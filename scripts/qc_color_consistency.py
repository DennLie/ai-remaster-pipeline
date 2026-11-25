#!/usr/bin/env python3
"""Color Consistency QC CLI for AI Remaster Pipeline.

Computes vectorized CIEDE2000 object color consistency, temporal color consistency
(CDC, optical flow warping chroma error, hue jumps), and mask bleed.
Updates shot manifests and writes QC events to SQLite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import uuid

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.color_bible_context import (
    canonical_shot_id,
    find_shot_mask_archive_candidates,
    shot_id_candidates,
)
from ai_remaster_gui.color_bible_db import (
    Film,
    QCEvent,
    ShotRecord,
    resolve_color_bible_db,
)
from ai_remaster_gui.color_bible_service import ColorBibleService
from ai_remaster_gui.color_qc import QCResult, evaluate_shot_qc
from ai_remaster_gui.manifests import read_manifest_details, write_manifest_details
from ai_remaster_gui.rle_masks import ShotMaskArchive


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="qc_color_consistency",
        description="Run automated Color Consistency Quality Control on shots.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to shot manifest CSV file.",
    )
    parser.add_argument(
        "--shot-id",
        type=str,
        default=None,
        help="Optional shot ID filter to process a specific shot.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory containing rendered frames or video file path.",
    )
    parser.add_argument(
        "--colorized-video",
        type=Path,
        default=None,
        help="Optional full colorized video file from which per-shot frames are extracted via FFmpeg.",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Directory containing .masks.npz multi-object mask archives.",
    )
    parser.add_argument(
        "--color-bible-db",
        type=Path,
        default=None,
        help="Path to Color Bible SQLite database (defaults to ARP_COLOR_BIBLE_DB or manifests/color_bible.db).",
    )
    parser.add_argument(
        "--film-id",
        type=str,
        default=None,
        help="Film identifier for Color Bible references and QC event logging.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal frame sampling stride (default 1; recommend 4 for 1080p speed).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write structured JSON Color QC report.",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="Flag to write qc_status, qc_delta_e, qc_cdc, and qc_failure_codes back to manifest CSV.",
    )
    parser.add_argument(
        "--record-db",
        action="store_true",
        help="Flag to insert QCEvent records into SQLite qc_events table.",
    )
    return parser


def _extract_shot_frames_from_video(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    stride: int = 1,
) -> list[Any] | None:
    """Decode shot frames ``[start_frame, end_frame)`` cleanly using FFmpeg (with CV2 fallback)."""
    import shutil
    import subprocess
    import cv2
    import numpy as np

    if not video_path.is_file():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if width <= 0 or height <= 0:
        return None

    start_idx = max(0, int(start_frame))
    end_idx = max(start_idx + 1, int(end_frame))
    step = max(1, int(stride))
    frames: list[np.ndarray] = []

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        vf = f"trim=start_frame={start_idx}:end_frame={end_idx},setpts=PTS-STARTPTS"
        if step > 1:
            vf += f",select=not(mod(n\\,{step}))"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = width * height * 3
        if proc.returncode == 0 and len(proc.stdout) >= frame_bytes:
            raw = np.frombuffer(proc.stdout, dtype=np.uint8)
            count = len(raw) // frame_bytes
            for i in range(count):
                arr = raw[i * frame_bytes : (i + 1) * frame_bytes].reshape((height, width, 3))
                frames.append(arr.copy())

    if not frames:
        cap = cv2.VideoCapture(str(video_path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            for idx in range(start_idx, end_idx):
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    break
                if (idx - start_idx) % step == 0:
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

    return frames or None


def resolve_frames_source(
    shot_id: str,
    frames_dir: Path | None,
    row: dict[str, str] | None = None,
    manifest_dir: Path | None = None,
    shot_index: int | None = None,
    film_id: str = "",
    manifest_stem: str = "",
    colorized_video: Path | None = None,
) -> Path | None:
    """Resolve the frames source path for a shot (supporting canonical shot IDs & chunk caches)."""
    candidate_ids: list[str] = [shot_id]
    if shot_index is not None:
        for cand in shot_id_candidates(film_id, shot_index):
            if cand not in candidate_ids:
                candidate_ids.append(cand)

    # 1. Check candidate subfolder, chunk cache, or video file in frames_dir
    if frames_dir is not None:
        if frames_dir.is_file():
            return frames_dir

        if frames_dir.is_dir():
            for cid in candidate_ids:
                shot_folder = frames_dir / cid
                if shot_folder.is_dir():
                    return shot_folder
                for ext in (".mp4", ".mkv", ".mov", ".avi", ".webm"):
                    shot_video = frames_dir / f"{cid}{ext}"
                    if shot_video.is_file():
                        return shot_video
            if shot_index is not None:
                chunk_matches = sorted(frames_dir.glob(f"segment_{shot_index:04d}_*.mkv"))
                if chunk_matches:
                    return chunk_matches[0]
            img_files = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
            if img_files:
                return frames_dir

    # 2. Check cached OSI / colorization chunk directory for the manifest
    if manifest_stem and shot_index is not None:
        chunk_root = REPO_ROOT / "intermediate" / "outpainted_colorized"
        if chunk_root.is_dir():
            for d in chunk_root.glob(f"{manifest_stem}*_chunks"):
                if d.is_dir():
                    matches = sorted(d.glob(f"segment_{shot_index:04d}_*.mkv"))
                    if matches:
                        return matches[0]

    # 3. Check row fields if available
    if row is not None:
        candidate_keys = [
            "rendered_path",
            "video_path",
            "frames_path",
            "prepared_path",
            "raw_path",
            "path",
        ]
        for key in candidate_keys:
            val = row.get(key, "").strip()
            if not val:
                continue
            p = Path(val)
            if p.is_file() or p.is_dir():
                return p
            p_repo = REPO_ROOT / val
            if p_repo.is_file() or p_repo.is_dir():
                return p_repo
            if manifest_dir is not None:
                p_man = manifest_dir / val
                if p_man.is_file() or p_man.is_dir():
                    return p_man

    if colorized_video is not None and colorized_video.is_file():
        return colorized_video

    return None


def resolve_mask_archive(
    shot_id: str,
    masks_dir: Path | None,
    shot_index: int | None = None,
    film_id: str = "",
) -> Path | None:
    """Resolve .masks.npz archive for a shot using canonical and legacy shot ID candidates."""
    if masks_dir is None or not masks_dir.is_dir():
        return None
    if shot_index is not None:
        found = find_shot_mask_archive_candidates(masks_dir, shot_index, film_id=film_id, shot_id=shot_id)
        if found is not None:
            return found

    candidates = [
        masks_dir / f"{shot_id}.masks.npz",
        masks_dir / f"{shot_id}.npz",
        masks_dir / f"{shot_id}_masks.npz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Color Consistency QC."""
    parser = build_parser()
    args = parser.parse_args(argv)

    film_id = (args.film_id or "").strip()
    if not film_id and args.manifest:
        film_id = args.manifest.stem.split("_")[0]
    if not film_id:
        film_id = "default_film"

    # Load reference stats from Color Bible database if provided
    ref_map: dict[str | int, dict[str, Any]] = {}
    db_path = resolve_color_bible_db(args.color_bible_db)
    service: ColorBibleService | None = None

    try:
        if args.color_bible_db or args.record_db:
            service = ColorBibleService(db_path=db_path)
            try:
                records = service.list_references(film_id=film_id)
                for rec in records:
                    try:
                        stats = json.loads(rec.lab_stats_json)
                        ref_map[rec.entity_id] = stats
                        ref_map[rec.entity_type] = stats
                        ref_map[f"{rec.entity_type}:{rec.entity_id}"] = stats
                        if rec.variant_id:
                            ref_map[f"{rec.entity_id}:{rec.variant_id}"] = stats
                    except Exception:
                        pass
            except Exception as exc:
                print(f"[WARN] Failed to load references from Color Bible: {exc}", file=sys.stderr)

        # Gather shots to evaluate
        shots_to_evaluate: list[dict[str, Any]] = []
        source_video = ""
        fieldnames: list[str] = []
        manifest_rows: list[dict[str, str]] = []

        if args.manifest is not None:
            manifest_path = Path(args.manifest)
            if not manifest_path.is_file():
                print(f"[ERROR] Manifest not found: {manifest_path}", file=sys.stderr)
                return 1
            source_video, fieldnames, manifest_rows = read_manifest_details(manifest_path)
            manifest_dir = manifest_path.parent

            for idx, row in enumerate(manifest_rows):
                try:
                    s_idx = int(row.get("shot_index", row.get("index", idx)))
                except (ValueError, TypeError):
                    s_idx = idx
                s_id = (
                    row.get("shot_id")
                    or row.get("id")
                    or canonical_shot_id(film_id, s_idx)
                )
                if args.shot_id and s_id != args.shot_id and args.shot_id not in shot_id_candidates(film_id, s_idx):
                    continue

                frames_p = resolve_frames_source(
                    shot_id=s_id,
                    frames_dir=args.frames_dir,
                    row=row,
                    manifest_dir=manifest_dir,
                    shot_index=s_idx,
                    film_id=film_id,
                    manifest_stem=manifest_path.stem,
                    colorized_video=args.colorized_video,
                )
                mask_p = resolve_mask_archive(shot_id=s_id, masks_dir=args.masks_dir, shot_index=s_idx, film_id=film_id)

                shots_to_evaluate.append({
                    "shot_id": s_id,
                    "shot_index": s_idx,
                    "frames_path": frames_p,
                    "mask_path": mask_p,
                    "row": row,
                    "row_index": idx,
                })
        elif args.frames_dir is not None:
            s_id = args.shot_id or args.frames_dir.stem
            mask_p = resolve_mask_archive(shot_id=s_id, masks_dir=args.masks_dir)
            shots_to_evaluate.append({
                "shot_id": s_id,
                "shot_index": 0,
                "frames_path": args.frames_dir,
                "mask_path": mask_p,
                "row": None,
                "row_index": None,
            })
        elif args.colorized_video is not None:
            s_id = args.shot_id or args.colorized_video.stem
            mask_p = resolve_mask_archive(shot_id=s_id, masks_dir=args.masks_dir)
            shots_to_evaluate.append({
                "shot_id": s_id,
                "shot_index": 0,
                "frames_path": args.colorized_video,
                "mask_path": mask_p,
                "row": None,
                "row_index": None,
            })
        else:
            print("[ERROR] Either --manifest or --frames-dir must be provided.", file=sys.stderr)
            return 1

        if not shots_to_evaluate:
            print("[WARN] No shots matched the criteria to evaluate.", file=sys.stderr)
            return 0

        print(f"=== ARP Color Consistency QC (Film: {film_id}, Shots: {len(shots_to_evaluate)}) ===")

        results: list[QCResult] = []
        has_failure = False

        for shot_info in shots_to_evaluate:
            shot_id = shot_info["shot_id"]
            shot_index = shot_info["shot_index"]
            frames_path = shot_info["frames_path"]
            mask_path = shot_info["mask_path"]

            if frames_path is None or not Path(frames_path).exists():
                print(f"[{shot_id}] WARN: Frames source not found. Skipping.")
                qc_res = QCResult(
                    shot_id=shot_id,
                    qc_status="warn",
                    failure_codes=["MISSING_FRAMES"],
                    metrics={"error": "Frames source not found"},
                )
            else:
                eval_input: Any = frames_path
                eval_stride = args.stride
                row_data = shot_info.get("row") or {}
                if (
                    args.colorized_video is not None
                    and Path(frames_path).resolve() == Path(args.colorized_video).resolve()
                    and str(row_data.get("start_frame", "")).strip().isdigit()
                    and str(row_data.get("end_frame", "")).strip().isdigit()
                ):
                    extracted = _extract_shot_frames_from_video(
                        Path(frames_path),
                        int(row_data["start_frame"]),
                        int(row_data["end_frame"]),
                        stride=args.stride,
                    )
                    if extracted is not None:
                        eval_input = extracted
                        eval_stride = 1
                qc_res = evaluate_shot_qc(
                    shot_id=shot_id,
                    frames=eval_input,
                    mask_archive=mask_path,
                    ref_lab_stats_map=ref_map,
                    stride=eval_stride,
                )

            results.append(qc_res)
            if qc_res.qc_status == "fail":
                has_failure = True

            status_tag = f"[{qc_res.qc_status.upper()}]"
            cdc_str = f"CDC: {qc_res.metrics.get('cdc', 0.0):.3f}"
            de_str = f"dE00: {qc_res.metrics.get('chroma_delta_e')}" if qc_res.metrics.get("chroma_delta_e") is not None else ""
            flow_str = f"FlowErr: {qc_res.metrics.get('flow_warp_error', 0.0):.2f}"
            fc_str = f"Codes: {','.join(qc_res.failure_codes)}" if qc_res.failure_codes else ""

            details = " | ".join(filter(bool, [cdc_str, de_str, flow_str, fc_str]))
            print(f"  {status_tag:8s} {shot_id:<16s} {details}")

            # Record QC event in SQLite database if requested
            if args.record_db and service is not None:
                try:
                    # Ensure film exists
                    film = service.repo.get_film(film_id)
                    if film is None:
                        service.repo.upsert_film(Film(film_id=film_id, title=film_id))

                    # Ensure shot exists
                    if shot_id and service.repo.get_shot(shot_id) is None:
                        service.repo.upsert_shot(
                            ShotRecord(
                                shot_id=shot_id,
                                film_id=film_id,
                                shot_index=shot_index,
                                start_frame=0,
                                end_frame=qc_res.metrics.get("total_frames", 0),
                                start_time=0.0,
                                end_time=0.0,
                            )
                        )

                    event = QCEvent(
                        event_id=f"qce_{uuid.uuid4().hex[:12]}",
                        film_id=film_id,
                        shot_id=shot_id,
                        shot_index=shot_index,
                        qc_status=qc_res.qc_status,
                        failure_codes=qc_res.failure_codes,
                        metrics=qc_res.metrics,
                    )
                    service.record_qc_event(event)
                except Exception as exc:
                    print(f"[WARN] Failed to record QC event for {shot_id}: {exc}", file=sys.stderr)

            # Update manifest row if requested
            if args.update_manifest and shot_info["row_index"] is not None:
                r_idx = shot_info["row_index"]
                manifest_rows[r_idx]["qc_status"] = qc_res.qc_status
                manifest_rows[r_idx]["qc_delta_e"] = (
                    f"{qc_res.metrics['chroma_delta_e']:.3f}"
                    if qc_res.metrics.get("chroma_delta_e") is not None
                    else ""
                )
                manifest_rows[r_idx]["qc_cdc"] = (
                    f"{qc_res.metrics['cdc']:.3f}"
                    if qc_res.metrics.get("cdc") is not None
                    else ""
                )
                manifest_rows[r_idx]["qc_flow_err"] = (
                    f"{qc_res.metrics['flow_warp_error']:.3f}"
                    if qc_res.metrics.get("flow_warp_error") is not None
                    else ""
                )
                manifest_rows[r_idx]["qc_bleed"] = (
                    f"{qc_res.metrics['bleed_score']:.3f}"
                    if qc_res.metrics.get("bleed_score") is not None
                    else ""
                )
                manifest_rows[r_idx]["qc_failure_codes"] = (
                    ";".join(qc_res.failure_codes) if qc_res.failure_codes else ""
                )

        # Write updated manifest CSV if requested
        if args.update_manifest and args.manifest is not None:
            target_fields = list(fieldnames)
            for col in ["qc_status", "qc_delta_e", "qc_cdc", "qc_flow_err", "qc_bleed", "qc_failure_codes"]:
                if col not in target_fields:
                    target_fields.append(col)
            write_manifest_details(args.manifest, source_video, target_fields, manifest_rows)
            print(f"[OK] Updated manifest CSV: {args.manifest}")

        # Write JSON report if requested
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            passed_count = sum(1 for r in results if r.qc_status == "pass")
            warned_count = sum(1 for r in results if r.qc_status == "warn")
            failed_count = sum(1 for r in results if r.qc_status == "fail")

            overall_status = "fail" if failed_count > 0 else ("warn" if warned_count > 0 else "pass")

            report = {
                "film_id": film_id,
                "manifest": str(args.manifest) if args.manifest else None,
                "summary": {
                    "total_shots": len(results),
                    "passed": passed_count,
                    "warned": warned_count,
                    "failed": failed_count,
                    "overall_status": overall_status,
                },
                "shots": [r.to_dict() for r in results],
            }
            with args.output_json.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[OK] Written QC report: {args.output_json}")

        print("==========================================================")
        if has_failure:
            print("[FAIL] Color consistency QC detected failure(s). Exit code 1.")
            return 1
        else:
            print("[PASS] Color consistency QC passed (or warned). Exit code 0.")
            return 0
    finally:
        if service is not None:
            try:
                service.close()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
