#!/usr/bin/env python3
"""Export Track C independent human-annotation pilot pack from raw archival frames only.

Strictly exports raw source frames I_t, raw 5-frame neighbor context strips
(I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}), and unreviewed (127) 3-tier template masks.
NEVER includes detector predictions, optical flow maps, or temporal differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from temporal_scratch_analyzer import load_clip

PACK_DIR = PROJECT_ROOT / ".cache" / "benchmarks" / "track_c_pilot_pack"
MANIFEST_OUT = PROJECT_ROOT / "docs" / "benchmarks" / "track_c_annotation_manifest.json"
STATUS_REPORT_MD = PROJECT_ROOT / "docs" / "benchmarks" / "track_c_pilot_status_report.md"
STATUS_REPORT_JSON = PROJECT_ROOT / "docs" / "benchmarks" / "track_c_pilot_status_report.json"

VALID_TIER_VALUES = np.array([0, 127, 255], dtype=np.uint8)
ALLOWED_REVIEW_STATUSES = (
    "pending",
    "first_pass_in_progress",
    "first_pass_complete",
    "second_pass_verified",
    "needs_revision",
)


def frame_review_key(clip_id: str, frame_index: int) -> str:
    return f"{clip_id}#{int(frame_index):06d}"


def resolve_manifest_path(
    rel_or_abs: str,
    *,
    pack_dir: Path | None = None,
    pilot_pack_rel: str = ".cache/benchmarks/track_c_pilot_pack",
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a manifest entry path against pack_dir (for isolated tests) or project_root."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    norm = rel_or_abs.replace("\\", "/")
    if pack_dir is not None:
        prefix = pilot_pack_rel.replace("\\", "/").rstrip("/") + "/"
        if norm.startswith(prefix):
            return pack_dir / norm[len(prefix) :]
        candidate = pack_dir / norm
        if candidate.exists():
            return candidate
    return project_root / norm


def resolve_pack_dir(data: dict[str, Any], pack_dir: Path | None = None, project_root: Path = PROJECT_ROOT) -> Path:
    if pack_dir is not None:
        return Path(pack_dir)
    rel = str(data.get("pilot_pack_dir") or ".cache/benchmarks/track_c_pilot_pack")
    p = Path(rel)
    return p if p.is_absolute() else (project_root / p)


def load_annotation_reviews(pack_dir: Path) -> dict[str, dict[str, Any]]:
    """Read optional provenance & second-pass review metadata without mutating files."""
    review_path = pack_dir / "annotation_reviews.json"
    if not review_path.exists():
        return {}
    try:
        raw = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    frames = raw.get("frames", {}) if isinstance(raw, dict) else {}
    return frames if isinstance(frames, dict) else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_neighbor_strip(frames: list[np.ndarray], idx: int, radius: int = 2) -> np.ndarray:
    """Build a raw horizontal contact strip [I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}] with frame labels."""
    h, w = frames[0].shape[:2]
    panels: list[np.ndarray] = []
    for offset in range(-radius, radius + 1):
        n_idx = idx + offset
        if 0 <= n_idx < len(frames):
            panel = frames[n_idx].copy()
            label = f"t{offset:+d} (f{n_idx})" if offset != 0 else f"TARGET t (f{n_idx})"
        else:
            panel = np.zeros((h, w, 3), dtype=np.uint8)
            label = f"t{offset:+d} (OUT_OF_RANGE)"
        hdr = np.zeros((28, w, 3), dtype=np.uint8)
        color = (0, 255, 255) if offset == 0 else (220, 220, 220)
        cv2.putText(hdr, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        panels.append(np.vstack([hdr, panel]))
    return np.hstack(panels)


def export_pilot_pack() -> dict[str, Any]:
    PACK_DIR.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            "clip_id": "neighbors_72",
            "video_path": PROJECT_ROOT / "evaluation" / "Neighbors_72.mkv",
            "frame_indices": [0, 12, 23, 25, 36, 48, 61, 62, 71],
            "split": "dev_diagnostic",
            "notes": (
                "Screen raw strip I_{t-2..t+2} for transient & persistent linear scratches plus intact architecture; "
                "Frame 25 ROI (x:180..192, y:48..282) is confirmed intact pipe (GT=0 negative control)."
            ),
        },
        {
            "clip_id": "paleface_seq_a",
            "video_path": PROJECT_ROOT / "evaluation" / "cross_film_validation" / "paleface_seq_a.mkv",
            "frame_indices": [8],
            "split": "dev_diagnostic",
            "notes": (
                "Screen for transient/persistent scratches across moving cabin exterior & verify intact actor silhouette "
                "negative-control ROI (x:258..268, y:188..381)."
            ),
        },
        {
            "clip_id": "paleface_seq_b",
            "video_path": PROJECT_ROOT / "evaluation" / "cross_film_validation" / "paleface_seq_b.mkv",
            "frame_indices": [7],
            "split": "dev_diagnostic",
            "notes": (
                "Screen for diagonal/branching/transient damage & verify intact static wall/curtain crack "
                "negative-control ROI (x:200..280, y:80..180)."
            ),
        },
        {
            "clip_id": "college_seq_c",
            "video_path": PROJECT_ROOT / "evaluation" / "cross_film_validation" / "college_seq_c.mkv",
            "frame_indices": [8],
            "split": "dev_diagnostic",
            "notes": (
                "Screen for scratches/dust across classroom sequence & verify classroom door jamb / moving student "
                "negative-control ROI (x:280..340, y:150..280)."
            ),
        },
        {
            "clip_id": "neighbors_seq_d",
            "video_path": PROJECT_ROOT / "evaluation" / "cross_film_validation" / "neighbors_seq_d.mkv",
            "frame_indices": [2, 30],
            "split": "dev_diagnostic",
            "notes": (
                "Screen for transient and multi-frame persistent street-scene defects & verify telephone pole "
                "negative-control ROI (x:150..220, y:200..400)."
            ),
        },
    ]

    exported_items: list[dict[str, Any]] = []

    for src in sources:
        vpath = src["video_path"]
        frames = load_clip(vpath)
        clip_dir = PACK_DIR / src["clip_id"]
        (clip_dir / "raw_frames").mkdir(parents=True, exist_ok=True)
        (clip_dir / "neighbor_strips").mkdir(parents=True, exist_ok=True)
        (clip_dir / "templates").mkdir(parents=True, exist_ok=True)

        for f_idx in src["frame_indices"]:
            if not (0 <= f_idx < len(frames)):
                continue
            frame_bgr = frames[f_idx]
            h, w = frame_bgr.shape[:2]
            fname = f"frame_{f_idx:06d}.png"

            raw_out = clip_dir / "raw_frames" / fname
            strip_out = clip_dir / "neighbor_strips" / f"strip_{f_idx:06d}.png"
            scr_tmpl = clip_dir / "templates" / f"frame_{f_idx:06d}_scratch_gt.png"
            dst_tmpl = clip_dir / "templates" / f"frame_{f_idx:06d}_dust_gt.png"
            neg_tmpl = clip_dir / "templates" / f"frame_{f_idx:06d}_neg_control_gt.png"

            cv2.imwrite(str(raw_out), frame_bgr)
            cv2.imwrite(str(strip_out), _build_neighbor_strip(frames, f_idx, radius=2))

            # Initialize template masks strictly to 127 (unreviewed) so blank 0 is never mistaken for reviewed clean
            unreviewed = np.full((h, w), 127, dtype=np.uint8)
            for tmpl in (scr_tmpl, dst_tmpl, neg_tmpl):
                if not tmpl.exists():
                    cv2.imwrite(str(tmpl), unreviewed)

            exported_items.append({
                "clip_id": src["clip_id"],
                "frame_index": f_idx,
                "split": src["split"],
                "resolution": [w, h],
                "source_video": str(vpath.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "source_sha256": _sha256(vpath),
                "raw_frame_path": str(raw_out.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "neighbor_strip_path": str(strip_out.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "scratch_template_path": str(scr_tmpl.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "dust_template_path": str(dst_tmpl.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "neg_control_template_path": str(neg_tmpl.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "annotation_status": "pending_human_annotation",
                "notes": src["notes"],
            })

    protocol = {
        "protocol_version": "2.0.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pilot_pack_dir": str(PACK_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "exported_keyframes_count": len(exported_items),
        "rules": [
            "Inspect ONLY raw_frames/ and neighbor_strips/ (raw I_{t-2..t+2} frames).",
            "NEVER inspect or derive masks from detector predictions, optical flow, temporal differences, or reconstruction residuals.",
            "Encode 3-tier PNG masks strictly as: 0 = confirmed intact, 127 = uncertain boundary / unreviewed, 255 = confirmed defect.",
            "Templates are initialized to 127 (unreviewed); annotators must explicitly paint confirmed intact regions as 0 and defects as 255.",
            "Include real transient scratches, real persistent film-locked scratches, and critical structural negative-control ROIs.",
            "Distinguish this 14-keyframe dev_diagnostic pilot from both scene-level holdouts and the final film-level locked holdout (>=30 unseen keyframes across >=3 unseen film titles).",
        ],
        "reclassified_dev_diagnostic_datasets": [
            "AbsoluteDegradation clip 000 (dev_diagnostic)",
            "AbsoluteDegradation clip 010 (dev_diagnostic)",
            "AbsoluteDegradation clip 020 & its 6 extended synthetic regimes (reclassified from locked_holdout to dev_diagnostic)",
            "Neighbors_72.mkv, paleface_seq_a.mkv, paleface_seq_b.mkv, college_seq_c.mkv, neighbors_seq_d.mkv (all dev_diagnostic)",
        ],
        "locked_holdout_plan": {
            "status": "RESERVED_UNTOUCHED",
            "policy": "Do NOT inspect, tune against, or run detector ablations on locked holdout assets until final candidate verification.",
            "scene_level_holdout_only": [
                "Uninspected late scenes of evaluation/Neighbors_original.webm (>03:00): same film title as dev Neighbors_72, so valid only as a scene-level holdout, NEVER as an independent film-level holdout."
            ],
            "track_b_unseen_synthetic_holdout": [
                "Uninspected clean archival video substrates (e.g. .cache/stabilization-benchmark/development-smoothing-20261105-01 Corner in Wheat clean frames after pre-defect screening, plus additional unseen film substrates outside Neighbors/Paleface/College)",
                "Fresh independent random seeds (20261201..20261210) covering bright/dark, vertical/diagonal/horizontal, fine/broad, transient/persistent, and telecine cadences",
            ],
            "track_c_unseen_archival_holdout": [
                "Minimum 3 genuinely unseen film titles (excluding Neighbors, The Paleface, and College) with >=30 independently human-annotated keyframes + critical structural negative controls.",
                "Neighbors_original.webm (>03:00) may be reported separately as a scene-level holdout only and must NOT count toward the 3 unseen film titles.",
            ],
        },
        "pilot_keyframes": exported_items,
    }

    (PACK_DIR / "annotation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUT.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def verify_human_labels(
    manifest_path: Path = MANIFEST_OUT,
    *,
    pack_dir: Path | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate manual Track C 3-tier PNG masks without auto-generating any labels."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Annotation manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_pack = resolve_pack_dir(data, pack_dir=pack_dir, project_root=project_root)
    pilot_pack_rel = str(data.get("pilot_pack_dir") or ".cache/benchmarks/track_c_pilot_pack")
    items = data.get("pilot_keyframes", [])
    reviews = load_annotation_reviews(resolved_pack)

    summary: dict[str, Any] = {
        "protocol_version": data.get("protocol_version", "2.0.0"),
        "manifest_path": str(manifest_path),
        "pack_dir": str(resolved_pack),
        "annotation_classification": "dev_diagnostic_pilot_only",
        "total_keyframes": len(items),
        "completed_keyframes": 0,
        "pending_keyframes": 0,
        "second_pass_verified_keyframes": 0,
        "keyframes_with_conflicts": 0,
        "partial_coverage_keyframes": 0,
        "unreviewed_keyframes": 0,
        "total_positive_scratch_px": 0,
        "total_positive_dust_px": 0,
        "total_reviewed_intact_px": 0,
        "total_neg_control_roi_px": 0,
        "total_uncertain_or_unreviewed_px": 0,
        "total_conflict_px": 0,
        "mean_reviewed_coverage_pct": 0.0,
        "raw_integrity_failures": [],
        "invalid_files": [],
        "ready_for_real_film_claim": False,
        "ready_with_second_pass": False,
        "per_frame": [],
    }

    coverage_sum = 0.0
    for item in items:
        clip_id = str(item["clip_id"])
        frame_idx = int(item["frame_index"])
        dataset_role = str(item.get("split", "dev_diagnostic"))
        key = frame_review_key(clip_id, frame_idx)
        rev = reviews.get(key, {})

        raw_rel = item.get("raw_frame_path")
        raw_shape: tuple[int, int] | None = None
        raw_integrity_ok = True
        if raw_rel:
            raw_p = resolve_manifest_path(
                str(raw_rel),
                pack_dir=resolved_pack,
                pilot_pack_rel=pilot_pack_rel,
                project_root=project_root,
            )
            if not raw_p.exists():
                raw_integrity_ok = False
                summary["raw_integrity_failures"].append({"clip_id": clip_id, "frame_index": frame_idx, "error": "missing raw frame"})
            else:
                raw_img = cv2.imread(str(raw_p), cv2.IMREAD_COLOR)
                if raw_img is None:
                    raw_integrity_ok = False
                    summary["raw_integrity_failures"].append({"clip_id": clip_id, "frame_index": frame_idx, "error": "unreadable raw frame"})
                else:
                    raw_shape = (int(raw_img.shape[0]), int(raw_img.shape[1]))
                    expected_sha = item.get("raw_sha256")
                    if expected_sha and _sha256(raw_p) != expected_sha:
                        raw_integrity_ok = False
                        summary["raw_integrity_failures"].append({"clip_id": clip_id, "frame_index": frame_idx, "error": "raw SHA256 mismatch"})

        arrays: dict[str, np.ndarray] = {}
        frame_invalid = False
        for ch_key in ("scratch_template_path", "dust_template_path", "neg_control_template_path"):
            p = resolve_manifest_path(
                str(item[ch_key]),
                pack_dir=resolved_pack,
                pilot_pack_rel=pilot_pack_rel,
                project_root=project_root,
            )
            if not p.exists():
                summary["invalid_files"].append({"path": str(p), "error": "missing"})
                frame_invalid = True
                continue
            arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if arr is None or arr.ndim != 2 or arr.dtype != np.uint8:
                summary["invalid_files"].append({"path": str(p), "error": "must be 2D uint8 PNG"})
                frame_invalid = True
                continue
            unexpected = np.setdiff1d(np.unique(arr), VALID_TIER_VALUES)
            if unexpected.size > 0:
                summary["invalid_files"].append({"path": str(p), "error": f"invalid values {unexpected.tolist()}"})
                frame_invalid = True
                continue
            if raw_shape is not None and arr.shape != raw_shape:
                summary["invalid_files"].append({"path": str(p), "error": f"shape {arr.shape} != raw {raw_shape}"})
                frame_invalid = True
                continue
            arrays[ch_key] = arr

        if frame_invalid or len(arrays) != 3:
            summary["pending_keyframes"] += 1
            summary["per_frame"].append(
                {
                    "clip_id": clip_id,
                    "frame_index": frame_idx,
                    "dataset_role": dataset_role,
                    "valid": False,
                    "raw_integrity_ok": raw_integrity_ok,
                    "review_status": "invalid",
                }
            )
            continue

        scr = arrays["scratch_template_path"]
        dst = arrays["dust_template_path"]
        neg = arrays["neg_control_template_path"]
        h, w = int(scr.shape[0]), int(scr.shape[1])
        total_px = h * w

        scratch_mask = scr == 255
        dust_mask = dst == 255
        neg_mask = neg == 255
        intact_mask = (scr == 0) & (dst == 0)
        both_reviewed_mask = (scr != 127) & (dst != 127)
        any_reviewed_mask = (scr != 127) | (dst != 127) | (neg != 127)

        scratch_px = int(np.count_nonzero(scratch_mask))
        dust_px = int(np.count_nonzero(dust_mask))
        neg_control_roi_px = int(np.count_nonzero(neg_mask))
        verified_intact_px = int(np.count_nonzero(intact_mask))
        both_reviewed_px = int(np.count_nonzero(both_reviewed_mask))
        any_reviewed_px = int(np.count_nonzero(any_reviewed_mask))
        uncertain_or_unreviewed_px = int(total_px - both_reviewed_px)
        reviewed_coverage_pct = round(100.0 * both_reviewed_px / max(total_px, 1), 2)
        any_channel_coverage_pct = round(100.0 * any_reviewed_px / max(total_px, 1), 2)

        scratch_dust_overlap_px = int(np.count_nonzero(scratch_mask & dust_mask))
        neg_control_defect_conflict_px = int(np.count_nonzero(neg_mask & (scratch_mask | dust_mask)))
        neg_control_uncleared_px = int(np.count_nonzero(neg_mask & ((scr == 127) | (dst == 127))))
        channel_review_mismatch_px = int(np.count_nonzero(((scr != 127) & (dst == 127)) | ((scr == 127) & (dst != 127))))
        hard_conflict_px = scratch_dust_overlap_px + neg_control_defect_conflict_px

        reviewed_any = any_reviewed_px > 0
        if not reviewed_any:
            coverage_state = "unreviewed"
            summary["unreviewed_keyframes"] += 1
        elif both_reviewed_px < total_px:
            coverage_state = "partial"
            summary["partial_coverage_keyframes"] += 1
        else:
            coverage_state = "complete"

        default_status = "pending" if not reviewed_any else "first_pass_in_progress"
        review_status = str(rev.get("review_status") or default_status)
        if review_status not in ALLOWED_REVIEW_STATUSES:
            review_status = default_status
        second_pass_verified = review_status == "second_pass_verified" and reviewed_any and hard_conflict_px == 0

        if reviewed_any:
            summary["completed_keyframes"] += 1
        else:
            summary["pending_keyframes"] += 1
        if second_pass_verified:
            summary["second_pass_verified_keyframes"] += 1
        if hard_conflict_px > 0:
            summary["keyframes_with_conflicts"] += 1

        summary["total_positive_scratch_px"] += scratch_px
        summary["total_positive_dust_px"] += dust_px
        summary["total_reviewed_intact_px"] += verified_intact_px
        summary["total_neg_control_roi_px"] += neg_control_roi_px
        summary["total_uncertain_or_unreviewed_px"] += uncertain_or_unreviewed_px
        summary["total_conflict_px"] += hard_conflict_px
        coverage_sum += reviewed_coverage_pct

        summary["per_frame"].append(
            {
                "clip_id": clip_id,
                "frame_index": frame_idx,
                "dataset_role": dataset_role,
                "valid": True,
                "raw_integrity_ok": raw_integrity_ok,
                "width": w,
                "height": h,
                "total_px": total_px,
                "scratch_px": scratch_px,
                "dust_px": dust_px,
                "neg_control_roi_px": neg_control_roi_px,
                "verified_intact_px": verified_intact_px,
                "both_defect_channels_reviewed_px": both_reviewed_px,
                "any_channel_reviewed_px": any_reviewed_px,
                "uncertain_or_unreviewed_px": uncertain_or_unreviewed_px,
                "reviewed_coverage_pct": reviewed_coverage_pct,
                "any_channel_coverage_pct": any_channel_coverage_pct,
                "coverage_state": coverage_state,
                "coverage_incomplete": both_reviewed_px < total_px,
                "scratch_dust_overlap_px": scratch_dust_overlap_px,
                "neg_control_defect_conflict_px": neg_control_defect_conflict_px,
                "neg_control_uncleared_px": neg_control_uncleared_px,
                "channel_review_mismatch_px": channel_review_mismatch_px,
                "hard_conflict_px": hard_conflict_px,
                "annotator": str(rev.get("annotator") or ""),
                "second_pass_reviewer": str(rev.get("second_pass_reviewer") or ""),
                "review_status": review_status,
                "second_pass_verified": second_pass_verified,
                "notes": str(rev.get("notes") or item.get("notes") or ""),
                "updated_at": str(rev.get("updated_at") or ""),
            }
        )

    if items:
        summary["mean_reviewed_coverage_pct"] = round(coverage_sum / len(items), 2)

    summary["ready_for_real_film_claim"] = (
        summary["completed_keyframes"] == summary["total_keyframes"]
        and summary["total_keyframes"] > 0
        and summary["total_positive_scratch_px"] > 0
        and len(summary["invalid_files"]) == 0
        and len(summary["raw_integrity_failures"]) == 0
        and summary["total_conflict_px"] == 0
    )
    summary["ready_with_second_pass"] = (
        summary["ready_for_real_film_claim"]
        and summary["second_pass_verified_keyframes"] == summary["total_keyframes"]
    )
    return summary


def render_status_report_markdown(report: dict[str, Any]) -> str:
    """Format a human-readable Markdown quality & completeness report for Track C pilot GT."""
    lines: list[str] = [
        "# Track C Pilot Human Annotation Status Report",
        "",
        f"- **Dataset classification:** `{report.get('annotation_classification', 'dev_diagnostic_pilot_only')}` (pilot diagnostic material only; NOT a locked cross-film generalization holdout)",
        f"- **Pack directory:** `{report.get('pack_dir', '')}`",
        f"- **Manifest:** `{report.get('manifest_path', '')}`",
        f"- **Keyframes with annotations:** `{report['completed_keyframes']} / {report['total_keyframes']}` (pending: `{report['pending_keyframes']}`)",
        f"- **Second-pass verified keyframes:** `{report['second_pass_verified_keyframes']} / {report['total_keyframes']}`",
        f"- **Mean reviewed coverage (both scratch & dust != 127):** `{report['mean_reviewed_coverage_pct']:.2f}%`",
        f"- **Total verified scratch px (`255`):** `{report['total_positive_scratch_px']}`",
        f"- **Total verified dust px (`255`):** `{report['total_positive_dust_px']}`",
        f"- **Total verified intact px (`0`):** `{report['total_reviewed_intact_px']}`",
        f"- **Total verified negative-control ROI px (`255`):** `{report['total_neg_control_roi_px']}`",
        f"- **Total uncertain / unreviewed px (`127`):** `{report['total_uncertain_or_unreviewed_px']}`",
        f"- **Total hard label conflicts:** `{report['total_conflict_px']}` across `{report['keyframes_with_conflicts']}` keyframe(s)",
        f"- **Raw frame SHA256 integrity failures:** `{len(report['raw_integrity_failures'])}`",
        f"- **Invalid / corrupted mask files:** `{len(report['invalid_files'])}`",
        f"- **Ready for pilot real-film evaluation:** `{report['ready_for_real_film_claim']}` (with full second-pass sign-off: `{report['ready_with_second_pass']}`)",
        "",
    ]

    warnings: list[str] = []
    if report["completed_keyframes"] == 0:
        warnings.append("INCOMPLETE GROUND TRUTH: 0 keyframes have human labels yet. No automated labels were generated; run the offline annotator first.")
    elif report["completed_keyframes"] < report["total_keyframes"]:
        warnings.append(f"INCOMPLETE PILOT COVERAGE: {report['pending_keyframes']} keyframe(s) are still completely unreviewed (`127` everywhere).")
    if report["partial_coverage_keyframes"] > 0:
        warnings.append(
            f"PARTIAL FRAME COVERAGE NOTICE: {report['partial_coverage_keyframes']} keyframe(s) retain `127` (uncertain or unreviewed) pixels; Track C evaluation excludes `127` and reports reviewed coverage % explicitly."
        )
    if report["total_conflict_px"] > 0:
        warnings.append(f"LABEL CONFLICTS DETECTED: {report['total_conflict_px']} pixel(s) have scratch+dust overlap or defect inside a negative-control ROI. Resolve before second-pass sign-off.")
    if report["raw_integrity_failures"]:
        warnings.append(f"RAW INTEGRITY ERROR: {len(report['raw_integrity_failures'])} raw frame(s) failed SHA256 verification.")
    if warnings:
        lines.append("## Audit Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- **WARNING:** {w}")
        lines.append("")

    lines.extend(
        [
            "## Per-Frame Annotation Coverage & Conflict Audit",
            "",
            "| Clip | Frame | Role | Scratch px | Dust px | NegCtrl ROI px | Verified Intact px | Uncertain/Unreviewed px | Coverage % | Conflicts | Review Status | Annotator / 2nd Pass |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for pf in report.get("per_frame", []):
        if not pf.get("valid", True):
            lines.append(
                f"| `{pf['clip_id']}` | {pf['frame_index']} | `{pf['dataset_role']}` | INVALID | - | - | - | - | - | - | `invalid` | - |"
            )
            continue
        reviewers = pf.get("annotator") or "-"
        if pf.get("second_pass_reviewer"):
            reviewers = f"{reviewers} / {pf['second_pass_reviewer']}"
        lines.append(
            f"| `{pf['clip_id']}` | {pf['frame_index']} | `{pf['dataset_role']}` | "
            f"{pf['scratch_px']} | {pf['dust_px']} | {pf['neg_control_roi_px']} | "
            f"{pf['verified_intact_px']} | {pf['uncertain_or_unreviewed_px']} | "
            f"{pf['reviewed_coverage_pct']:.2f}% ({pf['coverage_state']}) | "
            f"{pf['hard_conflict_px']} | `{pf['review_status']}` | {reviewers} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_status_report(
    manifest_path: Path = MANIFEST_OUT,
    *,
    pack_dir: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    md_out: Path | None = STATUS_REPORT_MD,
    json_out: Path | None = STATUS_REPORT_JSON,
) -> dict[str, Any]:
    """Generate JSON + Markdown annotation quality and completeness status reports."""
    report = verify_human_labels(manifest_path=manifest_path, pack_dir=pack_dir, project_root=project_root)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_status_report_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-human-labels",
        action="store_true",
        help="Validate human-completed 3-tier PNG masks against protocol without modifying labels.",
    )
    parser.add_argument(
        "--status-report",
        action="store_true",
        help="Write Markdown and JSON annotation quality/coverage status reports.",
    )
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_OUT)
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--report-md", type=Path, default=STATUS_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=STATUS_REPORT_JSON)
    args = parser.parse_args()
    if args.verify_human_labels or args.status_report:
        if args.status_report:
            report = generate_status_report(
                manifest_path=args.manifest_path,
                pack_dir=args.pack_dir,
                md_out=args.report_md,
                json_out=args.report_json,
            )
        else:
            report = verify_human_labels(manifest_path=args.manifest_path, pack_dir=args.pack_dir)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if (not report["invalid_files"] and not report["raw_integrity_failures"]) else 1)
    res = export_pilot_pack()
    print(f"Exported {res['exported_keyframes_count']} raw Track C pilot keyframes to {PACK_DIR}")
    print(f"Updated manifest: {MANIFEST_OUT}")
