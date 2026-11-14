"""Unified entry point for automated Dust & Scratch benchmark execution.

Automatically provisions eligible external/internal benchmark resources, runs
multi-candidate defect detector evaluation across standardized datasets, computes
spatial, per-class, nuisance-region, and temporal consistency metrics, measures
execution latency and memory, generates visual contact sheets, and produces
comprehensive comparative reports (JSON and Markdown).

STRICT ENVIRONMENT ISOLATION:
Never modifies ARP's `.venv` or ComfyUI's Python environment. External research
code and data extraction run strictly inside dedicated isolated environments.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    import psutil
except ImportError:
    psutil = None

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import benchmark_defect_masks as bdm
import candidate_detector_eval as cde
import dataset_adapters as da
import film_defect_detectors as fdd
import provision_benchmark_resources as pbr
import temporal_consistency_scorer as tcs


COMMERCIAL_CANDIDATES = ("farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy")
RESEARCH_CANDIDATES = ("dart_harness",)
DEFAULT_CANDIDATES = COMMERCIAL_CANDIDATES

MIN_FRAMES_PER_CANDIDATE = {
    "farneback_3f": 3,
    "dis_flow": 3,
    "ridge": 3,
    "consensus_5f": 2,
    "legacy": 1,
    "dart_harness": 2,
}


class EvaluationIntegrityError(RuntimeError):
    """Raised when evaluation integrity is compromised (e.g. missing or unreadable frames)."""


def load_dataset_records(
    dataset_dir: Path,
    *,
    return_invalid: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load adapted dataset manifest, frames, and ground-truth masks.

    Verifies every source frame and ground-truth mask exists, can be decoded,
    and shares identical spatial dimensions. Never fabricates black images.
    """
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.is_file():
        return ([], []) if return_invalid else []

    gt_root = dataset_dir / "ground_truth"
    frames_root = dataset_dir / "frames"

    records = []
    invalid_samples = []

    with manifest_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            clip = row["clip"]
            frame = row["frame"]
            split = row.get("split", "test")
            damage_type = row.get("damage_type", "all_damage")
            prov = row.get("label_provenance", "unknown")

            gt_file = gt_root / clip / frame
            frame_file = frames_root / clip / frame

            # Ground truth verification
            if not gt_file.is_file():
                invalid_samples.append({
                    "clip": clip,
                    "frame": frame,
                    "reason": f"Missing GT mask file: {gt_file}",
                })
                continue
            gt_mask = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
            if gt_mask is None:
                invalid_samples.append({
                    "clip": clip,
                    "frame": frame,
                    "reason": f"Corrupt or unreadable GT mask: {gt_file}",
                })
                continue

            # Source image verification - NEVER evaluate missing images as black frames
            if not frame_file.is_file():
                invalid_samples.append({
                    "clip": clip,
                    "frame": frame,
                    "reason": f"Missing source frame file: {frame_file}",
                })
                continue
            frame_img = cv2.imread(str(frame_file))
            if frame_img is None:
                invalid_samples.append({
                    "clip": clip,
                    "frame": frame,
                    "reason": f"Corrupt or unreadable source frame: {frame_file}",
                })
                continue

            # Dimension correspondence check
            if frame_img.shape[:2] != gt_mask.shape:
                invalid_samples.append({
                    "clip": clip,
                    "frame": frame,
                    "reason": (
                        f"Spatial dimension mismatch: image {frame_img.shape[:2]} "
                        f"vs ground truth {gt_mask.shape}"
                    ),
                })
                continue

            # Pixel variance check: reject fabricated all-zero black images when GT has defects
            if np.max(frame_img) == 0 and np.any(gt_mask == 255):
                invalid_samples.append({
                    "clip": clip,
                    "frame": frame,
                    "reason": f"Fabricated or corrupted black image (pixel variance 0 while GT has defects): {frame_file}",
                })
                continue

            # Load channel-specific ground-truth masks if available
            gt_scratch_file = dataset_dir / "ground_truth_scratch" / clip / frame
            gt_scratch_mask = None
            if gt_scratch_file.is_file():
                gt_scratch_mask = cv2.imread(str(gt_scratch_file), cv2.IMREAD_GRAYSCALE)
            elif damage_type == "scratch":
                gt_scratch_mask = gt_mask

            gt_dust_file = dataset_dir / "ground_truth_dust" / clip / frame
            gt_dust_mask = None
            if gt_dust_file.is_file():
                gt_dust_mask = cv2.imread(str(gt_dust_file), cv2.IMREAD_GRAYSCALE)
            elif damage_type == "dust":
                gt_dust_mask = gt_mask

            records.append(
                {
                    "clip": clip,
                    "frame": frame,
                    "split": split,
                    "damage_type": damage_type,
                    "label_provenance": prov,
                    "gt_mask": gt_mask,
                    "gt_scratch_mask": gt_scratch_mask,
                    "gt_dust_mask": gt_dust_mask,
                    "image": frame_img,
                    "gt_path": str(gt_file),
                    "frame_path": str(frame_file),
                }
            )

    # Deterministic frame sorting within clips
    records.sort(key=lambda r: (r["clip"], r["frame"]))

    if return_invalid:
        return records, invalid_samples
    return records


def run_detector_on_sequence(
    detector_name: str,
    frames_bgr: list[np.ndarray],
    *,
    frame_names: Sequence[str] | None = None,
    sensitivity: float = 0.65,
    external_mask_dir: Path | None = None,
    acknowledge_research: bool = False,
) -> dict[str, Any]:
    """Execute a candidate detector on a sequence of frames with strict provenance tracking."""
    if detector_name == "dart_harness":
        # Check if external precomputed masks exist for these exact frame names
        mask_dir = None
        if external_mask_dir is not None and frame_names and frames_bgr and frames_bgr[0].shape[:2] == (720, 960):
            mask_root = Path(external_mask_dir)
            if (mask_root / "masks").is_dir():
                mask_root = mask_root / "masks"
            all_exist = all(
                (mask_root / (fn if fn.endswith(".png") else f"{fn}.png")).is_file()
                or (mask_root / f"frame_{int(fn.split('.')[0]):06d}.png").is_file()
                for fn in frame_names
                if fn.split(".")[0].isdigit()
            )
            if all_exist:
                mask_dir = external_mask_dir

        # Run genuine inference or load precomputed masks; NEVER silently run fallback as DART
        return cde.detect_dart_harness(
            frames_bgr,
            external_mask_dir=mask_dir,
            frame_names=frame_names if mask_dir else None,
            sensitivity=sensitivity,
            run_inference=acknowledge_research,
            allow_fallback=False,
        )

    detector_fn = cde.CANDIDATE_REGISTRY.get(detector_name)
    if detector_fn is None:
        raise ValueError(f"Unknown candidate detector: {detector_name}")
    res = detector_fn(frames_bgr, sensitivity=sensitivity)
    if "provenance" not in res:
        res["provenance"] = f"in_tree_{detector_name}"
    return res


def evaluate_dataset(
    dataset_name: str,
    records: list[dict[str, Any]],
    candidate_names: Sequence[str],
    *,
    sensitivity: float = 0.65,
    threshold: int = 128,
    external_dart_dir: Path | None = None,
    contact_sheet_dir: Path | None = None,
    acknowledge_research: bool = False,
    invalid_samples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run candidate evaluation on a dataset group with temporal and channel integrity checks."""
    if not records:
        return {
            "dataset": dataset_name,
            "total_samples": 0,
            "clips": [],
            "candidates": {},
            "invalid_samples": invalid_samples or [],
        }

    # Group records by clip and sort by frame name
    clips: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        clips.setdefault(r["clip"], []).append(r)
    for c_records in clips.values():
        c_records.sort(key=lambda r: str(r["frame"]))

    dataset_results: dict[str, Any] = {
        "dataset": dataset_name,
        "total_samples": len(records),
        "clips": list(clips.keys()),
        "candidates": {},
        "invalid_samples": invalid_samples or [],
    }

    min_frames_map = MIN_FRAMES_PER_CANDIDATE

    for cand in candidate_names:
        # Commercial licensing check
        if not acknowledge_research and cand in RESEARCH_CANDIDATES:
            dataset_results["candidates"][cand] = {
                "status": "EXCLUDED_COMMERCIAL_STRICT",
                "provenance": "not_evaluated",
                "reason": "Requires explicit --acknowledge-research opt-in",
                "overall": {"status": "NOT_APPLICABLE"},
                "scratch": {"status": "NOT_APPLICABLE"},
                "dust": {"status": "NOT_APPLICABLE"},
            }
            continue

        min_req = min_frames_map.get(cand, 1)
        max_clip_len = max(len(c) for c in clips.values()) if clips else 0

        # Check if dataset has sufficient temporal context for this algorithm
        if max_clip_len < min_req:
            dataset_results["candidates"][cand] = {
                "status": "INSUFFICIENT_TEMPORAL_CONTEXT",
                "provenance": "not_evaluated",
                "required_frames": min_req,
                "actual_frames": max_clip_len,
                "overall": {
                    "status": "NOT_APPLICABLE",
                    "reason": f"Detector requires >= {min_req} frames; dataset clips have {max_clip_len} frame(s)",
                },
                "scratch": {"status": "NOT_APPLICABLE"},
                "dust": {"status": "NOT_APPLICABLE"},
                "nuisance": {},
                "temporal": {},
                "latency_ms_per_frame": 0.0,
                "peak_memory_mb": 0.0,
            }
            continue

        supports_dust = cand not in ("ridge", "legacy")
        supports_scratch = True

        cand_metrics_list = []
        scratch_metrics_list = []
        dust_metrics_list = []
        nuisance_list = []
        temporal_metrics_list = []
        total_time_ms = 0.0
        total_frames_processed = 0
        steady_state_time_ms = 0.0
        steady_state_frames = 0
        cold_start_ms = 0.0
        max_peak_mem_bytes = 0
        detected_provenance = "unknown"

        for clip_idx, (clip_name, clip_records) in enumerate(clips.items()):
            if len(clip_records) < min_req:
                continue

            # Build and verify frame list - never substitute black frames
            frames = []
            for r in clip_records:
                if r.get("image") is None:
                    raise EvaluationIntegrityError(
                        f"Missing source frame for clip '{clip_name}', frame '{r.get('frame')}'. "
                        "Evaluation integrity violated: fabricated black images are prohibited."
                    )
                frames.append(r["image"])

            # Run detector with memory profiling (tracking both tracemalloc and native memory)
            tracemalloc.start()
            proc = psutil.Process() if psutil else None
            mem_before = proc.memory_info().rss if proc else 0
            t0 = time.perf_counter()
            frame_names = [r["frame"] for r in clip_records]

            preds = run_detector_on_sequence(
                cand,
                frames,
                frame_names=frame_names,
                sensitivity=sensitivity,
                external_mask_dir=external_dart_dir,
                acknowledge_research=acknowledge_research,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            _, peak_py_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            mem_after = proc.memory_info().rss if proc else 0
            native_mem_diff = max(0, mem_after - mem_before)
            peak_wset = proc.memory_info().peak_wset if proc else 0
            # Track peak memory considering python heap, native working set delta, and process peak
            peak_mem = max(peak_py_mem, native_mem_diff, min(peak_wset, 1024 * 1024 * 1024))

            if clip_idx == 0:
                cold_start_ms = elapsed_ms
            else:
                steady_state_time_ms += elapsed_ms
                steady_state_frames += len(frames)

            max_peak_mem_bytes = max(max_peak_mem_bytes, peak_mem)
            total_time_ms += elapsed_ms
            total_frames_processed += len(frames)
            detected_provenance = preds.get("provenance", detected_provenance)

            scratch_preds = preds.get("scratch", [np.zeros(f.shape[:2], dtype=np.uint8) for f in frames])
            dust_preds = preds.get("dust", [np.zeros(f.shape[:2], dtype=np.uint8) for f in frames])

            # Combined prediction mask
            if supports_dust:
                combined_preds = [
                    np.maximum(s, d) for s, d in zip(scratch_preds, dust_preds)
                ]
            else:
                combined_preds = [s.copy() for s in scratch_preds]

            gt_masks = [r["gt_mask"] for r in clip_records]

            # 1. Pixel confusion across frames (evaluated exactly once per authentic frame)
            for idx, r in enumerate(clip_records):
                gt = gt_masks[idx]
                pred = combined_preds[idx]
                counts = bdm.confusion(gt, pred, threshold=threshold)
                cand_metrics_list.append(counts)

                # Channel specific metrics using authentic class ground-truth if available
                gt_s = r.get("gt_scratch_mask")
                if gt_s is not None and supports_scratch:
                    scratch_metrics_list.append(bdm.confusion(gt_s, scratch_preds[idx], threshold=threshold))
                elif r["damage_type"] == "scratch" and supports_scratch:
                    scratch_metrics_list.append(bdm.confusion(gt, scratch_preds[idx], threshold=threshold))

                gt_d = r.get("gt_dust_mask")
                if gt_d is not None and supports_dust:
                    dust_metrics_list.append(bdm.confusion(gt_d, dust_preds[idx], threshold=threshold))
                elif r["damage_type"] == "dust" and supports_dust:
                    dust_metrics_list.append(bdm.confusion(gt, dust_preds[idx], threshold=threshold))

                # Diagnostic nuisance breakdown
                prev_f = frames[idx - 1] if idx > 0 else None
                nuis = da.evaluate_nuisance_fp_breakdown(
                    frames[idx], gt, pred, prev_frame=prev_f, threshold=threshold
                )
                nuisance_list.append(nuis)

            # 2. Sequence-level temporal consistency (if sequence has at least 3 frames)
            if len(frames) >= 3:
                flicker = tcs.compute_mask_flicker_rate(gt_masks, combined_preds, threshold=threshold)
                traj = tcs.compute_persistent_trajectory_metrics(gt_masks, combined_preds, threshold=threshold)
                mot_res = tcs.compute_temporal_motion_fpr(frames, gt_masks, combined_preds, threshold=threshold)
                temporal_metrics_list.append(
                    {
                        "flicker_rate": flicker,
                        "continuity_score": traj["detection_continuity_score"],
                        "missed_defect_rate": traj["missed_persistent_defect_rate"],
                        "temporal_motion_fpr": mot_res["temporal_false_positive_rate"],
                    }
                )

            # 3. Generate Contact Sheet for first frame of clip
            if contact_sheet_dir is not None and frames:
                cs_dir = Path(contact_sheet_dir)
                cs_dir.mkdir(parents=True, exist_ok=True)
                cs_filename = f"{cand}_{dataset_name}_{clip_name}_frame0.png"
                cs_path = cs_dir / cs_filename

                scratch_gts = [
                    (r.get("gt_scratch_mask") if r.get("gt_scratch_mask") is not None
                     else (np.where(r["gt_mask"] == 255, 255, 0).astype(np.uint8) if r["damage_type"] == "scratch" else np.zeros_like(r["gt_mask"])))
                    for r in clip_records
                ]
                dust_gts = [
                    (r.get("gt_dust_mask") if r.get("gt_dust_mask") is not None
                     else (np.where(r["gt_mask"] == 255, 255, 0).astype(np.uint8) if r["damage_type"] == "dust" else np.zeros_like(r["gt_mask"])))
                    for r in clip_records
                ]

                cde.generate_contact_sheet(
                    frames,
                    scratch_gts,
                    dust_gts,
                    scratch_preds,
                    dust_preds,
                    frame_index=0,
                    method_name=cand,
                    output_path=cs_path,
                )

        # Aggregate metrics
        agg_counts = bdm.sum_counts(cand_metrics_list) if cand_metrics_list else {}
        agg_metrics = bdm.metrics(agg_counts) if agg_counts else {}

        if supports_scratch and scratch_metrics_list:
            agg_scratch = bdm.metrics(bdm.sum_counts(scratch_metrics_list))
        else:
            agg_scratch = {"status": "NOT_APPLICABLE"} if not supports_scratch else {}

        if supports_dust and dust_metrics_list:
            agg_dust = bdm.metrics(bdm.sum_counts(dust_metrics_list))
        else:
            agg_dust = {"status": "NOT_APPLICABLE"} if not supports_dust else {}

        # Average nuisance FPR
        avg_nuisance: dict[str, float] = {}
        if nuisance_list:
            for region in ("edges", "grain", "faces_foreground", "motion"):
                vals = [n.get(region, {}).get("false_positive_rate", 0.0) for n in nuisance_list if region in n]
                avg_nuisance[f"{region}_fpr"] = float(np.mean(vals)) if vals else 0.0

        # Average temporal metrics
        avg_temporal: dict[str, float] = {}
        if temporal_metrics_list:
            for met in ("flicker_rate", "continuity_score", "missed_defect_rate", "temporal_motion_fpr"):
                vals = [t[met] for t in temporal_metrics_list if met in t]
                avg_temporal[met] = float(np.mean(vals)) if vals else 0.0

        # Frame-weighted latency and steady-state latency across processed frames
        latency_ms_per_frame = (total_time_ms / max(1, total_frames_processed))
        steady_state_ms = (steady_state_time_ms / max(1, steady_state_frames)) if steady_state_frames > 0 else latency_ms_per_frame

        dataset_results["candidates"][cand] = {
            "status": "EVALUATED" if cand_metrics_list else "NO_ELIGIBLE_FRAMES",
            "provenance": detected_provenance,
            "overall": agg_metrics,
            "scratch": agg_scratch,
            "dust": agg_dust,
            "nuisance": avg_nuisance,
            "temporal": avg_temporal,
            "latency_ms_per_frame": round(latency_ms_per_frame, 3),
            "cold_start_ms": round(cold_start_ms, 3),
            "steady_state_ms_per_frame": round(steady_state_ms, 3),
            "peak_memory_mb": round(max_peak_mem_bytes / (1024 * 1024), 2),
            "frames_processed": total_frames_processed,
            "total_time_ms": round(total_time_ms, 3),
        }

    return dataset_results


def format_markdown_report(report_data: dict[str, Any]) -> str:
    """Generate comprehensive, scientific Markdown benchmark report."""
    lines = [
        "# Fully Automated Dust & Scratch Benchmark Report (Validated)",
        "",
        f"**Generated:** {report_data.get('timestamp')}",
        f"**Execution Mode:** `{report_data.get('execution_mode')}`",
        "",
        "## 1. Environment Isolation & Safety Status",
        "",
        f"- **ARP Python:** `{report_data['isolation']['current_python']}`",
        f"- **Strict Isolation Status:** `{'PASS (Clean)' if report_data['isolation']['is_isolated'] else 'FAIL'}`",
        f"- **Prohibited Packages in ARP:** `{'None (100% Clean)' if not report_data['isolation']['violations'] else report_data['isolation']['violations']}`",
        f"- **DART Isolated Runtime:** `{report_data['isolation']['dart_python'] or 'Not Used'}`",
        f"- **AbsoluteDegradation Runtime:** `{report_data['isolation']['absdeg_python'] or 'Not Used'}`",
        "",
        "## 2. Benchmark Resource Inventory & Provenance",
        "",
        "| Resource | Provenance Tier | License | Commercial Clearance | Status | Available Masks | Samples Evaluated | Skipped / Pending |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]

    for k, res in report_data.get("resources", {}).items():
        status = res.get("status", "UNKNOWN")
        avail = res.get("available_masks", res.get("samples_evaluated", 0))
        evaluated = res.get("samples_evaluated", 0)
        skipped = res.get("samples_skipped", 0)
        lines.append(
            f"| **{res['name']}** | `{res['provenance_tier']}` | {res['license']} | {res['commercial_suitability']} | `{status}` | {avail} | {evaluated} | {skipped} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Overall Multi-Candidate Detector Comparison",
        "",
        "Aggregated across all validly evaluated sequence benchmarks (frame-weighted latency and max peak memory):",
        "",
        "| Candidate Detector | Backend Provenance | Evaluated Datasets | Precision | Recall | Pixel F1 | Macro F1 | Intact-Pixel FPR | Frame Latency | Peak Memory |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])

    summary_cand = report_data.get("candidate_summary", {})
    for cand_name, metrics in summary_cand.items():
        prov = metrics.get("provenance", "unknown")
        ov = metrics.get("overall", {})
        ds_scope = ", ".join(metrics.get("evaluated_datasets", [])) or "None"
        macro_f1 = f"{metrics.get('macro_f1', 0.0):.4f}" if "macro_f1" in metrics else "N/A"
        if ov.get("status") == "NOT_APPLICABLE":
            lines.append(f"| **{cand_name}** | `{prov}` | `{ds_scope}` | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* | *N/A* |")
            continue

        prec_def = ov.get("precision_defined", True)
        prec_val = ov.get("precision", 0.0)
        prec = f"{prec_val:.4f}" if prec_def else f"{prec_val:.4f}*"
        rec = f"{ov.get('recall', 0.0):.4f}"
        f1 = f"{ov.get('f1', 0.0):.4f}"
        fpr = f"{ov.get('false_positive_rate', 0.0) * 100:.3f}%"
        lat = f"{metrics.get('latency_ms_per_frame', 0.0):.2f} ms"
        mem = f"{metrics.get('peak_memory_mb', 0.0):.1f} MB"
        lines.append(f"| **{cand_name}** | `{prov}` | `{ds_scope}` | {prec} | {rec} | **{f1}** | {macro_f1} | {fpr} | {lat} | {mem} |")

    lines.extend([
        "",
        "\\* *Precision is undefined (0/0) when TP=0 and FP=0; reported as 0.0000 rather than falsely claiming 100% precision.*",
        "",
        "---",
        "",
        "## 4. Per-Class & Nuisance False-Positive Resistance",
        "",
        "| Candidate Detector | Scratch F1 | Dust F1 | Edge Regime FPR | Film Grain FPR | Inter-Frame Motion FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for cand_name, metrics in summary_cand.items():
        sc = metrics.get("scratch", {})
        du = metrics.get("dust", {})
        sc_f1 = f"{sc.get('f1', 0.0):.4f}" if "f1" in sc else ("N/A (unsupported)" if sc.get("status") == "NOT_APPLICABLE" else "N/A")
        du_f1 = f"{du.get('f1', 0.0):.4f}" if "f1" in du else ("N/A (unsupported)" if du.get("status") == "NOT_APPLICABLE" else "N/A")
        nu = metrics.get("nuisance", {})
        edge_fpr = f"{nu.get('edges_fpr', 0.0) * 100:.3f}%"
        grain_fpr = f"{nu.get('grain_fpr', 0.0) * 100:.3f}%"
        mot_fpr = f"{nu.get('motion_fpr', 0.0) * 100:.3f}%"
        lines.append(f"| **{cand_name}** | {sc_f1} | {du_f1} | {edge_fpr} | {grain_fpr} | {mot_fpr} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Temporal Video Consistency Metrics",
        "",
        "Evaluated on multi-frame video sequences (drifting scratch trajectories and video):",
        "",
        "| Candidate Detector | Continuity Score (Recall along defect) | Missed Trajectory Rate | Mask Flicker Rate (`0->1->0`) | Motion Interval FPR |",
        "|---|---:|---:|---:|---:|",
    ])

    for cand_name, metrics in summary_cand.items():
        tm = metrics.get("temporal", {})
        cont = f"{tm.get('continuity_score', 0.0):.4f}" if tm else "N/A"
        drop = f"{tm.get('missed_defect_rate', 0.0):.4f}" if tm else "N/A"
        flick = f"{tm.get('flicker_rate', 0.0):.6f}" if tm else "N/A"
        mfpr = f"{tm.get('temporal_motion_fpr', 0.0) * 100:.4f}%" if tm else "N/A"
        lines.append(f"| **{cand_name}** | {cont} | {drop} | {flick} | {mfpr} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Per-Dataset Evaluation Breakdown & Scientific Integrity",
        "",
    ])

    for ds_name, ds_res in report_data.get("datasets", {}).items():
        lines.extend([
            f"### Dataset: `{ds_name}` (Total Samples: {ds_res.get('total_samples', 0)})",
            "",
            "| Candidate | Status | Backend Provenance | Overall F1 | Recall | Precision | Notes |",
            "|---|---|---|---:|---:|---:|---|",
        ])
        for cname, cinfo in ds_res.get("candidates", {}).items():
            status = cinfo.get("status", "EVALUATED")
            prov = cinfo.get("provenance", "unknown")
            ov = cinfo.get("overall", {})
            if "f1" in ov:
                f1_str = f"{ov['f1']:.4f}"
                rec_str = f"{ov.get('recall', 0.0):.4f}"
                p_val = ov.get('precision', 0.0)
                p_str = f"{p_val:.4f}" if ov.get("precision_defined", True) else f"{p_val:.4f}*"
                note = f"TP={ov.get('tp', 0)}, FP={ov.get('fp', 0)}, FN={ov.get('fn', 0)}"
            else:
                f1_str = "N/A"
                rec_str = "N/A"
                p_str = "N/A"
                note = ov.get("reason", cinfo.get("reason", status))
            lines.append(f"| **{cname}** | `{status}` | `{prov}` | {f1_str} | {rec_str} | {p_str} | {note} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 7. Real Dataset vs. Synthetic / Model Pseudo-Label Distinction",
        "",
        "1. **Real Human Ground Truth:**",
        "   - `FilmDamageSimulator` (`Scan (8)`, `Scan (10)`): Real physical film scans with human-labeled defect polygons. Single-frame scans lack temporal context for multi-frame optical flow detectors; evaluated with single-frame spatial detectors (`legacy`).",
        "   - `ARTeFACT` (`danielaivanova/damaged-media`): Real historical media pixel-level human segmentation PNGs (AFL-3.0; offline research only). Single-frame patches lack multi-frame context.",
        "   - `Neighbors (1920)`: 6 real native Buster Keaton frames (`evaluation/Neighbors_72.mkv`); status remains strictly `pending` human sign-off without substituting blank masks as negative labels.",
        "",
        "2. **Synthetic Parametric Ground Truth:**",
        "   - `AbsoluteDegradation`: Multi-frame parametric vertical scratches injected with variable curvature and drift, preserving float32 opacity masks (`opacity >= 0.10 -> 255`, `0 < opacity < 0.10 -> 127 ignore`).",
        "   - `Synthetic benchmark`: In-tree generator producing disjoint scratch trajectories and transient dust clusters on solid/textured backgrounds.",
        "",
        "3. **Model Pseudo-Labels (Strictly Isolated):**",
        "   - `DART MaskNet`: 72 precomputed Buster Keaton soft masks (`evaluation/dart_trial_01/masks`), evaluated strictly as Candidate D in an external harness. Never treated as ground truth or imported into commercial code.",
        "",
        "## 8. Visual Contact Sheets",
        "",
        f"Diagnostic 6-panel contact sheets (Source, Ground Truth, Scratch Separation, Dust Separation, Combined, Error Map) saved under `{report_data.get('contact_sheet_dir')}`.",
    ])

    return "\n".join(lines) + "\n"


def run_benchmark_suite(
    output_dir: Path,
    cache_dir: Path | None = None,
    *,
    candidate_names: Sequence[str] | None = None,
    acknowledge_research: bool = False,
    skip_provision: bool = False,
    sensitivity: float = 0.65,
    threshold: int = 128,
) -> dict[str, Any]:
    """Execute the complete end-to-end benchmark workflow."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir or ROOT_DIR / ".cache/benchmarks").resolve()

    # Determine candidate list and enforce commercial safety gating
    if candidate_names is None:
        candidates = list(COMMERCIAL_CANDIDATES if not acknowledge_research else (COMMERCIAL_CANDIDATES + RESEARCH_CANDIDATES))
    else:
        candidates = list(candidate_names)

    if not acknowledge_research:
        research_requested = [c for c in candidates if c in RESEARCH_CANDIDATES]
        if research_requested:
            raise fdd.CommercialLicenseError(
                f"Candidate(s) {research_requested} are research-restricted and blocked in commercial_strict mode. "
                "Pass --acknowledge-research to explicitly opt in for offline research evaluation."
            )

    # Step 1: Environment Isolation Audit
    iso = pbr.audit_environment_isolation()
    if not iso["is_isolated"]:
        raise RuntimeError(f"Strict Python environment isolation violated: {iso['violations']}")

    # Step 2: Automatic Provisioning (if needed)
    if not skip_provision:
        pbr.provision_all(
            cache,
            acknowledge_research=acknowledge_research,
            force=False,
        )

    # Step 3: Load Datasets
    contact_sheet_dir = output_dir / "contact_sheets"
    contact_sheet_dir.mkdir(parents=True, exist_ok=True)

    available_resources = pbr.audit_all_resources(cache)
    dataset_reports: dict[str, Any] = {}

    external_dart_dir = ROOT_DIR / "evaluation/dart_trial_01/masks"

    # Select datasets based on commercial vs research mode
    dataset_dirs = {
        "synthetic": cache / "synthetic",
        "film_damage_simulator": cache / "fds",
        "absolute_degradation": cache / "absolute_degradation",
    }
    if acknowledge_research:
        dataset_dirs["artefact"] = cache / "artefact"
        dataset_dirs["neighbors_pilot"] = cache / "neighbors_pilot"

    total_samples_per_res: dict[str, int] = {}
    all_dataset_evals = []

    for name, ddir in dataset_dirs.items():
        recs, invalid_samples = load_dataset_records(ddir, return_invalid=True)
        total_samples_per_res[name] = len(recs)
        if not recs:
            continue

        # Neighbors pilot: annotation_status is pending; skip confusing unreviewed frames as ground truth
        if name == "neighbors_pilot":
            continue

        eval_res = evaluate_dataset(
            name,
            recs,
            candidates,
            sensitivity=sensitivity,
            threshold=threshold,
            external_dart_dir=external_dart_dir,
            contact_sheet_dir=contact_sheet_dir,
            acknowledge_research=acknowledge_research,
            invalid_samples=invalid_samples,
        )
        dataset_reports[name] = eval_res
        all_dataset_evals.append(eval_res)

    # Step 4: Aggregate across all evaluated datasets
    candidate_summary: dict[str, Any] = {}
    for cand in candidates:
        cand_time_ms = 0.0
        cand_frames = 0
        cand_mems = [0.0]
        combined_overall_counts = []
        combined_scratch_counts = []
        combined_dust_counts = []
        combined_nuisance: dict[str, list[float]] = {}
        combined_temporal: dict[str, list[float]] = {}
        candidate_provenance = "unknown"

        evaluated_datasets = []
        dataset_f1s = []
        for ds_res in all_dataset_evals:
            cand_res = ds_res.get("candidates", {}).get(cand)
            if not cand_res:
                continue
            if cand_res.get("status") in ("INSUFFICIENT_TEMPORAL_CONTEXT", "EXCLUDED_COMMERCIAL_STRICT"):
                continue

            candidate_provenance = cand_res.get("provenance", candidate_provenance)
            cand_time_ms += cand_res.get("total_time_ms", 0.0)
            cand_frames += cand_res.get("frames_processed", 0)
            cand_mems.append(cand_res.get("peak_memory_mb", 0.0))
            evaluated_datasets.append(ds_res.get("dataset"))

            ov = cand_res.get("overall", {})
            if "tp" in ov:
                combined_overall_counts.append(ov)
                if "f1" in ov:
                    dataset_f1s.append(ov["f1"])

            sc = cand_res.get("scratch", {})
            if "tp" in sc:
                combined_scratch_counts.append(sc)

            du = cand_res.get("dust", {})
            if "tp" in du:
                combined_dust_counts.append(du)

            for n_reg, n_val in cand_res.get("nuisance", {}).items():
                combined_nuisance.setdefault(n_reg, []).append(n_val)

            for t_reg, t_val in cand_res.get("temporal", {}).items():
                combined_temporal.setdefault(t_reg, []).append(t_val)

        overall_metrics = bdm.metrics(bdm.sum_counts(combined_overall_counts)) if combined_overall_counts else {"status": "NOT_APPLICABLE"}
        scratch_metrics = bdm.metrics(bdm.sum_counts(combined_scratch_counts)) if combined_scratch_counts else {"status": "NOT_APPLICABLE"}
        dust_metrics = bdm.metrics(bdm.sum_counts(combined_dust_counts)) if combined_dust_counts else {"status": "NOT_APPLICABLE"}

        avg_nuis = {k: float(np.mean(v)) for k, v in combined_nuisance.items()}
        avg_temp = {k: float(np.mean(v)) for k, v in combined_temporal.items()}

        frame_weighted_latency = (cand_time_ms / max(1, cand_frames)) if cand_frames > 0 else 0.0
        macro_f1 = float(np.mean(dataset_f1s)) if dataset_f1s else 0.0

        candidate_summary[cand] = {
            "provenance": candidate_provenance,
            "overall": overall_metrics,
            "macro_f1": round(macro_f1, 4),
            "evaluated_datasets": evaluated_datasets,
            "scratch": scratch_metrics,
            "dust": dust_metrics,
            "nuisance": avg_nuis,
            "temporal": avg_temp,
            "latency_ms_per_frame": round(frame_weighted_latency, 2),
            "peak_memory_mb": round(float(np.max(cand_mems)), 2),
            "total_frames_evaluated": cand_frames,
        }

    # Annotate resources with exact sample counts
    resource_to_dataset_key = {
        "synthetic_benchmark": "synthetic",
        "film_damage_simulator": "film_damage_simulator",
        "absolute_degradation": "absolute_degradation",
        "artefact": "artefact",
        "dart_harness": "dart_harness",
        "neighbors_pilot": "neighbors_pilot",
    }
    for k, v in available_resources.items():
        ds_key = resource_to_dataset_key.get(k, k)
        if k == "dart_harness":
            # DART Buster Keaton pseudo-labels are pending human ground truth sign-off
            v["available_masks"] = 72
            v["valid_prediction_masks"] = 72
            v["samples_evaluated"] = total_samples_per_res.get("dart_harness", 0)
            v["samples_skipped"] = 72
            v["samples_without_verified_ground_truth"] = 72
        else:
            v["samples_evaluated"] = total_samples_per_res.get(ds_key, 0)

    final_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_mode": "research_offline_evaluation" if acknowledge_research else "commercial_strict",
        "isolation": iso,
        "resources": available_resources,
        "datasets": dataset_reports,
        "candidate_summary": candidate_summary,
        "contact_sheet_dir": str(contact_sheet_dir),
    }

    report_json_path = output_dir / "benchmark_report.json"
    report_md_path = output_dir / "benchmark_report.md"

    report_json_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    report_md_text = format_markdown_report(final_report)
    report_md_path.write_text(report_md_text, encoding="utf-8")

    return {
        "report_json": report_json_path,
        "report_md": report_md_path,
        "summary": candidate_summary,
        "isolation": iso,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / ".cache/benchmark_suite_report_validated_20261111",
        help="Destination directory for benchmark report artifacts (preserves baseline)",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT_DIR / ".cache/benchmarks")
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=COMMERCIAL_CANDIDATES + RESEARCH_CANDIDATES,
        default=None,
        help="Candidate detector identifiers to evaluate",
    )
    parser.add_argument(
        "--acknowledge-research",
        action="store_true",
        default=False,
        help="Explicitly acknowledge research terms for ARTeFACT (AFL-3.0) and DART (RTN research-only)",
    )
    parser.add_argument("--skip-provision", action="store_true", help="Skip provisioning and use existing provisioned data")
    parser.add_argument("--sensitivity", type=float, default=0.65)
    parser.add_argument("--threshold", type=int, default=128)
    args = parser.parse_args(argv)

    mode_label = "research_offline_evaluation" if args.acknowledge_research else "commercial_strict"
    print(f"Running Dust & Scratch Benchmark Suite [Mode: {mode_label}]...")
    res = run_benchmark_suite(
        args.output_dir,
        cache_dir=args.cache_dir,
        candidate_names=args.candidates,
        acknowledge_research=args.acknowledge_research,
        skip_provision=args.skip_provision,
        sensitivity=args.sensitivity,
        threshold=args.threshold,
    )
    print("Benchmark run complete!")
    print(f"  JSON Report:     {res['report_json']}")
    print(f"  Markdown Report: {res['report_md']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
