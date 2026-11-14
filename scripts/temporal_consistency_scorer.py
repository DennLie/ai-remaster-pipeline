"""Temporal consistency scorer for archival film defect detection masks.

Evaluates multi-frame predicted masks against ground-truth masks and video frames,
reusing `benchmark_defect_masks.py` for pixel-level confusion and spatial metrics
while adding four sequence-level temporal consistency metrics required by Phase 2:

1. `mask_flicker_rate`: Frequency of single-frame alternating prediction states
   (`0 -> 1 -> 0` or `1 -> 0 -> 1`) on static/unaffected background pixels (`gt == 0`).
2. `detection_continuity_score`: Proportion of active frames along a persistent
   multi-frame defect trajectory where the defect is detected (`>= min_frame_recall`).
3. `temporal_false_positive_rate`: False-positive rate on unaffected pixels during
   high-motion intervals (`|I_t - I_{t-1}| >= motion_diff_threshold`).
4. `missed_persistent_defect_rate`: Proportion of internal trajectory frames along a
   persistent multi-frame defect where detection drops out (`1.0 - detection_continuity_score`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import benchmark_defect_masks as bdm


def _as_uint8_2d_list(masks: Sequence[np.ndarray] | np.ndarray, label: str) -> list[np.ndarray]:
    if isinstance(masks, np.ndarray):
        if masks.ndim != 3:
            raise ValueError(f"{label} numpy array must be 3D (T,H,W), got {masks.shape}")
        seq = [masks[i] for i in range(masks.shape[0])]
    else:
        seq = [np.asarray(m) for m in masks]
    if len(seq) < 2:
        raise ValueError(f"{label} sequence requires at least 2 frames, got {len(seq)}")
    shape = seq[0].shape
    for idx, arr in enumerate(seq):
        if arr.ndim != 2 or arr.dtype != np.uint8 or arr.shape != shape:
            raise ValueError(f"{label}[{idx}] must be 2D uint8 with shape {shape}, got {arr.shape} ({arr.dtype})")
    return seq


def compute_mask_flicker_rate(
    gt_masks: Sequence[np.ndarray] | np.ndarray,
    pred_masks: Sequence[np.ndarray] | np.ndarray,
    *,
    threshold: int = 128,
) -> float:
    """Compute single-frame prediction state alternation rate on unaffected background pixels.

    For every interior frame `t in [1, T-2]`, a pixel is eligible if `gt[t-1] == 0`,
    `gt[t] == 0`, and `gt[t+1] == 0`. A flicker event occurs when the thresholded
    prediction at `t` disagrees with both `t-1` and `t+1` (i.e. `0->1->0` or `1->0->1`).
    """
    gts = _as_uint8_2d_list(gt_masks, "gt_masks")
    preds = _as_uint8_2d_list(pred_masks, "pred_masks")
    if len(gts) != len(preds):
        raise ValueError(f"Sequence lengths differ: gt={len(gts)} vs pred={len(preds)}")
    if len(gts) < 3:
        return 0.0

    flicker_events = 0
    eligible_triplets = 0
    bin_preds = [p >= threshold for p in preds]
    clean_gts = [g == 0 for g in gts]

    for t in range(1, len(gts) - 1):
        eligible = clean_gts[t - 1] & clean_gts[t] & clean_gts[t + 1]
        n_el = int(np.count_nonzero(eligible))
        if n_el == 0:
            continue
        p_prev, p_curr, p_next = bin_preds[t - 1], bin_preds[t], bin_preds[t + 1]
        alternating = (p_prev == p_next) & (p_curr != p_prev) & eligible
        flicker_events += int(np.count_nonzero(alternating))
        eligible_triplets += n_el

    return round(flicker_events / eligible_triplets, 8) if eligible_triplets > 0 else 0.0


def compute_persistent_trajectory_metrics(
    gt_masks: Sequence[np.ndarray] | np.ndarray,
    pred_masks: Sequence[np.ndarray] | np.ndarray,
    *,
    threshold: int = 128,
    min_frame_recall: float = 0.05,
    min_persistence_frames: int = 2,
) -> dict[str, float | int]:
    """Evaluate detection continuity and mid-trajectory dropout rate on persistent defects.

    Identifies contiguous runs of frames (length >= `min_persistence_frames`) containing
    ground-truth defect pixels (`gt == 255`), and measures the fraction of those frames
    where the detector achieves at least `min_frame_recall` on the ground-truth defect.
    """
    gts = _as_uint8_2d_list(gt_masks, "gt_masks")
    preds = _as_uint8_2d_list(pred_masks, "pred_masks")
    if len(gts) != len(preds):
        raise ValueError(f"Sequence lengths differ: gt={len(gts)} vs pred={len(preds)}")

    has_gt = [bool(np.any(g == 255)) for g in gts]
    frame_detected: list[bool] = []
    for gt, pr in zip(gts, preds):
        c = bdm.confusion(gt, pr, threshold=threshold)
        denom = c["tp"] + c["fn"]
        recall = (c["tp"] / denom) if denom > 0 else 0.0
        frame_detected.append(recall >= min_frame_recall)

    # Group contiguous active GT frames into persistent runs
    runs: list[list[int]] = []
    current_run: list[int] = []
    for idx, active in enumerate(has_gt):
        if active:
            current_run.append(idx)
        elif current_run:
            if len(current_run) >= min_persistence_frames:
                runs.append(current_run)
            current_run = []
    if current_run and len(current_run) >= min_persistence_frames:
        runs.append(current_run)

    total_persistent_frames = sum(len(r) for r in runs)
    detected_persistent_frames = sum(
        1 for r in runs for idx in r if frame_detected[idx]
    )
    if total_persistent_frames == 0:
        return {
            "persistent_runs": 0,
            "persistent_frames": 0,
            "detected_persistent_frames": 0,
            "detection_continuity_score": 1.0,
            "missed_persistent_defect_rate": 0.0,
        }

    continuity = round(detected_persistent_frames / total_persistent_frames, 6)
    missed = round(1.0 - continuity, 6)
    return {
        "persistent_runs": len(runs),
        "persistent_frames": total_persistent_frames,
        "detected_persistent_frames": detected_persistent_frames,
        "detection_continuity_score": continuity,
        "missed_persistent_defect_rate": missed,
    }


def compute_temporal_motion_fpr(
    frames: Sequence[np.ndarray] | np.ndarray,
    gt_masks: Sequence[np.ndarray] | np.ndarray,
    pred_masks: Sequence[np.ndarray] | np.ndarray,
    *,
    threshold: int = 128,
    motion_diff_threshold: float = 12.0,
) -> dict[str, float | int]:
    """Compute false-positive rate specifically on unaffected pixels undergoing inter-frame motion."""
    gts = _as_uint8_2d_list(gt_masks, "gt_masks")
    preds = _as_uint8_2d_list(pred_masks, "pred_masks")
    norm_frames = bdm._normalize_frame_sequence(frames)
    if not (len(norm_frames) == len(gts) == len(preds)):
        raise ValueError("frames, gt_masks, and pred_masks must have the same length")

    gray_frames = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
        for f in norm_frames
    ]

    motion_counts: list[dict[str, int]] = []
    static_counts: list[dict[str, int]] = []
    for t in range(1, len(gray_frames)):
        diff = cv2.absdiff(gray_frames[t], gray_frames[t - 1])
        is_motion = diff >= motion_diff_threshold
        gt_motion = np.where(is_motion, gts[t], np.uint8(127))
        gt_static = np.where(~is_motion, gts[t], np.uint8(127))
        motion_counts.append(bdm.confusion(gt_motion, preds[t], threshold=threshold))
        static_counts.append(bdm.confusion(gt_static, preds[t], threshold=threshold))

    motion_m = bdm.metrics(bdm.sum_counts(motion_counts))
    static_m = bdm.metrics(bdm.sum_counts(static_counts))
    return {
        "temporal_false_positive_rate": motion_m["false_positive_rate"],
        "static_false_positive_rate": static_m["false_positive_rate"],
        "motion_evaluated_pixels": motion_m["evaluated_pixels"],
        "motion_fp_pixels": motion_m["fp"],
        "static_evaluated_pixels": static_m["evaluated_pixels"],
        "static_fp_pixels": static_m["fp"],
    }


def evaluate_sequence_temporal_consistency(
    gt_masks: Sequence[np.ndarray] | np.ndarray,
    pred_masks: Sequence[np.ndarray] | np.ndarray,
    *,
    frames: Sequence[np.ndarray] | np.ndarray | None = None,
    threshold: int = 128,
    min_frame_recall: float = 0.05,
    motion_diff_threshold: float = 12.0,
) -> dict[str, Any]:
    """Compute full spatial + temporal consistency summary for a sequence of masks."""
    gts = _as_uint8_2d_list(gt_masks, "gt_masks")
    preds = _as_uint8_2d_list(pred_masks, "pred_masks")
    per_frame_counts = [bdm.confusion(g, p, threshold=threshold) for g, p in zip(gts, preds)]
    spatial_summary = bdm.metrics(bdm.sum_counts(per_frame_counts))
    flicker = compute_mask_flicker_rate(gts, preds, threshold=threshold)
    traj = compute_persistent_trajectory_metrics(
        gts, preds, threshold=threshold, min_frame_recall=min_frame_recall
    )
    if frames is not None:
        motion_info = compute_temporal_motion_fpr(
            frames,
            gts,
            preds,
            threshold=threshold,
            motion_diff_threshold=motion_diff_threshold,
        )
    else:
        motion_info = {
            "temporal_false_positive_rate": spatial_summary["false_positive_rate"],
            "static_false_positive_rate": spatial_summary["false_positive_rate"],
            "motion_evaluated_pixels": 0,
            "motion_fp_pixels": 0,
            "static_evaluated_pixels": spatial_summary["evaluated_pixels"],
            "static_fp_pixels": spatial_summary["fp"],
        }

    return {
        "spatial_metrics": spatial_summary,
        "mask_flicker_rate": flicker,
        "detection_continuity_score": traj["detection_continuity_score"],
        "missed_persistent_defect_rate": traj["missed_persistent_defect_rate"],
        "temporal_false_positive_rate": motion_info["temporal_false_positive_rate"],
        "trajectory_details": traj,
        "motion_details": motion_info,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True, help="Directory of sorted GT PNG masks")
    parser.add_argument("--pred-dir", type=Path, required=True, help="Directory of sorted predicted PNG masks")
    parser.add_argument("--frames-dir", type=Path, help="Optional directory of sorted source frames")
    parser.add_argument("--threshold", type=int, default=128, help="Prediction threshold 1..255")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report path")
    args = parser.parse_args(argv)

    gt_files = sorted(args.gt_dir.glob("*.png"))
    pred_files = sorted(args.pred_dir.glob("*.png"))
    if not gt_files or len(gt_files) != len(pred_files):
        parser.error("Expected matching non-empty *.png sequences in --gt-dir and --pred-dir")

    gts = [bdm.read_mask(p, ground_truth=True) for p in gt_files]
    preds = [bdm.read_mask(p, ground_truth=False) for p in pred_files]
    frames = None
    if args.frames_dir is not None:
        frame_files = sorted(args.frames_dir.glob("*.png"))
        if len(frame_files) != len(gt_files):
            parser.error("--frames-dir count must match --gt-dir count")
        frames = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in frame_files]

    result = evaluate_sequence_temporal_consistency(
        gts, preds, frames=frames, threshold=args.threshold
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Temporal Consistency: continuity={result['detection_continuity_score']:.3f} "
        f"missed={result['missed_persistent_defect_rate']:.3f} "
        f"flicker={result['mask_flicker_rate']:.6f} "
        f"temporal_FPR={result['temporal_false_positive_rate']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
