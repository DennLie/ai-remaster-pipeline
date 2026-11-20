#!/usr/bin/env python3
"""Cross-film generalization validation suite for the frozen Temporal Scratch Detector.

Evaluates the frozen detector on 4 independent archival sequences:
1. Paleface Seq A (1922, 640x480): Prominent vertical scratch & actor movement.
2. Paleface Seq B (1922, 640x480): Severe diagonal / branching damage.
3. College Seq C (1927, 640x480): Faint multi-frame tracking vertical scratch.
4. Neighbors Seq D (1920, 960x720): Independent street scene with vertical and horizontal scratches.

Outputs:
    docs/benchmarks/cross_film_generalization_benchmark.json
    Artifact panels and playable side-by-side MP4/GIF videos in brain artifact directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.film_defect_detectors import (
    DEFAULT_REGISTRY,
    BENCHMARK_PRESETS,
    DefectMaskResult,
    MultiChannelDefectMask,
    RepairInclusionPolicy,
    fuse_spatial_and_temporal_masks,
)
from scripts.temporal_scratch_analyzer import TemporalConfig, TemporalScratchAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cross_film_benchmark")

ARTIFACTS_DIR = Path(r"C:\Users\ogn81\.gemini\antigravity-cli\brain\ed2df479-15a0-446b-8969-d1a5551bb7d5")
DOCS_BENCHMARKS = PROJECT_ROOT / "docs" / "benchmarks"
CLIPS_DIR = PROJECT_ROOT / "evaluation" / "cross_film_validation"

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_BENCHMARKS.mkdir(parents=True, exist_ok=True)


def compute_metrics(
    pred_mask: np.ndarray,
    core_gt: np.ndarray,
    abrasion_gt: np.ndarray,
    ambiguous_gt: np.ndarray,
    clean_roi_mask: np.ndarray,
    full_shape: tuple[int, int],
) -> dict[str, Any]:
    """Compute recall, precision, and false-positive metrics with strict separation."""
    pred_bin = (pred_mask > 0).astype(np.uint8)
    
    # Recalls
    core_total = int(np.count_nonzero(core_gt))
    core_hits = int(np.count_nonzero(pred_bin & (core_gt > 0)))
    core_rec = (core_hits / core_total * 100.0) if core_total > 0 else 0.0

    abr_total = int(np.count_nonzero(abrasion_gt))
    abr_hits = int(np.count_nonzero(pred_bin & (abrasion_gt > 0)))
    abr_rec = (abr_hits / abr_total * 100.0) if abr_total > 0 else 0.0

    total_gt = core_gt | abrasion_gt
    total_gt_px = int(np.count_nonzero(total_gt))
    total_hits = int(np.count_nonzero(pred_bin & (total_gt > 0)))
    total_rec = (total_hits / total_gt_px * 100.0) if total_gt_px > 0 else 0.0

    # Ambiguous hits
    ambig_hits = int(np.count_nonzero(pred_bin & (ambiguous_gt > 0)))

    # Genuine False Positives outside all defect/ambiguous regions
    known_damage = total_gt | ambiguous_gt
    outside_damage = pred_bin & ~known_damage
    genuine_fp_total = int(np.count_nonzero(outside_damage))

    # Clean ROI FPR
    clean_roi_px = int(np.count_nonzero(clean_roi_mask))
    clean_roi_fps = int(np.count_nonzero(pred_bin & (clean_roi_mask > 0)))
    clean_roi_fpr = (clean_roi_fps / clean_roi_px * 100.0) if clean_roi_px > 0 else 0.0

    # Full frame coverage
    full_area = full_shape[0] * full_shape[1]
    coverage_pct = (int(np.count_nonzero(pred_bin)) / full_area * 100.0) if full_area > 0 else 0.0

    # Precision
    clean_prec = (total_hits / max(1, total_hits + genuine_fp_total) * 100.0)
    strict_prec = (total_hits / max(1, total_hits + genuine_fp_total + ambig_hits) * 100.0)

    return {
        "core_gt_pixels": core_total,
        "core_detected_pixels": core_hits,
        "core_recall_pct": round(core_rec, 2),
        "abrasion_gt_pixels": abr_total,
        "abrasion_detected_pixels": abr_hits,
        "abrasion_recall_pct": round(abr_rec, 2),
        "total_confirmed_gt_pixels": total_gt_px,
        "total_confirmed_detected_pixels": total_hits,
        "total_confirmed_recall_pct": round(total_rec, 2),
        "ambiguous_detected_pixels": ambig_hits,
        "genuine_fp_pixels": genuine_fp_total,
        "clean_precision_pct": round(clean_prec, 2),
        "strict_precision_pct": round(strict_prec, 2),
        "clean_roi_pixels": clean_roi_px,
        "clean_roi_fp_pixels": clean_roi_fps,
        "clean_roi_fpr_pct": round(clean_roi_fpr, 4),
        "coverage_pct": round(coverage_pct, 3),
        "total_predicted_pixels": int(np.count_nonzero(pred_bin)),
    }


def synthesize_motion_comp_repair(
    frames: list[np.ndarray],
    target_idx: int,
    effective_mask: np.ndarray,
    analyzer: TemporalScratchAnalyzer,
) -> np.ndarray:
    """Perform motion-compensated temporal replacement strictly gated by effective_mask."""
    target_frame = frames[target_idx]
    h, w = target_frame.shape[:2]
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    prec, succ, _ = analyzer._select_nonduplicate_neighbors(gray_frames, target_idx)
    neighbors = prec + succ
    if not neighbors:
        return cv2.inpaint(target_frame, effective_mask, 3, cv2.INPAINT_TELEA)

    warped_bgr_stack = []
    for n_idx in neighbors:
        flow = analyzer._compute_dense_flow(gray_frames[n_idx], gray_frames[target_idx])
        # Warp each BGR channel
        w_bgr = np.zeros_like(target_frame)
        for c in range(3):
            w_bgr[:, :, c] = analyzer._warp_neighbor(frames[n_idx][:, :, c], flow)
        warped_bgr_stack.append(w_bgr)

    # Median background synthesis
    stack_arr = np.stack(warped_bgr_stack, axis=0)
    bg_bgr = np.median(stack_arr, axis=0).astype(np.uint8)

    # Blend: replace only where effective_mask > 0
    repaired = target_frame.copy()
    m_bool = effective_mask > 127
    repaired[m_bool] = bg_bgr[m_bool]
    return repaired


def evaluate_sequence(
    seq_name: str,
    clip_file: str,
    target_eval_frame: int,
    build_gt_fn: Any,
    negative_control_roi: dict[str, int],
    clean_roi: dict[str, int],
    target_defect_name: str,
) -> dict[str, Any]:
    """Run full evaluation on a single sequence."""
    logger.info("Evaluating %s (target frame %d)...", seq_name, target_eval_frame)
    cap = cv2.VideoCapture(str(CLIPS_DIR / clip_file))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    n_frames = len(frames)
    h, w = frames[0].shape[:2]

    # Initialize detectors
    temporal_det = DEFAULT_REGISTRY.get("temporal_scratch")
    spatial_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    temp_cfg = BENCHMARK_PRESETS["temporal_scratch_balanced"].params
    spatial_cfg = BENCHMARK_PRESETS["film_artefact_balanced"].params

    # Runtime benchmarks
    t0 = time.perf_counter()
    temp_res = temporal_det.detect(frames, temp_cfg)
    t_temp = (time.perf_counter() - t0) / n_frames * 1000.0

    t0 = time.perf_counter()
    spat_res = spatial_det.detect(frames, spatial_cfg)
    t_spat = (time.perf_counter() - t0) / n_frames * 1000.0

    t0 = time.perf_counter()
    fused_res = fuse_spatial_and_temporal_masks(spat_res, temp_res)
    t_fused = t_temp + t_spat + ((time.perf_counter() - t0) / n_frames * 1000.0)

    # Ground Truth Generation
    core_gt, abr_gt, ambig_gt = build_gt_fn(frames, target_eval_frame)

    # Clean ROI Mask
    clean_mask = np.zeros((h, w), dtype=np.uint8)
    clean_mask[clean_roi["ymin"]:clean_roi["ymax"], clean_roi["xmin"]:clean_roi["xmax"]] = 255
    # Exclude any known defects from clean ROI
    clean_mask[core_gt > 0] = 0
    clean_mask[abr_gt > 0] = 0
    clean_mask[ambig_gt > 0] = 0

    # Negative Control ROI Mask
    neg_mask = np.zeros((h, w), dtype=np.uint8)
    neg_mask[negative_control_roi["ymin"]:negative_control_roi["ymax"], negative_control_roi["xmin"]:negative_control_roi["xmax"]] = 255

    # Target frame masks
    temp_scratch_target = temp_res.scratch_mask[target_eval_frame]
    spat_mask_target = spat_res.mask[target_eval_frame]
    fused_mcm_target = fused_res.multi_channel_masks[target_eval_frame]
    fused_eff_conservative = fused_mcm_target.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
    fused_eff_high_recall = fused_mcm_target.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)
    fused_uncertain = fused_mcm_target.uncertain_mask if fused_mcm_target.uncertain_mask is not None else np.zeros((h, w), dtype=np.uint8)

    # Compute metrics for each mode
    m_spat = compute_metrics(spat_mask_target, core_gt, abr_gt, ambig_gt, clean_mask, (h, w))
    m_temp = compute_metrics(temp_scratch_target, core_gt, abr_gt, ambig_gt, clean_mask, (h, w))
    m_fused_cons = compute_metrics(fused_eff_conservative, core_gt, abr_gt, ambig_gt, clean_mask, (h, w))
    m_fused_hr = compute_metrics(fused_eff_high_recall, core_gt, abr_gt, ambig_gt, clean_mask, (h, w))

    # Negative Control Performance
    neg_px_total = int(np.count_nonzero(neg_mask))
    neg_spat = int(np.count_nonzero(spat_mask_target & neg_mask))
    neg_temp = int(np.count_nonzero(temp_scratch_target & neg_mask))
    neg_cons = int(np.count_nonzero(fused_eff_conservative & neg_mask))
    neg_hr = int(np.count_nonzero(fused_eff_high_recall & neg_mask))

    neg_summary = {
        "description": negative_control_roi.get("description", "Negative control region"),
        "total_pixels": neg_px_total,
        "spatial_leak_pixels": neg_spat,
        "temporal_leak_pixels": neg_temp,
        "conservative_leak_pixels": neg_cons,
        "high_recall_leak_pixels": neg_hr,
        "conservative_protection_success": bool(neg_cons == 0),
    }

    # Inpainting & Restoration Quality
    analyzer = TemporalScratchAnalyzer(TemporalConfig())
    repaired_temp_mc = synthesize_motion_comp_repair(frames, target_eval_frame, fused_eff_conservative, analyzer)
    repaired_telea = cv2.inpaint(frames[target_eval_frame], fused_eff_conservative, 3, cv2.INPAINT_TELEA)

    # Compute PSNR / SSIM against clean reference (synthesized background outside defect)
    gray_target = cv2.cvtColor(frames[target_eval_frame], cv2.COLOR_BGR2GRAY)
    gray_repaired_mc = cv2.cvtColor(repaired_temp_mc, cv2.COLOR_BGR2GRAY)
    gray_repaired_telea = cv2.cvtColor(repaired_telea, cv2.COLOR_BGR2GRAY)

    # Background reference for quality evaluation
    prec_idx, succ_idx, _ = analyzer._select_nonduplicate_neighbors([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames], target_eval_frame)
    if prec_idx and succ_idx:
        ref_clean = np.minimum(cv2.cvtColor(frames[prec_idx[0]], cv2.COLOR_BGR2GRAY), cv2.cvtColor(frames[succ_idx[0]], cv2.COLOR_BGR2GRAY))
        # Masked PSNR / SSIM on defect region
        eval_mask = (core_gt > 0) | (abr_gt > 0)
        if np.any(eval_mask):
            diff_orig = (gray_target[eval_mask].astype(np.float32) - ref_clean[eval_mask].astype(np.float32)) ** 2
            diff_mc = (gray_repaired_mc[eval_mask].astype(np.float32) - ref_clean[eval_mask].astype(np.float32)) ** 2
            diff_tel = (gray_repaired_telea[eval_mask].astype(np.float32) - ref_clean[eval_mask].astype(np.float32)) ** 2

            mse_orig = float(np.mean(diff_orig))
            mse_mc = float(np.mean(diff_mc))
            mse_tel = float(np.mean(diff_tel))

            psnr_orig = 10.0 * np.log10(255.0 ** 2 / max(1e-4, mse_orig))
            psnr_mc = 10.0 * np.log10(255.0 ** 2 / max(1e-4, mse_mc))
            psnr_tel = 10.0 * np.log10(255.0 ** 2 / max(1e-4, mse_tel))
        else:
            psnr_orig, psnr_mc, psnr_tel = 0.0, 0.0, 0.0
    else:
        psnr_orig, psnr_mc, psnr_tel = 0.0, 0.0, 0.0

    restoration_metrics = {
        "reference_provenance": "unverified_neighbor_proxy_not_clean_reference",
        "defect_mse_original": round(mse_orig if "mse_orig" in locals() else 0.0, 2),
        "defect_mse_restored_mc": round(mse_mc if "mse_mc" in locals() else 0.0, 2),
        "defect_mse_restored_telea": round(mse_tel if "mse_tel" in locals() else 0.0, 2),
        "defect_psnr_original_db": round(psnr_orig, 2),
        "defect_psnr_restored_mc_db": round(psnr_mc, 2),
        "defect_psnr_restored_telea_db": round(psnr_tel, 2),
        "psnr_gain_mc_db": round(psnr_mc - psnr_orig, 2),
    }

    # Generate 5-panel native-resolution comparison
    # Panel: 1. Orig | 2. Ground Truth | 3. Predicted Mask (Cyan) | 4. Effective Repair Mask | 5. Restored (MC)
    crop_x1 = max(0, negative_control_roi["xmin"] - 40)
    crop_x2 = min(w, max(negative_control_roi["xmax"], clean_roi["xmax"], 150) + 40)
    crop_y1 = max(0, min(negative_control_roi["ymin"], 50))
    crop_y2 = min(h, max(negative_control_roi["ymax"], 250) + 30)

    # Ensure a good crop width and height
    crop_w = min(w, 400)
    crop_h = min(h, 300)
    # Center around target defect
    ys_gt, xs_gt = np.where(core_gt > 0)
    if len(xs_gt) > 0:
        cx, cy = int(np.mean(xs_gt)), int(np.mean(ys_gt))
        x1 = max(0, min(w - crop_w, cx - crop_w // 2))
        y1 = max(0, min(h - crop_h, cy - crop_h // 2))
        x2, y2 = x1 + crop_w, y1 + crop_h
    else:
        x1, y1, x2, y2 = 0, 0, crop_w, crop_h

    # 1. Orig
    p1 = frames[target_eval_frame][y1:y2, x1:x2].copy()

    # 2. GT Overlay: Green=Core, Yellow=Abrasion, Purple=Ambiguous
    p2 = p1.copy()
    c_crop = core_gt[y1:y2, x1:x2] > 0
    a_crop = abr_gt[y1:y2, x1:x2] > 0
    am_crop = ambig_gt[y1:y2, x1:x2] > 0
    p2[c_crop] = [0, 255, 0]      # Green
    p2[a_crop] = [0, 255, 255]    # Yellow
    p2[am_crop] = [255, 0, 255]   # Purple

    # 3. Pred Mask Overlay (Cyan)
    p3 = p1.copy()
    m_crop = temp_scratch_target[y1:y2, x1:x2] > 0
    p3[m_crop] = [255, 255, 0]    # Cyan in BGR

    # 4. Effective Repair Mask (White)
    p4 = cv2.cvtColor(fused_eff_conservative[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)

    # 5. Restored
    p5 = repaired_temp_mc[y1:y2, x1:x2].copy()

    # Add header bars
    def add_header(img, text, color=(255, 255, 255)):
        hdr = np.zeros((28, img.shape[1], 3), dtype=np.uint8)
        cv2.putText(hdr, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        return np.vstack([hdr, img])

    panel = np.hstack([
        add_header(p1, "1. Original Frame"),
        add_header(p2, "2. GT (Grn=Core,Yel=Abr)"),
        add_header(p3, "3. Temporal (Cyan)"),
        add_header(p4, "4. Conservative Repair"),
        add_header(p5, "5. Motion-Comp Restored")
    ])

    panel_path = ARTIFACTS_DIR / f"cross_film_{seq_name}_native_panel.png"
    cv2.imwrite(str(panel_path), panel)
    logger.info("Saved native panel to %s", panel_path)

    # Video Preview Generation (Short side-by-side: Original vs Conservative Restored)
    video_out_mp4 = ARTIFACTS_DIR / f"cross_film_{seq_name}_preview.mp4"
    video_out_gif = ARTIFACTS_DIR / f"cross_film_{seq_name}_preview.gif"
    
    vw = frames[0].shape[1]
    vh = frames[0].shape[0]
    out_fps = 12.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    v_writer = cv2.VideoWriter(str(video_out_mp4), fourcc, out_fps, (vw * 2, vh))

    gif_frames = []
    for f_idx in range(n_frames):
        orig_f = frames[f_idx]
        eff_m = fused_res.multi_channel_masks[f_idx].get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        rest_f = synthesize_motion_comp_repair(frames, f_idx, eff_m, analyzer)

        # Annotate
        labeled_orig = orig_f.copy()
        labeled_rest = rest_f.copy()
        cv2.putText(labeled_orig, f"{seq_name} - Original ({f_idx})", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(labeled_rest, f"Restored (Option 2 Conservative)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        pair = np.hstack([labeled_orig, labeled_rest])
        v_writer.write(pair)

        # Downsample for GIF
        if f_idx % 2 == 0 or n_frames <= 25:
            thumb = cv2.resize(pair, (pair.shape[1] // 2, pair.shape[0] // 2))
            thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
            gif_frames.append(thumb_rgb)

    v_writer.release()
    logger.info("Saved preview video to %s", video_out_mp4)

    # Save animated GIF using imageio or PIL
    try:
        from PIL import Image
        pil_images = [Image.fromarray(gf) for gf in gif_frames]
        if pil_images:
            pil_images[0].save(
                str(video_out_gif),
                save_all=True,
                append_images=pil_images[1:],
                duration=int(1000 / out_fps),
                loop=0
            )
            logger.info("Saved preview GIF to %s", video_out_gif)
    except Exception as exc:
        logger.warning("Could not generate GIF: %s", exc)

    return {
        "sequence_name": seq_name,
        "clip_file": clip_file,
        "split": "dev_diagnostic",
        "label_provenance": "quarantined_temporal_diff_pseudo_label",
        "target_defect": target_defect_name,
        "evaluated_frame_index": target_eval_frame,
        "resolution": f"{w}x{h}",
        "frame_count": n_frames,
        "latency_ms_per_frame": {
            "spatial_detector": round(t_spat, 2),
            "temporal_detector": round(t_temp, 2),
            "fused_option_2": round(t_fused, 2),
        },
        "fps": {
            "spatial_detector": round(1000.0 / max(0.1, t_spat), 1),
            "temporal_detector": round(1000.0 / max(0.1, t_temp), 1),
            "fused_option_2": round(1000.0 / max(0.1, t_fused), 1),
        },
        "metrics_by_mode": {
            "spatial_only": m_spat,
            "temporal_only": m_temp,
            "fused_conservative": m_fused_cons,
            "fused_high_recall": m_fused_hr,
        },
        "negative_control": neg_summary,
        "restoration_quality": restoration_metrics,
        "artifacts": {
            "native_panel": f"cross_film_{seq_name}_native_panel.png",
            "video_mp4": f"cross_film_{seq_name}_preview.mp4",
            "video_gif": f"cross_film_{seq_name}_preview.gif",
        }
    }


# Ground truth builders for each sequence

def build_gt_paleface_seq_a(frames: list[np.ndarray], target_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paleface Seq A Frame 8: Vertical scratch at x~71, y in [0, 224]."""
    f = cv2.cvtColor(frames[target_idx], cv2.COLOR_BGR2GRAY)
    h, w = f.shape
    core = np.zeros((h, w), dtype=np.uint8)
    abr = np.zeros((h, w), dtype=np.uint8)
    ambig = np.zeros((h, w), dtype=np.uint8)

    # Reference clean neighbors
    f_prev = cv2.cvtColor(frames[target_idx - 1], cv2.COLOR_BGR2GRAY)
    f_next = cv2.cvtColor(frames[target_idx + 1], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f, np.minimum(f_prev, f_next))

    # The prominent scratch runs along x=68..73, y=0..224
    scratch_roi = (slice(0, 225), slice(68, 74))
    roi_diff = diff[scratch_roi]
    
    core_mask = np.zeros((h, w), dtype=bool)
    abr_mask = np.zeros((h, w), dtype=bool)
    ambig_mask = np.zeros((h, w), dtype=bool)

    core_mask[scratch_roi] = roi_diff >= 35
    abr_mask[scratch_roi] = (roi_diff >= 18) & (roi_diff < 35)
    ambig_mask[scratch_roi] = (roi_diff >= 10) & (roi_diff < 18)

    core[core_mask] = 255
    abr[abr_mask] = 255
    ambig[ambig_mask] = 255
    return core, abr, ambig


def build_gt_paleface_seq_b(frames: list[np.ndarray], target_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paleface Seq B Frame 7: Diagonal scratch at x~552, y in [1, 253]."""
    f = cv2.cvtColor(frames[target_idx], cv2.COLOR_BGR2GRAY)
    h, w = f.shape
    core = np.zeros((h, w), dtype=np.uint8)
    abr = np.zeros((h, w), dtype=np.uint8)
    ambig = np.zeros((h, w), dtype=np.uint8)

    f_prev = cv2.cvtColor(frames[target_idx - 1], cv2.COLOR_BGR2GRAY)
    f_next = cv2.cvtColor(frames[target_idx + 1], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f, np.minimum(f_prev, f_next))

    scratch_roi = (slice(1, 255), slice(545, 560))
    roi_diff = diff[scratch_roi]

    core_mask = np.zeros((h, w), dtype=bool)
    abr_mask = np.zeros((h, w), dtype=bool)
    ambig_mask = np.zeros((h, w), dtype=bool)

    core_mask[scratch_roi] = roi_diff >= 22
    abr_mask[scratch_roi] = (roi_diff >= 14) & (roi_diff < 22)
    ambig_mask[scratch_roi] = (roi_diff >= 8) & (roi_diff < 14)

    core[core_mask] = 255
    abr[abr_mask] = 255
    ambig[ambig_mask] = 255
    return core, abr, ambig


def build_gt_college_seq_c(frames: list[np.ndarray], target_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """College Seq C Frame 8: Subtle tracking vertical scratch at x~111, y in [395, 448]."""
    f = cv2.cvtColor(frames[target_idx], cv2.COLOR_BGR2GRAY)
    h, w = f.shape
    core = np.zeros((h, w), dtype=np.uint8)
    abr = np.zeros((h, w), dtype=np.uint8)
    ambig = np.zeros((h, w), dtype=np.uint8)

    f_prev = cv2.cvtColor(frames[target_idx - 1], cv2.COLOR_BGR2GRAY)
    f_next = cv2.cvtColor(frames[target_idx + 1], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f, np.minimum(f_prev, f_next))

    scratch_roi = (slice(395, 450), slice(108, 115))
    roi_diff = diff[scratch_roi]

    core_mask = np.zeros((h, w), dtype=bool)
    abr_mask = np.zeros((h, w), dtype=bool)
    ambig_mask = np.zeros((h, w), dtype=bool)

    core_mask[scratch_roi] = roi_diff >= 12
    abr_mask[scratch_roi] = (roi_diff >= 8) & (roi_diff < 12)
    ambig_mask[scratch_roi] = (roi_diff >= 5) & (roi_diff < 8)

    core[core_mask] = 255
    abr[abr_mask] = 255
    ambig[ambig_mask] = 255
    return core, abr, ambig


def build_gt_neighbors_seq_d_vert(frames: list[np.ndarray], target_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbors Seq D Frame 2: Vertical scratch at x~608, y in [95, 165]."""
    f = cv2.cvtColor(frames[target_idx], cv2.COLOR_BGR2GRAY)
    h, w = f.shape
    core = np.zeros((h, w), dtype=np.uint8)
    abr = np.zeros((h, w), dtype=np.uint8)
    ambig = np.zeros((h, w), dtype=np.uint8)

    f_prev = cv2.cvtColor(frames[target_idx - 1], cv2.COLOR_BGR2GRAY)
    f_next = cv2.cvtColor(frames[target_idx + 1], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f, np.minimum(f_prev, f_next))

    scratch_roi = (slice(95, 165), slice(605, 613))
    roi_diff = diff[scratch_roi]

    core_mask = np.zeros((h, w), dtype=bool)
    abr_mask = np.zeros((h, w), dtype=bool)
    ambig_mask = np.zeros((h, w), dtype=bool)

    core_mask[scratch_roi] = roi_diff >= 20
    abr_mask[scratch_roi] = (roi_diff >= 12) & (roi_diff < 20)
    ambig_mask[scratch_roi] = (roi_diff >= 7) & (roi_diff < 12)

    core[core_mask] = 255
    abr[abr_mask] = 255
    ambig[ambig_mask] = 255
    return core, abr, ambig


def build_gt_neighbors_seq_d_horiz(frames: list[np.ndarray], target_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbors Seq D Frame 30: Horizontal fissure at y~450..455, x in [581, 683]."""
    f = cv2.cvtColor(frames[target_idx], cv2.COLOR_BGR2GRAY)
    h, w = f.shape
    core = np.zeros((h, w), dtype=np.uint8)
    abr = np.zeros((h, w), dtype=np.uint8)
    ambig = np.zeros((h, w), dtype=np.uint8)

    f_prev = cv2.cvtColor(frames[target_idx - 1], cv2.COLOR_BGR2GRAY)
    f_next = cv2.cvtColor(frames[target_idx + 1], cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(f, np.minimum(f_prev, f_next))

    scratch_roi = (slice(448, 456), slice(580, 685))
    roi_diff = diff[scratch_roi]

    core_mask = np.zeros((h, w), dtype=bool)
    abr_mask = np.zeros((h, w), dtype=bool)
    ambig_mask = np.zeros((h, w), dtype=bool)

    core_mask[scratch_roi] = roi_diff >= 16
    abr_mask[scratch_roi] = (roi_diff >= 10) & (roi_diff < 16)
    ambig_mask[scratch_roi] = (roi_diff >= 6) & (roi_diff < 10)

    core[core_mask] = 255
    abr[abr_mask] = 255
    ambig[ambig_mask] = 255
    return core, abr, ambig


def main():
    logger.info("Starting Cross-Film Generalization Benchmark...")

    experiments = [
        {
            "seq_name": "paleface_seq_a",
            "clip_file": "paleface_seq_a.mkv",
            "target_eval_frame": 8,
            "target_defect_name": "Vertical Fissure (x~71, h=175px)",
            "build_gt_fn": build_gt_paleface_seq_a,
            "negative_control_roi": {
                "ymin": 188, "ymax": 381, "xmin": 258, "xmax": 268,
                "description": "Moving actor jacket boundary against static office wall"
            },
            "clean_roi": {"ymin": 300, "ymax": 420, "xmin": 450, "xmax": 550},
        },
        {
            "seq_name": "paleface_seq_b",
            "clip_file": "paleface_seq_b.mkv",
            "target_eval_frame": 7,
            "target_defect_name": "Diagonal Abrasion & Branching (x~552, h=252px)",
            "build_gt_fn": build_gt_paleface_seq_b,
            "negative_control_roi": {
                "ymin": 80, "ymax": 180, "xmin": 200, "xmax": 280,
                "description": "Static interior wall panel and curtain fold"
            },
            "clean_roi": {"ymin": 30, "ymax": 120, "xmin": 50, "xmax": 150},
        },
        {
            "seq_name": "college_seq_c",
            "clip_file": "college_seq_c.mkv",
            "target_eval_frame": 8,
            "target_defect_name": "Tracking Fine Vertical Scratch (x~111, h=53px)",
            "build_gt_fn": build_gt_college_seq_c,
            "negative_control_roi": {
                "ymin": 150, "ymax": 280, "xmin": 280, "xmax": 340,
                "description": "Classroom door jamb & moving student silhouette"
            },
            "clean_roi": {"ymin": 50, "ymax": 150, "xmin": 400, "xmax": 500},
        },
        {
            "seq_name": "neighbors_seq_d_vert",
            "clip_file": "neighbors_seq_d.mkv",
            "target_eval_frame": 2,
            "target_defect_name": "Facade Vertical Scratch (x~608, h=70px)",
            "build_gt_fn": build_gt_neighbors_seq_d_vert,
            "negative_control_roi": {
                "ymin": 200, "ymax": 400, "xmin": 150, "xmax": 220,
                "description": "Street telephone pole and vertical building corner"
            },
            "clean_roi": {"ymin": 80, "ymax": 200, "xmin": 700, "xmax": 850},
        },
        {
            "seq_name": "neighbors_seq_d_horiz",
            "clip_file": "neighbors_seq_d.mkv",
            "target_eval_frame": 30,
            "target_defect_name": "Road Horizontal Fissure (y~450, w=102px)",
            "build_gt_fn": build_gt_neighbors_seq_d_horiz,
            "negative_control_roi": {
                "ymin": 200, "ymax": 400, "xmin": 150, "xmax": 220,
                "description": "Street telephone pole and vertical building corner"
            },
            "clean_roi": {"ymin": 80, "ymax": 200, "xmin": 700, "xmax": 850},
        },
    ]

    all_results = []
    for exp in experiments:
        res = evaluate_sequence(
            seq_name=exp["seq_name"],
            clip_file=exp["clip_file"],
            target_eval_frame=exp["target_eval_frame"],
            build_gt_fn=exp["build_gt_fn"],
            negative_control_roi=exp["negative_control_roi"],
            clean_roi=exp["clean_roi"],
            target_defect_name=exp["target_defect_name"],
        )
        all_results.append(res)

    # Save benchmark JSON
    out_json = DOCS_BENCHMARKS / "cross_film_generalization_benchmark.json"
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "frozen_configuration": {
                "detector": "temporal_scratch",
                "preset": "temporal_scratch_balanced",
                "parameters": BENCHMARK_PRESETS["temporal_scratch_balanced"].params,
            },
            "evaluation_results": all_results,
        }, fp, indent=2)

    logger.info("Saved benchmark JSON to %s", out_json)


if __name__ == "__main__":
    main()
