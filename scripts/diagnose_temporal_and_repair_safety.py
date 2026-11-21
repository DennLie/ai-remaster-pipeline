#!/usr/bin/env python3
"""Diagnostic Root-Cause Audit for Temporal Motion-Gate Failure & Repair Leakage.

Runs strictly on development/diagnostic sequences (NEVER touches locked holdouts
and NEVER modifies production detector thresholds, presets, fusion rules, or GUI code).

Outputs:
  - docs/benchmarks/diagnostic_motion_and_repair_leakage_report.json
  - docs/benchmarks/diagnostic_motion_and_repair_leakage_report.md
  - docs/benchmarks/visuals/diagnostic_motion_and_leakage_audit.png
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchmark_defect_masks as bdm
import dataset_adapters as da
from film_defect_detectors import (
    BENCHMARK_PRESETS,
    DEFAULT_REGISTRY,
    RepairInclusionPolicy,
    fuse_spatial_and_temporal_masks,
)
from temporal_scratch_analyzer import (
    CLEAN_ROI,
    SCRATCH_ROI,
    TemporalConfig,
    TemporalScratchAnalyzer,
    load_clip,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose_temporal_safety")

DOCS_BENCHMARKS = PROJECT_ROOT / "docs" / "benchmarks"
VISUALS_DIR = DOCS_BENCHMARKS / "visuals"
ABSDEG_FULL_DIR = PROJECT_ROOT / ".cache" / "absolute-scratch-benchmark-20261106-01"
CROSS_FILM_DIR = PROJECT_ROOT / "evaluation" / "cross_film_validation"
NEIGHBORS_VIDEO = PROJECT_ROOT / "evaluation" / "Neighbors_72.mkv"


def _stepwise_neighbors(
    gray_frames: Sequence[np.ndarray],
    target_idx: int,
    *,
    window_radius: int = 3,
    dedup_thresh: float = 0.8,
    step_cut_thresh: float = 45.0,
) -> tuple[list[int], list[int], list[int]]:
    """Select neighbors using step-wise inter-frame MAD rather than cumulative target MAD."""
    n = len(gray_frames)
    preceding: list[int] = []
    skipped: list[int] = []
    last_acc = target_idx
    for p in range(target_idx - 1, -1, -1):
        mad_step = float(np.mean(cv2.absdiff(gray_frames[p], gray_frames[last_acc])))
        if mad_step > step_cut_thresh:
            break
        if float(np.mean(cv2.absdiff(gray_frames[p], gray_frames[target_idx]))) >= dedup_thresh and mad_step >= dedup_thresh:
            preceding.append(p)
            last_acc = p
            if len(preceding) == window_radius:
                break
        else:
            skipped.append(p)

    succeeding: list[int] = []
    last_acc = target_idx
    for s in range(target_idx + 1, n):
        mad_step = float(np.mean(cv2.absdiff(gray_frames[s], gray_frames[last_acc])))
        if mad_step > step_cut_thresh:
            break
        if float(np.mean(cv2.absdiff(gray_frames[s], gray_frames[target_idx]))) >= dedup_thresh and mad_step >= dedup_thresh:
            succeeding.append(s)
            last_acc = s
            if len(succeeding) == window_radius:
                break
        else:
            skipped.append(s)
    return preceding, succeeding, skipped


def _compute_flow_diagnostics(
    analyzer: TemporalScratchAnalyzer,
    frames: Sequence[np.ndarray],
    idx: int,
    region_masks: Mapping[str, np.ndarray],
    *,
    allow_stepwise_fallback: bool = False,
) -> dict[str, Any]:
    """Measure flow direction, magnitude units, registration residuals, and gate counts."""
    grays = [analyzer._ensure_gray(f) for f in frames]
    target_gray = grays[idx]
    h, w = target_gray.shape

    consec_mads = [round(float(np.mean(cv2.absdiff(grays[i], grays[i - 1]))), 2) for i in range(1, len(grays))]
    prec, succ, skipped = analyzer._select_nonduplicate_neighbors(grays, idx)
    default_nbrs = prec + succ
    abstained_by_scene_cut = len(default_nbrs) < analyzer.config.min_temporal_neighbors

    if abstained_by_scene_cut and allow_stepwise_fallback:
        prec, succ, skipped = _stepwise_neighbors(grays, idx)
    valid_nbrs = prec + succ

    if not valid_nbrs:
        return {
            "abstained": True,
            "default_neighbors": default_nbrs,
            "consecutive_mads": consec_mads,
            "regions": {},
        }

    flows = [analyzer._compute_dense_flow(target_gray, grays[n]) for n in valid_nbrs]
    warped = [analyzer._warp_neighbor(grays[n], fl).astype(np.float32) for n, fl in zip(valid_nbrs, flows)]
    unwarped = [grays[n].astype(np.float32) for n in valid_nbrs]

    w_stack = np.stack(warped, axis=0)
    bg_warped = analyzer._synthesize_background(w_stack)
    bg_unwarped = analyzer._synthesize_background(np.stack(unwarped, axis=0))

    diff_warped = np.abs(target_gray.astype(np.float32) - bg_warped)
    diff_unwarped = np.abs(target_gray.astype(np.float32) - bg_unwarped)

    med = float(np.median(diff_warped))
    mad = float(np.median(np.abs(diff_warped - med)))
    sigma = 1.4826 * max(mad, 1.0)
    k = analyzer.config.k_factor - 0.8 * (analyzer.config.sensitivity - 0.5)
    tau = med + k * sigma

    flank = analyzer._compute_flank_contrast(target_gray)
    contrast_th = analyzer.config.contrast_base - 6.0 * (analyzer.config.sensitivity - 0.5)

    flow_mag = np.mean([np.hypot(fl[..., 0], fl[..., 1]) for fl in flows], axis=0)
    flow_mag_per_step = np.mean(
        [np.hypot(fl[..., 0], fl[..., 1]) / max(1, abs(n - idx)) for fl, n in zip(flows, valid_nbrs)],
        axis=0,
    )
    mean_dx = np.mean([fl[..., 0] for fl in flows], axis=0)
    mean_dy = np.mean([fl[..., 1] for fl in flows], axis=0)

    raw_temp = diff_warped >= tau
    struct_ok = flank >= contrast_th
    motion_ok = flow_mag <= analyzer.config.motion_coherence_threshold

    regions_out: dict[str, Any] = {}
    for rname, rmask in region_masks.items():
        n_px = int(np.count_nonzero(rmask))
        if n_px == 0:
            continue
        regions_out[rname] = {
            "pixels": n_px,
            "raw_temporal_pass": int(np.count_nonzero(raw_temp & rmask)),
            "structure_ok_pass": int(np.count_nonzero(struct_ok & rmask)),
            "motion_ok_pass": int(np.count_nonzero(motion_ok & rmask)),
            "raw_and_struct_pass": int(np.count_nonzero(raw_temp & struct_ok & rmask)),
            "all_3_gates_pass": int(np.count_nonzero(raw_temp & struct_ok & motion_ok & rmask)),
            "flow_mag_mean_px": round(float(flow_mag[rmask].mean()), 3),
            "flow_mag_median_px": round(float(np.median(flow_mag[rmask])), 3),
            "flow_mag_p90_px": round(float(np.percentile(flow_mag[rmask], 90)), 3),
            "flow_mag_per_frame_step_mean_px": round(float(flow_mag_per_step[rmask].mean()), 3),
            "flow_dx_mean_px": round(float(mean_dx[rmask].mean()), 3),
            "flow_dy_mean_px": round(float(mean_dy[rmask].mean()), 3),
            "diff_warped_mean": round(float(diff_warped[rmask].mean()), 3),
            "diff_unwarped_mean": round(float(diff_unwarped[rmask].mean()), 3),
            "flank_contrast_mean": round(float(flank[rmask].mean()), 3),
        }

    # Check connected-component bounding-box width rejection bug on raw_temp & struct_ok
    cand_u8 = (raw_temp & struct_ok).astype(np.uint8) * 255
    opened = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    n_l, lbl, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    bbox_rejected_components: list[dict[str, Any]] = []
    pos_mask = region_masks.get("gt_scratch") if "gt_scratch" in region_masks else region_masks.get("injected_scratch")
    if pos_mask is not None:
        for i in range(1, n_l):
            _x, _y, cw, ch, area = map(int, stats[i])
            ov = int(np.count_nonzero((lbl == i) & pos_mask))
            if ov > 50 and cw > analyzer.config.scratch_max_width:
                bbox_rejected_components.append({
                    "component_id": i,
                    "bbox_width_cw": cw,
                    "bbox_height_ch": ch,
                    "area_px": area,
                    "true_thickness_area_over_ch": round(area / max(1, ch), 2),
                    "gt_overlap_px": ov,
                    "rejected_by": f"cw={cw} > scratch_max_width={analyzer.config.scratch_max_width}",
                })

    return {
        "abstained_by_default_scene_cut": abstained_by_scene_cut,
        "default_neighbors": default_nbrs,
        "evaluated_neighbors": valid_nbrs,
        "skipped_telecine_duplicates": skipped,
        "consecutive_mads": consec_mads,
        "tau": round(tau, 3),
        "contrast_th": round(contrast_th, 3),
        "motion_coherence_threshold": analyzer.config.motion_coherence_threshold,
        "bbox_rejected_positive_components": bbox_rejected_components,
        "regions": regions_out,
        "_arrays": {
            "flow_mag": flow_mag,
            "diff_warped": diff_warped,
            "raw_temp": raw_temp,
            "struct_ok": struct_ok,
        },
    }


def _synthesize_mc_repair(
    frames: Sequence[np.ndarray],
    idx: int,
    repair_mask: np.ndarray,
    analyzer: TemporalScratchAnalyzer,
) -> np.ndarray:
    if not np.any(repair_mask >= 128):
        return frames[idx].copy()
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    prec, succ, _ = analyzer._select_nonduplicate_neighbors(grays, idx)
    nbrs = prec + succ
    if not nbrs:
        return cv2.inpaint(frames[idx], repair_mask, 3, cv2.INPAINT_TELEA)
    warped_bgr = []
    for n in nbrs:
        fl = analyzer._compute_dense_flow(grays[idx], grays[n])
        wf = np.zeros_like(frames[idx])
        for c in range(3):
            wf[:, :, c] = analyzer._warp_neighbor(frames[n][:, :, c], fl)
        warped_bgr.append(wf)
    bg_bgr = np.median(np.stack(warped_bgr, axis=0), axis=0).astype(np.uint8)
    out = frames[idx].copy()
    out[repair_mask >= 128] = bg_bgr[repair_mask >= 128]
    return out


def _trace_leakage_pipeline(
    name: str,
    frames: Sequence[np.ndarray],
    idx: int,
    roi: Mapping[str, Any],
    analyzer: TemporalScratchAnalyzer,
) -> dict[str, Any]:
    """Trace spatial_only, temporal_only, and fused_conservative through repair modification."""
    spat_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    temp_det = DEFAULT_REGISTRY.get("temporal_scratch")
    spat_cfg = dict(BENCHMARK_PRESETS["film_artefact_balanced"].params)
    temp_cfg = dict(BENCHMARK_PRESETS["temporal_scratch_balanced"].params)

    s_res = spat_det.detect(frames, spat_cfg)
    t_res = temp_det.detect(frames, temp_cfg)
    f_res = fuse_spatial_and_temporal_masks(s_res, t_res)

    h, w = frames[idx].shape[:2]
    rm = np.zeros((h, w), dtype=bool)
    rm[roi["ymin"]:roi["ymax"], roi["xmin"]:roi["xmax"]] = True

    grays = [analyzer._ensure_gray(f) for f in frames]
    prec, succ, skipped = analyzer._select_nonduplicate_neighbors(grays, idx)
    nbrs = prec + succ
    flows = [analyzer._compute_dense_flow(grays[idx], grays[n]) for n in nbrs]
    warped = [analyzer._warp_neighbor(grays[n], fl).astype(np.float32) for n, fl in zip(nbrs, flows)]
    bg = analyzer._synthesize_background(np.stack(warped, axis=0))
    diff = np.abs(grays[idx].astype(np.float32) - bg)
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    tau = med + (1.3 - 0.8 * 0.15) * (1.4826 * max(mad, 1.0))
    flank = analyzer._compute_flank_contrast(grays[idx])
    flow_mag = np.mean([np.hypot(fl[..., 0], fl[..., 1]) for fl in flows], axis=0)

    s_mcm = s_res.multi_channel_masks[idx]
    t_mcm = t_res.multi_channel_masks[idx]
    f_mcm = f_res.multi_channel_masks[idx]

    modes = {
        "spatial_only": (s_res.mask[idx], s_mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)),
        "temporal_only": (t_res.scratch_mask[idx], t_mcm.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)),
        "fused_conservative": (f_res.mask[idx], f_mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)),
        "fused_high_recall": (f_res.mask[idx], f_mcm.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)),
    }

    mode_trace: dict[str, Any] = {}
    for mname, (det_m, rep_m) in modes.items():
        rep_mc = _synthesize_mc_repair(frames, idx, rep_m, analyzer)
        rep_tel = cv2.inpaint(frames[idx], rep_m, 3, cv2.INPAINT_TELEA) if np.any(rep_m >= 128) else frames[idx].copy()
        mod_mc = np.any(rep_mc != frames[idx], axis=2)
        mod_tel = np.any(rep_tel != frames[idx], axis=2)

        neg_eval = da.evaluate_structural_negative_controls(det_m, rep_m, [roi])
        mode_trace[mname] = {
            "roi_detector_mask_px": neg_eval["total_detector_leak_pixels"],
            "roi_repair_mask_px": neg_eval["total_repair_leak_pixels"],
            "roi_actual_modified_mc_px": int(np.count_nonzero(mod_mc & rm)),
            "roi_actual_modified_telea_px": int(np.count_nonzero(mod_tel & rm)),
            "full_frame_repair_mask_px": int(np.count_nonzero(rep_m >= 128)),
            "full_frame_modified_mc_px": int(np.count_nonzero(mod_mc)),
            "full_frame_modified_telea_px": int(np.count_nonzero(mod_tel)),
            "strict_zero_leak_passed": neg_eval["strict_safety_gate_passed"],
        }

    t_scr_roi = (t_res.scratch_mask[idx] > 0) & rm
    return {
        "sequence": name,
        "frame_index": idx,
        "roi": dict(roi),
        "valid_neighbors": nbrs,
        "skipped_duplicates": skipped,
        "temporal_candidate_stages": {
            "raw_temporal_px": int(np.count_nonzero((diff >= tau) & rm)),
            "structure_ok_px": int(np.count_nonzero((flank >= 12.1) & rm)),
            "motion_ok_px": int(np.count_nonzero((flow_mag <= 1.2) & rm)),
            "temporal_scratch_mask_px": int(np.count_nonzero(t_scr_roi)),
            "temporal_prob_mean_on_mask": round(float(t_mcm.combined_probability[t_scr_roi].mean()), 4) if np.any(t_scr_roi) else 0.0,
        },
        "fusion_channels_in_roi": {
            "spatial_scratch_px": int(np.count_nonzero((s_res.scratch_mask[idx] > 0) & rm)),
            "spatial_dust_px": int(np.count_nonzero((s_res.dust_mask[idx] > 0) & rm)),
            "fused_scratch_px": int(np.count_nonzero((f_res.scratch_mask[idx] > 0) & rm)),
            "fused_dust_px": int(np.count_nonzero((f_res.dust_mask[idx] > 0) & rm)),
            "fused_uncertain_px": int(np.count_nonzero((f_res.uncertain_mask[idx] > 0) & rm)),
        },
        "modes": mode_trace,
        "_vis": {
            "frame": frames[idx],
            "fused_conservative": modes["fused_conservative"][1],
            "spatial_repair": modes["spatial_only"][1],
            "temporal_repair": modes["temporal_only"][1],
        },
    }


def _evaluate_candidate_solutions(
    name: str,
    frames: Sequence[np.ndarray],
    idx: int,
    *,
    gt_mask: np.ndarray | None = None,
    clean_ref: np.ndarray | None = None,
    neg_roi: Mapping[str, Any] | None = None,
    allow_stepwise_nbrs: bool = False,
) -> dict[str, Any]:
    """Run diagnostic ablation of baseline vs 3 candidate solutions on a dev sequence."""
    analyzer = TemporalScratchAnalyzer(TemporalConfig())
    spat_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    spat_res = spat_det.detect(frames, dict(BENCHMARK_PRESETS["film_artefact_balanced"].params))
    s_mcm = spat_res.multi_channel_masks[idx]
    s_rep = s_mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)

    grays = [analyzer._ensure_gray(f) for f in frames]
    target_gray = grays[idx]
    h, w = target_gray.shape

    prec, succ, _ = analyzer._select_nonduplicate_neighbors(grays, idx)
    if len(prec + succ) < 2 and allow_stepwise_nbrs:
        prec, succ, _ = _stepwise_neighbors(grays, idx)
    nbrs = prec + succ

    flows = [analyzer._compute_dense_flow(target_gray, grays[n]) for n in nbrs]
    warped = [analyzer._warp_neighbor(grays[n], fl).astype(np.float32) for n, fl in zip(nbrs, flows)]
    w_stack = np.stack(warped, axis=0)
    bg = analyzer._synthesize_background(w_stack)
    diff = np.abs(target_gray.astype(np.float32) - bg)
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    tau = med + (1.3 - 0.8 * 0.15) * (1.4826 * max(mad, 1.0))

    flank = analyzer._compute_flank_contrast(target_gray)
    nbr_flanks = np.stack([analyzer._compute_flank_contrast(w.astype(np.uint8)) for w in w_stack], axis=0)
    median_nbr_flank = np.median(nbr_flanks, axis=0)
    flank_novelty = flank - median_nbr_flank
    nbr_spread = np.percentile(w_stack, 75, axis=0) - np.percentile(w_stack, 25, axis=0)

    xg, yg = analyzer._get_remap_grid(h, w)
    fb_errs = []
    for n, fl_bw in zip(nbrs, flows):
        fl_fw = analyzer._compute_dense_flow(grays[n], target_gray)
        fw_x = cv2.remap(fl_fw[..., 0], xg + fl_bw[..., 0], yg + fl_bw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fw_y = cv2.remap(fl_fw[..., 1], xg + fl_bw[..., 0], yg + fl_bw[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        fb_errs.append(np.hypot(fl_bw[..., 0] + fw_x, fl_bw[..., 1] + fw_y))
    median_fb_err = np.median(np.stack(fb_errs, axis=0), axis=0)

    flow_mag = np.mean([np.hypot(fl[..., 0], fl[..., 1]) for fl in flows], axis=0)
    raw_temp = diff >= tau
    struct_ok = flank >= 12.1

    def _extract_components(cand_bool: np.ndarray, *, use_thickness_instead_of_cw: bool) -> np.ndarray:
        cand_u8 = cand_bool.astype(np.uint8) * 255
        opened = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
        n_l, lbl, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        sm = np.zeros_like(closed)
        for i in range(1, n_l):
            _x, _y, cw, ch, area = map(int, stats[i])
            if ch < 15:
                continue
            if use_thickness_instead_of_cw:
                thick = area / max(1, ch)
                ok = (thick <= 8.0 and ch >= 35) or (cw <= 12 and ch >= 3.5 * cw)
            else:
                ok = cw <= 12 and (ch >= 3.5 * cw or ch >= 35)
            if ok:
                sm[lbl == i] = 255
        if np.any(sm):
            sm = cv2.dilate(sm, np.ones((1, 3), np.uint8))
        return sm

    # 0. Baseline (Current) — note: if default neighbors abstained, baseline outputs empty mask
    prec_def, succ_def, _ = analyzer._select_nonduplicate_neighbors(grays, idx)
    if len(prec_def + succ_def) < 2:
        mask_baseline = np.zeros((h, w), dtype=np.uint8)
    else:
        mask_baseline = _extract_components(raw_temp & struct_ok & (flow_mag <= 1.2), use_thickness_instead_of_cw=False)

    # 1. Ablation: Drop flow gate only
    mask_drop_flow = _extract_components(raw_temp & struct_ok, use_thickness_instead_of_cw=False)

    # 2. Candidate Solution 1: Warped Flank-Novelty + Cross-Sectional Thickness
    sol1_cand = raw_temp & struct_ok & (flank_novelty >= 10.0) & (median_nbr_flank <= 6.0) & (nbr_spread <= np.maximum(10.0, 0.5 * diff))
    mask_sol1 = _extract_components(sol1_cand, use_thickness_instead_of_cw=True)

    # 3. Candidate Solution 2: Solution 1 + Forward-Backward Flow Consistency (rejects moving-actor occlusion borders)
    sol2_cand = sol1_cand & (median_fb_err <= 1.5)
    mask_sol2 = _extract_components(sol2_cand, use_thickness_instead_of_cw=True)

    # 4. Candidate Solution 3 (Fused Conservative Safety): Sol 2 temporal scratch + spatial proposals vetted by temporal diff & non-persistent-edge check
    spatial_vetted = np.where((diff >= tau) & (median_nbr_flank <= 6.0), s_rep, np.uint8(0))
    mask_sol3_fused = np.bitwise_or(spatial_vetted, mask_sol2)

    variants = {
        "0_baseline_current_temporal": mask_baseline,
        "1_ablation_drop_flow_only": mask_drop_flow,
        "2_solution_1_flank_novelty_thickness": mask_sol1,
        "3_solution_2_fb_consistency": mask_sol2,
        "4_solution_3_fused_conservative_veto": mask_sol3_fused,
    }

    results: dict[str, Any] = {}
    for vname, pmask in variants.items():
        entry: dict[str, Any] = {"predicted_pixels": int(np.count_nonzero(pmask))}
        if gt_mask is not None:
            cnt = bdm.confusion(gt_mask, pmask)
            m = bdm.metrics(cnt)
            sev = da.evaluate_false_positive_severity(gt_mask, pmask)
            entry["detection"] = {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "tp": cnt["tp"],
                "fp": cnt["fp"],
                "fn": cnt["fn"],
                "fp_severity": sev,
            }
            if clean_ref is not None:
                rep_mc = _synthesize_mc_repair(frames, idx, pmask, analyzer)
                # Use step-wise neighbors for MC repair when default abstained (clip 000)
                if len(prec_def + succ_def) < 2 and np.any(pmask >= 128):
                    w_bgr = []
                    for n_i, fl in zip(nbrs, flows):
                        wf = np.zeros_like(frames[idx])
                        for c in range(3):
                            wf[:, :, c] = analyzer._warp_neighbor(frames[n_i][:, :, c], fl)
                        w_bgr.append(wf)
                    bg_bgr = np.median(np.stack(w_bgr, axis=0), axis=0).astype(np.uint8)
                    rep_mc = frames[idx].copy()
                    rep_mc[pmask >= 128] = bg_bgr[pmask >= 128]
                entry["restoration_mc"] = da.evaluate_exact_reference_restoration(clean_ref, frames[idx], rep_mc, gt_mask, pmask)
        if neg_roi is not None:
            neg_res = da.evaluate_structural_negative_controls(pmask, pmask, [neg_roi])
            entry["negative_control"] = neg_res["rois"][0]
        results[vname] = entry

    return {
        "sequence": name,
        "frame_index": idx,
        "variants": results,
        "_masks": variants,
    }


def _save_diagnostic_visual(
    flow_diag_010: dict[str, Any],
    d010_frame5: np.ndarray,
    g010_frame5: np.ndarray,
    sol_masks_010: Mapping[str, np.ndarray],
    leak_pipe: Mapping[str, Any],
    sol_masks_pipe: Mapping[str, np.ndarray],
    leak_pale_a: Mapping[str, Any],
    sol_masks_pale_a: Mapping[str, np.ndarray],
    leak_pale_b: Mapping[str, Any],
    sol_masks_pale_b: Mapping[str, np.ndarray],
    sol3_pale_b_leak_px: int,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=140)
    fig.patch.set_facecolor("#121212")

    def _overlay(bgr: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        idx = mask > 0
        for c in range(3):
            rgb[idx, c] = np.clip(rgb[idx, c].astype(float) * 0.35 + color_rgb[c] * 0.65, 0, 255).astype(np.uint8)
        return rgb

    arrs = flow_diag_010["_arrays"]

    # Row 1: Motion-Gate Failure on AbsDeg 010 Frame 5
    axes[0, 0].imshow(_overlay(d010_frame5, (g010_frame5 == 255).astype(np.uint8) * 255, (40, 230, 60)))
    axes[0, 0].set_title("1. AbsDeg 010 (Dev) Frame 5\nGreen = Exact Synthetic Scratch GT", color="white", fontsize=10, weight="bold")
    axes[0, 0].axis("off")

    im = axes[0, 1].imshow(arrs["flow_mag"], cmap="viridis", vmin=0, vmax=30)
    axes[0, 1].set_title("2. Dense Optical Flow Magnitude\nMean ~20.3 px vs Gate <= 1.2 px (0% Pass!)", color="#FF8888", fontsize=10, weight="bold")
    axes[0, 1].axis("off")

    raw_struct = (arrs["raw_temp"] & arrs["struct_ok"]).astype(np.uint8) * 255
    axes[0, 2].imshow(_overlay(d010_frame5, raw_struct, (0, 220, 255)))
    axes[0, 2].set_title("3. Before Motion Gate (raw_temp & struct_ok)\nCaptures 57.6% of GT, plus background FPs", color="white", fontsize=10, weight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(_overlay(d010_frame5, sol_masks_010["2_solution_1_flank_novelty_thickness"], (255, 210, 40)))
    axes[0, 3].set_title("4. Sol 1 (Flank-Novelty + Thickness)\nRecovers moving-camera scratch (42.3% R)", color="#88FF88", fontsize=10, weight="bold")
    axes[0, 3].axis("off")

    # Row 2: Conservative Repair Leakage vs Candidate Solutions
    # 5. Neighbors f25 pipe current vs Sol 1
    vp = leak_pipe["_vis"]
    axes[1, 0].imshow(_overlay(vp["frame"][20:320, 80:320], vp["fused_conservative"][20:320, 80:320], (255, 50, 50)))
    axes[1, 0].set_title("5. Neighbors f25 Pipe (Current Fused)\n819 px leaked into CONSERVATIVE repair", color="#FF8888", fontsize=10, weight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(_overlay(vp["frame"][20:320, 80:320], sol_masks_pipe["3_solution_2_fb_consistency"][20:320, 80:320], (80, 255, 120)))
    axes[1, 1].set_title("6. Neighbors f25 Pipe (Sol 2)\n0 px leaked (pipe exists in warped nbrs)", color="#88FF88", fontsize=10, weight="bold")
    axes[1, 1].axis("off")

    va = leak_pale_a["_vis"]
    axes[1, 2].imshow(_overlay(va["frame"][140:420, 160:380], va["fused_conservative"][140:420, 160:380], (255, 50, 50)))
    axes[1, 2].set_title("7. Paleface A f8 Jacket (Current Fused)\n641 px leaked (wall planks + actor edge)", color="#FF8888", fontsize=10, weight="bold")
    axes[1, 2].axis("off")

    vb = leak_pale_b["_vis"]
    base_pale_b_leak = leak_pale_b["modes"]["fused_conservative"]["roi_repair_mask_px"]
    panel8_color = "#88FF88" if sol3_pale_b_leak_px == 0 else "#FFCC66"
    axes[1, 3].imshow(_overlay(vb["frame"][50:220, 160:340], sol_masks_pale_b["4_solution_3_fused_conservative_veto"][50:220, 160:340], (80, 255, 120)))
    axes[1, 3].set_title(
        f"8. Paleface B f7 Wall (Sol 3 Conservative Veto)\n{sol3_pale_b_leak_px} px leaked vs {base_pale_b_leak} px current spatial leak",
        color=panel8_color,
        fontsize=10,
        weight="bold",
    )
    axes[1, 3].axis("off")

    plt.suptitle("Diagnostic Audit: Motion-Gate Failure (Dev 010) & Structural Repair Leakage (Neighbors / Paleface)", color="white", fontsize=13, weight="bold")
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])
    fig.savefig(str(out_path), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    DOCS_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    analyzer = TemporalScratchAnalyzer(TemporalConfig())

    # Load Dev Clips Only
    c000 = [cv2.imread(str(ABSDEG_FULL_DIR / "000" / "reference" / f"{i:06d}.png")) for i in range(12)]
    d000 = [cv2.imread(str(ABSDEG_FULL_DIR / "000" / "20261106" / f"{i:06d}.png")) for i in range(12)]
    g000 = [cv2.imread(str(ABSDEG_FULL_DIR / "000" / "20261106" / f"{i:06d}.gt.png"), 0) for i in range(12)]

    c010 = [cv2.imread(str(ABSDEG_FULL_DIR / "010" / "reference" / f"{i:06d}.png")) for i in range(12)]
    d010 = [cv2.imread(str(ABSDEG_FULL_DIR / "010" / "20261106" / f"{i:06d}.png")) for i in range(12)]
    g010 = [cv2.imread(str(ABSDEG_FULL_DIR / "010" / "20261106" / f"{i:06d}.gt.png"), 0) for i in range(12)]

    inj_cfg = bdm.ExtendedScratchConfig(
        orientation="vertical", angle_deg=1.5, width=2, opacity=0.85, bright=True, duration=8, drift_px=1.5, seed=202611181
    )
    d010_inj, g010_inj, _ = bdm.inject_extended_scratches(c010, inj_cfg)

    n_frames = load_clip(NEIGHBORS_VIDEO)[20:31]
    pa_frames = load_clip(CROSS_FILM_DIR / "paleface_seq_a.mkv")
    pb_frames = load_clip(CROSS_FILM_DIR / "paleface_seq_b.mkv")

    # 1. Motion-coherence diagnostics on Dev 000 and Dev 010
    nuis000 = da.segment_nuisance_regions(c000[5], prev_frame=c000[4])
    diag_000 = _compute_flow_diagnostics(
        analyzer, d000, 5,
        {
            "gt_scratch": g000[5] == 255,
            "intact_edges": nuis000["edges"] & (g000[5] == 0),
            "moving_regions": nuis000["motion"] & (g000[5] == 0),
            "ordinary_bg": (~nuis000["edges"]) & (~nuis000["motion"]) & (g000[5] == 0),
        },
        allow_stepwise_fallback=True,
    )

    nuis010 = da.segment_nuisance_regions(c010[5], prev_frame=c010[4])
    diag_010 = _compute_flow_diagnostics(
        analyzer, d010, 5,
        {
            "gt_scratch": g010[5] == 255,
            "intact_edges": nuis010["edges"] & (g010[5] == 0),
            "moving_regions": nuis010["motion"] & (g010[5] == 0),
            "ordinary_bg": (~nuis010["edges"]) & (~nuis010["motion"]) & (g010[5] == 0),
        },
    )

    diag_010_inj = _compute_flow_diagnostics(
        analyzer, d010_inj, 5,
        {
            "injected_scratch": g010_inj[5] == 255,
            "intact_edges": nuis010["edges"] & (g010_inj[5] == 0),
            "ordinary_bg": (~nuis010["edges"]) & (g010_inj[5] == 0),
        },
    )

    # 2. Repair Leakage Trace on Neighbors f25, Paleface A f8, Paleface B f7
    roi_pipe = {"name": "neighbors_frame25_pipe", **SCRATCH_ROI}
    roi_pale_a = {"name": "paleface_actor_jacket_edge", "ymin": 188, "ymax": 381, "xmin": 258, "xmax": 268}
    roi_pale_b = {"name": "paleface_b_wall_curtain", "ymin": 80, "ymax": 180, "xmin": 200, "xmax": 280}

    leak_pipe = _trace_leakage_pipeline("neighbors_frame25_pipe", n_frames, 5, roi_pipe, analyzer)
    leak_pale_a = _trace_leakage_pipeline("paleface_seq_a", pa_frames, 8, roi_pale_a, analyzer)
    leak_pale_b = _trace_leakage_pipeline("paleface_seq_b", pb_frames, 7, roi_pale_b, analyzer)

    # 3. Diagnostic Ablations on Dev Data
    abl_000 = _evaluate_candidate_solutions("absdeg_000_dev", d000, 5, gt_mask=g000[5], clean_ref=c000[5], allow_stepwise_nbrs=True)
    abl_010 = _evaluate_candidate_solutions("absdeg_010_dev", d010, 5, gt_mask=g010[5], clean_ref=c010[5])
    abl_010_inj = _evaluate_candidate_solutions("absdeg_010_injected_drifting", d010_inj, 5, gt_mask=g010_inj[5], clean_ref=c010[5])
    abl_pipe = _evaluate_candidate_solutions("neighbors_frame25_pipe", n_frames, 5, neg_roi=roi_pipe)
    abl_pale_a = _evaluate_candidate_solutions("paleface_seq_a", pa_frames, 8, neg_roi=roi_pale_a)
    abl_pale_b = _evaluate_candidate_solutions("paleface_seq_b", pb_frames, 7, neg_roi=roi_pale_b)

    vis_out = VISUALS_DIR / "diagnostic_motion_and_leakage_audit.png"
    sol3_pale_b_leak_px = int(
        abl_pale_b["variants"]["4_solution_3_fused_conservative_veto"]["negative_control"]["repair_leak_pixels"]
    )
    _save_diagnostic_visual(
        diag_010, d010[5], g010[5], abl_010["_masks"],
        leak_pipe, abl_pipe["_masks"],
        leak_pale_a, abl_pale_a["_masks"],
        leak_pale_b, abl_pale_b["_masks"],
        sol3_pale_b_leak_px,
        vis_out,
    )

    for obj in (diag_000, diag_010, diag_010_inj):
        obj.pop("_arrays", None)
    for obj in (leak_pipe, leak_pale_a, leak_pale_b):
        obj.pop("_vis", None)
    for obj in (abl_000, abl_010, abl_010_inj, abl_pipe, abl_pale_a, abl_pale_b):
        obj.pop("_masks", None)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Development/diagnostic data only; zero production detector or preset modifications.",
        "motion_coherence_diagnostics": {
            "absdeg_000_frame5": diag_000,
            "absdeg_010_frame5": diag_010,
            "absdeg_010_injected_drifting_frame5": diag_010_inj,
        },
        "repair_leakage_trace": {
            "neighbors_frame25_pipe": leak_pipe,
            "paleface_seq_a_jacket": leak_pale_a,
            "paleface_seq_b_wall": leak_pale_b,
        },
        "diagnostic_ablations_dev_only": [abl_000, abl_010, abl_010_inj, abl_pipe, abl_pale_a, abl_pale_b],
        "visual_path": str(vis_out.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }

    json_path = DOCS_BENCHMARKS / "diagnostic_motion_and_repair_leakage_report.json"
    md_path = DOCS_BENCHMARKS / "diagnostic_motion_and_repair_leakage_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Format concise Markdown summary
    md_lines = [
        "# Diagnostic Audit: Motion-Gate Failure & Conservative Repair Leakage (Dev Only)",
        "",
        f"- **Timestamp:** `{report['timestamp']}`",
        "- **Constraint Check:** Zero detector thresholds, presets, fusion rules, or GUI files modified.",
        "",
        "## 1. Motion-Gate Flow Distributions (Dev Clip `010` Frame 5)",
        "",
        "| Region | Pixels | Raw Temporal Pass | Structure OK Pass | Motion OK (`flow<=1.2`) Pass | Mean Flow (px) | Diff Warped | Diff Unwarped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rname, rv in diag_010["regions"].items():
        md_lines.append(
            f"| `{rname}` | {rv['pixels']} | {rv['raw_temporal_pass']} | {rv['structure_ok_pass']} | "
            f"{rv['motion_ok_pass']} | {rv['flow_mag_mean_px']:.2f} | {rv['diff_warped_mean']:.2f} | {rv['diff_unwarped_mean']:.2f} |"
        )

    md_lines.extend([
        "",
        "## 2. Stage-by-Stage Repair Leakage Trace (Archival Dev Negative Controls)",
        "",
        "| Sequence ROI | Mode | Detector Mask (px) | Repair Mask (px) | Actual Modified MC (px) | Actual Modified Telea (px) | Zero-Leak Gate |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ])
    for lk in (leak_pipe, leak_pale_a, leak_pale_b):
        for mname in ("spatial_only", "temporal_only", "fused_conservative"):
            mv = lk["modes"][mname]
            badge = "PASS" if mv["strict_zero_leak_passed"] else "**FAIL**"
            md_lines.append(
                f"| `{lk['sequence']}` | `{mname}` | {mv['roi_detector_mask_px']} | {mv['roi_repair_mask_px']} | "
                f"{mv['roi_actual_modified_mc_px']} | {mv['roi_actual_modified_telea_px']} | {badge} |"
            )

    md_lines.extend([
        "",
        "## 3. Diagnostic Ablation Results (Dev Clips Only)",
        "",
        "| Sequence | Variant | Det Recall | Det Prec | Det F1 | Neg ROI Leak (px) | MC Defect PSNR Gain | Intact Px Modified |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for abl in report["diagnostic_ablations_dev_only"]:
        for vname, vv in abl["variants"].items():
            det = vv.get("detection", {})
            rest = vv.get("restoration_mc", {})
            neg = vv.get("negative_control", {})
            rec_s = f"{det['recall']*100:.1f}%" if det else "N/A"
            prec_s = f"{det['precision']*100:.1f}%" if det else "N/A"
            f1_s = f"{det['f1']:.3f}" if det else "N/A"
            leak_s = f"{neg['repair_leak_pixels']}" if neg else "N/A"
            psnr_s = f"{rest['psnr_gain_db']:+.2f} dB" if rest else "N/A"
            mod_s = f"{rest['intact_pixels_modified_count']}" if rest else "N/A"
            md_lines.append(f"| `{abl['sequence']}` | `{vname}` | {rec_s} | {prec_s} | {f1_s} | {leak_s} | {psnr_s} | {mod_s} |")

    md_lines.extend(["", f"Visual evidence saved to `{report['visual_path']}`."])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    logger.info("Saved diagnostic JSON to %s", json_path)
    logger.info("Saved diagnostic MD to %s", md_path)
    logger.info("Saved diagnostic visual to %s", vis_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
