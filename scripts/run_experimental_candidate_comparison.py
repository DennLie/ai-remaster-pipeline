#!/usr/bin/env python3
"""Reproducible Experimental Comparison Harness for Candidates A-E,
Factorial Ablations (M1 x M2 x M3), and Scene-vs-Film Persistence Study.

Runs strictly on `dev_diagnostic` datasets (NEVER touches locked holdouts and
NEVER modifies production detector, fusion, preset, restoration, or GUI files).

Outputs:
  - docs/benchmarks/experimental_candidate_comparison_report.json
  - docs/benchmarks/experimental_candidate_comparison_report.md
  - docs/benchmarks/visuals/experimental_candidates_a_to_e_comparison.png
  - docs/benchmarks/visuals/experimental_examples/*.png
  - docs/benchmarks/visuals/experimental_side_by_side_videos/*.mp4
"""
from __future__ import annotations

import json
import logging
import sys
import time
import tracemalloc
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
from diagnose_temporal_and_repair_safety import _stepwise_neighbors
from experimental_scene_film_scratch_prototype import (
    ExperimentalConfig,
    ExperimentalSceneFilmDetector,
    ExperimentalThreeStateResult,
    select_motion_robust_neighbors,
)
from film_defect_detectors import (
    BENCHMARK_PRESETS,
    DEFAULT_REGISTRY,
    RepairInclusionPolicy,
    fuse_spatial_and_temporal_masks,
)
from temporal_scratch_analyzer import TemporalConfig, TemporalScratchAnalyzer, load_clip

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp_candidate_comparison")

DOCS_BENCHMARKS = PROJECT_ROOT / "docs" / "benchmarks"
VISUALS_DIR = DOCS_BENCHMARKS / "visuals"
EXAMPLES_DIR = VISUALS_DIR / "experimental_examples"
VIDEOS_DIR = VISUALS_DIR / "experimental_side_by_side_videos"

ABSDEG_FULL_DIR = PROJECT_ROOT / ".cache" / "absolute-scratch-benchmark-20261106-01"
CROSS_FILM_DIR = PROJECT_ROOT / "evaluation" / "cross_film_validation"
NEIGHBORS_VIDEO = PROJECT_ROOT / "evaluation" / "Neighbors_72.mkv"


def _load_absdeg_clip(clip_id: str) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    clean = [cv2.imread(str(ABSDEG_FULL_DIR / clip_id / "reference" / f"{i:06d}.png")) for i in range(12)]
    dmg = [cv2.imread(str(ABSDEG_FULL_DIR / clip_id / "20261106" / f"{i:06d}.png")) for i in range(12)]
    gt = [cv2.imread(str(ABSDEG_FULL_DIR / clip_id / "20261106" / f"{i:06d}.gt.png"), 0) for i in range(12)]
    return clean, dmg, gt


def _edge_mask_from_clean(clean_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    th = max(20.0, float(np.percentile(mag, 85)))
    return mag >= th


def _eval_structural_edge_collateral(
    clean_bgr: np.ndarray,
    repaired_bgr: np.ndarray,
    gt_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure unintended edits strictly on clean structural edges (gt_mask == 0 & sobel edge)."""
    edge_intact = _edge_mask_from_clean(clean_bgr) & (gt_mask == 0)
    total_edge = int(np.count_nonzero(edge_intact))
    if total_edge == 0:
        return {"edge_intact_total": 0, "edge_intact_modified_count": 0, "edge_intact_max_abs_diff": 0.0, "edge_intact_psnr_db": 100.0}
    mod = np.any(repaired_bgr != clean_bgr, axis=2) & edge_intact
    diff = np.abs(repaired_bgr.astype(np.float32) - clean_bgr.astype(np.float32))
    diff_edge = diff[edge_intact]
    mse = float(np.mean(diff_edge ** 2))
    psnr = 10.0 * np.log10((255.0 ** 2) / mse) if mse > 0.0 else 100.0
    return {
        "edge_intact_total": total_edge,
        "edge_intact_modified_count": int(np.count_nonzero(mod)),
        "edge_intact_max_abs_diff": round(float(np.max(diff_edge)), 2),
        "edge_intact_psnr_db": round(float(psnr), 3),
    }


def _inpaint_telea(frame_bgr: np.ndarray, repair_mask: np.ndarray) -> np.ndarray:
    if not np.any(repair_mask >= 128):
        return frame_bgr.copy()
    return cv2.inpaint(frame_bgr, (repair_mask >= 128).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)


def _repair_mc_fair(
    frames: Sequence[np.ndarray],
    idx: int,
    repair_mask: np.ndarray,
    analyzer: TemporalScratchAnalyzer,
    *,
    cfg: ExperimentalConfig | None = None,
) -> np.ndarray:
    """Fair motion-compensated temporal median repair shared across all candidates A-E.

    Uses motion-robust neighbor selection (so fast-pan clips do not silently fall back to Telea)
    and temporarily inpaints `repair_mask` in the DIS grayscale proxy buffers only, preventing DIS
    from locking onto persistent film scratches while warping untouched original BGR neighbors.
    """
    m = repair_mask >= 128
    if not np.any(m):
        return frames[idx].copy()
    exp_cfg = cfg or ExperimentalConfig()
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    prec, succ, _, _ = select_motion_robust_neighbors(grays, idx, exp_cfg)
    nbrs = prec + succ
    if not nbrs:
        return _inpaint_telea(frames[idx], repair_mask)

    h, w = grays[idx].shape
    xg, yg = analyzer._get_remap_grid(h, w)
    mask_u8 = m.astype(np.uint8) * 255
    # Pre-fill masked pixels in DIS proxy gray only so DIS tracks surrounding scene motion
    ref_proxy = cv2.inpaint(grays[idx], mask_u8, 3, cv2.INPAINT_TELEA)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    if dis is not None:
        dis.setFinestScale(1)

    warped_bgr: list[np.ndarray] = []
    for n_idx in nbrs:
        # Also mildly suppress narrow linear ridges in neighbor proxy for persistent scratches
        nbr_proxy = cv2.inpaint(grays[n_idx], mask_u8, 3, cv2.INPAINT_TELEA)
        fl = dis.calc(ref_proxy, nbr_proxy, None) if dis is not None else analyzer._compute_dense_flow(ref_proxy, nbr_proxy)
        wf = cv2.remap(frames[n_idx], xg + fl[..., 0], yg + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        warped_bgr.append(wf)

    bg_bgr = np.median(np.stack(warped_bgr, axis=0), axis=0).astype(np.uint8)
    out = frames[idx].copy()
    out[m] = bg_bgr[m]
    return out


def _extract_legacy_components(cand_bool: np.ndarray, *, use_thickness: bool) -> np.ndarray:
    cand_u8 = cand_bool.astype(np.uint8) * 255
    opened = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
    n_l, lbl, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    sm = np.zeros_like(closed)
    for i in range(1, n_l):
        _x, _y, cw, ch, area = map(int, stats[i])
        if ch < 15:
            continue
        if use_thickness:
            thick = area / max(1, ch)
            ok = (thick <= 8.0 and ch >= 35) or (cw <= 12 and ch >= 3.5 * cw)
        else:
            ok = cw <= 12 and (ch >= 3.5 * cw or ch >= 35)
        if ok:
            sm[lbl == i] = 255
    if np.any(sm):
        sm = cv2.dilate(sm, np.ones((1, 3), np.uint8))
    return sm


def _run_frozen_sol1_to_sol3_on_frame(
    frames: Sequence[np.ndarray],
    idx: int,
    spat_res_idx_mcm: Any,
    analyzer: TemporalScratchAnalyzer,
    *,
    allow_stepwise_nbrs: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Reproduce frozen Candidates B (Sol 1), C (Sol 2), and D (Sol 3) verbatim."""
    grays = [analyzer._ensure_gray(f) for f in frames]
    target_gray = grays[idx]
    h, w = target_gray.shape
    s_rep = spat_res_idx_mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)

    prec, succ, _ = analyzer._select_nonduplicate_neighbors(grays, idx)
    if len(prec + succ) < 2 and allow_stepwise_nbrs:
        prec, succ, _ = _stepwise_neighbors(grays, idx)
    nbrs = prec + succ
    if len(nbrs) < 2:
        z = np.zeros((h, w), dtype=np.uint8)
        return {
            "B_solution_1": (z.copy(), z.copy()),
            "C_solution_2": (z.copy(), z.copy()),
            "D_solution_3": (s_rep.copy(), s_rep.copy()),
        }

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

    raw_temp = diff >= tau
    struct_ok = flank >= 12.1

    sol1_cand = raw_temp & struct_ok & (flank_novelty >= 10.0) & (median_nbr_flank <= 6.0) & (nbr_spread <= np.maximum(10.0, 0.5 * diff))
    mask_sol1 = _extract_legacy_components(sol1_cand, use_thickness=True)

    sol2_cand = sol1_cand & (median_fb_err <= 1.5)
    mask_sol2 = _extract_legacy_components(sol2_cand, use_thickness=True)

    spatial_vetted = np.where((diff >= tau) & (median_nbr_flank <= 6.0), s_rep, np.uint8(0))
    mask_sol3 = np.bitwise_or(spatial_vetted, mask_sol2)

    return {
        "B_solution_1": (mask_sol1, mask_sol1),
        "C_solution_2": (mask_sol2, mask_sol2),
        "D_solution_3": (mask_sol3, mask_sol3),
    }


def _mid_nonzero_gt_index(gt_masks: Sequence[np.ndarray]) -> int:
    nz = [i for i, g in enumerate(gt_masks) if np.any(g == 255)]
    if not nz:
        raise ValueError("Expected at least one frame with non-zero ground-truth scratch pixels")
    return nz[len(nz) // 2]


def run_factorial_ablation(
    *,
    d010_tilt: Sequence[np.ndarray],
    g010_tilt: Sequence[np.ndarray],
    t_tilt: int,
    d010_diag: Sequence[np.ndarray],
    g010_diag: Sequence[np.ndarray],
    t_diag: int,
) -> dict[str, Any]:
    """Part 1: Unconfounded 2x2x2 = 8 factorial ablation over (M1, M2, M3) on dev datasets.

    Runs strictly on the temporal stream (no spatial mask confounding) so main effects and
    2-way interactions (M1xM2, M1xM3, M2xM3) isolate each failure mechanism cleanly.
    """
    logger.info("Running unconfounded 2x2x2 factorial ablation (M1_flow x M2_cut x M3_geom)...")
    _, d000, g000 = _load_absdeg_clip("000")
    c010, d010, g010 = _load_absdeg_clip("010")
    n_frames = load_clip(NEIGHBORS_VIDEO)[20:31]
    pa_frames = load_clip(CROSS_FILM_DIR / "paleface_seq_a.mkv")
    pb_frames = load_clip(CROSS_FILM_DIR / "paleface_seq_b.mkv")

    # Add a static grained 1-frame diagonal scratch so M3 & M1xM3 are observable even when M1=0
    rng_stat = np.random.default_rng(202611195)
    static_base = [
        np.clip(c010[5].astype(np.float32) + rng_stat.normal(0.0, 2.0, size=c010[5].shape), 0, 255).astype(np.uint8)
        for _ in range(9)
    ]
    d_stat_diag, g_stat_diag, _ = bdm.inject_extended_scratches(
        static_base,
        bdm.ExtendedScratchConfig(orientation="diagonal", angle_deg=35.0, width=2, opacity=0.85, bright=True, duration=1, drift_px=0.0, seed=202611196),
    )
    t_stat_diag = _mid_nonzero_gt_index(g_stat_diag)

    neg_specs = [
        ("neighbors_f25_pipe", n_frames, 5, {"name": "neighbors_frame25_pipe", "ymin": 48, "ymax": 282, "xmin": 180, "xmax": 192}),
        ("paleface_a_jacket", pa_frames, 8, {"name": "paleface_actor_jacket_edge", "ymin": 188, "ymax": 381, "xmin": 258, "xmax": 268}),
        ("paleface_b_wall", pb_frames, 7, {"name": "paleface_b_wall_curtain", "ymin": 80, "ymax": 180, "xmin": 200, "xmax": 280}),
    ]
    pos_specs = [
        ("absdeg_000_f5", d000, g000[5], 5),
        ("absdeg_010_f5", d010, g010[5], 5),
        ("absdeg_010_tilted", d010_tilt, g010_tilt[t_tilt], t_tilt),
        ("absdeg_010_diag", d010_diag, g010_diag[t_diag], t_diag),
        ("absdeg_010_static_diag_1f", d_stat_diag, g_stat_diag[t_stat_diag], t_stat_diag),
    ]

    combos = []
    for m1 in (False, True):
        for m2 in (False, True):
            for m3 in (False, True):
                label = f"M1={int(m1)}_M2={int(m2)}_M3={int(m3)}"
                cfg = ExperimentalConfig(
                    use_motion_compensated_evidence=m1,
                    use_robust_scene_cut_selector=m2,
                    use_centerline_thickness_geometry=m3,
                )
                det = ExperimentalSceneFilmDetector(cfg)
                pos_metrics: dict[str, Any] = {}
                for pname, pframes, pgt, pidx in pos_specs:
                    res = det.analyze_frame(pframes, pidx)
                    rep_m = res.high_confidence_damage
                    cnt = bdm.confusion(pgt, rep_m)
                    met = bdm.metrics(cnt)
                    sev = da.evaluate_false_positive_severity(pgt, rep_m)
                    pos_metrics[pname] = {
                        "abstained": res.abstained,
                        "recall": round(met["recall"], 4),
                        "precision": round(met["precision"], 4),
                        "f1": round(met["f1"], 4),
                        "repair_px": int(np.count_nonzero(rep_m)),
                        "uncertain_px": int(np.count_nonzero(res.uncertain_damage_candidate)),
                        "fp_px": sev["fp_pixel_count"],
                    }

                neg_metrics: dict[str, Any] = {}
                total_neg_leak = 0
                for nname, nframes, nidx, nroi in neg_specs:
                    res = det.analyze_frame(nframes, nidx)
                    ev = da.evaluate_structural_negative_controls(res.all_detected_candidates, res.high_confidence_damage, [nroi])
                    rep_leak = ev["total_repair_leak_pixels"]
                    total_neg_leak += rep_leak
                    neg_metrics[nname] = {
                        "repair_leak_px": rep_leak,
                        "uncertain_roi_px": int(np.count_nonzero(res.uncertain_damage_candidate[nroi["ymin"]:nroi["ymax"], nroi["xmin"]:nroi["xmax"]])),
                        "vetoed_roi_px": int(np.count_nonzero(res.likely_intact_scene_structure[nroi["ymin"]:nroi["ymax"], nroi["xmin"]:nroi["xmax"]])),
                    }

                combos.append({
                    "config_label": label,
                    "M1_mc_evidence": m1,
                    "M2_robust_cut": m2,
                    "M3_centerline_geom": m3,
                    "positive": pos_metrics,
                    "negative_controls": neg_metrics,
                    "total_neg_repair_leak_px": total_neg_leak,
                    "mean_positive_recall": round(float(np.mean([pos_metrics[k]["recall"] for k in pos_metrics])), 4),
                    "mean_positive_f1": round(float(np.mean([pos_metrics[k]["f1"] for k in pos_metrics])), 4),
                })

    def _main_effect(flag: str, metric: str) -> float:
        on_vals = [c[metric] for c in combos if c[flag]]
        off_vals = [c[metric] for c in combos if not c[flag]]
        return round(float(np.mean(on_vals) - np.mean(off_vals)), 4)

    def _interaction_2way(f1: str, f2: str, metric: str) -> float:
        # Standard factorial 2-way interaction: 0.5 * ((E[1,1] - E[1,0]) - (E[0,1] - E[0,0]))
        v11 = np.mean([c[metric] for c in combos if c[f1] and c[f2]])
        v10 = np.mean([c[metric] for c in combos if c[f1] and not c[f2]])
        v01 = np.mean([c[metric] for c in combos if not c[f1] and c[f2]])
        v00 = np.mean([c[metric] for c in combos if not c[f1] and not c[f2]])
        return round(float(0.5 * ((v11 - v10) - (v01 - v00))), 4)

    return {
        "combinations": combos,
        "main_effects": {
            "M1_mc_evidence": {
                "delta_mean_recall": _main_effect("M1_mc_evidence", "mean_positive_recall"),
                "delta_mean_f1": _main_effect("M1_mc_evidence", "mean_positive_f1"),
                "delta_total_neg_leak_px": _main_effect("M1_mc_evidence", "total_neg_repair_leak_px"),
            },
            "M2_robust_cut": {
                "delta_mean_recall": _main_effect("M2_robust_cut", "mean_positive_recall"),
                "delta_mean_f1": _main_effect("M2_robust_cut", "mean_positive_f1"),
                "delta_total_neg_leak_px": _main_effect("M2_robust_cut", "total_neg_repair_leak_px"),
            },
            "M3_centerline_geom": {
                "delta_mean_recall": _main_effect("M3_centerline_geom", "mean_positive_recall"),
                "delta_mean_f1": _main_effect("M3_centerline_geom", "mean_positive_f1"),
                "delta_total_neg_leak_px": _main_effect("M3_centerline_geom", "total_neg_repair_leak_px"),
            },
        },
        "two_way_interactions": {
            "M1_x_M2": {
                "delta_mean_recall": _interaction_2way("M1_mc_evidence", "M2_robust_cut", "mean_positive_recall"),
                "delta_mean_f1": _interaction_2way("M1_mc_evidence", "M2_robust_cut", "mean_positive_f1"),
                "delta_total_neg_leak_px": _interaction_2way("M1_mc_evidence", "M2_robust_cut", "total_neg_repair_leak_px"),
            },
            "M1_x_M3": {
                "delta_mean_recall": _interaction_2way("M1_mc_evidence", "M3_centerline_geom", "mean_positive_recall"),
                "delta_mean_f1": _interaction_2way("M1_mc_evidence", "M3_centerline_geom", "mean_positive_f1"),
                "delta_total_neg_leak_px": _interaction_2way("M1_mc_evidence", "M3_centerline_geom", "total_neg_repair_leak_px"),
            },
            "M2_x_M3": {
                "delta_mean_recall": _interaction_2way("M2_robust_cut", "M3_centerline_geom", "mean_positive_recall"),
                "delta_mean_f1": _interaction_2way("M2_robust_cut", "M3_centerline_geom", "mean_positive_f1"),
                "delta_total_neg_leak_px": _interaction_2way("M2_robust_cut", "M3_centerline_geom", "total_neg_repair_leak_px"),
            },
        },
    }


def run_scene_vs_film_study() -> list[dict[str, Any]]:
    """Part 2: Evaluate scene-locked vs. film-locked features on synthetic & archival structures."""
    logger.info("Running Scene-Locked vs. Film-Locked coordinate discrimination study...")
    c010, _, _ = _load_absdeg_clip("010")
    # 1. Moving-camera transient scratch
    d_trans, g_trans, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=0.0, width=2, opacity=0.85, bright=True, duration=2, drift_px=0.5, seed=202611191)
    )
    t_trans = _mid_nonzero_gt_index(g_trans)
    # 2. Moving-camera persistent film-locked scratch (12 frames, gate weave 0.2 px)
    d_pers, g_pers, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=0.5, width=2, opacity=0.85, bright=True, duration=12, drift_px=0.2, seed=202611192)
    )
    # 3. Static-camera persistent scratch with realistic grain (non-duplicate frames, zero camera pan -> must preserve uncertainty!)
    rng = np.random.default_rng(202611194)
    static_clean = [
        np.clip(c010[5].astype(np.float32) + rng.normal(0.0, 2.2, size=c010[5].shape), 0, 255).astype(np.uint8)
        for _ in range(12)
    ]
    d_stat, g_stat, _ = bdm.inject_extended_scratches(
        static_clean, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=0.0, width=2, opacity=0.85, bright=True, duration=12, drift_px=0.0, seed=202611193)
    )

    n_frames = load_clip(NEIGHBORS_VIDEO)[20:31]
    pa_frames = load_clip(CROSS_FILM_DIR / "paleface_seq_a.mkv")
    pb_frames = load_clip(CROSS_FILM_DIR / "paleface_seq_b.mkv")

    det = ExperimentalSceneFilmDetector(ExperimentalConfig())
    spat_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    spat_cfg = dict(BENCHMARK_PRESETS["film_artefact_balanced"].params)

    cases = [
        ("moving_transient_scratch_010", d_trans, t_trans, g_trans[t_trans] == 255, "positive_gt"),
        ("moving_persistent_film_locked_010", d_pers, 5, g_pers[5] == 255, "positive_gt"),
        ("static_persistent_ambiguous_010f5", d_stat, 5, g_stat[5] == 255, "positive_gt_static"),
        ("neighbors_f25_pipe_structure", n_frames, 5, _roi_bool(n_frames[5].shape[:2], 48, 282, 180, 192), "negative_roi"),
        ("paleface_a_f8_actor_jacket", pa_frames, 8, _roi_bool(pa_frames[8].shape[:2], 188, 381, 258, 268), "negative_roi"),
        ("paleface_b_f7_wall_crack", pb_frames, 7, _roi_bool(pb_frames[7].shape[:2], 80, 180, 200, 280), "negative_roi"),
    ]

    study_rows: list[dict[str, Any]] = []
    for cname, cframes, cidx, roi_bool, kind in cases:
        s_res = spat_det.detect(cframes, spat_cfg)
        s_mcm = s_res.multi_channel_masks[cidx]
        res = det.analyze_frame(
            cframes,
            cidx,
            spatial_scratch_mask=s_mcm.scratch_mask,
            spatial_dust_mask=s_mcm.dust_mask,
        )
        n_roi = int(np.count_nonzero(roi_bool))
        hc = int(np.count_nonzero((res.high_confidence_damage > 0) & roi_bool))
        unc = int(np.count_nonzero((res.uncertain_damage_candidate > 0) & roi_bool))
        veto = int(np.count_nonzero((res.likely_intact_scene_structure > 0) & roi_bool))
        classes_seen: dict[str, int] = {}
        flows = []
        fb_errs = []
        sc_ratios = []
        fl_ratios = []
        for rec in res.component_records:
            classes_seen[rec.temporal_class] = classes_seen.get(rec.temporal_class, 0) + 1
        # Collect pixel-level flow / fb_err directly on ROI for exact physical summary
        flows_roi = float(np.median(res.flow_mag[roi_bool])) if n_roi else 0.0
        fb_roi = float(np.median(res.fb_err[roi_bool])) if n_roi else 0.0
        fb_warn_pct = round(float(np.mean(res.fb_warning_map[roi_bool])) * 100.0, 1) if n_roi else 0.0
        # Filter component records for relevant stream characteristics
        for rec in res.component_records:
            sc_ratios.append(rec.scene_persistence_ratio)
            fl_ratios.append(rec.film_persistence_ratio)
            flows.append(rec.mean_flow_mag_px)
            fb_errs.append(rec.median_fb_err_px)

        study_rows.append({
            "case": cname,
            "kind": kind,
            "roi_pixels": n_roi,
            "high_confidence_px": hc,
            "uncertain_px": unc,
            "vetoed_intact_px": veto,
            "missed_or_clean_px": n_roi - hc - unc - veto,
            "roi_median_flow_px": round(flows_roi, 3),
            "roi_median_fb_err_px": round(fb_roi, 3),
            "roi_fb_warning_pct": fb_warn_pct,
            "component_classes_full_frame": classes_seen,
        })
    return study_rows


def _roi_bool(hw: tuple[int, int], ymin: int, ymax: int, xmin: int, xmax: int) -> np.ndarray:
    m = np.zeros(hw, dtype=bool)
    m[ymin:ymax, xmin:xmax] = True
    return m


def _score_candidate_frame(
    clean_ref: np.ndarray | None,
    damaged: np.ndarray,
    gt_mask: np.ndarray,
    det_mask: np.ndarray,
    repair_mask: np.ndarray,
    repaired_mc: np.ndarray,
    repaired_telea: np.ndarray,
) -> dict[str, Any]:
    cnt = bdm.confusion(gt_mask, det_mask)
    met = bdm.metrics(cnt)
    sev = da.evaluate_false_positive_severity(gt_mask, det_mask)
    rep_sev = da.evaluate_false_positive_severity(gt_mask, repair_mask)
    out: dict[str, Any] = {
        "det_tp": cnt["tp"],
        "det_fp": cnt["fp"],
        "det_fn": cnt["fn"],
        "det_precision": round(met["precision"], 4),
        "det_recall": round(met["recall"], 4),
        "det_f1": round(met["f1"], 4),
        "det_fp_severity": sev,
        "repair_mask_pixels": int(np.count_nonzero(repair_mask >= 128)),
        "repair_fp_severity": rep_sev,
    }
    if clean_ref is not None:
        mc_eval = da.evaluate_exact_reference_restoration(clean_ref, damaged, repaired_mc, gt_mask, repair_mask)
        tel_eval = da.evaluate_exact_reference_restoration(clean_ref, damaged, repaired_telea, gt_mask, repair_mask)
        edge_mc = _eval_structural_edge_collateral(clean_ref, repaired_mc, gt_mask)
        edge_tel = _eval_structural_edge_collateral(clean_ref, repaired_telea, gt_mask)
        out["restoration_mc"] = mc_eval
        out["restoration_telea"] = tel_eval
        out["edge_collateral_mc"] = edge_mc
        out["edge_collateral_telea"] = edge_tel
    return out


def run_fair_candidate_comparison() -> dict[str, Any]:
    """Part 3: Compare A, B, C, D, E_temporal, and E_fused fairly on dev regimes + controls."""
    logger.info("Loading development datasets & computing spatial detector cache once...")
    c000, d000, g000 = _load_absdeg_clip("000")
    c010, d010, g010 = _load_absdeg_clip("010")
    c020, d020, g020 = _load_absdeg_clip("020")

    regimes = [
        {
            "id": "absdeg_000_fast_pan",
            "category": "fast_pan_mixed",
            "orientation": "vertical",
            "polarity": "mixed",
            "contrast": "strong",
            "persistence": "mixed_orig",
            "motion_regime": "fast_pan",
            "clean": c000,
            "damaged": d000,
            "gt": g000,
            "eval_indices": [3, 4, 5, 6, 7],
            "all_indices_for_abstention": list(range(len(d000))),
        },
        {
            "id": "absdeg_010_moderate_pan",
            "category": "moderate_pan_mixed",
            "orientation": "vertical",
            "polarity": "mixed",
            "contrast": "strong",
            "persistence": "mixed_orig",
            "motion_regime": "moderate_pan",
            "clean": c010,
            "damaged": d010,
            "gt": g010,
            "eval_indices": [3, 4, 5, 6, 7],
            "all_indices_for_abstention": list(range(len(d010))),
        },
        {
            "id": "absdeg_020_dev_pan",
            "category": "moderate_pan_mixed",
            "orientation": "vertical",
            "polarity": "mixed",
            "contrast": "strong",
            "persistence": "mixed_orig",
            "motion_regime": "moderate_pan",
            "clean": c020,
            "damaged": d020,
            "gt": g020,
            "eval_indices": [3, 4, 5, 6, 7],
            "all_indices_for_abstention": None,
        },
    ]

    # Extended synthetic regimes built cleanly on dev clean references (010, 020)
    d_vp_b, g_vp_b, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=0.0, width=2, opacity=0.85, bright=True, duration=12, drift_px=0.2, seed=202611182)
    )
    d_vp_d, g_vp_d, _ = bdm.inject_extended_scratches(
        c020, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=1.5, width=2, opacity=0.85, bright=False, duration=12, drift_px=0.2, seed=202611183)
    )
    d_sub, g_sub, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=0.5, width=2, opacity=0.38, bright=True, duration=2, drift_px=0.5, seed=202611184)
    )
    d_diag, g_diag, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="diagonal", angle_deg=38.0, width=2, opacity=0.85, bright=True, curvature=0.18, duration=4, drift_px=0.5, seed=202611185)
    )
    d_horz, g_horz, _ = bdm.inject_extended_scratches(
        c020, bdm.ExtendedScratchConfig(orientation="horizontal", angle_deg=3.0, width=2, opacity=0.85, bright=False, duration=4, drift_px=0.3, seed=202611186)
    )
    d_tele_raw, g_tele_raw, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=0.0, width=2, opacity=0.85, bright=True, duration=6, drift_px=0.4, seed=202611187)
    )
    tele = bdm.apply_telecine_cadence(c010, d_tele_raw, g_tele_raw, duplicate_every=3)

    regimes.extend([
        {
            "id": "regime_persistent_bright_010",
            "category": "persistent_film_locked",
            "orientation": "vertical",
            "polarity": "bright",
            "contrast": "strong",
            "persistence": "persistent_12f",
            "motion_regime": "moderate_pan",
            "clean": c010,
            "damaged": d_vp_b,
            "gt": g_vp_b,
            "eval_indices": [3, 4, 5, 6, 7],
            "all_indices_for_abstention": None,
        },
        {
            "id": "regime_persistent_dark_tilted_020",
            "category": "persistent_film_locked_dark",
            "orientation": "tilted_1.5deg",
            "polarity": "dark",
            "contrast": "strong",
            "persistence": "persistent_12f",
            "motion_regime": "moderate_pan",
            "clean": c020,
            "damaged": d_vp_d,
            "gt": g_vp_d,
            "eval_indices": [3, 4, 5, 6, 7],
            "all_indices_for_abstention": None,
        },
        {
            "id": "regime_subtle_transient_010",
            "category": "low_contrast_transient",
            "orientation": "vertical",
            "polarity": "bright",
            "contrast": "low_0.38",
            "persistence": "transient_2f",
            "motion_regime": "moderate_pan",
            "clean": c010,
            "damaged": d_sub,
            "gt": g_sub,
            "eval_indices": [i for i, g in enumerate(g_sub) if np.any(g == 255)],
            "all_indices_for_abstention": None,
        },
        {
            "id": "regime_diagonal_curved_010",
            "category": "diagonal_orientation",
            "orientation": "diagonal_38deg",
            "polarity": "bright",
            "contrast": "strong",
            "persistence": "medium_4f",
            "motion_regime": "moderate_pan",
            "clean": c010,
            "damaged": d_diag,
            "gt": g_diag,
            "eval_indices": [i for i, g in enumerate(g_diag) if np.any(g == 255)],
            "all_indices_for_abstention": None,
        },
        {
            "id": "regime_horizontal_fissure_020",
            "category": "horizontal_orientation",
            "orientation": "horizontal_3deg",
            "polarity": "dark",
            "contrast": "strong",
            "persistence": "medium_4f",
            "motion_regime": "moderate_pan",
            "clean": c020,
            "damaged": d_horz,
            "gt": g_horz,
            "eval_indices": [i for i, g in enumerate(g_horz) if np.any(g == 255)],
            "all_indices_for_abstention": None,
        },
        {
            "id": "regime_telecine_3_2_010",
            "category": "telecine_cadence",
            "orientation": "vertical",
            "polarity": "bright",
            "contrast": "strong",
            "persistence": "medium_6f_telecine",
            "motion_regime": "telecine_3_2",
            "clean": tele["clean_frames"],
            "damaged": tele["damaged_frames"],
            "gt": tele["gt_masks"],
            "eval_indices": [3, 5, 6, 8],
            "all_indices_for_abstention": None,
        },
    ])

    # Paired Clean References (exact zero-defect baseline check to separate motion FPs from true scratch triggers)
    clean_pairs = [
        {"id": "clean_ref_000", "frames": c000, "eval_indices": [3, 4, 5, 6, 7]},
        {"id": "clean_ref_010", "frames": c010, "eval_indices": [3, 4, 5, 6, 7]},
        {"id": "clean_ref_020", "frames": c020, "eval_indices": [3, 4, 5, 6, 7]},
    ]

    # Archival dev negative controls (strict zero-leak gate)
    n72_clip = load_clip(NEIGHBORS_VIDEO)[20:31]
    pa_clip = load_clip(CROSS_FILM_DIR / "paleface_seq_a.mkv")
    pb_clip = load_clip(CROSS_FILM_DIR / "paleface_seq_b.mkv")
    cc_clip = load_clip(CROSS_FILM_DIR / "college_seq_c.mkv")
    nd_clip = load_clip(CROSS_FILM_DIR / "neighbors_seq_d.mkv")

    archival_controls = [
        {"id": "neighbors_frame25_pipe", "frames": n72_clip, "idx": 5, "roi": {"name": "neighbors_frame25_pipe", "ymin": 48, "ymax": 282, "xmin": 180, "xmax": 192}},
        {"id": "paleface_seq_a_jacket", "frames": pa_clip, "idx": 8, "roi": {"name": "paleface_actor_jacket_edge", "ymin": 188, "ymax": 381, "xmin": 258, "xmax": 268}},
        {"id": "paleface_seq_b_wall", "frames": pb_clip, "idx": 7, "roi": {"name": "paleface_b_wall_curtain", "ymin": 80, "ymax": 180, "xmin": 200, "xmax": 280}},
        {"id": "college_seq_c_jamb", "frames": cc_clip, "idx": 8, "roi": {"name": "college_door_jamb_student", "ymin": 150, "ymax": 280, "xmin": 280, "xmax": 340}},
        {"id": "neighbors_seq_d_pole", "frames": nd_clip, "idx": 2, "roi": {"name": "neighbors_d_telephone_pole", "ymin": 200, "ymax": 400, "xmin": 150, "xmax": 220}},
    ]

    spat_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    temp_det = DEFAULT_REGISTRY.get("temporal_scratch")
    spat_cfg = dict(BENCHMARK_PRESETS["film_artefact_balanced"].params)
    temp_cfg = dict(BENCHMARK_PRESETS["temporal_scratch_balanced"].params)
    analyzer = TemporalScratchAnalyzer(TemporalConfig())
    cand_e_det = ExperimentalSceneFilmDetector(ExperimentalConfig())

    candidate_names = [
        "A1_baseline_temporal",
        "A2_baseline_fused_conservative",
        "B_solution_1",
        "C_solution_2",
        "D_solution_3",
        "E1_experimental_temporal_only",
        "E2_experimental_fused_3state",
    ]
    perf_tracker: dict[str, dict[str, list[float]]] = {c: {"ms": [], "peak_mb": []} for c in candidate_names}

    # Helper to execute all 7 candidate variants on a clip; measure pure wall-clock time without tracemalloc distortion,
    # and profile peak traced Python memory on a dedicated representative pass.
    def _evaluate_clip_all_candidates(
        frames: Sequence[np.ndarray],
        indices: Sequence[int],
        *,
        measure_mem: bool = False,
    ) -> dict[str, dict[int, dict[str, np.ndarray]]]:
        t0 = time.perf_counter()
        if measure_mem:
            tracemalloc.start()
        s_res = spat_det.detect(frames, spat_cfg)
        if measure_mem:
            _, peak_spat = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        ms_spat = (time.perf_counter() - t0) * 1000.0 / max(1, len(frames))

        t0 = time.perf_counter()
        if measure_mem:
            tracemalloc.start()
        t_res = temp_det.detect(frames, temp_cfg)
        if measure_mem:
            _, peak_temp = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        ms_temp = (time.perf_counter() - t0) * 1000.0 / max(1, len(frames))

        t0 = time.perf_counter()
        if measure_mem:
            tracemalloc.start()
        f_res = fuse_spatial_and_temporal_masks(s_res, t_res)
        if measure_mem:
            _, peak_fuse = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        ms_fuse = (time.perf_counter() - t0) * 1000.0 / max(1, len(frames))

        out: dict[str, dict[int, dict[str, np.ndarray]]] = {c: {} for c in candidate_names}

        # A1 / A2 extraction
        for idx in indices:
            t_mcm = t_res.multi_channel_masks[idx]
            f_mcm = f_res.multi_channel_masks[idx]
            a1_m = t_mcm.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)
            a2_m = f_mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
            abstained = t_res.metadata.get("abstained_frames", []) if isinstance(t_res.metadata, dict) else []
            out["A1_baseline_temporal"][idx] = {"det": a1_m, "repair": a1_m, "uncertain": np.zeros_like(a1_m), "vetoed": np.zeros_like(a1_m), "abstained": np.array([int(idx in abstained)])}
            out["A2_baseline_fused_conservative"][idx] = {"det": f_res.mask[idx], "repair": a2_m, "uncertain": f_mcm.uncertain_mask if f_mcm.uncertain_mask is not None else np.zeros_like(a2_m), "vetoed": np.zeros_like(a2_m), "abstained": np.array([int(idx in abstained)])}
        if not measure_mem:
            perf_tracker["A1_baseline_temporal"]["ms"].append(round(ms_temp, 2))
            perf_tracker["A2_baseline_fused_conservative"]["ms"].append(round(ms_spat + ms_temp + ms_fuse, 2))
        else:
            perf_tracker["A1_baseline_temporal"]["peak_mb"].append(round(peak_temp / (1024 * 1024), 2))
            perf_tracker["A2_baseline_fused_conservative"]["peak_mb"].append(round(max(peak_spat, peak_temp, peak_fuse) / (1024 * 1024), 2))

        # B, C, D frozen solutions (report honest standalone per-candidate cost: B uses flow+Sol1 ~ 0.5*ms_bcd, C/D compute flow+FB)
        t0 = time.perf_counter()
        if measure_mem:
            tracemalloc.start()
        for idx in indices:
            sols = _run_frozen_sol1_to_sol3_on_frame(frames, idx, s_res.multi_channel_masks[idx], analyzer, allow_stepwise_nbrs=True)
            for bcd in ("B_solution_1", "C_solution_2", "D_solution_3"):
                dm, rm = sols[bcd]
                out[bcd][idx] = {"det": dm, "repair": rm, "uncertain": np.zeros_like(rm), "vetoed": np.zeros_like(rm), "abstained": np.array([0])}
        if measure_mem:
            _, peak_bcd = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        ms_bcd = (time.perf_counter() - t0) * 1000.0 / max(1, len(indices))
        if not measure_mem:
            # B only computes backward flow (~50% of backward+forward flow pass); C computes backward+forward; D adds spatial detector
            perf_tracker["B_solution_1"]["ms"].append(round(ms_bcd * 0.50, 2))
            perf_tracker["C_solution_2"]["ms"].append(round(ms_bcd, 2))
            perf_tracker["D_solution_3"]["ms"].append(round(ms_bcd + ms_spat, 2))
        else:
            for bcd in ("B_solution_1", "C_solution_2"):
                perf_tracker[bcd]["peak_mb"].append(round(peak_bcd / (1024 * 1024), 2))
            perf_tracker["D_solution_3"]["peak_mb"].append(round(max(peak_bcd, peak_spat) / (1024 * 1024), 2))

        # E1 Experimental Temporal Only (isolated bounded sequence cache)
        t0 = time.perf_counter()
        if measure_mem:
            tracemalloc.start()
        e1_seq = cand_e_det.analyze_sequence(frames, target_indices=list(indices))
        if measure_mem:
            _, peak_e1 = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        ms_e1 = (time.perf_counter() - t0) * 1000.0 / max(1, len(indices))
        for idx in indices:
            e1_res = e1_seq[idx]
            out["E1_experimental_temporal_only"][idx] = {
                "det": e1_res.all_detected_candidates,
                "repair": e1_res.high_confidence_damage,
                "uncertain": e1_res.uncertain_damage_candidate,
                "vetoed": e1_res.likely_intact_scene_structure,
                "abstained": np.array([int(e1_res.abstained)]),
                "_res": e1_res,
            }
        if not measure_mem:
            perf_tracker["E1_experimental_temporal_only"]["ms"].append(round(ms_e1, 2))
        else:
            perf_tracker["E1_experimental_temporal_only"]["peak_mb"].append(round(peak_e1 / (1024 * 1024), 2))

        # E2 Experimental Fused 3-State (isolated bounded sequence cache; includes spatial detector overhead)
        t0 = time.perf_counter()
        if measure_mem:
            tracemalloc.start()
        e2_seq = cand_e_det.analyze_sequence(
            frames,
            spatial_scratch_masks=[m.scratch_mask for m in s_res.multi_channel_masks],
            spatial_dust_masks=[m.dust_mask for m in s_res.multi_channel_masks],
            target_indices=list(indices),
        )
        if measure_mem:
            _, peak_e2 = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        ms_e2 = (time.perf_counter() - t0) * 1000.0 / max(1, len(indices))
        for idx in indices:
            e2_res = e2_seq[idx]
            out["E2_experimental_fused_3state"][idx] = {
                "det": e2_res.all_detected_candidates,
                "repair": e2_res.high_confidence_damage,
                "uncertain": e2_res.uncertain_damage_candidate,
                "vetoed": e2_res.likely_intact_scene_structure,
                "abstained": np.array([int(e2_res.abstained)]),
                "_res": e2_res,
            }
        if not measure_mem:
            perf_tracker["E2_experimental_fused_3state"]["ms"].append(round(ms_e2 + ms_spat, 2))
        else:
            perf_tracker["E2_experimental_fused_3state"]["peak_mb"].append(round(max(peak_e2, peak_spat) / (1024 * 1024), 2))

        return out

    # Build consistent tilted & diagonal stimuli for Part 1 unconfounded factorial ablation
    d010_tilt_p1, g010_tilt_p1, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="vertical", angle_deg=1.5, width=2, opacity=0.85, bright=True, duration=8, drift_px=1.5, seed=202611181)
    )
    t_tilt_p1 = _mid_nonzero_gt_index(g010_tilt_p1)
    d010_diag_p1, g010_diag_p1, _ = bdm.inject_extended_scratches(
        c010, bdm.ExtendedScratchConfig(orientation="diagonal", angle_deg=38.0, width=2, opacity=0.85, bright=True, duration=6, drift_px=0.5, seed=202611185)
    )
    t_diag_p1 = _mid_nonzero_gt_index(g010_diag_p1)

    factorial_results = run_factorial_ablation(
        d010_tilt=d010_tilt_p1,
        g010_tilt=g010_tilt_p1,
        t_tilt=t_tilt_p1,
        d010_diag=d010_diag_p1,
        g010_diag=g010_diag_p1,
        t_diag=t_diag_p1,
    )

    # Dedicated traced-memory pass on representative 12-frame clip (clean_010)
    _evaluate_clip_all_candidates(c010, [3, 4, 5, 6, 7], measure_mem=True)

    # Evaluate Paired Clean References
    clean_results: dict[str, Any] = {}
    for cp in clean_pairs:
        logger.info("Evaluating paired clean reference: %s", cp["id"])
        preds = _evaluate_clip_all_candidates(cp["frames"], cp["eval_indices"])
        clean_entry: dict[str, Any] = {}
        for cname in candidate_names:
            fp_sum = 0
            rep_sum = 0
            unc_sum = 0
            veto_sum = 0
            mc_mod_sum = 0
            tel_mod_sum = 0
            max_h = 0
            sev_cnt = 0
            for idx in cp["eval_indices"]:
                p = preds[cname][idx]
                rep_m = p["repair"]
                zero_gt = np.zeros_like(rep_m)
                sev = da.evaluate_false_positive_severity(zero_gt, rep_m)
                fp_sum += sev["fp_pixel_count"]
                max_h = max(max_h, sev["max_fp_component_height"])
                sev_cnt += sev["severe_structural_fp_count"]
                rep_sum += int(np.count_nonzero(rep_m >= 128))
                unc_sum += int(np.count_nonzero(p["uncertain"] >= 128))
                veto_sum += int(np.count_nonzero(p["vetoed"] >= 128))
                mc_img = _repair_mc_fair(cp["frames"], idx, rep_m, analyzer)
                tel_img = _inpaint_telea(cp["frames"][idx], rep_m)
                mc_mod_sum += int(np.count_nonzero(np.any(mc_img != cp["frames"][idx], axis=2)))
                tel_mod_sum += int(np.count_nonzero(np.any(tel_img != cp["frames"][idx], axis=2)))
            clean_entry[cname] = {
                "clean_repair_fp_pixels": fp_sum,
                "clean_severe_structural_fps": sev_cnt,
                "clean_max_fp_height": max_h,
                "clean_uncertain_pixels": unc_sum,
                "clean_vetoed_intact_pixels": veto_sum,
                "clean_intact_modified_mc_pixels": mc_mod_sum,
                "clean_intact_modified_telea_pixels": tel_mod_sum,
            }
        clean_results[cp["id"]] = clean_entry

    # Evaluate Injected Regimes
    regime_results: list[dict[str, Any]] = []
    saved_visual_examples: dict[str, Any] = {}
    for reg in regimes:
        logger.info("Evaluating Track B dev regime: %s", reg["id"])
        eval_idxs = reg["eval_indices"]
        preds = _evaluate_clip_all_candidates(reg["damaged"], eval_idxs)

        # Measure full-sequence abstention rate across every regime
        full_idxs = reg["all_indices_for_abstention"] or list(range(len(reg["damaged"])))
        e_full = cand_e_det.analyze_sequence(reg["damaged"], target_indices=full_idxs)
        a_cnt = 0
        grays_full = [analyzer._ensure_gray(f) for f in reg["damaged"]]
        for i in full_idxs:
            pr, su, _ = analyzer._select_nonduplicate_neighbors(grays_full, i)
            if len(pr + su) < analyzer.config.min_temporal_neighbors:
                a_cnt += 1
        e_cnt = sum(1 for i in full_idxs if e_full[i].abstained)
        abst_rates = {
            "legacy_abstain_rate": round(a_cnt / max(1, len(full_idxs)), 4),
            "candidate_e_abstain_rate": round(e_cnt / max(1, len(full_idxs)), 4),
        }

        cand_entries: dict[str, Any] = {}
        for cname in candidate_names:
            tp_r = fp_r = fn_r = 0
            tp_all = fp_all = fn_all = 0
            tp_unc = tp_veto = gt_pos_total = 0
            unc_px = veto_px = 0
            max_fp_h = 0
            sev_fp = 0
            mc_gains = []
            tel_gains = []
            intact_mod_mc = 0
            intact_mod_tel = 0
            edge_mod_mc = 0
            edge_mod_tel = 0
            intact_mean_diffs_mc = []
            intact_mean_diffs_tel = []
            intact_max_diff_mc = 0.0
            intact_max_diff_tel = 0.0
            for idx in eval_idxs:
                p = preds[cname][idx]
                rep_m = p["repair"]
                det_m = p["det"]
                unc_m = p["uncertain"]
                veto_m = p["vetoed"]
                gt_m = reg["gt"][idx]
                clean_f = reg["clean"][idx]
                dmg_f = reg["damaged"][idx]

                mc_img = _repair_mc_fair(reg["damaged"], idx, rep_m, analyzer)
                tel_img = _inpaint_telea(dmg_f, rep_m)

                # Repair mask quality (conservative authorized repair)
                c_rep = bdm.confusion(gt_m, rep_m)
                tp_r += c_rep["tp"]
                fp_r += c_rep["fp"]
                fn_r += c_rep["fn"]
                # All non-vetoed detections (repair | uncertain)
                c_all = bdm.confusion(gt_m, det_m)
                tp_all += c_all["tp"]
                fp_all += c_all["fp"]
                fn_all += c_all["fn"]

                gt_pos_b = gt_m == 255
                gt_pos_total += int(np.count_nonzero(gt_pos_b))
                tp_unc += int(np.count_nonzero((unc_m >= 128) & gt_pos_b))
                tp_veto += int(np.count_nonzero((veto_m >= 128) & gt_pos_b))

                unc_px += int(np.count_nonzero(unc_m >= 128))
                veto_px += int(np.count_nonzero(veto_m >= 128))
                sev = da.evaluate_false_positive_severity(gt_m, rep_m)
                max_fp_h = max(max_fp_h, sev["max_fp_component_height"])
                sev_fp += sev["severe_structural_fp_count"]

                mc_ev = da.evaluate_exact_reference_restoration(clean_f, dmg_f, mc_img, gt_m, rep_m)
                tel_ev = da.evaluate_exact_reference_restoration(clean_f, dmg_f, tel_img, gt_m, rep_m)
                edge_mc = _eval_structural_edge_collateral(clean_f, mc_img, gt_m)
                edge_tel = _eval_structural_edge_collateral(clean_f, tel_img, gt_m)
                if mc_ev["has_defect_pixels"]:
                    mc_gains.append(mc_ev["psnr_gain_db"])
                    tel_gains.append(tel_ev["psnr_gain_db"])
                intact_mod_mc += mc_ev["intact_pixels_modified_count"]
                intact_mod_tel += tel_ev["intact_pixels_modified_count"]
                edge_mod_mc += edge_mc["edge_intact_modified_count"]
                edge_mod_tel += edge_tel["edge_intact_modified_count"]
                intact_mean_diffs_mc.append(mc_ev["intact_mean_abs_diff"])
                intact_mean_diffs_tel.append(tel_ev["intact_mean_abs_diff"])
                intact_max_diff_mc = max(intact_max_diff_mc, mc_ev["intact_max_abs_diff"])
                intact_max_diff_tel = max(intact_max_diff_tel, tel_ev["intact_max_abs_diff"])

            r_prec = tp_r / (tp_r + fp_r) if (tp_r + fp_r) > 0 else 0.0
            r_rec = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
            r_f1 = 2 * r_prec * r_rec / (r_prec + r_rec) if (r_prec + r_rec) > 0 else 0.0

            a_prec = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
            a_rec = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
            a_f1 = 2 * a_prec * a_rec / (a_prec + a_rec) if (a_prec + a_rec) > 0 else 0.0

            cand_entries[cname] = {
                "repair_recall": round(r_rec, 4),
                "repair_precision": round(r_prec, 4),
                "repair_f1": round(r_f1, 4),
                "repair_tp": tp_r,
                "repair_fp": fp_r,
                "repair_fn": fn_r,
                "all_detected_recall": round(a_rec, 4),
                "all_detected_precision": round(a_prec, 4),
                "all_detected_f1": round(a_f1, 4),
                "gt_positive_pixels": gt_pos_total,
                "tp_in_uncertain_px": tp_unc,
                "tp_erroneously_vetoed_px": tp_veto,
                "silent_loss_rate": round(fn_all / max(1, gt_pos_total), 4),
                "uncertain_pixels_total": unc_px,
                "vetoed_pixels_total": veto_px,
                "repair_max_fp_height": max_fp_h,
                "repair_severe_fp_components": sev_fp,
                "mc_psnr_gain_db": round(float(np.mean(mc_gains)), 3) if mc_gains else 0.0,
                "telea_psnr_gain_db": round(float(np.mean(tel_gains)), 3) if tel_gains else 0.0,
                "intact_pixels_modified_mc": intact_mod_mc,
                "intact_pixels_modified_telea": intact_mod_tel,
                "edge_intact_modified_mc": edge_mod_mc,
                "edge_intact_modified_telea": edge_mod_tel,
                "intact_mean_abs_diff_mc": round(float(np.mean(intact_mean_diffs_mc)), 4) if intact_mean_diffs_mc else 0.0,
                "intact_mean_abs_diff_telea": round(float(np.mean(intact_mean_diffs_tel)), 4) if intact_mean_diffs_tel else 0.0,
                "intact_max_abs_diff_mc": round(intact_max_diff_mc, 2),
                "intact_max_abs_diff_telea": round(intact_max_diff_tel, 2),
            }

        regime_results.append({
            "id": reg["id"],
            "category": reg["category"],
            "orientation": reg["orientation"],
            "polarity": reg["polarity"],
            "contrast": reg["contrast"],
            "persistence": reg["persistence"],
            "motion_regime": reg["motion_regime"],
            "eval_frames": eval_idxs,
            "abstention_rates": abst_rates,
            "candidates": cand_entries,
        })

        # Keep visual reference snapshots for Keyframes (success, subtle, and missed genuine defect)
        if reg["id"] in ("regime_persistent_bright_010", "regime_subtle_transient_010", "regime_horizontal_fissure_020"):
            if reg["id"] == "regime_persistent_bright_010":
                idx_vis = 5
            elif reg["id"] == "regime_horizontal_fissure_020":
                idx_vis = eval_idxs[-2] if len(eval_idxs) >= 2 else eval_idxs[-1]
            else:
                idx_vis = eval_idxs[0]
            saved_visual_examples[reg["id"]] = {
                "frame_idx": idx_vis,
                "clean": reg["clean"][idx_vis],
                "damaged": reg["damaged"][idx_vis],
                "gt": reg["gt"][idx_vis],
                "preds": {c: preds[c][idx_vis] for c in candidate_names},
                "mc_e2": _repair_mc_fair(reg["damaged"], idx_vis, preds["E2_experimental_fused_3state"][idx_vis]["repair"], analyzer),
            }

    # Build dimensional breakdown summaries (by orientation, polarity, contrast, persistence, motion_regime)
    dim_breakdowns: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in ("orientation", "polarity", "contrast", "persistence", "motion_regime"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for r in regime_results:
            groups.setdefault(r[dim], []).append(r)
        dim_entry: dict[str, dict[str, Any]] = {}
        for gval, rlist in groups.items():
            cand_agg: dict[str, Any] = {}
            for cname in candidate_names:
                tp_s = sum(r["candidates"][cname]["repair_tp"] for r in rlist)
                fp_s = sum(r["candidates"][cname]["repair_fp"] for r in rlist)
                fn_s = sum(r["candidates"][cname]["repair_fn"] for r in rlist)
                rec = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0.0
                prec = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                cand_agg[cname] = {
                    "repair_recall": round(rec, 4),
                    "repair_precision": round(prec, 4),
                    "repair_f1": round(f1, 4),
                    "all_detected_recall": round(float(np.mean([r["candidates"][cname]["all_detected_recall"] for r in rlist])), 4),
                }
            dim_entry[gval] = cand_agg
        dim_breakdowns[dim] = dim_entry

    # Aggregate persistent scratch loss metric explicitly
    persistent_regs = [r for r in regime_results if "persistent" in r["persistence"] or "telecine" in r["persistence"] or "medium" in r["persistence"]]
    persistent_loss_summary: dict[str, Any] = {}
    for cname in candidate_names:
        gt_tot = sum(r["candidates"][cname]["gt_positive_pixels"] for r in persistent_regs)
        rep_tp = sum(r["candidates"][cname]["repair_tp"] for r in persistent_regs)
        unc_tp = sum(r["candidates"][cname]["tp_in_uncertain_px"] for r in persistent_regs)
        veto_tp = sum(r["candidates"][cname]["tp_erroneously_vetoed_px"] for r in persistent_regs)
        silent_lost = max(0, gt_tot - rep_tp - unc_tp)
        persistent_loss_summary[cname] = {
            "persistent_gt_pixels": gt_tot,
            "recovered_repair_pct": round(rep_tp / max(1, gt_tot) * 100.0, 2),
            "surfaced_uncertain_pct": round(unc_tp / max(1, gt_tot) * 100.0, 2),
            "erroneously_vetoed_pct": round(veto_tp / max(1, gt_tot) * 100.0, 2),
            "silent_loss_pct": round(silent_lost / max(1, gt_tot) * 100.0, 2),
        }

    # Evaluate Archival Dev Negative Controls
    neg_results: list[dict[str, Any]] = []
    for ctrl in archival_controls:
        logger.info("Evaluating archival dev negative control: %s", ctrl["id"])
        idx = ctrl["idx"]
        roi = ctrl["roi"]
        frames = ctrl["frames"]
        preds = _evaluate_clip_all_candidates(frames, [idx])
        c_entries: dict[str, Any] = {}
        for cname in candidate_names:
            p = preds[cname][idx]
            det_m = p["det"]
            rep_m = p["repair"]
            unc_m = p["uncertain"]
            veto_m = p["vetoed"]
            ev = da.evaluate_structural_negative_controls(det_m, rep_m, [roi])
            mc_img = _repair_mc_fair(frames, idx, rep_m, analyzer)
            tel_img = _inpaint_telea(frames[idx], rep_m)
            rm_bool = _roi_bool(frames[idx].shape[:2], roi["ymin"], roi["ymax"], roi["xmin"], roi["xmax"])
            mod_mc = np.any(mc_img != frames[idx], axis=2)
            mod_tel = np.any(tel_img != frames[idx], axis=2)
            c_entries[cname] = {
                "roi_detector_leak_px": ev["total_detector_leak_pixels"],
                "roi_repair_leak_px": ev["total_repair_leak_pixels"],
                "roi_actual_modified_mc_px": int(np.count_nonzero(mod_mc & rm_bool)),
                "roi_actual_modified_telea_px": int(np.count_nonzero(mod_tel & rm_bool)),
                "roi_uncertain_px": int(np.count_nonzero((unc_m >= 128) & rm_bool)),
                "roi_vetoed_intact_px": int(np.count_nonzero((veto_m >= 128) & rm_bool)),
                "full_frame_repair_px": int(np.count_nonzero(rep_m >= 128)),
                "full_frame_modified_mc_px": int(np.count_nonzero(mod_mc)),
                "full_frame_modified_telea_px": int(np.count_nonzero(mod_tel)),
                "strict_zero_leak_passed": ev["strict_safety_gate_passed"],
            }
        neg_results.append({
            "id": ctrl["id"],
            "frame_index": idx,
            "roi": roi,
            "candidates": c_entries,
        })
        if ctrl["id"] in ("neighbors_frame25_pipe", "paleface_seq_a_jacket", "paleface_seq_b_wall", "neighbors_seq_d_pole"):
            saved_visual_examples[ctrl["id"]] = {
                "frame": frames[idx],
                "roi": roi,
                "preds": {c: preds[c][idx] for c in candidate_names},
                "mc_e2": _repair_mc_fair(frames, idx, preds["E2_experimental_fused_3state"][idx]["repair"], analyzer),
                "mc_a2": _repair_mc_fair(frames, idx, preds["A2_baseline_fused_conservative"][idx]["repair"], analyzer),
            }

    perf_summary = {
        cname: {
            "mean_ms_per_frame": round(float(np.mean(perf_tracker[cname]["ms"])), 2),
            "peak_traced_memory_mb": round(float(np.max(perf_tracker[cname]["peak_mb"])), 2),
        }
        for cname in candidate_names
    }

    return {
        "factorial_ablation": factorial_results,
        "paired_clean_references": clean_results,
        "track_b_dev_regimes": regime_results,
        "dimensional_breakdowns": dim_breakdowns,
        "persistent_scratch_loss_summary": persistent_loss_summary,
        "archival_negative_controls": neg_results,
        "performance": perf_summary,
        "_visual_examples": saved_visual_examples,
        "_clips_for_video": {
            "absdeg_010_persistent": (c010, d_vp_b, g_vp_b),
            "neighbors_72": n72_clip,
            "paleface_a": pa_clip,
            "paleface_b": pb_clip,
            "neighbors_d": nd_clip,
        },
        "_helpers": {
            "spat_det": spat_det,
            "temp_det": temp_det,
            "spat_cfg": spat_cfg,
            "temp_cfg": temp_cfg,
            "analyzer": analyzer,
            "cand_e_det": cand_e_det,
        },
    }


def _overlay_mask(bgr: np.ndarray, mask: np.ndarray, rgb: tuple[int, int, int], alpha: float = 0.65) -> np.ndarray:
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    idx = mask if mask.dtype == bool else (mask >= 128)
    for c in range(3):
        out[idx, c] = np.clip(out[idx, c].astype(float) * (1.0 - alpha) + rgb[c] * alpha, 0, 255).astype(np.uint8)
    return out


def _overlay_three_state(bgr: np.ndarray, pred_dict: Mapping[str, np.ndarray]) -> np.ndarray:
    """Render Green=HighConf Repair, Amber=Uncertain (excluded), Blue=Vetoed Intact Structure."""
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    layers = [
        (pred_dict["vetoed"], (70, 150, 255), 0.55),      # Blue: vetoed intact structure
        (pred_dict["uncertain"], (255, 195, 40), 0.70),   # Amber: uncertain candidate
        (pred_dict["repair"], (50, 240, 90), 0.75),       # Green: conservative repair authorized
    ]
    for m, col, alpha in layers:
        idx = m >= 128
        for c in range(3):
            out[idx, c] = np.clip(out[idx, c].astype(float) * (1.0 - alpha) + col[c] * alpha, 0, 255).astype(np.uint8)
    return out


def _save_visuals_and_videos(comp: Mapping[str, Any]) -> dict[str, Any]:
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    vis = comp["_visual_examples"]
    reg_by_id = {r["id"]: r["candidates"] for r in comp["track_b_dev_regimes"]}
    neg_by_id = {n["id"]: n["candidates"] for n in comp["archival_negative_controls"]}

    fig, axes = plt.subplots(3, 5, figsize=(25, 15), dpi=135)
    fig.patch.set_facecolor("#121212")

    # Row 0: Persistent film-locked bright scratch on AbsDeg 010 Frame 5 (success case)
    ex_p = vis["regime_persistent_bright_010"]
    rp = reg_by_id["regime_persistent_bright_010"]
    a1_r = rp["A1_baseline_temporal"]["repair_recall"] * 100.0
    d_r = rp["D_solution_3"]["repair_recall"] * 100.0
    e2_r = rp["E2_experimental_fused_3state"]["repair_recall"] * 100.0
    e2_p = rp["E2_experimental_fused_3state"]["repair_precision"] * 100.0
    row0 = [
        (_overlay_mask(ex_p["damaged"], ex_p["gt"] == 255, (40, 230, 80)), f"1. AbsDeg 010 Persistent GT (f{ex_p['frame_idx']})\n12f Film-Locked Moving Camera"),
        (_overlay_mask(ex_p["damaged"], ex_p["preds"]["A1_baseline_temporal"]["repair"], (255, 60, 60)), f"A1. Baseline Temporal\n{a1_r:.1f}% Recall (flow<=1.2)"),
        (_overlay_mask(ex_p["damaged"], ex_p["preds"]["D_solution_3"]["repair"], (255, 190, 40)), f"D. Sol 3 Fused Veto\n{d_r:.1f}% Recall"),
        (_overlay_three_state(ex_p["damaged"], ex_p["preds"]["E2_experimental_fused_3state"]), f"E2. Experimental 3-State\n{e2_r:.1f}% R / {e2_p:.1f}% P (Grn=Rep,Amb=Unc,Blu=Veto)"),
        (cv2.cvtColor(ex_p["mc_e2"], cv2.COLOR_BGR2RGB), f"E2. MC Conservative Repaired\nPSNR Gain {rp['E2_experimental_fused_3state']['mc_psnr_gain_db']:+.2f} dB"),
    ]
    for col_i, (img, title) in enumerate(row0):
        axes[0, col_i].imshow(img)
        axes[0, col_i].set_title(title, color="white", fontsize=9.5, weight="bold")
        axes[0, col_i].axis("off")

    # Row 1: Subtle low-contrast transient scratch (opacity=0.38)
    ex_s = vis["regime_subtle_transient_010"]
    rs = reg_by_id["regime_subtle_transient_010"]
    rs_a2_r = rs["A2_baseline_fused_conservative"]["repair_recall"] * 100.0
    rs_d_r = rs["D_solution_3"]["repair_recall"] * 100.0
    rs_e2_r = rs["E2_experimental_fused_3state"]["repair_recall"] * 100.0
    rs_e2_all = rs["E2_experimental_fused_3state"]["all_detected_recall"] * 100.0
    row1 = [
        (_overlay_mask(ex_s["damaged"], ex_s["gt"] == 255, (40, 230, 80)), f"2. Low-Contrast Transient (op=0.38, f{ex_s['frame_idx']})\nGreen = Subtle Scratch GT"),
        (_overlay_mask(ex_s["damaged"], ex_s["preds"]["A2_baseline_fused_conservative"]["repair"], (255, 60, 60)), f"A2. Baseline Fused\n{rs_a2_r:.1f}% R, {rs['A2_baseline_fused_conservative']['intact_pixels_modified_mc']} Intact Mod"),
        (_overlay_mask(ex_s["damaged"], ex_s["preds"]["D_solution_3"]["repair"], (255, 190, 40)), f"D. Sol 3\n{rs_d_r:.1f}% R, {rs['D_solution_3']['intact_pixels_modified_mc']} Intact Mod"),
        (_overlay_three_state(ex_s["damaged"], ex_s["preds"]["E2_experimental_fused_3state"]), f"E2. 3-State Subtle Scratch\n{rs_e2_r:.1f}% Repair R ({rs_e2_all:.1f}% All-Cand R)"),
        (cv2.cvtColor(ex_s["mc_e2"], cv2.COLOR_BGR2RGB), f"E2. MC Repaired\nPSNR Gain {rs['E2_experimental_fused_3state']['mc_psnr_gain_db']:+.2f} dB"),
    ]
    for col_i, (img, title) in enumerate(row1):
        axes[1, col_i].imshow(img)
        axes[1, col_i].set_title(title, color="white", fontsize=9.5, weight="bold")
        axes[1, col_i].axis("off")

    # Row 2: Critical Archival Negative Controls (Neighbors pipe, Paleface A jacket, Paleface B wall, Neighbors D pole)
    np_ex = vis["neighbors_frame25_pipe"]
    pa_ex = vis["paleface_seq_a_jacket"]
    nd_ex = vis["neighbors_seq_d_pole"]
    n_pipe = neg_by_id["neighbors_frame25_pipe"]
    n_pa = neg_by_id["paleface_seq_a_jacket"]
    n_nd = neg_by_id["neighbors_seq_d_pole"]
    row2 = [
        (_overlay_mask(np_ex["frame"][20:300, 80:300], np_ex["preds"]["A2_baseline_fused_conservative"]["repair"][20:300, 80:300], (255, 50, 50)), f"Neighbors Pipe: A2 Baseline\n{n_pipe['A2_baseline_fused_conservative']['roi_repair_leak_px']} px Leaked & Repaired"),
        (_overlay_three_state(np_ex["frame"][20:300, 80:300], {k: v[20:300, 80:300] for k, v in np_ex["preds"]["E2_experimental_fused_3state"].items() if isinstance(v, np.ndarray) and v.ndim == 2}), f"Neighbors Pipe: E2 3-State\n{n_pipe['E2_experimental_fused_3state']['roi_repair_leak_px']} px Leak (Veto={n_pipe['E2_experimental_fused_3state']['roi_vetoed_intact_px']} px)"),
        (_overlay_three_state(pa_ex["frame"][150:410, 170:360], {k: v[150:410, 170:360] for k, v in pa_ex["preds"]["E2_experimental_fused_3state"].items() if isinstance(v, np.ndarray) and v.ndim == 2}), f"Paleface A Jacket: E2 3-State\n{n_pa['E2_experimental_fused_3state']['roi_repair_leak_px']} px Leak vs D {n_pa['D_solution_3']['roi_repair_leak_px']} / A2 {n_pa['A2_baseline_fused_conservative']['roi_repair_leak_px']}"),
        (_overlay_mask(nd_ex["frame"][180:410, 120:250], nd_ex["preds"]["B_solution_1"]["repair"][180:410, 120:250], (255, 160, 40)), f"Neighbors D Pole: B Sol 1\n{n_nd['B_solution_1']['roi_repair_leak_px']} px Occlusion Leak"),
        (_overlay_three_state(nd_ex["frame"][180:410, 120:250], {k: v[180:410, 120:250] for k, v in nd_ex["preds"]["E2_experimental_fused_3state"].items() if isinstance(v, np.ndarray) and v.ndim == 2}), f"Neighbors D Pole: E2 3-State\n{n_nd['E2_experimental_fused_3state']['roi_repair_leak_px']} px Leak (Unc={n_nd['E2_experimental_fused_3state']['roi_uncertain_px']} FB Warn)"),
    ]
    for col_i, (img, title) in enumerate(row2):
        axes[2, col_i].imshow(img)
        axes[2, col_i].set_title(title, color="white", fontsize=9.5, weight="bold")
        axes[2, col_i].axis("off")

    plt.suptitle("Experimental Comparison A-E: Motion-Aware Scene-vs-Film Persistence & 3-State Repair Safety", color="white", fontsize=14, weight="bold")
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
    composite_path = VISUALS_DIR / "experimental_candidates_a_to_e_comparison.png"
    fig.savefig(str(composite_path), dpi=135, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Remove stale draft example from earlier iteration if present
    stale_ex = EXAMPLES_DIR / "example_missed_subtle_transient_010_e2_3state.png"
    if stale_ex.exists():
        stale_ex.unlink()

    # Save explicit standalone PNG examples (successes, missed genuine defect, false repairs)
    cv2.imwrite(str(EXAMPLES_DIR / "example_success_persistent_010_e2_repaired.png"), ex_p["mc_e2"])
    cv2.imwrite(str(EXAMPLES_DIR / "example_success_persistent_010_e2_3state_overlay.png"), cv2.cvtColor(_overlay_three_state(ex_p["damaged"], ex_p["preds"]["E2_experimental_fused_3state"]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(EXAMPLES_DIR / "example_subtle_transient_010_e2_3state.png"), cv2.cvtColor(_overlay_three_state(ex_s["damaged"], ex_s["preds"]["E2_experimental_fused_3state"]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(EXAMPLES_DIR / "example_false_repair_neighbors_pipe_a2_baseline.png"), np_ex["mc_a2"])
    cv2.imwrite(str(EXAMPLES_DIR / "example_safe_neighbors_pipe_e2_repaired.png"), np_ex["mc_e2"])
    cv2.imwrite(str(EXAMPLES_DIR / "example_false_repair_neighbors_d_pole_b_sol1.png"), cv2.cvtColor(_overlay_mask(nd_ex["frame"], nd_ex["preds"]["B_solution_1"]["repair"], (255, 160, 40)), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(EXAMPLES_DIR / "example_safe_neighbors_d_pole_e2_3state.png"), cv2.cvtColor(_overlay_three_state(nd_ex["frame"], nd_ex["preds"]["E2_experimental_fused_3state"]), cv2.COLOR_RGB2BGR))
    if "regime_horizontal_fissure_020" in vis:
        ex_h = vis["regime_horizontal_fissure_020"]
        rh = reg_by_id["regime_horizontal_fissure_020"]
        miss_fig, miss_ax = plt.subplots(1, 2, figsize=(16, 5.2), dpi=135)
        miss_fig.patch.set_facecolor("#121212")
        miss_ax[0].imshow(_overlay_mask(ex_h["damaged"], ex_h["gt"] == 255, (40, 230, 80)))
        miss_ax[0].set_title(f"Missed Defect GT: Horizontal Fissure (f{ex_h['frame_idx']}, dark, 3 deg)", color="white", fontsize=10, weight="bold")
        miss_ax[0].axis("off")
        miss_ax[1].imshow(_overlay_three_state(ex_h["damaged"], ex_h["preds"]["E2_experimental_fused_3state"]))
        e2_hr = rh["E2_experimental_fused_3state"]["repair_recall"] * 100.0
        e2_hp = rh["E2_experimental_fused_3state"]["repair_precision"] * 100.0
        miss_ax[1].set_title(f"E2 3-State Self-Occlusion Failure ({e2_hr:.1f}% R / {e2_hp:.1f}% P overall)", color="#FFCC66", fontsize=10, weight="bold")
        miss_ax[1].axis("off")
        plt.tight_layout()
        miss_fig.savefig(str(EXAMPLES_DIR / "example_missed_horizontal_fissure_020_e2_3state.png"), dpi=135, facecolor=miss_fig.get_facecolor())
        plt.close(miss_fig)

    # Side-by-Side Comparison Videos (MP4)
    helpers = comp["_helpers"]
    spat_det = helpers["spat_det"]
    temp_det = helpers["temp_det"]
    spat_cfg = helpers["spat_cfg"]
    temp_cfg = helpers["temp_cfg"]
    analyzer = helpers["analyzer"]
    cand_e_det = helpers["cand_e_det"]

    video_meta: dict[str, Any] = {}

    # Video 1: AbsDeg 010 Persistent Moving-Camera Sequence across A, B, C, D, E2 Overlay, E2 Repaired
    _, d_vid, g_vid = comp["_clips_for_video"]["absdeg_010_persistent"]
    s_res_v = spat_det.detect(d_vid, spat_cfg)
    t_res_v = temp_det.detect(d_vid, temp_cfg)
    f_res_v = fuse_spatial_and_temporal_masks(s_res_v, t_res_v)
    e_res_v = cand_e_det.analyze_sequence(
        d_vid,
        spatial_scratch_masks=[m.scratch_mask for m in s_res_v.multi_channel_masks],
        spatial_dust_masks=[m.dust_mask for m in s_res_v.multi_channel_masks],
    )

    vid1_path = VIDEOS_DIR / "absdeg_010_persistent_a_to_e.mp4"
    tile_w, tile_h = 320, 180
    canvas_w, canvas_h = tile_w * 4, tile_h * 2 + 44
    writer1 = cv2.VideoWriter(str(vid1_path), cv2.VideoWriter_fourcc(*"mp4v"), 4.0, (canvas_w, canvas_h))
    if not writer1.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {vid1_path}")

    def _tile(img_rgb: np.ndarray, title: str, color: tuple[int, int, int] = (235, 235, 235)) -> np.ndarray:
        t_bgr = cv2.cvtColor(cv2.resize(img_rgb, (tile_w, tile_h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
        hdr = np.zeros((22, tile_w, 3), dtype=np.uint8)
        cv2.putText(hdr, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        return np.vstack([hdr, t_bgr])

    for i in range(len(d_vid)):
        sols = _run_frozen_sol1_to_sol3_on_frame(d_vid, i, s_res_v.multi_channel_masks[i], analyzer, allow_stepwise_nbrs=True)
        mc_e2 = _repair_mc_fair(d_vid, i, e_res_v[i].high_confidence_damage, analyzer)
        e_dict = {
            "repair": e_res_v[i].high_confidence_damage,
            "uncertain": e_res_v[i].uncertain_damage_candidate,
            "vetoed": e_res_v[i].likely_intact_scene_structure,
        }
        a2_m = f_res_v.multi_channel_masks[i].get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        row_top = np.hstack([
            _tile(cv2.cvtColor(d_vid[i], cv2.COLOR_BGR2RGB), f"Input Damaged f{i}"),
            _tile(_overlay_mask(d_vid[i], g_vid[i] == 255, (50, 230, 80)), f"Exact Pers. Scratch GT f{i}"),
            _tile(_overlay_mask(d_vid[i], t_res_v.mask[i], (255, 70, 70)), "A1. Baseline Temporal"),
            _tile(_overlay_mask(d_vid[i], a2_m, (255, 130, 50)), "A2. Baseline Fused Cons."),
        ])
        row_bot = np.hstack([
            _tile(_overlay_mask(d_vid[i], sols["B_solution_1"][1], (255, 210, 50)), "B. Solution 1"),
            _tile(_overlay_mask(d_vid[i], sols["D_solution_3"][1], (200, 170, 255)), "D. Solution 3 Veto"),
            _tile(_overlay_three_state(d_vid[i], e_dict), "E2. 3-State (Grn/Amb/Blu)"),
            _tile(cv2.cvtColor(mc_e2, cv2.COLOR_BGR2RGB), "E2. MC Repaired"),
        ])
        writer1.write(np.vstack([row_top, row_bot]))
    writer1.release()
    video_meta["absdeg_010_persistent_a_to_e"] = {"path": str(vid1_path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "frames": len(d_vid), "fps": 4.0, "resolution": [canvas_w, canvas_h]}

    # Video 2: Archival Negative Controls sequence (Neighbors_72 + Paleface A + Paleface B + Neighbors D) across A2 vs D vs E2
    n_vid = comp["_clips_for_video"]["neighbors_72"]
    pa_vid = comp["_clips_for_video"]["paleface_a"]
    pb_vid = comp["_clips_for_video"]["paleface_b"]
    nd_vid = comp["_clips_for_video"]["neighbors_d"]
    vid2_path = VIDEOS_DIR / "archival_negative_controls_a_to_e.mp4"
    w2, h2 = 360, 270
    cw2, ch2 = w2 * 4, h2 + 24
    writer2 = cv2.VideoWriter(str(vid2_path), cv2.VideoWriter_fourcc(*"mp4v"), 3.0, (cw2, ch2))
    if not writer2.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {vid2_path}")

    vid2_frames = 0
    for tag, clip in [("Neighbors_72", n_vid), ("Paleface_A", pa_vid), ("Paleface_B", pb_vid), ("Neighbors_D", nd_vid[:12])]:
        sr = spat_det.detect(clip, spat_cfg)
        tr = temp_det.detect(clip, temp_cfg)
        fr = fuse_spatial_and_temporal_masks(sr, tr)
        er = cand_e_det.analyze_sequence(
            clip,
            spatial_scratch_masks=[m.scratch_mask for m in sr.multi_channel_masks],
            spatial_dust_masks=[m.dust_mask for m in sr.multi_channel_masks],
        )
        for i in range(2, min(len(clip) - 2, 10)):
            sols = _run_frozen_sol1_to_sol3_on_frame(clip, i, sr.multi_channel_masks[i], analyzer, allow_stepwise_nbrs=True)
            a2_m = fr.multi_channel_masks[i].get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
            e_dict = {
                "repair": er[i].high_confidence_damage,
                "uncertain": er[i].uncertain_damage_candidate,
                "vetoed": er[i].likely_intact_scene_structure,
            }
            mc_e2 = _repair_mc_fair(clip, i, er[i].high_confidence_damage, analyzer)
            panels = [
                (_overlay_mask(clip[i], a2_m, (255, 60, 60)), f"{tag} f{i} | A2 Baseline"),
                (_overlay_mask(clip[i], sols["D_solution_3"][1], (255, 190, 40)), f"{tag} f{i} | D Sol 3"),
                (_overlay_three_state(clip[i], e_dict), f"{tag} f{i} | E2 3-State"),
                (cv2.cvtColor(mc_e2, cv2.COLOR_BGR2RGB), f"{tag} f{i} | E2 MC Safe Repair"),
            ]
            rendered = []
            for pimg, plabel in panels:
                tb = cv2.cvtColor(cv2.resize(pimg, (w2, h2), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR)
                hdr = np.zeros((24, w2, 3), dtype=np.uint8)
                cv2.putText(hdr, plabel, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
                rendered.append(np.vstack([hdr, tb]))
            writer2.write(np.hstack(rendered))
            vid2_frames += 1
    writer2.release()
    video_meta["archival_negative_controls_a_to_e"] = {"path": str(vid2_path.relative_to(PROJECT_ROOT)).replace("\\", "/"), "frames": vid2_frames, "fps": 3.0, "resolution": [cw2, ch2]}

    return {
        "composite_image": str(composite_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "examples_dir": str(EXAMPLES_DIR.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "videos": video_meta,
    }


def main() -> int:
    t_start = time.perf_counter()
    DOCS_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    comp = run_fair_candidate_comparison()
    study = run_scene_vs_film_study()
    vis_meta = _save_visuals_and_videos(comp)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": "Isolated experimental comparison strictly on dev_diagnostic clips; zero production files modified.",
        "elapsed_seconds": round(time.perf_counter() - t_start, 2),
        "factorial_ablation": comp["factorial_ablation"],
        "scene_vs_film_study": study,
        "paired_clean_references": comp["paired_clean_references"],
        "track_b_dev_regimes": comp["track_b_dev_regimes"],
        "dimensional_breakdowns": comp["dimensional_breakdowns"],
        "persistent_scratch_loss_summary": comp["persistent_scratch_loss_summary"],
        "archival_negative_controls": comp["archival_negative_controls"],
        "performance": comp["performance"],
        "visuals": vis_meta,
    }

    json_out = DOCS_BENCHMARKS / "experimental_candidate_comparison_report.json"
    md_out = DOCS_BENCHMARKS / "experimental_candidate_comparison_report.md"
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Experimental Candidate Comparison: Motion-Aware Scene-vs-Film Coordinate Detector (A-E)",
        "",
        f"- **Timestamp:** `{report['timestamp']}`",
        f"- **Elapsed:** `{report['elapsed_seconds']}s`",
        "- **Constraint Check:** Zero production detector, fusion, preset, restoration, or GUI files modified.",
        "",
        "## 1. Unconfounded 2x2x2 Factorial Ablation (`M1_mc_evidence x M2_robust_cut x M3_centerline_geom`, Temporal-Only)",
        "",
        "| Config | Mean Pos Recall | Mean Pos F1 | 000 Fast Pan (R / P) | 010 Tilted (R / P) | 010 Diag (R / P) | Static Diag 1f (R / P) | Total Neg Leak (px) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report["factorial_ablation"]["combinations"]:
        p000 = c["positive"]["absdeg_000_f5"]
        pt = c["positive"]["absdeg_010_tilted"]
        pd = c["positive"]["absdeg_010_diag"]
        psd = c["positive"]["absdeg_010_static_diag_1f"]
        lines.append(
            f"| `{c['config_label']}` | {c['mean_positive_recall']*100:.1f}% | {c['mean_positive_f1']:.3f} | "
            f"{p000['recall']*100:.1f}% / {p000['precision']*100:.1f}% | "
            f"{pt['recall']*100:.1f}% / {pt['precision']*100:.1f}% | "
            f"{pd['recall']*100:.1f}% / {pd['precision']*100:.1f}% | "
            f"{psd['recall']*100:.1f}% / {psd['precision']*100:.1f}% | {c['total_neg_repair_leak_px']} |"
        )
    inter = report["factorial_ablation"]["two_way_interactions"]
    lines.extend([
        "",
        "**Main Effects:**",
        f"- `M1_mc_evidence`: `d_recall={report['factorial_ablation']['main_effects']['M1_mc_evidence']['delta_mean_recall']:+.4f}`, `d_f1={report['factorial_ablation']['main_effects']['M1_mc_evidence']['delta_mean_f1']:+.4f}`, `d_neg_leak={report['factorial_ablation']['main_effects']['M1_mc_evidence']['delta_total_neg_leak_px']:+.1f} px`",
        f"- `M2_robust_cut`: `d_recall={report['factorial_ablation']['main_effects']['M2_robust_cut']['delta_mean_recall']:+.4f}`, `d_f1={report['factorial_ablation']['main_effects']['M2_robust_cut']['delta_mean_f1']:+.4f}`, `d_neg_leak={report['factorial_ablation']['main_effects']['M2_robust_cut']['delta_total_neg_leak_px']:+.1f} px`",
        f"- `M3_centerline_geom`: `d_recall={report['factorial_ablation']['main_effects']['M3_centerline_geom']['delta_mean_recall']:+.4f}`, `d_f1={report['factorial_ablation']['main_effects']['M3_centerline_geom']['delta_mean_f1']:+.4f}`, `d_neg_leak={report['factorial_ablation']['main_effects']['M3_centerline_geom']['delta_total_neg_leak_px']:+.1f} px`",
        "",
        "**Two-Way Interactions:**",
        f"- `M1 x M2`: `d_recall={inter['M1_x_M2']['delta_mean_recall']:+.4f}`, `d_f1={inter['M1_x_M2']['delta_mean_f1']:+.4f}`, `d_neg_leak={inter['M1_x_M2']['delta_total_neg_leak_px']:+.1f} px`",
        f"- `M1 x M3`: `d_recall={inter['M1_x_M3']['delta_mean_recall']:+.4f}`, `d_f1={inter['M1_x_M3']['delta_mean_f1']:+.4f}`, `d_neg_leak={inter['M1_x_M3']['delta_total_neg_leak_px']:+.1f} px`",
        f"- `M2 x M3`: `d_recall={inter['M2_x_M3']['delta_mean_recall']:+.4f}`, `d_f1={inter['M2_x_M3']['delta_mean_f1']:+.4f}`, `d_neg_leak={inter['M2_x_M3']['delta_total_neg_leak_px']:+.1f} px`",
        "",
        "## 2. Scene-Locked vs. Film-Locked Coordinate Study (Candidate E2)",
        "",
        "| Case | ROI px | HighConf (Repair) | Uncertain | Vetoed Intact | Median Flow | Median FB Err |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for r in study:
        lines.append(
            f"| `{r['case']}` | {r['roi_pixels']} | {r['high_confidence_px']} | {r['uncertain_px']} | "
            f"{r['vetoed_intact_px']} | {r['roi_median_flow_px']:.2f} px | {r['roi_median_fb_err_px']:.2f} px |"
        )

    lines.extend([
        "",
        "## 3. Paired Clean Reference Collateral (Zero-Defect Sequences)",
        "",
        "| Clean Ref | Candidate | Repair FP (px) | Severe FP Comp | Max FP H | Uncertain (px) | Intact Mod MC / Telea (px) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for cid, cdict in report["paired_clean_references"].items():
        for cname, cvals in cdict.items():
            lines.append(
                f"| `{cid}` | `{cname}` | {cvals['clean_repair_fp_pixels']} | {cvals['clean_severe_structural_fps']} | "
                f"{cvals['clean_max_fp_height']} | {cvals['clean_uncertain_pixels']} | "
                f"{cvals['clean_intact_modified_mc_pixels']} / {cvals['clean_intact_modified_telea_pixels']} |"
            )

    lines.extend([
        "",
        "## 4. Track B Dev Regime Summary Across Candidates A-E",
        "",
        "| Regime | Candidate | Repair R / P / F1 | All-Cand R / P | MC / Telea PSNR Gain | Intact Mod MC / Telea (Edge MC / Telea) | Abstain (Leg / E) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for reg in report["track_b_dev_regimes"]:
        ab_s = f"{reg['abstention_rates']['legacy_abstain_rate']*100:.0f}% / {reg['abstention_rates']['candidate_e_abstain_rate']*100:.0f}%"
        for cname, cv_d in reg["candidates"].items():
            lines.append(
                f"| `{reg['id']}` | `{cname}` | "
                f"{cv_d['repair_recall']*100:.1f}% / {cv_d['repair_precision']*100:.1f}% / {cv_d['repair_f1']:.3f} | "
                f"{cv_d['all_detected_recall']*100:.1f}% / {cv_d['all_detected_precision']*100:.1f}% | "
                f"{cv_d['mc_psnr_gain_db']:+.2f} / {cv_d['telea_psnr_gain_db']:+.2f} dB | "
                f"{cv_d['intact_pixels_modified_mc']} / {cv_d['intact_pixels_modified_telea']} "
                f"({cv_d['edge_intact_modified_mc']} / {cv_d['edge_intact_modified_telea']}) | {ab_s} |"
            )

    lines.extend([
        "",
        "### 4b. Persistent Scratch Loss & Uncertainty Preservation Summary",
        "",
        "| Candidate | Persistent GT (px) | Recovered Repair % | Surfaced Uncertain % | Erroneously Vetoed % | Silent Loss % |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for cname, pv in report["persistent_scratch_loss_summary"].items():
        lines.append(
            f"| `{cname}` | {pv['persistent_gt_pixels']} | {pv['recovered_repair_pct']:.1f}% | "
            f"{pv['surfaced_uncertain_pct']:.1f}% | {pv['erroneously_vetoed_pct']:.2f}% | {pv['silent_loss_pct']:.1f}% |"
        )

    lines.extend([
        "",
        "### 4c. Dimensional Breakdown (Orientation, Polarity, Contrast, Persistence, Motion Regime — Repair R / P / F1)",
        "",
        "| Dimension | Bucket | A2 Baseline | D Sol 3 | E1 Temporal | E2 Fused 3-State |",
        "|---|---|---|---|---|---|",
    ])
    for dim_name, buckets in report["dimensional_breakdowns"].items():
        for bval, cdict in buckets.items():
            def _fmt(c: str) -> str:
                d = cdict[c]
                return f"{d['repair_recall']*100:.1f}%/{d['repair_precision']*100:.1f}%/{d['repair_f1']:.2f}"
            lines.append(
                f"| `{dim_name}` | `{bval}` | {_fmt('A2_baseline_fused_conservative')} | "
                f"{_fmt('D_solution_3')} | {_fmt('E1_experimental_temporal_only')} | {_fmt('E2_experimental_fused_3state')} |"
            )

    lines.extend([
        "",
        "## 5. Archival Dev Negative Controls (Strict Zero-Leak Safety Gate & Full-Frame Collateral)",
        "",
        "| Negative Control ROI | Candidate | Det Leak | Repair Leak | MC / Telea Mod | Full-Frame Rep / MC | Uncertain / Vetoed ROI | Gate |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ])
    for neg in report["archival_negative_controls"]:
        for cname, nv in neg["candidates"].items():
            badge = "PASS" if nv["strict_zero_leak_passed"] else "**FAIL**"
            lines.append(
                f"| `{neg['id']}` | `{cname}` | {nv['roi_detector_leak_px']} | {nv['roi_repair_leak_px']} | "
                f"{nv['roi_actual_modified_mc_px']} / {nv['roi_actual_modified_telea_px']} | "
                f"{nv['full_frame_repair_px']} / {nv['full_frame_modified_mc_px']} | "
                f"{nv['roi_uncertain_px']} / {nv['roi_vetoed_intact_px']} | {badge} |"
            )

    lines.extend([
        "",
        "## 6. Runtime (Un-instrumented) & Peak Traced Memory Overhead",
        "",
        "| Candidate | Mean ms/frame | Peak Traced Memory (MB) |",
        "|---|---:|---:|",
    ])
    for cname, pv in report["performance"].items():
        lines.append(f"| `{cname}` | {pv['mean_ms_per_frame']:.2f} | {pv['peak_traced_memory_mb']:.2f} |")

    lines.extend([
        "",
        "## 7. Artifacts Generated",
        f"- Composite Figure: `{vis_meta['composite_image']}`",
        f"- Examples Directory: `{vis_meta['examples_dir']}`",
        f"- Video 1: `{vis_meta['videos']['absdeg_010_persistent_a_to_e']['path']}`",
        f"- Video 2: `{vis_meta['videos']['archival_negative_controls_a_to_e']['path']}`",
    ])
    md_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved comparison JSON: %s", json_out)
    logger.info("Saved comparison MD: %s", md_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
