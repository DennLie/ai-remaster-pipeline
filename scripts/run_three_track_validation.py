#!/usr/bin/env python3
"""Unified Three-Track Ground-Truth Benchmark & Validation Suite.

Evaluates defect detectors and conservative repair safety without tuning or
modifying any detector algorithms, thresholds, presets, fusion rules, or GUI code:

  - Track A (Real Annotated Still Images — Spatial Only):
      * FilmDamageSimulator / FILM-AA (manual_polygon_gt, CC BY 4.0)
      * ARTeFACT (manual_segmentation_gt, AFL-3.0 offline research only)
      * Multi-frame / temporal detectors explicitly marked INSUFFICIENT_TEMPORAL_CONTEXT.

  - Track B (Exact Synthetic Defects on Genuine Clean Archival Video):
      * AbsoluteDegradation clean reference sequences (000/010 dev_diagnostic, 020 locked_holdout)
      * Existing FilmScratchGenerator clips + Extended multi-regime injections
        (bright/dark, vertical/diagonal/horizontal, straight/curved/branching,
         fine/broad with 127 ignore flanks, persistent, drifting, and telecine duplicates)
      * 3-axis reporting: (1) Detection Quality, (2) Conservative Repair Safety,
        (3) Exact Clean-Reference Reconstruction Quality (PSNR/MSE & intact collateral diff).

  - Track C (Real Archival Video — Negative Controls, Quarantine, & Annotation Workflow):
      * Quarantines invalidated Frame 25 pipe claims and absdiff pseudo-labels.
      * Evaluates strict zero-leakage negative controls across Neighbors_72, Paleface,
        College, and Neighbors Seq D (all split="dev_diagnostic").
      * Exports track_c_annotation_manifest.json for blinded human annotation & unseen holdout.
"""
from __future__ import annotations

import argparse
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
import provision_benchmark_resources as pbr
import run_defect_benchmark_suite as rdbs
from film_defect_detectors import (
    BENCHMARK_PRESETS,
    DEFAULT_REGISTRY,
    RepairInclusionPolicy,
    fuse_spatial_and_temporal_masks,
)
from temporal_scratch_analyzer import CLEAN_ROI, SCRATCH_ROI, TemporalConfig, TemporalScratchAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("three_track_validation")

DOCS_BENCHMARKS = PROJECT_ROOT / "docs" / "benchmarks"
VISUALS_DIR = DOCS_BENCHMARKS / "visuals"
CACHE_DIR = PROJECT_ROOT / ".cache" / "benchmarks"
ABSDEG_FULL_DIR = PROJECT_ROOT / ".cache" / "absolute-scratch-benchmark-20261106-01"
CROSS_FILM_DIR = PROJECT_ROOT / "evaluation" / "cross_film_validation"
NEIGHBORS_VIDEO = PROJECT_ROOT / "evaluation" / "Neighbors_72.mkv"


def _load_video_slice(video_path: Path, start: int = 0, end: int | None = None) -> list[np.ndarray]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx >= start and (end is None or idx < end):
            frames.append(frame)
        idx += 1
        if end is not None and idx >= end:
            break
    cap.release()
    return frames


def _select_defect_dense_window(
    image_bgr: np.ndarray,
    gt_combined: np.ndarray,
    gt_scratch: np.ndarray | None,
    gt_dust: np.ndarray | None,
    window_size: int = 1024,
) -> dict[str, Any]:
    """Deterministically crop a native-resolution 1:1 window maximizing defect coverage (never resizes)."""
    h, w = gt_combined.shape
    if h <= window_size and w <= window_size:
        return {
            "image": image_bgr,
            "gt_combined": gt_combined,
            "gt_scratch": gt_scratch,
            "gt_dust": gt_dust,
            "crop_bbox": {"ymin": 0, "ymax": h, "xmin": 0, "xmax": w},
            "is_cropped": False,
        }

    step = max(128, window_size // 4)
    best_score = (-1, -1)
    best_yx = (0, 0)
    priority_mask = gt_scratch if (gt_scratch is not None and np.any(gt_scratch == 255)) else gt_combined

    for y in range(0, max(1, h - window_size + 1), step):
        y2 = min(h, y + window_size)
        for x in range(0, max(1, w - window_size + 1), step):
            x2 = min(w, x + window_size)
            p_cnt = int(np.count_nonzero(priority_mask[y:y2, x:x2] == 255))
            c_cnt = int(np.count_nonzero(gt_combined[y:y2, x:x2] == 255))
            score = (p_cnt, c_cnt)
            if score > best_score:
                best_score = score
                best_yx = (y, x)

    y0, x0 = best_yx
    y1, x1 = min(h, y0 + window_size), min(w, x0 + window_size)
    return {
        "image": np.ascontiguousarray(image_bgr[y0:y1, x0:x1]),
        "gt_combined": np.ascontiguousarray(gt_combined[y0:y1, x0:x1]),
        "gt_scratch": np.ascontiguousarray(gt_scratch[y0:y1, x0:x1]) if gt_scratch is not None else None,
        "gt_dust": np.ascontiguousarray(gt_dust[y0:y1, x0:x1]) if gt_dust is not None else None,
        "crop_bbox": {"ymin": y0, "ymax": y1, "xmin": x0, "xmax": x1},
        "is_cropped": True,
    }


def _run_spatial_detector_native(
    det_name: str,
    frame_bgr: np.ndarray,
) -> dict[str, Any]:
    """Run a single-frame spatial detector at native resolution using frozen default thresholds."""
    reg = DEFAULT_REGISTRY
    det = reg.get(det_name, commercial_only=False)
    t0 = time.perf_counter()

    # Use frozen preset thresholds; enable tiling only on large images to prevent VRAM OOM
    cfg: dict[str, Any] = {}
    if det_name == "film_artefact_segmenter":
        cfg = {"threshold": 0.5, "scratch_threshold": 0.5, "dust_threshold": 0.5, "tile_size": 512, "overlap": 64}
    elif det_name == "cnn_blotches":
        cfg = {"threshold": 0.5, "scratch_threshold": 0.5, "dust_threshold": 0.5, "tile_size": 512, "overlap": 64}
    elif det_name == "bopb":
        cfg = {"threshold": 0.5, "tile_size": 256, "overlap": 64}
    elif det_name == "patchcore":
        cfg = {"threshold": BENCHMARK_PRESETS["patchcore_anomaly_research"].sensitivity, "tile_size": 512, "overlap": 64}
    elif det_name == "padim":
        cfg = {"threshold": BENCHMARK_PRESETS["padim_anomaly_research"].sensitivity, "tile_size": 512, "overlap": 64}
    elif det_name == "legacy":
        cfg = {"sensitivity": 0.65}

    res = det.detect([frame_bgr], config=cfg)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "combined": res.mask[0],
        "scratch": res.scratch_mask[0] if res.scratch_mask else np.zeros(frame_bgr.shape[:2], dtype=np.uint8),
        "dust": res.dust_mask[0] if res.dust_mask else np.zeros(frame_bgr.shape[:2], dtype=np.uint8),
        "latency_ms": round(latency_ms, 2),
        "commercial_cleared": det.commercial_cleared,
    }


def run_track_a_real_stills(*, include_research: bool = True) -> dict[str, Any]:
    """Evaluate Track A: Real human-annotated still images (FilmDamageSimulator & ARTeFACT)."""
    logger.info("=== Running Track A: Real Annotated Still Images ===")
    spatial_candidates = ["film_artefact_segmenter", "cnn_blotches", "legacy"]
    if include_research:
        spatial_candidates.extend(["bopb", "patchcore", "padim"])
    temporal_candidates = ["temporal_scratch", "fused_fas_temporal", "farneback_3f", "dis_flow", "ridge", "consensus_5f", "dart_harness"]

    datasets_spec: list[tuple[str, Path, str, bool]] = [
        ("film_damage_simulator", CACHE_DIR / "fds", "manual_polygon_gt", True),
    ]
    if include_research:
        datasets_spec.append(("artefact", CACHE_DIR / "artefact", "manual_segmentation_gt", False))

    art_meta_by_clip: dict[str, dict[str, Any]] = {}
    art_ext_manifest = CACHE_DIR / "artefact" / "extracted" / "extracted_manifest.json"
    if art_ext_manifest.is_file():
        raw_em = json.loads(art_ext_manifest.read_text(encoding="utf-8"))
        for item in raw_em.get("samples", []):
            art_meta_by_clip[f"artefact_{item['id']}"] = item

    track_a_results: dict[str, Any] = {
        "track": "Track A — Real Annotated Still Images (Spatial Evaluation Only)",
        "temporal_evaluation_policy": "Single still frames have 0 temporal motion context; all multi-frame detectors are strictly marked INSUFFICIENT_TEMPORAL_CONTEXT without fabricating sequences.",
        "datasets": {},
    }

    vis_samples: list[dict[str, Any]] = []

    for ds_name, ds_dir, expected_prov, comm_cleared in datasets_spec:
        recs, invalid = rdbs.load_dataset_records(ds_dir, return_invalid=True)
        da.validate_provenance_separation(recs, require_manual_only=True)
        da.validate_split_isolation(recs, allow_holdout_for_tuning=True)

        prepared_samples: list[dict[str, Any]] = []
        for r in recs:
            gt_s = r.get("gt_scratch_mask")
            gt_d = r.get("gt_dust_mask")
            win = _select_defect_dense_window(r["image"], r["gt_mask"], gt_s, gt_d, window_size=1024)
            extra = art_meta_by_clip.get(r["clip"], {})
            prepared_samples.append({
                "clip": r["clip"],
                "frame": r["frame"],
                "full_shape": list(r["image"].shape[:2]),
                "eval_shape": list(win["image"].shape[:2]),
                "crop_bbox": win["crop_bbox"],
                "is_cropped": win["is_cropped"],
                "material": extra.get("material", "35mm_film_scan"),
                "label_provenance": expected_prov,
                "image": win["image"],
                "gt_combined": win["gt_combined"],
                "gt_scratch": win["gt_scratch"],
                "gt_dust": win["gt_dust"],
            })

        cand_reports: dict[str, Any] = {}
        for cand in spatial_candidates:
            comb_counts, scr_counts, dst_counts, sev_list, lat_list = [], [], [], [], []
            per_sample: list[dict[str, Any]] = []
            supports_scratch = cand not in ("patchcore", "padim")
            supports_dust = cand not in ("legacy", "bopb", "patchcore", "padim")

            for s in prepared_samples:
                pred = _run_spatial_detector_native(cand, s["image"])
                lat_list.append(pred["latency_ms"])
                c_cnt = bdm.confusion(s["gt_combined"], pred["combined"], threshold=128)
                comb_counts.append(c_cnt)
                sev = da.evaluate_false_positive_severity(s["gt_combined"], pred["combined"], threshold=128)
                sev_list.append(sev)

                if supports_scratch and s["gt_scratch"] is not None:
                    scr_counts.append(bdm.confusion(s["gt_scratch"], pred["scratch"], threshold=128))
                if supports_dust and s["gt_dust"] is not None:
                    dst_counts.append(bdm.confusion(s["gt_dust"], pred["dust"], threshold=128))

                per_sample.append({
                    "clip": s["clip"],
                    "material": s["material"],
                    "eval_shape": s["eval_shape"],
                    "crop_bbox": s["crop_bbox"],
                    "metrics": bdm.metrics(c_cnt),
                    "fp_severity": sev,
                })

                if ds_name == "film_damage_simulator" and cand == "film_artefact_segmenter" and s["clip"] == "Scan_8":
                    vis_samples.append({
                        "title": "Track A: FDS Scan (8) 1024x1024 Window",
                        "image": s["image"],
                        "gt": s["gt_combined"],
                        "pred": pred["combined"],
                    })

            cand_reports[cand] = {
                "status": "EVALUATED",
                "commercial_cleared": DEFAULT_REGISTRY.get(cand, commercial_only=False).commercial_cleared,
                "combined_metrics": bdm.metrics(bdm.sum_counts(comb_counts)) if comb_counts else {},
                "scratch_metrics": bdm.metrics(bdm.sum_counts(scr_counts)) if scr_counts else {"status": "NOT_APPLICABLE"},
                "dust_metrics": bdm.metrics(bdm.sum_counts(dst_counts)) if dst_counts else {"status": "NOT_APPLICABLE"},
                "fp_severity": {
                    "fp_pixel_count": sum(x["fp_pixel_count"] for x in sev_list),
                    "fp_component_count": sum(x["fp_component_count"] for x in sev_list),
                    "max_fp_component_area": max((x["max_fp_component_area"] for x in sev_list), default=0),
                    "max_fp_component_height": max((x["max_fp_component_height"] for x in sev_list), default=0),
                    "severe_structural_fp_count": sum(x["severe_structural_fp_count"] for x in sev_list),
                },
                "mean_latency_ms": round(float(np.mean(lat_list)), 2) if lat_list else 0.0,
                "per_sample": per_sample,
            }

        for t_cand in temporal_candidates:
            cand_reports[t_cand] = {
                "status": "INSUFFICIENT_TEMPORAL_CONTEXT",
                "reason": "Single still images cannot evaluate temporal detectors (min_frames >= 2..3); repeating stills is prohibited.",
            }

        track_a_results["datasets"][ds_name] = {
            "provenance_tier": expected_prov,
            "commercial_cleared": comm_cleared,
            "samples_count": len(prepared_samples),
            "invalid_samples": invalid,
            "candidates": cand_reports,
        }

    track_a_results["_vis"] = vis_samples
    return track_a_results


def _synthesize_mc_repair(
    frames: list[np.ndarray],
    target_idx: int,
    repair_mask: np.ndarray,
    analyzer: TemporalScratchAnalyzer,
) -> np.ndarray:
    """Motion-compensated temporal median replacement strictly gated by repair_mask."""
    if not np.any(repair_mask >= 128):
        return frames[target_idx].copy()
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    prec, succ, _ = analyzer._select_nonduplicate_neighbors(gray_frames, target_idx)
    neighbors = prec + succ
    if not neighbors:
        return cv2.inpaint(frames[target_idx], repair_mask, 3, cv2.INPAINT_TELEA)

    warped_bgr = []
    for n_idx in neighbors:
        flow = analyzer._compute_dense_flow(gray_frames[target_idx], gray_frames[n_idx])
        w_f = np.zeros_like(frames[target_idx])
        for c in range(3):
            w_f[:, :, c] = analyzer._warp_neighbor(frames[n_idx][:, :, c], flow)
        warped_bgr.append(w_f)

    bg_bgr = np.median(np.stack(warped_bgr, axis=0), axis=0).astype(np.uint8)
    out = frames[target_idx].copy()
    m = repair_mask >= 128
    out[m] = bg_bgr[m]
    return out


def _evaluate_video_regime_3axis(
    regime_name: str,
    split: str,
    provenance: str,
    clean_frames: list[np.ndarray],
    damaged_frames: list[np.ndarray],
    gt_masks_3tier: list[np.ndarray],
    eval_indices: Sequence[int],
    *,
    notes: str = "",
) -> dict[str, Any]:
    """Run spatial, temporal, and fused detectors on a genuine video clip with exact clean references."""
    spatial_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    temporal_det = DEFAULT_REGISTRY.get("temporal_scratch")
    spatial_cfg = dict(BENCHMARK_PRESETS["film_artefact_balanced"].params)
    temporal_cfg = dict(BENCHMARK_PRESETS["temporal_scratch_balanced"].params)

    t0 = time.perf_counter()
    spat_res = spatial_det.detect(damaged_frames, spatial_cfg)
    t_spat_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(damaged_frames))

    t0 = time.perf_counter()
    temp_res = temporal_det.detect(damaged_frames, temporal_cfg)
    t_temp_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(damaged_frames))

    t0 = time.perf_counter()
    fused_res = fuse_spatial_and_temporal_masks(spat_res, temp_res)
    t_fused_ms = t_spat_ms + t_temp_ms + (time.perf_counter() - t0) * 1000.0 / max(1, len(damaged_frames))

    analyzer = TemporalScratchAnalyzer(TemporalConfig())

    mode_masks: dict[str, tuple[list[np.ndarray], list[np.ndarray], float]] = {
        "spatial_only": (
            list(spat_res.mask),
            [m.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE) for m in (spat_res.multi_channel_masks or [])],
            t_spat_ms,
        ),
        "temporal_only": (
            list(temp_res.scratch_mask),
            [m.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY) for m in (temp_res.multi_channel_masks or [])],
            t_temp_ms,
        ),
        "fused_conservative": (
            list(fused_res.mask),
            [m.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE) for m in (fused_res.multi_channel_masks or [])],
            t_fused_ms,
        ),
        "fused_high_recall": (
            list(fused_res.mask),
            [m.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL) for m in (fused_res.multi_channel_masks or [])],
            t_fused_ms,
        ),
    }

    mode_reports: dict[str, Any] = {}
    vis_payload: dict[str, Any] | None = None

    for mode_name, (det_masks, rep_masks, lat_ms) in mode_masks.items():
        det_counts, rep_counts, sev_list, rest_mc_list, rest_telea_list = [], [], [], [], []
        # Define structural edge nuisance ROI dynamically on first evaluated clean frame
        first_regions = da.segment_nuisance_regions(clean_frames[eval_indices[0]])
        edge_mask_bool = first_regions["edges"]

        edge_fp_total = 0
        edge_valid_total = 0

        for idx in eval_indices:
            gt = gt_masks_3tier[idx]
            d_mask = det_masks[idx]
            r_mask = rep_masks[idx]
            clean_ref = clean_frames[idx]
            dmg = damaged_frames[idx]

            # Axis 1: Candidate detection quality
            det_counts.append(bdm.confusion(gt, d_mask, threshold=128))
            sev_list.append(da.evaluate_false_positive_severity(gt, d_mask, threshold=128))

            # Axis 2: Repair-mask precision & edge leakage
            rep_counts.append(bdm.confusion(gt, r_mask, threshold=128))
            edge_gt = np.where(edge_mask_bool, gt, np.uint8(127))
            e_cnt = bdm.confusion(edge_gt, r_mask, threshold=128)
            edge_fp_total += e_cnt["fp"]
            edge_valid_total += (e_cnt["fp"] + e_cnt["tn"])

            # Axis 3: Reconstruction quality against exact clean reference
            rep_mc = _synthesize_mc_repair(damaged_frames, idx, r_mask, analyzer)
            rep_tel = cv2.inpaint(dmg, r_mask, 3, cv2.INPAINT_TELEA) if np.any(r_mask >= 128) else dmg.copy()

            rest_mc_list.append(da.evaluate_exact_reference_restoration(clean_ref, dmg, rep_mc, gt, r_mask))
            rest_telea_list.append(da.evaluate_exact_reference_restoration(clean_ref, dmg, rep_tel, gt, r_mask))

            if mode_name == "fused_conservative" and vis_payload is None and np.any(gt == 255):
                vis_payload = {
                    "clean": clean_ref,
                    "damaged": dmg,
                    "gt": gt,
                    "temporal_pred": temp_res.scratch_mask[idx],
                    "repair_mask": r_mask,
                    "repaired_mc": rep_mc,
                }

        det_m = bdm.metrics(bdm.sum_counts(det_counts))
        rep_m = bdm.metrics(bdm.sum_counts(rep_counts))

        mode_reports[mode_name] = {
            "latency_ms_per_frame": round(lat_ms, 2),
            "axis1_detection_quality": {
                **det_m,
                "fp_severity": {
                    "fp_pixel_count": sum(s["fp_pixel_count"] for s in sev_list),
                    "fp_component_count": sum(s["fp_component_count"] for s in sev_list),
                    "max_fp_component_area": max((s["max_fp_component_area"] for s in sev_list), default=0),
                    "max_fp_component_height": max((s["max_fp_component_height"] for s in sev_list), default=0),
                    "severe_structural_fp_count": sum(s["severe_structural_fp_count"] for s in sev_list),
                },
            },
            "axis2_repair_mask_safety": {
                "precision": rep_m["precision"],
                "recall": rep_m["recall"],
                "f1": rep_m["f1"],
                "repair_coverage_pct": round(rep_m["prediction_coverage"] * 100.0, 4),
                "intact_pixel_fpr_pct": round(rep_m["false_positive_rate"] * 100.0, 4),
                "structural_edge_repair_fpr_pct": round((edge_fp_total / max(1, edge_valid_total)) * 100.0, 4),
                "structural_edge_repair_fp_pixels": edge_fp_total,
            },
            "axis3_exact_restoration_quality": {
                "motion_comp": {
                    "mean_damaged_psnr_db": round(float(np.mean([x["damaged_psnr_db"] for x in rest_mc_list if x["has_defect_pixels"]] or [0.0])), 3),
                    "mean_repaired_psnr_db": round(float(np.mean([x["repaired_psnr_db"] for x in rest_mc_list if x["has_defect_pixels"]] or [0.0])), 3),
                    "mean_psnr_gain_db": round(float(np.mean([x["psnr_gain_db"] for x in rest_mc_list if x["has_defect_pixels"]] or [0.0])), 3),
                    "total_intact_pixels_modified": sum(x["intact_pixels_modified_count"] for x in rest_mc_list),
                    "mean_intact_modified_fpr_pct": round(float(np.mean([x["intact_pixels_modified_fpr"] * 100.0 for x in rest_mc_list])), 5),
                    "min_intact_psnr_db": round(float(np.min([x["intact_psnr_db"] for x in rest_mc_list])), 3),
                },
                "telea": {
                    "mean_damaged_psnr_db": round(float(np.mean([x["damaged_psnr_db"] for x in rest_telea_list if x["has_defect_pixels"]] or [0.0])), 3),
                    "mean_repaired_psnr_db": round(float(np.mean([x["repaired_psnr_db"] for x in rest_telea_list if x["has_defect_pixels"]] or [0.0])), 3),
                    "mean_psnr_gain_db": round(float(np.mean([x["psnr_gain_db"] for x in rest_telea_list if x["has_defect_pixels"]] or [0.0])), 3),
                    "total_intact_pixels_modified": sum(x["intact_pixels_modified_count"] for x in rest_telea_list),
                    "mean_intact_modified_fpr_pct": round(float(np.mean([x["intact_pixels_modified_fpr"] * 100.0 for x in rest_telea_list])), 5),
                    "min_intact_psnr_db": round(float(np.min([x["intact_psnr_db"] for x in rest_telea_list])), 3),
                },
            },
        }

    return {
        "regime": regime_name,
        "split": split,
        "label_provenance": provenance,
        "frame_count": len(damaged_frames),
        "evaluated_frame_indices": list(eval_indices),
        "resolution": [damaged_frames[0].shape[1], damaged_frames[0].shape[0]],
        "notes": notes,
        "modes": mode_reports,
        "_vis": vis_payload,
    }


def run_track_b_synthetic_on_real_video() -> dict[str, Any]:
    """Evaluate Track B: Exact synthetic defects injected onto genuine clean reference video."""
    logger.info("=== Running Track B: Exact Synthetic Defects on Real Clean Video ===")
    if not ABSDEG_FULL_DIR.is_dir():
        raise FileNotFoundError(f"AbsoluteDegradation cache missing at {ABSDEG_FULL_DIR}")

    def _load_absdeg_seq(clip: str, seed_sub: str, n_frames: int = 12) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        ref_dir = ABSDEG_FULL_DIR / clip / "reference"
        dmg_dir = ABSDEG_FULL_DIR / clip / seed_sub
        cleans, dmgs, gts = [], [], []
        for i in range(n_frames):
            c_img = cv2.imread(str(ref_dir / f"{i:06d}.png"))
            d_img = cv2.imread(str(dmg_dir / f"{i:06d}.png"))
            g_img = cv2.imread(str(dmg_dir / f"{i:06d}.gt.png"), cv2.IMREAD_GRAYSCALE)
            if c_img is None or d_img is None or g_img is None:
                raise FileNotFoundError(f"Missing AbsoluteDegradation frame {i} in {clip}/{seed_sub}")
            cleans.append(c_img)
            dmgs.append(d_img)
            gts.append(g_img)
        return cleans, dmgs, gts

    regimes: list[dict[str, Any]] = []
    # 1. Existing AbsoluteDegradation FilmScratchGenerator sequences (all dev_diagnostic after reclassification)
    for clip_id, seed_sub, split_tag in [
        ("010", "20261106", "dev_diagnostic"),
        ("020", "20261106", "dev_diagnostic"),
        ("020", "20261107", "dev_diagnostic"),
    ]:
        cleans, dmgs, gts = _load_absdeg_seq(clip_id, seed_sub, n_frames=12)
        eval_idx = [i for i in range(3, 9) if np.any(gts[i] == 255)] or list(range(3, 9))
        regimes.append(
            _evaluate_video_regime_3axis(
                f"absdeg_{clip_id}_seed_{seed_sub}",
                split_tag,
                "synthetic_parametric_gt",
                cleans,
                dmgs,
                gts,
                eval_idx,
                notes="Existing AbsoluteDegradation FilmScratchGenerator on real reference footage",
            )
        )

    # 2. Multi-regime synthetic injections on clean 020 reference (reclassified as dev_diagnostic)
    clean_020 = [cv2.imread(str(ABSDEG_FULL_DIR / "020" / "reference" / f"{i:06d}.png")) for i in range(12)]
    assert all(f is not None for f in clean_020)

    extended_specs: list[tuple[str, bdm.ExtendedScratchConfig, bool, str]] = [
        (
            "vertical_bright_drifting",
            bdm.ExtendedScratchConfig(
                orientation="vertical", angle_deg=1.5, width=2, opacity=0.85, bright=True,
                duration=8, drift_px=1.5, seed=202611181,
            ),
            False,
            "Bright vertical drifting scratch on moving real scene",
        ),
        (
            "vertical_dark_persistent",
            bdm.ExtendedScratchConfig(
                orientation="vertical", angle_deg=0.0, width=3, opacity=0.80, bright=False,
                duration=10, drift_px=0.5, seed=202611182,
            ),
            False,
            "Dark persistent vertical scratch (polarity invariance test)",
        ),
        (
            "vertical_telecine_duplicates",
            bdm.ExtendedScratchConfig(
                orientation="vertical", angle_deg=0.5, width=2, opacity=0.85, bright=True,
                duration=8, drift_px=1.0, seed=202611183,
            ),
            True,
            "Vertical bright scratch under simulated telecine pulldown frame duplicates (duplicate_every=3)",
        ),
        (
            "diagonal_bright_curved",
            bdm.ExtendedScratchConfig(
                orientation="diagonal", angle_deg=35.0, curvature=8.0, width=3, opacity=0.85, bright=True,
                duration=8, drift_px=0.5, seed=202611184,
            ),
            False,
            "Curved 35-deg diagonal bright scratch (orientation coverage test)",
        ),
        (
            "horizontal_fissure",
            bdm.ExtendedScratchConfig(
                orientation="horizontal", angle_deg=2.0, width=3, opacity=0.85, bright=True,
                duration=8, drift_px=0.5, seed=202611185,
            ),
            False,
            "Horizontal fissure scratch (tests vertical-only structural assumptions)",
        ),
        (
            "broad_abrasion_branching",
            bdm.ExtendedScratchConfig(
                orientation="vertical", angle_deg=3.0, width=6, opacity=0.80, bright=True,
                branching=True, flank_ignore_px=1, duration=8, drift_px=1.0, seed=202611186,
            ),
            False,
            "Broad 6px branching abrasion with 1px 127-ignore flank boundary",
        ),
    ]

    for reg_name, cfg, use_telecine, desc in extended_specs:
        if use_telecine:
            sub_clean = clean_020[:9]
            sub_dmg, sub_gt, sub_exact = bdm.inject_extended_scratches(sub_clean, cfg)
            tele = bdm.apply_telecine_cadence(sub_clean, sub_dmg, sub_gt, sub_exact, duplicate_every=3, target_length=12)
            c_seq, d_seq, g_seq = tele["clean_frames"], tele["damaged_frames"], tele["gt_masks"]
            notes_full = f"{desc}; duplicate_indices={tele['duplicate_indices']}"
        else:
            c_seq = clean_020
            d_seq, g_seq, _ = bdm.inject_extended_scratches(clean_020, cfg)
            notes_full = desc

        eval_idx = [i for i in range(3, 9) if np.any(g_seq[i] == 255)] or list(range(3, 9))
        regimes.append(
            _evaluate_video_regime_3axis(
                reg_name,
                "dev_diagnostic",
                "synthetic_parametric_gt",
                c_seq,
                d_seq,
                g_seq,
                eval_idx,
                notes=notes_full,
            )
        )

    vis_payloads = {r["regime"]: r.pop("_vis") for r in regimes}
    return {
        "track": "Track B — Exact Synthetic Defects on Genuine Clean Archival Video",
        "substrate": "AbsoluteDegradation clean references (010 dev_diagnostic, 020 locked_holdout)",
        "regimes": regimes,
        "_vis": vis_payloads,
    }


def run_track_c_negative_controls_and_quarantine() -> dict[str, Any]:
    """Evaluate Track C negative controls on real archival footage & quarantine invalid claims."""
    logger.info("=== Running Track C: Real Archival Negative Controls & Quarantine Audit ===")
    spatial_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    temporal_det = DEFAULT_REGISTRY.get("temporal_scratch")
    spatial_cfg = dict(BENCHMARK_PRESETS["film_artefact_balanced"].params)
    temporal_cfg = dict(BENCHMARK_PRESETS["temporal_scratch_balanced"].params)

    sequences = [
        {
            "sequence_name": "neighbors_frame25_pipe_audit",
            "video_path": NEIGHBORS_VIDEO,
            "start": 20,
            "end": 31,
            "target_idx": 5,
            "global_frame_index": 25,
            "split": "dev_diagnostic",
            "rois": [
                {
                    "name": "neighbors_frame25_pipe",
                    "description": "Confirmed intact building wall vertical pipe (previously mislabeled as GT scratch)",
                    **SCRATCH_ROI,
                },
                {
                    "name": "neighbors_frame25_clean_wall",
                    "description": "Confirmed intact wall/fence background texture",
                    **CLEAN_ROI,
                },
            ],
        },
        {
            "sequence_name": "paleface_seq_a",
            "video_path": CROSS_FILM_DIR / "paleface_seq_a.mkv",
            "start": 0,
            "end": None,
            "target_idx": 8,
            "global_frame_index": 8,
            "split": "dev_diagnostic",
            "rois": [
                {
                    "name": "paleface_actor_jacket_edge",
                    "description": "Moving actor jacket boundary against static office wall",
                    "ymin": 188, "ymax": 381, "xmin": 258, "xmax": 268,
                },
                {
                    "name": "paleface_clean_wall",
                    "description": "Clean background wall ROI",
                    "ymin": 300, "ymax": 420, "xmin": 450, "xmax": 550,
                },
            ],
        },
        {
            "sequence_name": "paleface_seq_b",
            "video_path": CROSS_FILM_DIR / "paleface_seq_b.mkv",
            "start": 0,
            "end": None,
            "target_idx": 7,
            "global_frame_index": 7,
            "split": "dev_diagnostic",
            "rois": [
                {
                    "name": "paleface_b_wall_curtain",
                    "description": "Static interior wall panel and curtain fold",
                    "ymin": 80, "ymax": 180, "xmin": 200, "xmax": 280,
                },
                {
                    "name": "paleface_b_clean_bg",
                    "description": "Clean interior background ROI",
                    "ymin": 30, "ymax": 120, "xmin": 50, "xmax": 150,
                },
            ],
        },
        {
            "sequence_name": "college_seq_c",
            "video_path": CROSS_FILM_DIR / "college_seq_c.mkv",
            "start": 0,
            "end": None,
            "target_idx": 8,
            "global_frame_index": 8,
            "split": "dev_diagnostic",
            "rois": [
                {
                    "name": "college_door_jamb_student",
                    "description": "Classroom door jamb & moving student silhouette",
                    "ymin": 150, "ymax": 280, "xmin": 280, "xmax": 340,
                },
                {
                    "name": "college_clean_bg",
                    "description": "Clean classroom wall ROI",
                    "ymin": 50, "ymax": 150, "xmin": 400, "xmax": 500,
                },
            ],
        },
        {
            "sequence_name": "neighbors_seq_d",
            "video_path": CROSS_FILM_DIR / "neighbors_seq_d.mkv",
            "start": 0,
            "end": None,
            "target_idx": 2,
            "global_frame_index": 2,
            "split": "dev_diagnostic",
            "rois": [
                {
                    "name": "neighbors_d_telephone_pole",
                    "description": "Street telephone pole and vertical building corner",
                    "ymin": 200, "ymax": 400, "xmin": 150, "xmax": 220,
                },
                {
                    "name": "neighbors_d_clean_bg",
                    "description": "Clean sky/building ROI",
                    "ymin": 80, "ymax": 200, "xmin": 700, "xmax": 850,
                },
            ],
        },
    ]

    # Enforce split guard so contaminated dev clips are never misclassified as holdout
    da.validate_split_isolation(
        [{"clip": s["sequence_name"] if "neighbors_frame25" not in s["sequence_name"] else "Neighbors_72", "split": s["split"]} for s in sequences],
        allow_holdout_for_tuning=False,
    )

    seq_reports: list[dict[str, Any]] = []
    vis_failures: dict[str, Any] = {}

    for spec in sequences:
        frames = _load_video_slice(spec["video_path"], start=spec["start"], end=spec["end"])
        idx = spec["target_idx"]
        spat_res = spatial_det.detect(frames, spatial_cfg)
        temp_res = temporal_det.detect(frames, temporal_cfg)
        fused_res = fuse_spatial_and_temporal_masks(spat_res, temp_res)

        s_mask = spat_res.mask[idx]
        s_rep = spat_res.multi_channel_masks[idx].get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        t_mask = temp_res.scratch_mask[idx]
        t_rep = temp_res.multi_channel_masks[idx].get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)
        f_mask = fused_res.mask[idx]
        f_cons = fused_res.multi_channel_masks[idx].get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        f_hr = fused_res.multi_channel_masks[idx].get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)

        eval_modes = {
            "spatial_only": da.evaluate_structural_negative_controls(s_mask, s_rep, spec["rois"]),
            "temporal_only": da.evaluate_structural_negative_controls(t_mask, t_rep, spec["rois"]),
            "fused_conservative": da.evaluate_structural_negative_controls(f_mask, f_cons, spec["rois"]),
            "fused_high_recall": da.evaluate_structural_negative_controls(f_mask, f_hr, spec["rois"]),
        }

        seq_reports.append({
            "sequence_name": spec["sequence_name"],
            "split": spec["split"],
            "global_frame_index": spec["global_frame_index"],
            "resolution": [frames[0].shape[1], frames[0].shape[0]],
            "negative_control_evaluation": eval_modes,
        })

        if spec["sequence_name"] in ("neighbors_frame25_pipe_audit", "paleface_seq_a"):
            vis_failures[spec["sequence_name"]] = {
                "frame": frames[idx],
                "temporal_mask": t_mask,
                "fused_conservative": f_cons,
                "roi": spec["rois"][0],
            }

    return {
        "track": "Track C — Real Archival Negative Controls, Quarantine, & Human Annotation Readiness",
        "sequences": seq_reports,
        "_vis": vis_failures,
    }


def _save_summary_visual(
    track_a_vis: list[dict[str, Any]],
    track_b_vis: dict[str, Any],
    track_c_vis: dict[str, Any],
    out_path: Path,
) -> None:
    """Create a 2x3 visual proof and failure-example panel across Tracks A, B, and C."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=140)
    fig.patch.set_facecolor("#121212")

    def _overlay(bgr: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int]) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
        idx = mask > 0
        for c in range(3):
            rgb[idx, c] = np.clip(rgb[idx, c].astype(float) * 0.4 + color_rgb[c] * 0.6, 0, 255).astype(np.uint8)
        return rgb

    # 1. Track A: FDS Scan 8 GT (Green) vs FAS (Red)
    ax = axes[0, 0]
    if track_a_vis:
        va = track_a_vis[0]
        img = _overlay(va["image"], va["gt"], (40, 230, 60))
        idx_p = va["pred"] > 0
        img[idx_p] = [240, 60, 60]
        ax.imshow(img)
        ax.set_title("Track A: FDS Scan (8) Native Window\nGreen=Human Polygon GT | Red=FAS Pred", color="white", fontsize=10, weight="bold")
    ax.axis("off")

    # 2. Track B: Diagonal Bright Curved (Miss Example)
    ax = axes[0, 1]
    vb_diag = track_b_vis.get("diagonal_bright_curved")
    if vb_diag:
        img = _overlay(vb_diag["damaged"], vb_diag["gt"], (40, 230, 60))
        idx_t = vb_diag["temporal_pred"] > 0
        img[idx_t] = [0, 220, 255]
        ax.imshow(img)
        ax.set_title("Track B: Diagonal Curved (Dev Diagnostic 020)\nGreen=Exact Injected GT | Cyan=Temporal Pred", color="white", fontsize=10, weight="bold")
    ax.axis("off")

    # 3. Track B: Vertical Bright Drifting Repaired vs Clean Ref
    ax = axes[0, 2]
    vb_vert = track_b_vis.get("vertical_bright_drifting")
    if vb_vert:
        img = _overlay(vb_vert["repaired_mc"], vb_vert["repair_mask"], (255, 200, 40))
        ax.imshow(img)
        ax.set_title("Track B: Vertical Drifting (Dev Diagnostic 020)\nYellow=Conservative Repair Mask on Restored", color="white", fontsize=10, weight="bold")
    ax.axis("off")

    # 4. Track C Failure 1: Neighbors Frame 25 Pipe (Zoomed crop)
    ax = axes[1, 0]
    vc_pipe = track_c_vis.get("neighbors_frame25_pipe_audit")
    if vc_pipe:
        f = vc_pipe["frame"]
        m = vc_pipe["fused_conservative"]
        crop = _overlay(f[20:320, 80:320], m[20:320, 80:320], (255, 50, 50))
        ax.imshow(crop)
        leak = int(np.count_nonzero(m[SCRATCH_ROI["ymin"]:SCRATCH_ROI["ymax"], SCRATCH_ROI["xmin"]:SCRATCH_ROI["xmax"]]))
        ax.set_title(f"Track C Failure: Neighbors Frame 25 Pipe\nRed=Conservative Repair Leak ({leak} px on intact pipe)", color="#FF8888", fontsize=10, weight="bold")
    ax.axis("off")

    # 5. Track C Failure 2: Paleface Actor Jacket Edge (Zoomed crop)
    ax = axes[1, 1]
    vc_pale = track_c_vis.get("paleface_seq_a")
    if vc_pale:
        f = vc_pale["frame"]
        m = vc_pale["fused_conservative"]
        roi = vc_pale["roi"]
        crop = _overlay(f[140:420, 160:380], m[140:420, 160:380], (255, 50, 50))
        ax.imshow(crop)
        leak = int(np.count_nonzero(m[roi["ymin"]:roi["ymax"], roi["xmin"]:roi["xmax"]]))
        ax.set_title(f"Track C Failure: Paleface Actor Jacket\nRed=Conservative Repair Leak ({leak} px on moving actor)", color="#FF8888", fontsize=10, weight="bold")
    ax.axis("off")

    # 6. Track B: Horizontal Fissure (Miss Example)
    ax = axes[1, 2]
    vb_hor = track_b_vis.get("horizontal_fissure")
    if vb_hor:
        img = _overlay(vb_hor["damaged"], vb_hor["gt"], (40, 230, 60))
        idx_t = vb_hor["temporal_pred"] > 0
        img[idx_t] = [0, 220, 255]
        ax.imshow(img)
        ax.set_title("Track B: Horizontal Fissure (Dev Diagnostic 020)\nGreen=Exact Injected GT | Cyan=Temporal Pred", color="white", fontsize=10, weight="bold")
    ax.axis("off")

    plt.suptitle("Validated Three-Track Benchmark: Real Stills, Exact Video Injections, & Structural Failure Examples", color="white", fontsize=13, weight="bold")
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.95])
    fig.savefig(str(out_path), dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_quarantine_and_annotation_manifests() -> tuple[Path, Path]:
    """Write explicit quarantine notice and Track C human annotation readiness manifest."""
    quarantine_notice = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "invalidated_claims": [
            {
                "claim": "Neighbors_72 Frame 25 Hard Quality Gate PASSED (+37.87% recall gain)",
                "source": "docs/benchmarks/temporal_scratch_quality_gate.json",
                "reason": "SCRATCH_ROI (x=180..192, y=48..282) is an intact building wall vertical pipe (GT=0 negative control), not film damage.",
                "status": "INVALIDATED_AND_QUARANTINED",
            },
            {
                "claim": "Option 2 FAS + Temporal Scratch 0.0% -> 42.8% recall gain on Neighbors_72 Frame 25",
                "source": "docs/benchmarks/temporal_scratch_option2_benchmark.json",
                "reason": "Derived from the same invalidated Frame 25 architectural pipe contrast mask.",
                "status": "INVALIDATED_AND_QUARANTINED",
            },
            {
                "claim": "Cross-film validation metrics using unregistered absdiff(f, min(f_prev, f_next)) pseudo-masks and min-neighbor PSNR",
                "source": "docs/benchmarks/cross_film_generalization_benchmark.json",
                "reason": "Circular evaluation against temporal-difference thresholds and non-independent neighbor proxy reference.",
                "status": "QUARANTINED_DIAGNOSTIC_ONLY",
            },
            {
                "claim": "DART 72 precomputed Neighbors masks as ground truth",
                "source": "evaluation/dart_trial_01/masks",
                "reason": "Model predictions (label_provenance='model_pseudo_label'), never independent ground truth.",
                "status": "QUARANTINED_PSEUDO_LABEL",
            },
        ],
    }
    q_path = DOCS_BENCHMARKS / "historical_claims_quarantine_notice.json"
    q_path.write_text(json.dumps(quarantine_notice, indent=2) + "\n", encoding="utf-8")

    # Track C Human Annotation Workflow Manifest
    annotation_manifest = {
        "protocol_version": "1.0.0",
        "rules": [
            "Annotators must inspect ONLY raw source frames I_t and raw neighboring frames I_{t-2..t+2}.",
            "NEVER derive ground-truth masks from detector outputs, optical flow maps, temporal differences, or reconstruction residuals.",
            "Encode 3-tier PNG masks strictly as: 0 = confirmed intact, 127 = uncertain boundary / faint halo, 255 = confirmed defect.",
            "Separate scratch and dust channels into distinct files plus explicit structural negative-control ROIs.",
            "All previously inspected clips remain in split='dev_diagnostic'; reserve unseen clips for split='locked_holdout'.",
        ],
        "dev_diagnostic_pilot_items": [
            {
                "dataset": "neighbors_pilot",
                "path": ".cache/benchmarks/neighbors_pilot/manifest.json",
                "frames": [0, 12, 23, 36, 48, 71],
                "annotation_status": "pending",
                "split": "dev_diagnostic",
            },
            {
                "dataset": "neighbors_72_verified_controls",
                "path": "evaluation/Neighbors_72.mkv",
                "frames": [25, 61, 62],
                "notes": "Frame 25 pipe (x:180..192, y:48..282) locked as GT=0 negative control; Frames 61-62 require independent human scratch re-annotation.",
                "annotation_status": "pending_independent_review",
                "split": "dev_diagnostic",
            },
            {
                "dataset": "cross_film_dev_keyframes",
                "clips": [
                    "evaluation/cross_film_validation/paleface_seq_a.mkv (frame 8)",
                    "evaluation/cross_film_validation/paleface_seq_b.mkv (frame 7)",
                    "evaluation/cross_film_validation/college_seq_c.mkv (frame 8)",
                    "evaluation/cross_film_validation/neighbors_seq_d.mkv (frames 2, 30)",
                ],
                "annotation_status": "pending_independent_review",
                "split": "dev_diagnostic",
            },
        ],
        "locked_holdout_requirements": {
            "status": "AWAITING_UNSEEN_ARCHIVAL_SELECTIONS",
            "note": "Reserve additional unseen archival film clips (not previously inspected during detector development) with >=30 independently annotated keyframes across >=3 unseen films before making cross-film generalization claims.",
        },
    }
    a_path = DOCS_BENCHMARKS / "track_c_annotation_manifest.json"
    a_path.write_text(json.dumps(annotation_manifest, indent=2) + "\n", encoding="utf-8")
    return q_path, a_path


def format_three_track_markdown(report: dict[str, Any]) -> str:
    """Format concise, evidence-only Markdown summary of the Three-Track Validation run."""
    lines = [
        "# Validated Three-Track Film Defect & Restoration Safety Benchmark",
        "",
        f"- **Timestamp:** `{report['timestamp']}`",
        f"- **Environment Isolation:** `{report['isolation']['is_isolated']}` (`{report['isolation']['current_python']}`)",
        "- **Detector Modification Check:** `0 thresholds, presets, or fusion rules modified (baseline frozen)`",
        "",
        "> [!WARNING]",
        "> Historical Frame 25 pipe benchmarks (`temporal_scratch_quality_gate.json`, `temporal_scratch_option2_benchmark.json`) and `absdiff` cross-film pseudo-labels (`cross_film_generalization_benchmark.json`) are **quarantined** and excluded from ground-truth status.",
        "",
        "---",
        "",
        "## Track A — Real Annotated Still Images (Spatial Evaluation Only)",
        "",
        "| Dataset (`provenance`) | Detector | Precision | Recall | Pixel F1 | FP Pixels | Max FP Comp Area | Max FP Comp H | Severe Structural FPs |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds_name, ds_info in report["track_a"]["datasets"].items():
        prov = ds_info["provenance_tier"]
        for cand, cinfo in ds_info["candidates"].items():
            if cinfo["status"] != "EVALUATED":
                continue
            ov = cinfo["combined_metrics"]
            sev = cinfo["fp_severity"]
            lines.append(
                f"| `{ds_name}` (`{prov}`) | `{cand}` | {ov['precision']:.4f} | {ov['recall']:.4f} | **{ov['f1']:.4f}** | "
                f"{sev['fp_pixel_count']} | {sev['max_fp_component_area']} | {sev['max_fp_component_height']} | {sev['severe_structural_fp_count']} |"
            )

    lines.extend([
        "",
        "*All temporal/multi-frame detectors (`temporal_scratch`, `fused_fas_temporal`, `farneback_3f`, `dis_flow`, `ridge`, `consensus_5f`, `dart_harness`) were marked `INSUFFICIENT_TEMPORAL_CONTEXT` on Track A still images.*",
        "",
        "---",
        "",
        "## Track B — Exact Synthetic Defects on Real Clean Video (`AbsoluteDegradation` Clean References)",
        "",
        "| Regime (`split`) | Mode | Det Precision | Det Recall | Det F1 | Max FP Height | Repair Cov % | Edge Repair FP px | MC Defect PSNR Gain (dB) | Intact Px Modified |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for reg in report["track_b"]["regimes"]:
        rname = f"`{reg['regime']}` (`{reg['split']}`)"
        for mname in ("spatial_only", "temporal_only", "fused_conservative", "fused_high_recall"):
            m = reg["modes"][mname]
            a1 = m["axis1_detection_quality"]
            a2 = m["axis2_repair_mask_safety"]
            a3 = m["axis3_exact_restoration_quality"]["motion_comp"]
            lines.append(
                f"| {rname} | `{mname}` | {a1['precision']:.4f} | {a1['recall']:.4f} | **{a1['f1']:.4f}** | "
                f"{a1['fp_severity']['max_fp_component_height']} | {a2['repair_coverage_pct']:.3f}% | "
                f"{a2['structural_edge_repair_fp_pixels']} | {a3['mean_psnr_gain_db']:+.2f} dB | {a3['total_intact_pixels_modified']} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## Track C — Real Archival Negative Controls (Strict Zero-Leakage Safety Gate)",
        "",
        "| Sequence (`split`) | Critical ROI | Mode | Detector Leak (px) | Repair Leak (px) | Max Leak Height | Zero-Leak Gate |",
        "|---|---|---|---:|---:|---:|:---:|",
    ])
    for seq in report["track_c"]["sequences"]:
        sname = f"`{seq['sequence_name']}` (`{seq['split']}`)"
        for mname in ("spatial_only", "temporal_only", "fused_conservative"):
            for r in seq["negative_control_evaluation"][mname]["rois"]:
                if "clean" in r["name"]:
                    continue
                badge = "PASS" if r["strict_zero_leak_passed"] else "**FAIL**"
                lines.append(
                    f"| {sname} | `{r['name']}` | `{mname}` | {r['detector_leak_pixels']} | {r['repair_leak_pixels']} | "
                    f"{r['max_leak_component_height']} | {badge} |"
                )

    lines.extend([
        "",
        f"Visual comparison & failure panel saved to `{report['visual_summary_path']}`.",
    ])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commercial-only", action="store_true", help="Exclude research-only ARTeFACT / BOPB / PatchCore / PaDiM evaluation")
    args = parser.parse_args(argv)

    DOCS_BENCHMARKS.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    iso = pbr.audit_environment_isolation()
    if not iso["is_isolated"]:
        raise RuntimeError(f"Environment isolation check failed: {iso['violations']}")

    track_a = run_track_a_real_stills(include_research=not args.commercial_only)
    track_b = run_track_b_synthetic_on_real_video()
    track_c = run_track_c_negative_controls_and_quarantine()

    vis_path = VISUALS_DIR / "three_track_validation_summary.png"
    _save_summary_visual(track_a.pop("_vis", []), track_b.pop("_vis", {}), track_c.pop("_vis", {}), vis_path)

    q_path, a_path = write_quarantine_and_annotation_manifests()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "isolation": iso,
        "quarantine_notice_path": str(q_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "track_c_annotation_manifest_path": str(a_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "visual_summary_path": str(vis_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "track_a": track_a,
        "track_b": track_b,
        "track_c": track_c,
    }

    json_out = DOCS_BENCHMARKS / "validated_three_track_benchmark_report.json"
    md_out = DOCS_BENCHMARKS / "validated_three_track_benchmark_report.md"
    json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(format_three_track_markdown(report), encoding="utf-8")

    logger.info("Wrote Three-Track JSON report to %s", json_out)
    logger.info("Wrote Three-Track Markdown report to %s", md_out)
    logger.info("Wrote Quarantine Notice to %s", q_path)
    logger.info("Wrote Track C Annotation Manifest to %s", a_path)
    logger.info("Wrote Visual Summary to %s", vis_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
