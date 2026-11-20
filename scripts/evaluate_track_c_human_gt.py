#!/usr/bin/env python3
"""Post-annotation Track C evaluation against independent human 3-tier ground truth.

Strictly read-only with respect to human labels (never creates or modifies masks).
Evaluates only human-verified regions (`0` intact, `255` scratch/dust/neg_control),
cleanly excludes `127` (uncertain / unreviewed), and discloses reviewed coverage %.

All Track C pilot sequences are `dev_diagnostic` material only — NOT locked
cross-film holdouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experimental_scene_film_scratch_prototype import (  # noqa: E402
    ExperimentalConfig,
    ExperimentalSceneFilmDetector,
)
from film_defect_detectors import (  # noqa: E402
    BENCHMARK_PRESETS,
    DEFAULT_REGISTRY,
    RepairInclusionPolicy,
    fuse_spatial_and_temporal_masks,
)
from prepare_track_c_annotation_pack import (  # noqa: E402
    MANIFEST_OUT,
    resolve_manifest_path,
    resolve_pack_dir,
    verify_human_labels,
)
from run_experimental_candidate_comparison import _repair_mc_fair  # noqa: E402
from temporal_scratch_analyzer import TemporalConfig, TemporalScratchAnalyzer, load_clip  # noqa: E402

DEFAULT_REPORT_JSON = PROJECT_ROOT / "docs" / "benchmarks" / "track_c_human_gt_evaluation_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "docs" / "benchmarks" / "track_c_human_gt_evaluation_report.md"
PRE_SAFETY_FIX_REPORT = PROJECT_ROOT / "docs" / "benchmarks" / "experimental_candidate_comparison_report_pre_safety_fix.json"


def _max_connected_component_area(mask_bool: np.ndarray) -> int:
    if not np.any(mask_bool):
        return 0
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8) * 255, connectivity=8)
    if num_labels <= 1:
        return 0
    return int(np.max(stats[1:, cv2.CC_STAT_AREA]))


def _safe_div(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return round(float(num) / float(den), 6)


def evaluate_frame_masks_against_human_gt(
    *,
    scr_gt: np.ndarray,
    dst_gt: np.ndarray,
    neg_gt: np.ndarray,
    det_mask: np.ndarray,
    repair_mask: np.ndarray,
    uncertain_mask: np.ndarray | None = None,
    actual_modified_mask: np.ndarray | None = None,
    severe_cc_threshold_px: int = 25,
) -> dict[str, Any]:
    """Compute strict 3-tier Track C metrics for one frame over human-verified pixels only."""
    scr_pos = scr_gt == 255
    dst_pos = dst_gt == 255
    neg_roi = neg_gt == 255
    any_pos = scr_pos | dst_pos
    verified_intact = (scr_gt == 0) & (dst_gt == 0)
    reviewed_both = ((scr_gt != 127) & (dst_gt != 127))

    det = det_mask > 0
    rep = repair_mask > 0
    unc = (uncertain_mask > 0) if uncertain_mask is not None else np.zeros_like(rep, dtype=bool)
    unc_only = unc & (~rep)
    mod = (actual_modified_mask > 0) if actual_modified_mask is not None else np.zeros_like(rep, dtype=bool)

    total_px = int(scr_gt.size)
    scr_gt_px = int(np.count_nonzero(scr_pos))
    dst_gt_px = int(np.count_nonzero(dst_pos))
    any_gt_px = int(np.count_nonzero(any_pos))
    intact_gt_px = int(np.count_nonzero(verified_intact))
    neg_roi_px = int(np.count_nonzero(neg_roi))
    reviewed_px = int(np.count_nonzero(reviewed_both))

    det_tp_any = int(np.count_nonzero(det & any_pos))
    det_tp_scr = int(np.count_nonzero(det & scr_pos))
    det_tp_dst = int(np.count_nonzero(det & dst_pos))
    det_fp = int(np.count_nonzero(det & verified_intact))

    rep_tp_any = int(np.count_nonzero(rep & any_pos))
    rep_tp_scr = int(np.count_nonzero(rep & scr_pos))
    rep_tp_dst = int(np.count_nonzero(rep & dst_pos))
    rep_fp = int(np.count_nonzero(rep & verified_intact))
    rep_roi_leak = int(np.count_nonzero(rep & neg_roi))

    scr_routed_unc = int(np.count_nonzero(unc_only & scr_pos))
    dst_routed_unc = int(np.count_nonzero(unc_only & dst_pos))
    intact_routed_unc = int(np.count_nonzero(unc_only & verified_intact))
    roi_routed_unc = int(np.count_nonzero(unc_only & neg_roi))

    mod_intact = int(np.count_nonzero(mod & verified_intact))
    mod_roi = int(np.count_nonzero(mod & neg_roi))
    mod_defect = int(np.count_nonzero(mod & any_pos))

    max_false_rep_cc = _max_connected_component_area(rep & verified_intact)
    severe_failure = (rep_roi_leak > 0) or (mod_roi > 0) or (max_false_rep_cc >= severe_cc_threshold_px)

    return {
        "total_px": total_px,
        "reviewed_px": reviewed_px,
        "reviewed_coverage_pct": round(100.0 * reviewed_px / max(1, total_px), 2),
        "uncertain_excluded_px": int(total_px - reviewed_px),
        "gt_scratch_px": scr_gt_px,
        "gt_dust_px": dst_gt_px,
        "gt_any_defect_px": any_gt_px,
        "gt_verified_intact_px": intact_gt_px,
        "gt_neg_control_roi_px": neg_roi_px,
        "det_tp_any_px": det_tp_any,
        "det_tp_scratch_px": det_tp_scr,
        "det_tp_dust_px": det_tp_dst,
        "det_fp_intact_px": det_fp,
        "det_precision": _safe_div(det_tp_any, det_tp_any + det_fp),
        "det_recall_any": _safe_div(det_tp_any, any_gt_px),
        "det_recall_scratch": _safe_div(det_tp_scr, scr_gt_px),
        "det_recall_dust": _safe_div(det_tp_dst, dst_gt_px),
        "repair_auth_total_px": int(np.count_nonzero(rep)),
        "repair_tp_any_px": rep_tp_any,
        "repair_tp_scratch_px": rep_tp_scr,
        "repair_tp_dust_px": rep_tp_dst,
        "repair_fp_intact_px": rep_fp,
        "repair_roi_leak_px": rep_roi_leak,
        "repair_precision": _safe_div(rep_tp_any, rep_tp_any + rep_fp),
        "repair_recall_any": _safe_div(rep_tp_any, any_gt_px),
        "repair_recall_scratch": _safe_div(rep_tp_scr, scr_gt_px),
        "repair_recall_dust": _safe_div(rep_tp_dst, dst_gt_px),
        "scratch_routed_to_uncertain_px": scr_routed_unc,
        "dust_routed_to_uncertain_px": dst_routed_unc,
        "intact_routed_to_uncertain_px": intact_routed_unc,
        "roi_routed_to_uncertain_px": roi_routed_unc,
        "actual_modified_total_px": int(np.count_nonzero(mod)),
        "actual_modified_intact_px": mod_intact,
        "actual_modified_roi_px": mod_roi,
        "actual_modified_defect_px": mod_defect,
        "max_false_repair_cc_px": max_false_rep_cc,
        "severe_localized_failure": bool(severe_failure),
    }


def _aggregate_frame_metrics(frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sum_keys = [
        "total_px",
        "reviewed_px",
        "uncertain_excluded_px",
        "gt_scratch_px",
        "gt_dust_px",
        "gt_any_defect_px",
        "gt_verified_intact_px",
        "gt_neg_control_roi_px",
        "det_tp_any_px",
        "det_tp_scratch_px",
        "det_tp_dust_px",
        "det_fp_intact_px",
        "repair_auth_total_px",
        "repair_tp_any_px",
        "repair_tp_scratch_px",
        "repair_tp_dust_px",
        "repair_fp_intact_px",
        "repair_roi_leak_px",
        "scratch_routed_to_uncertain_px",
        "dust_routed_to_uncertain_px",
        "intact_routed_to_uncertain_px",
        "roi_routed_to_uncertain_px",
        "actual_modified_total_px",
        "actual_modified_intact_px",
        "actual_modified_roi_px",
        "actual_modified_defect_px",
    ]
    agg = {k: int(sum(int(r[k]) for r in frame_rows)) for k in sum_keys}
    agg["evaluated_keyframes"] = len(frame_rows)
    agg["reviewed_coverage_pct"] = round(100.0 * agg["reviewed_px"] / max(1, agg["total_px"]), 2)
    agg["det_precision"] = _safe_div(agg["det_tp_any_px"], agg["det_tp_any_px"] + agg["det_fp_intact_px"])
    agg["det_recall_any"] = _safe_div(agg["det_tp_any_px"], agg["gt_any_defect_px"])
    agg["det_recall_scratch"] = _safe_div(agg["det_tp_scratch_px"], agg["gt_scratch_px"])
    agg["det_recall_dust"] = _safe_div(agg["det_tp_dust_px"], agg["gt_dust_px"])
    agg["repair_precision"] = _safe_div(agg["repair_tp_any_px"], agg["repair_tp_any_px"] + agg["repair_fp_intact_px"])
    agg["repair_recall_any"] = _safe_div(agg["repair_tp_any_px"], agg["gt_any_defect_px"])
    agg["repair_recall_scratch"] = _safe_div(agg["repair_tp_scratch_px"], agg["gt_scratch_px"])
    agg["repair_recall_dust"] = _safe_div(agg["repair_tp_dust_px"], agg["gt_dust_px"])
    agg["max_false_repair_cc_px"] = max((int(r["max_false_repair_cc_px"]) for r in frame_rows), default=0)
    agg["severe_localized_failure_frames"] = sum(1 for r in frame_rows if r["severe_localized_failure"])
    return agg


def _load_optional_previous_e2_masks(
    prev_dir: Path | None,
    clip_id: str,
    frame_idx: int,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Optionally load archived per-pixel Previous E2 masks if supplied via --previous-e2-dir."""
    if prev_dir is None:
        return None
    candidates = [
        prev_dir / f"{clip_id}_f{frame_idx}",
        prev_dir / clip_id / f"frame_{frame_idx:06d}",
        prev_dir,
    ]
    for base in candidates:
        if base.is_dir():
            scr_p = base / "repair_mask.png"
            hc_p = base / "state_hc.png"
            unc_p = base / "state_unc.png"
            rep_p = scr_p if scr_p.exists() else (base / f"{clip_id}_f{frame_idx}_repair.png")
            det_p = base / f"{clip_id}_f{frame_idx}_det.png"
            unc_alt = base / f"{clip_id}_f{frame_idx}_uncertain.png"
            if not rep_p.exists() and hc_p.exists():
                rep_p = hc_p
            if rep_p.exists():
                rep = cv2.imread(str(rep_p), cv2.IMREAD_GRAYSCALE)
                if rep is None or rep.shape[:2] != shape_hw:
                    continue
                unc = np.zeros(shape_hw, dtype=np.uint8)
                for up in (unc_p, unc_alt):
                    if up.exists():
                        u_arr = cv2.imread(str(up), cv2.IMREAD_GRAYSCALE)
                        if u_arr is not None and u_arr.shape[:2] == shape_hw:
                            unc = (u_arr > 0).astype(np.uint8) * 255
                            break
                if det_p.exists():
                    d_arr = cv2.imread(str(det_p), cv2.IMREAD_GRAYSCALE)
                    det = (d_arr > 0).astype(np.uint8) * 255 if d_arr is not None and d_arr.shape[:2] == shape_hw else np.bitwise_or((rep > 0).astype(np.uint8) * 255, unc)
                else:
                    det = np.bitwise_or((rep > 0).astype(np.uint8) * 255, unc)
                return det, (rep > 0).astype(np.uint8) * 255, unc
    return None


def _fmt_pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100.0:.2f}%"


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Track C Human Ground-Truth Evaluation Report",
        "",
        "> **Scope Notice:** Evaluates strictly over human-verified regions (`0 = intact`, `255 = scratch/dust/neg_control`). Pixels marked `127` (`uncertain / unreviewed`) are excluded cleanly from metrics, with reviewed coverage disclosed per frame and in aggregate. All 14 Track C pilot keyframes come from `dev_diagnostic` sequences—these results measure pilot real-film accuracy and safety, **not** cross-film holdout generalization.",
        "",
        f"- **Evaluated keyframes:** `{report['evaluated_keyframes']} / {report['total_keyframes']}`",
        f"- **Mean reviewed coverage (evaluated frames):** `{report['mean_evaluated_coverage_pct']:.2f}%`",
        f"- **Second-pass verified keyframes (out of evaluated):** `{report['second_pass_verified_evaluated']} / {report['evaluated_keyframes']}`",
        f"- **Human label conflicts:** `{report['label_audit']['total_conflict_px']}` px",
        "",
        "## Variant Comparison (Aggregated Over Evaluated Human-Verified Regions)",
        "",
        "| Variant | Det Prec | Det Recall (Any / Scr / Dst) | Auth Prec | Auth Recall (Any / Scr / Dst) | False Repair Auth (Intact px) | ROI Leak px | Scr->Uncertain px | Actual Modified Intact / ROI px | Severe Loc Failures |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for name, agg in report.get("variants", {}).items():
        if agg.get("status") == "not_archived_per_pixel":
            lines.append(f"| `{name}` | N/A | N/A | N/A | N/A | N/A | N/A | N/A | {agg.get('note', 'not archived')} | N/A |")
            continue
        lines.append(
            f"| `{name}` | {_fmt_pct(agg['det_precision'])} | "
            f"{_fmt_pct(agg['det_recall_any'])} / {_fmt_pct(agg['det_recall_scratch'])} / {_fmt_pct(agg['det_recall_dust'])} | "
            f"{_fmt_pct(agg['repair_precision'])} | "
            f"{_fmt_pct(agg['repair_recall_any'])} / {_fmt_pct(agg['repair_recall_scratch'])} / {_fmt_pct(agg['repair_recall_dust'])} | "
            f"{agg['repair_fp_intact_px']} | {agg['repair_roi_leak_px']} | {agg['scratch_routed_to_uncertain_px']} | "
            f"{agg['actual_modified_intact_px']} / {agg['actual_modified_roi_px']} | {agg['severe_localized_failure_frames']} |"
        )
    lines.append("")

    if report.get("historical_pre_safety_fix_reference"):
        ref = report["historical_pre_safety_fix_reference"]
        lines.extend(
            [
                "## Historical Pre-Safety-Fix E2 Archive Reference",
                "",
                f"- Source: `{ref['source_report']}` (evaluated previously on Track A/B + dev archival negative controls prior to the E2 temporal-reliability fix; NOT directly measured pixel-for-pixel on this user Track C pilot pack unless `--previous-e2-dir` mask PNGs are supplied).",
                f"- Archived note: `{ref.get('summary_note', '')}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Per-Frame Corrected E2 vs Baseline A2 Breakdown",
            "",
            "| Clip #Frame | Cov % | 2nd Pass | GT Scr / Dst / ROI px | A2 Auth FP / ROI / Mod | E2 Auth FP / ROI / Mod | E2 Scr Recall / Scr->Unc | Severe Loc Fail (A2 / E2) |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for pf in report.get("per_frame", []):
        a2 = pf["variants"].get("Conservative_baseline_A2", {})
        e2 = pf["variants"].get("Corrected_E2_fused_3state", {})
        lines.append(
            f"| `{pf['clip_id']}#{pf['frame_index']}` | {pf['reviewed_coverage_pct']:.2f}% | `{pf['second_pass_verified']}` | "
            f"{pf['gt_scratch_px']} / {pf['gt_dust_px']} / {pf['gt_neg_control_roi_px']} | "
            f"{a2.get('repair_fp_intact_px', '-')} / {a2.get('repair_roi_leak_px', '-')} / {a2.get('actual_modified_intact_px', '-')} | "
            f"{e2.get('repair_fp_intact_px', '-')} / {e2.get('repair_roi_leak_px', '-')} / {e2.get('actual_modified_intact_px', '-')} | "
            f"{_fmt_pct(e2.get('repair_recall_scratch'))} / {e2.get('scratch_routed_to_uncertain_px', '-')} | "
            f"{a2.get('severe_localized_failure', '-')} / {e2.get('severe_localized_failure', '-')} |"
        )
    lines.append("")
    return "\n".join(lines)


def run_track_c_evaluation(
    *,
    manifest_path: Path = MANIFEST_OUT,
    pack_dir: Path | None = None,
    project_root: Path = PROJECT_ROOT,
    allow_partial: bool = False,
    require_second_pass: bool = False,
    previous_e2_dir: Path | None = None,
    pre_safety_report_path: Path | None = PRE_SAFETY_FIX_REPORT,
    md_out: Path | None = DEFAULT_REPORT_MD,
    json_out: Path | None = DEFAULT_REPORT_JSON,
) -> dict[str, Any]:
    audit = verify_human_labels(manifest_path=manifest_path, pack_dir=pack_dir, project_root=project_root)
    if audit["invalid_files"] or audit["raw_integrity_failures"]:
        raise RuntimeError(f"Track C pack validation failed: {audit['invalid_files'] or audit['raw_integrity_failures']}")
    if audit["total_conflict_px"] > 0:
        raise RuntimeError(f"Resolve {audit['total_conflict_px']} human label conflicts before Track C evaluation.")
    if audit["completed_keyframes"] == 0:
        raise RuntimeError(
            "0 completed human annotations found. Annotate pilot keyframes first using scripts/track_c_annotator.py."
        )
    if not allow_partial and audit["completed_keyframes"] < audit["total_keyframes"]:
        raise RuntimeError(
            f"Only {audit['completed_keyframes']}/{audit['total_keyframes']} keyframes annotated. "
            "Pass --allow-partial to evaluate currently annotated frames only."
        )
    if require_second_pass and not audit["ready_with_second_pass"]:
        raise RuntimeError(
            f"Only {audit['second_pass_verified_keyframes']}/{audit['total_keyframes']} keyframes have second_pass_verified status."
        )

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved_pack = resolve_pack_dir(manifest_data, pack_dir=pack_dir, project_root=project_root)
    pilot_pack_rel = str(manifest_data.get("pilot_pack_dir") or ".cache/benchmarks/track_c_pilot_pack")
    items_by_key = {(str(it["clip_id"]), int(it["frame_index"])): it for it in manifest_data.get("pilot_keyframes", [])}

    evaluated_pf = [f for f in audit["per_frame"] if int(f.get("any_channel_reviewed_px", 0)) > 0]
    by_clip: dict[str, list[int]] = {}
    for pf in evaluated_pf:
        by_clip.setdefault(str(pf["clip_id"]), []).append(int(pf["frame_index"]))

    spat_det = DEFAULT_REGISTRY.get("film_artefact_segmenter")
    temp_det = DEFAULT_REGISTRY.get("temporal_scratch")
    spat_cfg = dict(BENCHMARK_PRESETS["film_artefact_balanced"].params)
    temp_cfg = dict(BENCHMARK_PRESETS["temporal_scratch_balanced"].params)
    analyzer = TemporalScratchAnalyzer(TemporalConfig())
    exp_det = ExperimentalSceneFilmDetector(ExperimentalConfig())

    variant_rows: dict[str, list[dict[str, Any]]] = {
        "Spatial_only_balanced": [],
        "Temporal_only_A1": [],
        "Conservative_baseline_A2": [],
        "Temporal_only_E1": [],
        "Corrected_E2_fused_3state": [],
    }
    prev_e2_rows: list[dict[str, Any]] = []
    per_frame_out: list[dict[str, Any]] = []

    for clip_id, frame_indices in by_clip.items():
        first_item = items_by_key[(clip_id, frame_indices[0])]
        video_path = resolve_manifest_path(
            str(first_item["source_video"]),
            pack_dir=resolved_pack,
            pilot_pack_rel=pilot_pack_rel,
            project_root=project_root,
        )
        frames = load_clip(video_path)
        s_res = spat_det.detect(frames, spat_cfg)
        t_res = temp_det.detect(frames, temp_cfg)
        f_res = fuse_spatial_and_temporal_masks(s_res, t_res)
        e1_seq = exp_det.analyze_sequence(frames, target_indices=frame_indices)
        e2_seq = exp_det.analyze_sequence(
            frames,
            spatial_scratch_masks=[m.scratch_mask for m in s_res.multi_channel_masks],
            spatial_dust_masks=[m.dust_mask for m in s_res.multi_channel_masks],
            target_indices=frame_indices,
        )

        pf_map = {(str(f["clip_id"]), int(f["frame_index"])): f for f in evaluated_pf}
        for idx in frame_indices:
            item = items_by_key[(clip_id, idx)]
            pf = pf_map[(clip_id, idx)]
            scr = cv2.imread(str(resolve_manifest_path(str(item["scratch_template_path"]), pack_dir=resolved_pack, pilot_pack_rel=pilot_pack_rel, project_root=project_root)), cv2.IMREAD_UNCHANGED)
            dst = cv2.imread(str(resolve_manifest_path(str(item["dust_template_path"]), pack_dir=resolved_pack, pilot_pack_rel=pilot_pack_rel, project_root=project_root)), cv2.IMREAD_UNCHANGED)
            neg = cv2.imread(str(resolve_manifest_path(str(item["neg_control_template_path"]), pack_dir=resolved_pack, pilot_pack_rel=pilot_pack_rel, project_root=project_root)), cv2.IMREAD_UNCHANGED)

            s_mcm = s_res.multi_channel_masks[idx]
            t_mcm = t_res.multi_channel_masks[idx]
            f_mcm = f_res.multi_channel_masks[idx]
            e1_r = e1_seq[idx]
            e2_r = e2_seq[idx]

            variant_masks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {
                "Spatial_only_balanced": (
                    s_res.mask[idx],
                    s_mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE),
                    s_mcm.uncertain_mask,
                ),
                "Temporal_only_A1": (
                    t_res.mask[idx],
                    t_mcm.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY),
                    None,
                ),
                "Conservative_baseline_A2": (
                    f_res.mask[idx],
                    f_mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE),
                    f_mcm.uncertain_mask,
                ),
                "Temporal_only_E1": (
                    e1_r.all_detected_candidates,
                    e1_r.high_confidence_damage,
                    e1_r.uncertain_damage_candidate,
                ),
                "Corrected_E2_fused_3state": (
                    e2_r.all_detected_candidates,
                    e2_r.high_confidence_damage,
                    e2_r.uncertain_damage_candidate,
                ),
            }

            frame_var_results: dict[str, dict[str, Any]] = {}
            for vname, (dm, rm, um) in variant_masks.items():
                repaired = _repair_mc_fair(frames, idx, rm, analyzer) if np.any(rm > 0) else frames[idx]
                actual_mod = np.any(repaired != frames[idx], axis=2) if np.any(rm > 0) else np.zeros(scr.shape, dtype=bool)
                metrics = evaluate_frame_masks_against_human_gt(
                    scr_gt=scr,
                    dst_gt=dst,
                    neg_gt=neg,
                    det_mask=dm,
                    repair_mask=rm,
                    uncertain_mask=um,
                    actual_modified_mask=actual_mod,
                )
                variant_rows[vname].append(metrics)
                frame_var_results[vname] = metrics

            prev_masks = _load_optional_previous_e2_masks(previous_e2_dir, clip_id, idx, scr.shape)
            if prev_masks is not None:
                p_det, p_rep, p_unc = prev_masks
                p_repaired = _repair_mc_fair(frames, idx, p_rep, analyzer) if np.any(p_rep > 0) else frames[idx]
                p_mod = np.any(p_repaired != frames[idx], axis=2) if np.any(p_rep > 0) else np.zeros(scr.shape, dtype=bool)
                pm = evaluate_frame_masks_against_human_gt(
                    scr_gt=scr,
                    dst_gt=dst,
                    neg_gt=neg,
                    det_mask=p_det,
                    repair_mask=p_rep,
                    uncertain_mask=p_unc,
                    actual_modified_mask=p_mod,
                )
                prev_e2_rows.append(pm)
                frame_var_results["Previous_E2_pre_safety_fix"] = pm

            per_frame_out.append(
                {
                    "clip_id": clip_id,
                    "frame_index": idx,
                    "dataset_role": str(pf["dataset_role"]),
                    "reviewed_coverage_pct": pf["reviewed_coverage_pct"],
                    "second_pass_verified": pf["second_pass_verified"],
                    "gt_scratch_px": pf["scratch_px"],
                    "gt_dust_px": pf["dust_px"],
                    "gt_neg_control_roi_px": pf["neg_control_roi_px"],
                    "variants": frame_var_results,
                }
            )

    variants_agg = {vname: _aggregate_frame_metrics(rows) for vname, rows in variant_rows.items()}
    if len(prev_e2_rows) == len(evaluated_pf) and prev_e2_rows:
        variants_agg["Previous_E2_pre_safety_fix"] = _aggregate_frame_metrics(prev_e2_rows)
    else:
        variants_agg["Previous_E2_pre_safety_fix"] = {
            "status": "not_archived_per_pixel",
            "note": "No per-frame pre-fix E2 binary masks provided (--previous-e2-dir). Cannot fabricate pixel counts.",
        }

    hist_ref: dict[str, Any] | None = None
    if pre_safety_report_path is not None and pre_safety_report_path.exists():
        hist_ref = {
            "source_report": str(pre_safety_report_path),
            "summary_note": "Archived pre-safety-fix comparison report prior to Paleface B f8 E2 temporal-reliability fix (historical Track A/B & negative-control context only).",
        }

    covs = [float(f["reviewed_coverage_pct"]) for f in per_frame_out]
    report: dict[str, Any] = {
        "protocol_version": audit.get("protocol_version", "2.0.0"),
        "annotation_classification": "dev_diagnostic_pilot_only",
        "total_keyframes": int(audit["total_keyframes"]),
        "evaluated_keyframes": len(per_frame_out),
        "second_pass_verified_evaluated": sum(1 for f in per_frame_out if f["second_pass_verified"]),
        "mean_evaluated_coverage_pct": round(sum(covs) / max(1, len(covs)), 2),
        "label_audit": {
            "completed_keyframes": audit["completed_keyframes"],
            "total_conflict_px": audit["total_conflict_px"],
            "ready_for_real_film_claim": audit["ready_for_real_film_claim"],
            "ready_with_second_pass": audit["ready_with_second_pass"],
        },
        "variants": variants_agg,
        "historical_pre_safety_fix_reference": hist_ref,
        "per_frame": per_frame_out,
    }

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if md_out is not None:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        md_out.write_text(render_evaluation_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_OUT)
    parser.add_argument("--pack-dir", type=Path, default=None)
    parser.add_argument("--allow-partial", action="store_true", help="Evaluate whatever keyframes currently have human annotations instead of requiring all 14.")
    parser.add_argument("--require-second-pass", action="store_true", help="Require second_pass_verified status on all evaluated keyframes.")
    parser.add_argument("--previous-e2-dir", type=Path, default=None, help="Optional directory containing archived pre-fix E2 binary masks.")
    parser.add_argument("--report-md", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    args = parser.parse_args()

    try:
        res = run_track_c_evaluation(
            manifest_path=args.manifest_path,
            pack_dir=args.pack_dir,
            allow_partial=args.allow_partial,
            require_second_pass=args.require_second_pass,
            previous_e2_dir=args.previous_e2_dir,
            md_out=args.report_md,
            json_out=args.report_json,
        )
        print(json.dumps(res["variants"], indent=2))
    except RuntimeError as exc:
        print(f"[Track C Evaluation Guard] {exc}", file=sys.stderr)
        raise SystemExit(2)
