#!/usr/bin/env python3
"""Isolated Experimental Prototype: Motion-Aware Scene-vs-Film Coordinate Scratch Detector
and Three-State Conservative Repair Authorization.

IMPORTANT:
- Completely isolated from production detector, fusion, preset, restoration, and GUI paths.
- Separates the three confirmed failure mechanisms into independently toggleable flags
  so factorial ablations can measure main effects and interactions cleanly:
    M1 (`use_motion_compensated_evidence`):
        Replace absolute optical-flow magnitude gate (`flow <= 1.2`) with FB-weighted
        motion-compensated background evidence + scene-locked vs. film-locked persistence.
    M2 (`use_robust_scene_cut_selector`):
        Distinguish fast camera pans from true scene cuts using telecine dedup +
        histogram L1 distance + coarse global phase-correlation aligned MAD.
    M3 (`use_centerline_thickness_geometry`):
        Orientation-aware multi-angle morphological bridging + Euclidean Distance Transform
        local centerline thickness and intrinsic elongation (replaces axis-aligned `cw <= 12`).
- Separates detection from repair authorization via three mutually exclusive outputs:
    1. `high_confidence_damage`      (authorized for automatic CONSERVATIVE repair)
    2. `uncertain_damage_candidate`  (surfaced for review / HIGH_RECALL; excluded from CONSERVATIVE)
    3. `likely_intact_scene_structure` (vetoed intact scene structure; excluded from repair)
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from temporal_scratch_analyzer import TemporalConfig, TemporalScratchAnalyzer

logger = logging.getLogger("experimental_scene_film_prototype")

_TEMPORAL_CLASSES = (
    "transient_defect",
    "persistent_film_locked",
    "persistent_scene_locked",
    "ambiguous_static_persistent",
)


@dataclass(frozen=True)
class ExperimentalConfig:
    """Configuration for the isolated experimental scene-vs-film prototype."""

    # Independently toggleable mechanisms for factorial ablations:
    use_motion_compensated_evidence: bool = True   # M1
    use_robust_scene_cut_selector: bool = True     # M2
    use_centerline_thickness_geometry: bool = True # M3

    # Temporal window & telecine handling
    window_radius: int = 3
    min_temporal_neighbors: int = 2
    telecine_dedup_threshold: float = 0.8
    flow_backend: str = "dis"

    # Legacy baseline fallbacks (when M1/M2/M3 are toggled False)
    legacy_scene_cut_threshold: float = 25.0
    legacy_motion_coherence_threshold: float = 1.2
    legacy_scratch_max_width: int = 12
    legacy_min_aspect_ratio: float = 3.5

    # M2 robust cut thresholds
    hist_cut_threshold: float = 0.38
    aligned_cut_mad_threshold: float = 24.0
    raw_hard_cut_mad: float = 65.0
    unaligned_cut_mad_threshold: float = 28.0
    min_cut_phase_response: float = 0.22
    allow_single_frame_flicker_skip: bool = True

    # Detection sensitivity & noise model
    sensitivity: float = 0.65
    k_factor: float = 1.3
    contrast_base: float = 12.0

    # Forward-backward flow reliability & fail-closed registration bounds
    fb_sigma: float = 2.0
    fb_warning_threshold: float = 1.8
    min_neighbor_in_bounds_frac: float = 0.40
    max_neighbor_fb_med: float = 12.0
    min_neighbor_fb_ok_frac: float = 0.15
    max_neighbor_warped_mad: float = 22.0
    max_global_fb_med: float = 6.0
    min_global_fb_ok_frac: float = 0.50
    max_override_fb_err: float = 16.0
    min_override_global_fb_ok_frac: float = 0.75
    max_override_global_fb_med: float = 1.2

    # Gate weave allowance in film coordinates (pixels)
    film_weave_x: int = 2
    film_weave_y: int = 1

    # M3 orientation-aware centerline geometry limits
    min_line_length: float = 15.0
    max_local_thickness: float = 8.5
    max_local_thickness_elongated: float = 14.0
    min_elongation: float = 3.5
    tall_override_length: float = 35.0
    dilate_result: bool = True


@dataclass
class ComponentRecord:
    """Audit record for a candidate connected component across temporal/spatial streams."""

    stream: str  # "temporal_scratch" | "spatial_scratch" | "spatial_dust"
    component_id: int
    area_px: int
    orientation_deg: float
    local_thickness_px: float
    arc_length_px: float
    elongation: float
    mean_flow_mag_px: float
    surround_flow_mag_px: float
    median_fb_err_px: float
    fb_warning: bool
    scene_persistence_ratio: float
    film_persistence_ratio: float
    flank_novelty_med: float
    r_scene_med: float
    z_scene_med: float
    bg_grad_med: float
    follows_scene_motion: bool
    temporal_class: str
    decision: str  # "high_confidence_damage" | "uncertain_damage_candidate" | "likely_intact_scene_structure"
    reason: str


@dataclass
class ExperimentalThreeStateResult:
    """Per-frame output cleanly separating detection proposals from repair authorization."""

    high_confidence_damage: np.ndarray        # uint8 {0, 255} — authorized for CONSERVATIVE repair
    uncertain_damage_candidate: np.ndarray    # uint8 {0, 255} — human review / HIGH_RECALL only
    likely_intact_scene_structure: np.ndarray # uint8 {0, 255} — vetoed intact structure
    temporal_candidate_mask: np.ndarray       # uint8 {0, 255} — all accepted geometric linear candidates
    background: np.ndarray                    # float32 (H, W) — reliability-weighted MC background
    flow_mag: np.ndarray                      # float32 (H, W)
    fb_err: np.ndarray                        # float32 (H, W)
    fb_warning_map: np.ndarray                # bool (H, W)
    abstained: bool
    abstention_reason: str | None
    valid_neighbors: list[int]
    skipped_duplicates: list[int]
    component_records: list[ComponentRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_detected_candidates(self) -> np.ndarray:
        """Union of high-confidence damage and uncertain candidates (all non-vetoed proposals)."""
        return np.bitwise_or(self.high_confidence_damage, self.uncertain_damage_candidate)


def _ensure_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _norm_hist(gray: np.ndarray, bins: int = 64) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).ravel().astype(np.float32)
    s = float(hist.sum())
    return hist / max(1.0, s)


def _coarse_aligned_mad(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
) -> tuple[float, float, float, tuple[float, float]]:
    """Measure global translation-compensated MAD via coarse FFT phase correlation.

    Distinguishes within-shot rapid camera pans (high raw MAD, low aligned MAD)
    from genuine scene cuts (high aligned MAD and/or high histogram distance).
    """
    h, w = gray_a.shape
    scale = min(1.0, 240.0 / float(max(h, w)))
    target_w = max(48, int(round(w * scale)))
    target_h = max(36, int(round(h * scale)))
    small_a = cv2.resize(gray_a, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_b = cv2.resize(gray_b, (target_w, target_h), interpolation=cv2.INTER_AREA).astype(np.float32)

    hann = cv2.createHanningWindow((target_w, target_h), cv2.CV_32F)
    (sx, sy), resp = cv2.phaseCorrelate(small_a, small_b, hann)

    def _shift_interior_mad(dx: float, dy: float) -> float:
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        warped = cv2.warpAffine(
            small_b,
            m,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=-1.0,
        )
        valid = warped >= 0.0
        # Require meaningful spatial overlap (at least 30% of frame)
        if int(np.count_nonzero(valid)) < int(0.30 * target_h * target_w):
            return 255.0
        return float(np.mean(np.abs(small_a[valid] - warped[valid])))

    # Evaluate zero shift plus both phase-correlation sign conventions so a
    # spurious FFT peak on static/low-texture shots never inflates aligned_mad above raw_mad.
    raw_mad = float(np.mean(np.abs(small_a - small_b)))
    mad_pos = _shift_interior_mad(float(sx), float(sy))
    mad_neg = _shift_interior_mad(float(-sx), float(-sy))
    aligned_mad = min(raw_mad, mad_pos, mad_neg)
    best_shift = (0.0, 0.0) if aligned_mad == raw_mad else (float(sx / scale), float(sy / scale))
    return aligned_mad, raw_mad, float(resp), best_shift


def select_motion_robust_neighbors(
    gray_frames: Sequence[np.ndarray],
    target_idx: int,
    cfg: ExperimentalConfig,
) -> tuple[list[int], list[int], list[int], dict[str, Any]]:
    """M2: Select non-duplicate temporal neighbors with global motion compensation.

    Steps over telecine duplicate frames and distinguishes fast camera pans from true
    scene cuts using histogram L1 distance + coarse phase-correlation aligned MAD.
    """
    n = len(gray_frames)
    r = cfg.window_radius
    dedup = cfg.telecine_dedup_threshold

    if not cfg.use_robust_scene_cut_selector:
        temp_cfg = TemporalConfig(
            window_radius=r,
            telecine_dedup_threshold=dedup,
            scene_cut_threshold=cfg.legacy_scene_cut_threshold,
        )
        helper = TemporalScratchAnalyzer(temp_cfg)
        p, s, skip = helper._select_nonduplicate_neighbors(gray_frames, target_idx)
        return p, s, skip, {"mode": "legacy_cumulative_mad", "cuts_detected": []}

    hists = [_norm_hist(g) for g in gray_frames]
    skipped: list[int] = []
    cuts_detected: list[dict[str, Any]] = []

    def _scan(direction: int) -> list[int]:
        chosen: list[int] = []
        last_acc = target_idx
        idx = target_idx + direction
        while 0 <= idx < n:
            diff_t = cv2.absdiff(gray_frames[idx], gray_frames[target_idx])
            mad_t = float(np.mean(diff_t))
            diff_step = cv2.absdiff(gray_frames[idx], gray_frames[last_acc])
            mad_step = float(np.mean(diff_step))
            # A true telecine duplicate repeats the whole frame; a static scene with a sparse
            # transient defect has low global MAD but >= 24 locally altered pixels (>= 15 gray levels).
            changed_t = int(np.count_nonzero(diff_t >= 15))
            changed_step = int(np.count_nonzero(diff_step >= 15))

            if (mad_t < dedup and changed_t < 24) or (mad_step < dedup and changed_step < 24):
                skipped.append(idx)
                idx += direction
                continue

            hist_dist = float(0.5 * np.sum(np.abs(hists[idx] - hists[last_acc])))
            aligned_mad, _raw_s, resp, shift_xy = _coarse_aligned_mad(gray_frames[last_acc], gray_frames[idx])

            is_unaligned_cut = (
                mad_step >= cfg.unaligned_cut_mad_threshold and resp < cfg.min_cut_phase_response
            )
            is_cut = (
                hist_dist > cfg.hist_cut_threshold
                or aligned_mad > cfg.aligned_cut_mad_threshold
                or (mad_step > cfg.raw_hard_cut_mad and resp < 0.15)
                or is_unaligned_cut
            )
            if is_cut:
                # If ONLY global exposure/vignette histogram shifted while spatial phase correlation
                # remains high and the next non-duplicate frame directly matches `last_acc`, skip this
                # isolated flicker frame instead of erecting a false scene-cut barrier.
                only_hist_trigger = (
                    cfg.allow_single_frame_flicker_skip
                    and hist_dist > cfg.hist_cut_threshold
                    and not is_unaligned_cut
                    and aligned_mad <= 12.0
                    and resp >= 0.75
                )
                if only_hist_trigger:
                    peek_idx = idx + direction
                    skipped_flicker = False
                    hops = 0
                    while 0 <= peek_idx < n and hops < 3:
                        p_diff = cv2.absdiff(gray_frames[peek_idx], gray_frames[idx])
                        if float(np.mean(p_diff)) < dedup and int(np.count_nonzero(p_diff >= 15)) < 24:
                            peek_idx += direction
                            hops += 1
                            continue
                        p_hist = float(0.5 * np.sum(np.abs(hists[peek_idx] - hists[last_acc])))
                        p_amad, p_raw, p_resp, _ = _coarse_aligned_mad(gray_frames[last_acc], gray_frames[peek_idx])
                        p_mad_step = float(np.mean(cv2.absdiff(gray_frames[peek_idx], gray_frames[last_acc])))
                        peek_ok = (
                            p_hist <= cfg.hist_cut_threshold
                            and p_amad <= min(cfg.aligned_cut_mad_threshold, 15.0)
                            and not (p_mad_step >= cfg.unaligned_cut_mad_threshold and p_resp < cfg.min_cut_phase_response)
                        )
                        if peek_ok:
                            skipped.append(idx)
                            idx += direction
                            skipped_flicker = True
                        break
                    if skipped_flicker:
                        continue

                cuts_detected.append({
                    "frame_idx": idx,
                    "against_idx": last_acc,
                    "mad_step": round(mad_step, 2),
                    "aligned_mad": round(aligned_mad, 2),
                    "hist_dist": round(hist_dist, 3),
                    "phase_response": round(resp, 3),
                    "estimated_shift_px": [round(shift_xy[0], 1), round(shift_xy[1], 1)],
                })
                break

            chosen.append(idx)
            last_acc = idx
            if len(chosen) == r:
                break
            idx += direction
        return chosen

    preceding = _scan(-1)
    succeeding = _scan(+1)
    return preceding, succeeding, skipped, {"mode": "robust_phase_hist", "cuts_detected": cuts_detected}


def _compute_oriented_signed_channels_i16(gray: np.ndarray) -> np.ndarray:
    """Compute signed multi-orientation, multi-scale ridge contrast (4, H, W) int16."""
    h, w = gray.shape
    if gray.dtype == np.int16:
        img = gray
    elif gray.dtype == np.uint8:
        img = gray.astype(np.int16)
    else:
        img = np.rint(gray).astype(np.int16)
    offsets = (3, 5, 8)
    pad_r = max(offsets)
    padded = cv2.copyMakeBorder(img, pad_r, pad_r, pad_r, pad_r, cv2.BORDER_REPLICATE)

    axes = ((0, 1), (1, 0), (-1, 1), (1, 1))
    signed_ch = np.empty((4, h, w), dtype=np.int16)
    diff_a = np.empty((h, w), dtype=np.int16)
    diff_b = np.empty((h, w), dtype=np.int16)
    b_max = np.empty((h, w), dtype=np.int16)
    d_max = np.empty((h, w), dtype=np.int16)

    for k, (oy, ox) in enumerate(axes):
        b_max.fill(0)
        d_max.fill(0)
        for d in offsets:
            ya, xa = pad_r + oy * d, pad_r + ox * d
            yb, xb = pad_r - oy * d, pad_r - ox * d
            flank_a = padded[ya : ya + h, xa : xa + w]
            flank_b = padded[yb : yb + h, xb : xb + w]

            cv2.subtract(img, flank_a, dst=diff_a)
            cv2.subtract(img, flank_b, dst=diff_b)
            cv2.min(diff_a, diff_b, dst=diff_a)
            cv2.max(b_max, diff_a, dst=b_max)

            cv2.subtract(flank_a, img, dst=diff_a)
            cv2.subtract(flank_b, img, dst=diff_b)
            cv2.min(diff_a, diff_b, dst=diff_a)
            cv2.max(d_max, diff_a, dst=d_max)

        np.negative(d_max, out=diff_a)
        signed_ch[k] = np.where(b_max >= d_max, b_max, diff_a)
    return signed_ch


def _compute_oriented_signed_channels(gray: np.ndarray) -> np.ndarray:
    """Compute signed multi-orientation, multi-scale ridge contrast (4, H, W) float32."""
    return _compute_oriented_signed_channels_i16(gray).astype(np.float32)


def compute_oriented_ridge_features(
    gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed multi-orientation, multi-scale ridge contrast across 4 axes.

    Axes: 0=vertical line (horiz flanks), 1=horizontal line (vert flanks),
          2=+45 deg diag, 3=-45 deg diag.
    Multi-scale flank offsets (3, 5, 8) cover hairline 1-2 px scratches and broad abrasions.

    Returns:
        signed_4ch: float32 (4, H, W) signed contrast (+ bright ridge, - dark ridge)
        max_mag:    float32 (H, W) max absolute ridge contrast across orientations
        best_dir:   int32 (H, W) index in {0, 1, 2, 3}
        best_sign:  float32 (H, W) +1.0 or -1.0
    """
    h, w = gray.shape
    signed_i16 = _compute_oriented_signed_channels_i16(gray)
    abs_ch = np.abs(signed_i16)
    best_dir = np.argmax(abs_ch, axis=0).astype(np.int32)
    yy, xx = np.ogrid[:h, :w]
    best_signed_map = signed_i16[best_dir, yy, xx]
    max_mag = np.abs(best_signed_map).astype(np.float32)
    best_sign = np.where(best_signed_map >= 0, 1.0, -1.0).astype(np.float32)
    return signed_i16.astype(np.float32), max_mag, best_dir, best_sign


def _compute_legacy_vertical_flank_features(
    gray: np.ndarray,
    analyzer: TemporalScratchAnalyzer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-axis horizontal flank contrast wrapper matching legacy shape conventions."""
    flank = analyzer._compute_flank_contrast(gray).astype(np.float32)
    h, w = gray.shape
    signed_ch = np.zeros((4, h, w), dtype=np.float32)
    signed_ch[0] = flank
    return (
        signed_ch,
        flank,
        np.zeros((h, w), dtype=np.int32),
        np.ones((h, w), dtype=np.float32),
    )


def _bridged_morphology_oriented(
    candidate_u8: np.ndarray,
    best_dir: np.ndarray | None = None,
) -> np.ndarray:
    """Close narrow collinear scratch gaps along dominant local stroke axis only.

    Uses strictly odd-sized centered kernels so opening is anti-extensive (zero 1-px shift
    thickening) and gates directional closing by local binary stroke orientation so vertical
    lines are never bridged horizontally into neighboring clean textures.
    """
    # Odd symmetric 3-px directional openings remove isolated 1-2 px noise specks cleanly
    k_open_v = np.ones((3, 1), dtype=np.uint8)
    k_open_h = np.ones((1, 3), dtype=np.uint8)
    k_open_d1 = np.eye(3, dtype=np.uint8)
    k_open_d2 = np.fliplr(k_open_d1).copy()

    opened = np.zeros_like(candidate_u8)
    for k in (k_open_v, k_open_h, k_open_d1, k_open_d2):
        opened = np.bitwise_or(opened, cv2.morphologyEx(candidate_u8, cv2.MORPH_OPEN, k))

    if not np.any(opened):
        return opened

    # Measure local 7-tap collinear stroke support per orientation on the binary opened mask
    opened_01 = (opened > 0).astype(np.float32)
    k_line_v = np.ones((7, 1), dtype=np.float32)
    k_line_h = np.ones((1, 7), dtype=np.float32)
    k_line_d1 = np.eye(7, dtype=np.float32)
    k_line_d2 = np.fliplr(k_line_d1).copy()

    s_v = cv2.filter2D(opened_01, cv2.CV_32F, k_line_v, borderType=cv2.BORDER_CONSTANT)
    s_h = cv2.filter2D(opened_01, cv2.CV_32F, k_line_h, borderType=cv2.BORDER_CONSTANT)
    s_d1 = cv2.filter2D(opened_01, cv2.CV_32F, k_line_d1, borderType=cv2.BORDER_CONSTANT)
    s_d2 = cv2.filter2D(opened_01, cv2.CV_32F, k_line_d2, borderType=cv2.BORDER_CONSTANT)

    # Allow closing along the locally dominant direction(s) only (never perpendicular)
    dom_dir = np.argmax(np.stack([s_v, s_h, s_d1, s_d2], axis=0), axis=0)
    k_close_v = np.ones((7, 1), dtype=np.uint8)
    k_close_h = np.ones((1, 7), dtype=np.uint8)
    k_close_d1 = np.eye(5, dtype=np.uint8)
    k_close_d2 = np.fliplr(k_close_d1).copy()

    closed = opened.copy()
    close_specs = (
        (0, k_close_v, s_v),
        (1, k_close_h, s_h),
        (2, k_close_d1, s_d1),
        (3, k_close_d2, s_d2),
    )
    for d_idx, k_cls, score_map in close_specs:
        seed_mask = (opened > 0) & ((dom_dir == d_idx) | (score_map >= 4.0))
        if best_dir is not None and d_idx in (0, 1):
            # Never bridge vertically on strictly horizontal ridges or horizontally on strictly vertical ridges
            opp_dir = 1 if d_idx == 0 else 0
            seed_mask &= (best_dir != opp_dir) | (score_map >= 5.0)
        if np.any(seed_mask):
            seed_u8 = seed_mask.astype(np.uint8) * 255
            closed = np.bitwise_or(closed, cv2.morphologyEx(seed_u8, cv2.MORPH_CLOSE, k_cls))
    return closed


def extract_orientation_aware_components(
    candidate_mask: np.ndarray,
    cfg: ExperimentalConfig,
    *,
    best_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """M3: Orientation-aware scratch geometry via EDT centerline thickness and moments.

    Returns (accepted_binary_mask_uint8, component_geometry_metadata_list).
    """
    cand_u8 = (candidate_mask > 0).astype(np.uint8) * 255
    if not cfg.use_centerline_thickness_geometry:
        opened = cv2.morphologyEx(cand_u8, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))
        n_l, lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        out = np.zeros_like(closed)
        geom: list[dict[str, Any]] = []
        for i in range(1, n_l):
            x, y, cw, ch, area = map(int, stats[i])
            if cw <= cfg.legacy_scratch_max_width and ch >= 15 and (ch >= cfg.legacy_min_aspect_ratio * cw or ch >= 35):
                out[lbl == i] = 255
                geom.append({
                    "component_id": i,
                    "bbox": (x, y, cw, ch),
                    "area_px": area,
                    "orientation_deg": 0.0,
                    "local_thickness_px": round(area / max(1, ch), 2),
                    "arc_length_px": float(ch),
                    "elongation": round(ch / max(1.0, area / max(1, ch)), 2),
                })
        return out, geom

    closed = _bridged_morphology_oriented(cand_u8, best_dir=best_dir)
    n_l, lbl, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out = np.zeros_like(closed)
    geom = []

    for i in range(1, n_l):
        x, y, cw, ch, area = map(int, stats[i])
        if area < 10:
            continue
        sub = (lbl[y : y + ch, x : x + cw] == i).astype(np.uint8) * 255
        # Pad 1 px around ROI so distanceTransform treats bbox edges accurately
        sub_pad = np.pad(sub, ((1, 1), (1, 1)), mode="constant", constant_values=0)
        edt = cv2.distanceTransform(sub_pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
        radii = edt[sub > 0]
        local_thickness = float(2.0 * np.percentile(radii, 75)) if radii.size > 0 else 1.0
        local_thickness = max(1.0, local_thickness)

        # Principal orientation and major span from rotated minAreaRect + central moments
        ys, xs = np.nonzero(sub > 0)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        rw, rh = rect[1]
        rect_major = float(max(rw, rh))
        arc_length = max(rect_major, float(area) / max(1.0, local_thickness))
        elongation = arc_length / max(1.0, local_thickness)

        # Angle relative to vertical (0 deg = pure vertical, +/-90 deg = horizontal)
        vx = xs.astype(np.float64) - float(xs.mean())
        vy = ys.astype(np.float64) - float(ys.mean())
        mu20 = float(np.mean(vx * vx))
        mu02 = float(np.mean(vy * vy))
        mu11 = float(np.mean(vx * vy))
        angle_rad = 0.5 * np.arctan2(2.0 * mu11, mu02 - mu20)
        orientation_deg = float(np.degrees(angle_rad))

        thick_ok = (local_thickness <= cfg.max_local_thickness) or (
            local_thickness <= cfg.max_local_thickness_elongated and elongation >= 4.5
        )
        len_ok = arc_length >= cfg.min_line_length
        elong_ok = elongation >= cfg.min_elongation or arc_length >= cfg.tall_override_length

        if len_ok and thick_ok and elong_ok:
            # Keep dilated mask compact by keeping bridged component pixels intersected with dilated core
            out[y : y + ch, x : x + cw][sub > 0] = 255
            geom.append({
                "component_id": len(geom) + 1,
                "bbox": (x, y, cw, ch),
                "area_px": area,
                "orientation_deg": round(orientation_deg, 2),
                "local_thickness_px": round(local_thickness, 2),
                "arc_length_px": round(arc_length, 2),
                "elongation": round(elongation, 2),
            })

    return out, geom


def _precondition_gray_for_flow(
    gray: np.ndarray,
    c_mag: np.ndarray,
    best_dir: np.ndarray,
    contrast_th: float,
) -> np.ndarray:
    """Suppress narrow symmetric ridges before dense flow so DIS tracks scene motion, not scratches."""
    ridge_raw = (c_mag >= contrast_th).astype(np.uint8)
    if not np.any(ridge_raw):
        return gray
    ridge_mask = cv2.dilate(ridge_raw, np.ones((3, 3), dtype=np.uint8)) > 0
    ys, xs = np.nonzero(ridge_mask)
    if ys.size == 0:
        return gray
    h, w = gray.shape
    out = gray.copy()
    is_horiz = best_dir[ys, xs] == 1
    nh = ~is_horiz
    if np.any(nh):
        yh, xh = ys[nh], xs[nh]
        vals_h = np.stack([gray[yh, np.clip(xh + d, 0, w - 1)] for d in (-3, -2, -1, 0, 1, 2, 3)], axis=0)
        out[yh, xh] = np.partition(vals_h, 3, axis=0)[3]
    if np.any(is_horiz):
        yv, xv = ys[is_horiz], xs[is_horiz]
        vals_v = np.stack([gray[np.clip(yv + d, 0, h - 1), xv] for d in (-3, -2, -1, 0, 1, 2, 3)], axis=0)
        out[yv, xv] = np.partition(vals_v, 3, axis=0)[3]
    return out


class ExperimentalSceneFilmDetector:
    """Isolated motion-aware, coordinate-system-aware experimental defect detector."""

    def __init__(self, config: ExperimentalConfig | None = None) -> None:
        self.config = config or ExperimentalConfig()
        self._base_analyzer = TemporalScratchAnalyzer(
            TemporalConfig(
                window_radius=self.config.window_radius,
                min_temporal_neighbors=self.config.min_temporal_neighbors,
                flow_backend=self.config.flow_backend,
                telecine_dedup_threshold=self.config.telecine_dedup_threshold,
                sensitivity=self.config.sensitivity,
            )
        )
        self._dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        if self._dis_flow is not None:
            self._dis_flow.setFinestScale(1)

    def _compute_flow(self, ref_gray: np.ndarray, nbr_gray: np.ndarray) -> np.ndarray:
        if self.config.flow_backend == "dis" and self._dis_flow is not None:
            return self._dis_flow.calc(ref_gray, nbr_gray, None)
        return self._base_analyzer._compute_dense_flow(ref_gray, nbr_gray)

    def _warp_with_flow(self, img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        xg, yg = self._base_analyzer._get_remap_grid(h, w)
        return cv2.remap(img, xg + flow[..., 0], yg + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _weighted_background_and_spread(
        self,
        warped_stack: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Synthesize reliability-weighted background excluding unreliable occlusion warps."""
        k = warped_stack.shape[0]
        if k == 1:
            return warped_stack[0].copy(), np.zeros_like(warped_stack[0])
        if not self.config.use_motion_compensated_evidence:
            bg = self._base_analyzer._synthesize_background(warped_stack)
            spread = np.percentile(warped_stack, 75, axis=0) - np.percentile(warped_stack, 25, axis=0)
            return bg.astype(np.float32), spread.astype(np.float32)

        # Robust trimmed + reliability-weighted average: discard extreme min/max warps when K>=4
        order = np.argsort(warped_stack, axis=0)
        sorted_w = np.take_along_axis(warped_stack, order, axis=0).astype(np.float32)
        if k >= 4:
            w_trimmed = sorted_w[1:-1]
            r_trimmed = np.take_along_axis(weights, order[1:-1], axis=0)
        else:
            w_trimmed = sorted_w
            r_trimmed = weights

        med = np.median(w_trimmed, axis=0)
        intensity_w = np.exp(-0.5 * ((w_trimmed - med[None, ...]) / 18.0) ** 2).astype(np.float32)
        total_w = np.maximum(1e-4, r_trimmed * intensity_w)
        bg = np.sum(w_trimmed * total_w, axis=0) / np.sum(total_w, axis=0)

        q75_pos = 0.75 * (k - 1)
        q25_pos = 0.25 * (k - 1)
        i75, f75 = int(q75_pos), q75_pos - int(q75_pos)
        i25, f25 = int(q25_pos), q25_pos - int(q25_pos)
        p75 = sorted_w[i75] if f75 == 0.0 else (1.0 - f75) * sorted_w[i75] + f75 * sorted_w[i75 + 1]
        p25 = sorted_w[i25] if f25 == 0.0 else (1.0 - f25) * sorted_w[i25] + f25 * sorted_w[i25 + 1]
        spread = (p75 - p25).astype(np.float32)
        return bg.astype(np.float32), spread

    def analyze_frame(
        self,
        frames: Sequence[np.ndarray],
        target_idx: int,
        *,
        spatial_scratch_mask: np.ndarray | None = None,
        spatial_dust_mask: np.ndarray | None = None,
        _flow_cache: dict[tuple[int, int], np.ndarray] | None = None,
        _ridge_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
        _flow_gray_cache: dict[int, np.ndarray] | None = None,
    ) -> ExperimentalThreeStateResult:
        """Analyze one target frame and produce 3-state repair authorization masks."""
        cfg = self.config
        grays = [_ensure_gray(f) for f in frames]
        target_gray = grays[target_idx]
        h, w = target_gray.shape

        if h < 16 or w < 16:
            z = np.zeros((h, w), dtype=np.uint8)
            zf = np.zeros((h, w), dtype=np.float32)
            return ExperimentalThreeStateResult(
                high_confidence_damage=z,
                uncertain_damage_candidate=z.copy(),
                likely_intact_scene_structure=z.copy(),
                temporal_candidate_mask=z.copy(),
                background=target_gray.astype(np.float32),
                flow_mag=zf,
                fb_err=zf.copy(),
                fb_warning_map=np.zeros((h, w), dtype=bool),
                abstained=True,
                abstention_reason="small_spatial_dimension",
                valid_neighbors=[],
                skipped_duplicates=[],
            )

        prec, succ, skipped, cut_meta = select_motion_robust_neighbors(grays, target_idx, cfg)
        nbrs = prec + succ
        if len(nbrs) < cfg.min_temporal_neighbors:
            z = np.zeros((h, w), dtype=np.uint8)
            zf = np.zeros((h, w), dtype=np.float32)
            unc_fallback = np.zeros((h, w), dtype=np.uint8)
            if spatial_scratch_mask is not None:
                unc_fallback[spatial_scratch_mask > 0] = 255
            if spatial_dust_mask is not None:
                unc_fallback[spatial_dust_mask > 0] = 255
            return ExperimentalThreeStateResult(
                high_confidence_damage=z,
                uncertain_damage_candidate=unc_fallback,
                likely_intact_scene_structure=z.copy(),
                temporal_candidate_mask=z.copy(),
                background=target_gray.astype(np.float32),
                flow_mag=zf,
                fb_err=zf.copy(),
                fb_warning_map=np.zeros((h, w), dtype=bool),
                abstained=True,
                abstention_reason="insufficient_temporal_neighbors",
                valid_neighbors=nbrs,
                skipped_duplicates=skipped,
                metadata={"cut_selector": cut_meta},
            )

        contrast_th = cfg.contrast_base - 6.0 * (cfg.sensitivity - 0.5)
        ridge_cache = _ridge_cache if _ridge_cache is not None else {}
        flow_gray_cache = _flow_gray_cache if _flow_gray_cache is not None else {}
        flow_cache = _flow_cache if _flow_cache is not None else {}

        def _get_ridge(i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            if i in ridge_cache:
                return ridge_cache[i]
            s_i16 = _compute_oriented_signed_channels_i16(grays[i])
            abs_ch = np.abs(s_i16)
            bd_u8 = np.argmax(abs_ch, axis=0).astype(np.uint8)
            yy_r, xx_r = np.ogrid[:h, :w]
            best_s = s_i16[bd_u8, yy_r, xx_r]
            mm = np.abs(best_s).astype(np.float32)
            bs_i8 = np.where(best_s >= 0, 1, -1).astype(np.int8)
            packed = (s_i16, mm, bd_u8, bs_i8)
            ridge_cache[i] = packed
            return packed

        def _get_flow_gray(i: int) -> np.ndarray:
            if not cfg.use_motion_compensated_evidence:
                return grays[i]
            if i in flow_gray_cache:
                return flow_gray_cache[i]
            _, cm, bd, _ = _get_ridge(i)
            fg = _precondition_gray_for_flow(grays[i], cm, bd, contrast_th)
            flow_gray_cache[i] = fg
            return fg

        def _get_flow(i: int, j: int) -> np.ndarray:
            key = (i, j)
            if key in flow_cache:
                return flow_cache[key].astype(np.float32)
            fl = self._compute_flow(_get_flow_gray(i), _get_flow_gray(j))
            flow_cache[key] = fl.astype(np.float16)
            return fl

        xg, yg = self._base_analyzer._get_remap_grid(h, w)
        k_nbrs = len(nbrs)
        w_list: list[np.ndarray] = []
        rel_list: list[np.ndarray] = []
        fb_list: list[np.ndarray] = []
        reliable_nbrs: list[int] = []
        rejected_neighbors: list[dict[str, Any]] = []
        in_bounds_count = np.zeros((h, w), dtype=np.int16)
        flow_x_sum = np.zeros((h, w), dtype=np.float32)
        flow_y_sum = np.zeros((h, w), dtype=np.float32)
        flow_mag_sum = np.zeros((h, w), dtype=np.float32)
        flank_mag_sum = np.zeros((h, w), dtype=np.float32)

        # Decouple M3 (orientation-aware ridge geometry) from M1 (motion-compensated temporal evidence)
        if cfg.use_centerline_thickness_geometry:
            _t_signed, c_target, best_dir, best_sign = _get_ridge(target_idx)
        else:
            _t_signed, c_target, best_dir, best_sign = _compute_legacy_vertical_flank_features(target_gray, self._base_analyzer)
        struct_ok = c_target >= contrast_th

        fd = min(32, max(16, w // 7))
        fb_nbr_ok_th = max(2.0, cfg.fb_warning_threshold * 1.1)
        is_vert = best_dir == 0
        is_horiz = best_dir == 1

        for n_idx in nbrs:
            fl_bw = _get_flow(target_idx, n_idx)
            local_mag = np.hypot(fl_bw[..., 0], fl_bw[..., 1])

            if cfg.use_motion_compensated_evidence:
                pad_x = np.pad(fl_bw, ((0, 0), (fd, fd), (0, 0)), mode="edge")
                u_l, u_r = pad_x[:, :w], pad_x[:, 2 * fd : 2 * fd + w]
                pad_y = np.pad(fl_bw, ((fd, fd), (0, 0), (0, 0)), mode="edge")
                u_u, u_d = pad_y[:h, :], pad_y[2 * fd : 2 * fd + h, :]

                mag_l = np.hypot(u_l[..., 0], u_l[..., 1])
                mag_r = np.hypot(u_r[..., 0], u_r[..., 1])
                mag_u = np.hypot(u_u[..., 0], u_u[..., 1])
                mag_d = np.hypot(u_d[..., 0], u_d[..., 1])

                diff_lr = np.hypot(u_l[..., 0] - u_r[..., 0], u_l[..., 1] - u_r[..., 1])
                diff_ud = np.hypot(u_u[..., 0] - u_d[..., 0], u_u[..., 1] - u_d[..., 1])

                fl_h = 0.5 * (u_l + u_r)
                fl_v = 0.5 * (u_u + u_d)
                mag_h_raw = np.hypot(fl_h[..., 0], fl_h[..., 1])
                mag_v_raw = np.hypot(fl_v[..., 0], fl_v[..., 1])

                # Require bilateral moving flanks (prevents a single adjacent moving object from
                # dragging a stationary structural edge like a wooden wall seam) and align with line normal.
                coh_h = (np.minimum(mag_l, mag_r) >= 2.2) & (diff_lr <= np.maximum(2.2, 0.35 * np.maximum(mag_l, mag_r)))
                coh_v = (np.minimum(mag_u, mag_d) >= 2.2) & (diff_ud <= np.maximum(2.2, 0.35 * np.maximum(mag_u, mag_d)))
                coh_h = coh_h & (~is_horiz)
                coh_v = coh_v & (~is_vert)

                mag_h = np.where(coh_h, mag_h_raw, 0.0)
                mag_v = np.where(coh_v, mag_v_raw, 0.0)
                use_h = mag_h >= mag_v
                fl_flank = np.where(use_h[..., None], fl_h, fl_v).astype(np.float32)
                mag_flank = np.maximum(mag_h, mag_v).astype(np.float32)

                # Un-lock narrow linear ridges only where BOTH orthogonal flanks coherently move
                unlock = struct_ok & (mag_flank >= 3.0) & (local_mag < 0.45 * mag_flank)
                fl_eff = np.where(unlock[..., None], fl_flank, fl_bw).astype(np.float32)
                del pad_x, pad_y, fl_h, fl_v, fl_flank, mag_h, mag_v, mag_l, mag_r, mag_u, mag_d
            else:
                fl_eff = fl_bw
                mag_flank = local_mag.astype(np.float32)

            wx = xg + fl_eff[..., 0]
            wy = yg + fl_eff[..., 1]
            in_b = (wx >= 0.0) & (wx <= float(w - 1)) & (wy >= 0.0) & (wy <= float(h - 1))
            warped_n = cv2.remap(grays[n_idx], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            if cfg.use_motion_compensated_evidence:
                fl_fw = _get_flow(n_idx, target_idx)
                fw_x = cv2.remap(fl_fw[..., 0], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                fw_y = cv2.remap(fl_fw[..., 1], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                err = np.hypot(fl_eff[..., 0] + fw_x, fl_eff[..., 1] + fw_y).astype(np.float32)
                inb_frac = float(np.mean(in_b))
                if int(np.count_nonzero(in_b)) > 0:
                    nbr_fb_med = float(np.median(err[in_b]))
                    nbr_fb_ok = float(np.mean(err[in_b] <= fb_nbr_ok_th))
                    nbr_wmad = float(np.mean(np.abs(target_gray[in_b].astype(np.float32) - warped_n[in_b].astype(np.float32))))
                else:
                    nbr_fb_med = 999.0
                    nbr_fb_ok = 0.0
                    nbr_wmad = 255.0

                nbr_ok = (
                    inb_frac >= cfg.min_neighbor_in_bounds_frac
                    and nbr_fb_med <= cfg.max_neighbor_fb_med
                    and nbr_fb_ok >= cfg.min_neighbor_fb_ok_frac
                    and nbr_wmad <= cfg.max_neighbor_warped_mad
                )
                if not nbr_ok:
                    rejected_neighbors.append({
                        "frame_idx": n_idx,
                        "inb_frac": round(inb_frac, 3),
                        "fb_med": round(nbr_fb_med, 2),
                        "fb_ok_frac": round(nbr_fb_ok, 3),
                        "warped_mad": round(nbr_wmad, 2),
                    })
                    continue

                rel_n = np.exp(-((err / max(0.5, cfg.fb_sigma)) ** 2)) * in_b.astype(np.float32)
            else:
                err = np.zeros((h, w), dtype=np.float32)
                rel_n = np.ones((h, w), dtype=np.float32)

            reliable_nbrs.append(n_idx)
            w_list.append(warped_n)
            fb_list.append(err)
            rel_list.append(rel_n)
            in_bounds_count += in_b
            flow_x_sum += fl_eff[..., 0]
            flow_y_sum += fl_eff[..., 1]
            flow_mag_sum += np.hypot(fl_eff[..., 0], fl_eff[..., 1])
            flank_mag_sum += mag_flank

        k_nbrs = len(reliable_nbrs)
        if k_nbrs < cfg.min_temporal_neighbors:
            z = np.zeros((h, w), dtype=np.uint8)
            zf = np.zeros((h, w), dtype=np.float32)
            unc_fallback = np.zeros((h, w), dtype=np.uint8)
            if spatial_scratch_mask is not None:
                unc_fallback[spatial_scratch_mask > 0] = 255
            if spatial_dust_mask is not None:
                unc_fallback[spatial_dust_mask > 0] = 255
            fb_ret = np.median(np.stack(fb_list, axis=0), axis=0).astype(np.float32) if fb_list else zf.copy()
            return ExperimentalThreeStateResult(
                high_confidence_damage=z,
                uncertain_damage_candidate=unc_fallback,
                likely_intact_scene_structure=z.copy(),
                temporal_candidate_mask=z.copy(),
                background=target_gray.astype(np.float32),
                flow_mag=zf,
                fb_err=fb_ret,
                fb_warning_map=np.ones((h, w), dtype=bool),
                abstained=True,
                abstention_reason="unreliable_temporal_registration",
                valid_neighbors=reliable_nbrs,
                skipped_duplicates=skipped,
                metadata={
                    "tau": 0.0,
                    "sigma_mad": 0.0,
                    "contrast_th": round(contrast_th, 3),
                    "cut_selector": cut_meta,
                    "rejected_neighbors": rejected_neighbors,
                    "high_confidence_pixels": 0,
                    "uncertain_pixels": int(np.count_nonzero(unc_fallback)),
                    "vetoed_intact_pixels": 0,
                },
            )

        w_stack = np.stack(w_list, axis=0)
        rel_stack = np.stack(rel_list, axis=0)
        fb_stack = np.stack(fb_list, axis=0)
        del w_list, rel_list, fb_list
        nbrs = reliable_nbrs

        bg_float, nbr_iqr = self._weighted_background_and_spread(w_stack, rel_stack)
        del rel_stack
        r_scene = np.abs(target_gray.astype(np.float32) - bg_float)
        med_r = float(np.median(r_scene))
        mad_r = float(np.median(np.abs(r_scene - med_r)))
        sigma_mad = 1.4826 * max(mad_r, 1.0)
        k_mult = cfg.k_factor - 0.8 * (cfg.sensitivity - 0.5)
        tau = max(6.5, med_r + k_mult * sigma_mad) if cfg.use_motion_compensated_evidence else (med_r + k_mult * sigma_mad)
        z_scene = (r_scene - med_r) / max(sigma_mad, 1.0)

        inv_k = np.float32(1.0 / max(1, k_nbrs))
        mean_fx = (flow_x_sum * inv_k).astype(np.float32)
        mean_fy = (flow_y_sum * inv_k).astype(np.float32)
        mean_flow_mag = (flow_mag_sum * inv_k).astype(np.float32)
        surround_flow_mag = np.maximum(flank_mag_sum * inv_k, mean_flow_mag).astype(np.float32)
        del flow_x_sum, flow_y_sum, flow_mag_sum, flank_mag_sum

        median_fb_err = np.median(fb_stack, axis=0).astype(np.float32)
        if k_nbrs >= 2:
            second_min_fb = np.partition(fb_stack, 1, axis=0)[1].astype(np.float32)
        else:
            second_min_fb = np.min(fb_stack, axis=0).astype(np.float32)
        min_in_bounds = min(k_nbrs, max(2, (k_nbrs + 1) // 2))
        oob_warning_map = in_bounds_count < min_in_bounds
        fb_warning_map = ((median_fb_err > cfg.fb_warning_threshold) & (second_min_fb > 1.5)) | oob_warning_map
        del fb_stack, second_min_fb, in_bounds_count

        global_flow_med = float(np.median(mean_flow_mag))
        global_fb_med = float(np.median(median_fb_err))
        global_fb_ok_frac = float(1.0 - np.mean(fb_warning_map))
        if cfg.use_motion_compensated_evidence and (
            global_fb_med > cfg.max_global_fb_med or global_fb_ok_frac < cfg.min_global_fb_ok_frac
        ):
            z = np.zeros((h, w), dtype=np.uint8)
            unc_fallback = np.zeros((h, w), dtype=np.uint8)
            if spatial_scratch_mask is not None:
                unc_fallback[spatial_scratch_mask > 0] = 255
            if spatial_dust_mask is not None:
                unc_fallback[spatial_dust_mask > 0] = 255
            raw_cand_unc = (r_scene >= tau) & struct_ok
            unc_fallback[raw_cand_unc] = 255
            return ExperimentalThreeStateResult(
                high_confidence_damage=z,
                uncertain_damage_candidate=unc_fallback,
                likely_intact_scene_structure=z.copy(),
                temporal_candidate_mask=z.copy(),
                background=bg_float,
                flow_mag=mean_flow_mag,
                fb_err=median_fb_err,
                fb_warning_map=fb_warning_map,
                abstained=True,
                abstention_reason="global_flow_registration_unreliable",
                valid_neighbors=nbrs,
                skipped_duplicates=skipped,
                metadata={
                    "tau": round(tau, 3),
                    "sigma_mad": round(sigma_mad, 3),
                    "contrast_th": round(contrast_th, 3),
                    "cut_selector": cut_meta,
                    "rejected_neighbors": rejected_neighbors,
                    "global_fb_med": round(global_fb_med, 3),
                    "global_fb_ok_frac": round(global_fb_ok_frac, 4),
                    "high_confidence_pixels": 0,
                    "uncertain_pixels": int(np.count_nonzero(unc_fallback)),
                    "vetoed_intact_pixels": 0,
                },
            )
        yy, xx = np.ogrid[:h, :w]

        # Build aligned orientation+polarity-matched ridge stack in scene & film coordinates
        k_tol1 = np.ones((3, 3), dtype=np.uint8)
        k_weave = np.ones((2 * cfg.film_weave_y + 1, 2 * cfg.film_weave_x + 1), dtype=np.uint8)
        best_sign_i8 = best_sign.astype(np.int8)

        if cfg.use_motion_compensated_evidence:
            c_scene_stack = np.empty((k_nbrs, h, w), dtype=np.uint8)
            c_film_stack = np.empty((k_nbrs, h, w), dtype=np.uint8)
            for k_i, n_idx in enumerate(nbrs):
                # Scene coordinate ridge on warped neighbor (uint8 via int16 ridge op)
                w_signed = _compute_oriented_signed_channels_i16(w_stack[k_i])
                w_matched = np.clip(best_sign_i8 * w_signed[best_dir, yy, xx], 0, 255).astype(np.uint8)
                # +/-1 px tolerance absorbs subpixel DIS warp softness on static pipes/planks
                c_scene_stack[k_i] = cv2.dilate(w_matched, k_tol1)

                # Film coordinate ridge: unwarped neighbor within gate weave +/-2 px horizontal, +/-1 px vertical
                n_signed, _, _, _ = _get_ridge(n_idx)
                n_matched = np.clip(best_sign_i8 * n_signed[best_dir, yy, xx], 0, 255).astype(np.uint8)
                c_film_stack[k_i] = cv2.dilate(n_matched, k_weave)

            c_scene_med = np.median(c_scene_stack, axis=0).astype(np.float32)
            c_film_med = np.median(c_film_stack, axis=0).astype(np.float32)
            if k_nbrs >= 4:
                c_scene_q2 = np.partition(c_scene_stack, 1, axis=0)[1].astype(np.float32)
            else:
                c_scene_q2 = c_scene_med
            del c_scene_stack, c_film_stack
        else:
            c_scene_med = np.zeros((h, w), dtype=np.float32)
            c_scene_q2 = np.zeros((h, w), dtype=np.float32)
            c_film_med = np.zeros((h, w), dtype=np.float32)
        del w_stack

        # Background gradient on reliability-weighted MC background
        gx = cv2.Sobel(bg_float, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(bg_float, cv2.CV_32F, 0, 1, ksize=3)
        bg_grad = cv2.magnitude(gx, gy)
        # Local 3x3 max gradient ensures edge halo doesn't slip beside an intact scene edge
        bg_grad_max3 = cv2.dilate(bg_grad, k_tol1)

        # For strong film-locked candidates whose ridge tangent is parallel to the camera pan vector
        # (e.g. horizontal scratch under horizontal pan, where the scratch overlaps itself across neighbors),
        # allow 2nd-smallest warped scene check when clean frames exist in the window and bg_grad isn't dominant.
        abs_fx = np.abs(mean_fx)
        abs_fy = np.abs(mean_fy)
        pan_parallel = (
            ((best_dir == 0) & (abs_fy >= 2.2 * np.maximum(abs_fx, 0.5)))
            | ((best_dir == 1) & (abs_fx >= 2.2 * np.maximum(abs_fy, 0.5)))
            | ((best_dir == 2) & (np.abs(mean_fx - mean_fy) <= 0.45 * mean_flow_mag))
            | ((best_dir == 3) & (np.abs(mean_fx + mean_fy) <= 0.45 * mean_flow_mag))
        )
        strong_film_moving = (
            pan_parallel
            & (c_film_med >= 0.60 * np.maximum(c_target, 1.0))
            & (surround_flow_mag >= 2.5)
            & (r_scene >= max(tau, 12.0))
            & (r_scene >= 0.45 * bg_grad_max3)
        )
        c_scene_eff = np.where(strong_film_moving, np.minimum(c_scene_med, c_scene_q2 * 1.15), c_scene_med).astype(np.float32)

        flank_novelty = np.maximum(0.0, c_target - c_scene_eff).astype(np.float32)

        # Raw pixel candidates for temporal scratch stream (struct_ok & contrast_th already computed above)
        raw_temp = r_scene >= tau
        if cfg.use_motion_compensated_evidence:
            # Allow transient/film-locked novel ridges without sweeping static low-flow background texture into transient blobs
            pixel_novel = (flank_novelty >= 8.0) & (c_scene_eff <= np.maximum(5.0, 0.25 * c_target))
            flow_locked_film_hint = (
                (mean_flow_mag < 1.2)
                & (surround_flow_mag >= 2.5)
                & (c_film_med >= 0.50 * np.maximum(c_target, 1.0))
                & (c_scene_eff <= np.maximum(6.0, 0.30 * c_target))
                & struct_ok
            )
            temporal_raw_cand = (raw_temp & struct_ok & pixel_novel) | flow_locked_film_hint
        else:
            motion_gate = mean_flow_mag <= cfg.legacy_motion_coherence_threshold
            temporal_raw_cand = raw_temp & struct_ok & motion_gate

        geom_mask, geom_records = extract_orientation_aware_components(
            temporal_raw_cand,
            cfg,
            best_dir=best_dir if cfg.use_centerline_thickness_geometry else None,
        )
        geom_by_bbox = {g["bbox"]: g for g in geom_records}

        high_conf = np.zeros((h, w), dtype=np.uint8)
        uncertain = np.zeros((h, w), dtype=np.uint8)
        vetoed_intact = np.zeros((h, w), dtype=np.uint8)
        comp_records: list[ComponentRecord] = []

        # Helper to score component scene-vs-film statistics & classify using ROI bbox slices
        def _measure_component(sub_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
            bx, by, bw, bh = bbox
            sl = (slice(by, by + bh), slice(bx, bx + bw))
            c_t = float(np.median(c_target[sl][sub_mask]))
            c_sc = float(np.median(c_scene_eff[sl][sub_mask]))
            c_fl = float(np.median(c_film_med[sl][sub_mask]))
            fl_n = float(np.median(flank_novelty[sl][sub_mask]))
            r_m = float(np.median(r_scene[sl][sub_mask]))
            z_m = float(np.median(z_scene[sl][sub_mask]))
            iqr_m = float(np.median(nbr_iqr[sl][sub_mask]))
            f_m = float(np.median(mean_flow_mag[sl][sub_mask]))
            sf_m = float(np.median(surround_flow_mag[sl][sub_mask]))
            fb_m = float(np.median(median_fb_err[sl][sub_mask]))
            fb_frac = float(np.mean(fb_warning_map[sl][sub_mask]))
            oob_frac = float(np.mean(oob_warning_map[sl][sub_mask]))
            bg_g = float(np.median(bg_grad_max3[sl][sub_mask]))

            scene_pers = c_sc / max(c_t, 1.0)
            film_pers = c_fl / max(c_t, 1.0)
            follows_scene = bool(
                max(f_m, sf_m) >= 1.5 and scene_pers >= 0.42 and c_sc >= 1.15 * max(c_fl, 1.0)
            )
            eff_flow = max(f_m, 0.8 * sf_m)

            if (eff_flow < 1.8 and scene_pers >= 0.35 and film_pers >= 0.35) or (
                global_flow_med < 1.2 and film_pers >= 0.45
            ):
                t_cls = "ambiguous_static_persistent"
            elif follows_scene or (scene_pers >= 0.42 and r_m < 1.35 * max(c_sc, 4.0)):
                t_cls = "persistent_scene_locked"
            elif (f_m >= 1.6 or sf_m >= 2.2) and film_pers >= 0.45 and scene_pers < 0.42:
                t_cls = "persistent_film_locked"
            elif scene_pers < 0.38 and film_pers < 0.45 and z_m >= 1.8:
                t_cls = "transient_defect"
            else:
                t_cls = "ambiguous_static_persistent"

            ginfo = geom_by_bbox.get(bbox)
            if ginfo is None:
                _a = int(np.count_nonzero(sub_mask))
                ginfo = {
                    "orientation_deg": 0.0,
                    "local_thickness_px": round(float(_a) / max(1, bh), 2),
                    "arc_length_px": float(max(bw, bh)),
                    "elongation": round(float(max(bw, bh)) / max(1.0, float(_a) / max(1, max(bw, bh))), 2),
                }

            return {
                "c_target_med": c_t,
                "c_scene_med": c_sc,
                "c_film_med": c_fl,
                "flank_novelty_med": fl_n,
                "r_scene_med": r_m,
                "z_scene_med": z_m,
                "iqr_med": iqr_m,
                "flow_mag_med": f_m,
                "surround_flow_med": sf_m,
                "eff_flow_med": eff_flow,
                "fb_err_med": fb_m,
                "fb_warn_frac": fb_frac,
                "oob_warn_frac": oob_frac,
                "bg_grad_med": bg_g,
                "scene_persistence_ratio": scene_pers,
                "film_persistence_ratio": film_pers,
                "follows_scene_motion": follows_scene,
                "temporal_class": t_cls,
                "geom": ginfo,
            }

        # Policy 1: Temporal Scratch Components
        if np.any(geom_mask > 0):
            n_t, lbl_t, stats_t, _ = cv2.connectedComponentsWithStats(geom_mask, connectivity=8)
            for i in range(1, n_t):
                x, y, cw, ch, area = map(int, stats_t[i])
                sl = (slice(y, y + ch), slice(x, x + cw))
                sub_m = lbl_t[sl] == i
                if not cfg.use_motion_compensated_evidence:
                    # Legacy behavior when M1 is turned off in factorial ablation
                    high_conf[sl][sub_m] = 255
                    continue

                m = _measure_component(sub_m, (x, y, cw, ch))
                fb_warn = (
                    m["oob_warn_frac"] >= 0.25
                    or m["fb_warn_frac"] >= 0.30
                    or m["fb_err_med"] > 2.8
                    or (m["fb_err_med"] > cfg.fb_warning_threshold and m["fb_warn_frac"] >= 0.20)
                )
                t_cls = m["temporal_class"]

                if t_cls == "persistent_scene_locked":
                    decision = "likely_intact_scene_structure"
                    reason = "scene_locked_structure"
                    vetoed_intact[sl][sub_m] = 255
                elif t_cls == "ambiguous_static_persistent":
                    # Truly static camera (<0.45 px): cannot resolve from motion alone -> preserve uncertainty
                    if m["eff_flow_med"] < 0.45:
                        decision = "uncertain_damage_candidate"
                        reason = "ambiguous_static_or_low_motion"
                        uncertain[sl][sub_m] = 255
                    elif m["scene_persistence_ratio"] >= 0.42 or (
                        m["bg_grad_med"] >= 18.0 and m["r_scene_med"] < 0.65 * m["bg_grad_med"]
                    ):
                        decision = "likely_intact_scene_structure"
                        reason = "static_persistent_edge_in_bg"
                        vetoed_intact[sl][sub_m] = 255
                    else:
                        decision = "uncertain_damage_candidate"
                        reason = "ambiguous_static_or_low_motion"
                        uncertain[sl][sub_m] = 255
                elif t_cls == "persistent_film_locked":
                    extreme_fb_override = (
                        global_fb_ok_frac >= cfg.min_override_global_fb_ok_frac
                        and global_fb_med <= cfg.max_override_global_fb_med
                        and m["oob_warn_frac"] < 0.25
                        and m["fb_err_med"] <= cfg.max_override_fb_err
                        and m["r_scene_med"] >= 70.0
                        and m["flank_novelty_med"] >= 60.0
                        and m["c_scene_med"] <= max(4.5, 0.18 * m["c_target_med"])
                        and m["scene_persistence_ratio"] <= 0.18
                        and m["film_persistence_ratio"] >= 0.48
                        and not (m["bg_grad_med"] >= 28.0 and m["r_scene_med"] < 1.8 * m["bg_grad_med"])
                    )
                    if fb_warn and not extreme_fb_override:
                        decision = "uncertain_damage_candidate"
                        reason = "film_locked_fb_warning"
                        uncertain[sl][sub_m] = 255
                    elif (
                        m["z_scene_med"] >= 2.0
                        and m["r_scene_med"] >= max(tau, 9.0)
                        and m["c_scene_med"] <= max(4.5, 0.24 * m["c_target_med"])
                        and m["scene_persistence_ratio"] <= 0.28
                        and m["flank_novelty_med"] >= 8.5
                        and not (m["bg_grad_med"] >= 22.0 and m["scene_persistence_ratio"] >= 0.20)
                        and not (m["bg_grad_med"] >= 18.0 and m["r_scene_med"] < 0.35 * m["bg_grad_med"])
                    ):
                        decision = "high_confidence_damage"
                        reason = "film_locked_persistent_scratch"
                        high_conf[sl][sub_m] = 255
                    elif m["r_scene_med"] >= max(6.5, 0.9 * tau) and m["flank_novelty_med"] >= 6.0:
                        decision = "uncertain_damage_candidate"
                        reason = "film_locked_needs_review"
                        uncertain[sl][sub_m] = 255
                    else:
                        decision = "likely_intact_scene_structure"
                        reason = "film_locked_low_residual_noise"
                        vetoed_intact[sl][sub_m] = 255
                else:  # transient_defect
                    extreme_fb_override = (
                        global_fb_ok_frac >= cfg.min_override_global_fb_ok_frac
                        and global_fb_med <= cfg.max_override_global_fb_med
                        and m["oob_warn_frac"] < 0.25
                        and m["fb_err_med"] <= cfg.max_override_fb_err
                        and m["r_scene_med"] >= 70.0
                        and m["flank_novelty_med"] >= 60.0
                        and m["c_scene_med"] <= max(4.5, 0.18 * m["c_target_med"])
                        and not (m["bg_grad_med"] >= 28.0 and m["r_scene_med"] < 1.8 * m["bg_grad_med"])
                    )
                    if fb_warn and not extreme_fb_override:
                        decision = "uncertain_damage_candidate"
                        reason = "fb_inconsistency_warning"
                        uncertain[sl][sub_m] = 255
                    elif (
                        m["r_scene_med"] >= max(tau, 9.0)
                        and m["flank_novelty_med"] >= 9.0
                        and m["c_scene_med"] <= max(4.5, 0.22 * m["c_target_med"])
                        and m["scene_persistence_ratio"] <= 0.26
                        and m["iqr_med"] <= max(10.0, 0.50 * m["r_scene_med"])
                        and m["z_scene_med"] >= 2.2
                        and not (m["bg_grad_med"] >= 18.0 and m["r_scene_med"] < 0.52 * m["bg_grad_med"])
                    ):
                        decision = "high_confidence_damage"
                        reason = "transient_novel_ridge"
                        high_conf[sl][sub_m] = 255
                    elif m["r_scene_med"] >= max(6.5, 0.9 * tau) and m["flank_novelty_med"] >= 6.0:
                        decision = "uncertain_damage_candidate"
                        reason = "transient_moderate_confidence"
                        uncertain[sl][sub_m] = 255
                    else:
                        decision = "likely_intact_scene_structure"
                        reason = "transient_low_residual_noise"
                        vetoed_intact[sl][sub_m] = 255

                comp_records.append(ComponentRecord(
                    stream="temporal_scratch",
                    component_id=i,
                    area_px=area,
                    orientation_deg=m["geom"]["orientation_deg"],
                    local_thickness_px=m["geom"]["local_thickness_px"],
                    arc_length_px=m["geom"]["arc_length_px"],
                    elongation=m["geom"]["elongation"],
                    mean_flow_mag_px=round(m["flow_mag_med"], 3),
                    surround_flow_mag_px=round(m["surround_flow_med"], 3),
                    median_fb_err_px=round(m["fb_err_med"], 3),
                    fb_warning=fb_warn,
                    scene_persistence_ratio=round(m["scene_persistence_ratio"], 3),
                    film_persistence_ratio=round(m["film_persistence_ratio"], 3),
                    flank_novelty_med=round(m["flank_novelty_med"], 2),
                    r_scene_med=round(m["r_scene_med"], 2),
                    z_scene_med=round(m["z_scene_med"], 2),
                    bg_grad_med=round(m["bg_grad_med"], 2),
                    follows_scene_motion=m["follows_scene_motion"],
                    temporal_class=t_cls,
                    decision=decision,
                    reason=reason,
                ))

        # Static-camera persistent ridge preservation: surface stationary linear candidates
        # into `uncertain_damage_candidate` (never auto-repaired, never merged into transient blobs).
        if cfg.use_motion_compensated_evidence:
            static_pers_cand = (
                (mean_flow_mag < 0.45)
                & (surround_flow_mag < 0.65)
                & (c_film_med >= 0.55 * np.maximum(c_target, 1.0))
                & (c_target >= max(contrast_th, 16.0))
                & (geom_mask == 0)
            )
            if np.any(static_pers_cand):
                st_geom, _ = extract_orientation_aware_components(
                    static_pers_cand,
                    cfg,
                    best_dir=best_dir if cfg.use_centerline_thickness_geometry else None,
                )
                if np.any(st_geom > 0):
                    uncertain[st_geom > 0] = 255

        # Policy 2: Spatial Scratch Components (separate scale-invariant policy — no blind global veto)
        if cfg.use_motion_compensated_evidence and spatial_scratch_mask is not None and np.any(spatial_scratch_mask > 0):
            s_scr_u8 = (spatial_scratch_mask > 0).astype(np.uint8) * 255
            n_ss, lbl_ss, stats_ss, _ = cv2.connectedComponentsWithStats(s_scr_u8, connectivity=8)
            for i in range(1, n_ss):
                x, y, cw, ch, area = map(int, stats_ss[i])
                sl = (slice(y, y + ch), slice(x, x + cw))
                sub_m = lbl_ss[sl] == i
                m = _measure_component(sub_m, (x, y, cw, ch))
                fb_warn = (
                    m["oob_warn_frac"] >= 0.25
                    or m["fb_warn_frac"] >= 0.30
                    or m["fb_err_med"] > 2.8
                    or (m["fb_err_med"] > cfg.fb_warning_threshold and m["fb_warn_frac"] >= 0.20)
                )
                t_cls = m["temporal_class"]

                if t_cls == "persistent_scene_locked" or m["scene_persistence_ratio"] >= 0.42 or m["r_scene_med"] < max(tau, 5.0):
                    if m["eff_flow_med"] < 0.45 and m["film_persistence_ratio"] >= 0.45:
                        decision = "uncertain_damage_candidate"
                        reason = "spatial_scratch_ambiguous_static"
                        uncertain[sl][sub_m] = 255
                    else:
                        decision = "likely_intact_scene_structure"
                        reason = "spatial_scratch_matches_scene_edge"
                        vetoed_intact[sl][sub_m] = 255
                elif t_cls == "ambiguous_static_persistent":
                    if m["eff_flow_med"] < 0.45 and m["film_persistence_ratio"] >= 0.45:
                        decision = "uncertain_damage_candidate"
                        reason = "spatial_scratch_ambiguous_static"
                        uncertain[sl][sub_m] = 255
                    elif m["scene_persistence_ratio"] >= 0.38 or (
                        m["bg_grad_med"] >= 18.0 and m["r_scene_med"] < 0.65 * m["bg_grad_med"]
                    ):
                        decision = "likely_intact_scene_structure"
                        reason = "spatial_scratch_matches_scene_edge"
                        vetoed_intact[sl][sub_m] = 255
                    else:
                        decision = "uncertain_damage_candidate"
                        reason = "spatial_scratch_ambiguous_static"
                        uncertain[sl][sub_m] = 255
                else:
                    if (m["bg_grad_med"] >= 22.0 and m["scene_persistence_ratio"] >= 0.22) or (
                        m["bg_grad_med"] >= 18.0 and m["r_scene_med"] < 0.35 * m["bg_grad_med"]
                    ):
                        decision = "likely_intact_scene_structure"
                        reason = "spatial_scratch_matches_scene_edge"
                        vetoed_intact[sl][sub_m] = 255
                    elif fb_warn and not (
                        global_fb_ok_frac >= cfg.min_override_global_fb_ok_frac
                        and global_fb_med <= cfg.max_override_global_fb_med
                        and m["oob_warn_frac"] < 0.25
                        and m["fb_err_med"] <= cfg.max_override_fb_err
                        and m["r_scene_med"] >= 60.0
                        and m["flank_novelty_med"] >= 50.0
                        and m["c_scene_med"] <= max(4.5, 0.18 * m["c_target_med"])
                        and not (m["bg_grad_med"] >= 28.0 and m["r_scene_med"] < 1.8 * m["bg_grad_med"])
                    ):
                        decision = "uncertain_damage_candidate"
                        reason = "spatial_scratch_fb_warning"
                        uncertain[sl][sub_m] = 255
                    elif (
                        m["z_scene_med"] >= 2.2
                        and m["r_scene_med"] >= max(tau, 8.0)
                        and m["c_scene_med"] <= max(4.5, 0.24 * m["c_target_med"])
                    ):
                        decision = "high_confidence_damage"
                        reason = "spatial_scratch_temporally_corroborated"
                        # Guard against fat spatial masks bleeding onto clean zero-residual flanks
                        px_ok = sub_m & (r_scene[sl] >= max(5.0, 0.75 * tau))
                        if np.any(px_ok):
                            high_conf[sl][px_ok] = 255
                        veto_rem = sub_m & ~px_ok
                        if np.any(veto_rem):
                            vetoed_intact[sl][veto_rem] = 255
                    else:
                        decision = "uncertain_damage_candidate"
                        reason = "spatial_scratch_uncorroborated"
                        uncertain[sl][sub_m] = 255

                comp_records.append(ComponentRecord(
                    stream="spatial_scratch",
                    component_id=i,
                    area_px=area,
                    orientation_deg=m["geom"]["orientation_deg"],
                    local_thickness_px=m["geom"]["local_thickness_px"],
                    arc_length_px=m["geom"]["arc_length_px"],
                    elongation=m["geom"]["elongation"],
                    mean_flow_mag_px=round(m["flow_mag_med"], 3),
                    surround_flow_mag_px=round(m["surround_flow_med"], 3),
                    median_fb_err_px=round(m["fb_err_med"], 3),
                    fb_warning=fb_warn,
                    scene_persistence_ratio=round(m["scene_persistence_ratio"], 3),
                    film_persistence_ratio=round(m["film_persistence_ratio"], 3),
                    flank_novelty_med=round(m["flank_novelty_med"], 2),
                    r_scene_med=round(m["r_scene_med"], 2),
                    z_scene_med=round(m["z_scene_med"], 2),
                    bg_grad_med=round(m["bg_grad_med"], 2),
                    follows_scene_motion=m["follows_scene_motion"],
                    temporal_class=t_cls,
                    decision=decision,
                    reason=reason,
                ))

        # Policy 3: Spatial Dust Components (separate blotch/opacity policy — NOT ridge novelty!)
        if cfg.use_motion_compensated_evidence and spatial_dust_mask is not None and np.any(spatial_dust_mask > 0):
            s_dst_u8 = (spatial_dust_mask > 0).astype(np.uint8) * 255
            n_sd, lbl_sd, stats_sd, _ = cv2.connectedComponentsWithStats(s_dst_u8, connectivity=8)
            for i in range(1, n_sd):
                x, y, cw, ch, area = map(int, stats_sd[i])
                sl = (slice(y, y + ch), slice(x, x + cw))
                sub_m = lbl_sd[sl] == i
                m = _measure_component(sub_m, (x, y, cw, ch))
                fb_warn = (
                    m["oob_warn_frac"] >= 0.25
                    or m["fb_warn_frac"] >= 0.30
                    or m["fb_err_med"] > 2.8
                    or (m["fb_err_med"] > cfg.fb_warning_threshold and m["fb_warn_frac"] >= 0.20)
                )

                # Dust/blotch criterion: must be transient opacity outlier vs B_scene and not a persistent crack/edge in B_scene.
                # Unlike bright long scratches, compact dust/blotches with FB flow warning MUST fail closed to Uncertain.
                if (
                    m["r_scene_med"] < max(tau, 5.0)
                    or (
                        m["bg_grad_med"] >= 14.0
                        and m["r_scene_med"] < 0.85 * m["bg_grad_med"]
                        and m["scene_persistence_ratio"] >= 0.25
                    )
                    or (
                        m["bg_grad_med"] >= 40.0
                        and m["r_scene_med"] < 0.70 * m["bg_grad_med"]
                    )
                ):
                    decision = "likely_intact_scene_structure"
                    reason = "spatial_dust_matches_bg_structure"
                    vetoed_intact[sl][sub_m] = 255
                elif fb_warn:
                    decision = "uncertain_damage_candidate"
                    reason = "spatial_dust_fb_warning"
                    uncertain[sl][sub_m] = 255
                elif m["z_scene_med"] >= 2.2 and m["iqr_med"] <= max(10.0, 0.55 * m["r_scene_med"]):
                    decision = "high_confidence_damage"
                    reason = "spatial_dust_transient_blotch"
                    # Guard against oversized spatial dust halos bleeding onto intact pixels
                    px_ok = sub_m & (r_scene[sl] >= max(5.0, 0.75 * tau))
                    if np.any(px_ok):
                        high_conf[sl][px_ok] = 255
                    veto_rem = sub_m & ~px_ok
                    if np.any(veto_rem):
                        vetoed_intact[sl][veto_rem] = 255
                else:
                    decision = "uncertain_damage_candidate"
                    reason = "spatial_dust_moderate_confidence"
                    uncertain[sl][sub_m] = 255

                comp_records.append(ComponentRecord(
                    stream="spatial_dust",
                    component_id=i,
                    area_px=area,
                    orientation_deg=m["geom"]["orientation_deg"],
                    local_thickness_px=m["geom"]["local_thickness_px"],
                    arc_length_px=m["geom"]["arc_length_px"],
                    elongation=m["geom"]["elongation"],
                    mean_flow_mag_px=round(m["flow_mag_med"], 3),
                    surround_flow_mag_px=round(m["surround_flow_med"], 3),
                    median_fb_err_px=round(m["fb_err_med"], 3),
                    fb_warning=fb_warn,
                    scene_persistence_ratio=round(m["scene_persistence_ratio"], 3),
                    film_persistence_ratio=round(m["film_persistence_ratio"], 3),
                    flank_novelty_med=round(m["flank_novelty_med"], 2),
                    r_scene_med=round(m["r_scene_med"], 2),
                    z_scene_med=round(m["z_scene_med"], 2),
                    bg_grad_med=round(m["bg_grad_med"], 2),
                    follows_scene_motion=m["follows_scene_motion"],
                    temporal_class=m["temporal_class"],
                    decision=decision,
                    reason=reason,
                ))

        # Ensure strict mutual exclusion: high_confidence > uncertain > vetoed_intact
        if cfg.dilate_result and np.any(high_conf > 0):
            # Conservative 1 px groove recovery ONLY on true residual defect shoulders (never bleed onto clean/vetoed pixels)
            dilated_hc = cv2.dilate(high_conf, np.ones((3, 3), np.uint8))
            defect_shoulder = (
                (r_scene >= max(8.0, 0.90 * tau))
                & (c_target >= 0.60 * contrast_th)
                & (c_scene_eff <= np.maximum(5.0, 0.30 * c_target))
            )
            shoulder_expand = (dilated_hc > 0) & (high_conf == 0) & defect_shoulder & (vetoed_intact == 0)
            high_conf = np.where((high_conf > 0) | shoulder_expand, np.uint8(255), np.uint8(0))

        uncertain[high_conf > 0] = 0
        vetoed_intact[high_conf > 0] = 0
        vetoed_intact[uncertain > 0] = 0

        temporal_cand = geom_mask.copy()
        if cfg.dilate_result and np.any(temporal_cand > 0) and not cfg.use_motion_compensated_evidence:
            temporal_cand = cv2.dilate(temporal_cand, np.ones((1, 3), np.uint8))

        return ExperimentalThreeStateResult(
            high_confidence_damage=np.ascontiguousarray(high_conf),
            uncertain_damage_candidate=np.ascontiguousarray(uncertain),
            likely_intact_scene_structure=np.ascontiguousarray(vetoed_intact),
            temporal_candidate_mask=np.ascontiguousarray(temporal_cand),
            background=bg_float,
            flow_mag=mean_flow_mag,
            fb_err=median_fb_err,
            fb_warning_map=fb_warning_map,
            abstained=False,
            abstention_reason=None,
            valid_neighbors=nbrs,
            skipped_duplicates=skipped,
            component_records=comp_records,
            metadata={
                "tau": round(tau, 3),
                "sigma_mad": round(sigma_mad, 3),
                "contrast_th": round(contrast_th, 3),
                "cut_selector": cut_meta,
                "rejected_neighbors": rejected_neighbors,
                "global_fb_med": round(global_fb_med, 3),
                "global_fb_ok_frac": round(global_fb_ok_frac, 4),
                "high_confidence_pixels": int(np.count_nonzero(high_conf)),
                "uncertain_pixels": int(np.count_nonzero(uncertain)),
                "vetoed_intact_pixels": int(np.count_nonzero(vetoed_intact)),
                "persistent_film_locked_components": sum(
                    1 for c in comp_records if c.temporal_class == "persistent_film_locked" and c.decision == "high_confidence_damage"
                ),
            },
        )

    def analyze_sequence(
        self,
        frames: Sequence[np.ndarray],
        *,
        spatial_scratch_masks: Sequence[np.ndarray] | None = None,
        spatial_dust_masks: Sequence[np.ndarray] | None = None,
        target_indices: Sequence[int] | None = None,
    ) -> dict[int, ExperimentalThreeStateResult]:
        """Run analyze_frame over target_indices with shared bounded sequence flow/ridge cache."""
        flow_cache: dict[tuple[int, int], np.ndarray] = {}
        ridge_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        flow_gray_cache: dict[int, np.ndarray] = {}
        indices = list(range(len(frames))) if target_indices is None else list(target_indices)
        results: dict[int, ExperimentalThreeStateResult] = {}
        seen_dups = False
        for pos, idx in enumerate(indices):
            s_scr = spatial_scratch_masks[idx] if spatial_scratch_masks is not None else None
            s_dst = spatial_dust_masks[idx] if spatial_dust_masks is not None else None
            res = self.analyze_frame(
                frames,
                idx,
                spatial_scratch_mask=s_scr,
                spatial_dust_mask=s_dst,
                _flow_cache=flow_cache,
                _ridge_cache=ridge_cache,
                _flow_gray_cache=flow_gray_cache,
            )
            results[idx] = res
            if res.skipped_duplicates:
                seen_dups = True
            if pos + 1 < len(indices):
                keep_radius = self.config.window_radius + (3 if seen_dups else 0)
                min_future = min(indices[pos + 1 :]) - keep_radius
                for k in [k for k in ridge_cache if k < min_future]:
                    del ridge_cache[k]
                for k in [k for k in flow_gray_cache if k < min_future]:
                    del flow_gray_cache[k]
                for fk in [fk for fk in flow_cache if fk[0] < min_future or fk[1] < min_future]:
                    del flow_cache[fk]
        return results
