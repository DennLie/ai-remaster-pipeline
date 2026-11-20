#!/usr/bin/env python3
"""Unit and safety-regression tests for the isolated experimental scene-vs-film prototype."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import benchmark_defect_masks as bdm
from experimental_scene_film_scratch_prototype import (
    ExperimentalConfig,
    ExperimentalSceneFilmDetector,
    extract_orientation_aware_components,
    select_motion_robust_neighbors,
)
from temporal_scratch_analyzer import TemporalConfig


def _make_textured_canvas(h: int = 160, w: int = 240, seed: int = 101) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(125.0, 22.0, size=(h + 60, w + 180)).astype(np.float32)
    base = cv2.GaussianBlur(base, (7, 7), 1.6)
    # Add deterministic architectural squares/edges in background
    cv2.rectangle(base, (90, 30), (140, 140), 70.0, 3)
    cv2.rectangle(base, (180, 20), (210, 155), 185.0, -1)
    cv2.line(base, (150, 15), (150, 175), 55.0, 3)
    return np.clip(base, 10, 245).astype(np.uint8)


def _crop_pan_sequence(
    canvas: np.ndarray,
    *,
    num_frames: int = 9,
    h: int = 120,
    w: int = 160,
    dx_per_frame: float = 10.0,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for t in range(num_frames):
        x0 = int(round(10 + t * dx_per_frame))
        y0 = 20
        crop = canvas[y0 : y0 + h, x0 : x0 + w]
        frames.append(cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR))
    return frames


def test_m2_robust_scene_cut_distinguishes_fast_pan_from_hard_cut() -> None:
    canvas = _make_textured_canvas(h=180, w=320, seed=7)
    pan_frames = _crop_pan_sequence(canvas, num_frames=9, h=140, w=180, dx_per_frame=11.0)
    pan_grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in pan_frames]

    # Legacy cumulative MAD aborts on fast pan when step MAD > 25; M2 accepts pan neighbors
    legacy_cfg = ExperimentalConfig(use_robust_scene_cut_selector=False, legacy_scene_cut_threshold=18.0)
    p_leg, s_leg, _, _ = select_motion_robust_neighbors(pan_grays, 4, legacy_cfg)
    assert len(p_leg + s_leg) < 4

    robust_cfg = ExperimentalConfig(use_robust_scene_cut_selector=True)
    p_rob, s_rob, _, _ = select_motion_robust_neighbors(pan_grays, 4, robust_cfg)
    assert p_rob == [3, 2, 1]
    assert s_rob == [5, 6, 7]

    # Inject hard scene cut after frame 5 (different tonal distribution & layout)
    rng = np.random.default_rng(999)
    cut_frames = [f.copy() for f in pan_frames]
    for i in range(6, 9):
        dark_scene = rng.integers(15, 55, size=(140, 180), dtype=np.uint8)
        cv2.circle(dark_scene, (90 + i * 3, 70), 30, 210, 3)
        cut_frames[i] = cv2.cvtColor(dark_scene, cv2.COLOR_GRAY2BGR)
    cut_grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in cut_frames]

    p_cut, s_cut, _, meta = select_motion_robust_neighbors(cut_grays, 4, robust_cfg)
    assert p_cut == [3, 2, 1]
    assert all(idx < 6 for idx in s_cut)
    assert meta["cuts_detected"]

    # Telecine duplicate skipping
    tele_grays = [pan_grays[0], pan_grays[1], pan_grays[1].copy(), pan_grays[2], pan_grays[3], pan_grays[3].copy(), pan_grays[4]]
    p_tel, s_tel, skip_tel, _ = select_motion_robust_neighbors(tele_grays, 3, robust_cfg)
    assert 1 in skip_tel and 5 in skip_tel
    assert 2 in p_tel and 4 in s_tel


def test_m3_orientation_aware_centerline_geometry() -> None:
    canvas = np.zeros((200, 240), dtype=np.uint8)
    # 1. Slightly tilted thin line spanning wide bbox (cw > 12), 2 px thick
    cv2.line(canvas, (30, 10), (50, 180), 255, 2)
    # 2. Diagonal line (~40 deg), 2 px thick
    cv2.line(canvas, (70, 20), (130, 110), 255, 2)
    # 3. Horizontal line (~88 deg), 2 px thick
    cv2.line(canvas, (90, 175), (210, 180), 255, 2)
    # 4. Compact square blob (must be rejected!)
    cv2.rectangle(canvas, (170, 30), (195, 55), 255, -1)

    legacy_mask, _ = extract_orientation_aware_components(
        canvas, ExperimentalConfig(use_centerline_thickness_geometry=False)
    )
    assert np.count_nonzero(legacy_mask[170:190, 90:210]) == 0  # legacy misses horizontal
    assert np.count_nonzero(legacy_mask[20:110, 70:130]) == 0   # legacy misses diagonal

    m3_mask, recs = extract_orientation_aware_components(
        canvas, ExperimentalConfig(use_centerline_thickness_geometry=True)
    )
    assert np.count_nonzero(m3_mask[10:180, 25:55]) > 150       # tilted kept
    assert np.count_nonzero(m3_mask[20:110, 70:135]) > 100      # diagonal kept
    assert np.count_nonzero(m3_mask[170:186, 90:210]) > 120     # horizontal kept
    assert np.count_nonzero(m3_mask[28:58, 168:198]) == 0       # compact square rejected
    assert len(recs) == 3


def test_scene_locked_vs_film_locked_and_static_uncertainty() -> None:
    canvas = _make_textured_canvas(h=180, w=320, seed=42)
    pan_clean = _crop_pan_sequence(canvas, num_frames=11, h=140, w=180, dx_per_frame=8.0)

    # Inject a persistent film-locked bright scratch at x ~ 95 (duration=11, low drift)
    pers_frames, pers_gt, _ = bdm.inject_extended_scratches(
        pan_clean,
        bdm.ExtendedScratchConfig(
            orientation="vertical",
            angle_deg=0.0,
            width=2,
            opacity=0.90,
            bright=True,
            duration=11,
            drift_px=0.1,
            seed=555,
        ),
    )

    det = ExperimentalSceneFilmDetector(ExperimentalConfig())
    res_pers = det.analyze_frame(pers_frames, 5)
    gt_f5 = pers_gt[5] == 255
    tp_hc = int(np.count_nonzero((res_pers.high_confidence_damage > 0) & gt_f5))
    assert tp_hc > 0.35 * int(np.count_nonzero(gt_f5)), "Must recover persistent film-locked scratch on panning scene"
    assert any(r.temporal_class == "persistent_film_locked" and r.decision == "high_confidence_damage" for r in res_pers.component_records)

    # Clean panning clip containing ONLY authentic scene vertical bar -> zero conservative leak
    res_clean = det.analyze_frame(pan_clean, 5)
    assert int(np.count_nonzero(res_clean.high_confidence_damage)) < 25

    # Static camera clip with stationary persistent vertical line -> cannot resolve from motion alone,
    # MUST NOT force into high_confidence_damage!
    static_base = [pan_clean[5].copy() for _ in range(11)]
    static_frames, static_gt, _ = bdm.inject_extended_scratches(
        static_base,
        bdm.ExtendedScratchConfig(
            orientation="vertical",
            angle_deg=0.0,
            width=2,
            opacity=0.90,
            bright=True,
            duration=11,
            drift_px=0.0,
            seed=777,
        ),
    )
    res_static = det.analyze_frame(static_frames, 5)
    static_gt_f5 = static_gt[5] == 255
    assert int(np.count_nonzero((res_static.high_confidence_damage > 0) & static_gt_f5)) == 0

    # Grained static camera clip -> must NOT trigger telecine dedup abstention;
    # stationary persistent line must land in uncertain_damage_candidate, not high_confidence_damage
    rng = np.random.default_rng(202611194)
    static_grained = [
        np.clip(
            f.astype(np.float32) + rng.normal(0.0, 2.2, size=f.shape).astype(np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)
        for f in static_frames
    ]
    res_static_grain = det.analyze_frame(static_grained, 5)
    assert res_static_grain.abstained is False
    assert int(np.count_nonzero((res_static_grain.high_confidence_damage > 0) & static_gt_f5)) == 0
    assert int(np.count_nonzero((res_static_grain.uncertain_damage_candidate > 0) & static_gt_f5)) > 0


def test_fused_spatial_overlap_preserves_high_confidence_film_locked_scratch() -> None:
    canvas = _make_textured_canvas(h=160, w=280, seed=101)
    pan_clean = _crop_pan_sequence(canvas, num_frames=11, h=120, w=160, dx_per_frame=8.0)
    scratched, scratched_gt, _ = bdm.inject_extended_scratches(
        pan_clean,
        bdm.ExtendedScratchConfig(
            orientation="vertical",
            angle_deg=0.0,
            width=2,
            opacity=0.90,
            bright=True,
            duration=11,
            drift_px=0.0,
            seed=303,
        ),
    )
    gt_f5 = scratched_gt[5] == 255
    assert int(np.count_nonzero(gt_f5)) > 20

    det = ExperimentalSceneFilmDetector(ExperimentalConfig())
    res_e1 = det.analyze_frame(scratched, 5)
    s_scratch = (scratched_gt[5] == 255).astype(np.uint8) * 255
    res_e2 = det.analyze_frame(scratched, 5, spatial_scratch_mask=s_scratch)

    e1_hc = int(np.count_nonzero((res_e1.high_confidence_damage > 0) & gt_f5))
    e2_hc = int(np.count_nonzero((res_e2.high_confidence_damage > 0) & gt_f5))
    assert e1_hc > 20
    assert e2_hc >= int(0.9 * e1_hc)


def test_fb_inconsistency_withholds_moderate_artifact_to_uncertain() -> None:
    canvas = _make_textured_canvas(h=160, w=280, seed=211)
    pan_clean = _crop_pan_sequence(canvas, num_frames=11, h=120, w=160, dx_per_frame=6.0)
    # Create an oscillating foreground occluder around [20:100, 45:115] across neighbors while keeping
    # the local background near x=79 smooth (~108 +/- 5), causing DIS forward-backward inconsistency
    # (>2.5 px) while frame 5 exhibits a moderate thin bright ridge (+16, intensity 124).
    frames = [f.copy() for f in pan_clean]
    yy, _ = np.mgrid[20:100, 45:115]
    for i, f in enumerate(frames):
        if i != 5:
            pat = (108 + 5 * np.sin((yy + (i - 5) * 9) * 0.5)).astype(np.uint8)
            for c in range(3):
                f[20:100, 45:115, c] = pat
            x_edge = 62 if i % 2 == 0 else 96
            cv2.rectangle(f, (x_edge - 8, 25), (x_edge + 8, 95), (20, 20, 20), -1)
        else:
            f[20:100, 45:115] = 108
            cv2.line(f, (79, 30), (79, 90), (124, 124, 124), 2)

    roi = np.zeros((120, 160), dtype=bool)
    roi[30:90, 77:83] = True

    det = ExperimentalSceneFilmDetector(ExperimentalConfig())
    res = det.analyze_frame(frames, 5)
    assert res.abstained is False
    assert float(np.median(res.fb_err[roi])) > 2.5
    assert int(np.count_nonzero((res.high_confidence_damage > 0) & roi)) == 0
    assert int(np.count_nonzero((res.uncertain_damage_candidate > 0) & roi)) > 0
    assert int(np.count_nonzero((res.likely_intact_scene_structure > 0) & roi)) == 0
    assert any(
        c.fb_warning and c.reason == "fb_inconsistency_warning" and c.decision == "uncertain_damage_candidate"
        for c in res.component_records
    )


def test_separate_spatial_scratch_and_dust_policies_and_frozen_production_defaults() -> None:
    canvas = _make_textured_canvas(h=180, w=320, seed=19)
    pan_clean = _crop_pan_sequence(canvas, num_frames=11, h=140, w=180, dx_per_frame=6.0)
    h, w = pan_clean[5].shape[:2]

    # Create synthetic static edge false-alarm proposal in spatial_dust + spatial_scratch on an intact dark line
    # Wait: frame 5 has the scene line at x0 = 10 + 5*6 = 40; original canvas vertical line at x = 150 -> image x = 110
    s_scratch = np.zeros((h, w), dtype=np.uint8)
    s_dust = np.zeros((h, w), dtype=np.uint8)
    cv2.line(s_scratch, (110, 10), (110, 120), 255, 2)  # false positive on intact panning scene line
    cv2.circle(s_dust, (110, 50), 3, 255, -1)           # false positive dust on intact panning scene line

    # Also inject a genuine single-frame bright dust dot at (35, 35)
    dmg = [f.copy() for f in pan_clean]
    cv2.circle(dmg[5], (35, 35), 3, (240, 240, 240), -1)
    cv2.circle(s_dust, (35, 35), 3, 255, -1)

    det = ExperimentalSceneFilmDetector(ExperimentalConfig())
    res = det.analyze_frame(dmg, 5, spatial_scratch_mask=s_scratch, spatial_dust_mask=s_dust)

    # False spatial scratch & dust on intact line at x=110 must be vetoed / excluded from high_confidence_damage
    assert np.count_nonzero(res.high_confidence_damage[:, 106:115]) == 0
    # Genuine transient dust blotch at (35, 35) must remain authorized
    assert np.count_nonzero(res.high_confidence_damage[30:40, 30:40]) > 10

    # Verify frozen baseline TemporalConfig constants were NOT mutated
    base_cfg = TemporalConfig()
    assert base_cfg.scene_cut_threshold == 25.0
    assert base_cfg.motion_coherence_threshold == 1.2
    assert base_cfg.scratch_max_width == 12


def test_bridged_morphology_oriented_preserves_diagonal_and_blocks_cross_gap_bridging():
    from experimental_scene_film_scratch_prototype import _bridged_morphology_oriented

    # 1. Isolated 2x2 noise speck is rejected
    speck = np.zeros((64, 64), dtype=np.uint8)
    speck[20:22, 20:22] = 255
    assert int(np.count_nonzero(_bridged_morphology_oriented(speck))) == 0

    # 2. Diagonal 2-px line preserves recall without even-kernel +1 px dilation bloat
    diag = np.zeros((96, 96), dtype=np.uint8)
    cv2.line(diag, (15, 15), (75, 75), 255, thickness=2)
    diag_area = int(np.count_nonzero(diag))
    diag_out = _bridged_morphology_oriented(diag)
    diag_tp = int(np.count_nonzero((diag_out > 0) & (diag > 0)))
    assert diag_tp / float(diag_area) >= 0.90
    assert int(np.count_nonzero(diag_out)) <= int(1.15 * diag_area)

    # 3. Adjacent clean vertical lines 6 px apart must NOT bridge horizontally across the gap,
    # while a 1-px vertical break along a vertical stroke IS bridged.
    lines = np.zeros((96, 96), dtype=np.uint8)
    lines[15:85, 40] = 255
    lines[15:85, 46] = 255
    lines[20:40, 20] = 255
    lines[42:62, 20] = 255
    lines_out = _bridged_morphology_oriented(lines)
    assert int(np.count_nonzero(lines_out[20:80, 40])) >= 50
    assert int(np.count_nonzero(lines_out[20:80, 46])) >= 50
    assert int(np.count_nonzero(lines_out[:, 42:45])) == 0
    assert int(lines_out[41, 20]) == 255


def test_static_low_grain_transient_scratch_not_skipped_as_telecine():
    rng = np.random.default_rng(202611195)
    base = np.full((128, 128), 105.0, dtype=np.float32)
    # Subtle static background structure + low grain sigma=0.4
    base[:, ::8] += 6.0
    frames = []
    for _ in range(11):
        noisy = np.clip(base + rng.normal(0.0, 0.4, size=base.shape), 0.0, 255.0).astype(np.uint8)
        frames.append(noisy)

    gt_mask = np.zeros((128, 128), dtype=bool)
    f5 = frames[5].astype(np.float32)
    f5[18:110, 64:66] = np.clip(f5[18:110, 64:66] + 42.0, 0.0, 255.0)
    frames[5] = f5.astype(np.uint8)
    gt_mask[18:110, 64:66] = True

    det = ExperimentalSceneFilmDetector(
        ExperimentalConfig(
            use_motion_compensated_evidence=True,
            use_robust_scene_cut_selector=True,
            use_centerline_thickness_geometry=True,
        )
    )
    res = det.analyze_frame(frames, 5)
    assert res.abstained is False
    assert res.abstention_reason != "telecine_duplicate_frames"
    tp_hc = int(np.count_nonzero((res.high_confidence_damage > 0) & gt_mask))
    assert tp_hc / float(np.count_nonzero(gt_mask)) > 0.50
    assert int(np.count_nonzero(res.high_confidence_damage > 0)) < 3 * int(np.count_nonzero(gt_mask))


def test_border_out_of_bounds_entry_does_not_leak_to_high_confidence():
    rng = np.random.default_rng(202611196)
    world = cv2.GaussianBlur(
        rng.normal(115.0, 18.0, size=(220, 320)).astype(np.float32), (5, 5), 1.1
    )
    # Place a sharp bright pole inside the world near x=155 so it enters from the right frame border
    world[20:200, 198:203] = 215.0
    frames = []
    # Camera pans left-to-right slowly? Shift window right (+10 px/frame) so new right-edge pixels in f5
    # were out of bounds in prior frames.
    for i in range(11):
        x0 = 40 + i * 10
        crop = world[30:190, x0 : x0 + 160]
        frames.append(np.clip(crop, 0.0, 255.0).astype(np.uint8))

    det = ExperimentalSceneFilmDetector(
        ExperimentalConfig(
            use_motion_compensated_evidence=True,
            use_robust_scene_cut_selector=True,
            use_centerline_thickness_geometry=True,
        )
    )
    res = det.analyze_frame(frames, 5)
    assert res.abstained is False
    assert int(np.count_nonzero(res.high_confidence_damage[:, :12])) == 0
    assert int(np.count_nonzero(res.high_confidence_damage[:, -12:])) == 0


def test_clean_panning_textured_canvas_stays_free_of_false_high_conf_repairs():
    rng = np.random.default_rng(202611197)
    world = cv2.GaussianBlur(
        rng.normal(120.0, 20.0, size=(240, 360)).astype(np.float32), (5, 5), 1.2
    )
    # Realistic vertical & diagonal architectural edges in scene coordinates
    cv2.line(world, (120, 20), (120, 220), 195.0, thickness=3)
    cv2.line(world, (170, 30), (240, 200), 60.0, thickness=2)
    frames = []
    for i in range(11):
        x0 = 30 + i * 4
        y0 = 20 + i
        crop = world[y0 : y0 + 160, x0 : x0 + 180]
        noisy = np.clip(crop + rng.normal(0.0, 1.5, size=crop.shape), 0.0, 255.0).astype(np.uint8)
        frames.append(noisy)

    det = ExperimentalSceneFilmDetector(
        ExperimentalConfig(
            use_motion_compensated_evidence=True,
            use_robust_scene_cut_selector=True,
            use_centerline_thickness_geometry=True,
        )
    )
    res = det.analyze_frame(frames, 5)
    assert res.abstained is False
    assert int(np.count_nonzero(res.high_confidence_damage > 0)) <= 25


def test_m2_catches_unaligned_intra_scene_zoom_cut_and_skips_single_flicker() -> None:
    rng = np.random.default_rng(202611198)
    # Construct high-contrast wide-shot planks vs 2x close-up (same tonal histogram, high MAD & low phase corr)
    wide = np.full((140, 180), 170.0, dtype=np.float32)
    for x in range(8, 178, 16):
        cv2.line(wide, (x, 0), (x, 140), 35.0, 4)
    cv2.rectangle(wide, (110, 60), (165, 130), 40.0, -1)
    wide_u8 = np.clip(wide, 0, 255).astype(np.uint8)

    # Shifted close-up re-framed cut keeping similar dark/light proportions
    closeup = np.full((140, 180), 170.0, dtype=np.float32)
    for x in range(4, 178, 30):
        cv2.line(closeup, (x, 0), (x, 140), 35.0, 7)
    cv2.rectangle(closeup, (15, 35), (85, 135), 40.0, -1)
    closeup_u8 = np.clip(closeup, 0, 255).astype(np.uint8)

    # Insert one mild exposure-flicker same-shot frame at index 3: sharp bimodal peaks shift slightly in
    # histogram (`hist_dist > 0.38`) while keeping high phase correlation (`resp >= 0.75`) and low `aligned_mad <= 12`.
    flicker = np.clip(wide_u8.astype(np.float32) * 0.94, 0, 255).astype(np.uint8)
    seq = [wide_u8.copy(), wide_u8.copy(), wide_u8.copy(), flicker, wide_u8.copy(), closeup_u8.copy(), closeup_u8.copy(), closeup_u8.copy()]
    noisy_seq = [
        np.clip(f.astype(np.float32) + rng.normal(0.0, 2.0, size=f.shape), 0, 255).astype(np.uint8)
        for f in seq
    ]

    det = ExperimentalSceneFilmDetector(ExperimentalConfig())
    p4, s4, skip4, meta4 = select_motion_robust_neighbors(noisy_seq, 4, det.config)
    # Must stop before frame 5 (intra-scene zoom cut!) AND skip flicker 3 to recover [2, 1, 0]
    assert all(i < 5 for i in p4 + s4)
    assert 5 not in (p4 + s4)
    assert 3 in skip4
    assert 2 in p4
    assert meta4["cuts_detected"]


def test_catastrophic_registration_failure_fails_closed_to_uncertain() -> None:
    rng = np.random.default_rng(202611199)
    # Build two unrelated striped shots whose neighbors pass a loose M2 threshold yet fail optical-flow registration.
    scene_a = np.full((140, 180), 185.0, dtype=np.float32)
    for x in range(10, 175, 20):
        cv2.line(scene_a, (x, 5), (x, 135), 30.0, 5)
    scene_b = np.full((140, 180), 185.0, dtype=np.float32)
    for y in range(10, 135, 20):
        cv2.line(scene_b, (5, y), (175, y), 30.0, 5)

    frames = [
        np.clip(scene_b + rng.normal(0.0, 2.0, size=scene_b.shape), 0, 255).astype(np.uint8)
        for _ in range(7)
    ]
    frames[3] = np.clip(scene_a + rng.normal(0.0, 2.0, size=scene_a.shape), 0, 255).astype(np.uint8)

    # Disable M2 selector with loose cut threshold so cross-shot neighbors reach registration verification directly!
    det = ExperimentalSceneFilmDetector(
        ExperimentalConfig(use_robust_scene_cut_selector=False, legacy_scene_cut_threshold=120.0)
    )
    dummy_spatial = np.zeros((140, 180), dtype=np.uint8)
    cv2.line(dummy_spatial, (30, 15), (30, 125), 255, 2)
    res = det.analyze_frame(frames, 3, spatial_scratch_mask=dummy_spatial)

    assert res.abstained is True
    assert res.abstention_reason in ("unreliable_temporal_registration", "global_flow_registration_unreliable")
    assert int(np.count_nonzero(res.high_confidence_damage)) == 0
    assert int(np.count_nonzero(res.uncertain_damage_candidate)) > 0


