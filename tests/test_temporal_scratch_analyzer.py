"""Unit tests for TemporalScratchAnalyzer (Phase 2 PoC).

Validates:
1. Configuration validation and parameter constraints.
2. Invariant enforcement on TemporalDefectResult (shapes, dtypes, value ranges).
3. Telecine duplicate skipping logic.
4. Edge-case handling (insufficient temporal neighbors, boundary frames).
5. Both optical flow backends ("dis" and "farneback").
6. Empirical detection on evaluation/Neighbors_72.mkv on frame 25.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
import cv2

from scripts.temporal_scratch_analyzer import (
    TemporalConfig,
    TemporalDefectResult,
    TemporalScratchAnalyzer,
    SCRATCH_ROI,
    CLEAN_ROI,
    load_clip,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "evaluation" / "Neighbors_72.mkv"


class TestTemporalConfig:
    """Test configuration validation."""

    def test_default_config(self) -> None:
        cfg = TemporalConfig()
        assert cfg.window_radius == 3
        assert cfg.min_temporal_neighbors == 2
        assert cfg.flow_backend == "dis"
        assert cfg.sensitivity == 0.65
        assert cfg.scratch_min_height == 15
        assert cfg.scratch_max_width == 12
        assert cfg.scene_cut_threshold == 25.0
        assert cfg.scratch_tall_override_height == 35

    def test_invalid_sensitivity(self) -> None:
        with pytest.raises(ValueError, match="sensitivity must be in"):
            TemporalConfig(sensitivity=1.5)
        with pytest.raises(ValueError, match="sensitivity must be in"):
            TemporalConfig(sensitivity=-0.1)

    def test_invalid_flow_backend(self) -> None:
        with pytest.raises(ValueError, match="Unsupported flow_backend"):
            TemporalConfig(flow_backend="invalid_flow")

    def test_invalid_window_radius(self) -> None:
        with pytest.raises(ValueError, match="window_radius must be >= 1"):
            TemporalConfig(window_radius=0)

    def test_invalid_scene_cut_threshold(self) -> None:
        with pytest.raises(ValueError, match="scene_cut_threshold must be > 0.0"):
            TemporalConfig(scene_cut_threshold=0.0)

    def test_invalid_scratch_tall_override_height(self) -> None:
        with pytest.raises(ValueError, match="scratch_tall_override_height must be >= 1"):
            TemporalConfig(scratch_tall_override_height=0)

    def test_dict_initialization(self) -> None:
        analyzer = TemporalScratchAnalyzer({"window_radius": 2, "sensitivity": 0.70})
        assert analyzer.config.window_radius == 2
        assert analyzer.config.sensitivity == 0.70


class TestTemporalDefectResultInvariants:
    """Test data type and invariant contracts of TemporalDefectResult."""

    def test_valid_result(self) -> None:
        h, w = 100, 120
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:30, 20:25] = 255
        prob = np.zeros((h, w), dtype=np.float32)
        prob[10:30, 20:25] = 0.85
        bg = np.zeros((h, w), dtype=np.uint8)

        res = TemporalDefectResult(
            scratch_mask=mask,
            probability_map=prob,
            background=bg,
            metadata={"status": "test"},
        )
        assert res.scratch_mask.dtype == np.uint8
        assert res.probability_map.dtype == np.float32
        assert res.background.dtype == np.uint8

    def test_invalid_mask_values(self) -> None:
        h, w = 50, 50
        mask = np.ones((h, w), dtype=np.uint8) * 128  # Invalid: not {0, 255}
        prob = np.zeros((h, w), dtype=np.float32)
        bg = np.zeros((h, w), dtype=np.uint8)

        with pytest.raises(ValueError, match="strictly binary values"):
            TemporalDefectResult(scratch_mask=mask, probability_map=prob, background=bg)

    def test_invalid_probability_range(self) -> None:
        h, w = 50, 50
        mask = np.zeros((h, w), dtype=np.uint8)
        prob = np.ones((h, w), dtype=np.float32) * 1.5  # Invalid: > 1.0
        bg = np.zeros((h, w), dtype=np.uint8)

        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            TemporalDefectResult(scratch_mask=mask, probability_map=prob, background=bg)

    def test_dimension_mismatch(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        prob = np.zeros((60, 50), dtype=np.float32)
        bg = np.zeros((50, 50), dtype=np.uint8)

        with pytest.raises(ValueError, match="Shape mismatch"):
            TemporalDefectResult(scratch_mask=mask, probability_map=prob, background=bg)


class TestTelecineNeighborSelection:
    """Test telecine duplicate skipping behavior."""

    def test_duplicate_skipping(self) -> None:
        # Create a sequence with intentional duplicates
        h, w = 64, 64
        base_frame = np.ones((h, w), dtype=np.uint8) * 50
        diff_frame = np.ones((h, w), dtype=np.uint8) * 65

        # frames: 0:base, 1:base(dup), 2:diff, 3:base, 4:base(dup)
        frames = [
            base_frame.copy(),          # 0
            base_frame.copy(),          # 1: dup of 0
            diff_frame.copy(),          # 2
            base_frame.copy(),          # 3
            base_frame.copy(),          # 4: dup of 3
        ]

        analyzer = TemporalScratchAnalyzer(TemporalConfig(window_radius=2, telecine_dedup_threshold=0.8))
        preceding, succeeding, skipped = analyzer._select_nonduplicate_neighbors(frames, target_idx=4)

        # Frame 3 is dup of target 4, so should be skipped
        assert 3 in skipped or 3 not in preceding
        # Frame 2 is different, so should be accepted
        assert 2 in preceding


class TestWindowBuffering:
    """Test window buffering radius limits and boundary frame handling."""

    def test_window_buffering_boundaries(self) -> None:
        h, w = 64, 64
        # 10 frames with distinct gray values (step 5 > telecine thresh, < scene cut thresh)
        frames = [np.ones((h, w), dtype=np.uint8) * (20 + i * 5) for i in range(10)]
        analyzer = TemporalScratchAnalyzer(TemporalConfig(window_radius=3, telecine_dedup_threshold=0.8))

        # Boundary: first frame (index 0)
        p0, s0, sk0 = analyzer._select_nonduplicate_neighbors(frames, target_idx=0)
        assert len(p0) == 0
        assert len(s0) == 3
        assert s0 == [1, 2, 3]

        # Boundary: last frame (index 9)
        p9, s9, sk9 = analyzer._select_nonduplicate_neighbors(frames, target_idx=9)
        assert len(s9) == 0
        assert len(p9) == 3
        assert p9 == [8, 7, 6]

        # Center frame (index 5)
        p5, s5, sk5 = analyzer._select_nonduplicate_neighbors(frames, target_idx=5)
        assert len(p5) == 3
        assert len(s5) == 3
        assert p5 == [4, 3, 2]
        assert s5 == [6, 7, 8]


class TestTargetExclusion:
    """Test strict target exclusion from synthesized background reference."""

    def test_target_frame_strictly_excluded_from_background(self) -> None:
        h, w = 80, 80
        # Background is solid 100 for all frames
        frames = [np.full((h, w), 100, dtype=np.uint8) for _ in range(7)]

        # Target frame (index 3) has an intense bright block defect at [20:50, 30:50]
        frames[3][20:50, 30:50] = 255

        analyzer = TemporalScratchAnalyzer(TemporalConfig(window_radius=2, min_temporal_neighbors=2))
        res = analyzer.analyze_frame(frames, target_idx=3)

        assert res.metadata["status"] == "success"
        # Background at defect location MUST NOT contain target defect (must be close to 100)
        bg_roi = res.background[20:50, 30:50]
        assert np.allclose(bg_roi, 100, atol=2), (
            f"Target defect leaked into synthesized background! Mean: {np.mean(bg_roi)}"
        )


class TestMotionCoherenceFiltering:
    """Test rejection of high-motion scene textures and retention of defects in stable regions."""

    def test_coherent_motion_rejection_vs_scratch_retention(self) -> None:
        h, w = 120, 160

        # 1. Subtle motion scene (smooth gradient, flow mag <= 1.2 px): static scratch at x=80 on frame 3
        frames_scratch = [np.tile(np.linspace(40, 100, w, dtype=np.uint8), (h, 1)) for _ in range(7)]
        frames_scratch[3][20:100, 80:82] = 220

        # 2. High-motion scene (horizontal roll at 4 px/frame, flow mag > 1.5 px): vertical texture lines
        frames_high_motion = [
            np.roll(np.tile(np.linspace(40, 100, w, dtype=np.uint8), (h, 1)), t * 4, axis=1)
            for t in range(7)
        ]

        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(
                window_radius=2,
                min_temporal_neighbors=2,
                motion_coherence_threshold=1.5,
                scratch_min_height=15,
                scratch_max_width=8,
            )
        )

        # In high-motion scene, vertical gradient edges moving with scene motion (>1.5 px) must be rejected
        res_mot = analyzer.analyze_frame(frames_high_motion, target_idx=3)
        assert np.count_nonzero(res_mot.scratch_mask) == 0

        # In stable scene, vertical defect must be detected
        res_sc = analyzer.analyze_frame(frames_scratch, target_idx=3)
        assert np.count_nonzero(res_sc.scratch_mask[20:100, 78:84]) > 30


class TestShortLivedDefectPreservation:
    """Test that defects persisting across 2-3 frames are preserved and not suppressed."""

    def test_multi_frame_defect_preservation(self) -> None:
        h, w = 100, 140
        frames = [np.full((h, w), 80, dtype=np.uint8) for _ in range(7)]

        # Defect persists on frames 2, 3, 4 (3-frame scratch)
        for t in (2, 3, 4):
            frames[t][15:85, 70:72] = 220

        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(
                window_radius=3,
                min_temporal_neighbors=2,
                scratch_min_height=15,
                scratch_max_width=8,
            )
        )

        # Target frame 3 should be detected because trimmed median background over
        # frames [0, 1, 5, 6] provides clean baseline
        res3 = analyzer.analyze_frame(frames, target_idx=3)
        assert res3.metadata["status"] == "success"
        detected_3 = np.count_nonzero(res3.scratch_mask[15:85, 68:74])
        assert detected_3 > 40, f"Expected multi-frame scratch to be detected, got {detected_3} px"


class TestInsufficientNeighborsAbstention:
    """Test boundary handling when not enough neighbors exist."""

    def test_single_frame_abstention(self) -> None:
        frame = np.ones((64, 64), dtype=np.uint8) * 100
        analyzer = TemporalScratchAnalyzer(TemporalConfig(min_temporal_neighbors=2))
        res = analyzer.analyze_frame([frame], target_idx=0)

        assert res.metadata["status"] == "abstain"
        assert res.metadata["reason"] == "insufficient_temporal_neighbors"
        assert np.all(res.scratch_mask == 0)
        assert np.all(res.probability_map == 0.0)


class TestSyntheticScratchDetection:
    """Test detection of an artificial vertical scratch on synthetic moving background."""

    def test_synthetic_vertical_scratch(self) -> None:
        h, w = 120, 160
        frames = []

        # Generate 7 frames with slight background gradient and horizontal shift
        for t in range(7):
            f = np.tile(np.linspace(40, 100, w, dtype=np.uint8), (h, 1))
            # Frame 3 gets an artificial vertical bright scratch at x=80
            if t == 3:
                f[20:100, 80:82] = 220
            frames.append(f)

        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(
                window_radius=2,
                min_temporal_neighbors=2,
                scratch_min_height=15,
                scratch_max_width=8,
                sensitivity=0.65,
            )
        )
        res = analyzer.analyze_frame(frames, target_idx=3)

        assert res.metadata["status"] == "success"
        # Scratch at x=80, y=20:100 should be detected
        detected_in_scratch = np.count_nonzero(res.scratch_mask[20:100, 78:84])
        assert detected_in_scratch > 30, f"Expected >30 scratch pixels, found {detected_in_scratch}"


@pytest.mark.skipif(not VIDEO_PATH.is_file(), reason=f"Video not found at {VIDEO_PATH}")
class TestNeighbors72EmpiricalBenchmark:
    """Run empirical ground-truth benchmark on Neighbors_72.mkv frame 25."""

    def test_frame_25_scratch_detection(self) -> None:
        frames = load_clip(VIDEO_PATH)
        assert len(frames) == 72

        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(
                sensitivity=0.65,
                flow_backend="dis",
                window_radius=3,
                min_temporal_neighbors=2,
            )
        )
        result = analyzer.analyze_frame(frames, target_idx=25)

        s_ymin, s_xmin, s_ymax, s_xmax = (
            SCRATCH_ROI["ymin"],
            SCRATCH_ROI["xmin"],
            SCRATCH_ROI["ymax"],
            SCRATCH_ROI["xmax"],
        )
        c_ymin, c_xmin, c_ymax, c_xmax = (
            CLEAN_ROI["ymin"],
            CLEAN_ROI["xmin"],
            CLEAN_ROI["ymax"],
            CLEAN_ROI["xmax"],
        )

        s_pixels = int(np.count_nonzero(result.scratch_mask[s_ymin:s_ymax, s_xmin:s_xmax]))
        c_pixels = int(np.count_nonzero(result.scratch_mask[c_ymin:c_ymax, c_xmin:c_xmax]))

        # Validate Phase 2 requirements:
        # At least 50+ scratch pixels inside ROI (compared to 0 baseline)
        assert s_pixels >= 50, f"Expected >= 50 scratch pixels in ROI, got {s_pixels}"
        # Low false alarms in clean ROI (<5 pixels)
        assert c_pixels < 5, f"Expected < 5 clean ROI false alarms, got {c_pixels}"

        # Contract assertions
        assert result.scratch_mask.dtype == np.uint8
        assert result.scratch_mask.shape == (720, 960)
        assert np.all((result.scratch_mask == 0) | (result.scratch_mask == 255))
        assert result.probability_map.dtype == np.float32
        assert 0.0 <= result.probability_map.min() and result.probability_map.max() <= 1.0

    def test_frame_25_farneback_backend(self) -> None:
        frames = load_clip(VIDEO_PATH)
        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(
                sensitivity=0.65,
                flow_backend="farneback",
                window_radius=3,
                min_temporal_neighbors=2,
            )
        )
        result = analyzer.analyze_frame(frames, target_idx=25)

        s_ymin, s_xmin, s_ymax, s_xmax = (
            SCRATCH_ROI["ymin"],
            SCRATCH_ROI["xmin"],
            SCRATCH_ROI["ymax"],
            SCRATCH_ROI["xmax"],
        )
        c_ymin, c_xmin, c_ymax, c_xmax = (
            CLEAN_ROI["ymin"],
            CLEAN_ROI["xmin"],
            CLEAN_ROI["ymax"],
            CLEAN_ROI["xmax"],
        )

        s_pixels = int(np.count_nonzero(result.scratch_mask[s_ymin:s_ymax, s_xmin:s_xmax]))
        c_pixels = int(np.count_nonzero(result.scratch_mask[c_ymin:c_ymax, c_xmin:c_xmax]))

        assert s_pixels >= 50, f"Expected >= 50 scratch pixels with Farneback, got {s_pixels}"
        assert c_pixels < 5, f"Expected < 5 clean ROI false alarms with Farneback, got {c_pixels}"


class TestSceneCutDetection:
    """Test scene cut boundary detection stopping neighbor selection."""

    def test_scene_cut_stops_preceding_neighbors(self) -> None:
        h, w = 64, 64
        # Scene 1: frames 0..2 (mean intensity 40)
        # Scene 2: frames 3..6 (mean intensity 180)
        frames = [
            np.full((h, w), 40, dtype=np.uint8),   # 0
            np.full((h, w), 42, dtype=np.uint8),   # 1
            np.full((h, w), 44, dtype=np.uint8),   # 2
            np.full((h, w), 180, dtype=np.uint8),  # 3 (target)
            np.full((h, w), 182, dtype=np.uint8),  # 4
            np.full((h, w), 184, dtype=np.uint8),  # 5
            np.full((h, w), 186, dtype=np.uint8),  # 6
        ]
        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(window_radius=3, scene_cut_threshold=25.0, telecine_dedup_threshold=0.8)
        )
        preceding, succeeding, skipped = analyzer._select_nonduplicate_neighbors(frames, target_idx=3)

        # Preceding search stepped from 2 back to 0: MAD(2, 3) is ~136 > 25.0 -> stops immediately!
        assert len(preceding) == 0
        # Succeeding search stays within Scene 2: MAD is ~2-6 < 25.0 -> collects 4, 5, 6
        assert succeeding == [4, 5, 6]

    def test_scene_cut_stops_succeeding_neighbors(self) -> None:
        h, w = 64, 64
        # Target is frame 2 (Scene 1)
        frames = [
            np.full((h, w), 40, dtype=np.uint8),   # 0
            np.full((h, w), 42, dtype=np.uint8),   # 1
            np.full((h, w), 44, dtype=np.uint8),   # 2 (target)
            np.full((h, w), 180, dtype=np.uint8),  # 3 (cut)
            np.full((h, w), 182, dtype=np.uint8),  # 4
        ]
        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(window_radius=2, scene_cut_threshold=25.0, telecine_dedup_threshold=0.8)
        )
        preceding, succeeding, skipped = analyzer._select_nonduplicate_neighbors(frames, target_idx=2)

        assert preceding == [1, 0]
        # Succeeding halts at frame 3
        assert len(succeeding) == 0

    def test_scene_cut_abstains_when_insufficient_neighbors(self) -> None:
        h, w = 64, 64
        # Single frame shot: frame 1 is completely isolated by scene cuts on both sides
        frames = [
            np.full((h, w), 30, dtype=np.uint8),   # 0 (Scene 1)
            np.full((h, w), 180, dtype=np.uint8),  # 1 (Target, Scene 2)
            np.full((h, w), 30, dtype=np.uint8),   # 2 (Scene 3)
        ]
        analyzer = TemporalScratchAnalyzer(
            TemporalConfig(window_radius=2, min_temporal_neighbors=2, scene_cut_threshold=25.0)
        )
        res = analyzer.analyze_frame(frames, target_idx=1)
        assert res.metadata["status"] == "abstain"
        assert res.metadata["reason"] == "insufficient_temporal_neighbors"
        assert np.all(res.scratch_mask == 0)


class TestMicroDimensionHandling:
    """Test dimension guard for small images (< 16x16) preventing optical flow crash."""

    @pytest.mark.parametrize("shape", [(10, 10), (12, 64), (64, 8), (4, 4)])
    def test_small_dimension_abstention(self, shape: tuple[int, int]) -> None:
        h, w = shape
        frames = [np.full((h, w), 100, dtype=np.uint8) for _ in range(5)]
        analyzer = TemporalScratchAnalyzer()
        res = analyzer.analyze_frame(frames, target_idx=2)

        assert res.metadata["status"] == "abstain"
        assert res.metadata["reason"] == "small_spatial_dimension"
        assert res.scratch_mask.shape == (h, w)
        assert res.scratch_mask.dtype == np.uint8
        assert np.all(res.scratch_mask == 0)
        assert res.probability_map.shape == (h, w)
        assert res.probability_map.dtype == np.float32
        assert np.all(res.probability_map == 0.0)
        assert res.background.shape == (h, w)

    def test_load_clip_failures(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_clip(tmp_path / "non_existent.mkv")

        empty_file = tmp_path / "empty.mkv"
        empty_file.write_bytes(b"")
        with pytest.raises(ValueError):
            load_clip(empty_file)
