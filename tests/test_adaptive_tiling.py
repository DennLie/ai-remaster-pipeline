"""Unit tests for Adaptive Tiling Engine and GPU VRAM Protection.

Validates:
1. Tile grid generation across 1080p, 2K, and 4K resolutions (100% coverage, no gaps).
2. Gaussian, distance_linear, and Hanning pre-threshold continuous blending (zero seam artifacts, in [0.0, 1.0]).
3. Deterministic temporal grid locking across consecutive frame sequences.
4. Hairline 1-pixel scratch continuity across overlapping tile boundaries.
5. VRAM budget monitoring, proactive batch size auto-throttling, and CPU fallback warnings.
6. Seamless integration with BaseDefectDetector and film defect detector registry.
"""
from __future__ import annotations

import warnings
import numpy as np
import pytest

from scripts.adaptive_tiling import (
    AdaptiveTilingConfig,
    AdaptiveTilingEngine,
    TileCoordinate,
    VRAMMonitor,
    create_blend_weight,
)
from scripts.film_defect_detectors import (
    AdaptiveTilingConfig as FddConfig,
    AdaptiveTilingEngine as FddEngine,
    Consensus5FDefectDetector,
    DefectMaskResult,
    FarnebackDefectDetector,
    MultiChannelDefectMask,
)


class TestAdaptiveTilingConfig:
    """Validate configuration instantiation, type coercion, and boundary assertions."""

    def test_default_config_attributes(self) -> None:
        cfg = AdaptiveTilingConfig()
        assert cfg.tile_size == (512, 512)
        assert cfg.overlap == (64, 64)
        assert cfg.context_padding == 32
        assert cfg.blend_mode == "gaussian"
        assert cfg.vram_budget_mb == 4096
        assert cfg.batch_size == 4
        assert cfg.device == "auto"

    def test_integer_coercion_for_sizes(self) -> None:
        cfg = AdaptiveTilingConfig(tile_size=256, overlap=32)
        assert cfg.tile_size == (256, 256)
        assert cfg.overlap == (32, 32)

    def test_invalid_tile_size_raises(self) -> None:
        with pytest.raises(ValueError, match="tile_size"):
            AdaptiveTilingConfig(tile_size=(0, 512))
        with pytest.raises(ValueError, match="tile_size"):
            AdaptiveTilingConfig(tile_size=(-10, 512))

    def test_invalid_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            AdaptiveTilingConfig(tile_size=(256, 256), overlap=(256, 64))
        with pytest.raises(ValueError, match="overlap"):
            AdaptiveTilingConfig(tile_size=(256, 256), overlap=(-1, 64))

    def test_invalid_blend_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="blend_mode"):
            AdaptiveTilingConfig(blend_mode="unsupported_filter")

    def test_cosine_alias_normalizes_to_hanning(self) -> None:
        cfg = AdaptiveTilingConfig(blend_mode="cosine")
        assert cfg.blend_mode == "hanning"

    def test_invalid_vram_budget_raises(self) -> None:
        with pytest.raises(ValueError, match="vram_budget_mb"):
            AdaptiveTilingConfig(vram_budget_mb=0)

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            AdaptiveTilingConfig(batch_size=0)


class TestTileGridGeneration:
    """Validate full spatial coverage and boundary invariants across resolutions."""

    @pytest.mark.parametrize(
        "res_name,height,width",
        [
            ("1080p_FHD", 1080, 1920),
            ("2K_DCI", 1080, 2048),
            ("1440p_QHD", 1440, 2560),
            ("4K_UHD", 2160, 3840),
            ("4K_DCI", 2160, 4096),
        ],
    )
    def test_tile_grid_coverage_standard_resolutions(
        self, res_name: str, height: int, width: int
    ) -> None:
        cfg = AdaptiveTilingConfig(tile_size=(512, 512), overlap=(64, 64), context_padding=32)
        engine = AdaptiveTilingEngine(cfg)
        coords = engine.generate_tiles((height, width))

        assert len(coords) > 0, f"No tiles generated for {res_name}"

        coverage = np.zeros((height, width), dtype=np.int32)
        for tile in coords:
            assert 0 <= tile.y < height
            assert 0 <= tile.x < width
            assert tile.y + tile.height <= height
            assert tile.x + tile.width <= width
            assert tile.height > 0
            assert tile.width > 0
            coverage[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += 1

        # Every pixel must be covered at least once (no gaps)
        assert np.all(coverage >= 1), f"Coverage gap found in {res_name}!"
        # Maximum overlap should not exceed 4 at internal tile corners
        assert int(coverage.max()) <= 4, f"Excessive overlap ({coverage.max()}) in {res_name}"

    def test_tile_grid_smaller_than_tile_size(self) -> None:
        cfg = AdaptiveTilingConfig(tile_size=(512, 512), overlap=(64, 64))
        engine = AdaptiveTilingEngine(cfg)
        coords = engine.generate_tiles((200, 300))

        assert len(coords) == 1
        tile = coords[0]
        assert tile.x == 0 and tile.y == 0
        assert tile.width == 300 and tile.height == 200

    def test_tile_grid_exact_tile_size_multiple(self) -> None:
        cfg = AdaptiveTilingConfig(tile_size=(256, 256), overlap=(64, 64))
        engine = AdaptiveTilingEngine(cfg)
        coords = engine.generate_tiles((512, 512))

        coverage = np.zeros((512, 512), dtype=np.int32)
        for t in coords:
            coverage[t.y : t.y + t.height, t.x : t.x + t.width] += 1
        assert np.all(coverage >= 1)


class TestSeamBlending:
    """Validate continuous probability blending and zero seam artifacts."""

    @pytest.mark.parametrize("blend_mode", ["gaussian", "distance_linear", "hanning"])
    def test_constant_probability_preserves_exact_values(self, blend_mode: str) -> None:
        target_h, target_w = 1080, 1920
        cfg = AdaptiveTilingConfig(
            tile_size=(512, 512),
            overlap=(64, 64),
            blend_mode=blend_mode,
        )
        engine = AdaptiveTilingEngine(cfg)
        coords = engine.generate_tiles((target_h, target_w))

        target_value = 0.725
        tiles = [
            np.full((coord.height, coord.width), target_value, dtype=np.float32)
            for coord in coords
        ]

        blended = engine.blend_probability_maps(tiles, coords, (target_h, target_w), blend_mode=blend_mode)

        assert blended.shape == (target_h, target_w)
        assert blended.dtype == np.float32
        # Verify strict probability bounds
        assert float(blended.min()) >= 0.0
        assert float(blended.max()) <= 1.0
        # Zero border seam discontinuity: output must be constant everywhere within 1e-5
        assert np.allclose(blended, target_value, atol=1e-5), (
            f"Seam artifact detected in {blend_mode} blend mode! "
            f"Min: {blended.min()}, Max: {blended.max()}, Expected: {target_value}"
        )

    def test_linear_gradient_continuity_across_seam(self) -> None:
        target_h, target_w = 600, 800
        cfg = AdaptiveTilingConfig(
            tile_size=(256, 256),
            overlap=(64, 64),
            blend_mode="gaussian",
        )
        engine = AdaptiveTilingEngine(cfg)
        coords = engine.generate_tiles((target_h, target_w))

        # Synthetic continuous horizontal gradient
        x_coords = np.arange(target_w, dtype=np.float32) / float(target_w)
        ground_truth = np.tile(x_coords, (target_h, 1))

        tiles = [ground_truth[c.y : c.y + c.height, c.x : c.x + c.width].copy() for c in coords]
        blended = engine.blend_probability_maps(tiles, coords, (target_h, target_w))

        assert np.allclose(blended, ground_truth, atol=1e-4)

    def test_multi_channel_blending_channel_first(self) -> None:
        target_h, target_w = 400, 600
        cfg = AdaptiveTilingConfig(tile_size=(256, 256), overlap=(32, 32))
        engine = AdaptiveTilingEngine(cfg)
        coords = engine.generate_tiles((target_h, target_w))

        # 2 channels: channel 0 = 0.3, channel 1 = 0.8
        tiles = [
            np.stack(
                [
                    np.full((c.height, c.width), 0.3, dtype=np.float32),
                    np.full((c.height, c.width), 0.8, dtype=np.float32),
                ],
                axis=0,
            )
            for c in coords
        ]

        blended = engine.blend_probability_maps(tiles, coords, (target_h, target_w))
        assert blended.shape == (2, target_h, target_w)
        assert np.allclose(blended[0], 0.3, atol=1e-5)
        assert np.allclose(blended[1], 0.8, atol=1e-5)


class TestTemporalGridLocking:
    """Validate deterministic spatial coordinates across sequential frames."""

    def test_deterministic_tile_coordinates_across_sequence(self) -> None:
        cfg = AdaptiveTilingConfig(tile_size=(512, 512), overlap=(64, 64))
        engine = AdaptiveTilingEngine(cfg)

        res = (1080, 1920)
        reference_coords = engine.generate_tiles(res)

        # Generate coordinates for 20 simulated consecutive frames
        for frame_idx in range(20):
            frame_coords = engine.generate_tiles(res)
            assert len(frame_coords) == len(reference_coords)
            for c_ref, c_cur in zip(reference_coords, frame_coords):
                assert c_cur.x == c_ref.x
                assert c_cur.y == c_ref.y
                assert c_cur.width == c_ref.width
                assert c_cur.height == c_ref.height
                assert c_cur.pad_top == c_ref.pad_top

    def test_sequence_inference_coordinate_stability(self) -> None:
        cfg = AdaptiveTilingConfig(tile_size=(128, 128), overlap=(32, 32), context_padding=16)
        engine = AdaptiveTilingEngine(cfg)

        frames = [np.full((200, 200, 3), 100 + i * 10, dtype=np.uint8) for i in range(3)]

        def mock_infer_fn(tile_frames: list[np.ndarray]) -> list[np.ndarray]:
            # Returns mean intensity normalized to [0, 1] for each frame
            return [np.full(f.shape[:2], float(f[0, 0, 0]) / 255.0, dtype=np.float32) for f in tile_frames]

        results = engine.run_tiled_inference_sequence(frames, mock_infer_fn, cfg)
        assert len(results) == 3
        for i, res in enumerate(results):
            expected = (100.0 + i * 10.0) / 255.0
            assert np.allclose(res, expected, atol=1e-4)


class TestHairlineScratchContinuity:
    """Validate that fine hairline scratches remain continuous across tile seams."""

    def test_vertical_hairline_scratch_across_tile_boundary(self) -> None:
        target_h, target_w = 1080, 1920
        # Tile grid with 512 tile_size and 64 overlap places boundary row at y=448..512
        cfg = AdaptiveTilingConfig(
            tile_size=(512, 512),
            overlap=(64, 64),
            context_padding=32,
            blend_mode="gaussian",
        )
        engine = AdaptiveTilingEngine(cfg)

        # Create synthetic image with a 1-pixel wide vertical scratch passing through the boundary
        frame = np.full((target_h, target_w), 128, dtype=np.uint8)
        scratch_x = 480
        scratch_y_start = 350
        scratch_y_end = 650
        frame[scratch_y_start:scratch_y_end, scratch_x] = 255  # Bright vertical scratch

        def detect_scratch(patch: np.ndarray) -> np.ndarray:
            # Detect pixels with intensity 255
            prob = np.where(patch == 255, 1.0, 0.0).astype(np.float32)
            return prob

        blended_prob = engine.run_tiled_inference(frame, detect_scratch, cfg)
        binary_mask = np.where(blended_prob >= 0.5, np.uint8(255), np.uint8(0))

        # Check scratch continuity:
        # Every pixel along the scratch path must be detected (strictly 255)
        scratch_slice = binary_mask[scratch_y_start:scratch_y_end, scratch_x]
        assert np.all(scratch_slice == 255), (
            f"Scratch interrupted across seam! Missing count: {np.count_nonzero(scratch_slice == 0)}"
        )

        # Check that no spurious blooming occurred outside the scratch path
        total_defect_pixels = np.count_nonzero(binary_mask)
        expected_pixels = scratch_y_end - scratch_y_start
        assert total_defect_pixels == expected_pixels, (
            f"Defect pixel count mismatch: expected {expected_pixels}, got {total_defect_pixels}"
        )

    def test_diagonal_scratch_crossing_tile_junction(self) -> None:
        target_h, target_w = 600, 600
        cfg = AdaptiveTilingConfig(tile_size=(256, 256), overlap=(64, 64), context_padding=16)
        engine = AdaptiveTilingEngine(cfg)

        frame = np.zeros((target_h, target_w), dtype=np.uint8)
        # Diagonal scratch passing through tile intersection around (200, 200)
        for i in range(150, 250):
            frame[i, i] = 255

        def detect_scratch(patch: np.ndarray) -> np.ndarray:
            return np.where(patch == 255, 1.0, 0.0).astype(np.float32)

        blended = engine.run_tiled_inference(frame, detect_scratch, cfg)
        binary = np.where(blended >= 0.5, np.uint8(255), np.uint8(0))

        for i in range(150, 250):
            assert binary[i, i] == 255, f"Diagonal scratch broken at ({i}, {i})"


class TestVRAMProtectionAndThrottling:
    """Validate GPU memory monitoring, batch throttling, and fallback warnings."""

    def test_available_vram_query(self) -> None:
        avail = VRAMMonitor.get_available_vram_mb()
        assert isinstance(avail, float)
        assert avail >= 0.0

    def test_estimate_tile_vram(self) -> None:
        # 512x512x3 float32
        est = VRAMMonitor.estimate_tile_vram_mb((512, 512))
        assert 10.0 <= est <= 100.0

    def test_batch_throttling_with_explicit_warning(self) -> None:
        monitor = VRAMMonitor(default_budget_mb=4096)
        # Simulate constrained GPU environment: 400 MB available, 300 MB overhead -> 100 MB usable
        # Each 512x512 tile ~50 MB, so batch of 8 cannot fit
        with pytest.warns(UserWarning, match="auto-throttled batch_size"):
            eff_bs, fell_back = monitor.determine_batch_size(
                requested_batch_size=8,
                tile_shape=(512, 512),
                vram_budget_mb=400,
                device="cuda",
                model_overhead_mb=300.0,
                available_vram_mb=400.0,
            )
        assert eff_bs < 8
        assert eff_bs >= 1
        assert not fell_back

    def test_cpu_fallback_warning_when_vram_zero(self) -> None:
        monitor = VRAMMonitor()
        with pytest.warns(UserWarning, match="falling back to CPU"):
            eff_bs, fell_back = monitor.determine_batch_size(
                requested_batch_size=4,
                tile_shape=(512, 512),
                device="auto",
                available_vram_mb=0.0,  # Simulate no CUDA
            )
        assert fell_back is True
        assert eff_bs in (1, 2)

    def test_run_tiled_inference_with_throttled_budget(self) -> None:
        # Small budget that forces throttling to 1
        cfg = AdaptiveTilingConfig(
            tile_size=(128, 128),
            overlap=(32, 32),
            context_padding=8,
            vram_budget_mb=350,
            batch_size=8,
        )
        monitor = VRAMMonitor()
        engine = AdaptiveTilingEngine(cfg, vram_monitor=monitor)

        frame = np.full((200, 200), 50, dtype=np.uint8)

        def mock_infer(patch: np.ndarray) -> np.ndarray:
            if patch.ndim == 3 and patch.shape[0] > 1:
                return np.full(patch.shape, 0.5, dtype=np.float32)
            return np.full(patch.shape[:2], 0.5, dtype=np.float32)

        out = engine.run_tiled_inference(frame, mock_infer, cfg)
        assert out.shape == (200, 200)
        assert np.allclose(out, 0.5, atol=1e-4)


class TestDetectorIntegration:
    """Validate AdaptiveTilingEngine integration with BaseDefectDetector hierarchy."""

    def test_engine_export_matches(self) -> None:
        assert FddConfig is AdaptiveTilingConfig
        assert FddEngine is AdaptiveTilingEngine

    def test_detector_detect_tiled_execution(self) -> None:
        detector = FarnebackDefectDetector()
        frames = [np.full((1080, 1920, 3), 128, dtype=np.uint8) for _ in range(3)]

        result = detector.detect_tiled(
            frames,
            config={"sensitivity": 0.65},
            tiling_config=AdaptiveTilingConfig(tile_size=(512, 512), overlap=(64, 64)),
        )

        assert isinstance(result, DefectMaskResult)
        assert len(result.mask) == 3
        assert result.mask[0].shape == (1080, 1920)
        assert result.multi_channel_masks is not None
        assert len(result.multi_channel_masks) == 3
        assert result.metadata.get("tiled_inference") is True

    def test_auto_tiling_flag_activation(self) -> None:
        detector = FarnebackDefectDetector()
        assert detector.should_use_tiling(1080, 1920, {"auto_tiling": True}) is True
        assert detector.should_use_tiling(2160, 3840, {"auto_tiling": True}) is True
        assert detector.should_use_tiling(480, 640, {"auto_tiling": True}) is False
        assert detector.should_use_tiling(480, 640, {"enable_tiling": True}) is True
        assert detector.should_use_tiling(1080, 1920, {"enable_tiling": False}) is False
