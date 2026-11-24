from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from ai_remaster_gui.rle_masks import MaskObjectMetadata, ShotMaskArchive
from scripts.osi_masked_lab import (
    apply_strict_l_lock,
    calculate_step_flow,
    colorize_shot_osi_masked_lab,
    fit_chroma_to_gamut,
    guided_filter,
    main,
    propagate_shot_chroma,
    warp_chroma_backward,
)


class TestGuidedFilter(unittest.TestCase):
    """Unit tests for the native OpenCV / NumPy Guided Filter implementation."""

    def test_guided_filter_edge_preservation_and_smoothing(self) -> None:
        """Verify filter smooths flat noise while preserving sharp high-contrast luminance step edges."""
        H, W = 64, 64
        # Create guidance image with a sharp vertical luminance step edge
        guide = np.zeros((H, W), dtype=np.float32)
        guide[:, W // 2 :] = 1.0  # Step from 0.0 to 1.0

        # Create source channel aligned with step edge plus Gaussian flat noise
        np.random.seed(42)
        noise = np.random.normal(0.0, 0.15, (H, W)).astype(np.float32)
        src = guide.copy() + noise

        # Flat region standard deviation before filtering
        raw_flat_std = float(np.std(src[:, : W // 2 - 5]))
        self.assertGreater(raw_flat_std, 0.10)

        # Run guided filter
        filtered = guided_filter(guide, src, radius=4, eps=1e-3)
        self.assertEqual(filtered.shape, (H, W))
        self.assertEqual(filtered.dtype, np.float32)

        # 1. Verify noise smoothing in flat regions: at least 4x reduction in std
        filtered_flat_std = float(np.std(filtered[:, : W // 2 - 5]))
        self.assertLess(filtered_flat_std, raw_flat_std * 0.25)

        # 2. Verify sharp step edge preservation across boundary
        # Jump from 2 pixels before boundary to 2 pixels after
        edge_jump = float(filtered[H // 2, W // 2 + 2] - filtered[H // 2, W // 2 - 3])
        self.assertGreaterEqual(edge_jump, 0.90)  # Preserved > 90% of full 1.0 step

    def test_guided_filter_numerical_stability(self) -> None:
        """Test on pure black (all 0) and pure white (all 255) frames, verifying zero NaNs or infinities."""
        shapes = [(16, 16), (32, 64)]
        for H, W in shapes:
            # Pure black guide (all 0.0)
            black_guide = np.zeros((H, W), dtype=np.float32)
            # Pure white guide (all 255.0 uint8 or 1.0 float32)
            white_guide = np.full((H, W), 255.0, dtype=np.float32)

            # Random and constant source channels
            src_random = np.random.uniform(-50.0, 50.0, (H, W)).astype(np.float32)
            src_constant = np.full((H, W), 30.0, dtype=np.float32)
            src_3d = np.random.uniform(-40.0, 40.0, (H, W, 2)).astype(np.float32)

            for g_name, g_img in [("black", black_guide), ("white", white_guide)]:
                for s_name, s_img in [
                    ("random", src_random),
                    ("constant", src_constant),
                    ("3d", src_3d),
                ]:
                    with self.subTest(guide=g_name, source=s_name, shape=(H, W)):
                        res = guided_filter(g_img, s_img, radius=5, eps=1e-3)
                        self.assertTrue(
                            np.all(np.isfinite(res)),
                            f"NaN or Inf encountered in guided_filter with {g_name} guide and {s_name} source",
                        )
                        self.assertEqual(res.shape, s_img.shape)


class TestDISOpticalFlowAndWarping(unittest.TestCase):
    """Unit tests for DIS optical flow calculation and backward chroma warping."""

    def test_dis_flow_backward_warp_translation(self) -> None:
        """Verify moving square is properly warped backward from curr to prev."""
        H, W = 80, 80
        # Prev frame: square at (20:40, 20:40)
        prev_gray = np.zeros((H, W), dtype=np.uint8)
        prev_gray[20:40, 20:40] = 200

        # Curr frame: square shifted by (+5, +5) to (25:45, 25:45)
        curr_gray = np.zeros((H, W), dtype=np.uint8)
        curr_gray[25:45, 25:45] = 200

        # Calculate step flow from curr to prev
        flow = calculate_step_flow(curr_gray, prev_gray)
        self.assertEqual(flow.shape, (H, W, 2))
        self.assertEqual(flow.dtype, np.float32)

        # Flow inside square should point back towards prev (-5, -5)
        inner_slice = (slice(28, 42), slice(28, 42))
        dx_mean = float(flow[inner_slice[0], inner_slice[1], 0].mean())
        dy_mean = float(flow[inner_slice[0], inner_slice[1], 1].mean())
        self.assertAlmostEqual(dx_mean, -5.0, delta=1.5)
        self.assertAlmostEqual(dy_mean, -5.0, delta=1.5)

        # Distinctive chroma in prev square: (a*=35.0, b*=-25.0), background (0, 0)
        prev_chroma = np.zeros((H, W, 2), dtype=np.float32)
        prev_chroma[20:40, 20:40] = [35.0, -25.0]

        # Backward warp chroma into current coordinates
        warped = warp_chroma_backward(prev_chroma, flow)
        self.assertEqual(warped.shape, (H, W, 2))

        # Check center of current square (35, 35)
        curr_center_chroma = warped[35, 35]
        np.testing.assert_allclose(curr_center_chroma, [35.0, -25.0], atol=2.0)

        # Check background area far from square is still close to 0
        bg_chroma = warped[5, 5]
        np.testing.assert_allclose(bg_chroma, [0.0, 0.0], atol=1.0)


class TestBidirectionalTemporalBlending(unittest.TestCase):
    """Unit tests for multi-keyframe bidirectional temporal blending."""

    def test_bidirectional_temporal_blending(self) -> None:
        """Test with 2 keyframes at frames 0 and 10; verify frame 5 receives a 50/50 blend."""
        H, W = 32, 32
        T = 11  # Frames 0..10

        # Uniform static grayscale frames
        frames_gray = [np.full((H, W), 128, dtype=np.uint8) for _ in range(T)]

        # Keyframe 0: Greenish (a*=-30, b*=20)
        # Keyframe 10: Purplish (a*=40, b*=-20)
        # Using apply_strict_l_lock to generate exact keyframe RGB
        chroma_k0 = np.full((H, W, 2), [-30.0, 20.0], dtype=np.float32)
        chroma_k10 = np.full((H, W, 2), [40.0, -20.0], dtype=np.float32)

        rgb_k0 = apply_strict_l_lock(frames_gray[0], chroma_k0)
        rgb_k10 = apply_strict_l_lock(frames_gray[10], chroma_k10)

        keyframes_color = {0: rgb_k0, 10: rgb_k10}

        output_frames = propagate_shot_chroma(
            frames_gray=frames_gray,
            keyframes_color=keyframes_color,
            radius=3,
            eps=1e-3,
        )

        self.assertEqual(len(output_frames), T)

        # Convert output frames to LAB float32 to check chroma values
        def _get_frame_chroma(rgb: np.ndarray) -> np.ndarray:
            rgb_f = rgb.astype(np.float32) / 255.0
            lab = cv2.cvtColor(rgb_f, cv2.COLOR_RGB2Lab)
            return lab[..., 1:3]

        chroma_f0 = _get_frame_chroma(output_frames[0])
        chroma_f5 = _get_frame_chroma(output_frames[5])
        chroma_f10 = _get_frame_chroma(output_frames[10])

        # Verify keyframes retain their distinct chroma
        np.testing.assert_allclose(chroma_f0.mean(axis=(0, 1)), [-30.0, 20.0], atol=1.5)
        np.testing.assert_allclose(chroma_f10.mean(axis=(0, 1)), [40.0, -20.0], atol=1.5)

        # Expected 50/50 blend at frame 5:
        # a* = 0.5 * (-30) + 0.5 * (40) = 5.0
        # b* = 0.5 * (20) + 0.5 * (-20) = 0.0
        mean_ch_f5 = chroma_f5.mean(axis=(0, 1))
        self.assertAlmostEqual(float(mean_ch_f5[0]), 5.0, delta=1.5)
        self.assertAlmostEqual(float(mean_ch_f5[1]), 0.0, delta=1.5)

        # Intermediate checks at frame 2 (80% k0, 20% k10) and frame 8 (20% k0, 80% k10)
        chroma_f2 = _get_frame_chroma(output_frames[2]).mean(axis=(0, 1))
        expected_f2 = 0.8 * np.array([-30.0, 20.0]) + 0.2 * np.array([40.0, -20.0])  # [-16, 12]
        np.testing.assert_allclose(chroma_f2, expected_f2, atol=2.0)

        chroma_f8 = _get_frame_chroma(output_frames[8]).mean(axis=(0, 1))
        expected_f8 = 0.2 * np.array([-30.0, 20.0]) + 0.8 * np.array([40.0, -20.0])  # [26, -12]
        np.testing.assert_allclose(chroma_f8, expected_f8, atol=2.0)


class TestStrictLuminanceLock(unittest.TestCase):
    """Unit tests for verbatim L* preservation ("Original pixels win")."""

    def test_strict_luminance_lock(self) -> None:
        """Verify that for every frame in output, the luminance channel matches the input grayscale frame within tight numerical tolerance (< 1.0 on 0-255 scale)."""
        H, W = 48, 48
        T = 4

        # Grayscale frames covering full dynamic range [0, 255]
        np.random.seed(123)
        frames_gray = []
        for _ in range(T):
            gray = np.random.randint(0, 256, (H, W), dtype=np.uint8)
            frames_gray.append(gray)

        # Reference keyframe with strong saturated colors
        k0_chroma = np.random.uniform(-40.0, 40.0, (H, W, 2)).astype(np.float32)
        k0_rgb = apply_strict_l_lock(frames_gray[0], k0_chroma)

        output_frames = propagate_shot_chroma(
            frames_gray=frames_gray,
            keyframes_color={0: k0_rgb},
            radius=4,
            eps=1e-3,
        )

        for t_idx in range(T):
            orig_gray = frames_gray[t_idx]
            out_rgb = output_frames[t_idx]

            # Compute L* of original gray frame
            orig_rgb_f = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            orig_lab = cv2.cvtColor(orig_rgb_f, cv2.COLOR_RGB2Lab)
            orig_L = orig_lab[..., 0]

            # Compute L* of reconstructed RGB output
            out_rgb_f = out_rgb.astype(np.float32) / 255.0
            out_lab = cv2.cvtColor(out_rgb_f, cv2.COLOR_RGB2Lab)
            out_L = out_lab[..., 0]

            # Guarantee: |L*(RGB_out) - L*(Gray_orig)| < 1.0 on [0, 100] CIE L* scale
            diff_L = np.abs(out_L - orig_L)
            max_diff = float(diff_L.max())
            mean_diff = float(diff_L.mean())

            self.assertLess(
                max_diff,
                1.0,
                f"Frame {t_idx} violated strict L* lock: max diff {max_diff:.4f} >= 1.0 (mean {mean_diff:.4f})",
            )


class TestTexturePreservingMaskedChromaTransfer(unittest.TestCase):
    """Unit tests for texture-preserving object mask chroma transfer and anchoring."""

    def test_texture_preserving_masked_object_transfer(self) -> None:
        """Verify object mask retains internal shading/texture rather than collapsing to a flat single color."""
        H, W = 60, 60
        T = 3

        # Create textured keyframe 0 with folds / shading variation inside the object mask
        yy, xx = np.indices((H, W))
        sin_texture = (np.sin(xx * 0.25) * 15.0).astype(np.float32)  # Folds amplitude +/- 15

        # Grayscale frames with shading folds inside the vest
        base_gray = np.full((H, W), 100, dtype=np.float32)
        base_gray[20:45, 20:45] += sin_texture[20:45, 20:45]
        gray_u8 = np.clip(np.round(base_gray), 0, 255).astype(np.uint8)
        frames_gray = [gray_u8.copy() for _ in range(T)]

        # Register mask archive with 1 object ("textured_vest", obj_id=1)
        archive = ShotMaskArchive("shot_vest", frame_count=T, height=H, width=W)
        obj_meta = MaskObjectMetadata(obj_id=1, name="textured_vest", entity_type="object")
        archive.add_object(obj_meta)

        # Mask rectangle (20:45, 20:45) across all frames
        mask_rect = np.zeros((H, W), dtype=np.uint8)
        mask_rect[20:45, 20:45] = 255
        for t in range(T):
            archive.set_mask(t, 1, mask_rect)

        chroma_k0 = np.zeros((H, W, 2), dtype=np.float32)
        # Background: neutral (0, 0)
        # Inside vest: mean a*=10 + sin_texture, b* = 5
        chroma_k0[20:45, 20:45, 0] = 10.0 + sin_texture[20:45, 20:45]
        chroma_k0[20:45, 20:45, 1] = 5.0

        raw_vest_a = chroma_k0[22:43, 22:43, 0]
        initial_vest_std = float(np.std(raw_vest_a))
        self.assertGreater(initial_vest_std, 8.0, "Initial texture must have strong standard deviation")

        rgb_k0 = apply_strict_l_lock(frames_gray[0], chroma_k0)

        # Color Bible anchor setting target median to (a*=35.0, b*=-15.0)
        color_bible_anchors = {
            1: {"a": 35.0, "b": -15.0}
        }

        output_frames = propagate_shot_chroma(
            frames_gray=frames_gray,
            keyframes_color={0: rgb_k0},
            mask_archive=archive,
            color_bible_anchors=color_bible_anchors,
            radius=3,
            eps=1e-3,
        )

        # Check frame 1 output inside object mask
        out_f1_rgb = output_frames[1]
        out_f1_lab = cv2.cvtColor(out_f1_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        vest_out_a = out_f1_lab[22:43, 22:43, 1]
        vest_out_b = out_f1_lab[22:43, 22:43, 2]

        # 1. Verify median is anchored close to Color Bible target (35.0, -15.0)
        median_out_a = float(np.median(vest_out_a))
        median_out_b = float(np.median(vest_out_b))
        self.assertAlmostEqual(median_out_a, 35.0, delta=4.0)
        self.assertAlmostEqual(median_out_b, -15.0, delta=4.0)

        # 2. Verify texture/shading variation was PRESERVED and did NOT collapse to flat 0
        out_vest_std = float(np.std(vest_out_a))
        self.assertGreater(
            out_vest_std,
            initial_vest_std * 0.70,
            f"Object chroma texture collapsed! out_std={out_vest_std:.2f} vs initial={initial_vest_std:.2f}",
        )


class TestSyntheticShotEndToEnd(unittest.TestCase):
    """End-to-end integration test on synthetic 6-frame motion sequence."""

    def test_synthetic_shot_end_to_end(self) -> None:
        """Run full propagate_shot_chroma on a 6-frame synthetic clip with moving circle + background gradient, verifying valid RGB output without color bleeding across mask boundary."""
        H, W = 80, 80
        T = 6

        archive = ShotMaskArchive("shot_moving_circle", frame_count=T, height=H, width=W)
        circle_obj = MaskObjectMetadata(obj_id=1, name="red_circle", entity_type="object")
        archive.add_object(circle_obj)

        yy, xx = np.indices((H, W))
        bg_gradient = np.tile(np.linspace(60, 200, W, dtype=np.uint8), (H, 1))

        frames_gray = []
        for t in range(T):
            cx = 25 + t * 5
            cy = 40
            circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= 12**2
            archive.set_mask(t, 1, circle_mask.astype(np.uint8) * 255)

            frame = bg_gradient.copy()
            frame[circle_mask] = 110  # Distinct object luminance
            frames_gray.append(frame)

        # Keyframe 0: Blue background (a*=-15, b*=-35), Red circle (a*=45, b*=25)
        cx0, cy0 = 25, 40
        circle0 = ((xx - cx0) ** 2 + (yy - cy0) ** 2) <= 12**2
        chroma_k0 = np.zeros((H, W, 2), dtype=np.float32)
        chroma_k0[~circle0] = [-15.0, -35.0]
        chroma_k0[circle0] = [45.0, 25.0]

        rgb_k0 = apply_strict_l_lock(frames_gray[0], chroma_k0)

        # Run complete pipeline
        output_frames = propagate_shot_chroma(
            frames_gray=frames_gray,
            keyframes_color={0: rgb_k0},
            mask_archive=archive,
            radius=4,
            eps=1e-3,
        )

        self.assertEqual(len(output_frames), T)

        # Verify all frames
        for t, out_rgb in enumerate(output_frames):
            self.assertEqual(out_rgb.shape, (H, W, 3))
            self.assertEqual(out_rgb.dtype, np.uint8)

            # Strict luminance lock verified on all frames
            orig_L = cv2.cvtColor(
                cv2.cvtColor(frames_gray[t], cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0,
                cv2.COLOR_RGB2Lab,
            )[..., 0]
            out_L = cv2.cvtColor(out_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)[..., 0]
            max_diff_L = float(np.abs(out_L - orig_L).max())
            self.assertLess(max_diff_L, 1.0)

        # Examine final frame 5: circle moved to cx = 25 + 5*5 = 50
        cx5, cy5 = 50, 40
        out_f5 = output_frames[5]
        out_f5_lab = cv2.cvtColor(out_f5.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)

        # 1. Circle center at (50, 40) must be distinctly red (a* > 25, b* > 10)
        center_a = float(out_f5_lab[cy5, cx5, 1])
        center_b = float(out_f5_lab[cy5, cx5, 2])
        self.assertGreater(center_a, 25.0, f"Circle center must be red: got a*={center_a:.2f}")
        self.assertGreater(center_b, 10.0, f"Circle center must be warm: got b*={center_b:.2f}")

        # 2. Background at (15, 15) must be distinctly blue (b* < -20)
        bg_b = float(out_f5_lab[15, 15, 2])
        self.assertLess(bg_b, -20.0, f"Background must be blue: got b*={bg_b:.2f}")

        # 3. No color bleeding: 2 pixels outside circle edge should be blue (b* < 0, a* < 0)
        outside_edge_a = float(out_f5_lab[cy5, cx5 + 15, 1])
        outside_edge_b = float(out_f5_lab[cy5, cx5 + 15, 2])
        self.assertLess(outside_edge_a, 0.0, "Outside circle edge must not bleed red")
        self.assertLess(outside_edge_b, -10.0, "Outside circle edge must retain blue background")


class TestTopLevelPipelineAndCLI(unittest.TestCase):
    """Unit tests for colorize_shot_osi_masked_lab file I/O and CLI execution."""

    def test_pipeline_directory_and_video_writing(self) -> None:
        """Test colorize_shot_osi_masked_lab with directory output and video writing."""
        H, W = 32, 32
        frames = [np.full((H, W), 120, dtype=np.uint8) for _ in range(4)]
        key_img = np.full((H, W, 3), [180, 50, 50], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Test video file output (.mp4)
            video_out = tmp_path / "output.mp4"
            res_vid = colorize_shot_osi_masked_lab(
                source=frames,
                keyframes={0: key_img},
                output_path=video_out,
                fps=24.0,
            )
            self.assertEqual(len(res_vid), 4)
            self.assertTrue(video_out.exists())
            self.assertGreater(video_out.stat().st_size, 0)

            # 2. Test directory output (frames)
            dir_out = tmp_path / "frames_out"
            res_dir = colorize_shot_osi_masked_lab(
                source=frames,
                keyframes={0: key_img},
                output_path=dir_out,
            )
            self.assertEqual(len(res_dir), 4)
            written_pngs = sorted(list(dir_out.glob("frame_*.png")))
            self.assertEqual(len(written_pngs), 4)

    def test_cli_execution_with_json_args(self) -> None:
        """Test main() CLI entry point with simulated argv."""
        H, W = 32, 32
        frames = [np.full((H, W), 100, dtype=np.uint8) for _ in range(3)]
        key_img = np.full((H, W, 3), [60, 160, 60], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Write source frames to directory
            src_dir = tmp_path / "source_frames"
            src_dir.mkdir()
            for idx, f in enumerate(frames):
                cv2.imwrite(str(src_dir / f"f_{idx:03d}.png"), f)

            # Write keyframe image
            ref_path = tmp_path / "ref_0.png"
            cv2.imwrite(str(ref_path), cv2.cvtColor(key_img, cv2.COLOR_RGB2BGR))

            # Output path
            out_file = tmp_path / "cli_output.mp4"

            # Execute CLI
            refs_json = json.dumps({"0": str(ref_path)})
            cli_argv = [
                "--source",
                str(src_dir),
                "--references",
                refs_json,
                "--output",
                str(out_file),
                "--preset",
                "ultrafast",
            ]
            exit_code = main(cli_argv)
            self.assertEqual(exit_code, 0)
            self.assertTrue(out_file.exists())
            self.assertGreater(out_file.stat().st_size, 0)

    def test_error_handling_and_validation(self) -> None:
        """Test fail-fast validation for invalid keyframes, empty sources, and dimension mismatches."""
        H, W = 32, 32
        frames = [np.zeros((H, W), dtype=np.uint8)]
        valid_key = np.zeros((H, W, 3), dtype=np.uint8)

        # Empty source
        with self.assertRaises(ValueError):
            colorize_shot_osi_masked_lab(source=[], keyframes={0: valid_key})

        # Keyframe index out of range
        with self.assertRaises(ValueError):
            colorize_shot_osi_masked_lab(source=frames, keyframes={5: valid_key})

        # Keyframe spatial mismatch
        mismatched_key = np.zeros((H + 10, W, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            colorize_shot_osi_masked_lab(source=frames, keyframes={0: mismatched_key})

    @mock.patch("cv2.VideoWriter.isOpened", return_value=False)
    def test_videowriter_unopened_raises_runtime_error(self, mock_is_opened: mock.MagicMock) -> None:
        """Verify RuntimeError is raised when cv2.VideoWriter fails to open for both codecs."""
        from scripts.osi_masked_lab import _save_output_frames

        dummy_frames = [np.zeros((32, 32, 3), dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_video = Path(tmp_dir) / "output.mp4"
            with self.assertRaises(RuntimeError) as ctx:
                _save_output_frames(dummy_frames, out_video)
            self.assertIn("Failed to open cv2.VideoWriter", str(ctx.exception))
            self.assertIn("mp4v and XVID codecs", str(ctx.exception))


class TestColorBibleAnchorResolution(unittest.TestCase):
    """Unit tests for _resolve_color_anchor enhanced parsing."""

    def test_resolve_percentile_dict(self) -> None:
        from scripts.osi_masked_lab import _resolve_color_anchor

        obj = MaskObjectMetadata(obj_id=1, name="character_suit", entity_type="character", entity_id="char_01", variant_id="costume_a")

        # Percentile list in "a" and "b"
        anchors = {
            "char_01": {
                "L": [20.0, 40.0, 50.0, 60.0, 80.0],
                "a": [10.0, 15.0, 25.0, 35.0, 40.0],
                "b": [-20.0, -10.0, -5.0, 0.0, 10.0],
            }
        }
        res = _resolve_color_anchor(anchors, obj)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res[0], 25.0)
        self.assertAlmostEqual(res[1], -5.0)

    def test_resolve_scalar_dict(self) -> None:
        from scripts.osi_masked_lab import _resolve_color_anchor

        obj = MaskObjectMetadata(obj_id=2, name="hat", entity_type="object", entity_id="obj_hat")
        anchors = {
            "hat": {"a": 18.2, "b": -33.4}
        }
        res = _resolve_color_anchor(anchors, obj)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res[0], 18.2)
        self.assertAlmostEqual(res[1], -33.4)

    def test_resolve_median_and_p50(self) -> None:
        from scripts.osi_masked_lab import _resolve_color_anchor

        obj1 = MaskObjectMetadata(obj_id=3, name="car", entity_type="object")
        anchors1 = {
            3: {"median_a": 12.0, "median_b": 42.0}
        }
        res1 = _resolve_color_anchor(anchors1, obj1)
        self.assertEqual(res1, (12.0, 42.0))

        obj2 = MaskObjectMetadata(obj_id=4, name="sky", entity_type="location")
        anchors2 = {
            "4": {"p50_a": [-5.0, 0.0, 5.0], "p50_b": [-20.0, -15.0, -10.0]}
        }
        res2 = _resolve_color_anchor(anchors2, obj2)
        self.assertEqual(res2, (0.0, -15.0))


class TestColourCommandWiring(unittest.TestCase):
    """Verify _colour_command includes osi_masked_lab flags: compliance dir, color bible db, film id."""

    def test_colour_command_osi_masked_lab_flags(self) -> None:
        from ai_remaster_gui.server import PipelineApp
        from ai_remaster_gui.config import ROOT

        app = PipelineApp()
        app.settings["global"]["color_bible_db"] = str(ROOT / "test_cb.db")
        app.settings["global"]["color_bible_film_id"] = "test_film"

        values = {
            "manifest": "manifests/test.csv",
            "method": "osi_masked_lab",
            "masks_dir": "intermediate/shot_masks",
            "guided_radius": "7",
            "guided_eps": "0.002",
        }
        cmd = app._colour_command({}, values)

        self.assertIn("--method", cmd)
        self.assertEqual(cmd[cmd.index("--method") + 1], "osi_masked_lab")
        self.assertIn("--compliance-output-dir", cmd)
        self.assertEqual(cmd[cmd.index("--compliance-output-dir") + 1], str(ROOT / "output"))
        self.assertIn("--color-bible-db", cmd)
        self.assertEqual(cmd[cmd.index("--color-bible-db") + 1], str(ROOT / "test_cb.db"))
        self.assertIn("--film-id", cmd)
        self.assertEqual(cmd[cmd.index("--film-id") + 1], "test_film")
        self.assertIn("--masks-dir", cmd)
        self.assertEqual(cmd[cmd.index("--masks-dir") + 1], "intermediate/shot_masks")
        self.assertIn("--guided-radius", cmd)
        self.assertEqual(cmd[cmd.index("--guided-radius") + 1], "7")
        self.assertIn("--guided-eps", cmd)
        self.assertEqual(cmd[cmd.index("--guided-eps") + 1], "0.002")


class TestColorizeVideoColorBibleIntegration(unittest.TestCase):
    """Verify scripts/colorize_video.py wiring to Color Bible DB and SAM2 masks."""

    def test_build_parser_includes_color_bible_flags(self) -> None:
        from scripts.colorize_video import build_parser

        parser = build_parser()
        args = parser.parse_args(["--manifest", "dummy.csv"])
        self.assertTrue(hasattr(args, "color_bible_db"))
        self.assertTrue(hasattr(args, "film_id"))
        self.assertIn("color_bible.db", args.color_bible_db)
        self.assertIsNone(args.film_id)

        args2 = parser.parse_args([
            "--manifest", "dummy.csv",
            "--color-bible-db", "custom.db",
            "--film-id", "film_42",
        ])
        self.assertEqual(args2.color_bible_db, "custom.db")
        self.assertEqual(args2.film_id, "film_42")

    def test_run_osi_masked_lab_resolves_film_id_from_manifest_comment(self) -> None:
        from scripts.colorize_video import build_parser, run_osi_masked_lab_colorization

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_file = tmp_path / "video.mp4"
            source_file.write_bytes(b"dummy video content")
            ref_file = tmp_path / "ref.png"
            ref_file.write_bytes(b"dummy image")
            manifest_file = tmp_path / "shots.csv"
            manifest_file.write_text(
                f"# source_video={source_file.as_posix()}\n"
                "# color_bible_film_id=film_from_comment\n"
                "index,start,end,reference,enabled\n"
                f"0,0.0,1.0,{ref_file.as_posix()},true\n",
                encoding="utf-8",
            )
            parser = build_parser()
            args = parser.parse_args([
                "--manifest", str(manifest_file),
                "--source-video", str(source_file),
                "--method", "osi_masked_lab",
                "--dry-run",
            ])
            # Dry run returns 0 without processing frames
            with mock.patch("scripts.colorize_video.video_info", return_value={"width": 64, "height": 64, "fps": 24.0, "frames": 24}):
                exit_code = run_osi_masked_lab_colorization(args)
            self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
