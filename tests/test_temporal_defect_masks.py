"""Synthetic, CPU-only tests for ARP's independent film damage masks."""

from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cleanup_video  # noqa: E402
import film_defect_masks as masks  # noqa: E402
from ai_remaster_gui import app  # noqa: E402
from ai_remaster_gui.runtime_settings import base_settings  # noqa: E402


def texture(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = np.clip(rng.normal(120, 9, size=(96, 128)), 0, 255).astype(np.uint8)
    gray[30:54, 80:86] = 180  # Real narrow scene feature must be retained.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


class TemporalFilmDefectTests(unittest.TestCase):
    def test_clean_static_video_has_no_mask(self):
        frame = texture()
        mask = masks.temporal_defect_mask(frame, frame, frame)
        self.assertEqual(mask.shape, (96, 128))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_isolated_transient_dust_is_localized(self):
        frame = texture()
        damaged = frame.copy()
        damaged[41:46, 27:32] = 245
        mask = masks.temporal_defect_mask(frame, damaged, frame, 0.65)
        self.assertGreater(int(np.count_nonzero(mask[40:47, 26:33])), 2)
        self.assertEqual(int(np.count_nonzero(mask[30:54, 80:86])), 0)
        self.assertLess(np.count_nonzero(mask) / mask.size, 0.025)

    def test_scene_cut_is_not_treated_as_damage(self):
        before = np.zeros((96, 128, 3), dtype=np.uint8)
        current = np.full_like(before, 255)
        after = np.zeros_like(before)
        mask = masks.temporal_defect_mask(before, current, after)
        self.assertFalse(np.any(mask))

    def test_moderate_archival_cut_abstains_before_optical_flow(self):
        before = texture()
        after = np.clip(before.astype(np.int16) + 30, 0, 255).astype(np.uint8)
        # A 30-level median luminance jump occurs at several hard cuts in
        # archival footage. No motion estimation should cross this boundary.
        with mock.patch.object(masks, 'aligned_neighbor', side_effect=AssertionError('flow across cut')):
            result = masks.temporal_defect_mask(before, after, after)
        self.assertFalse(np.any(result))

    def test_presentation_bars_and_borders_are_excluded(self):
        frame = texture()
        frame[:8] = 0
        frame[-8:] = 0
        damaged = frame.copy()
        damaged[0:5, 25:30] = 255
        mask = masks.temporal_defect_mask(frame, damaged, frame)
        self.assertFalse(np.any(mask[:3]))
        self.assertFalse(np.any(mask[-3:]))

    def test_different_geometry_fails_explicitly(self):
        frame = texture()
        with self.assertRaises(ValueError):
            masks.temporal_defect_mask(frame, frame[:80], frame)

    def test_sensitivity_is_bounded(self):
        frame = texture()
        with self.assertRaises(ValueError):
            masks.temporal_defect_mask(frame, frame, frame, 1.5)

    def test_analysis_dimensions_are_even_and_bounded(self):
        self.assertEqual(masks.defect_mask_size(4096, 3112, "720"), (948, 720))
        self.assertEqual(masks.defect_mask_size(640, 480, "1080"), (640, 480))
        self.assertEqual(masks.defect_mask_size(640, 480, "source"), (640, 480))

    def test_cli_accepts_mask_only_without_ltx(self):
        args = cleanup_video.build_parser().parse_args([
            "--source", "input/film.mp4", "--detect-defects", "--no-dearchive",
        ])
        self.assertTrue(args.detect_defects)
        self.assertFalse(args.ai_descratch)
        self.assertFalse(args.dearchive)

    def test_gui_and_worker_agree_on_mask_output_and_cli(self):
        settings = base_settings()
        settings["global"].update({"source": "input/film.mp4", "cleanup": "true"})
        settings["cleanup"].update({
            "detect_defects": "true", "dearchive": "false", "ai_descratch": "false",
        })
        old = app.APP.settings
        try:
            app.APP.settings = settings
            command = app.APP.command_for("cleanup")
            self.assertIn("--detect-defects", command)
            self.assertIn("--no-dearchive", command)
            self.assertNotIn("--ai-descratch", command)
            self.assertEqual(command[command.index("--defect-mask-height") + 1], "720")
            outputs = app.APP.expected_outputs("cleanup")
            self.assertEqual(len(outputs), 2)
            self.assertTrue(outputs[1].endswith("_defect_mask.mp4"))
            self.assertEqual(
                Path(outputs[1]).name,
                cleanup_video.defect_mask_preview_path(Path(outputs[0])).name,
            )
            args = cleanup_video.build_parser().parse_args(command[3:])
            self.assertEqual(
                Path(outputs[0]).name,
                cleanup_video.default_output(Path(args.source), args).name,
            )
        finally:
            app.APP.settings = old


if __name__ == "__main__":
    unittest.main()
