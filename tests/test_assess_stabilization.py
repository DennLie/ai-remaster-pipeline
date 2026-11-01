"""Tests for the archival stabilization measurement contract (no ML models)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assess_stabilization as assess  # noqa: E402


class StabilizationQaTests(unittest.TestCase):
    def test_feature_motion_recovers_known_translation(self):
        rng = np.random.default_rng(14)
        source = rng.integers(30, 230, (192, 256, 3), dtype=np.uint8)
        translate = np.float32([[1, 0, 4], [0, 1, -3]])
        target = cv2.warpAffine(source, translate, (256, 192), borderMode=cv2.BORDER_CONSTANT)
        measured = assess.estimate_motion(source, target)
        self.assertIsNotNone(measured, "Known textured global translation should be measurable")
        self.assertAlmostEqual(measured["tx_px"], 4, delta=0.6)
        self.assertAlmostEqual(measured["ty_px"], -3, delta=0.6)
        self.assertGreater(measured["inlier_points"], 24)

    def test_flat_frame_returns_untrusted_no_fake_zero(self):
        black = np.full((96, 128, 3), 60, np.uint8)
        self.assertIsNone(assess.estimate_motion(black, black))
        summary = assess.summarize_motion([None, None])
        self.assertIsNone(summary["median_translation_px"])
        self.assertEqual(summary["rejected_adjacent_pairs"], 2)

    def test_aligned_video_produces_report_and_rejects_mismatches(self):
        ffmpeg = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg, "CI needs FFmpeg for an actual encoded-video test")
        rng = np.random.default_rng(1234)
        texture = rng.integers(40, 200, (96, 128, 3), np.uint8)
        with tempfile.TemporaryDirectory(prefix="stabilization-measurement-") as td:
            root = Path(td)
            source = root / "source.mkv"
            copy = root / "copy.mkv"
            shorter = root / "shorter.mkv"
            base = [cv2.warpAffine(texture, np.float32([[1,0,i%3-1],[0,1,(i%2)*2-1]]),
                                   (128,96)) for i in range(5)]
            def encode(path, frames):
                cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "128x96",
                       "-r", "10", "-i", "pipe:0", "-an", "-c:v", "ffv1", str(path)]
                subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames),
                               capture_output=True, check=True, timeout=30)
            encode(source, base)
            encode(copy, base)
            encode(shorter, base[:4])
            result = assess.evaluate(source, {"same": copy}, root / "qa", cut_after_frames={2})
            self.assertEqual(result["frames"], 5)
            self.assertEqual(result["cut_after_frames_excluded"], [2])
            self.assertEqual(result["measurements"]["same"]["trusted_adjacent_pairs"],
                             result["measurements"]["original"]["trusted_adjacent_pairs"])
            self.assertTrue((root / "qa" / "stabilization_contact_sheet.jpg").exists())
            self.assertEqual(json.loads((root / "qa" / "stabilization_qa.json").read_text())["frames"], 5)
            with self.assertRaisesRegex(ValueError, "ended before frame"):
                assess.evaluate(source, {"short": shorter}, root / "wrong")

    def test_rejects_different_geometry(self):
        original = np.zeros((96, 128, 3), np.uint8)
        with self.assertRaises(ValueError):
            assess.estimate_motion(original, np.zeros((64, 128, 3), np.uint8))


if __name__ == "__main__":
    unittest.main()
