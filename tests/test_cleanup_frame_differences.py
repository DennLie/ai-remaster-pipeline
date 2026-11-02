"""Exact, source-bound FFmpeg frame difference chart tests (CPU only)."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import frame_differences as differences


class AppStub:
    def __init__(self, method: str = "ffmpeg_am", exists: bool = True):
        self.method = method
        self.exists = exists

    def cleanup_comparison_state(self):
        return {"method": self.method, "exists": "true" if self.exists else "false"}


class FFmpegFrameDifferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="arp_diff_")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.original = self.folder / "original.mkv"
        self.processed = self.folder / "processed.mkv"

    def _ffmpeg(self, *args):
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y", *map(str, args)],
            capture_output=True, text=True, timeout=25,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def _make_pair(self):
        self._ffmpeg("-f", "lavfi", "-i", "testsrc2=size=64x48:rate=10:duration=1.2",
                     "-an", "-c:v", "ffv1", self.original)
        self._ffmpeg("-i", self.original, "-vf", "eq=brightness=0.08", "-an",
                     "-c:v", "ffv1", self.processed)

    def test_difference_report_is_frame_aligned_and_cached(self):
        self._make_pair()
        with patch.object(differences, "_CACHE", self.folder / "cache"), \
             patch.object(differences, "_sources",
                          return_value=(self.original, self.processed)):
            app = AppStub()
            first = differences.cleanup_frame_differences(app)
            self.assertEqual(first["frames"], 12)
            self.assertEqual(len(first["samples"]), 12)
            self.assertFalse(first["sampled"])
            self.assertEqual([s["frame"] for s in first["samples"]], list(range(12)))
            self.assertTrue(all(0 <= s["mean_abs_luma"] <= 255 for s in first["samples"]))
            self.assertGreater(first["mean_abs_luma"], 0)
            self.assertEqual(first["metric"], "directional_mean_luma_8bit")
            self.assertGreater(first["mean_brighten_luma"], 0)
            self.assertTrue(all("brighten_mean_luma" in sample and "darken_mean_luma" in sample
                                for sample in first["samples"]))
            for sample in first["samples"]:
                self.assertAlmostEqual(sample["mean_abs_luma"],
                                       sample["brighten_mean_luma"] + sample["darken_mean_luma"],
                                       places=3)
            with patch.object(differences, "_compare_frames", side_effect=AssertionError("cache missed")):
                again = differences.cleanup_frame_differences(app)
            self.assertEqual(first, again)
            self.assertTrue(list((self.folder / "cache").glob("*.json")))

    def test_directional_luma_does_not_cancel_mixed_brightening_and_darkening(self):
        import numpy as np
        original = np.array([[100, 100], [100, 100]], dtype=np.uint8)
        processed = np.array([[140, 60], [130, 70]], dtype=np.uint8)
        up, down = differences._directional_luma(original, processed)
        self.assertEqual(up, 17.5)
        self.assertEqual(down, 17.5)
        self.assertEqual(up + down, 35.0)
        zero, zero_down = differences._directional_luma(original, original)
        self.assertEqual((zero, zero_down), (0.0, 0.0))
        with self.assertRaises(ValueError):
            differences._directional_luma(original, original[:1])

    def test_method_gate_does_not_allow_arbitrary_source(self):
        with self.assertRaises((FileNotFoundError, ValueError)):
            differences.cleanup_frame_differences(AppStub(exists=False))
        with self.assertRaises(ValueError):
            differences.cleanup_frame_differences(AppStub(method="arp"))

    def test_geometry_and_frame_count_mismatch_fail_closed(self):
        self._make_pair()
        shorter = self.folder / "shorter.mkv"
        self._ffmpeg("-i", self.original, "-frames:v", "8", "-an", "-c:v", "ffv1", shorter)
        with self.assertRaisesRegex(ValueError, "incompatible frames"):
            differences._validate_pair(self.original, shorter)
        wrong = self.folder / "wrong.mkv"
        self._ffmpeg("-i", self.original, "-vf", "scale=32:24", "-an", "-c:v", "ffv1", wrong)
        with self.assertRaisesRegex(ValueError, "incompatible width"):
            differences._validate_pair(self.original, wrong)

    def test_sparse_movie_chart_uses_real_exact_ffv1_samples(self):
        # Generates 2,430 frames at tiny resolution; compares only 240 exact
        # positions. Unlike short clips, it must not scan every decoded frame.
        self._ffmpeg(
            "-f", "lavfi", "-i",
            "testsrc2=size=32x24:rate=30:duration=81",
            "-an", "-c:v", "ffv1", self.original
        )
        self._ffmpeg(
            "-i", self.original, "-vf", "eq=brightness=0.06",
            "-an", "-c:v", "ffv1", self.processed
        )
        with patch.object(differences, "_CACHE", self.folder / "cache"), \
             patch.object(differences, "_sources", return_value=(
                 self.original, self.processed
             )):
            report = differences.cleanup_frame_differences(AppStub(method="localized_stream"))
        self.assertEqual(report["frames"], 2430)
        self.assertTrue(report["sampled"])
        self.assertEqual(len(report["samples"]), 240)
        self.assertEqual(report["samples"][0]["frame"], 0)
        self.assertEqual(report["samples"][-1]["frame"], 2429)
        self.assertGreater(report["mean_abs_luma"], 0)
        self.assertIn("NOT full-film", report["sampling_note"])
        self.assertTrue(all(
            x["frame"] < y["frame"] for x,y in zip(
                report["samples"], report["samples"][1:]
            )
        ))

    def test_refuse_same_file_and_bad_lengths(self):
        self._make_pair()
        with self.assertRaisesRegex(ValueError, "separate"):
            differences._validate_pair(self.original, self.original)
        with patch.object(differences, "_metadata", return_value={
            "width": 64, "height": 48, "fps_fraction": "10", "frames": 100000
        }):
            self.assertEqual(differences._validate_pair(self.original, self.processed), 100000)
            # Large films use verified, sparse frame samples instead of an
            # arbitrary 2,400-frame rejection or full-film decode for the UI.


if __name__ == "__main__":
    unittest.main()
