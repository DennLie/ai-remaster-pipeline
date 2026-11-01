"""Contract tests for frame-exact multi-method denoising comparison."""
from __future__ import annotations

import json
import subprocess
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_denoising as comparison  # noqa: E402


class CompareDenoisingTests(unittest.TestCase):
    def test_rejects_duplicate_method_and_bad_syntax(self):
        with self.assertRaises(ValueError):
            comparison.parse_options(["original=source.mkv"])
        with self.assertRaises(ValueError):
            comparison.parse_options(["x=one.mkv", "x=two.mkv"])
        with self.assertRaises(ValueError):
            comparison.parse_options(["../bad=one.mkv"])

    def test_video_comparison_writes_results_and_detects_missing_frames(self):
        ffmpeg = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, matching, short = root / "source.mkv", root / "same.mkv", root / "short.mkv"
            for path, duration in ((source, 1.0), (matching, 1.0), (short, 0.75)):
                subprocess.run([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", f"testsrc2=size=96x72:rate=12:duration={duration}",
                    "-an", "-c:v", "ffv1", str(path),
                ], check=True, capture_output=True, timeout=30)
            result = comparison.compare(source, {"same": matching}, root / "reports")
            self.assertEqual(result["frames"], 12)
            self.assertEqual(result["width"], 96)
            self.assertEqual(result["variants"]["same"]["mean_abs_change"], 0)
            self.assertTrue((root / "reports" / "denoising_comparison.jpg").is_file())
            report = json.loads((root / "reports" / "denoising_comparison.json").read_text())
            self.assertEqual(report["fps"], 12)
            with self.assertRaisesRegex(ValueError, "ended early"):
                comparison.compare(source, {"short": short}, root / "fail")


if __name__ == "__main__":
    unittest.main()
