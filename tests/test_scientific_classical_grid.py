"""Run the actual classical ARP model matrix, not fake prefilled candidates."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import restoration_evaluation as eval_tool
import run_scientific_classical_grid as grid


class ScientificGridTests(unittest.TestCase):
    def test_real_pipeline_algorithms_on_same_known_corruption(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-eval-grid-") as td:
            root = Path(td)
            source = root / "source.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=96x72:rate=12:duration=1",
                 "-c:v", "ffv1", str(source)],
                check=True, capture_output=True, timeout=40,
            )
            stage = root / "degraded"
            eval_tool.synthesize(
                source=source, output_dir=stage, film="testfilm", scene="edges",
                kind="gaussian_blur", preset="gentle", ffmpeg=ffmpeg,
                ffprobe=ffprobe,
            )
            result = grid.run(
                manifest=stage / "manifest.json",
                output_dir=root / "matrix",
                methods=["off", "cas", "richardson_lucy"],
                presets=["gentle"], mix=100,
                ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            scores = result["scores"]
            self.assertEqual(set(scores), {
                "degraded_baseline", "off_gentle",
                "cas_gentle", "richardson_lucy_gentle",
            })
            self.assertAlmostEqual(scores["off_gentle"]["delta_psnr_db_vs_degraded"], 0., places=4)
            self.assertEqual(scores["cas_gentle"]["frames"], 12)
            self.assertEqual(scores["richardson_lucy_gentle"]["frames"], 12)
            for method in ("off", "cas", "richardson_lucy"):
                p = root / "matrix" / f"{method}_gentle.mkv"
                self.assertTrue(p.is_file())
                self.assertTrue(p.with_suffix(".mkv.sig.json").is_file())
            self.assertTrue((root / "matrix" / "evaluation" / "evaluation.json").is_file())
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                grid.run(
                    manifest=stage / "manifest.json", output_dir=root / "matrix",
                    methods=["cas"], presets=["gentle"], ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )

    def test_invalid_grid_methods_are_rejected(self):
        with self.assertRaises(ValueError):
            grid.parse_names("cas,cas", grid.METHODS)
        with self.assertRaises(ValueError):
            grid.parse_names("untrusted", grid.METHODS)
        with self.assertRaises(ValueError):
            grid.parse_names("gentle,strong,bad", grid.PRESETS)


if __name__ == "__main__":
    unittest.main()
