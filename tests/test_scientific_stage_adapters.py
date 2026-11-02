"""Step 1/2/4/5 true ARP entrypoint tests on synthetic, frame-exact FFV1 footage."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import restoration_evaluation as ev
import scientific_stage_adapters as stages
import scientific_stabilization_eval as motion


class ScientificStageAdapterTests(unittest.TestCase):
    def test_exact_mask_precision_recall_f1_contract(self):
        a = np.zeros((8, 12), np.uint8)
        b = a.copy()
        a[:4, 2:4] = 255
        b[:4, 3:5] = 255
        x = stages.mask_scores([a], [b])
        self.assertEqual(x["tp_pixels"], 4)
        self.assertEqual(x["fp_pixels"], 4)
        self.assertEqual(x["fn_pixels"], 4)
        self.assertAlmostEqual(x["pixel_precision"], 0.5)
        self.assertAlmostEqual(x["pixel_recall"], 0.5)
        self.assertAlmostEqual(x["pixel_f1"], 0.5)
        with self.assertRaises(ValueError):
            stages.mask_scores([a], [a, b])
        with self.assertRaises(ValueError):
            stages.parse_methods("hqdn3d,hqdn3d")

    def test_real_ffmpeg_deflicker_denoise_and_scratch_detection(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-steps125-") as td:
            root = Path(td)
            source = root / "original.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=12:duration=2",
                 "-c:v", "ffv1", str(source)],
                check=True, capture_output=True, timeout=40,
            )
            for kind, methods in [
                ("temporal_flicker", ["off", "ffmpeg_deflicker"]),
                ("read_noise", ["off", "hqdn3d", "nlmeans"]),
            ]:
                pack = root / kind
                ev.synthesize(source=source, output_dir=pack, film="film_test",
                              scene="test_movement", kind=kind, preset="gentle",
                              ffmpeg=ffmpeg, ffprobe=ffprobe)
                grid = stages.run_grid(
                    manifest=pack / "manifest.json",
                    output_dir=root / (kind + "_grid"),
                    method_names=methods, presets=["gentle"], ffmpeg=ffmpeg, ffprobe=ffprobe,
                )
                self.assertEqual(set(grid["scores"]), {
                    "degraded_baseline", *(x + "_gentle" for x in methods),
                })
                self.assertEqual(grid["scores"]["off_gentle"]["frames"], 24)
                self.assertAlmostEqual(grid["scores"]["off_gentle"]["delta_psnr_db_vs_degraded"], 0., places=4)
                for m in methods:
                    file = root / (kind + "_grid") / (m + "_gentle.mkv")
                    self.assertEqual(ev.probe(file, ffprobe)["frames"], 24)
                    self.assertTrue(file.with_suffix(".mkv.sig.json").exists())
                with self.assertRaises(FileExistsError):
                    stages.run_grid(
                        manifest=pack / "manifest.json",
                        output_dir=root / (kind + "_grid"),
                        method_names=methods, presets=["gentle"],
                        ffmpeg=ffmpeg, ffprobe=ffprobe,
                    )
            mask_source = root / "vertical_scratches"
            ev.synthesize(source=source, output_dir=mask_source, film="film_test",
                          scene="scratch", kind="vertical_scratches", preset="balanced",
                          ffmpeg=ffmpeg, ffprobe=ffprobe)
            measured = stages.evaluate_scratch_mask(
                manifest=mask_source / "manifest.json",
                output_dir=root / "scratch_scores",
                ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            self.assertEqual(measured["frames"], 24)
            self.assertGreater(measured["metrics"]["ground_truth_fraction"], 0.)
            self.assertGreaterEqual(measured["metrics"]["pixel_f1"], 0.)
            self.assertLessEqual(measured["metrics"]["pixel_f1"], 1.)
            self.assertTrue((root / "scratch_scores" / "mask_evaluation.json").is_file())
            with self.assertRaisesRegex(ValueError, "Scratch detector needs"):
                stages.evaluate_scratch_mask(
                    manifest=root / "temporal_flicker" / "manifest.json",
                    output_dir=root / "not_scored",
                    ffmpeg=ffmpeg, ffprobe=ffprobe,
                )
            with self.assertRaisesRegex(ValueError, "Grid only supports"):
                stages.run_grid(
                    manifest=mask_source / "manifest.json",
                    output_dir=root / "wrong_grid", method_names=["off"],
                    presets=["gentle"], ffmpeg=ffmpeg, ffprobe=ffprobe,
                )

    def test_known_jitter_and_reference_validation(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-step4-") as td:
            root = Path(td)
            src = root / "clean.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12:duration=2",
                 "-c:v", "ffv1", str(src)],
                check=True, capture_output=True, timeout=40,
            )
            one, two = root / "a", root / "b"
            m = motion.prepare(source=src, output_dir=one, film="a",
                               scene="one", preset="balanced", seed=7,
                               ffmpeg=ffmpeg, ffprobe=ffprobe)
            motion.prepare(source=src, output_dir=two, film="a",
                           scene="one", preset="balanced", seed=7,
                           ffmpeg=ffmpeg, ffprobe=ffprobe)
            # Lossless encodes are bitwise identical on same FFmpeg instance.
            self.assertEqual(m["jitter_sha256"], ev.digest(two / "jitter.mkv"))
            self.assertEqual(m["frames"], 24)
            self.assertEqual(len(m["translation_per_frame"]), 24)
            baseline = motion.evaluate(manifest=one / "manifest.json",
                                       candidate=None, output_dir=root / "baseline",
                                       ffmpeg=ffmpeg, ffprobe=ffprobe)
            self.assertGreater(baseline["metrics"]["damaged_baseline"]["estimated_translation_rmse_pixels"], 0.1)
            # Simulate analytically perfect source restoration for reference
            # orientation check (not a claimed actual stabilizer result).
            corrected = root / "perfect.mkv"
            shutil.copyfile(one / "reference.mkv", corrected)
            corrected.with_suffix(".mkv.sig.json").write_text(json.dumps(
                {"source_fingerprint": {"sha256": m["jitter_sha256"]}}))
            report = motion.evaluate(manifest=one / "manifest.json",
                                     candidate=corrected, output_dir=root / "corrected",
                                     ffmpeg=ffmpeg, ffprobe=ffprobe)
            self.assertLess(report["metrics"]["candidate"]["reference_roi_mse_rgb"],
                            report["metrics"]["damaged_baseline"]["reference_roi_mse_rgb"])
            self.assertAlmostEqual(report["metrics"]["candidate"]["reference_roi_mse_rgb"], 0.)
            available_filters = subprocess.run(
                [ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True, check=True).stdout
            if "vidstabdetect" in available_filters and "vidstabtransform" in available_filters:
                # Actual ARP stabilizer, not just a hand-made perfect
                # answer. No claim it necessarily improves this clip.
                tested = motion.run_arp(
                    manifest=one / "manifest.json", output_dir=root / "arp_real",
                    smoothing=3, zoom=1., ffmpeg=ffmpeg, ffprobe=ffprobe)
                self.assertTrue((root / "arp_real" / "stabilized.mkv").is_file())
                self.assertEqual(tested["evaluation"]["candidate"]["frames"], 24)
                self.assertEqual(tested["jitter_sha256"], m["jitter_sha256"])
                self.assertIn("estimated_translation_rmse_pixels",
                              tested["evaluation"]["candidate"])
            corrected.with_suffix(".mkv.sig.json").write_text(json.dumps(
                {"source_fingerprint": {"sha256": "0" * 64}}))
            with self.assertRaisesRegex(ValueError, "source SHA mismatch"):
                motion.evaluate(manifest=one / "manifest.json", candidate=corrected,
                                output_dir=root / "bad_source", ffmpeg=ffmpeg, ffprobe=ffprobe)


if __name__ == "__main__":
    unittest.main()
