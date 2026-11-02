"""Scientific evaluation integrity, reproducible damage, FFmpeg and film statistics."""
from __future__ import annotations

import hashlib
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
import restoration_evaluation as research  # noqa: E402


class ScientificRestorationEvaluationTests(unittest.TestCase):
    def test_deterministic_damage_and_explicit_scratch_mask(self):
        image = np.full((48, 64, 3), 125, np.uint8)
        for kind in research.KINDS:
            for preset in research.PRESETS:
                a, ma = research.corrupt(image, kind=kind, strength=preset, index=3, seed=97)
                b, mb = research.corrupt(image, kind=kind, strength=preset, index=3, seed=97)
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(ma, mb)
                self.assertEqual(a.shape, image.shape)
                self.assertEqual(a.dtype, np.uint8)
                self.assertTrue(np.all((a >= 0) & (a <= 255)))
                if kind == "vertical_scratches":
                    self.assertGreater(np.count_nonzero(ma), 0)
                    self.assertTrue(np.all(a[ma > 0] != image[ma > 0]))
                else:
                    self.assertEqual(np.count_nonzero(ma), 0)
        with self.assertRaises(ValueError):
            research.corrupt(image, kind="unknown", strength="gentle", index=0, seed=1)

    def test_known_degradation_metrics_have_correct_orientation(self):
        rng = np.random.default_rng(41)
        frames = [rng.integers(10, 245, (48, 64, 3), np.uint8) for _ in range(4)]
        noisy = [research.corrupt(x, kind="read_noise", strength="strong",
                                  index=i, seed=42)[0] for i, x in enumerate(frames)]
        same = research._metrics(frames, frames)
        bad = research._metrics(frames, noisy)
        self.assertTrue(same["perfect_rgb_match"])
        self.assertEqual(same["psnr_rgb_db"], 120.)
        self.assertAlmostEqual(same["ssim_rgb"], 1., places=6)
        self.assertFalse(bad["perfect_rgb_match"])
        self.assertLess(bad["psnr_rgb_db"], same["psnr_rgb_db"])
        self.assertLess(bad["ssim_rgb"], same["ssim_rgb"])
        self.assertGreater(bad["temporal_reference_error_rmse"], 0.)

    def test_real_ffmpeg_synthetic_reference_stage_and_candidate_provenance(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-scientific-eval-") as d:
            root = Path(d)
            source = root / "film_excerpt.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=96x72:rate=12:duration=1",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source)],
                check=True, capture_output=True, timeout=40,
            )
            source_sha = research.digest(source)
            for kind in research.KINDS:
                synth_dir = root / kind
                m = research.synthesize(
                    source=source, output_dir=synth_dir, film="syntheticfilm",
                    scene="moving_edges", kind=kind, preset="gentle",
                    seed=17, ffmpeg=ffmpeg, ffprobe=ffprobe,
                )
                self.assertEqual(m["frames"], 12)
                self.assertEqual(m["source_sha256"], source_sha)
                self.assertTrue((synth_dir / "reference.mkv").is_file())
                self.assertTrue((synth_dir / "degraded.mkv").is_file())
                self.assertEqual(research.probe(synth_dir / "reference.mkv", ffprobe)["fps"], "12/1")
                self.assertEqual(research.probe(synth_dir / "degraded.mkv", ffprobe)["frames"], 12)
                # Audio tracks must not leak into synthetic evaluation media.
                raw = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=index", "-of", "csv=p=0",
                     str(synth_dir / "reference.mkv")],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
                self.assertEqual(raw, "")
                if kind == "vertical_scratches":
                    self.assertEqual(len(list((synth_dir / "masks").glob("*.png"))), 12)

                # Simulate perfect recovery only as a ground-truth calibration,
                # with a signed candidate pointing to EXACT damaged input.
                candidate = root / f"{kind}_perfect.mkv"
                shutil.copyfile(synth_dir / "reference.mkv", candidate)
                candidate.with_suffix(".mkv.sig.json").write_text(
                    json.dumps({"source_fingerprint": {"sha256": m["degraded_sha256"]}}))
                report = research.evaluate(
                    manifest=synth_dir / "manifest.json",
                    candidates={"perfect_recovery": candidate},
                    output_dir=root / (kind + "_scored"),
                    ffmpeg=ffmpeg, ffprobe=ffprobe,
                )
                baseline = report["scores"]["degraded_baseline"]
                perfect = report["scores"]["perfect_recovery"]
                self.assertGreater(perfect["psnr_rgb_db"], baseline["psnr_rgb_db"])
                self.assertGreater(perfect["ssim_rgb"], baseline["ssim_rgb"])
                self.assertGreater(perfect["delta_psnr_db_vs_degraded"], 0.)
                self.assertTrue(perfect["perfect_rgb_match"])
                self.assertEqual(perfect["frames"], 12)
                self.assertIn("evaluation_type", report)
                if kind == "vertical_scratches":
                    self.assertGreater(baseline["known_scratch_mask_mse"], 0.)
                    self.assertEqual(perfect["known_scratch_mask_mse"], 0.)
                # Same corrupted input is not an output with legitimate
                # provenance unless explicitly signed.
                wrong = root / f"{kind}_wrong.mkv"
                shutil.copyfile(candidate, wrong)
                with self.assertRaisesRegex(ValueError, "No matching source signature"):
                    research.evaluate(
                        manifest=synth_dir / "manifest.json",
                        candidates={"wrong": wrong},
                        output_dir=root / (kind + "_wrong_score"),
                        ffmpeg=ffmpeg, ffprobe=ffprobe,
                    )
                wrong.with_suffix(".mkv.sig.json").write_text(
                    json.dumps({"source_fingerprint": {"sha256": "0" * 64}}))
                with self.assertRaisesRegex(ValueError, "source SHA mismatch"):
                    research.evaluate(
                        manifest=synth_dir / "manifest.json",
                        candidates={"wrong": wrong},
                        output_dir=root / (kind + "_wrong_hash"),
                        ffmpeg=ffmpeg, ffprobe=ffprobe,
                    )
                # Verify synthetic input is immutable and stage is no-overwrite.
                self.assertEqual(research.digest(source), source_sha)
                with self.assertRaisesRegex(ValueError, "must not exist"):
                    research.synthesize(
                        source=source, output_dir=synth_dir, film="syntheticfilm",
                        scene="moving_edges", kind=kind, preset="gentle",
                        ffmpeg=ffmpeg, ffprobe=ffprobe,
                    )

    def test_hierarchical_resampling_not_per_frame_and_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            inputs = []
            # 3 films with 2,1,3 scenes. The first film has two correlated
            # shots: equal film-weighting means not a per-frame pooled mean.
            increments = {
                "film_a": [3., 5.], "film_b": [-1.],
                "film_c": [2., 4., 6.],
            }
            for film, values in increments.items():
                for i, delta in enumerate(values):
                    f = root / f"{film}-{i}.json"
                    f.write_text(json.dumps({
                        "evaluation_type": "synthetic_full_reference",
                        "film": film, "scene": f"scene_{i}",
                        "degradation": "gaussian_blur",
                        "scores": {
                            "degraded_baseline": {"psnr_rgb_db": 20},
                            "CAS": {"delta_psnr_db_vs_degraded": delta},
                        },
                    }))
                    inputs.append(f)
            a = research.summarize(inputs, n_bootstrap=350, seed=44)
            b = research.summarize(inputs, n_bootstrap=350, seed=44)
            self.assertEqual(a, b)
            entry = a["results"][0]
            self.assertEqual(entry["films"], 3)
            self.assertEqual(entry["scenes"], 6)
            # film means: 4, -1, 4 => equal-film mean = 2.3333
            self.assertAlmostEqual(entry["mean_scene_paired_psnr_delta_db"], 7 / 3)
            self.assertIsNotNone(entry["ci95"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                research.summarize(inputs + [inputs[0]], n_bootstrap=100)

    def test_two_films_report_exploratory_only_without_fake_ci(self):
        with tempfile.TemporaryDirectory() as td:
            files = []
            for i in range(2):
                f = Path(td) / f"{i}.json"
                f.write_text(json.dumps({
                    "evaluation_type": "synthetic_full_reference",
                    "film": f"film_{i}", "scene": "one", "degradation": "read_noise",
                    "scores": {
                        "degraded_baseline": {},
                        "unsharp": {"delta_psnr_db_vs_degraded": 0.2 + i},
                    },
                }))
                files.append(f)
            result = research.summarize(files, n_bootstrap=100)["results"][0]
            self.assertTrue(result["exploratory_only"])
            self.assertIsNone(result["ci95"])


if __name__ == "__main__":
    unittest.main()
