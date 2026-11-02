"""Real CPU ARP stage 1,2,4,5 scientific evaluation using known synthetic references."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_steps_1to5 as stage
import restoration_evaluation as eval_tool


class EvaluatingSteps1To5(unittest.TestCase):
    def test_all_four_stage_adapters_real_inputs_and_provenance(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-all-stages-eval-") as td:
            folder = Path(td)
            source = folder / "source.mkv"
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=12:duration=2",
                "-c:v", "ffv1", str(source)
            ], capture_output=True, check=True, timeout=40)
            original_sha = eval_tool.digest(source)
            definitions = [
                ("deflicker", "temporal_flicker"),
                ("denoise", "read_noise"),
                ("stabilize", "camera_jitter"),
                ("scratch_masks", "vertical_scratches"),
            ]
            for key, corruption in definitions:
                with self.subTest(stage=key):
                    input_dir = folder / ("generated_" + key)
                    meta = eval_tool.synthesize(
                        source=source, output_dir=input_dir, film="syntheticfilm",
                        scene="one", kind=corruption, preset="strong", seed=42,
                        ffmpeg=ffmpeg, ffprobe=ffprobe,
                    )
                    self.assertEqual(meta["frames"], 24)
                    actual = stage.run(
                        manifest=input_dir / "manifest.json",
                        output_dir=folder / ("output_" + key),
                        stage=key, ffmpeg=ffmpeg, ffprobe=ffprobe,
                        denoise_methods=("off", "hqdn3d", "nlmeans", "bm3d"),
                        denoise_presets=("gentle",),
                    )
                    self.assertEqual(actual["stage"], key)
                    self.assertEqual(actual["synthetic_degradation"], corruption)
                    self.assertIn("degraded_baseline", actual["scores"])
                    self.assertEqual(actual["scores"]["degraded_baseline"]["frames"], 24)
                    self.assertEqual(original_sha, eval_tool.digest(source))
                    self.assertTrue((folder / ("output_" + key) / "scores" / "evaluation.json").is_file())
                    if key == "denoise":
                        self.assertEqual(len(actual["scores"]), 5)
                        for method in ("off_gentle", "hqdn3d_gentle",
                                       "nlmeans_gentle", "bm3d_gentle"):
                            self.assertIn("ssim_rgb", actual["scores"][method])
                            self.assertTrue((folder / "output_denoise" / (method + ".mkv.sig.json")).is_file())
                    if key == "deflicker":
                        self.assertTrue((folder / "output_deflicker" / "light_deflicker.mkv").is_file())
                        self.assertLessEqual(actual["methods"]["light_deflicker"]["max_gain_deviation"], 0.080001)
                    if key == "stabilize":
                        drift = actual["methods"]["stabilization_drift"]
                        self.assertIn("drift_rmse_px", drift["damaged_baseline"])
                        self.assertEqual(drift["libvidstab_single_shot"]["total_frames"], 24)
                        self.assertTrue((folder / "output_stabilize" / "libvidstab_single_shot.mkv.sig.json").is_file())
                    if key == "scratch_masks":
                        det = actual["methods"]["scratch_mask_detection"]
                        for name in ("precision", "recall", "f1", "mask_iou"):
                            self.assertTrue(det[name] is None or 0 <= det[name] <= 1)
                        self.assertEqual(len(list((folder / "output_scratch_masks" / "predicted_masks").glob("*.png"))), 24)
                        self.assertIn("known_scratch_mask_mse", actual["scores"]["telea_detector_baseline"])
                        self.assertIn("NOT ProPainter", det["repair_baseline"])
                    with self.assertRaisesRegex(FileExistsError, "overwrite"):
                        stage.run(manifest=input_dir / "manifest.json",
                                  output_dir=folder / ("output_" + key),
                                  stage=key, ffmpeg=ffmpeg, ffprobe=ffprobe)

    def test_mismatched_ground_truth_task_and_invalid_options(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            missing = base / "missing.json"
            missing.write_text(json.dumps({
                "schema_version": 1,
                "reference_type": "decoded_existing_transfer_not_pristine_camera_negative",
                "synthetic_degradation": "gaussian_blur",
            }))
            with self.assertRaisesRegex(ValueError, "exact synthetic"):
                stage.assert_experiment(missing, "scratch_masks", "ffprobe")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                stage.assert_experiment(missing, "magic", "ffprobe")
            with self.assertRaisesRegex(ValueError, "same"):
                stage.residual_alignment([], [None])


if __name__ == "__main__":
    unittest.main()
