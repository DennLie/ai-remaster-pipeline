"""Scratch-mask diagnostic validity: real detector, FFV1, GT, overlays and integrity guards."""
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
import diagnose_scratch_masks as audit  # noqa: E402
import restoration_evaluation as ev  # noqa: E402


class ScratchMaskDiagnosticTests(unittest.TestCase):
    def test_exact_mask_confusion_and_separate_precision_recall(self):
        gt = np.zeros((12, 18), np.uint8)
        predicted = gt.copy()
        gt[:4, 3:5] = 255
        predicted[:4, 4:6] = 255
        values = audit.confusion(predicted, gt)
        self.assertEqual(values["tp"], 4)
        self.assertEqual(values["fp"], 4)
        self.assertEqual(values["fn"], 4)
        scored = audit.summarize_confusion([values, values])
        self.assertEqual(scored["tp"], 8)
        self.assertAlmostEqual(scored["precision"], 0.5)
        self.assertAlmostEqual(scored["recall"], 0.5)
        self.assertAlmostEqual(scored["pixel_f1"], 0.5)
        self.assertAlmostEqual(scored["pixel_iou"], 1 / 3)
        with self.assertRaises(ValueError):
            audit.confusion(predicted, np.zeros((11, 18), np.uint8))
        with self.assertRaises(ValueError):
            audit.summarize_confusion([])

    def test_sensitivity_rejects_invalid_or_duplicate_values(self):
        self.assertEqual(audit.sensitivities_arg("0.25, 0.65"), [0.25, 0.65])
        for invalid in ("0.65,0.65", "nan", "-0.1", "1.1", "x", "", "0.2,", "0.2," * 13):
            with self.assertRaises(ValueError):
                audit.sensitivities_arg(invalid)

    def test_lossless_real_video_masks_and_oracle_counterfactual(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-scratch-audit-") as td:
            root = Path(td)
            source = root / "native.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=96x72:rate=12:duration=1",
                 "-c:v", "ffv1", str(source)],
                capture_output=True, check=True, timeout=40,
            )
            manifest_root = root / "synthetic"
            ev.synthesize(source=source, output_dir=manifest_root, film="testfilm",
                          scene="still_shapes", kind="vertical_scratches",
                          preset="balanced", seed=2026,
                          ffmpeg=ffmpeg, ffprobe=ffprobe)
            report = audit.diagnose(
                manifest=manifest_root / "manifest.json",
                output_dir=root / "audit",
                sensitivities=[0.25, 0.65],
                ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            self.assertEqual(report["frame_count"], 12)
            self.assertEqual(len(report["ground_truth_png_sha256"]), 12)
            self.assertEqual([x["sensitivity"] for x in report["settings"]],
                             [0.25, 0.65])
            for item in report["settings"]:
                scores = item["mask_scores"]
                self.assertEqual(
                    sum(scores[k] for k in ("tp", "fp", "fn", "tn")), 96 * 72 * 12
                )
                self.assertGreater(scores["synthetic_ground_truth_fraction"], 0)
                self.assertGreaterEqual(scores["pixel_f1"], 0)
                self.assertLessEqual(scores["pixel_f1"], 1)
                self.assertGreaterEqual(item["clean_transfer_detector_coverage_fraction_mean"], 0)
                self.assertLessEqual(item["clean_transfer_detector_coverage_fraction_mean"], 1)
                self.assertEqual(len(item["visual_samples"]), 3)
                for path in item["visual_samples"]:
                    self.assertTrue((root / "audit" / path).is_file())
                self.assertTrue(item["oracle_is_non_deployable"])
                self.assertGreaterEqual(item["telea_using_oracle_gt_mask_known_scratch_mse"], 0.)
            with self.assertRaises(FileExistsError):
                audit.diagnose(
                    manifest=manifest_root / "manifest.json",
                    output_dir=root / "audit", sensitivities=[0.65],
                    ffmpeg=ffmpeg, ffprobe=ffprobe,
                )
            first = manifest_root / "masks" / "000000.png"
            first.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "mask missing/malformed"):
                audit.diagnose(
                    manifest=manifest_root / "manifest.json",
                    output_dir=root / "tampered", sensitivities=[0.65],
                    ffmpeg=ffmpeg, ffprobe=ffprobe,
                )


if __name__ == "__main__":
    unittest.main()
