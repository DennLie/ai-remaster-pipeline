"""GPU-free benchmarks with hand-constructed masks and ignored pixel regions."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_defect_masks as bench  # noqa: E402


class DefectBenchmarkTests(unittest.TestCase):
    def test_perfect_prediction_metrics(self):
        gt = np.zeros((4, 6), dtype=np.uint8)
        gt[1:3, 1:3] = 255
        result = bench.metrics(bench.confusion(gt, gt))
        self.assertEqual(result["tp"], 4)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 0)
        self.assertEqual(result["iou"], 1.0)
        self.assertEqual(result["precision"], 1.0)

    def test_mask_threshold_and_ignore_regions(self):
        gt = np.array([[0, 255, 127], [255, 0, 127]], dtype=np.uint8)
        pred = np.array([[200, 130, 255], [120, 0, 255]], dtype=np.uint8)
        counts = bench.confusion(gt, pred, 128)
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "ignored": 2})
        result = bench.metrics(counts)
        self.assertEqual(result["iou"], round(1/3, 6))
        self.assertEqual(result["false_positive_rate"], 0.5)

    def test_shapes_must_match_exactly(self):
        gt = np.zeros((8, 10), np.uint8)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            bench.confusion(gt, np.zeros((7, 10), np.uint8))

    def test_blank_frames_do_not_inflate_false_positive_rate(self):
        gt = np.zeros((5, 5), dtype=np.uint8)
        empty = bench.metrics(bench.confusion(gt, gt))
        self.assertEqual(empty["iou"], 1.0)
        self.assertEqual(empty["false_positive_rate"], 0.0)
        noisy = gt.copy()
        noisy[0, 0] = 255
        scored = bench.metrics(bench.confusion(gt, noisy))
        self.assertEqual(scored["iou"], 0.0)
        self.assertEqual(scored["false_positive_rate"], 0.04)

    def test_bad_manifest_and_paths_are_rejected(self):
        for candidate in ("../hello.png", "x/y.png", "", "."):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                bench.validate_component(candidate, "frame")
        with self.assertRaises(ValueError):
            bench.parse_methods(["wrong"])
        with self.assertRaises(ValueError):
            bench.parse_methods(["dart=x", "dart=y"])

    def test_real_manifest_and_two_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_root = root / "ground_truth"
            a_root = root / "classical"
            b_root = root / "dart"
            for directory in (gt_root, a_root, b_root):
                (directory / "clip01").mkdir(parents=True)
            names = ["000001.png", "000002.png"]
            for idx, name in enumerate(names):
                gt = np.zeros((10, 10), np.uint8)
                gt[2:5, 2:5] = 255 if idx == 0 else 0
                classical = np.zeros_like(gt)
                dart = gt.copy()
                cv2.imwrite(str(gt_root / "clip01" / name), gt)
                cv2.imwrite(str(a_root / "clip01" / name), classical)
                cv2.imwrite(str(b_root / "clip01" / name), dart)
            csvpath = root / "samples.csv"
            with csvpath.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=bench.REQUIRED_COLUMNS)
                writer.writeheader()
                writer.writerow({"clip": "clip01", "frame": names[0], "split": "evaluation", "damage_type": "dust"})
                writer.writerow({"clip": "clip01", "frame": names[1], "split": "evaluation", "damage_type": "clean"})
            data = bench.evaluate(
                bench.load_samples(csvpath), gt_root, {"classical": a_root, "dart": b_root}
            )
            self.assertEqual(data["samples"], 2)
            self.assertEqual(data["methods"]["dart"]["summary"]["iou"], 1.0)
            self.assertEqual(data["methods"]["classical"]["summary"]["recall"], 0.0)
            self.assertEqual(data["methods"]["dart"]["by"]["damage_type"]["dust"]["samples"], 1)
            # No silent interpolation or biased omission of missing frames.
            (a_root / "clip01" / names[1]).unlink()
            with self.assertRaises(FileNotFoundError):
                bench.evaluate(bench.load_samples(csvpath), gt_root, {"classical": a_root})

    def test_nonbinary_ground_truth_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.png"
            cv2.imwrite(str(path), np.full((6, 6), 100, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "only 0"):
                bench.read_mask(path, ground_truth=True)

    def test_end_to_end_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("gt", "dart"):
                (root / name / "a").mkdir(parents=True)
            image = np.zeros((8, 8), dtype=np.uint8)
            image[2:4, 2:4] = 255
            for name in ("gt", "dart"):
                cv2.imwrite(str(root / name / "a" / "001.png"), image)
            manifest = root / "manifest.csv"
            manifest.write_text("clip,frame,split,damage_type\na,001.png,test,dust\n")
            report = root / "report.json"
            code = bench.main([
                "--manifest", str(manifest), "--ground-truth", str(root / "gt"),
                "--prediction", "dart=" + str(root / "dart"),
                "--report", str(report),
            ])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(report.read_text())["methods"]["dart"]["summary"]["iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
