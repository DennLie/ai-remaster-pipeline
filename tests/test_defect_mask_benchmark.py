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
            payload = json.loads(report.read_text())
            self.assertEqual(payload["methods"]["dart"]["summary"]["iou"], 1.0)
            self.assertEqual(payload["methods"]["dart"]["dust_metrics"]["iou"], 1.0)
            self.assertEqual(payload["methods"]["dart"]["scratch_metrics"]["samples"], 0)

    def test_scratch_config_validation_bounds(self):
        with self.assertRaises(ValueError):
            bench.ScratchConfig(width=0)
        with self.assertRaises(ValueError):
            bench.ScratchConfig(width=9)
        with self.assertRaises(ValueError):
            bench.ScratchConfig(opacity=0.1)
        with self.assertRaises(ValueError):
            bench.ScratchConfig(angle_deg=20.0)
        with self.assertRaises(ValueError):
            bench.ScratchConfig(duration=0)
        with self.assertRaises(ValueError):
            bench.ScratchConfig(duration=31)
        with self.assertRaises(ValueError):
            bench.ScratchConfig(drift_px=-0.5)
        with self.assertRaises(ValueError):
            bench.ScratchConfig(drift_px=5.5)

    def test_dust_config_validation_bounds(self):
        with self.assertRaises(ValueError):
            bench.DustConfig(clusters_per_frame=0)
        with self.assertRaises(ValueError):
            bench.DustConfig(contrast=0.1)
        with self.assertRaises(ValueError):
            bench.DustConfig(affected_frame_ratio=0.0)

    def test_synthetic_scratch_generator_duration_drift_and_binary_gt(self):
        base = [np.full((48, 64), 100, dtype=np.uint8) for _ in range(8)]
        cfg = bench.ScratchConfig(
            width=3,
            opacity=0.75,
            angle_deg=10.0,
            duration=4,
            drift_px=2.0,
            bright=True,
            seed=123,
        )
        perturbed, masks = bench.inject_scratches(base, cfg)
        self.assertEqual(len(perturbed), 8)
        self.assertEqual(len(masks), 8)

        active_frames = [idx for idx, m in enumerate(masks) if np.any(m > 0)]
        self.assertEqual(len(active_frames), 4)
        for frame, mask in zip(perturbed, masks):
            self.assertTrue(set(np.unique(mask).tolist()).issubset({0, 255}))
            # Unmasked pixels must remain untouched
            self.assertTrue(np.array_equal(frame[mask == 0], np.full(np.count_nonzero(mask == 0), 100, dtype=np.uint8)))
            if np.any(mask == 255):
                self.assertTrue(np.all(frame[mask == 255] > 100))

        # Verify lateral drift shifts mean x-coordinate across consecutive active frames
        mean_xs = [float(np.mean(np.where(masks[t] == 255)[1])) for t in active_frames]
        self.assertGreater(mean_xs[-1] - mean_xs[0], 3.0)

        # Verify reproducibility with identical seed
        perturbed2, masks2 = bench.inject_scratches(base, cfg)
        for m1, m2 in zip(masks, masks2):
            self.assertTrue(np.array_equal(m1, m2))

    def test_synthetic_dust_generator_transient_clusters_and_binary_gt(self):
        base = [np.full((48, 48), 128, dtype=np.uint8) for _ in range(5)]
        cfg = bench.DustConfig(
            clusters_per_frame=3,
            particles_per_cluster=(2, 4),
            radius_range=(1, 3),
            contrast=0.8,
            seed=99,
        )
        perturbed, masks = bench.inject_dust(base, cfg)
        self.assertEqual(len(perturbed), 5)
        self.assertEqual(len(masks), 5)
        for idx, (frame, mask) in enumerate(zip(perturbed, masks)):
            self.assertTrue(set(np.unique(mask).tolist()).issubset({0, 255}))
            self.assertGreater(np.count_nonzero(mask == 255), 0)
            self.assertTrue(np.array_equal(frame[mask == 0], base[idx][mask == 0]))
            self.assertTrue(np.any(frame[mask == 255] != 128))
        # Transient dust masks must differ between consecutive frames
        self.assertFalse(np.array_equal(masks[0], masks[1]))

    def test_disjoint_scratch_vs_dust_metrics_and_cli_synthetic(self):
        base = [np.full((32, 32), 120, dtype=np.uint8) for _ in range(4)]
        _, scratch_gt = bench.inject_scratches(base, bench.ScratchConfig(duration=4, seed=7))
        _, dust_gt = bench.inject_dust(base, bench.DustConfig(clusters_per_frame=2, seed=8))
        empty_pred = [np.zeros_like(m) for m in dust_gt]

        report = bench.evaluate_category_masks(
            scratch_gt=scratch_gt,
            scratch_pred=scratch_gt,
            dust_gt=dust_gt,
            dust_pred=empty_pred,
        )
        self.assertEqual(report["scratch_metrics"]["iou"], 1.0)
        self.assertEqual(report["scratch_metrics"]["f1"], 1.0)
        self.assertEqual(report["dust_metrics"]["recall"], 0.0)
        self.assertEqual(report["dust_metrics"]["iou"], 0.0)
        self.assertLess(report["summary"]["iou"], 1.0)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synth_dir = root / "synth"
            report_path = root / "synth_report.json"
            code = bench.main([
                "--synthetic-dir", str(synth_dir),
                "--seed", "21",
                "--num-frames", "4",
                "--frame-height", "32",
                "--frame-width", "32",
                "--scratch-width", "3",
                "--scratch-opacity", "0.7",
                "--scratch-angle", "-5.0",
                "--scratch-duration", "3",
                "--scratch-drift", "1.5",
                "--dust-clusters", "2",
                "--dust-contrast", "0.9",
                "--report", str(report_path),
            ])
            self.assertEqual(code, 0)
            data = json.loads(report_path.read_text(encoding="utf-8"))
            oracle = data["methods"]["oracle"]
            self.assertEqual(oracle["summary"]["iou"], 1.0)
            self.assertEqual(oracle["scratch_metrics"]["samples"], 4)
            self.assertEqual(oracle["scratch_metrics"]["iou"], 1.0)
            self.assertEqual(oracle["dust_metrics"]["samples"], 4)
            self.assertEqual(oracle["dust_metrics"]["iou"], 1.0)

    def test_synthetic_generators_rgb_curvature_and_partial_overlap_metrics(self):
        # Verify 4D (T, H, W, C) RGB array input, curvature, dark scratches, and affected_frame_ratio
        rgb_frames = np.full((6, 40, 48, 3), 180, dtype=np.uint8)
        s_cfg = bench.ScratchConfig(
            width=4,
            opacity=0.6,
            angle_deg=-12.0,
            curvature=6.0,
            duration=3,
            drift_px=1.5,
            bright=False,
            seed=55,
        )
        perturbed_s, masks_s = bench.inject_scratches(rgb_frames, s_cfg, num_scratches=2)
        self.assertEqual(len(perturbed_s), 6)
        self.assertEqual(perturbed_s[0].shape, (40, 48, 3))
        self.assertEqual(masks_s[0].shape, (40, 48))
        has_dark = any(np.any(f[m == 255] < 180) for f, m in zip(perturbed_s, masks_s) if np.any(m == 255))
        self.assertTrue(has_dark)

        d_cfg = bench.DustConfig(
            clusters_per_frame=2,
            contrast=0.5,
            affected_frame_ratio=0.5,
            seed=77,
        )
        perturbed_d1, masks_d1 = bench.inject_dust(rgb_frames, d_cfg)
        perturbed_d2, masks_d2 = bench.inject_dust(rgb_frames, d_cfg)
        active_dust_frames = sum(1 for m in masks_d1 if np.any(m == 255))
        self.assertEqual(active_dust_frames, 3)
        for m1, m2 in zip(masks_d1, masks_d2):
            self.assertTrue(np.array_equal(m1, m2))

        # Verify partial-overlap P, R, F1, IoU, and unaffected-pixel FPR separated by category
        s_gt = [np.zeros((10, 10), dtype=np.uint8)]
        s_gt[0][0:4, 2] = 255  # 4 defect pixels, 96 unaffected pixels
        s_pred = [np.zeros((10, 10), dtype=np.uint8)]
        s_pred[0][0:2, 2] = 255  # 2 TP, 2 FN
        s_pred[0][0:3, 5] = 255  # 3 FP on unaffected pixels -> FPR = 3 / 96 = 0.03125

        d_gt = [np.zeros((10, 10), dtype=np.uint8)]
        d_gt[0][5:7, 5:7] = 255  # 4 defect pixels, 96 unaffected pixels
        d_pred = [np.zeros((10, 10), dtype=np.uint8)]
        d_pred[0][5:7, 5:7] = 255  # 4 TP, 0 FN
        d_pred[0][0, 0] = 255  # 1 FP -> FPR = 1 / 96

        cat_report = bench.evaluate_category_masks(s_gt, s_pred, d_gt, d_pred)
        sm = cat_report["scratch_metrics"]
        dm = cat_report["dust_metrics"]

        self.assertEqual(sm["tp"], 2)
        self.assertEqual(sm["fp"], 3)
        self.assertEqual(sm["fn"], 2)
        self.assertEqual(sm["tn"], 93)
        self.assertEqual(sm["precision"], round(2 / 5, 6))
        self.assertEqual(sm["recall"], round(2 / 4, 6))
        self.assertEqual(sm["f1"], round(4 / 9, 6))
        self.assertEqual(sm["iou"], round(2 / 7, 6))
        self.assertEqual(sm["false_positive_rate"], round(3 / 96, 8))

        self.assertEqual(dm["tp"], 4)
        self.assertEqual(dm["fp"], 1)
        self.assertEqual(dm["fn"], 0)
        self.assertEqual(dm["tn"], 95)
        self.assertEqual(dm["precision"], round(4 / 5, 6))
        self.assertEqual(dm["recall"], 1.0)
        self.assertEqual(dm["f1"], round(8 / 9, 6))
        self.assertEqual(dm["iou"], round(4 / 5, 6))
        self.assertEqual(dm["false_positive_rate"], round(1 / 96, 8))


if __name__ == "__main__":
    unittest.main()
