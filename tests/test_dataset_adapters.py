"""Unit and integration tests for published benchmark dataset adapters (`scripts/dataset_adapters.py`)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark_defect_masks as bdm
import dataset_adapters as da


class TestDatasetAdapters(unittest.TestCase):
    def test_fds_polygon_json_parsing_and_scan8_scale_factor(self) -> None:
        sample_json = {
            "0": {
                "points": [{"x": 10, "y": 20}, {"x": 20, "y": 20}, {"x": 20, "y": 30}, {"x": 10, "y": 30}],
                "label": {"name": "Dust"},
            },
            "1": {
                "points": [{"x": 40, "y": 5}, {"x": 40, "y": 50}],
                "label": {"name": "Scratch"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "Scan (1).json"
            p8 = Path(tmp) / "Scan (8).json"
            p1.write_text(json.dumps(sample_json), encoding="utf-8")
            p8.write_text(json.dumps(sample_json), encoding="utf-8")

            polys1 = da.parse_fds_polygon_json(p1)
            polys8 = da.parse_fds_polygon_json(p8)

            self.assertEqual(polys1[0]["scale_applied"], 1.0)
            self.assertEqual(polys1[0]["points"][0].tolist(), [10, 20])
            self.assertEqual(polys8[0]["scale_applied"], 1.5)
            self.assertEqual(polys8[0]["points"][0].tolist(), [15, 30])

    def test_fds_rasterization_separates_scratch_and_dust_with_ignore_for_other_classes(self) -> None:
        polygons = [
            {
                "id": "0",
                "label": "Dust",
                "channel": "dust",
                "points": np.array([[5, 5], [10, 5], [10, 10], [5, 10]], dtype=np.int32),
            },
            {
                "id": "1",
                "label": "Scratch",
                "channel": "scratch",
                "points": np.array([[20, 2], [20, 25]], dtype=np.int32),
            },
        ]
        masks = da.rasterize_fds_masks(polygons, height=32, width=32)
        # In dust mask: dust region is 255, scratch region is 127 (ignore), clean background is 0
        self.assertEqual(int(masks["dust"][7, 7]), 255)
        self.assertEqual(int(masks["dust"][10, 20]), 127)
        self.assertEqual(int(masks["dust"][0, 0]), 0)
        # In scratch mask: scratch region is 255, dust region is 127 (ignore)
        self.assertEqual(int(masks["scratch"][10, 20]), 255)
        self.assertEqual(int(masks["scratch"][7, 7]), 127)
        # Per-class dictionary contains all 5 FDS classes
        self.assertEqual(int(masks["by_class"]["Dust"][7, 7]), 255)
        self.assertEqual(int(masks["by_class"]["Scratch"][10, 20]), 255)

    def test_artefact_segmentation_conversion_maps_background_to_ignore(self) -> None:
        ann = np.zeros((16, 16), dtype=np.uint8)
        ann[0:2, :] = 255   # Background border -> 127 ignore
        ann[4:6, 4:6] = 3   # Dust -> 255 in dust & combined, 127 in scratch
        ann[8:12, 10] = 4   # Scratch -> 255 in scratch & combined, 127 in dust
        ann[13, 13] = 9     # Cracks (structural) -> 127 in combined/scratch/dust, 255 in all_damage

        comb = da.convert_artefact_segmentation_mask(ann, mode="combined")
        scratch = da.convert_artefact_segmentation_mask(ann, mode="scratch")
        dust = da.convert_artefact_segmentation_mask(ann, mode="dust")
        all_dmg = da.convert_artefact_segmentation_mask(ann, mode="all_damage")

        self.assertEqual(int(comb[0, 0]), 127)
        self.assertEqual(int(comb[4, 4]), 255)
        self.assertEqual(int(comb[9, 10]), 255)
        self.assertEqual(int(comb[13, 13]), 127)
        self.assertEqual(int(all_dmg[13, 13]), 255)

        self.assertEqual(int(scratch[9, 10]), 255)
        self.assertEqual(int(scratch[4, 4]), 127)
        self.assertEqual(int(dust[4, 4]), 255)
        self.assertEqual(int(dust[9, 10]), 127)

    def test_provenance_guard_prevents_mixing_manual_gt_with_pseudo_labels(self) -> None:
        manual_only = [
            {"clip": "c1", "frame": "001.png", "split": "test", "damage_type": "dust", "label_provenance": "manual_polygon_gt"},
            {"clip": "c2", "frame": "001.png", "split": "test", "damage_type": "scratch", "label_provenance": "manual_segmentation_gt"},
        ]
        counts = da.validate_provenance_separation(manual_only, require_manual_only=True)
        self.assertEqual(counts["manual_polygon_gt"], 1)
        self.assertEqual(counts["manual_segmentation_gt"], 1)

        with self.assertRaises(ValueError):
            da.validate_provenance_separation(
                manual_only + [{"clip": "c3", "frame": "001.png", "split": "test", "damage_type": "scratch", "label_provenance": "model_pseudo_label"}],
            )
        with self.assertRaises(ValueError):
            da.validate_provenance_separation(
                [{"clip": "c3", "frame": "001.png", "split": "test", "damage_type": "scratch", "label_provenance": "synthetic_parametric_gt"}],
                require_manual_only=True,
            )

    def test_export_and_evaluate_via_benchmark_defect_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "adapted"
            gt = np.zeros((24, 24), dtype=np.uint8)
            gt[5:10, 5:10] = 255
            gt[0, :] = 127
            records = [
                {
                    "clip": "fds_scan_10",
                    "frame": "000001.png",
                    "split": "test",
                    "damage_type": "dust",
                    "label_provenance": "manual_polygon_gt",
                    "gt_mask": gt,
                }
            ]
            exported = da.export_benchmark_dataset(out_dir, records, require_manual_only=True)
            self.assertTrue(exported["provenance_summary"]["manual_only_verified"])

            pred_dir = Path(tmp) / "preds" / "fds_scan_10"
            pred_dir.mkdir(parents=True)
            pred = np.zeros((24, 24), dtype=np.uint8)
            pred[5:10, 5:10] = 255
            cv2.imwrite(str(pred_dir / "000001.png"), pred)

            samples = bdm.load_samples(exported["manifest"])
            res = bdm.evaluate(samples, exported["ground_truth"], {"oracle": Path(tmp) / "preds"})
            self.assertEqual(res["methods"]["oracle"]["dust_metrics"]["f1"], 1.0)
            self.assertEqual(res["methods"]["oracle"]["dust_metrics"]["ignored"], 24)

    def test_nuisance_fp_breakdown_separates_edges_grain_faces_and_motion(self) -> None:
        rng = np.random.default_rng(123)
        frame = rng.integers(90, 130, size=(64, 64), dtype=np.uint8)
        # Add a sharp high-contrast vertical edge at x=48
        frame[:, 48:] = 230
        prev_frame = frame.copy()
        # Add motion in bottom-left corner
        prev_frame[45:60, 5:20] = 10

        gt = np.zeros((64, 64), dtype=np.uint8)
        pred = np.zeros((64, 64), dtype=np.uint8)
        # Put false positives specifically along the sharp edge at x=48 and motion box
        pred[10:40, 48] = 255
        pred[48:55, 8:15] = 255

        breakdown = da.evaluate_nuisance_fp_breakdown(frame, gt, pred, prev_frame=prev_frame)
        self.assertIn("edges", breakdown)
        self.assertIn("grain", breakdown)
        self.assertIn("faces_foreground", breakdown)
        self.assertIn("motion", breakdown)
        self.assertGreater(breakdown["edges"]["false_positive_rate"], 0.0)
        self.assertGreater(breakdown["motion"]["false_positive_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
