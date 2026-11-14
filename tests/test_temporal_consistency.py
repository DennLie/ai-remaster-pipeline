"""Unit tests for Phase 2 temporal consistency metrics (`scripts/temporal_consistency_scorer.py`)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import temporal_consistency_scorer as tcs


class TestTemporalConsistencyScorer(unittest.TestCase):
    def test_zero_flicker_on_stable_predictions(self) -> None:
        gt = [np.zeros((32, 32), dtype=np.uint8) for _ in range(6)]
        pred = [np.zeros((32, 32), dtype=np.uint8) for _ in range(6)]
        self.assertEqual(tcs.compute_mask_flicker_rate(gt, pred), 0.0)

    def test_detects_single_frame_alternating_flicker(self) -> None:
        gt = [np.zeros((10, 10), dtype=np.uint8) for _ in range(5)]
        pred = [np.zeros((10, 10), dtype=np.uint8) for _ in range(5)]
        # Introduce 0 -> 255 -> 0 flicker at frame 2 on a 4x4 block (16 pixels)
        pred[2][:4, :4] = 255
        rate = tcs.compute_mask_flicker_rate(gt, pred)
        # 3 interior frames (t=1,2,3), 100 eligible pixels per frame = 300 triplets.
        # At t=2, 16 pixels alternate (0 -> 1 -> 0).
        self.assertAlmostEqual(rate, 16 / 300, places=6)

    def test_persistent_trajectory_continuity_and_dropout(self) -> None:
        gt = [np.zeros((20, 20), dtype=np.uint8) for _ in range(6)]
        pred = [np.zeros((20, 20), dtype=np.uint8) for _ in range(6)]
        # Persistent vertical scratch across frames 1..4 (4 frames)
        for t in (1, 2, 3, 4):
            gt[t][:, 10] = 255
        # Detector catches frames 1, 2, 4 but drops out on frame 3 (3/4 = 0.75 continuity)
        for t in (1, 2, 4):
            pred[t][:, 10] = 255

        res = tcs.compute_persistent_trajectory_metrics(gt, pred)
        self.assertEqual(res["persistent_runs"], 1)
        self.assertEqual(res["persistent_frames"], 4)
        self.assertEqual(res["detected_persistent_frames"], 3)
        self.assertAlmostEqual(res["detection_continuity_score"], 0.75, places=6)
        self.assertAlmostEqual(res["missed_persistent_defect_rate"], 0.25, places=6)

    def test_temporal_motion_false_positive_separation(self) -> None:
        frames = [np.full((20, 20), 50, dtype=np.uint8) for _ in range(4)]
        # Introduce strong motion in top half of frame 2
        frames[2][:10, :] = 120
        gt = [np.zeros((20, 20), dtype=np.uint8) for _ in range(4)]
        pred = [np.zeros((20, 20), dtype=np.uint8) for _ in range(4)]
        # False positives occur exclusively inside the moving region of frame 2
        pred[2][:5, :10] = 255  # 50 FP pixels inside motion region

        res = tcs.compute_temporal_motion_fpr(frames, gt, pred, motion_diff_threshold=12.0)
        self.assertGreater(res["temporal_false_positive_rate"], 0.0)
        self.assertEqual(res["static_false_positive_rate"], 0.0)
        self.assertEqual(res["motion_fp_pixels"], 50)

    def test_cli_end_to_end_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gt_dir = root / "gt"
            pred_dir = root / "pred"
            frames_dir = root / "frames"
            gt_dir.mkdir()
            pred_dir.mkdir()
            frames_dir.mkdir()
            for i in range(4):
                g = np.zeros((16, 16), dtype=np.uint8)
                p = np.zeros((16, 16), dtype=np.uint8)
                f = np.full((16, 16), 80 + i * 5, dtype=np.uint8)
                g[:, 8] = 255
                p[:, 8] = 255
                cv2.imwrite(str(gt_dir / f"{i:04d}.png"), g)
                cv2.imwrite(str(pred_dir / f"{i:04d}.png"), p)
                cv2.imwrite(str(frames_dir / f"{i:04d}.png"), f)

            report_path = root / "report.json"
            rc = tcs.main(
                [
                    "--gt-dir",
                    str(gt_dir),
                    "--pred-dir",
                    str(pred_dir),
                    "--frames-dir",
                    str(frames_dir),
                    "--report",
                    str(report_path),
                ]
            )
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["detection_continuity_score"], 1.0)
            self.assertEqual(report["missed_persistent_defect_rate"], 0.0)
            self.assertEqual(report["mask_flicker_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
