"""Tiny synthetic-video contract for the source-resolution comparison tool."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_source_resolution as comparison  # noqa: E402


class SourceResolutionTests(unittest.TestCase):
    def test_resolution_transform_keeps_aspect_ratio(self):
        options = comparison.variants(960, 720)
        self.assertEqual(options["downsampled_480p"], (640, 480))
        self.assertEqual(options["digital_source"], (960, 720))
        self.assertEqual(options["lanczos_2x"], (1920, 1440))

    def test_soft_high_res_predictions_do_not_expand_on_reduction(self):
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        result = comparison.same_coordinates(mask, (4, 4))
        self.assertEqual(int(np.count_nonzero(result)), 1)

    def test_short_video_produces_report_and_poster(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "source.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 6, (96, 72))
            if not writer.isOpened():
                self.skipTest("No MJPEG AVI writer in this OpenCV build")
            base = np.full((72, 96, 3), 116, dtype=np.uint8)
            for i in range(9):
                frame = base.copy()
                if i == 4:
                    frame[36:39, 20:23] = 255
                writer.write(frame)
            writer.release()
            result = comparison.analyze_clip(video, root / "out", max_frames=9)
            self.assertEqual(result["processed_frames"], 9)
            self.assertFalse(result["ground_truth_available"])
            self.assertIn("digital_source", result["variants"])
            self.assertEqual(result["variants"]["digital_source"]["width"], 96)
            self.assertTrue((root / "out" / "source_resolution_contact_sheet.jpg").exists())
            stored = json.loads((root / "out" / "source_resolution_report.json").read_text())
            self.assertEqual(stored["processed_frames"], 9)


if __name__ == "__main__":
    unittest.main()
