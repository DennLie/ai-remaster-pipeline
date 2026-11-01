"""CPU contract tests for paired film deflicker diagnostic exports."""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_light_deflicker as comparison  # noqa: E402


def write_movie(path: Path, frames: int, offset: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (96, 72))
    if not writer.isOpened():
        raise RuntimeError("Unable to create test AVI")
    for i in range(frames):
        image = np.full((72, 96, 3), 105 + 4 * (i % 2) + offset, np.uint8)
        image[16:40, 22:35] = 145 + offset
        writer.write(image)
    writer.release()


class CompareLightDeflickerTests(unittest.TestCase):
    def test_aligned_video_pair_writes_image_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, adjusted = root / "before.avi", root / "after.avi"
            write_movie(source, frames=8, offset=0)
            write_movie(adjusted, frames=8, offset=5)
            result = comparison.compare_videos(source, adjusted, root / "out")
            self.assertEqual(result["frames"], 8)
            self.assertEqual(result["width"], 96)
            self.assertGreater(result["changed_median_luma_frames"], 0)
            self.assertTrue((root / "out" / "light_deflicker_before_after.jpg").exists())
            self.assertEqual(json.loads(
                (root / "out" / "light_deflicker_compare.json").read_text()
            )["frames"], 8)

    def test_missing_frame_refuses_misaligned_comparison(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, adjusted = root / "before.avi", root / "after.avi"
            write_movie(source, frames=8, offset=0)
            write_movie(adjusted, frames=7, offset=3)
            with self.assertRaisesRegex(ValueError, "Frame count mismatch"):
                comparison.compare_videos(source, adjusted, root / "out")


if __name__ == "__main__":
    unittest.main()
