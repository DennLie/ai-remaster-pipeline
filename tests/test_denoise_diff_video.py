"""Rendered Remove Noise difference previews: caching, validation and pixels."""
from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from ai_remaster_gui import denoise_diff_video as diff


def _write_clip(path: Path, pattern: str) -> None:
    subprocess.run(
        [
            diff.local_tool("ffmpeg") or "ffmpeg", "-y", "-f", "lavfi", "-i", pattern,
            "-t", "0.4", "-r", "24", "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class DiffFrameTests(unittest.TestCase):
    def test_magenta_mapping_matches_the_browser_canvas(self) -> None:
        before = np.zeros((1, 1, 3), dtype=np.uint8)
        after = np.array([[[10, 20, 30]]], dtype=np.uint8)
        out = diff._diff_frame(before, after)
        # (10+20+30)*4/3 = 80 -> R=80, G=0, B=48 in BGR order.
        self.assertEqual(out.tolist(), [[[48, 0, 80]]])

    def test_cached_token_rejects_paths_and_unknown_tokens(self) -> None:
        self.assertIsNone(diff.diff_video_path("../../etc/passwd"))
        self.assertIsNone(diff.diff_video_path("0" * 64))


class DiffPreviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="arp_diff_preview_")
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.first = self.folder / "first.mp4"
        self.second = self.folder / "second.mp4"

    def test_requires_two_different_existing_clips(self) -> None:
        with self.assertRaises(FileNotFoundError):
            diff.render_diff_preview(self.first, self.second, 0, 1)
        self.first.write_bytes(b"x")
        self.second.write_bytes(b"x")
        with self.assertRaises(ValueError):
            diff.render_diff_preview(self.first, self.first, 0, 1)
        with self.assertRaises(ValueError):
            diff.render_diff_preview(self.first, self.second, 2, 1)
        with self.assertRaises(ValueError):
            diff.render_diff_preview(self.first, self.second, 0, 13)

    @unittest.skipUnless(diff.local_tool("ffmpeg"), "FFmpeg required")
    def test_renders_a_cached_playable_difference_video(self) -> None:
        _write_clip(self.first, "testsrc2=size=320x240:rate=24")
        _write_clip(self.second, "smptebars=size=320x240:rate=24")
        cache = self.folder / "cache"
        with (
            mock.patch.object(diff, "_CACHE", cache),
            mock.patch.object(diff, "media_clip_path", side_effect=lambda source, start, end, key: source),
        ):
            result = diff.render_diff_preview(self.first, self.second, 0, 0.4)
            self.assertTrue(result["url"].startswith("/api/denoise-diff-video?token="))
            self.assertFalse(result["cached"])
            path = diff.diff_video_path(result["token"])
            self.assertIsNotNone(path)
            capture = cv2.VideoCapture(str(path))
            try:
                self.assertTrue(capture.isOpened())
                self.assertGreater(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape[:2], (240, 320))
                # The two generated patterns differ, so the magenta map is not empty.
                self.assertGreater(int(frame.max()), 0)
            finally:
                capture.release()
            again = diff.render_diff_preview(self.first, self.second, 0, 0.4)
            self.assertTrue(again["cached"])
            self.assertEqual(again["token"], result["token"])


if __name__ == "__main__":
    unittest.main()
