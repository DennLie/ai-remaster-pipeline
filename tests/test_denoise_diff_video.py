"""Rendered Remove Noise live change previews: caching, validation and pixels."""
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
    def test_memory_bounded_amplification_keeps_the_original_float64_mapping(self) -> None:
        values = np.random.default_rng(42).uniform(-255, 255, (256, 128))
        for amplification in (8, 32, 64):
            expected = np.floor(255.0 * np.sqrt(np.clip(np.abs(values).astype(np.float64) * amplification / 255.0, 0.0, 1.0)) + 0.5).astype(np.uint8)
            np.testing.assert_array_equal(diff._boost(values, amplification), expected)

    def test_direction_mapping_matches_the_frame_inspector(self) -> None:
        before = np.zeros((1, 1, 3), dtype=np.uint8)
        after = np.array([[[3, 2, 1]]], dtype=np.uint8)  # BGR: RGB 1,2,3
        out = diff._diff_frame(before, after, "direction", 32)
        # Signed Rec.709 luma delta 1.86 -> 123; positive is teal in BGR.
        self.assertEqual(out.tolist(), [[[123, 123, 6]]])

    def test_direction_mapping_marks_darkened_pixels_orange(self) -> None:
        before = np.array([[[30, 20, 10]]], dtype=np.uint8)  # RGB 10,20,30
        after = np.array([[[27, 18, 9]]], dtype=np.uint8)    # RGB 9,18,27
        out = diff._diff_frame(before, after, "direction", 32)
        self.assertEqual(out.tolist(), [[[7, 64, 123]]])

    def test_rgb_mapping_matches_the_frame_inspector(self) -> None:
        before = np.zeros((1, 1, 3), dtype=np.uint8)
        after = np.array([[[3, 2, 1]]], dtype=np.uint8)  # BGR: RGB 1,2,3
        out = diff._diff_frame(before, after, "rgb", 32)
        self.assertEqual(out.tolist(), [[[156, 128, 90]]])

    def test_unchanged_pixels_stay_neutral_graphite(self) -> None:
        before = np.array([[[12, 34, 56]]], dtype=np.uint8)
        out = diff._diff_frame(before, before.copy(), "direction", 32)
        self.assertEqual(out.tolist(), [[[31, 23, 15]]])

    def test_invalid_display_settings_are_rejected(self) -> None:
        before = np.zeros((1, 1, 3), dtype=np.uint8)
        after = np.ones((1, 1, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            diff._diff_frame(before, after, "rainbow", 32)
        with self.assertRaises(ValueError):
            diff._diff_frame(before, after, "direction", 16)

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
        with self.assertRaises(ValueError):
            diff.render_diff_preview(self.first, self.second, 0, 1, "rainbow", 32)
        with self.assertRaises(ValueError):
            diff.render_diff_preview(self.first, self.second, 0, 1, "direction", 16)

    @unittest.skipUnless(diff.local_tool("ffmpeg"), "FFmpeg required")
    def test_renders_a_cached_playable_live_change_video(self) -> None:
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
                # The two generated patterns differ, so the change map is not empty.
                self.assertGreater(int(frame.max()), 0)
            finally:
                capture.release()
            again = diff.render_diff_preview(self.first, self.second, 0, 0.4)
            self.assertTrue(again["cached"])
            self.assertEqual(again["token"], result["token"])
            rgb = diff.render_diff_preview(
                self.first, self.second, 0, 0.4, "rgb", 64,
            )
            self.assertFalse(rgb["cached"])
            self.assertNotEqual(rgb["token"], result["token"])
            self.assertIsNotNone(diff.diff_video_path(rgb["token"]))


if __name__ == "__main__":
    unittest.main()
