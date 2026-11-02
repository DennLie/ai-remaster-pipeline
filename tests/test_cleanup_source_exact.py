"""Original-only exact-frame inspection before a Deflicker variant exists."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import frame_inspector


class DummyApp:
    def __init__(self, available=True):
        self.available = available

    def cleanup_comparison_state(self):
        return {
            "source": "evaluation/short.mkv",
            "source_exists": "true" if self.available else "false",
            "output": "",
            "exists": "false",
        }


class OriginalOnlyInspectionTests(unittest.TestCase):
    def test_can_decode_exact_source_without_processed_result(self):
        meta = {"frames": 72, "fps": 30.0, "width": 960, "height": 720}
        with patch.object(frame_inspector, "_video_path", return_value=Path("short.mkv")), \
             patch.object(frame_inspector, "_metadata", return_value=meta), \
             patch.object(frame_inspector, "_decode", return_value=".cache/frame.png") as decode, \
             patch.object(frame_inspector, "_prune"):
            info = frame_inspector.inspection(DummyApp(), "cleanup_source", 31)
        self.assertEqual(info["frame"], 31)
        self.assertEqual(info["frames"], 72)
        self.assertEqual(info["before"], ".cache/frame.png")
        self.assertIsNone(info["after"])
        self.assertTrue(info["exact"])
        decode.assert_called_once_with(Path("short.mkv"), 31)

    def test_no_arbitrary_or_missing_source_allowed(self):
        with self.assertRaises(FileNotFoundError):
            frame_inspector.inspection(DummyApp(False), "cleanup_source", 0)
        with self.assertRaises(ValueError):
            frame_inspector.inspection(DummyApp(), "cleanup_source", -1)

    def test_frame_out_of_range_rejected(self):
        with patch.object(frame_inspector, "_video_path", return_value=Path("short.mkv")), \
             patch.object(frame_inspector, "_metadata", return_value={
                 "frames": 72, "fps": 30.0, "width": 960, "height": 720
             }):
            with self.assertRaises(IndexError):
                frame_inspector.inspection(DummyApp(), "cleanup_source", 72)


if __name__ == "__main__":
    unittest.main()
