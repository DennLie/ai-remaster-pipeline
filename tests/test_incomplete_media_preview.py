"""Regression: never expose a still-writing FFmpeg .partial.mkv to browser previews.

Prevents transient "EBML header parsing failed" errors during HQDN3D output
generation. The completed video and source remain untouched.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import app, media, server
from ai_remaster_gui.models import STAGES
from ai_remaster_gui.paths import is_unfinished_media_artifact


class IncompletePreviewTests(unittest.TestCase):
    def test_identify_ffmpeg_partial_extensions_case_insensitively(self):
        for name in (
            "movie.partial.mkv", "movie.partial.mp4", "render.tmp.mkv",
            "codec.PARTIAL.MKV", "movie.part.mkv", "frame.partial.png",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_unfinished_media_artifact(Path(name)))
        for name in ("movie.mkv", "movie.hqdn3d.mkv", "film_part2.mkv", "partiality.mp4"):
            with self.subTest(name=name):
                self.assertFalse(is_unfinished_media_artifact(Path(name)))

    def test_denoise_file_inventory_ignores_in_progress_mkv_and_skips_preview(self):
        with tempfile.TemporaryDirectory(prefix="arp_partial_preview_") as dirname:
            root = Path(dirname)
            folder = root / "intermediate" / "denoised"
            folder.mkdir(parents=True)
            finished = folder / "Neighbors_denoised_hqdn3d_gentle.mkv"
            partial = folder / "Neighbors_denoised_hqdn3d_gentle.partial.mkv"
            finished.write_bytes(b"completed-video-fixture")
            partial.write_bytes(b"unfinished-ebml-header")
            (folder / "other.tmp.mkv").write_bytes(b"unfinished")
            stage = next(s for s in STAGES if s.key == "denoise")
            with (
                patch.object(server, "ROOT", root),
                patch.object(server, "file_preview", return_value="cached_thumb.jpg") as preview,
                patch.object(server, "rel", side_effect=lambda path: path.name),
            ):
                listed = app.APP.files_for(stage)
            self.assertEqual([item["path"] for item in listed], [finished.name])
            self.assertEqual(preview.call_count, 1)
            self.assertEqual(preview.call_args.args[0], finished)

    def test_ffmpeg_preview_function_fails_closed_before_spawning_ffmpeg(self):
        with tempfile.TemporaryDirectory(prefix="arp_partial_media_") as dirname:
            partial = Path(dirname) / "clip.partial.mkv"
            partial.write_bytes(b"\x1a\x45")  # incomplete Matroska EBML signature
            with patch.object(media, "local_tool", side_effect=AssertionError(
                "Must not start FFmpeg on unfinished media"
            )):
                with self.assertRaisesRegex(FileNotFoundError, "Unfinished video"):
                    media.media_clip_path(partial, 0.0, 4.0)
            self.assertEqual(media.file_preview(partial), "")

    def test_http_media_handler_has_same_temporary_file_guard(self):
        handler = (Path(__file__).parents[1] / "ai_remaster_gui" / "http_handler.py"
                   ).read_text(encoding="utf-8")
        self.assertIn("if path is None or is_unfinished_media_artifact(path):", handler)


if __name__ == "__main__":
    unittest.main()
