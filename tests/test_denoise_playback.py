"""Native-size, source-frame-indexed comparison reels (real FFV1/H.264)."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ai_remaster_gui import denoise_playback as playback, denoise_review, frame_inspector


@unittest.skipUnless(playback.local_tool("ffmpeg"), "FFmpeg required")
class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="arp_playback_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        self.outputs = self.root / "denoised"
        self.outputs.mkdir()
        self.result = self.outputs / "result.mkv"
        for path, delta in ((self.source, 0), (self.result, 2)):
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 24, (64, 48))
            self.assertTrue(writer.isOpened())
            for frame in range(720):
                image = np.full((48, 64, 3), min(255, frame % 240 + delta), dtype=np.uint8)
                image[24:] = frame // 240 + delta
                writer.write(image)
            writer.release()
        stat = self.source.stat()
        self.result.with_suffix(".mkv.sig.json").write_text(json.dumps({
            "tool": "denoise_video.py", "version": 1, "source": str(self.source),
            "source_fingerprint": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
            "method": "bm3d", "preset": "balanced", "intermediate_profile": "lossless",
        }), encoding="utf-8")
        self.app = SimpleNamespace(settings={"global": {"source": str(self.source)},
            "cloud": {"intermediate_format": "lossless"}})
        for change in (patch.object(playback, "_CACHE", self.root / "cache"),
                       patch.object(denoise_review, "_OUTPUTS", self.outputs)):
            change.start()
            self.addCleanup(change.stop)

    def test_late_segment_keeps_native_pixels_and_subtracts_before_encoding(self):
        seen = []
        real = playback._diff_frame
        def inspect(before, after, mode, contrast):
            if not seen:
                seen.append((before.copy(), after.copy()))
            return real(before, after, mode, contrast)
        with patch.object(playback, "_diff_frame", side_effect=inspect):
            result = playback.playback_segment(self.app, str(self.result), 500)
        self.assertEqual(result["frames"], 720)
        self.assertEqual((result["width"], result["height"]), (64, 48))
        self.assertEqual(result["first_frame"], 384)
        self.assertEqual(result["end_frame"], 576)
        self.assertEqual(result["regions"]["difference"], [128, 0, 64, 48])
        # Frame 384, not the beginning of the film and not a lossy preview decode.
        self.assertEqual(int(seen[0][0][0, 0, 0]), 144)
        self.assertEqual(int(seen[0][1][0, 0, 0]), 146)
        self.assertEqual(int(seen[0][0][30, 0, 0]), 1)
        token = result["url"].split("=")[-1]
        encoded = playback._metadata(playback.segment_video_path(token))
        self.assertEqual(encoded["frames"], 192)
        self.assertEqual(encoded["fps_fraction"], "24")
        with patch.object(playback, "_render", side_effect=AssertionError("cache miss")):
            again = playback.playback_segment(self.app, str(self.result), 570)
        self.assertTrue(again["cached"])
        self.assertEqual(again["url"], result["url"])

    def test_modes_and_amplification_have_distinct_cache_entries(self):
        a = playback.playback_segment(self.app, str(self.result), 719)
        b = playback.playback_segment(self.app, str(self.result), 719, "rgb", 64)
        self.assertEqual(a["end_frame"], 720)
        self.assertNotEqual(a["url"], b["url"])
        self.assertEqual(a["difference_basis"], "decoded_processing_input_and_result")

    def test_original_only_and_validation_fail_closed(self):
        result = playback.playback_segment(self.app, "", 719)
        self.assertEqual(set(result["regions"]), {"original"})
        for frame in (-1, True, 720, "1"):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                playback.playback_segment(self.app, str(self.result), frame)
        with self.assertRaises(ValueError):
            playback.playback_segment(self.app, str(self.source), 0)
        with self.assertRaises(ValueError):
            playback.playback_segment(self.app, str(self.result), 0, "rgb", 16)
        self.assertIsNone(playback.segment_video_path("../../escape"))
        self.assertIsNone(playback.segment_video_path("0" * 64))

    def test_original_reference_uses_raw_pixels_and_section_frame_offset(self):
        original = self.root / 'original.mkv'
        writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*'FFV1'), 24, (64, 48))
        self.assertTrue(writer.isOpened())
        for frame in range(744):
            writer.write(np.full((48, 64, 3), (frame + 20) % 240, dtype=np.uint8))
        writer.release()
        self.app.processing_input = lambda stage: str(self.source)
        self.app.settings['global']['source'] = str(original)
        seen = []
        real = playback._diff_frame
        def inspect(before, after, mode, contrast):
            if not seen:
                seen.append((int(before[0, 0, 0]), int(after[0, 0, 0])))
            return real(before, after, mode, contrast)
        with patch('ai_remaster_gui.media.source_section_state', return_value={'enabled': True, 'start': 1}), \
             patch.object(playback, '_diff_frame', side_effect=inspect):
            result = playback.playback_segment(self.app, str(self.result), 5, reference='original')
        self.assertEqual(seen[0], (44, 2))  # original frame 24 against result frame 0
        self.assertEqual(result['frames'], 720)
        self.assertEqual(result['difference_basis'], 'decoded_original_source_and_result')

    def test_twenty_minute_input_can_seek_to_minute_nineteen(self):
        source = self.root / "twenty_minutes.mkv"
        subprocess.run([playback.local_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=gray:size=64x48:rate=10:duration=1200", "-c:v", "ffv1", str(source)], check=True)
        app = SimpleNamespace(settings={"global": {"source": str(source)}})
        result = playback.playback_segment(app, "", 11400)
        self.assertEqual(result["frames"], 12000)
        self.assertEqual(result["fps"], 10)
        self.assertEqual(result["first_frame"], 11360)
        self.assertEqual(result["end_frame"], 11440)

    def test_fractional_source_rate_is_preserved_in_the_encoded_frame_mapping(self):
        source = self.root / "fractional.mkv"
        subprocess.run([playback.local_tool("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=64x48:rate=24000/1001:duration=10", "-c:v", "ffv1", str(source)], check=True)
        app = SimpleNamespace(settings={"global": {"source": str(source)}})
        result = playback.playback_segment(app, "", 200)
        self.assertEqual(result["fps_fraction"], "24000/1001")
        self.assertEqual(result["first_frame"], 192)
        encoded = playback._metadata(playback.segment_video_path(result["url"].split("=")[-1]))
        self.assertEqual(encoded["fps_fraction"], "24000/1001")
        self.assertEqual(encoded["frames"], result["end_frame"] - 192)

    def test_native_4k_layout_is_not_downscaled(self):
        width, height, regions = playback._layout(3840, 2160, True)
        self.assertEqual((width, height), (3840, 6480))
        self.assertEqual(regions["result"], [0, 2160, 3840, 2160])
        width, height, regions = playback._layout(4096, 3072, True)
        self.assertEqual((width, height), (8192, 6144))
        self.assertEqual(regions["result"], [4096, 0, 4096, 3072])
        self.assertEqual(regions["difference"], [0, 3072, 4096, 3072])
        with self.assertRaises(ValueError):
            playback._layout(7680, 4320, True)

    def test_feature_length_frame_counts_are_not_limited_to_150000(self):
        probe = SimpleNamespace(stdout=json.dumps({"streams": [{"width": 1920, "height": 1080,
            "avg_frame_rate": "30/1", "r_frame_rate": "30/1", "nb_frames": "270000"}]}))
        with patch.object(frame_inspector.subprocess, "run", return_value=probe):
            metadata = frame_inspector._metadata_cached("feature_length_regression.mkv", 1, 123)
        self.assertEqual(metadata["frames"], 270000)

    def test_playback_cache_clearing_does_not_include_denoised_masters(self):
        from ai_remaster_gui import cache
        category = next(row for row in cache.cache_categories() if row["key"] == "denoise")
        self.assertEqual(category["folders"], (playback.ROOT / ".cache" / "denoise_playback",))


if __name__ == "__main__":
    unittest.main()
