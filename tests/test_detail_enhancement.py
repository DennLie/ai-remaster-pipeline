"""Real FFmpeg and GUI integration tests for optional native 1x detail enhancement."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import detail_enhance_video as detail  # noqa: E402


class DetailEnhancementTests(unittest.TestCase):
    def test_presets_and_blends_are_explicit_not_model_synthesis(self):
        for method in detail.METHODS:
            for preset in detail.PRESETS:
                if method == "richardson_lucy":
                    with self.assertRaisesRegex(ValueError, "dedicated CPU"):
                        detail.effect_for(method, preset)
                    continue
                effect = detail.effect_for(method, preset)
                self.assertIsNone(effect) if method == "off" else self.assertTrue(effect)
                self.assertEqual(detail.filter_graph(method, preset, 0), None)
                graph = detail.filter_graph(method, preset, 50)
                self.assertIsNone(graph) if method == "off" else self.assertIn("blend=all_expr", graph)
                self.assertFalse(effect and (";" in effect or "[" in effect))
        self.assertIn("chroma_amount=0", detail.effect_for("unsharp", "gentle"))
        self.assertIn("planes=1", detail.effect_for("cas", "gentle"))
        with self.assertRaises(ValueError):
            detail.filter_graph("arbitrary;script", "gentle", 100)
        with self.assertRaises(ValueError):
            detail.filter_graph("unsharp", "uncleared", 100)
        with self.assertRaises(ValueError):
            detail.filter_graph("cas", "gentle", 101)
        with self.assertRaises(ValueError):
            detail.filter_graph("cas", "gentle", -1)

    def test_gui_outputs_follow_denoise_cleanup_stabilize_and_detail(self):
        from ai_remaster_gui.models import STAGES
        from ai_remaster_gui.runtime_settings import base_settings
        from ai_remaster_gui.server import detail_output_for, outpaint_source_for_settings
        opts = base_settings()
        self.assertEqual(opts["global"]["detail"], "false")
        self.assertEqual(opts["detail"]["method"], "off")
        self.assertEqual([stage.key for stage in STAGES][:4],
                         ["denoise", "cleanup", "stabilize", "detail"])
        src = str(ROOT / "input" / "synthetic.mkv")
        a = detail_output_for(src, {"method": "unsharp", "preset": "gentle", "mix": "25"}, "lossless")
        b = detail_output_for(src, {"method": "unsharp", "preset": "gentle", "mix": "100"}, "lossless")
        c = detail_output_for(src, {"method": "cas", "preset": "gentle", "mix": "100"}, "lossless")
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        d = detail_output_for(src, {"method": "richardson_lucy", "preset": "gentle", "mix": "100"}, "lossless")
        self.assertNotEqual(c, d)
        opts["global"].update({"source": src, "denoise": "false", "cleanup": "false",
                               "stabilize": "false", "detail": "true"})
        opts["detail"].update({"method": "cas", "preset": "gentle", "mix": "100"})
        expected = detail_output_for(src, opts["detail"], opts["cloud"]["intermediate_format"])
        self.assertEqual(outpaint_source_for_settings(opts), expected)

    def test_real_video_backend_preserves_geometry_frames_audio_and_original(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-detail-") as folder:
            root = Path(folder)
            source = root / "source.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=12:duration=1",
                 "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                 "-map", "0:v:0", "-map", "1:a:0",
                 "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source)],
                check=True, capture_output=True, timeout=40,
            )
            original_bytes = source.read_bytes()
            original = detail.probe(ffmpeg, source)
            for method, mix in (
                ("off", 100), ("unsharp", 100), ("unsharp", 45),
                ("cas", 100), ("multiscale", 100), ("richardson_lucy", 100),
            ):
                target = root / f"{method}_{mix}.mkv"
                result = detail.run(source, target, method=method, preset="gentle",
                                    mix=mix, intermediate_profile="lossless", ffmpeg=ffmpeg)
                self.assertFalse(result["reused"])
                self.assertTrue(target.is_file())
                after = detail.probe(ffmpeg, target)
                self.assertEqual(
                    (after["width"], after["height"], after["avg_frame_rate"]),
                    (original["width"], original["height"], original["avg_frame_rate"]),
                )
                count = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                     "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(target)],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
                self.assertEqual(int(count), 12)
                audio = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries",
                     "stream=index", "-of", "csv=p=0", str(target)],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
                self.assertEqual(audio, "1")
                second = detail.run(source, target, method=method, preset="gentle",
                                    mix=mix, intermediate_profile="lossless", ffmpeg=ffmpeg)
                self.assertTrue(second["reused"])
            self.assertEqual(original_bytes, source.read_bytes())

    def test_reject_source_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "original.mkv"
            file.write_bytes(b"invalid")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                detail.run(file, file, method="unsharp", preset="gentle", mix=100,
                           intermediate_profile="lossless", ffmpeg="ffmpeg")


if __name__ == "__main__":
    unittest.main()
