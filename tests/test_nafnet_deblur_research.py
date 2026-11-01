"""Offline checks for opt-in NAFNet GoPro research deblurring.

No checkpoint execution, CUDA or external Python imports in CI.
Real FFmpeg source and FFV1 output are tested with passthrough images.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_nafnet_deblur_research as research  # noqa: E402


class NAFNetContractTests(unittest.TestCase):
    def test_sha_is_mandatory_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            weights = Path(td) / "research_mock.pth"
            weights.write_bytes(b"not a real checkpoint")
            good = hashlib.sha256(weights.read_bytes()).hexdigest()
            self.assertEqual(research.verify_weights(weights, good), good)
            with self.assertRaisesRegex(ValueError, "64-character"):
                research.verify_weights(weights, "unknown")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                research.verify_weights(weights, "a" * 64)

    def test_exact_checkout_and_clean_tree(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ("LICENSE", "basicsr/models/archs/NAFNet_arch.py",
                         "basicsr/models/archs/local_arch.py"):
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("mock only", encoding="utf-8")
            with patch.object(research, "command_text", side_effect=[research.REVISION, ""]) as check:
                research.verify_code_checkout(root)
                self.assertEqual(check.call_count, 2)
            with patch.object(research, "command_text", return_value="bad" * 13 + "b"):
                with self.assertRaisesRegex(ValueError, "Wrong NAFNet commit"):
                    research.verify_code_checkout(root)
            with patch.object(research, "command_text", side_effect=[research.REVISION, "?? new.py"]):
                with self.assertRaisesRegex(ValueError, "untracked"):
                    research.verify_code_checkout(root)

    def test_research_guard_precedes_external_code_or_neural_weights(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            video = root / "source.mkv"
            video.write_bytes(b"mock")
            a = argparse.Namespace(source=video, output=root / "research.mkv",
                                   checkout=root, weights=root / "not.pth",
                                   weights_sha256="a" * 64, research_only=False,
                                   width=32, frames=24, start_seconds=0.,
                                   mix=100, device="cuda", ffmpeg="ffmpeg", ffprobe="ffprobe")
            with self.assertRaisesRegex(ValueError, "research-only"):
                research.validate_inputs(a)
            a.research_only = True
            a.output = a.source
            with self.assertRaisesRegex(ValueError, "overwrite"):
                research.validate_inputs(a)
            a.output = root / "output.mkv"
            a.frames = 100
            with self.assertRaisesRegex(ValueError, "between 1"):
                research.validate_inputs(a)
            a.frames = 24
            a.mix = 101
            with self.assertRaisesRegex(ValueError, "0..100"):
                research.validate_inputs(a)
            a.mix = 100
            a.start_seconds = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                research.validate_inputs(a)

    def test_real_ffmpeg_rgb_decode_lossless_mkv_encode_and_frame_count(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            source, output = folder / "synthetic.mkv", folder / "copy.mkv"
            # Use RGB source to verify exact roundtrip, incl. odd dimensions.
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=size=95x71:rate=12:duration=0.5",
                 "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source)],
                capture_output=True, check=True, timeout=40,
            )
            w, h, fps = research.probe(ffprobe, source)
            self.assertEqual((w, h, fps), (95, 71, "12/1"))
            frames = research.decode_rgb(ffmpeg, source, 0., 6, w, h)
            self.assertEqual(len(frames), 6)
            self.assertEqual(frames[0].shape, (71, 95, 3))
            self.assertFalse(np.array_equal(frames[0], frames[1]))
            with self.assertRaisesRegex(ValueError, "ended"):
                research.decode_rgb(ffmpeg, source, 0., 7, w, h)
            research.write_lossless(ffmpeg, frames, fps, output)
            self.assertEqual(research.probe(ffprobe, output), (w, h, fps))
            decoded = research.decode_rgb(ffmpeg, output, 0., 6, w, h)
            self.assertEqual(len(decoded), 6)
            for before, after in zip(frames, decoded):
                np.testing.assert_array_equal(after, before)
            raw = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                                  "-count_frames", "-show_entries", "stream=nb_read_frames",
                                  "-of", "csv=p=0", str(output)],
                                 check=True, capture_output=True, text=True)
            self.assertEqual(raw.stdout.strip(), "6")

    def test_nafnet_code_is_not_loaded_by_import(self):
        # A strict safety regression: there must be no import of Torch or
        # the untrusted model code from module initialization.
        self.assertNotIn("torch", sys.modules)
        self.assertTrue(callable(research.prepare_model))


if __name__ == "__main__":
    unittest.main()
