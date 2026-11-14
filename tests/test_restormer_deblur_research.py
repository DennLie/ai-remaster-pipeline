"""Fail-closed Restormer research contracts; no weights, GPU or Torch needed."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_restormer_deblur_research as research


class RestormerResearchTests(unittest.TestCase):
    def test_checkpoint_integrity_is_mandatory(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "test_weights.pth"
            f.write_bytes(b"dummy bytes, never unpickle")
            checksum = hashlib.sha256(f.read_bytes()).hexdigest()
            self.assertEqual(research.verify_weights(f, checksum), checksum)
            with self.assertRaisesRegex(ValueError, "64-character"):
                research.verify_weights(f, "invalid")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                research.verify_weights(f, "0" * 64)

    def test_pinned_clean_checkout_or_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in ("LICENSE.md", research.ARCH_PATH):
                file = root / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("test fake", encoding="utf-8")
            with patch.object(research, "command_output", side_effect=[research.REVISION, ""]) as mock:
                research.verify_checkout(root)
                self.assertEqual(mock.call_count, 2)
            with patch.object(research, "command_output", return_value="c" * 40):
                with self.assertRaisesRegex(ValueError, "Unexpected"):
                    research.verify_checkout(root)
            with patch.object(research, "command_output", side_effect=[research.REVISION, "?? test.py"]):
                with self.assertRaisesRegex(ValueError, "dirty"):
                    research.verify_checkout(root)

    def test_research_only_optin_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "original.webm"
            source.write_bytes(b"not video")
            a = argparse.Namespace(
                research_only=False, source=source, output=root / "test.mkv",
                checkout=root, weights=root / "weights.pth",
                weights_sha256="0" * 64, frames=24, start_seconds=0.,
                mix=100, device="cuda", ffmpeg="ffmpeg", ffprobe="ffprobe",
            )
            with self.assertRaisesRegex(ValueError, "research-only"):
                research.validate(a)
            a.research_only = True
            a.output = a.source
            with self.assertRaisesRegex(ValueError, "overwrite"):
                research.validate(a)
            a.output = root / "test.mkv"
            a.frames = 49
            with self.assertRaisesRegex(ValueError, "between 1 and"):
                research.validate(a)
            a.frames = 24
            a.mix = 101
            with self.assertRaisesRegex(ValueError, "between 0 and 100"):
                research.validate(a)
            a.mix = 100
            a.start_seconds = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                research.validate(a)

    def test_real_rgb_frame_io_with_ffmpeg_keeps_native_geometry_and_pixels(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-restormer-") as td:
            root = Path(td)
            source, target = root / "sample.mkv", root / "restored.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=size=95x71:rate=12:duration=0.5",
                 "-c:v", "ffv1", "-pix_fmt", "bgr0", str(source)],
                check=True, capture_output=True, timeout=35,
            )
            w, h, fps = research.probe(ffprobe, source)
            self.assertEqual((w, h, fps), (95, 71, "12/1"))
            decoded = research.decode_frames(ffmpeg, source, 0., 6, w, h)
            self.assertEqual(len(decoded), 6)
            self.assertEqual(decoded[0].shape, (71, 95, 3))
            self.assertFalse(np.array_equal(decoded[0], decoded[1]))
            with self.assertRaisesRegex(ValueError, "ended before"):
                research.decode_frames(ffmpeg, source, 0., 7, w, h)
            research.encode_frames(ffmpeg, decoded, target, fps)
            self.assertTrue(target.is_file())
            self.assertEqual(research.probe(ffprobe, target), (w, h, fps))
            output = research.decode_frames(ffmpeg, target, 0., 6, w, h)
            for a, b in zip(decoded, output):
                np.testing.assert_array_equal(a, b)
            count = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                 "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(target)],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            self.assertEqual(count, "6")

    def test_import_does_not_import_torch_or_execute_external_model(self):
        self.assertTrue(callable(research.load_model))
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            f"sys.path.insert(0, r'{ROOT / 'scripts'}')\n"
            "import run_restormer_deblur_research as r\n"
            "assert callable(r.load_model)\n"
            "assert 'torch' not in sys.modules, f'torch was imported: {sys.modules.get(\"torch\")}'\n"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"Import isolation check failed: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
