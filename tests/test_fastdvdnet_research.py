"""CPU-only contract tests of the isolated FastDVDnet research adapter.

No network access, trained checkpoint, GPU, torch or external FastDVDnet
code is needed for these tests. Actual model inference needs RTX4090
and an explicitly verified research checkout/weight file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
import run_fastdvdnet_research as research  # noqa: E402


class FastDVDnetResearchContracts(unittest.TestCase):
    def test_reflect_index_exact_temporal_window_and_single_frame(self):
        self.assertEqual([research.reflect_index(i, 4) for i in (-2, -1, 0, 1, 2)],
                         [2, 1, 0, 1, 2])
        self.assertEqual([research.reflect_index(i, 4) for i in (2, 3, 4, 5)], [2, 3, 2, 1])
        self.assertEqual([research.reflect_index(i, 1) for i in (-10, 0, 10)], [0, 0, 0])
        with self.assertRaises(ValueError):
            research.reflect_index(0, 0)

    def test_checkpoint_hash_mandatory_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "checkpoint.pth"
            payload = b"research-only mock checkpoint, never load as PyTorch"
            file.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(research.verify_checkpoint(file, expected), expected)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                research.verify_checkpoint(file, "0" * 64)
            with self.assertRaisesRegex(ValueError, "full 64-character"):
                research.verify_checkpoint(file, "not-a-checksum")

    def test_clean_pinned_checkout_is_required(self):
        with tempfile.TemporaryDirectory() as root:
            external = Path(root)
            for name in research.MODEL_FILES:
                (external / name).write_text("# mock only\n", encoding="utf-8")
            with patch.object(research, "run_text", side_effect=[
                research.PINNED_COMMIT, ""
            ]) as runner:
                research.verify_checkout(external)
                self.assertEqual(runner.call_count, 2)
            with patch.object(research, "run_text", return_value="a" * 40):
                with self.assertRaisesRegex(ValueError, "Wrong FastDVDnet commit"):
                    research.verify_checkout(external)
            with patch.object(research, "run_text", side_effect=[
                research.PINNED_COMMIT, " M models.py"
            ]):
                with self.assertRaisesRegex(ValueError, "modifications"):
                    research.verify_checkout(external)

    def test_real_decoding_and_ffv1_frame_preserving_encode_without_ai(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-fastdvdnet-research-") as tmp:
            folder = Path(tmp)
            source, target = folder / "synthetic.mkv", folder / "passed.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc=size=97x71:rate=12:duration=0.5",
                 "-c:v", "ffv1", str(source)],
                check=True, capture_output=True, timeout=35,
            )
            w, h, fps = research.decode_info(ffprobe, source)
            self.assertEqual((w, h, fps), (97, 71, "12/1"))
            frames = research.decode_frames(
                ffmpeg, source, start_seconds=0, count=6, width=w, height=h
            )
            self.assertEqual(len(frames), 6)
            self.assertEqual(frames[0].shape, (71, 97, 3))
            self.assertFalse(np.array_equal(frames[0], frames[1]))
            with self.assertRaisesRegex(ValueError, "ended during frame"):
                research.decode_frames(ffmpeg, source, start_seconds=0,
                                       count=7, width=w, height=h)
            with patch.object(research, "denoise_one",
                              side_effect=lambda model, data, index, sigma, device: data[index]):
                research.encode_research_output(
                    ffmpeg, target, frames, None, 10, "cpu", w, h, fps
                )
            self.assertTrue(target.is_file())
            self.assertEqual(research.decode_info(ffprobe, target), (w, h, fps))
            count = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "csv=p=0", str(target)],
                capture_output=True, check=True, text=True, timeout=25,
            ).stdout.strip()
            self.assertEqual(int(count), 6)

    def test_research_optin_prevents_commercial_and_mutating_calls(self):
        with tempfile.TemporaryDirectory(prefix="arp-fastdvdnet-optin-") as tmp:
            root = Path(tmp)
            file = root / "source.mp4"
            file.write_bytes(b"fake")
            args = argparse.Namespace(
                source=file, output=root / "out.mkv", checkout=root,
                weights=root / "none.pth", weights_sha256="0" * 64,
                research_only=False, frames=24, sigma=10, start_seconds=0.0,
                ffmpeg="ffmpeg", ffprobe="ffprobe", device="cuda")
            with self.assertRaisesRegex(ValueError, "disabled by default"):
                research.inspect(args)
            args.research_only = True
            args.output = args.source
            with self.assertRaisesRegex(ValueError, "may not overwrite"):
                research.inspect(args)
            args.output = root / "output.mkv"
            args.frames = 121
            with self.assertRaisesRegex(ValueError, "between 1 and"):
                research.inspect(args)
            args.frames = 24
            args.sigma = 60
            with self.assertRaisesRegex(ValueError, "5..55"):
                research.inspect(args)
            args.sigma = 10
            args.start_seconds = float("nan")
            with self.assertRaisesRegex(ValueError, "between 0 and 86400"):
                research.inspect(args)


if __name__ == "__main__":
    unittest.main()
