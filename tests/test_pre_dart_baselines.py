"""CPU-only regression and real FFmpeg comparisons of pre-DART baselines.

Video tests use synthetic film-like texture with known flicker/gate-weave.
The report describes outputs and remains deliberately independent of
DART's weights and unresolved third-party licensing.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_pre_dart_baselines as compare  # noqa: E402


def write_video(ffmpeg: str, path: Path, frames: list[np.ndarray]) -> None:
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "-s:v", f"{w}x{h}",
           "-r", "12", "-i", "pipe:0", "-an", "-c:v", "ffv1",
           "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _out, err = process.communicate(b"".join(f.tobytes() for f in frames), timeout=50)
    if process.returncode:
        raise RuntimeError(err.decode(errors="replace"))


def synthetic_frames(count: int = 18, *, jitter: bool = False) -> list[np.ndarray]:
    rng = np.random.default_rng(19)
    background = rng.integers(35, 185, (128, 192), dtype=np.uint8)
    background[30:50, 80:100] = 230
    background[89:105, 42:62] = 10
    original = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    frames = []
    for i in range(count):
        if jitter:
            dx = 3 if i % 2 else -3
            dy = 2 if i % 3 == 0 else -2
            affine = np.float32([[1, 0, dx], [0, 1, dy]])
            frame = cv2.warpAffine(original, affine, (192, 128),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            illumination = 1.12 if i % 2 else 0.88
            frame = np.clip(np.round(original.astype(np.float32) * illumination),
                            0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


class PreDartBaselineUnitTests(unittest.TestCase):
    def test_validation_rejects_bad_variant_names(self):
        with self.assertRaises(ValueError):
            compare.parse_variants(["bad=name=video.mkv", "bad=other.mkv"])
        with self.assertRaises(ValueError):
            compare.parse_variants(["original=source.mkv"])
        with self.assertRaises(ValueError):
            compare.parse_variants(["../escape=file.mkv"])
        self.assertEqual(list(compare.parse_variants(["arp=movie.mkv"])), ["arp"])

    def test_video_alignment_and_independent_frame_metrics(self):
        ffmpeg = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg, "End-to-end video tests require ffmpeg")
        with tempfile.TemporaryDirectory(prefix="arp-pre-dart-") as folder:
            root = Path(folder)
            before, after = root / "original.mkv", root / "same.mkv"
            frames = synthetic_frames(8)
            write_video(ffmpeg, before, frames)
            write_video(ffmpeg, after, frames)
            result = compare.evaluate(before, {"same": after}, root / "results",
                                      cut_after_frames={2})
            self.assertEqual(result["frames"], 8)
            self.assertEqual(result["excluded_cut_after_frames"], [2])
            self.assertLess(result["variants"]["same"]["mean_absolute_pixel_difference_from_original"], 0.00001)
            self.assertTrue((root / "results" / "paired_pre_dart_frames.jpg").exists())
            self.assertEqual(json.loads((root / "results" / "pre_dart_baselines.json").read_text())["frames"], 8)
            with self.assertRaisesRegex(ValueError, "out of range"):
                compare.evaluate(before, {"same": after}, root / "invalid",
                                 cut_after_frames={9})
            truncated = root / "truncated.mkv"
            write_video(ffmpeg, truncated, frames[:6])
            with self.assertRaisesRegex(ValueError, "ended early"):
                compare.evaluate(before, {"truncated": truncated}, root / "invalid2")

    def test_actual_arp_and_ffmpeg_deflicker_comparison(self):
        ffmpeg = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg, "End-to-end video tests require ffmpeg")
        with tempfile.TemporaryDirectory(prefix="arp-pre-dart-") as folder:
            root = Path(folder)
            before, arp, builtin = root / "original.mkv", root / "arp.mkv", root / "ffmpeg.mkv"
            write_video(ffmpeg, before, synthetic_frames(18))
            command = [sys.executable, str(ROOT / "scripts" / "cleanup_video.py"),
                       "--source", str(before), "--output", str(arp),
                       "--light-deflicker", "--no-dearchive",
                       "--intermediate-profile", "lossless"]
            done = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=150)
            self.assertEqual(done.returncode, 0, done.stdout[-2200:] + done.stderr[-2200:])
            subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                            "-i", str(before), "-vf", "deflicker=size=5:mode=median",
                            "-an", "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(builtin)],
                           check=True, capture_output=True, timeout=60)
            metrics = compare.evaluate(before, {"arp": arp, "ffmpeg": builtin}, root / "qa")
            self.assertEqual(metrics["frames"], 18)
            self.assertIn("arp", metrics["variants"])
            self.assertIn("ffmpeg", metrics["variants"])
            self.assertGreater(metrics["original_mean_abs_adjacent_median_luma"], 2.0)
            self.assertLess(metrics["variants"]["arp"]["mean_abs_adjacent_median_luma"],
                            metrics["original_mean_abs_adjacent_median_luma"])
            self.assertLess(metrics["variants"]["ffmpeg"]["mean_abs_adjacent_median_luma"],
                            metrics["original_mean_abs_adjacent_median_luma"])

    def test_stabilization_real_video_no_zoom_vs_fixed_zoom(self):
        ffmpeg = shutil.which("ffmpeg")
        self.assertIsNotNone(ffmpeg, "End-to-end video tests require ffmpeg")
        with tempfile.TemporaryDirectory(prefix="arp-stabilize-qa-") as folder:
            root = Path(folder)
            original = root / "gate_weave.mkv"
            write_video(ffmpeg, original, synthetic_frames(24, jitter=True))
            variants = {}
            for zoom in (0, 3):
                target = root / f"stabilized_{zoom}.mkv"
                cmd = [sys.executable, str(ROOT / "scripts" / "stabilize_video.py"),
                       "--source", str(original), "--output", str(target),
                       "--single-shot", "--smoothing", "5", "--zoom", str(zoom),
                       "--intermediate-profile", "lossless"]
                result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=150)
                self.assertEqual(result.returncode, 0,
                                 f"zoom={zoom}\n" + result.stdout[-2200:] + result.stderr[-2200:])
                variants[f"zoom_{zoom}"] = target
            result = compare.evaluate(original, variants, root / "qa")
            self.assertEqual(result["frames"], 24)
            self.assertEqual(result["width"], 192)
            self.assertEqual(set(result["variants"]), {"zoom_0", "zoom_3"})
            self.assertTrue((root / "qa" / "pre_dart_baselines.json").is_file())


if __name__ == "__main__":
    unittest.main()
