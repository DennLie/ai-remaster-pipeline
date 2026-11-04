"""CPU-only Light Deflicker tests: cuts, fades, grain and complete video CLI."""
from __future__ import annotations

import argparse
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
import cleanup_video  # noqa: E402
import deflicker_video  # noqa: E402
import film_deflicker as flicker  # noqa: E402
from ai_remaster_gui import app  # noqa: E402
from ai_remaster_gui.runtime_settings import base_settings  # noqa: E402


def film_texture() -> np.ndarray:
    rng = np.random.default_rng(43)
    pixels = np.clip(rng.normal(115, 12, (96, 128)), 0, 255).astype(np.uint8)
    pixels[26:64, 67:73] = 175
    return cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)


def exposure_sequence() -> list[np.ndarray]:
    base = film_texture()
    return [
        np.clip(np.round(base.astype(np.float32) * (1.15 if i % 2 else 0.85)), 0, 255).astype(np.uint8)
        for i in range(18)
    ]


class LightDeflickerUnitTests(unittest.TestCase):
    def test_static_scene_remains_unchanged(self):
        frame = film_texture()
        metrics = [flicker.frame_luma(frame) for _ in range(15)]
        gains, cuts = flicker.calculate_gains(
            [m[0] for m in metrics], [m[1] for m in metrics]
        )
        self.assertEqual(cuts, [0])
        self.assertTrue(all(g == 1.0 for g in gains))
        self.assertTrue(np.array_equal(flicker.apply_gain(frame, 1.0), frame))

    def test_exposure_flicker_reduces_without_blurring_detail(self):
        source = exposure_sequence()
        metrics = [flicker.frame_luma(frame) for frame in source]
        values = [m[0] for m in metrics]
        gains, cuts = flicker.calculate_gains(values, [m[1] for m in metrics])
        self.assertEqual(cuts, [0], "Brightness-only flicker must not be called a scene cut")
        self.assertGreater(sum(g != 1.0 for g in gains), 5)
        self.assertLess(np.std([v*g for v, g in zip(values, gains)]), 0.80 * np.std(values))
        self.assertLessEqual(max(abs(g-1) for g in gains), 0.0800001)
        # A global gain can change contrast but does not spatially blur pixels.
        changed = flicker.apply_gain(source[4], gains[4])
        self.assertEqual(changed.shape, source[4].shape)
        self.assertGreater(np.std(cv2.cvtColor(changed, cv2.COLOR_BGR2GRAY)), 4)

    def test_monotonic_intentional_fade_is_unmodified(self):
        base = film_texture().astype(np.float32)
        frames = [
            np.clip(np.round(base * (0.76 + i * 0.022)), 0, 255).astype(np.uint8)
            for i in range(15)
        ]
        metrics = [flicker.frame_luma(f) for f in frames]
        gains, _cuts = flicker.calculate_gains(
            [m[0] for m in metrics], [m[1] for m in metrics]
        )
        self.assertTrue(all(abs(g-1) < 1e-5 for g in gains), gains)

    def test_hard_scene_cut_separates_temporal_windows(self):
        hist1 = np.eye(16, dtype=np.float32)[2]
        hist2 = np.eye(16, dtype=np.float32)[13]
        levels = [95.0]*7 + [165.0]*7
        histograms = [hist1]*7 + [hist2]*7
        gains, cuts = flicker.calculate_gains(levels, histograms)
        self.assertEqual(cuts, [0, 7])
        self.assertTrue(all(g == 1.0 for g in gains))

    def test_bounds_and_invalid_inputs(self):
        hist = np.eye(16, dtype=np.float32)[5]
        for overrides in ({"strength": -0.1}, {"strength": 1.1},
                          {"max_correction": 0.20}, {"radius": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                flicker.calculate_gains([90, 140, 90], [hist]*3, **overrides)
        with self.assertRaises(ValueError):
            flicker.calculate_gains([90], [hist]*2)

    def test_deflicker_methods_are_separate_and_default_identity_is_preserved(self):
        from scripts import artifact_ids as ids
        base = dict(light_deflicker=True, deflicker_strength=0.65, deflicker_max_correction=0.08)
        common = dict(
            source_name="film.mkv", lora="dearchive.safetensors", prompt="",
            negative_prompt="", strength=1.0, seed=42, dearchive=False,
        )
        old = ids.cleanup_identity(**common, **base)
        arp = ids.cleanup_identity(**common, **base, deflicker_method="arp", deflicker_window=9)
        median = ids.cleanup_identity(**common, **base, deflicker_method="ffmpeg_median", deflicker_window=5)
        median9 = ids.cleanup_identity(**common, **base, deflicker_method="ffmpeg_median", deflicker_window=9)
        mean = ids.cleanup_identity(**common, **base, deflicker_method="ffmpeg_am", deflicker_window=5)
        self.assertEqual(old, arp, "Old ARP filenames must remain unchanged")
        self.assertNotEqual(ids.artifact_key(arp), ids.artifact_key(median))
        self.assertNotEqual(ids.artifact_key(median), ids.artifact_key(median9))
        self.assertEqual(median["deflicker_algorithm"], "ffmpeg_median_v1")
        self.assertEqual(mean["deflicker_algorithm"], "ffmpeg_am_v1")
        self.assertNotEqual(ids.artifact_key(median), ids.artifact_key(mean))
        self.assertNotIn("deflicker_strength", median)
        with self.assertRaises(ValueError):
            ids.cleanup_identity(**common, **base, deflicker_method="ffmpeg_median", deflicker_window=11)

    def test_gui_ffmpeg_method_command_matches_cli_artifact(self):
        settings = base_settings()
        settings["global"].update({"source": "input/film.mp4", "deflicker": "true"})
        settings["deflicker"].update({"method": "ffmpeg_median", "window": "5"})
        old_settings = app.APP.settings
        try:
            app.APP.settings = settings
            command = app.APP.command_for("deflicker")
            self.assertEqual(command[command.index("--method") + 1], "ffmpeg_median")
            self.assertEqual(command[command.index("--window") + 1], "5")
            args = deflicker_video.build_parser().parse_args(command[3:])
            self.assertEqual(args.method, "ffmpeg_median")
            self.assertEqual(args.window, "5")
            name = Path(app.APP.deflicker_output()).name
            self.assertEqual(name, deflicker_video.output_for(Path(args.source), vars(args)).name)
            self.assertNotEqual(
                name,
                deflicker_video.output_for(Path("input/film.mp4"), dict(settings["deflicker"], method="arp")).name,
            )
        finally:
            app.APP.settings = old_settings

    def test_gui_arithmetic_mean_command_and_distinct_output(self):
        settings = base_settings()
        settings["global"].update({"source": "input/film.mp4", "deflicker": "true"})
        settings["deflicker"].update({"method": "ffmpeg_am", "window": "9"})
        prior = app.APP.settings
        try:
            app.APP.settings = settings
            command = app.APP.command_for("deflicker")
            args = deflicker_video.build_parser().parse_args(command[3:])
            self.assertEqual(args.method, "ffmpeg_am")
            self.assertEqual(args.window, "9")
            predicted = Path(app.APP.deflicker_output()).name
            actual = deflicker_video.output_for(Path(args.source), vars(args)).name
            self.assertEqual(predicted, actual)
            self.assertNotEqual(predicted, deflicker_video.output_for(Path("input/film.mp4"), dict(settings["deflicker"], method="ffmpeg_median")).name)
        finally:
            app.APP.settings = prior

    def test_cleanup_gui_command_identity_and_commercial_guard(self):
        settings = base_settings()
        settings["global"].update({"source": "input/film.mp4", "deflicker": "true",
                                   "commercial_model_guard": "true"})
        settings["deflicker"].update({"method": "arp", "strength": "0.65", "max_correction": "0.08"})
        old_settings = app.APP.settings
        try:
            app.APP.settings = settings
            command = app.APP.command_for("deflicker")
            self.assertEqual(command[command.index("--strength")+1], "0.65")
            self.assertEqual(command[command.index("--max-correction")+1], "0.08")
            from ai_remaster_gui.commercial_policy import stage_issue
            self.assertEqual(stage_issue(settings, "deflicker"), "")
            args = deflicker_video.build_parser().parse_args(command[3:])
            expected = Path(app.APP.deflicker_output()).name
            self.assertEqual(expected, deflicker_video.output_for(Path(args.source), vars(args)).name)
        finally:
            app.APP.settings = old_settings


class LightDeflickerVideoSmokeTests(unittest.TestCase):
    def test_real_ffmpeg_video_flicker_and_resume(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        self.assertIsNotNone(ffmpeg, "GitHub Actions must install FFmpeg")
        self.assertIsNotNone(ffprobe, "GitHub Actions must install ffprobe")
        with tempfile.TemporaryDirectory(prefix="arp-deflick-") as td:
            work = Path(td)
            source = work / "test.mkv"
            output = work / "cleaned.mkv"
            frames = exposure_sequence()
            encoder = subprocess.Popen([
                ffmpeg, "-loglevel", "error", "-y", "-f", "rawvideo",
                "-pix_fmt", "bgr24", "-s:v", "128x96", "-r", "12",
                "-i", "pipe:0", "-an", "-c:v", "ffv1", str(source),
            ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            raw = b"".join(f.tobytes() for f in frames)
            _stdout, errors = encoder.communicate(raw, timeout=30)
            self.assertEqual(encoder.returncode, 0, errors.decode(errors="replace"))
            cmd = [
                sys.executable, str(ROOT / "scripts" / "cleanup_video.py"),
                "--source", str(source), "--output", str(output),
                "--light-deflicker", "--no-dearchive",
                "--intermediate-profile", "lossless",
            ]
            first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=150)
            self.assertEqual(first.returncode, 0, first.stdout[-3000:] + first.stderr[-3000:])
            self.assertTrue(output.exists())
            self.assertIn("Light Deflicker:", first.stdout)
            self.assertIn("Wrote Clean Up video", first.stdout)
            second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=80)
            self.assertEqual(second.returncode, 0, second.stdout[-3000:]+second.stderr[-3000:])
            self.assertIn("Reuse Clean Up output", second.stdout)

            capture = cv2.VideoCapture(str(output))
            measured = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    measured.append(flicker.frame_luma(frame)[0])
            finally:
                capture.release()
            self.assertEqual(len(measured), len(frames))
            originals = [flicker.frame_luma(f)[0] for f in frames]
            self.assertLess(float(np.std(measured)), float(np.std(originals)) * 0.85)
            report_files = sorted((ROOT / ".cache" / "cleanup_chunks").glob(
                f"{source.stem}_*/light_deflicker_*_analysis.json"
            ))
            self.assertTrue(report_files, "Expected per-frame gains report in cleanup cache")
            report = json.loads(report_files[-1].read_text())
            self.assertEqual(report["frames"], 18)
            self.assertGreater(report["nonneutral_frames"], 0)

            # FFmpeg comparison uses the *same* source, independent output and cache.
            median = work / "median.mkv"
            cmd_median = [
                sys.executable, str(ROOT / "scripts" / "cleanup_video.py"),
                "--source", str(source), "--output", str(median),
                "--light-deflicker", "--deflicker-method", "ffmpeg_median",
                "--deflicker-window", "5", "--no-dearchive",
                "--intermediate-profile", "lossless",
            ]
            first_median = subprocess.run(
                cmd_median, cwd=ROOT, capture_output=True, text=True, timeout=150
            )
            self.assertEqual(
                first_median.returncode, 0,
                first_median.stdout[-2500:] + first_median.stderr[-2500:],
            )
            self.assertTrue(median.is_file())
            self.assertIn("FFmpeg Median Deflicker", first_median.stdout)
            self.assertTrue(output.is_file(), "ARP variant must still exist")
            cap = cv2.VideoCapture(str(median))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 18)
            finally:
                cap.release()
            second_median = subprocess.run(
                cmd_median, cwd=ROOT, capture_output=True, text=True, timeout=80
            )
            self.assertEqual(second_median.returncode, 0)
            self.assertIn("Reuse Clean Up output", second_median.stdout)

            mean = work / "mean.mkv"
            cmd_mean = [
                sys.executable, str(ROOT / "scripts" / "cleanup_video.py"),
                "--source", str(source), "--output", str(mean),
                "--light-deflicker", "--deflicker-method", "ffmpeg_am",
                "--deflicker-window", "5", "--no-dearchive",
                "--intermediate-profile", "lossless",
            ]
            first_mean = subprocess.run(
                cmd_mean, cwd=ROOT, capture_output=True, text=True, timeout=150
            )
            self.assertEqual(
                first_mean.returncode, 0,
                first_mean.stdout[-2500:] + first_mean.stderr[-2500:],
            )
            self.assertTrue(mean.is_file())
            self.assertIn("FFmpeg Arithmetic Mean Deflicker", first_mean.stdout)
            capture = cv2.VideoCapture(str(mean))
            try:
                self.assertEqual(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 18)
            finally:
                capture.release()
            self.assertTrue(median.is_file())
            self.assertTrue(output.is_file())
            again = subprocess.run(cmd_mean, cwd=ROOT, capture_output=True, text=True, timeout=80)
            self.assertEqual(again.returncode, 0)
            self.assertIn("Reuse Clean Up output", again.stdout)


if __name__ == "__main__":
    unittest.main()
