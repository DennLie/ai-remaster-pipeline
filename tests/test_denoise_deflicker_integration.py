"""Cross-stage ARP integration checks: optional denoise -> Localized -> stages.

Does not benchmark film quality or enable neural weights. Verifies the exact
existing stage order and output dependencies before any production merge.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cleanup_video  # noqa: E402
import denoise_video  # noqa: E402
from ai_remaster_gui import app  # noqa: E402
from ai_remaster_gui.runtime_settings import base_settings  # noqa: E402


class IntegratedNoiseDeflickerTests(unittest.TestCase):
    def test_gui_stage_order_and_source_dependency_are_explicit(self):
        settings = base_settings()
        source = "input/synthetic_long_movie.mkv"
        settings["global"].update({
            "source": source,
            "denoise": "true",
            "cleanup": "true",
            "stabilize": "true",
            "detail": "true",
            "expand_outpaint": "false",
            "colorize": "false",
            "upscale": "false",
            "add_soundtrack": "false",
        })
        settings["denoise"].update({"method": "nlmeans", "preset": "gentle"})
        settings["cleanup"].update({
            "light_deflicker": "true",
            "deflicker_method": "localized_stream",
            "localized_confirm_film_experiment": "true",
            "dearchive": "false",
            "ai_descratch": "false",
            "devignette": "false",
            "detect_defects": "false",
        })
        before = app.APP.settings
        try:
            app.APP.settings = settings
            self.assertEqual(
                [stage.key for stage in app.APP.active_stages()][:4],
                ["denoise", "cleanup", "stabilize", "detail"],
            )
            noise = app.APP.command_for("denoise")
            cleanup = app.APP.command_for("cleanup")
            stabilize = app.APP.command_for("stabilize")
            detail = app.APP.command_for("detail")
            self.assertEqual(noise[noise.index("--source") + 1], source)
            self.assertEqual(
                cleanup[cleanup.index("--source") + 1],
                noise[noise.index("--output") + 1],
            )
            self.assertEqual(
                stabilize[stabilize.index("--source") + 1],
                cleanup[cleanup.index("--output") + 1],
            )
            self.assertEqual(
                detail[detail.index("--source") + 1],
                stabilize[stabilize.index("--output") + 1],
            )
            self.assertEqual(
                denoise_video.build_parser().parse_args(noise[3:]).method,
                "nlmeans",
            )
            opts = cleanup_video.build_parser().parse_args(cleanup[3:])
            self.assertEqual(opts.deflicker_method, "localized_stream")
            self.assertTrue(opts.localized_confirm_film_experiment)
            self.assertFalse(opts.dearchive)

            # A different denoise preset must invalidate *downstream*
            # identities rather than accidentally reusing old cleanup pixels.
            cleanup_before = cleanup[cleanup.index("--output") + 1]
            settings["denoise"]["preset"] = "balanced"
            cleanup_after = app.APP.command_for("cleanup")
            self.assertNotEqual(
                cleanup_before, cleanup_after[cleanup_after.index("--output") + 1]
            )
            # Disabling denoise must route Clean Up directly to the original.
            settings["global"]["denoise"] = "false"
            direct = app.APP.command_for("cleanup")
            self.assertEqual(direct[direct.index("--source") + 1], source)
        finally:
            app.APP.settings = before

    def test_real_ffv1_audio_denoise_then_localized_stream(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if not (ffmpeg and ffprobe):
            self.skipTest("System FFmpeg + FFprobe required")
        with tempfile.TemporaryDirectory(prefix="arp_integrated_noise_") as tmp:
            folder = Path(tmp)
            original = folder / "original.mkv"
            denoised = folder / "denoised.mkv"
            cleaned = folder / "localized.mkv"
            created = subprocess.run(
                [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=12:duration=2",
                    "-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000:duration=2",
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "ffv1", "-c:a", "pcm_s16le",
                    str(original),
                ],
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(created.returncode, 0, created.stderr)

            run_noise = denoise_video.run(
                original, denoised,
                method="nlmeans", preset="gentle",
                intermediate_profile="lossless", ffmpeg=ffmpeg,
            )
            self.assertFalse(run_noise["reused"])
            self.assertTrue(denoised.with_suffix(".mkv.sig.json").is_file())
            noise_cached = denoise_video.run(
                original, denoised,
                method="nlmeans", preset="gentle",
                intermediate_profile="lossless", ffmpeg=ffmpeg,
            )
            self.assertTrue(noise_cached["reused"])

            cmd = [
                sys.executable, str(ROOT / "scripts" / "cleanup_video.py"),
                "--source", str(denoised), "--output", str(cleaned),
                "--no-dearchive", "--light-deflicker",
                "--deflicker-method", "localized_stream",
                "--localized-confirm-film-experiment",
                "--intermediate-profile", "lossless",
            ]
            process = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=150,
            )
            self.assertEqual(
                process.returncode, 0,
                process.stdout[-1800:] + process.stderr[-1800:],
            )
            self.assertIn("Localized streaming complete: 24 frames", process.stdout)
            self.assertTrue(original.exists())
            self.assertTrue(denoised.exists())
            self.assertTrue(cleaned.exists())
            self.assertTrue(cleaned.with_suffix(".mkv.sig.json").exists())

            for result in (original, denoised, cleaned):
                p = subprocess.run(
                    [
                        ffprobe, "-v", "error", "-count_frames",
                        "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,avg_frame_rate,nb_read_frames",
                        "-of", "default=noprint_wrappers=1", str(result),
                    ],
                    capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertIn("width=96", p.stdout)
                self.assertIn("height=64", p.stdout)
                self.assertIn("avg_frame_rate=12/1", p.stdout)
                self.assertIn("nb_read_frames=24", p.stdout)
                audio = subprocess.run(
                    [
                        ffprobe, "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=index", "-of", "csv=p=0",
                        str(result),
                    ], capture_output=True, text=True, timeout=20,
                )
                self.assertEqual(audio.returncode, 0, audio.stderr)
                self.assertTrue(audio.stdout.strip(), f"Audio missing from {result}")

            repeat = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=55,
            )
            self.assertEqual(repeat.returncode, 0, repeat.stderr)
            self.assertIn("Reuse Clean Up output", repeat.stdout)


if __name__ == "__main__":
    unittest.main()
