"""Fail-closed Localized Deflicker GUI integration tests (CPU / FFmpeg)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import artifact_ids as ids  # noqa: E402
import cleanup_video  # noqa: E402
import deflicker_video  # noqa: E402
from localized_deflicker_backend import validate_localized_request  # noqa: E402
from ai_remaster_gui import app  # noqa: E402
from ai_remaster_gui.runtime_settings import base_settings  # noqa: E402


def arguments(**overrides):
    values = {
        "localized_confirm_single_shot": True,
        "dearchive": False, "ai_descratch": False,
        "devignette": False, "detect_defects": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class LocalizedIntegrationTests(unittest.TestCase):
    def test_research_output_identity_preserves_existing_variants(self):
        common = dict(
            source_name="film.mkv", lora="a.safetensors", prompt="",
            negative_prompt="", strength=1.0, seed=42,
            light_deflicker=True, dearchive=False,
        )
        keys = {
            method: ids.artifact_key(ids.cleanup_identity(
                **common, deflicker_method=method
            ))
            for method in ("arp", "ffmpeg_am", "ffmpeg_median", "localized_research")
        }
        self.assertEqual(len(set(keys.values())), 4)
        local = ids.cleanup_identity(**common, deflicker_method="localized_research")
        self.assertEqual(local["deflicker_algorithm"], "localized_oscillation_guard_v1")
        self.assertEqual(local["temporal_guard"], "oscillation")
        self.assertEqual(local["local_cap"], 0.08)
        self.assertTrue(local["experimental"])

    def test_short_single_shot_limits_and_other_stages_fail_closed(self):
        info = {"frames":72, "width":960, "height":720, "fps":30}
        self.assertEqual(validate_localized_request(arguments(), info), (72, 960, 720, 30))
        with self.assertRaisesRegex(ValueError, "confirmation"):
            validate_localized_request(arguments(localized_confirm_single_shot=False), info)
        for field in ("dearchive", "ai_descratch", "devignette", "detect_defects"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "ALONE"):
                validate_localized_request(arguments(**{field:True}), info)
        for invalid in (
            dict(info, frames=241),
            dict(info, width=1920, height=1080, frames=72),
            dict(info, fps=0),
            dict(info, frames=0),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_localized_request(arguments(), invalid)

    def test_gui_and_cli_derive_identical_output_key(self):
        settings = base_settings()
        settings["global"].update({"source":"input/film.mp4", "deflicker":"true"})
        settings["deflicker"].update({
            "method":"localized_research",
            "localized_confirm_single_shot":"true",
        })
        old = app.APP.settings
        try:
            app.APP.settings = settings
            cmd = app.APP.command_for("deflicker")
            self.assertIn("--localized-confirm-single-shot", cmd)
            args = deflicker_video.build_parser().parse_args(cmd[3:])
            self.assertEqual(args.method, "localized_research")
            self.assertTrue(args.localized_confirm_single_shot)
            expected = Path(app.APP.deflicker_output()).name
            actual = deflicker_video.output_for(Path(args.source), vars(args), args.intermediate_profile).name
            self.assertEqual(expected, actual)
            settings["deflicker"]["localized_confirm_single_shot"] = "false"
            cmd_off = app.APP.command_for("deflicker")
            self.assertNotIn("--localized-confirm-single-shot", cmd_off)
        finally:
            app.APP.settings = old

    def test_actual_ffmpeg_render_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="local_gui_test_") as tmp:
            folder = Path(tmp)
            src = folder / "source.mkv"
            out = folder / "result.mkv"
            made = subprocess.run([
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=12:duration=1.5",
                "-an", "-c:v", "ffv1", "-y", str(src),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(made.returncode, 0, made.stderr)
            cmd = [
                sys.executable, str(ROOT / "scripts" / "cleanup_video.py"),
                "--source", str(src), "--output", str(out), "--no-dearchive",
                "--light-deflicker", "--deflicker-method", "localized_research",
                "--localized-confirm-single-shot", "--intermediate-profile", "lossless",
            ]
            first = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=150)
            self.assertEqual(first.returncode, 0, first.stdout[-2000:] + first.stderr[-2000:])
            self.assertTrue(out.is_file())
            self.assertIn("Experimental Localized Deflicker", first.stdout)
            cap = cv2.VideoCapture(str(out))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 18)
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 96)
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), 64)
            finally:
                cap.release()
            second = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=70)
            self.assertEqual(second.returncode, 0, second.stdout[-1200:] + second.stderr[-1200:])
            self.assertIn("Reuse Clean Up output", second.stdout)
            no_confirm = subprocess.run(
                [x for x in cmd if x != "--localized-confirm-single-shot"],
                cwd=ROOT, capture_output=True, text=True, timeout=50
            )
            self.assertNotEqual(no_confirm.returncode, 0)
            self.assertIn("confirmation", no_confirm.stdout + no_confirm.stderr)
            self.assertTrue(out.is_file(), "Failed request must not delete the valid result.")


if __name__ == "__main__":
    unittest.main()
