"""End-to-end CPU smoke test: real FFmpeg video through ARP Clean Up mask-only.

Unlike synthetic NumPy mask tests, this invokes the CLI and verifies the
produced MKV, browser preview, cache reuse and video/frame geometry.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def probe_frames(path: Path) -> dict[str, object]:
    command = [
        str(FFPROBE), "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,nb_read_frames", "-of", "json", str(path)
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(output.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
    }


class TemporalDefectPipelineSmokeTests(unittest.TestCase):
    @unittest.skipUnless(FFMPEG and FFPROBE, "FFmpeg and ffprobe required")
    def test_mask_only_cleanup_real_video_and_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="arp-temporal-video-") as folder:
            root = Path(folder)
            input_video = root / "sample.mkv"
            output_video = root / "cleaned.mkv"
            input_command = [
                str(FFMPEG), "-hide_banner", "-loglevel", "error",
                "-y", "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=12:duration=1",
                "-frames:v", "12", "-c:v", "ffv1", str(input_video),
            ]
            subprocess.run(input_command, check=True, capture_output=True, text=True)
            assert input_video.is_file()
            cmd = [
                sys.executable, str(ROOT / "scripts" / "cleanup_video.py"),
                "--source", str(input_video), "--output", str(output_video),
                "--detect-defects", "--defect-mask-height", "source",
                "--no-dearchive", "--intermediate-profile", "lossless",
            ]
            completed = subprocess.run(
                cmd, cwd=ROOT, capture_output=True, text=True, timeout=120
            )
            self.assertEqual(completed.returncode, 0, completed.stdout[-3000:] + completed.stderr[-3000:])
            self.assertTrue(output_video.is_file(), completed.stdout)
            mask_preview = root / "cleaned_defect_mask.mp4"
            self.assertTrue(mask_preview.is_file(), completed.stdout)
            expected = {"width": 128, "height": 96, "frames": 12}
            for source in (output_video, mask_preview):
                with self.subTest(file=source.name):
                    self.assertEqual(probe_frames(source), expected)

            # No AI checkpoint or ComfyUI is required for this path.
            self.assertIn("No source pixels were repaired.", completed.stdout)

            # An unchanged second run should reuse the signed processing master.
            again = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(again.returncode, 0, again.stdout[-3000:] + again.stderr[-3000:])
            self.assertIn("Reuse Clean Up output", again.stdout)

            # Missing preview must force a new preview instead of falsely
            # treating a partial mask-only operation as complete.
            mask_preview.unlink()
            regenerated = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertEqual(regenerated.returncode, 0, regenerated.stdout[-3000:] + regenerated.stderr[-3000:])
            self.assertTrue(mask_preview.is_file())
            self.assertEqual(probe_frames(mask_preview), expected)


if __name__ == "__main__":
    unittest.main()
