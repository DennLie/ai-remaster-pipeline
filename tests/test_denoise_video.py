"""Real FFmpeg integration and GUI identities for optional modular Remove Noise."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import denoise_video  # noqa: E402


class DenoiseMethodTests(unittest.TestCase):
    def test_method_presets_are_bounded_and_explicit(self):
        for method in denoise_video.METHODS:
            for preset in denoise_video.PRESETS:
                filt = denoise_video.filter_for(method, preset)
                self.assertIsNone(filt) if method == "off" else self.assertTrue(filt.startswith(method))
                self.assertFalse(filt and (";" in filt or "[" in filt))
        with self.assertRaises(ValueError):
            denoise_video.filter_for("arbitrary-filter-injection", "gentle")
        with self.assertRaises(ValueError):
            denoise_video.filter_for("nlmeans", "aggressive;rm-rf")

    def test_gui_defaults_and_independent_identifiers(self):
        from ai_remaster_gui.models import STAGES
        from ai_remaster_gui.runtime_settings import base_settings
        from ai_remaster_gui.server import denoise_output_for, outpaint_source_for_settings
        opts = base_settings()
        self.assertEqual(opts["global"]["denoise"], "false")
        self.assertEqual(opts["denoise"]["method"], "off")
        self.assertEqual(opts["global"]["deflicker"], "false")
        self.assertEqual([stage.key for stage in STAGES][:5],
                         ["denoise", "deflicker", 'enhancement', "cleanup", "stabilize"])
        source = str(ROOT / "input" / "fixture.mkv")
        first = denoise_output_for(source, {"method": "hqdn3d", "preset": "gentle"}, "lossless")
        second = denoise_output_for(source, {"method": "hqdn3d", "preset": "balanced"}, "lossless")
        third = denoise_output_for(source, {"method": "nlmeans", "preset": "gentle"}, "lossless")
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, third)
        opts["global"].update({"source": source, "denoise": "true", "cleanup": "false", "stabilize": "false"})
        opts["denoise"].update({"method": "hqdn3d", "preset": "gentle"})
        self.assertEqual(outpaint_source_for_settings(opts),
                         denoise_output_for(source, opts["denoise"], opts["cloud"]["intermediate_format"]))

    def test_end_to_end_all_classical_backends_and_reuse(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        self.assertIsNotNone(ffmpeg)
        self.assertIsNotNone(ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-denoise-") as tmp:
            root = Path(tmp)
            input_video = root / "noisy_source.mkv"
            # Source contains texture and a regular level change (not ground truth for
            # restoration); audio confirms -map 0:a? survives the filter.
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "testsrc2=size=128x96:rate=12:duration=1",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1",
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "ffv1", "-c:a", "pcm_s16le",
                str(input_video),
            ], capture_output=True, check=True, timeout=45)
            original = denoise_video.probe(ffmpeg, input_video)
            outputs = {}
            for method in denoise_video.METHODS:
                target = root / f"{method}.mkv"
                result = denoise_video.run(input_video, target, method=method, preset="gentle",
                                           intermediate_profile="lossless", ffmpeg=ffmpeg)
                self.assertEqual(result["method"], method)
                self.assertTrue(target.is_file())
                self.assertTrue(target.with_suffix(".mkv.sig.json").is_file())
                info = denoise_video.probe(ffmpeg, target)
                self.assertEqual((info["width"], info["height"]), (original["width"], original["height"]))
                self.assertEqual(info["avg_frame_rate"], original["avg_frame_rate"])
                self.assertEqual(len(subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "a", "-show_entries",
                     "stream=index", "-of", "csv=p=0", str(target)],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()), 1, f"Audio missing for {method}")
                output_meta = subprocess.run(
                    [ffprobe, "-v", "error", "-count_frames",
                     "-select_streams", "v:0", "-show_entries",
                     "stream=nb_read_frames", "-of", "default=nk=1:nw=1", str(target)],
                    capture_output=True, text=True, check=True,
                ).stdout.strip()
                self.assertEqual(int(output_meta), 12)
                outputs[method] = target
            second = denoise_video.run(input_video, outputs["hqdn3d"], method="hqdn3d",
                                       preset="gentle", intermediate_profile="lossless", ffmpeg=ffmpeg)
            self.assertTrue(second["reused"])
            stat = input_video.stat()
            os.utime(input_video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            target_stat = outputs['hqdn3d'].stat()
            reused = denoise_video.run(input_video, outputs['hqdn3d'], method='hqdn3d',
                preset='gentle', intermediate_profile='lossless', ffmpeg=ffmpeg)
            self.assertTrue(reused['reused'])
            self.assertEqual(outputs['hqdn3d'].stat().st_mtime_ns, target_stat.st_mtime_ns)
            provenance = json.loads(outputs['hqdn3d'].with_suffix('.mkv.sig.json').read_text())
            self.assertEqual(provenance['source_fingerprint']['mtime_ns'], input_video.stat().st_mtime_ns)
            target = outputs['hqdn3d']
            before = target.read_bytes()
            # An encoding failure before replacement preserves the last success.
            real_run = subprocess.run
            def fail_encoding(command, **kwargs):
                if str(command[-1]).endswith('.partial.mkv'):
                    raise subprocess.CalledProcessError(1, command)
                return real_run(command, **kwargs)
            with patch.object(denoise_video.subprocess, 'run', side_effect=fail_encoding), self.assertRaises(subprocess.CalledProcessError):
                denoise_video.run(input_video, target, method='hqdn3d', preset='gentle',
                    intermediate_profile='lossless', ffmpeg=ffmpeg, force=True)
            self.assertEqual(target.read_bytes(), before)
            self.assertTrue(target.with_suffix('.mkv.sig.json').is_file())
            # After replacement, failed publication cannot retain the old ready marker.
            with patch.object(denoise_video, 'write_signature', side_effect=OSError('publication failed')), self.assertRaises(OSError):
                denoise_video.run(input_video, target, method='hqdn3d', preset='gentle',
                    intermediate_profile='lossless', ffmpeg=ffmpeg, force=True)
            self.assertFalse(target.with_suffix('.mkv.sig.json').exists())

    def test_rejects_same_source_output(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "same.mkv"
            src.write_bytes(b"fake")
            with self.assertRaises(ValueError):
                denoise_video.run(src, src, method="hqdn3d", preset="gentle",
                                  intermediate_profile="lossless", ffmpeg="ffmpeg")


if __name__ == "__main__":
    unittest.main()
