"""Offline contracts for FlashVSR+ — NO model imports, weights or GPU needed."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_flashvsr_plus_research as flash  # noqa: E402


def fake_assets(root: Path, mode: str = "turbo") -> Path:
    paths = flash.checkpoint_paths(root, mode)
    hashes = {}
    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"test-only-{key}".encode()
        path.write_bytes(data)
        hashes[key] = hashlib.sha256(data).hexdigest()
    manifest = root.parent / "outside-flashvsr-assets.json"
    manifest.write_text(json.dumps(hashes), encoding="utf-8")
    return manifest


class FlashVSRPlusContracts(unittest.TestCase):
    def test_pin_checkout_and_untracked_code_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for key in ("LICENSE", "inference/run_stream.py",
                        "inference/presets.py", "prompt_tensor/posi_prompt.pth"):
                file = root / key
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_text("pinned fake test file")
            with patch.object(flash, "exec_text", side_effect=[flash.PINNED_COMMIT, ""]):
                flash.verify_checkout(root)
            with patch.object(flash, "exec_text", return_value="f" * 40):
                with self.assertRaisesRegex(ValueError, "Unexpected upstream revision"):
                    flash.verify_checkout(root)
            with patch.object(flash, "exec_text", side_effect=[flash.PINNED_COMMIT, " M inference/run_stream.py"]):
                with self.assertRaisesRegex(ValueError, "modifications"):
                    flash.verify_checkout(root)

    def test_four_mandatory_model_asset_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "external"
            root.mkdir()
            manifest = fake_assets(root)
            values = flash.verify_assets(root, "turbo", manifest)
            self.assertEqual(len(values), 4)
            for value in values.values():
                self.assertEqual(len(value), 64)
            bad = json.loads(manifest.read_text())
            bad["dit.safetensors"] = "a" * 64
            manifest.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                flash.verify_assets(root, "turbo", manifest)
            bad.pop("TCDecoder.ckpt")
            manifest.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "exactly four"):
                flash.verify_assets(root, "turbo", manifest)
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                flash.checkpoint_paths(root, "unrecognized")

    def test_dry_run_real_video_metadata_checks_and_cli_contract(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="flashvsr-offline-") as td:
            tmp = Path(td)
            source = tmp / "source.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12:duration=1",
                 "-c:v", "ffv1", str(source)],
                check=True, timeout=40, capture_output=True,
            )
            checkout = tmp / "pinned-code"
            checkout.mkdir()
            manifest = fake_assets(checkout)
            args = argparse.Namespace(
                source=source, output=tmp / "output.mp4", checkout=checkout,
                manifest=manifest, mode="turbo", width=1024, height=768,
                frames=8, start_seconds=0., temporal_compression=4,
                seed=42, research_only=True, dry_run=True,
                python=Path(sys.executable), ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            with patch.object(flash, "verify_checkout", return_value=None):
                report = flash.run(args)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["source_geometry"], [160, 120])
            self.assertEqual(report["output_geometry"], [1024, 768])
            self.assertEqual(report["fps"], "12/1")
            self.assertEqual(report["source_frames"], 12)
            self.assertEqual(report["frames_requested"], 8)
            cmd = report["representative_command"]
            self.assertIn("inference.run_stream", cmd)
            self.assertEqual(cmd[cmd.index("--attn-kernel") + 1], "sdpa")
            self.assertEqual(cmd[cmd.index("--temporal-compression") + 1], "4")
            self.assertIn("--no-color-correction", cmd)
            self.assertNotIn("pip", cmd)
            self.assertFalse(args.output.exists())
            args.research_only = False
            with self.assertRaisesRegex(ValueError, "research-only"):
                flash.validate(args)
            args.research_only = True
            args.output = args.source
            with self.assertRaisesRegex(ValueError, "overwrite"):
                flash.validate(args)
            args.output = tmp / "other.mp4"
            args.width = 1000
            with self.assertRaisesRegex(ValueError, "multiples of 128"):
                flash.validate(args)
            args.width = 1024
            args.height = 1024
            with patch.object(flash, "verify_checkout", return_value=None):
                with self.assertRaisesRegex(ValueError, "aspect ratio"):
                    flash.validate(args)
            args.height = 768
            args.start_seconds = 0.5
            args.frames = 10
            with patch.object(flash, "verify_checkout", return_value=None):
                with self.assertRaisesRegex(ValueError, "beyond"):
                    flash.validate(args)
            args.start_seconds = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                flash.validate(args)

    def test_no_torch_or_external_code_imported_at_module_level(self):
        self.assertNotIn("torch", sys.modules)
        self.assertTrue(callable(flash.build_upstream_command))


if __name__ == "__main__":
    unittest.main()
