"""Test pinned DART research launcher without downloading or running DART."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_dart_research as research  # noqa: E402


class ExternalDartRunnerTests(unittest.TestCase):
    def test_expected_pinned_git_revision_and_weight_hash(self):
        self.assertEqual(len(research.DART_COMMIT), 40)
        self.assertEqual(len(research.DART_WEIGHTS_SHA256), 64)
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "scripts").mkdir()
            (repo / "scripts" / "inference.py").write_text("# fixture")
            weights = repo / "dart.safetensors"
            weights.write_bytes(b"fixture")
            with mock.patch.object(research.subprocess, "run") as runner:
                runner.return_value.stdout = research.DART_COMMIT + "\n"
                with mock.patch.object(research, "sha256", return_value=research.DART_WEIGHTS_SHA256):
                    research.validate_checkout(repo, weights)
            runner.assert_called_once()

    def test_checkout_change_or_wrong_weights_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            (repo / "scripts").mkdir()
            (repo / "scripts" / "inference.py").write_text("# fixture")
            weights = repo / "dart.safetensors"
            weights.write_bytes(b"fixture")
            with mock.patch.object(research.subprocess, "run") as runner:
                runner.return_value.stdout = "f" * 40
                with self.assertRaisesRegex(ValueError, "Unreviewed DART checkout"):
                    research.validate_checkout(repo, weights)
                runner.return_value.stdout = research.DART_COMMIT
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    research.validate_checkout(repo, weights)

    def test_command_always_exports_restoration_and_soft_masks(self):
        args = research.build_command(
            Path("/venv/python"), Path("/external/DART"), Path("/models/dart.safetensors"),
            Path("/footage/shot.mkv"), Path("/output"), 72, "cuda", 5, 2,
        )
        self.assertEqual(args[0], "/venv/python")
        self.assertIn("--save-masks", args)
        self.assertIn("--save-video", args)
        self.assertEqual(args[args.index("--max-frames")+1], "72")
        self.assertEqual(args[args.index("--chunk-size")+1], "5")
        with self.assertRaises(ValueError):
            research.build_command(Path("p"), Path("r"), Path("w"), Path("s"), Path("o"),
                                   1, "cpu", 7, 3)
        with self.assertRaises(ValueError):
            research.build_command(Path("p"), Path("r"), Path("w"), Path("s"), Path("o"),
                                   72, "cpu", 5, 6)

    def test_cli_requires_explicit_research_acknowledgement(self):
        with self.assertRaises(SystemExit) as context:
            research.main(["--dart-repo", "unused", "--dart-python", "unused",
                           "--source", "unused", "--output-dir", "unused"])
        self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
