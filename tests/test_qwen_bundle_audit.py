"""Standard-library, no-GPU tests of Qwen 2511 bundle provenance checks."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_qwen_2511_bundle as auditor  # noqa: E402


class QwenBundleAuditTests(unittest.TestCase):
    def test_production_manifest_tracks_downloader(self):
        manifest = auditor.load_manifest()
        self.assertEqual(len(manifest["artifacts"]), 4)
        self.assertTrue(manifest["additional_dependencies_not_covered"])
        self.assertFalse(manifest["commercially_approved"])

    def test_manifest_fails_closed_if_downloader_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            m = auditor.load_manifest()
            m["artifacts"][0]["destination"] = "models/diffusion_models/other.gguf"
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "drifted"):
                auditor.load_manifest(path)

    def test_manifest_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as folder:
            m = auditor.load_manifest()
            m["artifacts"][0]["destination"] = "../outside/model.gguf"
            path = Path(folder) / "manifest.json"
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                auditor.load_manifest(path)

    def test_verify_bundle_hashes_files_and_keeps_licence_pending(self):
        manifest = auditor.load_manifest()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for i, row in enumerate(manifest["artifacts"]):
                target = root / row["destination"]
                target.parent.mkdir(parents=True, exist_ok=True)
                content = f"test-fixture-{i}".encode()
                target.write_bytes(content)
                row["sha256"] = hashlib.sha256(content).hexdigest()
            report = auditor.verify_bundle(root, manifest)
            self.assertTrue(report["all_files_verified"])
            self.assertFalse(report["commercially_approved"])
            self.assertTrue(all(row["status"] == "verified" for row in report["artifacts"]))
            (root / manifest["artifacts"][0]["destination"]).write_bytes(b"changed")
            (root / manifest["artifacts"][1]["destination"]).unlink()
            report = auditor.verify_bundle(root, manifest)
            self.assertFalse(report["all_files_verified"])
            self.assertEqual(report["artifacts"][0]["status"], "mismatch")
            self.assertEqual(report["artifacts"][1]["status"], "missing")

    def test_cli_manifest_only_never_hashes_or_downloads(self):
        with mock.patch.object(auditor, "sha256_file", side_effect=AssertionError("no hashing")):
            self.assertEqual(auditor.main(["--check-manifest-only"]), 0)


if __name__ == "__main__":
    unittest.main()
