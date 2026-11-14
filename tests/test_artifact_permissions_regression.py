from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_remaster_gui import dust, paths
from scripts import common


class TestArtifactPermissionsRegression(unittest.TestCase):
    """Windows-specific regression tests verifying that ARP artifact creation
    and promotion paths produce files and directories with standard inherited
    NTFS permissions accessible to normal, non-elevated users.
    """

    def setUp(self) -> None:
        self.test_dir = paths.ROOT / "intermediate" / f".perm_test_{os.getpid()}_{id(self)}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._cleanup)

    def _cleanup(self) -> None:
        if self.test_dir.exists():
            try:
                shutil.rmtree(self.test_dir, ignore_errors=True)
            except Exception:
                pass

    def _get_icacls_text(self, path: Path) -> str:
        """Run icacls on path and return stdout text."""
        res = subprocess.run(
            ["icacls", str(path)],
            capture_output=True,
            check=True,
        )
        return res.stdout.decode("cp850", errors="replace")

    def test_staging_directory_inherits_acls_without_0700_lockdown(self) -> None:
        """Verify staging_directory creates folders with inherited permissions
        rather than CPython's 0o700 explicit non-inheriting DACL.
        """
        with paths.staging_directory(prefix=".test-stage-", dir=self.test_dir) as stage:
            self.assertTrue(stage.exists())
            test_file = stage / "sample.mkv"
            test_file.write_bytes(b"dummy video content")

            if sys.platform == "win32":
                acl_text = self._get_icacls_text(stage)
                # In standard Windows NTFS, inherited permissions contain '(I)'
                self.assertIn("(I)", acl_text, f"Staging directory must inherit parent ACLs: {acl_text}")

                file_acl = self._get_icacls_text(test_file)
                self.assertIn("(I)", file_acl, f"File in staging directory must inherit parent ACLs: {file_acl}")

        # Verify cleanup on exit
        self.assertFalse(stage.exists())

    def test_ensure_inherited_permissions_resets_restrictive_dacl(self) -> None:
        """Verify ensure_inherited_permissions resets explicit non-inheriting DACLs
        (e.g., from tempfile.mkdtemp 0o700) back to inherited parent permissions.
        """
        if sys.platform != "win32":
            self.skipTest("Windows NTFS-specific test")

        # Create a directory with mode 0o700 to reproduce CPython's restrictive DACL
        restricted_dir = self.test_dir / "restricted_0700"
        os.mkdir(restricted_dir, 0o700)
        restricted_file = restricted_dir / "restricted.mkv"
        restricted_file.write_bytes(b"locked")

        try:
            before_acl = self._get_icacls_text(restricted_dir)
            # Before reset, 0o700 sets explicit ACEs without (I) inheritance
            self.assertNotIn("(I)", before_acl, f"0o700 directory should initially lack (I): {before_acl}")

            # Call ensure_inherited_permissions
            paths.ensure_inherited_permissions(restricted_dir)

            after_acl = self._get_icacls_text(restricted_dir)
            self.assertIn("(I)", after_acl, f"ensure_inherited_permissions must restore (I): {after_acl}")

            after_file_acl = self._get_icacls_text(restricted_file)
            self.assertIn("(I)", after_file_acl, f"Contained files must also inherit permissions: {after_file_acl}")
        finally:
            shutil.rmtree(restricted_dir, ignore_errors=True)

    def test_replace_with_retry_ensures_inherited_permissions(self) -> None:
        """Verify replace_with_retry guarantees the destination file inherits
        permissions even if the partial source came from a restricted origin.
        """
        if sys.platform != "win32":
            self.skipTest("Windows NTFS-specific test")

        source_partial = self.test_dir / "source.partial.mkv"
        target_final = self.test_dir / "final.mkv"

        # Create source with mode 0o700 directory staging simulation
        source_partial.write_bytes(b"encoded video bytes")
        # Explicitly make source non-inheriting to simulate move across contexts
        subprocess.run(["icacls", str(source_partial), "/inheritance:d"], capture_output=True, check=True)

        common.replace_with_retry(source_partial, target_final)
        self.assertTrue(target_final.exists())

        final_acl = self._get_icacls_text(target_final)
        self.assertIn("(I)", final_acl, f"Promoted target must have inherited permissions: {final_acl}")
        target_final.unlink()

    def test_real_dust_mask_and_repair_generates_readable_artifacts_with_inherited_acls(self) -> None:
        """Full end-to-end regression: generate real Dust & Scratches mask and repair
        artifacts in intermediate/dust_scratches, verifying that the produced folder
        and repaired.mkv inherit parent ACLs and can be read by a normal non-elevated user.
        """
        ffmpeg = dust.find_ffmpeg()
        source = self.test_dir / "test_input.mkv"

        # Generate a small 12-frame synthetic clip using ffmpeg
        subprocess.run(
            [ffmpeg, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=1",
             "-c:v", "ffv1", str(source)],
            check=True,
        )

        values = {"detector": "temporal", "repair": "telea"}
        # Execute real mask pass
        mask_report = dust.run(source, values, "mask")
        folder = dust.folder_for(source, values)
        self.addCleanup(lambda: shutil.rmtree(folder, ignore_errors=True) if folder.exists() else None)

        self.assertTrue(folder.exists(), f"Expected dust analysis folder at {folder}")
        self.assertFalse(mask_report["repair_complete"])

        # Execute real repair pass
        repair_report = dust.run(source, values, "repair")
        self.assertTrue(repair_report["repair_complete"])

        repaired_mkv = folder / "repaired.mkv"
        self.assertTrue(repaired_mkv.exists(), f"repaired.mkv must exist at {repaired_mkv}")

        # 1. Verify standard file reading in non-elevated process
        with open(repaired_mkv, "rb") as f:
            header = f.read(4)
            self.assertEqual(header, b"\x1a\x45\xdf\xa3", "repaired.mkv must be readable valid Matroska stream")

        # 2. On Windows, verify NTFS ACLs are inherited and not locked down
        if sys.platform == "win32":
            folder_acl = self._get_icacls_text(folder)
            self.assertIn("(I)", folder_acl, f"Artifact folder must inherit parent ACLs: {folder_acl}")

            file_acl = self._get_icacls_text(repaired_mkv)
            self.assertIn("(I)", file_acl, f"repaired.mkv must inherit parent ACLs: {file_acl}")

            # Verify that non-inherited (OI)(CI)(F) lockdown is absent
            # In 0o700, the output only had non-inherited SYSTEM:(OI)(CI)(F), Administrators:(OI)(CI)(F)
            lines = [line.strip() for line in file_acl.splitlines() if line.strip()]
            for line in lines:
                if "repaired.mkv" in line:
                    # After filename, entries follow
                    continue
                # If an entry exists for Users or Authenticated Users, it should be inherited
                if "Benutzer" in line or "Users" in line:
                    self.assertIn("(I)", line, f"User permission line must be inherited: {line}")
