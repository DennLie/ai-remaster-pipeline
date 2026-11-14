from __future__ import annotations

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_remaster_gui import paths, project_io, server, stage_stamps


class TestPermissionErrorHydration(unittest.TestCase):
    """Regression tests covering PermissionError during Path.is_file(),
    missing artifacts and persisted project-state recovery.
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.settings_file = Path(self.tmp_dir.name) / "test_settings.json"
        patcher = mock.patch.object(server, "SETTINGS_FILE", self.settings_file)
        patcher.start()
        self.addCleanup(patcher.stop)
        server.APP.log = []

    def test_paths_helpers_under_permission_error(self) -> None:
        """Verify check_artifact_access, safe_is_file, and safe_exists handle PermissionError."""
        fake_path = Path(self.tmp_dir.name) / "locked_file.mkv"
        perm_err = PermissionError(13, "Access is denied", str(fake_path), 5)

        with mock.patch("os.stat", side_effect=perm_err):
            # check_artifact_access with require_file
            ready, err = paths.check_artifact_access(fake_path, require_file=True)
            self.assertFalse(ready)
            self.assertIsInstance(err, PermissionError)
            self.assertEqual(err.winerror if hasattr(err, "winerror") else 13, 5 if hasattr(err, "winerror") else 13)

            # check_artifact_access without require_file
            ready_exists, err_exists = paths.check_artifact_access(fake_path, require_file=False)
            self.assertFalse(ready_exists)
            self.assertIsInstance(err_exists, PermissionError)

            # safe_is_file with callback
            logged: list[str] = []
            res = paths.safe_is_file(fake_path, on_error=lambda p, exc: logged.append(f"{p}: {exc}"))
            self.assertFalse(res)
            self.assertEqual(len(logged), 1)
            self.assertIn("Access is denied", logged[0])

            # safe_exists with callback
            logged_exists: list[str] = []
            res_exists = paths.safe_exists(fake_path, on_error=lambda p, exc: logged_exists.append(f"{p}: {exc}"))
            self.assertFalse(res_exists)
            self.assertEqual(len(logged_exists), 1)

            # resolve_video_source should not crash on PermissionError
            resolved = paths.resolve_video_source(str(fake_path))
            self.assertEqual(resolved, paths.resolve(str(fake_path)))

    def test_startup_hydration_handles_permission_error_and_preserves_state(self) -> None:
        """Verify startup hydration handles inaccessible artifact (PermissionError)
        without crashing the GUI, reports the error clearly, does not validate
        the inaccessible file, and preserves saved project state.
        """
        app = server.APP
        orig_log_len = len(app.log)

        # Create simulated project state referencing a dust repaired.mkv
        persisted_mkv = "intermediate/dust_scratches/test_dir/repaired.mkv"
        persisted_manifest = "manifests/references/test_shots.csv"
        initial_settings = {
            "global": {
                "source": "input/fixture_clip.mp4",
                "dust": "true",
                "expand_outpaint": "false",
                "colorize": "true",
            },
            "dust": {
                "detector": "farneback_3f",
                "repair": "telea",
            },
            "shots": {
                "outpainted_video": persisted_mkv,
                "manifest": persisted_manifest,
            },
            "references": {
                "manifest": persisted_manifest,
            },
            "colour": {
                "manifest": persisted_manifest,
            },
            "recomp": {
                "source": persisted_mkv,
                "manifest": persisted_manifest,
                "output": "output/reassembled/repaired_recomp.mp4",
            },
            "output": {
                "output": persisted_mkv,
            },
        }

        app.settings = copy.deepcopy(initial_settings)
        self.settings_file.write_text(json.dumps(app.settings, indent=2), encoding="utf-8")

        # Mock dust_output to return our test path
        perm_err = PermissionError(13, "Access is denied", persisted_mkv, 5)
        orig_os_stat = os.stat

        def conditional_stat(path, *args, **kwargs):
            if "repaired.mkv" in str(path) or "test_dir" in str(path):
                raise perm_err
            return orig_os_stat(path, *args, **kwargs)

        with mock.patch.object(app, "dust_output", return_value=persisted_mkv), \
             mock.patch("os.stat", side_effect=conditional_stat), \
             mock.patch("sys.stderr", new_callable=io.StringIO) as fake_stderr:

            # Execute startup hydration
            app.normalize_loaded_source_state()

            # 1. Did not crash
            # 2. Check diagnostic logging
            new_logs = app.log[orig_log_len:]
            inaccessible_logs = [entry for entry in new_logs if "Inaccessible artifact" in entry]
            self.assertTrue(len(inaccessible_logs) > 0, f"Expected inaccessible log entry, got: {new_logs}")
            self.assertIn("Access is denied", inaccessible_logs[0])
            self.assertIn(persisted_mkv, inaccessible_logs[0])

            # Check stderr warning output
            stderr_output = fake_stderr.getvalue()
            self.assertIn("[ARP WARNING]", stderr_output)
            self.assertIn("Access is denied", stderr_output)

            # 3. Artifact is NOT treated as validated/usable
            # (output_selection_state should not return kind='dust' as ready)
            sel = app.output_selection_state()
            self.assertNotEqual(sel.get("kind"), "dust")

            # 4. Preserved saved artifact references in settings
            self.assertEqual(app.settings.get("shots", {}).get("outpainted_video"), persisted_mkv)
            self.assertEqual(app.settings.get("recomp", {}).get("source"), persisted_mkv)
            self.assertEqual(app.settings.get("references", {}).get("manifest"), persisted_manifest)
            self.assertEqual(app.settings.get("recomp", {}).get("manifest"), persisted_manifest)
            self.assertEqual(app.settings.get("output", {}).get("output"), persisted_mkv)

            # Verify persisted settings on disk also preserved the references
            disk_settings = json.loads(self.settings_file.read_text(encoding="utf-8"))
            self.assertEqual(disk_settings.get("shots", {}).get("outpainted_video"), persisted_mkv)
            self.assertEqual(disk_settings.get("output", {}).get("output"), persisted_mkv)

    def test_missing_artifact_handling_without_permission_error(self) -> None:
        """Verify normal missing artifact does not log inaccessible warning."""
        app = server.APP
        orig_log_len = len(app.log)

        nonexistent_mkv = "intermediate/dust_scratches/nonexistent_hash/repaired.mkv"
        app.settings = {
            "global": {
                "source": "input/fixture_clip.mp4",
                "dust": "true",
                "expand_outpaint": "false",
            },
            "dust": {
                "detector": "farneback_3f",
                "repair": "telea",
            },
            "shots": {},
            "recomp": {},
            "output": {},
        }
        self.settings_file.write_text(json.dumps(app.settings, indent=2), encoding="utf-8")

        with mock.patch.object(app, "dust_output", return_value=nonexistent_mkv):
            app.normalize_loaded_source_state()

            # No inaccessible logs (since it's a normal missing file, not PermissionError)
            new_logs = app.log[orig_log_len:]
            inaccessible_logs = [entry for entry in new_logs if "Inaccessible artifact" in entry]
            self.assertEqual(len(inaccessible_logs), 0)

            # Not treated as ready
            self.assertEqual(app.settings.get("shots", {}).get("outpainted_video"), "")

    def test_persisted_project_state_recovery_when_permissions_restored(self) -> None:
        """Verify that when file permissions are restored, the preserved state hydrates cleanly."""
        app = server.APP
        real_file = Path(self.tmp_dir.name) / "repaired.mkv"
        real_file.write_bytes(b"dummy video")
        persisted_mkv = str(real_file)

        app.settings = {
            "global": {
                "source": "input/fixture_clip.mp4",
                "dust": "true",
                "expand_outpaint": "false",
            },
            "dust": {
                "detector": "farneback_3f",
                "repair": "telea",
            },
            "shots": {
                "outpainted_video": persisted_mkv,
            },
            "recomp": {
                "source": persisted_mkv,
            },
            "output": {
                "output": persisted_mkv,
            },
        }

        # Step 1: Inaccessible (PermissionError)
        perm_err = PermissionError(13, "Access is denied", persisted_mkv, 5)
        orig_os_stat = os.stat

        def conditional_stat(path, *args, **kwargs):
            if "repaired.mkv" in str(path):
                raise perm_err
            return orig_os_stat(path, *args, **kwargs)

        with mock.patch.object(app, "dust_output", return_value=persisted_mkv), \
             mock.patch("os.stat", side_effect=conditional_stat):
            app.hydrate_stage_inputs("global")
            self.assertEqual(app.settings.get("shots", {}).get("outpainted_video"), persisted_mkv)

        # Step 2: Permission restored (file on disk is readable without mock)
        with mock.patch.object(app, "dust_output", return_value=persisted_mkv):
            app.hydrate_stage_inputs("global")
            # Now properly validated and active
            self.assertEqual(app.settings.get("shots", {}).get("outpainted_video"), persisted_mkv)
            self.assertEqual(app.settings.get("output", {}).get("output"), persisted_mkv)

    def test_progress_handles_permission_error(self) -> None:
        """Verify that progress() does not crash and does not treat locked files as Ready."""
        app = server.APP
        persisted_mkv = "intermediate/dust_scratches/test_dir/repaired.mkv"
        perm_err = PermissionError(13, "Access is denied", persisted_mkv, 5)
        orig_stat = os.stat

        def conditional_stat(path, *args, **kwargs):
            if "repaired.mkv" in str(path) or "test_dir" in str(path):
                raise perm_err
            return orig_stat(path, *args, **kwargs)

        with mock.patch.object(app, "expected_outputs", return_value=[persisted_mkv]), \
             mock.patch("os.stat", side_effect=conditional_stat):
            rows = app.progress()
            self.assertIsInstance(rows, list)
            # Dust & Scratches should not be Ready
            dust_rows = [r for r in rows if r.get("stage") == "Dust & Scratches"]
            if dust_rows:
                self.assertEqual(dust_rows[0].get("status"), "Waiting")

    def test_files_for_handles_permission_error(self) -> None:
        """Verify that files_for() skips inaccessible files without aborting."""
        app = server.APP
        stage = next(s for s in server.STAGES if s.key == "dust")
        perm_err = PermissionError(13, "Access is denied", "repaired.mkv", 5)

        with mock.patch("pathlib.Path.is_file", side_effect=perm_err):
            # Should not raise
            res = app.files_for(stage)
            self.assertIsInstance(res, list)

    def test_api_state_views_under_permission_error(self) -> None:
        """Verify that state() for all views completes without uncaught PermissionError."""
        app = server.APP
        orig_stat = os.stat
        perm_err = PermissionError(13, "Access is denied", "intermediate/dust_scratches/48809ca17edbcadd4ad8d341/repaired.mkv", 5)

        def conditional_stat(path, *args, **kwargs):
            if "repaired.mkv" in str(path) or "48809ca17edbcadd4ad8d341" in str(path):
                raise perm_err
            return orig_stat(path, *args, **kwargs)

        views = [
            "overview", "denoise", "deflicker", "enhancement", "cleanup", "stabilize",
            "dust", "detail", "outpaint", "shots", "references", "colour", "recomp",
            "audio", "upscale", "output"
        ]

        with mock.patch("os.stat", side_effect=conditional_stat):
            for v in views:
                st = app.state(v)
                self.assertIsInstance(st, dict)
                self.assertIn("system_status", st)

    def test_project_io_under_permission_error(self) -> None:
        """Verify project_io helpers handle PermissionError gracefully."""
        perm_err = PermissionError(13, "Access is denied", "locked.mkv", 5)
        with mock.patch("os.stat", side_effect=perm_err):
            sig = project_io.source_signature("locked.mkv")
            self.assertIsNone(sig)

            bundleable = project_io.project_asset_is_bundleable(Path("locked.mkv"))
            self.assertFalse(bundleable)


if __name__ == "__main__":
    unittest.main()
