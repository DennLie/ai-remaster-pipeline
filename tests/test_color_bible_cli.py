"""Unit and integration tests for the Color Bible CLI and compliance integration."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from ai_remaster_gui.color_bible_service import ColorBibleService
from ai_remaster_gui.compliance import (
    append_compliance_record,
    load_compliance_report,
    resolve_color_bible_version,
)
from scripts.color_bible import build_parser, main as cli_main, parse_ranges

CLI_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "color_bible.py"


class ColorBibleCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)
        self.db_path = self.tmp_dir / "test_color_bible.db"

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def _run_cli(self, args: list[str]) -> tuple[int, str, str]:
        """Run CLI in-process while capturing stdout and stderr."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = stdout_buf
            sys.stderr = stderr_buf
            code = cli_main(args)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return code, stdout_buf.getvalue(), stderr_buf.getvalue()

    def _create_synthetic_image(
        self,
        path: Path,
        width: int = 120,
        height: int = 80,
        color_left: tuple[int, int, int] = (255, 0, 0),
        color_right: tuple[int, int, int] = (0, 0, 255),
    ) -> Path:
        """Create a split-color synthetic PNG image."""
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, : width // 2] = color_left
        arr[:, width // 2 :] = color_right
        img = Image.fromarray(arr, mode="RGB")
        img.save(path)
        return path

    def _create_synthetic_mask(
        self,
        path: Path,
        width: int = 120,
        height: int = 80,
    ) -> Path:
        """Create a binary mask selecting only the left half."""
        arr = np.zeros((height, width), dtype=np.uint8)
        arr[:, : width // 2] = 255
        img = Image.fromarray(arr, mode="L")
        img.save(path)
        return path

    def _create_sample_manifest(self, path: Path) -> Path:
        """Create a synthetic shot manifest CSV."""
        rows = [
            {
                "shot_index": 1,
                "start_time": "00:00:00.000",
                "end_time": "00:00:02.500",
                "source_reference": "shot_001.png",
                "qc_status": "approved",
            },
            {
                "shot_index": 2,
                "start_time": "00:00:02.500",
                "end_time": "00:00:05.000",
                "source_reference": "shot_002.png",
                "qc_status": "approved",
            },
            {
                "shot_index": 15,
                "start_time": "00:00:30.000",
                "end_time": "00:00:35.000",
                "source_reference": "shot_015.png",
                "qc_status": "approved",
            },
            {
                "shot_index": 420,
                "start_time": "00:15:00.000",
                "end_time": "00:15:05.000",
                "source_reference": "shot_420.png",
                "qc_status": "approved",
            },
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return path

    # --- Unit Tests ---

    def test_parse_ranges_helper(self) -> None:
        """Verify parse_ranges parses multi-range, single shot, and invalid formats."""
        self.assertEqual(parse_ranges("12-18,410-435"), [(12, 18), (410, 435)])
        self.assertEqual(parse_ranges("5"), [(5, 5)])
        self.assertEqual(parse_ranges("1-5, 10, 20-25"), [(1, 5), (10, 10), (20, 25)])

        with self.assertRaises(ValueError):
            parse_ranges("")
        with self.assertRaises(ValueError):
            parse_ranges("18-12")
        with self.assertRaises(ValueError):
            parse_ranges("invalid-range")

    def test_init_db(self) -> None:
        """Test init-db with custom --db path and verify schema initialization."""
        code, stdout, _stderr = self._run_cli(["init-db", "--db", str(self.db_path)])
        self.assertEqual(code, 0)
        self.assertTrue(self.db_path.is_file())
        self.assertIn(str(self.db_path), stdout)

        # Idempotent re-initialization
        code2, stdout2, _ = self._run_cli(["--db", str(self.db_path), "init-db"])
        self.assertEqual(code2, 0)
        self.assertIn("Initialized Color Bible database at", stdout2)

    def test_sync_manifest(self) -> None:
        """Test sync-manifest importing shots into the database."""
        manifest_file = self.tmp_dir / "test_manifest.csv"
        self._create_sample_manifest(manifest_file)

        code, stdout, _ = self._run_cli([
            "sync-manifest",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--manifest", str(manifest_file),
            "--title", "CLI Test Feature",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Synced 4 shot(s)", stdout)

        # Verify shots in service
        with ColorBibleService(db_path=self.db_path) as service:
            shots = service.list_shots("film_cli_test")
            self.assertEqual(len(shots), 4)
            shot_indices = {s.shot_index for s in shots}
            self.assertEqual(shot_indices, {1, 2, 15, 420})

    def test_characters_variants_and_assign_ranges(self) -> None:
        """Test add-character, add-variant, assign-range with non-contiguous intervals, and list-assignments."""
        # 1. Add character
        code, stdout, _ = self._run_cli([
            "add-character",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--name", "Elena",
            "--notes", "Lead investigator",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Created character: char_", stdout)
        char_id = stdout.strip().split()[-1]

        # 2. Add variant
        code, stdout, _ = self._run_cli([
            "add-variant",
            "--db", str(self.db_path),
            "--character-id", char_id,
            "--costume-id", "costume_trenchcoat",
            "--hair-profile", "auburn wavy",
            "--skin-reference", "fair",
            "--lighting-context", "exterior rainy day",
            "--notes", "Wool trench",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Created variant: var_", stdout)
        var_id = stdout.strip().split()[-1]

        # 3. Assign non-contiguous ranges: 12-18 and 410-435
        code, stdout, _ = self._run_cli([
            "assign-range",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--entity-type", "character",
            "--entity-id", char_id,
            "--variant-id", var_id,
            "--ranges", "12-18,410-435",
            "--confidence", "0.95",
            "--approval-status", "approved",
            "--provenance", "manual",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Created 2 assignment(s)", stdout)

        # 4. Assign single shot number: 5
        code, stdout, _ = self._run_cli([
            "assign-range",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--entity-type", "character",
            "--entity-id", char_id,
            "--ranges", "5",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Created 1 assignment(s)", stdout)

        # 5. List assignments (plain text)
        code, stdout, _ = self._run_cli([
            "list-assignments",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Assignments for film 'film_cli_test': (3 total)", stdout)
        self.assertIn("12-18", stdout)
        self.assertIn("410-435", stdout)
        self.assertIn("5-5", stdout)

        # 6. List assignments with --json
        code, stdout, _ = self._run_cli([
            "list-assignments",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--json",
        ])
        self.assertEqual(code, 0)
        data = json.loads(stdout)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 3)

        # 7. List active assignments for shot index in range (15)
        code, stdout, _ = self._run_cli([
            "list-assignments",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--shot-index", "15",
            "--json",
        ])
        self.assertEqual(code, 0)
        data_shot15 = json.loads(stdout)
        self.assertEqual(len(data_shot15), 1)
        self.assertEqual(data_shot15[0]["entity_id"], char_id)
        self.assertEqual(data_shot15[0]["variant_id"], var_id)

        # 8. List active assignments for shot index outside ranges (99)
        code, stdout, _ = self._run_cli([
            "list-assignments",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--shot-index", "99",
            "--json",
        ])
        self.assertEqual(code, 0)
        data_shot99 = json.loads(stdout)
        self.assertEqual(len(data_shot99), 0)

    def test_extract_palette(self) -> None:
        """Test extract-palette on synthetic image with and without mask, in text and JSON modes."""
        img_path = self.tmp_dir / "sample_keyframe.png"
        mask_path = self.tmp_dir / "sample_mask.png"
        self._create_synthetic_image(img_path, color_left=(255, 0, 0), color_right=(0, 255, 0))
        self._create_synthetic_mask(mask_path)

        # 1. Text mode without mask
        code, stdout, _ = self._run_cli([
            "extract-palette",
            "--image", str(img_path),
            "--num-colors", "4",
        ])
        self.assertEqual(code, 0)
        self.assertIn("CIE-LAB Percentiles", stdout)
        self.assertIn("Dominant RGB Palette", stdout)

        # 2. JSON mode with mask (should isolate only the red side)
        code, stdout, _ = self._run_cli([
            "extract-palette",
            "--image", str(img_path),
            "--mask", str(mask_path),
            "--num-colors", "2",
            "--json",
        ])
        self.assertEqual(code, 0)
        data = json.loads(stdout)
        self.assertIn("lab_percentiles", data)
        self.assertIn("rgb_palette", data)
        self.assertIn("L", data["lab_percentiles"])
        self.assertEqual(len(data["rgb_palette"]), 1)
        # Red side dominant swatch
        self.assertEqual(data["rgb_palette"][0]["hex"].lower(), "#ff0000")

    def test_register_reference_approve_reject_and_snapshot_hash(self) -> None:
        """Test register-reference, approve-ref, reject-ref, and snapshot-hash generation."""
        img_path = self.tmp_dir / "ref_keyframe.png"
        self._create_synthetic_image(img_path)

        # Create character first
        with ColorBibleService(db_path=self.db_path) as service:
            char = service.create_character("film_cli_test", "David")
            char_id = char.character_id
            service.assign_entity_ranges(
                film_id="film_cli_test",
                entity_type="character",
                entity_id=char_id,
                ranges=[(1, 10)],
            )

        # 1. Register reference (draft by default)
        code, stdout, _ = self._run_cli([
            "register-reference",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
            "--entity-type", "character",
            "--entity-id", char_id,
            "--keyframe", str(img_path),
            "--provenance", "manual",
        ])
        self.assertEqual(code, 0)
        self.assertIn("Created reference: ref_", stdout)
        ref_id = stdout.strip().split()[-1]

        # 2. Snapshot hash with no approved reference yet
        code, stdout_pre, _ = self._run_cli([
            "snapshot-hash",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
        ])
        self.assertEqual(code, 0)
        hash_pre = stdout_pre.strip()
        self.assertTrue(hash_pre.startswith("sha256:"))

        # 3. Approve reference
        code, stdout, _ = self._run_cli([
            "approve-ref",
            "--db", str(self.db_path),
            "--ref-id", ref_id,
            "--approved-by", "lead_colorist",
        ])
        self.assertEqual(code, 0)
        self.assertIn(f"Approved reference: {ref_id}", stdout)

        # 4. Snapshot hash should now reflect the approved reference
        code, stdout_post, _ = self._run_cli([
            "snapshot-hash",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
        ])
        self.assertEqual(code, 0)
        hash_post = stdout_post.strip()
        self.assertTrue(hash_post.startswith("sha256:"))
        self.assertNotEqual(hash_pre, hash_post)

        # 5. Reject reference
        code, stdout, _ = self._run_cli([
            "reject-ref",
            "--db", str(self.db_path),
            "--ref-id", ref_id,
        ])
        self.assertEqual(code, 0)
        self.assertIn(f"Rejected reference: {ref_id}", stdout)

        # 6. Snapshot hash after rejection reverts to excluding the reference
        code, stdout_rej, _ = self._run_cli([
            "snapshot-hash",
            "--db", str(self.db_path),
            "--film-id", "film_cli_test",
        ])
        self.assertEqual(code, 0)
        self.assertEqual(stdout_rej.strip(), hash_pre)

    def test_compliance_report_integration(self) -> None:
        """Verify compliance.append_compliance_record automatically resolves color_bible_version."""
        img_path = self.tmp_dir / "comp_keyframe.png"
        self._create_synthetic_image(img_path)

        # Setup Color Bible DB with approved reference
        film_id = "film_compliance_test"
        with ColorBibleService(db_path=self.db_path) as service:
            char = service.create_character(film_id, "Maria")
            service.assign_entity_ranges(film_id, "character", char.character_id, [(1, 20)])
            service.register_reference_image(
                film_id=film_id,
                entity_type="character",
                entity_id=char.character_id,
                keyframe_path=img_path,
                auto_approve=True,
            )
            expected_snapshot_hash = service.get_snapshot_hash(film_id)

        # 1. Automatic resolution via settings
        report_dir = self.tmp_dir / "compliance_out"
        settings = {
            "global": {
                "commercial_model_guard": "true",
                "color_bible_film_id": film_id,
                "color_bible_db": str(self.db_path),
            }
        }
        report_file = append_compliance_record(
            output_dir=report_dir,
            stage="colorize",
            models=[{"model_id": "ddcolor"}],
            settings=settings,
            color_bible_version=None,  # triggers auto-resolution
        )
        self.assertTrue(report_file.is_file())

        report_data = load_compliance_report(report_dir)
        self.assertIsNotNone(report_data)
        run = report_data["runs"][0]
        self.assertEqual(run["color_bible_version"], expected_snapshot_hash)

        # 2. Explicit override takes precedence
        append_compliance_record(
            output_dir=report_dir,
            stage="upscale",
            models=[{"model_id": "ddcolor"}],
            settings=settings,
            color_bible_version="explicit_version_tag",
        )
        report_data2 = load_compliance_report(report_dir)
        self.assertEqual(len(report_data2["runs"]), 2)
        self.assertEqual(report_data2["runs"][1]["color_bible_version"], "explicit_version_tag")

        # 3. Clean fallback to None when DB doesn't exist
        non_existent_db = self.tmp_dir / "does_not_exist.db"
        version_resolved = resolve_color_bible_version(
            film_id="any_film",
            db_path=non_existent_db,
        )
        self.assertIsNone(version_resolved)

    def test_subprocess_invocation(self) -> None:
        """Verify running scripts/color_bible.py via subprocess.run succeeds on Windows."""
        res = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Color Bible CLI", res.stdout)
        self.assertIn("init-db", res.stdout)
        self.assertIn("assign-range", res.stdout)

        # Subprocess error on missing required args
        res_err = subprocess.run(
            [sys.executable, str(CLI_SCRIPT), "sync-manifest"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(res_err.returncode, 0)

    def test_parse_ranges_whitespace_and_edge_cases(self) -> None:
        """Verify parse_ranges handles whitespace, zeros, trailing commas, and malformed inputs."""
        # Whitespace and zeros
        self.assertEqual(parse_ranges("  12 - 18 ,  410 - 435  "), [(12, 18), (410, 435)])
        self.assertEqual(parse_ranges("0-10"), [(0, 10)])
        self.assertEqual(parse_ranges("1-5, 10-15,"), [(1, 5), (10, 15)])
        self.assertEqual(parse_ranges(" 42 "), [(42, 42)])

        # Malformed strings
        with self.assertRaises(ValueError):
            parse_ranges("1-2-3")
        with self.assertRaises(ValueError):
            parse_ranges("abc-def")
        with self.assertRaises(ValueError):
            parse_ranges("5-3")
        with self.assertRaises(ValueError):
            parse_ranges("   ")
        with self.assertRaises(ValueError):
            parse_ranges(",,")

    def test_cli_error_paths(self) -> None:
        """Verify CLI error paths for invalid commands and missing files."""
        # Invalid subcommand
        code, _, stderr = self._run_cli(["unknown-command"])
        self.assertNotEqual(code, 0)

        # Missing manifest file in sync-manifest
        code, _, stderr = self._run_cli([
            "sync-manifest",
            "--db", str(self.db_path),
            "--film-id", "film_err",
            "--manifest", str(self.tmp_dir / "does_not_exist.csv"),
        ])
        self.assertNotEqual(code, 0)

        # Missing keyframe image in extract-palette
        code, _, stderr = self._run_cli([
            "extract-palette",
            "--image", str(self.tmp_dir / "does_not_exist.png"),
        ])
        self.assertNotEqual(code, 0)

        # Missing keyframe image in register-reference
        code, _, stderr = self._run_cli([
            "register-reference",
            "--db", str(self.db_path),
            "--film-id", "film_err",
            "--entity-type", "character",
            "--entity-id", "char_none",
            "--keyframe", str(self.tmp_dir / "does_not_exist.png"),
        ])
        self.assertNotEqual(code, 0)

    def test_cli_invalid_range_arguments(self) -> None:
        """Verify CLI error paths for inverted and malformed range inputs."""
        # Create character first
        code, stdout, _ = self._run_cli([
            "add-character",
            "--db", str(self.db_path),
            "--film-id", "film_cli_ranges",
            "--name", "Frank",
        ])
        self.assertEqual(code, 0)
        char_id = stdout.strip().split()[-1]

        # Inverted range (start > end)
        code_inv, _, stderr_inv = self._run_cli([
            "assign-range",
            "--db", str(self.db_path),
            "--film-id", "film_cli_ranges",
            "--entity-type", "character",
            "--entity-id", char_id,
            "--ranges", "20-10",
        ])
        self.assertNotEqual(code_inv, 0)
        self.assertIn("[ERROR]", stderr_inv)
        self.assertIn("start (20) > end (10)", stderr_inv)

        # Malformed range
        code_mal, _, stderr_mal = self._run_cli([
            "assign-range",
            "--db", str(self.db_path),
            "--film-id", "film_cli_ranges",
            "--entity-type", "character",
            "--entity-id", char_id,
            "--ranges", "invalid-range",
        ])
        self.assertNotEqual(code_mal, 0)
        self.assertIn("[ERROR]", stderr_mal)

    def test_cli_nonexistent_reference_approval_rejection(self) -> None:
        """Verify approve-ref and reject-ref exit with error code when reference does not exist."""
        code_app, _, stderr_app = self._run_cli([
            "approve-ref",
            "--db", str(self.db_path),
            "--ref-id", "ref_nonexistent",
        ])
        self.assertNotEqual(code_app, 0)
        self.assertIn("[ERROR]", stderr_app)
        self.assertIn("not found", stderr_app)

        code_rej, _, stderr_rej = self._run_cli([
            "reject-ref",
            "--db", str(self.db_path),
            "--ref-id", "ref_nonexistent",
        ])
        self.assertNotEqual(code_rej, 0)
        self.assertIn("[ERROR]", stderr_rej)
        self.assertIn("not found", stderr_rej)

    def test_cli_empty_database_snapshot_hash(self) -> None:
        """Verify snapshot-hash outputs valid sha256: hash even for empty database."""
        code, stdout, _ = self._run_cli([
            "snapshot-hash",
            "--db", str(self.db_path),
            "--film-id", "film_empty_hash",
        ])
        self.assertEqual(code, 0)
        hash_val = stdout.strip()
        self.assertTrue(hash_val.startswith("sha256:"))
        self.assertEqual(len(hash_val), 7 + 64)




if __name__ == "__main__":
    unittest.main()
