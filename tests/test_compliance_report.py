"""GPU-free tests for the Compliance Reporting Engine and CLI."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ai_remaster_gui.compliance import (
    DEFAULT_EU_AI_ACT_DISCLOSURE,
    append_compliance_record,
    determine_guard_mode,
    evaluate_overall_compliance,
    get_eu_ai_act_disclosure,
    is_run_compliant,
    load_compliance_report,
    normalize_model_record,
    validate_compliance_report,
)
from scripts.compliance_report import main as cli_main


class ComplianceReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir_obj.name)

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_single_run_recording_and_schema(self):
        settings = {
            "global": {
                "commercial_model_guard": "true",
                "non_commercial_benchmark_override": "false",
            }
        }
        report_path = append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor", "component_type": "checkpoint"}],
            settings=settings,
            seeds={"reference_seed": 42},
            color_bible_version="v1.0.0",
            conditional_attestations={},
        )

        self.assertTrue(report_path.is_file())
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertTrue(data["overall_commercial_compliant"])
        self.assertIn("report_id", data)
        self.assertIn("created_at", data)
        self.assertIn("updated_at", data)
        self.assertEqual(len(data["runs"]), 1)

        run = data["runs"][0]
        self.assertEqual(run["stage"], "references")
        self.assertEqual(run["guard_mode"], "strict")
        self.assertTrue(run["is_commercial_compliant"])
        self.assertEqual(run["seeds"], {"reference_seed": 42})
        self.assertEqual(run["color_bible_version"], "v1.0.0")
        self.assertEqual(run["conditional_attestations"], {})

        # EU AI Act Article 50 Disclosure validation
        disc = run["eu_ai_act_article_50_disclosure"]
        self.assertEqual(disc["role"], "deployer")
        self.assertTrue(disc["ai_generated_or_manipulated"])
        self.assertFalse(disc["machine_readable_watermark_or_metadata"])
        self.assertEqual(disc["compliance_effective_date"], "2026-08-02")
        self.assertEqual(disc["applicable_articles"], ["Art. 50(4)"])
        self.assertIn("EU AI Act Article 50(4)", disc["disclosure_notice"])

        # Model entry validation
        self.assertEqual(len(run["models"]), 1)
        model = run["models"][0]
        self.assertEqual(model["model_id"], "ddcolor")
        self.assertEqual(model["component_type"], "checkpoint")
        self.assertEqual(model["license"], "Apache-2.0")
        self.assertEqual(model["status"], "allowed_osi")
        self.assertEqual(model["source_url"], "https://github.com/piddnad/DDColor")
        self.assertIsNone(model["file_path"])
        self.assertIsNone(model["sha256"])
        self.assertTrue(len(model["reason"]) > 0)

    def test_seeds_and_nullable_color_bible_version(self):
        settings = {"global": {"commercial_model_guard": "true"}}

        # Test with multiple seeds and None for color_bible_version (nullable)
        append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor"}],
            settings=settings,
            seeds={"reference_seed": 42, "color_seed": 100, "denoise_seed": 777},
            color_bible_version=None,
        )

        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        run = data["runs"][0]
        self.assertEqual(run["seeds"], {"reference_seed": 42, "color_seed": 100, "denoise_seed": 777})
        self.assertIsNone(run["color_bible_version"])

    def test_eu_ai_act_custom_settings_override(self):
        settings = {
            "global": {
                "commercial_model_guard": "true",
                "eu_ai_act_role": "provider",
                "machine_readable_watermark": "true",
                "disclosure_notice": "Custom deployer Art 50 notice",
            }
        }
        append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor"}],
            settings=settings,
        )
        data = load_compliance_report(self.output_dir)
        disc = data["runs"][0]["eu_ai_act_article_50_disclosure"]
        self.assertEqual(disc["role"], "provider")
        self.assertTrue(disc["machine_readable_watermark_or_metadata"])
        self.assertEqual(disc["disclosure_notice"], "Custom deployer Art 50 notice")
        self.assertEqual(disc["compliance_effective_date"], "2026-08-02")
        self.assertEqual(disc["applicable_articles"], ["Art. 50(4)"])

    def test_multi_stage_cumulative_execution_history(self):
        settings = {
            "global": {
                "commercial_model_guard": "true",
                "non_commercial_benchmark_override": "false",
            }
        }

        # Stage 1: References (DDColor, compliant)
        append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor"}],
            settings=settings,
            seeds={"reference_seed": 101},
            color_bible_version="v1.2",
        )
        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["runs"]), 1)
        self.assertTrue(data["overall_commercial_compliant"])
        snapshot_stage_1 = dict(data["runs"][0])

        # Stage 2: Colour (DeOldify, compliant)
        append_compliance_record(
            output_dir=self.output_dir,
            stage="colour",
            models=[{"model_id": "deoldify"}],
            settings=settings,
            seeds={"colour_seed": 202},
            color_bible_version="v1.2",
        )
        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["runs"]), 2)
        # Adding stage 2 does not erase or mutate stage 1
        self.assertEqual(data["runs"][0], snapshot_stage_1)
        self.assertEqual(data["runs"][0]["stage"], "references")
        self.assertEqual(data["runs"][1]["stage"], "colour")
        self.assertTrue(data["overall_commercial_compliant"])
        snapshot_stage_2 = dict(data["runs"][1])

        # Stage 3: Cleanup (ProPainter, hard_blocked)
        append_compliance_record(
            output_dir=self.output_dir,
            stage="cleanup",
            models=[{"model_id": "propainter"}],
            settings=settings,
        )
        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["runs"]), 3)
        self.assertFalse(data["runs"][2]["is_commercial_compliant"])
        self.assertEqual(data["runs"][2]["models"][0]["status"], "hard_blocked")
        # Top-level overall_commercial_compliant flips to False!
        self.assertFalse(data["overall_commercial_compliant"])

        # Stage 4: Upscale (DDColor, compliant)
        append_compliance_record(
            output_dir=self.output_dir,
            stage="upscale",
            models=[{"model_id": "ddcolor"}],
            settings=settings,
        )
        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        self.assertEqual(len(data["runs"]), 4)
        self.assertTrue(data["runs"][3]["is_commercial_compliant"])
        # Overall compliance must STAY False because a prior run was non-compliant!
        self.assertFalse(data["overall_commercial_compliant"])

    def test_benchmark_override_flips_compliance(self):
        settings_override = {
            "global": {
                "commercial_model_guard": "true",
                "non_commercial_benchmark_override": "true",
            }
        }
        # Even with an allowed model, non_commercial_benchmark_override makes the run non-compliant
        append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor"}],
            settings=settings_override,
        )
        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        run = data["runs"][0]
        self.assertEqual(run["guard_mode"], "benchmark_override")
        self.assertFalse(run["is_commercial_compliant"])
        self.assertFalse(data["overall_commercial_compliant"])

    def test_sha256_computation_for_local_weights_file(self):
        weights_file = self.output_dir / "test_model_weights.pth"
        dummy_content = b"DDColor checkpoint test weights binary payload 123456"
        weights_file.write_bytes(dummy_content)
        expected_hash = hashlib.sha256(dummy_content).hexdigest()

        append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor", "file_path": str(weights_file)}],
            settings={"global": {"commercial_model_guard": "true"}},
        )

        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        model_rec = data["runs"][0]["models"][0]
        self.assertEqual(model_rec["file_path"], str(weights_file))
        self.assertEqual(model_rec["sha256"], expected_hash)

    def test_conditional_attestations_gating(self):
        # 1. Without revenue attestation: LTX is blocked
        append_compliance_record(
            output_dir=self.output_dir,
            stage="colour",
            models=[{"model_id": "ltx"}],
            settings={"global": {"commercial_model_guard": "true"}},
        )
        data = load_compliance_report(self.output_dir)
        self.assertIsNotNone(data)
        run1 = data["runs"][0]
        self.assertEqual(run1["models"][0]["status"], "conditional_blocked")
        self.assertFalse(run1["is_commercial_compliant"])
        self.assertFalse(data["overall_commercial_compliant"])

        # 2. In a separate dir with attestations: LTX is cleared
        cleared_dir = self.output_dir / "cleared_run"
        attestations = {
            "company_revenue_under_10m_usd": True,
            "ltx_components_audited": True,
        }
        append_compliance_record(
            output_dir=cleared_dir,
            stage="colour",
            models=[{"model_id": "ltx"}],
            settings={"global": {"commercial_model_guard": "true"}},
            conditional_attestations=attestations,
        )
        cleared_data = load_compliance_report(cleared_dir)
        self.assertIsNotNone(cleared_data)
        run2 = cleared_data["runs"][0]
        self.assertEqual(run2["models"][0]["status"], "conditional_cleared")
        self.assertTrue(run2["is_commercial_compliant"])
        self.assertTrue(cleared_data["overall_commercial_compliant"])

    def test_report_validation(self):
        append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor"}],
            settings={"global": {"commercial_model_guard": "true"}},
        )
        valid, errors = validate_compliance_report(self.output_dir)
        self.assertTrue(valid)
        self.assertEqual(errors, [])

        # Tampered data test
        report_file = self.output_dir / "compliance_report.json"
        with open(report_file, "r", encoding="utf-8") as f:
            corrupt = json.load(f)
        del corrupt["schema_version"]
        corrupt["overall_commercial_compliant"] = False  # Mismatch with run

        valid, errors = validate_compliance_report(corrupt)
        self.assertFalse(valid)
        self.assertTrue(len(errors) >= 2)

    def test_cli_subcommands(self):
        # 1. Generate via CLI
        code = cli_main([
            "generate",
            "--output-dir", str(self.output_dir),
            "--stage", "references",
            "--model-id", "ddcolor",
            "--seed", "seed_1=42",
            "--color-bible-version", "v1.0",
        ])
        self.assertEqual(code, 0)
        report_file = self.output_dir / "compliance_report.json"
        self.assertTrue(report_file.is_file())

        # 2. Inspect via CLI
        code = cli_main(["inspect", str(self.output_dir)])
        self.assertEqual(code, 0)

        # 3. Inspect JSON via CLI
        code = cli_main(["inspect", str(self.output_dir), "--json"])
        self.assertEqual(code, 0)

        # 4. Validate via CLI
        code = cli_main(["validate", str(self.output_dir)])
        self.assertEqual(code, 0)

        # 5. Root shortcuts
        code = cli_main(["--validate", str(self.output_dir)])
        self.assertEqual(code, 0)
        code = cli_main(["--inspect", str(self.output_dir)])
        self.assertEqual(code, 0)

    def test_cli_strict_and_no_strict_flags(self):
        # Create a report with a non-commercial run
        nc_dir = self.output_dir / "nc_report"
        settings_override = {
            "global": {
                "commercial_model_guard": "true",
                "non_commercial_benchmark_override": "true",
            }
        }
        append_compliance_record(
            output_dir=nc_dir,
            stage="references",
            models=[{"model_id": "ddcolor"}],
            settings=settings_override,
        )

        # Default validate has strict=True -> exit code 1
        code_default = cli_main(["validate", str(nc_dir), "--quiet"])
        self.assertEqual(code_default, 1)

        # Explicit --strict -> exit code 1
        code_strict = cli_main(["validate", str(nc_dir), "--strict", "--quiet"])
        self.assertEqual(code_strict, 1)

        # Explicit --no-strict -> exit code 0
        code_no_strict = cli_main(["validate", str(nc_dir), "--no-strict", "--quiet"])
        self.assertEqual(code_no_strict, 0)

        # Root flag --no-strict -> exit code 0
        code_root_no_strict = cli_main(["--validate", str(nc_dir), "--no-strict", "--quiet"])
        self.assertEqual(code_root_no_strict, 0)

        # Root flag --strict -> exit code 1
        code_root_strict = cli_main(["--validate", str(nc_dir), "--strict", "--quiet"])
        self.assertEqual(code_root_strict, 1)


if __name__ == "__main__":
    unittest.main()
