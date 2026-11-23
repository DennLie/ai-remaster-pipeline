"""GPU-free tests for the commercial model policy and registry engine."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ai_remaster_gui.commercial_policy import (
    CommercialPolicyError,
    PolicyDecision,
    compute_sha256,
    enabled,
    enforce_component,
    evaluate_component,
    load_policy_registry,
    pipeline_issues,
    stage_issue,
)
from ai_remaster_gui.config import ROOT
from ai_remaster_gui.runtime_settings import base_settings, normalize_settings


class CommercialModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = base_settings()

    def guard(self):
        self.settings["global"]["commercial_model_guard"] = "true"

    def test_default_on_and_override_behavior(self):
        # Commercial Guard defaults to ON (True)
        self.assertTrue(enabled(self.settings))
        # Popping commercial_model_guard still defaults to ON
        self.settings["global"].pop("commercial_model_guard")
        self.assertTrue(enabled(self.settings))

        # Explicit commercial_model_guard="false" turns it off
        self.settings["global"]["commercial_model_guard"] = "false"
        self.assertFalse(enabled(self.settings))

        # non_commercial_benchmark_override="true" overrides guard and disables enforcement
        self.settings["global"]["commercial_model_guard"] = "true"
        self.settings["global"]["non_commercial_benchmark_override"] = "true"
        self.assertFalse(enabled(self.settings))
        self.assertEqual(stage_issue(self.settings, "cleanup"), "")

    def test_normalize_settings_preserves_saved_values_while_applying_defaults(self):
        cfg = base_settings()
        cfg["global"].pop("commercial_model_guard")
        cfg["global"].pop("non_commercial_benchmark_override")
        normalized = normalize_settings(cfg)
        self.assertEqual(normalized["global"]["commercial_model_guard"], "true")
        self.assertEqual(normalized["global"]["non_commercial_benchmark_override"], "false")

        saved_cfg = base_settings()
        saved_cfg["global"]["commercial_model_guard"] = "false"
        saved_cfg["global"]["non_commercial_benchmark_override"] = "true"
        normalized_saved = normalize_settings(saved_cfg)
        self.assertEqual(normalized_saved["global"]["commercial_model_guard"], "false")
        self.assertEqual(normalized_saved["global"]["non_commercial_benchmark_override"], "true")

    def test_cleanup_blocks_both_restricted_models(self):
        self.guard()
        self.settings["cleanup"]["ai_descratch"] = "true"
        reason = stage_issue(self.settings, "cleanup")
        self.assertIn("ProPainter", reason)
        self.assertIn("Dearchive", reason)

    def test_devignette_alone_is_not_blocked(self):
        self.guard()
        self.settings["cleanup"].update({"ai_descratch": "false", "dearchive": "false", "devignette": "true"})
        self.assertEqual(stage_issue(self.settings, "cleanup"), "")

    def test_outpainting_fails_closed(self):
        self.guard()
        for model in ("h3", "oumoumad", "official", "ltx25", "wanvace", "future-model"):
            with self.subTest(model=model):
                self.settings["outpaint"]["outpaint_model"] = model
                self.assertTrue(stage_issue(self.settings, "outpaint"))

    def test_reference_override_fails_closed(self):
        self.guard()
        for model in ("qwen", "qwen21", "openai", "future-model"):
            with self.subTest(model=model):
                self.assertTrue(stage_issue(self.settings, "references", model_override=model))
        self.assertIn("research-only", stage_issue(self.settings, "references", model_override="qwen21"))

    def test_color_backends_fail_closed(self):
        self.guard()
        for model in ("deepexemplar", "colormnet", "cmnet2", "both", "ltx", "openai", "future-model"):
            with self.subTest(model=model):
                self.settings["colour"]["method"] = model
                self.assertTrue(stage_issue(self.settings, "colour"))

        # osi_masked_lab passes under commercial guard
        self.settings["colour"]["method"] = "osi_masked_lab"
        self.assertEqual(stage_issue(self.settings, "colour"), "")
        self.settings["colour"]["method"] = "osi-masked-lab"
        self.assertEqual(stage_issue(self.settings, "colour"), "")

    def test_osi_masked_lab_commercial_clearance(self):
        # 1. osi_masked_lab evaluates as allowed_osi with Apache-2.0 license
        decision = evaluate_component("osi_masked_lab")
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.status, "allowed_osi")
        self.assertEqual(decision.license, "Apache-2.0")
        self.assertIn("Apache-2.0", decision.reason)

        decision_alias = evaluate_component("osi-masked-lab")
        self.assertTrue(decision_alias.allowed)
        self.assertEqual(decision_alias.status, "allowed_osi")
        self.assertEqual(decision_alias.license, "Apache-2.0")

        # 2. In Commercial Mode (guard()), stage_issue returns "" (not blocked)
        self.guard()
        self.assertEqual(stage_issue(self.settings, "colour", model_override="osi_masked_lab"), "")
        self.assertEqual(stage_issue(self.settings, "colour", model_override="osi-masked-lab"), "")

        # 3. cmnet2, colormnet, and both remain strictly blocked
        for blocked in ("cmnet2", "colormnet", "both"):
            issue = stage_issue(self.settings, "colour", model_override=blocked)
            self.assertTrue(issue)
            self.assertIn("ColorMNet", issue)

        # enforce_component allows osi_masked_lab and blocks cmnet2/colormnet
        self.assertTrue(enforce_component("osi_masked_lab").allowed)
        with self.assertRaises(CommercialPolicyError):
            enforce_component("cmnet2")
        with self.assertRaises(CommercialPolicyError):
            enforce_component("colormnet")

    def test_utility_stages_are_not_blocked(self):
        self.guard()
        for stage in ("stabilize", "shots", "recomp", "output"):
            self.assertEqual(stage_issue(self.settings, stage), "")

    def test_audio_and_upscale_await_audit(self):
        self.guard()
        self.assertTrue(stage_issue(self.settings, "audio"))
        for model in ("flashvsr", "seedvr2", "ltx25", "ltx25cq"):
            self.settings["upscale"]["method"] = model
            self.assertTrue(stage_issue(self.settings, "upscale"))

    def test_whole_pipeline_preflight(self):
        self.guard()
        problems = pipeline_issues(self.settings, ["cleanup", "stabilize", "references", "colour"])
        self.assertEqual(len(problems), 3)
        self.assertIn("cleanup", problems[0])
        self.assertIn("references", problems[1])
        self.assertIn("colour", problems[2])

    def test_allowed_osi_models(self):
        for cid, expected_lic in (
            ("ddcolor", "Apache-2.0"),
            ("deoldify", "MIT"),
            ("osi_masked_lab", "Apache-2.0"),
            ("osi-masked-lab", "Apache-2.0"),
            ("qwen3-vl", "Apache-2.0"),
            ("sam2", "Apache-2.0"),
            ("sam2.1", "Apache-2.0"),
            ("bytetrack", "MIT"),
            ("pyscenedetect", "BSD-3-Clause"),
        ):
            with self.subTest(component_id=cid):
                decision = evaluate_component(cid)
                self.assertTrue(decision.allowed, f"{cid} should be allowed")
                self.assertEqual(decision.status, "allowed_osi")
                self.assertEqual(decision.license, expected_lic)

        # DDColor in references stage is not blocked
        self.assertEqual(stage_issue(self.settings, "references", model_override="ddcolor"), "")
        # DeOldify in colour stage is not blocked
        self.assertEqual(stage_issue(self.settings, "colour", model_override="deoldify"), "")
        # OSI Masked LAB in colour stage is not blocked
        self.assertEqual(stage_issue(self.settings, "colour", model_override="osi_masked_lab"), "")

        # Qwen-2511 requires audited SHA-256 or audit flag
        qwen_decision = evaluate_component("qwen-image-edit-2511")
        self.assertFalse(qwen_decision.allowed)
        self.assertEqual(qwen_decision.status, "un-audited")

        qwen_audited = evaluate_component("qwen-image-edit-2511", conditional_attestations={"qwen_2511_audited": True})
        self.assertTrue(qwen_audited.allowed)
        self.assertEqual(qwen_audited.status, "allowed_osi")

    def test_hard_block_commercial_models(self):
        hard_blocked = [
            "cmnet2",
            "colormnet",
            "qwen-image-2.1",
            "qwen21",
            "propainter",
            "h3",
            "minimax-h3",
            "sam3.1",
            "sam-3.1",
            "sam3",
        ]
        for cid in hard_blocked:
            with self.subTest(component_id=cid):
                # Even if conditional flags are passed, hard-blocked models must stay blocked
                decision = evaluate_component(
                    cid,
                    conditional_attestations={"company_revenue_under_10m_usd": True, "weight_vgg19_audited": True},
                )
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.status, "hard_blocked")
                with self.assertRaises(CommercialPolicyError):
                    enforce_component(cid)

        # Verify stage_issue blocks each hard-blocked model in relevant stages
        self.guard()
        # CMNET2 and ColorMNet in colour
        for m in ("cmnet2", "colormnet", "both"):
            self.assertTrue(stage_issue(self.settings, "colour", model_override=m))
            self.assertIn("ColorMNet", stage_issue(self.settings, "colour", model_override=m))
        # Qwen-2.1 in references
        self.assertTrue(stage_issue(self.settings, "references", model_override="qwen21"))
        self.assertIn("research-only", stage_issue(self.settings, "references", model_override="qwen21"))
        # ProPainter in cleanup and dust
        self.settings["cleanup"]["ai_descratch"] = "true"
        self.assertIn("ProPainter", stage_issue(self.settings, "cleanup"))
        self.settings["cleanup"]["ai_descratch"] = "false"
        self.assertIn("ProPainter", stage_issue(self.settings, "dust", model_override="propainter"))
        # MiniMax H3 in outpaint
        self.assertIn("MiniMax H3", stage_issue(self.settings, "outpaint", model_override="h3"))

    def test_conditional_commercial_ltx(self):
        # Default fails without revenue attestation and audited components
        default_dec = evaluate_component("ltx")
        self.assertFalse(default_dec.allowed)
        self.assertEqual(default_dec.status, "conditional_blocked")

        # Revenue alone is insufficient
        rev_only = evaluate_component("ltx", conditional_attestations={"company_revenue_under_10m_usd": True})
        self.assertFalse(rev_only.allowed)
        self.assertEqual(rev_only.status, "conditional_blocked")

        # Components alone are insufficient (when revenue un-attested or False)
        comp_only = evaluate_component("ltx", conditional_attestations={"ltx_components_audited": True})
        self.assertFalse(comp_only.allowed)
        self.assertEqual(comp_only.status, "conditional_blocked")

        # If revenue >= $10M (revenue flag explicitly False), it must be blocked even if components are audited
        rev_over_10m = evaluate_component(
            "ltx",
            conditional_attestations={
                "company_revenue_under_10m_usd": False,
                "ic_lora_audited": True,
                "gemma_encoder_audited": True,
            },
        )
        self.assertFalse(rev_over_10m.allowed)
        self.assertEqual(rev_over_10m.status, "conditional_blocked")
        self.assertIn("company_revenue_under_10m_usd", rev_over_10m.reason)

        # Partial component audits are insufficient
        ic_only = evaluate_component(
            "ltx",
            conditional_attestations={
                "company_revenue_under_10m_usd": True,
                "ic_lora_audited": True,
                "gemma_encoder_audited": False,
            },
        )
        self.assertFalse(ic_only.allowed)
        self.assertEqual(ic_only.status, "conditional_blocked")

        gemma_only = evaluate_component(
            "ltx",
            conditional_attestations={
                "company_revenue_under_10m_usd": True,
                "ic_lora_audited": False,
                "gemma_encoder_audited": True,
            },
        )
        self.assertFalse(gemma_only.allowed)
        self.assertEqual(gemma_only.status, "conditional_blocked")

        # Both revenue and audited components satisfy clearance (with individual flags)
        cleared = evaluate_component(
            "ltx",
            conditional_attestations={
                "company_revenue_under_10m_usd": True,
                "ic_lora_audited": True,
                "gemma_encoder_audited": True,
            },
        )
        self.assertTrue(cleared.allowed)
        self.assertEqual(cleared.status, "conditional_cleared")

        # Bundle flag also satisfies component requirement when revenue < $10M
        cleared_bundle = evaluate_component(
            "ltx",
            conditional_attestations={
                "company_revenue_under_10m_usd": True,
                "ltx_components_audited": True,
            },
        )
        self.assertTrue(cleared_bundle.allowed)
        self.assertEqual(cleared_bundle.status, "conditional_cleared")

        # Test LTX aliases resolve properly
        for alias in ("ltx", "ltx23", "ltx-2.3", "ltx25", "ltx-2.5", "dearchive", "oumoumad", "official"):
            with self.subTest(alias=alias):
                dec = evaluate_component(
                    alias,
                    conditional_attestations={
                        "company_revenue_under_10m_usd": True,
                        "ic_lora_audited": True,
                        "gemma_encoder_audited": True,
                    },
                )
                self.assertTrue(dec.allowed)
                self.assertEqual(dec.status, "conditional_cleared")

        # In stage_issue across stages:
        # 1. cleanup (dearchive alone)
        self.settings["cleanup"].update({"ai_descratch": "false", "dearchive": "true", "detect_defects": "false"})
        self.assertTrue(stage_issue(self.settings, "cleanup"))  # Blocked by default
        self.settings["global"]["company_revenue_under_10m_usd"] = "true"
        self.settings["global"]["ltx_components_audited"] = "true"
        self.assertEqual(stage_issue(self.settings, "cleanup"), "")  # Cleared with conditions

        # 2. outpaint (oumoumad, official, ltx25)
        for model in ("oumoumad", "official", "ltx25"):
            self.settings["outpaint"]["outpaint_model"] = model
            # With flags set, cleared
            self.assertEqual(stage_issue(self.settings, "outpaint"), "")
            # Clear revenue flag, blocked
            self.settings["global"]["company_revenue_under_10m_usd"] = "false"
            self.assertTrue(stage_issue(self.settings, "outpaint"))
            self.settings["global"]["company_revenue_under_10m_usd"] = "true"

        # 3. colour (ltx)
        self.settings["colour"]["method"] = "ltx"
        self.assertEqual(stage_issue(self.settings, "colour"), "")
        self.settings["global"]["company_revenue_under_10m_usd"] = "false"
        self.assertTrue(stage_issue(self.settings, "colour"))
        self.settings["global"]["company_revenue_under_10m_usd"] = "true"

        # 4. upscale (ltx25, ltx25cq)
        for model in ("ltx25", "ltx25cq"):
            self.settings["upscale"]["method"] = model
            self.assertEqual(stage_issue(self.settings, "upscale"), "")
            self.settings["global"]["company_revenue_under_10m_usd"] = "false"
            self.assertTrue(stage_issue(self.settings, "upscale"))
            self.settings["global"]["company_revenue_under_10m_usd"] = "true"

    def test_conditional_commercial_deep_exemplar(self):
        # Default fails without weight/VGG19 audit flag
        default_dec = evaluate_component("deepexemplar")
        self.assertFalse(default_dec.allowed)
        self.assertEqual(default_dec.status, "conditional_blocked")

        # Cleared when weight_vgg19_audited is true
        cleared = evaluate_component("deepexemplar", conditional_attestations={"weight_vgg19_audited": True})
        self.assertTrue(cleared.allowed)
        self.assertEqual(cleared.status, "conditional_cleared")

        # In stage_issue: colour with deepexemplar
        self.settings["colour"]["method"] = "deepexemplar"
        self.assertTrue(stage_issue(self.settings, "colour"))
        self.settings["global"]["weight_vgg19_audited"] = "true"
        self.assertEqual(stage_issue(self.settings, "colour"), "")

    def test_fail_closed_unknown_component(self):
        unknown_id = "unknown-vision-model-xyz-v9"
        decision = evaluate_component(unknown_id)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.status, "unknown")
        self.assertIn("has not been reviewed", decision.reason)

        with self.assertRaises(CommercialPolicyError):
            enforce_component(unknown_id)

    def test_diagnostic_error_for_unaudited_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_cases = [
                ("custom_lora", "custom_lora_experiment.safetensors", "lora", b"bytes for custom lora"),
                ("custom_checkpoint", "base_model_v1.ckpt", "checkpoint", b"bytes for checkpoint"),
                ("custom_encoder", "text_encoder.safetensors", "text_encoder", b"bytes for text encoder"),
            ]
            for cid, filename, comp_type, content in test_cases:
                with self.subTest(component_type=comp_type, filename=filename):
                    file_path = Path(tmp_dir) / filename
                    file_path.write_bytes(content)
                    expected_hash = hashlib.sha256(content).hexdigest()

                    # Evaluating an un-audited component with a local file
                    decision = evaluate_component(
                        cid,
                        file_path=file_path,
                        component_type=comp_type,
                    )
                    self.assertFalse(decision.allowed)
                    self.assertEqual(decision.sha256, expected_hash)
                    # The diagnostic message MUST include component type, filename, and computed SHA-256
                    self.assertIn(comp_type, decision.reason.lower())
                    self.assertIn(filename, decision.reason)
                    self.assertIn(expected_hash, decision.reason)

                    # enforce_component raises CommercialPolicyError with the exact diagnostic message
                    with self.assertRaises(CommercialPolicyError) as ctx:
                        enforce_component(cid, file_path=file_path, component_type=comp_type)
                    self.assertIn(comp_type, str(ctx.exception).lower())
                    self.assertIn(filename, str(ctx.exception))
                    self.assertIn(expected_hash, str(ctx.exception))

    def test_acceptance_gate_commercial_and_nc_rejection(self):
        """Acceptance Gate: verify DDColor and audited Qwen-2511 succeed while NC models fail cleanly."""
        self.guard()

        # 1. Commercial reference pass with DDColor succeeds
        ddcolor_issue = stage_issue(self.settings, "references", model_override="ddcolor")
        self.assertEqual(ddcolor_issue, "", "DDColor reference pass must succeed under commercial guard")
        ddcolor_decision = evaluate_component("ddcolor")
        self.assertTrue(ddcolor_decision.allowed)
        self.assertEqual(ddcolor_decision.status, "allowed_osi")
        self.assertEqual(ddcolor_decision.license, "Apache-2.0")

        # 2. Commercial reference pass with audited Qwen-2511 succeeds
        self.settings["global"]["qwen_2511_audited"] = "true"
        qwen_issue = stage_issue(self.settings, "references", model_override="qwen")
        self.assertEqual(qwen_issue, "", "Audited Qwen-2511 reference pass must succeed")
        qwen_decision = evaluate_component("qwen-image-edit-2511", conditional_attestations={"qwen_2511_audited": True})
        self.assertTrue(qwen_decision.allowed)

        # 3. Clean rejection when any Non-Commercial (NC) model is selected
        nc_cases = [
            ("references", "qwen21", "Qwen-Image-2.1"),
            ("colour", "colormnet", "ColorMNet"),
            ("colour", "cmnet2", "ColorMNet"),
            ("cleanup", "propainter", "ProPainter"),
            ("outpaint", "h3", "MiniMax H3"),
        ]
        for stage, model, expected_text in nc_cases:
            with self.subTest(stage=stage, model=model):
                if stage == "cleanup":
                    self.settings["cleanup"]["ai_descratch"] = "true"
                    issue = stage_issue(self.settings, "cleanup")
                    self.settings["cleanup"]["ai_descratch"] = "false"
                else:
                    issue = stage_issue(self.settings, stage, model_override=model)
                self.assertTrue(issue, f"{stage} with {model} must be blocked")
                self.assertIn(expected_text, issue)

                # enforce_component fails closed
                with self.assertRaises(CommercialPolicyError):
                    enforce_component(model)

    def test_zero_placeholder_hashes_in_policy_registry(self):
        reg = load_policy_registry()
        self.assertEqual(reg.get("schema_version"), "1.0.0")
        policies = reg.get("policies", {})

        for category_name, category in policies.items():
            models = category.get("models", {})
            for model_id, model_def in models.items():
                hashes = model_def.get("audited_hashes", [])
                for h in hashes:
                    self.assertIsInstance(h, str)
                    self.assertEqual(len(h), 64, f"Hash for {model_id} must be 64 characters")
                    self.assertFalse(all(c == "0" for c in h), f"Placeholder zero-hash found in {model_id}")
                    self.assertNotIn("placeholder", h.lower())
                    self.assertNotIn("todo", h.lower())

    def test_missing_or_corrupt_registry_fails_closed(self):
        missing_path = ROOT / "configs" / "non_existent_policy_registry_9999.json"
        reg = load_policy_registry(missing_path)
        decision = evaluate_component("ddcolor", registry=reg)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.status, "unknown")

    def test_load_policy_registry_coerces_string_paths(self):
        from ai_remaster_gui.commercial_policy import REGISTRY_PATH
        # Passing string path must not raise AttributeError
        reg_from_str = load_policy_registry(str(REGISTRY_PATH))
        self.assertEqual(reg_from_str.get("schema_version"), "1.0.0")
        self.assertIn("policies", reg_from_str)

        # Missing string path fails closed cleanly without AttributeError
        missing_str = str(ROOT / "configs" / "non_existent_policy_registry_9999.json")
        reg_missing = load_policy_registry(missing_str)
        self.assertEqual(reg_missing, {"schema_version": "1.0.0", "policies": {}})


if __name__ == "__main__":
    unittest.main()
