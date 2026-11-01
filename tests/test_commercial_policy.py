"""GPU-free tests for the opt-in commercial model policy."""

import unittest

from ai_remaster_gui.commercial_policy import enabled, pipeline_issues, stage_issue
from ai_remaster_gui.runtime_settings import base_settings


class CommercialModelPolicyTests(unittest.TestCase):
    def setUp(self):
        self.settings = base_settings()

    def guard(self):
        self.settings["global"]["commercial_model_guard"] = "true"

    def test_opt_in_preserves_legacy_projects(self):
        self.assertFalse(enabled(self.settings))
        self.assertEqual(stage_issue(self.settings, "cleanup"), "")
        self.settings["global"].pop("commercial_model_guard")
        self.assertFalse(enabled(self.settings))

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


if __name__ == "__main__":
    unittest.main()
