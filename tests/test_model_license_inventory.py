"""CPU-only checks that runtime model choices stay represented in the audit.

This prevents a new GUI backend from silently bypassing review when its
selector is added. It does NOT automatically grant permission to use models.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ai_remaster_gui.commercial_policy import stage_issue
from ai_remaster_gui.models import STAGES
from ai_remaster_gui.runtime_settings import base_settings


INVENTORY_PATH = Path(__file__).resolve().parents[1] / "docs" / "model-license-inventory.json"
MODEL_SELECTORS = {
    "outpaint": "outpaint_model",
    "references": "method",
    "colour": "method",
    "upscale": "method",
}
MODEL_FLAGS = {
    ("cleanup", "ai_descratch", "true"),
    ("cleanup", "dearchive", "true"),
    ("audio", "create_music", "true"),
    ("audio", "create_sfx", "true"),
}


class ModelLicenseInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))

    def test_each_gui_model_choice_is_inventory_covered(self):
        expected = set(MODEL_FLAGS)
        for stage in STAGES:
            selector = MODEL_SELECTORS.get(stage.key)
            if not selector:
                continue
            field = next(item for item in stage.fields if item[0] == selector)
            self.assertTrue(field[2].startswith("select:"), (stage.key, selector))
            expected.update((stage.key, selector, option) for option in field[2].split(":", 1)[1].split("|"))
        actual = {(row["stage"], row["field"], row["value"]) for row in self.inventory["entries"]}
        self.assertEqual(actual, expected, "A selectable AI model was added/removed without updating the licence audit.")

    def test_entries_have_unique_ids_and_primary_evidence(self):
        entries = self.inventory["entries"]
        ids = [row["id"] for row in entries]
        self.assertEqual(len(ids), len(set(ids)))
        for row in entries:
            self.assertEqual(row["id"], ".".join((row["stage"], row["field"], row["value"])))
            self.assertIn(row["status"], {"restricted", "incomplete", "external_terms"})
            self.assertTrue(row["reason"])
            self.assertTrue(row["evidence"])
            self.assertTrue(all(str(link).startswith("https://") for link in row["evidence"]))

    def test_all_unaudited_selected_backends_fail_closed(self):
        for row in self.inventory["entries"]:
            with self.subTest(model=row["id"]):
                settings = base_settings()
                settings["global"]["commercial_model_guard"] = "true"
                settings[row["stage"]][row["field"]] = row["value"]
                self.assertTrue(stage_issue(settings, row["stage"]), row["id"])

    def test_candidates_cannot_be_marked_cleared(self):
        candidates = self.inventory["candidates"]
        self.assertGreaterEqual(len(candidates), 3)
        for entry in candidates:
            self.assertNotEqual(entry["status"], "cleared")
            self.assertTrue(entry["evidence"])


if __name__ == "__main__":
    unittest.main()
