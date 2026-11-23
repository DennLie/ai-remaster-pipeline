"""API and security tests for the Color Bible HTTP endpoints."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from ai_remaster_gui import http_handler, state
from ai_remaster_gui.color_bible_db import SQLiteColorBibleRepository
from ai_remaster_gui.color_bible_service import ColorBibleService


class _MockHandler(http_handler.Handler):
    """Handler test double that records responses without real sockets."""

    def __init__(self, path: str, body: dict | None = None, headers: dict | None = None) -> None:
        self.path = path
        self.headers = headers or {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        self._body = body or {}
        self.responses: dict[str, object] = {}

    def read_json(self) -> dict:
        return self._body

    def send_json(self, payload):
        self.responses["json"] = payload

    def send_error(self, code, message=None, explain=None):
        self.responses["error"] = code
        self.responses["error_message"] = message


class FakeApp:
    def __init__(self) -> None:
        self.log: list[str] = []
        self.settings: dict = {"global": {}}

    def state(self, view: str = "") -> dict:
        return {"_view": view}


class ColorBibleApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_app = state.APP
        state.APP = FakeApp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "test_color_bible.db"
        self.patcher = mock.patch.dict("os.environ", {"ARP_COLOR_BIBLE_DB": str(self.db_path)})
        self.patcher.start()

        # Initialize repository
        with ColorBibleService(db_path=self.db_path) as svc:
            svc.create_character("film_test", "Elena", "Heroine")

    def tearDown(self) -> None:
        self.patcher.stop()
        self.temp_dir.cleanup()
        state.APP = self.old_app

    def test_get_state(self) -> None:
        handler = _MockHandler("/api/color-bible/state?film_id=film_test")
        handler.do_GET()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("ok"))
        self.assertEqual(res.get("film_id"), "film_test")
        self.assertEqual(len(res.get("characters", [])), 1)
        self.assertEqual(res["characters"][0]["name"], "Elena")

    def test_create_variant_and_assign_ranges(self) -> None:
        # Get character id
        with ColorBibleService(db_path=self.db_path) as svc:
            char = svc.list_characters("film_test")[0]
            char_id = char.character_id

        # Create variant
        var_handler = _MockHandler(
            "/api/color-bible/variant",
            body={
                "character_id": char_id,
                "costume_id": "costume_trench",
                "lighting_context": "exterior_overcast",
            },
        )
        var_handler.do_POST()
        var_res = var_handler.responses.get("json")
        self.assertIsNotNone(var_res)
        self.assertTrue(var_res.get("ok"))
        var_id = var_res["variant"]["variant_id"]

        # Assign ranges via string "12-18, 410-435"
        asgn_handler = _MockHandler(
            "/api/color-bible/assign-ranges",
            body={
                "film_id": "film_test",
                "entity_type": "character",
                "entity_id": char_id,
                "variant_id": var_id,
                "ranges": "12-18, 410-435",
                "confidence": 0.95,
            },
        )
        asgn_handler.do_POST()
        asgn_res = asgn_handler.responses.get("json")
        self.assertIsNotNone(asgn_res)
        self.assertTrue(asgn_res.get("ok"))
        self.assertEqual(len(asgn_res["assignments"]), 2)
        asgn_id = asgn_res["assignments"][0]["assignment_id"]

        # Shot assignments query
        shot_handler = _MockHandler("/api/color-bible/shot-assignments?film_id=film_test&shot_index=15")
        shot_handler.do_GET()
        shot_res = shot_handler.responses.get("json")
        self.assertIsNotNone(shot_res)
        self.assertTrue(shot_res.get("ok"))
        self.assertEqual(len(shot_res["assignments"]), 1)
        self.assertEqual(shot_res["assignments"][0]["variant"]["costume_id"], "costume_trench")
        self.assertFalse(shot_res.get("needs_review"))
        self.assertEqual(shot_res.get("approval_status"), "approved")

        # Delete single assignment
        del_handler = _MockHandler(
            "/api/color-bible/assign-ranges",
            body={
                "film_id": "film_test",
                "action": "delete",
                "delete_assignment_id": asgn_id,
            },
        )
        del_handler.do_POST()
        del_res = del_handler.responses.get("json")
        self.assertTrue(del_res.get("ok"))
        self.assertEqual(len(del_res["assignments"]), 1)

    def test_shot_assignments_needs_review_and_approval_status(self) -> None:
        with ColorBibleService(db_path=self.db_path) as svc:
            char = svc.list_characters("film_test")[0]
            svc.assign_entity_ranges(
                film_id="film_test",
                entity_type="character",
                entity_id=char.character_id,
                ranges=[(50, 55)],
                confidence=0.65,
                approval_status="pending_review",
            )
            svc.update_shot_qc_status("film_test", 50, "needs_review")

        handler = _MockHandler("/api/color-bible/shot-assignments?film_id=film_test&shot_index=50")
        handler.do_GET()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("ok"))
        self.assertTrue(res.get("needs_review"))
        self.assertEqual(res.get("qc_status"), "needs_review")
        self.assertEqual(res.get("approval_status"), "needs_review")
        self.assertTrue(len(res.get("review_reasons", [])) > 0)

    def test_objects_and_locations_post(self) -> None:
        # Create object
        obj_handler = _MockHandler(
            "/api/color-bible/object",
            body={
                "film_id": "film_test",
                "kind": "prop",
                "name": "Leather Briefcase",
                "palette": ["#5c3a21"],
            },
        )
        obj_handler.do_POST()
        self.assertTrue(obj_handler.responses.get("json", {}).get("ok"))

        # Create location
        loc_handler = _MockHandler(
            "/api/color-bible/location",
            body={
                "film_id": "film_test",
                "set_name": "Train Compartment",
                "lighting_context": "warm tungsten",
                "palette": ["#8a492d", "#1a2530"],
            },
        )
        loc_handler.do_POST()
        self.assertTrue(loc_handler.responses.get("json", {}).get("ok"))

    def test_sync_manifest_and_path_confinement(self) -> None:
        # Create a valid manifest inside the project / served path
        manifest_file = Path("test_shot_manifest_api.csv")
        manifest_file.write_text("index,start,end\n0,00:00:01.000,00:00:04.000\n1,00:00:04.000,00:00:08.500\n", encoding="utf-8")
        try:
            # Valid sync
            sync_handler = _MockHandler(
                "/api/color-bible/sync-manifest",
                body={
                    "film_id": "film_test",
                    "manifest": str(manifest_file),
                },
            )
            sync_handler.do_POST()
            self.assertTrue(sync_handler.responses.get("json", {}).get("ok"))
            self.assertEqual(sync_handler.responses["json"]["shots_synced"], 2)

            # Path traversal rejection (must return 400)
            traversal_handler = _MockHandler(
                "/api/color-bible/sync-manifest",
                body={
                    "film_id": "film_test",
                    "manifest": "../../etc/passwd",
                },
            )
            traversal_handler.do_POST()
            self.assertEqual(traversal_handler.responses.get("error"), 400)
        finally:
            if manifest_file.exists():
                manifest_file.unlink()

    def test_reference_registration_approval_and_snapshot_hash(self) -> None:
        # Create keyframe image in project
        keyframe_path = Path("test_keyframe_api.png")
        img = Image.new("RGB", (64, 64), color=(180, 50, 40))
        img.save(keyframe_path)

        try:
            # Register reference with auto_approve = False
            ref_handler = _MockHandler(
                "/api/color-bible/reference",
                body={
                    "film_id": "film_test",
                    "entity_type": "character",
                    "entity_id": "char_mock",
                    "keyframe_path": str(keyframe_path),
                    "auto_approve": False,
                },
            )
            ref_handler.do_POST()
            ref_res = ref_handler.responses.get("json")
            self.assertTrue(ref_res.get("ok"))
            ref_id = ref_res["reference"]["ref_id"]
            self.assertEqual(ref_res["reference"]["approval_status"], "draft")

            # Check snapshot hash before approval
            hash_handler = _MockHandler("/api/color-bible/snapshot-hash?film_id=film_test")
            hash_handler.do_GET()
            hash_before = hash_handler.responses.get("json")["snapshot_hash"]

            # Approve reference
            app_handler = _MockHandler(
                "/api/color-bible/reference-approval",
                body={"ref_id": ref_id, "action": "approve", "approved_by": "lead_colorist"},
            )
            app_handler.do_POST()
            app_res = app_handler.responses.get("json")
            self.assertTrue(app_res.get("ok"))
            self.assertEqual(app_res["reference"]["approval_status"], "approved")

            # Check snapshot hash after approval: should be different and start with sha256:
            hash_handler.do_GET()
            hash_after = hash_handler.responses.get("json")["snapshot_hash"]
            self.assertTrue(hash_after.startswith("sha256:"))
            self.assertNotEqual(hash_before, hash_after)

            # Reject reference
            rej_handler = _MockHandler(
                "/api/color-bible/reference-approval",
                body={"ref_id": ref_id, "action": "reject"},
            )
            rej_handler.do_POST()
            self.assertTrue(rej_handler.responses.get("json", {}).get("ok"))
            self.assertEqual(rej_handler.responses["json"]["reference"]["approval_status"], "rejected")

            # Keyframe path traversal rejection (must return 400)
            bad_ref_handler = _MockHandler(
                "/api/color-bible/reference",
                body={
                    "film_id": "film_test",
                    "entity_type": "character",
                    "entity_id": "char_mock",
                    "keyframe_path": "../outside.png",
                },
            )
            bad_ref_handler.do_POST()
            self.assertEqual(bad_ref_handler.responses.get("error"), 400)

            # Mask path traversal rejection (must return 400)
            bad_mask_handler = _MockHandler(
                "/api/color-bible/reference",
                body={
                    "film_id": "film_test",
                    "entity_type": "character",
                    "entity_id": "char_mock",
                    "keyframe_path": str(keyframe_path),
                    "mask_uri": "../../evil_mask.png",
                },
            )
            bad_mask_handler.do_POST()
            self.assertEqual(bad_mask_handler.responses.get("error"), 400)

        finally:
            if keyframe_path.exists():
                keyframe_path.unlink()

    def test_character_post_endpoint(self) -> None:
        """Verify POST /api/color-bible/character creation and validation."""
        # Valid creation
        handler = _MockHandler(
            "/api/color-bible/character",
            body={"film_id": "film_test", "name": "Viktor", "notes": "Antagonist"},
        )
        handler.do_POST()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("ok"))
        self.assertEqual(res["character"]["name"], "Viktor")
        self.assertEqual(res["character"]["notes"], "Antagonist")

        # Empty name should fail
        err_handler = _MockHandler(
            "/api/color-bible/character",
            body={"film_id": "film_test", "name": "  "},
        )
        err_handler.do_POST()
        err_res = err_handler.responses.get("json")
        self.assertIsNotNone(err_res)
        self.assertFalse(err_res.get("ok"))
        self.assertIn("Character name must not be empty", err_res.get("error", ""))

    def test_shot_assignments_invalid_index(self) -> None:
        """Verify GET /api/color-bible/shot-assignments handles invalid shot indices."""
        handler = _MockHandler("/api/color-bible/shot-assignments?film_id=film_test&shot_index=invalid_index")
        handler.do_GET()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("ok"))
        self.assertIn("Invalid shot_index", res.get("error", ""))

    def test_reference_approval_invalid_action(self) -> None:
        """Verify POST /api/color-bible/reference-approval rejects invalid action strings."""
        handler = _MockHandler(
            "/api/color-bible/reference-approval",
            body={"ref_id": "ref_dummy", "action": "invalid_action"},
        )
        handler.do_POST()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("ok"))
        self.assertIn("Invalid approval action", res.get("error", ""))

    def test_security_origin_and_host_rejection(self) -> None:
        """Verify non-local Origin or Host headers receive 403 Forbidden."""
        # Bad Origin
        origin_handler = _MockHandler(
            "/api/color-bible/character",
            body={"film_id": "film_test", "name": "Hacker"},
            headers={"Host": "127.0.0.1:8765", "Origin": "http://evil.com"},
        )
        origin_handler.do_POST()
        self.assertEqual(origin_handler.responses.get("error"), 403)

        # Bad Host (DNS rebinding)
        host_handler = _MockHandler(
            "/api/color-bible/character",
            body={"film_id": "film_test", "name": "Hacker"},
            headers={"Host": "evil.com", "Origin": "http://evil.com"},
        )
        host_handler.do_POST()
        self.assertEqual(host_handler.responses.get("error"), 403)

    def test_api_invalid_ranges(self) -> None:
        """Verify POST /api/color-bible/assign-ranges rejects invalid range formats."""
        handler = _MockHandler(
            "/api/color-bible/assign-ranges",
            body={
                "film_id": "film_test",
                "entity_type": "character",
                "entity_id": "char_mock",
                "ranges": "25-10",
            },
        )
        handler.do_POST()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("ok"))
        self.assertIn("Invalid range", res.get("error", ""))

    def test_api_missing_film_id_snapshot_hash(self) -> None:
        """Verify GET /api/color-bible/snapshot-hash requires film_id."""
        handler = _MockHandler("/api/color-bible/snapshot-hash")
        handler.do_GET()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertFalse(res.get("ok"))
        self.assertIn("film_id is required", res.get("error", ""))

    def test_api_validation_errors_object_and_location(self) -> None:
        """Verify object and location endpoints reject invalid payload parameters."""
        # Invalid object kind
        obj_handler = _MockHandler(
            "/api/color-bible/object",
            body={"film_id": "film_test", "kind": "spaceship", "name": "Falcon"},
        )
        obj_handler.do_POST()
        res_obj = obj_handler.responses.get("json")
        self.assertFalse(res_obj.get("ok"))
        self.assertIn("Invalid object kind", res_obj.get("error", ""))

        # Empty location name
        loc_handler = _MockHandler(
            "/api/color-bible/location",
            body={"film_id": "film_test", "set_name": "  "},
        )
        loc_handler.do_POST()
        res_loc = loc_handler.responses.get("json")
        self.assertFalse(res_loc.get("ok"))
        self.assertIn("Location set_name must not be empty", res_loc.get("error", ""))

    def test_api_variant_nonexistent_character(self) -> None:
        """Verify POST /api/color-bible/variant fails when character does not exist."""
        handler = _MockHandler(
            "/api/color-bible/variant",
            body={"character_id": "nonexistent_char", "costume_id": "Trench"},
        )
        handler.do_POST()
        res = handler.responses.get("json")
        self.assertFalse(res.get("ok"))
        self.assertIn("does not exist", res.get("error", ""))

    def test_api_state_empty_film(self) -> None:
        """Verify GET /api/color-bible/state handles non-existent film gracefully."""
        handler = _MockHandler("/api/color-bible/state?film_id=film_ghost")
        handler.do_GET()
        res = handler.responses.get("json")
        self.assertTrue(res.get("ok"))
        self.assertEqual(res.get("film_id"), "film_ghost")
        self.assertEqual(res.get("characters"), [])
        self.assertEqual(res.get("assignments"), [])
        self.assertEqual(res.get("references"), [])




    def test_post_film_creation(self) -> None:
        """Verify dedicated POST /api/color-bible/film creates and persists a new film."""
        handler = _MockHandler(
            "/api/color-bible/film",
            body={
                "film_id": "film_new",
                "title": "New Feature Film",
                "source_video": "vids/feat.mp4",
            },
        )
        handler.do_POST()
        res = handler.responses.get("json")
        self.assertIsNotNone(res)
        self.assertTrue(res.get("ok"))
        self.assertEqual(res["film"]["film_id"], "film_new")
        self.assertEqual(res["film"]["title"], "New Feature Film")
        self.assertEqual(res["film"]["source_video"], "vids/feat.mp4")

        with ColorBibleService(db_path=self.db_path) as svc:
            film = svc.get_film("film_new")
            self.assertIsNotNone(film)
            self.assertEqual(film.title, "New Feature Film")

    def test_post_film_validation(self) -> None:
        """Verify POST /api/color-bible/film rejects empty film_id."""
        handler = _MockHandler(
            "/api/color-bible/film",
            body={"film_id": "   ", "title": "Empty ID"},
        )
        handler.do_POST()
        res = handler.responses.get("json")
        self.assertFalse(res.get("ok"))
        self.assertIn("film_id must not be empty", res.get("error", ""))


if __name__ == "__main__":
    unittest.main()
