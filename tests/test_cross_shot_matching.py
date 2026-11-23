"""Unit tests for Phase 3: Semantic Reference Matching & Cross-Shot Color Consistency.

Comprehensive CPU-only unit test suite validating:
- Criterion (a): Cross-shot semantic reference retrieval and costume isolation (shot 520 vs shot 5 vs shot 519).
- Criterion (b): Backward compatibility when no Color Bible DB or film_id exists.
- Criterion (c): Priority C review gate on pending_review / low-confidence assignments before cache check.
- Criterion (d): Qwen3-VL structured prompt enrichment with RGB swatches and CIE-LAB statistics.
- Criterion (e): Two-stage reference router (DDColor draft -> Qwen-2511 escalation) and row overrides.
- Criterion (f): Scoped cache key invalidation affecting ONLY shots referencing updated entities.
- Criterion (g): Commercial Model Guard failure-closed policy and compliance_report.json provenance.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Ensure project root and scripts directories are on sys.path
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import qwen_colorize_references as qwen
from common import (
    file_fingerprint,
    resumable_output,
    signature_matches,
    signature_path,
    write_signature,
)
from ai_remaster_gui.color_bible_db import SQLiteColorBibleRepository
from ai_remaster_gui.color_bible_service import ColorBibleService
from ai_remaster_gui.commercial_policy import (
    CommercialPolicyError,
    enforce_component,
    evaluate_component,
)
from ai_remaster_gui.manifests import read_manifest, update_manifest_row, write_manifest_details
from ai_remaster_gui.semantic_matching import (
    ColorBibleMatcher,
    PromptEnricher,
    SemanticMatchResult,
    TwoStageReferenceRouter,
    build_reference_signature,
    compute_scoped_cache_key,
    enrich_prompt,
    record_reference_compliance,
    resolve_references,
    select_backend,
)


# ===========================================================================
# Criterion (a): Shot 520 retrieves Costume A master reference from shot 5,
#                ignoring adjacent shot 519's Costume B
# ===========================================================================


class TestCriterionACrossShotCostumeIsolation(unittest.TestCase):
    """Validate cross-shot semantic matching and costume isolation across film shots."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir_obj.name)
        self.db_path = self.tmp_path / "test_color_bible.db"
        self.service = ColorBibleService(db_path=self.db_path)
        self.film_id = "feature_film_cross_shot"

    def tearDown(self) -> None:
        self.service.close()
        self.tmp_dir_obj.cleanup()

    def test_shot_520_retrieves_costume_a_from_shot_5_ignoring_adjacent_shot_519_costume_b(self) -> None:
        """Shot 520 retrieves Costume A's master reference from shot 5 rather than adjacent shot 519's Costume B."""
        # 1. Setup Alice with Costume A (coat_blue)
        alice = self.service.create_character(self.film_id, "Alice", notes="Lead detective")
        costume_a = self.service.create_appearance_variant(
            alice.character_id, "costume_a", notes="Blue wool trench coat"
        )
        img_costume_a = self.tmp_path / "alice_costume_a_shot5.png"
        Image.new("RGB", (64, 64), (20, 60, 180)).save(img_costume_a)  # Dominant Blue

        ref_a = self.service.register_reference_image(
            self.film_id,
            entity_type="character",
            entity_id=alice.character_id,
            variant_id=costume_a.variant_id,
            keyframe_path=img_costume_a,
            auto_approve=True,
            approved_by="lead_colorist",
        )

        # Assign Alice Costume A to shots 0-10 (early film) and 520-525 (late climax)
        self.service.assign_entity_ranges(
            self.film_id,
            entity_type="character",
            entity_id=alice.character_id,
            ranges=[(0, 10), (520, 525)],
            variant_id=costume_a.variant_id,
            confidence=0.98,
            approval_status="approved",
        )

        # 2. Setup Bob with Costume B (suit_black) in adjacent shot 519
        bob = self.service.create_character(self.film_id, "Bob", notes="Antagonist")
        costume_b = self.service.create_appearance_variant(
            bob.character_id, "costume_b", notes="Black tuxedo suit"
        )
        img_costume_b = self.tmp_path / "bob_costume_b_shot519.png"
        Image.new("RGB", (64, 64), (15, 15, 15)).save(img_costume_b)  # Dominant Black

        ref_b = self.service.register_reference_image(
            self.film_id,
            entity_type="character",
            entity_id=bob.character_id,
            variant_id=costume_b.variant_id,
            keyframe_path=img_costume_b,
            auto_approve=True,
            approved_by="lead_colorist",
        )

        # Assign Bob Costume B to shot 519
        self.service.assign_entity_ranges(
            self.film_id,
            entity_type="character",
            entity_id=bob.character_id,
            ranges=[(515, 519)],
            variant_id=costume_b.variant_id,
            confidence=0.95,
            approval_status="approved",
        )

        # 3. Build manifest rows for shots 0 through 525 (526 shots total)
        rows: list[dict[str, str]] = []
        for s in range(526):
            row: dict[str, str] = {
                "_shot_index": str(s),
                "source_reference": f"references/shot_{s:04d}_src.png",
            }
            if s == 5:
                row["color_reference"] = str(img_costume_a)
            elif s == 519:
                row["color_reference"] = str(img_costume_b)
            else:
                row["color_reference"] = ""
            rows.append(row)

        # 4. Resolve references for shot 520 with Color Bible enabled
        # Test resolve_shot_references
        match_result: SemanticMatchResult = qwen.resolve_shot_references(
            rows=rows,
            index=520,
            count=2,
            film_id=self.film_id,
            color_bible_service=self.service,
        )

        self.assertIsInstance(match_result, SemanticMatchResult)
        self.assertFalse(match_result.needs_review)
        self.assertEqual(match_result.review_reasons, [])
        self.assertIn(ref_a.ref_id, match_result.retrieved_ref_ids)
        self.assertNotIn(ref_b.ref_id, match_result.retrieved_ref_ids)

        # Primary reference (index 0) MUST be Costume A's master reference from shot 5
        self.assertGreaterEqual(len(match_result.reference_paths), 1)
        self.assertEqual(match_result.reference_paths[0].resolve(), img_costume_a.resolve())

        # Test reference_images_for convenience function
        refs = qwen.reference_images_for(
            rows=rows,
            index=520,
            count=1,
            film_id=self.film_id,
            color_bible_service=self.service,
        )
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].resolve(), img_costume_a.resolve())

        # Contrast with legacy behavior when film_id is None (would have picked adjacent shot 519)
        legacy_refs = qwen.reference_images_for(
            rows=rows,
            index=520,
            count=1,
            film_id=None,
            color_bible_service=None,
        )
        self.assertEqual(len(legacy_refs), 1)
        # Without Color Bible, neighbor distance picks shot 519 (distance 1) over shot 5 (distance 515)
        self.assertEqual(legacy_refs[0].resolve(), img_costume_b.resolve())

    def test_costume_isolation_across_multiple_shots_in_range(self) -> None:
        """Verify that all shots in Alice's assigned range (520..525) retrieve Costume A."""
        char = self.service.create_character(self.film_id, "Alice")
        costume_a = self.service.create_appearance_variant(char.character_id, "costume_a")
        img_a = self.tmp_path / "alice_costume_a.png"
        Image.new("RGB", (32, 32), (50, 100, 200)).save(img_a)

        ref_a = self.service.register_reference_image(
            self.film_id,
            entity_type="character",
            entity_id=char.character_id,
            variant_id=costume_a.variant_id,
            keyframe_path=img_a,
            auto_approve=True,
        )
        self.service.assign_entity_ranges(
            self.film_id,
            entity_type="character",
            entity_id=char.character_id,
            ranges=[(520, 525)],
            variant_id=costume_a.variant_id,
        )

        rows = [{"_shot_index": str(s), "color_reference": ""} for s in range(530)]
        for test_shot in (520, 522, 525):
            res = ColorBibleMatcher.resolve_references(
                rows=rows,
                index=test_shot,
                count=1,
                film_id=self.film_id,
                service=self.service,
            )
            self.assertEqual(len(res.reference_paths), 1)
            self.assertEqual(res.reference_paths[0].resolve(), img_a.resolve())
            self.assertEqual(res.retrieved_ref_ids, [ref_a.ref_id])


# ===========================================================================
# Criterion (b): Backward compatibility when no Color Bible DB exists
# ===========================================================================


class TestCriterionBBackwardCompatibility(unittest.TestCase):
    """Validate 100% identical neighbor reference selection when Color Bible is not configured."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir_obj.name)

        # Create synthetic neighbor frames
        self.f0 = self.tmp_path / "ref_shot00.png"
        self.f1 = self.tmp_path / "ref_shot01.png"
        self.f2_a = self.tmp_path / "ref_shot02_a.png"
        self.f3 = self.tmp_path / "ref_shot03.png"

        for p in (self.f0, self.f1, self.f2_a, self.f3):
            p.write_bytes(b"synthetic_image_bytes")

        self.rows: list[dict[str, str]] = [
            {"_shot_index": "0", "color_reference": str(self.f0)},
            {"_shot_index": "1", "color_reference": str(self.f1)},
            {"_shot_index": "2", "color_reference": str(self.f2_a)},
            {"_shot_index": "2", "color_reference": ""},  # Same shot slot
            {"_shot_index": "3", "color_reference": str(self.f3)},
        ]

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_backward_compatibility_none_service_and_none_film_id(self) -> None:
        """When film_id=None and color_bible_service=None, preserves neighbor selection without errors."""
        # Query for shot 2 (second slot at index 3)
        refs = qwen.reference_images_for(
            rows=self.rows,
            index=3,
            count=2,
            film_id=None,
            color_bible_service=None,
        )
        expected = qwen._neighbor_reference_images_for(self.rows, index=3, count=2)

        self.assertEqual(refs, expected)
        self.assertEqual(len(refs), 2)
        # Own shot's other reference wins first (self.f2_a)
        self.assertEqual(refs[0], self.f2_a.resolve())
        # Closest adjacent shot wins next (self.f3 or self.f1, distance 1; earlier wins tie)
        self.assertEqual(refs[1], self.f1.resolve())

    def test_backward_compatibility_empty_string_film_id(self) -> None:
        """When film_id is empty string or whitespace, falls back to neighbor logic."""
        mock_service = MagicMock()
        refs = qwen.reference_images_for(
            rows=self.rows,
            index=1,
            count=2,
            film_id="   ",
            color_bible_service=mock_service,
        )
        mock_service.get_shot_palette_assignments.assert_not_called()
        self.assertEqual(refs, qwen._neighbor_reference_images_for(self.rows, index=1, count=2))

    def test_backward_compatibility_missing_service_with_film_id(self) -> None:
        """When film_id is provided but color_bible_service is None, falls back safely."""
        refs = qwen.reference_images_for(
            rows=self.rows,
            index=0,
            count=2,
            film_id="some_feature_film",
            color_bible_service=None,
        )
        expected = qwen._neighbor_reference_images_for(self.rows, index=0, count=2)
        self.assertEqual(refs, expected)

    def test_resolve_references_backward_compatibility_returns_empty_bible_metadata(self) -> None:
        """SemanticMatchResult returned during backward compatibility has clean defaults."""
        res = resolve_references(rows=self.rows, index=1, count=1, film_id=None, service=None)
        self.assertFalse(res.needs_review)
        self.assertEqual(res.review_reasons, [])
        self.assertEqual(res.retrieved_ref_ids, [])
        self.assertEqual(res.retrieved_ref_hashes, {})
        self.assertEqual(res.entities, [])
        self.assertEqual(res.color_bible_version, "")


# ===========================================================================
# Criterion (c): pending_review or low-confidence assignments trigger needs_review gate
# ===========================================================================


class TestCriterionCReviewGate(unittest.TestCase):
    """Validate Priority C review gate short-circuiting and manifest marking."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_review_gate_triggered_by_pending_review_status(self) -> None:
        """An assignment with approval_status == 'pending_review' marks needs_review=True."""
        mock_service = MagicMock()
        mock_service.get_snapshot_hash.return_value = "sha256:snap001"
        mock_service.get_shot_palette_assignments.return_value = [
            {
                "entity_type": "character",
                "entity_id": "charlie",
                "variant_id": "trench_coat",
                "entity": {"name": "Charlie"},
                "assignment": {"approval_status": "pending_review", "confidence": 0.95},
                "color_references": [],
            }
        ]

        result = ColorBibleMatcher.resolve_references(
            rows=[],
            index=10,
            count=2,
            film_id="film_noir",
            service=mock_service,
            confidence_threshold=0.80,
        )

        self.assertTrue(result.needs_review)
        self.assertTrue(any("pending_review" in r for r in result.review_reasons))
        self.assertTrue(any("Charlie" in r for r in result.review_reasons))

    def test_review_gate_triggered_by_confidence_below_threshold(self) -> None:
        """An assignment with confidence == 0.65 (< 0.80) marks needs_review=True."""
        mock_service = MagicMock()
        mock_service.get_snapshot_hash.return_value = "sha256:snap001"
        mock_service.get_shot_palette_assignments.return_value = [
            {
                "entity_type": "character",
                "entity_id": "diane",
                "variant_id": "evening_gown",
                "entity": {"name": "Diane"},
                "assignment": {"approval_status": "approved", "confidence": 0.65},
                "color_references": [],
            }
        ]

        result = ColorBibleMatcher.resolve_references(
            rows=[],
            index=12,
            count=2,
            film_id="film_noir",
            service=mock_service,
            confidence_threshold=0.80,
        )

        self.assertTrue(result.needs_review)
        self.assertTrue(any("0.65 < 0.80" in r for r in result.review_reasons))

    def test_review_gate_passes_when_confidence_meets_custom_threshold(self) -> None:
        """Assignments meeting a custom lower threshold pass cleanly."""
        mock_service = MagicMock()
        mock_service.get_snapshot_hash.return_value = "sha256:snap001"
        mock_service.get_shot_palette_assignments.return_value = [
            {
                "entity_type": "character",
                "entity_id": "diane",
                "entity": {"name": "Diane"},
                "assignment": {"approval_status": "approved", "confidence": 0.70},
                "color_references": [],
            }
        ]

        # With threshold 0.60: passes
        res_pass = ColorBibleMatcher.resolve_references(
            rows=[], index=12, count=2, film_id="film_noir", service=mock_service, confidence_threshold=0.60
        )
        self.assertFalse(res_pass.needs_review)

        # With threshold 0.80: fails
        res_fail = ColorBibleMatcher.resolve_references(
            rows=[], index=12, count=2, film_id="film_noir", service=mock_service, confidence_threshold=0.80
        )
        self.assertTrue(res_fail.needs_review)

    def test_main_with_args_review_gate_skips_inference_even_if_cached_output_exists(self) -> None:
        """Review Gate check runs BEFORE cache check: skips inference and marks qc_status='needs_review'."""
        src_img = self.tmp_path / "shot_0042_src.png"
        dst_img = self.tmp_path / "shot_0042_clr.png"
        Image.new("RGB", (32, 32), (128, 128, 128)).save(src_img)
        # Pre-create cached output image and matching signature
        Image.new("RGB", (32, 32), (128, 128, 128)).save(dst_img)

        wf_file = self.tmp_path / "workflow.json"
        wf_file.write_text("{}", encoding="utf-8")

        parser = qwen.build_parser()
        args = parser.parse_args(["--manifest", "dummy.csv", "--workflow", str(wf_file)])

        # Write pre-existing valid signature so cache hit would normally occur
        pre_sig = qwen.signature(
            args,
            workflow_path=wf_file,
            source_path=src_img,
            prompt="Initial prompt",
            shot_index=42,
            retrieved_ref_hashes={},
            color_bible_version="v1",
        )
        write_signature(dst_img, pre_sig)
        self.assertTrue(resumable_output(dst_img, pre_sig, image_like=src_img))

        manifest_file = self.tmp_path / "manifest.csv"
        manifest_file.write_text(
            f"enabled,source_reference,color_reference,qc_status\ntrue,{src_img},{dst_img},ok\n",
            encoding="utf-8",
        )

        cli_args = parser.parse_args([
            "--manifest", str(manifest_file),
            "--workflow", str(wf_file),
            "--film-id", "film_gate_test",
            "--confidence-threshold", "0.80",
            "--dry-run",
        ])

        # Mock ColorBibleService to return low-confidence assignment (0.65)
        mock_service = MagicMock()
        mock_service.get_snapshot_hash.return_value = "sha256:v1"
        mock_service.get_shot_palette_assignments.return_value = [
            {
                "entity_type": "character",
                "entity_id": "alice",
                "entity": {"name": "Alice"},
                "assignment": {"approval_status": "approved", "confidence": 0.65},
                "color_references": [],
            }
        ]

        with (
            patch("ai_remaster_gui.color_bible_service.ColorBibleService", return_value=mock_service),
            patch.object(qwen, "colorize_reference_shot") as mock_ddcolor,
            patch.object(qwen, "render") as mock_render,
        ):
            exit_code = qwen.main_with_args(cli_args)
            self.assertEqual(exit_code, 0)
            # Inference was cleanly skipped without executing rendering
            mock_ddcolor.assert_not_called()
            mock_render.assert_not_called()
            # Color Bible DB record was updated
            mock_service.update_shot_qc_status.assert_called_with("film_gate_test", 0, "needs_review")

        # Manifest CSV row was updated to 'needs_review'
        manifest_rows = read_manifest(manifest_file)
        self.assertEqual(len(manifest_rows), 1)
        self.assertEqual(manifest_rows[0].get("qc_status"), "needs_review")


# ===========================================================================
# Criterion (d): Qwen3-VL prompt enrichment includes structured Color Bible constraints
# ===========================================================================


class TestCriterionDPromptEnrichment(unittest.TestCase):
    """Validate prompt enrichment formatting with character, costume, lighting, RGB, and CIE-LAB stats."""

    def test_prompt_enrichment_structured_constraints(self) -> None:
        """Verify character names, costume IDs, lighting, RGB swatches, and LAB statistics in prompt."""
        base_prompt = "Colorize 1930s animated cartoon scene."
        entities = [
            {
                "entity_type": "character",
                "entity": {"name": "Mickey"},
                "variant": {"costume_id": "red_shorts", "lighting_context": "bright daylight"},
                "color_references": [
                    {
                        "rgb_palette_json": json.dumps([{"hex": "#D32F2F", "rgb": [211, 47, 47]}]),
                        "lab_stats_json": json.dumps({
                            "L": [20.0, 30.0, 45.2, 60.0, 70.0],
                            "a": [30.0, 40.0, 58.1, 65.0, 70.0],
                            "b": [10.0, 20.0, 39.4, 45.0, 50.0],
                        }),
                    }
                ],
            }
        ]

        enriched = PromptEnricher.enrich_prompt(
            base_prompt=base_prompt,
            entities=entities,
            lighting_context="warm afternoon daylight",
        )

        # Expected structured constraint clause:
        # [Color Bible Constraints: Character Mickey (costume: red_shorts, target RGB: #D32F2F / [211, 47, 47], LAB L*: 45.2, a*: 58.1, b*: 39.4); Scene Lighting: warm afternoon daylight].
        self.assertTrue(enriched.startswith("Colorize 1930s animated cartoon scene. [Color Bible Constraints:"))
        self.assertIn("Character Mickey", enriched)
        self.assertIn("costume: red_shorts", enriched)
        self.assertIn("target RGB: #D32F2F / [211, 47, 47]", enriched)
        self.assertIn("LAB L*: 45.2, a*: 58.1, b*: 39.4", enriched)
        self.assertIn("Scene Lighting: warm afternoon daylight", enriched)

    def test_prompt_enrichment_multiple_entities(self) -> None:
        """Verify enrichment formats multiple characters, costumes, and objects cleanly."""
        entities = [
            {
                "entity_type": "character",
                "entity": {"name": "Alice"},
                "variant": {"costume_id": "blue_coat"},
                "hex": "#1E3A8A",
                "rgb": [30, 58, 138],
                "lab": {"L": 27.5, "a": 12.0, "b": -45.2},
            },
            {
                "entity_type": "object",
                "entity": {"name": "Vintage Car"},
                "variant": {"costume_id": "ruby_red"},
                "hex": "#991B1B",
                "rgb": [153, 27, 27],
            },
        ]

        enriched = enrich_prompt(base_prompt="Street scene.", entities=entities)
        self.assertIn("Character Alice (costume: blue_coat, target RGB: #1E3A8A / [30, 58, 138], LAB L*: 27.5, a*: 12.0, b*: -45.2)", enriched)
        self.assertIn("Object Vintage Car (costume: ruby_red, target RGB: #991B1B / [153, 27, 27])", enriched)

    def test_prompt_enrichment_empty_entities_preserves_base_prompt(self) -> None:
        """Empty entities list and lighting context return original prompt untouched."""
        self.assertEqual(enrich_prompt("Monochrome frame.", []), "Monochrome frame.")
        self.assertEqual(enrich_prompt("  Monochrome frame.  ", []), "  Monochrome frame.  ")

    def test_prompt_enrichment_empty_base_prompt(self) -> None:
        """When base prompt is empty, returns constraint clause only."""
        entities = [{"entity_type": "character", "name": "Donald"}]
        result = enrich_prompt("", entities=entities)
        self.assertEqual(result, "[Color Bible Constraints: Character Donald].")


# ===========================================================================
# Criterion (e): Two-stage DDColor draft -> Qwen-2511 escalation logic
# ===========================================================================


class TestCriterionETwoStageRouting(unittest.TestCase):
    """Validate backend selection router: auto mode, escalation flags, and row overrides."""

    def test_auto_mode_unconstrained_shot_routes_to_ddcolor(self) -> None:
        """In reference_runner='auto', an unconstrained shot with no Color Bible entities routes to 'ddcolor'."""
        row = {"source_reference": "shot_01.png"}
        backend = TwoStageReferenceRouter.select_backend(
            row=row,
            has_color_bible_entities=False,
            reference_runner="auto",
            default_guided_backend="qwen2511",
        )
        self.assertEqual(backend, "ddcolor")

    def test_auto_mode_shot_with_color_bible_entities_routes_to_qwen2511(self) -> None:
        """In reference_runner='auto', a shot with assigned Color Bible entities routes to 'qwen2511'."""
        row = {"source_reference": "shot_01.png"}
        backend = TwoStageReferenceRouter.select_backend(
            row=row,
            has_color_bible_entities=True,
            reference_runner="auto",
            default_guided_backend="qwen2511",
        )
        self.assertEqual(backend, "qwen2511")

    def test_auto_mode_escalate_flags_route_to_qwen2511(self) -> None:
        """In reference_runner='auto', escalate_guided=true or escalate_qwen=true routes to 'qwen2511'."""
        for flag_key in ("escalate_guided", "escalate_qwen"):
            for val in ("true", "True", "1", "yes", "YES"):
                row = {"source_reference": "shot_01.png", flag_key: val}
                backend = select_backend(
                    row=row,
                    has_color_bible_entities=False,
                    reference_runner="auto",
                    default_guided_backend="qwen2511",
                )
                self.assertEqual(backend, "qwen2511", f"Failed for {flag_key}={val}")

    def test_per_row_reference_backend_override_takes_absolute_precedence(self) -> None:
        """Manifest column 'reference_backend' overrides runner settings unconditionally."""
        # 1. Force ddcolor even when runner is guided_only and entities exist
        row_ddcolor = {"reference_backend": "ddcolor"}
        backend1 = select_backend(
            row=row_ddcolor,
            has_color_bible_entities=True,
            reference_runner="guided_only",
        )
        self.assertEqual(backend1, "ddcolor")

        # 2. Custom backend
        row_custom = {"reference_backend": "custom_neural_model"}
        backend2 = select_backend(
            row=row_custom,
            has_color_bible_entities=False,
            reference_runner="ddcolor_only",
        )
        self.assertEqual(backend2, "custom_neural_model")

    def test_strict_runner_modes(self) -> None:
        """Strict single-backend modes ignore auto escalation unless overridden per row."""
        row = {"escalate_guided": "true"}
        # ddcolor_only forces ddcolor despite escalation flag
        self.assertEqual(select_backend(row, has_color_bible_entities=True, reference_runner="ddcolor_only"), "ddcolor")
        # guided_only forces guided model even without entities
        self.assertEqual(select_backend({}, has_color_bible_entities=False, reference_runner="guided_only"), "qwen2511")


# ===========================================================================
# Criterion (f): Cache signature invalidates ONLY shots affected by a Color Bible update
# ===========================================================================


class TestCriterionFScopedCacheInvalidation(unittest.TestCase):
    """Validate scoped cache keys: Costume B update invalidates Shot 100 but preserves Shot 520 cache hit."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir_obj.name)

        self.src_100 = self.tmp_path / "shot_0100_src.png"
        self.out_100 = self.tmp_path / "shot_0100_clr.png"
        self.src_520 = self.tmp_path / "shot_0520_src.png"
        self.out_520 = self.tmp_path / "shot_0520_clr.png"

        for p in (self.src_100, self.out_100, self.src_520, self.out_520):
            Image.new("RGB", (32, 32), (100, 100, 100)).save(p)

        self.wf_path = self.tmp_path / "wf.json"
        self.wf_path.write_text("{}", encoding="utf-8")

        self.parser = qwen.build_parser()
        self.args = self.parser.parse_args(["--manifest", "x.csv", "--workflow", str(self.wf_path)])

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_cache_signature_invalidates_only_affected_shot(self) -> None:
        """Updating Costume B invalidates Shot 100 cache, leaving Shot 520 as a valid cache hit."""
        # Initial Color Bible state:
        # Shot 100 uses Costume B ('ref_b_id' with hash 'hash_costume_b_v1')
        # Shot 520 uses Costume A ('ref_a_id' with hash 'hash_costume_a_v1')
        ref_b_hashes_v1 = {"ref_b_id": "hash_costume_b_v1"}
        ref_a_hashes = {"ref_a_id": "hash_costume_a_v1"}

        sig_100_v1 = qwen.signature(
            self.args,
            workflow_path=self.wf_path,
            source_path=self.src_100,
            prompt="Prompt for shot 100",
            shot_index=100,
            retrieved_ref_hashes=ref_b_hashes_v1,
            retrieved_ref_ids=["ref_b_id"],
            color_bible_version="sha256:color_bible_v1",
        )
        write_signature(self.out_100, sig_100_v1)

        sig_520_v1 = qwen.signature(
            self.args,
            workflow_path=self.wf_path,
            source_path=self.src_520,
            prompt="Prompt for shot 520",
            shot_index=520,
            retrieved_ref_hashes=ref_a_hashes,
            retrieved_ref_ids=["ref_a_id"],
            color_bible_version="sha256:color_bible_v1",
        )
        write_signature(self.out_520, sig_520_v1)

        # Both shots initially evaluate as cache hits
        self.assertTrue(resumable_output(self.out_100, sig_100_v1, image_like=self.src_100))
        self.assertTrue(resumable_output(self.out_520, sig_520_v1, image_like=self.src_520))

        # --- UPDATE EVENT: Costume B reference is edited/re-registered (new hash 'hash_costume_b_v2') ---
        # Global Color Bible snapshot hash advances to 'sha256:color_bible_v2'
        ref_b_hashes_v2 = {"ref_b_id": "hash_costume_b_v2"}

        new_sig_100 = qwen.signature(
            self.args,
            workflow_path=self.wf_path,
            source_path=self.src_100,
            prompt="Prompt for shot 100",
            shot_index=100,
            retrieved_ref_hashes=ref_b_hashes_v2,
            retrieved_ref_ids=["ref_b_id"],
            color_bible_version="sha256:color_bible_v2",
        )

        new_sig_520 = qwen.signature(
            self.args,
            workflow_path=self.wf_path,
            source_path=self.src_520,
            prompt="Prompt for shot 520",
            shot_index=520,
            retrieved_ref_hashes=ref_a_hashes,  # Unchanged
            retrieved_ref_ids=["ref_a_id"],
            color_bible_version="sha256:color_bible_v2",  # Global version changed
        )

        # 1. Shot 100 scoped cache key MUST change -> Cache Miss / Invalidation
        self.assertNotEqual(sig_100_v1["scoped_cache_key"], new_sig_100["scoped_cache_key"])
        self.assertFalse(resumable_output(self.out_100, new_sig_100, image_like=self.src_100))

        # 2. Shot 520 scoped cache key MUST remain identical -> Cache Hit Preserved
        self.assertEqual(sig_520_v1["scoped_cache_key"], new_sig_520["scoped_cache_key"])
        self.assertTrue(resumable_output(self.out_520, new_sig_520, image_like=self.src_520))


# ===========================================================================
# Criterion (g): Commercial Model Guard & compliance_report.json provenance
# ===========================================================================


class TestCriterionGCommercialPolicyAndComplianceReport(unittest.TestCase):
    """Validate Commercial Model Guard failure-closed policy and compliance report provenance."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_non_commercial_model_fails_closed_under_commercial_guard(self) -> None:
        """Non-commercial models fail closed and raise CommercialPolicyError when guard is active."""
        settings = {
            "global": {
                "commercial_model_guard": "true",
                "non_commercial_benchmark_override": "false",
            }
        }

        # Hard-blocked non-commercial research models
        for prohibited_model in ("qwen-image-2.1", "propainter", "sam3.1"):
            decision = evaluate_component(
                component_id=prohibited_model,
                component_type="checkpoint",
                settings=settings,
            )
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.status, "hard_blocked")

            with self.assertRaises(CommercialPolicyError):
                enforce_component(prohibited_model, settings=settings)

    def test_record_reference_compliance_writes_provenance(self) -> None:
        """record_reference_compliance records valid JSON with color_bible_version and retrieved_ref_ids."""
        output_file = self.tmp_path / "reference_005.png"
        output_file.write_bytes(b"image")

        settings = {"global": {"qwen_2511_audited": "true"}}
        report_path = record_reference_compliance(
            output_path=output_file,
            model_id="qwen-image-edit-2511",
            component_type="checkpoint",
            color_bible_version="sha256:cb_snapshot_9988",
            retrieved_ref_ids=["ref_alice_coat", "ref_mickey_shorts"],
            settings=settings,
            seed=42,
            execution_time_ms=1250.5,
        )

        self.assertTrue(report_path.is_file())
        data = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertTrue(data["overall_commercial_compliant"])
        self.assertEqual(len(data["runs"]), 1)

        run = data["runs"][0]
        self.assertEqual(run["stage"], "color_reference")
        self.assertEqual(run["color_bible_version"], "sha256:cb_snapshot_9988")
        self.assertEqual(run["metadata"]["retrieved_ref_ids"], ["ref_alice_coat", "ref_mickey_shorts"])
        self.assertEqual(run["metadata"]["execution_time_ms"], 1250.5)
        self.assertEqual(run["seeds"], {"reference_seed": 42})


if __name__ == "__main__":
    unittest.main()
