"""End-to-End CPU Integration Smoke Test for Commercial Remastering Pipeline.

Verifies the entire deterministic 100%-OSI commercial remastering integration chain:
1. SQLite Color Bible DB setup (films, entities, variants, references, shot assignments).
2. Manifest CSV formatting with metadata comments and shot specifications.
3. Semantic matching and two-stage reference runner router logic with review gates.
4. SAM2 RLE mask archiving and metadata tagging.
5. Deterministic OSI Masked LAB colorization with Color Bible anchor resolution,
   strict L* luminance lock, and unmasked optical flow fallback.
6. EU AI Act Article 50(4) compliant audit reporting with cryptographic snapshot hashing.

Runs 100% on CPU without GPU, network, or ffmpeg requirements.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Ensure repository root and scripts directory are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ai_remaster_gui.color_bible_db import AppearanceVariant, Character, Film
from ai_remaster_gui.color_bible_service import ColorBibleService
from ai_remaster_gui.compliance import append_compliance_record, load_compliance_report
from ai_remaster_gui.manifests import read_manifest_details
from ai_remaster_gui.rle_masks import (
    MaskObjectMetadata,
    ShotMaskArchive,
    load_shot_masks_npz,
    save_shot_masks_npz,
)
from ai_remaster_gui.semantic_matching import (
    ColorBibleMatcher,
    select_backend,
)
from qwen_colorize_references import reference_images_for
from scripts.osi_masked_lab import (
    apply_strict_l_lock,
    colorize_shot_osi_masked_lab,
    resolve_color_bible_anchors,
)


class TestEndToEndCommercialPipelineWiring(unittest.TestCase):
    """Rigorous end-to-end integration test verifying the commercial pipeline chain on CPU."""

    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp_dir_obj.name)
        self.db_path = self.tmp_path / "color_bible.db"
        self.output_dir = self.tmp_path / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir = self.tmp_path / "masks"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.film_id = "e2e_test_film"

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def _setup_color_bible_scenario(self) -> tuple[Path, str]:
        """Helper to construct synthetic Color Bible DB with Alice and Costume_A."""
        with ColorBibleService(db_path=self.db_path) as service:
            service.create_film(self.film_id, title="E2E Commercial Test Film")
            char = Character(character_id="alice", film_id=self.film_id, name="Alice")
            service.repo.upsert_character(char)
            var = AppearanceVariant(variant_id="costume_a", character_id="alice", costume_id="Costume_A")
            service.repo.upsert_variant(var)

            # Generate synthetic warm RGB image (64x64, L=50, a=35, b=25 in CIE-LAB)
            ref_path = self.tmp_path / "alice_costume_a_ref.png"
            ref_gray = np.full((64, 64), 128, dtype=np.uint8)
            ref_chroma = np.zeros((64, 64, 2), dtype=np.float32)
            ref_chroma[..., 0] = 35.0  # a*
            ref_chroma[..., 1] = 25.0  # b*
            ref_rgb = apply_strict_l_lock(ref_gray, ref_chroma)
            Image.fromarray(ref_rgb).save(ref_path)

            # Store and approve reference
            service.store_reference(
                film_id=self.film_id,
                entity_type="character",
                entity_id="alice",
                variant_id="costume_a",
                keyframe_path=ref_path,
                approval_status="approved",
            )

            # Assign Costume_A to Shot 0 (confidence=0.95) and Shot 1 (confidence=0.60)
            service.assign_palette_ranges(
                film_id=self.film_id,
                entity_type="character",
                entity_id="alice",
                ranges=[(0, 0)],
                variant_id="costume_a",
                confidence=0.95,
                approval_status="approved",
            )
            service.assign_palette_ranges(
                film_id=self.film_id,
                entity_type="character",
                entity_id="alice",
                ranges=[(1, 1)],
                variant_id="costume_a",
                confidence=0.60,
                approval_status="approved",
            )

            snapshot_hash = service.get_snapshot_hash(self.film_id)
        return ref_path, snapshot_hash

    def test_color_bible_setup_and_snapshot_hash(self) -> None:
        """Step 1.a: Verify SQLite Color Bible DB creation, reference storage, and snapshot hash."""
        ref_path, snapshot_hash = self._setup_color_bible_scenario()
        self.assertTrue(self.db_path.is_file())
        self.assertTrue(ref_path.is_file())
        self.assertTrue(snapshot_hash.startswith("sha256:"))

        with ColorBibleService(db_path=self.db_path) as service:
            film = service.get_film(self.film_id)
            self.assertIsNotNone(film)
            self.assertEqual(film.film_id, self.film_id)

            char = service.get_character("alice")
            self.assertIsNotNone(char)
            self.assertEqual(char.name, "Alice")

            var = service.get_variant("costume_a")
            self.assertIsNotNone(var)
            self.assertEqual(var.costume_id, "Costume_A")

            # Check shot palette assignments
            shot0_asgns = service.get_shot_palette_assignments(self.film_id, 0)
            self.assertEqual(len(shot0_asgns), 1)
            self.assertAlmostEqual(shot0_asgns[0]["assignment"]["confidence"], 0.95)
            self.assertEqual(len(shot0_asgns[0]["color_references"]), 1)
            self.assertEqual(shot0_asgns[0]["color_references"][0]["approval_status"], "approved")

            shot1_asgns = service.get_shot_palette_assignments(self.film_id, 1)
            self.assertEqual(len(shot1_asgns), 1)
            self.assertAlmostEqual(shot1_asgns[0]["assignment"]["confidence"], 0.60)

    def test_synthetic_manifest_and_semantic_matching_review_gate(self) -> None:
        """Step 1.b & 1.c: Verify manifest parsing, review gate triggers, and reference routing."""
        ref_path, _ = self._setup_color_bible_scenario()

        # Write synthetic 2-shot manifest CSV
        manifest_path = self.tmp_path / "shots_manifest.csv"
        manifest_text = (
            f"# source_video=test.mp4\n"
            f"# color_bible_film_id={self.film_id}\n"
            f"# color_bible_db={self.db_path.as_posix()}\n"
            "index,start,end,start_frame,end_frame,source_reference,color_reference\n"
            f"0,00:00:00.000,00:00:00.416,0,9,{ref_path.as_posix()},{ref_path.as_posix()}\n"
            f"1,00:00:00.416,00:00:00.833,10,19,{ref_path.as_posix()},{ref_path.as_posix()}\n"
        )
        manifest_path.write_text(manifest_text, encoding="utf-8")

        source_video, fields, rows = read_manifest_details(manifest_path)
        self.assertEqual(source_video, "test.mp4")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["start_frame"], "0")
        self.assertEqual(rows[1]["end_frame"], "19")

        with ColorBibleService(db_path=self.db_path) as service:
            # Shot 0: High confidence (0.95 >= 0.80) -> needs_review is False
            match_shot0 = ColorBibleMatcher.resolve_references(
                rows=rows,
                index=0,
                count=1,
                film_id=self.film_id,
                service=service,
                confidence_threshold=0.80,
            )
            self.assertFalse(match_shot0.needs_review)
            self.assertEqual(match_shot0.review_reasons, [])
            self.assertEqual(len(match_shot0.reference_paths), 1)
            self.assertEqual(match_shot0.reference_paths[0].resolve(), ref_path.resolve())

            # Shot 1: Low confidence (0.60 < 0.80) -> needs_review is True
            match_shot1 = ColorBibleMatcher.resolve_references(
                rows=rows,
                index=1,
                count=1,
                film_id=self.film_id,
                service=service,
                confidence_threshold=0.80,
            )
            self.assertTrue(match_shot1.needs_review)
            self.assertTrue(any("0.60 < 0.80" in r for r in match_shot1.review_reasons))
            self.assertEqual(len(match_shot1.reference_paths), 1)

            # Test reference_images_for convenience API
            refs_shot0 = reference_images_for(
                rows=rows,
                index=0,
                count=1,
                film_id=self.film_id,
                color_bible_service=service,
                confidence_threshold=0.80,
            )
            self.assertEqual(len(refs_shot0), 1)
            self.assertEqual(refs_shot0[0].resolve(), ref_path.resolve())

        # Router logic verification
        # 1. Auto mode with Color Bible entities -> guided (qwen2511)
        self.assertEqual(
            select_backend(rows[0], has_color_bible_entities=True, reference_runner="auto"),
            "qwen2511",
        )
        # 2. Auto mode without Color Bible entities -> ddcolor
        self.assertEqual(
            select_backend(rows[0], has_color_bible_entities=False, reference_runner="auto"),
            "ddcolor",
        )
        # 3. Explicit mode constraints
        self.assertEqual(
            select_backend(rows[0], has_color_bible_entities=True, reference_runner="ddcolor_only"),
            "ddcolor",
        )
        self.assertEqual(
            select_backend(rows[0], has_color_bible_entities=False, reference_runner="guided_only"),
            "qwen2511",
        )
        # 4. Explicit row override wins
        override_row = dict(rows[0])
        override_row["reference_backend"] = "custom_backend_v2"
        self.assertEqual(
            select_backend(override_row, has_color_bible_entities=True, reference_runner="auto"),
            "custom_backend_v2",
        )

    def test_mask_archive_creation_and_metadata(self) -> None:
        """Step 1.d: Verify synthetic .masks.npz creation with character/variant metadata."""
        archive_path = self.masks_dir / "shot_0000.masks.npz"
        archive = ShotMaskArchive(shot_id="shot_0000", frame_count=10, height=64, width=64)
        obj_meta = MaskObjectMetadata(
            obj_id=1,
            name="Alice",
            entity_id="alice",
            variant_id="costume_a",
            entity_type="character",
        )
        archive.add_object(obj_meta)

        # Center 32x32 binary mask for frames 0..9
        for t in range(10):
            m = np.zeros((64, 64), dtype=np.uint8)
            m[16:48, 16:48] = 255
            archive.set_mask(t, 1, m)

        save_shot_masks_npz(archive, archive_path)
        self.assertTrue(archive_path.is_file())

        # Verify round-trip load
        loaded = load_shot_masks_npz(archive_path)
        self.assertEqual(loaded.shot_id, "shot_0000")
        self.assertEqual(loaded.frame_count, 10)
        self.assertEqual(loaded.height, 64)
        self.assertEqual(loaded.width, 64)
        self.assertEqual(len(loaded.objects), 1)

        loaded_obj = loaded.get_object(1)
        self.assertIsNotNone(loaded_obj)
        self.assertEqual(loaded_obj.name, "Alice")
        self.assertEqual(loaded_obj.entity_id, "alice")
        self.assertEqual(loaded_obj.variant_id, "costume_a")
        self.assertEqual(loaded_obj.entity_type, "character")

        for t in range(10):
            self.assertTrue(loaded.has_mask(t, 1))
            mask_arr = loaded.get_mask(t, 1)
            self.assertIsNotNone(mask_arr)
            self.assertEqual(mask_arr.shape, (64, 64))
            self.assertEqual(np.count_nonzero(mask_arr), 32 * 32)
            self.assertEqual(mask_arr[32, 32], 255)
            self.assertEqual(mask_arr[0, 0], 0)

    def test_osi_masked_lab_colorization_and_unmasked_fallback(self) -> None:
        """Step 1.e: Verify anchor resolution from DB, strict L* lock, masked transfer, and unmasked fallback."""
        ref_path, _ = self._setup_color_bible_scenario()

        # Create mask archive for Shot 0
        archive_path = self.masks_dir / "shot_0000.masks.npz"
        archive = ShotMaskArchive(shot_id="shot_0000", frame_count=10, height=64, width=64)
        archive.add_object(
            MaskObjectMetadata(
                obj_id=1,
                name="Alice",
                entity_id="alice",
                variant_id="costume_a",
                entity_type="character",
            )
        )
        for t in range(10):
            m = np.zeros((64, 64), dtype=np.uint8)
            m[16:48, 16:48] = 255
            archive.set_mask(t, 1, m)
        save_shot_masks_npz(archive, archive_path)

        # 1. Verify anchor resolution directly from SQLite DB
        resolved_anchors = resolve_color_bible_anchors(
            color_bible_db=self.db_path,
            film_id=self.film_id,
            shot_index=0,
            mask_archive=archive,
        )
        self.assertIn("alice", resolved_anchors)
        self.assertIn("costume_a", resolved_anchors)
        self.assertIn(1, resolved_anchors)
        target_a, target_b = resolved_anchors["alice"]
        self.assertAlmostEqual(target_a, 35.0, delta=5.0)
        self.assertAlmostEqual(target_b, 25.0, delta=5.0)

        # 2. Synthetic grayscale frames (64x64, 10 frames)
        np.random.seed(42)
        gray_frames = [
            np.clip(100 + t * 3 + np.random.randint(-5, 6, (64, 64)), 0, 255).astype(np.uint8)
            for t in range(10)
        ]

        # 1 color keyframe at frame 0 (neutral / zero chroma)
        keyframe_0 = cv2.cvtColor(gray_frames[0], cv2.COLOR_GRAY2RGB)

        # 3. Execute colorize_shot_osi_masked_lab for Shot 0 with mask archive & Color Bible DB
        shot0_output_path = self.output_dir / "shot_0_color.mp4"
        shot0_out_frames = colorize_shot_osi_masked_lab(
            source=gray_frames,
            keyframes={0: keyframe_0},
            mask_archive=archive_path,
            output_path=shot0_output_path,
            color_bible_db=self.db_path,
            film_id=self.film_id,
            shot_index=0,
        )

        self.assertEqual(len(shot0_out_frames), 10)
        self.assertTrue(shot0_output_path.is_file())
        self.assertGreater(shot0_output_path.stat().st_size, 0)

        # Verify strict L* lock across all 10 frames (|L*(out) - L*(orig)| < 1.0)
        for t_idx in range(10):
            orig_gray = gray_frames[t_idx]
            out_rgb = shot0_out_frames[t_idx]

            orig_rgb_f = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            orig_L = cv2.cvtColor(orig_rgb_f, cv2.COLOR_RGB2Lab)[..., 0]

            out_rgb_f = out_rgb.astype(np.float32) / 255.0
            out_L = cv2.cvtColor(out_rgb_f, cv2.COLOR_RGB2Lab)[..., 0]

            max_l_diff = float(np.max(np.abs(out_L - orig_L)))
            self.assertLess(
                max_l_diff,
                1.0,
                f"Frame {t_idx} exceeded strict L* tolerance: {max_l_diff:.4f} >= 1.0",
            )

        # Verify chromaticity transfer into the masked region (center 32x32)
        # Check frame 5 (mid-shot):
        f5_rgb_f = shot0_out_frames[5].astype(np.float32) / 255.0
        f5_lab = cv2.cvtColor(f5_rgb_f, cv2.COLOR_RGB2Lab)
        masked_a = f5_lab[16:48, 16:48, 1]
        masked_b = f5_lab[16:48, 16:48, 2]
        unmasked_a = f5_lab[:10, :10, 1]
        unmasked_b = f5_lab[:10, :10, 2]

        # In masked region, chroma should reflect Alice's Costume_A warm anchor
        self.assertGreater(float(np.median(masked_a)), 20.0)
        self.assertGreater(float(np.median(masked_b)), 15.0)

        # In unmasked background region, chroma should remain neutral
        self.assertLess(abs(float(np.median(unmasked_a))), 3.0)
        self.assertLess(abs(float(np.median(unmasked_b))), 3.0)

        # 4. Test unmasked fallback for Shot 1 (no mask archive provided)
        shot1_output_path = self.output_dir / "shot_1_unmasked_color.mp4"
        shot1_out_frames = colorize_shot_osi_masked_lab(
            source=gray_frames,
            keyframes={0: keyframe_0},
            mask_archive=None,  # Unmasked fallback!
            output_path=shot1_output_path,
            color_bible_db=self.db_path,
            film_id=self.film_id,
            shot_index=1,
        )

        self.assertEqual(len(shot1_out_frames), 10)
        self.assertTrue(shot1_output_path.is_file())
        self.assertGreater(shot1_output_path.stat().st_size, 0)

        # Verify unmasked fallback also strictly preserves L*
        for t_idx in range(10):
            orig_gray = gray_frames[t_idx]
            out_rgb = shot1_out_frames[t_idx]
            orig_rgb_f = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
            orig_L = cv2.cvtColor(orig_rgb_f, cv2.COLOR_RGB2Lab)[..., 0]
            out_rgb_f = out_rgb.astype(np.float32) / 255.0
            out_L = cv2.cvtColor(out_rgb_f, cv2.COLOR_RGB2Lab)[..., 0]
            max_l_diff = float(np.max(np.abs(out_L - orig_L)))
            self.assertLess(max_l_diff, 1.0)

    def test_compliance_report_provenance_and_eu_ai_act_disclosure(self) -> None:
        """Step 1.f: Verify compliance reporting, snapshot hash linkage, and EU AI Act Art. 50 disclosure."""
        _, snapshot_hash = self._setup_color_bible_scenario()
        settings = {
            "global": {
                "commercial_model_guard": "true",
                "non_commercial_benchmark_override": "false",
            }
        }

        # Record Stage 1: references run
        append_compliance_record(
            output_dir=self.output_dir,
            stage="references",
            models=[{"model_id": "ddcolor", "component_type": "checkpoint"}],
            settings=settings,
            seeds={"reference_seed": 42},
            color_bible_version=snapshot_hash,
        )

        # Record Stage 2: colour run (deterministic OSI Masked LAB backend)
        osi_script_path = str(REPO_ROOT / "scripts" / "osi_masked_lab.py")
        append_compliance_record(
            output_dir=self.output_dir,
            stage="colour",
            models=[{
                "model_id": "osi_masked_lab",
                "component_type": "backend",
                "file_path": osi_script_path,
            }],
            settings=settings,
            seeds={"color_seed": 100},
            color_bible_version=snapshot_hash,
        )

        report_file = self.output_dir / "compliance_report.json"
        self.assertTrue(report_file.is_file())

        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Assert report schema and compliance
        self.assertEqual(data["schema_version"], "1.0.0")
        self.assertTrue(data["overall_commercial_compliant"])
        self.assertEqual(len(data["runs"]), 2)

        # Verify Color Bible snapshot hash linkage
        for run in data["runs"]:
            self.assertEqual(run["color_bible_version"], snapshot_hash)
            self.assertTrue(run["is_commercial_compliant"])

            # Verify EU AI Act Article 50 Disclosure block
            disclosure = run["eu_ai_act_article_50_disclosure"]
            self.assertEqual(disclosure["role"], "deployer")
            self.assertTrue(disclosure["ai_generated_or_manipulated"])
            self.assertFalse(disclosure["machine_readable_watermark_or_metadata"])
            self.assertEqual(disclosure["compliance_effective_date"], "2026-08-02")
            self.assertEqual(disclosure["applicable_articles"], ["Art. 50(4)"])
            self.assertIn("EU AI Act Article 50(4)", disclosure["disclosure_notice"])

    def test_end_to_end_full_chain_pipeline_integration(self) -> None:
        """Full end-to-end integration: DB setup -> Manifest -> Semantic Match -> OSI Colorize -> Compliance."""
        # 1. DB Setup & Reference Creation
        ref_path, snapshot_hash = self._setup_color_bible_scenario()

        # 2. Manifest Creation & Parsing
        manifest_path = self.tmp_path / "production_shots.csv"
        manifest_path.write_text(
            f"# source_video=feature_film.mp4\n"
            f"# color_bible_film_id={self.film_id}\n"
            f"# color_bible_db={self.db_path.as_posix()}\n"
            "index,start,end,start_frame,end_frame,source_reference,color_reference\n"
            f"0,00:00:00.000,00:00:00.416,0,9,{ref_path.as_posix()},{ref_path.as_posix()}\n"
            f"1,00:00:00.416,00:00:00.833,10,19,{ref_path.as_posix()},{ref_path.as_posix()}\n",
            encoding="utf-8",
        )
        _, _, rows = read_manifest_details(manifest_path)

        # 3. Semantic Reference Resolution with Color Bible
        with ColorBibleService(db_path=self.db_path) as service:
            res_shot0 = ColorBibleMatcher.resolve_references(
                rows=rows,
                index=0,
                count=1,
                film_id=self.film_id,
                service=service,
                confidence_threshold=0.80,
            )
            self.assertFalse(res_shot0.needs_review)

            res_shot1 = ColorBibleMatcher.resolve_references(
                rows=rows,
                index=1,
                count=1,
                film_id=self.film_id,
                service=service,
                confidence_threshold=0.80,
            )
            self.assertTrue(res_shot1.needs_review)

        # 4. SAM2 Mask Archive Creation
        archive_path = self.masks_dir / "shot_0000.masks.npz"
        archive = ShotMaskArchive("shot_0000", frame_count=10, height=64, width=64)
        archive.add_object(
            MaskObjectMetadata(
                obj_id=1,
                name="Alice",
                entity_id="alice",
                variant_id="costume_a",
                entity_type="character",
            )
        )
        for t in range(10):
            m = np.zeros((64, 64), dtype=np.uint8)
            m[16:48, 16:48] = 255
            archive.set_mask(t, 1, m)
        save_shot_masks_npz(archive, archive_path)

        # 5. Colorize Shot 0 (Masked) and Shot 1 (Unmasked Fallback)
        gray_frames = [np.full((64, 64), 110 + i * 2, dtype=np.uint8) for i in range(10)]
        key_0 = np.full((64, 64, 3), 110, dtype=np.uint8)

        out_shot0 = colorize_shot_osi_masked_lab(
            source=gray_frames,
            keyframes={0: key_0},
            mask_archive=archive_path,
            output_path=self.output_dir / "shot_0_final.mp4",
            color_bible_db=self.db_path,
            film_id=self.film_id,
            shot_index=0,
        )
        self.assertEqual(len(out_shot0), 10)

        out_shot1 = colorize_shot_osi_masked_lab(
            source=gray_frames,
            keyframes={0: key_0},
            mask_archive=None,
            output_path=self.output_dir / "shot_1_final.mp4",
            color_bible_db=self.db_path,
            film_id=self.film_id,
            shot_index=1,
        )
        self.assertEqual(len(out_shot1), 10)

        # 6. Record and Verify Compliance
        append_compliance_record(
            output_dir=self.output_dir,
            stage="colour",
            models=[{"model_id": "osi_masked_lab", "component_type": "backend"}],
            settings={"global": {"commercial_model_guard": "true"}},
            color_bible_version=snapshot_hash,
        )
        report = load_compliance_report(self.output_dir / "compliance_report.json")
        self.assertTrue(report["overall_commercial_compliant"])
        self.assertEqual(report["runs"][0]["color_bible_version"], snapshot_hash)
        self.assertEqual(
            report["runs"][0]["eu_ai_act_article_50_disclosure"]["applicable_articles"],
            ["Art. 50(4)"],
        )


if __name__ == "__main__":
    unittest.main()
