from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ai_remaster_gui.color_bible_db import SQLiteColorBibleRepository
from ai_remaster_gui.color_bible_service import ColorBibleService, parse_time_seconds
from ai_remaster_gui.color_science import (
    compute_lab_percentiles,
    extract_rgb_palette,
    filter_pixels_by_mask,
    linear_to_xyz,
    load_image_rgb,
    load_mask,
    srgb_to_lab,
    srgb_to_linear,
    xyz_to_lab,
)
from ai_remaster_gui.manifests import write_manifest_details


class TestColorScience:
    """Test mathematical color space transformations and extraction algorithms."""

    def test_synthetic_color_conversions_cie_lab(self) -> None:
        """Verify sRGB -> CIE XYZ -> CIE-LAB against standard D65 ground truth."""
        # Pure White
        white = np.array([255, 255, 255], dtype=np.uint8)
        lab_white = srgb_to_lab(white)
        assert pytest.approx(lab_white[0], abs=0.05) == 100.0
        assert pytest.approx(lab_white[1], abs=0.05) == 0.0
        assert pytest.approx(lab_white[2], abs=0.05) == 0.0

        # Pure Black
        black = np.array([0, 0, 0], dtype=np.uint8)
        lab_black = srgb_to_lab(black)
        assert pytest.approx(lab_black[0], abs=0.05) == 0.0
        assert pytest.approx(lab_black[1], abs=0.05) == 0.0
        assert pytest.approx(lab_black[2], abs=0.05) == 0.0

        # Pure Red: sRGB [255, 0, 0] -> Lab [53.24, 80.09, 67.20]
        red = np.array([255, 0, 0], dtype=np.uint8)
        lab_red = srgb_to_lab(red)
        assert pytest.approx(lab_red[0], abs=0.1) == 53.24
        assert pytest.approx(lab_red[1], abs=0.1) == 80.09
        assert pytest.approx(lab_red[2], abs=0.1) == 67.20

        # Pure Green: sRGB [0, 255, 0] -> Lab [87.73, -86.18, 83.18]
        green = np.array([0, 255, 0], dtype=np.uint8)
        lab_green = srgb_to_lab(green)
        assert pytest.approx(lab_green[0], abs=0.1) == 87.73
        assert pytest.approx(lab_green[1], abs=0.1) == -86.18
        assert pytest.approx(lab_green[2], abs=0.1) == 83.18

        # Pure Blue: sRGB [0, 0, 255] -> Lab [32.30, 79.19, -107.86]
        blue = np.array([0, 0, 255], dtype=np.uint8)
        lab_blue = srgb_to_lab(blue)
        assert pytest.approx(lab_blue[0], abs=0.1) == 32.30
        assert pytest.approx(lab_blue[1], abs=0.1) == 79.19
        assert pytest.approx(lab_blue[2], abs=0.1) == -107.86

        # 50% Neutral Gray: sRGB [128, 128, 128] -> Lab [53.59, 0.0, 0.0]
        gray = np.array([128, 128, 128], dtype=np.uint8)
        lab_gray = srgb_to_lab(gray)
        assert pytest.approx(lab_gray[0], abs=0.1) == 53.59
        assert pytest.approx(lab_gray[1], abs=0.05) == 0.0
        assert pytest.approx(lab_gray[2], abs=0.05) == 0.0

    def test_float_srgb_inputs(self) -> None:
        """Verify that floating point sRGB inputs in [0, 1] yield identical results."""
        red_u8 = np.array([255, 0, 0], dtype=np.uint8)
        red_flt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        np.testing.assert_allclose(srgb_to_lab(red_u8), srgb_to_lab(red_flt), atol=1e-5)

    def test_masked_vs_unmasked_color_extraction(self) -> None:
        """Verify color extraction and palette quantization with and without binary masks."""
        # 100x100 synthetic image: Left half is pure red [255, 0, 0], Right half is pure blue [0, 0, 255]
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :50] = [255, 0, 0]
        img[:, 50:] = [0, 0, 255]

        # 1. Unmasked extraction: 50% red, 50% blue
        pal_unmasked = extract_rgb_palette(img, mask=None, num_colors=2)
        assert len(pal_unmasked) == 2
        percentages = {p["hex"].lower(): p["percentage"] for p in pal_unmasked}
        assert "#ff0000" in percentages
        assert "#0000ff" in percentages
        assert pytest.approx(percentages["#ff0000"], abs=0.01) == 0.50
        assert pytest.approx(percentages["#0000ff"], abs=0.01) == 0.50

        # Unmasked percentiles: median L* reflects both channels
        pct_unmasked = compute_lab_percentiles(img, mask=None)
        assert len(pct_unmasked["L"]) == 5
        assert len(pct_unmasked["a"]) == 5
        assert len(pct_unmasked["b"]) == 5

        # 2. Masked extraction: Mask isolating only the left (red) half
        mask_left = np.zeros((100, 100), dtype=bool)
        mask_left[:, :50] = True

        pal_masked = extract_rgb_palette(img, mask=mask_left, num_colors=2)
        assert len(pal_masked) == 1
        assert pal_masked[0]["hex"].lower() == "#ff0000"
        assert pal_masked[0]["rgb"] == [255, 0, 0]
        assert pal_masked[0]["percentage"] == 1.0

        pct_masked = compute_lab_percentiles(img, mask=mask_left)
        # All percentiles should match pure red values
        for val in pct_masked["L"]:
            assert pytest.approx(val, abs=0.1) == 53.24
        for val in pct_masked["a"]:
            assert pytest.approx(val, abs=0.1) == 80.09
        for val in pct_masked["b"]:
            assert pytest.approx(val, abs=0.1) == 67.20

        # 3. Empty mask fallback: all False mask should safely fall back to all pixels
        mask_empty = np.zeros((100, 100), dtype=bool)
        pixels_fallback = filter_pixels_by_mask(img, mask=mask_empty)
        assert len(pixels_fallback) == 10000

    def test_image_and_mask_io(self, tmp_path: Path) -> None:
        """Verify image and mask loading via PIL."""
        img_path = tmp_path / "test_frame.png"
        mask_path = tmp_path / "test_mask.png"

        img = Image.new("RGB", (64, 64), color=(200, 100, 50))
        img.save(img_path)

        mask_img = Image.new("L", (64, 64), color=0)
        # Paint center white (> 127)
        for y in range(16, 48):
            for x in range(16, 48):
                mask_img.putpixel((x, y), 255)
        mask_img.save(mask_path)

        loaded_img = load_image_rgb(img_path)
        assert loaded_img.shape == (64, 64, 3)
        assert loaded_img[0, 0].tolist() == [200, 100, 50]

        loaded_mask = load_mask(mask_path)
        assert loaded_mask is not None
        assert loaded_mask.shape == (64, 64)
        assert not loaded_mask[0, 0]
        assert loaded_mask[32, 32]

        assert load_mask(None) is None
        assert load_mask(tmp_path / "non_existent.png") is None

    def test_full_mask_and_alpha_image_handling(self, tmp_path: Path) -> None:
        """Verify full mask equals unmasked extraction, and RGBA images are converted cleanly."""
        img_arr = np.zeros((32, 32, 3), dtype=np.uint8)
        img_arr[:] = [120, 60, 200]

        # Full mask (all True)
        mask_full = np.ones((32, 32), dtype=bool)
        pct_unmasked = compute_lab_percentiles(img_arr, mask=None)
        pct_full_mask = compute_lab_percentiles(img_arr, mask=mask_full)
        assert pct_unmasked == pct_full_mask

        pal_unmasked = extract_rgb_palette(img_arr, mask=None, num_colors=1)
        pal_full_mask = extract_rgb_palette(img_arr, mask=mask_full, num_colors=1)
        assert pal_unmasked == pal_full_mask

        # RGBA image conversion
        rgba_path = tmp_path / "test_rgba.png"
        Image.new("RGBA", (16, 16), (50, 100, 150, 200)).save(rgba_path)
        loaded = load_image_rgb(rgba_path)
        assert loaded.shape == (16, 16, 3)
        assert loaded.dtype == np.uint8
        assert loaded[0, 0].tolist() == [50, 100, 150]

        # 1x1 image test for percentiles and palette
        tiny = np.array([[[255, 128, 64]]], dtype=np.uint8)
        pct_tiny = compute_lab_percentiles(tiny)
        assert len(pct_tiny["L"]) == 5
        pal_tiny = extract_rgb_palette(tiny, num_colors=1)
        assert len(pal_tiny) == 1
        assert pal_tiny[0]["rgb"] == [255, 128, 64]

    def test_color_science_edge_cases(self) -> None:
        """Verify boundary cases: empty image, mismatched mask shape, zero colors, and custom percentiles."""
        # 1. Zero-size empty image
        empty_img = np.empty((0, 0, 3), dtype=np.uint8)
        assert filter_pixels_by_mask(empty_img).shape == (0, 3)
        pct_empty = compute_lab_percentiles(empty_img)
        assert pct_empty == {"L": [0.0] * 5, "a": [0.0] * 5, "b": [0.0] * 5}
        assert extract_rgb_palette(empty_img) == []

        # 2. Palette with num_colors <= 0
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        assert extract_rgb_palette(img, num_colors=0) == []
        assert extract_rgb_palette(img, num_colors=-1) == []

        # 3. Mask shape mismatch: mask is (5, 5), image is (10, 10) -> safe fallback to full image
        bad_mask = np.ones((5, 5), dtype=bool)
        pixels = filter_pixels_by_mask(img, mask=bad_mask)
        assert len(pixels) == 100

        # 4. 3D mask shape (H, W, 1) and (H, W, 3)
        mask_3d = np.ones((10, 10, 1), dtype=np.uint8) * 255
        pixels_3d = filter_pixels_by_mask(img, mask=mask_3d)
        assert len(pixels_3d) == 100

        # 5. Monochromatic image: single color everywhere returns exactly 1 item with 1.0 (100%)
        mono = np.zeros((20, 20, 3), dtype=np.uint8)
        mono[:] = [42, 84, 126]
        mono_pal = extract_rgb_palette(mono, num_colors=5)
        assert len(mono_pal) == 1
        assert mono_pal[0]["percentage"] == 1.0
        assert mono_pal[0]["rgb"] == [42, 84, 126]

        # 6. Custom percentiles
        custom_pct = compute_lab_percentiles(mono, percentiles=[0.0, 50.0, 100.0])
        assert len(custom_pct["L"]) == 3
        assert len(custom_pct["a"]) == 3
        assert len(custom_pct["b"]) == 3




class TestColorBibleService:
    """Test ColorBibleService business logic and orchestration."""

    @pytest.fixture
    def service(self) -> ColorBibleService:
        """Provide an in-memory ColorBibleService."""
        repo = SQLiteColorBibleRepository(":memory:")
        svc = ColorBibleService(repo=repo)
        yield svc
        svc.close()

    def test_manifest_sync_idempotency(self, service: ColorBibleService, tmp_path: Path) -> None:
        """Verify manifest synchronization imports shots and is fully idempotent."""
        manifest_file = tmp_path / "film_shots.csv"
        rows = [
            {
                "index": str(i),
                "start": f"00:00:{i * 2:02d}.000",
                "end": f"00:00:{(i * 2) + 2:02d}.000",
                "start_frame": str(i * 48),
                "end_frame": str((i * 48) + 47),
                "source_reference": f"references/shot_{i:04d}_src.png",
                "color_reference": f"references/shot_{i:04d}_clr.png",
            }
            for i in range(10)
        ]
        write_manifest_details(manifest_file, "videos/source_feature.mp4", list(rows[0].keys()), rows)

        # Initial sync
        result1 = service.sync_manifest("film_alpha", manifest_file, title="Alpha Feature")
        assert result1["shots_synced"] == 10

        shots = service.list_shots("film_alpha")
        assert len(shots) == 10
        assert shots[0].shot_id == "film_alpha_0"
        assert shots[0].start_frame == 0
        assert shots[0].end_frame == 47
        assert shots[9].shot_id == "film_alpha_9"
        assert shots[9].end_frame == 479

        film = service.get_film("film_alpha")
        assert film is not None
        assert film.title == "Alpha Feature"
        assert film.source_video == "videos/source_feature.mp4"

        # Idempotent re-sync
        result2 = service.sync_manifest("film_alpha", manifest_file)
        assert result2["shots_synced"] == 10
        shots_after = service.list_shots("film_alpha")
        assert len(shots_after) == 10

    def test_non_contiguous_range_assignments(self, service: ColorBibleService) -> None:
        """Verify entity assignments across disjoint shot ranges (e.g. 12-18 and 410-435)."""
        char = service.create_character("film_alpha", "Elena", notes="Lead protagonist")
        variant = service.create_appearance_variant(char.character_id, "Trench Coat A")

        # Assign Elena's Trench Coat to shots 12-18 and shots 410-435
        assignments = service.assign_entity_ranges(
            "film_alpha",
            entity_type="character",
            entity_id=char.character_id,
            ranges=[(12, 18), (410, 435)],
            variant_id=variant.variant_id,
        )
        assert len(assignments) == 2

        # Query shot index 15 (inside first range)
        active_15 = service.get_shot_palette_assignments("film_alpha", 15)
        assert len(active_15) == 1
        assert active_15[0]["entity"]["name"] == "Elena"
        assert active_15[0]["variant"]["costume_id"] == "Trench Coat A"

        # Query shot index 420 (inside second range)
        active_420 = service.get_shot_palette_assignments("film_alpha", 420)
        assert len(active_420) == 1
        assert active_420[0]["entity"]["name"] == "Elena"
        assert active_420[0]["variant"]["costume_id"] == "Trench Coat A"

        # Query shot index 50 (outside both ranges)
        active_50 = service.get_shot_palette_assignments("film_alpha", 50)
        assert len(active_50) == 0

    def test_variant_isolation_across_shots(self, service: ColorBibleService, tmp_path: Path) -> None:
        """Verify variant isolation (Costume A in shot 5 vs Costume B in shot 520)."""
        char = service.create_character("film_alpha", "Elena")
        var_a = service.create_appearance_variant(char.character_id, "Winter Trench Coat")
        var_b = service.create_appearance_variant(char.character_id, "Summer Evening Dress")

        # Assign Costume A to early shots [1, 10], Costume B to late shots [500, 550]
        service.assign_entity_ranges(
            "film_alpha",
            "character",
            char.character_id,
            ranges=[(1, 10)],
            variant_id=var_a.variant_id,
        )
        service.assign_entity_ranges(
            "film_alpha",
            "character",
            char.character_id,
            ranges=[(500, 550)],
            variant_id=var_b.variant_id,
        )

        # Create synthetic keyframe images for each costume
        img_a_path = tmp_path / "elena_trench.png"
        img_b_path = tmp_path / "elena_dress.png"
        Image.new("RGB", (64, 64), (120, 80, 40)).save(img_a_path)   # Brown trench
        Image.new("RGB", (64, 64), (40, 120, 200)).save(img_b_path)  # Blue dress

        ref_a = service.register_reference_image(
            "film_alpha",
            entity_type="character",
            entity_id=char.character_id,
            variant_id=var_a.variant_id,
            keyframe_path=img_a_path,
            auto_approve=True,
            approved_by="lead_colorist",
        )
        ref_b = service.register_reference_image(
            "film_alpha",
            entity_type="character",
            entity_id=char.character_id,
            variant_id=var_b.variant_id,
            keyframe_path=img_b_path,
            auto_approve=True,
            approved_by="lead_colorist",
        )

        # Inspect shot 5: Must contain only Costume A
        shot_5 = service.get_shot_palette_assignments("film_alpha", 5)
        assert len(shot_5) == 1
        assert shot_5[0]["variant"]["variant_id"] == var_a.variant_id
        assert shot_5[0]["variant"]["costume_id"] == "Winter Trench Coat"
        assert len(shot_5[0]["color_references"]) == 1
        assert shot_5[0]["color_references"][0]["ref_id"] == ref_a.ref_id

        # Inspect shot 520: Must contain only Costume B
        shot_520 = service.get_shot_palette_assignments("film_alpha", 520)
        assert len(shot_520) == 1
        assert shot_520[0]["variant"]["variant_id"] == var_b.variant_id
        assert shot_520[0]["variant"]["costume_id"] == "Summer Evening Dress"
        assert len(shot_520[0]["color_references"]) == 1
        assert shot_520[0]["color_references"][0]["ref_id"] == ref_b.ref_id

    def test_deterministic_snapshot_hash_stability(self, tmp_path: Path) -> None:
        """Verify deterministic snapshot hash computation across service instances."""
        db_file = tmp_path / "test_color_bible.db"

        # Instance 1: Create entities, ranges, and references
        with ColorBibleService(db_path=db_file) as svc1:
            char = svc1.create_character("film_hash", "Viktor")
            var = svc1.create_appearance_variant(char.character_id, "Suit 01")
            svc1.assign_entity_ranges(
                "film_hash",
                "character",
                char.character_id,
                ranges=[(1, 20), (50, 70)],
                variant_id=var.variant_id,
            )

            img_path = tmp_path / "viktor_suit.png"
            Image.new("RGB", (32, 32), (30, 30, 30)).save(img_path)

            ref = svc1.register_reference_image(
                "film_hash",
                entity_type="character",
                entity_id=char.character_id,
                variant_id=var.variant_id,
                keyframe_path=img_path,
                auto_approve=False,  # Initially draft
            )

            # Draft reference should NOT affect snapshot hash
            hash_before_approval = svc1.get_snapshot_hash("film_hash")
            assert hash_before_approval.startswith("sha256:")
            assert len(hash_before_approval) == 7 + 64

            # Repeat hash on svc1: must be identical
            assert svc1.get_snapshot_hash("film_hash") == hash_before_approval

            # Approve reference
            svc1.approve_reference(ref.ref_id, approved_by="senior_colorist")
            hash_after_approval = svc1.get_snapshot_hash("film_hash")
            assert hash_after_approval != hash_before_approval

        # Instance 2: Fresh instance on same database file (simulating new session/process)
        with ColorBibleService(db_path=db_file) as svc2:
            fresh_hash = svc2.get_snapshot_hash("film_hash")
            assert fresh_hash == hash_after_approval

            # Reject reference: should now revert back or change because approval_status is 'rejected'
            svc2.reject_reference(ref.ref_id)
            hash_after_reject = svc2.get_snapshot_hash("film_hash")
            assert hash_after_reject == hash_before_approval

    def test_objects_and_locations_management(self, service: ColorBibleService) -> None:
        """Verify object and location entity creation and range assignments."""
        prop = service.create_object(
            "film_alpha",
            kind="prop",
            name="Vintage Pocket Watch",
            palette=["#d4af37", "#4a3c18"],
            notes="Gold plated brass",
        )
        assert prop.kind == "prop"
        assert prop.name == "Vintage Pocket Watch"

        loc = service.create_location(
            "film_alpha",
            set_name="Grand Hotel Ballroom",
            lighting_context="warm candelabra",
            palette=["#f8e3c2", "#7b1113"],
            notes="Interior night",
        )
        assert loc.set_name == "Grand Hotel Ballroom"

        # Assign prop and location to shot 42
        service.assign_entity_ranges(
            "film_alpha",
            entity_type="object",
            entity_id=prop.object_id,
            ranges=[(40, 45)],
        )
        service.assign_entity_ranges(
            "film_alpha",
            entity_type="location",
            entity_id=loc.location_id,
            ranges=[(35, 55)],
        )

        shot_42 = service.get_shot_palette_assignments("film_alpha", 42)
        assert len(shot_42) == 2
        entity_types = {item["entity_type"] for item in shot_42}
        assert entity_types == {"object", "location"}

    def test_validation_errors(self, service: ColorBibleService) -> None:
        """Verify strict validation and defensive error handling."""
        with pytest.raises(ValueError, match="Character name must not be empty"):
            service.create_character("film_err", "   ")

        with pytest.raises(ValueError, match="Character 'non_existent' does not exist"):
            service.create_appearance_variant("non_existent", "Costume A")

        char = service.create_character("film_err", "Elena")
        with pytest.raises(ValueError, match="Costume ID must not be empty"):
            service.create_appearance_variant(char.character_id, "")

        with pytest.raises(ValueError, match="Invalid object kind"):
            service.create_object("film_err", kind="invalid_kind", name="Hat")

        with pytest.raises(ValueError, match="Location set_name must not be empty"):
            service.create_location("film_err", set_name="")

        with pytest.raises(ValueError, match="Invalid entity_type"):
            service.assign_entity_ranges("film_err", "invalid_type", "id_1", [(1, 5)])

        with pytest.raises(ValueError, match="Invalid range: start \\(10\\) > end \\(5\\)"):
            service.assign_entity_ranges("film_err", "character", char.character_id, [(10, 5)])

        with pytest.raises(ValueError, match="Color reference 'non_existent' not found"):
            service.approve_reference("non_existent")

        with pytest.raises(ValueError, match="Color reference 'non_existent' not found"):
            service.reject_reference("non_existent")

    def test_parse_time_seconds(self) -> None:
        """Verify timecode string parsing."""
        assert parse_time_seconds("00:01:23.500") == 83.5
        assert parse_time_seconds("02:15.000") == 135.0
        assert parse_time_seconds("45.2") == 45.2
        assert parse_time_seconds(30) == 30.0
        assert parse_time_seconds(None) == 0.0
        assert parse_time_seconds("") == 0.0
        assert parse_time_seconds("invalid") == 0.0

    def test_multiple_entities_in_same_shot(self, service: ColorBibleService) -> None:
        """Verify multiple distinct characters, objects, and locations assigned to the same shot."""
        char1 = service.create_character("film_multi", "Elena")
        var1 = service.create_appearance_variant(char1.character_id, "Costume Trench")
        char2 = service.create_character("film_multi", "Viktor")
        var2 = service.create_appearance_variant(char2.character_id, "Costume Tuxedo")
        prop = service.create_object("film_multi", kind="prop", name="Briefcase")
        loc = service.create_location("film_multi", set_name="Casino Floor")

        # Assign all 4 entities to overlap at shot 15
        service.assign_entity_ranges("film_multi", "character", char1.character_id, [(10, 20)], variant_id=var1.variant_id)
        service.assign_entity_ranges("film_multi", "character", char2.character_id, [(12, 18)], variant_id=var2.variant_id)
        service.assign_entity_ranges("film_multi", "object", prop.object_id, [(15, 25)])
        service.assign_entity_ranges("film_multi", "location", loc.location_id, [(1, 30)])

        shot_15 = service.get_shot_palette_assignments("film_multi", 15)
        assert len(shot_15) == 4
        entity_types = {item["entity_type"] for item in shot_15}
        assert entity_types == {"character", "object", "location"}

        entity_names = {item["entity"].get("name") or item["entity"].get("set_name") for item in shot_15}
        assert entity_names == {"Elena", "Viktor", "Briefcase", "Casino Floor"}

    def test_entity_without_variant_and_draft_reference_filtering(self, service: ColorBibleService, tmp_path: Path) -> None:
        """Verify that draft references are excluded from shot palettes and entities without variants are supported."""
        char = service.create_character("film_filter", "Agent X")
        var = service.create_appearance_variant(char.character_id, "Stealth Gear")
        service.assign_entity_ranges("film_filter", "character", char.character_id, [(1, 10)], variant_id=var.variant_id)

        img1 = tmp_path / "ref_approved.png"
        img2 = tmp_path / "ref_draft.png"
        Image.new("RGB", (32, 32), (100, 150, 200)).save(img1)
        Image.new("RGB", (32, 32), (200, 150, 100)).save(img2)

        # 1 approved reference and 1 draft reference
        ref_app = service.register_reference_image(
            "film_filter",
            "character",
            char.character_id,
            variant_id=var.variant_id,
            keyframe_path=img1,
            auto_approve=True,
            approved_by="supervisor",
        )
        ref_draft = service.register_reference_image(
            "film_filter",
            "character",
            char.character_id,
            variant_id=var.variant_id,
            keyframe_path=img2,
            auto_approve=False,
        )

        shot_data = service.get_shot_palette_assignments("film_filter", 5)
        assert len(shot_data) == 1
        refs = shot_data[0]["color_references"]
        assert len(refs) == 1
        assert refs[0]["ref_id"] == ref_app.ref_id

        # Reject the approved reference -> should now return empty references list
        service.reject_reference(ref_app.ref_id)
        shot_data_after_reject = service.get_shot_palette_assignments("film_filter", 5)
        assert len(shot_data_after_reject[0]["color_references"]) == 0

    def test_snapshot_hash_ordering_stability(self, tmp_path: Path) -> None:
        """Verify snapshot hash determinism regardless of reference/assignment insertion ordering."""
        db1_path = tmp_path / "db1.sqlite"
        db2_path = tmp_path / "db2.sqlite"

        from ai_remaster_gui.color_bible_db import (
            AppearanceVariant,
            Character,
            ColorReferenceRecord,
            Film,
            ShotEntityAssignment,
        )

        film = Film(film_id="film_ord", title="Ordering Test")
        char_a = Character(character_id="char_alpha", film_id="film_ord", name="Alpha", created_at="2026-01-01T00:00:00Z")
        char_b = Character(character_id="char_beta", film_id="film_ord", name="Beta", created_at="2026-01-01T00:00:00Z")
        var_a = AppearanceVariant(variant_id="var_alpha", character_id="char_alpha", costume_id="Costume A", created_at="2026-01-01T00:00:00Z")
        var_b = AppearanceVariant(variant_id="var_beta", character_id="char_beta", costume_id="Costume B", created_at="2026-01-01T00:00:00Z")

        # Fixed records
        asg1 = ShotEntityAssignment(
            assignment_id="asg_01",
            film_id="film_ord",
            entity_type="character",
            entity_id="char_alpha",
            variant_id="var_alpha",
            valid_shot_range_start=1,
            valid_shot_range_end=5,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        asg2 = ShotEntityAssignment(
            assignment_id="asg_02",
            film_id="film_ord",
            entity_type="character",
            entity_id="char_beta",
            variant_id="var_beta",
            valid_shot_range_start=6,
            valid_shot_range_end=10,
            created_at="2026-01-01T00:00:00Z",
            updated_at="2026-01-01T00:00:00Z",
        )
        ref1 = ColorReferenceRecord(
            ref_id="ref_01",
            film_id="film_ord",
            entity_type="character",
            entity_id="char_alpha",
            variant_id="var_alpha",
            source_keyframe_path="kf1.png",
            lab_stats_json='{"L": [50.0]}',
            rgb_palette_json='[{"hex": "#112233"}]',
            provenance="manual",
            approval_status="approved",
            created_at="2026-01-01T00:00:00Z",
        )
        ref2 = ColorReferenceRecord(
            ref_id="ref_02",
            film_id="film_ord",
            entity_type="character",
            entity_id="char_beta",
            variant_id="var_beta",
            source_keyframe_path="kf2.png",
            lab_stats_json='{"L": [60.0]}',
            rgb_palette_json='[{"hex": "#445566"}]',
            provenance="manual",
            approval_status="approved",
            created_at="2026-01-01T00:00:00Z",
        )

        # DB 1: Insert entities and then (asg1, asg2) and (ref1, ref2)
        with ColorBibleService(db_path=db1_path) as svc1:
            svc1.repo.upsert_film(film)
            svc1.repo.upsert_character(char_a)
            svc1.repo.upsert_character(char_b)
            svc1.repo.upsert_variant(var_a)
            svc1.repo.upsert_variant(var_b)
            svc1.repo.upsert_assignment(asg1)
            svc1.repo.upsert_assignment(asg2)
            svc1.repo.upsert_reference(ref1)
            svc1.repo.upsert_reference(ref2)
            hash1 = svc1.get_snapshot_hash("film_ord")

        # DB 2: Insert entities and in reverse order (asg2, asg1) and (ref2, ref1)
        with ColorBibleService(db_path=db2_path) as svc2:
            svc2.repo.upsert_film(film)
            svc2.repo.upsert_character(char_b)
            svc2.repo.upsert_character(char_a)
            svc2.repo.upsert_variant(var_b)
            svc2.repo.upsert_variant(var_a)
            svc2.repo.upsert_assignment(asg2)
            svc2.repo.upsert_assignment(asg1)
            svc2.repo.upsert_reference(ref2)
            svc2.repo.upsert_reference(ref1)
            hash2 = svc2.get_snapshot_hash("film_ord")

        assert hash1.startswith("sha256:")
        assert hash2.startswith("sha256:")
        assert hash1 == hash2

    def test_manifest_sync_with_missing_columns_and_empty_manifest(self, service: ColorBibleService, tmp_path: Path) -> None:
        """Verify manifest syncing with missing optional columns and empty manifest."""
        # Empty manifest with headers only
        empty_manifest = tmp_path / "empty_manifest.csv"
        empty_manifest.write_text("index,start,end\n", encoding="utf-8")
        res_empty = service.sync_manifest("film_empty", empty_manifest)
        assert res_empty["shots_synced"] == 0
        assert len(service.list_shots("film_empty")) == 0

        # Manifest with minimal columns (relying on start_time/end_time)
        minimal_manifest = tmp_path / "minimal_manifest.csv"
        rows = [
            {"shot_index": "0", "start_time": "00:00:01.000", "end_time": "00:00:03.000"},
            {"shot_index": "1", "start_time": "00:00:03.000", "end_time": "00:00:06.500"},
        ]
        import csv
        with open(minimal_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        res_min = service.sync_manifest("film_min", minimal_manifest)
        assert res_min["shots_synced"] == 2
        shots = service.list_shots("film_min")
        assert len(shots) == 2
        assert shots[0].start_frame == 24
        assert shots[0].end_frame == 72

    def test_service_deletion_and_cascade_lifecycle(self, service: ColorBibleService) -> None:
        """Verify lifecycle operations: entity creation, assignment deletion, and cascade deletions."""
        char = service.create_character("film_del", "Charlie")
        var = service.create_appearance_variant(char.character_id, "Tweed Jacket")
        asgns = service.assign_entity_ranges("film_del", "character", char.character_id, [(1, 5)], variant_id=var.variant_id)
        assert len(asgns) == 1
        asgn_id = asgns[0].assignment_id

        # Delete single assignment
        assert service.delete_assignment(asgn_id) is True
        assert len(service.list_assignments("film_del")) == 0

        # Re-assign and delete character
        service.assign_entity_ranges("film_del", "character", char.character_id, [(1, 5)], variant_id=var.variant_id)
        assert service.delete_character(char.character_id) is True
        assert service.get_character(char.character_id) is None
        assert service.get_variant(var.variant_id) is None

        # Film deletion
        assert service.delete_film("film_del") is True
        assert service.get_film("film_del") is None

    def test_snapshot_hash_updates_on_deletion(self, service: ColorBibleService, tmp_path: Path) -> None:
        """Verify that deleting an approved reference or assignment correctly updates the snapshot hash."""
        char = service.create_character("film_hash_del", "Diana")
        asgns = service.assign_entity_ranges("film_hash_del", "character", char.character_id, [(1, 10)])

        img_path = tmp_path / "diana_ref.png"
        Image.new("RGB", (32, 32), (80, 160, 240)).save(img_path)

        ref = service.register_reference_image(
            "film_hash_del",
            "character",
            char.character_id,
            keyframe_path=img_path,
            auto_approve=True,
        )
        hash_with_ref = service.get_snapshot_hash("film_hash_del")

        # Delete the reference -> hash must change
        assert service.delete_reference(ref.ref_id) is True
        hash_without_ref = service.get_snapshot_hash("film_hash_del")
        assert hash_without_ref != hash_with_ref

        # Delete assignment -> hash must change again
        service.delete_assignment(asgns[0].assignment_id)
        hash_empty = service.get_snapshot_hash("film_hash_del")
        assert hash_empty != hash_without_ref

    def test_register_reference_missing_mask_fallback(self, service: ColorBibleService, tmp_path: Path) -> None:
        """Verify registering a reference with a missing mask gracefully falls back without failing."""
        char = service.create_character("film_mask_fallback", "Eve")
        img_path = tmp_path / "eve_ref.png"
        Image.new("RGB", (32, 32), (200, 100, 50)).save(img_path)

        ref = service.register_reference_image(
            "film_mask_fallback",
            "character",
            char.character_id,
            keyframe_path=img_path,
            mask_uri=tmp_path / "non_existent_mask.png",
            auto_approve=False,
        )
        assert ref.ref_id.startswith("ref_")
        assert ref.approval_status == "draft"
        stats = json.loads(ref.lab_stats_json)
        assert "L" in stats

    def test_create_film_and_empty_film_id_validation(self, service: ColorBibleService, tmp_path: Path) -> None:
        """Verify dedicated create_film works and empty film_id is rejected across entity creation methods."""
        # Valid film creation
        film = service.create_film("film_standalone", title="Standalone Movie", source_video="media/movie.mp4")
        assert film.film_id == "film_standalone"
        assert film.title == "Standalone Movie"
        assert film.source_video == "media/movie.mp4"
        assert service.get_film("film_standalone") is not None

        # Repeat creation updates or returns existing
        film2 = service.create_film("film_standalone", title="Standalone Movie Renamed")
        assert film2.title == "Standalone Movie Renamed"

        # Empty film_id validation
        with pytest.raises(ValueError, match="film_id must not be empty"):
            service.create_film("   ")

        with pytest.raises(ValueError, match="film_id must not be empty"):
            service.create_character("  ", "Hero")

        with pytest.raises(ValueError, match="film_id must not be empty"):
            service.create_object("  ", "prop", "Ring")

        with pytest.raises(ValueError, match="film_id must not be empty"):
            service.create_location("  ", "Courtyard")

        with pytest.raises(ValueError, match="film_id must not be empty"):
            service.assign_entity_ranges("  ", "character", "char_1", [(1, 2)])

        img_path = tmp_path / "dummy.png"
        Image.new("RGB", (10, 10), (100, 100, 100)).save(img_path)
        with pytest.raises(ValueError, match="film_id must not be empty"):
            service.register_reference_image("  ", "character", "char_1", img_path)

        manifest_file = tmp_path / "dummy.csv"
        manifest_file.write_text("shot_index,start_time,end_time\n0,0,1\n")
        with pytest.raises(ValueError, match="film_id must not be empty"):
            service.sync_manifest("  ", manifest_file)

    def test_snapshot_hash_only_includes_approved_assignments(self, service: ColorBibleService) -> None:
        """Verify get_snapshot_hash only includes approved assignments, ignoring draft or rejected ones."""
        film_id = "film_hash_approval"
        service.create_film(film_id, "Hash Approval Film")
        char = service.create_character(film_id, "Hero")

        base_hash = service.get_snapshot_hash(film_id)

        # Add draft assignment -> snapshot hash should NOT change
        service.assign_entity_ranges(
            film_id, "character", char.character_id, [(1, 5)], approval_status="draft"
        )
        assert service.get_snapshot_hash(film_id) == base_hash

        # Add pending_review assignment -> snapshot hash should NOT change
        service.assign_entity_ranges(
            film_id, "character", char.character_id, [(6, 10)], approval_status="pending_review"
        )
        assert service.get_snapshot_hash(film_id) == base_hash

        # Add approved assignment -> snapshot hash MUST change
        service.assign_entity_ranges(
            film_id, "character", char.character_id, [(11, 15)], approval_status="approved"
        )
        approved_hash = service.get_snapshot_hash(film_id)
        assert approved_hash != base_hash



