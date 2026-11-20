"""Comprehensive Phase 9 End-to-End Restoration Integration & Mask Adoption Test Suite.

Verifies:
1. End-to-end execution:
   sample frame -> defect detection -> MultiChannelDefectMask -> adopt mask -> inpaint via Telea -> verified restored frame.
2. Bit-for-bit effective mask consumption:
   Effective mask consumed by Telea is bit-for-bit identical to MultiChannelDefectMask.get_effective_repair_mask(policy).
3. All RepairInclusionPolicy variants:
   - CONSERVATIVE: Scratch | Dust (Uncertain excluded).
   - HIGH_RECALL: Scratch | Dust | Uncertain.
   - SCRATCH_ONLY: Scratch only.
   - COMBINED_DEFECT: Combined proposal mask.
4. Engine dispatch across backends:
   - OpenCV Telea (strict mask gating, radius from preset).
   - ProPainter (temporal video inpainting patch, strict mask gating).
   - DART Composite (diagnostic frame replacement, strict mask gating).
5. Comprehensive provenance manifest (restoration_provenance.json):
   - detector (name, version, architecture)
   - checkpoint_sha256
   - license (status, commercial compatibility)
   - preset (id, version)
   - repair_policy
   - channel_provenance (native vs derived)
   - repair_backend
   - effective_mask_hash (SHA-256)
   - source_video
   - output_video
   - timestamp (ISO-8601)
6. Non-destructive invariance protection:
   Previewing a different detector or adjusting threshold NEVER alters or deletes an already accepted repair mask.
7. Restored video cache identity incorporates:
   detector, checkpoint_sha256, preset, repair_policy, repair_backend, and effective_mask_hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
GUI_DIR = ROOT_DIR / "ai_remaster_gui"

for p in (str(SCRIPTS_DIR), str(GUI_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ai_remaster_gui import dust
from ai_remaster_gui.dust_jobs import (
    adopt_defect_mask,
    build_restoration_provenance,
    compute_file_sha256,
    compute_mask_sha256,
    compute_restoration_cache_identity,
    dispatch_repair,
    protect_accepted_mask,
    resolve_effective_repair_mask,
)
from film_defect_detectors import (
    ChannelOrigin,
    ChannelProvenance,
    MultiChannelDefectMask,
    RepairInclusionPolicy,
    UnsupportedPolicyError,
)
from dust_native_composite import (
    dispatch_repair_frame,
    inpaint_frame_dart_composite,
    inpaint_frame_propainter,
    inpaint_frame_telea,
    write_restoration_provenance,
)


class EndToEndRestorationPhase9Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a synthetic 64x96 test image with background texture
        self.height, self.width = 64, 96
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        self.sample_image = ((x_coords * 2 + y_coords * 3) % 200 + 40).astype(np.uint8)
        self.sample_image = cv2.cvtColor(self.sample_image, cv2.COLOR_GRAY2BGR)

        # Create distinct defect patterns:
        # 1. Scratch: vertical line at col 20
        self.scratch = np.zeros((self.height, self.width), dtype=np.uint8)
        self.scratch[10:55, 20:22] = 255

        # 2. Dust: spot around row 40, col 70
        self.dust = np.zeros((self.height, self.width), dtype=np.uint8)
        self.dust[38:43, 68:73] = 255

        # 3. Uncertain: ambiguous artifact around row 15, col 80
        self.uncertain = np.zeros((self.height, self.width), dtype=np.uint8)
        self.uncertain[14:18, 78:83] = 255

        # Combined mask: bitwise union of all 3
        self.combined = np.bitwise_or(np.bitwise_or(self.scratch, self.dust), self.uncertain)
        self.prob = (self.combined.astype(np.float32) / 255.0)

        # Inject defects into the sample image (make defect pixels bright white 255)
        self.corrupted_image = self.sample_image.copy()
        self.corrupted_image[self.combined > 0] = 255

        # Build MultiChannelDefectMask container
        self.mask_container = MultiChannelDefectMask(
            combined_probability=self.prob,
            combined_mask=self.combined,
            scratch_mask=self.scratch,
            dust_mask=self.dust,
            uncertain_mask=self.uncertain,
            provenance={
                "scratch": ChannelProvenance(
                    origin=ChannelOrigin.NATIVE,
                    is_available=True,
                    source_algorithm="ridge_tracker",
                ),
                "dust": ChannelProvenance(
                    origin=ChannelOrigin.DERIVED,
                    is_available=True,
                    source_algorithm="temporal_diff",
                ),
                "uncertain": ChannelProvenance(
                    origin=ChannelOrigin.DERIVED,
                    is_available=True,
                    source_algorithm="ambiguity_filter",
                ),
            },
            metadata={"source": "phase9_synthetic_fixture"},
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_effective_mask_bit_for_bit_fidelity_across_policies(self):
        """Verify effective masks resolved via RepairInclusionPolicy are bit-for-bit exact."""
        # 1. CONSERVATIVE: Scratch | Dust (Uncertain excluded)
        eff_cons = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        expected_cons = np.bitwise_or(self.scratch, self.dust)
        np.testing.assert_array_equal(eff_cons, expected_cons)
        self.assertTrue(np.all(eff_cons[self.uncertain > 0] == 0), "Uncertain defect must be excluded in CONSERVATIVE")
        self.assertTrue(np.all(eff_cons[self.scratch > 0] == 255), "Scratch defect must be present in CONSERVATIVE")
        self.assertTrue(np.all(eff_cons[self.dust > 0] == 255), "Dust defect must be present in CONSERVATIVE")

        # 2. HIGH_RECALL: Scratch | Dust | Uncertain
        eff_hr = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)
        expected_hr = np.bitwise_or(np.bitwise_or(self.scratch, self.dust), self.uncertain)
        np.testing.assert_array_equal(eff_hr, expected_hr)
        self.assertTrue(np.all(eff_hr[self.uncertain > 0] == 255), "Uncertain defect must be included in HIGH_RECALL")

        # 3. SCRATCH_ONLY: Scratch only
        eff_scratch = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)
        np.testing.assert_array_equal(eff_scratch, self.scratch)
        self.assertTrue(np.all(eff_scratch[self.dust > 0] == 0), "Dust defect must be excluded in SCRATCH_ONLY")
        self.assertTrue(np.all(eff_scratch[self.uncertain > 0] == 0), "Uncertain defect must be excluded in SCRATCH_ONLY")

        # 4. COMBINED_DEFECT: Full combined mask
        eff_comb = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.COMBINED_DEFECT)
        np.testing.assert_array_equal(eff_comb, self.combined)

    def test_unsupported_policy_raises_error(self):
        """Verify UnsupportedPolicyError when requested policy requires missing channel."""
        mask_no_scratch = MultiChannelDefectMask(
            combined_probability=self.prob,
            combined_mask=self.combined,
            scratch_mask=None,
            dust_mask=self.dust,
            uncertain_mask=self.uncertain,
        )
        with self.assertRaises(UnsupportedPolicyError):
            mask_no_scratch.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)

    def test_end_to_end_telea_inpainting_and_provenance(self):
        """End-to-end: Corrupted image -> MultiChannelDefectMask -> adopt -> Telea inpainting -> Provenance."""
        target_folder = self.test_dir / "candidate_01"
        out_image_path = target_folder / "repaired.png"

        # 1. Adopt mask with CONSERVATIVE policy
        adoption = adopt_defect_mask(
            target_folder=target_folder,
            mask=self.mask_container,
            policy=RepairInclusionPolicy.CONSERVATIVE,
            preset="balanced",
            repair_backend="telea",
            checkpoint_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        )

        eff_mask_file = Path(adoption["effective_repair_mask"])
        self.assertTrue(eff_mask_file.is_file(), "effective_repair_mask.png must be created")
        adopted_eff = cv2.imread(str(eff_mask_file), cv2.IMREAD_GRAYSCALE)

        # Bit-for-bit check: adopted effective mask equals mask.get_effective_repair_mask(CONSERVATIVE)
        expected_eff = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        np.testing.assert_array_equal(adopted_eff, expected_eff)

        # Channel masks check
        self.assertTrue((target_folder / "scratch_mask.png").is_file())
        self.assertTrue((target_folder / "dust_mask.png").is_file())
        self.assertTrue((target_folder / "uncertain_mask.png").is_file())
        self.assertTrue((target_folder / "mask.png").is_file())

        # 2. Dispatch inpainting via Telea
        repaired, provenance = dispatch_repair(
            source=self.corrupted_image,
            effective_repair_mask=eff_mask_file,
            repair_backend="telea",
            output_path=out_image_path,
            radius=3,
            preset="balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            detector_info={"name": "farneback_3f", "version": "1.0.0", "architecture": "Optical Flow"},
            checkpoint_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            license_info={"status": "commercial_cleared", "commercial_compatible": True},
            channel_provenance=adoption["channel_provenance"],
        )

        # 3. Verify inpainting invariants
        support = adopted_eff > 127
        # Invariant 1: Outside of effective_repair_mask, NOT A SINGLE PIXEL is changed
        np.testing.assert_array_equal(
            repaired[~support],
            self.corrupted_image[~support],
            err_msg="Pixels outside effective repair mask must remain strictly bit-for-bit identical!",
        )

        # Invariant 2: Inside the effective_repair_mask, defects were inpainted
        changed_inside = np.any(repaired[support] != self.corrupted_image[support], axis=1)
        self.assertTrue(np.any(changed_inside), "Inpainting must alter defective pixels inside the mask")

        # Invariant 3: Uncertain region was NOT inpainted because CONSERVATIVE excluded it
        uncertain_support = self.uncertain > 127
        np.testing.assert_array_equal(
            repaired[uncertain_support],
            self.corrupted_image[uncertain_support],
            err_msg="Uncertain pixels must not be modified under CONSERVATIVE policy",
        )

        # 4. Verify restoration_provenance.json persistence
        prov_file = target_folder / "restoration_provenance.json"
        self.assertTrue(prov_file.is_file(), "restoration_provenance.json must be persisted")
        saved_prov = json.loads(prov_file.read_text(encoding="utf-8"))

        required_keys = [
            "detector",
            "checkpoint_sha256",
            "license",
            "preset",
            "repair_policy",
            "channel_provenance",
            "repair_backend",
            "effective_mask_hash",
            "source_video",
            "output_video",
            "timestamp",
        ]
        for key in required_keys:
            self.assertIn(key, saved_prov, f"Field '{key}' missing from restoration_provenance.json")

        self.assertEqual(saved_prov["repair_backend"], "telea")
        self.assertEqual(saved_prov["repair_policy"], "conservative")
        self.assertEqual(saved_prov["detector"]["name"], "farneback_3f")
        self.assertEqual(saved_prov["license"]["commercial_compatible"], True)

        # Hash check: effective_mask_hash matches exact SHA-256 of the mask file
        self.assertEqual(saved_prov["effective_mask_hash"], compute_file_sha256(eff_mask_file))

    def test_propainter_inpaint_dispatch_mask_consumption(self):
        """Verify ProPainter inpainting backend strictly consumes effective_repair_mask."""
        target_folder = self.test_dir / "propainter_cand"
        target_folder.mkdir(parents=True, exist_ok=True)

        eff_mask = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)
        support = eff_mask > 127

        # Synthetic model patch (e.g. from neural inpainter)
        model_patch = np.full_like(self.corrupted_image, 180)

        repaired, counts = inpaint_frame_propainter(
            image=self.corrupted_image,
            patch=model_patch,
            effective_repair_mask=eff_mask,
        )

        # Outside mask must be identical to source
        np.testing.assert_array_equal(repaired[~support], self.corrupted_image[~support])
        # Inside mask must be the model patch
        np.testing.assert_array_equal(repaired[support], model_patch[support])
        self.assertEqual(counts["changed_outside_pixels"], 0)

    def test_dart_composite_inpaint_dispatch_mask_consumption(self):
        """Verify DART composite inpainting strictly consumes effective_repair_mask."""
        target_folder = self.test_dir / "dart_cand"
        target_folder.mkdir(parents=True, exist_ok=True)

        eff_mask = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)
        support = eff_mask > 127

        # Synthetic DART full diagnostic frame
        dart_full = np.full_like(self.corrupted_image, 150)

        repaired, counts = inpaint_frame_dart_composite(
            image=self.corrupted_image,
            dart_full=dart_full,
            effective_repair_mask=eff_mask,
        )

        # Outside mask must be identical to source
        np.testing.assert_array_equal(repaired[~support], self.corrupted_image[~support])
        # Inside mask must be replaced by dart_full
        np.testing.assert_array_equal(repaired[support], dart_full[support])
        self.assertEqual(counts["changed_outside_pixels"], 0)

    def test_preview_switching_does_not_mutate_accepted_mask(self):
        """Invariant: Previewing a different detector or adjusting threshold NEVER alters an accepted mask."""
        adopted_dir = self.test_dir / "adopted_candidate"
        adopted_dir.mkdir(parents=True, exist_ok=True)

        # Simulate completed report and accepted mask
        report_data = {
            "identity": {"version": 3, "detector": "farneback_3f"},
            "source_sha256": "fake_source_sha",
            "repair_complete": True,
            "hashes": {},
        }
        report_file = adopted_dir / "report.json"
        report_file.write_text(json.dumps(report_data), encoding="utf-8")
        adopted_sha = compute_file_sha256(report_file)

        mask_file = adopted_dir / "effective_repair_mask.png"
        cv2.imwrite(str(mask_file), self.scratch)
        initial_mask_sha = compute_file_sha256(mask_file)

        # Configure settings as having adopted this report
        settings = {
            "dust": {
                "adopted_report": adopted_sha,
                "detector": "farneback_3f",
            }
        }

        # Attempting to mutate the accepted mask folder with a preview or new threshold must raise ValueError
        with self.assertRaisesRegex(ValueError, "cannot mutate an accepted mask"):
            protect_accepted_mask(adopted_dir, settings)

        with self.assertRaisesRegex(ValueError, "cannot mutate an accepted mask"):
            adopt_defect_mask(
                target_folder=adopted_dir,
                mask=self.mask_container,
                policy=RepairInclusionPolicy.HIGH_RECALL,
                settings=settings,
            )

        # Verify accepted mask is 100% untouched
        self.assertEqual(compute_file_sha256(mask_file), initial_mask_sha)
        self.assertEqual(compute_file_sha256(report_file), adopted_sha)

    def test_restoration_cache_identity_incorporates_required_invariants(self):
        """Verify cache identity incorporates detector, checkpoint, preset, policy, backend, and mask hash."""
        base_identity = compute_restoration_cache_identity(
            source=self.test_dir / "source.mkv",
            detector="farneback_3f",
            checkpoint_sha256="abc12345",
            preset="balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="deadbeef0123456789",
        )
        base_key = base_identity["cache_key"]

        # 1. Changing detector must change key
        id_det = compute_restoration_cache_identity(
            source=self.test_dir / "source.mkv",
            detector="bopb",
            checkpoint_sha256="abc12345",
            preset="balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="deadbeef0123456789",
        )
        self.assertNotEqual(base_key, id_det["cache_key"])

        # 2. Changing checkpoint_sha256 must change key
        id_cp = compute_restoration_cache_identity(
            source=self.test_dir / "source.mkv",
            detector="farneback_3f",
            checkpoint_sha256="changed_checkpoint_hash",
            preset="balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="deadbeef0123456789",
        )
        self.assertNotEqual(base_key, id_cp["cache_key"])

        # 3. Changing preset must change key
        id_preset = compute_restoration_cache_identity(
            source=self.test_dir / "source.mkv",
            detector="farneback_3f",
            checkpoint_sha256="abc12345",
            preset="high_precision",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="deadbeef0123456789",
        )
        self.assertNotEqual(base_key, id_preset["cache_key"])

        # 4. Changing repair_policy must change key
        id_policy = compute_restoration_cache_identity(
            source=self.test_dir / "source.mkv",
            detector="farneback_3f",
            checkpoint_sha256="abc12345",
            preset="balanced",
            repair_policy=RepairInclusionPolicy.HIGH_RECALL,
            repair_backend="telea",
            effective_mask_hash="deadbeef0123456789",
        )
        self.assertNotEqual(base_key, id_policy["cache_key"])

        # 5. Changing repair_backend must change key
        id_backend = compute_restoration_cache_identity(
            source=self.test_dir / "source.mkv",
            detector="farneback_3f",
            checkpoint_sha256="abc12345",
            preset="balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="propainter",
            effective_mask_hash="deadbeef0123456789",
        )
        self.assertNotEqual(base_key, id_backend["cache_key"])

        # 6. Changing effective_mask_hash must change key
        id_mask = compute_restoration_cache_identity(
            source=self.test_dir / "source.mkv",
            detector="farneback_3f",
            checkpoint_sha256="abc12345",
            preset="balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="different_mask_hash_9999",
        )
        self.assertNotEqual(base_key, id_mask["cache_key"])

    def test_full_video_pipeline_provenance_integration(self):
        """Verify video pipeline produces restoration_provenance.json upon repair completion."""
        source = self.test_dir / "source.mkv"
        meta = {"frames": 3, "width": 96, "height": 64, "fps": "12"}
        writer = dust.Writer(source, meta)
        for _ in range(3):
            writer.write(self.corrupted_image)
        writer.close()

        values = {
            "detector": "legacy",
            "repair": "telea",
            "radius": 3,
            "repair_policy": "conservative",
            "license_confirmed": "true",
        }

        with patch.object(dust, "ROOT", self.test_dir):
            # 1. Run mask stage
            mask_report = dust.run(source, values, "mask")
            target_folder = dust.folder_for(source, values)
            self.assertTrue((target_folder / "effective_mask.mkv").is_file())

            # 2. Run repair stage
            repair_report = dust.run(source, values, "repair")
            self.assertTrue(repair_report["repair_complete"])
            self.assertTrue((target_folder / "repaired.mkv").is_file())
            self.assertTrue((target_folder / "restoration_provenance.json").is_file())

            prov = json.loads((target_folder / "restoration_provenance.json").read_text(encoding="utf-8"))
            self.assertEqual(prov["repair_backend"], "telea")
            self.assertEqual(prov["repair_policy"], "conservative")
            self.assertEqual(prov["effective_mask_hash"], dust.digest(target_folder / "effective_mask.mkv"))


if __name__ == "__main__":
    unittest.main()
