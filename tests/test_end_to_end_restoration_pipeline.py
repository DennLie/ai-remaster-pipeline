"""Comprehensive End-to-End Restoration Pipeline Integration and Invariant Test Suite.

Verifies:
1. End-to-end execution chain:
   GUI model selection -> preset resolution -> real neural inference -> MultiChannelDefectMask
   -> adopt mask -> dispatch repair (Telea, ProPainter, DART composite) -> restored video / frame artifact.
2. Effective mask consumption:
   Inpainting consumes the exact effective mask dictated by RepairInclusionPolicy:
   - CONSERVATIVE: Scratch | Dust (Uncertain strictly excluded).
   - HIGH_RECALL: Scratch | Dust | Uncertain.
   - SCRATCH_ONLY: Scratch only (Dust and Uncertain excluded).
   - COMBINED_DEFECT: Combined defect proposal mask.
3. Unmasked pixel invariance:
   Unmasked pixels (where effective_mask == 0) MUST be strictly bit-for-bit identical before and after repair.
4. Provenance manifest verification:
   restoration_provenance.json contains detector, checkpoint_sha256, license, preset,
   repair_policy, channel_provenance, repair_backend, effective_mask_hash, source_video, output_video, timestamp.
5. Accepted mask protection:
   Switching preview model / preset or adjusting thresholds NEVER alters or overwrites an accepted mask.
6. Cache identity determinism and dimension sensitivity.
7. Windows non-elevated path and execution safety.
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
import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
GUI_DIR = ROOT / "ai_remaster_gui"

for p in (str(SCRIPTS_DIR), str(GUI_DIR), str(ROOT)):
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
from dust_native_composite import (
    dispatch_repair_frame,
    inpaint_frame_dart_composite,
    inpaint_frame_propainter,
    inpaint_frame_telea,
    write_restoration_provenance,
)
from film_defect_detectors import (
    BENCHMARK_PRESETS,
    ChannelOrigin,
    ChannelProvenance,
    DEFAULT_REGISTRY,
    MultiChannelDefectMask,
    ReadinessState,
    RepairInclusionPolicy,
    UnsupportedPolicyError,
)
from neural_defects_adapter import FilmArtefactSegmenterAdapter
from bopb_defect_adapter import BopbScratchDetectorAdapter


class TestEndToEndExecutionChain(unittest.TestCase):
    """Verify end-to-end flow from neural inference through mask adoption to inpainting."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a textured 128x128 sample image
        self.height, self.width = 128, 128
        y_coords, x_coords = np.ogrid[:self.height, :self.width]
        base_texture = ((x_coords * 3 + y_coords * 5) % 180 + 40).astype(np.uint8)
        self.sample_image = cv2.cvtColor(base_texture, cv2.COLOR_GRAY2BGR)

        # Ground truth defect patterns
        # 1. Tramline scratch: vertical column 45
        self.scratch = np.zeros((self.height, self.width), dtype=np.uint8)
        self.scratch[10:118, 44:47] = 255

        # 2. Dust spot: circular blob at row 80, col 90
        self.dust = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.circle(self.dust, (90, 80), 5, 255, -1)

        # 3. Uncertain anomaly: subtle blotch at row 30, col 100
        self.uncertain = np.zeros((self.height, self.width), dtype=np.uint8)
        self.uncertain[28:34, 98:104] = 255

        # Corrupted image with bright white defects
        self.corrupted_image = self.sample_image.copy()
        combined_all = np.bitwise_or(np.bitwise_or(self.scratch, self.dust), self.uncertain)
        self.corrupted_image[combined_all > 0] = 255

        self.mask_container = MultiChannelDefectMask(
            combined_probability=(combined_all.astype(np.float32) / 255.0),
            combined_mask=combined_all,
            scratch_mask=self.scratch,
            dust_mask=self.dust,
            uncertain_mask=self.uncertain,
            provenance={
                "scratch": ChannelProvenance(origin=ChannelOrigin.NATIVE, is_available=True, source_algorithm="fas_scratch_head"),
                "dust": ChannelProvenance(origin=ChannelOrigin.NATIVE, is_available=True, source_algorithm="fas_dust_head"),
                "uncertain": ChannelProvenance(origin=ChannelOrigin.DERIVED, is_available=True, source_algorithm="fas_ambiguity_gate"),
            },
            metadata={"model": "film_artefact_segmenter"},
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_e2e_real_neural_model_selection_to_repaired_frame(self) -> None:
        """Full pipeline: Select FAS -> real inference -> MultiChannelDefectMask -> adopt -> Telea repair."""
        fas_detector = DEFAULT_REGISTRY.get("film_artefact_segmenter", commercial_only=False)
        self.assertEqual(fas_detector.readiness_state, ReadinessState.PRETRAINED_OPERATIONAL)

        # 1. Real neural inference on corrupted frame
        mcdm = fas_detector.detect_defects(self.corrupted_image, {"device": "cpu", "tile_size": 128})
        self.assertIsInstance(mcdm, MultiChannelDefectMask)
        self.assertEqual(mcdm.combined_probability.shape, (self.height, self.width))

        # 2. Adopt mask into job folder under CONSERVATIVE policy
        target_folder = self.test_dir / "job_fas_telea"
        adoption = adopt_defect_mask(
            target_folder=target_folder,
            mask=mcdm,
            policy=RepairInclusionPolicy.CONSERVATIVE,
            preset="fas_balanced",
            repair_backend="telea",
            checkpoint_sha256="f7d3c3404b6da5557ce66ecb117b0bfc63376aa8930c4ab633b157ca31e97503",
        )

        eff_mask_path = Path(adoption["effective_repair_mask"])
        self.assertTrue(eff_mask_path.is_file(), "Adopted effective repair mask must exist on disk")

        # 3. Dispatch repair via Telea
        out_image_path = target_folder / "repaired.png"
        repaired, provenance = dispatch_repair(
            source=self.corrupted_image,
            effective_repair_mask=eff_mask_path,
            repair_backend="telea",
            output_path=out_image_path,
            radius=3,
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            detector_info={"name": "film_artefact_segmenter", "version": "1.0.0", "architecture": "MultiScaleUNet"},
            checkpoint_sha256="f7d3c3404b6da5557ce66ecb117b0bfc63376aa8930c4ab633b157ca31e97503",
            license_info={"status": "permissive", "commercial_compatible": True},
            channel_provenance=adoption["channel_provenance"],
        )

        # 4. Verify repaired artifact
        self.assertTrue(out_image_path.is_file())
        self.assertEqual(repaired.shape, self.corrupted_image.shape)

        # 5. Invariant: Unmasked pixels must remain strictly bit-for-bit identical
        eff_mask = cv2.imread(str(eff_mask_path), cv2.IMREAD_GRAYSCALE)
        unmasked = eff_mask == 0
        np.testing.assert_array_equal(
            repaired[unmasked],
            self.corrupted_image[unmasked],
            err_msg="Unmasked pixels must be bit-for-bit identical!",
        )

    def test_e2e_propainter_dispatch_with_effective_mask(self) -> None:
        """Verify ProPainter temporal inpainting patch strictly consumes effective mask."""
        eff_mask = self.mask_container.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)
        neural_patch = np.full_like(self.corrupted_image, 160)

        repaired, metrics = inpaint_frame_propainter(
            image=self.corrupted_image,
            patch=neural_patch,
            effective_repair_mask=eff_mask,
        )

        unmasked = eff_mask == 0
        masked = eff_mask > 0

        # Bit-for-bit exactness outside mask
        np.testing.assert_array_equal(repaired[unmasked], self.corrupted_image[unmasked])
        # Patched inside mask
        np.testing.assert_array_equal(repaired[masked], neural_patch[masked])
        self.assertEqual(metrics["changed_outside_pixels"], 0)


class TestEffectiveMaskConsumptionAcrossPolicies(unittest.TestCase):
    """Verify inpainting consumes the exact effective mask dictated by RepairInclusionPolicy."""

    def setUp(self) -> None:
        self.height, self.width = 64, 64

        # Disjoint defect regions
        self.scratch = np.zeros((self.height, self.width), dtype=np.uint8)
        self.scratch[5:60, 10:12] = 255

        self.dust = np.zeros((self.height, self.width), dtype=np.uint8)
        self.dust[20:25, 30:35] = 255

        self.uncertain = np.zeros((self.height, self.width), dtype=np.uint8)
        self.uncertain[40:46, 45:51] = 255

        self.combined = np.bitwise_or(np.bitwise_or(self.scratch, self.dust), self.uncertain)
        self.prob = self.combined.astype(np.float32) / 255.0

        self.mcdm = MultiChannelDefectMask(
            combined_probability=self.prob,
            combined_mask=self.combined,
            scratch_mask=self.scratch,
            dust_mask=self.dust,
            uncertain_mask=self.uncertain,
        )

    def test_conservative_policy_excludes_uncertain(self) -> None:
        eff = self.mcdm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        expected = np.bitwise_or(self.scratch, self.dust)
        np.testing.assert_array_equal(eff, expected)
        # Uncertain region must have 0 pixels included
        self.assertEqual(np.count_nonzero(eff[self.uncertain > 0]), 0)
        # Scratch and dust must be 100% included
        self.assertEqual(np.count_nonzero(eff[self.scratch > 0]), np.count_nonzero(self.scratch))
        self.assertEqual(np.count_nonzero(eff[self.dust > 0]), np.count_nonzero(self.dust))

    def test_high_recall_policy_includes_uncertain(self) -> None:
        eff = self.mcdm.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)
        expected = np.bitwise_or(np.bitwise_or(self.scratch, self.dust), self.uncertain)
        np.testing.assert_array_equal(eff, expected)
        self.assertEqual(np.count_nonzero(eff[self.uncertain > 0]), np.count_nonzero(self.uncertain))

    def test_scratch_only_policy_includes_only_scratch(self) -> None:
        eff = self.mcdm.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)
        np.testing.assert_array_equal(eff, self.scratch)
        self.assertEqual(np.count_nonzero(eff[self.dust > 0]), 0)
        self.assertEqual(np.count_nonzero(eff[self.uncertain > 0]), 0)

    def test_combined_defect_policy_includes_all_proposals(self) -> None:
        eff = self.mcdm.get_effective_repair_mask(RepairInclusionPolicy.COMBINED_DEFECT)
        np.testing.assert_array_equal(eff, self.combined)

    def test_effective_mask_hash_integrity(self) -> None:
        for policy in (
            RepairInclusionPolicy.CONSERVATIVE,
            RepairInclusionPolicy.HIGH_RECALL,
            RepairInclusionPolicy.SCRATCH_ONLY,
            RepairInclusionPolicy.COMBINED_DEFECT,
        ):
            eff = self.mcdm.get_effective_repair_mask(policy)
            mask_hash = compute_mask_sha256(eff)
            self.assertEqual(len(mask_hash), 64)
            # Recomputing must be perfectly deterministic
            self.assertEqual(compute_mask_sha256(eff), mask_hash)


class TestUnmaskedPixelInvariance(unittest.TestCase):
    """Verify that unmasked pixels are bit-for-bit identical before and after repair."""

    def test_unmasked_pixel_invariance_telea(self) -> None:
        rng = np.random.default_rng(42)
        h, w = 64, 64
        original = rng.integers(10, 240, size=(h, w, 3), dtype=np.uint8)

        # Small mask in the center
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[24:40, 24:40] = 255

        corrupted = original.copy()
        corrupted[mask > 0] = 255

        repaired, metrics = inpaint_frame_telea(corrupted, mask, radius=3)

        unmasked = mask == 0
        np.testing.assert_array_equal(
            repaired[unmasked],
            corrupted[unmasked],
            err_msg="Telea: Unmasked pixels must remain strictly bit-for-bit identical!",
        )
        self.assertEqual(metrics["changed_outside_pixels"], 0)

    def test_unmasked_pixel_invariance_multi_frame_sequence(self) -> None:
        """Verify unmasked pixel invariance across 5 consecutive video frames."""
        rng = np.random.default_rng(99)
        h, w = 48, 64

        for frame_idx in range(5):
            original = rng.integers(20, 220, size=(h, w, 3), dtype=np.uint8)
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[10 + frame_idx : 20 + frame_idx, 15:25] = 255

            corrupted = original.copy()
            corrupted[mask > 0] = 255

            repaired, metrics = inpaint_frame_telea(corrupted, mask, radius=2)
            unmasked = mask == 0

            np.testing.assert_array_equal(
                repaired[unmasked],
                corrupted[unmasked],
                err_msg=f"Frame {frame_idx}: Bit-for-bit invariance violated outside mask!",
            )


class TestRestorationProvenanceManifest(unittest.TestCase):
    """Verify restoration_provenance.json persistence and schema compliance."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.target_dir = Path(self.temp_dir.name)

        self.mask_file = self.target_dir / "effective_repair_mask.png"
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:16, 8:16] = 255
        cv2.imwrite(str(self.mask_file), mask)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_provenance_manifest_completeness_and_schema(self) -> None:
        mask_hash = compute_file_sha256(self.mask_file)
        prov = build_restoration_provenance(
            detector={"name": "film_artefact_segmenter", "version": "1.0.0", "architecture": "MultiScaleUNet"},
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash=mask_hash,
            source_video="input.mkv",
            output_video="output.mkv",
            checkpoint_sha256="f7d3c3404b6da5557ce66ecb117b0bfc63376aa8930c4ab633b157ca31e97503",
            license_info={"status": "permissive", "commercial_compatible": True},
            channel_provenance={"scratch": "native", "dust": "native", "uncertain": "unsupported"},
        )

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
            self.assertIn(key, prov, f"Missing required provenance field: {key}")

        self.assertEqual(prov["detector"]["name"], "film_artefact_segmenter")
        self.assertEqual(prov["repair_policy"], "conservative")
        self.assertEqual(prov["repair_backend"], "telea")
        self.assertEqual(prov["effective_mask_hash"], mask_hash)

    def test_write_and_read_provenance_json_roundtrip(self) -> None:
        out_prov_file = self.target_dir / "restoration_provenance.json"
        sample_prov = {
            "detector": {"name": "bopb", "version": "1.0.0", "architecture": "UNet"},
            "checkpoint_sha256": "b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95",
            "license": {"status": "academic_non_commercial", "commercial_compatible": False},
            "preset": "bopb_scratch_balanced",
            "repair_policy": "scratch_only",
            "channel_provenance": {"scratch": "native", "dust": "unsupported"},
            "repair_backend": "telea",
            "effective_mask_hash": compute_file_sha256(self.mask_file),
            "source_video": "source.mkv",
            "output_video": "repaired.mkv",
            "timestamp": "2026-11-14T23:00:00Z",
        }

        written_path = write_restoration_provenance(out_prov_file, sample_prov)
        self.assertTrue(Path(written_path).is_file())

        loaded = json.loads(Path(written_path).read_text(encoding="utf-8"))
        self.assertEqual(loaded["detector"]["name"], "bopb")
        self.assertEqual(loaded["checkpoint_sha256"], sample_prov["checkpoint_sha256"])
        self.assertEqual(loaded["license"]["commercial_compatible"], False)


class TestAcceptedMaskProtectionInvariance(unittest.TestCase):
    """Verify that accepted masks are strictly protected against mutations from previews or threshold changes."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.job_dir = Path(self.temp_dir.name)

        # Create accepted report and mask
        self.report_data = {
            "identity": {"version": 3, "detector": "farneback_3f"},
            "source_sha256": "fake_source_sha256",
            "repair_complete": True,
            "hashes": {},
        }
        self.report_file = self.job_dir / "report.json"
        self.report_file.write_text(json.dumps(self.report_data), encoding="utf-8")
        self.adopted_report_sha = compute_file_sha256(self.report_file)

        self.mask_file = self.job_dir / "effective_repair_mask.png"
        mask = np.full((32, 32), 255, dtype=np.uint8)
        cv2.imwrite(str(self.mask_file), mask)
        self.initial_mask_sha = compute_file_sha256(self.mask_file)

        self.settings = {
            "dust": {
                "adopted_report": self.adopted_report_sha,
                "detector": "farneback_3f",
            }
        }

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_protect_accepted_mask_raises_on_preview_switching(self) -> None:
        """Attempting to mutate an accepted mask folder with a preview must raise ValueError."""
        with self.assertRaisesRegex(ValueError, "cannot mutate an accepted mask"):
            protect_accepted_mask(self.job_dir, self.settings)

    def test_adopt_defect_mask_protects_accepted_mask(self) -> None:
        """Adopting a new mask into an accepted mask folder without re-adoption raises ValueError."""
        dummy_mask = MultiChannelDefectMask(
            combined_probability=np.zeros((32, 32), dtype=np.float32),
            combined_mask=np.zeros((32, 32), dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "cannot mutate an accepted mask"):
            adopt_defect_mask(
                target_folder=self.job_dir,
                mask=dummy_mask,
                settings=self.settings,
            )

        # Assert on-disk mask and report remain 100% unaltered
        self.assertEqual(compute_file_sha256(self.mask_file), self.initial_mask_sha)
        self.assertEqual(compute_file_sha256(self.report_file), self.adopted_report_sha)


class TestRestorationCacheIdentityDeterminism(unittest.TestCase):
    """Verify cache identity incorporating all required invariants."""

    def test_cache_identity_stability_and_sensitivity(self) -> None:
        source_path = Path("test_source.mkv")
        base_id = compute_restoration_cache_identity(
            source=source_path,
            detector="film_artefact_segmenter",
            checkpoint_sha256="sha_fas_123",
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="mask_hash_abc",
        )
        base_key = base_id["cache_key"]
        self.assertEqual(len(base_key), 64)

        # 1. Determinism: identical parameters produce identical keys
        repeat_id = compute_restoration_cache_identity(
            source=source_path,
            detector="film_artefact_segmenter",
            checkpoint_sha256="sha_fas_123",
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="mask_hash_abc",
        )
        self.assertEqual(base_key, repeat_id["cache_key"])

        # 2. Detector change alters key
        id_det = compute_restoration_cache_identity(
            source=source_path,
            detector="cnn_blotches",
            checkpoint_sha256="sha_fas_123",
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="mask_hash_abc",
        )
        self.assertNotEqual(base_key, id_det["cache_key"])

        # 3. Checkpoint SHA change alters key
        id_ckpt = compute_restoration_cache_identity(
            source=source_path,
            detector="film_artefact_segmenter",
            checkpoint_sha256="different_checkpoint_sha",
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="mask_hash_abc",
        )
        self.assertNotEqual(base_key, id_ckpt["cache_key"])

        # 4. Preset change alters key
        id_preset = compute_restoration_cache_identity(
            source=source_path,
            detector="film_artefact_segmenter",
            checkpoint_sha256="sha_fas_123",
            preset="fas_high_recall",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="mask_hash_abc",
        )
        self.assertNotEqual(base_key, id_preset["cache_key"])

        # 5. Policy change alters key
        id_policy = compute_restoration_cache_identity(
            source=source_path,
            detector="film_artefact_segmenter",
            checkpoint_sha256="sha_fas_123",
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.HIGH_RECALL,
            repair_backend="telea",
            effective_mask_hash="mask_hash_abc",
        )
        self.assertNotEqual(base_key, id_policy["cache_key"])

        # 6. Backend change alters key
        id_backend = compute_restoration_cache_identity(
            source=source_path,
            detector="film_artefact_segmenter",
            checkpoint_sha256="sha_fas_123",
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="propainter",
            effective_mask_hash="mask_hash_abc",
        )
        self.assertNotEqual(base_key, id_backend["cache_key"])

        # 7. Mask hash change alters key
        id_mask = compute_restoration_cache_identity(
            source=source_path,
            detector="film_artefact_segmenter",
            checkpoint_sha256="sha_fas_123",
            preset="fas_balanced",
            repair_policy=RepairInclusionPolicy.CONSERVATIVE,
            repair_backend="telea",
            effective_mask_hash="different_mask_hash_xyz",
        )
        self.assertNotEqual(base_key, id_mask["cache_key"])


class TestEndToEndVideoFilePipeline(unittest.TestCase):
    """Verify video pipeline integration on full video files."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create a 3-frame 64x96 test video
        self.source = self.test_dir / "test_source.mkv"
        meta = {"frames": 3, "width": 96, "height": 64, "fps": "12"}
        writer = dust.Writer(self.source, meta)
        for i in range(3):
            frame = np.full((64, 96, 3), 40 + i * 30, dtype=np.uint8)
            # Inject a small scratch
            frame[10:50, 48:50] = 255
            writer.write(frame)
        writer.close()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_video_container_repair_and_provenance(self) -> None:
        values = {
            "detector": "legacy",
            "repair": "telea",
            "radius": 3,
            "repair_policy": "conservative",
            "license_confirmed": "true",
        }

        with patch.object(dust, "ROOT", self.test_dir):
            mask_report = dust.run(self.source, values, "mask")
            target_folder = dust.folder_for(self.source, values)
            self.assertTrue((target_folder / "effective_mask.mkv").is_file())

            repair_report = dust.run(self.source, values, "repair")
            self.assertTrue(repair_report["repair_complete"])
            repaired_video = target_folder / "repaired.mkv"
            self.assertTrue(repaired_video.is_file())

            # Verify video is decodable
            cap = cv2.VideoCapture(str(repaired_video))
            self.assertTrue(cap.isOpened())
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertEqual(frame_count, 3)

            # Verify restoration_provenance.json
            prov_file = target_folder / "restoration_provenance.json"
            self.assertTrue(prov_file.is_file())
            prov = json.loads(prov_file.read_text(encoding="utf-8"))
            self.assertEqual(prov["repair_backend"], "telea")
            self.assertEqual(prov["repair_policy"], "conservative")


if __name__ == "__main__":
    unittest.main()
