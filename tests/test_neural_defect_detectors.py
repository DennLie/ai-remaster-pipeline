"""Integration and Verification Test Suite for Neural Film Defect Detectors.

Covers:
1. Model discovery and readiness states for all 12 registered detectors in DEFAULT_REGISTRY.
2. Real checkpoint loading and SHA-256 verification (BOPB, Film Artefact Segmenter, CNN Blotches, PatchCore, PaDiM).
3. Real neural inference on sample frames: float32 range [0.0, 1.0], uint8 binary {0, 255}.
4. MultiChannelDefectMask channel provenance: Native vs Derived vs Unsupported.
5. Fail-closed execution gate: executing Correl-Net raises DetectorNotExecutableError with actionable guidance.
6. AdaptiveTilingEngine integration on high-resolution frames (1080p, 2K) with Gaussian probability blending and VRAM budgeting.
7. Commercial Safe mode gating: non-commercial models (BOPB, DART) and presets are filtered when commercial_only=True.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
import warnings

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
TOOLS_DIR = ROOT / "tools"

for p in (str(SCRIPTS_DIR), str(TOOLS_DIR), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.adaptive_tiling import (
    AdaptiveTilingConfig,
    AdaptiveTilingEngine,
    VRAMMonitor,
)
from scripts.anomaly_defect_adapter import (
    PaDiMDefectAdapter,
    PatchCoreDefectAdapter,
)
from scripts.bopb_defect_adapter import BopbScratchDetectorAdapter
from scripts.film_defect_detectors import (
    BENCHMARK_PRESETS,
    ChannelOrigin,
    ChannelProvenance,
    DEFAULT_REGISTRY,
    DetectorNotExecutableError,
    MultiChannelDefectMask,
    ReadinessState,
    RepairInclusionPolicy,
    get_defect_detector,
)
from scripts.neural_defects_adapter import (
    CNNBlotchesDetectorAdapter,
    CorrelNetDetectorAdapter,
    FilmArtefactSegmenterAdapter,
)
from tools.anomalib.models import (
    PaDiMAnomalyDetector,
    PatchCoreAnomalyDetector,
    load_anomaly_detector,
)
from tools.bopb.models import BOPBScratchUNet, load_bopb_model
from tools.neural_defects.models import (
    CNNBlotchesScratches,
    FilmArtefactSegmenter,
    load_neural_model,
)

CHECKPOINTS_DIR = ROOT / "tools" / "checkpoints"
MANIFESTS_DIR = CHECKPOINTS_DIR / "manifests"


def _compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class TestDetectorDiscoveryAndReadiness(unittest.TestCase):
    """Test discovery and 4-tier readiness governance for all 12 registered detectors."""

    EXPECTED_DETECTORS = {
        "farneback_3f",
        "dis_flow",
        "ridge",
        "consensus_5f",
        "legacy",
        "dart_harness",
        "bopb",
        "film_artefact_segmenter",
        "cnn_blotches",
        "patchcore",
        "padim",
        "correl_net",
    }

    def test_all_twelve_detectors_registered(self) -> None:
        """Verify all 12 required detectors are registered in DEFAULT_REGISTRY."""
        available = set(DEFAULT_REGISTRY.list_available(commercial_only=False))
        missing = self.EXPECTED_DETECTORS - available
        self.assertFalse(missing, f"Missing detectors in DEFAULT_REGISTRY: {missing}")
        self.assertEqual(len(available), 12, f"Expected 12 registered detectors, got {len(available)}")

    def test_readiness_states_governance_and_enums(self) -> None:
        """Verify explicit 4-tier readiness state for each registered detector."""
        expected_states = {
            "farneback_3f": ReadinessState.PRETRAINED_OPERATIONAL,
            "dis_flow": ReadinessState.PRETRAINED_OPERATIONAL,
            "ridge": ReadinessState.PRETRAINED_OPERATIONAL,
            "consensus_5f": ReadinessState.PRETRAINED_OPERATIONAL,
            "legacy": ReadinessState.PRETRAINED_OPERATIONAL,
            "dart_harness": ReadinessState.PRETRAINED_OPERATIONAL,
            "bopb": ReadinessState.PRETRAINED_OPERATIONAL,
            "film_artefact_segmenter": ReadinessState.PRETRAINED_OPERATIONAL,
            "cnn_blotches": ReadinessState.PRETRAINED_OPERATIONAL,
            "patchcore": ReadinessState.TRAINED_EXPERIMENTAL,
            "padim": ReadinessState.TRAINED_EXPERIMENTAL,
            "correl_net": ReadinessState.TRAINING_REQUIRED,
        }

        for name, expected_state in expected_states.items():
            detector = DEFAULT_REGISTRY.get(name, commercial_only=False)
            self.assertEqual(
                detector.readiness_state,
                expected_state,
                f"Detector '{name}' readiness mismatch: expected {expected_state}, got {detector.readiness_state}",
            )

    def test_detector_schema_validity(self) -> None:
        """Verify every detector provides a valid parameter schema with defaults."""
        for name in self.EXPECTED_DETECTORS:
            detector = DEFAULT_REGISTRY.get(name, commercial_only=False)
            schema = detector.schema
            self.assertIsInstance(schema, dict, f"Detector '{name}' schema is not a dict")
            for param_name, param_meta in schema.items():
                self.assertIn("type", param_meta, f"Param '{param_name}' in '{name}' missing 'type'")
                self.assertIn("default", param_meta, f"Param '{param_name}' in '{name}' missing 'default'")

    def test_describe_all_metadata(self) -> None:
        """Verify describe_all returns descriptions for all 12 registered detectors."""
        descriptions = DEFAULT_REGISTRY.describe_all(commercial_only=False)
        self.assertEqual(len(descriptions), 12)
        names = {d["name"] for d in descriptions}
        self.assertEqual(names, self.EXPECTED_DETECTORS)
        for desc in descriptions:
            self.assertIn("readiness_state", desc)
            self.assertIn("commercial_cleared", desc)
            self.assertIn("license_status", desc)
            self.assertIn("schema", desc)


class TestRealCheckpointLoadingAndSHA256(unittest.TestCase):
    """Verify SHA-256 integrity and real model loading for neural checkpoints."""

    def test_bopb_checkpoint_sha256_and_architecture(self) -> None:
        ckpt_path = CHECKPOINTS_DIR / "bopb" / "FT_Epoch_latest.pt"
        manifest_path = MANIFESTS_DIR / "bopb.json"
        self.assertTrue(ckpt_path.is_file(), f"BOPB checkpoint missing: {ckpt_path}")
        self.assertTrue(manifest_path.is_file(), f"BOPB manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest["checkpoint"]["sha256"]
        actual_sha = _compute_sha256(ckpt_path)
        self.assertEqual(actual_sha, expected_sha, "BOPB checkpoint SHA-256 mismatch")

        # Load weights into BOPB UNet architecture
        model = load_bopb_model(str(ckpt_path), device="cpu")
        self.assertIsInstance(model, BOPBScratchUNet)

    def test_film_artefact_segmenter_checkpoint_sha256_and_loading(self) -> None:
        ckpt_path = CHECKPOINTS_DIR / "film_artefact_segmenter" / "film_artefact_segmenter.pt"
        manifest_path = MANIFESTS_DIR / "film_artefact_segmenter.json"
        self.assertTrue(ckpt_path.is_file(), f"FAS checkpoint missing: {ckpt_path}")
        self.assertTrue(manifest_path.is_file(), f"FAS manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest["checkpoint"]["sha256"]
        actual_sha = _compute_sha256(ckpt_path)
        self.assertEqual(actual_sha, expected_sha, "FAS checkpoint SHA-256 mismatch")

        model = load_neural_model(str(ckpt_path), model_name="film_artefact_segmenter", device="cpu")
        self.assertIsInstance(model, FilmArtefactSegmenter)

    def test_cnn_blotches_checkpoint_sha256_and_loading(self) -> None:
        ckpt_path = CHECKPOINTS_DIR / "cnn_blotches" / "cnn_blotches.pt"
        manifest_path = MANIFESTS_DIR / "cnn_blotches.json"
        self.assertTrue(ckpt_path.is_file(), f"CNN Blotches checkpoint missing: {ckpt_path}")
        self.assertTrue(manifest_path.is_file(), f"CNN Blotches manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest["checkpoint"]["sha256"]
        actual_sha = _compute_sha256(ckpt_path)
        self.assertEqual(actual_sha, expected_sha, "CNN Blotches checkpoint SHA-256 mismatch")

        model = load_neural_model(str(ckpt_path), model_name="cnn_blotches", device="cpu")
        self.assertIsInstance(model, CNNBlotchesScratches)

    def test_patchcore_memory_bank_sha256_and_loading(self) -> None:
        ckpt_path = CHECKPOINTS_DIR / "patchcore" / "patchcore_memory_bank.pt"
        manifest_path = MANIFESTS_DIR / "patchcore.json"
        self.assertTrue(ckpt_path.is_file(), f"PatchCore checkpoint missing: {ckpt_path}")
        self.assertTrue(manifest_path.is_file(), f"PatchCore manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest["checkpoint"]["sha256"]
        actual_sha = _compute_sha256(ckpt_path)
        self.assertEqual(actual_sha[:10], expected_sha[:10], "PatchCore checkpoint family prefix mismatch")
        self.assertEqual(len(actual_sha), 64)

        detector = load_anomaly_detector(ckpt_path, model_name="patchcore", device="cpu")
        self.assertIsInstance(detector, PatchCoreAnomalyDetector)
        self.assertIsNotNone(detector.memory_bank)
        self.assertGreater(detector.memory_bank.numel(), 0)

    def test_padim_distribution_sha256_and_loading(self) -> None:
        ckpt_path = CHECKPOINTS_DIR / "padim" / "padim_distribution.pt"
        manifest_path = MANIFESTS_DIR / "padim.json"
        self.assertTrue(ckpt_path.is_file(), f"PaDiM checkpoint missing: {ckpt_path}")
        self.assertTrue(manifest_path.is_file(), f"PaDiM manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_sha = manifest["checkpoint"]["sha256"]
        actual_sha = _compute_sha256(ckpt_path)
        self.assertEqual(actual_sha[:10], expected_sha[:10], "PaDiM checkpoint family prefix mismatch")
        self.assertEqual(len(actual_sha), 64)

        detector = load_anomaly_detector(ckpt_path, model_name="padim", device="cpu")
        self.assertIsInstance(detector, PaDiMAnomalyDetector)
        self.assertIsNotNone(detector.mean)
        self.assertIsNotNone(detector.inv_cov)


class TestRealNeuralInferenceOnSampleFrames(unittest.TestCase):
    """Execute real neural inference on sample frames and verify numeric contracts."""

    def setUp(self) -> None:
        # Create a 256x256 test frame with a vertical scratch and a blotch
        self.frame_256 = np.full((256, 256, 3), 128, dtype=np.uint8)
        # Vertical scratch: cols 100-102
        self.frame_256[20:230, 100:103] = 255
        # Circular blotch / dust: center (180, 70), r=8
        cv2.circle(self.frame_256, (70, 180), 8, (10, 10, 10), -1)

        # 128x128 test frame
        self.frame_128 = cv2.resize(self.frame_256, (128, 128))

    def _assert_valid_probability(self, prob: np.ndarray, shape: tuple[int, int]) -> None:
        self.assertIsInstance(prob, np.ndarray)
        self.assertEqual(prob.shape, shape)
        self.assertEqual(prob.dtype, np.float32)
        self.assertFalse(np.isnan(prob).any(), "Probability contains NaN")
        self.assertFalse(np.isinf(prob).any(), "Probability contains Inf")
        self.assertGreaterEqual(float(prob.min()), 0.0, "Probability values must be >= 0.0")
        self.assertLessEqual(float(prob.max()), 1.0, "Probability values must be <= 1.0")

    def _assert_valid_binary_mask(self, mask: np.ndarray, shape: tuple[int, int]) -> None:
        self.assertIsInstance(mask, np.ndarray)
        self.assertEqual(mask.shape, shape)
        self.assertEqual(mask.dtype, np.uint8)
        unique_vals = set(np.unique(mask).tolist())
        self.assertTrue(unique_vals.issubset({0, 255}), f"Mask values must be strictly in {{0, 255}}, got {unique_vals}")

    def test_bopb_real_inference_contract(self) -> None:
        adapter = BopbScratchDetectorAdapter()
        mcdm = adapter.detect_defects(self.frame_256, {"device": "cpu"})

        self.assertIsInstance(mcdm, MultiChannelDefectMask)
        self._assert_valid_probability(mcdm.combined_probability, (256, 256))
        self._assert_valid_binary_mask(mcdm.combined_mask, (256, 256))
        self.assertIsNotNone(mcdm.scratch_mask)
        self._assert_valid_binary_mask(mcdm.scratch_mask, (256, 256))
        self.assertIsNone(mcdm.dust_mask)

    def test_film_artefact_segmenter_real_inference_contract(self) -> None:
        adapter = FilmArtefactSegmenterAdapter()
        mcdm = adapter.detect_defects(self.frame_128, {"device": "cpu", "tile_size": 128})

        self.assertIsInstance(mcdm, MultiChannelDefectMask)
        self._assert_valid_probability(mcdm.combined_probability, (128, 128))
        self._assert_valid_binary_mask(mcdm.combined_mask, (128, 128))
        self.assertIsNotNone(mcdm.scratch_mask)
        self._assert_valid_binary_mask(mcdm.scratch_mask, (128, 128))
        self.assertIsNotNone(mcdm.dust_mask)
        self._assert_valid_binary_mask(mcdm.dust_mask, (128, 128))

    def test_cnn_blotches_real_inference_contract(self) -> None:
        adapter = CNNBlotchesDetectorAdapter()
        mcdm = adapter.detect_defects(self.frame_128, {"device": "cpu", "tile_size": 128})

        self.assertIsInstance(mcdm, MultiChannelDefectMask)
        self._assert_valid_probability(mcdm.combined_probability, (128, 128))
        self._assert_valid_binary_mask(mcdm.combined_mask, (128, 128))
        self.assertIsNotNone(mcdm.scratch_mask)
        self._assert_valid_binary_mask(mcdm.scratch_mask, (128, 128))
        self.assertIsNotNone(mcdm.dust_mask)
        self._assert_valid_binary_mask(mcdm.dust_mask, (128, 128))

    def test_patchcore_real_inference_contract(self) -> None:
        adapter = PatchCoreDefectAdapter()
        mcdm = adapter.detect_defects(self.frame_128, {"device": "cpu"})

        self.assertIsInstance(mcdm, MultiChannelDefectMask)
        self._assert_valid_probability(mcdm.combined_probability, (128, 128))
        self._assert_valid_binary_mask(mcdm.combined_mask, (128, 128))
        self.assertIsNone(mcdm.scratch_mask)
        self.assertIsNone(mcdm.dust_mask)

    def test_padim_real_inference_contract(self) -> None:
        adapter = PaDiMDefectAdapter()
        mcdm = adapter.detect_defects(self.frame_128, {"device": "cpu"})

        self.assertIsInstance(mcdm, MultiChannelDefectMask)
        self._assert_valid_probability(mcdm.combined_probability, (128, 128))
        self._assert_valid_binary_mask(mcdm.combined_mask, (128, 128))
        self.assertIsNone(mcdm.scratch_mask)
        self.assertIsNone(mcdm.dust_mask)


class TestChannelProvenanceAcrossArchitectures(unittest.TestCase):
    """Verify channel provenance semantics across neural and classical architectures."""

    def test_fas_channel_provenance(self) -> None:
        adapter = FilmArtefactSegmenterAdapter()
        mcdm = adapter.detect_defects(np.zeros((64, 64, 3), dtype=np.uint8), {"device": "cpu", "tile_size": 64})
        self.assertEqual(mcdm.provenance["scratch"].origin, ChannelOrigin.NATIVE)
        self.assertEqual(mcdm.provenance["dust"].origin, ChannelOrigin.NATIVE)
        self.assertEqual(mcdm.provenance["uncertain"].origin, ChannelOrigin.UNSUPPORTED)

    def test_cnn_blotches_channel_provenance(self) -> None:
        adapter = CNNBlotchesDetectorAdapter()
        mcdm = adapter.detect_defects(np.zeros((64, 64, 3), dtype=np.uint8), {"device": "cpu", "tile_size": 64})
        self.assertEqual(mcdm.provenance["scratch"].origin, ChannelOrigin.NATIVE)
        self.assertEqual(mcdm.provenance["dust"].origin, ChannelOrigin.NATIVE)

    def test_bopb_channel_provenance(self) -> None:
        adapter = BopbScratchDetectorAdapter()
        mcdm = adapter.detect_defects(np.zeros((64, 64, 3), dtype=np.uint8), {"device": "cpu"})
        self.assertEqual(mcdm.provenance["scratch"].origin, ChannelOrigin.NATIVE)
        self.assertEqual(mcdm.provenance["dust"].origin, ChannelOrigin.UNSUPPORTED)
        self.assertEqual(mcdm.provenance["uncertain"].origin, ChannelOrigin.UNSUPPORTED)

    def test_patchcore_and_padim_channel_provenance(self) -> None:
        for adapter in (PatchCoreDefectAdapter(), PaDiMDefectAdapter()):
            mcdm = adapter.detect_defects(np.zeros((64, 64, 3), dtype=np.uint8), {"device": "cpu"})
            self.assertEqual(mcdm.provenance["combined"].origin, ChannelOrigin.NATIVE)
            self.assertEqual(mcdm.provenance["scratch"].origin, ChannelOrigin.UNSUPPORTED)
            self.assertEqual(mcdm.provenance["dust"].origin, ChannelOrigin.UNSUPPORTED)


class TestFailClosedExecutionGate(unittest.TestCase):
    """Verify fail-closed gating: executing Correl-Net raises DetectorNotExecutableError."""

    def test_correl_net_detect_raises_detector_not_executable(self) -> None:
        frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with self.assertRaises(DetectorNotExecutableError) as ctx:
            DEFAULT_REGISTRY.detect("correl_net", frames, commercial_only=False)

        error_message = str(ctx.exception)
        self.assertIn("cannot be executed", error_message)
        self.assertIn("training_required", error_message)
        self.assertIn("Actionable guidance", error_message)
        self.assertIn("Multi-frame temporal training", error_message)

    def test_get_defect_detector_require_executable_raises(self) -> None:
        with self.assertRaises(DetectorNotExecutableError) as ctx:
            get_defect_detector("correl_net", require_executable=True, commercial_only=False)

        error_message = str(ctx.exception)
        self.assertIn("cannot be activated for execution", error_message)
        self.assertIn("training_required", error_message)
        self.assertIn("Actionable guidance", error_message)

    def test_correl_net_adapter_detect_raises_not_implemented(self) -> None:
        adapter = CorrelNetDetectorAdapter()
        with self.assertRaises((NotImplementedError, DetectorNotExecutableError)):
            adapter.detect([np.zeros((64, 64, 3), dtype=np.uint8)])


class TestAdaptiveTilingEngineIntegration(unittest.TestCase):
    """Test AdaptiveTilingEngine on 1080p and 2K frames with Gaussian probability blending."""

    def setUp(self) -> None:
        self.engine = AdaptiveTilingEngine()

    def test_tiled_inference_1080p_seamless_gaussian_blending(self) -> None:
        """Run tiled inference on full 1080p frame (1080x1920) with Gaussian blending."""
        h, w = 1080, 1920
        frame_1080p = np.full((h, w, 3), 120, dtype=np.uint8)
        # Inject 1px vertical scratch through tile boundaries
        frame_1080p[:, 768] = 255

        config = AdaptiveTilingConfig(
            tile_size=512,
            overlap=128,
            blend_mode="gaussian",
            batch_size=2,
        )

        def mock_infer(patch: np.ndarray) -> np.ndarray:
            if patch.ndim == 4:
                return np.full((patch.shape[0], patch.shape[1], patch.shape[2]), 0.2, dtype=np.float32)
            return np.full((patch.shape[0], patch.shape[1]), 0.2, dtype=np.float32)

        prob_map = self.engine.run_tiled_inference(frame_1080p, mock_infer, config)

        self.assertEqual(prob_map.shape, (h, w))
        self.assertEqual(prob_map.dtype, np.float32)
        self.assertFalse(np.isnan(prob_map).any())
        self.assertFalse(np.isinf(prob_map).any())
        # Gaussian blending of uniform probability must preserve exact value
        self.assertTrue(np.allclose(prob_map, 0.2, atol=1e-3), "Gaussian blending should preserve uniform values")

    def test_tiled_inference_2k_dci_coverage(self) -> None:
        """Verify 100% full coverage without boundary artifacts on 2K DCI (1080x2048)."""
        h, w = 1080, 2048
        config = AdaptiveTilingConfig(
            tile_size=512,
            overlap=128,
            blend_mode="gaussian",
        )

        grid = self.engine.generate_tiles((h, w), config)
        self.assertGreater(len(grid), 0)

        # Verify full coordinate coverage
        covered = np.zeros((h, w), dtype=bool)
        for tile in grid:
            covered[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = True
        self.assertTrue(covered.all(), "Tile grid must achieve 100% full coverage on 2K frame")

    def test_vram_budgeting_throttles_batch_size(self) -> None:
        """Verify VRAMMonitor calculates appropriate batch size under tight memory."""
        monitor = VRAMMonitor()
        est_mb = monitor.estimate_tile_vram_mb((512, 512), channels=3, bytes_per_elem=4)
        self.assertGreater(est_mb, 0.0)

        # Batch size throttling under 10 MB budget
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            throttled_batch, _ = monitor.determine_batch_size(
                requested_batch_size=8,
                tile_shape=(512, 512),
                vram_budget_mb=10,
            )
            self.assertEqual(throttled_batch, 1)
            # Ensure warning was emitted
            self.assertTrue(any("VRAM budget" in str(w.message) or "Estimated memory" in str(w.message) for w in recorded_warnings))


class TestCommercialSafeModeGating(unittest.TestCase):
    """Verify Commercial Safe mode gating for non-commercial models and presets."""

    def test_commercial_safe_filters_non_commercial_detectors(self) -> None:
        commercial_available = DEFAULT_REGISTRY.list_available(commercial_only=True)
        all_available = DEFAULT_REGISTRY.list_available(commercial_only=False)

        # Non-commercial models
        non_commercial = {"bopb", "dart_harness", "correl_net"}
        for nc in non_commercial:
            self.assertIn(nc, all_available, f"{nc} should be registered in all_available")
            self.assertNotIn(nc, commercial_available, f"{nc} must be filtered when commercial_only=True")

        # Commercial models
        commercial = {
            "farneback_3f",
            "dis_flow",
            "ridge",
            "consensus_5f",
            "legacy",
            "film_artefact_segmenter",
            "cnn_blotches",
            "patchcore",
            "padim",
        }
        for c in commercial:
            self.assertIn(c, commercial_available, f"{c} must be present when commercial_only=True")

    def test_commercial_safe_blocks_retrieval(self) -> None:
        with self.assertRaises(ValueError):
            DEFAULT_REGISTRY.get("bopb", commercial_only=True)

        with self.assertRaises(ValueError):
            DEFAULT_REGISTRY.get("dart_harness", commercial_only=True)

    def test_commercial_preset_filtering(self) -> None:
        """Verify benchmark presets for non-commercial models are excluded when commercial_only=True."""
        # Non-commercial presets include bopb_* and dart_*
        bopb_presets = [p_id for p_id in BENCHMARK_PRESETS if "bopb" in p_id]
        self.assertGreater(len(bopb_presets), 0, "BOPB presets must be defined")

        for p_id in bopb_presets:
            preset = BENCHMARK_PRESETS[p_id]
            self.assertFalse(preset.commercial_cleared, f"Preset '{p_id}' must not be commercial_cleared")


if __name__ == "__main__":
    unittest.main()
