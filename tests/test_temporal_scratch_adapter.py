"""Unit and integration tests for TemporalScratchDetectorAdapter and safety guards.

Verifies:
1. Adapter discovery, metadata, aliases, and commercial clearance in DefectDetectorRegistry.
2. Parameter schema validation and boundary checks.
3. Contract conformance for MultiChannelDefectMask and DefectMaskResult.
4. Channel provenance consistency (scratch=native/available, dust/uncertain=unsupported/unavailable).
5. All 4 RepairInclusionPolicy behaviors via to_legacy_compat_bridge.
6. 30% overmasking safety guardrail (zeroing mask, refusing auto-apply, metadata flags).
7. Tiling integration delegation.
8. Benchmark preset resolution and configuration.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import film_defect_detectors as fdd
from temporal_defect_adapter import TemporalScratchDetectorAdapter
from temporal_scratch_analyzer import CLEAN_ROI, SCRATCH_ROI, TemporalDefectResult, load_clip


class TemporalScratchDetectorAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = TemporalScratchDetectorAdapter()
        rng = np.random.default_rng(42)
        self.sample_frames = [
            rng.integers(100, 150, size=(64, 96, 3), dtype=np.uint8)
            for _ in range(5)
        ]

    def test_registry_discovery_and_metadata(self) -> None:
        registry = fdd.build_default_registry()
        self.assertTrue(registry.has("temporal_scratch", commercial_only=True))
        self.assertTrue(registry.has("temporal_fissure", commercial_only=True))
        self.assertTrue(registry.has("temporal_flow", commercial_only=True))

        det = registry.get("temporal_scratch")
        self.assertIsInstance(det, TemporalScratchDetectorAdapter)
        self.assertIs(det, registry.get("temporal_fissure"))
        self.assertIs(det, registry.get("temporal_flow"))

        info = det.info()
        self.assertEqual(info["name"], "temporal_scratch")
        self.assertTrue(info["commercial_cleared"])
        self.assertEqual(info["readiness_state"], fdd.ReadinessState.PRETRAINED_OPERATIONAL.value)
        self.assertEqual(info["supported_channels"], ["scratch"])
        self.assertIn("dust", info["unsupported_channels"])
        self.assertIn("uncertain", info["unsupported_channels"])
        self.assertEqual(info["temporal_radius"], 3)

    def test_parameter_schema_validation(self) -> None:
        defaults = self.adapter.validate_config(None)
        self.assertEqual(defaults["sensitivity"], 0.65)
        self.assertEqual(defaults["window_radius"], 3)
        self.assertEqual(defaults["flow_backend"], "dis")
        self.assertEqual(defaults["telecine_dedup_threshold"], 0.8)
        self.assertEqual(defaults["scene_cut_threshold"], 25.0)
        self.assertEqual(defaults["scratch_min_height"], 15)
        self.assertEqual(defaults["scratch_max_width"], 12)
        self.assertEqual(defaults["scratch_tall_override_height"], 35)
        self.assertEqual(defaults["motion_coherence_threshold"], 1.2)

        valid = self.adapter.validate_config({
            "sensitivity": 0.8,
            "window_radius": 2,
            "flow_backend": "farneback",
            "telecine_dedup_threshold": 1.5,
            "scene_cut_threshold": 30.0,
            "scratch_min_height": 20,
            "scratch_max_width": 8,
            "scratch_tall_override_height": 40,
            "motion_coherence_threshold": 2.0,
        })
        self.assertEqual(valid["sensitivity"], 0.8)
        self.assertEqual(valid["window_radius"], 2)
        self.assertEqual(valid["flow_backend"], "farneback")
        self.assertEqual(valid["scene_cut_threshold"], 30.0)
        self.assertEqual(valid["scratch_tall_override_height"], 40)

        with self.assertRaises(ValueError):
            self.adapter.validate_config({"sensitivity": 1.5})
        with self.assertRaises(ValueError):
            self.adapter.validate_config({"window_radius": 0})
        with self.assertRaises(ValueError):
            self.adapter.validate_config({"flow_backend": "invalid_backend"})
        with self.assertRaises(ValueError):
            self.adapter.validate_config({"scene_cut_threshold": -1.0})
        with self.assertRaises(ValueError):
            self.adapter.validate_config({"scratch_tall_override_height": 0})

    def test_small_dimensions_abstain_cleanly(self) -> None:
        # Micro-dimension frames: (10, 12, 3) < 16x16
        small_frames = [
            np.ones((10, 12, 3), dtype=np.uint8) * 128
            for _ in range(5)
        ]
        result = self.adapter.detect(small_frames)
        self.assertIsInstance(result, fdd.DefectMaskResult)
        self.assertEqual(result.metadata.get("status"), "abstain_small_dimension")
        self.assertEqual(len(result.scratch_mask), 5)
        for s in result.scratch_mask:
            self.assertEqual(s.shape, (10, 12))
            self.assertEqual(s.dtype, np.uint8)
            self.assertEqual(np.count_nonzero(s), 0)
        for m in result.mask:
            self.assertEqual(m.shape, (10, 12))
            self.assertEqual(np.count_nonzero(m), 0)
        self.assertIsNotNone(result.multi_channel_masks)
        self.assertEqual(len(result.multi_channel_masks), 5)
        for mcdm in result.multi_channel_masks:
            self.assertEqual(mcdm.combined_probability.shape, (10, 12))
            self.assertEqual(np.count_nonzero(mcdm.combined_mask), 0)

    def test_detect_returns_compliant_result_and_multi_channel_masks(self) -> None:
        result = self.adapter.detect(self.sample_frames)
        self.assertIsInstance(result, fdd.DefectMaskResult)
        self.assertEqual(len(result), len(self.sample_frames))
        self.assertEqual(len(result.scratch_mask), len(self.sample_frames))
        self.assertEqual(len(result.dust_mask), len(self.sample_frames))
        self.assertEqual(len(result.uncertain_mask), len(self.sample_frames))
        self.assertIsNotNone(result.multi_channel_masks)
        self.assertEqual(len(result.multi_channel_masks), len(self.sample_frames))

        h, w = self.sample_frames[0].shape[:2]
        for t, mcdm in enumerate(result.multi_channel_masks):
            self.assertIsInstance(mcdm, fdd.MultiChannelDefectMask)
            self.assertEqual(mcdm.combined_probability.shape, (h, w))
            self.assertEqual(mcdm.combined_probability.dtype, np.float32)
            self.assertTrue(np.all((mcdm.combined_mask == 0) | (mcdm.combined_mask == 255)))
            self.assertIsNotNone(mcdm.scratch_mask)
            self.assertIsNone(mcdm.dust_mask)
            self.assertIsNone(mcdm.uncertain_mask)

            # Check provenance contract
            self.assertIn("scratch", mcdm.provenance)
            self.assertEqual(mcdm.provenance["scratch"].origin, fdd.ChannelOrigin.NATIVE)
            self.assertTrue(mcdm.provenance["scratch"].is_available)
            self.assertEqual(mcdm.provenance["scratch"].source_algorithm, "temporal_scratch_analyzer")

            self.assertIn("dust", mcdm.provenance)
            self.assertEqual(mcdm.provenance["dust"].origin, fdd.ChannelOrigin.UNSUPPORTED)
            self.assertFalse(mcdm.provenance["dust"].is_available)

            self.assertIn("uncertain", mcdm.provenance)
            self.assertEqual(mcdm.provenance["uncertain"].origin, fdd.ChannelOrigin.UNSUPPORTED)
            self.assertFalse(mcdm.provenance["uncertain"].is_available)

            # Dust and uncertain masks in DefectMaskResult are zero arrays
            self.assertEqual(np.count_nonzero(result.dust_mask[t]), 0)
            self.assertEqual(np.count_nonzero(result.uncertain_mask[t]), 0)

    def test_legacy_compatibility_bridge_policies(self) -> None:
        result = self.adapter.detect(self.sample_frames)
        for policy in fdd.RepairInclusionPolicy:
            s_list, d_list, manifests = result.to_legacy_compat_bridge(policy)
            self.assertEqual(len(s_list), len(self.sample_frames))
            self.assertEqual(len(d_list), len(self.sample_frames))
            self.assertEqual(len(manifests), len(self.sample_frames))
            for s, d, m in zip(s_list, d_list, manifests):
                self.assertEqual(s.shape, (64, 96))
                self.assertEqual(d.shape, (64, 96))
                if policy in (fdd.RepairInclusionPolicy.CONSERVATIVE, fdd.RepairInclusionPolicy.SCRATCH_ONLY):
                    # Dust channel is strictly zero
                    self.assertEqual(np.count_nonzero(d), 0)

    def test_safety_guard_refuses_overmasking(self) -> None:
        h, w = 64, 96
        overmask = np.full((h, w), 255, dtype=np.uint8)  # 100% coverage (> 30%)
        prob = np.full((h, w), 0.95, dtype=np.float32)

        mock_defect_result = TemporalDefectResult(
            scratch_mask=overmask,
            probability_map=prob,
            background=np.zeros((h, w), dtype=np.uint8),
            metadata={"status": "synthetic_overmask"},
        )

        with patch("temporal_defect_adapter.TemporalScratchAnalyzer.analyze_frame", return_value=mock_defect_result):
            res = self.adapter.detect(self.sample_frames)

            # Safety guard triggers
            self.assertTrue(res.metadata["safety_guard_triggered"])
            self.assertEqual(res.metadata["safety_guard_violations"], len(self.sample_frames))
            self.assertFalse(res.metadata["safety_guard_compliant"])

            # Every scratch mask is zeroed to prevent damage
            for s in res.scratch_mask:
                self.assertEqual(np.count_nonzero(s), 0)
            for m in res.mask:
                self.assertEqual(np.count_nonzero(m), 0)

            # Metadata in MultiChannelDefectMask reflects the violation
            for mcdm in res.multi_channel_masks:
                self.assertTrue(mcdm.metadata["safety_guard_triggered"])
                self.assertEqual(mcdm.metadata["safety_guard_limit"], 0.30)
                self.assertGreater(mcdm.metadata["safety_guard_coverage"], 0.30)

    def test_preset_temporal_scratch_balanced(self) -> None:
        preset = fdd.get_preset("temporal_scratch_balanced")
        self.assertIsInstance(preset, fdd.DefectDetectorPreset)
        self.assertEqual(preset.detector, "temporal_scratch")
        self.assertEqual(preset.label, "Temporal Scratch (Balanced)")
        self.assertTrue(preset.commercial_cleared)
        self.assertEqual(preset.repair_policy, fdd.RepairInclusionPolicy.SCRATCH_ONLY)
        self.assertEqual(preset.params["sensitivity"], 0.65)
        self.assertEqual(preset.params["flow_backend"], "dis")
        self.assertEqual(preset.params["window_radius"], 3)
        self.assertIn("Motion-compensated", preset.description)

    @unittest.skipUnless((ROOT / "evaluation" / "Neighbors_72.mkv").is_file(), "Neighbors_72.mkv not found")
    def test_end_to_end_72_frame_execution(self) -> None:
        video_path = ROOT / "evaluation" / "Neighbors_72.mkv"
        frames = load_clip(video_path)
        self.assertEqual(len(frames), 72)

        detector = fdd.DEFAULT_REGISTRY.get("temporal_scratch")
        result = detector.detect(frames)

        self.assertIsInstance(result, fdd.DefectMaskResult)
        self.assertEqual(len(result), 72)

        # Confirm all 4 mask arrays are produced without crashes or dtype errors
        self.assertEqual(len(result.scratch_mask), 72)
        self.assertEqual(len(result.dust_mask), 72)
        self.assertEqual(len(result.uncertain_mask), 72)
        self.assertEqual(len(result.mask), 72)

        for t in range(72):
            s_mask = result.scratch_mask[t]
            d_mask = result.dust_mask[t]
            u_mask = result.uncertain_mask[t]
            eff_mask = result.mask[t]

            self.assertEqual(s_mask.shape, (720, 960))
            self.assertEqual(s_mask.dtype, np.uint8)
            self.assertTrue(np.all((s_mask == 0) | (s_mask == 255)))

            self.assertEqual(d_mask.shape, (720, 960))
            self.assertEqual(d_mask.dtype, np.uint8)
            self.assertEqual(np.count_nonzero(d_mask), 0)

            self.assertEqual(u_mask.shape, (720, 960))
            self.assertEqual(u_mask.dtype, np.uint8)
            self.assertEqual(np.count_nonzero(u_mask), 0)

            self.assertEqual(eff_mask.shape, (720, 960))
            self.assertEqual(eff_mask.dtype, np.uint8)
            self.assertTrue(np.all((eff_mask == 0) | (eff_mask == 255)))

        # Verify bridge policies yield matching effective masks
        for policy in fdd.RepairInclusionPolicy:
            s_list, d_list, manifests = result.to_legacy_compat_bridge(policy)
            self.assertEqual(len(s_list), 72)
            self.assertEqual(len(d_list), 72)
            for t in range(72):
                bridge_eff = np.bitwise_or(s_list[t], d_list[t])
                self.assertEqual(bridge_eff.dtype, np.uint8)
                self.assertEqual(bridge_eff.shape, (720, 960))

        # Check qualitative detection on Frame 25 ground truth ROI
        f25_scratch = result.scratch_mask[25]
        s_ymin, s_xmin, s_ymax, s_xmax = (
            SCRATCH_ROI["ymin"],
            SCRATCH_ROI["xmin"],
            SCRATCH_ROI["ymax"],
            SCRATCH_ROI["xmax"],
        )
        c_ymin, c_xmin, c_ymax, c_xmax = (
            CLEAN_ROI["ymin"],
            CLEAN_ROI["xmin"],
            CLEAN_ROI["ymax"],
            CLEAN_ROI["xmax"],
        )
        s_pixels = int(np.count_nonzero(f25_scratch[s_ymin:s_ymax, s_xmin:s_xmax]))
        c_pixels = int(np.count_nonzero(f25_scratch[c_ymin:c_ymax, c_xmin:c_xmax]))
        self.assertGreaterEqual(s_pixels, 50, f"Expected >= 50 scratch pixels in ROI, got {s_pixels}")
        self.assertLess(c_pixels, 5, f"Expected < 5 clean ROI false alarms, got {c_pixels}")

        # Check safety guard metadata
        self.assertTrue(result.metadata.get("safety_guard_compliant", False))
        self.assertFalse(result.metadata.get("safety_guard_triggered", True))


if __name__ == "__main__":
    unittest.main()
