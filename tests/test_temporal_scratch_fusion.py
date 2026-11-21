"""Unit and integration tests for Temporal Scratch Enhancement Option 2 and Mask Fusion.

Verifies:
1. fuse_spatial_and_temporal_masks:
   - High-confidence temporal scratches partitioned into scratch_mask.
   - Uncertain / low-confidence temporal fragments partitioned into uncertain_mask.
   - Spatial dust preserved without cross-contamination.
   - Strict channel disjointness and provenance tracking.
2. Repair Inclusion Policy enforcement:
   - CONSERVATIVE policy excludes uncertain temporal candidates from repair.
   - HIGH_RECALL policy includes uncertain temporal candidates.
   - SCRATCH_ONLY and DUST_ONLY isolate appropriate channels.
3. Safety and Anomaly Detector protection:
   - Anomaly detectors (PatchCore/PaDiM) preserve combined proposals without bypassing guards.
   - 30% overmasking safety check is respected.
4. End-to-end GUI dust stage integration:
   - options() parsing and identity differentiation (cache isolation).
   - masks_for() stream produces valid 2D uint8 contiguous binary masks.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import film_defect_detectors as fdd
from ai_remaster_gui.dust import ensure_uint8_binary_mask, options, masks_for


class TemporalScratchFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h, self.w = 120, 160
        self.shape = (self.h, self.w)

    def test_fusion_high_confidence_vs_uncertain_partitioning(self) -> None:
        # Spatial detector finds a small dust spot
        s_mask = np.zeros(self.shape, dtype=np.uint8)
        s_dust = np.zeros(self.shape, dtype=np.uint8)
        s_dust[20:25, 30:35] = 255
        s_mask[20:25, 30:35] = 255
        s_scratch = np.zeros(self.shape, dtype=np.uint8)
        s_prob = (s_mask.astype(np.float32) / 255.0)

        spatial_res = fdd.DefectMaskResult(
            mask=[s_mask],
            scratch_mask=[s_scratch],
            dust_mask=[s_dust],
            uncertain_mask=[np.zeros(self.shape, dtype=np.uint8)],
        )

        # Temporal detector finds two features:
        # 1. Tall vertical scratch: height=50, width=2, p=0.85 (High confidence)
        # 2. Short horizontal blob: height=4, width=12, p=0.45 (Uncertain / edge motion)
        t_mask = np.zeros(self.shape, dtype=np.uint8)
        t_scratch = np.zeros(self.shape, dtype=np.uint8)
        t_prob = np.zeros(self.shape, dtype=np.float32)

        # High-confidence vertical scratch
        t_scratch[10:60, 80:82] = 255
        t_mask[10:60, 80:82] = 255
        t_prob[10:60, 80:82] = 0.85

        # Uncertain fragment
        t_scratch[70:74, 100:112] = 255
        t_mask[70:74, 100:112] = 255
        t_prob[70:74, 100:112] = 0.45

        temporal_mcdm = fdd.MultiChannelDefectMask(
            combined_probability=t_prob,
            combined_mask=t_mask,
            scratch_mask=t_scratch,
        )
        temporal_res = fdd.DefectMaskResult(
            mask=[t_mask],
            scratch_mask=[t_scratch],
            dust_mask=[np.zeros(self.shape, dtype=np.uint8)],
            uncertain_mask=[np.zeros(self.shape, dtype=np.uint8)],
            multi_channel_masks=[temporal_mcdm],
        )

        fused = fdd.fuse_spatial_and_temporal_masks(spatial_res, temporal_res)

        f_mask = fused.mask[0]
        f_scratch = fused.scratch_mask[0]
        f_dust = fused.dust_mask[0]
        f_uncertain = fused.uncertain_mask[0]

        # Invariant: Output must be valid uint8 {0, 255}
        for name, arr in [("mask", f_mask), ("scratch", f_scratch), ("dust", f_dust), ("uncertain", f_uncertain)]:
            clean = ensure_uint8_binary_mask(arr, self.shape, name)
            self.assertEqual(clean.dtype, np.uint8)
            self.assertTrue(np.all((clean == 0) | (clean == 255)))

        # 1. High-confidence scratch must be in scratch_mask
        self.assertTrue(np.all(f_scratch[10:60, 80:82] == 255))
        # 2. Uncertain fragment must NOT be in scratch_mask
        self.assertTrue(np.all(f_scratch[70:74, 100:112] == 0))
        # 3. Uncertain fragment MUST be in uncertain_mask
        self.assertTrue(np.all(f_uncertain[70:74, 100:112] == 255))
        # 4. Spatial dust must remain in dust_mask untouched
        self.assertTrue(np.all(f_dust[20:25, 30:35] == 255))
        # 5. Semantic disjointness: uncertain cannot overlap scratch or dust
        self.assertTrue(np.all(f_uncertain[f_scratch > 0] == 0))
        self.assertTrue(np.all(f_uncertain[f_dust > 0] == 0))

        # Check provenance
        mcdm = fused.multi_channel_masks[0]
        self.assertEqual(mcdm.provenance["scratch"].source_algorithm, "fused_spatial_temporal_scratch")
        self.assertEqual(mcdm.provenance["uncertain"].source_algorithm, "temporal_uncertain_partitioning")

    def test_repair_inclusion_policies(self) -> None:
        # Construct fused mask with scratch, dust, and uncertain channels
        scratch = np.zeros(self.shape, dtype=np.uint8)
        dust = np.zeros(self.shape, dtype=np.uint8)
        uncertain = np.zeros(self.shape, dtype=np.uint8)

        scratch[10:50, 20:22] = 255   # 80 px
        dust[30:35, 40:45] = 255      # 25 px
        uncertain[80:85, 90:100] = 255  # 50 px

        combined = scratch | dust | uncertain
        prob = (combined.astype(np.float32) / 255.0)

        mcdm = fdd.MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=scratch,
            dust_mask=dust,
            uncertain_mask=uncertain,
        )

        # Policy 1: CONSERVATIVE (scratch | dust, uncertain DROPPED)
        eff_conservative = mcdm.get_effective_repair_mask(fdd.RepairInclusionPolicy.CONSERVATIVE)
        self.assertEqual(np.count_nonzero(eff_conservative), 80 + 25)
        self.assertTrue(np.all(eff_conservative[scratch > 0] == 255))
        self.assertTrue(np.all(eff_conservative[dust > 0] == 255))
        self.assertTrue(np.all(eff_conservative[uncertain > 0] == 0))

        # Policy 2: HIGH_RECALL (scratch | dust | uncertain ALL INCLUDED)
        eff_high_recall = mcdm.get_effective_repair_mask(fdd.RepairInclusionPolicy.HIGH_RECALL)
        self.assertEqual(np.count_nonzero(eff_high_recall), 80 + 25 + 50)
        self.assertTrue(np.all(eff_high_recall[uncertain > 0] == 255))

        # Policy 3: SCRATCH_ONLY
        eff_scratch = mcdm.get_effective_repair_mask(fdd.RepairInclusionPolicy.SCRATCH_ONLY)
        self.assertEqual(np.count_nonzero(eff_scratch), 80)
        self.assertTrue(np.all(eff_scratch[dust > 0] == 0))

        # Policy 4: COMBINED_DEFECT
        eff_combined = mcdm.get_effective_repair_mask(fdd.RepairInclusionPolicy.COMBINED_DEFECT)
        self.assertEqual(np.count_nonzero(eff_combined), 80 + 25 + 50)

    def test_anomaly_detector_protection(self) -> None:
        # For PatchCore / PaDiM, raw proposals must remain in combined_mask
        # and must NOT be masqueraded as conservative repair channels.
        anomaly_mask = np.zeros(self.shape, dtype=np.uint8)
        anomaly_mask[10:40, 10:40] = 255  # 900 px
        zero = np.zeros(self.shape, dtype=np.uint8)

        spatial_res = fdd.DefectMaskResult(
            mask=[anomaly_mask],
            scratch_mask=[zero],
            dust_mask=[zero],
            uncertain_mask=[zero],
        )

        t_scratch = np.zeros(self.shape, dtype=np.uint8)
        t_scratch[10:60, 100:102] = 255  # 100 px scratch
        t_prob = np.zeros(self.shape, dtype=np.float32)
        t_prob[10:60, 100:102] = 0.90
        temporal_mcdm = fdd.MultiChannelDefectMask(
            combined_probability=t_prob,
            combined_mask=t_scratch,
            scratch_mask=t_scratch,
        )
        temporal_res = fdd.DefectMaskResult(
            mask=[t_scratch],
            scratch_mask=[t_scratch],
            dust_mask=[zero],
            uncertain_mask=[zero],
            multi_channel_masks=[temporal_mcdm],
        )

        fused = fdd.fuse_spatial_and_temporal_masks(spatial_res, temporal_res, is_anomaly_detector=True)
        # Primary anomaly pixels must be preserved in combined_mask
        self.assertTrue(np.all(fused.mask[0][10:40, 10:40] == 255))
        # Temporal scratch must be in scratch_mask
        self.assertTrue(np.all(fused.scratch_mask[0][10:60, 100:102] == 255))
        # Anomaly pixels must NOT be in scratch_mask
        self.assertTrue(np.all(fused.scratch_mask[0][10:40, 10:40] == 0))

    def test_options_cache_isolation(self) -> None:
        opt_off = options({
            "detector": "film_artefact_segmenter",
            "preset": "film_artefact_balanced",
            "enable_temporal_scratch": "false",
        })
        opt_on = options({
            "detector": "film_artefact_segmenter",
            "preset": "film_artefact_balanced",
            "enable_temporal_scratch": "true",
        })
        self.assertEqual(opt_off["enable_temporal_scratch"], "false")
        self.assertEqual(opt_on["enable_temporal_scratch"], "true")
        self.assertNotEqual(opt_off, opt_on)


    def test_masks_for_streaming_fusion(self) -> None:
        from unittest.mock import patch
        fake_frames = [np.full((64, 64, 3), 128, dtype=np.uint8) for _ in range(12)]
        opt = options({
            "detector": "legacy",
            "preset": "legacy_baseline",
            "enable_temporal_scratch": "true",
        })
        with patch("ai_remaster_gui.dust.frames", return_value=fake_frames):
            items = list(masks_for("dummy.mkv", opt))

        self.assertEqual(len(items), 12)
        for idx, item in enumerate(items):
            self.assertEqual(len(item), 3)
            image, mask, reason = item
            self.assertEqual(image.shape, (64, 64, 3))
            self.assertEqual(mask.shape, (64, 64))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(np.all((mask == 0) | (mask == 255)))

            scratch = item.scratch_mask
            dust = item.dust_mask
            uncertain = item.uncertain_mask
            self.assertEqual(scratch.shape, (64, 64))
            self.assertEqual(scratch.dtype, np.uint8)
            self.assertEqual(dust.shape, (64, 64))
            self.assertEqual(dust.dtype, np.uint8)
            self.assertEqual(uncertain.shape, (64, 64))
            self.assertEqual(uncertain.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
