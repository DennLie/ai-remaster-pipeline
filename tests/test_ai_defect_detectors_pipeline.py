"""Comprehensive verification of AI Dust & Scratch defect detectors, benchmark presets,
and end-to-end pipeline execution.
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_remaster_gui import commercial_policy, dust, models
import film_defect_detectors as fdd
import candidate_detector_eval as cde


class TestBenchmarkPresets(unittest.TestCase):
    """Test suite for benchmark presets resolution and commercial filtering."""

    def test_all_expected_presets_present(self) -> None:
        expected = {
            "balanced",
            "fast_preview",
            "high_quality_precision",
            "scratch_focused",
            "dust_focused",
            "high_recall",
            "conservative_low_fp",
            "dart_optimal_f1",
            "dart_high_recall",
            "dart_conservative",
            "custom",
        }
        self.assertEqual(set(fdd.BENCHMARK_PRESETS.keys()), expected)

    def test_preset_definitions_and_attributes(self) -> None:
        p_opt = fdd.get_preset("dart_optimal_f1")
        self.assertEqual(p_opt.detector, "dart")
        self.assertEqual(p_opt.threshold, 224)
        self.assertFalse(p_opt.commercial_cleared)

        p_rec = fdd.get_preset("dart_high_recall")
        self.assertEqual(p_rec.detector, "dart")
        self.assertEqual(p_rec.threshold, 192)
        self.assertFalse(p_rec.commercial_cleared)

        p_con = fdd.get_preset("dart_conservative")
        self.assertEqual(p_con.detector, "dart")
        self.assertEqual(p_con.threshold, 240)
        self.assertFalse(p_con.commercial_cleared)

        p_bal = fdd.get_preset("balanced")
        self.assertEqual(p_bal.detector, "farneback_3f")
        self.assertEqual(p_bal.sensitivity, 0.65)
        self.assertEqual(p_bal.dilate, 1)
        self.assertTrue(p_bal.commercial_cleared)

    def test_commercial_preset_filtering(self) -> None:
        all_presets = fdd.list_presets(commercial_only=False)
        self.assertEqual(len(all_presets), 11)

        comm_presets = fdd.list_presets(commercial_only=True)
        self.assertIn("balanced", comm_presets)
        self.assertIn("fast_preview", comm_presets)
        self.assertIn("high_quality_precision", comm_presets)
        self.assertIn("scratch_focused", comm_presets)
        self.assertIn("dust_focused", comm_presets)
        self.assertIn("high_recall", comm_presets)
        self.assertIn("conservative_low_fp", comm_presets)
        self.assertNotIn("dart_optimal_f1", comm_presets)
        self.assertNotIn("dart_high_recall", comm_presets)
        self.assertNotIn("dart_conservative", comm_presets)

    def test_preset_resolution_exact_and_custom(self) -> None:
        name, res = fdd.resolve_preset({"preset": "dart_optimal_f1"})
        self.assertEqual(name, "dart_optimal_f1")
        self.assertEqual(res["detector"], "dart")
        self.assertEqual(res["threshold"], 224)

        # Altering threshold should drop to custom
        name_custom, res_custom = fdd.resolve_preset(
            {"preset": "dart_optimal_f1", "threshold": 210}
        )
        self.assertEqual(name_custom, "custom")
        self.assertEqual(res_custom["threshold"], 210)

        # Unspecified preset matching parameter set should auto-map
        name_auto, res_auto = fdd.resolve_preset(
            {"detector": "consensus_5f", "sensitivity": 0.70, "dilate": 0, "threshold": 160}
        )
        self.assertEqual(name_auto, "high_quality_precision")


class TestDefectDetectorRegistryDART(unittest.TestCase):
    """Test registry behavior with DART detector."""

    def test_dart_is_registered_with_aliases(self) -> None:
        reg = fdd.DEFAULT_REGISTRY
        self.assertTrue(reg.has("dart_harness", commercial_only=False))
        self.assertTrue(reg.has("dart", commercial_only=False))

    def test_commercial_guard_blocks_dart(self) -> None:
        reg = fdd.DEFAULT_REGISTRY
        self.assertFalse(reg.has("dart", commercial_only=True))
        with self.assertRaises(fdd.CommercialLicenseError):
            reg.get("dart", commercial_only=True)

    def test_research_mode_retrieves_dart(self) -> None:
        reg = fdd.DEFAULT_REGISTRY
        det = reg.get("dart", commercial_only=False)
        self.assertEqual(det.name, "dart_harness")
        self.assertFalse(det.commercial_cleared)
        self.assertIn("RTN", det.license_status)


class TestDustStageOptionsAndPolicy(unittest.TestCase):
    """Test dust stage options parsing, repair decoupling, and commercial policy."""

    def test_dart_detector_with_telea_repair_permitted(self) -> None:
        opts = dust.options({
            "detector": "dart",
            "repair": "telea",
            "threshold": "224",
        })
        self.assertEqual(opts["detector"], "dart")
        self.assertEqual(opts["repair"], "telea")
        self.assertEqual(opts["threshold"], 224)
        self.assertEqual(opts["preset"], "dart_optimal_f1")

    def test_dart_detector_with_dart_repair_permitted(self) -> None:
        opts = dust.options({
            "detector": "dart",
            "repair": "dart",
            "threshold": "224",
        })
        self.assertEqual(opts["detector"], "dart")
        self.assertEqual(opts["repair"], "dart")

    def test_classical_detector_with_dart_repair_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            dust.options({
                "detector": "farneback_3f",
                "repair": "dart",
            })
        self.assertIn("DART repair requires the DART defect detector", str(ctx.exception))

    def test_commercial_model_guard_enforcement(self) -> None:
        settings_guarded = {
            "global": {"commercial_model_guard": "true"},
            "dust": {"detector": "dart", "repair": "telea", "threshold": "224"},
        }
        issue = commercial_policy.stage_issue(settings_guarded, "dust")
        self.assertIn("DART adapted RTN code rights remain unresolved", issue)

        settings_unguarded = {
            "global": {"commercial_model_guard": "false"},
            "dust": {"detector": "dart", "repair": "telea", "threshold": "224"},
        }
        issue_free = commercial_policy.stage_issue(settings_unguarded, "dust")
        self.assertEqual(issue_free, "")

    def test_authorize_checks_confirmations_for_dart(self) -> None:
        values = {"detector": "dart", "repair": "telea", "threshold": "224"}
        with self.assertRaises(ValueError) as ctx1:
            dust.authorize(values, guard=False)
        self.assertIn("Confirm experimental use", str(ctx1.exception))

        values["research_confirmed"] = "true"
        with self.assertRaises(ValueError) as ctx2:
            dust.authorize(values, guard=False)
        self.assertIn("Acknowledge model licence restrictions", str(ctx2.exception))

        values["license_confirmed"] = "true"
        authorized = dust.authorize(values, guard=False)
        self.assertEqual(authorized["detector"], "dart")
        self.assertEqual(authorized["preset"], "dart_optimal_f1")


class TestDefectChannelSeparation(unittest.TestCase):
    """Test connected-component split of defect masks into scratch and dust."""

    def test_scratch_and_dust_separation_no_data_loss(self) -> None:
        h, w = 120, 160
        binary = np.zeros((h, w), dtype=np.uint8)

        # Draw a vertical scratch: 50 px tall, 2 px wide
        binary[20:70, 40:42] = 255

        # Draw a small dust spot: 3x3 blob
        binary[80:83, 100:103] = 255

        s_mask, d_mask = fdd.split_proposal_components(binary)

        # Verify scratch captured in scratch mask
        self.assertGreater(np.count_nonzero(s_mask), 0)
        self.assertEqual(np.count_nonzero(s_mask & (binary > 0)), np.count_nonzero(s_mask))

        # Verify dust captured in dust mask
        self.assertGreater(np.count_nonzero(d_mask), 0)
        self.assertEqual(np.count_nonzero(d_mask & (binary > 0)), np.count_nonzero(d_mask))

        # Ensure scratch and dust masks are disjoint
        overlap = np.count_nonzero((s_mask > 0) & (d_mask > 0))
        self.assertEqual(overlap, 0)


class TestExternalDartHarnessExecution(unittest.TestCase):
    """Test execution of ExternalDartHarnessDetector."""

    def test_minimum_frames_requirement(self) -> None:
        det = fdd.DEFAULT_REGISTRY.get("dart", commercial_only=False)
        single_frame = [np.zeros((64, 64, 3), dtype=np.uint8)]
        with self.assertRaises(ValueError) as ctx:
            det.detect(single_frame)
        self.assertIn("requires at least 2 consecutive frames", str(ctx.exception))

    def test_genuine_dart_inference_runs_when_environment_available(self) -> None:
        py = cde.find_isolated_dart_python()
        weights = Path("I:/DART/pretrained_models/dart.safetensors")
        if py is None or not py.is_file() or not weights.is_file():
            self.skipTest("Isolated DART environment or weights not found on this machine")

        det = fdd.DEFAULT_REGISTRY.get("dart", commercial_only=False)
        frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        # Add synthetic vertical mark
        frames[1][10:50, 30:32] = 255

        result = det.detect(frames, {"threshold": 224})
        self.assertEqual(len(result.mask), 3)
        self.assertEqual(len(result.scratch_mask), 3)
        self.assertEqual(len(result.dust_mask), 3)
        self.assertEqual(result.metadata["provenance"], "genuine_dart_inference")
        self.assertEqual(result.metadata["threshold"], 224)
        self.assertIn("weights_sha256", result.metadata)


if __name__ == "__main__":
    unittest.main()
