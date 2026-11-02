"""Quantitative digital-reference pilot and negative controls.

These tests use generated images, not archival ground-truth footage.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_local_deflicker_pilot import evaluate, synthetic_exposure_damage


def clean_frames(n=21):
    x = np.linspace(75, 170, 96, dtype=np.float32)
    y = np.linspace(0, 12, 64, dtype=np.float32)
    smooth = np.clip(np.rint(y[:, None] + x[None, :]), 0, 255).astype(np.uint8)
    frame = cv2.cvtColor(smooth, cv2.COLOR_GRAY2BGR)
    return [frame.copy() for _ in range(n)]


class LocalDeflickerPilotTests(unittest.TestCase):
    def test_synthetic_damage_affects_left_half_and_not_right(self):
        frames = clean_frames()
        damaged, mask = synthetic_exposure_damage(frames)
        self.assertEqual(len(damaged), len(frames))
        self.assertGreater(np.mean(np.abs(
            frames[0][:, :20].astype(np.int16) - damaged[0][:, :20].astype(np.int16))), 8)
        self.assertTrue(np.array_equal(frames[0][:, 75:], damaged[0][:, 75:]))
        self.assertEqual(mask[0], 1)
        self.assertEqual(mask[-1], 0)

    def test_known_digital_reference_and_untouched_negative_control(self):
        data = evaluate(clean_frames(), sigma=18.0, radius=2, strength=0.6, cap=0.08)
        self.assertEqual(data["frames"], 21)
        self.assertGreater(data["damaged_inside_mae"], 10)
        self.assertLess(data["corrected_inside_mae"], data["damaged_inside_mae"])
        self.assertGreater(data["inside_mae_improvement"], 0)
        self.assertEqual(data["damaged_outside_mae"], 0)
        self.assertEqual(data["untouched_original_modified_mae"], 0)
        self.assertEqual(data["untouched_outside_modified_mae"], 0)
        self.assertFalse(data["release_ready"])
        self.assertIn("NOT original film ground truth", data["reference"])

    def test_moving_scene_is_measured_not_assumed_to_improve(self):
        frames = clean_frames()
        for frame_id, frame in enumerate(frames):
            x = (frame_id * 3) % (frame.shape[1] - 10)
            frame[10:35, x:x+10, :] = 210
        result = evaluate(frames, sigma=12, strength=0.6, cap=0.08)
        self.assertEqual(result["frames"], len(frames))
        self.assertGreaterEqual(result["untouched_original_modified_mae"], 0)
        self.assertGreaterEqual(result["corrected_outside_mae"], 0)
        # Moving-object artifacts require review: do NOT assert 'success'.
        self.assertFalse(result["release_ready"])

    def test_motion_guard_comparison_is_measured_not_assumed(self):
        frames = clean_frames()
        for i, frame in enumerate(frames):
            x = 10 + i * 3
            frame[10:35, x:x+9] = 225
        baseline = evaluate(frames, sigma=9, temporal_guard="off")
        guarded = evaluate(frames, sigma=9, temporal_guard="oscillation")
        self.assertEqual(baseline["frames"], guarded["frames"])
        self.assertEqual(guarded["temporal_guard"], "oscillation")
        self.assertLessEqual(
            guarded["untouched_original_modified_mae"],
            baseline["untouched_original_modified_mae"]
        )
        self.assertFalse(guarded["release_ready"])
        self.assertIn("corrected_inside_mae", guarded)

    def test_invalid_injection_amplitudes_fail_closed(self):
        with self.assertRaises(ValueError):
            synthetic_exposure_damage(clean_frames(), amplitude=0.9)
        with self.assertRaises(ValueError):
            synthetic_exposure_damage(clean_frames(), amplitude=0)


if __name__ == "__main__":
    unittest.main()
