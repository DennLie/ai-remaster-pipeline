"""Guardrails for experimental localized deflicker. Not a production quality claim."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from local_deflicker_research import local_deflicker_frames  # noqa: E402


def base_frame() -> np.ndarray:
    y, x = np.mgrid[0:64, 0:96]
    luma = (90 + x / 3 + 8 * np.sin(y / 4)).astype(np.float32)
    luma[15:34, 10:21] += 55
    return cv2.cvtColor(np.clip(luma, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


class LocalizedDeflickerTests(unittest.TestCase):
    def test_static_scene_is_unchanged(self):
        original = base_frame()
        fixed, data = local_deflicker_frames([original.copy() for _ in range(11)])
        self.assertEqual(data["corrected_frames"], 0)
        self.assertTrue(all(np.array_equal(a, original) for a in fixed))

    def test_localized_flicker_improves_affected_region_but_spares_other_half(self):
        base = base_frame()
        source = []
        for i in range(21):
            f = base.copy().astype(np.float32)
            f[:, :40] *= 0.7 if i % 2 else 1.25
            source.append(np.uint8(np.clip(np.rint(f), 0, 255)))
        fixed, data = local_deflicker_frames(
            source, radius=2, sigma=6, strength=0.8, cap=0.12
        )
        inp = np.array([f[20:44, 7:27].mean() for f in source])
        out = np.array([f[20:44, 7:27].mean() for f in fixed])
        self.assertLess(np.std(out), np.std(inp) * 0.91)
        # The rightmost region receives no temporal correction.
        self.assertLess(
            np.mean([np.abs(f[:, 74:].astype(np.int16)-g[:, 74:].astype(np.int16)).mean()
                     for f, g in zip(source, fixed)]), 0.6
        )
        self.assertTrue(0 < data["corrected_frames"] <= len(source))
        self.assertLessEqual(data["max_abs_gain_pct"], 12.0001)

    def test_monotonic_fade_is_protected(self):
        base = base_frame().astype(np.float32)
        originals = [
            np.uint8(np.clip(np.rint(base * (0.60 + 0.035*i)), 0, 255))
            for i in range(16)
        ]
        fixed, data = local_deflicker_frames(originals)
        self.assertTrue(all(np.array_equal(a, b) for a,b in zip(originals, fixed)))
        self.assertEqual(data["protected_monotonic_fades"], [[0,16]])

    def test_human_cut_resets_temporal_neighbors(self):
        black = np.full((64,96,3), 65, np.uint8)
        white = np.full((64,96,3), 185, np.uint8)
        frames = [black.copy() for _ in range(10)] + [white.copy() for _ in range(10)]
        fixed, data = local_deflicker_frames(frames, cut_frames=(10,))
        self.assertEqual(data["corrected_frames"], 0)
        self.assertTrue(all(np.array_equal(a,b) for a,b in zip(frames, fixed)))

    def test_fail_closed_invalid_inputs(self):
        good = base_frame()
        for kwargs in ({"cap": 0.9}, {"sigma": 0}, {"radius": 0},
                       {"strength": -1}, {"cut_frames": (0,)},
                       {"cut_frames": (12,)}, {"cut_frames": (4,4)}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    local_deflicker_frames([good]*9, **kwargs)
        with self.assertRaises(ValueError):
            local_deflicker_frames([])
        with self.assertRaises(ValueError):
            local_deflicker_frames([good, good[:20]])


if __name__ == "__main__":
    unittest.main()
