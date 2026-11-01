"""Unit checks for assumed-PSF luminance Richardson–Lucy restoration."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

import cv2
import numpy as np
from skimage.restoration import richardson_lucy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import richardson_lucy_video as rl


class RichardsonLucyTests(unittest.TestCase):
    def test_psf_is_normalized_finite_and_symmetric(self):
        for preset, (sigma, steps) in rl.PSF_PRESETS.items():
            psf = rl.point_spread_function(sigma)
            self.assertEqual(psf.shape[0], psf.shape[1])
            self.assertGreater(steps, 0)
            self.assertAlmostEqual(float(psf.sum()), 1., places=10)
            self.assertTrue(np.all(psf >= 0))
            np.testing.assert_allclose(psf, psf[::-1, ::-1], rtol=1e-9)

    def test_mix_and_shape_and_color(self):
        rng = np.random.default_rng(1234)
        frame = rng.integers(20, 230, size=(64, 96, 3), dtype=np.uint8)
        kernel = rl.point_spread_function(0.75)
        out = rl.deblur_rgb(frame, kernel, 3, 80)
        self.assertEqual(out.shape, frame.shape)
        self.assertEqual(out.dtype, frame.dtype)
        self.assertGreater(float(np.mean(np.abs(out.astype(float)-frame.astype(float)))), 0)
        with self.assertRaises(ValueError):
            rl.deblur_rgb(frame, kernel, 0, 100)
        with self.assertRaises(ValueError):
            rl.deblur_rgb(frame, kernel, 3, 101)

    def test_assumed_kernel_changes_a_synthetic_blurred_edge(self):
        clean = np.zeros((80, 120), np.float64)
        clean[25:55, 25:95] = 0.9
        psf = rl.point_spread_function(1.0)
        blurred = cv2.filter2D(clean, -1, psf)
        deconv = richardson_lucy(blurred, psf, num_iter=5, clip=True)
        self.assertEqual(deconv.shape, clean.shape)
        self.assertTrue(np.all(np.isfinite(deconv)))
        self.assertLess(float(np.mean((deconv-clean)**2)),
                        float(np.mean((blurred-clean)**2)),
                        "A correctly specified synthetic PSF should help this calibration case")

    def test_refuses_invalid_presets_and_strength(self):
        with self.assertRaises(ValueError):
            rl.validate("unknown", 50)
        with self.assertRaises(ValueError):
            rl.validate("gentle", 101)


if __name__ == "__main__":
    unittest.main()
