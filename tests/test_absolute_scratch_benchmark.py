import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_absolute_scratches as benchmark


class LabelsTests(unittest.TestCase):
    def test_weak_opacity_is_ignored_not_healthy(self):
        soft = np.array([[0, .05, .1, 1]], np.float32)
        np.testing.assert_array_equal(benchmark.labels(soft), [[0, 127, 255, 255]])

    def test_invalid_ground_truth_rejected(self):
        for value in (float('nan'), -1, 2):
            with self.assertRaises(ValueError):
                benchmark.labels(np.array([[value]]))
