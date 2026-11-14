import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_ridge_pr24 as replay
import prepare_dust_annotation as annotation


class ReplayTests(unittest.TestCase):
    def test_short_annotation_source_creates_no_labels(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'invalid.mkv'
            source.write_bytes(b'not video')
            with self.assertRaises(ValueError):
                annotation.prepare(source, root / 'sample')
            self.assertFalse((root / 'sample').exists())

    def test_annotation_export_has_no_automatic_ground_truth(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / 'source'
            source.write_bytes(b'fixed source')
            with patch.object(annotation.cv2, 'VideoCapture') as capture:
                capture.return_value.read.return_value = (True, np.zeros((128, 256, 3), np.uint8))
                report = annotation.prepare(source, root / 'sample')
            self.assertEqual(len(report['samples']), 6)
            self.assertTrue(all(row['annotation_status'] == 'pending' for row in report['samples']))
            self.assertEqual(len(list((root / 'sample').glob('*.png'))), 6)
            with self.assertRaises(FileExistsError):
                annotation.prepare(source, root / 'sample')

    def test_missing_data_creates_no_output(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / 'output'
            with self.assertRaises(FileNotFoundError):
                replay.run(Path(td), output)
            self.assertFalse(output.exists())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            marker = Path(td) / 'keep'
            marker.write_text('original')
            with self.assertRaises(FileExistsError):
                replay.run(Path(td), Path(td))
            self.assertEqual(marker.read_text(), 'original')

    def test_pinned_metrics_count_false_positives_and_misses(self):
        score = replay.functions('scripts/diagnose_scratch_masks.py', ('confusion', 'summarize_confusion'))
        counts = score['confusion'](np.array([[255, 255, 0]], np.uint8),
                                    np.array([[255, 0, 255]], np.uint8))
        self.assertEqual(counts, {'tp': 1, 'fp': 1, 'fn': 1, 'tn': 0})
        self.assertEqual(score['summarize_confusion']([counts])['pixel_f1'], .5)

    def test_historical_legacy_has_its_column_helper(self):
        detector = replay.functions('scripts/cleanup_video.py',
                                    ('_contiguous_true_runs', 'ai_scratch_mask'))['ai_scratch_mask']
        frame = np.full((128, 256, 3), 100, np.uint8)
        frame[:, 40:43] = 240
        mask = detector(frame, .65)
        self.assertEqual(mask.shape, frame.shape[:2])
        self.assertEqual(mask.dtype, np.uint8)
