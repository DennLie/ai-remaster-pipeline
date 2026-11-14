import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ai_remaster_gui import denoise_review as review


class ComparisonReferenceTests(unittest.TestCase):
    def setUp(self):
        self.app = SimpleNamespace(settings={'global': {'source': 'original.mkv'}})
        self.meta = {'width': 320, 'height': 240, 'fps': 25, 'fps_fraction': '25/1', 'frames': 100}

    def test_original_reference_keeps_candidate_lineage_validation(self):
        with patch.object(review, 'review_pair', side_effect=ValueError('stale provenance')):
            with self.assertRaisesRegex(ValueError, 'stale provenance'):
                review.comparison_pair(self.app, 'result.mkv', 'enhancement', 'original')

    def test_original_reference_keeps_timeline_and_validates_geometry(self):
        with patch.object(review, 'review_pair', return_value=(Path('input.mkv'), Path('result.mkv'))), \
             patch.object(review, '_metadata', return_value=self.meta), \
             patch.object(review, '_validate_pair', return_value=100), \
             patch.object(review, '_video_path', side_effect=Path), \
             patch('ai_remaster_gui.media.source_section_state', return_value={'enabled': False}):
            source, output, metadata, count, offset = review.comparison_pair(self.app, 'result.mkv', 'enhancement', 'original')
            self.assertEqual((source, output, count, offset), (Path('original.mkv'), Path('result.mkv'), 100, 0))
            self.assertEqual(metadata, self.meta)

    def test_invalid_reference_is_rejected(self):
        with self.assertRaises(ValueError):
            review.comparison_pair(self.app, '', 'enhancement', '../arbitrary.mkv')

    def test_section_reference_offsets_only_original_not_candidate_timeline(self):
        original_meta = {**self.meta, 'frames': 200}
        with patch.object(review, 'review_pair', return_value=(Path('input.mkv'), Path('result.mkv'))), \
             patch.object(review, '_metadata', side_effect=[self.meta, original_meta]), \
             patch.object(review, '_validate_pair', return_value=100), \
             patch.object(review, '_video_path', side_effect=Path), \
             patch('ai_remaster_gui.media.source_section_state', return_value={'enabled': True, 'start': 2}):
            source, output, metadata, count, offset = review.comparison_pair(self.app, 'result.mkv', 'enhancement', 'original')
            self.assertEqual((count, offset), (100, 50))
            self.assertEqual(metadata['frames'], 100)

    def test_original_geometry_mismatch_does_not_silently_resize(self):
        with patch.object(review, 'review_pair', return_value=(Path('input.mkv'), Path('result.mkv'))), \
             patch.object(review, '_metadata', side_effect=[self.meta, {**self.meta, 'width': 640}]), \
             patch.object(review, '_validate_pair', return_value=100), \
             patch.object(review, '_video_path', side_effect=Path):
            with self.assertRaisesRegex(ValueError, 'geometry'):
                review.comparison_pair(self.app, 'result.mkv', 'enhancement', 'original')

    def test_non_frame_aligned_section_is_explicitly_rejected(self):
        with patch.object(review, 'review_pair', return_value=(Path('input.mkv'), Path('result.mkv'))), \
             patch.object(review, '_metadata', return_value=self.meta), \
             patch.object(review, '_validate_pair', return_value=100), \
             patch.object(review, '_video_path', side_effect=Path), \
             patch('ai_remaster_gui.media.source_section_state', return_value={'enabled': True, 'start': 0.01}):
            with self.assertRaisesRegex(ValueError, 'aligned'):
                review.comparison_pair(self.app, 'result.mkv', 'enhancement', 'original')
