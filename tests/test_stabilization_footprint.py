import tempfile
import unittest
from pathlib import Path

from ai_remaster_gui.stabilization import backend
from stabilization_footprint import replay


class FootprintTests(unittest.TestCase):
    def check_replay(self, x=0, zoom=0, fallback=False, angle=0, width=100, height=80):
        with tempfile.TemporaryDirectory(prefix='arp-footprint-') as folder:
            root = Path(folder)
            (root / 'shot_0000.trf').write_text(f'1 {x} 0 {angle} 0 0\n2 {x} 0 {angle} 0 0\n')
            (root / 'global_motions.trf').write_text('# controlled absolute transforms\n')
            transform = ("vidstabtransform=input='shot_0000.trf':smoothing=0:relative=0:"
                         f'optzoom=0:zoom={zoom}:crop=black:interpol=bicubic:debug=1')
            result = replay(backend.find_ffmpeg(), root, 0, 2, width, height, 24, transform, fallback)
            self.assertTrue((root / 'geometry' / result['analysis']).is_file())
            self.assertTrue((root / 'geometry' / result['global_motion']).is_file())
            return result['valid_pixels']

    def test_identity_retains_the_entire_canvas(self):
        self.assertEqual(self.check_replay(), [8000, 8000])

    def test_odd_dimensions_are_not_silently_rounded_by_the_mask_source(self):
        self.assertEqual(self.check_replay(width=97, height=81), [7857, 7857])

    def test_translation_has_missing_output_even_without_zoom(self):
        self.assertEqual(self.check_replay(x=10), [7200, 7200])

    def test_fixed_zoom_can_have_full_output_but_loses_source_area(self):
        self.assertEqual(self.check_replay(zoom=3), [8000, 8000])
        self.assertAlmostEqual((1 - .97 ** 2) * 100, 5.91)

    def test_safety_fallback_applies_neither_translation_nor_zoom(self):
        self.assertEqual(self.check_replay(x=10, zoom=3, fallback=True), [8000, 8000])

    def test_rotation_exposes_missing_corners(self):
        counts = self.check_replay(angle=.1)
        self.assertTrue(all(0 < count < 8000 for count in counts))
