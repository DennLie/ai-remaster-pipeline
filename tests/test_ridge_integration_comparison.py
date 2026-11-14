import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import compare_ridge_integration as comparison


class RidgeIntegrationComparisonTests(unittest.TestCase):
    def test_pinned_reference_matches_core_but_wrapper_abstains_at_endpoints(self):
        reference,_=comparison.reference_detector()
        frames=[]
        for index in range(24):
            frame=np.full((128,256,3),100,np.uint8)
            frame[:,20+index*7:23+index*7]=230
            frames.append(frame)
        before=[frame.copy() for frame in frames]
        masks,rows=comparison.compare_frames(frames,reference)
        self.assertTrue(all(row['different_pixels_core']==0 for row in rows))
        self.assertTrue(all(np.any(mask) for mask in masks['pr24']))
        self.assertEqual([row['frame'] for row in rows if row['different_pixels_wrapper']],[0,23])
        self.assertIn('insufficient context',rows[0]['wrapper_reason'])
        for old,new in zip(before,frames):np.testing.assert_array_equal(old,new)

    def test_static_vertical_structure_remains_empty_in_both_methods(self):
        reference,_=comparison.reference_detector()
        frame=np.full((128,256,3),100,np.uint8);frame[:,30:33]=230
        predictions,rows=comparison.compare_frames([frame.copy() for _ in range(24)],reference)
        self.assertTrue(all(not np.any(mask) for masks in predictions.values() for mask in masks))

    def test_actual_video_report_preserves_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);source=root/'source.mkv'
            writer=comparison.dust.Writer(source,{'frames':24,'width':256,'height':128,'fps':'12'})
            for _ in range(24):writer.write(np.full((128,256,3),100,np.uint8))
            writer.close();before=comparison.sha(source)
            report=comparison.compare(source,root/'results')
            self.assertEqual(comparison.sha(source),before)
            self.assertTrue(report['core_equal']);self.assertTrue(report['wrapper_equal'])
            self.assertEqual(len(report['both_empty_frames']),24)
            self.assertEqual(len(list((root/'results/pr24').glob('*.png'))),24)
            with patch.object(comparison,'reference_detector',side_effect=AssertionError('must refuse early')):
                with self.assertRaises(FileExistsError):comparison.compare(source,root/'results')


if __name__=='__main__':unittest.main()
