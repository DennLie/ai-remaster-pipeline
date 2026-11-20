"""Tests for Dust & Scratches GUI Phase 8 enhancements:
Tiered model selector, readiness badges, dynamic commercial safe filtering,
multi-channel preview toggles (scratch, dust, uncertain), and accepted mask protection invariant.
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai_remaster_gui import dust
from ai_remaster_gui.dust_jobs import protect_accepted_mask, propagate_defect_job_parameters


class DustGuiPhase8Tests(unittest.TestCase):
    def test_tiered_detector_metadata_and_readiness(self):
        # Tier categorizations
        self.assertEqual(dust.detector_tier('legacy'), 'Classical Detectors')
        self.assertEqual(dust.detector_tier('ridge'), 'Classical Detectors')
        self.assertEqual(dust.detector_tier('dart'), 'Neural Baseline (DART)')
        self.assertEqual(dust.detector_tier('bopb'), 'Neural Candidates')
        self.assertEqual(dust.detector_tier('film_artefact_segmenter'), 'Neural Candidates')
        self.assertEqual(dust.detector_tier('cnn_blotches'), 'Neural Candidates')
        self.assertEqual(dust.detector_tier('patchcore'), 'Neural Candidates')
        self.assertEqual(dust.detector_tier('padim'), 'Neural Candidates')
        self.assertEqual(dust.detector_tier('correl_net'), 'Neural Candidates')

        # available_detectors contains all models with tier and readiness
        detectors = dust.available_detectors(commercial_only=False)
        det_map = {d['name']: d for d in detectors}
        self.assertIn('legacy', det_map)
        self.assertIn('dart_harness', det_map)
        self.assertIn('bopb', det_map)
        self.assertIn('correl_net', det_map)

        self.assertEqual(det_map['legacy']['tier'], 'Classical Detectors')
        self.assertEqual(det_map['dart_harness']['tier'], 'Neural Baseline (DART)')
        self.assertEqual(det_map['bopb']['tier'], 'Neural Candidates')
        self.assertEqual(det_map['correl_net']['tier'], 'Neural Candidates')

        self.assertEqual(det_map['correl_net']['readiness_state'], 'training_required')

        # detector_metadata provides complete details
        meta_bopb = dust.detector_metadata('bopb')
        self.assertEqual(meta_bopb['tier'], 'Neural Candidates')
        self.assertIn('supported_channels', meta_bopb)
        self.assertEqual(meta_bopb['supported_channels'], ['scratch'])
        self.assertFalse(meta_bopb['commercial_cleared'])

    def test_authorization_blocks_training_required_and_unconfirmed_models(self):
        # Correl-Net is in training_required state -> must be blocked with actionable message
        with self.assertRaisesRegex(ValueError, "training_required"):
            dust.authorize({'detector': 'correl_net'}, kind='mask')

        # BOPB requires research confirmation
        with self.assertRaisesRegex(ValueError, "Confirm experimental use"):
            dust.authorize({'detector': 'bopb', 'research_confirmed': 'false'}, kind='mask')

        # BOPB requires license confirmation
        with self.assertRaisesRegex(ValueError, "Acknowledge model licence restrictions"):
            dust.authorize({
                'detector': 'bopb',
                'research_confirmed': 'true',
                'license_confirmed': 'false',
            }, kind='mask')

        # BOPB blocked by commercial guard
        with self.assertRaisesRegex(ValueError, "Commercial model guard blocks"):
            dust.authorize({
                'detector': 'bopb',
                'research_confirmed': 'true',
                'license_confirmed': 'true',
            }, guard=True, kind='mask')

        # BOPB authorized when acknowledged and guard is off
        auth = dust.authorize({
            'detector': 'bopb',
            'research_confirmed': 'true',
            'license_confirmed': 'true',
        }, guard=False, kind='mask')
        self.assertEqual(auth['detector'], 'bopb')

        # Legacy detector always succeeds
        auth_legacy = dust.authorize({'detector': 'legacy'}, kind='mask')
        self.assertEqual(auth_legacy['detector'], 'legacy')

    def test_protect_accepted_mask_invariant(self):
        """Invariant: Previewing a new detector or adjusting threshold NEVER overwrites an already accepted mask."""
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / 'candidate_dir'
            folder.mkdir(parents=True)
            report_file = folder / 'report.json'
            report_content = b'{"mock_report": true}'
            report_file.write_bytes(report_content)
            sha = hashlib.sha256(report_content).hexdigest()

            # Settings where this report is currently the accepted mask
            settings = {
                'dust': {
                    'adopted_report': sha,
                    'detector': 'legacy',
                }
            }

            # Mutating/running in this folder must raise ValueError
            with self.assertRaisesRegex(ValueError, "cannot mutate an accepted mask"):
                protect_accepted_mask(folder, settings)

            # Different adopted report allows operation
            other_settings = {
                'dust': {
                    'adopted_report': 'different_sha256_hash',
                    'detector': 'legacy',
                }
            }
            # Should not raise
            protect_accepted_mask(folder, other_settings)

    def test_propagate_defect_job_parameters(self):
        raw = {
            'detector': 'film_artefact_segmenter',
            'preset': 'film_artefact_balanced',
            'threshold': 0.45,
            'tile_size': 512,
            'overlap': 64,
            'ignored_none': None,
        }
        propagated = propagate_defect_job_parameters(raw)
        self.assertEqual(propagated['detector'], 'film_artefact_segmenter')
        self.assertEqual(propagated['preset'], 'film_artefact_balanced')
        self.assertEqual(propagated['threshold'], 0.45)
        self.assertEqual(propagated['tile_size'], 512)
        self.assertEqual(propagated['overlap'], 64)
        self.assertNotIn('ignored_none', propagated)

    def test_multichannel_uncertain_mask_roundtrip(self):
        with tempfile.TemporaryDirectory() as td, patch.object(dust, 'ROOT', Path(td)):
            source = Path(td) / 'source.mkv'
            writer = dust.Writer(source, {'frames': 3, 'width': 96, 'height': 64, 'fps': '12'})
            for _ in range(3):
                writer.write(np.full((64, 96, 3), 100, np.uint8))
            writer.close()

            values = {'detector': 'legacy'}
            dust.run(source, values, 'mask')

            target_folder = dust.folder_for(source, values)
            report = dust.manifest(target_folder, source, values, 'mask')

            # Verify multi-channel masks in manifest hashes and measurements
            self.assertIn('effective_mask.mkv', report['hashes'])
            self.assertIn('scratch_mask.mkv', report['hashes'])
            self.assertIn('dust_mask.mkv', report['hashes'])
            self.assertIn('uncertain_mask.mkv', report['hashes'])

            # Verify status exposes multi-channel files
            app = SimpleNamespace(
                settings={'global': {'source': str(source)}},
                project_path='p',
                stabilized_source_for_downstream=lambda: str(source),
                running_stage_key='',
                process=None,
            )
            st = dust.status(app, values)
            self.assertIn('scratch_mask.mkv', st['files'])
            self.assertIn('dust_mask.mkv', st['files'])
            self.assertIn('uncertain_mask.mkv', st['files'])

            # Verify inspect decodes multi-channel mask frames and counts
            inspected = dust.inspect(app, values, 0)
            self.assertIn('effective_mask.mkv', inspected['images'])
            self.assertIn('scratch_mask.mkv', inspected['images'])
            self.assertIn('dust_mask.mkv', inspected['images'])
            self.assertIn('uncertain_mask.mkv', inspected['images'])
            self.assertIn('uncertain_pixels', inspected['measurements'])


if __name__ == '__main__':
    unittest.main()
