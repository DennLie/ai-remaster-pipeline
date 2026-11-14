"""Stabilization comparisons are read-only until explicit, validated adoption."""
import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import stabilization as stage, server, denoise_review, frame_inspector
from ai_remaster_gui.runtime_settings import base_settings


class StabilizationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='arp-stabilization-review-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.mkv'
        self.source.write_bytes(b'original')
        self.settings = base_settings()
        self.settings['global'].update(source=str(self.source), denoise='false', deflicker='false',
            enhancement='false', cleanup='false', stabilize='true', detail='false')
        self.settings['cloud']['intermediate_format'] = 'lossless'
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for module in (stage, frame_inspector):
            self.stack.enter_context(patch.object(module, 'ROOT', self.root))
        self.stack.enter_context(patch.object(frame_inspector, 'CACHE', self.root / '.cache/stills'))
        self.stack.enter_context(patch.object(stage.backend, 'DEFAULT_OUTPUT_ROOT', self.root / 'intermediate' / 'stabilized'))
        self.stack.enter_context(patch.object(server.APP, 'settings', self.settings))
        self.stack.enter_context(patch.object(server.APP, 'process', None))
        self.stack.enter_context(patch.object(server.APP, 'queue_running', False, create=True))
        self.app = server.APP

    def candidate(self, method='zoom0'):
        args = stage.arguments(self.source, stage.variant_values(self.settings['stabilize'], method), 'lossless')
        output = stage.backend.default_output(self.source, args)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'completed')
        stage.backend.write_signature(output, stage.backend.output_signature(self.source, args, [(0, 12)]))
        return output

    def test_defaults_custom_zoom_and_variant_identity(self):
        self.assertEqual(float(self.settings['stabilize']['zoom']), 3)
        self.settings['stabilize']['zoom'] = '1.5'
        values = dict(self.settings['stabilize'])
        outputs = [stage.output_for(str(self.source), stage.variant_values(values, method), 'lossless') for method in stage.METHODS]
        self.assertEqual(len(set(outputs)), 3)
        self.assertEqual(self.settings['stabilize'], values)
        self.assertEqual(stage.arguments(self.source, values).zoom, 1.5)
        self.assertEqual(stage.arguments(self.source, {**values, 'max_shift': '0'}).max_shift, 0)
        for invalid in ('nan', 'inf', '-1', '21'):
            with self.assertRaises(ValueError):
                stage.arguments(self.source, {**values, 'zoom': invalid})
        for rows in ([], [{'method': 'evil', 'preset': 'settings'}], [{'method': 'zoom0', 'preset': 'settings'}] * 2):
            with self.assertRaises(ValueError):
                stage.validate_variants(rows)
        with patch.object(self.app, 'save') as save:
            for values in ({'zoom': 'nan'}, {'max_shift': '-2'}, {'scene_aware': 'unknown'}):
                with self.assertRaises(ValueError):
                    self.app.update_settings('stabilize', values)
            save.assert_not_called()
        self.assertEqual(self.settings['stabilize']['zoom'], '1.5')

    def test_read_only_pair_matrix_and_legacy_missing_report(self):
        output = self.candidate()
        before = json.dumps(self.settings, sort_keys=True)
        self.assertEqual(denoise_review.review_pair(self.app, str(output), 'stabilize'), (self.source, output))
        self.assertIsNone(stage.scene_report(output))
        matrix = self.app.stabilization_matrix_state()
        self.assertEqual(matrix['source'], str(self.source))
        self.assertEqual(matrix['variants'][0]['method'], 'zoom0')
        self.assertIsNone(matrix['variants'][0]['scenes'])
        self.assertEqual(json.dumps(self.settings, sort_keys=True), before)

    def test_rejects_stale_source_settings_format_partial_and_arbitrary_paths(self):
        output = self.candidate()
        for candidate in (str(self.source), str(output).replace('.mkv', '.partial.mkv')):
            with self.assertRaises((ValueError, FileNotFoundError)):
                stage.review_pair(self.app, candidate)
        for key, value in (('smoothing', '13'), ('scene_aware', 'false')):
            with patch.dict(self.settings['stabilize'], {key: value}):
                with self.assertRaises(ValueError):
                    stage.review_pair(self.app, str(output))
        with patch.dict(self.settings['cloud'], intermediate_format='high'):
            with self.assertRaises(ValueError):
                stage.review_pair(self.app, str(output))
        self.source.write_bytes(b'changed source')
        with self.assertRaises(ValueError):
            stage.review_pair(self.app, str(output))

    def test_invalid_or_missing_provenance_never_becomes_ready(self):
        output = self.candidate()
        sidecar = output.with_suffix('.mkv.sig.json')
        for text in ('[]', '{', '{}'):
            sidecar.write_text(text)
            self.assertEqual(self.app.stabilization_matrix_state()['variants'], [])
        sidecar.unlink()
        self.assertEqual(self.app.stabilization_matrix_state()['variants'], [])

    def test_adoption_revalidates_candidate_and_geometry_before_saving_only_zoom(self):
        output = self.candidate()
        with patch.object(self.app, 'save') as save, patch.object(self.app, 'hydrate_stage_inputs'), \
             patch('ai_remaster_gui.frame_differences._validate_pair', return_value=12) as validate:
            with self.assertRaises(ValueError):
                self.app.adopt_stabilization('zoom0', str(self.source))
            save.assert_not_called()
            before = dict(self.settings['stabilize'])
            self.app.adopt_stabilization('zoom0', str(output))
            validate.assert_called_once_with(self.source, output)
            self.assertEqual(self.settings['stabilize'], {**before, 'zoom': '0', 'force': 'false'})
            save.assert_called_once()
        with patch.object(self.app, 'queue_running', True):
            with self.assertRaisesRegex(ValueError, 'Stop processing'):
                self.app.adopt_stabilization('zoom0', str(output))

    def test_failed_geometry_does_not_save_or_adopt(self):
        output = self.candidate()
        with patch.object(self.app, 'save') as save, patch('ai_remaster_gui.frame_differences._validate_pair', side_effect=ValueError('frames differ')):
            with self.assertRaisesRegex(ValueError, 'frames differ'):
                self.app.adopt_stabilization('zoom0', str(output))
            self.assertEqual(float(self.settings['stabilize']['zoom']), 3)
            save.assert_not_called()

    def test_cleanup_input_is_never_replaced_by_raw_source(self):
        with patch.dict(self.settings['global'], cleanup='true'):
            info = self.app.processing_input_state('stabilize')
            self.assertNotEqual(info['source'], str(self.source))
            self.assertIn('Clean Up', info['problem'])
            self.assertFalse(info['original_fallback'])

    def test_scene_report_is_bound_to_the_completed_signature_and_output(self):
        output = self.candidate()
        signature = json.loads(output.with_suffix('.mkv.sig.json').read_text())
        scenes = [{'start': 0, 'end': 12, 'status': 'fallback', 'reason': 'unreliable fit', 'tracking_report_available': True}]
        report = {'signature': signature, 'output_fingerprint': stage.backend.file_fingerprint(output), 'scenes': scenes}
        sidecar = output.with_suffix('.mkv.scenes.json')
        sidecar.write_text(json.dumps(report))
        self.assertEqual(stage.scene_report(output), scenes)
        report['scenes'] = 'broken'
        sidecar.write_text(json.dumps(report))
        self.assertIsNone(stage.scene_report(output))
        report['scenes'] = scenes
        sidecar.write_text(json.dumps(report))
        output.write_bytes(b'replaced output')
        self.assertIsNone(stage.scene_report(output))

    def test_batch_preserves_settings_and_uses_only_requested_zoom_variants(self):
        rows = [{'method': 'zoom0', 'preset': 'settings'}, {'method': 'zoom3', 'preset': 'settings'}]
        before = json.dumps(self.settings, sort_keys=True)
        with patch.object(self.app, 'ensure_pipeline_source'), patch.object(self.app, '_start_denoise_matrix', return_value=(True, 'started')) as start:
            ok, _ = self.app.run_stabilization_batch(rows)
            self.assertTrue(ok)
            command = start.call_args.args[1]
            self.assertEqual(json.loads(command[command.index('--variants') + 1]), rows)
            self.assertEqual(start.call_args.args[2], 'stabilize')
        self.assertEqual(json.dumps(self.settings, sort_keys=True), before)
        with patch.dict(self.settings['stabilize'], compute='runpod'):
            ok, message = self.app.run_stabilization_batch(rows)
            self.assertFalse(ok)
            self.assertIn('RunPod', message)

    def test_real_zero_and_three_percent_outputs_share_clock_and_have_scene_reports(self):
        ffmpeg = stage.backend.find_ffmpeg()
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
            'testsrc2=size=96x64:rate=12:duration=1', '-c:v', 'ffv1', str(self.source)], check=True)
        before = self.source.read_bytes()
        for method in ('zoom0', 'zoom3'):
            args = stage.arguments(self.source, stage.variant_values(self.settings['stabilize'], method), 'lossless')
            args.single_shot = True
            with patch.dict(self.settings['stabilize'], scene_aware='false'):
                stage.backend.main_with_args(args)
                output = stage.backend.default_output(self.source, args)
                pair = denoise_review.comparison_pair(self.app, str(output), 'stabilize')
                self.assertEqual(pair[3], 12)
                report = stage.scene_report(output)
                self.assertEqual([(row['start'], row['end']) for row in report], [(0, 12)])
                self.assertIn(report[0]['status'], ('stabilized', 'fallback'))
                coverage = stage.footprint(self.app, str(output), 0)
                self.assertTrue(0 <= coverage['valid_output_pct'] <= 100)
                self.assertAlmostEqual(coverage['valid_output_pct'] + coverage['missing_output_pct'], 100)
                self.assertTrue((self.root / coverage['image']).is_file())
                with self.assertRaises(ValueError):
                    stage.footprint(self.app, str(output), 12)
                old_time = output.stat().st_mtime_ns
                stage.backend.main_with_args(args)
                self.assertEqual(output.stat().st_mtime_ns, old_time)
                geometry_path = output.with_suffix('.mkv.geometry.json')
                data = json.loads(geometry_path.read_text())
                original_geometry = geometry_path.read_bytes()
                data['scenes'][0]['mask'] = '../escape.mkv'
                geometry_path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'mask path'):
                    stage.footprint(self.app, str(output), 0)
                geometry_path.write_bytes(original_geometry)
                mask = output.with_suffix('.geometry') / json.loads(original_geometry)['scenes'][0]['mask']
                mask.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'mask has changed'):
                    stage.footprint(self.app, str(output), 0)
        self.assertEqual(self.source.read_bytes(), before)

    def test_black_film_and_fractional_fps_do_not_look_like_missing_source_area(self):
        ffmpeg = stage.backend.find_ffmpeg()
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
            'color=c=black:s=100x80:r=30000/1001', '-frames:v', '3', '-c:v', 'ffv1', str(self.source)], check=True)
        with patch.dict(self.settings['stabilize'], scene_aware='false'):
            args = stage.arguments(self.source, self.settings['stabilize'], 'lossless')
            stage.backend.main_with_args(args)
            output = stage.backend.default_output(self.source, args)
            coverage = stage.footprint(self.app, str(output), 2)
            self.assertEqual(coverage['valid_output_pct'], 100)
            self.assertEqual(coverage['missing_output_pct'], 0)
            self.assertAlmostEqual(coverage['cropped_source_area_estimate_pct'], 5.91)
