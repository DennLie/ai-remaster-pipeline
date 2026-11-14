"""The new stage is separate from late, legacy Detail Enhancement."""
import sys
import importlib.util
import unittest
import json
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class EnhancementTests(unittest.TestCase):
    def test_identity_distinguishes_current_parameters_without_changing_legacy(self):
        import enhancement_video as backend
        source = Path('input/film.mkv').resolve()
        first = backend.output_for(source, {'method': 'cas', 'preset': 'gentle', 'mix': '50'})
        second = backend.output_for(source, {'method': 'cas', 'preset': 'gentle', 'mix': '100'})
        self.assertNotEqual(first, second)
        self.assertEqual(first.parent.name, 'enhanced')
        for values in ({'method': 'unknown'}, {'mix': '-1'}, {'mix': 50.5}, {'mix': True}, {'preset': 'unknown'}):
            with self.assertRaises(ValueError):
                backend.options(values)

    def test_stage_is_optional_separate_and_immediately_after_deflicker(self):
        from ai_remaster_gui.models import STAGES
        from ai_remaster_gui.runtime_settings import base_settings
        keys = [stage.key for stage in STAGES]
        self.assertEqual(keys[keys.index('deflicker') + 1], 'enhancement')
        self.assertIn('detail', keys)
        self.assertEqual(base_settings()['global']['enhancement'], 'false')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch runtime required')
    def test_model_frame_preserves_odd_geometry_and_mix_zero(self):
        import numpy as np
        import torch
        from deblur_models import restore_rgb
        original = np.full((71, 95, 3), 127, dtype=np.uint8)
        seen = []
        class Model(torch.nn.Module):
            def forward(self, value):
                seen.append(tuple(value.shape))
                return value + .1
        result = restore_rgb(Model(), original, method='nafnet', mix=100, device='cpu')
        self.assertEqual(seen, [(1, 3, 80, 96)])
        self.assertEqual(result.shape, original.shape)
        self.assertTrue((result > original).all())
        self.assertTrue(np.array_equal(restore_rgb(Model(), original, method='nafnet', mix=0, device='cpu'), original))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Optional PyTorch runtime required')
    def test_pathological_neural_output_is_not_silently_clipped_into_a_ready_result(self):
        import numpy as np
        import torch
        from deblur_models import restore_rgb
        class Broken(torch.nn.Module):
            def forward(self, value):
                return value * 10000
        with self.assertRaisesRegex(ValueError, 'range'):
            restore_rgb(Broken(), np.full((64, 64, 3), 127, dtype=np.uint8), method='nafnet', mix=100, device='cpu')

    def test_commercial_guard_blocks_neural_but_not_classical_methods(self):
        from ai_remaster_gui.commercial_policy import stage_issue
        settings = {'global': {'commercial_model_guard': 'true'}}
        for method in ('off', 'unsharp', 'cas', 'multiscale', 'richardson_lucy'):
            self.assertEqual(stage_issue(settings, 'enhancement', model_override=method), '')
        for method in ('nafnet', 'restormer'):
            self.assertIn('audit', stage_issue(settings, 'enhancement', model_override=method))

    def test_real_classical_outputs_have_shared_review_provenance_and_preserve_audio(self):
        import enhancement_video as backend
        from ai_remaster_gui import denoise_review, frame_inspector
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            self.skipTest('FFmpeg required')
        with tempfile.TemporaryDirectory(prefix='arp_enhancement_') as folder:
            root = Path(folder)
            source = root / 'source.mkv'
            subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
                'testsrc2=size=96x64:rate=30000/1001:duration=0.21', '-f', 'lavfi', '-i',
                'sine=frequency=440:duration=0.21', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(source)], check=True)
            settings = {'global': {}, 'cloud': {'intermediate_format': 'lossless'},
                'enhancement': {'method': 'cas', 'preset': 'gentle', 'mix': '50'}}
            app = SimpleNamespace(settings=settings, deflickered_source_for_downstream=lambda: str(source))
            original = source.read_bytes()
            with patch.object(backend, 'ROOT', root), patch.object(denoise_review, 'ROOT', root), patch.object(frame_inspector, 'ROOT', root):
                for method in ('off', 'unsharp', 'cas', 'multiscale', 'richardson_lucy'):
                    values = {**settings['enhancement'], 'method': method}
                    output = backend.run(source, values, 'lossless')
                    app.enhancement_output = lambda: str(output)
                    settings['enhancement']['method'] = method
                    self.assertEqual(denoise_review.review_pair(app, str(output), 'enhancement'), (source, output))
                    self.assertTrue(denoise_review.output_ready(app, 'enhancement'))
                    stat = output.stat()
                    self.assertEqual(backend.run(source, values, 'lossless').stat().st_mtime_ns, stat.st_mtime_ns)
                    probe = str(Path(ffmpeg).with_name('ffprobe.exe' if Path(ffmpeg).suffix == '.exe' else 'ffprobe'))
                    data = json.loads(subprocess.check_output([probe, '-v', 'error', '-show_streams', '-of', 'json', str(output)], text=True))
                    self.assertIn('audio', [stream['codec_type'] for stream in data['streams']])
                    settings['enhancement']['mix'] = '51'
                    self.assertFalse(denoise_review.output_ready(app, 'enhancement'))
                    settings['enhancement']['mix'] = '50'
            self.assertEqual(source.read_bytes(), original)

    def test_new_stage_routes_to_legacy_without_migrating_legacy_settings(self):
        from ai_remaster_gui import server
        from ai_remaster_gui.runtime_settings import base_settings
        settings = base_settings()
        settings['global'].update(source='input/source.mkv', denoise='true', deflicker='true', enhancement='true',
            cleanup='false', stabilize='false', detail='false')
        settings['enhancement'].update(method='cas', preset='gentle', mix='50')
        legacy = dict(settings['detail'])
        with patch.object(server.APP, 'settings', settings), patch.object(server.APP, 'queue_running', True, create=True):
            active = [stage.key for stage in server.APP.active_stages()]
            self.assertEqual(active[:3], ['denoise', 'deflicker', 'enhancement'])
            expected = server.APP.enhancement_output()
            self.assertEqual(server.APP.cleaned_source_for_downstream(), expected)
            self.assertEqual(server.outpaint_source_for_settings(settings), expected)
            self.assertEqual(settings['detail'], legacy)
            settings['global']['denoise'] = settings['global']['deflicker'] = 'false'
            settings['global']['cleanup'] = 'true'
            ok, message = server.APP.run_stage('cleanup')
            self.assertFalse(ok)
            self.assertIn('Run Detail Enhancement / Deblur first', message)

    @unittest.skipUnless(os.environ.get('ARP_TEST_NEURAL') == '1', 'Explicit local pretrained CUDA integration test')
    def test_real_neural_video_on_the_identical_prepared_clip(self):
        import enhancement_video as backend
        ffmpeg = shutil.which('ffmpeg')
        demo = Path(__file__).resolve().parents[1] / '.cache/detail_deblur/smoke/official-nafnet-demo-blurry.jpg'
        with tempfile.TemporaryDirectory(prefix='arp_neural_video_') as folder:
            root = Path(folder)
            source = root / 'reference.mkv'
            subprocess.run([ffmpeg, '-v', 'error', '-y', '-loop', '1', '-i', str(demo), '-f', 'lavfi',
                '-i', 'sine=frequency=440:duration=0.17', '-vf', 'crop=960:720:0:0', '-t', '0.17',
                '-r', '24', '-c:v', 'ffv1', '-c:a', 'pcm_s16le', str(source)], check=True)
            before = source.read_bytes()
            for method in ('nafnet', 'restormer'):
                output = backend.run(source, {'method': method, 'mix': '50'}, 'lossless', output=root / f'{method}.mkv')
                self.assertTrue(output.is_file())
                probe = str(Path(ffmpeg).with_name('ffprobe.exe' if Path(ffmpeg).suffix.lower() == '.exe' else 'ffprobe'))
                data = json.loads(subprocess.check_output([probe, '-v', 'error', '-show_streams', '-of', 'json', str(output)], text=True))
                self.assertIn('audio', [stream['codec_type'] for stream in data['streams']])
                video = next(stream for stream in data['streams'] if stream['codec_type'] == 'video')
                self.assertEqual((video['width'], video['height']), (960, 720))
                self.assertEqual(video['avg_frame_rate'], '24/1')
                stat = output.stat()
                backend.run(source, {'method': method, 'mix': '50'}, 'lossless', output=output)
                self.assertEqual(stat.st_mtime_ns, output.stat().st_mtime_ns)
            self.assertEqual(source.read_bytes(), before)
