import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import benchmark_stabilization as bench


class StabilizationBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='arp-benchmark-stabilization-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.mkv'
        self.ffmpeg = bench.find_ffmpeg()
        subprocess.run([self.ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
            'testsrc2=size=160x128:rate=12:duration=1', '-c:v', 'ffv1', str(self.source)], check=True)
        self.manifest = self.root / 'cases.json'
        self.case = {'id': 'fixture', 'source': 'source.mkv', 'start_frame': 0, 'end_frame': 12,
            'sha256': bench.file_fingerprint(self.source)['sha256'], 'cut_after_frames': [],
            'synthetic': {'seed': 2026, 'translation_px': 1, 'rotation_deg': .25}}
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps({'version': 1, 'cases': [self.case]}))

    def test_validates_before_creating_any_output(self):
        for values in ({'id': '../escape'}, {'end_frame': 1}, {'sha256': 'wrong'},
                       {'cut_after_frames': [11]}, {'synthetic': {'seed': 1, 'rotation_deg': float('nan')}}):
            with self.subTest(values=values):
                original = dict(self.case)
                self.case.update(values);self.write_manifest()
                with self.assertRaises(ValueError):
                    bench.run(self.manifest, self.root / 'output')
                self.assertFalse((self.root / 'output').exists())
                self.case = original

    def test_existing_results_are_never_overwritten(self):
        output = self.root / 'output';output.mkdir()
        marker = output / 'keep.txt';marker.write_text('user data')
        with self.assertRaises(FileExistsError):
            bench.run(self.manifest, output)
        self.assertEqual(marker.read_text(), 'user data')

    def test_invalid_zoom_selections_create_no_output(self):
        for zooms in ((), (3, 3), (4,), (True,), (3.0,), '3'):
            with self.subTest(zooms=zooms), self.assertRaisesRegex(ValueError, 'unique zooms'):
                bench.run(self.manifest, self.root / 'invalid-zooms', zooms=zooms)
            self.assertFalse((self.root / 'invalid-zooms').exists())

    def test_cli_fixed_zoom_renders_only_the_requested_candidate(self):
        self.case.pop('synthetic');self.write_manifest()
        output = self.root / 'fixed-zoom'
        env = {**os.environ, 'PATH': str(Path(self.ffmpeg).parent) + os.pathsep + os.environ.get('PATH', '')}
        done = subprocess.run([sys.executable, str(Path(bench.__file__)), '--manifest', str(self.manifest),
            '--output-dir', str(output), '--ffmpeg', self.ffmpeg, '--zooms', '3'],
            capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr[-4000:])
        result = json.loads((output / 'benchmark.json').read_text())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['settings']['zooms'], [3])
        self.assertEqual(set(result['cases'][0]['variants']), {'zoom3'})
        self.assertFalse((output / 'fixture/real/zoom0.mkv').exists())

    def test_unannotated_cuts_are_not_treated_as_a_no_cut_ground_truth(self):
        del self.case['cut_after_frames'];self.write_manifest()
        self.assertFalse(bench.load_cases(self.manifest)[0]['cuts_annotated'])

    def test_seeded_rotation_and_translation_are_reproducible(self):
        config = self.case['synthetic']
        first, second = self.root / 'first.mkv', self.root / 'second.mkv'
        a = bench.encode_perturbation(self.source, first, self.ffmpeg, '12/1', config)
        b = bench.encode_perturbation(self.source, second, self.ffmpeg, '12/1', config)
        self.assertEqual(a, b);self.assertEqual(len(a), 12)
        self.assertNotEqual(a[0][0][1], 0)
        result = bench.reference_metrics(first, {'same': second}, self.root / 'reference.csv')
        self.assertEqual(result['same']['mean_mse_bgr8'], 0)
        self.assertTrue(result['same']['exact_match'])
        self.assertIsNone(result['same']['pooled_psnr_db'])

    def test_prepared_pr18_corpus_reuses_verified_clip_and_transfer_hashes(self):
        corpus = self.root / 'corpus'
        folder = corpus / 'film_a';clip_folder = folder / 'scene_a';clip_folder.mkdir(parents=True)
        clip = clip_folder / 'original_native.mkv';clip.write_bytes(self.source.read_bytes())
        metadata = {'schema_version': 1, 'id': 'film_a', 'source_provenance': {'downloaded_sha256': 'a'*64},
            'clips': [{'name': 'scene_a', 'frames': 12, 'files': {'original_native.mkv': {'sha256': bench.file_fingerprint(clip)['sha256']}}}]}
        (folder / 'corpus_manifest.json').write_text(json.dumps(metadata))
        self.case = {'id': 'existing_scene', 'film': 'film_a', 'scene': 'scene_a', 'transfer_sha256': 'a'*64, 'end_frame': 12}
        self.write_manifest()
        rows = bench.load_cases(self.manifest, corpus)
        self.assertEqual(rows[0]['source'], str(clip))
        self.assertIn('corpus_provenance', rows[0])
        clip.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            bench.load_cases(self.manifest, corpus)

    def test_failed_run_keeps_an_explicit_failed_status(self):
        output = self.root / 'failed'
        with patch.object(bench.backend, 'detect_shot_ranges', side_effect=ValueError('controlled failure')):
            with self.assertRaisesRegex(ValueError, 'controlled failure'):
                bench.run(self.manifest, output)
        self.assertEqual(json.loads((output / 'benchmark.json').read_text())['status'], 'failed')

    def test_real_backend_runs_all_zooms_without_mutating_sources(self):
        output = self.root / 'results'
        before = self.source.read_bytes()
        env = {**os.environ, 'PATH': str(Path(self.ffmpeg).parent) + os.pathsep + os.environ.get('PATH', '')}
        command = [sys.executable, str(Path(bench.__file__)), '--manifest', str(self.manifest), '--output-dir', str(output), '--ffmpeg', self.ffmpeg]
        done = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
        self.assertEqual(done.returncode, 0, done.stderr[-4000:] + done.stdout[-1000:])
        result = json.loads((output / 'benchmark.json').read_text())
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['cases']), 2)
        self.assertEqual(result['settings']['zooms'], [0, 1, 2, 3])
        self.assertIsNone(result['cases'][0]['reference_metrics'])
        self.assertEqual(set(result['cases'][1]['reference_metrics']), {'input', 'zoom0', 'zoom1', 'zoom2', 'zoom3'})
        for row in result['cases']:
            self.assertEqual(row['frames'], 12)
            self.assertEqual(set(row['variants']), {'zoom0', 'zoom1', 'zoom2', 'zoom3'})
            for name, metrics in row['variants'].items():
                self.assertIsNotNone(metrics['footprint'])
                self.assertIn('fallback_frames', metrics)
                self.assertEqual(json.loads(Path(metrics['scene_report']).read_text())['signature']['identity']['zoom'], int(name[-1]))
        self.assertTrue((output / 'fixture/real/motion_frames.csv').is_file())
        self.assertTrue((output / 'fixture/synthetic/reference_frames.csv').is_file())
        self.assertTrue((output / 'summary.csv').is_file())
        self.assertEqual(self.source.read_bytes(), before)
