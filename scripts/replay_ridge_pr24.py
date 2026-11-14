"""Offline five-film PR24 positive control. Requires existing clips and Git object.

No downloads, models, repairs, parameter tuning or application/project writes.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import types

import cv2
import numpy as np

import compare_ridge_integration as comparison

CASES = (
    ('lumiere_train', 'approach_probe', 'development', '17941b86bab117c937bc5ba7236451e933cdbb6864db66f4a6fd40ceebe78e46', .8073),
    ('great_train_robbery', 'interior_action_probe', 'development', 'd3e7b776921ba31017ccd703766111f320fecd9e2c570d27d0a6a4ca95bc1a9b', .3071),
    ('melies_moon', 'stage_probe', 'development', '5c107f8e73adca1dbebfeb34d80505c194c55254fc9db91cdb07eae4a98a89f4', .4976),
    ('caligari', 'expressionist_set_probe', 'holdout', 'ef0c2032dd95861a40d5fbe27f3fe2413343c0124d969c492803d23b8eb730ce', .7500),
    ('gertie', 'dinosaur_motion_probe', 'holdout', '9589d11d965194540b7e3cc5c9adb6f3c49fa45e66d4b2f909525a068c9e5bb0', .6110),
)


def pinned(path):
    return subprocess.check_output(['git', '-C', str(comparison.ROOT), 'show',
                                    f'{comparison.REFERENCE_COMMIT}:{path}'])


def functions(path, names):
    source = pinned(path)
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f'Missing pinned functions: {path}')
    namespace = {'np': np, 'cv2': cv2}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), namespace)
    return namespace


def inventory(corpus):
    rows = []
    for film, scene, split, expected, historical_f1 in CASES:
        source = corpus / film / scene / 'original_native.mkv'
        manifest = corpus / film / 'corpus_manifest.json'
        if not source.is_file() or not manifest.is_file():
            raise FileNotFoundError(f'Missing existing corpus clip/manifest: {source}')
        document = json.loads(manifest.read_text(encoding='utf-8'))
        clip = next(c for c in document['clips'] if c['name'] == scene)
        actual = comparison.sha(source)
        if (actual != expected or clip['files']['original_native.mkv']['sha256'] != expected
                or clip['frames'] != 24 or document['id'] != film):
            raise ValueError(f'Corpus integrity mismatch: {source}')
        rows.append({'film': film, 'scene': scene, 'split': split, 'source': str(source),
                     'source_sha256': actual, 'corpus_manifest_sha256': comparison.sha(manifest),
                     'transfer_sha256': document['source_provenance']['downloaded_sha256'],
                     'geometry': clip['geometry'], 'historical_rounded_f1': historical_f1})
    return rows


def run(corpus, output):
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output exists; refusing overwrite.')
    cases = inventory(corpus)
    # Run the pinned generator without importing its research detector/GT harness.
    ev = types.ModuleType('pinned_evaluation')
    generator = pinned('scripts/restoration_evaluation.py')
    exec(compile(generator, 'pinned/restoration_evaluation.py', 'exec'), ev.__dict__)
    scoring = functions('scripts/diagnose_scratch_masks.py', ('confusion', 'summarize_confusion'))
    legacy = functions('scripts/cleanup_video.py', ('_contiguous_true_runs', 'ai_scratch_mask'))['ai_scratch_mask']
    reference, _ = comparison.reference_detector()
    ffmpeg, ffprobe = ev.find_binary('ffmpeg'), ev.find_binary('ffprobe')
    for row in cases:
        info = ev.probe(Path(row['source']), ffprobe)
        if info != {**row['geometry'], 'frames': 24}:
            raise ValueError(f'Frame/cadence/geometry mismatch: {row["source"]}')
    output.mkdir(parents=True)
    for row in cases:
        folder = output / row['film']
        synthetic = folder / 'synthetic'
        manifest = ev.synthesize(source=Path(row['source']), output_dir=synthetic,
                                 film=row['film'], scene=row['scene'], kind='vertical_scratches',
                                 preset='gentle', seed=2026, ffmpeg=ffmpeg, ffprobe=ffprobe)
        info = {k: manifest[k] for k in ('width', 'height', 'fps', 'frames')}
        images = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                  for f in ev.decode(synthetic / 'degraded.mkv', ffmpeg, info)]
        predictions, agreement = comparison.compare_frames(images, reference)
        row['suspected_cuts'] = [
            {'before_frame': i, 'median_luma_difference': float(np.median(cv2.absdiff(
                cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(images[i-1], cv2.COLOR_BGR2GRAY))))}
            for i in range(1, 24) if float(np.median(cv2.absdiff(
                cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(images[i-1], cv2.COLOR_BGR2GRAY)))) >= 22]
        predictions['historical_legacy'] = [legacy(f, .65) for f in images]
        predictions['current_legacy'] = [comparison.dust.ai_scratch_mask(f, .65) for f in images]
        truth = [cv2.imread(str(synthetic / 'masks' / f'{i:06d}.png'), 0) for i in range(24)]
        methods = {}
        for name, masks in predictions.items():
            counts = [scoring['confusion'](mask, gt) for mask, gt in zip(masks, truth)]
            methods[name] = {'aggregate': scoring['summarize_confusion'](counts), 'per_frame': counts}
            target = folder / name
            target.mkdir()
            for i, mask in enumerate(masks):
                if not cv2.imwrite(str(target / f'{i:06d}.png'), mask):
                    raise OSError('Cannot save native mask')
        for i in (0, 12, 23):
            panels = [images[i], cv2.cvtColor(truth[i], cv2.COLOR_GRAY2BGR)]
            for name in ('historical_legacy', 'pr24', 'arp_wrapper'):
                overlay = images[i].copy()
                overlay[predictions[name][i] > 0] = (255, 0, 255)
                panels.append(overlay)
            if not cv2.imwrite(str(folder / f'native-input-gt-legacy-pr24-wrapper-{i:06d}.png'), np.hstack(panels)):
                raise OSError('Cannot save native comparison')
        row['methods'] = methods
        row['agreement'] = agreement
        row['synthetic_manifest_sha256'] = comparison.sha(synthetic / 'manifest.json')
        row['published_f1_reproduced_to_rounding'] = abs(
            methods['pr24']['aggregate']['pixel_f1'] - row['historical_rounded_f1']) <= .00005
        if comparison.sha(Path(row['source'])) != row['source_sha256']:
            raise ValueError('Source changed during experiment')
        (folder / 'comparison.json').write_text(json.dumps(row, indent=2) + '\n', encoding='utf-8')
        print(row['film'], {k: round(v['aggregate']['pixel_f1'], 6) for k, v in methods.items()}, flush=True)
    by_split = {split: {name: scoring['summarize_confusion']([
        counts for row in cases if row['split'] == split
        for counts in row['methods'][name]['per_frame']]) for name in cases[0]['methods']}
        for split in ('development', 'holdout')}
    report = {'reference_commit': comparison.REFERENCE_COMMIT, 'cases': cases, 'by_split': by_split,
              'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
              'ffmpeg': subprocess.check_output([ffmpeg, '-version'], text=True).splitlines()[0],
              'settings': {'sensitivity': .65, 'additional_dilation': 0, 'seed': 2026, 'preset': 'gentle'},
              'source_hashes': {p: hashlib.sha256(pinned(p)).hexdigest() for p in (
                  'scripts/restoration_evaluation.py', 'scripts/cleanup_video.py',
                  'scripts/diagnose_scratch_masks.py', comparison.REFERENCE_PATH)},
              'current_hashes': {p: comparison.sha(comparison.ROOT / p) for p in (
                  'scripts/film_scratch_ridge.py', 'scripts/cleanup_video.py', 'ai_remaster_gui/dust.py',
                  'scripts/compare_ridge_integration.py', 'scripts/replay_ridge_pr24.py')},
              'scope': 'Known synthetic damage only; historical holdouts are reproduction controls, not new validation. No repair or real-damage accuracy claim.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.corpus, args.output)
