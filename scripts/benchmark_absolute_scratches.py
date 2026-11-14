"""Bounded official AbsoluteDegradation scratch-channel experiment, not full pipeline.

Run generate in the isolated AbsoluteDegradation environment, score in ARP's.
Downloads only 72 reference PNGs, capped at 250 MB; never invokes GPU/models.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import urllib.request

import cv2
import numpy as np

PIN = '542856fd7e275a6dcec80311f7f7b4541bdc4f70'
CLIPS = ('000', '010', '020')
SEEDS = (20261106, 20261107)
LIMIT = 250_000_000
THRESHOLD = .1


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Cannot save {path}')


def labels(soft):
    if not np.isfinite(soft).all() or np.any((soft < 0) | (soft > 1)):
        raise ValueError('Invalid soft ground truth')
    result = np.zeros(soft.shape, np.uint8)
    result[soft > 0] = 127
    result[soft >= THRESHOLD] = 255
    return result


def generate(root, upstream):
    from absolute_degradation.scratches.film_scratch_generator import FilmScratchGenerator, ScratchConfig
    if root.exists():
        raise FileExistsError('Refusing to overwrite experiment')
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise ValueError('Generator checkout differs from experiment pin')
    dirty = subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if dirty.strip():
        raise ValueError('Generator checkout has tracked changes')
    root.mkdir(parents=True)
    protocol = {'generator_commit': PIN, 'clips': CLIPS, 'frames': 24, 'seeds': SEEDS,
                'split': {'000': 'development', '010': 'test', '020': 'test'},
                'gt_threshold': THRESHOLD, 'weak_nonzero_gt': 'ignore=127',
                'scope': 'Official scratch component only; no textures, dust, grain, blur or gate weave. No tuning.',
                'source': 'Kaggle Synthetic Degraded Eval version 6 gt/ reference images; not segmentation masks.',
                'numpy': np.__version__, 'opencv': cv2.__version__, 'python': platform.python_version()}
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    total, sources = 0, []
    for clip in CLIPS:
        folder = root / clip
        clean_dir = folder / 'reference'
        clean_dir.mkdir(parents=True)
        images = []
        for i in range(24):
            url = ('https://www.kaggle.com/api/v1/datasets/download/absolutedegradation/'
                   f'absolutecinema-synthetic-degraded-eval/gt%2F{clip}%2F{i:08d}.png?datasetVersionNumber=6')
            target = clean_dir / f'{i:06d}.png'
            with urllib.request.urlopen(url, timeout=60) as response, target.open('xb') as stream:
                while block := response.read(1 << 20):
                    total += len(block)
                    if total > LIMIT:
                        raise ValueError('Reference download cap exceeded; partial experiment retained')
                    stream.write(block)
            image = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError('Reference response is not a decodable image')
            images.append(image)
            sources.append({'url': url, 'sha256': digest(target), 'bytes': target.stat().st_size})
        for seed in SEEDS:
            target = folder / str(seed)
            target.mkdir()
            # Canonical YAML configuration, conditioned on damage presence for positive controls.
            config = ScratchConfig(probability=1.0, count_range=(1, 5), width_range_microns=(5., 40.),
                                   intensity_range=(.2, 1.), weave_amplitude_pixels=2., prob_curved=.2,
                                   lifespan_range=(2, 10), prob_white=.6)
            generator = FilmScratchGenerator(fps=24, config=config, seed=seed)
            damaged, masks = generator.apply_vertical_scratches(images, return_mask=True)
            again, again_masks = FilmScratchGenerator(fps=24, config=config, seed=seed).apply_vertical_scratches(images, return_mask=True)
            if not all(np.array_equal(a, b) for a, b in zip(damaged + masks, again + again_masks)):
                raise ValueError('Generator replay is not deterministic')
            for i, (image, mask) in enumerate(zip(damaged, masks)):
                save(target / f'{i:06d}.png', image)
                np.save(target / f'{i:06d}.soft.npy', mask, allow_pickle=False)
                save(target / f'{i:06d}.gt.png', labels(mask))
            (target / 'config.json').write_text(json.dumps(asdict(config), indent=2), encoding='utf-8')
        print('Generated', clip, '24 frames, two seeds, deterministic replay verified', flush=True)
    (root / 'acquisition.json').write_text(json.dumps({'bytes': total, 'sources': sources}, indent=2), encoding='utf-8')
    (root / 'generated.json').write_text(json.dumps({'complete': True, 'positive_frames': 144}), encoding='utf-8')


def score(root):
    from unittest.mock import patch
    import compare_ridge_integration as integration
    import benchmark_defect_masks as evaluation
    if not (root / 'generated.json').is_file():
        raise ValueError('Incomplete generated experiment')
    target = root / 'scored'
    if target.exists():
        raise FileExistsError('Refusing to overwrite scores')
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['gt_threshold'] != THRESHOLD or protocol['generator_commit'] != PIN:
        raise ValueError('Protocol differs from fixed experiment')
    target.mkdir()
    rows = []
    for clip in CLIPS:
        for condition in ('reference', *(str(s) for s in SEEDS)):
            folder = root / clip / condition
            images = [cv2.imread(str(folder / f'{i:06d}.png')) for i in range(24)]
            if any(f is None for f in images):
                raise ValueError('Missing input frame')
            truth = [np.zeros(images[0].shape[:2], np.uint8) if condition == 'reference'
                     else cv2.imread(str(folder / f'{i:06d}.gt.png'), 0) for i in range(24)]
            predictions = {'ridge_core': integration.detect_sequence(images, .65)}
            for name in ('legacy', 'temporal', 'ridge'):
                with patch.object(integration.dust, 'frames', side_effect=lambda _: iter(images)):
                    predictions[name] = [mask for _, mask, _ in integration.dust.masks_for(
                        Path('frozen-input'), integration.dust.options({'detector': name, 'sensitivity': .65, 'dilate': 0}))]
            for name, masks in predictions.items():
                counts = [evaluation.confusion(gt, pred) for gt, pred in zip(truth, masks)]
                destination = target / clip / condition / name
                destination.mkdir(parents=True)
                for i, mask in enumerate(masks):
                    save(destination / f'{i:06d}.png', mask)
                rows.append({'clip': clip, 'split': protocol['split'][clip], 'condition': condition,
                             'method': name, 'per_frame': counts,
                             'metrics': evaluation.metrics(evaluation.sum_counts(counts))})
            print('Scored', clip, condition, flush=True)
    report = {'protocol': protocol, 'rows': rows, 'numpy': np.__version__, 'opencv': cv2.__version__,
              'code_hashes': {p: digest(integration.ROOT / p) for p in (
                  'ai_remaster_gui/dust.py', 'scripts/film_scratch_ridge.py', 'scripts/film_defect_masks.py',
                  'scripts/cleanup_video.py', 'scripts/benchmark_absolute_scratches.py')},
              'limitations': 'Scratch-channel controls only. Reference activity is not certified real-damage FPR. No DART inference or tuning.'}
    (target / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('generate', 'score'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    args = parser.parse_args()
    if args.mode == 'generate':
        if args.upstream is None:
            parser.error('--upstream is required for generation')
        generate(args.root.resolve(), args.upstream.resolve())
    else:
        score(args.root.resolve())
