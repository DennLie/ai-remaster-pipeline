"""Offline 0/1/2/3% stabilization benchmark. No downloads or GUI/project writes."""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import re
import subprocess
import sys
from pathlib import Path
from fractions import Fraction

import cv2
import numpy as np

import assess_stabilization as assess
import stabilize_video as backend
from common import file_fingerprint, find_ffmpeg


WARNING = ('Apparent image motion includes pans, foreground motion and matching errors, not just gate weave. '
           'PSNR/MSE are full-frame decoded 8-bit BGR comparisons only for synthetic perturbations against '
           'their digital base; they include zoom, borders and resampling, not historical authenticity. '
           'Footprints are nearest-sampled raster estimates, not bicubic kernel support. '
           'Review cuts, pans, grain and detail visually; no automatic winner or default change.')


def write_json(path, data):
    partial = path.with_suffix(path.suffix + '.tmp')
    partial.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    partial.replace(path)


def load_cases(manifest, corpus_root=None):
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('version') != 1 or not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('Expected version 1 and a nonempty cases list.')
    result, ids = [], set()
    for row in data['cases']:
        if not isinstance(row, dict):
            raise ValueError('Each case must be an object.')
        if corpus_root is not None:
            film, scene = row.get('film'), row.get('scene')
            if any(not isinstance(value, str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,39}', value) for value in (film, scene)):
                raise ValueError('Invalid corpus film/scene ID.')
            root = corpus_root.resolve()
            folder = (root / film).resolve()
            if folder.parent != root:
                raise ValueError('Corpus folder escapes its root.')
            metadata_path = folder / 'corpus_manifest.json'
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            transfer = metadata.get('source_provenance', {})
            if (metadata.get('schema_version') != 1 or metadata.get('id') != film
                    or not re.fullmatch(r'[0-9a-f]{64}', row.get('transfer_sha256', ''))
                    or transfer.get('downloaded_sha256') != row['transfer_sha256']):
                raise ValueError(f'{film}: corpus identity/transfer hash mismatch.')
            choices = [item for item in metadata.get('clips', []) if item.get('name') == scene]
            if len(choices) != 1 or choices[0].get('frames') != row.get('end_frame'):
                raise ValueError(f'{film}/{scene}: curated clip missing or frame count changed.')
            clip = (folder / scene / 'original_native.mkv').resolve()
            if clip.parent.parent != folder:
                raise ValueError('Native clip escapes its corpus film directory.')
            expected = choices[0].get('files', {}).get('original_native.mkv', {}).get('sha256', '')
            if not re.fullmatch(r'[0-9a-f]{64}', expected):
                raise ValueError('Missing native-clip SHA-256.')
            row = {**row, 'source': str(clip), 'sha256': expected,
                   'corpus_provenance': {'manifest': str(metadata_path), 'manifest_fingerprint': file_fingerprint(metadata_path),
                       'transfer': transfer, 'source_page': metadata.get('source_page'),
                       'rights_status': metadata.get('rights_status'), 'catalog_origin': data.get('catalog_origin')}}
        name = row.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) or name.casefold() in ids:
            raise ValueError('Case IDs must be unique, simple directory names (1–64 characters).')
        ids.add(name.casefold())
        if not isinstance(row.get('source'), str) or not row['source']:
            raise ValueError(f'{name}: choose a source file.')
        source = (manifest.parent / row['source']).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        start, end = row.get('start_frame', 0), row.get('end_frame')
        if type(start) is not int or type(end) is not int or start < 0 or end - start < 3:
            raise ValueError(f'{name}: choose explicit start_frame/end_frame, end exclusive, at least 3 frames.')
        cuts = row.get('cut_after_frames', [])
        if not isinstance(cuts, list) or any(type(n) is not int or not 0 <= n < end-start-1 for n in cuts) or len(set(cuts)) != len(cuts):
            raise ValueError(f'{name}: cuts are unique zero-based indices relative to the selected clip.')
        jitter = row.get('synthetic')
        if jitter is not None:
            if (not isinstance(jitter, dict) or type(jitter.get('seed')) is not int or not 0 <= jitter['seed'] < 2**32
                    or any(type(jitter.get(key, 0)) not in (int, float) or not math.isfinite(jitter.get(key, 0))
                           or not 0 <= jitter.get(key, 0) <= limit for key, limit in [('translation_px', 50), ('rotation_deg', 5)])):
                raise ValueError(f'{name}: synthetic requires seed, translation_px 0–50 and rotation_deg 0–5.')
        fingerprint = file_fingerprint(source)
        if row.get('sha256') and row['sha256'] != fingerprint['sha256']:
            raise ValueError(f'{name}: source SHA-256 mismatch.')
        result.append({**row, 'source': str(source), 'start_frame': start, 'end_frame': end,
                       'cuts_annotated': 'cut_after_frames' in row, 'cut_after_frames': sorted(cuts), 'fingerprint': fingerprint})
    return result


def encode_perturbation(source, target, ffmpeg, fps, config):
    """Seeded translations AND rotations of the decoded digital base; stream, never hold a film in RAM."""
    capture, (width, height), _ = assess.open_video(source)
    command = [ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
               '-r', fps, '-i', 'pipe:0', '-an', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(target)]
    transforms = []
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
            try:
                rng = np.random.default_rng(config['seed'])
                while True:
                    ok, image = capture.read()
                    if not ok:
                        break
                    dx, dy = rng.uniform(-config.get('translation_px', 0), config.get('translation_px', 0), 2)
                    angle = rng.uniform(-config.get('rotation_deg', 0), config.get('rotation_deg', 0))
                    matrix = cv2.getRotationMatrix2D((width/2, height/2), float(angle), 1)
                    matrix[:, 2] += (dx, dy)
                    process.stdin.write(cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0).tobytes())
                    transforms.append(matrix.tolist())
            finally:
                process.stdin.close()
            if process.wait():
                raise RuntimeError('Synthetic fixture encoding failed.')
    finally:
        capture.release()
    return transforms


def reference_metrics(reference, variants, target):
    handles = {name: assess.open_video(path)[0] for name, path in {'reference': reference, **variants}.items()}
    totals = {name: [] for name in variants}
    residuals = {name: [] for name in variants}
    try:
        with target.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['frame', 'variant', 'mse_bgr8', 'psnr_db', 'exact_match',
                'reference_translation_px', 'phase_response'])
            writer.writeheader()
            frame = 0
            while True:
                reads = {name: cap.read() for name, cap in handles.items()}
                if not reads['reference'][0]:
                    if any(ok for name, (ok, _) in reads.items() if name != 'reference'):
                        raise ValueError('Reference timeline mismatch.')
                    break
                for name in variants:
                    ok, image = reads[name]
                    base = reads['reference'][1]
                    if not ok or image.shape != base.shape:
                        raise ValueError('Reference frame geometry/count mismatch.')
                    mse = float(np.mean((image.astype(np.float64) - base.astype(np.float64))**2))
                    psnr = 10*math.log10(255**2/mse) if mse else None
                    totals[name].append(mse)
                    h, w = base.shape[:2]
                    first = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)[h//10:h-h//10, w//10:w-w//10].astype(np.float32)
                    second = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[h//10:h-h//10, w//10:w-w//10].astype(np.float32)
                    shift, response = cv2.phaseCorrelate(first, second, cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_32F))
                    trusted = math.isfinite(response) and response >= .2 and all(math.isfinite(n) for n in shift)
                    distance = float(math.hypot(*shift)) if trusted else None
                    residuals[name].append(distance)
                    writer.writerow({'frame': frame, 'variant': name, 'mse_bgr8': mse,
                                     'psnr_db': psnr, 'exact_match': mse == 0, 'reference_translation_px': distance,
                                     'phase_response': response if math.isfinite(response) else None})
                frame += 1
        return {name: {'mean_mse_bgr8': float(np.mean(values)),
                      'pooled_psnr_db': 10*math.log10(255**2/np.mean(values)) if any(values) else None,
                      'exact_match': not any(values),
                      'median_reference_translation_px': float(np.median([n for n in residuals[name] if n is not None])) if any(n is not None for n in residuals[name]) else None,
                      'trusted_reference_frames': sum(n is not None for n in residuals[name]),
                      'rejected_reference_frames': sum(n is None for n in residuals[name])} for name, values in totals.items()}
    finally:
        for capture in handles.values():
            capture.release()


def diagnostics(output, source, frames):
    signature = json.loads(output.with_suffix('.mkv.sig.json').read_text())
    scene_data = json.loads(output.with_suffix('.mkv.scenes.json').read_text())
    if (signature['source_fingerprint'] != file_fingerprint(source) or scene_data['signature'] != signature
            or scene_data['output_fingerprint'] != file_fingerprint(output)):
        raise ValueError('Stale stabilization diagnostics.')
    scenes = scene_data['scenes']
    fallback = [row for row in scenes if row['status'] == 'fallback']
    result = {'fallback_scenes': len(fallback), 'scenes': len(scenes),
              'fallback_frames': sum(row['end']-row['start'] for row in fallback),
              'output_fingerprint': file_fingerprint(output), 'scene_report': str(output.with_suffix('.mkv.scenes.json'))}
    path = output.with_suffix('.mkv.geometry.json')
    if not path.is_file():
        return {**result, 'footprint': None, 'footprint_unavailable_reason': 'Missing geometry report.'}
    geometry = json.loads(path.read_text())
    if geometry['signature'] != signature or geometry['output_fingerprint'] != file_fingerprint(output):
        raise ValueError('Stale geometric footprint.')
    valid, cropped = [], []
    for row, span in zip(geometry['scenes'], signature['shots']):
        if row.get('error'):
            return {**result, 'footprint': None, 'footprint_unavailable_reason': row['error']}
        if [row['start'], row['end']] != span:
            raise ValueError('Footprint scene mismatch.')
        mask = output.with_suffix('.geometry') / row['mask']
        if mask.resolve().parent != output.with_suffix('.geometry').resolve() or file_fingerprint(mask) != row['mask_fingerprint']:
            raise ValueError('Changed footprint mask.')
        counts = row['valid_pixels']
        pixels = geometry['width'] * geometry['height']
        if len(counts) != span[1]-span[0] or any(type(n) is not int or not 0 <= n <= pixels for n in counts):
            raise ValueError('Invalid footprint counts.')
        fractions = np.array(counts) / pixels
        valid.extend(fractions * 100)
        cropped.extend(np.clip((1-fractions*(1-row['zoom']/100)**2)*100, 0, 100))
    if len(valid) != frames:
        raise ValueError('Incomplete footprint timeline.')
    return {**result, 'footprint': {'mean_valid_output_pct': float(np.mean(valid)),
        'worst_valid_output_pct': float(min(valid)), 'mean_cropped_source_area_estimate_pct': float(np.mean(cropped))}}


def cadence(path, ffmpeg):
    binary = str(Path(ffmpeg).with_name('ffprobe.exe' if Path(ffmpeg).suffix.lower() == '.exe' else 'ffprobe'))
    data = json.loads(subprocess.check_output([binary, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,r_frame_rate', '-of', 'json', str(path)], text=True))['streams'][0]
    rate = Fraction(data['avg_frame_rate'])
    if rate <= 0 or rate > 120 or rate != Fraction(data['r_frame_rate']):
        raise ValueError('Benchmark requires compatible constant-cadence metadata (1–120 fps).')
    return data['width'], data['height'], rate


def run(manifest, output_dir, *, smoothing=12, max_shift=48, max_angle=3, ffmpeg=None, corpus_root=None, zooms=(0, 1, 2, 3)):
    if type(smoothing) is not int or not 1 <= smoothing <= 1000 or type(max_shift) is not int or not 0 <= max_shift <= 500 or not math.isfinite(max_angle) or not 0 <= max_angle <= 180:
        raise ValueError('Invalid motion parameters.')
    if (not isinstance(zooms, (list, tuple)) or not zooms
            or any(type(zoom) is not int or zoom not in (0, 1, 2, 3) for zoom in zooms)
            or len(set(zooms)) != len(zooms)):
        raise ValueError('Choose unique zooms from 0, 1, 2, 3.')
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    cases = load_cases(manifest, corpus_root)
    ffmpeg = find_ffmpeg(ffmpeg)
    cv2.setRNGSeed(0)
    cv2.setNumThreads(1)
    # A new directory is mandatory: never replace an earlier benchmark or source.
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'version': 1, 'status': 'running', 'warning': WARNING, 'cases': [],
        'manifest': cases, 'manifest_fingerprint': file_fingerprint(manifest),
        'settings': {'smoothing': smoothing, 'max_shift': max_shift, 'max_angle': max_angle, 'zooms': list(zooms), 'profile': 'lossless'},
        'runtime': {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__,
                    'ffmpeg': subprocess.check_output([ffmpeg, '-version'], text=True)},
        'code': {name: file_fingerprint(Path(__file__).parent / name) for name in
                 ['benchmark_stabilization.py', 'assess_stabilization.py', 'stabilize_video.py', 'stabilization_footprint.py',
                  'common.py', 'artifact_ids.py', 'generate_references.py', 'intermediate_video.py']}}
    write_json(output_dir / 'benchmark.json', report)
    try:
        for case in cases:
            folder = output_dir / case['id']
            folder.mkdir()
            clip = folder / 'base.mkv'
            subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', case['source'], '-map', '0:v:0', '-vf',
                f"trim=start_frame={case['start_frame']}:end_frame={case['end_frame']},setpts=PTS-STARTPTS",
                '-an', '-fps_mode', 'passthrough', '-c:v', 'ffv1', str(clip)], check=True)
            info = backend.probe_video(clip)
            metadata = cadence(clip, ffmpeg)
            expected_frames = case['end_frame']-case['start_frame']
            if info.frame_count != expected_frames:
                raise ValueError(f"{case['id']}: selected source range is unavailable.")
            modes = [('real', clip)]
            if case.get('synthetic') is not None:
                rate = str(metadata[2])
                synthetic = folder / 'synthetic.mkv'
                matrices = encode_perturbation(clip, synthetic, ffmpeg, rate, case['synthetic'])
                write_json(folder / 'injected_transforms.json', {'config': case['synthetic'], 'matrices': matrices,
                    'note': 'Forward digital-base to perturbed-input affine matrices; positive angle follows OpenCV convention. Input and output synthesis use 8-bit BGR.'})
                modes.append(('synthetic', synthetic))
            for mode, source in modes:
                work = folder / mode
                work.mkdir()
                if cadence(source, ffmpeg) != metadata:
                    raise ValueError('Benchmark input geometry/cadence mismatch.')
                _, spans = backend.detect_shot_ranges(source, .075, 1)
                cuts = {start-1 for start, _ in spans[1:]}
                cut_manifest = work / 'cuts.csv'
                with cut_manifest.open('w', newline='') as file:
                    writer = csv.writer(file);writer.writerow(['start_frame', 'end_frame'])
                    writer.writerows(spans)
                variants = {}
                commands = {}
                for zoom in zooms:
                    name = f'zoom{zoom}'
                    target = work / (name + '.mkv')
                    command = [sys.executable, str(Path(backend.__file__)), '--source', str(source), '--output', str(target),
                        '--zoom', str(zoom), '--smoothing', str(smoothing), '--max-shift', str(max_shift), '--max-angle', str(max_angle),
                        '--shot-manifest', str(cut_manifest), '--intermediate-profile', 'lossless', '--ffmpeg', ffmpeg]
                    commands[name] = command
                    write_json(work / 'commands.json', commands)
                    with (work / (name + '.log')).open('w', encoding='utf-8') as log:
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if cadence(target, ffmpeg) != metadata:
                        raise ValueError('Stabilization output geometry/cadence mismatch.')
                    variants[name] = target
                qa = assess.evaluate(source, variants, work, cut_after_frames=set(case['cut_after_frames']) | cuts,
                    sample_frames={round((expected_frames-1)*part/4) for part in range(5)})
                if qa['frames'] != expected_frames:
                    raise ValueError('Decoded benchmark timeline differs from the selected range.')
                metrics = {name: {**qa['measurements'][name], **diagnostics(path, source, expected_frames), 'output': str(path)} for name, path in variants.items()}
                reference = reference_metrics(clip, {'input': source, **variants}, work / 'reference_frames.csv') if mode == 'synthetic' else None
                annotations = set(case['cut_after_frames'])
                result = {'case': case['id'], 'mode': mode, 'frames': expected_frames, 'input': str(source),
                    'input_fingerprint': file_fingerprint(source), 'detected_cut_after_frames': sorted(cuts),
                    'cut_detection': {'matched': len(cuts & annotations), 'false_positive': len(cuts-annotations), 'missed': len(annotations-cuts)} if case['cuts_annotated'] else None,
                    'baseline_motion': qa['measurements']['original'], 'variants': metrics, 'reference_metrics': reference}
                report['cases'].append(result)
                write_json(output_dir / 'benchmark.json', report)
            if file_fingerprint(Path(case['source'])) != case['fingerprint']:
                raise ValueError('Original source changed during the benchmark.')
        with (output_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
            fields = ['case', 'mode', 'variant', 'median_translation_px', 'median_rotation_deg', 'median_translation_acceleration_px',
                      'trusted_adjacent_pairs', 'rejected_adjacent_pairs', 'fallback_scenes', 'fallback_frames',
                      'mean_valid_output_pct', 'worst_valid_output_pct', 'mean_cropped_source_area_estimate_pct', 'pooled_psnr_db']
            writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore');writer.writeheader()
            for row in report['cases']:
                for variant, measurements in row['variants'].items():
                    writer.writerow({**measurements, **(measurements['footprint'] or {}), 'case': row['case'], 'mode': row['mode'],
                        'variant': variant, 'pooled_psnr_db': (row['reference_metrics'] or {}).get(variant, {}).get('pooled_psnr_db')})
        with (output_dir / 'visual_review.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file);writer.writerow(['case', 'mode', 'variant', 'pan_preserved', 'cuts_ok', 'grain_detail_ok', 'edges_ok', 'notes'])
            writer.writerows([row['case'], row['mode'], name, '', '', '', '', ''] for row in report['cases'] for name in row['variants'])
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(output_dir / 'benchmark.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--corpus-root', type=Path, help='Prepared PR #18 corpus; reads its manifests and native-clip hashes, never downloads')
    parser.add_argument('--output-dir', type=Path, required=True, help='New, nonexistent results directory')
    parser.add_argument('--smoothing', type=int, default=12)
    parser.add_argument('--max-shift', type=int, default=48)
    parser.add_argument('--max-angle', type=float, default=3)
    parser.add_argument('--zooms', type=int, nargs='+', choices=(0, 1, 2, 3), default=[0, 1, 2, 3],
                        help='Zoom percentages to render; use 3 for fixed-zoom motion comparison')
    parser.add_argument('--ffmpeg')
    args = parser.parse_args()
    run(args.manifest, args.output_dir.resolve(), smoothing=args.smoothing, max_shift=args.max_shift,
        max_angle=args.max_angle, ffmpeg=args.ffmpeg, corpus_root=args.corpus_root, zooms=args.zooms)
    print(f'Benchmark complete: {args.output_dir / "benchmark.json"}. {WARNING}')


if __name__ == '__main__':
    main()
