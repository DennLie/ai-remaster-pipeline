"""Stabilization zoom variants; inspection never changes the pipeline settings."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from .config import ROOT, SCRIPTS
from .paths import is_unfinished_media_artifact, rel, resolve, resolve_video_source

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import stabilize_video as backend

METHODS = ('zoom0', 'zoom3', 'current')


def variant_values(values, method):
    if method not in METHODS:
        raise ValueError('Choose 0%, 3% or current stabilization settings.')
    return {**values, **({'zoom': '0' if method == 'zoom0' else '3'} if method != 'current' else {})}


def arguments(source, values, profile='high'):
    args = backend.build_parser().parse_args(['--source', str(source), '--intermediate-profile', backend.canonical_profile(profile)])
    for key, lower, upper in (('smoothing', 1, 1000), ('max_shift', 0, 500), ('max_angle', 0, 180),
                              ('zoom', 0, 20), ('shot_threshold', .001, 2), ('min_shot_seconds', .1, 360000)):
        raw = values.get(key, getattr(args, key))
        if isinstance(raw, bool):
            raise ValueError(f'Invalid stabilization {key}.')
        value = float(raw)
        if not math.isfinite(value) or not lower <= value <= upper or key in ('smoothing', 'max_shift') and value != int(value):
            raise ValueError(f'Invalid stabilization {key}.')
        setattr(args, key, int(value) if key in ('smoothing', 'max_shift') else value)
    scene_aware = str(values.get('scene_aware', 'true')).lower()
    if scene_aware not in ('true', 'false'):
        raise ValueError('Choose scene resets on or off.')
    args.single_shot = scene_aware != 'true'
    return args


def output_for(source, values, profile='high'):
    return rel(backend.default_output(resolve_video_source(source), arguments(source, values, profile))) if source else ''


def validate_variants(rows):
    if not isinstance(rows, list) or not 1 <= len(rows) <= len(METHODS):
        raise ValueError('Choose 1–3 stabilization variants.')
    methods = []
    for row in rows:
        if not isinstance(row, dict) or row.get('method') not in METHODS or row.get('preset') != 'settings':
            raise ValueError('Choose supported stabilization zoom variants.')
        methods.append(row['method'])
    if len(set(methods)) != len(methods):
        raise ValueError('Duplicate stabilization variants.')


def review_pair(app, candidate):
    from .denoise_review import selected_output
    from .frame_inspector import _video_path
    source = _video_path(app.cleaned_source_for_downstream())
    problem = app.preprocessing_problem('stabilize')
    if problem:
        raise ValueError(problem)
    for upstream in app.preprocessing_upstream('stabilize'):
        selected_output(app, upstream)
    output = resolve(candidate)
    if (output.resolve().parent != (ROOT / 'intermediate' / 'stabilized').resolve()
            or output.suffix.lower() != '.mkv' or is_unfinished_media_artifact(output)):
        raise ValueError('Only finished ARP stabilization masters may be reviewed.')
    output = _video_path(str(output))
    provenance = json.loads(output.with_suffix(output.suffix + '.sig.json').read_text(encoding='utf-8'))
    if not isinstance(provenance, dict):
        raise ValueError('Invalid stabilization provenance.')
    profile = app.settings.get('cloud', {}).get('intermediate_format', 'high')
    values = app.settings.get('stabilize', {})
    for method in METHODS:
        args = arguments(source, variant_values(values, method), profile)
        expected = backend.default_output(source, args)
        if expected.resolve() != output.resolve():
            continue
        shots = provenance.get('shots', [])
        if (not isinstance(shots, list) or not shots or any(not isinstance(row, list) or len(row) != 2
                or any(type(n) is not int for n in row) or row[0] < 0 or row[1] <= row[0] for row in shots)
                or shots[0][0] != 0 or any(a[1] != b[0] for a, b in zip(shots, shots[1:]))):
            raise ValueError('Invalid stabilization scene provenance.')
        if provenance != backend.output_signature(source, args, shots):
            raise ValueError('Stabilization result has stale input, parameters or format.')
        return source, output
    raise ValueError('Stabilization result does not match current comparison settings.')


def scene_report(output):
    """Old cached results remain usable, but never invent a successful tracking report."""
    try:
        report = json.loads(output.with_suffix(output.suffix + '.scenes.json').read_text(encoding='utf-8'))
        signature = json.loads(output.with_suffix(output.suffix + '.sig.json').read_text(encoding='utf-8'))
        if report['signature'] == signature and report['output_fingerprint'] == backend.file_fingerprint(output):
            scenes = report['scenes']
            if (isinstance(scenes, list) and len(scenes) == len(signature['shots'])
                    and all(isinstance(row, dict) and [row.get('start'), row.get('end')] == span
                        and row.get('status') in ('stabilized', 'fallback') and isinstance(row.get('reason'), str)
                        and type(row.get('tracking_report_available')) is bool
                        for row, span in zip(scenes, signature['shots']))):
                return scenes
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def footprint(app, candidate, frame):
    """Read one verified frame's diagnostic mask; never trigger stabilization or replay."""
    from .frame_inspector import _decode, _metadata, _prune
    from .frame_differences import _validate_pair
    if type(frame) is not int or frame < 0:
        raise ValueError('Choose a nonnegative frame index.')
    source, output = review_pair(app, candidate)
    count = _validate_pair(source, output)
    if frame >= count:
        raise ValueError('Footprint frame is outside the comparison timeline.')
    target = output.with_suffix(output.suffix + '.geometry.json')
    if not target.is_file():
        raise ValueError('No geometric footprint for this cached result. Explicitly rerender to record it.')
    data = json.loads(target.read_text(encoding='utf-8'))
    signature = json.loads(output.with_suffix(output.suffix + '.sig.json').read_text(encoding='utf-8'))
    meta = _metadata(source)
    if (not isinstance(data, dict) or data.get('version') != 1
            or data.get('model') != 'libvidstab_white_canvas_nearest_pixel_centres'
            or data.get('signature') != signature or data.get('output_fingerprint') != backend.file_fingerprint(output)
            or (data.get('width'), data.get('height')) != (meta['width'], meta['height'])):
        raise ValueError('Stale or incompatible geometric footprint.')
    scenes = data.get('scenes')
    if not isinstance(scenes, list) or len(scenes) != len(signature['shots']):
        raise ValueError('Invalid footprint scene ranges.')
    for scene, span in zip(scenes, signature['shots']):
        if not isinstance(scene, dict) or [scene.get('start'), scene.get('end')] != span:
            raise ValueError('Invalid footprint scene ranges.')
        if not span[0] <= frame < span[1]:
            continue
        if scene.get('error'):
            raise ValueError('Footprint was not verified: ' + str(scene['error']))
        name = scene.get('mask', '')
        folder = output.with_suffix('.geometry').resolve()
        if folder.parent != output.resolve().parent:
            raise ValueError('Footprint folder escapes the stage output directory.')
        mask = (folder / name).resolve()
        if not name or mask.parent != folder or mask.suffix != '.mkv':
            raise ValueError('Invalid footprint mask path.')
        if scene.get('mask_fingerprint') != backend.file_fingerprint(mask):
            raise ValueError('Footprint mask has changed.')
        for key in ('analysis', 'global_motion'):
            saved = (folder / scene.get(key, '')).resolve()
            if saved.parent != folder or scene.get(key + '_fingerprint') != backend.file_fingerprint(saved):
                raise ValueError('Recorded motion analysis has changed.')
        counts = scene.get('valid_pixels')
        pixels = meta['width'] * meta['height']
        if (not isinstance(counts, list) or len(counts) != span[1] - span[0]
                or any(type(n) is not int or not 0 <= n <= pixels for n in counts)):
            raise ValueError('Invalid footprint pixel counts.')
        zoom = scene.get('zoom')
        if type(zoom) not in (int, float) or not math.isfinite(zoom) or not 0 <= zoom <= 20:
            raise ValueError('Invalid footprint zoom.')
        mask_meta = _metadata(mask)
        if (mask_meta['frames'] != len(counts) or any(mask_meta[key] != meta[key] for key in ('width', 'height', 'fps_fraction'))):
            raise ValueError('Footprint mask timeline differs from the stage input.')
        valid = counts[frame - span[0]] / pixels
        image = _decode(mask, frame - span[0])
        _prune()
        return {'frame': frame, 'image': image, 'valid_output_pct': valid * 100,
            'missing_output_pct': (1 - valid) * 100,
            'cropped_source_area_estimate_pct': max(0, min(100, (1 - valid * (1 - zoom / 100) ** 2) * 100)),
            'note': 'White: source-supported output pixel centres. Black: outside the source. Nearest-sampled geometric estimate, not bicubic kernel support or restoration quality. Source-area estimate uses the libvidstab planar zoom determinant; boundary quantization is resolution-dependent.'}
    raise ValueError('Footprint frame is not covered by a scene.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--variants', required=True)
    parser.add_argument('--profile', default='high')
    args = parser.parse_args()
    rows, values = json.loads(args.variants), json.loads(args.settings)
    validate_variants(rows)
    if not isinstance(values, dict):
        raise ValueError('Expected stabilization settings.')
    jobs = [arguments(args.source, variant_values(values, row['method']), args.profile) for row in rows]
    jobs = list({str(backend.default_output(resolve_video_source(args.source), job)): job for job in jobs}.values())
    for index, job in enumerate(jobs, 1):
        print(f'Stabilization batch {index}/{len(jobs)}: {job.zoom}% additional zoom', flush=True)
        backend.main_with_args(job)


if __name__ == '__main__':
    main()
