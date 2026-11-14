"""Opt-in native-resolution mask/repair stage. Reading never starts inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import ROOT, SCRIPTS
from .paths import ensure_inherited_permissions, rel, resolve, safe_is_file, staging_directory
from .dust_jobs import candidate_lock, check_cancel

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import cv2
import numpy as np
from common import find_ffmpeg
from cleanup_video import ai_scratch_mask, run_propainter_chunk
from film_defect_detectors import DEFAULT_REGISTRY as DETECTOR_REGISTRY, split_proposal_components, _split_temporal_mask_components
from film_defect_masks import temporal_defect_mask
from film_scratch_ridge import detect_sequence

_DEFAULT_TEMPORAL_FN = temporal_defect_mask
_DEFAULT_RIDGE_FN = detect_sequence

VERSION = 3


class MaskItem(tuple):
    def __new__(cls, image, mask, reason, scratch_mask=None, dust_mask=None):
        instance = super().__new__(cls, (image, mask, reason))
        instance.scratch_mask = scratch_mask if scratch_mask is not None else mask
        instance.dust_mask = dust_mask if dust_mask is not None else np.zeros_like(mask)
        return instance


def detector_limitations(det_name):
    target = 'dart_harness' if det_name == 'dart' else det_name
    det = DETECTOR_REGISTRY.get(target, commercial_only=False) if DETECTOR_REGISTRY.has(target, commercial_only=False) else None
    limits = ['BGR8 analysis, native geometry; no historical ground truth or accuracy score.']
    if det_name == 'ridge':
        limits.append('Temporal Ridge detects narrow vertical scratches only; dust and blotches are unsupported. Windowed/cut-abstaining Ridge is not the PR24 pilot.')
    elif det_name == 'legacy':
        limits.append('ARP legacy scratch mask is single-frame spatial only; no temporal motion tracking. Scratch channel only.')
    elif det_name == 'consensus_5f':
        limits.append('Consensus 5F requires 5 continuous frames (radius 2); boundary frames abstain with zero masks.')
    elif det_name == 'dis_flow':
        limits.append('DIS Flow uses 3-frame OpenCV DIS optical flow for fast boundary tracking.')
    elif det_name in ('temporal', 'farneback_3f'):
        limits.append('Farneback 3F uses 3-frame dense optical flow consensus; boundary frames abstain.')
    elif det_name == 'dart':
        limits.append('DART full frames are diagnostic; only the mask-gated composite can be adopted.')
    return limits
DETECTORS = tuple(
    dict.fromkeys(
        (
            'legacy',
            'temporal',
            'ridge',
            *DETECTOR_REGISTRY.list_available(commercial_only=True, include_aliases=True),
            'dart',
        )
    )
)
REPAIRS = ('telea', 'propainter', 'dart')


def available_detectors(commercial_only=True):
    """Return structured metadata and schemas for registered defect detectors."""
    return DETECTOR_REGISTRY.describe_all(commercial_only=commercial_only)


def detector_metadata(detector_name):
    """Return registry metadata for `detector_name` (or fallback descriptor)."""
    target = 'dart_harness' if detector_name == 'dart' else detector_name
    if DETECTOR_REGISTRY.has(target, commercial_only=False):
        return DETECTOR_REGISTRY.get(target, commercial_only=False).info()
    return {'name': str(detector_name), 'commercial_cleared': detector_name != 'dart', 'schema': {}}


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def options(values):
    preset_name = str(values.get('preset', '')).strip().lower()
    from film_defect_detectors import BENCHMARK_PRESETS, get_preset, resolve_preset
    preset_obj = BENCHMARK_PRESETS.get(preset_name) if preset_name in BENCHMARK_PRESETS and preset_name != 'custom' else None

    detector = values.get('detector', preset_obj.detector if preset_obj else 'temporal')
    repair = values.get('repair', preset_obj.repair if preset_obj else 'telea')
    supported = detector in DETECTORS or DETECTOR_REGISTRY.has(detector, commercial_only=True)
    if not supported or repair not in REPAIRS:
        raise ValueError('Choose a supported detector and repair.')
    if repair == 'dart' and detector not in ('dart', 'dart_harness'):
        raise ValueError('DART repair requires the DART defect detector.')
    if 'repair' not in values and not preset_obj and detector in ('dart', 'dart_harness'):
        raise ValueError('DART requires an explicit repair selection: telea, propainter, or dart.')
    result = {'detector': detector, 'repair': repair}
    source_mode = values.get('analysis_source', 'stage')
    if source_mode not in ('stage', 'original'):
        raise ValueError('Choose stage or original analysis input.')
    if source_mode == 'original':
        result['analysis_source'] = source_mode
    version = values.get('mask_version', '')
    if version:
        if not isinstance(version, str) or not re.fullmatch(r'[0-9a-f]{32}', version):
            raise ValueError('Invalid Dust mask version.')
        result['mask_version'] = version
    elif version != '':
        raise ValueError('Invalid Dust mask version.')

    sens_def = preset_obj.sensitivity if preset_obj else .65
    rad_def = preset_obj.radius if preset_obj else 3
    dil_def = preset_obj.dilate if preset_obj else (1 if detector in ('temporal', 'farneback_3f') else 0)
    thresh_def = preset_obj.threshold if preset_obj else 160

    for key, default, lo, hi in (('sensitivity', sens_def, 0, 1), ('radius', rad_def, 1, 9), ('dilate', dil_def, 0, 6), ('threshold', thresh_def, 1, 255)):
        raw = values.get(key, default)
        n = float(raw)
        if isinstance(raw, bool) or not math.isfinite(n) or not lo <= n <= hi or key != 'sensitivity' and n != int(n):
            raise ValueError(f'Invalid {key}.')
        result[key] = n if key == 'sensitivity' else int(n)

    actual_preset, _ = resolve_preset({**result, 'preset': preset_name or ''})
    result['preset'] = actual_preset

    for key in ('research_confirmed', 'license_confirmed'):
        result[key] = str(values.get(key, 'false')).lower()
    for key in ('dart_repo', 'dart_python'):
        result[key] = str(values.get(key, ''))
    if detector == 'dart':
        from .dart_runtime import installation
        result['dart_repo'], result['dart_python'] = installation(result,ROOT)
    return result


def authorize(values, guard=False, kind='repair'):
    o = options(values)
    if kind == 'repair' and o.get('analysis_source') == 'original':
        raise ValueError('Original input is analysis-only. Choose Stage input for repair and adoption.')
    if o['detector'] in ('ridge', 'dart') and o['research_confirmed'] != 'true':
        raise ValueError('Confirm experimental use; authentic damage accuracy is unvalidated.')
    if o['detector'] == 'dart' or kind == 'repair' and o['repair'] in ('propainter', 'dart'):
        if guard:
            raise ValueError('Commercial model guard blocks DART (unresolved RTN rights) and ProPainter (non-commercial).')
        if o['license_confirmed'] != 'true':
            raise ValueError('Acknowledge model licence restrictions before inference.')
    if guard and o['detector'] != 'dart' and DETECTOR_REGISTRY.has(o['detector'], commercial_only=False):
        DETECTOR_REGISTRY.get(o['detector'], commercial_only=True)
    return o


def identity(source, values):
    try:
        resolved = resolve(str(source)).resolve()
    except OSError:
        resolved = resolve(str(source))
    stat = None
    try:
        if resolved.is_file():
            stat = resolved.stat()
    except OSError:
        stat = None
    return {'version': VERSION, 'source': str(resolved), 'size': stat.st_size if stat else None,
            'mtime_ns': stat.st_mtime_ns if stat else None, 'options': options(values)}


def folder_for(source, values):
    key = hashlib.sha256(json.dumps(identity(source, values), sort_keys=True).encode()).hexdigest()[:24]
    return ROOT / 'intermediate' / 'dust_scratches' / key


def output_for(source, values):
    return rel(folder_for(source, values) / 'repaired.mkv') if source else ''


def metadata(source):
    ffprobe = str(Path(find_ffmpeg()).with_name('ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'))
    d = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,pix_fmt,color_transfer,color_space,color_range,color_primaries', '-of', 'json', str(source)], text=True))['streams'][0]
    from fractions import Fraction
    fps = Fraction(d['avg_frame_rate'])
    if fps <= 0 or int(d['nb_read_frames']) < 1:
        raise ValueError('Invalid source cadence/frame count.')
    return {'width': int(d['width']), 'height': int(d['height']), 'frames': int(d['nb_read_frames']), 'fps': str(fps), 'pixel_format': d['pix_fmt'], 'transfer': d.get('color_transfer', 'unknown'),
            **{key:d.get(key, 'unknown') for key in ('color_space','color_range','color_primaries')}}


def validate_repair_timeline(source, meta):
    from fractions import Fraction
    ffprobe = str(Path(find_ffmpeg()).with_name('ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'))
    proc = subprocess.Popen([ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'csv=p=0', str(source)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    count, fps = 0, float(Fraction(meta['fps']))
    try:
        for line in proc.stdout:
            value = line.strip().split(',')[0]
            if not value:
                continue
            if abs(float(value) - count/fps) > .002:
                raise ValueError('Repair requires a constant-rate, zero-origin video timeline; VFR/offset inputs are not silently retimed.')
            count += 1
        if proc.wait() or count != meta['frames']:
            raise ValueError('Incomplete source timestamp coverage.')
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError('Cannot decode video.')
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            yield image
    finally:
        cap.release()


class Writer:
    def __init__(self, path, meta):
        self.path, self.meta, self.count = path, meta, 0
        self.process = subprocess.Popen([find_ffmpeg(), '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f"{meta['width']}x{meta['height']}", '-r', meta['fps'], '-i', 'pipe:0',
            '-an', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(path)], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def write(self, image):
        if image.dtype != np.uint8 or image.shape != (self.meta['height'], self.meta['width'], 3):
            raise ValueError('Frame geometry/type mismatch.')
        self.process.stdin.write(image.tobytes())
        self.count += 1

    def close(self):
        try:
            self.process.stdin.close()
        finally:
            code = self.process.wait()
        if code or self.count != self.meta['frames']:
            raise ValueError('Incomplete lossless output.')


def masks_for(source, o):
    """Chunked defect detection stream preserving frame count across all detectors.

    Windowed temporal detectors require temporal_radius context frames on each side.
    Boundary frames and cut boundaries abstain and are explicitly reported.
    Spatial detectors require zero context and detect all frames without abstention.
    """
    det_name = o['detector']
    strategy = DETECTOR_REGISTRY.get(det_name, commercial_only=True) if DETECTOR_REGISTRY.has(det_name, commercial_only=True) else None
    radius = int(strategy.temporal_radius) if strategy is not None else 1
    pad = max(1, radius) if radius > 0 else 1
    is_spatial = (det_name == 'legacy') or (strategy is not None and strategy.temporal_radius == 0)

    block = []
    overlap = False
    previous = None

    def emit(items, left_context=False, following=False):
        if not items:
            return []
        images = [x[0] for x in items]
        n_images = len(images)
        min_context = max(2, 2 * radius + 1)

        pred_masks = None
        pred_scratch = None
        pred_dust = None
        if not is_spatial and strategy is not None and n_images >= min_context:
            if det_name == 'ridge':
                if detect_sequence is not _DEFAULT_RIDGE_FN:
                    pred_masks = detect_sequence(images, o['sensitivity'])
                    pred_scratch = pred_masks
                    pred_dust = [np.zeros(f.shape[:2], np.uint8) for f in images]
                else:
                    pred_res = strategy.detect(images, {'sensitivity': o['sensitivity']})
                    pred_masks = pred_res.mask
                    pred_scratch = pred_res.scratch_mask
                    pred_dust = pred_res.dust_mask
            elif det_name not in ('legacy', 'temporal', 'farneback_3f'):
                pred_res = strategy.detect(images, {'sensitivity': o['sensitivity']})
                pred_masks = pred_res.mask
                pred_scratch = pred_res.scratch_mask
                pred_dust = pred_res.dust_mask

        result = []
        first = pad if left_context else 0
        last = n_images - pad if following else n_images

        for i in range(first, last):
            image, cut_before = items[i]
            reason = ''
            h, w = image.shape[:2]
            if det_name == 'legacy':
                mask = ai_scratch_mask(image, o['sensitivity'])
                scratch_mask = mask.copy()
                dust_mask = np.zeros((h, w), np.uint8)
            elif is_spatial and strategy is not None:
                res = strategy.detect([image], {'sensitivity': o['sensitivity']})
                mask = res.mask[0]
                scratch_mask = res.scratch_mask[0]
                dust_mask = res.dust_mask[0]
            elif i < pad or i >= n_images - pad:
                mask = np.zeros((h, w), np.uint8)
                scratch_mask = np.zeros((h, w), np.uint8)
                dust_mask = np.zeros((h, w), np.uint8)
                reason = 'source_or_cut_boundary: insufficient context'
            elif det_name in ('temporal', 'farneback_3f'):
                if temporal_defect_mask is not _DEFAULT_TEMPORAL_FN or strategy is None:
                    mask = temporal_defect_mask(images[i-1], image, images[i+1], o['sensitivity'])
                    scratch_mask, dust_mask = _split_temporal_mask_components(mask)
                else:
                    res = strategy.detect([images[i-1], image, images[i+1]], {'sensitivity': o['sensitivity']})
                    mask = res.mask[1]
                    scratch_mask = res.scratch_mask[1]
                    dust_mask = res.dust_mask[1]
            elif pred_masks is not None:
                mask = pred_masks[i]
                scratch_mask = pred_scratch[i]
                dust_mask = pred_dust[i]
            else:
                mask = np.zeros((h, w), np.uint8)
                scratch_mask = np.zeros((h, w), np.uint8)
                dust_mask = np.zeros((h, w), np.uint8)
                reason = 'source_or_cut_boundary: insufficient context'

            if cut_before:
                reason = 'source_cut_boundary' + (': insufficient context' if not is_spatial else ': legacy spatial mask')
            result.append(MaskItem(image, mask, reason, scratch_mask=scratch_mask, dust_mask=dust_mask))
        return result

    for image in frames(source):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cut = previous is not None and float(np.median(cv2.absdiff(gray, previous))) >= 22
        if cut:
            yield from emit(block, overlap, False)
            block = []
            overlap = False
        block.append((image, cut))
        limit = 24 + 2 * pad if overlap else 24 + pad
        if len(block) == limit:
            yield from emit(block, overlap, True)
            block = block[-(2 * pad):]
            overlap = True
        previous = gray
    yield from emit(block, overlap, False)


def manifest(folder, source, values, kind):
    report = json.loads((folder / 'report.json').read_text(encoding='utf-8'))
    if report['identity'] != identity(source, values) or report['source_sha256'] != digest(resolve(str(source))):
        raise ValueError('Stale source, settings or detector version.')
    required = ['mask.mkv', 'effective_mask.mkv', 'input.mkv', 'measurements.csv']
    if kind == 'repair':
        if not report.get('repair_complete'):
            raise FileNotFoundError('No completely published Dust repair. Run or resume repair first.')
        required.extend(('repaired.mkv', 'repair_measurements.csv'))
        if report.get('native_composition'):
            required.append('native_measurements.csv')
    check_names = list(required)
    for name in ('scratch_mask.mkv', 'dust_mask.mkv'):
        if name in report.get('hashes', {}):
            check_names.append(name)
    if any(report.get('hashes', {}).get(name) != digest(folder / name) for name in check_names):
        raise ValueError('Missing or changed mask/result.')
    return report


def analysis_source(app, values):
    if options(values).get('analysis_source') == 'original':
        from .media import pipeline_source_text
        return pipeline_source_text(getattr(app, 'settings', {}))
    return app.stabilized_source_for_downstream()


def validate_analysis_context(app, values, source_expected=None, project_expected=None):
    source = analysis_source(app, values)
    if (source_expected is not None and source_expected != source or
            project_expected is not None and project_expected != str(getattr(app, 'project_path', '') or '')):
        raise ValueError('Dust project/input changed; reload before inspection.')


def status(app, values, source_expected=None, project_expected=None):
    validate_analysis_context(app, values, source_expected, project_expected)
    source = analysis_source(app, values)
    folder = folder_for(source, values)
    try:
        folder, report = saved_mask(source, values)
        exact = report['identity']['options'] == options(values)
        repair_error = ''
        complete = report['repair_complete'] and exact
        if report['repair_complete'] and exact:
            try:
                manifest(folder, source, values, 'repair')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                complete = False
                repair_error = str(exc)
        files = {name: rel(folder / name) for name in report['hashes']}
        if not complete:
            files = {name: path for name,path in files.items() if name in ('input.mkv','mask.mkv','effective_mask.mkv','scratch_mask.mkv','dust_mask.mkv','measurements.csv')}
        return {'source': source, 'files': files, 'frames': report['metadata']['frames'], 'repair_complete': complete, 'repair_error': repair_error, 'mask_values': report['identity']['options'], 'limitations': report['limitations'], 'detector_info': detector_metadata(report['identity']['options']['detector'])}
    except (OSError, ValueError, KeyError, TypeError):
        det_name = values.get('detector', 'temporal')
        result = {'source': source, 'files': {}, 'repair_complete': False, 'frames': 0, 'detector_info': detector_metadata(det_name)}
        if values.get('detector') == 'dart':
            opts = options(values)
            result['installation'] = {'available':all(safe_is_file(path) for path in (
                Path(opts['dart_repo'])/'scripts/inference.py',Path(opts['dart_python']),
                Path(opts['dart_repo'])/'pretrained_models/dart.safetensors')),
                'message':'Set up DART with install_windows.bat -DartOnly. ARP uses tools/dart automatically; no ComfyUI node is required.'}
        return result


def saved_mask(source, values):
    """Read an existing matching detector artifact; never infer or rewrite it."""
    opts = options(values)
    folder = folder_for(source, opts)
    try:
        return folder, manifest(folder, source, opts, 'mask')
    except (FileNotFoundError, OSError):
        pass
    if opts['detector'] != 'dart':
        for path in sorted((ROOT/'intermediate/dust_scratches').glob('*/report.json')):
            if path.parent.name.startswith('.'):
                continue
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                other = report['identity']['options']
                if other.get('mask_version', '') != opts.get('mask_version', ''):
                    continue
                if other.get('analysis_source') != opts.get('analysis_source'):
                    continue
                same_det = (other['detector'] == opts['detector']) or (
                    DETECTOR_REGISTRY.has(other['detector'], commercial_only=False)
                    and DETECTOR_REGISTRY.has(opts['detector'], commercial_only=False)
                    and DETECTOR_REGISTRY.get(other['detector'], commercial_only=False).name == DETECTOR_REGISTRY.get(opts['detector'], commercial_only=False).name
                )
                if not same_det:
                    continue
                if any(other[key] != opts[key] for key in ('sensitivity','dilate','threshold')):
                    continue
                return path.parent, manifest(path.parent, source, other, 'mask')
            except (OSError, ValueError, KeyError, TypeError):
                continue
    raise FileNotFoundError('No saved mask for this input and detector settings. Calculate a mask first.')


def matrix_state(app, original=False):
    source = analysis_source(app, {'analysis_source':'original'} if original else {})
    command = getattr(getattr(app,'process',None),'args',[])
    stop_requested = isinstance(command,list) and '--cancel-file' in command and Path(command[command.index('--cancel-file')+1]).exists()
    rows = []
    if source and safe_is_file(source):
        for path in sorted((ROOT/'intermediate/dust_scratches').glob('*/report.json')):
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                values = report['identity']['options']
                if (values.get('analysis_source') == 'original') != original:
                    continue
                if path.parent != folder_for(source, values):
                    continue
                report = manifest(path.parent, source, values, 'mask')
                complete = bool(report['repair_complete'])
                repair_error = ''
                if complete:
                    try:
                        manifest(path.parent, source, values, 'repair')
                    except (OSError, ValueError, KeyError, TypeError) as exc:
                        complete = False
                        repair_error = str(exc)
                job = json.loads((path.parent/'job-state.json').read_text()) if (path.parent/'job-state.json').exists() else {}
                rows.append({'method':path.parent.name, 'preset':'settings', 'values':values,
                    'label':f"{values['detector']} / {values['repair']} · {path.parent.name[:6]}", 'frames':report['metadata']['frames'],
                    'output':rel(path.parent/'repaired.mkv') if complete else '', 'repair_complete':complete,
                    'revision':digest(path), 'mask_sha256':report['hashes']['effective_mask.mkv'],
                    'job':job, 'repair_error':repair_error, 'adopted':complete and app.settings.get('dust',{}).get('adopted_report') == digest(path)
                    and options(app.settings.get('dust',{})) == values})
            except (OSError, ValueError, KeyError, TypeError):
                continue
    result = {'source':source,'source_revision':json.dumps(identity(source,{}),sort_keys=True) if source else '',
        'variants':rows,'ready':sum(r['repair_complete'] for r in rows),'running':app.running_stage_key=='dust','stop_requested':bool(stop_requested),
        'available_detectors':available_detectors(commercial_only=True)}
    if not original:
        result['original'] = matrix_state(app, True)
    return result


def review_pair(app, candidate):
    source = app.stabilized_source_for_downstream()
    output = resolve(candidate).resolve()
    if output.name != 'repaired.mkv' or output.parent.parent != (ROOT/'intermediate/dust_scratches').resolve():
        raise ValueError('Only completed mask-gated Dust repairs may be compared.')
    report = json.loads((output.parent/'report.json').read_text(encoding='utf-8'))
    values = report['identity']['options']
    if output.parent != folder_for(source,values):
        raise ValueError('Dust candidate belongs to a different source.')
    manifest(output.parent,source,values,'repair')
    return resolve(source),output


def inspect(app, values, frame, source_expected=None, project_expected=None):
    validate_analysis_context(app, values, source_expected, project_expected)
    if type(frame) is not int or frame < 0:
        raise ValueError('Choose a nonnegative frame index.')
    source = analysis_source(app, values)
    folder, report = saved_mask(source, values)
    complete = report['repair_complete'] and report['identity']['options'] == options(values)
    repair_error = ''
    if complete:
        try:
            manifest(folder, source, values, 'repair')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            complete = False
            repair_error = str(exc)
    if frame >= report['metadata']['frames']:
        raise ValueError('Frame outside the analyzed source.')
    from .frame_inspector import _decode, _prune
    names = ['input.mkv', 'mask.mkv', 'effective_mask.mkv']
    for extra in ('scratch_mask.mkv', 'dust_mask.mkv'):
        if extra in report.get('hashes', {}):
            names.append(extra)
    if complete:
        names.append('repaired.mkv')
    if 'dart_full.mkv' in report['hashes']:
        if digest(folder / 'dart_full.mkv') != report['hashes']['dart_full.mkv']:
            raise ValueError('DART diagnostic output changed.')
        names.append('dart_full.mkv')
    images = {name: _decode(folder / name, frame) for name in names}
    if 'scratch_mask.mkv' not in images and 'mask.mkv' in images:
        m_file = resolve(images['mask.mkv'])
        if m_file.is_file():
            raw_m = cv2.imread(str(m_file), cv2.IMREAD_GRAYSCALE)
            if raw_m is not None:
                s_m, d_m = split_proposal_components(raw_m)
                s_path = m_file.with_name(m_file.stem + '_scratch.png')
                d_path = m_file.with_name(m_file.stem + '_dust.png')
                if not s_path.is_file():
                    cv2.imwrite(str(s_path), s_m)
                if not d_path.is_file():
                    cv2.imwrite(str(d_path), d_m)
                images['scratch_mask.mkv'] = rel(s_path)
                images['dust_mask.mkv'] = rel(d_path)
    if complete:
        before = cv2.imread(str(resolve(images['input.mkv'])))
        after = cv2.imread(str(resolve(images['repaired.mkv'])))
        diff = cv2.absdiff(before, after)
        path = resolve(images['repaired.mkv']).with_name(resolve(images['repaired.mkv']).stem + '_dust_diff.png')
        if not cv2.imwrite(str(path), diff):
            raise ValueError('Could not save exact RGB difference.')
        images['difference'] = rel(path)
    counts = {}
    for name in ('measurements.csv', 'repair_measurements.csv', 'native_measurements.csv'):
        if not complete and name != 'measurements.csv':
            continue
        if name not in report['hashes']:
            continue
        with (folder / name).open(encoding='utf-8-sig', newline='') as f:
            counts.update(next(row for row in csv.DictReader(f) if int(row['frame']) == frame))
    _prune()
    return {'frame': frame, 'frames': report['metadata']['frames'], 'images': images, 'measurements': counts, 'repair_error':repair_error,
            'native_composition': report.get('native_composition'), 'detector_info': detector_metadata(report['identity']['options']['detector'])}


def run(source, values, kind='mask', guard=False, comfy_url='http://127.0.0.1:8188', comfy_output='', comfy_dir=None, cancel_file=None):
    if kind not in ('mask', 'repair'):
        raise ValueError('Choose mask analysis or repair.')
    o = authorize(values, guard, kind)
    with candidate_lock(folder_for(source, o)):
        check_cancel(cancel_file)
        try:
            report = _run(source, o, kind, guard, comfy_url, comfy_output, comfy_dir, cancel_file)
        except Exception as exc:
            target = folder_for(source,o)
            if (target/'report.json').exists():
                publish_json(target/'job-state.json',{'state':'stopped' if cancel_file and Path(cancel_file).exists() else 'failed','message':str(exc)})
            raise
        check_cancel(cancel_file)
        return report


def _run(source, values, kind, guard, comfy_url, comfy_output, comfy_dir, cancel_file):
    o = authorize(values, guard, kind)
    source = resolve(str(source)).resolve()
    target = folder_for(source, o)
    if target.exists():
        try:
            return manifest(target, source, o, kind)
        except FileNotFoundError:
            if kind == 'mask' or not (target / 'report.json').exists():
                raise
    if kind == 'mask':
        pending = list(target.parent.glob(f'.{target.name}.analysis-*'))
        if len(pending) > 1:
            raise ValueError('Multiple pending Dust analyses exist; inspect them before retrying. Nothing was overwritten.')
        if pending:
            return finish_analysis(pending[0], target, source, o, cancel_file)
    if kind == 'repair':
        if not target.exists():
            saved_mask(source, o)  # Refuse missing analysis; the mask pass only copies validated frozen bytes.
            _run(source, o, 'mask', guard, comfy_url, comfy_output, comfy_dir, cancel_file)
        report = manifest(target, source, o, 'mask')
        if (target / '.repair-ready').exists():
            return finish_repair(target, source, o, cancel_file)
        if (target / 'repaired.mkv').exists():
            raise ValueError('Refusing to overwrite an existing repair.')
        meta = report['metadata']
        from dust_native_composite import FORMATS, compose
        native = meta['pixel_format'] in FORMATS
        if not native and meta['pixel_format'] not in ('yuv420p', 'yuv422p', 'yuv444p', 'rgb24', 'bgr24', 'bgr0', 'rgb0', 'gray'):
            raise ValueError('Unsupported repair pixel format; no silent bit-depth reduction.')
        if meta.get('transfer') in ('smpte2084', 'arib-std-b67'):
            raise ValueError('HDR repair is not supported; no silent tone mapping.')
        validate_repair_timeline(source, meta)
        with staging_directory(prefix='.repair-', dir=target) as tmp:
            tmp = Path(tmp)
            model = None
            if o['repair'] == 'dart':
                model = frames(target / 'dart_full.mkv')
            elif o['repair'] == 'propainter':
                from dust_model_preflight import inspect_propainter, require_ready
                # The installed node can download absent checkpoints inside its
                # inference call. Refuse BEFORE queueing, rather than hiding that.
                report['model_preflight'] = require_ready(inspect_propainter(comfy_dir or Path(comfy_output).parent, comfy_url))
                # Bounded model jobs; no hidden mask regeneration or extra dilation.
                model = propainter_frames(target, tmp, meta, comfy_url, comfy_output, report, cancel_file)
            writer = Writer(tmp / 'repaired.mkv', meta)
            measurements = []
            try:
                for index, (image, m) in enumerate(zip(frames(target / 'input.mkv'), frames(target / 'effective_mask.mkv'), strict=True)):
                    check_cancel(cancel_file)
                    mask = m[:, :, 0] > 127
                    repaired = next(model) if model is not None else cv2.inpaint(image, mask.astype(np.uint8)*255, o['radius'], cv2.INPAINT_TELEA)
                    if repaired.shape != image.shape:
                        raise ValueError('Model output geometry mismatch.')
                    result = image.copy()
                    result[mask] = repaired[mask]
                    changed = np.any(result != image, axis=2)
                    measurements.append({'frame': index, 'changed_inside_pixels': int(np.count_nonzero(changed & mask)), 'changed_outside_pixels': int(np.count_nonzero(changed & ~mask))})
                    writer.write(result)
                if model is not None and next(model, None) is not None:
                    raise ValueError('Model output has extra frames.')
            finally:
                try:
                    if model is not None:
                        model.close()
                finally:
                    close_writers([writer])
            if native:
                native_rows = compose(find_ffmpeg(), source, tmp/'repaired.mkv', target/'effective_mask.mkv', tmp/'with_audio.mkv', meta)
                write_csv(tmp/'native_measurements.csv', native_rows)
                # Display metrics describe the actual decoded master, not the
                # pre-composition model copy. Native sample preservation is separate.
                measurements = []
                for index, (image, result, m) in enumerate(zip(frames(target/'input.mkv'), frames(tmp/'with_audio.mkv'), frames(target/'effective_mask.mkv'), strict=True)):
                    support = m[:, :, 0] > 127
                    changed = np.any(image != result, axis=2)
                    measurements.append({'frame':index, 'changed_inside_pixels':int((changed & support).sum()),
                        'changed_outside_pixels':int((changed & ~support).sum())})
            else:
                # Keep source audio; never silently replace an existing soundtrack.
                subprocess.run([find_ffmpeg(), '-v', 'error', '-n', '-i', str(tmp/'repaired.mkv'), '-i', str(source),
                    '-map', '0:v:0', '-map', '1:a?', '-c', 'copy', str(tmp/'with_audio.mkv')], check=True)
            check_cancel(cancel_file)
            (tmp / 'repaired.mkv').unlink()  # disposable BGR8 composite; master is already muxed
            (tmp / 'with_audio.mkv').rename(tmp / 'repaired.mkv')
            write_csv(tmp / 'repair_measurements.csv', measurements)
            if native:
                report['native_composition'] = {'pixel_format': meta['pixel_format'], 'model_domain':'BGR8',
                    'display_metrics':'BGR8 decoded master; chroma reconstruction may change displayed pixels outside the luma mask',
                    'support':'per-plane binary support; chroma uses any selected pixel in its subsampled block',
                    'encoded_raster_verified': True}
                report['hashes']['native_measurements.csv'] = digest(tmp/'native_measurements.csv')
            report['hashes'].update({name: digest(tmp / name) for name in ('repaired.mkv', 'repair_measurements.csv')})
            report['repair_complete'] = True
            publish_report(tmp, report)
            ready_dir = target / '.repair-ready'
            tmp.rename(ready_dir)
            ensure_inherited_permissions(ready_dir)
        return finish_repair(target, source, o, cancel_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError('Refusing to overwrite prior analysis.')
    meta = metadata(source)
    initial_identity, initial_hash = identity(source, o), digest(source)
    # A different repairer reuses the exact frozen CPU mask bytes, not another
    # detector invocation. Repair-specific configuration gets a separate result.
    if o['detector'] != 'dart':
        keys = ('detector', 'sensitivity', 'dilate', 'threshold')
        for other in sorted(target.parent.glob('*/report.json')):
            if other.parent.name.startswith('.'):
                continue
            try:
                old = json.loads(other.read_text(encoding='utf-8'))
                old_options = old['identity']['options']
                if old_options.get('mask_version', '') != o.get('mask_version', ''):
                    continue
                if old_options.get('analysis_source') != o.get('analysis_source'):
                    continue
                if old['identity']['source'] != str(source) or any(old_options[k] != o[k] for k in keys):
                    continue
                old = manifest(other.parent, source, old_options, 'mask')
            except (OSError, ValueError, KeyError, TypeError):
                continue
            with staging_directory(prefix=f'.{target.name}.analysis-', dir=target.parent, delete=False) as tmp:
                stage = Path(tmp)
                names = ['input.mkv', 'mask.mkv', 'effective_mask.mkv', 'measurements.csv']
                for name in ('scratch_mask.mkv', 'dust_mask.mkv'):
                    if (other.parent / name).exists():
                        names.append(name)
                for name in names:
                    shutil.copyfile(other.parent / name, stage / name)
                report = {**old, 'identity': initial_identity, 'repair_complete': False,
                    'mask_origin': str(other.parent), 'hashes': {name: digest(stage / name) for name in names}}
                publish_report(stage, report)
                return finish_analysis(stage, target, source, o, cancel_file)
    if meta['width']*meta['height']*26 > 96_000_000:
        raise ValueError('Native-resolution analysis exceeds the experimental window memory limit. No resizing applied.')
    # Retain staging on failure: Windows locks must not trigger destructive temp cleanup.
    with staging_directory(prefix=f'.{target.name}.analysis-', dir=target.parent, delete=False) as tmp:
        stage = Path(tmp)
        dart_masks, dart_images = None, None
        if o['detector'] == 'dart':
            if meta['frames'] > 96:
                raise ValueError('DART research adapter currently requires a reviewed single-shot source of 2–96 frames; full-film inference is not validated.')
            from .dart_runtime import require_installation
            require_installation(o, ROOT)
            previous = None
            for image in frames(source):
                check_cancel(cancel_file)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if previous is not None and float(np.median(cv2.absdiff(gray, previous))) >= 22:
                    raise ValueError('DART input contains a suspected cut/exposure discontinuity; use a reviewed continuous shot.')
                previous = gray
            from run_dart_research import build_command, validate_geometry
            from dust_model_preflight import inspect_dart, require_ready
            validate_geometry(meta['width'], meta['height'])
            repo = Path(o['dart_repo']).resolve()
            weights = repo / 'pretrained_models' / 'dart.safetensors'
            preflight = require_ready(inspect_dart(repo, Path(o['dart_python'])))
            external = stage / 'dart'
            subprocess.run(build_command(Path(o['dart_python']).resolve(), repo, weights, source, external, meta['frames'], 'auto', 7, 3), cwd=repo, check=True)
            check_cancel(cancel_file)
            from assess_dart_exports import read_mask, read_restored
            expected = {f'frame_{i:06d}.png' for i in range(meta['frames'])}
            for name in ('frames', 'masks'):
                if {p.name for p in (external / name).glob('*.png')} != expected:
                    raise ValueError('DART frame names/count do not match the source.')
            dart_masks, dart_images = external / 'masks', external / 'frames'
        writers = {
            'input.mkv': Writer(stage / 'input.mkv', meta),
            'mask.mkv': Writer(stage / 'mask.mkv', meta),
            'effective_mask.mkv': Writer(stage / 'effective_mask.mkv', meta),
            'scratch_mask.mkv': Writer(stage / 'scratch_mask.mkv', meta),
            'dust_mask.mkv': Writer(stage / 'dust_mask.mkv', meta),
        }
        if dart_images:
            writers['dart_full.mkv'] = Writer(stage / 'dart_full.mkv', meta)
        rows = []
        iterator = masks_for(source, o) if not dart_masks else ((f, None, '') for f in frames(source))
        try:
            for index, item in enumerate(iterator):
                check_cancel(cancel_file)
                image, mask, reason = item
                s_mask = getattr(item, 'scratch_mask', None)
                d_mask = getattr(item, 'dust_mask', None)
                if s_mask is None:
                    s_mask = mask
                if d_mask is None:
                    d_mask = np.zeros_like(mask)
                if dart_masks:
                    mask = read_mask(dart_masks / f'frame_{index:06d}.png', image.shape)
                    writers['dart_full.mkv'].write(read_restored(dart_images / f'frame_{index:06d}.png', image.shape))
                    raw_bin = (mask >= o['threshold']).astype(np.uint8) * 255
                    s_mask, d_mask = split_proposal_components(raw_bin)
                    categorized = (s_mask > 0) | (d_mask > 0)
                    uncategorized = (raw_bin > 0) & ~categorized
                    if np.any(uncategorized):
                        d_mask = np.where(uncategorized, np.uint8(255), d_mask)
                effective = (mask >= (o['threshold'] if dart_masks else 128)).astype(np.uint8)*255
                if o['dilate']:
                    effective = cv2.dilate(effective, np.ones((3, 3), np.uint8), iterations=o['dilate'])
                writers['input.mkv'].write(image)
                writers['mask.mkv'].write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
                writers['effective_mask.mkv'].write(cv2.cvtColor(effective, cv2.COLOR_GRAY2BGR))
                writers['scratch_mask.mkv'].write(cv2.cvtColor(s_mask, cv2.COLOR_GRAY2BGR))
                writers['dust_mask.mkv'].write(cv2.cvtColor(d_mask, cv2.COLOR_GRAY2BGR))
                rows.append({
                    'frame': index,
                    'mask_pixels': int(np.count_nonzero(mask >= (o['threshold'] if dart_masks else 128))),
                    'repair_pixels': int(np.count_nonzero(effective)),
                    'scratch_pixels': int(np.count_nonzero(s_mask >= 128)),
                    'dust_pixels': int(np.count_nonzero(d_mask >= 128)),
                    'abstention': reason,
                    'note': 'Empty proposal may also mean detector safety rejection; not proof of intact film.'
                })
        finally:
            close_writers(writers.values())
        write_csv(stage / 'measurements.csv', rows)
        if identity(source, o) != initial_identity or digest(source) != initial_hash:
            raise ValueError('Source changed during inference; refusing publication.')
        report = {
            'identity': initial_identity,
            'source_sha256': initial_hash,
            'metadata': meta,
            'repair_complete': False,
            'hashes': {name: digest(stage / name) for name in (*writers, 'measurements.csv')},
            'limitations': detector_limitations(o['detector']),
        }
        if o['detector'] == 'dart':
            report['model_preflight'] = preflight
            report['weights_sha256'] = 'd78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3'
        publish_report(stage, report)
        check_cancel(cancel_file)
        return finish_analysis(stage, target, source, o, cancel_file)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish_report(folder, report):
    partial = folder / 'report.partial.json'
    partial.write_text(json.dumps(report, indent=2), encoding='utf-8')
    partial.replace(folder / 'report.json')


def finish_analysis(stage, target, source, values, cancel_file=None):
    """Retry publication only; never regenerate or adopt a pending analysis."""
    try:
        report = manifest(stage, source, values, 'mask')
        if report.get('repair_complete'):
            raise ValueError('Pending analysis unexpectedly contains a completed repair.')
        if values.get('detector') == 'dart':
            if digest(stage/'dart_full.mkv') != report['hashes'].get('dart_full.mkv'):
                raise ValueError('Pending DART diagnostic output changed.')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'Pending Dust analysis is incomplete, changed or inaccessible: {stage}. '
                         'Retained for inspection; no new inference or overwrite was attempted.') from exc
    for attempt in range(5):
        check_cancel(cancel_file)
        if target.exists():
            raise FileExistsError(f'Dust destination already exists; pending analysis retained: {stage}')
        try:
            stage.rename(target)
            ensure_inherited_permissions(target)
            return report
        except PermissionError as exc:
            if attempt == 4:
                raise PermissionError(f'Dust publication blocked by file access: {stage}. '
                                      'Completed analysis retained. Release the lock and retry the same settings; '
                                      'no new inference is required for a valid pending analysis.') from exc
            time.sleep(.1 * (attempt + 1))


def close_writers(writers):
    failed = sys.exc_info()[0] is not None
    errors = []
    for writer in writers:
        try:
            writer.close()
        except (OSError, ValueError) as exc:
            errors.append(exc)
    if errors and not failed:
        raise errors[0]


def finish_repair(target, source, values, cancel_file=None):
    """Report is the commit marker; retry can finish a crash between file moves."""
    ready = target / '.repair-ready'
    report = json.loads((ready / 'report.json').read_text(encoding='utf-8'))
    mask = manifest(target, source, values, 'mask')
    if not report.get('repair_complete') or any(report[k] != mask[k] for k in ('identity', 'source_sha256', 'metadata')):
        raise ValueError('Stale or incomplete pending repair.')
    for name, sha in mask['hashes'].items():
        if report['hashes'].get(name) != sha or digest(target/name) != sha:
            raise ValueError('Pending repair does not match the frozen masks.')
    names = ['repaired.mkv', 'repair_measurements.csv']
    if report.get('native_composition'):
        names.append('native_measurements.csv')
    for name in ('scratch_mask.mkv', 'dust_mask.mkv'):
        if (ready / name).exists():
            names.append(name)
    for name in names:
        path = target/name if (target/name).exists() else ready/name
        if digest(path) != report['hashes'].get(name):
            raise ValueError('Missing or changed pending repair artifact.')
    master = target/'repaired.mkv' if (target/'repaired.mkv').exists() else ready/'repaired.mkv'
    actual = metadata(master)
    if any(actual[k] != report['metadata'][k] for k in ('frames', 'width', 'height', 'fps')):
        raise ValueError('Final repair timeline/geometry mismatch.')
    check_cancel(cancel_file)
    for name in names:
        if not (target/name).exists():
            (ready/name).rename(target/name)
            ensure_inherited_permissions(target/name)
    publish_report(target, report)
    ensure_inherited_permissions(target)
    return report


def propainter_frames(target, tmp, meta, url, output, report, cancel_file=None):
    from comfy_api import ensure_node_types
    from cleanup_video import AI_DESCRATCH_NODES
    from fractions import Fraction
    ensure_node_types(url, AI_DESCRATCH_NODES)
    args = argparse.Namespace(scratch_mask_dilate=0, ai_chunk_frames=25, intermediate_profile='lossless', poll_seconds=2)
    with (target/'measurements.csv').open(encoding='utf-8-sig', newline='') as f:
        cuts = [int(r['frame']) for r in csv.DictReader(f) if r['abstention'].startswith('source_cut_boundary')]
    boundaries = sorted({0, *cuts, meta['frames']})
    ranges = [(start, min(start+24, b)) for a,b in zip(boundaries,boundaries[1:]) for start in range(a,b,24)]
    if any(end-start < 2 for start,end in ranges):
        raise ValueError('ProPainter requires at least two frames per section, including cut/tail sections. No padding or altered boundaries applied.')
    for start, end in ranges:
        check_cancel(cancel_file)
        chunk = target / '.model-chunks' / str(start)
        chunk.mkdir(parents=True, exist_ok=True)
        signature = {'identity': report['identity'], 'source_sha256': report['source_sha256'],
            'hashes': {k: report['hashes'][k] for k in ('input.mkv', 'effective_mask.mkv', 'measurements.csv')},
            'preflight': report['model_preflight'], 'start': start, 'end': end}
        complete = chunk / 'complete.json'
        if complete.exists():
            saved = json.loads(complete.read_text(encoding='utf-8'))
            if saved['file'] not in ('model.mkv','model.mp4','model.mov','model.webm'):
                raise ValueError('Invalid ProPainter checkpoint path.')
            result = chunk / saved['file']
            if saved['signature'] != signature or digest(result) != saved['sha256']:
                raise ValueError('ProPainter checkpoint changed; refusing reuse or overwrite.')
            yield from frames(result)
            continue
        job = chunk / 'job.json'
        pending = json.loads(job.read_text(encoding='utf-8')) if job.exists() else None
        if pending and pending['signature'] != signature:
            raise ValueError('ProPainter pending job belongs to different inputs/runtime.')
        if pending:
            if not pending.get('prompt_id'):
                raise ValueError('ProPainter submission outcome is unknown. Inspect ComfyUI before retry; no duplicate was queued.')
            from comfy_api import http_json, completed_prompt_entry
            if completed_prompt_entry(http_json('GET', url.rstrip('/')+'/history/'+pending['prompt_id']), pending['prompt_id']) is None:
                raise ValueError('Saved ProPainter prompt is not completed/in history. No duplicate was queued.')
        args.propainter_prompt_id = pending['prompt_id'] if pending else None
        def queued(prompt_id):
            publish_json(job, {'signature': signature, 'prompt_id': prompt_id})
        args.propainter_queued = queued
        inputs = chunk/'inputs'
        if not inputs.exists():
            if pending:
                raise ValueError('Pending ProPainter section inputs are missing.')
            with staging_directory(prefix='.inputs-', dir=chunk) as stage:
                stage = Path(stage)
                for name, source in (('input.mkv',target/'input.mkv'),('mask.mkv',target/'effective_mask.mkv')):
                    subprocess.run([find_ffmpeg(), '-v', 'error', '-n', '-i', str(source),
                        '-vf', f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                        '-an', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(stage/name)], check=True)
                publish_json(stage/'report.json', {'signature': signature,
                    'hashes': {name:digest(stage/name) for name in ('input.mkv','mask.mkv')}})
                stage.rename(inputs)
                ensure_inherited_permissions(inputs)
        saved_inputs = json.loads((inputs/'report.json').read_text(encoding='utf-8'))
        if saved_inputs['signature'] != signature or any(digest(inputs/name) != saved_inputs['hashes'][name] for name in ('input.mkv','mask.mkv')):
            raise ValueError('ProPainter section inputs changed; refusing submission.')
        video, mask = inputs/'input.mkv', inputs/'mask.mkv'
        check_cancel(cancel_file)
        if not pending:
            publish_json(job, {'signature': signature, 'prompt_id': None})
        result = run_propainter_chunk(url, Path(output), video, mask, chunk/'model.mkv', meta['width'], meta['height'], float(Fraction(meta['fps'])), args, f'arp_dust_{target.name}_{start}')
        actual = metadata(result)
        if actual['frames'] != end-start or any(actual[k] != meta[k] for k in ('width', 'height', 'fps')):
            raise ValueError('ProPainter chunk frame count/geometry/cadence mismatch.')
        publish_json(complete, {'signature': signature, 'file': result.name, 'sha256': digest(result)})
        check_cancel(cancel_file)
        yield from frames(result)


def publish_json(path, data):
    partial = path.with_suffix('.partial.json')
    partial.write_text(json.dumps(data, indent=2), encoding='utf-8')
    partial.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--settings', required=True)
    p.add_argument('--kind', choices=('mask', 'repair'), default='repair')
    p.add_argument('--guard', action='store_true')
    p.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    p.add_argument('--comfy-output', default='')
    p.add_argument('--comfy-dir', type=Path)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--cancel-file', type=Path)
    a = p.parse_args()
    values = json.loads(a.settings)
    authorize(values, a.guard, a.kind)
    if not a.dry_run:
        run(a.source, values, a.kind, a.guard, a.comfy_url, a.comfy_output, a.comfy_dir, a.cancel_file)
    print('Dust & Scratches complete. Proposals and pixel changes are not quality scores.', flush=True)


if __name__ == '__main__':
    main()
