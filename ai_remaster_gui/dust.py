"""Opt-in native-resolution mask/repair stage. Reading never starts inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .config import ROOT, SCRIPTS
from .paths import rel, resolve
from .dust_jobs import candidate_lock, check_cancel

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import cv2
import numpy as np
from common import find_ffmpeg
from cleanup_video import ai_scratch_mask, run_propainter_chunk
from film_defect_masks import temporal_defect_mask
from film_scratch_ridge import detect_sequence

VERSION = 3
DETECTORS = ('legacy', 'temporal', 'ridge', 'dart')
REPAIRS = ('telea', 'propainter', 'dart')


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def options(values):
    detector, repair = values.get('detector', 'temporal'), values.get('repair', 'telea')
    if detector not in DETECTORS or repair not in REPAIRS:
        raise ValueError('Choose a supported detector and repair.')
    if (detector == 'dart') != (repair == 'dart'):
        raise ValueError('DART jointly generates its own soft mask and repaired frames; it does not accept the selected CPU mask.')
    result = {'detector': detector, 'repair': repair}
    for key, default, lo, hi in (('sensitivity', .65, 0, 1), ('radius', 3, 1, 9), ('dilate', 0, 0, 6), ('threshold', 160, 1, 255)):
        raw = values.get(key, default)
        n = float(raw)
        if isinstance(raw, bool) or not math.isfinite(n) or not lo <= n <= hi or key != 'sensitivity' and n != int(n):
            raise ValueError(f'Invalid {key}.')
        result[key] = n if key == 'sensitivity' else int(n)
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
    if o['detector'] in ('ridge', 'dart') and o['research_confirmed'] != 'true':
        raise ValueError('Confirm experimental use; authentic damage accuracy is unvalidated.')
    if o['detector'] == 'dart' or kind == 'repair' and o['repair'] == 'propainter':
        if guard:
            raise ValueError('Commercial model guard blocks DART (unresolved RTN rights) and ProPainter (non-commercial).')
        if o['license_confirmed'] != 'true':
            raise ValueError('Acknowledge model licence restrictions before inference.')
    return o


def identity(source, values):
    source = resolve(str(source)).resolve()
    stat = source.stat() if source.is_file() else None
    return {'version': VERSION, 'source': str(source), 'size': stat.st_size if stat else None,
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
    """24 output frames plus one neighbour at each side; no context crosses cuts.

    shortcut: windowed Ridge is not the PR24 pilot; validate separately before
    recommending it. Boundary frames abstain and are explicitly reported.
    """
    block = []
    overlap = False
    previous = None
    def emit(items, left_context=False, following=False):
        if not items:
            return []
        images = [x[0] for x in items]
        if o['detector'] == 'ridge' and len(images) >= 3:
            predicted = detect_sequence(images, o['sensitivity'])
        else:
            predicted = [np.zeros(f.shape[:2], np.uint8) for f in images]
        result = []
        first = 1 if left_context else 0
        last = len(items)-1 if following else len(items)
        for i in range(first, last):
            image, cut_before = items[i]
            reason = ''
            if o['detector'] == 'legacy':
                mask = ai_scratch_mask(image, o['sensitivity'])
            elif i == 0 or i == len(images)-1:
                mask, reason = np.zeros(image.shape[:2], np.uint8), 'source_or_cut_boundary: insufficient context'
            elif o['detector'] == 'temporal':
                mask = temporal_defect_mask(images[i-1], image, images[i+1], o['sensitivity'])
            else:
                mask = predicted[i]
            if cut_before:
                reason = 'source_cut_boundary' + (': insufficient context' if o['detector'] != 'legacy' else ': legacy spatial mask')
            result.append((image, mask, reason))
        return result
    for image in frames(source):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cut = previous is not None and float(np.median(cv2.absdiff(gray, previous))) >= 22
        if cut:
            yield from emit(block, overlap)
            block = []
            overlap = False
        block.append((image, cut))
        if len(block) == (26 if overlap else 25):
            yield from emit(block, overlap, True)
            block = block[-2:]
            overlap = True
        previous = gray
    yield from emit(block, overlap)


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
    if any(report.get('hashes', {}).get(name) != digest(folder / name) for name in required):
        raise ValueError('Missing or changed mask/result.')
    return report


def status(app, values):
    source = app.stabilized_source_for_downstream()
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
            files = {name: path for name,path in files.items() if name in ('input.mkv','mask.mkv','effective_mask.mkv','measurements.csv')}
        return {'source': source, 'files': files, 'frames': report['metadata']['frames'], 'repair_complete': complete, 'repair_error': repair_error, 'mask_values': report['identity']['options'], 'limitations': report['limitations']}
    except (OSError, ValueError, KeyError, TypeError):
        result = {'source': source, 'files': {}, 'repair_complete': False, 'frames': 0}
        if values.get('detector') == 'dart':
            opts = options(values)
            result['installation'] = {'available':all(path.is_file() for path in (
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
    except FileNotFoundError:
        pass
    if opts['detector'] != 'dart':
        for path in sorted((ROOT/'intermediate/dust_scratches').glob('*/report.json')):
            if path.parent.name.startswith('.'):
                continue
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                other = report['identity']['options']
                if any(other[key] != opts[key] for key in ('detector','sensitivity','dilate','threshold')):
                    continue
                return path.parent, manifest(path.parent, source, other, 'mask')
            except (OSError, ValueError, KeyError, TypeError):
                continue
    raise FileNotFoundError('No saved mask for this input and detector settings. Calculate a mask first.')


def matrix_state(app):
    source = app.stabilized_source_for_downstream()
    command = getattr(getattr(app,'process',None),'args',[])
    stop_requested = isinstance(command,list) and '--cancel-file' in command and Path(command[command.index('--cancel-file')+1]).exists()
    rows = []
    if source and resolve(source).is_file():
        for path in sorted((ROOT/'intermediate/dust_scratches').glob('*/report.json')):
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                values = report['identity']['options']
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
    return {'source':source,'source_revision':json.dumps(identity(source,{}),sort_keys=True) if source else '',
        'variants':rows,'ready':sum(r['repair_complete'] for r in rows),'running':app.running_stage_key=='dust','stop_requested':bool(stop_requested)}


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


def inspect(app, values, frame):
    if type(frame) is not int or frame < 0:
        raise ValueError('Choose a nonnegative frame index.')
    source = app.stabilized_source_for_downstream()
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
    if complete:
        names.append('repaired.mkv')
    if 'dart_full.mkv' in report['hashes']:
        if digest(folder / 'dart_full.mkv') != report['hashes']['dart_full.mkv']:
            raise ValueError('DART diagnostic output changed.')
        names.append('dart_full.mkv')
    images = {name: _decode(folder / name, frame) for name in names}
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
            'native_composition': report.get('native_composition')}


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
        with tempfile.TemporaryDirectory(prefix='.repair-', dir=target) as tmp:
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
            tmp.rename(target / '.repair-ready')
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
                if old['identity']['source'] != str(source) or any(old_options[k] != o[k] for k in keys):
                    continue
                old = manifest(other.parent, source, old_options, 'mask')
            except (OSError, ValueError, KeyError, TypeError):
                continue
            with tempfile.TemporaryDirectory(prefix=f'.{target.name}.analysis-', dir=target.parent, delete=False) as tmp:
                stage = Path(tmp)
                names = ('input.mkv', 'mask.mkv', 'effective_mask.mkv', 'measurements.csv')
                for name in names:
                    shutil.copyfile(other.parent / name, stage / name)
                report = {**old, 'identity': initial_identity, 'repair_complete': False,
                    'mask_origin': str(other.parent), 'hashes': {name: digest(stage / name) for name in names}}
                publish_report(stage, report)
                return finish_analysis(stage, target, source, o, cancel_file)
    if meta['width']*meta['height']*26 > 96_000_000:
        raise ValueError('Native-resolution analysis exceeds the experimental window memory limit. No resizing applied.')
    # Retain staging on failure: Windows locks must not trigger destructive temp cleanup.
    with tempfile.TemporaryDirectory(prefix=f'.{target.name}.analysis-', dir=target.parent, delete=False) as tmp:
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
        writers = {name: Writer(stage / name, meta) for name in ('input.mkv', 'mask.mkv', 'effective_mask.mkv')}
        if dart_images:
            writers['dart_full.mkv'] = Writer(stage / 'dart_full.mkv', meta)
        rows = []
        iterator = masks_for(source, o) if not dart_masks else ((f, None, '') for f in frames(source))
        try:
            for index, (image, mask, reason) in enumerate(iterator):
                check_cancel(cancel_file)
                if dart_masks:
                    mask = read_mask(dart_masks / f'frame_{index:06d}.png', image.shape)
                    writers['dart_full.mkv'].write(read_restored(dart_images / f'frame_{index:06d}.png', image.shape))
                effective = (mask >= (o['threshold'] if dart_masks else 128)).astype(np.uint8)*255
                if o['dilate']:
                    effective = cv2.dilate(effective, np.ones((3, 3), np.uint8), iterations=o['dilate'])
                for name, f in (('input.mkv', image), ('mask.mkv', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)), ('effective_mask.mkv', cv2.cvtColor(effective, cv2.COLOR_GRAY2BGR))):
                    writers[name].write(f)
                rows.append({'frame': index, 'mask_pixels': int(np.count_nonzero(mask >= 128)), 'repair_pixels': int(np.count_nonzero(effective)), 'abstention': reason,
                    'note': 'Empty proposal may also mean detector safety rejection; not proof of intact film.'})
        finally:
            close_writers(writers.values())
        write_csv(stage / 'measurements.csv', rows)
        if identity(source, o) != initial_identity or digest(source) != initial_hash:
            raise ValueError('Source changed during inference; refusing publication.')
        report = {'identity': initial_identity, 'source_sha256': initial_hash, 'metadata': meta,
            'repair_complete': False, 'hashes': {name: digest(stage / name) for name in (*writers, 'measurements.csv')},
            'limitations': ['BGR8 analysis, native geometry; no historical ground truth or accuracy score.', 'Windowed/cut-abstaining Ridge is not the original PR24 pilot.', 'DART full frames are diagnostic; only the mask-gated composite can be adopted.']}
        if o['detector'] == 'dart':
            report['model_preflight'] = preflight
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
    publish_report(target, report)
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
            with tempfile.TemporaryDirectory(prefix='.inputs-', dir=chunk) as stage:
                stage = Path(stage)
                for name, source in (('input.mkv',target/'input.mkv'),('mask.mkv',target/'effective_mask.mkv')):
                    subprocess.run([find_ffmpeg(), '-v', 'error', '-n', '-i', str(source),
                        '-vf', f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
                        '-an', '-c:v', 'ffv1', '-pix_fmt', 'bgr0', str(stage/name)], check=True)
                publish_json(stage/'report.json', {'signature': signature,
                    'hashes': {name:digest(stage/name) for name in ('input.mkv','mask.mkv')}})
                stage.rename(inputs)
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
