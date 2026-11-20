"""Frame-index-exact inspection of existing ARP outputs. No model execution.

The caller supplies stage names, never arbitrary filesystem paths.
A frame is decoded using FFmpeg's n-based select filter without approximate seeking.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from functools import lru_cache
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .config import ROOT, SCRIPTS
from .media import pipeline_source_text
from .paths import rel, resolve, resolve_video_source

EXTENSIONS = {'.mkv', '.mp4', '.webm', '.mov', '.m4v', '.avi'}
CACHE = ROOT / '.cache' / 'frame_inspector'
_METADATA_LOCK = threading.Lock()


def _executable(name: str) -> str:
    from shutil import which
    local = ROOT / '.cache' / 'tools' / 'ffmpeg' / (name + ('.exe' if os.name == 'nt' else ''))
    if local.is_file():
        return str(local)
    fallback = Path(__file__).resolve().parents[1] / '.cache' / 'tools' / 'ffmpeg' / (name + ('.exe' if os.name == 'nt' else ''))
    if fallback.is_file():
        return str(fallback)
    found = which(name)
    if found:
        return found
    raise RuntimeError(f'{name} not available. Install local FFmpeg tools.')


def _metadata(path: Path) -> dict:
    stat = path.stat()
    # Simultaneous visible variants must not each scan the same long source.
    with _METADATA_LOCK:
        return _metadata_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=16)
def _metadata_cached(path_text: str, _size: int, _mtime_ns: int) -> dict:
    path = Path(path_text)
    # Untrusted media may be malformed. Never rely on guessed frame counts for frame-exact review.
    proc = subprocess.run([
        _executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,avg_frame_rate',
        '-of', 'json', str(path)
    ], capture_output=True, text=True, timeout=20, check=True)
    streams = json.loads(proc.stdout).get('streams', [])
    if len(streams) != 1:
        raise ValueError('Expected one primary video stream.')
    d = streams[0]
    fps = Fraction(d.get('avg_frame_rate', '0/1'))
    r_fps = Fraction(d.get('r_frame_rate', '0/1'))
    if fps <= 0 or fps != r_fps:
        raise ValueError('Variable-frame-rate or ambiguous FPS video; frame-exact comparison needs a CFR export.')
    try:
        count = int(d.get('nb_frames') or '0')
    except ValueError:
        count = 0
    if count <= 0:
        # Some MKV/FFV1 streams omit nb_frames. Decode to count instead of guessing.
        probe = subprocess.run([
            _executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
            '-count_frames', '-show_entries', 'stream=nb_read_frames',
            '-of', 'default=noprint_wrappers=1:nokey=1', str(path)
        ], capture_output=True, text=True, timeout=600, check=True)
        count = int(probe.stdout.strip())
    if count < 1 or count > 2147483647:
        raise ValueError('Frame count is outside the decoder-supported integer range.')
    return {'width': int(d['width']), 'height': int(d['height']),
            'fps': float(fps), 'fps_fraction': str(fps), 'frames': count}


def _video_path(path_text: str) -> Path:
    p = resolve_video_source(path_text) if path_text else None
    if not p or not p.is_file() or p.suffix.lower() not in EXTENSIONS:
        raise FileNotFoundError('The selected video no longer exists or is not supported.')
    return p


def _sources(app, stage: str) -> tuple[Path, Path]:
    if stage == 'cleanup':
        info = app.cleanup_comparison_state()
        if info.get('exists') != 'true':
            raise FileNotFoundError('Run Clean Up first and reload the comparison.')
        before, after = info['source'], info['output']
    elif stage == 'stabilize':
        info = app.stabilization_comparison_state()
        if info.get('exists') != 'true':
            raise FileNotFoundError('Run Stabilization first and reload the comparison.')
        before = info.get('master_source') or info['source']
        after = info.get('master_output') or info['output']
    elif stage == 'detail':
        info = app.detail_comparison_state()
        if info.get('exists') != 'true':
            raise FileNotFoundError('Run Detail Enhancement first and reload the comparison.')
        before, after = info['source'], info['output']
    elif stage == 'denoise':
        before = pipeline_source_text(app.settings)
        after = app.denoise_output()
        if not after or not resolve(after).is_file():
            raise FileNotFoundError('Run Remove Noise first and reload the comparison.')
    else:
        raise ValueError('Unsupported stage.')
    return _video_path(before), _video_path(after)


def _decode(path: Path, frame: int) -> str:
    stamp = path.stat()
    identity = f'{path.resolve()}\n{stamp.st_size}\n{stamp.st_mtime_ns}\n{frame}'
    token = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:32]
    CACHE.mkdir(parents=True, exist_ok=True)
    output = CACHE / (token + '.png')
    if not output.is_file():
        temp = CACHE / (token + f'.{os.getpid()}.partial.png')
        try:
            # Frame N is selected by its decoded ordinal; no timestamp seeking.
            proc = subprocess.run([
                _executable('ffmpeg'), '-hide_banner', '-loglevel', 'error',
                '-nostdin', '-i', str(path),
                '-vf', f'select=eq(n\\,{frame})',
                '-frames:v', '1', '-f', 'image2', '-c:v', 'png', '-y', str(temp)
            ], capture_output=True, text=True, timeout=90)
            if proc.returncode != 0 or not temp.is_file() or temp.stat().st_size == 0:
                raise RuntimeError('FFmpeg could not decode frame: ' + (proc.stderr or '')[:450])
            os.replace(temp, output)
        finally:
            temp.unlink(missing_ok=True)
    try:
        os.utime(output, None)
    except OSError:
        pass
    return str(output.relative_to(ROOT)).replace('\\', '/')


def _prune() -> None:
    # Bound cached lossless stills. Errors while pruning never fail a valid inspection.
    try:
        paths = sorted(CACHE.glob('*.png'), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in paths[100:]:
            p.unlink(missing_ok=True)
    except OSError:
        pass


def inspection(app, stage: str, frame: int) -> dict:
    if frame < 0:
        raise ValueError('Frame index cannot be negative.')
    if stage == 'cleanup_source':
        # When the selected method has not been rendered, show original frames
        # without inventing a processed result or exposing arbitrary paths.
        comparison = app.cleanup_comparison_state()
        if comparison.get('source_exists') != 'true':
            raise FileNotFoundError('Choose a valid Clean Up input first.')
        src = _video_path(comparison['source'])
        meta = _metadata(src)
        if frame >= meta['frames']:
            raise IndexError(f'Frame {frame} exceeds total {meta["frames"]}.')
        original = _decode(src, frame)
        _prune()
        return {
            'stage': stage, 'frame': frame, 'frames': meta['frames'],
            'fps': meta['fps'], 'width': meta['width'],
            'height': meta['height'], 'before': original,
            'after': None, 'exact': True,
            'note': 'Source-only exact frame: selected cleanup variant not rendered yet.',
        }
    src, dst = _sources(app, stage)
    source_meta, output_meta = _metadata(src), _metadata(dst)
    for field in ('width', 'height', 'fps_fraction', 'frames'):
        if source_meta[field] != output_meta[field]:
            raise ValueError(f'Original and result have incompatible {field}; refuse unaligned comparison.')
    if frame >= source_meta['frames']:
        raise IndexError(f'Frame {frame} exceeds total {source_meta["frames"]}.')
    original = _decode(src, frame)
    processed = _decode(dst, frame)
    _prune()
    return {
        'stage': stage, 'frame': frame, 'frames': source_meta['frames'],
        'fps': source_meta['fps'], 'width': source_meta['width'],
        'height': source_meta['height'],
        'before': original, 'after': processed,
        'exact': True, 'note': 'Original and processed images use decoded frame ordinals.'
    }


def _media_url(rel_path: str) -> str:
    return f'/media?path={quote(rel_path, safe="")}' if rel_path else ''


def _resolve_cached_path(path_text: str, root: Path | str | None = None) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    for base in (ROOT, Path(root) if root is not None else None):
        if base is not None:
            cand = base / p
            if cand.exists():
                return cand
    return resolve(path_text)


def _rel_cached_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace('\\', '/')
    except ValueError:
        return rel(path)


def inspect_dust(
    folder: Path | str,
    report: Mapping[str, Any],
    frame: int,
    *,
    complete: bool = False,
    repair_error: str = '',
    base_folder: Path | str | None = None,
    base_report: Mapping[str, Any] | None = None,
    detector_info: Mapping[str, Any] | None = None,
    counts: Mapping[str, Any] | None = None,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Decode exact frame overlays for Dust & Scratches, including E2 vetoed and base vs composed layers."""
    if type(frame) is not int or frame < 0:
        raise ValueError('Choose a nonnegative frame index.')
    total_frames = int(report['metadata']['frames'])
    if frame >= total_frames:
        raise ValueError('Frame outside the analyzed source.')

    folder_path = Path(folder)
    hashes = report.get('hashes', {})
    names = ['input.mkv', 'mask.mkv', 'effective_mask.mkv']
    for extra in ('scratch_mask.mkv', 'dust_mask.mkv', 'uncertain_mask.mkv', 'vetoed_mask.mkv'):
        if extra in hashes or (extra == 'vetoed_mask.mkv' and (folder_path / extra).is_file()):
            names.append(extra)
    if complete:
        names.append('repaired.mkv')
    if 'dart_full.mkv' in hashes:
        names.append('dart_full.mkv')

    images = {name: _decode(folder_path / name, frame) for name in names}

    m_rel = images.get('mask.mkv', '')
    m_file = _resolve_cached_path(m_rel, root) if m_rel else None

    if ('scratch_mask.mkv' not in images or 'uncertain_mask.mkv' not in images) and m_file is not None and m_file.is_file():
        import cv2
        import numpy as np

        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        from film_defect_detectors import split_proposal_components

        if 'scratch_mask.mkv' not in images:
            raw_m = cv2.imread(str(m_file), cv2.IMREAD_GRAYSCALE)
            if raw_m is not None:
                s_m, d_m = split_proposal_components(raw_m)
                s_path = m_file.with_name(m_file.stem + '_scratch.png')
                d_path = m_file.with_name(m_file.stem + '_dust.png')
                if not s_path.is_file():
                    cv2.imwrite(str(s_path), s_m)
                if not d_path.is_file():
                    cv2.imwrite(str(d_path), d_m)
                images['scratch_mask.mkv'] = _rel_cached_path(s_path)
                images['dust_mask.mkv'] = _rel_cached_path(d_path)

        if 'uncertain_mask.mkv' not in images:
            u_path = m_file.with_name(m_file.stem + '_uncertain.png')
            if not u_path.is_file():
                raw_m = cv2.imread(str(m_file), cv2.IMREAD_GRAYSCALE)
                if raw_m is not None:
                    s_file = _resolve_cached_path(images.get('scratch_mask.mkv', ''), root) if images.get('scratch_mask.mkv') else None
                    d_file = _resolve_cached_path(images.get('dust_mask.mkv', ''), root) if images.get('dust_mask.mkv') else None
                    s_m = cv2.imread(str(s_file), cv2.IMREAD_GRAYSCALE) if s_file is not None and s_file.is_file() else None
                    d_m = cv2.imread(str(d_file), cv2.IMREAD_GRAYSCALE) if d_file is not None and d_file.is_file() else None
                    if s_m is not None and d_m is not None:
                        accounted = cv2.bitwise_or(s_m, d_m)
                        u_m = cv2.bitwise_and(raw_m, cv2.bitwise_not(accounted))
                    else:
                        u_m = np.zeros_like(raw_m)
                    cv2.imwrite(str(u_path), u_m)
            if u_path.is_file():
                images['uncertain_mask.mkv'] = _rel_cached_path(u_path)

    if 'vetoed_mask.mkv' in images and m_file is not None and m_file.is_file():
        v_file = _resolve_cached_path(images['vetoed_mask.mkv'], root)
        if v_file.is_file():
            v_alias = m_file.with_name(m_file.stem + '_vetoed.png')
            if not v_alias.is_file():
                tmp_v = v_alias.with_name(f'{v_alias.name}.{os.getpid()}.partial.png')
                try:
                    shutil.copyfile(v_file, tmp_v)
                    os.replace(tmp_v, v_alias)
                finally:
                    tmp_v.unlink(missing_ok=True)

    if complete:
        import cv2

        before_file = _resolve_cached_path(images['input.mkv'], root)
        after_file = _resolve_cached_path(images['repaired.mkv'], root)
        if before_file.is_file() and after_file.is_file():
            before = cv2.imread(str(before_file))
            after = cv2.imread(str(after_file))
            if before is None or after is None:
                raise ValueError('Could not decode input/repaired frame for difference image.')
            diff = cv2.absdiff(before, after)
            diff_path = after_file.with_name(after_file.stem + '_dust_diff.png')
            if not cv2.imwrite(str(diff_path), diff):
                raise ValueError('Could not save exact RGB difference.')
            images['difference'] = _rel_cached_path(diff_path)

    is_composed = bool(report.get('edit_signature'))
    base_images: dict[str, str] = {}
    if is_composed and base_folder is not None and base_report is not None:
        base_path = Path(base_folder)
        base_hashes = base_report.get('hashes', {})
        base_names = ['mask.mkv', 'effective_mask.mkv']
        for extra in ('scratch_mask.mkv', 'dust_mask.mkv', 'uncertain_mask.mkv', 'vetoed_mask.mkv'):
            if extra in base_hashes or (extra == 'vetoed_mask.mkv' and (base_path / extra).is_file()):
                base_names.append(extra)
        base_images = {
            name: _decode(base_path / name, frame)
            for name in base_names
            if (base_path / name).is_file() or name in base_hashes
        }

    urls = {name: _media_url(path) for name, path in images.items()}
    base_urls = {name: _media_url(path) for name, path in base_images.items()}
    proposal_keys = (
        'mask.mkv',
        'effective_mask.mkv',
        'scratch_mask.mkv',
        'dust_mask.mkv',
        'uncertain_mask.mkv',
        'vetoed_mask.mkv',
    )
    auto_proposal_images = (
        dict(base_images) if base_images else {k: v for k, v in images.items() if k in proposal_keys}
    )
    auto_proposal_urls = {name: _media_url(path) for name, path in auto_proposal_images.items()}

    _prune()
    return {
        'frame': frame,
        'frames': total_frames,
        'images': images,
        'urls': urls,
        'vetoed_url': urls.get('vetoed_mask.mkv', ''),
        'base_images': base_images,
        'base_urls': base_urls,
        'auto_proposal_images': auto_proposal_images,
        'auto_proposal_urls': auto_proposal_urls,
        'is_composed': is_composed,
        'edit_signature': str(report.get('edit_signature', '') or ''),
        'timeline_id': str(report.get('timeline_id', '') or ''),
        'base_folder_key': str(report.get('base_folder_key', '') or ''),
        'measurements': dict(counts) if counts is not None else {},
        'repair_error': repair_error,
        'native_composition': report.get('native_composition'),
        'detector_info': dict(detector_info) if detector_info is not None else {},
    }
