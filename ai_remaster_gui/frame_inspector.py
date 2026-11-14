"""Frame-index-exact inspection of existing ARP outputs. No model execution.

The caller supplies stage names, never arbitrary filesystem paths.
A frame is decoded using FFmpeg's n-based select filter without approximate seeking.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
import os
import subprocess
import threading
import time
from fractions import Fraction
from pathlib import Path

from .config import ROOT
from .media import pipeline_source_text
from .paths import resolve, resolve_video_source

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
