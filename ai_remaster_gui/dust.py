"""Opt-in native-resolution mask/repair stage. Reading never starts inference."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from functools import lru_cache
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from .config import ROOT, SCRIPTS
from .dust_e2_adapter import (
    DETECTOR_ARCHITECTURE as E2_DETECTOR_ARCHITECTURE,
    DETECTOR_NAME as E2_DETECTOR_NAME,
    DETECTOR_VERSION as E2_DETECTOR_VERSION,
    E2_DETECTOR_INFO,
    E2_PRESET_INFO,
    is_research_confirmed,
    run_experimental_e2_sequence,
)
from .dust_mask_edits import (
    BIT_PROTECT_INTACT,
    DUST_MASKS_REL_DIR,
    DustMaskEditStore,
    EditRevisionConflictError,
    EditStoreIntegrityError,
    EditTimelineMismatchError,
    GROUND_TRUTH_STATUS,
    build_edit_provenance_summary,
    build_timeline_binding,
    load_verified_store,
    resolve_effective_channels,
    store_compatibility_status,
)
from .paths import ensure_inherited_permissions, rel, resolve, safe_is_file, staging_directory
from .dust_jobs import candidate_lock, check_cancel

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import cv2
import numpy as np
from common import find_ffmpeg
from cleanup_video import ai_scratch_mask, run_propainter_chunk
from film_defect_detectors import (
    DEFAULT_REGISTRY as DETECTOR_REGISTRY,
    BENCHMARK_PRESETS,
    DefectMaskResult,
    RepairInclusionPolicy,
    resolve_preset,
    split_proposal_components,
    _split_temporal_mask_components,
    fuse_spatial_and_temporal_masks,
)
from film_defect_masks import temporal_defect_mask
from film_scratch_ridge import detect_sequence

_DEFAULT_TEMPORAL_FN = temporal_defect_mask
_DEFAULT_RIDGE_FN = detect_sequence

VERSION = 3

_THREAD_LOCK_STATE = threading.local()

_EDIT_SIGNATURE_RE = re.compile(r'^[0-9a-f]{64}:rev[1-9]\d*$')
_TIMELINE_ID_RE = re.compile(r'^[0-9a-f]{24}$')

DEFAULT_DETECTOR_PRESETS: dict[str, str] = {
    'farneback_3f': 'balanced',
    'temporal': 'balanced',
    'dis_flow': 'fast_preview',
    'consensus_5f': 'high_quality_precision',
    'ridge': 'scratch_focused',
    'legacy': 'legacy_baseline',
    'dart': 'dart_optimal_f1',
    'dart_harness': 'dart_optimal_f1',
    'bopb': 'bopb_scratch_balanced',
    'film_artefact_segmenter': 'film_artefact_balanced',
    'cnn_blotches': 'cnn_blotches_precision',
    'patchcore': 'patchcore_anomaly_research',
    'padim': 'padim_anomaly_research',
    'temporal_scratch': 'temporal_scratch_balanced',
    E2_DETECTOR_NAME: 'experimental_e2_research',
}

EXPERIMENTAL_PRESETS: dict[str, dict[str, Any]] = {
    'experimental_e2_research': dict(E2_PRESET_INFO),
}


def ensure_uint8_binary_mask(arr: Any, shape: tuple[int, int], name: str = "mask") -> np.ndarray:
    """Validate and enforce contiguous 2D uint8 binary mask contract strictly."""
    if arr is None or not isinstance(arr, np.ndarray):
        raise ValueError(f"Mask channel '{name}' must be a numpy.ndarray, got {type(arr)}")
    if arr.dtype == object or not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating) or arr.dtype == np.bool_
    ):
        raise ValueError(f"Mask channel '{name}' invalid dtype {arr.dtype}; object arrays are not supported")
    if arr.ndim != 2:
        raise ValueError(f"Mask channel '{name}' must be 2D, got shape {arr.shape}")
    if arr.shape != shape:
        raise ValueError(f"Mask channel '{name}' shape {arr.shape} does not match expected {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Mask channel '{name}' contains non-finite values (NaN/Inf)")
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8) * 255
    elif arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = (arr * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class MaskItem(tuple):
    def __new__(cls, image, mask, reason, scratch_mask=None, dust_mask=None, uncertain_mask=None, combined_probability=None, vetoed_mask=None):
        instance = super().__new__(cls, (image, mask, reason))
        h, w = image.shape[:2]
        instance.scratch_mask = scratch_mask if scratch_mask is not None else (mask.copy() if mask is not None else np.zeros((h, w), dtype=np.uint8))
        instance.dust_mask = dust_mask if dust_mask is not None else np.zeros((h, w), dtype=np.uint8)
        instance.uncertain_mask = uncertain_mask if uncertain_mask is not None else np.zeros((h, w), dtype=np.uint8)
        instance.vetoed_mask = vetoed_mask if vetoed_mask is not None else np.zeros((h, w), dtype=np.uint8)
        instance.combined_probability = combined_probability
        return instance


CLASSICAL_DETECTORS = {'legacy', 'temporal', 'farneback_3f', 'dis_flow', 'ridge', 'consensus_5f', 'temporal_scratch'}
NEURAL_BASELINE_DETECTORS = {'dart', 'dart_harness'}
EXPERIMENTAL_PROTOTYPE_DETECTORS = {E2_DETECTOR_NAME}


def detector_tier(det_name: str) -> str:
    target = 'dart_harness' if det_name == 'dart' else det_name
    if target in CLASSICAL_DETECTORS:
        return 'Classical Detectors'
    if target in NEURAL_BASELINE_DETECTORS:
        return 'Neural Baseline (DART)'
    if target in EXPERIMENTAL_PROTOTYPE_DETECTORS:
        return 'Experimental Prototypes'
    return 'Neural Candidates'


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
    elif det_name == 'bopb':
        limits.append('BOPB scratch detector is research-only under Microsoft Academic Non-Commercial License.')
    elif det_name == 'film_artefact_segmenter':
        limits.append('Film Artefact Segmenter provides multi-task scratch and dust segmentation (commercial-safe).')
    elif det_name == 'cnn_blotches':
        limits.append('CNN Blotches & Scratches provides high-precision facial and edge-preserving defect detection.')
    elif det_name in ('patchcore', 'padim'):
        limits.append(
            'EXPERIMENTAL / NOT RECOMMENDED: Distribution anomaly detectors have near-100% false-positive '
            'rates on authentic photographic film grain and unaligned scene textures. Safety guard blocks '
            'automatic repair when mask coverage exceeds 30%.'
        )
    elif det_name == 'correl_net':
        limits.append('Correl-Net architecture implemented; requires multi-frame temporal training before activation.')
    elif det_name == E2_DETECTOR_NAME:
        limits.append(
            'EXPERIMENTAL (OPT-IN): Motion-aware 3-state scene-vs-film scratch prototype '
            '(high_confidence_damage, uncertain_damage_candidate, likely_intact_scene_structure -> vetoed_mask.mkv). '
            'Requires explicit research_confirmed opt-in; dust channel unsupported; unvalidated on independent ground truth.'
        )
    return limits
DETECTORS = tuple(
    dict.fromkeys(
        (
            'legacy',
            'temporal',
            'ridge',
            *DETECTOR_REGISTRY.list_available(commercial_only=False, include_aliases=True),
            'dart',
            E2_DETECTOR_NAME,
        )
    )
)
REPAIRS = ('telea', 'propainter', 'dart')


def available_detectors(commercial_only=True):
    """Return structured metadata and schemas for registered defect detectors."""
    items = DETECTOR_REGISTRY.describe_all(commercial_only=commercial_only)
    for item in items:
        item['tier'] = detector_tier(item['name'])
    if (not commercial_only or E2_DETECTOR_INFO.get('commercial_cleared', True)) and not any(
        item.get('name') == E2_DETECTOR_NAME for item in items
    ):
        items.append(detector_metadata(E2_DETECTOR_NAME))
    return items


def detector_metadata(detector_name):
    """Return registry metadata for `detector_name` (or fallback descriptor)."""
    if detector_name == E2_DETECTOR_NAME:
        meta = dict(E2_DETECTOR_INFO)
        meta['tier'] = detector_tier(E2_DETECTOR_NAME)
        return meta
    target = 'dart_harness' if detector_name == 'dart' else detector_name
    if DETECTOR_REGISTRY.has(target, commercial_only=False):
        meta = dict(DETECTOR_REGISTRY.get(target, commercial_only=False).info())
        meta['tier'] = detector_tier(target)
        return meta
    return {
        'name': str(detector_name),
        'commercial_cleared': detector_name != 'dart',
        'tier': detector_tier(detector_name),
        'readiness_state': 'pretrained_operational',
        'schema': {},
    }


def detector_strategy_config(strategy, o):
    """Extract valid detector config parameters matching strategy.schema."""
    cfg = {}
    schema = getattr(strategy, 'schema', {})
    if 'sensitivity' in schema and 'sensitivity' in o:
        cfg['sensitivity'] = o['sensitivity']
    if 'threshold' in schema:
        th = o.get('threshold', 160)
        t_type = schema['threshold'].get('type')
        if t_type == 'float' and th > 1.0:
            cfg['threshold'] = float(th) / 255.0
        elif t_type == 'float':
            cfg['threshold'] = float(th)
        else:
            cfg['threshold'] = int(th)
    for k in ('scratch_threshold', 'dust_threshold'):
        if k in schema:
            cfg[k] = cfg.get('threshold', 0.5)
    for k in ('tile_size', 'overlap', 'boundary_mode', 'scales', 'multiscale', 'sigma', 'device'):
        if k in schema and k in o:
            cfg[k] = o[k]
    if 'preset' in schema and 'preset' in o:
        preset_val = o['preset']
        choices = schema['preset'].get('choices', ())
        if not choices or preset_val in choices:
            cfg['preset'] = preset_val
    return cfg


_DIGEST_LOCK = threading.Lock()
_DIGEST_CACHE_MIN_BYTES = 65536
_METADATA_LOCK = threading.Lock()


@lru_cache(maxsize=128)
def _digest_cached(path_text: str, _size: int, _mtime_ns: int, _ctime_ns: int) -> str:
    with Path(path_text).open('rb') as f:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(f, 'sha256').hexdigest()
        h = hashlib.sha256()
        while chunk := f.read(65536):
            h.update(chunk)
        return h.hexdigest()


def digest(path):
    target = Path(path)
    try:
        st = target.stat()
        resolved = str(target.resolve())
    except OSError:
        st = None
        resolved = ''
    if st is not None and st.st_size >= _DIGEST_CACHE_MIN_BYTES and resolved:
        with _DIGEST_LOCK:
            return _digest_cached(resolved, st.st_size, st.st_mtime_ns, getattr(st, 'st_ctime_ns', 0))
    with target.open('rb') as f:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(f, 'sha256').hexdigest()
        h = hashlib.sha256()
        while chunk := f.read(65536):
            h.update(chunk)
        return h.hexdigest()


def _resolve_root(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else ROOT


def _preset_entry(preset_name: str) -> dict[str, Any] | None:
    if not preset_name or preset_name == 'custom':
        return None
    if preset_name in BENCHMARK_PRESETS:
        return BENCHMARK_PRESETS[preset_name].as_dict()
    if preset_name in EXPERIMENTAL_PRESETS:
        return dict(EXPERIMENTAL_PRESETS[preset_name])
    return None


def _resolve_dust_preset(preset_name: str, result: dict[str, Any]) -> str:
    exp = EXPERIMENTAL_PRESETS.get(preset_name)
    if exp is not None:
        is_exact = (
            result.get('detector') == exp['detector']
            and abs(float(result.get('sensitivity', exp['sensitivity'])) - float(exp['sensitivity'])) < 1e-4
            and int(result.get('dilate', exp['dilate'])) == int(exp['dilate'])
            and int(result.get('threshold', exp['threshold'])) == int(exp['threshold'])
        )
        return preset_name if is_exact else 'custom'
    if result.get('detector') == E2_DETECTOR_NAME:
        exp_def = EXPERIMENTAL_PRESETS['experimental_e2_research']
        if (
            abs(float(result.get('sensitivity', 0.65)) - float(exp_def['sensitivity'])) < 1e-4
            and int(result.get('dilate', 0)) == int(exp_def['dilate'])
            and int(result.get('threshold', 160)) == int(exp_def['threshold'])
        ):
            return 'experimental_e2_research'
        return 'custom'
    actual_preset, _ = resolve_preset({**result, 'preset': preset_name or ''})
    return actual_preset


def options(values):
    preset_name = str(values.get('preset', '')).strip().lower()
    preset_obj = _preset_entry(preset_name)

    detector = values.get('detector', preset_obj['detector'] if preset_obj else 'temporal')
    repair = values.get('repair', preset_obj['repair'] if preset_obj else 'telea')
    supported = detector in DETECTORS or DETECTOR_REGISTRY.has(detector, commercial_only=False)
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

    edit_sig = values.get('edit_signature', '')
    if edit_sig:
        if not isinstance(edit_sig, str) or not _EDIT_SIGNATURE_RE.fullmatch(edit_sig):
            raise ValueError('Invalid Dust edit_signature.')
        result['edit_signature'] = edit_sig
    elif edit_sig != '':
        raise ValueError('Invalid Dust edit_signature.')

    timeline_id = values.get('timeline_id', '')
    if timeline_id:
        if not isinstance(timeline_id, str) or not _TIMELINE_ID_RE.fullmatch(timeline_id):
            raise ValueError('Invalid Dust timeline_id.')
        if result.get('edit_signature'):
            result['timeline_id'] = timeline_id
    elif timeline_id != '':
        raise ValueError('Invalid Dust timeline_id.')

    sens_def = preset_obj['sensitivity'] if preset_obj else .65
    rad_def = preset_obj['radius'] if preset_obj else 3
    dil_def = preset_obj['dilate'] if preset_obj else (1 if detector in ('temporal', 'farneback_3f') else 0)
    thresh_def = preset_obj['threshold'] if preset_obj else 160

    for key, default, lo, hi in (('sensitivity', sens_def, 0, 1), ('radius', rad_def, 1, 9), ('dilate', dil_def, 0, 6), ('threshold', thresh_def, 1, 255)):
        raw = values.get(key, default)
        n = float(raw)
        if isinstance(raw, bool) or not math.isfinite(n) or not lo <= n <= hi or key != 'sensitivity' and n != int(n):
            raise ValueError(f'Invalid {key}.')
        result[key] = n if key == 'sensitivity' else int(n)

    # Pass through any detector-specific configuration (tile_size, overlap, sigma, etc.)
    for param_key in ('tile_size', 'overlap', 'sigma', 'scratch_threshold', 'dust_threshold', 'device', 'boundary_mode', 'scales', 'multiscale'):
        if param_key in values:
            result[param_key] = values[param_key]

    result['preset'] = _resolve_dust_preset(preset_name, result)

    policy_val = values.get('repair_policy')
    if not policy_val and preset_obj:
        policy_val = preset_obj.get('repair_policy')
    if policy_val:
        try:
            if isinstance(policy_val, RepairInclusionPolicy):
                result['repair_policy'] = policy_val.value
            else:
                result['repair_policy'] = RepairInclusionPolicy(str(policy_val).lower()).value
        except Exception:
            result['repair_policy'] = 'conservative'
    else:
        result['repair_policy'] = 'conservative'

    cp_sha = values.get('checkpoint_sha256')
    if not cp_sha and preset_obj:
        cp_sha = preset_obj.get('checkpoint_sha256', '')
    result['checkpoint_sha256'] = str(cp_sha or '')

    for key in ('research_confirmed', 'license_confirmed', 'enable_temporal_scratch'):
        result[key] = str(values.get(key, 'false')).lower()
    for key in ('dart_repo', 'dart_python'):
        result[key] = str(values.get(key, ''))
    if detector == 'dart':
        from .dart_runtime import installation
        result['dart_repo'], result['dart_python'] = installation(result,ROOT)
    return result


def base_options(values):
    """Normalize options stripping edit-layer identifiers for immutable base detector caches."""
    cleaned = {k: v for k, v in dict(values or {}).items() if k not in ('edit_signature', 'timeline_id')}
    return options(cleaned)


def authorize(values, guard=False, kind='repair'):
    o = options(values)
    if kind == 'repair' and o.get('analysis_source') == 'original':
        raise ValueError('Original input is analysis-only. Choose Stage input for repair and adoption.')

    target = 'dart_harness' if o['detector'] == 'dart' else o['detector']
    if DETECTOR_REGISTRY.has(target, commercial_only=False):
        det_obj = DETECTOR_REGISTRY.get(target, commercial_only=False)
        r_state = det_obj.readiness_state.value if hasattr(det_obj.readiness_state, 'value') else str(det_obj.readiness_state)
        if r_state in ('training_required', 'blocked'):
            raise ValueError(
                f"Defect detector '{o['detector']}' cannot be executed because its readiness state is '{r_state}'. "
                "Actionable guidance: Multi-frame temporal training or checkpoint provisioning is required before activation."
            )
        if guard and not det_obj.commercial_cleared:
            raise ValueError(
                f"Commercial model guard blocks {det_obj.name} ({det_obj.license_status})."
            )

    if o['detector'] in ('ridge', 'dart', 'bopb', 'patchcore', 'padim', E2_DETECTOR_NAME) and o['research_confirmed'] != 'true':
        raise ValueError('Confirm experimental use; authentic damage accuracy is unvalidated.')
    if o['detector'] in ('dart', 'bopb') or kind == 'repair' and o['repair'] in ('propainter', 'dart'):
        if guard:
            raise ValueError(
                f"Commercial model guard blocks {o['detector']} (unresolved / non-commercial rights) and ProPainter (non-commercial)."
                if o['detector'] in ('dart', 'bopb') else
                'Commercial model guard blocks DART (unresolved RTN rights) and ProPainter (non-commercial).'
            )
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


def folder_for(source, values, root: Path | str | None = None):
    key = hashlib.sha256(json.dumps(identity(source, values), sort_keys=True).encode()).hexdigest()[:24]
    return _resolve_root(root) / 'intermediate' / 'dust_scratches' / key


def base_folder_for(source, values, root: Path | str | None = None):
    return folder_for(source, base_options(values), root=root)


def output_for(source, values):
    return rel(folder_for(source, values) / 'repaired.mkv') if source else ''


@contextmanager
def _ensure_candidate_lock(target: Path):
    held = getattr(_THREAD_LOCK_STATE, 'paths', None)
    if held is None:
        held = set()
        _THREAD_LOCK_STATE.paths = held
    key = str(target.resolve(strict=False))
    if key in held:
        yield
        return
    held.add(key)
    try:
        with candidate_lock(target):
            yield
    finally:
        held.discard(key)


def resolve_edit_context(
    source: Path | str | None,
    values: dict[str, Any] | None = None,
    *,
    root: Path | str | None = None,
) -> dict[str, Any]:
    """Verify and resolve active edit store binding and signature for (source, analysis_source)."""
    resolved_root = _resolve_root(root)
    raw_vals = dict(values or {})
    explicit_sig = raw_vals.get('edit_signature', '')
    if explicit_sig:
        if not isinstance(explicit_sig, str) or not _EDIT_SIGNATURE_RE.fullmatch(explicit_sig):
            raise ValueError('Invalid Dust edit_signature.')
    elif explicit_sig != '':
        raise ValueError('Invalid Dust edit_signature.')

    explicit_tid = raw_vals.get('timeline_id', '')
    if explicit_tid:
        if not isinstance(explicit_tid, str) or not _TIMELINE_ID_RE.fullmatch(explicit_tid):
            raise ValueError('Invalid Dust timeline_id.')
    elif explicit_tid != '':
        raise ValueError('Invalid Dust timeline_id.')

    source_mode = str(raw_vals.get('analysis_source', 'stage') or 'stage').strip().lower()
    if source_mode not in ('stage', 'original'):
        raise ValueError('Choose stage or original analysis input.')

    source_path = None
    if source:
        try:
            cand = resolve(str(source)).resolve()
            if cand.is_file():
                source_path = cand
        except OSError:
            source_path = None

    if source_path is None:
        if explicit_sig or explicit_tid:
            raise FileNotFoundError(f'Source video not found for edit verification: {source}')
        return {
            'has_edits': False,
            'edit_signature': '',
            'timeline_id': '',
            'store': None,
            'binding': None,
        }

    masks_dir = resolved_root / DUST_MASKS_REL_DIR
    has_archive_files = masks_dir.is_dir() and any(masks_dir.iterdir())
    if not explicit_sig and not explicit_tid and not has_archive_files:
        return {
            'has_edits': False,
            'edit_signature': '',
            'timeline_id': '',
            'store': None,
            'binding': None,
        }

    meta = metadata(source_path)
    source_sha = digest(source_path)
    binding = build_timeline_binding(
        source_path,
        source_mode,
        width=meta['width'],
        height=meta['height'],
        frame_count=meta['frames'],
        fps=meta['fps'],
        root=resolved_root,
        source_sha256=source_sha,
    )
    store = load_verified_store(binding, root=resolved_root)
    if explicit_tid and explicit_tid != store.timeline_id:
        raise EditTimelineMismatchError(
            f'Edit timeline_id mismatch: expected {store.timeline_id!r}, got {explicit_tid!r}.'
        )
    if explicit_sig and (not store.exists() or store.edit_signature != explicit_sig):
        raise EditStoreIntegrityError(
            f'Edit signature mismatch: requested {explicit_sig!r}, verified store has {store.edit_signature!r}.'
        )
    active_sig = store.edit_signature
    return {
        'has_edits': bool(active_sig),
        'edit_signature': active_sig,
        'timeline_id': store.timeline_id if active_sig else '',
        'store': store,
        'binding': binding,
    }


def resolve_edit_signature(
    source: Path | str | None,
    values: dict[str, Any] | None = None,
    *,
    root: Path | str | None = None,
) -> str:
    """Return the verified active edit_signature ('<zip_sha256>:rev<N>') or '' when unedited."""
    return str(resolve_edit_context(source, values, root=root)['edit_signature'])


@lru_cache(maxsize=64)
def _metadata_cached(path_text: str, _size: int, _mtime_ns: int, _ctime_ns: int) -> tuple[tuple[str, Any], ...]:
    ffprobe = str(Path(find_ffmpeg()).with_name('ffprobe.exe' if sys.platform == 'win32' else 'ffprobe'))
    d = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=width,height,avg_frame_rate,nb_read_frames,pix_fmt,color_transfer,color_space,color_range,color_primaries', '-of', 'json', path_text], text=True))['streams'][0]
    from fractions import Fraction
    fps = Fraction(d['avg_frame_rate'])
    if fps <= 0 or int(d['nb_read_frames']) < 1:
        raise ValueError('Invalid source cadence/frame count.')
    info = {
        'width': int(d['width']),
        'height': int(d['height']),
        'frames': int(d['nb_read_frames']),
        'fps': str(fps),
        'pixel_format': d['pix_fmt'],
        'transfer': d.get('color_transfer', 'unknown'),
        **{key: d.get(key, 'unknown') for key in ('color_space', 'color_range', 'color_primaries')},
    }
    return tuple(info.items())


def metadata(source):
    target = Path(source)
    try:
        st = target.stat()
        resolved = str(target.resolve())
    except OSError:
        resolved = str(target)
        st = None
    if st is not None:
        with _METADATA_LOCK:
            return dict(_metadata_cached(resolved, st.st_size, st.st_mtime_ns, getattr(st, 'st_ctime_ns', 0)))
    return dict(_metadata_cached.__wrapped__(resolved, -1, -1, -1))


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
    if det_name == E2_DETECTOR_NAME and not is_research_confirmed(o.get('research_confirmed')):
        raise ValueError('Confirm experimental use; authentic damage accuracy is unvalidated.')
    enable_temporal = str(o.get('enable_temporal_scratch', 'false')).lower() == 'true' and det_name not in ('temporal_scratch', E2_DETECTOR_NAME)
    target = 'dart_harness' if det_name == 'dart' else det_name
    strategy = DETECTOR_REGISTRY.get(target, commercial_only=False) if DETECTOR_REGISTRY.has(target, commercial_only=False) else None
    if det_name == E2_DETECTOR_NAME:
        radius = 3
        pad = 4
        is_spatial = False
        temporal_strategy = None
    else:
        radius = int(strategy.temporal_radius) if strategy is not None else 1
        temporal_strategy = None
        if enable_temporal:
            temporal_strategy = DETECTOR_REGISTRY.get('temporal_scratch', commercial_only=False)
            radius = max(radius, int(temporal_strategy.temporal_radius))
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
        first = pad if left_context else 0
        last = n_images - pad if following else n_images

        pred_masks = None
        pred_scratch = None
        pred_dust = None
        pred_uncertain = None
        pred_vetoed = None
        pred_prob = None
        pred_frame_meta = None

        if det_name == E2_DETECTOR_NAME:
            pred_res = run_experimental_e2_sequence(
                images,
                sensitivity=float(o.get('sensitivity', 0.65)),
                research_confirmed=o.get('research_confirmed', 'false'),
                target_indices=list(range(first, last)),
            )
            pred_masks = pred_res.mask
            pred_scratch = pred_res.scratch_mask
            pred_dust = pred_res.dust_mask
            pred_uncertain = getattr(pred_res, 'uncertain_mask', None)
            pred_vetoed = pred_res.metadata.get('vetoed_masks')
            pred_prob = [m.combined_probability for m in pred_res.multi_channel_masks] if getattr(pred_res, 'multi_channel_masks', None) else None
            pred_frame_meta = pred_res.metadata.get('frame_metadata')
        elif det_name == 'legacy':
            legacy_masks = [ai_scratch_mask(img, o['sensitivity']) for img in images]
            zeros = [np.zeros(img.shape[:2], np.uint8) for img in images]
            pred_res = DefectMaskResult(
                mask=legacy_masks,
                scratch_mask=legacy_masks,
                dust_mask=zeros,
                uncertain_mask=zeros,
            )
            if enable_temporal and temporal_strategy is not None:
                temp_cfg = BENCHMARK_PRESETS['temporal_scratch_balanced'].params
                temp_res = temporal_strategy.detect(images, temp_cfg)
                pred_res = fuse_spatial_and_temporal_masks(pred_res, temp_res, is_anomaly_detector=False)
            pred_masks = pred_res.mask
            pred_scratch = pred_res.scratch_mask
            pred_dust = pred_res.dust_mask
            pred_uncertain = getattr(pred_res, 'uncertain_mask', None)
            pred_prob = [m.combined_probability for m in pred_res.multi_channel_masks] if getattr(pred_res, 'multi_channel_masks', None) else None
        elif is_spatial and strategy is not None and images:
            cfg = detector_strategy_config(strategy, o)
            pred_res = strategy.detect(images, cfg)
            if enable_temporal and temporal_strategy is not None:
                temp_cfg = BENCHMARK_PRESETS['temporal_scratch_balanced'].params
                temp_res = temporal_strategy.detect(images, temp_cfg)
                is_anomaly = det_name in ('patchcore', 'padim')
                pred_res = fuse_spatial_and_temporal_masks(pred_res, temp_res, is_anomaly_detector=is_anomaly)
            pred_masks = pred_res.mask
            pred_scratch = pred_res.scratch_mask
            pred_dust = pred_res.dust_mask
            pred_uncertain = getattr(pred_res, 'uncertain_mask', None)
            pred_prob = [m.combined_probability for m in pred_res.multi_channel_masks] if getattr(pred_res, 'multi_channel_masks', None) else None
        elif not is_spatial and strategy is not None and n_images >= min_context:
            pred_res = None
            if det_name == 'ridge':
                if detect_sequence is not _DEFAULT_RIDGE_FN:
                    r_masks = detect_sequence(images, o['sensitivity'])
                    zeros = [np.zeros(f.shape[:2], np.uint8) for f in images]
                    pred_res = DefectMaskResult(mask=r_masks, scratch_mask=r_masks, dust_mask=zeros, uncertain_mask=zeros)
                else:
                    cfg = detector_strategy_config(strategy, o)
                    pred_res = strategy.detect(images, cfg)
            elif det_name not in ('legacy', 'temporal', 'farneback_3f'):
                cfg = detector_strategy_config(strategy, o)
                pred_res = strategy.detect(images, cfg)

            if pred_res is not None:
                if enable_temporal and temporal_strategy is not None:
                    temp_cfg = BENCHMARK_PRESETS['temporal_scratch_balanced'].params
                    temp_res = temporal_strategy.detect(images, temp_cfg)
                    pred_res = fuse_spatial_and_temporal_masks(pred_res, temp_res, is_anomaly_detector=False)
                pred_masks = pred_res.mask
                pred_scratch = pred_res.scratch_mask
                pred_dust = pred_res.dust_mask
                pred_uncertain = getattr(pred_res, 'uncertain_mask', None)
                pred_prob = [m.combined_probability for m in pred_res.multi_channel_masks] if getattr(pred_res, 'multi_channel_masks', None) else None

        result = []
        for i in range(first, last):
            image, cut_before = items[i]
            reason = ''
            h, w = image.shape[:2]
            uncertain_mask = np.zeros((h, w), np.uint8)
            vetoed_mask = np.zeros((h, w), np.uint8)
            combined_p = None
            if pred_masks is not None and i < len(pred_masks):
                mask = pred_masks[i]
                scratch_mask = pred_scratch[i] if pred_scratch is not None else mask.copy()
                dust_mask = pred_dust[i] if pred_dust is not None else np.zeros((h, w), np.uint8)
                uncertain_mask = pred_uncertain[i] if pred_uncertain is not None else np.zeros((h, w), np.uint8)
                if pred_vetoed is not None and i < len(pred_vetoed):
                    vetoed_mask = pred_vetoed[i]
                if pred_prob is not None and i < len(pred_prob):
                    combined_p = pred_prob[i]
                if pred_frame_meta is not None and i < len(pred_frame_meta):
                    fmeta = pred_frame_meta[i]
                    if fmeta.get('abstained') and fmeta.get('abstention_reason'):
                        reason = str(fmeta['abstention_reason'])
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
                    cfg = detector_strategy_config(strategy, o)
                    res = strategy.detect([images[i-1], image, images[i+1]], cfg)
                    mask = res.mask[1]
                    scratch_mask = res.scratch_mask[1]
                    dust_mask = res.dust_mask[1]
                    uncertain_mask = res.uncertain_mask[1] if getattr(res, 'uncertain_mask', None) is not None else np.zeros((h, w), np.uint8)
            else:
                mask = np.zeros((h, w), np.uint8)
                scratch_mask = np.zeros((h, w), np.uint8)
                dust_mask = np.zeros((h, w), np.uint8)
                reason = 'source_or_cut_boundary: insufficient context'

            if cut_before:
                reason = 'source_cut_boundary' + (': insufficient context' if not is_spatial else ': legacy spatial mask')
            result.append(
                MaskItem(
                    image,
                    mask,
                    reason,
                    scratch_mask=scratch_mask,
                    dust_mask=dust_mask,
                    uncertain_mask=uncertain_mask,
                    combined_probability=combined_p,
                    vetoed_mask=vetoed_mask,
                )
            )
        return result

    for image in frames(source):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # E2 implements M2 motion-robust cut selection internally so fast pans are not split prematurely.
        cut = False if det_name == E2_DETECTOR_NAME else (previous is not None and float(np.median(cv2.absdiff(gray, previous))) >= 22)
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
    for name in ('scratch_mask.mkv', 'dust_mask.mkv', 'uncertain_mask.mkv', 'vetoed_mask.mkv'):
        if name in report.get('hashes', {}):
            check_names.append(name)
    if kind == 'repair' and 'restoration_provenance.json' in report.get('hashes', {}):
        check_names.append('restoration_provenance.json')
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
            files = {
                name: path for name, path in files.items()
                if name in (
                    'input.mkv', 'mask.mkv', 'effective_mask.mkv',
                    'scratch_mask.mkv', 'dust_mask.mkv', 'uncertain_mask.mkv',
                    'vetoed_mask.mkv', 'measurements.csv',
                )
            }
        result_dict = {
            'source': source,
            'files': files,
            'frames': report['metadata']['frames'],
            'repair_complete': complete,
            'repair_error': repair_error,
            'mask_values': report['identity']['options'],
            'limitations': report['limitations'],
            'detector_info': detector_metadata(report['identity']['options']['detector']),
            'edit_signature': report.get('edit_signature', ''),
            'timeline_id': report.get('timeline_id', ''),
            'base_folder_key': report.get('base_folder_key', ''),
            'is_edited_composition': bool(report.get('edit_signature')),
        }
        if report.get('edit_provenance'):
            result_dict['edit_provenance'] = report['edit_provenance']
        if report.get('coverage_diagnostics'):
            result_dict['coverage_diagnostics'] = report['coverage_diagnostics']
            if report['coverage_diagnostics'].get('excessive_coverage'):
                result_dict['coverage_warning'] = report['coverage_diagnostics'].get('warning')
                if not complete:
                    result_dict['repair_blocked_reason'] = report['coverage_diagnostics'].get('warning')
        return result_dict
    except (OSError, ValueError, KeyError, TypeError):
        det_name = values.get('detector', 'temporal')
        result = {
            'source': source,
            'files': {},
            'repair_complete': False,
            'frames': 0,
            'detector_info': detector_metadata(det_name),
            'edit_signature': str(values.get('edit_signature', '') or ''),
            'timeline_id': str(values.get('timeline_id', '') or ''),
            'is_edited_composition': bool(values.get('edit_signature')),
        }
        if values.get('edit_signature'):
            try:
                saved_base_mask(source, values)
                result['base_mask_ready'] = True
            except (OSError, ValueError, FileNotFoundError):
                result['base_mask_ready'] = False
        if values.get('detector') == 'dart':
            opts = options(values)
            result['installation'] = {'available':all(safe_is_file(path) for path in (
                Path(opts['dart_repo'])/'scripts/inference.py',Path(opts['dart_python']),
                Path(opts['dart_repo'])/'pretrained_models/dart.safetensors')),
                'message':'Set up DART with install_windows.bat -DartOnly. ARP uses tools/dart automatically; no ComfyUI node is required.'}
        return result


def _same_canonical_detector(det_a: str, det_b: str) -> bool:
    if det_a == det_b:
        return True
    return (
        DETECTOR_REGISTRY.has(det_a, commercial_only=False)
        and DETECTOR_REGISTRY.has(det_b, commercial_only=False)
        and DETECTOR_REGISTRY.get(det_a, commercial_only=False).name == DETECTOR_REGISTRY.get(det_b, commercial_only=False).name
    )


def saved_mask(source, values, *, root: Path | str | None = None):
    """Read an existing matching detector artifact; never infer or rewrite it."""
    resolved_root = _resolve_root(root)
    opts = options(values)
    folder = folder_for(source, opts, root=resolved_root)
    try:
        return folder, manifest(folder, source, opts, 'mask')
    except (FileNotFoundError, OSError):
        pass
    if opts['detector'] != 'dart':
        edit_sig = opts.get('edit_signature', '')
        policy = opts.get('repair_policy', 'conservative')
        for path in sorted((resolved_root / 'intermediate' / 'dust_scratches').glob('*/report.json')):
            if path.parent.name.startswith('.'):
                continue
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                other = report['identity']['options']
                if other.get('edit_signature', '') != edit_sig:
                    continue
                if other.get('repair_policy', 'conservative') != policy:
                    continue
                if other.get('mask_version', '') != opts.get('mask_version', ''):
                    continue
                if other.get('analysis_source') != opts.get('analysis_source'):
                    continue
                if not _same_canonical_detector(other['detector'], opts['detector']):
                    continue
                if any(other.get(key) != opts.get(key) for key in ('sensitivity', 'dilate', 'threshold', 'enable_temporal_scratch')):
                    continue
                return path.parent, manifest(path.parent, source, other, 'mask')
            except (OSError, ValueError, KeyError, TypeError):
                continue
    raise FileNotFoundError('No saved mask for this input and detector settings. Calculate a mask first.')


def saved_base_mask(source, values, *, root: Path | str | None = None):
    """Find an immutable raw base detector folder (<base_key>/) without edits."""
    resolved_root = _resolve_root(root)
    base_opts = base_options(values)
    folder = folder_for(source, base_opts, root=resolved_root)
    try:
        return folder, manifest(folder, source, base_opts, 'mask')
    except (FileNotFoundError, OSError):
        pass
    if base_opts['detector'] != 'dart':
        for path in sorted((resolved_root / 'intermediate' / 'dust_scratches').glob('*/report.json')):
            if path.parent.name.startswith('.'):
                continue
            try:
                report = json.loads(path.read_text(encoding='utf-8'))
                other = report['identity']['options']
                if other.get('edit_signature', ''):
                    continue
                if other.get('mask_version', '') != base_opts.get('mask_version', ''):
                    continue
                if other.get('analysis_source') != base_opts.get('analysis_source'):
                    continue
                if not _same_canonical_detector(other['detector'], base_opts['detector']):
                    continue
                if any(other.get(key) != base_opts.get(key) for key in ('sensitivity', 'dilate', 'threshold', 'enable_temporal_scratch')):
                    continue
                return path.parent, manifest(path.parent, source, other, 'mask')
            except (OSError, ValueError, KeyError, TypeError):
                continue
    raise FileNotFoundError('No saved base mask for this input and detector settings. Calculate a mask first.')


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
                edit_sig = str(values.get('edit_signature', '') or '')
                label_suffix = f" · edited r{report.get('edit_revision', '')}" if edit_sig else ''
                rows.append({
                    'method': path.parent.name,
                    'preset': 'settings',
                    'values': values,
                    'label': f"{values['detector']} / {values['repair']} · {path.parent.name[:6]}{label_suffix}",
                    'frames': report['metadata']['frames'],
                    'output': rel(path.parent/'repaired.mkv') if complete else '',
                    'repair_complete': complete,
                    'revision': digest(path),
                    'mask_sha256': report['hashes']['effective_mask.mkv'],
                    'job': job,
                    'repair_error': repair_error,
                    'edit_signature': edit_sig,
                    'timeline_id': str(values.get('timeline_id', '') or ''),
                    'is_edited_composition': bool(edit_sig),
                    'base_folder_key': str(report.get('base_folder_key', '') or ''),
                    'edit_revision': int(report.get('edit_revision', 0) or 0),
                    'edited_frames': list(report.get('edited_frames', []) or []),
                    'adopted': (
                        complete
                        and app.settings.get('dust', {}).get('adopted_report') == digest(path)
                        and options(app.settings.get('dust', {})) == values
                    ),
                })
            except (OSError, ValueError, KeyError, TypeError):
                continue
    presets_dict = {name: p.as_dict() for name, p in BENCHMARK_PRESETS.items()}
    presets_dict.update({name: dict(preset) for name, preset in EXPERIMENTAL_PRESETS.items()})
    dust_edits_info = None
    if source and safe_is_file(source) and (ROOT / DUST_MASKS_REL_DIR).is_dir():
        try:
            src_path = resolve(str(source)).resolve()
            meta = metadata(src_path)
            binding = build_timeline_binding(
                src_path,
                'original' if original else 'stage',
                width=meta['width'],
                height=meta['height'],
                frame_count=meta['frames'],
                fps=meta['fps'],
                root=ROOT,
                source_sha256=digest(src_path),
            )
            compat = store_compatibility_status(binding, root=ROOT)
            compat.pop('store', None)
            dust_edits_info = compat
        except Exception as exc:
            dust_edits_info = {
                'compatible': False,
                'compatibility_reason': str(exc),
                'edit_signature': '',
                'timeline_id': '',
                'has_edits': False,
            }
    result = {
        'source': source,
        'source_revision': json.dumps(identity(source, {}), sort_keys=True) if source else '',
        'variants': rows,
        'ready': sum(r['repair_complete'] for r in rows),
        'running': app.running_stage_key == 'dust',
        'stop_requested': bool(stop_requested),
        'available_detectors': available_detectors(commercial_only=False),
        'presets': presets_dict,
        'available_presets': list(presets_dict.values()),
        'active_edit_signature': dust_edits_info.get('edit_signature', '') if dust_edits_info else '',
    }
    if dust_edits_info is not None:
        result['dust_edits'] = dust_edits_info
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
    if 'dart_full.mkv' in report['hashes']:
        if digest(folder / 'dart_full.mkv') != report['hashes']['dart_full.mkv']:
            raise ValueError('DART diagnostic output changed.')
    counts = {}
    for name in ('measurements.csv', 'repair_measurements.csv', 'native_measurements.csv'):
        if not complete and name != 'measurements.csv':
            continue
        if name not in report['hashes']:
            continue
        with (folder / name).open(encoding='utf-8-sig', newline='') as f:
            counts.update(next(row for row in csv.DictReader(f) if int(row['frame']) == frame))

    base_folder = None
    base_report = None
    if report.get('edit_signature'):
        base_opts = base_options(report['identity']['options'])
        base_key = str(report.get('base_folder_key', '') or '')
        if base_key:
            cand_base = folder.parent / base_key
            if cand_base.is_dir():
                try:
                    base_report = manifest(cand_base, source, base_opts, 'mask')
                    base_folder = cand_base
                except (OSError, ValueError, KeyError, TypeError):
                    base_folder = None
                    base_report = None
        if base_folder is None:
            try:
                base_folder, base_report = saved_base_mask(source, base_opts, root=ROOT)
            except (OSError, ValueError, FileNotFoundError):
                base_folder = None
                base_report = None

    from .frame_inspector import inspect_dust
    return inspect_dust(
        folder,
        report,
        frame,
        complete=complete,
        repair_error=repair_error,
        base_folder=base_folder,
        base_report=base_report,
        detector_info=detector_metadata(report['identity']['options']['detector']),
        counts=counts,
        root=ROOT,
    )


def _build_coverage_diagnostics(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    total_frame_pixels = int(meta['width']) * int(meta['height'])
    total_pixels_all = max(1, len(rows) * total_frame_pixels)
    mean_mask_cov = sum(int(r['mask_pixels']) for r in rows) / total_pixels_all
    mean_repair_cov = sum(int(r['repair_pixels']) for r in rows) / total_pixels_all
    max_repair_cov = max((int(r['repair_pixels']) for r in rows), default=0) / max(1, total_frame_pixels)

    coverage_diag: dict[str, Any] = {
        'mean_mask_coverage_fraction': round(mean_mask_cov, 5),
        'mean_repair_coverage_fraction': round(mean_repair_cov, 5),
        'max_repair_coverage_fraction': round(max_repair_cov, 5),
        'excessive_coverage': bool(mean_repair_cov > 0.30 or mean_mask_cov > 0.30),
        'safety_threshold': 0.30,
    }
    if coverage_diag['excessive_coverage']:
        coverage_diag['warning'] = (
            f"Excessive mask coverage detected: {max(mean_repair_cov, mean_mask_cov) * 100:.1f}% of frame area marked as defect "
            "(safety threshold is 30.0%). Automatic repair is blocked to prevent destructive inpainting of intact footage."
        )
    return coverage_diag


def _build_repair_provenance(o: dict[str, Any], report: dict[str, Any], target: Path, source: Path) -> dict[str, Any]:
    from .dust_jobs import build_restoration_provenance

    det_name = o['detector']
    if det_name == E2_DETECTOR_NAME:
        detector_arg: str | dict[str, Any] = {
            'name': E2_DETECTOR_NAME,
            'version': E2_DETECTOR_VERSION,
            'architecture': E2_DETECTOR_ARCHITECTURE,
        }
        channel_prov = {'scratch': 'native', 'dust': 'unsupported', 'uncertain': 'native'}
    else:
        detector_arg = det_name
        channel_prov = {
            'scratch': 'native' if det_name in ('ridge', 'bopb') else 'derived',
            'dust': 'derived',
            'uncertain': 'derived',
        }

    prov = build_restoration_provenance(
        detector=detector_arg,
        preset=o.get('preset'),
        repair_policy=o.get('repair_policy', 'conservative'),
        repair_backend=o['repair'],
        effective_mask_hash=digest(target / 'effective_mask.mkv'),
        source_video=source,
        output_video=target / 'repaired.mkv',
        checkpoint_sha256=o.get('checkpoint_sha256', ''),
        license_info=detector_metadata(det_name).get('license_status', ''),
        channel_provenance=channel_prov,
        resolved_parameters={
            'radius': o.get('radius', 3),
            'sensitivity': o.get('sensitivity', 0.65),
            'dilate': o.get('dilate', 0),
            'threshold': o.get('threshold', 160),
        },
    )

    edit_prov = report.get('edit_provenance') if isinstance(report.get('edit_provenance'), dict) else {}
    counts = edit_prov.get('counts', {}) if isinstance(edit_prov.get('counts'), dict) else {}
    human_added = int(counts.get('human_scratch_add', 0)) + int(counts.get('human_dust_add', 0))
    human_rejected = (
        int(counts.get('human_scratch_reject', 0))
        + int(counts.get('human_dust_reject', 0))
        + int(counts.get('human_uncertain_reject', 0))
    )
    protect_intact = int(counts.get('protect_intact', 0))
    operational_reviewed = int(counts.get('operational_reviewed', 0))

    edit_sig = str(report.get('edit_signature') or o.get('edit_signature') or '')
    timeline_id = str(report.get('timeline_id') or o.get('timeline_id') or '')

    prov['edit_signature'] = edit_sig
    prov['timeline_id'] = timeline_id
    prov['base_folder_key'] = str(report.get('base_folder_key', '') or '')
    prov['base_report_sha256'] = str(report.get('base_report_sha256', '') or '')
    prov['ground_truth_status'] = GROUND_TRUTH_STATUS
    prov['edit_provenance'] = {
        **edit_prov,
        'has_edits': bool(edit_sig),
        'edit_signature': edit_sig,
        'timeline_id': timeline_id,
        'human_added': human_added,
        'human_rejected': human_rejected,
        'protect_intact': protect_intact,
        'operational_reviewed': operational_reviewed,
        'ground_truth_status': GROUND_TRUTH_STATUS,
        'protect_or_reject_auto_promoted_to_gt': False,
        'untouched_auto_as_gt': False,
    }

    if det_name == E2_DETECTOR_NAME or report.get('e2_metadata'):
        e2_meta = dict(report.get('e2_metadata') or {})
        e2_meta.setdefault('detector', E2_DETECTOR_NAME)
        e2_meta.setdefault('version', E2_DETECTOR_VERSION)
        e2_meta.setdefault('architecture', E2_DETECTOR_ARCHITECTURE)
        e2_meta.setdefault(
            'three_state_outputs',
            ['high_confidence_damage', 'uncertain_damage_candidate', 'likely_intact_scene_structure'],
        )
        e2_meta['research_confirmed'] = is_research_confirmed(o.get('research_confirmed'))
        e2_meta['non_flattened_uncertainty'] = True
        e2_meta['vetoed_mask_hash'] = str(report.get('hashes', {}).get('vetoed_mask.mkv', '') or '')
        prov['e2_metadata'] = e2_meta

    return prov


def _frame_plane_iter(path: Path, count: int, h: int, w: int):
    if not path.is_file():
        zeros = np.zeros((h, w), dtype=np.uint8)
        for _ in range(count):
            yield zeros
        return
    yielded = 0
    for img in frames(path):
        yield img[:, :, 0]
        yielded += 1
    if yielded != count:
        raise ValueError(f'Mask stream {path.name} frame count {yielded} != expected {count}.')


def compose_edited_masks(
    source,
    values,
    root: Path | str | None = None,
    *,
    guard: bool = False,
    comfy_url: str = 'http://127.0.0.1:8188',
    comfy_output: str = '',
    comfy_dir=None,
    cancel_file=None,
):
    """Compose an edit-bound mask folder (<composed_key>/) without mutating <base_key>/."""
    resolved_root = _resolve_root(root)
    source_path = resolve(str(source)).resolve()
    o = authorize(values, guard, 'mask')
    ctx = resolve_edit_context(source_path, o, root=resolved_root)

    if not ctx['has_edits']:
        base_opts = base_options(o)
        try:
            _, base_report = saved_base_mask(source_path, base_opts, root=resolved_root)
            return base_report
        except (FileNotFoundError, OSError, ValueError):
            base_target = folder_for(source_path, base_opts, root=resolved_root)
            with _ensure_candidate_lock(base_target):
                return _run(
                    source_path,
                    base_opts,
                    'mask',
                    guard,
                    comfy_url,
                    comfy_output,
                    comfy_dir,
                    cancel_file,
                    _root=resolved_root,
                    _base_only=True,
                )

    composed_opts = options({
        **o,
        'edit_signature': ctx['edit_signature'],
        'timeline_id': ctx['timeline_id'],
    })
    target = folder_for(source_path, composed_opts, root=resolved_root)
    target.parent.mkdir(parents=True, exist_ok=True)

    with _ensure_candidate_lock(target):
        if target.exists():
            try:
                existing = manifest(target, source_path, composed_opts, 'mask')
                if existing.get('edit_signature') == ctx['edit_signature']:
                    return existing
            except (FileNotFoundError, OSError, ValueError):
                pass

        base_opts = base_options(composed_opts)
        try:
            base_folder, base_report = saved_base_mask(source_path, base_opts, root=resolved_root)
        except (FileNotFoundError, OSError, ValueError):
            base_folder = folder_for(source_path, base_opts, root=resolved_root)
            with _ensure_candidate_lock(base_folder):
                base_report = _run(
                    source_path,
                    base_opts,
                    'mask',
                    guard,
                    comfy_url,
                    comfy_output,
                    comfy_dir,
                    cancel_file,
                    _root=resolved_root,
                    _base_only=True,
                )
        base_report_sha256 = digest(base_folder / 'report.json')
        meta = base_report['metadata']
        h, w, frame_count = int(meta['height']), int(meta['width']), int(meta['frames'])

        abstentions: dict[int, str] = {}
        meas_path = base_folder / 'measurements.csv'
        if meas_path.is_file():
            with meas_path.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    abstentions[int(row['frame'])] = str(row.get('abstention', '') or '')

        store: DustMaskEditStore = ctx['store']
        edit_map = store.read_all_frames()
        if store.edit_signature != ctx['edit_signature']:
            raise EditRevisionConflictError(
                f"Edit revision changed during mask composition ({store.edit_signature!r} != {ctx['edit_signature']!r})"
            )
        policy_enum = RepairInclusionPolicy(composed_opts.get('repair_policy', 'conservative'))
        dilate_iters = int(composed_opts.get('dilate', 0))

        initial_identity = identity(source_path, composed_opts)
        initial_hash = digest(source_path)

        with staging_directory(prefix=f'.{target.name}.analysis-', dir=target.parent, delete=False) as tmp:
            stage = Path(tmp)
            shutil.copyfile(base_folder / 'input.mkv', stage / 'input.mkv')
            copied_optional: list[str] = []
            for opt_name in ('vetoed_mask.mkv', 'dart_full.mkv'):
                if opt_name in base_report.get('hashes', {}) and (base_folder / opt_name).is_file():
                    shutil.copyfile(base_folder / opt_name, stage / opt_name)
                    copied_optional.append(opt_name)

            writers = {
                'mask.mkv': Writer(stage / 'mask.mkv', meta),
                'effective_mask.mkv': Writer(stage / 'effective_mask.mkv', meta),
                'scratch_mask.mkv': Writer(stage / 'scratch_mask.mkv', meta),
                'dust_mask.mkv': Writer(stage / 'dust_mask.mkv', meta),
                'uncertain_mask.mkv': Writer(stage / 'uncertain_mask.mkv', meta),
            }

            has_vetoed = 'vetoed_mask.mkv' in copied_optional
            rows: list[dict[str, Any]] = []
            auto_frames_prov: dict[int, dict[str, np.ndarray]] = {}
            effective_frames_prov: dict[int, np.ndarray] = {}

            try:
                streams = zip(
                    _frame_plane_iter(base_folder / 'mask.mkv', frame_count, h, w),
                    _frame_plane_iter(base_folder / 'scratch_mask.mkv', frame_count, h, w),
                    _frame_plane_iter(base_folder / 'dust_mask.mkv', frame_count, h, w),
                    _frame_plane_iter(base_folder / 'uncertain_mask.mkv', frame_count, h, w),
                    _frame_plane_iter(base_folder / 'vetoed_mask.mkv', frame_count, h, w),
                    strict=True,
                )
                for index, (raw_m, raw_s, raw_d, raw_u, raw_v) in enumerate(streams):
                    check_cancel(cancel_file)
                    edit_u8 = edit_map.get(index)
                    resolved = resolve_effective_channels(
                        raw_s,
                        raw_d,
                        raw_u,
                        edit_u8,
                        policy_enum,
                        auto_combined=raw_m,
                        dilate=dilate_iters,
                    )
                    s_clean = ensure_uint8_binary_mask(resolved['scratch'], (h, w), 'scratch_mask')
                    d_clean = ensure_uint8_binary_mask(resolved['dust'], (h, w), 'dust_mask')
                    u_clean = ensure_uint8_binary_mask(resolved['uncertain'], (h, w), 'uncertain_mask')
                    eff_clean = ensure_uint8_binary_mask(resolved['effective'], (h, w), 'effective_mask')
                    prot_clean = ensure_uint8_binary_mask(resolved['protected'], (h, w), 'protected_mask')

                    if np.any((eff_clean > 0) & (prot_clean > 0)):
                        raise ValueError('Critical Invariant Violation: PROTECT_INTACT survived mask composition!')

                    comb_clean = np.bitwise_or(np.bitwise_or(s_clean, d_clean), u_clean)
                    if policy_enum == RepairInclusionPolicy.COMBINED_DEFECT:
                        comb_clean = np.bitwise_or(comb_clean, eff_clean)
                    comb_clean[prot_clean > 0] = 0
                    comb_clean = ensure_uint8_binary_mask(comb_clean, (h, w), 'mask')

                    writers['mask.mkv'].write(cv2.cvtColor(comb_clean, cv2.COLOR_GRAY2BGR))
                    writers['effective_mask.mkv'].write(cv2.cvtColor(eff_clean, cv2.COLOR_GRAY2BGR))
                    writers['scratch_mask.mkv'].write(cv2.cvtColor(s_clean, cv2.COLOR_GRAY2BGR))
                    writers['dust_mask.mkv'].write(cv2.cvtColor(d_clean, cv2.COLOR_GRAY2BGR))
                    writers['uncertain_mask.mkv'].write(cv2.cvtColor(u_clean, cv2.COLOR_GRAY2BGR))

                    if (
                        edit_u8 is not None
                        or np.any(raw_s > 0)
                        or np.any(raw_d > 0)
                        or np.any(raw_u > 0)
                        or np.any(raw_v > 0)
                        or np.any(eff_clean > 0)
                    ):
                        auto_frames_prov[index] = {
                            'scratch': raw_s.copy(),
                            'dust': raw_d.copy(),
                            'uncertain': raw_u.copy(),
                            'vetoed': raw_v.copy(),
                        }
                        effective_frames_prov[index] = eff_clean.copy()

                    row_entry: dict[str, Any] = {
                        'frame': index,
                        'mask_pixels': int(np.count_nonzero(comb_clean >= 128)),
                        'repair_pixels': int(np.count_nonzero(eff_clean)),
                        'scratch_pixels': int(np.count_nonzero(s_clean >= 128)),
                        'dust_pixels': int(np.count_nonzero(d_clean >= 128)),
                        'uncertain_pixels': int(np.count_nonzero(u_clean >= 128)),
                        'protected_pixels': int(np.count_nonzero(prot_clean >= 128)),
                    }
                    if has_vetoed:
                        row_entry['vetoed_pixels'] = int(np.count_nonzero(raw_v >= 128))
                    row_entry['abstention'] = abstentions.get(index, '')
                    row_entry['note'] = (
                        'Composed from immutable base detector cache and verified project mask edits; '
                        'operational edits only, not independent ground truth.'
                    )
                    rows.append(row_entry)
            finally:
                close_writers(writers.values())

            write_csv(stage / 'measurements.csv', rows)
            if identity(source_path, composed_opts) != initial_identity or digest(source_path) != initial_hash:
                raise ValueError('Source changed during composed mask resolution; refusing publication.')

            coverage_diag = _build_coverage_diagnostics(rows, meta)
            edit_prov_summary = build_edit_provenance_summary(
                store,
                auto_frames=auto_frames_prov,
                effective_frames=effective_frames_prov,
                include_per_frame=False,
            )

            all_hashed = ('input.mkv', *writers.keys(), *copied_optional, 'measurements.csv')
            report: dict[str, Any] = {
                'identity': initial_identity,
                'source_sha256': initial_hash,
                'metadata': meta,
                'repair_complete': False,
                'coverage_diagnostics': coverage_diag,
                'hashes': {name: digest(stage / name) for name in all_hashed},
                'limitations': detector_limitations(composed_opts['detector']),
                'base_folder_key': base_folder.name,
                'base_report_sha256': base_report_sha256,
                'edit_signature': ctx['edit_signature'],
                'timeline_id': ctx['timeline_id'],
                'edit_revision': store.edit_revision,
                'edit_zip_sha256': store.zip_sha256,
                'edited_frames': store.edited_frames(),
                'edit_provenance': edit_prov_summary,
            }
            for passthrough in ('model_preflight', 'weights_sha256', 'e2_metadata'):
                if passthrough in base_report:
                    report[passthrough] = base_report[passthrough]

            publish_report(stage, report)
            check_cancel(cancel_file)
            return finish_analysis(stage, target, source_path, composed_opts, cancel_file)


def run(source, values, kind='mask', guard=False, comfy_url='http://127.0.0.1:8188', comfy_output='', comfy_dir=None, cancel_file=None, ignore_edits=False):
    if kind not in ('mask', 'repair'):
        raise ValueError('Choose mask analysis or repair.')
    o = authorize(values, guard, kind)
    base_only = bool(ignore_edits and kind == 'mask')
    if kind == 'repair':
        ctx = resolve_edit_context(source, o, root=ROOT)
        if ignore_edits and ctx['has_edits']:
            raise ValueError(
                'Cannot ignore active manual mask edits during repair when edits exist on disk; '
                'reset manual edits first if you want an unedited repair.'
            )
        if ctx['has_edits'] and not o.get('edit_signature'):
            o = authorize(
                {**o, 'edit_signature': ctx['edit_signature'], 'timeline_id': ctx['timeline_id']},
                guard,
                kind,
            )
    elif base_only:
        o.pop('edit_signature', None)
        o.pop('timeline_id', None)
    with _ensure_candidate_lock(folder_for(source, o)):
        check_cancel(cancel_file)
        try:
            report = _run(source, o, kind, guard, comfy_url, comfy_output, comfy_dir, cancel_file, _base_only=base_only)
        except Exception as exc:
            target = folder_for(source,o)
            if (target/'report.json').exists():
                publish_json(target/'job-state.json',{'state':'stopped' if cancel_file and Path(cancel_file).exists() else 'failed','message':str(exc)})
            raise
        check_cancel(cancel_file)
        return report


def _run(source, values, kind, guard, comfy_url, comfy_output, comfy_dir, cancel_file, *, _root=None, _base_only=False):
    resolved_root = _resolve_root(_root)
    o = authorize(values, guard, kind)
    source = resolve(str(source)).resolve()
    if kind == 'mask' and _base_only:
        o.pop('edit_signature', None)
        o.pop('timeline_id', None)

    if kind == 'mask' and not _base_only and o.get('edit_signature'):
        return compose_edited_masks(
            source,
            o,
            root=resolved_root,
            guard=guard,
            comfy_url=comfy_url,
            comfy_output=comfy_output,
            comfy_dir=comfy_dir,
            cancel_file=cancel_file,
        )

    edit_ctx = {'has_edits': False, 'edit_signature': '', 'timeline_id': '', 'store': None, 'binding': None}
    if kind == 'repair':
        edit_ctx = resolve_edit_context(source, o, root=resolved_root)
        if edit_ctx['has_edits'] and not o.get('edit_signature'):
            o = authorize(
                {**o, 'edit_signature': edit_ctx['edit_signature'], 'timeline_id': edit_ctx['timeline_id']},
                guard,
                kind,
            )

    target = folder_for(source, o, root=resolved_root)
    if kind == 'repair' and o.get('edit_signature') and not target.exists():
        compose_edited_masks(
            source,
            o,
            root=resolved_root,
            guard=guard,
            comfy_url=comfy_url,
            comfy_output=comfy_output,
            comfy_dir=comfy_dir,
            cancel_file=cancel_file,
        )

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
            saved_mask(source, o, root=resolved_root)  # Refuse missing analysis; the mask pass only copies validated frozen bytes.
            _run(source, o, 'mask', guard, comfy_url, comfy_output, comfy_dir, cancel_file, _root=resolved_root)
        report = manifest(target, source, o, 'mask')
        cov_diag = report.get('coverage_diagnostics', {})
        if cov_diag.get('excessive_coverage') and values.get('allow_excessive_coverage') != 'true':
            raise ValueError(
                f"Repair refused by safety guard: {cov_diag.get('warning', 'Excessive mask coverage detected.')} "
                "Applying inpainting would destroy intact film footage. "
                "To bypass this safety guard, set allow_excessive_coverage=true."
            )
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

        edit_map: dict[int, np.ndarray] = {}
        if edit_ctx['has_edits'] and edit_ctx['store'] is not None:
            edit_map = edit_ctx['store'].read_all_frames()
            expected_sig = str(o.get('edit_signature') or '').strip()
            if expected_sig and edit_ctx['store'].edit_signature != expected_sig:
                raise EditRevisionConflictError(
                    f"Edit revision changed before repair ({edit_ctx['store'].edit_signature!r} != {expected_sig!r})"
                )

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
                    edit_u8 = edit_map.get(index)
                    protect_bool = ((edit_u8 & BIT_PROTECT_INTACT) != 0) if edit_u8 is not None else None
                    if protect_bool is not None and np.any(mask & protect_bool):
                        raise ValueError('Critical Invariant Violation: effective_mask intersects PROTECT_INTACT pixels!')

                    repaired = next(model) if model is not None else cv2.inpaint(image, mask.astype(np.uint8)*255, o['radius'], cv2.INPAINT_TELEA)
                    if repaired.shape != image.shape:
                        raise ValueError('Model output geometry mismatch.')
                    result = image.copy()
                    result[mask] = repaired[mask]
                    changed = np.any(result != image, axis=2)
                    changed_outside = int(np.count_nonzero(changed & ~mask))
                    if changed_outside != 0:
                        raise ValueError('Critical Invariant Violation: Pixels outside effective mask were modified!')
                    changed_protected = int(np.count_nonzero(changed & protect_bool)) if protect_bool is not None else 0
                    if changed_protected != 0:
                        raise ValueError('Critical Invariant Violation: PROTECT_INTACT pixels were modified!')

                    measurements.append({
                        'frame': index,
                        'changed_inside_pixels': int(np.count_nonzero(changed & mask)),
                        'changed_outside_pixels': changed_outside,
                        'changed_protected_pixels': changed_protected,
                    })
                    writer.write(result)
                    pct = int(100 * (index + 1) / max(meta['frames'], 1))
                    print(f"Dust repair frame {index + 1}/{meta['frames']} ({pct}%): repairing frame", flush=True)
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
                for n_row in native_rows:
                    if any(int(v) != 0 for k, v in n_row.items() if k.endswith('outside_samples')):
                        raise ValueError('Critical Invariant Violation: Native 10-bit composition modified outside-mask samples!')
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
                for index, (image, result, m) in enumerate(zip(frames(target/'input.mkv'), frames(tmp/'with_audio.mkv'), frames(target/'effective_mask.mkv'), strict=True)):
                    support = m[:, :, 0] > 127
                    changed = np.any(image != result, axis=2)
                    if int(np.count_nonzero(changed & ~support)) != 0:
                        raise ValueError('Critical Invariant Violation: Final output modified pixels outside effective mask!')
                    edit_u8 = edit_map.get(index)
                    if edit_u8 is not None and np.any(changed & ((edit_u8 & BIT_PROTECT_INTACT) != 0)):
                        raise ValueError('Critical Invariant Violation: Final output modified PROTECT_INTACT pixels!')
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
            prov = _build_repair_provenance(o, report, target, source)
            prov_file = tmp / 'restoration_provenance.json'
            publish_json(prov_file, prov)
            report['restoration_provenance'] = prov
            report['hashes']['restoration_provenance.json'] = digest(prov_file)
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
        keys = ('detector', 'sensitivity', 'dilate', 'threshold', 'enable_temporal_scratch')
        for other in sorted(target.parent.glob('*/report.json')):
            if other.parent.name.startswith('.'):
                continue
            try:
                old = json.loads(other.read_text(encoding='utf-8'))
                old_options = old['identity']['options']
                if old_options.get('edit_signature', '') != o.get('edit_signature', ''):
                    continue
                if old_options.get('repair_policy', 'conservative') != o.get('repair_policy', 'conservative'):
                    continue
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
                for name in ('scratch_mask.mkv', 'dust_mask.mkv', 'uncertain_mask.mkv', 'vetoed_mask.mkv'):
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
        det_name = o.get('detector', '')
        writers = {
            'input.mkv': Writer(stage / 'input.mkv', meta),
            'mask.mkv': Writer(stage / 'mask.mkv', meta),
            'effective_mask.mkv': Writer(stage / 'effective_mask.mkv', meta),
            'scratch_mask.mkv': Writer(stage / 'scratch_mask.mkv', meta),
            'dust_mask.mkv': Writer(stage / 'dust_mask.mkv', meta),
            'uncertain_mask.mkv': Writer(stage / 'uncertain_mask.mkv', meta),
        }
        if det_name == E2_DETECTOR_NAME:
            writers['vetoed_mask.mkv'] = Writer(stage / 'vetoed_mask.mkv', meta)
        if dart_images:
            writers['dart_full.mkv'] = Writer(stage / 'dart_full.mkv', meta)
        rows = []
        enable_temporal = str(o.get('enable_temporal_scratch', 'false')).lower() == 'true' and det_name not in ('temporal_scratch', E2_DETECTOR_NAME)
        target_name = 'dart_harness' if det_name == 'dart' else det_name
        strategy = DETECTOR_REGISTRY.get(target_name, commercial_only=False) if DETECTOR_REGISTRY.has(target_name, commercial_only=False) else None
        temporal_strategy = DETECTOR_REGISTRY.get('temporal_scratch', commercial_only=False) if enable_temporal else None
        is_anomaly = det_name in ('patchcore', 'padim')
        if det_name == E2_DETECTOR_NAME:
            scratch_supported = True
            dust_supported = False
            uncertain_supported = True
        elif strategy is not None:
            supp_chans = getattr(strategy, 'supported_channels', ('scratch', 'dust'))
            unsupp_chans = getattr(strategy, 'unsupported_channels', ())
            scratch_supported = (('scratch' in supp_chans) and ('scratch' not in unsupp_chans) and not is_anomaly) or (enable_temporal and not is_anomaly)
            dust_supported = ('dust' in supp_chans) and ('dust' not in unsupp_chans) and not is_anomaly
            uncertain_supported = (('uncertain' in supp_chans) and ('uncertain' not in unsupp_chans) and not is_anomaly) or (enable_temporal and not is_anomaly)
        elif is_anomaly:
            scratch_supported = False
            dust_supported = False
            uncertain_supported = False
        else:
            scratch_supported = True
            dust_supported = (det_name not in ('legacy', 'ridge'))
            uncertain_supported = enable_temporal
        iterator = masks_for(source, o) if not dart_masks else ((f, None, '') for f in frames(source))
        try:
            for index, item in enumerate(iterator):
                check_cancel(cancel_file)
                image, mask, reason = item
                h, w = image.shape[:2]
                v_mask = None

                if dart_masks:
                    raw_mask = read_mask(dart_masks / f'frame_{index:06d}.png', image.shape)
                    writers['dart_full.mkv'].write(read_restored(dart_images / f'frame_{index:06d}.png', image.shape))
                    mask = raw_mask
                    thresh = int(o.get('threshold', 160))
                    raw_bin = (mask >= thresh).astype(np.uint8) * 255
                    s_mask, d_mask = split_proposal_components(raw_bin)
                    categorized = (s_mask > 0) | (d_mask > 0)
                    uncategorized = (raw_bin > 0) & ~categorized
                    u_mask = np.where(uncategorized, np.uint8(255), np.uint8(0)).astype(np.uint8)
                    prob = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)
                else:
                    if mask is None:
                        raise ValueError(f"Detector '{det_name}' yielded None mask at frame {index}")
                    s_mask = getattr(item, 'scratch_mask', None)
                    d_mask = getattr(item, 'dust_mask', None)
                    u_mask = getattr(item, 'uncertain_mask', None)
                    v_mask = getattr(item, 'vetoed_mask', None)
                    if s_mask is None:
                        s_mask = mask.copy() if scratch_supported else np.zeros((h, w), dtype=np.uint8)
                    if d_mask is None:
                        d_mask = np.zeros((h, w), dtype=np.uint8)
                    if u_mask is None:
                        u_mask = np.zeros((h, w), dtype=np.uint8)
                    raw_bin = (mask >= 128).astype(np.uint8) * 255
                    prob = getattr(item, 'combined_probability', None)
                    if prob is None or not isinstance(prob, np.ndarray) or prob.shape != (h, w) or prob.dtype != np.float32:
                        prob = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)

                from film_defect_detectors import MultiChannelDefectMask
                try:
                    policy_enum = RepairInclusionPolicy(o.get('repair_policy', 'conservative'))
                    mcm = MultiChannelDefectMask(
                        combined_probability=prob,
                        combined_mask=raw_bin,
                        scratch_mask=((s_mask >= 128).astype(np.uint8) * 255) if (scratch_supported and s_mask is not None) else None,
                        dust_mask=((d_mask >= 128).astype(np.uint8) * 255) if (dust_supported and d_mask is not None) else None,
                        uncertain_mask=((u_mask >= 128).astype(np.uint8) * 255) if (uncertain_supported and u_mask is not None) else None,
                    )
                    effective = mcm.get_effective_repair_mask(policy_enum)
                except Exception:
                    effective = raw_bin

                if o.get('dilate'):
                    effective = cv2.dilate(effective, np.ones((3, 3), np.uint8), iterations=o['dilate'])

                mask_clean = ensure_uint8_binary_mask(mask, (h, w), "mask")
                effective_clean = ensure_uint8_binary_mask(effective, (h, w), "effective_mask")
                s_clean = ensure_uint8_binary_mask(s_mask, (h, w), "scratch_mask")
                d_clean = ensure_uint8_binary_mask(d_mask, (h, w), "dust_mask")
                u_clean = ensure_uint8_binary_mask(u_mask, (h, w), "uncertain_mask")

                writers['input.mkv'].write(image)
                writers['mask.mkv'].write(cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR))
                writers['effective_mask.mkv'].write(cv2.cvtColor(effective_clean, cv2.COLOR_GRAY2BGR))
                writers['scratch_mask.mkv'].write(cv2.cvtColor(s_clean, cv2.COLOR_GRAY2BGR))
                writers['dust_mask.mkv'].write(cv2.cvtColor(d_clean, cv2.COLOR_GRAY2BGR))
                writers['uncertain_mask.mkv'].write(cv2.cvtColor(u_clean, cv2.COLOR_GRAY2BGR))

                row_data: dict[str, Any] = {
                    'frame': index,
                    'mask_pixels': int(np.count_nonzero(mask_clean >= (o['threshold'] if dart_masks else 128))),
                    'repair_pixels': int(np.count_nonzero(effective_clean)),
                    'scratch_pixels': int(np.count_nonzero(s_clean >= 128)),
                    'dust_pixels': int(np.count_nonzero(d_clean >= 128)),
                    'uncertain_pixels': int(np.count_nonzero(u_clean >= 128)),
                }
                if 'vetoed_mask.mkv' in writers:
                    v_arr = v_mask if v_mask is not None else np.zeros((h, w), dtype=np.uint8)
                    v_clean = ensure_uint8_binary_mask(v_arr, (h, w), "vetoed_mask")
                    writers['vetoed_mask.mkv'].write(cv2.cvtColor(v_clean, cv2.COLOR_GRAY2BGR))
                    row_data['vetoed_pixels'] = int(np.count_nonzero(v_clean >= 128))
                row_data['abstention'] = reason
                row_data['note'] = 'Empty proposal may also mean detector safety rejection; not proof of intact film.'
                rows.append(row_data)

                pct = int(100 * (index + 1) / max(meta['frames'], 1))
                temporal_tag = " (temporal scratch enhancement)" if enable_temporal else ""
                print(f"Dust frame {index + 1}/{meta['frames']} ({pct}%): analyzing defects{temporal_tag}", flush=True)
        finally:
            close_writers(writers.values())
            if strategy is not None and hasattr(strategy, 'close'):
                try:
                    strategy.close()
                except Exception:
                    pass
            if temporal_strategy is not None and hasattr(temporal_strategy, 'close'):
                try:
                    temporal_strategy.close()
                except Exception:
                    pass
        write_csv(stage / 'measurements.csv', rows)
        if identity(source, o) != initial_identity or digest(source) != initial_hash:
            raise ValueError('Source changed during inference; refusing publication.')

        coverage_diag = _build_coverage_diagnostics(rows, meta)

        report = {
            'identity': initial_identity,
            'source_sha256': initial_hash,
            'metadata': meta,
            'repair_complete': False,
            'coverage_diagnostics': coverage_diag,
            'hashes': {name: digest(stage / name) for name in (*writers, 'measurements.csv')},
            'limitations': detector_limitations(o['detector']),
        }
        if o['detector'] == 'dart':
            report['model_preflight'] = preflight
            report['weights_sha256'] = 'd78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3'
        if o['detector'] == E2_DETECTOR_NAME:
            report['e2_metadata'] = {
                'detector': E2_DETECTOR_NAME,
                'version': E2_DETECTOR_VERSION,
                'architecture': E2_DETECTOR_ARCHITECTURE,
                'research_confirmed': True,
                'non_flattened_uncertainty': True,
                'three_state_outputs': [
                    'high_confidence_damage',
                    'uncertain_damage_candidate',
                    'likely_intact_scene_structure',
                ],
                'total_high_confidence_pixels': sum(int(r['scratch_pixels']) for r in rows),
                'total_uncertain_pixels': sum(int(r['uncertain_pixels']) for r in rows),
                'total_vetoed_pixels': sum(int(r.get('vetoed_pixels', 0)) for r in rows),
            }
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
    for name in ('scratch_mask.mkv', 'dust_mask.mkv', 'uncertain_mask.mkv', 'vetoed_mask.mkv', 'restoration_provenance.json'):
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
    p.add_argument('--edit-signature', default='')
    p.add_argument('--timeline-id', default='')
    p.add_argument('--ignore-edits', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--force', action='store_true')
    p.add_argument('--cancel-file', type=Path)
    a = p.parse_args()
    values = json.loads(a.settings)
    if a.edit_signature and not (a.ignore_edits and a.kind == 'mask'):
        values['edit_signature'] = a.edit_signature
    if a.timeline_id and not (a.ignore_edits and a.kind == 'mask'):
        values['timeline_id'] = a.timeline_id
    authorize(values, a.guard, a.kind)
    if not a.dry_run:
        run(a.source, values, a.kind, a.guard, a.comfy_url, a.comfy_output, a.comfy_dir, a.cancel_file, ignore_edits=a.ignore_edits)
    print('Dust & Scratches complete. Proposals and pixel changes are not quality scores.', flush=True)


if __name__ == '__main__':
    main()
