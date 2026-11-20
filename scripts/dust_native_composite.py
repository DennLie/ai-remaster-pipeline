"""Lossless 10-bit composition; only declared native-plane support is replaced.

Models still produce BGR8. This preserves native source samples elsewhere, not
10-bit model detail. Chroma support is the union of each subsampled pixel block.
"""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

try:
    from film_defect_detectors import (
        DEFAULT_REGISTRY as DETECTOR_REGISTRY,
        ChannelOrigin,
        ChannelProvenance,
        MultiChannelDefectMask,
        RepairInclusionPolicy,
        UnsupportedPolicyError,
    )
except ImportError:
    from scripts.film_defect_detectors import (
        DEFAULT_REGISTRY as DETECTOR_REGISTRY,
        ChannelOrigin,
        ChannelProvenance,
        MultiChannelDefectMask,
        RepairInclusionPolicy,
        UnsupportedPolicyError,
    )



FORMATS = {'yuv420p10le': (2, 2), 'yuv422p10le': (2, 1),
           'yuv444p10le': (1, 1), 'gbrp10le': (1, 1), 'gray10le': None}


def plane_shapes(meta):
    fmt = meta['pixel_format']
    if fmt not in FORMATS:
        raise ValueError('Unsupported native repair pixel format.')
    h, w = meta['height'], meta['width']
    chroma = FORMATS[fmt]
    return [(h, w)] if chroma is None else [(h, w)] + [((h+chroma[1]-1)//chroma[1], (w+chroma[0]-1)//chroma[0])] * 2


def plane_masks(mask, meta):
    shapes = plane_shapes(meta)
    result = [mask.astype(bool)]
    for h, w in shapes[1:]:
        sx = (meta['width'] + w-1)//w
        sy = (meta['height'] + h-1)//h
        padded = np.pad(mask, ((0, h*sy-mask.shape[0]), (0, w*sx-mask.shape[1])), constant_values=False)
        result.append(padded.reshape(h, sy, w, sx).any(axis=(1, 3)))
    return result


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size-len(data))
        if not chunk:
            raise ValueError('Native frame stream ended early.')
        data.extend(chunk)
    return bytes(data)


def composite_frame(original, patch, mask, meta):
    shapes = plane_shapes(meta)
    supports = plane_masks(mask, meta)
    chunks, counts, offset = [], [], 0
    for (h, w), support in zip(shapes, supports, strict=True):
        size = h*w*2
        before = np.frombuffer(original[offset:offset+size], '<u2').reshape(h, w)
        replacement = np.frombuffer(patch[offset:offset+size], '<u2').reshape(h, w)
        composed = before.copy()
        composed[support] = replacement[support]
        changed = composed != before
        counts.append({'support_samples': int(support.sum()), 'changed_inside_samples': int((changed & support).sum()),
                       'changed_outside_samples': int((changed & ~support).sum())})
        chunks.append(composed.tobytes())
        offset += size
    return b''.join(chunks), counts


def compose(ffmpeg, source, patch, mask_path, output, meta):
    """Stream native samples, verify encoded raster digest, retain source audio."""
    fmt = meta['pixel_format']
    size = sum(h*w*2 for h,w in plane_shapes(meta))
    with ExitStack() as stack:
        def process(args, **kwargs):
            log = stack.enter_context(tempfile.TemporaryFile())
            p = subprocess.Popen([ffmpeg, '-v', 'error', *args], stderr=log, **kwargs)
            def close():
                if p.poll() is None:
                    p.kill()
                p.wait()
                for stream in (p.stdin, p.stdout):
                    if stream:
                        stream.close()
            stack.callback(close)
            return p
        common = ['-map', '0:v:0', '-an', '-pix_fmt', fmt, '-f', 'rawvideo', 'pipe:1']
        base = process(['-i', str(source), *common], stdout=subprocess.PIPE)
        filters = []
        if fmt.startswith('yuv'):
            matrices = {'bt709':'bt709', 'bt470bg':'bt601', 'smpte170m':'bt601', 'bt2020nc':'bt2020'}
            space = meta.get('color_space', 'unknown')
            if space not in (*matrices, 'unknown'):
                raise ValueError('Unsupported source colour matrix; no implicit conversion.')
            filters = ['-vf', 'scale=out_color_matrix=' + matrices.get(space, 'bt601') + ':out_range=' + ('pc' if meta.get('color_range') == 'pc' else 'tv')]
        replacement = process(['-i', str(patch), *filters, *common], stdout=subprocess.PIPE)
        colour = []
        for field, flag in (('color_range','-color_range'), ('color_space','-colorspace'), ('transfer','-color_trc'), ('color_primaries','-color_primaries')):
            value = meta.get(field, 'unknown')
            if value not in ('unknown', 'unspecified', None):
                colour += [flag, value]
        # Tag the raw INPUT too: output-only colour tags can make FFmpeg insert
        # an unwanted range/matrix conversion into otherwise identical samples.
        writer = process(['-n', '-f', 'rawvideo', '-pixel_format', fmt, '-video_size', f"{meta['width']}x{meta['height']}",
            '-framerate', meta['fps'], *colour, '-i', 'pipe:0', '-i', str(source), '-map', '0:v:0', '-map', '1:a?',
            '-c:v', 'ffv1', '-pix_fmt', fmt, *colour, '-c:a', 'copy', str(output)], stdin=subprocess.PIPE)
        capture = cv2.VideoCapture(str(mask_path))
        stack.callback(capture.release)
        expected = hashlib.sha256()
        rows = []
        for frame in range(meta['frames']):
            ok, mask = capture.read()
            if not ok:
                raise ValueError('Effective mask ended early.')
            raw, counts = composite_frame(read_exact(base.stdout, size), read_exact(replacement.stdout, size), mask[:, :, 0] > 127, meta)
            writer.stdin.write(raw)
            expected.update(raw)
            rows.append({'frame': frame, **{f'plane{i}_{key}': value for i, c in enumerate(counts) for key,value in c.items()}})
        if base.stdout.read(1) or replacement.stdout.read(1) or capture.read()[0]:
            raise ValueError('Extra native/mask frames.')
        writer.stdin.close()
        if any(p.wait() for p in (base, replacement, writer)):
            raise ValueError('Native composition process failed.')
        verified = process(['-i', str(output), *common], stdout=subprocess.PIPE)
        actual = hashlib.sha256()
        while chunk := verified.stdout.read(1024*1024):
            actual.update(chunk)
        if verified.wait() or actual.digest() != expected.digest():
            raise ValueError('Encoded native samples differ from the verified composite.')
        return rows


def compute_file_sha256(path: Path | str) -> str:
    """Calculate SHA-256 hex digest of a file, compatible with Python < 3.11."""
    p = Path(path)
    with p.open('rb') as f:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(f, 'sha256').hexdigest()
        h = hashlib.sha256()
        while chunk := f.read(65536):
            h.update(chunk)
        return h.hexdigest()


def compute_mask_sha256(mask: np.ndarray) -> str:
    """Calculate SHA-256 hex digest of a numpy mask array."""
    return hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest()


def build_restoration_provenance(
    detector: str | dict[str, Any],
    preset: str | dict[str, Any] | None,
    repair_policy: RepairInclusionPolicy | str,
    repair_backend: str,
    effective_mask_hash: str,
    source_video: Path | str,
    output_video: Path | str,
    checkpoint_sha256: str = "",
    license_info: str | dict[str, Any] | None = None,
    channel_provenance: dict[str, Any] | None = None,
    resolved_parameters: dict[str, Any] | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Construct complete restoration provenance manifest according to Phase 9 specification.

    Fields:
      - detector: detector name, version, architecture
      - checkpoint_sha256: model weights hash
      - license: license status and commercial compatibility
      - preset: preset ID and version
      - repair_policy: resolved RepairInclusionPolicy
      - channel_provenance: native vs derived channel origins
      - repair_backend: 'telea', 'propainter', or 'dart_composite'
      - effective_mask_hash: SHA-256 of the effective repair mask used
      - source_video: source clip ID/path
      - output_video: restored video artifact ID/path
      - timestamp: ISO-8601
    """
    if isinstance(detector, str):
        det_name = detector
        det_obj = DETECTOR_REGISTRY.get(det_name, commercial_only=False) if DETECTOR_REGISTRY.has(det_name, commercial_only=False) else None
        det_version = getattr(det_obj, 'version', '1.0.0') if det_obj else '1.0.0'
        det_arch = getattr(det_obj, 'architecture', 'Classical Defect Detector') if det_obj else 'Classical Defect Detector'
        detector_data = {
            'name': det_name,
            'version': str(det_version),
            'architecture': str(det_arch),
        }
    elif isinstance(detector, dict):
        detector_data = {
            'name': str(detector.get('name', 'unknown')),
            'version': str(detector.get('version', '1.0.0')),
            'architecture': str(detector.get('architecture', 'Classical Defect Detector')),
        }
    else:
        detector_data = {'name': 'unknown', 'version': '1.0.0', 'architecture': 'unknown'}

    if isinstance(preset, str):
        preset_data = {'id': preset, 'version': '1.0.0'}
    elif isinstance(preset, dict):
        preset_data = {
            'id': str(preset.get('id', preset.get('name', 'custom'))),
            'version': str(preset.get('version', '1.0.0')),
        }
    elif preset is not None and hasattr(preset, 'name'):
        preset_data = {'id': getattr(preset, 'name', 'custom'), 'version': '1.0.0'}
    else:
        preset_data = {'id': 'custom', 'version': '1.0.0'}

    if isinstance(license_info, str):
        lic_stat = license_info
        is_comm = lic_stat.lower() in ('apache-2.0', 'mit', 'bsd-3-clause', 'commercial_cleared', 'unencumbered', 'commercial-safe')
        license_data = {'status': lic_stat, 'commercial_compatible': is_comm}
    elif isinstance(license_info, dict):
        license_data = {
            'status': str(license_info.get('status', 'commercial_cleared')),
            'commercial_compatible': bool(license_info.get('commercial_compatible', True)),
        }
    else:
        license_data = {'status': 'commercial_cleared', 'commercial_compatible': True}

    if isinstance(repair_policy, RepairInclusionPolicy):
        policy_str = repair_policy.value
    else:
        policy_str = str(repair_policy).lower()

    norm_backend = str(repair_backend).lower()
    if norm_backend not in ('telea', 'propainter', 'dart_composite', 'dart'):
        raise ValueError(f"Unsupported repair backend: '{repair_backend}'. Choose telea, propainter, or dart_composite.")
    if norm_backend == 'dart':
        norm_backend = 'dart_composite'

    prov_channels = {}
    if channel_provenance:
        for ch, origin in channel_provenance.items():
            if isinstance(origin, ChannelProvenance):
                prov_channels[ch] = origin.origin.value if hasattr(origin.origin, 'value') else str(origin.origin)
            elif isinstance(origin, ChannelOrigin):
                prov_channels[ch] = origin.value
            else:
                prov_channels[ch] = str(origin)
    else:
        prov_channels = {'scratch': 'native', 'dust': 'derived', 'uncertain': 'derived'}

    ts = timestamp or datetime.now(timezone.utc).isoformat()

    manifest_data = {
        'detector': detector_data,
        'checkpoint_sha256': str(checkpoint_sha256 or ''),
        'license': license_data,
        'preset': preset_data,
        'repair_policy': policy_str,
        'channel_provenance': prov_channels,
        'repair_backend': norm_backend,
        'effective_mask_hash': str(effective_mask_hash),
        'source_video': str(source_video),
        'output_video': str(output_video),
        'timestamp': ts,
    }
    if resolved_parameters:
        manifest_data['resolved_parameters'] = resolved_parameters
    return manifest_data


def write_restoration_provenance(output_path: Path | str, provenance_data: dict[str, Any]) -> Path:
    """Persist restoration provenance JSON file."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out.with_suffix('.partial.json')
    temp_file.write_text(json.dumps(provenance_data, indent=2), encoding='utf-8')
    temp_file.replace(out)
    return out


def compute_restoration_cache_identity(
    source: Path | str,
    detector: str | dict[str, Any],
    checkpoint_sha256: str,
    preset: str | dict[str, Any],
    repair_policy: str | RepairInclusionPolicy,
    repair_backend: str,
    effective_mask_hash: str,
    extra_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate cache identity for restored video artifact incorporating all required invariant fields."""
    det_id = detector['name'] if isinstance(detector, dict) else str(detector)
    preset_id = preset['id'] if isinstance(preset, dict) else str(preset)
    policy_str = repair_policy.value if isinstance(repair_policy, RepairInclusionPolicy) else str(repair_policy).lower()
    backend_str = 'dart_composite' if repair_backend == 'dart' else str(repair_backend).lower()

    identity = {
        'source': str(source),
        'detector': det_id,
        'checkpoint_sha256': str(checkpoint_sha256 or ''),
        'preset': preset_id,
        'repair_policy': policy_str,
        'repair_backend': backend_str,
        'effective_mask_hash': str(effective_mask_hash),
    }
    if extra_options:
        identity['options'] = extra_options
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity['cache_key'] = cache_key
    return identity


def inpaint_frame_telea(
    image: np.ndarray,
    effective_repair_mask: np.ndarray,
    radius: int = 3,
) -> tuple[np.ndarray, dict[str, int]]:
    """Inpaint a single frame with OpenCV Telea, strictly gated by effective_repair_mask.

    Invariant: Pixels outside the mask are 100% bit-for-bit identical to the source image.
    """
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError(f"image must be uint8 3D (H, W, 3), got shape {image.shape}, dtype {image.dtype}")
    h, w = image.shape[:2]

    if effective_repair_mask.ndim == 3:
        mask_2d = effective_repair_mask[:, :, 0]
    elif effective_repair_mask.ndim == 2:
        mask_2d = effective_repair_mask
    else:
        raise ValueError(f"effective_repair_mask must be 2D or 3D, got {effective_repair_mask.ndim}D")

    if mask_2d.shape != (h, w):
        raise ValueError(f"Shape mismatch: image {(h, w)} vs mask {mask_2d.shape}")

    mask_bin = (mask_2d > 127).astype(np.uint8) * 255
    support = mask_bin > 0

    if not np.any(support):
        return image.copy(), {'changed_inside_pixels': 0, 'changed_outside_pixels': 0}

    inpainted = cv2.inpaint(image, mask_bin, int(radius), cv2.INPAINT_TELEA)

    repaired = image.copy()
    repaired[support] = inpainted[support]

    changed = np.any(repaired != image, axis=2)
    counts = {
        'changed_inside_pixels': int(np.count_nonzero(changed & support)),
        'changed_outside_pixels': int(np.count_nonzero(changed & ~support)),
    }
    if counts['changed_outside_pixels'] != 0:
        raise ValueError('Critical Invariant Violation: Pixels outside effective mask were modified!')
    return repaired, counts


def inpaint_frame_propainter(
    image: np.ndarray,
    patch: np.ndarray,
    effective_repair_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply ProPainter inpainting patch strictly gated by effective_repair_mask."""
    if image.shape != patch.shape:
        raise ValueError(f"Geometry mismatch: image {image.shape} vs patch {patch.shape}")
    h, w = image.shape[:2]
    mask_2d = effective_repair_mask[:, :, 0] if effective_repair_mask.ndim == 3 else effective_repair_mask
    if mask_2d.shape != (h, w):
        raise ValueError(f"Shape mismatch: image {(h, w)} vs mask {mask_2d.shape}")

    mask_bin = (mask_2d > 127).astype(np.uint8) * 255
    support = mask_bin > 0

    repaired = image.copy()
    repaired[support] = patch[support]

    changed = np.any(repaired != image, axis=2)
    counts = {
        'changed_inside_pixels': int(np.count_nonzero(changed & support)),
        'changed_outside_pixels': int(np.count_nonzero(changed & ~support)),
    }
    if counts['changed_outside_pixels'] != 0:
        raise ValueError('Critical Invariant Violation: Pixels outside effective mask were modified!')
    return repaired, counts


def inpaint_frame_dart_composite(
    image: np.ndarray,
    dart_full: np.ndarray,
    effective_repair_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Apply DART composite restoration strictly gated by effective_repair_mask."""
    return inpaint_frame_propainter(image, dart_full, effective_repair_mask)


def dispatch_repair_frame(
    image: np.ndarray,
    effective_repair_mask: np.ndarray,
    repair_backend: str = 'telea',
    radius: int = 3,
    model_patch: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Single frame dispatch to selected repair engine."""
    norm_backend = 'dart_composite' if repair_backend == 'dart' else str(repair_backend).lower()
    if norm_backend == 'telea':
        return inpaint_frame_telea(image, effective_repair_mask, radius=radius)
    elif norm_backend == 'propainter':
        if model_patch is None:
            raise ValueError('ProPainter repair requires model_patch frame.')
        return inpaint_frame_propainter(image, model_patch, effective_repair_mask)
    elif norm_backend == 'dart_composite':
        if model_patch is None:
            raise ValueError('DART composite repair requires model_patch (dart_full) frame.')
        return inpaint_frame_dart_composite(image, model_patch, effective_repair_mask)
    else:
        raise ValueError(f"Unsupported repair backend: '{repair_backend}'")


def execute_repair(
    source_frames: Sequence[np.ndarray] | np.ndarray,
    effective_repair_mask: Sequence[np.ndarray] | np.ndarray,
    repair_backend: str = 'telea',
    radius: int = 3,
    model_patches: Sequence[np.ndarray] | np.ndarray | None = None,
    output_path: Path | str | None = None,
    provenance_metadata: dict[str, Any] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Execute complete repair across frame sequence with mask gating and write provenance."""
    if isinstance(source_frames, np.ndarray) and source_frames.ndim == 3:
        source_seq = [source_frames]
    else:
        source_seq = list(source_frames)

    if isinstance(effective_repair_mask, np.ndarray) and effective_repair_mask.ndim in (2, 3) and len(source_seq) == 1:
        mask_seq = [effective_repair_mask]
    else:
        mask_seq = list(effective_repair_mask)

    if len(source_seq) != len(mask_seq):
        raise ValueError(f"Frame count mismatch: {len(source_seq)} source frames vs {len(mask_seq)} mask frames")

    patch_seq = list(model_patches) if model_patches is not None else [None] * len(source_seq)
    if len(patch_seq) != len(source_seq):
        raise ValueError(f"Model patch count mismatch: {len(patch_seq)} vs {len(source_seq)} source frames")

    repaired_frames = []
    total_inside = 0
    total_outside = 0
    for img, m, p in zip(source_seq, mask_seq, patch_seq, strict=True):
        repaired, counts = dispatch_repair_frame(
            img, m, repair_backend=repair_backend, radius=radius, model_patch=p
        )
        repaired_frames.append(repaired)
        total_inside += counts['changed_inside_pixels']
        total_outside += counts['changed_outside_pixels']

    provenance = {}
    if output_path and provenance_metadata:
        out = Path(output_path)
        provenance = dict(provenance_metadata)
        if 'effective_mask_hash' not in provenance and mask_seq:
            all_mask_bytes = b''.join(np.ascontiguousarray(m).tobytes() for m in mask_seq)
            provenance['effective_mask_hash'] = hashlib.sha256(all_mask_bytes).hexdigest()
        prov_file = out.parent / 'restoration_provenance.json'
        write_restoration_provenance(prov_file, provenance)

    summary = {
        'frames': len(repaired_frames),
        'total_changed_inside_pixels': total_inside,
        'total_changed_outside_pixels': total_outside,
        'provenance': provenance,
    }
    return repaired_frames, summary

