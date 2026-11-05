"""Exact source-aligned review of completed Remove Noise / Light Deflicker variants.

The browser may send only a candidate already created under its stage's
output directory with a matching, finalized provenance sidecar for the active
pipeline source. No arbitrary input paths and no access to in-progress MKVs.
This is measurement, not restoration-quality or grain-preservation proof.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from .config import ROOT
from .frame_differences import _compare_sampled_frames, _file_identity, _validate_pair
from .frame_inspector import _decode, _metadata, _prune, _video_path
from .media import pipeline_source_text
from .paths import is_unfinished_media_artifact, resolve

_CACHE = ROOT / ".cache" / "denoise_review_differences"
_OUTPUTS = ROOT / "intermediate" / "denoised"
_REPORT_VERSION = 1
_READINESS_SNAPSHOT = ContextVar('comparison_readiness_snapshot', default=None)


def readiness_snapshot(function):
    """Reuse signed readiness only within one state response, never across requests."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _READINESS_SNAPSHOT.set({})
        try:
            return function(*args, **kwargs)
        finally:
            _READINESS_SNAPSHOT.reset(token)
    return wrapped


def comparison_source(app, stage: str = "denoise") -> str:
    if stage not in ("denoise", "deflicker", 'enhancement'):
        raise ValueError("Choose a supported restoration comparison stage.")
    if hasattr(app, 'processing_input'):
        return app.processing_input(stage)
    if stage == 'enhancement':
        return app.deflickered_source_for_downstream()
    return app.denoised_source_for_downstream() if stage == "deflicker" else pipeline_source_text(app.settings)


def review_pair(app, candidate: str, stage: str = "denoise", *, method: str | None = None,
                preset: str | None = None) -> tuple[Path, Path]:
    """Enforce stage-bound candidate and signed source lineage before decoding."""
    if not isinstance(candidate, str) or not candidate or len(candidate) > 2048:
        raise ValueError("Choose a finished comparison-stage output.")
    source_text = comparison_source(app, stage)
    if not source_text:
        raise FileNotFoundError("Choose the original film source.")
    source = _video_path(source_text)
    if hasattr(app, 'preprocessing_upstream'):
        for upstream in app.preprocessing_upstream(stage):
            selected_output(app, upstream)
    elif stage == 'deflicker' and str(app.settings.get('global', {}).get('denoise', 'false')) == 'true':
        selected_output(app, 'denoise')
    if stage == 'enhancement' and not hasattr(app, 'preprocessing_upstream'):
        for upstream in ('denoise', 'deflicker'):
            if app.settings.get('global', {}).get(upstream) == 'true':
                selected_output(app, upstream)
    output = resolve(candidate)
    # A path existing *somewhere* under the repository is not a legal output.
    # Never permit symlinks/.. escapes or another stage's files.
    folder = {'denoise': _OUTPUTS, 'deflicker': ROOT / 'intermediate' / 'deflickered',
              'enhancement': ROOT / 'intermediate' / 'enhanced'}[stage]
    if output.resolve().parent != folder.resolve():
        raise ValueError("Only finished ARP variants from the requested comparison stage may be reviewed.")
    if is_unfinished_media_artifact(output) or output.suffix.lower() != ".mkv":
        raise ValueError("An unfinished or unsupported comparison output cannot be inspected.")
    output = _video_path(str(output))
    if output.resolve() == source.resolve():
        raise ValueError("Original and processed source must be different.")
    sidecar = output.with_suffix(output.suffix + ".sig.json")
    if not sidecar.is_file():
        raise ValueError("This output has no completed comparison-stage provenance report.")
    try:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        fp = provenance.get("source_fingerprint") or {}
        source_stat = source.stat()
        if (
            provenance.get("tool") != {'denoise': 'denoise_video.py', 'deflicker': 'deflicker_video.py',
                                       'enhancement': 'enhancement_video.py'}[stage]
            or resolve(str(provenance.get("source", ""))).resolve() != source.resolve()
            or fp.get("size") != source_stat.st_size
            or fp.get("mtime_ns") != source_stat.st_mtime_ns
            or stage == "denoise" and (provenance.get("method") not in ("off", "hqdn3d", "nlmeans", "bm3d")
                                      or provenance.get("preset") not in ("gentle", "balanced", "strong"))
        ):
            raise ValueError("This result belongs to a different source or has stale provenance.")
        if stage in ('deflicker', 'enhancement'):
            if stage == 'deflicker':
                from . import deflicker as processing
            else:
                from . import enhancement as processing
            values = {**app.settings.get(stage, {}), "method": provenance.get("method")}
            profile = processing.backend.canonical_profile(app.settings.get("cloud", {}).get("intermediate_format"))
            if (provenance.get("version") != processing.backend.VERSION or provenance.get("parameters") != processing.backend.options(values)
                    or provenance.get("intermediate_profile") != profile
                    or resolve(processing.output_for(source_text, values, profile)).resolve() != output.resolve()):
                title = 'Light Deflicker' if stage == 'deflicker' else 'enhancement'
                raise ValueError(f'This {title} result has stale parameters, format or provenance.')
        else:
            from .deflicker import backend
            from denoise_video import ALGORITHM_VERSION
            profile = backend.canonical_profile(app.settings.get('cloud', {}).get('intermediate_format'))
            if (provenance.get('version') != ALGORITHM_VERSION or not provenance.get('intermediate_profile')
                    or backend.canonical_profile(provenance.get('intermediate_profile')) != profile):
                raise ValueError('This Remove Noise result has stale parameters, format or provenance.')
        if (method is not None and provenance.get('method') != method
                or preset is not None and provenance.get('preset') != preset):
            raise ValueError('This result belongs to a different selected variant.')
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid comparison-stage output provenance.") from exc
    return source, output


def selected_output(app, stage: str) -> Path:
    """The selected pipeline result, using the same rules as comparison variants."""
    values = app.settings.get(stage, {})
    getter = {'denoise': 'denoise_output', 'deflicker': 'deflicker_output', 'enhancement': 'enhancement_output'}.get(stage)
    if getter is None:
        raise ValueError('Choose a supported restoration comparison stage.')
    candidate = getattr(app, getter)()
    return review_pair(app, candidate, stage, method=values.get('method', 'arp' if stage == 'deflicker' else 'off'),
                       preset=values.get('preset', 'gentle') if stage == 'denoise' else None)[1]


def output_ready(app, stage: str) -> bool:
    snapshot = _READINESS_SNAPSHOT.get()
    key = (id(app), stage)
    if snapshot is not None and key in snapshot:
        return snapshot[key]
    try:
        selected_output(app, stage)
        ready = True
    except (OSError, ValueError, TypeError):
        ready = False
    if snapshot is not None:
        snapshot[key] = ready
    return ready


def comparison_pair(app, candidate: str, stage: str, reference: str = 'input') -> tuple[Path, Path | None, dict, int, int]:
    """Validate processing provenance first; reference choice never changes lineage."""
    if reference not in ('input', 'original'):
        raise ValueError('Choose stage input or original source as the comparison reference.')
    source, output = review_pair(app, candidate, stage) if candidate else (_video_path(comparison_source(app, stage)), None)
    metadata = _metadata(source)
    count = _validate_pair(source, output) if output is not None else metadata['frames']
    offset = 0
    if reference == 'original':
        from .media import source_section_state
        original = _video_path(app.settings.get('global', {}).get('source', ''))
        original_metadata = _metadata(original)
        if any(metadata[key] != original_metadata[key] for key in ('width', 'height', 'fps_fraction')):
            raise ValueError('Original comparison unavailable: geometry or frame rate differs from the stage input.')
        section = source_section_state(app.settings)
        start = section['start'] * metadata['fps'] if section['enabled'] else 0
        if abs(start - round(start)) > 0.001:
            raise ValueError('Original comparison requires a section start aligned to a source frame. Stage input comparison remains available.')
        offset = round(start)
        if original_metadata['frames'] < offset + count:
            raise ValueError('Original source does not cover the verified comparison timeline.')
        source = original
    return source, output, metadata, count, offset


def inspect_denoise(app, candidate: str, frame: int, stage: str = "denoise", reference: str = 'input') -> dict:
    if type(frame) is not int or frame < 0:
        raise ValueError("Frame must be a nonnegative integer.")
    if not isinstance(candidate, str):
        raise ValueError("Choose a finished comparison-stage output.")
    source, output, metadata, count, offset = comparison_pair(app, candidate, stage, reference)
    if frame >= count:
        raise IndexError(f"Frame {frame} exceeds total {count}.")
    before = _decode(source, frame + offset)
    after = _decode(output, frame) if output is not None else before
    _prune()
    return {
        "stage": stage, "reference": reference, "source_frame": frame + offset, "frame": frame, "frames": count,
        "fps": metadata["fps"], "width": metadata["width"],
        "height": metadata["height"], "before": before, "after": after,
        "exact": True,
        "method": json.loads(output.with_suffix(output.suffix + ".sig.json").read_text(encoding="utf-8"))["method"] if output is not None else "off",
        "note": f"Frame-aligned {reference} reference; processing provenance is validated independently.",
    }


def denoise_frame_differences(app, candidate: str, stage: str = "denoise") -> dict:
    source, output = review_pair(app, candidate, stage)
    identity = {
        "version": _REPORT_VERSION,
        "source": _file_identity(source),
        "output": _file_identity(output),
    }
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = _CACHE / (token + ".json")
    if path.is_file():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("identity") == identity and cached.get("report", {}).get("metric") == "directional_mean_luma_8bit":
                return cached["report"]
        except (OSError, ValueError, TypeError):
            pass
    count = _validate_pair(source, output)
    report = _compare_sampled_frames(source, output, count)
    _CACHE.mkdir(parents=True, exist_ok=True)
    temp = _CACHE / f"{token}.{os.getpid()}.tmp"
    try:
        temp.write_text(json.dumps({"identity": identity, "report": report}), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return report
