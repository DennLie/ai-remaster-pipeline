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
from pathlib import Path

from .config import ROOT
from .frame_differences import _compare_sampled_frames, _file_identity, _validate_pair
from .frame_inspector import _decode, _metadata, _prune, _video_path
from .media import pipeline_source_text
from .paths import is_unfinished_media_artifact, resolve

_CACHE = ROOT / ".cache" / "denoise_review_differences"
_OUTPUTS = ROOT / "intermediate" / "denoised"
_REPORT_VERSION = 1


def comparison_source(app, stage: str = "denoise") -> str:
    if stage not in ("denoise", "deflicker"):
        raise ValueError("Choose Remove Noise or Light Deflicker for comparison.")
    return app.denoised_source_for_downstream() if stage == "deflicker" else pipeline_source_text(app.settings)


def review_pair(app, candidate: str, stage: str = "denoise") -> tuple[Path, Path]:
    """Enforce stage-bound candidate and signed source lineage before decoding."""
    if not candidate or len(candidate) > 2048:
        raise ValueError("Choose a finished comparison-stage output.")
    source_text = comparison_source(app, stage)
    if not source_text:
        raise FileNotFoundError("Choose the original film source.")
    source = _video_path(source_text)
    output = resolve(candidate)
    # A path existing *somewhere* under the repository is not a legal output.
    # Never permit symlinks/.. escapes or another stage's files.
    folder = ROOT / "intermediate" / "deflickered" if stage == "deflicker" else _OUTPUTS
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
            provenance.get("tool") != ("deflicker_video.py" if stage == "deflicker" else "denoise_video.py")
            or resolve(str(provenance.get("source", ""))).resolve() != source.resolve()
            or fp.get("size") != source_stat.st_size
            or fp.get("mtime_ns") != source_stat.st_mtime_ns
            or stage == "denoise" and (provenance.get("method") not in ("off", "hqdn3d", "nlmeans", "bm3d")
                                      or provenance.get("preset") not in ("gentle", "balanced", "strong"))
        ):
            raise ValueError("This result belongs to a different source or has stale provenance.")
        if stage == "deflicker":
            from .deflicker import backend, output_for
            values = {**app.settings.get("deflicker", {}), "method": provenance.get("method")}
            profile = backend.canonical_profile(app.settings.get("cloud", {}).get("intermediate_format"))
            if (provenance.get("version") != backend.VERSION or provenance.get("parameters") != backend.options(values)
                    or provenance.get("intermediate_profile") != profile
                    or resolve(output_for(source_text, values, profile)).resolve() != output.resolve()):
                raise ValueError("This Light Deflicker result has stale parameters, format or provenance.")
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid comparison-stage output provenance.") from exc
    return source, output


def inspect_denoise(app, candidate: str, frame: int, stage: str = "denoise") -> dict:
    if type(frame) is not int or frame < 0:
        raise ValueError("Frame must be a nonnegative integer.")
    if not isinstance(candidate, str):
        raise ValueError("Choose a finished comparison-stage output.")
    source_text = comparison_source(app, stage)
    source, output = review_pair(app, candidate, stage) if candidate else (_video_path(source_text), None)
    metadata = _metadata(source)
    count = _validate_pair(source, output) if output is not None else metadata["frames"]
    if frame >= count:
        raise IndexError(f"Frame {frame} exceeds total {count}.")
    before = _decode(source, frame)
    after = _decode(output, frame) if output is not None else before
    _prune()
    return {
        "stage": stage, "frame": frame, "frames": count,
        "fps": metadata["fps"], "width": metadata["width"],
        "height": metadata["height"], "before": before, "after": after,
        "exact": True,
        "method": json.loads(output.with_suffix(output.suffix + ".sig.json").read_text(encoding="utf-8"))["method"] if output is not None else "off",
        "note": "Matched decoded frame indices of the actual processing input and the selected signed output." if output is not None else "Decoded frame of the actual processing input.",
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
