"""Exact source-aligned review of selected, completed Remove Noise variants.

The browser may send only a candidate already created under ARP's denoise
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


def review_pair(app, candidate: str) -> tuple[Path, Path]:
    """Enforce stage-bound candidate and signed source lineage before decoding."""
    if not candidate or len(candidate) > 2048:
        raise ValueError("Choose a finished Remove Noise output (B or C).")
    source_text = pipeline_source_text(app.settings)
    if not source_text:
        raise FileNotFoundError("Choose the original film source.")
    source = _video_path(source_text)
    output = resolve(candidate)
    # A path existing *somewhere* under the repository is not a legal output.
    # Never permit symlinks/.. escapes or another stage's files.
    if output.resolve().parent != _OUTPUTS.resolve():
        raise ValueError("Only finished ARP Remove Noise variants may be reviewed.")
    if is_unfinished_media_artifact(output) or output.suffix.lower() != ".mkv":
        raise ValueError("An unfinished or unsupported Remove Noise output cannot be inspected.")
    output = _video_path(str(output))
    if output.resolve() == source.resolve():
        raise ValueError("Original and processed source must be different.")
    sidecar = output.with_suffix(output.suffix + ".sig.json")
    if not sidecar.is_file():
        raise ValueError("This output has no completed denoise provenance report.")
    try:
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        fp = provenance.get("source_fingerprint") or {}
        source_stat = source.stat()
        if (
            provenance.get("tool") != "denoise_video.py"
            or resolve(str(provenance.get("source", ""))).resolve() != source.resolve()
            or fp.get("size") != source_stat.st_size
            or fp.get("mtime_ns") != source_stat.st_mtime_ns
            or provenance.get("method") not in ("off", "hqdn3d", "nlmeans", "bm3d")
            or provenance.get("preset") not in ("gentle", "balanced", "strong")
        ):
            raise ValueError("This result belongs to a different source or has stale provenance.")
    except (OSError, json.JSONDecodeError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid Remove Noise output provenance.") from exc
    return source, output


def inspect_denoise(app, candidate: str, frame: int) -> dict:
    if not isinstance(frame, int) or frame < 0:
        raise ValueError("Frame must be a nonnegative integer.")
    source, output = review_pair(app, candidate)
    count = _validate_pair(source, output)
    if frame >= count:
        raise IndexError(f"Frame {frame} exceeds total {count}.")
    before, after = _decode(source, frame), _decode(output, frame)
    _prune()
    metadata = _metadata(source)
    return {
        "stage": "denoise", "frame": frame, "frames": count,
        "fps": metadata["fps"], "width": metadata["width"],
        "height": metadata["height"], "before": before, "after": after,
        "exact": True,
        "method": json.loads(output.with_suffix(output.suffix + ".sig.json").read_text(encoding="utf-8"))["method"],
        "note": "Matched decoded frame indices of the actual Remove Noise processing input and the selected signed output.",
    }


def denoise_frame_differences(app, candidate: str) -> dict:
    source, output = review_pair(app, candidate)
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
