"""Source-bound per-frame luminance differences for FFmpeg Clean Up variants.

Unlike ARP Conservative gains, FFmpeg's deflicker does not export per-frame
gain factors. These measurements describe *output pixel changes*, including
possible codec differences, NOT applied exposure corrections or quality gains.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT
from .frame_inspector import _metadata, _sources

_CACHE = ROOT / ".cache" / "cleanup_frame_differences"
_REPORT_VERSION = 3
_MAX_FRAMES = 2400
_MAX_SECONDS = 30.0
_MAX_BARS = 240


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _validate_pair(source: Path, output: Path) -> int:
    if source.resolve() == output.resolve():
        raise ValueError("Original and result must be separate files.")
    before = _metadata(source)
    after = _metadata(output)
    for key in ("width", "height", "fps_fraction", "frames"):
        if before[key] != after[key]:
            raise ValueError(f"Frame-difference videos have incompatible {key}.")
    count = int(before["frames"])
    if count < 1:
        raise ValueError("Cannot analyze a video with no verified frames.")
    return count


def _directional_luma(y_before: np.ndarray, y_after: np.ndarray) -> tuple[float, float]:
    """Per-pixel brightening and darkening, each averaged over ALL frame pixels.

    Both are nonnegative and can coexist within one frame. Their sum is mean
    absolute luma change. Pixel values describe differences, not FFmpeg gains.
    """
    if y_before.shape != y_after.shape or y_before.dtype != np.uint8 or y_after.dtype != np.uint8:
        raise ValueError("Expected aligned 8-bit grayscale frames.")
    diff = y_after.astype(np.int16) - y_before.astype(np.int16)
    bright = float(np.maximum(diff, 0).mean())
    dark = float(np.maximum(-diff, 0).mean())
    return bright, dark


def _compare_frames(source: Path, output: Path, frames: int) -> dict:
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(output))
    try:
        if not a.isOpened() or not b.isOpened():
            raise RuntimeError("Unable to open both videos for frame-difference analysis.")
        stride = max(1, math.ceil(frames / _MAX_BARS))
        buckets: list[dict] = []
        current: dict | None = None
        total = 0.0
        total_bright = 0.0
        total_dark = 0.0
        largest = 0.0
        largest_bright = 0.0
        largest_dark = 0.0
        start = time.monotonic()
        for frame in range(frames):
            if time.monotonic() - start > _MAX_SECONDS:
                raise RuntimeError("Frame-difference analysis exceeded its time budget.")
            ok0, im0 = a.read()
            ok1, im1 = b.read()
            if not ok0 or not ok1 or im0 is None or im1 is None:
                raise ValueError(f"Frame-difference decode ended before frame {frame+1}/{frames}.")
            if im0.shape != im1.shape or im0.ndim != 3 or im0.shape[2] != 3:
                raise ValueError("Videos have incompatible decoded frame geometries.")
            # For each fully decoded matched ordinal, average absolute brightness
            # difference (BT.601 OpenCV 8-bit luma), units 0..255.
            y0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            y1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            bright, dark = _directional_luma(y0, y1)
            score = bright + dark
            total += score
            total_bright += bright
            total_dark += dark
            largest = max(largest, score)
            largest_bright = max(largest_bright, bright)
            largest_dark = max(largest_dark, dark)
            candidate = {
                "frame": frame,
                "mean_abs_luma": round(score, 4),
                "brighten_mean_luma": round(bright, 4),
                "darken_mean_luma": round(dark, 4),
            }
            if frame % stride == 0:
                if current is not None:
                    buckets.append(current)
                current = candidate
            elif current is not None and score > current["mean_abs_luma"]:
                current = candidate
        if current is not None:
            buckets.append(current)
        # Reject hidden additional frames even if metadata was incorrect.
        if a.grab() or b.grab():
            raise ValueError("Decoded video exceeds the verified frame count.")
        return {
            "frames": frames,
            "samples": buckets,
            "sampled": stride > 1,
            "mean_abs_luma": round(total / frames, 4),
            "mean_brighten_luma": round(total_bright / frames, 4),
            "mean_darken_luma": round(total_dark / frames, 4),
            "max_mean_abs_luma": round(largest, 4),
            "max_brighten_luma": round(largest_bright, 4),
            "max_darken_luma": round(largest_dark, 4),
            "metric": "directional_mean_luma_8bit",
        }
    finally:
        a.release()
        b.release()


def _compare_sampled_frames(source: Path, output: Path, frames: int) -> dict:
    """Sample up to 240 exact frame ordinals, never decode the whole film.

    A sparse chart is not a per-frame maximum or a full-film mean.
    Fail closed if backend random seek cannot verify the requested ordinal.
    """
    if frames <= _MAX_FRAMES:
        return _compare_frames(source, output, frames)
    positions = sorted({round(i * (frames-1) / (_MAX_BARS-1)) for i in range(_MAX_BARS)})
    a, b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(output))
    samples = []
    start = time.monotonic()
    try:
        if not a.isOpened() or not b.isOpened():
            raise RuntimeError("Unable to open videos for sampled frame analysis.")
        for index in positions:
            if time.monotonic() - start > 300:
                raise RuntimeError("Sampled film chart exceeded compute budget; try a short section.")
            images = []
            for capture in (a,b):
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                    raise RuntimeError("Frame-accurate seek is not supported by the selected video.")
                ok, image = capture.read()
                if not ok or image is None:
                    raise ValueError(f"Cannot decode chart sample {index+1}/{frames}.")
                # OpenCV reports the *next* frame position after decoding. Skip
                # this sample rather than silently compare wrong timestamps.
                after = capture.get(cv2.CAP_PROP_POS_FRAMES)
                if not math.isfinite(after) or abs(after-(index+1)) > 0.5:
                    raise ValueError(
                        f"Video seek did not resolve exact frame {index} (position {after})."
                    )
                images.append(image)
            im0, im1 = images
            if im0.shape != im1.shape:
                raise ValueError(f"Different video shapes at frame {index}.")
            y0 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            y1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
            bright, dark = _directional_luma(y0, y1)
            samples.append({
                "frame": index,
                "mean_abs_luma": round(bright+dark,4),
                "brighten_mean_luma": round(bright,4),
                "darken_mean_luma": round(dark,4)
            })
        return {
            "frames":frames, "samples":samples, "sampled":True,
            "sample_count":len(samples),
            "sampling_note":"Sparse exact frame samples; NOT full-film statistics or within-block maxima.",
            "mean_abs_luma":round(sum(x["mean_abs_luma"] for x in samples)/len(samples),4),
            "mean_brighten_luma":round(sum(x["brighten_mean_luma"] for x in samples)/len(samples),4),
            "mean_darken_luma":round(sum(x["darken_mean_luma"] for x in samples)/len(samples),4),
            "max_mean_abs_luma":max(x["mean_abs_luma"] for x in samples),
            "max_brighten_luma":max(x["brighten_mean_luma"] for x in samples),
            "max_darken_luma":max(x["darken_mean_luma"] for x in samples),
            "metric":"directional_mean_luma_8bit",
        }
    finally:
        a.release()
        b.release()


def cleanup_frame_differences(app) -> dict:
    """Measure only the app-selected cleanup result; never accept client paths."""
    comparison = app.cleanup_comparison_state()
    if comparison.get("exists") != "true":
        raise FileNotFoundError("Run the selected Clean Up method first.")
    if comparison.get("method") not in ("ffmpeg_median", "ffmpeg_am", "localized_research", "localized_stream"):
        raise ValueError("Pixel-difference bars require a rendered non-ARP deflicker variant.")
    source, output = _sources(app, "cleanup")
    identity = {
        "version": _REPORT_VERSION,
        "source": _file_identity(source),
        "output": _file_identity(output),
        "method": comparison["method"],
    }
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = _CACHE / f"{token}.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("identity") == identity and cached.get("report", {}).get("metric") == "directional_mean_luma_8bit":
                return cached["report"]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    count = _validate_pair(source, output)
    report = _compare_sampled_frames(source, output, count)
    _CACHE.mkdir(parents=True, exist_ok=True)
    temp = _CACHE / f"{token}.{os.getpid()}.tmp"
    try:
        temp.write_text(json.dumps({"identity": identity, "report": report}, separators=(",", ":")), encoding="utf-8")
        os.replace(temp, cache_path)
    finally:
        temp.unlink(missing_ok=True)
    return report
