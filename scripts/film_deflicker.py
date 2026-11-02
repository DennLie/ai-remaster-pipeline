"""Conservative per-shot illumination deflicker for archival video.

No learned model, frame interpolation, denoising or detail synthesis. Small,
bounded whole-frame gains minimize changes to film texture. A two-pass scan
estimates per-frame exposure and refuses to propagate corrections across cuts.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ALGORITHM_VERSION = 1


def frame_luma(frame: np.ndarray) -> tuple[float, np.ndarray]:
    """Robust central-crop exposure and a normalized 16-bin scene histogram."""
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected one uint8 BGR video frame")
    height, width = frame.shape[:2]
    y0, y1 = height // 10, height - height // 10
    x0, x1 = width // 10, width - width // 10
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (min(width, 160), min(height, 120)), interpolation=cv2.INTER_AREA)
    valid = thumb[(thumb >= 18) & (thumb <= 242)]
    # For near-black or clipped-white frames, do not try to invent exposure.
    median = float(np.median(valid)) if valid.size >= thumb.size * 0.08 else 0.0
    # Standardize contrast before comparing histograms. A global exposure
    # flicker changes both the mean and standard deviation of intensities,
    # but should NOT be classified as a shot cut based on that change alone.
    values = thumb.astype(np.float32)
    standardized = (values - float(np.mean(values))) / max(1.0, float(np.std(values)))
    hist = np.histogram(standardized, bins=16, range=(-4.0, 4.0))[0].astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    return median, hist


def calculate_gains(
    luminances: list[float],
    histograms: list[np.ndarray],
    *,
    strength: float = 0.65,
    max_correction: float = 0.08,
    radius: int = 5,
) -> tuple[list[float], list[int]]:
    """Return bounded per-frame gain and frame numbers starting a new shot."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Deflicker strength must be in [0,1]")
    if not 0.0 <= max_correction <= 0.15:
        raise ValueError("Maximum correction must be between 0 and 0.15")
    if radius < 1 or radius > 100:
        raise ValueError("Temporal radius must be between 1 and 100")
    if len(luminances) != len(histograms):
        raise ValueError("Luminance/histogram frame count mismatch")

    boundaries = [0] if luminances else []
    for i in range(1, len(luminances)):
        previous, current = luminances[i - 1], luminances[i]
        # Do not smooth across edits, camera flashes, sudden exposure jumps
        # or significant changes in the overall tonal distribution.
        distance = float(cv2.compareHist(
            np.asarray(histograms[i-1], dtype=np.float32),
            np.asarray(histograms[i], dtype=np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        ))
        if distance > 0.35 or (
            previous > 0 and current > 0 and abs(previous-current) > 22
            and distance > 0.15
        ):
            boundaries.append(i)
    boundaries.append(len(luminances))

    gains = [1.0] * len(luminances)
    if strength == 0 or max_correction == 0:
        return gains, boundaries[:-1]
    for start, end in zip(boundaries, boundaries[1:]):
        for i in range(start, end):
            value = luminances[i]
            if value < 18.0 or value > 240.0 or end-start < 3:
                continue
            left = max(start, i-radius)
            right = min(end, i+radius+1)
            window = [(j, v) for j in range(left, right)
                      if 18.0 <= (v := luminances[j]) <= 240.0]
            if len(window) < 4:
                continue
            # Preserve real exposure ramps: monotonic sequences (fades,
            # opening a door, deliberate lighting) are not frame flicker.
            exposure = [v for _, v in window]
            steps = np.diff(exposure)
            if len(steps) >= 3 and (
                np.all(steps >= -0.5) or np.all(steps <= 0.5)
            ):
                continue
            # Excluding the current value prevents an alternating two-frame
            # flicker from making the odd-window median equal the flicker.
            neighbors = [v for j, v in window if j != i]
            if len(neighbors) < 3:
                continue
            reference = float(np.median(neighbors))
            if abs(reference - value) < 1.5:
                continue
            adjustment = strength * (reference / value - 1.0)
            gains[i] = 1.0 + float(np.clip(adjustment, -max_correction, max_correction))
    return gains, boundaries[:-1]


def apply_gain(frame: np.ndarray, gain: float) -> np.ndarray:
    """Apply uniform per-channel gain; do not filter high-frequency film texture."""
    if abs(gain - 1.0) < 1e-4:
        return frame.copy()
    lut = np.clip(np.round(np.arange(256, dtype=np.float32) * gain), 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut)


def analyze_video(source: Path, strength: float, max_correction: float, radius: int = 5) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open Deflicker video: {source}")
    luminances: list[float] = []
    histograms: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            v, hist = frame_luma(frame)
            luminances.append(v)
            histograms.append(hist)
    finally:
        capture.release()
    if not luminances:
        raise ValueError(f"No decodable video frames: {source}")
    gains, cuts = calculate_gains(
        luminances, histograms, strength=strength,
        max_correction=max_correction, radius=radius,
    )
    return {
        "algorithm": f"light_deflicker_v{ALGORITHM_VERSION}",
        "frames": len(gains),
        "cuts": cuts,
        "strength": strength,
        "max_correction": max_correction,
        "radius": radius,
        "nonneutral_frames": sum(abs(g-1) >= 1e-4 for g in gains),
        "max_applied_deviation": max(abs(g-1) for g in gains),
        "gains": gains,
    }


def prepare_light_deflicker(ffmpeg: str, source: Path, work_dir: Path, info: dict[str, Any], args: Any) -> Path:
    """Produce a cached lossless/intermediate deflickered movie, or pass source through."""
    if not args.light_deflicker:
        return source
    from common import file_fingerprint, replace_with_retry, resumable_output, root_relative, write_signature
    from intermediate_video import codec_args, container_args

    width, height = int(info["width"]), int(info["height"])
    fps = float(info.get("fps") or 24.0)
    signature = {
        "tool": "film_deflicker.py",
        "version": ALGORITHM_VERSION,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "geometry": [width, height],
        "fps": fps,
        "strength": float(args.deflicker_strength),
        "max_correction": float(args.deflicker_max_correction),
        "radius": 5,
        "intermediate_profile": args.intermediate_profile,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    output = work_dir / f"light_deflicker_{key}.mkv"
    if not args.force and resumable_output(output, signature, width=width, height=height, video_like=source):
        print(f"Reuse Light Deflicker: {output}", flush=True)
        return output
    analysis = analyze_video(source, args.deflicker_strength, args.deflicker_max_correction)
    work_dir.mkdir(parents=True, exist_ok=True)
    report = work_dir / f"light_deflicker_{key}_analysis.json"
    report.write_text(json.dumps({
        **analysis,
        "source": str(source),
        "disclaimer": "Per-frame exposure estimates are not a film-restoration quality metric.",
    }, indent=2) + "\n", encoding="utf-8")

    print(
        f"Light Deflicker: {analysis['frames']} frames, "
        f"{analysis['nonneutral_frames']} bounded corrections, "
        f"{max(0,len(analysis['cuts'])-1)} cut boundaries, "
        f"maximum gain change {analysis['max_applied_deviation']:.2%}.",
        flush=True,
    )
    if analysis["nonneutral_frames"] == 0:
        return source
    partial = output.with_suffix(".partial.mkv")
    partial.unlink(missing_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not reopen Light Deflicker input: {source}")
    encoder = None
    processed = 0
    failure: BaseException | None = None
    try:
        encoder = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s:v", f"{width}x{height}", "-r", f"{fps:.8f}", "-i", "pipe:0", "-an",
             *codec_args(args.intermediate_profile, fast=True),
             *container_args(str(partial), args.intermediate_profile), str(partial)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if encoder.stdin is None:
            raise RuntimeError("Could not open Light Deflicker FFmpeg encoder pipe")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if processed >= len(analysis["gains"]):
                raise RuntimeError("Deflicker second pass decoded more frames than the first")
            frame = apply_gain(frame, analysis["gains"][processed])
            encoder.stdin.write(frame.tobytes())
            processed += 1
    except BaseException as exc:
        failure = exc
    finally:
        capture.release()
        if encoder is not None and encoder.stdin is not None:
            encoder.stdin.close()
    if encoder is None:
        raise RuntimeError("Light Deflicker encoder did not start")
    errors = (encoder.stderr.read() if encoder.stderr else b"").decode("utf-8", errors="replace")
    if encoder.stderr is not None:
        encoder.stderr.close()
    code = encoder.wait()
    if failure is not None:
        partial.unlink(missing_ok=True)
        raise failure
    if code != 0 or processed != analysis["frames"] or not partial.exists():
        partial.unlink(missing_ok=True)
        raise RuntimeError(errors or f"Deflicker video failed; {processed} of {analysis['frames']} frames encoded")
    replace_with_retry(partial, output, "Light Deflicker")
    write_signature(output, signature)
    return output


def prepare_ffmpeg_deflicker(
    ffmpeg: str, source: Path, work_dir: Path, info: dict[str, Any], args: Any
) -> Path:
    """Build a cached FFmpeg median or arithmetic-mean deflicker variant.

    Unlike ARP's shot-aware gain model this is FFmpeg's temporal median filter.
    It can alter intentional fades and scene transitions. No gain report is
    fabricated because the filter does not apply one global per-frame gain.
    """
    from common import file_fingerprint, replace_with_retry, resumable_output, root_relative, write_signature
    from intermediate_video import codec_args, container_args

    method = getattr(args, 'deflicker_method', 'ffmpeg_median')
    if method not in ('ffmpeg_median', 'ffmpeg_am'):
        raise ValueError(f'Unsupported FFmpeg deflicker method: {method!r}')
    mode = 'median' if method == 'ffmpeg_median' else 'am'
    window = int(args.deflicker_window)
    if window not in (3, 5, 7, 9):
        raise ValueError("FFmpeg deflicker window must be 3, 5, 7 or 9")
    width, height = int(info["width"]), int(info["height"])
    fps = float(info.get("fps") or 24.0)
    signature = {
        "tool": f"film_deflicker.py/{method}",
        "version": 1,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "geometry": [width, height],
        "fps": fps,
        "window": window,
        "mode": mode,
        "intermediate_profile": args.intermediate_profile,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    output = work_dir / f"{method}_{window}_{key}.mkv"
    if not args.force and resumable_output(
        output, signature, width=width, height=height, video_like=source
    ):
        print(f"Reuse FFmpeg {mode} Deflicker: {output}", flush=True)
        return output
    work_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mkv")
    partial.unlink(missing_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(source),
        "-vf", f"deflicker=size={window}:mode={mode}",
        "-an", "-r", f"{fps:.8f}",
        *codec_args(args.intermediate_profile, fast=True),
        *container_args(str(partial), args.intermediate_profile),
        str(partial),
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not partial.is_file():
            error = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or f"FFmpeg Median failed with exit code {result.returncode}")
        replace_with_retry(partial, output, f"FFmpeg {mode} Deflicker")
        write_signature(output, signature)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    print(
        f"FFmpeg {'Median' if mode == 'median' else 'Arithmetic Mean'} Deflicker: {window}-frame temporal {mode}; "
        "not shot-aware, no global-gain analysis. Inspect cuts and fades.",
        flush=True,
    )
    return output


def prepare_ffmpeg_median_deflicker(
    ffmpeg: str, source: Path, work_dir: Path, info: dict[str, Any], args: Any
) -> Path:
    """Compatibility wrapper for callers of the original Median-only helper."""
    from argparse import Namespace
    return prepare_ffmpeg_deflicker(ffmpeg, source, work_dir, info,
                                   Namespace(**{**vars(args), "deflicker_method": "ffmpeg_median"}))
