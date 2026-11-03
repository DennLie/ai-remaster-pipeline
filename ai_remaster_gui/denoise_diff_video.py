"""Rendered live pixel-difference previews for the Remove Noise A/B/C view.

The browser cannot lock two independently decoded H.264 previews to the same
frame. This module renders a stable difference video from the same preview
clips the panels play, so playback shows a real per-frame difference instead of
a motion-offset canvas flicker.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import threading
from pathlib import Path

import cv2
import numpy as np

from .config import ROOT
from .media import local_tool, media_clip_path
from .paths import resolve

_VERSION = 1
_MAX_SECONDS = 12.0
_MAX_WIDTH = 960
_CACHE = ROOT / ".cache" / "denoise_diff_previews"
_LOCK_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


def _lock_for(token: str) -> threading.Lock:
    with _LOCK_GUARD:
        return _LOCKS.setdefault(token, threading.Lock())


def _even(value: float) -> int:
    return max(2, (int(round(value)) // 2) * 2)


def _identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _token(first: Path, second: Path, start: float, end: float) -> str:
    payload = {
        "version": _VERSION,
        "first": _identity(first),
        "second": _identity(second),
        "start": round(start, 3),
        "end": round(end, 3),
        "max_width": _MAX_WIDTH,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def diff_video_path(token: str) -> Path | None:
    """Return a validated cached diff video; never accept a filesystem path."""
    if not re.fullmatch(r"[0-9a-f]{64}", str(token or "")):
        return None
    path = _CACHE / f"{token}.mp4"
    return path if path.is_file() and path.stat().st_size > 0 else None


def _diff_frame(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Match the browser's preview mapping: 4x channel delta -> magenta."""
    if before.shape != after.shape:
        raise ValueError("Preview clips have incompatible frame geometry.")
    delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
    strength = np.clip(
        (delta[:, :, 0].astype(np.int32) +
         delta[:, :, 1].astype(np.int32) +
         delta[:, :, 2].astype(np.int32)) * 4 // 3,
        0, 255,
    ).astype(np.uint8)
    out = np.empty_like(before)
    out[:, :, 0] = (strength.astype(np.uint16) * 3 // 5).astype(np.uint8)  # blue
    out[:, :, 1] = 0                                                       # green
    out[:, :, 2] = strength                                                # red
    return out


def _render(first_clip: Path, second_clip: Path, target: Path, token: str) -> None:
    partial = target.with_suffix(".partial.mp4")
    if partial.exists():
        partial.unlink()
    first = cv2.VideoCapture(str(first_clip))
    second = cv2.VideoCapture(str(second_clip))
    process: subprocess.Popen | None = None
    try:
        if not first.isOpened() or not second.isOpened():
            raise RuntimeError("Could not open both preview clips for the difference video.")
        width = int(round(first.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(first.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        second_width = int(round(second.get(cv2.CAP_PROP_FRAME_WIDTH)))
        second_height = int(round(second.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width < 2 or height < 2 or (width, height) != (second_width, second_height):
            raise ValueError("Preview clips have incompatible dimensions.")
        fps = float(first.get(cv2.CAP_PROP_FPS) or second.get(cv2.CAP_PROP_FPS) or 0)
        if not (fps > 0 and fps <= 240):
            raise ValueError("Preview clips do not expose a usable frame rate.")
        scale = min(1.0, _MAX_WIDTH / width)
        out_width = _even(width * scale)
        out_height = _even(height * scale)
        max_frames = int(round(_MAX_SECONDS * fps)) + 2
        ffmpeg = local_tool("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is required to render a difference preview.")
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg, "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{out_width}x{out_height}",
            "-r", f"{fps:.6f}",
            "-i", "-",
            "-an",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(partial),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        frames = 0
        while frames < max_frames:
            ok_first, before = first.read()
            ok_second, after = second.read()
            if not ok_first or not ok_second:
                break
            if before is None or after is None:
                raise ValueError("A preview clip returned an empty frame.")
            if (out_width, out_height) != (width, height):
                before = cv2.resize(before, (out_width, out_height), interpolation=cv2.INTER_AREA)
                after = cv2.resize(after, (out_width, out_height), interpolation=cv2.INTER_AREA)
            process.stdin.write(_diff_frame(before, after).tobytes())
            frames += 1
        if frames == 0:
            raise ValueError("The selected preview range contains no frames.")
        if frames < max_frames:
            extra_first = first.grab()
            extra_second = second.grab()
            if extra_first != extra_second:
                raise ValueError("Preview clips end at different frames; choose matching clips.")
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
        code = process.wait()
        if code != 0:
            raise RuntimeError("FFmpeg could not render the difference preview: " + stderr[-500:].strip())
        os.replace(partial, target)
    finally:
        first.release()
        second.release()
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stderr and not process.stderr.closed:
                process.stderr.close()
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    pass


def render_diff_preview(first: Path, second: Path, start: float, end: float) -> dict:
    """Render (or reuse) a cached difference video for two known previews."""
    first = resolve(str(first))
    second = resolve(str(second))
    if first.resolve() == second.resolve():
        raise ValueError("Choose two different clips for a pixel difference.")
    if not first.is_file() or not second.is_file():
        raise FileNotFoundError("One of the selected clips is no longer available.")
    try:
        start = float(start)
        end = float(end)
    except (TypeError, ValueError):
        raise ValueError("The difference preview needs numeric start and end times.") from None
    if not (0 <= start < end):
        raise ValueError("The difference preview needs an end time after its start.")
    if end - start > _MAX_SECONDS + 1e-6:
        raise ValueError(f"The difference preview is limited to {_MAX_SECONDS:.0f} seconds.")
    first_clip = media_clip_path(first, start, end, "denoise-compare-v1")
    second_clip = media_clip_path(second, start, end, "denoise-compare-v1")
    token = _token(first, second, start, end)
    target = _CACHE / f"{token}.mp4"
    if target.is_file() and target.stat().st_size > 0:
        return {"token": token, "url": f"/api/denoise-diff-video?token={token}", "cached": True}
    with _lock_for(token):
        if target.is_file() and target.stat().st_size > 0:
            return {"token": token, "url": f"/api/denoise-diff-video?token={token}", "cached": True}
        _render(first_clip, second_clip, target, token)
    return {"token": token, "url": f"/api/denoise-diff-video?token={token}", "cached": False}
