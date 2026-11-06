"""Frame-indexed, native-size comparison reels, cached in short internal segments.

Each encoded frame contains the original, result and pre-compression difference.
Changing the displayed region therefore cannot change the decoded frame index.
The source/master files remain read-only. Display RGB is 8-bit, like the inspector.
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
from .denoise_diff_video import _diff_frame, _DIFF_MODES, _CONTRASTS
from .denoise_review import comparison_pair
from .frame_inspector import _metadata
from .media import local_tool

_CACHE = ROOT / ".cache" / "denoise_playback"
_VERSION = 1
_SECONDS = 8
_RENDER_LOCK = threading.Lock()


def segment_video_path(token: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(token or "")):
        return None
    path = _CACHE / f"{token}.mp4"
    return path if path.is_file() and path.stat().st_size else None


def _identity(path: Path) -> list:
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]


def _layout(width: int, height: int, paired: bool) -> tuple[int, int, dict]:
    # shortcut: native reels fit 8192px decoder axes; larger scans need separate-plane decoders.
    if not paired:
        cols, rows = 1, 1
    elif width * 3 <= 8192:
        cols, rows = 3, 1
    elif height * 3 <= 8192:
        cols, rows = 1, 3
    elif width * 2 <= 8192 and height * 2 <= 8192:
        cols, rows = 2, 2
    else:
        raise ValueError("Native comparison exceeds the browser decoder size. Use the exact frame inspector for this source.")
    regions = {"original": [0, 0, width, height]}
    if paired:
        regions["result"] = [1 % cols * width, 1 // cols * height, width, height]
        regions["difference"] = [2 % cols * width, 2 // cols * height, width, height]
    # Only the atlas padding is rounded; source pixels and regions are not resized.
    return (width * cols + 1) // 2 * 2, (height * rows + 1) // 2 * 2, regions


def _seek(capture, frame: int) -> None:
    if frame and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame):
        raise ValueError("Decoder cannot seek to the requested source frame.")
    if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - frame) > .1:
        raise ValueError("Decoder could not verify the requested frame position.")


def _render(source: Path, output: Path | None, target: Path, first: int, count: int,
            metadata: dict, atlas_width: int, atlas_height: int, regions: dict,
            mode: str, contrast: int, source_offset: int = 0) -> None:
    ffmpeg = local_tool("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg is required for comparison playback.")
    captures = [cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
                for path in (source, output) if path is not None]
    process = None
    partial = target.with_suffix(".partial.mp4")
    try:
        for slot, capture in enumerate(captures):
            if not capture.isOpened():
                raise ValueError("Could not decode the processing input or result.")
            _seek(capture, first + (source_offset if slot == 0 else 0))
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", f"{atlas_width}x{atlas_height}", "-r", metadata["fps_fraction"], "-i", "-",
                   "-an", "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                   "-threads", "2", "-tune", "zerolatency",
                   "-pix_fmt", "yuv420p", "-g", str(max(1, round(metadata["fps"]))),
                   "-bf", "0", "-movflags", "+faststart", str(partial)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        atlas = np.zeros((atlas_height, atlas_width, 3), dtype=np.uint8)
        for index in range(count):
            images = []
            for slot, capture in enumerate(captures):
                ok, image = capture.read()
                if not ok or image is None:
                    raise ValueError("Decode ended before the verified segment frame count.")
                if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - (first + index + 1 + (source_offset if slot == 0 else 0))) > .1:
                    raise ValueError("Decoder lost the verified source frame index.")
                if image.shape != (metadata["height"], metadata["width"], 3):
                    raise ValueError("Decoded frame dimensions changed.")
                images.append(image)
            planes = {"original": images[0]}
            if output is not None:
                planes["result"] = images[1]
                # Subtract master decodes before any scaling or H.264 encoding.
                planes["difference"] = _diff_frame(images[0], images[1], mode, contrast)
            for name, image in planes.items():
                x, y, width, height = regions[name]
                atlas[y:y + height, x:x + width] = image
            process.stdin.write(atlas.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        process.stderr.close()
        if process.wait(timeout=120) != 0:
            raise RuntimeError("Comparison encoding failed: " + error[-500:].strip())
        encoded = _metadata(partial)
        if (encoded["frames"] != count or encoded["fps_fraction"] != metadata["fps_fraction"] or
                (encoded["width"], encoded["height"]) != (atlas_width, atlas_height)):
            raise ValueError("Encoded comparison lost its frame mapping or dimensions.")
        os.replace(partial, target)
    finally:
        for capture in captures:
            capture.release()
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            if process.stderr and not process.stderr.closed:
                process.stderr.close()
        partial.unlink(missing_ok=True)


def playback_segment(app, candidate: str, frame: int, mode: str = "direction", contrast: int = 32, stage: str = "denoise", reference: str = 'input', mask_values=None) -> dict:
    if not isinstance(candidate, str):
        raise ValueError("Choose a finished Remove Noise variant.")
    if type(frame) is not int or frame < 0:
        raise ValueError("Choose a nonnegative source frame index.")
    if mode not in _DIFF_MODES or type(contrast) is not int or contrast not in _CONTRASTS:
        raise ValueError("Choose a supported difference mode and amplification.")
    if mask_values is not None:
        if stage != 'dust' or candidate or reference != 'input' or not isinstance(mask_values, dict):
            raise ValueError('Mask playback requires Dust stage input and saved detector settings.')
        from . import dust
        folder, report = dust.saved_mask(app.stabilized_source_for_downstream(), mask_values)
        source, output = folder/'input.mkv', folder/'effective_mask.mkv'
        metadata = _metadata(source)
        mask_meta = _metadata(output)
        if any(mask_meta[key] != metadata[key] for key in ('frames','width','height','fps_fraction')):
            raise ValueError('Saved mask does not match the stage-input timeline or dimensions.')
        total, offset = metadata['frames'], 0
    else:
        source, output, metadata, total, offset = comparison_pair(app, candidate, stage, reference)
    if frame >= total:
        raise ValueError("The requested frame is outside the processing input.")
    size = max(1, round(metadata["fps"] * _SECONDS))
    first = frame // size * size
    end = min(total, first + size)
    width, height, regions = _layout(metadata["width"], metadata["height"], output is not None)
    if mask_values is not None:
        regions['mask'] = regions['result']
    identity = [_VERSION, _identity(source), _identity(output) if output else None,
                first, end, metadata["fps_fraction"], mode, contrast, reference, offset, mask_values is not None]
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    target = _CACHE / f"{token}.mp4"
    cached = bool(segment_video_path(token))
    if not cached:
        with _RENDER_LOCK:
            cached = bool(segment_video_path(token))
            if not cached:
                _render(source, output, target, first, end - first, metadata, width, height, regions, mode, contrast, offset)
    return {"url": f"/api/denoise-playback-video?token={token}", "cached": cached,
            "first_frame": first, "end_frame": end, "frames": total,
            "fps": metadata["fps"], "fps_fraction": metadata["fps_fraction"],
            "width": metadata["width"], "height": metadata["height"],
            "atlas_width": width, "atlas_height": height, "regions": regions,
            "difference_basis": "decoded_original_source_and_result" if reference == 'original' else "decoded_processing_input_and_result", "reference": reference, "compressed": True}
