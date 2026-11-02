"""Explicitly opt-in, short single-shot Localized Deflicker GUI adapter.

Uses the research prototype unchanged, with oscillation guard forced ON.
Not a production restoration backend: no motion compensation or automatic
scene-cut protection. Refuses long or unconfirmed/multi-operation jobs.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import file_fingerprint, replace_with_retry, resumable_output, root_relative, write_signature
from intermediate_video import codec_args, container_args
from local_deflicker_research import local_deflicker_frames

MAX_FRAMES = 240
MAX_TOTAL_PIXELS = 75_000_000
ALGORITHM = "localized_oscillation_guard_v1"
RADIUS = 2
SIGMA = 18.0
STRENGTH = 0.6
CAP = 0.08


def validate_localized_request(args: Any, info: dict[str, Any]) -> tuple[int, int, int, float]:
    if not getattr(args, "localized_confirm_single_shot", False):
        raise ValueError(
            "Localized Deflicker requires explicit confirmation that this is a single-shot "
            "test clip (no cuts, transitions or intended lighting changes)."
        )
    if any(bool(getattr(args, key, False)) for key in (
        "dearchive", "ai_descratch", "devignette", "detect_defects"
    )):
        raise ValueError(
            "Experimental Localized Deflicker must run ALONE. Switch off Dearchive, "
            "DeScratch, DeVignette and Detect Film Damage."
        )
    frames = int(info.get("frames") or 0)
    width = int(info.get("width") or 0)
    height = int(info.get("height") or 0)
    fps = float(info.get("fps") or 0)
    if not (1 <= frames <= MAX_FRAMES) or width <= 0 or height <= 0:
        raise ValueError(f"Localized Deflicker permits only 1–{MAX_FRAMES} exactly counted frames.")
    if width * height * frames > MAX_TOTAL_PIXELS:
        raise ValueError(
            "Experimental Localized Deflicker memory limit exceeded "
            f"({width}×{height}×{frames} > {MAX_TOTAL_PIXELS:,} pixels). "
            "Use a shorter or lower-resolution test section."
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Localized Deflicker requires a verified positive frame rate.")
    return frames, width, height, fps


def prepare_localized_deflicker(
    ffmpeg: str, source: Path, work_dir: Path, info: dict[str, Any], args: Any
) -> Path:
    frames, width, height, fps = validate_localized_request(args, info)
    signature = {
        "tool": "localized_deflicker_backend.py",
        "algorithm": ALGORITHM,
        "version": 1,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "geometry": [width, height, frames],
        "fps": fps,
        "temporal_guard": "oscillation",
        "radius": RADIUS, "sigma": SIGMA, "strength": STRENGTH, "cap": CAP,
        "intermediate_profile": args.intermediate_profile,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    output = work_dir / f"localized_oscillation_{key}.mkv"
    if not args.force and resumable_output(
        output, signature, width=width, height=height, video_like=source
    ):
        print(f"Reuse experimental Localized Deflicker: {output}", flush=True)
        return output

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot decode Localized Deflicker source: {source}")
    images: list[np.ndarray] = []
    try:
        while len(images) <= frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                raise ValueError("Unexpected decoded Localized Deflicker geometry or pixel type.")
            images.append(frame)
    finally:
        cap.release()
    if len(images) != frames:
        raise ValueError(
            f"Localized Deflicker frame count differs from metadata ({len(images)} != {frames})."
        )

    result, report = local_deflicker_frames(
        images, radius=RADIUS, sigma=SIGMA, strength=STRENGTH,
        cap=CAP, temporal_guard="oscillation"
    )
    if len(result) != frames:
        raise RuntimeError("Localized Deflicker returned an inconsistent frame count.")
    work_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mkv")
    if partial.exists():
        raise FileExistsError(f"Stale localized output exists; inspect before retry: {partial}")

    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
        "-r", f"{fps:.10g}", "-i", "pipe:0", "-an",
        *codec_args(args.intermediate_profile, fast=True),
        *container_args(str(partial), args.intermediate_profile),
        str(partial),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        if process.stdin is None:
            raise RuntimeError("Localized Deflicker encoder pipe unavailable.")
        for frame in result:
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        stderr = (process.stderr.read() if process.stderr else b"").decode(
            "utf-8", errors="replace"
        )
        if process.stderr is not None:
            process.stderr.close()
        if process.wait(timeout=120) != 0 or not partial.is_file():
            raise RuntimeError(stderr or "Localized Deflicker encoding failed.")
        replace_with_retry(partial, output, "Experimental Localized Deflicker")
        write_signature(output, signature)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        partial.unlink(missing_ok=True)
        raise

    report_path = output.with_name(output.stem + "_analysis.json")
    report_path.write_text(
        json.dumps({
            **report, "stage": "cleanup", "method": "localized_research",
            "algorithm": ALGORITHM, "experimental": True,
            "single_shot_user_confirmed": True,
            "release_ready": False,
            "source": str(source), "output": str(output),
            "warning": "No motion compensation; inspect cuts, fades and moving objects.",
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
    )
    print(
        f"Experimental Localized Deflicker: {frames} frames, guard=oscillation, "
        f"{report['corrected_frames']} adjusted, max local gain {report['max_abs_gain_pct']:.2f}%. "
        "Research only, NOT production-validated.",
        flush=True,
    )
    return output
