"""Bounded-memory, two-pass experimental Localized Deflicker for whole movies.

This is a research candidate, not validated film restoration. Pass 1 scans
lightweight per-frame shot statistics without retaining decoded video frames.
Pass 2 decodes/encodes sequentially with a sliding five-frame lookahead.
Frame count is not artificially capped. Optical-flow / subtle-transition
handling is NOT solved: automatic cut detection and fade guard are conservative
approximations and require human review of whole-film results.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import file_fingerprint, replace_with_retry, resumable_output, root_relative, write_signature
from film_deflicker import frame_luma
from intermediate_video import codec_args, container_args

ALGORITHM = "localized_stream_oscillation_v1"
RADIUS = 2
SIGMA = 18.0
STRENGTH = 0.6
CAP = 0.08
CUT_HISTOGRAM_THRESHOLD = 0.35
MAX_BUFFER_FRAMES = 5


def validate_stream_request(args: Any, info: dict[str, Any]) -> tuple[int, int, int, float]:
    if not getattr(args, "localized_confirm_film_experiment", False):
        raise ValueError(
            "Localized streaming is experimental: enable explicit film-experiment "
            "confirmation to acknowledge possible motion and transition artifacts."
        )
    if any(bool(getattr(args, key, False)) for key in (
        "dearchive", "ai_descratch", "devignette", "detect_defects"
    )):
        raise ValueError(
            "Experimental Localized Deflicker must run ALONE; disable Dearchive, "
            "DeScratch, DeVignette and Detect Film Damage."
        )
    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    frames = int(info.get("frames") or 0)
    fps = float(info.get("fps") or 0)
    if width <= 0 or height <= 0 or frames <= 0:
        raise ValueError("Film stream must have positive dimensions and a verified frame count.")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Film stream must have a verified positive frame rate.")
    return frames, width, height, fps


def scan_shot_boundaries(
    source: Path, frames: int, width: int, height: int
) -> tuple[list[tuple[int, int, bool]], dict[str, Any]]:
    """Sequential cut/fade scan. Stores tiny scalars, not decoded frames.

    Reuses the same conservative histogram cut signal as ARP Conservative,
    complemented by coarse full-picture median for fade protection.
    """
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot scan Localized film: {source}")
    boundaries = [0]
    image_medians: list[float] = []
    prior_level = 0.0
    prior_hist = None
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                raise ValueError(f"Film geometry changed during shot scan at frame {count}.")
            level, hist = frame_luma(frame)
            thumb = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                (min(width, 160), min(height, 120)), interpolation=cv2.INTER_AREA
            )
            image_medians.append(float(np.median(thumb)))
            if prior_hist is not None:
                distance = float(cv2.compareHist(
                    prior_hist, hist, cv2.HISTCMP_BHATTACHARYYA
                ))
                if distance > CUT_HISTOGRAM_THRESHOLD or (
                    prior_level > 0 and level > 0 and
                    abs(prior_level - level) > 22 and distance > 0.15
                ):
                    boundaries.append(count)
            prior_level, prior_hist = level, hist
            count += 1
            if count % 3000 == 0:
                print(f"Localized scene scan: {count}/{frames} frames", flush=True)
    finally:
        cap.release()
    if count != frames:
        raise ValueError(
            f"Shot scan decoded {count} frames, but source metadata declared {frames}."
        )
    boundaries.append(frames)
    shots = []
    protected = 0
    for start, end in zip(boundaries, boundaries[1:]):
        seq = np.asarray(image_medians[start:end], dtype=np.float32)
        diffs = np.diff(seq)
        fade = bool(
            len(seq) >= 4 and np.ptp(seq) >= 8 and
            (np.all(diffs >= -0.75) or np.all(diffs <= 0.75))
        )
        shots.append((start, end, fade))
        protected += int(fade)
    return shots, {
        "frames": frames, "detected_shots": len(shots),
        "cut_boundaries": max(0, len(shots)-1),
        "protected_monotonic_shots": protected,
        "shot_detector": "arp_histogram_bhattacharyya_v1",
        "fade_detector": "coarse_full_picture_median_monotone_v1",
        "warnings": [
            "Shot and fade heuristics can miss crossfades, flashes and moving edits.",
            "No optical flow or motion compensation: manually inspect motion/lighting.",
        ],
    }


def _blur_luma(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32),
        (0, 0), sigmaX=SIGMA, sigmaY=SIGMA
    )


def _correct_center(
    image: np.ndarray, center: np.ndarray,
    neighboring: list[np.ndarray], previous: np.ndarray, following: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Match research pilot math but only for a bounded local temporal window."""
    target = np.mean(np.stack(neighboring), axis=0)
    ratio = np.divide(target, np.maximum(center, 8), dtype=np.float32)
    gain = np.clip(1 + STRENGTH * (ratio - 1), 1-CAP, 1+CAP)
    adjacent_center = 0.5 * (previous + following)
    deviation = np.abs(adjacent_center - center)
    spread = np.abs(previous - following)
    confidence = (
        (deviation >= 3.0) &
        (spread <= np.maximum(2.0, 0.5 * deviation))
    ).astype(np.float32)
    confidence = cv2.GaussianBlur(
        confidence, (0, 0), sigmaX=max(1.0, SIGMA * 0.25)
    )
    gain = 1 + (gain - 1) * confidence
    delta = float(np.max(np.abs(gain - 1)))
    if delta < 1e-4:
        return image, 0.0
    fixed = np.clip(
        np.rint(image.astype(np.float32) * gain[..., None]), 0, 255
    ).astype(np.uint8)
    return fixed, delta


def stream_corrected_frames(
    capture: cv2.VideoCapture, shots: list[tuple[int, int, bool]],
    width: int, height: int
):
    """Yield exact frame order with <=5 decoded frame/blur pairs in memory.

    Frames and luma maps are dropped as soon as no later frame can need them.
    No arbitrary frame-count threshold or memory proportional to film duration.
    """
    next_read = 0
    stored: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for start, end, fade in shots:
        for i in range(start, end):
            to_read = min(end-1, i+RADIUS)
            while next_read <= to_read:
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(f"Film decoder stopped early at frame {next_read}.")
                if frame.dtype != np.uint8 or frame.shape != (height, width, 3):
                    raise ValueError(f"Film geometry changed at frame {next_read}.")
                stored[next_read] = (frame, _blur_luma(frame))
                next_read += 1
            if len(stored) > MAX_BUFFER_FRAMES:
                raise AssertionError("Localized streaming buffer exceeded five frames.")
            image, center = stored[i]
            modified = image
            gain_delta = 0.0
            if not fade and start < i < end-1:
                neighbors = [
                    stored[j][1]
                    for j in range(max(start,i-RADIUS), min(end,i+RADIUS+1))
                    if j != i
                ]
                if len(neighbors) >= 2:
                    modified, gain_delta = _correct_center(
                        image, center, neighbors, stored[i-1][1], stored[i+1][1]
                    )
            yield i, modified, gain_delta
            expired = i - RADIUS
            if expired in stored:
                del stored[expired]
        stored.clear()
    # Reject hidden extra frames, even if the metadata count matched pass 1.
    if capture.grab():
        raise ValueError("Film decoder produced more frames than scan metadata.")


def prepare_stream_localized_deflicker(
    ffmpeg: str, source: Path, work_dir: Path, info: dict[str, Any], args: Any,
) -> Path:
    frames, width, height, fps = validate_stream_request(args, info)
    signature = {
        "tool": "localized_deflicker_stream.py",
        "algorithm": ALGORITHM,
        "version": 1,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "geometry": [width, height, frames], "fps": fps,
        "params": {
            "radius": RADIUS, "sigma": SIGMA, "strength": STRENGTH, "cap": CAP,
            "guard": "oscillation",
            "cuts": "arp_histogram_bhattacharyya_v1",
            "fades": "coarse_median_monotone_v1",
        },
        "intermediate_profile": args.intermediate_profile,
    }
    token = hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:12]
    output = work_dir / f"localized_stream_{token}.mkv"
    if not args.force and resumable_output(
        output, signature, width=width, height=height, video_like=source
    ):
        print(f"Reuse long-form experimental Localized Deflicker: {output}", flush=True)
        return output
    shots, scan_report = scan_shot_boundaries(source, frames, width, height)
    work_dir.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".partial.mkv")
    if partial.exists():
        raise FileExistsError(f"Existing partial output requires inspection: {partial}")
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
        "-r", f"{fps:.10g}", "-i", "pipe:0", "-an",
        *codec_args(args.intermediate_profile, fast=True),
        *container_args(str(partial), args.intermediate_profile), str(partial),
    ]
    reader = cv2.VideoCapture(str(source))
    if not reader.isOpened():
        raise RuntimeError(f"Cannot reopen film for Localized streaming: {source}")
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    changed = 0
    biggest_gain = 0.0
    processed = 0
    try:
        if process.stdin is None:
            raise RuntimeError("Cannot open encoder pipe.")
        for index, image, delta in stream_corrected_frames(reader, shots, width, height):
            process.stdin.write(image.tobytes())
            changed += int(delta > 0)
            biggest_gain = max(biggest_gain, delta)
            processed = index + 1
            if processed % 3000 == 0:
                print(f"Localized streaming: {processed}/{frames} frames", flush=True)
        process.stdin.close()
        errors = (process.stderr.read() if process.stderr else b"").decode(
            "utf-8", errors="replace"
        )
        if process.stderr:
            process.stderr.close()
        if process.wait() != 0 or not partial.is_file() or processed != frames:
            raise RuntimeError(
                errors or f"Localized streaming failed; wrote {processed}/{frames} frames."
            )
        replace_with_retry(partial, output, "Localized streaming experimental")
        write_signature(output, signature)
    except BaseException:
        if process.poll() is None:
            process.kill()
            process.wait()
        partial.unlink(missing_ok=True)
        raise
    finally:
        reader.release()
    report = {
        **scan_report,
        "algorithm": ALGORITHM, "method": "localized_stream",
        "research_only": True, "release_ready": False,
        "corrected_frames": changed,
        "max_applied_local_gain_pct": round(biggest_gain * 100, 3),
        "max_buffer_frames": MAX_BUFFER_FRAMES,
        "source": str(source), "output": str(output),
        "warning": "Cut detector and oscillation gate are heuristic; check full film.",
    }
    output.with_name(output.stem + "_analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Localized streaming complete: {processed} frames, {len(shots)} estimated shots, "
        f"{changed} frames altered; experimental, not production validated.",
        flush=True,
    )
    return output
