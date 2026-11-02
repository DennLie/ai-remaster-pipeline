"""Research-only spatially varying illumination deflicker (CPU, non-generative).

NOT A GUI BACKEND. Designed only for controlled, mostly static single-shot
experiments. Motion/occlusions can create false positive exposure corrections.
Writes a new, separate video; never overwrites the source or any ARP render.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


def _fading_shot(frames: list[np.ndarray]) -> bool:
    """Conservative negative control: protect smooth, global monotonic fades."""
    if len(frames) < 4:
        return False
    levels = np.array([float(np.median(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))) for f in frames])
    increments = np.diff(levels)
    # A substantial monotonic ramp is more likely an intentional transition.
    return bool(np.ptp(levels) >= 8 and
                (np.all(increments >= -0.75) or np.all(increments <= 0.75)))


def local_deflicker_frames(
    frames: list[np.ndarray], *, radius: int = 2,
    sigma: float = 18.0, strength: float = 0.6, cap: float = 0.08,
    cut_frames: tuple[int, ...] = ()
) -> tuple[list[np.ndarray], dict]:
    """Smooth spatial gain maps against temporal neighbours, preserving detail.

    Frame ordinals in cut_frames indicate *the first frame of a new shot*.
    No motion compensation: do not use for unvalidated production restoration.
    """
    if not frames:
        raise ValueError("At least one frame is required.")
    if radius < 1 or radius > 15 or sigma < 1 or not 0 <= strength <= 1 or not 0 < cap <= 0.15:
        raise ValueError("Invalid radius, sigma, strength or correction cap.")
    shape = frames[0].shape
    if len(shape) != 3 or shape[2] != 3 or any(
        f.shape != shape or f.dtype != np.uint8 for f in frames
    ):
        raise ValueError("Frames must be matching 8-bit BGR images.")
    boundaries = [0, *sorted(set(cut_frames)), len(frames)]
    if len(boundaries) != len(set(boundaries)) or any(x <= 0 or x >= len(frames) for x in cut_frames):
        raise ValueError("Cut boundaries must be unique internal frame indices.")
    low_frequency = [
        cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32),
                         (0, 0), sigmaX=sigma, sigmaY=sigma)
        for f in frames
    ]
    corrected = [f.copy() for f in frames]
    changed = []
    max_gain = 0.0
    protected = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        shot = frames[start:stop]
        if _fading_shot(shot):
            protected.append([start, stop])
            continue
        for i in range(start, stop):
            neighbors = [j for j in range(max(start, i-radius), min(stop, i+radius+1))
                         if j != i]
            if len(neighbors) < 2:
                continue
            # Leave texture intact. A low-frequency illumination estimate is
            # applied as a spatial gain; no source pixels are synthesized.
            target = np.mean(np.stack([low_frequency[j] for j in neighbors]), axis=0)
            original = low_frequency[i]
            ratio = np.divide(target, np.maximum(original, 8), dtype=np.float32)
            gain = np.clip(1 + strength * (ratio - 1), 1-cap, 1+cap)
            if np.max(np.abs(gain - 1)) < 1e-4:
                continue
            corrected[i] = np.clip(
                np.rint(frames[i].astype(np.float32) * gain[..., None]), 0, 255
            ).astype(np.uint8)
            changed.append(i)
            max_gain = max(max_gain, float(np.max(np.abs(gain - 1))))
    return corrected, {
        "algorithm": "localized_blurred_luma_temporal_neighbor_mean_v1",
        "status": "research_only_no_motion_compensation",
        "frames": len(frames),
        "corrected_frames": len(changed),
        "corrected_indices_zero_based": changed,
        "protected_monotonic_fades": protected,
        "max_abs_gain_pct": round(100 * max_gain, 4),
        "radius": radius,
        "sigma_px": sigma,
        "strength": strength,
        "cap": cap,
        "cut_frames": list(cut_frames),
    }


def run_cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True, help="New FFV1 MKV output; never overwrite")
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=18.0)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--cap", type=float, default=0.08)
    parser.add_argument("--cut-frame", type=int, action="append", default=[])
    parser.add_argument("--max-frames", type=int, default=240)
    args = parser.parse_args()
    source = Path(args.source).resolve(strict=True)
    dest = Path(args.output).resolve()
    if source == dest or dest.exists() or dest.with_suffix(".report.json").exists():
        raise FileExistsError("Output must be a new filename different from the source.")
    if dest.suffix.lower() != ".mkv":
        raise ValueError("Research output must be an MKV file.")
    capture = cv2.VideoCapture(str(source))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Source FPS could not be verified.")
        frames = []
        while len(frames) <= args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) > args.max_frames:
            raise ValueError("Clip too long for bounded research evaluation.")
    finally:
        capture.release()
    fixed, report = local_deflicker_frames(
        frames, radius=args.radius, sigma=args.sigma, strength=args.strength,
        cap=args.cap, cut_frames=tuple(args.cut_frame)
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.stem + ".partial.mkv")
    if partial.exists():
        raise FileExistsError(f"Stale intermediate output exists: {partial}")
    h, w = frames[0].shape[:2]
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", f"{w}x{h}", "-r", f"{fps:.12g}",
        "-i", "pipe:0", "-an", "-c:v", "ffv1", "-level", "3", str(partial),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for frame in fixed:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        assert proc.stderr is not None
        error = proc.stderr.read()
        if proc.wait(timeout=90) != 0:
            raise RuntimeError(error.decode("utf-8", errors="replace")[:800])
        partial.replace(dest)
        report["source"] = str(source)
        report["output"] = str(dest)
        report["fps"] = fps
        report["dimensions"] = [w, h]
        dest.with_suffix(".report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except BaseException:
        proc.kill() if proc.poll() is None else None
        partial.unlink(missing_ok=True)
        raise
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
