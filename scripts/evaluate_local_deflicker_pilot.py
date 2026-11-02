"""Controlled real-footage pilot for the *research-only* localized Deflicker.

Loads a short digital film-transfer clip (no source redistribution), injects a
known, localized temporal exposure disturbance, and measures whether the
experimental algorithm recovers the original digital frames. Separately
measures false modifications to untouched footage. No historical ground
truth or restoration-quality claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from local_deflicker_research import local_deflicker_frames

MAX_FRAMES = 240
MAX_TOTAL_PIXELS = 75_000_000


def load_video(path: Path, max_frames: int) -> tuple[list[np.ndarray], float]:
    if not 3 <= max_frames <= MAX_FRAMES:
        raise ValueError("max_frames must be between 3 and 240.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Could not open the source film clip.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Could not verify input frame rate.")
        frames: list[np.ndarray] = []
        total_pixels = 0
        while len(frames) < max_frames + 1:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Input requires 8-bit BGR video frames.")
            total_pixels += frame.shape[0] * frame.shape[1]
            if total_pixels > MAX_TOTAL_PIXELS:
                raise ValueError("Research clip exceeds bounded memory budget.")
            frames.append(frame)
        if len(frames) > max_frames:
            raise ValueError("Use a shorter source section; this pilot refuses to silently truncate.")
        if len(frames) < 3:
            raise ValueError("At least three frames required.")
        return frames, fps
    finally:
        capture.release()


def synthetic_exposure_damage(
    clean: list[np.ndarray], amplitude: float = 0.18
) -> tuple[list[np.ndarray], np.ndarray]:
    """Alternating exposure in a left-side region with a feathered border.

    This simulates *injected digital exposure error*, not a real film flaw.
    """
    if not 0 < amplitude <= 0.40:
        raise ValueError("Amplitude must be >0 and <=0.4.")
    height, width = clean[0].shape[:2]
    ramp = np.linspace(0, 1, width, dtype=np.float32)
    mask_line = np.clip((0.60 - ramp) / 0.24, 0, 1)
    mask = np.repeat(mask_line[None, :, None], height, axis=0)
    damaged = []
    for i, image in enumerate(clean):
        direction = -1.0 if i % 2 else 1.0
        gain = 1 + direction * amplitude * mask
        damaged.append(np.clip(np.rint(image.astype(np.float32) * gain), 0, 255).astype(np.uint8))
    return damaged, mask_line


def mean_abs_error(reference: list[np.ndarray], images: list[np.ndarray],
                   x0: int, x1: int) -> float:
    total, samples = 0.0, 0
    for clean, image in zip(reference, images):
        difference = np.abs(clean[:, x0:x1].astype(np.int16) -
                            image[:, x0:x1].astype(np.int16))
        total += float(difference.sum())
        samples += difference.size
    return round(total / samples, 5) if samples else 0.0


def evaluate(clean: list[np.ndarray], *, radius: int = 2, sigma: float = 18,
             strength: float = 0.6, cap: float = 0.08,
             cut_frames: tuple[int, ...] = ()) -> dict:
    damaged, _ = synthetic_exposure_damage(clean)
    recovered, run_report = local_deflicker_frames(
        damaged, radius=radius, sigma=sigma, strength=strength,
        cap=cap, cut_frames=cut_frames
    )
    # Crucial negative control: does the same algorithm modify unaltered film?
    untouched_output, negative_report = local_deflicker_frames(
        clean, radius=radius, sigma=sigma, strength=strength,
        cap=cap, cut_frames=cut_frames
    )
    width = clean[0].shape[1]
    inside = (0, max(1, width // 3))
    outside = (max(1, width * 3 // 4), width)
    degraded = mean_abs_error(clean, damaged, *inside)
    restored = mean_abs_error(clean, recovered, *inside)
    return {
        "method": "localized_blurred_luma_temporal_neighbor_mean_v1",
        "status": "research_only",
        "reference": "original digital frames before synthetic damage injection; NOT original film ground truth",
        "damage": "alternating 18% localized digital gain on left region with feathering",
        "frames": len(clean),
        "frame_dimensions": [clean[0].shape[1], clean[0].shape[0]],
        "damaged_inside_mae": degraded,
        "corrected_inside_mae": restored,
        "inside_mae_improvement": round(degraded - restored, 5),
        "damaged_outside_mae": mean_abs_error(clean, damaged, *outside),
        "corrected_outside_mae": mean_abs_error(clean, recovered, *outside),
        "untouched_original_modified_mae": mean_abs_error(clean, untouched_output, 0, width),
        "untouched_outside_modified_mae": mean_abs_error(clean, untouched_output, *outside),
        "guardrails": {
            "candidate": run_report,
            "negative_control": negative_report,
        },
        "release_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Locally available short archival or synthetic clip")
    parser.add_argument("--report", required=True, help="New JSON report path")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=18.0)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--cap", type=float, default=0.08)
    parser.add_argument("--cut-frame", type=int, action="append", default=[])
    args = parser.parse_args()
    source = Path(args.source).resolve(strict=True)
    report_path = Path(args.report).resolve()
    if source == report_path or report_path.exists() or report_path.suffix.lower() != ".json":
        raise FileExistsError("Choose a new JSON report file; input and existing files are protected.")
    frames, fps = load_video(source, args.max_frames)
    result = evaluate(frames, radius=args.radius, sigma=args.sigma,
                      strength=args.strength, cap=args.cap,
                      cut_frames=tuple(args.cut_frame))
    result["fps"] = fps
    result["source"] = str(source)
    result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")
    print(json.dumps({k:v for k,v in result.items() if k != "guardrails"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
