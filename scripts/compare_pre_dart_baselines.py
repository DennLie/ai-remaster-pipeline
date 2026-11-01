"""Reproducible, frame-exact visual baselines for pre-DART preprocessing.

Compare the same original clip with alternative Deflicker / stabilization
outputs. This describes image differences, NOT historical authenticity or
restoration accuracy. No DART, GPU model or archived movie is downloaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_variants(values: list[str]) -> dict[str, Path]:
    if not values:
        raise ValueError("Provide at least one NAME=VIDEO candidate")
    variants = {}
    for pair in values:
        label, sep, filename = pair.partition("=")
        if not sep or not label or not filename or not label.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Expected simple label=video, got {pair!r}")
        if label in variants or label == "original":
            raise ValueError(f"Duplicate or reserved variant label: {label}")
        variants[label] = Path(filename)
    return variants


def open_video(path: Path) -> tuple[cv2.VideoCapture, tuple[int, int], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not decode video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        raise ValueError(f"Bad geometry or cadence: {path} ({width}x{height} at {fps})")
    return cap, (width, height), fps


def median_luma(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    central = gray[h // 10:h - h // 10, w // 10:w - w // 10]
    return float(np.median(central))


def tile(image: np.ndarray, label: str, width: int = 320) -> np.ndarray:
    h = max(64, round(image.shape[0] * width / image.shape[1]))
    small = cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
    bar = np.full((28, width, 3), 24, np.uint8)
    cv2.putText(bar, label[:34], (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.47, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([small, bar], axis=0)


def evaluate(original: Path, variants: dict[str, Path], output_dir: Path,
             *, cut_after_frames: set[int] | None = None) -> dict:
    """Reject missing/extra frames or altered frame geometry instead of aligning."""
    if not variants:
        raise ValueError("No candidate videos supplied")
    cuts = set(cut_after_frames or ())
    if any(not isinstance(i, int) or i < 0 for i in cuts):
        raise ValueError("Cut boundaries must be nonnegative frame indices")
    handles = {}
    try:
        for label, path in {"original": original, **variants}.items():
            cap, size, fps = open_video(path)
            handles[label] = (cap, size, fps)
        size = handles["original"][1]
        fps = handles["original"][2]
        for label, (_, dims, rate) in handles.items():
            if dims != size or abs(rate-fps) > 0.02:
                raise ValueError(f"{label} mismatches source geometry or frame rate: {dims}, {rate:g} vs {size}, {fps:g}")
        labels = list(variants)
        data = {
            label: {"luma": [], "mean_absolute_pixel_difference": [],
                    "fraction_new_clipped_pixels": []}
            for label in labels
        }
        original_values = []
        previews: list[tuple[int, np.ndarray]] = []
        frames = 0
        while True:
            ok, ref = handles["original"][0].read()
            candidates = {label: handles[label][0].read() for label in labels}
            if not ok:
                if any(success for success, _ in candidates.values()):
                    raise ValueError("Candidate has extra frames")
                break
            if not all(success for success, _ in candidates.values()):
                raise ValueError(f"Candidate ended early at source frame {frames}")
            if ref.shape != (size[1], size[0], 3):
                raise ValueError("Source contains inconsistent frame geometry")
            median_ref = median_luma(ref)
            original_values.append(median_ref)
            sheet = [tile(ref, f"Frame {frames} | Original")]
            for label in labels:
                frame = candidates[label][1]
                if frame.shape != ref.shape or frame.dtype != np.uint8:
                    raise ValueError(f"{label} has unsupported frame geometry/precision at {frames}")
                diff = cv2.absdiff(ref, frame)
                new_clipped = (
                    ((frame <= 1) & (ref > 1)) |
                    ((frame >= 254) & (ref < 254))
                )
                data[label]["luma"].append(median_luma(frame))
                data[label]["mean_absolute_pixel_difference"].append(float(np.mean(diff)))
                data[label]["fraction_new_clipped_pixels"].append(float(np.mean(new_clipped)))
                sheet.append(tile(frame, f"Frame {frames} | {label}"))
            # First 3 frames provide a deterministic same-frame visual baseline.
            if frames < 3:
                previews.append((frames, np.concatenate(sheet, axis=1)))
            frames += 1
        if frames < 3:
            raise ValueError("Need at least three frames for temporal comparison")
        if cuts and max(cuts) >= frames - 1:
            raise ValueError("Cut-after frame index out of range")
        eligible_pairs = [i for i in range(frames - 1) if i not in cuts]
        if not eligible_pairs:
            raise ValueError("All adjacent frame pairs were excluded")
        original_variation = np.mean([
            abs(original_values[i+1] - original_values[i]) for i in eligible_pairs
        ])
        result = {
            "schema_version": 1,
            "purpose": "Paired-frame descriptive QA, not a film-fidelity ground truth or SOTA benchmark",
            "source": original.name, "width": size[0], "height": size[1],
            "fps": fps, "frames": frames,
            "excluded_cut_after_frames": sorted(cuts),
            "original_mean_abs_adjacent_median_luma": round(float(original_variation), 5),
            "variants": {},
            "warning": (
                "Lower adjacent luma changes are not necessarily more authentic; "
                "pixel differences include video reencoding, intentional motion, "
                "valid stabilization geometry and defects. Manually review crops and grain."
            ),
        }
        for label in labels:
            d = data[label]
            variation = np.mean([
                abs(d["luma"][i+1] - d["luma"][i]) for i in eligible_pairs
            ])
            result["variants"][label] = {
                "video": variants[label].name,
                "mean_abs_adjacent_median_luma": round(float(variation), 5),
                "mean_absolute_pixel_difference_from_original": round(
                    float(np.mean(d["mean_absolute_pixel_difference"])), 5
                ),
                "fraction_new_clipped_channels": round(
                    float(np.mean(d["fraction_new_clipped_pixels"])), 7
                ),
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        poster = np.concatenate([frame for _, frame in previews], axis=0)
        if not cv2.imwrite(str(output_dir / "paired_pre_dart_frames.jpg"), poster,
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError("Could not write comparison sheet")
        (output_dir / "pre_dart_baselines.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return result
    finally:
        for cap, _, _ in handles.values():
            cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", required=True, type=Path)
    p.add_argument("--variant", action="append", required=True, metavar="NAME=VIDEO")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--cut-after-frame", action="append", type=int, default=[])
    args = p.parse_args()
    report = evaluate(args.original, parse_variants(args.variant), args.output_dir,
                      cut_after_frames=set(args.cut_after_frame))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
