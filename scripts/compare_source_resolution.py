"""Compare film defect mask proposals before/after resolution changes.

This script is intentionally descriptive rather than a benchmark of accuracy:
no annotated ground truth -> no precision/recall, no claim that upscaling helps.
Input is a short excerpt of the same digital transfer for every variant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from film_defect_masks import temporal_defect_mask


def resized(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    method = cv2.INTER_AREA if height < frame.shape[0] else cv2.INTER_LANCZOS4
    return cv2.resize(frame, size, interpolation=method)


def variants(width: int, height: int) -> dict[str, tuple[int, int]]:
    def scale(h: int) -> tuple[int, int]:
        w = max(16, int(round(width * h / height)))
        return (w - w % 2, h)
    low = min(height, 480)
    return {
        "downsampled_480p": scale(low - low % 2),
        "digital_source": (width, height),
        "lanczos_2x": (width * 2, height * 2),
    }


def same_coordinates(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    # For a high-resolution mask, average binary subpixels so isolated,
    # single-subpixel interpolation responses do not expand to a full source pixel.
    if (mask.shape[1], mask.shape[0]) == size:
        return mask > 0
    method = cv2.INTER_AREA if mask.shape[0] > size[1] else cv2.INTER_NEAREST
    adjusted = cv2.resize(mask, size, interpolation=method)
    return adjusted >= 128


def overlay(frame: np.ndarray, binary_mask: np.ndarray, title: str) -> np.ndarray:
    bg = frame.copy()
    # Highlight proposed (not proven) damage; preserve a faint source beneath.
    bg[binary_mask, 0] = (bg[binary_mask, 0] * 0.25).astype(np.uint8)
    bg[binary_mask, 1] = (bg[binary_mask, 1] * 0.25).astype(np.uint8)
    bg[binary_mask, 2] = 255
    h, w = bg.shape[:2]
    # Text belongs on a black footer, not across historical imagery.
    footer = np.full((32, w, 3), 28, dtype=np.uint8)
    cv2.putText(footer, title, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([bg, footer], axis=0)


def analyze_clip(input_video: Path, output_dir: Path, max_frames: int = 72, sensitivity: float = 0.65) -> dict[str, Any]:
    if max_frames < 3:
        raise ValueError("Need at least three consecutive video frames")
    video = cv2.VideoCapture(str(input_video))
    if not video.isOpened():
        raise FileNotFoundError(f"Video not readable: {input_video}")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(video.get(cv2.CAP_PROP_FPS))
    sizes = variants(width, height)
    output_dir.mkdir(parents=True, exist_ok=True)

    def read() -> np.ndarray | None:
        ok, frame = video.read()
        return frame if ok else None

    previous = None
    current = read()
    following = read()
    totals = {name: {"selected_native": 0, "pixels_native": 0, "selected_source_coordinates": 0, "candidate_frames": 0}
              for name in sizes}
    disagreement = {"downsampled_480p": {"intersection": 0, "union": 0},
                    "lanczos_2x": {"intersection": 0, "union": 0}}
    snapshots = []
    sample_indices = {max_frames // 5, max_frames // 2, (max_frames * 4) // 5}
    processed = 0
    try:
        while current is not None and processed < max_frames:
            predictions: dict[str, np.ndarray] = {}
            for name, size in sizes.items():
                image = resized(current, size)
                mask = (temporal_defect_mask(
                    resized(previous, size), image, resized(following, size), sensitivity
                ) if previous is not None and following is not None
                        else np.zeros((size[1], size[0]), dtype=np.uint8))
                canonical = same_coordinates(mask, (width, height))
                predictions[name] = canonical
                value = totals[name]
                value["selected_native"] += int(np.count_nonzero(mask))
                value["pixels_native"] += mask.size
                value["selected_source_coordinates"] += int(np.count_nonzero(canonical))
                value["candidate_frames"] += bool(np.any(mask))
            basis = predictions["digital_source"]
            for name in disagreement:
                pred = predictions[name]
                disagreement[name]["intersection"] += int(np.count_nonzero(basis & pred))
                disagreement[name]["union"] += int(np.count_nonzero(basis | pred))
            if processed in sample_indices:
                snapshots.append((processed, current.copy(), {k: v.copy() for k, v in predictions.items()}))
            processed += 1
            previous, current, following = current, following, read()
    finally:
        video.release()

    if processed < 3:
        raise ValueError("Source clip contains fewer than three usable frames")
    results = {
        "source_file": input_video.name,
        "dimensions": {"width": width, "height": height},
        "fps": fps,
        "processed_frames": processed,
        "sensitivity": sensitivity,
        "ground_truth_available": False,
        "warning": "Detection coverage and cross-scale agreement are not detection accuracy or restoration quality.",
        "variants": {
            name: {
                "width": sizes[name][0],
                "height": sizes[name][1],
                **values,
                "native_mask_coverage_fraction": round(values["selected_native"] / max(1, values["pixels_native"]), 8),
                "source_coordinate_mask_coverage_fraction": round(
                    values["selected_source_coordinates"] / max(1, processed * width * height), 8
                ),
            }
            for name, values in totals.items()
        },
        "jaccard_vs_digital_source": {
            name: round(value["intersection"] / value["union"], 6) if value["union"] else None
            for name, value in disagreement.items()
        },
    }
    (output_dir / "source_resolution_report.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Small poster is the only binary artifact; never upload a complete source film.
    poster_rows = []
    labels = [("Original frame", None), ("480p downsample", "downsampled_480p"),
              ("Native source", "digital_source"), ("Lanczos 2x", "lanczos_2x")]
    for index, image, predictions in snapshots:
        tiles = []
        for label, key in labels:
            mask = predictions[key] if key else np.zeros((height, width), dtype=bool)
            tile = overlay(image, mask, f"frame {index:03d}: {label}")
            tile = cv2.resize(tile, (480, int(tile.shape[0] * 480 / width)), interpolation=cv2.INTER_AREA)
            tiles.append(tile)
        poster_rows.append(np.concatenate(tiles, axis=1))
    if poster_rows:
        poster = np.concatenate(poster_rows, axis=0)
        cv2.imwrite(str(output_dir / "source_resolution_contact_sheet.jpg"), poster,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--sensitivity", type=float, default=0.65)
    args = parser.parse_args()
    report = analyze_clip(args.input, args.output_dir, args.max_frames, args.sensitivity)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
