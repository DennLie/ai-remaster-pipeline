"""Evaluate externally generated DART masks AND restored frames side by side.

This module never imports, downloads or runs DART/RTN. Its optional
mask-gated composite is EXPERIMENTAL and RESEARCH-ONLY: matching output
files or high mask scores do not grant any commercial model rights.

Input: a short same-shot video OR a directory of source PNGs.
DART outputs: <dart-dir>/frames/<name>.png and masks/<name>.png.
Masks are soft 8-bit confidences. No size, crop, or time interpolation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator, Any

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
DEFAULT_THRESHOLD = 160
CHANGE_THRESHOLD = 8


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_frames(source: Path, max_frames: int | None = None) -> Iterator[tuple[str, np.ndarray]]:
    """Follow DART's official iter_frames() naming and color convention."""
    if max_frames is not None and max_frames < 2:
        raise ValueError("DART requires at least two consecutive frames")
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"No image frames found in {source}")
        for i, path in enumerate(files):
            if max_frames is not None and i >= max_frames:
                break
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot decode source frame {path}")
            yield path.stem, bgr
    elif source.suffix.lower() in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot decode source video {source}")
        try:
            i = 0
            while max_frames is None or i < max_frames:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield f"frame_{i:06d}", bgr
                i += 1
        finally:
            cap.release()
    else:
        raise ValueError(f"Source must be a video file or frame directory: {source}")


def read_restored(path: Path, target_shape: tuple[int, int, int]) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(f"Missing DART restored frame: {path}")
    if bgr.dtype != np.uint8 or bgr.shape != target_shape:
        raise ValueError(f"DART frame must be exact uint8 3-channel geometry {target_shape}: {path} got {bgr.shape}")
    return bgr


def read_mask(path: Path, image_shape: tuple[int, int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Missing DART soft mask: {path}")
    if mask.ndim != 2 or mask.dtype != np.uint8 or mask.shape != image_shape[:2]:
        raise ValueError(f"DART mask must be uint8 single-channel source geometry {image_shape[:2]}: {path}")
    return mask


def masked_composite(original: np.ndarray, restored: np.ndarray, mask: np.ndarray, threshold: int = DEFAULT_THRESHOLD) -> np.ndarray:
    """Keep exact source pixels wherever DART confidence < threshold."""
    if not 1 <= threshold <= 255:
        raise ValueError("threshold must be from 1 to 255")
    if original.dtype != np.uint8 or restored.dtype != np.uint8 or mask.dtype != np.uint8:
        raise ValueError("Expected uint8 images")
    if original.shape != restored.shape or original.shape[:2] != mask.shape or original.ndim != 3:
        raise ValueError("Image/mask geometries must match exactly")
    # Binary gate is intentional: feathering would modify unmasked pixels.
    return np.where((mask >= threshold)[..., None], restored, original)


def scores(original: np.ndarray, restored: np.ndarray, mask: np.ndarray, threshold: int) -> dict[str, int | float]:
    changed = np.max(cv2.absdiff(original, restored), axis=2) >= CHANGE_THRESHOLD
    selected = mask >= threshold
    selected_count = int(np.count_nonzero(selected))
    unselected_count = int(np.count_nonzero(~selected))
    return {
        "pixels": mask.size,
        "mask_pixels": selected_count,
        "changed_masked": int(np.count_nonzero(changed & selected)),
        "changed_unmasked": int(np.count_nonzero(changed & ~selected)),
        "mean_abs_difference": round(float(np.mean(cv2.absdiff(original, restored))), 5),
        "max_abs_difference": int(np.max(cv2.absdiff(original, restored))),
        "unmasked_pixels": unselected_count,
    }


def sheet_tile(original: np.ndarray, restored: np.ndarray, mask: np.ndarray, threshold: int, name: str) -> np.ndarray:
    h, w = mask.shape
    overlay = original.copy()
    chosen = mask >= threshold
    overlay[chosen, 0] = (overlay[chosen, 0] // 4)
    overlay[chosen, 1] = (overlay[chosen, 1] // 4)
    overlay[chosen, 2] = 255
    difference = cv2.absdiff(original, restored)
    difference = cv2.applyColorMap(np.max(difference, axis=2), cv2.COLORMAP_TURBO)
    composite = masked_composite(original, restored, mask, threshold)
    panels = [
        (original, "source"), (restored, "DART restored"),
        (overlay, "mask overlay"), (composite, "masked composite"),
        (difference, "difference heatmap"),
    ]
    tiles = []
    for image, label in panels:
        small = cv2.resize(image, (320, max(120, int(h * 320 / w))), interpolation=cv2.INTER_AREA)
        footer = np.full((26, small.shape[1], 3), 25, np.uint8)
        cv2.putText(footer, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(np.concatenate([small, footer], axis=0))
    return np.concatenate(tiles, axis=1)


def evaluate(
    source: Path,
    dart_output: Path,
    output_dir: Path,
    *,
    threshold: int = DEFAULT_THRESHOLD,
    max_frames: int | None = None,
    write_masked_composite: bool = False,
    research_only: bool = False,
) -> dict[str, Any]:
    if not 1 <= threshold <= 255:
        raise ValueError("threshold must be 1..255")
    if write_masked_composite and not research_only:
        raise ValueError("Masked DART composition is research-only and requires --research-only")
    if not (dart_output / "frames").is_dir() or not (dart_output / "masks").is_dir():
        raise FileNotFoundError("Expected DART output folders 'frames/' and 'masks/'")
    if output_dir.resolve() == dart_output.resolve():
        raise ValueError("The report directory must differ from the original DART outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    if write_masked_composite:
        (output_dir / "masked_composite_frames").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    gallery: list[tuple[float, int, np.ndarray]] = []
    expected_names: set[str] = set()
    first_shape = None
    for name, original in source_frames(source, max_frames):
        if name in expected_names:
            raise ValueError(f"Duplicate source frame name: {name}")
        expected_names.add(name)
        if first_shape is None:
            first_shape = original.shape
        elif first_shape != original.shape:
            raise ValueError("Variable-resolution source frames are not supported")
        restored = read_restored(dart_output / "frames" / f"{name}.png", original.shape)
        mask = read_mask(dart_output / "masks" / f"{name}.png", original.shape)
        row = scores(original, restored, mask, threshold)
        row["frame"] = name
        rows.append(row)
        if write_masked_composite:
            image = masked_composite(original, restored, mask, threshold)
            if not cv2.imwrite(str(output_dir / "masked_composite_frames" / f"{name}.png"), image):
                raise IOError(f"Could not write masked composite {name}")
        # Score using changed intact pixels, so the poster prioritizes
        # potential preservation problems rather than just damaged areas.
        rank = row["changed_unmasked"] / max(1, row["unmasked_pixels"])
        gallery.append((rank, len(rows), sheet_tile(original, restored, mask, threshold, name)))
        gallery.sort(key=lambda item: item[0], reverse=True)
        if len(gallery) > 3:
            gallery.pop()
    if len(rows) < 2:
        raise ValueError("DART evaluation requires at least two frames")

    # No stale, unrelated frame files can silently distort the diagnosis.
    for dirname in ("frames", "masks"):
        present = {p.stem for p in (dart_output / dirname).glob("*.png")}
        extra = present - expected_names
        if extra:
            raise ValueError(f"DART {dirname} has unexpected frames: {sorted(extra)[:8]}")

    total = {key: sum(int(row[key]) for row in rows)
             for key in ("pixels", "mask_pixels", "changed_masked", "changed_unmasked", "unmasked_pixels")}
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "DART mask+restoration quality-control diagnostics, NOT commercial clearance or accuracy",
        "source": source.name,
        "source_file_sha256": file_sha256(source) if source.is_file() else None,
        "dart_export": str(dart_output),
        "frames": len(rows),
        "width": first_shape[1],
        "height": first_shape[0],
        "threshold": threshold,
        "change_detection_threshold_8bit": CHANGE_THRESHOLD,
        "mask_coverage_fraction": round(total["mask_pixels"] / total["pixels"], 7),
        "changed_unmasked_fraction": round(total["changed_unmasked"] / max(1, total["unmasked_pixels"]), 7),
        "changed_masked_fraction": round(total["changed_masked"] / max(1, total["mask_pixels"]), 7),
        "commercially_cleared": False,
        "masked_composite_written": write_masked_composite,
        "gated_pixels_unchanged_outside_mask": bool(write_masked_composite),
        "warning": "Unmasked-change rate is a fidelity diagnostic, not a defect-repair success score. Real changes can be desirable; masked changes can be wrong. Evaluate annotated ground truth and actual footage.",
        "totals": total,
        "per_frame": rows,
    }
    gallery.sort(key=lambda item: item[1])
    poster = np.concatenate([item[2] for item in gallery], axis=0)
    if not cv2.imwrite(str(output_dir / "dart_mask_restoration_contact_sheet.jpg"), poster,
                       [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise IOError("Could not write DART quality-control sheet")
    (output_dir / "dart_restoration_diagnostics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Original video or source frame folder")
    parser.add_argument("--dart-output", required=True, type=Path, help="DART export with frames/ and masks/")
    parser.add_argument("--output-dir", required=True, type=Path, help="QC report and image directory")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="DART soft mask gate 1..255")
    parser.add_argument("--max-frames", type=int, default=None, help="Match DART's --max-frames exactly")
    parser.add_argument("--write-masked-composite", action="store_true", help="Research-only image outputs; use with --research-only")
    parser.add_argument("--research-only", action="store_true", help="Acknowledge no commercial redistribution clearance")
    args = parser.parse_args(argv)
    result = evaluate(args.source, args.dart_output, args.output_dir, threshold=args.threshold,
                      max_frames=args.max_frames, write_masked_composite=args.write_masked_composite,
                      research_only=args.research_only)
    print(json.dumps({k: v for k, v in result.items() if k not in {"per_frame", "totals"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
