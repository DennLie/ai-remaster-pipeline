"""Benchmark pixel-accurate archival film defect masks from multiple methods.

This tool DOES NOT run or import DART/RTN. Evaluate masks exported separately
by any model against manually labeled ground truth. PNG 0=clean, 255=defect,
127=ignore in ground truth; predicted PNG values 0..255 are thresholded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REQUIRED_COLUMNS = ("clip", "frame", "split", "damage_type")
GT_VALUES = (0, 127, 255)


def validate_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Invalid {label} {value!r}: only one safe folder or file name is allowed")
    return value


def load_samples(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not all(field in reader.fieldnames for field in REQUIRED_COLUMNS):
            raise ValueError(f"Manifest needs columns: {', '.join(REQUIRED_COLUMNS)}")
        rows = []
        seen: set[tuple[str, str]] = set()
        for row in reader:
            clip = validate_component((row.get("clip") or "").strip(), "clip")
            frame = validate_component((row.get("frame") or "").strip(), "frame")
            if not frame.lower().endswith(".png"):
                raise ValueError(f"Frame names must be PNG filenames: {frame!r}")
            identity = (clip, frame)
            if identity in seen:
                raise ValueError(f"Duplicate sample in manifest: {identity}")
            seen.add(identity)
            split = (row.get("split") or "").strip()
            damage_type = (row.get("damage_type") or "").strip()
            if not split or not damage_type:
                raise ValueError(f"Missing split or damage_type for {identity}")
            rows.append({"clip": clip, "frame": frame, "split": split, "damage_type": damage_type})
    if not rows:
        raise ValueError("No benchmark samples found")
    return rows


def read_mask(path: Path, *, ground_truth: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Missing or unreadable PNG mask: {path}")
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"Mask must be single-channel 8-bit PNG (not RGB or 16-bit): {path}")
    if ground_truth:
        unexpected = np.setdiff1d(np.unique(image), GT_VALUES)
        if unexpected.size:
            raise ValueError(
                f"Ground truth must use only 0=clean, 127=ignore, 255=damaged: {path} "
                f"(found values {unexpected.tolist()[:10]})"
            )
    return image


def confusion(gt: np.ndarray, pred: np.ndarray, threshold: int = 128) -> dict[str, int]:
    if gt.shape != pred.shape:
        raise ValueError(f"Mask dimensions differ: gt={gt.shape}, prediction={pred.shape}; never resize silently")
    if gt.dtype != np.uint8 or pred.dtype != np.uint8 or gt.ndim != 2:
        raise ValueError("Expected two single-channel uint8 masks")
    if not 1 <= threshold <= 255:
        raise ValueError("Threshold must be between 1 and 255")
    valid = gt != 127
    true_defect = (gt == 255) & valid
    predicted_defect = (pred >= threshold) & valid
    tp = int(np.count_nonzero(true_defect & predicted_defect))
    fp = int(np.count_nonzero(~true_defect & predicted_defect))
    fn = int(np.count_nonzero(true_defect & ~predicted_defect & valid))
    tn = int(np.count_nonzero(~true_defect & ~predicted_defect & valid))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ignored": int(np.count_nonzero(~valid))}


def metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (counts.get(k, 0) for k in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    return {
        **counts,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(2 * tp / (2 * tp + fp + fn), 6) if tp + fp + fn else 1.0,
        "iou": round(iou, 6),
        "false_positive_rate": round(fp / (fp + tn), 8) if fp + tn else 0.0,
        "prediction_coverage": round((tp + fp) / total, 6) if total else 0.0,
        "ground_truth_coverage": round((tp + fn) / total, 6) if total else 0.0,
        "evaluated_pixels": total,
    }


def sum_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn", "ignored")}


def evaluate(
    samples: list[dict[str, str]],
    ground_truth_root: Path,
    predictions: dict[str, Path],
    *,
    threshold: int = 128,
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("At least one prediction method is required")
    result: dict[str, Any] = {"schema_version": 1, "threshold": threshold, "samples": len(samples), "methods": {}}
    for method, root in sorted(predictions.items()):
        rows: list[dict[str, Any]] = []
        for sample in samples:
            relpath = Path(sample["clip"]) / sample["frame"]
            ground_truth = read_mask(ground_truth_root / relpath, ground_truth=True)
            prediction = read_mask(root / relpath, ground_truth=False)
            counts = confusion(ground_truth, prediction, threshold)
            rows.append({**sample, **metrics(counts)})
        summary = metrics(sum_counts(rows))
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for field in ("clip", "split", "damage_type"):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[row[field]].append(row)
            grouped[field] = {
                key: {"samples": len(items), **metrics(sum_counts(items))}
                for key, items in sorted(groups.items())
            }
        result["methods"][method] = {"summary": summary, "by": grouped, "frames": rows}
    return result


def parse_methods(entries: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in entries:
        name, separator, directory = item.partition("=")
        if not separator or not name.strip() or not directory.strip():
            raise ValueError(f"Specify each method as NAME=FOLDER, not {item!r}")
        name = validate_component(name.strip(), "method")
        if name in result:
            raise ValueError(f"Duplicate prediction method {name!r}")
        result[name] = Path(directory)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="CSV clip,frame,split,damage_type")
    parser.add_argument("--ground-truth", type=Path, required=True, help="GT PNG root, organized clip/frame.png")
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=FOLDER",
                        help="Prediction root arranged clip/frame.png; repeat for each method")
    parser.add_argument("--threshold", type=int, default=128, help="Soft mask threshold 1..255 (default 128)")
    parser.add_argument("--report", type=Path, required=True, help="JSON report path; no report if validation fails")
    args = parser.parse_args(argv)
    if not 1 <= args.threshold <= 255:
        parser.error("--threshold must be between 1 and 255")
    data = evaluate(load_samples(args.manifest), args.ground_truth, parse_methods(args.prediction),
                    threshold=args.threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, value in data["methods"].items():
        summary = value["summary"]
        print(f"{name}: IoU={summary['iou']:.3f} F1={summary['f1']:.3f} "
              f"precision={summary['precision']:.3f} recall={summary['recall']:.3f} "
              f"intact-pixel FPR={summary['false_positive_rate']:.6f}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
