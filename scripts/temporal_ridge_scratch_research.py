#!/usr/bin/env python3
"""Opt-in scratch-mask research: spatial *ridge* AND temporal novelty.

A spatial two-sided ridge test rejects ordinary step edges; a
per-pixel robust temporal percentile rejects persistent scene lines.
Predictions use only the *damaged input sequence*, NEVER clean
references or ground-truth masks. No learned weights or downloads.

CAUTION: Unregistered camera motion / changes in lighting and actual
static scratches can be missed or misclassified. This deliberately
conservative research method is NOT activated in production.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

import restoration_evaluation as ev
from cleanup_video import ai_scratch_mask
from diagnose_scratch_masks import confusion, summarize_confusion, visualize, ssim_psnr_roi


def _as_stack(frames_bgr: list[np.ndarray]) -> np.ndarray:
    if not 3 <= len(frames_bgr) <= 96:
        raise ValueError("Require 3..96 aligned input frames")
    first = frames_bgr[0]
    if first.dtype != np.uint8 or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Expected BGR8 video")
    height, width = first.shape[:2]
    if height < 32 or width < 32 or height * width * len(frames_bgr) > 96_000_000:
        raise ValueError("Invalid geometry or video memory cap exceeded")
    for item in frames_bgr:
        if item.shape != first.shape or item.dtype != np.uint8:
            raise ValueError("Video has inconsistent frame geometry")
    return np.stack([
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        for frame in frames_bgr
    ], axis=0)


def detect_sequence(frames_bgr: list[np.ndarray], *, sensitivity: float = 0.65) -> list[np.ndarray]:
    """Predict full-resolution binary masks without observing GT/reference.

    Fixed hyperparameters established before holdout experiments.
    Run on short clips; no behavior on a single frame is asserted.
    """
    if not math.isfinite(sensitivity) or not 0 <= sensitivity <= 1:
        raise ValueError("Sensitivity must be finite in [0,1]")
    gray = _as_stack(frames_bgr)
    count, height, width = gray.shape

    # For bright/dark transient pixels, reference percentiles use
    # opposite tails to tolerate repeated synthetic scratch positions.
    lower = np.percentile(gray, 20, axis=0).astype(np.float32)
    upper = np.percentile(gray, 80, axis=0).astype(np.float32)

    # Pre-registered conservative spatial/temporal thresholds; not
    # optimized from labels or any held-out source.
    contrast_floor = 19.0 - 7.0 * sensitivity
    novelty_floor = 24.0 - 8.0 * sensitivity
    flank = min(5, max(3, width // 140))
    result: list[np.ndarray] = []
    min_height = max(13, int(round(height * 0.18)))
    vertical_open = max(7, min(13, (height // 28) | 1))

    for index in range(count):
        image = gray[index]
        padded = np.pad(image, ((0, 0), (flank, flank)), mode="edge")
        left = padded[:, :width]
        right = padded[:, 2 * flank:2 * flank + width]

        # Thin bright or dark ridge must contrast with BOTH flanks.
        bright_flanks = np.minimum(image - left, image - right)
        dark_flanks = np.minimum(left - image, right - image)
        bright = (bright_flanks >= contrast_floor) & ((image - lower) >= novelty_floor)
        dark = (dark_flanks >= contrast_floor) & ((upper - image) >= novelty_floor)
        evidence = np.asarray(bright | dark, np.uint8) * 255

        evidence[:, :flank + 1] = 0
        evidence[:, -flank - 1:] = 0
        evidence = cv2.morphologyEx(
            evidence, cv2.MORPH_OPEN, np.ones((vertical_open, 1), np.uint8))
        evidence = cv2.morphologyEx(
            evidence, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
        number, labels, stats, _ = cv2.connectedComponentsWithStats(evidence, 8)
        filtered = np.zeros_like(evidence)
        for component in range(1, number):
            x, y, w, h, area = stats[component]
            if (w <= 7 and h >= min_height and h >= 8 * w
                    and area >= min_height // 2):
                filtered[labels == component] = 255

        # Dilation recovers one/two neighboring pixels from a detected
        # 3px scratch core. It never expands to a full-height column.
        if np.any(filtered):
            filtered = cv2.dilate(filtered, np.ones((1, 3), np.uint8))
        # Fail closed on pathological masks (production detector allows 28%).
        if np.mean(filtered > 0) > 0.03:
            filtered[:] = 0
        result.append(filtered)
    return result


def _mean_masked_mse(values: list[dict], region: str) -> float | None:
    pixels = sum(x[region + "_pixels"] for x in values)
    return (sum(x[region + "_mse"] * x[region + "_pixels"] for x in values
                if x[region + "_mse"] is not None) / pixels) if pixels else None


def benchmark(*, manifest: Path, output_dir: Path,
              split: str, sensitivity: float = 0.65,
              ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    if split not in ("development", "holdout"):
        raise ValueError("Expected explicit development/holdout designation")
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite benchmark")
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if (m.get("schema_version") != 1 or
            m.get("synthetic_degradation") != "vertical_scratches" or
            m.get("reference_type") != "decoded_existing_transfer_not_pristine_camera_negative" or
            m.get("mask_dir") != "masks"):
        raise ValueError("Not a controlled known-mask scratch experiment")
    root = manifest.parent
    info = {key: m[key] for key in ("width", "height", "fps", "frames")}
    clean_file, damaged_file = root / "reference.mkv", root / "degraded.mkv"
    for path, key in ((clean_file, "reference_sha256"), (damaged_file, "degraded_sha256")):
        if not path.is_file() or ev.digest(path) != m[key] or ev.probe(path, ffprobe) != info:
            raise ValueError("Synthetic input/reference integrity or frame count mismatch")
    ref_rgb = list(ev.decode(clean_file, ffmpeg, info))
    damaged_rgb = list(ev.decode(damaged_file, ffmpeg, info))
    gt = []
    for index in range(info["frames"]):
        target = root / "masks" / f"{index:06d}.png"
        actual = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
        if actual is None or actual.shape != (info["height"], info["width"]):
            raise ValueError(f"Ground-truth scratch mask missing/mismatched: {target}")
        gt.append(actual)
    damaged_bgr = [cv2.cvtColor(item, cv2.COLOR_RGB2BGR) for item in damaged_rgb]
    clean_bgr = [cv2.cvtColor(item, cv2.COLOR_RGB2BGR) for item in ref_rgb]
    new_masks = detect_sequence(damaged_bgr, sensitivity=sensitivity)
    old_masks = [ai_scratch_mask(frame, sensitivity=sensitivity) for frame in damaged_bgr]
    clean_new = detect_sequence(clean_bgr, sensitivity=sensitivity)
    clean_old = [ai_scratch_mask(frame, sensitivity=sensitivity) for frame in clean_bgr]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arp-ridge-", dir=output_dir.parent) as td:
        stage = Path(td)
        per_method = {}
        for label, masks, neg in (("existing_production", old_masks, clean_old),
                                  ("research_temporal_ridge", new_masks, clean_new)):
            counts = [confusion(p, t) for p, t in zip(masks, gt)]
            repair_inside, repair_outside = [], []
            for src, corrupt, pred, known in zip(ref_rgb, damaged_rgb, masks, gt):
                repaired = cv2.cvtColor(
                    cv2.inpaint(cv2.cvtColor(corrupt, cv2.COLOR_RGB2BGR),
                                pred, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
                val = ssim_psnr_roi(src, repaired, known)
                repair_inside.append(val)
                repair_outside.append(val)
            per_method[label] = {
                "mask_metrics": summarize_confusion(counts),
                "per_frame_counts": counts,
                "negative_control_coverage_mean": float(np.mean([np.mean(x > 127) for x in neg])),
                "telea_known_scratch_mse": _mean_masked_mse(repair_inside, "known_scratch"),
                "telea_outside_scratch_mse": _mean_masked_mse(repair_outside, "outside_known_scratch"),
            }
        # Difference maps from actual data show both false positives and missed pixels.
        for index in sorted(set((0, len(gt)//2, len(gt)-1))):
            for label, masks in (("existing_production", old_masks),
                                 ("research_temporal_ridge", new_masks)):
                contact = visualize(ref_rgb[index], damaged_rgb[index], gt[index],
                                    masks[index],
                                    title=f"{m['film']} / {m['scene']} | {split} | {label}")
                path = stage / f"{label}_{index:06d}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR)):
                    raise IOError("Could not write diagnostic contact sheet")
        report = {
            "schema_version": 1,
            "task": "research_temporal_ridge_scratch_mask",
            "split": split,
            "film": m["film"], "scene": m["scene"],
            "original_type": m["reference_type"], "synthetic_degradation": m["synthetic_degradation"],
            "sensitivity": sensitivity, "model_selection_on_holdout": False,
            "manifest_sha256": ev.digest(manifest),
            "source_damaged_sha256": m["degraded_sha256"],
            "source_reference_sha256": m["reference_sha256"],
            "frames": info["frames"], "methods": per_method,
            "limitations": [
                "Temporal filtering uses only damaged frames, no clean/reference data at inference.",
                "Fixed spatial ridge / temporal percentile assumptions can miss static scratches and fail under motion.",
                "Synthetic full-height scratches are not representative of all historic film defects.",
                "No model or threshold is optimized based on independent holdout result.",
                "This is a CPU research-only comparison; production scratch mask and DART/ProPainter not changed.",
            ],
        }
        (stage / "comparison.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Diagnostic output appeared during run")
        stage.rename(output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--sensitivity", type=float, default=0.65)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    data = benchmark(manifest=args.manifest, output_dir=args.output_dir,
                     split=args.split, sensitivity=args.sensitivity,
                     ffmpeg=ev.find_binary(args.ffmpeg), ffprobe=ev.find_binary(args.ffprobe))
    for name, item in data["methods"].items():
        print(name, json.dumps(item["mask_metrics"], sort_keys=True))
    print("source", data["film"], data["scene"], "split", args.split)


if __name__ == "__main__":
    main()
