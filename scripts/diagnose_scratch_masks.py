#!/usr/bin/env python3
"""Auditable scratch-mask diagnosis on *known* synthetic defects.

Evaluate the existing ARP CPU scratch detector across FIXED sensitivity
settings, against exact per-pixel ground truth for synthetic scratches.
Provide per-frame errors and sampled RGB+mask overlays; compare
classical Telea inpainting using the predicted mask against a
non-deployable oracle GT mask to isolate mask vs inpainting failures.

The clean digital transfer is a NEGATIVE CONTROL for detector activity,
not a true 'no historical damage' guarantee. Sensitivity sweeps on the
test films are diagnostics only -- not validated preset selection.
No model weights, DART inference, commercial clearance, or changes
to the existing production scratch detector.
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

DEFAULT_SENSITIVITIES = (0.25, 0.45, 0.65, 0.85)


def sensitivities_arg(text: str) -> list[float]:
    try:
        values = [float(x.strip()) for x in text.split(",")]
    except ValueError as e:
        raise ValueError("Sensitivities must be comma-separated 0..1 values") from e
    if not values or len(values) > 12 or len(set(values)) != len(values):
        raise ValueError("Expected 1..12 unique sensitivity values")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("All sensitivities must be finite and in 0..1")
    return values


def confusion(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    if pred.ndim != 2 or pred.shape != truth.shape:
        raise ValueError("Invalid synthetic scratch prediction shape")
    p, t = pred > 127, truth > 127
    return {
        "tp": int(np.count_nonzero(p & t)),
        "fp": int(np.count_nonzero(p & ~t)),
        "fn": int(np.count_nonzero(~p & t)),
        "tn": int(np.count_nonzero(~p & ~t)),
    }


def summarize_confusion(counts: list[dict[str, int]]) -> dict:
    if not counts:
        raise ValueError("No frames evaluated")
    keys = ("tp", "fp", "fn", "tn")
    total = {k: sum(v[k] for v in counts) for k in keys}
    tp, fp, fn, tn = (total[k] for k in keys)
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    total_pixels = tp + fp + fn + tn
    return {
        **total, "precision": precision, "recall": recall, "pixel_f1": f1,
        "pixel_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.,
        "false_positive_fraction_of_healthy_pixels": fp / (fp + tn) if fp + tn else 0.,
        "predicted_coverage_fraction": (tp + fp) / total_pixels,
        "synthetic_ground_truth_fraction": (tp + fn) / total_pixels,
    }


def ssim_psnr_roi(reference: np.ndarray, estimate: np.ndarray,
                  mask: np.ndarray) -> dict:
    """Reference RGB squared errors inside/outside GT in gamma-encoded 8-bit space."""
    if reference.shape != estimate.shape or mask.shape != reference.shape[:2]:
        raise ValueError("Reference/estimate/mask shape mismatch")
    errors = np.mean(((reference.astype(np.float32) -
                       estimate.astype(np.float32)) / 255.) ** 2, axis=2)
    result = {}
    for label, region in (("known_scratch", mask > 127),
                          ("outside_known_scratch", mask <= 127)):
        result[label + "_pixels"] = int(np.count_nonzero(region))
        result[label + "_mse"] = float(np.mean(errors[region])) if np.any(region) else None
    return result


def visualize(clean: np.ndarray, damaged: np.ndarray, gt: np.ndarray,
              predicted: np.ndarray, *, title: str) -> np.ndarray:
    """Single frame RGB reference / synthetic damage / GT / TP-FP-FN overlay."""
    if not (clean.shape == damaged.shape and clean.shape[:2] == gt.shape == predicted.shape):
        raise ValueError("Mismatched contact sheet shapes")
    h, w = gt.shape
    truth, pred = gt > 127, predicted > 127
    overlay = damaged.copy()
    overlay[truth & pred] = [0, 220, 0]  # green true positives
    overlay[~truth & pred] = [245, 48, 48]  # red false positives
    overlay[truth & ~pred] = [55, 105, 255]  # blue false negatives
    gt_rgb = np.repeat(gt[..., None], 3, axis=2)
    sheet = np.zeros((h + 34, w * 4, 3), np.uint8)
    for idx, image in enumerate((clean, damaged, gt_rgb, overlay)):
        sheet[34:, idx * w:(idx + 1) * w] = image
    cv2.putText(sheet, title[:70], (6, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (255, 255, 255), 1, cv2.LINE_AA)
    labels = ("Digital reference", "Synthetic damage", "Known GT", "TP green / FP red / FN blue")
    for idx, label in enumerate(labels):
        cv2.putText(sheet, label, (idx * w + 5, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                    (235, 235, 235), 1, cv2.LINE_AA)
    return sheet


def diagnose(*, manifest: Path, output_dir: Path,
             sensitivities: list[float] | None = None,
             ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Diagnostic output exists; refusing overwrite")
    settings = sensitivities if sensitivities is not None else list(DEFAULT_SENSITIVITIES)
    if (not settings or len(settings) > 12 or len(set(settings)) != len(settings) or
            any(not math.isfinite(x) or x < 0 or x > 1 for x in settings)):
        raise ValueError("Invalid/duplicate detector sensitivity")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (data.get("schema_version") != 1 or
            data.get("synthetic_degradation") != "vertical_scratches" or
            data.get("reference_type") != "decoded_existing_transfer_not_pristine_camera_negative" or
            data.get("mask_dir") != "masks"):
        raise ValueError("Only scratch ground truth from the scientific synthetic harness is supported")
    root = manifest.parent
    clean, damaged, masks_dir = root / "reference.mkv", root / "degraded.mkv", root / "masks"
    meta = {k: data[k] for k in ("width", "height", "fps", "frames")}
    for path, sha in ((clean, data["reference_sha256"]), (damaged, data["degraded_sha256"])):
        if not path.is_file() or ev.digest(path) != sha or ev.probe(path, ffprobe) != meta:
            raise ValueError(f"Reference/synthetic source was modified: {path}")
    original_frames = list(ev.decode(clean, ffmpeg, meta))
    damaged_frames = list(ev.decode(damaged, ffmpeg, meta))
    truth = []
    for i in range(meta["frames"]):
        mask_path = masks_dir / f"{i:06d}.png"
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gt is None or gt.shape != (meta["height"], meta["width"]) or not np.all((gt == 0) | (gt == 255)):
            raise ValueError(f"Ground truth mask missing/malformed: {mask_path}")
        truth.append(gt)
    reference_gt_hashes = [ev.digest(masks_dir / f"{i:06d}.png") for i in range(meta["frames"])]
    sample_indices = sorted(set((0, meta["frames"] // 2, meta["frames"] - 1)))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tempfile.TemporaryDirectory(prefix=".arp-scratch-diagnostic-", dir=output_dir.parent) as td:
        stage = Path(td)
        for strength in settings:
            predictions, clean_detections, frame_confusions = [], [], []
            ablated_predictions, ablated_confusions, ablated_clean_coverage = [], [], []
            detector_inpaint_error, oracle_inpaint_error, ablated_inpaint_error = [], [], []
            for i, (src, dst, gt) in enumerate(zip(original_frames, damaged_frames, truth)):
                pred = ai_scratch_mask(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR), sensitivity=strength)
                if pred.shape != gt.shape or pred.dtype != np.uint8:
                    raise ValueError("Real detector returned unexpected shape/type")
                clean_pred = ai_scratch_mask(cv2.cvtColor(src, cv2.COLOR_RGB2BGR), sensitivity=strength)
                if clean_pred.shape != gt.shape:
                    raise ValueError("Clean-transfer negative control shape mismatch")
                ablated = ai_scratch_mask(
                    cv2.cvtColor(dst, cv2.COLOR_RGB2BGR), sensitivity=strength,
                    full_height_column_fallback=False)
                ablated_clean = ai_scratch_mask(
                    cv2.cvtColor(src, cv2.COLOR_RGB2BGR), sensitivity=strength,
                    full_height_column_fallback=False)
                ablated_predictions.append(ablated)
                ablated_confusions.append(confusion(ablated, gt))
                ablated_clean_coverage.append(float(np.mean(ablated_clean > 127)))
                predictions.append(pred)
                clean_detections.append(float(np.mean(clean_pred > 127)))
                frame_confusions.append(confusion(pred, gt))
                detector_repair = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR),
                                                          pred, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
                oracle_repair = cv2.cvtColor(cv2.inpaint(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR),
                                                        gt, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
                ablated_repair = cv2.cvtColor(
                    cv2.inpaint(cv2.cvtColor(dst, cv2.COLOR_RGB2BGR),
                                ablated, 3, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
                detector_inpaint_error.append(ssim_psnr_roi(src, detector_repair, gt))
                ablated_inpaint_error.append(ssim_psnr_roi(src, ablated_repair, gt))
                oracle_inpaint_error.append(ssim_psnr_roi(src, oracle_repair, gt))
            aggregate = summarize_confusion(frame_confusions)
            label = f"sensitivity_{strength:.2f}"
            destination = stage / label
            destination.mkdir()
            # Contact sheets are a *diagnostic* sampling, not a replacement
            # for blinded, independently annotated full-length inspection.
            for idx in sample_indices:
                sheet = visualize(original_frames[idx], damaged_frames[idx],
                                  truth[idx], predictions[idx],
                                  title=f"{data['film']} / {data['scene']} | frame={idx} | sens={strength:.2f}")
                if not cv2.imwrite(str(destination / f"overlay_{idx:06d}.png"),
                                   cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)):
                    raise IOError("Could not export scratch diagnostics")
                if abs(strength - 0.65) < 1e-9:
                    experimental = visualize(original_frames[idx], damaged_frames[idx],
                                             truth[idx], ablated_predictions[idx],
                                             title=f"{data['film']} / {data['scene']} | frame={idx} | column fallback OFF")
                    if not cv2.imwrite(str(destination / f"no_column_fallback_{idx:06d}.png"),
                                       cv2.cvtColor(experimental, cv2.COLOR_RGB2BGR)):
                        raise IOError("Could not export column fallback ablation")
            def pooled_mse(values: list[dict], region: str) -> float | None:
                pixels = sum(d[region + "_pixels"] for d in values)
                return (sum(d[region + "_mse"] * d[region + "_pixels"] for d in values
                            if d[region + "_mse"] is not None) / pixels) if pixels else None
            entry = {
                "sensitivity": strength, "mask_scores": aggregate,
                "clean_transfer_detector_coverage_fraction_mean": float(np.mean(clean_detections)),
                "per_frame_confusion": frame_confusions,
                "no_full_height_column_fallback_ablation": {
                    "mask_scores": summarize_confusion(ablated_confusions),
                    "clean_transfer_detector_coverage_fraction_mean": float(np.mean(ablated_clean_coverage)),
                    "telea_known_scratch_mse": pooled_mse(ablated_inpaint_error, "known_scratch"),
                    "telea_outside_gt_mse": pooled_mse(ablated_inpaint_error, "outside_known_scratch"),
                    "experimental_only": True,
                },
                "telea_using_detector_mask_known_scratch_mse": pooled_mse(detector_inpaint_error, "known_scratch"),
                "telea_using_oracle_gt_mask_known_scratch_mse": pooled_mse(oracle_inpaint_error, "known_scratch"),
                "telea_using_detector_mask_outside_gt_mse": pooled_mse(detector_inpaint_error, "outside_known_scratch"),
                "telea_using_oracle_gt_mask_outside_gt_mse": pooled_mse(oracle_inpaint_error, "outside_known_scratch"),
                "oracle_is_non_deployable": True,
                "visual_samples": [f"{label}/overlay_{i:06d}.png" for i in sample_indices],
            }
            entries.append(entry)
        result = {
            "schema_version": 1, "experiment": "ARP-scratch-mask-fixed-sensitivity-audit",
            "film": data["film"], "scene": data["scene"],
            "original_reference_type": data["reference_type"],
            "synthetic_manifest_sha256": ev.digest(manifest),
            "degraded_sha256": data["degraded_sha256"],
            "reference_sha256": data["reference_sha256"],
            "ground_truth_png_sha256": reference_gt_hashes,
            "frame_count": meta["frames"], "geometry": [meta["width"], meta["height"]],
            "settings": entries,
            "limitations": [
                "Sensitivity sweep is exploratory on evaluation data; do NOT select/tune a production threshold from these same test clips.",
                "Clean digital reference can contain actual historic scratches; flagged clean pixels are a detector activity control, not certified false positives.",
                "Oracle masks use withheld synthetic ground truth, cannot be generated at inference, and demonstrate only possible inpainting upper-bound/control.",
                "Synthetic straight scratches do not establish real-film accuracy or DART/ProPainter GPU reconstruction quality.",
                "No production detector, model weights, source movies or default restoration settings are changed.",
            ],
        }
        (stage / "scratch_diagnostics.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError("Diagnostic target appeared during computation")
        stage.rename(output_dir)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--sensitivities", default="0.25,0.45,0.65,0.85")
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args()
    result = diagnose(manifest=args.manifest, output_dir=args.output_dir,
                      sensitivities=sensitivities_arg(args.sensitivities),
                      ffmpeg=ev.find_binary(args.ffmpeg),
                      ffprobe=ev.find_binary(args.ffprobe))
    print(json.dumps({
        "film": result["film"], "scene": result["scene"],
        "settings": [{
            "sensitivity": x["sensitivity"],
            "precision": x["mask_scores"]["precision"],
            "recall": x["mask_scores"]["recall"],
            "f1": x["mask_scores"]["pixel_f1"],
            "clean_transfer_detector_coverage": x["clean_transfer_detector_coverage_fraction_mean"],
        } for x in result["settings"]],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
