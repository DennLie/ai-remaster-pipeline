#!/usr/bin/env python3
"""Evidence-oriented automated evaluation adapters for ARP stages 1/2/4/5.

Stage 1 uses the exact CPU Light Deflicker gain estimator from PR #6.
Stage 2 invokes the real FFmpeg HQDN3D/NLMeans/BM3D ARP backend.
Stage 4 invokes the real ARP libvidstab single-shot video backend and
reports both known-reference image fidelity and residual translation.
Stage 5 evaluates ARP's ACTUAL CPU scratch mask detector, plus an
explicitly separate OpenCV inpainting baseline. ProPainter/DART AI
inference is never mislabeled as having run.
All runs require a known-reference *synthetic* damage manifest.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

import restoration_evaluation as ev

TASKS = {
    "deflicker": "temporal_flicker",
    "denoise": "read_noise",
    "stabilize": "camera_jitter",
    "scratch_masks": "vertical_scratches",
}


def assert_experiment(manifest: Path, stage: str, ffprobe: str) -> tuple[dict, Path]:
    if stage not in TASKS:
        raise ValueError(f"Unsupported evaluation stage {stage}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (data.get("schema_version") != ev.SCHEMA
            or data.get("reference_type") != "decoded_existing_transfer_not_pristine_camera_negative"
            or data.get("synthetic_degradation") != TASKS[stage]):
        raise ValueError(f"{stage} needs its exact synthetic {TASKS[stage]} manifest")
    damaged, clean = manifest.parent / "degraded.mkv", manifest.parent / "reference.mkv"
    info = {k: data[k] for k in ("width", "height", "fps", "frames")}
    for path, checksum in ((damaged, data["degraded_sha256"]),
                           (clean, data["reference_sha256"])):
        if not path.is_file() or ev.digest(path) != checksum or ev.probe(path, ffprobe) != info:
            raise ValueError(f"Synthetic source/ground truth mismatch: {path}")
    return data, damaged


def sign(target: Path, source_sha: str, *, method: str, args: dict) -> None:
    """Research-only source identity sidecar, matching existing ARP scorer."""
    sidecar = target.with_suffix(target.suffix + ".sig.json")
    sidecar.write_text(json.dumps({
        "tool": "evaluate_steps_1to5.py", "version": 1,
        "source_fingerprint": {"sha256": source_sha},
        "method": method, "parameters": args,
        "output_sha256": ev.digest(target),
    }, indent=2) + "\n", encoding="utf-8")


def deflicker(*, source: Path, target: Path, ffmpeg: str, ffprobe: str,
              max_correction: float = 0.08, strength: float = 0.65) -> dict:
    from film_deflicker import apply_gain, calculate_gains, frame_luma
    info = ev.probe(source, ffprobe)
    frames = list(ev.decode(source, ffmpeg, info))
    # Model is intentionally the PR #6 estimator; OpenCV expects BGR.
    bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    data = [frame_luma(f) for f in bgr]
    gains, cuts = calculate_gains(
        [x[0] for x in data], [x[1] for x in data],
        strength=strength, max_correction=max_correction,
    )
    results = [
        cv2.cvtColor(apply_gain(f, g), cv2.COLOR_BGR2RGB)
        for f, g in zip(bgr, gains)
    ]
    ev.encode(target, ffmpeg, info, results)
    sign(target, ev.digest(source), method="light_deflicker_pr6",
         args={"strength": strength, "max_correction": max_correction})
    return {
        "source_branch": "feature/light-deflicker-cpu (draft PR #6)",
        "nonneutral_frames": sum(abs(g - 1) > 1e-4 for g in gains),
        "max_gain_deviation": max(abs(g - 1) for g in gains),
        "shot_starts": cuts, "strength": strength,
        "max_correction": max_correction,
    }


def denoise(*, source: Path, directory: Path, ffmpeg: str,
            methods: list[str], presets: list[str]) -> tuple[dict, dict]:
    import denoise_video
    candidates, notes = {}, {}
    for method in methods:
        for preset in (["gentle"] if method == "off" else presets):
            name = f"{method}_{preset}"
            path = directory / f"{name}.mkv"
            result = denoise_video.run(
                source, path, method=method, preset=preset,
                intermediate_profile="lossless", ffmpeg=ffmpeg,
            )
            candidates[name] = path
            notes[name] = {"backend": "scripts/denoise_video.py", **result}
    return candidates, notes


def stabilize(*, source: Path, target: Path, ffmpeg: str) -> dict:
    import stabilize_video
    args = stabilize_video.build_parser().parse_args([
        "--source", str(source), "--output", str(target),
        "--single-shot", "--smoothing", "3", "--zoom", "0",
        "--intermediate-profile", "lossless", "--ffmpeg", ffmpeg,
    ])
    if stabilize_video.main_with_args(args) != 0:
        raise RuntimeError("ARP libvidstab stabilization failed")
    return {"backend": "scripts/stabilize_video.py", "single_shot": True,
            "smoothing": 3, "zoom": 0,
            "caveat": "Stabilized frames can be resampled/cropped; motion preservation and geometric damage must be inspected"}


def residual_alignment(reference_frames: list, frames: list) -> dict:
    """Paired frame-to-reference translation diagnostic on a central crop.

    NOT an optical-flow/warp error and not comparable across dissimilar scenes.
    Phase correlation may fail for featureless images, strong motion blur or
    altered texture. We expose response confidence and never hide failures.
    """
    if len(reference_frames) != len(frames):
        raise ValueError("Paired scene length mismatch")
    displacements = []
    for a, b in zip(reference_frames, frames):
        if a.shape != b.shape:
            raise ValueError("Stabilization changed frame geometry")
        h, w = a.shape[:2]
        margin_y, margin_x = max(4, h // 8), max(4, w // 8)
        window = (slice(margin_y, h - margin_y), slice(margin_x, w - margin_x))
        first = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)[window].astype(np.float32)
        second = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)[window].astype(np.float32)
        taper = cv2.createHanningWindow((first.shape[1], first.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(first, second, taper)
        if not np.isfinite([dx, dy, response]).all():
            raise ValueError("Non-finite phase-correlation result")
        displacements.append((float(dx), float(dy), float(response)))
    shift = np.asarray(displacements, dtype=float)
    reliable = shift[:, 2] >= 0.08
    if reliable.sum() < max(3, math.ceil(0.6 * len(frames))):
        return {"status": "INSUFFICIENT_PHASE_CORRELATION",
                "reliable_frames": int(reliable.sum()),
                "total_frames": len(frames), "drift_rmse_px": None,
                "median_response": float(np.median(shift[:, 2]))}
    # Reference for a still camera: frame-specific displacement should be
    # constant. Subtract the median to ignore harmless static border shift.
    center = np.median(shift[reliable, :2], axis=0)
    deviations = shift[reliable, :2] - center
    return {
        "status": "MEASURABLE",
        "drift_rmse_px": float(np.sqrt(np.mean(np.sum(deviations ** 2, axis=1)))),
        "median_response": float(np.median(shift[:, 2])),
        "reliable_frames": int(reliable.sum()),
        "total_frames": len(frames),
        "method": "central-crop Hanning phase-correlation, per-frame vs known clean reference",
    }


def scratch_detection(*, source: Path, mask_dir: Path, reference_dir: Path,
                      ffmpeg: str, ffprobe: str, sensitivity: float = 0.65,
                      repair_baseline: Path | None = None) -> dict:
    """Evaluate CPU scratch segmentation on exact synthetic known masks."""
    from cleanup_video import ai_scratch_mask
    if not 0 <= sensitivity <= 1:
        raise ValueError("Scratch sensitivity outside 0..1")
    info = ev.probe(source, ffprobe)
    input_frames = list(ev.decode(source, ffmpeg, info))
    mask_dir.mkdir(parents=True, exist_ok=False)
    tp = fp = fn = tn = 0
    predicted_frames = []
    for index, rgb in enumerate(input_frames):
        gt_file = reference_dir / f"{index:06d}.png"
        truth = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
        if truth is None or truth.shape != rgb.shape[:2]:
            raise ValueError(f"Missing synthetic binary ground truth mask: {gt_file}")
        pred = ai_scratch_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                               sensitivity=sensitivity)
        if pred.shape != truth.shape:
            raise ValueError("Predicted scratch mask shape changed")
        pred_bin = pred > 127
        true_bin = truth > 127
        tp += int(np.sum(pred_bin & true_bin))
        fp += int(np.sum(pred_bin & ~true_bin))
        fn += int(np.sum(~pred_bin & true_bin))
        tn += int(np.sum(~pred_bin & ~true_bin))
        if not cv2.imwrite(str(mask_dir / f"{index:06d}.png"), pred):
            raise IOError("Could not preserve scratch-mask prediction")
        if repair_baseline is not None:
            repaired = cv2.inpaint(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   pred, 3, cv2.INPAINT_TELEA)
            predicted_frames.append(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    metrics = {
        "backend": "cleanup_video.ai_scratch_mask (actual ARP CPU detector)",
        "true_positive_pixels": tp, "false_positive_pixels": fp,
        "false_negative_pixels": fn, "true_negative_pixels": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.,
        "mask_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "sensitivity": sensitivity,
        "warning": "Pixel-level synthetic scratch masks, not annotated real-film DART/ProPainter segmentation",
    }
    if repair_baseline is not None:
        ev.encode(repair_baseline, ffmpeg, info, predicted_frames)
        sign(repair_baseline, ev.digest(source), method="opencv_telea_detector_mask_baseline",
             args={"sensitivity": sensitivity, "radius": 3})
        metrics["repair_baseline"] = (
            "OpenCV Telea with ARP detector mask; NOT ProPainter, DART or any AI inpainting"
        )
    return metrics


def run(*, manifest: Path, output_dir: Path, stage: str,
        ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
        denoise_methods: tuple[str, ...] = ("off", "hqdn3d", "nlmeans", "bm3d"),
        denoise_presets: tuple[str, ...] = ("gentle",),
        scratch_sensitivity: float = 0.65) -> dict:
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite previous stage evaluation")
    data, damaged = assert_experiment(manifest, stage, ffprobe)
    if stage == "denoise":
        import denoise_video
        if not denoise_methods or len(set(denoise_methods)) != len(denoise_methods) or any(
                x not in denoise_video.METHODS for x in denoise_methods):
            raise ValueError("Unsupported/duplicate denoise algorithm")
        if not denoise_presets or len(set(denoise_presets)) != len(denoise_presets) or any(
                x not in denoise_video.PRESETS for x in denoise_presets):
            raise ValueError("Unsupported/duplicate denoise preset")
    else:
        if stage == "scratch_masks" and not 0 <= scratch_sensitivity <= 1:
            raise ValueError("Invalid scratch detection sensitivity")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = {}
    extra = {}
    if stage == "deflicker":
        target = output_dir / "light_deflicker.mkv"
        extra["light_deflicker"] = deflicker(source=damaged, target=target,
                                            ffmpeg=ffmpeg, ffprobe=ffprobe)
        candidates["light_deflicker"] = target
    elif stage == "denoise":
        candidates, extra = denoise(source=damaged, directory=output_dir,
                                    ffmpeg=ffmpeg, methods=list(denoise_methods),
                                    presets=list(denoise_presets))
    elif stage == "stabilize":
        target = output_dir / "libvidstab_single_shot.mkv"
        extra["libvidstab_single_shot"] = stabilize(source=damaged, target=target,
                                                   ffmpeg=ffmpeg)
        candidates["libvidstab_single_shot"] = target
    elif stage == "scratch_masks":
        candidate = output_dir / "telea_detector_baseline.mkv"
        extra["scratch_mask_detection"] = scratch_detection(
            source=damaged, mask_dir=output_dir / "predicted_masks",
            reference_dir=manifest.parent / "masks",
            ffmpeg=ffmpeg, ffprobe=ffprobe, sensitivity=scratch_sensitivity,
            repair_baseline=candidate,
        )
        candidates["telea_detector_baseline"] = candidate
    if stage == "stabilize":
        info = {k: data[k] for k in ("width", "height", "fps", "frames")}
        clean = list(ev.decode(manifest.parent / "reference.mkv", ffmpeg, info))
        before = list(ev.decode(damaged, ffmpeg, info))
        after = list(ev.decode(candidates["libvidstab_single_shot"], ffmpeg, info))
        extra["stabilization_drift"] = {
            "damaged_baseline": residual_alignment(clean, before),
            "libvidstab_single_shot": residual_alignment(clean, after),
            "warning": "Image-based phase-correlation diagnostics require a textured stationary reference; low-confidence frames are excluded",
        }
    scores = ev.evaluate(manifest=manifest, candidates=candidates,
                         output_dir=output_dir / "scores",
                         ffmpeg=ffmpeg, ffprobe=ffprobe)
    report = {
        "schema_version": 1,
        "stage": stage, "film": data["film"], "scene": data["scene"],
        "synthetic_degradation": data["synthetic_degradation"],
        "input_sha256": data["degraded_sha256"],
        "methods": extra, "scores": scores["scores"],
        "important": "Scores measure ONLY controlled synthetic degradation of a digital transfer. No GPU, model-weights, DART or ProPainter inference.",
    }
    (output_dir / "stage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage", choices=tuple(TASKS), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", default="off,hqdn3d,nlmeans,bm3d",
                   help="Stage2 only: comma separated options")
    p.add_argument("--presets", default="gentle", help="Stage2 only")
    p.add_argument("--scratch-sensitivity", type=float, default=0.65)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args()
    print(json.dumps(run(
        manifest=args.manifest, output_dir=args.output_dir,
        stage=args.stage, ffmpeg=ev.find_binary(args.ffmpeg),
        ffprobe=ev.find_binary(args.ffprobe),
        denoise_methods=tuple(x.strip() for x in args.methods.split(",")),
        denoise_presets=tuple(x.strip() for x in args.presets.split(",")),
        scratch_sensitivity=args.scratch_sensitivity,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
