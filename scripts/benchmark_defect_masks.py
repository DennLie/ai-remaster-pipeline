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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


REQUIRED_COLUMNS = ("clip", "frame", "split", "damage_type")
GT_VALUES = (0, 127, 255)
EMPTY_COUNTS = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "ignored": 0}


@dataclass(frozen=True)
class ScratchConfig:
    """Configuration for parametric synthetic film scratch injection."""

    width: int = 2
    opacity: float = 0.8
    angle_deg: float = 0.0
    curvature: float = 0.0
    duration: int = 5
    drift_px: float = 1.0
    bright: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.width <= 8:
            raise ValueError(f"Scratch width must be in [1, 8], got {self.width}")
        if not 0.2 <= self.opacity <= 1.0:
            raise ValueError(f"Scratch opacity must be in [0.2, 1.0], got {self.opacity}")
        if not -15.0 <= self.angle_deg <= 15.0:
            raise ValueError(f"Scratch angle_deg must be in [-15.0, 15.0], got {self.angle_deg}")
        if not 1 <= self.duration <= 30:
            raise ValueError(f"Scratch duration must be in [1, 30], got {self.duration}")
        if not 0.0 <= self.drift_px <= 5.0:
            raise ValueError(f"Scratch drift_px must be in [0.0, 5.0], got {self.drift_px}")


@dataclass(frozen=True)
class DustConfig:
    """Configuration for synthetic transient dust and chemical blotch injection."""

    clusters_per_frame: int = 3
    particles_per_cluster: tuple[int, int] = (2, 6)
    radius_range: tuple[int, int] = (1, 5)
    cluster_spread_px: float = 8.0
    contrast: float = 0.85
    bright_probability: float = 0.3
    affected_frame_ratio: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.clusters_per_frame < 1:
            raise ValueError(f"clusters_per_frame must be >= 1, got {self.clusters_per_frame}")
        min_p, max_p = self.particles_per_cluster
        if min_p < 1 or max_p < min_p:
            raise ValueError(f"Invalid particles_per_cluster range: {self.particles_per_cluster}")
        min_r, max_r = self.radius_range
        if min_r < 1 or max_r < min_r:
            raise ValueError(f"Invalid radius_range: {self.radius_range}")
        if self.cluster_spread_px <= 0:
            raise ValueError(f"cluster_spread_px must be > 0, got {self.cluster_spread_px}")
        if not 0.2 <= self.contrast <= 1.0:
            raise ValueError(f"Dust contrast must be in [0.2, 1.0], got {self.contrast}")
        if not 0.0 <= self.bright_probability <= 1.0:
            raise ValueError(f"bright_probability must be in [0.0, 1.0], got {self.bright_probability}")
        if not 0.0 < self.affected_frame_ratio <= 1.0:
            raise ValueError(f"affected_frame_ratio must be in (0.0, 1.0], got {self.affected_frame_ratio}")


def _normalize_frame_sequence(frames: np.ndarray | Sequence[np.ndarray]) -> list[np.ndarray]:
    if isinstance(frames, np.ndarray):
        if frames.ndim not in (3, 4):
            raise ValueError(f"Expected 3D (T,H,W) or 4D (T,H,W,C) array, got shape {frames.shape}")
        frame_list = [frames[i].copy() for i in range(frames.shape[0])]
    else:
        frame_list = [ np.asarray(f).copy() for f in frames ]
    if not frame_list:
        raise ValueError("Frame sequence must not be empty")
    first_shape = frame_list[0].shape
    for idx, frame in enumerate(frame_list):
        if frame.dtype != np.uint8:
            raise ValueError(f"Frame {idx} must have uint8 dtype, got {frame.dtype}")
        if frame.ndim not in (2, 3) or frame.shape != first_shape:
            raise ValueError(f"Frame {idx} shape {frame.shape} does not match first frame {first_shape}")
    return frame_list


def _apply_mask_perturbation(
    frame: np.ndarray,
    defect_mask: np.ndarray,
    *,
    strength: float,
    bright: bool,
) -> np.ndarray:
    out = frame.copy()
    active = defect_mask == 255
    if not np.any(active):
        return out
    target = 255.0 if bright else 0.0
    blended = (1.0 - strength) * out.astype(np.float32) + strength * target
    out[active] = np.clip(np.round(blended[active]), 0, 255).astype(np.uint8)
    return out


def inject_scratches(
    frames: np.ndarray | Sequence[np.ndarray],
    config: ScratchConfig | None = None,
    *,
    num_scratches: int = 1,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Inject synthetic vertical/diagonal scratches into a frame sequence.

    Returns (perturbed_frames, ground_truth_masks) where each mask is a 2D uint8
    array with 255 at scratch pixels and 0 elsewhere.
    """
    cfg = config or ScratchConfig()
    if num_scratches < 1:
        raise ValueError(f"num_scratches must be >= 1, got {num_scratches}")
    perturbed = _normalize_frame_sequence(frames)
    num_frames = len(perturbed)
    height, width = perturbed[0].shape[:2]
    gt_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]
    rng = np.random.default_rng(cfg.seed)

    span = min(cfg.duration, num_frames)
    max_start = max(0, num_frames - span)
    slope = math.tan(math.radians(cfg.angle_deg))
    margin = max(cfg.width, min(width // 8, 8))
    low_x = margin if width > 2 * margin else 0
    high_x = (width - margin) if width > 2 * margin else width

    ys = np.linspace(0, height - 1, num=max(height, 2))
    norm_y = (ys - (height - 1) / 2.0) / max((height - 1) / 2.0, 1.0)

    for _ in range(num_scratches):
        t0 = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
        x0 = float(rng.uniform(low_x, max(low_x + 1, high_x)))
        drift_sign = 1.0 if num_scratches == 1 else float(rng.choice([-1.0, 1.0]))

        for dt in range(span):
            t = t0 + dt
            center_x = x0 + drift_sign * cfg.drift_px * dt
            xs = center_x + (ys - (height - 1) / 2.0) * slope + cfg.curvature * (norm_y**2 - 0.5)
            pts = np.stack([np.round(xs).astype(np.int32), np.round(ys).astype(np.int32)], axis=1)
            pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)

            scratch_layer = np.zeros((height, width), dtype=np.uint8)
            cv2.polylines(
                scratch_layer,
                [pts.reshape(-1, 1, 2)],
                isClosed=False,
                color=255,
                thickness=cfg.width,
                lineType=cv2.LINE_8,
            )
            gt_masks[t] = np.where(scratch_layer > 0, np.uint8(255), gt_masks[t])
            perturbed[t] = _apply_mask_perturbation(
                perturbed[t],
                scratch_layer,
                strength=cfg.opacity,
                bright=cfg.bright,
            )

    return perturbed, gt_masks


def inject_dust(
    frames: np.ndarray | Sequence[np.ndarray],
    config: DustConfig | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Inject synthetic transient single-frame dust and blotch clusters.

    Returns (perturbed_frames, ground_truth_masks) where each mask is a 2D uint8
    array with 255 at dust/blotch pixels and 0 elsewhere.
    """
    cfg = config or DustConfig()
    perturbed = _normalize_frame_sequence(frames)
    num_frames = len(perturbed)
    height, width = perturbed[0].shape[:2]
    gt_masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]
    rng = np.random.default_rng(cfg.seed)

    num_affected = max(1, int(round(num_frames * cfg.affected_frame_ratio)))
    affected_indices = set(
        rng.choice(num_frames, size=min(num_frames, num_affected), replace=False).tolist()
    )

    for t in sorted(affected_indices):
        frame_mask = np.zeros((height, width), dtype=np.uint8)
        for _ in range(cfg.clusters_per_frame):
            cx = float(rng.uniform(0, width))
            cy = float(rng.uniform(0, height))
            n_particles = int(
                rng.integers(cfg.particles_per_cluster[0], cfg.particles_per_cluster[1] + 1)
            )
            cluster_bright = bool(rng.random() < cfg.bright_probability)
            cluster_layer = np.zeros((height, width), dtype=np.uint8)

            for _ in range(n_particles):
                px = int(np.clip(round(rng.normal(cx, cfg.cluster_spread_px)), 0, width - 1))
                py = int(np.clip(round(rng.normal(cy, cfg.cluster_spread_px)), 0, height - 1))
                rx = int(rng.integers(cfg.radius_range[0], cfg.radius_range[1] + 1))
                ry = int(rng.integers(cfg.radius_range[0], cfg.radius_range[1] + 1))
                angle = float(rng.uniform(0.0, 180.0))
                cv2.ellipse(
                    cluster_layer,
                    center=(px, py),
                    axes=(rx, ry),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=255,
                    thickness=-1,
                    lineType=cv2.LINE_8,
                )

            frame_mask = np.where(cluster_layer > 0, np.uint8(255), frame_mask)
            perturbed[t] = _apply_mask_perturbation(
                perturbed[t],
                cluster_layer,
                strength=cfg.contrast,
                bright=cluster_bright,
            )

        gt_masks[t] = frame_mask

    return perturbed, gt_masks


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


def _is_scratch_damage(damage_type: str) -> bool:
    token = damage_type.strip().lower()
    return token in {"scratch", "scratches"} or token.startswith("scratch")


def _is_dust_damage(damage_type: str) -> bool:
    token = damage_type.strip().lower()
    return token in {"dust", "blotch", "blotches", "dust_blotch"} or token.startswith(("dust", "blotch"))


def evaluate_category_masks(
    scratch_gt: Sequence[np.ndarray] | np.ndarray,
    scratch_pred: Sequence[np.ndarray] | np.ndarray,
    dust_gt: Sequence[np.ndarray] | np.ndarray,
    dust_pred: Sequence[np.ndarray] | np.ndarray,
    *,
    threshold: int = 128,
) -> dict[str, dict[str, float | int]]:
    """Compute disjoint spatial metrics for scratch vs. dust mask sequences."""
    s_gt = _normalize_frame_sequence(scratch_gt)
    s_pr = _normalize_frame_sequence(scratch_pred)
    d_gt = _normalize_frame_sequence(dust_gt)
    d_pr = _normalize_frame_sequence(dust_pred)
    if len(s_gt) != len(s_pr):
        raise ValueError(f"Scratch GT/pred sequence lengths differ: {len(s_gt)} vs {len(s_pr)}")
    if len(d_gt) != len(d_pr):
        raise ValueError(f"Dust GT/pred sequence lengths differ: {len(d_gt)} vs {len(d_pr)}")

    scratch_counts = [confusion(gt, pr, threshold) for gt, pr in zip(s_gt, s_pr)]
    dust_counts = [confusion(gt, pr, threshold) for gt, pr in zip(d_gt, d_pr)]
    scratch_summary = metrics(sum_counts(scratch_counts))
    dust_summary = metrics(sum_counts(dust_counts))
    combined_summary = metrics(sum_counts(scratch_counts + dust_counts))
    return {
        "scratch_metrics": scratch_summary,
        "dust_metrics": dust_summary,
        "summary": combined_summary,
    }


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
        scratch_rows = [row for row in rows if _is_scratch_damage(row["damage_type"])]
        dust_rows = [row for row in rows if _is_dust_damage(row["damage_type"])]
        scratch_metrics = {
            "samples": len(scratch_rows),
            **metrics(sum_counts(scratch_rows) if scratch_rows else dict(EMPTY_COUNTS)),
        }
        dust_metrics = {
            "samples": len(dust_rows),
            **metrics(sum_counts(dust_rows) if dust_rows else dict(EMPTY_COUNTS)),
        }
        grouped: dict[str, dict[str, dict[str, Any]]] = {}
        for field in ("clip", "split", "damage_type"):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                groups[row[field]].append(row)
            grouped[field] = {
                key: {"samples": len(items), **metrics(sum_counts(items))}
                for key, items in sorted(groups.items())
            }
        result["methods"][method] = {
            "summary": summary,
            "scratch_metrics": scratch_metrics,
            "dust_metrics": dust_metrics,
            "by": grouped,
            "frames": rows,
        }
    return result


def generate_synthetic_dataset(
    output_dir: Path,
    *,
    num_frames: int = 10,
    height: int = 64,
    width: int = 64,
    scratch_config: ScratchConfig | None = None,
    dust_config: DustConfig | None = None,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate a synthetic benchmark dataset with disjoint scratch and dust clips."""
    if num_frames < 1 or height < 8 or width < 8:
        raise ValueError("num_frames >= 1 and height, width >= 8 are required")
    rng = np.random.default_rng(seed)
    base_frames = [
        rng.integers(96, 160, size=(height, width), dtype=np.uint8)
        for _ in range(num_frames)
    ]
    s_cfg = scratch_config or ScratchConfig(seed=seed)
    d_cfg = dust_config or DustConfig(seed=seed + 1)

    scratch_frames, scratch_masks = inject_scratches(base_frames, s_cfg)
    dust_frames, dust_masks = inject_dust(base_frames, d_cfg)

    gt_root = output_dir / "ground_truth"
    frames_root = output_dir / "frames"
    oracle_root = output_dir / "oracle_prediction"
    for root in (gt_root, frames_root, oracle_root):
        (root / "scratch_clip").mkdir(parents=True, exist_ok=True)
        (root / "dust_clip").mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for idx in range(num_frames):
            fname = f"{idx + 1:06d}.png"
            cv2.imwrite(str(frames_root / "scratch_clip" / fname), scratch_frames[idx])
            cv2.imwrite(str(gt_root / "scratch_clip" / fname), scratch_masks[idx])
            cv2.imwrite(str(oracle_root / "scratch_clip" / fname), scratch_masks[idx])
            writer.writerow(
                {
                    "clip": "scratch_clip",
                    "frame": fname,
                    "split": "synthetic",
                    "damage_type": "scratch",
                }
            )
        for idx in range(num_frames):
            fname = f"{idx + 1:06d}.png"
            cv2.imwrite(str(frames_root / "dust_clip" / fname), dust_frames[idx])
            cv2.imwrite(str(gt_root / "dust_clip" / fname), dust_masks[idx])
            cv2.imwrite(str(oracle_root / "dust_clip" / fname), dust_masks[idx])
            writer.writerow(
                {
                    "clip": "dust_clip",
                    "frame": fname,
                    "split": "synthetic",
                    "damage_type": "dust",
                }
            )

    return {
        "manifest": manifest_path,
        "ground_truth": gt_root,
        "frames": frames_root,
        "oracle_prediction": oracle_root,
    }


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
    parser.add_argument("--manifest", type=Path, help="CSV clip,frame,split,damage_type")
    parser.add_argument("--ground-truth", type=Path, help="GT PNG root, organized clip/frame.png")
    parser.add_argument("--prediction", action="append", metavar="NAME=FOLDER",
                        help="Prediction root arranged clip/frame.png; repeat for each method")
    parser.add_argument("--threshold", type=int, default=128, help="Soft mask threshold 1..255 (default 128)")
    parser.add_argument("--report", type=Path, required=True, help="JSON report path; no report if validation fails")
    parser.add_argument("--synthetic-dir", type=Path,
                        help="Generate synthetic scratch and dust sequences into this folder before evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic defect generation")
    parser.add_argument("--num-frames", type=int, default=10, help="Frames per synthetic clip (default 10)")
    parser.add_argument("--frame-height", type=int, default=64, help="Synthetic frame height (default 64)")
    parser.add_argument("--frame-width", type=int, default=64, help="Synthetic frame width (default 64)")
    parser.add_argument("--scratch-width", type=int, default=2, help="Synthetic scratch width 1..8 px")
    parser.add_argument("--scratch-opacity", type=float, default=0.8, help="Synthetic scratch opacity 0.2..1.0")
    parser.add_argument("--scratch-angle", type=float, default=0.0, help="Synthetic scratch angle -15..15 deg")
    parser.add_argument("--scratch-duration", type=int, default=5, help="Synthetic scratch duration 1..30 frames")
    parser.add_argument("--scratch-drift", type=float, default=1.0, help="Synthetic scratch drift 0..5 px/frame")
    parser.add_argument("--dust-clusters", type=int, default=3, help="Synthetic dust clusters per frame")
    parser.add_argument("--dust-contrast", type=float, default=0.85, help="Synthetic dust contrast 0.2..1.0")
    args = parser.parse_args(argv)
    if not 1 <= args.threshold <= 255:
        parser.error("--threshold must be between 1 and 255")

    if args.synthetic_dir is not None:
        try:
            s_cfg = ScratchConfig(
                width=args.scratch_width,
                opacity=args.scratch_opacity,
                angle_deg=args.scratch_angle,
                duration=args.scratch_duration,
                drift_px=args.scratch_drift,
                seed=args.seed,
            )
            d_cfg = DustConfig(
                clusters_per_frame=args.dust_clusters,
                contrast=args.dust_contrast,
                seed=args.seed + 1,
            )
        except ValueError as exc:
            parser.error(str(exc))
        synth_paths = generate_synthetic_dataset(
            args.synthetic_dir,
            num_frames=args.num_frames,
            height=args.frame_height,
            width=args.frame_width,
            scratch_config=s_cfg,
            dust_config=d_cfg,
            seed=args.seed,
        )
        manifest_path = args.manifest or synth_paths["manifest"]
        gt_path = args.ground_truth or synth_paths["ground_truth"]
        methods = (
            parse_methods(args.prediction)
            if args.prediction
            else {"oracle": synth_paths["oracle_prediction"]}
        )
    else:
        if args.manifest is None or args.ground_truth is None or not args.prediction:
            parser.error("--manifest, --ground-truth, and --prediction are required unless --synthetic-dir is set")
        manifest_path = args.manifest
        gt_path = args.ground_truth
        methods = parse_methods(args.prediction)

    data = evaluate(load_samples(manifest_path), gt_path, methods, threshold=args.threshold)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, value in data["methods"].items():
        summary = value["summary"]
        scratch_m = value["scratch_metrics"]
        dust_m = value["dust_metrics"]
        print(f"{name}: IoU={summary['iou']:.3f} F1={summary['f1']:.3f} "
              f"precision={summary['precision']:.3f} recall={summary['recall']:.3f} "
              f"intact-pixel FPR={summary['false_positive_rate']:.6f}")
        print(f"  scratch_metrics: IoU={scratch_m['iou']:.3f} F1={scratch_m['f1']:.3f} "
              f"P={scratch_m['precision']:.3f} R={scratch_m['recall']:.3f} "
              f"FPR={scratch_m['false_positive_rate']:.6f}")
        print(f"  dust_metrics:    IoU={dust_m['iou']:.3f} F1={dust_m['f1']:.3f} "
              f"P={dust_m['precision']:.3f} R={dust_m['recall']:.3f} "
              f"FPR={dust_m['false_positive_rate']:.6f}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
