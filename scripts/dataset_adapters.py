"""Dataset adapters and nuisance-region false-positive analysis for film defect benchmarks.

Adapts published film-damage benchmarks into the canonical `benchmark_defect_masks.py`
manifest (`clip,frame,split,damage_type`) and 3-tier PNG mask format (`0=clean`,
`127=ignore`, `255=defect`) without duplicating metric formulas:

1. FilmDamageSimulator / FILM-AA (`daniela997/FilmDamageSimulator`, MIT / Figshare CC BY 4.0):
   Human-annotated polygon contours (`Dust`, `Dirt`, `Scratch`, `Long hair`, `Short hair`)
   on 35mm scans, including the upstream 1.5x coordinate scaling for `Scan (8..10)`.
2. ARTeFACT (`danielaivanova/damaged-media`, AFL-3.0):
   15-class pixel-level segmentation masks (`0=Clean`, `1..15=damage classes`,
   `255=Background -> 127 ignore`).
3. AbsoluteDegradation (`TytanMikJas/AbsoluteDegradation`, MIT code):
   Float32 parametric scratch soft masks (`opacity >= threshold -> 255`,
   `0 < opacity < threshold -> 127 ignore`, `0 -> 0`).
4. Provenance Separation & Pseudo-Label Guard:
   Strictly separates `manual_polygon_gt` and `manual_segmentation_gt` from
   `synthetic_parametric_gt`, `restoration_diff_proxy` (e.g. Figshare 21803304
   damaged/restored pairs), and `model_pseudo_label` (e.g. DART MaskNet outputs).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

import benchmark_defect_masks as bdm


PROVENANCE_TIERS = frozenset({
    "manual_polygon_gt",
    "manual_segmentation_gt",
    "synthetic_parametric_gt",
    "restoration_diff_proxy",
    "model_pseudo_label",
    "quarantined_temporal_diff_pseudo_label",
})

MANUAL_PROVENANCE_TIERS = frozenset({
    "manual_polygon_gt",
    "manual_segmentation_gt",
})

QUARANTINED_PROVENANCE_TIERS = frozenset({
    "restoration_diff_proxy",
    "model_pseudo_label",
    "quarantined_temporal_diff_pseudo_label",
})

DEV_DIAGNOSTIC_SPLITS = frozenset({
    "dev",
    "development",
    "diagnostic",
    "dev_diagnostic",
    "pilot",
})

LOCKED_HOLDOUT_SPLITS = frozenset({
    "locked_holdout",
    "final_holdout",
})

SPLIT_ROLES = DEV_DIAGNOSTIC_SPLITS | LOCKED_HOLDOUT_SPLITS | frozenset({
    "test",
    "eval",
    "unassigned",
})

CONTAMINATED_DEV_CLIPS = frozenset({
    "Neighbors_72",
    "neighbors_pilot",
    "dart_neighbors_72",
    "paleface_seq_a",
    "paleface_seq_b",
    "college_seq_c",
    "neighbors_seq_d",
    "neighbors_seq_d_vert",
    "neighbors_seq_d_horiz",
    "absdeg_000",
    "000",
    "absdeg_020",
    "020",
})

# FilmDamageSimulator (`daniela997/FilmDamageSimulator`) class mapping
FDS_CLASS_TO_CHANNEL: dict[str, str] = {
    "Dust": "dust",
    "Dirt": "dust",
    "Short hair": "dust",
    "Long hair": "scratch",
    "Scratch": "scratch",
}

# Scans annotated at 2/3 resolution in upstream `damage_generator/scans.py`
FDS_SCALED_SCAN_STEMS = frozenset({"Scan (8)", "Scan (9)", "Scan (10)"})
FDS_SCALE_FACTOR = 1.5

# ARTeFACT (`danielaivanova/damaged-media`) pixel IDs from upstream dataset card
ARTEFACT_ID_TO_CLASS: dict[int, str] = {
    0: "Clean",
    1: "Material loss",
    2: "Peel",
    3: "Dust",
    4: "Scratch",
    5: "Hair",
    6: "Dirt",
    7: "Fold",
    8: "Writing",
    9: "Cracks",
    10: "Staining",
    11: "Stamp",
    12: "Sticker",
    13: "Puncture",
    14: "Burn marks",
    15: "Lightleak",
    255: "Background",
}

ARTEFACT_DUST_IDS = frozenset({3, 5, 6})       # Dust, Hair, Dirt
ARTEFACT_SCRATCH_IDS = frozenset({4})          # Scratch
ARTEFACT_STRUCTURAL_IDS = frozenset({1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15})
ARTEFACT_BACKGROUND_ID = 255


def validate_provenance_separation(
    samples: Sequence[Mapping[str, Any]],
    *,
    require_manual_only: bool = False,
) -> dict[str, int]:
    """Verify every sample declares a valid provenance tier and prevent silent mixing.

    When `require_manual_only=True`, raises `ValueError` if any synthetic,
    restoration-difference proxy, or model pseudo-label sample is present.
    Also raises `ValueError` if a single manifest mixes manual ground truth with
    model pseudo-labels or restoration-difference proxies without explicit split isolation.
    """
    counts: dict[str, int] = {}
    for idx, row in enumerate(samples):
        prov = (row.get("label_provenance") or "").strip()
        if prov not in PROVENANCE_TIERS:
            raise ValueError(
                f"Sample {idx} has invalid or missing label_provenance {prov!r}; "
                f"expected one of {sorted(PROVENANCE_TIERS)}"
            )
        counts[prov] = counts.get(prov, 0) + 1

    if require_manual_only:
        non_manual = sorted(set(counts) - MANUAL_PROVENANCE_TIERS)
        if non_manual:
            raise ValueError(
                f"Manual-only ground truth evaluation rejected non-manual provenance tiers: {non_manual}"
            )

    has_manual = bool(set(counts) & MANUAL_PROVENANCE_TIERS)
    has_unverified = bool(set(counts) & QUARANTINED_PROVENANCE_TIERS)
    if has_manual and has_unverified:
        raise ValueError(
            "Refusing to mix manual ground-truth annotations with quarantined pseudo-labels or "
            f"restoration_diff_proxy in the same evaluation cohort: {counts}"
        )
    return counts


def validate_split_isolation(
    samples: Sequence[Mapping[str, Any]],
    *,
    allow_holdout_for_tuning: bool = False,
) -> dict[str, int]:
    """Validate split roles and reject contaminated development clips in locked holdouts."""
    counts: dict[str, int] = {}
    contaminated_lower = {c.lower() for c in CONTAMINATED_DEV_CLIPS}
    for idx, row in enumerate(samples):
        clip = str(row.get("clip") or "").strip()
        split = str(row.get("split") or "").strip()
        if split and split not in SPLIT_ROLES:
            raise ValueError(
                f"Sample {idx} (clip={clip!r}) has unrecognized split {split!r}; "
                f"expected one of {sorted(SPLIT_ROLES)}"
            )
        if clip.lower() in contaminated_lower and split in LOCKED_HOLDOUT_SPLITS:
            raise ValueError(
                f"Contaminated dev/diagnostic clip {clip!r} cannot be placed in locked holdout split {split!r}"
            )
        if not allow_holdout_for_tuning and split in LOCKED_HOLDOUT_SPLITS:
            raise ValueError(
                f"Locked holdout split {split!r} (clip={clip!r}) is disallowed during tuning/calibration"
            )
        key = split or "unspecified"
        counts[key] = counts.get(key, 0) + 1
    return counts


def parse_fds_polygon_json(
    json_path: Path,
    *,
    scan_stem: str | None = None,
) -> list[dict[str, Any]]:
    """Parse a `FilmDamageSimulator` polygon annotation JSON (`Scan (N).json`).

    Applies the upstream 1.5x coordinate scaling for `Scan (8)`, `Scan (9)`, and `Scan (10)`
    as specified in `daniela997/FilmDamageSimulator/blob/main/damage_generator/scans.py`.
    """
    stem = scan_stem if scan_stem is not None else json_path.stem
    scale = FDS_SCALE_FACTOR if stem in FDS_SCALED_SCAN_STEMS else 1.0
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    polygons: list[dict[str, Any]] = []
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "points" not in entry or "label" not in entry:
            continue
        label_name = str(entry["label"].get("name", "")).strip()
        pts_raw = entry["points"]
        if not pts_raw or not label_name:
            continue
        pts = np.array(
            [[round(float(p["x"]) * scale), round(float(p["y"]) * scale)] for p in pts_raw],
            dtype=np.int32,
        )
        channel = FDS_CLASS_TO_CHANNEL.get(label_name, "other")
        polygons.append(
            {
                "id": str(key),
                "label": label_name,
                "channel": channel,
                "scale_applied": scale,
                "points": pts,
            }
        )
    return polygons


def rasterize_fds_masks(
    polygons: Sequence[Mapping[str, Any]],
    height: int,
    width: int,
    *,
    boundary_ignore_px: int = 0,
) -> dict[str, np.ndarray]:
    """Rasterize FilmDamageSimulator polygons into 3-tier uint8 masks (`0`, `127`, `255`).

    Returns a dictionary of masks keyed by:
    - `'combined'`: all annotated defects = 255
    - `'scratch'`: `Scratch` and `Long hair` = 255; other defect classes = 127 (ignore)
    - `'dust'`: `Dust`, `Dirt`, `Short hair` = 255; other defect classes = 127 (ignore)
    - `'by_class'`: dict mapping each raw class name (`Dust`, `Dirt`, `Scratch`,
      `Long hair`, `Short hair`) to its class-specific 3-tier mask (target class = 255,
      other annotated defect classes = 127 ignore, clean background = 0).
    """
    if height < 1 or width < 1:
        raise ValueError(f"Invalid canvas shape ({height}, {width})")

    class_layers: dict[str, np.ndarray] = {
        cls_name: np.zeros((height, width), dtype=np.uint8)
        for cls_name in FDS_CLASS_TO_CHANNEL
    }
    for poly in polygons:
        label = str(poly["label"])
        pts = np.asarray(poly["points"], dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 1 or pts.shape[1] != 2:
            continue
        pts_clipped = pts.copy()
        pts_clipped[:, 0] = np.clip(pts_clipped[:, 0], 0, width - 1)
        pts_clipped[:, 1] = np.clip(pts_clipped[:, 1], 0, height - 1)
        layer = class_layers.setdefault(label, np.zeros((height, width), dtype=np.uint8))
        if pts_clipped.shape[0] == 1:
            x, y = int(pts_clipped[0, 0]), int(pts_clipped[0, 1])
            layer[y, x] = 255
        elif pts_clipped.shape[0] == 2:
            cv2.line(
                layer,
                tuple(pts_clipped[0]),
                tuple(pts_clipped[1]),
                color=255,
                thickness=1,
                lineType=cv2.LINE_8,
            )
        else:
            cv2.fillPoly(layer, [pts_clipped.reshape(-1, 1, 2)], color=255)

    any_defect = np.zeros((height, width), dtype=bool)
    for layer in class_layers.values():
        any_defect |= layer > 0

    def _build_channel_mask(target_labels: set[str]) -> np.ndarray:
        target = np.zeros((height, width), dtype=bool)
        for lbl in target_labels:
            if lbl in class_layers:
                target |= class_layers[lbl] > 0
        other_defect = any_defect & ~target
        mask = np.zeros((height, width), dtype=np.uint8)
        if boundary_ignore_px > 0 and np.any(target):
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * boundary_ignore_px + 1, 2 * boundary_ignore_px + 1),
            )
            dilated = cv2.dilate(target.astype(np.uint8) * 255, kernel) > 0
            mask[dilated & ~target] = 127
        mask[other_defect] = 127
        mask[target] = 255
        return mask

    combined_mask = _build_channel_mask(set(class_layers.keys()))
    scratch_mask = _build_channel_mask({k for k, v in FDS_CLASS_TO_CHANNEL.items() if v == "scratch"})
    dust_mask = _build_channel_mask({k for k, v in FDS_CLASS_TO_CHANNEL.items() if v == "dust"})
    by_class = {
        cls_name: _build_channel_mask({cls_name})
        for cls_name in class_layers
    }
    return {
        "combined": combined_mask,
        "scratch": scratch_mask,
        "dust": dust_mask,
        "by_class": by_class,
    }


def convert_artefact_segmentation_mask(
    annotation: np.ndarray,
    *,
    mode: str = "combined",
) -> np.ndarray:
    """Convert an ARTeFACT (`danielaivanova/damaged-media`) pixel-ID mask to 3-tier GT.

    In ARTeFACT:
    - `0` = Clean -> `0`
    - `255` = Background (outside media border) -> `127` (ignore)
    - `1..15` = Damage classes:
      - `mode='combined'`: `3, 4, 5, 6` (Dust, Scratch, Hair, Dirt) -> `255`,
        other structural media damage (`1, 2, 7..15`) -> `127` (ignore)
      - `mode='all_damage'`: all `1..15` -> `255`
      - `mode='scratch'`: `4` (Scratch) -> `255`; other damage `1..15` -> `127` (ignore)
      - `mode='dust'`: `3, 5, 6` (Dust, Hair, Dirt) -> `255`; other damage `1..15` -> `127` (ignore)
    """
    if annotation.ndim != 2 or annotation.dtype != np.uint8:
        raise ValueError("ARTeFACT annotation must be a 2D uint8 array")

    out = np.zeros_like(annotation, dtype=np.uint8)
    out[annotation == ARTEFACT_BACKGROUND_ID] = 127

    if mode == "all_damage":
        target_ids = ARTEFACT_DUST_IDS | ARTEFACT_SCRATCH_IDS | ARTEFACT_STRUCTURAL_IDS
        ignore_ids: frozenset[int] = frozenset()
    elif mode == "combined":
        target_ids = ARTEFACT_DUST_IDS | ARTEFACT_SCRATCH_IDS
        ignore_ids = ARTEFACT_STRUCTURAL_IDS
    elif mode == "scratch":
        target_ids = ARTEFACT_SCRATCH_IDS
        ignore_ids = ARTEFACT_DUST_IDS | ARTEFACT_STRUCTURAL_IDS
    elif mode == "dust":
        target_ids = ARTEFACT_DUST_IDS
        ignore_ids = ARTEFACT_SCRATCH_IDS | ARTEFACT_STRUCTURAL_IDS
    else:
        raise ValueError(f"Unsupported ARTeFACT conversion mode: {mode!r}")

    for cid in ignore_ids:
        out[annotation == cid] = 127
    for cid in target_ids:
        out[annotation == cid] = 255
    return out


def convert_soft_degradation_mask(
    soft_mask: np.ndarray,
    *,
    defect_threshold: float = 0.10,
) -> np.ndarray:
    """Convert an AbsoluteDegradation float32 soft mask in [0, 1] into 3-tier uint8 GT.

    Pixels with `0 < soft < defect_threshold` become `127` (ignore);
    pixels with `soft >= defect_threshold` become `255` (defect); `0` stays `0` (clean).
    """
    if not np.isfinite(soft_mask).all() or np.any((soft_mask < 0.0) | (soft_mask > 1.0)):
        raise ValueError("Soft mask must be finite and within [0.0, 1.0]")
    if not 0.0 < defect_threshold <= 1.0:
        raise ValueError("defect_threshold must be in (0.0, 1.0]")
    out = np.zeros(soft_mask.shape[:2], dtype=np.uint8)
    out[soft_mask > 0.0] = 127
    out[soft_mask >= defect_threshold] = 255
    return out


def derive_restoration_diff_proxy_mask(
    damaged_gray: np.ndarray,
    restored_gray: np.ndarray,
    *,
    low_diff: int = 6,
    high_diff: int = 18,
) -> dict[str, Any]:
    """Derive a strictly-tagged proxy mask from paired damaged/restored scans (e.g. Figshare 21803304).

    Never labelled as manual ground truth (`label_provenance='restoration_diff_proxy'`),
    because image differences between damaged and restored JPEG scans include JPEG compression
    and retouching transitions. Pixels in `[low_diff, high_diff)` are marked `127` (ignore).
    """
    if damaged_gray.shape != restored_gray.shape or damaged_gray.ndim != 2:
        raise ValueError("Expected matching 2D grayscale damaged and restored arrays")
    if not 0 < low_diff < high_diff <= 255:
        raise ValueError("Expected 0 < low_diff < high_diff <= 255")
    diff = cv2.absdiff(damaged_gray, restored_gray)
    mask = np.zeros_like(diff, dtype=np.uint8)
    mask[(diff >= low_diff) & (diff < high_diff)] = 127
    mask[diff >= high_diff] = 255
    return {
        "mask": mask,
        "label_provenance": "restoration_diff_proxy",
        "low_diff": low_diff,
        "high_diff": high_diff,
    }


def export_benchmark_dataset(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    require_manual_only: bool = False,
) -> dict[str, Any]:
    """Write adapted ground-truth masks and a `benchmark_defect_masks.py`-compatible manifest.

    Each item in `records` must provide:
    - `clip`: safe folder name
    - `frame`: safe PNG filename (e.g. `'000001.png'`)
    - `split`: split identifier (e.g. `'val'`, `'test'`)
    - `damage_type`: `'scratch'`, `'dust'`, or class name
    - `label_provenance`: one of `PROVENANCE_TIERS`
    - `gt_mask`: 2D uint8 array using only `GT_VALUES` (`0, 127, 255`)
    - optional `image`: 2D or 3D uint8 source frame to save under `frames/<clip>/<frame>`
    """
    provenance_counts = validate_provenance_separation(
        records, require_manual_only=require_manual_only
    )
    gt_root = output_dir / "ground_truth"
    frames_root = output_dir / "frames"
    manifest_path = output_dir / "manifest.csv"
    provenance_path = output_dir / "provenance.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = (*bdm.REQUIRED_COLUMNS, "label_provenance")
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            clip = bdm.validate_component(str(rec["clip"]).strip(), "clip")
            frame = bdm.validate_component(str(rec["frame"]).strip(), "frame")
            split = str(rec["split"]).strip()
            damage_type = str(rec["damage_type"]).strip()
            prov = str(rec["label_provenance"]).strip()
            gt_mask = np.asarray(rec["gt_mask"], dtype=np.uint8)
            unexpected = np.setdiff1d(np.unique(gt_mask), bdm.GT_VALUES)
            if unexpected.size:
                raise ValueError(f"Invalid GT mask values {unexpected.tolist()} for {clip}/{frame}")

            clip_gt_dir = gt_root / clip
            clip_gt_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clip_gt_dir / frame), gt_mask)

            if "gt_scratch_mask" in rec and rec["gt_scratch_mask"] is not None:
                gt_s_dir = output_dir / "ground_truth_scratch" / clip
                gt_s_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(gt_s_dir / frame), np.asarray(rec["gt_scratch_mask"], dtype=np.uint8))

            if "gt_dust_mask" in rec and rec["gt_dust_mask"] is not None:
                gt_d_dir = output_dir / "ground_truth_dust" / clip
                gt_d_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(gt_d_dir / frame), np.asarray(rec["gt_dust_mask"], dtype=np.uint8))

            if "image" in rec and rec["image"] is not None:
                clip_frame_dir = frames_root / clip
                clip_frame_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(clip_frame_dir / frame), np.asarray(rec["image"], dtype=np.uint8))

            writer.writerow(
                {
                    "clip": clip,
                    "frame": frame,
                    "split": split,
                    "damage_type": damage_type,
                    "label_provenance": prov,
                }
            )

    prov_meta = {
        "samples": len(records),
        "provenance_counts": provenance_counts,
        "manual_only_verified": all(k in MANUAL_PROVENANCE_TIERS for k in provenance_counts),
    }
    provenance_path.write_text(json.dumps(prov_meta, indent=2) + "\n", encoding="utf-8")
    return {
        "manifest": manifest_path,
        "ground_truth": gt_root,
        "frames": frames_root,
        "provenance": provenance_path,
        "provenance_summary": prov_meta,
    }


def segment_nuisance_regions(
    frame: np.ndarray,
    *,
    prev_frame: np.ndarray | None = None,
    edge_percentile: float = 85.0,
    grain_std_percentile: float = 50.0,
    motion_diff_threshold: float = 12.0,
) -> dict[str, np.ndarray]:
    """Partition a frame into diagnostic nuisance masks (boolean 2D arrays) for FP analysis.

    Regions:
    - `'edges'`: strong structural gradients (Sobel magnitude >= `edge_percentile`).
    - `'grain'`: non-edge textured film-grain regions (local 5x5 standard deviation >=
      `grain_std_percentile` and not in `'edges'`).
    - `'faces_foreground'`: central portrait/subject zone (inner 50% ellipse where faces and
      primary foreground subjects reside in archival framing).
    - `'motion'`: pixels with inter-frame difference `|I_t - I_{t-1}| >= motion_diff_threshold`
      when `prev_frame` is supplied (empty mask for single still images).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected 2D uint8 grayscale or 3D BGR uint8 frame")
    h, w = gray.shape
    gf = gray.astype(np.float32)

    gx = cv2.Sobel(gf, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gf, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    edge_thresh = max(15.0, float(np.percentile(grad_mag, edge_percentile)))
    edges = grad_mag >= edge_thresh

    local_mean = cv2.blur(gf, (5, 5))
    local_sq_mean = cv2.blur(gf * gf, (5, 5))
    local_std = np.sqrt(np.maximum(0.0, local_sq_mean - local_mean * local_mean))
    grain_thresh = max(1.5, float(np.percentile(local_std, grain_std_percentile)))
    grain = (local_std >= grain_thresh) & ~edges

    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    ry, rx = max(1.0, h * 0.28), max(1.0, w * 0.22)
    faces_foreground = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if prev_frame.ndim == 3 else prev_frame
        if prev_gray.shape != gray.shape:
            raise ValueError("prev_frame shape must match frame shape")
        motion = cv2.absdiff(gray, prev_gray) >= motion_diff_threshold
    else:
        motion = np.zeros((h, w), dtype=bool)

    return {
        "edges": edges,
        "grain": grain,
        "faces_foreground": faces_foreground,
        "motion": motion,
    }


def evaluate_nuisance_fp_breakdown(
    frame: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    prev_frame: np.ndarray | None = None,
    threshold: int = 128,
) -> dict[str, dict[str, float | int]]:
    """Evaluate detector false-positive rates on unaffected pixels stratified by nuisance regime.

    Delegates all confusion and metric math to `benchmark_defect_masks.confusion` and
    `benchmark_defect_masks.metrics` (marking out-of-region pixels as `127 = ignore`)
    so metric definitions are never duplicated.
    """
    regions = segment_nuisance_regions(frame, prev_frame=prev_frame)
    overall_counts = bdm.confusion(gt_mask, pred_mask, threshold=threshold)
    out: dict[str, dict[str, float | int]] = {
        "overall": bdm.metrics(overall_counts),
    }
    for region_name, region_bool in regions.items():
        masked_gt = np.where(region_bool, gt_mask, np.uint8(127))
        counts = bdm.confusion(masked_gt, pred_mask, threshold=threshold)
        out[region_name] = bdm.metrics(counts)
    return out


def evaluate_false_positive_severity(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    threshold: int = 128,
) -> dict[str, Any]:
    """Quantify connected-component severity of false positives on intact pixels.

    Prevents isolated severe structural errors from being hidden by whole-frame averaging.
    """
    if gt_mask.shape != pred_mask.shape or gt_mask.ndim != 2 or pred_mask.ndim != 2:
        raise ValueError(f"Expected matching 2D masks, got gt={gt_mask.shape}, pred={pred_mask.shape}")
    if gt_mask.dtype != np.uint8 or pred_mask.dtype != np.uint8:
        raise ValueError("Expected uint8 masks")

    valid = gt_mask != 127
    fp_map = ((pred_mask >= threshold) & (gt_mask == 0) & valid).astype(np.uint8) * 255
    fp_pixel_count = int(np.count_nonzero(fp_map))
    if fp_pixel_count == 0:
        return {
            "fp_pixel_count": 0,
            "fp_component_count": 0,
            "max_fp_component_area": 0,
            "max_fp_component_height": 0,
            "max_fp_component_width": 0,
            "severe_structural_fp_count": 0,
        }

    n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(fp_map, connectivity=8)
    max_area = 0
    max_h = 0
    max_w = 0
    severe_count = 0
    for i in range(1, n_labels):
        cw = int(stats[i, cv2.CC_STAT_WIDTH])
        ch = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        max_area = max(max_area, area)
        max_h = max(max_h, ch)
        max_w = max(max_w, cw)
        if ch >= 15 or area >= 25:
            severe_count += 1

    return {
        "fp_pixel_count": fp_pixel_count,
        "fp_component_count": max(0, n_labels - 1),
        "max_fp_component_area": max_area,
        "max_fp_component_height": max_h,
        "max_fp_component_width": max_w,
        "severe_structural_fp_count": severe_count,
    }


def evaluate_structural_negative_controls(
    pred_mask: np.ndarray,
    repair_mask: np.ndarray,
    negative_control_rois: Sequence[Mapping[str, Any]],
    threshold: int = 128,
) -> dict[str, Any]:
    """Evaluate detector and conservative repair-mask leakage on critical negative-control ROIs.

    Requires strict zero repair-mask leakage (`repair_leak_pixels == 0`) across all
    designated critical structural ROIs.
    """
    if pred_mask.shape != repair_mask.shape or pred_mask.ndim != 2 or repair_mask.ndim != 2:
        raise ValueError(f"Expected matching 2D masks, got pred={pred_mask.shape}, repair={repair_mask.shape}")
    if pred_mask.dtype != np.uint8 or repair_mask.dtype != np.uint8:
        raise ValueError("Expected uint8 masks")

    h, w = pred_mask.shape
    pred_bin = pred_mask >= threshold
    repair_bin = repair_mask >= threshold
    rois_out: list[dict[str, Any]] = []
    total_det_leak = 0
    total_rep_leak = 0

    for roi in negative_control_rois:
        ymin = max(0, int(roi["ymin"]))
        ymax = min(h, int(roi["ymax"]))
        xmin = max(0, int(roi["xmin"]))
        xmax = min(w, int(roi["xmax"]))
        roi_area = max(0, (ymax - ymin) * (xmax - xmin))
        pred_slice = pred_bin[ymin:ymax, xmin:xmax]
        rep_slice = repair_bin[ymin:ymax, xmin:xmax]

        det_leak = int(np.count_nonzero(pred_slice))
        rep_leak = int(np.count_nonzero(rep_slice))
        max_comp_h = 0
        if rep_leak > 0:
            n_lbl, _lbl, stats, _ = cv2.connectedComponentsWithStats(rep_slice.astype(np.uint8) * 255, connectivity=8)
            if n_lbl > 1:
                max_comp_h = int(np.max(stats[1:, cv2.CC_STAT_HEIGHT]))

        total_det_leak += det_leak
        total_rep_leak += rep_leak
        rois_out.append({
            "name": str(roi.get("name", "negative_control")),
            "description": str(roi.get("description", "")),
            "bbox": {"ymin": ymin, "ymax": ymax, "xmin": xmin, "xmax": xmax},
            "roi_pixels": roi_area,
            "detector_leak_pixels": det_leak,
            "repair_leak_pixels": rep_leak,
            "detector_leak_fpr": round(det_leak / max(1, roi_area), 6),
            "repair_leak_fpr": round(rep_leak / max(1, roi_area), 6),
            "max_leak_component_height": max_comp_h,
            "strict_zero_leak_passed": bool(rep_leak == 0),
        })

    return {
        "rois": rois_out,
        "total_detector_leak_pixels": total_det_leak,
        "total_repair_leak_pixels": total_rep_leak,
        "strict_safety_gate_passed": all(r["strict_zero_leak_passed"] for r in rois_out),
    }


def evaluate_exact_reference_restoration(
    clean_ref_bgr: np.ndarray,
    damaged_bgr: np.ndarray,
    repaired_bgr: np.ndarray,
    gt_mask: np.ndarray,
    repair_mask: np.ndarray,
) -> dict[str, Any]:
    """Measure reconstruction quality and intact-region collateral safety against exact clean reference."""
    if clean_ref_bgr.shape != damaged_bgr.shape or clean_ref_bgr.shape != repaired_bgr.shape:
        raise ValueError("Clean, damaged, and repaired frames must have identical shapes")
    if gt_mask.shape != clean_ref_bgr.shape[:2] or repair_mask.shape != clean_ref_bgr.shape[:2]:
        raise ValueError("Masks must match frame spatial dimensions")

    clean_f = clean_ref_bgr.astype(np.float32)
    damaged_f = damaged_bgr.astype(np.float32)
    repaired_f = repaired_bgr.astype(np.float32)

    defect_idx = gt_mask == 255
    intact_idx = gt_mask == 0
    valid_idx = gt_mask != 127

    has_defect = bool(np.any(defect_idx))
    if has_defect:
        dmg_mse = float(np.mean((damaged_f[defect_idx] - clean_f[defect_idx]) ** 2))
        rep_mse = float(np.mean((repaired_f[defect_idx] - clean_f[defect_idx]) ** 2))
        dmg_psnr = 10.0 * math.log10((255.0 ** 2) / max(dmg_mse, 1e-6))
        rep_psnr = 10.0 * math.log10((255.0 ** 2) / max(rep_mse, 1e-6))
        psnr_gain = rep_psnr - dmg_psnr
    else:
        dmg_mse = 0.0
        rep_mse = 0.0
        dmg_psnr = 0.0
        rep_psnr = 0.0
        psnr_gain = 0.0

    intact_total = int(np.count_nonzero(intact_idx))
    if intact_total > 0:
        if clean_ref_bgr.ndim == 3:
            modified_map = np.any(repaired_bgr != clean_ref_bgr, axis=2) & intact_idx
        else:
            modified_map = (repaired_bgr != clean_ref_bgr) & intact_idx
        intact_modified_px = int(np.count_nonzero(modified_map))
        intact_diff = np.abs(repaired_f[intact_idx] - clean_f[intact_idx])
        intact_max_diff = float(np.max(intact_diff))
        intact_mean_diff = float(np.mean(intact_diff))
        intact_mse = float(np.mean((repaired_f[intact_idx] - clean_f[intact_idx]) ** 2))
        intact_psnr = 10.0 * math.log10((255.0 ** 2) / intact_mse) if intact_mse > 0.0 else 100.0
    else:
        intact_modified_px = 0
        intact_max_diff = 0.0
        intact_mean_diff = 0.0
        intact_mse = 0.0
        intact_psnr = 100.0

    rep_bin = repair_mask >= 128
    rep_valid = rep_bin & valid_idx
    rep_tp = int(np.count_nonzero(rep_valid & defect_idx))
    rep_fp = int(np.count_nonzero(rep_valid & intact_idx))
    rep_prec = (rep_tp / (rep_tp + rep_fp)) if (rep_tp + rep_fp) > 0 else 0.0

    return {
        "has_defect_pixels": has_defect,
        "defect_pixels": int(np.count_nonzero(defect_idx)),
        "damaged_mse": round(dmg_mse, 4),
        "repaired_mse": round(rep_mse, 4),
        "damaged_psnr_db": round(dmg_psnr, 3),
        "repaired_psnr_db": round(rep_psnr, 3),
        "psnr_gain_db": round(psnr_gain, 3),
        "repair_mask_pixels": int(np.count_nonzero(rep_bin)),
        "repair_mask_tp": rep_tp,
        "repair_mask_fp": rep_fp,
        "repair_mask_precision": round(rep_prec, 6),
        "intact_total_pixels": intact_total,
        "intact_pixels_modified_count": intact_modified_px,
        "intact_pixels_modified_fpr": round(intact_modified_px / max(1, intact_total), 8),
        "intact_max_abs_diff": round(intact_max_diff, 4),
        "intact_mean_abs_diff": round(intact_mean_diff, 6),
        "intact_mse": round(intact_mse, 6),
        "intact_psnr_db": round(intact_psnr, 3),
    }
