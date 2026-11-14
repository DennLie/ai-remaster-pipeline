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
})

MANUAL_PROVENANCE_TIERS = frozenset({
    "manual_polygon_gt",
    "manual_segmentation_gt",
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
    has_unverified = bool(set(counts) & {"restoration_diff_proxy", "model_pseudo_label"})
    if has_manual and has_unverified:
        raise ValueError(
            "Refusing to mix manual ground-truth annotations with model_pseudo_label or "
            f"restoration_diff_proxy in the same evaluation cohort: {counts}"
        )
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
