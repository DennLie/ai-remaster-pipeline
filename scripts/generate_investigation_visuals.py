"""Generate high-resolution diagnostic crop comparisons for DART defect and class audit.

Generates 7-panel native crop comparisons:
1. Source Frame Crop
2. Ground Truth Mask
3. Raw DART Soft Mask (0..255)
4. Thresholded Binary Mask (threshold 160)
5. Predicted Scratch Channel
6. Predicted Dust Channel
7. Error Map Overlay (Green=TP, Red=FP, Blue=FN)
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
import cv2
import numpy as np

import candidate_detector_eval as cde
import run_defect_benchmark_suite as rdbs


def build_error_map(gt: np.ndarray, pred: np.ndarray, ignore_val: int = 127) -> np.ndarray:
    """Build BGR color-coded error map: TP=Green, FP=Red, FN=Blue, Ignore=Gray, TN=Dark Gray."""
    h, w = gt.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[:, :] = (30, 30, 30)
    
    valid = gt != ignore_val
    gt_pos = (gt == 255) & valid
    pred_pos = (pred > 0) & valid
    
    # TP: bright green
    out[gt_pos & pred_pos] = (0, 230, 0)
    # FP: bright red
    out[~gt_pos & pred_pos] = (0, 0, 230)
    # FN: electric blue
    out[gt_pos & ~pred_pos] = (230, 120, 0)
    # Ignore: dim gray
    if ignore_val is not None:
        out[gt == ignore_val] = (70, 70, 70)
    return out


def make_labeled_panel(img: np.ndarray, title: str, banner_h: int = 28) -> np.ndarray:
    """Add a dark header banner with crisp white title text to a panel."""
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()
    h, w, _ = img_bgr.shape
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (20, 20, 20)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42 if w >= 160 else 0.32
    thick = 1
    cv2.putText(banner, title, (6, banner_h - 8), font, scale, (240, 240, 240), thick, cv2.LINE_AA)
    return np.vstack([banner, img_bgr])


def create_7panel_comparison(
    source: np.ndarray,
    gt: np.ndarray,
    raw_dart: np.ndarray,
    thresh_mask: np.ndarray,
    scratch_ch: np.ndarray,
    dust_ch: np.ndarray,
    crop_box: tuple[int, int, int, int] | None = None,  # (y0, y1, x0, x1)
) -> np.ndarray:
    """Create a unified 7-panel composite."""
    if crop_box is not None:
        y0, y1, x0, x1 = crop_box
        source = source[y0:y1, x0:x1]
        gt = gt[y0:y1, x0:x1]
        raw_dart = raw_dart[y0:y1, x0:x1]
        thresh_mask = thresh_mask[y0:y1, x0:x1]
        scratch_ch = scratch_ch[y0:y1, x0:x1]
        dust_ch = dust_ch[y0:y1, x0:x1]

    # Combined prediction for error map
    comb = np.maximum(scratch_ch, dust_ch)
    error_map = build_error_map(gt, comb)
    
    # GT: Green
    gt_vis = np.zeros_like(source)
    gt_vis[gt == 255] = (0, 255, 0)
    if np.any(gt == 127):
        gt_vis[gt == 127] = (127, 127, 127)
        
    # Raw DART
    raw_vis = cv2.cvtColor(raw_dart, cv2.COLOR_GRAY2BGR)
    
    # Thresh: white
    thresh_vis = cv2.cvtColor(thresh_mask, cv2.COLOR_GRAY2BGR)
    
    # Scratch channel: Gold/Yellow
    s_vis = np.zeros_like(source)
    s_vis[scratch_ch > 0] = (0, 215, 255)
    
    # Dust channel: Cyan
    d_vis = np.zeros_like(source)
    d_vis[dust_ch > 0] = (255, 200, 0)
    
    panels = [
        make_labeled_panel(source, "1. Source"),
        make_labeled_panel(gt_vis, "2. Ground Truth"),
        make_labeled_panel(raw_vis, "3. Raw DART (0..255)"),
        make_labeled_panel(thresh_vis, "4. Binary (>160)"),
        make_labeled_panel(s_vis, "5. Scratch Ch"),
        make_labeled_panel(d_vis, "6. Dust Ch"),
        make_labeled_panel(error_map, "7. Error Map"),
    ]
    return np.hstack(panels)


def split_baseline(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the baseline bug: axis-aligned bounding box clamp + 2.5% clamp + uncat dump to dust."""
    h, w = binary_mask.shape
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    dust_mask = np.zeros((h, w), dtype=np.uint8)
    if not np.any(binary_mask):
        return scratch_mask, dust_mask

    rough = (binary_mask > 0).astype(np.uint8) * 255
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, min(15, h // 25))))
    closed_vert = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k_vert)
    total, labels, stats, _ = cv2.connectedComponentsWithStats(closed_vert, connectivity=8)
    max_blob_area = max(8, min(180, int(round(h * w * 0.00035))))
    max_blob_edge = max(7, min(25, int(round(min(h, w) * 0.028))))

    for component in range(1, total):
        _x, _y, cw, ch, area = map(int, stats[component])
        comp_mask = labels == component
        fine_scratch = (
            cw <= max(8, int(w * 0.06))
            and ch >= 10
            and ch <= max(12, int(h * 0.60))
            and area <= max(40, int(h * 0.75))
        )
        small_spot = (
            not fine_scratch
            and area <= max_blob_area
            and cw <= max_blob_edge
            and ch <= max_blob_edge
        )
        if fine_scratch:
            scratch_mask[comp_mask & (rough > 0)] = 255
        elif small_spot:
            dust_mask[comp_mask & (rough > 0)] = 255

    # 2.5% area cutoff
    if np.count_nonzero(scratch_mask) / scratch_mask.size > 0.025:
        scratch_mask[:] = 0
    if np.count_nonzero(dust_mask) / dust_mask.size > 0.025:
        dust_mask[:] = 0

    # Uncat dump bug
    categorized = (scratch_mask > 0) | (dust_mask > 0)
    uncat = (rough > 0) & ~categorized
    if np.any(uncat):
        dust_mask = np.where(uncat, np.uint8(255), dust_mask)
    return scratch_mask, dust_mask


def split_corrected(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Corrected split using rotated bounding box, compactness check, and no uncat dump."""
    return cde.split_defect_components(binary_mask)


def run_dart_inference_clip(frames: list[np.ndarray]) -> list[np.ndarray]:
    dart_py = cde.find_isolated_dart_python()
    weights = Path("I:/DART/pretrained_models/dart.safetensors")
    with tempfile.TemporaryDirectory(prefix="dart-crops-") as tmp:
        tpath = Path(tmp)
        in_dir = tpath / "in"
        out_dir = tpath / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)
        for idx, f in enumerate(frames):
            cv2.imwrite(str(in_dir / f"frame_{idx:06d}.png"), f)
        cmd = [
            str(dart_py), r"I:\DART\scripts\inference.py",
            "--input", str(in_dir), "--output", str(out_dir),
            "--weights", str(weights), "--save-masks", "--device", "cuda"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        masks_dir = out_dir / "masks"
        return [cv2.imread(str(masks_dir / f"frame_{idx:06d}.png"), cv2.IMREAD_UNCHANGED) for idx in range(len(frames))]


def main():
    out_dir = Path("docs/investigation/visual_crops")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating investigation visual crops...")
    
    # 1. Synthetic Scratch (10 frames)
    synth_records = rdbs.load_dataset_records(Path(".cache/benchmarks/synthetic"))
    s_records = [r for r in synth_records if r["clip"] == "synthetic_scratch"]
    s_frames = [r["image"] for r in s_records]
    s_raw = run_dart_inference_clip(s_frames)
    
    for f_idx in [0, 2]:
        raw = s_raw[f_idx]
        gt = s_records[f_idx]["gt_mask"]
        src = s_frames[f_idx]
        b160 = (raw >= 160).astype(np.uint8) * 255
        
        # Baseline Bug
        s_old, d_old = split_baseline(b160)
        comp_old = create_7panel_comparison(src, gt, raw, b160, s_old, d_old)
        cv2.imwrite(str(out_dir / f"synthetic_scratch_frame_{f_idx}_baseline_bug.png"), comp_old)
        
        # Corrected
        s_new, d_new = split_corrected(b160)
        comp_new = create_7panel_comparison(src, gt, raw, b160, s_new, d_new)
        cv2.imwrite(str(out_dir / f"synthetic_scratch_frame_{f_idx}_corrected.png"), comp_new)
        
    # 2. Synthetic Dust (frame 0)
    d_records = [r for r in synth_records if r["clip"] == "synthetic_dust"]
    d_frames = [r["image"] for r in d_records]
    d_raw = run_dart_inference_clip(d_frames)
    
    raw_d0 = d_raw[0]
    gt_d0 = d_records[0]["gt_mask"]
    src_d0 = d_frames[0]
    b160_d0 = (raw_d0 >= 160).astype(np.uint8) * 255
    
    # Baseline Bug
    s_old_d, d_old_d = split_baseline(b160_d0)
    comp_d0_old = create_7panel_comparison(src_d0, gt_d0, raw_d0, b160_d0, s_old_d, d_old_d)
    cv2.imwrite(str(out_dir / "synthetic_dust_frame_0_baseline_bug.png"), comp_d0_old)
    
    # Corrected
    s_new_d, d_new_d = split_corrected(b160_d0)
    comp_d0_new = create_7panel_comparison(src_d0, gt_d0, raw_d0, b160_d0, s_new_d, d_new_d)
    cv2.imwrite(str(out_dir / "synthetic_dust_frame_0_corrected.png"), comp_d0_new)

    # 3. AbsoluteDegradation clip 010 (frame 0 and frame 1)
    absdeg_records = rdbs.load_dataset_records(Path(".cache/benchmarks/absolute_degradation"))
    c010_records = [r for r in absdeg_records if r["clip"] == "absdeg_010"]
    c010_frames = [r["image"] for r in c010_records]
    c010_raw = run_dart_inference_clip(c010_frames)
    
    # High-resolution crop around vertical scratches (y: 100..500, x: 250..650)
    crop_box = (100, 500, 250, 650)
    for f_idx in [0, 1]:
        raw = c010_raw[f_idx]
        gt = c010_records[f_idx]["gt_mask"]
        src = c010_frames[f_idx]
        b160 = (raw >= 160).astype(np.uint8) * 255
        
        # Baseline Bug
        s_old, d_old = split_baseline(b160)
        comp_old = create_7panel_comparison(src, gt, raw, b160, s_old, d_old, crop_box=crop_box)
        cv2.imwrite(str(out_dir / f"absdeg_010_frame_{f_idx}_crop_baseline_bug.png"), comp_old)
        
        # Corrected
        s_new, d_new = split_corrected(b160)
        comp_new = create_7panel_comparison(src, gt, raw, b160, s_new, d_new, crop_box=crop_box)
        cv2.imwrite(str(out_dir / f"absdeg_010_frame_{f_idx}_crop_corrected.png"), comp_new)
        
    print(f"Visual crops successfully generated in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
