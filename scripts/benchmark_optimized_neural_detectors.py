#!/usr/bin/env python3
"""
Benchmark Optimized Neural Defect Detectors on 72-frame Reference Video
======================================================================
Tests:
  1. BOPB Scratch UNet (Research)
  2. Film Artefact Segmenter (Commercial)
  3. CNN Blotches & Scratches (Commercial)

Evaluates:
  - Total wall-clock time
  - Effective FPS
  - Real CUDA execution & Peak VRAM
  - Output mask equivalence against original unoptimized baseline runs
"""

import sys
import os
import time
import json
import uuid
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai_remaster_gui"))
sys.path.insert(0, str(ROOT / "scripts"))

from ai_remaster_gui import dust
import torch

REFERENCE_VIDEO = ROOT / "intermediate" / "stabilized" / "Neighbors_stabilized_eb46d58c.mkv"

BASELINES = {
    "bopb": {
        "folder": ROOT / "intermediate" / "dust_scratches" / "e21fbefffe9e5d1807efd1b4",
        "preset": "bopb_scratch_balanced",
        "threshold": 128,
        "sensitivity": 0.5,
        "dilate": 0,
        "repair_policy": "scratch_only",
        "baseline_runtime_s": 673.0,
        "baseline_fps": 72.0 / 673.0,
        "research_confirmed": "true",
        "license_confirmed": "true",
    },
    "film_artefact_segmenter": {
        "folder": ROOT / "intermediate" / "dust_scratches" / "fb2b8d095e66add710c430c3",
        "preset": "film_artefact_balanced",
        "threshold": 128,
        "sensitivity": 0.5,
        "dilate": 0,
        "repair_policy": "conservative",
        "baseline_runtime_s": 260.0,
        "baseline_fps": 72.0 / 260.0,
        "research_confirmed": "true",
        "license_confirmed": "true",
    },
    "cnn_blotches": {
        "folder": ROOT / "intermediate" / "dust_scratches" / "d0ed547a00531de7a8f1b588",
        "preset": "cnn_blotches_precision",
        "threshold": 204,
        "sensitivity": 0.8,
        "dilate": 0,
        "repair_policy": "conservative",
        "baseline_runtime_s": 360.0,
        "baseline_fps": 72.0 / 360.0,
        "research_confirmed": "true",
        "license_confirmed": "true",
    },
}

def compare_masks(mask_a_path: Path, mask_b_path: Path) -> dict:
    cap_a = cv2.VideoCapture(str(mask_a_path))
    cap_b = cv2.VideoCapture(str(mask_b_path))

    frame_count = 0
    ious = []
    pixel_matches = []
    mae_list = []

    while True:
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            break
        frame_count += 1
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY) if frame_a.ndim == 3 else frame_a
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY) if frame_b.ndim == 3 else frame_b

        bin_a = gray_a > 127
        bin_b = gray_b > 127

        intersection = np.logical_and(bin_a, bin_b).sum()
        union = np.logical_or(bin_a, bin_b).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)

        match_pct = float((gray_a == gray_b).sum()) / float(gray_a.size) * 100.0
        pixel_matches.append(match_pct)
        mae_list.append(float(np.mean(np.abs(gray_a.astype(float) - gray_b.astype(float)))))

    cap_a.release()
    cap_b.release()

    return {
        "frames_compared": frame_count,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "min_iou": float(np.min(ious)) if ious else 0.0,
        "mean_pixel_match_pct": float(np.mean(pixel_matches)) if pixel_matches else 0.0,
        "mean_absolute_error": float(np.mean(mae_list)) if mae_list else 0.0,
    }

def get_peak_vram_mib() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0

def run_benchmark():
    assert REFERENCE_VIDEO.is_file(), f"Reference video missing: {REFERENCE_VIDEO}"
    print(f"=== Starting Optimized Neural Detectors Benchmark ===")
    print(f"Reference Video: {REFERENCE_VIDEO}")
    print(f"CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}\n")

    results = {}

    for det_key, base_info in BASELINES.items():
        print(f"--------------------------------------------------")
        print(f"Benchmarking: {det_key} (Preset: {base_info['preset']})")
        print(f"Baseline unoptimized: {base_info['baseline_runtime_s']:.1f}s ({base_info['baseline_fps']:.2f} FPS)")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        # Generate a unique 32-hex mask_version to guarantee fresh inference without cache hit
        mask_version = uuid.uuid4().hex

        values = {
            "detector": det_key,
            "preset": base_info["preset"],
            "threshold": base_info["threshold"],
            "sensitivity": base_info["sensitivity"],
            "dilate": base_info["dilate"],
            "repair_policy": base_info["repair_policy"],
            "mask_version": mask_version,
            "research_confirmed": base_info["research_confirmed"],
            "license_confirmed": base_info["license_confirmed"],
        }

        start_time = time.perf_counter()
        report = dust.run(REFERENCE_VIDEO, values, kind="mask")
        elapsed_s = time.perf_counter() - start_time

        peak_vram_mib = get_peak_vram_mib()
        fps = 72.0 / elapsed_s
        speedup = base_info["baseline_runtime_s"] / elapsed_s

        print(f"Completed {det_key}: {elapsed_s:.2f}s ({fps:.2f} FPS), Peak VRAM: {peak_vram_mib:.1f} MiB")
        print(f"Speedup vs Baseline: {speedup:.2f}x")

        # Compare generated masks against baseline
        gen_folder = dust.folder_for(REFERENCE_VIDEO, values)
        base_folder = base_info["folder"]

        mask_cmp = compare_masks(gen_folder / "mask.mkv", base_folder / "mask.mkv")
        print(f"Mask Equivalence (mask.mkv): Mean IoU={mask_cmp['mean_iou']:.4f}, Pixel Match={mask_cmp['mean_pixel_match_pct']:.2f}%, MAE={mask_cmp['mean_absolute_error']:.4f}")

        # If scratch mask exists, compare scratch mask too
        scratch_cmp = None
        if (gen_folder / "scratch_mask.mkv").is_file() and (base_folder / "scratch_mask.mkv").is_file():
            scratch_cmp = compare_masks(gen_folder / "scratch_mask.mkv", base_folder / "scratch_mask.mkv")
            print(f"Scratch Mask Equivalence: Mean IoU={scratch_cmp['mean_iou']:.4f}, Pixel Match={scratch_cmp['mean_pixel_match_pct']:.2f}%")

        results[det_key] = {
            "preset": base_info["preset"],
            "baseline_runtime_s": base_info["baseline_runtime_s"],
            "baseline_fps": base_info["baseline_fps"],
            "optimized_runtime_s": elapsed_s,
            "optimized_fps": fps,
            "speedup_factor": speedup,
            "peak_vram_mib": peak_vram_mib,
            "mask_equivalence": mask_cmp,
            "scratch_mask_equivalence": scratch_cmp,
            "output_folder": str(gen_folder),
            "baseline_folder": str(base_folder),
        }

    output_json = ROOT / "intermediate" / "optimized_neural_detectors_benchmark.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Benchmark Complete! Saved results to {output_json} ===")

if __name__ == "__main__":
    run_benchmark()
