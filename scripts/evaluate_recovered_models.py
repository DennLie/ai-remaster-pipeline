"""Evaluate recovered PatchCore and PaDiM models against original broken baselines
on both held-out benchmark GT and authentic historical archival footage (Neighbors 1920).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.anomalib.models import load_anomaly_detector
from tools.anomalib.run_anomaly_detector import predict_frame

def evaluate_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Evaluation running on device: {device}")

    # Load new recovered models
    pc_path = ROOT / "tools/checkpoints/patchcore/patchcore_memory_bank.pt"
    padim_path = ROOT / "tools/checkpoints/padim/padim_distribution.pt"

    pc_detector = load_anomaly_detector(pc_path, model_name="patchcore", device=device)
    padim_detector = load_anomaly_detector(padim_path, model_name="padim", device=device)

    # Load original broken checkpoints for direct before/after comparison
    pc_old_path = ROOT / "tools/checkpoints/patchcore/patchcore_memory_bank.pt.original_broken_fit.bak"
    padim_old_path = ROOT / "tools/checkpoints/padim/padim_distribution.pt.original_broken_fit.bak"

    pc_old_detector = load_anomaly_detector(pc_old_path, model_name="patchcore", device=device)
    padim_old_detector = load_anomaly_detector(padim_old_path, model_name="padim", device=device)

    # 1. Benchmark Ground Truth Test (000/20261106/000000.png)
    dmg_path = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/20261106/000000.png"
    gt_path = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/20261106/000000.gt.png"

    dmg_img = cv2.imread(str(dmg_path))
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    gt_defect = (gt_mask == 255)
    gt_intact = (gt_mask == 0)

    print("\n" + "=" * 75)
    print("PART 1: HELD-OUT BENCHMARK GROUND TRUTH EVALUATION (000000.png)")
    print("=" * 75)

    detectors = [
        ("PatchCore (BEFORE: Broken)", pc_old_detector, 256),
        ("PatchCore (AFTER: Recovered)", pc_detector, 0),
        ("PaDiM (BEFORE: Broken)", padim_old_detector, 256),
        ("PaDiM (AFTER: Recovered)", padim_detector, 0),
    ]

    bench_results = {}
    for name, det, tile_sz in detectors:
        t0 = time.time()
        prob_map, mask = predict_frame(det, dmg_img, tile_size=tile_sz, gaussian_sigma=2.0)
        dt = (time.time() - t0) * 1000.0

        pred_defect = (mask == 255)
        tp = int(np.count_nonzero(pred_defect & gt_defect))
        fp = int(np.count_nonzero(pred_defect & gt_intact))
        fn = int(np.count_nonzero(~pred_defect & gt_defect))
        tn = int(np.count_nonzero(~pred_defect & gt_intact))

        coverage = float(np.mean(pred_defect))
        intact_fpr = fp / max(tn + fp, 1)
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        iou = tp / max(tp + fp + fn, 1)

        mean_intact_prob = float(np.mean(prob_map[gt_intact]))
        mean_defect_prob = float(np.mean(prob_map[gt_defect]))

        bench_results[name] = {
            "coverage_pct": round(coverage * 100.0, 2),
            "intact_fpr_pct": round(intact_fpr * 100.0, 3),
            "recall_pct": round(recall * 100.0, 2),
            "precision_pct": round(precision * 100.0, 2),
            "pixel_f1": round(f1, 4),
            "iou": round(iou, 4),
            "mean_intact_score": round(mean_intact_prob, 4),
            "mean_defect_score": round(mean_defect_prob, 4),
            "latency_ms": round(dt, 1),
        }

        print(f"\nModel: {name}")
        print(f"  Coverage: {coverage * 100.0:.2f}% | Intact FPR: {intact_fpr * 100.0:.3f}%")
        print(f"  Recall: {recall * 100.0:.2f}% | Precision: {precision * 100.0:.2f}% | Pixel-F1: {f1:.4f} | IoU: {iou:.4f}")
        print(f"  Mean Intact Score: {mean_intact_prob:.4f} | Mean Defect Score: {mean_defect_prob:.4f}")
        print(f"  Inference Latency: {dt:.1f}ms")

    # 2. Historical Archival Footage Evaluation (Neighbors 1920)
    print("\n" + "=" * 75)
    print("PART 2: AUTHENTIC ARCHIVAL FILM CLIP EVALUATION (Neighbors 1920)")
    print("=" * 75)
    video_path = ROOT / "intermediate/stabilized/Neighbors_stabilized_eb46d58c.mkv"
    if not video_path.is_file():
        video_path = ROOT / "evaluation/Neighbors_72.mkv"

    cap = cv2.VideoCapture(str(video_path))
    archival_frames = []
    # Test on 10 spaced frames from Neighbors
    frame_indices = [0, 8, 16, 24, 32, 40, 48, 56, 64, 71]
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret and f is not None:
            archival_frames.append((idx, f))
    cap.release()

    print(f"[*] Loaded {len(archival_frames)} frames from {video_path.name}")

    neighbors_results = {}
    for name, det, tile_sz in detectors:
        coverages = []
        latencies = []
        for idx, f in archival_frames:
            t0 = time.time()
            prob_map, mask = predict_frame(det, f, tile_size=tile_sz, gaussian_sigma=2.0)
            dt = (time.time() - t0) * 1000.0
            cov = float(np.mean(mask == 255))
            coverages.append(cov)
            latencies.append(dt)

        mean_cov = float(np.mean(coverages))
        max_cov = float(np.max(coverages))
        mean_lat = float(np.mean(latencies))
        fps = 1000.0 / max(mean_lat, 1e-3)

        neighbors_results[name] = {
            "mean_coverage_pct": round(mean_cov * 100.0, 2),
            "max_coverage_pct": round(max_cov * 100.0, 2),
            "mean_latency_ms": round(mean_lat, 1),
            "effective_fps": round(fps, 1),
            "safety_guard_triggered (>30%)": bool(mean_cov > 0.30),
        }

        print(f"\nModel: {name}")
        print(f"  Mean Mask Coverage: {mean_cov * 100.0:.2f}% (Max: {max_cov * 100.0:.2f}%)")
        print(f"  Latency: {mean_lat:.1f}ms ({fps:.1f} FPS)")
        print(f"  Safety Guard (30% threshold): {'TRIGGERED (Destructive)' if mean_cov > 0.30 else 'SAFE (Passed)'}")

    # Save summary report
    out_dir = ROOT / "docs/benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_gt": bench_results,
        "neighbors_archival": neighbors_results,
    }
    with open(out_dir / "anomalib_recovery_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[✓] Evaluation report written to docs/benchmarks/anomalib_recovery_evaluation.json")

if __name__ == "__main__":
    evaluate_models()
