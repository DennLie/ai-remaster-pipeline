"""Calibrate PatchCore and PaDiM presets across multiple ground truth frames and clean negative controls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.anomalib.models import load_anomaly_detector
from tools.anomalib.run_anomaly_detector import predict_frame

def run_calibration():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Calibration sweep running on device: {device}")

    pc_path = ROOT / "tools/checkpoints/patchcore/patchcore_memory_bank.pt"
    padim_path = ROOT / "tools/checkpoints/padim/padim_distribution.pt"

    pc_detector = load_anomaly_detector(pc_path, model_name="patchcore", device=device)
    padim_detector = load_anomaly_detector(padim_path, model_name="padim", device=device)

    # Collect validation frames:
    # 1. Damaged frames with GT: 000/20261106/000000.png, 000001.png, 000002.png
    bench_dir = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/20261106"
    dmg_records = []
    for idx in range(3):
        stem = f"{idx:06d}"
        d_p = bench_dir / f"{stem}.png"
        g_p = bench_dir / f"{stem}.gt.png"
        if d_p.is_file() and g_p.is_file():
            img = cv2.imread(str(d_p))
            gt = cv2.imread(str(g_p), cv2.IMREAD_GRAYSCALE)
            dmg_records.append((f"000_{stem}", img, gt))

    # 2. Clean negative controls: 000/reference/000010.png, 000011.png
    ref_dir = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/reference"
    clean_records = []
    for idx in [10, 11]:
        stem = f"{idx:06d}"
        r_p = ref_dir / f"{stem}.png"
        if r_p.is_file():
            img = cv2.imread(str(r_p))
            gt = np.zeros(img.shape[:2], dtype=np.uint8)
            clean_records.append((f"clean_{stem}", img, gt))

    print(f"[*] Evaluation set: {len(dmg_records)} damaged frames with GT, {len(clean_records)} clean negative controls")

    for model_name, detector in [("PatchCore", pc_detector), ("PaDiM", padim_detector)]:
        print(f"\n" + "=" * 60)
        print(f"CALIBRATION SWEEP: {model_name}")
        print("=" * 60)

        # Precompute probability maps for all frames
        prob_maps = []
        for name, img, gt in dmg_records + clean_records:
            p_map, _ = predict_frame(detector, img, tile_size=0, gaussian_sigma=2.0)
            prob_maps.append((name, p_map, gt))

        # Sweep thresholds
        thresholds = np.linspace(0.40, 0.85, 10)
        best_f1 = -1.0
        best_thresh = 0.5

        for th in thresholds:
            tps, fps, fns, tns = 0, 0, 0, 0
            clean_fps, clean_total = 0, 0

            for name, p_map, gt in prob_maps:
                pred = (p_map >= th)
                gt_def = (gt == 255)
                gt_int = (gt == 0)

                if "clean_" in name:
                    clean_fps += np.count_nonzero(pred)
                    clean_total += pred.size
                else:
                    tps += np.count_nonzero(pred & gt_def)
                    fps += np.count_nonzero(pred & gt_int)
                    fns += np.count_nonzero(~pred & gt_def)
                    tns += np.count_nonzero(~pred & gt_int)

            prec = tps / max(tps + fps, 1)
            rec = tps / max(tps + fns, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-6)
            intact_fpr = fps / max(tns + fps, 1)
            clean_fpr = clean_fps / max(clean_total, 1)

            print(
                f"Thresh {th:.3f} | Prec: {prec*100:5.2f}% | Rec: {rec*100:5.2f}% | "
                f"F1: {f1:.4f} | Intact FPR: {intact_fpr*100:5.2f}% | Clean Neg FPR: {clean_fpr*100:5.2f}%"
            )

if __name__ == "__main__":
    run_calibration()
