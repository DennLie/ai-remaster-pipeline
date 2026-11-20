"""Diagnostic script to compare feature layers, color spaces, scale, and distribution models
for PatchCore and PaDiM on archival film defect detection.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

class DiagnosticExtractor(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        weights = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.eval()
        for p in self.parameters():
            p.requires_grad = False
        self.to(self.device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))

    def forward(self, x: torch.Tensor):
        x = x.to(self.device)
        x_norm = (x - self.mean) / self.std
        h0 = self.maxpool(self.relu(self.bn1(self.conv1(x_norm))))
        f1 = self.layer1(h0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return {"layer1": f1, "layer2": f2, "layer3": f3}

def run_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running diagnostic on {device}")
    extractor = DiagnosticExtractor(device=device)

    # Load 000 reference (clean) and 000 damaged frame with GT
    bench_dir = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000"
    ref_path = bench_dir / "reference/000000.png"
    dmg_path = bench_dir / "20261106/000000.png"
    gt_path = bench_dir / "20261106/000000.gt.png"

    if not (ref_path.is_file() and dmg_path.is_file() and gt_path.is_file()):
        print("[-] Test files missing:", ref_path, dmg_path, gt_path)
        return

    ref_bgr = cv2.imread(str(ref_path))
    dmg_bgr = cv2.imread(str(dmg_path))
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    h, w = ref_bgr.shape[:2]
    print(f"[*] Loaded frame: {w}x{h}, defect pixels: {np.count_nonzero(gt_mask == 255)}")

    # Test 1: Color Space impact (BGR vs RGB)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    t_bgr = torch.from_numpy(ref_bgr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t_rgb = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        out_bgr = extractor(t_bgr)
        out_rgb = extractor(t_rgb)
    diff_l1 = torch.norm(out_bgr["layer1"] - out_rgb["layer1"]).item() / out_rgb["layer1"].numel()
    diff_l2 = torch.norm(out_bgr["layer2"] - out_rgb["layer2"]).item() / out_rgb["layer2"].numel()
    print(f"[!] Color Space Bug Impact (BGR vs RGB mean diff): layer1={diff_l1:.4f}, layer2={diff_l2:.4f}")

    # Test 2: Contrast & Separation of Defect vs Intact under different layer combinations
    # Compare feature distance between clean frame and damaged frame at defect locations vs intact locations
    dmg_rgb = cv2.cvtColor(dmg_bgr, cv2.COLOR_BGR2RGB)
    t_dmg = torch.from_numpy(dmg_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        clean_feats = extractor(t_rgb)
        dmg_feats = extractor(t_dmg)

    pool = nn.AvgPool2d(3, stride=1, padding=1)

    layer_configs = {
        "layer1 (stride 4, dim 64)": (clean_feats["layer1"], dmg_feats["layer1"]),
        "layer2 (stride 8, dim 128)": (clean_feats["layer2"], dmg_feats["layer2"]),
        "layer3 (stride 16, dim 256)": (clean_feats["layer3"], dmg_feats["layer3"]),
        "layer1+layer2 (stride 4, dim 192)": (
            torch.cat([clean_feats["layer1"], F.interpolate(clean_feats["layer2"], size=clean_feats["layer1"].shape[-2:], mode="bilinear", align_corners=False)], dim=1),
            torch.cat([dmg_feats["layer1"], F.interpolate(dmg_feats["layer2"], size=dmg_feats["layer1"].shape[-2:], mode="bilinear", align_corners=False)], dim=1)
        ),
        "layer2+layer3 (stride 8, dim 384 - old setup)": (
            torch.cat([clean_feats["layer2"], F.interpolate(clean_feats["layer3"], size=clean_feats["layer2"].shape[-2:], mode="bilinear", align_corners=False)], dim=1),
            torch.cat([dmg_feats["layer2"], F.interpolate(dmg_feats["layer3"], size=dmg_feats["layer2"].shape[-2:], mode="bilinear", align_corners=False)], dim=1)
        ),
    }

    print("\n--- Defect vs Intact Signal-to-Noise Ratio (Clean vs Damaged frame at identical coordinates) ---")
    gt_defect_mask = (gt_mask == 255)
    gt_intact_mask = (gt_mask == 0)

    for name, (f_clean, f_dmg) in layer_configs.items():
        # Apply neighborhood pooling
        p_clean = pool(f_clean)
        p_dmg = pool(f_dmg)
        # Compute L2 distance per spatial cell
        cell_dist = torch.norm(p_clean - p_dmg, dim=1, keepdim=True)  # [1, 1, Hf, Wf]
        # Upsample to full resolution
        dist_up = F.interpolate(cell_dist, size=(h, w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        defect_score = float(np.mean(dist_up[gt_defect_mask])) if np.any(gt_defect_mask) else 0.0
        intact_score = float(np.mean(dist_up[gt_intact_mask]))
        snr = defect_score / max(intact_score, 1e-6)
        print(f"{name:45s} | Defect Dist: {defect_score:.4f} | Intact Dist: {intact_score:.4f} | Ratio: {snr:.2f}x")

    # Test 3: PaDiM Spatially Invariant Distribution vs Position-Wise
    # Let's test a global Gaussian distribution on layer1+layer2
    f_c = torch.cat([clean_feats["layer1"], F.interpolate(clean_feats["layer2"], size=clean_feats["layer1"].shape[-2:], mode="bilinear", align_corners=False)], dim=1)
    f_d = torch.cat([dmg_feats["layer1"], F.interpolate(dmg_feats["layer2"], size=dmg_feats["layer1"].shape[-2:], mode="bilinear", align_corners=False)], dim=1)
    b, c, h1, w1 = f_c.shape
    flat_clean = f_c.squeeze(0).permute(1, 2, 0).reshape(-1, c) # [N, C]
    
    # Global mean and covariance across all normal patches
    mu_global = flat_clean.mean(dim=0)
    diff = flat_clean - mu_global.unsqueeze(0)
    cov_global = torch.mm(diff.t(), diff) / (flat_clean.shape[0] - 1)
    cov_reg = cov_global + torch.eye(c, device=device) * 0.05
    inv_cov_global = torch.linalg.pinv(cov_reg)

    # Compute Mahalanobis on clean and damaged
    flat_dmg = f_d.squeeze(0).permute(1, 2, 0).reshape(-1, c)
    diff_dmg = flat_dmg - mu_global.unsqueeze(0)
    temp = torch.mm(diff_dmg, inv_cov_global)
    mahal_dmg = torch.sqrt(torch.clamp((temp * diff_dmg).sum(dim=1), min=0.0)).reshape(1, 1, h1, w1)
    mahal_up = F.interpolate(mahal_dmg, size=(h, w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

    m_defect = float(np.mean(mahal_up[gt_defect_mask]))
    m_intact = float(np.mean(mahal_up[gt_intact_mask]))
    print(f"\nPaDiM Spatially-Invariant Global Gaussian (layer1+layer2):")
    print(f"  Defect Mahalanobis: {m_defect:.4f} | Intact Mahalanobis: {m_intact:.4f} | Ratio: {m_defect / max(m_intact, 1e-6):.2f}x")

if __name__ == "__main__":
    run_diagnostic()
