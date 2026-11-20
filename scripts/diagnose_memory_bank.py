"""Test cross-frame PatchCore nearest neighbor retrieval using clean archival corpus.
"""
from __future__ import annotations

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

class FeatureExtractor(nn.Module):
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

def test_cross_frame():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fe = FeatureExtractor(device=device)
    pool = nn.AvgPool2d(3, stride=1, padding=1)

    # 1. Collect clean training frames from 000 reference (frames 1 to 20)
    ref_dir = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/reference"
    clean_paths = sorted(list(ref_dir.glob("*.png")))[1:15] # 14 clean frames

    # Extract clean patches for layer1+layer2 and layer2+layer3
    bank_12 = []
    bank_23 = []

    for p in clean_paths:
        img = cv2.imread(str(p))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            feats = fe(t)
            # layer1+layer2 (stride 4, dim 192)
            f1 = feats["layer1"]
            f2 = F.interpolate(feats["layer2"], size=f1.shape[-2:], mode="bilinear", align_corners=False)
            cat12 = pool(torch.cat([f1, f2], dim=1)) # [1, 192, H1, W1]
            flat12 = cat12.squeeze(0).permute(1, 2, 0).reshape(-1, 192)
            # Subsample randomly
            idx = torch.randperm(flat12.shape[0])[:1000]
            bank_12.append(flat12[idx])

            # layer2+layer3 (stride 8, dim 384)
            f2_raw = feats["layer2"]
            f3 = F.interpolate(feats["layer3"], size=f2_raw.shape[-2:], mode="bilinear", align_corners=False)
            cat23 = pool(torch.cat([f2_raw, f3], dim=1))
            flat23 = cat23.squeeze(0).permute(1, 2, 0).reshape(-1, 384)
            idx23 = torch.randperm(flat23.shape[0])[:1000]
            bank_23.append(flat23[idx23])

    full_bank_12 = torch.cat(bank_12, dim=0) # [14000, 192]
    full_bank_23 = torch.cat(bank_23, dim=0) # [14000, 384]

    # Test on held-out damaged frame 000000.png with GT
    dmg_path = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/20261106/000000.png"
    gt_path = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/20261106/000000.gt.png"
    dmg_img = cv2.imread(str(dmg_path))
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    h, w = dmg_img.shape[:2]

    rgb_dmg = cv2.cvtColor(dmg_img, cv2.COLOR_BGR2RGB)
    t_dmg = torch.from_numpy(rgb_dmg).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        d_feats = fe(t_dmg)
        # Query 12
        q1 = d_feats["layer1"]
        q2 = F.interpolate(d_feats["layer2"], size=q1.shape[-2:], mode="bilinear", align_corners=False)
        q12 = pool(torch.cat([q1, q2], dim=1)).squeeze(0).permute(1, 2, 0).reshape(-1, 192)

        # Batch NN distance to full_bank_12
        # Use torch.cdist in chunks to avoid OOM
        dists_12 = []
        chunk_sz = 2000
        for i in range(0, q12.shape[0], chunk_sz):
            q_c = q12[i : i + chunk_sz]
            # dist = min over memory bank
            d = torch.cdist(q_c, full_bank_12).min(dim=1).values
            dists_12.append(d)
        all_dists_12 = torch.cat(dists_12, dim=0).reshape(1, 1, q1.shape[-2], q1.shape[-1])
        up_12 = F.interpolate(all_dists_12, size=(h, w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        # Query 23
        q2_raw = d_feats["layer2"]
        q3 = F.interpolate(d_feats["layer3"], size=q2_raw.shape[-2:], mode="bilinear", align_corners=False)
        q23 = pool(torch.cat([q2_raw, q3], dim=1)).squeeze(0).permute(1, 2, 0).reshape(-1, 384)
        dists_23 = []
        for i in range(0, q23.shape[0], chunk_sz):
            q_c = q23[i : i + chunk_sz]
            d = torch.cdist(q_c, full_bank_23).min(dim=1).values
            dists_23.append(d)
        all_dists_23 = torch.cat(dists_23, dim=0).reshape(1, 1, q2_raw.shape[-2], q2_raw.shape[-1])
        up_23 = F.interpolate(all_dists_23, size=(h, w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

    gt_defect = (gt_mask == 255)
    gt_intact = (gt_mask == 0)

    print("\n=== Cross-Frame Nearest Neighbor Retrieval Comparison ===")
    for name, up_map in [("layer1+layer2 (stride 4)", up_12), ("layer2+layer3 (stride 8 - old)", up_23)]:
        d_mean = float(np.mean(up_map[gt_defect]))
        i_mean = float(np.mean(up_map[gt_intact]))
        i_p95 = float(np.percentile(up_map[gt_intact], 95))
        i_p99 = float(np.percentile(up_map[gt_intact], 99))
        print(f"\n{name}:")
        print(f"  Defect distance mean: {d_mean:.4f}")
        print(f"  Intact distance mean: {i_mean:.4f} (p95={i_p95:.4f}, p99={i_p99:.4f})")
        print(f"  Separation (Defect / Intact): {d_mean / i_mean:.2f}x")
        # Check AUC or thresholding
        threshold = (d_mean + i_p95) / 2.0
        pred = (up_map >= threshold)
        tp = np.count_nonzero(pred & gt_defect)
        fp = np.count_nonzero(pred & gt_intact)
        fn = np.count_nonzero(~pred & gt_defect)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        fpr = fp / max(np.count_nonzero(gt_intact), 1)
        print(f"  At threshold {threshold:.4f}: Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, Intact FPR={fpr:.3%}")

if __name__ == "__main__":
    test_cross_frame()
