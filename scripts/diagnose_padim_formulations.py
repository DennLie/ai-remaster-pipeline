"""Compare PaDiM formulations for film defect detection.
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

class PaDiMExtractor(nn.Module):
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
        f1 = self.layer1(h0) # [B, 64, H/4, W/4]
        f2 = self.layer2(f1) # [B, 128, H/8, W/8]
        f2_up = F.interpolate(f2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f1, f2_up], dim=1) # [B, 192, H/4, W/4]

def test_padim_formulations():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pe = PaDiMExtractor(device=device)

    # Clean frames
    ref_dir = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/reference"
    clean_paths = sorted(list(ref_dir.glob("*.png")))[1:15]

    all_patches = []
    for p in clean_paths:
        img = cv2.imread(str(p))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            f = pe(t) # [1, 192, H1, W1]
            flat = f.squeeze(0).permute(1, 2, 0).reshape(-1, 192)
            sub = flat[torch.randperm(flat.shape[0])[:2000]]
            all_patches.append(sub)

    train_data = torch.cat(all_patches, dim=0) # [28000, 192]
    n_samples, d = train_data.shape

    # Random dimension selection (standard PaDiM reduction)
    # Reduce 192 to 64 or 80 dims
    g = torch.Generator()
    g.manual_seed(42)
    selected_dims = torch.randperm(d, generator=g)[:80]
    train_sub = train_data[:, selected_dims] # [N, 80]
    dim_d = 80

    # Model A: Single Global Gaussian
    mu_global = train_sub.mean(dim=0, keepdim=True)
    diff = train_sub - mu_global
    cov = torch.mm(diff.t(), diff) / (n_samples - 1)
    eps = 0.05
    cov_reg = cov + torch.eye(dim_d, device=device) * eps
    inv_cov_global = torch.linalg.pinv(cov_reg)

    # Model B: Multi-prototype / K-Means Mixture of Gaussians (K=8 clusters)
    # Run K-means on GPU
    k = 8
    # Random init
    init_idx = torch.randperm(n_samples)[:k]
    centroids = train_sub[init_idx].clone()
    for _ in range(15):
        dists = torch.cdist(train_sub, centroids) # [N, K]
        labels = dists.argmin(dim=1)
        for i in range(k):
            mask_i = (labels == i)
            if mask_i.any():
                centroids[i] = train_sub[mask_i].mean(dim=0)

    # For each cluster, fit local mean and inv_cov
    cluster_means = []
    cluster_inv_covs = []
    for i in range(k):
        mask_i = (labels == i)
        c_samples = train_sub[mask_i]
        c_mu = c_samples.mean(dim=0, keepdim=True)
        c_diff = c_samples - c_mu
        c_cov = torch.mm(c_diff.t(), c_diff) / max(1, c_samples.shape[0] - 1)
        c_inv = torch.linalg.pinv(c_cov + torch.eye(dim_d, device=device) * eps)
        cluster_means.append(c_mu)
        cluster_inv_covs.append(c_inv)

    # Evaluate on held-out damaged frame
    dmg_path = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/20261106/000000.png"
    gt_path = ROOT / ".cache/absolute-scratch-benchmark-20261106-01/000/20261106/000000.gt.png"
    dmg_img = cv2.imread(str(dmg_path))
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    h, w = dmg_img.shape[:2]

    rgb_dmg = cv2.cvtColor(dmg_img, cv2.COLOR_BGR2RGB)
    t_dmg = torch.from_numpy(rgb_dmg).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        d_feats = pe(t_dmg)
        d_sub = d_feats[:, selected_dims] # [1, 80, H1, W1]
        b, c, h1, w1 = d_sub.shape
        flat_q = d_sub.squeeze(0).permute(1, 2, 0).reshape(-1, dim_d)

        # Eval Model A: Single Global Gaussian
        diff_a = flat_q - mu_global
        temp_a = torch.mm(diff_a, inv_cov_global)
        mahal_a = torch.sqrt(torch.clamp((temp_a * diff_a).sum(dim=1), min=0.0)).reshape(1, 1, h1, w1)
        up_a = F.interpolate(mahal_a, size=(h, w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        # Eval Model B: Mixture of Gaussians (min Mahalanobis to closest cluster)
        cluster_mahals = []
        for i in range(k):
            diff_k = flat_q - cluster_means[i]
            temp_k = torch.mm(diff_k, cluster_inv_covs[i])
            m_k = torch.sqrt(torch.clamp((temp_k * diff_k).sum(dim=1), min=0.0))
            cluster_mahals.append(m_k.unsqueeze(1))
        all_km = torch.cat(cluster_mahals, dim=1) # [N_patches, K]
        min_km = all_km.min(dim=1).values.reshape(1, 1, h1, w1)
        up_b = F.interpolate(min_km, size=(h, w), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

    gt_defect = (gt_mask == 255)
    gt_intact = (gt_mask == 0)

    print("\n=== PaDiM Spatially-Invariant Models Comparison ===")
    for name, up_map in [("Model A: Single Global Gaussian", up_a), ("Model B: 8-Cluster GMM / Multi-Prototype Gaussian", up_b)]:
        d_mean = float(np.mean(up_map[gt_defect]))
        i_mean = float(np.mean(up_map[gt_intact]))
        i_p95 = float(np.percentile(up_map[gt_intact], 95))
        i_p99 = float(np.percentile(up_map[gt_intact], 99))
        print(f"\n{name}:")
        print(f"  Defect distance mean: {d_mean:.4f}")
        print(f"  Intact distance mean: {i_mean:.4f} (p95={i_p95:.4f}, p99={i_p99:.4f})")
        print(f"  Separation (Defect / Intact): {d_mean / i_mean:.2f}x")
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
    test_padim_formulations()
