import sys
from pathlib import Path
import tempfile
import subprocess
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import candidate_detector_eval as cde

dart_py = cde.find_isolated_dart_python()
weights = Path("I:/DART/pretrained_models/dart.safetensors")

def run_dart_clip(frames):
    with tempfile.TemporaryDirectory(prefix="dart-neg-") as tmp:
        tpath = Path(tmp)
        in_dir = tpath / "in"
        out_dir = tpath / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        for i, f in enumerate(frames):
            cv2.imwrite(str(in_dir / f"frame_{i:06d}.png"), f)
        cmd = [
            str(dart_py),
            r"I:\DART\scripts\inference.py",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--weights", str(weights),
            "--save-masks",
            "--device", "cuda",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        masks = []
        for i in range(len(frames)):
            mf = out_dir / "masks" / f"frame_{i:06d}.png"
            masks.append(cv2.imread(str(mf), cv2.IMREAD_UNCHANGED))
        return masks

h, w = 256, 256
n_frames = 6
total_px = h * w * n_frames

# 1. Clean Flat Gray
flat_frames = [np.full((h, w, 3), 128, dtype=np.uint8) for _ in range(n_frames)]

# 2. Film Grain (sigma=15)
rng = np.random.default_rng(42)
grain_frames = []
for _ in range(n_frames):
    noise = rng.normal(0, 15, (h, w, 3))
    frame = np.clip(128.0 + noise, 0, 255).astype(np.uint8)
    grain_frames.append(frame)

# 3. High-Contrast Static Edges (3 vertical bars: x=64, 128, 192)
edge_frames = []
for _ in range(n_frames):
    f = np.full((h, w, 3), 60, dtype=np.uint8)
    for bx in (64, 128, 192):
        f[:, bx - 4 : bx + 4] = 220
    edge_frames.append(f)

# 4. Portrait Face (oval with skin tone)
face_frames = []
for _ in range(n_frames):
    f = np.full((h, w, 3), 50, dtype=np.uint8)
    # Draw oval face tone
    cv2.ellipse(f, (128, 128), (50, 75), 0, 0, 360, (140, 170, 210), -1) # BGR
    # Add soft gradient / features
    cv2.circle(f, (110, 110), 8, (60, 80, 120), -1)
    cv2.circle(f, (146, 110), 8, (60, 80, 120), -1)
    cv2.ellipse(f, (128, 150), (16, 8), 0, 0, 360, (90, 100, 160), -1)
    face_frames.append(f)

# 5. Moving Objects (translating vertical bar dx=20 px/frame)
move_frames = []
for t in range(n_frames):
    f = np.full((h, w, 3), 80, dtype=np.uint8)
    bx = 30 + t * 20
    f[:, bx : bx + 16] = 200
    move_frames.append(f)

regimes = [
    ("Clean Flat Gray", flat_frames),
    ("Film Grain", grain_frames),
    ("High-Contrast Edges", edge_frames),
    ("Portrait Face", face_frames),
    ("Moving Objects", move_frames),
]

print(f"{'Regime':22s} | {'FP (>160)':10s} | {'FPR':8s} | {'Soft Mean':9s}")
print("-" * 56)
for name, frames in regimes:
    masks = run_dart_clip(frames)
    fp_count = sum(int(np.count_nonzero(m >= 160)) for m in masks)
    fpr = fp_count / total_px * 100
    soft_mean = np.mean([np.mean(m) for m in masks])
    print(f"{name:22s} | {fp_count:10d} | {fpr:7.3f}% | {soft_mean:8.1f}/255")
