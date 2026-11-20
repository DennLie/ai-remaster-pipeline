"""Comprehensive Profiling and Diagnostics for Neural Film Defect Detectors.

Measures all operational neural defect detectors in ARP:
1. Python executable and virtual environment resolution.
2. torch version, torch.version.cuda, torch.cuda.is_available(), parameter and tensor devices.
3. Real CUDA kernel execution verification using CUDA events and synchronize.
4. Silent CPU fallback detection.
5. Component-by-component timing on the actual 72-frame reference video:
   - Model initialization
   - Frame decoding
   - Preprocessing
   - CPU-to-GPU transfer
   - Forward inference
   - GPU-to-CPU transfer
   - Tiling (slice + merge)
   - Postprocessing
   - Subprocess launch overhead
6. Tile counts, forward passes per frame, batch size, effective FPS.
7. Subprocess / per-frame overhead analysis.
8. CPU-bound vs GPU-bound analysis.
9. GPU memory and concurrent workload inspection.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE_VIDEO = ROOT / "intermediate/stabilized/Neighbors_stabilized_eb46d58c.mkv"


def get_gpu_info() -> dict[str, Any]:
    """Inspect GPU hardware and active processes via nvidia-smi and torch."""
    cuda_avail = torch.cuda.is_available()
    info: dict[str, Any] = {
        "cuda_available": cuda_avail,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_avail else "None",
        "total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 2) if cuda_avail else 0,
        "allocated_memory_mb": round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2) if cuda_avail else 0,
        "reserved_memory_mb": round(torch.cuda.memory_reserved(0) / (1024 * 1024), 2) if cuda_avail else 0,
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if proc.returncode == 0:
            gpu_u, mem_u, mem_used, mem_free = [float(x.strip()) for x in proc.stdout.strip().split(",")]
            info["nvidia_smi"] = {
                "gpu_utilization_percent": gpu_u,
                "memory_utilization_percent": mem_u,
                "memory_used_mb": mem_used,
                "memory_free_mb": mem_free,
            }
    except Exception as exc:
        info["nvidia_smi_error"] = str(exc)
    return info


def verify_real_cuda_execution(model: torch.nn.Module, sample_tensor: torch.Tensor) -> dict[str, Any]:
    """Verify that actual CUDA kernels execute and record timing on GPU hardware."""
    device = next(model.parameters()).device
    if device.type != "cuda":
        return {
            "real_cuda_executed": False,
            "reason": f"Model is on {device.type}, not cuda",
            "kernel_time_ms": 0.0,
        }

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    input_on_gpu = sample_tensor.to(device)
    torch.cuda.synchronize()

    start_event.record()
    with torch.no_grad():
        out = model(input_on_gpu)
    end_event.record()
    torch.cuda.synchronize()

    kernel_time = start_event.elapsed_time(end_event)
    out_dev = "unknown"
    if isinstance(out, dict) and out:
        first_val = next(iter(out.values()))
        out_dev = str(first_val.device) if torch.is_tensor(first_val) else "dict"
    elif torch.is_tensor(out):
        out_dev = str(out.device)

    return {
        "real_cuda_executed": True,
        "parameter_device": str(device),
        "input_tensor_device": str(input_on_gpu.device),
        "kernel_time_ms": round(kernel_time, 3),
        "output_device": out_dev,
    }


def profile_bopb(frames_bgr: list[np.ndarray], device_str: str = "cuda") -> dict[str, Any]:
    """Deep profile of BOPB scratch detector."""
    sys.path.insert(0, str(ROOT / "tools/bopb"))
    from tools.bopb.models import load_bopb_model
    from tools.bopb.run_bopb_detector import predict_frame_tiled

    weights_path = ROOT / "tools/checkpoints/bopb/FT_Epoch_latest.pt"
    dev = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    # 1. Model init timing
    t0 = time.perf_counter()
    model = load_bopb_model(str(weights_path), device=dev)
    init_time_s = time.perf_counter() - t0

    # 2. Verify CUDA execution
    sample_patch = torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=dev)
    cuda_verif = verify_real_cuda_execution(model, sample_patch)

    # 3. Component timing across frames (sample 5 frames to get precise per-frame averages)
    test_frames = frames_bgr[:5]
    h, w = test_frames[0].shape[:2]

    # Preprocessing timing
    t_pre_list = []
    t_tile_slice_list = []
    t_transfer_h2d_list = []
    t_infer_list = []
    t_transfer_d2h_list = []
    t_tile_merge_list = []

    # Compute tiling steps
    tile_size = 256
    overlap = 64
    step = tile_size - overlap
    y_steps = list(range(0, max(1, h - tile_size + 1), step))
    if not y_steps or y_steps[-1] + tile_size < h:
        y_steps.append(max(0, h - tile_size))
    x_steps = list(range(0, max(1, w - tile_size + 1), step))
    if not x_steps or x_steps[-1] + tile_size < w:
        x_steps.append(max(0, w - tile_size))
    y_steps = sorted(list(set(y_steps)))
    x_steps = sorted(list(set(x_steps)))
    total_tiles_per_frame = len(y_steps) * len(x_steps)

    for frame in test_frames:
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t_pre_list.append(time.perf_counter() - t0)

        # Measure slice, H2D, infer, D2H, merge for each tile
        t_slice_total = 0.0
        t_h2d_total = 0.0
        t_infer_total = 0.0
        t_d2h_total = 0.0
        t_merge_total = 0.0

        prob_accum = np.zeros((h, w), dtype=np.float32)
        weight_accum = np.zeros((h, w), dtype=np.float32)
        win_y = np.hanning(tile_size + 2)[1:-1]
        win_x = np.hanning(tile_size + 2)[1:-1]
        tile_weight = (win_y[:, None] * win_x[None, :]).astype(np.float32)

        for y in y_steps:
            for x in x_steps:
                t_s0 = time.perf_counter()
                patch = gray[y : y + tile_size, x : x + tile_size]
                t_slice_total += time.perf_counter() - t_s0

                t_h0 = time.perf_counter()
                tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
                tensor = (tensor / 255.0 - 0.5) / 0.5
                tensor = tensor.to(dev)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_h2d_total += time.perf_counter() - t_h0

                t_inf0 = time.perf_counter()
                with torch.no_grad():
                    out = torch.sigmoid(model(tensor))
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                t_infer_total += time.perf_counter() - t_inf0

                t_d0 = time.perf_counter()
                patch_prob = out.squeeze().cpu().numpy()
                t_d2h_total += time.perf_counter() - t_d0

                t_m0 = time.perf_counter()
                cur_w = tile_weight[:patch.shape[0], :patch.shape[1]]
                prob_accum[y : y + patch.shape[0], x : x + patch.shape[1]] += patch_prob * cur_w
                weight_accum[y : y + patch.shape[0], x : x + patch.shape[1]] += cur_w
                t_merge_total += time.perf_counter() - t_m0

        t_tile_slice_list.append(t_slice_total)
        t_transfer_h2d_list.append(t_h2d_total)
        t_infer_list.append(t_infer_total)
        t_transfer_d2h_list.append(t_d2h_total)
        t_tile_merge_list.append(t_merge_total)

    # Postprocessing
    t_post_list = []
    for frame in test_frames:
        t0 = time.perf_counter()
        binary = np.where(prob_accum >= 0.5, np.uint8(255), np.uint8(0))
        t_post_list.append(time.perf_counter() - t0)

    # Subprocess launch overhead check
    py_venv = ROOT / "tools/bopb/.venv/Scripts/python.exe"
    py_root = ROOT / ".venv/Scripts/python.exe"
    py_sys = Path(sys.executable)

    def measure_subproc_startup(py_path: Path) -> float:
        if not py_path.is_file():
            return -1.0
        t0 = time.perf_counter()
        subprocess.run([str(py_path), "-c", "import torch; torch.cuda.is_available()"], capture_output=True, timeout=15)
        return time.perf_counter() - t0

    subproc_venv_time = measure_subproc_startup(py_venv)
    subproc_root_time = measure_subproc_startup(py_root)

    return {
        "detector": "bopb",
        "weights_path": str(weights_path),
        "device": str(dev),
        "model_init_time_s": round(init_time_s, 4),
        "cuda_verification": cuda_verif,
        "tiles_per_frame": total_tiles_per_frame,
        "tile_size": tile_size,
        "overlap": overlap,
        "batch_size": 1,
        "forward_passes_per_frame": total_tiles_per_frame,
        "avg_timings_ms": {
            "preprocessing": round(float(np.mean(t_pre_list)) * 1000, 2),
            "tile_slicing": round(float(np.mean(t_tile_slice_list)) * 1000, 2),
            "cpu_to_gpu_transfer": round(float(np.mean(t_transfer_h2d_list)) * 1000, 2),
            "gpu_inference": round(float(np.mean(t_infer_list)) * 1000, 2),
            "gpu_to_cpu_transfer": round(float(np.mean(t_transfer_d2h_list)) * 1000, 2),
            "tile_merging": round(float(np.mean(t_tile_merge_list)) * 1000, 2),
            "postprocessing": round(float(np.mean(t_post_list)) * 1000, 2),
            "total_in_process_frame_ms": round(
                (float(np.mean(t_pre_list)) + float(np.mean(t_tile_slice_list)) +
                 float(np.mean(t_transfer_h2d_list)) + float(np.mean(t_infer_list)) +
                 float(np.mean(t_transfer_d2h_list)) + float(np.mean(t_tile_merge_list)) +
                 float(np.mean(t_post_list))) * 1000, 2
            ),
        },
        "subprocess_overhead": {
            "tools_bopb_venv_startup_s": round(subproc_venv_time, 3),
            "root_venv_startup_s": round(subproc_root_time, 3),
        },
    }


def profile_neural_models(frames_bgr: list[np.ndarray], model_name: str) -> dict[str, Any]:
    """Profile FilmArtefactSegmenter or CNNBlotches."""
    sys.path.insert(0, str(ROOT / "tools/neural_defects"))
    from tools.neural_defects.models import load_neural_model
    from tools.neural_defects.run_neural_detector import run_tiled_inference

    weights_file = "film_artefact_segmenter.pt" if model_name == "film_artefact_segmenter" else "cnn_blotches.pt"
    weights_path = ROOT / f"tools/checkpoints/{model_name}/{weights_file}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    model = load_neural_model(str(weights_path), model_name=model_name, device=dev)
    init_time_s = time.perf_counter() - t0

    sample_patch = torch.zeros((1, 3, 256, 256), dtype=torch.float32, device=dev)
    cuda_verif = verify_real_cuda_execution(model, sample_patch)

    test_frames = frames_bgr[:5]
    h, w = test_frames[0].shape[:2]

    # Tile calculations
    tile_size = 256
    overlap = 64
    step = tile_size - overlap
    y_coords = list(range(0, max(1, h - tile_size + 1), step))
    if y_coords[-1] + tile_size < h:
        y_coords.append(h - tile_size)
    x_coords = list(range(0, max(1, w - tile_size + 1), step))
    if x_coords[-1] + tile_size < w:
        x_coords.append(w - tile_size)
    total_tiles = len(y_coords) * len(x_coords)

    t_frame_list = []
    for frame in test_frames:
        t0 = time.perf_counter()
        _ = run_tiled_inference(model, frame, tile_size=tile_size, overlap=overlap, device=dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_frame_list.append(time.perf_counter() - t0)

    return {
        "detector": model_name,
        "weights_path": str(weights_path),
        "device": str(dev),
        "model_init_time_s": round(init_time_s, 4),
        "cuda_verification": cuda_verif,
        "tiles_per_frame": total_tiles,
        "tile_size": tile_size,
        "overlap": overlap,
        "batch_size": 1,
        "forward_passes_per_frame": total_tiles,
        "avg_in_process_frame_ms": round(float(np.mean(t_frame_list)) * 1000, 2),
        "in_process_fps": round(1.0 / float(np.mean(t_frame_list)), 2),
    }


def main():
    print("=" * 70)
    print("Neural Defect Detector Profiling & Hardware Diagnostics")
    print("=" * 70)

    # 1. GPU hardware diagnostics
    gpu_info = get_gpu_info()
    print("\n[1] GPU Environment:")
    print(json.dumps(gpu_info, indent=2))

    # 2. Python runtime inspection
    print("\n[2] Python Interpreter & Virtual Environment Diagnostics:")
    interpreters = {
        "current_process": Path(sys.executable),
        "root_venv": ROOT / ".venv/Scripts/python.exe",
        "tools_bopb_venv": ROOT / "tools/bopb/.venv/Scripts/python.exe",
        "tools_dart_venv": ROOT / "tools/dart/.venv/Scripts/python.exe",
    }
    env_results = {}
    for name, py_path in interpreters.items():
        if py_path.is_file():
            res = subprocess.run(
                [str(py_path), "-c", "import sys, torch; print(f'{sys.executable}|{torch.__version__}|{getattr(torch.version, \"cuda\", None)}|{torch.cuda.is_available()}')"],
                capture_output=True, text=True, timeout=10
            )
            if res.returncode == 0:
                parts = res.stdout.strip().split("|")
                env_results[name] = {
                    "path": parts[0],
                    "torch_version": parts[1],
                    "torch_cuda": parts[2],
                    "cuda_available": parts[3] == "True",
                }
            else:
                env_results[name] = {"path": str(py_path), "error": res.stderr.strip()}
        else:
            env_results[name] = {"path": str(py_path), "exists": False}
    print(json.dumps(env_results, indent=2))

    # 3. Read test video frames
    print("\n[3] Reading test video frames from:", SOURCE_VIDEO)
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(SOURCE_VIDEO))
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    decode_time = time.perf_counter() - t0
    n_frames = len(frames)
    h, w = frames[0].shape[:2] if frames else (0, 0)
    print(f"Decoded {n_frames} frames ({w}x{h}) in {decode_time:.3f} s ({round(n_frames/decode_time, 1)} FPS)")

    # 4. Profile BOPB
    print("\n[4] Profiling BOPB Scratch Detector...")
    bopb_prof = profile_bopb(frames, device_str="cuda")
    print(json.dumps(bopb_prof, indent=2))

    # 5. Profile FilmArtefactSegmenter
    print("\n[5] Profiling Film Artefact Segmenter...")
    fas_prof = profile_neural_models(frames, "film_artefact_segmenter")
    print(json.dumps(fas_prof, indent=2))

    # 6. Profile CNN Blotches
    print("\n[6] Profiling CNN Blotches & Scratches...")
    cnn_prof = profile_neural_models(frames, "cnn_blotches")
    print(json.dumps(cnn_prof, indent=2))

    # 7. Summary
    report = {
        "gpu": gpu_info,
        "interpreters": env_results,
        "video": {
            "path": str(SOURCE_VIDEO),
            "frames": n_frames,
            "width": w,
            "height": h,
            "decode_time_s": round(decode_time, 3),
            "decode_fps": round(n_frames / decode_time, 1),
        },
        "detectors": {
            "bopb": bopb_prof,
            "film_artefact_segmenter": fas_prof,
            "cnn_blotches": cnn_prof,
        },
    }

    out_file = ROOT / "intermediate/neural_detector_profile.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote full profiling report to: {out_file}")


if __name__ == "__main__":
    main()
