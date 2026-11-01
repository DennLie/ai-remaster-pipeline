#!/usr/bin/env python3
"""Research-only, opt-in FastDVDnet denoising of real archival video frames.

This is deliberately separate from ARP's production GUI and commercial
allowlist. It imports no external neural code until the exact FastDVDnet
Git revision and user-supplied checkpoint SHA-256 have been verified.
It NEVER adds synthetic noise to input frames (unlike upstream's
test_fastdvdnet.py which benchmarks synthetic Gaussian noise).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

UPSTREAM = "https://github.com/m-tassano/fastdvdnet"
PINNED_COMMIT = "c8fdf6182a0340e89dd18f5df25b47337cbede6f"
MODEL_FILES = ("models.py", "fastdvdnet.py", "LICENSE")
MAX_CHECKPOINT_BYTES = 1_000_000_000
MAX_RESEARCH_FRAMES = 120


def run_text(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkout(checkout: Path) -> None:
    """Fail closed on wrong revision or locally modified/importable code."""
    if not checkout.is_dir():
        raise FileNotFoundError(f"Missing external checkout: {checkout}")
    revision = run_text(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if revision.lower() != PINNED_COMMIT:
        raise ValueError(f"Wrong FastDVDnet commit {revision}; require {PINNED_COMMIT}")
    changes = run_text(["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=normal"])
    if changes:
        raise ValueError("External checkout has tracked/untracked modifications: use a clean pinned clone")
    for name in MODEL_FILES:
        if not (checkout / name).is_file():
            raise ValueError(f"Pinned FastDVDnet checkout missing {name}")
    # Pinned checkout + clean tree is necessary, not a license grant.


def verify_checkpoint(checkpoint: Path, expected_sha256: str) -> str:
    """A mandatory user-supplied hash is checked before importing torch."""
    expected = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Supply the full 64-character SHA-256 of your checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint {checkpoint}")
    size = checkpoint.stat().st_size
    if not 0 < size <= MAX_CHECKPOINT_BYTES:
        raise ValueError(f"Checkpoint file size is invalid or over {MAX_CHECKPOINT_BYTES} bytes")
    actual = sha256_file(checkpoint)
    if actual != expected:
        raise ValueError(f"Checkpoint SHA-256 mismatch: actual {actual}, expected {expected}")
    return actual


def decode_info(ffprobe: str, source: Path) -> tuple[int, int, str]:
    raw = run_text([ffprobe, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,avg_frame_rate",
                    "-of", "json", str(source)])
    streams = json.loads(raw).get("streams", [])
    if not streams:
        raise ValueError("Source has no decodable video stream")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    fps = stream["avg_frame_rate"]
    number, separator, denominator = fps.partition("/")
    if not separator or int(number) <= 0 or int(denominator) <= 0:
        raise ValueError(f"Source has unsupported nonconstant/invalid frame rate: {fps}")
    if min(width, height) < 4 or max(width, height) > 8192:
        raise ValueError("Unsupported video dimensions")
    return width, height, fps


def reflect_index(index: int, length: int) -> int:
    """Reflect temporal neighbours at both ends; length=1 is permitted."""
    if length <= 0:
        raise ValueError("Empty input video")
    if length == 1:
        return 0
    period = 2 * length - 2
    k = index % period
    return k if k < length else period - k


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            break
        chunks.extend(part)
    return bytes(chunks)


def decode_frames(ffmpeg: str, source: Path, *, start_seconds: float,
                  count: int, width: int, height: int):
    """Yield exactly count native-size 8-bit RGB frames from a short source clip."""
    # Lazy import: import of this script requires neither OpenCV nor PyTorch.
    import numpy as np
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
               "-ss", f"{start_seconds:.6f}", "-i", str(source),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-frames:v", str(count),
               "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    bytes_per_frame = width * height * 3
    frames = []
    try:
        assert process.stdout is not None
        for index in range(count):
            payload = read_exact(process.stdout, bytes_per_frame)
            if len(payload) != bytes_per_frame:
                raise ValueError(f"Source ended during frame {index}; requested {count} frames")
            frames.append(np.frombuffer(payload, np.uint8).reshape(height, width, 3).copy())
        process.stdout.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        status = process.wait(timeout=30)
        if status:
            raise RuntimeError(f"FFmpeg decoding failed: {stderr[-1400:]}")
        return frames
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def load_model(checkout: Path, weights: Path, device: str):
    import torch
    # Loading the external model definition executes external Python code.
    # The caller has verified a clean, pinned checkout and opted in explicitly.
    spec = importlib.util.spec_from_file_location("_arp_fastdvdnet_pinned_models", checkout / "models.py")
    if spec is None or spec.loader is None:
        raise ImportError("Could not load verified external FastDVDnet models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # weights_only=True is mandatory; never fall back to unsafe pickle loading.
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a non-empty tensor state dictionary")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Checkpoint contains non-tensor objects")
    if all(k.startswith("module.") for k in state):
        state = {k.removeprefix("module."): v for k, v in state.items()}
    model = module.FastDVDnet(num_input_frames=5)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def denoise_one(model, frames, index: int, sigma: int, device: str):
    """Feed the ORIGINAL RGB frames, not a synthetically degraded benchmark."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    count, height, width = len(frames), *frames[0].shape[:2]
    indices = [reflect_index(index + relative, count) for relative in (-2, -1, 0, 1, 2)]
    sample = np.concatenate([frames[k].transpose(2, 0, 1) for k in indices], axis=0)
    x = torch.from_numpy(sample.copy()).unsqueeze(0).to(device=device, dtype=torch.float32) / 255.
    noise_map = torch.full((1, 1, height, width), sigma / 255., dtype=torch.float32, device=device)
    # Reproduce upstream temp_denoise's reflect padding for dimensions not divisible by 4.
    pad_h = (-height) % 4
    pad_w = (-width) % 4
    if pad_h or pad_w:
        if height <= pad_h or width <= pad_w:
            raise ValueError("Input too small for reflect padding")
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        noise_map = F.pad(noise_map, (0, pad_w, 0, pad_h), mode="reflect")
    with torch.inference_mode():
        predicted = torch.clamp(model(x, noise_map), 0., 1.)
    if pad_h:
        predicted = predicted[:, :, :-pad_h, :]
    if pad_w:
        predicted = predicted[:, :, :, :-pad_w]
    if tuple(predicted.shape) != (1, 3, height, width):
        raise ValueError(f"Unexpected model output shape {tuple(predicted.shape)}")
    rgb = predicted.squeeze(0).permute(1, 2, 0).mul(255.).round().to(torch.uint8).cpu().numpy()
    return rgb


def encode_research_output(ffmpeg: str, target: Path, frames, model,
                           sigma: int, device: str, width: int, height: int, fps: str):
    tmp = target.with_name(target.stem + ".partial.mkv")
    tmp.unlink(missing_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
           "-r", fps, "-i", "pipe:0", "-map", "0:v:0", "-an",
           "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0", "-f", "matroska", str(tmp)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for index in range(len(frames)):
            corrected = denoise_one(model, frames, index, sigma, device)
            process.stdin.write(corrected.tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        status = process.wait(timeout=180)
        if status:
            raise RuntimeError(f"FFmpeg video encoding failed: {stderr[-1400:]}")
        tmp.replace(target)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        tmp.unlink(missing_ok=True)


def inspect(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    target = args.output.resolve()
    if not args.research_only:
        raise ValueError("Research adapter disabled by default; pass --research-only explicitly")
    if source == target:
        raise ValueError("Research output may not overwrite the original source")
    if not source.is_file():
        raise FileNotFoundError(f"Missing source {source}")
    if target.suffix.lower() != ".mkv":
        raise ValueError("Research output must be a new .mkv file")
    if not 1 <= args.frames <= MAX_RESEARCH_FRAMES:
        raise ValueError(f"Frames must be between 1 and {MAX_RESEARCH_FRAMES}")
    if not 5 <= args.sigma <= 55:
        raise ValueError("Conditioning noise sigma (8-bit units) must be 5..55")
    if not math.isfinite(args.start_seconds) or not 0 <= args.start_seconds <= 86400:
        raise ValueError("Source start time must be between 0 and 86400 seconds")
    verify_checkout(args.checkout.resolve())
    model_sha = verify_checkpoint(args.weights.resolve(), args.weights_sha256)
    ffmpeg = shutil.which(args.ffmpeg) if args.ffmpeg else shutil.which("ffmpeg")
    ffprobe = shutil.which(args.ffprobe) if args.ffprobe else shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("Both FFmpeg and FFprobe are required")
    width, height, fps = decode_info(ffprobe, source)
    return {
        "model": "FastDVDnet", "upstream": UPSTREAM,
        "external_checkout": str(args.checkout.resolve()), "revision": PINNED_COMMIT,
        "checkpoint_sha256": model_sha, "source": str(source), "source_sha256": sha256_file(source),
        "output": str(target), "frames_requested": args.frames,
        "start_seconds": args.start_seconds, "fps": fps,
        "width": width, "height": height, "noise_sigma_8bit": args.sigma,
        "processing": "original RGB8 frames without synthetic noise; five-frame reflected window",
        "research_only": True, "commercial_clearance": False,
        "audio": "omitted intentionally for benchmark clip",
        "limitations": "Research-quality 8-bit RGB, short constant-cadence excerpt. No claim of film-grain preservation.",
        "_ffmpeg": ffmpeg, "_ffprobe": ffprobe,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True, help="Manually prepared pinned external FastDVDnet Git checkout")
    p.add_argument("--weights", type=Path, required=True, help="Manually provided checkpoint; never downloaded automatically")
    p.add_argument("--weights-sha256", required=True, help="Expected SHA-256 recorded by the researcher")
    p.add_argument("--research-only", action="store_true", help="Explicit opt-in; not commercial-cleared")
    p.add_argument("--frames", type=int, default=24, help=f"Maximum {MAX_RESEARCH_FRAMES} frames / run")
    p.add_argument("--start-seconds", type=float, default=0.0)
    p.add_argument("--sigma", type=int, default=10, help="FastDVDnet noise conditioning in 8-bit units, 5–55")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")
    p.add_argument("--dry-run", action="store_true", help="Verify inputs and provenance only; never import torch or load weights")
    args = p.parse_args()
    report = inspect(args)
    ffmpeg, ffprobe = report.pop("_ffmpeg"), report.pop("_ffprobe")
    if args.dry_run:
        report["dry_run"] = True
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    if args.output.exists():
        raise FileExistsError(f"Research output already exists: {args.output}; choose another filename")
    import torch  # optional, isolated; required only for a real research run
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no silent CPU fallback")
    model = load_model(args.checkout.resolve(), args.weights.resolve(), args.device)
    frames = decode_frames(ffmpeg, args.source.resolve(), start_seconds=args.start_seconds,
                           count=args.frames, width=report["width"], height=report["height"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_research_output(ffmpeg, args.output.resolve(), frames, model,
                           args.sigma, args.device, report["width"], report["height"], report["fps"])
    # Confirm physical output and cadence before writing research metadata.
    w2, h2, fps2 = decode_info(ffprobe, args.output.resolve())
    if (w2, h2, fps2) != (report["width"], report["height"], report["fps"]):
        args.output.unlink(missing_ok=True)
        raise ValueError("Output dimensions or cadence differ from the original")
    report["actual_frames"] = len(frames)
    report["device"] = args.device
    report["output_sha256"] = sha256_file(args.output)
    args.output.with_suffix(".research.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
