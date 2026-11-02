#!/usr/bin/env python3
"""Isolated Restormer single-image deblur research on ORIGINAL video frames.

This is NOT a commercial or production ARP backend. Verifies a clean,
pinned external repo and separately supplied checkpoint SHA-256 before
importing Restormer code or PyTorch. Never downloads files itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
import shutil
import subprocess
import tempfile
from pathlib import Path

REVISION = "68dc6ac472db26f16361150cb7a96a1bc87da93f"
UPSTREAM = "https://github.com/swz30/Restormer"
ARCH_PATH = "basicsr/models/archs/restormer_arch.py"
MAX_FRAMES = 48
MAX_RAW_BYTES = 512 * 1024 * 1024


def command_output(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def verify_checkout(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"External Restormer checkout missing: {path}")
    head = command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
    if head != REVISION:
        raise ValueError(f"Unexpected Restormer commit {head}; require {REVISION}")
    changes = command_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"])
    if changes:
        raise ValueError("Restormer checkout is dirty or contains untracked files")
    for name in ("LICENSE.md", ARCH_PATH):
        if not (path / name).is_file():
            raise ValueError(f"Pinned Restormer checkout missing {name}")


def verify_weights(path: Path, expected_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256.strip().lower()):
        raise ValueError("Full 64-character SHA-256 is required for Restormer weights")
    if not path.is_file():
        raise FileNotFoundError(f"Missing externally reviewed checkpoint: {path}")
    if not 0 < path.stat().st_size <= 1_000_000_000:
        raise ValueError("Checkpoint has invalid size")
    actual = sha256_file(path)
    if actual != expected_sha256.strip().lower():
        raise ValueError(f"Restormer checkpoint SHA-256 mismatch: {actual}")
    return actual


def probe(ffprobe: str, video: Path) -> tuple[int, int, str]:
    raw = command_output([ffprobe, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate",
                          "-of", "json", str(video)])
    streams = json.loads(raw).get("streams", [])
    if not streams:
        raise ValueError("No video stream")
    x = streams[0]
    w, h, fps = int(x["width"]), int(x["height"]), str(x["avg_frame_rate"])
    num, sep, den = fps.partition("/")
    if w < 32 or h < 32 or not sep or int(num) <= 0 or int(den) <= 0:
        raise ValueError("Invalid video geometry or cadence")
    if x.get("r_frame_rate") != fps:
        raise ValueError("Variable frame rate not supported by research decoder")
    return w, h, fps


def validate(a: argparse.Namespace) -> dict:
    if not a.research_only:
        raise ValueError("Restormer research adapter requires --research-only")
    source, output = a.source.resolve(), a.output.resolve()
    if source == output:
        raise ValueError("Refusing to overwrite original video")
    if not source.is_file():
        raise FileNotFoundError(f"Missing source: {source}")
    if output.exists():
        raise FileExistsError(f"Research output already exists: {output}")
    if output.suffix.lower() != ".mkv":
        raise ValueError("Research output must have .mkv suffix")
    if not 1 <= a.frames <= MAX_FRAMES:
        raise ValueError(f"Frames must be between 1 and {MAX_FRAMES}")
    if not math.isfinite(a.start_seconds) or not 0 <= a.start_seconds <= 86400:
        raise ValueError("Start time must be finite between 0 and 86400 seconds")
    if not isinstance(a.mix, int) or not 0 <= a.mix <= 100:
        raise ValueError("Mix must be between 0 and 100")
    verify_checkout(a.checkout.resolve())
    model_sha = verify_weights(a.weights.resolve(), a.weights_sha256)
    ffmpeg = shutil.which(a.ffmpeg) if a.ffmpeg else shutil.which("ffmpeg")
    ffprobe = shutil.which(a.ffprobe) if a.ffprobe else shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("FFmpeg/FFprobe required")
    width, height, fps = probe(ffprobe, source)
    if a.frames * width * height * 3 > MAX_RAW_BYTES:
        raise ValueError("Research sample too large in decoded RGB8 memory")
    return {
        "tool": "run_restormer_deblur_research.py",
        "upstream": UPSTREAM, "revision": REVISION,
        "task": "Restormer single-image motion deblurring (GoPro-trained)",
        "source": str(source), "source_sha256": sha256_file(source),
        "output": str(output), "checkout": str(a.checkout.resolve()),
        "weights_sha256": model_sha, "frames": a.frames,
        "start_seconds": a.start_seconds, "width": width,
        "height": height, "fps": fps, "mix": a.mix,
        "device": a.device, "research_only": True,
        "commercial_cleared": False, "audio": "omitted from research sample",
        "warning": "Framewise Restormer can alter film grain and flicker; no historical fidelity guarantee.",
        "_ffmpeg": ffmpeg, "_ffprobe": ffprobe,
    }


def read_exact(stream, count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < count:
        packet = stream.read(count - len(chunks))
        if not packet:
            break
        chunks.extend(packet)
    return bytes(chunks)


def decode_frames(ffmpeg: str, source: Path, start: float, count: int, width: int, height: int):
    import numpy as np

    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-ss", f"{start:.6f}", "-i", str(source),
           "-map", "0:v:0", "-an", "-sn", "-dn",
           "-frames:v", str(count), "-fps_mode", "passthrough",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames = []
    try:
        assert p.stdout is not None
        for i in range(count):
            raw = read_exact(p.stdout, width * height * 3)
            if len(raw) != width * height * 3:
                raise ValueError(f"Video ended before requested frame {i+1}/{count}")
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3).copy())
        p.stdout.close()
        if p.wait(timeout=45):
            raise RuntimeError("FFmpeg decode failure")
        return frames
    finally:
        if p.poll() is None:
            p.kill()
            p.communicate()


def load_model(checkout: Path, weights: Path, device: str):
    import torch
    # Import only after strict revision, clean tree and weight-hash checks.
    arch = runpy.run_path(str(checkout / ARCH_PATH))
    kwargs = {
        "inp_channels": 3, "out_channels": 3, "dim": 48,
        "num_blocks": [4, 6, 6, 8], "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8], "ffn_expansion_factor": 2.66,
        "bias": False, "LayerNorm_type": "WithBias",
        "dual_pixel_task": False,
    }
    model = arch["Restormer"](**kwargs)
    loaded = torch.load(str(weights), map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError("Restormer checkpoint must contain a dict")
    params = loaded.get("params")
    if not isinstance(params, dict) or not params:
        raise ValueError("Expected motion-deblurring checkpoint with 'params' tensor map")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in params.items()):
        raise ValueError("Invalid/non-tensor checkpoint state")
    model.load_state_dict(params, strict=True)
    return model.eval().to(device)


def deblur_frame(model, rgb, *, mix: int, device: str):
    import numpy as np
    import torch
    import torch.nn.functional as F
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError("Restormer input must be RGB8")
    h, w = arr.shape[:2]
    x = torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0).to(
        device=device, dtype=torch.float32) / 255.
    # The official demo reflects borders to the next multiple of 8.
    pad_h, pad_w = (-h) % 8, (-w) % 8
    padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x
    with torch.inference_mode():
        out = model(padded).clamp_(0., 1.)[:, :, :h, :w]
        if tuple(out.shape) != (1, 3, h, w):
            raise ValueError("Model output changed source geometry")
        if mix != 100:
            out = out * (mix / 100) + x * (1 - mix / 100)
        return out.squeeze(0).permute(1, 2, 0).mul(255).round().to(torch.uint8).cpu().numpy()


def encode_frames(ffmpeg: str, frames, target: Path, fps: str) -> None:
    import cv2
    import numpy as np

    partial = target.with_name(target.stem + ".partial.mkv")
    partial.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="arp-restormer-") as tmp:
        folder = Path(tmp)
        try:
            for i, frame in enumerate(frames):
                if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
                    raise ValueError("Invalid encoded video frame")
                if not cv2.imwrite(str(folder / f"{i:06d}.png"),
                                   cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise IOError(f"Could not save lossless frame {i}")
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-framerate", fps, "-start_number", "0",
                 "-i", str(folder / "%06d.png"), "-an",
                 "-c:v", "ffv1", "-level", "3",
                 "-pix_fmt", "bgr0", "-f", "matroska", str(partial)],
                capture_output=True, check=True, timeout=180,
            )
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--weights-sha256", required=True)
    p.add_argument("--research-only", action="store_true")
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--start-seconds", type=float, default=0.)
    p.add_argument("--mix", type=int, default=100)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    report = validate(args)
    ffmpeg, ffprobe = report.pop("_ffmpeg"), report.pop("_ffprobe")
    if args.dry_run:
        report["dry_run"] = True
        print(json.dumps(report, indent=2))
        return
    import torch  # Not imported in --dry-run or CPU-only safety tests.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; no silent fallback")
    model = load_model(args.checkout.resolve(), args.weights.resolve(), args.device)
    frames = decode_frames(ffmpeg, args.source.resolve(), args.start_seconds, args.frames,
                           report["width"], report["height"])
    restored = [deblur_frame(model, frame, mix=args.mix, device=args.device) for frame in frames]
    target = args.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encode_frames(ffmpeg, restored, target, report["fps"])
    if probe(ffprobe, target) != (report["width"], report["height"], report["fps"]):
        target.unlink(missing_ok=True)
        raise ValueError("Encoded research artifact failed geometry/cadence check")
    report["result_sha256"] = sha256_file(target)
    report["completed_frames"] = len(restored)
    target.with_suffix(".research.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
