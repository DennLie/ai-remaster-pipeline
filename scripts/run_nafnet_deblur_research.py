#!/usr/bin/env python3
"""Run official pinned NAFNet GoPro deblurring on *real* film frames.

RESEARCH ONLY. Separate from the commercial ARP workflow and default GUI.
Pinned external code, local user-provided weights, mandatory SHA-256.
No auto-fetch, no `torch.load` pickle fallback, no original overwrites.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UPSTREAM = "https://github.com/megvii-research/NAFNet"
REVISION = "2b4af71ebe098a92a75910c233a3965a3e93ede4"
MAX_FRAMES = 72
MAX_TOTAL_RAW_BYTES = 512 * 1024 * 1024
WIDTHS = (32, 64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while part := stream.read(1024 * 1024):
            digest.update(part)
    return digest.hexdigest()


def command_text(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def verify_code_checkout(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing NAFNet checkout: {path}")
    rev = command_text(["git", "-C", str(path), "rev-parse", "HEAD"])
    if rev != REVISION:
        raise ValueError(f"Wrong NAFNet commit {rev}; expected {REVISION}")
    dirty = command_text(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"])
    if dirty:
        raise ValueError("NAFNet checkout is modified or includes untracked files")
    for name in ("LICENSE", "basicsr/models/archs/NAFNet_arch.py",
                 "basicsr/models/archs/local_arch.py"):
        if not (path / name).is_file():
            raise ValueError(f"Missing external source file {name}")


def verify_weights(path: Path, expected_hash: str) -> str:
    expected = expected_hash.lower().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("The complete 64-character weight SHA-256 is required")
    if not path.is_file():
        raise FileNotFoundError(f"Missing NAFNet weights {path}")
    if not (0 < path.stat().st_size <= 1_000_000_000):
        raise ValueError("NAFNet checkpoint too large or empty")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"NAFNet weight SHA-256 mismatch: actual {actual}")
    return actual


def probe(ffprobe: str, path: Path) -> tuple[int, int, str]:
    raw = command_text([ffprobe, "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,avg_frame_rate",
                        "-of", "json", str(path)])
    streams = json.loads(raw).get("streams") or []
    if not streams:
        raise ValueError(f"Missing video stream in {path}")
    info = streams[0]
    w, h, fps = int(info["width"]), int(info["height"]), str(info["avg_frame_rate"])
    num, sep, den = fps.partition("/")
    if w <= 0 or h <= 0 or not sep or int(num) <= 0 or int(den) <= 0:
        raise ValueError(f"Invalid video metadata {w}x{h} fps={fps}")
    return w, h, fps


def validate_inputs(a: argparse.Namespace) -> dict:
    if not a.research_only:
        raise ValueError("Research-only adapter requires explicit --research-only flag")
    source, output = a.source.resolve(), a.output.resolve()
    if source == output:
        raise ValueError("Output cannot overwrite original source")
    if not source.is_file():
        raise FileNotFoundError(f"Missing video {source}")
    if output.exists():
        raise FileExistsError(f"Research output exists already: {output}")
    if output.suffix.lower() != ".mkv":
        raise ValueError("Output must be a separate MKV research artifact")
    if a.width not in WIDTHS:
        raise ValueError(f"Only NAFNet GoPro widths {WIDTHS} supported")
    if not (1 <= a.frames <= MAX_FRAMES):
        raise ValueError(f"Frames must be between 1 and {MAX_FRAMES}")
    if not math.isfinite(a.start_seconds) or not (0 <= a.start_seconds <= 86400):
        raise ValueError("Start time must be finite and between 0 and 86400 seconds")
    if not isinstance(a.mix, int) or not 0 <= a.mix <= 100:
        raise ValueError("Mix must be 0..100 percent")
    verify_code_checkout(a.checkout.resolve())
    checkpoint_hash = verify_weights(a.weights.resolve(), a.weights_sha256)
    ffmpeg = shutil.which(a.ffmpeg) if a.ffmpeg else shutil.which("ffmpeg")
    ffprobe = shutil.which(a.ffprobe) if a.ffprobe else shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("FFmpeg and FFprobe required")
    w, h, fps = probe(ffprobe, source)
    if w * h * 3 * a.frames > MAX_TOTAL_RAW_BYTES:
        raise ValueError("Selected sample too large; shorten the clip")
    if w < 32 or h < 32:
        raise ValueError("Video too small for GoPro-pretrained NAFNet")
    return {
        "source": str(source), "source_sha256": sha256_file(source),
        "output": str(output), "checkout": str(a.checkout.resolve()),
        "revision": REVISION, "model": f"NAFNet-GoPro-width{a.width}",
        "checkpoint": str(a.weights.resolve()), "checkpoint_sha256": checkpoint_hash,
        "frames_requested": a.frames, "start_seconds": a.start_seconds,
        "width": w, "height": h, "fps": fps, "mix_percent": a.mix,
        "device": a.device, "research_only": True, "commercial_clearance": False,
        "temporal_model": False, "audio": "omitted for short research comparison",
        "warning": "GoPro single-image deblur model may alter film grain and flicker between adjacent frames.",
        "_ffmpeg": ffmpeg, "_ffprobe": ffprobe,
    }


def read_exact(stream, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        part = stream.read(n - len(data))
        if not part:
            break
        data.extend(part)
    return bytes(data)


def decode_rgb(ffmpeg: str, source: Path, start: float, frames: int, w: int, h: int):
    import numpy as np
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start:.6f}", "-i", str(source), "-map", "0:v:0",
            "-an", "-sn", "-dn", "-frames:v", str(frames),
            "-fps_mode", "passthrough", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    seq = []
    try:
        assert p.stdout is not None
        for index in range(frames):
            sample = read_exact(p.stdout, w * h * 3)
            if len(sample) != w * h * 3:
                raise ValueError(f"Video ended before frame {index + 1} of {frames}")
            seq.append(np.frombuffer(sample, dtype=np.uint8).reshape(h, w, 3).copy())
        p.stdout.close()
        err = p.stderr.read().decode(errors="replace") if p.stderr else ""
        if p.wait(timeout=30):
            raise RuntimeError(f"FFmpeg decode error: {err[-900:]}")
        return seq
    finally:
        if p.poll() is None:
            p.kill()
            p.communicate()


def prepare_model(checkout: Path, weights: Path, width: int, device: str):
    import torch
    # External code is imported only after SHA/revision/clean-tree validation.
    sys.path.insert(0, str(checkout))
    from basicsr.models.archs.NAFNet_arch import NAFNet
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected dictionary of tensor weights")
    params = checkpoint.get("params", checkpoint.get("params_ema", checkpoint))
    if not isinstance(params, dict) or not params:
        raise ValueError("NAFNet checkpoint has no tensor state_dict")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in params.items()):
        raise ValueError("NAFNet checkpoint contains non-tensor entries")
    # Upstream uses the same NAFNet weights for NAFNetLocal evaluation.
    model = NAFNet(width=width, enc_blk_nums=[1, 1, 1, 28],
                   middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    state = {k.removeprefix("module."): v for k, v in params.items()}
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def restore_frame(model, frame, device: str, mix: int):
    import numpy as np
    import torch
    input_frame = np.asarray(frame, dtype=np.uint8)
    h, w = input_frame.shape[:2]
    x = torch.from_numpy(input_frame.copy()).permute(2, 0, 1).unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32).div_(255.)
    with torch.inference_mode():
        predicted = model(x).clamp(0., 1.)
    if tuple(predicted.shape) != (1, 3, h, w):
        raise ValueError(f"NAFNet changed frame geometry {tuple(predicted.shape)}")
    # Blend in float32 to avoid repeat 8-bit rounding before final encode.
    if mix != 100:
        predicted = (mix / 100) * predicted + (1 - mix / 100) * x
    arr = predicted.squeeze(0).permute(1, 2, 0).mul(255.).round()
    return arr.to(torch.uint8).cpu().numpy()


def write_lossless(ffmpeg: str, frames: list, fps: str, target: Path) -> None:
    import cv2
    with tempfile.TemporaryDirectory(prefix="arp-nafnet-") as work:
        folder = Path(work)
        for i, rgb in enumerate(frames):
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(f"Invalid frame shape {rgb.shape}")
            filename = folder / f"{i:06d}.png"
            if not cv2.imwrite(str(filename), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise IOError(f"Could not write research frame {i}")
        partial = target.with_name(target.stem + ".partial.mkv")
        partial.unlink(missing_ok=True)
        try:
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-framerate", fps, "-start_number", "0",
                 "-i", str(folder / "%06d.png"), "-an", "-c:v", "ffv1",
                 "-level", "3", "-pix_fmt", "bgr0", "-f", "matroska",
                 str(partial)],
                capture_output=True, text=True, check=True, timeout=180,
            )
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True, help="Explicitly prepared pinned external NAFNet source")
    p.add_argument("--weights", type=Path, required=True, help="GoPro model weights; never auto-fetched")
    p.add_argument("--weights-sha256", required=True)
    p.add_argument("--research-only", action="store_true")
    p.add_argument("--width", type=int, choices=WIDTHS, default=32)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--start-seconds", type=float, default=0.)
    p.add_argument("--mix", type=int, default=100)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    details = validate_inputs(a)
    ffmpeg, ffprobe = details.pop("_ffmpeg"), details.pop("_ffprobe")
    if a.dry_run:
        details["dry_run"] = True
        print(json.dumps(details, indent=2))
        return
    import torch
    if a.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; refusing implicit CPU fallback")
    model = prepare_model(a.checkout.resolve(), a.weights.resolve(), a.width, a.device)
    source = Path(details["source"])
    decoded = decode_rgb(ffmpeg, source, a.start_seconds, a.frames,
                         details["width"], details["height"])
    result_frames = [restore_frame(model, frame, a.device, a.mix) for frame in decoded]
    output = Path(details["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    write_lossless(ffmpeg, result_frames, details["fps"], output)
    w, h, fps = probe(ffprobe, output)
    if (w, h, fps) != (details["width"], details["height"], details["fps"]):
        output.unlink(missing_ok=True)
        raise ValueError("Result failed original geometry and cadence validation")
    details["frames_completed"] = len(result_frames)
    details["output_sha256"] = sha256_file(output)
    output.with_suffix(".research.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
