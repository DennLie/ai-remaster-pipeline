#!/usr/bin/env python3
"""Fail-closed FlashVSR+ research wrapper. No model code/weights are bundled.

Runs the pinned upstream CUDA-only streaming CLI ONLY after --research-only,
a clean exact Git checkout, and full SHA-256 verification of ALL FOUR
model assets. The source is clipped without scaling; the official
inference backend handles aspect-preserving resizing at the requested
output geometry. It is generative VSR, NOT historical 1x deblurring.
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

UPSTREAM = "https://github.com/OpenImagingLab/FlashVSR_Plus"
PINNED_COMMIT = "b548a2cc2c929d3d4676f2973abe27876ec4dee7"
MODES = ("turbo", "base")
MODEL_FILES = ("dit.safetensors", "lq_proj_in.safetensors",
               "TCDecoder.ckpt", "posi_prompt.pth")
MAX_FRAMES = 48
MAX_SOURCE_RGB_BYTES = 512 * 1024 * 1024
MAX_MODEL_FILE_BYTES = 12 * 1024 * 1024 * 1024


def exec_text(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True,
                          check=True).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkout(checkout: Path) -> None:
    """Pin the code; ignored ckpts/ are checked separately with SHA-256."""
    if not checkout.is_dir():
        raise FileNotFoundError(f"FlashVSR+ checkout missing: {checkout}")
    head = exec_text(["git", "-C", str(checkout), "rev-parse", "HEAD"])
    if head.lower() != PINNED_COMMIT:
        raise ValueError(f"Unexpected upstream revision {head}; expected {PINNED_COMMIT}")
    dirty = exec_text(["git", "-C", str(checkout), "status",
                       "--porcelain", "--untracked-files=normal"])
    if dirty:
        raise ValueError("External FlashVSR+ checkout has modifications or untracked files")
    for p in ("LICENSE", "inference/run_stream.py", "inference/presets.py",
              "prompt_tensor/posi_prompt.pth"):
        if not (checkout / p).is_file():
            raise ValueError(f"Pinned FlashVSR+ checkout missing {p}")


def checkpoint_paths(checkout: Path, mode: str) -> dict[str, Path]:
    if mode not in MODES:
        raise ValueError(f"Unsupported FlashVSR+ research mode: {mode}")
    return {
        "dit.safetensors": checkout / "ckpts" / mode / "dit.safetensors",
        "lq_proj_in.safetensors": checkout / "ckpts" / mode / "lq_proj_in.safetensors",
        "TCDecoder.ckpt": checkout / "ckpts" / "TCDecoder.ckpt",
        "posi_prompt.pth": checkout / "prompt_tensor" / "posi_prompt.pth",
    }


def verify_assets(checkout: Path, mode: str, manifest: Path) -> dict[str, str]:
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing external checksum manifest {manifest}")
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != set(MODEL_FILES):
        raise ValueError("Manifest requires exactly four named FlashVSR+ asset hashes")
    assets = checkpoint_paths(checkout, mode)
    verified = {}
    for name, path in assets.items():
        expected = raw[name]
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
            raise ValueError(f"Full 64-character SHA-256 required for {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing FlashVSR+ model asset: {path}")
        if not 0 < path.stat().st_size <= MAX_MODEL_FILE_BYTES:
            raise ValueError(f"Invalid or oversized FlashVSR+ model asset {name}")
        actual = sha256_file(path)
        if actual != expected.lower():
            raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
        verified[name] = actual
    return verified


def probe(ffprobe: str, video: Path) -> tuple[int, int, str, int]:
    output = exec_text([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
        "-of", "json", str(video),
    ])
    streams = json.loads(output).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found: {video}")
    row = streams[0]
    w, h = int(row["width"]), int(row["height"])
    fps = str(row["avg_frame_rate"])
    numerator, sep, denominator = fps.partition("/")
    if w < 16 or h < 16 or w > 8192 or h > 8192 or not sep or int(numerator) <= 0 or int(denominator) <= 0:
        raise ValueError(f"Invalid source video size/rate: {w}x{h}@{fps}")
    if row.get("r_frame_rate") != fps:
        raise ValueError("Variable-rate input is not supported in this frame-exact research wrapper")
    frame_count = int(row.get("nb_read_frames", 0) or 0)
    if frame_count < 1:
        raise ValueError("No decodable input frames")
    return w, h, fps, frame_count


def validate(a: argparse.Namespace) -> dict:
    if not a.research_only:
        raise ValueError("FlashVSR+ is research-only and requires explicit --research-only")
    source = a.source.resolve()
    output = a.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video missing: {source}")
    if source == output:
        raise ValueError("Refusing to overwrite original video")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() != ".mp4":
        raise ValueError("Upstream research CLI outputs lossy preview .mp4")
    if a.mode not in MODES or not isinstance(a.frames, int) or not 1 <= a.frames <= MAX_FRAMES:
        raise ValueError(f"Use Turbo/Base and 1..{MAX_FRAMES} frames")
    if not math.isfinite(a.start_seconds) or not 0 <= a.start_seconds <= 86400:
        raise ValueError("Start seconds must be finite, 0..86400")
    if a.width <= 0 or a.height <= 0 or a.width % 128 or a.height % 128:
        raise ValueError("FlashVSR+ output width and height must be multiples of 128")
    if a.width > 4096 or a.height > 4096:
        raise ValueError("Research output capped at 4096 pixels per dimension")
    if a.temporal_compression not in (1, 2, 4):
        raise ValueError("Temporal compression must be 1, 2 or 4")
    checkout = a.checkout.resolve()
    verify_checkout(checkout)
    checksums = verify_assets(checkout, a.mode, a.manifest.resolve())
    ffmpeg = shutil.which(a.ffmpeg) if a.ffmpeg else shutil.which("ffmpeg")
    ffprobe = shutil.which(a.ffprobe) if a.ffprobe else shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise FileNotFoundError("FFmpeg/FFprobe missing")
    w, h, fps, count = probe(ffprobe, source)
    # No silent center cropping or stretching. 960x720 -> 1024x768 is 4:3.
    if w * a.height != h * a.width:
        raise ValueError("Output aspect ratio differs from source; refusing upstream center crop")
    if w * h * a.frames * 3 > MAX_SOURCE_RGB_BYTES:
        raise ValueError("Research frame window exceeds decoded RGB memory cap")
    fps_number = int(fps.split("/")[0]) / int(fps.split("/")[1])
    if a.start_seconds * fps_number + a.frames > count + 1:
        raise ValueError("Selected clip extends beyond verified source frame count")
    return {
        "upstream": UPSTREAM, "pinned_commit": PINNED_COMMIT,
        "mode": a.mode, "code_checkout": str(checkout),
        "assets_sha256": checksums,
        "original_path": str(source), "original_sha256": sha256_file(source),
        "output": str(output), "output_geometry": [a.width, a.height],
        "source_geometry": [w, h], "fps": fps, "source_frames": count,
        "frames_requested": a.frames, "start_seconds": a.start_seconds,
        "temporal_compression": a.temporal_compression,
        "seed": a.seed, "hardware_profile": "generic",
        "attention": "sdpa", "cache_offload": "auto",
        "research_only": True, "commercial_clearance": False,
        "video_codec": "upstream H.264 MP4; lossy preview only",
        "audio": "intentionally omitted from research excerpt",
        "limitation": "Generative VSR at fixed 4:3, not native-frame reconstruction or archival ground truth",
        "_ffmpeg": ffmpeg, "_ffprobe": ffprobe,
    }


def build_upstream_command(a: argparse.Namespace, input_clip: Path, output: Path) -> list[str]:
    return [
        str(a.python.resolve()), "-m", "inference.run_stream",
        "--input", str(input_clip), "--output", str(output),
        "--mode", a.mode, "--height", str(a.height),
        "--width", str(a.width), "--max-frames", str(a.frames),
        "--seed", str(a.seed), "--hardware-profile", "generic",
        "--attn-kernel", "sdpa", "--cache-offload-mode", "auto",
        "--tiny-decoder-mode", "serial",
        "--temporal-compression", str(a.temporal_compression),
        "--no-color-correction",
        "--ckpt-dir", str(a.checkout.resolve() / "ckpts" / a.mode),
    ]


def run(a: argparse.Namespace) -> dict:
    report = validate(a)  # exact code, all external files, and source first
    ffmpeg, ffprobe = report.pop("_ffmpeg"), report.pop("_ffprobe")
    # Dry-run validates the same full provenance without importing CUDA or dependencies.
    if a.dry_run:
        report["dry_run"] = True
        report["representative_command"] = build_upstream_command(
            a, Path("<temporary-lossless-original-clip.mkv>"), Path(report["output"]))
        return report
    source, target = a.source.resolve(), a.output.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="arp-flashvsr-plus-") as temp:
        directory = Path(temp)
        clip = directory / "original_excerpt.mkv"
        result = directory / "flashvsr_plus_preview.mp4"
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{a.start_seconds:.6f}", "-i", str(source),
            "-frames:v", str(a.frames), "-map", "0:v:0", "-an",
            "-fps_mode", "passthrough", "-c:v", "ffv1", "-pix_fmt", "yuv420p",
            str(clip),
        ], capture_output=True, text=True, check=True)
        cw, ch, cfps, n = probe(ffprobe, clip)
        if [cw, ch] != report["source_geometry"] or cfps != report["fps"] or n != a.frames:
            raise ValueError("Source extraction changed frame count, geometry or cadence")
        command = build_upstream_command(a, clip, result)
        subprocess.run(command, cwd=a.checkout.resolve(), check=True)
        if not result.is_file():
            raise FileNotFoundError("Upstream inference did not produce its expected MP4")
        rw, rh, rfps, rn = probe(ffprobe, result)
        if (rw, rh) != (a.width, a.height) or rfps != report["fps"] or rn != a.frames:
            raise ValueError(f"Output mismatch: {rw}x{rh}, fps={rfps}, frames={rn}; expected {a.width}x{a.height}, {report['fps']}, {a.frames}")
        if target.exists():
            raise FileExistsError("Research result path appeared during model inference")
        result.replace(target)
        report["output_sha256"] = sha256_file(target)
        report["completed_frames"] = rn
        report["effective_input_sha256"] = sha256_file(clip)
        report["actual_command"] = command
    target.with_suffix(".research.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True, help="Manually cloned pinned clean code tree")
    p.add_argument("--manifest", type=Path, required=True, help="JSON of 4 reviewed SHA-256 digests; stored outside checkout")
    p.add_argument("--mode", choices=MODES, default="turbo")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--frames", type=int, default=24)
    p.add_argument("--start-seconds", type=float, default=300.)
    p.add_argument("--temporal-compression", type=int, choices=(1, 2, 4), default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--research-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--python", type=Path, default=Path(sys.executable), help="Separate reviewed CUDA/Python 3.10 env")
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")
    return p


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
