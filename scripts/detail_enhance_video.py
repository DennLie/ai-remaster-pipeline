#!/usr/bin/env python3
"""Optional native-resolution detail enhancement for ARP.

No learned models, no invented pixels, no automatic pre-DART sharpening.
Runs as an independent post-Clean Up / post-Stabilization stage.
The original is never overwritten and audio is stream-copied.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from common import (
    file_fingerprint, find_ffmpeg, resolve_path, resumable_output,
    root_relative, write_signature,
)
from intermediate_video import canonical_profile, codec_args, container_args

VERSION = 2
METHODS = ("off", "unsharp", "cas", "multiscale", "richardson_lucy")
PRESETS = ("gentle", "balanced", "strong")

# Luminance-only, at native resolution; all preset values are bounded.
UNSHARP = {
    "gentle": (3, 0.25), "balanced": (5, 0.55), "strong": (5, 0.95),
}
CAS = {"gentle": 0.18, "balanced": 0.38, "strong": 0.60}
# Multi-scale here means two sequential unsharp radii, NOT a detail
# synthesis network or mask-aware edge/texture isolation.
MULTISCALE = {
    "gentle": (0.12, 0.10),
    "balanced": (0.30, 0.22),
    "strong": (0.52, 0.38),
}


def validate(method: str, preset: str, mix: int) -> None:
    if method not in METHODS:
        raise ValueError(f"Unknown detail enhancement method: {method}")
    if preset not in PRESETS:
        raise ValueError(f"Unknown detail enhancement preset: {preset}")
    if not isinstance(mix, int) or not 0 <= mix <= 100:
        raise ValueError("Detail mix must be an integer from 0 to 100")


def effect_for(method: str, preset: str) -> str | None:
    validate(method, preset, 100)
    if method == "off":
        return None
    if method == "richardson_lucy":
        raise ValueError("Richardson–Lucy uses the dedicated CPU deconvolution runner, not FFmpeg filter graph")
    if method == "unsharp":
        size, amount = UNSHARP[preset]
        return f"unsharp=luma_msize_x={size}:luma_msize_y={size}:luma_amount={amount:g}:chroma_amount=0"
    if method == "cas":
        return f"cas=strength={CAS[preset]:g}:planes=1"
    local, broad = MULTISCALE[preset]
    return (
        f"unsharp=luma_msize_x=3:luma_msize_y=3:luma_amount={local:g}:chroma_amount=0,"
        f"unsharp=luma_msize_x=7:luma_msize_y=7:luma_amount={broad:g}:chroma_amount=0"
    )


def filter_graph(method: str, preset: str, mix: int) -> str | None:
    validate(method, preset, mix)
    effect = effect_for(method, preset)
    if not effect or mix == 0:
        return None
    if mix == 100:
        return f"[0:v]{effect}[enhanced]"
    fraction = mix / 100
    original = f"{1 - fraction:.2f}"
    processed = f"{fraction:.2f}"
    return (
        f"[0:v]split=2[original][working];"
        f"[working]{effect}[filtered];"
        f"[original][filtered]blend=all_expr='A*{original}+B*{processed}'[enhanced]"
    )


def probe(ffmpeg: str, path: Path) -> dict:
    tool = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe")
    result = subprocess.run(
        [str(tool), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
        text=True, capture_output=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"Video stream not found: {path}")
    return streams[0]


def signature(source: Path, method: str, preset: str, mix: int, profile: str) -> dict:
    validate(method, preset, mix)
    return {
        "tool": "detail_enhance_video.py", "version": VERSION,
        "source": root_relative(source), "source_fingerprint": file_fingerprint(source),
        "method": method, "preset": preset, "mix_percent": mix,
        "profile": canonical_profile(profile),
    }


def run(source: Path, target: Path, *, method: str, preset: str, mix: int,
        intermediate_profile: str, ffmpeg: str, force: bool = False,
        dry_run: bool = False) -> dict:
    source, target = source.resolve(), target.resolve()
    validate(method, preset, mix)
    if method == "richardson_lucy":
        from richardson_lucy_video import run as run_rl
        return run_rl(source, target, method=method, preset=preset, mix=mix,
                      intermediate_profile=intermediate_profile, ffmpeg=ffmpeg,
                      force=force, dry_run=dry_run)
    graph = filter_graph(method, preset, mix)
    profile = canonical_profile(intermediate_profile)
    if source == target:
        raise ValueError("Refusing to overwrite original source")
    if target.suffix.lower() != ".mkv":
        raise ValueError("Output must be a distinct .mkv intermediate")
    if not source.is_file():
        raise FileNotFoundError(f"Missing source: {source}")
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg unavailable")
    before = probe(ffmpeg, source)
    sig = signature(source, method, preset, mix, profile)
    if not force and resumable_output(target, sig, video_like=source):
        return {"output": str(target), "method": method, "preset": preset,
                "mix_percent": mix, "reused": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.mkv")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source)]
    if graph:
        cmd += ["-filter_complex", graph, "-map", "[enhanced]", "-map", "0:a?",
                "-fps_mode", "passthrough", *codec_args(profile, fast=True),
                "-c:a", "copy", *container_args(str(partial), profile), str(partial)]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a?", "-c", "copy", str(partial)]
    if dry_run:
        return {"output": str(target), "method": method, "preset": preset,
                "mix_percent": mix, "filter_graph": graph, "command": cmd, "dry_run": True}
    partial.unlink(missing_ok=True)
    try:
        subprocess.run(cmd, check=True)
        after = probe(ffmpeg, partial)
        for key in ("width", "height", "avg_frame_rate"):
            if str(after.get(key)) != str(before.get(key)):
                raise ValueError(f"Detail enhancement changed {key}")
        if (before.get("nb_frames") not in (None, "N/A")
                and after.get("nb_frames") not in (None, "N/A")
                and int(before["nb_frames"]) != int(after["nb_frames"])):
            raise ValueError("Detail enhancement changed frame count")
        partial.replace(target)
        write_signature(target, sig)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"output": str(target), "method": method, "preset": preset,
            "mix_percent": mix, "filter_graph": graph, "reused": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="off")
    parser.add_argument("--preset", choices=PRESETS, default="gentle")
    parser.add_argument("--mix", type=int, default=100, help="0=untouched, 100=full enhancement")
    parser.add_argument("--intermediate-profile", choices=("low", "medium", "high", "lossless"),
                        default="lossless")
    parser.add_argument("--ffmpeg", help="FFmpeg executable override")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(resolve_path(args.source), resolve_path(args.output), method=args.method,
                 preset=args.preset, mix=args.mix, intermediate_profile=args.intermediate_profile,
                 ffmpeg=find_ffmpeg(args.ffmpeg), force=args.force, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
