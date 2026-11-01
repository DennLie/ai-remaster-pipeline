#!/usr/bin/env python3
"""Optional model-free noise reduction for film comparison in ARP.

Each backend has method-specific gentle/balanced/strong presets.
This stage preserves a source file, its geometry, cadence and audio,
but it is NOT a claim that original grain survives unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from common import (
    ROOT, file_fingerprint, find_ffmpeg, resolve_path, resumable_output,
    root_relative, write_signature,
)
from intermediate_video import canonical_profile, codec_args, container_args

ALGORITHM_VERSION = 1
METHODS = ("off", "hqdn3d", "nlmeans", "bm3d")
PRESETS = ("gentle", "balanced", "strong")

# Each preset uses native FFmpeg filter units; there is deliberately no
# misleading algorithm-independent strength percentage.
HQDN3D = {
    "gentle": (0.9, 0.7, 1.2, 0.8),
    "balanced": (1.8, 1.4, 2.4, 1.6),
    "strong": (3.0, 2.4, 4.0, 3.2),
}
NLMEANS = {"gentle": 1.0, "balanced": 2.0, "strong": 4.0}
BM3D = {"gentle": 1.5, "balanced": 3.0, "strong": 5.0}


def filter_for(method: str, preset: str) -> str | None:
    if method not in METHODS:
        raise ValueError(f"Unsupported denoise method: {method}")
    if preset not in PRESETS:
        raise ValueError(f"Unsupported denoise preset: {preset}")
    if method == "off":
        return None
    if method == "hqdn3d":
        ls, cs, lt, ct = HQDN3D[preset]
        return f"hqdn3d=luma_spatial={ls:g}:chroma_spatial={cs:g}:luma_tmp={lt:g}:chroma_tmp={ct:g}"
    if method == "nlmeans":
        return f"nlmeans=s={NLMEANS[preset]:g}:p=3:r=7"
    return f"bm3d=sigma={BM3D[preset]:g}:block=4:bstep=2:group=1:estim=basic"


def probe(ffmpeg: str, video: Path) -> dict:
    ffprobe = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe")
    result = subprocess.run(
        [str(ffprobe), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,pix_fmt",
         "-of", "json", str(video)],
        check=True, text=True, capture_output=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No decodable video stream: {video}")
    return streams[0]


def signature(source: Path, method: str, preset: str, profile: str) -> dict:
    return {
        "tool": "denoise_video.py", "version": ALGORITHM_VERSION,
        "source": root_relative(source),
        "source_fingerprint": file_fingerprint(source),
        "method": method, "preset": preset,
        "intermediate_profile": canonical_profile(profile),
    }


def run(source: Path, target: Path, *, method: str, preset: str,
        intermediate_profile: str, ffmpeg: str, force: bool = False,
        dry_run: bool = False) -> dict:
    source = source.resolve()
    target = target.resolve()
    effect = filter_for(method, preset)
    profile = canonical_profile(intermediate_profile)
    if source == target:
        raise ValueError("Denoise output must be different from the original source")
    if target.suffix.lower() != ".mkv":
        raise ValueError("Denoise output must be .mkv (Matroska intermediate)")
    if not source.is_file():
        raise FileNotFoundError(f"Source video missing: {source}")
    if not ffmpeg:
        raise ValueError("FFmpeg is required")
    info = probe(ffmpeg, source)
    sig = signature(source, method, preset, profile)
    if not force and resumable_output(target, sig, video_like=source):
        return {"output": str(target), "reused": True, "method": method, "preset": preset}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.mkv")
    if method == "off":
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
               "-c", "copy", str(partial)]
    else:
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-i", str(source), "-map", "0:v:0", "-map", "0:a?",
               "-vf", effect, "-fps_mode", "passthrough",
               *codec_args(profile, fast=True), "-c:a", "copy",
               *container_args(str(partial), profile), str(partial)]
    if dry_run:
        return {"output": str(target), "method": method, "preset": preset,
                "command": cmd, "dry_run": True}
    partial.unlink(missing_ok=True)
    try:
        subprocess.run(cmd, check=True)
        finished = probe(ffmpeg, partial)
        if (int(finished["width"]), int(finished["height"])) != (
            int(info["width"]), int(info["height"])
        ):
            raise ValueError("Denoiser changed frame geometry unexpectedly")
        if finished.get("avg_frame_rate") != info.get("avg_frame_rate"):
            raise ValueError("Denoiser changed frame cadence unexpectedly")
        if (info.get("nb_frames") not in (None, "N/A")
                and finished.get("nb_frames") not in (None, "N/A")
                and int(info["nb_frames"]) != int(finished["nb_frames"])):
            raise ValueError("Denoiser changed frame count unexpectedly")
        partial.replace(target)
        write_signature(target, sig)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return {"output": str(target), "reused": False,
            "method": method, "preset": preset,
            "filter": effect, "profile": profile}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Original or selected source video")
    parser.add_argument("--output", required=True, help="Non-destructive .mkv output path")
    parser.add_argument("--method", choices=METHODS, default="off")
    parser.add_argument("--preset", choices=PRESETS, default="gentle")
    parser.add_argument("--intermediate-profile", choices=("low", "medium", "high", "lossless"), default="lossless")
    parser.add_argument("--ffmpeg", help="Optional explicit FFmpeg executable")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outcome = run(
        resolve_path(args.source), resolve_path(args.output),
        method=args.method, preset=args.preset,
        intermediate_profile=args.intermediate_profile,
        ffmpeg=find_ffmpeg(args.ffmpeg),
        force=args.force, dry_run=args.dry_run,
    )
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
