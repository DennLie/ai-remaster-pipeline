#!/usr/bin/env python3
"""Opt-in, CPU-only luminance Richardson–Lucy deconvolution for ARP.

The Gaussian point-spread function (PSF) below is ASSUMED, not measured.
This does not establish recovery of historical detail; it can amplify grain,
scratches and ringing. Preserves frame geometry, nominal CFR and audio.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from common import file_fingerprint, find_ffmpeg, resolve_path, resumable_output, root_relative, write_signature
from intermediate_video import canonical_profile, codec_args, container_args

VERSION = 1
PSF_PRESETS = {
    "gentle": (0.75, 3),
    "balanced": (1.00, 5),
    "strong": (1.35, 8),
}


def validate(preset: str, mix: int) -> tuple[float, int]:
    if preset not in PSF_PRESETS:
        raise ValueError(f"Invalid Richardson–Lucy preset: {preset}")
    if not isinstance(mix, int) or not 0 <= mix <= 100:
        raise ValueError("Mix must be in range 0..100")
    return PSF_PRESETS[preset]


def point_spread_function(sigma: float):
    """Finite, normalized isotropic Gaussian; explicitly an assumed blur kernel."""
    import cv2
    import numpy as np
    radius = max(1, int(math.ceil(3 * sigma)))
    axis = cv2.getGaussianKernel(2 * radius + 1, sigma).reshape(-1)
    kernel = np.outer(axis, axis)
    return (kernel / kernel.sum()).astype(np.float64)


def deblur_rgb(frame, psf, iterations: int, mix: int):
    import cv2
    import numpy as np
    from skimage.restoration import richardson_lucy

    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected RGB8 frame")
    if not 1 <= iterations <= 12:
        raise ValueError("Iterations must be between 1 and 12")
    if not 0 <= mix <= 100:
        raise ValueError("Mix must be between 0 and 100")
    # Keep chroma unchanged as far as RGB round-trip permits.
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
    luminance = ycrcb[:, :, 0].astype(np.float64) / 255
    deconvolved = richardson_lucy(luminance, psf, num_iter=iterations,
                                  clip=True, filter_epsilon=1e-8)
    if not np.all(np.isfinite(deconvolved)):
        raise ValueError("Deblur produced non-finite luminance")
    ycrcb[:, :, 0] = np.clip(
        np.rint((1 - mix / 100) * luminance * 255 +
                (mix / 100) * deconvolved * 255), 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def probe(ffmpeg: str, source: Path) -> dict:
    companion = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe")
    obj = subprocess.run(
        [str(companion), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
         "-of", "json", str(source)], capture_output=True, text=True, check=True
    )
    items = json.loads(obj.stdout).get("streams", [])
    if not items:
        raise ValueError("No video stream")
    return items[0]


def signature(source: Path, preset: str, mix: int, profile: str) -> dict:
    sigma, iterations = validate(preset, mix)
    return {
        "tool": "richardson_lucy_video.py", "version": VERSION,
        "source": root_relative(source), "source_fingerprint": file_fingerprint(source),
        "method": "richardson_lucy", "psf_type": "assumed_gaussian",
        "psf_sigma_pixels": sigma, "iterations": iterations,
        "preset": preset, "mix_percent": mix,
        "profile": canonical_profile(profile),
    }


def read_frame(pipe, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        block = pipe.read(n - len(chunks))
        if not block:
            break
        chunks.extend(block)
    return bytes(chunks)


def run(source: Path, target: Path, *, method: str = "richardson_lucy",
        preset: str, mix: int, intermediate_profile: str, ffmpeg: str,
        force: bool = False, dry_run: bool = False) -> dict:
    import numpy as np
    source, target = source.resolve(), target.resolve()
    if method != "richardson_lucy":
        raise ValueError("This runner supports only Richardson–Lucy")
    sigma, iterations = validate(preset, mix)
    if source == target:
        raise ValueError("Refusing to overwrite original source")
    if target.suffix.lower() != ".mkv" or not source.is_file():
        raise ValueError("Requires an existing source and distinct .mkv output")
    if not ffmpeg:
        raise FileNotFoundError("FFmpeg unavailable")
    metadata = probe(ffmpeg, source)
    if metadata.get("avg_frame_rate") != metadata.get("r_frame_rate"):
        raise ValueError("Variable frame rate unsupported for frame-exact Richardson–Lucy processing")
    fps = str(metadata["avg_frame_rate"])
    w, h = int(metadata["width"]), int(metadata["height"])
    if w < 16 or h < 16 or max(w, h) > 8192:
        raise ValueError("Unsupported image geometry")
    profile = canonical_profile(intermediate_profile)
    sig = signature(source, preset, mix, profile)
    if not force and resumable_output(target, sig, video_like=source):
        return {"output": str(target), "reused": True, "method": method, "preset": preset}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.stem + ".partial.mkv")
    decode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
              "-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
              "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    encode = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
              "-f", "rawvideo", "-pixel_format", "rgb24",
              "-video_size", f"{w}x{h}", "-framerate", fps, "-i", "pipe:0",
              "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
              "-fps_mode", "passthrough",
              *codec_args(profile, fast=True), "-c:a", "copy",
              *container_args(str(partial), profile), str(partial)]
    if dry_run:
        return {"output": str(target), "method": method, "preset": preset,
                "psf_sigma": sigma, "iterations": iterations, "mix": mix,
                "decode": decode, "encode": encode, "dry_run": True}
    kernel = point_spread_function(sigma)
    partial.unlink(missing_ok=True)
    decoder = subprocess.Popen(decode, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(encode, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    count = 0
    try:
        assert decoder.stdout is not None and encoder.stdin is not None
        while True:
            raw = read_frame(decoder.stdout, w * h * 3)
            if not raw:
                break
            if len(raw) != w * h * 3:
                raise ValueError("Truncated decoded frame")
            rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            processed = deblur_rgb(rgb, kernel, iterations, mix)
            encoder.stdin.write(processed.tobytes())
            count += 1
        decoder.stdout.close()
        if decoder.wait(timeout=45):
            raise RuntimeError("FFmpeg decoder failed")
        encoder.stdin.close()
        if encoder.wait(timeout=90):
            raise RuntimeError("FFmpeg encoder failed")
        if not count:
            raise ValueError("No decoded frames")
        after = probe(ffmpeg, partial)
        for key in ("width", "height", "avg_frame_rate"):
            if str(after.get(key)) != str(metadata.get(key)):
                raise ValueError(f"Output changed {key}")
        if metadata.get("nb_frames") not in (None, "N/A"):
            # Matroska may omit nb_frames; explicitly count output when needed.
            companion = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix.lower() == ".exe" else "ffprobe")
            tally = subprocess.run([str(companion), "-v", "error", "-select_streams", "v:0",
                                    "-count_frames", "-show_entries", "stream=nb_read_frames",
                                    "-of", "csv=p=0", str(partial)],
                                   capture_output=True, text=True, check=True)
            if int(tally.stdout.strip()) != count:
                raise ValueError("Encoded frame count mismatch")
        partial.replace(target)
        write_signature(target, sig)
        return {"output": str(target), "reused": False, "frames": count,
                "method": method, "preset": preset, "mix": mix,
                "psf_sigma": sigma, "iterations": iterations}
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        for proc in (decoder, encoder):
            if proc.poll() is None:
                proc.kill()
                proc.communicate()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preset", choices=tuple(PSF_PRESETS), default="gentle")
    p.add_argument("--mix", type=int, default=100)
    p.add_argument("--intermediate-profile", choices=("low", "medium", "high", "lossless"), default="lossless")
    p.add_argument("--ffmpeg")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    print(json.dumps(run(resolve_path(args.source), resolve_path(args.output),
                         preset=args.preset, mix=args.mix,
                         intermediate_profile=args.intermediate_profile,
                         ffmpeg=find_ffmpeg(args.ffmpeg), force=args.force,
                         dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
