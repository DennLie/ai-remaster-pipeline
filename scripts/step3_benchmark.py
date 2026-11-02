#!/usr/bin/env python3
"""Reproducible Step-3 video-detail benchmarks, without running GPU models.

PREPARE: make lossless 1x source excerpts plus matched-size Lanczos
reference clips for VSR. Models must consume these exact excerpts
from frame zero; NEVER independently seek the long film in each model.

EVALUATE: reject mismatched geometry/cadence/frames and, by default,
missing or nonmatching research/signature provenance. Report descriptive
metrics separately for native 1x and generative VSR target geometry.
There is no quality ranking or commercial-rights certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

SCHEMA_VERSION = 1
MAX_SHOTS = 8
MAX_SHOT_FRAMES = 72
MAX_RESOLUTION = 4096
SHOT_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_shots(specs: list[str]) -> list[dict]:
    if not 1 <= len(specs) <= MAX_SHOTS:
        raise ValueError(f"Provide between one and {MAX_SHOTS} --shot entries")
    shots = []
    used = set()
    for spec in specs:
        name, eq, timing = spec.partition("=")
        seconds, sep, frames = timing.partition(":")
        if not eq or not sep or not SHOT_LABEL.fullmatch(name):
            raise ValueError(f"Invalid shot NAME=START_SECONDS:FRAMES: {spec!r}")
        if name.lower() in used:
            raise ValueError(f"Duplicate shot name: {name}")
        try:
            start = float(seconds)
            count = int(frames)
        except ValueError as e:
            raise ValueError(f"Invalid numeric shot values: {spec}") from e
        if not math.isfinite(start) or start < 0 or start > 86400:
            raise ValueError(f"Invalid shot start time: {spec}")
        if not 1 <= count <= MAX_SHOT_FRAMES:
            raise ValueError(f"Shot frames must be 1..{MAX_SHOT_FRAMES}: {spec}")
        used.add(name.lower())
        shots.append({"name": name, "start_seconds": start, "frames": count})
    return shots


def executable(name: str | None, default: str) -> str:
    value = name or default
    found = shutil.which(value)
    if not found:
        raise FileNotFoundError(f"Required executable not found: {value}")
    return found


def video_probe(ffprobe: str, path: Path) -> dict:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
         "-of", "json", str(path)], capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"Video stream missing: {path}")
    info = streams[0]
    w, h = int(info["width"]), int(info["height"])
    if min(w, h) < 16 or max(w, h) > 8192:
        raise ValueError(f"Unexpected geometry {w}x{h}: {path}")
    try:
        fps = Fraction(info["avg_frame_rate"])
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"Invalid video frame rate: {path}")
    if fps <= 0 or fps > 240:
        raise ValueError(f"Invalid frame rate: {fps}")
    if Fraction(info.get("r_frame_rate", "0/1")) != fps:
        raise ValueError(f"Variable/nonmatching frame rate: {path}")
    try:
        count = int(info["nb_read_frames"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"Cannot verify decoded frame count: {path}")
    if count < 1:
        raise ValueError(f"Source has zero frames: {path}")
    return {"width": w, "height": h, "fps": f"{fps.numerator}/{fps.denominator}",
            "frame_count": count}


def validate_geometry(original: dict, width: int, height: int) -> None:
    if width <= 0 or height <= 0 or width > MAX_RESOLUTION or height > MAX_RESOLUTION:
        raise ValueError("Unsupported upscale reference dimensions")
    if width % 128 or height % 128:
        raise ValueError("VSR target dimensions must be multiples of 128")
    if original["width"] * height != original["height"] * width:
        raise ValueError("Target aspect ratio does not match source; no center cropping")


def verify_video(path: Path, ffprobe: str, *, width: int, height: int,
                 fps: str, frames: int) -> dict:
    actual = video_probe(ffprobe, path)
    if (actual["width"], actual["height"], actual["fps"], actual["frame_count"]) != (
        width, height, fps, frames
    ):
        raise ValueError(f"Frame-exact video mismatch at {path}: {actual}; "
                         f"expected {width}x{height}, {fps}, {frames} frames")
    return actual


def checked_relative(manifest_dir: Path, value: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Missing relative clip path")
    target = (manifest_dir / value).resolve()
    try:
        target.relative_to(manifest_dir.resolve())
    except ValueError as e:
        raise ValueError(f"Clip path escapes benchmark folder: {value}") from e
    return target


def prepare(*, source: Path, destination: Path, shots: list[dict],
            width: int = 1024, height: int = 768,
            expected_source_sha256: str | None = None,
            ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    source, destination = source.resolve(), destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Original film source missing: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark folder: {destination}")
    # Confirm all requested shots before creating anything.
    shots = parse_shots([f"{s['name']}={s['start_seconds']}:{s['frames']}" for s in shots])
    original = video_probe(ffprobe, source)
    validate_geometry(original, width, height)
    original_sha256 = file_hash(source)
    if expected_source_sha256 is not None:
        if not re.fullmatch("[0-9a-fA-F]{64}", expected_source_sha256):
            raise ValueError("Source SHA-256 must be 64 hex characters")
        if original_sha256 != expected_source_sha256.lower():
            raise ValueError(f"Source SHA-256 mismatch: {original_sha256}")
    rate = Fraction(original["fps"])
    for shot in shots:
        # Verify frame bounds conservatively; actual frame count is always rechecked.
        if int(shot["start_seconds"] * rate) + shot["frames"] > original["frame_count"] + 1:
            raise ValueError(f"Shot {shot['name']} extends beyond original film")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step3-staging-", dir=destination.parent) as td:
        staging = Path(td)
        result = {
            "schema_version": SCHEMA_VERSION,
            "kind": "ARP-Step3-frame-aligned-benchmark",
            "original": {"path": str(source), "sha256": original_sha256, **original},
            "target_geometry": {"width": width, "height": height},
            "shots": [],
            "warning": (
                "Fixed frame-exact source clips; video-super-resolution and native-1x "
                "scores are separate. Metrics do not prove historical detail recovery."
            ),
        }
        for shot in shots:
            sub = staging / shot["name"]
            sub.mkdir()
            native = sub / "native.mkv"
            lanczos = sub / "lanczos_reference.mkv"
            decode_cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-ss", f"{shot['start_seconds']:.6f}", "-i", str(source),
                "-map", "0:v:0", "-frames:v", str(shot["frames"]),
                "-an", "-fps_mode", "passthrough", "-c:v", "ffv1",
                "-pix_fmt", "bgr0", str(native),
            ]
            subprocess.run(decode_cmd, capture_output=True, check=True)
            verify_video(native, ffprobe, width=original["width"],
                         height=original["height"], fps=original["fps"],
                         frames=shot["frames"])
            scale_cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(native),
                "-map", "0:v:0", "-an", "-vf",
                f"scale={width}:{height}:flags=lanczos,format=bgr0",
                "-fps_mode", "passthrough", "-c:v", "ffv1", str(lanczos),
            ]
            subprocess.run(scale_cmd, capture_output=True, check=True)
            verify_video(lanczos, ffprobe, width=width, height=height,
                         fps=original["fps"], frames=shot["frames"])
            result["shots"].append({
                **shot,
                "native_path": f"{shot['name']}/native.mkv",
                "native_sha256": file_hash(native),
                "lanczos_path": f"{shot['name']}/lanczos_reference.mkv",
                "lanczos_sha256": file_hash(lanczos),
            })
        (staging / "manifest.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise FileExistsError("Benchmark destination appeared while preparing")
        staging.rename(destination)
    return result


def validate_provenance(variant: Path, source_sha256: str, frames: int,
                        *, allow_unverified: bool = False) -> dict:
    research = variant.with_suffix(".research.json")
    sig = variant.with_suffix(variant.suffix + ".sig.json")
    if research.is_file():
        data = json.loads(research.read_text(encoding="utf-8"))
        actual_source_hash = data.get("source_sha256", data.get("original_sha256"))
        if actual_source_hash != source_sha256:
            raise ValueError(f"Research source SHA mismatch for {variant}: "
                             "must process the exact prepared native.mkv clip")
        if data.get("start_seconds") != 0:
            raise ValueError(f"Variant {variant} was not processed from frame zero of native.mkv")
        declared = data.get("frames_requested", data.get("frames"))
        if declared != frames:
            raise ValueError(f"Research frame-count mismatch for {variant}")
        reported_output = data.get("output_sha256", data.get("result_sha256"))
        if reported_output and reported_output != file_hash(variant):
            raise ValueError(f"Research output digest mismatch for {variant}")
        return {"status": "research_sidecar_verified", "sidecar": str(research)}
    if sig.is_file():
        data = json.loads(sig.read_text(encoding="utf-8"))
        fingerprint = data.get("source_fingerprint", {})
        if not isinstance(fingerprint, dict) or fingerprint.get("sha256") != source_sha256:
            raise ValueError(f"Signature source SHA mismatch for {variant}")
        return {"status": "arp_stage_signature_verified", "sidecar": str(sig)}
    if not allow_unverified:
        raise ValueError(
            f"No matching source signature for {variant}. Supply the model's .research.json "
            "or ARP .mkv.sig.json, or opt in to --allow-unverified (manual alignment required)."
        )
    return {"status": "UNVERIFIED_MANUAL_ALIGNMENT", "sidecar": None}


def evaluate(*, manifest_path: Path, shot_name: str, native: dict[str, Path],
             vsr: dict[str, Path], destination: Path, ffprobe: str = "ffprobe",
             allow_unverified: bool = False) -> dict:
    from compare_denoising import compare

    manifest_path, destination = manifest_path.resolve(), destination.resolve()
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported benchmark manifest schema")
    row = next((s for s in manifest.get("shots", []) if s["name"] == shot_name), None)
    if row is None:
        raise ValueError(f"Shot {shot_name!r} not in prepared benchmark")
    if not native and not vsr:
        raise ValueError("At least one native or VSR candidate required")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation folder: {destination}")
    original = manifest["original"]
    target = manifest["target_geometry"]
    refs = [
        (checked_relative(root, row["native_path"]), row["native_sha256"],
         original["width"], original["height"]),
        (checked_relative(root, row["lanczos_path"]), row["lanczos_sha256"],
         target["width"], target["height"]),
    ]
    for path, expected, w, h in refs:
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Prepared benchmark reference missing or changed: {path}")
        verify_video(path, ffprobe, width=w, height=h, fps=original["fps"], frames=row["frames"])
    provenance = {}
    for group, variants, w, h in [
        ("native", native, original["width"], original["height"]),
        ("vsr", vsr, target["width"], target["height"]),
    ]:
        if len(variants) > 8:
            raise ValueError("Maximum eight variants per comparison group")
        for label, path in variants.items():
            if not SHOT_LABEL.fullmatch(label) or label == "original":
                raise ValueError(f"Invalid variant label {label}")
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Variant missing: {path}")
            verify_video(path, ffprobe, width=w, height=h,
                         fps=original["fps"], frames=row["frames"])
            provenance[group + ":" + label] = {
                **validate_provenance(path, row["native_sha256"], row["frames"],
                                     allow_unverified=allow_unverified),
                "file": str(path), "sha256": file_hash(path),
            }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".step3-evaluation-", dir=destination.parent) as td:
        staging = Path(td)
        reports = {}
        if native:
            reports["native"] = compare(refs[0][0], native, staging / "native")
        if vsr:
            reports["vsr"] = compare(refs[1][0], vsr, staging / "vsr")
        summary = {
            "schema_version": SCHEMA_VERSION,
            "shot": shot_name, "prepared_manifest": str(manifest_path),
            "native_reference_sha256": row["native_sha256"],
            "upscaled_lanczos_reference_sha256": row["lanczos_sha256"],
            "candidates": provenance, "diagnostics": reports,
            "warning": (
                "Native and VSR diagnostics intentionally separated by geometry. "
                "These are pixel-difference/texture proxies, not reference restoration "
                "scores or commercial-model license clearance."
            ),
        }
        (staging / "step3_evaluation.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if destination.exists():
            raise FileExistsError("Evaluation destination appeared during comparison")
        staging.rename(destination)
    return summary


def parse_variants(pairs: list[str]) -> dict[str, Path]:
    result = {}
    for item in pairs:
        name, sep, value = item.partition("=")
        if not sep or not SHOT_LABEL.fullmatch(name) or name == "original" or not value:
            raise ValueError(f"Variant must be LABEL=PATH: {item}")
        if name in result:
            raise ValueError(f"Duplicate variant label: {name}")
        result[name] = Path(value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    p = sub.add_parser("prepare", help="Generate fixed native and Lanczos-baseline excerpts")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--shot", action="append", required=True, help="NAME=START_SECONDS:FRAMES")
    p.add_argument("--target-width", type=int, default=1024)
    p.add_argument("--target-height", type=int, default=768)
    p.add_argument("--source-sha256")
    p.add_argument("--ffmpeg")
    p.add_argument("--ffprobe")
    e = sub.add_parser("evaluate", help="Separate native 1x and VSR comparisons")
    e.add_argument("--manifest", type=Path, required=True)
    e.add_argument("--shot", required=True)
    e.add_argument("--native", action="append", default=[], help="LABEL=PATH in native geometry")
    e.add_argument("--vsr", action="append", default=[], help="LABEL=PATH at VSR target size")
    e.add_argument("--output-dir", type=Path, required=True)
    e.add_argument("--allow-unverified", action="store_true",
                   help="Explicitly accept absent provenance; results flagged as unverified")
    e.add_argument("--ffprobe")
    args = parser.parse_args()
    ffprobe = executable(args.ffprobe, "ffprobe")
    if args.operation == "prepare":
        result = prepare(
            source=args.source, destination=args.output_dir, shots=parse_shots(args.shot),
            width=args.target_width, height=args.target_height,
            expected_source_sha256=args.source_sha256,
            ffmpeg=executable(args.ffmpeg, "ffmpeg"), ffprobe=ffprobe,
        )
    else:
        result = evaluate(
            manifest_path=args.manifest, shot_name=args.shot,
            native=parse_variants(args.native), vsr=parse_variants(args.vsr),
            destination=args.output_dir, allow_unverified=args.allow_unverified,
            ffprobe=ffprobe,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
