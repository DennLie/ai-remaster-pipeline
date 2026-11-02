#!/usr/bin/env python3
"""Research-only Step 4 synthetic camera-jitter test.

Create known per-frame translational camera jitter with a fixed seed,
keeping an aligned lossless digital-transfer reference. Score candidate
outputs with a conservative central-ROI reference comparison and
phase-correlation translation diagnostics. Temporal flow, real physical
camera motion and authentic original-film stability are NOT measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

import restoration_evaluation as ev
from step3_benchmark import validate_provenance

MAX_JITTER = {"gentle": 1.5, "balanced": 3.0, "strong": 5.0}


def prepare(*, source: Path, output_dir: Path, film: str, scene: str,
            preset: str = "gentle", seed: int = 2026,
            ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    if preset not in MAX_JITTER or not ev.NAME.fullmatch(film) or not ev.NAME.fullmatch(scene):
        raise ValueError("Invalid jitter strength / film / scene")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be 0..2^32-1")
    source, output_dir = source.resolve(), output_dir.resolve()
    if not source.is_file() or output_dir.exists():
        raise ValueError("Source must exist and output directory must be unused")
    info = ev.probe(source, ffprobe)
    if min(info["width"], info["height"]) < 48:
        raise ValueError("Insufficient geometry for stable central ROI")
    crop = max(8, int(round(min(info["width"], info["height"]) * 0.17)))
    if crop * 2 >= min(info["width"], info["height"]) // 2:
        raise ValueError("Too little usable central image area")
    original_frames = list(ev.decode(source, ffmpeg, info))
    scene_seed = int.from_bytes(hashlib.sha256(f"{film}/{scene}/{seed}/jitter".encode()).digest()[:4], "big")
    rng = np.random.default_rng(scene_seed)
    amplitude = MAX_JITTER[preset]
    translations = []
    jitter_frames = []
    for rgb in original_frames:
        dx, dy = [float(x) for x in rng.uniform(-amplitude, amplitude, size=2)]
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        transformed = cv2.warpAffine(
            rgb, matrix, (info["width"], info["height"]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        jitter_frames.append(transformed)
        translations.append({"dx_pixels": dx, "dy_pixels": dy})
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arp-jitter-", dir=output_dir.parent) as td:
        temp = Path(td)
        ref, jitter = temp / "reference.mkv", temp / "jitter.mkv"
        ev.encode(ref, ffmpeg, info, original_frames)
        ev.encode(jitter, ffmpeg, info, jitter_frames)
        for path in (ref, jitter):
            if ev.probe(path, ffprobe) != info:
                raise ValueError("Encoded jitter clips did not preserve geometry/cadence/frames")
        manifest = {
            "schema_version": 1, "experiment": "controlled_translation_jitter",
            "film": film, "scene": scene, "preset": preset, "seed": seed,
            "effective_scene_seed": scene_seed, "max_shift_pixels": amplitude,
            "source_sha256": ev.digest(source),
            "reference_sha256": ev.digest(ref),
            "jitter_sha256": ev.digest(jitter),
            "translation_per_frame": translations,
            "crop_margin_pixels": crop, **info,
            "limitations": "Translation-only synthetic jitter. Original is decoded transfer, not true camera-negative ground truth. Central ROI ignores border losses; report ROI fraction. This is not general video stabilization evaluation.",
        }
        (temp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if output_dir.exists():
            raise FileExistsError("Output folder appeared during processing")
        temp.rename(output_dir)
    return manifest


def _pair_metrics(ref: list[np.ndarray], frames: list[np.ndarray], crop: int) -> dict:
    if len(ref) != len(frames):
        raise ValueError("Misaligned frame count")
    shifts = []
    mse = []
    responses = []
    for orig, frame in zip(ref, frames):
        if orig.shape != frame.shape:
            raise ValueError("Misaligned frame geometry")
        # Inner ROI removes the known synthetic translation's blank borders.
        a = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)[crop:-crop, crop:-crop].astype(np.float32)
        b = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[crop:-crop, crop:-crop].astype(np.float32)
        # Hanning weighting limits artificial ROI edge artifacts.
        window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(a, b, window)
        if not (math.isfinite(dx) and math.isfinite(dy)):
            raise ValueError("Invalid motion estimate")
        shifts.append((dx, dy))
        responses.append(float(response))
        diff = (orig[crop:-crop, crop:-crop].astype(np.float32) -
                frame[crop:-crop, crop:-crop].astype(np.float32)) / 255.0
        mse.append(float(np.mean(diff ** 2)))
    arr = np.asarray(shifts, np.float64)
    magnitude = np.linalg.norm(arr, axis=1)
    return {
        "reference_roi_mse_rgb": float(np.mean(mse)),
        "estimated_translation_rmse_pixels": float(np.sqrt(np.mean(magnitude ** 2))),
        "estimated_translation_p95_pixels": float(np.percentile(magnitude, 95)),
        "mean_phase_correlation_response": float(np.mean(responses)),
        "frame_translation_estimates": [{"dx": float(a), "dy": float(b)} for a, b in shifts],
        "frames": len(shifts),
    }


def evaluate(*, manifest: Path, candidate: Path | None,
             output_dir: Path, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
             allow_unverified: bool = False) -> dict:
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    m = json.loads(manifest.read_text())
    if m.get("schema_version") != 1 or m.get("experiment") != "controlled_translation_jitter":
        raise ValueError("Invalid controlled jitter manifest")
    if output_dir.exists():
        raise FileExistsError("Evaluation output folder exists")
    root = manifest.parent
    ref, jitter = root / "reference.mkv", root / "jitter.mkv"
    for path, key in ((ref, "reference_sha256"), (jitter, "jitter_sha256")):
        if not path.is_file() or ev.digest(path) != m.get(key):
            raise ValueError(f"Corrupted jitter/reference source: {path}")
    info = {k: m[k] for k in ("width", "height", "fps", "frames")}
    for f in (ref, jitter):
        if ev.probe(f, ffprobe) != info:
            raise ValueError("Mismatch in sample frame count/geometry/cadence")
    provenance = None
    if candidate is not None:
        candidate = candidate.resolve()
        if not candidate.is_file() or ev.probe(candidate, ffprobe) != info:
            raise ValueError("Candidate frame count/geometry/cadence mismatch")
        provenance = validate_provenance(candidate, m["jitter_sha256"],
                                         m["frames"], allow_unverified=allow_unverified)
    clean = list(ev.decode(ref, ffmpeg, info))
    jitter_frames = list(ev.decode(jitter, ffmpeg, info))
    c = int(m["crop_margin_pixels"])
    scores = {"damaged_baseline": _pair_metrics(clean, jitter_frames, c)}
    if candidate is not None:
        scores["candidate"] = _pair_metrics(clean, list(ev.decode(candidate, ffmpeg, info)), c)
    report = {
        "schema_version": 1, "experiment": "controlled_translation_jitter",
        "film": m["film"], "scene": m["scene"], "frames": m["frames"],
        "crop_margin_pixels": c,
        "roi_fraction_of_full_frame": ((m["width"] - 2*c) * (m["height"] - 2*c)) / (m["width"] * m["height"]),
        "candidate_provenance": provenance, "metrics": scores,
        "known_added_translations": m["translation_per_frame"],
        "warning": "Scores are phase-correlation translation PROXIES relative to the known digital reference; unreliable in flat/moving/zoomed footage. No general stabilization ground truth, no blur or edge-crop assessment outside ROI.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--film", required=True)
    a.add_argument("--scene", required=True)
    a.add_argument("--preset", choices=tuple(MAX_JITTER), default="gentle")
    a.add_argument("--seed", type=int, default=2026)
    b = sub.add_parser("evaluate")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--candidate", type=Path)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--allow-unverified", action="store_true")
    for subparser in (a, b):
        subparser.add_argument("--ffmpeg", default="ffmpeg")
        subparser.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args()
    if args.action == "prepare":
        result = prepare(source=args.source, output_dir=args.output_dir,
                         film=args.film, scene=args.scene,
                         preset=args.preset, seed=args.seed,
                         ffmpeg=ev.find_binary(args.ffmpeg), ffprobe=ev.find_binary(args.ffprobe))
    else:
        result = evaluate(manifest=args.manifest, candidate=args.candidate,
                          output_dir=args.output_dir, allow_unverified=args.allow_unverified,
                          ffmpeg=ev.find_binary(args.ffmpeg), ffprobe=ev.find_binary(args.ffprobe))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
