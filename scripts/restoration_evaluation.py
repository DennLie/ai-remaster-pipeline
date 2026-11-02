#!/usr/bin/env python3
"""Reference-based ARP restoration benchmark (research, never historical ground truth).

SYNTHESIZE: apply seeded, explicit degradations to the SAME decoded clean
digital-transfer frames, create independent lossless FFV1 reference/damaged
videos, and record provenance. The input transfer is NOT pristine film.

EVALUATE: require the synthetic manifest, exact candidate frame count,
FPS and dimensions, and input-file hash sidecars (unless explicitly
overridden). Compare known-reference RGB PSNR, RGB SSIM, and a temporal
reference-error diagnostic. Scratches have known binary masks.

SUMMARIZE: mean paired candidate-minus-degraded score by scene, then film;
with >=3 independent films, give hierarchical film/scene bootstrap CIs.
No cross-task quality leaderboard or inferred historical accuracy.
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
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

SCHEMA = 1
KINDS = ("gaussian_blur", "read_noise", "temporal_flicker", "vertical_scratches")
PRESETS = ("gentle", "balanced", "strong")
SETTINGS = {
    "gaussian_blur": (0.65, 1.1, 1.65),
    "read_noise": (5.0, 10.0, 18.0),
    "temporal_flicker": (0.05, 0.10, 0.17),
    "vertical_scratches": (1, 2, 4),
}
NAME = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,40}$")
MAX_FRAMES = 240
MAX_RAW_BYTES = 1024 * 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fd:
        while block := fd.read(2 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def find_binary(name: str) -> str:
    result = shutil.which(name)
    if not result:
        raise FileNotFoundError(f"Executable unavailable: {name}")
    return result


def probe(path: Path, ffprobe: str) -> dict:
    p = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(p.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream in {path}")
    info = streams[0]
    w, h = int(info["width"]), int(info["height"])
    rate = Fraction(info["avg_frame_rate"])
    nominal = Fraction(info["r_frame_rate"])
    count = int(info["nb_read_frames"])
    if min(w, h) < 16 or rate <= 0 or nominal != rate or not 2 <= count <= MAX_FRAMES:
        raise ValueError(f"Unsupported frame rate/geometry/count: {path}")
    if w * h * count * 3 > MAX_RAW_BYTES:
        raise ValueError("Scene sample would exceed RGB memory limit")
    return {"width": w, "height": h,
            "fps": f"{rate.numerator}/{rate.denominator}", "frames": count}


def decode(path: Path, ffmpeg: str, info: dict):
    """Read exactly n RGB8 frames; fail on any truncated or extra frame."""
    w, h, count = info["width"], info["height"], info["frames"]
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        assert proc.stdout is not None
        for index in range(count):
            needed = w * h * 3
            data = bytearray()
            while len(data) < needed:
                chunk = proc.stdout.read(needed - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) != needed:
                raise ValueError(f"Video ended early at frame {index}: {path}")
            yield np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()
        if proc.stdout.read(1):
            raise ValueError(f"Extra decoded frames: {path}")
        proc.stdout.close()
        if proc.wait(timeout=60):
            raise RuntimeError(f"FFmpeg decode failed: {path}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def encode(path: Path, ffmpeg: str, info: dict, frames) -> None:
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{info['width']}x{info['height']}",
            "-framerate", info["fps"], "-i", "pipe:0",
            "-an", "-c:v", "ffv1", "-level", "3", "-pix_fmt", "bgr0",
            "-f", "matroska", str(path)]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    wrote = 0
    try:
        assert proc.stdin is not None
        for frame in frames:
            if (frame.shape != (info["height"], info["width"], 3)
                    or frame.dtype != np.uint8):
                raise ValueError("Unexpected input frame type/geometry")
            proc.stdin.write(frame.tobytes())
            wrote += 1
        proc.stdin.close()
        if proc.wait(timeout=90):
            raise RuntimeError("FFmpeg FFV1 encoder failed")
        if wrote != info["frames"]:
            raise ValueError(f"Expected {info['frames']} frames; got {wrote}")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()


def corrupt(frame: np.ndarray, *, kind: str, strength: str,
            index: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic *known* degradation of a decoded digital reference."""
    if kind not in KINDS or strength not in PRESETS:
        raise ValueError(f"Unknown kind/preset {kind}/{strength}")
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Expected RGB8 source")
    x = frame.astype(np.float32)
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    amount = SETTINGS[kind][PRESETS.index(strength)]
    # Seed is deliberately mixed with frame index, so film identity and
    # order, not execution scheduling, determine exact synthetic damage.
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    if kind == "gaussian_blur":
        x = cv2.GaussianBlur(x, (0, 0), sigmaX=amount, sigmaY=amount,
                             borderType=cv2.BORDER_REFLECT_101)
    elif kind == "read_noise":
        # Homoscedastic read noise only. It does not represent realistic
        # photon statistics, dust or actual historic negative grain.
        x += rng.normal(0.0, amount, size=x.shape).astype(np.float32)
    elif kind == "temporal_flicker":
        # Deterministic brightness oscillation. No claim it matches
        # projector/exposure-flicker statistics or intentional fades.
        x *= 1.0 + amount * math.sin((index + 1) * 2 * math.pi / 7.0)
    else:
        # Sparse approximately vertical synthetically drawn scratches.
        # Time-correlated center plus a bounded per-frame x displacement.
        count = int(amount)
        for line in range(count):
            base = int(w * (line + 1) / (count + 1))
            shift = int(rng.integers(-2, 3))
            center = min(w - 1, max(0, base + shift))
            radius = 1 if strength != "strong" else 2
            x0, x1 = max(0, center - radius), min(w, center + radius + 1)
            mask[:, x0:x1] = 255
            x[:, x0:x1, :] = 248.0 if line % 2 == 0 else 5.0
    return np.clip(np.rint(x), 0, 255).astype(np.uint8), mask


def synthesize(*, source: Path, output_dir: Path, film: str, scene: str,
               kind: str, preset: str, seed: int = 2026,
               ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if kind not in KINDS or preset not in PRESETS:
        raise ValueError("Unknown synthetic degradation")
    if not NAME.fullmatch(film) or not NAME.fullmatch(scene):
        raise ValueError("Unsafe film or scene identifier")
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be 0..2^32-1")
    if not source.is_file() or output_dir.exists():
        raise ValueError("Source must exist; output directory must not exist")
    source_info = probe(source, ffprobe)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # The source is re-encoded as a reference in the SAME RGB8/FFV1
    # path as damaged video to avoid encoder asymmetry.
    with tempfile.TemporaryDirectory(prefix=".arp-eval-", dir=output_dir.parent) as td:
        stage = Path(td)
        clean = stage / "reference.mkv"
        degraded = stage / "degraded.mkv"
        masks = stage / "masks"
        if kind == "vertical_scratches":
            masks.mkdir()
        src_frames = list(decode(source, ffmpeg, source_info))
        # A scene-specific seed prevents identical static scratch/noise
        # patterns across films even if the same user seed is supplied.
        combined_seed = int.from_bytes(
            hashlib.sha256(f"{film}/{scene}/{seed}".encode()).digest()[:4], "big")
        damaged_frames = []
        affected_fraction = []
        for i, frame in enumerate(src_frames):
            output, mask = corrupt(frame, kind=kind, strength=preset,
                                   index=i, seed=combined_seed)
            damaged_frames.append(output)
            affected_fraction.append(float(np.mean(mask > 0)))
            if kind == "vertical_scratches":
                if not cv2.imwrite(str(masks / f"{i:06d}.png"), mask):
                    raise IOError("Could not write scratch ground truth mask")
        encode(clean, ffmpeg, source_info, src_frames)
        encode(degraded, ffmpeg, source_info, damaged_frames)
        for f in (clean, degraded):
            if probe(f, ffprobe) != source_info:
                raise ValueError("Encoded synthetic video changed geometry/cadence/frames")
        report = {
            "schema_version": SCHEMA,
            "reference_type": "decoded_existing_transfer_not_pristine_camera_negative",
            "synthetic_degradation": kind, "preset": preset,
            "strength_parameter": SETTINGS[kind][PRESETS.index(preset)],
            "film": film, "scene": scene, "user_seed": seed,
            "effective_scene_seed": combined_seed,
            "source_path": str(source), "source_sha256": digest(source),
            "reference_path": "reference.mkv", "reference_sha256": digest(clean),
            "degraded_path": "degraded.mkv", "degraded_sha256": digest(degraded),
            "mask_dir": "masks" if kind == "vertical_scratches" else None,
            "synthetic_mask_fraction": float(np.mean(affected_fraction)),
            **source_info,
            "limitations": (
                "This is a controlled degradation of the decoded historical transfer. "
                "A known synthetic target is not a ground-truth original film image. "
                "No audio, no original-film or commercial-rights certification."
            ),
        }
        (stage / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        if output_dir.exists():
            raise FileExistsError("Output directory appeared while preparing")
        stage.rename(output_dir)
    return report


def _read_frames(path: Path, ffmpeg: str, meta: dict):
    return list(decode(path, ffmpeg, meta))


def _metrics(reference: list, variant: list, mask_dir: Path | None = None) -> dict:
    if len(reference) != len(variant):
        raise ValueError("Frame count mismatch")
    mse_sum, ssim_sum, n = 0., 0., 0
    masked_mse, masked_pixels = 0., 0
    temporal_mse, temporal_n = 0., 0
    prev_ref, prev_var = None, None
    for i, (a, b) in enumerate(zip(reference, variant)):
        if a.shape != b.shape:
            raise ValueError("Frame geometry mismatch")
        ref = a.astype(np.float32) / 255.
        pred = b.astype(np.float32) / 255.
        error = pred - ref
        mse_sum += float(np.mean(error ** 2))
        ssim_sum += float(structural_similarity(
            a, b, data_range=255, channel_axis=2, gaussian_weights=True,
            sigma=1.5, use_sample_covariance=False))
        if mask_dir:
            mask = cv2.imread(str(mask_dir / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.shape != a.shape[:2]:
                raise ValueError("Missing/mismatched synthetic scratch mask")
            region = mask > 127
            if np.any(region):
                masked_mse += float(np.sum(np.mean(error ** 2, axis=2)[region]))
                masked_pixels += int(np.sum(region))
        if prev_ref is not None:
            # Temporal *reference-error* diagnostic: unlike warp error
            # on real films this has a clean paired reference. Distinct
            # from optical-flow based motion preservation measurement.
            dt_error = (pred - prev_var) - (ref - prev_ref)
            temporal_mse += float(np.mean(dt_error ** 2))
            temporal_n += 1
        prev_ref, prev_var = ref, pred
        n += 1
    if not n:
        raise ValueError("No frames to score")
    mse = mse_sum / n
    result = {
        "mse_rgb_linear_0_1_range": mse,
        "psnr_rgb_db": None if mse == 0 else float(10 * math.log10(1 / mse)),
        "perfect_rgb_match": mse == 0,
        "ssim_rgb": ssim_sum / n,
        "temporal_reference_error_rmse": (
            None if not temporal_n else float(math.sqrt(temporal_mse / temporal_n))
        ),
        "frames": n,
    }
    if mask_dir:
        result["known_scratch_mask_mse"] = (
            masked_mse / masked_pixels if masked_pixels else None)
        result["known_scratch_mask_pixel_count"] = masked_pixels
    return result


def evaluate(*, manifest: Path, candidates: dict[str, Path],
             output_dir: Path, ffmpeg: str = "ffmpeg",
             ffprobe: str = "ffprobe",
             allow_unverified: bool = False) -> dict:
    from step3_benchmark import validate_provenance
    manifest = manifest.resolve()
    data = json.loads(manifest.read_text())
    root = manifest.parent
    if data.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported synthetic manifest schema")
    if data.get("reference_type") != "decoded_existing_transfer_not_pristine_camera_negative":
        raise ValueError("Not a known-source synthetic experiment")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Evaluation output directory exists")
    clean, degraded = root / "reference.mkv", root / "degraded.mkv"
    for path, checksum in ((clean, data["reference_sha256"]),
                           (degraded, data["degraded_sha256"])):
        if not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Reference/degradation checksum mismatch: {path}")
    metadata = {k: data[k] for k in ("width", "height", "fps", "frames")}
    if probe(clean, ffprobe) != metadata or probe(degraded, ffprobe) != metadata:
        raise ValueError("Synthetic video changed, missing or misaligned")
    maskdir = root / "masks" if data["synthetic_degradation"] == "vertical_scratches" else None
    results = {}
    all_paths = {"degraded_baseline": degraded}
    for label, path in candidates.items():
        if not NAME.fullmatch(label) or label == "degraded_baseline":
            raise ValueError(f"Unsafe or reserved algorithm label: {label}")
        path = path.resolve()
        if not path.is_file() or probe(path, ffprobe) != metadata:
            raise ValueError(f"Candidate not frame-aligned with synthetic reference: {path}")
        provenance = validate_provenance(
            path, data["degraded_sha256"], data["frames"],
            allow_unverified=allow_unverified,
        )
        results[label] = {"provenance": provenance, "file_sha256": digest(path)}
        all_paths[label] = path
    reference_frames = _read_frames(clean, ffmpeg, metadata)
    scores = {}
    for label, path in all_paths.items():
        outputs = _read_frames(path, ffmpeg, metadata)
        scores[label] = _metrics(reference_frames, outputs, maskdir)
    base = scores["degraded_baseline"]
    for label, metrics in scores.items():
        if label == "degraded_baseline":
            continue
        metrics["delta_psnr_db_vs_degraded"] = (
            metrics["psnr_rgb_db"] - base["psnr_rgb_db"]
            if metrics["psnr_rgb_db"] is not None and base["psnr_rgb_db"] is not None
            else None
        )
        metrics["delta_ssim_vs_degraded"] = metrics["ssim_rgb"] - base["ssim_rgb"]
    report = {
        "schema_version": SCHEMA,
        "evaluation_type": "synthetic_full_reference",
        "film": data["film"], "scene": data["scene"],
        "degradation": data["synthetic_degradation"],
        "preset": data["preset"], "frames": data["frames"],
        "synthetic_manifest_sha256": digest(manifest),
        "candidate_provenance": results, "scores": scores,
        "important": (
            "Reference is a pre-existing decoded transfer, not the camera negative. "
            "Scores measure recovery of deliberately introduced defects only. "
            "Motion/frame errors require independent review. No automatic quality winner."
        ),
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arp-scores-", dir=output_dir.parent) as td:
        stage = Path(td)
        (stage / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        if output_dir.exists():
            raise FileExistsError("Evaluation output appeared mid-run")
        stage.rename(output_dir)
    return report


def summarize(paths: list[Path], *, n_bootstrap: int = 1000,
              seed: int = 2026) -> dict:
    if not paths or not 100 <= n_bootstrap <= 20000:
        raise ValueError("At least one evaluation and 100..20000 resamples required")
    rng = np.random.default_rng(seed)
    # Paired delta per *scene* is the statistical unit, not video frames.
    observed = defaultdict(lambda: defaultdict(dict))
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("evaluation_type") != "synthetic_full_reference":
            raise ValueError(f"Not a full-reference evaluation: {path}")
        film, scene, deg = report["film"], report["scene"], report["degradation"]
        for label, metrics in report["scores"].items():
            if label == "degraded_baseline":
                continue
            value = metrics.get("delta_psnr_db_vs_degraded")
            if value is None or not math.isfinite(value):
                continue
            key = (deg, label)
            if scene in observed[key][film]:
                raise ValueError(f"Duplicate film/scene/algorithm {film}/{scene}/{label}")
            observed[key][film][scene] = float(value)
    outcome = []
    for (degradation, algorithm), per_film in sorted(observed.items()):
        films = sorted(per_film)
        # Equal film weight regardless of length / number of sampled frames.
        film_means = [float(np.mean(list(per_film[f].values()))) for f in films]
        point = float(np.mean(film_means))
        entry = {
            "degradation": degradation, "algorithm": algorithm,
            "mean_scene_paired_psnr_delta_db": point,
            "films": len(films),
            "scenes": sum(len(per_film[f]) for f in films),
            "film_means_db": {f: v for f, v in zip(films, film_means)},
            "ci95": None,
            "exploratory_only": len(films) < 3,
        }
        if len(films) >= 3:
            resamples = []
            for _ in range(n_bootstrap):
                sampled_films = rng.choice(films, size=len(films), replace=True)
                film_sample_means = []
                for f in sampled_films:
                    scene_values = list(per_film[f].values())
                    reselected = rng.choice(scene_values, size=len(scene_values), replace=True)
                    film_sample_means.append(float(np.mean(reselected)))
                resamples.append(float(np.mean(film_sample_means)))
            entry["ci95"] = list(map(float, np.quantile(resamples, [0.025, 0.975])))
        outcome.append(entry)
    return {
        "schema_version": SCHEMA,
        "method": "equal-film weighting; paired scene scores; hierarchical film/scene bootstrap",
        "bootstrap_resamples": n_bootstrap, "seed": seed,
        "results": outcome,
        "caveat": (
            "Intervals are descriptive bootstrap uncertainty over sampled films/scenes, "
            "not proofs of algorithm superiority, not across-mode rankings, and "
            "not calibrated historical image authenticity."
        ),
    }


def parse_pairs(pairs: list[str]) -> dict[str, Path]:
    result = {}
    for entry in pairs:
        name, sep, path = entry.partition("=")
        if not sep or not NAME.fullmatch(name) or not path or name in result:
            raise ValueError(f"Expected unique LABEL=PATH: {entry}")
        result[name] = Path(path)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("synthesize")
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--output-dir", type=Path, required=True)
    a.add_argument("--film", required=True)
    a.add_argument("--scene", required=True)
    a.add_argument("--kind", choices=KINDS, required=True)
    a.add_argument("--preset", choices=PRESETS, default="gentle")
    a.add_argument("--seed", type=int, default=2026)
    a.add_argument("--ffmpeg", default="ffmpeg")
    a.add_argument("--ffprobe", default="ffprobe")
    b = sub.add_parser("evaluate")
    b.add_argument("--manifest", type=Path, required=True)
    b.add_argument("--variant", action="append", default=[])
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--allow-unverified", action="store_true")
    b.add_argument("--ffmpeg", default="ffmpeg")
    b.add_argument("--ffprobe", default="ffprobe")
    c = sub.add_parser("summarize")
    c.add_argument("--evaluation", type=Path, action="append", required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--bootstrap", type=int, default=1000)
    c.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.action == "synthesize":
        result = synthesize(source=args.source, output_dir=args.output_dir,
                            film=args.film, scene=args.scene, kind=args.kind,
                            preset=args.preset, seed=args.seed,
                            ffmpeg=find_binary(args.ffmpeg), ffprobe=find_binary(args.ffprobe))
    elif args.action == "evaluate":
        result = evaluate(manifest=args.manifest, candidates=parse_pairs(args.variant),
                          output_dir=args.output_dir,
                          ffmpeg=find_binary(args.ffmpeg), ffprobe=find_binary(args.ffprobe),
                          allow_unverified=args.allow_unverified)
    else:
        result = summarize(args.evaluation, n_bootstrap=args.bootstrap, seed=args.seed)
        if args.output.exists():
            raise FileExistsError("Summary output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
