#!/usr/bin/env python3
"""CPU-only scientific probes for ARP Steps 1 (deflicker), 2 (denoise), 5 (scratch masks).

Tasks have distinct metrics and outputs; no "overall restoration score".
Uses an EXACT synthetic experiment from restoration_evaluation.py with
known digital-reference frames. All processing paths preserve the
original, reject stale source hashes, and use lossless FFV1 intermediates.

Step 1 is explicitly an experimental FFmpeg deflicker baseline, not an
already integrated production deflicker stage. Step 2 invokes ARP's real
denoise_video.run. Step 5 scores the actual ai_scratch_mask detector
against synthetic scratch GT, NOT DART/ProPainter inpainting fidelity.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import restoration_evaluation as ev
import denoise_video
from cleanup_video import ai_scratch_mask
from common import write_signature

DEF_MODE = ("am", "median")
DEF_SIZES = {"gentle": 3, "balanced": 5, "strong": 9}
DENOISE_METHODS = ("off", "hqdn3d", "nlmeans", "bm3d")
STRENGTHS = ("gentle", "balanced", "strong")


def synthetic_input(manifest: Path, *, ffprobe: str) -> tuple[dict, Path]:
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if (m.get("schema_version") != 1 or
        m.get("reference_type") != "decoded_existing_transfer_not_pristine_camera_negative"):
        raise ValueError("Expected known-source synthetic reference manifest")
    damaged = manifest.parent / "degraded.mkv"
    clean = manifest.parent / "reference.mkv"
    expected = {k: m[k] for k in ("width", "height", "fps", "frames")}
    for path, checksum in ((damaged, m["degraded_sha256"]),
                           (clean, m["reference_sha256"])):
        if not path.is_file() or ev.digest(path) != checksum:
            raise ValueError(f"Source/reference SHA mismatch: {path}")
        if ev.probe(path, ffprobe) != expected:
            raise ValueError(f"Source/reference frames or geometry mismatch: {path}")
    return m, damaged


def deflicker_video(source: Path, output: Path, *,
                    mode: str, preset: str, ffmpeg: str, ffprobe: str) -> dict:
    if mode not in DEF_MODE or preset not in DEF_SIZES:
        raise ValueError("Unsupported experimental FFmpeg deflicker preset")
    source, output = source.resolve(), output.resolve()
    if output.exists() or output == source:
        raise FileExistsError("Deflicker output must be new and distinct")
    original = ev.probe(source, ffprobe)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".deflicker-", dir=output.parent) as td:
        tmp = Path(td) / "tmp.mkv"
        args = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:v:0", "-an",
            "-vf", f"deflicker=size={DEF_SIZES[preset]}:mode={mode}",
            "-fps_mode", "passthrough", "-c:v", "ffv1",
            "-level", "3", "-pix_fmt", "bgr0", "-f", "matroska", str(tmp),
        ]
        subprocess.run(args, check=True, capture_output=True, text=True, timeout=180)
        if ev.probe(tmp, ffprobe) != original:
            raise ValueError("Deflicker changed frame count/fps/geometry")
        if output.exists():
            raise FileExistsError("Deflicker target appeared during processing")
        tmp.replace(output)
    # Compatible source hash for restoration_evaluation.evaluate.
    write_signature(output, {
        "tool": "scientific_stage_adapters.py/experimental_deflicker",
        "version": 1, "method": "ffmpeg_deflicker", "mode": mode,
        "preset": preset, "source_fingerprint": {
            "sha256": ev.digest(source), "size": source.stat().st_size,
        }, "ffmpeg": ffmpeg,
    })
    return {"output": str(output), "filter": f"deflicker=size={DEF_SIZES[preset]}:mode={mode}"}


def run_grid(*, manifest: Path, output_dir: Path,
             method_names: list[str], presets: list[str],
             deflicker_mode: str = "am",
             ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Grid output directory exists; no overwrite")
    m, damaged = synthetic_input(manifest, ffprobe=ffprobe)
    if not method_names or len(method_names) != len(set(method_names)):
        raise ValueError("Methods required; duplicates rejected")
    if not presets or len(presets) != len(set(presets)) or any(p not in STRENGTHS for p in presets):
        raise ValueError("Unknown/duplicate strength")
    if m["synthetic_degradation"] == "temporal_flicker":
        if any(n != "ffmpeg_deflicker" and n != "off" for n in method_names):
            raise ValueError("Deflicker experiment accepts off/ffmpeg_deflicker only")
    elif m["synthetic_degradation"] == "read_noise":
        if any(n not in DENOISE_METHODS for n in method_names):
            raise ValueError("Denoise experiment accepts existing off/HQDN3D/NLMeans/BM3D only")
    else:
        raise ValueError("Grid only supports temporal_flicker or read_noise; no task mixing")
    if "ffmpeg_deflicker" in method_names and deflicker_mode not in DEF_MODE:
        raise ValueError("Unsupported deflicker mode")
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates: dict[str, Path] = {}
    configurations = []
    for method in method_names:
        for strength in (["gentle"] if method == "off" else presets):
            label = method + "_" + strength
            dest = output_dir / (label + ".mkv")
            if method == "ffmpeg_deflicker":
                record = deflicker_video(damaged, dest, mode=deflicker_mode,
                                         preset=strength, ffmpeg=ffmpeg, ffprobe=ffprobe)
            else:
                record = denoise_video.run(damaged, dest, method=method, preset=strength,
                                           ffmpeg=ffmpeg, intermediate_profile="lossless")
            expected = {k: m[k] for k in ("width", "height", "fps", "frames")}
            if ev.probe(dest, ffprobe) != expected:
                raise ValueError("Denoise or deflicker output lost frames/changed geometry")
            candidates[label] = dest
            configurations.append({"algorithm": label, "settings": record})
    report = ev.evaluate(manifest=manifest, candidates=candidates,
                         output_dir=output_dir / "evaluation",
                         ffmpeg=ffmpeg, ffprobe=ffprobe)
    result = {
        "schema_version": 1, "film": m["film"], "scene": m["scene"],
        "synthetic_degradation": m["synthetic_degradation"],
        "degraded_sha256": ev.digest(damaged),
        "algorithms": configurations,
        "evaluation_path": "evaluation/evaluation.json",
        "warning": "Only controlled synthetic reference accuracy; not original archive truth or human preference.",
    }
    (output_dir / "grid.json").write_text(json.dumps(result, indent=2) + "\n")
    return {"grid": result, "scores": report["scores"]}


def mask_scores(pred: list[np.ndarray], truth: list[np.ndarray]) -> dict:
    if not pred or len(pred) != len(truth):
        raise ValueError("Mask frame count mismatch")
    tp = fp = fn = tn = 0
    for p, t in zip(pred, truth):
        if p.shape != t.shape:
            raise ValueError("Mask shape mismatch")
        a, b = p > 0, t > 0
        tp += int(np.count_nonzero(a & b))
        fp += int(np.count_nonzero(a & ~b))
        fn += int(np.count_nonzero(~a & b))
        tn += int(np.count_nonzero(~a & ~b))
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {
        "tp_pixels": tp, "fp_pixels": fp, "fn_pixels": fn, "tn_pixels": tn,
        "pixel_precision": precision, "pixel_recall": recall,
        "pixel_f1": f1,
        "detected_fraction": (tp + fp) / max(1, tp + fp + fn + tn),
        "ground_truth_fraction": (tp + fn) / max(1, tp + fp + fn + tn),
    }


def evaluate_scratch_mask(*, manifest: Path, output_dir: Path,
                          sensitivity: float = 0.65,
                          ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    manifest, output_dir = manifest.resolve(), output_dir.resolve()
    if not 0 <= sensitivity <= 1:
        raise ValueError("Scratch detector sensitivity must be between 0 and 1")
    if output_dir.exists():
        raise FileExistsError("Mask evaluation output already exists")
    m, damaged = synthetic_input(manifest, ffprobe=ffprobe)
    if m["synthetic_degradation"] != "vertical_scratches" or m.get("mask_dir") != "masks":
        raise ValueError("Scratch detector needs synthetic vertical_scratches and masks")
    meta = {k: m[k] for k in ("width", "height", "fps", "frames")}
    generated, known = [], []
    frames = list(ev.decode(damaged, ffmpeg, meta))
    for i, rgb in enumerate(frames):
        expected_mask = cv2.imread(str(manifest.parent / "masks" / f"{i:06d}.png"),
                                   cv2.IMREAD_GRAYSCALE)
        if expected_mask is None or expected_mask.shape != rgb.shape[:2]:
            raise ValueError(f"Missing/mismatched ground-truth scratch mask {i}")
        estimate = ai_scratch_mask(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                   sensitivity=sensitivity)
        generated.append(estimate)
        known.append(expected_mask)
    result = {
        "schema_version": 1, "task": "scratch_detection_only_not_inpainting",
        "film": m["film"], "scene": m["scene"],
        "synthetic_manifest_sha256": ev.digest(manifest),
        "degraded_sha256": m["degraded_sha256"],
        "detector": "cleanup_video.ai_scratch_mask",
        "sensitivity": sensitivity, "frames": len(generated),
        "metrics": mask_scores(generated, known),
        "warning": "Pixel-level synthetic ground truth, not real-film scratch fidelity; ProPainter repair and DART not measured.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "mask_evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parse_methods(value: str) -> list[str]:
    items = [part.strip() for part in value.split(",")]
    if not items or any(not item for item in items) or len(set(items)) != len(items):
        raise ValueError("Expected nonempty unique comma-separated method names")
    return items


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--manifest", required=True, type=Path)
    g.add_argument("--output-dir", required=True, type=Path)
    g.add_argument("--methods", default="off,ffmpeg_deflicker")
    g.add_argument("--presets", default="gentle,balanced,strong")
    g.add_argument("--deflicker-mode", choices=DEF_MODE, default="am")
    g.add_argument("--ffmpeg", default="ffmpeg")
    g.add_argument("--ffprobe", default="ffprobe")
    s = sub.add_parser("scratch-mask")
    s.add_argument("--manifest", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--sensitivity", type=float, default=0.65)
    s.add_argument("--ffmpeg", default="ffmpeg")
    s.add_argument("--ffprobe", default="ffprobe")
    a = p.parse_args()
    if a.action == "grid":
        result = run_grid(manifest=a.manifest, output_dir=a.output_dir,
                          method_names=parse_methods(a.methods),
                          presets=parse_methods(a.presets), deflicker_mode=a.deflicker_mode,
                          ffmpeg=ev.find_binary(a.ffmpeg), ffprobe=ev.find_binary(a.ffprobe))
    else:
        result = evaluate_scratch_mask(
            manifest=a.manifest, output_dir=a.output_dir,
            sensitivity=a.sensitivity,
            ffmpeg=ev.find_binary(a.ffmpeg), ffprobe=ev.find_binary(a.ffprobe))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
