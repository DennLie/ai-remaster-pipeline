#!/usr/bin/env python3
"""Automated opt-in CPU benchmark grid on one synthetic-degraded film clip.

Runs existing *classical 1x* ARP methods on EXACTLY the damaged FFV1
input, then evaluates every variant against the held-out reference.
No downloaded weights and no generative interpretation. Algorithm
quality must not be ranked using one short scene alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import detail_enhance_video as detail
import restoration_evaluation as evaluation

METHODS = ("off", "unsharp", "cas", "multiscale", "richardson_lucy")
PRESETS = ("gentle", "balanced", "strong")


def parse_names(spec: str, allowed: tuple[str, ...]) -> list[str]:
    result = [x.strip() for x in spec.split(",")]
    if not result or len(set(result)) != len(result) or any(x not in allowed for x in result):
        raise ValueError(f"Expected unique comma-separated options from: {allowed}")
    return result


def run(*, manifest: Path, output_dir: Path,
        methods: list[str], presets: list[str], mix: int = 100,
        ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
        intermediate_profile: str = "lossless") -> dict:
    manifest = manifest.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Refusing to overwrite earlier algorithm-grid results")
    if not 0 <= mix <= 100:
        raise ValueError("Mix must be 0..100")
    if not methods or any(x not in METHODS for x in methods) or len(methods) != len(set(methods)):
        raise ValueError("Invalid/duplicate method")
    if not presets or any(x not in PRESETS for x in presets) or len(presets) != len(set(presets)):
        raise ValueError("Invalid/duplicate preset")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("reference_type") != "decoded_existing_transfer_not_pristine_camera_negative":
        raise ValueError("Only explicit synthetic full-reference manifests are supported")
    damaged = manifest.parent / "degraded.mkv"
    if not damaged.is_file() or evaluation.digest(damaged) != data.get("degraded_sha256"):
        raise ValueError("Synthetic degraded source checksum mismatch")
    expected = {k: data[k] for k in ("width", "height", "fps", "frames")}
    if evaluation.probe(damaged, ffprobe) != expected:
        raise ValueError("Damaged video geometry/fps/frame count mismatch")
    # Keep outputs under the final directory so sidecar provenance remains
    # valid after scoring. Fail rather than overwrite earlier results.
    output_dir.mkdir(parents=True, exist_ok=False)
    candidates = {}
    configs = []
    for method in methods:
        for preset in (["gentle"] if method == "off" else presets):
            label = f"{method}_{preset}"
            path = output_dir / f"{label}.mkv"
            result = detail.run(
                damaged, path, method=method, preset=preset, mix=mix,
                ffmpeg=ffmpeg, intermediate_profile=intermediate_profile,
            )
            if evaluation.probe(path, ffprobe) != expected:
                raise ValueError(f"Algorithm {label} changed scene geometry or cadence")
            candidates[label] = path
            configs.append({"label": label, "method": method, "preset": preset,
                            "mix": mix, "result": result})
    report = evaluation.evaluate(
        manifest=manifest, candidates=candidates, output_dir=output_dir / "evaluation",
        ffmpeg=ffmpeg, ffprobe=ffprobe,
        allow_unverified=False,
    )
    summary = {
        "schema_version": 1,
        "film": data["film"], "scene": data["scene"],
        "degradation": data["synthetic_degradation"],
        "input_sha256": evaluation.digest(damaged),
        "algorithms": configs,
        "evaluation_path": str(output_dir / "evaluation" / "evaluation.json"),
        "warning": "Full-reference scores quantify recovery of ADDED synthetic damage only; no historical fidelity ranking.",
    }
    (output_dir / "grid.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"grid": summary, "scores": report["scores"]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--methods", default="off,unsharp,cas,multiscale",
                   help="Comma-separated methods; add richardson_lucy explicitly")
    p.add_argument("--presets", default="gentle,balanced,strong")
    p.add_argument("--mix", type=int, default=100)
    p.add_argument("--intermediate-profile", default="lossless",
                   choices=("low", "medium", "high", "lossless"))
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    args = p.parse_args()
    print(json.dumps(run(
        manifest=args.manifest, output_dir=args.output_dir,
        methods=parse_names(args.methods, METHODS),
        presets=parse_names(args.presets, PRESETS),
        mix=args.mix, intermediate_profile=args.intermediate_profile,
        ffmpeg=evaluation.find_binary(args.ffmpeg),
        ffprobe=evaluation.find_binary(args.ffprobe),
    ), indent=2))


if __name__ == "__main__":
    main()
