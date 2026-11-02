#!/usr/bin/env python3
"""Experimental multi-film Step 1-5 CPU benchmark on verified local film clips.

Uses the actual parent-PR ARP adapters, not simulated algorithm results:
Step1 film_deflicker; Step2 ARP denoise; Step3 native detail; Step4
libvidstab; Step5 scratch-mask detector plus labeled Telea control.

The evaluated clean references are *existing digitized film transfers*
to which we add *known synthetic damage*. This does not estimate
historical fidelity or establish commercial rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import evaluate_steps_1to5 as other
import restoration_evaluation as ev
import run_scientific_classical_grid as detail

TASKS = {
    "deflicker": "temporal_flicker",
    "denoise": "read_noise",
    "detail": "gaussian_blur",
    "stabilize": "camera_jitter",
    "scratch_masks": "vertical_scratches",
}
LABEL = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


def checked_member(root: Path, name: str) -> Path:
    if not LABEL.fullmatch(name):
        raise ValueError(f"Unsafe experiment label: {name}")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes corpus root")
    return path


def source_shots(corpus_root: Path, films: list[str], scenes_per_film: int) -> list[dict]:
    if (not 1 <= len(films) <= 12 or len(films) != len(set(films)) or
        not 1 <= scenes_per_film <= 4):
        raise ValueError("Expected 1–12 distinct film IDs, 1–4 scenes per film")
    entries = []
    for film in films:
        folder = checked_member(corpus_root, film)
        manifest = folder / "corpus_manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"Film corpus manifest missing: {manifest}")
        report = json.loads(manifest.read_text(encoding="utf-8"))
        if report.get("id") != film or report.get("schema_version") != 1:
            raise ValueError(f"Film corpus identity mismatch: {film}")
        source = report.get("source_provenance", {})
        if not isinstance(source.get("downloaded_sha256"), str):
            raise ValueError(f"Missing source transfer SHA-256 for {film}")
        choices = report.get("clips", [])[:scenes_per_film]
        if len(choices) < scenes_per_film:
            raise ValueError(f"Not enough curated scenes in {film}")
        for shot in choices:
            name = shot["name"]
            folder_path = checked_member(folder, name)
            original = folder_path / "original_native.mkv"
            expected = shot.get("files", {}).get("original_native.mkv", {}).get("sha256")
            if not original.is_file() or ev.digest(original) != expected:
                raise ValueError(f"Native film scene changed or missing: {original}")
            entries.append({
                "film": film, "scene": name, "source": original,
                "sha256": expected, "source_transfer_sha256": source["downloaded_sha256"],
                "rights_status": report.get("rights_status"),
            })
    return entries


def execute(*, corpus_root: Path, output_root: Path, films: list[str],
            stages: list[str], scenes_per_film: int = 1, seed: int = 2026,
            ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    corpus_root, output_root = corpus_root.resolve(), output_root.resolve()
    if output_root.exists():
        raise FileExistsError("Evaluation target exists; no existing benchmark is overwritten")
    if not stages or len(stages) != len(set(stages)) or any(s not in TASKS for s in stages):
        raise ValueError("Unsupported/duplicate evaluation stage")
    sources = source_shots(corpus_root, films, scenes_per_film)
    # This preparatory validation runs before the first frame is modified.
    for row in sources:
        ev.probe(row["source"], ffprobe)
    output_root.mkdir(parents=True, exist_ok=False)
    evaluation_paths = []
    completed = []
    for item in sources:
        for stage in stages:
            name = TASKS[stage]
            stem = output_root / item["film"] / item["scene"] / stage
            synthetic = stem / "source"
            out = stem / "algorithms"
            ev.synthesize(
                source=item["source"], output_dir=synthetic,
                film=item["film"], scene=item["scene"],
                kind=name, preset="gentle", seed=seed,
                ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            if stage == "detail":
                report = detail.run(
                    manifest=synthetic / "manifest.json", output_dir=out,
                    methods=["off", "cas", "richardson_lucy"],
                    presets=["gentle"], mix=100,
                    ffmpeg=ffmpeg, ffprobe=ffprobe,
                )
                evaluated = out / "evaluation" / "evaluation.json"
                labels = list(report["scores"])
            else:
                report = other.run(
                    manifest=synthetic / "manifest.json", output_dir=out,
                    stage=stage, ffmpeg=ffmpeg, ffprobe=ffprobe,
                    denoise_methods=("off", "hqdn3d", "nlmeans", "bm3d"),
                    denoise_presets=("gentle",), scratch_sensitivity=0.65,
                )
                evaluated = out / "scores" / "evaluation.json"
                labels = list(report["scores"])
            if not evaluated.is_file():
                raise ValueError(f"Missing true scientific score report: {evaluated}")
            evaluation_paths.append(evaluated)
            completed.append({
                "film": item["film"], "scene": item["scene"],
                "stage": stage, "degradation": name,
                "source_clip_sha256": item["sha256"],
                "source_transfer_sha256": item["source_transfer_sha256"],
                "evaluation_sha256": ev.digest(evaluated),
                "evaluation_file": str(evaluated.relative_to(output_root)),
                "candidates": labels,
            })
    summary = ev.summarize(evaluation_paths, n_bootstrap=1000, seed=seed)
    result = {
        "schema_version": 1, "experiment": "ARP-multifilm-known-synthetic-degradations",
        "films": films, "scenes_per_film": scenes_per_film, "stages": stages,
        "seed": seed, "stage_runs": len(completed),
        "outputs": completed, "film_level_summary": summary,
        "limitations": [
            "Only controlled defects inserted into digitized transfers, never ground truth of historic camera original.",
            "Three films with few scenes are a pilot, not a representative population; film bootstrap is exploratory.",
            "ProPainter, DART neural repair and hallucinations are not measured. Telea is a labeled classical control only.",
            "Metrics across different degradation families may not be pooled into one universal ranking.",
            "Commercial rights to digital transfers and restored editions are not determined by this experiment.",
        ],
    }
    (output_root / "multifilm_report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def parse_csv(value: str) -> list[str]:
    result = [s.strip() for s in value.split(",")]
    if not result or any(not s for s in result) or len(set(result)) != len(result):
        raise ValueError("Provide nonempty unique comma-separated names")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus-root", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--films", required=True,
                   help="Comma-separated IDs from catalog, e.g. lumiere_train,gertie,caligari")
    p.add_argument("--stages", default="deflicker,denoise,detail,stabilize,scratch_masks")
    p.add_argument("--scenes-per-film", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--ffprobe", default="ffprobe")
    a = p.parse_args()
    report = execute(
        corpus_root=a.corpus_root, output_root=a.output_dir,
        films=parse_csv(a.films), stages=parse_csv(a.stages),
        scenes_per_film=a.scenes_per_film, seed=a.seed,
        ffmpeg=ev.find_binary(a.ffmpeg), ffprobe=ev.find_binary(a.ffprobe),
    )
    print(json.dumps({
        "films": report["films"], "stage_runs": report["stage_runs"],
        "summary_rows": len(report["film_level_summary"]["results"]),
        "result": str(a.output_dir / "multifilm_report.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
