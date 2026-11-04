"""Independent, model-free Light Deflicker stage using the existing algorithms."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from common import ROOT, file_fingerprint, find_ffmpeg, resolve_path, resumable_output, root_relative, signature_path, write_signature
from intermediate_video import canonical_profile

METHODS = ("arp", "ffmpeg_median", "ffmpeg_am", "localized_research", "localized_stream")
VERSION = 1


def options(values: dict) -> dict:
    method = values.get("method", "arp")
    if method not in ("off", *METHODS):
        raise ValueError("Choose a supported Light Deflicker method.")
    result = {"method": method}
    if method == "arp":
        strength = float(values.get("strength", 0.65))
        limit = float(values.get("max_correction", 0.08))
        if not math.isfinite(strength) or not 0 <= strength <= 1 or not math.isfinite(limit) or not 0 <= limit <= .15:
            raise ValueError("ARP strength must be 0–1 and maximum correction 0–0.15.")
        result.update(strength=strength, max_correction=limit)
    elif method in ("ffmpeg_median", "ffmpeg_am"):
        window = int(values.get("window", 5))
        if str(values.get("window", 5)) != str(window) or window not in (3, 5, 7, 9):
            raise ValueError("FFmpeg window must be 3, 5, 7 or 9 frames.")
        result["window"] = window
    return result


def output_for(source: Path, values: dict, profile: str = "high") -> Path:
    selected = options(values)
    identity = [VERSION, str(source.resolve()), selected, canonical_profile(profile)]
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    from artifact_ids import source_word
    return ROOT / "intermediate" / "deflickered" / f"{source_word(source.name)}_deflickered_{selected['method']}_{token}.mkv"


def work_directory(source: Path, values: dict, profile: str = "high") -> Path:
    return ROOT / ".cache" / "deflicker" / output_for(source, values, profile).stem


def arguments(values: dict, profile: str = "high", force: bool = False) -> argparse.Namespace:
    selected = options(values)
    return argparse.Namespace(
        light_deflicker=selected["method"] != "off", deflicker_method=selected["method"], deflicker_strength=selected.get("strength", .65),
        deflicker_max_correction=selected.get("max_correction", .08), deflicker_window=selected.get("window", 5),
        localized_confirm_single_shot=str(values.get("localized_confirm_single_shot", "false")) == "true",
        localized_confirm_film_experiment=str(values.get("localized_confirm_film_experiment", "false")) == "true",
        dearchive=False, ai_descratch=False, devignette=False, detect_defects=False,
        intermediate_profile=canonical_profile(profile), force=force,
    )


def validate_request(values: dict, info: dict) -> None:
    args = arguments(values)
    if args.deflicker_method == "localized_research":
        from localized_deflicker_backend import validate_localized_request
        validate_localized_request(args, info)
    elif args.deflicker_method == "localized_stream":
        from localized_deflicker_stream import validate_stream_request
        validate_stream_request(args, info)


def run(source: Path, values: dict, profile: str = "high", *, output: Path | None = None,
        force: bool = False, dry_run: bool = False) -> Path:
    from prepare_outpaint_input import probe_video
    from cleanup_video import finalize_output
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError("Light Deflicker input is missing. Run the selected upstream stage first.")
    profile = canonical_profile(profile)
    selected = options(values)
    info = probe_video(source)
    validate_request(values, info)
    output = output or output_for(source, values, profile)
    if output.resolve() == source:
        raise ValueError("Light Deflicker never overwrites its processing input.")
    signature = {"tool": "deflicker_video.py", "version": VERSION, "source": root_relative(source),
                 "source_fingerprint": file_fingerprint(source), "method": selected["method"],
                 "parameters": selected, "intermediate_profile": profile}
    if not force and resumable_output(output, signature, width=int(info["width"]), height=int(info["height"]), video_like=source):
        # Content-based reuse can survive a source touch; refresh the review's stat-bound lineage.
        if not dry_run:
            write_signature(output, signature)
        print(f"Reuse Light Deflicker output: {output}", flush=True)
        return output
    if dry_run:
        print(f"Would run Light Deflicker {selected}: {source} -> {output}", flush=True)
        return output
    ffmpeg = find_ffmpeg()
    args = arguments(values, profile, force)
    workdir = work_directory(source, values, profile)
    if selected["method"] == "arp":
        from film_deflicker import prepare_light_deflicker
        processed = prepare_light_deflicker(ffmpeg, source, workdir, info, args)
    elif selected["method"] in ("ffmpeg_median", "ffmpeg_am"):
        from film_deflicker import prepare_ffmpeg_deflicker
        processed = prepare_ffmpeg_deflicker(ffmpeg, source, workdir, info, args)
    elif selected["method"] == "localized_research":
        from localized_deflicker_backend import prepare_localized_deflicker
        processed = prepare_localized_deflicker(ffmpeg, source, workdir, info, args)
    elif selected["method"] == "localized_stream":
        from localized_deflicker_stream import prepare_stream_localized_deflicker
        processed = prepare_stream_localized_deflicker(ffmpeg, source, workdir, info, args)
    else:
        processed = source
    # A failed replacement or validation must not leave the previous ready marker attached.
    signature_path(output).unlink(missing_ok=True)
    finalize_output(ffmpeg, processed, source, output, info, intermediate_profile=profile)
    if not resumable_geometry(source, output):
        raise ValueError("Light Deflicker output lost its frame mapping or dimensions; it was not published as ready.")
    write_signature(output, signature)
    print(f"Wrote Light Deflicker video: {output}", flush=True)
    return output


def resumable_geometry(source: Path, output: Path) -> bool:
    from prepare_outpaint_input import probe_video
    before, after = probe_video(source), probe_video(output)
    return all(before[key] == after[key] for key in ("width", "height", "frames")) and abs(float(before["fps"]) - float(after["fps"])) < .00001


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output")
    parser.add_argument("--method", choices=("off", *METHODS), default="arp")
    parser.add_argument("--window", default="5")
    parser.add_argument("--strength", default="0.65")
    parser.add_argument("--max-correction", default="0.08")
    parser.add_argument("--localized-confirm-single-shot", action="store_true")
    parser.add_argument("--localized-confirm-film-experiment", action="store_true")
    parser.add_argument("--intermediate-profile", choices=("low", "medium", "high", "lossless"), default="high")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    values = {**vars(args), "localized_confirm_single_shot": str(args.localized_confirm_single_shot).lower(),
              "localized_confirm_film_experiment": str(args.localized_confirm_film_experiment).lower()}
    run(resolve_path(args.source), values, args.intermediate_profile,
        output=resolve_path(args.output) if args.output else None, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
