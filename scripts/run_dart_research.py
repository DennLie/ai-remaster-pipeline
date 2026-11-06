"""Opt-in isolated DART research runner for an already-installed external checkout.

NEVER downloads model/code or mixes third-party RTN code into ARP. Refuses to
run unless source commit and weight bytes match known DART v1.0.0 artifacts.
No commercial rights are asserted. Use separate DART Python environment.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

# Primary source: DART v1.0.0 tag and scripts/download_weights.py (2026-11-01).
DART_COMMIT = "fc6f3e722c74a3b0e3a43ef40bb1d5303fa5e332"
DART_WEIGHTS_SHA256 = "d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3"


def validate_geometry(width: int, height: int) -> None:
    # Four RAFT correlation levels at 1/8 resolution must not reach a
    # singleton axis: bilinear coordinate normalization divides by size-1.
    if width < 128 or height < 128:
        raise ValueError('DART requires at least 128x128 for safe RAFT geometry; no automatic upscaling.')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkout(repo: Path, weights: Path) -> None:
    script = repo / "scripts" / "inference.py"
    if not script.is_file():
        raise FileNotFoundError(f"External DART inference.py missing: {script}")
    git = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True)
    revision = git.stdout.strip()
    if revision != DART_COMMIT:
        raise ValueError(f"Unreviewed DART checkout {revision}; expected pinned {DART_COMMIT}")
    if not weights.is_file():
        raise FileNotFoundError(f"DART weights missing: {weights}")
    actual_hash = sha256(weights)
    if actual_hash != DART_WEIGHTS_SHA256:
        raise ValueError(f"DART weights SHA-256 mismatch: {actual_hash}")


def build_command(
    dart_python: Path, dart_repo: Path, weights: Path, source: Path, output: Path,
    max_frames: int, device: str, chunk_size: int, stride: int,
) -> list[str]:
    if max_frames < 2:
        raise ValueError("DART requires at least two frames")
    if not 0 < stride <= chunk_size <= 15:
        raise ValueError("Choose 1 <= stride <= chunk_size <= 15")
    return [
        str(dart_python), '-W', 'error:invalid value encountered in cast:RuntimeWarning',
        str(dart_repo / "scripts" / "inference.py"),
        "--input", str(source), "--output", str(output),
        "--weights", str(weights), "--save-masks", "--save-video",
        "--max-frames", str(max_frames), "--chunk-size", str(chunk_size),
        "--stride", str(stride), "--device", device,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dart-repo", required=True, type=Path, help="Existing DART v1.0.0 external checkout")
    parser.add_argument("--dart-python", required=True, type=Path, help="Python executable from DART's isolated venv")
    parser.add_argument("--source", required=True, type=Path, help="Exact source video or PNG frame folder")
    parser.add_argument("--output-dir", required=True, type=Path, help="Separate empty DART output folder")
    parser.add_argument("--weights", type=Path, help="Pinned DART checkpoint; default external checkout/pretrained_models/dart.safetensors")
    parser.add_argument("--max-frames", type=int, default=72)
    parser.add_argument("--chunk-size", type=int, default=7)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--research-only", action="store_true", help="Acknowledge licence uncertainty and prevent production use")
    args = parser.parse_args(argv)
    if not args.research_only:
        parser.error("DART test execution is research-only; explicitly pass --research-only")
    if not args.source.exists():
        parser.error("Source footage not found")
    repo = args.dart_repo.resolve()
    source = args.source.resolve()
    output = args.output_dir.resolve()
    weights = (args.weights or repo / "pretrained_models" / "dart.safetensors").resolve()
    if output == repo or output == source or output == source.parent or output == repo.parent:
        parser.error("DART test output must be a separate directory")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty to prevent stale DART frames")
    validate_checkout(repo, weights)
    from assess_dart_exports import evaluate, source_frames
    _, first = next(source_frames(source, max_frames=2))
    validate_geometry(first.shape[1], first.shape[0])
    command = build_command(
        args.dart_python.resolve(), repo, weights, source, output,
        args.max_frames, args.device, args.chunk_size, args.stride
    )
    print("Running pinned external DART v1.0.0 for research only.", flush=True)
    print(f"Source: {source}  |  DART revision: {DART_COMMIT}", flush=True)
    subprocess.run(command, cwd=repo, check=True)
    result = evaluate(source, output, output / "arp_qa", max_frames=args.max_frames)
    print(f"DART restored {result['frames']} frames; "
          f"changed-unmasked fraction: {result['changed_unmasked_fraction']:.4f}", flush=True)
    print("NO commercial clearance. Review contact sheet and model provenance before any deployment.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
