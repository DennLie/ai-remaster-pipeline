#!/usr/bin/env python3
"""Multi-Worker Shot-Parallel Batch Dispatcher and Shard Partitioning.

Partitions shots cleanly across N workers once a Color Bible snapshot
(color_bible_version or hash) is frozen.

Supports:
1. Local concurrent worker pool: --workers N.
2. RunPod-compatible shard manifest generator: --worker-index i --worker-count N.
3. Partition strategies: interleaved (default round-robin) and contiguous.
4. Deterministic manifest merging sorted by shot_index.
5. Frozen Color Bible snapshot version validation/injection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import inspect
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.manifests import read_manifest_details, write_manifest_details


def partition_manifest_rows(
    rows: list[dict[str, str]],
    worker_index: int,
    worker_count: int,
    strategy: str = "interleaved",
) -> list[dict[str, str]]:
    """Partition manifest rows for a specific worker index.

    Parameters:
        rows: List of shot rows from manifest CSV.
        worker_index: 0-indexed worker integer.
        worker_count: Total worker integer count (>= 1).
        strategy: "interleaved" (round-robin) or "contiguous" (block-sliced).

    Returns:
        List of row dictionaries assigned to this worker.
    """
    if worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError(
            f"worker_index must be in range [0, {worker_count - 1}], got {worker_index}"
        )

    strat = strategy.strip().lower()
    if strat == "interleaved":
        return [row for idx, row in enumerate(rows) if idx % worker_count == worker_index]
    elif strat == "contiguous":
        total = len(rows)
        start_idx = (worker_index * total) // worker_count
        end_idx = ((worker_index + 1) * total) // worker_count
        return rows[start_idx:end_idx]
    else:
        raise ValueError(
            f"Unsupported partition strategy: '{strategy}'. Choose 'interleaved' or 'contiguous'."
        )


def generate_shard_manifests(
    manifest_path: Path | str,
    worker_count: int,
    output_dir: Path | str | None = None,
    strategy: str = "interleaved",
    color_bible_version: str | None = None,
) -> list[Path]:
    """Generate N partitioned shard manifest CSVs from a master manifest.

    Parameters:
        manifest_path: Path to source manifest CSV.
        worker_count: Number of worker shards (>= 1).
        output_dir: Directory to save shard manifests (defaults to manifest parent).
        strategy: Partition strategy ("interleaved" or "contiguous").
        color_bible_version: Optional Color Bible snapshot hash to freeze into manifests.

    Returns:
        List of Path objects for all generated shard manifests.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    source_video, fieldnames, rows = read_manifest_details(manifest_path)
    out_dir = Path(output_dir) if output_dir is not None else manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = list(fieldnames)
    if color_bible_version is not None:
        if "color_bible_version" not in fieldnames:
            fieldnames.append("color_bible_version")
        for row in rows:
            row["color_bible_version"] = str(color_bible_version)

    shard_paths: list[Path] = []
    stem = manifest_path.stem

    for i in range(worker_count):
        shard_rows = partition_manifest_rows(rows, i, worker_count, strategy=strategy)
        shard_file = out_dir / f"{stem}_worker_{i}_of_{worker_count}.csv"
        write_manifest_details(shard_file, source_video, fieldnames, shard_rows)
        shard_paths.append(shard_file)

    return shard_paths


def merge_shard_manifests(
    shard_paths: list[Path | str],
    output_path: Path | str,
) -> Path:
    """Combine multiple worker shard manifests into a unified manifest sorted by shot_index.

    Parameters:
        shard_paths: List of paths to shard manifest CSV files.
        output_path: Path to write the merged manifest CSV.

    Returns:
        Path object pointing to the written merged manifest CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_video: str = ""
    fieldnames: list[str] = []
    all_rows: list[dict[str, str]] = []

    for shard in shard_paths:
        shard_p = Path(shard)
        if not shard_p.exists():
            continue
        src, fields, rows = read_manifest_details(shard_p)
        if not source_video and src:
            source_video = src
        for f in fields:
            if f not in fieldnames:
                fieldnames.append(f)
        all_rows.extend(rows)

    def sort_key(row: dict[str, str]) -> tuple[int, int, str]:
        idx_str = row.get("shot_index", "").strip()
        if idx_str.isdigit():
            return (0, int(idx_str), row.get("shot_id", ""))
        return (1, 0, row.get("shot_id", ""))

    # Deduplicate by shot identifier if same shot appeared in multiple shards
    seen_ids: set[str] = set()
    unique_rows: list[dict[str, str]] = []
    for r in all_rows:
        sid = r.get("shot_id") or r.get("shot_index") or str(id(r))
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_rows.append(r)

    unique_rows.sort(key=sort_key)
    write_manifest_details(output_path, source_video, fieldnames, unique_rows)
    return output_path


def dispatch_local_workers(
    manifest_path: Path | str,
    worker_count: int,
    method: str = "osi_masked_lab",
    execute_fn: Callable | None = None,
) -> dict[str, Any]:
    """Execute concurrent local workers across partitioned shard manifests.

    Parameters:
        manifest_path: Path to master manifest CSV.
        worker_count: Concurrency level / number of worker shards.
        method: Colorization method (default: "osi_masked_lab").
        execute_fn: Optional callable to execute a shard manifest.

    Returns:
        dict containing execution status and manifest paths.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    _src, _f, rows = read_manifest_details(manifest_path)
    shard_paths = generate_shard_manifests(manifest_path, worker_count, strategy="interleaved")

    def _worker_job(shard_p: Path) -> Path:
        if execute_fn is not None:
            sig = inspect.signature(execute_fn)
            if len(sig.parameters) == 1:
                execute_fn(shard_p)
            else:
                try:
                    execute_fn(shard_p)
                except TypeError:
                    execute_fn(shard_p, method)
        else:
            colorizer_script = REPO_ROOT / "scripts" / "colorize_video.py"
            cmd = [
                sys.executable,
                str(colorizer_script),
                "--manifest",
                str(shard_p),
                "--method",
                method,
            ]
            subprocess.run(cmd, check=True)
        return shard_p

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_worker_job, p) for p in shard_paths]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    merged_path = manifest_path.parent / f"{manifest_path.stem}_merged.csv"
    merge_shard_manifests(shard_paths, merged_path)

    return {
        "worker_count": worker_count,
        "method": method,
        "shard_manifests": [str(p) for p in shard_paths],
        "merged_manifest": str(merged_path),
        "total_shots": len(rows),
        "status": "completed",
    }


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for parallel_colorize_dispatch."""
    parser = argparse.ArgumentParser(
        prog="parallel_colorize_dispatch",
        description="Multi-Worker Shot-Parallel Batch Dispatcher and Shard Partitioning.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to master shot manifest CSV file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers for local concurrent pool execution.",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=None,
        help="0-indexed worker index for RunPod shard manifest generation.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Total worker count for RunPod shard manifest generation.",
    )
    parser.add_argument(
        "--partition-strategy",
        choices=["interleaved", "contiguous"],
        default="interleaved",
        help="Partitioning strategy: interleaved (round-robin) or contiguous (block-sliced).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated shard manifests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for merged manifest CSV file.",
    )
    parser.add_argument(
        "--merge-manifests",
        nargs="*",
        default=None,
        help="List of shard manifest paths to merge into a single manifest.",
    )
    parser.add_argument(
        "--color-bible-snapshot",
        type=str,
        default=None,
        help="Color Bible snapshot hash/version to freeze into shard manifests.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="osi_masked_lab",
        help="Colorization method (default: osi_masked_lab).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Merge Mode
    if args.merge_manifests is not None:
        if not args.merge_manifests:
            parser.error("--merge-manifests requires at least one shard manifest file path.")
        output_path = args.output or Path("manifest_merged.csv")
        merged = merge_shard_manifests(args.merge_manifests, output_path)
        print(f"Merged {len(args.merge_manifests)} shard manifests into: {merged}")
        return 0

    # 2. Local Concurrent Worker Pool Mode
    if args.workers is not None:
        if not args.manifest:
            parser.error("--workers mode requires --manifest.")
        result = dispatch_local_workers(
            manifest_path=args.manifest,
            worker_count=args.workers,
            method=args.method,
        )
        print(json.dumps(result, indent=2))
        return 0

    # 3. Single RunPod Shard Generator Mode
    if args.worker_index is not None and args.worker_count is not None:
        if not args.manifest:
            parser.error("--worker-index and --worker-count require --manifest.")
        source_video, fieldnames, rows = read_manifest_details(args.manifest)
        fieldnames = list(fieldnames)
        if args.color_bible_snapshot:
            if "color_bible_version" not in fieldnames:
                fieldnames.append("color_bible_version")
            for r in rows:
                r["color_bible_version"] = args.color_bible_snapshot

        shard_rows = partition_manifest_rows(
            rows, args.worker_index, args.worker_count, strategy=args.partition_strategy
        )
        out_dir = Path(args.output_dir) if args.output_dir else args.manifest.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        shard_path = out_dir / f"{args.manifest.stem}_worker_{args.worker_index}_of_{args.worker_count}.csv"
        write_manifest_details(shard_path, source_video, fieldnames, shard_rows)
        print(f"Generated shard manifest: {shard_path} ({len(shard_rows)} shots)")
        return 0

    # 4. Generate all shards for worker_count
    if args.worker_count is not None:
        if not args.manifest:
            parser.error("--worker-count requires --manifest.")
        shards = generate_shard_manifests(
            manifest_path=args.manifest,
            worker_count=args.worker_count,
            output_dir=args.output_dir,
            strategy=args.partition_strategy,
            color_bible_version=args.color_bible_snapshot,
        )
        print(f"Generated {len(shards)} shard manifests in {args.output_dir or args.manifest.parent}:")
        for s in shards:
            print(f"  {s}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
