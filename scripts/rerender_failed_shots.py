#!/usr/bin/env python3
"""Selective Re-Render CLI and API for Failed Quality Control Shots.

Inspects a shot manifest CSV and/or Color Bible DB qc_events table,
extracts shots flagged qc_status == 'fail', strictly preserves passing
shots ('pass' or 'warn') untouched, and re-renders ONLY those failed
shots using signature-based caching and subprocess or callable execution.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable, Sequence

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.color_bible_db import get_connection, list_qc_events
from ai_remaster_gui.manifests import read_manifest_details, write_manifest_details


def extract_shot_identifier(row: dict[str, str], fallback_index: int = 0) -> str:
    """Extract a consistent shot identifier from a manifest row."""
    shot_id = row.get("shot_id", "").strip()
    if shot_id:
        return shot_id
    shot_index = row.get("shot_index", "").strip()
    if shot_index:
        return f"shot_{shot_index}"
    return f"shot_{fallback_index}"


def find_failed_shots(
    manifest_rows: list[dict[str, str]],
    color_bible_db: Path | str | None = None,
    film_id: str | None = None,
) -> tuple[list[str], list[str]]:
    """Identify failed shots from manifest rows and/or Color Bible SQLite DB.

    Returns:
        (failed_shot_ids, preserved_shot_ids) maintaining original manifest order.
    """
    manifest_shot_ids = [extract_shot_identifier(row, idx) for idx, row in enumerate(manifest_rows)]
    failed_set: set[str] = set()

    # 1. Check manifest rows for explicit 'fail' status
    for idx, row in enumerate(manifest_rows):
        shot_id = manifest_shot_ids[idx]
        status = row.get("qc_status", "").strip().lower()
        if status == "fail":
            failed_set.add(shot_id)

    # 2. Check Color Bible DB qc_events table if database path is provided
    if color_bible_db is not None:
        db_path = Path(color_bible_db)
        if db_path.exists():
            conn = get_connection(db_path)
            try:
                events = list_qc_events(conn, film_id=film_id)
                # list_qc_events returns events ordered by created_at DESC
                latest_by_shot: dict[str, str] = {}
                for ev in events:
                    ev_shot = ev.shot_id.strip() if ev.shot_id else ""
                    if not ev_shot and ev.shot_index is not None:
                        ev_shot = f"shot_{ev.shot_index}"
                    if ev_shot and ev_shot not in latest_by_shot:
                        latest_by_shot[ev_shot] = ev.qc_status.strip().lower()

                for sid, status in latest_by_shot.items():
                    if status == "fail" and sid in manifest_shot_ids:
                        failed_set.add(sid)
            finally:
                conn.close()

    failed_shot_ids = [sid for sid in manifest_shot_ids if sid in failed_set]
    preserved_shot_ids = [sid for sid in manifest_shot_ids if sid not in failed_set]
    return failed_shot_ids, preserved_shot_ids


def rerender_failed_shots(
    manifest_path: Path | str,
    color_bible_db: Path | str | None = None,
    film_id: str | None = None,
    method: str = "osi_masked_lab",
    dry_run: bool = False,
    execute_fn: Callable | None = None,
    python_exe: str | None = None,
    extra_args: Sequence[str] | str | None = None,
) -> dict[str, Any]:
    """Selectively re-render only shots that failed Quality Control.

    Parameters:
        manifest_path: Path to shot manifest CSV.
        color_bible_db: Optional path to Color Bible SQLite DB.
        film_id: Optional film identifier.
        method: Colorization method (default: "osi_masked_lab").
        dry_run: If True, prints plan and returns without executing.
        execute_fn: Optional callable for mock or custom execution.
        python_exe: Python interpreter for subprocesses (default: sys.executable).
        extra_args: Additional command line arguments forwarded to colorizer.

    Returns:
        dict containing:
            total_shots: int
            failed_shot_ids: list[str]
            rerendered_shot_ids: list[str]
            preserved_shot_ids: list[str]
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    source_video, fieldnames, rows = read_manifest_details(manifest_path)
    total_shots = len(rows)

    failed_shot_ids, preserved_shot_ids = find_failed_shots(
        rows, color_bible_db=color_bible_db, film_id=film_id
    )

    if dry_run:
        print(
            f"[DRY-RUN] Re-render plan for {manifest_path.name}: "
            f"{len(failed_shot_ids)} of {total_shots} shot(s) failed QC and scheduled for re-render. "
            f"{len(preserved_shot_ids)} shot(s) strictly preserved."
        )
        if failed_shot_ids:
            print(f"[DRY-RUN] Target failed shot IDs: {failed_shot_ids}")
        return {
            "total_shots": total_shots,
            "failed_shot_ids": failed_shot_ids,
            "rerendered_shot_ids": [],
            "preserved_shot_ids": preserved_shot_ids,
        }

    if not failed_shot_ids:
        print(f"All {total_shots} shot(s) passed or warned. Zero re-renders needed.")
        return {
            "total_shots": total_shots,
            "failed_shot_ids": [],
            "rerendered_shot_ids": [],
            "preserved_shot_ids": preserved_shot_ids,
        }

    # Filter rows strictly to failed shots
    failed_set = set(failed_shot_ids)
    failed_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        sid = extract_shot_identifier(row, idx)
        if sid in failed_set:
            row_copy = dict(row)
            # Ensure row is marked enabled for colorizer processing
            row_copy["enabled"] = "true"
            failed_rows.append(row_copy)

    temp_dir = tempfile.mkdtemp(prefix="arp_rerender_")
    filtered_manifest_path = Path(temp_dir) / f"{manifest_path.stem}_failed_shots.csv"

    try:
        write_manifest_details(filtered_manifest_path, source_video, fieldnames, failed_rows)

        if execute_fn is not None:
            # Check callable signature to support flexible execution harnesses
            sig = inspect.signature(execute_fn)
            params = list(sig.parameters.values())
            if len(params) == 1 and params[0].name in ("shots", "failed_shots", "rows"):
                execute_fn(failed_rows)
            elif len(params) == 1 and params[0].name in ("shot_ids", "failed_shot_ids"):
                execute_fn(failed_shot_ids)
            else:
                try:
                    execute_fn(filtered_manifest_path)
                except TypeError:
                    execute_fn(filtered_manifest_path, failed_shot_ids)
        else:
            exe = python_exe or sys.executable
            colorizer_script = REPO_ROOT / "scripts" / "colorize_video.py"
            cmd = [
                exe,
                str(colorizer_script),
                "--manifest",
                str(filtered_manifest_path),
                "--method",
                method,
            ]
            if color_bible_db:
                cmd.extend(["--color-bible-db", str(color_bible_db)])
            if film_id:
                cmd.extend(["--film-id", str(film_id)])
            if extra_args:
                if isinstance(extra_args, str):
                    cmd.extend(shlex.split(extra_args))
                else:
                    cmd.extend(list(extra_args))

            print(f"Invoking re-render subprocess: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        # Merge updated rows from filtered manifest back into master manifest
        if filtered_manifest_path.exists():
            _sub_src, sub_fields, sub_rows = read_manifest_details(filtered_manifest_path)
            for f in sub_fields:
                if f not in fieldnames:
                    fieldnames.append(f)
            for r in sub_rows:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)

            sub_by_id: dict[str, dict[str, str]] = {}
            for idx, r in enumerate(sub_rows):
                sid = extract_shot_identifier(r, idx)
                sub_by_id[sid] = r

            for idx, row in enumerate(rows):
                sid = extract_shot_identifier(row, idx)
                if sid in sub_by_id:
                    row.update(sub_by_id[sid])

            write_manifest_details(manifest_path, source_video, fieldnames, rows)

        rerendered_shot_ids = list(failed_shot_ids)
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    return {
        "total_shots": total_shots,
        "failed_shot_ids": failed_shot_ids,
        "rerendered_shot_ids": rerendered_shot_ids,
        "preserved_shot_ids": preserved_shot_ids,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser for rerender_failed_shots."""
    parser = argparse.ArgumentParser(
        prog="rerender_failed_shots",
        description="Selectively re-render only shots that failed Quality Control.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to shot manifest CSV file.",
    )
    parser.add_argument(
        "--color-bible-db",
        type=Path,
        default=None,
        help="Optional path to Color Bible SQLite DB.",
    )
    parser.add_argument(
        "--film-id",
        type=str,
        default=None,
        help="Film identifier.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="osi_masked_lab",
        help="Colorization method (default: osi_masked_lab).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prints failed shot IDs and re-render plan without executing.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=None,
        help="Python executable to invoke for subprocesses (default sys.executable).",
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Additional arguments forwarded to colorizer script.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    result = rerender_failed_shots(
        manifest_path=args.manifest,
        color_bible_db=args.color_bible_db,
        film_id=args.film_id,
        method=args.method,
        dry_run=args.dry_run,
        python_exe=args.python_exe,
        extra_args=args.extra_args,
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
