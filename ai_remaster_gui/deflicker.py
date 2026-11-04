"""Stage-owned Light Deflicker variants and explicit subset generation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import SCRIPTS
from .paths import rel, resolve_video_source

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import deflicker_video as backend

METHODS = backend.METHODS


def output_for(source: str, values: dict, profile: str = "high") -> str:
    return rel(backend.output_for(resolve_video_source(source), values, profile)) if source else ""


def validate_variants(rows: list[dict]) -> None:
    if not isinstance(rows, list) or not 1 <= len(rows) <= len(METHODS):
        raise ValueError("Choose 1–5 Light Deflicker variants.")
    methods = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("method"), str) or row["method"] not in METHODS or row.get("preset", "settings") != "settings":
            raise ValueError("Choose supported Light Deflicker methods using their current settings.")
        methods.append(row["method"])
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate Light Deflicker variants.")


def validate_requests(source: Path, rows: list[dict], values: dict) -> None:
    from .frame_inspector import _metadata
    validate_variants(rows)
    info = None
    for row in rows:
        selected = {**values, "method": row["method"]}
        backend.options(selected)
        if row["method"].startswith("localized_"):
            info = info or _metadata(source)
            backend.validate_request(selected, info)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--profile", default="high")
    args = parser.parse_args()
    source = resolve_video_source(args.source)
    rows, values = json.loads(args.variants), json.loads(args.settings)
    if not isinstance(values, dict):
        raise ValueError("Expected Light Deflicker settings.")
    validate_requests(source, rows, values)
    for index, row in enumerate(rows):
        print(f"Light Deflicker batch {index + 1}/{len(rows)}: {row['method']}", flush=True)
        backend.run(source, {**values, "method": row["method"]}, args.profile)
        print(f"Light Deflicker batch complete {index + 1}/{len(rows)}: {row['method']}", flush=True)


if __name__ == "__main__":
    main()
