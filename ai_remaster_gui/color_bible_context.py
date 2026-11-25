"""Shared Color Bible, canonical shot-ID, and shot-mask path resolution.

Provides a single source of truth used by `server.py`, `references.py`,
`http_handler.py`, `colorize_video.py`, and `qc_color_consistency.py` so the
active Color Bible DB, Film ID, shot identifiers, and SAM 2.1 mask archives
stay synchronized across every pipeline stage without duplicate heuristics.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

DEFAULT_COLOR_BIBLE_DB = "manifests/color_bible.db"


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _root_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def resolve_color_bible_context(
    settings: dict[str, Any] | None,
    root: Path,
    *,
    stage: str | None = None,
    explicit_db: str = "",
    explicit_film_id: str = "",
) -> dict[str, Any]:
    """Resolve the active Color Bible SQLite path and Film ID across stages.

    Precedence for DB path:
      1. ``explicit_db``
      2. ``settings[stage]["color_bible_db"]`` (if ``stage`` is provided)
      3. ``settings["global"]["color_bible_db"]``
      4. ``settings["references"]["color_bible_db"]``
      5. ``settings["colour"]["color_bible_db"]``
      6. ``DEFAULT_COLOR_BIBLE_DB`` (``manifests/color_bible.db``)

    Precedence for Film ID:
      1. ``explicit_film_id``
      2. ``settings[stage]["film_id"]`` or ``settings[stage]["color_bible_film_id"]``
      3. ``settings["global"]["color_bible_film_id"]`` or ``settings["global"]["film_id"]``
      4. ``settings["references"]["film_id"]`` or ``settings["references"]["color_bible_film_id"]``
      5. ``settings["colour"]["film_id"]`` or ``settings["colour"]["color_bible_film_id"]``
    """
    cfg = settings if isinstance(settings, dict) else {}
    stage_cfg = cfg.get(stage, {}) if stage and isinstance(cfg.get(stage), dict) else {}
    global_cfg = cfg.get("global", {}) if isinstance(cfg.get("global"), dict) else {}
    refs_cfg = cfg.get("references", {}) if isinstance(cfg.get("references"), dict) else {}
    colour_cfg = cfg.get("colour", {}) if isinstance(cfg.get("colour"), dict) else {}

    env_db = os.environ.get("ARP_COLOR_BIBLE_DB", "").strip()
    configured_db = _first_nonempty(
        explicit_db,
        stage_cfg.get("color_bible_db"),
        global_cfg.get("color_bible_db"),
        refs_cfg.get("color_bible_db"),
        colour_cfg.get("color_bible_db"),
    )
    if explicit_db:
        db_raw = str(explicit_db).strip()
    elif str(stage_cfg.get("color_bible_db") or "").strip():
        db_raw = str(stage_cfg["color_bible_db"]).strip()
    elif env_db and (not configured_db or configured_db == DEFAULT_COLOR_BIBLE_DB):
        db_raw = env_db
    else:
        db_raw = configured_db or DEFAULT_COLOR_BIBLE_DB
    candidate_path = Path(db_raw)
    db_path = candidate_path if candidate_path.is_absolute() else (root / candidate_path)
    db_rel = _root_relative(root, db_path)

    film_id = _first_nonempty(
        explicit_film_id,
        stage_cfg.get("film_id"),
        stage_cfg.get("color_bible_film_id"),
        global_cfg.get("color_bible_film_id"),
        global_cfg.get("film_id"),
        refs_cfg.get("film_id"),
        refs_cfg.get("color_bible_film_id"),
        colour_cfg.get("film_id"),
        colour_cfg.get("color_bible_film_id"),
    )

    return {
        "db_path": db_path,
        "db_raw": db_raw,
        "db_rel": db_rel,
        "film_id": film_id,
        "exists": db_path.exists(),
    }


def canonical_shot_id(film_id: str, shot_index: int) -> str:
    """Return the canonical Color Bible shot_id for a film and zero-based shot index."""
    idx = int(shot_index)
    film = str(film_id or "").strip()
    if film:
        return f"{film}_{idx}"
    return f"shot_{idx:04d}"


def shot_id_candidates(film_id: str, shot_index: int) -> list[str]:
    """Return all legacy and canonical shot_id spellings in priority order."""
    idx = int(shot_index)
    film = str(film_id or "").strip()
    raw: list[str] = []
    if film:
        raw.extend(
            [
                f"{film}_{idx}",
                f"{film}_{idx:04d}",
                f"{film}_{idx:03d}",
            ]
        )
    raw.extend(
        [
            f"shot_{idx:04d}",
            f"shot_{idx:03d}",
            f"shot_{idx}",
            str(idx),
        ]
    )
    seen: set[str] = set()
    ordered: list[str] = []
    for item in raw:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def parse_shot_index_from_id(shot_id: str) -> int | None:
    """Extract the numeric shot index suffix from canonical or legacy shot IDs."""
    text = str(shot_id or "").strip()
    if not text:
        return None
    match = re.search(r"(\d+)$", text)
    if not match:
        return None
    return int(match.group(1))


def find_shot_mask_archive_candidates(
    masks_dir: Path | str,
    shot_index: int,
    film_id: str = "",
    shot_id: str = "",
) -> Path | None:
    """Find the first existing ``*.masks.npz`` archive matching canonical or legacy shot IDs."""
    base = Path(masks_dir)
    if not base.exists() or not base.is_dir():
        return None
    candidates: list[str] = []
    explicit = str(shot_id or "").strip()
    if explicit:
        candidates.append(explicit)
    for cand in shot_id_candidates(film_id, shot_index):
        if cand not in candidates:
            candidates.append(cand)
    for stem in candidates:
        path = base / f"{stem}.masks.npz"
        if path.exists() and path.is_file():
            return path
    return None
