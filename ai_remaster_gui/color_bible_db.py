from __future__ import annotations

import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .config import ROOT


def _utcnow_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def resolve_color_bible_db(path: Path | str | None = None) -> Path:
    """Resolve the path to the Color Bible SQLite database.

    Resolution order:
    1. Explicit path parameter if provided.
    2. Environment variable `ARP_COLOR_BIBLE_DB` if set and non-empty.
    3. Default path: `ROOT / "manifests" / "color_bible.db"`.
    """
    if path is not None:
        return Path(path)
    env_path = os.environ.get("ARP_COLOR_BIBLE_DB", "").strip()
    if env_path:
        return Path(env_path)
    return ROOT / "manifests" / "color_bible.db"


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Open and configure a SQLite connection with WAL mode and foreign key enforcement.

    - Foreign keys are always enabled (`PRAGMA foreign_keys = ON;`).
    - File databases enable WAL journal mode and busy timeout.
    - Row factory is set to `sqlite3.Row`.
    """
    path_str = str(db_path)
    is_memory = path_str == ":memory:" or path_str.startswith("file::memory:")
    if not is_memory and not path_str.startswith("file:"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path_str)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    if not is_memory:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
    return conn


# --- Schema Migrations ---

MIGRATION_1_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS films (
    film_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_video TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sequences (
    sequence_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    sequence_index INTEGER NOT NULL,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shots (
    shot_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    sequence_id TEXT REFERENCES sequences(sequence_id) ON DELETE SET NULL,
    shot_index INTEGER NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    source_reference TEXT,
    color_reference TEXT,
    qc_status TEXT DEFAULT 'pending',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_shots_film_index ON shots(film_id, shot_index);

CREATE TABLE IF NOT EXISTS characters (
    character_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appearance_variants (
    variant_id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(character_id) ON DELETE CASCADE,
    costume_id TEXT NOT NULL,
    hair_profile TEXT,
    skin_reference TEXT,
    lighting_context TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS objects (
    object_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    kind TEXT CHECK(kind IN ('prop', 'vehicle', 'other')) NOT NULL,
    name TEXT NOT NULL,
    palette_json TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS locations (
    location_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    set_name TEXT NOT NULL,
    lighting_context TEXT,
    palette_json TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS shot_entity_assignments (
    assignment_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    shot_id TEXT REFERENCES shots(shot_id) ON DELETE SET NULL,
    entity_type TEXT CHECK(entity_type IN ('character', 'object', 'location')) NOT NULL,
    entity_id TEXT NOT NULL,
    variant_id TEXT REFERENCES appearance_variants(variant_id) ON DELETE SET NULL,
    valid_shot_range_start INTEGER NOT NULL,
    valid_shot_range_end INTEGER NOT NULL,
    confidence REAL DEFAULT 1.0,
    approval_status TEXT CHECK(approval_status IN ('draft', 'pending_review', 'approved', 'rejected')) NOT NULL DEFAULT 'approved',
    provenance TEXT DEFAULT 'manual',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_assignments_shot ON shot_entity_assignments(film_id, shot_id);
CREATE INDEX IF NOT EXISTS idx_assignments_range ON shot_entity_assignments(film_id, valid_shot_range_start, valid_shot_range_end);

CREATE TABLE IF NOT EXISTS color_references (
    ref_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    entity_type TEXT CHECK(entity_type IN ('character', 'object', 'location')) NOT NULL,
    entity_id TEXT NOT NULL,
    variant_id TEXT REFERENCES appearance_variants(variant_id) ON DELETE SET NULL,
    source_shot_id TEXT REFERENCES shots(shot_id) ON DELETE SET NULL,
    source_keyframe_path TEXT NOT NULL,
    mask_uri TEXT,
    lab_stats_json TEXT NOT NULL,
    rgb_palette_json TEXT NOT NULL,
    provenance TEXT CHECK(provenance IN ('historical_research', 'ddcolor', 'qwen2511', 'manual')) NOT NULL,
    approval_status TEXT CHECK(approval_status IN ('draft', 'pending_review', 'approved', 'rejected')) NOT NULL DEFAULT 'draft',
    approved_by TEXT,
    approved_at TEXT,
    version INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refs_entity ON color_references(film_id, entity_type, entity_id, variant_id);
"""

MIGRATIONS: list[tuple[int, str]] = [
    (1, MIGRATION_1_SQL),
]


def apply_migrations(conn: sqlite3.Connection) -> list[int]:
    """Apply any pending schema migrations idempotently inside transactions."""
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            );
            """
        )
    applied_versions = {
        int(row["version"])
        for row in conn.execute("SELECT version FROM schema_migrations;").fetchall()
    }
    newly_applied: list[int] = []
    for version, sql_script in sorted(MIGRATIONS, key=lambda m: m[0]):
        if version not in applied_versions:
            with conn:
                conn.executescript(sql_script)
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?);",
                    (version, _utcnow_iso()),
                )
            newly_applied.append(version)
    return newly_applied


# --- Data Models ---


@dataclass
class Film:
    film_id: str
    title: str
    source_video: str | None = None
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> Film:
        return cls(
            film_id=row["film_id"],
            title=row["title"],
            source_video=row["source_video"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Sequence:
    sequence_id: str
    film_id: str
    sequence_index: int
    name: str

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> Sequence:
        return cls(
            sequence_id=row["sequence_id"],
            film_id=row["film_id"],
            sequence_index=int(row["sequence_index"]),
            name=row["name"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShotRecord:
    shot_id: str
    film_id: str
    shot_index: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    sequence_id: str | None = None
    source_reference: str | None = None
    color_reference: str | None = None
    qc_status: str = "pending"
    updated_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> ShotRecord:
        return cls(
            shot_id=row["shot_id"],
            film_id=row["film_id"],
            shot_index=int(row["shot_index"]),
            start_frame=int(row["start_frame"]),
            end_frame=int(row["end_frame"]),
            start_time=float(row["start_time"]),
            end_time=float(row["end_time"]),
            sequence_id=row["sequence_id"],
            source_reference=row["source_reference"],
            color_reference=row["color_reference"],
            qc_status=row["qc_status"] if row["qc_status"] is not None else "pending",
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Character:
    character_id: str
    film_id: str
    name: str
    notes: str | None = None
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> Character:
        return cls(
            character_id=row["character_id"],
            film_id=row["film_id"],
            name=row["name"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AppearanceVariant:
    variant_id: str
    character_id: str
    costume_id: str
    hair_profile: str | None = None
    skin_reference: str | None = None
    lighting_context: str | None = None
    notes: str | None = None
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> AppearanceVariant:
        return cls(
            variant_id=row["variant_id"],
            character_id=row["character_id"],
            costume_id=row["costume_id"],
            hair_profile=row["hair_profile"],
            skin_reference=row["skin_reference"],
            lighting_context=row["lighting_context"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColorBibleObject:
    object_id: str
    film_id: str
    kind: str  # 'prop', 'vehicle', 'other'
    name: str
    palette_json: str | None = None
    notes: str | None = None
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> ColorBibleObject:
        return cls(
            object_id=row["object_id"],
            film_id=row["film_id"],
            kind=row["kind"],
            name=row["name"],
            palette_json=row["palette_json"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Location:
    location_id: str
    film_id: str
    set_name: str
    lighting_context: str | None = None
    palette_json: str | None = None
    notes: str | None = None
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> Location:
        return cls(
            location_id=row["location_id"],
            film_id=row["film_id"],
            set_name=row["set_name"],
            lighting_context=row["lighting_context"],
            palette_json=row["palette_json"],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ShotEntityAssignment:
    assignment_id: str
    film_id: str
    entity_type: str  # 'character', 'object', 'location'
    entity_id: str
    valid_shot_range_start: int
    valid_shot_range_end: int
    shot_id: str | None = None
    variant_id: str | None = None
    confidence: float = 1.0
    approval_status: str = "approved"  # 'draft', 'pending_review', 'approved', 'rejected'
    provenance: str = "manual"
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> ShotEntityAssignment:
        return cls(
            assignment_id=row["assignment_id"],
            film_id=row["film_id"],
            shot_id=row["shot_id"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            variant_id=row["variant_id"],
            valid_shot_range_start=int(row["valid_shot_range_start"]),
            valid_shot_range_end=int(row["valid_shot_range_end"]),
            confidence=float(row["confidence"]) if row["confidence"] is not None else 1.0,
            approval_status=row["approval_status"] if row["approval_status"] is not None else "approved",
            provenance=row["provenance"] if row["provenance"] is not None else "manual",
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColorReferenceRecord:
    ref_id: str
    film_id: str
    entity_type: str  # 'character', 'object', 'location'
    entity_id: str
    source_keyframe_path: str
    lab_stats_json: str
    rgb_palette_json: str
    provenance: str  # 'historical_research', 'ddcolor', 'qwen2511', 'manual'
    variant_id: str | None = None
    source_shot_id: str | None = None
    mask_uri: str | None = None
    approval_status: str = "draft"  # 'draft', 'pending_review', 'approved', 'rejected'
    approved_by: str | None = None
    approved_at: str | None = None
    version: int = 1
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_row(cls, row: sqlite3.Row | dict[str, Any]) -> ColorReferenceRecord:
        return cls(
            ref_id=row["ref_id"],
            film_id=row["film_id"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
            variant_id=row["variant_id"],
            source_shot_id=row["source_shot_id"],
            source_keyframe_path=row["source_keyframe_path"],
            mask_uri=row["mask_uri"],
            lab_stats_json=row["lab_stats_json"],
            rgb_palette_json=row["rgb_palette_json"],
            provenance=row["provenance"],
            approval_status=row["approval_status"] if row["approval_status"] is not None else "draft",
            approved_by=row["approved_by"],
            approved_at=row["approved_at"],
            version=int(row["version"]) if row["version"] is not None else 1,
            created_at=row["created_at"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Repository Interface ---


class ColorBibleRepository(Protocol):
    """Protocol defining the Color Bible persistence operations."""

    def init_db(self) -> list[int]: ...
    def close(self) -> None: ...

    # Film
    def upsert_film(self, film: Film) -> Film: ...
    def get_film(self, film_id: str) -> Film | None: ...
    def list_films(self) -> list[Film]: ...
    def delete_film(self, film_id: str) -> bool: ...

    # Sequence
    def upsert_sequence(self, sequence: Sequence) -> Sequence: ...
    def get_sequence(self, sequence_id: str) -> Sequence | None: ...
    def list_sequences(self, film_id: str) -> list[Sequence]: ...
    def delete_sequence(self, sequence_id: str) -> bool: ...

    # Shot
    def upsert_shot(self, shot: ShotRecord) -> ShotRecord: ...
    def upsert_shots(self, shots: list[ShotRecord]) -> int: ...
    def get_shot(self, shot_id: str) -> ShotRecord | None: ...
    def get_shot_by_index(self, film_id: str, shot_index: int) -> ShotRecord | None: ...
    def list_shots(self, film_id: str) -> list[ShotRecord]: ...
    def delete_shot(self, shot_id: str) -> bool: ...

    # Character
    def upsert_character(self, character: Character) -> Character: ...
    def get_character(self, character_id: str) -> Character | None: ...
    def list_characters(self, film_id: str) -> list[Character]: ...
    def delete_character(self, character_id: str) -> bool: ...

    # Appearance Variant
    def upsert_variant(self, variant: AppearanceVariant) -> AppearanceVariant: ...
    def get_variant(self, variant_id: str) -> AppearanceVariant | None: ...
    def list_variants(self, character_id: str) -> list[AppearanceVariant]: ...
    def list_variants_for_film(self, film_id: str) -> list[AppearanceVariant]: ...
    def delete_variant(self, variant_id: str) -> bool: ...

    # Object
    def upsert_object(self, obj: ColorBibleObject) -> ColorBibleObject: ...
    def get_object(self, object_id: str) -> ColorBibleObject | None: ...
    def list_objects(self, film_id: str) -> list[ColorBibleObject]: ...
    def delete_object(self, object_id: str) -> bool: ...

    # Location
    def upsert_location(self, location: Location) -> Location: ...
    def get_location(self, location_id: str) -> Location | None: ...
    def list_locations(self, film_id: str) -> list[Location]: ...
    def delete_location(self, location_id: str) -> bool: ...

    # Shot Entity Assignment
    def upsert_assignment(self, assignment: ShotEntityAssignment) -> ShotEntityAssignment: ...
    def get_assignment(self, assignment_id: str) -> ShotEntityAssignment | None: ...
    def list_assignments(
        self,
        film_id: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        approval_status: str | None = None,
    ) -> list[ShotEntityAssignment]: ...
    def find_assignments_for_shot_index(
        self,
        film_id: str,
        shot_index: int,
    ) -> list[ShotEntityAssignment]: ...
    def find_assignments_for_range(
        self,
        film_id: str,
        start_index: int,
        end_index: int,
    ) -> list[ShotEntityAssignment]: ...
    def delete_assignment(self, assignment_id: str) -> bool: ...
    def delete_assignments_for_entity(
        self,
        film_id: str,
        entity_type: str,
        entity_id: str,
        variant_id: str | None = None,
    ) -> int: ...

    # Color Reference
    def upsert_reference(self, reference: ColorReferenceRecord) -> ColorReferenceRecord: ...
    def get_reference(self, ref_id: str) -> ColorReferenceRecord | None: ...
    def list_references(
        self,
        film_id: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        variant_id: str | None = None,
        approval_status: str | None = None,
    ) -> list[ColorReferenceRecord]: ...
    def list_approved_references(self, film_id: str) -> list[ColorReferenceRecord]: ...
    def update_reference_approval(
        self,
        ref_id: str,
        approval_status: str,
        approved_by: str | None = None,
        approved_at: str | None = None,
    ) -> bool: ...
    def delete_reference(self, ref_id: str) -> bool: ...


# --- SQLite Repository Implementation ---


class SQLiteColorBibleRepository:
    """SQLite implementation of the ColorBibleRepository."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if connection is not None:
            self._conn = connection
            self.db_path = Path(":memory:")
            self._owns_conn = False
        else:
            self.db_path = resolve_color_bible_db(db_path) if db_path != ":memory:" else Path(":memory:")
            self._conn = get_connection(self.db_path)
            self._owns_conn = True

    def get_connection(self) -> sqlite3.Connection:
        """Return the active SQLite connection."""
        return self._conn

    def close(self) -> None:
        """Close the underlying connection if owned by this repository."""
        if self._owns_conn and self._conn:
            self._conn.close()

    def __enter__(self) -> SQLiteColorBibleRepository:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def init_db(self) -> list[int]:
        """Apply all pending schema migrations idempotently inside transactions."""
        return apply_migrations(self._conn)

    def get_applied_migrations(self) -> list[int]:
        """Return list of all applied migration versions."""
        try:
            rows = self._conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version ASC;"
            ).fetchall()
            return [int(row["version"]) for row in rows]
        except sqlite3.OperationalError:
            return []

    # --- Films ---

    def upsert_film(self, film: Film) -> Film:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO films (film_id, title, source_video, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(film_id) DO UPDATE SET
                    title = excluded.title,
                    source_video = excluded.source_video,
                    updated_at = excluded.updated_at;
                """,
                (film.film_id, film.title, film.source_video, film.created_at, film.updated_at),
            )
        return film

    def get_film(self, film_id: str) -> Film | None:
        row = self._conn.execute(
            "SELECT * FROM films WHERE film_id = ?;", (film_id,)
        ).fetchone()
        return Film.from_row(row) if row else None

    def list_films(self) -> list[Film]:
        rows = self._conn.execute(
            "SELECT * FROM films ORDER BY created_at ASC;"
        ).fetchall()
        return [Film.from_row(row) for row in rows]

    def delete_film(self, film_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM films WHERE film_id = ?;", (film_id,)
            )
        return cur.rowcount > 0

    # --- Sequences ---

    def upsert_sequence(self, sequence: Sequence) -> Sequence:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO sequences (sequence_id, film_id, sequence_index, name)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sequence_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    sequence_index = excluded.sequence_index,
                    name = excluded.name;
                """,
                (sequence.sequence_id, sequence.film_id, sequence.sequence_index, sequence.name),
            )
        return sequence

    def get_sequence(self, sequence_id: str) -> Sequence | None:
        row = self._conn.execute(
            "SELECT * FROM sequences WHERE sequence_id = ?;", (sequence_id,)
        ).fetchone()
        return Sequence.from_row(row) if row else None

    def list_sequences(self, film_id: str) -> list[Sequence]:
        rows = self._conn.execute(
            "SELECT * FROM sequences WHERE film_id = ? ORDER BY sequence_index ASC;",
            (film_id,),
        ).fetchall()
        return [Sequence.from_row(row) for row in rows]

    def delete_sequence(self, sequence_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM sequences WHERE sequence_id = ?;", (sequence_id,)
            )
        return cur.rowcount > 0

    # --- Shots ---

    def upsert_shot(self, shot: ShotRecord) -> ShotRecord:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO shots (
                    shot_id, film_id, sequence_id, shot_index, start_frame, end_frame,
                    start_time, end_time, source_reference, color_reference, qc_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(shot_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    sequence_id = excluded.sequence_id,
                    shot_index = excluded.shot_index,
                    start_frame = excluded.start_frame,
                    end_frame = excluded.end_frame,
                    start_time = excluded.start_time,
                    end_time = excluded.end_time,
                    source_reference = excluded.source_reference,
                    color_reference = excluded.color_reference,
                    qc_status = excluded.qc_status,
                    updated_at = excluded.updated_at;
                """,
                (
                    shot.shot_id,
                    shot.film_id,
                    shot.sequence_id,
                    shot.shot_index,
                    shot.start_frame,
                    shot.end_frame,
                    shot.start_time,
                    shot.end_time,
                    shot.source_reference,
                    shot.color_reference,
                    shot.qc_status,
                    shot.updated_at,
                ),
            )
        return shot

    def upsert_shots(self, shots: list[ShotRecord]) -> int:
        if not shots:
            return 0
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO shots (
                    shot_id, film_id, sequence_id, shot_index, start_frame, end_frame,
                    start_time, end_time, source_reference, color_reference, qc_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(shot_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    sequence_id = excluded.sequence_id,
                    shot_index = excluded.shot_index,
                    start_frame = excluded.start_frame,
                    end_frame = excluded.end_frame,
                    start_time = excluded.start_time,
                    end_time = excluded.end_time,
                    source_reference = excluded.source_reference,
                    color_reference = excluded.color_reference,
                    qc_status = excluded.qc_status,
                    updated_at = excluded.updated_at;
                """,
                [
                    (
                        s.shot_id,
                        s.film_id,
                        s.sequence_id,
                        s.shot_index,
                        s.start_frame,
                        s.end_frame,
                        s.start_time,
                        s.end_time,
                        s.source_reference,
                        s.color_reference,
                        s.qc_status,
                        s.updated_at,
                    )
                    for s in shots
                ],
            )
        return len(shots)

    def get_shot(self, shot_id: str) -> ShotRecord | None:
        row = self._conn.execute(
            "SELECT * FROM shots WHERE shot_id = ?;", (shot_id,)
        ).fetchone()
        return ShotRecord.from_row(row) if row else None

    def get_shot_by_index(self, film_id: str, shot_index: int) -> ShotRecord | None:
        row = self._conn.execute(
            "SELECT * FROM shots WHERE film_id = ? AND shot_index = ?;",
            (film_id, shot_index),
        ).fetchone()
        return ShotRecord.from_row(row) if row else None

    def list_shots(self, film_id: str) -> list[ShotRecord]:
        rows = self._conn.execute(
            "SELECT * FROM shots WHERE film_id = ? ORDER BY shot_index ASC;",
            (film_id,),
        ).fetchall()
        return [ShotRecord.from_row(row) for row in rows]

    def delete_shot(self, shot_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM shots WHERE shot_id = ?;", (shot_id,)
            )
        return cur.rowcount > 0

    # --- Characters ---

    def upsert_character(self, character: Character) -> Character:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO characters (character_id, film_id, name, notes, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(character_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    name = excluded.name,
                    notes = excluded.notes;
                """,
                (
                    character.character_id,
                    character.film_id,
                    character.name,
                    character.notes,
                    character.created_at,
                ),
            )
        return character

    def get_character(self, character_id: str) -> Character | None:
        row = self._conn.execute(
            "SELECT * FROM characters WHERE character_id = ?;", (character_id,)
        ).fetchone()
        return Character.from_row(row) if row else None

    def list_characters(self, film_id: str) -> list[Character]:
        rows = self._conn.execute(
            "SELECT * FROM characters WHERE film_id = ? ORDER BY name ASC;",
            (film_id,),
        ).fetchall()
        return [Character.from_row(row) for row in rows]

    def delete_character(self, character_id: str) -> bool:
        char = self.get_character(character_id)
        if char is not None:
            self.delete_assignments_for_entity(char.film_id, "character", character_id)
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM characters WHERE character_id = ?;", (character_id,)
            )
        return cur.rowcount > 0

    # --- Appearance Variants ---

    def upsert_variant(self, variant: AppearanceVariant) -> AppearanceVariant:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO appearance_variants (
                    variant_id, character_id, costume_id, hair_profile, skin_reference,
                    lighting_context, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(variant_id) DO UPDATE SET
                    character_id = excluded.character_id,
                    costume_id = excluded.costume_id,
                    hair_profile = excluded.hair_profile,
                    skin_reference = excluded.skin_reference,
                    lighting_context = excluded.lighting_context,
                    notes = excluded.notes;
                """,
                (
                    variant.variant_id,
                    variant.character_id,
                    variant.costume_id,
                    variant.hair_profile,
                    variant.skin_reference,
                    variant.lighting_context,
                    variant.notes,
                    variant.created_at,
                ),
            )
        return variant

    def get_variant(self, variant_id: str) -> AppearanceVariant | None:
        row = self._conn.execute(
            "SELECT * FROM appearance_variants WHERE variant_id = ?;", (variant_id,)
        ).fetchone()
        return AppearanceVariant.from_row(row) if row else None

    def list_variants(self, character_id: str) -> list[AppearanceVariant]:
        rows = self._conn.execute(
            "SELECT * FROM appearance_variants WHERE character_id = ? ORDER BY costume_id ASC;",
            (character_id,),
        ).fetchall()
        return [AppearanceVariant.from_row(row) for row in rows]

    def list_variants_for_film(self, film_id: str) -> list[AppearanceVariant]:
        rows = self._conn.execute(
            """
            SELECT av.* FROM appearance_variants av
            JOIN characters c ON av.character_id = c.character_id
            WHERE c.film_id = ?
            ORDER BY c.name ASC, av.costume_id ASC;
            """,
            (film_id,),
        ).fetchall()
        return [AppearanceVariant.from_row(row) for row in rows]

    def delete_variant(self, variant_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM appearance_variants WHERE variant_id = ?;", (variant_id,)
            )
        return cur.rowcount > 0

    # --- Objects ---

    def upsert_object(self, obj: ColorBibleObject) -> ColorBibleObject:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO objects (object_id, film_id, kind, name, palette_json, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    kind = excluded.kind,
                    name = excluded.name,
                    palette_json = excluded.palette_json,
                    notes = excluded.notes;
                """,
                (
                    obj.object_id,
                    obj.film_id,
                    obj.kind,
                    obj.name,
                    obj.palette_json,
                    obj.notes,
                    obj.created_at,
                ),
            )
        return obj

    def get_object(self, object_id: str) -> ColorBibleObject | None:
        row = self._conn.execute(
            "SELECT * FROM objects WHERE object_id = ?;", (object_id,)
        ).fetchone()
        return ColorBibleObject.from_row(row) if row else None

    def list_objects(self, film_id: str) -> list[ColorBibleObject]:
        rows = self._conn.execute(
            "SELECT * FROM objects WHERE film_id = ? ORDER BY kind ASC, name ASC;",
            (film_id,),
        ).fetchall()
        return [ColorBibleObject.from_row(row) for row in rows]

    def delete_object(self, object_id: str) -> bool:
        obj = self.get_object(object_id)
        if obj is not None:
            self.delete_assignments_for_entity(obj.film_id, "object", object_id)
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM objects WHERE object_id = ?;", (object_id,)
            )
        return cur.rowcount > 0

    # --- Locations ---

    def upsert_location(self, location: Location) -> Location:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO locations (
                    location_id, film_id, set_name, lighting_context, palette_json, notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(location_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    set_name = excluded.set_name,
                    lighting_context = excluded.lighting_context,
                    palette_json = excluded.palette_json,
                    notes = excluded.notes;
                """,
                (
                    location.location_id,
                    location.film_id,
                    location.set_name,
                    location.lighting_context,
                    location.palette_json,
                    location.notes,
                    location.created_at,
                ),
            )
        return location

    def get_location(self, location_id: str) -> Location | None:
        row = self._conn.execute(
            "SELECT * FROM locations WHERE location_id = ?;", (location_id,)
        ).fetchone()
        return Location.from_row(row) if row else None

    def list_locations(self, film_id: str) -> list[Location]:
        rows = self._conn.execute(
            "SELECT * FROM locations WHERE film_id = ? ORDER BY set_name ASC;",
            (film_id,),
        ).fetchall()
        return [Location.from_row(row) for row in rows]

    def delete_location(self, location_id: str) -> bool:
        loc = self.get_location(location_id)
        if loc is not None:
            self.delete_assignments_for_entity(loc.film_id, "location", location_id)
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM locations WHERE location_id = ?;", (location_id,)
            )
        return cur.rowcount > 0

    # --- Shot Entity Assignments ---

    def upsert_assignment(self, assignment: ShotEntityAssignment) -> ShotEntityAssignment:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO shot_entity_assignments (
                    assignment_id, film_id, shot_id, entity_type, entity_id, variant_id,
                    valid_shot_range_start, valid_shot_range_end, confidence, approval_status,
                    provenance, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assignment_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    shot_id = excluded.shot_id,
                    entity_type = excluded.entity_type,
                    entity_id = excluded.entity_id,
                    variant_id = excluded.variant_id,
                    valid_shot_range_start = excluded.valid_shot_range_start,
                    valid_shot_range_end = excluded.valid_shot_range_end,
                    confidence = excluded.confidence,
                    approval_status = excluded.approval_status,
                    provenance = excluded.provenance,
                    updated_at = excluded.updated_at;
                """,
                (
                    assignment.assignment_id,
                    assignment.film_id,
                    assignment.shot_id,
                    assignment.entity_type,
                    assignment.entity_id,
                    assignment.variant_id,
                    assignment.valid_shot_range_start,
                    assignment.valid_shot_range_end,
                    assignment.confidence,
                    assignment.approval_status,
                    assignment.provenance,
                    assignment.created_at,
                    assignment.updated_at,
                ),
            )
        return assignment

    def get_assignment(self, assignment_id: str) -> ShotEntityAssignment | None:
        row = self._conn.execute(
            "SELECT * FROM shot_entity_assignments WHERE assignment_id = ?;",
            (assignment_id,),
        ).fetchone()
        return ShotEntityAssignment.from_row(row) if row else None

    def list_assignments(
        self,
        film_id: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        approval_status: str | None = None,
    ) -> list[ShotEntityAssignment]:
        clauses = ["film_id = ?"]
        params: list[Any] = [film_id]
        if entity_type is not None:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if approval_status is not None:
            clauses.append("approval_status = ?")
            params.append(approval_status)
        where_sql = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM shot_entity_assignments WHERE {where_sql} ORDER BY valid_shot_range_start ASC, valid_shot_range_end ASC;",
            params,
        ).fetchall()
        return [ShotEntityAssignment.from_row(row) for row in rows]

    def find_assignments_for_shot_index(
        self,
        film_id: str,
        shot_index: int,
    ) -> list[ShotEntityAssignment]:
        rows = self._conn.execute(
            """
            SELECT * FROM shot_entity_assignments
            WHERE film_id = ?
              AND valid_shot_range_start <= ?
              AND valid_shot_range_end >= ?
            ORDER BY valid_shot_range_start ASC, valid_shot_range_end ASC;
            """,
            (film_id, shot_index, shot_index),
        ).fetchall()
        return [ShotEntityAssignment.from_row(row) for row in rows]

    def find_assignments_for_range(
        self,
        film_id: str,
        start_index: int,
        end_index: int,
    ) -> list[ShotEntityAssignment]:
        rows = self._conn.execute(
            """
            SELECT * FROM shot_entity_assignments
            WHERE film_id = ?
              AND valid_shot_range_start <= ?
              AND valid_shot_range_end >= ?
            ORDER BY valid_shot_range_start ASC, valid_shot_range_end ASC;
            """,
            (film_id, end_index, start_index),
        ).fetchall()
        return [ShotEntityAssignment.from_row(row) for row in rows]

    def delete_assignment(self, assignment_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM shot_entity_assignments WHERE assignment_id = ?;",
                (assignment_id,),
            )
        return cur.rowcount > 0

    def delete_assignments_for_entity(
        self,
        film_id: str,
        entity_type: str,
        entity_id: str,
        variant_id: str | None = None,
    ) -> int:
        with self._conn:
            if variant_id is not None:
                cur = self._conn.execute(
                    """
                    DELETE FROM shot_entity_assignments
                    WHERE film_id = ? AND entity_type = ? AND entity_id = ? AND variant_id = ?;
                    """,
                    (film_id, entity_type, entity_id, variant_id),
                )
            else:
                cur = self._conn.execute(
                    """
                    DELETE FROM shot_entity_assignments
                    WHERE film_id = ? AND entity_type = ? AND entity_id = ?;
                    """,
                    (film_id, entity_type, entity_id),
                )
        return cur.rowcount

    # --- Color References ---

    def upsert_reference(self, reference: ColorReferenceRecord) -> ColorReferenceRecord:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO color_references (
                    ref_id, film_id, entity_type, entity_id, variant_id, source_shot_id,
                    source_keyframe_path, mask_uri, lab_stats_json, rgb_palette_json,
                    provenance, approval_status, approved_by, approved_at, version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ref_id) DO UPDATE SET
                    film_id = excluded.film_id,
                    entity_type = excluded.entity_type,
                    entity_id = excluded.entity_id,
                    variant_id = excluded.variant_id,
                    source_shot_id = excluded.source_shot_id,
                    source_keyframe_path = excluded.source_keyframe_path,
                    mask_uri = excluded.mask_uri,
                    lab_stats_json = excluded.lab_stats_json,
                    rgb_palette_json = excluded.rgb_palette_json,
                    provenance = excluded.provenance,
                    approval_status = excluded.approval_status,
                    approved_by = excluded.approved_by,
                    approved_at = excluded.approved_at,
                    version = excluded.version;
                """,
                (
                    reference.ref_id,
                    reference.film_id,
                    reference.entity_type,
                    reference.entity_id,
                    reference.variant_id,
                    reference.source_shot_id,
                    reference.source_keyframe_path,
                    reference.mask_uri,
                    reference.lab_stats_json,
                    reference.rgb_palette_json,
                    reference.provenance,
                    reference.approval_status,
                    reference.approved_by,
                    reference.approved_at,
                    reference.version,
                    reference.created_at,
                ),
            )
        return reference

    def get_reference(self, ref_id: str) -> ColorReferenceRecord | None:
        row = self._conn.execute(
            "SELECT * FROM color_references WHERE ref_id = ?;", (ref_id,)
        ).fetchone()
        return ColorReferenceRecord.from_row(row) if row else None

    def list_references(
        self,
        film_id: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        variant_id: str | None = None,
        approval_status: str | None = None,
    ) -> list[ColorReferenceRecord]:
        clauses = ["film_id = ?"]
        params: list[Any] = [film_id]
        if entity_type is not None:
            clauses.append("entity_type = ?")
            params.append(entity_type)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if variant_id is not None:
            clauses.append("variant_id = ?")
            params.append(variant_id)
        if approval_status is not None:
            clauses.append("approval_status = ?")
            params.append(approval_status)
        where_sql = " AND ".join(clauses)
        rows = self._conn.execute(
            f"SELECT * FROM color_references WHERE {where_sql} ORDER BY entity_type ASC, entity_id ASC, variant_id ASC, version DESC;",
            params,
        ).fetchall()
        return [ColorReferenceRecord.from_row(row) for row in rows]

    def list_approved_references(self, film_id: str) -> list[ColorReferenceRecord]:
        rows = self._conn.execute(
            """
            SELECT * FROM color_references
            WHERE film_id = ? AND approval_status = 'approved'
            ORDER BY entity_type ASC, entity_id ASC, variant_id ASC, version DESC;
            """,
            (film_id,),
        ).fetchall()
        return [ColorReferenceRecord.from_row(row) for row in rows]

    def update_reference_approval(
        self,
        ref_id: str,
        approval_status: str,
        approved_by: str | None = None,
        approved_at: str | None = None,
    ) -> bool:
        if approval_status not in ("draft", "pending_review", "approved", "rejected"):
            raise ValueError(f"Invalid approval status: {approval_status}")
        resolved_approved_at = approved_at
        if approval_status == "approved" and resolved_approved_at is None:
            resolved_approved_at = _utcnow_iso()
        with self._conn:
            cur = self._conn.execute(
                """
                UPDATE color_references
                SET approval_status = ?,
                    approved_by = ?,
                    approved_at = ?
                WHERE ref_id = ?;
                """,
                (approval_status, approved_by, resolved_approved_at, ref_id),
            )
        return cur.rowcount > 0

    def delete_reference(self, ref_id: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM color_references WHERE ref_id = ?;", (ref_id,)
            )
        return cur.rowcount > 0
