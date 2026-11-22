from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from .color_bible_db import (
    AppearanceVariant,
    Character,
    ColorBibleObject,
    ColorBibleRepository,
    ColorReferenceRecord,
    Film,
    Location,
    ShotEntityAssignment,
    ShotRecord,
    SQLiteColorBibleRepository,
    _utcnow_iso,
    resolve_color_bible_db,
)
from .color_science import (
    compute_lab_percentiles,
    extract_rgb_palette,
    load_image_rgb,
    load_mask,
)
from .manifests import read_manifest_details


def parse_time_seconds(value: str | float | int | None) -> float:
    """Parse a timecode string ('HH:MM:SS.mmm' or 'MM:SS.mmm') or float into seconds."""
    if value is None:
        return 0.0
    val_str = str(value).strip()
    if not val_str:
        return 0.0
    parts = val_str.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(val_str)
    except ValueError:
        return 0.0


class ColorBibleService:
    """Domain service orchestrating the Color Bible module business logic."""

    def __init__(
        self,
        repo: ColorBibleRepository | None = None,
        db_path: Path | str | None = None,
    ) -> None:
        if repo is not None:
            self.repo = repo
        else:
            resolved_path = resolve_color_bible_db(db_path) if db_path != ":memory:" else Path(":memory:")
            self.repo = SQLiteColorBibleRepository(resolved_path)
        self.repo.init_db()

    def close(self) -> None:
        """Close the underlying repository connection."""
        self.repo.close()

    def __enter__(self) -> ColorBibleService:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # --- Manifest Synchronization ---

    def sync_manifest(
        self,
        film_id: str,
        manifest_path: Path | str,
        title: str | None = None,
    ) -> dict[str, int]:
        """Synchronize an ARP shot manifest into films and shots idempotently."""
        if not film_id or not film_id.strip():
            raise ValueError("film_id must not be empty.")
        film_id = film_id.strip()
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        source_video, _fields, rows = read_manifest_details(path)
        resolved_title = title if title else path.stem

        existing_film = self.repo.get_film(film_id)
        if existing_film:
            film = Film(
                film_id=film_id,
                title=title or existing_film.title,
                source_video=source_video or existing_film.source_video,
                created_at=existing_film.created_at,
                updated_at=_utcnow_iso(),
            )
        else:
            film = Film(
                film_id=film_id,
                title=resolved_title,
                source_video=source_video or None,
            )
        self.repo.upsert_film(film)

        shots: list[ShotRecord] = []
        for idx, row in enumerate(rows):
            # Parse shot index
            raw_index = row.get("shot_index") or row.get("index")
            if raw_index is not None and str(raw_index).strip() != "":
                try:
                    shot_index = int(float(str(raw_index).strip()))
                except ValueError:
                    shot_index = idx
            else:
                shot_index = idx

            # Parse start and end times
            if "start_time" in row and str(row["start_time"]).strip() != "":
                try:
                    start_time = float(row["start_time"])
                except ValueError:
                    start_time = parse_time_seconds(row.get("start_time"))
            else:
                start_time = parse_time_seconds(row.get("start"))

            if "end_time" in row and str(row["end_time"]).strip() != "":
                try:
                    end_time = float(row["end_time"])
                except ValueError:
                    end_time = parse_time_seconds(row.get("end_time"))
            else:
                end_time = parse_time_seconds(row.get("end"))
                if end_time == 0.0 and start_time > 0.0:
                    end_time = start_time

            # Parse start and end frames
            raw_sf = row.get("start_frame")
            if raw_sf is not None and str(raw_sf).strip() != "":
                try:
                    start_frame = int(float(str(raw_sf).strip()))
                except ValueError:
                    start_frame = int(round(start_time * 24.0))
            else:
                start_frame = int(round(start_time * 24.0))

            raw_ef = row.get("end_frame")
            if raw_ef is not None and str(raw_ef).strip() != "":
                try:
                    end_frame = int(float(str(raw_ef).strip()))
                except ValueError:
                    end_frame = max(start_frame, int(round(end_time * 24.0)))
            else:
                end_frame = max(start_frame, int(round(end_time * 24.0)))

            shot_id = f"{film_id}_{shot_index}"
            shots.append(
                ShotRecord(
                    shot_id=shot_id,
                    film_id=film_id,
                    shot_index=shot_index,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=start_time,
                    end_time=end_time,
                    sequence_id=row.get("sequence_id") or None,
                    source_reference=row.get("source_reference") or row.get("anchor_image") or None,
                    color_reference=row.get("color_reference") or None,
                    qc_status=row.get("qc_status") or "pending",
                )
            )

        synced_count = self.repo.upsert_shots(shots)
        return {"shots_synced": synced_count}

    # --- Film Management ---

    def create_film(
        self,
        film_id: str,
        title: str | None = None,
        source_video: str | None = None,
    ) -> Film:
        """Create and store a film in the Color Bible repository."""
        if not film_id or not film_id.strip():
            raise ValueError("film_id must not be empty.")
        film_id = film_id.strip()
        existing_film = self.repo.get_film(film_id)
        resolved_title = title.strip() if title and title.strip() else film_id
        resolved_video = source_video.strip() if source_video and source_video.strip() else None

        if existing_film:
            film = Film(
                film_id=film_id,
                title=title.strip() if title and title.strip() else existing_film.title,
                source_video=resolved_video or existing_film.source_video,
                created_at=existing_film.created_at,
                updated_at=_utcnow_iso(),
            )
        else:
            film = Film(
                film_id=film_id,
                title=resolved_title,
                source_video=resolved_video,
            )
        return self.repo.upsert_film(film)

    # --- Character and Appearance Variants ---

    def create_character(
        self,
        film_id: str,
        name: str,
        notes: str = "",
    ) -> Character:
        """Create and store a character for a film."""
        if not film_id or not film_id.strip():
            raise ValueError("film_id must not be empty.")
        film_id = film_id.strip()
        if not name or not name.strip():
            raise ValueError("Character name must not be empty.")

        if self.repo.get_film(film_id) is None:
            self.repo.upsert_film(Film(film_id=film_id, title=film_id))

        char_id = f"char_{uuid.uuid4().hex[:8]}"
        character = Character(
            character_id=char_id,
            film_id=film_id,
            name=name.strip(),
            notes=notes.strip() if notes else None,
        )
        return self.repo.upsert_character(character)

    def create_appearance_variant(
        self,
        character_id: str,
        costume_id: str,
        hair_profile: str = "",
        skin_reference: str = "",
        lighting_context: str = "",
        notes: str = "",
    ) -> AppearanceVariant:
        """Create and store an appearance variant for a character."""
        character = self.repo.get_character(character_id)
        if character is None:
            raise ValueError(f"Character '{character_id}' does not exist.")
        if not costume_id or not costume_id.strip():
            raise ValueError("Costume ID must not be empty.")

        variant_id = f"var_{uuid.uuid4().hex[:8]}"
        variant = AppearanceVariant(
            variant_id=variant_id,
            character_id=character_id,
            costume_id=costume_id.strip(),
            hair_profile=hair_profile.strip() if hair_profile else None,
            skin_reference=skin_reference.strip() if skin_reference else None,
            lighting_context=lighting_context.strip() if lighting_context else None,
            notes=notes.strip() if notes else None,
        )
        return self.repo.upsert_variant(variant)

    # --- Objects and Locations ---

    def create_object(
        self,
        film_id: str,
        kind: str,
        name: str,
        palette: list[str] | None = None,
        notes: str = "",
    ) -> ColorBibleObject:
        """Create and store an object (prop, vehicle, or other) for a film."""
        if not film_id or not film_id.strip():
            raise ValueError("film_id must not be empty.")
        film_id = film_id.strip()
        if kind not in ("prop", "vehicle", "other"):
            raise ValueError(f"Invalid object kind '{kind}'. Must be one of: 'prop', 'vehicle', 'other'.")
        if not name or not name.strip():
            raise ValueError("Object name must not be empty.")

        if self.repo.get_film(film_id) is None:
            self.repo.upsert_film(Film(film_id=film_id, title=film_id))

        palette_json = json.dumps(palette) if palette is not None else None
        obj = ColorBibleObject(
            object_id=f"obj_{uuid.uuid4().hex[:8]}",
            film_id=film_id,
            kind=kind,
            name=name.strip(),
            palette_json=palette_json,
            notes=notes.strip() if notes else None,
        )
        return self.repo.upsert_object(obj)

    def create_location(
        self,
        film_id: str,
        set_name: str,
        lighting_context: str = "",
        palette: list[str] | None = None,
        notes: str = "",
    ) -> Location:
        """Create and store a location for a film."""
        if not film_id or not film_id.strip():
            raise ValueError("film_id must not be empty.")
        film_id = film_id.strip()
        if not set_name or not set_name.strip():
            raise ValueError("Location set_name must not be empty.")

        if self.repo.get_film(film_id) is None:
            self.repo.upsert_film(Film(film_id=film_id, title=film_id))

        palette_json = json.dumps(palette) if palette is not None else None
        loc = Location(
            location_id=f"loc_{uuid.uuid4().hex[:8]}",
            film_id=film_id,
            set_name=set_name.strip(),
            lighting_context=lighting_context.strip() if lighting_context else None,
            palette_json=palette_json,
            notes=notes.strip() if notes else None,
        )
        return self.repo.upsert_location(loc)

    # --- Shot Entity Assignments ---

    def assign_entity_ranges(
        self,
        film_id: str,
        entity_type: str,
        entity_id: str,
        ranges: list[tuple[int, int]],
        variant_id: str | None = None,
        confidence: float = 1.0,
        approval_status: str = "approved",
        provenance: str = "manual",
    ) -> list[ShotEntityAssignment]:
        """Assign an entity (and optional variant) across multiple shot index ranges."""
        if not film_id or not film_id.strip():
            raise ValueError("film_id must not be empty.")
        film_id = film_id.strip()
        if entity_type not in ("character", "object", "location"):
            raise ValueError(f"Invalid entity_type '{entity_type}'. Must be 'character', 'object', or 'location'.")
        if approval_status not in ("draft", "pending_review", "approved", "rejected"):
            raise ValueError(f"Invalid approval_status '{approval_status}'.")

        if self.repo.get_film(film_id) is None:
            self.repo.upsert_film(Film(film_id=film_id, title=film_id))

        assignments: list[ShotEntityAssignment] = []
        for start_idx, end_idx in ranges:
            if start_idx > end_idx:
                raise ValueError(f"Invalid range: start ({start_idx}) > end ({end_idx})")
            asgn = ShotEntityAssignment(
                assignment_id=f"asgn_{uuid.uuid4().hex[:8]}",
                film_id=film_id,
                entity_type=entity_type,
                entity_id=entity_id,
                variant_id=variant_id,
                valid_shot_range_start=start_idx,
                valid_shot_range_end=end_idx,
                confidence=confidence,
                approval_status=approval_status,
                provenance=provenance,
            )
            self.repo.upsert_assignment(asgn)
            assignments.append(asgn)
        return assignments

    def get_shot_palette_assignments(
        self,
        film_id: str,
        shot_index: int,
    ) -> list[dict[str, Any]]:
        """Return all active entity assignments, metadata, and approved color references for a shot."""
        assignments = self.repo.find_assignments_for_shot_index(film_id, shot_index)
        results: list[dict[str, Any]] = []

        for asgn in assignments:
            item: dict[str, Any] = {
                "assignment": asgn.to_dict(),
                "entity_type": asgn.entity_type,
                "entity_id": asgn.entity_id,
                "variant_id": asgn.variant_id,
                "entity": None,
                "variant": None,
                "color_references": [],
            }
            if asgn.entity_type == "character":
                char = self.repo.get_character(asgn.entity_id)
                if char:
                    item["entity"] = char.to_dict()
                if asgn.variant_id:
                    var = self.repo.get_variant(asgn.variant_id)
                    if var:
                        item["variant"] = var.to_dict()
                refs = self.repo.list_references(
                    film_id,
                    entity_type="character",
                    entity_id=asgn.entity_id,
                    variant_id=asgn.variant_id,
                    approval_status="approved",
                )
                item["color_references"] = [r.to_dict() for r in refs]
            elif asgn.entity_type == "object":
                obj = self.repo.get_object(asgn.entity_id)
                if obj:
                    item["entity"] = obj.to_dict()
                refs = self.repo.list_references(
                    film_id,
                    entity_type="object",
                    entity_id=asgn.entity_id,
                    approval_status="approved",
                )
                item["color_references"] = [r.to_dict() for r in refs]
            elif asgn.entity_type == "location":
                loc = self.repo.get_location(asgn.entity_id)
                if loc:
                    item["entity"] = loc.to_dict()
                refs = self.repo.list_references(
                    film_id,
                    entity_type="location",
                    entity_id=asgn.entity_id,
                    approval_status="approved",
                )
                item["color_references"] = [r.to_dict() for r in refs]
            results.append(item)

        return results

    # --- Color References ---

    def register_reference_image(
        self,
        film_id: str,
        entity_type: str,
        entity_id: str,
        keyframe_path: Path | str,
        mask_uri: Path | str | None = None,
        variant_id: str | None = None,
        provenance: str = "manual",
        source_shot_id: str | None = None,
        auto_approve: bool = False,
        approved_by: str | None = None,
    ) -> ColorReferenceRecord:
        """Analyze a keyframe image, extract CIE-LAB percentiles and RGB palette, and store a reference."""
        if not film_id or not film_id.strip():
            raise ValueError("film_id must not be empty.")
        film_id = film_id.strip()
        if entity_type not in ("character", "object", "location"):
            raise ValueError(f"Invalid entity_type '{entity_type}'. Must be 'character', 'object', or 'location'.")
        if provenance not in ("historical_research", "ddcolor", "qwen2511", "manual"):
            raise ValueError(f"Invalid provenance '{provenance}'.")

        if self.repo.get_film(film_id) is None:
            self.repo.upsert_film(Film(film_id=film_id, title=film_id))

        keyframe_file = Path(keyframe_path)
        img_rgb = load_image_rgb(keyframe_file)
        mask = load_mask(Path(mask_uri)) if mask_uri else None

        lab_stats = compute_lab_percentiles(img_rgb, mask)
        rgb_palette = extract_rgb_palette(img_rgb, mask)

        existing_refs = self.repo.list_references(
            film_id,
            entity_type=entity_type,
            entity_id=entity_id,
            variant_id=variant_id,
        )
        version = max([r.version for r in existing_refs], default=0) + 1

        ref_id = f"ref_{uuid.uuid4().hex[:8]}"
        approval_status = "approved" if auto_approve else "draft"
        approved_by_val = (approved_by or "colorist") if auto_approve else None
        approved_at_val = _utcnow_iso() if auto_approve else None

        record = ColorReferenceRecord(
            ref_id=ref_id,
            film_id=film_id,
            entity_type=entity_type,
            entity_id=entity_id,
            variant_id=variant_id,
            source_shot_id=source_shot_id,
            source_keyframe_path=str(keyframe_path),
            mask_uri=str(mask_uri) if mask_uri else None,
            lab_stats_json=json.dumps(lab_stats),
            rgb_palette_json=json.dumps(rgb_palette),
            provenance=provenance,
            approval_status=approval_status,
            approved_by=approved_by_val,
            approved_at=approved_at_val,
            version=version,
        )
        return self.repo.upsert_reference(record)

    def approve_reference(
        self,
        ref_id: str,
        approved_by: str = "colorist",
    ) -> ColorReferenceRecord:
        """Mark a reference as approved."""
        ok = self.repo.update_reference_approval(ref_id, "approved", approved_by=approved_by)
        if not ok:
            raise ValueError(f"Color reference '{ref_id}' not found.")
        ref = self.repo.get_reference(ref_id)
        if ref is None:
            raise ValueError(f"Color reference '{ref_id}' not found.")
        return ref

    def reject_reference(
        self,
        ref_id: str,
    ) -> ColorReferenceRecord:
        """Mark a reference as rejected."""
        ok = self.repo.update_reference_approval(ref_id, "rejected")
        if not ok:
            raise ValueError(f"Color reference '{ref_id}' not found.")
        ref = self.repo.get_reference(ref_id)
        if ref is None:
            raise ValueError(f"Color reference '{ref_id}' not found.")
        return ref

    # --- Snapshot Hashing ---

    def get_snapshot_hash(self, film_id: str) -> str:
        """Compute a canonical, deterministic SHA-256 snapshot hash across approved references and assignments."""
        assignments = self.repo.list_assignments(film_id, approval_status="approved")
        references = self.repo.list_approved_references(film_id)

        sorted_assignments = sorted(
            assignments,
            key=lambda a: (
                a.valid_shot_range_start,
                a.valid_shot_range_end,
                a.entity_type,
                a.entity_id,
                a.variant_id or "",
                a.assignment_id,
            ),
        )

        sorted_references = sorted(
            references,
            key=lambda r: (
                r.entity_type,
                r.entity_id,
                r.variant_id or "",
                r.version,
                r.ref_id,
            ),
        )

        canonical_payload = {
            "assignments": [a.to_dict() for a in sorted_assignments],
            "film_id": film_id,
            "references": [r.to_dict() for r in sorted_references],
        }

        raw = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"sha256:{hashlib.sha256(raw).hexdigest()}"

    # --- Passthrough Convenience Methods ---

    def get_film(self, film_id: str) -> Film | None:
        return self.repo.get_film(film_id)

    def list_films(self) -> list[Film]:
        return self.repo.list_films()

    def delete_film(self, film_id: str) -> bool:
        return self.repo.delete_film(film_id)

    def get_character(self, character_id: str) -> Character | None:
        return self.repo.get_character(character_id)

    def list_characters(self, film_id: str) -> list[Character]:
        return self.repo.list_characters(film_id)

    def delete_character(self, character_id: str) -> bool:
        return self.repo.delete_character(character_id)

    def get_variant(self, variant_id: str) -> AppearanceVariant | None:
        return self.repo.get_variant(variant_id)

    def list_variants(self, character_id: str) -> list[AppearanceVariant]:
        return self.repo.list_variants(character_id)

    def list_variants_for_film(self, film_id: str) -> list[AppearanceVariant]:
        return self.repo.list_variants_for_film(film_id)

    def delete_variant(self, variant_id: str) -> bool:
        return self.repo.delete_variant(variant_id)

    def get_object(self, object_id: str) -> ColorBibleObject | None:
        return self.repo.get_object(object_id)

    def list_objects(self, film_id: str) -> list[ColorBibleObject]:
        return self.repo.list_objects(film_id)

    def delete_object(self, object_id: str) -> bool:
        return self.repo.delete_object(object_id)

    def get_location(self, location_id: str) -> Location | None:
        return self.repo.get_location(location_id)

    def list_locations(self, film_id: str) -> list[Location]:
        return self.repo.list_locations(film_id)

    def delete_location(self, location_id: str) -> bool:
        return self.repo.delete_location(location_id)

    def get_shot(self, shot_id: str) -> ShotRecord | None:
        return self.repo.get_shot(shot_id)

    def get_shot_by_index(self, film_id: str, shot_index: int) -> ShotRecord | None:
        return self.repo.get_shot_by_index(film_id, shot_index)

    def list_shots(self, film_id: str) -> list[ShotRecord]:
        return self.repo.list_shots(film_id)

    def delete_shot(self, shot_id: str) -> bool:
        return self.repo.delete_shot(shot_id)

    def get_assignment(self, assignment_id: str) -> ShotEntityAssignment | None:
        return self.repo.get_assignment(assignment_id)

    def list_assignments(
        self,
        film_id: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        approval_status: str | None = None,
    ) -> list[ShotEntityAssignment]:
        return self.repo.list_assignments(
            film_id,
            entity_type=entity_type,
            entity_id=entity_id,
            approval_status=approval_status,
        )

    def delete_assignment(self, assignment_id: str) -> bool:
        return self.repo.delete_assignment(assignment_id)

    def delete_assignments_for_entity(
        self,
        film_id: str,
        entity_type: str,
        entity_id: str,
        variant_id: str | None = None,
    ) -> int:
        return self.repo.delete_assignments_for_entity(film_id, entity_type, entity_id, variant_id)

    def get_reference(self, ref_id: str) -> ColorReferenceRecord | None:
        return self.repo.get_reference(ref_id)

    def list_references(
        self,
        film_id: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        variant_id: str | None = None,
        approval_status: str | None = None,
    ) -> list[ColorReferenceRecord]:
        return self.repo.list_references(
            film_id,
            entity_type=entity_type,
            entity_id=entity_id,
            variant_id=variant_id,
            approval_status=approval_status,
        )

    def list_approved_references(self, film_id: str) -> list[ColorReferenceRecord]:
        return self.repo.list_approved_references(film_id)

    def delete_reference(self, ref_id: str) -> bool:
        return self.repo.delete_reference(ref_id)
