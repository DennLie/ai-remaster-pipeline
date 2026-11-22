from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
import pytest

from ai_remaster_gui.color_bible_db import (
    AppearanceVariant,
    Character,
    ColorBibleObject,
    ColorReferenceRecord,
    Film,
    Location,
    Sequence,
    ShotEntityAssignment,
    ShotRecord,
    SQLiteColorBibleRepository,
    get_connection,
    resolve_color_bible_db,
)
from ai_remaster_gui.config import ROOT


@pytest.fixture
def mem_repo() -> SQLiteColorBibleRepository:
    repo = SQLiteColorBibleRepository(":memory:")
    repo.init_db()
    yield repo
    repo.close()


def test_resolve_color_bible_db(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Explicit path
    p = resolve_color_bible_db("custom/path/db.sqlite")
    assert p == Path("custom/path/db.sqlite")

    p2 = resolve_color_bible_db(Path("another/path.db"))
    assert p2 == Path("another/path.db")

    # 2. Env variable
    monkeypatch.setenv("ARP_COLOR_BIBLE_DB", "env/path/color.db")
    p_env = resolve_color_bible_db()
    assert p_env == Path("env/path/color.db")

    # 3. Default fallback
    monkeypatch.delenv("ARP_COLOR_BIBLE_DB", raising=False)
    p_def = resolve_color_bible_db()
    assert p_def == ROOT / "manifests" / "color_bible.db"


def test_get_connection_memory_and_file() -> None:
    # In-memory connection
    conn_mem = get_connection(":memory:")
    cur = conn_mem.execute("PRAGMA foreign_keys;")
    assert cur.fetchone()[0] == 1
    conn_mem.close()

    # File database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_file = Path(tmpdir) / "sub" / "test.db"
        conn_file = get_connection(db_file)
        cur_fk = conn_file.execute("PRAGMA foreign_keys;")
        assert cur_fk.fetchone()[0] == 1

        cur_wal = conn_file.execute("PRAGMA journal_mode;")
        assert str(cur_wal.fetchone()[0]).lower() == "wal"

        cur_busy = conn_file.execute("PRAGMA busy_timeout;")
        assert cur_busy.fetchone()[0] == 5000
        conn_file.close()


def test_schema_migrations_idempotency() -> None:
    repo = SQLiteColorBibleRepository(":memory:")
    first_run = repo.init_db()
    assert first_run == [1]
    assert repo.get_applied_migrations() == [1]

    # Running second time must be a no-op
    second_run = repo.init_db()
    assert second_run == []
    assert repo.get_applied_migrations() == [1]
    repo.close()


def test_film_and_sequence_crud(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_001", title="Test Noir Film", source_video="media/test.mp4")
    mem_repo.upsert_film(film)

    retrieved = mem_repo.get_film("film_001")
    assert retrieved is not None
    assert retrieved.title == "Test Noir Film"
    assert retrieved.source_video == "media/test.mp4"

    # Update film
    film.title = "Updated Title"
    mem_repo.upsert_film(film)
    retrieved_updated = mem_repo.get_film("film_001")
    assert retrieved_updated is not None
    assert retrieved_updated.title == "Updated Title"

    # Sequences
    seq1 = Sequence(sequence_id="seq_1", film_id="film_001", sequence_index=1, name="Opening")
    seq2 = Sequence(sequence_id="seq_2", film_id="film_001", sequence_index=2, name="Docks")
    mem_repo.upsert_sequence(seq1)
    mem_repo.upsert_sequence(seq2)

    seqs = mem_repo.list_sequences("film_001")
    assert len(seqs) == 2
    assert seqs[0].sequence_id == "seq_1"
    assert seqs[1].sequence_id == "seq_2"

    assert mem_repo.delete_sequence("seq_2") is True
    assert mem_repo.get_sequence("seq_2") is None


def test_shot_crud_and_batch_upsert(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_001", title="Test Film")
    mem_repo.upsert_film(film)

    shot1 = ShotRecord(
        shot_id="shot_0001",
        film_id="film_001",
        shot_index=1,
        start_frame=0,
        end_frame=120,
        start_time=0.0,
        end_time=5.0,
    )
    mem_repo.upsert_shot(shot1)

    s = mem_repo.get_shot("shot_0001")
    assert s is not None
    assert s.shot_index == 1

    by_idx = mem_repo.get_shot_by_index("film_001", 1)
    assert by_idx is not None
    assert by_idx.shot_id == "shot_0001"

    # Batch upsert
    batch = [
        ShotRecord(
            shot_id=f"shot_{i:04d}",
            film_id="film_001",
            shot_index=i,
            start_frame=i * 100,
            end_frame=(i + 1) * 100,
            start_time=float(i * 4),
            end_time=float((i + 1) * 4),
        )
        for i in range(2, 6)
    ]
    count = mem_repo.upsert_shots(batch)
    assert count == 4

    all_shots = mem_repo.list_shots("film_001")
    assert len(all_shots) == 5

    assert mem_repo.delete_shot("shot_0001") is True
    assert mem_repo.get_shot("shot_0001") is None


def test_characters_and_appearance_variants(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_001", title="Noir Mystery")
    mem_repo.upsert_film(film)

    char = Character(character_id="char_elena", film_id="film_001", name="Elena", notes="Protagonist")
    mem_repo.upsert_character(char)

    retrieved_char = mem_repo.get_character("char_elena")
    assert retrieved_char is not None
    assert retrieved_char.name == "Elena"

    # Appearance variants
    var1 = AppearanceVariant(
        variant_id="var_elena_trench",
        character_id="char_elena",
        costume_id="trench_coat",
        hair_profile="dark_updo",
        lighting_context="exterior_rain",
    )
    var2 = AppearanceVariant(
        variant_id="var_elena_gown",
        character_id="char_elena",
        costume_id="evening_gown",
        hair_profile="dark_curled",
        lighting_context="interior_ballroom",
    )
    mem_repo.upsert_variant(var1)
    mem_repo.upsert_variant(var2)

    variants = mem_repo.list_variants("char_elena")
    assert len(variants) == 2

    film_variants = mem_repo.list_variants_for_film("film_001")
    assert len(film_variants) == 2

    assert mem_repo.delete_variant("var_elena_gown") is True
    assert mem_repo.get_variant("var_elena_gown") is None


def test_objects_and_locations(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_001", title="Action Thriller")
    mem_repo.upsert_film(film)

    # Object
    obj = ColorBibleObject(
        object_id="obj_sedan",
        film_id="film_001",
        kind="vehicle",
        name="Vintage 1948 Sedan",
        palette_json='{"primary": "#2b3a42"}',
    )
    mem_repo.upsert_object(obj)

    retrieved_obj = mem_repo.get_object("obj_sedan")
    assert retrieved_obj is not None
    assert retrieved_obj.kind == "vehicle"

    objs = mem_repo.list_objects("film_001")
    assert len(objs) == 1

    # Location
    loc = Location(
        location_id="loc_warehouse",
        film_id="film_001",
        set_name="Abandoned Dock Warehouse",
        lighting_context="low_key_tungsten",
    )
    mem_repo.upsert_location(loc)

    retrieved_loc = mem_repo.get_location("loc_warehouse")
    assert retrieved_loc is not None
    assert retrieved_loc.set_name == "Abandoned Dock Warehouse"

    locs = mem_repo.list_locations("film_001")
    assert len(locs) == 1


def test_shot_entity_assignments_and_ranges(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_001", title="Feature Film")
    mem_repo.upsert_film(film)

    char = Character(character_id="char_elena", film_id="film_001", name="Elena")
    mem_repo.upsert_character(char)

    var1 = AppearanceVariant(variant_id="var_trench", character_id="char_elena", costume_id="trench")
    var2 = AppearanceVariant(variant_id="var_gown", character_id="char_elena", costume_id="gown")
    mem_repo.upsert_variant(var1)
    mem_repo.upsert_variant(var2)

    # Disjoint shot range assignments: Elena in Trench at shots 12-18 and shots 410-435
    assign1 = ShotEntityAssignment(
        assignment_id="asg_01",
        film_id="film_001",
        entity_type="character",
        entity_id="char_elena",
        variant_id="var_trench",
        valid_shot_range_start=12,
        valid_shot_range_end=18,
    )
    assign2 = ShotEntityAssignment(
        assignment_id="asg_02",
        film_id="film_001",
        entity_type="character",
        entity_id="char_elena",
        variant_id="var_trench",
        valid_shot_range_start=410,
        valid_shot_range_end=435,
    )
    # Elena in Gown at shots 500-550
    assign3 = ShotEntityAssignment(
        assignment_id="asg_03",
        film_id="film_001",
        entity_type="character",
        entity_id="char_elena",
        variant_id="var_gown",
        valid_shot_range_start=500,
        valid_shot_range_end=550,
    )
    mem_repo.upsert_assignment(assign1)
    mem_repo.upsert_assignment(assign2)
    mem_repo.upsert_assignment(assign3)

    # Query by shot index
    matches_15 = mem_repo.find_assignments_for_shot_index("film_001", 15)
    assert len(matches_15) == 1
    assert matches_15[0].variant_id == "var_trench"

    matches_50 = mem_repo.find_assignments_for_shot_index("film_001", 50)
    assert len(matches_50) == 0

    matches_420 = mem_repo.find_assignments_for_shot_index("film_001", 420)
    assert len(matches_420) == 1
    assert matches_420[0].variant_id == "var_trench"

    matches_520 = mem_repo.find_assignments_for_shot_index("film_001", 520)
    assert len(matches_520) == 1
    assert matches_520[0].variant_id == "var_gown"

    # Query by overlapping range [10, 14]
    overlap_matches = mem_repo.find_assignments_for_range("film_001", 10, 14)
    assert len(overlap_matches) == 1
    assert overlap_matches[0].assignment_id == "asg_01"

    # Delete assignment for entity
    deleted = mem_repo.delete_assignments_for_entity("film_001", "character", "char_elena", "var_gown")
    assert deleted == 1
    assert len(mem_repo.list_assignments("film_001", "character", "char_elena")) == 2


def test_color_references_and_approval_workflow(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_001", title="Feature Film")
    mem_repo.upsert_film(film)

    char = Character(character_id="char_elena", film_id="film_001", name="Elena")
    mem_repo.upsert_character(char)

    var = AppearanceVariant(variant_id="var_trench", character_id="char_elena", costume_id="trench")
    mem_repo.upsert_variant(var)

    ref = ColorReferenceRecord(
        ref_id="ref_01",
        film_id="film_001",
        entity_type="character",
        entity_id="char_elena",
        variant_id="var_trench",
        source_keyframe_path="references/keyframe_01.png",
        lab_stats_json='{"L": {"p50": 62.5}, "a": {"p50": 14.2}, "b": {"p50": 18.9}}',
        rgb_palette_json='[{"hex": "#a3705c", "ratio": 0.45}]',
        provenance="historical_research",
        approval_status="draft",
    )
    mem_repo.upsert_reference(ref)

    retrieved = mem_repo.get_reference("ref_01")
    assert retrieved is not None
    assert retrieved.approval_status == "draft"

    # List approved should be empty initially
    assert len(mem_repo.list_approved_references("film_001")) == 0

    # Approve reference
    ok = mem_repo.update_reference_approval("ref_01", "approved", approved_by="colorist_alice")
    assert ok is True

    approved_list = mem_repo.list_approved_references("film_001")
    assert len(approved_list) == 1
    assert approved_list[0].approval_status == "approved"
    assert approved_list[0].approved_by == "colorist_alice"
    assert approved_list[0].approved_at is not None

    # Invalid status should raise ValueError
    with pytest.raises(ValueError):
        mem_repo.update_reference_approval("ref_01", "non_existent_status")

    # Reject reference
    mem_repo.update_reference_approval("ref_01", "rejected")
    assert len(mem_repo.list_approved_references("film_001")) == 0
    ref_rejected = mem_repo.get_reference("ref_01")
    assert ref_rejected is not None
    assert ref_rejected.approval_status == "rejected"


def test_foreign_key_enforcement_and_cascades(mem_repo: SQLiteColorBibleRepository) -> None:
    # Inserting shot with non-existent film_id must fail
    invalid_shot = ShotRecord(
        shot_id="shot_invalid",
        film_id="non_existent_film",
        shot_index=1,
        start_frame=0,
        end_frame=100,
        start_time=0.0,
        end_time=4.0,
    )
    with pytest.raises(sqlite3.IntegrityError):
        mem_repo.upsert_shot(invalid_shot)

    # Create film, character, variant, shot, assignment, reference
    film = Film(film_id="film_cascade", title="Cascade Test Film")
    mem_repo.upsert_film(film)

    char = Character(character_id="char_casc", film_id="film_cascade", name="Bob")
    mem_repo.upsert_character(char)

    var = AppearanceVariant(variant_id="var_casc", character_id="char_casc", costume_id="suit")
    mem_repo.upsert_variant(var)

    shot = ShotRecord(
        shot_id="shot_casc",
        film_id="film_cascade",
        shot_index=1,
        start_frame=0,
        end_frame=50,
        start_time=0.0,
        end_time=2.0,
    )
    mem_repo.upsert_shot(shot)

    asg = ShotEntityAssignment(
        assignment_id="asg_casc",
        film_id="film_cascade",
        shot_id="shot_casc",
        entity_type="character",
        entity_id="char_casc",
        variant_id="var_casc",
        valid_shot_range_start=1,
        valid_shot_range_end=1,
    )
    mem_repo.upsert_assignment(asg)

    ref = ColorReferenceRecord(
        ref_id="ref_casc",
        film_id="film_cascade",
        entity_type="character",
        entity_id="char_casc",
        variant_id="var_casc",
        source_shot_id="shot_casc",
        source_keyframe_path="test.png",
        lab_stats_json="{}",
        rgb_palette_json="[]",
        provenance="manual",
    )
    mem_repo.upsert_reference(ref)

    # Deleting shot should SET NULL on shot_id in asg and source_shot_id in ref
    mem_repo.delete_shot("shot_casc")
    asg_after_shot = mem_repo.get_assignment("asg_casc")
    assert asg_after_shot is not None
    assert asg_after_shot.shot_id is None

    ref_after_shot = mem_repo.get_reference("ref_casc")
    assert ref_after_shot is not None
    assert ref_after_shot.source_shot_id is None

    # Deleting character should CASCADE delete appearance variant
    mem_repo.delete_character("char_casc")
    assert mem_repo.get_variant("var_casc") is None

    # Deleting film should CASCADE delete everything remaining
    mem_repo.delete_film("film_cascade")
    assert mem_repo.get_assignment("asg_casc") is None
    assert mem_repo.get_reference("ref_casc") is None


def test_check_constraints(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_check", title="Check Constraints")
    mem_repo.upsert_film(film)

    # Invalid object kind
    invalid_obj = ColorBibleObject(
        object_id="obj_bad",
        film_id="film_check",
        kind="invalid_kind",
        name="Bad Object",
    )
    with pytest.raises(sqlite3.IntegrityError):
        mem_repo.upsert_object(invalid_obj)

    # Invalid assignment entity_type
    invalid_asg = ShotEntityAssignment(
        assignment_id="asg_bad",
        film_id="film_check",
        entity_type="invalid_type",
        entity_id="some_id",
        valid_shot_range_start=1,
        valid_shot_range_end=2,
    )
    with pytest.raises(sqlite3.IntegrityError):
        mem_repo.upsert_assignment(invalid_asg)

    # Invalid reference provenance
    invalid_ref = ColorReferenceRecord(
        ref_id="ref_bad",
        film_id="film_check",
        entity_type="character",
        entity_id="some_id",
        source_keyframe_path="bad.png",
        lab_stats_json="{}",
        rgb_palette_json="[]",
        provenance="unsupported_source",
    )
    with pytest.raises(sqlite3.IntegrityError):
        mem_repo.upsert_reference(invalid_ref)

    # Invalid assignment approval_status
    invalid_asg_status = ShotEntityAssignment(
        assignment_id="asg_bad_status",
        film_id="film_check",
        entity_type="character",
        entity_id="some_id",
        valid_shot_range_start=1,
        valid_shot_range_end=2,
        approval_status="unapproved_unknown",
    )
    with pytest.raises(sqlite3.IntegrityError):
        mem_repo.upsert_assignment(invalid_asg_status)

    # Invalid reference approval_status
    invalid_ref_status = ColorReferenceRecord(
        ref_id="ref_bad_status",
        film_id="film_check",
        entity_type="character",
        entity_id="some_id",
        source_keyframe_path="bad.png",
        lab_stats_json="{}",
        rgb_palette_json="[]",
        provenance="manual",
        approval_status="unknown_status",
    )
    with pytest.raises(sqlite3.IntegrityError):
        mem_repo.upsert_reference(invalid_ref_status)

    # Invalid reference entity_type
    invalid_ref_type = ColorReferenceRecord(
        ref_id="ref_bad_type",
        film_id="film_check",
        entity_type="unknown_entity_type",
        entity_id="some_id",
        source_keyframe_path="bad.png",
        lab_stats_json="{}",
        rgb_palette_json="[]",
        provenance="manual",
    )
    with pytest.raises(sqlite3.IntegrityError):
        mem_repo.upsert_reference(invalid_ref_type)


def test_database_persistence_and_reopen(tmp_path: Path) -> None:
    db_file = tmp_path / "persistent_color_bible.db"

    # Step 1: Open repository and populate entities
    repo1 = SQLiteColorBibleRepository(db_file)
    repo1.init_db()

    film = Film(film_id="film_persist", title="Persistent Noir Feature", source_video="vids/feature.mp4")
    repo1.upsert_film(film)

    char = Character(character_id="char_persist", film_id="film_persist", name="Marcus", notes="Investigator")
    repo1.upsert_character(char)

    var = AppearanceVariant(variant_id="var_persist", character_id="char_persist", costume_id="fedora_coat")
    repo1.upsert_variant(var)

    shot = ShotRecord(
        shot_id="shot_persist_1",
        film_id="film_persist",
        shot_index=1,
        start_frame=0,
        end_frame=48,
        start_time=0.0,
        end_time=2.0,
    )
    repo1.upsert_shot(shot)

    asg = ShotEntityAssignment(
        assignment_id="asg_persist_1",
        film_id="film_persist",
        shot_id="shot_persist_1",
        entity_type="character",
        entity_id="char_persist",
        variant_id="var_persist",
        valid_shot_range_start=1,
        valid_shot_range_end=10,
    )
    repo1.upsert_assignment(asg)

    ref = ColorReferenceRecord(
        ref_id="ref_persist_1",
        film_id="film_persist",
        entity_type="character",
        entity_id="char_persist",
        variant_id="var_persist",
        source_keyframe_path="ref_1.png",
        lab_stats_json='{"L": [50.0]}',
        rgb_palette_json='[{"hex": "#334455"}]',
        provenance="manual",
        approval_status="approved",
    )
    repo1.upsert_reference(ref)
    repo1.close()

    # Step 2: Open fresh repository on the existing database file
    repo2 = SQLiteColorBibleRepository(db_file)
    applied = repo2.init_db()
    assert applied == []  # Migration must be idempotent

    retrieved_film = repo2.get_film("film_persist")
    assert retrieved_film is not None
    assert retrieved_film.title == "Persistent Noir Feature"

    retrieved_char = repo2.get_character("char_persist")
    assert retrieved_char is not None
    assert retrieved_char.name == "Marcus"

    retrieved_var = repo2.get_variant("var_persist")
    assert retrieved_var is not None
    assert retrieved_var.costume_id == "fedora_coat"

    retrieved_shot = repo2.get_shot("shot_persist_1")
    assert retrieved_shot is not None
    assert retrieved_shot.shot_index == 1

    retrieved_asg = repo2.get_assignment("asg_persist_1")
    assert retrieved_asg is not None
    assert retrieved_asg.variant_id == "var_persist"

    retrieved_ref = repo2.get_reference("ref_persist_1")
    assert retrieved_ref is not None
    assert retrieved_ref.approval_status == "approved"

    repo2.close()


def test_boundary_shot_ranges(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_boundary", title="Boundary Testing")
    mem_repo.upsert_film(film)

    char = Character(character_id="char_b", film_id="film_boundary", name="Hero")
    mem_repo.upsert_character(char)

    var1 = AppearanceVariant(variant_id="var_b1", character_id="char_b", costume_id="Costume 1")
    var2 = AppearanceVariant(variant_id="var_b2", character_id="char_b", costume_id="Costume 2")
    mem_repo.upsert_variant(var1)
    mem_repo.upsert_variant(var2)

    # Disjoint ranges: [12, 18] and [410, 435] for var1; [500, 500] (single shot) for var2
    asg1 = ShotEntityAssignment(
        assignment_id="asg_b1",
        film_id="film_boundary",
        entity_type="character",
        entity_id="char_b",
        variant_id="var_b1",
        valid_shot_range_start=12,
        valid_shot_range_end=18,
    )
    asg2 = ShotEntityAssignment(
        assignment_id="asg_b2",
        film_id="film_boundary",
        entity_type="character",
        entity_id="char_b",
        variant_id="var_b1",
        valid_shot_range_start=410,
        valid_shot_range_end=435,
    )
    asg3 = ShotEntityAssignment(
        assignment_id="asg_b3",
        film_id="film_boundary",
        entity_type="character",
        entity_id="char_b",
        variant_id="var_b2",
        valid_shot_range_start=500,
        valid_shot_range_end=500,
    )
    mem_repo.upsert_assignment(asg1)
    mem_repo.upsert_assignment(asg2)
    mem_repo.upsert_assignment(asg3)

    # Test boundary of [12, 18]
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 11)) == 0
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 12)) == 1
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 15)) == 1
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 18)) == 1
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 19)) == 0

    # Test boundary of [410, 435]
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 409)) == 0
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 410)) == 1
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 420)) == 1
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 435)) == 1
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 436)) == 0

    # Test single-shot boundary [500, 500]
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 499)) == 0
    match_500 = mem_repo.find_assignments_for_shot_index("film_boundary", 500)
    assert len(match_500) == 1
    assert match_500[0].variant_id == "var_b2"
    assert len(mem_repo.find_assignments_for_shot_index("film_boundary", 501)) == 0

    # Overlapping range queries
    assert len(mem_repo.find_assignments_for_range("film_boundary", 1, 11)) == 0
    assert len(mem_repo.find_assignments_for_range("film_boundary", 10, 12)) == 1
    assert len(mem_repo.find_assignments_for_range("film_boundary", 18, 20)) == 1
    assert len(mem_repo.find_assignments_for_range("film_boundary", 19, 400)) == 0
    assert len(mem_repo.find_assignments_for_range("film_boundary", 10, 500)) == 3


def test_variant_deletion_sets_null_on_assignments_and_references(mem_repo: SQLiteColorBibleRepository) -> None:
    film = Film(film_id="film_fk_null", title="FK Null Test")
    mem_repo.upsert_film(film)

    char = Character(character_id="char_fk", film_id="film_fk_null", name="Detective")
    mem_repo.upsert_character(char)

    var = AppearanceVariant(variant_id="var_fk", character_id="char_fk", costume_id="Overcoat")
    mem_repo.upsert_variant(var)

    asg = ShotEntityAssignment(
        assignment_id="asg_fk",
        film_id="film_fk_null",
        entity_type="character",
        entity_id="char_fk",
        variant_id="var_fk",
        valid_shot_range_start=1,
        valid_shot_range_end=5,
    )
    mem_repo.upsert_assignment(asg)

    ref = ColorReferenceRecord(
        ref_id="ref_fk",
        film_id="film_fk_null",
        entity_type="character",
        entity_id="char_fk",
        variant_id="var_fk",
        source_keyframe_path="ref_fk.png",
        lab_stats_json="{}",
        rgb_palette_json="[]",
        provenance="manual",
    )
    mem_repo.upsert_reference(ref)

    # Deleting the appearance variant must SET NULL on variant_id in assignment and reference
    assert mem_repo.delete_variant("var_fk") is True
    assert mem_repo.get_variant("var_fk") is None

    asg_after = mem_repo.get_assignment("asg_fk")
    assert asg_after is not None
    assert asg_after.variant_id is None

    ref_after = mem_repo.get_reference("ref_fk")
    assert ref_after is not None
    assert ref_after.variant_id is None


def test_empty_database_and_nonexistent_entity_queries(mem_repo: SQLiteColorBibleRepository) -> None:
    # Query non-existent entities
    assert mem_repo.get_film("ghost_film") is None
    assert mem_repo.get_shot("ghost_shot") is None
    assert mem_repo.get_character("ghost_char") is None
    assert mem_repo.get_variant("ghost_var") is None
    assert mem_repo.get_object("ghost_obj") is None
    assert mem_repo.get_location("ghost_loc") is None
    assert mem_repo.get_assignment("ghost_asg") is None
    assert mem_repo.get_reference("ghost_ref") is None

    # Deleting non-existent entities returns False
    assert mem_repo.delete_film("ghost_film") is False
    assert mem_repo.delete_shot("ghost_shot") is False
    assert mem_repo.delete_character("ghost_char") is False
    assert mem_repo.delete_variant("ghost_var") is False
    assert mem_repo.delete_object("ghost_obj") is False
    assert mem_repo.delete_location("ghost_loc") is False
    assert mem_repo.delete_assignment("ghost_asg") is False
    assert mem_repo.delete_reference("ghost_ref") is False

    # Listing for non-existent film returns empty lists
    assert mem_repo.list_shots("ghost_film") == []
    assert mem_repo.list_characters("ghost_film") == []
    assert mem_repo.list_variants_for_film("ghost_film") == []
    assert mem_repo.list_objects("ghost_film") == []
    assert mem_repo.list_locations("ghost_film") == []
    assert mem_repo.list_assignments("ghost_film") == []
    assert mem_repo.list_references("ghost_film") == []
    assert mem_repo.list_approved_references("ghost_film") == []

    # Out of bounds and inverted range queries
    assert mem_repo.find_assignments_for_shot_index("ghost_film", -1) == []
    assert mem_repo.find_assignments_for_shot_index("ghost_film", 999999) == []
    assert mem_repo.find_assignments_for_range("ghost_film", 100, 50) == []


def test_film_isolation_multiple_films(mem_repo: SQLiteColorBibleRepository) -> None:
    film_a = Film(film_id="film_a", title="Film Alpha")
    film_b = Film(film_id="film_b", title="Film Beta")
    mem_repo.upsert_film(film_a)
    mem_repo.upsert_film(film_b)

    char_a = Character(character_id="char_a", film_id="film_a", name="Alice")
    char_b = Character(character_id="char_b", film_id="film_b", name="Bob")
    mem_repo.upsert_character(char_a)
    mem_repo.upsert_character(char_b)

    var_a = AppearanceVariant(variant_id="var_a", character_id="char_a", costume_id="Costume Alpha")
    var_b = AppearanceVariant(variant_id="var_b", character_id="char_b", costume_id="Costume Beta")
    mem_repo.upsert_variant(var_a)
    mem_repo.upsert_variant(var_b)

    # Verify listing variants for film A only contains var_a
    vars_a = mem_repo.list_variants_for_film("film_a")
    assert len(vars_a) == 1
    assert vars_a[0].variant_id == "var_a"

    vars_b = mem_repo.list_variants_for_film("film_b")
    assert len(vars_b) == 1
    assert vars_b[0].variant_id == "var_b"

    # Delete film A, ensure film B is untouched
    assert mem_repo.delete_film("film_a") is True
    assert mem_repo.get_film("film_a") is None
    assert mem_repo.get_character("char_a") is None
    assert mem_repo.get_variant("var_a") is None

    assert mem_repo.get_film("film_b") is not None
    assert mem_repo.get_character("char_b") is not None
    assert mem_repo.get_variant("var_b") is not None


def test_polymorphic_assignment_cleanup_on_entity_deletion(mem_repo: SQLiteColorBibleRepository) -> None:
    """Verify delete_character, delete_object, and delete_location invoke polymorphic assignment cleanup."""
    film = Film(film_id="film_poly", title="Polymorphic Test")
    mem_repo.upsert_film(film)

    char = Character(character_id="char_p1", film_id="film_poly", name="Test Char")
    mem_repo.upsert_character(char)
    asg_char = ShotEntityAssignment(
        assignment_id="asg_p_char",
        film_id="film_poly",
        entity_type="character",
        entity_id="char_p1",
        valid_shot_range_start=1,
        valid_shot_range_end=5,
    )
    mem_repo.upsert_assignment(asg_char)

    obj = ColorBibleObject(object_id="obj_p1", film_id="film_poly", kind="prop", name="Test Prop")
    mem_repo.upsert_object(obj)
    asg_obj = ShotEntityAssignment(
        assignment_id="asg_p_obj",
        film_id="film_poly",
        entity_type="object",
        entity_id="obj_p1",
        valid_shot_range_start=2,
        valid_shot_range_end=6,
    )
    mem_repo.upsert_assignment(asg_obj)

    loc = Location(location_id="loc_p1", film_id="film_poly", set_name="Test Set")
    mem_repo.upsert_location(loc)
    asg_loc = ShotEntityAssignment(
        assignment_id="asg_p_loc",
        film_id="film_poly",
        entity_type="location",
        entity_id="loc_p1",
        valid_shot_range_start=3,
        valid_shot_range_end=7,
    )
    mem_repo.upsert_assignment(asg_loc)

    assert len(mem_repo.list_assignments("film_poly")) == 3

    # Delete character -> assignment for char_p1 must be gone
    assert mem_repo.delete_character("char_p1") is True
    assert mem_repo.get_assignment("asg_p_char") is None
    assert len(mem_repo.list_assignments("film_poly")) == 2

    # Delete object -> assignment for obj_p1 must be gone
    assert mem_repo.delete_object("obj_p1") is True
    assert mem_repo.get_assignment("asg_p_obj") is None
    assert len(mem_repo.list_assignments("film_poly")) == 1

    # Delete location -> assignment for loc_p1 must be gone
    assert mem_repo.delete_location("loc_p1") is True
    assert mem_repo.get_assignment("asg_p_loc") is None
    assert len(mem_repo.list_assignments("film_poly")) == 0



