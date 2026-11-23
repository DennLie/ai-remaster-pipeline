"""Unit tests for ai_remaster_gui.semantic_matching on CPU."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai_remaster_gui.semantic_matching import (
    ColorBibleMatcher,
    GuidedReferenceBackend,
    PromptEnricher,
    Qwen2511Backend,
    ReferenceBackendRegistry,
    ReferenceRequest,
    ReferenceResult,
    SemanticMatchResult,
    TwoStageReferenceRouter,
    build_reference_signature,
    compute_scoped_cache_key,
    enrich_prompt,
    get_backend,
    list_backends,
    record_reference_compliance,
    register_backend,
    registry,
    resolve_references,
    select_backend,
)


# ---------------------------------------------------------------------------
# 1. GuidedReferenceBackend Protocol & ReferenceBackendRegistry
# ---------------------------------------------------------------------------


def test_qwen2511_backend_registered():
    """Verify built-in Qwen2511Backend is registered and implements protocol."""
    backend = get_backend("qwen2511")
    assert isinstance(backend, GuidedReferenceBackend)
    assert backend.name == "qwen2511"
    assert "qwen2511" in list_backends()


def test_qwen2511_colorize_shot_command(tmp_path):
    """Verify Qwen2511Backend builds expected command line arguments."""
    backend = get_backend("qwen2511")
    src = tmp_path / "source.png"
    out = tmp_path / "output.png"
    ref1 = tmp_path / "ref1.png"
    ref2 = tmp_path / "ref2.png"

    request = ReferenceRequest(
        shot_index=4,
        source_image_path=src,
        output_path=out,
        reference_image_paths=[ref1, ref2],
        prompt="Restore colors",
        prompt_suffix="cinematic grade",
        workflow="custom_workflow.json",
        comfy_url="http://127.0.0.1:8188",
        comfy_dir="/comfyui",
        model_backend="gguf",
        gguf_model="qwen-edit-v1",
        seed=42,
    )

    result = backend.colorize_shot(request)
    assert isinstance(result, ReferenceResult)
    assert result.success is True
    assert result.backend_name == "qwen2511"
    assert str(src) in result.command
    assert str(out) in result.command
    assert "--seed" in result.command
    assert "42" in result.command
    assert "--workflow" in result.command
    assert "--reference-image" in result.command


def test_custom_backend_registration():
    """Verify custom backend registration via decorator and direct call."""
    custom_registry = ReferenceBackendRegistry()

    @custom_registry.register("mock_guided")
    class MockBackend:
        name = "mock_guided"

        def colorize_shot(self, request: ReferenceRequest) -> ReferenceResult:
            return ReferenceResult(output_path=request.output_path, backend_name=self.name)

    assert "mock_guided" in custom_registry.list_backends()
    instance = custom_registry.get("mock_guided")
    assert isinstance(instance, MockBackend)

    with pytest.raises(KeyError):
        custom_registry.get("nonexistent")


# ---------------------------------------------------------------------------
# 2. ColorBibleMatcher: 3-Tier Retrieval & Review Gate
# ---------------------------------------------------------------------------


def test_color_bible_matcher_backward_compatibility(tmp_path):
    """When film_id or service is None, falls back cleanly to neighbor frames."""
    f1 = tmp_path / "frame1.png"
    f2 = tmp_path / "frame2.png"
    f1.write_bytes(b"img1")
    f2.write_bytes(b"img2")

    rows = [
        {"_shot_index": "0", "color_reference": str(f1)},
        {"_shot_index": "1", "color_reference": str(f2)},
    ]

    # Film ID missing -> fallback
    result = ColorBibleMatcher.resolve_references(rows=rows, index=1, count=2, film_id=None, service=None)
    assert isinstance(result, SemanticMatchResult)
    assert result.needs_review is False
    assert result.review_reasons == []
    assert result.color_bible_version == ""
    assert len(result.reference_paths) == 1
    assert result.reference_paths[0] == f1.resolve()

    # Service missing -> fallback
    result2 = resolve_references(rows=rows, index=1, count=2, film_id="film_123", service=None)
    assert result2.needs_review is False
    assert len(result2.reference_paths) == 1


def test_color_bible_matcher_priority_a_and_grouping(tmp_path):
    """Priority A prioritizes characters with variants over characters, objects, locations."""
    # Create real image files for references
    char_var_img = tmp_path / "char_var.png"
    char_base_img = tmp_path / "char_base.png"
    obj_img = tmp_path / "obj.png"
    loc_img = tmp_path / "loc.png"
    for img in (char_var_img, char_base_img, obj_img, loc_img):
        img.write_bytes(b"dummy")

    mock_service = MagicMock()
    mock_service.get_snapshot_hash.return_value = "sha256:abc12345"

    mock_service.get_shot_palette_assignments.return_value = [
        {
            "entity_type": "location",
            "entity_id": "loc_1",
            "variant_id": None,
            "assignment": {"confidence": 0.99, "approval_status": "approved"},
            "color_references": [
                {"ref_id": "ref_loc", "approval_status": "approved", "source_keyframe_path": str(loc_img), "version": 1}
            ],
        },
        {
            "entity_type": "object",
            "entity_id": "obj_1",
            "variant_id": None,
            "assignment": {"confidence": 0.95, "approval_status": "approved"},
            "color_references": [
                {"ref_id": "ref_obj", "approval_status": "approved", "source_keyframe_path": str(obj_img), "version": 1}
            ],
        },
        {
            "entity_type": "character",
            "entity_id": "char_1",
            "variant_id": None,
            "assignment": {"confidence": 0.90, "approval_status": "approved"},
            "color_references": [
                {"ref_id": "ref_char_base", "approval_status": "approved", "source_keyframe_path": str(char_base_img), "version": 1}
            ],
        },
        {
            "entity_type": "character",
            "entity_id": "char_1",
            "variant_id": "var_red_coat",
            "assignment": {"confidence": 0.85, "approval_status": "approved"},
            "color_references": [
                {"ref_id": "ref_char_var", "approval_status": "approved", "source_keyframe_path": str(char_var_img), "version": 2}
            ],
        },
    ]

    # Request count=2: should pick char_var first, then char_base second (characters before objects/locations)
    result = ColorBibleMatcher.resolve_references(
        rows=[],
        index=0,
        count=2,
        film_id="film_test",
        service=mock_service,
    )

    assert result.color_bible_version == "sha256:abc12345"
    assert result.needs_review is False
    assert len(result.reference_paths) == 2
    assert result.reference_paths[0] == char_var_img.resolve()
    assert result.reference_paths[1] == char_base_img.resolve()
    assert result.retrieved_ref_ids == ["ref_char_var", "ref_char_base"]
    assert "ref_char_var" in result.retrieved_ref_hashes


def test_color_bible_matcher_missing_files_and_priority_b_fill(tmp_path):
    """Missing reference files are skipped cleanly; remaining slots fill from Priority B neighbor rows."""
    missing_img = tmp_path / "does_not_exist.png"
    valid_master_img = tmp_path / "valid_master.png"
    valid_master_img.write_bytes(b"master")

    neighbor_img = tmp_path / "neighbor.png"
    neighbor_img.write_bytes(b"neighbor")

    mock_service = MagicMock()
    mock_service.get_snapshot_hash.return_value = "sha256:snapshot1"
    mock_service.get_shot_palette_assignments.return_value = [
        {
            "entity_type": "character",
            "entity_id": "char_missing",
            "assignment": {"confidence": 0.95, "approval_status": "approved"},
            "color_references": [
                {"ref_id": "ref_missing", "approval_status": "approved", "source_keyframe_path": str(missing_img)}
            ],
        },
        {
            "entity_type": "character",
            "entity_id": "char_valid",
            "assignment": {"confidence": 0.90, "approval_status": "approved"},
            "color_references": [
                {"ref_id": "ref_valid", "approval_status": "approved", "source_keyframe_path": str(valid_master_img)}
            ],
        },
    ]

    rows = [
        {"_shot_index": "0", "color_reference": str(neighbor_img)},
        {"_shot_index": "1", "color_reference": str(tmp_path / "current.png")},
    ]

    result = ColorBibleMatcher.resolve_references(
        rows=rows,
        index=1,
        count=2,
        film_id="film_test",
        service=mock_service,
    )

    assert len(result.reference_paths) == 2
    # First is valid_master (missing was skipped)
    assert result.reference_paths[0] == valid_master_img.resolve()
    # Second is Priority B neighbor fill
    assert result.reference_paths[1] == neighbor_img.resolve()
    assert result.retrieved_ref_ids == ["ref_valid"]


def test_color_bible_matcher_review_gate_triggers():
    """Priority C review gate triggers on pending_review or confidence below threshold."""
    mock_service = MagicMock()
    mock_service.get_snapshot_hash.return_value = "sha256:v1"

    # Case 1: Low confidence
    mock_service.get_shot_palette_assignments.return_value = [
        {
            "entity_type": "character",
            "entity_id": "mickey",
            "variant_id": "coat_red",
            "entity": {"name": "Mickey"},
            "assignment": {"confidence": 0.72, "approval_status": "approved"},
            "color_references": [],
        }
    ]

    result = ColorBibleMatcher.resolve_references(
        rows=[],
        index=0,
        count=2,
        film_id="film_test",
        service=mock_service,
        confidence_threshold=0.80,
    )
    assert result.needs_review is True
    assert any("0.72 < 0.80" in r for r in result.review_reasons)

    # Case 2: Pending review status
    mock_service.get_shot_palette_assignments.return_value = [
        {
            "entity_type": "character",
            "entity_id": "donald",
            "variant_id": "sailor_blue",
            "entity": {"name": "Donald"},
            "assignment": {"confidence": 0.95, "approval_status": "pending_review"},
            "color_references": [],
        }
    ]

    result2 = ColorBibleMatcher.resolve_references(
        rows=[],
        index=0,
        count=2,
        film_id="film_test",
        service=mock_service,
    )
    assert result2.needs_review is True
    assert any("pending_review" in r for r in result2.review_reasons)


# ---------------------------------------------------------------------------
# 3. PromptEnricher
# ---------------------------------------------------------------------------


def test_prompt_enricher_formatting():
    """PromptEnricher formats characters, costume IDs, RGB swatches, LAB stats, and lighting."""
    entities = [
        {
            "entity_type": "character",
            "entity": {"name": "Mickey"},
            "variant": {"costume_id": "red_shorts"},
            "color_references": [
                {
                    "rgb_palette_json": json.dumps([{"hex": "#D32F2F", "rgb": [211, 47, 47]}]),
                    "lab_stats_json": json.dumps({
                        "L": [20.0, 30.0, 45.2, 60.0, 70.0],
                        "a": [30.0, 40.0, 58.1, 65.0, 70.0],
                        "b": [10.0, 20.0, 39.4, 45.0, 50.0],
                    }),
                }
            ],
        }
    ]

    enriched = enrich_prompt(
        base_prompt="A black and white cartoon scene.",
        entities=entities,
        lighting_context="warm afternoon daylight",
    )

    expected_clause = "[Color Bible Constraints: Character Mickey (costume: red_shorts, target RGB: #D32F2F / [211, 47, 47], LAB L*: 45.2, a*: 58.1, b*: 39.4); Scene Lighting: warm afternoon daylight]."
    assert expected_clause in enriched
    assert enriched.startswith("A black and white cartoon scene. ")


def test_prompt_enricher_empty_entities():
    """PromptEnricher returns base prompt intact when entities list and lighting are empty."""
    base = "Clean monochrome shot."
    assert enrich_prompt(base, []) == base
    assert enrich_prompt(base, [], "") == base


# ---------------------------------------------------------------------------
# 4. TwoStageReferenceRouter
# ---------------------------------------------------------------------------


def test_two_stage_reference_router_overrides_and_modes():
    """Verify backend selection honors row overrides, runner modes, and auto escalation."""
    # Per-row override wins unconditionally
    row_override = {"reference_backend": "my_custom_model"}
    assert select_backend(row_override, has_color_bible_entities=False, reference_runner="ddcolor_only") == "my_custom_model"

    # Strict runner modes
    row_normal = {}
    assert select_backend(row_normal, has_color_bible_entities=True, reference_runner="ddcolor_only") == "ddcolor"
    assert select_backend(row_normal, has_color_bible_entities=False, reference_runner="guided_only") == "qwen2511"

    # Auto mode: no entities -> ddcolor
    assert select_backend(row_normal, has_color_bible_entities=False, reference_runner="auto") == "ddcolor"

    # Auto mode: with Color Bible entities -> guided
    assert select_backend(row_normal, has_color_bible_entities=True, reference_runner="auto") == "qwen2511"

    # Auto mode: escalate flags in row -> guided
    assert select_backend({"escalate_guided": "true"}, has_color_bible_entities=False, reference_runner="auto") == "qwen2511"
    assert select_backend({"escalate_qwen": "1"}, has_color_bible_entities=False, reference_runner="auto") == "qwen2511"


# ---------------------------------------------------------------------------
# 5. Scoped Cache Key & Signature Builders
# ---------------------------------------------------------------------------


def test_scoped_cache_key_invariance():
    """Updates to unrelated entity references or global color bible version do not invalidate cache."""
    shot_hashes = {"ref_mickey_costume": "v2"}
    key1 = compute_scoped_cache_key(
        shot_index=10,
        source_fingerprint="fp_shot10",
        retrieved_ref_hashes=shot_hashes,
        prompt="Restore Mickey",
        backend_params={"seed": 42},
    )

    # Recomputing with identical shot-specific params gives same hash
    key2 = compute_scoped_cache_key(
        shot_index=10,
        source_fingerprint="fp_shot10",
        retrieved_ref_hashes=shot_hashes,
        prompt="Restore Mickey",
        backend_params={"seed": 42},
    )
    assert key1 == key2

    # Changing an unrelated entity in the Color Bible does NOT change this shot's key
    sig1 = build_reference_signature(
        shot_index=10,
        source_fingerprint="fp_shot10",
        retrieved_ref_hashes=shot_hashes,
        prompt="Restore Mickey",
        backend_params={"seed": 42},
        color_bible_version="sha256:global_v1",
    )

    sig2 = build_reference_signature(
        shot_index=10,
        source_fingerprint="fp_shot10",
        retrieved_ref_hashes=shot_hashes,
        prompt="Restore Mickey",
        backend_params={"seed": 42},
        color_bible_version="sha256:global_v2_unrelated_entity_changed",
    )

    assert sig1["scoped_cache_key"] == sig2["scoped_cache_key"]
    # But provenance color_bible_version reflects the actual version
    assert sig1["color_bible_version"] != sig2["color_bible_version"]


def test_scoped_cache_key_normalization_and_serialization():
    """Verify dictionary source fingerprints and non-primitive objects are handled safely."""
    dict_fp_sha = {"sha256": "abc12345", "size": 1024, "mtime_ns": 9999}
    dict_fp_size = {"size": 2048, "mtime_ns": 8888}
    dict_fp_mtime = {"mtime_ns": 7777}

    wf_path = Path("/tmp/workflow.json")
    key_sha = compute_scoped_cache_key(
        shot_index=1,
        source_fingerprint=dict_fp_sha,
        retrieved_ref_hashes={},
        prompt="Test",
        backend_params={"workflow_path": wf_path},
    )
    assert isinstance(key_sha, str)
    assert len(key_sha) == 64

    # Direct string matching sha256 produces identical key
    key_direct = compute_scoped_cache_key(
        shot_index=1,
        source_fingerprint="abc12345",
        retrieved_ref_hashes={},
        prompt="Test",
        backend_params={"workflow_path": wf_path},
    )
    assert key_sha == key_direct

    # Non-primitive objects in backend_params do not raise TypeError
    key_size = compute_scoped_cache_key(
        shot_index=1,
        source_fingerprint=dict_fp_size,
        retrieved_ref_hashes={},
        prompt="Test",
        backend_params={"path": Path("/some/path")},
    )
    assert isinstance(key_size, str)

    key_mtime = compute_scoped_cache_key(
        shot_index=1,
        source_fingerprint=dict_fp_mtime,
        retrieved_ref_hashes={},
        prompt="Test",
        backend_params={},
    )
    assert isinstance(key_mtime, str)


def test_color_bible_matcher_skips_rejected_assignments():
    """Verify rejected assignments are omitted from match_result.entities and do not trigger review."""
    mock_service = MagicMock()
    mock_service.get_snapshot_hash.return_value = "v1"
    mock_service.get_shot_palette_assignments.return_value = [
        {
            "entity_type": "character",
            "entity_id": "rejected_char",
            "variant_id": "rejected_variant",
            "entity": {"name": "RejectedGuy"},
            "assignment": {"confidence": 0.20, "approval_status": "rejected"},
            "color_references": [
                {"ref_id": "ref_rej", "approval_status": "approved", "source_keyframe_path": "rej.png"}
            ],
        },
        {
            "entity_type": "character",
            "entity_id": "approved_char",
            "variant_id": "approved_variant",
            "entity": {"name": "ApprovedGuy"},
            "assignment": {"confidence": 0.95, "approval_status": "approved"},
            "color_references": [],
        },
    ]

    res = ColorBibleMatcher.resolve_references(
        rows=[],
        index=0,
        film_id="film_test",
        service=mock_service,
        confidence_threshold=0.80,
    )

    # Rejected assignment must be completely filtered out
    entity_ids = [e.get("entity_id") for e in res.entities]
    assert "rejected_char" not in entity_ids
    assert "approved_char" in entity_ids
    # And its low confidence (0.20) must NOT trigger a review gate
    assert res.needs_review is False


# ---------------------------------------------------------------------------
# 6. Commercial Model Guard & Compliance Recording
# ---------------------------------------------------------------------------


def test_record_reference_compliance(tmp_path):
    """Verify commercial policy evaluation and compliance report appending."""
    out_img = tmp_path / "cut_0005.png"
    out_img.write_bytes(b"fake image")

    report_path = record_reference_compliance(
        output_path=out_img,
        model_id="qwen-image-edit-2511",
        component_type="checkpoint",
        color_bible_version="sha256:cb_snapshot_123",
        retrieved_ref_ids=["ref_mickey_1", "ref_mickey_2"],
        settings={"global": {"qwen_2511_audited": "true"}},
        seed=100,
        execution_time_ms=1845.2,
    )

    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))

    assert data["schema_version"] == "1.0.0"
    assert len(data["runs"]) == 1
    run = data["runs"][0]

    assert run["stage"] == "color_reference"
    assert run["color_bible_version"] == "sha256:cb_snapshot_123"
    assert "metadata" in run
    assert run["metadata"]["retrieved_ref_ids"] == ["ref_mickey_1", "ref_mickey_2"]
    assert run["metadata"]["execution_time_ms"] == 1845.2
    assert run["seeds"] == {"reference_seed": 100}


def test_record_reference_compliance_fail_closed(tmp_path):
    """Verify fail_closed=True raises CommercialPolicyError on blocked models, and fail_closed=False records it."""
    from ai_remaster_gui.commercial_policy import CommercialPolicyError

    out_img = tmp_path / "blocked_shot.png"
    out_img.write_bytes(b"data")

    # Fail closed by default
    with pytest.raises(CommercialPolicyError):
        record_reference_compliance(
            output_path=out_img,
            model_id="flux",
            component_type="checkpoint",
            color_bible_version="v1",
            retrieved_ref_ids=[],
            fail_closed=True,
        )

    # With fail_closed=False, records the blocked run without throwing
    report_path = record_reference_compliance(
        output_path=out_img,
        model_id="flux",
        component_type="checkpoint",
        color_bible_version="v1",
        retrieved_ref_ids=[],
        fail_closed=False,
    )
    assert report_path.is_file()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["runs"][0]["models"][0]["status"] in ("hard_blocked", "unknown", "un-audited")
