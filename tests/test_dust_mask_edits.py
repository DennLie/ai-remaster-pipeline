"""Unit and integration tests for lossless dust mask edits, precedence, E2 opt-in, and provenance."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock
import zipfile

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
for p in (str(ROOT_DIR), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ai_remaster_gui import dust_mask_edits as dme
from ai_remaster_gui.dust_e2_adapter import (
    DETECTOR_NAME,
    E2_DETECTOR_INFO,
    E2_PRESET_INFO,
    ExperimentalOptInRequiredError,
    is_research_confirmed,
    run_experimental_e2_sequence,
)
from ai_remaster_gui.dust_mask_edits import (
    BIT_DUST_ADD,
    BIT_DUST_REJECT,
    BIT_PROTECT_INTACT,
    BIT_REVIEWED,
    BIT_SCRATCH_ADD,
    BIT_SCRATCH_REJECT,
    BIT_UNCERTAIN_REJECT,
    EDIT_SCHEMA_VERSION,
    GROUND_TRUTH_STATUS,
    KNOWN_EDIT_BITS_MASK,
    VALID_STROKE_TOOLS,
    DustMaskEditStore,
    EditRevisionConflictError,
    EditStoreIntegrityError,
    EditTimelineMismatchError,
    TimelineBinding,
    apply_stroke_transition,
    build_edit_provenance_summary,
    build_timeline_binding,
    compute_edit_signature,
    compute_timeline_id,
    decode_bitmask_b64,
    decode_bitmask_png,
    encode_bitmask_b64,
    encode_bitmask_png,
    export_provenance_bundle,
    load_verified_store,
    resolve_effective_channels,
    store_compatibility_status,
)
from scripts.film_defect_detectors import ChannelOrigin, RepairInclusionPolicy
import benchmark_defect_masks as bdm


def _make_source_file(root: Path, rel_name: str = "input/clip.mkv", payload: bytes = b"synthetic-video-payload") -> Path:
    path = root / rel_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _make_binding(
    root: Path,
    source_path: Path,
    *,
    analysis_source: str = "stage",
    width: int = 32,
    height: int = 24,
    frame_count: int = 5,
    fps: str = "24",
) -> TimelineBinding:
    return build_timeline_binding(
        source_path,
        analysis_source,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        root=root,
    )


def _make_panning_clip_with_scratch_and_scene_bar() -> tuple[list[np.ndarray], np.ndarray]:
    h, w = 128, 160
    rng = np.random.default_rng(123)
    canvas = rng.normal(125.0, 18.0, size=(h + 40, w + 140)).astype(np.float32)
    canvas = cv2.GaussianBlur(canvas, (7, 7), 1.4)
    cv2.rectangle(canvas, (85, 15), (110, 145), 60.0, 3)
    cv2.line(canvas, (140, 10), (140, 155), 195.0, 3)
    canvas_u8 = np.clip(canvas, 15, 240).astype(np.uint8)

    pan_clean = [
        cv2.cvtColor(canvas_u8[20 : 20 + h, 10 + t * 8 : 10 + t * 8 + w], cv2.COLOR_GRAY2BGR)
        for t in range(9)
    ]
    frames_scratched, gt_masks, _ = bdm.inject_extended_scratches(
        pan_clean,
        bdm.ExtendedScratchConfig(
            orientation="vertical",
            angle_deg=0.0,
            width=2,
            opacity=0.9,
            bright=True,
            duration=9,
            drift_px=0.1,
            seed=77,
        ),
    )
    return frames_scratched, gt_masks[4]


def test_precedence_policies_and_protect_veto_survives_dilation() -> None:
    shape = (12, 16)
    auto_scratch = np.zeros(shape, dtype=np.uint8)
    auto_dust = np.zeros(shape, dtype=np.uint8)
    auto_uncertain = np.zeros(shape, dtype=np.uint8)
    auto_combined = np.zeros(shape, dtype=np.uint8)
    edit_u8 = np.zeros(shape, dtype=np.uint8)

    # Pixel layout:
    # (1, 1): auto scratch only
    auto_scratch[1, 1] = 255
    # (1, 3): auto dust only
    auto_dust[1, 3] = 255
    # (1, 5): auto uncertain only
    auto_uncertain[1, 5] = 255
    # (1, 7): auto combined only
    auto_combined[1, 7] = 255
    # (2, 2): auto scratch + scratch_reject
    auto_scratch[2, 2] = 255
    edit_u8[2, 2] = BIT_SCRATCH_REJECT
    # (2, 4): conflicting scratch_add | scratch_reject -> REJECT wins (fail-closed)
    edit_u8[2, 4] = BIT_SCRATCH_ADD | BIT_SCRATCH_REJECT
    # (2, 6): conflicting dust_add | dust_reject -> REJECT wins (fail-closed)
    edit_u8[2, 6] = BIT_DUST_ADD | BIT_DUST_REJECT
    # (2, 8): auto uncertain + uncertain_reject
    auto_uncertain[2, 8] = 255
    edit_u8[2, 8] = BIT_UNCERTAIN_REJECT
    # (4, 4): human scratch_add without auto
    edit_u8[4, 4] = BIT_SCRATCH_ADD
    # (4, 6): human dust_add without auto (and overlaps auto_uncertain)
    auto_uncertain[4, 6] = 255
    edit_u8[4, 6] = BIT_DUST_ADD
    # (6, 6): human scratch_add + PROTECT_INTACT -> PROTECT wins
    edit_u8[6, 6] = BIT_SCRATCH_ADD | BIT_PROTECT_INTACT
    # (8, 8): auto scratch + PROTECT_INTACT -> PROTECT wins
    auto_scratch[8, 8] = 255
    edit_u8[8, 8] = BIT_PROTECT_INTACT
    # (4, 5): PROTECT_INTACT immediately adjacent to human dust_add at (4, 6) and scratch_add at (4, 4)
    edit_u8[4, 5] = BIT_PROTECT_INTACT
    # (4, 7): explicit dust_reject + scratch_reject immediately adjacent to human dust_add at (4, 6)
    edit_u8[4, 7] = BIT_DUST_REJECT | BIT_SCRATCH_REJECT

    cons = resolve_effective_channels(
        auto_scratch,
        auto_dust,
        auto_uncertain,
        edit_u8,
        policy=RepairInclusionPolicy.CONSERVATIVE,
        auto_combined=auto_combined,
        dilate=0,
    )
    assert cons["effective"][1, 1] == 255
    assert cons["effective"][1, 3] == 255
    assert cons["effective"][1, 5] == 0
    assert cons["effective"][1, 7] == 0
    assert cons["effective"][2, 2] == 0
    assert cons["effective"][2, 4] == 0
    assert cons["effective"][2, 6] == 0
    assert cons["effective"][2, 8] == 0
    assert cons["effective"][4, 4] == 255
    assert cons["effective"][4, 6] == 255
    assert cons["uncertain"][4, 6] == 0  # promoted to dust_add
    assert cons["effective"][6, 6] == 0
    assert cons["effective"][8, 8] == 0
    assert cons["protected"][4, 5] == 255

    high_recall = resolve_effective_channels(
        auto_scratch,
        auto_dust,
        auto_uncertain,
        edit_u8,
        policy=RepairInclusionPolicy.HIGH_RECALL,
        auto_combined=auto_combined,
        dilate=0,
    )
    assert high_recall["effective"][1, 5] == 255
    assert high_recall["effective"][2, 8] == 0
    assert high_recall["effective"][6, 6] == 0
    assert high_recall["effective"][8, 8] == 0

    scratch_only = resolve_effective_channels(
        auto_scratch,
        auto_dust,
        auto_uncertain,
        edit_u8,
        policy=RepairInclusionPolicy.SCRATCH_ONLY,
        auto_combined=auto_combined,
        dilate=0,
    )
    assert scratch_only["effective"][1, 1] == 255
    assert scratch_only["effective"][4, 4] == 255
    assert scratch_only["effective"][1, 3] == 0
    assert scratch_only["effective"][4, 6] == 0

    combined = resolve_effective_channels(
        auto_scratch,
        auto_dust,
        auto_uncertain,
        edit_u8,
        policy=RepairInclusionPolicy.COMBINED_DEFECT,
        auto_combined=auto_combined,
        dilate=0,
    )
    assert combined["effective"][1, 7] == 255
    assert combined["effective"][6, 6] == 0

    # Dilation expands repair into unprotected neighbors while strictly re-enforcing PROTECT_INTACT & REJECT.
    dilated = resolve_effective_channels(
        auto_scratch,
        auto_dust,
        auto_uncertain,
        edit_u8,
        policy=RepairInclusionPolicy.CONSERVATIVE,
        auto_combined=auto_combined,
        dilate=2,
    )
    assert dilated["effective"][4, 3] == 255  # expanded from (4, 4)
    assert dilated["effective"][4, 5] == 0    # protected pixel between (4, 4) and (4, 6) stays vetoed
    assert dilated["effective"][4, 7] == 0    # explicit reject stays excluded
    assert dilated["effective"][6, 6] == 0
    assert dilated["effective"][8, 8] == 0
    assert np.count_nonzero((dilated["effective"] > 0) & (dilated["protected"] > 0)) == 0


def test_apply_stroke_transition_conflict_reversal_and_provenance_preservation() -> None:
    base = np.zeros((6, 6), dtype=np.uint8)
    sel_a = np.zeros((6, 6), dtype=bool)
    sel_a[1:3, 1:3] = True
    sel_b = np.zeros((6, 6), dtype=bool)
    sel_b[2:4, 2:4] = True

    # Scratch reject -> scratch add clears reject while preserving reviewed/dust bits
    m = apply_stroke_transition(base, sel_a, "mark_reviewed")
    m = apply_stroke_transition(m, sel_a, "dust_add")
    m = apply_stroke_transition(m, sel_a, "scratch_reject")
    assert int(m[1, 1]) == (BIT_REVIEWED | BIT_DUST_ADD | BIT_SCRATCH_REJECT)

    m2 = apply_stroke_transition(m, sel_a, "scratch_add")
    assert int(m2[1, 1]) == (BIT_REVIEWED | BIT_DUST_ADD | BIT_SCRATCH_ADD)
    # Input array was never mutated
    assert int(m[1, 1]) & BIT_SCRATCH_REJECT

    # Scratch add -> scratch reject clears scratch_add
    m3 = apply_stroke_transition(m2, sel_a, "scratch_reject")
    assert int(m3[1, 1]) == (BIT_REVIEWED | BIT_DUST_ADD | BIT_SCRATCH_REJECT)

    # Dust add <-> dust reject conflict reversal
    m4 = apply_stroke_transition(m3, sel_a, "dust_reject")
    assert int(m4[1, 1]) == (BIT_REVIEWED | BIT_DUST_REJECT | BIT_SCRATCH_REJECT)
    m5 = apply_stroke_transition(m4, sel_a, "dust_add")
    assert int(m5[1, 1]) == (BIT_REVIEWED | BIT_DUST_ADD | BIT_SCRATCH_REJECT)

    # protect preserves existing provenance flags; unprotect removes only protect
    m6 = apply_stroke_transition(m5, sel_b, "protect")
    assert int(m6[2, 2]) == (BIT_REVIEWED | BIT_DUST_ADD | BIT_SCRATCH_REJECT | BIT_PROTECT_INTACT)
    m7 = apply_stroke_transition(m6, sel_b, "unprotect")
    assert int(m7[2, 2]) == (BIT_REVIEWED | BIT_DUST_ADD | BIT_SCRATCH_REJECT)

    # reject_all clears scratch_add and dust_add while setting all reject flags
    m8 = apply_stroke_transition(m5, sel_a, "reject_all")
    assert (int(m8[1, 1]) & (BIT_SCRATCH_ADD | BIT_DUST_ADD)) == 0
    assert (int(m8[1, 1]) & (BIT_SCRATCH_REJECT | BIT_DUST_REJECT | BIT_UNCERTAIN_REJECT)) == (
        BIT_SCRATCH_REJECT | BIT_DUST_REJECT | BIT_UNCERTAIN_REJECT
    )

    # clear_edits resets all bits on selected region (whole-frame clear resets frame bits)
    all_px = np.ones((6, 6), dtype=bool)
    cleared = apply_stroke_transition(m6, all_px, "clear_edits")
    assert int(np.count_nonzero(cleared)) == 0

    with pytest.raises(ValueError, match="Unsupported stroke tool"):
        apply_stroke_transition(base, sel_a, "not_a_tool")


def test_js_python_exact_bitmask_b64_and_png_roundtrip() -> None:
    # Cover every valid flag combination 0..127 (all 7 active bits) in a 16x16 grid
    grid = (np.arange(256, dtype=np.uint8) & KNOWN_EDIT_BITS_MASK).reshape((16, 16))
    b64_py = encode_bitmask_b64(grid)

    # Also verify lossless PNG encode/decode preserves 0..127 bit-for-bit
    png_bytes = encode_bitmask_png(grid)
    decoded_png = decode_bitmask_png(png_bytes, width=16, height=16)
    np.testing.assert_array_equal(decoded_png, grid)

    editor_js = (ROOT_DIR / "ai_remaster_gui" / "static" / "js" / "dust-mask-editor.js").read_text(encoding="utf-8")
    node_script = """
const fs = require('node:fs');
const vm = require('node:vm');
const payload = JSON.parse(fs.readFileSync(0, 'utf8'));
const ctx = vm.createContext({ Buffer, Uint8Array, ArrayBuffer, Number, Object, String, Math, Error });
vm.runInContext(payload.code, ctx);
const arr = ctx.dustBase64ToBitmask(payload.b64, payload.width, payload.height);
const values = Array.from(arr);
const toolB64s = {};
for (const tool of ctx.DUST_EDIT_TOOLS) {
  const stroked = ctx.dustApplyStrokeTransition(arr, payload.indices, tool);
  toolB64s[tool] = ctx.dustBitmaskToBase64(stroked);
}
process.stdout.write(JSON.stringify({ values, jsTools: Array.from(ctx.DUST_EDIT_TOOLS), toolB64s }));
"""
    proc = subprocess.run(
        ["node", "-e", node_script],
        input=json.dumps(
            {
                "code": editor_js,
                "b64": b64_py,
                "width": 16,
                "height": 16,
                "indices": [0, 17, 127, 255],
            }
        ),
        text=True,
        capture_output=True,
        check=True,
    )
    js_out = json.loads(proc.stdout)
    assert js_out["values"] == grid.ravel().tolist()
    assert set(js_out["jsTools"]) == set(VALID_STROKE_TOOLS)

    sel = np.zeros((16, 16), dtype=bool)
    sel.ravel()[[0, 17, 127, 255]] = True
    for tool in sorted(VALID_STROKE_TOOLS):
        py_from_js = decode_bitmask_b64(js_out["toolB64s"][tool], 16, 16, validate_known_bits=True)
        expected_py = apply_stroke_transition(grid, sel, tool)
        np.testing.assert_array_equal(py_from_js, expected_py, err_msg=f"JS<->Python mismatch on tool {tool!r}")

    expected_py = apply_stroke_transition(grid, sel, "protect")

    # Full ZIP store PNG round-trip preserves 0..127 exactly
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _make_source_file(root)
        binding = _make_binding(root, src, width=16, height=16, frame_count=3)
        store = DustMaskEditStore(binding, root=root)
        store.save_frame(1, expected_py, expected_revision=0)
        reloaded = load_verified_store(binding, root=root)
        np.testing.assert_array_equal(reloaded.read_frame(1), expected_py)
        assert reloaded.read_frame_b64(1) == js_out["toolB64s"]["protect"]


def test_atomic_write_crash_simulation_and_integrity_tamper_checks() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _make_source_file(root)
        binding = _make_binding(root, src, width=20, height=16, frame_count=4)
        store = DustMaskEditStore(binding, root=root)

        mask_rev1 = np.zeros((16, 20), dtype=np.uint8)
        mask_rev1[3:7, 4:8] = BIT_SCRATCH_ADD | BIT_REVIEWED
        summary1 = store.save_frame(0, mask_rev1, expected_revision=0)
        assert summary1["edit_revision"] == 1
        assert store.zip_path.is_file()
        assert store.meta_path.is_file()

        rev1_zip_bytes = store.zip_path.read_bytes()
        rev1_meta_bytes = store.meta_path.read_bytes()

        # 1. Simulate crash inside os.replace mid-save on revision 2
        mask_rev2 = mask_rev1.copy()
        mask_rev2[8:11, 10:14] = BIT_PROTECT_INTACT
        with mock.patch.object(dme.os, "replace", side_effect=OSError("simulated power loss")):
            with pytest.raises(OSError, match="simulated power loss"):
                store.save_frame(0, mask_rev2, expected_revision=1)

        # Revision 1 archive and metadata remain untouched and readable
        assert store.zip_path.read_bytes() == rev1_zip_bytes
        assert store.meta_path.read_bytes() == rev1_meta_bytes
        recovered = load_verified_store(binding, root=root)
        assert recovered.edit_revision == 1
        np.testing.assert_array_equal(recovered.read_frame(0), mask_rev1)

        # Stale .tmp / .partial leftovers do not break subsequent loads or saves
        (store.masks_dir / f"{store.timeline_id}.zip.partial.stale.tmp").write_bytes(b"garbage")
        assert load_verified_store(binding, root=root).edit_revision == 1

        # 2. Stale expected_revision fails closed with EditRevisionConflictError
        with pytest.raises(EditRevisionConflictError):
            recovered.save_frame(1, mask_rev2, expected_revision=0)

        # 3. Tampered zip_sha256 in .meta.json fails closed
        meta_obj = json.loads(rev1_meta_bytes.decode("utf-8"))
        tampered_sha_meta = {**meta_obj, "zip_sha256": "0" * 64}
        store.meta_path.write_text(json.dumps(tampered_sha_meta), encoding="utf-8")
        with pytest.raises(EditStoreIntegrityError, match="SHA-256 mismatch"):
            load_verified_store(binding, root=root)

        # 4. Tampered edit_revision in .meta.json (mismatches embedded edit_meta.json) fails closed
        tampered_rev_meta = {**meta_obj, "edit_revision": 99}
        store.meta_path.write_text(json.dumps(tampered_rev_meta), encoding="utf-8")
        with pytest.raises(EditStoreIntegrityError, match="revision mismatch"):
            load_verified_store(binding, root=root)
        store.meta_path.write_bytes(rev1_meta_bytes)

        # 5. Corrupt ZIP payload fails closed with EditStoreIntegrityError
        store.zip_path.write_bytes(b"corrupt-zip-bytes")
        with pytest.raises(EditStoreIntegrityError):
            load_verified_store(binding, root=root)
        store.zip_path.write_bytes(rev1_zip_bytes)

        # 6. Corrupt PNG member bytes inside ZIP (with recomputed zip_sha256 -> frame SHA mismatch) fails closed
        with zipfile.ZipFile(store.zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            zf.writestr("f000000.png", b"not-a-png")
            zf.writestr("edit_meta.json", json.dumps({
                "edit_schema_version": EDIT_SCHEMA_VERSION,
                "timeline_id": store.timeline_id,
                "timeline_binding": binding.to_dict(),
                "edit_revision": 1,
                "edited_frames": [0],
                "frame_sha256s": meta_obj["frame_sha256s"],
            }))
        bad_zip_sha = dme.compute_file_sha256(store.zip_path)
        store.meta_path.write_text(json.dumps({**meta_obj, "zip_sha256": bad_zip_sha}), encoding="utf-8")
        with pytest.raises(EditStoreIntegrityError):
            load_verified_store(binding, root=root)

        # Restore valid rev1 and verify timeline mismatch checks
        store.zip_path.write_bytes(rev1_zip_bytes)
        store.meta_path.write_bytes(rev1_meta_bytes)

        bad_dims = TimelineBinding(
            source_path=binding.source_path,
            analysis_source=binding.analysis_source,
            source_sha256=binding.source_sha256,
            source_size=binding.source_size,
            source_mtime_ns=binding.source_mtime_ns,
            width=40,
            height=16,
            frame_count=4,
            fps="24",
        )
        # Note: compute_timeline_id hashes geometry/source, so check direct verify against mismatched binding
        # AND check store_compatibility_status when a source file replaces the previous clip under same path.
        mismatched_store = DustMaskEditStore(binding, root=root)
        mismatched_store.binding = bad_dims
        with pytest.raises(EditTimelineMismatchError):
            mismatched_store.verify()

        # Simulate replacing source clip on disk at same path (changing sha/size/frame_count)
        src_changed = _make_source_file(root, payload=b"different-source-content-bytes")
        new_binding = _make_binding(root, src_changed, width=20, height=16, frame_count=7)
        # Copy rev1 archive under new_binding's timeline_id to test load_verified_store + compatibility check
        target_store = DustMaskEditStore(new_binding, root=root)
        target_store.zip_path.write_bytes(rev1_zip_bytes)
        target_store.meta_path.write_bytes(rev1_meta_bytes)
        with pytest.raises(EditTimelineMismatchError):
            load_verified_store(new_binding, root=root)
        compat = store_compatibility_status(new_binding, root=root)
        assert compat["compatible"] is False
        assert compat["compatibility_reason"]


def test_dust_e2_adapter_opt_in_and_three_state_non_flattening() -> None:
    assert DETECTOR_NAME == "experimental_e2"
    assert E2_DETECTOR_INFO["experimental_opt_in_required"] is True
    assert E2_PRESET_INFO["detector"] == "experimental_e2"

    for falsey in (False, "false", "", None, "yes", "1"):
        assert is_research_confirmed(falsey) is False
    for truthy in (True, "true", " TRUE "):
        assert is_research_confirmed(truthy) is True

    frames, _gt = _make_panning_clip_with_scratch_and_scene_bar()

    # Fail closed without explicit research_confirmed == 'true'
    for bad_flag in (False, "false", "", "1"):
        with pytest.raises(ExperimentalOptInRequiredError):
            run_experimental_e2_sequence(frames, research_confirmed=bad_flag)

    result = run_experimental_e2_sequence(frames, sensitivity=0.65, research_confirmed="true", target_indices=[4])
    assert result.metadata["detector"] == "experimental_e2"
    assert result.metadata["research_confirmed"] is True
    assert len(result.scratch_mask) == 9
    assert len(result.uncertain_mask) == 9
    assert len(result.metadata["vetoed_masks"]) == 9
    assert len(result.dust_mask) == 9

    # Dust channel must remain all-zero and marked UNSUPPORTED
    for idx in range(9):
        assert int(np.count_nonzero(result.dust_mask[idx])) == 0
        assert result.multi_channel_masks[idx].provenance["dust"].origin == ChannelOrigin.UNSUPPORTED
        assert result.multi_channel_masks[idx].provenance["dust"].is_available is False

    # 3-state channels at frame 4 are mutually exclusive and never flattened
    hc = result.scratch_mask[4] > 0
    unc = result.uncertain_mask[4] > 0
    vet = result.metadata["vetoed_masks"][4] > 0
    assert not np.any(hc & unc)
    assert not np.any(hc & vet)
    assert not np.any(unc & vet)

    # CONSERVATIVE repair authorization only includes high_confidence_damage, never uncertain or vetoed
    eff_cons = result.multi_channel_masks[4].get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
    np.testing.assert_array_equal(eff_cons, result.scratch_mask[4])
    assert not np.any((eff_cons > 0) & unc)
    assert not np.any((eff_cons > 0) & vet)


def test_mandatory_non_gt_provenance_and_untouched_frames_export() -> None:
    assert GROUND_TRUTH_STATUS == "operational_edits_only_not_independent_gt"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _make_source_file(root)
        binding = _make_binding(root, src, width=16, height=12, frame_count=4)
        store = DustMaskEditStore(binding, root=root)

        # Frame 1 edited with add, reject, protect, and reviewed marker; frames 0, 2, 3 untouched
        edit1 = np.zeros((12, 16), dtype=np.uint8)
        edit1[1:3, 1:3] = BIT_SCRATCH_ADD | BIT_REVIEWED
        edit1[4:6, 4:6] = BIT_SCRATCH_REJECT
        edit1[7:9, 7:9] = BIT_DUST_REJECT | BIT_PROTECT_INTACT
        store.save_frame(1, edit1, expected_revision=0)

        # Untouched auto predictions exist on frame 0 and frame 2
        auto_frames = {
            0: {
                "scratch": np.pad(np.full((2, 2), 255, dtype=np.uint8), ((2, 8), (2, 12))),
                "dust": np.zeros((12, 16), dtype=np.uint8),
                "uncertain": np.zeros((12, 16), dtype=np.uint8),
                "vetoed": np.zeros((12, 16), dtype=np.uint8),
            },
            1: {
                "scratch": np.pad(np.full((2, 2), 255, dtype=np.uint8), ((4, 6), (4, 10))),
                "dust": np.pad(np.full((2, 2), 255, dtype=np.uint8), ((7, 3), (7, 7))),
                "uncertain": np.zeros((12, 16), dtype=np.uint8),
                "vetoed": np.zeros((12, 16), dtype=np.uint8),
            },
        }

        summary = build_edit_provenance_summary(store, auto_frames=auto_frames, include_per_frame=True)
        assert summary["ground_truth_status"] == "operational_edits_only_not_independent_gt"
        assert summary["protect_or_reject_auto_promoted_to_gt"] is False
        assert summary["untouched_auto_as_gt"] is False

        counts = summary["counts"]
        assert counts["total_pixels"] == 4 * 12 * 16
        assert counts["human_scratch_add"] == 4
        assert counts["human_scratch_reject"] == 4
        assert counts["human_dust_reject"] == 4
        assert counts["protect_intact"] == 4
        assert counts["operational_reviewed"] == 12
        assert counts["unreviewed"] == (4 * 12 * 16) - 12
        # Untouched auto predictions on frame 0 must NOT show up as human edits
        assert counts["auto_predicted"] == 12

        # Export bundle works cleanly when untouched frames exist (including frame 0 with auto only & frames 2,3 untouched)
        export_dir = root / "exports" / "bundle_with_untouched"
        exported = export_provenance_bundle(
            store,
            export_dir,
            policy=RepairInclusionPolicy.CONSERVATIVE,
            auto_frames=auto_frames,
            frames=[0, 1, 2],
        )
        prov_data = json.loads(Path(exported["provenance_path"]).read_text(encoding="utf-8"))
        assert prov_data["ground_truth_status"] == "operational_edits_only_not_independent_gt"
        assert prov_data["protect_or_reject_auto_promoted_to_gt"] is False
        assert prov_data["untouched_auto_as_gt"] is False
        assert (export_dir / "human_edits" / "f000002.png").is_file()
        assert (export_dir / "human_edits" / "scratch_reject" / "f000001.png").is_file()
        assert (export_dir / "protect_intact" / "f000001.png").is_file()
        assert (export_dir / "effective_repair" / "f000001.png").is_file()

        # Frame 1 effective repair has scratch_add (4px) and excluded rejected/protected auto proposals
        eff1 = cv2.imread(str(export_dir / "effective_repair" / "f000001.png"), cv2.IMREAD_GRAYSCALE)
        assert int(np.count_nonzero(eff1)) == 4
        assert int(np.count_nonzero(eff1[4:6, 4:6])) == 0
        assert int(np.count_nonzero(eff1[7:9, 7:9])) == 0
        # Untouched frame 2 has all-zero human edit PNG and all-zero effective repair
        human2 = cv2.imread(str(export_dir / "human_edits" / "f000002.png"), cv2.IMREAD_GRAYSCALE)
        assert int(np.count_nonzero(human2)) == 0

        # Even an empty store (zero edited frames across entire timeline) exports cleanly
        store.clear_all()
        empty_export_dir = root / "exports" / "empty_bundle"
        empty_exp = export_provenance_bundle(store, empty_export_dir, auto_frames=auto_frames)
        empty_prov = json.loads(Path(empty_exp["provenance_path"]).read_text(encoding="utf-8"))
        assert empty_prov["ground_truth_status"] == "operational_edits_only_not_independent_gt"
        assert empty_prov["counts"]["operational_reviewed"] == 0
        assert empty_prov["counts"]["unreviewed"] == 4 * 12 * 16


def test_timeline_id_determinism_edit_signature_and_reset_frame() -> None:
    assert compute_edit_signature("a" * 64, 0, [0]) == ""
    assert compute_edit_signature("a" * 64, 1, []) == ""
    assert compute_edit_signature("a" * 64, 2, [0, 3]) == ("a" * 64) + ":rev2"

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _make_source_file(root)
        b1 = _make_binding(root, src, width=16, height=12, frame_count=3, fps="24")
        # Different mtime_ns (same sha256) yields identical timeline_id
        b2 = TimelineBinding(
            source_path=b1.source_path,
            analysis_source=b1.analysis_source,
            source_sha256=b1.source_sha256,
            source_size=b1.source_size,
            source_mtime_ns=b1.source_mtime_ns + 999999,
            width=16,
            height=12,
            frame_count=3,
            fps="24/1",
        )
        assert compute_timeline_id(b1) == compute_timeline_id(b2)

        # RGB PNG is rejected by decode_bitmask_png
        rgb_png = cv2.imencode(".png", np.zeros((12, 16, 3), dtype=np.uint8))[1].tobytes()
        with pytest.raises(EditStoreIntegrityError, match="single-channel 8-bit grayscale"):
            decode_bitmask_png(rgb_png, width=16, height=12)

        store = DustMaskEditStore(b1, root=root)
        m0 = np.zeros((12, 16), dtype=np.uint8)
        m0[2:5, 3:6] = BIT_SCRATCH_ADD
        m1 = np.zeros((12, 16), dtype=np.uint8)
        m1[4:7, 5:8] = BIT_DUST_ADD | BIT_REVIEWED
        store.save_frames({0: m0, 1: m1}, expected_revision=0)
        assert store.edited_frames() == [0, 1]

        res = store.reset_frame(0, expected_revision=1)
        assert res["edited_frames"] == [1]
        assert int(np.count_nonzero(store.read_frame(0))) == 0
        np.testing.assert_array_equal(store.read_frame(1), m1)

        cleared = store.reset_frame(1, expected_revision=2)
        assert cleared["has_edits"] is False
        assert not store.zip_path.exists()
        assert not store.meta_path.exists()


def test_compose_edited_masks_detects_toctou_revision_drift() -> None:
    from ai_remaster_gui import dust as dust_stage

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "input" / "clip.mkv"
        source.parent.mkdir(parents=True, exist_ok=True)
        writer = dust_stage.Writer(source, {"frames": 3, "width": 32, "height": 24, "fps": "12"})
        for _ in range(3):
            writer.write(np.full((24, 32, 3), 120, dtype=np.uint8))
        writer.close()

        with mock.patch.object(dust_stage, "ROOT", root), mock.patch.object(dme, "ROOT", root):
            dust_stage.run(source, {"detector": "legacy", "repair": "telea"}, "mask")
            binding = _make_binding(root, source, width=32, height=24, frame_count=3, fps="12")
            store = DustMaskEditStore(binding, root=root)
            edit0 = np.zeros((24, 32), dtype=np.uint8)
            edit0[2:6, 2:6] = BIT_SCRATCH_ADD
            store.save_frame(0, edit0, expected_revision=0)

            orig_read_all = DustMaskEditStore.read_all_frames

            def _drift_on_read(self_store):
                data = orig_read_all(self_store)
                # Simulate a concurrent edit committing rev2 after ctx was resolved at rev1
                edit1 = np.zeros((24, 32), dtype=np.uint8)
                edit1[8:12, 8:12] = BIT_DUST_ADD
                self_store.save_frame(1, edit1, expected_revision=1)
                return data

            with mock.patch.object(DustMaskEditStore, "read_all_frames", _drift_on_read):
                with pytest.raises(EditRevisionConflictError, match="Edit revision changed during mask composition"):
                    dust_stage.compose_edited_masks(source, {"detector": "legacy", "repair": "telea"}, root=root)


def test_commercial_policy_allows_e2_and_blocks_unknown_or_noncommercial_detectors() -> None:
    from ai_remaster_gui.commercial_policy import stage_issue

    base_cfg = {"global": {"commercial_model_guard": "true"}}
    assert stage_issue({**base_cfg, "dust": {"detector": "experimental_e2", "repair": "telea"}}, stage="dust") == ""
    assert stage_issue({**base_cfg, "dust": {"detector": "farneback_3f", "repair": "telea"}}, stage="dust") == ""

    unknown_msg = stage_issue({**base_cfg, "dust": {"detector": "unlisted_detector", "repair": "telea"}}, stage="dust")
    assert "unlisted_detector" in unknown_msg and "not cleared for commercial use" in unknown_msg

    dart_msg = stage_issue({**base_cfg, "dust": {"detector": "dart", "repair": "telea"}}, stage="dust")
    assert "DART" in dart_msg

    pp_msg = stage_issue({**base_cfg, "dust": {"detector": "experimental_e2", "repair": "propainter"}}, stage="dust")
    assert "ProPainter" in pp_msg


def test_export_provenance_bundle_validates_frames_before_mkdir() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        src = _make_source_file(root)
        binding = _make_binding(root, src, width=16, height=12, frame_count=4)
        store = DustMaskEditStore(binding, root=root)
        edit = np.zeros((12, 16), dtype=np.uint8)
        edit[1:3, 1:3] = BIT_SCRATCH_ADD
        store.save_frame(0, edit, expected_revision=0)

        export_dir = root / "exports" / "invalid_frames_target"
        for bad_frames in ([True], "01", b"01", [-1], ["bad"]):
            with pytest.raises((ValueError, TypeError)):
                export_provenance_bundle(store, export_dir, frames=bad_frames)
            assert not export_dir.exists(), f"export_dir must not be created for invalid frames={bad_frames!r}"


def test_ignore_edits_fails_closed_on_repair_and_adoption_while_mask_keeps_edits_intact() -> None:
    import threading
    from ai_remaster_gui import dust as dust_stage, server
    from ai_remaster_gui.runtime_settings import base_settings

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "input" / "clip.mkv"
        source.parent.mkdir(parents=True, exist_ok=True)
        writer = dust_stage.Writer(source, {"frames": 3, "width": 32, "height": 24, "fps": "12"})
        for _ in range(3):
            writer.write(np.full((24, 32, 3), 120, dtype=np.uint8))
        writer.close()

        settings = base_settings()
        settings["global"].update(source=str(source), dust="true", expand_outpaint="false", colorize="false")
        settings["dust"].update(detector="legacy", repair="telea")

        app = server.PipelineApp.__new__(server.PipelineApp)
        app.settings = settings
        app.project_path = None
        app.log = []
        app.process = None
        app.quitting = False
        app.queue_running = False
        app.running_stage = ""
        app.running_stage_key = ""
        app.running_reference_manifest = ""
        app.running_reference_index = None
        app.denoise_grid_request = {}
        app.run_started_at = 0.0
        app.recomp_total_frames = 0
        app.recomp_progress_run = 0.0
        app.lock = threading.Lock()
        app.source_analysis_lock = threading.Lock()
        app.source_analysis_status = {}
        app.source_analysis_results = {}
        app.source_analysis_threads = set()
        app.source_tone_default_keys = set()
        app.save = mock.Mock()
        app.ensure_pipeline_source = mock.Mock()
        app.remember_project_path = mock.Mock()

        with (
            mock.patch.object(server, "ROOT", root),
            mock.patch.object(dust_stage, "ROOT", root),
            mock.patch.object(dme, "ROOT", root),
        ):
            # Seed base repair + save manual edit on frame 1
            base_vals = {"detector": "legacy", "repair": "telea"}
            dust_stage.run(source, base_vals, "mask")
            dust_stage.run(source, base_vals, "repair")
            base_folder = dust_stage.folder_for(source, base_vals)

            edit1 = np.zeros((24, 32), dtype=np.uint8)
            edit1[5:9, 6:10] = BIT_SCRATCH_ADD | BIT_PROTECT_INTACT
            save_res = app.save_dust_mask_frame(1, encode_bitmask_b64(edit1), analysis_source="stage", expected_revision=0)
            assert save_res["edit_revision"] == 1

            # ignore_edits='true' MUST fail closed on repair and adoption when manual edits exist on disk
            with pytest.raises(ValueError, match="Cannot ignore active manual mask edits during repair"):
                app.run_dust({**base_vals, "ignore_edits": "true"}, "repair")

            with pytest.raises(ValueError, match="Cannot ignore active manual mask edits during adoption"):
                app.adopt_dust({**base_vals, "ignore_edits": "true"})

            # ignore_edits='true' on kind='mask' forwards --ignore-edits without --edit-signature
            with mock.patch.object(app, "_start_denoise_matrix", return_value=(True, "launched")) as launch:
                ok, _msg = app.run_dust({**base_vals, "ignore_edits": "true"}, "mask")
            assert ok is True
            cmd = launch.call_args.args[1]
            assert "--ignore-edits" in cmd
            assert "--edit-signature" not in cmd

            # Regenerating base mask with ignore_edits=True restores <base_key>/ without touching manual edits
            import shutil
            shutil.rmtree(base_folder)
            assert not base_folder.exists()
            dust_stage.run(source, base_vals, "mask", ignore_edits=True)
            assert (base_folder / "effective_mask.mkv").is_file()
            loaded = app.load_dust_mask_frame(1, analysis_source="stage")
            assert loaded["edit_revision"] == 1
            np.testing.assert_array_equal(decode_bitmask_b64(loaded["edit_mask_b64"], 32, 24), edit1)
