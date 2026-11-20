---
title: "Editable Defect Masks in ARP Workflow Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-19-editable-defect-masks-arp-gui-design.md"
created: "2026-11-19T16:41:26+01:00"
status: "approved"
total_phases: 5
estimated_files: 17
task_complexity: "medium"
---

# Editable Defect Masks in ARP Workflow Implementation Plan

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `coder` (Phases 1–4), `tester` (Phase 5)
- **Estimated effort**: Focused MVP spanning non-destructive sparse ZIP storage & `.arpp` persistence (Phase 1), pipeline composed-mask adapter & opt-in E2 (Phase 2), parallel backend HTTP API (Phase 3) and frontend canvas editor (Phase 4), and end-to-end/round-trip verification + Windows guide (Phase 5).

## Dependency Graph

```
Phase 1 (Storage & .arpp Persistence)
   |
   v
Phase 2 (Pipeline Adapter & Opt-In E2)
   |
   +--------------------+
   |                    |
   v                    v
Phase 3              Phase 4
(Backend HTTP API)   (Frontend Canvas UI)
   |                    |
   +---------+----------+
             |
             v
Phase 5 (E2E Tests & Windows Verification Docs)
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1 | Phase 1 | Sequential | 1 | Core bitmask codec, atomic ZIP store, `.arpp` allowlist |
| 2 | Phase 2 | Sequential | 1 | Pipeline composed-mask adapter, cache keying, opt-in E2 |
| 3 | Phase 3, Phase 4 | Parallel eligible | 2 | Disjoint backend (`http_handler`/`server`) and frontend (`static/`) files |
| 4 | Phase 5 | Sequential | 1 | E2E regression tests, JS↔Python PNG round-trip, Windows guide |

---

## Phase 1: Project Bitmask Edit Store and Bundle Persistence

### Objective
Implement the non-destructive sparse packed-bitmask ZIP store, checksum/timeline verification, deterministic edit & resolution precedence algebra, mandatory provenance builder, optional provenance exporter, and `.arpp` bundle integration.

### Agent: `coder`
### Parallel: No

### Files to Create
- `ai_remaster_gui/dust_mask_edits.py` — Bit constants (`BIT_SCRATCH_ADD=1`, `BIT_SCRATCH_REJECT=2`, `BIT_DUST_ADD=4`, `BIT_DUST_REJECT=8`, `BIT_PROTECT_INTACT=16`, `BIT_UNCERTAIN_REJECT=32`, `BIT_REVIEWED=64`), `apply_stroke_transition(...)`, `resolve_effective_channels(...)`, `TimelineBinding`, `compute_timeline_id(...)`, atomic `DustMaskEditStore` (`manifests/dust_masks/<timeline_id>.zip` + `edit_meta.json` inside ZIP + external `<timeline_id>.meta.json` with `zip_sha256` + `edit_revision`), `load_verified_store(...)` (fails closed on checksum or timeline/dimension mismatch), raw `Uint8Array` base64 ↔ grayscale PNG codecs, `build_edit_provenance_summary(...)`, and `export_provenance_bundle(...)`.

### Files to Modify
- `ai_remaster_gui/project_io.py` — Add `DUST_MASK_BUNDLE_DIR = "manifests/dust_masks/"`, update `bundle_name_allowed()` for `manifests/dust_masks/*.zip` + `.json`, update `project_asset_paths()` to package project edit archives and `.meta.json` sidecars without bundling `intermediate/dust_scratches/`.
- `ai_remaster_gui/cache.py` — Ensure `cache_categories()` never targets `manifests/dust_masks/` for cleanup deletion.

### Implementation Details
- `resolve_effective_channels(auto_scratch, auto_dust, auto_uncertain, edit_u8, policy, dilate=0)`:
  - `protected = (edit_u8 & BIT_PROTECT_INTACT) != 0`
  - For scratch/dust: if `protected` -> `0`; elif corresponding `REJECT` bit set -> `0` (`REJECT > ADD > AUTO` fail-closed); elif corresponding `ADD` bit set -> `255`; else `auto`.
  - Uncertain: if `protected` or `BIT_UNCERTAIN_REJECT` or promoted via non-rejected `ADD` -> `0`; else `auto_uncertain`.
  - Apply `RepairInclusionPolicy` (`CONSERVATIVE` = `scratch | dust`; `HIGH_RECALL` = `scratch | dust | uncertain`; `SCRATCH_ONLY` = `scratch`; `COMBINED_DEFECT` = `scratch | dust` + non-rejected auto combined), apply `dilate` if `> 0`, then re-enforce `effective &= ~protected` (and `~explicitly_rejected`) post-dilation.
- Atomic write protocol: write ZIP to same-directory temp file -> `flush()` + `os.fsync()` -> compute `zip_sha256` -> `os.replace(tmp_zip, final_zip)` -> write `.meta.json.tmp` -> `flush()` + `os.fsync()` -> `os.replace(tmp_meta, final_meta)`. Verify `external_meta.zip_sha256 == actual_zip_sha256` and `external_meta.edit_revision == embedded_meta.edit_revision` on every read.
- Provenance distinguishes: `auto_predicted`, `human_added`, `human_rejected`, `protect_intact`, `operational_reviewed`, and `unreviewed`. Never converts `PROTECT_INTACT`, `REJECT`, or untouched auto outputs into independent ground truth (`ground_truth_status: "operational_edits_only_not_independent_gt"`).

### Validation
- `.\.venv\Scripts\python.exe -m pytest tests/test_gui_smoke.py -k "project"`
- Unit smoke script check via Python verifying atomic replace, checksum mismatch fail-closed, and post-dilation `PROTECT_INTACT` dominance.

### Dependencies
- Blocked by: None
- Blocks: 2

---

## Phase 2: Composed Mask Restoration Adapter and Opt-In E2

### Objective
Wire composed edit mask resolution cleanly into the existing `dust.py` detection/repair/10-bit pipeline without mutating base detector caches, bind `edit_signature` into composed cache identities, and expose Experimental E2 as an opt-in 3-state detector.

### Agent: `coder`
### Parallel: No

### Files to Create
- `ai_remaster_gui/dust_e2_adapter.py` — Opt-in wrapper invoking `scripts/experimental_scene_film_scratch_prototype.ExperimentalSceneFilmDetector` ONLY when detector `experimental_e2` is explicitly selected and `research_confirmed == 'true'`. Returns separate `scratch_mask` (`high_confidence_damage`), `uncertain_mask` (`uncertain_damage_candidate`), `vetoed_mask` (`likely_intact_scene_structure`), and empty `dust_mask` (`UNSUPPORTED`), never flattening uncertain or vetoed pixels into confirmed damage.

### Files to Modify
- `ai_remaster_gui/models.py` — Add `experimental_e2` choice to the `dust` stage detector selector and preset/description metadata cleanly marked `(Experimental)` while keeping `farneback_3f` default.
- `ai_remaster_gui/dust.py` — Register/dispatch `experimental_e2` (gated behind `research_confirmed`); save optional `vetoed_mask.mkv` in base `<base_key>/`; extend `options()`/`identity()` cleanly with optional `edit_signature` (`"<zip_sha256>:<edit_revision>"`) when active edits exist so `<composed_key>/` is strictly separate from immutable `<base_key>/` and stale repairs are never reused; add `ensure_composed_mask_folder(...)` reusing `<base_key>/` detector masks (without re-running detection) and writing composed `effective_mask.mkv`, `scratch_mask.mkv`, `dust_mask.mkv`, `uncertain_mask.mkv`, `measurements.csv`, and `report.json`; ensure `_run('repair')` consumes composed `effective_mask.mkv` across `telea`, `propainter`, `dart`, and `dust_native_composite.compose()` with `changed_outside_pixels == 0` including `PROTECT_INTACT`, and records detector + human-edit provenance in `restoration_provenance.json`.
- `ai_remaster_gui/frame_inspector.py` — Support inspecting optional `vetoed` overlay PNG (`_vetoed.png`) and composed effective/scratch/dust channels alongside raw base proposals.

### Validation
- `.\.venv\Scripts\python.exe -m pytest tests/test_dust_stage.py tests/test_end_to_end_repair_phase9.py tests/test_mask_coverage_failsafe.py -v`

### Dependencies
- Blocked by: 1
- Blocks: 3, 4

---

## Phase 3: GUI HTTP Routes, State Hydration, and Export API

### Objective
Expose secure loopback endpoints and `PipelineApp` state hydration to load/save/reset editable frame bitmasks, inspect composed vs. automatic layers, adopt edit-bound repairs, and trigger optional provenance export.

### Agent: `coder`
### Parallel: Yes

### Files to Modify
- `ai_remaster_gui/http_handler.py` — Add `GET /api/dust-mask-frame` and `POST /api/dust-mask-frame`, `/api/dust-mask-reset`, `/api/dust-mask-export` with existing localhost/CSRF guards and no raw filesystem path acceptance.
- `ai_remaster_gui/server.py` — Wire `dust_edit_state()`, `load_dust_mask_frame()`, `save_dust_mask_frame()`, `reset_dust_masks()`, `export_dust_provenance()` on `PipelineApp`, hydrate active `edit_signature` in `APP.state('dust')` and `validate_dust_adoption()` / `run_stage('dust')`, pass active edit binding cleanly during `/api/dust-run` / `/api/dust-inspect` even when detection reruns or switches detectors, and re-bind on `load_project()`.
- `tests/test_http_routes.py` — Update `EXPECTED_GET_ROUTES` and `EXPECTED_POST_ROUTES` and add payload/route checks for the new endpoints.

### API Contract (Shared with Phase 4)
- `GET /api/dust-mask-frame?frame=<int>&analysis_source=<stage|original>` -> `{ ok, frame, width, height, frame_count, timeline_id, edit_signature, edit_revision, compatible, compatibility_reason, edited_frames, edit_mask_b64, stats }`
- `POST /api/dust-mask-frame` (`{ frame, analysis_source, edit_mask_b64, expected_revision? }`) -> `{ ok, frame, timeline_id, edit_signature, edit_revision, edited_frames, stats, state }`
- `POST /api/dust-mask-reset` (`{ scope: "frame"|"all", frame?, analysis_source }`) -> `{ ok, timeline_id, edit_signature, edit_revision, edited_frames, state }`
- `POST /api/dust-mask-export` (`{ analysis_source }`) -> `{ ok, export_dir, summary }`
- `state.dust_edits` in `/api/state?active=dust`: `{ timeline_id, edit_signature, edit_revision, compatible, compatibility_reason, edited_frames, summary }`.

### Validation
- `.\.venv\Scripts\python.exe -m pytest tests/test_http_routes.py tests/test_http_security.py tests/test_dust_stage_data_api.py -v`

### Dependencies
- Blocked by: 2
- Blocks: 5

---

## Phase 4: Interactive Dust Mask Editor Frontend UI

### Objective
Integrate an interactive, non-destructive multi-layer mask editor into the existing **Dust & Scratches** canvas workbench without replacing ARP's UI architecture or breaking existing comparison/playback features.

### Agent: `coder`
### Parallel: Yes

### Files to Create
- `ai_remaster_gui/static/js/dust-mask-editor.js` — Lossless base64 `Uint8Array` bitmask decoder/encoder, deterministic brush transitions (`Add Scratch`, `Reject Scratch`, `Add Dust`, `Reject Dust`, `Reject All`, `Protect Intact`, `Unprotect`, `Clear Pixel`), zoom-aware `getBoundingClientRect()` pointer strokes, 30-step `Uint8Array` Undo/Redo stack (`Ctrl+Z`/`Ctrl+Y`), distinct color-coded overlay rendering (Auto Proposal vs. Human Added Scratch/Dust vs. Rejected vs. Protected Intact vs. E2 Uncertain/Vetoed), dirty-state auto-save queue across frame changes (`goToDustFrame`), and poll-guard (`dustMaskEditorIsBusy()`).

### Files to Modify
- `ai_remaster_gui/static/index.html` — Include `/static/js/dust-mask-editor.js` before `render-dust.js` / `app.js`.
- `ai_remaster_gui/static/dust.css` — Minimal layout/toolbar/badge styling for edit mode, layer toggles, brush size, protect tint, and compatibility banner.
- `ai_remaster_gui/static/js/render-dust.js` — Mount edit controls (`Edit Mode` toggle, `Scratch / Dust / Protect / Reject` tool buttons, brush size slider + `[`/`]`, Undo/Redo, Reset Frame/All, Save Now, Export Provenance…), add `(Experimental)` E2 badge in `DUST_DETECTOR_META`, load/render edit bitmask overlay & `vetoed` layer in exact still inspection, yield A/B split drag when paint tool active, and preserve before/after repaired inspection using the composed repair output.

### Validation
- `node --check ai_remaster_gui/static/js/dust-mask-editor.js`
- `node --check ai_remaster_gui/static/js/render-dust.js`
- `node --test tests/test_dust_workspace_browser.test.cjs`

### Dependencies
- Blocked by: 2
- Blocks: 5

---

## Phase 5: End-to-End Tests, Round-Trip Verification, and Windows Guide

### Objective
Deliver comprehensive automated tests covering JS↔Python exact bitmask round-tripping, atomic failure recovery, pipeline invariance (`PROTECT_INTACT` and `changed_outside_pixels == 0` across all supported repair backends), full `.arpp` save/close/reopen + detector switching + restore E2E workflow, and clear Windows user verification instructions.

### Agent: `tester`
### Parallel: No

### Files to Create
- `tests/test_dust_mask_edits.py` — Bitmask precedence (`PROTECT_INTACT > REJECT > ADD > AUTO`, post-dilation veto), interactive stroke conflict reversal, JS↔Python `Uint8Array` ↔ PNG round-trip (invoking Node helper + Python codec), atomic ZIP crash simulation & `.meta.json` checksum fail-closed tests, timeline/dimension mismatch tests, E2 3-state separation (`high_confidence`, `uncertain`, `vetoed`), and mandatory provenance non-GT verification.
- `tests/test_dust_e2e_mask_workflow.py` — Full workflow E2E tests: production (`farneback_3f`) & opt-in `experimental_e2` detection -> HTTP load/edit/save -> close & reopen `.arpp` -> detector switch & rerun -> confirm edits & `PROTECT_INTACT` persist -> edit cache key invalidation -> repair across `telea`, `propainter` (test double), `dart` (test double), and `dust_native_composite.compose()` with `changed_outside_pixels == 0` and 0 modified pixels inside `PROTECT_INTACT`, originals unchanged, and optional provenance export check.
- `docs/dust-mask-editing-guide.md` — Clear Windows guide (`launch_gui.bat`) covering project open, detection (`farneback_3f` / opt-in `experimental_e2`), editing scratch/dust/protect/reject masks, saving/reopening `.arpp`, restoring with composed masks, A/B inspection, and optional provenance export.

### Files to Modify
- `tests/test_dust_workspace_browser.test.cjs` — Extend Node `vm` contract tests to cover `dust-mask-editor.js` bitmask round-trip encoding, deterministic brush precedence/reversal, undo/redo, and E2 opt-in UI metadata.

### Validation
- `.\.venv\Scripts\python.exe -m pytest tests/test_dust_mask_edits.py tests/test_dust_e2e_mask_workflow.py tests/test_http_routes.py tests/test_http_security.py tests/test_dust_stage.py tests/test_dust_stage_data_api.py tests/test_end_to_end_repair_phase9.py tests/test_gui_smoke.py tests/test_experimental_scene_film_prototype.py -v`
- `node --test tests/test_dust_workspace_browser.test.cjs`

### Dependencies
- Blocked by: 3, 4
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `ai_remaster_gui/dust_mask_edits.py` | 1 | Sparse bitmask delta ZIP storage, precedence, provenance, export |
| 2 | `ai_remaster_gui/project_io.py` | 1 | `.arpp` bundling allowlist & asset collection for `manifests/dust_masks/` |
| 3 | `ai_remaster_gui/cache.py` | 1 | Preserve project mask manifests from cache sweeps |
| 4 | `ai_remaster_gui/dust_e2_adapter.py` | 2 | Isolated opt-in E2 3-state detector adapter |
| 5 | `ai_remaster_gui/dust.py` | 2 | Composed mask folder derivation, `edit_signature` keys, repair & provenance |
| 6 | `ai_remaster_gui/models.py` | 2 | Register opt-in `experimental_e2` detector choice in Dust UI model |
| 7 | `ai_remaster_gui/frame_inspector.py` | 2 | Inspect composed layers & optional `vetoed` mask PNG |
| 8 | `ai_remaster_gui/http_handler.py` | 3 | Loopback REST routes for edit load/save/reset/export |
| 9 | `ai_remaster_gui/server.py` | 3 | App state hydration, edit adoption binding, `.arpp` reload wiring |
| 10 | `tests/test_http_routes.py` | 3 | Update expected GET/POST route assertions |
| 11 | `ai_remaster_gui/static/js/dust-mask-editor.js` | 4 | Bitmask brush editor, undo/redo, lossless base64 codec, save queue |
| 12 | `ai_remaster_gui/static/js/render-dust.js` | 4 | Toolbar/controls integration, E2 badge, canvas paint & A/B compare |
| 13 | `ai_remaster_gui/static/dust.css` | 4 | Styling for edit toolbar, layer indicators, protect banner |
| 14 | `ai_remaster_gui/static/index.html` | 4 | Load `/static/js/dust-mask-editor.js` |
| 15 | `tests/test_dust_mask_edits.py` | 5 | Unit/property tests for bitmask algebra, ZIP safety, JS↔Py round-trip |
| 16 | `tests/test_dust_e2e_mask_workflow.py` | 5 | Full workflow E2E tests across detectors, reloads, & all repair backends |
| 17 | `tests/test_dust_workspace_browser.test.cjs` | 5 | Frontend JS unit/VM tests for bitmask editor & UI state |
| 18 | `docs/dust-mask-editing-guide.md` | 5 | Step-by-step Windows testing & usage guide |

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | MEDIUM | Core binary bitmask schema, atomic file replacement, and `.arpp` portability |
| 2 | MEDIUM | Touches `dust.py` cache key invariants, multi-backend repair gating, & E2 isolation |
| 3 | LOW | Additive HTTP endpoints + `APP.state('dust')` extensions following existing patterns |
| 4 | MEDIUM | Canvas pointer math under `review-zoom.js` & safe coexistence with 4s UI refresh |
| 5 | LOW | Isolated tests in temp directories + user guide |

## Execution Profile

```
Execution Profile:
- Total phases: 5
- Parallelizable phases: 2 (Phases 3 and 4 in Stage 3)
- Sequential-only phases: 3 (Phases 1, 2, 5)
- Estimated parallel wall time: 4 sequential batches
- Estimated sequential wall time: 5 serial phases

Note: Native subagents currently run without user approval gates.
All tool calls are auto-approved without user confirmation.
```
