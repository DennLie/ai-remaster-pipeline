---
title: "Integrate Editable Defect Masks Into the Main ARP Workflow"
created: "2026-11-19T16:40:26+01:00"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "standard"
task_complexity: "medium"
session_id: "2026-11-19-editable-defect-masks-arp-gui"
---

# Integrate Editable Defect Masks Into the Main ARP Workflow Design Document

## Problem Statement

ARP currently treats film defect detection inside **Dust & Scratches** (`ai_remaster_gui/dust.py`, `ai_remaster_gui/static/js/render-dust.js`) as a read-only black box: once a detector produces `mask.mkv` and `effective_mask.mkv` under `intermediate/dust_scratches/<key>/`, users can preview masks, inspect stills, and adjust presets, but cannot add missed defects, suppress false positives, or lock intact scene regions against repair. Directly overwriting cached `.mkv` files would break SHA-256 integrity checks (`report.json`) and discard manual edits whenever detection reruns or detector parameters change.

Rather than building a separate browser annotation tool or continuing open-ended detector tuning, this design integrates a non-destructive, persistent human-in-the-loop mask editing workflow directly into the existing **Dust & Scratches** workbench—reusing existing canvas zoom/pan (`review-zoom.js`), sparse per-frame ZIP persistence (`scripts/frame_masks.py`), `.arpp` project bundling (`project_io.py`), and verified repair/composition (`dust.py`, `dust_native_composite.py`).

## Requirements

### Functional Requirements
1. **REQ-1 (Interactive Workbench Editing)**: Add non-destructive mask editing inside **Dust & Scratches** with explicit automatic detection, separate Scratch / Dust layers, an explicit **Protect (Intact)** brush, Add/Remove brush tools, overlay visibility/opacity controls, zoom/pan, frame navigation, undo/redo, visual separation of automatic proposals vs. manual edits vs. protected zones, and before/after restoration inspection.
2. **REQ-2 (Non-Destructive Persistent Storage)**: Persist sparse per-frame packed bitmask PNGs (`manifests/dust_masks/<timeline_id>.zip`) and versioned metadata (`<timeline_id>.meta.json`) inside `.arpp` bundles using temp-file + `os.fsync` + `os.replace` atomic replacement. Bind archives to stable source/timeline identity, frame dimensions, frame count, and archive SHA-256 (`zip_sha256` + `edit_revision` checksum verification) so incompatible or partially written edits fail closed instead of applying silently.
3. **REQ-3 (Deterministic Precedence & Pipeline Consumption)**: Resolve final masks deterministically (`PROTECT_INTACT > REJECT > ADD > AUTO`, re-enforcing `~PROTECT_INTACT` post-dilation). Generate composed repair caches separately from immutable base detector caches (`<base_key>/` untouched), include the active edit signature in composed cache identity so stale restoration outputs are never reused after edits, and feed composed `effective_mask.mkv` through existing Telea/ProPainter/DART + 10-bit `dust_native_composite.compose()`.
4. **REQ-4 (Detector Isolation & Mandatory Provenance)**: Keep existing production detectors (`farneback_3f`, `dis_flow`, `ridge`, `consensus_5f`, `legacy`, `temporal_scratch`) as defaults. Expose Experimental E2 (`ExperimentalSceneFilmDetector`) strictly as an opt-in `(Experimental)` detector option gated by `research_confirmed`, preserving its `high_confidence_damage`, `uncertain_damage_candidate`, and `likely_intact_scene_structure` (`vetoed_mask.mkv`) outputs separately without flattening uncertainty or vetoed regions into confirmed damage. Record mandatory provenance distinguishing auto predictions, manual additions, manual rejections, explicit protections, explicit review markers, and unreviewed pixels; keep dataset export optional and never promote operational edit strokes or untouched auto predictions into independent verified ground truth.

### Constraints
- Do not modify underlying production restoration algorithms (`film_defect_detectors.py` / Telea / ProPainter / `dust_native_composite.py` math) beyond an explicit adapter to consume resolved masks.
- Do not overwrite source frames or immutable `intermediate/dust_scratches/<base_key>/` caches.
- Maintain exact compatibility with existing `.arpp` v2 projects (`project_io.py`) and ARP security/route guards (`http_handler.py`, `tests/test_http_routes.py`, `tests/test_http_security.py`).

## Approach

### Selected Approach: Approach 1 — Single Packed Bitmask Delta ZIP
Human corrections are stored in a project-owned archive (`manifests/dust_masks/<timeline_id>.zip`) paired with `<timeline_id>.meta.json` — *separates human work from SHA-256-locked detector caches so edits survive reloads and detector switches*. Each edited frame is stored as one lossless single-channel grayscale PNG (`f{frame:06d}.png`) plus embedded `edit_meta.json`, packing bit flags:
- `BIT_SCRATCH_ADD = 1` (`0x01`)
- `BIT_SCRATCH_REJECT = 2` (`0x02`)
- `BIT_DUST_ADD = 4` (`0x04`)
- `BIT_DUST_REJECT = 8` (`0x08`)
- `BIT_PROTECT_INTACT = 16` (`0x10`)
- `BIT_UNCERTAIN_REJECT = 32` (`0x20`)
- `BIT_REVIEWED = 64` (`0x40`) — operational review marker only; NEVER treated alone as independent ground truth.

**Deterministic Edit Transitions & Precedence**:
- Interactive Add reverses prior Reject on that class (`Add Scratch` sets `BIT_SCRATCH_ADD` and clears `BIT_SCRATCH_REJECT`; `Reject Scratch` sets `BIT_SCRATCH_REJECT` and clears `BIT_SCRATCH_ADD`; `Reject All` clears both `ADD` bits and sets `SCRATCH_REJECT | DUST_REJECT | UNCERTAIN_REJECT`). `PROTECT_INTACT` is set by `Protect` and cleared only by explicit `Unprotect` / `Clear Frame`.
- Resolution precedence for every pixel: `PROTECT_INTACT` (vetoes all channels + post-dilation repair) > explicit `REJECT` > human `ADD` > active automatic proposal.

### Alternatives Rejected
- **Approach 2: Separate Binary ZIPs per Channel**: Avoids bit packing, but triples `.arpp` archive coordination and weakens atomic cross-channel undo.
- **Approach 3: Mutable Copied `.mkv` Cache**: Breaks compact `.arpp` portability (`bundle_name_allowed` blocks `.mkv`), orphans edits across detector switches, and loses proposal vs. human provenance.

### Decision Matrix

| Criterion | Weight | Approach 1 (Bitmask ZIP) | Approach 2 (Multi-Zip) | Approach 3 (Mutable MKV) |
| --- | --- | --- | --- | --- |
| Non-Destructive Safety & Persistence | 30% | 5: Immutable caches, atomic ZIP, survives reruns | 4: Immutable caches, multi-zip sync surface | 2: Coupled to one cache key; fragile on rerun |
| Reuse & MVP Simplicity | 25% | 4: Clean 1-PNG/frame model; small bit-pack helper | 4: Binary PNGs, 4–5 archives to coordinate | 3: Simple repair read, complex `.arpp` exclusion |
| Pipeline Compatibility & User Control | 25% | 5: Edit-bound cache keys + protect override | 4: Works at repair time, heavier inspect | 2: Breaks SHA-256 report invariants if mutated |
| Provenance Fidelity | 20% | 5: Exact state separation without GT conflation | 3: Weak FP-rejection vs unreviewed tracking | 1: Overwrites auto vs human history |
| **Weighted Total** | **100%** | **4.75** | **3.80** | **2.05** |

## Architecture

### Component Diagram
```
[Browser: render-dust.js + dust-mask-editor.js + review-zoom.js]
   |  (lossless base64 Uint8Array bitmask + /api/dust-inspect + /api/dust-run)
   v
[GUI Backend: http_handler.py + server.py (PipelineApp)]
   |--> [dust_mask_edits.py + project_io.py] <--> manifests/dust_masks/<timeline_id>.zip + .meta.json + .arpp
   |--> [dust.py + dust_e2_adapter.py + frame_inspector.py]
           |--> Immutable base detector cache: intermediate/dust_scratches/<base_key>/ (incl. vetoed_mask.mkv for E2)
           |--> Separate composed repair cache: intermediate/dust_scratches/<composed_key>/ (keyed by edit_signature)
                   `--> Telea / ProPainter / DART + dust_native_composite.compose() -> repaired.mkv + restoration_provenance.json
```

### Key Data Contracts & Separation
1. **Bitmask Transport & Storage (`ai_remaster_gui/dust_mask_edits.py`)**: HTTP routes exchange exact raw base64 `Uint8Array` bytes for edited frames, encoded/decoded server-side as lossless single-channel PNG (`f{frame:06d}.png`) inside `manifests/dust_masks/<timeline_id>.zip` so HTML5 Canvas premultiplied alpha never mutates bit flags.
2. **Checksum & Timeline Integrity**: External `<timeline_id>.meta.json` stores `edit_schema_version`, `timeline_identity` (`source_path`, `analysis_source`, `source_sha256`/fingerprint, `width`, `height`, `frame_count`), `zip_sha256`, and `edit_revision`, mirrored inside the ZIP's `edit_meta.json`. Loading or repairing fails closed on checksum or timeline mismatch.
3. **Cache Key Invalidation (`ai_remaster_gui/dust.py`)**: Base detector caches `<base_key>/` stay untouched. Active edits bind `edit_signature = "<zip_sha256>:<edit_revision>"` into `dust.options()` / `dust.identity()`, writing composed masks and repairs to a distinct `<composed_key>/` (reusing `<base_key>/` raw detector `.mkv` streams without re-running detection) and verifying `changed_outside_pixels == 0` including `PROTECT_INTACT`.
4. **Provenance & Optional Export**: `restoration_provenance.json` and optional `export_provenance_bundle()` record separate layers (`auto_proposals` including E2's unflattened 3 states, `human_edits`, `protect_intact`, `effective_repair`, `edit_signature`) and label `ground_truth_status: "operational_edits_only_not_independent_gt"`.

## Agent Team

| Phase | Agent | Parallel | Deliverables |
| --- | --- | --- | --- |
| 1 | `coder` | No | `dust_mask_edits.py`, `project_io.py`, `cache.py` atomic ZIP storage & `.arpp` bundling |
| 2 | `coder` | No | `dust_e2_adapter.py`, `dust.py`, `models.py`, `frame_inspector.py` composed mask & E2 opt-in |
| 3 | `coder` | Yes | `http_handler.py`, `server.py`, `test_http_routes.py` API routes & project hydration |
| 4 | `coder` | Yes | `dust-mask-editor.js`, `render-dust.js`, `dust.css`, `index.html` interactive canvas UI |
| 5 | `tester` | No | `test_dust_mask_edits.py`, `test_dust_e2_mask_workflow.py`, JS tests, `docs/dust-mask-editing-guide.md` |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
| --- | --- | --- | --- |
| Partial or mismatched ZIP vs `.meta.json` | HIGH | LOW | Same-dir temp file + `fsync` + `os.replace`, `zip_sha256` & `edit_revision` binding, embedded `edit_meta.json`, fail closed on mismatch |
| Stale repair reuse after manual edits | HIGH | MEDIUM | Hash `edit_signature` into `dust.identity()` and `validate_dust_adoption()` so composed repairs always rebuild cleanly |
| Source / crop / stabilize drift misaligning frames | HIGH | MEDIUM | Bind archive to `(source_sha256, analysis_source, W, H, frames, schema_version)`; block silent mismatch application |
| JS Canvas alpha premultiplication corrupting bits | MEDIUM | HIGH | Exchange raw `Uint8Array` base64 buffers; server encodes/decodes `L`-mode PNG; verify with JS↔Python round-trip test |
| 4s `/api/state` poll or A/B drag interrupting paint | MEDIUM | MEDIUM | Guard DOM rebuild while stroke/dirty pending and yield A/B drag handler when brush tool active |
| Conflating E2 uncertainty or operational strokes with GT | HIGH | LOW | `PROTECT_INTACT` veto, separate `vetoed_mask.mkv`/`uncertain_mask.mkv`, explicit `operational_edits_only` provenance |

## Success Criteria

1. Exact JS `Uint8Array` ↔ Python PNG bitmask round-trip preserves all bit flags (`0..127`) without corruption.
2. Interrupted ZIP saves leave previous archive intact; checksum or timeline mismatches fail closed.
3. Final restoration across `telea`, `propainter` (test double), `dart` (test double), and `dust_native_composite.compose()` writes zero pixels outside the composed `effective_mask` and zero pixels inside `PROTECT_INTACT`.
4. Full workflow round-trip passes automated tests: detect (production + opt-in E2) -> edit scratch/dust/reject/protect -> save `.arpp` -> close -> reopen `.arpp` -> switch/rerun detector -> confirm edits & protect persist -> restore using composed mask -> optional provenance export with originals untouched.
