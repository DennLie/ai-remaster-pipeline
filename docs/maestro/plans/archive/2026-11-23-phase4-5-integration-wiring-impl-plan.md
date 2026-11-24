---
title: "Phase 4.5 Integration & Wiring Polish Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-23-phase4-5-integration-wiring-design.md"
created: "2026-11-23"
status: "draft"
total_phases: 4
estimated_files: 9
task_complexity: "medium"
---

# Phase 4.5 Integration & Wiring Polish Implementation Plan

## Plan Overview
- **Total phases**: 4
- **Agents involved**: `coder`, `tester`
- **Estimated effort**: Focused, high-precision integration pass closing the 4 seams across GUI, HTTP server, CLI dispatch, and downstream execution engines.
- **Execution mode**: Sequential (High Precision).

## Dependency Graph
```
[Phase 1: Seam 1 (DDColor Weight Loader & Guard Toggle)]
                     │
                     ▼
[Phase 2: Seam 2 (Phase 3 References CLI & GUI Wiring)]
                     │
                     ▼
[Phase 3: Seam 3 (Phase 4 Color Bible DB + SAM2 -> osi_masked_lab)]
                     │
                     ▼
[Phase 4: Seam 4 (End-to-End CPU Integration Smoke Test)]
```

## Execution Strategy
| Stage | Phase | Execution | Agent | Purpose |
|---|---|---|---|---|
| 1 | Phase 1 | Sequential | `coder` | Real DDColor weights loading, Commercial Guard GUI toggle with confirmation, and server completion provenance hook |
| 2 | Phase 2 | Sequential | `coder` | Wire Phase 3 options into `models.py`, `server.py`, and References GUI (badges, reasons, Color Bible jump) |
| 3 | Phase 3 | Sequential | `coder` | Wire `color_bible.db` + SAM 2.1 tracking into `colorize_video.py`, `osi_masked_lab.py`, and `http_handler.py` |
| 4 | Phase 4 | Sequential | `tester` | Comprehensive end-to-end CPU smoke test in `tests/test_e2e_commercial_pipeline_wiring.py` + full test suite pass |

---

## Phase 1: Seam 1 — Real DDColor Weight Loading & Commercial Guard Toggle

### Objective
Provide clean real DDColor architecture / loader support in `scripts/ddcolor_reference.py` that handles real `piddnad/DDColor` checkpoints and fails with an actionable message if `ddcolor` is missing (never faking colors outside `--dry-run`); add a clear visual confirmation warning when toggling the Commercial Model Guard OFF in the browser while syncing `non_commercial_benchmark_override="true"`; and add a server completion hook to record stage compliance in `output/compliance_report.json`.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `scripts/ddcolor_reference.py`
  - In `load_model`: Detect official `piddnad/DDColor` checkpoint keys (`encoder.backbone`, `color_decoder`, `pixel_decoder`).
  - Attempt importing `ddcolor` builder/model. If missing, raise `RuntimeError` advising installation (`pip install ddcolor` or clone) or using `--dry-run`.
  - Maintain support for `DDColorModel()` state dicts and TorchScript models.
- `ai_remaster_gui/static/js/render-overview.js`
  - On `globalCommercialModelGuard` click/change, if unchecking, show confirmation warning modal/prompt explaining that non-commercial/unaudited models will be permitted and output cannot be certified commercial-safe.
  - Revert if cancelled.
- `ai_remaster_gui/static/js/actions.js`
  - In `saveGlobalPipelineOptions`: When `commercial_model_guard` is false, send `non_commercial_benchmark_override="true"`. When true, send `non_commercial_benchmark_override="false"`.
- `ai_remaster_gui/server.py`
  - In `update_settings`: Synchronize `non_commercial_benchmark_override` when `commercial_model_guard` is updated.
  - In `_collect_output`: Call `_record_stage_compliance(stage_key)` on exit code 0 to record provenance in `output/compliance_report.json`.

### Validation
- Run `python -m pytest tests/test_ddcolor_reference.py tests/test_compliance_report.py tests/test_commercial_policy.py -q`.
- Verify `scripts/ddcolor_reference.py` fails with actionable error on missing weights/package and never fakes colors outside `--dry-run`.

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: Seam 2 — Phase 3 References CLI & GUI Wiring

### Objective
Wire Phase 3's new options (`--reference-runner=auto|guided_only|ddcolor_only`, `--guided-backend`, `--color-bible-db`, `--film-id`, `--confidence-threshold`) into `models.py`, `server.py`, and References GUI tab; display `needs_review` badge, `review_reasons`, and assigned `color_bible_entities` in References shot cards with a direct jump to the Color Bible tab.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `ai_remaster_gui/models.py`
  - Add fields to `references` Stage: `reference_runner`, `guided_backend`, `color_bible_db`, `film_id`, `confidence_threshold`.
- `ai_remaster_gui/server.py`
  - In `_references_command`: Read `reference_runner`, `guided_backend`, `color_bible_db`, `film_id`, `confidence_threshold` from `values` or `global` settings and pass them to `ddcolor_reference.py` and `qwen_colorize_references.py`.
- `ai_remaster_gui/static/js/render-shots.js`
  - In `referenceCard`:
    - If `row.needs_review` is true, render `<span class="badge badge-warn">Needs Review</span>` and list `row.review_reasons`.
    - Render assigned `row.color_bible_entities` chips with confidence score and status.
    - Add button/link: `Review in Color Bible &rarr;` calling `openColorBibleForShot`.
- `ai_remaster_gui/static/js/actions.js`
  - Add `openColorBibleForShot(filmId, shotIndex)` to switch `active = 'color_bible'`, highlight the relevant shot, and re-render.

### Validation
- Run `python -m pytest tests/test_cross_shot_matching.py tests/test_semantic_matching.py tests/test_gui_smoke.py -q`.

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 3

---

## Phase 3: Seam 3 — Phase 4 End-to-End Wiring (`color_bible.db` + SAM 2.1 Tracking -> `osi_masked_lab`)

### Objective
Enable `--color-bible-db` and `--film-id` to be passed end-to-end from `server.py` -> `colorize_video.py` -> `osi_masked_lab.py`, loading approved LAB anchor statistics from SQLite DB; support percentile-dict LAB anchors in `_resolve_color_anchor`; and expose `POST /api/sam2-track-shot` in `http_handler.py` gated by installation and commercial policy.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `scripts/colorize_video.py`
  - Add `--color-bible-db` and `--film-id` arguments to parser.
  - In `run_osi_masked_lab_colorization`: For each shot, if `row.get("color_bible_anchors")` is not present, query `ColorBibleService` for shot assignments and extract median LAB anchors from approved references.
  - Maintain clean unmasked fallback when no mask archive or anchors exist.
- `scripts/osi_masked_lab.py`
  - In `_resolve_color_anchor`: Support dicts with list values `{"L": [...], "a": [...], "b": [...]}` by taking the median (50th percentile).
- `ai_remaster_gui/server.py`
  - In `_colour_command`: Pass `--color-bible-db`, `--film-id`, `--masks-dir`, `--guided-radius`, `--guided-eps`, and `--compliance-output-dir` when `method == "osi_masked_lab"`.
- `ai_remaster_gui/http_handler.py`
  - Implement route `POST /api/sam2-track-shot`:
    - Gate by `verify_sam2_installation()` and `evaluate_component("sam2")`.
    - Run `run_shot_tracking_pipeline(...)`.
- `tests/test_http_routes.py`
  - Add `"/api/sam2-track-shot"` to `EXPECTED_POST_ROUTES`.

### Validation
- Run `python -m pytest tests/test_osi_masked_lab.py tests/test_sam2_video_tracker.py tests/test_http_routes.py tests/test_http_security.py -q`.

### Dependencies
- Blocked by: Phase 2
- Blocks: Phase 4

---

## Phase 4: Seam 4 — End-to-End CPU Integration Smoke Test & Full Validation

### Objective
Author `tests/test_e2e_commercial_pipeline_wiring.py` exercising the full Phase 1-4 pipeline chain on CPU with synthetic 2-shot test data: Color Bible DB -> semantic cross-shot matching & reference dispatch -> RLE `.masks.npz` archive -> `osi_masked_lab` (loading DB anchors) -> cumulative `compliance_report.json` (`overall_commercial_compliant: true`, `color_bible_version` populated, EU AI Act Art. 50 disclosure present).

### Agent: `tester`
### Parallel: No

### Files to Create
- `tests/test_e2e_commercial_pipeline_wiring.py`
  - Synthetic 2-shot pipeline test with SQLite Color Bible, character with approved reference, RLE mask archive, and `osi_masked_lab` colorization.
  - Verify all 4 seams execute seamlessly and produce expected output frames, anchors, and compliance report.

### Validation
- Run acceptance test command:
  `python -m pytest tests/test_e2e_commercial_pipeline_wiring.py tests/test_rle_masks.py tests/test_sam2_video_tracker.py tests/test_osi_masked_lab.py tests/test_frame_masks.py tests/test_temporal_consistency.py tests/test_reference_luminance.py tests/test_ltx_colorize.py tests/test_video_order.py tests/test_cross_shot_matching.py tests/test_semantic_matching.py tests/test_color_bible_api.py tests/test_color_bible_cli.py tests/test_color_bible_db.py tests/test_color_bible_service.py tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_ddcolor_reference.py tests/test_http_routes.py tests/test_http_security.py tests/test_gui_smoke.py tests/test_pipeline_order.py -q`
- 100% pass on CPU without GPU or network access.

### Dependencies
- Blocked by: Phase 3
- Blocks: None

---

## File Inventory
| # | File | Phase | Purpose |
|---|---|---|---|
| 1 | `scripts/ddcolor_reference.py` | 1 | Real DDColor weight detection, package error handling, fail-closed inference |
| 2 | `ai_remaster_gui/static/js/render-overview.js` | 1 | Confirmation warning modal when toggling Commercial Model Guard OFF |
| 3 | `ai_remaster_gui/static/js/actions.js` | 1, 2 | Synchronize benchmark override and provide `openColorBibleForShot` navigation |
| 4 | `ai_remaster_gui/server.py` | 1, 2, 3 | Stage completion compliance hook, Phase 3 CLI args in references, and Phase 4 args in colour |
| 5 | `ai_remaster_gui/models.py` | 2 | Register Phase 3 options on `references` Stage |
| 6 | `ai_remaster_gui/static/js/render-shots.js` | 2 | Render `needs_review` badge, review reasons, entity chips, and jump link |
| 7 | `scripts/colorize_video.py` | 3 | Add `--color-bible-db` & `--film-id` and resolve DB anchors per shot |
| 8 | `scripts/osi_masked_lab.py` | 3 | Handle percentile-dict LAB anchors in `_resolve_color_anchor` |
| 9 | `ai_remaster_gui/http_handler.py` | 3 | Add `POST /api/sam2-track-shot` route with policy checks |
| 10 | `tests/test_http_routes.py` | 3 | Keep `EXPECTED_POST_ROUTES` synchronized |
| 11 | `tests/test_e2e_commercial_pipeline_wiring.py` | 4 | Comprehensive CPU end-to-end integration smoke test |

## Risk Classification
| Phase | Risk | Rationale |
|---|---|---|
| 1 | LOW | Changes are additive in model loader and GUI events; zero changes to mathematical core. |
| 2 | LOW | Pure wiring of existing backend options and UI badging already supported in data layer. |
| 3 | MEDIUM | Involves query integration in `colorize_video.py` and route addition in `http_handler.py`; risk mitigated by strict route tests and fallback logic. |
| 4 | LOW | Verification test only; exercises all components on synthetic CPU data. |

## Execution Profile
```
Execution Profile:
- Total phases: 4
- Parallelizable phases: 0 (sequential requested)
- Sequential-only phases: 4
- Estimated sequential wall time: ~10-15 minutes
```
