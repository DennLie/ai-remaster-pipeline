---
title: "Phase 4.5 End-to-End Integration & Wiring Polish Design Document"
session_id: "2026-11-23-phase4-5-integration-wiring"
created: "2026-11-23"
status: "draft"
task_complexity: "medium"
---

# Phase 4.5 End-to-End Integration & Wiring Polish Design Document

## Context & Problem Statement
Phases 1 through 4 introduced:
- Commercial Model Guard & policy enforcement with DDColor still reference generator (`scripts/ddcolor_reference.py`)
- Color Bible SQLite DB, character/variant entities, shot palette assignments, and Color Bible GUI tab (`ai_remaster_gui/color_bible_*.py`)
- Semantic cross-shot matching with review gating (`ai_remaster_gui/semantic_matching.py`)
- RLE mask compression and SAM 2.1 video segmentation tracker (`ai_remaster_gui/rle_masks.py`, `scripts/sam2_video_tracker.py`, `scripts/osi_masked_lab.py`)

While each subsystem is thoroughly covered by isolated unit tests (720 passing tests), several critical integration seams between GUI, HTTP server, CLI dispatch, and downstream execution remain disconnected or partially wired:
1. **Seam 1**: Real `piddnad/DDColor` checkpoints fail with raw architecture mismatches unless the optional package is detected or clean model conversion is supported; GUI toggling of the Commercial Model Guard does not prompt confirmation or synchronize `non_commercial_benchmark_override`; and `server.py` lacks a reliable multi-stage provenance completion hook.
2. **Seam 2**: GUI References dispatch in `server.py` does not forward Phase 3 CLI arguments (`--reference-runner`, `--guided-backend`, `--color-bible-db`, `--film-id`, `--confidence-threshold`), and the References tab does not display the `needs_review` badge, review reasons, or entity badges with a direct jump to the Color Bible tab.
3. **Seam 3**: `colorize_video.py` does not accept or propagate `--color-bible-db` or `--film-id` to `osi_masked_lab.py`, preventing automatic lookup of approved LAB anchor medians from `manifests/color_bible.db`; furthermore, SAM 2.1 tracking lacks an authenticated API route in `http_handler.py` / `server.py`.
4. **Seam 4**: No full end-to-end CPU smoke test exercises the continuous pipeline chain from Color Bible DB -> semantic matching / reference dispatch -> RLE `.masks.npz` archive -> `osi_masked_lab` colorization with DB anchors -> cumulative `compliance_report.json`.

---

## Architectural Seam Specifications

### 1. Seam 1: Real `piddnad/DDColor` Weight Loading & Commercial Guard Toggle

#### 1.1 `scripts/ddcolor_reference.py` Model Loading
- **Detection & Dispatch**: In `load_model(model_path, device)`:
  - If the checkpoint is an `nn.Module` (TorchScript / unpickled model), use directly.
  - If `checkpoint` is a `state_dict`:
    - First attempt loading into `DDColorModel()`.
    - If zero keys match, inspect state dict keys. If keys start with `encoder.backbone`, `color_decoder`, or `pixel_decoder` (hallmarks of the official ICCV 2023 `piddnad/DDColor` dual-decoder architecture):
      - Attempt `from ddcolor import DDColor` or `from ddcolor.builder import build_model`.
      - If the `ddcolor` package is not installed in the environment, raise a clear, actionable `RuntimeError`:
        `"Checkpoint at '{model_path}' is an official piddnad/DDColor model, which requires the optional 'ddcolor' package. Install it via 'pip install ddcolor' (or clone https://github.com/piddnad/DDColor), or supply a compatible DDColorModel state dict. Pass --dry-run for synthetic mock mode."`
    - Outside of `--dry-run`, never produce fake or synthetic colors; fail closed.

#### 1.2 GUI Commercial Model Guard Toggle
- **UI Confirmation**: In `ai_remaster_gui/static/js/render-overview.js` and `actions.js`:
  - When the user unchecks `globalCommercialModelGuard`, intercept with a confirmation dialog:
    `"Disabling the Commercial Model Guard permits non-commercial, academic, and unaudited AI models (e.g., Qwen-Image-2.1, ColorMNet). Outputs generated in this mode cannot be certified commercial-safe under Apache-2.0 / MIT terms. Proceed with benchmark override?"`
  - If cancelled, revert the checkbox to checked.
  - If confirmed, send `commercial_model_guard="false"` AND `non_commercial_benchmark_override="true"`.
  - When toggled back ON (checked), automatically set `commercial_model_guard="true"` AND `non_commercial_benchmark_override="false"`.
- **Server State Handling**: In `ai_remaster_gui/server.py` (`update_settings`), ensure that setting `commercial_model_guard` synchronizes `non_commercial_benchmark_override` to preserve consistency across restarts.

#### 1.3 Stage Completion Provenance Hook
- In `server.py`:
  - When a stage subprocess completes with `code == 0` in `_collect_output`:
    - Invoke `_record_stage_compliance(stage_key)`.
    - For active pipeline stages (e.g. `references`, `colour`), record a compliance run record into `output/compliance_report.json` via `ai_remaster_gui.compliance.append_compliance_record`, evaluating guard mode and resolving Color Bible snapshot hash if present.

---

### 2. Seam 2: Phase 3 References CLI & GUI Wiring

#### 2.1 Server Dispatch (`ai_remaster_gui/server.py` & `models.py`)
- In `ai_remaster_gui/models.py`, register the Phase 3 fields on the `references` Stage:
  - `reference_runner` (`select:auto|guided_only|ddcolor_only`, default `"auto"`)
  - `guided_backend` (`select:qwen2511|qwen21|openai`, default `"qwen2511"`)
  - `color_bible_db` (`file`, default `""`)
  - `film_id` (`text`, default `""`)
  - `confidence_threshold` (`number`, default `"0.8"`)
- In `_references_command(config, values)` in `server.py`:
  - Resolve `reference_runner`, `guided_backend`, `color_bible_db`, `film_id`, and `confidence_threshold` from `values` or fall back to `global` settings.
  - Forward these arguments to `ddcolor_reference.py`, `qwen_colorize_references.py`, and `qwen21_generate_reference.py`.

#### 2.2 References UI Badges & Navigation (`render-shots.js`)
- In `referenceCard(context)`:
  - Check `row.needs_review`:
    - If `true`, render `<span class="badge badge-warn">Needs Review</span>`.
    - Render `row.review_reasons` as an alert list.
  - Check `row.color_bible_entities`:
    - Render chips for assigned entities showing name, variant/costume, and confidence score.
  - Add button: `<button type="button" class="compact-btn" onclick="openColorBibleForShot('${esc(film_id)}', ${idx})">Review in Color Bible &rarr;</button>`.
  - In `actions.js`, implement `openColorBibleForShot(filmId, shotIndex)` to switch `active = 'color_bible'`, highlight the relevant shot/character, and re-render.

---

### 3. Seam 3: Phase 4 End-to-End Wiring (`color_bible.db` + SAM 2.1 -> `osi_masked_lab`)

#### 3.1 `colorize_video.py` Parameter Passing & Anchor Resolution
- Add CLI arguments in `build_parser()`:
  - `--color-bible-db`: path to Color Bible SQLite DB (defaults to `ROOT / "manifests" / "color_bible.db"`).
  - `--film-id`: target film ID.
- In `run_osi_masked_lab_colorization`:
  - For each shot row, resolve `anchors`:
    - If `row.get("color_bible_anchors")` is present, parse it.
    - If not, and `color_bible_db` exists and `film_id` is provided (via CLI or manifest comments `# color_bible_film_id=` / `# color_bible=`):
      - Query `ColorBibleService(db_path=args.color_bible_db).get_shot_palette_assignments(film_id, index)`.
      - For each assigned entity with approved references, parse `lab_stats` / `lab_stats_json` and extract median chromaticity `(a*, b*)`.
      - Map into `anchors` by `entity_id`, `variant_id`, `name`, and `obj_id`.
  - If no mask archive or anchors exist, cleanly perform unmasked global optical flow propagation.

#### 3.2 `scripts/osi_masked_lab.py` Percentile Anchor Support
- In `_resolve_color_anchor(anchors, obj)`:
  - If anchor entry has `{"L": [...], "a": [...], "b": [...]}`, extract the 50th percentile (index `len(arr) // 2`).

#### 3.3 Server Dispatch & SAM 2.1 Route
- In `_colour_command` in `server.py`:
  - Forward `--color-bible-db`, `--film-id`, `--masks-dir`, `--guided-radius`, `--guided-eps`, and `--compliance-output-dir` when `method == "osi_masked_lab"`.
- In `ai_remaster_gui/http_handler.py`:
  - Add route `POST /api/sam2-track-shot`:
    - Check `verify_sam2_installation()` and `evaluate_component("sam2", settings=..., component_type="tool")`.
    - Accept `shot_id`, `frame_paths` or `video_path`, `prompts`, and `output_npz_path`.
    - Run `run_shot_tracking_pipeline(...)` from `ai_remaster_gui.sam2_tracker`.
  - Add `/api/sam2-track-shot` to `EXPECTED_POST_ROUTES` in `tests/test_http_routes.py`.

---

### 4. Seam 4: End-to-End CPU Integration Smoke Test

#### 4.1 Test Suite (`tests/test_e2e_commercial_pipeline_wiring.py`)
- Create synthetic 2-shot pipeline test:
  1. Initialize SQLite Color Bible with Film `"e2e_test_film"`, Character `"Alice"`, Variant `"Costume_Red"` (approved reference with LAB median `a=35.0, b=25.0`).
  2. Assign `"Costume_Red"` to Shot 0 and Shot 1.
  3. Run `semantic_matching.reference_images_for` to verify entity resolution, distant matching, and review gating.
  4. Generate a synthetic `.masks.npz` archive via `rle_masks.save_shot_masks_npz` with metadata pointing to `"alice"` / `"costume_red"`.
  5. Run `colorize_shot_osi_masked_lab` or `run_osi_masked_lab_colorization` on CPU:
     - Verify it loads LAB anchors directly from SQLite DB.
     - Verify colorization applies expected chroma to masked foreground pixels and propagates across frames.
  6. Verify `append_compliance_record`:
     - Assert `overall_commercial_compliant: True`.
     - Assert `color_bible_version` is present and matches the DB snapshot hash.
     - Assert `eu_ai_act_article_50_disclosure` block is present and conforms to Article 50(4).

---

## Validation Strategy
1. Run targeted pytest suite on CPU:
   `python -m pytest tests/test_e2e_commercial_pipeline_wiring.py tests/test_rle_masks.py tests/test_sam2_video_tracker.py tests/test_osi_masked_lab.py tests/test_frame_masks.py tests/test_temporal_consistency.py tests/test_reference_luminance.py tests/test_ltx_colorize.py tests/test_video_order.py tests/test_cross_shot_matching.py tests/test_semantic_matching.py tests/test_color_bible_api.py tests/test_color_bible_cli.py tests/test_color_bible_db.py tests/test_color_bible_service.py tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_ddcolor_reference.py tests/test_http_routes.py tests/test_http_security.py tests/test_gui_smoke.py tests/test_pipeline_order.py -q`
2. Ensure 100% pass rate with zero skips (or only existing environmental skips) and zero regressions.
