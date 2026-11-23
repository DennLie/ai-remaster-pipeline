---
title: "Phase 3: Semantic Reference Matching Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-22-phase3-semantic-reference-matching-design.md"
created: "2026-11-22T17:35:00+01:00"
status: "draft"
total_phases: 3
estimated_files: 6
task_complexity: "complex"
---

# Phase 3: Semantic Reference Matching Implementation Plan

## Plan Overview

- **Total phases**: 3
- **Agents involved**: `coder`, `tester`, `code-reviewer`
- **Estimated effort**: High Precision Sequential Implementation & Comprehensive Verification
- **Target Deliverables**:
  1. `ai_remaster_gui/semantic_matching.py` (New module)
  2. `scripts/qwen_colorize_references.py` (Upgrade)
  3. `scripts/ddcolor_reference.py` (Upgrade)
  4. `ai_remaster_gui/references.py` & `ai_remaster_gui/http_handler.py` (Upgrade)
  5. `tests/test_cross_shot_matching.py` (New comprehensive test suite)

## Dependency Graph

```
[Phase 1: Foundation & Semantic Engine] (coder)
       |
       v
[Phase 2: Pipeline Integration & CLI/GUI] (coder)
       |
       v
[Phase 3: Comprehensive Cross-Shot Test Suite] (tester)
       |
       v
[Completion: Code Review Gate] (code-reviewer)
```

## Execution Strategy

| Stage | Phase | Execution | Agent | Notes |
|---|---|---|---|---|
| 1 | Phase 1 | Sequential | `coder` | Semantic Matching Subsystem, 3-Tier Retrieval, Review Gate, Registry, Prompt Enricher, Scoped Cache |
| 2 | Phase 2 | Sequential | `coder` | Wire into `qwen_colorize_references.py`, `ddcolor_reference.py`, `references.py`, `http_handler.py` |
| 3 | Phase 3 | Sequential | `tester` | 100% CPU pytest suite covering acceptance criteria (a)-(g) |

---

## Phase 1: Foundation and Semantic Engine

### Objective
Create `ai_remaster_gui/semantic_matching.py` implementing the core 3-tier retrieval logic, Review Gate (Priority C), pluggable `GuidedReferenceBackend` protocol/registry, `PromptEnricher`, two-stage reference router, scoped cache key generator, and compliance auditing.

### Agent: `coder`
### Parallel: No

### Files to Create
- `ai_remaster_gui/semantic_matching.py` — Central domain module for Phase 3 containing:
  - `ColorBibleMatcher`: 3-tier resolver with path deduplication, fallback, and review gate short-circuiting.
  - `PromptEnricher`: Formatter for character names, costume IDs, lighting, RGB swatches, and LAB statistics.
  - `GuidedReferenceBackend` protocol & `ReferenceBackendRegistry`: Registry holding `Qwen2511Backend` and future backends.
  - `TwoStageReferenceRouter`: Automatic escalation router respecting `--reference-runner` and per-row `reference_backend` overrides.
  - `compute_scoped_cache_key(...)` & `build_reference_signature(...)`: Scoped cache key hashing only shot-specific `ref_id`s, versions, and prompt parameters.
  - `record_reference_compliance(...)`: Audits commercial policy and logs provenance record to `output/compliance_report.json`.

### Files to Modify
None in this phase.

### Implementation Details
- Deduplicate normalized file paths between Priority A and Priority B.
- Priority C check (`pending_review` or `confidence < 0.8`) short-circuits to `needs_review` BEFORE cache-hit check.
- Keep film-wide `color_bible_version` as provenance metadata outside the cache-hit equality key.
- Strict type annotations, dataclasses, and zero external network/GPU requirements.

### Validation
- Run isolated import check:
  `python -c "import ai_remaster_gui.semantic_matching as sm; print(sm.__name__)"`

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: Pipeline Integration and CLI/GUI

### Objective
Integrate `semantic_matching` into `scripts/qwen_colorize_references.py`, `scripts/ddcolor_reference.py`, `ai_remaster_gui/references.py`, and `ai_remaster_gui/http_handler.py`.

### Agent: `coder`
### Parallel: No

### Files to Modify
- `scripts/qwen_colorize_references.py`:
  - Update `reference_images_for()` to delegate to `ColorBibleMatcher.resolve_references()`.
  - Add CLI arguments: `--reference-runner`, `--guided-backend`, `--color-bible-db`, `--film-id`, `--confidence-threshold`.
  - Reorder execution flow: resolve metadata -> evaluate Review Gate -> select backend -> enrich prompt -> check scoped cache -> check commercial policy -> run backend -> write signature & compliance.
  - When `needs_review` is detected, mark manifest and Color Bible record and cleanly skip inference.
- `scripts/ddcolor_reference.py`:
  - Expose helper `colorize_reference_shot(...)` for programmatic two-stage execution by the router.
- `ai_remaster_gui/references.py`:
  - Update `shot_rows()` to expose `needs_review` flag, review reasons, and assigned entity summaries.
- `ai_remaster_gui/http_handler.py`:
  - Ensure `/api/color-bible/shot-assignments` and related endpoints expose approval and review status for shots.

### Validation
- Run existing test suites:
  `python -m pytest tests/test_qwen_references.py tests/test_references.py tests/test_frozen_reference.py tests/test_http_routes.py tests/test_http_security.py tests/test_gui_smoke.py`

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 3

---

## Phase 3: Comprehensive Cross-Shot Unit Test Suite

### Objective
Author and execute `tests/test_cross_shot_matching.py` validating criteria (a)-(g) strictly on CPU without GPU or network calls.

### Agent: `tester`
### Parallel: No

### Files to Create
- `tests/test_cross_shot_matching.py`:
  - (a) Shot 520 retrieves Costume A's approved master reference from shot 5, ignoring adjacent shot 519's Costume B.
  - (b) 100% backward compatibility when no Color Bible DB or `film_id` is configured.
  - (c) `pending_review` or confidence < threshold triggers `needs_review` gate and cleanly skips inference *before* cache hits.
  - (d) `Qwen3-VL` prompt enrichment correctly structures character names, costume IDs, lighting, RGB swatches, and LAB stats.
  - (e) Two-stage DDColor draft -> Qwen2511 escalation router (with mocked inference) and per-row `reference_backend` override.
  - (f) Cache signature invalidates ONLY shots affected by a Color Bible update (Costume B edit does NOT invalidate Costume A shot 520).
  - (g) Commercial Model Guard fails closed on prohibited models and records `color_bible_version` + `ref_id`s in `compliance_report.json`.

### Validation
- Execute targeted test command:
  `python -m pytest tests/test_cross_shot_matching.py tests/test_qwen_references.py tests/test_references.py tests/test_reference_luminance.py tests/test_frozen_reference.py tests/test_color_bible_api.py tests/test_color_bible_cli.py tests/test_color_bible_db.py tests/test_color_bible_service.py tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_ddcolor_reference.py tests/test_http_routes.py tests/test_http_security.py tests/test_gui_smoke.py tests/test_pipeline_order.py`

### Dependencies
- Blocked by: Phase 2
- Blocks: Completion

---

## File Inventory

| # | File | Phase | Purpose |
|---|---|---|---|
| 1 | `ai_remaster_gui/semantic_matching.py` | 1 | [NEW] Central Phase 3 domain logic, retrieval, review gate, registry, router, caching, compliance |
| 2 | `scripts/qwen_colorize_references.py` | 2 | [MODIFY] Upgrade reference retrieval, CLI flags, prompt enrichment, flow reordering, review gate short-circuit |
| 3 | `scripts/ddcolor_reference.py` | 2 | [MODIFY] Expose callable helper for two-stage reference runner |
| 4 | `ai_remaster_gui/references.py` | 2 | [MODIFY] Expose `needs_review` status and entity palettes in `shot_rows()` |
| 5 | `ai_remaster_gui/http_handler.py` | 2 | [MODIFY] Expose review gate and shot assignment status in HTTP API |
| 6 | `tests/test_cross_shot_matching.py` | 3 | [NEW] Comprehensive unit test suite covering criteria (a)-(g) on CPU |

## Risk Classification

| Phase | Risk | Rationale |
|---|---|---|
| 1 | LOW | Self-contained new module with zero circular dependencies and clean abstractions |
| 2 | MEDIUM | Modifies CLI script and GUI modules; mitigated by backward-compatible defaults and thorough existing tests |
| 3 | LOW | Test-only suite executing purely on CPU using isolated temporary directories and mocks |

## Execution Profile

```
Execution Profile:
- Total phases: 3
- Parallelizable phases: 0
- Sequential-only phases: 3 (Sequential High Precision mode requested by user)
- Estimated execution time: ~5-10 minutes across sequential agent turns
```
