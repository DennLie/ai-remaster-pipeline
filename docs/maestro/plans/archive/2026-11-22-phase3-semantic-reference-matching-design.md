---
title: "Phase 3: Semantic Reference Matching"
created: "2026-11-22T17:34:00+01:00"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "standard"
task_complexity: "complex"
---

# Phase 3: Semantic Reference Matching Design Document

## Problem Statement

Currently, `reference_images_for()` in `scripts/qwen_colorize_references.py` relies solely on same-shot and adjacent-shot neighbor lookups. In a full-length feature film (e.g. 90 minutes, 720+ shots), this neighbor-only heuristic causes severe cross-scene color drift: for example, shot 520 in minute 80 may inadvertently adopt an unrelated costume palette from adjacent shot 519, rather than retrieving the character's approved master costume reference established in shot 5 (minute 4). Furthermore, when new entities or variants appear with unverified or low-confidence assignments, the pipeline has no review gate, silently guessing references and baking color errors into the remaster. Additionally, running expensive 20B neural image editing (Qwen-Image-Edit-2511 at ~18s/still) for simple unconstrained backgrounds is inefficient compared to fast deterministic drafts (DDColor at ~0.4s/still), and updating a single character variant in the Color Bible currently risks invalidating all reference caches across the film.

Phase 3 resolves this by replacing adjacent-shot dependency with film-wide semantic reference matching:
1. A 3-Tier Retrieval Engine prioritizing approved Color Bible master references (Priority A) with clean fallback to local neighbor frames (Priority B), gated by a strict Human-in-the-Loop review check (Priority C) evaluated prior to cache lookups.
2. Structured Qwen3-VL prompt enrichment injecting character names, costume IDs, lighting context, target RGB swatches, and CIE-LAB statistics into guided neural generation.
3. A two-stage reference runner (`--reference-runner=auto|guided_only|ddcolor_only`) that dispatches fast DDColor drafts for unconstrained shots and automatically escalates constrained/flagged shots to a pluggable guided backend (`qwen2511`).
4. Scoped cache signatures recording retrieved `ref_id`s and versions, and Commercial Model Guard auditing logged to `compliance_report.json`.

## Requirements

### Functional Requirements

- **REQ-1 (3-Tier Semantic Retrieval):** `reference_images_for()` must query `ColorBibleService` for all assigned entities for the target shot (Priority A), prioritizing character costume appearance variants over objects and locations, deduplicating normalized file paths, and filling remaining slots (up to `MAX_REFERENCE_IMAGES = 2`) with local adjacent shot references (Priority B).
- **REQ-2 (Human-in-the-Loop Review Gate):** If any assigned entity has `approval_status == 'pending_review'` or `confidence < threshold` (default `0.8`), the shot is marked as `needs_review` in the manifest and Color Bible shot status, cleanly skipping inference. Priority C MUST be evaluated *before* checking cache hits to prevent reusing stale cached outputs on unapproved assignments.
- **REQ-3 (Structured Qwen3-VL Prompt Enrichment):** Inject character names, costume IDs, lighting context, target RGB swatches, and CIE-LAB statistics into the prompt builder to enforce consistent colors across scenes.
- **REQ-4 (Pluggable Two-Stage Runner):** Implement a `GuidedReferenceBackend` protocol/registry (`qwen2511` default) and runner CLI (`--reference-runner=auto|guided_only|ddcolor_only`, `--guided-backend=qwen2511`, manifest flag `escalate_guided`). In `auto`, unconstrained shots execute fast DDColor (0.4s), while shots with Color Bible entities escalate directly to the guided backend from grayscale, honoring per-row `reference_backend` overrides.
- **REQ-5 (Granular Scoped Caching):** Cache signature must record retrieved `ref_id`s, their versions/hashes, and active `color_bible_version`. The cache-hit equality key (`compute_scoped_cache_key`) hashes only shot-specific retrieved `ref_id`s and versions so modifying one entry invalidates only shots referencing that entity.
- **REQ-6 (Commercial Policy & Provenance):** Guard every backend call with Phase 1's `commercial_policy` and record `color_bible_version` and retrieved `ref_id`s into `output/compliance_report.json`.
- **REQ-7 (GUI/API Visibility):** Update `shot_rows()` in `references.py` and HTTP endpoints to expose `needs_review` flags and assigned entity palettes.

### Non-Functional Requirements & Constraints

- 100% backward compatible when no Color Bible DB or film is configured.
- 100% CPU-testable on Windows without GPU or live ComfyUI/network calls.
- Strictly typed, zero placeholders, zero TODOs, zero silent fallbacks to non-commercial models.

## Approach

### Selected Approach

**Approach 1: Modular Semantic Matching Subsystem with Pluggable Guided Backend Registry**
All Phase 3 domain logic is encapsulated in a dedicated module `ai_remaster_gui/semantic_matching.py`, which is imported and utilized by `scripts/qwen_colorize_references.py` and `ai_remaster_gui/references.py`.
- **3-Tier Semantic Engine**: `ColorBibleMatcher.resolve_references()` queries `ColorBibleService.get_shot_palette_assignments()`, sorting character appearance variants ahead of objects and locations, deduplicating normalized file paths, and filling remaining slots up to `MAX_REFERENCE_IMAGES` from adjacent shots.
- **Review Gate (Priority C)**: Checks `approval_status == 'pending_review'` or `confidence < threshold` *prior* to cache lookup; unapproved shots are marked `needs_review` and cleanly skipped without polluting colors.
- **Pluggable Guided Backend Registry**: Defines a `GuidedReferenceBackend` protocol (`name`, `colorize_shot(...)`) with `Qwen2511Backend` as the default. The two-stage runner (`--reference-runner=auto|guided_only|ddcolor_only`, `--guided-backend=qwen2511`) dispatches unconstrained shots to DDColor and constrained/flagged shots directly from grayscale to the guided backend, honoring per-row `reference_backend` manifest overrides.
- **Scoped Cache & Invalidation**: `compute_scoped_cache_key()` hashes only shot-specific retrieved `ref_id`s, individual content hashes/versions, and shot parameters. Global film `color_bible_version` is stored as provenance metadata outside the equality check, ensuring updates to one character never invalidate unrelated shots.

### Alternatives Considered

- *Approach 2 (Monolithic In-Place)*: Duplicating Color Bible logic in `qwen_colorize_references.py` and GUI files. Rejected due to maintainability risks and difficult isolated unit testing.
- *Approach 3 (DB-Centric Pre-Materialization)*: Adding mandatory SQLite tables for reference caching. Rejected because it breaks zero-config backward compatibility for non-Color-Bible projects.

### Decision Matrix

| Criterion | Weight | Approach 1 (Subsystem) | Approach 2 (Monolithic) | Approach 3 (DB-Centric) |
|---|---|---|---|---|
| Backward Compatibility | 25% | 5/5: Transparent fallback | 4/5: Harder fallbacks | 2/5: Requires DB setup |
| Precision & Review Gate | 25% | 5/5: Gate checked before cache | 4/5: Dispersed gate checks | 4/5: Rigid table gating |
| Granular Invalidation | 20% | 5/5: Scoped entity hash | 4/5: Global hash drift | 3/5: Table-level invalidation |
| CPU Testability & Modularity | 15% | 5/5: 100% CPU isolated tests | 3/5: Script globals patching | 3/5: DB migrations in tests |
| Extensibility (Pluggable Registry) | 15% | 5/5: Protocol-driven backends | 2/5: Hardcoded Qwen | 3/5: Custom schemas |
| **Weighted Total** | 100% | **5.00** | **3.55** | **2.95** |

## Architecture

### Component Diagram & Flow

```
[Shot Manifest Row (index i)]
           |
           v
+-------------------------------------------------------------+
| Step 1: ColorBibleMatcher.resolve_references()              |
| -> Priority A (DB): Characters > Objects > Locations        |
| -> Priority B (Local): Neighbor shots if slots remain       |
| -> Path deduplication & missing file fallback               |
+-------------------------------------------------------------+
           |
           v
+-------------------------------------------------------------+
| Step 2: Priority C Review Gate Check                        |
| -> If pending_review or confidence < threshold:             |
|    Mark 'needs_review' in shot & manifest; log entity info; |
|    Cleanly skip inference (Short-circuit before cache hit)  |
+-------------------------------------------------------------+
           | (Pass)
           v
+-------------------------------------------------------------+
| Step 3: TwoStageReferenceRouter.select_backend()            |
| -> Selects backend (DDColor or GuidedBackend Qwen2511)      |
|    respecting row reference_backend overrides               |
+-------------------------------------------------------------+
           |
           v
+-------------------------------------------------------------+
| Step 4: PromptEnricher.enrich_prompt()                      |
| -> Injects target RGB swatches, CIE-LAB stats, lighting ctx |
+-------------------------------------------------------------+
           |
           v
+-------------------------------------------------------------+
| Step 5: compute_scoped_cache_key() & Check Resumption       |
| -> Uses shot-specific retrieved_ref_hashes & prompt params  |
+-------------------------------------------------------------+
      | (Cache Hit)                           | (Cache Miss)
      v                                       v
[Reuse Existing Output]          +-------------------------------+
                                 | Step 6: Commercial Policy     |
                                 | evaluate_component() check    |
                                 +-------------------------------+
                                              | (Pass)
                                              v
                                 +-------------------------------+
                                 | Step 7: Execute Backend       |
                                 | (DDColor or Guided Qwen2511)  |
                                 +-------------------------------+
                                              |
                                              v
                                 +-------------------------------+
                                 | Step 8: Write Scoped Sig &    |
                                 | append_compliance_record()    |
                                 +-------------------------------+
```

### Key Interfaces (`ai_remaster_gui/semantic_matching.py`)

```python
class GuidedReferenceBackend(Protocol):
    name: str
    def colorize_shot(self, request: ReferenceRequest) -> ReferenceResult: ...

@dataclass(frozen=True)
class SemanticMatchResult:
    reference_paths: list[Path]
    retrieved_ref_ids: list[str]
    retrieved_ref_hashes: dict[str, str]
    entities: list[dict[str, Any]]
    needs_review: bool
    review_reasons: list[str]
    color_bible_version: str

def compute_scoped_cache_key(
    shot_index: int,
    source_fingerprint: str,
    retrieved_ref_hashes: dict[str, str],
    prompt: str,
    backend_params: dict[str, Any],
) -> str: ...
```

## Agent Team

| Phase | Agent | Mode | Deliverables |
|---|---|---|---|
| **Phase 1: Foundation & Semantic Engine** | `coder` | Sequential | Create `ai_remaster_gui/semantic_matching.py` with 3-tier retrieval, Priority C review gate, `GuidedReferenceBackend` protocol/registry (`qwen2511`), two-stage router, scoped cache key generator, and compliance recorder. |
| **Phase 2: Pipeline Integration & CLI/GUI** | `coder` | Sequential | Upgrade `scripts/qwen_colorize_references.py` (`reference_images_for`, prompt builder, CLI flags, flow ordering, review gate short-circuit before cache), update `scripts/ddcolor_reference.py`, and integrate `needs_review` status into `ai_remaster_gui/references.py` and `http_handler.py`. |
| **Phase 3: Comprehensive Test Suite** | `tester` | Sequential | Build `tests/test_cross_shot_matching.py` validating criteria (a)-(g) on CPU: cross-shot distant retrieval, backward compatibility, review gating, prompt enrichment, two-stage escalation, scoped cache invalidation, and commercial policy compliance. |
| **Completion: Quality & Security Gate** | `code-reviewer` | Sequential | Rigorous review of code quality, type hints, edge cases, error propagation, and zero-TODO/zero-placeholder enforcement. |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| **R1: Backward Compatibility Regression** in existing scripts or non-Color-Bible projects | HIGH | LOW | `ColorBibleMatcher` transparently falls back to existing adjacent-shot neighbor logic if `film_id` or Color Bible DB is unset. Existing CLI flags and arguments remain 100% supported. |
| **R2: Silent Color Drift via Stale Cache** when new unapproved assignments are added | HIGH | MEDIUM | Priority C Review Gate is evaluated strictly *before* cache lookup. Shots with `pending_review` or low-confidence assignments short-circuit immediately to `needs_review` and skip inference. |
| **R3: Full-Film Cache Invalidation** when a single character/costume is updated | MEDIUM | MEDIUM | The cache-hit equality key (`compute_scoped_cache_key`) hashes only the shot-specific retrieved `ref_id`s, their versions/hashes, and enriched parameters. The global `color_bible_version` is kept as provenance metadata only. |
| **R4: Commercial Policy Breach** (unauthorized or non-commercial model invocation) | HIGH | LOW | Every backend call (DDColor and Guided Backend) must pass `commercial_policy.evaluate_component(...)` before invocation, appending audit records to `output/compliance_report.json`. |
| **R5: Test Suite Flakiness or Hardware Dependency** on Windows CPU | HIGH | LOW | All unit tests in `tests/test_cross_shot_matching.py` mock out external ComfyUI network and GPU calls, executing purely on CPU in < 5 seconds. |

## Success Criteria

1. **Distant Semantic Retrieval:** In a multi-shot project, shot 520 assigned Costume A retrieves Costume A's approved master reference from shot 5, ignoring adjacent shot 519's Costume B.
2. **Backward Compatibility:** When no Color Bible DB or film is configured, `reference_images_for()` preserves 100% identical behavior to existing neighbor-shot selection with zero crashes or regressions.
3. **Review Gate Short-Circuit:** Any shot containing an entity assignment with `approval_status == 'pending_review'` or `confidence < 0.8` is flagged `needs_review`, emits a clear warning, and cleanly skips inference *before* evaluating any cache hits.
4. **Structured Qwen3-VL Prompt Enrichment:** Color Bible constraints (entity names, costume IDs, lighting context, target RGB swatches in hex/RGB, CIE-LAB percentiles, provenance) are systematically formatted into the guided prompt.
5. **Pluggable Two-Stage Draft & Escalation:** In `auto` mode, unconstrained shots route to fast DDColor, while shots with Color Bible entities or `escalate_guided=true` route directly to `qwen2511` from grayscale, honoring per-row `reference_backend` overrides.
6. **Scoped Cache Invalidation:** Updating Costume B in shot 100 invalidates shot 100, while shot 520 (referencing Costume A) retains a valid cache hit without requiring a re-render.
7. **Commercial Model Guard & Provenance:** Non-commercial models fail closed; every approved run logs `color_bible_version` and retrieved `ref_id`s into `output/compliance_report.json`.
8. **100% CPU Test Suite Pass:** All tests in `tests/test_cross_shot_matching.py` and the existing suites pass 100% on Windows CPU without live GPU or network access.
