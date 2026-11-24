---
title: "Phase 5 Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-24-phase5-color-qc-design.md"
created: "2026-11-24T10:51:00Z"
status: "approved"
total_phases: 5
estimated_files: 10
task_complexity: "complex"
---

# Phase 5: Automated Cross-Shot Color QC, Selective Re-Render, Multi-Worker & Benchmark Implementation Plan

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `data-engineer`, `coder`, `tester`
- **Estimated effort**: High precision sequential execution across database migration, pure vectorized color science math, batch dispatch, CLI wrappers, and non-commercial benchmarking.

## Dependency Graph

```
Phase 1: Persistence & Schema Layer (data-engineer)
    |
    v
Phase 2: Color QC Vectorized Engine & CLI (coder)
    |
    v
Phase 3: Selective Re-Render & Batch Dispatch (coder)
    |
    v
Phase 4: Non-Commercial Benchmark & HTTP Route (coder)
    |
    v
Phase 5: Full Acceptance Gate Verification (tester)
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1 | Phase 1: Persistence & Schema Layer | Sequential | 1 | SQLite Migration v2 (`qc_events`), `ColorBibleService` methods, models, tests |
| 2 | Phase 2: Color QC Vectorized Engine & CLI | Sequential | 1 | Pure vectorized CIEDE2000, DIS flow, CDC, outer/far ring mask bleed, CLI, wrapper, tests |
| 3 | Phase 3: Selective Re-Render & Batch Dispatch | Sequential | 1 | `rerender_failed_shots.py`, `parallel_colorize_dispatch.py`, tests |
| 4 | Phase 4: Non-Commercial Benchmark & HTTP Route | Sequential | 1 | `benchmark_colorizers.py`, `/api/color-bible/qc-summary`, tests |
| 5 | Phase 5: Full Acceptance Gate Verification | Sequential | 1 | Complete 25+ suite CPU acceptance gate |

---

## Phase 1: Persistence & Schema Layer

### Objective
Implement an idempotent SQLite migration `version = 2` in `ai_remaster_gui/color_bible_db.py` to create the `qc_events` table (including optional `shot_index INTEGER`), define the `QCEvent` data model, implement CRUD helper functions (`record_qc_event`, `list_qc_events`, `get_latest_qc_event`), expose `record_qc_event`, `list_qc_events`, and `get_qc_summary` on `ColorBibleService` (`ai_remaster_gui/color_bible_service.py`), and expand unit tests in `tests/test_color_bible_db.py` and `tests/test_color_bible_service.py`.

### Agent: `data-engineer`
### Parallel: No

### Files to Modify
- `ai_remaster_gui/color_bible_db.py` — Add `MIGRATION_2_SQL` (creating `qc_events` with `shot_index INTEGER` and index on `(film_id, shot_id)`), append `(2, MIGRATION_2_SQL)` to `MIGRATIONS`, define `QCEvent` dataclass, and implement `record_qc_event`, `list_qc_events`, `get_latest_qc_event`.
- `ai_remaster_gui/color_bible_service.py` — Expose `record_qc_event`, `list_qc_events`, and `get_qc_summary` on `ColorBibleService`.
- `tests/test_color_bible_db.py` — Add unit tests verifying migration v2 runs idempotently on fresh and pre-existing v1 databases without data loss, and verify `QCEvent` persistence and query filtering.

### Implementation Details
```python
MIGRATION_2_SQL = """
CREATE TABLE IF NOT EXISTS qc_events (
    event_id TEXT PRIMARY KEY,
    film_id TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
    shot_id TEXT REFERENCES shots(shot_id) ON DELETE SET NULL,
    shot_index INTEGER,
    qc_status TEXT CHECK(qc_status IN ('pass', 'warn', 'fail')) NOT NULL,
    failure_codes_json TEXT NOT NULL DEFAULT '[]',
    metrics_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qc_events_film_shot ON qc_events(film_id, shot_id);
"""
```

### Validation
```bash
python -m pytest tests/test_color_bible_db.py -q
```

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: Color QC Vectorized Engine & CLI

### Objective
Implement `ai_remaster_gui/color_qc.py` containing vectorized CIEDE2000 math (Sharma et al., 2005) with reference-pinned $L^*$ normalization, Color Distribution Consistency (CDC), OpenCV DIS optical-flow backward chroma warping error, sudden frame-to-frame hue jump detection, and the refined two-ring mask bleed heuristic comparing the immediate outer ring (`dilate(R) - mask`) against a farther background ring (`dilate(3*R) - dilate(R)`) to detect true color spill into the background. Implement CLI `scripts/qc_color_consistency.py`, batch wrapper `wrappers/qc_color_consistency.bat`, and test suite `tests/test_color_qc.py`.

### Agent: `coder`
### Parallel: No

### Files to Create
- `ai_remaster_gui/color_qc.py` — Vectorized color metrics, thresholding, and shot QC evaluator.
- `scripts/qc_color_consistency.py` — Standalone CLI evaluating shots or entire manifests, dual-writing results to manifest CSV and `qc_events`.
- `wrappers/qc_color_consistency.bat` — Windows batch wrapper matching existing ARP wrapper conventions.
- `tests/test_color_qc.py` — Unit tests for CIE 34-pair ground-truth dataset, $L^*$ illumination invariance, injected chromatic error detection, CDC, DIS warping error, hue jumps, and mask bleed.

### Implementation Details
- `ciede2000(lab1, lab2, kL=1.0, kC=1.0, kH=1.0, normalize_lightness=False)`:
  - Vectorized calculation handling $C'=0$ and hue difference $\Delta h'$ wrap-around across $360^\circ$ and negative values.
  - When `normalize_lightness=True`, sets $L_1 = L_2$ to isolate pure chromatic distance ($\Delta C', \Delta H'$).
- `compute_cdc(frames_rgb, stride=1)`: Pairwise histogram intersection across consecutive strided frames.
- `compute_optical_flow_chroma_error(frames_gray, frames_lab, stride=1, preset=DISOPTICAL_FLOW_PRESET_FAST)`:
  - Backward warps $a^*, b^*$ channels from frame $t-1$ to frame $t$ using OpenCV DIS flow.
  - Measures residual Euclidean distance in chroma space.
- `detect_mask_bleed(frame_lab, mask, radius=3, threshold=8.0)`:
  - Immediate outer ring: $M_{near} = \text{dilate}(mask, R) \land \neg mask$.
  - Far background ring: $M_{far} = \text{dilate}(mask, 3R) \land \neg \text{dilate}(mask, R)$.
  - Interior object: $M_{in} = mask$.
  - Computes chroma spill: projection of $(C_{near} - C_{far})$ onto unit vector $(C_{in} - C_{far})$ in $a^*b^*$ space. Positive values indicate true color bleeding into the background.

### Validation
```bash
python -m pytest tests/test_color_qc.py -q
```

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 3

---

## Phase 3: Selective Re-Render & Batch Dispatch

### Objective
Implement `scripts/rerender_failed_shots.py` to inspect manifests or `qc_events`, extract shots with `qc_status == 'fail'`, and trigger `scripts/colorize_video.py` targeting ONLY those shots with signature-based caching preserving passing shots. Implement `scripts/parallel_colorize_dispatch.py` for concurrent local subprocess pooling (`--workers N`) and RunPod shard manifest generation (`--worker-index i --worker-count N`) using frozen Color Bible snapshots. Add unit tests in `tests/test_parallel_dispatch_and_benchmark.py`.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/rerender_failed_shots.py` — Reads manifest/DB, identifies failed shots, and issues selective colorization.
- `scripts/parallel_colorize_dispatch.py` — Partitions shots across workers, executes local pool or emits RunPod shard manifests, merges outputs deterministically.
- `tests/test_parallel_dispatch_and_benchmark.py` — Tests for selective re-render isolation and multi-worker partitioning determinism ($N=2, N=4$, reversed order).

### Implementation Details
- Shard partitioning: Deterministic interleaved or contiguous chunk slicing:
  `worker_shots = [shot for idx, shot in enumerate(shots) if idx % worker_count == worker_index]`.
- Selective re-render writes filtered manifest containing only failed shot rows, invokes `colorize_video.py`, and merges back updated rows into master manifest.

### Validation
```bash
python -m pytest tests/test_parallel_dispatch_and_benchmark.py -q
```

### Dependencies
- Blocked by: Phase 2
- Blocks: Phase 4

---

## Phase 4: Non-Commercial Benchmark & HTTP Route

### Objective
Implement `scripts/benchmark_colorizers.py` gated behind `--allow-non-commercial-benchmark`, restricted to non-commercial output sandboxes, comparing `osi_masked_lab` against `CMNET2` (with CPU mock support for unit testing). Implement read-only route `GET /api/color-bible/qc-summary` in `ai_remaster_gui/http_handler.py` consuming `ColorBibleService.get_qc_summary()`. Add tests in `tests/test_http_routes.py` and expand `tests/test_parallel_dispatch_and_benchmark.py`.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/benchmark_colorizers.py` — Benchmark runner computing PSNR, mean/p90 $\Delta E_{2000}$, and CDC.

### Files to Modify
- `ai_remaster_gui/http_handler.py` — Register `GET /api/color-bible/qc-summary` using `ColorBibleService.get_qc_summary()` with host loopback validation, origin check, and parameterized film/shot queries.
- `tests/test_http_routes.py` — Add `/api/color-bible/qc-summary` to `EXPECTED_GET_ROUTES` and test response shapes.
- `tests/test_parallel_dispatch_and_benchmark.py` — Add benchmark security gate tests (fail without flag, output path constraint, valid report generation with mocked CMNET2).

### Validation
```bash
python -m pytest tests/test_http_routes.py tests/test_parallel_dispatch_and_benchmark.py -q
```

### Dependencies
- Blocked by: Phase 3
- Blocks: Phase 5

---

## Phase 5: Full Acceptance Gate Verification

### Objective
Run the complete acceptance test gate across all 25+ test suites on Windows CPU, validating zero regressions and 100% pass rate.

### Agent: `tester`
### Parallel: No

### Files to Create / Modify
- None.

### Validation Command
```bash
python -m pytest tests/test_color_qc*.py tests/test_parallel_dispatch*.py tests/test_e2e_commercial_pipeline_wiring.py tests/test_rle_masks.py tests/test_sam2_video_tracker.py tests/test_osi_masked_lab.py tests/test_frame_masks.py tests/test_temporal_consistency.py tests/test_reference_luminance.py tests/test_ltx_colorize.py tests/test_video_order.py tests/test_runpod_stage.py tests/test_chunk_plan.py tests/test_cross_shot_matching.py tests/test_semantic_matching.py tests/test_color_bible_api.py tests/test_color_bible_cli.py tests/test_color_bible_db.py tests/test_color_bible_service.py tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_ddcolor_reference.py tests/test_model_license_inventory.py tests/test_http_routes.py tests/test_http_security.py tests/test_gui_smoke.py tests/test_pipeline_order.py -q
```

### Dependencies
- Blocked by: Phase 4
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `ai_remaster_gui/color_bible_db.py` | 1 | Schema migration v2 (`qc_events`), `QCEvent` model, CRUD queries |
| 2 | `ai_remaster_gui/color_bible_service.py` | 1 | Expose `record_qc_event`, `list_qc_events`, `get_qc_summary` on `ColorBibleService` |
| 3 | `tests/test_color_bible_db.py` | 1 | Unit tests for migration v2 and `qc_events` table |
| 4 | `ai_remaster_gui/color_qc.py` | 2 | Vectorized CIEDE2000, CDC, DIS optical flow, hue jump, refined mask bleed |
| 5 | `scripts/qc_color_consistency.py` | 2 | Standalone CLI for shot & manifest color QC evaluation |
| 6 | `wrappers/qc_color_consistency.bat` | 2 | Windows batch wrapper for color QC CLI |
| 7 | `tests/test_color_qc.py` | 2 | CIE ground-truth test suite, illumination invariance, temporal metrics |
| 8 | `scripts/rerender_failed_shots.py` | 3 | Selective re-render of `qc_status == 'fail'` shots |
| 9 | `scripts/parallel_colorize_dispatch.py` | 3 | Multi-worker local pool & RunPod shard manifest dispatcher |
| 10 | `tests/test_parallel_dispatch_and_benchmark.py` | 3 & 4 | Tests for re-render, worker partitioning determinism, and benchmark |
| 11 | `scripts/benchmark_colorizers.py` | 4 | Standalone non-commercial benchmark comparing `osi_masked_lab` vs `CMNET2` |
| 12 | `ai_remaster_gui/http_handler.py` | 4 | Read-only `/api/color-bible/qc-summary` route |
| 13 | `tests/test_http_routes.py` | 4 | Route registry and security tests for `/api/color-bible/qc-summary` |

## Risk Classification

| Phase | Risk | Rationale |
|---|---|---|
| Phase 1 | LOW | Non-destructive SQLite migration v2 adding a new table; tested against existing databases. |
| Phase 2 | MEDIUM | Complex vectorized color math and optical flow; mitigated by testing against standard CIE ground-truth dataset and background ring differential. |
| Phase 3 | MEDIUM | Process concurrency and manifest partitioning; mitigated by deterministic integer indexing and signature cache re-use. |
| Phase 4 | LOW | Strict security flags and mockable CPU backend preventing licensing or network leakage. |
| Phase 5 | LOW | Read-only test execution confirming all acceptance criteria. |

## Execution Profile

```
Execution Profile:
- Total phases: 5
- Parallelizable phases: 0 (Strict Sequential Mode chosen for High Precision)
- Sequential-only phases: 5
- Estimated sequential wall time: ~4-6 minutes across implementation and verification
```
