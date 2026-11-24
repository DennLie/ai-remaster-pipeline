---
title: "Phase 5: Automated Cross-Shot Color QC, Selective Re-Render, Multi-Worker & Benchmark"
created: "2026-11-24T10:45:00Z"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "standard"
task_complexity: "complex"
---

# Phase 5: Automated Cross-Shot Color QC, Selective Re-Render, Multi-Worker & Benchmark Design Document

## Problem Statement

In commercial feature-length film remastering (60–120 minutes, 480–1,600 shots), maintaining chromatic consistency across non-adjacent shots separated by scenes or hours of runtime is a critical challenge. Pure memory-based video models lose state across cut boundaries, leading to gradual color drift where a character's costume or prop changes appearance between scenes. Furthermore, temporal artifacts such as color flickering, sudden frame-to-frame hue jumps, and chroma bleeding across object mask boundaries degrade visual quality.

Currently, detecting these discrepancies requires labor-intensive manual inspection across tens of thousands of frames. While Phases 1–4.5 established the foundational Color Bible relational database, RLE mask tracking, and deterministic OSI masked LAB colorization, the pipeline lacks:
1. An automated, perceptual cross-shot quality control engine capable of verifying tracked objects against Color Bible references without false alarms from intentional lighting and shadow shifts.
2. Intra-shot temporal consistency metrics that operate in real time (>=20 fps on 1080p).
3. A selective re-render loop that targets only failed shots while preserving valid renders via signature-based caching.
4. Multi-worker shot-parallel batch dispatching across local cores and RunPod cloud shards.
5. An isolated, compliance-safe internal benchmark comparing OSI colorization against state-of-the-art non-commercial models like CMNET2 without licensing risks.

## Requirements

### Functional Requirements

- **REQ-1 (Vectorized CIEDE2000 Engine)**: Pure NumPy implementation of Sharma et al. (2005) CIEDE2000 matching CIE test pairs within $10^{-4}$ tolerance. Evaluates tracked objects via RLE `.masks.npz` with reference-pinned $L^*$ normalization ($L_{test}^* = L_{ref}^*$) to isolate chroma/hue drift without false alarms from shadows or lighting changes, alongside raw full $\Delta E_{2000}$.
- **REQ-2 (Intra-Shot Temporal Metrics)**: Color Distribution Consistency (CDC), OpenCV DIS optical-flow backward chroma warping error, frame-to-frame hue jumps, and morphological boundary mask-bleed detection, with configurable `--stride` achieving $\ge 20$ fps on 1080p clips.
- **REQ-3 (Persistence & API Integration)**: Idempotent SQLite migration v2 creating `qc_events` table in `ai_remaster_gui/color_bible_db.py`. Dual-writes structured status (`pass`, `warn`, `fail`), failure reason codes, and metrics to shot manifest CSV and `qc_events`. Exposes read-only `/api/color-bible/qc-summary` in `http_handler.py` with loopback/origin/path-safety validation.
- **REQ-4 (Selective Re-Render)**: `scripts/rerender_failed_shots.py` re-runs ONLY shots flagged `qc_status == 'fail'` via signature-based caching, leaving passing shots untouched.
- **REQ-5 (Multi-Worker Batch Dispatcher)**: `scripts/parallel_colorize_dispatch.py` providing dual local worker pool (`--workers N`) and RunPod shard manifest generation (`--worker-index i --worker-count N`) against frozen Color Bible snapshots with order-independent deterministic outputs.
- **REQ-6 (Non-Commercial Benchmark Comparator)**: `scripts/benchmark_colorizers.py` gated behind `--allow-non-commercial-benchmark`, restricted to non-commercial output sandboxes, reporting PSNR, mean/p90 $\Delta E_{2000}$, and CDC against mocked CMNET2.

### Non-Functional Requirements & Constraints

- Zero heavy dependencies (pure NumPy + OpenCV DIS).
- 100% CPU-testable on Windows without GPU.
- Commercial Model Guard fails closed against unauthorized models in commercial mode.
- Fully typed and structured with zero TODOs or placeholders.

## Approach

### Selected Approach: Unified Modular Engine Architecture

We adopt a unified modular approach that cleanly decouples mathematical perception, persistence, CLI tooling, distributed orchestration, and non-commercial evaluation:
1. **Core QC Engine (`ai_remaster_gui/color_qc.py`)**: Pure vectorized NumPy CIEDE2000 math, DIS optical flow chroma warping error, CDC computation, hue jump detection, and mask-bleed heuristic returning structured dataclasses (`QCResult`, `ObjectQCScore`, `TemporalQCScore`).
2. **Persistence Layer (`ai_remaster_gui/color_bible_db.py`)**: Migration v2 adding `qc_events` table (`event_id`, `film_id`, `shot_id`, `qc_status`, `failure_codes`, `metrics_json`, `created_at`).
3. **HTTP API (`ai_remaster_gui/http_handler.py`)**: Read-only `/api/color-bible/qc-summary` route with loopback host validation, origin check, and parameterized filtering.
4. **Tooling & Dispatch (`scripts/qc_color_consistency.py`, `scripts/rerender_failed_shots.py`, `scripts/parallel_colorize_dispatch.py`, `wrappers/qc_color_consistency.bat`)**: Standalone CLIs for QC evaluation, signature-cached selective re-rendering, and dual-mode batch dispatching.
5. **Benchmark Comparator (`scripts/benchmark_colorizers.py`)**: Fail-closed compliance guarded comparator between `osi_masked_lab` and mockable `CMNET2`.

### Alternatives Considered

#### Monolithic Script-Centric Architecture
- **Description**: Embedding QC routines and SQLite writes directly into `scripts/colorize_video.py` or a single monolithic script.
- **Pros**: Fewer files to touch.
- **Cons**: High code duplication, tightly coupled, violates separation of concerns, cannot be queried via API or GUI.
- **Rejected Because**: Unmaintainable in a production pipeline; prevents isolated unit testing and clean architectural layering.

#### External Microservice / Daemon Architecture
- **Description**: Running an external message queue daemon (Celery/Redis) to process QC jobs asynchronously.
- **Pros**: Independent scaling across clusters.
- **Cons**: Severe operational complexity and introduces heavy dependencies outside ARP's lightweight architecture.
- **Rejected Because**: Disproportionate complexity; violates zero-extra-dependency and standalone CLI design.

### Decision Matrix

| Criterion | Weight | Approach 1 (Unified Modular) | Approach 2 (Monolithic Script) | Approach 3 (Microservice Daemon) |
|---|---|---|---|---|
| Perceptual Accuracy & Math | 25% | 5/5: Vectorized CIE standard | 4/5: Script-bound math | 4/5: Async queue overhead |
| Cohesion & ARP Integration | 25% | 5/5: Seamless GUI/DB/CLI fits | 2/5: Fragmented CLI-only | 2/5: Complex external daemon |
| Speed & CPU Testability | 20% | 5/5: >=20 fps, 100% CPU unit test | 3/5: Harder to test in isolation | 2/5: Requires brokers/services |
| Compliance & License Safety | 20% | 5/5: Fail-closed hard guards | 3/5: Ad-hoc script flags | 4/5: Isolated sandbox |
| Simplicity & Maintainability | 10% | 4/5: Clean modular boundaries | 3/5: Monolithic script bloat | 1/5: High operational complexity |
| **Weighted Total** | **100%** | **4.90 / 5.00** | **3.10 / 5.00** | **2.70 / 5.00** |

## Architecture

### Component Diagram

```
+-----------------------------------------------------------------------------------+
| Color Bible DB (SQLite v2: qc_events) <----+------------------------------------+  |
+--------------------------------------------|-----------------------------------|--+
                                             |                                   |
                                             v                                   |
+------------------------------+     +-------------------+     +-----------------+--+
| Tracked Masks (.masks.npz)   | --> | color_qc.py       | --> | Shot Manifest CSV  |
| Colorized Frames (RGB/LAB)   |     | (Vectorized Math, |     +--------------------+
+------------------------------+     |  DIS Flow, CDC)   |
                                     +-------------------+
                                             |
                   +-------------------------+-------------------------+
                   v                                                   v
+------------------------------------+               +----------------------------------+
| rerender_failed_shots.py           |               | http_handler.py                  |
| (Selectively reruns fail shots     |               | (GET /api/color-bible/qc-summary |
|  via signature-based caching)      |               |  with loopback security guards)  |
+------------------------------------+               +----------------------------------+
                   |
                   v
+------------------------------------+               +----------------------------------+
| parallel_colorize_dispatch.py      |               | benchmark_colorizers.py          |
| (Local ProcessPool or RunPod       |               | (Non-commercial comparison:      |
|  sharded manifest generation)      |               |  osi_masked_lab vs CMNET2)       |
+------------------------------------+               +----------------------------------+
```

### Key Interfaces

- `ciede2000(lab1: np.ndarray, lab2: np.ndarray, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0, normalize_lightness: bool = False) -> np.ndarray`
- `compute_cdc(frames_rgb: Sequence[np.ndarray], stride: int = 1) -> float`
- `compute_optical_flow_chroma_error(frames_gray: Sequence[np.ndarray], frames_lab: Sequence[np.ndarray], stride: int = 1) -> float`
- `detect_hue_jumps(frames_lab: Sequence[np.ndarray], threshold: float = 25.0, stride: int = 1) -> list[int]`
- `detect_mask_bleed(frame_lab: np.ndarray, mask: np.ndarray, dilation_radius: int = 3, threshold: float = 12.0) -> float`
- `evaluate_shot_qc(...) -> QCResult`
- `record_qc_event(conn: Connection, event: QCEvent) -> None`

## Agent Team

Sequential execution is pinned to ensure 100% precision and avoid file lock collisions:

| Stage / Phase | Assigned Agent | Capability Tier | Deliverables |
|---|---|---|---|
| **Phase 1: Persistence & Schema Layer** | `data-engineer` | Full Access | SQLite migration v2 (`qc_events`), DB models, queries, and `test_color_bible_db.py` updates |
| **Phase 2: Color QC Vectorized Engine & CLI** | `coder` | Full Access | `ai_remaster_gui/color_qc.py`, `scripts/qc_color_consistency.py`, `wrappers/qc_color_consistency.bat`, and `tests/test_color_qc.py` |
| **Phase 3: Selective Re-Render & Batch Dispatch** | `coder` | Full Access | `scripts/rerender_failed_shots.py`, `scripts/parallel_colorize_dispatch.py`, deterministic partitioning & re-render test suite |
| **Phase 4: Non-Commercial Benchmark & HTTP Route** | `coder` | Full Access | `scripts/benchmark_colorizers.py`, read-only route in `http_handler.py`, route security and mock benchmark tests |
| **Phase 5: Full Acceptance Gate Verification** | `tester` | Full Access | Full acceptance test gate execution across all 25+ suites on Windows CPU |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation Strategy |
|---|---|---|---|
| **CIEDE2000 Edge Cases & Singularities** | HIGH | MEDIUM | Strictly implement Sharma et al. (2005) formulation with safe atan2 handling; validate against published CIE 34-pair ground-truth benchmark with tolerance $\le 10^{-4}$. |
| **False Alarms from Lighting / Shadow Changes** | HIGH | MEDIUM | Reference-pinned $L^*$ normalization ($L_{test}^* = L_{ref}^*$) during object chroma evaluation to isolate chromatic drift ($\Delta C', \Delta H'$), while recording raw $\Delta E_{2000}$ for reference. |
| **1080p Intra-Shot Flow Throughput Bottleneck** | MEDIUM | MEDIUM | Utilize OpenCV DIS Optical Flow (`FAST` preset) on downscaled/strided grayscale frames (`--stride`) and vectorized morphological operations to guarantee $\ge 20$ fps on CPU. |
| **Commercial License Contamination from CMNET2 Benchmark** | CRITICAL | LOW | Benchmark tool fails closed without `--allow-non-commercial-benchmark`, prohibits writing to commercial output paths, records `NON_COMMERCIAL_BENCHMARK` compliance entries, and supports a mockable CPU execution mode. |
| **SQLite Migration v2 Data Loss on Existing v1 Databases** | HIGH | LOW | Transactional idempotent schema migration checking `schema_migrations`; unit test verifies upgrading active v1 database without schema or data regression. |

## Success Criteria

1. **CIE Ground-Truth Accuracy**: Vectorized CIEDE2000 math matches standard Sharma et al. (2005) 34-pair ground-truth color dataset within $\le 10^{-4}$ numerical tolerance.
2. **Illumination Invariance**: Modifying $L^*$ on tracked objects while preserving chroma ($a^*, b^*$) produces $\Delta E \approx 0$ under reference-pinned normalization, preventing lighting false alarms.
3. **Actionable Defect Detection**: Injecting an incorrect hue on a distant shot triggers `qc_status == 'fail'`, logs failure reason codes in the manifest CSV, and inserts a structured record into `qc_events`.
4. **Selective Re-Render Isolation**: `scripts/rerender_failed_shots.py` targets only failed shots; previously rendered passing shots remain completely untouched with valid cached signatures.
5. **Temporal Defect Sensitivity**: CDC, optical flow chroma warping error, frame-to-frame hue jumps, and mask boundary chroma bleed heuristics catch injected temporal defects with $\ge 20$ fps throughput on strided 1080p clips.
6. **Zero-Data-Loss Migration v2**: SQLite migration v2 applies idempotently to fresh and populated v1 databases without data loss.
7. **Order-Independent Multi-Worker Determinism**: Local concurrent dispatch and RunPod shard execution ($N=2, N=4$, reversed order) against a frozen Color Bible snapshot generate deterministic, order-independent per-shot outputs.
8. **Compliance Fail-Closed Gate**: `scripts/benchmark_colorizers.py` rejects runs without `--allow-non-commercial-benchmark`, prohibits writing into commercial master directories, and generates valid PSNR / CIEDE2000 / CDC reports on CPU with mocked CMNET2.
9. **Full Acceptance Gate Pass**: Targeted pytest command passes 100% on Windows CPU.
