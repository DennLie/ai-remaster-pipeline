---
title: "Scratch and Dust Mask Detection System Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-09-scratch-dust-investigation-design.md"
created: "2026-11-09T16:02:03+01:00"
status: "approved"
total_phases: 5
estimated_files: 13
task_complexity: "medium"
---

# Scratch and Dust Mask Detection System Implementation Plan

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `coder`, `tester`
- **Estimated effort**: Medium (empirical benchmark infrastructure, multi-candidate evaluation, modular strategy architecture, and end-to-end verification)
- **Execution Target**: Designed specifically for handoff and execution by **Claude Sonnet 5.5**.

## Dependency Graph

```
[Phase 1: Benchmark Engine & Synthetic Damage]
                      │
                      ▼
[Phase 2: Archival Pilot Dataset & Temporal Scorer]
                      │
                      ▼
[Phase 3: Multi-Candidate Empirical Scorer & Visuals]
                      │
                      ▼
[Phase 4: Pluggable Architecture & Pipeline Integration]
                      │
                      ▼
[Phase 5: End-to-End Validation & Verification Suite]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|---|---|---|---|---|
| 1 | Phase 1 | Sequential | 1 | Foundation: Synthetic defect generator and spatial scoring engine |
| 2 | Phase 2 | Sequential | 1 | Dataset grounding: Archival annotation protocol and temporal consistency metrics |
| 3 | Phase 3 | Sequential | 1 | Empirical research: Candidate method benchmarking and diagnostic visual diffs |
| 4 | Phase 4 | Sequential | 1 | Architecture: Decoupled strategy pattern and backward-compatible integration |
| 5 | Phase 5 | Sequential | 1 | Quality Gate: End-to-end regression testing and benchmark summary |

---

## Phase 1: Benchmark Engine & Synthetic Damage Generator

### Objective
Implement a modular, automated benchmark engine capable of generating calibrated synthetic damage (variable-width vertical/diagonal scratches and transient dust blotches) with exact binary ground truth masks ($y_{true}$), computing spatial metrics (Precision, Recall, Pixel F1, IoU, and Unaffected Pixel False Positive Rate) strictly separated by defect category.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/benchmark_defect_masks.py` — Benchmark execution engine, synthetic injection harness, metric calculation, and CLI reporting.
- `tests/test_defect_mask_benchmark.py` — Unit tests for synthetic generation, mask binarization, confusion matrix math, and edge cases.

### Implementation Details
- **Synthetic Damage Engine**:
  - Scratch injector: Parametric lines with configurable width (1–8px), opacity (0.2–1.0), curvature/angle ($\pm 15^\circ$ from vertical), length, and inter-frame drift (0–5px/frame) lasting 1 to 30 frames.
  - Dust/Blotch injector: Random spatial clusters of irregular morphology and high contrast simulating dust specks and chemical emulsion blotches on isolated single frames.
- **Scoring Engine**:
  - Accepts predicted binary masks and ground truth masks.
  - Computes True Positives ($TP$), False Positives ($FP$), False Negatives ($FN$), and True Negatives ($TN$).
  - Outputs metrics separated into two distinct channels: `scratch_metrics` and `dust_metrics`.

### Validation
- Run unit test suite: `python -m unittest tests/test_defect_mask_benchmark.py` (all tests passing).
- Run CLI sanity check: `python scripts/benchmark_defect_masks.py --help` (exits 0 with valid flag specification).

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: Archival Pilot Dataset & Temporal Consistency Scorer

### Objective
Formalize the archival ground truth annotation protocol and integrate the 5 curated *Neighbors* (1920) pilot sequences under sequence-level train/validation/test splits, while implementing explicit temporal consistency metrics to measure mask stability across frames.

### Agent: `coder`
### Parallel: No

### Files to Create
- `docs/archival-ground-truth-protocol.md` — Formal specification for 3-tier archival annotation (Defect, Clean, Ambiguous/Ignore) and expansion criteria.
- `scripts/temporal_consistency_scorer.py` — Temporal metric evaluator computing mask flicker, detection continuity, and trajectory dropouts.
- `tests/test_temporal_consistency.py` — Unit tests verifying temporal consistency metrics on known flickering/drifting test patterns.

### Implementation Details
- **Annotation & Dataset Convention**:
  - Annotate authentic archival film damage explicitly to ensure background defects are never treated as clean film.
  - Sequence-level train/val/test splits to eliminate temporal leakage across neighboring frames.
- **Temporal Metrics Suite**:
  - `Mask Flicker Rate`: Frequency of single-frame alternating mask states in static/unaffected regions.
  - `Detection Continuity Score`: Proportion of frames along a persistent scratch's ground-truth lifespan where the defect is actively detected.
  - `Temporal False Positive Rate`: False positive spikes correlated with sudden camera or subject motion.
  - `Missed Persistent Defect Rate`: Frequency of mid-trajectory dropouts on multi-frame scratch lines.

### Validation
- Verify archival protocol documentation matches repo standards.
- Run unit tests: `python -m unittest tests/test_temporal_consistency.py`.

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 3

---

## Phase 3: Multi-Candidate Empirical Scorer & Visual Diagnostics

### Objective
Execute an empirical head-to-head comparison of candidate detection algorithms across the identical synthetic and archival test material, logging quantitative performance tables and generating multi-panel side-by-side visual diagnostic contact sheets.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/candidate_detector_eval.py` — Evaluator runner wrapping each candidate detector (Farneback baseline, DIS flow, Multi-scale Ridge, 5-frame consensus, and DART evaluation harness).
- `docs/defect-detector-empirical-comparison.md` — Detailed empirical report with comparative metric tables, throughput baselines, and trade-off analysis.
- `tests/test_candidate_detectors.py` — Integration tests validating consistent input/output contracts across all evaluated candidates.

### Implementation Details
- Standardized execution interface ensuring all candidates receive identical preprocessed frames and configuration parameters.
- Multi-panel visual artifact generator:
  - Panel A: Source frame.
  - Panel B: Ground truth label mask.
  - Panel C: Predicted scratch mask.
  - Panel D: Predicted dust mask.
  - Panel E: Color-coded Error Map (Green=TP, Red=FP, Blue=FN).
  - Panel F: Temporal difference strip ($t-2 \dots t+2$).
- Profiling: Record mean execution time per frame (ms), peak RAM (MB), and GPU VRAM (if applicable).

### Validation
- Run integration tests: `python -m unittest tests/test_candidate_detectors.py`.
- Run full comparative evaluation script on pilot datasets and verify generation of markdown metrics tables and diagnostic contact sheets.

### Dependencies
- Blocked by: Phase 2
- Blocks: Phase 4

---

## Phase 4: Pluggable BaseDefectDetector Architecture & Pipeline Integration

### Objective
Define and implement the unified `BaseDefectDetector` strategy pattern and registry. Integrate the empirically validated detector backends into `scripts/cleanup_video.py` and `ai_remaster_gui/dust.py`, guaranteeing backward compatibility and complete licensing safety.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/film_defect_detectors.py` — `BaseDefectDetector` abstract base class, detector configuration schemas, and `DefectDetectorRegistry`.
- `tests/test_defect_detector_registry.py` — Unit tests for registry lookup, parameter validation, and detector execution.

### Files to Modify
- `scripts/cleanup_video.py` — Add optional `--defect-detector` selection flag while retaining existing CLI flags and default behavior.
- `ai_remaster_gui/dust.py` — Connect detector registry to GUI inspection routes, supporting detector switching and inspection without breaking existing cached masks.

### Implementation Details
- Ensure core production code imports only commercial-safe (MIT/BSD/Apache-2.0) libraries (OpenCV, NumPy, SciPy).
- Any neural evaluation code (e.g. DART) remains strictly isolated in external evaluation harnesses, never imported in production paths without verified commercial rights.
- Seamless fallback to baseline Farneback detector if no alternate backend is specified.

### Validation
- Run registry tests: `python -m unittest tests/test_defect_detector_registry.py`.
- Run CLI regression tests: verify `scripts/cleanup_video.py --detect-defects` works as expected.
- Run GUI tests: `pytest tests/test_dust_stage.py`.

### Dependencies
- Blocked by: Phase 3
- Blocks: Phase 5

---

## Phase 5: End-to-End Benchmark Validation & Verification Suite

### Objective
Perform end-to-end system verification across all unit, integration, and browser test suites. Verify that no regressions were introduced to existing pipeline stages, and document the final benchmark summary.

### Agent: `tester`
### Parallel: No

### Files to Create
- `docs/defect-detector-benchmark-results.md` — Final summary document detailing benchmark results, acceptance criteria verification, and recommendations for downstream inpainting integration.
- `tests/test_defect_pipeline_integration.py` — End-to-end integration test validating video processing, mask generation, lossless MKV caching, and GUI inspection compatibility.

### Implementation Details
- Execute all related test suites:
  - `tests/test_defect_mask_benchmark.py`
  - `tests/test_temporal_consistency.py`
  - `tests/test_candidate_detectors.py`
  - `tests/test_defect_detector_registry.py`
  - `tests/test_defect_pipeline_integration.py`
  - Existing GUI and pipeline tests (`tests/test_dust_stage.py`, `tests/test_video_order.py`).
- Confirm zero test regressions and full compliance with commercial licensing rules.

### Validation
- All unit and integration test suites pass with zero failures.
- Final documentation reviewed and verified.

### Dependencies
- Blocked by: Phase 4
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|---|---|---|
| 1 | `scripts/benchmark_defect_masks.py` | 1 | Benchmark execution runner and synthetic damage generation engine |
| 2 | `tests/test_defect_mask_benchmark.py` | 1 | Unit tests for synthetic damage injection and spatial scoring metrics |
| 3 | `docs/archival-ground-truth-protocol.md` | 2 | Specification for archival defect annotation and sequence-level splitting |
| 4 | `scripts/temporal_consistency_scorer.py` | 2 | Scorer computing mask flicker, continuity, and trajectory dropouts |
| 5 | `tests/test_temporal_consistency.py` | 2 | Unit tests for temporal consistency metrics |
| 6 | `scripts/candidate_detector_eval.py` | 3 | Comparative evaluation runner across baseline and candidate detectors |
| 7 | `docs/defect-detector-empirical-comparison.md` | 3 | Empirical evaluation report comparing candidate methods |
| 8 | `tests/test_candidate_detectors.py` | 3 | Integration tests for candidate detector contracts |
| 9 | `scripts/film_defect_detectors.py` | 4 | Abstract `BaseDefectDetector` strategy interface and registry |
| 10 | `tests/test_defect_detector_registry.py` | 4 | Unit tests for detector registry and dynamic backend dispatch |
| 11 | `scripts/cleanup_video.py` | 4 | CLI integration for selectable defect detector backends |
| 12 | `ai_remaster_gui/dust.py` | 4 | GUI integration for selectable defect detector backends |
| 13 | `docs/defect-detector-benchmark-results.md` | 5 | Final benchmark summary report and production readiness evaluation |
| 14 | `tests/test_defect_pipeline_integration.py` | 5 | End-to-end verification suite across CLI, GUI, and cache layers |

---

## Risk Classification

| Phase | Risk | Rationale |
|---|---|---|
| 1 | LOW | Independent test tooling and synthetic injection; zero production code touched. |
| 2 | MEDIUM | Ground truth annotation on historical footage requires careful edge handling to avoid subjective bias. |
| 3 | MEDIUM | Evaluating multiple algorithmic backends requires rigorous control of preprocessing to prevent unfair bias. |
| 4 | HIGH | Touches core pipeline and GUI files (`scripts/cleanup_video.py`, `ai_remaster_gui/dust.py`); must maintain strict backward compatibility. |
| 5 | LOW | Verification, regression testing, and documentation. |

---

## Execution Profile

```
Execution Profile:
- Total phases: 5
- Parallelizable phases: 0 (strict sequential dependency order)
- Sequential-only phases: 5
- Primary target runtime: Claude Sonnet 5.5
- Safety constraints: Zero commercial-risk dependencies in production code; all AI models sandboxed in evaluation harnesses.
```
