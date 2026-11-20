---
title: "Temporal Scratch Detection POC & Integration Implementation Plan (Revised: Quality Gate First)"
design_ref: "docs/maestro/plans/2026-11-16-temporal-scratch-detection-design.md"
created: "2026-11-16T15:10:00+01:00"
status: "approved_ready"
total_phases: 5
estimated_files: 7
task_complexity: "complex"
---

# Temporal Scratch Detection Implementation Plan: Quality Validation Precedes Integration

## Plan Overview

- **Total phases**: 5
- **Agents involved**: `coder`, `tester`
- **Estimated effort**: 5 sequential stages strictly gating pipeline integration behind empirical detection improvement on real archival footage.

## Dependency Graph

```
[Phase 1: Audit and Ground Truth Identification]
                         │
                         ▼
[Phase 2: Isolated Temporal PoC (TemporalScratchAnalyzer)]
                         │
                         ▼
[Phase 3: Quantitative Quality Gate (Real Neighbors Validation)]
                         │ (GATE: Must demonstrate meaningful recall improvement)
                         ▼
[Phase 4: Conditional ARP Integration & Safety Guards]
                         │
                         ▼
[Phase 5: Regression & 1080p Scaling Verification]
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|---|---|---|---|---|
| 1 | Phase 1 | Sequential | 1 (`coder`) | Isolate actual missed scratch frames/region on *Neighbors_72.mkv*, trace failure stages |
| 2 | Phase 2 | Sequential | 1 (`coder`) | Standalone `TemporalScratchAnalyzer`, OpenCV motion-compensated reference, short-lived defect logic |
| 3 | Phase 3 | Sequential | 1 (`coder`) | **Quality Gate**: Compare spatial-only vs temporal-only vs fused against GT ROI, intact FPR verification |
| 4 | Phase 4 | Sequential | 1 (`coder`) | Conditional integration into `DefectDetectorRegistry`, `MultiChannelDefectMask`, 30% guard |
| 5 | Phase 5 | Sequential | 1 (`tester`) | Pytest regression suite, full 72-frame run, 1080p scaling, CPU/GPU flow profiling |

---

## Phase 1: Audit and Ground Truth Identification

### Objective
Examine all 72 frames of the real `evaluation/Neighbors_72.mkv` clip to locate the exact frame indices and pixel bounding boxes of the missed long scratch. Trace the existing detection pipeline across all stages (raw probability maps, binarized scratch masks, uncertain masks, and effective repair masks) for DART, FAS, BOPB, CNN Blotches, PatchCore, PaDiM, Farneback, and Consensus to determine exactly where and why the scratch is lost.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/audit_neighbors_scratches.py` — Diagnostic script tracing detector outputs per stage and logging intermediate tensor values and data types.
- `docs/benchmarks/temporal_scratch_audit.json` — Empirical log of per-frame scratch coordinates, stage-by-stage defect loss categorization, and verified GT ROI.

### Implementation Details
- Extract frames 0–71 of `evaluation/Neighbors_72.mkv`.
- Measure Sobel vertical gradient profile, line continuity, and frame-to-frame intensity differences to locate the persistent/intermittent long vertical scratches.
- Trace raw probability/anomaly scores, thresholding, morphological filtering (e.g., `component_h <= 0.60 * h`), uncertain channel extraction, and repair policy mask generation.
- Record exact failure mode: raw sensitivity failure vs threshold cutoff vs morphological rejection vs repair exclusion.
- Establish verified scratch ROI and intact background ROI for quantitative evaluation.

### Validation
- Script executes and produces `docs/benchmarks/temporal_scratch_audit.json` containing frame numbers, ROI coordinates, and failure point analysis.

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: Isolated Temporal PoC (`TemporalScratchAnalyzer`)

### Objective
Implement a standalone, testable `TemporalScratchAnalyzer` module using OpenCV infrastructure. Register neighboring frames into the reference coordinate system, construct a robust motion-compensated background reference (excluding the target frame), detect transient/drifting outliers with adaptive noise thresholds, and preserve short-lived scratches lasting only 2–3 frames without requiring temporal persistence.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/temporal_scratch_analyzer.py` — Standalone class `TemporalScratchAnalyzer`:
  - Sliding window buffer ($R \in [1, 5]$, default $R=3 \implies 7$ frames).
  - Dense optical flow via OpenCV `DISOpticalFlow` (with fallback to `calcOpticalFlowFarneback`).
  - Backward warping of neighboring frames to reference frame coordinates.
  - Robust temporal background synthesis: trimmed median / weighted reference excluding target frame $t$.
  - Adaptive MAD noise floor estimation ($\mu_{grain} + \lambda \cdot \sigma_{MAD}$).
  - Scratch-oriented spatial evidence (vertical structure enhancement, line continuity).
  - Motion-coherence filtering: distinguish camera/scene motion from stationary/drifting film scratches.
  - Short-lived scratch preservation: logic allowing defects present in 2–3 consecutive frames to be captured without being suppressed as background.

### Implementation Details
- Self-contained implementation without modifying GUI or production registry yet.
- Strict data type hygiene: all masks returned as contiguous `uint8` with values strictly in $\{0, 255\}$.

### Validation
- Standalone verification script runs on sample frames and outputs non-empty, high-continuity scratch proposals.

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 3

---

## Phase 3: Quantitative Quality Gate (Real Neighbors Validation)

### Objective
**Hard Quality Gate**: Before modifying GUI code or integrating into the production pipeline, test the PoC on the real 72-frame *Neighbors* clip. Compare spatial-only, temporal-only, and fused results against the manually verified scratch ROI and intact background regions. Prove that the analyzer delivers a meaningful improvement in scratch recall without unacceptable false positives on intact areas.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/benchmark_temporal_quality_gate.py` — Comprehensive evaluation script calculating:
  - Scratch Recall inside target ROI.
  - Precision and Pixel-F1 / IoU.
  - Intact-Pixel False Alarm Rate (FPR) on clean background regions.
  - Total frame mask coverage vs 30% guardrail limit.
  - Line continuity and connected component statistics.
- `docs/benchmarks/temporal_scratch_quality_gate.json` — Quantitative metrics before and after temporal enhancement.
- Visual proof panel: `docs/benchmarks/visuals/temporal_scratch_quality_gate.png` showing raw frames, warped references, spatial masks, temporal candidate masks, and fused repair masks.

### Implementation Details
- Evaluate across multiple baseline spatial detectors (FAS, PatchCore, BOPB, CNN Blotches).
- Test fusion strategies (Logical OR, confidence-weighted blending, spatial confirmation).
- Gate Decision: Integration proceeds to Phase 4 ONLY IF scratch recall measurably improves (e.g. >25% absolute gain on the missed scratch) and intact FPR remains low (<3.0%).

### Validation
- Output JSON report and visual panel saved. Explicit verification of the quality criteria.

### Dependencies
- Blocked by: Phase 2
- Blocks: Phase 4

---

## Phase 4: Conditional ARP Integration & Safety Guards

### Objective
Following successful quality gate clearance, integrate `TemporalScratchAnalyzer` into ARP as an optional, non-destructive detector and candidate enhancement branch. Preserve existing detector outputs, `MultiChannelDefectMask` contracts, preview identity, repair policies, and the 30% overmasking safety guard.

### Agent: `coder`
### Parallel: No

### Files to Create
- `scripts/temporal_defect_adapter.py` — Pluggable adapter implementing `BaseDefectDetector`.

### Files to Modify
- `scripts/film_defect_detectors.py` — Register `temporal_scratch` detector and add temporal enhancement hook for spatial detectors.
- Maintain `MultiChannelDefectMask` invariants: place temporal scratch in `scratch_mask` and `combined_mask`. Never silently add unclassified temporal pixels to `uncertain_mask` or effective repair mask unless configured by policy.
- Enforce the 30% frame coverage failsafe: automatically throttle or suppress temporal addition if total mask exceeds 30%, logging diagnostic warnings.

### Validation
- Verify `pytest tests/test_defect_detector_registry.py` and `pytest tests/test_dust_stage_data_api.py` pass cleanly.

### Dependencies
- Blocked by: Phase 3
- Blocks: Phase 5

---

## Phase 5: Regression and 1080p Scaling Verification

### Objective
Run complete regression tests on the temporal analyzer and pipeline integration. Verify the full 72-frame video pipeline without crashes or data-type regressions. Test performance scaling at 1080p ($1440 \times 1080$), verify actual OpenCV backend (CPU vs OpenCL/CUDA), and document timings and peak memory.

### Agent: `tester`
### Parallel: No

### Files to Create
- `tests/test_temporal_scratch_analyzer.py` — Automated test suite:
  - Unit tests for sliding window buffering, frame warping, and boundary handling.
  - Motion-coherence tests rejecting moving foreground objects.
  - Short-lived 2–3 frame defect preservation tests.
  - Strict `uint8` shape and dtype invariant tests.
  - 30% coverage guardrail enforcement tests.
- `docs/benchmarks/temporal_scratch_scaling_1080p.json` — Performance and memory profile comparing 720p ($960 \times 720$) vs 1080p ($1440 \times 1080$), documenting actual OpenCV backend execution mode.

### Validation
- `python -m pytest tests/test_temporal_scratch_analyzer.py` passes 100%. Full 72-frame pipeline run completes without errors.

### Dependencies
- Blocked by: Phase 4
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|---|---|---|
| 1 | `scripts/audit_neighbors_scratches.py` | 1 | Audit script inspecting failure stages on *Neighbors* |
| 2 | `docs/benchmarks/temporal_scratch_audit.json` | 1 | Audit findings and scratch ROI coordinates |
| 3 | `scripts/temporal_scratch_analyzer.py` | 2 | Core motion-compensated temporal scratch detection engine |
| 4 | `scripts/benchmark_temporal_quality_gate.py` | 3 | Quality gate benchmarking harness |
| 5 | `docs/benchmarks/temporal_scratch_quality_gate.json` | 3 | Quality gate quantitative report |
| 6 | `scripts/temporal_defect_adapter.py` | 4 | Adapter wrapping TemporalScratchAnalyzer as BaseDefectDetector |
| 7 | `scripts/film_defect_detectors.py` | 4 | Registry registration and multichannel fusion integration |
| 8 | `tests/test_temporal_scratch_analyzer.py` | 5 | Pytest regression suite |
| 9 | `docs/benchmarks/temporal_scratch_scaling_1080p.json` | 5 | 1080p scaling and backend timing report |

---

## Risk Classification

| Phase | Risk | Rationale |
|---|---|---|
| Phase 1 | LOW | Non-destructive diagnostic analysis. |
| Phase 2 | MEDIUM | Dense optical flow tuning and motion coherence thresholding. |
| Phase 3 | HIGH | Quality Gate: determines whether temporal PoC is viable before touching production code. |
| Phase 4 | LOW | Pluggable adapter architecture ensures zero regression on existing models. |
| Phase 5 | LOW | Automated testing and scaling analysis. |

---

## Execution Profile

```
Execution Profile:
- Total phases: 5
- Parallelizable phases: 0
- Sequential-only phases: 5
- Estimated sequential wall time: ~10-15 minutes

Note: Quality validation occurs at Phase 3 prior to any pipeline modifications in Phase 4.
```
