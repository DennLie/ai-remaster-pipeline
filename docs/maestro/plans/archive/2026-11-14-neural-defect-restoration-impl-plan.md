---
title: "ARP Next-Generation Neural Dust & Scratch Detection Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-14-neural-defect-restoration-design.md"
created: "2026-11-14T20:36:40+01:00"
status: "approved"
total_phases: 10
estimated_files: 14
task_complexity: "complex"
---

# ARP — Next-Generation Neural Dust & Scratch Detection Implementation Plan

## Plan Overview

- **Total Phases**: 10
- **Agents Involved**: `coder` (Lead Integration Owner), `ml-engineer`, `tester`
- **Estimated Effort**: Enterprise-grade multi-model neural vision integration, active training pipelines, empirical benchmarking, GUI wiring, and end-to-end restoration validation.
- **Single Integration Owner Guard**: `coder` exclusively manages writes to shared core files (`scripts/film_defect_detectors.py`, `ai_remaster_gui/dust.py`, `ai_remaster_gui/dust_jobs.py`).

## Dependency Graph

```
Phase 1: MultiChannelDefectMask Contract & Uncertainty (coder)
   ├──> Phase 2: Contract Unit Tests & Legacy Suite (tester) ────────────────────────────────────────────────────────┐
   ├──> Phase 3: BOPB Scratch Detector Runner & Checkpoint Licensing Audit (ml-engineer)                             │
   │       └──> Phase 5: Adaptive Tiling Engine & VRAM Protection (coder)                                            │
   │               │                                                                                                 │
   └──> Phase 4: Neural Candidates Architecture & Active Model Training (ml-engineer)                                │
           └──┬────┘                                                                                                 │
              └──> Phase 6: Comparative Defect Benchmarking & Preset Calibration (tester)                            │
                      └──> Phase 7: DefectDetectorRegistry Expansion, Calibrated Presets & Provenance (coder)        │
                              └──> Phase 8: GUI Workspace Integration & A/B Slider (coder)                           │
                                      └──> Phase 9: End-to-End Restoration Integration & Mask Adoption (coder)       │
                                              └──┬───────────────────────────────────────────────────────────────────┘
                                                 └──> Phase 10: Comprehensive Integration & Regression Suite (tester)
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
| :---: | :---: | :---: | :---: | :--- |
| **Stage 1** | Phase 1 | Sequential | 1 | Foundation: Contract, validation, bitwise union, uncertainty |
| **Stage 2** | Phase 2, 3, 4 | Parallel / Sequential | 3 | Contract test suite, BOPB isolation, Active neural candidate training |
| **Stage 3** | Phase 5 | Sequential | 1 | Adaptive high-resolution tiling engine with Gaussian blending |
| **Stage 4** | Phase 6 | Sequential | 1 | Real comparative benchmarking & empirical preset calibration |
| **Stage 5** | Phase 7 | Sequential | 1 | Registry expansion, 4-tier readiness lifecycle, calibrated presets |
| **Stage 6** | Phase 8 | Sequential | 1 | GUI workspace integration, A/B viewport split-slider, Commercial Safe |
| **Stage 7** | Phase 9 | Sequential | 1 | End-to-end mask adoption and inpainting (Telea/ProPainter) handoff |
| **Stage 8** | Phase 10 | Sequential | 1 | Full integration verification, non-elevated Windows checks, regressions |

---

## Phase Details

### Phase 1: MultiChannelDefectMask Contract & Safe Uncertainty Handling
* **Objective**: Implement the `MultiChannelDefectMask` contract, strict validation (2D, dtypes, finite values, [0, 1] range, strict {0, 255} binary set), bitwise OR combination, and policy-aware legacy bridge. Replace DART's unclassified-as-dust flaw with explicit uncertainty handling.
* **Agent**: `coder`
* **Parallel**: No
* **Files to Modify**: `scripts/film_defect_detectors.py`
* **Implementation Details**:
  - Implement `ReadinessState`, `ChannelOrigin`, `RepairInclusionPolicy`, `UnsupportedPolicyError`, `MaskContractValidationError`, `ChannelProvenance`.
  - Implement `MultiChannelDefectMask` with zero silent policy overrides.
  - Implement `to_legacy_compat_bridge(policy)` with JSON-serializable manifest and invariant assertion that `bitwise_or(scratch, dust)` is 100% pixel-identical to `get_effective_repair_mask(policy)`.
  - Update DART defect decomposition to route unclassified components meeting calibrated criteria to `uncertain` rather than `dust`.
* **Validation**: Run Python verification on two-pixel scenario and invalid value assertions.
* **Dependencies**: Blocked by: None; Blocks: Phase 2, 3, 4.

### Phase 2: Contract Unit Tests & Legacy Compatibility Suite
* **Objective**: Deliver comprehensive unit tests validating the mask contract against all edge cases and legacy requirements.
* **Agent**: `tester`
* **Parallel**: No (depends on Phase 1)
* **Files to Create**: `tests/test_multichannel_defect_mask.py`
* **Implementation Details**:
  - Test all 4 policies (`CONSERVATIVE`, `HIGH_RECALL`, `SCRATCH_ONLY`, `COMBINED_DEFECT`).
  - Two-pixel regression scenario: Scratch=[255, 0], Dust=[0, 0], Uncertain=[0, 255].
  - Validate exception raising for NaN/Inf, non-binary masks, shape mismatches, and unsupported policies.
  - Verify pixel-identical legacy union for all policies and combined-only models.
* **Validation**: `pytest tests/test_multichannel_defect_mask.py`
* **Dependencies**: Blocked by: Phase 1; Blocks: Phase 10.

### Phase 3: BOPB Scratch Detector Runner & Checkpoint Licensing Audit
* **Objective**: Isolate Microsoft Bringing Old Photos Back to Life scratch detector (`Global/detection.py`), verify upstream weights and license, and implement a standalone pinned CLI runner.
* **Agent**: `ml-engineer`
* **Parallel**: No
* **Files to Create**: `tools/bopb/run_bopb_detector.py`, `scripts/setup_bopb.py`
* **Implementation Details**:
  - Implement `setup_bopb.py` managing the isolated virtualenv and downloading/verifying official weights into `tools/checkpoints/` with SHA-256 validation.
  - Implement `run_bopb_detector.py` executing standalone scratch detection on input frames, exporting continuous float32 scratch probabilities and binary masks.
  - Record non-commercial research license attribution in checkpoint manifest.
* **Validation**: Run `python scripts/setup_bopb.py --check` and verify test inference on a sample frame.
* **Dependencies**: Blocked by: Phase 1; Blocks: Phase 5.

### Phase 4: Neural Candidates Architecture Reproduction & Active Model Training
* **Objective**: Reconstruct architectures and execute active model training and memory-bank fitting for research candidates where data and compute permit.
* **Agent**: `ml-engineer`
* **Parallel**: No
* **Files to Create**: `tools/neural_defects/models.py`, `tools/neural_defects/train_defect_segmenter.py`, `tools/anomalib/run_anomaly_detector.py`
* **Implementation Details**:
  - Implement clean-room PyTorch architectures: Film Artefact Segmenter (Ivanova et al.), Correl-Net (Renaudeau et al.), and CNN Blotches & Scratches (Yous/Serir).
  - Implement reproducible training pipelines with fixed seeds and validation loss logging using existing ARP datasets (`FilmDamageSimulator`, `AbsoluteDegradation`).
  - Implement PatchCore and PaDiM feature memory-bank fitting and anomaly localization using clean archival film patches.
  - Execute training/fitting runs to generate candidate checkpoints; validate on independent held-out frames.
  - Record training logs, curves, and honest readiness states (`TRAINED_EXPERIMENTAL`, `TRAINING_REQUIRED`, or `BLOCKED`).
* **Validation**: Verify checkpoint generation, loss convergence, and inference execution on validation frames.
* **Dependencies**: Blocked by: Phase 1; Blocks: Phase 6.

### Phase 5: Adaptive Tiling Engine & VRAM Protection
* **Objective**: Implement `AdaptiveTilingEngine` supporting 1080p/2K/4K film scans without seam artifacts or CUDA OOM.
* **Agent**: `coder`
* **Parallel**: No
* **Files to Create**: `scripts/adaptive_tiling.py`
* **Files to Modify**: `scripts/film_defect_detectors.py`
* **Implementation Details**:
  - Model-specific tile sizes (256px BOPB, 512px DART/segmenter), context padding, and overlap.
  - Distance/Gaussian weighted probability blending across overlapping seams *prior* to thresholding.
  - Spatially locked tile coordinates across temporal frame sequences.
  - Dynamic VRAM budgeting with explicit non-silent logging for batch throttling or CPU fallback.
* **Validation**: Run tiling tests on 1080p and 4K synthetic frames; verify seam continuity and memory bounds.
* **Dependencies**: Blocked by: Phase 3; Blocks: Phase 6.

### Phase 6: Comparative Defect Benchmarking & Preset Calibration
* **Objective**: Run comparative benchmarks of all operational models (DART, BOPB, and newly trained models from Phase 4) against classical baselines, calibrating optimal thresholds and presets.
* **Agent**: `tester`
* **Parallel**: No
* **Files to Create**: `scripts/run_defect_benchmark_suite.py`
* **Implementation Details**:
  - Reuse existing benchmark datasets (AbsoluteDegradation, synthetic scratches/dust, clean negative controls with faces/architecture/motion).
  - Measure Scratch/Dust Precision, Recall, F1, IoU, clean-region false-positive rate, temporal consistency, latency, and VRAM.
  - Maintain strict separation between synthetic ground truth, manual annotations, and pseudo-labels.
  - Empirically calibrate optimal thresholds for Fast Preview, Balanced, High Precision, High Recall, and Archival Preservation.
* **Validation**: Generate structured benchmark report and calibrated preset parameter dictionary.
* **Dependencies**: Blocked by: Phase 4, Phase 5; Blocks: Phase 7.

### Phase 7: DefectDetectorRegistry Expansion, Calibrated Presets & Provenance
* **Objective**: Register all 7 neural approaches with explicit 4-tier readiness states and integrate empirical presets.
* **Agent**: `coder`
* **Parallel**: No
* **Files to Modify**: `scripts/film_defect_detectors.py`
* **Implementation Details**:
  - Register DART, BOPB, Correl-Net, Film Artefact Segmenter, CNN Blotches, PatchCore, and PaDiM in `DefectDetectorRegistry`.
  - Classify readiness: `PRETRAINED_OPERATIONAL` (DART, BOPB), validated `TRAINED_EXPERIMENTAL` models, and `TRAINING_REQUIRED` / `BLOCKED`.
  - Preserve all 10 existing ARP presets intact.
  - Add benchmark-backed presets for operational models (e.g. `bopb_scratch_balanced`, `bopb_scratch_high_recall`, `dart_archival_conservative`).
  - Attach full SHA-256 checkpoint hashes, license metadata, and `RepairInclusionPolicy`.
* **Validation**: Run registry validation tests ensuring all models register correctly and active selector gates untrained models.
* **Dependencies**: Blocked by: Phase 6; Blocks: Phase 8.

### Phase 8: GUI Workspace Integration & Interactive A/B Comparison
* **Objective**: Update Dust & Scratches GUI with tiered dropdown, readiness badges, A/B viewport split-slider, and Commercial Safe toggle.
* **Agent**: `coder`
* **Parallel**: No
* **Files to Modify**: `ai_remaster_gui/dust.py`, `ai_remaster_gui/dust_jobs.py`, `ai_remaster_gui/static/js/render-dust.js`
* **Implementation Details**:
  - Tiered optgroups in detector selector (`Classical`, `Neural Baseline`, `Neural Candidates`) with live readiness badges.
  - Interactive A/B viewport split-slider for comparative review (Model A vs B, or Raw vs Mask vs Restored).
  - Dynamic `Commercial Safe` toggle gating research-only models (BOPB) with clear license attribution.
  - Multi-channel preview toggles (Combined, Scratch, Dust, Uncertain).
  - Backend job dispatch wiring selected model, preset, and parameters.
* **Validation**: Verify GUI endpoints, parameter serialization, and preview rendering.
* **Dependencies**: Blocked by: Phase 7; Blocks: Phase 9.

### Phase 9: End-to-End Restoration Integration & Mask Adoption
* **Objective**: Connect adopted masks to downstream inpainting backends (OpenCV Telea, ProPainter, DART composite) with full provenance.
* **Agent**: `coder`
* **Parallel**: No
* **Files to Modify**: `ai_remaster_gui/dust_jobs.py`, `scripts/dust_native_composite.py`
* **Implementation Details**:
  - Deliver exact effective mask resolved by `RepairInclusionPolicy` to the chosen inpainting backend.
  - Guarantee preview adjustments never overwrite accepted masks.
  - Maintain independent cache identities incorporating detector ID, checkpoint SHA-256, and policy.
  - Persist complete provenance manifest into project state and artifact cache.
* **Validation**: Execute real video inpainting run producing restored video artifacts.
* **Dependencies**: Blocked by: Phase 8; Blocks: Phase 10.

### Phase 10: Comprehensive Integration, Benchmark & Regression Test Suite
* **Objective**: Execute end-to-end integration tests, non-elevated Windows checks, and full regression test suite.
* **Agent**: `tester`
* **Parallel**: No
* **Files to Create**: `tests/test_neural_defect_detectors.py`, `tests/test_end_to_end_restoration_pipeline.py`
* **Implementation Details**:
  - Verify non-elevated Windows execution and path safety.
  - Real neural inference tests on sample film clips for all operational models.
  - Verify effective repair mask propagation to Telea and ProPainter.
  - Validate Commercial Safe filtering and provenance manifests.
  - Run full existing ARP regression tests (`test_dust_stage.py`, `test_dart_exports.py`, `test_ai_defect_detectors_pipeline.py`, etc.).
* **Validation**: 100% test pass rate across new suites and legacy regression tests.
* **Dependencies**: Blocked by: Phase 2, Phase 9; Blocks: Completion.

---

## File Inventory

| # | File | Phase | Primary Owner | Purpose |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `scripts/film_defect_detectors.py` | 1, 5, 7 | `coder` | MultiChannelDefectMask, Adaptive Tiling integration, DefectDetectorRegistry expansion, Presets |
| 2 | `tests/test_multichannel_defect_mask.py` | 2 | `tester` | Unit tests for mask contract, policies, two-pixel regression, and legacy compatibility |
| 3 | `scripts/setup_bopb.py` | 3 | `ml-engineer` | Managed installation script for tools/bopb virtualenv and checkpoint downloading |
| 4 | `tools/bopb/run_bopb_detector.py` | 3 | `ml-engineer` | Standalone CLI runner for BOPB scratch detection |
| 5 | `tools/neural_defects/models.py` | 4 | `ml-engineer` | Clean-room architectures for Correl-Net, Film Artefact Segmenter, and CNN Blotches |
| 6 | `tools/neural_defects/train_defect_segmenter.py` | 4 | `ml-engineer` | Reproducible training pipeline with FilmDamageSimulator / AbsoluteDegradation loaders |
| 7 | `tools/anomalib/run_anomaly_detector.py` | 4 | `ml-engineer` | Memory bank fitting and anomaly localization runner for PatchCore and PaDiM |
| 8 | `scripts/adaptive_tiling.py` | 5 | `coder` | Adaptive overlapping tiling engine with pre-threshold Gaussian probability blending |
| 9 | `scripts/run_defect_benchmark_suite.py` | 6 | `tester` | Comparative benchmarking suite measuring Precision, Recall, F1, IoU, FP, latency, and VRAM |
| 10 | `ai_remaster_gui/dust.py` | 8 | `coder` | GUI backend endpoints for model selector, readiness states, and parameter dispatch |
| 11 | `ai_remaster_gui/dust_jobs.py` | 8, 9 | `coder` | Restoration job runner, mask adoption, inpainting dispatch (Telea/ProPainter) |
| 12 | `ai_remaster_gui/static/js/render-dust.js` | 8 | `coder` | UI components: tiered optgroups, readiness badges, A/B viewport split-slider, channel toggles |
| 13 | `scripts/dust_native_composite.py` | 9 | `coder` | Pipeline integration for mask handoff and provenance persistence |
| 14 | `tests/test_neural_defect_detectors.py` | 10 | `tester` | Integration tests for neural model inference, readiness gating, and tiling |
| 15 | `tests/test_end_to_end_restoration_pipeline.py` | 10 | `tester` | End-to-end video repair tests, mask-to-repair fidelity, and provenance verification |

## Execution Profile

```
Execution Profile:
- Total phases: 10
- Parallelizable phases: 3 (Phase 2, 3, 4 in Stage 2)
- Sequential-only phases: 7
- Estimated parallel wall time: Optimized across ML training, contract testing, and tiling
- Estimated sequential wall time: Serialized execution across single integration owner
```
