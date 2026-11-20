---
title: "AI Dust & Scratch Preset Evidence Calibration and Acceptance Alignment Design"
created: "2026-11-14T19:35:00+01:00"
status: "approved"
authors: ["TechLead", "User"]
type: "design"
design_depth: "standard"
task_complexity: "simple"
---

# AI Dust & Scratch Preset Evidence Calibration and Acceptance Alignment Design

## Problem Statement

Following the full integration of commercial and neural defect detectors (DART MaskNet, Farneback 3F, DIS Flow, Temporal Ridge, Consensus 5F) and 10 benchmark-calibrated presets into the ARP Dust & Scratches pipeline and GUI, an independent acceptance audit identified specific points requiring empirical clarification and evidence precision:
1. **Metric Ambiguity**: The preset `dart_optimal_f1` cited both `F1 = 0.9032 (synthetic)` and `0.7802 (AbsoluteDegradation)` without specifying the exact metric types (synthetic localization F1 vs. photographic scratch recall) or test configurations.
2. **Defect Channel Architecture**: Clarifying that DART MaskNet outputs a continuous single-class probability gate, and that the scratch/dust channel split is performed via morphological postprocessing rather than native multi-class neural heads.
3. **Traceability Alignment**: Pointers to historical research, benchmark suites, and licensing inventories must be consolidated so all preset claims are directly verifiable against held-out benchmark artifacts.

## Requirements

### Functional Requirements
- **REQ-1 (Evidence Precision)**: Update `dart_optimal_f1` preset metadata in `scripts/film_defect_detectors.py` to explicitly describe `F1 = 0.9032 (synthetic localization at threshold 224)` and `78.8% scratch recall (AbsoluteDegradation test clips 010 & 020)`.
- **REQ-2 (Zero Regression)**: Ensure all 10 presets continue to resolve deterministically, preserve idempotence (`options(options(v)) == options(v)`), and maintain correct repair artifact bindings (`effective_mask.mkv`).
- **REQ-3 (Commercial & Licensing Guard)**: Verify that commercial guard invariants remain intact (DART blocked under `commercial_model_guard=true`).

### Non-Functional Requirements
- **REQ-N1 (Windows NTFS Safety)**: Maintain inherited permissions on all generated artifacts (`paths.staging_directory` and `paths.ensure_inherited_permissions`).
- **REQ-N2 (Preserve Working Tree)**: No resetting, cleaning, stashing, or overwriting of uncommitted files.

## Selected Approach

Update the single string in `scripts/film_defect_detectors.py` for the `dart_optimal_f1` preset to make the benchmark distinction crystal clear. Then execute the full test suite to guarantee 100% regression-free behavior across GUI, CLI, and pipeline.

## Component Diagram & Data Flow

```
+-------------------------------------------------------------------+
|                        BENCHMARK_PRESETS                          |
|  - balanced (Farneback 3F)        - dart_optimal_f1 (DART T224)   |
|  - fast_dis (DIS Flow)            - dart_high_recall (DART T192)  |
|  - persistent_scratches (Ridge)   - dart_conservative (DART T240) |
|  - heavy_damage (Consensus 5F)    - archive_legacy                |
|  - motion_safe                    - pure_scratch_removal          |
+---------------------------------+---------------------------------+
                                  |
                   scripts/film_defect_detectors.py
                                  |
                                  v
                   ai_remaster_gui/dust.py: options()
                                  |
                   +--------------+---------------+
                   |                              |
                   v                              v
            effective_mask.mkv              repair method
       (Binary defect mask gate)        (Telea / ProPainter / DART)
```

## Agent Team

| Phase | Agent | Deliverables |
|---|---|---|
| Phase 1 | coder | Update `scripts/film_defect_detectors.py` with calibrated evidence string |
| Phase 2 | tester | Run complete regression test suite (`test_ai_defect_detectors_pipeline.py`, `test_dust_stage.py`, `test_gui_smoke.py`, etc.) |

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|---|---|---|---|
| String mismatch in tests | LOW | LOW | Audited codebase: `0.7802` string is not hardcoded in test assertions. |
| Working tree pollution | LOW | LOW | Targeted editing only on `scripts/film_defect_detectors.py`. |

## Success Criteria

1. `scripts/film_defect_detectors.py` contains calibrated evidence for `dart_optimal_f1`.
2. All regression test suites pass without error.
3. Maestro orchestration session completes cleanly and archives state.
