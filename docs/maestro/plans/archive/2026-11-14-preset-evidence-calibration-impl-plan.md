---
title: "AI Dust & Scratch Preset Evidence Calibration Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-14-preset-evidence-calibration-design.md"
created: "2026-11-14T19:37:00+01:00"
status: "approved"
total_phases: 2
estimated_files: 1
task_complexity: "simple"
---

# AI Dust & Scratch Preset Evidence Calibration Implementation Plan

## Plan Overview

- **Total phases**: 2
- **Agents involved**: coder, tester
- **Estimated effort**: Calibrate single preset evidence string and run exhaustive regression validation.

## Dependency Graph

```
+-----------------------------------------------------------+
| Phase 1: Calibrate Preset Evidence Description (coder)     |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
| Phase 2: Comprehensive Test Suite Verification (tester)   |
+-----------------------------------------------------------+
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|---|---|---|---|---|
| 1 | Phase 1 | Sequential | 1 | Edit `scripts/film_defect_detectors.py` |
| 2 | Phase 2 | Sequential | 1 | Execute regression tests across all stages |

---

## Phase 1: Calibrate Preset Evidence Description

### Objective
Update `dart_optimal_f1` preset in `scripts/film_defect_detectors.py` to specify synthetic localization F1 (threshold 224) vs. AbsoluteDegradation scratch recall (test clips 010 & 020).

### Agent: coder
### Parallel: No

### Files to Modify
- `scripts/film_defect_detectors.py` — Update `dart_optimal_f1` `evidence` field.

### Validation
- Syntax check and unit test verification.

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: Comprehensive Test Suite Verification

### Objective
Verify full test suite passes cleanly, confirming no regressions in preset resolution, pipeline execution, or GUI smoke tests.

### Agent: tester
### Parallel: No

### Validation
- `tests/test_ai_defect_detectors_pipeline.py`
- `tests/test_dust_stage.py`
- `tests/test_defect_detector_registry.py`
- `tests/test_commercial_defect_detectors_integration.py`
- `tests/test_artifact_permissions_regression.py`
- `tests/test_permission_error_hydration.py`
- `tests/test_defect_class_assignment_audit.py`
- `tests/test_gui_smoke.py`

### Dependencies
- Blocked by: Phase 1
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|---|---|---|
| 1 | `scripts/film_defect_detectors.py` | 1 | Calibrate `dart_optimal_f1` evidence text |

## Risk Classification

| Phase | Risk | Rationale |
|---|---|---|
| 1 | LOW | Minor text clarification in preset metadata. |
| 2 | LOW | Pure test suite execution and validation. |

## Execution Profile

```
Execution Profile:
- Total phases: 2
- Parallelizable phases: 0
- Sequential-only phases: 2
- Estimated sequential wall time: ~90s
```
