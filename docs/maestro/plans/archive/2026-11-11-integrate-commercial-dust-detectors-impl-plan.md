---
title: "Commercial-Safe Dust & Scratch Detector Integration Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-11-integrate-commercial-dust-detectors-design.md"
created: "2026-11-11T14:59:00Z"
status: "draft"
total_phases: 4
estimated_files: 5
task_complexity: "complex"
---

# Commercial-Safe Dust & Scratch Detector Integration Implementation Plan

## Plan Overview

- **Total phases**: 4
- **Agents involved**: `coder`, `tester`, `code-reviewer`
- **Estimated effort**: Serialized integration across backend models, pipeline runner, frontend UI, and targeted regression tests.
- **Safety guarantee**: All shared files are edited sequentially by a single integration coder. Uncommitted work (Issue #37) is strictly preserved.

## Dependency Graph

```
Phase 1: Backend Registry & Schema Contracts
    │
    ▼
Phase 2: Pipeline Runner & Artifact Identity Dispatch
    │
    ▼
Phase 3: Frontend UI, Multi-Channel Tabs & Mask Comparison
    │
    ▼
Phase 4: Targeted Integration & Full Regression Testing
```

## Execution Strategy

| Stage | Phases | Execution | Agent | Notes |
|-------|--------|-----------|-------|-------|
| 1 | Phase 1 | Sequential | `coder` | Backend fields, schemas, and separate detector/repair validation |
| 2 | Phase 2 | Sequential | `coder` | Cleanup video pipeline runner and artifact identity hashing |
| 3 | Phase 3 | Sequential | `coder` | UI dropdown, limitation badges, and channel tabs in `render-dust.js` |
| 4 | Phase 4 | Sequential | `tester` | New integration test suite and full ARP regression suite |

---

## Phase 1: Backend Registry & Schema Contracts

### Objective
Expose commercial-safe detectors (`legacy`, `temporal`/`farneback_3f`, `dis_flow`, `ridge`, `consensus_5f`) in GUI stage models and ensure strict separation between defect detection and defect repair in options validation.

### Agent: `coder`
### Parallel: false

### Files to Modify
- `ai_remaster_gui/models.py` — Update `Stage("dust", ...)` field definition for `detector` to include all commercial detectors.
- `ai_remaster_gui/dust.py` — Validate supported detectors, preserve separate repair selection (Telea, ProPainter), and enrich detector metadata with channel and temporal capabilities.

### Implementation Details
- Expose commercial detectors in `models.py`: `select:temporal|farneback_3f|dis_flow|ridge|consensus_5f|legacy`.
- In `ai_remaster_gui/dust.py:options()`:
  - Treat missing `detector` as `temporal` for backward compatibility.
  - Normalize `temporal` $\leftrightarrow$ `farneback_3f`.
  - Validate that detector and repair methods remain completely decoupled (e.g., Telea and ProPainter are usable with any detector).
  - Enforce `commercial_model_guard` blocking DART.

### Validation
- `python -c "from ai_remaster_gui.dust import options; print(options({'detector': 'dis_flow', 'repair': 'telea'}))"`

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: Pipeline Runner & Artifact Identity Dispatch

### Objective
Ensure selected detectors dispatch correctly through `scripts/cleanup_video.py` and generate discrete cache signatures via `scripts/artifact_ids.py`.

### Agent: `coder`
### Parallel: false

### Files to Modify
- `scripts/cleanup_video.py` — Support `--defect-detector` dispatch through `DEFAULT_REGISTRY`.
- `scripts/artifact_ids.py` — Ensure `defect_detector` is hashed into cleanup identity and artifact names.

### Implementation Details
- Ensure `--defect-detector` cleanly accepts any registered commercial detector name or alias.
- Write detector metadata into the stage stamp JSON.
- Verify switching detectors invalidates preview and processing without deleting accepted prior outputs.

### Validation
- Run existing cleanup video unit tests: `pytest tests/test_cleanup_video.py -k test_identity`

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 3

---

## Phase 3: Frontend UI, Multi-Channel Tabs & Mask Comparison

### Objective
Implement the detector selection dropdown with honest limitation badges, multi-channel mask tabs (`Combined`, `Scratch`, `Dust`, `Overlay`), and preserve non-destructive A/B variant comparison.

### Agent: `coder`
### Parallel: false

### Files to Modify
- `ai_remaster_gui/static/js/render-dust.js` — Update detector labels, limitation notices, channel tabs, and preview rendering.

### Implementation Details
- Display user-friendly labels in `dustNames`:
  - `temporal` / `farneback_3f`: "Temporal Optical Flow (Farneback)"
  - `dis_flow`: "Dense Inverse Search (DIS Flow)"
  - `ridge`: "Multi-Scale Ridge (Scratches only · experimental)"
  - `consensus_5f`: "Temporal Consensus (5-frame)"
  - `legacy`: "ARP Legacy Scratch Mask"
- Display limitation notices (e.g., "Dust channel: N/A (scratch-only)" for Ridge, or "Requires >= 3 consecutive frames").
- Add channel selector buttons in the mask inspection viewport (`Combined`, `Scratch`, `Dust`, `Overlay`).
- Retrieve authentic channel masks from detector artifacts without placeholder fabrication.
- Preserve repair options (Telea, ProPainter) in the separate Repair section.

### Validation
- Syntax and lint validation: `node -c ai_remaster_gui/static/js/render-dust.js`

### Dependencies
- Blocked by: Phase 2
- Blocks: Phase 4

---

## Phase 4: Targeted Integration & Full Regression Testing

### Objective
Author comprehensive integration tests validating detector selection, backward compatibility, cache keys, and run full test suites.

### Agent: `tester`
### Parallel: false

### Files to Create
- `tests/test_dust_detector_gui_integration.py` — Comprehensive integration tests.

### Implementation Details
- Test all 5 commercial detectors for valid options parsing and registry dispatch.
- Test backward compatibility: legacy presets without detector key default to `temporal`.
- Test artifact ID separation: switching detector changes hash key.
- Test preview isolation: changing draft detector does not alter adopted pipeline output.
- Test commercial safety firewall: DART is blocked when `commercial_model_guard=True`.
- Test operational limitations: Ridge dust absence and short clips report diagnostics without silent substitution.
- Run complete test suite: `pytest tests/`.

### Validation
- `pytest tests/test_dust_detector_gui_integration.py -v`
- `pytest tests/` (1,037+ tests passing)

### Dependencies
- Blocked by: Phase 3
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `ai_remaster_gui/models.py` | 1 | Expose commercial detectors in GUI stage field schema |
| 2 | `ai_remaster_gui/dust.py` | 1 | Decouple detection from repair and enrich metadata |
| 3 | `scripts/cleanup_video.py` | 2 | Wire CLI/pipeline runner to detector registry |
| 4 | `scripts/artifact_ids.py` | 2 | Distinct artifact signatures per detector |
| 5 | `ai_remaster_gui/static/js/render-dust.js` | 3 | Detector dropdown, limitation badges, and channel tabs |
| 6 | `tests/test_dust_detector_gui_integration.py` | 4 | Targeted integration and regression suite |

---

## Risk Classification

| Phase | Risk | Rationale |
|---|---|---|
| Phase 1 | LOW | Non-breaking schema expansion with transparent fallbacks |
| Phase 2 | LOW | Purely additive CLI flag and identity key mapping |
| Phase 3 | MEDIUM | Frontend UI state must preserve existing canvas rendering and variant selection |
| Phase 4 | LOW | Isolated new test file and non-destructive suite execution |

---

## Execution Profile

```
Execution Profile:
- Total phases: 4
- Parallelizable phases: 0 (All 4 serialized to protect shared GUI/pipeline files)
- Sequential-only phases: 4
- Estimated sequential wall time: ~10-15 minutes
```
