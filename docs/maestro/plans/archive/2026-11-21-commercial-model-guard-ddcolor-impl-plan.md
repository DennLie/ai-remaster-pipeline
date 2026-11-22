---
title: "Commercial Model Guard Upgrade & DDColor Reference Integration Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-21-commercial-model-guard-ddcolor-design.md"
created: "2026-11-21T18:14:30+01:00"
status: "draft"
total_phases: 4
estimated_files: 11
task_complexity: "complex"
---

# Commercial Model Guard Upgrade & DDColor Reference Integration Implementation Plan

## Plan Overview

- **Total phases**: 4
- **Agents involved**: `coder`, `tester`
- **Estimated effort**: 11 files (6 new, 5 modified), 4 execution stages
- **Primary Goal**: Complete Phase 1 of the commercial film colorization roadmap: upgrade Commercial Model Guard to granular per-checkpoint/LoRA/encoder enforcement, set Commercial Mode ON by default with non-commercial benchmark override, produce cumulative `compliance_report.json` with EU AI Act Art. 50 disclosure, integrate DDColor (Apache-2.0) deterministic still reference draft generator, and establish a comprehensive CPU test suite.

## Dependency Graph

```
Phase 1: Policy Registry & Guard Engine (coder)
   │
   ├───► Phase 2: Compliance Reporting Engine (coder) ──┐
   │                                                     │
   └───► Phase 3: DDColor Reference Generator (coder) ───┼──► Phase 4: Verification Test Suites (tester)
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|---|---|---|---|---|
| 1 | Phase 1 | Sequential | 1 | Foundation: Registry config, policy evaluation engine, and settings default |
| 2 | Phase 2, Phase 3 | Parallel Eligible | 2 | Disjoint deliverables: Compliance reporter & DDColor generator |
| 3 | Phase 4 | Sequential | 1 | Complete test suite coverage across all new and updated components |

---

## Phase 1: Policy Registry & Guard Engine

### Objective
Create the versioned policy registry JSON (`configs/commercial_policy_registry.json`) and upgrade `ai_remaster_gui/commercial_policy.py` and `ai_remaster_gui/runtime_settings.py` so Commercial Mode is ON by default, non-commercial benchmarking requires an explicit opt-in override, and policy checks evaluate granular checkpoints, LoRAs, and encoders with fail-closed SHA-256 diagnostics.

### Agent: `coder`
### Parallel: No (Foundation)

### Files to Create
- `configs/commercial_policy_registry.json` — Version `1.0.0` policy registry defining `ALLOWED_OSI`, `CONDITIONAL_COMMERCIAL`, and `HARD_BLOCK_COMMERCIAL` with verified model IDs, source repositories, license tags, and conditional criteria schemas.

### Files to Modify
- `ai_remaster_gui/commercial_policy.py` — Refactor to load `configs/commercial_policy_registry.json`. Update `enabled()` to default to `True` unless `non_commercial_benchmark_override` is `"true"`. Implement granular component evaluation with computed SHA-256 diagnostics for un-audited files. Enforce hard blocks and conditional revenue checks.
- `ai_remaster_gui/runtime_settings.py` — Set `defaults["global"]["commercial_model_guard"] = "true"` and `defaults["global"]["non_commercial_benchmark_override"] = "false"`.

### Implementation Details
```python
@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    status: str
    license: str
    reason: str
    sha256: str | None = None
    component_id: str = ""

def evaluate_component(
    component_id: str,
    file_path: Path | None = None,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    conditional_attestations: Mapping[str, bool] | None = None,
) -> PolicyDecision: ...
```

### Validation
- Run existing commercial policy tests: `python -m unittest tests.test_commercial_policy`
- Ensure missing/corrupt registry fails closed.

### Dependencies
- Blocked by: None
- Blocks: Phase 2, Phase 3, Phase 4

---

## Phase 2: Compliance Reporting Engine

### Objective
Implement the automated compliance reporter in `ai_remaster_gui/compliance.py` and CLI script `scripts/compliance_report.py` to record multi-stage run executions into `output/compliance_report.json`, accumulating run history, calculating top-level `overall_commercial_compliant`, and generating an EU AI Act Art. 50 disclosure metadata block.

### Agent: `coder`
### Parallel: Yes (Depth 1, with Phase 3)

### Files to Create
- `ai_remaster_gui/compliance.py` — Core library for atomic read-modify-write of `output/compliance_report.json`. Captures model IDs, file paths, SHA-256 hashes, source URLs, licenses, guard status, RNG seeds (`seeds: Mapping[str, int]`), Color Bible version (nullable), conditional attestations, and EU AI Act Art. 50 disclosure metadata block.
- `scripts/compliance_report.py` — Standalone CLI tool to generate, inspect, and validate `compliance_report.json` for headless/batch runs.

### Implementation Details
```python
def append_compliance_record(
    output_dir: Path,
    stage: str,
    models: list[dict[str, Any]],
    settings: Mapping[str, Mapping[str, str]],
    seeds: Mapping[str, int] | None = None,
    color_bible_version: str | None = None,
    conditional_attestations: Mapping[str, bool] | None = None,
) -> Path: ...
```

### Validation
- Execute `python scripts/compliance_report.py --help`
- Generate sample compliance report in temporary directory and assert JSON schema validity.

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 4

---

## Phase 3: DDColor Reference Generator

### Objective
Integrate DDColor (Apache-2.0) as a fast, deterministic still reference draft generator with standalone CLI script `scripts/ddcolor_reference.py`, wrapper `wrappers/ddcolor_reference.bat`, and GUI integration in `ai_remaster_gui/models.py`, `ai_remaster_gui/server.py`, and `ai_remaster_gui/references.py`.

### Agent: `coder`
### Parallel: Yes (Depth 1, with Phase 2)

### Files to Create
- `scripts/ddcolor_reference.py` — Standalone PyTorch inference script supporting `--manifest`, `--input-image`, `--output`, `--model-path`, `--device` (auto/cuda/cpu), and `--dry-run`. Fails closed if model weights are missing without `--dry-run`.
- `wrappers/ddcolor_reference.bat` — Windows CMD/PowerShell wrapper calling `_python.bat` and `scripts/ddcolor_reference.py`.

### Files to Modify
- `ai_remaster_gui/models.py` — Update `Stage("references")` method selector: `"select:ddcolor|qwen|qwen21|openai"`.
- `ai_remaster_gui/server.py` — Update `_references_command` to dispatch `scripts/ddcolor_reference.py` when `method == "ddcolor"`.
- `ai_remaster_gui/references.py` — Support DDColor in reference generation routines and preview generation.

### Implementation Details
```python
def colorize_frame(
    image: np.ndarray,
    model_path: Path,
    device: str = "auto",
    dry_run: bool = False,
) -> np.ndarray: ...
```

### Validation
- Execute `python scripts/ddcolor_reference.py --dry-run --input-image tests/fixtures/test.png --output output/test_out.png`
- Verify wrapper execution via `wrappers/ddcolor_reference.bat --help`

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 4

---

## Phase 4: Verification Test Suites

### Objective
Author comprehensive `pytest` test suites verifying all Phase 1 constraints: hard blocks, fail-closed diagnostics with computed SHA-256, conditional revenue gating, cumulative compliance reporting schema, EU AI Act Art. 50 disclosure, and DDColor CLI/wrapper invocation with mocked inference.

### Agent: `tester`
### Parallel: No (Final Verification)

### Files to Create
- `tests/test_compliance_report.py` — Tests for `compliance.py` and `compliance_report.py`: schema verification, SHA-256 hashing of generated artifacts, cumulative multi-stage run appending, `overall_commercial_compliant` boolean evaluation, and EU AI Act Art. 50 metadata block integrity.
- `tests/test_ddcolor_reference.py` — Tests for `scripts/ddcolor_reference.py` and wrapper: CLI parameter parsing, `--dry-run` synthetic output, mocked neural inference on synthetic image, and fail-closed `FileNotFoundError` when weights are missing without `--dry-run`.

### Files to Modify
- `tests/test_commercial_policy.py` — Update and expand tests: Commercial Guard default ON, `non_commercial_benchmark_override` behavior, hard blocks (CMNET2, ColorMNet, Qwen-2.1, ProPainter, MiniMax H3, SAM 3.1), fail-closed on unknown LoRAs with filename + computed SHA-256 error diagnostics, and conditional LTX revenue gating.
- `tests/test_model_license_inventory.py` — Update test coverage to ensure GUI dropdown additions (`ddcolor`) remain in sync with audit records.

### Validation
- Run full pytest suite: `python -m pytest tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_ddcolor_reference.py tests/test_model_license_inventory.py`
- Verify 100% pass on CPU without network/GPU.

### Dependencies
- Blocked by: Phase 2, Phase 3
- Blocks: None

---

## File Inventory

| # | File | Phase | Status | Purpose |
|---|---|---|---|---|
| 1 | `configs/commercial_policy_registry.json` | 1 | NEW | Version 1.0.0 policy registry config |
| 2 | `ai_remaster_gui/commercial_policy.py` | 1 | MODIFY | Granular per-component guard engine |
| 3 | `ai_remaster_gui/runtime_settings.py` | 1 | MODIFY | Commercial guard defaults & override settings |
| 4 | `ai_remaster_gui/compliance.py` | 2 | NEW | Cumulative compliance report & EU AI Act disclosure |
| 5 | `scripts/compliance_report.py` | 2 | NEW | CLI compliance generator & validator |
| 6 | `scripts/ddcolor_reference.py` | 3 | NEW | Standalone DDColor still reference generator |
| 7 | `wrappers/ddcolor_reference.bat` | 3 | NEW | Windows CLI wrapper for DDColor reference |
| 8 | `ai_remaster_gui/models.py` | 3 | MODIFY | GUI references.method selector update |
| 9 | `ai_remaster_gui/server.py` | 3 | MODIFY | Server dispatch for DDColor references |
| 10 | `ai_remaster_gui/references.py` | 3 | MODIFY | GUI reference processing integration |
| 11 | `tests/test_commercial_policy.py` | 4 | MODIFY | Unit tests for upgraded guard & registry |
| 12 | `tests/test_compliance_report.py` | 4 | NEW | Unit tests for compliance reporting |
| 13 | `tests/test_ddcolor_reference.py` | 4 | NEW | Unit tests for DDColor CLI & mocked inference |
| 14 | `tests/test_model_license_inventory.py` | 4 | MODIFY | Inventory coverage tests sync |

---

## Risk Classification

| Phase | Risk | Rationale |
|---|---|---|
| Phase 1 | MEDIUM | Core policy logic change; must preserve compatibility with existing pipelines while defaulting to fail-closed safety. |
| Phase 2 | LOW | Purely additive logging and provenance reporting; non-intrusive to existing video processing. |
| Phase 3 | MEDIUM | Adds new model backend; must strictly fail closed on missing weights while allowing mocked tests in CI. |
| Phase 4 | LOW | Comprehensive test suite implementation; validates that all requirements are met and no regressions exist. |

## Execution Profile

```
Execution Profile:
- Total phases: 4
- Parallelizable phases: 2 (in 1 batch: Phase 2 & Phase 3)
- Sequential-only phases: 2 (Phase 1, Phase 4)
- Estimated parallel wall time: ~4-6 minutes
- Estimated sequential wall time: ~6-9 minutes

Note: Native subagents run without user approval gates during Phase 3.
```
