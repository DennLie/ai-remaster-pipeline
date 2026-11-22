---
title: "Commercial Model Guard Upgrade & DDColor Reference Integration Design"
created: "2026-11-21T18:12:00+01:00"
status: "approved"
design_depth: "standard"
task_complexity: "complex"
---

# Commercial Model Guard Upgrade & DDColor Reference Integration Design

## Problem Statement

DennLie/ai-remaster-pipeline (ARP) operates a multi-stage historical film remastering pipeline covering denoising, deflicker, defect restoration, outpainting, keyframe reference generation, video colorization, and upscaling. Prior to this upgrade, the repository's Commercial Model Guard was disabled by default (`commercial_model_guard: "false"`), coarse-grained at the stage level, and blocked almost all neural backends without differentiating between verified permissive open-source models, conditional commercial licenses, and strictly non-commercial research models.

Under the November 2026 legal framework (EU AI Act Article 50 transparency obligations and German copyright §§ 2, 23 UrhG) and production requirements for 60–120 minute feature film colorization:
1. Production runs must never silently invoke non-commercial models (such as CMNET2, ColorMNet, Qwen-Image-2.1, ProPainter, MiniMax H3, or SAM 3.1) which invalidate commercial distribution rights.
2. The pipeline requires Commercial Mode to be ON (`True`) by default across CLI, GUI, and batch runs, with a fail-closed architecture on any unknown or un-audited checkpoint, LoRA, or text encoder. Non-commercial models may only be loaded if an explicit non-commercial benchmarking override is affirmatively provided.
3. Every production render must automatically generate an immutable, machine-readable audit report (`compliance_report.json`) capturing exact model identifiers, SHA-256 hashes, source URLs, license tags, conditional attestations (e.g. LTX revenue thresholds), and an EU AI Act Art. 50 disclosure metadata block.
4. Fast, deterministic still reference draft generation is missing a certified OSI option: DDColor (Apache-2.0) must be integrated into CLI and GUI alongside Qwen-2511 to drastically reduce GPU draft rendering time from ~18s to ~0.4s per frame.

## Requirements

### Functional Requirements

- **REQ-1 (Default Commercial Guard & Override)**: `commercial_model_guard` defaults to `"true"` (Commercial Mode ON) in `base_settings()` and runtime configurations. In the GUI, the toggle defaults to ON; toggling it OFF displays a warning and automatically sets `non_commercial_benchmark_override: "true"` for that session/project. Headless CLI/batch scripts require `--allow-non-commercial-benchmark` to bypass the guard.
- **REQ-2 (Policy Registry)**: Encode a versioned JSON policy in `configs/commercial_policy_registry.json` specifying:
  - `ALLOWED_OSI`: DDColor (Apache-2.0), DeOldify (MIT), Qwen-Image-Edit-2511 (Apache-2.0, audited SHA-256/GGUF/LoRA), Qwen3-VL (Apache-2.0), SAM 2 / SAM 2.1 (Apache-2.0), ByteTrack (MIT), PySceneDetect (BSD-3-Clause).
  - `CONDITIONAL_COMMERCIAL`: LTX-2.3 / LTX-2.5 (requires `company_revenue_under_10m_usd: true` and separate audited entries for IC-LoRA + Gemma-3 encoder), Deep Exemplar (requires explicit weight/VGG19 audit flag).
  - `HARD_BLOCK_COMMERCIAL`: CMNET2, ColorMNet, Qwen-Image-2.1, ProPainter, MiniMax H3, SAM 3.1.
  - Fail-closed: Any model, LoRA, encoder, or checkpoint not explicitly whitelisted in the active mode is rejected with a fatal error.
  - Diagnostic rejection: When rejecting an un-audited local checkpoint, LoRA, or encoder, the error message must include the component name, file path, and its computed SHA-256 hash so engineers can audit and register legitimate Apache-2.0 quants.
- **REQ-3 (Compliance & Disclosure Reporting)**: Produce `compliance_report.json` in each run's `output/` directory. Rather than overwriting prior runs, it appends stage entries to a cumulative `runs` list, records RNG seeds (`seeds: Mapping[str, int]`), Color Bible version (nullable for now), conditional attestations, computes a top-level `overall_commercial_compliant` boolean, and includes an EU AI Act Art. 50 disclosure metadata block.
- **REQ-4 (DDColor Reference Integration)**: Provide `scripts/ddcolor_reference.py` and `wrappers/ddcolor_reference.bat`. Expose `ddcolor` in GUI `Stage("references")` method selector (`ddcolor|qwen|qwen21|openai`). Provide native CPU/GPU execution and deterministic fallback with mocked inference strictly gated behind `--dry-run` or pytest mocks. Missing weights during normal execution raise a fatal `FileNotFoundError`.

### Non-Functional Requirements

- **NFR-1 (GPU-Free Testing)**: The test suite must execute cleanly in CPU-only CI environments without requiring GPU hardware, external model downloads, or network access.
- **NFR-2 (Modular & Well-Typed)**: Zero placeholders, zero `TODO`s, zero silent fallbacks. Python 3.10+ type hints across all new and modified modules.
- **NFR-3 (High Test Coverage)**: Full pytest suite verifying hard blocks, fail-closed diagnostics, conditional gating, cumulative report schema & SHA-256 hashing, and DDColor CLI/wrapper invocation.

## Approach & Decision Matrix

### Selected Approach: Granular Policy Engine & Cumulative Audit Reporter

1. **Policy Specification**: Centralize rules in `configs/commercial_policy_registry.json` (schema version `1.0.0`). It contains explicit dictionaries for `ALLOWED_OSI`, `CONDITIONAL_COMMERCIAL`, and `HARD_BLOCK_COMMERCIAL` with required audit attributes (known SHA-256 fingerprints, source repo, license, conditional schema). Never fabricate placeholder SHA-256 hashes—only populate verified hashes from the audit.
2. **Evaluation & Rejection Diagnostics**: `ai_remaster_gui/commercial_policy.py` parses the registry and validates runtime selections against active settings. When rejecting an un-audited checkpoint, LoRA, or encoder, the exception explicitly reports the component type, filename, and computed SHA-256 hash.
3. **Cumulative Compliance Lifecycle**: `ai_remaster_gui/compliance.py` maintains `output/compliance_report.json`. Rather than clobbering earlier stages, it reads existing reports and appends stage execution entries to a cumulative `runs` array. An aggregate top-level boolean `overall_commercial_compliant` evaluates the entire pipeline's provenance, flipping to `false` if any stage in the project history used `non_commercial_benchmark_override` or an un-cleared model.
4. **DDColor Implementation**: `scripts/ddcolor_reference.py` provides standalone inference loading DDColor PyTorch weights. Missing weights raise a fatal error (`FileNotFoundError`) during production runs. Mocked inference is strictly gated behind `--dry-run` or explicit test mocks.

### Decision Matrix

| Criterion | Weight | Selected Approach (Modular Registry) | Monolithic Python Policy | External Validation Daemon |
|---|---|---|---|---|
| Granular Compliance & Auditability | 35% | 5/5: JSON registry, SHA-256 reporting, cumulative multi-stage run history | 3/5: Hardcoded Python logic, rigid to audit | 4/5: Strong audit but heavy operational overhead |
| Offline & CI Testability | 25% | 5/5: 100% offline, GPU-free mocked unit tests pass without weights | 4/5: Offline but tightly coupled | 2/5: Requires mock daemon in CI |
| Safety & Fail-Closed Integrity | 20% | 5/5: Commercial default True, fail-closed on unknown hashes, strict error messages | 3/5: Risk of missing checks on dynamic LoRAs | 4/5: High safety, high latency |
| Maintainability & Standards | 20% | 5/5: Pure Python standard library + PyTorch, zero TODOs, clean CLI + GUI wiring | 4/5: Python code edits needed for policy changes | 3/5: Multi-service complexity |
| **Weighted Total** | **100%** | **5.00** | **3.50** | **2.95** |

## Architecture

### Component Diagram

```
+----------------------------------------------------------------------------------------------------+
|                                    GUI / CLI / Batch Invocation                                    |
|   ai_remaster_gui / wrappers / scripts/*.py (with --allow-non-commercial-benchmark override flag)  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+────────────────────────────────────────────────────────────────────────────────────────────────────+
|                        Commercial Policy Engine (commercial_policy.py)                            |
|  - Reads configs/commercial_policy_registry.json (ALLOWED_OSI, CONDITIONAL, HARD_BLOCK)            |
|  - Validates exact Checkpoint, GGUF, LoRA, and Text-Encoder against active configuration           |
|  - Computes SHA-256 for unknown candidates & rejects with [component, filename, sha256]           |
|  - Evaluates conditional rules (company_revenue_under_10m_usd, weight_vgg19_audited)               |
+────────────────────────────────────────────────────────────────────────────────────────────────────+
                         │                                              │
                 [PASS]  ▼                                      [PASS]  ▼
+──────────────────────────────────────────────────────+  +──────────────────────────────────────────+
|           DDColor Reference Generator                |  |        Audited Qwen-2511 / SAM 2.1       |
|  - scripts/ddcolor_reference.py                      |  |  - Verified GGUF & Lightning LoRA        |
|  - wrappers/ddcolor_reference.bat                    |  |  - ComfyUI / PyTorch pipelines           |
|  - Deterministic PyTorch inference (GPU/CPU)         |  +──────────────────────────────────────────+
|  - Strict fail-closed on missing weights             |                        │
|  - --dry-run for mockable CI verification            |                        │
+──────────────────────────────────────────────────────+                        │
                         │                                                      │
                         └──────────────────────┬───────────────────────────────┘
                                                │
                                                ▼
+────────────────────────────────────────────────────────────────────────────────────────────────────+
|                           Audit & Provenance Engine (compliance.py)                               |
|  - Updates output/compliance_report.json cumulatively across all pipeline stages                   |
|  - Maintains runs: [] history with models, paths, hashes, license tags, and seeds                  |
|  - Computes top-level overall_commercial_compliant boolean                                         |
|  - Generates EU AI Act Art. 50 disclosure metadata block                                            |
+────────────────────────────────────────────────────────────────────────────────────────────────────+
```

### Key Signatures and Data Models

1. **Policy Evaluation Contract**:
   ```python
   @dataclass(frozen=True)
   class PolicyDecision:
       allowed: bool
       status: str  # "allowed_osi", "conditional_cleared", "conditional_blocked", "hard_blocked", "unknown"
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

2. **Compliance Reporting Contract**:
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

3. **DDColor Draft Invocation**:
   ```python
   def colorize_frame(
       image: np.ndarray,
       model_path: Path,
       device: str = "auto",
       dry_run: bool = False,
   ) -> np.ndarray: ...
   ```

## Agent Team

- **`compliance-reviewer`** (Read-Only Analysis): Verification of policy rules, license mappings, and EU AI Act Art. 50 disclosure metadata schema.
- **`coder`** (Implementation Specialist): Writes `configs/commercial_policy_registry.json`, refactors `ai_remaster_gui/commercial_policy.py`, implements `ai_remaster_gui/compliance.py` and `scripts/compliance_report.py`, authors `scripts/ddcolor_reference.py` and `wrappers/ddcolor_reference.bat`, and wires GUI and CLI components.
- **`tester`** (Verification Specialist): Develops comprehensive pytest test suites in `tests/test_commercial_policy.py`, `tests/test_compliance_report.py`, and `tests/test_ddcolor_reference.py`.
- **`code-reviewer`** (Quality Gate): Reviews all changes for correctness, security, typing, and zero TODOs.

## Risk Assessment

| Risk ID | Description | Severity | Likelihood | Mitigation Strategy |
|---|---|---|---|---|
| **R-1** | Existing projects break due to Commercial Guard default changing to `True` | Medium | Medium | `normalize_settings()` preserves saved project intent; GUI provides clear visual toggle that sets `non_commercial_benchmark_override: "true"` seamlessly. |
| **R-2** | Silent fallbacks to mock output or un-audited models | High | Low | Zero placeholders/silent fallbacks. Missing DDColor weights raise a fatal `FileNotFoundError` in production; mocks are gated behind `--dry-run` or pytest mocks. |
| **R-3** | Fabricated model fingerprints in policy config | High | Low | Zero placeholder SHA-256 hashes. Only verified hashes from the audit are populated; un-audited files compute real SHA-256 at evaluation time and reject cleanly with that hash. |

## Success Criteria

1. All existing and new tests pass under `python -m pytest tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_ddcolor_reference.py tests/test_model_license_inventory.py`.
2. Hard blocks enforced for `CMNET2`, `ColorMNet`, `Qwen-Image-2.1`, `ProPainter`, `MiniMax H3`, and `SAM 3.1`.
3. Fail-closed rejection verified on unknown LoRAs, printing filename and computed SHA-256.
4. Conditional policy gating verified for LTX revenue flag (`company_revenue_under_10m_usd`) and Deep Exemplar.
5. Cumulative `output/compliance_report.json` records multi-stage execution, RNG seeds, and EU AI Act Art. 50 disclosure metadata.
6. DDColor reference generator works deterministically in CLI, wrapper, and GUI.
