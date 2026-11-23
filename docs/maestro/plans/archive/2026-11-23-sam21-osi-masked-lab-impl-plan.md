---
title: "Phase 4 SAM 2.1 Video Tracking & 100%-OSI Video Propagation Implementation Plan"
design_ref: "docs/maestro/plans/2026-11-23-sam21-osi-masked-lab-design.md"
created: "2026-11-23T13:47:00Z"
status: "draft"
total_phases: 6
estimated_files: 14
task_complexity: "complex"
---

# Phase 4 SAM 2.1 Video Tracking & 100%-OSI Video Propagation Implementation Plan

## Plan Overview

- **Total phases**: 6
- **Agents involved**: `coder`, `tester`
- **Estimated effort**: Complex (14 files across RLE mask codec, SAM 2.1 lifecycle/guard, DIS optical flow + guided filter propagation engine, commercial model guard, and end-to-end regression testing)

## Dependency Graph

```
Phase 1: RLE Multi-Object Mask Codec & Storage
   │
   ▼
Phase 2: SAM 2.1 Video Tracker Lifecycle & Sanity Guard
   │
   ▼
Phase 3: 100%-OSI Video Propagation Core Engine (osi_masked_lab)
   │
   ▼
Phase 4: Commercial Model Guard & Compliance Integration
   │
   ▼
Phase 5: Pipeline Orchestration, GUI & CLI Integration
   │
   ▼
Phase 6: Comprehensive Test Verification & End-to-End Regression Validation
```

## Execution Strategy

| Stage | Phases | Execution | Agent Count | Notes |
|-------|--------|-----------|-------------|-------|
| 1     | Phase 1 | Sequential | 1 | Foundation RLE codec and `.masks.npz` format |
| 2     | Phase 2 | Sequential | 1 | SAM 2.1 lifecycle, fail-closed checkpoint guard, ByteTrack hook |
| 3     | Phase 3 | Sequential | 1 | Core `osi_masked_lab` engine, DIS flow, Guided Filter, $L^*$ lock |
| 4     | Phase 4 | Sequential | 1 | Commercial policy registry, model guard, compliance records |
| 5     | Phase 5 | Sequential | 1 | CLI, GUI, server, and wrapper integrations |
| 6     | Phase 6 | Sequential | 1 | Full 20-suite regression and CPU validation gate |

---

## Phase 1: RLE Multi-Object Mask Codec & Storage

### Objective
Implement a high-performance, bit-exact Run-Length-Encoded (RLE) multi-object mask codec and compressed `.masks.npz` archive storage for shots, achieving $>50\times$ compression over raw uint8 arrays and embedding structured Color Bible entity links.

### Agent: coder
### Parallel: No

### Files to Create
- `ai_remaster_gui/rle_masks.py` — Core RLE codec (`encode_mask_rle`, `decode_mask_rle`, `ShotMaskArchive`, `save_shot_masks_npz`, `load_shot_masks_npz`)
- `scripts/rle_masks.py` — Script-level re-export and CLI helper for inspecting/extracting `.masks.npz` archives
- `tests/test_rle_masks.py` — Unit tests verifying bit-exactness, $>50\times$ compression, metadata schema validation, and random-access slicing

### Implementation Details
- Alternating run-length encoding starting with count of 0s, stored as 1D `int32` array.
- Archive metadata JSON with `version`, `shot_id`, `frame_count`, `height`, `width`, `objects: [{ obj_id, name, entity_type, entity_id, variant_id }]`.
- `entity_type` restricted to Phase 2 enum: `"character" | "object" | "location"`.

### Validation
- `python -m pytest tests/test_rle_masks.py -q`

### Dependencies
- Blocked by: None
- Blocks: Phase 2

---

## Phase 2: SAM 2.1 Video Tracker Lifecycle & Sanity Guard

### Objective
Wrap the official `facebookresearch/sam2` `SAM2VideoPredictor` with fail-closed installation/checkpoint sanity checks, per-shot lifecycle memory cleanup, float32 `(3, H, W)` handling with empty-background heuristics, and a pluggable `PreTrackerAdapter` protocol hook for ByteTrack.

### Agent: coder
### Parallel: No

### Files to Create
- `scripts/sam2_video_tracker.py` — `ShotVideoTracker`, `verify_sam2_installation`, `PreTrackerAdapter`, `NullTrackerAdapter`, `ByteTrackAdapterHook`
- `ai_remaster_gui/sam2_tracker.py` — GUI integration and service bridge for shot-local video tracking
- `tests/test_sam2_video_tracker.py` — Unit tests for sanity check failures (PyPI fork, SAM 3.1), per-shot lifecycle cleanup, empty-background heuristics, and ByteTrack adapter degradation

### Implementation Details
- `verify_sam2_installation()` verifies package origin and checkpoint path (`sam2.1_hiera_base_plus.pt` / `sam2.1_hiera_large.pt`).
- Explicit per-shot lifecycle: `init_state` -> track -> `reset_state` -> `del inference_state` -> GPU cache clearing.
- Discard candidate masks with area $< 16$ px or point prompts on uniform background.

### Validation
- `python -m pytest tests/test_sam2_video_tracker.py -q`

### Dependencies
- Blocked by: Phase 1
- Blocks: Phase 3

---

## Phase 3: 100%-OSI Video Propagation Core Engine (`osi_masked_lab`)

### Objective
Implement the deterministic 100%-OSI video colorization backend with step-by-step bidirectional OpenCV DIS Optical Flow background propagation, texture-preserving object chroma transfer, native `cv2.boxFilter` Guided Filter edge-smoothing, and strict $L^*$ preservation.

### Agent: coder
### Parallel: No

### Files to Create
- `scripts/osi_masked_lab.py` — `guided_filter`, `calculate_step_flow`, `warp_chroma_backward`, `propagate_shot_chroma`, `osi_masked_lab_colorize_shot`
- `tests/test_osi_masked_lab.py` — Tests for guided filter smoothing, bidirectional flow warping, strict $L^*$ luminance invariance, and texture-preserving mask chroma transfer

### Implementation Details
- `guided_filter(guide_gray, src_channel, radius=5, eps=1e-3)` via `cv2.boxFilter` without `cv2.ximgproc`.
- Backward lookup flow: `dis.calc(curr_gray, prev_gray, None)` with `map = grid + flow`.
- Consecutive step-by-step warping along forward and backward keyframe chains, temporally blended by distance.
- $L^*_{\text{out}}(x, y) = L^*_{\text{orig}}(x, y)$ within $10^{-4}$ tolerance.

### Validation
- `python -m pytest tests/test_osi_masked_lab.py -q`

### Dependencies
- Blocked by: Phase 2
- Blocks: Phase 4

---

## Phase 4: Commercial Model Guard & Compliance Integration

### Objective
Register `osi_masked_lab` as `ALLOWED_OSI` in `configs/commercial_policy_registry.json`, ensure Commercial Model Guard strictly blocks non-commercial engines (`cmnet2`, `colormnet`) while clearing `osi_masked_lab`, and log audit records into `output/compliance_report.json`.

### Agent: coder
### Parallel: No

### Files to Modify
- `configs/commercial_policy_registry.json` — Add `osi_masked_lab` entry under `ALLOWED_OSI`
- `ai_remaster_gui/commercial_policy.py` — Update evaluation mappings if necessary
- `ai_remaster_gui/compliance.py` — Ensure `osi_masked_lab` run records include `color_bible_version` and component hashes
- `tests/test_commercial_policy.py` — Tests verifying `osi_masked_lab` is allowed and `cmnet2`/`colormnet` are hard-blocked in Commercial Mode
- `tests/test_compliance_report.py` — Tests verifying compliance records for `osi_masked_lab`

### Implementation Details
- Evaluate `osi_masked_lab` as `allowed_osi` (Apache-2.0).
- Reject `cmnet2` and `colormnet` when `commercial_model_guard` is active.
- Verify `output/compliance_report.json` accumulates compliant records.

### Validation
- `python -m pytest tests/test_commercial_policy.py tests/test_compliance_report.py -q`

### Dependencies
- Blocked by: Phase 3
- Blocks: Phase 5

---

## Phase 5: Pipeline Orchestration, GUI & CLI Integration

### Objective
Integrate `osi_masked_lab` into `scripts/colorize_video.py` (`--method osi_masked_lab`), add UI configuration options in `ai_remaster_gui/models.py`, update output resolution in `server.py` and `references.py`, and update CLI batch wrappers.

### Agent: coder
### Parallel: No

### Files to Modify
- `scripts/colorize_video.py` — Add `osi_masked_lab` option, CLI args, execution handler, and compliance logging call
- `ai_remaster_gui/models.py` — Add `osi_masked_lab` to method options in `colour` and `recomp` stages
- `ai_remaster_gui/server.py` — Support `osi_masked_lab` artifact tracking and preview generation
- `ai_remaster_gui/references.py` — Update colorized output helpers for `osi_masked_lab`
- `wrappers/colorize_video.bat` — Expose `--method osi_masked_lab` in batch script

### Implementation Details
- In `scripts/colorize_video.py`, wire `run_osi_masked_lab_colorization(...)` to invoke `scripts/osi_masked_lab.py`.
- Ensure `--method osi_masked_lab` passes through Commercial Model Guard and logs compliance report.

### Validation
- `python -m pytest tests/test_http_routes.py tests/test_gui_smoke.py tests/test_pipeline_order.py -q`

### Dependencies
- Blocked by: Phase 4
- Blocks: Phase 6

---

## Phase 6: Comprehensive Test Verification & End-to-End Regression Validation

### Objective
Execute the full 20-suite automated test command on CPU, verifying all tests pass 100% with zero regressions across the codebase.

### Agent: tester
### Parallel: No

### Files to Modify
- `tests/test_osi_masked_lab.py` — Final edge-case assertion polish if needed

### Validation
- Execute the full acceptance gate:
  `python -m pytest tests/test_rle_masks.py tests/test_sam2_video_tracker.py tests/test_osi_masked_lab.py tests/test_frame_masks.py tests/test_temporal_consistency.py tests/test_reference_luminance.py tests/test_ltx_colorize.py tests/test_video_order.py tests/test_cross_shot_matching.py tests/test_semantic_matching.py tests/test_color_bible_api.py tests/test_color_bible_cli.py tests/test_color_bible_db.py tests/test_color_bible_service.py tests/test_commercial_policy.py tests/test_compliance_report.py tests/test_http_routes.py tests/test_http_security.py tests/test_gui_smoke.py tests/test_pipeline_order.py -q`

### Dependencies
- Blocked by: Phase 5
- Blocks: None

---

## File Inventory

| # | File | Phase | Purpose |
|---|------|-------|---------|
| 1 | `ai_remaster_gui/rle_masks.py` | 1 | Bit-exact RLE codec and ShotMaskArchive format |
| 2 | `scripts/rle_masks.py` | 1 | Script CLI utilities and re-exports for RLE masks |
| 3 | `tests/test_rle_masks.py` | 1 | Unit tests for RLE mask compression and bit-exactness |
| 4 | `scripts/sam2_video_tracker.py` | 2 | SAM 2.1 video tracker runner, lifecycle & sanity guard |
| 5 | `ai_remaster_gui/sam2_tracker.py` | 2 | GUI bridge for SAM 2.1 video tracking |
| 6 | `tests/test_sam2_video_tracker.py` | 2 | Unit tests for SAM 2.1 tracker lifecycle and guards |
| 7 | `scripts/osi_masked_lab.py` | 3 | Deterministic 100%-OSI video propagation engine |
| 8 | `tests/test_osi_masked_lab.py` | 3 | Tests for DIS flow, guided filter, and $L^*$ lock |
| 9 | `configs/commercial_policy_registry.json` | 4 | Add `osi_masked_lab` under `ALLOWED_OSI` |
| 10 | `ai_remaster_gui/commercial_policy.py` | 4 | Policy evaluation for `osi_masked_lab` |
| 11 | `ai_remaster_gui/compliance.py` | 4 | Compliance record generation for `osi_masked_lab` |
| 12 | `tests/test_commercial_policy.py` | 4 | Policy tests for `osi_masked_lab` vs `cmnet2`/`colormnet` |
| 13 | `tests/test_compliance_report.py` | 4 | Tests for compliance reporting with `osi_masked_lab` |
| 14 | `scripts/colorize_video.py` | 5 | Support `--method osi_masked_lab` |
| 15 | `ai_remaster_gui/models.py` | 5 | Add `osi_masked_lab` to GUI stage configuration |
| 16 | `ai_remaster_gui/server.py` | 5 | Route `osi_masked_lab` outputs and previews |
| 17 | `ai_remaster_gui/references.py` | 5 | Manifest colorization output mapping for `osi_masked_lab` |
| 18 | `wrappers/colorize_video.bat` | 5 | Batch file CLI integration |

---

## Risk Classification

| Phase | Risk | Rationale |
|-------|------|-----------|
| 1 | LOW | Self-contained NumPy codec; standard mathematical algorithms. |
| 2 | MEDIUM | Requires careful mocking in unit tests so it runs 100% on CPU without GPU or pretrained weights. |
| 3 | MEDIUM | Optical flow boundary handling and guided filter precision require numerical stability. |
| 4 | LOW | Standard policy registry extension and compliance logging. |
| 5 | MEDIUM | Integration across GUI server, CLI flags, and stage configuration. |
| 6 | LOW | Test runner and validation gate. |

---

## Execution Profile

```
Execution Profile:
- Total phases: 6
- Parallelizable phases: 0
- Sequential-only phases: 6
- Estimated parallel wall time: N/A (Sequential mode requested by user)
- Estimated sequential wall time: ~15-20 minutes

Note: Native subagents run sequentially under the orchestrator's oversight.
```
