# Design Document: Commercial-Safe Dust & Scratch Detector Integration in ARP GUI & Presets

**Session ID:** `2026-11-11-integrate-commercial-dust-detectors`  
**Date:** 2026-11-11  
**Status:** Approved  
**Author:** TechLead Orchestrator  

---

## 1. Overview & Architecture Scope

### 1.1 Objective
Integrate commercial-safe film defect detectors (`legacy`, `temporal`/`farneback_3f`, `dis_flow`, `ridge`, `consensus_5f`) into the existing AI Remaster Pipeline Dust & Scratches stage (`ai_remaster_gui/dust.py`, `ai_remaster_gui/static/js/render-dust.js`, `scripts/cleanup_video.py`, and `scripts/artifact_ids.py`).

### 1.2 Core Architectural Principles
1. **Strict Preset & Project Backward Compatibility:**
   - Every existing saved user preset, project configuration, and explicitly selected detector setting is preserved exactly without mutation or overwriting.
   - `temporal` / `farneback_3f` remains the default **only** when a project or preset has no explicit detector specified.
2. **No Silent Detector Substitutions:**
   - If a selected detector cannot process a short clip ($\text{frames} < \text{min\_frames}$) or lacks a required output channel (e.g., `ridge` lacking dust), the limitation is explicitly reported in diagnostic metadata and clearly surfaced in the GUI.
   - Under no circumstances will the system silently substitute another algorithm. Any operational boundary returns an explicit, diagnostic unsupported-context state.
3. **Parameter Schema Fidelity:**
   - Each detector declares and binds only its supported parameters (`sensitivity`, `threshold`, `dilate`, `radius`). Non-interchangeable controls (such as neural soft-mask thresholds) are strictly quarantined from classical detectors.
4. **Commercial Safety Firewall:**
   - DART remains strictly research-only and firewalled from commercial runtime, production presets, and commercial GUI workflows via `ai_remaster_gui/commercial_policy.py`.
5. **Working-Tree Safety & Single Integration Owner:**
   - A single integration coder owns all edits to shared GUI, preset, and pipeline files (`ai_remaster_gui/dust.py`, `ai_remaster_gui/models.py`, `scripts/cleanup_video.py`, `scripts/artifact_ids.py`). All edits are strictly serialized, and uncommitted Issue #37 work is preserved untouched.

---

## 2. Detailed Backend, Registry, Preset & Pipeline Contracts

1. **GUI Models & Registry Contract (`ai_remaster_gui/models.py`):**
   - Update `Stage("dust", ...)` field definition from:
     ```python
     ("detector", "Defect detector", "select:legacy|temporal|ridge|dart", "temporal")
     ```
     to dynamically reflect registered commercial detectors:
     ```python
     ("detector", "Defect detector", "select:temporal|farneback_3f|dis_flow|ridge|consensus_5f|legacy", "temporal")
     ```
   - In `options()` (`ai_remaster_gui/dust.py`):
     - Normalize `temporal` $\leftrightarrow$ `farneback_3f` transparently while validating against `DETECTOR_REGISTRY.list_available(commercial_only=True)`.
     - Reject DART when `commercial_model_guard` is active.
2. **Artifact ID & Cache Key Integrity (`scripts/artifact_ids.py`):**
   - Incorporate `defect_detector` into cleanup identity:
     ```python
     normalized = str(defect_detector or "farneback_3f").strip().lower()
     defect_algo = "temporal_motion_v1" if normalized in ("", "farneback_3f", "temporal") else f"{normalized}_v1"
     ```
   - Switching detectors cleanly produces distinct cache keys and directories without invalidating or overwriting previously accepted repair artifacts.
3. **Pipeline Runner Dispatch (`scripts/cleanup_video.py`):**
   - CLI flag `--defect-detector` dispatches directly through `DETECTOR_REGISTRY.get(name, commercial_only=True)`.
   - Propagate detector metadata (`channel_support`, `min_frames`, `provenance`) into the cleanup stage stamp JSON sidecar.
4. **Preset System Preservation:**
   - Existing saved preset dictionaries (`ai_remaster_gui/server.py` and project files) remain byte-compatible; omission of `detector` key evaluates via non-destructive `.get('detector', 'temporal')`.

---

## 3. Frontend UI, Interactive Mask Comparison & Channels

1. **Detector Selection Dropdown & Labels (`ai_remaster_gui/static/js/render-dust.js`):**
   - Populate friendly, descriptive labels with honest limitation badges:
     - `temporal` / `farneback_3f`: Temporal Optical Flow (Farneback)
     - `dis_flow`: Dense Inverse Search (DIS Flow)
     - `ridge`: Multi-Scale Ridge (Scratches only · experimental)
     - `consensus_5f`: Temporal Consensus (5-frame)
     - `legacy`: ARP Legacy Scratch Mask
   - Detector limitations (e.g. Ridge lacking dust, or methods requiring $\ge 3$ frames) are dynamically indicated below the dropdown when selected.
2. **Interactive Mask Comparison & Multi-Channel Viewport:**
   - Add channel selector buttons inside the mask viewport header:
     - `[ Combined ]` (Default full proposal)
     - `[ Scratch ]` (Elongated scratch channel, where supported)
     - `[ Dust ]` (Transient cluster channel, where supported; disabled / marked N/A for `ridge`)
     - `[ Overlay ]` (Mask overlaid in translucent red on source video frame)
   - Channels are retrieved from the detector's multi-channel result artifact (`scratch_mask.png`, `dust_mask.png`) without running a full video restoration.
3. **Non-Destructive Variant A/B Comparison:**
   - The existing `Variants` matrix card displays each calculated mask/repair as a discrete tile labeled with its detector and parameters.
   - Users can switch between Variant A and Variant B in the player to compare predictions on the shared frame clock without modifying the currently adopted pipeline output.

---

## 4. Testing & Verification Strategy

1. **Targeted Integration Test Suite (`tests/test_dust_detector_gui_integration.py`):**
   - **Test 1:** Selection and registration of all 5 commercial detectors (`temporal`/`farneback_3f`, `dis_flow`, `ridge`, `consensus_5f`, `legacy`).
   - **Test 2:** GUI-to-backend parameter propagation via `ai_remaster_gui/dust.py:options()`.
   - **Test 3:** Backward compatibility: existing saved presets and projects without a detector setting resolve cleanly to `temporal`.
   - **Test 4:** Cache key and artifact identity divergence between detectors.
   - **Test 5:** Preview isolation: generating a preview mask for detector B does not mutate or overwrite accepted masks for detector A.
   - **Test 6:** Commercial safety firewall: DART is blocked when `commercial_model_guard=True`.
   - **Test 7:** Operational limitations: single-frame clips on temporal detectors and Ridge dust channel absence return clear diagnostic indicators without silent substitution.
2. **Full Regression Execution:**
   - Run existing Dust stage and Issue #37 tests (`tests/test_dust_stage.py`, `tests/test_dust_workspace_browser.test.cjs`).
   - Run complete ARP test suite (`pytest tests/`) ensuring zero regressions across all existing tests.
3. **GUI Smoke Verification:**
   - Verify that switching detector selection in the GUI genuinely changes the calculated mask pixels on test frames rather than merely changing a label.
