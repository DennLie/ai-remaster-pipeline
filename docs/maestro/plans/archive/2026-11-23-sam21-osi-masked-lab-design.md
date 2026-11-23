# Phase 4: SAM 2.1 Video Tracking & 100%-OSI Deterministic Video Propagation Backend Design

- **Date**: 2026-11-23
- **Session ID**: 2026-11-23-sam21-osi-masked-lab
- **Status**: Approved

## 1. Overview & Architecture

Phase 4 of the Commercial Remaster Audit implements a **100%-OSI compliant, deterministic, reference-guided video colorization pipeline** for long-form films (60–120 minutes, 480–1,600 shots) that eliminates dependency on non-commercial models (`cmnet2`, `colormnet`) while preserving the historical black-and-white luminance ($L^*$) verbatim (*"Original pixels win"*).

### Core Components
1. **RLE Mask Codec & Multi-Object Shot Storage** (`ai_remaster_gui/rle_masks.py` and `scripts/rle_masks.py`):
   - Fast, bit-exact Run-Length-Encoded multi-object mask storage inside compressed `.masks.npz` archives.
   - Per-shot storage at `intermediate/shot_masks/{shot_id}.masks.npz`.
   - Explicit metadata schema mapping `obj_id` to Color Bible entity identifiers (`entity_type: character | object | location`, `entity_id`, `variant_id`), `width`, `height`, `frame_count`.
   - Achieves $>50\times$ compression vs raw uint8 arrays (<2 GB for 90-min 1080p film).
2. **Shot-Local SAM 2.1 Video Tracker Engine** (`scripts/sam2_video_tracker.py` and `ai_remaster_gui/sam2_tracker.py`):
   - Official `facebookresearch/sam2` wrapper with fail-closed installation/checkpoint sanity check (rejects unofficial PyPI `sam2` fork and non-Apache `SAM 3.1` checkpoint; accepts `sam2.1_hiera_base_plus.pt` and `sam2.1_hiera_large.pt`).
   - Strict per-shot lifecycle: `init_state` -> prompt/track -> `reset_state` -> explicit cleanup (`del inference_state`, GPU cache release).
   - Float32 `(3, H, W)` prediction handling with minimum foreground area and prompt-presence heuristics to reject empty-background false-positive masks.
   - Pluggable `PreTrackerAdapter` protocol hook for optional `ByteTrack` bounding-box stabilization during occlusions, with graceful degradation when absent.
3. **100%-OSI Deterministic Video Propagation Backend (`osi_masked_lab`)** (`scripts/osi_masked_lab.py`):
   - Dual-path propagation:
     - Tracked object masks: transfer approved CIE-LAB $a^*, b^*$ chroma from Color Bible / shot reference keyframe masks, spatially warped by optical flow to preserve shading and textile texture.
     - Unmasked background: propagate chroma from reference keyframes using OpenCV DIS Optical Flow (`cv2.DISOpticalFlow_create`) computed step-by-step between consecutive frames, with bidirectional temporal distance blending when multiple reference keyframes exist.
     - Luminance preservation: $L^*$ strictly locked to original B&W frame verbatim.
     - Luminance-guided edge-aware chroma filtering: He et al. Guided Filter implemented natively via standard `cv2.boxFilter` and NumPy (zero `cv2.ximgproc` dependency) to eliminate boundary color bleeding and temporal chatter.
   - Integrated as `--method osi_masked_lab` across `scripts/colorize_video.py`, `ai_remaster_gui/models.py`, `ai_remaster_gui/server.py`, and `wrappers/colorize_video.bat`.
   - Provenance and audit logging wired into `ai_remaster_gui/compliance.py` (`append_compliance_record`).

## 2. Data Flow & Interface Contracts

### 2.1 RLE `.masks.npz` Storage Format
Per-shot archive: `intermediate/shot_masks/{shot_id}.masks.npz`
- `metadata`: JSON string encoded as utf-8 bytes:
  ```json
  {
    "version": 1,
    "shot_id": "shot_001",
    "frame_count": 120,
    "height": 1080,
    "width": 1920,
    "objects": [
      {
        "obj_id": 1,
        "name": "protagonist_coat",
        "entity_type": "character",
        "entity_id": "char_01",
        "variant_id": "costume_02"
      }
    ],
    "compression": "rle_npz"
  }
  ```
- `rle_lengths`: 1D `int32` array containing run lengths of alternating 0s and 1s.
- `rle_index`: 2D `int64` array of shape `(num_objects, num_frames, 2)` storing `[offset, count]` slice pointers into `rle_lengths`.
- Roundtrip property: `decode_mask(encode_mask(m)) == m` is strictly bit-exact.

### 2.2 SAM 2.1 Video Tracker Interfaces
- `verify_sam2_installation()`: validates package and checkpoint, failing closed if unofficial PyPI fork or SAM 3.1 is detected.
- `ShotVideoTracker`: wraps `SAM2VideoPredictor` with per-shot state initialization and teardown.
- `PreTrackerAdapter`: protocol for optional bounding box tracking (e.g. ByteTrack).

### 2.3 `osi_masked_lab` Interfaces
- `guided_filter(guide_gray, src_channel, radius=5, eps=1e-3)`: native OpenCV box-filter implementation.
- `propagate_shot_chroma(...)`: executes bidirectional optical flow warping, texture-preserving object chroma transfer, background blending, guided filtering, and recombination with original $L^*$.

## 3. Commercial Compliance & Audit
- Registered under `ALLOWED_OSI` in `configs/commercial_policy_registry.json`.
- When Commercial Model Guard is enabled, `cmnet2` and `colormnet` are strictly blocked.
- Each `osi_masked_lab` run appends a record in `output/compliance_report.json` with status `allowed_osi` and license `Apache-2.0`.

## 4. Verification Plan
- Unit tests for RLE mask compression and bit-exact roundtrip (`tests/test_rle_masks.py`).
- Unit tests for SAM 2.1 tracker lifecycle, checkpoint sanity, and ByteTrack adapter (`tests/test_sam2_video_tracker.py`).
- Unit tests for `osi_masked_lab` optical flow, guided filtering, and luminance lock (`tests/test_osi_masked_lab.py`).
- Full regression test run across ARP suites.
