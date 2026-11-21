"""Isolated opt-in adapter for Experimental E2 (`ExperimentalSceneFilmDetector`).

Exposes the 3-state experimental prototype cleanly to the GUI/pipeline layer:
  - `high_confidence_damage` -> `scratch_mask` (strictly `{0, 255}`)
  - `uncertain_damage_candidate` -> `uncertain_mask` (strictly `{0, 255}`)
  - `likely_intact_scene_structure` -> `vetoed_masks` (separate veto layer)
  - `dust_mask` -> zeros with `ChannelOrigin.UNSUPPORTED`

Fails closed unless `research_confirmed` is explicitly true (`True` or `'true'`).
Never promotes `uncertain_damage_candidate` or `likely_intact_scene_structure`
into confirmed damage or independent ground truth.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import sys
from typing import Any

import numpy as np

from .config import SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    from experimental_scene_film_scratch_prototype import (
        ExperimentalConfig,
        ExperimentalSceneFilmDetector,
    )
except ImportError:
    from scripts.experimental_scene_film_scratch_prototype import (
        ExperimentalConfig,
        ExperimentalSceneFilmDetector,
    )

try:
    from film_defect_detectors import (
        ChannelOrigin,
        ChannelProvenance,
        DefectMaskResult,
        MultiChannelDefectMask,
        normalize_frames,
    )
except ImportError:
    from scripts.film_defect_detectors import (
        ChannelOrigin,
        ChannelProvenance,
        DefectMaskResult,
        MultiChannelDefectMask,
        normalize_frames,
    )


DETECTOR_NAME = "experimental_e2"
DETECTOR_LABEL = "Experimental E2 · Scene-vs-Film (Experimental)"
DETECTOR_VERSION = "0.1.0-experimental"
DETECTOR_ARCHITECTURE = "ExperimentalSceneFilmDetector (3-State Scene-vs-Film)"
E2_TEMPORAL_RADIUS = 3

E2_DETECTOR_INFO: dict[str, Any] = {
    "name": DETECTOR_NAME,
    "aliases": [],
    "label": DETECTOR_LABEL,
    "description": (
        "Opt-in experimental 3-state motion-aware scene-vs-film scratch detector "
        "(high-confidence damage, uncertain candidates, and vetoed intact scene structure). "
        "Requires explicit research confirmation."
    ),
    "license_status": "Apache-2.0 / BSD-3-Clause (OpenCV + NumPy; Experimental Unvalidated)",
    "commercial_cleared": True,
    "temporal_radius": E2_TEMPORAL_RADIUS,
    "readiness_state": "trained_experimental",
    "supported_channels": ["scratch", "uncertain"],
    "unsupported_channels": ["dust"],
    "experimental_opt_in_required": True,
    "research_confirmed_required": True,
    "three_state_outputs": [
        "high_confidence_damage",
        "uncertain_damage_candidate",
        "likely_intact_scene_structure",
    ],
    "schema": {
        "sensitivity": {
            "type": "float",
            "default": 0.65,
            "min": 0.0,
            "max": 1.0,
            "description": "Detection sensitivity from 0.0 (conservative) to 1.0 (aggressive).",
        },
    },
}

E2_PRESET_INFO: dict[str, Any] = {
    "name": "experimental_e2_research",
    "label": DETECTOR_LABEL,
    "detector": DETECTOR_NAME,
    "repair": "telea",
    "repair_policy": "conservative",
    "sensitivity": 0.65,
    "dilate": 0,
    "threshold": 160,
    "radius": 3,
    "commercial_cleared": True,
    "description": (
        "Opt-in experimental 3-state motion-aware scene-vs-film scratch detector "
        "(high confidence, uncertain, and vetoed intact structure)."
    ),
    "target_use_case": (
        "Experimental review of panning and static film scratches with explicit "
        "research opt-in and human mask editing."
    ),
    "evidence": "Prototype ablation only; unvalidated on independent historical ground truth.",
    "checkpoint_sha256": "",
    "license_status": E2_DETECTOR_INFO["license_status"],
    "params": {"sensitivity": 0.65, "window_radius": E2_TEMPORAL_RADIUS, "flow_backend": "dis"},
}


class ExperimentalOptInRequiredError(PermissionError, ValueError):
    """Raised when Experimental E2 is invoked without explicit research_confirmed opt-in."""


def is_research_confirmed(value: Any) -> bool:
    """Return True only when explicit experimental opt-in is confirmed."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return False


def require_experimental_opt_in(research_confirmed: Any) -> None:
    """Fail closed unless `research_confirmed` is explicitly true."""
    if not is_research_confirmed(research_confirmed):
        raise ExperimentalOptInRequiredError(
            "Experimental E2 detector is strictly opt-in; set research_confirmed='true' to execute."
        )


def _to_strict_binary_u8(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if not isinstance(arr, np.ndarray) or arr.shape != shape:
        raise ValueError(f"Expected 2D array with shape {shape}, got {getattr(arr, 'shape', None)}")
    return np.ascontiguousarray(np.where(arr > 0, np.uint8(255), np.uint8(0)), dtype=np.uint8)


def run_experimental_e2_sequence(
    frames_bgr: Sequence[np.ndarray] | np.ndarray,
    sensitivity: float = 0.65,
    research_confirmed: bool | str = False,
    *,
    window_radius: int = E2_TEMPORAL_RADIUS,
    flow_backend: str = "dis",
    target_indices: Sequence[int] | None = None,
    spatial_scratch_masks: Sequence[np.ndarray] | None = None,
    spatial_dust_masks: Sequence[np.ndarray] | None = None,
    config_overrides: Mapping[str, Any] | None = None,
) -> DefectMaskResult:
    """Execute Experimental E2 3-state detection across a frame sequence.

    Strictly preserves three non-flattened proposal outputs:
      - `scratch_mask`: `high_confidence_damage` (authorized for CONSERVATIVE repair)
      - `uncertain_mask`: `uncertain_damage_candidate` (review / HIGH_RECALL only)
      - `vetoed_masks` (on `metadata['vetoed_masks']` and `.vetoed_mask`):
        `likely_intact_scene_structure` (never promoted to damage)
      - `dust_mask`: zeros, marked `ChannelOrigin.UNSUPPORTED` in `ChannelProvenance`
    """
    require_experimental_opt_in(research_confirmed)

    if isinstance(sensitivity, bool):
        raise ValueError("sensitivity must be a finite float in [0.0, 1.0], not bool.")
    sens = float(sensitivity)
    if not math.isfinite(sens) or not 0.0 <= sens <= 1.0:
        raise ValueError(f"sensitivity must be finite in [0.0, 1.0], got {sensitivity!r}")

    seq = normalize_frames(frames_bgr)
    n_frames = len(seq.bgr_frames)
    h, w = seq.height, seq.width

    if target_indices is not None:
        eval_indices = [int(i) for i in target_indices]
        for idx in eval_indices:
            if idx < 0 or idx >= n_frames:
                raise ValueError(f"target_index {idx} out of range [0, {n_frames})")
    else:
        eval_indices = list(range(n_frames))

    cfg_kwargs: dict[str, Any] = {
        "sensitivity": sens,
        "window_radius": int(window_radius),
        "flow_backend": str(flow_backend),
    }
    if config_overrides:
        cfg_kwargs.update(dict(config_overrides))
    cfg = ExperimentalConfig(**cfg_kwargs)
    detector = ExperimentalSceneFilmDetector(cfg)

    raw_results = detector.analyze_sequence(
        seq.bgr_frames,
        spatial_scratch_masks=spatial_scratch_masks,
        spatial_dust_masks=spatial_dust_masks,
        target_indices=eval_indices,
    )

    combined_masks: list[np.ndarray] = []
    scratch_masks: list[np.ndarray] = []
    dust_masks: list[np.ndarray] = []
    uncertain_masks: list[np.ndarray] = []
    vetoed_masks: list[np.ndarray] = []
    multi_channel_masks: list[MultiChannelDefectMask] = []
    frame_metadata: list[dict[str, Any]] = []

    for idx in range(n_frames):
        res = raw_results.get(idx)
        zeros = np.zeros((h, w), dtype=np.uint8)
        if res is None:
            s_arr = zeros
            u_arr = zeros.copy()
            d_arr = zeros.copy()
            v_arr = zeros.copy()
            c_arr = zeros.copy()
            abstained = True
            abstention_reason = "skipped_context_frame"
            res_meta: dict[str, Any] = {}
        else:
            s_arr = _to_strict_binary_u8(res.high_confidence_damage, (h, w))
            u_arr = _to_strict_binary_u8(res.uncertain_damage_candidate, (h, w))
            u_arr[s_arr > 0] = 0
            v_arr = _to_strict_binary_u8(res.likely_intact_scene_structure, (h, w))
            v_arr[s_arr > 0] = 0
            v_arr[u_arr > 0] = 0
            d_arr = zeros
            c_arr = np.ascontiguousarray(np.bitwise_or(s_arr, u_arr), dtype=np.uint8)
            abstained = bool(res.abstained)
            abstention_reason = res.abstention_reason or ""
            res_meta = dict(res.metadata or {})

        prob = np.zeros((h, w), dtype=np.float32)
        if np.any(u_arr > 0):
            prob[u_arr > 0] = 0.5
        if np.any(s_arr > 0):
            prob[s_arr > 0] = 1.0

        prov = {
            "scratch": ChannelProvenance(
                origin=ChannelOrigin.NATIVE,
                is_available=True,
                source_algorithm="experimental_scene_film_prototype",
                derivation_parameters={
                    "sensitivity": sens,
                    "window_radius": cfg.window_radius,
                    "flow_backend": cfg.flow_backend,
                },
                confidence_threshold=0.5,
            ),
            "dust": ChannelProvenance(
                origin=ChannelOrigin.UNSUPPORTED,
                is_available=False,
                source_algorithm="none",
            ),
            "uncertain": ChannelProvenance(
                origin=ChannelOrigin.NATIVE,
                is_available=True,
                source_algorithm="experimental_scene_film_prototype",
                derivation_parameters={
                    "sensitivity": sens,
                    "window_radius": cfg.window_radius,
                    "flow_backend": cfg.flow_backend,
                },
                confidence_threshold=0.5,
            ),
        }

        mcdm_meta: dict[str, Any] = {
            "frame_index": idx,
            "detector": DETECTOR_NAME,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "high_confidence_pixels": int(np.count_nonzero(s_arr)),
            "uncertain_pixels": int(np.count_nonzero(u_arr)),
            "vetoed_intact_pixels": int(np.count_nonzero(v_arr)),
        }
        mcdm = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=c_arr,
            scratch_mask=s_arr,
            dust_mask=None,
            uncertain_mask=u_arr,
            provenance=prov,
            metadata=mcdm_meta,
        )

        combined_masks.append(c_arr)
        scratch_masks.append(s_arr)
        dust_masks.append(d_arr)
        uncertain_masks.append(u_arr)
        vetoed_masks.append(v_arr)
        multi_channel_masks.append(mcdm)
        frame_metadata.append(
            {
                **mcdm_meta,
                "cut_selector": res_meta.get("cut_selector"),
                "global_fb_med": res_meta.get("global_fb_med"),
                "global_fb_ok_frac": res_meta.get("global_fb_ok_frac"),
            }
        )

    result = DefectMaskResult(
        mask=combined_masks,
        scratch_mask=scratch_masks,
        dust_mask=dust_masks,
        uncertain_mask=uncertain_masks,
        multi_channel_masks=multi_channel_masks,
        metadata={
            "detector": DETECTOR_NAME,
            "version": DETECTOR_VERSION,
            "architecture": DETECTOR_ARCHITECTURE,
            "research_confirmed": True,
            "experimental_opt_in": True,
            "frames": n_frames,
            "height": h,
            "width": w,
            "config": {
                "sensitivity": sens,
                "window_radius": cfg.window_radius,
                "flow_backend": cfg.flow_backend,
            },
            "vetoed_masks": vetoed_masks,
            "frame_metadata": frame_metadata,
            "total_high_confidence_pixels": sum(int(np.count_nonzero(m)) for m in scratch_masks),
            "total_uncertain_pixels": sum(int(np.count_nonzero(m)) for m in uncertain_masks),
            "total_vetoed_pixels": sum(int(np.count_nonzero(m)) for m in vetoed_masks),
        },
    )
    # Auxiliary sequence attribute for callers inspecting the 3-state veto layer directly.
    setattr(result, "vetoed_mask", vetoed_masks)
    setattr(result, "vetoed_masks", vetoed_masks)
    return result


__all__ = [
    "DETECTOR_NAME",
    "DETECTOR_LABEL",
    "DETECTOR_VERSION",
    "DETECTOR_ARCHITECTURE",
    "E2_TEMPORAL_RADIUS",
    "E2_DETECTOR_INFO",
    "E2_PRESET_INFO",
    "ExperimentalOptInRequiredError",
    "is_research_confirmed",
    "require_experimental_opt_in",
    "run_experimental_e2_sequence",
]
