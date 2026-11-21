#!/usr/bin/env python3
"""Temporal Scratch Detector Adapter for the ARP defect detector registry.

Integrates the motion-compensated `TemporalScratchAnalyzer` into the ARP
pipeline and registry as a commercial-cleared, non-destructive detector
strategy conforming to BaseDefectDetector, DefectMaskResult, and
MultiChannelDefectMask contracts.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

logger = logging.getLogger("temporal_defect_adapter")

# Dual import compatibility
try:
    from scripts.film_defect_detectors import (
        BaseDefectDetector,
        ChannelOrigin,
        ChannelProvenance,
        DefectMaskResult,
        MultiChannelDefectMask,
        ReadinessState,
        normalize_frames,
    )
except ImportError:
    from film_defect_detectors import (
        BaseDefectDetector,
        ChannelOrigin,
        ChannelProvenance,
        DefectMaskResult,
        MultiChannelDefectMask,
        ReadinessState,
        normalize_frames,
    )

try:
    from scripts.temporal_scratch_analyzer import TemporalConfig, TemporalScratchAnalyzer
except ImportError:
    from temporal_scratch_analyzer import TemporalConfig, TemporalScratchAnalyzer


class TemporalScratchDetectorAdapter(BaseDefectDetector):
    """Adapter bridging TemporalScratchAnalyzer into ARP's BaseDefectDetector contract."""

    name = "temporal_scratch"
    aliases = ("temporal_fissure", "temporal_flow")
    description = "Motion-compensated optical flow temporal scratch detector with telecine de-duplication."
    license_status = "Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)"
    commercial_cleared = True
    readiness_state = ReadinessState.PRETRAINED_OPERATIONAL
    supported_channels = ("scratch",)
    unsupported_channels = ("dust", "uncertain")
    temporal_radius = 3

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "sensitivity": {
                "type": "float",
                "default": 0.65,
                "min": 0.0,
                "max": 1.0,
                "description": "Detection sensitivity from 0.0 (conservative) to 1.0 (aggressive).",
            },
            "window_radius": {
                "type": "int",
                "default": 3,
                "min": 1,
                "max": 5,
                "description": "Temporal neighbor window radius (R preceding and R succeeding frames).",
            },
            "flow_backend": {
                "type": "str",
                "default": "dis",
                "choices": ("dis", "farneback"),
                "description": "Dense optical flow backend ('dis' or 'farneback').",
            },
            "telecine_dedup_threshold": {
                "type": "float",
                "default": 0.8,
                "min": 0.0,
                "max": 50.0,
                "description": "Inter-frame MAD difference threshold below which frames are treated as telecine repeats.",
            },
            "scratch_min_height": {
                "type": "int",
                "default": 15,
                "min": 1,
                "max": 1000,
                "description": "Minimum vertical component height in pixels.",
            },
            "scratch_max_width": {
                "type": "int",
                "default": 12,
                "min": 1,
                "max": 200,
                "description": "Maximum horizontal component width in pixels.",
            },
            "motion_coherence_threshold": {
                "type": "float",
                "default": 1.2,
                "min": 0.0,
                "max": 100.0,
                "description": "Scene motion threshold in pixels to reject coherent scene textures.",
            },
            "scene_cut_threshold": {
                "type": "float",
                "default": 25.0,
                "min": 0.0,
                "max": 255.0,
                "description": "Inter-frame MAD difference threshold above which neighbor selection stops at a scene cut.",
            },
            "scratch_tall_override_height": {
                "type": "int",
                "default": 35,
                "min": 1,
                "max": 10000,
                "description": "Component height threshold to bypass aspect ratio filtering.",
            },
        }

    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        seq = normalize_frames(frames)
        if self.should_use_tiling(seq.height, seq.width, config):
            return self.detect_tiled(frames, config=config)

        params = self.validate_config(config)
        n_frames = len(seq.gray_frames)
        h, w = seq.height, seq.width

        # Micro-dimension guard: reject tiny frames/tiles that crash dense optical flow
        if seq.height < 16 or seq.width < 16:
            zeros = [np.zeros((seq.height, seq.width), dtype=np.uint8) for _ in range(n_frames)]
            zero_prob = np.zeros((seq.height, seq.width), dtype=np.float32)
            mcdm_list = [
                MultiChannelDefectMask(
                    combined_probability=zero_prob.copy(),
                    combined_mask=zeros[i].copy(),
                    scratch_mask=zeros[i].copy(),
                    dust_mask=None,
                    uncertain_mask=None,
                    provenance={
                        "scratch": ChannelProvenance(
                            origin=ChannelOrigin.NATIVE,
                            is_available=True,
                            source_algorithm="temporal_scratch_analyzer",
                            derivation_parameters=params,
                            confidence_threshold=0.5,
                        ),
                        "dust": ChannelProvenance(
                            origin=ChannelOrigin.UNSUPPORTED,
                            is_available=False,
                            source_algorithm="none",
                        ),
                        "uncertain": ChannelProvenance(
                            origin=ChannelOrigin.UNSUPPORTED,
                            is_available=False,
                            source_algorithm="none",
                        ),
                    },
                    metadata={"frame_index": i, "detector": self.name, "status": "abstain_small_dimension"},
                )
                for i in range(n_frames)
            ]
            return DefectMaskResult(
                mask=[m.copy() for m in zeros],
                scratch_mask=[m.copy() for m in zeros],
                dust_mask=[m.copy() for m in zeros],
                uncertain_mask=[m.copy() for m in zeros],
                multi_channel_masks=mcdm_list,
                metadata={"detector": self.name, "config": params, "frames": n_frames, "status": "abstain_small_dimension"},
            )

        sensitivity = float(params["sensitivity"])
        window_radius = int(params["window_radius"])
        flow_backend = str(params["flow_backend"])
        telecine_dedup_threshold = float(params["telecine_dedup_threshold"])
        scene_cut_threshold = float(params["scene_cut_threshold"])
        scratch_min_height = int(params["scratch_min_height"])
        scratch_max_width = int(params["scratch_max_width"])
        scratch_tall_override_height = int(params["scratch_tall_override_height"])
        motion_coherence_threshold = float(params["motion_coherence_threshold"])

        temporal_cfg = TemporalConfig(
            window_radius=window_radius,
            flow_backend=flow_backend,
            telecine_dedup_threshold=telecine_dedup_threshold,
            scene_cut_threshold=scene_cut_threshold,
            sensitivity=sensitivity,
            scratch_min_height=scratch_min_height,
            scratch_max_width=scratch_max_width,
            scratch_tall_override_height=scratch_tall_override_height,
            motion_coherence_threshold=motion_coherence_threshold,
        )
        analyzer = TemporalScratchAnalyzer(config=temporal_cfg)

        n_frames = len(seq.gray_frames)
        h, w = seq.height, seq.width

        scratch_masks: list[np.ndarray] = []
        dust_masks: list[np.ndarray] = []
        uncertain_masks: list[np.ndarray] = []
        multi_channel_masks: list[MultiChannelDefectMask] = []
        safety_guard_violations = 0

        for t in range(n_frames):
            defect_res = analyzer.analyze_frame(seq.gray_frames, t)
            raw_s_mask = defect_res.scratch_mask
            prob_map = defect_res.probability_map

            # Safety guard: maximum 30% mask coverage
            coverage = float(np.count_nonzero(raw_s_mask)) / float(raw_s_mask.size) if raw_s_mask.size > 0 else 0.0
            safety_guard_triggered = bool(coverage > 0.30)

            if safety_guard_triggered:
                logger.warning(
                    "Temporal scratch safety guard triggered on frame %d: coverage %.2f%% exceeds 30.0%% limit. "
                    "Refusing automatic repair; zeroing defect mask.",
                    t,
                    coverage * 100.0,
                )
                safety_guard_violations += 1
                effective_s_mask = np.zeros((h, w), dtype=np.uint8)
                effective_prob = np.zeros((h, w), dtype=np.float32)
            else:
                effective_s_mask = raw_s_mask
                effective_prob = prob_map

            # Ensure contiguous uint8 array
            if not effective_s_mask.flags["C_CONTIGUOUS"]:
                effective_s_mask = np.ascontiguousarray(effective_s_mask)
            if not effective_prob.flags["C_CONTIGUOUS"]:
                effective_prob = np.ascontiguousarray(effective_prob)

            scratch_masks.append(effective_s_mask)
            zero_channel = np.zeros((h, w), dtype=np.uint8)
            dust_masks.append(zero_channel)
            uncertain_masks.append(zero_channel.copy())

            prov = {
                "scratch": ChannelProvenance(
                    origin=ChannelOrigin.NATIVE,
                    is_available=True,
                    source_algorithm="temporal_scratch_analyzer",
                    derivation_parameters={
                        "sensitivity": sensitivity,
                        "window_radius": window_radius,
                        "flow_backend": flow_backend,
                    },
                    confidence_threshold=0.5,
                ),
                "dust": ChannelProvenance(
                    origin=ChannelOrigin.UNSUPPORTED,
                    is_available=False,
                    source_algorithm="none",
                ),
                "uncertain": ChannelProvenance(
                    origin=ChannelOrigin.UNSUPPORTED,
                    is_available=False,
                    source_algorithm="none",
                ),
            }

            mcdm_meta: dict[str, Any] = {
                "frame_index": t,
                "detector": self.name,
                "safety_guard_triggered": safety_guard_triggered,
                "safety_guard_coverage": round(coverage, 5),
                "safety_guard_limit": 0.30,
            }
            if hasattr(defect_res, "metadata") and isinstance(defect_res.metadata, dict):
                mcdm_meta.update(defect_res.metadata)

            mcdm = MultiChannelDefectMask(
                combined_probability=effective_prob,
                combined_mask=effective_s_mask,
                scratch_mask=effective_s_mask,
                dust_mask=None,
                uncertain_mask=None,
                provenance=prov,
                metadata=mcdm_meta,
            )
            multi_channel_masks.append(mcdm)

        return DefectMaskResult(
            mask=scratch_masks,
            scratch_mask=scratch_masks,
            dust_mask=dust_masks,
            uncertain_mask=uncertain_masks,
            multi_channel_masks=multi_channel_masks,
            metadata={
                "detector": self.name,
                "config": params,
                "frames": n_frames,
                "height": h,
                "width": w,
                "commercial_cleared": self.commercial_cleared,
                "safety_guard_triggered": bool(safety_guard_violations > 0),
                "safety_guard_violations": safety_guard_violations,
                "safety_guard_compliant": bool(safety_guard_violations == 0),
            },
        )


__all__ = [
    "TemporalScratchDetectorAdapter",
]
