"""GUI and service layer integration for video segmentation tracking.

Re-exports core tracker components from scripts.sam2_video_tracker and delegates
run_shot_tracking_pipeline to ai_remaster_gui.video_segmenters while keeping 100%
backward compatibility with SAM 2.1 callers.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from scripts.rle_masks import MaskObjectMetadata, ShotMaskArchive
from scripts.sam2_video_tracker import (
    ByteTrackAdapterHook,
    NullTrackerAdapter,
    PreTrackerAdapter,
    ShotVideoTracker,
    verify_sam2_installation,
)
from .video_segmenters import DEFAULT_SEGMENTER_REGISTRY, run_shot_segmentation

logger = logging.getLogger(__name__)

__all__ = [
    "ByteTrackAdapterHook",
    "DEFAULT_SEGMENTER_REGISTRY",
    "NullTrackerAdapter",
    "PreTrackerAdapter",
    "ShotVideoTracker",
    "verify_sam2_installation",
    "run_shot_tracking_pipeline",
    "run_shot_segmentation",
]


def run_shot_tracking_pipeline(
    shot_id: str,
    frames: list[np.ndarray] | Path | str,
    prompts: dict[int, dict[str, Any]] | None,
    output_npz_path: Path | str,
    checkpoint_path: Path | str | None = None,
    device: str = "auto",
    predictor: Any = None,
    object_metadatas: list[MaskObjectMetadata] | None = None,
    min_mask_area: int = 16,
    iou_threshold: float = 0.5,
    *,
    segmenter: str = "sam21",
    text_prompts: list[str] | str | None = None,
    alpha_threshold: float = 0.5,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    allow_non_commercial_benchmark: bool = False,
    compliance_output_dir: Path | str | None = None,
    backend_kwargs: Mapping[str, Any] | None = None,
) -> Path:
    """Execute end-to-end shot tracking with the chosen segmenter and write compressed .masks.npz."""
    extra_kwargs = dict(backend_kwargs or {})
    if predictor is not None:
        extra_kwargs.setdefault("predictor", predictor)

    return run_shot_segmentation(
        shot_id=shot_id,
        frames=frames,
        prompts=prompts,
        output_npz_path=output_npz_path,
        segmenter=segmenter,
        text_prompts=text_prompts,
        checkpoint_path=checkpoint_path,
        device=device,
        object_metadatas=object_metadatas,
        min_mask_area=min_mask_area,
        iou_threshold=iou_threshold,
        alpha_threshold=alpha_threshold,
        settings=settings,
        allow_non_commercial_benchmark=allow_non_commercial_benchmark,
        compliance_output_dir=compliance_output_dir,
        **extra_kwargs,
    )
