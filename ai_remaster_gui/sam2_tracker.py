"""GUI and service layer integration for SAM 2.1 video segmentation tracking.

Re-exports core tracker components from scripts.sam2_video_tracker and provides
the high-level run_shot_tracking_pipeline execution wrapper.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

__all__ = [
    "ByteTrackAdapterHook",
    "NullTrackerAdapter",
    "PreTrackerAdapter",
    "ShotVideoTracker",
    "verify_sam2_installation",
    "run_shot_tracking_pipeline",
]


def run_shot_tracking_pipeline(
    shot_id: str,
    frames: list[np.ndarray] | Path | str,
    prompts: dict[int, dict[str, Any]],
    output_npz_path: Path | str,
    checkpoint_path: Path | str | None = None,
    device: str = "auto",
    predictor: Any = None,
    object_metadatas: list[MaskObjectMetadata] | None = None,
    min_mask_area: int = 16,
    iou_threshold: float = 0.5,
) -> Path:
    """Execute end-to-end SAM 2 shot tracking and write compressed .masks.npz archive.

    Parameters
    ----------
    shot_id : str
        Shot identifier.
    frames : list[np.ndarray] | Path | str
        List of video frame arrays, or path to image directory or video file.
    prompts : dict[int, dict[str, Any]]
        Prompt specifications keyed by object ID (or frame index).
    output_npz_path : Path | str
        Destination path for .masks.npz file.
    checkpoint_path : Path | str | None
        Optional model checkpoint path or HuggingFace ID.
    device : str
        Execution device ('auto', 'cuda', 'cpu').
    predictor : Any | None
        Optional injected predictor instance (for CPU tests).
    object_metadatas : list[MaskObjectMetadata] | None
        Optional explicit object metadata.
    min_mask_area : int
        Minimum foreground pixels threshold for empty-background heuristic.
    iou_threshold : float
        Threshold for binarizing candidate probability/logits.

    Returns
    -------
    Path
        Path to the saved .masks.npz archive file.
    """
    target_path = Path(output_npz_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with ShotVideoTracker(
        checkpoint_path=checkpoint_path,
        device=device,
        predictor=predictor,
    ) as tracker:
        tracker.init_shot(frames)

        metas: list[MaskObjectMetadata] = list(object_metadatas) if object_metadatas else []
        known_meta_ids = {m.obj_id for m in metas}

        for key, pdict in prompts.items():
            if "obj_id" in pdict:
                obj_id = int(pdict["obj_id"])
                frame_idx = int(pdict.get("frame_idx", key))
            else:
                obj_id = int(key)
                frame_idx = int(pdict.get("frame_idx", 0))

            points = pdict.get("points")
            labels = pdict.get("labels")
            mask = pdict.get("mask")
            box = pdict.get("box")

            tracker.add_object_points_or_mask(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
                mask=mask,
                box=box,
            )

            if obj_id not in known_meta_ids:
                metas.append(
                    MaskObjectMetadata(
                        obj_id=obj_id,
                        name=str(pdict.get("name", f"obj_{obj_id}")),
                        entity_type=str(pdict.get("entity_type", "object")),
                        entity_id=str(pdict.get("entity_id", "")),
                        variant_id=str(pdict.get("variant_id", "")),
                    )
                )
                known_meta_ids.add(obj_id)

        tracker.track_and_save_archive(
            shot_id=shot_id,
            output_npz_path=target_path,
            object_metadatas=metas,
            min_mask_area=min_mask_area,
            iou_threshold=iou_threshold,
        )

    return target_path
