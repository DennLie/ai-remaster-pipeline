#!/usr/bin/env python3
"""SAM 2.1 Video Tracker Lifecycle & Sanity Guard.

Wraps official facebookresearch/sam2 SAM2VideoPredictor with:
- Fail-closed commercial policy & unofficial PyPI fork rejection.
- Strict per-shot inference lifecycle and VRAM/RAM cleanup.
- Float32 (3, H, W) candidate selection with empty-background heuristics.
- Pluggable PreTrackerAdapter protocol with ByteTrack hook fallback.
- Export to compressed .masks.npz archives via ShotMaskArchive.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Generator, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.rle_masks import MaskObjectMetadata, ShotMaskArchive

try:
    from ai_remaster_gui.commercial_policy import (
        CommercialPolicyError,
        evaluate_component,
    )
except ImportError:
    class CommercialPolicyError(RuntimeError):  # type: ignore[no-redef]
        """Raised when a model component fails commercial policy validation."""

logger = logging.getLogger(__name__)

VALID_SAM2_CHECKPOINTS: frozenset[str] = frozenset({
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large.pt",
    "sam2_hiera_tiny.pt",
    "sam2_hiera_small.pt",
    "sam2_hiera_base_plus.pt",
    "sam2_hiera_large.pt",
})

VALID_HF_MODELS: frozenset[str] = frozenset({
    "facebook/sam2.1-hiera-tiny",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2.1-hiera-large",
    "facebook/sam2-hiera-tiny",
    "facebook/sam2-hiera-small",
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2-hiera-large",
})

HARD_BLOCKED_PATTERNS: tuple[str, ...] = (
    "sam3",
    "sam-3",
    "sam 3",
)

# Optional ByteTrack imports
_BYTE_TRACK_AVAILABLE = False
try:
    from yolox.tracker.byte_tracker import BYTETracker  # type: ignore
    _BYTE_TRACK_AVAILABLE = True
except ImportError:
    try:
        from bytetrack.byte_tracker import BYTETracker  # type: ignore
        _BYTE_TRACK_AVAILABLE = True
    except ImportError:
        BYTETracker = None  # type: ignore[misc, assignment]


def verify_sam2_installation(checkpoint_path: Path | str | None = None) -> dict[str, str]:
    """Verify official facebookresearch/sam2 (Apache-2.0) installation and checkpoint compliance.

    Fails closed with CommercialPolicyError or RuntimeError if:
      - The checkpoint refers to SAM 3.1 or non-Apache checkpoint.
      - The installed package is the unofficial PyPI sam2 fork or lacks hiera/video components.

    Parameters
    ----------
    checkpoint_path : Path | str | None
        Optional model checkpoint path or HuggingFace ID.

    Returns
    -------
    dict[str, str]
        Sanity verification summary dictionary.
    """
    # 1. Checkpoint sanity checks
    if checkpoint_path is not None:
        ckpt_str = str(checkpoint_path).replace("\\", "/").strip()
        ckpt_lower = ckpt_str.lower()
        ckpt_name = Path(ckpt_str).name.lower()

        # Check for hard-blocked SAM 3 / 3.1 patterns
        for pattern in HARD_BLOCKED_PATTERNS:
            if pattern in ckpt_lower or pattern in ckpt_name:
                raise CommercialPolicyError(
                    f"Commercial policy blocked checkpoint '{checkpoint_path}': "
                    f"SAM 3 / 3.1 models are restricted under non-commercial research terms."
                )

        # Check if valid SAM 2.1 checkpoint or HuggingFace model ID
        is_valid_filename = ckpt_name in VALID_SAM2_CHECKPOINTS or any(
            ckpt_name.startswith(v.replace(".pt", "")) and ckpt_name.endswith((".pt", ".pth"))
            for v in VALID_SAM2_CHECKPOINTS
        )
        is_valid_hf = ckpt_lower in VALID_HF_MODELS or any(
            ckpt_lower.startswith(hf) for hf in VALID_HF_MODELS
        )
        is_valid_pattern = (
            ("sam2.1" in ckpt_lower or "sam2" in ckpt_lower)
            and ("hiera" in ckpt_lower or ckpt_lower.endswith((".pt", ".pth")))
        )

        if not (is_valid_filename or is_valid_hf or is_valid_pattern):
            raise CommercialPolicyError(
                f"Commercial policy blocked checkpoint '{checkpoint_path}': "
                f"Checkpoint is not a recognized official SAM 2.1 Apache-2.0 checkpoint."
            )

    # 2. Package verification
    sam2_mod = sys.modules.get("sam2")
    if sam2_mod is None:
        try:
            import sam2 as sam2_mod
        except ImportError as exc:
            raise RuntimeError(
                "Official facebookresearch/sam2 package is not installed. "
                "Install via git+https://github.com/facebookresearch/sam2.git"
            ) from exc

    # Check for unofficial PyPI fork indicators
    if getattr(sam2_mod, "_is_unofficial_pypi", False):
        raise RuntimeError(
            "Unofficial PyPI 'sam2' fork detected; official facebookresearch/sam2 is required."
        )

    mod_file = str(getattr(sam2_mod, "__file__", "") or "").lower()
    if "unofficial" in mod_file:
        raise RuntimeError(
            f"Unofficial PyPI 'sam2' fork detected at '{mod_file}'; official facebookresearch/sam2 is required."
        )

    # Check for presence of hiera / video components
    has_video = (
        hasattr(sam2_mod, "sam2_video_predictor")
        or hasattr(sam2_mod, "build_sam2_video_predictor")
        or hasattr(sam2_mod, "SAM2VideoPredictor")
    )
    has_builder = hasattr(sam2_mod, "build_sam")
    has_modeling = hasattr(sam2_mod, "modeling")

    if not (has_video and (has_builder or has_modeling)):
        raise RuntimeError(
            "Unofficial PyPI 'sam2' fork detected: missing official video predictor and hiera backbone components."
        )

    pkg_version = getattr(sam2_mod, "__version__", "2.1")
    return {
        "status": "verified",
        "package": "facebookresearch/sam2",
        "license": "Apache-2.0",
        "version": str(pkg_version),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else "none",
    }


@runtime_checkable
class PreTrackerAdapter(Protocol):
    """Protocol for bounding-box pre-tracking before SAM 2 segmentation."""

    def update_and_associate(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        """Associate detections across frames and assign consistent tracking IDs."""
        ...


class NullTrackerAdapter:
    """Default pass-through adapter preserving box IDs without external tracker."""

    def is_available(self) -> bool:
        """Always available as standard pass-through."""
        return True

    def update_and_associate(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        """Preserve bounding boxes and assign sequential 0-based index IDs."""
        return [
            (i, (float(b[0]), float(b[1]), float(b[2]), float(b[3])))
            for i, b in enumerate(bboxes)
        ]


class ByteTrackAdapterHook:
    """Pluggable ByteTrack adapter hook.

    Uses BYTETracker to associate bounding boxes across frames during occlusions when installed.
    Falls back cleanly to NullTrackerAdapter with is_available() == False if not installed.
    """

    def __init__(
        self,
        tracker: Any = None,
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
    ) -> None:
        self._null_adapter = NullTrackerAdapter()
        self._tracker = tracker

        if self._tracker is None and _BYTE_TRACK_AVAILABLE and BYTETracker is not None:
            try:
                class _ByteArgs:
                    def __init__(self) -> None:
                        self.track_thresh = track_thresh
                        self.match_thresh = match_thresh
                        self.track_buffer = track_buffer
                        self.mot20 = False
                self._tracker = BYTETracker(_ByteArgs(), frame_rate=frame_rate)
            except Exception:
                self._tracker = None

    def is_available(self) -> bool:
        """Return True if an underlying ByteTrack instance is active, False otherwise."""
        return self._tracker is not None

    def update_and_associate(
        self,
        frame_idx: int,
        frame_bgr: np.ndarray,
        bboxes: list[tuple[float, float, float, float]],
    ) -> list[tuple[int, tuple[float, float, float, float]]]:
        """Update tracker with frame detections or fallback cleanly to NullTrackerAdapter."""
        if self._tracker is None:
            return self._null_adapter.update_and_associate(frame_idx, frame_bgr, bboxes)

        if hasattr(self._tracker, "update"):
            if not bboxes:
                return []

            h, w = frame_bgr.shape[:2]
            dets = np.array([[b[0], b[1], b[2], b[3], 1.0] for b in bboxes], dtype=np.float32)

            try:
                online_targets = self._tracker.update(dets, [h, w], [h, w])
                results: list[tuple[int, tuple[float, float, float, float]]] = []
                for t in online_targets:
                    tlbr = getattr(t, "tlbr", None)
                    if tlbr is None and hasattr(t, "to_tlbr"):
                        tlbr = t.to_tlbr()
                    track_id = int(getattr(t, "track_id", 0))
                    if tlbr is not None:
                        results.append((track_id, (float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3]))))
                return results
            except Exception:
                return self._null_adapter.update_and_associate(frame_idx, frame_bgr, bboxes)

        if hasattr(self._tracker, "update_and_associate"):
            return self._tracker.update_and_associate(frame_idx, frame_bgr, bboxes)

        return self._null_adapter.update_and_associate(frame_idx, frame_bgr, bboxes)


def select_best_candidate_mask(
    mask_logits_3d: np.ndarray,
    scores: np.ndarray | Sequence[float] | None = None,
    min_mask_area: int = 16,
    threshold: float = 0.0,
) -> np.ndarray:
    """Select the best candidate mask channel from float32 (3, H, W) SAM 2 output.

    Parameters
    ----------
    mask_logits_3d : np.ndarray
        3D float array of shape (3, H, W) containing candidate mask logits.
    scores : Sequence[float] | None
        Optional IoU prediction scores for each candidate channel.
    min_mask_area : int
        Minimum foreground pixels threshold.
    threshold : float
        Logit threshold for foreground detection (default 0.0).

    Returns
    -------
    np.ndarray
        Selected 2D candidate mask logits of shape (H, W).
    """
    if mask_logits_3d.ndim != 3:
        raise ValueError(f"Expected 3D candidate array (3, H, W), got shape {mask_logits_3d.shape}")

    num_candidates = mask_logits_3d.shape[0]
    if num_candidates == 1:
        return mask_logits_3d[0]

    if scores is not None and len(scores) == num_candidates:
        best_idx = int(np.argmax(scores))
        return mask_logits_3d[best_idx]

    best_idx = 0
    best_score = -float("inf")

    for c in range(num_candidates):
        cand = mask_logits_3d[c]
        fg = cand > threshold
        fg_count = int(np.count_nonzero(fg))

        if fg_count < min_mask_area:
            # Below minimum area: penalize heavily
            score = -1e6 + fg_count
        else:
            # Score based on mean positive logit activation
            mean_logit = float(np.mean(cand[fg]))
            score = mean_logit

        if score > best_score:
            best_score = score
            best_idx = c

    return mask_logits_3d[best_idx]


class ShotVideoTracker:
    """Wraps SAM2VideoPredictor with fail-closed sanity checks, memory cleanup, and archive export."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        model_cfg: str | None = None,
        device: str = "auto",
        pre_tracker: PreTrackerAdapter | None = None,
        predictor: Any = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.model_cfg = model_cfg
        self.pre_tracker: PreTrackerAdapter = pre_tracker if pre_tracker is not None else NullTrackerAdapter()

        # State tracking
        self.inference_state: Any | None = None
        self._frame_count: int = 0
        self._height: int = 0
        self._width: int = 0
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._registered_objects: set[int] = set()
        self._empty_background_objects: set[int] = set()
        self._is_closed: bool = False

        self._owns_predictor = predictor is None
        if predictor is not None:
            self.predictor = predictor
            self.device = device
            # If checkpoint_path is provided, still verify compliance
            if checkpoint_path is not None:
                verify_sam2_installation(checkpoint_path)
        else:
            # Verify official SAM 2 installation and checkpoint
            verify_sam2_installation(checkpoint_path)

            try:
                import torch
                if device == "auto":
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    self.device = device
            except ImportError:
                self.device = "cpu"

            try:
                from sam2.build_sam import build_sam2_video_predictor
            except ImportError as exc:
                raise RuntimeError("Could not import build_sam2_video_predictor from sam2.build_sam") from exc

            resolved_cfg = model_cfg or "configs/sam2.1/sam2.1_hiera_l.yaml"
            self.predictor = build_sam2_video_predictor(
                resolved_cfg,
                str(checkpoint_path) if checkpoint_path else None,
                device=self.device,
            )

    def init_shot(
        self,
        video_frames: list[Path] | list[np.ndarray] | Path | str,
    ) -> Any:
        """Initialize inference state for a new shot, strictly cleaning any previous shot state."""
        # Clean any prior shot state
        self._cleanup_shot()
        self._is_closed = False

        # Handle different video_frames input types
        if isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], np.ndarray):
            self._frame_count = len(video_frames)
            self._height, self._width = video_frames[0].shape[:2]

            # Try passing directly to predictor if supported (e.g. mock predictor)
            try:
                self.inference_state = self.predictor.init_state(video_path=video_frames)
            except (TypeError, ValueError):
                # Official SAM 2 requires path to directory of JPEG images
                import cv2
                self._temp_dir = tempfile.TemporaryDirectory()
                temp_path = Path(self._temp_dir.name)
                for i, frame in enumerate(video_frames):
                    frame_file = temp_path / f"{i:05d}.jpg"
                    cv2.imwrite(str(frame_file), frame)
                self.inference_state = self.predictor.init_state(video_path=str(temp_path))

        elif isinstance(video_frames, list) and len(video_frames) > 0 and isinstance(video_frames[0], (Path, str)):
            frame_paths = [Path(f) for f in video_frames]
            self._frame_count = len(frame_paths)
            first_path = frame_paths[0]
            try:
                from PIL import Image
                with Image.open(first_path) as img:
                    self._width, self._height = img.size
            except Exception:
                pass

            parent_dir = first_path.parent
            parent_images = []
            if parent_dir.is_dir():
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
                parent_images = [
                    f for f in parent_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in valid_exts
                ]

            if len(parent_images) != len(frame_paths):
                # video_frames is a subset/slice of a folder with multiple shots or sequence frames
                self._temp_dir = tempfile.TemporaryDirectory()
                temp_path = Path(self._temp_dir.name)
                for i, src_path in enumerate(frame_paths):
                    dst_path = temp_path / f"{i:05d}.jpg"
                    try:
                        os.link(str(src_path), str(dst_path))
                    except (OSError, NotImplementedError):
                        shutil.copyfile(str(src_path), str(dst_path))
                self.inference_state = self.predictor.init_state(video_path=self._temp_dir.name)
            else:
                video_dir = str(parent_dir)
                self.inference_state = self.predictor.init_state(video_path=video_dir)

        elif isinstance(video_frames, (Path, str)):
            p = Path(video_frames)
            if p.is_dir():
                frames = sorted([
                    f for f in p.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ])
                self._frame_count = len(frames)
                if frames:
                    try:
                        from PIL import Image
                        with Image.open(frames[0]) as img:
                            self._width, self._height = img.size
                    except Exception:
                        pass
                self.inference_state = self.predictor.init_state(video_path=str(p))
            else:
                # Video file
                try:
                    import cv2
                    cap = cv2.VideoCapture(str(p))
                    self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                except Exception:
                    pass
                self.inference_state = self.predictor.init_state(video_path=str(p))
        else:
            raise ValueError(f"Unsupported video_frames type or empty sequence: {type(video_frames)}")

        return self.inference_state

    def add_object_points_or_mask(
        self,
        frame_idx: int,
        obj_id: int,
        points: Any | None = None,
        labels: Any | None = None,
        mask: Any | None = None,
        box: Any | None = None,
    ) -> Any:
        """Register object prompt in the active inference_state."""
        if self.inference_state is None:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")

        obj_id = int(obj_id)
        frame_idx = int(frame_idx)
        self._registered_objects.add(obj_id)

        # Check for empty-background prompt heuristics
        if mask is not None:
            mask_arr = np.asarray(mask)
            if not np.any(mask_arr) or np.count_nonzero(mask_arr) < 16:
                self._empty_background_objects.add(obj_id)
            else:
                self._empty_background_objects.discard(obj_id)
            if hasattr(self.predictor, "add_new_mask"):
                return self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )
            elif hasattr(self.predictor, "add_object_points_or_mask"):
                return self.predictor.add_object_points_or_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask,
                )

        if points is not None or box is not None:
            points_arr = np.asarray(points, dtype=np.float32) if points is not None else None
            box_arr = np.asarray(box, dtype=np.float32) if box is not None else None

            has_valid_box = box_arr is not None and box_arr.size >= 4
            has_positive_point = False

            if points_arr is not None and len(points_arr) > 0:
                if labels is None:
                    labels_arr = np.ones(len(points_arr), dtype=np.int32)
                else:
                    labels_arr = np.asarray(labels, dtype=np.int32)
                # If all labels are 0, this object prompt is background only
                if np.any(labels_arr == 1):
                    has_positive_point = True
                elif np.all(labels_arr == 0) and not has_valid_box:
                    self._empty_background_objects.add(obj_id)
            else:
                labels_arr = None

            if has_positive_point or has_valid_box:
                self._empty_background_objects.discard(obj_id)

            if hasattr(self.predictor, "add_new_points_or_box"):
                return self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points_arr,
                    labels=labels_arr,
                    box=box_arr,
                )
            elif hasattr(self.predictor, "add_object_points_or_mask"):
                return self.predictor.add_object_points_or_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points_arr,
                    labels=labels_arr,
                    box=box_arr,
                )

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        """Propagate prompts across all frames in the shot and yield binarized masks.

        Yields (frame_idx, {obj_id: binary_mask_2d}).
        """
        if self.inference_state is None:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")

        try:
            import torch
            inf_mode = torch.inference_mode()
        except (ImportError, AttributeError, TypeError):
            inf_mode = nullcontext()

        with inf_mode:
            propagation_stream = self.predictor.propagate_in_video(self.inference_state)

            for item in propagation_stream:
                if len(item) == 3:
                    out_frame_idx, out_obj_ids, out_mask_logits = item
                elif len(item) == 2:
                    out_frame_idx, data = item
                    if isinstance(data, dict):
                        out_obj_ids = list(data.keys())
                        out_mask_logits = list(data.values())
                    else:
                        out_obj_ids = list(range(len(data)))
                        out_mask_logits = data
                else:
                    raise ValueError(f"Unexpected propagate_in_video output tuple of length {len(item)}")

                # Convert out_mask_logits to numpy
                if hasattr(out_mask_logits, "detach"):
                    logits_np = out_mask_logits.detach().cpu().numpy()
                elif hasattr(out_mask_logits, "numpy"):
                    logits_np = out_mask_logits.numpy()
                elif isinstance(out_mask_logits, (list, tuple)):
                    logits_np = [
                        (x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))
                        for x in out_mask_logits
                    ]
                else:
                    logits_np = np.asarray(out_mask_logits)

                if isinstance(out_obj_ids, (int, np.integer)):
                    out_obj_ids = [int(out_obj_ids)]

                frame_masks: dict[int, np.ndarray] = {}

                for idx, obj_id in enumerate(out_obj_ids):
                    obj_id_int = int(obj_id)
                    obj_data = logits_np[idx] if isinstance(logits_np, (list, tuple, np.ndarray)) else logits_np

                    # Squeeze or select best candidate
                    if obj_data.ndim == 4:
                        obj_data = np.squeeze(obj_data, axis=0)

                    if obj_data.ndim == 3:
                        if obj_data.shape[0] == 3:
                            # Float32 (3, H, W) 3-candidate multimask output from SAM 2
                            cand_2d = select_best_candidate_mask(
                                obj_data,
                                min_mask_area=min_mask_area,
                                threshold=0.0,
                            )
                        elif obj_data.shape[0] == 1:
                            cand_2d = obj_data[0]
                        elif obj_data.shape[-1] in (1, 3):
                            cand_2d = obj_data[..., 0]
                        else:
                            cand_2d = obj_data[0]
                    elif obj_data.ndim == 2:
                        cand_2d = obj_data
                    else:
                        raise ValueError(f"Unsupported mask output dimensions {obj_data.shape}")

                    # Binarize
                    val_min = float(np.min(cand_2d)) if cand_2d.size > 0 else 0.0
                    val_max = float(np.max(cand_2d)) if cand_2d.size > 0 else 0.0

                    if val_min < 0.0 or val_max > 1.0:
                        # Logits
                        if 0.0 < iou_threshold < 1.0:
                            logit_thresh = math.log(iou_threshold / (1.0 - iou_threshold))
                        else:
                            logit_thresh = 0.0
                        binary = cand_2d > logit_thresh
                    else:
                        # Probability in [0, 1]
                        binary = cand_2d > iou_threshold

                    fg_pixels = int(np.count_nonzero(binary))

                    # Empty-background heuristics:
                    # 1. Fewer than min_mask_area foreground pixels -> all zeros
                    # 2. Prompt placed on empty background -> all zeros
                    # 3. Spurious full-frame mask (>98% coverage) when background flagged -> all zeros
                    is_empty_bg = (obj_id_int in self._empty_background_objects)
                    is_spurious_full = (fg_pixels >= int(cand_2d.size * 0.98)) and is_empty_bg

                    if fg_pixels < min_mask_area or is_empty_bg or is_spurious_full:
                        final_mask = np.zeros(cand_2d.shape, dtype=np.uint8)
                    else:
                        final_mask = (binary.astype(np.uint8) * 255)

                    frame_masks[obj_id_int] = final_mask

                yield (int(out_frame_idx), frame_masks)

    def _cleanup_shot(self) -> None:
        """Internal cleanup helper for per-shot lifecycle."""
        if getattr(self, "inference_state", None) is not None:
            try:
                if hasattr(self.predictor, "reset_state"):
                    self.predictor.reset_state(self.inference_state)
            except Exception:
                pass
            self.inference_state = None

        if getattr(self, "_temp_dir", None) is not None:
            try:
                self._temp_dir.cleanup()
            except Exception:
                pass
            self._temp_dir = None

        if hasattr(self, "_registered_objects"):
            self._registered_objects.clear()
        if hasattr(self, "_empty_background_objects"):
            self._empty_background_objects.clear()

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def close(self) -> None:
        """Close tracker, free inference state and owned predictor, and purge GPU cache."""
        self._cleanup_shot()
        if getattr(self, "_owns_predictor", False) and self.predictor is not None:
            predictor = self.predictor
            self.predictor = None
            try:
                if hasattr(predictor, "to"):
                    predictor.to("cpu")
                elif hasattr(getattr(predictor, "model", None), "to"):
                    predictor.model.to("cpu")
            except Exception:
                pass
            del predictor
            gc.collect()
            torch_mod = sys.modules.get("torch")
            if torch_mod is not None:
                try:
                    if torch_mod.cuda.is_available():
                        torch_mod.cuda.empty_cache()
                except Exception:
                    pass
        self._is_closed = True

    def __enter__(self) -> ShotVideoTracker:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def track_and_save_archive(
        self,
        shot_id: str,
        output_npz_path: Path | str,
        object_metadatas: list[MaskObjectMetadata] | None = None,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> ShotMaskArchive:
        """Track full shot and serialize results to compressed .masks.npz archive."""
        collected: dict[int, dict[int, np.ndarray]] = {}
        for f_idx, f_masks in self.track_shot(min_mask_area=min_mask_area, iou_threshold=iou_threshold):
            collected[f_idx] = f_masks

        # Frame count
        if self._frame_count > 0:
            total_frames = self._frame_count
        elif collected:
            total_frames = max(collected.keys()) + 1
        else:
            total_frames = 0

        # Height & width
        h, w = self._height, self._width
        if h == 0 or w == 0:
            for f_masks in collected.values():
                for m in f_masks.values():
                    if m is not None and m.ndim == 2:
                        h, w = m.shape
                        break
                if h > 0 and w > 0:
                    break

        # Object metadata
        all_obj_ids = set(self._registered_objects)
        for f_masks in collected.values():
            all_obj_ids.update(f_masks.keys())

        meta_map: dict[int, MaskObjectMetadata] = {}
        if object_metadatas is not None:
            for om in object_metadatas:
                meta_map[om.obj_id] = om

        for oid in sorted(all_obj_ids):
            if oid not in meta_map:
                meta_map[oid] = MaskObjectMetadata(
                    obj_id=oid,
                    name=f"obj_{oid}",
                    entity_type="object",
                )

        objects = [meta_map[oid] for oid in sorted(meta_map.keys())]

        archive = ShotMaskArchive(
            shot_id=shot_id,
            frame_count=total_frames,
            height=h,
            width=w,
            objects=objects,
        )

        for t in range(total_frames):
            f_masks = collected.get(t, {})
            for obj in objects:
                m = f_masks.get(obj.obj_id)
                if m is not None and np.any(m):
                    archive.set_mask(t, obj.obj_id, m)
                else:
                    archive.set_mask(t, obj.obj_id, None)

        archive.save_npz(output_npz_path)
        return archive


def main(argv: Sequence[str] | None = None) -> int:
    """CLI interface for SAM 2 installation verification and shot tracking."""
    parser = argparse.ArgumentParser(description="SAM 2.1 Video Tracker CLI")
    parser.add_argument("--verify", action="store_true", help="Verify SAM 2 installation and exit.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to SAM 2 checkpoint or HF ID.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu').")
    args = parser.parse_args(argv)

    if args.verify:
        try:
            info = verify_sam2_installation(args.checkpoint)
            print(json.dumps(info, indent=2))
            return 0
        except Exception as e:
            print(f"SAM 2 Verification Failed: {e}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
