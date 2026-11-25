"""Pluggable VideoSegmenterBackend protocol, registry, and adapters for shot-local segmentation.

Supports 9 shot-local segmentation/matting backends all outputting ARP's compressed
RLE .masks.npz format (ai_remaster_gui/rle_masks.py) for seamless consumption by
osi_masked_lab.py and color_qc.py:

Commercial-safe OSI (ALLOWED_OSI):
  1. sam21         - Official facebookresearch/sam2 SAM 2.1 (Apache-2.0)
  2. samurai       - Kalman motion/affinity-stabilized wrapper over SAM 2.1 (Apache-2.0)
  3. edgetam       - Meta EdgeTAM fast video tracker adapter (Apache-2.0)
  4. cutie         - hkchengrex/Cutie object-memory VOS tracker adapter (MIT)
  5. grounded_sam2 - Grounding DINO (Apache-2.0) + SAM 2.1 / EdgeTAM / SAMURAI (Apache-2.0)
  6. rf_detr_seg   - Roboflow RF-DETR-Seg instance segmentation adapter (Apache-2.0)
  7. videomama     - VideoMaMa video matting adapter binarized to RLE masks (Apache-2.0)

Non-Commercial / Benchmark Only (HARD_BLOCK_COMMERCIAL unless explicitly overridden):
  8. sam3          - Meta SAM 3 text+video promptable tracker (Meta SAM License)
  9. sam31         - Meta SAM 3.1 text+video promptable tracker (Meta SAM License)
"""

from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import math
import sys
import tempfile
from collections.abc import Generator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from .commercial_policy import (
    CommercialPolicyError,
    PolicyDecision,
    enabled as commercial_guard_enabled,
    evaluate_component,
)
from .config import ROOT
from .managed_model_envs import (
    env_for_model,
    inspect_env,
    is_managed_env_active,
    run_in_managed_env,
)
from .rle_masks import MaskObjectMetadata, ShotMaskArchive

VIDEO_EXTS: frozenset[str] = frozenset({".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"})
IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"})


def release_gpu_memory() -> None:
    """Force garbage collection and purge PyTorch CUDA allocator cache if loaded."""
    gc.collect()
    torch_mod = sys.modules.get("torch")
    if torch_mod is None:
        return
    try:
        if torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
    except Exception:
        pass


def normalize_text_prompts(text_prompts: list[str] | str | None) -> list[str]:
    """Normalize string / sequence text prompts into a clean list of non-empty strings."""
    if text_prompts is None:
        return []
    if isinstance(text_prompts, str):
        raw = [part.strip() for part in text_prompts.replace("\n", ".").split(".")]
        return [p for p in raw if p]
    return [str(p).strip() for p in text_prompts if str(p).strip()]


def load_video_frames_bgr(video_frames: list[Path] | list[np.ndarray] | Path | str) -> list[np.ndarray]:
    """Load frames as BGR uint8 numpy arrays from arrays, file lists, directories, or a video."""
    if isinstance(video_frames, list):
        if not video_frames:
            raise ValueError("video_frames sequence must not be empty.")
        if isinstance(video_frames[0], np.ndarray):
            return [np.asarray(f) for f in video_frames]
        import cv2

        loaded: list[np.ndarray] = []
        for item in video_frames:
            frame = cv2.imread(str(item), cv2.IMREAD_COLOR)
            if frame is None:
                raise FileNotFoundError(f"Could not read frame image: {item}")
            loaded.append(frame)
        return loaded

    if isinstance(video_frames, (Path, str)):
        p = Path(video_frames)
        if not p.exists():
            raise FileNotFoundError(f"Video frames path does not exist: {p}")
        import cv2

        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            if not files:
                raise ValueError(f"No image frames found in directory: {p}")
            return load_video_frames_bgr(files)

        if p.suffix.lower() in IMAGE_EXTS:
            return load_video_frames_bgr([p])

        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file for mask tracking: {p}")
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"Decoded 0 frames from video file: {p}")
        return frames

    raise ValueError(f"Unsupported video_frames type: {type(video_frames).__name__}")


def _to_numpy(tensor_or_array: Any) -> np.ndarray:
    if hasattr(tensor_or_array, "detach"):
        return tensor_or_array.detach().cpu().numpy()
    if hasattr(tensor_or_array, "cpu") and hasattr(tensor_or_array, "numpy"):
        return tensor_or_array.cpu().numpy()
    if hasattr(tensor_or_array, "numpy"):
        return tensor_or_array.numpy()
    return np.asarray(tensor_or_array)


def _select_2d_candidate(mask_data: np.ndarray, min_mask_area: int = 16) -> np.ndarray:
    arr = np.asarray(mask_data)
    while arr.ndim > 3:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim == 3:
        if arr.shape[0] == 3:
            from scripts.sam2_video_tracker import select_best_candidate_mask

            return select_best_candidate_mask(arr, min_mask_area=min_mask_area, threshold=0.0)
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] in (1, 3):
            return arr[..., 0]
        return arr[0]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Unsupported mask array shape {arr.shape}")


def binarize_mask(
    raw_mask: Any,
    *,
    min_mask_area: int = 16,
    threshold: float = 0.5,
    is_logit: bool | None = None,
) -> np.ndarray:
    """Binarize logits, probabilities, alpha mattes, or uint8 arrays to 2D uint8 (0 or 255)."""
    cand_2d = _select_2d_candidate(_to_numpy(raw_mask), min_mask_area=min_mask_area)
    if cand_2d.size == 0:
        return np.zeros(cand_2d.shape, dtype=np.uint8)

    if cand_2d.dtype == np.bool_:
        binary = cand_2d
    else:
        cand_f = cand_2d.astype(np.float32, copy=False)
        val_min = float(np.min(cand_f))
        val_max = float(np.max(cand_f))
        if is_logit is True or (is_logit is None and (val_min < 0.0 or (1.0 < val_max <= 30.0))):
            if 0.0 < threshold < 1.0:
                logit_thresh = math.log(threshold / (1.0 - threshold))
            else:
                logit_thresh = 0.0
            binary = cand_f > logit_thresh
        elif val_max > 1.0:
            # 0..255 alpha / uint8 range
            scaled_thresh = threshold * 255.0 if threshold <= 1.0 else threshold
            binary = cand_f >= scaled_thresh
        else:
            binary = cand_f >= threshold

    if int(np.count_nonzero(binary)) < int(min_mask_area):
        return np.zeros(cand_2d.shape, dtype=np.uint8)
    return binary.astype(np.uint8) * 255


def _inference_context() -> Any:
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and hasattr(torch_mod, "inference_mode"):
        return torch_mod.inference_mode()
    return nullcontext()


def normalize_prompts(prompts: dict[int, dict[str, Any]] | None) -> list[tuple[int, int, dict[str, Any], MaskObjectMetadata]]:
    """Normalize prompt dictionaries into (obj_id, frame_idx, prompt_dict, MaskObjectMetadata)."""
    if not prompts:
        return []
    normalized: list[tuple[int, int, dict[str, Any], MaskObjectMetadata]] = []
    for key, raw_pdict in prompts.items():
        pdict = dict(raw_pdict or {})
        if "obj_id" in pdict:
            obj_id = int(pdict["obj_id"])
            frame_idx = int(pdict.get("frame_idx", key))
        else:
            obj_id = int(key)
            frame_idx = int(pdict.get("frame_idx", 0))
        meta = MaskObjectMetadata(
            obj_id=obj_id,
            name=str(pdict.get("name", f"obj_{obj_id}")),
            entity_type=str(pdict.get("entity_type", "object")),
            entity_id=str(pdict.get("entity_id", "")),
            variant_id=str(pdict.get("variant_id", "")),
        )
        normalized.append((obj_id, frame_idx, pdict, meta))
    return normalized


def _parse_predictor_stream_item(item: Any) -> tuple[int, list[int], list[np.ndarray]]:
    """Normalize SAM2 / EdgeTAM / SAM3 propagate_in_video stream tuples."""
    if len(item) == 3:
        out_frame_idx, out_obj_ids, out_mask_logits = item
    elif len(item) == 2:
        out_frame_idx, data = item
        if isinstance(data, Mapping):
            out_obj_ids = list(data.keys())
            out_mask_logits = list(data.values())
        else:
            out_obj_ids = list(range(len(data)))
            out_mask_logits = data
    else:
        raise ValueError(f"Unexpected propagate_in_video output tuple of length {len(item)}")

    if isinstance(out_obj_ids, (int, np.integer)):
        obj_ids = [int(out_obj_ids)]
    else:
        obj_ids = [int(oid) for oid in _to_numpy(out_obj_ids).tolist()] if hasattr(out_obj_ids, "detach") else [int(oid) for oid in out_obj_ids]

    if isinstance(out_mask_logits, (list, tuple)):
        masks_list = [_to_numpy(x) for x in out_mask_logits]
    else:
        arr = _to_numpy(out_mask_logits)
        masks_list = [arr[i] for i in range(len(obj_ids))]

    return int(out_frame_idx), obj_ids, masks_list


def _box_from_mask(mask_2d: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask_2d > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0.0 else 0.0


def _solve_linear_assignment(cost_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Exact pure-NumPy Hungarian / Jonker-Volgenant style shortest augmenting path assignment.

    Minimizes total cost for any rectangular (rows x cols) matrix with no SciPy dependency.
    """
    n_rows, n_cols = cost_matrix.shape
    if n_rows == 0 or n_cols == 0:
        return []

    transposed = False
    mat = cost_matrix.astype(np.float64, copy=True)
    if n_rows > n_cols:
        mat = mat.T
        n_rows, n_cols = mat.shape
        transposed = True

    u = np.zeros(n_rows + 1, dtype=np.float64)
    v = np.zeros(n_cols + 1, dtype=np.float64)
    p = np.zeros(n_cols + 1, dtype=np.int64)
    way = np.zeros(n_cols + 1, dtype=np.int64)

    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n_cols + 1, float("inf"), dtype=np.float64)
        used = np.zeros(n_cols + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = float("inf")
            j1 = 0
            row_vals = mat[i0 - 1]
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = row_vals[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: list[tuple[int, int]] = []
    for j in range(1, n_cols + 1):
        if p[j] > 0:
            r = int(p[j] - 1)
            c = int(j - 1)
            pairs.append((c, r) if transposed else (r, c))
    pairs.sort()
    return pairs


class KalmanBoxTracker:
    """8-state constant-velocity linear Kalman filter over bounding boxes [cx, cy, w, h]."""

    def __init__(self, obj_id: int, bbox: tuple[float, float, float, float]) -> None:
        self.obj_id = int(obj_id)
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h

        self.x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            self.H[i, i] = 1.0

        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 250.0, 250.0, 100.0, 100.0])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 2.0, 2.0])
        self.R = np.diag([4.0, 4.0, 6.0, 6.0])
        self.hits = 1
        self.time_since_update = 0

    def predict(self) -> tuple[float, float, float, float]:
        self.x = self.F @ self.x
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        z = np.array([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dtype=np.float64)

        y = z - (self.H @ self.x)
        s = self.H @ self.P @ self.H.T + self.R
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.x[2] = max(1.0, self.x[2])
        self.x[3] = max(1.0, self.x[3])
        identity = np.eye(8, dtype=np.float64)
        self.P = (identity - k @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        return (float(cx - 0.5 * w), float(cy - 0.5 * h), float(cx + 0.5 * w), float(cy + 0.5 * h))


class SamuraiMotionGate:
    """Kalman bbox motion & appearance-free affinity gating to prevent ID swaps across occlusions."""

    def __init__(
        self,
        *,
        min_affinity: float = 0.15,
        iou_weight: float = 0.65,
        dist_weight: float = 0.25,
        area_weight: float = 0.10,
        max_age: int = 30,
        allow_new_tracks: bool = False,
    ) -> None:
        self.min_affinity = float(min_affinity)
        self.iou_weight = float(iou_weight)
        self.dist_weight = float(dist_weight)
        self.area_weight = float(area_weight)
        self.max_age = int(max_age)
        self.allow_new_tracks = bool(allow_new_tracks)
        self.tracks: dict[int, KalmanBoxTracker] = {}
        self._next_track_id: int = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_track_id = 1

    def register_track(self, obj_id: int, bbox: tuple[float, float, float, float]) -> None:
        oid = int(obj_id)
        self.tracks[oid] = KalmanBoxTracker(oid, bbox)
        self._next_track_id = max(self._next_track_id, oid + 1)

    def _compute_affinity(
        self,
        pred_box: tuple[float, float, float, float],
        det_box: tuple[float, float, float, float],
        time_since_update: int,
    ) -> float:
        iou = _bbox_iou(pred_box, det_box)
        pcx = 0.5 * (pred_box[0] + pred_box[2])
        pcy = 0.5 * (pred_box[1] + pred_box[3])
        dcx = 0.5 * (det_box[0] + det_box[2])
        dcy = 0.5 * (det_box[1] + det_box[3])
        pw = max(1.0, pred_box[2] - pred_box[0])
        ph = max(1.0, pred_box[3] - pred_box[1])
        dw = max(1.0, det_box[2] - det_box[0])
        dh = max(1.0, det_box[3] - det_box[1])

        scale = max(math.hypot(pw, ph) * (1.0 + 0.15 * max(0, time_since_update - 1)), 1.0)
        dist_norm = math.hypot(dcx - pcx, dcy - pcy) / scale
        dist_aff = math.exp(-dist_norm)

        area_p = pw * ph
        area_d = dw * dh
        area_aff = min(area_p, area_d) / max(area_p, area_d, 1.0)

        return self.iou_weight * iou + self.dist_weight * dist_aff + self.area_weight * area_aff

    def select_candidate(self, obj_id: int, candidates_3d: np.ndarray, min_mask_area: int = 16) -> np.ndarray:
        """Pick the candidate channel maximizing logits weighted by Kalman predicted bbox IoU."""
        if candidates_3d.ndim != 3:
            return _select_2d_candidate(candidates_3d, min_mask_area=min_mask_area)
        tracker = self.tracks.get(int(obj_id))
        if tracker is None or candidates_3d.shape[0] == 1:
            from scripts.sam2_video_tracker import select_best_candidate_mask

            return select_best_candidate_mask(candidates_3d, min_mask_area=min_mask_area, threshold=0.0)

        pred_box = tracker.bbox
        best_idx = 0
        best_score = -float("inf")
        for c in range(candidates_3d.shape[0]):
            cand = candidates_3d[c]
            fg = cand > 0.0
            fg_count = int(np.count_nonzero(fg))
            if fg_count < min_mask_area:
                score = -1e6 + fg_count
            else:
                box = _box_from_mask(fg.astype(np.uint8))
                iou = _bbox_iou(pred_box, box) if box is not None else 0.0
                mean_logit = float(np.mean(cand[fg]))
                score = mean_logit + 4.0 * iou
            if score > best_score:
                best_score = score
                best_idx = c
        return candidates_3d[best_idx]

    def associate_frame(
        self,
        detections: Sequence[tuple[int | None, np.ndarray]],
        *,
        frame_hw: tuple[int, int] | None = None,
    ) -> dict[int, np.ndarray]:
        """Match predicted Kalman tracks to candidate detections and return {stable_obj_id: mask_2d}."""
        valid_dets: list[tuple[int | None, np.ndarray, tuple[float, float, float, float]]] = []
        for raw_id, mask in detections:
            if mask is None or not np.any(mask):
                continue
            box = _box_from_mask(mask)
            if box is not None:
                valid_dets.append((raw_id, mask, box))

        if not self.tracks and not valid_dets:
            return {}

        if not self.tracks:
            result: dict[int, np.ndarray] = {}
            for raw_id, mask, box in valid_dets:
                oid = int(raw_id) if raw_id is not None else self._next_track_id
                self._next_track_id = max(self._next_track_id, oid + 1)
                self.register_track(oid, box)
                result[oid] = mask
            return result

        # Predict every active track forward 1 step
        track_ids = sorted(self.tracks.keys())
        pred_boxes: dict[int, tuple[float, float, float, float]] = {}
        for tid in track_ids:
            pred_boxes[tid] = self.tracks[tid].predict()

        output: dict[int, np.ndarray] = {}
        matched_det_indices: set[int] = set()

        if valid_dets:
            n_t = len(track_ids)
            n_d = len(valid_dets)
            aff_matrix = np.zeros((n_t, n_d), dtype=np.float64)
            for ti, tid in enumerate(track_ids):
                t_obj = self.tracks[tid]
                for di, (_raw_id, _mask, dbox) in enumerate(valid_dets):
                    aff_matrix[ti, di] = self._compute_affinity(pred_boxes[tid], dbox, t_obj.time_since_update)

            pairs = _solve_linear_assignment(-aff_matrix)
            for ti, di in pairs:
                score = aff_matrix[ti, di]
                if score >= self.min_affinity:
                    tid = track_ids[ti]
                    _raw_id, mask, dbox = valid_dets[di]
                    self.tracks[tid].update(dbox)
                    output[tid] = mask
                    matched_det_indices.add(di)

        if self.allow_new_tracks:
            for di, (raw_id, mask, dbox) in enumerate(valid_dets):
                if di in matched_det_indices:
                    continue
                oid = int(raw_id) if (raw_id is not None and int(raw_id) not in self.tracks) else self._next_track_id
                self._next_track_id = max(self._next_track_id, oid + 1)
                self.register_track(oid, dbox)
                output[oid] = mask

        if frame_hw is not None:
            h, w = frame_hw
            for tid in track_ids:
                if tid not in output and self.tracks[tid].time_since_update <= self.max_age:
                    output[tid] = np.zeros((h, w), dtype=np.uint8)

        return output


@runtime_checkable
class VideoSegmenterBackend(Protocol):
    """Protocol implemented by every pluggable shot-local video segmentation backend."""

    name: str
    component_id: str
    supports_text_prompts: bool

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> Any:
        """Initialize per-shot inference state for a single video shot."""
        ...

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        """Register object prompts (points, boxes, masks, and optional text prompts)."""
        ...

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        """Propagate prompts across all frames in the initialized shot."""
        ...

    def close(self) -> None:
        """Free per-shot state, temporary folders, model references, and CUDA cache."""
        ...


class BaseVideoSegmenter:
    """Shared lifecycle, VRAM cleanup, and ShotMaskArchive serializer for all backends."""

    name: str = "base"
    component_id: str = "base"
    supports_text_prompts: bool = False

    def __init__(self, *, checkpoint_path: Path | str | None = None, device: str = "auto") -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._frame_count: int = 0
        self._height: int = 0
        self._width: int = 0
        self._object_metadatas: dict[int, MaskObjectMetadata] = {}
        self._is_closed: bool = False

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        torch_mod = sys.modules.get("torch")
        if torch_mod is None:
            try:
                import torch as torch_mod  # noqa: F401
            except ImportError:
                return "cpu"
        try:
            return "cuda" if torch_mod.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _release_gpu_memory(self) -> None:
        release_gpu_memory()

    def close(self) -> None:
        self._object_metadatas.clear()
        self._is_closed = True
        self._release_gpu_memory()

    def __enter__(self) -> BaseVideoSegmenter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if not getattr(self, "_is_closed", False):
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
        """Execute track_shot and serialize results into compressed .masks.npz archive."""
        collected: dict[int, dict[int, np.ndarray]] = {}
        for f_idx, f_masks in self.track_shot(min_mask_area=min_mask_area, iou_threshold=iou_threshold):
            collected[int(f_idx)] = f_masks

        total_frames = self._frame_count
        if total_frames <= 0:
            total_frames = (max(collected.keys()) + 1) if collected else 0

        h, w = self._height, self._width
        if h == 0 or w == 0:
            for f_masks in collected.values():
                for m in f_masks.values():
                    if m is not None and m.ndim == 2:
                        h, w = m.shape
                        break
                if h > 0 and w > 0:
                    break

        all_obj_ids = set(self._object_metadatas.keys())
        for f_masks in collected.values():
            all_obj_ids.update(f_masks.keys())

        meta_map: dict[int, MaskObjectMetadata] = dict(self._object_metadatas)
        if object_metadatas is not None:
            for om in object_metadatas:
                meta_map[om.obj_id] = om

        for oid in sorted(all_obj_ids):
            if oid not in meta_map:
                meta_map[oid] = MaskObjectMetadata(obj_id=oid, name=f"obj_{oid}", entity_type="object")

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


def _require_optional_module(module_name: str, install_hint: str) -> Any:
    mod = sys.modules.get(module_name)
    if mod is not None:
        return mod
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"Optional segmenter dependency '{module_name}' is not installed. {install_hint}"
        ) from exc


def _stage_frames_if_needed(video_frames: list[Path] | list[np.ndarray] | Path | str) -> tuple[Any, tuple[int, int, int], tempfile.TemporaryDirectory | None]:
    """Prepare video_frames for SAM-style init_state(video_path=...) predictors."""
    if isinstance(video_frames, list) and video_frames and isinstance(video_frames[0], np.ndarray):
        return video_frames, (len(video_frames), video_frames[0].shape[0], video_frames[0].shape[1]), None
    if isinstance(video_frames, list) and video_frames and isinstance(video_frames[0], (Path, str)):
        paths = [Path(f) for f in video_frames]
        h, w = 0, 0
        try:
            from PIL import Image

            with Image.open(paths[0]) as im:
                w, h = im.size
        except Exception:
            pass
        td = tempfile.TemporaryDirectory()
        temp_path = Path(td.name)
        import cv2

        for i, src in enumerate(paths):
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is not None:
                if h == 0 or w == 0:
                    h, w = img.shape[:2]
                cv2.imwrite(str(temp_path / f"{i:05d}.jpg"), img)
        return td.name, (len(paths), h, w), td
    if isinstance(video_frames, (Path, str)):
        p = Path(video_frames)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            h, w = 0, 0
            if files:
                try:
                    from PIL import Image

                    with Image.open(files[0]) as im:
                        w, h = im.size
                except Exception:
                    pass
            return str(p), (len(files), h, w), None
        h, w, count = 0, 0, 0
        try:
            import cv2

            cap = cv2.VideoCapture(str(p))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        except Exception:
            pass
        return str(p), (count, h, w), None
    raise ValueError(f"Unsupported or empty video_frames input: {type(video_frames).__name__}")


class SAM21SegmenterBackend(BaseVideoSegmenter):
    """Official facebookresearch/sam2 SAM 2.1 backend wrapping ShotVideoTracker (Apache-2.0)."""

    name = "sam21"
    component_id = "sam2"
    supports_text_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        model_cfg: str | None = None,
        device: str = "auto",
        predictor: Any = None,
        pre_tracker: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        from scripts.sam2_video_tracker import ShotVideoTracker

        self._tracker = ShotVideoTracker(
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            device=device,
            pre_tracker=pre_tracker,
            predictor=predictor,
        )

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> Any:
        self._is_closed = False
        self._object_metadatas.clear()
        state = self._tracker.init_shot(video_frames)
        self._frame_count = self._tracker._frame_count
        self._height = self._tracker._height
        self._width = self._tracker._width
        return state

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        if normalize_text_prompts(text_prompts):
            raise ValueError(f"Segmenter '{self.name}' does not support text prompts; use grounded_sam2, sam3, or sam31.")
        metas: list[MaskObjectMetadata] = []
        for obj_id, frame_idx, pdict, meta in normalize_prompts(prompts):
            self._object_metadatas[obj_id] = meta
            metas.append(meta)
            self._tracker.add_object_points_or_mask(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=pdict.get("points"),
                labels=pdict.get("labels"),
                mask=pdict.get("mask"),
                box=pdict.get("box"),
            )
        return metas

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        yield from self._tracker.track_shot(min_mask_area=min_mask_area, iou_threshold=iou_threshold)

    def close(self) -> None:
        if getattr(self, "_tracker", None) is not None:
            self._tracker.close()
        super().close()


class SAMURAISegmenterBackend(BaseVideoSegmenter):
    """Kalman-stabilized SAMURAI wrapper over official SAM 2.1 Apache-2.0 weights."""

    name = "samurai"
    component_id = "samurai"
    supports_text_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        model_cfg: str | None = None,
        device: str = "auto",
        predictor: Any = None,
        min_affinity: float = 0.15,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        from scripts.sam2_video_tracker import ShotVideoTracker

        self._tracker = ShotVideoTracker(
            checkpoint_path=checkpoint_path,
            model_cfg=model_cfg,
            device=device,
            predictor=predictor,
        )
        self._gate = SamuraiMotionGate(min_affinity=min_affinity, allow_new_tracks=False)
        self._seeded_boxes: dict[int, tuple[float, float, float, float]] = {}

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> Any:
        self._is_closed = False
        self._object_metadatas.clear()
        self._seeded_boxes.clear()
        self._gate.reset()
        state = self._tracker.init_shot(video_frames)
        self._frame_count = self._tracker._frame_count
        self._height = self._tracker._height
        self._width = self._tracker._width
        return state

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        if normalize_text_prompts(text_prompts):
            raise ValueError("Segmenter 'samurai' does not support text prompts; use grounded_sam2, sam3, or sam31.")
        metas: list[MaskObjectMetadata] = []
        for obj_id, frame_idx, pdict, meta in normalize_prompts(prompts):
            self._object_metadatas[obj_id] = meta
            metas.append(meta)
            self._tracker.add_object_points_or_mask(
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=pdict.get("points"),
                labels=pdict.get("labels"),
                mask=pdict.get("mask"),
                box=pdict.get("box"),
            )
            seed_box: tuple[float, float, float, float] | None = None
            if pdict.get("box") is not None and len(pdict["box"]) >= 4:
                b = [float(x) for x in pdict["box"][:4]]
                seed_box = (b[0], b[1], b[2], b[3])
            elif pdict.get("mask") is not None:
                seed_box = _box_from_mask(np.asarray(pdict["mask"]))
            elif pdict.get("points") is not None and len(pdict["points"]) > 0:
                pts = np.asarray(pdict["points"], dtype=np.float32)
                xs, ys = pts[:, 0], pts[:, 1]
                seed_box = (float(xs.min() - 12.0), float(ys.min() - 12.0), float(xs.max() + 12.0), float(ys.max() + 12.0))
            if seed_box is not None:
                self._seeded_boxes[obj_id] = seed_box
                self._gate.register_track(obj_id, seed_box)
        return metas

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        if self._tracker.inference_state is None:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")

        with _inference_context():
            stream = self._tracker.predictor.propagate_in_video(self._tracker.inference_state)
            for item in stream:
                frame_idx, out_obj_ids, out_masks = _parse_predictor_stream_item(item)
                detections: list[tuple[int | None, np.ndarray]] = []
                frame_hw: tuple[int, int] | None = None

                for idx, raw_oid in enumerate(out_obj_ids):
                    raw_arr = out_masks[idx]
                    arr = raw_arr
                    while arr.ndim > 3:
                        arr = np.squeeze(arr, axis=0)
                    if arr.ndim == 3 and arr.shape[0] == 3:
                        cand_2d = self._gate.select_candidate(raw_oid, arr, min_mask_area=min_mask_area)
                    else:
                        cand_2d = _select_2d_candidate(arr, min_mask_area=min_mask_area)
                    bin_mask = binarize_mask(cand_2d, min_mask_area=min_mask_area, threshold=iou_threshold)
                    if frame_hw is None and bin_mask.ndim == 2:
                        frame_hw = bin_mask.shape
                    if raw_oid in self._tracker._empty_background_objects:
                        bin_mask = np.zeros_like(bin_mask)
                    detections.append((raw_oid, bin_mask))

                associated = self._gate.associate_frame(detections, frame_hw=frame_hw)
                if not associated and frame_hw is not None:
                    for raw_oid, bmask in detections:
                        if raw_oid is not None:
                            associated[int(raw_oid)] = bmask
                yield int(frame_idx), associated

    def close(self) -> None:
        if getattr(self, "_tracker", None) is not None:
            self._tracker.close()
        self._gate.reset()
        self._seeded_boxes.clear()
        super().close()


class _PredictorVideoSegmenterBase(BaseVideoSegmenter):
    """Shared SAM-predictor style lifecycle for EdgeTAM, SAM 3, and SAM 3.1."""

    _pkg_name: str = ""
    _install_hint: str = ""
    _default_cfg: str = ""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        model_cfg: str | None = None,
        device: str = "auto",
        predictor: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.model_cfg = model_cfg
        self._owns_predictor = predictor is None
        self.predictor = predictor
        self.inference_state: Any | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._registered_objects: set[int] = set()
        if self.predictor is None:
            self.predictor = self._build_predictor()

    def _build_predictor(self) -> Any:
        mod = _require_optional_module(self._pkg_name, self._install_hint)
        resolved_device = self._resolve_device()
        builder = (
            getattr(mod, "build_sam3_video_predictor", None)
            or getattr(mod, "build_edgetam_video_predictor", None)
            or getattr(mod, "build_sam2_video_predictor", None)
            or getattr(mod, "build_video_predictor", None)
        )
        if builder is None:
            raise RuntimeError(
                f"Package '{self._pkg_name}' does not expose a video predictor builder. {self._install_hint}"
            )
        if self.checkpoint_path is None:
            try:
                return builder(self.model_cfg or self._default_cfg, device=resolved_device)
            except TypeError:
                return builder(device=resolved_device)
        ckpt = str(self.checkpoint_path)
        try:
            return builder(self.model_cfg or self._default_cfg, ckpt, device=resolved_device)
        except TypeError:
            return builder(checkpoint=ckpt, device=resolved_device)

    def _cleanup_shot(self) -> None:
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
        self._registered_objects.clear()
        self._release_gpu_memory()

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> Any:
        self._cleanup_shot()
        self._is_closed = False
        self._object_metadatas.clear()
        staged_arg, (count, h, w), td = _stage_frames_if_needed(video_frames)
        self._temp_dir = td
        self._frame_count = count
        self._height = h
        self._width = w
        try:
            self.inference_state = self.predictor.init_state(video_path=staged_arg)
        except TypeError:
            loaded = load_video_frames_bgr(video_frames)
            self._frame_count = len(loaded)
            self._height, self._width = loaded[0].shape[:2]
            self.inference_state = self.predictor.init_state(video_path=loaded)
        return self.inference_state

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        if self.inference_state is None:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")
        texts = normalize_text_prompts(text_prompts)
        if texts and not self.supports_text_prompts:
            raise ValueError(
                f"Segmenter '{self.name}' does not support text prompts; use grounded_sam2, sam3, or sam31."
            )

        metas: list[MaskObjectMetadata] = []
        for obj_id, frame_idx, pdict, meta in normalize_prompts(prompts):
            self._registered_objects.add(obj_id)
            self._object_metadatas[obj_id] = meta
            metas.append(meta)
            text = str(pdict.get("text") or "").strip() or None

            if pdict.get("mask") is not None and hasattr(self.predictor, "add_new_mask"):
                self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=pdict["mask"],
                )
                continue

            if text and hasattr(self.predictor, "add_prompt"):
                self.predictor.add_prompt(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    text=text,
                    box=pdict.get("box"),
                    points=pdict.get("points"),
                    labels=pdict.get("labels"),
                )
                continue

            if hasattr(self.predictor, "add_new_points_or_box"):
                pts = np.asarray(pdict["points"], dtype=np.float32) if pdict.get("points") is not None else None
                lbls = np.asarray(pdict["labels"], dtype=np.int32) if pdict.get("labels") is not None else (np.ones(len(pts), dtype=np.int32) if pts is not None else None)
                box = np.asarray(pdict["box"], dtype=np.float32) if pdict.get("box") is not None else None
                self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pts,
                    labels=lbls,
                    box=box,
                )
            elif hasattr(self.predictor, "add_prompt"):
                self.predictor.add_prompt(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pdict.get("points"),
                    labels=pdict.get("labels"),
                    box=pdict.get("box"),
                )

        if texts:
            next_id = max(self._registered_objects, default=0) + 1
            for offset, text in enumerate(texts):
                obj_id = next_id + offset
                meta = MaskObjectMetadata(obj_id=obj_id, name=text.replace(" ", "_"), entity_type="object")
                self._registered_objects.add(obj_id)
                self._object_metadatas[obj_id] = meta
                metas.append(meta)
                if hasattr(self.predictor, "add_text_prompt"):
                    self.predictor.add_text_prompt(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        text=text,
                    )
                elif hasattr(self.predictor, "add_prompt"):
                    self.predictor.add_prompt(
                        inference_state=self.inference_state,
                        frame_idx=0,
                        obj_id=obj_id,
                        text=text,
                    )
                else:
                    raise RuntimeError(
                        f"{self.name} predictor does not expose add_text_prompt/add_prompt for text query {text!r}."
                    )
        return metas

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        if self.inference_state is None:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")
        with _inference_context():
            for item in self.predictor.propagate_in_video(self.inference_state):
                frame_idx, obj_ids, masks_list = _parse_predictor_stream_item(item)
                frame_masks: dict[int, np.ndarray] = {}
                for idx, oid in enumerate(obj_ids):
                    frame_masks[int(oid)] = binarize_mask(
                        masks_list[idx],
                        min_mask_area=min_mask_area,
                        threshold=iou_threshold,
                    )
                yield int(frame_idx), frame_masks

    def close(self) -> None:
        self._cleanup_shot()
        if self._owns_predictor and self.predictor is not None:
            pred = self.predictor
            self.predictor = None
            try:
                if hasattr(pred, "to"):
                    pred.to("cpu")
            except Exception:
                pass
            del pred
        super().close()


class EdgeTAMSegmenterBackend(_PredictorVideoSegmenterBase):
    """Meta EdgeTAM fast video object tracker adapter (Apache-2.0)."""

    name = "edgetam"
    component_id = "edgetam"
    supports_text_prompts = False
    _pkg_name = "edgetam"
    _install_hint = "Install official Meta EdgeTAM from https://github.com/facebookresearch/EdgeTAM and supply an Apache-2.0 checkpoint."
    _default_cfg = "configs/edgetam.yaml"


class CutieSegmenterBackend(BaseVideoSegmenter):
    """hkchengrex/Cutie object-memory VOS tracker adapter (MIT)."""

    name = "cutie"
    component_id = "cutie"
    supports_text_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()
        self._frames: list[np.ndarray] = []
        self._initial_masks: dict[int, tuple[int, np.ndarray]] = {}

    def _build_runner(self) -> Any:
        cutie_mod = _require_optional_module(
            "cutie",
            "Install Cutie from https://github.com/hkchengrex/Cutie (MIT) before running the cutie segmenter.",
        )
        inference_mod = sys.modules.get("cutie.inference.inference_core")
        if inference_mod is None:
            try:
                inference_mod = importlib.import_module("cutie.inference.inference_core")
            except ImportError as exc:
                raise RuntimeError("cutie.inference.inference_core is unavailable in the installed cutie package.") from exc

        get_model = getattr(cutie_mod, "get_default_model", None)
        if get_model is None:
            model_mod = importlib.import_module("cutie.model.cutie")
            get_model = getattr(model_mod, "get_default_model", None) or getattr(model_mod, "CUTIE", None)
        if get_model is None:
            raise RuntimeError("Could not locate Cutie model factory in installed cutie package.")
        model = get_model(str(self.checkpoint_path)) if self.checkpoint_path else get_model()
        if hasattr(model, "to"):
            model = model.to(self._resolve_device())
        core_cls = getattr(inference_mod, "InferenceCore")
        return core_cls(model, cfg=getattr(model, "cfg", None))

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> dict[str, Any]:
        self._is_closed = False
        self._object_metadatas.clear()
        self._initial_masks.clear()
        if hasattr(self.runner, "clear_memory"):
            self.runner.clear_memory()
        self._frames = load_video_frames_bgr(video_frames)
        self._frame_count = len(self._frames)
        self._height, self._width = self._frames[0].shape[:2]
        return {"frame_count": self._frame_count, "height": self._height, "width": self._width}

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        if normalize_text_prompts(text_prompts):
            raise ValueError("Segmenter 'cutie' does not support text prompts; supply an initial mask or bbox prompt.")
        if not self._frames:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")
        metas: list[MaskObjectMetadata] = []
        for obj_id, frame_idx, pdict, meta in normalize_prompts(prompts):
            self._object_metadatas[obj_id] = meta
            metas.append(meta)
            mask = pdict.get("mask")
            if mask is not None:
                seed_mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
            elif pdict.get("box") is not None:
                seed_mask = np.zeros((self._height, self._width), dtype=np.uint8)
                x1, y1, x2, y2 = [int(round(v)) for v in pdict["box"][:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(self._width, x2), min(self._height, y2)
                seed_mask[y1:y2, x1:x2] = 255
            else:
                raise ValueError(f"Cutie requires an explicit 'mask' or 'box' prompt for object {obj_id}.")
            self._initial_masks[obj_id] = (frame_idx, seed_mask)
        return metas

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        if not self._frames:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")

        # Support injected mock runner exposing step(frame_bgr, mask=..., objects=...)
        # or real Cutie InferenceCore accepting torch tensors.
        active_objects = sorted(self._initial_masks.keys())
        prompt_frame = min((f for f, _ in self._initial_masks.values()), default=0)
        label_map = np.zeros((self._height, self._width), dtype=np.uint8)
        for oid in active_objects:
            f_idx, m = self._initial_masks[oid]
            if f_idx == prompt_frame:
                label_map[m > 0] = oid

        with _inference_context():
            for t in range(prompt_frame, self._frame_count):
                frame = self._frames[t]
                if t == prompt_frame and active_objects:
                    out_probs = self._step_runner(frame, label_map=label_map, object_ids=active_objects)
                else:
                    out_probs = self._step_runner(frame, label_map=None, object_ids=active_objects)
                yield t, self._decode_step_output(out_probs, active_objects, min_mask_area=min_mask_area, threshold=iou_threshold)

    def _step_runner(self, frame: np.ndarray, *, label_map: np.ndarray | None, object_ids: list[int]) -> Any:
        if label_map is not None:
            try:
                return self.runner.step(frame, mask=label_map, objects=object_ids)
            except TypeError:
                return self.runner.step(frame, label_map, objects=object_ids)
        return self.runner.step(frame)

    def _decode_step_output(
        self,
        output: Any,
        object_ids: list[int],
        *,
        min_mask_area: int,
        threshold: float,
    ) -> dict[int, np.ndarray]:
        if isinstance(output, Mapping):
            return {
                int(oid): binarize_mask(mask, min_mask_area=min_mask_area, threshold=threshold, is_logit=False)
                for oid, mask in output.items()
            }
        arr = _to_numpy(output)
        if arr.ndim == 2:
            # Label map output
            return {
                int(oid): binarize_mask((arr == oid).astype(np.uint8) * 255, min_mask_area=min_mask_area)
                for oid in object_ids
            }
        if arr.ndim == 3:
            # Cutie prob volume: channel 0 = background, channels 1..K = objects (or C == len(object_ids))
            if arr.shape[0] == len(object_ids) + 1:
                return {
                    int(oid): binarize_mask(arr[idx + 1], min_mask_area=min_mask_area, threshold=threshold, is_logit=False)
                    for idx, oid in enumerate(object_ids)
                }
            if arr.shape[0] == len(object_ids):
                return {
                    int(oid): binarize_mask(arr[idx], min_mask_area=min_mask_area, threshold=threshold, is_logit=False)
                    for idx, oid in enumerate(object_ids)
                }
        raise ValueError(f"Unsupported Cutie step output shape {arr.shape}")

    def close(self) -> None:
        if getattr(self, "runner", None) is not None and hasattr(self.runner, "clear_memory"):
            try:
                self.runner.clear_memory()
            except Exception:
                pass
        if getattr(self, "_owns_runner", False) and self.runner is not None:
            del self.runner
            self.runner = None
        self._frames.clear()
        self._initial_masks.clear()
        super().close()


class GroundedSAM2SegmenterBackend(BaseVideoSegmenter):
    """Grounding DINO (Apache-2.0) text-prompt box detector feeding sam21/edgetam/samurai."""

    name = "grounded_sam2"
    component_id = "grounded_sam2"
    supports_text_prompts = True

    def __init__(
        self,
        *,
        inner_tracker: str = "sam21",
        checkpoint_path: Path | str | None = None,
        grounding_checkpoint: Path | str | None = None,
        device: str = "auto",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
        detector: Any = None,
        inner_backend: VideoSegmenterBackend | None = None,
        predictor: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        norm_inner = str(inner_tracker or "sam21").strip().lower()
        if norm_inner not in {"sam21", "edgetam", "samurai"}:
            raise ValueError(
                f"Grounded-SAM-2 inner_tracker must be 'sam21', 'edgetam', or 'samurai'; got {inner_tracker!r}."
            )
        self.inner_tracker = norm_inner
        self.grounding_checkpoint = grounding_checkpoint
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._owns_detector = detector is None
        self._detector = detector if detector is not None else self._build_detector()

        if inner_backend is not None:
            self._inner: VideoSegmenterBackend = inner_backend
        elif norm_inner == "samurai":
            self._inner = SAMURAISegmenterBackend(checkpoint_path=checkpoint_path, device=device, predictor=predictor)
        elif norm_inner == "edgetam":
            self._inner = EdgeTAMSegmenterBackend(checkpoint_path=checkpoint_path, device=device, predictor=predictor)
        else:
            self._inner = SAM21SegmenterBackend(checkpoint_path=checkpoint_path, device=device, predictor=predictor)

        self._frames_ref: list[Path] | list[np.ndarray] | Path | str | None = None
        self._loaded_frames: list[np.ndarray] | None = None

    def _build_detector(self) -> Any:
        gd_mod = sys.modules.get("groundingdino")
        if gd_mod is not None:
            return gd_mod
        try:
            return importlib.import_module("groundingdino.util.inference")
        except ImportError:
            pass
        try:
            transformers_mod = importlib.import_module("transformers")
            model_id = str(self.grounding_checkpoint or "IDEA-Research/grounding-dino-tiny")
            processor = transformers_mod.AutoProcessor.from_pretrained(model_id)
            model = transformers_mod.AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            if hasattr(model, "to"):
                model = model.to(self._resolve_device())
            return {"transformers_processor": processor, "transformers_model": model}
        except ImportError as exc:
            raise RuntimeError(
                "Grounding DINO is not installed. Install GroundingDINO (Apache-2.0) or HuggingFace transformers "
                "(https://github.com/IDEA-Research/Grounded-SAM-2) before using grounded_sam2."
            ) from exc

    def _detect_boxes(self, frame_bgr: np.ndarray, text_queries: list[str]) -> list[tuple[str, tuple[float, float, float, float]]]:
        if hasattr(self._detector, "predict_boxes"):
            raw = self._detector.predict_boxes(
                frame_bgr,
                text_queries,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
            )
            results: list[tuple[str, tuple[float, float, float, float]]] = []
            for item in raw:
                if isinstance(item, Mapping):
                    label = str(item.get("label") or text_queries[0])
                    b = [float(x) for x in item["box"][:4]]
                    results.append((label, (b[0], b[1], b[2], b[3])))
                else:
                    label, b = item[0], item[1]
                    results.append((str(label), (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
            return results

        if isinstance(self._detector, Mapping) and "transformers_model" in self._detector:
            import cv2
            from PIL import Image

            proc = self._detector["transformers_processor"]
            model = self._detector["transformers_model"]
            pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            caption = ". ".join(text_queries) + "."
            inputs = proc(images=pil_img, text=caption, return_tensors="pt")
            if hasattr(model, "device"):
                inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with _inference_context():
                outputs = model(**inputs)
            target_sizes = [(pil_img.height, pil_img.width)]
            parsed = proc.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=target_sizes,
            )[0]
            out: list[tuple[str, tuple[float, float, float, float]]] = []
            for lbl, box_t in zip(parsed["labels"], parsed["boxes"]):
                b = _to_numpy(box_t).tolist()
                out.append((str(lbl), (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
            return out

        raise RuntimeError(
            "Configured Grounding DINO detector does not expose predict_boxes() or HuggingFace zero-shot interface."
        )

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> Any:
        self._is_closed = False
        self._object_metadatas.clear()
        self._frames_ref = video_frames
        self._loaded_frames = None
        state = self._inner.init_shot(video_frames)
        self._frame_count = getattr(self._inner, "_frame_count", 0)
        self._height = getattr(self._inner, "_height", 0)
        self._width = getattr(self._inner, "_width", 0)
        return state

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        texts = normalize_text_prompts(text_prompts)
        merged_prompts: dict[int, dict[str, Any]] = dict(prompts) if prompts else {}

        if texts:
            if self._frames_ref is None:
                raise RuntimeError("No active shot initialized. Call init_shot() first.")
            if self._loaded_frames is None:
                self._loaded_frames = load_video_frames_bgr(self._frames_ref)
                if self._frame_count == 0:
                    self._frame_count = len(self._loaded_frames)
                if self._height == 0 or self._width == 0:
                    self._height, self._width = self._loaded_frames[0].shape[:2]

            detections = self._detect_boxes(self._loaded_frames[0], texts)
            next_obj_id = max((int(k) for k in merged_prompts.keys()), default=0) + 1
            for offset, (label, box) in enumerate(detections):
                oid = next_obj_id + offset
                merged_prompts[oid] = {
                    "obj_id": oid,
                    "frame_idx": 0,
                    "box": list(box),
                    "name": label.strip().replace(" ", "_") or f"text_obj_{oid}",
                    "entity_type": "object",
                }

        metas = self._inner.add_prompts(merged_prompts)
        for m in metas:
            self._object_metadatas[m.obj_id] = m
        return metas

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        yield from self._inner.track_shot(min_mask_area=min_mask_area, iou_threshold=iou_threshold)

    def close(self) -> None:
        if getattr(self, "_inner", None) is not None:
            self._inner.close()
        if getattr(self, "_owns_detector", False) and self._detector is not None:
            if isinstance(self._detector, Mapping):
                model = self._detector.get("transformers_model")
                if hasattr(model, "to"):
                    try:
                        model.to("cpu")
                    except Exception:
                        pass
            self._detector = None
        self._loaded_frames = None
        self._frames_ref = None
        super().close()


class RFDETRSegSegmenterBackend(BaseVideoSegmenter):
    """Roboflow RF-DETR-Seg instance segmentation adapter paired with motion association (Apache-2.0)."""

    name = "rf_detr_seg"
    component_id = "rf_detr_seg"
    supports_text_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        confidence_threshold: float = 0.4,
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.confidence_threshold = float(confidence_threshold)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()
        self._frames: list[np.ndarray] = []
        self._gate = SamuraiMotionGate(min_affinity=0.15, allow_new_tracks=True)
        self._target_classes: set[str] = set()

    def _build_runner(self) -> Any:
        rfdetr_mod = _require_optional_module(
            "rfdetr",
            "Install Roboflow rfdetr (Apache-2.0, https://github.com/roboflow/rf-detr) before using rf_detr_seg.",
        )
        cls = (
            getattr(rfdetr_mod, "RFDETRSegPreview", None)
            or getattr(rfdetr_mod, "RFDETRSegMedium", None)
            or getattr(rfdetr_mod, "RFDETRSegLarge", None)
        )
        if cls is None:
            raise RuntimeError("Installed 'rfdetr' package does not expose RFDETRSeg* segmentation classes.")
        if self.checkpoint_path:
            return cls(pretrain_weights=str(self.checkpoint_path), device=self._resolve_device())
        return cls(device=self._resolve_device())

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> dict[str, Any]:
        self._is_closed = False
        self._object_metadatas.clear()
        self._gate.reset()
        self._target_classes.clear()
        self._frames = load_video_frames_bgr(video_frames)
        self._frame_count = len(self._frames)
        self._height, self._width = self._frames[0].shape[:2]
        return {"frame_count": self._frame_count, "height": self._height, "width": self._width}

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        if normalize_text_prompts(text_prompts):
            raise ValueError("Segmenter 'rf_detr_seg' does not support free-form text prompts; use grounded_sam2, sam3, or sam31.")
        metas: list[MaskObjectMetadata] = []
        for obj_id, _frame_idx, pdict, meta in normalize_prompts(prompts):
            self._object_metadatas[obj_id] = meta
            metas.append(meta)
            if pdict.get("box") is not None and len(pdict["box"]) >= 4:
                b = [float(x) for x in pdict["box"][:4]]
                self._gate.register_track(obj_id, (b[0], b[1], b[2], b[3]))
            elif pdict.get("mask") is not None:
                box = _box_from_mask(np.asarray(pdict["mask"]))
                if box is not None:
                    self._gate.register_track(obj_id, box)
            if pdict.get("class_name"):
                self._target_classes.add(str(pdict["class_name"]).strip().lower())
        return metas

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        if not self._frames:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")
        import cv2

        with _inference_context():
            for t, frame_bgr in enumerate(self._frames):
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                preds = self.runner.predict(rgb, threshold=self.confidence_threshold)
                masks_attr = getattr(preds, "mask", None)
                if masks_attr is None and isinstance(preds, Mapping):
                    masks_attr = preds.get("masks")
                raw_ids = getattr(preds, "tracker_id", None) or (preds.get("obj_ids") if isinstance(preds, Mapping) else None)

                detections: list[tuple[int | None, np.ndarray]] = []
                if masks_attr is not None:
                    masks_np = _to_numpy(masks_attr)
                    for i in range(masks_np.shape[0]):
                        bmask = binarize_mask(masks_np[i], min_mask_area=min_mask_area, threshold=iou_threshold)
                        rid = int(raw_ids[i]) if raw_ids is not None and i < len(raw_ids) and raw_ids[i] is not None else None
                        detections.append((rid, bmask))

                associated = self._gate.associate_frame(detections, frame_hw=(self._height, self._width))
                for oid in associated:
                    if oid not in self._object_metadatas:
                        self._object_metadatas[oid] = MaskObjectMetadata(obj_id=oid, name=f"rfdetr_obj_{oid}", entity_type="object")
                yield t, associated

    def close(self) -> None:
        if self._owns_runner and self.runner is not None:
            try:
                if hasattr(self.runner, "model") and hasattr(self.runner.model, "to"):
                    self.runner.model.to("cpu")
            except Exception:
                pass
            del self.runner
            self.runner = None
        self._frames.clear()
        self._gate.reset()
        super().close()


class VideoMaMaSegmenterBackend(BaseVideoSegmenter):
    """VideoMaMa video matting adapter (Apache-2.0) binarizing alpha mattes cleanly into RLE masks."""

    name = "videomama"
    component_id = "videomama"
    supports_text_prompts = False

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str = "auto",
        alpha_threshold: float = 0.5,
        runner: Any = None,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, device=device)
        self.alpha_threshold = float(alpha_threshold)
        self._owns_runner = runner is None
        self.runner = runner if runner is not None else self._build_runner()
        self._frames: list[np.ndarray] = []
        self._guide_masks: dict[int, tuple[int, np.ndarray]] = {}

    def _build_runner(self) -> Any:
        mod = _require_optional_module(
            "videomama",
            "Install VideoMaMa (Apache-2.0, https://github.com/cvlab-kaist/VideoMaMa) before using videomama.",
        )
        factory = getattr(mod, "build_videomama_pipeline", None) or getattr(mod, "VideoMaMaPipeline", None)
        if factory is None:
            raise RuntimeError("Installed 'videomama' package does not expose VideoMaMaPipeline or build_videomama_pipeline.")
        if self.checkpoint_path:
            return factory(checkpoint_path=str(self.checkpoint_path), device=self._resolve_device())
        return factory(device=self._resolve_device())

    def init_shot(self, video_frames: list[Path] | list[np.ndarray] | Path | str) -> dict[str, Any]:
        self._is_closed = False
        self._object_metadatas.clear()
        self._guide_masks.clear()
        self._frames = load_video_frames_bgr(video_frames)
        self._frame_count = len(self._frames)
        self._height, self._width = self._frames[0].shape[:2]
        return {"frame_count": self._frame_count, "height": self._height, "width": self._width}

    def add_prompts(
        self,
        prompts: dict[int, dict[str, Any]] | None = None,
        *,
        text_prompts: list[str] | str | None = None,
    ) -> list[MaskObjectMetadata]:
        if normalize_text_prompts(text_prompts):
            raise ValueError("Segmenter 'videomama' does not support text prompts.")
        if not self._frames:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")
        metas: list[MaskObjectMetadata] = []
        for obj_id, frame_idx, pdict, meta in normalize_prompts(prompts):
            self._object_metadatas[obj_id] = meta
            metas.append(meta)
            mask = pdict.get("mask")
            if mask is not None:
                self._guide_masks[obj_id] = (frame_idx, np.asarray(mask, dtype=np.float32))
            elif pdict.get("box") is not None:
                guide = np.zeros((self._height, self._width), dtype=np.float32)
                x1, y1, x2, y2 = [int(round(v)) for v in pdict["box"][:4]]
                guide[max(0, y1) : min(self._height, y2), max(0, x1) : min(self._width, x2)] = 1.0
                self._guide_masks[obj_id] = (frame_idx, guide)
        if not metas:
            default_meta = MaskObjectMetadata(obj_id=1, name="foreground", entity_type="object")
            self._object_metadatas[1] = default_meta
            metas.append(default_meta)
        return metas

    def track_shot(
        self,
        min_mask_area: int = 16,
        iou_threshold: float = 0.5,
    ) -> Generator[tuple[int, dict[int, np.ndarray]], None, None]:
        if not self._frames:
            raise RuntimeError("No active shot initialized. Call init_shot() first.")
        thresh = self.alpha_threshold if self.alpha_threshold > 0.0 else iou_threshold
        target_ids = sorted(self._object_metadatas.keys()) or [1]
        if 1 not in self._object_metadatas:
            self._object_metadatas[1] = MaskObjectMetadata(obj_id=1, name="foreground", entity_type="object")

        with _inference_context():
            for oid in target_ids:
                guide = self._guide_masks.get(oid)
                try:
                    alphas = self.runner.matte_shot(self._frames, guide_mask=(guide[1] if guide else None))
                except TypeError:
                    alphas = self.runner(self._frames, mask=(guide[1] if guide else None))
                if not hasattr(self, "_cached_mattes"):
                    self._cached_mattes: dict[int, list[np.ndarray]] = {}
                self._cached_mattes[oid] = [_to_numpy(a) for a in alphas]

            try:
                for t in range(self._frame_count):
                    frame_masks: dict[int, np.ndarray] = {}
                    for oid in target_ids:
                        matte_seq = self._cached_mattes.get(oid, [])
                        if t < len(matte_seq):
                            frame_masks[oid] = binarize_mask(
                                matte_seq[t],
                                min_mask_area=min_mask_area,
                                threshold=thresh,
                                is_logit=False,
                            )
                    yield t, frame_masks
            finally:
                if hasattr(self, "_cached_mattes"):
                    self._cached_mattes.clear()

    def close(self) -> None:
        if self._owns_runner and self.runner is not None:
            try:
                if hasattr(self.runner, "to"):
                    self.runner.to("cpu")
            except Exception:
                pass
            del self.runner
            self.runner = None
        self._frames.clear()
        self._guide_masks.clear()
        super().close()


class SAM3SegmenterBackend(_PredictorVideoSegmenterBase):
    """Meta SAM 3 text+video promptable tracker (Meta SAM License - non-commercial/benchmark only)."""

    name = "sam3"
    component_id = "sam3"
    supports_text_prompts = True
    _pkg_name = "sam3"
    _install_hint = "Install official Meta SAM 3 from https://github.com/facebookresearch/sam3 (Meta SAM License, benchmark-only)."
    _default_cfg = "configs/sam3/sam3_video.yaml"


class SAM31SegmenterBackend(SAM3SegmenterBackend):
    """Meta SAM 3.1 text+video promptable tracker (Meta SAM License - non-commercial/benchmark only)."""

    name = "sam31"
    component_id = "sam3.1"
    supports_text_prompts = True
    _default_cfg = "configs/sam3/sam3.1_video.yaml"


@dataclass(frozen=True)
class SegmenterSpec:
    """Descriptor for a registered shot-local video segmentation backend."""

    name: str
    label: str
    component_id: str
    license: str
    commercial_tier: str  # "allowed_osi" or "hard_blocked"
    supports_text_prompts: bool
    factory: Callable[..., VideoSegmenterBackend]
    aliases: tuple[str, ...] = ()
    extra_component_ids: tuple[str, ...] = ()
    description: str = ""


class VideoSegmenterRegistry:
    """Pluggable registry mapping segmenter names/aliases to VideoSegmenterBackend factories."""

    def __init__(self) -> None:
        self._specs: dict[str, SegmenterSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: SegmenterSpec) -> None:
        canonical = spec.name.strip().lower()
        self._specs[canonical] = spec
        self._aliases[canonical] = canonical
        for alias in spec.aliases:
            self._aliases[alias.strip().lower()] = canonical

    def resolve_name(self, name: str) -> str:
        norm = str(name or "").strip().lower().replace("-", "_")
        # Also test direct lowercase lookup before underscore normalization
        direct = str(name or "").strip().lower()
        for candidate in (direct, norm):
            if candidate in self._aliases:
                return self._aliases[candidate]
        available = ", ".join(self._specs.keys())
        raise ValueError(f"Unknown video segmenter {name!r}. Supported segmenters: {available}")

    def has(self, name: str) -> bool:
        try:
            self.resolve_name(name)
            return True
        except ValueError:
            return False

    def get_spec(self, name: str) -> SegmenterSpec:
        return self._specs[self.resolve_name(name)]

    def list_specs(self) -> list[SegmenterSpec]:
        return list(self._specs.values())

    def is_available(self, name: str) -> bool:
        return self.availability_info(name)["available"]

    def availability_info(
        self,
        name: str,
        *,
        settings: Mapping[str, Mapping[str, str]] | None = None,
        inner_tracker: str = "sam21",
    ) -> dict[str, Any]:
        spec = self.get_spec(name)
        env_name = env_for_model(spec.name) or ""
        local_ok = _video_backend_locally_available(spec.name, inner_tracker=inner_tracker)
        if local_ok:
            return {
                "available": True,
                "runtime": "in_process",
                "env_name": env_name,
                "env_status": inspect_env(env_name, settings=settings)["status"] if env_name else "missing",
            }
        if env_name:
            env_info = inspect_env(env_name, settings=settings)
            if env_info["ready"]:
                return {
                    "available": True,
                    "runtime": f"managed_env:{env_name}",
                    "env_name": env_name,
                    "env_status": "ready",
                }
            return {
                "available": False,
                "runtime": "missing",
                "env_name": env_name,
                "env_status": env_info["status"],
            }
        return {
            "available": False,
            "runtime": "missing",
            "env_name": "",
            "env_status": "missing",
        }

    def check_policy(
        self,
        name: str,
        *,
        settings: Mapping[str, Mapping[str, str]] | None = None,
        allow_non_commercial_benchmark: bool = False,
        checkpoint_path: Path | str | None = None,
    ) -> list[PolicyDecision]:
        """Validate a segmenter (and any composite components) against Commercial Model Guard."""
        spec = self.get_spec(name)
        all_ids = [spec.component_id, *spec.extra_component_ids]
        decisions = [
            evaluate_component(
                cid,
                file_path=(checkpoint_path if cid == spec.component_id else None),
                settings=settings,
                component_type="tool",
            )
            for cid in all_ids
        ]

        guard_on = (not allow_non_commercial_benchmark) and commercial_guard_enabled(settings)
        if guard_on:
            for dec in decisions:
                if not dec.allowed:
                    raise CommercialPolicyError(
                        f"Commercial Model Guard blocked segmenter '{spec.name}' ({dec.component_id}): {dec.reason}"
                    )
        return decisions

    def create(
        self,
        name: str,
        *,
        settings: Mapping[str, Mapping[str, str]] | None = None,
        allow_non_commercial_benchmark: bool = False,
        **kwargs: Any,
    ) -> VideoSegmenterBackend:
        """Enforce policy and instantiate the requested VideoSegmenterBackend."""
        spec = self.get_spec(name)
        self.check_policy(
            spec.name,
            settings=settings,
            allow_non_commercial_benchmark=allow_non_commercial_benchmark,
            checkpoint_path=kwargs.get("checkpoint_path"),
        )
        return spec.factory(**kwargs)


def _build_default_registry() -> VideoSegmenterRegistry:
    reg = VideoSegmenterRegistry()
    reg.register(
        SegmenterSpec(
            name="sam21",
            label="SAM 2.1 (Official Apache-2.0)",
            component_id="sam2",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            factory=SAM21SegmenterBackend,
            aliases=("sam2", "sam2.1", "sam-2.1"),
            description="Official facebookresearch/sam2 SAM 2.1 video object predictor.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="samurai",
            label="SAMURAI (Kalman-stabilized SAM 2.1, Apache-2.0)",
            component_id="samurai",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            factory=SAMURAISegmenterBackend,
            aliases=("samurai-sam21",),
            description="Kalman bbox motion & affinity gating over official SAM 2.1 Apache-2.0 weights.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="edgetam",
            label="Meta EdgeTAM (Fast VOS, Apache-2.0)",
            component_id="edgetam",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            factory=EdgeTAMSegmenterBackend,
            aliases=("edge-tam",),
            description="Meta EdgeTAM fast shot-local video segmentation adapter.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="cutie",
            label="Cutie VOS (Object Memory, MIT)",
            component_id="cutie",
            license="MIT",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            factory=CutieSegmenterBackend,
            aliases=("hkchengrex-cutie",),
            description="Independent MIT-licensed object-memory video object segmentation tracker.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="grounded_sam2",
            label="Grounded-SAM-2 (Text + SAM 2.1, Apache-2.0)",
            component_id="grounded_sam2",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=True,
            factory=GroundedSAM2SegmenterBackend,
            aliases=("grounded-sam2", "grounded-sam-2"),
            extra_component_ids=("grounding_dino", "sam2"),
            description="Grounding DINO zero-shot text detector feeding SAM 2.1 / EdgeTAM / SAMURAI.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="rf_detr_seg",
            label="RF-DETR-Seg (Real-Time Instance Seg, Apache-2.0)",
            component_id="rf_detr_seg",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            factory=RFDETRSegSegmenterBackend,
            aliases=("rf-detr-seg", "rfdetr-seg", "rfdetr"),
            description="Roboflow Apache-2.0 RF-DETR-Seg frame instance masks with Kalman ID association.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="videomama",
            label="VideoMaMa (Video Matting, Apache-2.0)",
            component_id="videomama",
            license="Apache-2.0",
            commercial_tier="allowed_osi",
            supports_text_prompts=False,
            factory=VideoMaMaSegmenterBackend,
            aliases=("video-mama",),
            description="Apache-2.0 VideoMaMa video matting adapter binarized into compressed RLE masks.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="sam3",
            label="SAM 3 (Meta SAM License — Non-Commercial / Benchmark Only)",
            component_id="sam3",
            license="Meta SAM License",
            commercial_tier="hard_blocked",
            supports_text_prompts=True,
            factory=SAM3SegmenterBackend,
            aliases=("sam-3",),
            description="Meta SAM 3 text+video promptable tracker restricted to non-commercial benchmark runs.",
        )
    )
    reg.register(
        SegmenterSpec(
            name="sam31",
            label="SAM 3.1 (Meta SAM License — Non-Commercial / Benchmark Only)",
            component_id="sam3.1",
            license="Meta SAM License",
            commercial_tier="hard_blocked",
            supports_text_prompts=True,
            factory=SAM31SegmenterBackend,
            aliases=("sam3.1", "sam-3.1"),
            description="Meta SAM 3.1 text+video promptable tracker restricted to non-commercial benchmark runs.",
        )
    )
    return reg


DEFAULT_SEGMENTER_REGISTRY: VideoSegmenterRegistry = _build_default_registry()


def _record_segmentation_compliance(
    *,
    compliance_output_dir: Path | str,
    spec: SegmenterSpec,
    shot_id: str,
    output_npz_path: Path,
    checkpoint_path: Path | str | None,
    settings: Mapping[str, Any] | None,
    allow_non_commercial_benchmark: bool,
) -> None:
    from .compliance import append_compliance_record

    eff_settings: dict[str, Any] = {
        k: dict(v) if isinstance(v, Mapping) else v
        for k, v in (settings.items() if settings else [])
    }
    global_cfg = dict(eff_settings.get("global", {})) if isinstance(eff_settings.get("global"), Mapping) else {}
    if allow_non_commercial_benchmark:
        global_cfg["non_commercial_benchmark_override"] = "true"
    elif "commercial_model_guard" not in global_cfg:
        global_cfg["commercial_model_guard"] = "true"
    eff_settings["global"] = global_cfg

    models: list[dict[str, Any]] = [
        {
            "model_id": spec.component_id,
            "component_type": "tool",
            "file_path": str(checkpoint_path) if checkpoint_path else None,
        }
    ]
    for extra_cid in spec.extra_component_ids:
        models.append({"model_id": extra_cid, "component_type": "tool"})

    is_nc_run = bool(
        allow_non_commercial_benchmark
        or spec.commercial_tier == "hard_blocked"
        or str(global_cfg.get("non_commercial_benchmark_override", "false")).lower() == "true"
        or str(global_cfg.get("commercial_model_guard", "true")).lower() == "false"
    )

    append_compliance_record(
        output_dir=compliance_output_dir,
        stage="segmentation",
        models=models,
        settings=eff_settings,
        metadata={
            "shot_id": shot_id,
            "segmenter": spec.name,
            "output_npz_path": str(output_npz_path),
            "non_commercial_benchmark_run": is_nc_run,
        },
    )


_VIDEO_PROBE_MODULES: dict[str, tuple[str, ...]] = {
    "sam21": ("sam2",),
    "samurai": ("sam2",),
    "edgetam": ("edgetam",),
    "cutie": ("cutie",),
    "grounded_sam2": ("groundingdino", "transformers"),
    "rf_detr_seg": ("rfdetr",),
    "videomama": ("videomama",),
    "sam3": ("sam3",),
    "sam31": ("sam3",),
}


def _video_module_available(module_names: Sequence[str]) -> bool:
    for name in module_names:
        if name in sys.modules:
            return True
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except Exception:
            continue
    return False


def _video_backend_locally_available(name: str, *, inner_tracker: str = "sam21") -> bool:
    canonical = str(name or "").strip().lower()
    probes = _VIDEO_PROBE_MODULES.get(canonical, ())
    if not _video_module_available(probes):
        return False
    if canonical == "grounded_sam2":
        inner_probes = _VIDEO_PROBE_MODULES.get(str(inner_tracker or "sam21").strip().lower(), ("sam2",))
        return _video_module_available(inner_probes)
    return True


def _serialize_prompts_for_subprocess(
    prompts: dict[int, dict[str, Any]] | None,
    stage_dir: Path,
) -> Path | None:
    if not prompts:
        return None
    from PIL import Image

    serialized: dict[str, Any] = {}
    for key, raw in prompts.items():
        entry: dict[str, Any] = {}
        for k, v in dict(raw or {}).items():
            if k == "mask" and v is not None:
                mask_arr = (np.asarray(v) > 0).astype(np.uint8) * 255
                mask_path = stage_dir / f"prompt_{key}_seed.png"
                Image.fromarray(mask_arr, mode="L").save(mask_path, format="PNG")
                entry["mask_path"] = str(mask_path)
            elif isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            else:
                entry[k] = v
        serialized[str(key)] = entry
    prompts_file = stage_dir / "prompts.json"
    prompts_file.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
    return prompts_file


def _dispatch_shot_in_managed_env(
    *,
    env_name: str,
    spec: SegmenterSpec,
    shot_id: str,
    frames: list[np.ndarray] | list[Path] | Path | str,
    prompts: dict[int, dict[str, Any]] | None,
    target_path: Path,
    text_prompts: list[str] | str | None,
    checkpoint_path: Path | str | None,
    device: str,
    min_mask_area: int,
    iou_threshold: float,
    alpha_threshold: float,
    settings: Mapping[str, Mapping[str, str]] | None,
    allow_non_commercial_benchmark: bool,
) -> Path:
    import cv2

    with tempfile.TemporaryDirectory(prefix="arp-vid-env-") as tmpdir:
        tmp_path = Path(tmpdir)
        if isinstance(frames, list):
            if not frames:
                raise ValueError("video_frames sequence must not be empty.")
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(frames[0], np.ndarray):
                for idx, arr in enumerate(frames):
                    cv2.imwrite(str(frames_dir / f"{idx:05d}.png"), np.asarray(arr))
            else:
                for idx, src_p in enumerate(frames):
                    src = Path(src_p)
                    dst = frames_dir / f"{idx:05d}{src.suffix or '.png'}"
                    dst.write_bytes(src.read_bytes())
            frames_arg = str(frames_dir)
        else:
            frames_arg = str(frames)

        prompts_file = _serialize_prompts_for_subprocess(prompts, tmp_path)
        argv: list[str] = [
            "--segmenter",
            spec.name,
            "--shot-id",
            shot_id,
            "--frames",
            frames_arg,
            "--output",
            str(target_path),
            "--device",
            str(device),
            "--min-mask-area",
            str(int(min_mask_area)),
            "--iou-threshold",
            str(float(iou_threshold)),
            "--alpha-threshold",
            str(float(alpha_threshold)),
            "--compliance-output-dir",
            "",
        ]
        if prompts_file is not None:
            argv.extend(["--prompts-json", str(prompts_file)])
        for t_prompt in normalize_text_prompts(text_prompts):
            argv.extend(["--text-prompts", t_prompt])
        if checkpoint_path is not None:
            argv.extend(["--checkpoint", str(checkpoint_path)])
        if allow_non_commercial_benchmark or spec.commercial_tier == "hard_blocked":
            argv.append("--allow-non-commercial-benchmark")

        run_in_managed_env(
            env_name,
            ROOT / "scripts" / "track_shot_masks.py",
            argv,
            settings=settings,
            allow_non_commercial=bool(allow_non_commercial_benchmark or spec.commercial_tier == "hard_blocked"),
        )

    if not target_path.is_file():
        raise RuntimeError(
            f"Managed env '{env_name}' did not produce expected mask archive at '{target_path}'."
        )
    return target_path


def run_shot_segmentation(
    *,
    shot_id: str,
    frames: list[np.ndarray] | list[Path] | Path | str,
    prompts: dict[int, dict[str, Any]] | None = None,
    output_npz_path: Path | str,
    segmenter: str = "sam21",
    text_prompts: list[str] | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: str = "auto",
    object_metadatas: list[MaskObjectMetadata] | None = None,
    min_mask_area: int = 16,
    iou_threshold: float = 0.5,
    alpha_threshold: float = 0.5,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    allow_non_commercial_benchmark: bool = False,
    compliance_output_dir: Path | str | None = None,
    registry: VideoSegmenterRegistry | None = None,
    **backend_kwargs: Any,
) -> Path:
    """Run a registered VideoSegmenterBackend end-to-end and write .masks.npz + optional compliance log."""
    reg = registry or DEFAULT_SEGMENTER_REGISTRY
    spec = reg.get_spec(segmenter)

    # Always evaluate commercial policy FIRST (fail-closed).
    reg.check_policy(
        spec.name,
        settings=settings,
        allow_non_commercial_benchmark=allow_non_commercial_benchmark,
        checkpoint_path=checkpoint_path,
    )

    init_kwargs: dict[str, Any] = dict(backend_kwargs)
    if checkpoint_path is not None:
        init_kwargs.setdefault("checkpoint_path", checkpoint_path)
    init_kwargs.setdefault("device", device)
    if spec.name == "videomama":
        init_kwargs.setdefault("alpha_threshold", alpha_threshold)

    target_path = Path(output_npz_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    has_injected_runner = any(
        init_kwargs.get(k) is not None
        for k in ("predictor", "pre_tracker", "runner", "detector", "inner_backend")
    )
    local_ok = _video_backend_locally_available(
        spec.name,
        inner_tracker=str(init_kwargs.get("inner_tracker", "sam21")),
    )

    if not has_injected_runner and not local_ok:
        env_name = env_for_model(spec.name)
        if env_name and not is_managed_env_active():
            env_info = inspect_env(env_name, settings=settings)
            if env_info["ready"]:
                _dispatch_shot_in_managed_env(
                    env_name=env_name,
                    spec=spec,
                    shot_id=shot_id,
                    frames=frames,
                    prompts=prompts,
                    target_path=target_path,
                    text_prompts=text_prompts,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    min_mask_area=min_mask_area,
                    iou_threshold=iou_threshold,
                    alpha_threshold=alpha_threshold,
                    settings=settings,
                    allow_non_commercial_benchmark=allow_non_commercial_benchmark,
                )
                if compliance_output_dir is not None:
                    _record_segmentation_compliance(
                        compliance_output_dir=compliance_output_dir,
                        spec=spec,
                        shot_id=shot_id,
                        output_npz_path=target_path,
                        checkpoint_path=checkpoint_path,
                        settings=settings,
                        allow_non_commercial_benchmark=allow_non_commercial_benchmark,
                    )
                return target_path
        target_env = env_name or "sam2_family"
        raise RuntimeError(
            f"Video segmenter '{spec.name}' dependencies are not installed locally and managed environment '{target_env}' is not ready. "
            f"Install it via 'python scripts/setup_model_env.py install {target_env}' or the GUI 'Install Environment' button."
        )

    backend = reg.create(
        spec.name,
        settings=settings,
        allow_non_commercial_benchmark=allow_non_commercial_benchmark,
        **init_kwargs,
    )
    with backend:
        backend.init_shot(frames)
        if prompts or normalize_text_prompts(text_prompts):
            backend.add_prompts(prompts, text_prompts=text_prompts)
        backend.track_and_save_archive(
            shot_id=shot_id,
            output_npz_path=target_path,
            object_metadatas=object_metadatas,
            min_mask_area=min_mask_area,
            iou_threshold=iou_threshold,
        )

    if compliance_output_dir is not None:
        _record_segmentation_compliance(
            compliance_output_dir=compliance_output_dir,
            spec=spec,
            shot_id=shot_id,
            output_npz_path=target_path,
            checkpoint_path=checkpoint_path,
            settings=settings,
            allow_non_commercial_benchmark=allow_non_commercial_benchmark,
        )

    return target_path


__all__ = [
    "BaseVideoSegmenter",
    "CutieSegmenterBackend",
    "DEFAULT_SEGMENTER_REGISTRY",
    "EdgeTAMSegmenterBackend",
    "GroundedSAM2SegmenterBackend",
    "KalmanBoxTracker",
    "RFDETRSegSegmenterBackend",
    "SAM21SegmenterBackend",
    "SAM31SegmenterBackend",
    "SAM3SegmenterBackend",
    "SAMURAISegmenterBackend",
    "SamuraiMotionGate",
    "SegmenterSpec",
    "VideoMaMaSegmenterBackend",
    "VideoSegmenterBackend",
    "VideoSegmenterRegistry",
    "binarize_mask",
    "load_video_frames_bgr",
    "normalize_prompts",
    "normalize_text_prompts",
    "release_gpu_memory",
    "run_shot_segmentation",
]
