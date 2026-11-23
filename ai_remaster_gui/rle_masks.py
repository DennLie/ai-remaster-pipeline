from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

VALID_ENTITY_TYPES: frozenset[str] = frozenset({"character", "object", "location"})


def encode_mask_rle(mask: np.ndarray) -> np.ndarray:
    """Encode a 2D binary, boolean, or uint8 mask into alternating run lengths starting with 0s.

    Parameters
    ----------
    mask : np.ndarray
        2D binary, boolean, or integer/float mask of shape (H, W).
        Non-zero values are considered foreground (1).

    Returns
    -------
    np.ndarray
        1D int32 array of run lengths alternating between 0s and 1s,
        starting with the count of 0s at the start.
    """
    if not isinstance(mask, np.ndarray):
        mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-dimensional (H, W), got shape {mask.shape}")

    flat = mask.ravel()
    n = flat.size
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    bflat = (flat != 0)
    diff_idx = np.nonzero(bflat[:-1] != bflat[1:])[0]
    if diff_idx.size == 0:
        # Uniform mask: either all 0s or all 1s
        if bflat[0]:
            # Starts with 0 zeros, followed by n ones
            return np.array([0, n], dtype=np.int32)
        else:
            # Starts with n zeros
            return np.array([n], dtype=np.int32)

    # Compute run lengths between transition indices
    runs = np.diff(np.concatenate(([-1], diff_idx, [n - 1]))).astype(np.int32)
    if bflat[0]:
        # Starts with 1: prepend count of 0 zeros
        runs = np.concatenate(([np.int32(0)], runs))

    return runs


def decode_mask_rle(
    rle_lengths: np.ndarray | Sequence[int],
    height: int,
    width: int,
    dtype: np.dtype | type = np.uint8,
    values_255: bool = True,
) -> np.ndarray:
    """Decode a 1D int32 alternating run-length array back into a 2D mask.

    Parameters
    ----------
    rle_lengths : np.ndarray | Sequence[int]
        1D int32 array of alternating run lengths starting with count of 0s.
    height : int
        Height (H) of the target mask.
    width : int
        Width (W) of the target mask.
    dtype : np.dtype | type
        Output array dtype. Defaults to np.uint8.
    values_255 : bool
        If True and dtype is uint8, foreground (1s) is mapped to 255.
        If False and dtype is uint8, foreground is mapped to 1.

    Returns
    -------
    np.ndarray
        2D mask of shape (height, width).
    """
    if height < 0 or width < 0:
        raise ValueError(f"Mask dimensions must be non-negative, got height={height}, width={width}")

    total_pixels = int(height) * int(width)
    if total_pixels == 0:
        return np.zeros((height, width), dtype=dtype)

    rle = np.asarray(rle_lengths, dtype=np.int32)
    if rle.ndim != 1:
        rle = rle.ravel()

    if rle.size == 0:
        raise ValueError(f"Empty RLE lengths cannot decode non-empty mask ({height}x{width})")

    if (rle < 0).any():
        raise ValueError("RLE run lengths must be non-negative")

    total_rle_sum = int(np.sum(rle, dtype=np.int64))
    if total_rle_sum != total_pixels:
        raise ValueError(
            f"RLE run lengths sum ({total_rle_sum}) does not match expected total pixels ({total_pixels} = {height}x{width})"
        )

    # Values alternate: 0, foreground, 0, foreground, ...
    vals = np.zeros(len(rle), dtype=dtype)
    is_bool = (dtype is bool) or (dtype == np.bool_)
    if is_bool:
        fg_val = True
    elif np.issubdtype(dtype, np.integer) and values_255 and np.dtype(dtype).type == np.uint8:
        fg_val = 255
    else:
        fg_val = 1
    vals[1::2] = fg_val

    flat = np.repeat(vals, rle)
    return flat.reshape((height, width))


@dataclass
class MaskObjectMetadata:
    """Metadata describing a tracked entity in the Color Bible and mask archives."""

    obj_id: int
    name: str
    entity_type: str
    entity_id: str = ""
    variant_id: str = ""

    def __post_init__(self) -> None:
        self.obj_id = int(self.obj_id)
        self.name = str(self.name)
        norm_type = str(self.entity_type).strip().lower()
        if norm_type not in VALID_ENTITY_TYPES:
            raise ValueError(
                f"Invalid entity_type: '{self.entity_type}'. Must be one of: {sorted(VALID_ENTITY_TYPES)}"
            )
        self.entity_type = norm_type
        self.entity_id = str(self.entity_id)
        self.variant_id = str(self.variant_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "obj_id": self.obj_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "variant_id": self.variant_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MaskObjectMetadata:
        """Construct MaskObjectMetadata from dictionary."""
        return cls(
            obj_id=int(data["obj_id"]),
            name=str(data.get("name", f"obj_{data['obj_id']}")),
            entity_type=str(data.get("entity_type", "object")),
            entity_id=str(data.get("entity_id", "")),
            variant_id=str(data.get("variant_id", "")),
        )


class ShotMaskArchive:
    """Holds per-shot multi-object Run-Length-Encoded (RLE) masks across T frames for K objects.

    Supports compressed NPZ storage with random-access slice decoding.
    """

    def __init__(
        self,
        shot_id: str,
        frame_count: int,
        height: int,
        width: int,
        objects: list[MaskObjectMetadata] | None = None,
    ) -> None:
        self.shot_id = str(shot_id)
        self.frame_count = int(frame_count)
        self.height = int(height)
        self.width = int(width)
        if self.frame_count < 0 or self.height < 0 or self.width < 0:
            raise ValueError("frame_count, height, and width must be non-negative")

        self.objects: list[MaskObjectMetadata] = []
        self._object_map: dict[int, MaskObjectMetadata] = {}
        self._obj_id_to_idx: dict[int, int] = {}
        if objects:
            for obj in objects:
                self.add_object(obj)

        # In-memory transient masks: mapping (obj_id, frame_idx) -> np.ndarray (1D int32 RLE) or None
        self._masks: dict[tuple[int, int], np.ndarray | None] = {}

        # Backing compressed data when loaded from NPZ
        self._rle_lengths: np.ndarray | None = None
        self._rle_index: np.ndarray | None = None
        self._archive_path: Path | None = None

    def add_object(self, metadata: MaskObjectMetadata) -> None:
        """Register or update an object's metadata in the archive."""
        if not isinstance(metadata, MaskObjectMetadata):
            raise TypeError(f"Expected MaskObjectMetadata, got {type(metadata)}")
        if metadata.obj_id in self._object_map:
            idx = self._obj_id_to_idx[metadata.obj_id]
            self.objects[idx] = metadata
            self._object_map[metadata.obj_id] = metadata
            return
        self._obj_id_to_idx[metadata.obj_id] = len(self.objects)
        self.objects.append(metadata)
        self._object_map[metadata.obj_id] = metadata

    def get_object(self, obj_id: int) -> MaskObjectMetadata | None:
        """Retrieve metadata for an object ID."""
        return self._object_map.get(int(obj_id))

    def set_mask(self, frame_idx: int, obj_id: int, mask_2d: np.ndarray | None) -> None:
        """Store mask for a given frame index and object ID."""
        frame_idx = int(frame_idx)
        obj_id = int(obj_id)
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"frame_idx {frame_idx} out of range [0, {self.frame_count})")
        if obj_id not in self._object_map:
            raise ValueError(f"Object ID {obj_id} is not registered in archive objects")

        if mask_2d is None:
            self._masks[(obj_id, frame_idx)] = None
            return

        if not isinstance(mask_2d, np.ndarray) or mask_2d.ndim != 2:
            raise ValueError(f"mask_2d must be a 2D numpy array, got shape {getattr(mask_2d, 'shape', None)}")
        if mask_2d.shape != (self.height, self.width):
            raise ValueError(
                f"Mask shape {mask_2d.shape} does not match archive dimensions ({self.height}, {self.width})"
            )

        rle = encode_mask_rle(mask_2d)
        self._masks[(obj_id, frame_idx)] = rle

    def clear_mask(self, frame_idx: int, obj_id: int) -> None:
        """Remove/clear the mask for given frame index and object ID."""
        self.set_mask(frame_idx, obj_id, None)

    def has_mask(self, frame_idx: int, obj_id: int) -> bool:
        """Check whether a mask is present for given frame index and object ID."""
        frame_idx = int(frame_idx)
        obj_id = int(obj_id)
        if not (0 <= frame_idx < self.frame_count) or (obj_id not in self._object_map):
            return False
        if (obj_id, frame_idx) in self._masks:
            return self._masks[(obj_id, frame_idx)] is not None
        if self._rle_index is not None:
            k = self._obj_id_to_idx.get(obj_id)
            if k is not None and k < self._rle_index.shape[0]:
                offset, length = self._rle_index[k, frame_idx]
                return bool(length > 0 and offset >= 0)
        return False

    def get_mask(
        self,
        frame_idx: int,
        obj_id: int,
        dtype: np.dtype | type = np.uint8,
        values_255: bool = True,
    ) -> np.ndarray | None:
        """Retrieve decoded 2D mask for given frame index and object ID.

        Returns None if no mask is present for that object at that frame.
        """
        frame_idx = int(frame_idx)
        obj_id = int(obj_id)
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"frame_idx {frame_idx} out of range [0, {self.frame_count})")
        if obj_id not in self._object_map:
            return None

        # Check in-memory overrides first
        if (obj_id, frame_idx) in self._masks:
            rle = self._masks[(obj_id, frame_idx)]
            if rle is None:
                return None
            return decode_mask_rle(rle, self.height, self.width, dtype=dtype, values_255=values_255)

        # Check loaded archive index
        if self._rle_index is not None and self._rle_lengths is not None:
            k = self._obj_id_to_idx.get(obj_id)
            if k is not None and k < self._rle_index.shape[0]:
                offset, length = self._rle_index[k, frame_idx]
                if length <= 0 or offset < 0:
                    return None
                rle = self._rle_lengths[offset : offset + length]
                return decode_mask_rle(rle, self.height, self.width, dtype=dtype, values_255=values_255)

        return None

    def get_frame_masks(
        self,
        frame_idx: int,
        dtype: np.dtype | type = np.uint8,
        values_255: bool = True,
    ) -> dict[int, np.ndarray]:
        """Retrieve all object masks present at given frame index."""
        if not (0 <= frame_idx < self.frame_count):
            raise ValueError(f"frame_idx {frame_idx} out of range [0, {self.frame_count})")
        result: dict[int, np.ndarray] = {}
        for obj in self.objects:
            m = self.get_mask(frame_idx, obj.obj_id, dtype=dtype, values_255=values_255)
            if m is not None:
                result[obj.obj_id] = m
        return result

    def stored_mask_count(self) -> int:
        """Return total number of active masks stored across all objects and frames."""
        count = 0
        for obj in self.objects:
            for t in range(self.frame_count):
                if self.has_mask(t, obj.obj_id):
                    count += 1
        return count

    def _prepare_archive_arrays(self) -> tuple[str, np.ndarray, np.ndarray]:
        """Consolidate internal mask state into (metadata_json, rle_lengths, rle_index)."""
        num_objects = len(self.objects)
        new_rle_index = np.full((num_objects, self.frame_count, 2), -1, dtype=np.int64)
        new_rle_index[:, :, 1] = 0

        rle_chunks: list[np.ndarray] = []
        current_offset = 0

        for k, obj in enumerate(self.objects):
            obj_id = obj.obj_id
            for t in range(self.frame_count):
                rle = None
                if (obj_id, t) in self._masks:
                    rle = self._masks[(obj_id, t)]
                elif self._rle_index is not None and self._rle_lengths is not None:
                    orig_k = self._obj_id_to_idx.get(obj_id)
                    if orig_k is not None and orig_k < self._rle_index.shape[0]:
                        off, ln = self._rle_index[orig_k, t]
                        if ln > 0 and off >= 0:
                            rle = self._rle_lengths[off : off + ln]

                if rle is not None and len(rle) > 0:
                    chunk = np.asarray(rle, dtype=np.int32)
                    chunk_len = len(chunk)
                    rle_chunks.append(chunk)
                    new_rle_index[k, t, 0] = current_offset
                    new_rle_index[k, t, 1] = chunk_len
                    current_offset += chunk_len
                else:
                    new_rle_index[k, t, 0] = -1
                    new_rle_index[k, t, 1] = 0

        consolidated_rle = np.concatenate(rle_chunks) if rle_chunks else np.zeros(0, dtype=np.int32)

        metadata_dict = {
            "version": 1,
            "shot_id": self.shot_id,
            "frame_count": int(self.frame_count),
            "height": int(self.height),
            "width": int(self.width),
            "objects": [obj.to_dict() for obj in self.objects],
            "compression": "rle_npz",
        }
        metadata_json = json.dumps(metadata_dict, indent=2)
        return metadata_json, consolidated_rle, new_rle_index

    def save_npz(self, path: Path | str) -> None:
        """Save multi-object mask archive to compressed .masks.npz format."""
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        metadata_json, consolidated_rle, new_rle_index = self._prepare_archive_arrays()

        np.savez_compressed(
            target_path,
            metadata=metadata_json,
            rle_lengths=consolidated_rle,
            rle_index=new_rle_index,
        )

        self._rle_lengths = consolidated_rle
        self._rle_index = new_rle_index
        self._masks.clear()
        self._obj_id_to_idx = {obj.obj_id: i for i, obj in enumerate(self.objects)}
        self._archive_path = target_path.resolve()

    @classmethod
    def load_npz(cls, path: Path | str) -> ShotMaskArchive:
        """Load a compressed .masks.npz archive."""
        target_path = Path(path)
        if not target_path.is_file():
            raise FileNotFoundError(f"Mask archive file not found: {target_path}")

        with np.load(target_path, allow_pickle=False) as npz:
            if "metadata" not in npz:
                raise ValueError(f"Invalid .masks.npz archive (missing 'metadata'): {target_path}")
            if "rle_lengths" not in npz:
                raise ValueError(f"Invalid .masks.npz archive (missing 'rle_lengths'): {target_path}")
            if "rle_index" not in npz:
                raise ValueError(f"Invalid .masks.npz archive (missing 'rle_index'): {target_path}")

            meta_raw = npz["metadata"].item()
            if isinstance(meta_raw, bytes):
                meta_raw = meta_raw.decode("utf-8")
            meta_dict = json.loads(str(meta_raw))

            rle_lengths = np.array(npz["rle_lengths"], dtype=np.int32)
            rle_index = np.array(npz["rle_index"], dtype=np.int64)

        shot_id = str(meta_dict.get("shot_id", ""))
        frame_count = int(meta_dict.get("frame_count", 0))
        height = int(meta_dict.get("height", 0))
        width = int(meta_dict.get("width", 0))

        objects_data = meta_dict.get("objects", [])
        objects = [MaskObjectMetadata.from_dict(d) for d in objects_data]

        if rle_index.shape != (len(objects), frame_count, 2):
            raise ValueError(
                f"rle_index shape {rle_index.shape} does not match expected ({len(objects)}, {frame_count}, 2)"
            )

        archive = cls(
            shot_id=shot_id,
            frame_count=frame_count,
            height=height,
            width=width,
            objects=objects,
        )
        archive._rle_lengths = rle_lengths
        archive._rle_index = rle_index
        archive._archive_path = target_path.resolve()
        return archive

    def compression_ratio(self, raw_dtype: np.dtype | type = np.uint8) -> float:
        """Calculate ratio of uncompressed raw mask bytes to compressed archive file size on disk."""
        num_objs = len(self.objects)
        if num_objs == 0 or self.frame_count == 0 or self.height == 0 or self.width == 0:
            return 0.0

        itemsize = np.dtype(raw_dtype).itemsize
        raw_bytes = int(num_objs) * int(self.frame_count) * int(self.height) * int(self.width) * int(itemsize)

        archive_bytes = 0
        if self._archive_path is not None and self._archive_path.is_file():
            archive_bytes = self._archive_path.stat().st_size
        else:
            buf = io.BytesIO()
            metadata_json, consolidated_rle, new_rle_index = self._prepare_archive_arrays()
            np.savez_compressed(
                buf,
                metadata=metadata_json,
                rle_lengths=consolidated_rle,
                rle_index=new_rle_index,
            )
            archive_bytes = len(buf.getvalue())

        if archive_bytes == 0:
            return 0.0
        return float(raw_bytes / archive_bytes)

    def __len__(self) -> int:
        return self.frame_count

    def __repr__(self) -> str:
        return (
            f"ShotMaskArchive(shot_id={self.shot_id!r}, "
            f"frame_count={self.frame_count}, "
            f"height={self.height}, "
            f"width={self.width}, "
            f"objects={len(self.objects)})"
        )


def save_shot_masks_npz(archive: ShotMaskArchive, path: Path | str) -> None:
    """Save ShotMaskArchive to .masks.npz file."""
    archive.save_npz(path)


def load_shot_masks_npz(path: Path | str) -> ShotMaskArchive:
    """Load ShotMaskArchive from .masks.npz file."""
    return ShotMaskArchive.load_npz(path)
