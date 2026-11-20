"""Non-destructive sparse bitmask delta ZIP store, precedence algebra, and provenance."""
from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import threading
import time
import uuid
import zipfile
import zlib
from typing import Any

import cv2
import numpy as np

from .config import ROOT, SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

try:
    from film_defect_detectors import RepairInclusionPolicy
except ImportError:
    from scripts.film_defect_detectors import RepairInclusionPolicy


EDIT_SCHEMA_VERSION = 1

BIT_SCRATCH_ADD = 1
BIT_SCRATCH_REJECT = 2
BIT_DUST_ADD = 4
BIT_DUST_REJECT = 8
BIT_PROTECT_INTACT = 16
BIT_UNCERTAIN_REJECT = 32
BIT_REVIEWED = 64

KNOWN_EDIT_BITS_MASK = (
    BIT_SCRATCH_ADD
    | BIT_SCRATCH_REJECT
    | BIT_DUST_ADD
    | BIT_DUST_REJECT
    | BIT_PROTECT_INTACT
    | BIT_UNCERTAIN_REJECT
    | BIT_REVIEWED
)
EDIT_ACTION_MASK = (
    BIT_SCRATCH_ADD
    | BIT_SCRATCH_REJECT
    | BIT_DUST_ADD
    | BIT_DUST_REJECT
    | BIT_PROTECT_INTACT
    | BIT_UNCERTAIN_REJECT
)

VALID_STROKE_TOOLS = frozenset(
    {
        "scratch_add",
        "scratch_reject",
        "dust_add",
        "dust_reject",
        "uncertain_reject",
        "reject_all",
        "protect",
        "unprotect",
        "clear_edits",
        "mark_reviewed",
    }
)

DUST_MASKS_REL_DIR = Path("manifests") / "dust_masks"
MEMBER_PATTERN = re.compile(r"^f(\d{6})\.png$")
EDIT_META_MEMBER = "edit_meta.json"
MAX_PNG_MEMBER_BYTES = 64 * 1024 * 1024
GROUND_TRUTH_STATUS = "operational_edits_only_not_independent_gt"

_STORE_LOCKS_GUARD = threading.Lock()
_STORE_LOCKS: dict[str, threading.RLock] = {}


class DustMaskEditError(ValueError):
    """Base error for dust mask edit store validation and verification failures."""


class EditStoreIntegrityError(DustMaskEditError):
    """Raised when archive files, checksums, revisions, or PNG payloads fail verification."""


class EditTimelineMismatchError(DustMaskEditError):
    """Raised when stored edit archive dimensions, frame count, or source identity mismatch."""


class EditRevisionConflictError(DustMaskEditError):
    """Raised when an edit write targets a stale edit_revision."""


def _store_lock_for(zip_path: Path) -> threading.RLock:
    key = str(zip_path.resolve(strict=False))
    with _STORE_LOCKS_GUARD:
        lock = _STORE_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _STORE_LOCKS[key] = lock
        return lock


def member_name(frame: int) -> str:
    idx = int(frame)
    if idx < 0:
        raise ValueError(f"Frame index must be non-negative, got {frame}")
    return f"f{idx:06d}.png"


def frame_of(name: str) -> int | None:
    match = MEMBER_PATTERN.match(name)
    return int(match.group(1)) if match else None


def normalize_fps(fps: str | int | float | Fraction) -> str:
    """Normalize frame rate representations (int, float, Fraction, or string) deterministically."""
    if isinstance(fps, Fraction):
        frac = fps
    elif isinstance(fps, bool):
        raise ValueError("fps cannot be a boolean")
    elif isinstance(fps, int):
        frac = Fraction(fps, 1)
    elif isinstance(fps, float):
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"fps must be positive and finite, got {fps!r}")
        frac = Fraction(fps).limit_denominator(1001)
    elif isinstance(fps, str):
        text = fps.strip()
        if not text:
            raise ValueError("fps cannot be empty")
        try:
            if "/" in text:
                frac = Fraction(text)
            else:
                val = float(text)
                if not np.isfinite(val) or val <= 0:
                    raise ValueError(f"fps must be positive and finite, got {fps!r}")
                frac = Fraction(text)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Invalid fps value: {fps!r}") from exc
    else:
        raise ValueError(f"Unsupported fps type: {type(fps).__name__}")

    if frac <= 0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    return str(frac)


def normalize_source_ref(source_path: str | Path, root: Path | str = ROOT) -> str:
    """Normalize source path relative to workspace root when inside it, using POSIX separators."""
    text = str(source_path or "").strip()
    if not text:
        return ""
    candidate = Path(text)
    if not candidate.is_absolute():
        return Path(text.replace("\\", "/")).as_posix()
    root_resolved = Path(root).resolve(strict=False)
    cand_resolved = candidate.resolve(strict=False)
    try:
        return cand_resolved.relative_to(root_resolved).as_posix()
    except ValueError:
        return cand_resolved.as_posix()


def compute_file_sha256(path: Path | str) -> str:
    """Compute SHA-256 hex digest of a file using chunked reads."""
    target = Path(path)
    with target.open("rb") as handle:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(handle, "sha256").hexdigest()
        hasher = hashlib.sha256()
        while chunk := handle.read(65536):
            hasher.update(chunk)
        return hasher.hexdigest()


@dataclass(frozen=True)
class TimelineBinding:
    """Immutable identity contract binding a bitmask archive to one exact video timeline."""

    source_path: str
    analysis_source: str
    source_sha256: str
    source_size: int
    source_mtime_ns: int
    width: int
    height: int
    frame_count: int
    fps: str | int | float | Fraction
    edit_schema_version: int = EDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        source_mode = str(self.analysis_source or "").strip().lower()
        if source_mode not in ("stage", "original"):
            raise ValueError(f"analysis_source must be 'stage' or 'original', got {self.analysis_source!r}")
        width = int(self.width)
        height = int(self.height)
        frame_count = int(self.frame_count)
        if width <= 0 or height <= 0 or frame_count < 1:
            raise ValueError(
                f"Invalid video dimensions/frame_count: width={width}, height={height}, frame_count={frame_count}"
            )
        norm_fps = normalize_fps(self.fps)
        object.__setattr__(self, "source_path", str(self.source_path or "").strip())
        object.__setattr__(self, "analysis_source", source_mode)
        object.__setattr__(self, "source_sha256", str(self.source_sha256 or "").strip().lower())
        object.__setattr__(self, "source_size", int(self.source_size))
        object.__setattr__(self, "source_mtime_ns", int(self.source_mtime_ns))
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "fps", norm_fps)
        object.__setattr__(self, "edit_schema_version", int(self.edit_schema_version))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "analysis_source": self.analysis_source,
            "source_sha256": self.source_sha256,
            "source_size": self.source_size,
            "source_mtime_ns": self.source_mtime_ns,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "fps": str(self.fps),
            "edit_schema_version": self.edit_schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TimelineBinding":
        if not isinstance(data, Mapping):
            raise EditStoreIntegrityError("timeline_binding must be a JSON object.")
        required = (
            "source_path",
            "analysis_source",
            "source_sha256",
            "source_size",
            "source_mtime_ns",
            "width",
            "height",
            "frame_count",
            "fps",
        )
        missing = [key for key in required if key not in data]
        if missing:
            raise EditStoreIntegrityError(f"timeline_binding missing required fields: {missing}")
        try:
            return cls(
                source_path=str(data["source_path"]),
                analysis_source=str(data["analysis_source"]),
                source_sha256=str(data["source_sha256"]),
                source_size=int(data["source_size"]),
                source_mtime_ns=int(data["source_mtime_ns"]),
                width=int(data["width"]),
                height=int(data["height"]),
                frame_count=int(data["frame_count"]),
                fps=str(data["fps"]),
                edit_schema_version=int(data.get("edit_schema_version", EDIT_SCHEMA_VERSION)),
            )
        except (TypeError, ValueError) as exc:
            raise EditStoreIntegrityError(f"Invalid timeline_binding fields: {exc}") from exc


def build_timeline_binding(
    source_path: Path | str,
    analysis_source: str,
    *,
    width: int,
    height: int,
    frame_count: int,
    fps: str | int | float | Fraction,
    root: Path | str = ROOT,
    source_sha256: str | None = None,
) -> TimelineBinding:
    """Build a TimelineBinding from a source file on disk and probed geometry."""
    raw_path = Path(source_path)
    resolved = raw_path if raw_path.is_absolute() else Path(root) / raw_path
    if not resolved.is_file():
        raise FileNotFoundError(f"Source file for TimelineBinding does not exist: {resolved}")
    stat = resolved.stat()
    sha = source_sha256 if source_sha256 is not None else compute_file_sha256(resolved)
    return TimelineBinding(
        source_path=normalize_source_ref(resolved, root=root),
        analysis_source=analysis_source,
        source_sha256=sha,
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        edit_schema_version=EDIT_SCHEMA_VERSION,
    )


def compute_timeline_id(binding: TimelineBinding) -> str:
    """Compute deterministic 24-hex timeline ID independent of detector settings and file mtime."""
    norm_source = normalize_source_ref(binding.source_path, root=ROOT)
    payload: dict[str, Any] = {
        "edit_schema_version": binding.edit_schema_version,
        "analysis_source": binding.analysis_source,
        "source_path": norm_source,
        "source_sha256": binding.source_sha256,
        "width": binding.width,
        "height": binding.height,
        "frame_count": binding.frame_count,
        "fps": normalize_fps(binding.fps),
    }
    if not binding.source_sha256:
        payload["source_size"] = binding.source_size
        payload["source_mtime_ns"] = binding.source_mtime_ns
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def check_compatibility(
    expected: TimelineBinding,
    stored: TimelineBinding,
    *,
    root: Path | str = ROOT,
) -> None:
    """Verify stored timeline binding matches expected source/geometry; raise EditTimelineMismatchError otherwise."""
    if stored.edit_schema_version != expected.edit_schema_version:
        raise EditTimelineMismatchError(
            f"Edit schema version mismatch: stored {stored.edit_schema_version} vs expected {expected.edit_schema_version}."
        )
    if stored.analysis_source != expected.analysis_source:
        raise EditTimelineMismatchError(
            f"Analysis source mismatch: stored {stored.analysis_source!r} vs expected {expected.analysis_source!r}."
        )
    if (stored.width, stored.height) != (expected.width, expected.height):
        raise EditTimelineMismatchError(
            f"Frame geometry mismatch: stored {stored.width}x{stored.height} vs expected {expected.width}x{expected.height}."
        )
    if stored.frame_count != expected.frame_count:
        raise EditTimelineMismatchError(
            f"Frame count mismatch: stored {stored.frame_count} vs expected {expected.frame_count}."
        )
    if normalize_fps(stored.fps) != normalize_fps(expected.fps):
        raise EditTimelineMismatchError(
            f"Frame rate mismatch: stored {stored.fps} vs expected {expected.fps}."
        )
    if stored.source_sha256 != expected.source_sha256:
        raise EditTimelineMismatchError(
            f"Source SHA-256 mismatch: stored {stored.source_sha256[:12]}... vs expected {expected.source_sha256[:12]}...."
        )
    if expected.source_size > 0 and stored.source_size > 0 and stored.source_size != expected.source_size:
        raise EditTimelineMismatchError(
            f"Source file size mismatch: stored {stored.source_size} vs expected {expected.source_size}."
        )
    if not expected.source_sha256 and stored.source_mtime_ns != expected.source_mtime_ns:
        raise EditTimelineMismatchError("Source timestamp mismatch without SHA-256 verification.")
    norm_expected = normalize_source_ref(expected.source_path, root=root)
    norm_stored = normalize_source_ref(stored.source_path, root=root)
    if norm_expected != norm_stored:
        raise EditTimelineMismatchError(
            f"Source path mismatch: stored {norm_stored!r} vs expected {norm_expected!r}."
        )


def compute_edit_signature(zip_sha256: str, edit_revision: int, edited_frames: Sequence[int]) -> str:
    """Compute cache-key edit signature '<zip_sha256>:rev<edit_revision>', or '' when no edits exist."""
    if not edited_frames or not zip_sha256 or int(edit_revision) <= 0:
        return ""
    return f"{zip_sha256}:rev{int(edit_revision)}"


def _validate_uint8_2d(arr: Any, name: str = "edit_u8", expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{name} must be a numpy.ndarray, got {type(arr).__name__}")
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{name} must have dtype uint8, got {arr.dtype}")
    if expected_shape is not None and arr.shape != expected_shape:
        raise ValueError(f"{name} shape {arr.shape} does not match expected {expected_shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _to_bool_mask(arr: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    if arr is None:
        return np.zeros(expected_shape, dtype=bool)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Mask must be a numpy.ndarray, got {type(arr).__name__}")
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        plane = arr[:, :, 0]
    elif arr.ndim == 2:
        plane = arr
    else:
        raise ValueError(f"Mask must be 2D or 3/4-channel image, got shape {arr.shape}")
    if plane.shape != expected_shape:
        raise ValueError(f"Mask shape {plane.shape} does not match expected {expected_shape}")
    if plane.dtype == np.bool_:
        return plane.copy()
    if np.issubdtype(plane.dtype, np.floating):
        if not np.all(np.isfinite(plane)):
            raise ValueError("Floating mask contains NaN/Inf")
        threshold = 0.5 if float(plane.max(initial=0.0)) <= 1.0 else 127.0
        return plane > threshold
    if np.issubdtype(plane.dtype, np.integer):
        return plane > 127
    raise ValueError(f"Unsupported mask dtype: {plane.dtype}")


def apply_stroke_transition(
    current_u8: np.ndarray,
    mask_pixels_bool: np.ndarray,
    tool: str,
) -> np.ndarray:
    """Apply deterministic bitmask stroke transition with conflict reversal without mutating inputs."""
    out = _validate_uint8_2d(current_u8, "current_u8").copy()
    if not isinstance(mask_pixels_bool, np.ndarray):
        sel = np.asarray(mask_pixels_bool, dtype=bool)
    elif mask_pixels_bool.dtype == np.bool_:
        sel = mask_pixels_bool
    else:
        sel = mask_pixels_bool != 0
    if sel.shape != out.shape:
        raise ValueError(f"mask_pixels_bool shape {sel.shape} does not match current_u8 shape {out.shape}")

    tool_norm = str(tool).strip().lower()
    if tool_norm not in VALID_STROKE_TOOLS:
        raise ValueError(f"Unsupported stroke tool {tool!r}; expected one of {sorted(VALID_STROKE_TOOLS)}")

    if not np.any(sel):
        return out

    if tool_norm == "scratch_add":
        out[sel] = (out[sel] & ~np.uint8(BIT_SCRATCH_REJECT)) | np.uint8(BIT_SCRATCH_ADD)
    elif tool_norm == "scratch_reject":
        out[sel] = (out[sel] & ~np.uint8(BIT_SCRATCH_ADD)) | np.uint8(BIT_SCRATCH_REJECT)
    elif tool_norm == "dust_add":
        out[sel] = (out[sel] & ~np.uint8(BIT_DUST_REJECT)) | np.uint8(BIT_DUST_ADD)
    elif tool_norm == "dust_reject":
        out[sel] = (out[sel] & ~np.uint8(BIT_DUST_ADD)) | np.uint8(BIT_DUST_REJECT)
    elif tool_norm == "uncertain_reject":
        out[sel] = out[sel] | np.uint8(BIT_UNCERTAIN_REJECT)
    elif tool_norm == "reject_all":
        out[sel] = (out[sel] & ~np.uint8(BIT_SCRATCH_ADD | BIT_DUST_ADD)) | np.uint8(
            BIT_SCRATCH_REJECT | BIT_DUST_REJECT | BIT_UNCERTAIN_REJECT
        )
    elif tool_norm == "protect":
        out[sel] = out[sel] | np.uint8(BIT_PROTECT_INTACT)
    elif tool_norm == "unprotect":
        out[sel] = out[sel] & ~np.uint8(BIT_PROTECT_INTACT)
    elif tool_norm == "clear_edits":
        out[sel] = np.uint8(0)
    elif tool_norm == "mark_reviewed":
        out[sel] = out[sel] | np.uint8(BIT_REVIEWED)
    return out


def _normalize_policy(policy: RepairInclusionPolicy | str) -> RepairInclusionPolicy:
    if isinstance(policy, RepairInclusionPolicy):
        return policy
    try:
        return RepairInclusionPolicy(str(policy).strip().lower())
    except ValueError as exc:
        raise ValueError(f"Unknown RepairInclusionPolicy: {policy!r}") from exc


def _infer_shape_2d(*arrays: Any) -> tuple[int, int]:
    for arr in arrays:
        if arr is not None:
            if not isinstance(arr, np.ndarray) or arr.ndim not in (2, 3):
                raise ValueError("Mask inputs must be 2D or 3D numpy arrays.")
            return int(arr.shape[0]), int(arr.shape[1])
    raise ValueError("At least one mask or edit array must be provided to infer frame shape.")


def resolve_effective_channels(
    auto_scratch: np.ndarray | None,
    auto_dust: np.ndarray | None,
    auto_uncertain: np.ndarray | None,
    edit_u8: np.ndarray | None,
    policy: RepairInclusionPolicy | str = RepairInclusionPolicy.CONSERVATIVE,
    *,
    auto_combined: np.ndarray | None = None,
    dilate: int = 0,
) -> dict[str, np.ndarray]:
    """Resolve scratch, dust, uncertain, and effective repair masks with strict precedence.

    Precedence per pixel: PROTECT_INTACT > explicit REJECT (fail-closed if ADD & REJECT coexist)
    > human ADD > AUTO. Re-enforces PROTECT_INTACT and explicit rejections after dilation.
    """
    policy_enum = _normalize_policy(policy)
    dilate_iters = int(dilate)
    if dilate_iters < 0:
        raise ValueError(f"dilate must be >= 0, got {dilate}")

    shape = _infer_shape_2d(edit_u8, auto_scratch, auto_dust, auto_uncertain, auto_combined)
    edit = _validate_uint8_2d(edit_u8, "edit_u8", expected_shape=shape) if edit_u8 is not None else np.zeros(shape, dtype=np.uint8)

    auto_s = _to_bool_mask(auto_scratch, shape)
    auto_d = _to_bool_mask(auto_dust, shape)
    auto_u = _to_bool_mask(auto_uncertain, shape)
    auto_c = _to_bool_mask(auto_combined, shape)

    protected = (edit & BIT_PROTECT_INTACT) != 0
    scratch_add = (edit & BIT_SCRATCH_ADD) != 0
    scratch_reject = (edit & BIT_SCRATCH_REJECT) != 0
    dust_add = (edit & BIT_DUST_ADD) != 0
    dust_reject = (edit & BIT_DUST_REJECT) != 0
    uncertain_reject = (edit & BIT_UNCERTAIN_REJECT) != 0

    resolved_scratch = (~protected) & (~scratch_reject) & (scratch_add | auto_s)
    resolved_dust = (~protected) & (~dust_reject) & (dust_add | auto_d)

    promoted = (scratch_add & (~scratch_reject)) | (dust_add & (~dust_reject))
    resolved_uncertain = (~protected) & (~uncertain_reject) & (~promoted) & auto_u

    vetoed_combined = (
        protected
        | (auto_s & scratch_reject)
        | (auto_d & dust_reject)
        | (auto_u & uncertain_reject)
        | ((~auto_s) & (~auto_d) & (~auto_u) & (scratch_reject | dust_reject | uncertain_reject))
    )
    non_rejected_combined = auto_c & (~vetoed_combined)

    def _maybe_dilate(mask_bool: np.ndarray, veto_bool: np.ndarray) -> np.ndarray:
        if dilate_iters == 0:
            return mask_bool & (~veto_bool)
        if not np.any(mask_bool):
            return np.zeros(shape, dtype=bool)
        dilated = cv2.dilate(
            mask_bool.astype(np.uint8) * 255,
            np.ones((3, 3), np.uint8),
            iterations=dilate_iters,
        )
        return (dilated > 0) & (~veto_bool)

    if policy_enum == RepairInclusionPolicy.SCRATCH_ONLY:
        effective_bool = _maybe_dilate(resolved_scratch, protected | scratch_reject)
    elif policy_enum == RepairInclusionPolicy.CONSERVATIVE:
        effective_bool = _maybe_dilate(resolved_scratch, protected | scratch_reject) | _maybe_dilate(
            resolved_dust, protected | dust_reject
        )
    elif policy_enum == RepairInclusionPolicy.HIGH_RECALL:
        effective_bool = (
            _maybe_dilate(resolved_scratch, protected | scratch_reject)
            | _maybe_dilate(resolved_dust, protected | dust_reject)
            | _maybe_dilate(resolved_uncertain, protected | uncertain_reject)
        )
    elif policy_enum == RepairInclusionPolicy.COMBINED_DEFECT:
        effective_bool = (
            _maybe_dilate(resolved_scratch, protected | scratch_reject)
            | _maybe_dilate(resolved_dust, protected | dust_reject)
            | _maybe_dilate(non_rejected_combined, vetoed_combined)
        )
    else:
        raise ValueError(f"Unsupported RepairInclusionPolicy: {policy_enum}")

    # Final hard invariant: protected pixels can never be selected for repair.
    effective_bool &= ~protected

    return {
        "scratch": np.where(resolved_scratch, np.uint8(255), np.uint8(0)),
        "dust": np.where(resolved_dust, np.uint8(255), np.uint8(0)),
        "uncertain": np.where(resolved_uncertain, np.uint8(255), np.uint8(0)),
        "effective": np.where(effective_bool, np.uint8(255), np.uint8(0)),
        "protected": np.where(protected, np.uint8(255), np.uint8(0)),
    }


def encode_bitmask_b64(mask_u8: np.ndarray) -> str:
    """Encode a 2D uint8 bitmask array into raw base64 bytes (exact Uint8Array transport)."""
    arr = _validate_uint8_2d(mask_u8, "mask_u8")
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_bitmask_b64(
    b64_text: str,
    width: int,
    height: int,
    *,
    validate_known_bits: bool = False,
) -> np.ndarray:
    """Decode raw base64 Uint8Array bytes into a contiguous 2D uint8 bitmask array."""
    w = int(width)
    h = int(height)
    if w <= 0 or h <= 0:
        raise ValueError(f"width and height must be positive, got {w}x{h}")
    if not isinstance(b64_text, str):
        raise ValueError("edit_mask_b64 must be a base64 string.")
    expected_len = w * h
    max_b64_len = ((expected_len + 2) // 3) * 4 + 16
    if len(b64_text) > max_b64_len:
        raise ValueError("Bitmask base64 payload exceeds expected frame dimensions.")
    try:
        raw = base64.b64decode(b64_text, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid base64 bitmask payload.") from exc
    if len(raw) != expected_len:
        raise ValueError(
            f"Decoded bitmask byte length {len(raw)} does not match {w}x{h} ({expected_len} bytes)."
        )
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w)).copy()
    if validate_known_bits and np.any(arr & np.uint8(~KNOWN_EDIT_BITS_MASK & 0xFF)):
        raise ValueError("Bitmask payload contains unsupported high bit flags.")
    return arr


def encode_bitmask_png(mask_u8: np.ndarray) -> bytes:
    """Encode a 2D uint8 bitmask array as a lossless 8-bit single-channel grayscale PNG."""
    arr = _validate_uint8_2d(mask_u8, "mask_u8")
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("Failed to encode bitmask PNG.")
    return buf.tobytes()


def decode_bitmask_png(
    png_bytes: bytes,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Decode a lossless 8-bit grayscale PNG into an exact 2D uint8 bitmask array."""
    if not png_bytes:
        raise EditStoreIntegrityError("Empty PNG payload.")
    arr = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None or arr.ndim != 2 or arr.dtype != np.uint8:
        raise EditStoreIntegrityError("Stored mask PNG must be a single-channel 8-bit grayscale image.")
    if width is not None and height is not None and arr.shape != (int(height), int(width)):
        raise EditStoreIntegrityError(
            f"Stored mask PNG geometry {arr.shape[1]}x{arr.shape[0]} does not match expected {width}x{height}."
        )
    return np.ascontiguousarray(arr, dtype=np.uint8)


def png_bytes_to_bitmask_b64(png_bytes: bytes, width: int, height: int) -> str:
    return encode_bitmask_b64(decode_bitmask_png(png_bytes, width=width, height=height))


def bitmask_b64_to_png_bytes(
    b64_text: str,
    width: int,
    height: int,
    *,
    validate_known_bits: bool = False,
) -> bytes:
    return encode_bitmask_png(
        decode_bitmask_b64(b64_text, width=width, height=height, validate_known_bits=validate_known_bits)
    )


def _validate_png_ihdr(png_bytes: bytes, width: int, height: int, member: str) -> None:
    if len(png_bytes) < 33 or png_bytes[:8] != b"\x89PNG\r\n\x1a\n":
        raise EditStoreIntegrityError(f"Corrupt PNG header in member {member}.")
    length = struct.unpack(">I", png_bytes[8:12])[0]
    chunk_type = png_bytes[12:16]
    if chunk_type != b"IHDR" or length != 13:
        raise EditStoreIntegrityError(f"Missing IHDR chunk in member {member}.")
    w, h, bit_depth, color_type = struct.unpack(">IIBB", png_bytes[16:26])
    if (w, h) != (width, height):
        raise EditStoreIntegrityError(
            f"PNG member {member} dimensions {w}x{h} do not match timeline {width}x{height}."
        )
    if bit_depth != 8 or color_type != 0:
        raise EditStoreIntegrityError(
            f"PNG member {member} must be 8-bit grayscale (got bit_depth={bit_depth}, color_type={color_type})."
        )


def _replace_with_retry(src: Path, dst: Path) -> None:
    for attempt in range(5):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _cleanup_leftovers(masks_dir: Path, timeline_id: str) -> None:
    if not masks_dir.is_dir():
        return
    for pattern in (
        f"{timeline_id}.*.tmp",
        f"{timeline_id}.*.partial",
        f"{timeline_id}.*.partial.*",
        f".{timeline_id}.*.tmp*",
    ):
        for leftover in masks_dir.glob(pattern):
            try:
                if leftover.is_file():
                    leftover.unlink(missing_ok=True)
            except OSError:
                pass


def _parse_and_verify_archive(
    zip_path: Path,
    meta_path: Path,
) -> dict[str, Any]:
    """Strictly verify .meta.json and .zip consistency, checksums, and members without trusting partial data."""
    zip_exists = zip_path.is_file()
    meta_exists = meta_path.is_file()
    if not zip_exists and not meta_exists:
        return {
            "exists": False,
            "edit_revision": 0,
            "zip_sha256": "",
            "edited_frames": [],
            "frame_sha256s": {},
            "external_meta": None,
            "embedded_meta": None,
            "stored_binding": None,
        }
    if zip_exists and not meta_exists:
        raise EditStoreIntegrityError(f"Missing sidecar metadata file for edit archive: {meta_path.name}")
    if meta_exists and not zip_exists:
        raise EditStoreIntegrityError(f"Missing edit ZIP archive for sidecar metadata: {zip_path.name}")

    try:
        external_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EditStoreIntegrityError(f"Cannot parse dust mask sidecar metadata {meta_path.name}: {exc}") from exc
    if not isinstance(external_meta, dict):
        raise EditStoreIntegrityError("External .meta.json must contain a JSON object.")

    required_ext = (
        "edit_schema_version",
        "zip_sha256",
        "edit_revision",
        "timeline_binding",
        "edited_frames",
        "frame_sha256s",
    )
    missing_ext = [key for key in required_ext if key not in external_meta]
    if missing_ext:
        raise EditStoreIntegrityError(f"External .meta.json missing required fields: {missing_ext}")

    try:
        ext_schema = int(external_meta["edit_schema_version"])
        ext_rev = int(external_meta["edit_revision"])
    except (TypeError, ValueError) as exc:
        raise EditStoreIntegrityError("External .meta.json has non-integer schema_version or edit_revision.") from exc
    if ext_rev < 1:
        raise EditStoreIntegrityError(f"Invalid external edit_revision {ext_rev}; persisted archives must be >= 1.")

    ext_zip_sha = str(external_meta["zip_sha256"] or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", ext_zip_sha):
        raise EditStoreIntegrityError("External .meta.json contains invalid zip_sha256.")

    stored_binding = TimelineBinding.from_dict(external_meta["timeline_binding"])

    ext_frames_raw = external_meta["edited_frames"]
    ext_frame_shas = external_meta["frame_sha256s"]
    if not isinstance(ext_frames_raw, list) or not isinstance(ext_frame_shas, dict):
        raise EditStoreIntegrityError("edited_frames must be a list and frame_sha256s must be an object.")
    try:
        ext_frames = [int(f) for f in ext_frames_raw]
    except (TypeError, ValueError) as exc:
        raise EditStoreIntegrityError("edited_frames contains non-integer frame index.") from exc
    if ext_frames != sorted(set(ext_frames)):
        raise EditStoreIntegrityError("edited_frames must be strictly sorted and unique.")
    if any(f < 0 or f >= stored_binding.frame_count for f in ext_frames):
        raise EditStoreIntegrityError("edited_frames contains out-of-range frame index.")

    actual_zip_sha = compute_file_sha256(zip_path)
    if actual_zip_sha != ext_zip_sha:
        raise EditStoreIntegrityError(
            f"Edit ZIP SHA-256 mismatch: sidecar expects {ext_zip_sha[:12]}... but ZIP is {actual_zip_sha[:12]}...."
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise EditStoreIntegrityError(f"Corrupt ZIP CRC detected on member: {bad_member}")

            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise EditStoreIntegrityError("Edit ZIP contains duplicate member names.")
            if EDIT_META_MEMBER not in names:
                raise EditStoreIntegrityError(f"Edit ZIP is missing {EDIT_META_MEMBER}.")

            try:
                embedded_meta = json.loads(archive.read(EDIT_META_MEMBER).decode("utf-8"))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise EditStoreIntegrityError(f"Corrupt embedded {EDIT_META_MEMBER}: {exc}") from exc
            if not isinstance(embedded_meta, dict):
                raise EditStoreIntegrityError(f"Embedded {EDIT_META_MEMBER} must be a JSON object.")

            if int(embedded_meta.get("edit_revision", -1)) != ext_rev:
                raise EditStoreIntegrityError(
                    f"Edit revision mismatch between external meta ({ext_rev}) and embedded meta ({embedded_meta.get('edit_revision')})."
                )
            if int(embedded_meta.get("edit_schema_version", -1)) != ext_schema:
                raise EditStoreIntegrityError("Edit schema version mismatch between external and embedded metadata.")
            if embedded_meta.get("timeline_binding") != external_meta["timeline_binding"]:
                raise EditStoreIntegrityError("timeline_binding mismatch between external and embedded metadata.")
            if embedded_meta.get("edited_frames") != ext_frames_raw:
                raise EditStoreIntegrityError("edited_frames mismatch between external and embedded metadata.")
            if embedded_meta.get("frame_sha256s") != ext_frame_shas:
                raise EditStoreIntegrityError("frame_sha256s mismatch between external and embedded metadata.")

            zip_frame_set: set[int] = set()
            for info in infos:
                if info.filename == EDIT_META_MEMBER:
                    continue
                if "\\" in info.filename or info.filename.startswith("/") or ".." in Path(info.filename).parts:
                    raise EditStoreIntegrityError(f"Disallowed member path in edit ZIP: {info.filename!r}")
                frame_idx = frame_of(info.filename)
                if frame_idx is None:
                    raise EditStoreIntegrityError(f"Unexpected member in edit ZIP: {info.filename!r}")
                if info.file_size > MAX_PNG_MEMBER_BYTES:
                    raise EditStoreIntegrityError(f"Member {info.filename} exceeds max allowed PNG size.")
                png_bytes = archive.read(info)
                _validate_png_ihdr(png_bytes, stored_binding.width, stored_binding.height, info.filename)
                expected_frame_sha = str(ext_frame_shas.get(str(frame_idx), "") or "").strip().lower()
                actual_frame_sha = hashlib.sha256(png_bytes).hexdigest()
                if expected_frame_sha != actual_frame_sha:
                    raise EditStoreIntegrityError(
                        f"Frame {frame_idx} SHA-256 mismatch in edit ZIP."
                    )
                zip_frame_set.add(frame_idx)

            if zip_frame_set != set(ext_frames) or set(ext_frame_shas.keys()) != {str(f) for f in ext_frames}:
                raise EditStoreIntegrityError("Edit ZIP members do not match edited_frames / frame_sha256s manifest.")

    except (zipfile.BadZipFile, OSError, EOFError, zlib.error) as exc:
        raise EditStoreIntegrityError(f"Invalid or unreadable dust mask edit ZIP {zip_path.name}: {exc}") from exc

    return {
        "exists": True,
        "edit_revision": ext_rev,
        "zip_sha256": actual_zip_sha,
        "edited_frames": ext_frames,
        "frame_sha256s": {str(k): str(v) for k, v in ext_frame_shas.items()},
        "external_meta": external_meta,
        "embedded_meta": embedded_meta,
        "stored_binding": stored_binding,
    }


def verify_archive_pair(zip_path: Path | str, meta_path: Path | str) -> dict[str, Any]:
    """Verify an archive (.zip) and sidecar (.meta.json) pair directly from disk."""
    return _parse_and_verify_archive(Path(zip_path), Path(meta_path))


class DustMaskEditStore:
    """Atomic, checksum-verified sparse per-frame bitmask delta store."""

    def __init__(self, binding: TimelineBinding, root: Path | str = ROOT) -> None:
        self.binding = binding
        self.root = Path(root)
        self.timeline_id = compute_timeline_id(binding)
        self.masks_dir = self.root / DUST_MASKS_REL_DIR
        self.zip_path = self.masks_dir / f"{self.timeline_id}.zip"
        self.meta_path = self.masks_dir / f"{self.timeline_id}.meta.json"
        self._lock = _store_lock_for(self.zip_path)
        self.edit_revision: int = 0
        self.zip_sha256: str = ""
        self._edited_frames: list[int] = []
        self._frame_sha256s: dict[str, str] = {}

    @property
    def edit_signature(self) -> str:
        return compute_edit_signature(self.zip_sha256, self.edit_revision, self._edited_frames)

    def exists(self) -> bool:
        return self.zip_path.is_file() and self.meta_path.is_file()

    def edited_frames(self) -> list[int]:
        with self._lock:
            return list(self._edited_frames)

    def verify(self) -> dict[str, Any]:
        """Verify on-disk ZIP, sidecar JSON, checksums, revisions, and timeline compatibility."""
        with self._lock:
            state = _parse_and_verify_archive(self.zip_path, self.meta_path)
            if not state["exists"]:
                self.edit_revision = 0
                self.zip_sha256 = ""
                self._edited_frames = []
                self._frame_sha256s = {}
                return state
            stored_binding: TimelineBinding = state["stored_binding"]
            check_compatibility(self.binding, stored_binding, root=self.root)
            self.edit_revision = int(state["edit_revision"])
            self.zip_sha256 = str(state["zip_sha256"])
            self._edited_frames = list(state["edited_frames"])
            self._frame_sha256s = dict(state["frame_sha256s"])
            return state

    def _validate_frame_index(self, frame: int) -> int:
        if isinstance(frame, bool) or not isinstance(frame, (int, np.integer)):
            raise ValueError(f"frame index must be an integer, got {type(frame).__name__}")
        idx = int(frame)
        if idx < 0 or idx >= self.binding.frame_count:
            raise ValueError(f"Frame index {idx} out of range [0, {self.binding.frame_count})")
        return idx

    def read_frame(self, frame: int) -> np.ndarray:
        """Read one frame's uint8 bitmask (zeros when unedited); verifies archive integrity on read."""
        idx = self._validate_frame_index(frame)
        with self._lock:
            state = self.verify()
            if not state["exists"] or idx not in self._edited_frames:
                return np.zeros((self.binding.height, self.binding.width), dtype=np.uint8)
            with zipfile.ZipFile(self.zip_path, "r") as archive:
                png_bytes = archive.read(member_name(idx))
            expected_sha = self._frame_sha256s.get(str(idx), "")
            if hashlib.sha256(png_bytes).hexdigest() != expected_sha:
                raise EditStoreIntegrityError(f"Frame {idx} SHA-256 verification failed on read.")
            return decode_bitmask_png(png_bytes, width=self.binding.width, height=self.binding.height)

    def read_frame_b64(self, frame: int, *, omit_empty: bool = False) -> str:
        """Return base64 raw Uint8Array for frame (or '' if unedited and omit_empty=True)."""
        idx = self._validate_frame_index(frame)
        with self._lock:
            arr = self.read_frame(idx)
            if omit_empty and idx not in self._edited_frames:
                return ""
            return encode_bitmask_b64(arr)

    def read_all_frames(self) -> dict[int, np.ndarray]:
        """Verify and decode all edited frames in one pass."""
        with self._lock:
            state = self.verify()
            if not state["exists"] or not self._edited_frames:
                return {}
            result: dict[int, np.ndarray] = {}
            with zipfile.ZipFile(self.zip_path, "r") as archive:
                for idx in self._edited_frames:
                    png_bytes = archive.read(member_name(idx))
                    expected_sha = self._frame_sha256s.get(str(idx), "")
                    if hashlib.sha256(png_bytes).hexdigest() != expected_sha:
                        raise EditStoreIntegrityError(f"Frame {idx} SHA-256 verification failed on read.")
                    result[idx] = decode_bitmask_png(
                        png_bytes, width=self.binding.width, height=self.binding.height
                    )
            return result

    def save_frame(
        self,
        frame: int,
        edit_u8: np.ndarray,
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Persist one frame's bitmask atomically; zero-only masks omit/remove the frame."""
        idx = self._validate_frame_index(frame)
        arr = _validate_uint8_2d(
            edit_u8,
            "edit_u8",
            expected_shape=(self.binding.height, self.binding.width),
        )
        return self.save_frames({idx: arr}, expected_revision=expected_revision)

    def save_frames(
        self,
        updates: Mapping[int, np.ndarray | None],
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Atomically persist multiple frame updates (None or all-zero removes frame)."""
        validated: dict[int, np.ndarray | None] = {}
        for frame_key, mask in updates.items():
            idx = self._validate_frame_index(frame_key)
            if mask is None:
                validated[idx] = None
            else:
                arr = _validate_uint8_2d(
                    mask,
                    f"edit_u8[{idx}]",
                    expected_shape=(self.binding.height, self.binding.width),
                )
                validated[idx] = arr if np.any(arr != 0) else None

        with self._lock:
            state = self.verify()
            if expected_revision is not None and int(expected_revision) != self.edit_revision:
                raise EditRevisionConflictError(
                    f"Expected edit_revision {expected_revision}, but current revision is {self.edit_revision}."
                )

            existing_pngs: dict[int, bytes] = {}
            if state["exists"] and self._edited_frames:
                with zipfile.ZipFile(self.zip_path, "r") as archive:
                    for idx in self._edited_frames:
                        existing_pngs[idx] = archive.read(member_name(idx))

            kept_pngs = dict(existing_pngs)
            changed = False
            for idx, arr in validated.items():
                if arr is None:
                    if idx in kept_pngs:
                        kept_pngs.pop(idx)
                        changed = True
                else:
                    new_png = encode_bitmask_png(arr)
                    if kept_pngs.get(idx) != new_png:
                        kept_pngs[idx] = new_png
                        changed = True

            if not changed:
                return self.summary()

            if not kept_pngs:
                self._remove_store_files_locked()
                return self.summary()

            self._commit_atomic_locked(kept_pngs)
            return self.summary()

    def reset_frame(self, frame: int, *, expected_revision: int | None = None) -> dict[str, Any]:
        """Clear all edits on a single frame atomically."""
        idx = self._validate_frame_index(frame)
        return self.save_frames({idx: None}, expected_revision=expected_revision)

    def clear_all(self, *, expected_revision: int | None = None) -> dict[str, Any]:
        """Remove all edited frames and delete archive/sidecar files safely."""
        with self._lock:
            if expected_revision is not None:
                self.verify()
                if int(expected_revision) != self.edit_revision:
                    raise EditRevisionConflictError(
                        f"Expected edit_revision {expected_revision}, but current revision is {self.edit_revision}."
                    )
            self._remove_store_files_locked()
            return self.summary()

    def _remove_store_files_locked(self) -> None:
        self.zip_path.unlink(missing_ok=True)
        self.meta_path.unlink(missing_ok=True)
        _cleanup_leftovers(self.masks_dir, self.timeline_id)
        self.edit_revision = 0
        self.zip_sha256 = ""
        self._edited_frames = []
        self._frame_sha256s = {}

    def _commit_atomic_locked(self, kept_pngs: Mapping[int, bytes]) -> None:
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        _cleanup_leftovers(self.masks_dir, self.timeline_id)

        sorted_frames = sorted(kept_pngs)
        frame_sha256s = {
            str(frame): hashlib.sha256(kept_pngs[frame]).hexdigest() for frame in sorted_frames
        }
        new_revision = self.edit_revision + 1
        binding_dict = self.binding.to_dict()

        embedded_meta = {
            "edit_schema_version": EDIT_SCHEMA_VERSION,
            "timeline_id": self.timeline_id,
            "timeline_binding": binding_dict,
            "edit_revision": new_revision,
            "edited_frames": sorted_frames,
            "frame_sha256s": frame_sha256s,
        }

        token = f"{os.getpid()}.{uuid.uuid4().hex[:8]}"
        tmp_zip = self.masks_dir / f"{self.timeline_id}.zip.partial.{token}.tmp"
        tmp_meta = self.masks_dir / f"{self.timeline_id}.meta.json.partial.{token}.tmp"
        try:
            with tmp_zip.open("wb") as handle:
                with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_STORED) as archive:
                    for frame in sorted_frames:
                        archive.writestr(member_name(frame), kept_pngs[frame])
                    archive.writestr(
                        EDIT_META_MEMBER,
                        json.dumps(embedded_meta, indent=2, sort_keys=True) + "\n",
                    )
                handle.flush()
                os.fsync(handle.fileno())

            zip_sha256 = compute_file_sha256(tmp_zip)

            external_meta = {
                "edit_schema_version": EDIT_SCHEMA_VERSION,
                "timeline_id": self.timeline_id,
                "zip_sha256": zip_sha256,
                "edit_revision": new_revision,
                "timeline_binding": binding_dict,
                "edited_frames": sorted_frames,
                "frame_sha256s": frame_sha256s,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            with tmp_meta.open("w", encoding="utf-8") as meta_handle:
                meta_handle.write(json.dumps(external_meta, indent=2, sort_keys=True) + "\n")
                meta_handle.flush()
                os.fsync(meta_handle.fileno())

            _replace_with_retry(tmp_zip, self.zip_path)
            _replace_with_retry(tmp_meta, self.meta_path)
        finally:
            tmp_zip.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)

        self.edit_revision = new_revision
        self.zip_sha256 = zip_sha256
        self._edited_frames = list(sorted_frames)
        self._frame_sha256s = dict(frame_sha256s)

    def frame_stats(self, frame: int) -> dict[str, int]:
        arr = self.read_frame(frame)
        return {
            "scratch_add_pixels": int(np.count_nonzero((arr & BIT_SCRATCH_ADD) != 0)),
            "scratch_reject_pixels": int(np.count_nonzero((arr & BIT_SCRATCH_REJECT) != 0)),
            "dust_add_pixels": int(np.count_nonzero((arr & BIT_DUST_ADD) != 0)),
            "dust_reject_pixels": int(np.count_nonzero((arr & BIT_DUST_REJECT) != 0)),
            "uncertain_reject_pixels": int(np.count_nonzero((arr & BIT_UNCERTAIN_REJECT) != 0)),
            "protect_intact_pixels": int(np.count_nonzero((arr & BIT_PROTECT_INTACT) != 0)),
            "reviewed_pixels": int(np.count_nonzero((arr & BIT_REVIEWED) != 0)),
            "edited_pixels": int(np.count_nonzero(arr != 0)),
        }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "timeline_id": self.timeline_id,
                "edit_schema_version": EDIT_SCHEMA_VERSION,
                "edit_revision": self.edit_revision,
                "zip_sha256": self.zip_sha256,
                "edit_signature": self.edit_signature,
                "edited_frames": list(self._edited_frames),
                "edited_frame_count": len(self._edited_frames),
                "has_edits": bool(self._edited_frames),
            }


def load_verified_store(binding: TimelineBinding, root: Path | str = ROOT) -> DustMaskEditStore:
    """Instantiate and strictly verify a DustMaskEditStore; fails closed on corruption or mismatch."""
    store = DustMaskEditStore(binding, root=root)
    store.verify()
    return store


def store_compatibility_status(binding: TimelineBinding, root: Path | str = ROOT) -> dict[str, Any]:
    """Check store compatibility non-destructively for GUI state hydration."""
    store = DustMaskEditStore(binding, root=root)
    try:
        store.verify()
        return {
            "compatible": True,
            "compatibility_reason": "",
            "store": store,
            **store.summary(),
        }
    except (EditStoreIntegrityError, EditTimelineMismatchError) as exc:
        return {
            "compatible": False,
            "compatibility_reason": str(exc),
            "store": None,
            "timeline_id": store.timeline_id,
            "edit_schema_version": EDIT_SCHEMA_VERSION,
            "edit_revision": 0,
            "zip_sha256": "",
            "edit_signature": "",
            "edited_frames": [],
            "edited_frame_count": 0,
            "has_edits": False,
        }


def build_frame_provenance(
    frame: int,
    edit_u8: np.ndarray | None,
    *,
    auto_scratch: np.ndarray | None = None,
    auto_dust: np.ndarray | None = None,
    auto_uncertain: np.ndarray | None = None,
    auto_vetoed: np.ndarray | None = None,
    effective_mask: np.ndarray | None = None,
    shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Compute per-frame pixel provenance separating auto predictions, human edits, protect, and review state."""
    frame_shape = shape if shape is not None else _infer_shape_2d(
        edit_u8, auto_scratch, auto_dust, auto_uncertain, auto_vetoed, effective_mask
    )
    h, w = int(frame_shape[0]), int(frame_shape[1])
    total_pixels = h * w

    edit = (
        _validate_uint8_2d(edit_u8, "edit_u8", expected_shape=(h, w))
        if edit_u8 is not None
        else np.zeros((h, w), dtype=np.uint8)
    )
    auto_s = _to_bool_mask(auto_scratch, (h, w))
    auto_d = _to_bool_mask(auto_dust, (h, w))
    auto_u = _to_bool_mask(auto_uncertain, (h, w))
    auto_v = _to_bool_mask(auto_vetoed, (h, w))
    auto_any = auto_s | auto_d | auto_u

    protect_mask = (edit & BIT_PROTECT_INTACT) != 0
    scratch_add_mask = (edit & BIT_SCRATCH_ADD) != 0
    scratch_reject_mask = (edit & BIT_SCRATCH_REJECT) != 0
    dust_add_mask = (edit & BIT_DUST_ADD) != 0
    dust_reject_mask = (edit & BIT_DUST_REJECT) != 0
    uncertain_reject_mask = (edit & BIT_UNCERTAIN_REJECT) != 0
    reviewed_marker_mask = (edit & BIT_REVIEWED) != 0
    operational_reviewed_mask = edit != 0

    operational_reviewed = int(np.count_nonzero(operational_reviewed_mask))
    unreviewed = total_pixels - operational_reviewed
    effective_count = (
        int(np.count_nonzero(_to_bool_mask(effective_mask, (h, w))))
        if effective_mask is not None
        else None
    )

    return {
        "frame": int(frame),
        "total_pixels": int(total_pixels),
        "auto_predicted": int(np.count_nonzero(auto_any)),
        "auto_scratch": int(np.count_nonzero(auto_s)),
        "auto_dust": int(np.count_nonzero(auto_d)),
        "auto_uncertain": int(np.count_nonzero(auto_u)),
        "auto_vetoed": int(np.count_nonzero(auto_v)),
        "human_scratch_add": int(np.count_nonzero(scratch_add_mask)),
        "human_scratch_reject": int(np.count_nonzero(scratch_reject_mask)),
        "human_dust_add": int(np.count_nonzero(dust_add_mask)),
        "human_dust_reject": int(np.count_nonzero(dust_reject_mask)),
        "human_uncertain_reject": int(np.count_nonzero(uncertain_reject_mask)),
        "protect_intact": int(np.count_nonzero(protect_mask)),
        "operational_reviewed": operational_reviewed,
        "explicit_reviewed_marker": int(np.count_nonzero(reviewed_marker_mask)),
        "unreviewed": unreviewed,
        "effective_repair": effective_count,
    }


def build_edit_provenance_summary(
    store: DustMaskEditStore,
    *,
    auto_frames: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    effective_frames: Mapping[int, np.ndarray] | None = None,
    include_per_frame: bool = True,
) -> dict[str, Any]:
    """Aggregate timeline-wide provenance while preserving operational non-GT semantics."""
    store.verify()
    edit_map = store.read_all_frames()
    frame_shape = (store.binding.height, store.binding.width)
    frame_pixels = store.binding.height * store.binding.width
    total_timeline_pixels = store.binding.frame_count * frame_pixels

    relevant_frames = sorted(
        set(edit_map.keys())
        | set(int(k) for k in (auto_frames or {}).keys())
        | set(int(k) for k in (effective_frames or {}).keys())
    )

    totals: dict[str, int] = {
        "total_pixels": int(total_timeline_pixels),
        "auto_predicted": 0,
        "auto_scratch": 0,
        "auto_dust": 0,
        "auto_uncertain": 0,
        "auto_vetoed": 0,
        "human_scratch_add": 0,
        "human_scratch_reject": 0,
        "human_dust_add": 0,
        "human_dust_reject": 0,
        "human_uncertain_reject": 0,
        "protect_intact": 0,
        "operational_reviewed": 0,
        "explicit_reviewed_marker": 0,
        "unreviewed": 0,
        "effective_repair": 0,
    }

    per_frame_rows: list[dict[str, Any]] = []
    has_effective = effective_frames is not None
    for idx in relevant_frames:
        auto_entry = (auto_frames or {}).get(idx, {})
        eff_arr = (effective_frames or {}).get(idx)
        row = build_frame_provenance(
            idx,
            edit_map.get(idx),
            auto_scratch=auto_entry.get("scratch"),
            auto_dust=auto_entry.get("dust"),
            auto_uncertain=auto_entry.get("uncertain"),
            auto_vetoed=auto_entry.get("vetoed"),
            effective_mask=eff_arr,
            shape=frame_shape,
        )
        for key in (
            "auto_predicted",
            "auto_scratch",
            "auto_dust",
            "auto_uncertain",
            "auto_vetoed",
            "human_scratch_add",
            "human_scratch_reject",
            "human_dust_add",
            "human_dust_reject",
            "human_uncertain_reject",
            "protect_intact",
            "operational_reviewed",
            "explicit_reviewed_marker",
        ):
            totals[key] += int(row[key])
        if row["effective_repair"] is not None:
            totals["effective_repair"] += int(row["effective_repair"])
        if include_per_frame:
            per_frame_rows.append(row)

    totals["unreviewed"] = total_timeline_pixels - totals["operational_reviewed"]
    if not has_effective:
        totals["effective_repair"] = 0

    return {
        "edit_schema_version": EDIT_SCHEMA_VERSION,
        "timeline_id": store.timeline_id,
        "timeline_binding": store.binding.to_dict(),
        "edit_revision": store.edit_revision,
        "zip_sha256": store.zip_sha256,
        "edit_signature": store.edit_signature,
        "edited_frames": store.edited_frames(),
        "edited_frame_count": len(store.edited_frames()),
        "ground_truth_status": GROUND_TRUTH_STATUS,
        "protect_or_reject_auto_promoted_to_gt": False,
        "untouched_auto_as_gt": False,
        "counts": totals,
        "frames": per_frame_rows if include_per_frame else [],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.partial.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_png_file(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_bitmask_png(arr))


def validate_frame_selection(
    store: DustMaskEditStore,
    frames: Iterable[int] | None,
) -> list[int] | None:
    """Validate an optional frame selection without coercing strings, bytes, floats, or booleans."""
    if frames is None:
        return None
    if isinstance(frames, (str, bytes, bytearray)):
        raise ValueError("frames must be an iterable of integer frame indices, not a string or bytes")
    raw_frames = list(frames)
    return sorted({store._validate_frame_index(frame) for frame in raw_frames})


def export_provenance_bundle(
    store: DustMaskEditStore,
    export_dir: Path | str,
    *,
    policy: RepairInclusionPolicy | str = RepairInclusionPolicy.CONSERVATIVE,
    dilate: int = 0,
    auto_frames: Mapping[int, Mapping[str, np.ndarray]] | None = None,
    detector_metadata: Mapping[str, Any] | None = None,
    frames: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Export separated PNG layers, source metadata, and non-GT provenance.json cleanly."""
    store.verify()
    policy_enum = _normalize_policy(policy)
    dilate_iters = int(dilate)
    if dilate_iters < 0:
        raise ValueError(f"dilate must be >= 0, got {dilate}")

    validated_frames = validate_frame_selection(store, frames)

    out_dir = Path(export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edit_map = store.read_all_frames()
    auto_map = {int(k): dict(v) for k, v in (auto_frames or {}).items()}

    if validated_frames is not None:
        target_frames = validated_frames
    else:
        target_frames = sorted(set(edit_map.keys()) | set(auto_map.keys()))

    human_edits_dir = out_dir / "human_edits"
    protect_dir = out_dir / "protect_intact"
    effective_dir = out_dir / "effective_repair"
    human_edits_dir.mkdir(parents=True, exist_ok=True)
    protect_dir.mkdir(parents=True, exist_ok=True)
    effective_dir.mkdir(parents=True, exist_ok=True)

    include_auto = bool(auto_map)
    auto_dir = out_dir / "auto_proposals"
    if include_auto:
        for channel in ("scratch", "dust", "uncertain", "vetoed"):
            (auto_dir / channel).mkdir(parents=True, exist_ok=True)

    h, w = store.binding.height, store.binding.width
    zeros = np.zeros((h, w), dtype=np.uint8)
    effective_frames: dict[int, np.ndarray] = {}

    for idx in target_frames:
        edit_u8 = edit_map.get(idx, zeros)
        auto_entry = auto_map.get(idx, {})
        auto_s = auto_entry.get("scratch")
        auto_d = auto_entry.get("dust")
        auto_u = auto_entry.get("uncertain")
        auto_v = auto_entry.get("vetoed")
        auto_c = auto_entry.get("combined")

        resolved = resolve_effective_channels(
            auto_s,
            auto_d,
            auto_u,
            edit_u8,
            policy_enum,
            auto_combined=auto_c,
            dilate=dilate_iters,
        )
        effective_frames[idx] = resolved["effective"]

        fname = member_name(idx)
        _write_png_file(human_edits_dir / fname, edit_u8)
        _write_png_file(
            human_edits_dir / "scratch_add" / fname,
            np.where((edit_u8 & BIT_SCRATCH_ADD) != 0, np.uint8(255), np.uint8(0)),
        )
        _write_png_file(
            human_edits_dir / "scratch_reject" / fname,
            np.where((edit_u8 & BIT_SCRATCH_REJECT) != 0, np.uint8(255), np.uint8(0)),
        )
        _write_png_file(
            human_edits_dir / "dust_add" / fname,
            np.where((edit_u8 & BIT_DUST_ADD) != 0, np.uint8(255), np.uint8(0)),
        )
        _write_png_file(
            human_edits_dir / "dust_reject" / fname,
            np.where((edit_u8 & BIT_DUST_REJECT) != 0, np.uint8(255), np.uint8(0)),
        )
        _write_png_file(
            human_edits_dir / "uncertain_reject" / fname,
            np.where((edit_u8 & BIT_UNCERTAIN_REJECT) != 0, np.uint8(255), np.uint8(0)),
        )
        _write_png_file(protect_dir / fname, resolved["protected"])
        _write_png_file(effective_dir / fname, resolved["effective"])

        if include_auto:
            _write_png_file(
                auto_dir / "scratch" / fname,
                np.where(_to_bool_mask(auto_s, (h, w)), np.uint8(255), np.uint8(0)),
            )
            _write_png_file(
                auto_dir / "dust" / fname,
                np.where(_to_bool_mask(auto_d, (h, w)), np.uint8(255), np.uint8(0)),
            )
            _write_png_file(
                auto_dir / "uncertain" / fname,
                np.where(_to_bool_mask(auto_u, (h, w)), np.uint8(255), np.uint8(0)),
            )
            _write_png_file(
                auto_dir / "vetoed" / fname,
                np.where(_to_bool_mask(auto_v, (h, w)), np.uint8(255), np.uint8(0)),
            )

    summary = build_edit_provenance_summary(
        store,
        auto_frames=auto_map if include_auto else None,
        effective_frames=effective_frames,
        include_per_frame=True,
    )
    exported_at = datetime.now(timezone.utc).isoformat()
    source_ref = store.binding.to_dict()
    det_meta = dict(detector_metadata or {})

    metadata_payload = {
        "exported_at": exported_at,
        "timeline_id": store.timeline_id,
        "edit_schema_version": EDIT_SCHEMA_VERSION,
        "edit_revision": store.edit_revision,
        "zip_sha256": store.zip_sha256,
        "edit_signature": store.edit_signature,
        "source_reference": source_ref,
        "repair_policy": policy_enum.value,
        "dilate": dilate_iters,
        "detector_metadata": det_meta,
        "exported_frames": target_frames,
    }
    provenance_payload = {
        **summary,
        "exported_at": exported_at,
        "source_reference": source_ref,
        "repair_policy": policy_enum.value,
        "dilate": dilate_iters,
        "detector_metadata": det_meta,
        "exported_frames": target_frames,
        "exported_directories": {
            "human_edits": "human_edits",
            "protect_intact": "protect_intact",
            "effective_repair": "effective_repair",
            "auto_proposals": "auto_proposals" if include_auto else None,
        },
        "ground_truth_status": GROUND_TRUTH_STATUS,
        "protect_or_reject_auto_promoted_to_gt": False,
        "untouched_auto_as_gt": False,
    }

    metadata_path = out_dir / "metadata.json"
    provenance_path = out_dir / "provenance.json"
    _write_json_atomic(metadata_path, metadata_payload)
    _write_json_atomic(provenance_path, provenance_payload)

    return {
        "export_dir": str(out_dir),
        "metadata_path": str(metadata_path),
        "provenance_path": str(provenance_path),
        "exported_frames": target_frames,
        "summary": provenance_payload,
    }
