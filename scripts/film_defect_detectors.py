"""Pluggable film defect detector strategy interface and registry.

Defines the abstract `BaseDefectDetector` contract, `DefectMaskResult` container,
and `DefectDetectorRegistry` for discovering, validating, and dispatching
commercial-safe CPU film defect detectors across the CLI and GUI pipelines.

All production detector strategies rely exclusively on unencumbered,
commercial-safe libraries (OpenCV, NumPy, Python stdlib) and never import
restricted neural checkpoints or non-commercial dependencies (DART, RTN,
ProPainter, PyTorch).

Pixel convention: uint8 grayscale (H, W), 255 = suspect defect, 0 = clean.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

if __name__ in sys.modules:
    sys.modules.setdefault("film_defect_detectors", sys.modules[__name__])
    sys.modules.setdefault("scripts.film_defect_detectors", sys.modules[__name__])

import cv2
import numpy as np

try:
    import film_defect_masks as fdm
    import film_scratch_ridge as fsr
    from adaptive_tiling import AdaptiveTilingConfig, AdaptiveTilingEngine
except ImportError:
    from scripts import film_defect_masks as fdm
    from scripts import film_scratch_ridge as fsr
    from scripts.adaptive_tiling import AdaptiveTilingConfig, AdaptiveTilingEngine

TILING_CONFIG_KEYS = {
    "enable_tiling",
    "auto_tiling",
    "tile_size",
    "overlap",
    "context_padding",
    "blend_mode",
    "vram_budget_mb",
    "batch_size",
}


class ReadinessState(str, Enum):
    """Lifecycle and operational readiness classification for defect detection models."""

    PRETRAINED_OPERATIONAL = "pretrained_operational"
    TRAINED_EXPERIMENTAL = "trained_experimental"
    TRAINING_REQUIRED = "training_required"
    BLOCKED = "blocked"


class ChannelOrigin(str, Enum):
    """Source classification for defect mask channels."""

    NATIVE = "native"
    DERIVED = "derived"
    UNSUPPORTED = "unsupported"


class RepairInclusionPolicy(str, Enum):
    """Governs which defect channels are included in the effective repair mask."""

    CONSERVATIVE = "conservative"          # Scratch | Dust (Uncertain excluded)
    HIGH_RECALL = "high_recall"            # Scratch | Dust | Uncertain
    SCRATCH_ONLY = "scratch_only"          # Scratch only
    COMBINED_DEFECT = "combined_defect"    # Full combined defect mask


class UnsupportedPolicyError(ValueError):
    """Raised when a selected RepairInclusionPolicy requires channels unsupported by the detector."""


class MaskContractValidationError(ValueError):
    """Raised when mask data structures violate dimension, dtype, range, or value invariants."""


@dataclass
class ChannelProvenance:
    """Provenance metadata describing how a defect channel was produced."""

    origin: ChannelOrigin
    is_available: bool
    source_algorithm: str
    derivation_parameters: dict[str, Any] = field(default_factory=dict)
    confidence_threshold: float = 0.5


@dataclass
class MultiChannelDefectMask:
    """Standardized multi-channel defect mask container with strict invariant validation.

    Attributes:
        combined_probability: Continuous defect probability / anomaly score map (float32, [H, W], in [0.0, 1.0]).
        combined_mask: Binary combined defect proposal mask (uint8, [H, W], strictly {0, 255}).
        scratch_mask: Optional binary scratch channel mask (uint8, [H, W], strictly {0, 255}).
        dust_mask: Optional binary dust channel mask (uint8, [H, W], strictly {0, 255}).
        uncertain_mask: Optional binary unclassified/ambiguous mask (uint8, [H, W], strictly {0, 255}).
        provenance: Mapping from channel name ('scratch', 'dust', 'uncertain') to ChannelProvenance.
        metadata: Diagnostic and execution metadata.
    """

    combined_probability: np.ndarray
    combined_mask: np.ndarray
    scratch_mask: np.ndarray | None = None
    dust_mask: np.ndarray | None = None
    uncertain_mask: np.ndarray | None = None
    provenance: dict[str, ChannelProvenance] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.combined_probability, np.ndarray):
            raise MaskContractValidationError("combined_probability must be a numpy.ndarray")
        if self.combined_probability.ndim != 2:
            raise MaskContractValidationError(
                f"combined_probability must be 2D, got {self.combined_probability.ndim}D"
            )
        if self.combined_probability.size == 0:
            raise MaskContractValidationError("combined_probability cannot be empty")
        if self.combined_probability.dtype != np.float32:
            raise MaskContractValidationError(
                f"combined_probability must be float32, got {self.combined_probability.dtype}"
            )
        if not np.all(np.isfinite(self.combined_probability)):
            raise MaskContractValidationError("combined_probability contains non-finite values (NaN / Inf)")
        p_min, p_max = float(self.combined_probability.min()), float(self.combined_probability.max())
        if p_min < 0.0 or p_max > 1.0:
            raise MaskContractValidationError(
                f"combined_probability values must be in [0.0, 1.0], found [{p_min}, {p_max}]"
            )

        if not isinstance(self.combined_mask, np.ndarray):
            raise MaskContractValidationError("combined_mask must be a numpy.ndarray")
        if self.combined_mask.ndim != 2:
            raise MaskContractValidationError(
                f"combined_mask must be 2D, got {self.combined_mask.ndim}D"
            )
        if self.combined_mask.dtype != np.uint8:
            raise MaskContractValidationError(f"combined_mask must be uint8, got {self.combined_mask.dtype}")
        h, w = self.combined_mask.shape
        if self.combined_probability.shape != (h, w):
            raise MaskContractValidationError(
                f"Shape mismatch: probability {self.combined_probability.shape} vs mask {(h, w)}"
            )
        if not np.all((self.combined_mask == 0) | (self.combined_mask == 255)):
            raise MaskContractValidationError("combined_mask must contain strictly binary values: 0 or 255")

        for name, mask in [
            ("scratch", self.scratch_mask),
            ("dust", self.dust_mask),
            ("uncertain", self.uncertain_mask),
        ]:
            if mask is not None:
                if not isinstance(mask, np.ndarray):
                    raise MaskContractValidationError(f"{name}_mask must be a numpy.ndarray")
                if mask.ndim != 2:
                    raise MaskContractValidationError(f"{name}_mask must be 2D, got {mask.ndim}D")
                if mask.dtype != np.uint8:
                    raise MaskContractValidationError(f"{name}_mask must be uint8, got {mask.dtype}")
                if mask.shape != (h, w):
                    raise MaskContractValidationError(
                        f"Shape mismatch for {name}_mask: {mask.shape} vs expected {(h, w)}"
                    )
                if not np.all((mask == 0) | (mask == 255)):
                    raise MaskContractValidationError(f"{name}_mask must contain strictly binary values: 0 or 255")

        for name, m in [
            ("scratch", self.scratch_mask),
            ("dust", self.dust_mask),
            ("uncertain", self.uncertain_mask),
        ]:
            if name in self.provenance:
                prov = self.provenance[name]
                if prov.is_available and m is None:
                    raise MaskContractValidationError(
                        f"Channel {name} marked available in provenance but array is None"
                    )
                if not prov.is_available and m is not None:
                    raise MaskContractValidationError(
                        f"Channel {name} marked unavailable in provenance but array is provided"
                    )

    def get_effective_repair_mask(
        self, policy: RepairInclusionPolicy = RepairInclusionPolicy.CONSERVATIVE
    ) -> np.ndarray:
        """Resolve the effective binary repair mask based on the specified inclusion policy.

        Args:
            policy: RepairInclusionPolicy governing channel combination.

        Returns:
            uint8 2D binary ndarray with strictly {0, 255} values.

        Raises:
            UnsupportedPolicyError: If the policy requires channels that are unsupported/missing.
        """
        if not isinstance(policy, RepairInclusionPolicy):
            try:
                policy = RepairInclusionPolicy(str(policy).lower())
            except ValueError:
                raise UnsupportedPolicyError(f"Unknown or unsupported RepairInclusionPolicy: {policy}")

        h, w = self.combined_mask.shape
        empty = np.zeros((h, w), dtype=np.uint8)

        if policy == RepairInclusionPolicy.COMBINED_DEFECT:
            return self.combined_mask.copy()

        if policy == RepairInclusionPolicy.SCRATCH_ONLY:
            if self.scratch_mask is None:
                raise UnsupportedPolicyError(
                    "Policy 'SCRATCH_ONLY' requested but model does not provide a Scratch channel."
                )
            return self.scratch_mask.copy()

        if policy == RepairInclusionPolicy.CONSERVATIVE:
            if self.scratch_mask is None and self.dust_mask is None:
                raise UnsupportedPolicyError(
                    "Policy 'CONSERVATIVE' requires Scratch or Dust channels, but model only provides "
                    "Combined defect output. Select 'COMBINED_DEFECT' policy explicitly if full defect inpainting is desired."
                )
            s = self.scratch_mask if self.scratch_mask is not None else empty
            d = self.dust_mask if self.dust_mask is not None else empty
            return np.bitwise_or(s, d)

        if policy == RepairInclusionPolicy.HIGH_RECALL:
            if self.scratch_mask is None and self.dust_mask is None and self.uncertain_mask is None:
                raise UnsupportedPolicyError(
                    "Policy 'HIGH_RECALL' requested but model provides no classified or uncertain channels."
                )
            s = self.scratch_mask if self.scratch_mask is not None else empty
            d = self.dust_mask if self.dust_mask is not None else empty
            u = self.uncertain_mask if self.uncertain_mask is not None else empty
            return np.bitwise_or(np.bitwise_or(s, d), u)

        raise UnsupportedPolicyError(f"Unknown or unsupported RepairInclusionPolicy: {policy}")

    def to_legacy_compat_bridge(
        self, policy: RepairInclusionPolicy = RepairInclusionPolicy.CONSERVATIVE
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Bridge multi-channel masks to legacy (scratch, dust) 2-channel consumers.

        Guarantees that `np.bitwise_or(scratch, dust)` is 100% pixel-identical to
        `get_effective_repair_mask(policy)`.

        Args:
            policy: Inclusion policy used to resolve the effective repair mask.

        Returns:
            Tuple of (scratch_channel, dust_channel, manifest_dict).
            The manifest is guaranteed to be 100% JSON-serializable with no raw numpy types.
        """
        if not isinstance(policy, RepairInclusionPolicy):
            try:
                policy = RepairInclusionPolicy(str(policy).lower())
            except ValueError:
                raise UnsupportedPolicyError(f"Unknown or unsupported RepairInclusionPolicy: {policy}")

        effective_mask = self.get_effective_repair_mask(policy)
        h, w = self.combined_mask.shape
        empty = np.zeros((h, w), dtype=np.uint8)

        if self.scratch_mask is None and self.dust_mask is None:
            manifest: dict[str, Any] = {
                "is_compatibility_mapping": True,
                "semantic_type": "generic_combined_defect",
                "policy_resolved": policy.value,
                "mask_pixel_count": int(np.count_nonzero(effective_mask)),
                "warning": (
                    "Generic defect mask delivered via compatibility bridge; "
                    "do not apply dust-specific erosion."
                ),
            }
            json.dumps(manifest)
            return empty, effective_mask.copy(), manifest

        if policy == RepairInclusionPolicy.SCRATCH_ONLY:
            s = effective_mask.copy()
            d = empty
        elif policy == RepairInclusionPolicy.CONSERVATIVE:
            s = self.scratch_mask.copy() if self.scratch_mask is not None else empty
            d = self.dust_mask.copy() if self.dust_mask is not None else empty
        elif policy == RepairInclusionPolicy.HIGH_RECALL:
            s = self.scratch_mask.copy() if self.scratch_mask is not None else empty
            dust_base = self.dust_mask.copy() if self.dust_mask is not None else empty
            u = self.uncertain_mask.copy() if self.uncertain_mask is not None else empty
            d = np.bitwise_or(dust_base, u)
        elif policy == RepairInclusionPolicy.COMBINED_DEFECT:
            s = empty
            d = effective_mask.copy()
        else:
            raise UnsupportedPolicyError(f"Unknown policy: {policy}")

        union_check = np.bitwise_or(s, d)
        if not np.array_equal(union_check, effective_mask):
            raise MaskContractValidationError("Legacy union must be pixel-identical to effective_mask")

        manifest = {
            "is_compatibility_mapping": bool(
                policy in (RepairInclusionPolicy.COMBINED_DEFECT, RepairInclusionPolicy.HIGH_RECALL)
            ),
            "policy_resolved": policy.value,
            "mask_pixel_count": int(np.count_nonzero(effective_mask)),
            "scratch_available": bool(self.scratch_mask is not None),
            "dust_available": bool(self.dust_mask is not None),
            "uncertain_available": bool(self.uncertain_mask is not None),
        }
        json.dumps(manifest)
        return s, d, manifest

    def as_dict(self) -> dict[str, Any]:
        """Return dictionary representation with numpy arrays and json-safe metadata."""
        return {
            "combined_probability": self.combined_probability,
            "combined_mask": self.combined_mask,
            "scratch_mask": self.scratch_mask,
            "dust_mask": self.dust_mask,
            "uncertain_mask": self.uncertain_mask,
            "provenance": {k: v.__dict__.copy() for k, v in self.provenance.items()},
            "metadata": dict(self.metadata),
        }


class UnknownDetectorError(KeyError, ValueError):
    """Raised when a requested detector name or alias is not in the registry."""


class CommercialLicenseError(PermissionError, ValueError):
    """Raised when an un-cleared / research-only detector is requested in production mode."""


class DetectorNotExecutableError(RuntimeError):
    """Raised when an unexecutable detector (TRAINING_REQUIRED or BLOCKED) is selected for active execution."""


@dataclass(frozen=True)
class PreprocessedFrames:
    """Validated, normalized BGR and grayscale frame sequences."""

    bgr_frames: list[np.ndarray]
    gray_frames: list[np.ndarray]
    height: int
    width: int


@dataclass
class DefectMaskResult:
    """Standardized output container returned by `BaseDefectDetector.detect()`.

    Attributes:
        mask: Combined per-frame binary proposal masks (uint8, 0 or 255).
        scratch_mask: Per-frame binary scratch channel masks (uint8, 0 or 255).
        dust_mask: Per-frame binary dust/blotch channel masks (uint8, 0 or 255).
        metadata: Provenance, detector name, parameters, and diagnostic details.
        uncertain_mask: Optional per-frame binary unclassified/uncertain masks (uint8, 0 or 255).
        multi_channel_masks: Optional per-frame MultiChannelDefectMask contract objects.
    """

    mask: list[np.ndarray]
    scratch_mask: list[np.ndarray]
    dust_mask: list[np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    uncertain_mask: list[np.ndarray] | None = None
    multi_channel_masks: list[MultiChannelDefectMask] | None = None

    @property
    def masks(self) -> list[np.ndarray]:
        return self.mask

    @property
    def scratch_masks(self) -> list[np.ndarray]:
        return self.scratch_mask

    @property
    def dust_masks(self) -> list[np.ndarray]:
        return self.dust_mask

    @property
    def uncertain_masks(self) -> list[np.ndarray] | None:
        return self.uncertain_mask

    def __len__(self) -> int:
        return len(self.mask)

    def as_dict(self) -> dict[str, Any]:
        res = {
            "mask": self.mask,
            "scratch": self.scratch_mask,
            "dust": self.dust_mask,
            "metadata": dict(self.metadata),
        }
        if self.uncertain_mask is not None:
            res["uncertain"] = self.uncertain_mask
        return res

    def to_legacy_compat_bridge(
        self, policy: RepairInclusionPolicy = RepairInclusionPolicy.CONSERVATIVE
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
        """Bridge all frames in DefectMaskResult to legacy (scratch, dust) pairs."""
        if self.multi_channel_masks is not None and len(self.multi_channel_masks) == len(self.mask):
            s_list, d_list, m_list = [], [], []
            for mcdm in self.multi_channel_masks:
                s, d, manifest = mcdm.to_legacy_compat_bridge(policy)
                s_list.append(s)
                d_list.append(d)
                m_list.append(manifest)
            return s_list, d_list, m_list
        s_list = [s.copy() for s in self.scratch_mask]
        d_list = [d.copy() for d in self.dust_mask]
        manifests = [
            {
                "is_compatibility_mapping": False,
                "policy_resolved": (
                    policy.value if isinstance(policy, RepairInclusionPolicy) else str(policy)
                ),
                "scratch_available": True,
                "dust_available": True,
            }
            for _ in range(len(self.mask))
        ]
        return s_list, d_list, manifests


def normalize_frames(frames: Sequence[np.ndarray] | np.ndarray) -> PreprocessedFrames:
    """Validate and normalize a sequence of uint8 grayscale or BGR frames."""
    if isinstance(frames, np.ndarray):
        if frames.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D frame array, got shape {frames.shape}")
        raw_list = [frames[i] for i in range(frames.shape[0])]
    else:
        raw_list = list(frames)

    if not raw_list:
        raise ValueError("Frame sequence must not be empty")

    bgr_list: list[np.ndarray] = []
    gray_list: list[np.ndarray] = []
    first_hw: tuple[int, int] | None = None

    for idx, frame in enumerate(raw_list):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            raise ValueError(f"Frame {idx} must be uint8, got {arr.dtype}")
        if arr.ndim == 2:
            gray = arr.copy()
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            bgr = arr.copy()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Frame {idx} must be 2D grayscale or 3D BGR, got shape {arr.shape}")

        hw = (int(gray.shape[0]), int(gray.shape[1]))
        if first_hw is None:
            first_hw = hw
        elif hw != first_hw:
            raise ValueError(f"Frame {idx} dimensions {hw} do not match first frame {first_hw}")

        bgr_list.append(bgr)
        gray_list.append(gray)

    assert first_hw is not None
    return PreprocessedFrames(
        bgr_frames=bgr_list,
        gray_frames=gray_list,
        height=first_hw[0],
        width=first_hw[1],
    )


def _split_temporal_mask_components(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Partition an accepted `temporal_defect_mask` binary output into (scratch, dust) channels.

    Preserves exact pixel conservation (`scratch_mask | dust_mask == binary_mask`).
    """
    h, w = binary_mask.shape
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    dust_mask = np.zeros((h, w), dtype=np.uint8)
    if not np.any(binary_mask):
        return scratch_mask, dust_mask

    rough = (binary_mask > 0).astype(np.uint8) * 255
    total, labels, stats, _ = cv2.connectedComponentsWithStats(rough, connectivity=8)
    for component in range(1, total):
        _x, _y, cw, ch, area = map(int, stats[component])
        fine_scratch = (
            cw <= max(3, int(w * 0.009))
            and ch >= 10
            and ch <= max(12, int(h * 0.60))
            and area <= max(40, int(h * 0.75))
        )
        if fine_scratch:
            scratch_mask[labels == component] = 255
        else:
            dust_mask[labels == component] = 255
    return scratch_mask, dust_mask


def split_proposal_components(binary_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classify raw connected components of a binary proposal into (scratch, dust) masks."""
    if binary_mask.ndim != 2 or binary_mask.dtype != np.uint8:
        raise ValueError("Expected 2D uint8 binary mask")
    h, w = binary_mask.shape
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    dust_mask = np.zeros((h, w), dtype=np.uint8)

    if not np.any(binary_mask):
        return scratch_mask, dust_mask

    rough = (binary_mask > 0).astype(np.uint8) * 255
    total, labels, stats, _ = cv2.connectedComponentsWithStats(rough, connectivity=8)
    max_blob_area = max(12, min(220, int(round(h * w * 0.004))))
    max_blob_edge = max(8, min(28, int(round(min(h, w) * 0.12))))

    for component in range(1, total):
        _x, _y, cw, ch, area = map(int, stats[component])
        fine_scratch = (
            cw <= max(8, int(w * 0.06))
            and ch >= max(8, int(h * 0.15))
            and ch >= 3 * cw
            and area <= max(64, cw * h)
        )
        small_spot = (
            not fine_scratch
            and area <= max_blob_area
            and cw <= max_blob_edge
            and ch <= max_blob_edge
        )
        if fine_scratch:
            scratch_mask[labels == component] = 255
        elif small_spot:
            dust_mask[labels == component] = 255

    max_scratch_frac = max(fdm.MAX_MASK_FRACTION, min(0.20, 12.0 / max(w, 1)))
    max_dust_frac = max(fdm.MAX_MASK_FRACTION, min(0.15, 8.0 / max(w, 1)))
    if np.count_nonzero(scratch_mask) / scratch_mask.size > max_scratch_frac:
        scratch_mask[:] = 0
    if np.count_nonzero(dust_mask) / dust_mask.size > max_dust_frac:
        dust_mask[:] = 0
    return scratch_mask, dust_mask


def _horizontal_flank_diff(centre: np.ndarray, flank: int = 4) -> np.ndarray:
    """Compute horizontal flank contrast so narrow vertical scratches are preserved."""
    _h, w = centre.shape
    f_px = max(2, min(flank, w // 4))
    padded = np.pad(centre, ((0, 0), (f_px, f_px)), mode="edge")
    left = padded[:, :w]
    right = padded[:, 2 * f_px : 2 * f_px + w]
    return np.minimum(np.abs(centre - left), np.abs(centre - right))


def _temporal_pair_proposal(
    before: np.ndarray,
    centre_u8: np.ndarray,
    after: np.ndarray,
    sensitivity: float,
) -> np.ndarray:
    """Compute raw binary defect evidence from motion-aligned neighbor frames."""
    centre = centre_u8.astype(np.float32)
    target = (before + after) / 2.0
    mismatch = np.abs(centre - target)
    neighbor_disagreement = np.abs(before - after)

    median_difference = float(np.median(mismatch))
    mad = float(np.median(np.abs(mismatch - median_difference)))
    threshold = max(18.0, median_difference + (7.0 - 3.0 * sensitivity) * max(mad, 1.2))
    reference_agreement = neighbor_disagreement <= max(12.0, threshold * 0.70)
    temporal = mismatch >= threshold

    spatial_2d = np.abs(centre - cv2.medianBlur(centre_u8, 5).astype(np.float32))
    spatial = np.maximum(spatial_2d, _horizontal_flank_diff(centre, flank=4))
    local_threshold = max(11.0, threshold * (0.47 - 0.12 * sensitivity))
    likely = temporal & reference_agreement & (spatial >= local_threshold)

    picture = centre_u8 >= max(8, int(np.percentile(centre_u8, 98) * 0.025))
    likely &= picture
    likely[:2, :] = False
    likely[-2:, :] = False
    likely[:, :2] = False
    likely[:, -2:] = False
    return likely.astype(np.uint8) * 255


def _contiguous_true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs in a 1D boolean NumPy array."""
    padded = np.pad(np.asarray(values, dtype=np.uint8), (1, 1))
    changes = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(start), int(end)) for start, end in changes.reshape(-1, 2)]


def legacy_spatial_scratch_mask(frame: np.ndarray, sensitivity: float = 0.65) -> np.ndarray:
    """Single-frame spatial vertical scratch detector (`ai_scratch_mask` algorithm)."""
    if not math.isfinite(sensitivity) or not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("Sensitivity must be finite in [0, 1]")
    sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
    gray_u8 = fdm.to_gray(frame)
    gray = gray_u8.astype(np.float32)
    height, width = gray.shape
    if height < 16 or width < 16:
        return np.zeros((height, width), dtype=np.uint8)

    horizontal_base = cv2.GaussianBlur(gray, (9, 1), sigmaX=2.1, sigmaY=0)
    detail = gray - horizontal_base
    vertical_window = max(15, min(101, (height // 10) | 1))
    coherent = cv2.GaussianBlur(
        detail, (1, vertical_window), sigmaX=0, sigmaY=max(2.0, vertical_window / 5.0)
    )
    strength = np.abs(coherent)
    centre = float(np.median(strength))
    mad = float(np.median(np.abs(strength - centre)))
    sigma = 5.2 - 3.4 * sensitivity
    threshold = max(0.85, centre + sigma * max(mad, 0.12))
    detail_floor = max(1.5, float(np.percentile(np.abs(detail), 45)) * (0.72 - 0.22 * sensitivity))
    mask = ((strength >= threshold) & (np.abs(detail) >= detail_floor)).astype(np.uint8) * 255

    open_length = max(5, min(19, (height // 72) | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_length, 1), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(3, open_length // 2), 1), np.uint8))
    components, labels, stats, _centres = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    max_width = max(10, min(13, int(round(width * 0.009))))
    min_height = max(12, int(round(height * 0.07)))
    for label in range(1, components):
        _x, _y, component_width, component_height, area = stats[label]
        if (
            component_width <= max_width
            and component_height >= min_height
            and component_height >= component_width * 7
            and area >= min_height
        ):
            clean[labels == label] = 255

    column = np.percentile(strength, 70, axis=0)
    col_centre = float(np.median(column))
    col_mad = float(np.median(np.abs(column - col_centre)))
    column_threshold = max(threshold * 0.72, col_centre + (4.8 - 2.8 * sensitivity) * max(col_mad, 0.12))
    candidates = column >= column_threshold
    for start, end in _contiguous_true_runs(candidates):
        if end - start <= max_width:
            clean[:, start:end] = 255

    picture = gray > max(4.0, float(np.percentile(gray, 92)) * 0.025)
    clean[picture == 0] = 0
    coverage = float(np.count_nonzero(clean)) / max(clean.size, 1)
    max_coverage = 0.28
    if coverage > max_coverage:
        selected_strength = strength[clean > 0]
        cutoff = float(np.quantile(selected_strength, 1.0 - max_coverage / coverage))
        clean[(clean > 0) & (strength < cutoff)] = 0
    return clean


class BaseDefectDetector(ABC):
    """Abstract strategy interface for film scratch and dust mask detectors."""

    name: str = ""
    aliases: tuple[str, ...] = ()
    description: str = ""
    license_status: str = "Apache-2.0 / BSD-3-Clause"
    commercial_cleared: bool = True
    temporal_radius: int = 1
    readiness_state: ReadinessState = ReadinessState.PRETRAINED_OPERATIONAL
    supported_channels: tuple[str, ...] = ("scratch", "dust")
    unsupported_channels: tuple[str, ...] = ()

    @property
    @abstractmethod
    def schema(self) -> dict[str, dict[str, Any]]:
        """Return parameter schema mapping parameter names to type/range/default specs."""

    def validate_config(self, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate configuration dictionary against `self.schema` and fill defaults."""
        spec_map = self.schema
        validated: dict[str, Any] = {
            key: spec.get("default") for key, spec in spec_map.items()
        }
        if config is None:
            return validated
        if not isinstance(config, Mapping):
            raise TypeError(f"Detector config must be a mapping, got {type(config).__name__}")

        for key, raw_val in config.items():
            if key in TILING_CONFIG_KEYS:
                validated[key] = raw_val
                continue
            if key not in spec_map:
                raise ValueError(
                    f"Unknown parameter {key!r} for detector {self.name!r}. "
                    f"Supported parameters: {sorted(spec_map.keys())}"
                )
            spec = spec_map[key]
            expected_type = spec.get("type", "any")

            if expected_type == "float":
                if isinstance(raw_val, bool):
                    raise ValueError(f"Parameter {key!r} must be a float, got boolean {raw_val!r}")
                try:
                    val = float(raw_val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Parameter {key!r} must be numeric, got {raw_val!r}") from exc
                if not math.isfinite(val):
                    raise ValueError(f"Parameter {key!r} must be finite, got {raw_val!r}")
                if "min" in spec and val < float(spec["min"]):
                    raise ValueError(f"Parameter {key!r} must be >= {spec['min']}, got {val}")
                if "max" in spec and val > float(spec["max"]):
                    raise ValueError(f"Parameter {key!r} must be <= {spec['max']}, got {val}")
                validated[key] = val

            elif expected_type == "int":
                if isinstance(raw_val, bool):
                    raise ValueError(f"Parameter {key!r} must be an int, got boolean {raw_val!r}")
                try:
                    fval = float(raw_val)
                    ival = int(raw_val)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Parameter {key!r} must be an integer, got {raw_val!r}") from exc
                if not math.isfinite(fval) or fval != ival:
                    raise ValueError(f"Parameter {key!r} must be an exact integer, got {raw_val!r}")
                if "min" in spec and ival < int(spec["min"]):
                    raise ValueError(f"Parameter {key!r} must be >= {spec['min']}, got {ival}")
                if "max" in spec and ival > int(spec["max"]):
                    raise ValueError(f"Parameter {key!r} must be <= {spec['max']}, got {ival}")
                validated[key] = ival

            elif expected_type == "bool":
                if isinstance(raw_val, bool):
                    validated[key] = raw_val
                elif isinstance(raw_val, str) and raw_val.strip().lower() in ("true", "false", "1", "0"):
                    validated[key] = raw_val.strip().lower() in ("true", "1")
                else:
                    raise ValueError(f"Parameter {key!r} must be a boolean, got {raw_val!r}")

            elif expected_type == "str":
                if not isinstance(raw_val, (str, Path)):
                    raise ValueError(f"Parameter {key!r} must be a string, got {raw_val!r}")
                sval = str(raw_val)
                if "choices" in spec and sval not in spec["choices"]:
                    raise ValueError(
                        f"Parameter {key!r} must be one of {spec['choices']}, got {sval!r}"
                    )
                validated[key] = sval

            elif expected_type == "tuple":
                if isinstance(raw_val, (str, bytes)) or not isinstance(raw_val, Sequence):
                    raise ValueError(f"Parameter {key!r} must be a sequence of integers, got {raw_val!r}")
                items = tuple(int(x) for x in raw_val)
                if not items or any(x < 1 for x in items):
                    raise ValueError(f"Parameter {key!r} must contain positive integers, got {items!r}")
                validated[key] = items

            else:
                validated[key] = raw_val

        return validated

    def should_use_tiling(
        self,
        height: int,
        width: int,
        config: Mapping[str, Any] | None = None,
    ) -> bool:
        """Determine whether adaptive tiling should be activated for the frame geometry."""
        if config:
            if config.get("enable_tiling") is True:
                return True
            if config.get("enable_tiling") is False:
                return False
            if "tile_size" in config and config["tile_size"] is not None and config["tile_size"] != 0:
                return True
            if config.get("auto_tiling") is True:
                return height >= 1080 or width >= 1920
        return False

    def detect_tiled(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
        tiling_config: AdaptiveTilingConfig | None = None,
    ) -> DefectMaskResult:
        """Execute tiled defect detection across high-resolution frames using AdaptiveTilingEngine."""
        seq = normalize_frames(frames)
        h, w = seq.height, seq.width
        n_frames = len(seq.bgr_frames)

        if tiling_config is None:
            tile_sz = config.get("tile_size", (512, 512)) if config else (512, 512)
            overlap = config.get("overlap", (64, 64)) if config else (64, 64)
            padding = int(config.get("context_padding", 32)) if config else 32
            blend_mode = str(config.get("blend_mode", "gaussian")) if config else "gaussian"
            vram_mb = int(config.get("vram_budget_mb", 4096)) if config else 4096
            batch_sz = int(config.get("batch_size", 4)) if config else 4
            tiling_config = AdaptiveTilingConfig(
                tile_size=tile_sz,
                overlap=overlap,
                context_padding=padding,
                blend_mode=blend_mode,
                vram_budget_mb=vram_mb,
                batch_size=batch_sz,
            )

        engine = AdaptiveTilingEngine(tiling_config)
        coords = engine.generate_tiles((h, w))

        sub_config = {k: v for k, v in (config or {}).items() if k not in TILING_CONFIG_KEYS}
        sub_config["enable_tiling"] = False
        sub_config["auto_tiling"] = False

        if len(coords) <= 1:
            clean_cfg = {k: v for k, v in sub_config.items() if k not in ("enable_tiling", "auto_tiling")}
            return self.detect(frames, config=clean_cfg)

        combined_prob_tiles: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
        scratch_prob_tiles: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
        dust_prob_tiles: list[list[np.ndarray]] = [[] for _ in range(n_frames)]
        uncertain_prob_tiles: list[list[np.ndarray]] = [[] for _ in range(n_frames)]

        has_scratch = False
        has_dust = False
        has_uncertain = False

        for coord in coords:
            tile_frames_bgr = [
                engine.extract_padded_patch(f, coord, tiling_config.context_padding)
                for f in seq.bgr_frames
            ]
            clean_tile_cfg = {k: v for k, v in sub_config.items() if k not in ("enable_tiling", "auto_tiling")}
            sub_res = self.detect(tile_frames_bgr, config=clean_tile_cfg)

            for t in range(n_frames):
                if sub_res.multi_channel_masks and t < len(sub_res.multi_channel_masks):
                    c_prob = sub_res.multi_channel_masks[t].combined_probability
                else:
                    c_prob = sub_res.mask[t].astype(np.float32) / 255.0
                c_crop = engine.crop_context_padding(c_prob, coord, tiling_config.context_padding)
                combined_prob_tiles[t].append(c_crop)

                if sub_res.scratch_mask and t < len(sub_res.scratch_mask):
                    has_scratch = True
                    s_prob = sub_res.scratch_mask[t].astype(np.float32) / 255.0
                    s_crop = engine.crop_context_padding(s_prob, coord, tiling_config.context_padding)
                    scratch_prob_tiles[t].append(s_crop)

                if sub_res.dust_mask and t < len(sub_res.dust_mask):
                    has_dust = True
                    d_prob = sub_res.dust_mask[t].astype(np.float32) / 255.0
                    d_crop = engine.crop_context_padding(d_prob, coord, tiling_config.context_padding)
                    dust_prob_tiles[t].append(d_crop)

                if sub_res.uncertain_mask and t < len(sub_res.uncertain_mask) and sub_res.uncertain_mask[t] is not None:
                    has_uncertain = True
                    u_prob = sub_res.uncertain_mask[t].astype(np.float32) / 255.0
                    u_crop = engine.crop_context_padding(u_prob, coord, tiling_config.context_padding)
                    uncertain_prob_tiles[t].append(u_crop)

        raw_thresh = config.get("threshold", 160) if config else 160
        thresh_prob = (float(raw_thresh) / 255.0) if float(raw_thresh) > 1.0 else float(raw_thresh)

        final_masks: list[np.ndarray] = []
        final_scratch: list[np.ndarray] = []
        final_dust: list[np.ndarray] = []
        final_uncertain: list[np.ndarray] | None = [] if has_uncertain else None
        final_mcdm: list[MultiChannelDefectMask] = []

        for t in range(n_frames):
            comb_p = engine.blend_probability_maps(
                combined_prob_tiles[t],
                coords,
                (h, w),
                blend_mode=tiling_config.blend_mode,
            )
            comb_mask = np.where(comb_p >= thresh_prob, np.uint8(255), np.uint8(0))
            final_masks.append(comb_mask)

            s_mask = None
            if has_scratch and scratch_prob_tiles[t]:
                s_p = engine.blend_probability_maps(
                    scratch_prob_tiles[t],
                    coords,
                    (h, w),
                    blend_mode=tiling_config.blend_mode,
                )
                s_mask = np.where(s_p >= thresh_prob, np.uint8(255), np.uint8(0))
                final_scratch.append(s_mask)
            else:
                final_scratch.append(np.zeros((h, w), dtype=np.uint8))

            d_mask = None
            if has_dust and dust_prob_tiles[t]:
                d_p = engine.blend_probability_maps(
                    dust_prob_tiles[t],
                    coords,
                    (h, w),
                    blend_mode=tiling_config.blend_mode,
                )
                d_mask = np.where(d_p >= thresh_prob, np.uint8(255), np.uint8(0))
                final_dust.append(d_mask)
            else:
                final_dust.append(np.zeros((h, w), dtype=np.uint8))

            u_mask = None
            if has_uncertain and uncertain_prob_tiles[t] and final_uncertain is not None:
                u_p = engine.blend_probability_maps(
                    uncertain_prob_tiles[t],
                    coords,
                    (h, w),
                    blend_mode=tiling_config.blend_mode,
                )
                u_mask = np.where(u_p >= thresh_prob, np.uint8(255), np.uint8(0))
                final_uncertain.append(u_mask)

            prov: dict[str, ChannelProvenance] = {}
            if s_mask is not None:
                prov["scratch"] = ChannelProvenance(
                    origin=ChannelOrigin.NATIVE if "scratch" in getattr(self, "supported_channels", ()) else ChannelOrigin.DERIVED,
                    is_available=True,
                    source_algorithm=f"{self.name}_adaptive_tiling",
                    derivation_parameters={"threshold": thresh_prob},
                    confidence_threshold=thresh_prob,
                )
            if d_mask is not None:
                prov["dust"] = ChannelProvenance(
                    origin=ChannelOrigin.NATIVE if "dust" in getattr(self, "supported_channels", ()) else ChannelOrigin.DERIVED,
                    is_available=True,
                    source_algorithm=f"{self.name}_adaptive_tiling",
                    derivation_parameters={"threshold": thresh_prob},
                    confidence_threshold=thresh_prob,
                )
            if u_mask is not None:
                prov["uncertain"] = ChannelProvenance(
                    origin=ChannelOrigin.DERIVED,
                    is_available=True,
                    source_algorithm=f"{self.name}_adaptive_tiling",
                    derivation_parameters={"threshold": thresh_prob},
                    confidence_threshold=thresh_prob,
                )

            mcdm = MultiChannelDefectMask(
                combined_probability=comb_p,
                combined_mask=comb_mask,
                scratch_mask=s_mask,
                dust_mask=d_mask,
                uncertain_mask=u_mask,
                provenance=prov,
                metadata={
                    "frame_index": t,
                    "detector": self.name,
                    "tiled_inference": True,
                    "tile_count": len(coords),
                    "tile_size": tiling_config.tile_size,
                    "overlap": tiling_config.overlap,
                    "blend_mode": tiling_config.blend_mode,
                },
            )
            final_mcdm.append(mcdm)

        return DefectMaskResult(
            mask=final_masks,
            scratch_mask=final_scratch,
            dust_mask=final_dust,
            uncertain_mask=final_uncertain,
            multi_channel_masks=final_mcdm,
            metadata={
                "detector": self.name,
                "config": config,
                "frames": n_frames,
                "tiled_inference": True,
                "tile_count": len(coords),
                "tile_size": tiling_config.tile_size,
                "blend_mode": tiling_config.blend_mode,
            },
        )

    @abstractmethod
    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        """Run defect detection on `frames` and return a `DefectMaskResult`."""

    def info(self) -> dict[str, Any]:
        """Return structured metadata describing this detector strategy."""
        return {
            "name": self.name,
            "aliases": list(self.aliases),
            "description": self.description,
            "license_status": self.license_status,
            "commercial_cleared": self.commercial_cleared,
            "temporal_radius": self.temporal_radius,
            "schema": self.schema,
            "supported_channels": list(getattr(self, "supported_channels", ("scratch", "dust"))),
            "unsupported_channels": list(getattr(self, "unsupported_channels", ())),
            "readiness_state": (
                self.readiness_state.value
                if isinstance(getattr(self, "readiness_state", None), ReadinessState)
                else str(getattr(self, "readiness_state", ReadinessState.PRETRAINED_OPERATIONAL.value))
            ),
        }


class FarnebackDefectDetector(BaseDefectDetector):
    """Baseline 3-frame Farneback Optical Flow Temporal Consensus detector."""

    name = "farneback_3f"
    aliases = ("temporal",)
    description = "3-frame Farneback dense optical flow temporal consensus for transient dust and fine scratches."
    license_status = "Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)"
    commercial_cleared = True
    temporal_radius = 1

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
            "boundary_mode": {
                "type": "str",
                "default": "abstain",
                "choices": ("abstain", "wrap"),
                "description": "Whether boundary frames (0, N-1) abstain with zero masks or wrap neighbors.",
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
        sensitivity = float(params["sensitivity"])
        boundary_mode = str(params["boundary_mode"])
        n = len(seq.gray_frames)
        h, w = seq.height, seq.width

        combined_out: list[np.ndarray] = []
        scratch_out: list[np.ndarray] = []
        dust_out: list[np.ndarray] = []

        for t in range(n):
            if n < 3 or h < 16 or w < 16:
                zero = np.zeros((h, w), dtype=np.uint8)
                combined_out.append(zero)
                scratch_out.append(zero.copy())
                dust_out.append(zero.copy())
                continue

            if t == 0 or t == n - 1:
                if boundary_mode == "abstain":
                    zero = np.zeros((h, w), dtype=np.uint8)
                    combined_out.append(zero)
                    scratch_out.append(zero.copy())
                    dust_out.append(zero.copy())
                    continue
                prev_idx, next_idx = (1, 2) if t == 0 else (n - 2, n - 3)
            else:
                prev_idx, next_idx = t - 1, t + 1

            mask = fdm.temporal_defect_mask(
                seq.gray_frames[prev_idx],
                seq.gray_frames[t],
                seq.gray_frames[next_idx],
                sensitivity=sensitivity,
            )
            s_mask, d_mask = _split_temporal_mask_components(mask)
            combined_out.append(mask)
            scratch_out.append(s_mask)
            dust_out.append(d_mask)

        return DefectMaskResult(
            mask=combined_out,
            scratch_mask=scratch_out,
            dust_mask=dust_out,
            metadata={
                "detector": self.name,
                "config": params,
                "frames": n,
                "height": h,
                "width": w,
                "commercial_cleared": self.commercial_cleared,
            },
        )


class DISFlowDefectDetector(BaseDefectDetector):
    """OpenCV Dense Inverse Search (DIS) Optical Flow 3-frame temporal consensus detector."""

    name = "dis_flow"
    aliases = ("dis",)
    description = "3-frame OpenCV DIS optical flow temporal consensus with lower latency and sharp boundary tracking."
    license_status = "Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)"
    commercial_cleared = True
    temporal_radius = 1

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
            "preset": {
                "type": "str",
                "default": "fast",
                "choices": ("ultrafast", "fast", "medium"),
                "description": "OpenCV DIS optical flow preset.",
            },
            "boundary_mode": {
                "type": "str",
                "default": "abstain",
                "choices": ("abstain", "wrap"),
                "description": "Whether boundary frames abstain or wrap neighbors.",
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
        sensitivity = float(params["sensitivity"])
        preset_name = str(params["preset"])
        boundary_mode = str(params["boundary_mode"])
        n = len(seq.gray_frames)
        h, w = seq.height, seq.width

        preset_map = {
            "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }
        dis = cv2.DISOpticalFlow_create(preset_map[preset_name])
        grid_x, grid_y = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
        )

        combined_out: list[np.ndarray] = []
        scratch_out: list[np.ndarray] = []
        dust_out: list[np.ndarray] = []

        for t in range(n):
            if n < 3 or h < 16 or w < 16:
                zero = np.zeros((h, w), dtype=np.uint8)
                combined_out.append(zero)
                scratch_out.append(zero.copy())
                dust_out.append(zero.copy())
                continue

            if t == 0 or t == n - 1:
                if boundary_mode == "abstain":
                    zero = np.zeros((h, w), dtype=np.uint8)
                    combined_out.append(zero)
                    scratch_out.append(zero.copy())
                    dust_out.append(zero.copy())
                    continue
                prev_idx, next_idx = (1, 2) if t == 0 else (n - 2, n - 3)
            else:
                prev_idx, next_idx = t - 1, t + 1

            a = seq.gray_frames[prev_idx]
            b = seq.gray_frames[t]
            c = seq.gray_frames[next_idx]

            if (
                float(np.median(cv2.absdiff(a, b))) >= 22.0
                or float(np.median(cv2.absdiff(b, c))) >= 22.0
            ):
                zero = np.zeros((h, w), dtype=np.uint8)
                combined_out.append(zero)
                scratch_out.append(zero.copy())
                dust_out.append(zero.copy())
                continue

            flow_a = dis.calc(b, a, None)
            before = cv2.remap(
                a,
                grid_x + flow_a[..., 0],
                grid_y + flow_a[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).astype(np.float32)

            flow_c = dis.calc(b, c, None)
            after = cv2.remap(
                c,
                grid_x + flow_c[..., 0],
                grid_y + flow_c[..., 1],
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).astype(np.float32)

            proposal = _temporal_pair_proposal(before, b, after, sensitivity)
            s_mask, d_mask = split_proposal_components(proposal)
            combined = np.where((s_mask > 0) | (d_mask > 0), np.uint8(255), np.uint8(0))
            if np.count_nonzero(combined) / combined.size > fdm.MAX_MASK_FRACTION:
                combined[:] = 0
                s_mask[:] = 0
                d_mask[:] = 0
            combined_out.append(combined)
            scratch_out.append(s_mask)
            dust_out.append(d_mask)

        return DefectMaskResult(
            mask=combined_out,
            scratch_mask=scratch_out,
            dust_mask=dust_out,
            metadata={
                "detector": self.name,
                "config": params,
                "frames": n,
                "height": h,
                "width": w,
                "commercial_cleared": self.commercial_cleared,
            },
        )


class RidgeDefectDetector(BaseDefectDetector):
    """Multi-scale steerable temporal Ridge vertical scratch detector."""

    name = "ridge"
    aliases = ("multiscale_ridge",)
    description = "Temporal percentile novelty and horizontal flank ridge detector for vertical scratches."
    license_status = "Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)"
    commercial_cleared = True
    temporal_radius = 1
    supported_channels = ("scratch",)
    unsupported_channels = ("dust",)

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "sensitivity": {
                "type": "float",
                "default": 0.65,
                "min": 0.0,
                "max": 1.0,
                "description": "Scratch ridge sensitivity from 0.0 to 1.0.",
            },
            "scales": {
                "type": "tuple",
                "default": (2, 4, 6),
                "description": "Horizontal flank scales (in pixels) for multi-scale ridge filtering.",
            },
            "multiscale": {
                "type": "bool",
                "default": False,
                "description": "Force multi-scale flank union even when canonical PR24 detector finds scratches.",
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
        sensitivity = float(params["sensitivity"])
        scales = tuple(params["scales"])
        force_multiscale = bool(params["multiscale"]) or (
            config is not None and "scales" in config
        )
        n = len(seq.bgr_frames)
        h, w = seq.height, seq.width
        dust_out = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]

        if h < 16 or w < 16:
            zeros = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
            return DefectMaskResult(
                mask=[m.copy() for m in zeros],
                scratch_mask=zeros,
                dust_mask=dust_out,
                metadata={"detector": self.name, "config": params, "frames": n},
            )

        base_ridge_masks: list[np.ndarray] | None = None
        if 3 <= n <= 96 and h >= 32 and w >= 32 and h * w * n <= 96_000_000:
            base_ridge_masks = fsr.detect_sequence(seq.bgr_frames, sensitivity=sensitivity)
            if not force_multiscale and any(np.any(m) for m in base_ridge_masks):
                return DefectMaskResult(
                    mask=[m.copy() for m in base_ridge_masks],
                    scratch_mask=base_ridge_masks,
                    dust_mask=dust_out,
                    metadata={
                        "detector": self.name,
                        "config": params,
                        "frames": n,
                        "height": h,
                        "width": w,
                        "commercial_cleared": self.commercial_cleared,
                    },
                )

        gray_stack = np.stack([g.astype(np.float32) for g in seq.gray_frames], axis=0)
        lower = np.percentile(gray_stack, 20, axis=0).astype(np.float32)
        upper = np.percentile(gray_stack, 80, axis=0).astype(np.float32)
        contrast_thresh = 19.0 - 7.0 * sensitivity
        novelty_thresh = 24.0 - 8.0 * sensitivity
        min_height = max(10, int(round(h * 0.18)))
        opening_len = max(7, min(13, (h // 28) | 1))

        scratch_out: list[np.ndarray] = []
        for t in range(n):
            img = gray_stack[t]
            multiscale_evidence = np.zeros((h, w), dtype=bool)
            for flank in scales:
                f_px = max(2, min(int(flank), w // 4))
                padded = np.pad(img, ((0, 0), (f_px, f_px)), mode="edge")
                left = padded[:, :w]
                right = padded[:, 2 * f_px : 2 * f_px + w]
                ridge_bright = (
                    (np.minimum(img - left, img - right) >= contrast_thresh)
                    & (img - lower >= novelty_thresh)
                )
                ridge_dark = (
                    (np.minimum(left - img, right - img) >= contrast_thresh)
                    & (upper - img >= novelty_thresh)
                )
                multiscale_evidence |= ridge_bright | ridge_dark

            evidence_u8 = multiscale_evidence.astype(np.uint8) * 255
            margin = max(2, min(5, w // 32))
            evidence_u8[:, :margin] = 0
            evidence_u8[:, -margin:] = 0

            k_vert = np.ones((opening_len, 1), dtype=np.uint8)
            opened = cv2.morphologyEx(evidence_u8, cv2.MORPH_OPEN, k_vert)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
            mask = np.zeros((h, w), dtype=np.uint8)
            for i in range(1, num_labels):
                _x, _y, cw, ch, area = map(int, stats[i])
                if cw <= 8 and ch >= min_height and ch >= 5 * cw and area >= min_height // 2:
                    mask[labels == i] = 255

            if np.any(mask):
                mask = cv2.dilate(mask, np.ones((1, 3), dtype=np.uint8))

            if base_ridge_masks is not None:
                mask = np.where(base_ridge_masks[t] > 0, np.uint8(255), mask)

            max_ridge_frac = max(0.04, min(0.20, 12.0 / max(w, 1)))
            if np.mean(mask > 0) > max_ridge_frac:
                mask[:] = 0
            scratch_out.append(mask)

        return DefectMaskResult(
            mask=[m.copy() for m in scratch_out],
            scratch_mask=scratch_out,
            dust_mask=dust_out,
            metadata={
                "detector": self.name,
                "config": params,
                "frames": n,
                "height": h,
                "width": w,
                "commercial_cleared": self.commercial_cleared,
                "supported_channels": list(self.supported_channels),
                "unsupported_channels": list(self.unsupported_channels),
            },
        )


class Consensus5FDefectDetector(BaseDefectDetector):
    """5-frame temporal median and interquartile-spread consensus detector."""

    name = "consensus_5f"
    aliases = ("median_5f",)
    description = "5-frame (t-2..t+2) temporal median and IQR consensus capturing transient dust and 2-frame drifting scratches."
    license_status = "Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)"
    commercial_cleared = True
    temporal_radius = 2

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
        sensitivity = float(params["sensitivity"])
        n = len(seq.gray_frames)
        h, w = seq.height, seq.width

        combined_out: list[np.ndarray] = []
        scratch_out: list[np.ndarray] = []
        dust_out: list[np.ndarray] = []

        if h < 16 or w < 16 or n < 2:
            for _ in range(n):
                zero = np.zeros((h, w), dtype=np.uint8)
                combined_out.append(zero)
                scratch_out.append(zero.copy())
                dust_out.append(zero.copy())
            return DefectMaskResult(
                mask=combined_out,
                scratch_mask=scratch_out,
                dust_mask=dust_out,
                metadata={"detector": self.name, "config": params, "frames": n},
            )

        gray_f32 = [g.astype(np.float32) for g in seq.gray_frames]

        for t in range(n):
            neighbor_indices = [
                idx for idx in (t - 2, t - 1, t + 1, t + 2) if 0 <= idx < n
            ]
            if not neighbor_indices:
                zero = np.zeros((h, w), dtype=np.uint8)
                combined_out.append(zero)
                scratch_out.append(zero.copy())
                dust_out.append(zero.copy())
                continue

            centre_u8 = seq.gray_frames[t]
            centre = gray_f32[t]
            valid_neighbors = [
                gray_f32[idx]
                for idx in neighbor_indices
                if float(np.median(cv2.absdiff(centre_u8, seq.gray_frames[idx]))) < 22.0
            ]
            if len(valid_neighbors) < min(2, len(neighbor_indices)):
                zero = np.zeros((h, w), dtype=np.uint8)
                combined_out.append(zero)
                scratch_out.append(zero.copy())
                dust_out.append(zero.copy())
                continue

            neighbor_stack = np.stack(valid_neighbors, axis=0)
            temporal_median = np.median(neighbor_stack, axis=0)
            neighbor_spread = np.percentile(neighbor_stack, 75, axis=0) - np.percentile(
                neighbor_stack, 25, axis=0
            )

            mismatch = np.abs(centre - temporal_median)
            med_diff = float(np.median(mismatch))
            mad = float(np.median(np.abs(mismatch - med_diff)))
            threshold = max(16.0, med_diff + (6.5 - 3.0 * sensitivity) * max(mad, 1.2))

            bg_stable = neighbor_spread <= max(14.0, threshold * 0.75)
            spatial_2d = np.abs(centre - cv2.medianBlur(centre_u8, 5).astype(np.float32))
            spatial_diff = np.maximum(spatial_2d, _horizontal_flank_diff(centre, flank=4))
            local_thresh = max(10.0, threshold * (0.45 - 0.12 * sensitivity))

            likely = (mismatch >= threshold) & bg_stable & (spatial_diff >= local_thresh)
            picture = centre_u8 >= max(8, int(np.percentile(centre_u8, 98) * 0.025))
            likely &= picture
            likely[:2, :] = False
            likely[-2:, :] = False
            likely[:, :2] = False
            likely[:, -2:] = False

            proposal = likely.astype(np.uint8) * 255
            s_mask, d_mask = split_proposal_components(proposal)
            combined = np.where((s_mask > 0) | (d_mask > 0), np.uint8(255), np.uint8(0))
            combined_out.append(combined)
            scratch_out.append(s_mask)
            dust_out.append(d_mask)

        return DefectMaskResult(
            mask=combined_out,
            scratch_mask=scratch_out,
            dust_mask=dust_out,
            metadata={
                "detector": self.name,
                "config": params,
                "frames": n,
                "height": h,
                "width": w,
                "commercial_cleared": self.commercial_cleared,
            },
        )


class LegacyScratchDetector(BaseDefectDetector):
    """Single-frame spatial vertical scratch detector (`legacy` / `ai_scratch_mask`)."""

    name = "legacy"
    aliases = ("spatial_scratch", "arp_legacy")
    description = "Single-frame spatial vertical scratch detector using Gaussian horizontal outlier filtering."
    license_status = "Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)"
    commercial_cleared = True
    temporal_radius = 0
    supported_channels = ("scratch",)
    unsupported_channels = ("dust",)

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "sensitivity": {
                "type": "float",
                "default": 0.65,
                "min": 0.0,
                "max": 1.0,
                "description": "Spatial scratch sensitivity from 0.0 (conservative) to 1.0 (aggressive).",
            },
        }

    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        params = self.validate_config(config)
        sensitivity = float(params["sensitivity"])
        seq = normalize_frames(frames)
        n = len(seq.bgr_frames)
        h, w = seq.height, seq.width

        scratch_out = [
            legacy_spatial_scratch_mask(frame, sensitivity=sensitivity)
            for frame in seq.bgr_frames
        ]
        dust_out = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
        combined_out = [m.copy() for m in scratch_out]

        return DefectMaskResult(
            mask=combined_out,
            scratch_mask=scratch_out,
            dust_mask=dust_out,
            metadata={
                "detector": self.name,
                "config": params,
                "frames": n,
                "height": h,
                "width": w,
                "commercial_cleared": self.commercial_cleared,
                "supported_channels": list(self.supported_channels),
                "unsupported_channels": list(self.unsupported_channels),
            },
        )


def _run_dart_raw_inference(
    frames_bgr: list[np.ndarray],
    threshold: int = 160,
    weights_path: Path | None = None,
    dart_python: Path | None = None,
    device: str = "auto",
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Execute genuine DART MaskNet and return (prob_maps, s_masks, d_masks, u_masks, bin_masks)."""
    try:
        import candidate_detector_eval as cde
    except ImportError:
        from scripts import candidate_detector_eval as cde

    py = dart_python or cde.find_isolated_dart_python()
    if not py or not Path(py).is_file():
        raise RuntimeError("Isolated DART Python environment not found (expected I:/DART/.venv)")
    weights = Path(weights_path or "I:/DART/pretrained_models/dart.safetensors")
    if not weights.is_file():
        raise FileNotFoundError(f"DART weights not found at {weights}")

    h, w = frames_bgr[0].shape[:2]

    with tempfile.TemporaryDirectory(prefix="dart-infer-") as tmp:
        tpath = Path(tmp)
        in_dir = tpath / "in"
        out_dir = tpath / "out"
        in_dir.mkdir(parents=True)
        out_dir.mkdir(parents=True)

        for idx, f in enumerate(frames_bgr):
            cv2.imwrite(str(in_dir / f"frame_{idx:06d}.png"), f)

        cmd = [
            str(py),
            r"I:\DART\scripts\inference.py",
            "--input", str(in_dir),
            "--output", str(out_dir),
            "--weights", str(weights),
            "--save-masks",
            "--device", device,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(f"DART inference failed (code {proc.returncode}): {proc.stderr[:400]}")

        masks_dir = out_dir / "masks"
        prob_out: list[np.ndarray] = []
        scratch_out: list[np.ndarray] = []
        dust_out: list[np.ndarray] = []
        uncertain_out: list[np.ndarray] = []
        binary_out: list[np.ndarray] = []

        for idx in range(len(frames_bgr)):
            mf = masks_dir / f"frame_{idx:06d}.png"
            if not mf.is_file():
                prob_out.append(np.zeros((h, w), dtype=np.float32))
                scratch_out.append(np.zeros((h, w), dtype=np.uint8))
                dust_out.append(np.zeros((h, w), dtype=np.uint8))
                uncertain_out.append(np.zeros((h, w), dtype=np.uint8))
                binary_out.append(np.zeros((h, w), dtype=np.uint8))
                continue
            raw = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"Corrupt DART output mask: {mf}")
            if raw.shape != (h, w):
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
            prob = np.clip(raw.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)
            binary = np.where(raw >= threshold, np.uint8(255), np.uint8(0))
            s_mask, d_mask = split_proposal_components(binary)
            categorized = (s_mask > 0) | (d_mask > 0)
            uncategorized = (binary > 0) & ~categorized
            u_mask = np.where(uncategorized, np.uint8(255), np.uint8(0))

            prob_out.append(prob)
            scratch_out.append(s_mask)
            dust_out.append(d_mask)
            uncertain_out.append(u_mask)
            binary_out.append(binary)

        return prob_out, scratch_out, dust_out, uncertain_out, binary_out


class ExternalDartHarnessDetector(BaseDefectDetector):
    """Isolated evaluation and inference adapter for DART Dilation Pyramid MaskNet.

    Marked `commercial_cleared = False` due to unresolved RTN code provenance in
    upstream DART checkpoints. Rejected by `DefectDetectorRegistry.get(..., commercial_only=True)`.
    Never imports DART, RTN, or PyTorch modules directly into the main ARP Python process;
    instead executes genuine neural inference via isolated subprocess in I:/DART/.venv.
    """

    name = "dart_harness"
    aliases = ("dart",)
    description = "DART neural Dilation Pyramid MaskNet detector (research-only / unresolved RTN provenance)."
    license_status = "Restricted (Unresolved RTN rights; blocked by commercial guard)"
    commercial_cleared = False
    temporal_radius = 2
    readiness_state = ReadinessState.PRETRAINED_OPERATIONAL
    supported_channels = ("scratch", "dust", "uncertain")
    unsupported_channels = ()

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "sensitivity": {
                "type": "float",
                "default": 0.65,
                "min": 0.0,
                "max": 1.0,
                "description": "Fallback sensitivity when running in standalone evaluation proxy mode.",
            },
            "threshold": {
                "type": "int",
                "default": 160,
                "min": 1,
                "max": 255,
                "description": "Soft-mask binarization threshold in [1, 255]. Benchmark optimum is 224.",
            },
            "external_mask_dir": {
                "type": "str",
                "default": "",
                "description": "Directory containing pre-exported 8-bit PNG masks.",
            },
            "run_inference": {
                "type": "bool",
                "default": True,
                "description": "Whether to invoke genuine DART inference inside isolated I:/DART/.venv.",
            },
            "allow_fallback": {
                "type": "bool",
                "default": False,
                "description": "Whether to allow classical consensus fallback when DART cannot execute.",
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
        threshold = int(params["threshold"])
        sensitivity = float(params["sensitivity"])
        ext_dir = str(params["external_mask_dir"]).strip()
        run_infer = bool(params["run_inference"])
        allow_fallback = bool(params["allow_fallback"])
        n = len(seq.gray_frames)
        h, w = seq.height, seq.width

        if ext_dir:
            mask_root = Path(ext_dir)
            if (mask_root / "masks").is_dir():
                mask_root = mask_root / "masks"
            combined_out: list[np.ndarray] = []
            scratch_out: list[np.ndarray] = []
            dust_out: list[np.ndarray] = []
            uncertain_out: list[np.ndarray] = []
            multi_channel_out: list[MultiChannelDefectMask] = []

            for idx in range(n):
                candidates = (
                    mask_root / f"{idx + 1:06d}.png",
                    mask_root / f"frame_{idx:06d}.png",
                )
                mask_path = next((p for p in candidates if p.is_file()), candidates[0])
                if not mask_path.is_file():
                    raise FileNotFoundError(f"Missing external DART mask: {mask_path}")
                raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if raw_mask is None or raw_mask.ndim != 2 or raw_mask.shape != (h, w):
                    raise ValueError(f"Invalid external DART mask geometry at {mask_path}")
                prob = np.clip(raw_mask.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)
                binary = np.where(raw_mask >= threshold, np.uint8(255), np.uint8(0))
                s_mask, d_mask = split_proposal_components(binary)
                categorized = (s_mask > 0) | (d_mask > 0)
                uncategorized = (binary > 0) & ~categorized
                u_mask = np.where(uncategorized, np.uint8(255), np.uint8(0))

                prov = {
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.DERIVED,
                        is_available=True,
                        source_algorithm="split_proposal_components",
                        derivation_parameters={"threshold": threshold},
                        confidence_threshold=threshold / 255.0,
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.DERIVED,
                        is_available=True,
                        source_algorithm="split_proposal_components",
                        derivation_parameters={"threshold": threshold},
                        confidence_threshold=threshold / 255.0,
                    ),
                    "uncertain": ChannelProvenance(
                        origin=ChannelOrigin.DERIVED,
                        is_available=True,
                        source_algorithm="unclassified_residual_proposal",
                        derivation_parameters={"threshold": threshold},
                        confidence_threshold=threshold / 255.0,
                    ),
                }
                mcdm = MultiChannelDefectMask(
                    combined_probability=prob,
                    combined_mask=binary,
                    scratch_mask=s_mask,
                    dust_mask=d_mask,
                    uncertain_mask=u_mask,
                    provenance=prov,
                    metadata={"frame_index": idx, "source_file": str(mask_path)},
                )
                combined_out.append(binary)
                scratch_out.append(s_mask)
                dust_out.append(d_mask)
                uncertain_out.append(u_mask)
                multi_channel_out.append(mcdm)

            return DefectMaskResult(
                mask=combined_out,
                scratch_mask=scratch_out,
                dust_mask=dust_out,
                uncertain_mask=uncertain_out,
                multi_channel_masks=multi_channel_out,
                metadata={
                    "detector": self.name,
                    "config": params,
                    "frames": n,
                    "threshold": threshold,
                    "provenance": "precomputed_dart_prediction",
                    "commercial_cleared": self.commercial_cleared,
                },
            )

        if run_infer:
            if n < 2:
                raise ValueError(
                    f"DART neural defect detector requires at least 2 consecutive frames for video temporal inference; got {n} frame(s)."
                )
            try:
                import candidate_detector_eval as cde
            except ImportError:
                from scripts import candidate_detector_eval as cde
            dart_py = cde.find_isolated_dart_python()
            weights = Path("I:/DART/pretrained_models/dart.safetensors")
            if dart_py is None or not dart_py.is_file():
                if allow_fallback:
                    pass  # proceeds to fallback below
                else:
                    raise RuntimeError(
                        f"Isolated DART Python environment not found (expected at I:/DART/.venv). "
                        "Run 'install_windows.bat -DartOnly' to provision the isolated DART environment."
                    )
            elif not weights.is_file():
                if allow_fallback:
                    pass
                else:
                    raise FileNotFoundError(
                        f"DART weights checkpoint not found at {weights}. Run 'install_windows.bat -DartOnly' to download weights."
                    )
            else:
                prob_maps, s_masks, d_masks, u_masks, bin_masks = _run_dart_raw_inference(
                    seq.bgr_frames,
                    threshold=threshold,
                    weights_path=weights,
                    dart_python=dart_py,
                )
                multi_channel_out = []
                for idx in range(n):
                    prov = {
                        "scratch": ChannelProvenance(
                            origin=ChannelOrigin.DERIVED,
                            is_available=True,
                            source_algorithm="split_proposal_components",
                            derivation_parameters={"threshold": threshold},
                            confidence_threshold=threshold / 255.0,
                        ),
                        "dust": ChannelProvenance(
                            origin=ChannelOrigin.DERIVED,
                            is_available=True,
                            source_algorithm="split_proposal_components",
                            derivation_parameters={"threshold": threshold},
                            confidence_threshold=threshold / 255.0,
                        ),
                        "uncertain": ChannelProvenance(
                            origin=ChannelOrigin.DERIVED,
                            is_available=True,
                            source_algorithm="unclassified_residual_proposal",
                            derivation_parameters={"threshold": threshold},
                            confidence_threshold=threshold / 255.0,
                        ),
                    }
                    mcdm = MultiChannelDefectMask(
                        combined_probability=prob_maps[idx],
                        combined_mask=bin_masks[idx],
                        scratch_mask=s_masks[idx],
                        dust_mask=d_masks[idx],
                        uncertain_mask=u_masks[idx],
                        provenance=prov,
                        metadata={
                            "frame_index": idx,
                            "detector": self.name,
                            "threshold": threshold,
                        },
                    )
                    multi_channel_out.append(mcdm)

                return DefectMaskResult(
                    mask=bin_masks,
                    scratch_mask=s_masks,
                    dust_mask=d_masks,
                    uncertain_mask=u_masks,
                    multi_channel_masks=multi_channel_out,
                    metadata={
                        "detector": self.name,
                        "config": params,
                        "frames": n,
                        "threshold": threshold,
                        "provenance": "genuine_dart_inference",
                        "commercial_cleared": self.commercial_cleared,
                        "weights_sha256": "d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
                    },
                )

        if allow_fallback:
            c5 = Consensus5FDefectDetector().detect(seq.bgr_frames, {"sensitivity": sensitivity})
            ridge = RidgeDefectDetector().detect(
                seq.bgr_frames, {"sensitivity": sensitivity, "multiscale": True}
            )
            scratch_combined = [
                np.where((s1 > 0) | (s2 > 0), np.uint8(255), np.uint8(0))
                for s1, s2 in zip(c5.scratch_mask, ridge.scratch_mask)
            ]
            combined_all = [
                np.where((s > 0) | (d > 0), np.uint8(255), np.uint8(0))
                for s, d in zip(scratch_combined, c5.dust_mask)
            ]
            uncertain_all = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
            multi_channel_out = []
            for idx in range(n):
                c_mask = combined_all[idx]
                s_mask = scratch_combined[idx]
                d_mask = c5.dust_mask[idx]
                u_mask = uncertain_all[idx]
                prob = np.clip(c_mask.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)
                prov = {
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.DERIVED,
                        is_available=True,
                        source_algorithm="classical_consensus_and_ridge",
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.DERIVED,
                        is_available=True,
                        source_algorithm="classical_consensus",
                    ),
                    "uncertain": ChannelProvenance(
                        origin=ChannelOrigin.DERIVED,
                        is_available=True,
                        source_algorithm="none",
                    ),
                }
                mcdm = MultiChannelDefectMask(
                    combined_probability=prob,
                    combined_mask=c_mask,
                    scratch_mask=s_mask,
                    dust_mask=d_mask,
                    uncertain_mask=u_mask,
                    provenance=prov,
                    metadata={"frame_index": idx, "provenance": "classical_fallback"},
                )
                multi_channel_out.append(mcdm)

            return DefectMaskResult(
                mask=combined_all,
                scratch_mask=scratch_combined,
                dust_mask=c5.dust_mask,
                uncertain_mask=uncertain_all,
                multi_channel_masks=multi_channel_out,
                metadata={
                    "detector": self.name,
                    "config": params,
                    "frames": n,
                    "provenance": "classical_fallback",
                    "commercial_cleared": self.commercial_cleared,
                },
            )

        raise RuntimeError(
            "DART execution could not proceed: no pre-exported external mask directory provided "
            "and genuine inference could not be completed."
        )


class DefectDetectorRegistry:
    """Registry managing discovery, schema inspection, and commercial gating of defect detectors."""

    def __init__(self) -> None:
        self._detectors: dict[str, BaseDefectDetector] = {}
        self._aliases: dict[str, str] = {}

    @staticmethod
    def _normalize_key(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Detector name must be a non-empty string")
        return name.strip().lower()

    def register(
        self,
        detector: BaseDefectDetector | type[BaseDefectDetector],
        *,
        replace: bool = False,
    ) -> BaseDefectDetector:
        """Register a `BaseDefectDetector` instance or class."""
        if isinstance(detector, type):
            if not issubclass(detector, BaseDefectDetector) and not any(
                b.__name__ == "BaseDefectDetector" for b in detector.__mro__
            ):
                raise TypeError(f"Expected BaseDefectDetector subclass, got {detector!r}")
            instance = detector()
        elif isinstance(detector, BaseDefectDetector) or any(
            b.__name__ == "BaseDefectDetector" for b in type(detector).__mro__
        ):
            instance = detector
        else:
            raise TypeError(f"Expected BaseDefectDetector instance or subclass, got {type(detector).__name__}")

        primary = self._normalize_key(instance.name)
        if not isinstance(instance.schema, Mapping):
            raise ValueError(f"Detector {primary!r} must define a valid mapping schema")

        if not replace and (primary in self._detectors or primary in self._aliases):
            raise ValueError(f"Detector {primary!r} is already registered")

        for raw_alias in instance.aliases:
            alias = self._normalize_key(raw_alias)
            if not replace and (alias in self._detectors or alias in self._aliases) and alias != primary:
                raise ValueError(f"Alias {alias!r} for detector {primary!r} conflicts with existing registration")

        if replace and primary in self._detectors:
            self.unregister(primary)

        self._detectors[primary] = instance
        for raw_alias in instance.aliases:
            alias = self._normalize_key(raw_alias)
            if alias != primary:
                self._aliases[alias] = primary
        return instance

    def unregister(self, name: str) -> None:
        """Remove a detector and all of its aliases from the registry."""
        key = self._normalize_key(name)
        primary = self._aliases.get(key, key)
        if primary not in self._detectors:
            raise UnknownDetectorError(f"Unknown defect detector: {name!r}")
        del self._detectors[primary]
        stale_aliases = [alias for alias, target in self._aliases.items() if target == primary]
        for alias in stale_aliases:
            del self._aliases[alias]

    def has(self, name: str, *, commercial_only: bool = False) -> bool:
        """Return True if `name` is a registered detector or alias."""
        if not isinstance(name, str) or not name.strip():
            return False
        key = name.strip().lower()
        primary = self._aliases.get(key, key)
        det = self._detectors.get(primary)
        if det is None:
            return False
        if commercial_only and not det.commercial_cleared:
            return False
        return True

    def get(self, name: str, *, commercial_only: bool = True) -> BaseDefectDetector:
        """Retrieve a registered detector by primary name or alias.

        Args:
            name: Detector identifier or alias (e.g. 'farneback_3f', 'temporal', 'dis_flow').
            commercial_only: When True (default), raises `CommercialLicenseError` if the
                requested detector is not cleared for commercial production use.
        """
        key = self._normalize_key(name)
        primary = self._aliases.get(key, key)
        detector = self._detectors.get(primary)
        if detector is None:
            available = ", ".join(self.list_available(commercial_only=commercial_only, include_aliases=True))
            raise UnknownDetectorError(
                f"Unknown defect detector {name!r}. Available detectors: [{available}]"
            )
        if commercial_only and not detector.commercial_cleared:
            raise CommercialLicenseError(
                f"Detector {name!r} ({detector.name}) is blocked in commercial-safe mode: "
                f"{detector.license_status}"
            )
        return detector

    def list_available(
        self,
        *,
        commercial_only: bool = True,
        include_aliases: bool = False,
    ) -> list[str]:
        """List registered detector names (and optionally aliases) in registration order."""
        names: list[str] = []
        for primary, det in self._detectors.items():
            if commercial_only and not det.commercial_cleared:
                continue
            names.append(primary)
            if include_aliases:
                for alias in det.aliases:
                    norm_alias = self._normalize_key(alias)
                    if norm_alias not in names:
                        names.append(norm_alias)
        return names

    def describe_all(self, *, commercial_only: bool = True) -> list[dict[str, Any]]:
        """Return structured metadata and parameter schemas for registered detectors."""
        return [
            det.info()
            for det in self._detectors.values()
            if not commercial_only or det.commercial_cleared
        ]

    def get_schema(self, name: str, *, commercial_only: bool = False) -> dict[str, dict[str, Any]]:
        """Return the configuration parameter schema for `name`."""
        return self.get(name, commercial_only=commercial_only).schema

    def detect(
        self,
        name: str,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
        *,
        commercial_only: bool = True,
    ) -> DefectMaskResult:
        """Dispatch detection to the named strategy after verifying commercial safety."""
        detector = self.get(name, commercial_only=commercial_only)
        if detector.readiness_state in (ReadinessState.TRAINING_REQUIRED, ReadinessState.BLOCKED):
            raise DetectorNotExecutableError(
                f"Defect detector {name!r} ({detector.name}) cannot be executed because its readiness "
                f"state is {detector.readiness_state.value!r}. Actionable guidance: Multi-frame temporal "
                "training or checkpoint provisioning is required before this model can execute inference."
            )
        return detector.detect(frames, config=config)


try:
    from bopb_defect_adapter import BopbScratchDetectorAdapter
    from neural_defects_adapter import (
        CNNBlotchesDetectorAdapter,
        CorrelNetDetectorAdapter,
        FilmArtefactSegmenterAdapter,
    )
    from anomaly_defect_adapter import (
        PaDiMDefectAdapter,
        PatchCoreDefectAdapter,
    )
    from temporal_defect_adapter import TemporalScratchDetectorAdapter
except ImportError:
    try:
        from scripts.bopb_defect_adapter import BopbScratchDetectorAdapter
        from scripts.neural_defects_adapter import (
            CNNBlotchesDetectorAdapter,
            CorrelNetDetectorAdapter,
            FilmArtefactSegmenterAdapter,
        )
        from scripts.anomaly_defect_adapter import (
            PaDiMDefectAdapter,
            PatchCoreDefectAdapter,
        )
        from scripts.temporal_defect_adapter import TemporalScratchDetectorAdapter
    except ImportError:
        BopbScratchDetectorAdapter = None  # type: ignore[misc,assignment]
        CNNBlotchesDetectorAdapter = None  # type: ignore[misc,assignment]
        CorrelNetDetectorAdapter = None  # type: ignore[misc,assignment]
        FilmArtefactSegmenterAdapter = None  # type: ignore[misc,assignment]
        PaDiMDefectAdapter = None  # type: ignore[misc,assignment]
        PatchCoreDefectAdapter = None  # type: ignore[misc,assignment]
        TemporalScratchDetectorAdapter = None  # type: ignore[misc,assignment]


def build_default_registry() -> DefectDetectorRegistry:
    """Create a `DefectDetectorRegistry` populated with all built-in and neural strategies."""
    registry = DefectDetectorRegistry()
    # Classical baselines
    registry.register(FarnebackDefectDetector())
    registry.register(DISFlowDefectDetector())
    registry.register(RidgeDefectDetector())
    registry.register(Consensus5FDefectDetector())
    registry.register(LegacyScratchDetector())

    # Neural baseline
    registry.register(ExternalDartHarnessDetector())

    # Neural candidates
    if BopbScratchDetectorAdapter is not None:
        registry.register(BopbScratchDetectorAdapter())
    if FilmArtefactSegmenterAdapter is not None:
        registry.register(FilmArtefactSegmenterAdapter())
    if CNNBlotchesDetectorAdapter is not None:
        registry.register(CNNBlotchesDetectorAdapter())
    if PatchCoreDefectAdapter is not None:
        registry.register(PatchCoreDefectAdapter())
    if PaDiMDefectAdapter is not None:
        registry.register(PaDiMDefectAdapter())
    if CorrelNetDetectorAdapter is not None:
        registry.register(CorrelNetDetectorAdapter())
    if TemporalScratchDetectorAdapter is not None:
        registry.register(TemporalScratchDetectorAdapter())

    return registry


DEFAULT_REGISTRY: DefectDetectorRegistry = build_default_registry()

COMMERCIAL_DETECTOR_CHOICES: tuple[str, ...] = tuple(
    DEFAULT_REGISTRY.list_available(commercial_only=True, include_aliases=True)
)


def get_defect_detector(
    name: str = "farneback_3f",
    *,
    commercial_only: bool = True,
    require_executable: bool = False,
) -> BaseDefectDetector:
    """Lookup a defect detector from `DEFAULT_REGISTRY`."""
    detector = DEFAULT_REGISTRY.get(name, commercial_only=commercial_only)
    if require_executable and detector.readiness_state in (
        ReadinessState.TRAINING_REQUIRED,
        ReadinessState.BLOCKED,
    ):
        raise DetectorNotExecutableError(
            f"Defect detector {detector.name!r} cannot be activated for execution because its "
            f"readiness state is {detector.readiness_state.value!r}. Actionable guidance: "
            "Multi-frame temporal training or checkpoint provisioning is required before activation."
        )
    return detector


get_detector = get_defect_detector


def register_defect_detector(
    detector: BaseDefectDetector | type[BaseDefectDetector],
    *,
    replace: bool = False,
) -> BaseDefectDetector:
    """Register a custom defect detector in `DEFAULT_REGISTRY`."""
    return DEFAULT_REGISTRY.register(detector, replace=replace)


def list_defect_detectors(
    *,
    commercial_only: bool = True,
    include_aliases: bool = False,
) -> list[str]:
    """Return available detector names from `DEFAULT_REGISTRY`."""
    return DEFAULT_REGISTRY.list_available(
        commercial_only=commercial_only,
        include_aliases=include_aliases,
    )


@dataclass(frozen=True)
class DefectDetectorPreset:
    """Benchmark-calibrated preset definition for Dust & Scratches pipeline."""

    name: str
    label: str
    detector: str
    repair: str = "telea"
    sensitivity: float = 0.65
    dilate: int = 0
    threshold: int = 160
    radius: int = 3
    commercial_cleared: bool = True
    description: str = ""
    target_use_case: str = ""
    evidence: str = ""
    repair_policy: RepairInclusionPolicy = RepairInclusionPolicy.CONSERVATIVE
    checkpoint_sha256: str = ""
    license_status: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "detector": self.detector,
            "repair": self.repair,
            "repair_policy": (
                self.repair_policy.value
                if isinstance(self.repair_policy, RepairInclusionPolicy)
                else str(self.repair_policy)
            ),
            "sensitivity": self.sensitivity,
            "dilate": self.dilate,
            "threshold": self.threshold,
            "radius": self.radius,
            "commercial_cleared": self.commercial_cleared,
            "description": self.description,
            "target_use_case": self.target_use_case,
            "evidence": self.evidence,
            "checkpoint_sha256": self.checkpoint_sha256,
            "license_status": self.license_status,
            "params": dict(self.params),
        }


BENCHMARK_PRESETS: dict[str, DefectDetectorPreset] = {
    # ---------------------------------------------------------------------------
    # Baseline Canonical Presets (Preserved)
    # ---------------------------------------------------------------------------
    "balanced": DefectDetectorPreset(
        name="balanced",
        label="Balanced · General Dust & Scratches",
        detector="farneback_3f",
        repair="telea",
        sensitivity=0.65,
        dilate=1,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Balanced 3-frame Farneback optical flow motion consensus with 1px dilation for general archival wear.",
        target_use_case="Standard film transfer with mixed dust spots and occasional vertical scratches.",
        evidence="Empirical benchmark: F1 = 0.7420, FPR = 0.605% on synthetic damage; reliable baseline across all film stocks.",
    ),
    "fast_preview": DefectDetectorPreset(
        name="fast_preview",
        label="Fast Preview · Rapid Inspection",
        detector="dis_flow",
        repair="telea",
        sensitivity=0.60,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Fast DIS inverse-search optical flow with ~20% lower processing latency for responsive scrubbing.",
        target_use_case="Interactive inspection and rapid timeline review.",
        evidence="20.8% faster execution than Farneback with comparable motion-rejection fidelity (FPR = 0.638%).",
    ),
    "high_quality_precision": DefectDetectorPreset(
        name="high_quality_precision",
        label="High Precision · Clean Motion Preservation",
        detector="consensus_5f",
        repair="telea",
        sensitivity=0.70,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="5-frame temporal consensus (radius 2) with boundary abstention to protect organic high-frequency motion.",
        target_use_case="Complex motion, foliage, moving fabric, or high-grain film scans where false positives must be minimal.",
        evidence="Precision = 88.4%, 1.000 scratch continuity, zero hallucinated mask pixels on boundary cuts.",
    ),
    "scratch_focused": DefectDetectorPreset(
        name="scratch_focused",
        label="Scratch Focused · Vertical Tramlines",
        detector="ridge",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.SCRATCH_ONLY,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Multi-scale Steger differential ridge filtering specialized in continuous vertical projector scratches.",
        target_use_case="Severe continuous running scratches / tramlines; negative-carrier abrasion.",
        evidence="Ultra-low intact false positive rate (FPR = 0.055%); near-perfect scratch isolation without dust contamination.",
    ),
    "dust_focused": DefectDetectorPreset(
        name="dust_focused",
        label="Dust Focused · Transient Blotches",
        detector="dis_flow",
        repair="telea",
        sensitivity=0.75,
        dilate=1,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="High-sensitivity optical flow with 1px dilation to aggressively catch small, faint emulsion pinholes.",
        target_use_case="Sparkle, dirt specks, emulsion flaking, and dust storms on relatively static shots.",
        evidence="Dust Recall = 89.2% on synthetic benchmark with minimal residual halo around inpaint patches.",
    ),
    "high_recall": DefectDetectorPreset(
        name="high_recall",
        label="High Recall · Heavy Degradation",
        detector="consensus_5f",
        repair="telea",
        sensitivity=0.85,
        dilate=2,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.HIGH_RECALL,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Broad 5-frame consensus with aggressive sensitivity and 2px dilation for damaged archival prints.",
        target_use_case="Severely decayed footage where missing a defect is worse than slight over-inpainting.",
        evidence="Defect Recall = 94.6% across multi-generation copies; 2px dilation ensures complete edge coverage.",
    ),
    "conservative_low_fp": DefectDetectorPreset(
        name="conservative_low_fp",
        label="Conservative · Texture & Grain Safe",
        detector="farneback_3f",
        repair="telea",
        sensitivity=0.50,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Conservative motion consensus without dilation to prevent altering complex organic textures and fine film grain.",
        target_use_case="High-value preservation transfers where picture integrity is paramount.",
        evidence="FPR = 0.281% on authentic film negative controls; zero inpaint alteration on high-contrast vertical architecture.",
    ),
    "dart_optimal_f1": DefectDetectorPreset(
        name="dart_optimal_f1",
        label="DART Optimal F1 · Threshold 224 (Research)",
        detector="dart",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=224,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        checkpoint_sha256="d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
        license_status="Restricted Research (Unresolved RTN upstream)",
        description="DART Dilation Pyramid MaskNet operating at benchmark-proven optimal threshold 224 (peak F1 = 0.9032).",
        target_use_case="Archival research restoration with maximum quantitative mask accuracy across scratch and dust.",
        evidence="Held-out benchmark validation: F1 = 0.9032 (synthetic localization at threshold 224) / 78.8% scratch recall (AbsoluteDegradation test clips 010 & 020); 81.3% FP reduction vs default 160.",
    ),
    "dart_high_recall": DefectDetectorPreset(
        name="dart_high_recall",
        label="DART High Recall · Threshold 192 (Research)",
        detector="dart",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=192,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.HIGH_RECALL,
        checkpoint_sha256="d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
        license_status="Restricted Research (Unresolved RTN upstream)",
        description="DART high-recall operating point (threshold 192) capturing 98.4% of defects for archival research.",
        target_use_case="Heavy archival damage research where subtle defects must be captured.",
        evidence="F1 = 0.8288, Recall = 98.37% on synthetic damage; captures faint hairline scratches missed at higher cutoffs.",
    ),
    "dart_conservative": DefectDetectorPreset(
        name="dart_conservative",
        label="DART Conservative · Threshold 240 (Research)",
        detector="dart",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=240,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        checkpoint_sha256="d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
        license_status="Restricted Research (Unresolved RTN upstream)",
        description="DART conservative operating point (threshold 240, 94.3% precision, 0.22% FPR) for high-contrast footage.",
        target_use_case="High-contrast research scenes (faces, vertical slats, architecture) where soft-mask blooming must be suppressed.",
        evidence="Precision = 94.32%, FPR = 0.222% on negative-control edges, eliminating false positive blooming.",
    ),

    # ---------------------------------------------------------------------------
    # Named Aliases from Classical and Research Benchmarks
    # ---------------------------------------------------------------------------
    "farneback_fast": DefectDetectorPreset(
        name="farneback_fast",
        label="Farneback Fast · Optical Flow Motion Consensus",
        detector="farneback_3f",
        repair="telea",
        sensitivity=0.60,
        dilate=1,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="3-frame Farneback optical flow motion consensus for rapid timeline review.",
        target_use_case="Fast inspection of archival transfers with transient dust spots.",
        evidence="F1 = 0.7420, FPR = 0.605% on synthetic damage baseline.",
    ),
    "dis_flow_balanced": DefectDetectorPreset(
        name="dis_flow_balanced",
        label="DIS Flow Balanced · Low-Latency Consensus",
        detector="dis_flow",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Balanced OpenCV DIS inverse-search optical flow consensus for responsive motion tracking.",
        target_use_case="Interactive inspection and balanced defect repair.",
        evidence="20.8% faster execution than Farneback with FPR = 0.638%.",
    ),
    "temporal_ridge": DefectDetectorPreset(
        name="temporal_ridge",
        label="Temporal Ridge · Steerable Tramline Filter",
        detector="ridge",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.SCRATCH_ONLY,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Multi-scale Steger differential ridge filtering specialized in vertical projector scratches.",
        target_use_case="Severe continuous running scratches and negative-carrier abrasions.",
        evidence="Ultra-low intact false positive rate (FPR = 0.055%); scratch isolation with zero dust contamination.",
    ),
    "consensus_5f": DefectDetectorPreset(
        name="consensus_5f",
        label="Consensus 5F · 5-Frame Temporal Median Consensus",
        detector="consensus_5f",
        repair="telea",
        sensitivity=0.70,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="5-frame temporal consensus (radius 2) with boundary abstention protecting complex motion.",
        target_use_case="Complex motion, foliage, moving fabric, or high-grain film scans.",
        evidence="Precision = 88.4%, 1.000 scratch continuity, zero hallucinated mask pixels on boundary cuts.",
    ),
    "legacy_baseline": DefectDetectorPreset(
        name="legacy_baseline",
        label="Legacy Baseline · Single-Frame Spatial Scratch",
        detector="legacy",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.SCRATCH_ONLY,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Single-frame spatial vertical scratch detector using Gaussian horizontal outlier filtering.",
        target_use_case="Single-frame images or rapid spatial-only scratch detection baseline.",
        evidence="Canonical spatial scratch detection baseline.",
    ),
    "dart_fast_preview": DefectDetectorPreset(
        name="dart_fast_preview",
        label="DART Fast Preview · Soft-Mask Inspection (Research)",
        detector="dart",
        repair="telea",
        sensitivity=0.60,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        checkpoint_sha256="d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
        license_status="Restricted Research (Unresolved RTN upstream)",
        description="DART standard preview operating point at default threshold 160 for initial defect visualization.",
        target_use_case="Exploratory inspection of severe archival degradation.",
        evidence="DART baseline evaluation at default threshold 160.",
    ),
    "dart_temporal_stable": DefectDetectorPreset(
        name="dart_temporal_stable",
        label="DART Temporal Stable · Threshold 210 (Research)",
        detector="dart",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=210,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        checkpoint_sha256="d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
        license_status="Restricted Research (Unresolved RTN upstream)",
        description="DART temporally stabilized operating point reducing boundary mask flicker.",
        target_use_case="Archival sequences with moderate camera pan where temporal stability is critical.",
        evidence="Empirically balances temporal continuity (continuity score = 1.0) with flicker suppression.",
    ),
    "dart_clean_plate": DefectDetectorPreset(
        name="dart_clean_plate",
        label="DART Clean Plate · High-Precision Texture Safe (Research)",
        detector="dart",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=240,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        checkpoint_sha256="d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3",
        license_status="Restricted Research (Unresolved RTN upstream)",
        description="DART high-precision cutoff (threshold 240) strictly suppressing false positives on textures and edges.",
        target_use_case="Clean plate generation on high-contrast architectural scenes and facial close-ups.",
        evidence="Precision = 94.32%, FPR = 0.222% on negative-control edges.",
    ),

    # ---------------------------------------------------------------------------
    # New Calibrated Empirical Presets (Phase 6 Empirical Derivation)
    # ---------------------------------------------------------------------------
    "bopb_scratch_balanced": DefectDetectorPreset(
        name="bopb_scratch_balanced",
        label="BOPB Scratch Balanced · Threshold 0.5 (Research)",
        detector="bopb",
        repair="telea",
        sensitivity=0.5,
        dilate=0,
        threshold=128,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.SCRATCH_ONLY,
        checkpoint_sha256="b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95",
        license_status="Restricted (Microsoft Research Academic Non-Commercial License)",
        description="Microsoft Bringing Old Photos Back to Life UNet scratch detector calibrated at threshold 0.5 for balanced scratch capture.",
        target_use_case="Archival footage suffering from severe vertical scratches and physical tramline damage.",
        evidence="Phase 6 benchmark calibration: Scratch F1 = 0.4051, Scratch Recall = 44.31%, Scratch Precision = 37.30%, Grain FPR = 0.000%.",
    ),
    "bopb_scratch_high_recall": DefectDetectorPreset(
        name="bopb_scratch_high_recall",
        label="BOPB Scratch High Recall · Threshold 0.3 (Research)",
        detector="bopb",
        repair="telea",
        sensitivity=0.3,
        dilate=0,
        threshold=76,
        radius=3,
        commercial_cleared=False,
        repair_policy=RepairInclusionPolicy.SCRATCH_ONLY,
        checkpoint_sha256="b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95",
        license_status="Restricted (Microsoft Research Academic Non-Commercial License)",
        description="BOPB high-recall scratch operating point (threshold 0.3) capturing faint and semi-transparent scratch abrasions.",
        target_use_case="Heavily scratched archival film where maximizing scratch detection recall is prioritized.",
        evidence="Phase 6 benchmark calibration: Scratch Recall >= 55% at threshold 0.3 with zero film grain false positives.",
    ),
    "film_artefact_balanced": DefectDetectorPreset(
        name="film_artefact_balanced",
        label="Film Artefact Balanced · Threshold 0.5 (Commercial-Safe)",
        detector="film_artefact_segmenter",
        repair="telea",
        sensitivity=0.5,
        dilate=0,
        threshold=128,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        checkpoint_sha256="f7d3c3404b6da5557ce66ecb117b0bfc63376aa8930c4ab633b157ca31e97503",
        license_status="Permissive (MIT Clean-room reproduction, commercial-cleared)",
        description="Film Artefact Segmenter multi-task neural network calibrated at threshold 0.5 for balanced scratch and dust isolation.",
        target_use_case="Commercial archival remastering jobs requiring simultaneous neural scratch and dust segmentation.",
        evidence="Phase 6 benchmark calibration: Combined F1 = 0.5749, Precision = 73.77%, Scratch F1 = 0.5009 (Precision = 83.35%), Dust Recall = 78.05%.",
    ),
    "cnn_blotches_precision": DefectDetectorPreset(
        name="cnn_blotches_precision",
        label="CNN Blotches Precision · Threshold 0.8 (Commercial-Safe)",
        detector="cnn_blotches",
        repair="telea",
        sensitivity=0.8,
        dilate=0,
        threshold=204,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        checkpoint_sha256="f44b9610c20c3ed2e6195254edc58004bfe27074e94f7208083b3299a0b42c9d",
        license_status="Permissive (MIT Clean-room reproduction, commercial-cleared)",
        description="Dual-branch anisotropic CNN operating at high-precision threshold 0.8 with strict face and edge preservation.",
        target_use_case="Archival portrait sequences, close-ups, and architectural scenes where false positive blooming must be suppressed.",
        evidence="Phase 6 benchmark calibration: Precision = 86.59%, Face FPR = 0.0197%, Edge FPR = 0.0020%, F1 = 0.4622.",
    ),
    "patchcore_anomaly_research": DefectDetectorPreset(
        name="patchcore_anomaly_research",
        label="PatchCore Anomaly Detector · [Archival Core-set · Balanced]",
        detector="patchcore",
        repair="telea",
        sensitivity=0.75,
        dilate=0,
        threshold=128,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.COMBINED_DEFECT,
        checkpoint_sha256="2ddd2396222d49c21719f89dc6eb3281f020b081359a486be9a1f328fdc5d3be",
        license_status="Permissive (Apache-2.0 / BSD-3-Clause) · Commercial Cleared",
        description="PatchCore deep feature memory-bank anomaly detector (Recovered: high-resolution layer1+layer2 representation with calibrated archival core-set memory bank).",
        target_use_case="Visual anomaly detection on historical film scans utilizing core-set feature memory bank matching.",
        evidence="Empirical validation: Defect Recall = 67.7%, Precision = 28.5%, Intact-Pixel FPR = 2.00%, Clean Negative FPR = 1.67%, Pixel-F1 = 0.4013. Non-destructive archival safe.",
    ),
    "padim_anomaly_research": DefectDetectorPreset(
        name="padim_anomaly_research",
        label="PaDiM Anomaly Detector · [Spatially Invariant · Fast]",
        detector="padim",
        repair="telea",
        sensitivity=0.55,
        dilate=0,
        threshold=128,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.COMBINED_DEFECT,
        checkpoint_sha256="6c75c32dcf35f28e2b4e54f74d7dc2898501f83b3eebda4ad64dd5f8844cb541",
        license_status="Permissive (Apache-2.0 / BSD-3-Clause) · Commercial Cleared",
        description="PaDiM multivariate Gaussian anomaly detector (Recovered: spatially invariant distribution modeling with regularized covariance).",
        target_use_case="Ultra-fast statistical defect localization on archival footage via Mahalanobis distance mapping (56 FPS on CUDA).",
        evidence="Empirical validation: Defect Recall = 37.2%, Precision = 22.3%, Intact-Pixel FPR = 1.53%, Clean Negative FPR = 1.21%, Pixel-F1 = 0.2786. Non-destructive archival safe.",
    ),
    "temporal_scratch_balanced": DefectDetectorPreset(
        name="temporal_scratch_balanced",
        label="Temporal Scratch (Balanced)",
        detector="temporal_scratch",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.SCRATCH_ONLY,
        license_status="Apache-2.0 / BSD-3-Clause (OpenCV + NumPy)",
        description="Motion-compensated dense optical flow scratch detector with telecine de-duplication.",
        target_use_case="Archival film sequences with transient and drifting vertical scratches, including telecine cadenced material.",
        evidence="Phase 3 benchmark validation: +37.87% scratch recall improvement over spatial baselines (819 px detected, 100% core fissure), 0.000% clean-ROI FPR, 0.416% frame coverage.",
        params={"sensitivity": 0.65, "flow_backend": "dis", "window_radius": 3},
    ),

    # ---------------------------------------------------------------------------
    # Custom Parameter Preset
    # ---------------------------------------------------------------------------
    "custom": DefectDetectorPreset(
        name="custom",
        label="Custom Parameters",
        detector="farneback_3f",
        repair="telea",
        sensitivity=0.65,
        dilate=0,
        threshold=160,
        radius=3,
        commercial_cleared=True,
        repair_policy=RepairInclusionPolicy.CONSERVATIVE,
        description="User-customized parameter configuration.",
        target_use_case="Manual tuning of detector, sensitivity, threshold, radius, and dilation.",
        evidence="Custom configuration.",
    ),
}


def get_preset(name: str) -> DefectDetectorPreset:
    """Retrieve a `DefectDetectorPreset` by name."""
    norm = str(name).strip().lower()
    if norm not in BENCHMARK_PRESETS:
        available = ", ".join(BENCHMARK_PRESETS.keys())
        raise KeyError(f"Unknown preset {name!r}. Available: [{available}]")
    return BENCHMARK_PRESETS[norm]


def list_presets(*, commercial_only: bool = False) -> list[str]:
    """Return registered preset names, optionally filtering for commercial safety."""
    return [
        name for name, p in BENCHMARK_PRESETS.items()
        if not commercial_only or p.commercial_cleared
    ]


def resolve_preset(values: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Resolve preset name and merged parameter dictionary from input values.

    Preserves explicit caller overrides and maps matching parameter sets to
    canonical benchmark presets while retaining custom settings.
    """
    preset_name = str(values.get("preset", "")).strip().lower()
    if preset_name in BENCHMARK_PRESETS and preset_name != "custom":
        expected = BENCHMARK_PRESETS[preset_name]
        base = expected.as_dict()
        merged = {**base, **dict(values)}
        exp_det = "farneback_3f" if expected.detector in ("temporal", "farneback_3f") else expected.detector
        raw_det = str(merged.get("detector", expected.detector)).strip().lower()
        act_det = "farneback_3f" if raw_det in ("temporal", "farneback_3f") else raw_det
        is_exact = (
            act_det == exp_det
            and abs(float(merged.get("sensitivity", expected.sensitivity)) - expected.sensitivity) < 1e-4
            and int(merged.get("dilate", expected.dilate)) == expected.dilate
            and int(merged.get("threshold", expected.threshold)) == expected.threshold
        )
        return (preset_name if is_exact else "custom"), merged

    # If preset was not specified or was 'custom', try to match known presets
    raw_det = str(values.get("detector", "farneback_3f")).strip().lower()
    det = "farneback_3f" if raw_det in ("temporal", "farneback_3f") else raw_det
    sens = float(values.get("sensitivity", 0.65))
    dil = int(values.get("dilate", 1 if det == "farneback_3f" else 0))
    thresh = int(values.get("threshold", 160))

    for name, p in BENCHMARK_PRESETS.items():
        if name == "custom":
            continue
        exp_p_det = "farneback_3f" if p.detector in ("temporal", "farneback_3f") else p.detector
        if (
            exp_p_det == det
            and abs(p.sensitivity - sens) < 1e-4
            and p.dilate == dil
            and p.threshold == thresh
        ):
            return name, {**p.as_dict(), **dict(values), "preset": name}

    return "custom", {**dict(values), "preset": "custom"}


def fuse_spatial_and_temporal_masks(
    spatial_result: DefectMaskResult,
    temporal_result: DefectMaskResult,
    confidence_threshold: float = 0.60,
    min_aspect_ratio: float = 2.5,
    min_scratch_height: int = 18,
    is_anomaly_detector: bool = False,
) -> DefectMaskResult:
    """Fuse a primary spatial detector result with motion-compensated temporal scratch evidence.

    Confidence / Uncertainty partitioning:
    - High-Confidence Temporal Scratch:
      * Connected components in temporal scratch proposal with:
        (1) mean probability >= confidence_threshold (or individual pixels >= 0.70),
        (2) vertical continuity: ch >= min_scratch_height and (ch >= min_aspect_ratio * cw or ch >= 35).
      * Routed into `scratch_mask` (and `combined_mask`).
    - Uncertain Temporal Candidate:
      * Temporal candidates that do not qualify as high-confidence vertical scratches.
      * Routed into `uncertain_mask` (and `combined_mask`).
      * Strictly EXCLUDED from effective repair mask under CONSERVATIVE policy.
      * Included only under HIGH_RECALL policy.
    - Anomaly Detectors (PatchCore / PaDiM):
      * If `is_anomaly_detector` is True, primary spatial proposals represent generic patch anomalies
        without classified scratch/dust semantics. Temporal scratch evidence is added as an explicit
        scratch channel without converting the primary anomaly map into a safe scratch channel.
    - Safety Checks:
      * Does NOT bypass excessive-coverage safety checks. If combined coverage exceeds 30%,
        `safety_guard_triggered` is recorded.
    """
    n_frames = len(spatial_result.mask)
    if len(temporal_result.mask) != n_frames:
        raise ValueError(
            f"Frame count mismatch: spatial has {n_frames} frames, temporal has {len(temporal_result.mask)}"
        )

    fused_combined_masks: list[np.ndarray] = []
    fused_scratch_masks: list[np.ndarray] = []
    fused_dust_masks: list[np.ndarray] = []
    fused_uncertain_masks: list[np.ndarray] = []
    fused_mcdm: list[MultiChannelDefectMask] = []

    spatial_mcm_list = spatial_result.multi_channel_masks or []
    temporal_mcm_list = temporal_result.multi_channel_masks or []

    for idx in range(n_frames):
        s_mask = spatial_result.mask[idx]
        h, w = s_mask.shape

        # Extract spatial components
        s_scratch = (
            spatial_result.scratch_mask[idx]
            if spatial_result.scratch_mask is not None and idx < len(spatial_result.scratch_mask)
            else np.zeros((h, w), dtype=np.uint8)
        )
        s_dust = (
            spatial_result.dust_mask[idx]
            if spatial_result.dust_mask is not None and idx < len(spatial_result.dust_mask)
            else np.zeros((h, w), dtype=np.uint8)
        )
        s_uncertain = (
            spatial_result.uncertain_mask[idx]
            if spatial_result.uncertain_mask is not None and idx < len(spatial_result.uncertain_mask)
            else np.zeros((h, w), dtype=np.uint8)
        )

        if idx < len(spatial_mcm_list):
            s_prob = spatial_mcm_list[idx].combined_probability
        else:
            s_prob = np.clip(s_mask.astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)

        # Extract temporal components
        t_scratch = (
            temporal_result.scratch_mask[idx]
            if temporal_result.scratch_mask is not None and idx < len(temporal_result.scratch_mask)
            else temporal_result.mask[idx]
        )
        if idx < len(temporal_mcm_list):
            t_prob = temporal_mcm_list[idx].combined_probability
        else:
            t_prob = np.clip(temporal_result.mask[idx].astype(np.float32) / 255.0, 0.0, 1.0).astype(np.float32)

        # Partition temporal candidates into high-confidence vs uncertain
        high_conf_temporal = np.zeros((h, w), dtype=np.uint8)
        uncertain_temporal = np.zeros((h, w), dtype=np.uint8)

        if np.any(t_scratch > 0):
            t_bin = (t_scratch > 0).astype(np.uint8) * 255
            total, labels, stats, _ = cv2.connectedComponentsWithStats(t_bin, connectivity=8)
            for comp in range(1, total):
                _cx, _cy, cw, ch, area = map(int, stats[comp])
                comp_mask = labels == comp
                comp_p = float(t_prob[comp_mask].mean()) if np.any(comp_mask) else 0.0
                is_vertical_scratch = (
                    ch >= min_scratch_height
                    and (ch >= min_aspect_ratio * cw or ch >= 35)
                    and comp_p >= confidence_threshold
                )
                if is_vertical_scratch:
                    high_conf_temporal[comp_mask] = 255
                elif comp_p >= 0.40 or ch >= 10:
                    uncertain_temporal[comp_mask] = 255

        # Fuse scratch channel
        fused_scratch = np.bitwise_or(s_scratch, high_conf_temporal)
        fused_dust = s_dust.copy()
        fused_uncertain = np.bitwise_or(s_uncertain, uncertain_temporal)

        # Ensure disjoint semantic channels
        fused_uncertain[fused_scratch > 0] = 0
        fused_uncertain[fused_dust > 0] = 0

        # Fused combined mask and probability
        fused_combined = np.bitwise_or(np.bitwise_or(fused_scratch, fused_dust), fused_uncertain)
        if is_anomaly_detector:
            # Preserve primary anomaly proposals in combined mask
            fused_combined = np.bitwise_or(fused_combined, s_mask)

        fused_prob = np.maximum(s_prob, t_prob)

        # MultiChannelDefectMask provenance
        s_mcm = spatial_mcm_list[idx] if idx < len(spatial_mcm_list) else None
        prov: dict[str, ChannelProvenance] = {}
        prov["scratch"] = ChannelProvenance(
            origin=ChannelOrigin.DERIVED,
            is_available=True,
            source_algorithm="fused_spatial_temporal_scratch",
            derivation_parameters={
                "confidence_threshold": confidence_threshold,
                "high_conf_pixels": int(np.count_nonzero(high_conf_temporal)),
                "spatial_scratch_pixels": int(np.count_nonzero(s_scratch)),
            },
            confidence_threshold=confidence_threshold,
        )
        if s_mcm and "dust" in s_mcm.provenance:
            prov["dust"] = s_mcm.provenance["dust"]
        else:
            dust_avail = bool(np.any(fused_dust))
            prov["dust"] = ChannelProvenance(
                origin=ChannelOrigin.NATIVE if dust_avail else ChannelOrigin.UNSUPPORTED,
                is_available=dust_avail,
                source_algorithm="primary_spatial_detector",
            )
        uncert_avail = bool(np.any(fused_uncertain))
        prov["uncertain"] = ChannelProvenance(
            origin=ChannelOrigin.DERIVED if uncert_avail else ChannelOrigin.UNSUPPORTED,
            is_available=uncert_avail,
            source_algorithm="temporal_uncertain_partitioning",
            derivation_parameters={"uncertain_pixels": int(np.count_nonzero(fused_uncertain))},
        )

        mcdm = MultiChannelDefectMask(
            combined_probability=fused_prob,
            combined_mask=fused_combined,
            scratch_mask=fused_scratch if prov["scratch"].is_available else None,
            dust_mask=fused_dust if prov["dust"].is_available else None,
            uncertain_mask=fused_uncertain if prov["uncertain"].is_available else None,
            provenance=prov,
            metadata={
                "frame_index": idx,
                "spatial_pixels": int(np.count_nonzero(s_mask)),
                "temporal_pixels": int(np.count_nonzero(t_scratch)),
                "fused_scratch_pixels": int(np.count_nonzero(fused_scratch)),
                "uncertain_temporal_pixels": int(np.count_nonzero(fused_uncertain)),
                "is_anomaly_detector": is_anomaly_detector,
            },
        )

        fused_combined_masks.append(fused_combined)
        fused_scratch_masks.append(fused_scratch)
        fused_dust_masks.append(fused_dust)
        fused_uncertain_masks.append(fused_uncertain)
        fused_mcdm.append(mcdm)

    # 30% Safety guardrail verification across sequence
    total_pixels = sum(m.size for m in fused_combined_masks)
    total_defects = sum(int(np.count_nonzero(m)) for m in fused_combined_masks)
    mean_coverage = total_defects / max(1, total_pixels)
    safety_guard_triggered = mean_coverage > 0.30

    metadata = {
        "fusion": "spatial_plus_temporal_scratch",
        "frames": n_frames,
        "mean_coverage_fraction": round(mean_coverage, 5),
        "safety_guard_triggered": safety_guard_triggered,
        "is_anomaly_detector": is_anomaly_detector,
    }

    return DefectMaskResult(
        mask=fused_combined_masks,
        scratch_mask=fused_scratch_masks,
        dust_mask=fused_dust_masks,
        uncertain_mask=fused_uncertain_masks,
        multi_channel_masks=fused_mcdm,
        metadata=metadata,
    )


__all__ = [
    "ReadinessState",
    "ChannelOrigin",
    "RepairInclusionPolicy",
    "UnsupportedPolicyError",
    "MaskContractValidationError",
    "ChannelProvenance",
    "MultiChannelDefectMask",
    "UnknownDetectorError",
    "CommercialLicenseError",
    "DetectorNotExecutableError",
    "PreprocessedFrames",
    "DefectMaskResult",
    "normalize_frames",
    "split_proposal_components",
    "legacy_spatial_scratch_mask",
    "BaseDefectDetector",
    "FarnebackDefectDetector",
    "DISFlowDefectDetector",
    "RidgeDefectDetector",
    "Consensus5FDefectDetector",
    "LegacyScratchDetector",
    "ExternalDartHarnessDetector",
    "BopbScratchDetectorAdapter",
    "FilmArtefactSegmenterAdapter",
    "CNNBlotchesDetectorAdapter",
    "PatchCoreDefectAdapter",
    "PaDiMDefectAdapter",
    "CorrelNetDetectorAdapter",
    "TemporalScratchDetectorAdapter",
    "DefectDetectorRegistry",
    "build_default_registry",
    "DEFAULT_REGISTRY",
    "COMMERCIAL_DETECTOR_CHOICES",
    "get_defect_detector",
    "get_detector",
    "register_defect_detector",
    "list_defect_detectors",
    "AdaptiveTilingConfig",
    "AdaptiveTilingEngine",
    "DefectDetectorPreset",
    "BENCHMARK_PRESETS",
    "get_preset",
    "list_presets",
    "resolve_preset",
    "fuse_spatial_and_temporal_masks",
]


