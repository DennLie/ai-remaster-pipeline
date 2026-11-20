#!/usr/bin/env python3
"""Isolated Temporal Scratch Analyzer PoC for Phase 2.

Provides a standalone, testable `TemporalScratchAnalyzer` using dense optical
flow registration, robust background synthesis excluding the target frame,
telecine cadence de-duplication, dynamic noise floor thresholding via MAD,
motion-coherence filtering, and vertical structure enhancement.

Designed to detect transient and drifting film scratches (including those
persisting across 2-3 or more frames) without suppression by telecine duplicate
frames or legacy morphological size limits.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("temporal_scratch_analyzer")

# Ground truth ROIs from Phase 1 audit
SCRATCH_ROI = {
    "ymin": 48,
    "xmin": 180,
    "ymax": 282,
    "xmax": 192,
    "height": 234,
    "width": 12,
    "center_x": 184,
}

CLEAN_ROI = {
    "ymin": 80,
    "xmin": 620,
    "ymax": 200,
    "xmax": 700,
    "height": 120,
    "width": 80,
}


@dataclass
class TemporalConfig:
    """Configuration for TemporalScratchAnalyzer.

    Attributes:
        window_radius: Number of preceding and succeeding neighbor frames (window = 2R + 1).
        min_temporal_neighbors: Minimum number of valid non-duplicate neighbors required.
        flow_backend: Dense optical flow backend: "dis" | "farneback".
        telecine_dedup_threshold: Inter-frame MAD threshold below which frames are deemed telecine duplicates.
        scene_cut_threshold: Inter-frame MAD threshold above which neighbor selection stops (scene cut).
        sensitivity: Detection sensitivity in [0.0, 1.0].
        scratch_min_height: Minimum vertical component height in pixels.
        scratch_max_width: Maximum horizontal component width in pixels.
        scratch_tall_override_height: Component height above which aspect ratio filtering is bypassed.
        motion_coherence_threshold: Scene motion threshold in pixels to reject coherent scene textures.
        contrast_base: Base horizontal flank contrast threshold.
        k_factor: Base multiplier for MAD dynamic noise threshold.
        min_aspect_ratio: Minimum aspect ratio (ch / cw) for fine scratches.
        dilate_result: Whether to dilate binary scratch masks horizontally for complete fissure coverage.
    """

    window_radius: int = 3
    min_temporal_neighbors: int = 2
    flow_backend: str = "dis"  # "dis" | "farneback"
    telecine_dedup_threshold: float = 0.8
    scene_cut_threshold: float = 25.0
    sensitivity: float = 0.65
    scratch_min_height: int = 15
    scratch_max_width: int = 12
    scratch_tall_override_height: int = 35
    motion_coherence_threshold: float = 1.2
    contrast_base: float = 13.0
    k_factor: float = 1.3
    min_aspect_ratio: float = 3.5
    dilate_result: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.sensitivity <= 1.0):
            raise ValueError(f"sensitivity must be in [0.0, 1.0], got {self.sensitivity}")
        if self.window_radius < 1:
            raise ValueError(f"window_radius must be >= 1, got {self.window_radius}")
        if self.min_temporal_neighbors < 1:
            raise ValueError(f"min_temporal_neighbors must be >= 1, got {self.min_temporal_neighbors}")
        if self.scene_cut_threshold <= 0.0:
            raise ValueError(f"scene_cut_threshold must be > 0.0, got {self.scene_cut_threshold}")
        if self.scratch_tall_override_height < 1:
            raise ValueError(f"scratch_tall_override_height must be >= 1, got {self.scratch_tall_override_height}")
        if self.flow_backend not in ("dis", "farneback"):
            raise ValueError(f"Unsupported flow_backend '{self.flow_backend}'. Choose 'dis' or 'farneback'.")


@dataclass
class TemporalDefectResult:
    """Result of temporal defect analysis on a single frame.

    Attributes:
        scratch_mask: Binary scratch mask (contiguous uint8, strictly {0, 255}).
        probability_map: Continuous scratch probability map (contiguous float32, in [0.0, 1.0]).
        background: Synthesized robust background reference (contiguous uint8).
        metadata: Execution statistics and diagnostics.
    """

    scratch_mask: np.ndarray
    probability_map: np.ndarray
    background: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Strict invariant validation
        if not isinstance(self.scratch_mask, np.ndarray):
            raise TypeError("scratch_mask must be a numpy.ndarray")
        if self.scratch_mask.dtype != np.uint8:
            raise TypeError(f"scratch_mask must be uint8, got {self.scratch_mask.dtype}")
        if not self.scratch_mask.flags["C_CONTIGUOUS"]:
            self.scratch_mask = np.ascontiguousarray(self.scratch_mask)
        if not np.all((self.scratch_mask == 0) | (self.scratch_mask == 255)):
            raise ValueError("scratch_mask must contain strictly binary values: {0, 255}")

        if not isinstance(self.probability_map, np.ndarray):
            raise TypeError("probability_map must be a numpy.ndarray")
        if self.probability_map.dtype != np.float32:
            raise TypeError(f"probability_map must be float32, got {self.probability_map.dtype}")
        if not self.probability_map.flags["C_CONTIGUOUS"]:
            self.probability_map = np.ascontiguousarray(self.probability_map)
        if not np.all(np.isfinite(self.probability_map)):
            raise ValueError("probability_map contains non-finite values (NaN / Inf)")
        p_min, p_max = float(self.probability_map.min()), float(self.probability_map.max())
        if p_min < 0.0 or p_max > 1.0:
            raise ValueError(f"probability_map values must be in [0.0, 1.0], got [{p_min}, {p_max}]")

        if not isinstance(self.background, np.ndarray):
            raise TypeError("background must be a numpy.ndarray")
        if self.background.dtype != np.uint8:
            raise TypeError(f"background must be uint8, got {self.background.dtype}")
        if not self.background.flags["C_CONTIGUOUS"]:
            self.background = np.ascontiguousarray(self.background)

        h, w = self.scratch_mask.shape
        if self.probability_map.shape != (h, w):
            raise ValueError(f"Shape mismatch: probability {self.probability_map.shape} vs mask {(h, w)}")
        if self.background.shape != (h, w):
            raise ValueError(f"Shape mismatch: background {self.background.shape} vs mask {(h, w)}")


class TemporalScratchAnalyzer:
    """Isolated temporal film scratch analyzer.

    Registers neighboring non-duplicate frames using dense optical flow,
    synthesizes a robust reference background excluding the target frame,
    computes outlier differences against a dynamic MAD noise floor,
    applies motion-coherence filtering, and extracts high-aspect-ratio vertical scratches.
    """

    def __init__(self, config: TemporalConfig | dict[str, Any] | None = None) -> None:
        if isinstance(config, dict):
            # Extract valid fields
            valid_keys = TemporalConfig.__dataclass_fields__.keys()
            filtered = {k: v for k, v in config.items() if k in valid_keys}
            self.config = TemporalConfig(**filtered)
        elif isinstance(config, TemporalConfig):
            self.config = config
        elif config is None:
            self.config = TemporalConfig()
        else:
            raise TypeError(f"Expected TemporalConfig, dict, or None; got {type(config).__name__}")

        # Initialize optical flow engine
        self._dis_flow: cv2.DISOpticalFlow | None = None
        if self.config.flow_backend == "dis":
            try:
                self._dis_flow = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            except Exception as e:
                logger.warning("DISOpticalFlow creation failed (%s), falling back to Farneback.", e)
                self.config.flow_backend = "farneback"

        # Remap grid cache
        self._grid_cache_shape: tuple[int, int] | None = None
        self._cached_x_grid: np.ndarray | None = None
        self._cached_y_grid: np.ndarray | None = None

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        """Convert input frame to contiguous uint8 grayscale."""
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(frame).__name__}")
        if frame.dtype != np.uint8:
            raise TypeError(f"Expected uint8 frame, got {frame.dtype}")
        if frame.ndim == 2:
            return np.ascontiguousarray(frame)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        elif frame.ndim == 3 and frame.shape[2] == 1:
            return np.ascontiguousarray(frame[:, :, 0])
        else:
            raise ValueError(f"Unsupported frame geometry: shape={frame.shape}")

    def _get_remap_grid(self, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """Return cached meshgrid coordinate arrays for remap."""
        if self._grid_cache_shape != (h, w) or self._cached_x_grid is None or self._cached_y_grid is None:
            x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._cached_x_grid = x
            self._cached_y_grid = y
            self._grid_cache_shape = (h, w)
        return self._cached_x_grid, self._cached_y_grid

    def _select_nonduplicate_neighbors(
        self, gray_frames: Sequence[np.ndarray], target_idx: int
    ) -> tuple[list[int], list[int], list[int]]:
        """Select preceding and succeeding non-duplicate neighbors stepping over telecine repeats.

        Halts search in either direction if inter-frame MAD exceeds scene_cut_threshold
        to prevent selecting frames across camera cuts or scene transitions.

        Returns:
            (preceding_indices, succeeding_indices, skipped_duplicate_indices)
        """
        n = len(gray_frames)
        r = self.config.window_radius
        thresh = self.config.telecine_dedup_threshold
        cut_thresh = self.config.scene_cut_threshold

        skipped_duplicates: list[int] = []

        # Preceding neighbors (stepping backwards)
        preceding: list[int] = []
        last_accepted = target_idx
        for p in range(target_idx - 1, -1, -1):
            mad_t = float(np.mean(cv2.absdiff(gray_frames[p], gray_frames[target_idx])))
            if mad_t > cut_thresh:
                break
            mad_last = float(np.mean(cv2.absdiff(gray_frames[p], gray_frames[last_accepted])))
            if mad_last > cut_thresh:
                break
            if mad_t >= thresh and mad_last >= thresh:
                preceding.append(p)
                last_accepted = p
                if len(preceding) == r:
                    break
            else:
                skipped_duplicates.append(p)

        # Succeeding neighbors (stepping forwards)
        succeeding: list[int] = []
        last_accepted = target_idx
        for s in range(target_idx + 1, n):
            mad_t = float(np.mean(cv2.absdiff(gray_frames[s], gray_frames[target_idx])))
            if mad_t > cut_thresh:
                break
            mad_last = float(np.mean(cv2.absdiff(gray_frames[s], gray_frames[last_accepted])))
            if mad_last > cut_thresh:
                break
            if mad_t >= thresh and mad_last >= thresh:
                succeeding.append(s)
                last_accepted = s
                if len(succeeding) == r:
                    break
            else:
                skipped_duplicates.append(s)

        return preceding, succeeding, skipped_duplicates

    def _compute_dense_flow(self, ref_gray: np.ndarray, nbr_gray: np.ndarray) -> np.ndarray:
        """Compute dense backward optical flow from ref_gray to nbr_gray."""
        if self.config.flow_backend == "dis" and self._dis_flow is not None:
            return self._dis_flow.calc(ref_gray, nbr_gray, None)
        return cv2.calcOpticalFlowFarneback(
            ref_gray,
            nbr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )

    def _warp_neighbor(self, nbr_gray: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Backward warp neighbor to align with reference coordinate frame."""
        h, w = nbr_gray.shape
        x_grid, y_grid = self._get_remap_grid(h, w)
        return cv2.remap(
            nbr_gray,
            x_grid + flow[..., 0],
            y_grid + flow[..., 1],
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _synthesize_background(self, warped_stack: np.ndarray) -> np.ndarray:
        """Synthesize robust reference background strictly excluding target frame.

        Args:
            warped_stack: float32 array of shape (K, H, W) containing K warped neighbors.

        Returns:
            float32 array of shape (H, W).
        """
        k = warped_stack.shape[0]
        if k >= 4:
            # Trimmed median: sort along neighbor axis, discard extremes, average central values
            sorted_stack = np.sort(warped_stack, axis=0)
            return np.mean(sorted_stack[1:-1], axis=0)
        return np.median(warped_stack, axis=0)

    def _compute_flank_contrast(self, gray: np.ndarray, flank: int = 5) -> np.ndarray:
        """Compute horizontal flank contrast (vertical ridge profile evidence)."""
        h, w = gray.shape
        flank = min(flank, max(3, w // 140))
        padded = np.pad(gray.astype(np.float32), ((0, 0), (flank, flank)), mode="edge")
        left = padded[:, :w]
        right = padded[:, 2 * flank : 2 * flank + w]
        img_f = gray.astype(np.float32)
        bright = np.minimum(img_f - left, img_f - right)
        dark = np.minimum(left - img_f, right - img_f)
        return np.maximum(bright, dark)

    def analyze_frame(
        self,
        frames: Sequence[np.ndarray],
        target_idx: int,
        gray_frames: Sequence[np.ndarray] | None = None,
    ) -> TemporalDefectResult:
        """Analyze a target frame within a temporal sequence for scratches.

        Args:
            frames: Sequence of BGR or grayscale video frames.
            target_idx: Zero-based index of frame to analyze.
            gray_frames: Optional pre-converted sequence of grayscale frames.
                         If omitted, frames will be converted automatically.

        Returns:
            TemporalDefectResult with scratch mask, probability map, background, and metadata.
        """
        t0 = time.perf_counter()
        n_frames = len(frames)
        if not (0 <= target_idx < n_frames):
            raise IndexError(f"target_idx {target_idx} out of range for sequence of length {n_frames}")

        if gray_frames is None:
            gray_frames = [self._ensure_gray(f) for f in frames]
        target_gray = gray_frames[target_idx]
        h, w = target_gray.shape

        # Micro-dimension guard: reject tiny frames/tiles that crash dense optical flow
        if h < 16 or w < 16:
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            zero_prob = np.zeros((h, w), dtype=np.float32)
            return TemporalDefectResult(
                scratch_mask=zero_mask,
                probability_map=zero_prob,
                background=target_gray.copy(),
                metadata={
                    "status": "abstain",
                    "reason": "small_spatial_dimension",
                    "target_idx": target_idx,
                    "height": h,
                    "width": w,
                    "scratch_pixels": 0,
                    "execution_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                },
            )

        # Select non-duplicate temporal neighbors
        preceding, succeeding, skipped_dups = self._select_nonduplicate_neighbors(gray_frames, target_idx)
        valid_neighbors = preceding + succeeding

        # Boundary / insufficient neighbors abstention
        if len(valid_neighbors) < self.config.min_temporal_neighbors:
            zero_mask = np.zeros((h, w), dtype=np.uint8)
            zero_prob = np.zeros((h, w), dtype=np.float32)
            return TemporalDefectResult(
                scratch_mask=zero_mask,
                probability_map=zero_prob,
                background=target_gray.copy(),
                metadata={
                    "status": "abstain",
                    "reason": "insufficient_temporal_neighbors",
                    "target_idx": target_idx,
                    "valid_neighbors": valid_neighbors,
                    "preceding_neighbors": preceding,
                    "succeeding_neighbors": succeeding,
                    "telecine_skipped": skipped_dups,
                    "scratch_pixels": 0,
                    "execution_time_ms": round((time.perf_counter() - t0) * 1000, 2),
                },
            )

        # Dense optical flow & backward warping for each neighbor
        warped_list: list[np.ndarray] = []
        flows: list[np.ndarray] = []
        for n_idx in valid_neighbors:
            fl = self._compute_dense_flow(target_gray, gray_frames[n_idx])
            w_img = self._warp_neighbor(gray_frames[n_idx], fl)
            warped_list.append(w_img.astype(np.float32))
            flows.append(fl)

        # Robust Background Synthesis (Target Excluded)
        warped_stack = np.stack(warped_list, axis=0)  # Shape: (K, H, W)
        bg_float = self._synthesize_background(warped_stack)
        bg_uint8 = np.ascontiguousarray(np.clip(bg_float, 0, 255).astype(np.uint8))

        # Dynamic Noise Floor & Outlier Difference Map
        diff = np.abs(target_gray.astype(np.float32) - bg_float)
        med_diff = float(np.median(diff))
        mad_diff = float(np.median(np.abs(diff - med_diff)))
        sigma_mad = 1.4826 * max(mad_diff, 1.0)

        # Adaptive threshold k based on sensitivity
        k = self.config.k_factor - 0.8 * (self.config.sensitivity - 0.5)
        tau = med_diff + k * sigma_mad
        raw_temporal = diff >= tau

        # Vertical Structure Evidence (Horizontal Flank Contrast)
        flank_contrast = self._compute_flank_contrast(target_gray)
        contrast_th = self.config.contrast_base - 6.0 * (self.config.sensitivity - 0.5)
        structure_ok = flank_contrast >= contrast_th

        # Motion Coherence Filtering:
        # Compare local scene motion magnitude against threshold.
        # Defect textures moving coherently with scene motion (> threshold) are rejected.
        scene_flow_mag = np.mean([np.hypot(fl[..., 0], fl[..., 1]) for fl in flows], axis=0)
        motion_ok = scene_flow_mag <= self.config.motion_coherence_threshold

        # Initial candidate mask
        candidate = (raw_temporal & structure_ok & motion_ok).astype(np.uint8) * 255
        candidate_count = int(np.count_nonzero(candidate))

        # Morphological continuity enhancement: bridge small vertical fissures
        opened = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((5, 1), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 1), np.uint8))

        # Connected component filtering: aspect ratio and geometry
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        scratch_mask = np.zeros_like(closed, dtype=np.uint8)

        min_h = self.config.scratch_min_height
        max_w = self.config.scratch_max_width
        min_aspect = self.config.min_aspect_ratio
        tall_override = self.config.scratch_tall_override_height

        for i in range(1, n_labels):
            _x, _y, cw, ch, area = stats[i]
            # Valid scratch: within width limit, exceeds minimum height, and has high vertical aspect ratio
            if cw <= max_w and ch >= min_h and (ch >= min_aspect * cw or ch >= tall_override):
                scratch_mask[labels == i] = 255

        # Horizontal dilation for full scratch groove coverage if enabled
        if self.config.dilate_result and np.any(scratch_mask):
            scratch_mask = cv2.dilate(scratch_mask, np.ones((1, 3), np.uint8))

        scratch_mask = np.ascontiguousarray(scratch_mask)
        final_scratch_pixels = int(np.count_nonzero(scratch_mask))

        # Continuous probability map synthesis:
        # Standardize score into continuous [0.0, 1.0]
        z_score = (diff - med_diff) / max(sigma_mad, 1.0)
        s_temp = np.clip(z_score / 4.0, 0.0, 1.0)
        s_ridge = np.clip(flank_contrast / 30.0, 0.0, 1.0)
        s_motion = np.clip(1.0 - scene_flow_mag / max(self.config.motion_coherence_threshold, 0.1), 0.0, 1.0)
        s_combined = (s_temp * 0.45 + s_ridge * 0.40 + s_motion * 0.15).astype(np.float32)

        probability_map = np.zeros((h, w), dtype=np.float32)
        scratch_idx = scratch_mask > 0
        probability_map[scratch_idx] = np.clip(0.5 + 0.5 * s_combined[scratch_idx], 0.5, 1.0)
        probability_map[~scratch_idx] = np.clip(0.48 * s_combined[~scratch_idx], 0.0, 0.48)
        probability_map = np.ascontiguousarray(probability_map)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        metadata = {
            "status": "success",
            "target_idx": target_idx,
            "valid_neighbors": valid_neighbors,
            "preceding_neighbors": preceding,
            "succeeding_neighbors": succeeding,
            "telecine_skipped": skipped_dups,
            "flow_backend": self.config.flow_backend,
            "noise_median": round(med_diff, 3),
            "noise_mad": round(mad_diff, 3),
            "noise_tau": round(tau, 3),
            "candidate_pixels": candidate_count,
            "scratch_pixels": final_scratch_pixels,
            "execution_time_ms": round(elapsed_ms, 2),
        }

        return TemporalDefectResult(
            scratch_mask=scratch_mask,
            probability_map=probability_map,
            background=bg_uint8,
            metadata=metadata,
        )

    def analyze_clip(self, frames: Sequence[np.ndarray]) -> list[TemporalDefectResult]:
        """Analyze all frames in a video sequence sequentially."""
        gray_frames = [self._ensure_gray(f) for f in frames]
        return [self.analyze_frame(frames, i, gray_frames=gray_frames) for i in range(len(frames))]


def load_clip(video_path: Path) -> list[np.ndarray]:
    """Load all frames of video clip using OpenCV."""
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found at: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file at: {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from video file at: {video_path}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal Scratch Analyzer PoC CLI")
    parser.add_argument(
        "--video",
        type=Path,
        default=PROJECT_ROOT / "evaluation" / "Neighbors_72.mkv",
        help="Path to evaluation video file",
    )
    parser.add_argument("--target-frame", type=int, default=25, help="Single target frame to benchmark")
    parser.add_argument("--start-frame", type=int, default=20, help="Start frame for range benchmark")
    parser.add_argument("--end-frame", type=int, default=36, help="End frame (exclusive) for range benchmark")
    parser.add_argument("--sensitivity", type=float, default=0.65, help="Sensitivity in [0.0, 1.0]")
    parser.add_argument("--flow-backend", choices=["dis", "farneback"], default="dis", help="Flow backend")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"Loading evaluation video: {args.video}")
    frames = load_clip(args.video)
    print(f"Loaded {len(frames)} frames of size {frames[0].shape[1]}x{frames[0].shape[0]}")

    config = TemporalConfig(
        sensitivity=args.sensitivity,
        flow_backend=args.flow_backend,
        window_radius=3,
        min_temporal_neighbors=2,
    )
    analyzer = TemporalScratchAnalyzer(config)

    # 1. Primary Benchmark Target: Frame 25
    print(f"\n================================================================================")
    print(f"PRIMARY BENCHMARK EVALUATION: Target Frame {args.target_frame}")
    print(f"================================================================================")
    result = analyzer.analyze_frame(frames, args.target_frame)

    s_ymin, s_xmin, s_ymax, s_xmax = (
        SCRATCH_ROI["ymin"],
        SCRATCH_ROI["xmin"],
        SCRATCH_ROI["ymax"],
        SCRATCH_ROI["xmax"],
    )
    c_ymin, c_xmin, c_ymax, c_xmax = (
        CLEAN_ROI["ymin"],
        CLEAN_ROI["xmin"],
        CLEAN_ROI["ymax"],
        CLEAN_ROI["xmax"],
    )

    s_pixels = int(np.count_nonzero(result.scratch_mask[s_ymin:s_ymax, s_xmin:s_xmax]))
    c_pixels = int(np.count_nonzero(result.scratch_mask[c_ymin:c_ymax, c_xmin:c_xmax]))
    total_scratch_pixels = int(np.count_nonzero(result.scratch_mask))

    print(f"Frame {args.target_frame} Metrics:")
    print(f"  Scratch ROI Detected: {s_pixels} pixels (Requirement: >= 50 pixels)")
    print(f"  Clean ROI False Alarms: {c_pixels} pixels (Requirement: < 5 pixels)")
    print(f"  Full-Frame Scratch Pixels: {total_scratch_pixels}")
    print(f"  Execution Time: {result.metadata['execution_time_ms']} ms")
    print(f"  Valid Neighbors: {result.metadata['valid_neighbors']}")
    print(f"  Telecine Skipped: {result.metadata['telecine_skipped']}")
    print(f"  Noise Baseline: Median={result.metadata['noise_median']}, MAD={result.metadata['noise_mad']}, Tau={result.metadata['noise_tau']}")
    print(f"  Mask Dtype: {result.scratch_mask.dtype}, Shape: {result.scratch_mask.shape}")
    print(f"  Prob Range: [{result.probability_map.min():.3f}, {result.probability_map.max():.3f}]")

    # Contract assertion checks
    assert result.scratch_mask.dtype == np.uint8, "Mask must be uint8"
    assert result.scratch_mask.shape == (720, 960), "Shape must be (720, 960)"
    assert np.all((result.scratch_mask == 0) | (result.scratch_mask == 255)), "Mask values must be in {0, 255}"
    assert result.probability_map.dtype == np.float32, "Probability map must be float32"
    assert 0.0 <= result.probability_map.min() and result.probability_map.max() <= 1.0, "Prob must be in [0, 1]"
    assert s_pixels >= 50, f"Validation failure: Scratch pixels {s_pixels} < 50"
    assert c_pixels < 5, f"Validation failure: Clean ROI false alarms {c_pixels} >= 5"
    print("\n>>> PRIMARY BENCHMARK VALIDATION PASSED: Frame 25 meets all criteria! <<<")

    # 2. Sequence Benchmark: Frames 20..35
    print(f"\n================================================================================")
    print(f"RANGE BENCHMARK EVALUATION: Target Frames {args.start_frame}..{args.end_frame - 1}")
    print(f"================================================================================")
    range_detected = 0
    total_scratch_roi_pixels = 0
    total_clean_roi_pixels = 0

    print(f"{'Frame':<8}{'Scratch ROI':<16}{'Clean ROI (FP)':<18}{'Total Mask':<14}{'Exec (ms)':<12}{'Neighbors'}")
    print("-" * 80)
    for t in range(args.start_frame, args.end_frame):
        res = analyzer.analyze_frame(frames, t)
        s_cnt = int(np.count_nonzero(res.scratch_mask[s_ymin:s_ymax, s_xmin:s_xmax]))
        c_cnt = int(np.count_nonzero(res.scratch_mask[c_ymin:c_ymax, c_xmin:c_xmax]))
        tot = int(np.count_nonzero(res.scratch_mask))
        nbrs_str = str(res.metadata.get("valid_neighbors", []))

        if s_cnt >= 20:
            range_detected += 1
        total_scratch_roi_pixels += s_cnt
        total_clean_roi_pixels += c_cnt

        print(f"Frame {t:<2d} {s_cnt:<16d}{c_cnt:<18d}{tot:<14d}{res.metadata['execution_time_ms']:<12.1f}{nbrs_str}")

    print("-" * 80)
    print(
        f"Range Summary: {range_detected}/{args.end_frame - args.start_frame} frames with strong scratch detection "
        f"(Total Scratch ROI: {total_scratch_roi_pixels} px, Total Clean ROI: {total_clean_roi_pixels} px)"
    )


if __name__ == "__main__":
    main()
