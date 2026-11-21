"""Adaptive Tiling Engine and GPU VRAM Protection for Neural Film Defect Restoration.

Provides seamless full-resolution (1080p, 2K, 4K) inference across neural defect models
with context padding (eliminating edge convolution cutoff artifacts), pre-threshold
continuous probability blending (2D Gaussian and distance weighting with zero seam artifacts),
deterministic temporal grid locking, and proactive VRAM monitoring/throttling.
"""
from __future__ import annotations

import logging
import math
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

if __name__ in sys.modules:
    sys.modules.setdefault("adaptive_tiling", sys.modules[__name__])
    sys.modules.setdefault("scripts.adaptive_tiling", sys.modules[__name__])

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TileCoordinate:
    """Coordinate and context padding metadata for a single tile within an image.

    Attributes:
        x: Horizontal offset (column index) of the core tile in the full image.
        y: Vertical offset (row index) of the core tile in the full image.
        width: Width of the core tile in pixels.
        height: Height of the core tile in pixels.
        pad_top: Context padding in pixels added to the top border.
        pad_bottom: Context padding in pixels added to the bottom border.
        pad_left: Context padding in pixels added to the left border.
        pad_right: Context padding in pixels added to the right border.
        tile_index: Zero-based sequential tile index.
    """

    x: int
    y: int
    width: int
    height: int
    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0
    tile_index: int = 0

    @property
    def core_bounds(self) -> tuple[int, int, int, int]:
        """Return (y_start, y_end, x_start, x_end) slice bounds for the core tile."""
        return (self.y, self.y + self.height, self.x, self.x + self.width)

    @property
    def padded_width(self) -> int:
        """Return total width of tile including context padding."""
        return self.width + self.pad_left + self.pad_right

    @property
    def padded_height(self) -> int:
        """Return total height of tile including context padding."""
        return self.height + self.pad_top + self.pad_bottom


@dataclass
class AdaptiveTilingConfig:
    """Configuration contract for adaptive neural tiling and VRAM protection.

    Attributes:
        tile_size: Core tile dimensions (height, width) in pixels (e.g., (256, 256) or (512, 512)).
        overlap: Overlap border dimensions (height, width) between adjacent tiles in pixels.
        context_padding: Extra pixel border around tile passed to neural net to eliminate convolution cutoff.
        blend_mode: Weighting function for continuous probability blending ("gaussian", "distance_linear", "hanning").
        vram_budget_mb: Target maximum GPU memory in MB before batch throttling or CPU fallback.
        batch_size: Target batch size for neural tile evaluation.
        device: Target execution device ("auto", "cuda", "cpu").
    """

    tile_size: tuple[int, int] = (512, 512)
    overlap: tuple[int, int] = (64, 64)
    context_padding: int = 32
    blend_mode: str = "gaussian"
    vram_budget_mb: int = 4096
    batch_size: int = 4
    device: str = "auto"

    def __post_init__(self) -> None:
        if isinstance(self.tile_size, int):
            self.tile_size = (int(self.tile_size), int(self.tile_size))
        elif isinstance(self.tile_size, (list, tuple)) and len(self.tile_size) == 2:
            self.tile_size = (int(self.tile_size[0]), int(self.tile_size[1]))
        else:
            raise ValueError(f"tile_size must be a 2-tuple of positive integers, got {self.tile_size!r}")

        if self.tile_size[0] <= 0 or self.tile_size[1] <= 0:
            raise ValueError(f"tile_size elements must be > 0, got {self.tile_size}")

        if isinstance(self.overlap, int):
            self.overlap = (int(self.overlap), int(self.overlap))
        elif isinstance(self.overlap, (list, tuple)) and len(self.overlap) == 2:
            self.overlap = (int(self.overlap[0]), int(self.overlap[1]))
        else:
            raise ValueError(f"overlap must be a 2-tuple of integers, got {self.overlap!r}")

        if self.overlap[0] < 0 or self.overlap[1] < 0:
            raise ValueError(f"overlap elements must be >= 0, got {self.overlap}")
        if self.overlap[0] >= self.tile_size[0] or self.overlap[1] >= self.tile_size[1]:
            raise ValueError(
                f"overlap {self.overlap} must be strictly smaller than tile_size {self.tile_size}"
            )

        if not isinstance(self.context_padding, int) or self.context_padding < 0:
            raise ValueError(f"context_padding must be a non-negative integer, got {self.context_padding!r}")

        valid_modes = {"gaussian", "distance_linear", "hanning", "cosine"}
        if str(self.blend_mode).strip().lower() not in valid_modes:
            raise ValueError(f"blend_mode must be one of {sorted(valid_modes)}, got {self.blend_mode!r}")
        self.blend_mode = str(self.blend_mode).strip().lower()
        if self.blend_mode == "cosine":
            self.blend_mode = "hanning"

        if not isinstance(self.vram_budget_mb, (int, float)) or self.vram_budget_mb <= 0:
            raise ValueError(f"vram_budget_mb must be a positive number, got {self.vram_budget_mb!r}")
        self.vram_budget_mb = int(self.vram_budget_mb)

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be an integer >= 1, got {self.batch_size!r}")


class VRAMMonitor:
    """Proactive GPU VRAM monitor and batch throttling controller."""

    def __init__(self, default_budget_mb: int = 4096) -> None:
        self.default_budget_mb = default_budget_mb

    @staticmethod
    def get_available_vram_mb(device: int | str | None = None) -> float:
        """Return available GPU VRAM in megabytes, or 0.0 if CUDA is unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                idx = 0
                if isinstance(device, int):
                    idx = device
                elif isinstance(device, str) and ":" in device:
                    try:
                        idx = int(device.split(":")[1])
                    except ValueError:
                        idx = 0
                free_bytes, _ = torch.cuda.mem_get_info(idx)
                return float(free_bytes) / (1024.0 * 1024.0)
        except Exception as exc:
            logger.debug("Failed to query PyTorch CUDA memory: %s", exc)
        return 0.0

    @staticmethod
    def estimate_tile_vram_mb(
        tile_shape: tuple[int, int],
        channels: int = 3,
        bytes_per_elem: int = 4,
        activation_multiplier: float = 16.0,
    ) -> float:
        """Estimate VRAM footprint in MB for a single tile during neural inference."""
        h, w = tile_shape[:2]
        raw_input_bytes = h * w * channels * bytes_per_elem
        est_total_bytes = raw_input_bytes * (1.0 + activation_multiplier)
        return float(est_total_bytes) / (1024.0 * 1024.0)

    def determine_batch_size(
        self,
        requested_batch_size: int,
        tile_shape: tuple[int, int],
        vram_budget_mb: int | None = None,
        device: str = "auto",
        model_overhead_mb: float = 300.0,
        available_vram_mb: float | None = None,
    ) -> tuple[int, bool]:
        """Calculate optimal execution batch size and determine if CPU fallback is required.

        Args:
            requested_batch_size: Desired batch size from user or config.
            tile_shape: Spatial dimensions of tile passed to neural net (including context padding).
            vram_budget_mb: Explicit VRAM budget ceiling in MB.
            device: Compute device string ("auto", "cuda", "cpu").
            model_overhead_mb: Estimated memory reserved for model weights and baseline workspace.
            available_vram_mb: Optional explicit memory override (used for deterministic testing).

        Returns:
            Tuple of (effective_batch_size: int, fell_back_to_cpu: bool).
        """
        budget = float(vram_budget_mb if vram_budget_mb is not None else self.default_budget_mb)
        avail_gpu = available_vram_mb if available_vram_mb is not None else self.get_available_vram_mb(device)

        is_cpu_device = isinstance(device, str) and device.strip().lower() == "cpu"
        if is_cpu_device or avail_gpu <= 0.0:
            fell_back_to_cpu = not is_cpu_device
            if fell_back_to_cpu:
                msg = (
                    "GPU VRAM not available or CUDA not detected; "
                    "falling back to CPU execution with minimal batch allocation."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
                logger.warning(msg)
            effective_bs = max(1, min(requested_batch_size, 2))
            return effective_bs, fell_back_to_cpu

        effective_budget = min(avail_gpu, budget)
        usable_budget = max(50.0, effective_budget - model_overhead_mb)
        est_per_tile = self.estimate_tile_vram_mb(tile_shape)

        if est_per_tile <= 0.0:
            return requested_batch_size, False

        max_possible = max(1, int(usable_budget // est_per_tile))
        effective_bs = max(1, min(requested_batch_size, max_possible))

        if effective_bs < requested_batch_size:
            msg = (
                f"VRAM budget ({effective_budget:.0f} MB, usable ~{usable_budget:.0f} MB) is insufficient "
                f"for requested batch_size={requested_batch_size} (~{requested_batch_size * est_per_tile:.1f} MB); "
                f"auto-throttled batch_size to {effective_bs}."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            logger.warning(msg)

        if usable_budget < est_per_tile:
            msg = (
                f"Estimated memory for single tile ({est_per_tile:.1f} MB) exceeds available usable budget "
                f"({usable_budget:.0f} MB); proceeding with batch_size=1 with explicit VRAM pressure warning."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            logger.warning(msg)

        return effective_bs, False


def create_blend_weight(shape: tuple[int, int], blend_mode: str = "gaussian") -> np.ndarray:
    """Create a 2D continuous weighting window normalized in [1e-4, 1.0].

    Args:
        shape: Spatial tile dimensions (height, width).
        blend_mode: "gaussian", "distance_linear", or "hanning".

    Returns:
        float32 2D ndarray of weights with shape `(height, width)`.
    """
    h, w = shape
    mode = str(blend_mode).strip().lower()
    if mode == "cosine":
        mode = "hanning"

    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid tile shape for weight creation: {shape}")

    if mode == "gaussian":
        sigma_y = max(1.0, float(h) / 4.0)
        sigma_x = max(1.0, float(w) / 4.0)
        y = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
        x = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
        gy = np.exp(-0.5 * (y / sigma_y) ** 2)
        gx = np.exp(-0.5 * (x / sigma_x) ** 2)
        weight = np.outer(gy, gx).astype(np.float32)
    elif mode == "distance_linear":
        y = np.minimum(np.arange(h, dtype=np.float32) + 1.0, float(h) - np.arange(h, dtype=np.float32))
        x = np.minimum(np.arange(w, dtype=np.float32) + 1.0, float(w) - np.arange(w, dtype=np.float32))
        weight = np.outer(y, x).astype(np.float32)
    elif mode == "hanning":
        if h == 1:
            wy = np.ones(1, dtype=np.float32)
        else:
            wy = np.hanning(h + 2)[1:-1].astype(np.float32)
        if w == 1:
            wx = np.ones(1, dtype=np.float32)
        else:
            wx = np.hanning(w + 2)[1:-1].astype(np.float32)
        weight = np.outer(wy, wx).astype(np.float32)
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode!r}. Expected 'gaussian', 'distance_linear', or 'hanning'.")

    # Guard against zero boundary weights to prevent numerical instability
    return np.maximum(weight, 1e-4).astype(np.float32)


class AdaptiveTilingEngine:
    """Seamless high-resolution tiling engine with continuous pre-threshold blending."""

    def __init__(
        self,
        config: AdaptiveTilingConfig | None = None,
        vram_monitor: VRAMMonitor | None = None,
    ) -> None:
        self.config = config or AdaptiveTilingConfig()
        self.vram_monitor = vram_monitor or VRAMMonitor(default_budget_mb=self.config.vram_budget_mb)

    def generate_tiles(
        self,
        image_shape: tuple[int, int],
        config: AdaptiveTilingConfig | None = None,
    ) -> list[TileCoordinate]:
        """Generate overlapping tile coordinates covering the full image resolution.

        Guarantees:
        1. 100% spatial coverage of [0, H) x [0, W) with zero uncovered pixels.
        2. Deterministic, spatially locked coordinates across consecutive video frames.
        3. Uniform context padding attributes per tile.

        Args:
            image_shape: (height, width) of the target image.
            config: Optional config override.

        Returns:
            List of `TileCoordinate` objects ordered in row-major order.
        """
        cfg = config or self.config
        h, w = int(image_shape[0]), int(image_shape[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"Image dimensions must be positive, got {image_shape}")

        tile_h, tile_w = cfg.tile_size
        overlap_y, overlap_x = cfg.overlap
        pad = cfg.context_padding

        def compute_axis_steps(length: int, tile_len: int, overlap_len: int) -> list[int]:
            if length <= tile_len:
                return [0]
            stride = max(1, tile_len - overlap_len)
            num_tiles = math.ceil((length - tile_len) / stride) + 1
            if num_tiles <= 1:
                return [0]
            steps = [int(round(i * (length - tile_len) / (num_tiles - 1))) for i in range(num_tiles)]
            return sorted(list(set(steps)))

        y_coords = compute_axis_steps(h, tile_h, overlap_y)
        x_coords = compute_axis_steps(w, tile_w, overlap_x)

        tiles: list[TileCoordinate] = []
        idx = 0
        for y in y_coords:
            cur_h = min(tile_h, h - y)
            for x in x_coords:
                cur_w = min(tile_w, w - x)
                tiles.append(
                    TileCoordinate(
                        x=x,
                        y=y,
                        width=cur_w,
                        height=cur_h,
                        pad_top=pad,
                        pad_bottom=pad,
                        pad_left=pad,
                        pad_right=pad,
                        tile_index=idx,
                    )
                )
                idx += 1

        return tiles

    @staticmethod
    def extract_padded_patch(
        frame: np.ndarray,
        coord: TileCoordinate,
        context_padding: int | None = None,
    ) -> np.ndarray:
        """Extract tile region with reflection context padding beyond image boundaries.

        Args:
            frame: Input frame ndarray (2D [H, W], 3D [H, W, C], or 4D [T, H, W, C]).
            coord: TileCoordinate defining the core crop.
            context_padding: Border padding size (defaults to coord.pad_top).

        Returns:
            Extracted ndarray with shape expanded by 2 * padding along spatial dimensions.
        """
        p = context_padding if context_padding is not None else coord.pad_top
        h_frame, w_frame = frame.shape[:2] if frame.ndim in (2, 3) else frame.shape[1:3]

        if p == 0:
            if frame.ndim in (2, 3):
                return frame[coord.y : coord.y + coord.height, coord.x : coord.x + coord.width].copy()
            # 4D [T, H, W, C]
            return frame[:, coord.y : coord.y + coord.height, coord.x : coord.x + coord.width].copy()

        y_start = coord.y - p
        y_end = coord.y + coord.height + p
        x_start = coord.x - p
        x_end = coord.x + coord.width + p

        pad_top = max(0, -y_start)
        pad_bottom = max(0, y_end - h_frame)
        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - w_frame)

        valid_y_start = max(0, y_start)
        valid_y_end = min(h_frame, y_end)
        valid_x_start = max(0, x_start)
        valid_x_end = min(w_frame, x_end)

        if frame.ndim == 2:
            valid_slice = frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                padded = cv2.copyMakeBorder(
                    valid_slice,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_REFLECT_101,
                )
            else:
                padded = valid_slice.copy()
            return padded

        if frame.ndim == 3:
            valid_slice = frame[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
            if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                padded = cv2.copyMakeBorder(
                    valid_slice,
                    pad_top,
                    pad_bottom,
                    pad_left,
                    pad_right,
                    cv2.BORDER_REFLECT_101,
                )
            else:
                padded = valid_slice.copy()
            return padded

        if frame.ndim == 4:
            # Sequence: [T, H, W, C]
            slices = []
            for t in range(frame.shape[0]):
                f = frame[t]
                valid_slice = f[valid_y_start:valid_y_end, valid_x_start:valid_x_end]
                if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                    p_t = cv2.copyMakeBorder(
                        valid_slice,
                        pad_top,
                        pad_bottom,
                        pad_left,
                        pad_right,
                        cv2.BORDER_REFLECT_101,
                    )
                else:
                    p_t = valid_slice.copy()
                slices.append(p_t)
            return np.stack(slices, axis=0)

        raise ValueError(f"Unsupported frame dimensionality: {frame.ndim}")

    @staticmethod
    def crop_context_padding(
        tile_output: np.ndarray,
        coord: TileCoordinate,
        context_padding: int | None = None,
    ) -> np.ndarray:
        """Crop away context padding from tile prediction to restore core tile geometry.

        Args:
            tile_output: Neural model output array (2D [H, W], 3D [C, H, W] or [H, W, C]).
            coord: Corresponding TileCoordinate with core height and width.
            context_padding: Extra border size to remove.

        Returns:
            Cropped array of exact spatial dimensions `(coord.height, coord.width)`.
        """
        p = context_padding if context_padding is not None else coord.pad_top
        h_core = coord.height
        w_core = coord.width

        if p == 0:
            if tile_output.ndim == 2:
                return tile_output[:h_core, :w_core]
            if tile_output.ndim == 3:
                if tile_output.shape[0] <= 16 and tile_output.shape[1] >= h_core:
                    # [C, H, W]
                    return tile_output[:, :h_core, :w_core]
                # [H, W, C]
                return tile_output[:h_core, :w_core, :]
            return tile_output

        if tile_output.ndim == 2:
            return tile_output[p : p + h_core, p : p + w_core]

        if tile_output.ndim == 3:
            # Check whether channel is first or last
            if tile_output.shape[0] <= 16 and tile_output.shape[1] >= (h_core + 2 * p):
                # [C, H, W]
                return tile_output[:, p : p + h_core, p : p + w_core]
            # [H, W, C]
            return tile_output[p : p + h_core, p : p + w_core, :]

        if tile_output.ndim == 4:
            # [T, C, H, W] or [T, H, W, C]
            if tile_output.shape[1] <= 16:
                return tile_output[:, :, p : p + h_core, p : p + w_core]
            return tile_output[:, p : p + h_core, p : p + w_core, :]

        raise ValueError(f"Unsupported tile_output ndim: {tile_output.ndim}")

    def blend_probability_maps(
        self,
        tiles: list[np.ndarray],
        coordinates: list[TileCoordinate],
        target_shape: tuple[int, int],
        blend_mode: str | None = None,
    ) -> np.ndarray:
        """Blend overlapping tile probability maps using continuous 2D window weighting.

        Weights are accumulated prior to binarization and normalized by the total
        accumulated weight map. Guarantees output is float32 normalized in [0.0, 1.0]
        with zero border seam artifacts.

        Args:
            tiles: List of cropped tile probability maps matching `coordinates`.
            coordinates: Tile coordinates specifying spatial locations in target frame.
            target_shape: Target full-frame (height, width).
            blend_mode: Optional blend weighting mode ("gaussian", "distance_linear", "hanning").

        Returns:
            Seamless blended float32 ndarray with values strictly bounded in [0.0, 1.0].
        """
        if not tiles:
            raise ValueError("tiles list must not be empty")
        if len(tiles) != len(coordinates):
            raise ValueError(
                f"Mismatch between number of tiles ({len(tiles)}) and coordinates ({len(coordinates)})"
            )

        target_h, target_w = int(target_shape[0]), int(target_shape[1])
        mode = blend_mode or self.config.blend_mode

        first_tile = tiles[0]
        # Detect channel orientation
        is_channel_first = False
        is_channel_last = False
        num_channels = 1

        if first_tile.ndim == 3:
            if first_tile.shape[0] <= 16:
                is_channel_first = True
                num_channels = first_tile.shape[0]
            else:
                is_channel_last = True
                num_channels = first_tile.shape[2]

        if is_channel_first:
            accum_prob = np.zeros((num_channels, target_h, target_w), dtype=np.float32)
            accum_weight = np.zeros((1, target_h, target_w), dtype=np.float32)
        elif is_channel_last:
            accum_prob = np.zeros((target_h, target_w, num_channels), dtype=np.float32)
            accum_weight = np.zeros((target_h, target_w, 1), dtype=np.float32)
        else:
            accum_prob = np.zeros((target_h, target_w), dtype=np.float32)
            accum_weight = np.zeros((target_h, target_w), dtype=np.float32)

        weight_cache: dict[tuple[int, int], np.ndarray] = {}

        for tile, coord in zip(tiles, coordinates):
            th, tw = coord.height, coord.width
            key = (th, tw)
            if key not in weight_cache:
                weight_cache[key] = create_blend_weight(key, mode)
            w2d = weight_cache[key]

            y, x = coord.y, coord.x

            if is_channel_first:
                accum_prob[:, y : y + th, x : x + tw] += tile.astype(np.float32) * w2d[None, :, :]
                accum_weight[0, y : y + th, x : x + tw] += w2d
            elif is_channel_last:
                accum_prob[y : y + th, x : x + tw, :] += tile.astype(np.float32) * w2d[:, :, None]
                accum_weight[y : y + th, x : x + tw, 0] += w2d
            else:
                accum_prob[y : y + th, x : x + tw] += tile.astype(np.float32) * w2d
                accum_weight[y : y + th, x : x + tw] += w2d

        # Normalize by accumulated weights
        denom = np.maximum(accum_weight, 1e-7)
        blended = accum_prob / denom
        return np.clip(blended, 0.0, 1.0).astype(np.float32)

    def run_tiled_inference(
        self,
        frame: np.ndarray,
        infer_fn: Callable[[np.ndarray], np.ndarray],
        config: AdaptiveTilingConfig | None = None,
    ) -> np.ndarray:
        """Orchestrate tile extraction, context padding, batched inference, crop, and blending.

        Args:
            frame: Input image array [H, W] or [H, W, C].
            infer_fn: Callable executing model prediction on a tile patch or batch of patches.
            config: Optional AdaptiveTilingConfig override.

        Returns:
            Full-resolution float32 blended probability map normalized in [0.0, 1.0].
        """
        cfg = config or self.config
        target_h, target_w = frame.shape[:2]
        coords = self.generate_tiles((target_h, target_w), config=cfg)

        p = cfg.context_padding
        tile_h, tile_w = cfg.tile_size
        padded_spatial = (tile_h + 2 * p, tile_w + 2 * p)

        effective_bs, _ = self.vram_monitor.determine_batch_size(
            cfg.batch_size,
            padded_spatial,
            cfg.vram_budget_mb,
            device=cfg.device,
        )

        # 1. Extract padded patches
        padded_patches = [self.extract_padded_patch(frame, coord, p) for coord in coords]

        # 2. Run batched inference
        predicted_patches: list[np.ndarray] = []
        n_tiles = len(padded_patches)

        for i in range(0, n_tiles, effective_bs):
            chunk = padded_patches[i : i + effective_bs]
            # Attempt batched evaluation if multiple tiles
            executed = False
            if len(chunk) > 1:
                try:
                    batch_in = np.stack(chunk, axis=0)
                    batch_out = infer_fn(batch_in)
                    if isinstance(batch_out, np.ndarray) and batch_out.shape[0] == len(chunk):
                        for b_idx in range(len(chunk)):
                            predicted_patches.append(batch_out[b_idx])
                        executed = True
                    elif isinstance(batch_out, (list, tuple)) and len(batch_out) == len(chunk):
                        predicted_patches.extend(list(batch_out))
                        executed = True
                except Exception as exc:
                    logger.debug("Batched inference failed, falling back to per-tile inference: %s", exc)

            if not executed:
                for patch in chunk:
                    predicted_patches.append(infer_fn(patch))

        # 3. Crop context padding from each prediction
        cropped_tiles = [
            self.crop_context_padding(pred, coord, p)
            for pred, coord in zip(predicted_patches, coords)
        ]

        # 4. Blend probability maps
        return self.blend_probability_maps(cropped_tiles, coords, (target_h, target_w), blend_mode=cfg.blend_mode)

    def run_tiled_inference_sequence(
        self,
        frames: Sequence[np.ndarray],
        infer_fn: Callable[[list[np.ndarray]], Sequence[np.ndarray]],
        config: AdaptiveTilingConfig | None = None,
    ) -> list[np.ndarray]:
        """Execute sequence tiled inference with locked temporal grid coordinates across frames.

        Args:
            frames: Sequence of input frames of identical geometry [H, W] or [H, W, C].
            infer_fn: Temporal model inference callable accepting a list of padded tile frames
                      and returning a list of tile probability outputs for each frame.
            config: Optional AdaptiveTilingConfig override.

        Returns:
            List of full-resolution float32 blended probability maps for each frame.
        """
        if not frames:
            return []

        cfg = config or self.config
        target_h, target_w = frames[0].shape[:2]
        coords = self.generate_tiles((target_h, target_w), config=cfg)
        p = cfg.context_padding
        n_frames = len(frames)

        # For each frame, we maintain a list of cropped tiles
        frame_tiles: list[list[np.ndarray]] = [[] for _ in range(n_frames)]

        for coord in coords:
            # Extract corresponding tile from each frame
            tile_frames = [self.extract_padded_patch(f, coord, p) for f in frames]
            tile_outputs = infer_fn(tile_frames)
            for t_idx, tile_out in enumerate(tile_outputs):
                cropped = self.crop_context_padding(tile_out, coord, p)
                frame_tiles[t_idx].append(cropped)

        # Blend each frame independently using locked coordinates
        blended_frames = [
            self.blend_probability_maps(
                frame_tiles[t_idx],
                coords,
                (target_h, target_w),
                blend_mode=cfg.blend_mode,
            )
            for t_idx in range(n_frames)
        ]
        return blended_frames


__all__ = [
    "TileCoordinate",
    "AdaptiveTilingConfig",
    "VRAMMonitor",
    "create_blend_weight",
    "AdaptiveTilingEngine",
]
