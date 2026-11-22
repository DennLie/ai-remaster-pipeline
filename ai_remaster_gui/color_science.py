from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Standard sRGB D65 to CIE-XYZ conversion matrix
SRGB_TO_XYZ_MATRIX: np.ndarray = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

# Standard CIE 1931 D65 reference white point (2° standard observer)
D65_WHITE: np.ndarray = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# CIE L*a*b* standard constants
_CIE_DELTA: float = 6.0 / 29.0
_CIE_DELTA_CUBE: float = _CIE_DELTA**3  # (6/29)^3 ≈ 0.00885645
_CIE_T_DIV: float = 3.0 * (_CIE_DELTA**2)  # 3 * (6/29)^2 ≈ 0.12841855
_CIE_T_ADD: float = 4.0 / 29.0  # ≈ 0.13793103


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB array to linear RGB using standard piecewise inverse companding.

    Accepts uint8 values in [0, 255] or float values in [0.0, 1.0].
    Returns float64 array with linear values in [0.0, 1.0].
    """
    arr = np.asarray(rgb, dtype=np.float64)
    if np.issubdtype(rgb.dtype, np.integer) or (arr.size > 0 and np.nanmax(arr) > 1.0):
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return np.where(arr <= 0.04045, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)


def linear_to_xyz(rgb_linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB to CIE 1931 XYZ using the sRGB D65 reference matrix."""
    arr = np.asarray(rgb_linear, dtype=np.float64)
    return arr @ SRGB_TO_XYZ_MATRIX.T


def xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """Convert CIE 1931 XYZ to CIE L*a*b* normalized against standard D65 illuminant."""
    arr = np.asarray(xyz, dtype=np.float64)
    norm = arr / D65_WHITE

    f_t = np.where(
        norm > _CIE_DELTA_CUBE,
        np.cbrt(np.maximum(norm, 0.0)),
        (norm / _CIE_T_DIV) + _CIE_T_ADD,
    )

    fx = f_t[..., 0]
    fy = f_t[..., 1]
    fz = f_t[..., 2]

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def srgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 [0, 255] or float [0, 1] RGB to CIE L*a*b* array."""
    linear = srgb_to_linear(rgb_uint8)
    xyz = linear_to_xyz(linear)
    return xyz_to_lab(xyz)


def load_image_rgb(path: Path | str) -> np.ndarray:
    """Open an image via PIL, convert to RGB mode, and return as (H, W, 3) uint8 array."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def load_mask(path: Path | str | None) -> np.ndarray | None:
    """Open a mask image via PIL, convert to grayscale, and return a 2D boolean mask.

    Pixels with intensity > 127 are considered foreground (True).
    If path is None or does not exist, returns None.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with Image.open(p) as img:
        gray = img.convert("L")
        arr = np.array(gray, dtype=np.uint8)
        return arr > 127


def filter_pixels_by_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return an (N, 3) array of RGB pixels inside the mask.

    If mask is None, empty, or contains zero foreground pixels, falls back
    safely to all pixels in the image.
    """
    if image_rgb.size == 0:
        return np.empty((0, 3), dtype=image_rgb.dtype)

    if image_rgb.ndim == 3:
        h, w, _ = image_rgb.shape
        if mask is not None and mask.size > 0:
            m = mask
            if m.ndim == 3:
                m = m[..., 0]
            fg = m > 127 if m.dtype == np.uint8 else m.astype(bool)
            if fg.shape == (h, w) and np.count_nonzero(fg) > 0:
                return image_rgb[fg]
        return image_rgb.reshape(-1, 3)

    if image_rgb.ndim == 2 and image_rgb.shape[1] == 3:
        if mask is not None and mask.size > 0:
            m = mask
            if m.ndim > 1:
                m = m.flatten()
            fg = m > 127 if m.dtype == np.uint8 else m.astype(bool)
            if fg.shape[0] == image_rgb.shape[0] and np.count_nonzero(fg) > 0:
                return image_rgb[fg]
        return image_rgb

    return image_rgb.reshape(-1, 3)


def compute_lab_percentiles(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    percentiles: list[float] | None = None,
) -> dict[str, list[float]]:
    """Compute CIE L*a*b* percentiles for an image, optionally restricted by mask.

    Defaults to [10.0, 25.0, 50.0, 75.0, 90.0] percentiles, rounded to 2 decimal places.
    Returns: {"L": [...], "a": [...], "b": [...]}
    """
    if percentiles is None:
        percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]

    pixels = filter_pixels_by_mask(image_rgb, mask)
    if pixels.size == 0:
        return {
            "L": [0.0] * len(percentiles),
            "a": [0.0] * len(percentiles),
            "b": [0.0] * len(percentiles),
        }

    lab = srgb_to_lab(pixels)
    return {
        "L": [round(float(v), 2) for v in np.percentile(lab[:, 0], percentiles)],
        "a": [round(float(v), 2) for v in np.percentile(lab[:, 1], percentiles)],
        "b": [round(float(v), 2) for v in np.percentile(lab[:, 2], percentiles)],
    }


def extract_rgb_palette(
    image_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    num_colors: int = 6,
) -> list[dict[str, Any]]:
    """Extract a dominant deterministic RGB palette from an image, optionally masked.

    Quantizes pixels deterministically, computes percentage share of each cluster,
    and returns a list sorted by percentage descending:
    [{"hex": "#rrggbb", "rgb": [r, g, b], "percentage": round(pct, 4)}]
    """
    if num_colors <= 0:
        return []

    pixels = filter_pixels_by_mask(image_rgb, mask)
    if pixels.size == 0:
        return []

    # If uint8 or float, normalize to uint8 for palette quantization
    if np.issubdtype(pixels.dtype, np.floating):
        if np.nanmax(pixels) <= 1.0:
            pixels = np.clip(np.round(pixels * 255.0), 0, 255).astype(np.uint8)
        else:
            pixels = np.clip(np.round(pixels), 0, 255).astype(np.uint8)
    else:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    total_pixels = len(pixels)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    # If unique colors are fewer than or equal to requested count, return exact distribution
    if len(unique_colors) <= num_colors:
        order = np.argsort(-counts)
        palette: list[dict[str, Any]] = []
        for idx in order:
            c = unique_colors[idx].astype(int).tolist()
            cnt = int(counts[idx])
            pct = round(cnt / total_pixels, 4)
            h = f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"
            palette.append({"hex": h, "rgb": c, "percentage": pct})
        return palette

    # For continuous/photographic images, use deterministic PIL median-cut quantization
    clamped_num = min(num_colors, 256)
    img = Image.fromarray(pixels.reshape(1, -1, 3))
    q = img.quantize(colors=clamped_num, method=Image.Quantize.MEDIANCUT)
    labels = np.array(q).flatten()

    unique_labels, label_counts = np.unique(labels, return_counts=True)
    order = np.argsort(-label_counts)

    palette = []
    for label_idx in order:
        lbl = unique_labels[label_idx]
        cnt = int(label_counts[label_idx])
        pct = round(cnt / total_pixels, 4)
        cluster_pixels = pixels[labels == lbl]
        mean_rgb = np.round(cluster_pixels.mean(axis=0)).astype(int).clip(0, 255).tolist()
        h = f"#{mean_rgb[0]:02x}{mean_rgb[1]:02x}{mean_rgb[2]:02x}"
        palette.append({"hex": h, "rgb": mean_rgb, "percentage": pct})

    return palette
