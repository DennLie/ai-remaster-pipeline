"""Conservative, CPU-based temporal film-defect masks without AI model weights.

Per-frame masks mark transient dust/blotches and short scratches. Repeated
scratches and torn regions need separate review. This code does not inpaint
and does not depend on DART, RTN, ProPainter, or generative checkpoints.

Pixel convention: uint8 grayscale, 255 = suspect defect, 0 = keep original.
"""
from __future__ import annotations

from typing import Any


MASK_VERSION = 1
MAX_MASK_FRACTION = 0.025


def to_gray(frame: Any) -> Any:
    import cv2
    import numpy as np

    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Expected grayscale or BGR frame")
    if gray.dtype != np.uint8:
        raise ValueError("Expected uint8 video frames")
    return gray


def aligned_neighbor(reference: Any, neighbor: Any) -> Any:
    """Warp a neighbor into reference-frame coordinates using optical flow."""
    import cv2
    import numpy as np

    h, w = reference.shape
    flow = cv2.calcOpticalFlowFarneback(
        reference, neighbor, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0,
    )
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        neighbor, x + flow[..., 0], y + flow[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def temporal_defect_mask(previous: Any, current: Any, following: Any, sensitivity: float = 0.65) -> Any:
    """Locate isolated single-frame damage, preferring false negatives to lost detail.

    Compare the middle frame with *both* motion-compensated neighbors;
    rejection on scene cuts / occlusions avoids many moving-object artifacts.
    Adjacent frames remain unmodified, and output is a binary proposal only.
    """
    import cv2
    import numpy as np

    if not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("Sensitivity must be between 0 and 1")
    a, b, c = (to_gray(f) for f in (previous, current, following))
    if a.shape != b.shape or b.shape != c.shape:
        raise ValueError("All temporal frames must have matching dimensions")
    h, w = b.shape
    zero = np.zeros_like(b)
    if h < 16 or w < 16:
        return zero

    # Never interpolate a cut, strong fade, or scene transition.
    if (float(np.median(cv2.absdiff(a, b))) > 35
            or float(np.median(cv2.absdiff(b, c))) > 35):
        return zero

    before = aligned_neighbor(b, a).astype(np.float32)
    after = aligned_neighbor(b, c).astype(np.float32)
    centre = b.astype(np.float32)
    target = (before + after) / 2.0
    mismatch = np.abs(centre - target)
    neighbor_disagreement = np.abs(before - after)

    # Robust global noise floor: no mask on ordinary film grain.
    median_difference = float(np.median(mismatch))
    mad = float(np.median(np.abs(mismatch - median_difference)))
    threshold = max(18.0, median_difference + (7.0 - 3.0 * sensitivity) * max(mad, 1.2))
    reference_agreement = neighbor_disagreement <= max(12.0, threshold * 0.70)
    temporal = mismatch >= threshold

    # Impulse-like damage is also distinct from spatially adjacent material.
    # The second test rejects soft motion boundaries and exposure pumping.
    spatial = np.abs(centre - cv2.medianBlur(b, 5).astype(np.float32))
    local_threshold = max(11.0, threshold * (0.47 - 0.12 * sensitivity))
    likely = temporal & reference_agreement & (spatial >= local_threshold)

    # Presentation bars and outer registration boundaries are never selected.
    picture = b >= max(8, int(np.percentile(b, 98) * 0.025))
    likely &= picture
    likely[:2, :] = False
    likely[-2:, :] = False
    likely[:, :2] = False
    likely[:, -2:] = False

    rough = likely.astype(np.uint8) * 255
    total, labels, stats, _centroids = cv2.connectedComponentsWithStats(rough, connectivity=8)
    result = zero.copy()
    max_blob_area = max(8, min(180, int(round(h * w * 0.00035))))
    max_blob_edge = max(7, min(25, int(round(min(h, w) * 0.028))))
    for component in range(1, total):
        _x, _y, component_w, component_h, area = map(int, stats[component])
        small_spot = area <= max_blob_area and component_w <= max_blob_edge and component_h <= max_blob_edge
        fine_scratch = (
            component_w <= max(3, int(w * 0.009))
            and component_h >= 10
            and component_h <= max(12, int(h * 0.60))
            and area <= max(40, int(h * 0.75))
        )
        if small_spot or fine_scratch:
            result[labels == component] = 255

    # If registration failed, abstain rather than propose extensive repairs.
    if np.count_nonzero(result) / result.size > MAX_MASK_FRACTION:
        return zero
    return result


def defect_mask_size(width: int, height: int, selected_height: str) -> tuple[int, int]:
    """Use the source size or a smaller even-sized mask analysis canvas."""
    if str(selected_height) == "source":
        return width, height
    target_h = min(height, max(16, int(selected_height)))
    target_h -= target_h % 2
    target_w = max(16, int(round(width * target_h / height)))
    target_w -= target_w % 2
    return target_w, target_h
