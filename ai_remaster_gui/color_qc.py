from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from .color_science import srgb_to_lab
from .rle_masks import MaskObjectMetadata, ShotMaskArchive


def ciede2000(
    lab1: np.ndarray | Sequence[float],
    lab2: np.ndarray | Sequence[float],
    k_L: float = 1.0,
    k_C: float = 1.0,
    k_H: float = 1.0,
    normalize_lightness: bool = False,
) -> np.ndarray | float:
    """Compute CIEDE2000 total color difference between CIE L*a*b* coordinates.

    Implementation based on:
    Sharma, Wu, Dalal (2005): "The CIEDE2000 Color-Difference Formula:
    Implementation Notes, Supplementary Test Data, and Mathematical Observations",
    Color Research & Application, 30(1): 21-30.

    Parameters
    ----------
    lab1 : array_like
        Reference color(s) CIE L*a*b*. Can be 1D (3,), 2D (N, 3), or N-D (..., 3).
    lab2 : array_like
        Test color(s) CIE L*a*b*. Must be broadcastable with `lab1`.
    k_L : float
        Parametric weighting factor for lightness difference (default 1.0).
    k_C : float
        Parametric weighting factor for chroma difference (default 1.0).
    k_H : float
        Parametric weighting factor for hue difference (default 1.0).
    normalize_lightness : bool
        When True, forces L_test* = L_ref* (i.e. Delta L' = 0), so the distance
        strictly measures perceptual chromatic/hue deviation (Delta C', Delta H'),
        isolating color drift from intentional scene lighting or shadow shifts.

    Returns
    -------
    np.ndarray or float
        CIEDE2000 color difference Delta E_00. Returns float for 1D (3,) inputs,
        or np.ndarray with shape matching broadcasted batch dimensions (...,).
    """
    arr1 = np.asarray(lab1, dtype=np.float64)
    arr2 = np.asarray(lab2, dtype=np.float64)

    is_1d = (arr1.ndim == 1 and arr1.shape[0] == 3 and arr2.ndim == 1 and arr2.shape[0] == 3)

    L1, a1, b1 = arr1[..., 0], arr1[..., 1], arr1[..., 2]
    L2, a2, b2 = arr2[..., 0], arr2[..., 1], arr2[..., 2]

    if normalize_lightness:
        # Match shapes for broadcasting if needed
        b_shape = np.broadcast_shapes(np.shape(L1), np.shape(L2))
        L1 = np.broadcast_to(L1, b_shape)
        L2 = L1.copy()

    # Step 1: Calculate C' and a'
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_bar = 0.5 * (C1 + C2)

    C_bar7 = C_bar**7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0**7)))

    a1_prime = (1.0 + G) * a1
    a2_prime = (1.0 + G) * a2

    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)

    # Calculate h_prime in degrees in [0, 360)
    h1_prime = np.rad2deg(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.rad2deg(np.arctan2(b2, a2_prime)) % 360.0

    # Handle C' == 0 where h' = 0
    h1_prime = np.where(C1_prime == 0.0, 0.0, h1_prime)
    h2_prime = np.where(C2_prime == 0.0, 0.0, h2_prime)

    # Step 2: Calculate Delta L', Delta C', Delta H'
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime

    # Delta h_prime calculation with proper wrap-around
    h_diff = h2_prime - h1_prime
    abs_h_diff = np.abs(h_diff)

    delta_h_prime = np.zeros_like(h_diff)
    mask_c = (C1_prime * C2_prime) != 0.0

    m1 = mask_c & (abs_h_diff <= 180.0)
    delta_h_prime = np.where(m1, h_diff, delta_h_prime)

    m2 = mask_c & (h_diff > 180.0)
    delta_h_prime = np.where(m2, h_diff - 360.0, delta_h_prime)

    m3 = mask_c & (h_diff < -180.0)
    delta_h_prime = np.where(m3, h_diff + 360.0, delta_h_prime)

    # Delta H_prime = 2 * sqrt(C1' * C2') * sin(Delta h' / 2)
    delta_H_prime = 2.0 * np.sqrt(C1_prime * C2_prime) * np.sin(np.deg2rad(0.5 * delta_h_prime))

    # Step 3: Calculate CIEDE2000 total color difference
    L_bar_prime = 0.5 * (L1 + L2)
    C_bar_prime = 0.5 * (C1_prime + C2_prime)

    h_sum = h1_prime + h2_prime
    h_bar_prime = np.zeros_like(h_diff)

    # When C1' * C2' == 0: h_bar_prime = h1' + h2'
    h_bar_prime = np.where(~mask_c, h_sum, h_bar_prime)

    m_h1 = mask_c & (abs_h_diff <= 180.0)
    h_bar_prime = np.where(m_h1, 0.5 * h_sum, h_bar_prime)

    m_h2 = mask_c & (abs_h_diff > 180.0) & (h_sum < 360.0)
    h_bar_prime = np.where(m_h2, 0.5 * (h_sum + 360.0), h_bar_prime)

    m_h3 = mask_c & (abs_h_diff > 180.0) & (h_sum >= 360.0)
    h_bar_prime = np.where(m_h3, 0.5 * (h_sum - 360.0), h_bar_prime)

    T = (
        1.0
        - 0.17 * np.cos(np.deg2rad(h_bar_prime - 30.0))
        + 0.24 * np.cos(np.deg2rad(2.0 * h_bar_prime))
        + 0.32 * np.cos(np.deg2rad(3.0 * h_bar_prime + 6.0))
        - 0.20 * np.cos(np.deg2rad(4.0 * h_bar_prime - 63.0))
    )

    delta_theta = 30.0 * np.exp(-(((h_bar_prime - 275.0) / 25.0) ** 2))

    C_bar_prime7 = C_bar_prime**7
    R_C = 2.0 * np.sqrt(C_bar_prime7 / (C_bar_prime7 + 25.0**7))

    S_L = 1.0 + (0.015 * (L_bar_prime - 50.0) ** 2) / np.sqrt(20.0 + (L_bar_prime - 50.0) ** 2)
    S_C = 1.0 + 0.045 * C_bar_prime
    S_H = 1.0 + 0.015 * C_bar_prime * T
    R_T = -np.sin(np.deg2rad(2.0 * delta_theta)) * R_C

    term_L = delta_L_prime / (k_L * S_L)
    term_C = delta_C_prime / (k_C * S_C)
    term_H = delta_H_prime / (k_H * S_H)

    dE2 = term_L**2 + term_C**2 + term_H**2 + R_T * term_C * term_H
    dE = np.sqrt(np.maximum(dE2, 0.0))

    if is_1d:
        return float(dE.item())
    return dE


def compute_cdc(frames_rgb: Sequence[np.ndarray], stride: int = 1) -> float:
    """Compute Color Distribution Consistency (CDC) across strided frames.

    Uses normalized 3D color histogram intersection across consecutive strided
    frames. Returns a consistency score in [0.0, 1.0] where 1.0 is perfectly consistent.

    Parameters
    ----------
    frames_rgb : Sequence[np.ndarray]
        Sequence of (H, W, 3) RGB uint8 image arrays.
    stride : int
        Temporal sampling stride (default 1).

    Returns
    -------
    float
        CDC score between 0.0 (completely inconsistent) and 1.0 (identical color distribution).
    """
    if len(frames_rgb) < 2:
        return 1.0

    stride = max(1, int(stride))
    strided = frames_rgb[::stride]
    if len(strided) < 2:
        return 1.0

    channels = [0, 1, 2]
    hist_size = [16, 16, 16]
    ranges = [0, 256, 0, 256, 0, 256]

    scores: list[float] = []
    prev_hist: np.ndarray | None = None

    for frame in strided:
        if frame.dtype != np.uint8:
            frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_uint8 = frame

        hist = cv2.calcHist([frame_uint8], channels, None, hist_size, ranges)
        cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)

        if prev_hist is not None:
            score = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_INTERSECT))
            scores.append(float(np.clip(score, 0.0, 1.0)))

        prev_hist = hist

    return float(np.mean(scores)) if scores else 1.0


def compute_optical_flow_chroma_error(
    frames_gray: Sequence[np.ndarray],
    frames_lab: Sequence[np.ndarray],
    stride: int = 1,
    preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST,
) -> float:
    """Compute mean optical flow chroma warping error across strided frames.

    Calculates grayscale OpenCV DIS optical flow backward warping chroma (a*, b*)
    from frame t-1 to frame t, computing the mean residual Euclidean distance
    sqrt(Delta a^2 + Delta b^2) on non-occluded / valid warped pixels.

    Parameters
    ----------
    frames_gray : Sequence[np.ndarray]
        Sequence of 2D uint8 grayscale frames.
    frames_lab : Sequence[np.ndarray]
        Sequence of (H, W, 3) float64 CIE L*a*b* frames.
    stride : int
        Temporal sampling stride (default 1).
    preset : int
        OpenCV DISOpticalFlow preset (default cv2.DISOPTICAL_FLOW_PRESET_FAST).

    Returns
    -------
    float
        Mean residual Euclidean chroma error in a*b* units across frame pairs.
    """
    if len(frames_gray) < 2 or len(frames_lab) < 2:
        return 0.0

    stride = max(1, int(stride))
    strided_gray = frames_gray[::stride]
    strided_lab = frames_lab[::stride]
    n = min(len(strided_gray), len(strided_lab))
    if n < 2:
        return 0.0

    dis = cv2.DISOpticalFlow_create(preset)
    errors: list[float] = []

    h, w = strided_gray[0].shape[:2]
    x_coords, y_coords = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )

    for i in range(1, n):
        prev_gray = strided_gray[i - 1]
        curr_gray = strided_gray[i]
        prev_lab = strided_lab[i - 1]
        curr_lab = strided_lab[i]

        # Backward optical flow from curr to prev: mapping each curr pixel (x, y)
        # to displacement (dx, dy) in prev frame
        flow = dis.calc(curr_gray, prev_gray, None)
        map_x = x_coords + flow[..., 0]
        map_y = y_coords + flow[..., 1]

        valid_mask = (map_x >= 0) & (map_x <= (w - 1)) & (map_y >= 0) & (map_y <= (h - 1))
        if not np.any(valid_mask):
            continue

        prev_chroma = prev_lab[..., 1:3].astype(np.float32)
        curr_chroma = curr_lab[..., 1:3].astype(np.float32)

        warped_chroma = cv2.remap(
            prev_chroma,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        diff = curr_chroma - warped_chroma
        residual = np.sqrt(diff[..., 0] ** 2 + diff[..., 1] ** 2)
        mean_err = float(np.mean(residual[valid_mask]))
        errors.append(mean_err)

    return float(np.mean(errors)) if errors else 0.0


def detect_hue_jumps(
    frames_lab: Sequence[np.ndarray],
    threshold_degrees: float = 25.0,
    stride: int = 1,
) -> list[int]:
    """Detect sudden spikes in frame-to-frame mean hue angle across strided frames.

    Parameters
    ----------
    frames_lab : Sequence[np.ndarray]
        Sequence of (H, W, 3) float64 CIE L*a*b* frames.
    threshold_degrees : float
        Angular jump threshold in degrees (default 25.0).
    stride : int
        Temporal sampling stride (default 1).

    Returns
    -------
    list[int]
        Indices of frames in `frames_lab` where a hue jump was detected.
    """
    if len(frames_lab) < 2:
        return []

    stride = max(1, int(stride))
    sampled_indices = list(range(0, len(frames_lab), stride))
    if len(sampled_indices) < 2:
        return []

    mean_hues: list[float] = []
    chroma_mags: list[float] = []

    for idx in sampled_indices:
        f = frames_lab[idx]
        a = f[..., 1]
        b = f[..., 2]
        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        mag = float(np.hypot(mean_a, mean_b))
        chroma_mags.append(mag)
        hue = float(np.rad2deg(np.arctan2(mean_b, mean_a)) % 360.0)
        mean_hues.append(hue)

    jumps: list[int] = []
    for i in range(1, len(mean_hues)):
        # If both consecutive frames are near-neutral grayscale, hue angle is noisy/undefined
        if chroma_mags[i - 1] < 1.0 and chroma_mags[i] < 1.0:
            continue

        h1 = mean_hues[i - 1]
        h2 = mean_hues[i]
        diff = abs(h2 - h1) % 360.0
        angular_diff = min(diff, 360.0 - diff)
        if angular_diff > threshold_degrees:
            jumps.append(sampled_indices[i])

    return jumps


def detect_mask_bleed(
    frame_lab: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    threshold: float = 8.0,
) -> float:
    """Measure chromatic spill of a foreground object into its surrounding background.

    Compares the immediate boundary ring M_near against a farther reference ring M_far
    and the interior object M_in in a*b* chroma space. Computes the projection of
    (C_near - C_far) along the unit vector from C_far to C_in.

    Parameters
    ----------
    frame_lab : np.ndarray
        (H, W, 3) float64 CIE L*a*b* frame.
    mask : np.ndarray
        2D binary or uint8 object mask of shape (H, W).
    radius : int
        Near boundary dilation radius in pixels (default 3). Far ring uses 3*radius.
    threshold : float
        Spill threshold in chroma units (default 8.0).

    Returns
    -------
    float
        Chromatic spill distance in a*b* units toward the object chroma.
        Returns 0.0 if no spill or no valid background pixels exist.
    """
    if mask is None or mask.size == 0 or frame_lab.size == 0:
        return 0.0

    if mask.ndim == 3:
        mask = mask[..., 0]

    binary_mask = (mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(binary_mask) == 0:
        return 0.0

    r_near = max(1, int(radius))
    r_far = max(r_near + 1, int(3 * radius))

    k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_near + 1, 2 * r_near + 1))
    k_far = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_far + 1, 2 * r_far + 1))

    dilated_near = cv2.dilate(binary_mask, k_near)
    dilated_far = cv2.dilate(binary_mask, k_far)

    m_in = binary_mask > 0
    m_near = (dilated_near > 0) & (~m_in)
    m_far = (dilated_far > 0) & (~(dilated_near > 0))

    if not np.any(m_near) or not np.any(m_far) or not np.any(m_in):
        return 0.0

    a = frame_lab[..., 1].astype(np.float64)
    b = frame_lab[..., 2].astype(np.float64)

    c_in = np.array([np.mean(a[m_in]), np.mean(b[m_in])])
    c_near = np.array([np.mean(a[m_near]), np.mean(b[m_near])])
    c_far = np.array([np.mean(a[m_far]), np.mean(b[m_far])])

    v = c_in - c_far
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-6:
        return 0.0

    u = v / v_norm
    w = c_near - c_far
    spill = float(np.dot(w, u))
    return max(0.0, spill)


@dataclass
class ObjectQCScore:
    """Detailed Color QC metrics and status for an individual object."""

    mean_lab: tuple[float, float, float]
    p10_lab: tuple[float, float, float]
    p50_lab: tuple[float, float, float]
    p90_lab: tuple[float, float, float]
    raw_delta_e: float
    chroma_delta_e: float
    status: str  # 'pass' | 'warn' | 'fail'
    failure_codes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_lab": [round(x, 2) for x in self.mean_lab],
            "p10_lab": [round(x, 2) for x in self.p10_lab],
            "p50_lab": [round(x, 2) for x in self.p50_lab],
            "p90_lab": [round(x, 2) for x in self.p90_lab],
            "raw_delta_e": round(self.raw_delta_e, 3),
            "chroma_delta_e": round(self.chroma_delta_e, 3),
            "status": self.status,
            "failure_codes": list(self.failure_codes),
            "metadata": self.metadata,
        }


def extract_reference_lab(ref_lab_stats: dict[str, Any]) -> tuple[float, float, float]:
    """Extract (L*, a*, b*) reference coordinates from various Color Bible dictionary layouts."""
    if "mean_lab" in ref_lab_stats and isinstance(ref_lab_stats["mean_lab"], (list, tuple)):
        m = ref_lab_stats["mean_lab"]
        return float(m[0]), float(m[1]), float(m[2])

    stats = ref_lab_stats.get("lab_percentiles", ref_lab_stats)

    def _val(ch: str) -> float:
        v = stats.get(ch, 0.0)
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return 0.0
            return float(v[len(v) // 2])
        return float(v)

    return _val("L"), _val("a"), _val("b")


def evaluate_object_color(
    object_pixels_lab: np.ndarray,
    ref_lab_stats: dict[str, Any],
    thresholds: dict[str, float] | None = None,
) -> ObjectQCScore:
    """Evaluate an object's CIE L*a*b* distribution against reference Color Bible statistics.

    Computes mean and percentiles, evaluating both raw CIEDE2000 and reference-pinned
    lightness-normalized CIEDE2000.

    Parameters
    ----------
    object_pixels_lab : np.ndarray
        (N, 3) array of CIE L*a*b* coordinates for the object.
    ref_lab_stats : dict[str, Any]
        Reference Color Bible statistics or percentiles dictionary.
    thresholds : dict[str, float], optional
        Threshold overrides for 'chroma_delta_e_pass' (default 5.0) and
        'chroma_delta_e_warn' (default 8.0).

    Returns
    -------
    ObjectQCScore
        Object QC score containing CIEDE2000 metrics and status.
    """
    th = thresholds or {}
    pass_thresh = float(th.get("chroma_delta_e_pass", 5.0))
    warn_thresh = float(th.get("chroma_delta_e_warn", 8.0))

    if object_pixels_lab is None or len(object_pixels_lab) == 0:
        return ObjectQCScore(
            mean_lab=(0.0, 0.0, 0.0),
            p10_lab=(0.0, 0.0, 0.0),
            p50_lab=(0.0, 0.0, 0.0),
            p90_lab=(0.0, 0.0, 0.0),
            raw_delta_e=0.0,
            chroma_delta_e=0.0,
            status="pass",
            failure_codes=[],
            metadata={"empty_pixels": True},
        )

    mean_lab = tuple(float(x) for x in np.mean(object_pixels_lab, axis=0))
    p10_lab = tuple(float(x) for x in np.percentile(object_pixels_lab, 10, axis=0))
    p50_lab = tuple(float(x) for x in np.percentile(object_pixels_lab, 50, axis=0))
    p90_lab = tuple(float(x) for x in np.percentile(object_pixels_lab, 90, axis=0))

    ref_L, ref_a, ref_b = extract_reference_lab(ref_lab_stats)
    ref_coord = np.array([ref_L, ref_a, ref_b], dtype=np.float64)
    obj_coord = np.array(mean_lab, dtype=np.float64)

    raw_delta_e = float(ciede2000(ref_coord, obj_coord, normalize_lightness=False))
    chroma_delta_e = float(ciede2000(ref_coord, obj_coord, normalize_lightness=True))

    failure_codes: list[str] = []
    if chroma_delta_e > warn_thresh:
        status = "fail"
        failure_codes.append("CHROMA_DELTA_E_FAIL")
    elif chroma_delta_e > pass_thresh:
        status = "warn"
        failure_codes.append("CHROMA_DELTA_E_WARN")
    else:
        status = "pass"

    return ObjectQCScore(
        mean_lab=mean_lab,
        p10_lab=p10_lab,
        p50_lab=p50_lab,
        p90_lab=p90_lab,
        raw_delta_e=raw_delta_e,
        chroma_delta_e=chroma_delta_e,
        status=status,
        failure_codes=failure_codes,
        metadata={
            "pixel_count": len(object_pixels_lab),
            "ref_lab": [round(ref_L, 2), round(ref_a, 2), round(ref_b, 2)],
        },
    )


@dataclass
class QCResult:
    """Comprehensive shot-level Color QC results and metrics."""

    shot_id: str
    qc_status: str  # 'pass' | 'warn' | 'fail'
    failure_codes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    object_scores: dict[str, ObjectQCScore] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shot_id": self.shot_id,
            "qc_status": self.qc_status,
            "failure_codes": list(self.failure_codes),
            "metrics": self.metrics,
            "object_scores": {k: v.to_dict() for k, v in self.object_scores.items()},
        }


def load_shot_frames(
    frames_source: str | Path | Sequence[np.ndarray],
    stride: int = 1,
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Load frames from a directory, video file, or sequence of arrays.

    Parameters
    ----------
    frames_source : str, Path, or Sequence[np.ndarray]
        Directory of image files, video file path, or in-memory arrays.
    stride : int
        Temporal sampling stride (default 1).
    max_frames : int, optional
        Maximum number of frames to load.

    Returns
    -------
    list[np.ndarray]
        List of (H, W, 3) RGB uint8 image arrays.
    """
    stride = max(1, int(stride))

    if isinstance(frames_source, (list, tuple)):
        frames = list(frames_source[::stride])
        if max_frames is not None:
            frames = frames[:max_frames]
        return frames

    path = Path(frames_source)
    if not path.exists():
        raise FileNotFoundError(f"Frames source not found: {path}")

    # Case A: Video file
    if path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {path}")

        frames_list: list[np.ndarray] = []
        frame_idx = 0
        try:
            while True:
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                if frame_idx % stride == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    frames_list.append(rgb)
                    if max_frames is not None and len(frames_list) >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()
        return frames_list

    # Case B: Directory of image files
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts],
        key=lambda p: (len(p.stem), p.stem),
    )

    if not image_paths:
        return []

    sampled_paths = image_paths[::stride]
    if max_frames is not None:
        sampled_paths = sampled_paths[:max_frames]

    frames_list = []
    for img_p in sampled_paths:
        bgr = cv2.imread(str(img_p))
        if bgr is not None:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames_list.append(rgb)

    return frames_list


def evaluate_shot_qc(
    shot_id: str,
    frames: str | Path | Sequence[np.ndarray],
    mask_archive: ShotMaskArchive | Path | str | None = None,
    ref_lab_stats_map: dict[str | int, dict[str, Any]] | None = None,
    stride: int = 1,
    thresholds: dict[str, float] | None = None,
) -> QCResult:
    """Perform comprehensive shot-level Color QC evaluation.

    Combines cross-shot object CIEDE2000 (reading RLE masks) and intra-shot
    temporal metrics (Color Distribution Consistency CDC, DIS optical flow
    warping chroma error, hue jumps, and mask bleed).

    Parameters
    ----------
    shot_id : str
        Shot identifier.
    frames : str, Path, or Sequence[np.ndarray]
        Shot frames (video path, frames directory, or list of RGB arrays).
    mask_archive : ShotMaskArchive, Path, or str, optional
        Pre-loaded ShotMaskArchive or path to .masks.npz archive.
    ref_lab_stats_map : dict, optional
        Mapping from object identifier/name/entity_id to reference Lab statistics.
    stride : int
        Temporal sampling stride (default 1).
    thresholds : dict[str, float], optional
        Threshold overrides for metric boundaries.

    Returns
    -------
    QCResult
        Structured QC result including qc_status ('pass' | 'warn' | 'fail'),
        failure codes, and detailed metrics.
    """
    th = thresholds or {}
    cdc_warn_thresh = float(th.get("cdc_warn", 0.70))
    cdc_fail_thresh = float(th.get("cdc_fail", 0.50))
    flow_warn_thresh = float(th.get("flow_warp_warn", 6.0))
    flow_fail_thresh = float(th.get("flow_warp_fail", 12.0))
    hue_jump_thresh = float(th.get("hue_jump_deg", 25.0))
    mask_bleed_warn_thresh = float(th.get("mask_bleed_warn", 5.0))
    mask_bleed_fail_thresh = float(th.get("mask_bleed_fail", 8.0))

    stride = max(1, int(stride))

    # Load frames
    if isinstance(frames, (str, Path)):
        # Load all frames so stride can be applied synchronously with mask indexing
        all_frames = load_shot_frames(frames, stride=1)
    else:
        all_frames = list(frames)

    total_frame_count = len(all_frames)
    if total_frame_count == 0:
        return QCResult(
            shot_id=shot_id,
            qc_status="warn",
            failure_codes=["NO_FRAMES"],
            metrics={"frame_count": 0, "error": "No frames provided"},
            object_scores={},
        )

    # Subsample frames according to stride
    sampled_indices = list(range(0, total_frame_count, stride))
    sampled_rgb = [all_frames[idx] for idx in sampled_indices]
    sampled_gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in sampled_rgb]
    sampled_lab = [srgb_to_lab(f) for f in sampled_rgb]

    # 1. Intra-shot temporal metrics
    cdc = compute_cdc(sampled_rgb, stride=1)
    flow_warp_error = compute_optical_flow_chroma_error(sampled_gray, sampled_lab, stride=1)
    hue_jumps = detect_hue_jumps(sampled_lab, threshold_degrees=hue_jump_thresh, stride=1)

    # 2. Resolve mask archive if path provided
    archive: ShotMaskArchive | None = None
    if mask_archive is not None:
        if isinstance(mask_archive, ShotMaskArchive):
            archive = mask_archive
        else:
            p = Path(mask_archive)
            if p.is_file():
                archive = ShotMaskArchive.load_npz(p)

    # 3. Cross-Shot Object Evaluation and Mask Bleed Detection
    object_scores: dict[str, ObjectQCScore] = {}
    max_mask_bleed = 0.0
    ref_map = ref_lab_stats_map or {}

    if archive is not None and len(archive.objects) > 0:
        for obj in archive.objects:
            accumulated_pixels: list[np.ndarray] = []
            for sample_i, orig_t in enumerate(sampled_indices):
                if orig_t >= archive.frame_count:
                    continue
                mask_2d = archive.get_mask(orig_t, obj.obj_id)
                if mask_2d is None or np.count_nonzero(mask_2d) == 0:
                    continue

                lab_f = sampled_lab[sample_i]
                fg = mask_2d > 0
                obj_pixels = lab_f[fg]
                if obj_pixels.size > 0:
                    accumulated_pixels.append(obj_pixels)

                bleed = detect_mask_bleed(
                    lab_f,
                    mask_2d,
                    radius=3,
                    threshold=mask_bleed_fail_thresh,
                )
                if bleed > max_mask_bleed:
                    max_mask_bleed = bleed

            # Resolve reference stats for object
            ref_stats = (
                ref_map.get(obj.name)
                or ref_map.get(obj.entity_id)
                or ref_map.get(obj.obj_id)
                or ref_map.get(str(obj.obj_id))
            )

            if accumulated_pixels and ref_stats:
                all_obj_pixels = np.concatenate(accumulated_pixels, axis=0)
                score = evaluate_object_color(all_obj_pixels, ref_stats, thresholds=th)
                object_scores[obj.name] = score

    # 4. Status mapping & failure codes
    failure_codes: list[str] = []
    statuses: list[str] = []

    # Object Chroma Delta E
    for obj_name, score in object_scores.items():
        if score.status == "fail":
            if "CHROMA_DELTA_E_FAIL" not in failure_codes:
                failure_codes.append("CHROMA_DELTA_E_FAIL")
            statuses.append("fail")
        elif score.status == "warn":
            if "CHROMA_DELTA_E_WARN" not in failure_codes:
                failure_codes.append("CHROMA_DELTA_E_WARN")
            statuses.append("warn")

    # CDC
    if cdc < cdc_fail_thresh:
        failure_codes.append("CDC_FAIL")
        statuses.append("fail")
    elif cdc < cdc_warn_thresh:
        failure_codes.append("CDC_WARN")
        statuses.append("warn")

    # Optical Flow Warp Error
    if flow_warp_error > flow_fail_thresh:
        failure_codes.append("FLOW_WARP_FAIL")
        statuses.append("fail")
    elif flow_warp_error > flow_warn_thresh:
        failure_codes.append("FLOW_WARP_WARN")
        statuses.append("warn")

    # Hue jumps
    if len(hue_jumps) > 0:
        failure_codes.append("HUE_JUMP_FAIL")
        statuses.append("fail")

    # Mask bleed
    if max_mask_bleed > mask_bleed_fail_thresh:
        failure_codes.append("MASK_BLEED_FAIL")
        statuses.append("fail")
    elif max_mask_bleed > mask_bleed_warn_thresh:
        failure_codes.append("MASK_BLEED_WARN")
        statuses.append("warn")

    if "fail" in statuses:
        qc_status = "fail"
    elif "warn" in statuses:
        qc_status = "warn"
    else:
        qc_status = "pass"

    max_chroma_delta_e = (
        max([s.chroma_delta_e for s in object_scores.values()]) if object_scores else None
    )
    mean_raw_delta_e = (
        float(np.mean([s.raw_delta_e for s in object_scores.values()])) if object_scores else None
    )

    metrics: dict[str, Any] = {
        "chroma_delta_e": round(max_chroma_delta_e, 3) if max_chroma_delta_e is not None else None,
        "raw_delta_e": round(mean_raw_delta_e, 3) if mean_raw_delta_e is not None else None,
        "cdc": round(cdc, 4),
        "flow_warp_error": round(flow_warp_error, 4),
        "hue_jumps": hue_jumps,
        "mask_bleed": round(max_mask_bleed, 4),
        "total_frames": total_frame_count,
        "sampled_frames": len(sampled_rgb),
        "stride": stride,
    }

    return QCResult(
        shot_id=shot_id,
        qc_status=qc_status,
        failure_codes=failure_codes,
        metrics=metrics,
        object_scores=object_scores,
    )
