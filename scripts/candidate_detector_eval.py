"""Empirical multi-candidate film defect detector evaluation and visual diagnostics.

Wraps and evaluates candidate detection algorithms under identical preprocessing:
1. Baseline (`farneback_3f`): 3-frame Farneback Optical Flow Temporal Consensus.
2. Candidate A (`dis_flow`): OpenCV DIS Optical Flow temporal consensus.
3. Candidate B (`ridge`): Multi-scale steerable Ridge filter with directional tracking.
4. Candidate C (`consensus_5f`): 5-frame temporal median/rank consensus (t-2 .. t+2).
5. Candidate D (`dart_harness`): Isolated external DART MaskNet evaluation harness.

All production/classical candidates use 100% commercial-safe (BSD/MIT/Apache-2.0)
OpenCV and NumPy primitives. Candidate D never imports DART or RTN code; it only
reads pre-exported external mask files in an isolated research harness.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

import benchmark_defect_masks as bench
import film_defect_masks as fdm
import film_scratch_ridge as fsr


STANDARD_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4K": (3840, 2160),
}

CANDIDATE_NAMES: tuple[str, ...] = (
    "farneback_3f",
    "dis_flow",
    "ridge",
    "consensus_5f",
    "dart_harness",
)


@dataclass(frozen=True)
class PreprocessedSequence:
    """Standardized frame sequence representation shared across all detectors."""

    bgr_frames: list[np.ndarray]
    gray_frames: list[np.ndarray]
    height: int
    width: int


def preprocess_frames(frames: Sequence[np.ndarray] | np.ndarray) -> PreprocessedSequence:
    """Normalize input frames into validated uint8 BGR and grayscale sequences.

    Ensures all candidates receive identical frame representations without
    silent resizing or channel ambiguity.
    """
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

        hw = ( int(gray.shape[0]), int(gray.shape[1]) )
        if first_hw is None:
            first_hw = hw
        elif hw != first_hw:
            raise ValueError(f"Frame {idx} dimensions {hw} do not match first frame {first_hw}")

        bgr_list.append(bgr)
        gray_list.append(gray)

    assert first_hw is not None
    return PreprocessedSequence(
        bgr_frames=bgr_list,
        gray_frames=gray_list,
        height=first_hw[0],
        width=first_hw[1],
    )


def split_defect_components(
    binary_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify connected components of a binary proposal mask into (scratch, dust).

    Uses the geometric criteria from `film_defect_masks.py` to separate elongated
    vertical/near-vertical scratches from compact transient dust/blotch spots.
    Connects vertically aligned scratch fragments before component classification.
    """
    if binary_mask.ndim != 2 or binary_mask.dtype != np.uint8:
        raise ValueError("Expected 2D uint8 binary mask")
    h, w = binary_mask.shape
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    dust_mask = np.zeros((h, w), dtype=np.uint8)

    if not np.any(binary_mask):
        return scratch_mask, dust_mask

    rough = (binary_mask > 0).astype(np.uint8) * 255
    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, min(15, h // 25))))
    closed_vert = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, k_vert)
    total, labels, stats, _ = cv2.connectedComponentsWithStats(closed_vert, connectivity=8)
    max_blob_area = max(1000, int(round(h * w * 0.08)))
    max_blob_edge = max(60, int(round(min(h, w) * 0.50)))

    for component in range(1, total):
        _x, _y, cw, ch, area = map(int, stats[component])
        comp_mask = labels == component
        pts = cv2.findNonZero((comp_mask & (rough > 0)).astype(np.uint8))
        if pts is not None and len(pts) >= 5:
            rect = cv2.minAreaRect(pts)
            (rcx, rcy), (rw, rh), angle = rect
            thick = min(rw, rh)
            length = max(rw, rh)
        else:
            thick = float(min(cw, ch))
            length = float(max(cw, ch))

        fine_scratch = (
            (length >= max(6, int(h * 0.08)))
            and (thick <= max(14, int(w * 0.12)))
            and (
                (length >= 2.5 * max(thick, 1.0) and ch >= 1.4 * cw)
                or (ch >= 2.5 * cw)
            )
            and area <= max(64, cw * h)
        )
        compact = (length < 2.5 * max(thick, 1.0)) or (area <= 60 and max(cw, ch) <= 15)
        small_spot = (
            not fine_scratch
            and area <= max_blob_area
            and cw <= max_blob_edge
            and ch <= max_blob_edge
            and compact
        )
        if fine_scratch:
            scratch_mask[comp_mask & (rough > 0)] = 255
        elif small_spot:
            dust_mask[comp_mask & (rough > 0)] = 255

    max_scratch_frac = max(0.10, min(0.25, 24.0 / max(w, 1)))
    max_dust_frac = max(0.15, min(0.30, 24.0 / max(w, 1)))
    if np.count_nonzero(scratch_mask) / scratch_mask.size > max_scratch_frac:
        scratch_mask[:] = 0
    if np.count_nonzero(dust_mask) / dust_mask.size > max_dust_frac:
        dust_mask[:] = 0
    return scratch_mask, dust_mask


def _horizontal_flank_diff(centre: np.ndarray, flank: int = 4) -> np.ndarray:
    """Compute horizontal flank contrast so vertical scratches >=3px wide are not suppressed by 5x5 2D median."""
    h, w = centre.shape
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


def detect_farneback_3f(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    sensitivity: float = 0.65,
    **_: Any,
) -> dict[str, list[np.ndarray]]:
    """Baseline: 3-frame Farneback Optical Flow Temporal Consensus."""
    if not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("Sensitivity must be between 0 and 1")
    seq = preprocess_frames(frames)
    n = len(seq.gray_frames)
    h, w = seq.height, seq.width
    scratch_out: list[np.ndarray] = []
    dust_out: list[np.ndarray] = []

    for t in range(n):
        if n < 3 or h < 16 or w < 16:
            scratch_out.append(np.zeros((h, w), dtype=np.uint8))
            dust_out.append(np.zeros((h, w), dtype=np.uint8))
            continue

        prev_idx = max(0, t - 1) if t > 0 else min(n - 1, 1)
        next_idx = min(n - 1, t + 1) if t < n - 1 else max(0, n - 2)
        if t == 0 and n >= 3:
            prev_idx, next_idx = 1, 2
        elif t == n - 1 and n >= 3:
            prev_idx, next_idx = n - 2, n - 3

        a = seq.gray_frames[prev_idx]
        b = seq.gray_frames[t]
        c = seq.gray_frames[next_idx]

        if (
            float(np.median(cv2.absdiff(a, b))) >= 22.0
            or float(np.median(cv2.absdiff(b, c))) >= 22.0
        ):
            scratch_out.append(np.zeros((h, w), dtype=np.uint8))
            dust_out.append(np.zeros((h, w), dtype=np.uint8))
            continue

        before = fdm.aligned_neighbor(b, a).astype(np.float32)
        after = fdm.aligned_neighbor(b, c).astype(np.float32)
        proposal = _temporal_pair_proposal(before, b, after, sensitivity)
        s_mask, d_mask = split_defect_components(proposal)
        scratch_out.append(s_mask)
        dust_out.append(d_mask)

    return {"scratch": scratch_out, "dust": dust_out}


def _aligned_neighbor_dis(reference: np.ndarray, neighbor: np.ndarray, dis: Any) -> np.ndarray:
    """Warp a neighbor frame into reference coordinates using OpenCV DIS optical flow."""
    h, w = reference.shape
    flow = dis.calc(reference, neighbor, None)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(
        neighbor,
        x + flow[..., 0],
        y + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def detect_dis_flow(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    sensitivity: float = 0.65,
    preset: int | None = None,
    **_: Any,
) -> dict[str, list[np.ndarray]]:
    """Candidate A: OpenCV DIS Optical Flow 3-frame temporal consensus."""
    if not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("Sensitivity must be between 0 and 1")
    seq = preprocess_frames(frames)
    n = len(seq.gray_frames)
    h, w = seq.height, seq.width
    scratch_out: list[np.ndarray] = []
    dust_out: list[np.ndarray] = []

    dis_preset = cv2.DISOPTICAL_FLOW_PRESET_FAST if preset is None else preset
    dis = cv2.DISOpticalFlow_create(dis_preset)

    for t in range(n):
        if n < 3 or h < 16 or w < 16:
            scratch_out.append(np.zeros((h, w), dtype=np.uint8))
            dust_out.append(np.zeros((h, w), dtype=np.uint8))
            continue

        prev_idx = max(0, t - 1) if t > 0 else 1
        next_idx = min(n - 1, t + 1) if t < n - 1 else n - 2
        if t == 0:
            prev_idx, next_idx = 1, 2
        elif t == n - 1:
            prev_idx, next_idx = n - 2, n - 3

        a = seq.gray_frames[prev_idx]
        b = seq.gray_frames[t]
        c = seq.gray_frames[next_idx]

        if (
            float(np.median(cv2.absdiff(a, b))) >= 22.0
            or float(np.median(cv2.absdiff(b, c))) >= 22.0
        ):
            scratch_out.append(np.zeros((h, w), dtype=np.uint8))
            dust_out.append(np.zeros((h, w), dtype=np.uint8))
            continue

        before = _aligned_neighbor_dis(b, a, dis).astype(np.float32)
        after = _aligned_neighbor_dis(b, c, dis).astype(np.float32)
        proposal = _temporal_pair_proposal(before, b, after, sensitivity)
        s_mask, d_mask = split_defect_components(proposal)
        scratch_out.append(s_mask)
        dust_out.append(d_mask)

    return {"scratch": scratch_out, "dust": dust_out}


def detect_multiscale_ridge(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    sensitivity: float = 0.65,
    scales: Sequence[int] = (2, 4, 6),
    **_: Any,
) -> dict[str, list[np.ndarray]]:
    """Candidate B: Multi-scale steerable Ridge filter with directional temporal tracking.

    Combines `film_scratch_ridge.detect_sequence` (when geometry permits) with
    multi-scale horizontal flank ridge response and directional vertical tracking
    across frames so persistent and drifting vertical/near-vertical scratches are
    captured across scales. Dust output is explicitly empty (0) since Ridge is a
    specialized scratch channel detector.
    """
    if not math.isfinite(sensitivity) or not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("Sensitivity must be finite in [0, 1]")
    seq = preprocess_frames(frames)
    n = len(seq.bgr_frames)
    h, w = seq.height, seq.width
    dust_out = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]

    if h < 16 or w < 16:
        return {
            "scratch": [np.zeros((h, w), dtype=np.uint8) for _ in range(n)],
            "dust": dust_out,
        }

    # Run canonical PR #24 ridge detector if within its geometry/length envelope
    base_ridge_masks: list[np.ndarray] | None = None
    if 3 <= n <= 96 and h >= 32 and w >= 32 and h * w * n <= 96_000_000:
        base_ridge_masks = fsr.detect_sequence(seq.bgr_frames, sensitivity=sensitivity)

    gray_stack = np.stack([g.astype(np.float32) for g in seq.gray_frames], axis=0)
    lower = np.percentile(gray_stack, 20, axis=0).astype(np.float32)
    upper = np.percentile(gray_stack, 80, axis=0).astype(np.float32)
    contrast_thresh = 19.0 - 7.0 * float(sensitivity)
    novelty_thresh = 24.0 - 8.0 * float(sensitivity)
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

        # Directional morphological filtering along vertical axis
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

    return {"scratch": scratch_out, "dust": dust_out}


def detect_consensus_5f(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    sensitivity: float = 0.65,
    **_: Any,
) -> dict[str, list[np.ndarray]]:
    """Candidate C: 5-frame temporal median/rank consensus (t-2 .. t+2).

    Evaluates a 5-frame temporal window around each frame t using temporal rank
    statistics (median and interquartile range across neighbors) to isolate both
    single-frame dust impulses and 2-frame persistent blotches/scratches.
    """
    if not 0.0 <= float(sensitivity) <= 1.0:
        raise ValueError("Sensitivity must be between 0 and 1")
    seq = preprocess_frames(frames)
    n = len(seq.gray_frames)
    h, w = seq.height, seq.width
    scratch_out: list[np.ndarray] = []
    dust_out: list[np.ndarray] = []

    if h < 16 or w < 16 or n < 2:
        for _ in range(n):
            scratch_out.append(np.zeros((h, w), dtype=np.uint8))
            dust_out.append(np.zeros((h, w), dtype=np.uint8))
        return {"scratch": scratch_out, "dust": dust_out}

    gray_f32 = [g.astype(np.float32) for g in seq.gray_frames]

    for t in range(n):
        # Gather up to 4 neighbor indices in [t-2, t-1, t+1, t+2]
        neighbor_indices = [
            idx for idx in (t - 2, t - 1, t + 1, t + 2) if 0 <= idx < n
        ]
        if not neighbor_indices:
            scratch_out.append(np.zeros((h, w), dtype=np.uint8))
            dust_out.append(np.zeros((h, w), dtype=np.uint8))
            continue

        centre_u8 = seq.gray_frames[t]
        centre = gray_f32[t]
        valid_neighbors = [
            gray_f32[idx]
            for idx in neighbor_indices
            if float(np.median(cv2.absdiff(centre_u8, seq.gray_frames[idx]))) < 22.0
        ]
        if len(valid_neighbors) < min(2, len(neighbor_indices)):
            scratch_out.append(np.zeros((h, w), dtype=np.uint8))
            dust_out.append(np.zeros((h, w), dtype=np.uint8))
            continue

        neighbor_stack = np.stack(valid_neighbors, axis=0)
        temporal_median = np.median(neighbor_stack, axis=0)
        neighbor_spread = np.percentile(neighbor_stack, 75, axis=0) - np.percentile(
            neighbor_stack, 25, axis=0
        )

        mismatch = np.abs(centre - temporal_median)
        med_diff = float(np.median(mismatch))
        mad = float(np.median(np.abs(mismatch - med_diff)))
        threshold = max(16.0, med_diff + (6.5 - 3.0 * float(sensitivity)) * max(mad, 1.2))

        bg_stable = neighbor_spread <= max(14.0, threshold * 0.75)
        spatial_2d = np.abs(centre - cv2.medianBlur(centre_u8, 5).astype(np.float32))
        spatial_diff = np.maximum(spatial_2d, _horizontal_flank_diff(centre, flank=4))
        local_thresh = max(10.0, threshold * (0.45 - 0.12 * float(sensitivity)))

        likely = (mismatch >= threshold) & bg_stable & (spatial_diff >= local_thresh)
        picture = centre_u8 >= max(8, int(np.percentile(centre_u8, 98) * 0.025))
        likely &= picture
        likely[:2, :] = False
        likely[-2:, :] = False
        likely[:, :2] = False
        likely[:, -2:] = False

        proposal = likely.astype(np.uint8) * 255
        s_mask, d_mask = split_defect_components(proposal)
        scratch_out.append(s_mask)
        dust_out.append(d_mask)

    return {"scratch": scratch_out, "dust": dust_out}


def find_isolated_dart_python() -> Path | None:
    """Locate isolated DART virtual environment Python executable."""
    candidate = Path("I:/DART/.venv/Scripts/python.exe")
    if candidate.is_file():
        return candidate
    candidate_unix = Path("I:/DART/.venv/bin/python")
    if candidate_unix.is_file():
        return candidate_unix
    managed = Path(__file__).resolve().parent.parent / "tools/dart/.venv/Scripts/python.exe"
    if managed.is_file():
        return managed
    return None


def run_genuine_dart_inference(
    frames_bgr: list[np.ndarray],
    *,
    threshold: int = 160,
    weights_path: Path | None = None,
    dart_python: Path | None = None,
    device: str = "auto",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Execute genuine DART MaskNet inference inside isolated DART .venv."""
    py = dart_python or find_isolated_dart_python()
    if not py or not py.is_file():
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
        scratch_out = []
        dust_out = []
        for idx in range(len(frames_bgr)):
            mf = masks_dir / f"frame_{idx:06d}.png"
            if not mf.is_file():
                zero = np.zeros((h, w), dtype=np.uint8)
                scratch_out.append(zero)
                dust_out.append(zero.copy())
                continue
            raw = cv2.imread(str(mf), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"Corrupt DART output mask: {mf}")
            if raw.shape != (h, w):
                raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
            binary = np.where(raw >= threshold, np.uint8(255), np.uint8(0))
            s_mask, d_mask = split_defect_components(binary)
            categorized = (s_mask > 0) | (d_mask > 0)
            uncategorized = (binary > 0) & ~categorized
            if np.any(uncategorized):
                d_mask = np.where(uncategorized, np.uint8(255), d_mask)
            scratch_out.append(s_mask)
            dust_out.append(d_mask)

        return scratch_out, dust_out


def detect_dart_harness(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    external_mask_dir: Path | None = None,
    frame_names: Sequence[str] | None = None,
    threshold: int = 160,
    sensitivity: float = 0.65,
    run_inference: bool = False,
    allow_fallback: bool = True,
    **_: Any,
) -> dict[str, Any]:
    """Candidate D: Isolated external DART MaskNet evaluation harness.

    Strictly preserves provenance tier:
    - `precomputed_dart_prediction`: Reads pre-exported 8-bit soft masks from external_mask_dir.
    - `genuine_dart_inference`: Runs genuine MaskNet inference in isolated I:/DART/.venv.
    - `classical_fallback`: Runs classical proxy combining 5-frame consensus and multi-scale ridge.
    - `not_evaluated`: When neither precomputed masks nor genuine inference nor fallback can run.
    """
    if not 1 <= int(threshold) <= 255:
        raise ValueError("DART threshold must be between 1 and 255")
    seq = preprocess_frames(frames)
    n = len(seq.gray_frames)
    h, w = seq.height, seq.width

    # 1. External precomputed masks
    if external_mask_dir is not None:
        mask_root = Path(external_mask_dir)
        if (mask_root / "masks").is_dir():
            mask_root = mask_root / "masks"
        names = (
            list(frame_names)
            if frame_names is not None
            else [f"{i + 1:06d}.png" for i in range(n)]
        )
        if len(names) != n:
            raise ValueError(f"Expected {n} frame_names, got {len(names)}")

        scratch_out: list[np.ndarray] = []
        dust_out: list[np.ndarray] = []
        for name in names:
            fname = name if name.lower().endswith(".png") else f"{name}.png"
            mask_path = mask_root / fname
            if not mask_path.is_file():
                stem = Path(fname).stem
                alt_name = f"frame_{int(stem):06d}.png" if stem.isdigit() else f"frame_{stem}.png"
                if (mask_root / alt_name).is_file():
                    mask_path = mask_root / alt_name
                else:
                    raise FileNotFoundError(f"Missing external DART mask: {mask_path}")
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if raw_mask is None or raw_mask.ndim != 2 or raw_mask.shape != (h, w):
                raise ValueError(f"Invalid external DART mask geometry at {mask_path}")
            binary = np.where(raw_mask >= threshold, np.uint8(255), np.uint8(0))
            s_mask, d_mask = split_defect_components(binary)
            scratch_out.append(s_mask)
            dust_out.append(d_mask)
        return {
            "scratch": scratch_out,
            "dust": dust_out,
            "provenance": "precomputed_dart_prediction",
        }

    # 2. Genuine isolated DART MaskNet inference if explicitly requested
    if run_inference:
        dart_py = find_isolated_dart_python()
        weights = Path("I:/DART/pretrained_models/dart.safetensors")
        if dart_py is not None and weights.is_file() and n >= 2:
            try:
                s_out, d_out = run_genuine_dart_inference(
                    seq.bgr_frames,
                    threshold=threshold,
                    weights_path=weights,
                    dart_python=dart_py,
                )
                return {
                    "scratch": s_out,
                    "dust": d_out,
                    "provenance": "genuine_dart_inference",
                }
            except Exception as exc:
                if not allow_fallback:
                    raise RuntimeError(f"Genuine DART inference failed: {exc}") from exc

    # 3. Controlled classical fallback proxy
    if allow_fallback:
        c5 = detect_consensus_5f(seq.bgr_frames, sensitivity=sensitivity)
        ridge = detect_multiscale_ridge(seq.bgr_frames, sensitivity=sensitivity)
        scratch_combined = [
            np.where((s1 > 0) | (s2 > 0), np.uint8(255), np.uint8(0))
            for s1, s2 in zip(c5["scratch"], ridge["scratch"])
        ]
        return {
            "scratch": scratch_combined,
            "dust": c5["dust"],
            "provenance": "classical_fallback",
        }

    # 4. Not evaluated
    return {
        "scratch": [np.zeros((h, w), dtype=np.uint8) for _ in range(n)],
        "dust": [np.zeros((h, w), dtype=np.uint8) for _ in range(n)],
        "provenance": "not_evaluated",
    }


def detect_legacy_spatial_scratch(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    sensitivity: float = 0.65,
    **_: Any,
) -> dict[str, Any]:
    """Single-frame spatial vertical scratch detector (legacy/ai_scratch_mask)."""
    seq = preprocess_frames(frames)
    import film_defect_detectors as fdd
    scratch_out = [
        fdd.legacy_spatial_scratch_mask(f, sensitivity=sensitivity)
        for f in seq.bgr_frames
    ]
    dust_out = [np.zeros((seq.height, seq.width), dtype=np.uint8) for _ in seq.bgr_frames]
    return {
        "scratch": scratch_out,
        "dust": dust_out,
        "provenance": "in_tree_spatial_scratch",
    }


CANDIDATE_REGISTRY: dict[str, Callable[..., dict[str, Any]]] = {
    "farneback_3f": detect_farneback_3f,
    "dis_flow": detect_dis_flow,
    "ridge": detect_multiscale_ridge,
    "consensus_5f": detect_consensus_5f,
    "legacy": detect_legacy_spatial_scratch,
    "dart_harness": detect_dart_harness,
}


def compute_temporal_metrics(
    gt_masks: Sequence[np.ndarray],
    pred_masks: Sequence[np.ndarray],
    *,
    threshold: int = 128,
) -> dict[str, float]:
    """Compute temporal consistency metrics across a sequence of masks.

    Metrics:
    - `mask_flicker_rate`: Fraction of unaffected static pixels that alternate
      mask state (0 -> 1 -> 0) across consecutive triples (t-1, t, t+1).
    - `detection_continuity`: Mean fraction of active ground-truth frames in a
      persistent defect trajectory where the detector predicts at least one TP pixel.
    - `missed_persistent_rate`: Complement of detection_continuity (1.0 - continuity)
      on multi-frame defect sequences (0.0 if no persistent defects exist).
    """
    if len(gt_masks) != len(pred_masks) or not gt_masks:
        raise ValueError("GT and predicted mask sequences must have matching non-zero length")
    n = len(gt_masks)
    gt_bin = [(m == 255) for m in gt_masks]
    pr_bin = [(m >= threshold) for m in pred_masks]

    # Mask flicker rate on unaffected pixels across 3-frame windows
    flicker_events = 0
    unaffected_triplets = 0
    for t in range(1, n - 1):
        unaffected = ~(gt_bin[t - 1] | gt_bin[t] | gt_bin[t + 1])
        n_unaff = int(np.count_nonzero(unaffected))
        if n_unaff == 0:
            continue
        oscillated = (
            unaffected
            & (pr_bin[t] != pr_bin[t - 1])
            & (pr_bin[t] != pr_bin[t + 1])
        )
        flicker_events += int(np.count_nonzero(oscillated))
        unaffected_triplets += n_unaff

    flicker_rate = flicker_events / unaffected_triplets if unaffected_triplets > 0 else 0.0

    # Detection continuity across active ground-truth frames
    active_gt_frames = [t for t in range(n) if np.any(gt_bin[t])]
    if active_gt_frames:
        detected_frames = sum(
            1 for t in active_gt_frames if np.any(gt_bin[t] & pr_bin[t])
        )
        continuity = detected_frames / len(active_gt_frames)
        missed_rate = 1.0 - continuity
    else:
        continuity = 1.0
        missed_rate = 0.0

    return {
        "mask_flicker_rate": round(float(flicker_rate), 8),
        "detection_continuity": round(float(continuity), 6),
        "missed_persistent_rate": round(float(missed_rate), 6),
    }


def build_error_map(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    threshold: int = 128,
) -> np.ndarray:
    """Create a BGR color-coded error map (Green=TP, Red=FP, Blue=FN).

    OpenCV BGR channel convention:
    - True Positive (TP): Green = (0, 255, 0)
    - False Positive (FP): Red = (0, 0, 255)
    - False Negative (FN): Blue = (255, 0, 0)
    - Ignored (127 in GT): Dark gray = (64, 64, 64)
    - True Negative (TN): Black = (0, 0, 0)
    """
    if gt_mask.shape != pred_mask.shape or gt_mask.ndim != 2:
        raise ValueError(f"Expected matching 2D masks, got {gt_mask.shape} and {pred_mask.shape}")
    h, w = gt_mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    valid = gt_mask != 127
    gt_pos = (gt_mask == 255) & valid
    pr_pos = (pred_mask >= threshold) & valid

    tp = gt_pos & pr_pos
    fp = (~gt_pos) & pr_pos & valid
    fn = gt_pos & (~pr_pos) & valid
    ignored = ~valid

    canvas[tp] = (0, 255, 0)       # Green = TP
    canvas[fp] = (0, 0, 255)       # Red = FP
    canvas[fn] = (255, 0, 0)       # Blue = FN
    canvas[ignored] = (64, 64, 64)
    return canvas


def build_temporal_strip(
    masks: Sequence[np.ndarray],
    center_index: int,
    *,
    panel_width: int,
    panel_height: int,
) -> np.ndarray:
    """Build Panel F: 5-frame temporal evolution strip (t-2 .. t+2) resized to panel geometry."""
    if not masks:
        raise ValueError("Mask sequence must not be empty")
    n = len(masks)
    if not 0 <= center_index < n:
        raise ValueError(f"center_index {center_index} out of bounds for length {n}")

    offsets = (-2, -1, 0, 1, 2)
    sub_w = max(8, panel_width // len(offsets))
    strips: list[np.ndarray] = []

    for dt in offsets:
        idx = min(max(0, center_index + dt), n - 1)
        m = masks[idx]
        bgr = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) if m.ndim == 2 else m.copy()
        if dt == 0:
            # Highlight center frame t border in yellow/cyan
            bgr[:, 0:1] = (0, 255, 255)
            bgr[:, -1:] = (0, 255, 255)
        resized = cv2.resize(bgr, (sub_w, panel_height), interpolation=cv2.INTER_NEAREST)
        strips.append(resized)

    combined = np.concatenate(strips, axis=1)
    if combined.shape[1] != panel_width:
        combined = cv2.resize(combined, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)
    return combined


def generate_contact_sheet(
    source_frames: Sequence[np.ndarray],
    scratch_gt: Sequence[np.ndarray],
    dust_gt: Sequence[np.ndarray],
    scratch_pred: Sequence[np.ndarray],
    dust_pred: Sequence[np.ndarray],
    *,
    frame_index: int = 0,
    method_name: str = "candidate",
    output_path: Path | None = None,
    panel_width: int = 240,
) -> np.ndarray:
    """Generate a 6-panel side-by-side diagnostic contact sheet (Panels A-F).

    Panels:
    - Panel A: Source Frame
    - Panel B: Ground Truth Mask (Scratch + Dust combined/colorized)
    - Panel C: Predicted Scratch Mask
    - Panel D: Predicted Dust Mask
    - Panel E: Color-Coded Error Map (Green=TP, Red=FP, Blue=FN)
    - Panel F: Temporal Strip (t-2 .. t+2)
    """
    seq = preprocess_frames(source_frames)
    n = len(seq.bgr_frames)
    if not 0 <= frame_index < n:
        raise ValueError(f"frame_index {frame_index} out of range [0, {n - 1}]")
    for seq_arr in (scratch_gt, dust_gt, scratch_pred, dust_pred):
        if len(seq_arr) != n:
            raise ValueError("All GT and prediction mask lists must match source_frames length")

    h, w = seq.height, seq.width
    panel_height = max(96, int(round(h * panel_width / max(w, 1))))

    src_bgr = seq.bgr_frames[frame_index]
    s_gt = scratch_gt[frame_index]
    d_gt = dust_gt[frame_index]
    s_pr = scratch_pred[frame_index]
    d_pr = dust_pred[frame_index]

    combined_gt = np.where((s_gt == 255) | (d_gt == 255), np.uint8(255), np.uint8(0))
    combined_gt = np.where((s_gt == 127) | (d_gt == 127), np.uint8(127), combined_gt)
    combined_pr = np.where((s_pr >= 128) | (d_pr >= 128), np.uint8(255), np.uint8(0))

    all_combined_pr = [
        np.where((sp >= 128) | (dp >= 128), np.uint8(255), np.uint8(0))
        for sp, dp in zip(scratch_pred, dust_pred)
    ]

    gt_bgr = cv2.cvtColor(combined_gt, cv2.COLOR_GRAY2BGR)
    s_pr_bgr = cv2.cvtColor(s_pr, cv2.COLOR_GRAY2BGR)
    d_pr_bgr = cv2.cvtColor(d_pr, cv2.COLOR_GRAY2BGR)
    error_bgr = build_error_map(combined_gt, combined_pr)
    temporal_bgr = build_temporal_strip(
        all_combined_pr,
        frame_index,
        panel_width=panel_width,
        panel_height=panel_height,
    )

    raw_panels = [
        (src_bgr, f"A: Source ({method_name})"),
        (gt_bgr, "B: Ground Truth"),
        (s_pr_bgr, "C: Pred Scratch"),
        (d_pr_bgr, "D: Pred Dust"),
        (error_bgr, "E: Error (G=TP R=FP B=FN)"),
        (temporal_bgr, "F: Temporal (t-2..t+2)"),
    ]

    tiled_panels: list[np.ndarray] = []
    for img, label in raw_panels:
        if img.shape[0] != panel_height or img.shape[1] != panel_width:
            resized = cv2.resize(img, (panel_width, panel_height), interpolation=cv2.INTER_NEAREST)
        else:
            resized = img.copy()
        footer = np.full((28, panel_width, 3), 24, dtype=np.uint8)
        cv2.putText(
            footer,
            label,
            (6, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        tiled_panels.append(np.concatenate([resized, footer], axis=0))

    sheet = np.concatenate(tiled_panels, axis=1)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), sheet):
            raise IOError(f"Failed to write contact sheet to {output_path}")
    return sheet


def profile_detector_performance(
    detector_fn: Callable[..., dict[str, list[np.ndarray]]],
    *,
    resolutions: dict[str, tuple[int, int]] | None = None,
    num_frames: int = 5,
    seed: int = 42,
    **detector_kwargs: Any,
) -> dict[str, dict[str, float | int]]:
    """Measure execution latency (ms/frame) and peak memory footprint (MB) per resolution."""
    if num_frames < 3:
        raise ValueError("num_frames must be >= 3 for temporal profiling")
    target_res = resolutions or STANDARD_RESOLUTIONS
    rng = np.random.default_rng(seed)
    results: dict[str, dict[str, float | int]] = {}

    for label, (width, height) in target_res.items():
        frames = [
            rng.integers(64, 192, size=(height, width, 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]
        tracemalloc.start()
        t0 = time.perf_counter()
        _ = detector_fn(frames, **detector_kwargs)
        elapsed_s = time.perf_counter() - t0
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        ms_per_frame = (elapsed_s * 1000.0) / num_frames
        peak_ram_mb = peak_bytes / (1024.0 * 1024.0)
        results[label] = {
            "width": width,
            "height": height,
            "frames": num_frames,
            "total_ms": round(elapsed_s * 1000.0, 3),
            "ms_per_frame": round(ms_per_frame, 3),
            "peak_ram_mb": round(max(peak_ram_mb, 0.01), 3),
        }

    return results


def evaluate_all_candidates(
    *,
    candidates: Sequence[str] = CANDIDATE_NAMES,
    num_frames: int = 8,
    height: int = 96,
    width: int = 128,
    seed: int = 42,
    sensitivity: float = 0.65,
    contact_sheet_dir: Path | None = None,
    profile_resolutions: dict[str, tuple[int, int]] | None = None,
    external_dart_mask_dir: Path | None = None,
) -> dict[str, Any]:
    """Run head-to-head empirical evaluation across all candidate detectors."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.linspace(90.0, 140.0, height, dtype=np.float32),
        np.linspace(95.0, 145.0, width, dtype=np.float32),
        indexing="ij",
    )
    base_canvas = 0.5 * (yy + xx)
    clean_frames = []
    for _ in range(num_frames):
        grain = rng.normal(0.0, 2.5, size=(height, width)).astype(np.float32)
        gray = np.clip(np.round(base_canvas + grain), 0, 255).astype(np.uint8)
        clean_frames.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    scratch_cfg = bench.ScratchConfig(
        width=2,
        opacity=0.85,
        angle_deg=0.0,
        duration=min(5, num_frames),
        drift_px=3.0,
        bright=True,
        seed=seed,
    )
    dust_cfg = bench.DustConfig(
        clusters_per_frame=3,
        particles_per_cluster=(2, 4),
        radius_range=(2, 4),
        contrast=0.85,
        seed=seed + 1,
    )

    scratch_frames, scratch_gt = bench.inject_scratches(clean_frames, scratch_cfg)
    dust_frames, dust_gt = bench.inject_dust(clean_frames, dust_cfg)

    # Also build a combined clip containing both scratches and dust for unified contact sheets
    combined_frames, combined_dust_gt = bench.inject_dust(scratch_frames, dust_cfg)
    empty_gt = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]

    # Pick middle frame for diagnostic contact sheet
    sheet_frame_idx = num_frames // 2

    report: dict[str, Any] = {
        "schema_version": 1,
        "parameters": {
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "seed": seed,
            "sensitivity": sensitivity,
        },
        "candidates": {},
    }

    for name in candidates:
        if name not in CANDIDATE_REGISTRY:
            raise ValueError(f"Unknown candidate detector: {name!r}")
        detector = CANDIDATE_REGISTRY[name]
        kwargs: dict[str, Any] = {"sensitivity": sensitivity}
        if name == "dart_harness" and external_dart_mask_dir is not None:
            kwargs["external_mask_dir"] = external_dart_mask_dir

        # Evaluate on scratch clip and dust clip under identical preprocessing
        pred_on_scratch = detector(scratch_frames, **kwargs)
        pred_on_dust = detector(dust_frames, **kwargs)
        pred_on_combined = detector(combined_frames, **kwargs)

        # Score spatial metrics separated by defect category
        spatial_scores = bench.evaluate_category_masks(
            scratch_gt=scratch_gt,
            scratch_pred=pred_on_scratch["scratch"],
            dust_gt=dust_gt,
            dust_pred=pred_on_dust["dust"],
        )

        # Score temporal consistency metrics
        scratch_temporal = compute_temporal_metrics(scratch_gt, pred_on_scratch["scratch"])
        dust_temporal = compute_temporal_metrics(dust_gt, pred_on_dust["dust"])

        sheet_path: Path | None = None
        if contact_sheet_dir is not None:
            sheet_path = contact_sheet_dir / f"contact_sheet_{name}.png"
            generate_contact_sheet(
                combined_frames,
                scratch_gt,
                combined_dust_gt,
                pred_on_combined["scratch"],
                pred_on_combined["dust"],
                frame_index=sheet_frame_idx,
                method_name=name,
                output_path=sheet_path,
            )

        perf_profile: dict[str, dict[str, float | int]] = {}
        if profile_resolutions is not None:
            perf_profile = profile_detector_performance(
                detector,
                resolutions=profile_resolutions,
                num_frames=max(3, min(5, num_frames)),
                seed=seed,
                **kwargs,
            )

        report["candidates"][name] = {
            "scratch_metrics": spatial_scores["scratch_metrics"],
            "dust_metrics": spatial_scores["dust_metrics"],
            "summary": spatial_scores["summary"],
            "temporal_metrics": {
                "scratch": scratch_temporal,
                "dust": dust_temporal,
            },
            "performance_profile": perf_profile,
            "contact_sheet": str(sheet_path) if sheet_path is not None else None,
        }

    return report


def format_markdown_summary(report: dict[str, Any]) -> str:
    """Render comparative evaluation results as a Markdown table."""
    lines = [
        "# Candidate Defect Detector Empirical Comparison",
        "",
        "## Spatial Metrics by Defect Channel",
        "",
        "| Candidate | Scratch P | Scratch R | Scratch F1 | Scratch IoU | Scratch FPR | Dust P | Dust R | Dust F1 | Dust IoU | Dust FPR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, data in report.get("candidates", {}).items():
        sm = data["scratch_metrics"]
        dm = data["dust_metrics"]
        lines.append(
            f"| `{name}` | {sm['precision']:.4f} | {sm['recall']:.4f} | {sm['f1']:.4f} | "
            f"{sm['iou']:.4f} | {sm['false_positive_rate']:.6f} | "
            f"{dm['precision']:.4f} | {dm['recall']:.4f} | {dm['f1']:.4f} | "
            f"{dm['iou']:.4f} | {dm['false_positive_rate']:.6f} |"
        )

    lines.extend([
        "",
        "## Temporal Consistency Metrics",
        "",
        "| Candidate | Scratch Flicker Rate | Scratch Continuity | Missed Persistent Rate | Dust Flicker Rate |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, data in report.get("candidates", {}).items():
        st = data["temporal_metrics"]["scratch"]
        dt = data["temporal_metrics"]["dust"]
        lines.append(
            f"| `{name}` | {st['mask_flicker_rate']:.6f} | {st['detection_continuity']:.4f} | "
            f"{st['missed_persistent_rate']:.4f} | {dt['mask_flicker_rate']:.6f} |"
        )

    # Add performance table if present
    any_perf = any(data.get("performance_profile") for data in report.get("candidates", {}).values())
    if any_perf:
        lines.extend([
            "",
            "## Execution Latency & Memory Footprint",
            "",
            "| Candidate | Resolution | Latency (ms/frame) | Peak RAM (MB) |",
            "|---|---|---:|---:|",
        ])
        for name, data in report.get("candidates", {}).items():
            for res_label, perf in data.get("performance_profile", {}).items():
                lines.append(
                    f"| `{name}` | {res_label} ({perf['width']}x{perf['height']}) | "
                    f"{perf['ms_per_frame']:.2f} | {perf['peak_ram_mb']:.2f} |"
                )

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=list(CANDIDATE_NAMES),
        choices=CANDIDATE_NAMES,
        help="Candidate detectors to evaluate",
    )
    parser.add_argument("--num-frames", type=int, default=8, help="Number of frames per test sequence")
    parser.add_argument("--height", type=int, default=96, help="Test sequence frame height")
    parser.add_argument("--width", type=int, default=128, help="Test sequence frame width")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sensitivity", type=float, default=0.65, help="Detector sensitivity in [0, 1]")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON report and contact sheets")
    parser.add_argument("--markdown-out", type=Path, help="Optional path to write Markdown comparison summary")
    parser.add_argument(
        "--profile-standard-resolutions",
        action="store_true",
        help="Profile latency and memory across 720p, 1080p, and 4K",
    )
    parser.add_argument(
        "--external-dart-masks",
        type=Path,
        help="Optional external directory of pre-computed DART masks",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    res_map = STANDARD_RESOLUTIONS if args.profile_standard_resolutions else {
        "eval_res": (args.width, args.height),
    }

    report = evaluate_all_candidates(
        candidates=args.candidates,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=args.seed,
        sensitivity=args.sensitivity,
        contact_sheet_dir=args.output_dir / "contact_sheets",
        profile_resolutions=res_map,
        external_dart_mask_dir=args.external_dart_masks,
    )

    report_path = args.output_dir / "candidate_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    md_text = format_markdown_summary(report)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(md_text, encoding="utf-8")

    print(md_text)
    print(f"Report written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
