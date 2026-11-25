#!/usr/bin/env python3
"""Deterministic 100%-OSI Video Colorization Backend.

Features:
1. Native OpenCV Guided Filter (He et al., ECCV 2010 / TPAMI 2013) implemented using
   cv2.boxFilter and NumPy (zero cv2.ximgproc contrib dependency).
2. Step-by-step consecutive frame OpenCV DIS Optical Flow backward warping.
3. Bidirectional temporal blending across multiple reference keyframes in a shot.
4. Texture-preserving object mask chroma transfer with Color Bible anchoring.
5. Verbatim L* preservation ("Original pixels win" principle) with gamut fitting.
6. Clean Python API and standalone CLI entry point.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.color_science import D65_WHITE, SRGB_TO_XYZ_MATRIX
from ai_remaster_gui.rle_masks import MaskObjectMetadata, ShotMaskArchive, load_shot_masks_npz

# Precompute inverse matrix from XYZ to Linear sRGB
_XYZ_TO_SRGB_MATRIX: np.ndarray = np.linalg.inv(SRGB_TO_XYZ_MATRIX)

# Optical flow preset lookup
DIS_PRESETS: dict[str, int] = {
    "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
    "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
    "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
}


def _guided_filter_2d(
    I: np.ndarray,
    p: np.ndarray,
    radius: int = 5,
    eps: float = 1e-3,
) -> np.ndarray:
    """Internal 2D implementation of He et al. Guided Filter."""
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(I, -1, ksize, borderType=cv2.BORDER_REFLECT)
    corr_I = cv2.boxFilter(I * I, -1, ksize, borderType=cv2.BORDER_REFLECT)
    var_I = np.maximum(0.0, corr_I - mean_I * mean_I)

    mean_p = cv2.boxFilter(p, -1, ksize, borderType=cv2.BORDER_REFLECT)
    corr_Ip = cv2.boxFilter(I * p, -1, ksize, borderType=cv2.BORDER_REFLECT)
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, ksize, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, ksize, borderType=cv2.BORDER_REFLECT)

    q = mean_a * I + mean_b
    return np.nan_to_num(q, nan=0.0).astype(np.float32)


def guided_filter(
    guide_gray: np.ndarray,
    src_channel: np.ndarray,
    radius: int = 5,
    eps: float = 1e-3,
) -> np.ndarray:
    """Edge-preserving Guided Image Filter (He, Sun, Tang, ECCV 2010 / TPAMI 2013).

    Smooths noise and compression artifacts in src_channel while transferring and
    preserving the sharp high-contrast structural edges of guide_gray. Implemented
    strictly via native cv2.boxFilter and NumPy, requiring zero cv2.ximgproc builds.

    Parameters
    ----------
    guide_gray : np.ndarray
        2D guidance grayscale array of shape (H, W). Normalized to [0.0, 1.0].
        If passed as uint8 in [0, 255] or float with max > 1.0, automatically scaled.
    src_channel : np.ndarray
        Filtering input array of shape (H, W) or (H, W, C), typically chroma a* or b*.
    radius : int, optional
        Filter window radius in pixels, by default 5 (kernel size 2*r + 1).
    eps : float, optional
        Regularization parameter penalizing large variance coefficients, by default 1e-3.

    Returns
    -------
    np.ndarray
        Filtered output array with identical shape and float32 dtype.
    """
    if radius < 1:
        radius = 1
    if eps <= 0.0:
        eps = 1e-6

    # Normalize guidance image to float32 in [0.0, 1.0]
    I = np.asarray(guide_gray, dtype=np.float32)
    if I.ndim == 3 and I.shape[-1] == 1:
        I = I[..., 0]
    elif I.ndim != 2:
        raise ValueError(f"guide_gray must be a 2D array (H, W), got shape {guide_gray.shape}")

    if np.nanmax(I) > 1.0:
        I = I / 255.0
    I = np.clip(np.nan_to_num(I, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    p = np.asarray(src_channel, dtype=np.float32)
    if p.ndim == 2:
        if I.shape != p.shape:
            raise ValueError(f"guide_gray shape {I.shape} does not match src_channel shape {p.shape}")
        return _guided_filter_2d(I, p, radius=radius, eps=eps)
    elif p.ndim == 3:
        H, W, C = p.shape
        if I.shape != (H, W):
            raise ValueError(f"guide_gray spatial shape {I.shape} does not match src_channel shape ({H}, {W})")
        channels = [
            _guided_filter_2d(I, p[..., c], radius=radius, eps=eps)
            for c in range(C)
        ]
        return np.stack(channels, axis=-1)
    else:
        raise ValueError(f"src_channel must be 2D (H, W) or 3D (H, W, C), got shape {src_channel.shape}")


def calculate_step_flow(
    curr_gray: np.ndarray,
    prev_gray: np.ndarray,
    preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST,
    flow_scale: float = 1.0,
    dis_instance: Any | None = None,
) -> np.ndarray:
    """Compute dense optical flow from curr_gray to prev_gray using OpenCV DIS.

    Supports optional downscaled flow computation via ``flow_scale`` and reusing
    a pre-initialized ``dis_instance`` across frames.
    """
    g_curr = _to_uint8_gray(curr_gray)
    g_prev = _to_uint8_gray(prev_gray)

    if g_curr.shape != g_prev.shape:
        raise ValueError(f"Frame shapes do not match: curr {g_curr.shape} vs prev {g_prev.shape}")

    h, w = g_curr.shape
    scale = float(flow_scale) if flow_scale else 1.0
    dis = dis_instance if dis_instance is not None else cv2.DISOpticalFlow_create(preset)

    if 0.05 < scale < 0.99 and min(h, w) >= 32:
        sw = max(16, int(round(w * scale)))
        sh = max(16, int(round(h * scale)))
        sc = cv2.resize(g_curr, (sw, sh), interpolation=cv2.INTER_AREA)
        sp = cv2.resize(g_prev, (sw, sh), interpolation=cv2.INTER_AREA)
        low_flow = dis.calc(sc, sp, None).astype(np.float32)
        flow = cv2.resize(low_flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= float(w) / float(sw)
        flow[..., 1] *= float(h) / float(sh)
        return flow.astype(np.float32)

    flow: np.ndarray = dis.calc(g_curr, g_prev, None)
    return flow.astype(np.float32)


def warp_chroma_backward(
    prev_chroma: np.ndarray,
    flow_curr_to_prev: np.ndarray,
) -> np.ndarray:
    """Backward warp prev_chroma into current frame coordinates using dense flow.

    Constructs sampling grid map_x = grid_x + flow[..., 0], map_y = grid_y + flow[..., 1],
    and remaps using bilinear interpolation and reflection boundary condition.

    Parameters
    ----------
    prev_chroma : np.ndarray
        Chroma array in previous frame coords, shape (H, W) or (H, W, C), float32.
    flow_curr_to_prev : np.ndarray
        Dense optical flow from curr to prev, shape (H, W, 2), float32.

    Returns
    -------
    np.ndarray
        Warped chroma array in current frame coords with matching shape and dtype.
    """
    H, W = flow_curr_to_prev.shape[:2]
    if prev_chroma.shape[:2] != (H, W):
        raise ValueError(
            f"Chroma spatial shape {prev_chroma.shape[:2]} does not match flow spatial shape ({H}, {W})"
        )

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    map_x = grid_x + flow_curr_to_prev[..., 0]
    map_y = grid_y + flow_curr_to_prev[..., 1]

    warped: np.ndarray = cv2.remap(
        prev_chroma.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return warped


def lab_to_linear_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* coordinates directly to unclipped Linear sRGB [0.0, 1.0].

    Used for gamut boundary detection and chroma scale factor calculation.
    """
    fy = (np.clip(L, 0.0, 100.0) + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    delta = 6.0 / 29.0
    t_div = 3.0 * (delta**2)
    t_add = 4.0 / 29.0

    xr = np.where(fx > delta, np.maximum(fx, 0.0) ** 3, (fx - t_add) * t_div)
    yr = np.where(fy > delta, np.maximum(fy, 0.0) ** 3, (fy - t_add) * t_div)
    zr = np.where(fz > delta, np.maximum(fz, 0.0) ** 3, (fz - t_add) * t_div)

    X = xr * D65_WHITE[0]
    Y = yr * D65_WHITE[1]
    Z = zr * D65_WHITE[2]

    XYZ = np.stack([X, Y, Z], axis=-1)
    return XYZ @ _XYZ_TO_SRGB_MATRIX.T


def fit_chroma_to_gamut(
    L: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    num_steps: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically scale chroma (a*, b*) towards neutral gray if outside sRGB gamut.

    Because neutral gray (L, 0, 0) is guaranteed to lie inside [0.0, 1.0] for any valid
    grayscale L*, scaling (a*, b*) by factor s in [0.0, 1.0] strictly prevents channel
    clipping in RGB space and guarantees exact L* preservation within 1.0 integer tolerance.
    """
    lin = lab_to_linear_rgb(L, a, b)
    out_of_gamut = np.any((lin < 0.0) | (lin > 1.0), axis=-1)
    if not np.any(out_of_gamut):
        return a.copy(), b.copy()

    low = np.zeros_like(L)
    high = np.ones_like(L)

    for _ in range(num_steps):
        mid = (low + high) * 0.5
        test_a = np.where(out_of_gamut, mid * a, a)
        test_b = np.where(out_of_gamut, mid * b, b)
        lin_test = lab_to_linear_rgb(L, test_a, test_b)
        in_g = np.all((lin_test >= 0.0) & (lin_test <= 1.0), axis=-1)
        low = np.where(in_g, mid, low)
        high = np.where(in_g, high, mid)

    scale = np.where(out_of_gamut, low, 1.0)
    return a * scale, b * scale


def apply_strict_l_lock(
    gray_orig: np.ndarray,
    chroma: np.ndarray,
) -> np.ndarray:
    """Enforce strict L* luminance lock: 'Original pixels win'.

    Extracts verbatim L* from the original input grayscale frame, pairs it with
    gamut-fitted chroma (a*, b*), and reconstructs standard uint8 RGB.

    Guarantee: |L*(RGB_out) - L*(Gray_orig)| < 1.0 on a [0, 100] scale, strictly within
    8-bit integer quantization tolerance.

    Parameters
    ----------
    gray_orig : np.ndarray
        2D uint8 original grayscale frame of shape (H, W).
    chroma : np.ndarray
        3D float32 chroma array of shape (H, W, 2) representing (a*, b*).

    Returns
    -------
    np.ndarray
        Reconstructed uint8 RGB frame of shape (H, W, 3).
    """
    g_u8 = _to_uint8_gray(gray_orig)
    H, W = g_u8.shape

    # Extract exact L* from gray_orig
    gray_rgb_f = cv2.cvtColor(g_u8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    orig_L = cv2.cvtColor(gray_rgb_f, cv2.COLOR_RGB2Lab)[..., 0]

    a_raw = chroma[..., 0].astype(np.float32)
    b_raw = chroma[..., 1].astype(np.float32)

    # Fit chroma to sRGB gamut boundary to prevent RGB clipping from altering luminance
    a_fit, b_fit = fit_chroma_to_gamut(orig_L, a_raw, b_raw)

    lab_composed = np.stack([orig_L, a_fit, b_fit], axis=-1)
    rgb_f = cv2.cvtColor(lab_composed, cv2.COLOR_Lab2RGB)
    rgb_u8 = np.clip(np.round(rgb_f * 255.0), 0, 255).astype(np.uint8)
    return rgb_u8


def propagate_shot_chroma(
    frames_gray: list[np.ndarray],
    keyframes_color: dict[int, np.ndarray],
    mask_archive: ShotMaskArchive | None = None,
    color_bible_anchors: dict[Any, Any] | None = None,
    radius: int = 5,
    eps: float = 1e-3,
    flow_preset: int = cv2.DISOPTICAL_FLOW_PRESET_FAST,
    flow_scale: float = 1.0,
    anchor_strength: float = 1.0,
) -> list[np.ndarray]:
    """Colorize a full video shot via bidirectional optical flow propagation and guided filtering.

    Parameters
    ----------
    frames_gray : list[np.ndarray]
        List of 2D uint8 grayscale frames (H, W) for the entire shot.
    keyframes_color : dict[int, np.ndarray]
        Mapping of frame_idx -> (H, W, 3) uint8 RGB for approved reference keyframes.
    mask_archive : ShotMaskArchive | None, optional
        Per-shot multi-object tracked RLE mask archive, by default None.
    color_bible_anchors : dict | None, optional
        Target median chroma (a*, b*) per object ID/name/entity to anchor mean color.
    radius : int, optional
        Guided filter window radius in pixels, by default 5.
    eps : float, optional
        Guided filter regularization epsilon, by default 1e-3.
    flow_preset : int, optional
        OpenCV DIS optical flow preset, by default cv2.DISOPTICAL_FLOW_PRESET_FAST.

    Returns
    -------
    list[np.ndarray]
        List of colorized uint8 RGB frames of shape (H, W, 3) for the entire shot.
    """
    if not frames_gray:
        return []

    T = len(frames_gray)
    if not keyframes_color:
        raise ValueError("At least one reference keyframe is required in keyframes_color")

    # Validate frames_gray
    cleaned_frames_gray: list[np.ndarray] = [_to_uint8_gray(f) for f in frames_gray]
    H, W = cleaned_frames_gray[0].shape
    for idx, f in enumerate(cleaned_frames_gray):
        if f.shape != (H, W):
            raise ValueError(f"Frame {idx} shape {f.shape} does not match initial frame ({H}, {W})")

    # Validate keyframes_color
    cleaned_keyframes: dict[int, np.ndarray] = {}
    key_chroma: dict[int, np.ndarray] = {}
    for k_idx, k_img in keyframes_color.items():
        k_idx_int = int(k_idx)
        if not (0 <= k_idx_int < T):
            raise ValueError(f"Keyframe index {k_idx_int} is out of bounds for shot length {T}")
        k_arr = np.asarray(k_img, dtype=np.uint8)
        if k_arr.shape != (H, W, 3):
            raise ValueError(f"Keyframe {k_idx_int} shape {k_arr.shape} does not match shot ({H}, {W}, 3)")
        cleaned_keyframes[k_idx_int] = k_arr

        # Extract float32 LAB chroma (a*, b*)
        k_rgb_f = k_arr.astype(np.float32) / 255.0
        k_lab = cv2.cvtColor(k_rgb_f, cv2.COLOR_RGB2Lab)
        key_chroma[k_idx_int] = k_lab[..., 1:3].astype(np.float32)

    # 1. Compute consecutive step optical flows between adjacent frames (reusing DIS instance)
    dis_instance = cv2.DISOpticalFlow_create(flow_preset)
    flow_forward: dict[int, np.ndarray] = {}
    for t in range(1, T):
        flow_forward[t] = calculate_step_flow(
            curr_gray=cleaned_frames_gray[t],
            prev_gray=cleaned_frames_gray[t - 1],
            preset=flow_preset,
            flow_scale=flow_scale,
            dis_instance=dis_instance,
        )

    flow_backward: dict[int, np.ndarray] = {}
    for t in range(0, T - 1):
        flow_backward[t] = calculate_step_flow(
            curr_gray=cleaned_frames_gray[t],
            prev_gray=cleaned_frames_gray[t + 1],
            preset=flow_preset,
            flow_scale=flow_scale,
            dis_instance=dis_instance,
        )

    # 2. Refine motion fields inside tracked object masks if mask archive is present
    if mask_archive is not None:
        yy, xx = np.indices((H, W), dtype=np.float32)
        for obj in mask_archive.objects:
            obj_id = obj.obj_id

            # Forward mask tracking alignment
            for t in range(1, T):
                if mask_archive.has_mask(t, obj_id) and mask_archive.has_mask(t - 1, obj_id):
                    m_curr = mask_archive.get_mask(t, obj_id)
                    m_prev = mask_archive.get_mask(t - 1, obj_id)
                    if m_curr is not None and m_prev is not None:
                        fg_curr = (m_curr > 0)
                        fg_prev = (m_prev > 0)
                        if np.count_nonzero(fg_curr) > 0 and np.count_nonzero(fg_prev) > 0:
                            c_curr_x = float(xx[fg_curr].mean())
                            c_curr_y = float(yy[fg_curr].mean())
                            c_prev_x = float(xx[fg_prev].mean())
                            c_prev_y = float(yy[fg_prev].mean())
                            exp_dx = c_prev_x - c_curr_x
                            exp_dy = c_prev_y - c_curr_y
                            cur_mean_dx = float(flow_forward[t][fg_curr, 0].mean())
                            cur_mean_dy = float(flow_forward[t][fg_curr, 1].mean())
                            flow_forward[t][fg_curr, 0] += (exp_dx - cur_mean_dx)
                            flow_forward[t][fg_curr, 1] += (exp_dy - cur_mean_dy)

            # Backward mask tracking alignment
            for t in range(0, T - 1):
                if mask_archive.has_mask(t, obj_id) and mask_archive.has_mask(t + 1, obj_id):
                    m_curr = mask_archive.get_mask(t, obj_id)
                    m_next = mask_archive.get_mask(t + 1, obj_id)
                    if m_curr is not None and m_next is not None:
                        fg_curr = (m_curr > 0)
                        fg_next = (m_next > 0)
                        if np.count_nonzero(fg_curr) > 0 and np.count_nonzero(fg_next) > 0:
                            c_curr_x = float(xx[fg_curr].mean())
                            c_curr_y = float(yy[fg_curr].mean())
                            c_next_x = float(xx[fg_next].mean())
                            c_next_y = float(yy[fg_next].mean())
                            exp_dx = c_next_x - c_curr_x
                            exp_dy = c_next_y - c_curr_y
                            cur_mean_dx = float(flow_backward[t][fg_curr, 0].mean())
                            cur_mean_dy = float(flow_backward[t][fg_curr, 1].mean())
                            flow_backward[t][fg_curr, 0] += (exp_dx - cur_mean_dx)
                            flow_backward[t][fg_curr, 1] += (exp_dy - cur_mean_dy)

    # 3. Bidirectional Temporal Propagation across multiple reference keyframes
    sorted_keys = sorted(cleaned_keyframes.keys())
    chroma_fields: list[np.ndarray | None] = [None] * T

    # Range before first keyframe: propagate backward from K_0 down to 0
    k_first = sorted_keys[0]
    chroma_fields[k_first] = key_chroma[k_first].copy()
    c = key_chroma[k_first].copy()
    for t in range(k_first - 1, -1, -1):
        c = warp_chroma_backward(c, flow_backward[t])
        chroma_fields[t] = c.copy()

    # Range after last keyframe: propagate forward from K_last up to T - 1
    k_last = sorted_keys[-1]
    chroma_fields[k_last] = key_chroma[k_last].copy()
    c = key_chroma[k_last].copy()
    for t in range(k_last + 1, T):
        c = warp_chroma_backward(c, flow_forward[t])
        chroma_fields[t] = c.copy()

    # Intervals between consecutive keyframes: linear bidirectional distance blend
    for i in range(len(sorted_keys) - 1):
        k_a = sorted_keys[i]
        k_b = sorted_keys[i + 1]
        span = k_b - k_a

        if span == 1:
            chroma_fields[k_a] = key_chroma[k_a].copy()
            chroma_fields[k_b] = key_chroma[k_b].copy()
            continue

        # Forward run from k_a to k_b
        fwd_seq: dict[int, np.ndarray] = {k_a: key_chroma[k_a].copy()}
        curr_c = key_chroma[k_a].copy()
        for t in range(k_a + 1, k_b + 1):
            curr_c = warp_chroma_backward(curr_c, flow_forward[t])
            fwd_seq[t] = curr_c.copy()

        # Backward run from k_b down to k_a
        bwd_seq: dict[int, np.ndarray] = {k_b: key_chroma[k_b].copy()}
        curr_c = key_chroma[k_b].copy()
        for t in range(k_b - 1, k_a - 1, -1):
            curr_c = warp_chroma_backward(curr_c, flow_backward[t])
            bwd_seq[t] = curr_c.copy()

        # Occlusion-aware bidirectional temporal blending
        occ_fwd: dict[int, np.ndarray] = {}
        occ_bwd: dict[int, np.ndarray] = {}
        for t in range(k_a + 1, k_b):
            fb_f = flow_forward[t] + warp_chroma_backward(flow_backward[t - 1], flow_forward[t])
            fb_b = flow_backward[t] + warp_chroma_backward(flow_forward[t + 1], flow_backward[t])
            occ_fwd[t] = np.exp(-np.sum(fb_f ** 2, axis=-1) / 16.0).astype(np.float32)
            occ_bwd[t] = np.exp(-np.sum(fb_b ** 2, axis=-1) / 16.0).astype(np.float32)

        for t in range(k_a, k_b + 1):
            if t == k_a:
                chroma_fields[t] = key_chroma[k_a].copy()
                continue
            if t == k_b:
                chroma_fields[t] = key_chroma[k_b].copy()
                continue
            w_fwd = (float(k_b - t) / float(span)) * np.maximum(occ_fwd[t], 0.05)
            w_bwd = (float(t - k_a) / float(span)) * np.maximum(occ_bwd[t], 0.05)
            denom = np.maximum(w_fwd + w_bwd, 1e-6)[..., None]
            blended = (w_fwd[..., None] * fwd_seq[t] + w_bwd[..., None] * bwd_seq[t]) / denom
            chroma_fields[t] = blended.astype(np.float32)

    # 4. Texture-preserving object mask chroma transfer with soft feathering and Color Bible anchoring
    strength = max(0.0, min(1.0, float(anchor_strength)))
    if mask_archive is not None and strength > 0.0:
        for t in range(T):
            cur_ch = chroma_fields[t]
            if cur_ch is None:
                continue

            for obj in mask_archive.objects:
                if not mask_archive.has_mask(t, obj.obj_id):
                    continue
                mask = mask_archive.get_mask(t, obj.obj_id)
                if mask is None:
                    continue
                fg = (mask > 0)
                if np.count_nonzero(fg) == 0:
                    continue

                # Check if Color Bible anchor is defined for this object
                anchor = _resolve_color_anchor(color_bible_anchors, obj)
                if anchor is not None:
                    target_a, target_b = anchor
                    cur_a = cur_ch[fg, 0]
                    cur_b = cur_ch[fg, 1]
                    med_a = float(np.median(cur_a))
                    med_b = float(np.median(cur_b))
                    shift_a = target_a - med_a
                    shift_b = target_b - med_b
                    # Feather mask boundary smoothly inside foreground to avoid sharp chroma halos
                    soft_fg = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), sigmaX=1.5)
                    soft_fg = np.where(fg, np.clip(soft_fg * 1.15, 0.25, 1.0), soft_fg * 0.25) * strength
                    cur_ch[..., 0] += soft_fg * shift_a
                    cur_ch[..., 1] += soft_fg * shift_b

            chroma_fields[t] = cur_ch

    # 5. Edge-aware Guided Filter smoothing & Strict L* Lock reconstruction
    output_frames: list[np.ndarray] = []
    for t in range(T):
        raw_chroma = chroma_fields[t]
        if raw_chroma is None:
            raw_chroma = np.zeros((H, W, 2), dtype=np.float32)

        guide = cleaned_frames_gray[t].astype(np.float32) / 255.0
        filtered_a = guided_filter(guide, raw_chroma[..., 0], radius=radius, eps=eps)
        filtered_b = guided_filter(guide, raw_chroma[..., 1], radius=radius, eps=eps)
        filtered_chroma = np.stack([filtered_a, filtered_b], axis=-1)

        # Enforce verbatim L* preservation
        rgb_frame = apply_strict_l_lock(cleaned_frames_gray[t], filtered_chroma)
        output_frames.append(rgb_frame)

    return output_frames


def colorize_shot_osi_masked_lab(
    source: str | Path | Sequence[np.ndarray],
    keyframes: dict[int, np.ndarray | str | Path],
    mask_archive: ShotMaskArchive | str | Path | None = None,
    output_path: str | Path | None = None,
    color_bible_anchors: dict[Any, Any] | str | Path | None = None,
    color_bible_db: str | Path | None = None,
    film_id: str | None = None,
    shot_index: int = 0,
    radius: int = 5,
    eps: float = 1e-3,
    fps: float = 24.0,
    flow_preset: str | int = "fast",
) -> list[np.ndarray]:
    """Top-level pipeline interface for 100%-OSI video colorization.

    Accepts source video file, directory of frames, or list of arrays. Writes
    reconstructed color video or saved frames if output_path is provided.
    """
    # 1. Load source grayscale frames
    frames_gray = _load_source_frames(source)
    if not frames_gray:
        raise ValueError("Source contains zero frames")

    # 2. Resolve keyframe color images
    resolved_keyframes: dict[int, np.ndarray] = {}
    for k_idx, k_val in keyframes.items():
        if isinstance(k_val, (str, Path)):
            p = Path(k_val)
            if not p.exists():
                raise FileNotFoundError(f"Keyframe image path not found: {p}")
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Failed to read keyframe image: {p}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resolved_keyframes[int(k_idx)] = rgb
        elif isinstance(k_val, np.ndarray):
            resolved_keyframes[int(k_idx)] = _to_uint8_rgb(k_val)
        else:
            raise TypeError(f"Unsupported keyframe image type: {type(k_val)}")

    # 3. Resolve mask archive
    resolved_archive: ShotMaskArchive | None = None
    if isinstance(mask_archive, (str, Path)):
        p = Path(mask_archive)
        if not p.exists():
            raise FileNotFoundError(f"Mask archive file not found: {p}")
        resolved_archive = load_shot_masks_npz(p)
    elif isinstance(mask_archive, ShotMaskArchive):
        resolved_archive = mask_archive

    # 4. Resolve color bible anchors
    resolved_anchors: dict[Any, Any] | None = None
    if isinstance(color_bible_anchors, (str, Path)):
        p = Path(color_bible_anchors)
        if not p.exists():
            raise FileNotFoundError(f"Color bible anchors file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            resolved_anchors = json.load(f)
    elif isinstance(color_bible_anchors, dict):
        resolved_anchors = color_bible_anchors
    elif resolved_anchors is None and film_id and color_bible_db:
        resolved_anchors = resolve_color_bible_anchors(
            color_bible_db=color_bible_db,
            film_id=film_id,
            shot_index=shot_index,
            mask_archive=resolved_archive,
        )

    # 5. Resolve optical flow preset
    preset_int = flow_preset
    if isinstance(flow_preset, str):
        preset_int = DIS_PRESETS.get(flow_preset.lower(), cv2.DISOPTICAL_FLOW_PRESET_FAST)

    # 6. Execute propagation
    output_frames = propagate_shot_chroma(
        frames_gray=frames_gray,
        keyframes_color=resolved_keyframes,
        mask_archive=resolved_archive,
        color_bible_anchors=resolved_anchors,
        radius=radius,
        eps=eps,
        flow_preset=int(preset_int),
    )

    # 7. Write output if requested
    if output_path is not None:
        _save_output_frames(output_frames, output_path, fps=fps)

    return output_frames


# ==============================================================================
# Helper Utilities
# ==============================================================================


def _to_uint8_gray(img: np.ndarray) -> np.ndarray:
    """Normalize input array to 2D uint8 grayscale."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[-1] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Cannot convert array of shape {arr.shape} to grayscale")
    elif arr.ndim != 2:
        raise ValueError(f"Grayscale image must be 2D, got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    """Normalize input array to 3D uint8 RGB."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    elif arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"RGB image must have shape (H, W, 3), got shape {arr.shape}")

    if np.issubdtype(arr.dtype, np.floating):
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def resolve_color_bible_anchors(
    color_bible_db: str | Path,
    film_id: str,
    shot_index: int = 0,
    mask_archive: ShotMaskArchive | None = None,
) -> dict[Any, Any]:
    """Resolve target median (a*, b*) anchors from SQLite Color Bible DB."""
    db_path = Path(color_bible_db)
    if not db_path.is_file():
        return {}

    try:
        from ai_remaster_gui.color_bible_service import ColorBibleService
    except ImportError:
        try:
            from color_bible_service import ColorBibleService  # type: ignore[no-redef]
        except ImportError:
            return {}

    service = ColorBibleService(db_path=db_path)
    try:
        assignments = service.get_shot_palette_assignments(film_id, shot_index)
        anchors_dict: dict[Any, Any] = {}
        for asgn_item in assignments:
            refs = asgn_item.get("color_references") or []
            target_chroma = None
            for ref in refs:
                lab_stats = ref.get("lab_stats")
                if not lab_stats and ref.get("lab_stats_json"):
                    try:
                        lab_stats = json.loads(ref["lab_stats_json"])
                    except Exception:
                        lab_stats = None
                if isinstance(lab_stats, dict):
                    med_a, med_b = None, None
                    if "a" in lab_stats and "b" in lab_stats:
                        a_val, b_val = lab_stats["a"], lab_stats["b"]
                        if isinstance(a_val, (list, tuple)) and isinstance(b_val, (list, tuple)):
                            if len(a_val) > 0 and len(b_val) > 0:
                                med_a = float(a_val[len(a_val) // 2])
                                med_b = float(b_val[len(b_val) // 2])
                        else:
                            try:
                                med_a = float(a_val)
                                med_b = float(b_val)
                            except (ValueError, TypeError):
                                pass
                    elif "median_a" in lab_stats and "median_b" in lab_stats:
                        try:
                            med_a = float(lab_stats["median_a"])
                            med_b = float(lab_stats["median_b"])
                        except (ValueError, TypeError):
                            pass
                    elif "p50_a" in lab_stats and "p50_b" in lab_stats:
                        try:
                            med_a = float(lab_stats["p50_a"])
                            med_b = float(lab_stats["p50_b"])
                        except (ValueError, TypeError):
                            pass

                    if med_a is not None and med_b is not None:
                        target_chroma = (med_a, med_b)
                        break

            if target_chroma is not None:
                entity_id = asgn_item.get("entity_id")
                variant_id = asgn_item.get("variant_id")
                name = (asgn_item.get("entity") or {}).get("name")
                obj_id = asgn_item.get("obj_id") or (asgn_item.get("assignment") or {}).get("obj_id") or (asgn_item.get("entity") or {}).get("obj_id")
                if entity_id:
                    anchors_dict[entity_id] = target_chroma
                if variant_id:
                    anchors_dict[variant_id] = target_chroma
                if name:
                    anchors_dict[name] = target_chroma
                if obj_id is not None:
                    anchors_dict[obj_id] = target_chroma
                    anchors_dict[str(obj_id)] = target_chroma
                if mask_archive is not None:
                    for m_obj in mask_archive.objects:
                        if (entity_id and m_obj.entity_id == entity_id) or \
                           (variant_id and m_obj.variant_id == variant_id) or \
                           (name and m_obj.name == name):
                            anchors_dict[m_obj.obj_id] = target_chroma
                            anchors_dict[str(m_obj.obj_id)] = target_chroma
        return anchors_dict
    finally:
        service.close()


def _resolve_color_anchor(
    anchors: dict[Any, Any] | None,
    obj: MaskObjectMetadata,
) -> tuple[float, float] | None:
    """Resolve target median (a*, b*) from color_bible_anchors dictionary."""
    if not anchors:
        return None

    # Search candidates in priority order: obj_id, str(obj_id), name, entity_id, variant_id
    candidate_keys = [
        obj.obj_id,
        str(obj.obj_id),
        obj.name,
        obj.entity_id,
        obj.variant_id,
    ]

    target_entry = None
    for k in candidate_keys:
        if k in anchors:
            target_entry = anchors[k]
            break

    if target_entry is None:
        return None

    # Parse (a*, b*) from various representations: tuple, list, or dict
    if isinstance(target_entry, (tuple, list)) and len(target_entry) >= 2:
        return float(target_entry[0]), float(target_entry[1])

    if isinstance(target_entry, dict):
        if "median_a" in target_entry and "median_b" in target_entry:
            val_a = target_entry["median_a"]
            val_b = target_entry["median_b"]
            med_a = float(val_a[len(val_a) // 2]) if isinstance(val_a, (list, tuple)) else float(val_a)
            med_b = float(val_b[len(val_b) // 2]) if isinstance(val_b, (list, tuple)) else float(val_b)
            return (med_a, med_b)
        if "p50_a" in target_entry and "p50_b" in target_entry:
            val_a = target_entry["p50_a"]
            val_b = target_entry["p50_b"]
            med_a = float(val_a[len(val_a) // 2]) if isinstance(val_a, (list, tuple)) else float(val_a)
            med_b = float(val_b[len(val_b) // 2]) if isinstance(val_b, (list, tuple)) else float(val_b)
            return (med_a, med_b)
        if "a" in target_entry and "b" in target_entry:
            if isinstance(target_entry["a"], (list, tuple)):
                med_a = float(target_entry["a"][len(target_entry["a"]) // 2])
            else:
                med_a = float(target_entry["a"])
            if isinstance(target_entry["b"], (list, tuple)):
                med_b = float(target_entry["b"][len(target_entry["b"]) // 2])
            else:
                med_b = float(target_entry["b"])
            return (med_a, med_b)
        if "a_star" in target_entry and "b_star" in target_entry:
            return float(target_entry["a_star"]), float(target_entry["b_star"])

    return None


def _load_source_frames(source: str | Path | Sequence[np.ndarray]) -> list[np.ndarray]:
    """Load grayscale frames from video file, directory, or sequence of arrays."""
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        return [_to_uint8_gray(f) for f in source]

    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source path does not exist: {p}")

    if p.is_dir():
        # Load directory of sorted image files
        valid_exts = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
        image_files = sorted([f for f in p.iterdir() if f.suffix.lower() in valid_exts])
        if not image_files:
            raise ValueError(f"No image files found in source directory: {p}")
        frames = []
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image frame: {img_path}")
            frames.append(img)
        return frames

    # Video file via cv2.VideoCapture
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {p}")

    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
    finally:
        cap.release()

    return frames


def _save_output_frames(
    frames: list[np.ndarray],
    output_path: str | Path,
    fps: float = 24.0,
) -> None:
    """Save RGB output frames to video file or image directory."""
    if not frames:
        return

    p = Path(output_path)
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}

    if p.suffix.lower() in video_exts:
        p.parent.mkdir(parents=True, exist_ok=True)
        H, W = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(p), fourcc, fps, (W, H))
        if not writer.isOpened():
            # Fallback to XVID
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(p), fourcc, fps, (W, H))

        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to open cv2.VideoWriter for {output_path} with both mp4v and XVID codecs."
            )

        try:
            for rgb_frame in frames:
                bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()
    else:
        # Directory of frames
        p.mkdir(parents=True, exist_ok=True)
        for idx, rgb_frame in enumerate(frames):
            frame_path = p / f"frame_{idx:05d}.png"
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), bgr)


# ==============================================================================
# CLI Entry Point
# ==============================================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line interface for 100%-OSI video colorization backend."""
    parser = argparse.ArgumentParser(
        description="Deterministic 100%-OSI Video Colorization Backend with Optical Flow & Guided Filter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help="Path to source video file or directory of grayscale frames.",
    )
    parser.add_argument(
        "--references",
        "-r",
        required=True,
        help="JSON file or JSON string mapping frame index to reference image path.",
    )
    parser.add_argument(
        "--masks",
        "-m",
        default=None,
        help="Optional path to .npz ShotMaskArchive containing tracked object masks.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output video file (.mp4) or directory for reconstructed color frames.",
    )
    parser.add_argument(
        "--anchors",
        "-a",
        default=None,
        help="Optional JSON file or JSON string of Color Bible median chroma anchors.",
    )
    parser.add_argument(
        "--color-bible-db",
        default=None,
        help="Optional path to SQLite Color Bible DB.",
    )
    parser.add_argument(
        "--film-id",
        default=None,
        help="Optional film ID for Color Bible lookups.",
    )
    parser.add_argument(
        "--shot-index",
        type=int,
        default=0,
        help="Shot index for Color Bible palette lookups.",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=5,
        help="Guided filter radius in pixels.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-3,
        help="Guided filter regularization epsilon.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Frame rate for output video.",
    )
    parser.add_argument(
        "--preset",
        choices=["ultrafast", "fast", "medium"],
        default="fast",
        help="DIS optical flow speed/quality preset.",
    )

    args = parser.parse_args(argv)

    # Parse references JSON
    ref_arg = args.references.strip()
    if os.path.exists(ref_arg):
        with open(ref_arg, "r", encoding="utf-8") as f:
            ref_dict = json.load(f)
    else:
        ref_dict = json.loads(ref_arg)

    keyframes = {int(k): v for k, v in ref_dict.items()}

    # Parse anchors JSON if provided
    anchors = None
    if args.anchors:
        anchor_arg = args.anchors.strip()
        if os.path.exists(anchor_arg):
            with open(anchor_arg, "r", encoding="utf-8") as f:
                anchors = json.load(f)
        else:
            anchors = json.loads(anchor_arg)

    print("=" * 64)
    print("100%-OSI Video Propagation Core Engine")
    print("=" * 64)
    print(f"Source:      {args.source}")
    print(f"Keyframes:   {list(keyframes.keys())}")
    print(f"Masks:       {args.masks or 'None (Global flow)'}")
    print(f"Output:      {args.output}")
    print(f"Filter:      radius={args.radius}, eps={args.eps}")
    print(f"Preset:      {args.preset}")
    print("=" * 64)

    output_frames = colorize_shot_osi_masked_lab(
        source=args.source,
        keyframes=keyframes,
        mask_archive=args.masks,
        output_path=args.output,
        color_bible_anchors=anchors,
        color_bible_db=args.color_bible_db,
        film_id=args.film_id,
        shot_index=args.shot_index,
        radius=args.radius,
        eps=args.eps,
        fps=args.fps,
        flow_preset=args.preset,
    )

    print(f"Successfully colorized {len(output_frames)} frames.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
