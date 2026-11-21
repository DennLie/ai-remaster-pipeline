"""Regression and Contract Tests for DART Mask Channel Integration and Fail-Safes."""

import numpy as np
import pytest
import cv2
from ai_remaster_gui.dust import ensure_uint8_binary_mask, split_proposal_components
from film_defect_detectors import MultiChannelDefectMask, RepairInclusionPolicy


def test_ensure_uint8_binary_mask_valid():
    shape = (720, 960)
    arr = np.zeros(shape, dtype=np.uint8)
    arr[10:50, 20:30] = 255
    res = ensure_uint8_binary_mask(arr, shape, "test_mask")
    assert res.dtype == np.uint8
    assert res.shape == shape
    assert res.flags.c_contiguous
    # Must succeed in OpenCV
    bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    assert bgr.shape == (720, 960, 3)


def test_ensure_uint8_binary_mask_rejects_object_dtype():
    shape = (720, 960)
    obj_arr = np.array(0, dtype=object)
    with pytest.raises(ValueError, match="object arrays are not supported"):
        ensure_uint8_binary_mask(obj_arr, shape, "uncertain_mask")
    obj_2d = np.zeros(shape, dtype=object)
    with pytest.raises(ValueError, match="object arrays are not supported"):
        ensure_uint8_binary_mask(obj_2d, shape, "uncertain_mask")


def test_ensure_uint8_binary_mask_rejects_none_and_non_arrays():
    shape = (720, 960)
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        ensure_uint8_binary_mask(None, shape, "scratch_mask")
    with pytest.raises(ValueError, match="must be a numpy.ndarray"):
        ensure_uint8_binary_mask([0, 255], shape, "dust_mask")


def test_ensure_uint8_binary_mask_rejects_nans():
    shape = (720, 960)
    arr = np.zeros(shape, dtype=np.float32)
    arr[0, 0] = float("nan")
    with pytest.raises(ValueError, match="contains non-finite values"):
        ensure_uint8_binary_mask(arr, shape, "mask")


def test_ensure_uint8_binary_mask_rejects_shape_mismatch():
    arr = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError, match="does not match expected"):
        ensure_uint8_binary_mask(arr, (720, 960), "mask")


def test_ensure_uint8_binary_mask_converts_bool():
    shape = (720, 960)
    b_arr = np.zeros(shape, dtype=bool)
    b_arr[5, 5] = True
    res = ensure_uint8_binary_mask(b_arr, shape, "mask")
    assert res.dtype == np.uint8
    assert res[5, 5] == 255
    assert res[0, 0] == 0


def test_dart_empty_and_full_component_splitting_contracts():
    """Verify DART proposal partitioning produces contiguous uint8 arrays for all channels."""
    h, w = 720, 960
    # Case 1: Empty frame (no defects)
    raw_bin_empty = np.zeros((h, w), dtype=np.uint8)
    s_mask, d_mask = split_proposal_components(raw_bin_empty)
    categorized = (s_mask > 0) | (d_mask > 0)
    uncategorized = (raw_bin_empty > 0) & ~categorized
    u_mask = np.where(uncategorized, np.uint8(255), np.uint8(0)).astype(np.uint8)

    assert u_mask.dtype == np.uint8
    assert u_mask.shape == (h, w)
    # cv2.cvtColor must succeed without throwing 'src data type = object is not supported'
    bgr_u = cv2.cvtColor(u_mask, cv2.COLOR_GRAY2BGR)
    assert bgr_u.shape == (h, w, 3)

    # MultiChannelDefectMask contract
    prob = np.zeros((h, w), dtype=np.float32)
    mcm = MultiChannelDefectMask(
        combined_probability=prob,
        combined_mask=raw_bin_empty,
        scratch_mask=s_mask,
        dust_mask=d_mask,
        uncertain_mask=u_mask,
    )
    eff = mcm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
    assert eff.shape == (h, w)
    assert eff.dtype == np.uint8

    # Case 2: Artificial hairline scratch + dust spot + large ambiguous defect
    raw_bin = np.zeros((h, w), dtype=np.uint8)
    # Hairline vertical scratch (width 4, height 300)
    raw_bin[100:400, 200:204] = 255
    # Small dust spot (5x5)
    raw_bin[50:55, 50:55] = 255
    # Large arbitrary blob (80x80) -> should land in uncertain
    raw_bin[500:580, 500:580] = 255

    s_mask, d_mask = split_proposal_components(raw_bin)
    categorized = (s_mask > 0) | (d_mask > 0)
    uncategorized = (raw_bin > 0) & ~categorized
    u_mask = np.where(uncategorized, np.uint8(255), np.uint8(0)).astype(np.uint8)

    assert s_mask.dtype == np.uint8
    assert d_mask.dtype == np.uint8
    assert u_mask.dtype == np.uint8

    # Channels must be disjoint and sum to raw_bin
    assert np.all((s_mask > 0) + (d_mask > 0) + (u_mask > 0) == (raw_bin > 0))
    # Scratch detected
    assert np.any(s_mask > 0)
    # Dust detected
    assert np.any(d_mask > 0)
    # Uncategorized blob in u_mask
    assert np.any(u_mask > 0)

    # All pass cv2.cvtColor without error
    cv2.cvtColor(s_mask, cv2.COLOR_GRAY2BGR)
    cv2.cvtColor(d_mask, cv2.COLOR_GRAY2BGR)
    cv2.cvtColor(u_mask, cv2.COLOR_GRAY2BGR)
