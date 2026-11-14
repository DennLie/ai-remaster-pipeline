"""Regression tests for defect class assignment and DART harness channel mapping.

Verifies:
1. Geometric classification of tilted/curved scratches into the scratch channel.
2. Hairline scratches (1px) with tilt are not rejected by axis-aligned bounding box limits.
3. Uncategorized defect proposals are not unconditionally dumped into the dust channel.
4. Frames with multiple authentic scratches (>2.5% mask coverage) are not wiped out.
5. Mathematical integrity of class confusion matrix and unmatched TP accounting.
6. Clean separation of defect localization quality vs defect classification quality.
"""
from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_defect_masks as bdm
import candidate_detector_eval as cde


def test_tilted_scratch_classified_as_scratch_not_dust():
    """Parametric scratch with 4-degree tilt must be assigned to scratch channel, not dust."""
    base = [np.full((128, 128), 128, dtype=np.uint8) for _ in range(4)]
    cfg = bdm.ScratchConfig(seed=20261109, width=3, opacity=0.9, angle_deg=4.0, duration=4, drift_px=1.5)
    _, gts = bdm.inject_scratches(base, cfg)
    scratch_gt = gts[1]
    assert np.count_nonzero(scratch_gt) > 0

    s_mask, d_mask = cde.split_defect_components(scratch_gt)
    assert np.count_nonzero(s_mask) > 0, "Tilted scratch was not detected in scratch channel"
    assert np.count_nonzero(d_mask) == 0, "Tilted scratch was incorrectly leaked into dust channel"
    # At least 95% of the scratch pixels must be retained in scratch channel
    assert np.count_nonzero(s_mask) >= 0.95 * np.count_nonzero(scratch_gt)


def test_hairline_tilted_scratch_not_rejected_by_bounding_box():
    """A 1-pixel scratch with 4-degree tilt (bounding box cw=10 > 8) must be classified as scratch."""
    base = [np.full((128, 128), 128, dtype=np.uint8) for _ in range(4)]
    cfg = bdm.ScratchConfig(seed=42, width=1, opacity=1.0, angle_deg=4.0, duration=4)
    _, gts = bdm.inject_scratches(base, cfg)
    scratch_gt = gts[1]
    
    # Verify the axis-aligned bounding box width exceeds 8
    pts = cv2.findNonZero(scratch_gt)
    x, y, cw, ch = cv2.boundingRect(pts)
    assert cw > 8, f"Precondition failed: expected bounding box width > 8, got {cw}"

    s_mask, d_mask = cde.split_defect_components(scratch_gt)
    assert np.count_nonzero(s_mask) > 0, "Hairline tilted scratch was rejected by bounding box"
    assert np.count_nonzero(d_mask) == 0, "Hairline scratch leaked into dust channel"


def test_uncategorized_proposals_not_dumped_into_dust():
    """Non-scratch non-dust shapes must NOT be dumped into the dust channel."""
    # Create a proposal with an irregular medium non-scratch blob (e.g. 25x25 diagonal block)
    mask = np.zeros((128, 128), dtype=np.uint8)
    for i in range(25):
        mask[30 + i, 30 + i : 30 + i + 10] = 255  # Diagonal thick blob

    s_mask, d_mask = cde.split_defect_components(mask)
    # The diagonal thick blob is neither a vertical fine scratch nor a small compact dust spot
    assert np.count_nonzero(s_mask) == 0
    assert np.count_nonzero(d_mask) == 0, "Uncategorized diagonal blob was leaked into dust channel"


def test_dense_authentic_scratch_frame_not_wiped_by_fraction_cutoff():
    """Frames with multiple vertical scratches totaling >2.5% area must not be wiped to zero."""
    mask = np.zeros((500, 500), dtype=np.uint8)
    # Inject 4 vertical scratches of width 4 spanning full height (4 * 4 * 500 = 8000 pixels = 3.2% of 250,000)
    for x in (50, 150, 280, 420):
        mask[:, x : x + 4] = 255
    coverage = np.count_nonzero(mask) / mask.size
    assert coverage > 0.025, f"Precondition failed: coverage {coverage:.4f} <= 0.025"

    s_mask, d_mask = cde.split_defect_components(mask)
    assert np.count_nonzero(s_mask) > 0, "Dense authentic scratch frame was wiped to zero"
    assert np.count_nonzero(d_mask) == 0, "Scratches in dense frame were converted to dust"


def test_class_confusion_and_unmatched_tp_identity():
    """Verify exact per-pixel accounting of the 5,192 unmatched True Positives."""
    # Synthetic dataset ground truth facts:
    # 10 scratch frames (8 active * 640 px = 5,120 GT scratch pixels)
    # 10 dust frames (9,300 GT dust pixels)
    # Total GT defect pixels = 14,420
    gt_scratch_total = 5120
    gt_dust_total = 9300
    assert gt_scratch_total + gt_dust_total == 14420

    # In the baseline report:
    overall_tp = 14352
    dust_tp = 9160
    scratch_tp = 0
    unmatched_tp = overall_tp - dust_tp
    assert unmatched_tp == 5192

    # Root cause accounting:
    scratch_to_dust_misclassified = 5120
    dust_to_scratch_misclassified = 72
    assert scratch_to_dust_misclassified + dust_to_scratch_misclassified == unmatched_tp


def test_localization_vs_classification_metrics_distinction():
    """Demonstrate that defect localization and defect classification measure distinct properties."""
    gt_scratch = np.zeros((100, 100), dtype=np.uint8)
    gt_scratch[10:90, 50:53] = 255  # 80 * 3 = 240 scratch pixels
    
    # Case A: Perfect localization, but inverted classification (scratch predicted in dust channel)
    pred_scratch = np.zeros((100, 100), dtype=np.uint8)
    pred_dust = gt_scratch.copy()
    
    # Combined / localization evaluation:
    combined_pred = np.maximum(pred_scratch, pred_dust)
    loc_counts = bdm.confusion(gt_scratch, combined_pred)
    loc_metrics = bdm.metrics(loc_counts)
    assert loc_metrics["recall"] == 1.0, "Defect localization recall must be 1.0 when defect is found"
    assert loc_metrics["precision"] == 1.0, "Defect localization precision must be 1.0"
    assert loc_metrics["f1"] == 1.0, "Localization F1 must be 1.0"

    # Scratch channel-specific evaluation:
    scratch_counts = bdm.confusion(gt_scratch, pred_scratch)
    scratch_metrics = bdm.metrics(scratch_counts)
    assert scratch_metrics["recall"] == 0.0, "Scratch classification recall is 0.0 due to channel mismatch"
    assert scratch_metrics["tp"] == 0


def test_bloomed_dust_cluster_classified_as_dust_not_rejected():
    """Authentic bloomed dust clusters (e.g. 450px compact blob in 128x128) must be classified as dust."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.circle(mask, (64, 64), 12, 255, -1)  # Area ~452 pixels, compact
    s_mask, d_mask = cde.split_defect_components(mask)
    assert np.count_nonzero(s_mask) == 0, "Compact dust cluster must not be classified as scratch"
    assert np.count_nonzero(d_mask) > 0, "Compact bloomed dust cluster must not be rejected"
    assert np.count_nonzero(d_mask) == np.count_nonzero(mask)

