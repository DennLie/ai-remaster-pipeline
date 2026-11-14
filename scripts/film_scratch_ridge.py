"""Temporal Ridge from PR #24 (15af6526), isolated from its evaluation harness.

Only damaged BGR8 input is observed. Windowing/cut abstention belongs to the
caller and is a NEW experimental configuration, not the published pilot.
"""
import math

import cv2
import numpy as np


def detect_sequence(frames, sensitivity=0.65):
    if not math.isfinite(sensitivity) or not 0 <= sensitivity <= 1:
        raise ValueError('Sensitivity must be finite in [0,1].')
    if not 3 <= len(frames) <= 96:
        raise ValueError('Require 3..96 frames.')
    h, w = frames[0].shape[:2]
    if h < 32 or w < 32 or h*w*len(frames) > 96_000_000:
        raise ValueError('Ridge geometry/memory limit exceeded; no automatic resizing.')
    if any(f.dtype != np.uint8 or f.shape != (h, w, 3) for f in frames):
        raise ValueError('Expected matching BGR8 frames.')
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    lower = np.percentile(gray, 20, axis=0).astype(np.float32)
    upper = np.percentile(gray, 80, axis=0).astype(np.float32)
    contrast, novelty = 19-7*sensitivity, 24-8*sensitivity
    flank = min(5, max(3, w//140))
    min_height = max(13, round(h*0.18))
    opening = max(7, min(13, (h//28) | 1))
    result = []
    for image in gray:
        padded = np.pad(image, ((0, 0), (flank, flank)), mode='edge')
        left, right = padded[:, :w], padded[:, 2*flank:2*flank+w]
        bright = (np.minimum(image-left, image-right) >= contrast) & (image-lower >= novelty)
        dark = (np.minimum(left-image, right-image) >= contrast) & (upper-image >= novelty)
        evidence = (bright | dark).astype(np.uint8)*255
        evidence[:, :flank+1] = 0
        evidence[:, -flank-1:] = 0
        evidence = cv2.morphologyEx(evidence, cv2.MORPH_OPEN, np.ones((opening, 1), np.uint8))
        evidence = cv2.morphologyEx(evidence, cv2.MORPH_CLOSE, np.ones((5, 1), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(evidence, 8)
        mask = np.zeros_like(evidence)
        for i in range(1, n):
            _, _, cw, ch, area = stats[i]
            if cw <= 7 and ch >= min_height and ch >= 8*cw and area >= min_height//2:
                mask[labels == i] = 255
        if np.any(mask):
            mask = cv2.dilate(mask, np.ones((1, 3), np.uint8))
        if np.mean(mask > 0) > .03:
            mask[:] = 0
        result.append(mask)
    return result
