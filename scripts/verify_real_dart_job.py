#!/usr/bin/env python3
"""Verify a complete real 72-frame DART mask job on Neighbors_stabilized_eb46d58c.mkv."""

import sys
import time
import json
import uuid
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai_remaster_gui"))

from ai_remaster_gui import dust

REFERENCE_VIDEO = ROOT / "intermediate" / "stabilized" / "Neighbors_stabilized_eb46d58c.mkv"

def run_real_dart_verification():
    assert REFERENCE_VIDEO.is_file(), f"Reference video not found: {REFERENCE_VIDEO}"
    print(f"=== Starting Real 72-Frame DART Mask Verification ===")
    print(f"Reference Video: {REFERENCE_VIDEO}")

    mask_version = uuid.uuid4().hex
    values = {
        "detector": "dart",
        "repair": "telea",
        "preset": "dart_optimal_f1",
        "threshold": 224,
        "sensitivity": 0.65,
        "dilate": 0,
        "repair_policy": "conservative",
        "mask_version": mask_version,
        "research_confirmed": "true",
        "license_confirmed": "true",
    }

    t0 = time.perf_counter()
    report = dust.run(REFERENCE_VIDEO, values, kind="mask")
    elapsed = time.perf_counter() - t0

    target_folder = dust.folder_for(REFERENCE_VIDEO, values)
    print(f"\n[✓] DART analysis completed in {elapsed:.2f}s ({72.0 / elapsed:.2f} FPS)")
    print(f"[✓] Output folder: {target_folder}")

    # Verify all expected artifacts exist and are readable
    expected_files = [
        "input.mkv",
        "mask.mkv",
        "effective_mask.mkv",
        "scratch_mask.mkv",
        "dust_mask.mkv",
        "uncertain_mask.mkv",
        "dart_full.mkv",
        "measurements.csv",
        "report.json",
    ]

    for fname in expected_files:
        fpath = target_folder / fname
        assert fpath.is_file(), f"Expected artifact missing: {fpath}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fpath}"
        print(f"  - {fname}: {fpath.stat().st_size} bytes")

    # Verify video streams decode correctly and have exactly 72 frames
    for vname in ["mask.mkv", "scratch_mask.mkv", "dust_mask.mkv", "uncertain_mask.mkv", "effective_mask.mkv", "dart_full.mkv"]:
        cap = cv2.VideoCapture(str(target_folder / vname))
        assert cap.isOpened(), f"Cannot open video {vname}"
        count = 0
        non_zero_frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if np.any(frame > 0):
                non_zero_frames += 1
        cap.release()
        assert count == 72, f"Video {vname} frame count {count} != 72"
        print(f"  - {vname}: Decoded 72 frames successfully ({non_zero_frames} non-zero frames)")

    # Verify report diagnostics
    report_data = json.loads((target_folder / "report.json").read_text(encoding="utf-8"))
    assert "coverage_diagnostics" in report_data, "coverage_diagnostics missing in report"
    cov = report_data["coverage_diagnostics"]
    print(f"[✓] Coverage Diagnostics: Mean Repair Coverage = {cov['mean_repair_coverage_fraction']*100:.2f}%, Excessive = {cov['excessive_coverage']}")

    print("\n=== Real DART Mask Verification PASSED Successfully! ===")

if __name__ == "__main__":
    run_real_dart_verification()
