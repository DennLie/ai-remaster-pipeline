#!/usr/bin/env python3
"""Verify PatchCore mask coverage diagnostics and repair safety fail-safe on reference video."""

import sys
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ai_remaster_gui"))

from ai_remaster_gui import dust

REFERENCE_VIDEO = ROOT / "intermediate" / "stabilized" / "Neighbors_stabilized_eb46d58c.mkv"

def run_anomaly_failsafe_verification():
    assert REFERENCE_VIDEO.is_file(), f"Reference video not found: {REFERENCE_VIDEO}"
    print(f"=== Starting PatchCore Mask Coverage & Safety Fail-Safe Verification ===")

    mask_version = uuid.uuid4().hex
    values = {
        "detector": "patchcore",
        "repair": "telea",
        "preset": "patchcore_anomaly_research",
        "threshold": 128,
        "sensitivity": 0.5,
        "dilate": 0,
        "repair_policy": "combined_defect",
        "mask_version": mask_version,
        "research_confirmed": "true",
        "license_confirmed": "true",
    }

    # Run analysis pass
    report = dust.run(REFERENCE_VIDEO, values, kind="mask")
    target_folder = dust.folder_for(REFERENCE_VIDEO, values)
    print(f"[✓] PatchCore analysis completed.")
    print(f"[✓] Output folder: {target_folder}")

    report_data = json.loads((target_folder / "report.json").read_text(encoding="utf-8"))
    cov = report_data.get("coverage_diagnostics", {})
    print(f"Coverage diagnostics: {cov}")
    assert cov.get("excessive_coverage") is True, "Expected excessive_coverage to be True for PatchCore"
    assert cov.get("mean_repair_coverage_fraction", 0.0) > 0.30, "Expected mean repair coverage > 30%"
    assert "warning" in cov, "Expected warning string in coverage diagnostics"
    print(f"[✓] Coverage Diagnostic Warning: {cov['warning']}")

    # Verify run(..., kind='repair') refuses repair without explicit bypass
    try:
        dust.run(REFERENCE_VIDEO, values, kind="repair")
        raise AssertionError("Expected repair to be blocked by safety guard!")
    except ValueError as e:
        assert "Repair refused by safety guard" in str(e)
        print(f"[✓] Repair Safety Guard successfully blocked destructive inpainting: {e}")

    print("\n=== Anomaly Detector Fail-Safe Verification PASSED Successfully! ===")

if __name__ == "__main__":
    run_anomaly_failsafe_verification()
