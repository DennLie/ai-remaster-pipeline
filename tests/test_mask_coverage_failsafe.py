"""Tests for Mask Coverage Diagnostics and Repair Safety Fail-Safe."""

import json
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

import numpy as np
from ai_remaster_gui import dust


def test_coverage_diagnostics_calculation():
    """Verify coverage diagnostics correctly calculate mean and flag excessive coverage (>30%)."""
    meta = {'width': 100, 'height': 100}
    # 2 frames, total pixels = 20,000
    # Case A: Low coverage (100 pixels per frame = 1%)
    rows_low = [
        {'frame': 0, 'mask_pixels': 100, 'repair_pixels': 100, 'scratch_pixels': 50, 'dust_pixels': 50, 'uncertain_pixels': 0},
        {'frame': 1, 'mask_pixels': 100, 'repair_pixels': 100, 'scratch_pixels': 50, 'dust_pixels': 50, 'uncertain_pixels': 0},
    ]
    total_pixels = 2 * 100 * 100
    mean_repair_cov = sum(r['repair_pixels'] for r in rows_low) / total_pixels
    assert mean_repair_cov == 0.01

    # Case B: Excessive coverage (9,900 pixels per frame = 99%)
    rows_high = [
        {'frame': 0, 'mask_pixels': 9900, 'repair_pixels': 9900, 'scratch_pixels': 0, 'dust_pixels': 0, 'uncertain_pixels': 9900},
        {'frame': 1, 'mask_pixels': 9900, 'repair_pixels': 9900, 'scratch_pixels': 0, 'dust_pixels': 0, 'uncertain_pixels': 9900},
    ]
    mean_repair_high = sum(r['repair_pixels'] for r in rows_high) / total_pixels
    assert mean_repair_high == 0.99
    assert mean_repair_high > 0.30


def test_repair_safety_failsafe_blocks_excessive_mask(tmp_path):
    """Verify run(kind='repair') refuses inpainting when mask coverage exceeds 30% without explicit bypass."""
    # Create fake analysis directory with excessive coverage report
    target = tmp_path / "target_analysis"
    target.mkdir()

    report = {
        'identity': {'version': 3, 'source': str(tmp_path / 'source.mkv'), 'options': {'detector': 'patchcore', 'repair': 'telea'}},
        'metadata': {'width': 100, 'height': 100, 'frames': 1, 'pixel_format': 'yuv420p', 'fps': '24/1'},
        'repair_complete': False,
        'coverage_diagnostics': {
            'mean_mask_coverage_fraction': 0.998,
            'mean_repair_coverage_fraction': 0.998,
            'excessive_coverage': True,
            'warning': 'Excessive mask coverage detected: 99.8% of frame area marked as defect (safety threshold is 30.0%). Automatic repair is blocked to prevent destructive inpainting of intact footage.',
        },
        'hashes': {'input.mkv': 'fake', 'mask.mkv': 'fake', 'effective_mask.mkv': 'fake', 'measurements.csv': 'fake'},
    }
    (target / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    (target / 'mask.mkv').write_bytes(b'fake')
    (target / 'input.mkv').write_bytes(b'fake')
    (target / 'effective_mask.mkv').write_bytes(b'fake')

    def mock_manifest(tgt, src, opts, kind):
        if kind == 'repair':
            raise FileNotFoundError("repaired.mkv not ready")
        return report

    with patch('ai_remaster_gui.dust.folder_for', return_value=target), \
         patch('ai_remaster_gui.dust.manifest', side_effect=mock_manifest), \
         patch('ai_remaster_gui.dust.validate_repair_timeline'):
        
        # Should raise ValueError blocking repair
        values = {'detector': 'patchcore', 'repair': 'telea', 'research_confirmed': 'true'}
        with pytest.raises(ValueError, match="Repair refused by safety guard: Excessive mask coverage detected"):
            dust._run(tmp_path / 'source.mkv', values, kind='repair', guard=False, comfy_url='', comfy_output='', comfy_dir=None, cancel_file=None)


def test_status_reports_coverage_warning(tmp_path):
    """Verify status() propagates coverage warning and repair_blocked_reason to the GUI payload."""
    target = tmp_path / "target_analysis"
    target.mkdir()

    warning_text = "Excessive mask coverage detected: 99.8% of frame area marked as defect."
    report = {
        'identity': {'version': 3, 'source': str(tmp_path / 'source.mkv'), 'options': {'detector': 'patchcore', 'repair': 'telea', 'sensitivity': 0.5, 'radius': 3, 'dilate': 0, 'threshold': 128, 'repair_policy': 'combined_defect', 'preset': 'custom', 'checkpoint_sha256': '', 'research_confirmed': 'true', 'license_confirmed': 'true', 'dart_repo': '', 'dart_python': ''}},
        'metadata': {'width': 100, 'height': 100, 'frames': 1, 'pixel_format': 'yuv420p', 'fps': '24/1'},
        'repair_complete': False,
        'limitations': [],
        'coverage_diagnostics': {
            'mean_mask_coverage_fraction': 0.998,
            'mean_repair_coverage_fraction': 0.998,
            'excessive_coverage': True,
            'warning': warning_text,
        },
        'hashes': {'input.mkv': 'fake', 'mask.mkv': 'fake', 'effective_mask.mkv': 'fake', 'measurements.csv': 'fake'},
    }
    (target / 'report.json').write_text(json.dumps(report), encoding='utf-8')

    mock_app = MagicMock()
    mock_app.stabilized_source_for_downstream.return_value = str(tmp_path / 'source.mkv')
    mock_app.project_path = ""

    with patch('ai_remaster_gui.dust.folder_for', return_value=target), \
         patch('ai_remaster_gui.dust.saved_mask', return_value=(target, report)):
        
        values = {'detector': 'patchcore', 'repair': 'telea', 'research_confirmed': 'true', 'license_confirmed': 'true'}
        st = dust.status(mock_app, values)
        assert st.get('coverage_warning') == warning_text
        assert st.get('repair_blocked_reason') == warning_text
