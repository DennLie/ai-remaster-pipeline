import time
import pytest
from ai_remaster_gui.server import PipelineApp

def test_dust_progress_estimation_analysis():
    app = PipelineApp()
    app.running_stage_key = "dust"
    app.running_stage = "Dust & Scratches"
    app.run_started_at = time.time() - 10.0
    app.log = ["Starting dust analysis...", "Dust frame 18/72 (25%): analyzing defects"]
    
    res = app.estimate_running_progress()
    pct = res["percent"]
    label = res["label"]
    assert pct == int(95 * 18 / 72)
    assert "Analyzing defects 18/72" in label
    assert "ETA" in label
    assert pct < 90

    # Near end of analysis
    app.log = ["Dust frame 72/72 (100%): analyzing defects"]
    app.run_started_at = time.time() - 20.0
    res = app.estimate_running_progress()
    assert res["percent"] == 95
    assert "Analyzing defects 72/72" in res["label"]

    # Complete analysis
    app.log = ["Dust analysis complete", "wrote effective mask"]
    res = app.estimate_running_progress()
    assert res["percent"] == 100
    assert res["label"] == "Dust analysis complete"

def test_dust_progress_estimation_repair():
    app = PipelineApp()
    app.running_stage_key = "dust"
    app.running_stage = "Dust & Scratches"
    app.run_started_at = time.time() - 15.0
    app.log = ["Dust repair frame 36/72 (50%): repairing frame"]
    
    res = app.estimate_running_progress()
    pct = res["percent"]
    label = res["label"]
    assert pct == 50 + int(49 * 36 / 72)
    assert "Repairing defects 36/72" in label
    assert "ETA" in label

    # Complete repair
    app.log = ["Dust repair complete", "wrote repaired video"]
    res = app.estimate_running_progress()
    assert res["percent"] == 100
    assert res["label"] == "Dust repair complete"

def test_dust_progress_does_not_stall_at_90():
    app = PipelineApp()
    app.running_stage_key = "dust"
    app.running_stage = "Dust & Scratches"
    # Over 5 minutes elapsed (which previously saturated time-based formula at 90%)
    app.run_started_at = time.time() - 320.0
    app.log = ["Dust frame 60/72 (83%): analyzing defects"]
    
    res = app.estimate_running_progress()
    pct = res["percent"]
    label = res["label"]
    # 95 * 60 / 72 = 79% (actual frame progress, not clamped at 90%)
    assert pct == 79
    assert "Analyzing defects 60/72 (79%)" in label
