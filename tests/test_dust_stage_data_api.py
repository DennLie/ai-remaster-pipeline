"""Regression and verification tests for Dust & Scratches stage data API.

Verifies:
1. `app.state('dust')` returns valid `dust_matrix`, `available_detectors`, `presets`,
   and `available_presets`.
2. All 10 existing presets + 6 newly integrated neural detector presets are preserved.
3. Switching between DART, BOPB, Film Artefact Segmenter, and CNN Blotches & Scratches
   correctly resolves the corresponding presets and detector parameters.
4. HTTP handler response contains complete stage data and valid fields schema.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
import pytest

from ai_remaster_gui.config import ROOT
from ai_remaster_gui.server import PipelineApp
import ai_remaster_gui.dust as dust_stage
from scripts.film_defect_detectors import BENCHMARK_PRESETS


def test_dust_stage_data_api_structure():
    """Verify app.state('dust') returns all required stage data fields."""
    app = PipelineApp()
    app.settings.setdefault("global", {})["dust"] = "true"
    state = app.state("dust")
    assert "stages" in state
    assert "dust_matrix" in state
    dm = state["dust_matrix"]

    # Verify dust_matrix structure
    assert "available_detectors" in dm
    assert "presets" in dm
    assert "available_presets" in dm

    # Check detector choices (note: dart is registered as dart_harness in registry)
    det_names = {d["name"] for d in dm["available_detectors"]}
    expected_detectors = {
        "farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy",
        "dart_harness", "bopb", "film_artefact_segmenter", "cnn_blotches",
        "patchcore", "padim", "correl_net",
    }
    assert expected_detectors.issubset(det_names), f"Missing detectors: {expected_detectors - det_names}"

    # Verify all 16 canonical presets + custom are exposed
    presets = dm["presets"]
    expected_presets = [
        "balanced", "fast_preview", "high_quality_precision", "scratch_focused",
        "dust_focused", "high_recall", "conservative_low_fp",
        "dart_optimal_f1", "dart_high_recall", "dart_conservative",
        "bopb_scratch_balanced", "bopb_scratch_high_recall",
        "film_artefact_balanced", "cnn_blotches_precision",
        "patchcore_anomaly_research", "padim_anomaly_research",
        "custom"
    ]
    for p_name in expected_presets:
        assert p_name in presets, f"Preset {p_name!r} missing from state.dust_matrix.presets"
        p_cfg = presets[p_name]
        assert "detector" in p_cfg
        assert "sensitivity" in p_cfg
        assert "threshold" in p_cfg
        assert "dilate" in p_cfg
        assert "commercial_cleared" in p_cfg


def test_dust_stage_field_definitions():
    """Verify Stage('dust') fields include all preset choices and detector choices."""
    app = PipelineApp()
    app.settings.setdefault("global", {})["dust"] = "true"
    state = app.state("dust")
    dust_stage_def = next((s for s in state["stages"] if s["key"] == "dust"), None)
    assert dust_stage_def is not None, "Dust stage definition missing from state['stages']"

    field_map = {f[0]: f for f in dust_stage_def["fields"]}
    assert "preset" in field_map
    assert "detector" in field_map

    preset_kind = field_map["preset"][2]
    assert preset_kind.startswith("select:")
    preset_choices = set(preset_kind.removeprefix("select:").split("|"))
    expected_presets = {
        "balanced", "fast_preview", "high_quality_precision", "scratch_focused",
        "dust_focused", "high_recall", "conservative_low_fp",
        "dart_optimal_f1", "dart_high_recall", "dart_conservative",
        "bopb_scratch_balanced", "bopb_scratch_high_recall",
        "film_artefact_balanced", "cnn_blotches_precision",
        "patchcore_anomaly_research", "padim_anomaly_research",
        "custom"
    }
    assert expected_presets.issubset(preset_choices), f"Missing preset choices: {expected_presets - preset_choices}"

    det_kind = field_map["detector"][2]
    det_choices = set(det_kind.removeprefix("select:").split("|"))
    expected_dets = {
        "farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy",
        "dart", "bopb", "film_artefact_segmenter", "cnn_blotches",
        "patchcore", "padim", "correl_net"
    }
    assert expected_dets.issubset(det_choices), f"Missing detector choices: {expected_dets - det_choices}"


def test_switching_neural_detectors_resolves_presets_and_parameters():
    """Verify switching detector to DART, BOPB, Film Artefact Segmenter, and CNN Blotches

    correctly resolves the corresponding preset and calibrated parameters.
    """
    # 1. DART (preset resolution)
    dart_opts = dust_stage.options({"preset": "dart_optimal_f1"})
    assert dart_opts["detector"] == "dart"
    assert dart_opts["preset"] == "dart_optimal_f1"
    assert dart_opts["threshold"] == 224
    assert dart_opts["sensitivity"] == 0.65
    assert dart_opts["repair_policy"] == "conservative"

    # 2. BOPB (preset resolution)
    bopb_opts = dust_stage.options({"preset": "bopb_scratch_balanced"})
    assert bopb_opts["detector"] == "bopb"
    assert bopb_opts["preset"] == "bopb_scratch_balanced"
    assert bopb_opts["threshold"] == 128
    assert bopb_opts["sensitivity"] == 0.5
    assert bopb_opts["repair_policy"] == "scratch_only"

    # 3. Film Artefact Segmenter (preset resolution)
    fas_opts = dust_stage.options({"preset": "film_artefact_balanced"})
    assert fas_opts["detector"] == "film_artefact_segmenter"
    assert fas_opts["preset"] == "film_artefact_balanced"
    assert fas_opts["threshold"] == 128
    assert fas_opts["sensitivity"] == 0.5
    assert fas_opts["repair_policy"] == "conservative"

    # 4. CNN Blotches & Scratches (preset resolution)
    cnn_opts = dust_stage.options({"preset": "cnn_blotches_precision"})
    assert cnn_opts["detector"] == "cnn_blotches"
    assert cnn_opts["preset"] == "cnn_blotches_precision"
    assert cnn_opts["threshold"] == 204
    assert cnn_opts["sensitivity"] == 0.8
    assert cnn_opts["repair_policy"] == "conservative"


def test_explicit_preset_selection_overrides_detector():
    """Verify explicitly choosing a preset sets the detector and calibrated parameters."""
    bopb_hr = dust_stage.options({"preset": "bopb_scratch_high_recall"})
    assert bopb_hr["detector"] == "bopb"
    assert bopb_hr["threshold"] == 76
    assert bopb_hr["sensitivity"] == 0.3
    assert bopb_hr["preset"] == "bopb_scratch_high_recall"

    dart_hr = dust_stage.options({"preset": "dart_high_recall"})
    assert dart_hr["detector"] == "dart"
    assert dart_hr["threshold"] == 192
    assert dart_hr["preset"] == "dart_high_recall"


def test_frontend_node_eval_against_real_state_payload(tmp_path):
    """Evaluate render-dust.js in Node against real state payload to prevent ReferenceError regressions."""
    app = PipelineApp()
    app.settings.setdefault("global", {})["dust"] = "true"
    state = app.state("dust")

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps(state), encoding="utf-8")

    js_file = tmp_path / "eval_runner.cjs"
    js_content = f"""
    const assert = require('node:assert/strict');
    const fs = require('node:fs');
    const vm = require('node:vm');
    const path = require('node:path');

    const state = JSON.parse(fs.readFileSync({repr(str(state_file))}, 'utf8'));
    const context = vm.createContext({{
      state,
      active: 'dust',
      parseDuration: Number,
      esc: String,
      location: {{ pathname: '/' }},
      document: {{
        querySelector: () => null,
        querySelectorAll: () => [],
        getElementById: () => null
      }}
    }});

    // Load scripts
    for (const f of ['review-zoom.js', 'render-denoise.js', 'denoise-workspace.js', 'comparison-layout.js', 'render-dust.js']) {{
      vm.runInContext(fs.readFileSync(path.join('{ROOT.as_posix()}', 'ai_remaster_gui/static/js', f), 'utf8'), context);
    }}

    // Check that DUST_PRESET_MAP is defined and contains presets
    const map = vm.runInContext('DUST_PRESET_MAP', context);
    assert.ok(map, 'DUST_PRESET_MAP must be defined');
    assert.equal(typeof map, 'object');
    assert.ok(map.balanced);
    assert.ok(map.bopb_scratch_balanced);
    assert.ok(map.film_artefact_balanced);
    assert.ok(map.cnn_blotches_precision);
    assert.ok(map.dart_optimal_f1);

    // Render preset field through dustFieldHtml for farneback_3f
    vm.runInContext("dustDraft = {{detector:'farneback_3f', preset:'balanced', repair:'telea', sensitivity:0.65, threshold:160, dilate:1}};", context);
    const dustStage = state.stages.find(s => s.key === 'dust');
    assert.ok(dustStage, 'dust stage must be present in state.stages');
    const presetField = dustStage.fields.find(f => f[0] === 'preset');
    assert.ok(presetField, 'preset field must be present');
    const farnebackHtml = context.dustFieldHtml(presetField);
    assert.ok(farnebackHtml.includes('select data-dust="preset"'));
    assert.ok(farnebackHtml.includes('value="balanced"'));

    // Render preset field for bopb detector
    vm.runInContext("dustDraft.detector = 'bopb'; dustDraft.preset = 'bopb_scratch_balanced';", context);
    const bopbHtml = context.dustFieldHtml(presetField);
    assert.ok(bopbHtml.includes('bopb_scratch_balanced'));

    // Render preset field for film_artefact_segmenter detector
    vm.runInContext("dustDraft.detector = 'film_artefact_segmenter'; dustDraft.preset = 'film_artefact_balanced';", context);
    const fasHtml = context.dustFieldHtml(presetField);
    assert.ok(fasHtml.includes('film_artefact_balanced'));
    """
    js_file.write_text(js_content, encoding="utf-8")

    res = subprocess.run(["node", str(js_file)], capture_output=True, text=True)
    assert res.returncode == 0, f"Node script failed:\n{res.stdout}\n{res.stderr}"
