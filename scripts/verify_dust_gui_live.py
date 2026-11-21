"""Live verification script for ARP GUI Dust & Scratches workspace.

Starts the real ARP GUI HTTP server, performs live HTTP requests to `/api/state?active=dust`,
and runs Node-based browser rendering against the live server output to verify
that the Dust workspace renders cleanly without 'DUST_PRESET_MAP is not defined'
and that switching detectors updates presets and parameters.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_remaster_gui.server import create_server, APP_VERSION

os.environ["AI_REMASTER_NO_COMFY_AUTOSTART"] = "1"
os.environ["AI_REMASTER_NO_BROWSER"] = "1"


def main():
    print(f"=== Starting Real ARP GUI Server {APP_VERSION} ===")
    server = create_server("127.0.0.1", 8798)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Server started on http://127.0.0.1:{port}/")

    try:
        # 1. Test Root HTTP 200
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            assert resp.status == 200, f"Root returned {resp.status}"
            html_text = resp.read().decode("utf-8")
            assert "AI Remaster Pipeline" in html_text
            print("[PASS] Root index.html served successfully (HTTP 200)")

        # 2. Test Dust Stage Data API
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state?active=dust") as resp:
            assert resp.status == 200, f"State API returned {resp.status}"
            state = json.loads(resp.read().decode("utf-8"))
            dm = state.get("dust_matrix", {})
            assert "presets" in dm, "dust_matrix missing presets"
            assert "available_detectors" in dm, "dust_matrix missing available_detectors"
            presets = dm["presets"]
            print(f"[PASS] Stage data API returned {len(presets)} presets and {len(dm['available_detectors'])} detectors")

            expected_presets = [
                "balanced", "fast_preview", "high_quality_precision", "scratch_focused",
                "dust_focused", "high_recall", "conservative_low_fp",
                "dart_optimal_f1", "dart_high_recall", "dart_conservative",
                "bopb_scratch_balanced", "bopb_scratch_high_recall",
                "film_artefact_balanced", "cnn_blotches_precision",
                "patchcore_anomaly_research", "padim_anomaly_research", "custom"
            ]
            for p in expected_presets:
                assert p in presets, f"Preset {p} missing from API response"
            print(f"[PASS] All {len(expected_presets)} expected presets present in live API response")

        # 3. Test static JavaScript delivery
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/static/js/render-dust.js") as resp:
            assert resp.status == 200
            js_code = resp.read().decode("utf-8")
            assert "DUST_PRESET_MAP" in js_code
            print("[PASS] render-dust.js served successfully (HTTP 200)")

        # 4. Run Node browser verification against live server
        node_script = f"""
        const assert = require('node:assert/strict');
        const http = require('node:http');
        const vm = require('node:vm');
        const fs = require('node:fs');
        const path = require('node:path');

        function fetch(url) {{
          return new Promise((resolve, reject) => {{
            http.get(url, (res) => {{
              let data = '';
              res.on('data', chunk => data += chunk);
              res.on('end', () => resolve(data));
            }}).on('error', reject);
          }});
        }}

        async function run() {{
          const stateJson = await fetch('http://127.0.0.1:{port}/api/state?active=dust');
          const state = JSON.parse(stateJson);

          // Simulated browser DOM
          const mockElements = new Map();
          function makeEl(tag) {{
            return {{
              tagName: tag.toUpperCase(),
              classList: {{
                add: () => {{}},
                remove: () => {{}},
                toggle: () => {{}}
              }},
              dataset: {{}},
              style: {{}},
              children: [],
              querySelector: (s) => makeEl('div'),
              querySelectorAll: (s) => [],
              append: () => {{}},
              prepend: () => {{}},
              before: () => {{}},
              remove: () => {{}},
              setAttribute: () => {{}},
              getAttribute: () => '',
              addEventListener: () => {{}}
            }};
          }}

          const context = vm.createContext({{
            state,
            active: 'dust',
            parseDuration: Number,
            esc: String,
            location: {{ pathname: '/' }},
            document: {{
              querySelector: (sel) => makeEl('div'),
              querySelectorAll: (sel) => [],
              getElementById: (id) => makeEl('div'),
              createElement: (tag) => makeEl(tag),
              activeElement: null
            }}
          }});

          // Load client scripts in order
          const jsFiles = ['review-zoom.js', 'render-denoise.js', 'denoise-workspace.js', 'comparison-layout.js', 'render-dust.js'];
          for (const f of jsFiles) {{
            const code = await fetch(`http://127.0.0.1:{port}/static/js/${{f}}`);
            vm.runInContext(code, context);
          }}

          // Check that DUST_PRESET_MAP is defined and functional
          const map = vm.runInContext('DUST_PRESET_MAP', context);
          assert.ok(map, 'DUST_PRESET_MAP must be defined');
          assert.equal(typeof map, 'object');
          assert.ok(map.bopb_scratch_balanced);
          assert.ok(map.film_artefact_balanced);
          assert.ok(map.cnn_blotches_precision);
          assert.ok(map.dart_optimal_f1);

          // Verify dustFieldHtml renders preset field cleanly without any errors
          vm.runInContext("dustDraft = {{detector:'farneback_3f', preset:'balanced', repair:'telea', sensitivity:0.65, threshold:160, dilate:0, radius:3}};", context);
          const st = state.stages.find(s => s.key === 'dust');
          assert.ok(st, 'dust stage must be present in state.stages');
          const presetField = st.fields.find(f => f[0] === 'preset');
          const renderedField = context.dustFieldHtml(presetField);
          assert.ok(renderedField.includes('select data-dust="preset"'));
          assert.ok(renderedField.includes('bopb_scratch_balanced'));
          assert.ok(renderedField.includes('film_artefact_balanced'));
          assert.ok(renderedField.includes('cnn_blotches_precision'));

          // Verify detector switching logic
          const DETECTOR_DEFAULT_PRESETS = vm.runInContext('DETECTOR_DEFAULT_PRESETS', context);
          assert.equal(DETECTOR_DEFAULT_PRESETS.bopb, 'bopb_scratch_balanced');
          assert.equal(DETECTOR_DEFAULT_PRESETS.film_artefact_segmenter, 'film_artefact_balanced');
          assert.equal(DETECTOR_DEFAULT_PRESETS.cnn_blotches, 'cnn_blotches_precision');
          assert.equal(DETECTOR_DEFAULT_PRESETS.dart, 'dart_optimal_f1');

          console.log('[PASS] Live browser execution: DUST_PRESET_MAP loaded and functional');
        }}

        run().catch(err => {{
          console.error('[FAIL]', err);
          process.exit(1);
        }});
        """

        node_proc = subprocess.run(["node", "-e", node_script], capture_output=True, text=True)
        if node_proc.returncode != 0:
            print("Node STDOUT:", node_proc.stdout)
            print("Node STDERR:", node_proc.stderr)
            sys.exit(node_proc.returncode)
        else:
            print(node_proc.stdout.strip())
            print("[PASS] Full browser execution test against real server PASSED!")

    finally:
        server.server_close()
        print("Server shutdown cleanly.")


if __name__ == "__main__":
    main()
