#!/usr/bin/env python3
"""HTML5 Offline Visual Restoration Audit Viewer Generator.

Generates `docs/benchmarks/visual_audit/index.html` (strictly read-only inspection UI,
completely separate from the raw Track C independent human annotation workflow).
Supports direct `file:///` opening on Windows via `<script src="audit_data.js">`
without requiring any database or network server.
"""
from __future__ import annotations

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Visual Restoration Audit — E1/E2 Experimental vs Baselines</title>
<style>
  :root {
    --bg: #111317;
    --panel: #191d24;
    --panel-alt: #212630;
    --border: #2e3442;
    --text: #e6eaf2;
    --muted: #98a2b3;
    --accent: #4aa3ff;
    --green: #32f05a;
    --amber: #ffc328;
    --blue: #4696ff;
    --cyan: #00dcff;
    --magenta: #ff46dc;
    --red: #ff4141;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%; width: 100%;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px; overflow: hidden;
  }
  .layout {
    display: grid;
    grid-template-columns: 290px 1fr 330px;
    grid-template-rows: 54px 1fr;
    height: 100vh; width: 100vw;
  }
  header {
    grid-column: 1 / -1;
    background: var(--panel); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 14px; gap: 12px;
  }
  header h1 { margin: 0; font-size: 15px; font-weight: 600; letter-spacing: 0.2px; }
  .trackc-guard {
    font-size: 11px; padding: 4px 8px; border-radius: 4px;
    background: #232a36; border: 1px solid #394457; color: #b7c3d6;
  }
  .sidebar {
    background: var(--panel); border-right: 1px solid var(--border);
    display: flex; flex-direction: column; overflow: hidden;
  }
  .filter-bar {
    padding: 8px; border-bottom: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 4px;
  }
  .btn {
    background: var(--panel-alt); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
    padding: 4px 8px; cursor: pointer; font-size: 11.5px;
  }
  .btn:hover { border-color: var(--accent); }
  .btn.active { background: #1e3a5f; border-color: var(--accent); color: #fff; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .case-list {
    overflow-y: auto; padding: 6px; flex: 1;
  }
  .case-group-label {
    font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.7px;
    color: var(--muted); padding: 8px 6px 4px; font-weight: 700;
  }
  .case-item {
    padding: 7px 8px; border-radius: 5px; cursor: pointer;
    border: 1px solid transparent; margin-bottom: 4px; background: rgba(255,255,255,0.02);
  }
  .case-item:hover { background: rgba(74, 163, 255, 0.08); }
  .case-item.selected { background: rgba(74, 163, 255, 0.16); border-color: var(--accent); }
  .case-title { font-weight: 600; font-size: 12px; line-height: 1.3; }
  .case-sub { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .badge {
    display: inline-block; font-size: 10px; font-weight: 700; padding: 1px 6px;
    border-radius: 3px; text-transform: uppercase; letter-spacing: 0.4px;
  }
  .badge-gt-synth { background: rgba(50,240,90,0.16); color: #74f791; border: 1px solid rgba(50,240,90,0.35); }
  .badge-gt-clean { background: rgba(70,150,255,0.16); color: #86bdff; border: 1px solid rgba(70,150,255,0.35); }
  .badge-gt-none  { background: rgba(255,195,40,0.16); color: #ffd466; border: 1px solid rgba(255,195,40,0.38); }
  .badge-pass { background: rgba(50,240,90,0.18); color: #6ef78d; }
  .badge-fail { background: rgba(255,65,65,0.22); color: #ff8e8e; }

  .center {
    display: flex; flex-direction: column; overflow: hidden; background: #0c0d10;
  }
  .toolbar {
    background: var(--panel); border-bottom: 1px solid var(--border);
    padding: 6px 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
  }
  .toolbar-row2 {
    background: #15181e; border-bottom: 1px solid var(--border);
    padding: 5px 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
  }
  .toggle-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11.5px; background: var(--panel-alt); border: 1px solid var(--border);
    padding: 3px 7px; border-radius: 4px; cursor: pointer; user-select: none;
  }
  .toggle-pill input { margin: 0; cursor: pointer; }
  .swatch { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }

  .canvas-wrap {
    position: relative; flex: 1; overflow: hidden; cursor: grab;
    background: repeating-conic-gradient(#161920 0% 25%, #12141a 0% 50%) 50% / 16px 16px;
  }
  .canvas-wrap:active { cursor: grabbing; }
  canvas {
    display: block; width: 100%; height: 100%;
    image-rendering: pixelated; image-rendering: crisp-edges;
  }
  .slider-bar {
    display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--muted);
  }
  .coord-readout {
    position: absolute; bottom: 8px; left: 8px;
    background: rgba(12, 14, 18, 0.85); border: 1px solid var(--border);
    padding: 4px 8px; border-radius: 4px; font-family: Consolas, monospace; font-size: 11px;
    pointer-events: none;
  }
  .wipe-label {
    position: absolute; top: 8px; right: 8px;
    background: rgba(12, 14, 18, 0.85); border: 1px solid var(--border);
    padding: 4px 8px; border-radius: 4px; font-size: 11px; pointer-events: none;
  }

  .inspector {
    background: var(--panel); border-left: 1px solid var(--border);
    overflow-y: auto; padding: 10px; display: flex; flex-direction: column; gap: 10px;
  }
  .card {
    background: var(--panel-alt); border: 1px solid var(--border);
    border-radius: 6px; padding: 8px 10px;
  }
  .card h3 {
    margin: 0 0 6px; font-size: 12px; font-weight: 700;
    color: #dbe4f3; text-transform: uppercase; letter-spacing: 0.4px;
  }
  .kv-table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  .kv-table td { padding: 3px 2px; border-bottom: 1px solid rgba(255,255,255,0.05); }
  .kv-table td:first-child { color: var(--muted); }
  .kv-table td:last-child { text-align: right; font-family: Consolas, monospace; font-weight: 600; }
  .legend-row { display: flex; align-items: center; gap: 7px; font-size: 11.5px; margin-bottom: 4px; }
  .roi-btn-list { display: flex; flex-wrap: wrap; gap: 5px; }
  .asset-links a {
    display: block; color: var(--accent); text-decoration: none;
    font-size: 11.5px; margin-bottom: 4px; word-break: break-all;
  }
  .asset-links a:hover { text-decoration: underline; }
  video { width: 100%; border-radius: 4px; border: 1px solid var(--border); margin-top: 6px; background: #000; }
  .modal-backdrop {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.82);
    z-index: 50; align-items: center; justify-content: center; padding: 24px;
  }
  .modal-backdrop.open { display: flex; }
  .modal-box {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    max-width: 94vw; max-height: 92vh; overflow: auto; padding: 14px;
  }
  .modal-box img { max-width: 100%; height: auto; image-rendering: pixelated; }
</style>
</head>
<body>
<div class="layout">
  <header>
    <div style="display:flex;align-items:center;gap:12px;">
      <h1>Visual Restoration Audit — E1 / E2 vs Baselines</h1>
      <span id="hdr-provenance" class="badge badge-gt-none">Loading...</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <button class="btn" id="btn-prev-frame" title="Previous neighboring audited frame">&larr; Prev Frame</button>
      <button class="btn" id="btn-next-frame" title="Next neighboring audited frame">Next Frame &rarr;</button>
      <span class="trackc-guard">Read-Only Restoration Audit (Separate from Track C Manual GT)</span>
    </div>
  </header>

  <aside class="sidebar">
    <div class="filter-bar">
      <button class="btn active" data-filter="all">All</button>
      <button class="btn" data-filter="existing_example">8 Examples</button>
      <button class="btn" data-filter="archival_control">Archival</button>
      <button class="btn" data-filter="synthetic_defect">Synthetic</button>
      <button class="btn" data-filter="clean_reference">Clean Ref</button>
    </div>
    <div class="case-list" id="case-list"></div>
  </aside>

  <main class="center">
    <div class="toolbar">
      <label class="slider-bar">
        Mode:
        <select id="view-mode" class="btn">
          <option value="slider">Before / After Wipe Slider</option>
          <option value="orig">Original Input Only</option>
          <option value="repaired">Repaired Image Only</option>
          <option value="side_by_side">Side-by-Side (Orig | Repaired)</option>
        </select>
      </label>
      <label class="slider-bar" id="wipe-control">
        Wipe: <input type="range" id="wipe-slider" min="0" max="100" value="50" style="width:130px;" />
        <span id="wipe-pct">50%</span>
      </label>
      <div style="display:flex;gap:4px;margin-left:auto;">
        <button class="btn" data-zoom="fit">Fit</button>
        <button class="btn" data-zoom="1">1x</button>
        <button class="btn" data-zoom="2">2x</button>
        <button class="btn" data-zoom="4">4x</button>
        <button class="btn" data-zoom="8">8x</button>
        <button class="btn" id="btn-reset-view">Reset Pan</button>
      </div>
    </div>

    <div class="toolbar-row2">
      <label class="toggle-pill" id="wrap-gt"><input type="checkbox" id="ov-gt" /> <span class="swatch" style="background:var(--green)"></span>Exact GT</label>
      <label class="toggle-pill"><input type="checkbox" id="ov-cand" /> <span class="swatch" style="background:var(--cyan)"></span>Spatial / <span class="swatch" style="background:var(--magenta)"></span>Temporal Cand</label>
      <label class="toggle-pill"><input type="checkbox" id="ov-3state" checked /> <span class="swatch" style="background:var(--green)"></span>3-State (HC/Unc/Veto)</label>
      <label class="toggle-pill"><input type="checkbox" id="ov-repair" /> <span class="swatch" style="background:#ffffff"></span>Repair Mask Rim</label>
      <label class="toggle-pill"><input type="checkbox" id="ov-diff-bin" /> <span class="swatch" style="background:var(--red)"></span>Actual Mod (&gt;0)</label>
      <label class="toggle-pill"><input type="checkbox" id="ov-diff-amp" /> <span class="swatch" style="background:#ff944d"></span>Diff Heatmap (8x)</label>
      <label class="toggle-pill" id="wrap-ref-err"><input type="checkbox" id="ov-ref-err" /> <span class="swatch" style="background:#c47aff"></span>Ref Error |Rep-Clean|</label>
      <label class="toggle-pill"><input type="checkbox" id="ov-roi" checked /> ROI Box</label>
    </div>

    <div class="canvas-wrap" id="canvas-wrap">
      <canvas id="audit-canvas"></canvas>
      <div class="wipe-label" id="wipe-label">Left: Original Input | Right: Repaired Image</div>
      <div class="coord-readout" id="coord-readout">Zoom: 1.0x | Pixel: (-, -)</div>
    </div>
  </main>

  <aside class="inspector">
    <div class="card">
      <h3>Source Provenance</h3>
      <table class="kv-table" id="tbl-prov"></table>
      <div id="gt-note" style="margin-top:6px;font-size:11px;color:#ffd466;"></div>
    </div>

    <div class="card">
      <h3>Permanent Color Legend</h3>
      <div class="legend-row"><span class="swatch" style="background:rgb(50,240,90)"></span>High-Confidence Damage (Repair Authorized)</div>
      <div class="legend-row"><span class="swatch" style="background:rgb(255,195,40)"></span>Uncertain Candidate (Review Only, No Repair)</div>
      <div class="legend-row"><span class="swatch" style="background:rgb(70,150,255)"></span>Likely Intact Scene Structure (Vetoed)</div>
      <div class="legend-row"><span class="swatch" style="background:rgb(0,220,255)"></span>Spatial Candidate (FAS) / <span class="swatch" style="background:rgb(255,70,220)"></span>Temporal Cand</div>
      <div class="legend-row"><span class="swatch" style="background:rgb(255,65,65)"></span>Actual Changed Pixels (|Orig - Repaired| &gt; 0)</div>
      <div class="legend-row"><span class="swatch" style="border:2px solid rgb(40,230,80);background:transparent"></span>Exact Synthetic Injection GT (where available)</div>
      <div style="font-size:10.5px;color:var(--muted);margin-top:4px;">Note: Legacy `B_solution_1` orange overlays in prior standalone PNGs represented binary repair proposals; `A2`/`D` repair masks lack true 3-state veto separation.</div>
    </div>

    <div class="card">
      <h3>Critical ROI Shortcuts</h3>
      <div class="roi-btn-list" id="roi-shortcuts"></div>
    </div>

    <div class="card">
      <h3>True Cost &amp; Pixel Audit</h3>
      <table class="kv-table" id="tbl-metrics"></table>
    </div>

    <div class="card">
      <h3>Audit Sheets, Crops &amp; Video</h3>
      <div class="asset-links" id="asset-links"></div>
      <div id="video-box"></div>
    </div>
  </aside>
</div>

<div class="modal-backdrop" id="modal-backdrop">
  <div class="modal-box">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
      <strong id="modal-title">Inspection Sheet</strong>
      <button class="btn" id="modal-close">Close</button>
    </div>
    <img id="modal-img" src="" alt="Audit visual" />
  </div>
</div>

<script src="audit_data.js"></script>
<script>
(function() {
  const DATA = window.AUDIT_MANIFEST;
  if (!DATA || !DATA.cases || !DATA.cases.length) {
    document.getElementById("case-list").innerHTML = "<div style='padding:12px;color:#ff8e8e;'>Missing audit_data.js</div>";
    return;
  }

  const casesById = Object.fromEntries(DATA.cases.map(c => [c.case_id, c]));
  const hashId = window.location.hash ? window.location.hash.replace(/^#/, "") : "";
  let selectedId = casesById[hashId] ? hashId : (casesById["paleface_b_f8_e2"] ? "paleface_b_f8_e2" : DATA.cases[0].case_id);
  let activeFilter = "all";
  let zoom = 1.0, panX = 0.0, panY = 0.0, dragging = false, lastX = 0, lastY = 0;

  const canvas = document.getElementById("audit-canvas");
  const ctx = canvas.getContext("2d");
  const wrap = document.getElementById("canvas-wrap");
  const loadedImgs = {};

  function loadLayer(url) {
    return new Promise((resolve) => {
      if (!url) { resolve(null); return; }
      if (loadedImgs[url] && loadedImgs[url].complete) { resolve(loadedImgs[url]); return; }
      const img = new Image();
      img.onload = () => { loadedImgs[url] = img; resolve(img); };
      img.onerror = () => resolve(null);
      img.src = url;
    });
  }

  let currentLayers = {};

  async function selectCase(caseId, preserveView=false) {
    const c = casesById[caseId];
    if (!c) return;
    selectedId = caseId;
    renderSidebar();

    // Update header provenance & neighbor nav
    const provBadge = document.getElementById("hdr-provenance");
    if (c.gt_provenance === "EXACT_SYNTHETIC_INJECTION") {
      provBadge.className = "badge badge-gt-synth";
      provBadge.textContent = "EXACT SYNTHETIC INJECTION GT";
    } else if (c.gt_provenance === "CLEAN_ZERO_DEFECT_REFERENCE") {
      provBadge.className = "badge badge-gt-clean";
      provBadge.textContent = "CLEAN ZERO-DEFECT REFERENCE (GT=0)";
    } else {
      provBadge.className = "badge badge-gt-none";
      provBadge.textContent = "NO INDEPENDENT HUMAN GROUND TRUTH AVAILABLE";
    }

    const prevBtn = document.getElementById("btn-prev-frame");
    const nextBtn = document.getElementById("btn-next-frame");
    prevBtn.disabled = !c.prev_case_id;
    nextBtn.disabled = !c.next_case_id;
    prevBtn.onclick = () => c.prev_case_id && selectCase(c.prev_case_id, true);
    nextBtn.onclick = () => c.next_case_id && selectCase(c.next_case_id, true);

    const p = c.paths;
    const [orig, repaired, gtMask, spatCand, tempCand, hc, unc, veto, repMask, diffBin, diffAmp, refErr] = await Promise.all([
      loadLayer(p.orig), loadLayer(p.repaired), loadLayer(p.gt_mask),
      loadLayer(p.cand_spatial), loadLayer(p.cand_temporal),
      loadLayer(p.state_hc), loadLayer(p.state_unc), loadLayer(p.state_veto),
      loadLayer(p.repair_mask), loadLayer(p.diff_binary), loadLayer(p.diff_amp),
      loadLayer(p.ref_err_amp)
    ]);
    currentLayers = { orig, repaired, gtMask, spatCand, tempCand, hc, unc, veto, repMask, diffBin, diffAmp, refErr };

    document.getElementById("wrap-gt").style.opacity = gtMask ? "1" : "0.45";
    document.getElementById("ov-gt").disabled = !gtMask;
    if (!gtMask) document.getElementById("ov-gt").checked = false;

    document.getElementById("wrap-ref-err").style.opacity = refErr ? "1" : "0.45";
    document.getElementById("ov-ref-err").disabled = !refErr;
    if (!refErr) document.getElementById("ov-ref-err").checked = false;

    populateInspector(c);
    if (!preserveView) fitView();
    else redraw();
  }

  function fitView() {
    const img = currentLayers.orig;
    if (!img) return;
    const cw = wrap.clientWidth, ch = wrap.clientHeight;
    zoom = Math.min(cw / img.width, ch / img.height) * 0.96;
    panX = (cw - img.width * zoom) * 0.5;
    panY = (ch - img.height * zoom) * 0.5;
    redraw();
  }

  function focusROI(roi, targetZoom=6.0) {
    const img = currentLayers.orig;
    if (!img || !roi) return;
    const cw = wrap.clientWidth, ch = wrap.clientHeight;
    const rw = Math.max(8, roi.xmax - roi.xmin);
    const rh = Math.max(8, roi.ymax - roi.ymin);
    zoom = Math.min(12.0, Math.max(targetZoom, Math.min(cw / (rw * 2.2), ch / (rh * 2.2))));
    const cx = (roi.xmin + roi.xmax) * 0.5;
    const cy = (roi.ymin + roi.ymax) * 0.5;
    panX = cw * 0.5 - cx * zoom;
    panY = ch * 0.5 - cy * zoom;
    redraw();
  }

  function redraw() {
    canvas.width = wrap.clientWidth;
    canvas.height = wrap.clientHeight;
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const orig = currentLayers.orig;
    const rep = currentLayers.repaired;
    if (!orig || !rep) return;

    const c = casesById[selectedId];
    const mode = document.getElementById("view-mode").value;
    const wipePct = parseInt(document.getElementById("wipe-slider").value, 10) / 100.0;
    document.getElementById("wipe-pct").textContent = Math.round(wipePct * 100) + "%";
    document.getElementById("wipe-control").style.display = mode === "slider" ? "flex" : "none";

    const drawW = orig.width * zoom, drawH = orig.height * zoom;
    const wipeLabel = document.getElementById("wipe-label");

    function drawOverlayLayers(offsetX, clipX, clipW) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(clipX, 0, clipW, canvas.height);
      ctx.clip();
      ctx.imageSmoothingEnabled = false;

      if (document.getElementById("ov-gt").checked && currentLayers.gtMask) {
        ctx.globalAlpha = 0.65;
        ctx.drawImage(currentLayers.gtMask, offsetX, panY, drawW, drawH);
      }
      if (document.getElementById("ov-cand").checked) {
        ctx.globalAlpha = 0.72;
        if (currentLayers.spatCand) ctx.drawImage(currentLayers.spatCand, offsetX, panY, drawW, drawH);
        if (currentLayers.tempCand) ctx.drawImage(currentLayers.tempCand, offsetX, panY, drawW, drawH);
      }
      if (document.getElementById("ov-3state").checked) {
        if (currentLayers.veto) { ctx.globalAlpha = 0.65; ctx.drawImage(currentLayers.veto, offsetX, panY, drawW, drawH); }
        if (currentLayers.unc) { ctx.globalAlpha = 0.78; ctx.drawImage(currentLayers.unc, offsetX, panY, drawW, drawH); }
        if (currentLayers.hc) { ctx.globalAlpha = 0.78; ctx.drawImage(currentLayers.hc, offsetX, panY, drawW, drawH); }
      }
      if (document.getElementById("ov-repair").checked && currentLayers.repMask) {
        ctx.globalAlpha = 0.90;
        ctx.drawImage(currentLayers.repMask, offsetX, panY, drawW, drawH);
      }
      if (document.getElementById("ov-diff-bin").checked && currentLayers.diffBin) {
        ctx.globalAlpha = 0.85;
        ctx.drawImage(currentLayers.diffBin, offsetX, panY, drawW, drawH);
      }
      if (document.getElementById("ov-diff-amp").checked && currentLayers.diffAmp) {
        ctx.globalAlpha = 0.95;
        ctx.drawImage(currentLayers.diffAmp, offsetX, panY, drawW, drawH);
      }
      if (document.getElementById("ov-ref-err").checked && currentLayers.refErr) {
        ctx.globalAlpha = 0.95;
        ctx.drawImage(currentLayers.refErr, offsetX, panY, drawW, drawH);
      }
      if (document.getElementById("ov-roi").checked && c.rois && c.rois.length) {
        ctx.globalAlpha = 0.95;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 4]);
        c.rois.forEach((r, idx) => {
          ctx.strokeStyle = idx === 0 ? "#ffd466" : "#6ef78d";
          ctx.strokeRect(offsetX + r.xmin * zoom, panY + r.ymin * zoom, (r.xmax - r.xmin) * zoom, (r.ymax - r.ymin) * zoom);
        });
      }
      ctx.restore();
    }

    if (mode === "orig") {
      ctx.drawImage(orig, panX, panY, drawW, drawH);
      drawOverlayLayers(panX, 0, canvas.width);
      wipeLabel.textContent = "Showing: Original Input";
    } else if (mode === "repaired") {
      ctx.drawImage(rep, panX, panY, drawW, drawH);
      drawOverlayLayers(panX, 0, canvas.width);
      wipeLabel.textContent = "Showing: Repaired Output";
    } else if (mode === "slider") {
      const splitX = Math.round(panX + drawW * wipePct);
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, splitX, canvas.height); ctx.clip();
      ctx.drawImage(orig, panX, panY, drawW, drawH); ctx.restore();
      ctx.save(); ctx.beginPath(); ctx.rect(splitX, 0, canvas.width - splitX, canvas.height); ctx.clip();
      ctx.drawImage(rep, panX, panY, drawW, drawH); ctx.restore();
      drawOverlayLayers(panX, 0, canvas.width);
      ctx.strokeStyle = "#4aa3ff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(splitX, 0); ctx.lineTo(splitX, canvas.height); ctx.stroke();
      wipeLabel.textContent = "Left: Original | Right: Repaired (" + Math.round(wipePct * 100) + "%)";
    } else {
      const halfW = Math.floor(canvas.width * 0.5);
      const leftPanX = panX - halfW * 0.5;
      const rightPanX = panX + halfW * 0.5;
      ctx.save(); ctx.beginPath(); ctx.rect(0, 0, halfW - 1, canvas.height); ctx.clip();
      ctx.drawImage(orig, leftPanX, panY, drawW, drawH); ctx.restore();
      drawOverlayLayers(leftPanX, 0, halfW - 1);
      ctx.save(); ctx.beginPath(); ctx.rect(halfW + 1, 0, halfW - 1, canvas.height); ctx.clip();
      ctx.drawImage(rep, rightPanX, panY, drawW, drawH); ctx.restore();
      drawOverlayLayers(rightPanX, halfW + 1, halfW - 1);
      ctx.fillStyle = "#2e3442"; ctx.fillRect(halfW - 1, 0, 2, canvas.height);
      wipeLabel.textContent = "Left Viewport: Original | Right Viewport: Repaired (Synced)";
    }
    document.getElementById("coord-readout").textContent = "Zoom: " + zoom.toFixed(2) + "x | Image: " + orig.width + "x" + orig.height;
  }

  function populateInspector(c) {
    const provTbl = document.getElementById("tbl-prov");
    const m = c.metrics;
    provTbl.innerHTML = [
      ["Case ID", c.case_id],
      ["Variant", c.variant],
      ["Source Clip", c.source_clip],
      ["Source Frame #", c.source_frame_index + " (seq idx " + c.seq_frame_index + ")"],
      ["Resolution", c.resolution[0] + "x" + c.resolution[1]],
      ["GT Origin", c.gt_provenance]
    ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

    document.getElementById("gt-note").textContent = c.gt_note || "";

    const roiDiv = document.getElementById("roi-shortcuts");
    roiDiv.innerHTML = "";
    if (c.rois && c.rois.length) {
      c.rois.forEach(r => {
        const b = document.createElement("button");
        b.className = "btn";
        b.textContent = `${r.label} (${r.xmin}..${r.xmax}, ${r.ymin}..${r.ymax})`;
        b.onclick = () => focusROI(r, 6.0);
        roiDiv.appendChild(b);
      });
    } else {
      roiDiv.innerHTML = "<span style='color:var(--muted);font-size:11px;'>No critical ROI registered for this case.</span>";
    }

    const fmt = (v, suffix="") => (v === null || v === undefined) ? "N/A" : `${v}${suffix}`;
    const metRows = [
      ["Total Candidate px", m.total_candidate_pixels],
      ["HighConf Damage px", m.high_confidence_pixels],
      ["Uncertain px", m.uncertain_pixels],
      ["Vetoed Intact px", m.vetoed_intact_pixels],
      ["Authorized Repair px", m.authorized_repair_mask_pixels],
      ["Actually Modified px", m.actually_modified_pixels],
      ["Modified Outside Repair", m.modified_outside_repair_mask_pixels],
      ["FP Repair Mask px (GT)", fmt(m.fp_repair_mask_pixels)],
      ["Intact Px Modified (GT)", fmt(m.intact_pixels_modified)],
      ["Intact Edge Modified", fmt(m.intact_structural_edge_pixels_modified)],
      ["Largest FP Comp (px)", fmt(m.largest_fp_component_area_px)],
      ["Max Pixel Mod Mag", m.max_modification_magnitude],
      ["PSNR vs Clean", fmt(m.psnr_repaired_vs_clean_db, " dB")],
      ["SSIM vs Clean", fmt(m.ssim_repaired_vs_clean)],
      ["Neg Control ROI Leak", fmt(m.neg_roi_repair_leak_pixels, " px")],
      ["Neg Control Actual Mod", fmt(m.neg_roi_actual_modified_pixels, " px")]
    ];
    document.getElementById("tbl-metrics").innerHTML = metRows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

    const linksDiv = document.getElementById("asset-links");
    linksDiv.innerHTML = "";
    if (c.sheet_path) {
      const a = document.createElement("a");
      a.href = c.sheet_path; a.textContent = "Open Full-Frame 10-Panel Sheet (Modal)";
      a.onclick = (e) => { e.preventDefault(); openModal(c.title + " — Full Verification Sheet", c.sheet_path); };
      linksDiv.appendChild(a);
    }
    if (DATA.paleface_b_f8_comparison_sheet && c.case_id.indexOf("paleface_b_f8") === 0) {
      const pbs = DATA.paleface_b_f8_comparison_sheet;
      const a = document.createElement("a");
      a.href = pbs; a.textContent = "Paleface B f8: Old Destructive vs Corrected E2 Comparison Sheet";
      a.onclick = (e) => { e.preventDefault(); openModal("Paleface B f8 — Old Destructive vs Corrected E2 Comparison", pbs); };
      linksDiv.appendChild(a);
    }
    if (DATA.horizontal_fissure_investigation && DATA.horizontal_fissure_investigation.diagnostic_sheet_path && c.case_id.indexOf("horizontal_") === 0) {
      const sp = DATA.horizontal_fissure_investigation.diagnostic_sheet_path;
      const a = document.createElement("a");
      a.href = sp; a.textContent = "Section 4: 9-Panel Horizontal Fissure Sheet";
      a.onclick = (e) => { e.preventDefault(); openModal("Section 4 Horizontal Fissure Root-Cause Sheet", sp); };
      linksDiv.appendChild(a);
    }
    if (c.crop_paths && c.crop_paths.length) {
      c.crop_paths.forEach(cp => {
        const a = document.createElement("a");
        a.href = cp.path; a.textContent = "Inspect 4x/8x Zoom: " + cp.label;
        a.onclick = (e) => { e.preventDefault(); openModal(cp.label, cp.path); };
        linksDiv.appendChild(a);
      });
    }
    if (c.paths.orig_example_ref) {
      const a = document.createElement("a");
      a.href = c.paths.orig_example_ref; a.textContent = "Prior Screenshot Reference";
      a.onclick = (e) => { e.preventDefault(); openModal("Prior Screenshot: " + c.paths.orig_example_ref, c.paths.orig_example_ref); };
      linksDiv.appendChild(a);
    }

    const vbox = document.getElementById("video-box");
    vbox.innerHTML = "";
    if (c.video_browser_path) {
      vbox.innerHTML = `<div style="font-size:11px;color:var(--muted);margin-top:4px;">Side-by-Side Comparison Video:</div>
        <video controls loop preload="metadata" src="${c.video_browser_path}"></video>
        <div style="margin-top:4px;"><a style="color:var(--accent);font-size:11px;" href="${c.video_original_path}" target="_blank">Original MP4 Link (${c.video_original_path})</a></div>`;
    }
  }

  function openModal(title, src) {
    document.getElementById("modal-title").textContent = title;
    document.getElementById("modal-img").src = src;
    document.getElementById("modal-backdrop").classList.add("open");
  }
  document.getElementById("modal-close").onclick = () => document.getElementById("modal-backdrop").classList.remove("open");
  document.getElementById("modal-backdrop").onclick = (e) => {
    if (e.target.id === "modal-backdrop") document.getElementById("modal-backdrop").classList.remove("open");
  };

  function renderSidebar() {
    const list = document.getElementById("case-list");
    list.innerHTML = "";
    const filtered = DATA.cases.filter(c => activeFilter === "all" || c.tags.includes(activeFilter));
    const groups = [
      ["existing_example", "8 Existing Experimental Examples"],
      ["critical_comparison", "Critical Comparisons & Clean Refs"],
      ["neighbor_frame", "Adjacent Sequence Frames"]
    ];
    groups.forEach(([gkey, glabel]) => {
      const items = filtered.filter(c => c.group === gkey);
      if (!items.length) return;
      const lbl = document.createElement("div");
      lbl.className = "case-group-label";
      lbl.textContent = glabel;
      list.appendChild(lbl);
      items.forEach(c => {
        const div = document.createElement("div");
        div.className = "case-item" + (c.case_id === selectedId ? " selected" : "");
        const gateHtml = c.metrics.neg_roi_repair_leak_pixels === null
          ? ""
          : (c.metrics.neg_roi_repair_leak_pixels === 0
              ? `<span class="badge badge-pass">0 leak</span>`
              : `<span class="badge badge-fail">${c.metrics.neg_roi_repair_leak_pixels} leak</span>`);
        div.innerHTML = `<div class="case-title">${c.title} ${gateHtml}</div>
          <div class="case-sub">${c.variant} &bull; f${c.source_frame_index} &bull; Mod: ${c.metrics.actually_modified_pixels} px</div>`;
        div.onclick = () => selectCase(c.case_id, false);
        list.appendChild(div);
      });
    });
  }

  document.querySelectorAll(".filter-bar button").forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll(".filter-bar button").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      activeFilter = btn.dataset.filter;
      renderSidebar();
    };
  });

  document.querySelectorAll("[data-zoom]").forEach(btn => {
    btn.onclick = () => {
      const z = btn.dataset.zoom;
      if (z === "fit") { fitView(); return; }
      const img = currentLayers.orig;
      if (!img) return;
      const cw = wrap.clientWidth, ch = wrap.clientHeight;
      const cx = (cw * 0.5 - panX) / zoom, cy = (ch * 0.5 - panY) / zoom;
      zoom = parseFloat(z);
      panX = cw * 0.5 - cx * zoom;
      panY = ch * 0.5 - cy * zoom;
      redraw();
    };
  });
  document.getElementById("btn-reset-view").onclick = fitView;
  document.getElementById("view-mode").onchange = redraw;
  document.getElementById("wipe-slider").oninput = redraw;
  document.querySelectorAll(".toolbar-row2 input[type=checkbox]").forEach(el => { el.onchange = redraw; });

  wrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.25 : 0.8;
    const nextZoom = Math.min(20.0, Math.max(0.25, zoom * factor));
    const ix = (mx - panX) / zoom, iy = (my - panY) / zoom;
    zoom = nextZoom;
    panX = mx - ix * zoom;
    panY = my - iy * zoom;
    redraw();
  }, { passive: false });

  wrap.addEventListener("mousedown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener("mouseup", () => { dragging = false; });
  window.addEventListener("mousemove", (e) => {
    const img = currentLayers.orig;
    if (dragging) {
      panX += e.clientX - lastX;
      panY += e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      redraw();
    }
    if (img) {
      const rect = canvas.getBoundingClientRect();
      const ix = Math.floor((e.clientX - rect.left - panX) / zoom);
      const iy = Math.floor((e.clientY - rect.top - panY) / zoom);
      const inB = ix >= 0 && iy >= 0 && ix < img.width && iy < img.height;
      document.getElementById("coord-readout").textContent =
        `Zoom: ${zoom.toFixed(2)}x | Pixel: ${inB ? "(" + ix + ", " + iy + ")" : "(-, -)"} | Image: ${img.width}x${img.height}`;
    }
  });
  window.addEventListener("resize", redraw);

  renderSidebar();
  selectCase(selectedId, false);
})();
</script>
</body>
</html>
"""
