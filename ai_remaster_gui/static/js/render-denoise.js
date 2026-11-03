/* Optional synchronized denoising A/B/C video previews.
 *
 * Uses only cached local ARP videos served by /media?clip_start=... .
 * Every preview is an independently encoded H.264 clip; time sync is
 * approximate in the browser, never described as frame-exact validation.
 */
let denoiseComparePaths = ['', '', ''];
let denoiseCompareStart = 0;
let denoiseCompareLength = 4;
let denoiseCompareZoom = 1;
let denoiseCompareLayout = 'abc';

function denoisePreviewIsFinal(path) {
  // Server-side stage discovery excludes partials; defend against stale
  // browser snapshots or pre-fix cached state as well.
  const filename = String(path || '').split(/[\\/]/).pop();
  return !/\.(?:partial|tmp|part)(?:\.|$)/i.test(filename);
}

function denoiseCompareCandidates(st) {
  const global = (state && state.settings && state.settings.global) || {};
  const source = st.key === 'detail'
    ? ((state && state.detail_comparison && state.detail_comparison.source) || '')
    : (global.source_sequence_preview || global.source || '');
  const candidates = [];
  if (source && denoisePreviewIsFinal(source) && /\.(mp4|mkv|webm|mov|m4v|avi)$/i.test(source)) {
    candidates.push({ path: source, label: 'Original — ' + source });
  }
  for (const file of (st.files || [])) {
    const path = file.path || '';
    if (denoisePreviewIsFinal(path) && /\.(mp4|mkv|webm|mov|m4v|avi)$/i.test(path) && !candidates.some(v => v.path === path)) {
      candidates.push({ path, label: path });
    }
  }
  return candidates;
}

function denoiseCompareClip(start, duration) {
  const a = Number(start);
  const b = Number(duration);
  if (!Number.isFinite(a) || a < 0 || a > 360000 ||
      !Number.isFinite(b) || b < 1 || b > 12) {
    return null;
  }
  return { start: a, duration: b };
}

function denoiseCompareInitializePaths(candidates) {
  const names = new Set(candidates.map(v => v.path));
  const source = candidates[0] ? candidates[0].path : '';
  const variants = candidates.slice(1).map(v => v.path);
  // Preserve manual selections without repeating them when files disappear.
  // The original is preferred in slot A and processed variants in B/C.
  const used = new Set();
  denoiseComparePaths = denoiseComparePaths.map((path, index) => {
    const preferences = index === 0 ? [source, ...variants] : [...variants, source];
    const current = path && names.has(path) && !used.has(path);
    const selected = current ? path : (preferences.find(x => x && !used.has(x)) || (index === 0 ? source : ''));
    if (selected) used.add(selected);
    return selected;
  });
}

function drawDenoise(st, settings, expected, progress) {
  const candidates = denoiseCompareCandidates(st);
  denoiseCompareInitializePaths(candidates);
  document.getElementById('app').innerHTML = `
    <div class="editor-page ${st.key === 'denoise' ? 'denoise-editor-page' : ''}">
      <section class="card">
        <h2>${esc(st.title)}</h2>
        <p>${esc(st.description)}</p>
        ${progressHtml(progress.percent, progress.label)}
        ${st.fields.map(field => fieldHtml(st, field)).join('')}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(settings)}
        <div class="actions">
          <button class="primary" onclick="runStage('${st.key}')" ${state.running ? 'disabled' : ''}>Run ${esc(st.title)}</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      ${st.key === 'denoise' ? denoiseMatrixHtml() : ''}
      <section class="card ${st.key === 'denoise' ? 'denoise-compare-card' : ''}">
        <h2>Synchronized A/B/C Comparison</h2>
        <p class="shot-empty">Choose the original and completed ${st.key === 'detail' ? 'detail enhancement' : 'denoising'} outputs. Processing files are hidden until the encoder finishes. Clips share the same source-relative start time; browser seeking is approximate.</p>
        <div class="denoise-compare-toolbar">
          <label>Start second <input id="denoiseCompareStart" type="number" min="0" max="360000" step="0.1" value="${denoiseCompareStart}"></label>
          <label>Clip duration <select id="denoiseCompareLength">
            ${[2, 4, 8, 12].map(x => `<option value="${x}" ${x === denoiseCompareLength ? 'selected' : ''}>${x}s</option>`).join('')}
          </select></label>
          <button type="button" onclick="denoiseCompareApplyClip()">Load same section</button>
          <label>Ansicht <select id="denoiseCompareLayout" onchange="denoiseCompareSetLayout(this.value)">
            <option value="abc">A / B / C · drei Ansichten</option>
            <option value="ab">A / B · groß</option>
            <option value="ac">A / C · groß</option>
          </select></label>
          <label>Display zoom <select id="denoiseCompareZoom" onchange="denoiseCompareSetZoom(this.value)">
            ${[1, 2, 4].map(x => `<option value="${x}" ${x === denoiseCompareZoom ? 'selected' : ''}>${x}x</option>`).join('')}
          </select></label>
        </div>
        <div id="denoiseComparePlayers"></div>
        <p id="denoiseCompareNotice" class="shot-empty" role="status">The playback preview is re-encoded; the original files are never overwritten.</p>
        ${st.key === 'denoise' ? denoiseReviewHtml(candidates) : ''}
      </section>
    </div>
  `;
  bindStageFields(st.key);
  showCommand(st.key);
  renderDenoiseComparePlayers(candidates);
  if (st.key === 'denoise') {
    denoiseReviewStart();
    updateDenoiseMatrixStatus();
  }
}

let denoiseMatrixRenderedReady = null;

function denoiseMatrixHtml() {
  const matrix = state.denoise_matrix || {};
  denoiseMatrixRenderedReady = Number(matrix.ready || 0);
  const count = denoiseMatrixRenderedReady;
  const busy = !!state.running;
  return `
    <section class="card denoise-matrix-card">
      <h2>Remove Noise · alle Methoden vergleichen</h2>
      <p>HQDN3D, NLMeans und BM3D jeweils in Gentle, Balanced und Strong: 9 unabhängige Render-Varianten.
        Fertige Varianten werden anhand ihrer Signatur wiederverwendet; das Original bleibt unverändert.</p>
      <div class="actions">
        <button id="denoiseMatrixRun" class="primary" type="button" onclick="denoiseMatrixRunAll()" ${busy ? 'disabled' : ''}>
          Alle 9 Varianten rendern
        </button>
      </div>
      <output id="denoiseMatrixStatus" class="denoise-matrix-status" role="status">${count}/9 Varianten fertig.</output>
      <div class="denoise-matrix-controls">
        <label for="denoiseMatrixFrames">Frames im Vergleich (1-basiert, kommasepariert, maximal 5)</label>
        <input id="denoiseMatrixFrames" type="text" placeholder="Leer = automatisch 10 %, 50 %, 90 %" autocomplete="off" spellcheck="false">
        <div class="actions">
          <button type="button" id="denoiseMatrixExport" onclick="denoiseMatrixExport()" ${busy || !count ? 'disabled' : ''}>
            Frame-Grids als ZIP erstellen
          </button>
          <a id="denoiseMatrixDownload" href="/api/denoise-grid-download" download class="denoise-grid-download" ${matrix.download ? '' : 'hidden'}>
            ZIP herunterladen · Vergleich aller Frames
          </a>
        </div>
      </div>
      <p class="shot-empty">Die ZIP enthält ein großes Übersichtsgrid, ein Grid je Frame, die Pixeldifferenzen als einzelne
        PNGs in Quellauflösung, Messwerte als CSV und eine Legende. Nur fertige Varianten kommen hinein.
        Die farblich verstärkte Differenz zeigt Pixeländerungen, nicht nachgewiesen entferntes Rauschen.</p>
    </section>`;
}

function updateDenoiseMatrixStatus() {
  if (active !== 'denoise') return;
  const matrix = state?.denoise_matrix || {};
  const count = Number(matrix.ready || 0);
  const status = document.getElementById('denoiseMatrixStatus');
  if (status) {
    const progress = (state?.phase_progress?.stages || []).find(s => s.key === 'denoise');
    const running = state.running && (state.running_stage === 'Remove Noise Batch' || state.running_stage === 'Remove Noise Grid');
    status.textContent = `${count}/9 Varianten fertig${running && progress ? ' · ' + progress.label : ''}`;
  }
  const download = document.getElementById('denoiseMatrixDownload');
  if (download) {
    download.hidden = !matrix.download;
    if (matrix.download) download.textContent = 'ZIP herunterladen · ' + (matrix.filename || 'Frame-Grids');
  }
  const run = document.getElementById('denoiseMatrixRun');
  if (run) run.disabled = !!state.running;
  const exportButton = document.getElementById('denoiseMatrixExport');
  if (exportButton) exportButton.disabled = !!state.running || !count;
  // The batch completed while polling preserved the video DOM. Rebuild once to
  // discover all new variants, then preserve interactive videos again.
  if (!state.running && denoiseMatrixRenderedReady !== null && count !== denoiseMatrixRenderedReady) {
    denoiseMatrixRenderedReady = count;
    void refresh(true);
  }
}

async function denoiseMatrixRunAll() {
  const result = await postJson('/api/denoise-batch', {});
  if (!result.ok) return alert(result.error || 'Batch konnte nicht gestartet werden.');
  await refresh(true);
}

async function denoiseMatrixExport() {
  const frames = document.getElementById('denoiseMatrixFrames')?.value.trim() || '';
  if (frames && !/^\d+(\s*,\s*\d+){0,4}$/.test(frames)) {
    return alert('Bitte 1 bis 5 unterschiedliche Frame-Nummern angeben, mit Kommas getrennt.');
  }
  const result = await postJson('/api/denoise-grid', {frames});
  if (!result.ok) return alert(result.error || 'Grid konnte nicht erstellt werden.');
  // A cached grid is immediately downloadable, a new grid appears once complete.
  if (result.state) state = result.state;
  updateDenoiseMatrixStatus();
}

function renderDenoiseComparePlayers(candidates) {
  const host = document.getElementById('denoiseComparePlayers');
  if (!host) return;
  const optionHtml = candidates.map(v =>
    `<option value="${esc(v.path)}">${esc(v.label)}</option>`).join('');
  host.innerHTML = `
    <div class="denoise-compare-grid denoise-layout-${denoiseCompareLayout}">
      ${denoiseComparePaths.map((path, i) => `
        <div class="denoise-compare-panel">
          <label for="denoiseCompareVariant${i}">${'ABC'[i]} · Variant</label>
          <select id="denoiseCompareVariant${i}" onchange="denoiseCompareChoose(${i}, this.value)" ${candidates.length ? '' : 'disabled'}>
            ${optionHtml}
          </select>
          <div class="denoise-compare-video-wrap">
            ${path ? `<video id="denoiseCompareVideo${i}" preload="metadata" muted playsinline
              src="${mediaClip(path, denoiseCompareStart, denoiseCompareStart + denoiseCompareLength, 'denoise-compare-v1')}" aria-label="Denoising comparison variant ${'ABC'[i]}"></video>` : '<p class="shot-empty">Select a video to compare.</p>'}
          </div>
        </div>
      `).join('')}
    </div>
    <div class="denoise-compare-actions">
      <button type="button" id="denoiseComparePlay" onclick="denoiseComparePlayPause()">Play all</button>
      <button type="button" onclick="denoiseCompareStep(-1)">Previous frame</button>
      <button type="button" onclick="denoiseCompareStep(1)">Next frame</button>
      <input id="denoiseCompareSeek" type="range" min="0" max="${denoiseCompareLength}" step="0.001" value="0" oninput="denoiseCompareSeek(this.value)" aria-label="Shared comparison position">
      <span id="denoiseCompareTime">0.00 / ${denoiseCompareLength.toFixed(2)} s</span>
    </div>
    <div class="denoise-compare-actions">
      <label><input id="denoiseCompareDiff" type="checkbox" onchange="denoiseCompareDifference()"> Show A–B pixel differences (preview only)</label>
    </div>
    <canvas id="denoiseCompareDiffCanvas" class="denoise-compare-diff hidden" aria-label="Pixel difference between A and B"></canvas>
  `;
  for (let i = 0; i < 3; i++) {
    const select = document.getElementById(`denoiseCompareVariant${i}`);
    if (select) select.value = denoiseComparePaths[i] || '';
  }
  const layout = document.getElementById('denoiseCompareLayout');
  if (layout) layout.value = denoiseCompareLayout;
  denoiseCompareSetZoom(denoiseCompareZoom);
  denoiseCompareWireVideos();
}

function denoiseCompareVideos() {
  return [0, 1, 2].map(i => document.getElementById(`denoiseCompareVideo${i}`)).filter(Boolean);
}

function denoiseCompareChoose(index, path) {
  if (![0, 1, 2].includes(index)) return;
  const st = (state.stages || []).find(v => v.key === active);
  const candidates = st ? denoiseCompareCandidates(st) : [];
  if (!candidates.some(v => v.path === path)) return;
  denoiseComparePaths[index] = path;
  renderDenoiseComparePlayers(candidates);
  if (active === 'denoise' && index > 0 && path !== denoiseReviewVariant && denoisePreviewIsFinal(path)) {
    denoiseReviewChoose(path);
  }
}

function denoiseCompareSetLayout(value) {
  denoiseCompareLayout = ['abc','ab','ac'].includes(value) ? value : 'abc';
  const grid = document.querySelector('.denoise-compare-grid');
  if (grid) {
    grid.classList.remove('denoise-layout-abc','denoise-layout-ab','denoise-layout-ac');
    grid.classList.add('denoise-layout-' + denoiseCompareLayout);
  }
}

function denoiseCompareApplyClip() {
  const start = document.getElementById('denoiseCompareStart');
  const duration = document.getElementById('denoiseCompareLength');
  const clipped = denoiseCompareClip(start?.value, duration?.value);
  if (!clipped) {
    denoiseCompareNotice('Use a start time between 0 and 360000 seconds and a clip length from 1 to 12 seconds.');
    return;
  }
  denoiseCompareStart = clipped.start;
  denoiseCompareLength = clipped.duration;
  const st = (state.stages || []).find(v => v.key === active);
  renderDenoiseComparePlayers(st ? denoiseCompareCandidates(st) : []);
}

function denoiseCompareNotice(message) {
  const target = document.getElementById('denoiseCompareNotice');
  if (target) target.textContent = message;
}

function denoiseCompareSetZoom(value) {
  const zoom = Number(value);
  denoiseCompareZoom = [1, 2, 4].includes(zoom) ? zoom : 1;
  document.querySelectorAll('.denoise-compare-panel video').forEach(video => {
    video.style.transform = `scale(${denoiseCompareZoom})`;
  });
}

function denoiseCompareCompatible(videos) {
  if (videos.length !== 3 || videos.some(v => v.readyState < 1)) return false;
  const first = videos[0];
  return videos.every(v =>
    v.videoWidth === first.videoWidth &&
    v.videoHeight === first.videoHeight &&
    Math.abs(v.duration - first.duration) <= 0.15 &&
    Number.isFinite(v.duration));
}

function denoiseCompareWireVideos() {
  const videos = denoiseCompareVideos();
  if (!videos.length) return;
  const master = videos[0];
  const sync = () => {
    const t = master.currentTime || 0;
    for (const other of videos.slice(1)) {
      if (other.readyState && Math.abs(other.currentTime - t) > 0.09) {
        try { other.currentTime = t; } catch (_) { /* not yet seekable */ }
      }
    }
    const seek = document.getElementById('denoiseCompareSeek');
    if (seek) seek.value = String(t);
    const label = document.getElementById('denoiseCompareTime');
    if (label) label.textContent = `${t.toFixed(2)} / ${denoiseCompareLength.toFixed(2)} s`;
    denoiseCompareDifference();
  };
  master.addEventListener('timeupdate', sync);
  master.addEventListener('seeked', sync);
  master.addEventListener('pause', () => {
    videos.slice(1).forEach(v => v.pause());
    const button = document.getElementById('denoiseComparePlay');
    if (button) button.textContent = 'Play all';
    sync();
  });
  master.addEventListener('ended', () => videos.slice(1).forEach(v => v.pause()));
  videos.forEach(video => {
    video.addEventListener('loadedmetadata', () => {
      if (videos.every(v => v.readyState >= 1)) {
        denoiseCompareNotice(
          denoiseCompareCompatible(videos)
            ? 'Same dimensions and duration. Browser synchronization is approximate; the master files are unchanged.'
            : 'Source variants differ in dimensions or duration. Synchronized playback is disabled; select matching clips.'
        );
      }
    });
    video.addEventListener('error', () => denoiseCompareNotice(
      'Could not decode a comparison preview. Check FFmpeg, source availability and the ARP log.'
    ));
  });
}

function denoiseCompareSeek(value) {
  const videos = denoiseCompareVideos();
  const t = Number(value);
  if (!Number.isFinite(t) || t < 0 || t > denoiseCompareLength) return;
  videos.forEach(v => {
    v.pause();
    if (v.readyState) {
      try { v.currentTime = t; } catch (_) { /* not yet seekable */ }
    }
  });
  const label = document.getElementById('denoiseCompareTime');
  if (label) label.textContent = `${t.toFixed(2)} / ${denoiseCompareLength.toFixed(2)} s`;
}

function denoiseComparePlayPause() {
  const videos = denoiseCompareVideos();
  if (!denoiseCompareCompatible(videos)) {
    denoiseCompareNotice('All three clips must load with matching dimensions and durations before synchronized playback.');
    return;
  }
  const master = videos[0];
  if (!master.paused) {
    videos.forEach(v => v.pause());
    return;
  }
  const t = master.ended ? 0 : master.currentTime;
  for (const v of videos) v.currentTime = t;
  Promise.all(videos.map(v => v.play())).then(() => {
    const button = document.getElementById('denoiseComparePlay');
    if (button) button.textContent = 'Pause all';
  }).catch(() => {
    videos.forEach(v => v.pause());
    denoiseCompareNotice('Browser blocked playback or a codec failed. Use the playback controls to retry.');
  });
}

function denoiseCompareStep(direction) {
  const videos = denoiseCompareVideos();
  if (!videos.length || !videos[0].readyState) return;
  const sourceFps = Number(String((state.source_info || {}).fps || '').split(' ')[0]) || 30;
  const fps = Math.max(1, Math.min(240, sourceFps));
  const delta = direction < 0 ? -1/fps : 1/fps;
  denoiseCompareSeek(Math.max(0, Math.min(denoiseCompareLength, videos[0].currentTime + delta)));
  denoiseCompareNotice('Frame step is an approximate browser seek; exact source-frame evaluation uses FFmpeg.');
}

function denoiseCompareDifference() {
  const enabled = document.getElementById('denoiseCompareDiff');
  const canvas = document.getElementById('denoiseCompareDiffCanvas');
  if (!enabled || !canvas) return;
  canvas.classList.toggle('hidden', !enabled.checked);
  if (!enabled.checked) return;
  const videos = denoiseCompareVideos();
  if (!denoiseCompareCompatible(videos)) return;
  const width = Math.min(512, videos[0].videoWidth), height = Math.round(width * videos[0].videoHeight / videos[0].videoWidth);
  if (!width || !height) return;
  // The two clips are same-origin H.264 previews; pixel differences include preview compression.
  const tmp = document.createElement('canvas');
  tmp.width = width; tmp.height = height;
  const ctx = tmp.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;
  try {
    ctx.drawImage(videos[0], 0, 0, width, height);
    const a = ctx.getImageData(0, 0, width, height);
    ctx.drawImage(videos[1], 0, 0, width, height);
    const b = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < a.data.length; i += 4) {
      const d = Math.min(255, 4 * (
        Math.abs(a.data[i] - b.data[i]) +
        Math.abs(a.data[i+1] - b.data[i+1]) +
        Math.abs(a.data[i+2] - b.data[i+2])) / 3);
      a.data[i] = d; a.data[i+1] = 0; a.data[i+2] = Math.round(d * 0.6); a.data[i+3] = 255;
    }
    canvas.width = width; canvas.height = height;
    canvas.getContext('2d').putImageData(a, 0, 0);
  } catch (_) {
    denoiseCompareNotice('Difference preview is unavailable for these video sources.');
  }
}
