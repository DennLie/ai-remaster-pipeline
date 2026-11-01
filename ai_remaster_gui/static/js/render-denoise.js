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

function denoiseCompareCandidates(st) {
  const global = (state && state.settings && state.settings.global) || {};
  const source = global.source_sequence_preview || global.source || '';
  const candidates = [];
  if (source && /\.(mp4|mkv|webm|mov|m4v|avi)$/i.test(source)) {
    candidates.push({ path: source, label: 'Original — ' + source });
  }
  for (const file of (st.files || [])) {
    const path = file.path || '';
    if (/\.(mp4|mkv|webm|mov|m4v|avi)$/i.test(path) && !candidates.some(v => v.path === path)) {
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
  // Avoid hidden replacements: preserve a previous explicit selection if it still exists.
  denoiseComparePaths = denoiseComparePaths.map((path, index) =>
    path && names.has(path) ? path : (index === 0 ? source : (variants[index - 1] || source))
  );
}

function drawDenoise(st, settings, expected, progress) {
  const candidates = denoiseCompareCandidates(st);
  denoiseCompareInitializePaths(candidates);
  document.getElementById('app').innerHTML = `
    <div class="editor-page">
      <section class="card">
        <h2>Remove Noise</h2>
        <p>${esc(st.description)}</p>
        ${progressHtml(progress.percent, progress.label)}
        ${st.fields.map(field => fieldHtml(st, field)).join('')}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(settings)}
        <div class="actions">
          <button class="primary" onclick="runStage('denoise')" ${state.running ? 'disabled' : ''}>Run Remove Noise</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <div class="command" id="cmd"></div>
      </section>
      <section class="card">
        <h2>Synchronized A/B/C Comparison</h2>
        <p class="shot-empty">Choose original and any cached denoising outputs. All three preview clips use the same source-relative start time. The browser's video seeking is approximate; use the offline frame-exact report for scientific comparisons.</p>
        <div class="denoise-compare-toolbar">
          <label>Start second <input id="denoiseCompareStart" type="number" min="0" max="360000" step="0.1" value="${denoiseCompareStart}"></label>
          <label>Clip duration <select id="denoiseCompareLength">
            ${[2, 4, 8, 12].map(x => `<option value="${x}" ${x === denoiseCompareLength ? 'selected' : ''}>${x}s</option>`).join('')}
          </select></label>
          <button type="button" onclick="denoiseCompareApplyClip()">Load same section</button>
          <label>Display zoom <select id="denoiseCompareZoom" onchange="denoiseCompareSetZoom(this.value)">
            ${[1, 2, 4].map(x => `<option value="${x}" ${x === denoiseCompareZoom ? 'selected' : ''}>${x}x</option>`).join('')}
          </select></label>
        </div>
        <div id="denoiseComparePlayers"></div>
        <p id="denoiseCompareNotice" class="shot-empty" role="status">The playback preview is re-encoded; the original files are never overwritten.</p>
      </section>
    </div>
  `;
  bindStageFields('denoise');
  showCommand('denoise');
  renderDenoiseComparePlayers(candidates);
}

function renderDenoiseComparePlayers(candidates) {
  const host = document.getElementById('denoiseComparePlayers');
  if (!host) return;
  const optionHtml = candidates.map(v =>
    `<option value="${esc(v.path)}">${esc(v.label)}</option>`).join('');
  host.innerHTML = `
    <div class="denoise-compare-grid">
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
  denoiseCompareSetZoom(denoiseCompareZoom);
  denoiseCompareWireVideos();
}

function denoiseCompareVideos() {
  return [0, 1, 2].map(i => document.getElementById(`denoiseCompareVideo${i}`)).filter(Boolean);
}

function denoiseCompareChoose(index, path) {
  if (![0, 1, 2].includes(index)) return;
  const st = (state.stages || []).find(v => v.key === 'denoise');
  const candidates = st ? denoiseCompareCandidates(st) : [];
  if (!candidates.some(v => v.path === path)) return;
  denoiseComparePaths[index] = path;
  renderDenoiseComparePlayers(candidates);
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
  const st = (state.stages || []).find(v => v.key === 'denoise');
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
