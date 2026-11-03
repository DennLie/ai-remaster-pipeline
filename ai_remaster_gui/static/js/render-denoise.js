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
let denoiseCompareEffectiveLength = 0;
let denoiseCompareLengthSection = '';

const DENOISE_METHODS = [
  { key: 'hqdn3d', label: 'HQDN3D' },
  { key: 'nlmeans', label: 'NLMeans' },
  { key: 'bm3d', label: 'BM3D' },
];
const DENOISE_PRESETS = [
  { key: 'gentle', label: 'Gentle' },
  { key: 'balanced', label: 'Balanced' },
  { key: 'strong', label: 'Strong' },
];
let denoiseMatrixSelected = null;
let denoiseMatrixRenderedReady = null;
let denoiseUseVariantPending = false;
let denoiseBatchPending = false;
let denoiseGridPending = false;

function denoiseVariantLabelFor(method, preset) {
  if (method === 'off') return 'Off';
  const m = DENOISE_METHODS.find(x => x.key === method);
  const p = DENOISE_PRESETS.find(x => x.key === preset);
  return m && p ? `${m.label} · ${p.label}` : 'Variant';
}

function denoiseVariantFromPath(path) {
  const match = String(path || '').match(/_denoised_(hqdn3d|nlmeans|bm3d)_(gentle|balanced|strong)_[0-9a-f]+\.mkv$/i);
  if (!match) return null;
  return { method: match[1].toLowerCase(), preset: match[2].toLowerCase() };
}

function denoiseVariantLabel(path) {
  const parsed = denoiseVariantFromPath(path);
  if (parsed) return denoiseVariantLabelFor(parsed.method, parsed.preset);
  const name = String(path || '').split(/[\\/]/).pop();
  return name || 'Variant';
}

function denoiseVariantPath(method, preset) {
  const st = (state?.stages || []).find(x => x.key === 'denoise');
  const files = st?.files || [];
  const pattern = new RegExp(`_denoised_${method}_${preset}_[0-9a-f]+\\.mkv$`, 'i');
  return (files.find(file => pattern.test(file.path || '')) || {}).path || '';
}

function denoiseVariantReady(method, preset) {
  if (method === 'off') return true;
  const variants = state?.denoise_matrix?.variants;
  if (Array.isArray(variants)) return variants.some(v => v.method === method && v.preset === preset);
  return Boolean(denoiseVariantPath(method, preset));
}

function denoiseStageVariant() {
  const values = state?.settings?.denoise || {};
  const method = values.method || 'off';
  const preset = values.preset || 'gentle';
  if (method === 'off' || !DENOISE_METHODS.some(x => x.key === method) ||
      !DENOISE_PRESETS.some(x => x.key === preset)) return { method: 'off', preset: '' };
  return { method, preset };
}

function denoiseSelectedVariant() {
  if (denoiseMatrixSelected &&
      (denoiseMatrixSelected.method === 'off' ||
       (DENOISE_METHODS.some(x => x.key === denoiseMatrixSelected.method) &&
        DENOISE_PRESETS.some(x => x.key === denoiseMatrixSelected.preset)))) {
    return denoiseMatrixSelected;
  }
  const current = denoiseStageVariant();
  if (current.method === 'off') {
    denoiseMatrixSelected = { method: 'off', preset: '' };
    return denoiseMatrixSelected;
  }
  if (denoiseVariantReady(current.method, current.preset)) {
    denoiseMatrixSelected = { method: current.method, preset: current.preset };
    return denoiseMatrixSelected;
  }
  for (const method of DENOISE_METHODS) {
    for (const preset of DENOISE_PRESETS) {
      if (denoiseVariantReady(method.key, preset.key)) {
        denoiseMatrixSelected = { method: method.key, preset: preset.key };
        return denoiseMatrixSelected;
      }
    }
  }
  denoiseMatrixSelected = { method: 'bm3d', preset: 'balanced' };
  return denoiseMatrixSelected;
}

function denoiseSelectedPath() {
  const selected = denoiseSelectedVariant();
  return selected.method === 'off' ? '' : denoiseVariantPath(selected.method, selected.preset);
}

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
    candidates.push({ path: source, label: 'Original — ' + source, kind: 'source' });
  }
  for (const file of (st.files || [])) {
    const path = file.path || '';
    if (denoisePreviewIsFinal(path) && /\.(mp4|mkv|webm|mov|m4v|avi)$/i.test(path) && !candidates.some(v => v.path === path)) {
      candidates.push({ path, label: denoiseVariantLabel(path), kind: 'variant' });
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

function denoiseCompareAvailableLength() {
  const section = state?.source_section || {};
  const sourceDuration = parseDuration(state?.source_info?.duration || '0');
  const start = Number(section.enabled ? section.start : 0) || 0;
  const end = Number(section.enabled ? section.end : sourceDuration) || sourceDuration;
  return Math.max(0, end - start);
}

function denoiseCompareSectionKey() {
  const section = state?.source_section || {};
  return `${section.enabled ? section.start : 0}-${section.enabled ? section.end : ''}`;
}

function denoiseCompareFullDuration() {
  const available = denoiseCompareAvailableLength();
  if (!(available >= 1)) return 0;
  return Math.min(12, available);
}

function denoiseCompareEnsureLength() {
  const full = denoiseCompareFullDuration();
  const key = denoiseCompareSectionKey();
  if (key !== denoiseCompareLengthSection) {
    denoiseCompareLengthSection = key;
    if (full) denoiseCompareLength = full;
  }
  if (full && denoiseCompareLength > full) denoiseCompareLength = full;
  if (!Number.isFinite(denoiseCompareLength) || denoiseCompareLength < 1) {
    denoiseCompareLength = full || 4;
  }
  if (denoiseCompareLength > 12) denoiseCompareLength = 12;
  denoiseCompareEffectiveLength = full
    ? Math.min(denoiseCompareLength, full)
    : denoiseCompareLength;
}

function denoiseCompareDurationOptions() {
  const full = denoiseCompareFullDuration();
  const values = [2, 4, 8, 12].filter(value => !full || value < full - 0.001);
  if (full) values.push(full);
  const unique = [...new Set(values)].sort((a, b) => a - b);
  return unique.map(value => {
    const isFull = full > 0 && Math.abs(value - full) < 0.001;
    const suffix = isFull ? (full < 12 ? ' · full clip' : ' · max preview') : '';
    return `<option value="${value}" ${Math.abs(value - denoiseCompareLength) < 0.001 ? 'selected' : ''}>${value}s${suffix}</option>`;
  }).join('');
}

function denoiseCompareSetEffectiveLength(value) {
  const actual = Number(value);
  const maximum = Number.isFinite(actual) && actual > 0 ? actual : denoiseCompareLength;
  denoiseCompareEffectiveLength = Math.max(0, Math.min(denoiseCompareLength, maximum));
  const seek = document.getElementById('denoiseCompareSeek');
  if (seek) {
    seek.max = String(denoiseCompareEffectiveLength);
    if (Number(seek.value) > denoiseCompareEffectiveLength) {
      seek.value = String(denoiseCompareEffectiveLength);
    }
  }
  const label = document.getElementById('denoiseCompareTime');
  if (label) {
    const current = Math.min(Number(seek?.value || 0), denoiseCompareEffectiveLength);
    label.textContent = `${current.toFixed(2)} / ${denoiseCompareEffectiveLength.toFixed(2)} s`;
  }
}

function denoiseCompareInitializePaths(candidates) {
  const names = new Set(candidates.map(v => v.path));
  const source = candidates[0] ? candidates[0].path : '';
  const variants = candidates.slice(1).map(v => v.path);
  // Preserve manual selections without repeating them when files disappear.
  // The original is preferred in slot A, the matrix choice in B, and the rest in C.
  const preferred = typeof denoiseSelectedPath === 'function' ? denoiseSelectedPath() : '';
  const used = new Set();
  denoiseComparePaths = denoiseComparePaths.map((path, index) => {
    const preferences = index === 0
      ? [source, ...variants]
      : [...(index === 1 && preferred ? [preferred] : []), ...variants, source];
    const current = path && names.has(path) && !used.has(path);
    const selected = current ? path : (preferences.find(x => x && !used.has(x)) || (index === 0 ? source : ''));
    if (selected) used.add(selected);
    return selected;
  });
}

function drawDenoise(st, settings, expected, progress) {
  const candidates = denoiseCompareCandidates(st);
  denoiseCompareInitializePaths(candidates);
  denoiseCompareEnsureLength();
  document.getElementById('app').innerHTML = `
    <div class="editor-page ${st.key === 'denoise' ? 'denoise-editor-page' : ''}">
      <aside class="card denoise-settings-sidebar" aria-label="${st.key === 'denoise' ? 'Remove Noise settings' : 'Detail enhancement settings'}">
        <h2>${esc(st.title)}</h2>
        <p>${esc(st.description)}</p>
        ${progressHtml(progress.percent, progress.label)}
        ${st.key === 'denoise' ? denoiseMatrixHtml() : st.fields.map(field => fieldHtml(st, field)).join('')}
        ${shotOutputList(expected, null)}
        ${stageCheckboxes(settings)}
        <div class="actions">
          <button class="primary" onclick="runStage('${st.key}')" ${state.running ? 'disabled' : ''}>Run ${esc(st.title)}</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <details class="command-details"><summary>Command</summary><div class="command" id="cmd"></div></details>
      </aside>
      <section class="card denoise-compare-card">
        <h2>Synchronized A/B/C Comparison</h2>
        <p class="shot-empty">Choose the original and completed ${st.key === 'detail' ? 'detail enhancement' : 'denoising'} outputs. Processing files are hidden until the encoder finishes. Clips share the same source-relative start time; browser seeking is approximate.</p>
        <div class="denoise-compare-toolbar">
          <label>Start time (s) <input id="denoiseCompareStart" type="number" min="0" max="360000" step="0.1" value="${denoiseCompareStart}"></label>
          <label>Clip duration <select id="denoiseCompareLength">
            ${denoiseCompareDurationOptions()}
          </select></label>
          <button type="button" onclick="denoiseCompareApplyClip()">Apply to all three</button>
          <label>View <select id="denoiseCompareLayout" onchange="denoiseCompareSetLayout(this.value)">
            <option value="abc">A / B / C · three views</option>
            <option value="ab">A / B · large</option>
            <option value="ac">A / C · large</option>
          </select></label>
          <label>Display zoom <select id="denoiseCompareZoom" onchange="denoiseCompareSetZoom(this.value)">
            ${[1, 2, 4].map(x => `<option value="${x}" ${x === denoiseCompareZoom ? 'selected' : ''}>${x}x</option>`).join('')}
          </select></label>
        </div>
        <div id="denoiseComparePlayers"></div>
        <p id="denoiseCompareNotice" class="shot-empty" role="status">The playback preview is re-encoded; the original files are never overwritten.</p>
      </section>
      ${st.key === 'denoise' ? denoiseReviewHtml(candidates) : ''}
    </div>
  `;
  bindStageFields(st.key);
  showCommand(st.key);
  renderDenoiseComparePlayers(candidates);
  if (st.key === 'denoise') {
    denoiseReviewStart();
    denoiseMatrixUpdateUi();
    updateDenoiseMatrixStatus();
  }
}

function denoiseMatrixHtml() {
  const matrix = state.denoise_matrix || {};
  denoiseMatrixRenderedReady = Number(matrix.ready || 0);
  const count = denoiseMatrixRenderedReady;
  const busy = !!state.running;
  const selected = denoiseSelectedVariant();
  const stage = denoiseStageVariant();
  const raw = state?.settings?.denoise || {};
  const selectedLabel = denoiseVariantLabelFor(selected.method, selected.preset);
  const stageLabel = denoiseVariantLabelFor(stage.method, stage.preset);
  return `
    <section class="denoise-variant-picker" aria-labelledby="denoiseVariantHeading">
      <input type="hidden" data-field="method" value="${esc(raw.method || 'off')}">
      <input type="hidden" data-field="preset" value="${esc(raw.preset || 'gentle')}">
      <div class="denoise-variant-head">
        <div>
          <h3 id="denoiseVariantHeading">Variant</h3>
          <p class="denoise-variant-help">Nine method and strength combinations. Ready variants compare immediately; missing variants render on demand.</p>
        </div>
        <output id="denoiseMatrixStatus" class="denoise-matrix-status" role="status">${count}/9 rendered</output>
      </div>
      <div class="denoise-variant-grid" role="radiogroup" aria-label="Denoise variant" onkeydown="denoiseMatrixKeyboard(event)">
        <span class="denoise-variant-corner" aria-hidden="true"></span>
        ${DENOISE_METHODS.map(method => `<span class="denoise-variant-col">${method.label}</span>`).join('')}
        ${DENOISE_PRESETS.map(preset => `
          <span class="denoise-variant-row">${preset.label}</span>
          ${DENOISE_METHODS.map(method => denoiseMatrixCellHtml(method, preset, selected)).join('')}
        `).join('')}
      </div>
      <button type="button" class="denoise-off-option ${stage.method === 'off' ? 'is-current' : ''}" onclick="denoiseMatrixUseOff()" ${busy ? 'disabled' : ''} aria-pressed="${stage.method === 'off'}">
        Off · keep the source unchanged
      </button>
      <div class="denoise-selected-variant" data-denoise-selected aria-live="polite">
        <div class="denoise-selected-main">
          <strong class="denoise-selected-name" data-dm-selected-label>${selectedLabel}</strong>
          <span class="denoise-selected-state" data-dm-selected-state></span>
        </div>
        <div class="denoise-selected-meta" data-dm-selected-meta></div>
        <div class="denoise-selected-output" data-dm-selected-output></div>
        <div class="denoise-selected-actions">
          <button type="button" class="primary" id="denoiseUseVariant" onclick="denoiseMatrixUseSelected()" ${busy ? 'disabled' : ''}>Use as stage output</button>
          <button type="button" id="denoiseCopyVariant" onclick="denoiseMatrixCopySelected()" ${busy ? 'disabled' : ''}>Copy settings</button>
        </div>
        <small class="denoise-selected-stage">Stage output: <strong data-dm-stage-output>${stageLabel}</strong></small>
        <output class="denoise-selected-notice" data-dm-notice role="status"></output>
      </div>
      <details class="denoise-batch-export">
        <summary>Batch export · ${count}/9 rendered</summary>
        <p>Render all nine variants at once, reuse matching signatures, then export frame grids and pixel differences as a ZIP.</p>
        <div class="actions">
          <button id="denoiseMatrixRun" class="primary" type="button" onclick="denoiseMatrixRunAll()" ${busy ? 'disabled' : ''}>Render all 9 variants</button>
        </div>
        <div class="denoise-matrix-controls">
          <label for="denoiseMatrixFrames">Frames to compare (1-based, comma-separated, max 5)</label>
          <input id="denoiseMatrixFrames" type="text" placeholder="Empty = automatic 10%, 50%, 90%" autocomplete="off" spellcheck="false">
          <div class="actions">
            <button type="button" id="denoiseMatrixExport" onclick="denoiseMatrixExport()" ${busy || !count ? 'disabled' : ''}>Create frame grids as ZIP</button>
            <a id="denoiseMatrixDownload" href="/api/denoise-grid-download" download class="denoise-grid-download" ${matrix.download ? '' : 'hidden'}>Download ZIP · all-frame comparison</a>
          </div>
        </div>
        <p class="shot-empty">The ZIP contains an overview grid, one grid per frame, pixel-difference PNGs at source resolution, metrics as CSV and a legend. Only finished variants are included. The amplified difference shows pixel changes, not proven noise removal.</p>
      </details>
    </section>`;
}

function denoiseMatrixCellHtml(method, preset, selected) {
  const ready = denoiseVariantReady(method.key, preset.key);
  const isSelected = selected.method === method.key && selected.preset === preset.key;
  const noSelection = selected.method === 'off';
  const firstCell = method.key === DENOISE_METHODS[0].key && preset.key === DENOISE_PRESETS[0].key;
  const tabIndex = isSelected || (noSelection && firstCell) ? '0' : '-1';
  const label = `${method.label} · ${preset.label}`;
  const stateLabel = ready ? 'ready' : 'not rendered';
  return `<button type="button" role="radio" aria-checked="${isSelected}" aria-label="${label}, ${stateLabel}" tabindex="${tabIndex}"
    class="denoise-variant-cell ${ready ? 'is-ready' : 'is-missing'} ${isSelected ? 'is-selected' : ''}"
    data-denoise-cell data-method="${method.key}" data-preset="${preset.key}"
    onclick="denoiseMatrixSelect('${method.key}','${preset.key}')" ${state.running ? 'disabled' : ''}>
    <span class="denoise-variant-cell-state">${ready ? 'Ready' : '—'}</span>
  </button>`;
}

function denoiseMatrixKeyboard(event) {
  const cell = event.target.closest('[data-denoise-cell]');
  if (!cell) return;
  const cells = [...document.querySelectorAll('[data-denoise-cell]')];
  const index = cells.indexOf(cell);
  if (index < 0) return;
  let next = -1;
  if (event.key === 'ArrowRight') next = index + 1;
  else if (event.key === 'ArrowLeft') next = index - 1;
  else if (event.key === 'ArrowDown') next = index + 3;
  else if (event.key === 'ArrowUp') next = index - 3;
  else if (event.key === 'Home') next = 0;
  else if (event.key === 'End') next = cells.length - 1;
  if (next >= 0 && next < cells.length) {
    event.preventDefault();
    cells[next].focus();
    cells[next].click();
  }
}

function denoiseSameVariant(a, b) {
  return a.method === b.method && (a.method === 'off' || a.preset === b.preset);
}

function denoiseMatrixUpdateUi() {
  const selected = denoiseSelectedVariant();
  const stage = denoiseStageVariant();
  const ready = selected.method !== 'off' && denoiseVariantReady(selected.method, selected.preset);
  const path = ready ? denoiseVariantPath(selected.method, selected.preset) : '';
  const same = denoiseSameVariant(selected, stage);
  const label = denoiseVariantLabelFor(selected.method, selected.preset);
  const busy = !!state.running;

  const set = (selector, value) => {
    const element = document.querySelector(selector);
    if (element) element.textContent = value;
  };
  set('[data-dm-selected-label]', label);
  set('[data-dm-stage-output]', denoiseVariantLabelFor(stage.method, stage.preset));
  // This number is measured on one decoded frame. Say so, or it reads as a
  // property of the whole variant.
  const frameMetric = path === denoiseReviewVariant && denoiseReviewMetrics
    ? `Frame ${denoiseReviewFrame + 1}: mean |ΔY| ${denoiseReviewMetrics.mean.toFixed(2)} / 255 · ${denoiseReviewMetrics.changedPercent.toFixed(2)}% pixels changed`
    : '';

  if (selected.method === 'off') {
    set('[data-dm-selected-state]', 'Remove Noise is off.');
    set('[data-dm-selected-meta]', 'The source passes through unchanged.');
    set('[data-dm-selected-output]', 'No denoised output.');
  } else if (!ready) {
    set('[data-dm-selected-state]', 'Not rendered yet.');
    set('[data-dm-selected-meta]', 'Render this variant to compare exact frames.');
    set('[data-dm-selected-output]', 'No output yet.');
  } else if (same) {
    set('[data-dm-selected-state]', 'Current stage output.');
    set('[data-dm-selected-meta]', frameMetric || 'Exact-frame metrics load in the review below.');
    set('[data-dm-selected-output]', path || 'No output path.');
  } else {
    set('[data-dm-selected-state]', 'Ready to use.');
    set('[data-dm-selected-meta]', frameMetric || 'Exact-frame metrics load in the review below.');
    set('[data-dm-selected-output]', path || 'No output path.');
  }

  const action = document.getElementById('denoiseUseVariant');
  if (action) {
    if (selected.method === 'off') {
      action.textContent = same ? 'Remove Noise is off' : 'Turn off Remove Noise';
      action.disabled = busy || same;
    } else if (!ready) {
      action.textContent = 'Render this variant';
      action.disabled = busy;
    } else if (same) {
      action.textContent = 'Current stage output';
      action.disabled = true;
    } else {
      action.textContent = 'Use as stage output';
      action.disabled = busy;
    }
  }
  const copy = document.getElementById('denoiseCopyVariant');
  if (copy) copy.disabled = busy || selected.method === 'off' || !ready;

  document.querySelectorAll('[data-denoise-cell]').forEach(cell => {
    const isSelected = cell.dataset.method === selected.method && cell.dataset.preset === selected.preset;
    cell.classList.toggle('is-selected', isSelected);
    cell.setAttribute('aria-checked', String(isSelected));
    cell.tabIndex = isSelected || (selected.method === 'off' &&
      cell.dataset.method === DENOISE_METHODS[0].key && cell.dataset.preset === DENOISE_PRESETS[0].key) ? 0 : -1;
  });
  // One selection drives the exact review. It either shows that variant or an
  // explicit no-result state; it never keeps older pixels under a new label.
  if (typeof denoiseReviewSyncSelection === 'function') denoiseReviewSyncSelection(selected);
}

function denoiseMatrixSelect(method, preset) {
  if (state.running) return;
  if (!DENOISE_METHODS.some(x => x.key === method) || !DENOISE_PRESETS.some(x => x.key === preset)) return;
  denoiseMatrixSelected = { method, preset };
  const path = denoiseVariantPath(method, preset);
  if (path && denoisePreviewIsFinal(path) && denoiseComparePaths[1] !== path) {
    denoiseComparePaths[1] = path;
    const st = (state.stages || []).find(x => x.key === 'denoise');
    renderDenoiseComparePlayers(denoiseCompareCandidates(st));
  }
  // denoiseMatrixUpdateUi syncs the exact review: a ready variant loads its
  // pixels, Off or an unrendered variant replaces the review with a hold state.
  denoiseMatrixUpdateUi();
}

function denoiseMatrixSyncFromPath(path) {
  const parsed = denoiseVariantFromPath(path);
  if (!parsed) return;
  denoiseMatrixSelected = parsed;
  denoiseMatrixUpdateUi();
}

async function denoiseMatrixUseSelected() {
  if (state.running || denoiseUseVariantPending) return;
  const selected = denoiseSelectedVariant();
  const method = document.querySelector('[data-field="method"]');
  const preset = document.querySelector('[data-field="preset"]');
  const force = document.querySelector('[data-field="force"]');
  if (method) method.value = selected.method;
  if (preset) preset.value = selected.preset || 'gentle';
  if (force) force.checked = false;
  denoiseUseVariantPending = true;
  const action = document.getElementById('denoiseUseVariant');
  if (action) action.disabled = true;
  try {
    await saveStage('denoise', true);
    if (selected.method !== 'off' && !denoiseVariantReady(selected.method, selected.preset)) {
      await runStage('denoise');
    }
  } catch (error) {
    const notice = document.querySelector('[data-dm-notice]');
    if (notice) notice.textContent = 'The stage output could not be changed: ' + (error?.message || 'unknown error');
  } finally {
    denoiseUseVariantPending = false;
    if (typeof denoiseMatrixUpdateUi === 'function') denoiseMatrixUpdateUi();
  }
}

async function denoiseMatrixUseOff() {
  if (state.running) return;
  denoiseMatrixSelected = { method: 'off', preset: '' };
  await denoiseMatrixUseSelected();
}

async function denoiseMatrixCopySelected() {
  const selected = denoiseSelectedVariant();
  if (selected.method === 'off') return;
  const path = denoiseVariantPath(selected.method, selected.preset);
  const text = `method=${selected.method} preset=${selected.preset}${path ? ` output=${path}` : ''}`;
  const notice = document.querySelector('[data-dm-notice]');
  try {
    await navigator.clipboard.writeText(text);
    if (notice) notice.textContent = 'Settings copied.';
  } catch (_) {
    if (notice) notice.textContent = 'Copy failed; select the output path manually.';
  }
}

function updateDenoiseMatrixStatus() {
  if (active !== 'denoise') return;
  const matrix = state?.denoise_matrix || {};
  const count = Number(matrix.ready || 0);
  const status = document.getElementById('denoiseMatrixStatus');
  if (status) {
    const progress = (state?.phase_progress?.stages || []).find(s => s.key === 'denoise');
    const running = state.running && (state.running_stage === 'Remove Noise Batch' || state.running_stage === 'Remove Noise Grid');
    status.textContent = `${count}/9 rendered${running && progress ? ' · ' + progress.label : ''}`;
  }
  const summary = document.querySelector('.denoise-batch-export > summary');
  if (summary) summary.textContent = `Batch export · ${count}/9 rendered`;
  const download = document.getElementById('denoiseMatrixDownload');
  if (download) {
    download.hidden = !matrix.download;
    if (matrix.download) download.textContent = 'Download ZIP · ' + (matrix.filename || 'frame grids');
  }
  const run = document.getElementById('denoiseMatrixRun');
  if (run) {
    run.disabled = !!state.running || denoiseBatchPending;
    if (!denoiseBatchPending) run.textContent = 'Render all 9 variants';
  }
  const exportButton = document.getElementById('denoiseMatrixExport');
  if (exportButton) exportButton.disabled = !!state.running || denoiseGridPending || !count;
  // The batch completed while polling preserved the video DOM. Rebuild once to
  // discover all new variants, then preserve interactive videos again.
  if (!state.running && denoiseMatrixRenderedReady !== null && count !== denoiseMatrixRenderedReady) {
    denoiseMatrixRenderedReady = count;
    void refresh(true);
  }
}

async function denoiseMatrixRunAll() {
  if (state.running || denoiseBatchPending) return;
  denoiseBatchPending = true;
  const button = document.getElementById('denoiseMatrixRun');
  if (button) {
    button.disabled = true;
    button.textContent = 'Starting batch...';
  }
  try {
    const result = await postJson('/api/denoise-batch', {});
    if (!result || !result.ok) {
      alert((result && result.error) || 'The batch could not be started.');
      return;
    }
    await refresh(true);
  } catch (error) {
    alert('The batch request failed: ' + (error?.message || 'the local server did not answer'));
  } finally {
    denoiseBatchPending = false;
    updateDenoiseMatrixStatus();
  }
}

async function denoiseMatrixExport() {
  if (state.running || denoiseGridPending) return;
  const frames = document.getElementById('denoiseMatrixFrames')?.value.trim() || '';
  if (frames) {
    const values = frames.split(',').map(value => Number(value.trim()));
    if (values.length > 5 || values.some(value => !Number.isInteger(value) || value < 1)) {
      return alert('Enter up to five whole frame numbers starting at 1, separated by commas.');
    }
    if (new Set(values).size !== values.length) {
      return alert('Each frame can only appear once. Remove the duplicate before creating the grid.');
    }
  }
  denoiseGridPending = true;
  const button = document.getElementById('denoiseMatrixExport');
  if (button) {
    button.disabled = true;
    button.textContent = 'Creating grids...';
  }
  try {
    const result = await postJson('/api/denoise-grid', {frames});
    if (!result || !result.ok) {
      alert((result && result.error) || 'The grid could not be created.');
      return;
    }
    // A cached grid is immediately downloadable, a new grid appears once complete.
    if (result.state && typeof result.state === 'object') state = result.state;
  } catch (error) {
    alert('The grid request failed: ' + (error?.message || 'the local server did not answer'));
  } finally {
    denoiseGridPending = false;
    const current = document.getElementById('denoiseMatrixExport');
    if (current) current.textContent = 'Create frame grids as ZIP';
    updateDenoiseMatrixStatus();
  }
}

function renderDenoiseComparePlayers(candidates) {
  const host = document.getElementById('denoiseComparePlayers');
  if (!host) return;
  const compareDuration = denoiseCompareEffectiveLength || denoiseCompareLength;
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
              src="${mediaClip(path, denoiseCompareStart, denoiseCompareStart + compareDuration, 'denoise-compare-v1')}" aria-label="Denoising comparison variant ${'ABC'[i]}"></video>` : '<p class="shot-empty">Select a video to compare.</p>'}
          </div>
        </div>
      `).join('')}
    </div>
    <div class="denoise-compare-actions">
      <button type="button" id="denoiseComparePlay" onclick="denoiseComparePlayPause()">Play all</button>
      <button type="button" onclick="denoiseCompareStep(-1)">Previous frame</button>
      <button type="button" onclick="denoiseCompareStep(1)">Next frame</button>
      <input id="denoiseCompareSeek" type="range" min="0" max="${compareDuration}" step="0.001" value="0" oninput="denoiseCompareSeek(this.value)" aria-label="Shared comparison position">
      <span id="denoiseCompareTime">0.00 / ${compareDuration.toFixed(2)} s</span>
    </div>
    <div class="denoise-compare-actions">
      <label><input id="denoiseCompareDiff" type="checkbox" onchange="denoiseCompareDifference()"> Show <span data-dc-pair>${denoiseComparePairLabel()}</span> pixel differences (preview only)</label>
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
  denoiseCompareSyncDiffAvailability();
  denoiseCompareWireVideos();
  denoiseCompareSetEffectiveLength(compareDuration);
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
    denoiseMatrixSyncFromPath(path);
  }
}

function denoiseComparePairIndex() {
  return denoiseCompareLayout === 'ac' ? 2 : 1;
}

function denoiseComparePairLabel() {
  return denoiseCompareLayout === 'ac' ? 'A–C' : 'A–B';
}

function denoiseCompareSyncDiffAvailability() {
  // The previewed difference must describe the pair the user can actually see.
  const toggle = document.getElementById('denoiseCompareDiff');
  const canvas = document.getElementById('denoiseCompareDiffCanvas');
  const pair = denoiseComparePairIndex();
  const second = denoiseComparePaths[pair] || '';
  const duplicate = !second || second === denoiseComparePaths[0];
  if (toggle) {
    toggle.disabled = duplicate;
    if (duplicate) toggle.checked = false;
  }
  const label = document.querySelector('[data-dc-pair]');
  if (label) label.textContent = denoiseComparePairLabel();
  if (canvas) {
    if (duplicate) canvas.classList.add('hidden');
    canvas.setAttribute('aria-label', `Pixel difference between A and ${'ABC'[pair]}`);
  }
  if (duplicate && second) {
    denoiseCompareNotice(`A and ${'ABC'[pair]} are the same file. Choose a different variant to compare pixels.`);
  }
}

function denoiseComparePairReady(first, second) {
  return Boolean(first && second && first.readyState >= 1 && second.readyState >= 1 &&
    first.videoWidth > 0 && first.videoHeight > 0 &&
    first.videoWidth === second.videoWidth && first.videoHeight === second.videoHeight);
}

function denoiseCompareSetLayout(value) {
  denoiseCompareLayout = ['abc','ab','ac'].includes(value) ? value : 'abc';
  const grid = document.querySelector('.denoise-compare-grid');
  if (grid) {
    grid.classList.remove('denoise-layout-abc','denoise-layout-ab','denoise-layout-ac');
    grid.classList.add('denoise-layout-' + denoiseCompareLayout);
  }
  denoiseCompareSyncDiffAvailability();
  const toggle = document.getElementById('denoiseCompareDiff');
  if (toggle && toggle.checked) denoiseCompareDifference();
}

function denoiseCompareApplyClip() {
  const start = document.getElementById('denoiseCompareStart');
  const duration = document.getElementById('denoiseCompareLength');
  const clipped = denoiseCompareClip(start?.value, duration?.value);
  if (!clipped) {
    denoiseCompareNotice('Use a start time between 0 and 360000 seconds and a clip length from 1 to 12 seconds.');
    return;
  }
  const full = denoiseCompareFullDuration();
  const length = full ? Math.min(clipped.duration, full) : clipped.duration;
  const maxStart = full ? Math.max(0, full - length) : 360000;
  denoiseCompareStart = Math.min(clipped.start, maxStart);
  denoiseCompareLength = length;
  denoiseCompareEffectiveLength = length;
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
    const total = denoiseCompareEffectiveLength || denoiseCompareLength;
    const t = Math.min(master.currentTime || 0, total);
    for (const other of videos.slice(1)) {
      if (other.readyState && Math.abs(other.currentTime - t) > 0.09) {
        try { other.currentTime = t; } catch (_) { /* not yet seekable */ }
      }
    }
    const seek = document.getElementById('denoiseCompareSeek');
    if (seek) seek.value = String(t);
    const label = document.getElementById('denoiseCompareTime');
    if (label) label.textContent = `${t.toFixed(2)} / ${total.toFixed(2)} s`;
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
  videos.forEach((video, index) => {
    video.addEventListener('loadedmetadata', () => {
      if (videos.every(v => v.readyState >= 1)) {
        const durations = videos.map(v => Number(v.duration)).filter(d => Number.isFinite(d) && d > 0);
        if (durations.length) denoiseCompareSetEffectiveLength(Math.min(...durations));
        denoiseCompareNotice(
          denoiseCompareCompatible(videos)
            ? 'Same dimensions and duration. Browser synchronization is approximate; the master files are unchanged.'
            : 'Source variants differ in dimensions or duration. Synchronized playback is disabled; select matching clips.'
        );
      }
      if (document.getElementById('denoiseCompareDiff')?.checked) denoiseCompareDifference();
    });
    video.addEventListener('error', () => {
      const slot = 'ABC'[index] || '?';
      const name = denoiseVariantLabel(denoiseComparePaths[index]);
      denoiseCompareNotice(`Clip ${slot} (${name}) could not be decoded. Pick another variant, or check FFmpeg and the ARP log; the other previews stay usable.`);
    });
  });
}

function denoiseCompareSeek(value) {
  const videos = denoiseCompareVideos();
  const total = denoiseCompareEffectiveLength || denoiseCompareLength;
  const t = Number(value);
  if (!Number.isFinite(t) || t < 0 || t > total) return;
  videos.forEach(v => {
    v.pause();
    if (v.readyState) {
      try { v.currentTime = t; } catch (_) { /* not yet seekable */ }
    }
  });
  const label = document.getElementById('denoiseCompareTime');
  if (label) label.textContent = `${t.toFixed(2)} / ${total.toFixed(2)} s`;
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
  const total = denoiseCompareEffectiveLength || denoiseCompareLength;
  denoiseCompareSeek(Math.max(0, Math.min(total, videos[0].currentTime + delta)));
  denoiseCompareNotice('Frame step is an approximate browser seek; exact source-frame evaluation uses FFmpeg.');
}

function denoiseCompareDifference() {
  const enabled = document.getElementById('denoiseCompareDiff');
  const canvas = document.getElementById('denoiseCompareDiffCanvas');
  if (!enabled || !canvas) return;
  const pair = denoiseComparePairIndex();
  const first = document.getElementById('denoiseCompareVideo0');
  const second = document.getElementById(`denoiseCompareVideo${pair}`);
  const sameFile = !second || denoiseComparePaths[0] === denoiseComparePaths[pair];
  if (sameFile) {
    enabled.checked = false;
    enabled.disabled = true;
  }
  canvas.classList.toggle('hidden', !enabled.checked);
  if (!enabled.checked) return;
  if (!denoiseComparePairReady(first, second)) {
    denoiseCompareNotice(`A and ${'ABC'[pair]} must both load with matching dimensions before a pixel difference.`);
    return;
  }
  const width = Math.min(512, first.videoWidth), height = Math.round(width * first.videoHeight / first.videoWidth);
  if (!width || !height) return;
  // The two clips are same-origin H.264 previews; pixel differences include preview compression.
  const tmp = document.createElement('canvas');
  tmp.width = width; tmp.height = height;
  const ctx = tmp.getContext('2d', { willReadFrequently: true });
  if (!ctx) return;
  try {
    ctx.drawImage(first, 0, 0, width, height);
    const a = ctx.getImageData(0, 0, width, height);
    ctx.drawImage(second, 0, 0, width, height);
    const b = ctx.getImageData(0, 0, width, height);
    for (let i = 0; i < a.data.length; i += 4) {
      const d = Math.min(255, 4 * (
        Math.abs(a.data[i] - b.data[i]) +
        Math.abs(a.data[i+1] - b.data[i+1]) +
        Math.abs(a.data[i+2] - b.data[i+2])) / 3);
      a.data[i] = d; a.data[i+1] = 0; a.data[i+2] = Math.round(d * 0.6); a.data[i+3] = 255;
    }
    canvas.width = width; canvas.height = height;
    const output = canvas.getContext('2d');
    if (!output) throw new Error('No 2D canvas context.');
    output.putImageData(a, 0, 0);
  } catch (_) {
    denoiseCompareNotice('Difference preview is unavailable for these video sources.');
  }
}
