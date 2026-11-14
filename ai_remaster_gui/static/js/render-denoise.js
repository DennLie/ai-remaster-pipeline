/* Synchronized denoising previews; Detail keeps the legacy A/B/C selector.
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
let denoiseComparePan = { x: 0, y: 0 };
let denoiseCompareZoomView = null;
let denoiseCompareNative = false;
let denoiseCompareGrid = false;
const denoiseCompareDiffState = {
  1: {enabled: false, url: '', key: '', request: 0, loading: false},
  2: {enabled: false, url: '', key: '', request: 0, loading: false},
};
let denoiseCompareDiffMode = 'direction';
let denoiseCompareDiffContrast = 32;

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
let denoiseBatchPending = false;
let denoiseGridPending = false;

const DEFLICKER_METHODS = [
  {key:'arp',label:'ARP Conservative'},
  {key:'ffmpeg_median',label:'FFmpeg Median'},
  {key:'ffmpeg_am',label:'FFmpeg Arithmetic Mean'},
  {key:'localized_research',label:'Localized · short experiment'},
  {key:'localized_stream',label:'Localized · film streaming'},
];
const ENHANCEMENT_METHODS = [
  {key:'unsharp',label:'Unsharp'}, {key:'cas',label:'CAS'}, {key:'multiscale',label:'Multiscale'},
  {key:'richardson_lucy',label:'Richardson–Lucy'}, {key:'nafnet',label:'NAFNet · GoPro'}, {key:'restormer',label:'Restormer · GoPro'},
];
const STABILIZATION_METHODS = [{key:'zoom0',label:'0% additional zoom'}, {key:'zoom3',label:'3% additional zoom'}, {key:'current',label:'Current pipeline settings'}];
function comparisonStageKey() { return typeof active==='string' && ['dust','deflicker','enhancement','stabilize'].includes(active) ? active : 'denoise'; }
function isRestorationComparison() { return typeof active==='string' && ['dust','denoise','deflicker','enhancement','stabilize'].includes(active); }
function denoiseMethods(key=comparisonStageKey()) { return key==='dust' ? (denoiseMatrixState(key).variants || []).map(v=>({key:v.method,label:dustLabel(v.values)})) : key==='stabilize' ? STABILIZATION_METHODS : key==='enhancement' ? ENHANCEMENT_METHODS : key==='deflicker' ? DEFLICKER_METHODS : DENOISE_METHODS; }
function denoisePresets(key=comparisonStageKey()) { return key!=='denoise' ? [{key:'settings',label:'Current settings'}] : DENOISE_PRESETS; }
function denoiseMatrixState(key=comparisonStageKey()) { return key==='dust' && typeof dustMatrix==='function' ? dustMatrix() : state?.[key+'_matrix'] || {}; }

function denoiseVariantLabelFor(method, preset, key=comparisonStageKey()) {
  if (method === 'off') return 'Off';
  const m = denoiseMethods(key).find(x => x.key === method);
  if(key!=='denoise') return m?.label || 'Variant';
  const p = denoisePresets(key).find(x => x.key === preset);
  return m && p ? `${m.label} · ${p.label}` : 'Variant';
}

function denoiseVariantFromPath(path) {
  if(comparisonStageKey()!=='denoise') {
    const row=denoiseMatrixState().variants?.find(item=>item.output===path);
    return row ? {method:row.method,preset:'settings'} : null;
  }
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

function denoiseVariantPath(method, preset, key=comparisonStageKey(), matrix=denoiseMatrixState(key)) {
  const variants = matrix.variants;
  if (Array.isArray(variants)) {
    const variant = variants.find(v => v.method === method && v.preset === preset);
    if (!variant) return '';
    if (variant.output) return denoisePreviewIsFinal(variant.output) ? variant.output : '';
  }
  const st = (state?.stages || []).find(x => x.key === key);
  const files = st?.files || [];
  const pattern = new RegExp(`_denoised_${method}_${preset}_[0-9a-f]+\\.mkv$`, 'i');
  return (files.find(file => pattern.test(file.path || '')) || {}).path || '';
}

function denoiseVariantReady(method, preset, key=comparisonStageKey(), matrix=denoiseMatrixState(key)) {
  if (method === 'off') return true;
  const variants = matrix.variants;
  if (Array.isArray(variants)) return variants.some(v => v.method === method && v.preset === preset && (key!=='dust' || v.repair_complete));
  return Boolean(denoiseVariantPath(method, preset,key,matrix));
}

function denoiseStageVariant(key=comparisonStageKey()) {
  const values = state?.settings?.[key] || {};
  if(key==='dust') return {method:state.dust_matrix?.variants?.find(v=>v.adopted)?.method || 'off',preset:'settings'};
  if(key==='stabilize') return {method:Number(values.zoom ?? 3)===0 ? 'zoom0' : Number(values.zoom ?? 3)===3 ? 'zoom3' : 'current',preset:'settings'};
  const method = values.method || (key==='deflicker' ? 'arp' : 'off');
  const preset = key!=='denoise' ? 'settings' : values.preset || 'gentle';
  if (method === 'off' || !denoiseMethods(key).some(x => x.key === method) ||
      !denoisePresets(key).some(x => x.key === preset)) return { method: 'off', preset: '' };
  return { method, preset };
}

function denoiseSelectedVariant() {
  if(isRestorationComparison()) return denoiseWorkspaceSnapshot().selected;
  if (denoiseMatrixSelected &&
      (denoiseMatrixSelected.method === 'off' ||
        (denoiseMethods().some(x => x.key === denoiseMatrixSelected.method) &&
         denoisePresets().some(x => x.key === denoiseMatrixSelected.preset)))) {
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
  for (const method of denoiseMethods()) {
    for (const preset of denoisePresets()) {
      if (denoiseVariantReady(method.key, preset.key)) {
        denoiseMatrixSelected = { method: method.key, preset: preset.key };
        return denoiseMatrixSelected;
      }
    }
  }
  denoiseMatrixSelected = comparisonStageKey()==='deflicker' ? current : { method: 'bm3d', preset: 'balanced' };
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
  const source = st.key === 'detail'
    ? ((state && state.detail_comparison && state.detail_comparison.source) || '')
    : (denoiseMatrixState().source || '');
  const candidates = [];
  if (source && denoisePreviewIsFinal(source) && /\.(mp4|mkv|webm|mov|m4v|avi)$/i.test(source)) {
    candidates.push({ path: source, label: 'Stage input — ' + source, kind: 'source' });
  }
  for (const file of (st.files || [])) {
    const path = file.path || '';
    if (['dust','denoise','deflicker','enhancement','stabilize'].includes(st.key) && Array.isArray(denoiseMatrixState().variants) &&
        !denoiseMatrixState().variants.some(v => denoiseVariantPath(v.method, v.preset) === path)) continue;
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
  if (isRestorationComparison()) return available;
  if (!(available >= 1)) return 0;
  return Math.min(12, available);
}

function denoiseCompareEnsureLength() {
  const full = denoiseCompareFullDuration();
  if (isRestorationComparison()) {
    return;
  }
  const key = denoiseCompareSectionKey();
  if (key !== denoiseCompareLengthSection) {
    denoiseCompareLengthSection = key;
    denoiseComparePan = { x: 0, y: 0 };
    denoiseCompareZoom = 1;
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
  if (isRestorationComparison()) {
    denoiseWorkspaceInitialize(candidates);
    return;
  }
  const names = new Set(candidates.map(v => v.path));
  const source = candidates[0] ? candidates[0].path : '';
  const variants = candidates.slice(1).map(v => v.path);
  // Preserve manual selections without repeating them when files disappear.
  // The original is preferred in slot A, the matrix choice in B, and the rest in C.
  const preferred = typeof denoiseSelectedPath === 'function' ? denoiseSelectedPath() : '';
  const used = new Set();
  denoiseComparePaths = denoiseComparePaths.slice(0,3).map((path, index) => {
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
  const shared=['dust','denoise','deflicker','enhancement','stabilize'].includes(st.key);
  if(shared) denoiseWorkspaceActivate(st.key);
  const candidates = denoiseCompareCandidates(st);
  denoiseCompareInitializePaths(candidates);
  denoiseCompareEnsureLength();
  document.getElementById('app').innerHTML = `
    <div class="editor-page ${shared ? 'denoise-editor-page' : ''}">
      <aside class="card denoise-settings-sidebar" aria-label="${esc(st.title)} settings">
        <h2>${esc(st.title)}</h2>
        <p>${esc(shared ? `Compare ${st.key==='dust' ? 'mask-gated dust repair' : st.key==='stabilize' ? 'stabilization zoom' : st.key==='enhancement' ? 'enhancement and deblurring' : st.key==='deflicker' ? 'brightness-correction' : 'noise-reduction'} variants against the processing input, then choose one for the pipeline.` : st.description)}</p>
        ${progressHtml(progress.percent, progress.label)}
        ${shared ? denoiseMatrixHtml() : st.fields.map(field => fieldHtml(st, field)).join('')}
        <div data-dm-expected>${shotOutputList(expected, null)}</div>
        ${stageCheckboxes(settings)}
        <div class="actions">
          <button class="primary" onclick="runStage('${st.key}')" ${state.running ? 'disabled' : ''}>Run ${esc(st.title)}</button>
          <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
        </div>
        <details class="command-details"><summary>Command</summary><div class="command" id="cmd"></div></details>
      </aside>
      <section class="card denoise-compare-card">
        <div class="denoise-compare-heading">
          <h2>Comparison</h2>
          <span class="denoise-compare-info" ${shared ? 'data-ds-quality' : ''} title="${shared ? 'Frame-locked playback is compressed. Exact still decodes the same frame without lossy preview compression, in these viewports.' : 'Browser synchronization is approximate.'}">${shared ? 'Frame-locked · compressed display' : 'Preview · approximate sync'}</span>
          ${shared ? '<button type="button" data-ds-exact aria-pressed="false" onclick="denoiseWorkspaceSetExact()" title="Pause and decode this frame without lossy preview compression">Exact still</button><button type="button" data-ds-retry hidden onclick="denoiseWorkspaceRetry()">Retry exact still</button>' : ''}
          ${reviewZoomControlsHtml('denoiseCompareZoomControls', 'denoiseCompareSetZoom', 'denoiseCompareFit',shared)}
          <div class="denoise-compare-transport" role="group" aria-label="Comparison playback">
            <button type="button" onclick="denoiseCompareStep(-10)" title="Back 10 frames" aria-label="Back 10 frames">${denoiseCompareTransportIcon('previous10')}</button>
            <button type="button" onclick="denoiseCompareStep(-1)" title="Previous frame" aria-label="Previous frame">${denoiseCompareTransportIcon('previous')}</button>
            <button type="button" id="denoiseComparePlay" onclick="denoiseComparePlayPause()" title="Play all" aria-label="Play all">${denoiseCompareTransportIcon('play')}</button>
            <button type="button" onclick="denoiseCompareStep(1)" title="Next frame" aria-label="Next frame">${denoiseCompareTransportIcon('next')}</button>
            <button type="button" onclick="denoiseCompareStep(10)" title="Forward 10 frames" aria-label="Forward 10 frames">${denoiseCompareTransportIcon('next10')}</button>
          </div>
        </div>
        ${shared ? '<div data-processing-input>'+processingInputHtml(st.key)+'</div>' : ''}
        <details class="denoise-preview-settings"><summary>Preview settings</summary>
        ${!shared ? `
        <div class="denoise-compare-toolbar">
          <label>Start time (s) <input id="denoiseCompareStart" type="number" min="0" max="360000" step="0.1" value="${denoiseCompareStart}"></label>
          <label>Clip duration <select id="denoiseCompareLength">
            ${denoiseCompareDurationOptions()}
          </select></label>
          <button type="button" onclick="denoiseCompareApplyClip()">Apply to all previews</button>
          ${st.key !== 'denoise' ? `<label>View <select id="denoiseCompareLayout" onchange="denoiseCompareSetLayout(this.value)">
            <option value="abc">A / B / C · three views</option>
            <option value="ab">A / B · large</option>
            <option value="ac">A / C · large</option>
          </select></label>` : ''}
        </div>
        ` : '<p class="shot-empty">Full stage input. Internal caching never limits the timeline. Stage input and result pixels are compared before display compression.</p>'}
        ${shared ? denoiseCompareDisplayHtml() : ''}
        ${shared ? '<label class="denoise-time-jump">Position (s) <input data-dp-jump type="number" min="0" step="0.001" value="0" onchange="denoiseCompareSeek(this.value)"></label>' : ''}
        </details>
        <p class="review-zoom-hint">Drag image to pan · ${shared ? 'Alt + scroll' : 'Scroll'} to zoom · 1:1 maps source pixels to display pixels · Percent is relative to Fit.</p>
        ${shared ? `<div class="denoise-compact-rail">
          <div class="denoise-compact-rail-heading"><strong>Frame changes</strong>
            <span data-dc-chart-variant>Choose a result to inspect.</span>
            <output data-dr-chart-status role="status">Choose a finished variant.</output>
          </div>
          <div class="denoise-review-chart" data-dr-chart role="group" aria-label="Luminance change per frame" title="Teal: brightened; orange: darkened. Click a bar to move the shared frame; Exact still decodes its pixels on request. Long films are sampled; changes are not a defect mask."></div>
        </div>` : ''}
        <div class="denoise-compare-position">
          <input id="denoiseCompareSeek" type="range" min="0" max="${shared ? denoiseWorkspaceSnapshot().length : denoiseCompareEffectiveLength || denoiseCompareLength}" step="0.001" value="0" oninput="denoiseCompareSeek(this.value)" aria-label="Shared comparison position">
          <span id="denoiseCompareTime">0.00 / ${(shared ? denoiseWorkspaceSnapshot().length : denoiseCompareEffectiveLength || denoiseCompareLength).toFixed(2)} s</span>
        </div>
        <div id="denoiseComparePlayers"></div>
        <p id="denoiseCompareNotice" class="shot-empty" role="status">The playback preview is re-encoded; the original files are never overwritten.</p>
      </section>
    </div>
  `;
  if(shared) mountComparisonLayout(st.title);
  bindStageFields(st.key);
  showCommand(st.key);
  renderDenoiseComparePlayers(candidates);
  if (shared) {
    denoiseWorkspaceFor().syncChart();
    denoiseMatrixUpdateUi();
    updateDenoiseMatrixStatus();
  }
}

function denoiseMatrixHtml() {
  if(comparisonStageKey()==='dust') return dustPickerHtml();
  if(comparisonStageKey()==='stabilize') return stabilizationPickerHtml();
  if(comparisonStageKey()==='enhancement') return enhancementPickerHtml();
  if(comparisonStageKey()==='deflicker') return deflickerPickerHtml();
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
          <h3 id="denoiseVariantHeading">Compare variants</h3>
          <p class="denoise-variant-help">Select up to nine variants. Comparing does not change the pipeline.</p>
        </div>
        <output id="denoiseMatrixStatus" class="denoise-matrix-status" role="status">${count}/9 rendered</output>
      </div>
      <div class="denoise-variant-grid" role="group" aria-label="Variants included in comparison" onkeydown="denoiseMatrixKeyboard(event)">
        <span class="denoise-variant-corner" aria-hidden="true"></span>
        ${DENOISE_METHODS.map(method => `<span class="denoise-variant-col">${method.label}</span>`).join('')}
        ${DENOISE_PRESETS.map(preset => `
          <span class="denoise-variant-row">${preset.label}</span>
          ${DENOISE_METHODS.map(method => denoiseMatrixCellHtml(method, preset, selected)).join('')}
        `).join('')}
      </div>
      <div class="denoise-selection-tools">
        <span data-dm-comparison-count role="status"></span>
        <button type="button" onclick="denoiseWorkspaceSelectAll()">Select all</button>
        <button type="button" onclick="denoiseWorkspaceClear()">Clear</button>
      </div>
      <div class="denoise-generation-prompt" data-dm-generation hidden>
        <p data-dm-generation-text></p>
        <div class="actions">
          <button type="button" class="primary" data-dm-generate onclick="denoiseWorkspaceGenerate()">Render missing variants</button>
          <button type="button" onclick="denoiseWorkspaceDismissGeneration()">Not now</button>
        </div>
        <output data-dm-generation-status role="status"></output>
      </div>
      <button type="button" class="denoise-off-option ${stage.method === 'off' ? 'is-current' : ''}" onclick="denoiseMatrixUseOff()" ${busy ? 'disabled' : ''} aria-pressed="${stage.method === 'off'}">
        Pipeline: use unchanged source
      </button>
      ${comparisonSelectedHtml(selectedLabel,stageLabel,busy)}
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

function comparisonSelectedHtml(selectedLabel,stageLabel,busy) {
  return `<div class="denoise-selected-variant" data-denoise-selected aria-live="polite">
    <div class="denoise-selected-main"><strong class="denoise-selected-name" data-dm-selected-label>${esc(selectedLabel)}</strong><span class="denoise-selected-state" data-dm-selected-state></span></div>
    <div class="denoise-selected-meta" data-dm-selected-meta></div><div class="denoise-selected-output" data-dm-selected-output></div>
    <div class="denoise-selected-actions"><button type="button" class="primary" id="denoiseUseVariant" onclick="denoiseMatrixUseSelected()" ${busy ? 'disabled' : ''}>Use for pipeline</button>
      <button type="button" id="denoiseCopyVariant" onclick="denoiseMatrixCopySelected()" ${busy ? 'disabled' : ''}>Copy settings</button></div>
    <small class="denoise-selected-stage">Selected for pipeline: <strong data-dm-stage-output>${esc(stageLabel)}</strong></small><output class="denoise-selected-notice" data-dm-notice role="status"></output>
  </div>`;
}

function denoiseMatrixCellHtml(method, preset, selected) {
  const ready = denoiseVariantReady(method.key, preset.key);
  const isSelected = denoiseWorkspaceHas(method.key, preset.key);
  const pipeline = denoiseSameVariant({method:method.key,preset:preset.key}, denoiseStageVariant());
  const label = denoiseVariantLabelFor(method.key,preset.key);
  const stateLabel = ready ? 'ready' : 'not rendered';
  return `<button type="button" role="checkbox" aria-checked="${isSelected}" aria-label="${label}, ${stateLabel}${pipeline ? ', selected for pipeline' : ''}"
    class="denoise-variant-cell ${ready ? 'is-ready' : 'is-missing'} ${isSelected ? 'is-selected' : ''}"
    data-denoise-cell data-method="${method.key}" data-preset="${preset.key}"
    onclick="denoiseMatrixSelect('${method.key}','${preset.key}')">
    <svg class="denoise-selection-check" viewBox="0 0 16 16" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><path d="m3 8 3 3 7-7"/></svg>
    ${comparisonStageKey()!=='denoise' ? `<strong>${esc(method.label)}</strong>` : ''}
    <span class="denoise-variant-cell-state">${ready ? 'Ready' : 'Not rendered'}</span>
    <span class="denoise-variant-pipeline" ${pipeline ? '' : 'hidden'}>Pipeline</span>
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
  else if (event.key === 'ArrowDown') next = index + (comparisonStageKey()!=='denoise' ? 1 : 3);
  else if (event.key === 'ArrowUp') next = index - (comparisonStageKey()!=='denoise' ? 1 : 3);
  else if (event.key === 'Home') next = 0;
  else if (event.key === 'End') next = cells.length - 1;
  if (next >= 0 && next < cells.length) {
    event.preventDefault();
    cells[next].focus();
  }
}

function denoiseSameVariant(a, b) {
  return a.method === b.method && (a.method === 'off' || a.preset === b.preset);
}

function denoiseMatrixUpdateUi() {
  const selected = denoiseSelectedVariant();
  invalidateStabilizationFootprint();
  // Focus selects the chart, not pipeline settings or comparison membership.
  denoiseWorkspaceFor().syncChart();
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
  const measured=denoiseWorkspaceSnapshot().metrics;
  const frameMetric = measured
    ? `Frame ${measured.frame + 1}: mean |ΔY| ${measured.mean.toFixed(2)} / 255 · ${measured.changedPercent.toFixed(2)}% pixels changed`
    : '';

  if (selected.method === 'off') {
    set('[data-dm-selected-label]', 'No result selected');
    set('[data-dm-selected-state]', 'Original only.');
    set('[data-dm-selected-meta]', 'Select variants in the matrix to compare.');
    set('[data-dm-selected-output]', '');
  } else if (!ready) {
    set('[data-dm-selected-state]', 'Not rendered yet.');
    set('[data-dm-selected-meta]', 'Confirm rendering below the matrix to compare this variant.');
    set('[data-dm-selected-output]', 'No output yet.');
  } else if (same) {
    set('[data-dm-selected-state]', 'Selected for pipeline.');
    set('[data-dm-selected-meta]', frameMetric || 'Choose Exact still to measure the current frame.');
    set('[data-dm-selected-output]', path || 'No output path.');
  } else {
    set('[data-dm-selected-state]', 'Inspecting.');
    set('[data-dm-selected-meta]', frameMetric || 'Choose Exact still to measure the current frame.');
    set('[data-dm-selected-output]', path || 'No output path.');
  }

  const action = document.getElementById('denoiseUseVariant');
  if (action) {
    if (selected.method === 'off') {
      action.textContent = 'Select a result first';
      action.disabled = true;
    } else if (!ready) {
      action.textContent = 'Render before using for pipeline';
      action.disabled = true;
    } else if (same) {
      action.textContent = 'Selected for pipeline';
      action.disabled = true;
    } else {
      action.textContent = denoiseWorkspaceSnapshot().adoptionPending ? 'Adopting…' : 'Use for pipeline';
      action.disabled = busy || denoiseWorkspaceSnapshot().adoptionPending;
    }
  }
  const copy = document.getElementById('denoiseCopyVariant');
  if (copy) copy.disabled = busy || selected.method === 'off' || !ready;

  document.querySelectorAll('[data-denoise-cell]').forEach(cell => {
    const isSelected = denoiseWorkspaceHas(cell.dataset.method, cell.dataset.preset);
    const ready = denoiseVariantReady(cell.dataset.method, cell.dataset.preset);
    const pipeline = denoiseSameVariant({method:cell.dataset.method,preset:cell.dataset.preset},stage);
    cell.classList.toggle('is-selected', isSelected);
    cell.classList.toggle('is-ready', ready);
    cell.classList.toggle('is-missing', !ready);
    cell.setAttribute('aria-checked', String(isSelected));
    cell.setAttribute('aria-label', `${denoiseVariantLabelFor(cell.dataset.method,cell.dataset.preset)}, ${ready ? 'ready' : 'not rendered'}${pipeline ? ', selected for pipeline' : ''}`);
    const row=cell.closest('.comparison-method-row') || cell;
    const status = row.querySelector('.denoise-variant-cell-state');
    if (status) status.textContent = comparisonStageKey()==='dust' ? ready ? 'Repair ready' : 'Mask ready' : ready ? 'Rendered' : 'Not rendered';
    const badge = row.querySelector('.denoise-variant-pipeline');
    if (badge) badge.hidden = !pipeline;
  });
  denoiseWorkspaceUpdateSelectionUi();
  const analysis=document.querySelector('[data-df-analysis]');
  if(analysis) {
    const gains=denoiseMatrixState().variants?.find(item=>item.method===selected.method)?.analysis;
    analysis.hidden=!gains;
    analysis.textContent=gains ? `Applied ARP gains: ${gains.corrected}/${gains.frames} frames · maximum ${gains.max_gain_change_pct}% · mean ${gains.mean_abs_gain_change_pct}% · ${gains.cut_boundaries} detected cuts. Estimates, not proof of improved restoration.` : '';
  }
  const scenes=document.querySelector('[data-stabilization-scenes]');
  if(scenes) {
    const variant=denoiseMatrixState().variants?.find(item=>item.method===selected.method);
    const reports=variant?.scenes;
    const summary=document.querySelector('[data-stabilization-scene-summary]');
    if(summary) summary.textContent='Stabilization scene report · '+(!variant ? 'Not rendered' : !reports ? 'Report unavailable' :
      `${reports.filter(row=>row.status==='fallback').length}/${reports.length} scenes: unstabilized safety fallback`);
    const html=!variant ? '<p>Render this variant to inspect its scene report.</p>' : !reports ? '<p>Scene report unavailable for this cached result. No tracking success is inferred; explicitly rerender to record it.</p>' :
      `<p>${reports.filter(row=>row.status==='fallback').length}/${reports.length} scenes used the unstabilized safety fallback. Applied stabilization is not proof of improved restoration.</p><ol>${reports.map(row=>`<li>Frames ${row.start+1}–${row.end}: ${row.status==='fallback' ? 'Unstabilized safety fallback' : 'Stabilization applied'}${row.reason ? ' · '+esc(row.reason) : ''}${!row.tracking_report_available ? ' · Tracking report unavailable' : ''}</li>`).join('')}</ol>`;
    if(scenes.innerHTML!==html) scenes.innerHTML=html;
  }
}

function stabilizationFootprintIdentity() {
  const snapshot=denoiseWorkspaceSnapshot(),selected=snapshot.selected;
  const row=denoiseMatrixState().variants?.find(item=>item.method===selected.method);
  return JSON.stringify([comparisonStageKey(),denoiseMatrixState().source_revision,row?.output,row?.revision,snapshot.frame]);
}
function invalidateStabilizationFootprint() {
  const host=document.querySelector('[data-stabilization-footprint]');
  if(!host || host.dataset.identity===stabilizationFootprintIdentity()) return;
  host.dataset.identity=stabilizationFootprintIdentity();
  host.querySelector('[data-footprint-status]').textContent='Select Inspect footprint to read this frame; no render is started.';
  const image=host.querySelector('[data-footprint-image]');image.hidden=true;image.removeAttribute('src');
}
async function inspectStabilizationFootprint() {
  if(comparisonStageKey()!=='stabilize') return;
  const host=document.querySelector('[data-stabilization-footprint]');
  if(!host) return;
  const identity=stabilizationFootprintIdentity(),frame=denoiseWorkspaceSnapshot().frame,path=denoiseSelectedPath();
  const status=host.querySelector('[data-footprint-status]'),image=host.querySelector('[data-footprint-image]');
  host.dataset.identity=identity;image.hidden=true;
  if(!path) {status.textContent='Choose a finished stabilization result first.';return;}
  status.textContent='Reading geometric footprint…';
  try {
    const result=await api('/api/stabilization-footprint?'+new URLSearchParams({variant:path,frame}));
    if(!host.isConnected || identity!==stabilizationFootprintIdentity()) return;
    if(!result.ok) throw Error(result.error || 'Footprint unavailable.');
    status.textContent=`Frame ${frame+1}: valid output ${result.valid_output_pct.toFixed(2)}% · missing output ${result.missing_output_pct.toFixed(2)}% · estimated cropped source area ${result.cropped_source_area_estimate_pct.toFixed(2)}%. ${result.note}`;
    image.src=media(result.image);image.hidden=false;
  }catch(error) {
    if(host.isConnected && identity===stabilizationFootprintIdentity()) status.textContent=error.message || 'Footprint unavailable.';
  }
}

function stabilizationPickerHtml() {
  const stage=state.stages.find(item=>item.key==='stabilize'),values=state.settings.stabilize || {};
  const selected=denoiseSelectedVariant(),pipeline=denoiseStageVariant();
  return `<section class="denoise-variant-picker" aria-labelledby="denoiseVariantHeading">
    <div class="denoise-variant-head"><h3 id="denoiseVariantHeading">Zoom variants</h3><output id="denoiseMatrixStatus" class="denoise-matrix-status" role="status"></output></div>
    <div class="deflicker-variant-list" role="group" aria-label="Stabilization variants included in comparison" onkeydown="denoiseMatrixKeyboard(event)">
      ${STABILIZATION_METHODS.map(method=>denoiseMatrixCellHtml(method,{key:'settings'},selected)).join('')}
    </div>
    <div class="denoise-selection-tools"><span data-dm-comparison-count role="status"></span><button type="button" onclick="denoiseWorkspaceSelectAll()">Select all</button><button type="button" onclick="denoiseWorkspaceClear()">Clear</button></div>
    <div class="denoise-generation-prompt" data-dm-generation hidden><p data-dm-generation-text></p><div class="actions"><button type="button" class="primary" data-dm-generate onclick="denoiseWorkspaceGenerate()">Render missing variants</button><button type="button" onclick="denoiseWorkspaceDismissGeneration()">Not now</button></div><output data-dm-generation-status role="status"></output></div>
    ${comparisonSelectedHtml(denoiseVariantLabelFor(selected.method,selected.preset),denoiseVariantLabelFor(pipeline.method,pipeline.preset),Boolean(state.running))}
    <div class="deflicker-parameters"><h3>Motion settings</h3>
      ${stage.fields.map(field=>fieldHtml(stage,field)).join('')}
      <p>0% and 3% override only additional zoom; motion settings stay identical. Current pipeline settings retain custom zoom values (currently <span data-stabilization-current-zoom>${esc(values.zoom ?? '3')}</span>%). Matching results are reused.</p>
    </div>
  </section>`;
}

function denoiseMatrixSelect(method, preset) {
  if (!denoiseMethods().some(x => x.key === method) || !denoisePresets().some(x => x.key === preset)) return;
  denoiseWorkspaceToggle(method, preset);
}

function denoiseMatrixSyncFromPath(path) {
  const parsed = denoiseVariantFromPath(path);
  if (!parsed) return;
  if(isRestorationComparison()) {
    const slot=denoiseWorkspaceSnapshot().variants.findIndex(v=>denoiseSameVariant(v,parsed))+1;
    denoiseWorkspaceFocus(slot);return;
  }
  denoiseMatrixSelected = parsed;
  denoiseMatrixUpdateUi();
}

async function denoiseMatrixUseSelected(selected = denoiseSelectedVariant()) {
  return denoiseWorkspaceFor().useSelected(selected);
}

async function denoiseMatrixUseOff() {
  if (state.running) return;
  await denoiseMatrixUseSelected({ method: 'off', preset: '' });
}

async function denoiseMatrixCopySelected() {
  const selected = denoiseSelectedVariant();
  if (selected.method === 'off') return;
  const path = denoiseVariantPath(selected.method, selected.preset);
  const settings=state?.settings?.enhancement || {};
  const parameters=comparisonStageKey()==='enhancement'
    ? `${['nafnet','restormer'].includes(selected.method) ? '' : ` preset=${settings.preset || 'gentle'}`} mix=${settings.mix ?? 100}`
    : ` preset=${selected.preset}`;
  const stabilizationValues=comparisonStageKey()==='stabilize' ? {...state.settings.stabilize,
    zoom:selected.method==='zoom0' ? '0' : selected.method==='zoom3' ? '3' : state.settings.stabilize.zoom} : null;
  const text = stabilizationValues ? JSON.stringify({stage:'stabilize',parameters:stabilizationValues,output:path})
    : `method=${selected.method}${parameters}${path ? ` output=${path}` : ''}`;
  const notice = document.querySelector('[data-dm-notice]');
  try {
    await navigator.clipboard.writeText(text);
    if (notice) {notice.dataset.state='success';notice.textContent = 'Settings copied.';}
  } catch (_) {
    if (notice) {notice.dataset.state='error';notice.textContent = 'Copy failed; select the output path manually.';}
  }
}

function updateDenoiseMatrixStatus() {
  if (!isRestorationComparison()) return;
  const matrix = denoiseMatrixState();
  const count = Number(matrix.ready || 0);
  const status = document.getElementById('denoiseMatrixStatus');
  if (status) {
    const progress = (state?.phase_progress?.stages || []).find(s => s.key === comparisonStageKey());
    const running = state.running && matrix.running;
    status.textContent = `${count}/${denoiseWorkspaceSnapshot().variants.length} rendered${comparisonStageKey()!=='enhancement' && running && progress ? ' · ' + progress.label : ''}`;
  }
  const progress = (state?.phase_progress?.stages || []).find(s => s.key === comparisonStageKey());
  const sidebar = document.querySelector('.denoise-settings-sidebar');
  if (sidebar && progress) {
    const percent = Math.max(0,Math.min(100,Number(progress.percent) || 0));
    const label = sidebar.querySelector('[data-progress-label]');
    const number = sidebar.querySelector('[data-progress-percent]');
    const bar = sidebar.querySelector('progress');
    if (label) label.textContent = progress.label || 'Waiting';
    if (number) number.textContent = percent + '%';
    if (bar) bar.value = percent;
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
  denoiseMatrixRenderedReady = count;
  denoiseWorkspaceRefresh();
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
    await refresh();
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

function denoiseCompareDisplayHtml() {
  const {diffMode,diffContrast}=denoiseWorkspaceSnapshot();
  return `<div class="denoise-compare-display">
      <label>Difference colors <select id="denoiseCompareDiffMode" onchange="denoiseCompareSetDiffMode(this.value)">
        <option value="direction" ${diffMode === 'direction' ? 'selected' : ''}>False color: brighter / darker</option>
        <option value="rgb" ${diffMode === 'rgb' ? 'selected' : ''}>RGB: change per channel</option>
      </select></label>
      <label>Amplification <select id="denoiseCompareDiffContrast" onchange="denoiseCompareSetDiffContrast(this.value)">
        ${[8,32,64].map(n => `<option value="${n}" ${n === diffContrast ? 'selected' : ''}>${n}× + gamma</option>`).join('')}
      </select></label>
      <small>Display amplification only. Exact still uses decoded pixels; measurements stay unchanged.</small>
    </div>`;
}

function renderDenoiseComparePlayers(candidates) {
  if (isRestorationComparison()) {
    denoiseWorkspaceRender(candidates);
    return;
  }
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
  `;
  for (let i = 0; i < 3; i++) {
    const select = document.getElementById(`denoiseCompareVariant${i}`);
    if (select) select.value = denoiseComparePaths[i] || '';
  }
  const layout = document.getElementById('denoiseCompareLayout');
  if (layout) layout.value = denoiseCompareLayout;
  denoiseCompareSetZoom(denoiseCompareZoom);
  denoiseCompareWirePan();
  denoiseCompareSyncAllDiffAvailability();
  denoiseCompareWireVideos();
  denoiseCompareSetEffectiveLength(compareDuration);
  denoiseCompareSetPlaying(false);
  denoiseCompareUpdateChartChoices();
}

function denoiseCompareTransportIcon(action) {
  const paths = {
    previous10: '<path d="m12 5-8 7 8 7V5Zm8 0-8 7 8 7V5Z"/>',
    next10: '<path d="m4 5 8 7-8 7V5Zm8 0 8 7-8 7V5Z"/>',
    previous: '<path d="M6 5v14M18 5l-9 7 9 7Z"/>',
    next: '<path d="M18 5v14M6 5l9 7-9 7Z"/>',
    play: '<path d="m8 5 11 7-11 7Z"/>',
    pause: '<path d="M8 5v14M16 5v14"/>',
  };
  return `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" aria-hidden="true">${paths[action]}</svg>`;
}

function denoiseCompareSetPlaying(playing) {
  const button = document.getElementById('denoiseComparePlay');
  if (!button) return;
  const label = playing ? 'Pause all' : 'Play all';
  button.innerHTML = denoiseCompareTransportIcon(playing ? 'pause' : 'play');
  button.setAttribute('aria-label', label);
  button.title = label;
}

function denoiseCompareUpdateChartChoices() {
  const label = document.querySelector('[data-dc-chart-variant]');
  const path=isRestorationComparison() ? denoiseWorkspaceSnapshot().chartPath : '';
  if (label) label.textContent = path ? denoiseVariantLabel(path) : 'No result selected';
}

function denoiseCompareVideos() {
  return denoiseComparePaths.map((path, i) => document.getElementById(`denoiseCompareVideo${i}`)).filter(Boolean);
}

function denoiseCompareResultSlots() {
  if(isRestorationComparison()) return denoiseWorkspaceSnapshot().paths.slice(1).map((path,i)=>i+1);
  return denoiseComparePaths.slice(1).map((path, i) => i + 1);
}

function denoiseCompareChoose(index, path) {
  if (![0, 1, 2].includes(index)) return;
  const st = (state.stages || []).find(v => v.key === active);
  const candidates = st ? denoiseCompareCandidates(st) : [];
  if (!candidates.some(v => v.path === path)) return;
  denoiseComparePaths[index] = path;
  renderDenoiseComparePlayers(candidates);
  if (isRestorationComparison() && index > 0 && denoisePreviewIsFinal(path)) {
    denoiseMatrixSyncFromPath(path);
  }
}

function denoiseCompareDiffLabel(index) {
  if (isRestorationComparison()) return 'Original / ' + denoiseVariantLabel(denoiseWorkspaceSnapshot().paths[index]);
  return Number(index) === 2 ? 'A–C' : 'A–B';
}

function denoiseCompareDiffVisible(index) {
  if (isRestorationComparison()) return denoiseWorkspaceSlotVisible(index);
  return Number(index) === 2 ? denoiseCompareLayout !== 'ab' : denoiseCompareLayout !== 'ac';
}

function denoiseCompareSetDiffMode(value) {
  const mode = ['direction', 'rgb'].includes(value) ? value : 'direction';
  if(isRestorationComparison()) return denoiseWorkspaceFor().setDifference(mode,denoiseWorkspaceSnapshot().diffContrast);
  if (mode === denoiseCompareDiffMode) return;
  denoiseCompareDiffMode = mode;
  denoiseCompareResetDiffCaches();
}

function denoiseCompareSetDiffContrast(value) {
  const contrast = [8, 32, 64].includes(Number(value)) ? Number(value) : 32;
  if(isRestorationComparison()) return denoiseWorkspaceFor().setDifference(denoiseWorkspaceSnapshot().diffMode,contrast);
  if (contrast === denoiseCompareDiffContrast) return;
  denoiseCompareDiffContrast = contrast;
  denoiseCompareResetDiffCaches();
}

function denoiseCompareResetDiffCaches() {
  if (isRestorationComparison()) return denoiseWorkspaceFor().setDifference(denoiseWorkspaceSnapshot().diffMode,denoiseWorkspaceSnapshot().diffContrast);
  for (const index of denoiseCompareResultSlots()) {
    const state = denoiseCompareDiffState[index];
    if (!state) continue;
    state.url = '';
    state.key = '';
    state.request++;
    denoiseCompareHideDiff(index);
    if (state.enabled && denoiseCompareDiffVisible(index)) void denoiseCompareEnsureDiffVideo(index);
  }
}

function denoiseCompareFpsValue(value) {
  const text = String(value ?? '').trim().split(/\s+/)[0];
  const fraction = text.match(/^(\d+(?:\.\d+)?)\/(\d+(?:\.\d+)?)$/);
  const fps = fraction ? Number(fraction[1]) / Number(fraction[2]) : Number(text);
  return Number.isFinite(fps) && fps > 0 ? fps : 0;
}

function denoiseCompareFps() {
  const reviewFps = isRestorationComparison() ? denoiseWorkspaceSnapshot().fps : 0;
  const sourceFps = typeof state === 'object' && state ? state.source_info?.fps : 0;
  return denoiseCompareFpsValue(reviewFps) || denoiseCompareFpsValue(sourceFps);
}

function denoiseCompareFrameAt(seconds, start = denoiseCompareStart, fps = denoiseCompareFps()) {
  const time = Number(seconds);
  const origin = Number(start);
  const rate = Number(fps);
  if (!Number.isFinite(time) || !Number.isFinite(origin) || !(rate > 0)) return null;
  return Math.max(0, Math.round((origin + Math.max(0, time)) * rate));
}

function denoiseCompareSyncDiffAvailability(index) {
  if (typeof active === 'string' && active !== 'denoise') return;
  const state = denoiseCompareDiffState[index];
  if (!state) return;
  const toggle = document.getElementById(`denoiseCompareDiffToggle${index}`);
  const video = document.getElementById(`denoiseCompareDiffVideo${index}`);
  const second = denoiseComparePaths[index] || '';
  const duplicate = !second || second === denoiseComparePaths[0];
  if (toggle) {
    toggle.disabled = duplicate;
    if (duplicate) toggle.checked = false;
    toggle.title = duplicate
      ? (second ? 'Choose a result different from the original for a live diff.' : 'Render this variant before showing its live diff.')
      : '';
  }
  if (duplicate) {
    state.enabled = false;
    state.request++;
    denoiseCompareHideDiff(index);
    return;
  }
  if (video) video.setAttribute('aria-label', `Rendered live change preview: ${denoiseCompareDiffLabel(index)}`);
  if (state.enabled && denoiseCompareDiffVisible(index)) void denoiseCompareEnsureDiffVideo(index);
}

function denoiseCompareSyncAllDiffAvailability() {
  denoiseCompareResultSlots().forEach(denoiseCompareSyncDiffAvailability);
}

function denoiseCompareSetDiffEnabled(index, enabled) {
  if (typeof active === 'string' && active !== 'denoise') return;
  const state = denoiseCompareDiffState[index];
  if (!state) return;
  const toggle = document.getElementById(`denoiseCompareDiffToggle${index}`);
  const path=denoiseComparePaths[index];
  if (!path || path===denoiseComparePaths[0] || !denoisePreviewIsFinal(path)) {
    state.enabled = false;
    if (toggle) toggle.checked = false;
    return;
  }
  state.enabled = Boolean(enabled);
  if (toggle) toggle.checked = state.enabled;
  if (!state.enabled) {
    state.request++;
    denoiseCompareHideDiff(index);
    return;
  }
  if (denoiseCompareDiffVisible(index)) void denoiseCompareEnsureDiffVideo(index);
}

function denoiseCompareHideDiff(index) {
  const state = denoiseCompareDiffState[index];
  const slot = document.querySelector(`[data-dc-diff-slot="${index}"]`);
  const video = document.getElementById(`denoiseCompareDiffVideo${index}`);
  const status = document.getElementById(`denoiseCompareDiffStatus${index}`);
  if (state) state.loading = false;
  if (video) {
    video.pause();
    video.classList.add('hidden');
    video.removeAttribute('src');
    try { video.load(); } catch (_) { /* no source to release */ }
  }
  if (slot) slot.classList.add('hidden');
  if (status) status.textContent = '';
}

function denoiseCompareShowDiffVideo(index) {
  const slot = document.querySelector(`[data-dc-diff-slot="${index}"]`);
  const video = document.getElementById(`denoiseCompareDiffVideo${index}`);
  if (!slot || !video) return;
  slot.classList.remove('hidden');
  video.classList.remove('hidden');
  denoiseCompareZoomView?.apply();
  const master = document.getElementById('denoiseCompareVideo0');
  const syncPlayback = () => {
    const state = denoiseCompareDiffState[index];
    if (!state?.enabled || !denoiseCompareDiffVisible(index)) {
      video.pause();
      return;
    }
    if (!master || video.readyState < 1) return;
    const total = Math.min(
      denoiseCompareEffectiveLength || denoiseCompareLength,
      Number.isFinite(video.duration) ? video.duration : Number.POSITIVE_INFINITY,
    );
    const target = Math.max(0, Math.min(master.currentTime || 0, total));
    try { video.currentTime = target; } catch (_) { /* not seekable yet */ }
    if (!master.paused) video.play().catch(() => { /* autoplay can be refused */ });
    else video.pause();
  };
  if (video.readyState >= 1) syncPlayback();
  else video.addEventListener('loadedmetadata', syncPlayback, {once: true});
}

function denoiseCompareSyncDiffVideo(index, time) {
  const state = denoiseCompareDiffState[index];
  if (!state?.enabled || !denoiseCompareDiffVisible(index)) return;
  const video = document.getElementById(`denoiseCompareDiffVideo${index}`);
  const master = document.getElementById('denoiseCompareVideo0');
  if (!video || video.classList.contains('hidden') || video.readyState < 1) return;
  const total = Math.min(
    denoiseCompareEffectiveLength || denoiseCompareLength,
    Number.isFinite(video.duration) ? video.duration : Number.POSITIVE_INFINITY,
  );
  const target = Math.max(0, Math.min(Number(time) || 0, total));
  if (Math.abs(video.currentTime - target) > 0.09) {
    try { video.currentTime = target; } catch (_) { /* not seekable yet */ }
  }
  if (master && !master.paused) {
    if (video.paused) video.play().catch(() => { /* autoplay can be refused */ });
  } else if (!video.paused) {
    video.pause();
  }
}

function denoiseCompareSyncAllDiffVideos(time) {
  denoiseCompareResultSlots().forEach(index => denoiseCompareSyncDiffVideo(index, time));
}

async function denoiseCompareEnsureDiffVideo(index) {
  if (typeof active === 'string' && active !== 'denoise') return;
  const state = denoiseCompareDiffState[index];
  if (!state?.enabled) return;
  const video = document.getElementById(`denoiseCompareDiffVideo${index}`);
  const slot = document.querySelector(`[data-dc-diff-slot="${index}"]`);
  const status = document.getElementById(`denoiseCompareDiffStatus${index}`);
  if (!video || !slot) return;
  const first = denoiseComparePaths[0];
  const second = denoiseComparePaths[index];
  const start = denoiseCompareStart;
  const end = denoiseCompareStart + (denoiseCompareEffectiveLength || denoiseCompareLength);
  const label = denoiseCompareDiffLabel(index);
  if (!first || !second || first === second) {
    state.enabled = false;
    denoiseCompareHideDiff(index);
    return;
  }
  const key = denoiseCompareDiffKeyFor(first, second, start, end);
  if (state.loading && state.key === key) return;
  if (state.url && state.key === key) {
    slot.classList.remove('hidden');
    if (video.getAttribute('src') !== state.url) {
      video.src = state.url;
      video.load();
    }
    denoiseCompareShowDiffVideo(index);
    if (status) status.textContent = '';
    return;
  }
  const requestId = ++state.request;
  state.key = key;
  state.loading = true;
  video.pause();
  slot.classList.remove('hidden');
  video.classList.add('hidden');
  if (status) status.textContent = `Rendering ${label} live change…`;
  try {
    const result = await postJson('/api/denoise-diff', {
      first, second, start, end,
      mode: denoiseCompareDiffMode,
      contrast: denoiseCompareDiffContrast,
    });
    if (requestId !== state.request || !state.enabled || !video.isConnected) return;
    if (!result || !result.ok) {
      throw new Error((result && result.error) || 'The live change preview could not be rendered.');
    }
    state.url = result.url;
    state.key = key;
    const loaded = new Promise((resolve, reject) => {
      const onReady = () => { cleanup(); resolve(); };
      const onError = () => { cleanup(); reject(new Error('The rendered live change preview could not be loaded.')); };
      const cleanup = () => {
        video.removeEventListener('loadedmetadata', onReady);
        video.removeEventListener('error', onError);
      };
      video.addEventListener('loadedmetadata', onReady, {once: true});
      video.addEventListener('error', onError, {once: true});
    });
    video.src = result.url;
    video.load();
    await loaded;
    if (requestId !== state.request || !state.enabled) return;
    if (status) status.textContent = '';
    denoiseCompareShowDiffVideo(index);
  } catch (error) {
    if (requestId !== state.request) return;
    state.url = '';
    state.key = '';
    if (video) {
      video.pause();
      video.classList.add('hidden');
    }
    if (slot) slot.classList.remove('hidden');
    if (status) status.textContent = 'Live change unavailable: ' + (error?.message || 'unknown error');
  } finally {
    if (requestId === state.request) state.loading = false;
  }
}

function denoiseCompareSetLayout(value) {
  denoiseCompareLayout = ['abc','ab','ac'].includes(value) ? value : 'abc';
  const grid = document.querySelector('.denoise-compare-grid');
  if (grid) {
    grid.classList.remove('denoise-layout-abc','denoise-layout-ab','denoise-layout-ac');
    grid.classList.add('denoise-layout-' + denoiseCompareLayout);
  }
  denoiseCompareSyncAllDiffAvailability();
  const master = document.getElementById('denoiseCompareVideo0');
  const time = Number(master?.currentTime) || 0;
  for (const index of [1, 2]) {
    const state = denoiseCompareDiffState[index];
    if (!state?.enabled) continue;
    if (denoiseCompareDiffVisible(index)) {
      denoiseCompareSyncDiffVideo(index, time);
    } else {
      document.getElementById(`denoiseCompareDiffVideo${index}`)?.pause();
    }
  }
  denoiseCompareApplyTransform();
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
  const available = denoiseCompareAvailableLength();
  const maxStart = available ? Math.max(0, available - length) : 360000;
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

function denoiseCompareApplyTransform() {
  if (!denoiseCompareZoomView) return;
  Object.assign(denoiseCompareZoomView.state, {zoom:denoiseCompareZoom, ...denoiseComparePan});
  denoiseCompareZoomView.apply();
}

function denoiseCompareSetZoom(value) {
  if(isRestorationComparison()) return denoiseWorkspaceFor().setViewport({zoom:value,native:false});
  denoiseCompareZoom = reviewZoomClamp(value, 1/32, 32);
  denoiseCompareZoomView?.setZoom(denoiseCompareZoom);
}

function denoiseCompareFit() {
  if(isRestorationComparison()) return denoiseWorkspaceFor().setViewport({zoom:1,pan:{x:0,y:0},native:false});
  denoiseCompareZoom = 1;
  denoiseComparePan = {x:0, y:0};
  denoiseCompareZoomView?.fit();
}

function denoiseCompareWirePan() {
  if(isRestorationComparison()) return;
  const host = document.getElementById('denoiseComparePlayers');
  if (!host) return;
  denoiseCompareZoomView?.destroy();
  denoiseCompareZoomView = createReviewZoom(host,
    '.denoise-compare-video-wrap, .denoise-compare-diff-frame',
    document.getElementById('denoiseCompareZoomControls'),
    {zoom:denoiseCompareZoom, ...denoiseComparePan, native:denoiseCompareNative,grid:denoiseCompareGrid}, value => {
      denoiseCompareZoom = value.zoom;
      denoiseCompareNative = Boolean(value.native);
      denoiseCompareGrid = Boolean(value.grid);
      denoiseComparePan = {x:value.x, y:value.y};
    });
}

function denoiseCompareCompatible(videos) {
  if (!videos.length || videos.some(v => v.readyState < 1)) return false;
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
  if (!master.dataset.masterWired) {
    master.dataset.masterWired = 'true';
    master.addEventListener('timeupdate', denoiseCompareSyncPlayback);
    master.addEventListener('seeked', denoiseCompareSyncPlayback);
    master.addEventListener('play', denoiseCompareSyncPlayback);
    master.addEventListener('pause', denoiseCompareSyncPlayback);
    master.addEventListener('ended', denoiseCompareSyncPlayback);
  }
  videos.forEach((video, index) => {
    if (video.dataset.playbackWired) return;
    video.dataset.playbackWired = 'true';
    video.addEventListener('loadedmetadata', () => {
      if (!video.isConnected) return;
      const current = denoiseCompareVideos();
      if (current.every(v => v.readyState >= 1)) {
        const durations = current.map(v => Number(v.duration)).filter(d => Number.isFinite(d) && d > 0);
        if (durations.length) denoiseCompareSetEffectiveLength(Math.min(...durations));
        denoiseCompareNotice(
          denoiseCompareCompatible(current)
            ? 'Same dimensions and duration. Browser synchronization is approximate; the master files are unchanged.'
            : 'Some previews differ in dimensions or duration. Only matching previews play together.'
        );
      }
      denoiseCompareSyncPlayback();
    });
    video.addEventListener('error', () => {
      if (!video.isConnected) return;
      denoiseCompareNotice('A preview could not be decoded. Check FFmpeg and the ARP log, then remove and reselect that variant to retry; the other previews stay usable.');
    });
  });
}

function denoiseCompareSyncPlayback() {
  const master = document.getElementById('denoiseCompareVideo0');
  if (!master) return;
  const total = denoiseCompareEffectiveLength || denoiseCompareLength;
  const time = Math.min(master.currentTime || 0,total);
  for (const other of denoiseCompareVideos().filter(video => video !== master)) {
    const slot = Number(other.closest('[data-dc-slot]')?.dataset.dcSlot);
    if (active === 'denoise' && !denoiseWorkspaceSlotVisible(slot)) { other.pause(); continue; }
    if (other.readyState < 1 || other.error) continue;
    if (Math.abs(other.currentTime-time) > .09) {
      try {other.currentTime = time;} catch (_) { /* not seekable yet */ }
    }
    if (!master.paused && denoiseCompareCompatible([master,other])) {
      if (other.paused) other.play().catch(() => denoiseCompareNotice('A preview could not play. Check the ARP log or reselect that variant.'));
    } else other.pause();
  }
  const seek = document.getElementById('denoiseCompareSeek');
  if (seek) seek.value = String(time);
  const label = document.getElementById('denoiseCompareTime');
  if (label) label.textContent = `${time.toFixed(2)} / ${total.toFixed(2)} s`;
  denoiseCompareSetPlaying(!master.paused);
  denoiseCompareSyncAllDiffVideos(time);
}

function denoiseCompareSeek(value) {
  if (isRestorationComparison()) return denoiseWorkspaceFor().seek(value);
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
  denoiseCompareResultSlots().forEach(slot => {
    const state = denoiseCompareDiffState[slot];
    const diffVideo = document.getElementById(`denoiseCompareDiffVideo${slot}`);
    if (diffVideo && state?.enabled && diffVideo.readyState) {
      diffVideo.pause();
      try { diffVideo.currentTime = Math.min(t, diffVideo.duration || total); } catch (_) { /* not yet seekable */ }
    }
  });
  const label = document.getElementById('denoiseCompareTime');
  if (label) label.textContent = `${t.toFixed(2)} / ${total.toFixed(2)} s`;
  const seek = document.getElementById('denoiseCompareSeek');
  if (seek) seek.value = String(t);
  denoiseCompareSetPlaying(false);
}

function denoiseComparePlayPause() {
  if (isRestorationComparison()) return denoiseWorkspaceFor().playPause();
  const videos = denoiseCompareVideos();
  const master = document.getElementById('denoiseCompareVideo0');
  if (!master || master.readyState < 1 || master.error) {
    denoiseCompareNotice('Wait for the original preview to load before playing.');
    return;
  }
  if (!master.paused) {
    videos.forEach(v => v.pause());
    return;
  }
  const t = master.ended ? 0 : master.currentTime;
  const visible = videos.filter(video => video === master ||
    (video.readyState >= 1 && denoiseCompareCompatible([master,video]) &&
      (active !== 'denoise' || denoiseWorkspaceSlotVisible(Number(video.closest('[data-dc-slot]')?.dataset.dcSlot)))));
  for (const v of visible) v.currentTime = t;
  Promise.all(visible.map(v => v.play())).then(() => {
    denoiseCompareSetPlaying(true);
    denoiseCompareSyncAllDiffVideos(t);
  }).catch(() => {
    videos.forEach(v => v.pause());
    denoiseCompareNotice('Browser blocked playback or a codec failed. Use the playback controls to retry.');
  });
}

function denoiseCompareStep(frames) {
  if (isRestorationComparison()) return denoiseWorkspaceFor().step(frames);
  const videos = denoiseCompareVideos();
  if (!videos.length || !videos[0].readyState) return;
  const sourceFps = Number(String((state.source_info || {}).fps || '').split(' ')[0]) || 30;
  const fps = Math.max(1, Math.min(240, sourceFps));
  const delta = frames / fps;
  const total = denoiseCompareEffectiveLength || denoiseCompareLength;
  denoiseCompareSeek(Math.max(0, Math.min(total, videos[0].currentTime + delta)));
  denoiseCompareNotice('Frame step is an approximate browser seek; exact source-frame evaluation uses FFmpeg.');
}

function denoiseCompareDiffKeyFor(first, second, start, end,
  mode = denoiseCompareDiffMode, contrast = denoiseCompareDiffContrast) {
  return [first, second, Number(start).toFixed(3), Number(end).toFixed(3),
    mode, Number(contrast)].join('|');
}
