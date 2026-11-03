/* Extended, frame-exact Remove Noise review. Complements rather than replaces
 * the existing approximate A/B/C H.264 playback previews.
 * No writable operations or user-controlled filesystem APIs.
 */
let denoiseReviewVariant = '';
let denoiseReviewFrame = 0;
let denoiseReviewCount = 0;
let denoiseReviewFps = 0;
let denoiseReviewEpoch = 0;
let denoiseReviewChartEpoch = 0;
let denoiseReviewEntries = [];
let denoiseReviewZoom = 1;
let denoiseReviewWipe = 50;
let denoiseReviewDiffMode = 'direction';
let denoiseReviewDiffContrast = 32;
let denoiseReviewImagePair = null;
let denoiseReviewMetrics = null;
// The review label describes the pixels that are actually on screen. 'hold'
// hides the stills whenever the selection has no result, so stale evidence can
// never sit under a newer label.
let denoiseReviewPhase = 'idle';
let denoiseReviewHold = null;
let denoiseReviewHasFrame = false;
let denoiseReviewInFlight = false;
let denoiseReviewPending = null;
let denoiseWipePointer = null;

// Shared difference helpers predate the English review and throw German
// validation text. Never leak that into the English tab.
const DENOISE_REVIEW_ERROR_TEXT = [
  [/Ungültige Differenzdaten oder Kontrasteinstellungen\.?/i,
    'The pixel-difference data was invalid for these settings.'],
  [/Bildgrößen stimmen nicht überein\.?/i,
    'The still images do not have identical dimensions.'],
  [/Nicht deckungsgleiche Einzelbilder\.?/i,
    'The still images are not aligned.'],
  [/Canvas nicht verfügbar\.?/i,
    'This browser cannot provide a 2D canvas for the difference view.'],
];

function denoiseReviewErrorText(error) {
  const text = String((error && error.message) || error || 'Unknown error.').trim();
  for (const [pattern, replacement] of DENOISE_REVIEW_ERROR_TEXT) {
    if (pattern.test(text)) return replacement;
  }
  return text.length > 300 ? text.slice(0, 300) + '…' : text;
}

function denoiseReviewVariants(candidates) {
  // A missing film source drops the "Original" entry; filtering by kind keeps
  // every rendered variant reviewable instead of silently losing the first.
  // Legacy snapshots and tests have no kind, so their first entry is the source.
  const items = candidates || [];
  const hasKinds = items.some(v => v.kind);
  return items.filter((v, index) => {
    if (hasKinds ? v.kind === 'source' : index === 0) return false;
    return denoisePreviewIsFinal(v.path);
  });
}

function denoiseReviewPathValid(path) {
  const st = state?.stages?.find(x => x.key === 'denoise');
  return Boolean(path) && denoiseReviewVariants(st ? denoiseCompareCandidates(st) : [])
    .some(x => x.path === path);
}

function denoiseReviewHoldMessage(kind, label, detail) {
  if (kind === 'off') {
    return 'Remove Noise is off. The source passes through unchanged, so there is no denoised result to review. The original frames stay available in the A/B/C comparison above.';
  }
  if (kind === 'missing') {
    return `${label || 'This variant'} has not been rendered. Render it from the Variant panel; exact frames, the difference map and the wipe appear here when it finishes.`;
  }
  if (kind === 'unavailable') {
    return `${label || 'This variant'} is out of step with the current file list, so its exact frames cannot be opened. Reload the page; if it persists, render the variant again.`;
  }
  return `${label ? 'The exact frames for ' + label + ' could not be loaded.' : 'The exact frames could not be loaded.'}${detail ? ' ' + denoiseReviewErrorText(detail) : ''} Select the variant again to retry.`;
}

function denoiseReviewHtml(candidates) {
  const options = denoiseReviewVariants(candidates);
  if (!options.length) {
    return `<section class="card denoise-review-card denoise-review" data-denoise-review data-review-lang="en" aria-label="Exact frame review">
      <h3>Exact frame review</h3>
      <p class="shot-empty">No finished Remove Noise variant yet. After rendering, exact original/result frames, pixel difference, frame bars and the wipe comparison appear here.</p>
    </section>`;
  }
  const selected = typeof denoiseSelectedVariant === 'function' ? denoiseSelectedVariant() : null;
  let label = '';
  let holdKind = '';
  let holdText = '';
  let target = '';
  if (selected && selected.method === 'off') {
    label = 'Off';
    holdKind = 'off';
  } else if (selected && (!denoiseVariantReady(selected.method, selected.preset) || !denoiseVariantPath(selected.method, selected.preset))) {
    label = denoiseVariantLabelFor(selected.method, selected.preset);
    holdKind = 'missing';
  } else {
    const preferred = selected ? denoiseVariantPath(selected.method, selected.preset) : '';
    target = options.some(x => x.path === preferred) ? preferred : options[0].path;
    label = denoiseVariantLabel(target);
    if (selected && preferred !== target) holdKind = 'unavailable';
  }
  if (holdKind) {
    holdText = denoiseReviewHoldMessage(holdKind, label);
    denoiseReviewVariant = '';
    denoiseReviewPhase = 'hold';
    denoiseReviewHold = { kind: holdKind, label };
  } else {
    denoiseReviewVariant = target;
    denoiseReviewPhase = 'loading';
    denoiseReviewHold = null;
  }
  denoiseReviewHasFrame = false;
  return `<section class="card denoise-review-card denoise-review" data-denoise-review data-review-lang="en" aria-label="Exact frame review">
    <div class="denoise-review-heading">
      <div><h3>Exact frame review</h3>
        <span>Beyond the A/B/C preview: decoded source frames, not H.264 previews.</span></div>
      <span class="denoise-review-current">Selected: <strong data-dr-current-variant>${esc(label)}</strong></span>
    </div>
    <div class="denoise-review-hold" data-dr-hold ${holdKind ? '' : 'hidden'}>
      <p data-dr-hold-text>${esc(holdText)}</p>
    </div>
    <div class="denoise-review-body" data-dr-body ${holdKind ? 'hidden' : ''}>
    <div class="denoise-review-toolbar">
      <span class="denoise-review-toolbar-note">Frame-exact stills from the actual processing input and the selected variant.</span>
      <label>Zoom <select data-dr-zoom onchange="denoiseReviewSetZoom(this.value)">
        ${[1,2,4].map(v => `<option value="${v}" ${v === denoiseReviewZoom ? 'selected' : ''}>${v*100}%</option>`).join('')}
      </select></label>
    </div>
    <div class="denoise-review-rail">
      <div class="cleanup-frame-rail-heading"><strong>Frame changes</strong>
        <span>Teal = brightened · Orange = darkened · Bars select an exact frame · arrow keys step</span></div>
      <div class="denoise-review-chart" data-dr-chart role="group" aria-label="Luminance change per frame"></div>
      <output data-dr-chart-status role="status">Calculating frame differences...</output>
      <small>Measured pixel changes are not proof of noise or grain removal. Long films are sampled.</small>
    </div>
    <section class="denoise-review-inspector" aria-label="Three exact frame views">
      <div class="fi-heading"><strong>Frame inspector</strong>
        <small>Frame-exact from the actual processing input and the selected variant</small></div>
      <div class="fi-buttons">
        <button type="button" data-dr-prev10 disabled>−10</button>
        <button type="button" data-dr-prev disabled>−1 frame</button>
        <button type="button" data-dr-next disabled>+1 frame</button>
        <button type="button" data-dr-next10 disabled>+10</button>
      </div>
      <input data-dr-timeline type="range" min="0" max="0" value="0" step="1" disabled aria-label="Exact frame position in the clip">
      <output data-dr-status role="status">Preparing exact frames.</output>
      <div class="denoise-review-diff-controls">
        <label>Pixel difference <select data-dr-diff-mode onchange="denoiseReviewSetDiffMode(this.value)">
          <option value="direction" ${denoiseReviewDiffMode === 'direction' ? 'selected' : ''}>False color: brighter / darker</option>
          <option value="rgb" ${denoiseReviewDiffMode === 'rgb' ? 'selected' : ''}>RGB: change per channel</option>
        </select></label>
        <label>Amplification <select data-dr-diff-contrast onchange="denoiseReviewSetDiffContrast(this.value)">
          ${[8,32,64].map(n => `<option value="${n}" ${n === denoiseReviewDiffContrast ? 'selected' : ''}>${n}× + gamma</option>`).join('')}
        </select></label>
        <small>Display amplification only — original frames, results and measurements stay unchanged.</small>
      </div>
      <div class="denoise-review-stills">
        <figure><figcaption>ORIGINAL · render input</figcaption>
          <div class="denoise-review-media"><img data-dr-before tabindex="0" role="button" alt="Original frame before noise reduction" title="Click to enlarge"></div></figure>
        <figure><figcaption>RESULT · selected variant</figcaption>
          <div class="denoise-review-media"><img data-dr-after tabindex="0" role="button" alt="Exact processed frame" title="Click to enlarge"></div></figure>
        <figure><figcaption>PIXEL DIFFERENCE · amplified</figcaption>
          <div class="denoise-review-media"><canvas data-dr-difference tabindex="0" role="button" aria-label="Amplified pixel differences, click to enlarge" title="Click to enlarge"></canvas></div>
          <small data-dr-diff-caption>Teal = brightened · Orange = darkened · Dark gray = unchanged. Not a defect mask.</small></figure>
      </div>
      <output data-dr-readout role="status"></output>
      <details class="denoise-review-wipe" open>
        <summary><strong>Wipe comparison</strong> · exact original/result slider</summary>
        <p class="denoise-review-wipe-hint">Left <strong>ORIGINAL</strong>, right <strong>RESULT</strong>.
          Drag the vertical line in the image or use the slider below.</p>
        <output class="wipe-current-view" data-dr-wipe-current-view aria-live="off">50% ORIGINAL · 50% RESULT</output>
        <div class="denoise-review-wipe-picture" data-dr-wipe-picture>
          <img data-dr-wipe-before alt="Unchanged original frame">
          <img data-dr-wipe-after alt="Processed frame">
          <div class="denoise-review-wipe-divider" data-dr-wipe-divider aria-hidden="true">
            <span class="denoise-review-wipe-handle" data-dr-wipe-handle><span class="wipe-grip-side wipe-grip-original">ORIGINAL</span><span class="wipe-grip-side wipe-grip-result">RESULT</span></span>
          </div>
        </div>
        <label>Original left · Result right
          <input data-dr-wipe-position type="range" min="0" max="100" step="1"
                 value="${denoiseReviewWipe}" oninput="denoiseReviewSetWipe(this.value)">
        </label>
        <output data-dr-wipe-status>50% original left / 50% result right</output>
      </details>
    </section>
    </div>
  </section>`;
}

function denoiseReviewRoot() { return document.querySelector('[data-denoise-review]'); }
function denoiseReviewElement(name) { return denoiseReviewRoot()?.querySelector('[data-dr-' + name + ']'); }

function denoiseReviewSetLabel(label) {
  const current = denoiseReviewElement('current-variant');
  if (current) current.textContent = label;
}

function denoiseReviewClearEvidence() {
  // Never leave a decoded still or a measurement behind while a different
  // variant loads; an empty picture is honest, a stale one is not.
  for (const name of ['before', 'after', 'wipe-before', 'wipe-after']) {
    const image = denoiseReviewElement(name);
    if (image) image.removeAttribute('src');
  }
  const difference = denoiseReviewElement('difference');
  if (difference) {
    difference.width = 0;
    difference.height = 0;
  }
  const readout = denoiseReviewElement('readout');
  if (readout) readout.textContent = '';
  const timeline = denoiseReviewElement('timeline');
  if (timeline) {
    timeline.disabled = true;
    timeline.max = '0';
    timeline.value = '0';
  }
  for (const key of ['prev', 'prev10', 'next', 'next10']) {
    const button = denoiseReviewElement(key);
    if (button) button.disabled = true;
  }
  denoiseReviewEntries = [];
  denoiseReviewHighlight();
}

function denoiseReviewShowBody() {
  const hold = denoiseReviewElement('hold');
  const body = denoiseReviewElement('body');
  if (hold) hold.hidden = true;
  if (body) body.hidden = false;
  denoiseReviewHold = null;
}

function denoiseReviewShowHold(kind, label, detail) {
  const hold = denoiseReviewElement('hold');
  const body = denoiseReviewElement('body');
  if (!hold || !body) return;
  denoiseReviewPhase = 'hold';
  denoiseReviewHold = { kind, label };
  denoiseReviewVariant = '';
  denoiseReviewHasFrame = false;
  denoiseReviewInFlight = false;
  denoiseReviewPending = null;
  denoiseReviewEpoch++;
  denoiseReviewChartEpoch++;
  denoiseReviewImagePair = null;
  denoiseReviewMetrics = null;
  denoiseReviewEntries = [];
  hold.hidden = false;
  body.hidden = true;
  const text = hold.querySelector('[data-dr-hold-text]');
  if (text) text.textContent = denoiseReviewHoldMessage(kind, label, detail);
  denoiseReviewSetLabel(label);
  denoiseReviewRoot()?.removeAttribute('aria-busy');
  const chart = denoiseReviewElement('chart');
  if (chart) chart.replaceChildren();
  const chartStatus = denoiseReviewElement('chart-status');
  if (chartStatus) chartStatus.textContent = 'No result selected; the frame chart is paused.';
}

function denoiseReviewSyncSelection(selected) {
  const root = denoiseReviewRoot();
  if (!root || !root.querySelector('[data-dr-hold]')) return;
  if (!selected || selected.method === 'off') {
    if (denoiseReviewHold?.kind !== 'off') denoiseReviewShowHold('off', 'Off');
    return;
  }
  const label = denoiseVariantLabelFor(selected.method, selected.preset);
  if (!denoiseVariantReady(selected.method, selected.preset)) {
    if (denoiseReviewHold?.kind !== 'missing' || denoiseReviewHold.label !== label)
      denoiseReviewShowHold('missing', label);
    return;
  }
  const path = denoiseVariantPath(selected.method, selected.preset);
  if (!path || !denoisePreviewIsFinal(path) || !denoiseReviewPathValid(path)) {
    if (denoiseReviewHold?.kind !== 'unavailable' || denoiseReviewHold.label !== label)
      denoiseReviewShowHold('unavailable', label);
    return;
  }
  denoiseReviewShowBody();
  denoiseReviewChoose(path);
}

function denoiseReviewSetZoom(value) {
  denoiseReviewZoom = [1,2,4].includes(Number(value)) ? Number(value) : 1;
  denoiseReviewRoot()?.style.setProperty('--denoise-review-zoom', String(denoiseReviewZoom));
  denoiseReviewSetWipe(denoiseReviewWipe);
}

function denoiseReviewSetWipe(value) {
  denoiseReviewWipe = Math.max(0, Math.min(100, Number(value) || 0));
  const after = denoiseReviewElement('wipe-after');
  const line = denoiseReviewElement('wipe-divider');
  // Both pictures use centered scaling; align the clip with the physical handle.
  const edge = 50 + (denoiseReviewWipe - 50) / denoiseReviewZoom;
  if (after) after.style.clipPath = `inset(0 0 0 ${edge}%)`;
  if (line) line.style.left = denoiseReviewWipe + '%';
  const picture = denoiseReviewElement('wipe-picture');
  const grip = denoiseReviewElement('wipe-handle');
  if (picture && grip) {
    const width = picture.getBoundingClientRect().width;
    grip.style.opacity = String(comparisonWipeGripOpacity(width, grip.offsetWidth || 180, denoiseReviewWipe));
  }
  const view = denoiseReviewElement('wipe-current-view');
  if (view) view.textContent = comparisonWipeVisibleState(denoiseReviewWipe, 'RESULT', 'en');
  const control = denoiseReviewElement('wipe-position');
  if (control) control.value = String(denoiseReviewWipe);
  const status = denoiseReviewElement('wipe-status');
  if (status) status.textContent = `${denoiseReviewWipe.toFixed(0)}% original left · ${(100-denoiseReviewWipe).toFixed(0)}% result right`;
}

function denoiseReviewChoose(path, preserveFrame = true) {
  if (!denoiseReviewPathValid(path)) return;
  if (path === denoiseReviewVariant &&
      (denoiseReviewPhase === 'ready' || denoiseReviewPhase === 'loading')) return;
  denoiseReviewVariant = path;
  denoiseReviewPhase = 'loading';
  denoiseReviewHasFrame = false;
  denoiseReviewHold = null;
  denoiseReviewEpoch++;
  denoiseReviewChartEpoch++;
  denoiseReviewCount = 0;
  denoiseReviewImagePair = null;
  denoiseReviewEntries = [];
  denoiseReviewMetrics = null;
  if (!preserveFrame) denoiseReviewFrame = 0;
  denoiseReviewShowBody();
  denoiseReviewSetLabel(denoiseVariantLabel(path));
  denoiseReviewClearEvidence();
  denoiseReviewRoot()?.setAttribute('aria-busy', 'true');
  const status = denoiseReviewElement('status');
  if (status) status.textContent = `Loading exact frames for ${denoiseVariantLabel(path)}...`;
  // Reuse the existing event handlers; re-running Start would attach them twice.
  void denoiseReviewSelectFrame(denoiseReviewFrame, false);
  void denoiseReviewLoadChart(path);
}

function denoiseReviewChartRows(report) {
  if (!report || report.metric !== 'directional_mean_luma_8bit' ||
      !Number.isInteger(report.frames) || report.frames < 1 ||
      !Array.isArray(report.samples) || report.samples.length > 240 || !report.samples.length) {
    throw new Error('Invalid frame report.');
  }
  let prev = -1, peak = 0;
  const rows = report.samples.map(x => {
    const frame = Number(x.frame), bright = Number(x.brighten_mean_luma);
    const dark = Number(x.darken_mean_luma), total = Number(x.mean_abs_luma);
    if (!Number.isInteger(frame) || frame <= prev || frame >= report.frames ||
        ![bright,dark,total].every(v => Number.isFinite(v) && v >= 0 && v <= 255) ||
        Math.abs(total-bright-dark) > 0.001) throw new Error('Invalid frame measurements.');
    prev = frame;
    peak = Math.max(peak, bright, dark);
    return {frame, bright, dark, total};
  });
  return rows.map(x => ({...x, brightHeight: x.bright ? Math.max(2,100*x.bright/(peak||1)) : 0,
    darkHeight: x.dark ? Math.max(2,100*x.dark/(peak||1)) : 0}));
}

function denoiseReviewHighlight() {
  const root = denoiseReviewRoot();
  if (!root) return;
  for (const bar of root.querySelectorAll('[data-dr-bar]')) {
    const active = Number(bar.dataset.drBar) === denoiseReviewFrame;
    bar.classList.toggle('is-current', active);
    bar.tabIndex = active ? 0 : -1;
    bar.setAttribute('aria-current', active ? 'true' : 'false');
  }
  const selected = denoiseReviewEntries.find(x => x.frame === denoiseReviewFrame);
  const readout = denoiseReviewElement('chart-status');
  if (readout && selected) {
    readout.textContent = `Frame ${denoiseReviewFrame+1}: brightened ${selected.bright.toFixed(2)} / 255 · darkened ${selected.dark.toFixed(2)} / 255`;
  } else if (readout && denoiseReviewEntries.length) {
    readout.textContent = `Frame ${denoiseReviewFrame+1}: not in the chart sample; the exact stills are still available.`;
  }
}

async function denoiseReviewLoadChart(path) {
  const ticket = ++denoiseReviewChartEpoch;
  const chart = denoiseReviewElement('chart');
  if (!chart) return;
  chart.replaceChildren();
  const chartStatus = denoiseReviewElement('chart-status');
  if (chartStatus) chartStatus.textContent = 'Calculating luminance changes...';
  try {
    const response = await fetch('/api/denoise-frame-differences?variant=' + encodeURIComponent(path), {cache:'no-store'});
    if (!response.ok) throw new Error('The chart endpoint answered with HTTP ' + response.status + '.');
    const report = await response.json().catch(() => { throw new Error('The chart endpoint returned malformed JSON.'); });
    if (!report || typeof report !== 'object') throw new Error('The chart endpoint returned no report.');
    if (!report.ok) throw new Error(String(report.error || 'The chart is unavailable.'));
    if (ticket !== denoiseReviewChartEpoch || denoiseReviewVariant !== path || !chart.isConnected) return;
    const rows = denoiseReviewChartRows(report);
    denoiseReviewEntries = rows;
    const parts = document.createDocumentFragment();
    for (const item of rows) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'denoise-review-bar';
      button.dataset.drBar = String(item.frame);
      button.tabIndex = -1;
      button.style.setProperty('--bright-height',item.brightHeight.toFixed(2)+'%');
      button.style.setProperty('--dark-height',item.darkHeight.toFixed(2)+'%');
      button.title = `Frame ${item.frame+1} · bright +${item.bright.toFixed(2)} · dark −${item.dark.toFixed(2)} / 255`;
      button.setAttribute('aria-label',button.title);
      button.addEventListener('click', () => denoiseReviewSelectFrame(item.frame, true));
      parts.appendChild(button);
    }
    chart.replaceChildren(parts);
    if (report.sampled) chart.title = 'Up to 240 real frame samples, not a complete film scan';
    denoiseReviewHighlight();
  } catch (error) {
    if (ticket === denoiseReviewChartEpoch && chart.isConnected && chartStatus)
      chartStatus.textContent = 'Chart unavailable: ' + denoiseReviewErrorText(error);
  }
}

function denoiseReviewLoadImage(src) {
  return new Promise((resolve,reject) => {
    if (typeof src !== 'string' || !src) {
      reject(new Error('The server did not return a still image path.'));
      return;
    }
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The still image could not be loaded from the cache.'));
    image.src = '/media?path=' + encodeURIComponent(src);
  });
}

function denoiseReviewSetDiffMode(mode) {
  denoiseReviewDiffMode = ['direction','rgb'].includes(mode) ? mode : 'direction';
  denoiseReviewRefreshDiff();
}

function denoiseReviewSetDiffContrast(n) {
  denoiseReviewDiffContrast = [8,32,64].includes(Number(n)) ? Number(n) : 32;
  denoiseReviewRefreshDiff();
}

function denoiseReviewRefreshDiff() {
  if (!denoiseReviewImagePair || !denoiseReviewRoot()?.isConnected) return;
  try { denoiseReviewRenderDiff(denoiseReviewImagePair.before,denoiseReviewImagePair.after); }
  catch (error) {
    const readout = denoiseReviewElement('readout');
    if (readout) readout.textContent = 'Pixel difference unavailable: ' + denoiseReviewErrorText(error);
  }
}

function denoiseReviewRenderDiff(before,after) {
  const width = before.naturalWidth, height = before.naturalHeight;
  if (!(width > 0 && height > 0) || width !== after.naturalWidth || height !== after.naturalHeight)
    throw new Error('The still images do not have identical dimensions.');
  // Identical pixel metric to Clean Up. Use only decoded original images;
  // separate A/B/C playback previews are lossy H.264 and not measured here.
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d',{willReadFrequently:true});
  if (!ctx) throw new Error('A 2D canvas is not available.');
  ctx.drawImage(before,0,0);
  const a = ctx.getImageData(0,0,width,height);
  ctx.clearRect(0,0,width,height);
  ctx.drawImage(after,0,0);
  const b = ctx.getImageData(0,0,width,height);
  const diff = cleanupDifferencePixels(a.data,b.data,width,height,8);
  // Only the shown pixels change; mean and changed-percent use the original
  // unamplified measurements (same as the existing Deflicker inspector).
  const visible = comparisonContrastPixels(a.data,b.data,width,height,
    denoiseReviewDiffMode,denoiseReviewDiffContrast);
  const output = denoiseReviewElement('difference');
  const outputContext = output?.getContext('2d');
  if (!output || !outputContext) throw new Error('The difference canvas is no longer on screen.');
  output.width = width; output.height = height;
  outputContext.putImageData(new ImageData(visible,width,height),0,0);
  denoiseReviewMetrics = {mean: diff.mean, changedPercent: diff.changedPercent};
  const readout = denoiseReviewElement('readout');
  if (readout) readout.textContent =
    `Mean absolute luminance difference: ${diff.mean.toFixed(2)} / 255 · Pixels with |ΔY| ≥ 1: ${diff.changedPercent.toFixed(2)}% · Display ${denoiseReviewDiffContrast}× + gamma. Judge grain separately.`;
  const caption = denoiseReviewElement('diff-caption');
  if (caption) caption.textContent = comparisonContrastCaption(denoiseReviewDiffMode, 'en');
  if (typeof denoiseMatrixUpdateUi === 'function') denoiseMatrixUpdateUi();
}

function denoiseReviewAlignVideo(frame) {
  if (!(denoiseReviewFps>0)) return;
  const t = frame / denoiseReviewFps;
  const delta = t - denoiseCompareStart;
  if (delta >= 0 && delta < (denoiseCompareEffectiveLength || denoiseCompareLength) - 0.001) {
    denoiseCompareSeek(delta);
  } else {
    // A long-film frame selected outside the 4-second A/B/C preview:
    // reload all preview clips at the corresponding source-relative time.
    denoiseCompareStart = Math.max(0,t-0.5);
    const st = state?.stages?.find(x => x.key === active);
    if (st) renderDenoiseComparePlayers(denoiseCompareCandidates(st));
    const videos = denoiseCompareVideos();
    for (const video of videos) {
      video.addEventListener('loadedmetadata', () => {
        try { video.pause(); video.currentTime = Math.max(0,t-denoiseCompareStart); }
        catch (_) {}
      },{once:true});
    }
    const control = document.getElementById('denoiseCompareStart');
    if (control) control.value = String(denoiseCompareStart);
  }
}

async function denoiseReviewSelectFrame(frame, syncVideos = false) {
  const path = denoiseReviewVariant;
  if (!path) return;
  const requested = Math.max(0, Math.min(denoiseReviewCount ? denoiseReviewCount - 1 : Number.MAX_SAFE_INTEGER,
    Math.round(Number(frame) || 0)));
  // A frame decode is expensive. Keep one request in flight and remember only
  // the newest frame the user asked for, so fast stepping cannot stack FFmpeg
  // decodes that are then discarded.
  if (denoiseReviewInFlight) {
    denoiseReviewPending = { path, frame: requested, syncVideos, ticket: denoiseReviewEpoch };
    return;
  }
  denoiseReviewInFlight = true;
  const ticket = ++denoiseReviewEpoch;
  const status = denoiseReviewElement('status');
  if (!status) {
    denoiseReviewInFlight = false;
    return;
  }
  status.textContent = `Decoding frame ${requested+1}...`;
  try {
    const response = await fetch('/api/denoise-inspector?variant=' + encodeURIComponent(path) +
      '&frame=' + requested,{cache:'no-store'});
    if (!response.ok) throw new Error('The inspection endpoint answered with HTTP ' + response.status + '.');
    const data = await response.json().catch(() => { throw new Error('The inspection endpoint returned malformed JSON.'); });
    if (!data || typeof data !== 'object') throw new Error('The inspection endpoint returned no frame.');
    if (!data.ok) throw new Error(String(data.error || 'The exact frame is unavailable.'));
    if (!data.exact) throw new Error('The server did not return an exact decoded frame.');
    const total = Number(data.frames), fps = Number(data.fps), actual = Number(data.frame);
    if (!Number.isInteger(total) || total < 1 || total > 150000 ||
        !Number.isFinite(fps) || fps <= 0 ||
        !Number.isInteger(actual) || actual < 0 || actual >= total ||
        !(Number(data.width) > 0) || !(Number(data.height) > 0)) {
      throw new Error('The server returned invalid frame metadata.');
    }
    if (typeof data.before !== 'string' || typeof data.after !== 'string' || !data.before || !data.after) {
      throw new Error('The server did not return both still images.');
    }
    const [original, processed] = await Promise.all([
      denoiseReviewLoadImage(data.before),denoiseReviewLoadImage(data.after)
    ]);
    if (ticket !== denoiseReviewEpoch || path !== denoiseReviewVariant || !denoiseReviewRoot()?.isConnected) return;
    denoiseReviewCount = total;
    denoiseReviewFps = fps;
    denoiseReviewFrame = actual;
    denoiseReviewHasFrame = true;
    denoiseReviewPhase = 'ready';
    denoiseReviewSetLabel(denoiseVariantLabel(path));
    denoiseReviewRoot()?.removeAttribute('aria-busy');
    denoiseReviewElement('before').src = original.src;
    denoiseReviewElement('after').src = processed.src;
    denoiseReviewElement('wipe-before').src = original.src;
    denoiseReviewElement('wipe-after').src = processed.src;
    denoiseReviewImagePair = {before:original,after:processed};
    const wipe = denoiseReviewElement('wipe-picture');
    if (wipe) wipe.style.aspectRatio = `${data.width} / ${data.height}`;
    const slider = denoiseReviewElement('timeline');
    slider.disabled = false; slider.max = String(total - 1);
    slider.value = String(actual);
    for(const key of ['prev','prev10','next','next10'])
      denoiseReviewElement(key).disabled = false;
    status.textContent = `Exact: frame ${actual+1}/${total} · ${(actual/fps).toFixed(3)} s · ${data.width}×${data.height}`;
    denoiseReviewSetZoom(denoiseReviewZoom);
    try { denoiseReviewRenderDiff(original,processed); }
    catch(error) {
      denoiseReviewElement('readout').textContent = 'Pixel difference unavailable: ' + denoiseReviewErrorText(error);
    }
    denoiseReviewHighlight();
    if (syncVideos) denoiseReviewAlignVideo(denoiseReviewFrame);
  } catch(error) {
    if (ticket === denoiseReviewEpoch && path === denoiseReviewVariant && denoiseReviewRoot()?.isConnected) {
      const message = denoiseReviewErrorText(error);
      if (!denoiseReviewHasFrame) {
        // The variant was never shown: replace the body with an explicit
        // no-result state rather than leaving a mislabelled blank panel.
        denoiseReviewShowHold('error', denoiseVariantLabel(path), message);
      } else if (status.isConnected) {
        status.textContent = `Frame ${requested+1} could not be decoded: ${message} Still showing frame ${denoiseReviewFrame+1}.`;
      }
    }
  } finally {
    denoiseReviewInFlight = false;
    const queued = denoiseReviewPending;
    denoiseReviewPending = null;
    if (queued && queued.path === denoiseReviewVariant &&
        queued.ticket === denoiseReviewEpoch) {
      void denoiseReviewSelectFrame(queued.frame, queued.syncVideos);
    }
  }
}

function denoiseReviewStart() {
  const root = denoiseReviewRoot();
  if (!root) return;
  denoiseReviewEpoch++;
  denoiseReviewChartEpoch++;
  denoiseReviewCount = 0;
  denoiseReviewHasFrame = false;
  denoiseReviewImagePair = null;
  denoiseReviewEntries = [];
  denoiseReviewMetrics = null;
  denoiseReviewInFlight = false;
  denoiseReviewPending = null;
  denoiseWipePointer = null;
  root.style.setProperty('--denoise-review-zoom',String(denoiseReviewZoom));
  if (!root.querySelector('[data-dr-wipe-picture]')) return;
  // Bind the body even when a no-result hold hides it: choosing a rendered
  // variant later reveals the same elements without a full redraw.
  for(const [key,delta] of [['prev10',-10],['prev',-1],['next',1],['next10',10]]) {
    root.querySelector('[data-dr-'+key+']')?.addEventListener('click',() =>
      denoiseReviewSelectFrame(denoiseReviewFrame+delta,true));
  }
  root.querySelector('[data-dr-timeline]')?.addEventListener('change',event =>
    denoiseReviewSelectFrame(Number(event.target.value),true));
  const chart = denoiseReviewElement('chart');
  chart?.addEventListener('keydown', event => {
    const keys = {ArrowLeft:-1, ArrowRight:1, Home:0, End:0, PageUp:-10, PageDown:10};
    if (!(event.key in keys)) return;
    const bars = [...chart.querySelectorAll('[data-dr-bar]')];
    if (!bars.length) return;
    const current = Math.max(0, bars.findIndex(bar => bar.tabIndex === 0));
    const next = event.key === 'Home' ? 0
      : event.key === 'End' ? bars.length - 1
      : Math.max(0, Math.min(bars.length - 1, current + keys[event.key]));
    event.preventDefault();
    bars[next].focus();
    bars[next].click();
  });
  const picture = denoiseReviewElement('wipe-picture');
  if (picture) {
    const setFromPointer = event => {
      const bounds = picture.getBoundingClientRect();
      if (bounds.width <= 0) return;
      const percent = Math.round(100 * (event.clientX - bounds.left) / bounds.width);
      denoiseReviewSetWipe(Math.max(0, Math.min(100, percent)));
    };
    picture.addEventListener('pointerdown', event => {
      if (denoiseWipePointer !== null) return;
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      event.preventDefault();
      denoiseWipePointer = event.pointerId;
      picture.setPointerCapture(event.pointerId);
      setFromPointer(event);
    });
    picture.addEventListener('pointermove', event => {
      if (event.pointerId !== denoiseWipePointer) return;
      if (picture.hasPointerCapture(event.pointerId)) setFromPointer(event);
    });
    const endDrag = event => {
      if (event.type !== 'blur' && event.pointerId !== denoiseWipePointer) return;
      if (event.pointerId != null && typeof picture.hasPointerCapture === 'function' &&
          picture.hasPointerCapture(event.pointerId)) {
        try { picture.releasePointerCapture(event.pointerId); } catch (_) { /* already gone */ }
      }
      denoiseWipePointer = null;
    };
    picture.addEventListener('pointerup', endDrag);
    picture.addEventListener('pointercancel', endDrag);
    // A browser-cancelled gesture, a lost capture or a window switch must not
    // leave the wipe stuck mid-drag for the next pointer.
    picture.addEventListener('lostpointercapture', endDrag);
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('blur', endDrag);
    }
  }
  const wipeFold = root.querySelector('.denoise-review-wipe');
  wipeFold?.addEventListener('toggle', () => { if (wipeFold.open) denoiseReviewSetWipe(denoiseReviewWipe); });
  denoiseReviewSetWipe(denoiseReviewWipe);
  if (denoiseReviewPhase === 'hold' || !denoiseReviewVariant) {
    const chartStatus = denoiseReviewElement('chart-status');
    if (chartStatus) chartStatus.textContent = 'No result selected; the frame chart is paused.';
    return;
  }
  void denoiseReviewSelectFrame(denoiseReviewFrame,false);
  void denoiseReviewLoadChart(denoiseReviewVariant);
}
