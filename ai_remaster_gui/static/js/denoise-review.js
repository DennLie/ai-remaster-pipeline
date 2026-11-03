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

function denoiseReviewVariants(candidates) {
  return (candidates || []).slice(1).filter(v => denoisePreviewIsFinal(v.path));
}

function denoiseReviewHtml(candidates) {
  const options = denoiseReviewVariants(candidates);
  if (!options.length) {
    return `<section class="card denoise-review-card denoise-review" data-denoise-review data-review-lang="en" aria-label="Exact frame review">
      <h3>Exact frame review</h3>
      <p class="shot-empty">No finished Remove Noise variant yet. After rendering, exact original/result frames, pixel difference, frame bars and the wipe comparison appear here.</p>
    </section>`;
  }
  const preferred = typeof denoiseSelectedPath === 'function' ? denoiseSelectedPath() : '';
  if (!options.some(x => x.path === denoiseReviewVariant)) {
    denoiseReviewVariant = options.find(x => x.path === (preferred || denoiseComparePaths[1]))?.path || options[0].path;
  }
  const label = typeof denoiseVariantLabel === 'function'
    ? denoiseVariantLabel(denoiseReviewVariant)
    : denoiseReviewVariant;
  return `<section class="card denoise-review-card denoise-review" data-denoise-review data-review-lang="en" aria-label="Exact frame review">
    <div class="denoise-review-heading">
      <div><h3>Exact frame review</h3>
        <span>Beyond the A/B/C preview: decoded source frames, not H.264 previews.</span></div>
      <span class="denoise-review-current">Selected: <strong data-dr-current-variant>${esc(label)}</strong></span>
    </div>
    <div class="denoise-review-toolbar">
      <span class="denoise-review-toolbar-note">Frame-exact stills from the actual processing input and the selected variant.</span>
      <label>Zoom <select data-dr-zoom onchange="denoiseReviewSetZoom(this.value)">
        ${[1,2,4].map(v => `<option value="${v}" ${v === denoiseReviewZoom ? 'selected' : ''}>${v*100}%</option>`).join('')}
      </select></label>
    </div>
    <div class="denoise-review-rail">
      <div class="cleanup-frame-rail-heading"><strong>Frame changes</strong>
        <span>Teal = brightened · Orange = darkened · Bars select an exact frame</span></div>
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
  </section>`;
}

function denoiseReviewRoot() { return document.querySelector('[data-denoise-review]'); }
function denoiseReviewElement(name) { return denoiseReviewRoot()?.querySelector('[data-dr-' + name + ']'); }

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
  const st = state?.stages?.find(x => x.key === 'denoise');
  const valid = st ? denoiseReviewVariants(denoiseCompareCandidates(st)).some(x => x.path === path) : false;
  if (!valid) return;
  denoiseReviewVariant = path;
  denoiseReviewEpoch++;
  denoiseReviewChartEpoch++;
  denoiseReviewCount = 0;
  denoiseReviewImagePair = null;
  denoiseReviewEntries = [];
  denoiseReviewMetrics = null;
  if (!preserveFrame) denoiseReviewFrame = 0;
  const current = denoiseReviewElement('current-variant');
  if (current) current.textContent = denoiseVariantLabel(path);
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
  denoiseReviewElement('chart-status').textContent = 'Calculating luminance changes...';
  try {
    const response = await fetch('/api/denoise-frame-differences?variant=' + encodeURIComponent(path), {cache:'no-store'});
    if (!response.ok) throw new Error('Server error ' + response.status);
    const report = await response.json();
    if (!report.ok) throw new Error(report.error || 'Chart unavailable.');
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
    if (ticket === denoiseReviewChartEpoch && chart.isConnected)
      denoiseReviewElement('chart-status').textContent = 'Chart unavailable: ' + error.message;
  }
}

function denoiseReviewLoadImage(src) {
  return new Promise((resolve,reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Still image file unavailable.'));
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
    if (readout) readout.textContent = 'Difference view: ' + error.message;
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
  output.width = width; output.height = height;
  output.getContext('2d').putImageData(new ImageData(visible,width,height),0,0);
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
  if (delta >= 0 && delta < denoiseCompareLength - 0.001) {
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
  const requested = Math.max(0,Math.min(denoiseReviewCount ? denoiseReviewCount-1 : Number.MAX_SAFE_INTEGER,
    Math.round(Number(frame) || 0)));
  const ticket = ++denoiseReviewEpoch;
  const status = denoiseReviewElement('status');
  if (!status) return;
  status.textContent = `Decoding frame ${requested+1}...`;
  try {
    const response = await fetch('/api/denoise-inspector?variant=' + encodeURIComponent(path) +
      '&frame=' + requested,{cache:'no-store'});
    if (!response.ok) throw new Error('Server error '+response.status);
    const data = await response.json();
    if (!data.ok || !data.exact) throw new Error(data.error || 'Exact frame unavailable.');
    const [original, processed] = await Promise.all([
      denoiseReviewLoadImage(data.before),denoiseReviewLoadImage(data.after)
    ]);
    if (ticket !== denoiseReviewEpoch || path !== denoiseReviewVariant || !denoiseReviewRoot()?.isConnected) return;
    denoiseReviewCount = Number(data.frames);
    denoiseReviewFps = Number(data.fps);
    denoiseReviewFrame = Number(data.frame);
    denoiseReviewElement('before').src = original.src;
    denoiseReviewElement('after').src = processed.src;
    denoiseReviewElement('wipe-before').src = original.src;
    denoiseReviewElement('wipe-after').src = processed.src;
    denoiseReviewImagePair = {before:original,after:processed};
    const wipe = denoiseReviewElement('wipe-picture');
    if (wipe) wipe.style.aspectRatio = `${data.width} / ${data.height}`;
    const slider = denoiseReviewElement('timeline');
    slider.disabled = false; slider.max = String(data.frames-1);
    slider.value = String(data.frame);
    for(const key of ['prev','prev10','next','next10'])
      denoiseReviewElement(key).disabled = false;
    status.textContent = `Exact: frame ${data.frame+1}/${data.frames} · ${(data.frame/data.fps).toFixed(3)} s · ${data.width}×${data.height}`;
    denoiseReviewSetZoom(denoiseReviewZoom);
    try { denoiseReviewRenderDiff(original,processed); }
    catch(error) { denoiseReviewElement('readout').textContent = 'Pixel difference unavailable: '+error.message; }
    denoiseReviewHighlight();
    if (syncVideos) denoiseReviewAlignVideo(denoiseReviewFrame);
  } catch(error) {
    if (ticket === denoiseReviewEpoch && status.isConnected)
      status.textContent = 'Still review: ' + error.message;
  }
}

function denoiseReviewStart() {
  const root = denoiseReviewRoot();
  if (!root || !root.querySelector('[data-dr-wipe-picture]')) return;
  denoiseReviewEpoch++;
  denoiseReviewChartEpoch++;
  denoiseReviewCount = 0;
  denoiseReviewImagePair = null;
  denoiseReviewEntries = [];
  denoiseReviewMetrics = null;
  root.style.setProperty('--denoise-review-zoom',String(denoiseReviewZoom));
  for(const [key,delta] of [['prev10',-10],['prev',-1],['next',1],['next10',10]]) {
    root.querySelector('[data-dr-'+key+']')?.addEventListener('click',() =>
      denoiseReviewSelectFrame(denoiseReviewFrame+delta,true));
  }
  root.querySelector('[data-dr-timeline]')?.addEventListener('change',event =>
    denoiseReviewSelectFrame(Number(event.target.value),true));
  const picture = denoiseReviewElement('wipe-picture');
  if (picture) {
    const setFromPointer = event => {
      const bounds = picture.getBoundingClientRect();
      if (bounds.width <= 0) return;
      const percent = Math.round(100 * (event.clientX - bounds.left) / bounds.width);
      denoiseReviewSetWipe(Math.max(0, Math.min(100, percent)));
    };
    picture.addEventListener('pointerdown', event => {
      if (event.pointerType === 'mouse' && event.button !== 0) return;
      event.preventDefault();
      picture.setPointerCapture(event.pointerId);
      setFromPointer(event);
    });
    picture.addEventListener('pointermove', event => {
      if (picture.hasPointerCapture(event.pointerId)) setFromPointer(event);
    });
    const endDrag = event => {
      if (picture.hasPointerCapture(event.pointerId)) picture.releasePointerCapture(event.pointerId);
    };
    picture.addEventListener('pointerup', endDrag);
    picture.addEventListener('pointercancel', endDrag);
  }
  const wipeFold = root.querySelector('.denoise-review-wipe');
  wipeFold?.addEventListener('toggle', () => { if (wipeFold.open) denoiseReviewSetWipe(denoiseReviewWipe); });
  denoiseReviewSetWipe(denoiseReviewWipe);
  void denoiseReviewSelectFrame(denoiseReviewFrame,false);
  void denoiseReviewLoadChart(denoiseReviewVariant);
}
