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

function denoiseReviewVariants(candidates) {
  return (candidates || []).slice(1).filter(v => denoisePreviewIsFinal(v.path));
}

function denoiseReviewHtml(candidates) {
  const options = denoiseReviewVariants(candidates);
  if (!options.length) {
    return `<section class="denoise-review" data-denoise-review>
      <h3>Erweiterte Einzelbildansicht</h3>
      <p class="shot-empty">Noch keine fertige Rauschreduzierungs-Variante vorhanden. Nach dem Rendern erscheinen exakte Original-/Ergebnisframes, Pixeldifferenz, Frame-Balken und Wischvergleich.</p>
    </section>`;
  }
  if (!options.some(x => x.path === denoiseReviewVariant)) {
    denoiseReviewVariant = options.find(x => x.path === denoiseComparePaths[1])?.path || options[0].path;
    denoiseReviewFrame = 0;
  }
  return `<section class="denoise-review" data-denoise-review aria-label="Erweiterter bildgenauer Remove-Noise-Vergleich">
    <div class="denoise-review-heading"><h3>Erweiterte Ansicht · Einzelbildgenau</h3>
      <span>Zusätzlich zum A/B/C-Vergleich – echte dekodierte Frames, keine H.264-Vorschauen</span></div>
    <div class="denoise-review-toolbar">
      <label>Ergebnis vergleichen <select data-dr-variant onchange="denoiseReviewChoose(this.value)">
        ${options.map(x => `<option value="${esc(x.path)}" ${x.path === denoiseReviewVariant ? 'selected' : ''}>${esc(x.label)}</option>`).join('')}
      </select></label>
      <label>Zoom <select data-dr-zoom onchange="denoiseReviewSetZoom(this.value)">
        ${[1,2,4].map(v => `<option value="${v}" ${v === denoiseReviewZoom ? 'selected' : ''}>${v*100}%</option>`).join('')}
      </select></label>
    </div>
    <div class="denoise-review-rail">
      <div class="cleanup-frame-rail-heading"><strong>Frame-Änderungen</strong>
        <span>Türkis = aufgehellt · Orange = abgedunkelt · Balken wählen einen exakten Frame</span></div>
      <div class="denoise-review-chart" data-dr-chart role="group" aria-label="Helligkeitsänderungen je Einzelbild"></div>
      <output data-dr-chart-status role="status">Berechne Frame-Unterschiede ...</output>
      <small>Gemessene Pixeländerungen sind kein Beweis für Rausch- oder Kornentfernung. Bei Langfilmen nur Stichproben.</small>
    </div>
    <section class="denoise-review-inspector" aria-label="Drei Einzelbildansichten">
      <div class="fi-heading"><strong>Frame Inspector</strong>
        <small>Frame-indexgenau aus tatsächlichem Verarbeitungseingang und ausgewählter Variante</small></div>
      <div class="fi-buttons">
        <button type="button" data-dr-prev10 disabled>−10</button>
        <button type="button" data-dr-prev disabled>−1 Frame</button>
        <button type="button" data-dr-next disabled>+1 Frame</button>
        <button type="button" data-dr-next10 disabled>+10</button>
      </div>
      <input data-dr-timeline type="range" min="0" max="0" value="0" step="1" disabled aria-label="Exakte Frameposition im Clip">
      <output data-dr-status role="status">Bereite Einzelbilder vor.</output>
      <div class="denoise-review-stills">
        <figure><figcaption>ORIGINAL · Render-Eingang</figcaption>
          <div class="denoise-review-media"><img data-dr-before alt="Originalframe vor Rauschreduzierung"></div></figure>
        <figure><figcaption>ERGEBNIS · ausgewählte Variante</figcaption>
          <div class="denoise-review-media"><img data-dr-after alt="Exaktes bearbeitetes Einzelbild"></div></figure>
        <figure><figcaption>PIXELDIFFERENZ ×8</figcaption>
          <div class="denoise-review-media"><canvas data-dr-difference aria-label="Verstärkte Pixelhelligkeitsunterschiede"></canvas></div>
          <small>Türkis = heller · Orange = dunkler · Schwarz = unverändert. Keine Defektmaske.</small></figure>
      </div>
      <output data-dr-readout role="status"></output>
      <details class="denoise-review-wipe">
        <summary><strong>Wischvergleich</strong> · exakter Original/Ergebnis-Schieber</summary>
        <div class="denoise-review-wipe-picture" data-dr-wipe-picture>
          <img data-dr-wipe-before alt="Unverändertes Originalframe">
          <img data-dr-wipe-after alt="Bearbeitetes Frame">
          <div class="denoise-review-wipe-divider" data-dr-wipe-divider aria-hidden="true"></div>
        </div>
        <label>Original links · Ergebnis rechts
          <input data-dr-wipe-position type="range" min="0" max="100" step="1"
                 value="${denoiseReviewWipe}" oninput="denoiseReviewSetWipe(this.value)">
        </label>
        <output data-dr-wipe-status>50 % Original links / 50 % Ergebnis rechts</output>
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
  const control = denoiseReviewElement('wipe-position');
  if (control) control.value = String(denoiseReviewWipe);
  const status = denoiseReviewElement('wipe-status');
  if (status) status.textContent = `${denoiseReviewWipe.toFixed(0)} % Original links · ${(100-denoiseReviewWipe).toFixed(0)} % Ergebnis rechts`;
}

function denoiseReviewChoose(path) {
  const st = state?.stages?.find(x => x.key === 'denoise');
  const valid = st ? denoiseReviewVariants(denoiseCompareCandidates(st)).some(x => x.path === path) : false;
  if (!valid) return;
  denoiseReviewVariant = path;
  denoiseReviewEpoch++;
  denoiseReviewFrame = 0;
  denoiseReviewEntries = [];
  denoiseReviewStart();
}

function denoiseReviewChartRows(report) {
  if (!report || report.metric !== 'directional_mean_luma_8bit' ||
      !Number.isInteger(report.frames) || report.frames < 1 ||
      !Array.isArray(report.samples) || report.samples.length > 240 || !report.samples.length) {
    throw new Error('Ungültiger Framebericht.');
  }
  let prev = -1, peak = 0;
  const rows = report.samples.map(x => {
    const frame = Number(x.frame), bright = Number(x.brighten_mean_luma);
    const dark = Number(x.darken_mean_luma), total = Number(x.mean_abs_luma);
    if (!Number.isInteger(frame) || frame <= prev || frame >= report.frames ||
        ![bright,dark,total].every(v => Number.isFinite(v) && v >= 0 && v <= 255) ||
        Math.abs(total-bright-dark) > 0.001) throw new Error('Ungültige Frame-Messwerte.');
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
    bar.setAttribute('aria-pressed', String(active));
  }
  const selected = denoiseReviewEntries.find(x => x.frame === denoiseReviewFrame);
  const readout = denoiseReviewElement('chart-status');
  if (readout && selected) {
    readout.textContent = `Frame ${denoiseReviewFrame+1}: aufgehellt ${selected.bright.toFixed(2)} / 255 · abgedunkelt ${selected.dark.toFixed(2)} / 255`;
  } else if (readout && denoiseReviewEntries.length) {
    readout.textContent = `Frame ${denoiseReviewFrame+1}: nicht in der Diagramm-Stichprobe; die Einzelbilder sind trotzdem exakt.`;
  }
}

async function denoiseReviewLoadChart(path) {
  const ticket = ++denoiseReviewChartEpoch;
  const chart = denoiseReviewElement('chart');
  if (!chart) return;
  chart.replaceChildren();
  denoiseReviewElement('chart-status').textContent = 'Berechne Helligkeitsänderungen ...';
  try {
    const response = await fetch('/api/denoise-frame-differences?variant=' + encodeURIComponent(path), {cache:'no-store'});
    if (!response.ok) throw new Error('Serverfehler ' + response.status);
    const report = await response.json();
    if (!report.ok) throw new Error(report.error || 'Diagramm nicht verfügbar.');
    if (ticket !== denoiseReviewChartEpoch || denoiseReviewVariant !== path || !chart.isConnected) return;
    const rows = denoiseReviewChartRows(report);
    denoiseReviewEntries = rows;
    const parts = document.createDocumentFragment();
    for (const item of rows) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'denoise-review-bar';
      button.dataset.drBar = String(item.frame);
      button.style.setProperty('--bright-height',item.brightHeight.toFixed(2)+'%');
      button.style.setProperty('--dark-height',item.darkHeight.toFixed(2)+'%');
      button.title = `Frame ${item.frame+1} · hell +${item.bright.toFixed(2)} · dunkel −${item.dark.toFixed(2)} / 255`;
      button.setAttribute('aria-label',button.title);
      button.addEventListener('click', () => denoiseReviewSelectFrame(item.frame, true));
      parts.appendChild(button);
    }
    chart.replaceChildren(parts);
    if (report.sampled) chart.title = 'Bis zu 240 echte Einzelbild-Stichproben, kein vollständiges Filmprotokoll';
    denoiseReviewHighlight();
  } catch (error) {
    if (ticket === denoiseReviewChartEpoch && chart.isConnected)
      denoiseReviewElement('chart-status').textContent = 'Diagramm nicht verfügbar: ' + error.message;
  }
}

function denoiseReviewLoadImage(src) {
  return new Promise((resolve,reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Einzelbilddatei nicht verfügbar.'));
    image.src = '/media?path=' + encodeURIComponent(src);
  });
}

function denoiseReviewRenderDiff(before,after) {
  const width = before.naturalWidth, height = before.naturalHeight;
  if (!(width > 0 && height > 0) || width !== after.naturalWidth || height !== after.naturalHeight)
    throw new Error('Nicht deckungsgleiche Einzelbilder.');
  // Identical pixel metric to Clean Up. Use only decoded original images;
  // separate A/B/C playback previews are lossy H.264 and not measured here.
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d',{willReadFrequently:true});
  if (!ctx) throw new Error('2D-Canvas ist nicht verfügbar.');
  ctx.drawImage(before,0,0);
  const a = ctx.getImageData(0,0,width,height);
  ctx.clearRect(0,0,width,height);
  ctx.drawImage(after,0,0);
  const b = ctx.getImageData(0,0,width,height);
  const diff = cleanupDifferencePixels(a.data,b.data,width,height,8);
  const output = denoiseReviewElement('difference');
  output.width = width; output.height = height;
  output.getContext('2d').putImageData(new ImageData(diff.rgba,width,height),0,0);
  const readout = denoiseReviewElement('readout');
  if (readout) readout.textContent =
    `Mittlere absolute Luminanzdifferenz: ${diff.mean.toFixed(2)} / 255 · Pixel mit |ΔY| ≥ 1: ${diff.changedPercent.toFixed(2)} % · Anzeige ×8. Kornbeurteilung separat vornehmen.`;
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
  status.textContent = `Dekodiere Frame ${requested+1} …`;
  try {
    const response = await fetch('/api/denoise-inspector?variant=' + encodeURIComponent(path) +
      '&frame=' + requested,{cache:'no-store'});
    if (!response.ok) throw new Error('Serverfehler '+response.status);
    const data = await response.json();
    if (!data.ok || !data.exact) throw new Error(data.error || 'Exakter Frame nicht verfügbar.');
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
    const wipe = denoiseReviewElement('wipe-picture');
    if (wipe) wipe.style.aspectRatio = `${data.width} / ${data.height}`;
    const slider = denoiseReviewElement('timeline');
    slider.disabled = false; slider.max = String(data.frames-1);
    slider.value = String(data.frame);
    for(const key of ['prev','prev10','next','next10'])
      denoiseReviewElement(key).disabled = false;
    status.textContent = `Exakt: Frame ${data.frame+1}/${data.frames} · ${(data.frame/data.fps).toFixed(3)} s · ${data.width}×${data.height}`;
    denoiseReviewSetZoom(denoiseReviewZoom);
    try { denoiseReviewRenderDiff(original,processed); }
    catch(error) { denoiseReviewElement('readout').textContent = 'Pixeldifferenz nicht verfügbar: '+error.message; }
    denoiseReviewHighlight();
    if (syncVideos) denoiseReviewAlignVideo(denoiseReviewFrame);
  } catch(error) {
    if (ticket === denoiseReviewEpoch && status.isConnected)
      status.textContent = 'Einzelbildansicht: ' + error.message;
  }
}

function denoiseReviewStart() {
  const root = denoiseReviewRoot();
  if (!root || !root.querySelector('[data-dr-variant]')) return;
  denoiseReviewEpoch++;
  denoiseReviewChartEpoch++;
  denoiseReviewCount = 0;
  denoiseReviewEntries = [];
  root.style.setProperty('--denoise-review-zoom',String(denoiseReviewZoom));
  for(const [key,delta] of [['prev10',-10],['prev',-1],['next',1],['next10',10]]) {
    root.querySelector('[data-dr-'+key+']')?.addEventListener('click',() =>
      denoiseReviewSelectFrame(denoiseReviewFrame+delta,true));
  }
  root.querySelector('[data-dr-timeline]')?.addEventListener('change',event =>
    denoiseReviewSelectFrame(Number(event.target.value),true));
  const picture = denoiseReviewElement('wipe-picture');
  picture?.addEventListener('pointerdown',event => {
    if(event.button !== 0 && event.pointerType === 'mouse') return;
    const bounds=picture.getBoundingClientRect();
    if(bounds.width>0) denoiseReviewSetWipe(100*(event.clientX-bounds.left)/bounds.width);
  });
  denoiseReviewSetWipe(denoiseReviewWipe);
  void denoiseReviewSelectFrame(denoiseReviewFrame,false);
  void denoiseReviewLoadChart(denoiseReviewVariant);
}
