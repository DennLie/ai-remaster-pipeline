/* ARP frame-inspector v1: exact stills, browser preview loop and speed.
 * Injected into existing stage comparison panels. No arbitrary media path requests.
 */
function frameInspectorHtml(stage) {
  const cleanupWorkspace = stage === 'cleanup' ? `
    <section class="fi-exact-workspace" data-fi-workspace hidden aria-label="Drei exakte Einzelbildansichten">
      <div class="fi-exact-heading"><strong>Einzelbildvergleich</strong><span>Alle drei Ansichten zeigen denselben dekodierten Frame. Die Videowiedergabe bleibt verfügbar.</span></div>
      <div class="fi-exact-grid">
        <figure class="fi-exact-panel">
          <figcaption>ORIGINAL – Einzelbild</figcaption>
          <div class="fi-exact-media" data-fi-still-before></div>
        </figure>
        <figure class="fi-exact-panel">
          <figcaption>LIGHT DEFLICKER – Einzelbild</figcaption>
          <div class="fi-exact-media" data-fi-still-after></div>
        </figure>
        <figure class="fi-exact-panel">
          <figcaption>PIXELDIFFERENZ ×8</figcaption>
          <div class="fi-exact-media"><canvas data-fi-difference aria-label="Verstärkte Helligkeitsunterschiede zwischen Original und Ergebnis"></canvas></div>
          <small>Türkis = aufgehellt · Orange = abgedunkelt · Schwarz = keine Differenz. Keine Defektmaske.</small>
        </figure>
      </div>
      <output class="fi-difference-readout" data-fi-difference-readout aria-live="polite"></output>
    </section>` : '';
  return `<section class="frame-inspector" data-fi-stage="${stage}" aria-label="Einzelbildanalyse">
    <div class="fi-heading"><strong>Frame Inspector</strong><small>Exakte Einzelbilder aus Original und Ergebnis (FFmpeg)</small></div>
    <div class="fi-buttons">
      <button type="button" data-fi-open>${stage === 'cleanup' ? 'Einzelbilder anzeigen' : 'Einzelbildmodus starten'}</button>
      <button type="button" data-fi-close disabled>${stage === 'cleanup' ? 'Einzelbilder ausblenden' : 'Zum Video zurück'}</button>
      <button type="button" data-fi-prev10 disabled>-10</button>
      <button type="button" data-fi-prev disabled>-1 Frame</button>
      <button type="button" data-fi-next disabled>+1 Frame</button>
      <button type="button" data-fi-next10 disabled>+10</button>
      <label>Zoom <select data-fi-zoom><option value="1">100%</option><option value="2">200%</option><option value="4">400%</option></select></label>
    </div>
    <input data-fi-timeline type="range" min="0" max="0" step="1" value="0" disabled aria-label="Exakte Frameposition">
    <div class="fi-status" data-fi-status role="status" aria-live="polite">Einzelbildansicht laden, um exakt dekodierte Frames anzuzeigen.</div>
    ${cleanupWorkspace}
    <div class="fi-buttons fi-playback">
      <label>Tempo <select data-fi-speed><option value="0.25">0,25×</option><option value="0.5">0,5×</option><option value="1" selected>1×</option></select></label>
      <label>Abschnitt von Frame <input data-fi-in type="number" min="1" step="1" value="1"></label>
      <label>bis <input data-fi-out type="number" min="1" step="1" value="1"></label>
      <label><input data-fi-loop type="checkbox"> Loop in Videowiedergabe</label>
    </div>
    <p class="shot-empty">${stage === 'cleanup' ? 'Video und exakte Einzelbilder sind gleichzeitig sichtbar. Beim Abspielen bleibt der zuletzt gewählte Einzelbild-Frame angeheftet; ein Diagrammklick oder Frame-Schritt wählt einen neuen Frame.' : 'Die Einzelbildansicht ist bildindexgenau. Wiedergabe, Tempo und Loop nutzen den Browser und sind zeitlich nur annähernd synchron.'}</p>
  </section>`;
}

function setupFrameInspector(sliderId, player, before, after, slider) {
  const stage = sliderId === 'cleanupCompareSlider' ? 'cleanup' :
    sliderId === 'stabilizationCompareSlider' ? 'stabilize' : '';
  if (!stage || !player) return;
  const root = player.parentElement.querySelector(`.frame-inspector[data-fi-stage="${stage}"]`);
  if (!root || root.dataset.fiWired) return;
  root.dataset.fiWired = 'true';
  const $ = s => root.querySelector(`[data-fi-${s}]`);
  const open = $('open'), close = $('close'), timeline = $('timeline'), status = $('status');
  const start = $('in'), end = $('out'), loop = $('loop');
  const zoom = $('zoom'), speed = $('speed');
  const workspace = root.querySelector('[data-fi-workspace]');
  let exact = false, busy = false, epoch = 0, count = 0, index = 0, fps = 0;
  let pendingFrame = null;
  let beforeImage, afterImage;
  const clamp = (n,lo,hi) => Math.max(lo,Math.min(hi,n));

  const applyZoom = () => {
    const value = Number(zoom.value) || 1;
    player.style.setProperty('--fi-zoom', String(value));
    syncClip();
  };
  const syncClip = () => {
    if (!afterImage || stage === 'cleanup') return;
    const z = Number(zoom.value) || 1;
    const edge = 50 + (Number(slider.value) - 50) / z;
    // Keep the split at the physical handle position even when images are zoomed.
    const rightSide = stage === 'cleanup';
    afterImage.style.clipPath = rightSide
      ? `inset(0 0 0 ${edge}%)`
      : `inset(0 ${100-edge}% 0 0)`;
  };
  const imagesReady = urls => Promise.all(urls.map(url => new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error('Einzelbild konnte nicht geladen werden.'));
    img.src = url;
  })));
  async function showFrame(n) {
    if (busy) { pendingFrame = n; return; }
    const requested = count
      ? clamp(Math.round(Number(n) || 0), 0, count - 1)
      : Math.max(0, Math.round(Number(n) || 0));
    const requestId = ++epoch;
    busy = true;
    status.textContent = `Lade Frame ${requested+1} ...`;
    try {
      const response = await fetch('/api/frame-inspector?stage=' + encodeURIComponent(stage) + '&frame=' + requested);
      if (!response.ok) throw new Error('Serverfehler ' + response.status);
      const data = await response.json();
      if (!data.ok) throw new Error(data.error || 'Frame-Zugriff fehlgeschlagen.');
      if (requestId !== epoch) return;
      if (data.frames < 1 || !data.fps) throw new Error('Ungültige Frame-Metadaten.');
      const urls = [data.before, data.after].map(p => '/media?path=' + encodeURIComponent(p));
      const [a,b] = await imagesReady(urls);
      if (requestId !== epoch) return;
      if (!beforeImage) {
        beforeImage = document.createElement('img');
        afterImage = document.createElement('img');
        beforeImage.className = 'fi-photo fi-photo-before';
        afterImage.className = 'fi-photo fi-photo-after';
        beforeImage.alt = 'Originalframe'; afterImage.alt = 'Bearbeiteter Frame';
        // Cleanup uses separate still panels; never replace the running video images.
        if (stage === 'cleanup' && workspace) {
          workspace.querySelector('[data-fi-still-before]').appendChild(beforeImage);
          workspace.querySelector('[data-fi-still-after]').appendChild(afterImage);
        } else {
          player.append(beforeImage, afterImage);
        }
      }
      before.pause(); after.pause();
      beforeImage.src = a.src; afterImage.src = b.src;
      if (workspace) {
        workspace.hidden = false;
        try {
          renderCleanupDifference(a, b, workspace);
        } catch (error) {
          workspace.querySelector('[data-fi-difference-readout]').textContent =
            'Pixeldifferenz nicht verfügbar: ' + error.message;
        }
      }
      count = data.frames; fps = data.fps; index = requested;
      timeline.disabled = false; timeline.max = String(count-1); timeline.value = String(index);
      start.max = end.max = String(count);
      if (Number(end.value) <= 1) end.value = String(count);
      for (const selector of ['prev','prev10','next','next10']) $(selector).disabled = false;
      exact = true; close.disabled = false; open.disabled = true;
      player.classList.add('frame-inspector-active');
      syncClip(); applyZoom();
      status.textContent = `Exakt: Frame ${index+1} / ${count} · ${(index/fps).toFixed(3)} s · ${data.width}×${data.height}`;
      document.dispatchEvent(new CustomEvent('arp-frame-inspected', {detail:{stage,frame:index,frames:count,fps}}));
    } catch (err) {
      status.textContent = 'Einzelbildansicht: ' + err.message;
    } finally {
      busy = false;
      if (pendingFrame !== null && root.isConnected) {
        const next = pendingFrame;
        pendingFrame = null;
        void showFrame(next);
      }
    }
  }
  // Other GUI controls can request an exact frame without touching video currentTime.
  // Frame numbers are zero-based, matching the saved Deflicker report.
  document.addEventListener('arp-request-exact-frame', event => {
    if (!root.isConnected || event.detail?.stage !== stage) return;
    const target = Number(event.detail.frame);
    if (!Number.isInteger(target) || target < 0) return;
    void showFrame(target);
  });
  open.addEventListener('click', () => showFrame(index));
  close.addEventListener('click', () => {
    ++epoch; pendingFrame = null; exact = false; player.classList.remove('frame-inspector-active');
    open.disabled = false; close.disabled = true;
    if (workspace) workspace.hidden = true;
    status.textContent = 'Einzelbilder ausgeblendet. Video bleibt unabhängig bedienbar.';
    if (stage !== 'cleanup' && fps > 0) {
      try { before.currentTime = index / fps; } catch (_) {}
    }
  });
  $('prev10').addEventListener('click', () => showFrame(index-10));
  $('prev').addEventListener('click', () => showFrame(index-1));
  $('next').addEventListener('click', () => showFrame(index+1));
  $('next10').addEventListener('click', () => showFrame(index+10));
  timeline.addEventListener('change', () => showFrame(Number(timeline.value)));
  zoom.addEventListener('change', applyZoom);
  slider.addEventListener('input', syncClip);
  speed.addEventListener('change', () => {
    before.playbackRate = after.playbackRate = Number(speed.value);
  });
  // The video loop is deliberately approximate; exact inspection always uses FFmpeg output.
  before.addEventListener('timeupdate', () => {
    if (stage === 'cleanup' || exact || !loop.checked || !fps || !count) return;
    const low = clamp((Number(start.value)||1)-1,0,count-1)/fps;
    const high = clamp(Number(end.value)||count,1,count)/fps;
    if (high > low && before.currentTime >= high - 0.015) before.currentTime = low;
  });
  before.addEventListener('loadedmetadata', () => {
    if (stage !== 'cleanup' && count === 0 && Number.isFinite(before.duration) && before.duration > 0) {
      // Only a provisional UI limit until the exact server metadata becomes available.
      end.value = String(Math.max(1,Math.round(before.duration*30)));
    }
  });
  applyZoom();
  // Clean Up keeps the inspector visible next to the video from the beginning.
  if (stage === 'cleanup') void showFrame(0);
}

// Exact comparison: pixel luminance difference, not a defect detection mask.
function cleanupDifferencePixels(before, after, width, height, amplification = 8) {
  if (!(width > 0 && height > 0) || before.length !== width * height * 4 ||
      after.length !== before.length) throw new Error('Bildgrößen stimmen nicht überein.');
  const rgba = new Uint8ClampedArray(before.length);
  let sum = 0, changed = 0;
  const total = width * height;
  for (let p = 0; p < before.length; p += 4) {
    const y0 = 0.2126 * before[p] + 0.7152 * before[p+1] + 0.0722 * before[p+2];
    const y1 = 0.2126 * after[p] + 0.7152 * after[p+1] + 0.0722 * after[p+2];
    const delta = y1 - y0;
    const abs = Math.abs(delta);
    sum += abs;
    if (abs >= 1) changed++;
    const intensity = Math.min(255, Math.round(abs * amplification));
    if (delta > 0) {
      rgba[p] = 0; rgba[p+1] = intensity; rgba[p+2] = intensity;
    } else {
      rgba[p] = intensity; rgba[p+1] = Math.round(intensity * 0.50); rgba[p+2] = 0;
    }
    rgba[p+3] = 255;
  }
  return {rgba, mean: sum / total, changedPercent: changed / total * 100};
}

function renderCleanupDifference(before, after, workspace) {
  const width = before.naturalWidth, height = before.naturalHeight;
  if (!width || !height || width !== after.naturalWidth || height !== after.naturalHeight)
    throw new Error('Nicht deckungsgleiche Einzelbilder.');
  const scratch = document.createElement('canvas');
  scratch.width = width; scratch.height = height;
  const ctx = scratch.getContext('2d', {willReadFrequently: true});
  if (!ctx) throw new Error('Canvas nicht verfügbar.');
  ctx.drawImage(before, 0, 0);
  const a = ctx.getImageData(0, 0, width, height);
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(after, 0, 0);
  const b = ctx.getImageData(0, 0, width, height);
  const diff = cleanupDifferencePixels(a.data, b.data, width, height);
  const canvas = workspace.querySelector('[data-fi-difference]');
  canvas.width = width; canvas.height = height;
  const output = canvas.getContext('2d');
  output.putImageData(new ImageData(diff.rgba, width, height), 0, 0);
  workspace.querySelector('[data-fi-difference-readout]').textContent =
    `Mittlere absolute Luminanzdifferenz: ${diff.mean.toFixed(2)} / 255 · Pixel mit |ΔY| ≥ 1: ${diff.changedPercent.toFixed(2)} % · Darstellung ×8, keine Defektmaske.`;
}

// The optional existing Deflicker diagnostics react to an exactly inspected frame.
document.addEventListener('arp-frame-inspected', event => {
  if (event.detail?.stage !== 'cleanup') return;
  const d = (typeof state !== 'undefined' && state?.cleanup_diagnostics?.deflicker) || null;
  const readout = document.getElementById('cleanupGainReadout');
  if (!d || !readout || !Array.isArray(d.samples)) return;
  const {frame,frames} = event.detail;
  if (Number(d.frames) !== frames) return;
  const sample = d.samples.find(item => Number(item.frame) === frame);
  if (!sample) {
    readout.textContent = `Frame ${frame + 1}/${frames}: kein Einzelwert im komprimierten Analysebericht.`;
    return;
  }
  const delta = 100 * (Number(sample.gain) - 1);
  readout.textContent = `Frame ${frame+1}/${frames}: ${delta>=0?'+':''}${delta.toFixed(2)} % Helligkeitskorrektur${d.sampled ? ' (Stichprobe)' : ''}`;
});
