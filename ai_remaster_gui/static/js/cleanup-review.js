/* cleanup-two-up-v1 */
/* ARP Clean Up two-up review v1: permanent side-by-side presentation.
 * Browser playback is synchronized approximately; exact FFmpeg stills are
 * frame-index matched and rendered in the same two panels.
 * The original and the processed file are never modified by this viewer.
 */
// Exact frame ordinal to video timestamp; 0-based internally, 1-based in UI.
function cleanupFrameTimestamp(frame, fps, totalFrames) {
  if (!Number.isInteger(frame) || !Number.isFinite(fps) || fps <= 0 ||
      !Number.isInteger(totalFrames) || totalFrames < 1) return null;
  return Math.max(0, Math.min(frame, totalFrames - 1)) / fps;
}


// FFmpeg does not report applied frame gains. Show both directions of
// measured pixel brightness changes; a frame may contain *both* colors.
function cleanupDifferenceBarData(report) {
  const frames = Number(report?.frames);
  const samples = report?.samples;
  if (!Number.isInteger(frames) || frames < 1 || !Array.isArray(samples) ||
      samples.length < 1 || samples.length > 240 ||
      report?.metric !== 'directional_mean_luma_8bit') {
    throw new Error('Ungültiger gerichteter FFmpeg-Framebericht.');
  }
  let previous = -1, maximumDirection = 0, maximumTotal = 0;
  const values = samples.map(item => {
    const frame = Number(item.frame);
    const value = Number(item.mean_abs_luma);
    const bright = Number(item.brighten_mean_luma);
    const dark = Number(item.darken_mean_luma);
    if (!Number.isInteger(frame) || frame <= previous || frame >= frames ||
        ![value, bright, dark].every(x => Number.isFinite(x) && x >= 0 && x <= 255) ||
        Math.abs(value - bright - dark) > 0.001) {
      throw new Error('Inkonsistente Helligkeitswerte im FFmpeg-Bericht.');
    }
    previous = frame;
    maximumDirection = Math.max(maximumDirection, bright, dark);
    maximumTotal = Math.max(maximumTotal, value);
    return {frame, value, bright, dark};
  });
  return values.map(item => ({
    ...item,
    height: Math.max(2, Math.min(100, 100 * item.value / (maximumTotal || 1))),
    brightHeight: item.bright === 0 ? 0 : Math.max(2, 100 * item.bright / (maximumDirection || 1)),
    darkHeight: item.dark === 0 ? 0 : Math.max(2, 100 * item.dark / (maximumDirection || 1))
  }));
}

function setupFFmpegDifferenceRail(slider, player, before) {
  const method = player.dataset.deflickerMethod;
  if (!['ffmpeg_median', 'ffmpeg_am', 'localized_research'].includes(method)) return;
  const rail = document.createElement('section');
  rail.className = 'cleanup-frame-rail cleanup-difference-rail';
  rail.setAttribute('aria-label', 'Aufhellungen und Abdunklungen je Frame');
  rail.innerHTML = `
    <div class="cleanup-frame-rail-heading">
      <strong>Frame-Änderungen</strong>
      <span>Original ↔ Ergebnis · Balken anklicken: Frame auswählen</span>
    </div>
    <div class="cleanup-gain-plot" role="group" aria-label="Türkis für hellere, Orange für dunklere Pixel pro Frame"></div>
    <div class="cleanup-diag-legend">
      <span class="cleanup-bright-key">Türkis: aufgehellt</span>
      <span class="cleanup-dark-key">Orange: abgedunkelt</span>
      <span>Je Richtung mittlere Pixelhelligkeitsänderung (0–255); keine Deflicker-Gain-Werte.</span>
    </div>
    <p class="cleanup-diag-readout" role="status">Berechne gerichtete Pixelunterschiede aus den vorhandenen Videos …</p>`;
  slider.insertAdjacentElement('afterend', rail);
  player.closest('.editor-viewer')?.classList.add('has-cleanup-frame-rail');
  const plot = rail.querySelector('.cleanup-gain-plot');
  const readout = rail.querySelector('.cleanup-diag-readout');
  let bars = [], reportFrames = 0, selectedFrame = 0;
  const highlight = frame => {
    selectedFrame = frame;
    for (const bar of bars) {
      const active = Number(bar.dataset.cleanupFrame) === frame;
      bar.classList.toggle('is-current', active);
      bar.setAttribute('aria-pressed', String(active));
    }
    const match = bars.find(bar => Number(bar.dataset.cleanupFrame) === frame);
    if (match) {
      readout.textContent =
        `Frame ${frame+1}/${reportFrames}: ` +
        `aufgehellt ${Number(match.dataset.brightLuma).toFixed(2)} / 255 · ` +
        `abgedunkelt ${Number(match.dataset.darkLuma).toFixed(2)} / 255`;
    }
  };
  document.addEventListener('arp-frame-inspected', event => {
    if (!rail.isConnected || event.detail?.stage !== 'cleanup') return;
    highlight(Number(event.detail.frame));
  });
  before.addEventListener('timeupdate', () => {
    if (!rail.isConnected || !(reportFrames > 0) || !(before.duration > 0) || before.paused) return;
    const at = Math.min(reportFrames-1, Math.max(0, Math.floor(before.currentTime / before.duration * reportFrames)));
    highlight(at);
  });
  fetch('/api/cleanup-frame-differences', {cache:'no-store'})
    .then(response => {
      if (!response.ok) throw new Error('Serverfehler ' + response.status);
      return response.json();
    })
    .then(data => {
      if (!rail.isConnected) return;
      if (!data.ok) throw new Error(data.error || 'Analyse nicht verfügbar.');
      const entries = cleanupDifferenceBarData(data);
      reportFrames = data.frames;
      const fragment = document.createDocumentFragment();
      for (const entry of entries) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'cleanup-gain-bar difference';
        button.dataset.cleanupFrame = String(entry.frame);
        button.dataset.brightLuma = String(entry.bright);
        button.dataset.darkLuma = String(entry.dark);
        button.style.setProperty('--bright-height', entry.brightHeight.toFixed(1) + '%');
        button.style.setProperty('--dark-height', entry.darkHeight.toFixed(1) + '%');
        button.title = `Frame ${entry.frame+1}: aufgehellt ${entry.bright.toFixed(2)} / 255, abgedunkelt ${entry.dark.toFixed(2)} / 255`;
        button.setAttribute('aria-label', button.title);
        button.setAttribute('aria-pressed', 'false');
        button.addEventListener('click', () => {
          highlight(entry.frame);
          document.dispatchEvent(new CustomEvent('arp-request-exact-frame', {
            detail: {stage:'cleanup', frame:entry.frame}
          }));
        });
        fragment.appendChild(button);
        bars.push(button);
      }
      plot.replaceChildren(fragment);
      const legend = rail.querySelector('.cleanup-diag-legend');
      if (legend) {
        const note = document.createElement('span');
        note.textContent = `${data.sampled ? 'Stichprobe pro Block · ' : ''}Balkenhöhe je Farbe relativ zum größten sichtbaren Richtungswert.`;
        legend.appendChild(note);
      }
      highlight(selectedFrame);
    }).catch(error => {
      if (rail.isConnected) readout.textContent = 'Frame-Änderungen nicht verfügbar: ' + error.message;
    });
}

function setupCleanupReview(sliderId, player, before, after, slider) {
  if (sliderId !== 'cleanupCompareSlider' || player.dataset.cleanupTwoUpReady) return;
  const inspector = player.parentElement?.querySelector('.frame-inspector[data-fi-stage="cleanup"]');
  if (!inspector) return;
  player.dataset.cleanupTwoUpReady = 'true';
  player.classList.add('cleanup-two-up');
  // The correction chart belongs immediately underneath both video previews.
  // Move only the interactive chart; advanced metrics stay above the video.
  const diagnostics = player.parentElement.querySelector('.cleanup-diagnostics');
  const plot = diagnostics?.querySelector('.cleanup-gain-plot');
  const legend = diagnostics?.querySelector('.cleanup-diag-legend');
  const readout = diagnostics?.querySelector('#cleanupGainReadout');
  if (!plot && ['ffmpeg_median', 'ffmpeg_am', 'localized_research'].includes(player.dataset.deflickerMethod)) {
    setupFFmpegDifferenceRail(slider, player, before);
  }
  if (plot && legend && readout) {
    const oldTitle = plot.previousElementSibling;
    if (oldTitle?.tagName === 'H4') oldTitle.remove();
    const rail = document.createElement('section');
    rail.className = 'cleanup-frame-rail';
    rail.setAttribute('aria-label', 'Helligkeitskorrektur pro Einzelbild');
    rail.innerHTML = '<div class="cleanup-frame-rail-heading"><strong>Frame-Korrekturen</strong><span>Balken anklicken: exakten Frame anzeigen</span></div>';
    rail.append(plot, legend, readout);
    // Keep the hidden slider directly behind player: bindCleanupDiagnostics uses
    // slider.previousElementSibling to find the video pair.
    slider.insertAdjacentElement('afterend', rail);
    player.closest('.editor-viewer')?.classList.add('has-cleanup-frame-rail');
  }
  player.querySelectorAll('.cleanup-compare-label').forEach(node => node.remove());
  slider.hidden = true;
  slider.setAttribute('aria-hidden', 'true');

  // Move the *existing* video elements. No new video URL or second set of
  // decoders, and the pre-existing ARP stage binding retains its references.
  const layout = document.createElement('div');
  layout.className = 'cleanup-two-up-grid';
  const resultLabel = player.dataset.deflickerMethod === 'ffmpeg_median' ? 'FFMPEG MEDIAN'
    : player.dataset.deflickerMethod === 'ffmpeg_am' ? 'FFMPEG ARITHMETIC MEAN'
    : player.dataset.deflickerMethod === 'localized_research' ? 'LOCALIZED · EXPERIMENTELL'
    : player.dataset.deflickerMethod === 'arp' ? 'ARP CONSERVATIVE' : 'CLEAN-UP-ERGEBNIS';
  layout.innerHTML = `
    <div class="cleanup-side" data-compare-panel="before">
      <div class="cleanup-side-heading"><strong>ORIGINAL</strong><span>Vorher · unverändert</span></div>
      <div class="cleanup-side-media" data-compare-side="before"></div>
    </div>
    <div class="cleanup-side" data-compare-panel="after">
      <div class="cleanup-side-heading"><strong>${resultLabel}</strong><span>Nachher · bearbeitet</span></div>
      <div class="cleanup-side-media" data-compare-side="after"></div>
    </div>`;
  player.appendChild(layout);
  const beforeMedia = layout.querySelector('[data-compare-side="before"]');
  const afterMedia = layout.querySelector('[data-compare-side="after"]');
  beforeMedia.appendChild(before);
  afterMedia.appendChild(after);
  // The previous wipe implementation clipped the processed video. Never crop
  // either panel, even when the old slider event fires.
  after.style.clipPath = 'none';
  before.controls = false;
  after.controls = false;
  before.muted = false;
  after.muted = true;
  for (const video of [before, after]) {
    video.playsInline = true;
    video.preload = 'auto';
    video.addEventListener('loadedmetadata', () => {
      if (video.videoWidth > 0 && video.videoHeight > 0) {
        video.parentElement.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
      }
    });
    if (video.readyState > 0 && video.videoWidth > 0 && video.videoHeight > 0) {
      video.parentElement.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    }
  }

  const controls = document.createElement('section');
  controls.className = 'cleanup-two-up-controls';
  controls.innerHTML = `
    <div class="cleanup-two-up-tools">
      <button type="button" data-two-play aria-label="Beide Videos abspielen">Beide abspielen</button>
      <button type="button" data-two-pause>Pause</button>
      <button type="button" data-two-rewind>Anfang</button>
      <span data-two-clock class="cleanup-two-up-clock">0.000 s</span>
    </div>
    <label class="cleanup-two-up-seek-label" for="cleanupTwoUpSeek">Gemeinsame Video-Timeline</label>
    <input type="range" min="0" max="1" step="0.001" value="0" id="cleanupTwoUpSeek" data-two-seek aria-label="Gemeinsame Videoposition">
    <p data-two-mode role="status">Beide Videos sind gleichzeitig sichtbar. Bei laufender Wiedergabe ist die Synchronität näherungsweise; der Frame Inspector zeigt exakt denselben Frame in beiden Spalten.</p>
    <p data-two-gain class="cleanup-two-up-gain" aria-live="polite"></p>`;
  player.parentElement.insertBefore(controls, player);
  const control = name => controls.querySelector(`[data-two-${name}]`);
  const play = control('play');
  const pause = control('pause');
  const rewind = control('rewind');
  const seek = control('seek');
  const clock = control('clock');
  const mode = control('mode');
  const gain = control('gain');
  const loop = inspector.querySelector('[data-fi-loop]');
  const loopIn = inspector.querySelector('[data-fi-in]');
  const loopOut = inspector.querySelector('[data-fi-out]');
  const speed = inspector.querySelector('[data-fi-speed]');
  const exact = () => player.classList.contains('frame-inspector-active');
  let fps = 0, frameCount = 0, seeking = false, syncing = false;
  let pendingVideoFrame = null;
  const validDuration = () => Number.isFinite(before.duration) && before.duration > 0 ? before.duration : 0;
  const currentTime = () => Number.isFinite(before.currentTime) ? before.currentTime : 0;
  function limits() {
    if (!(fps > 0 && frameCount > 0)) return null;
    const first = Math.max(1, Math.min(frameCount, Math.round(Number(loopIn?.value) || 1)));
    const last = Math.max(first, Math.min(frameCount, Math.round(Number(loopOut?.value) || frameCount)));
    return { start: (first - 1) / fps, end: last / fps };
  }
  function moveBoth(position) {
    if (!Number.isFinite(position)) return;
    const duration = validDuration();
    const target = Math.max(0, Math.min(position, duration || position));
    for (const v of [before, after]) {
      try { v.currentTime = target; } catch (_) {}
    }
    updateClock();
  }
  // A frame selection must update both video clocks and the shared time slider.
  // Metadata can still be loading when the first chart bar is selected.
  function applyFrameSeek(frame) {
    const seconds = cleanupFrameTimestamp(frame, fps, frameCount);
    if (seconds === null) {
      pendingVideoFrame = frame;
      return;
    }
    pendingVideoFrame = null;
    pauseBoth();
    moveBoth(seconds);
  }
  document.addEventListener('arp-cleanup-seek-video', event => {
    if (!player.isConnected || event.detail?.stage !== 'cleanup') return;
    const requested = Number(event.detail.frame);
    if (!Number.isInteger(requested) || requested < 0) return;
    applyFrameSeek(frameCount > 0 ? Math.min(requested, frameCount - 1) : requested);
  });
  function updateClock() {
    const duration = validDuration();
    const t = Math.max(0, currentTime());
    if (!seeking) {
      seek.max = String(duration || 1);
      seek.value = String(Math.min(duration || 1, t));
    }
    clock.textContent = `${t.toFixed(3)} s / ${duration.toFixed(3)} s`;
  }
  function updateButtons() {
    // Playback must never depend on whether exact stills are displayed.
    play.disabled = false;
    pause.disabled = false;
    rewind.disabled = false;
    seek.disabled = false;
    mode.textContent = exact()
      ? 'Videos: annähernd synchron · Einzelbilder: exakt und angeheftet.'
      : 'Videos: annähernd synchron · Einzelbilder: exakt.';
  }
  async function playBoth() {
    const l = loop?.checked ? limits() : null;
    if (l && (currentTime() < l.start || currentTime() >= l.end - 0.010)) moveBoth(l.start);
    before.playbackRate = after.playbackRate = Number(speed?.value) || 1;
    // At the exact frame boundary start, align positions before starting.
    try { after.currentTime = currentTime(); } catch (_) {}
    const results = await Promise.allSettled([before.play(), after.play()]);
    const rejected = results.find(x => x.status === 'rejected');
    if (rejected) {
      before.pause(); after.pause();
      mode.textContent = 'Wiedergabe fehlgeschlagen: ' + String(rejected.reason?.message || rejected.reason);
    }
    updateButtons();
  }
  function pauseBoth() {before.pause(); after.pause(); updateClock();}
  play.addEventListener('click', playBoth);
  pause.addEventListener('click', pauseBoth);
  rewind.addEventListener('click', () => {
    pauseBoth();
    moveBoth(loop?.checked && limits() ? limits().start : 0);
  });
  seek.addEventListener('input', () => {
    seeking = true;
    pauseBoth();
    moveBoth(Number(seek.value));
  });
  seek.addEventListener('change', () => {
    seeking = false;
    moveBoth(Number(seek.value));
    // Scrubbing the shared video timeline also updates the exact inspector.
    if (fps > 0 && frameCount > 0) {
      const frame = Math.max(0, Math.min(frameCount - 1, Math.floor(currentTime() * fps + 0.00001)));
      document.dispatchEvent(new CustomEvent('arp-request-exact-frame', {
        detail: {stage:'cleanup', frame}
      }));
    }
  });
  function checkLoop() {
    if (!loop?.checked || before.paused) return;
    const bounds = limits();
    if (bounds && currentTime() >= bounds.end - 0.015) moveBoth(bounds.start);
  }
  before.addEventListener('timeupdate', () => {
    updateClock();
    checkLoop();
    if (syncing || after.readyState < 1) return;
    const skew = Math.abs((after.currentTime || 0) - currentTime());
    if (skew > 0.08) {
      syncing = true;
      try { after.currentTime = currentTime(); } catch (_) {}
      syncing = false;
    }
  });
  before.addEventListener('seeked', () => {
    if (after.readyState < 1) return;
    try { after.currentTime = currentTime(); } catch (_) {}
  });
  before.addEventListener('ended', () => {
    if (!loop?.checked || !limits()) {after.pause(); return;}
    moveBoth(limits().start);
    playBoth();
  });
  after.addEventListener('ended', () => {
    // Only the original is the playback clock and decides loop bounds.
    if (!before.paused && after.readyState >= 1) {
      try {after.currentTime = currentTime(); after.play().catch(() => {});} catch (_) {}
    }
  });
  before.addEventListener('pause', () => {
    if (!after.paused) after.pause();
  });
  before.addEventListener('ratechange', () => { after.playbackRate = before.playbackRate; });
  for (const input of [loop, loopIn, loopOut]) input?.addEventListener('change', checkLoop);

  const open = inspector.querySelector('[data-fi-open]');
  const close = inspector.querySelector('[data-fi-close]');
  if (open) open.addEventListener('click', () => {
    pauseBoth();
  });
  if (close) close.addEventListener('click', () => {
    setTimeout(() => {
      if (fps > 0) moveBoth(currentTime());
      updateButtons();
    }, 0);
  });
  document.addEventListener('arp-frame-inspected', event => {
    if (!player.isConnected || event.detail?.stage !== 'cleanup') return;
    fps = Number(event.detail.fps) || fps;
    frameCount = Number(event.detail.frames) || frameCount;
    if (pendingVideoFrame !== null) applyFrameSeek(pendingVideoFrame);
    updateButtons();
    const report = (typeof state !== 'undefined' && state?.cleanup_diagnostics?.deflicker) || null;
    if (report && Number(report.frames) === frameCount && Array.isArray(report.samples)) {
      const selected = report.samples.find(x => Number(x.frame) === Number(event.detail.frame));
      if (selected) {
        const change = (Number(selected.gain) - 1) * 100;
        gain.textContent = `Frame ${Number(event.detail.frame)+1}: ${change>=0?'+':''}${change.toFixed(2)} % Helligkeitskorrektur`;
      } else {
        gain.textContent = 'Für diesen Frame ist kein genauer Korrekturwert gespeichert.';
      }
    }
  });
  // As in the old loop fix, obtain actual frame metadata from ARP's safe
  // stage-bound endpoint, rather than assuming 24/30 fps.
  fetch('/api/frame-inspector?stage=cleanup&frame=0', {cache:'no-store'})
    .then(r => r.json())
    .then(meta => {
      if (!player.isConnected) return;
      if (!meta.ok || !(Number(meta.fps)>0) || !(Number(meta.frames)>0)) throw new Error(meta.error || 'Ungültige Frame-Metadaten');
      fps = Number(meta.fps);
      frameCount = Number(meta.frames);
      if (pendingVideoFrame !== null) applyFrameSeek(pendingVideoFrame);
      if (loopIn) loopIn.max = String(frameCount);
      if (loopOut) {
        loopOut.max = String(frameCount);
        if (Number(loopOut.value) < 2 || Number(loopOut.value) > frameCount) loopOut.value = String(frameCount);
      }
    }).catch(err => {
      mode.textContent = 'Frame-Metadaten für den Loop nicht verfügbar: ' + err.message;
      if (loop) loop.disabled = true;
    });
  for (const video of [before,after]) video.addEventListener('loadedmetadata', updateClock);
  updateClock();
  updateButtons();
  setupCleanupExactWipe(player);
}

/* arp-exact-wipe-v1 */
/* Optional exact still-image wipe. Two-up video and still panels remain visible. */
function setupCleanupExactWipe(player) {
  if (!player || player.dataset.cleanupExactWipeReady) return;
  player.dataset.cleanupExactWipeReady = 'true';
  // Keep the exact Wipe available on demand without forcing another tall panel.
  const resultLabel = player.dataset.deflickerMethod === 'ffmpeg_median' ? 'FFmpeg Median'
    : player.dataset.deflickerMethod === 'ffmpeg_am' ? 'FFmpeg Arithmetic Mean'
    : player.dataset.deflickerMethod === 'localized_research' ? 'Localized · experimentell'
    : player.dataset.deflickerMethod === 'arp' ? 'ARP Conservative' : 'Clean-Up-Ergebnis';
  const fold = document.createElement('details');
  fold.className = 'cleanup-wipe-fold';
  fold.innerHTML = '<summary><strong>Wischvergleich</strong><span>Exakter Vorher/Nachher-Schieber · bei Bedarf öffnen</span></summary><section class="cleanup-exact-wipe"></section>';
  const card = fold.querySelector('.cleanup-exact-wipe');
  card.setAttribute('aria-label', 'Zusätzlicher Vorher-Nachher-Wischvergleich');
  card.innerHTML = `
    <div class="cleanup-exact-wipe-heading">
      <strong>Einzelbild: Wischvergleich</strong>
      <span>Dasselbe Einzelbild aus beiden Videos</span>
    </div>
    <p>Links <strong>ORIGINAL</strong>, rechts <strong>${resultLabel}</strong>.
      Ziehe die senkrechte Linie im Bild oder benutze den Schieber darunter.</p>
    <div class="cleanup-exact-wipe-image" data-wipe-picture
         aria-label="Wischvergleich von Original und bearbeitetem Einzelbild">
      <img data-wipe-before alt="Original, unberührter Frame">
      <img data-wipe-after alt="Derselbe Frame nach ${resultLabel}">
      <div class="cleanup-exact-wipe-divider" data-wipe-divider aria-hidden="true">
        <span class="cleanup-exact-wipe-handle"></span>
      </div>
    </div>
    <label class="cleanup-exact-wipe-control">Trennlinie
      <input data-wipe-position type="range" min="0" max="100" step="1"
             value="50" aria-label="Bildvergleich, links Original, rechts Ergebnis">
    </label>
    <output data-wipe-status aria-live="polite">50 % Original links · 50 % Ergebnis rechts</output>
    <p class="cleanup-exact-wipe-hint">Ganz links = nur Ergebnis. Ganz rechts = nur Original.
      Die beiden oberen Einzelbilder bleiben weiterhin nebeneinander sichtbar.</p>`;
  const inspector = player.parentElement.querySelector('.frame-inspector[data-fi-stage="cleanup"]');
  if (!inspector) return;
  inspector.appendChild(fold);
  // Wipe visibility is independent from video controls. Opening the details
  // never switches playback mode or changes the selected frame.
  const updateWipeVisibility = () => {
    const isInspecting = player.classList.contains('frame-inspector-active');
    fold.hidden = !isInspecting;
    card.classList.toggle('fi-visible', isInspecting);
  };
  inspector.querySelector('[data-fi-open]')?.addEventListener('click', () => setTimeout(updateWipeVisibility, 0));
  inspector.querySelector('[data-fi-close]')?.addEventListener('click', () => setTimeout(updateWipeVisibility, 0));
  const picture = card.querySelector('[data-wipe-picture]');
  const original = card.querySelector('[data-wipe-before]');
  const result = card.querySelector('[data-wipe-after]');
  const divider = card.querySelector('[data-wipe-divider]');
  const position = card.querySelector('[data-wipe-position]');
  const status = card.querySelector('[data-wipe-status]');

  function update() {
    const amount = Math.max(0, Math.min(100, Number(position.value) || 0));
    const zoom = Math.max(1, Number(getComputedStyle(player).getPropertyValue('--fi-zoom')) || 1);
    // Both pictures may be zoomed. Compensate the post-transform clip edge.
    const edge = 50 + (amount - 50) / zoom;
    result.style.clipPath = `inset(0 0 0 ${edge}%)`;
    divider.style.left = amount + '%';
    status.textContent = `${amount.toFixed(0)} % Original links · ${(100-amount).toFixed(0)} % Ergebnis rechts`;
  }
  function setFromPointer(event) {
    const bounds = picture.getBoundingClientRect();
    if (bounds.width <= 0) return;
    const amount = Math.max(0, Math.min(100, Math.round((event.clientX - bounds.left) / bounds.width * 100)));
    position.value = String(amount);
    update();
  }
  picture.addEventListener('pointerdown', event => {
    if (event.button !== 0 && event.pointerType === 'mouse') return;
    event.preventDefault();
    picture.setPointerCapture(event.pointerId);
    setFromPointer(event);
  });
  picture.addEventListener('pointermove', event => {
    if (picture.hasPointerCapture(event.pointerId)) setFromPointer(event);
  });
  picture.addEventListener('pointerup', event => {
    if (picture.hasPointerCapture(event.pointerId)) picture.releasePointerCapture(event.pointerId);
  });
  picture.addEventListener('pointercancel', event => {
    if (picture.hasPointerCapture(event.pointerId)) picture.releasePointerCapture(event.pointerId);
  });
  position.addEventListener('input', update);
  // The existing FFmpeg inspector emits this event only after both exact stills
  // have been decoded and inserted into the two-up panels.
  const onFrame = event => {
    if (!player.isConnected) {
      document.removeEventListener('arp-frame-inspected', onFrame);
      return;
    }
    if (event.detail?.stage !== 'cleanup') return;
    updateWipeVisibility();
    const srcBefore = inspector.querySelector('.fi-photo-before');
    const srcAfter = inspector.querySelector('.fi-photo-after');
    if (!srcBefore?.src || !srcAfter?.src) return;
    original.src = srcBefore.src;
    result.src = srcAfter.src;
    if (srcBefore.naturalWidth && srcBefore.naturalHeight) {
      picture.style.aspectRatio = `${srcBefore.naturalWidth} / ${srcBefore.naturalHeight}`;
    }
    update();
  };
  document.addEventListener('arp-frame-inspected', onFrame);
  const zoom = inspector.querySelector('[data-fi-zoom]');
  zoom?.addEventListener('change', update);
  updateWipeVisibility();
  update();
}
