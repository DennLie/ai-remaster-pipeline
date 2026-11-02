/* cleanup-two-up-v1 */
/* ARP Clean Up two-up review v1: permanent side-by-side presentation.
 * Browser playback is synchronized approximately; exact FFmpeg stills are
 * frame-index matched and rendered in the same two panels.
 * The original and the processed file are never modified by this viewer.
 */
function setupCleanupReview(sliderId, player, before, after, slider) {
  if (sliderId !== 'cleanupCompareSlider' || player.dataset.cleanupTwoUpReady) return;
  const inspector = player.parentElement?.querySelector('.frame-inspector[data-fi-stage="cleanup"]');
  if (!inspector) return;
  player.dataset.cleanupTwoUpReady = 'true';
  player.classList.add('cleanup-two-up');
  player.closest('.editor-viewer')?.classList.add('cleanup-two-up-wide');
  player.querySelectorAll('.cleanup-compare-label').forEach(node => node.remove());
  slider.hidden = true;
  slider.setAttribute('aria-hidden', 'true');

  // Move the *existing* video elements. No new video URL or second set of
  // decoders, and the pre-existing ARP stage binding retains its references.
  const layout = document.createElement('div');
  layout.className = 'cleanup-two-up-grid';
  layout.innerHTML = `
    <div class="cleanup-side" data-compare-panel="before">
      <div class="cleanup-side-heading"><strong>ORIGINAL</strong><span>Vorher · unverändert</span></div>
      <div class="cleanup-side-media" data-compare-side="before"></div>
    </div>
    <div class="cleanup-side" data-compare-panel="after">
      <div class="cleanup-side-heading"><strong>LIGHT DEFLICKER</strong><span>Nachher · bearbeitet</span></div>
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
      <button type="button" data-two-play aria-label="Beide Videos abspielen">▶ Beide abspielen</button>
      <button type="button" data-two-pause>⏸ Pause</button>
      <button type="button" data-two-rewind>⟲ Anfang</button>
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
      ? 'Beide Videos sind abspielbar. Die Einzelbilder darunter bleiben auf dem ausgewählten Frame stehen (exakt per FFmpeg).'
      : 'Beide Videos gleichzeitig sichtbar. Gemeinsame Wiedergabe ist näherungsweise synchron; exakte Unterschiede im Frame Inspector beurteilen.';
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
    <p>Links <strong>ORIGINAL</strong>, rechts <strong>LIGHT DEFLICKER</strong>.
      Ziehe die senkrechte Linie im Bild oder benutze den Schieber darunter.</p>
    <div class="cleanup-exact-wipe-image" data-wipe-picture
         aria-label="Wischvergleich von Original und bearbeitetem Einzelbild">
      <img data-wipe-before alt="Original, unberührter Frame">
      <img data-wipe-after alt="Derselbe Frame nach Light Deflicker">
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
