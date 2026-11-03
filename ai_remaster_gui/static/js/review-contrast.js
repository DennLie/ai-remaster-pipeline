/* Shared read-only film-review visualization.
 * These are display-only colors. Never use them as measurement values, as a
 * trained defect detector, or to infer that the source has been changed.
 */
function comparisonContrastPixels(before, after, width, height, mode = 'direction', amplification = 32) {
  if (!(Number.isInteger(width) && width > 0 && Number.isInteger(height) && height > 0) ||
      before?.length !== width * height * 4 || after?.length !== before.length ||
      !['direction', 'rgb'].includes(mode) || ![8,32,64].includes(Number(amplification))) {
    throw new Error('Ungültige Differenzdaten oder Kontrasteinstellungen.');
  }
  const rgba = new Uint8ClampedArray(before.length);
  const boost = delta => delta === 0 ? 0 :
    Math.round(255 * Math.sqrt(Math.min(1, Math.abs(delta) * Number(amplification) / 255)));
  for (let p = 0; p < before.length; p += 4) {
    const dr = after[p] - before[p];
    const dg = after[p+1] - before[p+1];
    const db = after[p+2] - before[p+2];
    // Neutral graphite is clearly different from the strengthened deltas.
    // Exactly identical RGB pixels remain neutral, never falsely highlighted.
    let r = 15, g = 23, b = 31;
    if (mode === 'rgb') {
      if (dr !== 0 || dg !== 0 || db !== 0) {
        // Each output channel shows the absolute difference in that channel.
        r = boost(dr);
        g = boost(dg);
        b = boost(db);
      }
    } else {
      const signed = 0.2126*dr + 0.7152*dg + 0.0722*db;
      if (Math.abs(signed) > 1e-9) {
        const v = boost(signed);
        if (signed > 0) { r = Math.round(0.05*v); g = v; b = v; }
        else { r = v; g = Math.round(0.52*v); b = Math.round(0.06*v); }
      }
    }
    rgba[p] = r;
    rgba[p+1] = g;
    rgba[p+2] = b;
    rgba[p+3] = 255;
  }
  return rgba;
}

function comparisonContrastCaption(mode) {
  return mode === 'rgb'
    ? 'RGB = absoluter Kanalunterschied |ΔR|, |ΔG|, |ΔB| · Dunkelgrau = identische Pixel. Keine Defektmaske.'
    : 'Türkis = aufgehellt · Orange = abgedunkelt · Dunkelgrau = unverändert. Keine Defektmaske.';
}

/* In-browser focusable enlargement of genuine still images or difference
 * canvases; no extra server routes, re-encoding or edits to source files.
 * Clicking the enlarged picture again closes it. Escape works via dialog.
 */
const comparisonLightboxSelector =
  '.denoise-review-media img, .denoise-review-media canvas, ' +
  '.fi-exact-media img, .fi-exact-media canvas, .cleanup-source-still img';

function comparisonLightboxOpen(target) {
  if (!target || typeof document === 'undefined') return false;
  const isCanvas = target.tagName?.toLowerCase() === 'canvas';
  if (isCanvas && (!target.width || !target.height)) return false;
  if (!isCanvas && (!target.complete || !target.naturalWidth)) return false;
  let src;
  try {
    src = isCanvas ? target.toDataURL('image/png') : (target.currentSrc || target.src);
  } catch (_) {
    return false;
  }
  if (!src) return false;
  const existing = document.querySelector('.comparison-review-lightbox');
  if (existing) existing.close();
  const previousFocus = document.activeElement;
  const dialog = document.createElement('dialog');
  dialog.className = 'comparison-review-lightbox';
  dialog.setAttribute('aria-label', 'Vergrößerte Einzelbildansicht');
  dialog.innerHTML = `<div class="comparison-lightbox-head">
    <strong data-lightbox-title></strong>
    <button type="button" data-lightbox-close aria-label="Großansicht schließen">Schließen ×</button>
  </div>
  <img data-lightbox-image alt="Vergrößertes ausgewähltes Einzelbild" title="Erneut klicken zum Schließen">
  <p>Klick ins Bild oder auf Schließen: zurück zur Übersicht · Esc: schließen.</p>`;
  const name = target.closest('figure')?.querySelector('figcaption')?.textContent?.trim()
    || target.getAttribute('aria-label') || target.getAttribute('alt') || 'Einzelbild';
  dialog.querySelector('[data-lightbox-title]').textContent = name;
  dialog.querySelector('[data-lightbox-image]').src = src;
  dialog.addEventListener('click', event => {
    if (event.target === dialog || event.target.closest('[data-lightbox-close]') ||
        event.target === dialog.querySelector('[data-lightbox-image]')) dialog.close();
  });
  dialog.addEventListener('close', () => {
    dialog.remove();
    if (previousFocus?.isConnected && typeof previousFocus.focus === 'function') previousFocus.focus();
  }, {once:true});
  document.body.appendChild(dialog);
  dialog.showModal();
  dialog.querySelector('[data-lightbox-close]').focus();
  return true;
}

if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
  document.addEventListener('click', event => {
    const target = event.target?.closest?.(comparisonLightboxSelector);
    if (target) comparisonLightboxOpen(target);
  });
  document.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    const target = event.target?.closest?.(comparisonLightboxSelector);
    if (target) {
      event.preventDefault();
      comparisonLightboxOpen(target);
    }
  });
}
