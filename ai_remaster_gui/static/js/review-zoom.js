/* Shared, normalized crop navigation. Fit refers to the image, not its letterbox. */
function reviewZoomClamp(value, min, max) {
  const number = Number(value);
  return Math.max(min, Math.min(max, Number.isFinite(number) ? number : min));
}

function reviewZoomGeometry(width, height, imageWidth, imageHeight) {
  if (!(width > 0 && height > 0 && imageWidth > 0 && imageHeight > 0)) return null;
  const scale = Math.min(width / imageWidth, height / imageHeight);
  return {width: imageWidth * scale, height: imageHeight * scale};
}

function reviewZoomAnchor(pan, oldZoom, newZoom, point) {
  // point is normalized to the fitted image, with its center at zero.
  if (newZoom <= 1) return 0;
  return reviewZoomClamp(((oldZoom - 1) * pan * newZoom / oldZoom +
    2 * point * (1 - newZoom / oldZoom)) / (newZoom - 1), -1, 1);
}

function reviewZoomNativeScale(imageWidth, fittedWidth, pixelRatio = 1) {
  return imageWidth > 0 && fittedWidth > 0 && pixelRatio > 0 ? imageWidth / (fittedWidth * pixelRatio) : null;
}

function reviewZoomControlsHtml(id, setter, fit, exact = false) {
  return `<div class="review-zoom-controls" id="${id}" role="group" aria-label="Comparison display zoom">
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6"/></svg>
    <input data-zoom-range type="range" min="-1000" max="1000" step="1" value="0" aria-label="Comparison zoom" oninput="${setter}(Math.pow(32,Number(this.value)/1000))">
    <label><input data-zoom-percent type="number" min="3.125" max="3200" step="0.1" value="100" aria-label="Zoom percent relative to fit" title="100% fits the complete image; use 1:1 for native display pixels" onchange="${setter}(Number(this.value)/100)"><span>%</span></label>
    <button type="button" onclick="${fit}()" title="Reset zoom and pan">Fit</button>
    <button type="button" data-zoom-native aria-pressed="false" title="One decoded source pixel per physical display pixel">1:1</button>
    ${exact ? '<label class="review-grid-toggle"><input data-zoom-grid type="checkbox">Pixel grid</label>' : ''}
  </div>`;
}

function createReviewZoom(host, selector, controls, state, changed) {
  let drag = null, suppressClick = false;
  const abort = new AbortController();
  const options = {signal: abort.signal};
  function nativeScale() {
    const focused = typeof document !== 'undefined' ? document.activeElement?.closest?.(selector) : null;
    const wrap = focused && host.contains?.(focused) ? focused : [...host.querySelectorAll(selector)]
      .find(wrap => wrap.clientWidth > 0 && wrap.clientHeight > 0);
    const media = wrap?.querySelector('video, img, canvas');
    const iw = media?.videoWidth || media?.naturalWidth || media?.width;
    const ih = media?.videoHeight || media?.naturalHeight || media?.height;
    const fitted = wrap && reviewZoomGeometry(wrap.clientWidth,wrap.clientHeight,iw,ih);
    return fitted ? reviewZoomNativeScale(iw,fitted.width,typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1) : null;
  }
  function apply() {
    const native = nativeScale();
    if (state.native && native !== null) state.zoom = native;
    state.zoom = reviewZoomClamp(state.zoom, 1/32, 32);
    state.x = state.zoom <= 1 ? 0 : reviewZoomClamp(state.x, -1, 1);
    state.y = state.zoom <= 1 ? 0 : reviewZoomClamp(state.y, -1, 1);
    host.querySelectorAll(selector).forEach(wrap => {
      const media = wrap.querySelector('video, img, canvas');
      if (!media) return;
      const iw = media.videoWidth || media.naturalWidth || media.width;
      const ih = media.videoHeight || media.naturalHeight || media.height;
      const size = reviewZoomGeometry(wrap.clientWidth, wrap.clientHeight, iw, ih);
      if (!size) return;
      let window = wrap.querySelector('.review-zoom-window');
      if (!window) {
        window = document.createElement('div');
        window.className = 'review-zoom-window';
        const plane = document.createElement('div');
        plane.className = 'review-zoom-plane';
        plane.appendChild(media);
        const grid = document.createElement('div');
        grid.className = 'review-pixel-grid';
        grid.setAttribute('aria-hidden', 'true');
        plane.appendChild(grid);
        window.appendChild(plane);
        wrap.appendChild(window);
      }
      window.style.width = `${size.width}px`;
      window.style.height = `${size.height}px`;
      const plane = window.firstElementChild;
      const pixelRatio = globalThis.devicePixelRatio || 1;
      const zoom = state.native ? reviewZoomNativeScale(iw,size.width,pixelRatio) : state.zoom;
      let x = state.x * size.width * (zoom - 1) / 2;
      let y = state.y * size.height * (zoom - 1) / 2;
      if (state.native) {
        const rect = window.getBoundingClientRect();
        const left = rect.left + size.width * (1-zoom)/2 + x;
        const top = rect.top + size.height * (1-zoom)/2 + y;
        x += Math.round(left*pixelRatio)/pixelRatio-left;
        y += Math.round(top*pixelRatio)/pixelRatio-top;
      }
      plane.style.transform = `translate(${x}px, ${y}px) scale(${zoom})`;
      wrap.classList.toggle('is-zoomed', zoom > 1);
      wrap.classList.toggle('is-pixel-zoom', state.native || size.width * zoom / iw >= 4);
      wrap.tabIndex = 0;
      wrap.setAttribute('aria-label', 'Image viewport. Scroll to zoom; drag or use arrow keys to pan when zoomed.');
      wrap.title = state.zoom > 1 ? 'Drag image to pan · Scroll to zoom · Double-click to fit' : 'Scroll to zoom';
      const grid = plane.lastElementChild;
      grid.hidden = !state.grid || size.width * zoom / iw < 8;
      grid.style.backgroundSize = `${size.width / iw}px ${size.height / ih}px`;
      grid.style.setProperty('--pixel-line', `${1 / zoom}px`);
    });
    const range = controls?.querySelector('[data-zoom-range]');
    const percent = controls?.querySelector('[data-zoom-percent]');
    if (range) {
      range.value = String(1000 * Math.log(state.zoom) / Math.log(32));
      range.setAttribute('aria-valuetext', `${Math.round(state.zoom * 100)} percent`);
    }
    if (percent) percent.value = String(Math.round(state.zoom * 100));
    const checkbox = controls?.querySelector('[data-zoom-grid]');
    if (checkbox) checkbox.checked = Boolean(state.grid);
    const nativeButton = controls?.querySelector('[data-zoom-native]');
    if (nativeButton) {
      nativeButton.disabled = native === null || native < 1/32 || native > 32;
      nativeButton.setAttribute('aria-pressed',String(Boolean(state.native)));
    }
    changed?.(state);
  }
  function setZoom(value, point = {x:0, y:0}) {
    state.native = false;
    const next = reviewZoomClamp(value, 1/32, 32);
    state.x = reviewZoomAnchor(state.x, state.zoom, next, point.x);
    state.y = reviewZoomAnchor(state.y, state.zoom, next, point.y);
    state.zoom = next;
    apply();
  }
  function fit() { state.native = false; state.x = 0; state.y = 0; state.zoom = 1; apply(); }
  function native() {
    const scale = nativeScale();
    if (scale === null || scale < 1/32 || scale > 32) return;
    state.native = true; apply();
  }
  host.addEventListener('wheel', event => {
    if (event.ctrlKey || event.metaKey) return; // Preserve browser accessibility zoom.
    const wrap = event.target.closest(selector);
    const window = wrap?.querySelector('.review-zoom-window');
    if (!window) return;
    event.preventDefault();
    const rect = window.getBoundingClientRect();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? rect.height : 1;
    setZoom(state.zoom * Math.exp(-reviewZoomClamp(event.deltaY * unit, -200, 200) * .003), {
      x: reviewZoomClamp((event.clientX - rect.left) / rect.width - .5, -.5, .5),
      y: reviewZoomClamp((event.clientY - rect.top) / rect.height - .5, -.5, .5),
    });
  }, {...options, passive:false});
  host.addEventListener('pointerdown', event => {
    if (state.zoom <= 1 || drag || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const wrap = event.target.closest(selector);
    const window = wrap?.querySelector('.review-zoom-window');
    if (!window) return;
    const rect = window.getBoundingClientRect();
    try { host.setPointerCapture(event.pointerId); } catch (_) { return; }
    drag = {id:event.pointerId, wrap, startX:event.clientX, startY:event.clientY,
      x:state.x, y:state.y, width:rect.width, height:rect.height};
    suppressClick = false;
    wrap.classList.add('is-panning');
    event.preventDefault();
  }, options);
  host.addEventListener('pointermove', event => {
    if (!drag || event.pointerId !== drag.id) return;
    const dx = event.clientX - drag.startX, dy = event.clientY - drag.startY;
    if (Math.hypot(dx, dy) > 3) suppressClick = true;
    state.x = drag.x + dx * 2 / (drag.width * (state.zoom - 1));
    state.y = drag.y + dy * 2 / (drag.height * (state.zoom - 1));
    apply();
    event.preventDefault();
  }, options);
  function end(event) {
    if (!drag || (event.pointerId != null && event.pointerId !== drag.id)) return;
    drag.wrap.classList.remove('is-panning');
    const id = drag.id;
    drag = null;
    if (host.hasPointerCapture(id)) host.releasePointerCapture(id);
  }
  for (const name of ['pointerup','pointercancel','lostpointercapture']) host.addEventListener(name, end, options);
  if (typeof window !== 'undefined') window.addEventListener('blur', end, options);
  host.addEventListener('click', event => {
    const image = event.target.closest(selector);
    if (image && (suppressClick || state.zoom > 1)) { event.preventDefault(); event.stopPropagation(); }
    suppressClick = false;
  }, {...options, capture:true});
  host.addEventListener('keydown', event => {
    if (state.zoom <= 1 || !event.target.closest(selector)) return;
    const directions = {ArrowLeft:[1,0], ArrowRight:[-1,0], ArrowUp:[0,1], ArrowDown:[0,-1]};
    const direction = directions[event.key];
    if (!direction) return;
    event.preventDefault();
    state.x += direction[0] * .1 / (state.zoom - 1);
    state.y += direction[1] * .1 / (state.zoom - 1);
    apply();
  }, options);
  host.addEventListener('dblclick', event => { if (event.target.closest(selector)) fit(); }, options);
  host.addEventListener('dragstart', event => { if (event.target.closest(selector)) event.preventDefault(); }, options);
  host.addEventListener('load', apply, {...options, capture:true});
  host.addEventListener('loadedmetadata', apply, {...options, capture:true});
  controls?.querySelector('[data-zoom-grid]')?.addEventListener('change', event => {
    state.grid = event.target.checked; apply();
  }, options);
  controls?.querySelector('[data-zoom-native]')?.addEventListener('click',native,options);
  const resize = new ResizeObserver(() => { if (host.isConnected) apply(); else resize.disconnect(); });
  resize.observe(host);
  apply();
  return {host, state, apply, setZoom, fit, native, destroy(){abort.abort(); resize.disconnect();}};
}
