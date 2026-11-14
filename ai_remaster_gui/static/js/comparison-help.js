/* Read-only contextual help. Mount again after rendering; no processing state is read or written. */
const comparisonHelpMethods = Object.freeze({
  zoom0: 'FFmpeg/libvidstab with no additional fixed zoom. Motion correction can still expose borders and clip source content. Bicubic resampling changes pixels; inspect pans, grain and frame edges.',
  zoom3: 'The same FFmpeg/libvidstab motion settings, with 3% additional fixed zoom to conceal transformed edges. Crops the field of view and does not guarantee border-free output.',
  current: 'Uses the current pipeline stabilization parameters, including any custom zoom. This can match the 0% or 3% variant; matching renders are reused.',
  hqdn3d: 'HQDN3D reduces noise across space and time. Inspect moving edges for trails and fine grain for smoothing.',
  nlmeans: 'NLMeans averages similar image patches. Inspect fine texture and grain for loss of detail.',
  bm3d: 'BM3D groups similar patches for noise reduction. Inspect fine textures and edges against the input.',
  arp: 'ARP Conservative limits brightness correction. Inspect cuts, fades and intentional lighting changes.',
  ffmpeg_median: 'FFmpeg Median uses a temporal median brightness reference. The window is measured in frames.',
  ffmpeg_am: 'FFmpeg Arithmetic Mean uses an average temporal brightness reference. Inspect fades and cuts.',
  localized_research: 'Localized short experiment corrects regional brightness in one confirmed shot. Limited to 240 frames and 75 million total pixels.',
  localized_stream: 'Localized film streaming uses a bounded five-frame buffer. Scene and fade detection can fail; review the full output.',
  unsharp: 'Unsharp increases edge contrast using a blurred reference. Inspect halos and amplified grain.',
  cas: 'CAS applies contrast-adaptive sharpening at native resolution. Sharper edges do not prove recovered detail.',
  multiscale: 'Multiscale enhances detail at several spatial scales. Inspect halos and amplified texture.',
  richardson_lucy: 'Richardson–Lucy deconvolution assumes Gaussian blur, not a measured film blur. Inspect ringing and grain.',
  nafnet: 'NAFNet uses a GoPro-trained image deblurring model. Inspect faces, text, grain and frame-to-frame consistency.',
  restormer: 'Restormer uses a GoPro-trained image deblurring model. Inspect faces, text, grain and frame-to-frame consistency.',
});
// Qualitative cost of ARP's implementations, not measured runtimes or quality scores.
const comparisonEffort = Object.freeze({
  zoom0: ['medium','CPU','Two-pass motion analysis and transformation per scene; resolution, film length and scene count drive work.'],
  zoom3: ['medium','CPU','The same two-pass analysis as 0%, with fixed zoom during transformation.'],
  current: ['medium','CPU','Two-pass libvidstab at the current motion and zoom settings.'],
  hqdn3d: ['low','CPU','Simple spatial/temporal filtering; resolution and film length drive total work.'],
  nlmeans: ['high','CPU','Searching similar patches is expensive; resolution and the search region drive work.'],
  bm3d: ['high','CPU','Block matching and grouped filtering are expensive, even with the basic preset implementation.'],
  arp: ['low','CPU','Global brightness analysis and correction; decoding and encoding can dominate.'],
  ffmpeg_median: ['low','CPU','Temporal brightness statistics; a wider frame window adds work and buffering.'],
  ffmpeg_am: ['low','CPU','Temporal brightness statistics; a wider frame window adds work and buffering.'],
  localized_research: ['medium','CPU','Regional analysis across a short shot; frame count and pixel count increase memory and work.'],
  localized_stream: ['medium','CPU','Regional analysis with a bounded five-frame buffer; film length still increases total work.'],
  unsharp: ['low','CPU','One FFmpeg sharpening filter; resolution and film length drive work.'],
  cas: ['low','CPU','FFmpeg contrast-adaptive sharpening at native resolution; no neural model.'],
  multiscale: ['medium','CPU','Two sequential sharpening radii for small and broader structures; no synthesis network.'],
  richardson_lucy: ['high','CPU','Iterative deconvolution; stronger presets use more iterations. Test a short section first.'],
  nafnet: ['high','CUDA GPU','Native-resolution FP32 inference per frame. Prepared weights required; resolution strongly affects VRAM. No CPU fallback.'],
  restormer: ['high','CUDA GPU','Native-resolution FP32 inference per frame. Prepared weights required; resolution strongly affects VRAM. No CPU fallback.'],
});
function comparisonMethodHelp(method) {
  const effort=comparisonEffort[method];
  return (comparisonHelpMethods[method] || '')+(effort ? ` Effort: ${effort[0]} · ${effort[1]}. ${effort[2]} Qualitative estimate, not a benchmark or quality ranking. Total time also depends on hardware, settings and delivery encoding.` : '');
}
const comparisonHelpFields = Object.freeze({
  smoothing: 'Motion smoothing radius in frames. Inspect intentional pans and transitions; stronger smoothing is not automatically more faithful.',
  max_shift: 'Maximum correction in pixels. IMPORTANT: 0 means unlimited correction, not disabled translation.',
  max_angle: 'Maximum rotation correction in degrees. IMPORTANT: 0 means unlimited correction, not disabled rotation.',
  zoom: 'Additional fixed zoom for the current pipeline settings. The 0% and 3% comparison variants override only this value. No zoom does not guarantee no image loss.',
  scene_aware: 'Restart motion analysis at detected hard cuts. Turn off only for a genuinely continuous shot; cut detection can fail.',
  shot_threshold: 'Hard-cut detection threshold. Inspect the boundaries; image-content changes can be mistaken for cuts.',
  min_shot_seconds: 'Minimum detected scene duration in seconds. Short cuts may be missed; each accepted boundary resets smoothing.',
  preset: 'Gentle, Balanced and Strong are method-specific strengths, not comparable measurements. Enhancement presets affect classical methods only.',
  window: 'Temporal window in frames for the FFmpeg deflicker methods. Inspect transitions and deliberate lighting changes.',
  strength: 'Correction strength for ARP Conservative. Higher values apply more of the estimated brightness correction.',
  max_correction: 'Maximum brightness correction for ARP Conservative. This limits how far correction can move the picture.',
  mix: 'Blends the enhanced picture with the stage input: 0% is input, 100% is the enhanced result. Applies to every enhancement method.',
  force: 'Regenerate instead of reusing matching cached output. This is a processing option, not a comparison selection.',
  dry_run: 'Show the stage command without running the processing step.',
  localized_confirm_single_shot: 'Required acknowledgement for the short localized experiment. Keep the original warning visible and check the shot before confirming.',
  localized_confirm_film_experiment: 'Required acknowledgement for localized streaming. Scene and fade detection are fallible; review the entire result.',
});
const comparisonHelpControls = Object.freeze({
  '#denoiseComparePlay': 'Play or pause the comparison views together. Playback uses compressed display frames; use Exact still to inspect pixels without lossy preview compression.',
  '#denoiseCompareSeek': 'Move the shared comparison position. This does not trim the source or change pipeline settings.',
  '[data-dp-jump]': 'Jump to a position in seconds on the shared comparison timeline.',
  '[data-zoom-native]': '1:1 maps one decoded source pixel to one physical display pixel. This is not the same as 100% of Fit.',
  '[data-zoom-percent]': 'Zoom relative to Fit: 100% fits the whole image. Display zoom does not resize the processed output.',
  '[data-zoom-range]': 'Change display magnification around Fit. Alt+wheel zooms the image; ordinary wheel scrolls. Drag to pan. Ctrl/Meta+wheel stays with browser zoom. Processing resolution is unchanged.',
  '[data-zoom-grid]': 'Overlay a pixel grid for close inspection. The grid changes the display only.',
  '[data-ds-exact]': 'Pause and decode the shared frame without lossy preview compression. This does not change the processed output.',
  '#denoiseUseVariant': 'Adopt the inspected variant for the pipeline. Comparison membership alone does not adopt a result.',
  '[data-dc-focus]': 'Inspect this result. Inspecting a result does not adopt it for the pipeline.',
  '[data-dr-chart]': 'Luminance changes per frame, not a defect mask. Long films are sampled; click a bar to move the shared position.',
});

const comparisonHelp = (() => {
  const content = new WeakMap();
  const decorated = new WeakSet();
  let installed = false, tooltip, anchor, pending, pinned = false, timer, dismissed;
  const tooltipId = 'comparison-context-help';
  function describe(element, add) {
    const ids = (element.getAttribute('aria-describedby') || '').split(/\s+/).filter(id => id && id !== tooltipId);
    if (add) ids.push(tooltipId);
    if (ids.length) element.setAttribute('aria-describedby', ids.join(' '));
    else element.removeAttribute('aria-describedby');
  }
  function hide() {
    clearTimeout(timer);
    pending = null;
    if (anchor) {
      describe(anchor, false);
      if (anchor.classList.contains('comparison-help-info')) anchor.setAttribute('aria-expanded', 'false');
    }
    anchor = null;
    pinned = false;
    if (tooltip) tooltip.hidden = true;
  }
  function position() {
    if (!anchor) return;
    if (!anchor.isConnected) return hide();
    const viewport = window.visualViewport;
    const left = viewport?.offsetLeft || 0, top = viewport?.offsetTop || 0;
    const width = viewport?.width || window.innerWidth, height = viewport?.height || window.innerHeight;
    tooltip.style.maxWidth = `min(34ch, ${Math.max(0, width - 16)}px)`;
    tooltip.style.maxHeight = `${Math.max(0, height - 16)}px`;
    const box = anchor.getBoundingClientRect(), tip = tooltip.getBoundingClientRect();
    const group = anchor.closest('.comparison-noise-method') || anchor.closest('.comparison-method-row') || anchor.closest('.comparison-param-row');
    const side = group?.getBoundingClientRect() || box;
    const right = side.left + side.width + 8;
    const x = right + tip.width <= left + width - 8 ? right : side.left - tip.width - 8;
    tooltip.style.left = `${Math.max(left + 8, Math.min(x, left + width - tip.width - 8))}px`;
    tooltip.style.top = `${Math.max(top + 8, Math.min(box.top, top + height - tip.height - 8))}px`;
  }
  function show(element) {
    clearTimeout(timer);
    pending = null;
    if (!element.isConnected) return hide();
    if (dismissed === element || (pinned && anchor !== element)) return;
    if (anchor !== element) hide();
    const host = element.closest('dialog[open]') || document.body;
    if (tooltip.parentElement !== host) host.append(tooltip);
    anchor = element;
    tooltip.textContent = content.get(element);
    tooltip.hidden = false;
    describe(element, true);
    position();
  }
  function target(node) {
    for (let element = node; element; element = element.parentElement) {
      if (content.has(element)) return element;
    }
    return null;
  }
  function leave(event) {
    const element = target(event.target);
    if (element && element.contains(event.relatedTarget)) return;
    if (dismissed === element) dismissed = null;
    if (pinned || tooltip.contains(event.relatedTarget) || anchor?.contains(event.relatedTarget)) return;
    clearTimeout(timer);
    pending = null;
    timer = setTimeout(() => {
      if (!anchor?.contains(document.activeElement)) hide();
    }, 160);
  }
  function install() {
    if (installed) return;
    installed = true;
    tooltip = document.createElement('div');
    tooltip.id = tooltipId;
    tooltip.className = 'comparison-help-tooltip';
    tooltip.setAttribute('role', 'tooltip');
    tooltip.hidden = true;
    document.body.append(tooltip);
    document.addEventListener('pointerover', event => {
      if (event.pointerType === 'touch') return;
      if (tooltip.contains(event.target)) return clearTimeout(timer);
      const element = target(event.target);
      if (!element || element === anchor || element === pending || pinned) return;
      hide();
      pending = element;
      timer = setTimeout(() => show(element), 3000);
    });
    document.addEventListener('pointerout', event => {
      if (event.pointerType !== 'touch') leave(event);
    });
    document.addEventListener('focusin', event => {
      const element = target(event.target);
      if(element && element!==anchor) pinned=false;
      if (element) show(element);
    });
    document.addEventListener('focusout', leave);
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && (anchor || pending)) {
        event.preventDefault();
        dismissed = anchor || pending;
        hide();
      }
    });
    // Capture only our own buttons: no inline selection or pipeline handler can run.
    document.addEventListener('click', event => {
      const element = target(event.target);
      if (!element?.classList.contains('comparison-help-info')) {
        if (!tooltip.contains(event.target)) hide();
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      if (anchor === element && pinned) { dismissed = element; hide(); return; }
      hide();
      dismissed = null;
      show(element);
      pinned = true;
      element.setAttribute('aria-expanded', 'true');
    }, true);
    window.addEventListener('resize', position);
    document.addEventListener('scroll', position, true);
    window.visualViewport?.addEventListener('resize', position);
    window.visualViewport?.addEventListener('scroll', position);
  }
  function info(element, text, label, location) {
    if (decorated.has(element)) return;
    decorated.add(element);
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'comparison-help-info';
    button.setAttribute('aria-label', `Help: ${label}`);
    button.setAttribute('aria-expanded', 'false');
    button.innerHTML = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="10" cy="10" r="7.5"/><path d="M10 9v5M10 5.5v1.5"/></svg>';
    content.set(button, text);
    location(button);
  }
  function mount(root, compact=false) {
    install();
    hide();
    if(compact) {
      root.querySelectorAll('[data-inspect-method], [data-effort-method]').forEach(element=>{
        content.set(element,comparisonMethodHelp(element.dataset.inspectMethod || element.dataset.effortMethod));
      });
      root.querySelectorAll('[data-field]').forEach(field=>{
        if(field.type==='hidden' || !comparisonHelpFields[field.dataset.field]) return;
        const text=comparisonHelpFields[field.dataset.field];
        content.set(field,text);
        const label=field.closest('label') || (field.previousElementSibling?.tagName==='LABEL' ? field.previousElementSibling : null);
        if(label) content.set(label,text);
      });
      return;
    }
    root.querySelectorAll('[data-denoise-cell][data-method]').forEach(cell => {
      const text = comparisonHelpMethods[cell.dataset.method];
      if (!text) return;
      content.set(cell, text);
      if (cell.closest('.denoise-variant-grid')) return;
      info(cell, text, cell.querySelector('strong')?.textContent || cell.dataset.method, button => {
        const row = document.createElement('div');
        row.className = 'comparison-help-method-row';
        cell.before(row);
        row.append(cell, button);
      });
    });
    root.querySelectorAll('.denoise-variant-col').forEach(heading => {
      const method = heading.textContent.trim().toLowerCase();
      const text = comparisonHelpMethods[method];
      if (text) info(heading, text, heading.textContent.trim(), button => heading.append(button));
    });
    root.querySelectorAll('[data-field]').forEach(field => {
      if (field.type === 'hidden') return;
      const text = comparisonHelpFields[field.dataset.field];
      if (!text) return;
      content.set(field, text);
      const label = field.closest('label');
      const name = field.getAttribute('aria-label') || label?.textContent.trim() || field.dataset.field.replaceAll('_', ' ');
      info(field, text, name, button => (label || field).after(button));
    });
    for (const [selector, text] of Object.entries(comparisonHelpControls)) {
      root.querySelectorAll(selector).forEach(element => content.set(element, text));
    }
  }
  return {mount};
})();

function mountComparisonHelp(root = document, compact=false) {
  comparisonHelp.mount(root,compact);
}
