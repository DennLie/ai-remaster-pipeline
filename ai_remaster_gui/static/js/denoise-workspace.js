/* Matrix-owned comparison membership. Slot IDs are stable across selection changes. */
let denoiseWorkspaceVariants = DENOISE_PRESETS.flatMap(preset =>
  DENOISE_METHODS.map(method => ({method:method.key, preset:preset.key})));
let denoiseWorkspaceSelection = null;
let denoiseWorkspaceSource = '';
let denoiseWorkspaceObserver = null;
let denoiseWorkspaceHost = null;
let denoiseWorkspaceGenerationDismissed = false;
let denoiseWorkspaceGenerationPending = false;
let denoiseWorkspaceGenerationError = '';
let denoiseWorkspacePlayback = null;
let denoiseWorkspaceStill = null;
let denoiseWorkspaceExact = false;
let denoiseWorkspaceFrame = 0;
const denoiseWorkspaceViews = new Map();
const denoiseWorkspaceWipes = new Map();
let denoiseWorkspaceStage = 'denoise';
const denoiseWorkspaceSaved = new Map();

function denoiseWorkspaceDeactivate() {
  if (!denoiseWorkspaceStage) return;
  denoiseWorkspaceSaved.set(denoiseWorkspaceStage,{selection:denoiseWorkspaceSelection,source:denoiseWorkspaceSource,
    selected:denoiseMatrixSelected,frame:denoiseWorkspaceFrame,views:new Map(denoiseWorkspaceViews),wipes:new Map(denoiseWorkspaceWipes),
    zoom:denoiseCompareZoom,pan:{...denoiseComparePan},native:denoiseCompareNative,exact:denoiseWorkspaceExact,grid:denoiseCompareGrid,
    diffMode:denoiseCompareDiffMode,diffContrast:denoiseCompareDiffContrast});
  denoiseWorkspaceStill?.destroy();denoiseWorkspaceStill=null;
  denoiseWorkspacePlayback?.destroy();denoiseWorkspacePlayback=null;
  denoiseWorkspaceObserver?.disconnect();denoiseWorkspaceObserver=null;denoiseWorkspaceHost=null;
  denoiseCompareZoomView?.destroy();denoiseCompareZoomView=null;
  denoiseWorkspaceStage='';
}

function denoiseWorkspaceActivate(key) {
  if(!['denoise','deflicker'].includes(key) || key===denoiseWorkspaceStage) return;
  denoiseWorkspaceDeactivate();
  const saved=denoiseWorkspaceSaved.get(key);
  denoiseWorkspaceStage=key;
  denoiseWorkspaceVariants=denoisePresets().flatMap(preset=>denoiseMethods().map(method=>({method:method.key,preset:preset.key})));
  denoiseWorkspaceSelection=saved?.selection ?? null;denoiseWorkspaceSource=saved?.source || '';
  denoiseMatrixSelected=saved?.selected || null;denoiseWorkspaceFrame=saved?.frame || 0;
  denoiseCompareZoom=saved?.zoom || 1;denoiseComparePan=saved?.pan || {x:0,y:0};
  denoiseCompareNative=Boolean(saved?.native);denoiseWorkspaceExact=Boolean(saved?.exact);
  denoiseCompareGrid=Boolean(saved?.grid);
  denoiseCompareDiffMode=saved?.diffMode || 'direction';denoiseCompareDiffContrast=saved?.diffContrast || 32;
  denoiseWorkspaceViews.clear();denoiseWorkspaceWipes.clear();
  for(const [slot,view] of saved?.views || []) denoiseWorkspaceViews.set(slot,view);
  for(const [slot,wipe] of saved?.wipes || []) denoiseWorkspaceWipes.set(slot,wipe);
  denoiseWorkspaceGenerationDismissed=false;denoiseWorkspaceGenerationError='';
  denoiseReviewFps=0;
}

function denoiseWorkspaceKey(variant) { return variant.method + '/' + variant.preset; }
function denoiseWorkspaceHas(method, preset) {
  return Boolean(denoiseWorkspaceSelection?.has(denoiseWorkspaceKey({method,preset})));
}
function denoiseWorkspaceSelected() {
  return denoiseWorkspaceVariants.filter(v => denoiseWorkspaceHas(v.method,v.preset));
}
function denoiseWorkspaceMissing() {
  return denoiseWorkspaceSelected().filter(v => !denoiseVariantReady(v.method,v.preset));
}

function denoiseWorkspaceInitialize(candidates) {
  denoiseWorkspaceActivate(comparisonStageKey());
  const source = candidates.find(v => v.kind === 'source')?.path || '';
  const identity = [source, state?.settings?.global?.source, denoiseCompareSectionKey()].join('|');
  if (denoiseWorkspaceSelection === null) {
    const stage = denoiseStageVariant();
    const ready = denoiseWorkspaceVariants.filter(v => denoiseVariantReady(v.method,v.preset));
    const initial = stage.method === 'off' ? ready.slice(0,2)
      : [stage, ...ready.filter(v => !denoiseSameVariant(v,stage)).slice(0,1)];
    denoiseWorkspaceSelection = new Set(initial.map(denoiseWorkspaceKey));
    denoiseMatrixSelected = initial[0] || {method:'off',preset:''};
  }
  if (identity !== denoiseWorkspaceSource) {
    denoiseWorkspaceSource = identity;
    denoiseWorkspaceGenerationDismissed = false;
    denoiseWorkspaceGenerationError = '';
    denoiseCompareStart = 0;
    denoiseCompareZoom = 1;
    denoiseCompareNative = false;
    denoiseWorkspaceFrame = 0;
    denoiseComparePan = {x:0,y:0};
  }
  denoiseComparePaths = [source, ...denoiseWorkspaceVariants.map((v,index) => {
    denoiseCompareDiffState[index+1] ||= {enabled:false,url:'',key:'',request:0,loading:false};
    return denoiseWorkspaceHas(v.method,v.preset) && denoiseVariantReady(v.method,v.preset)
      ? denoiseVariantPath(v.method,v.preset) : '';
  })];
}

function denoiseWorkspaceToggle(method,preset) {
  const key = denoiseWorkspaceKey({method,preset});
  denoiseWorkspaceSelection ||= new Set();
  if (denoiseWorkspaceSelection.has(key)) {
    denoiseWorkspaceSelection.delete(key);
    if (denoiseSameVariant(denoiseSelectedVariant(),{method,preset}))
      denoiseMatrixSelected = denoiseWorkspaceSelected()[0] || {method:'off',preset:''};
  } else {
    denoiseWorkspaceSelection.add(key);
    denoiseMatrixSelected = {method,preset};
  }
  denoiseWorkspaceGenerationDismissed = false;
  denoiseWorkspaceGenerationError = '';
  denoiseWorkspaceRefresh();
}

function denoiseWorkspaceSelectAll() {
  denoiseWorkspaceSelection = new Set(denoiseWorkspaceVariants.map(denoiseWorkspaceKey));
  if (denoiseSelectedVariant().method === 'off') denoiseMatrixSelected = denoiseWorkspaceVariants[0];
  denoiseWorkspaceGenerationDismissed = false;
  denoiseWorkspaceGenerationError = '';
  denoiseWorkspaceRefresh();
}

function denoiseWorkspaceClear() {
  denoiseWorkspaceSelection = new Set();
  denoiseMatrixSelected = {method:'off',preset:''};
  denoiseWorkspaceGenerationError = '';
  denoiseWorkspaceRefresh();
}

function denoiseWorkspaceDismissGeneration() {
  denoiseWorkspaceGenerationDismissed = true;
  denoiseWorkspaceUpdateSelectionUi();
}

function denoiseWorkspaceRequestGeneration() {
  denoiseWorkspaceGenerationDismissed = false;
  denoiseWorkspaceUpdateSelectionUi();
  const prompt = document.querySelector('[data-dm-generation]');
  prompt?.scrollIntoView({block:'nearest',behavior:'smooth'});
  document.querySelector('[data-dm-generate]')?.focus({preventScroll:true});
}

async function denoiseWorkspaceGenerate() {
  if (state.running || denoiseWorkspaceGenerationPending) return;
  const variants = denoiseWorkspaceMissing();
  if (!variants.length) return;
  denoiseWorkspaceGenerationPending = true;
  denoiseWorkspaceGenerationError = '';
  denoiseWorkspaceUpdateSelectionUi();
  try {
    const result = await postJson('/api/'+comparisonStageKey()+'-batch', {variants});
    if (!result?.ok) throw new Error(result?.error || 'The renderer could not be started.');
    await refresh();
  } catch (error) {
    denoiseWorkspaceGenerationError = 'Rendering could not start: ' + (error?.message || 'local server unavailable');
  } finally {
    denoiseWorkspaceGenerationPending = false;
    if (isRestorationComparison()) denoiseWorkspaceUpdateSelectionUi();
  }
}

function denoiseWorkspaceUpdateSelectionUi() {
  const selected = denoiseSelectedVariant();
  const count = document.querySelector('[data-dm-comparison-count]');
  if (count) count.textContent = `${denoiseWorkspaceSelection?.size || 0}/${denoiseWorkspaceVariants.length} selected`;
  const missing = denoiseWorkspaceMissing();
  const prompt = document.querySelector('[data-dm-generation]');
  if (prompt) prompt.hidden = !missing.length || denoiseWorkspaceGenerationDismissed;
  const text = document.querySelector('[data-dm-generation-text]');
  if (text) text.textContent = 'Render for the current input? ' + missing.map(v => denoiseVariantLabelFor(v.method,v.preset)).join(', ') + '. Pipeline selection stays unchanged.';
  const button = document.querySelector('[data-dm-generate]');
  if (button) {
    button.textContent = `Render ${missing.length} missing variant${missing.length === 1 ? '' : 's'}`;
    button.disabled = Boolean(state.running || denoiseWorkspaceGenerationPending || !missing.length);
  }
  const status = document.querySelector('[data-dm-generation-status]');
  if (status) status.textContent = denoiseWorkspaceGenerationError ||
    (denoiseWorkspaceGenerationPending ? 'Starting selected variants…' :
      state.running && missing.length ? 'A job is running. Finished results appear automatically; unrendered selections can be rendered when it finishes.' : '');
  document.querySelectorAll('[data-dc-slot]').forEach(panel => {
    const variant = denoiseWorkspaceVariants[Number(panel.dataset.dcSlot)-1];
    if (!variant) return;
    const focused = denoiseSameVariant(variant,selected);
    panel.classList.toggle('is-inspected',focused);
    const focus = panel.querySelector('[data-dc-focus]');
    if (focus) focus.setAttribute('aria-pressed',String(focused));
    const pipeline = panel.querySelector('[data-dc-pipeline]');
    if (pipeline) pipeline.hidden = !denoiseSameVariant(variant,denoiseStageVariant());
  });
  const off = document.querySelector('.denoise-off-option');
  if (off) {
    const isOff = denoiseStageVariant().method === 'off';
    off.textContent = isOff ? 'Pipeline: unchanged input' : 'Pipeline: use unchanged input';
    off.classList.toggle('is-current',isOff);
    off.setAttribute('aria-pressed',String(isOff));
    off.disabled = Boolean(state.running || isOff || denoiseUseVariantPending);
  }
}

function denoiseWorkspaceRefresh() {
  const stage = state?.stages?.find(v => v.key === comparisonStageKey());
  if (!stage || !document.getElementById('denoiseComparePlayers')) return;
  denoiseWorkspaceRender(denoiseCompareCandidates(stage));
  denoiseMatrixUpdateUi();
  const method = document.querySelector('[data-field="method"]');
  const preset = document.querySelector('[data-field="preset"]');
  const pipelineChanged = method && method.value !== denoiseStageVariant().method ||
    preset && preset.value !== (denoiseStageVariant().preset || 'gentle');
  if (method) method.value = denoiseStageVariant().method;
  if (preset) preset.value = denoiseStageVariant().preset || 'gentle';
  if (pipelineChanged) showCommand(comparisonStageKey());
  const expected = document.querySelector('[data-dm-expected]');
  if (expected) {
    const html = shotOutputList(state.expected_outputs?.[comparisonStageKey()] || [],null);
    if (expected.innerHTML !== html) expected.innerHTML = html;
  }
  document.querySelectorAll('.denoise-settings-sidebar .actions > button').forEach(button => {
    if (button.getAttribute('onclick') === `runStage('${comparisonStageKey()}')`) button.disabled = Boolean(state.running);
    if (button.getAttribute('onclick') === 'stopRun()') button.disabled = !state.running;
  });
}

function denoiseWorkspacePanelHtml(slot, path) {
  const variant = denoiseWorkspaceVariants[slot-1];
  const label = slot ? denoiseVariantLabelFor(variant.method,variant.preset) : 'Original';
  return `<div class="denoise-result-heading">
      ${slot ? `<button type="button" data-dc-focus aria-pressed="false" onclick="denoiseWorkspaceFocus(${slot})" title="Inspect ${esc(label)} and its frame changes">${esc(label)}</button>
      <small data-dc-pipeline hidden>Pipeline</small>` : '<strong>Original</strong><small>Shared reference</small>'}
    </div>
    ${slot ? `<div class="denoise-view-modes" role="group" aria-label="${esc(label)} display">
      ${[['original','Original'],['result','Result'],['difference','Difference'],['wipe','Wipe']].map(([mode,text])=>`<button type="button" data-dp-mode="${mode}" aria-pressed="false" onclick="denoiseWorkspaceSetView(${slot},'${mode}')" ${path ? '' : 'disabled'} ${mode==='wipe' ? 'title="Original left / Result right · exact still"' : ''}>${text}</button>`).join('')}
    </div>` : '<div class="denoise-original-mode-spacer" aria-hidden="true"></div>'}
    <div class="denoise-compare-video-wrap">
      ${path ? `<canvas data-dp-canvas="${slot}" width="0" height="0" aria-label="${esc(label)} frame-locked preview"></canvas><span class="denoise-view-badge" data-dp-badge></span>` :
        `<p class="shot-empty">${slot ? 'No matching output for the current input.' : 'Prepare the source material to preview it here.'}</p>`}
      ${slot && path ? `<div class="denoise-inline-wipe" data-ds-wipe="${slot}" role="slider" tabindex="0" aria-label="${esc(label)} exact wipe: original left, result right" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" aria-orientation="horizontal" hidden><span>Original / Result</span></div>` : ''}
    </div>
    <output data-dp-status="${slot}" class="denoise-diff-status" role="status"></output>
    ${slot && path ? `<label class="denoise-inline-wipe-control" data-ds-wipe-control="${slot}" title="Original left · Result right" hidden><span data-ds-wipe-state>50% ORIGINAL · 50% RESULT</span>
      <input data-ds-wipe-range="${slot}" type="range" min="0" max="100" step="1" value="50" aria-label="${esc(label)} exact wipe position"></label>
      <output data-ds-readout="${slot}" class="denoise-exact-readout" hidden></output>` : slot ? '<button type="button" onclick="denoiseWorkspaceRequestGeneration()">Review render request</button>' : ''}`;
}

function denoiseWorkspaceSetView(slot, mode) {
  if (!['original','result','difference','wipe'].includes(mode)) return;
  if(mode==='wipe' && !denoiseWorkspacePlayback?.state.exact) denoiseWorkspaceSetExact(true);
  denoiseWorkspaceViews.set(slot,mode);
  denoiseWorkspacePlayback?.setView(slot,mode);
  denoiseWorkspaceStill?.draw();
  denoiseWorkspaceUpdateModes();
}

function denoiseWorkspaceSetExact(enabled) {
  denoiseWorkspaceExact=Boolean(enabled);
  denoiseWorkspacePlayback?.setExact(denoiseWorkspaceExact);
  denoiseWorkspaceUpdateModes();
}

function denoiseWorkspaceUpdateModes() {
  document.querySelectorAll('[data-dc-slot]').forEach(panel=>{
    const slot=Number(panel.dataset.dcSlot);
    const exact=Boolean(denoiseWorkspacePlayback?.state.exact);
    let mode=slot ? denoiseWorkspaceViews.get(slot) || 'result' : 'original';
    if(mode==='wipe' && !exact) mode='result';
    panel.querySelectorAll('[data-dp-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dpMode===mode)));
    const badge=panel.querySelector('[data-dp-badge]');
    if (badge) badge.textContent=(mode==='difference' ? `Difference · ${denoiseCompareDiffContrast}× · ${denoiseCompareDiffMode==='rgb' ? 'RGB' : 'false color'}` : mode==='original' ? 'Original' : mode==='wipe' ? 'Original / Result' : 'Result')+(exact ? ' · exact' : '');
    const control=panel.querySelector('[data-ds-wipe-control]');
    if(control) control.hidden=!exact || mode!=='wipe';
    const slider=panel.querySelector('[data-ds-wipe-range]');
    if(slider) slider.value=String(denoiseWorkspaceStill?.wipe(slot) ?? 50);
    const wipeState=panel.querySelector('[data-ds-wipe-state]');
    if(wipeState) wipeState.textContent=comparisonWipeVisibleState(denoiseWorkspaceStill?.wipe(slot) ?? 50,'RESULT','en');
    const readout=panel.querySelector('[data-ds-readout]');
    const still=denoiseWorkspaceStill?.state;
    const pair=still?.frame===denoiseWorkspacePlayback?.state.frame && !still.loading && !still.error ? still.pairs.get(slot) : null;
    if(readout) {
      readout.hidden=!exact || !pair?.metrics;
      readout.textContent=pair?.metrics ? `Frame ${still.frame+1} · mean |ΔY| ${pair.metrics.mean.toFixed(2)} / 255 · ${pair.metrics.changedPercent.toFixed(2)}% pixels changed` : '';
    }
  });
}

function denoiseWorkspacePlaybackSlots(host) {
  const visible=[...host.querySelectorAll('[data-dc-slot]')].filter(panel=>
    panel.dataset.visible==='true' && denoiseComparePaths[Number(panel.dataset.dcSlot)]);
  const results=visible.map(panel=>Number(panel.dataset.dcSlot)).filter(Boolean);
  return results.length ? results : visible.some(panel=>panel.dataset.dcSlot==='0') ? [0] : [];
}

function denoiseWorkspaceWirePlayback(host) {
  if (denoiseWorkspacePlayback?.host===host) return;
  denoiseWorkspacePlayback?.destroy();
  denoiseWorkspaceStill?.destroy();
  const fps=denoiseCompareFps() || 30;
  const stageKey=comparisonStageKey();
  const identity=slot=>[stageKey,denoiseWorkspaceSource,denoiseMatrixState().source_revision,
    slot ? denoiseMatrixState().variants?.find(v=>denoiseSameVariant(v,denoiseWorkspaceVariants[slot-1]))?.revision : ''].join('|');
  denoiseWorkspacePlayback=createDenoisePlayback(host,{
    stage:()=>stageKey,
    fps,count:Math.max(1,Math.round(denoiseCompareAvailableLength()*fps)),exact:denoiseWorkspaceExact,
    slots:()=>denoiseWorkspacePlaybackSlots(host),
    path:slot=>slot ? denoiseComparePaths[slot] || '' : '',
    identity,
    diffMode:()=>denoiseCompareDiffMode,contrast:()=>denoiseCompareDiffContrast,
    painted:()=>denoiseCompareZoomView?.apply(),
    changed:(playback,message)=>{
      denoiseWorkspaceExact=playback.exact;
      denoiseWorkspaceStill?.reconcile();
      denoiseWorkspaceUpdatePlayback(playback,message);
    },
  });
  denoiseWorkspaceStill=createDenoiseStillReview(host,{
    stage:()=>stageKey,
    wipes:denoiseWorkspaceWipes,
    playback:()=>denoiseWorkspacePlayback,slots:()=>denoiseWorkspacePlaybackSlots(host),
    path:slot=>slot ? denoiseComparePaths[slot] || '' : '',identity,
    view:slot=>denoiseWorkspaceViews.get(slot) || 'result',
    diffMode:()=>denoiseCompareDiffMode,contrast:()=>denoiseCompareDiffContrast,
    painted:()=>denoiseCompareZoomView?.apply(),
    changed:()=>{denoiseWorkspaceUpdatePlayback(denoiseWorkspacePlayback.state);denoiseMatrixUpdateUi();},
  });
  denoiseWorkspacePlayback.state.frame=denoiseWorkspaceFrame;
  for (const [slot,mode] of denoiseWorkspaceViews) denoiseWorkspacePlayback.setView(slot,mode);
}

function denoiseWorkspaceUpdatePlayback(playback,message='') {
  const host=denoiseWorkspacePlayback?.host;
  if(!host?.isConnected) return;
  const still=denoiseWorkspaceStill?.state,exact=playback.exact;
  const loading=exact ? Boolean(still?.loading) : playback.loading;
  const error=exact ? still?.error || '' : playback.error;
  denoiseWorkspaceFrame=playback.frame;
  host.classList.toggle('is-buffering',loading);host.classList.toggle('has-playback-error',Boolean(error));
  host.setAttribute('aria-busy',String(loading));
  host.querySelectorAll('[data-dp-canvas]').forEach(canvas=>{
    const current=canvas.dataset.dpFrame===String(playback.frame) && canvas.dataset.dpQuality===(exact ? 'exact' : 'preview');
    canvas.classList.toggle('is-awaiting-frame',Boolean(error || !current || loading && exact));
  });
  denoiseCompareEffectiveLength=playback.count/playback.fps;denoiseCompareLength=denoiseCompareEffectiveLength;
  const seek=document.getElementById('denoiseCompareSeek');
  if(seek) {seek.max=String((playback.count-1)/playback.fps);seek.step=String(1/playback.fps);seek.value=String(playback.frame/playback.fps);}
  const label=document.getElementById('denoiseCompareTime');
  if(label) label.textContent=`${denoisePlaybackTime(playback.frame/playback.fps)} / ${denoisePlaybackTime(playback.count/playback.fps)} · ${playback.frame+1}/${playback.count}`;
  const jump=document.querySelector('[data-dp-jump]');
  if(jump) {jump.max=String((playback.count-1)/playback.fps);if(document.activeElement!==jump) jump.value=(playback.frame/playback.fps).toFixed(3);}
  const toggle=document.querySelector('[data-ds-exact]');
  if(toggle) {toggle.setAttribute('aria-pressed',String(exact));toggle.disabled=!denoiseComparePaths[0];}
  const quality=document.querySelector('[data-ds-quality]');
  if(quality) quality.textContent=exact ? 'Exact still · 8-bit RGB' : 'Frame-locked · compressed display';
  const retry=document.querySelector('[data-ds-retry]');
  if(retry) retry.hidden=!exact || !error;
  const status=error ? exact ? error+' Retry the exact still, or press Play to return to compressed playback.' : message || error+' Retry playback, or choose Exact still.'
    : exact ? loading ? `Decoding exact frame ${playback.frame+1}…` : 'Exact 8-bit RGB still · no lossy preview compression. Press Play to return to compressed playback.'
    : message || 'Frame-locked views · compressed display. Differences are computed before display compression.';
  host.querySelectorAll('[data-dp-status]').forEach(output=>{output.textContent=loading || error ? status : '';});
  denoiseCompareSetPlaying(playback.playing);denoiseReviewHighlight(playback.frame);
  denoiseWorkspaceUpdateModes();denoiseCompareNotice(status);
}

function denoiseWorkspaceFocus(slot) {
  const variant = denoiseWorkspaceVariants[slot-1];
  if (!variant || !denoiseWorkspaceHas(variant.method,variant.preset)) return;
  denoiseMatrixSelected = variant;
  denoiseMatrixUpdateUi();
}

function denoiseWorkspaceSlotVisible(slot) {
  const panel = document.querySelector(`[data-dc-slot="${slot}"]`);
  return Boolean(panel && panel.dataset.visible !== 'false');
}

function denoiseWorkspaceObserve(host) {
  if (denoiseWorkspaceHost !== host) {
    denoiseWorkspaceObserver?.disconnect();
    denoiseWorkspaceHost = host;
    if (typeof IntersectionObserver === 'function') {
      const observer = new IntersectionObserver(entries => {
        if (!host.isConnected || !isRestorationComparison()) {
          observer.disconnect();
          if (denoiseWorkspacePlayback?.host===host) {
            denoiseWorkspacePlayback.destroy(); denoiseWorkspacePlayback=null;
            denoiseWorkspaceStill?.destroy();denoiseWorkspaceStill=null;
          }
          return;
        }
        for (const {target,isIntersecting} of entries) {
          const slot = Number(target.dataset.dcSlot);
          target.dataset.visible = String(isIntersecting);
        }
        denoiseWorkspacePlayback?.reconcile();
      },{threshold:0});
      denoiseWorkspaceObserver = observer;
    }
  }
  host.querySelectorAll('[data-dc-slot]').forEach(panel => {
    if (!panel.dataset.observed) {
      panel.dataset.observed = 'true';
      // Avoid starting offscreen decoders before the first intersection callback.
      panel.dataset.visible = denoiseWorkspaceObserver ? 'false' : 'true';
      denoiseWorkspaceObserver?.observe(panel);
    }
  });
}

function denoiseWorkspaceRender(candidates) {
  const host = document.getElementById('denoiseComparePlayers');
  if (!host) return;
  const sourceBefore = denoiseWorkspaceSource;
  denoiseWorkspaceInitialize(candidates);
  denoiseCompareEnsureLength();
  if (sourceBefore !== denoiseWorkspaceSource) {
    denoiseWorkspacePlayback?.destroy(); denoiseWorkspacePlayback=null;
    denoiseWorkspaceStill?.destroy();denoiseWorkspaceStill=null;
  }
  if (!host.querySelector('.denoise-workspace')) {
    host.innerHTML = '<div class="denoise-workspace"><div class="denoise-original-column"></div><div class="denoise-results-grid" aria-label="Selected comparison results"></div></div>';
  }
  const original = host.querySelector('.denoise-original-column');
  const results = host.querySelector('.denoise-results-grid');
  const slots = [0,...denoiseWorkspaceVariants.map((v,i) => denoiseWorkspaceHas(v.method,v.preset) ? i+1 : 0).filter(Boolean)];
  let changed = false;
  host.querySelectorAll('[data-dc-slot]').forEach(panel => {
    const slot = Number(panel.dataset.dcSlot);
    if (slots.includes(slot)) return;
    denoiseWorkspaceObserver?.unobserve(panel);
    panel.querySelectorAll('video').forEach(video => {video.pause(); video.removeAttribute('src'); video.load();});
    denoiseCompareDiffState[slot].request++;
    denoiseCompareDiffState[slot].loading = false;
    panel.remove();
    changed = true;
  });
  for (const slot of slots) {
    const variant = denoiseWorkspaceVariants[slot-1];
    const path = denoiseComparePaths[slot];
    const row = slot ? denoiseMatrixState().variants?.find(v => denoiseSameVariant(v,variant)) : null;
    const key = [path, row?.revision, denoiseMatrixState().source_revision].join('|');
    let panel = host.querySelector(`[data-dc-slot="${slot}"]`);
    if (!panel) {
      panel = document.createElement('article');
      panel.className = 'denoise-compare-panel';
      panel.dataset.dcSlot = String(slot);
      (slot ? results : original).appendChild(panel);
    }
    if (panel.dataset.mediaKey === key) continue;
    panel.querySelectorAll('video').forEach(video => {video.pause(); video.removeAttribute('src'); video.load();});
    if (slot) {
      const diff = denoiseCompareDiffState[slot];
      diff.request++;
      diff.url = ''; diff.key = ''; diff.loading = false;
    }
    panel.innerHTML = denoiseWorkspacePanelHtml(slot,path);
    panel.dataset.mediaKey = key;
    changed = true;
  }
  // Canonical matrix order, without moving unchanged tiles during ordinary polling.
  slots.slice(1).forEach((slot,index) => {
    const panel = host.querySelector(`[data-dc-slot="${slot}"]`);
    if (results.children[index] !== panel) results.insertBefore(panel,results.children[index] || null);
  });
  let empty = results.querySelector('[data-dc-empty]');
  if (slots.length === 1 && !empty) {
    results.insertAdjacentHTML('beforeend','<p class="shot-empty" data-dc-empty>Select variants in the matrix to build your comparison. The original stays unchanged.</p>');
  } else if (slots.length > 1) empty?.remove();
  results.classList.toggle('has-one-result',slots.length === 2);
  host.querySelector('.denoise-workspace').classList.toggle('has-one-result',slots.length === 2);
  denoiseWorkspaceObserve(host);
  denoiseWorkspaceWirePlayback(host);
  if (changed || denoiseCompareZoomView?.host !== host) {
    denoiseCompareWirePan();
  }
  denoiseWorkspacePlayback.reconcile();
  denoiseWorkspaceStill.reconcile();
  denoiseWorkspaceUpdateModes();
  denoiseCompareUpdateChartChoices();
}
