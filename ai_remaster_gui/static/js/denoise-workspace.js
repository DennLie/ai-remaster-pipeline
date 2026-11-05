/* Each tab owns its state; only the mounted workspace owns decoders and DOM work. */
const denoiseWorkspaces = new Map();
let denoiseWorkspaceActive = null;
let denoiseWorkspaceStarting = null;

function denoiseWorkspaceFor(key = comparisonStageKey()) {
  if (!denoiseWorkspaces.has(key)) denoiseWorkspaces.set(key, createDenoiseWorkspace(key));
  return denoiseWorkspaces.get(key);
}

function denoiseWorkspaceActivate(key) {
  if (!['denoise','deflicker','enhancement'].includes(key)) return;
  const workspace = denoiseWorkspaceFor(key);
  if (workspace === denoiseWorkspaceActive) return;
  denoiseWorkspaceDeactivate();
  denoiseWorkspaceActive = workspace;
}

function denoiseWorkspaceDeactivate() {
  denoiseWorkspaceActive?.deactivate();
  denoiseWorkspaceActive = null;
}

function createDenoiseWorkspace(stageKey) {
const methods = denoiseMethods(stageKey);
const presets = denoisePresets(stageKey);
const denoiseWorkspaceVariants = presets.flatMap(preset => methods.map(method => ({method:method.key,preset:preset.key})));
let matrix = state?.[stageKey+'_matrix'] || {};
const denoiseMatrixState = () => matrix;
const pipelineVariant = () => denoiseStageVariant(stageKey);
const variantPath = (method,preset) => denoiseVariantPath(method,preset,stageKey,matrix);
const variantReady = (method,preset) => denoiseVariantReady(method,preset,stageKey,matrix);
const labelFor = (method,preset) => denoiseVariantLabelFor(method,preset,stageKey);
let denoiseMatrixSelected = null;
let denoiseComparePaths = [];
let denoiseCompareZoom = 1, denoiseComparePan = {x:0,y:0};
let denoiseCompareNative = false, denoiseCompareGrid = false;
let denoiseCompareZoomView = null;
let denoiseCompareDiffMode = 'direction', denoiseCompareDiffContrast = 32;
let fps = 0, count = 1, length = 0;
let chart = null, metrics = null, chartPath = '';
let mounted = false, mountEpoch = 0, inputEpoch = 0;
let adoptionPending = false, adoptionError = '';
let denoiseWorkspaceSelection = null;
let denoiseWorkspaceSource = '';
let denoiseWorkspaceObserver = null;
let denoiseWorkspaceHost = null;
let denoiseWorkspaceGenerationDismissed = false;
let denoiseWorkspaceGenerationPending = false;
let denoiseWorkspaceGenerationError = '';
let generationContext = '', generationVariants = [];
let denoiseWorkspacePlayback = null;
let denoiseWorkspaceStill = null;
let denoiseWorkspaceExact = false;
let comparisonReference = 'input';
let denoiseWorkspaceFrame = 0;
const denoiseWorkspaceViews = new Map();
const denoiseWorkspaceWipes = new Map();
function deactivate() {
  mounted = false; mountEpoch++;
  denoiseWorkspaceStill?.destroy();denoiseWorkspaceStill=null;
  denoiseWorkspacePlayback?.destroy();denoiseWorkspacePlayback=null;
  denoiseWorkspaceObserver?.disconnect();denoiseWorkspaceObserver=null;denoiseWorkspaceHost=null;
  denoiseCompareZoomView?.destroy();denoiseCompareZoomView=null;
  chart?.destroy();chart=null;
}

function selectedVariant() { return denoiseMatrixSelected || pipelineVariant(); }
function snapshot() {
  return {stage:stageKey,variants:denoiseWorkspaceVariants.map(v=>({...v})),selected:{...selectedVariant()},
    selection:[...(denoiseWorkspaceSelection || [])],paths:[...denoiseComparePaths],frame:denoiseWorkspaceFrame,
    zoom:denoiseCompareZoom,pan:{...denoiseComparePan},native:denoiseCompareNative,grid:denoiseCompareGrid,
    exact:denoiseWorkspaceExact,reference:comparisonReference,views:new Map(denoiseWorkspaceViews),wipes:new Map(denoiseWorkspaceWipes),
    diffMode:denoiseCompareDiffMode,diffContrast:denoiseCompareDiffContrast,fps,count,length,
    generationPending:denoiseWorkspaceGenerationPending,generationError:denoiseWorkspaceGenerationError,
    generationDismissed:denoiseWorkspaceGenerationDismissed,adoptionPending,chartPath,metrics:metrics ? {...metrics} : null,
    loading:Boolean(denoiseWorkspaceExact ? denoiseWorkspaceStill?.state.loading : denoiseWorkspacePlayback?.state.loading),
    error:(denoiseWorkspaceExact ? denoiseWorkspaceStill?.state.error : denoiseWorkspacePlayback?.state.error) || ''};
}

function denoiseWorkspaceKey(variant) { return variant.method + '/' + variant.preset; }
function denoiseWorkspaceHas(method, preset) {
  return Boolean(denoiseWorkspaceSelection?.has(denoiseWorkspaceKey({method,preset})));
}
function denoiseWorkspaceSelected() {
  return denoiseWorkspaceVariants.filter(v => denoiseWorkspaceHas(v.method,v.preset));
}
function denoiseWorkspaceMissing() {
  const review=typeof comparisonProfile==='function' && comparisonProfile()==='review';
  const variants=review ? (selectedVariant().method==='off' ? [] : [selectedVariant()]) : denoiseWorkspaceSelected();
  return variants.filter(v => !variantReady(v.method,v.preset));
}

function denoiseWorkspaceInitialize(candidates) {
  const latest = state?.[stageKey+'_matrix'];
  if (latest && Object.hasOwn(latest,'source')) matrix = latest;
  if(generationContext && generationContext!==renderContext(generationVariants)) denoiseWorkspaceGenerationError='';
  const source = candidates.find(v => v.kind === 'source')?.path || '';
  const identity = [source, matrix.source_revision, state?.settings?.global?.source, denoiseCompareSectionKey()].join('|');
  if (denoiseWorkspaceSelection === null) {
    const stage = pipelineVariant();
    const ready = denoiseWorkspaceVariants.filter(v => variantReady(v.method,v.preset));
    const initial = stage.method === 'off' ? ready.slice(0,2)
      : [stage, ...ready.filter(v => !denoiseSameVariant(v,stage)).slice(0,1)];
    denoiseWorkspaceSelection = new Set(initial.map(denoiseWorkspaceKey));
    denoiseMatrixSelected = initial[0] || {method:'off',preset:''};
  }
  if (identity !== denoiseWorkspaceSource) {
    denoiseWorkspaceSource = identity;
    inputEpoch++;
    denoiseWorkspaceGenerationDismissed = false;
    denoiseWorkspaceGenerationError = '';
    denoiseCompareZoom = 1;
    denoiseCompareNative = false;
    denoiseWorkspaceFrame = 0;
    denoiseComparePan = {x:0,y:0};
    fps=0;count=1;metrics=null;
  }
  length=denoiseCompareAvailableLength();
  denoiseComparePaths = [source, ...denoiseWorkspaceVariants.map((v,index) => {
    return (denoiseWorkspaceHas(v.method,v.preset) || denoiseSameVariant(v,selectedVariant())) && variantReady(v.method,v.preset)
      ? variantPath(v.method,v.preset) : '';
  })];
}

function denoiseWorkspaceToggle(method,preset) {
  const key = denoiseWorkspaceKey({method,preset});
  denoiseWorkspaceSelection ||= new Set();
  if (denoiseWorkspaceSelection.has(key)) {
    denoiseWorkspaceSelection.delete(key);
    if (denoiseSameVariant(selectedVariant(),{method,preset}))
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
  if (selectedVariant().method === 'off') denoiseMatrixSelected = denoiseWorkspaceVariants[0];
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
  if (state.running || denoiseWorkspaceStarting) return;
  if(state.processing_input?.problem) {
    denoiseWorkspaceGenerationError=state.processing_input.problem;
    denoiseWorkspaceUpdateSelectionUi();return;
  }
  const variants = denoiseWorkspaceMissing();
  if (!variants.length) return;
  denoiseWorkspaceGenerationPending = true;
  const request={context:renderContext(variants),epoch:inputEpoch};
  generationContext=request.context;generationVariants=variants;
  denoiseWorkspaceStarting=request;
  denoiseWorkspaceGenerationError = '';
  denoiseWorkspaceUpdateSelectionUi();
  try {
    const result = await postJson('/api/'+stageKey+'-batch', {variants});
    if (!result?.ok) throw new Error(result?.error || 'The renderer could not be started.');
    await refresh();
  } catch (error) {
    if(request.epoch===inputEpoch && request.context===renderContext(variants))
      denoiseWorkspaceGenerationError = 'Rendering could not start: ' + (error?.message || 'local server unavailable');
  } finally {
    denoiseWorkspaceGenerationPending = false;
    if(denoiseWorkspaceStarting===request) denoiseWorkspaceStarting=null;
    if (denoiseWorkspaceActive===workspace) denoiseWorkspaceUpdateSelectionUi();
  }
}

function renderContext(variants) {
  const settings=state?.settings || {},values=settings[stageKey] || {};
  const parameters=variants.map(({method})=>stageKey==='enhancement' ? [method,values.mix,['nafnet','restormer'].includes(method) ? null : values.preset] : method==='arp' ? [method,values.strength,values.max_correction]
    : method.startsWith('ffmpeg_') ? [method,values.window]
    : [method,values.localized_confirm_single_shot,values.localized_confirm_film_experiment]);
  return JSON.stringify([settings.global,stageKey!=='denoise' ? settings.denoise : null,stageKey==='enhancement' ? settings.deflicker : null,
    settings.cloud?.intermediate_format,parameters]);
}

function denoiseWorkspaceUpdateSelectionUi() {
  const selected = selectedVariant();
  const count = document.querySelector('[data-dm-comparison-count]');
  if (count) count.textContent = `${denoiseWorkspaceSelection?.size || 0}/${denoiseWorkspaceVariants.length} selected`;
  const missing = denoiseWorkspaceMissing();
  const prompt = document.querySelector('[data-dm-generation]');
  if (prompt) prompt.hidden = !missing.length || denoiseWorkspaceGenerationDismissed;
  const text = document.querySelector('[data-dm-generation-text]');
  if (text) text.textContent = 'Render for the current input? ' + missing.map(v => labelFor(v.method,v.preset)).join(', ') + '. Pipeline selection stays unchanged.';
  const button = document.querySelector('[data-dm-generate]');
  if (button) {
    button.textContent = `Render ${missing.length} missing variant${missing.length === 1 ? '' : 's'}`;
    button.disabled = Boolean(state.running || state.processing_input?.problem || denoiseWorkspaceStarting || !missing.length);
  }
  const status = document.querySelector('[data-dm-generation-status]');
  if (status) {status.dataset.state=denoiseWorkspaceGenerationError ? 'error' : 'running';status.textContent = denoiseWorkspaceGenerationError ||
    (denoiseWorkspaceGenerationPending ? 'Starting selected variants…' :
      state.running && missing.length ? 'A job is running. Finished results appear automatically; unrendered selections can be rendered when it finishes.' : '');}
  document.querySelectorAll('[data-dc-slot]').forEach(panel => {
    const variant = denoiseWorkspaceVariants[Number(panel.dataset.dcSlot)-1];
    if (!variant) return;
    const focused = denoiseSameVariant(variant,selected);
    panel.classList.toggle('is-inspected',focused);
    const focus = panel.querySelector('[data-dc-focus]');
    if (focus) focus.setAttribute('aria-pressed',String(focused));
    const pipeline = panel.querySelector('[data-dc-pipeline]');
    if (pipeline) pipeline.hidden = !denoiseSameVariant(variant,pipelineVariant());
  });
  const off = document.querySelector('.denoise-off-option');
  if (off) {
    const isOff = pipelineVariant().method === 'off';
    off.textContent = isOff ? 'Pipeline: unchanged input' : 'Pipeline: use unchanged input';
    off.classList.toggle('is-current',isOff);
    off.setAttribute('aria-pressed',String(isOff));
    off.disabled = Boolean(state.running || isOff || adoptionPending);
  }
  const notice=document.querySelector('[data-dm-notice]');
  if(notice) {notice.dataset.state=adoptionError ? 'error' : 'idle';notice.textContent=adoptionError;}
  if(typeof updateComparisonInspector==='function') updateComparisonInspector();
}

function denoiseWorkspaceRefresh() {
  const stage = state?.stages?.find(v => v.key === stageKey);
  if (!stage || !document.getElementById('denoiseComparePlayers')) return;
  denoiseWorkspaceRender(denoiseCompareCandidates(stage));
  denoiseMatrixUpdateUi();
  const method = document.querySelector('[data-field="method"]');
  const preset = document.querySelector('[data-field="preset"]');
  const processingPreset = stageKey==='denoise' ? pipelineVariant().preset || 'gentle' : state.settings?.[stageKey]?.preset || 'gentle';
  const pipelineChanged = method && method.value !== pipelineVariant().method ||
    preset && preset.value !== processingPreset;
  if (method) method.value = pipelineVariant().method;
  if (preset) preset.value = processingPreset;
  if (pipelineChanged) showCommand(stageKey);
  const expected = document.querySelector('[data-dm-expected]');
  const input=document.querySelector('[data-processing-input]');
  if(input && typeof processingInputHtml==='function') {
    const html=processingInputHtml(stageKey);
    if(input.dataset.signature!==html) {
      const expanded=input.querySelector('details')?.open;
      input.innerHTML=html;input.dataset.signature=html;
      if(expanded && input.querySelector('details')) input.querySelector('details').open=true;
      if(typeof updateComparisonInputContext==='function') updateComparisonInputContext();
    }
  }
  if (expected) {
    const html = shotOutputList(state.expected_outputs?.[stageKey] || [],null);
    if (expected.innerHTML !== html) expected.innerHTML = html;
  }
  document.querySelectorAll('.denoise-settings-sidebar .actions > button').forEach(button => {
    if (button.getAttribute('onclick') === `runStage('${stageKey}')`) button.disabled = Boolean(state.running || state.processing_input?.problem);
    if (button.getAttribute('onclick') === 'stopRun()') button.disabled = !state.running;
  });
}

function denoiseWorkspacePanelHtml(slot, path) {
  const variant = denoiseWorkspaceVariants[slot-1];
  const label = slot ? labelFor(variant.method,variant.preset) : 'Stage input';
  return `<div class="denoise-result-heading">
      ${slot ? `<button type="button" data-dc-focus aria-pressed="false" onclick="denoiseWorkspaceFocus(${slot})" title="Inspect ${esc(label)} and its frame changes">${esc(label)}</button>
      <small data-dc-pipeline hidden>Pipeline</small>` : '<strong>Stage input</strong><small>Reference for this stage</small>'}
    </div>
    ${slot ? `<div class="denoise-view-modes" role="group" aria-label="${esc(label)} display">
      ${[['original','Stage input'],['result','Result'],['difference','Difference vs input'],['wipe','Wipe']].map(([mode,text])=>`<button type="button" data-dp-mode="${mode}" aria-pressed="false" onclick="denoiseWorkspaceSetView(${slot},'${mode}')" ${path ? '' : 'disabled'} ${mode==='wipe' ? 'title="Stage input left / Result right · exact still"' : ''}>${text}</button>`).join('')}
    </div>` : '<div class="denoise-original-mode-spacer" aria-hidden="true"></div>'}
    <div class="denoise-compare-video-wrap">
      ${path ? `<canvas data-dp-canvas="${slot}" width="0" height="0" aria-label="${esc(label)} frame-locked preview"></canvas><span class="denoise-view-badge" data-dp-badge></span>` :
        `<p class="shot-empty">${slot ? 'No matching output for the current input.' : 'Prepare the source material to preview it here.'}</p>`}
      ${slot && path ? `<div class="denoise-inline-wipe" data-ds-wipe="${slot}" role="slider" tabindex="0" aria-label="${esc(label)} exact wipe: stage input left, result right" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50" aria-orientation="horizontal" hidden><span>Stage input / Result</span></div>` : ''}
    </div>
    <output data-dp-status="${slot}" class="denoise-diff-status" role="status"></output>
    ${slot && path ? `<label class="denoise-inline-wipe-control" data-ds-wipe-control="${slot}" title="Stage input left · Result right" hidden><span data-ds-wipe-state>50% STAGE INPUT · 50% RESULT</span>
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
  const referenceLabel=comparisonReference==='original' ? 'Original source' : 'Stage input';
  const page=document.querySelector('.comparison-workbench');
  if(page) page.dataset.reference=comparisonReference;
  document.querySelectorAll('[data-reference]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.reference===comparisonReference)));
  const heading=document.querySelector('[data-dc-slot="0"] .denoise-result-heading strong');
  if(heading) heading.textContent=referenceLabel;
  const referenceCanvas=document.querySelector('[data-dp-canvas="0"]');
  if(referenceCanvas) referenceCanvas.setAttribute('aria-label',referenceLabel+' frame-locked comparison reference');
  const referenceNote=document.querySelector('[data-dc-slot="0"] .denoise-result-heading small');
  if(referenceNote) referenceNote.textContent=comparisonReference==='original' ? 'Unprocessed project source' : 'Input to this stage';
  document.querySelectorAll('[data-dc-slot]').forEach(panel=>{
    const slot=Number(panel.dataset.dcSlot);
    const exact=Boolean(denoiseWorkspacePlayback?.state.exact);
    let mode=slot ? denoiseWorkspaceViews.get(slot) || 'result' : 'original';
    if(mode==='wipe' && !exact) mode='result';
    panel.querySelectorAll('[data-dp-mode]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dpMode===mode)));
    const originalButton=panel.querySelector('[data-dp-mode="original"]'),differenceButton=panel.querySelector('[data-dp-mode="difference"]');
    if(originalButton) originalButton.textContent=referenceLabel;
    if(differenceButton) differenceButton.textContent='Difference vs '+(comparisonReference==='original' ? 'original' : 'input');
    const wipeButton=panel.querySelector('[data-dp-mode="wipe"]');
    if(wipeButton) wipeButton.title=referenceLabel+' left / Result right · exact still';
    const badge=panel.querySelector('[data-dp-badge]');
    if (badge) badge.textContent=(mode==='difference' ? `Difference vs ${referenceLabel} · ${denoiseCompareDiffContrast}× · ${denoiseCompareDiffMode==='rgb' ? 'RGB' : 'false color'}` : mode==='original' ? referenceLabel : mode==='wipe' ? referenceLabel+' / Result' : 'Result')+(exact ? ' · exact' : '');
    const control=panel.querySelector('[data-ds-wipe-control]');
    if(control) control.hidden=!exact || mode!=='wipe';
    const slider=panel.querySelector('[data-ds-wipe-range]');
    if(slider) {slider.value=String(denoiseWorkspaceStill?.wipe(slot) ?? 50);slider.setAttribute('aria-label',referenceLabel+' left, result right: wipe position');}
    const wipeState=panel.querySelector('[data-ds-wipe-state]');
    if(wipeState) wipeState.textContent=comparisonWipeVisibleState(denoiseWorkspaceStill?.wipe(slot) ?? 50,'RESULT','en').replaceAll('ORIGINAL',referenceLabel.toUpperCase());
    const handle=panel.querySelector('[data-ds-wipe]');
    if(handle) {handle.setAttribute('aria-label',referenceLabel+' left, result right');if(handle.firstElementChild) handle.firstElementChild.textContent=referenceLabel+' / Result';}
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
    !panel.hidden && panel.dataset.visible==='true' && denoiseComparePaths[Number(panel.dataset.dcSlot)]);
  const results=visible.map(panel=>Number(panel.dataset.dcSlot)).filter(Boolean);
  return results.length ? results : visible.some(panel=>panel.dataset.dcSlot==='0') ? [0] : [];
}

function denoiseWorkspaceWirePlayback(host) {
  if (denoiseWorkspacePlayback?.host===host) return;
  denoiseWorkspacePlayback?.destroy();
  denoiseWorkspaceStill?.destroy();
  const rate=fps || denoiseCompareFpsValue(state?.source_info?.fps) || 30;
  const epoch=mountEpoch;
  const current=()=>mounted && mountEpoch===epoch && denoiseWorkspaceActive===workspace && host.isConnected;
  const identity=slot=>[stageKey,comparisonReference,denoiseWorkspaceSource,denoiseMatrixState().source_revision,
    slot ? denoiseMatrixState().variants?.find(v=>denoiseSameVariant(v,denoiseWorkspaceVariants[slot-1]))?.revision : ''].join('|');
  denoiseWorkspacePlayback=createDenoisePlayback(host,{
    stage:()=>stageKey,
    reference:()=>comparisonReference,
    fps:rate,count:Math.max(1,Math.round(length*rate)),frame:denoiseWorkspaceFrame,exact:denoiseWorkspaceExact,
    slots:()=>denoiseWorkspacePlaybackSlots(host),
    path:slot=>slot ? denoiseComparePaths[slot] || '' : '',
    identity,
    diffMode:()=>denoiseCompareDiffMode,contrast:()=>denoiseCompareDiffContrast,
    painted:()=>{if(current()) denoiseCompareZoomView?.apply();},
    changed:(playback,message)=>{
      if(!current()) return;
      denoiseWorkspaceExact=playback.exact;
      denoiseWorkspaceStill?.reconcile();
      denoiseWorkspaceUpdatePlayback(playback,message);
    },
  });
  denoiseWorkspaceStill=createDenoiseStillReview(host,{
    stage:()=>stageKey,
    reference:()=>comparisonReference,
    wipes:denoiseWorkspaceWipes,
    playback:()=>denoiseWorkspacePlayback,slots:()=>denoiseWorkspacePlaybackSlots(host),
    path:slot=>slot ? denoiseComparePaths[slot] || '' : '',identity,
    view:slot=>denoiseWorkspaceViews.get(slot) || 'result',
    diffMode:()=>denoiseCompareDiffMode,contrast:()=>denoiseCompareDiffContrast,
    painted:()=>{if(current()) denoiseCompareZoomView?.apply();},
    metadata:info=>{if(current()) denoiseWorkspacePlayback.setMetadata(info);},
    measured:(frameMetrics,slot)=>{
      if(!frameMetrics) {if(current()) metrics=null;return;}
      if(current() && variantPath(selectedVariant().method,selectedVariant().preset)===denoiseComparePaths[slot])
        metrics={...frameMetrics,frame:denoiseWorkspaceFrame};
    },
    changed:()=>{if(current()) {denoiseWorkspaceUpdatePlayback(denoiseWorkspacePlayback.state);denoiseMatrixUpdateUi();}},
  });
  for (const [slot,mode] of denoiseWorkspaceViews) denoiseWorkspacePlayback.setView(slot,mode);
}

function denoiseWorkspaceUpdatePlayback(playback,message='') {
  const host=denoiseWorkspacePlayback?.host;
  if(!host?.isConnected) return;
  const still=denoiseWorkspaceStill?.state,exact=playback.exact;
  const loading=exact ? Boolean(still?.loading) : playback.loading;
  const error=exact ? still?.error || '' : playback.error;
  denoiseWorkspaceFrame=playback.frame;
  fps=playback.fps;count=playback.count;
  host.classList.toggle('is-buffering',loading);host.classList.toggle('has-playback-error',Boolean(error));
  host.setAttribute('aria-busy',String(loading));
  host.querySelectorAll('[data-dp-canvas]').forEach(canvas=>{
    const current=canvas.dataset.dpFrame===String(playback.frame) && canvas.dataset.dpQuality===(exact ? 'exact' : 'preview');
    canvas.classList.toggle('is-awaiting-frame',Boolean(error || !current || loading && exact));
  });
  length=playback.count/playback.fps;
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
  denoiseCompareSetPlaying(playback.playing);chart?.highlight(playback.frame);
  denoiseWorkspaceUpdateModes();denoiseCompareNotice(status);
}

function denoiseWorkspaceFocus(slot) {
  const variant = denoiseWorkspaceVariants[slot-1];
  if (!variant) return;
  denoiseMatrixSelected = variant;
  metrics=null;
  denoiseWorkspaceGenerationDismissed=false;
  denoiseWorkspaceRefresh();
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
      const epoch=mountEpoch;
      const observer = new IntersectionObserver(entries => {
        if(epoch!==mountEpoch) return;
        if (!host.isConnected || !mounted || denoiseWorkspaceActive!==workspace) {
          observer.disconnect();
          if (denoiseWorkspacePlayback?.host===host) {
            denoiseWorkspacePlayback.destroy(); denoiseWorkspacePlayback=null;
            denoiseWorkspaceStill?.destroy();denoiseWorkspaceStill=null;
          }
          return;
        }
        for (const {target,isIntersecting} of entries) {
          const slot = Number(target.dataset.dcSlot);
          target.dataset.visible = String(isIntersecting && !target.hidden);
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
  // Keep the reference controls when a source revision replaces the panel markup.
  const reference=host.closest('.comparison-workbench')?.querySelector('.comparison-reference');
  if(denoiseWorkspaceHost && denoiseWorkspaceHost!==host) deactivate();
  const sourceBefore = denoiseWorkspaceSource;
  denoiseWorkspaceInitialize(candidates);
  mounted=true;
  if (sourceBefore !== denoiseWorkspaceSource) {
    deactivate();mounted=true;
  }
  if (!host.querySelector('.denoise-workspace')) {
    host.innerHTML = '<div class="denoise-workspace"><div class="denoise-original-column"></div><div class="denoise-results-grid" aria-label="Selected comparison results"></div></div>';
  }
  const original = host.querySelector('.denoise-original-column');
  const results = host.querySelector('.denoise-results-grid');
  const slots = [0,...denoiseWorkspaceVariants.map((v,i) => denoiseWorkspaceHas(v.method,v.preset) || denoiseSameVariant(v,selectedVariant()) ? i+1 : 0).filter(Boolean)];
  let changed = false;
  host.querySelectorAll('[data-dc-slot]').forEach(panel => {
    const slot = Number(panel.dataset.dcSlot);
    if (slots.includes(slot)) return;
    denoiseWorkspaceObserver?.unobserve(panel);
    panel.querySelectorAll('video').forEach(video => {video.pause(); video.removeAttribute('src'); video.load();});
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
    panel.innerHTML = denoiseWorkspacePanelHtml(slot,path);
    panel.dataset.mediaKey = key;
    changed = true;
  }
  const referenceHeading=original.querySelector('.denoise-result-heading');
  if(reference && referenceHeading && !referenceHeading.contains(reference)) {
    referenceHeading.querySelectorAll('strong,small').forEach(node=>node.classList.add('sr-only'));
    referenceHeading.append(reference);
  }
  // Canonical matrix order, without moving unchanged tiles during ordinary polling.
  slots.slice(1).forEach((slot,index) => {
    const panel = host.querySelector(`[data-dc-slot="${slot}"]`);
    if (results.children[index] !== panel) results.insertBefore(panel,results.children[index] || null);
  });
  const review=typeof comparisonProfile==='function' && comparisonProfile()==='review';
  const displayed=slots.filter(slot=>!slot || (review ? denoiseSameVariant(denoiseWorkspaceVariants[slot-1],selectedVariant())
    : denoiseWorkspaceHas(denoiseWorkspaceVariants[slot-1].method,denoiseWorkspaceVariants[slot-1].preset)));
  let empty = results.querySelector('[data-dc-empty]');
  if (displayed.length === 1 && !empty) {
    results.insertAdjacentHTML('beforeend','<p class="shot-empty" data-dc-empty>Select a method in Review or include variants in Labor. The stage input stays unchanged.</p>');
  } else if (displayed.length > 1) empty?.remove();
  results.classList.toggle('has-one-result',displayed.length === 2);
  host.querySelector('.denoise-workspace').classList.toggle('has-one-result',displayed.length === 2);
  if(typeof updateComparisonInspector==='function') updateComparisonInspector();
  denoiseWorkspaceObserve(host);
  denoiseWorkspaceWirePlayback(host);
  if (changed || denoiseCompareZoomView?.host !== host) {
    wireZoom(host);
  }
  denoiseWorkspacePlayback.reconcile();
  denoiseWorkspaceStill.reconcile();
  denoiseWorkspaceUpdateModes();
  denoiseCompareUpdateChartChoices();
}

function wireZoom(host) {
  denoiseCompareZoomView?.destroy();
  const epoch=mountEpoch;
  denoiseCompareZoomView=createReviewZoom(host,'.denoise-compare-video-wrap',document.getElementById('denoiseCompareZoomControls'),
    {zoom:denoiseCompareZoom,...denoiseComparePan,native:denoiseCompareNative,grid:denoiseCompareGrid},value=>{
      if(epoch!==mountEpoch || !mounted) return;
      denoiseCompareZoom=value.zoom;denoiseComparePan={x:value.x,y:value.y};
      denoiseCompareNative=Boolean(value.native);denoiseCompareGrid=Boolean(value.grid);
      denoiseWorkspaceStill?.updateWipes();
    });
}

function setViewport(value) {
  if(value.zoom!==undefined) denoiseCompareZoom=reviewZoomClamp(value.zoom,1/32,32);
  if(value.pan) denoiseComparePan={...value.pan};
  if(value.native!==undefined) denoiseCompareNative=Boolean(value.native);
  if(value.grid!==undefined) denoiseCompareGrid=Boolean(value.grid);
  if(denoiseCompareZoomView) {
    denoiseCompareZoomView.setViewport({zoom:denoiseCompareZoom,...denoiseComparePan,native:denoiseCompareNative,grid:denoiseCompareGrid});
  }
}

async function useSelected(selected=selectedVariant()) {
  if(state.running || adoptionPending || selected.method!=='off' && !variantReady(selected.method,selected.preset)) return;
  adoptionPending=true;adoptionError='';
  const context=renderContext([]),epoch=inputEpoch;
  const button=document.getElementById('denoiseUseVariant');
  if(button) {button.disabled=true;button.textContent='Adopting…';}
  denoiseWorkspaceUpdateSelectionUi();
  try {
    const values=stageKey!=='denoise' ? {method:selected.method,force:'false'}
      : {method:selected.method,preset:selected.preset || 'gentle',force:'false'};
    const result=await postJson('/api/settings',{stage:stageKey,values});
    if(!result?.ok) throw new Error(result?.error || 'Settings could not be saved.');
    if(denoiseWorkspaceActive===workspace && epoch===inputEpoch && context===renderContext([])) {
      const force=document.querySelector('[data-field="force"]');
      if(force) force.checked=false;
    }
    await refresh();
  }catch(error) {
    if(epoch===inputEpoch && context===renderContext([])) adoptionError='The pipeline selection could not be changed: '+(error?.message || 'unknown error');
  }finally {
    adoptionPending=false;
    if(denoiseWorkspaceActive===workspace) denoiseMatrixUpdateUi();
  }
}

const workspace={initialize:denoiseWorkspaceInitialize,deactivate,snapshot,
  has:denoiseWorkspaceHas,toggle:denoiseWorkspaceToggle,selectAll:denoiseWorkspaceSelectAll,clear:denoiseWorkspaceClear,
  generate:denoiseWorkspaceGenerate,dismiss:denoiseWorkspaceDismissGeneration,request:denoiseWorkspaceRequestGeneration,
  updateSelectionUi:denoiseWorkspaceUpdateSelectionUi,refresh:denoiseWorkspaceRefresh,render:denoiseWorkspaceRender,
  setView:denoiseWorkspaceSetView,setExact:denoiseWorkspaceSetExact,updateModes:denoiseWorkspaceUpdateModes,
  setReference:value=>{
    if(!['input','original'].includes(value) || value===comparisonReference) return;
    comparisonReference=value;metrics=null;
    denoiseWorkspacePlayback?.pause();
    denoiseWorkspacePlayback?.reconcile();denoiseWorkspaceStill?.reconcile(true);
    denoiseWorkspaceUpdateModes();
  },
  focus:denoiseWorkspaceFocus,slotVisible:denoiseWorkspaceSlotVisible,setViewport,useSelected,
  retry:()=>denoiseWorkspaceStill?.reconcile(true),
  seek:seconds=>denoiseWorkspacePlayback?.seekSeconds(seconds),
  seekFrame:frame=>{denoiseWorkspaceFrame=Math.max(0,Math.round(frame));return denoiseWorkspacePlayback?.seekFrame(frame);},
  step:frames=>denoiseWorkspacePlayback?.step(frames),playPause:()=>denoiseWorkspacePlayback?.toggle(),
  setWipe:(slot,value)=>{denoiseWorkspaceWipes.set(slot,Math.max(0,Math.min(100,Number(value)||0)));denoiseWorkspaceStill?.draw();},
  setDifference:(mode,contrast)=>{denoiseCompareDiffMode=mode;denoiseCompareDiffContrast=contrast;denoiseWorkspacePlayback?.reconcile();denoiseWorkspaceStill?.draw();denoiseWorkspaceUpdateModes();},
  syncChart:()=>{
    if(!mounted || !denoiseWorkspaceHost) return;
    if(!chart) chart=createDenoiseReviewChart(denoiseWorkspaceHost.parentElement || denoiseWorkspaceHost,{
      stage:stageKey,selection:()=>{
        const selected=selectedVariant(),path=selected.method==='off' ? '' : variantPath(selected.method,selected.preset);
        const row=matrix.variants?.find(item=>denoiseSameVariant(item,selected));
        return {path,identity:[denoiseWorkspaceSource,row?.revision].join('|')};
      },frame:()=>denoiseWorkspaceFrame,seek:frame=>{denoiseWorkspacePlayback?.pause();return workspace.seekFrame(frame);},
      changed:path=>{chartPath=path;metrics=null;denoiseWorkspaceStill?.draw();},
    });
    chart.reconcile();
  },
};
return workspace;
}

// Existing inline controls enter the workspace here; asynchronous work never resolves through active.
function denoiseWorkspaceSnapshot(key) {return denoiseWorkspaceFor(key).snapshot();}
function denoiseWorkspaceInitialize(candidates) {denoiseWorkspaceActivate(comparisonStageKey());return denoiseWorkspaceFor().initialize(candidates);}
function denoiseWorkspaceHas(method,preset) {return denoiseWorkspaceFor().has(method,preset);}
function denoiseWorkspaceToggle(method,preset) {return denoiseWorkspaceFor().toggle(method,preset);}
function denoiseWorkspaceSelectAll() {return denoiseWorkspaceFor().selectAll();}
function denoiseWorkspaceClear() {return denoiseWorkspaceFor().clear();}
function denoiseWorkspaceGenerate() {return denoiseWorkspaceFor().generate();}
function denoiseWorkspaceDismissGeneration() {return denoiseWorkspaceFor().dismiss();}
function denoiseWorkspaceRequestGeneration() {return denoiseWorkspaceFor().request();}
function denoiseWorkspaceUpdateSelectionUi() {return denoiseWorkspaceFor().updateSelectionUi();}
function denoiseWorkspaceRefresh() {return denoiseWorkspaceFor().refresh();}
function denoiseWorkspaceRender(candidates) {denoiseWorkspaceActivate(comparisonStageKey());return denoiseWorkspaceFor().render(candidates);}
function denoiseWorkspaceSetView(slot,mode) {return denoiseWorkspaceFor().setView(slot,mode);}
function denoiseWorkspaceSetExact(enabled=!denoiseWorkspaceSnapshot().exact) {return denoiseWorkspaceFor().setExact(enabled);}
function denoiseWorkspaceRetry() {return denoiseWorkspaceFor().retry();}
function denoiseWorkspaceUpdateModes() {return denoiseWorkspaceFor().updateModes();}
function denoiseWorkspaceFocus(slot) {return denoiseWorkspaceFor().focus(slot);}
function denoiseWorkspaceSlotVisible(slot) {return denoiseWorkspaceFor().slotVisible(slot);}
