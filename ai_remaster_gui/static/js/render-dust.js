/* Drafts, inspection and pipeline adoption are separate decisions. */
let dustDraft=null,dustProject='',dustWorkspaceCatalog='',dustRunPending=false,dustUpdateStatus=null;
let dustEpoch=0,dustInspectEpoch=0,dustInspectionKey='',dustView='mask',dustStatus=null,dustFollowDraft=false;
let dustRepairOpen=false,dustInputMode='input',dustOpacity=0.5;
let dustPaintFrame=null;
let dustAlignmentObserver=null;
const dustNames={legacy:'ARP legacy scratch mask',temporal:'Temporal Dust',ridge:'Temporal Ridge · experimental',dart:'DART · joint research inference',telea:'OpenCV Telea',propainter:'ProPainter · non-commercial'};
function dustContext() {return JSON.stringify([state.project_path,state.settings.global.source,state.dust_matrix?.source_revision]);}
function dustCatalog() {return JSON.stringify([dustContext(),(state.dust_matrix?.variants || []).map(v=>[v.method,v.repair_complete])]);}
function dustLabel(values) {return values.detector==='dart' ? 'DART · joint restoration' : `${dustNames[values.detector] || values.detector} · ${dustNames[values.repair] || values.repair}`;}
function dustMatches(values,draft=dustDraft) {return Boolean(draft && Object.entries(values).every(([key,value])=>String(draft[key] ?? '')===String(value)));}
function dustActionState(result,busy=false,values=dustDraft || {},repairRequested=dustRepairOpen) {
  if(busy) return {kind:'',label:'Processing…',message:'A job is running. No new processing can start. Stop waits for the current operation.'};
  if(result?.frames && !repairRequested) return {kind:'',label:'Mask ready · Saved',message:'Mask ready · Saved. Inspect or open the mask. Creating a repair is optional.'};
  if(result?.repair_error) return {kind:'',label:'Repair unavailable',message:'The saved repair is missing or changed. Its mask is still usable. Existing files will not be overwritten.'};
  if(result?.repair_complete) return {kind:'',label:'Repair ready',message:'Repair ready. Compare the result before explicitly using it for the pipeline.'};
  if(result?.frames) return {kind:'repair',label:values.detector==='dart' ? 'Compose masked repair' : 'Repair masked areas',message:'Saved mask ready for this repair. Processing starts only on request.'};
  return {kind:'mask',label:values.detector==='dart' ? 'Calculate DART mask & restoration' : 'Calculate mask',message:'No mask for these settings. Calculate a mask first; no repair starts automatically.'};
}
function dustFieldHtml(field) {
  const [key,label,kind]=field,value=dustDraft[key] ?? '';
  if(kind==='checkbox') return `<label><input type="checkbox" data-dust="${key}" ${value==='true'?'checked':''}> ${esc(label)}</label>`;
  const options=kind.startsWith('select:') ? kind.slice(7).split('|').filter(v=>key!=='repair' || v!=='dart') : [];
  const control=options.length ? `<select data-dust="${key}">${options.map(v=>`<option value="${v}" ${value===v?'selected':''}>${esc(dustNames[v] || v)}</option>`).join('')}</select>`
    : `<input data-dust="${key}" type="${kind==='text'?'text':'number'}" ${key==='sensitivity'?'min="0" max="1" step="0.05"':''} value="${esc(value)}">`;
  return `<label>${esc(label)}${control}</label>`;
}
function dustPickerHtml() {
  const selected=denoiseSelectedVariant(),st=state.stages.find(s=>s.key==='dust'),dart=dustDraft.detector==='dart';
  const fields=keys=>st.fields.filter(([key])=>keys.includes(key)).map(dustFieldHtml).join('');
  return `<section class="denoise-variant-picker"><h3 id="denoiseVariantHeading">Variants</h3>
    <output id="denoiseMatrixStatus" role="status"></output>
    <div class="denoise-variant-grid">${denoiseMethods().map(m=>denoiseMatrixCellHtml(m,{key:'settings'},selected)).join('') || '<p>No saved variants for this input.</p>'}</div>
    <section class="dust-parameter-group"><h3>Detection</h3><p>Finds possible defects and marks areas for repair.</p><div class="dust-fields">${fields(dart ? ['detector','threshold'] : ['detector','sensitivity','dilate'])}</div></section>
    ${dustRepairOpen ? `<section class="dust-parameter-group"><h3>Repair</h3><p>${dart ? 'DART creates its own mask and restoration together. The next step composes only the masked areas.' : 'Repairs the marked areas using the saved mask.'}</p><div class="dust-fields">${dart ? '' : fields(dustDraft.repair==='telea' ? ['repair','radius'] : ['repair'])}</div><button type="button" data-dust-close-repair>Return to masks</button></section>` : ''}
    ${['ridge','dart'].includes(dustDraft.detector) ? `<div class="dust-fields">${fields(['research_confirmed'])}</div>` : ''}
    ${dart || dustRepairOpen && dustDraft.repair==='propainter' ? `<p class="inline-warning">${dart ? 'DART is research-only; adapted RTN rights remain unresolved. Mask creation also runs restoration inference.' : 'ProPainter is non-commercial.'}</p><div class="dust-fields">${fields(['license_confirmed'])}</div>` : ''}
    ${dart ? `<details><summary>DART installation · managed by ARP</summary><p>Default: tools/dart with its own Python 3.12 environment. Set up once with install_windows.bat -DartOnly. No ComfyUI node required. Leave both overrides blank to use the managed installation.</p><div class="dust-fields">${fields(['dart_repo','dart_python'])}</div></details>` : ''}
    <p data-dust-draft role="status">Settings are a draft. Saved variants and pipeline selection stay unchanged.</p>
    ${comparisonSelectedHtml(denoiseVariantLabelFor(selected.method,'settings'),denoiseVariantLabelFor(denoiseStageVariant().method,'settings'),Boolean(state.running))}</section>`;
}
function dustSetView(view) {
  dustView=dustInputMode==='input' ? 'result' : 'mask';
  const host=document.querySelector('[data-dust-catalog]');if(!host) return;
  const mask=dustView==='mask',repair=Boolean(dustRepairOpen && dustStatus?.repair_complete);
  host.classList.toggle('dust-has-repair',repair);host.classList.toggle('dust-input-mask',mask);
  host.querySelector('[data-dust-mask-view]').hidden=!mask;
  host.querySelector('#denoiseComparePlayers').hidden=mask && !repair;
  host.querySelector('[data-dust-opacity]').hidden=dustInputMode!=='overlay';
  const play=host.querySelector('#denoiseComparePlay');if(play) {play.disabled=mask && !dustStatus?.frames;play.title=mask?'Play stage input and saved mask on the shared frame clock':'Play repair comparison';}
  host.querySelectorAll('[data-dust-input]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dustInput===dustInputMode)));
  if(mask) {
    dustInvalidateInspection();
  }
  const playbackKey=dustPlaybackIdentity();
  if(host.dataset.maskPlayback!==playbackKey) {host.dataset.maskPlayback=playbackKey;denoiseWorkspaceRefresh();}
  if(mask && !denoiseWorkspaceSnapshot().exact) dustPaintFrame?.();
  dustUpdateInspectionContext();
  dustAlignImages();
}
function dustAlignImages() {
  const host=document.querySelector('.dust-input-mask.dust-has-repair');if(!host) return;
  const grid=host.querySelector('.dust-comparison-images'),caption=grid?.querySelector('[data-dust-images] figcaption');
  const wrap=[...host.querySelectorAll('#denoiseComparePlayers [data-dc-slot]')].find(panel=>Number(panel.dataset.dcSlot)>0 && !panel.hidden)?.querySelector('.denoise-compare-video-wrap');
  if(!caption || !wrap) return;
  caption.style.minHeight=innerWidth>600 ? Math.max(0,wrap.getBoundingClientRect().top-grid.getBoundingClientRect().top)+'px' : '';
}
function dustUpdateInspectionContext() {
  if(active!=='dust') return;
  const host=document.querySelector('[data-dust-catalog]');if(!host) return;
  const rows=state.dust_matrix?.variants || [],summary=host.querySelector('#denoiseMatrixStatus');
  if(summary) summary.textContent=`${new Set(rows.map(row=>row.mask_sha256 || row.method)).size} saved masks · ${rows.filter(row=>row.repair_complete).length} repairs`;
  const selected=denoiseSelectedVariant(),row=state.dust_matrix?.variants?.find(v=>v.method===selected.method);
  if(dustView==='mask' && (!row || !dustMatches(row.values))) {
    const context=host.querySelector('[data-review-context]');if(context) context.textContent=`Inspecting: ${dustStatus?.frames?'saved':'draft'} mask · ${dustNames[dustDraft.detector]} · Pipeline selection unchanged`;
    host.querySelectorAll('[data-method-viewed]').forEach(badge=>badge.hidden=true);
    const label=host.querySelector('[data-dm-selected-label]');if(label) label.textContent='Draft mask · '+dustLabel(dustDraft);
    const use=host.querySelector('#denoiseUseVariant');if(use) use.hidden=true;
  }
  const notice=host.querySelector('#denoiseCompareNotice');
  if(notice && dustView==='mask') notice.textContent=denoiseWorkspaceSnapshot().exact?'Exact mask still · press Play for frame-locked compressed playback.':'Frame-locked mask playback · compressed display. Use Exact still to inspect mask boundaries.';
}
function dustMaskPlaybackActive() {return active==='dust' && dustInputMode!=='input' && Boolean(dustStatus?.frames);}
function dustPlaybackIdentity() {return dustMaskPlaybackActive() ? JSON.stringify([dustContext(),dustDraft]) : '';}
function dustMaskCanvas() {
  const images=document.querySelector('[data-dust-images]');if(!images) return null;
  if(!images.querySelector('canvas')) images.innerHTML='<figure><figcaption></figcaption><div class="dust-image-viewport"><div class="dust-image-layer"><canvas hidden aria-label="Stage input mask inspection"></canvas></div></div></figure>';
  return images.querySelector('canvas');
}
function dustPaintPlayback(atlas,manifest,frame) {
  if(!dustMaskPlaybackActive() || !manifest.regions?.mask) return;
  const identity=dustPlaybackIdentity(),canvas=dustMaskCanvas();if(!canvas) return;
  const paint=()=>{
    if(identity!==dustPlaybackIdentity() || !canvas.isConnected || !atlas.width) return;
    const input=manifest.regions.original,mask=manifest.regions.mask;
    canvas.width=manifest.width;canvas.height=manifest.height;
    const ctx=canvas.getContext('2d');
    ctx.drawImage(atlas,...mask,0,0,canvas.width,canvas.height);
    if(dustInputMode==='overlay') {
      const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
      for(let i=0;i<pixels.data.length;i+=4) {const on=pixels.data[i]>=128;pixels.data[i]=255;pixels.data[i+1]=0;pixels.data[i+2]=255;pixels.data[i+3]=on?Math.round(255*dustOpacity):0;}
      const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;overlay.getContext('2d').putImageData(pixels,0,0);
      ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(atlas,...input,0,0,canvas.width,canvas.height);ctx.drawImage(overlay,0,0);
    }
    canvas.hidden=false;canvas.dataset.dpFrame=String(frame);canvas.dataset.dpQuality='preview';
    canvas.closest('figure').querySelector('figcaption').textContent=`${dustInputMode==='mask'?'Effective mask':'Stage input + magenta mask'} · frame ${frame+1} · compressed display`;
    const status=document.querySelector('[data-dust-frame-status]');if(status) status.textContent=`Saved mask · frame ${frame+1}/${manifest.frames}. Frame-locked compressed playback; no detection or repair.`;
    dustAlignImages();
  };
  paint();dustPaintFrame=paint;
}
async function dustRun() {
  if(state.running || dustRunPending) return;
  const action=dustActionState(dustStatus,false);if(!action.kind) return;
  const context=dustContext(),values={...dustDraft};dustRunPending=true;dustFollowDraft=true;
  const host=document.querySelector('[data-dust-catalog]'),button=host?.querySelector('[data-dust-next]');
  if(button) button.disabled=true;
  try {
    const result=await postJson('/api/dust-run',{values,kind:action.kind,source:state.dust_matrix.source,project:state.project_path || ''});
    if(context!==dustContext() || active!=='dust') return;
    if(!result?.ok) throw Error(result?.error || result?.message || 'Processing could not start.');
    await refresh();
  } catch(error) {if(context===dustContext() && host?.isConnected) host.querySelector('[data-dust-state]').textContent=error.message;}
  finally {dustRunPending=false;if(button?.isConnected) button.disabled=false;}
}
function drawDust(st) {
  const openDialog=document.querySelector('.comparison-inspector-dialog[open]');
  const scroll=document.querySelector('.comparison-inspector-content')?.scrollTop || 0;
  const focus=document.activeElement?.dataset?.dust;
  const project=dustContext();
  const changedProject=!dustDraft || dustProject!==project;
  if(changedProject) {dustDraft={...state.settings.dust};dustProject=project;dustView='result';dustInputMode='input';dustRepairOpen=false;dustStatus=null;dustInspectionKey='';dustInspectEpoch++;}
  const epoch=++dustEpoch;
  const previous=denoiseWorkspaces.has('dust') ? denoiseWorkspaceSnapshot() : null;
  denoiseWorkspaceDeactivate();
  if(dustWorkspaceCatalog!==dustCatalog()) denoiseWorkspaces.delete('dust');
  dustWorkspaceCatalog=dustCatalog();
  drawDenoise(st,state.settings.dust,state.expected_outputs?.dust || [],{percent:0,label:state.running?'Processing':'Idle'});
  if(previous && !changedProject) {denoiseWorkspaceFor().restoreComparison(previous);denoiseWorkspaceRefresh();denoiseWorkspaceFor().seekFrame(previous.frame);denoiseWorkspaceFor().setViewport({zoom:previous.zoom,pan:previous.pan,native:previous.native,grid:previous.grid});}
  const host=document.querySelector('.comparison-workbench');host.classList.add('dust-editor');host.dataset.dustCatalog=dustCatalog();
  dustInspectionKey='';dustInspectEpoch++;dustPaintFrame=null;
  host.querySelector('#denoiseVariantHeading').textContent='Variants';
  host.querySelector('[data-dm-generation]')?.remove();
  host.querySelector('#denoiseCopyVariant')?.remove();
  const use=host.querySelector('#denoiseUseVariant');if(use) use.hidden=true;
  host.querySelector('[onclick="runStage(\'dust\')"]')?.remove();
  const footer=host.querySelector('.comparison-inspector-footer');
  const actions=document.createElement('div');actions.className='dust-next-action';
  actions.innerHTML='<p data-dust-state role="status">Checking saved mask…</p><button type="button" data-dust-next disabled>Checking mask…</button><a data-dust-open-mask hidden target="_blank" rel="noopener">Open saved mask</a><button type="button" data-dust-create-repair hidden>Create repair…</button>';
  footer.prepend(actions);actions.querySelector('button').onclick=dustRun;
  actions.querySelector('[data-dust-create-repair]').onclick=()=>{dustRepairOpen=true;drawDust(st);};
  host.querySelector('[data-dust-close-repair]')?.addEventListener('click',()=>{dustRepairOpen=false;drawDust(st);});
  const details=document.createElement('details');details.className='comparison-input-context';
  details.innerHTML='<summary>Mask measurements & technical details</summary><p data-dust-candidate-context></p><div data-dust-measurements></div><div data-dust-files></div><p>Counts describe proposals and changes, not detection accuracy. Models use BGR8 copies; native 10-bit preservation is verified separately. DART full restoration is diagnostic only.</p>';
  host.querySelector('[data-comparison-analysis]').append(details);
  const viewer=document.createElement('section');viewer.className='dust-mask-review';
  viewer.innerHTML=`<div class="dust-view-switch" role="group" aria-label="Stage input display"><strong>Stage input</strong>${[['input','Input'],['overlay','Mask overlay'],['mask','Mask only']].map(([key,label])=>`<button type="button" data-dust-input="${key}">${label}</button>`).join('')}<label data-dust-opacity hidden>Overlay opacity <input type="range" min="0" max="100" value="${dustOpacity*100}" aria-label="Mask overlay opacity"><output>${Math.round(dustOpacity*100)}%</output></label></div><div class="dust-comparison-images"><div data-dust-mask-view hidden><p data-dust-frame-status role="status">Calculate a mask to inspect marked areas here.</p><button type="button" data-dust-retry hidden>Retry mask frame</button><div class="dust-images" data-dust-images></div></div></div>`;
  host.querySelector('#denoiseComparePlayers').before(viewer);
  viewer.before(viewer.querySelector('[data-dust-frame-status]'));
  viewer.querySelector('.dust-comparison-images').append(host.querySelector('#denoiseComparePlayers'));
  dustAlignmentObserver?.disconnect();
  if(typeof ResizeObserver==='function') {dustAlignmentObserver=new ResizeObserver(dustAlignImages);dustAlignmentObserver.observe(viewer);}
  host.querySelectorAll('[data-dust-input]').forEach(button=>button.onclick=()=>{dustInputMode=button.dataset.dustInput;dustInspectionKey='';dustSetView();});
  host.querySelector('[data-dust-opacity] input').oninput=event=>{dustOpacity=Number(event.target.value)/100;host.querySelector('[data-dust-opacity] output').textContent=Math.round(dustOpacity*100)+'%';dustPaintFrame?.();};
  host.querySelector('[data-dust-retry]').onclick=()=>{if(!denoiseWorkspaceSnapshot().exact) {denoiseWorkspaceFor().playPause();return;}dustInspectionKey='';dustInvalidateInspection();};
  const valid=()=>epoch===dustEpoch && host.isConnected && project===dustContext() && active==='dust';
  let statusEpoch=0;
  async function status() {
    const values=JSON.stringify(dustDraft),request=++statusEpoch;
    try {
      const result=await postJson('/api/dust-status',{values:{...dustDraft},source:state.dust_matrix.source,project:state.project_path || ''});
      if(!valid() || request!==statusEpoch || values!==JSON.stringify(dustDraft)) return;
      if(!result?.ok) throw Error(result?.error || result?.message || 'Could not check this mask.');
      if(result.frames && !dustStatus?.frames) dustInspectionKey='';
      dustStatus=result;
      if(dustDraft.detector==='dart' && result.mask_values) {
        for(const key of ['dart_repo','dart_python']) if(!dustDraft[key]) dustDraft[key]=result.mask_values[key];
      }
      const action=dustActionState(result,state.running || dustRunPending);
      host.querySelector('[data-dust-state]').textContent=state.dust_matrix?.stop_requested ? 'Stop requested. Waiting for the current operation.' : action.message;
      const next=host.querySelector('[data-dust-next]');next.textContent=action.label;next.hidden=!action.kind && !state.running;next.disabled=!action.kind || Boolean(state.running || dustRunPending || state.processing_input?.problem);
      if(result.installation?.available===false && action.kind==='mask') {next.disabled=true;next.textContent='Set up DART first';host.querySelector('[data-dust-state]').textContent=result.installation.message;}
      const row=state.dust_matrix?.variants?.find(v=>dustMatches(v.values));
      const create=host.querySelector('[data-dust-create-repair]');create.hidden=!result.frames || dustRepairOpen;create.disabled=Boolean(state.running);
      const open=host.querySelector('[data-dust-open-mask]');open.hidden=!result.files?.['effective_mask.mkv'];if(!open.hidden) open.href='/media?path='+encodeURIComponent(result.files['effective_mask.mkv']);
      const use=host.querySelector('#denoiseUseVariant');if(use) use.hidden=!row?.repair_complete || denoiseSelectedVariant().method!==row.method;
      if(use && !dustRepairOpen) use.hidden=true;
      host.querySelector('[data-dust-draft]').textContent=row ? 'Settings match a saved variant. Pipeline selection is unchanged.' : 'Draft settings. Saved variants and pipeline selection are unchanged.';
      const context=host.querySelector('[data-review-context]');if(context && (!row || row.method!==denoiseSelectedVariant().method)) context.textContent=`Inspecting: draft mask · ${dustLabel(dustDraft)} · Pipeline selection unchanged`;
      host.querySelectorAll('[data-method-viewed]').forEach(badge=>{if(!row || row.method!==denoiseSelectedVariant().method) badge.hidden=true;});
      host.querySelectorAll('.comparison-method-row').forEach(element=>{
        const candidate=state.dust_matrix?.variants?.find(v=>v.method===element.querySelector('[data-denoise-cell]')?.dataset.method);
        const label=element.querySelector('.denoise-variant-cell-state');
        if(label && candidate) label.textContent=candidate.repair_complete?'Repair ready':candidate.job?.state==='stopped'?'Stopped · resume':candidate.job?.state==='failed'?'Failed · inspect':'Mask ready';
      });
      host.querySelector('[data-dust-files]').innerHTML=Object.entries(result.files || {}).filter(([name])=>/csv|mkv/.test(name)).map(([name,path])=>`<a href="/media?path=${encodeURIComponent(path)}" target="_blank" rel="noopener">${esc(name)}</a>`).join(' · ');
      host.querySelector('[data-dust-candidate-context]').textContent=row ? `Effective mask SHA-256: ${row.mask_sha256}. Saved parameters: ${JSON.stringify(row.values)}` : 'No saved mask for this draft.';
      if(dustFollowDraft && row && !state.running && !dustRunPending) {
        dustFollowDraft=false;inspectComparisonVariant(row.method,'settings');if(!row.repair_complete) dustInputMode='overlay';
      }
      if(!row?.repair_complete && dustView==='result') dustView='mask';
      dustSetView(dustView);
    } catch(error) {if(valid() && request===statusEpoch) host.querySelector('[data-dust-state]').textContent=error.message;}
  }
  dustUpdateStatus=status;
  host.querySelectorAll('[data-dust]').forEach(input=>input.onchange=()=>{
    dustDraft[input.dataset.dust]=input.type==='checkbox'?String(input.checked):input.value;
    if(input.dataset.dust==='detector') {if(input.value==='dart') dustDraft.repair='dart';else if(dustDraft.repair==='dart') dustDraft.repair='telea';}
    dustStatus=null;dustView='mask';dustInspectionKey='';dustInspectEpoch++;drawDust(st);
  });
  host.addEventListener('click',event=>{
    const rowElement=event.target.closest('.comparison-method-row');if(!rowElement) return;
    const row=state.dust_matrix.variants.find(v=>v.method===rowElement.querySelector('[data-denoise-cell]')?.dataset.method);
    if(!row || event.target.closest('[data-denoise-cell]')) return;
    dustDraft={...row.values};dustStatus=null;dustRepairOpen=Boolean(row.repair_complete);dustInputMode=row.repair_complete?'input':'overlay';dustInspectionKey='';drawDust(st);
  });
  if(openDialog && !changedProject) showComparisonInspector();
  const inspector=host.querySelector('.comparison-inspector-content');if(inspector) inspector.scrollTop=scroll;
  if(focus) host.querySelector(`[data-dust="${focus}"]`)?.focus({preventScroll:true});
  status();
}
function dustInvalidateInspection() {
  if(active!=='dust' || dustView!=='mask') return;
  const host=document.querySelector('[data-dust-mask-view]');if(!host) return;
  const snapshot=denoiseWorkspaceSnapshot(),frame=snapshot.frame;
  if(!snapshot.exact) {
    dustInspectEpoch++;dustInspectionKey='';
    const canvas=dustMaskCanvas();
    canvas?.classList.toggle('is-awaiting-frame',Boolean(snapshot.loading || snapshot.error || canvas.dataset.dpFrame!==String(frame)));
    const status=document.querySelector('[data-dust-frame-status]'),retry=host.querySelector('[data-dust-retry]');
    if(status && (snapshot.error || snapshot.loading)) status.textContent=snapshot.error ? snapshot.error+' Retry mask playback or choose Exact still.' : 'Preparing frame-locked saved mask playback…';
    retry.hidden=!snapshot.error;
    return;
  }
  const inspectionKey=()=>JSON.stringify([dustContext(),dustDraft,denoiseWorkspaceSnapshot().frame,dustInputMode]);
  const key=inspectionKey();
  if(key===dustInspectionKey) return;
  dustInspectionKey=key;const epoch=++dustInspectEpoch;
  const images=host.querySelector('[data-dust-images]'),message=document.querySelector('[data-dust-frame-status]');
  const retry=host.querySelector('[data-dust-retry]');retry.hidden=true;
  images.replaceChildren();
  dustPaintFrame=null;
  if(!dustStatus?.frames) {message.textContent='No mask for these settings. Choose Calculate mask to inspect marked areas.';return;}
  message.textContent=`Decoding mask and stage input at frame ${frame+1}…`;
  postJson('/api/dust-inspect',{values:{...dustDraft},source:state.dust_matrix.source,project:state.project_path || '',frame}).then(result=>{
    if(epoch!==dustInspectEpoch || !host.isConnected || dustView!=='mask' || key!==inspectionKey()) return;
    if(!result?.ok) throw Error(result?.error || result?.message || 'Mask frame unavailable.');
    images.innerHTML=`<figure><figcaption>${dustInputMode==='mask'?'Effective mask · white marks detected areas':'Stage input + mask overlay · magenta marks detected areas'} · frame ${frame+1}</figcaption><div class="dust-image-viewport"><div class="dust-image-layer"><canvas hidden aria-label="Stage input mask inspection"></canvas></div></div></figure>`;
    const mask=new Image();mask.src='/media?path='+encodeURIComponent(result.images['effective_mask.mkv']);
    const input=new Image();input.src='/media?path='+encodeURIComponent(result.images['input.mkv']);
    input.onerror=mask.onerror=()=>{if(epoch===dustInspectEpoch && host.isConnected) {message.textContent='Could not load this mask frame. Retry the frame.';retry.hidden=false;}};
    const paint=()=>{
      if(epoch!==dustInspectEpoch || !mask.naturalWidth || !input.naturalWidth) return;
      const canvas=images.querySelector('canvas');canvas.width=mask.naturalWidth;canvas.height=mask.naturalHeight;
      const ctx=canvas.getContext('2d');ctx.drawImage(mask,0,0);const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
      for(let i=0;i<pixels.data.length;i+=4) {const on=pixels.data[i]>=128;pixels.data[i]=255;pixels.data[i+1]=0;pixels.data[i+2]=255;pixels.data[i+3]=on?Math.round(255*dustOpacity):0;}
      ctx.putImageData(pixels,0,0);
      const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;overlay.getContext('2d').putImageData(pixels,0,0);
      ctx.clearRect(0,0,canvas.width,canvas.height);if(dustInputMode==='mask') ctx.drawImage(mask,0,0);else {ctx.drawImage(input,0,0);ctx.drawImage(overlay,0,0);}
      canvas.hidden=false;
      const viewport=denoiseWorkspaceSnapshot();
      denoiseWorkspaceFor().setViewport({zoom:viewport.zoom,pan:viewport.pan,native:viewport.native,grid:viewport.grid});
      dustPaintFrame=paint;
      dustAlignImages();
    };
    mask.onload=paint;input.onload=paint;if(mask.complete && input.complete) paint();
    message.textContent=`Saved mask · exact frame ${frame+1}/${result.frames}. Display only; creating a repair is optional.`;
    const measures=document.querySelector('[data-dust-measurements]');if(measures) measures.textContent=JSON.stringify(result.measurements || {});
  }).catch(error=>{if(epoch===dustInspectEpoch && host.isConnected) {message.textContent=error.message;retry.hidden=false;}});
}
