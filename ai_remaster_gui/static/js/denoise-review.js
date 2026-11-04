/* Exact 8-bit RGB stills share the comparison's canvases, clock and crop. */
let denoiseReviewVariant = '';
let denoiseReviewFrame = 0;
let denoiseReviewFps = 0;
let denoiseReviewMetrics = null;
let denoiseReviewEntries = [];
let denoiseReviewChartEpoch = 0;
let denoiseReviewChartKey = '';

function denoiseReviewErrorText(error) {
  return String(error?.message || error || 'The exact still could not be loaded.')
    .replace(/Ungültige Differenzdaten oder Kontrasteinstellungen\.?/i,'The pixel-difference settings are invalid.')
    .replace(/Bildgrößen stimmen nicht überein\.?/i,'The still images do not have identical dimensions.')
    .replace(/Nicht deckungsgleiche Einzelbilder\.?/i,'The still images are not aligned.')
    .replace(/Canvas nicht verfügbar\.?/i,'This browser cannot provide a difference canvas.').slice(0,300);
}

function denoiseReviewVariants(candidates) {
  const items=candidates || [], hasKinds=items.some(item=>item.kind);
  return items.filter((item,index)=>(hasKinds ? item.kind!=='source' : index>0) && denoisePreviewIsFinal(item.path));
}

function denoiseReviewSyncSelection(selected) {
  const path=selected && selected.method!=='off' && denoiseVariantReady(selected.method,selected.preset)
    ? denoiseVariantPath(selected.method,selected.preset) : '';
  const row=denoiseMatrixState().variants?.find(item=>denoiseSameVariant(item,selected));
  const key=JSON.stringify([comparisonStageKey(),path,denoiseMatrixState().source_revision,row?.revision]);
  if(key===denoiseReviewChartKey) return;
  denoiseReviewChartKey=key;
  denoiseReviewVariant=path;
  denoiseReviewMetrics=null;
  denoiseReviewEntries=[];
  denoiseReviewChartEpoch++;
  const chart=document.querySelector('[data-dr-chart]');
  chart?.replaceChildren();
  const status=document.querySelector('[data-dr-chart-status]');
  if(status) status.textContent=path ? 'Calculating luminance changes…' : 'Choose a finished variant.';
  denoiseCompareUpdateChartChoices();
  if(path) void denoiseReviewLoadChart(path);
  denoiseWorkspaceStill?.draw();
}

function denoiseReviewStart() {
  denoiseReviewVariant='';
  denoiseReviewChartKey='';
  denoiseReviewMetrics=null;
  denoiseReviewChartEpoch++;
  const chart=document.querySelector('[data-dr-chart]');
  chart?.addEventListener('keydown',event=>{
    const keys={ArrowLeft:-1,ArrowRight:1,Home:0,End:0,PageUp:-10,PageDown:10};
    if(!(event.key in keys)) return;
    const bars=[...chart.querySelectorAll('[data-dr-bar]')];
    if(!bars.length) return;
    const current=Math.max(0,bars.findIndex(bar=>bar.tabIndex===0));
    const next=event.key==='Home' ? 0 : event.key==='End' ? bars.length-1
      : Math.max(0,Math.min(bars.length-1,current+keys[event.key]));
    event.preventDefault();bars[next].focus();bars[next].click();
  });
  denoiseReviewSyncSelection(denoiseSelectedVariant());
}

function denoiseReviewChartRows(report) {
  if(!report || report.metric!=='directional_mean_luma_8bit' ||
    !Number.isInteger(report.frames) || report.frames<1 || !Array.isArray(report.samples) ||
    !report.samples.length || report.samples.length>240) throw new Error('Invalid frame report.');
  let prev=-1,peak=0;
  const rows=report.samples.map(item=>{
    const frame=Number(item.frame),bright=Number(item.brighten_mean_luma);
    const dark=Number(item.darken_mean_luma),total=Number(item.mean_abs_luma);
    if(!Number.isInteger(frame) || frame<=prev || frame>=report.frames ||
      ![bright,dark,total].every(value=>Number.isFinite(value) && value>=0 && value<=255) ||
      Math.abs(total-bright-dark)>.001) throw new Error('Invalid frame measurements.');
    prev=frame;peak=Math.max(peak,bright,dark);
    return {frame,bright,dark,total};
  });
  return rows.map(item=>({...item,brightHeight:item.bright ? Math.max(2,100*item.bright/(peak||1)) : 0,
    darkHeight:item.dark ? Math.max(2,100*item.dark/(peak||1)) : 0}));
}

function denoiseReviewHighlight(frame=denoiseWorkspacePlayback?.state.frame ?? denoiseReviewFrame) {
  const chart=document.querySelector('[data-dr-chart]');
  if(!chart) return;
  const bars=[...chart.querySelectorAll('[data-dr-bar]')];
  const nearest=bars.reduce((best,bar)=>!best || Math.abs(Number(bar.dataset.drBar)-frame)<Math.abs(Number(best.dataset.drBar)-frame) ? bar : best,null);
  for(const bar of bars) {
    const active=Number(bar.dataset.drBar)===frame;
    bar.classList.toggle('is-current',active);bar.tabIndex=bar===nearest ? 0 : -1;
    bar.setAttribute('aria-current',String(active));
  }
  const selected=denoiseReviewEntries.find(item=>item.frame===frame);
  const readout=document.querySelector('[data-dr-chart-status]');
  if(readout && selected) readout.textContent=`Frame ${frame+1}: brightened ${selected.bright.toFixed(2)} / 255 · darkened ${selected.dark.toFixed(2)} / 255`;
  else if(readout && denoiseReviewEntries.length) readout.textContent=`Frame ${frame+1}: not in the chart sample; exact stills are available on request.`;
}

async function denoiseReviewLoadChart(path) {
  const ticket=++denoiseReviewChartEpoch,chart=document.querySelector('[data-dr-chart]');
  if(!chart) return;
  const status=document.querySelector('[data-dr-chart-status]');
  try {
    const response=await fetch('/api/denoise-frame-differences?variant='+encodeURIComponent(path)+'&stage='+comparisonStageKey(),{cache:'no-store'});
    if(!response.ok) throw new Error('The chart endpoint answered with HTTP '+response.status+'.');
    const report=await response.json();
    if(!report?.ok) throw new Error(report?.error || 'The chart is unavailable.');
    if(ticket!==denoiseReviewChartEpoch || path!==denoiseReviewVariant || !chart.isConnected) return;
    denoiseReviewEntries=denoiseReviewChartRows(report);
    const parts=document.createDocumentFragment();
    for(const item of denoiseReviewEntries) {
      const button=document.createElement('button');
      button.type='button';button.className='denoise-review-bar';button.dataset.drBar=String(item.frame);button.tabIndex=-1;
      button.style.setProperty('--bright-height',item.brightHeight.toFixed(2)+'%');
      button.style.setProperty('--dark-height',item.darkHeight.toFixed(2)+'%');
      button.title=`Frame ${item.frame+1} · bright +${item.bright.toFixed(2)} · dark −${item.dark.toFixed(2)} / 255`;
      button.setAttribute('aria-label',button.title);
      button.addEventListener('click',()=>{
        denoiseWorkspacePlayback?.pause();
        void denoiseWorkspacePlayback?.seekFrame(item.frame);
      });
      parts.appendChild(button);
    }
    chart.replaceChildren(parts);
    chart.title=report.sampled ? 'Up to 240 real frame samples, not a complete film scan' : 'Click a bar to move the shared comparison frame';
    denoiseReviewHighlight();
  }catch(error) {
    if(ticket===denoiseReviewChartEpoch && chart.isConnected && status) status.textContent='Chart unavailable: '+denoiseReviewErrorText(error);
  }
}

function denoiseReviewLoadImage(path) {
  return new Promise((resolve,reject)=>{
    if(typeof path!=='string' || !path) {reject(new Error('The server did not return a still image path.'));return;}
    const image=new Image();
    image.onload=()=>resolve(image);
    image.onerror=()=>reject(new Error('The decoded still image could not be loaded.'));
    image.src='/media?path='+encodeURIComponent(path);
  });
}

function denoiseReviewPixels(pair, mode, contrast) {
  const canvas=document.createElement('canvas');
  canvas.width=pair.before.naturalWidth;canvas.height=pair.before.naturalHeight;
  const context=canvas.getContext('2d',{willReadFrequently:true});
  if(!context) throw new Error('This browser cannot provide a difference canvas.');
  context.drawImage(pair.before,0,0);
  const a=context.getImageData(0,0,canvas.width,canvas.height);
  context.drawImage(pair.after,0,0);
  const b=context.getImageData(0,0,canvas.width,canvas.height);
  if(!pair.metrics) {
    const measured=cleanupDifferencePixels(a.data,b.data,canvas.width,canvas.height,8);
    pair.metrics={mean:measured.mean,changedPercent:measured.changedPercent};
  }
  const pixels=comparisonContrastPixels(a.data,b.data,canvas.width,canvas.height,mode,contrast);
  context.putImageData(new ImageData(pixels,canvas.width,canvas.height),0,0);
  return canvas;
}

function createDenoiseStillReview(host,options) {
  const state={loading:false,error:'',frame:-1,pairs:new Map()};
  let epoch=0,key='',inFlight=false,pending=null,destroyed=false,drag=null;
  const wipes=options.wipes || new Map(),abort=new AbortController();
  function clear() {
    for(const pair of state.pairs.values()) if(pair.difference) {pair.difference.width=0;pair.difference.height=0;}
    state.pairs.clear();denoiseReviewMetrics=null;
  }
  function current(ticket) {return !destroyed && ticket===epoch && options.playback().state.exact && host.isConnected;}
  function notify() {options.changed();}
  function updateWipes() {
    host.querySelectorAll('[data-ds-wipe]').forEach(handle=>{
      const slot=Number(handle.dataset.dsWipe),canvas=host.querySelector(`[data-dp-canvas="${slot}"]`);
      const enabled=options.playback().state.exact && !state.loading && !state.error && state.pairs.has(slot) && options.view(slot)==='wipe';
      handle.hidden=!enabled;
      if(!enabled || !canvas) return;
      const percent=wipes.get(slot) ?? 50,rect=canvas.getBoundingClientRect();
      const bounds=canvas.closest('.denoise-compare-video-wrap').getBoundingClientRect();
      const left=rect.left-bounds.left+rect.width*percent/100;
      handle.style.left=left+'px';
      handle.hidden=left<0 || left>bounds.width;
      handle.setAttribute('aria-valuenow',String(percent));
      handle.setAttribute('aria-valuetext',comparisonWipeVisibleState(percent,'RESULT','en'));
      const grip=handle.firstElementChild;
      if(grip) grip.style.opacity=String(comparisonWipeGripOpacity(rect.width,grip.offsetWidth||120,percent));
    });
  }
  function draw() {
    const playback=options.playback().state;
    if(!playback.exact || state.loading || state.error || state.frame!==playback.frame) {updateWipes();return;}
    const reference=state.pairs.values().next().value;
    if(!reference) return;
    const paint=(slot,pair,mode)=>{
      const canvas=host.querySelector(`[data-dp-canvas="${slot}"]`);
      if(!canvas) return;
      const width=pair.before.naturalWidth,height=pair.before.naturalHeight;
      if(canvas.width!==width || canvas.height!==height) {canvas.width=width;canvas.height=height;}
      const context=canvas.getContext('2d',{alpha:false});
      if(mode==='difference') {
        const diffKey=options.diffMode()+'|'+options.contrast();
        if(pair.diffKey!==diffKey) {
          if(pair.difference) {pair.difference.width=0;pair.difference.height=0;}
          pair.difference=denoiseReviewPixels(pair,options.diffMode(),options.contrast());pair.diffKey=diffKey;
        }
        context.drawImage(pair.difference,0,0);
      }else {
        context.drawImage(mode==='original' || mode==='wipe' ? pair.before : pair.after,0,0);
        if(mode==='wipe') {
          const edge=Math.round(width*(wipes.get(slot) ?? 50)/100);
          if(edge<width) context.drawImage(pair.after,edge,0,width-edge,height,edge,0,width-edge,height);
        }
      }
      canvas.dataset.dpFrame=String(state.frame);canvas.dataset.dpQuality='exact';
      if(options.path(slot)===denoiseReviewVariant) {denoiseReviewMetrics=pair.metrics;denoiseReviewFrame=state.frame;}
    };
    try {
      paint(0,reference,'original');
      for(const [slot,pair] of state.pairs) if(slot) paint(slot,pair,options.view(slot));
      options.painted();updateWipes();
    }catch(error) {state.error=denoiseReviewErrorText(error);notify();}
  }
  async function load(request) {
    inFlight=true;
    const pairs=new Map();
    try {
      let reference=null;
      for(const slot of request.slots) {
        if(!current(request.ticket)) return;
        const response=await fetch('/api/denoise-inspector?variant='+encodeURIComponent(options.path(slot))+'&frame='+request.frame+(options.stage?.()==='deflicker' ? '&stage=deflicker' : ''),{cache:'no-store'});
        if(!response.ok) throw new Error('The inspection endpoint answered with HTTP '+response.status+'.');
        const data=await response.json();
        if(!data?.ok) throw new Error(data?.error || 'The exact frame is unavailable.');
        if(data.exact!==true || data.frame!==request.frame || !Number.isInteger(data.frames) || data.frames<1 || data.frames>2147483647 ||
          data.frame>=data.frames || !(data.fps>0 && Number.isFinite(data.fps)) || !Number.isInteger(data.width) || data.width<1 ||
          !Number.isInteger(data.height) || data.height<1 || typeof data.before!=='string' || !data.before || typeof data.after!=='string' || !data.after)
          throw new Error('The server returned invalid exact-frame metadata.');
        if(reference && ['before','frames','fps','width','height'].some(name=>reference[name]!==data[name]))
          throw new Error('The exact comparison variants have different frame mappings.');
        if(!current(request.ticket)) return;
        const before=reference ? pairs.values().next().value.before : await denoiseReviewLoadImage(data.before);
        const after=data.after===data.before ? before : await denoiseReviewLoadImage(data.after);
        if([before,after].some(image=>image.naturalWidth!==data.width || image.naturalHeight!==data.height))
          throw new Error('The decoded still dimensions do not match the frame metadata.');
        if(!current(request.ticket)) return;
        const pair={before,after,metrics:null,difference:null,diffKey:''};
        if(slot) {
          pair.difference=denoiseReviewPixels(pair,options.diffMode(),options.contrast());
          pair.diffKey=options.diffMode()+'|'+options.contrast();
        }
        pairs.set(slot,pair);reference=data;
      }
      if(!current(request.ticket)) return;
      state.pairs=pairs;state.frame=request.frame;state.loading=false;state.error='';
      const playback=options.playback().state;
      if(reference) {playback.count=reference.frames;playback.fps=reference.fps;denoiseReviewFps=reference.fps;}
      draw();notify();
    }catch(error) {
      if(current(request.ticket)) {state.loading=false;state.error=denoiseReviewErrorText(error);clear();notify();}
    }finally {
      if(state.pairs!==pairs) for(const pair of pairs.values()) if(pair.difference) {pair.difference.width=0;pair.difference.height=0;}
      inFlight=false;
      const next=pending;pending=null;
      if(next && current(next.ticket)) void load(next);
    }
  }
  function reconcile(retry=false) {
    const playback=options.playback().state,slots=options.slots();
    const next=playback.exact ? JSON.stringify([playback.frame,slots.map(slot=>[slot,options.identity(slot),options.path(slot)])]) : 'preview';
    if(!retry && next===key) {if(playback.exact) draw();return;}
    key=next;epoch++;pending=null;clear();state.frame=-1;state.error='';
    state.loading=Boolean(playback.exact && slots.length);
    if(state.loading) {
      const request={frame:playback.frame,slots,ticket:epoch};
      if(inFlight) pending=request;else void load(request);
    }
    updateWipes();notify();
  }
  function setWipe(slot,value) {wipes.set(slot,Math.max(0,Math.min(100,Number(value)||0)));draw();}
  host.addEventListener('input',event=>{
    const slider=event.target.closest('[data-ds-wipe-range]');
    if(slider) {setWipe(Number(slider.dataset.dsWipeRange),slider.value);notify();}
  },{signal:abort.signal});
  host.addEventListener('pointerdown',event=>{
    const handle=event.target.closest('[data-ds-wipe]');
    if(!handle || handle.hidden || drag || event.pointerType==='mouse' && event.button!==0) return;
    event.preventDefault();event.stopPropagation();
    try {handle.setPointerCapture(event.pointerId);}catch(_) {return;}
    drag={handle,id:event.pointerId};
  },{signal:abort.signal,capture:true});
  host.addEventListener('pointermove',event=>{
    if(!drag || drag.id!==event.pointerId) return;
    const slot=Number(drag.handle.dataset.dsWipe),canvas=host.querySelector(`[data-dp-canvas="${slot}"]`);
    const rect=canvas.getBoundingClientRect();
    if(rect.width>0) setWipe(slot,100*(event.clientX-rect.left)/rect.width);
    event.preventDefault();event.stopPropagation();notify();
  },{signal:abort.signal,capture:true});
  function end(event) {
    if(!drag || event.pointerId!=null && event.pointerId!==drag.id) return;
    const {handle,id}=drag;drag=null;
    if(handle.hasPointerCapture(id)) handle.releasePointerCapture(id);
  }
  for(const name of ['pointerup','pointercancel','lostpointercapture']) host.addEventListener(name,end,{signal:abort.signal,capture:true});
  if(typeof window!=='undefined') window.addEventListener('blur',end,{signal:abort.signal});
  host.addEventListener('keydown',event=>{
    const handle=event.target.closest('[data-ds-wipe]');
    const keys={ArrowLeft:-1,ArrowRight:1,PageUp:-10,PageDown:10,Home:0,End:100};
    if(!handle || !(event.key in keys)) return;
    event.preventDefault();event.stopPropagation();
    const slot=Number(handle.dataset.dsWipe);
    setWipe(slot,event.key==='Home' || event.key==='End' ? keys[event.key] : (wipes.get(slot) ?? 50)+keys[event.key]);notify();
  },{signal:abort.signal,capture:true});
  return {state,reconcile,draw,updateWipes,wipe:slot=>wipes.get(slot) ?? 50,
    destroy(){destroyed=true;epoch++;pending=null;end({});abort.abort();clear();}};
}
