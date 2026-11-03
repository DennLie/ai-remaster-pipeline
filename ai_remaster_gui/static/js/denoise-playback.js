/* Publish only common decoded frame ordinals. Each reel contains all three views. */
function denoisePlaybackOrdinal(manifest, mediaTime) {
  const frame = manifest.first_frame + Math.round(Number(mediaTime) * manifest.fps);
  return Number.isFinite(frame) && frame >= manifest.first_frame && frame < manifest.end_frame ? frame : null;
}

function denoisePlaybackCommonFrame(queues, minimum, paused) {
  if (!queues.length) return null;
  const common = [...queues[0].keys()].filter(frame => frame >= minimum &&
    (!paused || frame === minimum) && queues.every(queue => queue.has(frame)));
  return common.length ? Math.max(...common) : null;
}

function denoisePlaybackTime(seconds) {
  const milliseconds = Math.max(0,Math.round((Number(seconds) || 0)*1000));
  const hours = Math.floor(milliseconds/3600000);
  const minutes = Math.floor(milliseconds/60000)%60;
  const rest = (milliseconds%60000/1000).toFixed(3).padStart(6,'0');
  return (hours ? String(hours).padStart(2,'0')+':' : '') + String(minutes).padStart(2,'0')+':'+rest;
}

function createDenoisePlayback(host, options) {
  const state = {frame:0, fps:options.fps || 30, count:Math.max(1,options.count || 1),
    playing:false, loading:false, error:'', exact:Boolean(options.exact), views:new Map()};
  let epoch=0, streams=[], next=[], abort=null, key='', requested=0, transition=false, destroyed=false;
  let recovering=false, watchdog=null;
  const manifests=new Map();
  const streamKey=slot=>JSON.stringify([options.identity(slot),options.path(slot),options.diffMode(),options.contrast()]);
  const decoders=document.createElement('div');
  decoders.className='denoise-playback-decoders';
  decoders.setAttribute('aria-hidden','true');
  host.appendChild(decoders);

  function notify(message='') { options.changed?.(state,message); }
  function dispose(stream) {
    stream.dead=true;
    stream.video.pause();
    if (stream.callback != null) stream.video.cancelVideoFrameCallback(stream.callback);
    stream.video.removeAttribute('src'); stream.video.load(); stream.video.remove();
    for(const image of new Set([...stream.queue.values(),stream.presented])) if(image) {image.width=0;image.height=0;}
    stream.queue.clear(); stream.presented=null;
  }
  function stopDecoders() { for(const stream of streams) stream.video.pause(); }
  function fail(error, version=epoch) {
    if(destroyed || version!==epoch || error?.name==='AbortError') return;
    state.error=error?.message || 'Comparison playback failed.';
    manifests.clear();
    state.playing=false; state.loading=false; stopDecoders();
    notify(state.error+' Retry playback, or deselect the affected variant.');
  }
  function draw() {
    if(state.exact) return;
    const first=streams[0];
    if(!first?.presented) return;
    let resized=false;
    const paint=(slot,stream,mode)=>{
      const canvas=host.querySelector(`[data-dp-canvas="${slot}"]`);
      const region=stream.manifest.regions[mode];
      if(!canvas || !region || !stream.presented) return;
      if(canvas.width!==region[2] || canvas.height!==region[3]) {
        canvas.width=region[2]; canvas.height=region[3];
        resized=true;
      }
      canvas.getContext('2d',{alpha:false}).drawImage(stream.presented,...region,0,0,region[2],region[3]);
      canvas.dataset.dpFrame=String(state.frame);
      canvas.dataset.dpQuality='preview';
    };
    paint(0,first,'original');
    for(const stream of streams) if(stream.slot) {
      const mode=state.views.get(stream.slot) || 'result';
      paint(stream.slot,stream,mode==='wipe' ? 'result' : mode);
    }
    if(resized) options.painted?.();
  }
  function publish() {
    if(!streams.length || streams.some(stream=>stream.dead)) return;
    const frame=denoisePlaybackCommonFrame(streams.map(stream=>stream.queue),requested,state.loading || !state.playing);
    if(frame===null) return;
    if(!state.loading && streams.every(stream=>stream.presented===stream.queue.get(frame))) return;
    for(const stream of streams) {
      const old=stream.presented;
      stream.presented=stream.queue.get(frame);
      stream.presentedFrame=frame;
      for(const [ordinal,image] of stream.queue) if(ordinal<frame) {
        stream.queue.delete(ordinal); if(image!==stream.presented) {image.width=0;image.height=0;}
      }
      if(old && old!==stream.presented && ![...stream.queue.values()].includes(old)) {old.width=0;old.height=0;}
    }
    state.frame=frame; requested=frame; state.loading=false; recovering=false;
    draw(); notify();
    if(state.playing) {
      startDecoders();
      prefetch();
      if(frame>=streams[0].manifest.end_frame-1) void advance();
    }
  }
  function capture(stream,metadata) {
    const ordinal=denoisePlaybackOrdinal(stream.manifest,metadata.mediaTime);
    if(ordinal===null || ordinal<requested) return;
    if(stream.seekingTarget!=null) {
      if(ordinal!==stream.seekingTarget) return;
      stream.seekingTarget=null;
    }
    if(stream.queue.has(ordinal)) return;
    while(stream.queue.size>=3) {
      const [old,image]=stream.queue.entries().next().value;
      stream.queue.delete(old);
      if(image!==stream.presented) {image.width=0;image.height=0;}
    }
    const image=document.createElement('canvas');
    image.width=stream.manifest.atlas_width; image.height=stream.manifest.atlas_height;
    image.getContext('2d',{alpha:false}).drawImage(stream.video,0,0);
    stream.queue.set(ordinal,image);
    if(streams.includes(stream)) {
      publish();
      const latest=streams.map(s=>Math.max(-1,...s.queue.keys()));
      if(state.playing && !state.loading && !recovering && Math.min(...latest)>=0 &&
          Math.max(...latest)-Math.min(...latest)>2) {
        recovering=true;
        const target=Math.max(state.frame+1,Math.min(...latest)), version=epoch;
        queueMicrotask(()=>{
          if(!destroyed && version===epoch && recovering && state.playing && state.frame<target) void seekFrame(target);
        });
      }
    }
  }
  async function manifestFor(slot,frame,signal) {
    const body={variant:options.path(slot), frame, mode:options.diffMode(), contrast:options.contrast()};
    const cacheKey=JSON.stringify([options.identity(slot),body.variant,body.mode,body.contrast]);
    const cached=[...manifests.values()].find(v=>v.key===cacheKey && frame>=v.data.first_frame && frame<v.data.end_frame);
    if(cached) return cached.data;
    const response=await fetch('/api/denoise-playback',{method:'POST',signal,
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const result=await response.json();
    if(!result.ok) throw new Error(result.error || 'The comparison could not be prepared.');
    if(!(result.fps>0 && result.frames>0 && result.first_frame<=frame && result.end_frame>frame &&
         result.width>0 && result.height>0 && result.regions?.original &&
         (!slot || result.regions.result && result.regions.difference))) throw new Error('Invalid comparison frame mapping.');
    manifests.set(cacheKey+'|'+result.first_frame,{key:cacheKey,data:result});
    if(manifests.size>32) manifests.delete(manifests.keys().next().value);
    return result;
  }
  async function prepare(slot,frame,signal,version) {
    const manifest=await manifestFor(slot,frame,signal);
    if(version!==epoch || destroyed) throw new DOMException('Superseded','AbortError');
    const video=document.createElement('video');
    video.muted=true; video.playsInline=true; video.preload='auto';
    video.tabIndex=-1;
    const stream={slot,manifest,video,queue:new Map(),presented:null,callback:null,dead:false,version,key:streamKey(slot)};
    decoders.appendChild(video);
    const onFrame=(_now,metadata)=>{
      if(stream.dead || destroyed || stream.version!==epoch) return;
      try {capture(stream,metadata);} catch(error) {fail(error,stream.version);}
      if(!stream.dead) stream.callback=video.requestVideoFrameCallback(onFrame);
    };
    if(typeof video.requestVideoFrameCallback!=='function') {
      dispose(stream); throw new Error('Frame-locked playback needs a browser with requestVideoFrameCallback support.');
    }
    stream.callback=video.requestVideoFrameCallback(onFrame);
    try {
      await new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>done(new Error('The comparison video did not load.')),45000);
        const ready=()=>done();
        const error=()=>done(new Error('The browser could not decode this native-size comparison.'));
        const cancelled=()=>done(new DOMException('Superseded','AbortError'));
        function done(error) {
          clearTimeout(timer); video.removeEventListener('loadeddata',ready); video.removeEventListener('error',errorHandler);
          signal.removeEventListener('abort',cancelled); error?reject(error):resolve();
        }
        const errorHandler=error;
        video.addEventListener('loadeddata',ready,{once:true}); video.addEventListener('error',errorHandler,{once:true});
        signal.addEventListener('abort',cancelled,{once:true});
        video.src=manifest.url; video.load();
        if(signal.aborted) cancelled();
      });
      video.addEventListener('error',()=>{if(!stream.dead) fail(new Error('A comparison decoder failed.'),stream.version);});
      video.addEventListener('ended',()=>{publish();if(state.playing && !state.loading) void advance();});
      return stream;
    } catch(error) {dispose(stream);throw error;}
  }
  function position(stream,frame) {
    if(stream.queue.has(frame) && Math.floor(stream.video.currentTime*stream.manifest.fps)===frame-stream.manifest.first_frame) {
      stream.seekingTarget=null;
      return;
    }
    // Choose the interior of the encoded frame, not its floating-point boundary.
    stream.seekingTarget=frame;
    stream.video.currentTime=(frame-stream.manifest.first_frame+.25)/stream.manifest.fps;
  }
  function startDecoders() {
    if(!state.playing || state.loading || state.error || document.hidden) return;
    for(const stream of streams) if(stream.video.paused && !stream.video.ended)
      stream.video.play().catch(error=>fail(error));
    clearTimeout(watchdog);
    const frame=state.frame, version=epoch;
    watchdog=setTimeout(()=>{
      if(!destroyed && version===epoch && state.playing && !state.loading && frame===state.frame)
        fail(new Error('Frame-locked playback stalled. Try fewer visible variants.'));
    },5000);
  }
  function prefetch() {
    if(document.hidden || next.length || !streams.length || streams[0].manifest.end_frame>=state.count || !abort) return;
    const version=epoch, frame=streams[0].manifest.end_frame;
    next=streams.map(stream=>prepare(stream.slot,frame,abort.signal,version));
    // Failed speculative work must not invalidate the current visible frame.
    for(const promise of next) promise.catch(()=>{});
  }
  async function advance() {
    if(transition || !streams.length || state.frame<streams[0].manifest.end_frame-1) return;
    if(state.frame>=state.count-1) {pause();return;}
    transition=true;
    const version=epoch, target=streams[0].manifest.end_frame;
    stopDecoders(); state.loading=true; requested=target; notify('Buffering the next frames…');
    try {
      if(!next.length) prefetch();
      const prepared=await Promise.all(next); next=[];
      if(version!==epoch || destroyed) {prepared.forEach(dispose);return;}
      streams.forEach(dispose); streams=prepared;
      for(const stream of streams) position(stream,target);
      publish();
    } catch(error) {fail(error,version);}
    finally {if(version===epoch) transition=false;}
  }
  async function seekFrame(value) {
    const target=Math.max(0,Math.min(state.count-1,Math.round(Number(value)||0)));
    const slots=options.slots();
    requested=target; state.error=''; state.loading=Boolean(slots.length) && !state.exact;
    recovering=false;
    state.frame=target;
    stopDecoders(); clearTimeout(watchdog);
    const version=++epoch;
    abort?.abort(); abort=new AbortController();
    for(const promise of next) promise.then(dispose,()=>{});
    next=[]; transition=false;
    const old=streams; streams=[];
    const retireOld=()=>old.forEach(stream=>{if(stream.version<=version && !streams.includes(stream)) dispose(stream);});
    if(state.exact) {retireOld();notify();return;}
    notify(slots.length?'Verifying and preparing frame '+(target+1)+'…':'');
    if(!slots.length) {retireOld();state.loading=false;notify();return;}
    try {
      const prepared=await Promise.allSettled(slots.map(slot=>{
        const reusable=old.find(stream=>stream.slot===slot && !stream.dead && stream.key===streamKey(slot) &&
          target>=stream.manifest.first_frame && target<stream.manifest.end_frame);
        if(reusable) {
          reusable.version=version;
          const retained=reusable.queue.get(target) || (reusable.presentedFrame===target ? reusable.presented : null);
          for(const image of new Set([...reusable.queue.values(),reusable.presented]))
            if(image && image!==retained) {image.width=0;image.height=0;}
          reusable.queue.clear(); reusable.presented=null;
          if(retained) reusable.queue.set(target,retained);
          return reusable;
        }
        return prepare(slot,target,abort.signal,version);
      }));
      const ready=prepared.filter(p=>p.status==='fulfilled').map(p=>p.value);
      const failed=prepared.find(p=>p.status==='rejected');
      if(failed || version!==epoch || destroyed) {
        ready.filter(stream=>stream.version===version).forEach(dispose);
        retireOld();
        if(failed) throw failed.reason;
        return;
      }
      const reference=ready[0].manifest;
      if(ready.some(s=>s.manifest.frames!==reference.frames || s.manifest.fps_fraction!==reference.fps_fraction ||
        s.manifest.first_frame!==reference.first_frame || s.manifest.end_frame!==reference.end_frame)) {
        ready.forEach(dispose);throw new Error('Comparison variants have different frame mappings.');
      }
      state.count=reference.frames; state.fps=reference.fps;
      streams=ready;
      retireOld();
      for(const stream of streams) position(stream,target);
      publish();
    } catch(error) {retireOld();fail(error,version);}
  }
  function reconcile() {
    const current=JSON.stringify([options.slots().map(slot=>[slot,options.path(slot),options.identity(slot)]),options.diffMode(),options.contrast()]);
    if(current===key) return;
    key=current;
    void seekFrame(state.frame);
  }
  function pause() {state.playing=false;recovering=false;stopDecoders();clearTimeout(watchdog);notify();}
  function toggle() {
    if(state.playing) {pause();return;}
    if(state.exact) {
      if(state.frame>=state.count-1) state.frame=0;
      setExact(false);state.playing=true;notify();return;
    }
    state.playing=true;
    if(state.error || !streams.length || state.frame>=state.count-1) void seekFrame(state.frame>=state.count-1?0:state.frame);
    else {startDecoders();prefetch();notify();}
  }
  function setView(slot,view) {
    if(!['original','result','difference','wipe'].includes(view)) return;
    state.views.set(slot,view); draw();
  }
  const visibility=()=>{
    if(document.hidden) {stopDecoders();clearTimeout(watchdog);}
    else if(state.playing) void seekFrame(state.frame);
  };
  document.addEventListener?.('visibilitychange',visibility);
  function setExact(enabled) {
    if(state.exact===Boolean(enabled)) return;
    pause(); state.exact=Boolean(enabled); key='';
    void seekFrame(state.frame);
  }
  return {host,state,reconcile,seekFrame,seekSeconds(value){return seekFrame(Number(value)*state.fps);},
    step(delta){pause();return seekFrame(state.frame+delta);},pause,toggle,setView,setExact,draw,
    destroy(){destroyed=true;epoch++;abort?.abort();clearTimeout(watchdog);streams.forEach(dispose);
      document.removeEventListener?.('visibilitychange',visibility);
      for(const promise of next) promise.then(dispose,()=>{});decoders.remove();}};
}
