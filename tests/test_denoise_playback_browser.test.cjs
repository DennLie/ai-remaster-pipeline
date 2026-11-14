const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const script=fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/denoise-playback.js'),'utf8');
const settle=async()=>{for(let i=0;i<5;i++)await new Promise(setImmediate)};

function setup(options={}) {
  const videos=[],canvases=new Map(),requests=[];
  let slots=[1,2];
  const element=()=>({children:[],className:'',setAttribute(){},appendChild(child){this.children.push(child)},remove(){}});
  const output=slot=>({dataset:{},width:0,height:0,getContext(){return{drawImage(source,...region){this.owner.last={source,region}},owner:this}}});
  for(const slot of [0,1,2])canvases.set(slot,output(slot));
  class Video {
    constructor(){this.listeners=new Map();this.callbacks=new Map();this.id=videos.length;this.readyState=0;this.paused=true;this.ended=false;this._time=0;videos.push(this)}
    addEventListener(name,callback){if(!this.listeners.has(name))this.listeners.set(name,new Set());this.listeners.get(name).add(callback)}
    removeEventListener(name,callback){this.listeners.get(name)?.delete(callback)}
    emit(name){for(const callback of [...this.listeners.get(name)||[]])callback()}
    requestVideoFrameCallback(callback){const id=Math.random();this.callbacks.set(id,callback);return id}
    cancelVideoFrameCallback(id){this.callbacks.delete(id)}
    frame(ordinal){const pair=this.callbacks.entries().next().value;if(pair){this.callbacks.delete(pair[0]);pair[1](0,{mediaTime:ordinal/30})}}
    set currentTime(value){this._time=value;queueMicrotask(()=>this.frame(Math.floor(value*30)))}
    get currentTime(){return this._time}
    load(){if(this.src)queueMicrotask(()=>{this.readyState=2;this.emit('loadeddata');this.frame(0)})}
    pause(){this.paused=true}
    play(){this.paused=false;return Promise.resolve()}
    removeAttribute(){this.src=''}
    remove(){this.removed=true}
  }
  const host={...element(),querySelector(selector){return canvases.get(Number(selector.match(/"(\d+)"/)[1]))}};
  const context=vm.createContext({Map,Set,Math,JSON,Number,Promise,AbortController,DOMException,
    setTimeout,clearTimeout,queueMicrotask,document:{createElement(tag){return tag==='video'?new Video():tag==='canvas'?output():element()}},
    fetch:async(url,options)=>{
      const body=JSON.parse(options.body);requests.push(body);
      const first=Math.floor(body.frame/240)*240;
      return{json:async()=>({ok:true,url:'video-'+body.variant+'-'+first,first_frame:first,end_frame:first+240,
        fps:30,fps_fraction:'30',frames:36000,width:320,height:240,atlas_width:960,atlas_height:240,
         regions:{original:[0,0,320,240],result:[320,0,320,240],difference:[640,0,320,240],...(body.mask_values?{mask:[320,0,320,240]}:{})}})};
    },
  });
  vm.runInContext(script,context);
  const engine=context.createDenoisePlayback(host,{fps:30,count:36000,slots:()=>slots,path:slot=>'variant'+slot,
     identity:slot=>'v1-'+slot,diffMode:()=>'direction',contrast:()=>32,changed(){},painted(){},...options});
  return{context,engine,videos,canvases,requests,setSlots(value){slots=value;engine.reconcile()}};
}

test('common-frame selection never pairs adjacent decoded frames',()=>{
  const context=vm.createContext({});vm.runInContext(script,context);
  const queues=[new Map([[100,'a'],[101,'b']]),new Map([[100,'a'],[102,'b']])];
  assert.equal(context.denoisePlaybackCommonFrame(queues,100,false),100);
  assert.equal(context.denoisePlaybackCommonFrame(queues,101,false),null);
  assert.equal(context.denoisePlaybackCommonFrame([new Map([[101,'a']]),new Map([[101,'b']])],100,true),null);
  assert.equal(context.denoisePlaybackOrdinal({first_frame:300,end_frame:540,fps:30},1.2),336);
  assert.equal(context.denoisePlaybackOrdinal({first_frame:300,end_frame:540,fps:30},8),null);
  assert.equal(context.denoisePlaybackOrdinal({first_frame:192,end_frame:240,fps:24000/1001},8*1001/24000),200);
  assert.equal(context.denoisePlaybackTime(1140.123),'19:00.123');
});

test('mask input and repair retain their own reference and publish only common frames',async()=>{
  const painted=[];
  const run=setup({stage:()=>'dust',path:slot=>slot?'repair.mkv':'',
    reference:slot=>slot?'original':'input',request:slot=>slot?{}:{mask_values:{detector:'temporal'}},
    sourcePainted:(_image,manifest,frame)=>{assert.ok(manifest.regions.mask);painted.push(frame);}});
  try {
    run.setSlots([0,1]);await settle();
    assert.equal(run.requests.find(body=>body.mask_values).reference,'input');
    assert.equal(run.requests.find(body=>body.variant==='repair.mkv').reference,'original');
    run.engine.toggle();await settle();
    const before=painted.length;run.videos[0].frame(1);assert.equal(painted.length,before);
    run.videos[1].frame(1);assert.equal(painted.at(-1),1);
  } finally {run.engine.destroy();}
});

test('mode switches reuse the very same decoded atlas without seeking or fetching',async()=>{
  const {engine,canvases,requests}=setup();
  try {
    engine.reconcile();await settle();
    assert.equal(engine.state.loading,false);
    assert.equal(engine.state.count,36000);
    const result=canvases.get(1).last.source;
    const count=requests.length;
    engine.setView(1,'difference');
    assert.equal(canvases.get(1).last.source,result);
    assert.deepEqual(canvases.get(1).last.region,[640,0,320,240,0,0,320,240]);
    engine.setView(1,'original');
    assert.equal(canvases.get(1).last.source,result);
    assert.equal(requests.length,count);
    assert.equal(engine.state.frame,0);
  } finally {engine.destroy()}
});

test('visible canvases advance atomically and seeking reaches minute ten without a twelve-second cap',async()=>{
  const {engine,canvases,videos,requests}=setup();
  try {
    engine.reconcile();await settle();engine.toggle();await settle();
    videos[0].frame(1);
    assert.equal(canvases.get(1).dataset.dpFrame,'0');
    assert.equal(canvases.get(2).dataset.dpFrame,'0');
    videos[1].frame(1);
    assert.equal(canvases.get(0).dataset.dpFrame,'1');
    assert.equal(canvases.get(1).dataset.dpFrame,'1');
    assert.equal(canvases.get(2).dataset.dpFrame,'1');
    engine.pause();
    await engine.seekSeconds(600);await settle();
    assert.equal(engine.state.loading,false);
    assert.equal(engine.state.frame,18000);
    assert.equal(canvases.get(0).dataset.dpFrame,'18000');
    assert.ok(requests.some(r=>r.frame===18000));
  } finally {engine.destroy()}
});

test('last seek wins when an older manifest request finishes late',async()=>{
  const {context,engine,canvases}=setup();
  try {
    engine.reconcile();await settle();
    const original=context.fetch,pending=[];
    context.fetch=(url,options)=>new Promise(resolve=>pending.push({url,options,resolve}));
    const older=engine.seekFrame(900);
    const newer=engine.seekFrame(1600);
    const finish=async target=>{
      for(const item of pending.filter(item=>JSON.parse(item.options.body).frame===target))
        item.resolve(await original(item.url,item.options));
      await settle();
    };
    await finish(1600);await newer;
    await finish(900);await older;
    assert.equal(engine.state.frame,1600);
    assert.equal(canvases.get(1).dataset.dpFrame,'1600');
    assert.equal(canvases.get(2).dataset.dpFrame,'1600');
    assert.equal(engine.state.error,'');
  } finally {engine.destroy()}
});

test('comparison membership changes keep the unchanged decoder and current frame',async()=>{
  const {engine,videos,requests,setSlots}=setup();
  try {
    engine.reconcile();await settle();
    const first=videos[0],count=requests.length;
    setSlots([1]);await settle();
    assert.equal(first.removed,undefined);
    assert.equal(videos.length,2);
    setSlots([1,2]);await settle();
    assert.equal(first.removed,undefined);
    assert.equal(requests.length,count);
    assert.equal(engine.state.frame,0);
    assert.equal(engine.state.loading,false);
  }finally{engine.destroy()}
});

test('a backward seek discards future buffers instead of jumping forward on resume',async()=>{
  const {engine,videos}=setup();
  try {
    engine.reconcile();await settle();engine.toggle();await settle();
    videos[0].frame(5);videos[1].frame(5);
    assert.equal(engine.state.frame,5);
    engine.pause();await engine.seekFrame(1);await settle();
    assert.equal(engine.state.frame,1);
    engine.toggle();
    videos[0].frame(2);
    assert.equal(engine.state.frame,1);
    videos[1].frame(2);
    assert.equal(engine.state.frame,2);
  }finally{engine.destroy()}
});

test('exact still mode pauses and releases reel decoders, shares seeking, then resumes compressed playback',async()=>{
  const {engine,videos,requests,canvases}=setup();
  try {
    engine.reconcile();await settle();engine.toggle();await settle();
    engine.pause();await engine.seekFrame(12);await settle();
    engine.setExact(true);await settle();const count=requests.length;
    assert.equal(engine.state.exact,true);assert.equal(engine.state.playing,false);
    assert.ok(videos.every(video=>video.removed));
    await engine.step(1);await settle();assert.equal(engine.state.frame,13);assert.equal(requests.length,count);
    const before=canvases.get(1).dataset.dpFrame;engine.draw();assert.equal(canvases.get(1).dataset.dpFrame,before);
    engine.toggle();await settle();assert.equal(engine.state.exact,false);assert.equal(engine.state.playing,true);
    assert.equal(engine.state.frame,13);assert.equal(canvases.get(1).dataset.dpFrame,'13');assert.equal(canvases.get(1).dataset.dpQuality,'preview');
  }finally{engine.destroy()}
});
