const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const base=path.join(__dirname,'../ai_remaster_gui/static/js');
const settle=async()=>{for(let index=0;index<12;index++) await new Promise(setImmediate)};

function setup() {
  const listeners={},requests=[],canvases=new Map(),handles=new Map();
  const playback={state:{frame:4,exact:true,count:720000,fps:24,playing:false}};
  let slots=[1,2],mode='direction',contrast=32,revision='v1',notifications=0;
  const revisions=new Map();
  const views=new Map([[1,'result'],[2,'difference']]);
  function canvas() {
    const element={width:4,height:2,dataset:{},pixels:new Uint8ClampedArray(32),draws:[],
      getBoundingClientRect(){return{left:40,width:800}},closest(){return{getBoundingClientRect:()=>({left:100,width:400})}}};
    const context={drawImage(image,...args){element.draws.push({image,args});if(image.pixels) element.pixels=new Uint8ClampedArray(image.pixels)},
      getImageData(){return{data:new Uint8ClampedArray(element.pixels)}},
      putImageData(image){element.pixels=new Uint8ClampedArray(image.data)}};
    element.getContext=()=>context;
    return element;
  }
  for(const slot of [0,1,2]) {
    canvases.set(slot,canvas());
    const captures=new Set();
    handles.set(slot,{dataset:{dsWipe:String(slot)},style:{},firstElementChild:{style:{},offsetWidth:120},hidden:true,
      setAttribute(key,value){this[key]=value},setPointerCapture(id){captures.add(id)},hasPointerCapture:id=>captures.has(id),releasePointerCapture(id){captures.delete(id)}});
  }
  const host={isConnected:true,addEventListener(name,callback){listeners[name]=callback},
    querySelectorAll(selector){return selector==='[data-ds-wipe]'?[...handles.values()]:[]},
    querySelector(selector){return canvases.get(Number(selector.match(/"(\d+)"/)[1]))}};
  const context=vm.createContext({AbortController,Map,Set,Promise,Number,Math,JSON,Uint8ClampedArray,
    ImageData:class{constructor(data,width,height){this.data=data;this.width=width;this.height=height}},
    document:{addEventListener(){},createElement(tag){assert.equal(tag,'canvas');return canvas()},querySelector(){return null}},
    Image:class {
      set src(value) {
        this._src=value;this.naturalWidth=4;this.naturalHeight=2;
        const frame=Number(decodeURIComponent(value).match(/(?:before|after)-(\d+)/)?.[1] || 0);
        this.pixels=new Uint8ClampedArray(32);
        for(let index=0;index<32;index+=4) this.pixels.set([100+frame%50+(value.includes('after-')?1:0),100,100,255],index);
        queueMicrotask(()=>this.onload());
      }
      get src(){return this._src}
    },
    fetch:async url=>{
      const query=new URL('http://localhost'+url).searchParams,frame=Number(query.get('frame')),variant=query.get('variant');
      requests.push({frame,variant});
      return {ok:true,json:async()=>({ok:true,exact:true,frame,frames:720000,fps:24000/1001,width:4,height:2,
        before:`before-${frame}.png`,after:variant ? `after-${frame}-${variant}.png` : `before-${frame}.png`})};
    },
  });
  for(const file of ['review-contrast.js','frame-inspector.js','denoise-review.js']) vm.runInContext(fs.readFileSync(path.join(base,file),'utf8'),context);
  const controller=context.createDenoiseStillReview(host,{playback:()=>playback,slots:()=>slots,path:slot=>slot?'variant'+slot:'',identity:slot=>revisions.get(slot)||revision,
    view:slot=>views.get(slot)||'result',diffMode:()=>mode,contrast:()=>contrast,painted(){},changed(){notifications++},
    metadata(info){playback.state.count=info.frames;playback.state.fps=info.fps;}});
  return {context,controller,playback,canvases,handles,listeners,requests,views,
    setSlots(value){slots=value},setDiff(value,boost){mode=value;contrast=boost},revise(slot){if(slot===undefined) revision='v2';else revisions.set(slot,'v2')},notifications:()=>notifications};
}

test('exact stills replace the existing canvases atomically at the shared frame, at native dimensions',async()=>{
  const {controller,playback,canvases,requests}=setup();
  try {
    controller.reconcile();await settle();
    assert.equal(controller.state.error,'');assert.equal(controller.state.loading,false);
    assert.equal(controller.state.pairs.size,2);assert.equal(requests.length,2);
    for(const canvas of canvases.values()) {assert.equal(canvas.dataset.dpQuality,'exact');assert.equal(canvas.dataset.dpFrame,'4');assert.equal(canvas.width,4);assert.equal(canvas.height,2)}
    assert.equal(playback.state.frame,4);assert.equal(playback.state.count,720000);assert.equal(playback.state.fps,24000/1001);
    assert.notDeepEqual(Array.from(canvases.get(0).pixels),Array.from(canvases.get(1).pixels));
  }finally{controller.destroy()}
});

test('Original, Result, Difference and Wipe reuse the same exact frame, with unchanged measurements',async()=>{
  const {controller,playback,views,canvases,requests,setDiff}=setup();
  try {
    controller.reconcile();await settle();
    const pair=controller.state.pairs.get(1),metrics=JSON.stringify(pair.metrics),count=requests.length;
    views.set(1,'original');controller.draw();assert.deepEqual(Array.from(canvases.get(1).pixels),Array.from(canvases.get(0).pixels));
    views.set(1,'difference');controller.draw();const first=Array.from(canvases.get(1).pixels);
    setDiff('rgb',64);controller.reconcile();assert.notDeepEqual(Array.from(canvases.get(1).pixels),first);
    views.set(1,'wipe');controller.draw();
    assert.equal(canvases.get(1).draws.at(-1).args[0],2);
    assert.equal(playback.state.frame,4);assert.equal(requests.length,count);assert.equal(JSON.stringify(pair.metrics),metrics);
  }finally{controller.destroy()}
});

test('latest requested frame wins and expensive decodes are serialized',async()=>{
  const {context,controller,playback,canvases,requests}=setup();
  try {
    const original=context.fetch;let finish;
    context.fetch=url=>new Promise(resolve=>{finish=async()=>resolve(await original(url))});
    controller.reconcile();
    playback.state.frame=5;controller.reconcile();playback.state.frame=6;controller.reconcile();
    assert.equal(requests.length,0);
    context.fetch=original;await finish();await settle();
    assert.deepEqual(requests.map(item=>item.frame),[4,6,6]);
    assert.equal(controller.state.frame,6);assert.equal(canvases.get(1).dataset.dpFrame,'6');
    assert.equal(controller.state.loading,false);
  }finally{controller.destroy()}
});

test('failed or mismatched frame metadata cannot publish stale or mixed evidence and can be retried',async()=>{
  const {context,controller,playback,canvases}=setup();
  try {
    controller.reconcile();await settle();
    const original=context.fetch;
    context.fetch=async url=>{const response=await original(url),data=await response.json();return{ok:true,json:async()=>({...data,frame:data.frame-1})}};
    playback.state.frame=10;controller.reconcile();await settle();
    assert.match(controller.state.error,/invalid exact-frame metadata/);assert.equal(controller.state.pairs.size,0);
    assert.equal(canvases.get(1).dataset.dpFrame,'4');
    context.fetch=original;controller.reconcile(true);await settle();
    assert.equal(controller.state.error,'');assert.equal(canvases.get(1).dataset.dpFrame,'10');
  }finally{controller.destroy()}
});

test('replaced output revisions invalidate exact pixels even at the same frame',async()=>{
  const {controller,requests,revise}=setup();
  try {
    controller.reconcile();await settle();const pair=controller.state.pairs.get(1);
    revise();controller.reconcile();assert.equal(controller.state.pairs.size,0);await settle();
    assert.notEqual(controller.state.pairs.get(1),pair);assert.equal(requests.length,4);
  }finally{controller.destroy()}
});

test('changing one variant retains other exact frames instead of decoding them again',async()=>{
  const {controller,requests,revise}=setup();
  try {
    controller.reconcile();await settle();
    const unaffected=controller.state.pairs.get(2),replaced=controller.state.pairs.get(1);
    revise(1);controller.reconcile();
    assert.equal(controller.state.pairs.get(2),unaffected);
    assert.equal(controller.state.pairs.has(1),false);
    assert.equal(replaced.difference.width,0);
    await settle();
    assert.equal(controller.state.pairs.get(2),unaffected);
    assert.equal(requests.length,3);
    assert.equal(requests.at(-1).variant,'variant1');
  }finally{controller.destroy()}
});

test('variants decoded from different processing inputs are never published together',async()=>{
  const {context,controller,canvases}=setup();
  try {
    const original=context.fetch;
    context.fetch=async url=>{
      const response=await original(url),data=await response.json();
      return {ok:true,json:async()=>url.includes('variant2') ? {...data,before:'unrelated-input.png'} : data};
    };
    controller.reconcile();await settle();
    assert.match(controller.state.error,/different frame mappings/);
    assert.equal(controller.state.pairs.size,0);
    assert.equal(canvases.get(0).dataset.dpQuality,undefined);
    assert.equal(canvases.get(1).dataset.dpQuality,undefined);
  }finally{controller.destroy()}
});

test('source-only exact inspection does not require a rendered variant',async()=>{
  const {controller,setSlots,requests,canvases}=setup();
  try {
    setSlots([0]);controller.reconcile();await settle();
    assert.equal(requests[0].variant,'');assert.equal(controller.state.error,'');
    assert.equal(canvases.get(0).dataset.dpFrame,'4');assert.equal(canvases.get(0).dataset.dpQuality,'exact');
  }finally{controller.destroy()}
});

test('returning to compressed playback invalidates late exact responses and frees difference buffers',async()=>{
  const {context,controller,playback,canvases}=setup();
  try {
    controller.reconcile();await settle();const difference=controller.state.pairs.get(1).difference;
    const original=context.fetch;let finish;
    context.fetch=url=>new Promise(resolve=>{finish=async()=>resolve(await original(url))});
    playback.state.frame=8;controller.reconcile();
    assert.equal(difference.width,0);
    playback.state.exact=false;controller.reconcile();await finish();await settle();
    assert.equal(controller.state.pairs.size,0);assert.equal(controller.state.loading,false);
    assert.equal(canvases.get(1).dataset.dpFrame,'4');
  }finally{controller.destroy()}
});

test('wipe dragging respects the zoomed and panned image, keyboard endpoints and pointer cleanup',async()=>{
  const {controller,views,handles,listeners}=setup();
  const event=(x,id=7)=>({clientX:x,pointerId:id,pointerType:'mouse',button:0,target:{closest:()=>handles.get(1)},preventDefault(){},stopPropagation(){}});
  try {
    views.set(1,'wipe');controller.reconcile();await settle();
    const handle=handles.get(1);assert.equal(handle.hidden,false);assert.equal(handle.style.left,'340px');
    listeners.pointerdown(event(440));assert.equal(handle.hasPointerCapture(7),true);
    listeners.pointerdown(event(0,9));listeners.pointermove(event(0,9));assert.equal(controller.wipe(1),50);
    listeners.pointermove(event(240));assert.equal(controller.wipe(1),25);
    listeners.pointermove(event(900));assert.equal(controller.wipe(1),100);
    listeners.pointermove(event(0));assert.equal(controller.wipe(1),0);
    listeners.pointerup(event(0));assert.equal(handle.hasPointerCapture(7),false);
    listeners.pointermove(event(440));assert.equal(controller.wipe(1),0);
    listeners.keydown({...event(0),key:'End'});assert.equal(controller.wipe(1),100);
    listeners.keydown({...event(0),key:'Home'});assert.equal(controller.wipe(1),0);
    listeners.pointerdown(event(440));listeners.pointercancel(event(440));assert.equal(handle.hasPointerCapture(7),false);
    listeners.pointerdown(event(440));listeners.lostpointercapture(event(440));assert.equal(handle.hasPointerCapture(7),false);
  }finally{controller.destroy()}
});
