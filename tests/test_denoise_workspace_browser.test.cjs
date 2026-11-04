const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const base = path.join(__dirname,'../ai_remaster_gui/static/js');
const script = ['review-zoom.js','render-denoise.js','denoise-workspace.js','denoise-review.js'].map(file => fs.readFileSync(path.join(base,file),'utf8')).join('\n');

function setup() {
  const source = {path:'input/source.mkv',kind:'source'};
  const variants = ['hqdn3d','nlmeans','bm3d'].flatMap(method =>
    ['gentle','balanced','strong'].map(preset => ({method,preset,
      output:`intermediate/denoised/source_denoised_${method}_${preset}_abc123.mkv`,revision:'v1'})));
  const state = {settings:{global:{source:source.path},denoise:{method:'bm3d',preset:'balanced'}},
    source_info:{duration:'2.4'},
    stages:[{key:'denoise',files:variants.map(v => ({path:v.output}))}],
    denoise_matrix:{source:source.path,source_revision:'s1',variants}};
  const context = vm.createContext({state,active:'denoise',AbortController,
    document:{querySelector:()=>null,querySelectorAll:()=>[],getElementById:()=>null},
    esc:String,parseDuration:Number,mediaClip:(path,start,end,key)=>`${path}?${start}:${end}:${key}`});
  vm.runInContext(script,context);
  const evaluate = expression => vm.runInContext(expression,context);
  return {context,state,variants,evaluate};
}

test('all nine selected variants have stable independent slots; empty selection stays empty',()=>{
  const {context,evaluate}=setup();
  context.denoiseWorkspaceSelectAll();
  const candidates=context.denoiseCompareCandidates(context.state.stages[0]);
  context.denoiseCompareInitializePaths(candidates);
  const first=Array.from(context.denoiseWorkspaceSnapshot().paths);
  assert.equal(first.length,10);
  assert.equal(new Set(first).size,10);
  context.denoiseMatrixSelect('hqdn3d','gentle');
  context.denoiseCompareInitializePaths(candidates);
  const second=Array.from(context.denoiseWorkspaceSnapshot().paths);
  assert.equal(second[1],'');
  assert.deepEqual(second.slice(2),first.slice(2));
  context.denoiseWorkspaceClear();
  context.denoiseCompareInitializePaths(candidates);
  assert.equal(context.denoiseWorkspaceSnapshot().paths.filter(Boolean).length,1);
  assert.equal(context.denoiseWorkspaceSnapshot().selection.length,0);
});

test('switching to Detail restores exactly three legacy slots',()=>{
  const {context,evaluate}=setup();
  context.denoiseWorkspaceSelectAll();
  const candidates=context.denoiseCompareCandidates(context.state.stages[0]);
  context.denoiseCompareInitializePaths(candidates);
  assert.equal(context.denoiseWorkspaceSnapshot().paths.length,10);
  context.active='detail';
  context.denoiseCompareInitializePaths(candidates);
  assert.equal(evaluate('denoiseComparePaths.length'),3);
  context.active='denoise';
  context.denoiseCompareInitializePaths(candidates);
  assert.equal(context.denoiseWorkspaceSnapshot().paths.length,10);
  assert.equal(context.denoiseWorkspaceSnapshot().selection.length,9);
});

test('full-film duration is not capped; legacy clip start clamps against the whole input',()=>{
  const {context,state,evaluate}=setup();
  state.source_info.duration='1200';
  assert.equal(context.denoiseCompareFullDuration(),1200);
  context.denoiseCompareEnsureLength();
  context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages[0]));
  assert.equal(context.denoiseWorkspaceSnapshot().length,1200);
  context.active='detail';
  context.renderDenoiseComparePlayers=()=>{};
  context.document.getElementById=id=>id==='denoiseCompareStart'?{value:'900'}:
    id==='denoiseCompareLength'?{value:'4'}:null;
  context.denoiseCompareApplyClip();
  assert.equal(evaluate('denoiseCompareStart'),900);
});

test('stale output files are excluded even if count is unchanged; server-selected output wins',()=>{
  const {context,state,variants}=setup();
  const stale='intermediate/denoised/older_denoised_bm3d_balanced_def456.mkv';
  state.stages[0].files.unshift({path:stale});
  assert.equal(context.denoiseVariantPath('bm3d','balanced'),variants[7].output);
  assert.ok(!context.denoiseCompareCandidates(state.stages[0]).some(v => v.path===stale));
  state.denoise_matrix.variants=variants.filter(v => v.method!=='bm3d');
  assert.equal(context.denoiseVariantPath('bm3d','balanced'),'');
  assert.equal(context.denoiseVariantReady('bm3d','balanced'),false);
});

test('selection and arrows never generate or adopt; explicit confirmation renders missing pairs only',async()=>{
  const {context,state,evaluate}=setup();
  state.denoise_matrix.variants=[];
  const before=JSON.stringify(state.settings);
  const calls=[];
  context.postJson=async(url,body)=>{calls.push({url,body});return{ok:true}};
  context.refresh=async()=>{};
  context.denoiseWorkspaceSelectAll();
  assert.equal(calls.length,0);
  context.denoiseWorkspaceDismissGeneration();
  assert.equal(context.denoiseWorkspaceSnapshot().generationDismissed,true);
  await context.denoiseWorkspaceGenerate();
  assert.equal(calls.length,1);
  assert.equal(calls[0].url,'/api/denoise-batch');
  assert.equal(calls[0].body.variants.length,9);
  assert.equal(JSON.stringify(state.settings),before);
  state.running=true;
  await context.denoiseWorkspaceGenerate();
  assert.equal(calls.length,1);
  const focused=[];
  context.document.querySelectorAll=()=>[{}, {focus(){focused.push('next')}}];
  const cell={closest(){return this}};
  context.document.querySelectorAll=()=>[cell,{focus(){focused.push('next')}}];
  context.denoiseMatrixKeyboard({target:cell,key:'ArrowRight',preventDefault(){}});
  assert.deepEqual(focused,['next']);
  assert.equal(calls.length,1);
});

test('generation failures preserve selections and expose a retryable error',async()=>{
  const {context,state,evaluate}=setup();
  state.denoise_matrix.variants=[];
  context.denoiseMatrixSelect('nlmeans','strong');
  context.postJson=async()=>({ok:false,error:'A command is already running.'});
  await context.denoiseWorkspaceGenerate();
  assert.match(context.denoiseWorkspaceSnapshot().generationError,/already running/);
  assert.equal(context.denoiseWorkspaceSnapshot().generationPending,false);
  assert.equal(context.denoiseWorkspaceHas('nlmeans','strong'),true);
});

test('pipeline adoption is explicit and neither regenerates nor changes comparison membership',async()=>{
  const {context,evaluate}=setup();
  context.denoiseWorkspaceSelectAll();
  context.denoiseWorkspaceFocus(3);
  const calls=[];
  context.postJson=async(url,body)=>{calls.push({url,body});return{ok:true}};
  context.refresh=async()=>{};
  await context.denoiseMatrixUseSelected();
  assert.equal(calls.length,1);
  assert.equal(calls[0].url,'/api/settings');
  assert.deepEqual(JSON.parse(JSON.stringify(calls[0].body)),{
    stage:'denoise',values:{method:'bm3d',preset:'gentle',force:'false'},
  });
  assert.equal(context.denoiseWorkspaceSnapshot().selection.length,9);
  assert.equal(context.denoiseWorkspaceSnapshot().adoptionPending,false);
});

test('live diff requests are deduplicated and late responses cannot resurrect a disabled diff',async()=>{
  const {context,evaluate}=setup();
  evaluate("denoiseComparePaths=['source.mkv','variant.mkv'];denoiseCompareDiffState[1].enabled=true");
  const video={isConnected:true,src:'',classList:{add(){},remove(){}},pause(){},removeAttribute(){},load(){}};
  const slot={classList:{add(){},remove(){}}};
  const status={textContent:''};
  context.document.getElementById=id=>id==='denoiseCompareDiffVideo1'?video:
    id==='denoiseCompareDiffStatus1'?status:null;
  context.document.querySelector=()=>slot;
  let finish,calls=0;
  context.postJson=()=>{calls++;return new Promise(resolve=>{finish=resolve})};
  const pending=context.denoiseCompareEnsureDiffVideo(1);
  await context.denoiseCompareEnsureDiffVideo(1);
  assert.equal(calls,1);
  context.denoiseCompareSetDiffEnabled(1,false);
  finish({ok:true,url:'late-diff.mp4'});
  await pending;
  assert.equal(video.src,'');
  assert.equal(evaluate('denoiseCompareDiffState[1].enabled'),false);
  assert.equal(evaluate('denoiseCompareDiffState[1].loading'),false);
});

function fakeWorkspace(context) {
  const classes=()=>({toggle(){},add(){},remove(){}});
  const column=()=>({children:[],appendChild(p){p.parent=this;this.children.push(p)},
    insertBefore(p,before){p.remove();p.parent=this;const i=this.children.indexOf(before);this.children.splice(i<0?this.children.length:i,0,p)},
    querySelector(){return null},classList:classes()});
  const original=column(),results=column();
  const panels=()=>[...original.children,...results.children];
  const host={isConnected:true,classList:classes(),setAttribute(){},querySelector(selector){
    if(selector==='.denoise-workspace')return{classList:classes()};
    if(selector==='.denoise-original-column')return original;
    if(selector==='.denoise-results-grid')return results;
    const slot=selector.match(/data-dc-slot="(\d+)"/);
    return slot?panels().find(p=>p.dataset.dcSlot===slot[1])||null:null;
  },querySelectorAll(selector){return selector==='[data-dc-slot]'?panels():[]}};
  context.document={
    getElementById(id){return id==='denoiseComparePlayers'?host:panels().map(p=>p.video).find(v=>v?.id===id)||null},
    querySelector(selector){return host.querySelector(selector)},querySelectorAll:()=>[],
    createElement(){return{dataset:{},classList:classes(),parent:null,canvas:null,
      set innerHTML(html){
        const id=html.match(/data-dp-canvas="(\d+)"/);
        this.canvas=id?{dataset:{dpCanvas:id[1]},width:0,height:0}:null;
      },
      querySelector(selector){return selector.startsWith('[data-dp-canvas')?this.canvas:null},
      querySelectorAll(){return[]},
      remove(){if(this.parent){this.parent.children.splice(this.parent.children.indexOf(this),1);this.parent=null}},
    }},
  };
  context.denoiseCompareWirePan=()=>{};
  context.denoiseCompareWireVideos=()=>{};
  context.denoiseCompareSyncAllDiffAvailability=()=>{};
  context.denoiseCompareUpdateChartChoices=()=>{};
  context.createReviewZoom=(host)=>({host,setViewport(){},apply(){},destroy(){}});
  context.createDenoisePlayback=(host,options)=>({host,state:{frame:options.frame},reconcile(){},setView(){},destroy(){},
    seekFrame(frame){this.state.frame=frame;options.changed({...this.state,fps:30,count:72,exact:false});}});
  context.createDenoiseStillReview=()=>({state:{pairs:new Map()},reconcile(){},draw(){},destroy(){},wipe(){return 50}});
  return {panels,host};
}

test('polling, additions and same-count replacements preserve unaffected canvases and frame position',()=>{
  const {context,state,variants,evaluate}=setup();
  context.denoiseWorkspaceSelectAll();
  const {panels}=fakeWorkspace(context);
  const render=()=>context.denoiseWorkspaceRender(context.denoiseCompareCandidates(state.stages[0]));
  render();
  const master=panels()[0].canvas;
  const first=panels().find(p=>p.dataset.dcSlot==='1').canvas;
  const other=panels().find(p=>p.dataset.dcSlot==='2').canvas;
   context.denoiseWorkspaceFor().seekFrame(36);
   context.denoiseWorkspaceFor().setViewport({zoom:4,pan:{x:.2,y:.3}});
  render();
  assert.equal(panels()[0].canvas,master);
  assert.equal(panels().find(p=>p.dataset.dcSlot==='1').canvas,first);
  assert.equal(context.denoiseWorkspaceSnapshot().frame,36);
  variants[0].revision='v2';
  render();
  assert.notEqual(panels().find(p=>p.dataset.dcSlot==='1').canvas,first);
  assert.equal(panels().find(p=>p.dataset.dcSlot==='2').canvas,other);
  assert.equal(panels()[0].canvas,master);
  assert.equal(context.denoiseWorkspaceSnapshot().zoom,4);
  assert.equal(context.denoiseWorkspaceSnapshot().pan.x,.2);
  context.denoiseMatrixSelect('hqdn3d','strong');
  assert.equal(panels().length,9);
  assert.equal(panels()[0].canvas,master);
});

test('leaving a mounted workspace disposes its display adapters and rejects their late callbacks',()=>{
  const {context,state}=setup();context.denoiseWorkspaceSelectAll();fakeWorkspace(context);
  const callbacks=[],metadata=[],destroyed=[];
  for(const name of ['createDenoisePlayback','createDenoiseStillReview','createReviewZoom']) {
    const create=context[name];
    context[name]=(host,options,...args)=>{
      if(name==='createDenoisePlayback') callbacks.push(options.changed);
      if(name==='createDenoiseStillReview') metadata.push(options.metadata);
      const adapter=create(host,options,...args);
      const destroy=adapter.destroy;
      adapter.destroy=()=>{destroyed.push(name);destroy.call(adapter)};
      return adapter;
    };
  }
  context.denoiseWorkspaceRender(context.denoiseCompareCandidates(state.stages[0]));
  context.denoiseWorkspaceFor().seekFrame(17);
  context.denoiseWorkspaceDeactivate();context.active='deflicker';
  state.deflicker_matrix={source:'denoised.mkv',source_revision:'v1',variants:[]};
  context.denoiseWorkspaceInitialize([{path:'denoised.mkv',kind:'source'}]);
  callbacks[0]({frame:55,fps:1,count:60,exact:true});metadata[0]({frames:60,fps:1});
  assert.deepEqual(destroyed.sort(),['createDenoisePlayback','createDenoiseStillReview','createReviewZoom'].sort());
  assert.equal(context.denoiseWorkspaceSnapshot('denoise').frame,17);
  assert.equal(context.denoiseWorkspaceSnapshot().frame,0);assert.equal(context.denoiseWorkspaceSnapshot().fps,0);
});
