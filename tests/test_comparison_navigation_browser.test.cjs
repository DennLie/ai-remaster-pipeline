const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

function setup() {
  const pending=[],draws=[],app={innerHTML:'',children:[]},timers=new Map();let timer=0;
  const context=vm.createContext({URLSearchParams,AbortController,esc:String,window:{addEventListener(){}},
    setInterval:fn=>{timers.set(++timer,fn);return timer},clearInterval:id=>timers.delete(id),
    document:{querySelectorAll:()=>[],querySelector:selector=>selector==='#app > .stage-loading'?app.innerHTML.includes('stage-loading')?{}:null:selector==='.denoise-workspace'?{}:null,getElementById:id=>id==='app'?app:id==='root'?{textContent:''}:null},
    fetch:(url,options)=>new Promise((resolve,reject)=>pending.push({url,signal:options?.signal,reject,reply:payload=>resolve({json:async()=>payload}),finish:marker=>resolve({json:async()=>({marker,stages:[],settings:{global:{}}})})})),
    isRestorationComparison:()=>true,denoiseWorkspaceDeactivate(){},
    captureScrollState(){},isEditingField:()=>false,hasMediaOnPage:()=>false,
    drawTabs(){},draw:()=>draws.push(vm.runInContext('state.marker',context)),
    wireColourShotVideos(){},renderSignature:()=>'',outpaintVisualSignature:()=>'',
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/core.js'),'utf8'),context);
  for(const name of ['drawTabs','pruneSelected','notifyNewLogErrors','updateSystemStatus','updateRunLogs','restoreScrollState']) context[name]=()=>{};
  context.availableTabs=()=>['global','denoise','deflicker'];
  context.updateDenoiseMatrixStatus=()=>draws.push(vm.runInContext('state.marker',context));
  return {context,pending,draws,app,timers};
}

test('out-of-order tab responses and an older poll cannot replace the current comparison',async()=>{
  const {context,pending,draws}=setup();
  const poll=context.refresh();
  const noise=context.selectTab('denoise');const flicker=context.selectTab('deflicker');
  pending[2].finish('deflicker');await flicker;
  pending[1].finish('obsolete-denoise');await noise;
  pending[0].finish('obsolete-overview');await poll;
  assert.deepEqual(draws,['deflicker']);
});

test('slow overlapping polls still publish progress, but cannot overwrite a newer applied response',async()=>{
  const {context,pending,draws}=setup();
  const first=context.refresh(),second=context.refresh();
  pending[0].finish('first progress');await first;assert.deepEqual(draws,['first progress']);
  pending[1].finish('second progress');await second;
  const third=context.refresh(),fourth=context.refresh();
  pending[3].finish('latest progress');await fourth;
  pending[2].finish('obsolete progress');await third;
  assert.deepEqual(draws,['first progress','second progress','latest progress']);
});

test('tab click immediately replaces old controls with a navigable loading shell and aborts the old request',async()=>{
  const {context,pending,app,timers}=setup();app.innerHTML='<button>Old stage action</button>';
  const noise=context.selectTab('denoise');
  assert.match(app.innerHTML,/<h1>denoise<\/h1>/);assert.match(app.innerHTML,/Loading input and result status/);
  assert.doesNotMatch(app.innerHTML,/Old stage action/);assert.equal(timers.size,1);
  const flicker=context.selectTab('deflicker');assert.equal(pending[0].signal.aborted,true);
  pending[0].finish('obsolete');await noise;
  assert.match(app.innerHTML,/<h1>deflicker<\/h1>/);assert.equal(timers.size,1);
  pending[1].finish('flicker');await flicker;assert.equal(timers.size,0);
});

test('failed tab request exposes an error and retry; retry can succeed',async()=>{
  const {context,pending,app,timers,draws}=setup();
  const first=context.selectTab('denoise');pending[0].reject(Error('Server unavailable'));await first;
  assert.match(app.innerHTML,/role="alert"/);assert.match(app.innerHTML,/Server unavailable/);
  assert.match(app.innerHTML,/Try again/);assert.equal(timers.size,0);
  assert.equal(vm.runInContext('tabLoading',context),false);
  const retry=context.selectTab('denoise');pending[1].finish('recovered');await retry;
  assert.deepEqual(draws,['recovered']);assert.equal(timers.size,0);
});

test('a late failed request cannot replace the new tab or its loading status',async()=>{
  const {context,pending,app,draws}=setup();
  const noise=context.selectTab('denoise'),flicker=context.selectTab('deflicker');
  pending[0].reject(Error('Obsolete failure'));await noise;
  assert.doesNotMatch(app.innerHTML,/Obsolete failure/);assert.equal(vm.runInContext('tabLoading',context),true);
  pending[1].finish('flicker');await flicker;assert.deepEqual(draws,['flicker']);
});

test('a JSON server error is rejected before publishing a malformed state',async()=>{
  const {context,pending,app}=setup();
  const request=context.selectTab('denoise');pending[0].reply({error:'Input check failed'});await request;
  assert.match(app.innerHTML,/Input check failed/);assert.equal(vm.runInContext('state',context),null);
});

test('a successful poll replaces a failed shell even when the render signature is unchanged',async()=>{
  const {context,pending,draws}=setup();
  const request=context.selectTab('denoise');pending[0].reject(Error('Temporary failure'));await request;
  const poll=context.refresh();pending[1].finish('recovered by poll');await poll;
  assert.deepEqual(draws,['recovered by poll']);
});
