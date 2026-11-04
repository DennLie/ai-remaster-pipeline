const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');

function setup() {
  const pending=[],draws=[];
  const context=vm.createContext({URLSearchParams,window:{addEventListener(){}},
    document:{querySelector:selector=>selector==='.denoise-workspace'?{}:null,getElementById:id=>id==='root'?{textContent:''}:null},
    fetch:url=>new Promise(resolve=>pending.push({url,finish:marker=>resolve({json:async()=>({marker,stages:[]})})})),
    isRestorationComparison:()=>true,denoiseWorkspaceDeactivate(){},
    captureScrollState(){},isEditingField:()=>false,hasMediaOnPage:()=>false,
    drawTabs(){},draw:()=>draws.push(vm.runInContext('state.marker',context)),
    wireColourShotVideos(){},renderSignature:()=>'',outpaintVisualSignature:()=>'',
  });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/core.js'),'utf8'),context);
  for(const name of ['drawTabs','pruneSelected','notifyNewLogErrors','updateSystemStatus','updateRunLogs']) context[name]=()=>{};
  context.availableTabs=()=>['global','denoise','deflicker'];
  context.updateDenoiseMatrixStatus=()=>draws.push(vm.runInContext('state.marker',context));
  return {context,pending,draws};
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
