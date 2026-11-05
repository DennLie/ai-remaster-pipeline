const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const read=name=>fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js',name),'utf8');

test('runStage never requests rendering after failed settings persistence',async()=>{
  const calls=[];
  const context=vm.createContext({saveStage:async()=>false,postJson:async()=>{calls.push('run');return {ok:true}},confirmOverwrite:async()=>true});
  vm.runInContext('async function runStage('+read('actions.js').split('async function runStage(')[1].split('\nasync function generateUpscalePreview')[0],context);
  await context.runStage('deflicker');assert.deepEqual(calls,[]);
});

test('shared refresh keeps a real processing preset instead of the synthetic comparison preset',()=>{
  const source=read('denoise-workspace.js').split('function denoiseWorkspaceRefresh() {')[1].split('\nfunction ')[0];
  for(const stageKey of ['enhancement','deflicker']) {
    const preset={value:'strong'},method={value:'cas'};
    const context=vm.createContext({stageKey,state:{stages:[{key:stageKey}],settings:{[stageKey]:{preset:'strong'}}},
      document:{getElementById:()=>({}),querySelector:s=>s==='[data-field="preset"]'?preset:s==='[data-field="method"]'?method:null,querySelectorAll:()=>[]},
      denoiseWorkspaceRender(){},denoiseCompareCandidates(){return []},denoiseMatrixUpdateUi(){},
      pipelineVariant:()=>({method:'cas',preset:'settings'}),showCommand(){}});
    vm.runInContext('function refresh() {'+source,context); context.refresh();
    assert.equal(preset.value,'strong');
  }
});

function saveSetup() {
  const calls=[];
  const context=vm.createContext({active:'enhancement',state:{},captureScrollState:()=>({}),stateUrl:()=>'/state?stage=enhancement',
    formValues:()=>({mix:'51'}),postJson:async()=>({ok:true}),api:async()=>({saved:true}),pruneSelected(){},
    denoiseWorkspaceRefresh:()=>calls.push('reconcile'),draw:()=>calls.push('draw'),restoreScrollState:()=>calls.push('scroll'),
    showCommand:()=>calls.push('command'),alert:()=>calls.push('error')});
  vm.runInContext('async function saveStage('+read('actions.js').split('async function saveStage(')[1].split('\nfunction formValues')[0],context);
  return {context,calls};
}
test('saving shared settings reconciles without rebuilding DOM or restoring scroll',async()=>{
  const {context,calls}=saveSetup(); await context.saveStage('enhancement',true);
  assert.deepEqual(calls,['reconcile','command']); assert.equal(context.state.saved,true);
});
test('late settings responses cannot replace another tab state',async()=>{
  const {context,calls}=saveSetup(),original=context.state;
  context.api=async()=>{context.active='deflicker';return {saved:true}};
  await context.saveStage('enhancement',true);
  assert.equal(context.state,original);assert.deepEqual(calls,[]);
});
test('failed settings saves leave state and comparison DOM untouched',async()=>{
  const {context,calls}=saveSetup(),original=context.state;
  context.postJson=async()=>({ok:false,error:'Invalid preset'});
  await context.saveStage('enhancement',true);
  assert.equal(context.state,original);assert.deepEqual(calls,['error']);
});
