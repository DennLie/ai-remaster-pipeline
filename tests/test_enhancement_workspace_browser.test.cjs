const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const base=path.join(__dirname,'../ai_remaster_gui/static/js');
function setup() {
  const methods=['unsharp','cas','multiscale','richardson_lucy','nafnet','restormer'];
  const variants=methods.map(method=>({method,preset:'settings',output:`intermediate/enhanced/film_enhanced_${method}_123.mkv`,revision:'v1'}));
  const state={settings:{global:{source:'raw.mkv'},enhancement:{method:'off',preset:'gentle',mix:'50'},deflicker:{method:'arp'}},
    source_info:{duration:'1200',fps:'30000/1001'},enhancement_matrix:{source:'intermediate/deflickered/selected.mkv',source_revision:'s1',variants},
    deflicker_matrix:{source:'raw.mkv',source_revision:'s0',variants:[]},
    stages:[{key:'enhancement',files:variants.map(v=>({path:v.output})),fields:[['method','Method'],['preset','Preset'],['mix','Mix']]},{key:'deflicker',files:[]}]};
  const context=vm.createContext({state,active:'enhancement',parseDuration:Number,esc:String,
    document:{querySelector:()=>null,querySelectorAll:()=>[],getElementById:()=>null},
    fieldHtml:(stage,field)=>`<input data-field="${field[0]}">`});
  for(const file of ['review-zoom.js','render-denoise.js','render-enhancement.js','denoise-workspace.js'])
    vm.runInContext(fs.readFileSync(path.join(base,file),'utf8'),context);
  const initialize=key=>{context.active=key;context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages.find(s=>s.key===key)))};
  initialize('enhancement');return {context,state,initialize};
}
test('enhancement compares six current-setting methods against the actual input, not raw source',()=>{
  const {context,state}=setup();context.denoiseWorkspaceSelectAll();
  const view=context.denoiseWorkspaceSnapshot();
  assert.equal(view.variants.length,6);assert.equal(view.paths.length,7);
  assert.equal(view.paths[0],state.enhancement_matrix.source);assert.notEqual(view.paths[0],state.settings.global.source);
  const html=context.denoiseMatrixHtml();
  assert.equal((html.match(/role="checkbox"/g)||[]).length,6);
  assert.match(html,/NAFNet/);assert.match(html,/Restormer/);assert.match(html,/data-field="mix"/);
  assert.match(html,/not to the neural checkpoints/);
});
test('enhancement retains tab state and resets only frame/fit for a changed processing input',()=>{
  const {context,state,initialize}=setup();context.denoiseWorkspaceSelectAll();
  context.denoiseWorkspaceFor().seekFrame(17);context.denoiseCompareSetZoom(3);context.denoiseWorkspaceSetView(2,'wipe');
  initialize('deflicker');initialize('enhancement');
  assert.equal(context.denoiseWorkspaceSnapshot().frame,17);assert.equal(context.denoiseWorkspaceSnapshot().zoom,3);
  state.settings.enhancement.preset='strong';initialize('enhancement');
  assert.equal(context.denoiseWorkspaceSnapshot().frame,17);
  state.enhancement_matrix.source_revision='new';initialize('enhancement');
  const view=context.denoiseWorkspaceSnapshot();assert.equal(view.frame,0);assert.equal(view.zoom,1);assert.equal(view.selection.length,6);
});
test('pipeline adoption never writes the synthetic settings preset into real enhancement settings',async()=>{
  const {context,state}=setup();const calls=[];
  context.denoiseWorkspaceSelectAll();context.denoiseWorkspaceFocus(5);
  context.postJson=async(url,body)=>{calls.push({url,body});return {ok:true}};context.refresh=async()=>{};
  await context.denoiseMatrixUseSelected();
  assert.equal(calls.length,1);assert.equal(calls[0].body.stage,'enhancement');
  assert.equal(calls[0].body.values.method,'nafnet');assert.equal(calls[0].body.values.preset,undefined);
  assert.equal(state.settings.enhancement.preset,'gentle');assert.equal(state.settings.enhancement.mix,'50');
});
