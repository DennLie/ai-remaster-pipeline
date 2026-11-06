const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const base=path.join(__dirname,'../ai_remaster_gui/static/js');
function setup() {
  const variants=['zoom0','zoom3','current'].map(method=>({method,preset:'settings',output:`intermediate/stabilized/${method}.mkv`,revision:'v1'}));
  const state={settings:{global:{source:'raw.mkv'},stabilize:{zoom:'1.5',smoothing:'12',max_shift:'48',scene_aware:'true'},cloud:{intermediate_format:'lossless'}},
    source_info:{duration:'600',fps:'24'},stabilize_matrix:{source:'intermediate/cleaned/selected.mkv',source_revision:'s1',variants},
    denoise_matrix:{source:'raw.mkv',variants:[]},stages:[{key:'stabilize',files:variants.map(v=>({path:v.output})),fields:[['zoom','Zoom'],['smoothing','Smoothing']]},{key:'denoise',files:[]}]};
  const context=vm.createContext({state,active:'stabilize',parseDuration:Number,esc:String,location:{pathname:'/'},
    document:{querySelector:()=>null,querySelectorAll:()=>[],getElementById:()=>null},
    fieldHtml:(stage,field)=>`<input data-field="${field[0]}">`});
  for(const file of ['review-zoom.js','render-denoise.js','denoise-workspace.js','comparison-layout.js']) vm.runInContext(fs.readFileSync(path.join(base,file),'utf8'),context);
  const initialize=key=>{context.active=key;context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages.find(s=>s.key===key)))};
  initialize('stabilize');return {context,state,initialize};
}
test('stabilization uses the real input, retains custom zoom and offers explicit 0/3 zoom choices',()=>{
  const {context,state}=setup(),before=JSON.stringify(state.settings);
  context.denoiseWorkspaceSelectAll();
  const snapshot=context.denoiseWorkspaceSnapshot();
  assert.equal(snapshot.stage,'stabilize');assert.equal(snapshot.paths[0],state.stabilize_matrix.source);
  assert.equal(snapshot.variants.length,3);assert.equal(snapshot.selected.method,'current');
  const html=context.denoiseMatrixHtml();
  assert.match(html,/0% additional zoom/);assert.match(html,/3% additional zoom/);assert.match(html,/custom zoom/);
  assert.equal(JSON.stringify(state.settings),before);
});
test('Review inspection and reference selection do not change Labor, settings or start rendering',()=>{
  const {context,state}=setup(),calls=[];
  context.postJson=async(...args)=>{calls.push(args);return {ok:true}};
  context.denoiseWorkspaceClear();const before=JSON.stringify(state.settings);
  context.inspectComparisonVariant('zoom0','settings');context.denoiseWorkspaceFor().setReference('original');
  assert.equal(context.denoiseWorkspaceSnapshot().selected.method,'zoom0');
  assert.equal(context.denoiseWorkspaceSnapshot().selection.length,0);assert.equal(calls.length,0);
  assert.equal(JSON.stringify(state.settings),before);
});
test('missing stabilization variants render only after explicit confirmation',async()=>{
  const {context,state}=setup(),calls=[];
  state.stabilize_matrix.variants=[];context.inspectComparisonVariant('zoom0','settings');
  context.postJson=async(url,body)=>{calls.push({url,body});return {ok:true}};context.refresh=async()=>{};
  await context.denoiseWorkspaceGenerate();
  assert.deepEqual(JSON.parse(JSON.stringify(calls)),[{url:'/api/stabilize-batch',body:{variants:[{method:'zoom0',preset:'settings'}]}}]);
  assert.equal(state.settings.stabilize.zoom,'1.5');
});
test('adoption sends the inspected path for server validation and synchronizes the real zoom field',async()=>{
  const {context,state}=setup(),calls=[],zoom={value:'1.5'},label={textContent:'1.5'},readout={textContent:'1.5'};
  context.inspectComparisonVariant('zoom0','settings');
  context.document.querySelector=selector=>selector==='[data-field="zoom"]' ? zoom : selector==='[data-stabilization-current-zoom]' ? label : null;
  context.document.getElementById=id=>id==='zoomValue' ? readout : null;
  context.postJson=async(url,body)=>{calls.push({url,body});return {ok:true}};
  context.refresh=async()=>{state.settings.stabilize.zoom='0'};
  await context.denoiseMatrixUseSelected();
  assert.deepEqual(JSON.parse(JSON.stringify(calls)),[{url:'/api/stabilize-adopt',body:{method:'zoom0',variant:state.stabilize_matrix.variants[0].output}}]);
  assert.equal(zoom.value,'0');assert.equal(label.textContent,'0');assert.equal(readout.textContent,'0');assert.equal(state.settings.stabilize.smoothing,'12');
});
test('stabilization retains its own view and input changes reset only frame/fit',()=>{
  const {context,state,initialize}=setup();context.denoiseWorkspaceSelectAll();
  context.denoiseWorkspaceFor().seekFrame(31);context.denoiseCompareSetZoom(3);context.denoiseWorkspaceSetView(1,'difference');
  initialize('denoise');initialize('stabilize');
  assert.equal(context.denoiseWorkspaceSnapshot().frame,31);assert.equal(context.denoiseWorkspaceSnapshot().zoom,3);
  state.stabilize_matrix.source_revision='new';initialize('stabilize');
  const view=context.denoiseWorkspaceSnapshot();assert.equal(view.frame,0);assert.equal(view.zoom,1);assert.equal(view.selection.length,3);
});
test('fallback count and missing-report state remain visible even with scene details collapsed',()=>{
  const {context,state}=setup(),summary={textContent:''},report={innerHTML:''};
  context.document.querySelector=selector=>selector==='[data-stabilization-scene-summary]' ? summary : selector==='[data-stabilization-scenes]' ? report : null;
  context.inspectComparisonVariant('zoom0','settings');
  state.stabilize_matrix.variants[0].scenes=[{start:0,end:12,status:'fallback',reason:'failed fit',tracking_report_available:true}];
  context.denoiseMatrixUpdateUi();
  assert.match(summary.textContent,/1\/1 scenes: unstabilized safety fallback/);
  assert.match(report.innerHTML,/Frames 1–12.*Unstabilized safety fallback.*failed fit/);
  state.stabilize_matrix.variants[0].scenes=null;context.denoiseMatrixUpdateUi();
  assert.match(summary.textContent,/Report unavailable/);assert.match(report.innerHTML,/No tracking success is inferred/);
});
test('footprint inspection is read-only and late frame responses cannot publish stale masks',async()=>{
  const {context,state}=setup(),status={textContent:''},image={hidden:true,removeAttribute(){delete this.src}},host={dataset:{},isConnected:true};
  host.querySelector=selector=>selector==='[data-footprint-status]' ? status : image;
  context.document.querySelector=selector=>selector==='[data-stabilization-footprint]' ? host : null;
  context.URLSearchParams=URLSearchParams;context.media=path=>'/media?path='+path;
  let resolve,calls=[];context.api=url=>{calls.push(url);return new Promise(r=>resolve=r)};
  const before=JSON.stringify(state.settings),request=context.inspectStabilizationFootprint();
  context.denoiseWorkspaceFor().seekFrame(1);context.invalidateStabilizationFootprint();
  resolve({ok:true,image:'old.png',valid_output_pct:100,missing_output_pct:0,cropped_source_area_estimate_pct:5.91,note:'Raster estimate'});
  await request;assert.equal(image.hidden,true);assert.equal(image.src,undefined);
  const next=context.inspectStabilizationFootprint();
  resolve({ok:true,image:'new.png',valid_output_pct:100,missing_output_pct:0,cropped_source_area_estimate_pct:5.91,note:'Raster estimate'});
  await next;assert.equal(image.hidden,false);assert.match(image.src,/new.png/);
  assert.match(status.textContent,/Frame 2:.*estimated cropped source area 5.91%/);
  assert.equal(calls.length,2);assert.equal(JSON.stringify(state.settings),before);
});
