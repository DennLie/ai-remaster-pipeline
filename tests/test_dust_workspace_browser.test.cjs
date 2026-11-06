const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
function setup(){
 const rows=['telea','propainter'].map((repair,i)=>({method:'candidate'+i,preset:'settings',label:'legacy / '+repair,repair_complete:true,output:'result'+i+'.mkv',revision:'report'+i,values:{detector:'legacy',repair},mask_sha256:'same-mask'}));
 const state={project_path:'A',settings:{global:{source:'raw.mkv'},dust:{detector:'legacy',repair:'telea'},cloud:{}},source_info:{duration:'10',fps:'24'},dust_matrix:{source:'stabilized.mkv',source_revision:'v1',variants:rows},stages:[{key:'dust',files:[]}]};
 const context=vm.createContext({state,active:'dust',parseDuration:Number,esc:String,location:{pathname:'/'},document:{querySelector:()=>null,querySelectorAll:()=>[],getElementById:()=>null}});
 for(const file of ['review-zoom.js','render-denoise.js','denoise-workspace.js','comparison-layout.js','render-dust.js'])vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js',file),'utf8'),context);
 context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages[0]));return {context,state};
}
test('Dust uses saved identities and the actual upstream input in the shared workspace',()=>{
 const {context,state}=setup();const view=context.denoiseWorkspaceSnapshot();assert.equal(view.stage,'dust');assert.equal(view.paths[0],'stabilized.mkv');assert.equal(view.variants.length,2);assert.equal(context.denoiseStageVariant().method,'off');assert.equal(state.dust_matrix.variants[0].mask_sha256,state.dust_matrix.variants[1].mask_sha256);
});
test('inspection and lab membership never render or change pipeline settings',()=>{
 const {context,state}=setup(),before=JSON.stringify(state.settings);context.postJson=()=>{throw Error('unexpected request')};context.denoiseWorkspaceSelectAll();context.inspectComparisonVariant('candidate1','settings');assert.equal(context.denoiseWorkspaceSnapshot().selected.method,'candidate1');assert.equal(JSON.stringify(state.settings),before);
});
test('mask-only candidates are selectable but not adoptable/playable',async()=>{
 const {context,state}=setup();state.dust_matrix.variants[1].repair_complete=false;state.dust_matrix.variants[1].output='';context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages[0]));assert.equal(context.denoiseVariantReady('candidate1','settings'),false);context.postJson=()=>{throw Error('must not adopt')};await context.denoiseWorkspaceFor().useSelected({method:'candidate1',preset:'settings'});
});
test('adoption binds saved options to the project and input, not the edited draft',async()=>{
 const {context,state}=setup(),calls=[];context.postJson=async(url,body)=>{calls.push({url,body});return {ok:true}};context.refresh=async()=>{};await context.denoiseWorkspaceFor().useSelected({method:'candidate1',preset:'settings'});assert.deepEqual(JSON.parse(JSON.stringify(calls)),[{url:'/api/dust-adopt',body:{values:{detector:'legacy',repair:'propainter'},source:'stabilized.mkv',project:'A',report:'report1'}}]);assert.equal(state.settings.dust.repair,'telea');
});
test('same-source project changes reset the shared frame and zoom',()=>{
 const {context,state}=setup();context.denoiseWorkspaceFor().seekFrame(23);context.denoiseCompareSetZoom(2);state.project_path='B';context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages[0]));assert.equal(context.denoiseWorkspaceSnapshot().frame,0);assert.equal(context.denoiseWorkspaceSnapshot().zoom,1);
});
test('Dust has one next processing action and never repairs or adopts a mask implicitly',()=>{
 const {context}=setup();
 assert.equal(context.dustActionState(null,false,{detector:'temporal'}).kind,'mask');
  const done=context.dustActionState({frames:12},false,{detector:'temporal'});
  assert.equal(done.kind,'');assert.match(done.message,/Mask ready · Saved/);
  const masked=context.dustActionState({frames:12},false,{detector:'temporal'},true);
 assert.equal(masked.kind,'repair');assert.equal(masked.label,'Repair masked areas');
 assert.match(masked.message,/only on request/);
 assert.equal(context.dustActionState({frames:12,repair_complete:true},false,{}).kind,'');
 assert.equal(context.dustActionState({frames:12},true,{}).kind,'');
});
test('DART labels distinguish joint inference from masked composition',()=>{
 const {context}=setup();
 assert.match(context.dustActionState(null,false,{detector:'dart'}).label,/mask & restoration/);
 assert.equal(context.dustActionState({frames:12},false,{detector:'dart'},true).label,'Compose masked repair');
 assert.equal(context.dustLabel({detector:'temporal',repair:'telea'}),'Temporal Dust · OpenCV Telea');
});
test('saved numeric values match draft strings, and changed options do not match',()=>{
 const {context}=setup();
 assert.equal(context.dustMatches({radius:3,repair:'telea'},{radius:'3',repair:'telea'}),true);
 assert.equal(context.dustMatches({radius:3,repair:'telea'},{radius:'4',repair:'telea'}),false);
});
test('rebuilding the candidate catalogue retains deliberate comparison membership',()=>{
 const {context,state}=setup();context.denoiseWorkspaceClear();context.denoiseWorkspaceToggle('candidate1','settings');
 const before=context.denoiseWorkspaceSnapshot();
 state.dust_matrix.variants.push({...state.dust_matrix.variants[0],method:'new-mask',output:'',repair_complete:false});
 vm.runInContext("denoiseWorkspaces.delete('dust')",context);
 context.denoiseWorkspaceFor('dust').restoreComparison(before);
 context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages[0]));
 assert.deepEqual(Array.from(context.denoiseWorkspaceSnapshot().selection),Array.from(before.selection));
 assert.equal(context.denoiseWorkspaceSnapshot().selected.method,'candidate1');
});
