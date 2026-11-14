const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const base=path.join(__dirname,'../ai_remaster_gui/static/js');
function setup() {
  const methods=['arp','ffmpeg_median','ffmpeg_am','localized_research','localized_stream'];
  const noise=['hqdn3d','nlmeans','bm3d'].flatMap(method=>['gentle','balanced','strong'].map(preset=>({method,preset,
    output:`intermediate/denoised/source_denoised_${method}_${preset}_abc123.mkv`,revision:'v1'})));
  const flicker=methods.map(method=>({method,preset:'settings',output:`intermediate/deflickered/source_deflickered_${method}_abc123.mkv`,revision:'v1'}));
  const state={settings:{global:{source:'raw.mkv'},denoise:{method:'bm3d',preset:'balanced'},deflicker:{method:'arp',strength:'.65',max_correction:'.08',window:'5'}},
    source_info:{duration:'1200',fps:'30'},
    denoise_matrix:{source:'raw.mkv',source_revision:'s1',variants:noise},
    deflicker_matrix:{source:noise[7].output,source_revision:'s2',variants:flicker},
    stages:[{key:'denoise',files:noise.map(item=>({path:item.output}))},
      {key:'deflicker',files:flicker.map(item=>({path:item.output})),fields:[['method','Method'],['strength','ARP strength'],['window','FFmpeg window'],['localized_confirm_single_shot','Confirm short'],['localized_confirm_film_experiment','Confirm streaming']]}]};
  const context=vm.createContext({state,active:'denoise',denoiseReviewVariant:'',denoiseReviewFps:0,
    document:{querySelector(){return null},querySelectorAll(){return[]},getElementById(){return null}},
    parseDuration:Number,esc:String,fieldHtml:(stage,field)=>`<input data-field="${field[0]}">`});
  for(const file of ['review-zoom.js','render-denoise.js','render-deflicker.js','denoise-workspace.js']) vm.runInContext(fs.readFileSync(path.join(base,file),'utf8'),context);
  const evaluate=expression=>vm.runInContext(expression,context);
  const initialize=key=>{
    context.active=key;
    context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages.find(stage=>stage.key===key)));
  };
  return{context,state,evaluate,initialize};
}

test('Deflicker uses five real methods, the actual upstream input and the full comparison timeline',()=>{
  const {context,state,evaluate,initialize}=setup();initialize('deflicker');
  context.denoiseWorkspaceSelectAll();context.denoiseCompareInitializePaths(context.denoiseCompareCandidates(state.stages[1]));
  assert.equal(context.denoiseWorkspaceSnapshot().variants.length,5);
  assert.equal(context.denoiseWorkspaceSnapshot().paths.length,6);
  assert.equal(context.denoiseWorkspaceSnapshot().paths[0],state.deflicker_matrix.source);
  assert.notEqual(context.denoiseWorkspaceSnapshot().paths[0],state.settings.global.source);
  assert.equal(context.denoiseCompareFullDuration(),1200);
  const markup=context.denoiseMatrixHtml();
  assert.equal((markup.match(/role="checkbox"/g)||[]).length,5);
  assert.match(markup,/Localized methods are experimental/);
  assert.match(markup,/at most 240 frames/);
  assert.match(markup,/data-field="localized_confirm_single_shot"/);
  assert.doesNotMatch(markup.replace(/<[^>]*>/g,''),/gentle|balanced|strong|Render all 9/);
});

test('Noise and Deflicker retain independent membership, focus, playhead and viewport settings',()=>{
  const {context,evaluate,initialize}=setup();initialize('denoise');context.denoiseWorkspaceSelectAll();
  context.denoiseWorkspaceFor().seekFrame(18);context.denoiseWorkspaceFor().setViewport({zoom:3,pan:{x:.2,y:-.1}});
  context.denoiseWorkspaceSetView(1,'difference');
  initialize('deflicker');context.denoiseWorkspaceSelectAll();context.denoiseWorkspaceToggle('ffmpeg_am','settings');
  context.denoiseWorkspaceFor().seekFrame(7);context.denoiseWorkspaceFor().setViewport({native:true,grid:true});
  context.denoiseWorkspaceSetView(1,'wipe');context.denoiseWorkspaceFor().setWipe(1,62);
  initialize('denoise');
  let view=context.denoiseWorkspaceSnapshot();
  assert.equal(view.selection.length,9);assert.equal(view.frame,18);
  assert.equal(view.zoom,3);assert.equal(view.pan.x,.2);
  assert.equal(view.views.get(1),'difference');assert.equal(view.exact,false);assert.equal(view.grid,false);
  initialize('deflicker');view=context.denoiseWorkspaceSnapshot();assert.equal(view.selection.length,4);
  assert.equal(view.frame,7);assert.equal(view.exact,true);
  assert.equal(view.native,true);assert.equal(view.views.get(1),'wipe');
  assert.equal(view.wipes.get(1),62);assert.equal(view.grid,true);
  context.denoiseWorkspaceDeactivate();context.active='detail';evaluate('denoiseCompareZoom=5;denoiseCompareGrid=false');
  initialize('deflicker');view=context.denoiseWorkspaceSnapshot();assert.equal(view.grid,true);assert.equal(view.frame,7);
});

test('selection never adopts or generates; missing Deflicker outputs require an explicit subset request',async()=>{
  const {context,state,initialize}=setup();state.deflicker_matrix.variants=[];initialize('deflicker');
  const before=JSON.stringify(state.settings),calls=[];
  context.postJson=async(url,body)=>{calls.push({url,body});return{ok:true}};context.refresh=async()=>{};
  context.denoiseWorkspaceSelectAll();assert.equal(calls.length,0);assert.equal(JSON.stringify(state.settings),before);
  await context.denoiseWorkspaceGenerate();assert.equal(calls.length,1);assert.equal(calls[0].url,'/api/deflicker-batch');
  assert.equal(calls[0].body.variants.length,5);
  assert.ok(calls[0].body.variants.every(item=>item.preset==='settings'&&!('output' in item)));
  assert.equal(JSON.stringify(state.settings),before);
});

test('a late render error belongs to its initiating tab, including after returning to it',async()=>{
  const {context,state,initialize}=setup();
  const status={textContent:'',dataset:{}};
  context.document.querySelector=selector=>selector==='[data-dm-generation-status]'?status:null;
  state.denoise_matrix.variants=[];
  initialize('denoise');context.denoiseWorkspaceSelectAll();
  let finish;
  context.postJson=()=>new Promise(resolve=>{finish=resolve});
  const pending=context.denoiseWorkspaceGenerate();
  initialize('deflicker');
  finish({ok:false,error:'Remove Noise renderer unavailable'});
  await pending;
  context.denoiseWorkspaceUpdateSelectionUi();
  assert.doesNotMatch(status.textContent,/Remove Noise renderer unavailable/);
  initialize('denoise');context.denoiseWorkspaceUpdateSelectionUi();
  assert.match(status.textContent,/Remove Noise renderer unavailable/);
});

test('parameter changes preserve the viewport and selection but reject obsolete generation errors',async()=>{
  const {context,state,initialize}=setup();
  state.deflicker_matrix.variants=[];initialize('deflicker');context.denoiseWorkspaceSelectAll();
  context.denoiseWorkspaceFor().seekFrame(17);context.denoiseCompareSetZoom(4);
  context.denoiseWorkspaceSetView(1,'difference');
  let finish;const calls=[];
  context.postJson=(url,body)=>{calls.push({url,body});return new Promise(resolve=>{finish=resolve})};
  const pending=context.denoiseWorkspaceGenerate();
  state.settings.deflicker.window='7';initialize('deflicker');
  finish({ok:false,error:'obsolete window'});await pending;
  const view=context.denoiseWorkspaceSnapshot();
  assert.equal(view.frame,17);assert.equal(view.zoom,4);assert.equal(view.selection.length,5);
  assert.equal(view.views.get(1),'difference');assert.equal(view.generationError,'');
  assert.equal(calls.length,1);
});

test('a new processing input resets frame and fit, retains membership, and cannot receive an old error',async()=>{
  const {context,state,initialize}=setup();
  state.deflicker_matrix.variants=[];initialize('deflicker');context.denoiseWorkspaceSelectAll();
  context.denoiseWorkspaceFor().seekFrame(17);context.denoiseWorkspaceFor().setViewport({zoom:4,pan:{x:.2,y:.1},native:true});
  let finish;context.postJson=()=>new Promise(resolve=>{finish=resolve});
  const pending=context.denoiseWorkspaceGenerate();
  state.deflicker_matrix.source='another-denoised.mkv';state.deflicker_matrix.source_revision='new-input';
  initialize('deflicker');finish({ok:false,error:'old input'});await pending;
  const view=context.denoiseWorkspaceSnapshot();
  assert.equal(view.frame,0);assert.equal(view.zoom,1);assert.equal(view.pan.x,0);assert.equal(view.native,false);
  assert.equal(view.selection.length,5);assert.equal(view.generationError,'');
});

test('unmounting releases the view, not the render request or the single-renderer lock',async()=>{
  const {context,state,initialize}=setup();state.denoise_matrix.variants=[];state.deflicker_matrix.variants=[];
  initialize('denoise');context.denoiseWorkspaceSelectAll();
  let finish,calls=0;context.refresh=async()=>{};
  context.postJson=()=>{calls++;return new Promise(resolve=>{finish=resolve})};
  const pending=context.denoiseWorkspaceGenerate();
  context.denoiseWorkspaceDeactivate();initialize('deflicker');context.denoiseWorkspaceSelectAll();
  await context.denoiseWorkspaceGenerate();assert.equal(calls,1);
  assert.equal(context.denoiseWorkspaceSnapshot().generationPending,false);
  assert.equal(context.denoiseWorkspaceSnapshot('denoise').generationPending,true);
  finish({ok:true});await pending;
  assert.equal(context.denoiseWorkspaceSnapshot('denoise').generationPending,false);
});

test('a returned snapshot cannot mutate tab state',()=>{
  const {context,initialize}=setup();initialize('deflicker');context.denoiseWorkspaceSelectAll();
  const view=context.denoiseWorkspaceSnapshot();view.pan.x=1;view.selected.method='off';view.selection.length=0;view.views.set(1,'wipe');
  const next=context.denoiseWorkspaceSnapshot();
  assert.equal(next.pan.x,0);assert.notEqual(next.selected.method,'off');assert.equal(next.selection.length,5);assert.equal(next.views.size,0);
});

test('pipeline adoption changes only the Deflicker method, without resetting its parameters or comparison selection',async()=>{
  const {context,state,evaluate,initialize}=setup();initialize('deflicker');context.denoiseWorkspaceSelectAll();context.denoiseWorkspaceFocus(3);
  const calls=[];context.postJson=async(url,body)=>{calls.push({url,body});return{ok:true}};context.refresh=async()=>{};
  await context.denoiseMatrixUseSelected();
  assert.deepEqual(JSON.parse(JSON.stringify(calls)),[{url:'/api/settings',body:{stage:'deflicker',values:{method:'ffmpeg_am',force:'false'}}}]);
  assert.equal(context.denoiseWorkspaceSnapshot().selection.length,5);assert.equal(state.settings.deflicker.window,'5');
});

test('late pipeline-adoption failures cannot write into a different comparison tab',async()=>{
  const {context,initialize}=setup();const notice={textContent:'',dataset:{}};
  context.document.querySelector=selector=>selector==='[data-dm-notice]'?notice:null;
  initialize('denoise');context.denoiseWorkspaceSelectAll();context.denoiseWorkspaceFocus(2);
  let finish;context.postJson=()=>new Promise(resolve=>{finish=resolve});
  const pending=context.denoiseMatrixUseSelected();
  initialize('deflicker');finish({ok:false,error:'owned adoption failure'});await pending;
  context.denoiseWorkspaceUpdateSelectionUi();assert.equal(notice.textContent,'');
  initialize('denoise');context.denoiseWorkspaceUpdateSelectionUi();assert.match(notice.textContent,/owned adoption failure/);
});

test('Clean Up no longer renders Deflicker controls, while both tabs use the same viewer factory',()=>{
  const helpers=fs.readFileSync(path.join(base,'render-helpers.js'),'utf8');
  const cleanup=helpers.slice(helpers.indexOf('function drawCleanup('),helpers.indexOf('function drawStabilization('));
  assert.doesNotMatch(cleanup,/data-deflicker-only|light_deflicker|deflicker_method|localized_confirm/);
  assert.match(helpers,/st.key === 'deflicker'\) return drawDenoise/);
  const playback=fs.readFileSync(path.join(base,'denoise-playback.js'),'utf8');
  assert.match(playback,/body.stage=options.stage\(\)/);
  const still=fs.readFileSync(path.join(base,'denoise-review.js'),'utf8');
  assert.match(still,/encodeURIComponent\(options.stage\?\.\(\) \|\| 'denoise'\)/);
});
