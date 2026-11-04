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
  for(const file of ['render-denoise.js','render-deflicker.js','denoise-workspace.js']) vm.runInContext(fs.readFileSync(path.join(base,file),'utf8'),context);
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
  assert.equal(evaluate('denoiseWorkspaceVariants.length'),5);
  assert.equal(evaluate('denoiseComparePaths.length'),6);
  assert.equal(evaluate('denoiseComparePaths[0]'),state.deflicker_matrix.source);
  assert.notEqual(evaluate('denoiseComparePaths[0]'),state.settings.global.source);
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
  evaluate("denoiseWorkspaceFrame=18;denoiseCompareZoom=3;denoiseComparePan={x:.2,y:-.1};denoiseWorkspaceViews.set(1,'difference')");
  initialize('deflicker');context.denoiseWorkspaceSelectAll();context.denoiseWorkspaceToggle('ffmpeg_am','settings');
  evaluate("denoiseWorkspaceFrame=7;denoiseWorkspaceExact=true;denoiseCompareNative=true;denoiseCompareGrid=true;denoiseWorkspaceViews.set(1,'wipe');denoiseWorkspaceWipes.set(1,62)");
  initialize('denoise');
  assert.equal(evaluate('denoiseWorkspaceSelection.size'),9);assert.equal(evaluate('denoiseWorkspaceFrame'),18);
  assert.equal(evaluate('denoiseCompareZoom'),3);assert.equal(evaluate('denoiseComparePan.x'),.2);
  assert.equal(evaluate('denoiseWorkspaceViews.get(1)'),'difference');assert.equal(evaluate('denoiseWorkspaceExact'),false);
  assert.equal(evaluate('denoiseCompareGrid'),false);
  initialize('deflicker');assert.equal(evaluate('denoiseWorkspaceSelection.size'),4);
  assert.equal(evaluate('denoiseWorkspaceFrame'),7);assert.equal(evaluate('denoiseWorkspaceExact'),true);
  assert.equal(evaluate('denoiseCompareNative'),true);assert.equal(evaluate('denoiseWorkspaceViews.get(1)'),'wipe');
  assert.equal(evaluate('denoiseWorkspaceWipes.get(1)'),62);
  assert.equal(evaluate('denoiseCompareGrid'),true);
  context.denoiseWorkspaceDeactivate();context.active='detail';evaluate('denoiseCompareZoom=5;denoiseCompareGrid=false');
  initialize('deflicker');assert.equal(evaluate('denoiseCompareGrid'),true);assert.equal(evaluate('denoiseWorkspaceFrame'),7);
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

test('pipeline adoption changes only the Deflicker method, without resetting its parameters or comparison selection',async()=>{
  const {context,state,evaluate,initialize}=setup();initialize('deflicker');context.denoiseWorkspaceSelectAll();context.denoiseWorkspaceFocus(3);
  const calls=[];context.postJson=async(url,body)=>{calls.push({url,body});return{ok:true}};context.refresh=async()=>{};
  await context.denoiseMatrixUseSelected();
  assert.deepEqual(JSON.parse(JSON.stringify(calls)),[{url:'/api/settings',body:{stage:'deflicker',values:{method:'ffmpeg_am',force:'false'}}}]);
  assert.equal(evaluate('denoiseWorkspaceSelection.size'),5);assert.equal(state.settings.deflicker.window,'5');
});

test('Clean Up no longer renders Deflicker controls, while both tabs use the same viewer factory',()=>{
  const helpers=fs.readFileSync(path.join(base,'render-helpers.js'),'utf8');
  const cleanup=helpers.slice(helpers.indexOf('function drawCleanup('),helpers.indexOf('function drawStabilization('));
  assert.doesNotMatch(cleanup,/data-deflicker-only|light_deflicker|deflicker_method|localized_confirm/);
  assert.match(helpers,/st.key === 'deflicker'\) return drawDenoise/);
  const playback=fs.readFileSync(path.join(base,'denoise-playback.js'),'utf8');
  assert.match(playback,/body.stage='deflicker'/);
  const still=fs.readFileSync(path.join(base,'denoise-review.js'),'utf8');
  assert.match(still,/&stage=deflicker/);
});
