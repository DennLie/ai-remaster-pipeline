/* Unified Remove Noise comparison; Clean Up and Detail keep their own paths. */
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const read=file=>fs.readFileSync(path.join(root,file),'utf8');
const compare=read('ai_remaster_gui/static/js/render-denoise.js');
const workspace=read('ai_remaster_gui/static/js/denoise-workspace.js');
const review=read('ai_remaster_gui/static/js/denoise-review.js');
const contrast=read('ai_remaster_gui/static/js/review-contrast.js');
const html=read('ai_remaster_gui/static/index.html');
const context=vm.createContext({state:{settings:{global:{}},stages:[]},
  document:{querySelector:()=>null,querySelectorAll:()=>[],getElementById:()=>null},
  esc:x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;')});
vm.runInContext(read('ai_remaster_gui/static/js/review-zoom.js')+'\n'+compare+'\n'+workspace+'\n'+contrast+'\n'+review,context);

test('one comparison owns exact stills, shared transport, zoom, grid and wipe',()=>{
  assert.doesNotMatch(compare,/denoiseReviewHtml|Exact frame review|Inspect this frame exactly/);
  assert.doesNotMatch(review,/createReviewZoom|data-dr-timeline|data-dr-prev|scrollIntoView/);
  assert.equal((compare.match(/id="denoiseComparePlay"/g)||[]).length,1);
  assert.equal((compare.match(/id="denoiseCompareSeek"/g)||[]).length,1);
  assert.match(compare,/data-ds-exact/);
  assert.match(workspace,/\['wipe','Wipe'\]/);
  assert.match(workspace,/data-ds-wipe="\$\{slot\}" role="slider"/);
  assert.match(workspace,/denoiseWorkspaceSetExact\(true\)/);
  assert.match(workspace,/denoise-results-grid/);
  assert.match(workspace,/denoise-original-column/);
  assert.doesNotMatch(workspace,/<select/);
  assert.ok(html.indexOf('review-contrast.js')<html.indexOf('denoise-review.js'));
  assert.ok(html.indexOf('frame-inspector.js')<html.indexOf('denoise-review.js'));
});

test('Detail still uses legacy A/B/C layout, independent of matrix membership',()=>{
  assert.match(compare,/if \(active === 'denoise'\) \{\s*denoiseWorkspaceRender\(candidates\)/);
  assert.match(compare,/denoise-layout-\$\{denoiseCompareLayout\}/);
  const previous=context.document;
  const classes=new Set(['denoise-layout-abc']);
  context.document={getElementById:()=>null,querySelector:()=>({classList:{add:value=>classes.add(value),remove:(...values)=>values.forEach(value=>classes.delete(value))}})};
  try {
    context.denoiseCompareSetLayout('ac');assert.ok(classes.has('denoise-layout-ac'));
    context.denoiseCompareSetLayout('junk');assert.ok(classes.has('denoise-layout-abc'));
  }finally{context.document=previous}
});

test('batch UI keeps validated exports and separate pipeline adoption',()=>{
  context.state.denoise_matrix={ready:4,total:9,download:true};
  const markup=context.denoiseMatrixHtml();
  assert.match(markup,/Render all 9 variants/);
  assert.match(markup,/Create frame grids as ZIP/);
  assert.match(markup,/\/api\/denoise-grid-download/);
  assert.match(markup,/4\/9 rendered/);
  assert.equal((markup.match(/role="checkbox"/g)||[]).length,9);
  assert.match(markup,/Use for pipeline/);
  assert.match(compare,/values\.length > 5/);
  assert.match(compare,/new Set\(values\)\.size/);
  context.state.denoise_matrix={};
});

test('review candidates exclude Original and partial outputs',()=>{
  const candidates=[{path:'source.mkv',kind:'source'},
    {path:'result.mkv',kind:'variant',label:'Ready'},
    {path:'result.partial.mkv',kind:'variant',label:'Unfinished'}];
  assert.deepEqual(Array.from(context.denoiseReviewVariants(candidates),item=>item.label),['Ready']);
  assert.equal(context.denoiseReviewVariants([{path:'source.mkv',kind:'source'}]).length,0);
  assert.match(workspace,/No matching output for the current input/);
});

test('chart validates simultaneous brightening and darkening, ordering and totals',()=>{
  const samples=[{frame:0,mean_abs_luma:4,brighten_mean_luma:3,darken_mean_luma:1},
    {frame:1,mean_abs_luma:5,brighten_mean_luma:2,darken_mean_luma:3}];
  const rows=context.denoiseReviewChartRows({metric:'directional_mean_luma_8bit',frames:2,samples});
  assert.equal(rows[0].brightHeight,100);assert.ok(rows[0].darkHeight>0);
  assert.ok(rows[1].brightHeight>0);assert.equal(rows[1].darkHeight,100);
  assert.throws(()=>context.denoiseReviewChartRows({metric:'directional_mean_luma_8bit',frames:2,samples:[samples[1],samples[0]]}),/Invalid/);
  assert.throws(()=>context.denoiseReviewChartRows({metric:'directional_mean_luma_8bit',frames:2,samples:[{...samples[0],mean_abs_luma:8}]}),/Invalid/);
});

test('chart highlighting follows the shared playhead without changing exact-frame measurements',()=>{
  const previous=context.document;
  let selected=false;
  const bar={dataset:{drBar:'4'},classList:{toggle(key,value){selected=value}},setAttribute(){}};
  const chart={querySelectorAll:()=>[bar]},status={textContent:''};
  context.document={querySelector:selector=>selector==='[data-dr-chart]'?chart:selector==='[data-dr-chart-status]'?status:null};
  try {
    vm.runInContext('denoiseReviewFrame=4;denoiseReviewEntries=[{frame:4,bright:1,dark:2}]',context);
    context.denoiseReviewHighlight(4);assert.equal(selected,true);assert.equal(bar.tabIndex,0);assert.match(status.textContent,/Frame 5/);
    context.denoiseReviewHighlight(8);assert.equal(selected,false);assert.equal(bar.tabIndex,0);assert.match(status.textContent,/Frame 9: not in the chart sample/);
    assert.equal(vm.runInContext('denoiseReviewFrame',context),4);
  }finally{context.document=previous}
});

test('chart navigation seeks the shared clock rather than decoding an independent inspector',()=>{
  assert.match(review,/button\.addEventListener\('click',[\s\S]*?denoiseWorkspacePlayback\?\.seekFrame\(item.frame\)/);
  assert.doesNotMatch(review,/denoiseReviewSelectFrame|denoiseReviewAlignVideo/);
  assert.ok(compare.indexOf('data-dr-chart role')<compare.indexOf('id="denoiseComparePlayers"'));
});

test('transport icon and accessible label change together',()=>{
  const previous=context.document,button={setAttribute(key,value){this[key]=value}};
  context.document={getElementById:()=>button};
  try {
    context.denoiseCompareSetPlaying(true);assert.equal(button['aria-label'],'Pause all');assert.match(button.innerHTML,/M8 5v14M16 5v14/);
    context.denoiseCompareSetPlaying(false);assert.equal(button['aria-label'],'Play all');assert.match(button.innerHTML,/m8 5 11 7-11 7Z/);
  }finally{context.document=previous}
});

test('legacy stepping clamps one- and ten-frame moves to clip boundaries',()=>{
  const previous={videos:context.denoiseCompareVideos,seek:context.denoiseCompareSeek,state:context.state};
  const video={readyState:4,currentTime:1};let target;
  context.denoiseCompareVideos=()=>[video];context.denoiseCompareSeek=value=>{target=value};context.state={source_info:{fps:'25'}};
  try {
    vm.runInContext("active='detail';denoiseCompareEffectiveLength=2",context);
    for(const [frames,expected] of [[-10,.6],[-1,.96],[1,1.04],[10,1.4]]) {context.denoiseCompareStep(frames);assert.ok(Math.abs(target-expected)<1e-9)}
    video.currentTime=.1;context.denoiseCompareStep(-10);assert.equal(target,0);
    video.currentTime=1.9;context.denoiseCompareStep(10);assert.equal(target,2);
  }finally{context.denoiseCompareVideos=previous.videos;context.denoiseCompareSeek=previous.seek;context.state=previous.state}
});

test('difference modes and amplification reject unsupported values',()=>{
  context.denoiseCompareSetDiffMode('rainbow');assert.equal(vm.runInContext('denoiseCompareDiffMode',context),'direction');
  context.denoiseCompareSetDiffMode('rgb');assert.equal(vm.runInContext('denoiseCompareDiffMode',context),'rgb');
  context.denoiseCompareSetDiffContrast(16);assert.equal(vm.runInContext('denoiseCompareDiffContrast',context),32);
  context.denoiseCompareSetDiffContrast(64);assert.equal(vm.runInContext('denoiseCompareDiffContrast',context),64);
  context.denoiseCompareSetDiffMode('direction');context.denoiseCompareSetDiffContrast(32);
});

test('pixel differences preserve raw inputs and distinguish signed luma from RGB changes',()=>{
  const a=Uint8ClampedArray.from([100,100,100,255,100,100,100,255,100,100,100,255]);
  const b=Uint8ClampedArray.from([100,100,100,255,101,101,101,255,99,99,99,255]);
  const original=Array.from(a),result=Array.from(b);
  const mapped=context.comparisonContrastPixels(a,b,3,1,'direction',32);
  assert.deepEqual(Array.from(mapped.slice(0,4)),[15,23,31,255]);
  assert.ok(mapped[5]>60 && mapped[6]>60 && mapped[4]<mapped[5]);
  assert.ok(mapped[8]>60 && mapped[8]>mapped[9]);
  assert.deepEqual(Array.from(a),original);assert.deepEqual(Array.from(b),result);
  assert.ok(context.comparisonContrastPixels(a,b,3,1,'direction',8)[5]<mapped[5]);
  const rgb=context.comparisonContrastPixels(Uint8ClampedArray.from([10,20,30,255]),Uint8ClampedArray.from([11,20,30,255]),1,1,'rgb',32);
  assert.ok(rgb[0]>60);assert.equal(rgb[1],0);assert.equal(rgb[2],0);
  assert.throws(()=>context.comparisonContrastPixels(a,b,3,1,'rainbow',32));
  assert.throws(()=>context.comparisonContrastPixels(a,b,6,1,'rgb',32));
});

test('wipe endpoints are factual and the grip fades at both image edges',()=>{
  assert.equal(context.comparisonWipeGripOpacity(640,180,0),0);
  assert.equal(context.comparisonWipeGripOpacity(640,180,50),1);
  assert.equal(context.comparisonWipeGripOpacity(640,180,100),0);
  assert.equal(context.comparisonWipeVisibleState(0,'RESULT','en'),'ONLY RESULT · 100%');
  assert.equal(context.comparisonWipeVisibleState(100,'RESULT','en'),'ONLY ORIGINAL · 100%');
});

test('decoded frame rate takes priority over source metadata, including fractional rates',()=>{
  const previous=context.state;
  try {
    context.state={source_info:{fps:'30000/1001'}};
    vm.runInContext('denoiseReviewFps=0',context);assert.ok(Math.abs(context.denoiseCompareFps()-29.97)<.001);
    vm.runInContext('denoiseReviewFps=24000/1001',context);assert.ok(Math.abs(context.denoiseCompareFps()-23.976)<.001);
  }finally{context.state=previous;vm.runInContext('denoiseReviewFps=0',context)}
});

test('server inspection remains read-only and verifies signed output lineage',()=>{
  const api=read('ai_remaster_gui/http_handler.py'),backend=read('ai_remaster_gui/denoise_review.py');
  assert.match(api,/parsed\.path == "\/api\/denoise-inspector"/);
  assert.match(backend,/provenance\.get\("tool"\) != "denoise_video\.py"/);
  assert.match(backend,/output\.resolve\(\)\.parent != _OUTPUTS\.resolve\(\)/);
  assert.match(backend,/is_unfinished_media_artifact\(output\)/);
  assert.match(review,/cleanupDifferencePixels/);
});

test('Clean Up retains exact stills, contrast controls and its lightbox',()=>{
  const frame=read('ai_remaster_gui/static/js/frame-inspector.js');
  assert.match(frame,/cleanupSetDifferenceMode/);assert.match(frame,/cleanupSetDifferenceContrast/);
  assert.match(frame,/const diff = cleanupDifferencePixels\(a\.data, b\.data, width, height\)/);
  assert.match(contrast,/dialog\.showModal\(\)/);
  assert.match(contrast,/event\.target === dialog\.querySelector\('\[data-lightbox-image\]'\)/);
});
