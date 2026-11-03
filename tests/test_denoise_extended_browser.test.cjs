/* Extended Remove Noise (A/B/C + exact still + wipe + directional bars). */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const compare = fs.readFileSync(path.join(root,'ai_remaster_gui/static/js/render-denoise.js'),'utf8');
const review = fs.readFileSync(path.join(root,'ai_remaster_gui/static/js/denoise-review.js'),'utf8');
const html = fs.readFileSync(path.join(root,'ai_remaster_gui/static/index.html'),'utf8');
const css = fs.readFileSync(path.join(root,'ai_remaster_gui/static/styles.css'),'utf8');
const api = fs.readFileSync(path.join(root,'ai_remaster_gui/http_handler.py'),'utf8');

const context = vm.createContext({
  state:{settings:{global:{source:'evaluation/Neighbors_72.mkv'}},stages:[]},
  document:{querySelector:()=>null,querySelectorAll:()=>[],getElementById:()=>null},
  esc:x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;'),
});
vm.runInContext(compare+'\n'+review, context);

test('main page retains A/B/C and adds exact review under it only for denoise',()=>{
  assert.match(compare,/st\.key === 'denoise' \? denoiseReviewHtml\(candidates\)/);
  assert.match(compare,/denoiseCompareLayout/);
  assert.match(compare,/denoise-layout-\$\{denoiseCompareLayout\}/);
  assert.match(compare,/denoiseReviewStart\(\)/);
  assert.match(html, /<script src="\/static\/js\/frame-inspector.js"><\/script>/);
  assert.match(html, /<script src="\/static\/js\/denoise-review.js"><\/script>/);
  assert.ok(html.indexOf('frame-inspector.js') < html.indexOf('denoise-review.js'));
  assert.match(css,/\.denoise-review-stills/);
  assert.match(css,/\.denoise-layout-ac .denoise-compare-panel:nth-child\(2\)/);
});

test('variant selection excludes Original/partials; no output means a factual empty state',()=>{
  const withNoOutput=context.denoiseReviewHtml([{path:'evaluation/Neighbors_72.mkv',label:'Original'}]);
  assert.match(withNoOutput,/Noch keine fertige/);
  const choices=[
    {path:'evaluation/Neighbors_72.mkv',label:'Original'},
    {path:'intermediate/denoised/Neighbors_hqdn3d_gentle.mkv',label:'HQDN3D'},
    {path:'intermediate/denoised/Neighbors_nlmeans_gentle.partial.mkv',label:'Unfinished'},
    {path:'intermediate/denoised/Neighbors_nlmeans_gentle.mkv',label:'NLMeans'},
  ];
  const variants=context.denoiseReviewVariants(choices);
  assert.deepEqual(variants.map(x=>x.label),['HQDN3D','NLMeans']);
  const markup=context.denoiseReviewHtml(choices);
  assert.match(markup,/Frame-Änderungen/);
  assert.match(markup,/Wischvergleich/);
  assert.match(markup,/PIXELDIFFERENZ ×8/);
  assert.match(markup,/Exakte.*Frame/);
  assert.doesNotMatch(markup,/Unfinished/);
});

test('directional chart shows simultaneous brightening and darkening',()=>{
  const samples=[
    {frame:0,mean_abs_luma:4,brighten_mean_luma:3,darken_mean_luma:1},
    {frame:1,mean_abs_luma:5,brighten_mean_luma:2,darken_mean_luma:3},
    {frame:2,mean_abs_luma:2,brighten_mean_luma:0,darken_mean_luma:2}
  ];
  const rows=context.denoiseReviewChartRows({metric:'directional_mean_luma_8bit',frames:3,samples});
  assert.equal(rows.length,3);
  assert.equal(rows[0].brightHeight,100);
  assert.ok(rows[0].darkHeight>0);
  assert.ok(rows[1].brightHeight>0);
  assert.equal(rows[2].brightHeight,0);
  assert.throws(()=>context.denoiseReviewChartRows({
    metric:'directional_mean_luma_8bit',frames:3,
    samples:[samples[1],samples[0]]
  }),/Ungültige/);
  assert.throws(()=>context.denoiseReviewChartRows({
    metric:'directional_mean_luma_8bit',frames:3,
    samples:[{frame:0,mean_abs_luma:5,brighten_mean_luma:5,darken_mean_luma:5}]
  }),/Ungültige/);
});

test('server API remains read-only and validates signed denoise output',()=>{
  assert.match(api,/parsed\.path == "\/api\/denoise-inspector"/);
  assert.match(api,/parsed\.path == "\/api\/denoise-frame-differences"/);
  const backend=fs.readFileSync(path.join(root,'ai_remaster_gui/denoise_review.py'),'utf8');
  assert.match(backend,/provenance\.get\("tool"\) != "denoise_video\.py"/);
  assert.match(backend,/provenance\.get\("source"/);
  assert.match(backend,/is_unfinished_media_artifact\(output\)/);
  assert.match(backend,/output\.resolve\(\)\.parent != _OUTPUTS\.resolve\(\)/);
  assert.match(backend,/inspect_denoise/);
  assert.match(review,/cleanupDifferencePixels/);
  assert.match(review,/denoiseReviewAlignVideo/);
});

test('layout controls select enlarged A/B or A/C without removing video data',()=>{
  vm.runInContext("denoiseCompareLayout = 'abc'",context);
  const classes=new Set(['denoise-compare-grid','denoise-layout-abc']);
  context.document.querySelector=()=>({classList:{
    add(x){classes.add(x)},remove(...values){values.forEach(v=>classes.delete(v))}
  }});
  context.denoiseCompareSetLayout('ac');
  assert.equal(vm.runInContext('denoiseCompareLayout',context),'ac');
  assert.ok(classes.has('denoise-layout-ac'));
  assert.ok(!classes.has('denoise-layout-abc'));
  context.denoiseCompareSetLayout('junk');
  assert.equal(vm.runInContext('denoiseCompareLayout',context),'abc');
});
