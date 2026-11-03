/* Extended Remove Noise (A/B/C + exact still + wipe + directional bars). */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const compare = fs.readFileSync(path.join(root,'ai_remaster_gui/static/js/render-denoise.js'),'utf8');
const review = fs.readFileSync(path.join(root,'ai_remaster_gui/static/js/denoise-review.js'),'utf8');
const contrast = fs.readFileSync(path.join(root,'ai_remaster_gui/static/js/review-contrast.js'),'utf8');
const html = fs.readFileSync(path.join(root,'ai_remaster_gui/static/index.html'),'utf8');
const css = fs.readFileSync(path.join(root,'ai_remaster_gui/static/styles.css'),'utf8');
const api = fs.readFileSync(path.join(root,'ai_remaster_gui/http_handler.py'),'utf8');

const context = vm.createContext({
  state:{settings:{global:{source:'evaluation/Neighbors_72.mkv'}},stages:[]},
  document:{querySelector:()=>null,querySelectorAll:()=>[],getElementById:()=>null},
  esc:x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('"','&quot;'),
});
vm.runInContext(compare+'\n'+contrast+'\n'+review, context);

test('main page retains A/B/C and adds exact review under it only for denoise',()=>{
  assert.match(compare,/st\.key === 'denoise' \? denoiseReviewHtml\(candidates\)/);
  assert.match(compare,/denoiseCompareLayout/);
  assert.match(compare,/denoise-layout-\$\{denoiseCompareLayout\}/);
  assert.match(compare,/denoiseReviewStart\(\)/);
  assert.match(html, /<script src="\/static\/js\/frame-inspector.js"><\/script>/);
  assert.match(html, /<script src="\/static\/js\/denoise-review.js"><\/script>/);
  assert.ok(html.indexOf('frame-inspector.js') < html.indexOf('denoise-review.js'));
  assert.match(css,/\.denoise-review-stills/);
  assert.match(css,/dialog\.comparison-review-lightbox/);
  assert.match(html,/<script src="\/static\/js\/review-contrast.js"><\/script>/);
  assert.ok(html.indexOf('review-contrast.js') < html.indexOf('frame-inspector.js'));
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
  assert.match(markup,/PIXELDIFFERENZ · verstärkt/);
  assert.match(markup,/data-dr-diff-mode/);
  assert.match(markup,/RGB: Veränderung je Farbkanal/);
  assert.match(markup,/data-dr-diff-contrast/);
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

test('raw image differences remain exact while RGB and signed false colors gain contrast',()=>{
  const a=Uint8ClampedArray.from([100,100,100,255,100,100,100,255,100,100,100,255]);
  const b=Uint8ClampedArray.from([100,100,100,255,101,101,101,255,99,99,99,255]);
  const orig=Array.from(a), rendered=Array.from(b);
  const directional=context.comparisonContrastPixels(a,b,3,1,'direction',32);
  const brighter=directional.slice(4,8);
  const darker=directional.slice(8,12);
  assert.deepEqual(Array.from(directional.slice(0,4)),[15,23,31,255]);
  assert.ok(brighter[1]>60 && brighter[2]>60 && brighter[0]<brighter[1]);
  assert.ok(darker[0]>60 && darker[1]>20 && darker[0]>darker[1]);
  assert.deepEqual(Array.from(a),orig);
  assert.deepEqual(Array.from(b),rendered);
  const lower=context.comparisonContrastPixels(a,b,3,1,'direction',8);
  assert.ok(lower[5]<directional[5]);
  const rgb=context.comparisonContrastPixels(
    Uint8ClampedArray.from([10,20,30,255]),
    Uint8ClampedArray.from([11,20,30,255]),1,1,'rgb',32);
  assert.ok(rgb[0]>60);
  assert.equal(rgb[1],0);
  assert.equal(rgb[2],0);
  assert.equal(rgb[3],255);
  assert.throws(()=>context.comparisonContrastPixels(a,b,3,1,'rainbow',32),/Ungültige/);
  assert.throws(()=>context.comparisonContrastPixels(a,b,3,1,'rgb',0),/Ungültige/);
  assert.throws(()=>context.comparisonContrastPixels(a,b,6,1,'rgb',32),/Ungültige/);
});

test('the enlarged still dialog closes on clicking the image again',()=>{
  const title={textContent:''};
  const image={src:'',id:'large'};
  const closer={focus(){this.focused=true}};
  const handlers={};
  let modal=false, removed=false, restored=false;
  const dialog={
    className:'',innerHTML:'',attrs:{},
    setAttribute(key,value){this.attrs[key]=value},
    querySelector(key){
      return key==='[data-lightbox-title]' ? title
        : key==='[data-lightbox-image]' ? image : closer;
    },
    addEventListener(kind,func){handlers[kind]=func},
    showModal(){modal=true},
    close(){modal=false;handlers.close?.()},
    remove(){removed=true},
  };
  const old=context.document;
  context.document={
    querySelector(){return null},
    createElement(tag){assert.equal(tag,'dialog');return dialog},
    activeElement:{isConnected:true,focus(){restored=true}},
    body:{appendChild(element){assert.equal(element,dialog)}},
  };
  const clicked={
    tagName:'CANVAS',width:960,height:720,
    toDataURL(){return 'data:image/png;base64,iVBORw0KGgo='},
    closest(){return {querySelector(){return {textContent:'PIXELDIFFERENZ'}}}},
    getAttribute(){return null},
  };
  try {
    assert.equal(context.comparisonLightboxOpen(clicked),true);
    assert.equal(modal,true);
    assert.equal(title.textContent,'PIXELDIFFERENZ');
    assert.ok(image.src.startsWith('data:image/png;'));
    assert.equal(closer.focused,true);
    handlers.click({target:image});
    assert.equal(modal,false);
    assert.equal(removed,true);
    assert.equal(restored,true);
  } finally {
    context.document=old;
  }
});

test('Clean Up receives the same contrast chooser without changing measured luma',()=>{
  const frame=fs.readFileSync(path.join(root,'ai_remaster_gui/static/js/frame-inspector.js'),'utf8');
  assert.match(frame,/cleanupSetDifferenceMode/);
  assert.match(frame,/cleanupSetDifferenceContrast/);
  assert.match(frame,/comparisonContrastPixels\(a\.data,b\.data,width,height/);
  assert.match(frame,/const diff = cleanupDifferencePixels\(a\.data, b\.data, width, height\)/);
  assert.match(frame,/data-fi-diff-mode/);
  assert.match(frame,/data-fi-difference/);
  assert.match(contrast,/dialog\.showModal\(\)/);
  assert.match(contrast,/event\.target === dialog\.querySelector\('\[data-lightbox-image\]'\)/);
});
