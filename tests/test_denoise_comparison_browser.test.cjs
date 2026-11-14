/* Contract tests for optional browser A/B/C denoising comparison. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const script = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/render-denoise.js'), 'utf8');
const context = vm.createContext({
  state: {
    settings: { global: { source: 'input/original.webm' } },
    denoise_matrix: {source: 'input/original.webm'},
    stages: [{key: 'denoise', files: [
      {path: 'intermediate/denoised/arp_hqdn3d_gentle.mkv'},
      {path: 'intermediate/denoised/arp_nlmeans_gentle.mkv'},
      {path: 'intermediate/denoised/arp_notes.txt'},
    ]}],
    source_info: {fps: '30 fps'},
  },
  document: {querySelectorAll: () => [], getElementById: () => null},
});
vm.runInContext(script, context);

test('browser view is wired into active GUI, without touching the original', () => {
  const html = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/index.html'), 'utf8');
  const helpers = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/render-helpers.js'), 'utf8');
  assert.match(html, /<script src="\/static\/js\/render-denoise.js"><\/script>/);
  assert.match(helpers, /st.key === 'denoise'\) return drawDenoise/);
  assert.match(script, /mediaClip\(path, denoiseCompareStart/);
  assert.match(script, /Browser synchronization is approximate/);
});

test('only video candidates are selectable, and source stays first', () => {
  const found = context.denoiseCompareCandidates(context.state.stages[0]);
  assert.deepEqual(Array.from(found.map(x => x.path)), [
    'input/original.webm',
    'intermediate/denoised/arp_hqdn3d_gentle.mkv',
    'intermediate/denoised/arp_nlmeans_gentle.mkv',
  ]);
  context.denoiseCompareInitializePaths(found);
  assert.deepEqual(Array.from(vm.runInContext('denoiseComparePaths', context)), Array.from(found, x => x.path));
});

test('missing stage input never falls back to the raw original',()=>{
  const source=context.state.denoise_matrix.source;
  try {
    context.state.denoise_matrix.source='';
    const found=context.denoiseCompareCandidates({key:'denoise',files:[]});
    assert.equal(found.length,0);
  } finally {context.state.denoise_matrix.source=source}
});

test('reuse a manual choice, but reject removed cached files', () => {
  vm.runInContext("denoiseComparePaths = ['input/original.webm', 'intermediate/denoised/arp_nlmeans_gentle.mkv', 'deleted.mkv']", context);
  const options = context.denoiseCompareCandidates(context.state.stages[0]);
  context.denoiseCompareInitializePaths(options);
  const result = Array.from(vm.runInContext('denoiseComparePaths', context));
  assert.equal(result[1], 'intermediate/denoised/arp_nlmeans_gentle.mkv');
  assert.equal(result[2], 'intermediate/denoised/arp_hqdn3d_gentle.mkv');
});

test('clip and playback contracts reject mismatched geometry or duration', () => {
  assert.deepEqual({...context.denoiseCompareClip('300', '4')}, {start: 300, duration: 4});
  assert.equal(context.denoiseCompareClip('-1', '4'), null);
  assert.equal(context.denoiseCompareClip('abc', '4'), null);
  assert.equal(context.denoiseCompareClip('10', '13'), null);
  assert.equal(context.denoiseCompareClip('10', '0'), null);
  const a = {readyState: 2, videoWidth: 960, videoHeight: 720, duration: 4};
  const b = {...a};
  const c = {...a};
  assert.equal(context.denoiseCompareCompatible([a,b,c]), true);
  assert.equal(context.denoiseCompareCompatible([a,{...b, videoWidth: 1920},c]), false);
  assert.equal(context.denoiseCompareCompatible([a,b,{...c, duration: 3.6}]), false);
  assert.equal(context.denoiseCompareCompatible([a,b,{...c, readyState: 0}]), false);
});

test('unfinished FFmpeg files never appear in A/B/C, even in stale GUI snapshots', () => {
  const stage = {key:'denoise',files:[
    {path:'intermediate/denoised/Neighbors_hqdn3d_gentle.partial.mkv'},
    {path:'intermediate/denoised/Neighbors_hqdn3d_gentle.mkv'},
    {path:'intermediate/denoised/another.TMP.MKV'},
  ]};
  assert.equal(context.denoisePreviewIsFinal('C:\\film\\Neighbors.partial.mkv'), false);
  assert.equal(context.denoisePreviewIsFinal('intermediate/denoised/Neighbors_hqdn3d_gentle.mkv'), true);
  let items = context.denoiseCompareCandidates(stage);
  assert.deepEqual(Array.from(items, x => x.path), [
    'input/original.webm',
    'intermediate/denoised/Neighbors_hqdn3d_gentle.mkv',
  ]);
  vm.runInContext("denoiseComparePaths = ['input/original.webm', 'intermediate/denoised/Neighbors_hqdn3d_gentle.partial.mkv', '']", context);
  context.denoiseCompareInitializePaths(items);
  assert.deepEqual(Array.from(vm.runInContext('denoiseComparePaths', context)), [
    'input/original.webm',
    'intermediate/denoised/Neighbors_hqdn3d_gentle.mkv',
    '',
  ]);
  // While the encoder writes its partial output, show the original once and
  // let B/C wait instead of showing false "duplicate originals" or errors.
  stage.files = [{path:'intermediate/denoised/Neighbors_hqdn3d_gentle.partial.mkv'}];
  items = context.denoiseCompareCandidates(stage);
  context.denoiseCompareInitializePaths(items);
  assert.deepEqual(Array.from(vm.runInContext('denoiseComparePaths', context)), [
    'input/original.webm', '', '',
  ]);
});
