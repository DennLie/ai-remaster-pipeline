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
test('original source selection is analysis-only and keeps catalogues and settings separate',()=>{
 const {context,state}=setup(),before=JSON.stringify(state.settings),calls=[];
 state.dust_matrix.original={source:'raw.mkv',source_revision:'original-v1',variants:[]};
 context.drawDust=()=>{};context.postJson=(...args)=>{calls.push(args);throw Error('no processing on source change')};
 const old=context.dustContext();context.dustSelectSource('original');
 assert.notEqual(context.dustContext(),old);assert.equal(context.dustMatrix().source,'raw.mkv');
 assert.equal(context.denoiseMethods().length,0);assert.equal(context.denoiseCompareCandidates(state.stages[0])[0].path,'raw.mkv');
 state.processing_input={problem:'Run stabilization first'};assert.equal(context.dustInputBlocked(),false);
 assert.equal(JSON.stringify(state.settings),before);assert.equal(calls.length,0);
 context.dustSelectSource('stage');assert.equal(context.dustMatrix().source,'stabilized.mkv');assert.equal(context.dustInputBlocked(),true);
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
  assert.equal(context.dustMatches({repair:'telea'},{repair:'telea',mask_version:'a'.repeat(32)}),false);
  assert.equal(context.dustMatches({repair:'telea',mask_version:'a'.repeat(32)},{repair:'telea',mask_version:'a'.repeat(32)}),true);
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

test('recalculate requires consent and starts only a new mask version, never adoption',async()=>{
  const {context,state}=setup();const original=JSON.stringify(state.settings),requests=[];
  context.confirm=()=>false;
  vm.runInContext("dustDraft={detector:'dart',repair:'dart'};dustStatus={frames:12,repair_complete:true}",context);
  context.postJson=async(url,body)=>{requests.push({url,body});return {ok:false,error:'fixture launch rejected'}};
  await context.dustRun(true);assert.equal(requests.length,0);
  let question='';context.confirm=text=>{question=text;return true;};
  context.crypto={randomUUID:()=> 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'};
  await context.dustRun(true);
  assert.match(question,/GPU inference again/);assert.match(question,/pipeline selection stay unchanged/);
  assert.equal(requests.length,1);assert.equal(requests[0].url,'/api/dust-run');
  assert.equal(requests[0].body.kind,'mask');
  assert.equal(requests[0].body.values.mask_version,'aaaaaaaabbbbccccddddeeeeeeeeeeee');
  assert.equal(requests[0].body.source,'stabilized.mkv');assert.equal(requests[0].body.project,'A');
  assert.equal(JSON.stringify(state.settings),original);
  assert.equal(vm.runInContext('dustDraft.mask_version',context),undefined,'rejected request changed inspected version');
  state.running=true;await context.dustRun(true);assert.equal(requests.length,1);
});

test('saved versions have distinguishable labels without changing detector names',()=>{
  const {context}=setup();
  assert.match(context.dustLabel({detector:'dart',repair:'dart',mask_version:'a'.repeat(32)}),/mask aaaaaa/);
});

test('Dust presets are scoped to the selected detector and retain custom tuning',()=>{
  const {context}=setup();
  vm.runInContext("dustDraft={detector:'dis_flow',preset:'fast_preview'}",context);
  const html=context.dustFieldHtml(['preset','Preset','select:balanced|fast_preview|dust_focused|dart_optimal_f1|custom']);
  assert.match(html,/value="fast_preview"/);assert.match(html,/value="dust_focused"/);assert.match(html,/value="custom"/);
  assert.doesNotMatch(html,/value="balanced"|value="dart_optimal_f1"/);
});

test('unsupported channels never claim derived provenance',()=>{
  const {context}=setup();
  assert.equal(context.channelOrigin('dust','ridge'),'No native channel');
  assert.equal(context.channelOrigin('scratch','ridge'),'Native channel');
  assert.equal(context.channelOrigin('mask','ridge'),'Combined mask');
});

test('individual channels cannot silently paint a combined playback mask',()=>{
  const {context}=setup();const canvas={hidden:false},status={textContent:''};
  context.document.querySelector=selector=>selector==='[data-dust-frame-status]'?status:null;
  context.dustMaskCanvas=()=>canvas;
  vm.runInContext("dustInputMode='scratch';dustStatus={frames:12}",context);
  context.dustPaintPlayback({}, {regions:{mask:[0,0,1,1]}},0);
  assert.equal(canvas.hidden,true);assert.match(status.textContent,/Exact still/);
});

test('accepted recalculation inspects the new version without changing pipeline settings',async()=>{
  const {context,state}=setup(),before=JSON.stringify(state.settings),requests=[];
  vm.runInContext("dustDraft={detector:'legacy',repair:'telea'};dustStatus={frames:12,repair_complete:true};dustRepairOpen=true",context);
  context.confirm=()=>true;context.crypto={randomUUID:()=> 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'};
  context.postJson=async(url,body)=>{requests.push({url,body});return {ok:true}};
  context.refresh=async()=>{};context.drawDust=()=>{};
  await context.dustRun(true);
  assert.equal(requests.length,1);assert.equal(requests[0].body.kind,'mask');
  assert.equal(vm.runInContext('dustDraft.mask_version',context),requests[0].body.values.mask_version);
  assert.equal(vm.runInContext('dustRepairOpen',context),false);
  assert.equal(JSON.stringify(state.settings),before);
  state.queue_running=true;await context.dustRun(true);assert.equal(requests.length,1);
});

test('defect detector select options render human-readable labels and map historical temporal',()=>{
  const context=vm.createContext({esc:String,tooltipTitle:()=>'',fieldDescription:()=>'',fieldTooltip:()=>''});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/render-helpers.js'),'utf8'),context);
  assert.equal(context.selectOptionLabel('defect_detector','farneback_3f'),'Farneback 3F (optical flow consensus)');
  assert.equal(context.selectOptionLabel('defect_detector','dis_flow'),'DIS Flow (fast optical flow)');
  assert.equal(context.selectOptionLabel('defect_detector','ridge'),'Temporal Ridge (vertical scratches)');
  assert.equal(context.selectOptionLabel('defect_detector','consensus_5f'),'Consensus 5F (5-frame median)');
  assert.equal(context.selectOptionLabel('defect_detector','legacy'),'Legacy (single-frame spatial)');

  // selectFieldHtml maps 'temporal' to farneback_3f option selected
  const html=context.selectFieldHtml('defect_detector','Defect detector','select:farneback_3f|dis_flow|ridge|consensus_5f|legacy','temporal');
  assert.match(html,/<option value="farneback_3f" selected>Farneback 3F/);
});

test('regression: dustFieldHtml renders preset selector without ReferenceError on DUST_PRESET_MAP', () => {
  const {context, state} = setup();
  vm.runInContext("dustDraft = {detector:'farneback_3f', preset:'balanced', repair:'telea', sensitivity:0.65, threshold:160, dilate:1};", context);

  // Invoke dustFieldHtml for preset field
  const presetField = [
    'preset',
    'Detector preset',
    'select:balanced|fast_preview|high_quality_precision|scratch_focused|dust_focused|high_recall|conservative_low_fp|dart_optimal_f1|dart_high_recall|dart_conservative|bopb_scratch_balanced|bopb_scratch_high_recall|film_artefact_balanced|cnn_blotches_precision|patchcore_anomaly_research|padim_anomaly_research|custom',
    'balanced'
  ];
  let rendered = '';
  assert.doesNotThrow(() => {
    rendered = context.dustFieldHtml(presetField);
  }, /DUST_PRESET_MAP is not defined/);

  assert.match(rendered, /<select data-dust="preset">/);
  assert.match(rendered, /value="balanced"/);
  for(const [detector,preset] of [['bopb','bopb_scratch_balanced'],['film_artefact_segmenter','film_artefact_balanced'],['cnn_blotches','cnn_blotches_precision'],['dart','dart_optimal_f1']]) {
    vm.runInContext(`dustDraft.detector=${JSON.stringify(detector)}`,context);
    assert.match(context.dustFieldHtml(presetField),new RegExp(`value="${preset}"`));
  }
});

test('switching between DART, BOPB, Film Artefact Segmenter and CNN Blotches correctly resolves presets and parameters', () => {
  const {context, state} = setup();
  // Simulate DOM element container with querySelectorAll
  const inputs = [];
  const handlers = {};
  const mockHost = {
    querySelectorAll: (sel) => {
      if (sel === '[data-dust]') return inputs;
      return [];
    },
    querySelector: () => null,
    addEventListener: () => {}
  };

  // Set initial draft to DART
  vm.runInContext("dustDraft = {detector:'dart', preset:'dart_optimal_f1', sensitivity:0.65, dilate:0, threshold:224, repair:'telea'};", context);
  const DUST_PRESET_MAP = context.DUST_PRESET_MAP;
  const DETECTOR_DEFAULT_PRESETS = context.DETECTOR_DEFAULT_PRESETS;

  assert.ok(DUST_PRESET_MAP.dart_optimal_f1);
  assert.equal(DUST_PRESET_MAP.dart_optimal_f1.threshold, 224);
  assert.ok(DUST_PRESET_MAP.bopb_scratch_balanced);
  assert.equal(DUST_PRESET_MAP.bopb_scratch_balanced.threshold, 128);
  assert.equal(DUST_PRESET_MAP.bopb_scratch_balanced.sensitivity, 0.5);

  // Test simulation of changing detector to BOPB
  function simulateDetectorChange(newDet) {
    const draft = vm.runInContext('dustDraft', context);
    const curP = DUST_PRESET_MAP[draft.preset];
    const curDet = curP?.detector === 'temporal' ? 'farneback_3f' : curP?.detector;
    const normDet = newDet === 'temporal' ? 'farneback_3f' : newDet;
    draft.detector = newDet;
    if (!curP || curDet !== normDet) {
      const defPresetName = DETECTOR_DEFAULT_PRESETS[newDet] || DETECTOR_DEFAULT_PRESETS[normDet];
      if (defPresetName && DUST_PRESET_MAP[defPresetName]) {
        const pCfg = DUST_PRESET_MAP[defPresetName];
        draft.detector = pCfg.detector;
        draft.sensitivity = pCfg.sensitivity;
        draft.dilate = pCfg.dilate;
        draft.threshold = pCfg.threshold;
        if (pCfg.repair) draft.repair = pCfg.repair;
        draft.preset = defPresetName;
      }
    }
  }

  // Switch to BOPB
  simulateDetectorChange('bopb');
  let draft = vm.runInContext('dustDraft', context);
  assert.equal(draft.detector, 'bopb');
  assert.equal(draft.preset, 'bopb_scratch_balanced');
  assert.equal(draft.threshold, 128);
  assert.equal(draft.sensitivity, 0.5);

  // Switch to Film Artefact Segmenter
  simulateDetectorChange('film_artefact_segmenter');
  draft = vm.runInContext('dustDraft', context);
  assert.equal(draft.detector, 'film_artefact_segmenter');
  assert.equal(draft.preset, 'film_artefact_balanced');
  assert.equal(draft.threshold, 128);
  assert.equal(draft.sensitivity, 0.5);

  // Switch to CNN Blotches
  simulateDetectorChange('cnn_blotches');
  draft = vm.runInContext('dustDraft', context);
  assert.equal(draft.detector, 'cnn_blotches');
  assert.equal(draft.preset, 'cnn_blotches_precision');
  assert.equal(draft.threshold, 204);
  assert.equal(draft.sensitivity, 0.8);

  // Switch to DART
  simulateDetectorChange('dart');
  draft = vm.runInContext('dustDraft', context);
  assert.equal(draft.detector, 'dart');
  assert.equal(draft.preset, 'dart_optimal_f1');
  assert.equal(draft.threshold, 224);
  assert.equal(draft.sensitivity, 0.65);
});

test('DUST_PRESET_MAP dynamically reflects live state.dust_matrix.presets from canonical backend', () => {
  const {context, state} = setup();
  // Backend populates state.dust_matrix.presets
  state.dust_matrix.presets = {
    custom_calibrated: {
      detector: 'film_artefact_segmenter',
      sensitivity: 0.55,
      dilate: 1,
      threshold: 135,
      repair: 'telea',
      commercial_cleared: true
    }
  };

  assert.ok(context.DUST_PRESET_MAP.custom_calibrated, 'Proxy should resolve live preset from state');
  assert.equal(context.DUST_PRESET_MAP.custom_calibrated.threshold, 135);
  // Canonical fallback also accessible
  assert.ok(context.DUST_PRESET_MAP.bopb_scratch_balanced, 'Canonical fallback preset should remain accessible');
});

test('dustPickerHtml renders full Dust & Scratches inspector controls without error', () => {
  const {context, state} = setup();
  state.stages[0].fields = [
    ["preset", "Detector preset", "select:balanced|fast_preview|high_quality_precision|scratch_focused|dust_focused|high_recall|conservative_low_fp|dart_optimal_f1|dart_high_recall|dart_conservative|bopb_scratch_balanced|bopb_scratch_high_recall|film_artefact_balanced|cnn_blotches_precision|patchcore_anomaly_research|padim_anomaly_research|custom", "balanced"],
    ["detector", "Defect detector", "select:farneback_3f|dis_flow|ridge|consensus_5f|legacy|dart|bopb|film_artefact_segmenter|cnn_blotches|patchcore|padim|correl_net", "farneback_3f"],
    ["repair", "Repair method", "select:telea|propainter|dart", "telea"],
    ["sensitivity", "Detection sensitivity", "range:0|1|0.05", "0.65"],
    ["radius", "Telea radius (px)", "number", "3"],
    ["dilate", "Repair mask expansion (px)", "number", "0"],
    ["threshold", "DART soft-mask threshold", "number", "160"],
    ["research_confirmed", "I acknowledge experimental detection and unvalidated real-damage accuracy", "checkbox", "false"],
    ["license_confirmed", "I acknowledge non-commercial ProPainter / unresolved DART RTN rights", "checkbox", "false"],
    ["dart_repo", "External pinned DART checkout override (optional)", "text", ""],
    ["dart_python", "External DART Python override (optional)", "text", ""]
  ];

  vm.runInContext("dustDraft = {detector:'farneback_3f', preset:'balanced', repair:'telea', sensitivity:0.65, threshold:160, dilate:0, radius:3};", context);

  let html = '';
  assert.doesNotThrow(() => {
    html = context.dustPickerHtml();
  });

  assert.ok(html.includes('data-dust="preset"'));
  assert.ok(html.includes('data-dust="detector"'));
  assert.ok(html.includes('data-dust="sensitivity"'));
  assert.ok(html.includes('data-dust="dilate"'));
  assert.ok(html.indexOf('data-dust="detector"') < html.indexOf('data-dust="preset"'));
  assert.ok(html.indexOf('data-dust="preset"') < html.indexOf('Saved masks & repairs'));
  assert.match(html, /<details class="dust-fine-tuning">/);

  // Switch to DART and verify threshold is rendered
  vm.runInContext("dustDraft.detector = 'dart'; dustDraft.preset = 'dart_optimal_f1'; dustDraft.threshold = 224; dustDraft.sensitivity = 0.65;", context);
  assert.doesNotThrow(() => {
    html = context.dustPickerHtml();
  });
  assert.ok(html.includes('data-dust="threshold"'));
  assert.ok(html.includes('DART'));
});

function setupWithEditor() {
  const rows = [
    {
      method: 'candidate0',
      preset: 'settings',
      label: 'farneback_3f / telea',
      repair_complete: true,
      output: 'result0.mkv',
      revision: 'report0',
      values: { detector: 'farneback_3f', preset: 'balanced', repair: 'telea' },
      mask_sha256: 'same-mask',
    },
  ];
  const state = {
    project_path: 'P1',
    settings: {
      global: { source: 'raw.mkv', commercial_model_guard: 'false' },
      dust: { detector: 'farneback_3f', preset: 'balanced', repair: 'telea' },
      cloud: {},
    },
    source_info: { duration: '1', fps: '4' },
    dust_matrix: { source: 'stabilized.mkv', source_revision: 'v1', variants: rows },
    stages: [
      {
        key: 'dust',
        files: [],
        fields: [
          ['preset', 'Detector preset', 'select:balanced|fast_preview|experimental_e2_research|custom', 'balanced'],
          ['detector', 'Defect detector', 'select:farneback_3f|dis_flow|ridge|consensus_5f|legacy|dart|bopb|film_artefact_segmenter|cnn_blotches|patchcore|padim|temporal_scratch|correl_net|experimental_e2', 'farneback_3f'],
          ['enable_temporal_scratch', 'Temporal scratch enhancement (motion-compensated)', 'checkbox', 'false'],
          ['repair', 'Repair method', 'select:telea|propainter|dart', 'telea'],
          ['sensitivity', 'Detection sensitivity', 'range:0|1|0.05', '0.65'],
          ['radius', 'Telea radius (px)', 'number', '3'],
          ['dilate', 'Repair mask expansion (px)', 'number', '1'],
          ['threshold', 'DART soft-mask threshold', 'number', '160'],
          ['research_confirmed', 'I acknowledge experimental detection and unvalidated real-damage accuracy', 'checkbox', 'false'],
          ['license_confirmed', 'I acknowledge non-commercial ProPainter / unresolved DART RTN rights', 'checkbox', 'false'],
        ],
      },
    ],
  };

  function makeFakeCanvas(w = 0, h = 0) {
    const c = {
      width: w,
      height: h,
      isConnected: true,
      style: {},
      dataset: {},
      getContext() {
        return {
          imageSmoothingEnabled: false,
          createImageData(iw, ih) {
            return { width: iw, height: ih, data: new Uint8ClampedArray(iw * ih * 4) };
          },
          putImageData(imgData) {
            c._lastImageData = imgData;
          },
          clearRect() {},
          fillRect() {},
          drawImage() {},
          save() {},
          restore() {},
          beginPath() {},
          rect() {},
          clip() {},
        };
      },
    };
    return c;
  }

  const maskViewEl = {
    isConnected: true,
    innerHTML: '',
    hidden: false,
    _imagesEl: {
      isConnected: true,
      children: [],
      replaceChildren() { this.children = []; },
      set innerHTML(_v) {},
      querySelector() { return null; },
    },
    _retryEl: { hidden: true },
    querySelector(sel) {
      if (sel === '[data-dust-images]') return this._imagesEl;
      if (sel === '[data-dust-retry]') return this._retryEl;
      return null;
    },
  };
  const frameStatusEl = { textContent: '' };

  const listeners = new Map();
  const doc = {
    querySelector(sel) {
      if (sel === '[data-dust-mask-view]') return maskViewEl;
      if (sel === '[data-dust-frame-status]') return frameStatusEl;
      return null;
    },
    querySelectorAll() { return []; },
    getElementById() { return null; },
    createElement(tag) {
      if (tag === 'canvas') return makeFakeCanvas();
      return { style: {}, dataset: {}, classList: { toggle() {}, add() {}, remove() {} }, querySelector: () => null, querySelectorAll: () => [] };
    },
    addEventListener(name, fn) {
      if (!listeners.has(name)) listeners.set(name, []);
      listeners.get(name).push(fn);
    },
  };

  const context = vm.createContext({
    state,
    active: 'dust',
    parseDuration: Number,
    esc: String,
    location: { pathname: '/' },
    document: doc,
    window: { addEventListener() {}, removeEventListener() {} },
    Buffer,
    Uint8Array,
    Uint8ClampedArray,
    ArrayBuffer,
    URLSearchParams,
    setTimeout,
    clearTimeout,
    Math,
    Number,
    String,
    Object,
    Array,
    JSON,
    Map,
    Set,
    Error,
    Promise,
    api: async () => ({ ok: true }),
    postJson: async () => ({ ok: true }),
    refresh: async () => {},
  });

  for (const file of ['review-zoom.js', 'render-denoise.js', 'denoise-workspace.js', 'comparison-layout.js', 'dust-mask-editor.js', 'render-dust.js']) {
    vm.runInContext(fs.readFileSync(path.join(__dirname, '../ai_remaster_gui/static/js', file), 'utf8'), context);
  }
  context.denoiseWorkspaceInitialize(context.denoiseCompareCandidates(state.stages[0]));
  return { context, state, makeFakeCanvas, maskViewEl };
}

test('dust mask editor Uint8Array <-> base64 round-trips all bit combinations 0..127 losslessly', () => {
  const { context } = setupWithEditor();
  const w = 16, h = 8;
  const src = new Uint8Array(w * h);
  for (let i = 0; i < src.length; i += 1) {
    src[i] = i & context.KNOWN_EDIT_BITS_MASK;
  }
  const b64 = context.dustBitmaskToBase64(src);
  const decoded = context.dustBase64ToBitmask(b64, w, h);
  assert.equal(context.dustMaskEditorIsUint8(decoded), true);
  assert.deepEqual(Array.from(decoded), Array.from(src));

  // Empty string decodes to zero-filled Uint8Array of exact frame size
  const zeros = context.dustBase64ToBitmask('', w, h);
  assert.equal(zeros.length, w * h);
  assert.equal(zeros.every(v => v === 0), true);

  // Length mismatch rejects cleanly
  assert.throws(() => context.dustBase64ToBitmask(b64, w + 1, h), /does not match/);
});

test('dustApplyStrokeTransition, paintBrushSegment, zoomed coordinates, and overlay precedence', () => {
  const { context, makeFakeCanvas } = setupWithEditor();
  const w = 10, h = 10;
  let mask = new Uint8Array(w * h);

  // Conflict reversal & provenance preservation
  mask = context.dustApplyStrokeTransition(mask, [11], 'mark_reviewed');
  mask = context.dustApplyStrokeTransition(mask, [11], 'scratch_reject');
  assert.equal(mask[11], context.BIT_REVIEWED | context.BIT_SCRATCH_REJECT);

  mask = context.dustApplyStrokeTransition(mask, [11], 'scratch_add');
  assert.equal(mask[11], context.BIT_REVIEWED | context.BIT_SCRATCH_ADD);

  mask = context.dustApplyStrokeTransition(mask, [11], 'dust_add');
  mask = context.dustApplyStrokeTransition(mask, [11], 'protect');
  assert.equal(
    mask[11],
    context.BIT_REVIEWED | context.BIT_SCRATCH_ADD | context.BIT_DUST_ADD | context.BIT_PROTECT_INTACT
  );

  mask = context.dustApplyStrokeTransition(mask, [11], 'unprotect');
  assert.equal(mask[11], context.BIT_REVIEWED | context.BIT_SCRATCH_ADD | context.BIT_DUST_ADD);

  mask = context.dustApplyStrokeTransition(mask, [11], 'reject_all');
  assert.equal(
    mask[11],
    context.BIT_REVIEWED | context.BIT_SCRATCH_REJECT | context.BIT_DUST_REJECT | context.BIT_UNCERTAIN_REJECT
  );

  mask = context.dustApplyStrokeTransition(mask, [11], 'clear_edits');
  assert.equal(mask[11], 0);

  // paintBrushSegment Bresenham + radius > 0
  const opts = { inPlace: true };
  context.paintBrushSegment(mask, w, h, 1, 1, 4, 1, 0, 'scratch_add', opts);
  for (let x = 1; x <= 4; x += 1) {
    assert.equal(mask[1 * w + x] & context.BIT_SCRATCH_ADD, context.BIT_SCRATCH_ADD);
  }
  assert.deepEqual(JSON.parse(JSON.stringify(opts.lastDirtyRect)), { minX: 1, minY: 1, maxX: 4, maxY: 1 });

  context.paintBrushSegment(mask, w, h, 6, 6, 6, 6, 1, 'protect', opts);
  assert.equal(mask[6 * w + 6] & context.BIT_PROTECT_INTACT, context.BIT_PROTECT_INTACT);
  assert.equal(mask[6 * w + 7] & context.BIT_PROTECT_INTACT, context.BIT_PROTECT_INTACT);

  // Zoomed + panned canvas coordinate mapping
  const zoomedCanvas = {
    width: 100,
    height: 80,
    getBoundingClientRect: () => ({ left: 50, top: 20, width: 200, height: 160 }),
  };
  assert.deepEqual(JSON.parse(JSON.stringify(context.dustCanvasClientToPixel(zoomedCanvas, 50, 20))), { x: 0, y: 0 });
  assert.deepEqual(JSON.parse(JSON.stringify(context.dustCanvasClientToPixel(zoomedCanvas, 150, 100))), { x: 50, y: 40 });
  assert.deepEqual(JSON.parse(JSON.stringify(context.dustCanvasClientToPixel(zoomedCanvas, 999, 999))), { x: 99, y: 79 });

  // Overlay render precedence: PROTECT_INTACT overrides scratch_add & autoScratch; autoVetoed renders separately
  const renderCanvas = makeFakeCanvas(w, h);
  const autoScratch = new Uint8Array(w * h);
  const autoVetoed = new Uint8Array(w * h);
  autoScratch[1 * w + 1] = 255;
  autoVetoed[8 * w + 8] = 255;
  mask[1 * w + 1] = context.BIT_SCRATCH_ADD | context.BIT_PROTECT_INTACT;
  const imgData = context.renderDustMaskOverlayCanvas(renderCanvas, {
    width: w,
    height: h,
    overlayOnly: true,
    bitmask: mask,
    autoScratch,
    autoVetoed,
  });
  const protPx = (1 * w + 1) * 4;
  assert.ok(imgData.data[protPx + 1] > imgData.data[protPx], 'PROTECT_INTACT green veto must dominate red scratch_add');
  const vetoPx = (8 * w + 8) * 4;
  assert.equal(imgData.data[vetoPx], 56);
  assert.equal(imgData.data[vetoPx + 1], 178);
  assert.equal(imgData.data[vetoPx + 2], 172);
});

test('undoDustMaskEdit and redoDustMaskEdit respect undo cap and clear redo on new stroke', async () => {
  const { context, makeFakeCanvas } = setupWithEditor();
  const postCalls = [];
  context.api = async () => ({
    ok: true,
    frame: 0,
    frame_count: 3,
    width: 8,
    height: 8,
    analysis_source: 'stage',
    timeline_id: 't'.repeat(24),
    edit_revision: 0,
    zip_sha256: '',
    edit_signature: '',
    compatible: true,
    compatibility_reason: '',
    edited_frames: [],
    edit_mask_b64: '',
    reviewed: false,
  });
  context.postJson = async (url, body) => {
    postCalls.push({ url, body });
    return {
      ok: true,
      frame: body.frame,
      timeline_id: body.timeline_id,
      edit_revision: 1,
      zip_sha256: 'a'.repeat(64),
      edit_signature: 'a'.repeat(64) + ':rev1',
      edited_frames: [0],
      reviewed: false,
      frame_stats: { reviewed_pixels: 0 },
    };
  };

  await context.dustMaskEditorLoadFrame(0, 'stage');
  const canvas = makeFakeCanvas(8, 8);
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 8, height: 8 });
  context.dustMaskEditorAttachFrameInspection(canvas, { frame: 0, width: 8, height: 8 });
  context.dustMaskEditorSetEditMode(true);
  context.dustMaskEditorSetBrushRadius(0);
  context.dustMaskEditorSetTool('scratch_add');

  const maxUndo = context.DUST_EDITOR_MAX_UNDO;
  for (let i = 0; i < maxUndo + 4; i += 1) {
    const px = i % 8;
    const py = Math.floor(i / 8) % 8;
    context.dustMaskEditorHandlePointerDown(canvas, {
      pointerType: 'mouse',
      button: 0,
      clientX: px + 0.1,
      clientY: py + 0.1,
      stopPropagation() {},
      preventDefault() {},
    });
    vm.runInContext('_finishStroke(null)', context);
  }
  assert.equal(context.dustMaskEditor.undoStack.length, maxUndo);

  const beforeUndoPixel = context.dustMaskEditor.bitmask[(Math.floor((maxUndo + 3) / 8) % 8) * 8 + ((maxUndo + 3) % 8)];
  assert.equal(beforeUndoPixel, context.BIT_SCRATCH_ADD);

  assert.equal(context.undoDustMaskEdit(), true);
  assert.equal(context.dustMaskEditor.redoStack.length, 1);
  assert.equal(
    context.dustMaskEditor.bitmask[(Math.floor((maxUndo + 3) / 8) % 8) * 8 + ((maxUndo + 3) % 8)],
    0
  );

  assert.equal(context.redoDustMaskEdit(), true);
  assert.equal(context.dustMaskEditor.redoStack.length, 0);
  assert.equal(
    context.dustMaskEditor.bitmask[(Math.floor((maxUndo + 3) / 8) % 8) * 8 + ((maxUndo + 3) % 8)],
    context.BIT_SCRATCH_ADD
  );

  // Undo once, then start a new stroke -> redo stack must clear
  assert.equal(context.undoDustMaskEdit(), true);
  assert.equal(context.dustMaskEditor.redoStack.length, 1);
  context.dustMaskEditorHandlePointerDown(canvas, {
    pointerType: 'mouse',
    button: 0,
    clientX: 7.1,
    clientY: 7.1,
    stopPropagation() {},
    preventDefault() {},
  });
  vm.runInContext('_finishStroke(null)', context);
  assert.equal(context.dustMaskEditor.redoStack.length, 0);

  await context.saveCurrentDustMaskEdit();
  assert.equal(postCalls.length, 1);
  assert.equal(postCalls[0].url, '/api/dust-mask-frame');
  assert.equal(postCalls[0].body.expected_revision, 0);
  assert.equal(context.dustMaskEditor.editRevision, 1);
  assert.equal(context.dustMaskEditor.editSignature, 'a'.repeat(64) + ':rev1');
});

test('DUST_DETECTOR_META and experimental_e2 opt-in gating keep farneback_3f balanced default', () => {
  const { context } = setupWithEditor();
  const meta = context.DUST_DETECTOR_META;
  assert.equal(context.DETECTOR_DEFAULT_PRESETS.farneback_3f, 'balanced');
  assert.equal(context.CANONICAL_DUST_PRESETS.balanced.detector, 'farneback_3f');
  assert.equal(meta.experimental_e2.experimental_opt_in_required, true);
  assert.equal(meta.experimental_e2.research_confirmed_required, true);
  assert.deepEqual(Array.from(meta.experimental_e2.unsupported_channels), ['dust']);
  assert.deepEqual(Array.from(meta.experimental_e2.three_state_outputs), [
    'high_confidence_damage',
    'uncertain_damage_candidate',
    'likely_intact_scene_structure',
  ]);

  // Blocked without research_confirmed === 'true'
  const blocked = context.dustActionState(null, false, {
    detector: 'experimental_e2',
    preset: 'experimental_e2_research',
    research_confirmed: 'false',
  });
  assert.equal(blocked.kind, '');
  assert.equal(blocked.label, 'Experimental Opt-In Required');

  // Allowed when research_confirmed === 'true'
  const allowed = context.dustActionState(null, false, {
    detector: 'experimental_e2',
    preset: 'experimental_e2_research',
    research_confirmed: 'true',
  });
  assert.equal(allowed.kind, 'mask');
  assert.equal(allowed.label, 'Calculate mask');

  // Picker defaults to farneback_3f / balanced and shows Experimental Prototypes optgroup
  vm.runInContext("dustDraft = { detector: 'farneback_3f', preset: 'balanced', repair: 'telea', sensitivity: 0.65, dilate: 1, research_confirmed: 'false' };", context);
  const pickerDefault = context.dustPickerHtml();
  assert.match(pickerDefault, /<optgroup label="Experimental Prototypes">/);
  assert.match(pickerDefault, /value="experimental_e2"/);
  assert.match(pickerDefault, /<option value="farneback_3f" selected/);
  assert.doesNotMatch(pickerDefault, /data-dust="research_confirmed"/);

  // Selecting experimental_e2 reveals research_confirmed checkbox and notice
  vm.runInContext("dustDraft = { detector: 'experimental_e2', preset: 'experimental_e2_research', repair: 'telea', sensitivity: 0.65, dilate: 0, research_confirmed: 'false' };", context);
  const pickerE2 = context.dustPickerHtml();
  assert.match(pickerE2, /data-dust="research_confirmed"/);
  assert.match(pickerE2, /vetoed_mask\.mkv/);
});

test('dustCatalog and dustInspectionKey stay stable across edit revisions while dustActiveEditSignature updates', () => {
  const { context, state } = setupWithEditor();
  const inspectCalls = [];
  context.postJson = async (url, body) => {
    inspectCalls.push({ url, body });
    return { ok: true, frames: 4, urls: {}, vetoed_url: '', base_urls: {}, auto_proposal_urls: {} };
  };

  vm.runInContext("dustDraft = { detector: 'farneback_3f', preset: 'balanced', repair: 'telea' }; dustStatus = { frames: 4 }; dustView = 'mask'; dustInputMode = 'overlay';", context);
  context.denoiseWorkspaceFor('dust').setExact(true);
  context.dustInvalidateInspection();

  const initialCatalog = context.dustCatalog();
  const initialInspectionKey = vm.runInContext('dustInspectionKey', context);
  assert.equal(context.dustActiveEditSignature(), '');
  assert.equal(inspectCalls.length, 1);

  // Revision 1 arrives via state sync
  const sig1 = 'b'.repeat(64) + ':rev1';
  context.dustMaskEditorSyncFromState({
    dust_edits: {
      timeline_id: 't'.repeat(24),
      edit_revision: 1,
      zip_sha256: 'b'.repeat(64),
      edit_signature: sig1,
      edited_frames: [0],
      compatible: true,
    },
  });
  assert.equal(context.dustActiveEditSignature(), sig1);
  assert.equal(context.dustCatalog(), initialCatalog);
  assert.equal(vm.runInContext('dustInspectionKey', context), initialInspectionKey);
  context.dustInvalidateInspection();
  assert.equal(inspectCalls.length, 1, 'Edit revision update must not force duplicate /api/dust-inspect');

  // Revision 2 arrives and marks previous repair stale for repair action without mutating catalog/inspection key
  const sig2 = 'c'.repeat(64) + ':rev2';
  context.dustMaskEditorSyncFromState({
    dust_edits: {
      timeline_id: 't'.repeat(24),
      edit_revision: 2,
      zip_sha256: 'c'.repeat(64),
      edit_signature: sig2,
      edited_frames: [0, 1],
      compatible: true,
    },
  });
  assert.equal(context.dustActiveEditSignature(), sig2);
  assert.equal(context.dustCatalog(), initialCatalog);
  assert.equal(vm.runInContext('dustInspectionKey', context), initialInspectionKey);

  const action = context.dustActionState(
    { frames: 4, repair_complete: true, edit_signature: sig1 },
    false,
    vm.runInContext('dustDraft', context),
    true
  );
  assert.equal(action.kind, 'repair');
  assert.equal(action.label, 'Repair masked areas');
});

test('dustAvailableFrames and dustActionState recognize base_mask_ready when composed frames is 0', () => {
  const { context } = setupWithEditor();
  assert.equal(context.dustAvailableFrames({ frames: 0, base_mask_ready: true, base_frames: 4 }), 4);
  assert.equal(context.dustAvailableFrames({ frames: 7, base_mask_ready: true, base_frames: 4 }), 7);
  assert.equal(context.dustAvailableFrames({ frames: 0, base_mask_ready: false, base_frames: 4 }), 0);
  assert.equal(context.dustAvailableFrames(null), 0);

  const action = context.dustActionState(
    { frames: 0, base_mask_ready: true, base_frames: 4 },
    false,
    { detector: 'farneback_3f', preset: 'balanced' },
    true
  );
  assert.equal(action.kind, 'repair');
  assert.equal(action.label, 'Repair masked areas');
});

test('dustMatches refuses unedited base variants while matching variants with active edit_signature and timeline_id', () => {
  const { context, state } = setupWithEditor();
  const sig = 'd'.repeat(64) + ':rev1';
  const tid = 'e'.repeat(24);
  context.dustMaskEditorResetLocalCache();
  vm.runInContext("dustMaskEditor.editSignature = ''; dustMaskEditor.timelineId = ''; dustMaskEditor.analysisSource = 'stage';", context);

  state.dust_edits = {
    stage: { timeline_id: tid, edit_signature: sig, edit_revision: 1, has_edits: true },
    original: { timeline_id: '', edit_signature: '', edit_revision: 0, has_edits: false },
  };
  const draft = { detector: 'farneback_3f', preset: 'balanced', repair: 'telea' };
  const baseVariant = { detector: 'farneback_3f', preset: 'balanced', repair: 'telea' };
  const composedVariant = {
    detector: 'farneback_3f',
    preset: 'balanced',
    repair: 'telea',
    edit_signature: sig,
    timeline_id: tid,
  };
  const wrongTidVariant = { ...composedVariant, timeline_id: 'f'.repeat(24) };

  assert.equal(context.dustMatches(baseVariant, draft), false);
  assert.equal(context.dustMatches(composedVariant, draft), true);
  assert.equal(context.dustMatches(wrongTidVariant, draft), false);
});

test('dustActiveEditSignature isolates original source mode metadata from stage dust_edits', () => {
  const { context, state } = setupWithEditor();
  const stageSig = '1'.repeat(64) + ':rev1';
  const stageTid = 'a'.repeat(24);
  const origSig = '2'.repeat(64) + ':rev1';
  const origTid = 'b'.repeat(24);

  vm.runInContext(
    `dustMaskEditor.analysisSource = 'stage'; dustMaskEditor.editSignature = ${JSON.stringify(stageSig)}; dustMaskEditor.timelineId = ${JSON.stringify(stageTid)};`,
    context
  );
  state.dust_edits = {
    edit_signature: stageSig,
    timeline_id: stageTid,
    stage: { edit_signature: stageSig, timeline_id: stageTid },
    original: { edit_signature: '', timeline_id: '' },
  };

  assert.equal(context.dustActiveEditSignature('stage'), stageSig);
  assert.equal(context.dustActiveEditSignature('original'), '');

  state.dust_edits.original = { edit_signature: origSig, timeline_id: origTid };
  state.dust_matrix.original = {
    source: 'raw.mkv',
    dust_edits: { edit_signature: origSig, timeline_id: origTid },
  };
  assert.equal(context.dustActiveEditSignature('original'), origSig);
  assert.equal(context.dustActiveEditSignature('stage'), stageSig);
});

test('dustWireCanvasInteraction stops propagation on A/B split drag, allows shift-pan, and supports edit mode', () => {
  const { context, makeFakeCanvas } = setupWithEditor();
  const canvas = makeFakeCanvas(100, 80);
  canvas.getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 80 });
  canvas.setPointerCapture = () => {};
  canvas.hasPointerCapture = () => false;
  canvas.releasePointerCapture = () => {};

  let redraws = 0;
  vm.runInContext('dustAbSplitActive = true; dustSplitPos = 50;', context);
  context.dustMaskEditorSetEditMode(false);
  context.dustWireCanvasInteraction(canvas, () => { redraws += 1; }, true);
  assert.equal(canvas.style.cursor, 'ew-resize');
  assert.equal(canvas.style.pointerEvents, 'auto');

  let stopped = 0;
  let prevented = 0;
  canvas.onpointerdown({
    type: 'pointerdown',
    pointerType: 'mouse',
    button: 0,
    pointerId: 1,
    clientX: 25,
    clientY: 20,
    shiftKey: false,
    stopPropagation() { stopped += 1; },
    preventDefault() { prevented += 1; },
  });
  assert.equal(stopped, 1);
  assert.equal(prevented, 1);
  assert.equal(vm.runInContext('dustSplitPos', context), 25);
  assert.equal(redraws, 1);

  // Holding shift yields to zoom pan without moving split or stopping propagation
  let shiftStopped = 0;
  let shiftPrevented = 0;
  canvas.onpointerdown({
    type: 'pointerdown',
    pointerType: 'mouse',
    button: 0,
    pointerId: 2,
    clientX: 75,
    clientY: 20,
    shiftKey: true,
    stopPropagation() { shiftStopped += 1; },
    preventDefault() { shiftPrevented += 1; },
  });
  assert.equal(shiftStopped, 0);
  assert.equal(shiftPrevented, 0);
  assert.equal(vm.runInContext('dustSplitPos', context), 25);

  // Turning Edit Mask ON switches cursor to crosshair and routes pointerdown to brush painting
  context.dustMaskEditorSetEditMode(true);
  vm.runInContext('dustMaskEditor.compatible = true; dustMaskEditor.loading = false;', context);
  context.dustMaskEditorAttachFrameInspection(canvas, { frame: 0, width: 100, height: 80 });
  context.dustWireCanvasInteraction(canvas, () => { redraws += 1; }, true);
  assert.equal(canvas.style.cursor, 'crosshair');

  let editStopped = 0;
  let editPrevented = 0;
  canvas.onpointerdown({
    type: 'pointerdown',
    pointerType: 'mouse',
    button: 0,
    pointerId: 3,
    clientX: 60,
    clientY: 40,
    shiftKey: false,
    stopPropagation() { editStopped += 1; },
    preventDefault() { editPrevented += 1; },
  });
  assert.equal(editStopped, 1);
  assert.equal(editPrevented, 1);
  assert.equal(vm.runInContext('dustSplitPos', context), 25);
  assert.equal(vm.runInContext('dustMaskEditor.bitmask[40 * 100 + 60]', context) & context.BIT_SCRATCH_ADD, context.BIT_SCRATCH_ADD);
});


