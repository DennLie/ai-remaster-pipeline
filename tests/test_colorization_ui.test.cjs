const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function loadJsHelpers() {
  const jsDir = path.join(__dirname, '..', 'ai_remaster_gui', 'static', 'js');
  const context = {
    console,
    state: {
      settings: { global: {}, references: {}, colour: {} },
      expected_outputs: {},
      shot_views: {},
      color_bible_context: { film_id: 'film_01', db_path: 'manifests/color_bible.db' },
    },
    settings(key) {
      return (context.state.settings && context.state.settings[key]) || {};
    },
    esc(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
      }[ch]));
    },
    formatNum(v) {
      return (v === undefined || v === null || Number.isNaN(Number(v))) ? '—' : Number(v).toFixed(2);
    },
    jsArg(v) {
      return JSON.stringify(v);
    },
    document: { addEventListener() {} },
    api: async () => ({ ok: true }),
    postJson: async () => ({ ok: true }),
    stateUrl: () => '/api/state',
  };
  vm.createContext(context);
  for (const file of ['render-helpers.js', 'render-shots.js', 'render-color-bible.js', 'settings.js']) {
    const src = fs.readFileSync(path.join(jsDir, file), 'utf8');
    vm.runInContext(src, context, { filename: file });
  }
  return context;
}

test('shotStageVisibleFields filters osi_masked_lab and references cleanly', () => {
  const ctx = loadJsHelpers();
  const colourStage = {
    key: 'colour',
    fields: [
      ['compute'], ['manifest'], ['method'], ['processing_height'],
      ['openai_image_model'], ['frame_propagate'], ['masks_dir'],
      ['color_bible_db'], ['film_id'], ['osi_flow_scale'],
      ['osi_anchor_strength'], ['guided_radius'], ['guided_eps'],
      ['osi_run_qc'], ['osi_workers'],
    ],
  };
  const visibleOsi = ctx.shotStageVisibleFields(colourStage, { method: 'osi_masked_lab' }).map(f => f[0]);
  assert.ok(visibleOsi.includes('masks_dir'));
  assert.ok(visibleOsi.includes('color_bible_db'));
  assert.ok(visibleOsi.includes('film_id'));
  assert.ok(visibleOsi.includes('osi_flow_scale'));
  assert.ok(!visibleOsi.includes('openai_image_model'));
  assert.ok(!visibleOsi.includes('frame_propagate'));

  const refStage = {
    key: 'references',
    fields: [
      ['compute'], ['method'], ['manifest'], ['prompt'],
      ['reference_runner'], ['ddcolor_model'], ['ddcolor_model_path'],
      ['color_bible_db'], ['film_id'], ['confidence_threshold'],
    ],
  };
  const visibleGuided = ctx.shotStageVisibleFields(refStage, { method: 'qwen', reference_runner: 'guided_only' }).map(f => f[0]);
  assert.ok(!visibleGuided.includes('ddcolor_model'));
  const visibleDd = ctx.shotStageVisibleFields(refStage, { method: 'ddcolor', reference_runner: 'guided_only' }).map(f => f[0]);
  assert.ok(visibleDd.includes('ddcolor_model'));
  assert.ok(!visibleDd.includes('prompt'));
});

test('colorizationLabel, selectOptionLabel, and labPercentileValue handle new modes and arrays', () => {
  const ctx = loadJsHelpers();
  assert.equal(ctx.colorizationLabel('osi_masked_lab'), 'OSI Masked LAB');
  assert.ok(ctx.selectOptionLabel('method', 'osi_masked_lab').includes('OSI Masked LAB'));
  assert.ok(ctx.selectOptionLabel('reference_runner', 'guided_only').includes('Guided only'));
  assert.equal(ctx.labPercentileValue([10, 25, 50, 75, 90], 2, 'p50'), 50);
  assert.equal(ctx.labPercentileValue({ p50: 42 }, 2, 'p50'), 42);
});

test('commercialPolicySettingsHtml includes auto_free_vram and core updateSystemStatus renders Free VRAM button', () => {
  const ctx = loadJsHelpers();
  const html = ctx.commercialPolicySettingsHtml();
  assert.ok(html.includes('id="setAutoFreeVram"'));
  assert.ok(html.includes('checked'));

  const jsDir = path.join(__dirname, '..', 'ai_remaster_gui', 'static', 'js');
  const sysStatusEl = { innerHTML: '' };
  const coreCtx = {
    console,
    esc: ctx.esc,
    window: { addEventListener() {} },
    document: {
      addEventListener() {},
      getElementById(id) { return id === 'systemStatus' ? sysStatusEl : null; },
      querySelector() { return null; },
      documentElement: { style: { setProperty() {} } },
    },
  };
  vm.createContext(coreCtx);
  vm.runInContext(fs.readFileSync(path.join(jsDir, 'core.js'), 'utf8'), coreCtx, { filename: 'core.js' });
  vm.runInContext(`state = ${JSON.stringify({
    running: false,
    system_status: {
      cuda: { available: true, capability: '8.9', detail: 'RTX 4090' },
      cpu: { name: 'CPU', percent: 12 },
      gpu: { name: 'GPU', percent: 0 },
      ram: { label: 'RAM 8/64 GB', percent: 12.5 },
      vram: { label: 'VRAM 12.0/24.0 GB', percent: 50, detail: '12.0 / 24.0 GB | ComfyUI models: 11.6 GB' },
    },
  })};`, coreCtx);
  coreCtx.updateSystemStatus();
  assert.ok(sysStatusEl.innerHTML.includes('id="freeVramBtn"'));
  assert.ok(sysStatusEl.innerHTML.includes('onclick="freeVramNow()"'));
  assert.ok(!sysStatusEl.innerHTML.includes('disabled'));
  assert.ok(sysStatusEl.innerHTML.includes('ComfyUI models: 11.6 GB'));

  vm.runInContext('state.running = true;', coreCtx);
  coreCtx.updateSystemStatus();
  assert.ok(sysStatusEl.innerHTML.includes('disabled'));
});

test('availableTabs places color_bible between shots and references', () => {
  const jsDir = path.join(__dirname, '..', 'ai_remaster_gui', 'static', 'js');
  const coreCtx = {
    console,
    esc: v => String(v ?? ''),
    window: { addEventListener() {} },
    document: {
      addEventListener() {},
      getElementById() { return null; },
      querySelector() { return null; },
      documentElement: { style: { setProperty() {} } },
    },
  };
  vm.createContext(coreCtx);
  vm.runInContext(fs.readFileSync(path.join(jsDir, 'core.js'), 'utf8'), coreCtx, { filename: 'core.js' });
  vm.runInContext(`state = ${JSON.stringify({
    stages: [
      { key: 'outpaint' },
      { key: 'shots' },
      { key: 'references' },
      { key: 'colour' },
      { key: 'recomp' },
      { key: 'output' },
    ],
  })};`, coreCtx);
  const tabs = Array.from(coreCtx.availableTabs());
  assert.deepEqual(tabs, ['global', 'outpaint', 'shots', 'color_bible', 'references', 'colour', 'recomp', 'cache', 'output', 'settings']);
});

test('reference hints, shotFieldOptions, osiMaskStatusHtml, and Color QC summary bar behave cleanly', () => {
  const ctx = loadJsHelpers();

  // Updated reference hint banner mentions OSI Masked LAB
  const cardHtml = ctx.referenceCard({
    mode: 'references',
    manifest: 'manifests/shots.csv',
    row: { index: 0, start_label: '00:00:00.000', end_label: '00:00:02.000', reference_items: [] },
    idx: 0,
    enabled: true,
    sourceUrl: '',
    colorUrl: '',
    sourceReady: false,
    colorReady: false,
  });
  assert.ok(cardHtml.includes('OSI Masked LAB'));

  // Contradictory reference controls are dimmed/disabled cleanly
  assert.equal(ctx.shotFieldOptions('references', ['reference_runner'], { method: 'ddcolor', reference_runner: 'guided_only' }).disabled, true);
  assert.equal(ctx.shotFieldOptions('references', ['reference_runner'], { method: 'qwen21', reference_runner: 'auto' }).disabled, true);
  assert.equal(ctx.shotFieldOptions('references', ['prompt'], { method: 'qwen', reference_runner: 'ddcolor_only' }).disabled, true);
  assert.ok(ctx.shotFieldOptions('colour', ['film_id'], {}).placeholder.includes('Auto: film_01'));
  assert.ok(ctx.shotFieldOptions('colour', ['masks_dir'], {}).placeholder.includes('intermediate/shot_masks'));

  // osiMaskStatusHtml renders SAM 2.1 controls for osi_masked_lab only and respects Commercial Model Guard
  ctx.state.shot_views = {
    colour: [
      { index: 0, shot_id: 'film_01_0', has_masks: true },
      { index: 1, shot_id: 'film_01_1', has_masks: false },
    ],
  };
  ctx.state.sam2_policy = { allowed: true, reason: '' };
  const osiHtml = ctx.osiMaskStatusHtml({ method: 'osi_masked_lab' });
  assert.ok(osiHtml.includes('Partial masks (1/2)'));
  assert.ok(osiHtml.includes('Track Shot Masks (SAM 2.1)'));
  assert.equal(ctx.osiMaskStatusHtml({ method: 'ltx' }), '');

  ctx.state.sam2_policy = { allowed: false, reason: 'Blocked for commercial test' };
  const blockedHtml = ctx.osiMaskStatusHtml({ method: 'osi_masked_lab' });
  assert.ok(blockedHtml.includes('Blocked for commercial test'));
  assert.ok(blockedHtml.includes('disabled'));

  // Color QC summary bar renders counts
  const qcHtml = ctx.renderColorBibleQcSummaryBar({
    total_events: 5,
    status_counts: { pass: 3, warn: 1, fail: 1 },
    failure_codes: { CDC_DRIFT: 1 },
  }, 'film_01');
  assert.ok(qcHtml.includes('Pass: 3'));
  assert.ok(qcHtml.includes('Warn: 1'));
  assert.ok(qcHtml.includes('Fail: 1'));
  assert.ok(qcHtml.includes('CDC_DRIFT (1)'));
});

test('segmenter dropdown visibility, text prompt toggling, and per-segmenter policy hints work cleanly', () => {
  const ctx = loadJsHelpers();
  const colourStage = {
    key: 'colour',
    fields: [
      ['compute'], ['manifest'], ['method'], ['processing_height'],
      ['masks_dir'], ['segmenter'], ['segmenter_text_prompts'],
      ['color_bible_db'], ['film_id'],
    ],
  };

  const visibleOsi = ctx.shotStageVisibleFields(colourStage, { method: 'osi_masked_lab' }).map(f => f[0]);
  assert.ok(visibleOsi.includes('segmenter'));
  assert.ok(visibleOsi.includes('segmenter_text_prompts'));

  // Non-text segmenter disables text prompts; text-capable segmenters enable them
  assert.equal(ctx.shotFieldOptions('colour', ['segmenter_text_prompts'], { segmenter: 'sam21' }).disabled, true);
  assert.equal(ctx.shotFieldOptions('colour', ['segmenter_text_prompts'], { segmenter: 'grounded_sam2' }).disabled, undefined);
  assert.ok(ctx.shotFieldOptions('colour', ['segmenter_text_prompts'], { segmenter: 'sam3' }).placeholder.includes('coat'));
  assert.ok(ctx.selectOptionLabel('segmenter', 'sam31').includes('Meta SAM License'));
  assert.ok(ctx.selectOptionLabel('segmenter', 'samurai').includes('SAMURAI'));

  // osiMaskStatusHtml respects segmenter_policies for sam3/sam31 vs grounded_sam2
  ctx.state.shot_views = { colour: [{ index: 0, shot_id: 'film_01_0', has_masks: false }] };
  ctx.state.segmenter_policies = {
    sam31: { allowed: false, reason: 'SAM 3.1 blocked under Meta SAM License' },
    grounded_sam2: { allowed: true, reason: '' },
  };
  const blockedSam31 = ctx.osiMaskStatusHtml({ method: 'osi_masked_lab', segmenter: 'sam31' });
  assert.ok(blockedSam31.includes('Track Shot Masks (SAM 3.1 (Benchmark Only))'));
  assert.ok(blockedSam31.includes('SAM 3.1 blocked under Meta SAM License'));
  assert.ok(blockedSam31.includes('disabled'));

  const allowedGrounded = ctx.osiMaskStatusHtml({ method: 'osi_masked_lab', segmenter: 'grounded_sam2' });
  assert.ok(allowedGrounded.includes('Track Shot Masks (Grounded-SAM-2)'));
  assert.ok(!allowedGrounded.includes('SAM 3.1 blocked'));
});

test('two-step Keyframe Image Segmenter vs Video Mask Tracker UI separation works cleanly', () => {
  const ctx = loadJsHelpers();

  // osiMaskStatusHtml cleanly separates Step 1 Keyframe Still from Step 2 Video Tracking
  ctx.state.shot_views = { colour: [{ index: 0, shot_id: 'film_01_0', has_masks: false }] };
  ctx.state.image_segmenter_policies = {
    evf_sam2: { allowed: true, reason: '' },
    mask2former: { allowed: true, reason: '' },
    sam31_image: { allowed: false, reason: 'SAM 3.1 Still-Image blocked by Meta SAM License' },
  };

  const textHtml = ctx.osiMaskStatusHtml({ method: 'osi_masked_lab', image_segmenter: 'evf_sam2', segmenter: 'sam21' });
  assert.ok(textHtml.includes('Step 1 · Keyframe Still'));
  assert.ok(textHtml.includes('Step 2 · Video Tracking'));
  assert.ok(textHtml.includes('Keyframe Image Segmenter (EVF-SAM-2)'));
  assert.ok(textHtml.includes('Text prompt (e.g. man in coat)'));
  assert.ok(textHtml.includes('Segment Reference Still'));
  assert.ok(textHtml.includes('Track Shot Masks (SAM 2.1)'));

  const panopticHtml = ctx.osiMaskStatusHtml({ method: 'osi_masked_lab', image_segmenter: 'mask2former', segmenter: 'samurai' });
  assert.ok(panopticHtml.includes('Scene classes (e.g. person, coat, sky)'));
  assert.ok(panopticHtml.includes('Track Shot Masks (SAMURAI)'));

  const ncImageHtml = ctx.osiMaskStatusHtml({ method: 'osi_masked_lab', image_segmenter: 'sam31_image', segmenter: 'sam21' });
  assert.ok(ncImageHtml.includes('Non-Commercial'));
  assert.ok(ncImageHtml.includes('SAM 3.1 Still-Image blocked by Meta SAM License'));

  // Color Bible Keyframe Image Segmenter box includes both still segmentation & handoff to Video Tracker
  const cbBoxHtml = ctx.renderCbKeyframeSegmenterBox();
  assert.ok(cbBoxHtml.includes('id="cbKeyframeSegSelect"'));
  assert.ok(cbBoxHtml.includes('Segment Reference Still'));
  assert.ok(cbBoxHtml.includes('Send to Video Tracker'));
  assert.ok(ctx.selectOptionLabel('image_segmenter', 'sam3_image').includes('Non-Commercial'));
});
