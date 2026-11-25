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
