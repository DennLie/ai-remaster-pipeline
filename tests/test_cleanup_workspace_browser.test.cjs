'use strict';
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const root = path.join(__dirname, '..');
const inspector = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/frame-inspector.js'), 'utf8');
const review = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/cleanup-review.js'), 'utf8');
const css = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/styles.css'), 'utf8');
const helpers = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/render-helpers.js'), 'utf8');

function browserFunctions() {
  const context = {document: {addEventListener() {}}, Uint8ClampedArray};
  vm.createContext(context);
  vm.runInContext(inspector, context, {filename: 'frame-inspector.js'});
  return context;
}

test('Clean Up provides permanent three-up exact frame inspection independently of stabilization', () => {
  const api = browserFunctions();
  const cleanup = api.frameInspectorHtml('cleanup');
  assert.match(cleanup, /data-fi-still-before/);
  assert.match(cleanup, /data-fi-still-after/);
  assert.match(cleanup, /data-fi-difference/);
  assert.match(cleanup, /PIXELDIFFERENZ · verstärkt/);
  assert.match(cleanup, /data-fi-diff-mode/);
  assert.match(cleanup, /data-fi-diff-contrast/);
  assert.doesNotMatch(cleanup, /Einzelbilder (anzeigen|ausblenden)/);
  assert.doesNotMatch(cleanup, /data-fi-open|data-fi-close/);
  const stabilize = api.frameInspectorHtml('stabilize');
  assert.doesNotMatch(stabilize, /data-fi-difference/);
  assert.match(stabilize, /Zum Video zurück/);
});

test('identical frames produce a dark difference and zero change', () => {
  const api = browserFunctions();
  const pixels = Uint8ClampedArray.from([70, 70, 70, 255, 0, 0, 0, 255]);
  const result = api.cleanupDifferencePixels(pixels, pixels, 2, 1);
  assert.deepEqual(Array.from(result.rgba), [0, 0, 0, 255, 0, 0, 0, 255]);
  assert.equal(result.mean, 0);
  assert.equal(result.changedPercent, 0);
});

test('brightened and darkened pixels have different diagnostic colors', () => {
  const api = browserFunctions();
  const original = Uint8ClampedArray.from([100, 100, 100, 255, 100, 100, 100, 255]);
  const processed = Uint8ClampedArray.from([110, 110, 110, 255, 90, 90, 90, 255]);
  const result = api.cleanupDifferencePixels(original, processed, 2, 1);
  assert.deepEqual(Array.from(result.rgba), [0, 80, 80, 255, 80, 40, 0, 255]);
  assert.ok(Math.abs(result.mean - 10) < 1e-9);
  assert.equal(result.changedPercent, 100);
});

test('pixel difference refuses different frame dimensions', () => {
  const api = browserFunctions();
  assert.throws(() => api.cleanupDifferencePixels(new Uint8ClampedArray(8), new Uint8ClampedArray(4), 2, 1),
    /Bildgrößen/);
});

test('cleanup playback has no mode lock and the exact wipe uses inspector stills', () => {
  assert.match(review, /play\.disabled = false/);
  assert.match(review, /seek\.disabled = false/);
  assert.doesNotMatch(review, /if \(exact\(\)\) return;/);
  assert.match(review, /inspector\.appendChild\(fold\)/);
  assert.match(review, /inspector\.querySelector\('\.fi-photo-before'\)/);
  assert.match(css, /\.comparison-player\.cleanup-two-up\.frame-inspector-active \.compare-before,[\s\S]*?visibility:visible !important;/);
});

test('Clean Up wipe grip tracks exact split, fades near edges and labels final state', () => {
  assert.match(review, /class="cleanup-exact-wipe-handle"/);
  assert.match(review, /wipe-grip-original">ORIGINAL<\/span>/);
  assert.match(review, /wipe-grip-result">ERGEBNIS<\/span>/);
  assert.match(review, /data-wipe-current-view/);
  assert.match(review, /comparisonWipeGripOpacity\(width, grip\.offsetWidth \|\| 180, amount\)/);
  assert.match(review, /comparisonWipeVisibleState\(amount, resultLabel\)/);
  assert.doesNotMatch(review, /comparisonWipeGripOffset/);
  assert.match(css, /\.cleanup-exact-wipe-handle \{/);
  assert.match(css, /\.wipe-current-view\s*\{/);
  assert.match(css, /\.wipe-grip-side\s*\{[\s\S]*?color:#fff/);
  const contrast = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/review-contrast.js'), 'utf8');
  const ctx = vm.createContext({});
  vm.runInContext(contrast, ctx);
  assert.equal(ctx.comparisonWipeGripOpacity(640, 180, 0), 0);
  assert.equal(ctx.comparisonWipeGripOpacity(640, 180, 50), 1);
  assert.equal(ctx.comparisonWipeGripOpacity(640, 180, 100), 0);
  assert.equal(ctx.comparisonWipeGripOpacity(640, 180, 15), 0);
  assert.equal(ctx.comparisonWipeGripOpacity(640, 180, 20), 1);
  assert.equal(ctx.comparisonWipeVisibleState(0, 'Localized · experimentell'), 'NUR Localized · experimentell · 100 %');
  assert.equal(ctx.comparisonWipeVisibleState(100), 'NUR ORIGINAL · 100 %');
  assert.equal(ctx.comparisonWipeVisibleState(50), '50 % ORIGINAL · 50 % ERGEBNIS');
});

test('Clean Up settings are a permanent left sidebar and metrics are above the video', () => {
  const start = helpers.indexOf('function drawCleanup(');
  const end = helpers.indexOf('function drawStabilization(', start);
  assert.ok(start >= 0 && end > start);
  const markup = helpers.slice(start, end);
  assert.match(markup, /class="editor-page cleanup-editor-page"/);
  assert.match(markup, /class="card cleanup-settings-sidebar"/);
  assert.match(markup, /class="card editor-viewer cleanup-workspace-viewer"/);
  assert.match(markup, /class="cleanup-results-overview"/);
  assert.match(markup, /class="cleanup-results-metrics"/);
  assert.match(markup, /Was wurde verändert\?/);
  assert.match(markup, /data-cleanup-fold="analysis"/);
  assert.doesNotMatch(markup, /cleanup-params-fold/);
  assert.ok(markup.indexOf('cleanup-settings-sidebar') < markup.indexOf('cleanup-workspace-viewer'));
  assert.ok(markup.indexOf('cleanup-results-overview') < markup.indexOf('cleanupComparisonHtml(comparison)'));
  assert.match(css, /grid-template-columns:clamp\(302px,22vw,370px\) minmax\(0,1fr\)/);
  assert.match(css, /\.cleanup-settings-sidebar\s*\{[\s\S]*?position:sticky/);
});

test('compact layout removes oversized video and still rows', () => {
  assert.match(css, /\.editor-page\.cleanup-editor-page\s*\{\s*display:grid;/);
  assert.match(css, /\.comparison-player\.cleanup-two-up \.cleanup-side-media\s*\{[\s\S]*?height:clamp\(190px,31vh,340px\)/);
  assert.match(css, /\.frame-inspector\[data-fi-stage="cleanup"\] \.fi-exact-media\s*\{[\s\S]*?height:clamp\(175px,27vh,285px\)/);
  assert.match(css, /\.cleanup-wipe-fold:not\(\[open\]\) \.cleanup-exact-wipe\s*\{[\s\S]*?display:none !important/);
  assert.match(review, /fold\.className = 'cleanup-wipe-fold'/);
  assert.match(review, /fold\.hidden = !isInspecting/);
  assert.match(helpers, /class="cleanup-media-fold"/);
});

test('loop and speed controls are above the still image grid', () => {
  const api = browserFunctions();
  const cleanup = api.frameInspectorHtml('cleanup');
  assert.ok(cleanup.indexOf('data-fi-loop') < cleanup.indexOf('data-fi-still-before'));
});

test('Impeccable hierarchy: report status and density toggle share the review header', () => {
  const markup = helpers.slice(
    helpers.indexOf('function drawCleanup('),
    helpers.indexOf('function drawStabilization(')
  );
  assert.match(markup, /class="cleanup-heading-title"/);
  assert.match(markup, /data-cleanup-density-toggle/);
  assert.match(markup, /aria-pressed=/);
  assert.match(markup, /page\.dataset\.cleanupDensity/);
  assert.match(markup, /next\.density = expanded \? 'focus' : 'compact'/);
  assert.match(markup, /Weitere Messwerte und Masken/);
  assert.match(markup, /cleanup-results-overview/);
});

test('Frame correction rail is adjacent to the inspector, not buried in advanced analysis', () => {
  assert.match(review, /className = 'cleanup-frame-rail'/);
  assert.match(review, /slider\.insertAdjacentElement\('afterend', rail\)/);
  assert.match(review, /rail\.append\(plot, legend, readout\)/);
  assert.match(review, /classList\.add\('has-cleanup-frame-rail'\)/);
  assert.match(css, /\.cleanup-frame-rail \.cleanup-gain-plot/);
  assert.match(css, /\.has-cleanup-frame-rail \[data-two-gain\] \{display:none\}/);
});

test('compact and large layouts preserve video and exact comparison panels', () => {
  assert.match(css, /data-cleanup-density="focus"/);
  assert.match(css, /--cleanup-compact-video:clamp\(160px,25vh,285px\)/);
  assert.match(css, /--cleanup-compact-still:clamp\(160px,23vh,260px\)/);
  assert.match(css, /height:var\(--cleanup-compact-video\)/);
  assert.match(css, /height:var\(--cleanup-compact-still\)/);
  assert.match(css, /@media\(pointer:coarse\)/);
  assert.match(css, /\.cleanup-editor-page button:focus-visible/);
  assert.match(review, /Beide abspielen/);
  assert.match(review, /Balken anklicken: exakten Frame anzeigen/);
});

test('correction bars have full-height click target while color encodes amplitude', () => {
  assert.match(helpers, /--correction-height:\$\{height\.toFixed\(1\)\}%/);
  assert.match(css, /\.cleanup-frame-rail \.cleanup-gain-bar::after/);
  assert.match(css, /height:var\(--correction-height,2%\)/);
  assert.match(css, /min-height:100%;height:100%/);
  assert.match(css, /\.cleanup-frame-rail \.cleanup-gain-bar\.bright::after/);
  assert.match(css, /\.cleanup-frame-rail \.cleanup-gain-bar\.dark::after/);
});

test('correction bar and still inspector seek both videos using a shared frame clock', () => {
  const context = {document:{addEventListener() {}}, Uint8ClampedArray};
  vm.createContext(context);
  vm.runInContext(review, context, {filename: 'cleanup-review.js'});
  assert.equal(context.cleanupFrameTimestamp(39, 30, 72), 1.3);
  assert.equal(context.cleanupFrameTimestamp(0, 30, 72), 0);
  assert.equal(context.cleanupFrameTimestamp(71, 30, 72), 71/30);
  assert.equal(context.cleanupFrameTimestamp(72, 30, 72), 71/30);
  assert.equal(context.cleanupFrameTimestamp(2, 0, 72), null);
  assert.match(inspector, /arp-cleanup-seek-video/);
  assert.match(review, /document\.addEventListener\('arp-cleanup-seek-video'/);
  assert.match(review, /pauseBoth\(\);\s*moveBoth\(seconds\)/);
  assert.match(review, /seek\.addEventListener\('change',[\s\S]*?arp-request-exact-frame/);
  assert.match(css, /\.cleanup-results-overview/);
});
test('chart keeps scroll position while selecting frames', () => {
  const recomp = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/render-recomp.js'), 'utf8');
  const clickHandler = recomp.slice(recomp.indexOf("bars.forEach(bar =>"), recomp.indexOf("document.addEventListener('arp-frame-inspected'", recomp.indexOf("bars.forEach(bar =>")));
  assert.doesNotMatch(clickHandler, /scrollIntoView/);
  assert.match(clickHandler, /arp-request-exact-frame/);
});

test('sidebar execution actions stay visible near the top of lengthy settings', () => {
  const markup = helpers.slice(
    helpers.indexOf('function drawCleanup('),
    helpers.indexOf('function drawStabilization(')
  );
  assert.ok(markup.indexOf('cleanup-primary-actions') > markup.indexOf('progressHtml('));
  assert.ok(markup.indexOf('cleanup-primary-actions') < markup.indexOf('st.fields.map('));
  assert.match(css, /\.cleanup-settings-sidebar \.cleanup-primary-actions\s*\{[\s\S]*?position:sticky/);
});

test('source-only exact frame UI remains available without a processed output', () => {
  const api = browserFunctions();
  const sourceHtml = api.cleanupSourceInspectorHtml();
  assert.match(sourceHtml, /Original · Einzelbildansicht/);
  assert.match(sourceHtml, /data-source-prev10/);
  assert.match(sourceHtml, /data-source-next/);
  assert.match(sourceHtml, /data-source-zoom/);
  assert.match(sourceHtml, /data-source-timeline/);
  assert.doesNotMatch(sourceHtml, /data-fi-difference/);
  assert.match(helpers, /cleanupSourceInspectorHtml\(\)/);
  assert.match(helpers, /cleanup-source-video/);
  const recomp = fs.readFileSync(path.join(root, 'ai_remaster_gui/static/js/render-recomp.js'), 'utf8');
  assert.match(recomp, /setupCleanupSourceInspector\(\)/);
});

test('variant labels distinguish FFmpeg arithmetic mean from median and ARP', () => {
  assert.match(helpers, /FFmpeg Arithmetic Mean/);
  assert.match(helpers, /ffmpeg_am/);
  assert.match(review, /FFMPEG ARITHMETIC MEAN/);
  assert.match(review, /FFmpeg Arithmetic Mean/);
});

test('FFmpeg bar shows turquoise brightening and orange darkening in the same frame', () => {
  const context = {document: {addEventListener() {}}, Uint8ClampedArray};
  vm.createContext(context);
  vm.runInContext(review, context, {filename: 'cleanup-review.js'});
  const rows = context.cleanupDifferenceBarData({
    frames: 3, metric:'directional_mean_luma_8bit', sampled:false,
    samples:[
      {frame:0,mean_abs_luma:2,brighten_mean_luma:2,darken_mean_luma:0},
      {frame:1,mean_abs_luma:5,brighten_mean_luma:3,darken_mean_luma:2},
      {frame:2,mean_abs_luma:3,brighten_mean_luma:0,darken_mean_luma:3},
    ],
  });
  assert.equal(rows.length, 3);
  assert.equal(rows[0].frame, 0);
  assert.equal(rows[1].height, 100);
  assert.equal(rows[0].height, 40);
  assert.equal(rows[1].brightHeight, 100);
  assert.ok(rows[1].darkHeight > 0);
  assert.equal(rows[0].darkHeight, 0);
  assert.equal(rows[2].brightHeight, 0);
  assert.throws(() => context.cleanupDifferenceBarData({
    frames:3,metric:'directional_mean_luma_8bit',samples:[
      {frame:1,mean_abs_luma:4,brighten_mean_luma:3,darken_mean_luma:1},
      {frame:1,mean_abs_luma:3,brighten_mean_luma:2,darken_mean_luma:1}
    ]
  }), /Inkonsistente/);
  assert.match(review, /fetch\('\/api\/cleanup-frame-differences'/);
  assert.match(review, /slider\.insertAdjacentElement\('afterend', rail\)/);
  assert.match(review, /setupFFmpegDifferenceRail\(slider, player, before\)/);
  assert.match(review, /document\.dispatchEvent\(new CustomEvent\('arp-request-exact-frame'/);
  assert.match(review, /--bright-height/);
  assert.match(review, /--dark-height/);
  assert.match(review, /Türkis: aufgehellt/);
  assert.match(review, /Orange: abgedunkelt/);
  assert.match(css, /\.cleanup-difference-rail \.cleanup-gain-bar\.difference::before/);
  assert.match(css, /\.cleanup-difference-rail \.cleanup-gain-bar\.difference::after/);
  const handler = fs.readFileSync(path.join(root, 'ai_remaster_gui/http_handler.py'), 'utf8');
  assert.match(handler, /parsed\.path == "\/api\/cleanup-frame-differences"/);
});

test('Localized research is opt-in and clearly identified as not production-ready', () => {
  const models = fs.readFileSync(path.join(root, 'ai_remaster_gui/models.py'), 'utf8');
  assert.match(models, /select:off\|arp\|ffmpeg_median\|ffmpeg_am\|localized_research/);
  assert.match(models, /localized_confirm_single_shot/);
  assert.match(helpers, /Localized · Langfilm-Streaming \(experimentell\)/);
  const picker=fs.readFileSync(path.join(root,'ai_remaster_gui/static/js/render-deflicker.js'),'utf8');
  assert.match(picker, /Localized methods are experimental/);
  assert.match(picker, /at most 240 frames/);
  assert.doesNotMatch(helpers, /data-deflicker-only=/);
  assert.doesNotMatch(helpers, /value="localized_research" disabled/);
  assert.match(review, /LOCALIZED · EXPERIMENTELL/);
  assert.match(review, /setupFFmpegDifferenceRail/);
  assert.match(review, /'localized_research'/);
});
