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

test('Clean Up provides permanent three-up exact frame inspection; stabilization remains unchanged', () => {
  const api = browserFunctions();
  const cleanup = api.frameInspectorHtml('cleanup');
  assert.match(cleanup, /data-fi-still-before/);
  assert.match(cleanup, /data-fi-still-after/);
  assert.match(cleanup, /data-fi-difference/);
  assert.match(cleanup, /PIXELDIFFERENZ ×8/);
  assert.match(cleanup, /Einzelbilder ausblenden/);
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

test('comparison workspace appears before collapsed settings and full diagnostics', () => {
  const start = helpers.indexOf('function drawCleanup(');
  const end = helpers.indexOf('function drawStabilization(', start);
  assert.ok(start >= 0 && end > start);
  const markup = helpers.slice(start, end);
  assert.match(markup, /class="editor-page cleanup-editor-page"/);
  assert.match(markup, /class="card editor-viewer cleanup-workspace-viewer"/);
  assert.match(markup, /class="cleanup-params-fold"/);
  assert.match(markup, /data-cleanup-fold="analysis"/);
  assert.match(markup, /data-cleanup-fold="settings"/);
  assert.ok(markup.indexOf('cleanup-workspace-viewer') < markup.indexOf('cleanup-params-fold'));
  assert.match(markup, /const settingsOpen = typeof prefs\.settings === 'boolean'/);
  assert.match(markup, /comparison\.exists === 'true'/);
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
});

test('Frame correction rail is adjacent to the inspector, not buried in advanced analysis', () => {
  assert.match(review, /className = 'cleanup-frame-rail'/);
  assert.match(review, /inspector\.insertAdjacentElement\('afterend', rail\)/);
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
