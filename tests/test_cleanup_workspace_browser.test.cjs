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
  assert.match(review, /inspector\.appendChild\(card\)/);
  assert.match(review, /inspector\.querySelector\('\.fi-photo-before'\)/);
  assert.match(css, /\.comparison-player\.cleanup-two-up\.frame-inspector-active \.compare-before,[\s\S]*?visibility:visible !important;/);
});
