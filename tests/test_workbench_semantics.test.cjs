const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const read=name=>fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static',name),'utf8');
test('workbench uses one explicit method help instead of decorating every control',()=>{
 const layout=read('js/comparison-layout.js');
 assert.match(layout,/data-method-help/);assert.doesNotMatch(layout,/mountComparisonHelp\(page\)/);
 assert.match(layout,/heading\.after\(zoomHint\)/);
 assert.doesNotMatch(layout,/analysis\.append\([^;]*review-zoom-hint/);
});
test('pipeline and membership have independent state markers',()=>{
 const layout=read('js/comparison-layout.js'),css=read('comparison-workbench.css');
 assert.match(layout,/classList\.toggle\('is-pipeline'/);assert.match(layout,/data-comparison-membership|comparisonMembership/);
 assert.match(css,/\.comparison-method-row\.is-pipeline\s*\{[^}]*pipeline-fill/);
 assert.match(css,/data-zoom-native[^}]*color:#142638; background:#c6def2/);
});
test('state palette foregrounds have AA text contrast',()=>{
 const luminance=hex=>{const c=hex.match(/\w\w/g).map(v=>parseInt(v,16)/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4);return c[0]*.2126+c[1]*.7152+c[2]*.0722};
  for(const [background,foreground] of [['244c70','eef6ff'],['205358','d6f0f1'],['2c593d','edf7ef'],['c6def2','142638'],['214c35','e0f5e7'],['5b481b','fff0be'],['642d2a','ffe4de']]){
  const a=luminance(background),b=luminance(foreground);
  assert.ok((Math.max(a,b)+.05)/(Math.min(a,b)+.05)>=4.5,`${background}/${foreground}`);
 }
});
