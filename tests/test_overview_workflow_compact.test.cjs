const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/render-overview.js'),'utf8');
function setup(){
 const context=vm.createContext({state:{settings:{global:{deflicker:'true',enhancement:'true'}}},pipelineOrderPending:false,esc:value=>String(value).replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;'),pipelineOrder:()=>['deflicker','denoise','enhancement']});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/render-helpers.js'),'utf8'),context);
 vm.runInContext(source,context);return context;
}
test('compact workflow retains eleven unique real inputs and a separate preprocessing order host',()=>{
 const context=setup(),html=context.workflowPickerHtml(true,false,true,true,true,false,false,false,'Black and white');
 const ids=[...html.matchAll(/<input id="([^"]+)"/g)].map(m=>m[1]);
  assert.equal(ids.length,11);assert.equal(new Set(ids).size,11);assert.ok(ids.includes('globalDust'));
  assert.match(html,/data-preprocessing-stages/);assert.match(source,/<h2>Processing stages/);
 assert.match(html,/<ol class="workflow-sequence-list" data-preprocessing-stages>/);
  assert.equal([...html.matchAll(/<li class="workflow-stage-tile/g)].length,11);
 assert.doesNotMatch(html,/workflow-stage-grid/);
 assert.match(html,/Detail Enhancement · legacy/);assert.match(html,/workflow-tiles/);
   assert.equal([...html.matchAll(/class="workflow-thumb /g)].length,11);
   assert.equal([...html.matchAll(/class="workflow-paired-row"/g)].length,11);
   assert.equal([...html.matchAll(/data-workflow-stage-status=/g)].length,11);
 assert.doesNotMatch(html,/pipeline-order-controls/);
  assert.equal([...html.matchAll(/data-workflow-order-key=/g)].length,22);
 for(const label of html.matchAll(/<label\b[^>]*>[\s\S]*?<\/label>/g)) assert.doesNotMatch(label[0],/<button/);
});
test('tile ordering disables boundary arrows and all arrows during processing or pending save',()=>{
 const context=setup();
 const html=context.workflowOrderControlsHtml('deflicker','Light Deflicker');
 assert.match(html,/direction="-1"[^>]* disabled/);
 assert.doesNotMatch(html,/direction="1"[^>]* disabled/);
 for(const flag of ['running','queue_running']) {
  context.state[flag]=true;
  assert.equal([...context.workflowOrderControlsHtml('denoise','Remove Noise').matchAll(/ disabled/g)].length,2);
  context.state[flag]=false;
 }
 context.pipelineOrderPending=true;
 assert.equal([...context.workflowOrderControlsHtml('denoise','Remove Noise').matchAll(/ disabled/g)].length,2);
});
test('short summaries retain full descriptions for assistive technology and safe tooltip text',()=>{
 const context=setup(),html=context.workflowOptionHtml({id:'globalDenoise',title:'Remove Noise',body:'Protect "grain" <carefully>',checked:true});
 assert.match(html,/aria-describedby="globalDenoiseDescription" checked/);
 assert.match(html,/id="globalDenoiseDescription">Protect &quot;grain&quot; &lt;carefully>/);
 assert.match(html,/Reduce noise; protect grain/);
 assert.match(html,/aria-labelledby="globalDenoiseTitle"/);
 assert.match(html,/class="workflow-stage-state" aria-hidden="true"/);
});
test('source samples have individual captions, explain their purpose and handle an empty preview set',()=>{
 const context=setup();context.media=value=>'/media?path='+encodeURIComponent(value);
 context.state.source_previews=['first "frame".jpg','second.jpg','third.jpg'];
 const html=context.overviewFilmstripInner();
 assert.equal([...html.matchAll(/<figure>/g)].length,3);
 assert.equal([...html.matchAll(/<figcaption>Source sample \d<\/figcaption>/g)].length,3);
 assert.match(html,/quick visual check\. Not restored outputs/);
 assert.match(html,/first%20%22frame%22\.jpg/);
 assert.equal(context.overviewFilmstrip(),html);
 context.state.source_previews=[];assert.equal(context.overviewFilmstripInner(),'');
});
test('source samples live beside the video before the unchanged section controls',()=>{
 const context=setup();context.media=String;context.parseDuration=()=>12;context.formatSeconds=String;
 context.state.source_previews=['sample.jpg'];
 const html=context.overviewSectionPicker({section_start:'1',section_end:'7'},'source.mp4','playback.mp4');
 assert.match(html,/overview-source-layout/);
 assert.ok(html.indexOf('id="overviewFilmstrip"')<html.indexOf('class="editor-controls"'));
 for(const id of ['sourceSectionVideo','sectionStart','sectionEnd','sectionStartLabel','sectionEndLabel','overviewFilmstrip','overviewSourceInfo']) assert.equal(html.split('id="'+id+'"').length,2);
 assert.match(html,/value="1"/);assert.match(html,/value="7"/);
 assert.doesNotMatch(context.overviewSectionPicker({},'',''),/<video|class="source-samples"/);
});
test('source metadata keeps every supported value and moves into the shared source layout',()=>{
 const context=setup();context.media=String;context.parseDuration=()=>12;context.formatSeconds=String;
 context.state.source_info={resolution:'960x720',aspect:'1.333:1',duration:'0:02',frame_rate:'30 fps',frames:'72',video_codec:'ffv1',pixel_format:'yuv420p',colour:'bt470bg / tv',audio:'none',container:'matroska',overall_bitrate:'47 Mbps',video_bitrate:'46 Mbps',size:'13 MB',codec_note:'<unsafe> "note"'};
 const info=context.sourceInfoHtml(context.state.source_info);
 assert.equal([...info.matchAll(/<dt>/g)].length,14);assert.equal([...info.matchAll(/<dd>/g)].length,14);
 assert.match(info,/Metadata of the selected source clip/);assert.match(info,/&lt;unsafe> &quot;note&quot;/);
 const html=context.overviewSectionPicker({},'source.mp4','playback.mp4');
 assert.ok(html.indexOf('id="overviewSourceInfo"')<html.indexOf('class="editor-controls"'));
 assert.match(html,/class="source-metadata"/);assert.equal(context.sourceInfoHtml({}),'');
});
test('whole-remaster actions retain their handlers and dynamic overall progress host',()=>{
 const context=setup();context.progressHtml=(percent,label)=>`<progress value="${percent}">${label}</progress>`;
 context.state.phase_progress={global:{percent:37,label:'37% complete'}};
 const html=context.overviewActions();
 assert.match(html,/onclick="runAll\(\)"/);assert.match(html,/onclick="stopRun\(\)" disabled/);
 assert.match(html,/id="overviewPipelineProgress"/);assert.match(html,/value="37"/);
  assert.match(source,/Force enabled always reruns/);
 context.state.running=true;assert.doesNotMatch(context.overviewActions(),/onclick="stopRun\(\)" disabled/);
});
test('paired status follows the stage key, not the position or similar stage names',()=>{
 const context=setup();
 context.state.phase_progress={stages:[{key:'enhancement',stage:'Detail Enhancement / Deblur',percent:61,label:'Working'},{key:'detail',stage:'Detail Enhancement',percent:100,label:'Ready'}]};
 context.state.progress=[{stage:'Detail Enhancement',status:'Ready',latest:'legacy.mkv'},{stage:'Detail Enhancement / Deblur',status:'Running',latest:'current/<frame>.mkv'}];
 const html=context.workflowStageProgressHtml('globalEnhancement',true);
 assert.match(html,/Running/);assert.match(html,/value="61"/);assert.match(html,/current\/&lt;frame>\.mkv/);
 assert.doesNotMatch(html,/legacy\.mkv|value="100"/);
});
test('disabled and unreported stages keep truthful paired rows without invented percentages',()=>{
 const context=setup();
 context.state.phase_progress={stages:[{key:'detail',stage:'Detail Enhancement',percent:100,label:'Ready'}]};
 const off=context.workflowStageProgressHtml('globalDetail',false);
 assert.match(off,/Off · skipped/);assert.doesNotMatch(off,/<progress|100|Ready/);
 const pending=context.workflowStageProgressHtml('globalUpscale',true);
 assert.match(pending,/Selected/);assert.match(pending,/No status reported yet/);assert.doesNotMatch(pending,/<progress/);
});
test('colorization retains all three real substeps while unmatched pipeline work remains visible',()=>{
 const context=setup();
 context.state.phase_progress={stages:[{key:'shots',stage:'Shot Detection',percent:100,label:'Ready'},{key:'references',stage:'Reference Generation',percent:32,label:'Working'},{key:'colour',stage:'Colorization',percent:0,label:'Waiting'},{key:'recomp',stage:'Recomposition',percent:0,label:'Waiting'}]};
 context.state.progress=[{stage:'Shot Detection',status:'Ready',latest:'shots.json'},{stage:'Reference Generation',status:'Running',latest:'reference.png'},{stage:'Colorization',status:'Waiting',latest:''},{stage:'Recomposition',status:'Waiting',latest:''}];
 const grouped=context.workflowStageProgressHtml('globalColorize',true);
 for(const title of ['Shot Detection','Reference Generation','Colorization']) assert.match(grouped,new RegExp(title));
 assert.equal([...grouped.matchAll(/class="workflow-status-entry"/g)].length,3);
 const extra=context.overviewProgressTable();assert.match(extra,/Recomposition/);assert.doesNotMatch(extra,/Reference Generation|Shot Detection|Colorization/);
 context.state.progress=[];assert.equal(context.overviewProgressTable(),'');
});
test('stage status rows use current progress and escape output without hard-coded stages',()=>{
 const context=setup();context.stageProgressByTitle=()=>({percent:61,label:'Current work'});
 context.progressHtml=(percent,label)=>`<progress value="${percent}">${label}</progress>`;
 context.state.progress=[{stage:'Test <stage>',status:'Running',latest:'output/"frame"<1>.mkv'},{stage:'Another stage',status:'Waiting',latest:''}];
 const html=context.overviewProgressTable();
 assert.equal([...html.matchAll(/class="overview-progress-stage"/g)].length,2);
 assert.match(html,/Status: Running/);assert.match(html,/value="61"/);
 assert.match(html,/Test &lt;stage>/);assert.match(html,/output\/&quot;frame&quot;&lt;1>\.mkv/);
 assert.match(html,/Latest output: —/);assert.doesNotMatch(html,/<table|Neighbors|29%/);
});
