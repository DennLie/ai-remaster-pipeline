const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
function setup() {
  const elements={tabs:{innerHTML:''}},calls=[];
  const context=vm.createContext({window:{addEventListener(){}},document:{getElementById:id=>elements[id],querySelector:()=>null,querySelectorAll:()=>[]},esc:String,alert:message=>{throw Error(message)}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/core.js'),'utf8'),context);
  vm.runInContext(`state={running:false,settings:{global:{}},stages:[{key:'denoise',title:'Remove Noise'},{key:'deflicker',title:'Light Deflicker'},{key:'enhancement',title:'Detail Enhancement / Deblur'},{key:'cleanup',title:'Clean Up'}]};active='deflicker'`,context);
  context.stage=key=>vm.runInContext(`state.stages.find(item=>item.key==='${key}')`,context);
  context.postJson=async(url,body)=>{calls.push({url,body});return {ok:true}};
  context.refresh=async force=>calls.push({force});
  return {context,elements,calls};
}
test('tab bar has draggable video stages, keyboard menu and fixed artifact tabs',()=>{
  const {context,elements}=setup();context.drawTabs();
  assert.equal((elements.tabs.innerHTML.match(/draggable="true"/g)||[]).length,4);
  assert.match(elements.tabs.innerHTML,/Reorder Light Deflicker/);
  assert.match(elements.tabs.innerHTML,/Move earlier/);assert.match(elements.tabs.innerHTML,/Fixed pipeline position/);
});
test('moving a tab saves actual global order and refreshes Overview as well as navigation',async()=>{
  const {context,calls}=setup();await context.movePipelineStage('deflicker',-1);
  assert.equal(calls[0].body.values.pipeline_order,'deflicker,denoise,enhancement,cleanup,stabilize,dust,detail,outpaint,shots,references,colour,recomp,audio,upscale');
  assert.equal(calls[0].body.stage,'global');assert.equal(calls[1].force,true);
});
test('drop saves the same order, foreign drags and busy changes cannot submit',async()=>{
  const {context,calls,elements}=setup();
  context.pipelineDrop({preventDefault(){},dataTransfer:{getData:()=> 'deflicker'}},'denoise');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(calls[0].body.values.pipeline_order,'deflicker,denoise,enhancement,cleanup,stabilize,dust,detail,outpaint,shots,references,colour,recomp,audio,upscale');
  vm.runInContext('state.running=true',context);await context.movePipelineStage('deflicker',-1);
  context.drawTabs();assert.doesNotMatch(elements.tabs.innerHTML,/draggable="true"/);
  assert.equal(calls.length,2);
});
test('reference metadata distinguishes original source from selected upstream output',()=>{
  const {context}=setup();
  vm.runInContext(`state.processing_input={label:'Light Deflicker',from_stage:'deflicker',source:'intermediate/flicker.mkv',parameters:{method:'arp'},chain:[{key:'deflicker',title:'Light Deflicker'},{key:'denoise',title:'Remove Noise'}]}`,context);
  const html=context.processingInputHtml('denoise');
  assert.match(html,/Stage input:/);assert.match(html,/Selected for pipeline/);assert.match(html,/Light Deflicker/);
  assert.match(html,/method: arp/);assert.match(html,/aria-current="step">Remove Noise/);
});
