const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const settle=async()=>{for(let i=0;i<6;i++) await new Promise(setImmediate)};

function setup() {
  const requests=[],seeks=[],bars=[];
  let selection={path:'first.mkv',identity:'source-v1'},frame=4;
  const status={textContent:''},chart={title:'',addEventListener(){},querySelectorAll:()=>bars,
    replaceChildren(fragment){bars.splice(0,bars.length,...(fragment?.children || []));}};
  const host={isConnected:true,querySelector:selector=>selector==='[data-dr-chart]'?chart:status};
  const context=vm.createContext({AbortController,document:{
    createDocumentFragment:()=>({children:[],appendChild(child){this.children.push(child)}}),
    createElement:()=>({dataset:{},style:{setProperty(){}},classList:{toggle(key,value){this[key]=value}},
      setAttribute(key,value){this[key]=value},addEventListener(name,callback){this[name]=callback}}),
  },fetch:(url,options)=>new Promise(resolve=>requests.push({url,signal:options.signal,resolve}))});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/denoise-review.js'),'utf8'),context);
  const controller=context.createDenoiseReviewChart(host,{stage:'deflicker',selection:()=>selection,frame:()=>frame,
    seek:value=>seeks.push(value),changed(){}});
  const finish=index=>requests[index].resolve({ok:true,json:async()=>({ok:true,metric:'directional_mean_luma_8bit',frames:10,
    samples:[{frame:4,mean_abs_luma:3,brighten_mean_luma:1,darken_mean_luma:2}]})});
  return {controller,host,bars,status,requests,seeks,finish,
    select(value){selection=value},frame(value){frame=value}};
}

test('chart navigation uses the shared clock and highlighting does not change measured values',async()=>{
  const view=setup();
  try {
    view.controller.reconcile();view.finish(0);await settle();
    assert.match(view.requests[0].url,/stage=deflicker/);
    view.controller.highlight();assert.equal(view.bars[0].classList['is-current'],true);
    assert.match(view.status.textContent,/brightened 1.00.*darkened 2.00/);
    view.bars[0].click();assert.deepEqual(view.seeks,[4]);
    view.frame(8);view.controller.highlight();assert.equal(view.bars[0].classList['is-current'],false);
    assert.match(view.status.textContent,/Frame 9: not in the chart sample/);
    view.controller.highlight(4);assert.match(view.status.textContent,/brightened 1.00.*darkened 2.00/);
  }finally{view.controller.destroy()}
});

test('changed source revisions and disposal reject late chart responses and old bar clicks',async()=>{
  const view=setup();
  view.controller.reconcile();view.finish(0);await settle();const old=view.bars[0];
  view.select({path:'first.mkv',identity:'source-v2'});view.controller.reconcile();
  assert.equal(view.requests[0].signal.aborted,true);old.click();assert.deepEqual(view.seeks,[]);
  view.select({path:'second.mkv',identity:'source-v2'});view.controller.reconcile();
  view.finish(1);await settle();assert.equal(view.bars.length,0);
  view.controller.destroy();assert.equal(view.requests[2].signal.aborted,true);
  view.finish(2);await settle();assert.equal(view.bars.length,0);
});
