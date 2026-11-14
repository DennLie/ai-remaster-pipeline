const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const context = vm.createContext({AbortController, ResizeObserver:class{observe(){} disconnect(){}}});
vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/review-zoom.js'),'utf8'),context);

function fixture() {
  const events={},capture=new Set();
  const wrappers=[{w:400,h:300},{w:400,h:130}].map(({w,h})=>{
    const classes=new Set();
    const media={videoWidth:960,videoHeight:720};
    const grid={style:{setProperty(){}},hidden:true};
    const plane={style:{},lastElementChild:grid};
    const window={style:{},firstElementChild:plane,getBoundingClientRect(){
      return {left:0,top:0,width:parseFloat(this.style.width),height:parseFloat(this.style.height)};
    }};
    return {clientWidth:w,clientHeight:h,classes,plane,grid,window,setAttribute(){},
      classList:{add(c){classes.add(c)},remove(c){classes.delete(c)},toggle(c,v){v?classes.add(c):classes.delete(c)}},
      querySelector(s){return s==='video, img, canvas'?media:window},
    };
  });
  const host={isConnected:true,querySelectorAll:()=>wrappers,
    addEventListener(name,callback){events[name]=callback},
    setPointerCapture(id){capture.add(id)},hasPointerCapture:id=>capture.has(id),releasePointerCapture(id){capture.delete(id)},
  };
  const state={zoom:1,x:0,y:0,grid:false};
  const controller=context.createReviewZoom(host,'.viewport',null,state);
  const event=(wrap,x,y,id=1)=>({pointerId:id,pointerType:'mouse',button:0,clientX:x,clientY:y,
    target:{closest:()=>wrap},preventDefault(){this.prevented=true},stopPropagation(){this.stopped=true}});
  return {wrappers,events,capture,state,controller,event};
}

test('fit geometry preserves image aspect ratio in a short diff box',()=>{
  const normal=context.reviewZoomGeometry(400,300,960,720);
  const diff=context.reviewZoomGeometry(400,130,960,720);
  assert.ok(Math.abs(normal.width/normal.height-diff.width/diff.height)<1e-9);
  assert.equal(diff.height,130);
  assert.equal(context.reviewZoomGeometry(0,300,960,720),null);
  assert.equal(context.reviewZoomClamp(0,-1,1),0);
});

test('continuous zoom clamps to fit and 32x; fit resets shared pan',()=>{
  const {controller,state}=fixture();
  controller.setZoom(3.57);
  assert.equal(state.zoom,3.57);
  controller.setZoom(100);
  assert.equal(state.zoom,32);
  state.x=.6;state.y=-.3;
  controller.fit();
  assert.deepEqual(state,{zoom:1,x:0,y:0,grid:false,native:false});
});

test('native zoom maps a source pixel to one physical display pixel, independently of Fit',()=>{
  assert.equal(context.reviewZoomNativeScale(960,400,1),2.4);
  assert.equal(context.reviewZoomNativeScale(960,400,2),1.2);
  assert.equal(context.reviewZoomNativeScale(64,400,1),.16);
  const {controller,state,wrappers}=fixture();
  controller.native();
  assert.equal(state.zoom,2.4);
  assert.equal(state.native,true);
  for(const wrap of wrappers) {
    const scale=Number(wrap.plane.style.transform.match(/scale\(([^)]+)\)/)[1]);
    assert.ok(Math.abs(parseFloat(wrap.window.style.width)*scale/960-1)<1e-9);
  }
  controller.setZoom(.5);
  assert.equal(state.zoom,.5);
  assert.equal(state.native,false);
});

test('all differently sized views show the same normalized crop',()=>{
  const {controller,state,wrappers}=fixture();
  state.zoom=4;state.x=.5;state.y=-.4;
  controller.apply();
  const crops=wrappers.map(w=>{
    const [x,y]=w.plane.style.transform.match(/translate\(([-\d.]+)px, ([-\d.]+)px/).slice(1).map(Number);
    return [x/parseFloat(w.window.style.width),y/parseFloat(w.window.style.height)];
  });
  assert.ok(Math.abs(crops[0][0]-crops[1][0])<1e-9);
  assert.ok(Math.abs(crops[0][1]-crops[1][1])<1e-9);
});

test('dragging a live diff pans the shared crop, clamps, and suppresses lightbox click',()=>{
  const {controller,state,wrappers,events,event,capture}=fixture();
  controller.setZoom(2);
  const down=event(wrappers[1],50,50);
  events.pointerdown(down);
  assert.ok(capture.has(1));
  assert.ok(wrappers[1].classes.has('is-panning'));
  events.pointermove(event(wrappers[1],100,100,2));
  assert.equal(state.x,0);
  events.pointermove(event(wrappers[1],500,500));
  assert.equal(state.x,1);assert.equal(state.y,1);
  events.pointercancel(event(wrappers[1],500,500));
  assert.equal(capture.size,0);
  assert.equal(wrappers[1].classes.has('is-panning'),false);
  const click=event(wrappers[1],500,500);
  events.click(click);
  assert.equal(click.stopped,true);
});

test('wheel zoom is anchored at the cursor and preserves browser ctrl-wheel',()=>{
  const {state,wrappers,events,event}=fixture();
  const wheel=event(wrappers[0],300,75);
  Object.assign(wheel,{deltaY:-100,deltaMode:0});
  events.wheel(wheel);
  const z=state.zoom;
  const imagePoint=(.25-state.x*(z-1)/2)/z;
  assert.ok(Math.abs(imagePoint-.25)<1e-9);
  const ctrl=event(wrappers[0],300,75);
  Object.assign(ctrl,{ctrlKey:true,deltaY:-100});
  events.wheel(ctrl);
  assert.equal(state.zoom,z);
  assert.equal(ctrl.prevented,undefined);
});

test('grid only appears when individual decoded pixels are large enough',()=>{
  const {controller,state,wrappers}=fixture();
  state.grid=true;
  controller.setZoom(2);
  assert.equal(wrappers[0].grid.hidden,true);
  controller.setZoom(32);
  assert.equal(wrappers[0].grid.hidden,false);
  assert.equal(wrappers[1].grid.hidden,true);
});

test('keyboard panning shares the same crop and stops at its boundaries',()=>{
  const {controller,state,wrappers,events,event}=fixture();
  controller.setZoom(2);
  const key=event(wrappers[0],0,0);
  key.key='ArrowLeft';
  events.keydown(key);
  assert.equal(key.prevented,true);
  assert.equal(state.x,.1);
  for(let i=0;i<30;i++) events.keydown(key);
  assert.equal(state.x,1);
  controller.fit();
  events.keydown(key);
  assert.equal(state.x,0);
});
