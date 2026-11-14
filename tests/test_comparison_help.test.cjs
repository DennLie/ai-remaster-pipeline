const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../ai_remaster_gui/static/js/comparison-help.js'), 'utf8');

function setup() {
  class Element {
    constructor(tag = 'div') {
      this.tagName = tag.toUpperCase(); this.children = []; this.attrs = new Map();
      this.dataset = {}; this.style = {}; this.className = ''; this.textContent = '';
      this.isConnected = true; this.rect = {left: 310, top: 210, bottom: 240, width: 200, height: 80};
      this.classList = {contains: name => this.className.split(' ').includes(name)};
    }
    setAttribute(key, value) { this.attrs.set(key, value); }
    getAttribute(key) { return this.attrs.get(key) || null; }
    removeAttribute(key) { this.attrs.delete(key); }
    append(...elements) {
      for (const element of elements) {
        if (element.parentElement) element.parentElement.children.splice(element.parentElement.children.indexOf(element), 1);
        element.parentElement = this; this.children.push(element);
      }
    }
    before(element) {
      const parent = this.parentElement;
      element.parentElement = parent; parent.children.splice(parent.children.indexOf(this), 0, element);
    }
    after(element) {
      const parent = this.parentElement;
      element.parentElement = parent; parent.children.splice(parent.children.indexOf(this) + 1, 0, element);
    }
    contains(element) { return element === this || this.children.some(child => child.contains(element)); }
    closest(selector) {
      if (selector === 'dialog[open]' && this.tagName === 'DIALOG' && this.open) return this;
      if (selector === 'label' && this.tagName === 'LABEL') return this;
      if (selector.startsWith('.') && this.classList.contains(selector.slice(1))) return this;
      return this.parentElement?.closest(selector) || null;
    }
    querySelector() { return null; }
    getBoundingClientRect() { return this.rect; }
  }
  const listeners = {}, windowListeners = {}, timers = new Map();
  let nextTimer = 0, now = 0;
  const body = new Element('body');
  const list = new Element(); list.className = 'deflicker-variant-list'; body.append(list);
  const cell = new Element('button'); cell.dataset.method = 'cas'; cell.setAttribute('aria-label', 'CAS, ready'); list.append(cell);
  const label = new Element('label'), field = new Element('input'); field.dataset.field = 'mix'; label.textContent = 'Mix'; label.append(field); body.append(label);
  const play = new Element('button'); body.append(play);
  const selections = {
    '[data-denoise-cell][data-method]': [cell], '.denoise-variant-col': [], '[data-field]': [field], '#denoiseComparePlay': [play],
  };
  const document = {
    body, activeElement: null, createElement: tag => new Element(tag),
    querySelectorAll: selector => selections[selector] || [],
    addEventListener: (type, fn, capture) => (listeners[type] ||= []).push({fn, capture}),
  };
  const context = vm.createContext({
    document, window: {innerWidth: 360, innerHeight: 280, addEventListener: (type, fn) => windowListeners[type] = fn},
    setTimeout: (fn,delay=0) => { timers.set(++nextTimer, {fn,at:now+delay}); return nextTimer; }, clearTimeout: id => timers.delete(id),
  });
  vm.runInContext(source, context);
  const fire = (type, target, extra = {}) => {
    const event = {target, relatedTarget: null, preventDefault() { this.prevented = true; }, stopPropagation() { this.stopped = true; }, ...extra};
    for (const {fn} of listeners[type] || []) fn(event);
    return event;
  };
  return {context, document, body, list, cell, field, play, listeners, fire, selections,
    flush() { const pending = [...timers.values()]; timers.clear(); pending.forEach(({fn}) => fn()); },
    advance(ms) {
      const until=now+ms;
      while(true) {
        const next=[...timers].filter(([,timer])=>timer.at<=until).sort((a,b)=>a[1].at-b[1].at)[0];
        if(!next) break;
        timers.delete(next[0]);now=next[1].at;next[1].fn();
      }
      now=until;
    },
    tooltip: () => body.children.find(element => element.id === 'comparison-context-help'),
    info: () => list.children[0].children[1],
  };
}

test('content is factual, complete for current methods, and executable without a DOM', () => {
  const context = vm.createContext({}); vm.runInContext(source, context);
  const methods = vm.runInContext('comparisonHelpMethods', context);
  assert.equal(Object.keys(methods).length, 17);
  assert.match(methods.zoom0, /clip source content/);
  assert.match(methods.zoom3, /does not guarantee/);
  assert.match(methods.richardson_lucy, /not a measured/);
  assert.match(methods.nafnet, /GoPro-trained/);
  assert.match(methods.localized_stream, /detection can fail/);
  assert.match(vm.runInContext('comparisonHelpFields.mix', context), /0% is input/);
  assert.match(vm.runInContext('comparisonHelpControls["[data-zoom-range]"]', context), /Alt\+wheel.*ordinary wheel scrolls.*Ctrl\/Meta\+wheel/);
});

test('mount is idempotent with one set of delegated handlers and sibling info buttons', () => {
  const s = setup(); s.context.mountComparisonHelp(); const button = s.info();
  s.context.mountComparisonHelp();
  assert.equal(s.info(), button);
  assert.equal(s.list.children.length, 1);
  assert.equal(s.cell.parentElement, button.parentElement);
  assert.equal(s.cell.contains(button), false);
  const label = s.field.closest('label');
  const fieldInfo = label.parentElement.children[label.parentElement.children.indexOf(label) + 1];
  assert.equal(fieldInfo.className, 'comparison-help-info');
  assert.equal(label.contains(fieldInfo), false);
  for (const handlers of Object.values(s.listeners)) assert.equal(handlers.length, 1);
  assert.equal(s.listeners.click[0].capture, true);
  assert.equal(s.cell.getAttribute('aria-label'), 'CAS, ready');
});

test('focus describes without replacing existing labels or descriptions; Escape dismisses', () => {
  const s = setup(); s.context.mountComparisonHelp();
  s.cell.setAttribute('aria-describedby', 'existing');
  s.fire('focusin', s.cell);
  assert.equal(s.cell.getAttribute('aria-describedby'), 'existing comparison-context-help');
  assert.equal(s.tooltip().getAttribute('role'), 'tooltip');
  assert.match(s.tooltip().textContent, /contrast-adaptive/);
  const escape = s.fire('keydown', s.cell, {key: 'Escape'});
  assert.equal(escape.prevented, true);
  assert.equal(s.tooltip().hidden, true);
  assert.equal(s.cell.getAttribute('aria-describedby'), 'existing');
  s.fire('pointerover', s.cell); s.advance(3000); assert.equal(s.tooltip().hidden, true);
  s.fire('pointerout', s.cell); s.fire('pointerover', s.cell);
  s.advance(3000);
  assert.equal(s.tooltip().hidden, false);
});

test('pointer can cross the gap into the tooltip and leave to dismiss', () => {
  const s = setup(); s.context.mountComparisonHelp();
  s.fire('pointerover', s.cell);
  s.advance(3000);
  s.fire('pointerout', s.cell);
  s.fire('pointerover', s.tooltip()); s.flush();
  assert.equal(s.tooltip().hidden, false);
  s.fire('pointerout', s.tooltip()); s.flush();
  assert.equal(s.tooltip().hidden, true);
});

test('method hover help does not cover the pointer route to the next method row',()=>{
  const s=setup();s.context.window.innerWidth=1200;s.context.window.innerHeight=900;
  s.list.className='comparison-method-row';
  s.list.rect={left:20,right:316,top:100,bottom:136,width:296,height:36};
  s.cell.rect={left:20,right:220,top:100,bottom:136,width:200,height:36};
  s.context.mountComparisonHelp();
  s.tooltip().rect={left:0,top:0,right:260,bottom:160,width:260,height:160};
  s.fire('pointerover',s.cell);
  s.advance(3000);
  const tip=s.tooltip(),x=120,y=160,left=parseFloat(tip.style.left),top=parseFloat(tip.style.top);
  assert.equal(x>=left&&x<left+tip.rect.width&&y>=top&&y<top+tip.rect.height,false);
});

test('help appears beside the hovered element and follows that element on scroll',()=>{
  const s=setup();s.context.window.innerWidth=1200;s.context.window.innerHeight=900;
  s.cell.rect={left:20,right:220,top:100,bottom:136,width:200,height:36};
  s.context.mountComparisonHelp();s.fire('pointerover',s.cell);
  s.advance(3000);
  const tip=s.tooltip();
  assert.equal(parseFloat(tip.style.left),228);assert.equal(parseFloat(tip.style.top),100);
  s.play.rect={left:30,right:70,top:700,bottom:736,width:40,height:36};
  s.fire('focusin',s.play);
  assert.equal(parseFloat(tip.style.left),78);assert.equal(parseFloat(tip.style.top),700);
  s.play.rect.top=600;
  s.fire('scroll',s.body);
  assert.equal(parseFloat(tip.style.top),600);
});

test('hover help needs three continuous seconds and cancels when leaving early',()=>{
  const s=setup();s.context.mountComparisonHelp();s.fire('pointerover',s.cell);
  assert.equal(s.tooltip().hidden,true);
  s.advance(2999);assert.equal(s.tooltip().hidden,true);
  s.advance(1);assert.equal(s.tooltip().hidden,false);
  s.fire('pointerout',s.cell);s.advance(160);assert.equal(s.tooltip().hidden,true);
  s.fire('pointerover',s.cell);s.advance(2500);s.fire('pointerout',s.cell,{relatedTarget:s.body});
  s.advance(10000);assert.equal(s.tooltip().hidden,true);
});

test('preset help sits outside its selection group instead of covering adjacent presets',()=>{
  const s=setup();s.context.window.innerWidth=1200;s.context.window.innerHeight=900;
  s.list.className='comparison-noise-method';
  s.list.rect={left:20,right:316,top:80,bottom:160,width:296,height:80};
  s.cell.rect={left:20,right:112,top:100,bottom:136,width:92,height:36};
  s.context.mountComparisonHelp();s.fire('pointerover',s.cell);s.advance(3000);
  assert.equal(parseFloat(s.tooltip().style.left),324);
  assert.equal(parseFloat(s.tooltip().style.top),100);
});

test('moving to another field starts a fresh dwell instead of retaining the previous help',()=>{
  const s=setup();s.context.mountComparisonHelp();s.fire('pointerover',s.cell);s.advance(3000);
  s.fire('pointerout',s.cell,{relatedTarget:s.field});s.fire('pointerover',s.field,{relatedTarget:s.cell});
  assert.equal(s.tooltip().hidden,true);
  s.advance(2999);assert.equal(s.tooltip().hidden,true);
  s.advance(1);assert.equal(s.tooltip().hidden,false);assert.match(s.tooltip().textContent,/0% is input/);
});

test('moving within a control does not restart its dwell and Escape cancels pending help',()=>{
  const s=setup();s.context.mountComparisonHelp();const child=s.document.createElement('span');s.cell.append(child);
  s.fire('pointerover',s.cell);s.advance(2000);
  s.fire('pointerout',s.cell,{relatedTarget:child});s.fire('pointerover',child,{relatedTarget:s.cell});
  s.advance(1000);assert.equal(s.tooltip().hidden,false);
  s.fire('pointerout',s.cell);s.advance(160);s.fire('pointerover',s.cell);s.advance(1500);
  const escape=s.fire('keydown',s.cell,{key:'Escape'});
  assert.equal(escape.prevented,true);s.advance(3000);assert.equal(s.tooltip().hidden,true);
});

test('a control removed during the dwell cannot open stale help',()=>{
  const s=setup();s.context.mountComparisonHelp();s.fire('pointerover',s.cell);s.advance(2500);
  s.cell.isConnected=false;s.advance(500);assert.equal(s.tooltip().hidden,true);
});

test('touch info toggles independently; normal controls are not intercepted', () => {
  const s = setup(); s.context.mountComparisonHelp(); const info = s.info();
  s.fire('pointerover', info, {pointerType: 'touch'}); assert.equal(s.tooltip().hidden, true);
  const event = s.fire('click', info);
  assert.equal(event.prevented, true); assert.equal(event.stopped, true);
  assert.equal(info.getAttribute('aria-expanded'), 'true');
  s.fire('click', info); assert.equal(s.tooltip().hidden, true);
  assert.equal(info.getAttribute('aria-expanded'), 'false');
  s.fire('click', info); s.fire('click', s.body); assert.equal(s.tooltip().hidden, true);
  const normal = s.fire('click', s.cell);
  assert.equal(normal.prevented, undefined); assert.equal(normal.stopped, undefined);
  assert.equal(s.cell.getAttribute('aria-checked'), null);
});

test('position stays inside viewport and disconnected anchors clean up on scroll', () => {
  const s = setup(); s.context.mountComparisonHelp(); s.fire('focusin', s.play);
  const tip = s.tooltip();
  assert.equal(tip.parentElement, s.body);
  assert.ok(parseFloat(tip.style.left) >= 8);
  assert.ok(parseFloat(tip.style.left) + tip.rect.width <= 352);
  assert.ok(parseFloat(tip.style.top) + tip.rect.height <= 272);
  s.play.isConnected = false; s.fire('scroll', s.body);
  assert.equal(tip.hidden, true);
  assert.equal(s.play.getAttribute('aria-describedby'), null);
});

test('fonts are self-hosted and checksum matches the official source', () => {
  const crypto = require('node:crypto');
  const font = fs.readFileSync(path.join(__dirname, '../ai_remaster_gui/static/fonts/source-sans-3/SourceSans3-variable.ttf'));
  assert.equal(crypto.createHash('sha256').update(font).digest('hex'), '042fe2cc0b933e328410d7acbd0aa6a1873dca5aef81875f4bc214b08825c7b9');
  const css = fs.readFileSync(path.join(__dirname, '../ai_remaster_gui/static/comparison-help.css'), 'utf8');
  assert.match(css, /font-weight: 200 900/); assert.match(css, /font-display: swap/);
  assert.doesNotMatch(css, /https?:/);
});

test('modal help lives in the dialog top layer, not the inert page', () => {
  const s = setup(); s.context.mountComparisonHelp();
  const dialog = s.document.createElement('dialog'); dialog.open = true;
  s.body.append(dialog); dialog.append(s.list);
  s.fire('focusin', s.info());
  const tip = dialog.children.find(element => element.id === 'comparison-context-help');
  assert.ok(tip); assert.equal(tip.hidden, false);
  s.fire('focusin', s.play);
  assert.equal(s.tooltip(), tip);
});

test('every method has qualitative effort, compute path and cost drivers, never a quality rank',()=>{
 const context=vm.createContext({});vm.runInContext(source,context);
 const methods=vm.runInContext('comparisonHelpMethods',context),costs=vm.runInContext('comparisonEffort',context);
 assert.deepEqual(Object.keys(costs).sort(),Object.keys(methods).sort());
 for(const method of Object.keys(methods)) {
  assert.ok(['low','medium','high'].includes(costs[method][0]));assert.ok(costs[method][2].length>30);
  assert.match(context.comparisonMethodHelp(method),/not a benchmark or quality ranking/);
 }
 assert.equal(costs.nafnet[1],'CUDA GPU');assert.equal(costs.richardson_lucy[1],'CPU');
 assert.match(context.comparisonMethodHelp('multiscale'),/Two sequential/);
});

test('compact mounting decorates only method actions, effort badges and relevant fields without adding icons',()=>{
 const s=setup();s.cell.dataset.inspectMethod='cas';
 s.selections['[data-inspect-method], [data-effort-method]']=[s.cell];
 s.context.mountComparisonHelp(s.document,true);s.context.mountComparisonHelp(s.document,true);
 assert.equal(s.list.children[0],s.cell);assert.equal(s.list.children.length,1);
 s.fire('focusin',s.cell);assert.match(s.tooltip().textContent,/Effort: low · CPU/);
 s.fire('focusin',s.field);assert.match(s.tooltip().textContent,/0% is input/);
 const click=s.fire('click',s.cell);assert.equal(click.prevented,undefined);
});

test('keyboard focus can replace pinned help with the newly focused control',()=>{
 const s=setup();s.context.mountComparisonHelp();s.fire('click',s.info());
 s.fire('focusin',s.field);assert.match(s.tooltip().textContent,/0% is input/);
});
