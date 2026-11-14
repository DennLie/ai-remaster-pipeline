const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
function setup() {
 const stored=new Map(),inspector={},analysis={after(){}},buttons=['review','lab'].map(profile=>({dataset:{profile},setAttribute(key,value){this[key]=value}}));
 const page={dataset:{},querySelector:selector=>selector.includes('inspector')?inspector:analysis,querySelectorAll:()=>buttons};
 const state={root:'repo',project_path:'film.arp',settings:{method:'cas'}};
 const context=vm.createContext({state,innerWidth:1600,location:{pathname:'/'},localStorage:{getItem:key=>stored.get(key),setItem:(key,value)=>stored.set(key,value)},document:{querySelector:()=>page},window:{dispatchEvent(){}},Event:class {}});
 vm.runInContext(fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/comparison-layout.js'),'utf8'),context);
 return {context,state,page,stored,inspector,analysis,buttons};
}
test('Review is default; profiles persist independently of processing settings',()=>{
 const {context,state,page,analysis,buttons}=setup(),before=JSON.stringify(state);
 assert.equal(context.comparisonProfile(),'review');context.setComparisonProfile('lab');
 assert.equal(page.dataset.profile,'lab');assert.equal(analysis.open,true);assert.equal(buttons[1]['aria-pressed'],'true');
 assert.equal(context.comparisonProfile(),'lab');assert.equal(JSON.stringify(state),before);
 context.setComparisonProfile('review');assert.equal(analysis.open,false);
});
test('profile storage is project-scoped and unavailable storage has a safe fallback',()=>{
 const {context,state}=setup();context.setComparisonProfile('lab');state.project_path='other.arp';assert.equal(context.comparisonProfile(),'review');
 context.localStorage.getItem=()=>{throw Error('blocked')};assert.equal(context.comparisonProfile(),'review');
});
test('profile changes retain the mobile inspector disclosure without renderer or adoption calls',()=>{
 const {context,inspector}=setup();context.innerWidth=480;
 context.setComparisonProfile('review');assert.equal(inspector.open,false);
 inspector.open=true;context.setComparisonProfile('lab');assert.equal(inspector.open,true);
 context.setComparisonProfile('review');assert.equal(inspector.open,true);
});
test('unavailable browser storage still switches the current profile and keeps it project-scoped',()=>{
 const {context,state,page}=setup();
 context.localStorage.getItem=context.localStorage.setItem=()=>{throw Error('blocked')};
 context.setComparisonProfile('lab');assert.equal(context.comparisonProfile(),'lab');assert.equal(page.dataset.profile,'lab');
 state.project_path='other.arp';assert.equal(context.comparisonProfile(),'review');
 state.project_path='film.arp';assert.equal(context.comparisonProfile(),'lab');
});
test('the shared play bar keeps its 57px height and scrolls instead of wrapping controls',()=>{
 const css=fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/comparison-workbench.css'),'utf8');
 const rule=css.match(/\.comparison-transport-strip \{([^}]+)\}/)[1];
 for(const property of ['height:57px','min-height:57px','max-height:57px','flex-wrap:nowrap','overflow-x:auto','position:fixed','top:auto','bottom:env(safe-area-inset-bottom,0px)']) assert.ok(rule.includes(property),property);
 assert.match(css,/#app:has\(\.comparison-workbench\) \{ padding-bottom:calc\(100px \+ env\(safe-area-inset-bottom,0px\)\)/);
 assert.doesNotMatch(css,/\.comparison-transport-strip \.denoise-compare-position \{[^}]*flex-basis:100%/);
});
test('input context keeps its current label and moves blocking warnings outside the collapsed details',()=>{
 const {context,state}=setup(),label={textContent:''};
 let nested=null,visible=null,removed=0;
 const input={querySelector:()=>nested};
 const page={querySelector:selector=>selector==='[data-processing-input]'?input:selector==='[data-processing-input-label]'?label:
   selector==='[data-comparison-input-warning]'?visible:{before(warning){visible=warning;nested=null}}};
 context.document.querySelector=()=>page;
 const warning=()=>({dataset:{},remove(){visible=null;removed++}});
 nested=warning();const first=nested;state.processing_input={label:'Upstream <frame>'};
 context.updateComparisonInputContext();
 assert.equal(label.textContent,'Upstream <frame>');assert.equal(visible,first);
 assert.equal(visible.dataset.comparisonInputWarning,'');assert.equal(nested,null);
 nested=warning();const next=nested;context.updateComparisonInputContext();
 assert.equal(removed,1);assert.equal(visible,next);
 state.processing_input={};context.updateComparisonInputContext();
 assert.equal(visible,null);assert.equal(label.textContent,'Selected processing input');
});
