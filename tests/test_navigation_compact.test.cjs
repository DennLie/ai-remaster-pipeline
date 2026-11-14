const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const html=fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/index.html'),'utf8');
const core=fs.readFileSync(path.join(__dirname,'../ai_remaster_gui/static/js/core.js'),'utf8');
test('one navigation row contains the original dynamic tab host, system details and explicit actions',()=>{
 const header=html.match(/<header class="workbench-navigation">[\s\S]*?<\/header>/)[0];
 assert.match(header,/<nav id="tabs" class="tabs" aria-label="Pipeline stages"[^>]*><\/nav>/);
 assert.match(header,/onfocusin="[^\"]*scrollIntoView/);
 for(const id of ['tabs','workbenchRunStatus','version','root','systemStatus']) assert.equal(html.split('id="'+id+'"').length,2);
 assert.match(header,/onclick="stopRun\(\)" disabled/);
 assert.match(header,/<details class="workbench-system">[\s\S]*onclick="refresh\(true\)"[\s\S]*onclick="quitArp\(\)"/);
 assert.match(header,/buymeacoffee/);
 assert.doesNotMatch(html.slice(html.indexOf('</header>')+9),/<nav id="tabs"/);
});
test('sticky offsets count a contained tab row once and retain separate-row compatibility',()=>{
 const styles=new Map(),tabs={getBoundingClientRect:()=>({height:51})};
 let combined=true;
 const header={getBoundingClientRect:()=>({height:48}),contains:element=>combined&&element===tabs};
 const context=vm.createContext({window:{addEventListener(){}},document:{querySelector:()=>header,getElementById:()=>tabs,documentElement:{style:{setProperty:(name,value)=>styles.set(name,value)}}}});
 vm.runInContext(core,context);
 context.updateStickyOffsets();assert.equal(styles.get('--header-height'),'48px');assert.equal(styles.get('--workbench-chrome-height'),'48px');
 combined=false;context.updateStickyOffsets();assert.equal(styles.get('--workbench-chrome-height'),'99px');
});
