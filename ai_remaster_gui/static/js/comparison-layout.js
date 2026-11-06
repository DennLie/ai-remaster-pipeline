/* Display-only preference: excluded from processing settings and signatures. */
const comparisonProfiles=new Map();
function comparisonProfileKey() { return 'arp.comparison.profile:'+String(state?.project_path || state?.root || location.pathname); }
function comparisonProfile() {
  const key=comparisonProfileKey();
  if(comparisonProfiles.has(key)) return comparisonProfiles.get(key);
  try { return localStorage.getItem(key)==='lab' ? 'lab' : 'review'; }
  catch (_) { return 'review'; }
}
function setComparisonProfile(profile) {
  if(!['review','lab'].includes(profile)) return;
  comparisonProfiles.set(comparisonProfileKey(),profile);
  try { localStorage.setItem(comparisonProfileKey(),profile); } catch (_) {}
  const page=document.querySelector('.comparison-workbench');
  if(!page) return;
  const previous=page.dataset.profile;
  page.dataset.profile=profile;
  page.querySelectorAll('[data-profile]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.profile===profile)));
  if(!previous) page.querySelector('[data-comparison-inspector]').open=innerWidth>1100;
  page.querySelector('[data-comparison-analysis]').open=profile==='lab';
  const analysis=page.querySelector('[data-comparison-analysis]');
  if(profile==='lab') page.querySelector('.comparison-transport-strip').after(analysis);
  else page.querySelector('#denoiseCompareNotice').after(analysis);
  window.dispatchEvent(new Event('resize'));
  updateComparisonInspector();
  if(typeof denoiseWorkspaceRefresh==='function') denoiseWorkspaceRefresh();
}
function updateComparisonInputContext() {
  const page=document.querySelector('.comparison-workbench'),input=page?.querySelector('[data-processing-input]');
  if(!input) return;
  const label=page.querySelector('[data-processing-input-label]');
  if(label) label.textContent=state.processing_input?.label || 'Selected processing input';
  const warning=input.querySelector('.inline-warning');
  page.querySelector('[data-comparison-input-warning]')?.remove();
  if(warning) {
    warning.dataset.comparisonInputWarning='';
    page.querySelector('#denoiseComparePlayers').before(warning);
  }
}
function showComparisonInspector() {
  const inspector=document.querySelector('[data-comparison-inspector]');
  if(innerWidth<=1100) {
    const page=document.querySelector('.comparison-workbench'),sidebar=page.querySelector('.denoise-settings-sidebar');
    let dialog=page.querySelector('.comparison-inspector-dialog');
    if(!dialog) {
      dialog=document.createElement('dialog');dialog.className='comparison-inspector-dialog';
      dialog.setAttribute('aria-label','Variants and parameters');
      dialog.innerHTML='<form method="dialog"><button type="submit" autofocus>Return to comparison</button></form>';
      dialog.addEventListener('close',()=>page.append(sidebar));page.append(dialog);
    }
    dialog.append(sidebar);inspector.open=true;dialog.showModal();return;
  }
  inspector.open=true;inspector.scrollIntoView({block:'nearest'});
  inspector.querySelector('summary').focus();
}
function inspectComparisonVariant(method,preset) {
  const variants=denoiseWorkspaceSnapshot().variants;
  const index=variants.findIndex(value=>value.method===method&&value.preset===preset);
  if(index<0) return;
  if(comparisonProfile()==='lab' && !denoiseWorkspaceHas(method,preset)) denoiseWorkspaceToggle(method,preset);
  denoiseWorkspaceFocus(index+1);
  const page=document.querySelector('.comparison-workbench');
  if(page?.dataset.profile==='lab') page.querySelector(`[data-dc-slot="${index+1}"]`)?.scrollIntoView({block:'nearest'});
}
function comparisonEffortButton(method) {
  const effort=comparisonEffort[method];
  const button=document.createElement('button');button.type='button';
  button.className='comparison-help-info comparison-effort';button.dataset.effortMethod=method;button.dataset.effort=effort[0];
  button.textContent='Effort '+effort[0];button.setAttribute('aria-label',`${method}: ${effort[0]} processing effort. Show explanation`);
  button.setAttribute('aria-expanded','false');return button;
}
function updateComparisonInspector() {
  const page=document.querySelector('.comparison-workbench');
  if(!page || typeof denoiseSelectedVariant!=='function') return;
  page.dataset.jobRunning=String(Boolean(state.running));
  page.dataset.inputBlocked=String(Boolean(state.processing_input?.problem));
  const selected=denoiseSelectedVariant(),pipeline=denoiseStageVariant();
  page.dataset.inspectionAdopted=String(denoiseSameVariant(selected,pipeline));
  page.dataset.inspectionReady=String(selected.method!=='off' && denoiseVariantReady(selected.method,selected.preset));
  page.querySelectorAll('[data-inspect-method]').forEach(button=>{
    const value={method:button.dataset.inspectMethod,preset:button.dataset.inspectPreset};
    button.disabled=false;
    button.setAttribute('aria-pressed',String(denoiseSameVariant(value,selected)));
  });
  page.querySelectorAll('.comparison-method-row').forEach(row=>{
    const cell=row.querySelector('[data-denoise-cell]');
    if(!cell) return;
    const member=denoiseWorkspaceHas(cell.dataset.method,cell.dataset.preset);
    row.classList.toggle('is-pipeline',denoiseSameVariant({method:cell.dataset.method,preset:cell.dataset.preset},pipeline));
    row.classList.toggle('is-viewed',denoiseSameVariant({method:cell.dataset.method,preset:cell.dataset.preset},selected));
    const viewed=row.querySelector('[data-method-viewed]');
    if(viewed) {
      viewed.hidden=!denoiseSameVariant({method:cell.dataset.method,preset:cell.dataset.preset},selected);
      viewed.textContent=page.dataset.profile==='lab' ? 'Inspecting' : 'In view';
    }
    const label=cell.querySelector('[data-comparison-membership]');
    if(label) label.textContent=comparisonStageKey()==='denoise' ? (member ? 'In lab' : 'Compare') : (member ? 'In lab comparison' : 'Compare in lab');
  });
  const helpText=page.querySelector('[data-method-help]');
  if(helpText && typeof comparisonHelpMethods!=='undefined') {
    const text=comparisonMethodHelp(selected.method) || 'Inspect a method to read its scope and limitations here.';
    if(helpText.textContent!==text) helpText.textContent=text;
  }
  const warning=page.querySelector('[data-method-warning]');
  if(warning) {
    warning.textContent=selected.method.startsWith('localized_') ? 'Experimental. Confirm the scope below; review cuts, fades and the full output.'
      : ['nafnet','restormer'].includes(selected.method) ? 'GoPro-trained image model · CUDA/weights required. Inspect grain and temporal consistency; commercial guard restrictions apply.'
      : selected.method==='richardson_lucy' ? 'Assumed Gaussian blur, not a measured film PSF. Check ringing and grain.' : '';
    warning.hidden=!warning.textContent;
  }
  page.querySelectorAll('.comparison-param-row[data-methods]').forEach(row=>{
    row.hidden=!row.dataset.methods.split(',').some(method=>selected.method===method || page.dataset.profile==='lab' && denoiseWorkspaceHas(method,'settings'));
  });
  const preset=page.querySelector('.comparison-param-row:has([data-field="preset"])');
  if(preset && comparisonStageKey()==='enhancement') preset.hidden=['nafnet','restormer'].includes(selected.method);
  const summary=page.querySelector('[data-review-context]');
  if(summary) summary.textContent=`Inspecting: ${denoiseVariantLabelFor(selected.method,selected.preset)} · Pipeline: ${denoiseVariantLabelFor(pipeline.method,pipeline.preset)}`;
  page.querySelectorAll('#denoiseComparePlayers [data-dc-slot]').forEach(panel=>{
    const slot=Number(panel.dataset.dcSlot),variant=denoiseWorkspaceSnapshot().variants[slot-1];
    const hidden=slot>0 && (page.dataset.profile==='review' ? !denoiseSameVariant(variant,selected) : !denoiseWorkspaceHas(variant.method,variant.preset));
    panel.hidden=hidden;
    if(hidden) panel.dataset.visible='false';
  });
  if(comparisonStageKey()==='dust' && typeof dustUpdateInspectionContext==='function') dustUpdateInspectionContext();
}
function mountComparisonLayout(title) {
  const page=document.querySelector('.denoise-editor-page');
  page.classList.add('comparison-workbench');
  const bar=document.createElement('div');bar.className='comparison-workbench-title';
  bar.innerHTML=`<div><h1>${esc(title)}</h1><span>Compare the current stage input with selected results</span></div>
    <button type="button" class="comparison-inspector-shortcut" onclick="showComparisonInspector()">Variants & settings</button>
    <div class="comparison-profile" role="group" aria-label="Workspace profile">
      <button type="button" data-profile="review" aria-pressed="false" onclick="setComparisonProfile('review')">Review</button>
      <button type="button" data-profile="lab" aria-pressed="false" onclick="setComparisonProfile('lab')">Labor</button>
    </div>`;
  page.prepend(bar);
  const sidebar=page.querySelector('.denoise-settings-sidebar');
  const inspector=document.createElement('details');inspector.dataset.comparisonInspector='';
  const summary=document.createElement('summary');summary.textContent='Methods & settings';
  const content=document.createElement('div');content.className='comparison-inspector-content';
  while(sidebar.firstChild) content.append(sidebar.firstChild);
  inspector.append(summary,content);sidebar.append(inspector);
  content.classList.add('compact-inspector');
  const help=content.querySelector('.denoise-variant-help');
  if(help) help.remove();
  const methodHeading=content.querySelector('#denoiseVariantHeading');
  if(methodHeading) methodHeading.textContent='Methods';
  const footer=document.createElement('div');footer.className='comparison-inspector-footer';
  const selectedPanel=content.querySelector('[data-denoise-selected]');
  const generation=content.querySelector('[data-dm-generation]');
  if(generation) footer.append(generation);
  if(selectedPanel) {
    const details=document.createElement('details');details.className='comparison-result-details';
    details.innerHTML='<summary>More actions & details</summary>';
    for(const selector of ['[data-dm-selected-meta]','[data-dm-selected-output]','#denoiseCopyVariant']) {
      const node=selectedPanel.querySelector(selector);if(node) details.append(node);
    }
    selectedPanel.append(details);footer.append(selectedPanel);
    const bypass=content.querySelector('.denoise-off-option');
    if(bypass) details.append(bypass);
  }
  sidebar.append(footer);
  const jobActions=content.querySelector(':scope > .actions');
  if(jobActions) {
    jobActions.querySelector('button').textContent='Render pipeline selection';
    const stop=jobActions.querySelector('[onclick="stopRun()"]');
    if(stop) {stop.dataset.compactStop='';footer.prepend(stop);}
    footer.querySelector('.comparison-result-details').append(jobActions);
  }
  const jobProgress=content.querySelector(':scope > .phase-progress');
  if(jobProgress) footer.prepend(jobProgress);
  const advanced=document.createElement('details');advanced.className='comparison-advanced';advanced.innerHTML='<summary>Job & technical details</summary>';
  for(const node of [...content.children]) {
    if(node.matches('h2,p,[data-dm-expected],.actions,.command-details') || node.querySelector?.('[data-field="force"]')) advanced.append(node);
  }
  footer.querySelector('.comparison-result-details').append(advanced);
  content.querySelectorAll('.denoise-variant-picker > p').forEach(node=>advanced.append(node));
  content.querySelectorAll('[data-denoise-cell]').forEach(cell=>{
    const row=document.createElement('div');row.className='comparison-method-row';cell.before(row);row.append(cell);
    row.addEventListener('click',event=>{if(!event.target.closest('button,a,input,summary')) inspectComparisonVariant(cell.dataset.method,cell.dataset.preset);});
    const button=document.createElement('button');button.type='button';button.className='comparison-inspect-method';
    button.dataset.inspectMethod=cell.dataset.method;button.dataset.inspectPreset=cell.dataset.preset;
    button.textContent=comparisonStageKey()==='denoise' ? cell.dataset.preset[0].toUpperCase()+cell.dataset.preset.slice(1) : 'Inspect';button.setAttribute('aria-label','Inspect '+cell.getAttribute('aria-label'));
    button.addEventListener('click',()=>inspectComparisonVariant(cell.dataset.method,cell.dataset.preset));row.append(button);
    if(comparisonStageKey()!=='denoise') {
      button.textContent=cell.querySelector('strong')?.textContent || cell.dataset.method;
      cell.querySelector('strong')?.remove();
      row.prepend(button);
      if(comparisonStageKey()!=='dust') row.append(comparisonEffortButton(cell.dataset.method));
      const status=cell.querySelector('.denoise-variant-cell-state'),badge=cell.querySelector('.denoise-variant-pipeline');
      if(status) row.append(status);if(badge) row.append(badge);
    }
    cell.querySelector('.denoise-selection-check')?.remove();
    const membership=document.createElement('span');membership.dataset.comparisonMembership='';cell.prepend(membership);
    if(comparisonStageKey()==='denoise') {
      for(const selector of ['.denoise-variant-cell-state','.denoise-variant-pipeline']) { const node=cell.querySelector(selector);if(node) row.append(node); }
    }
    const viewed=document.createElement('small');viewed.dataset.methodViewed='';viewed.textContent='In view';viewed.hidden=true;row.append(viewed);
    const status=document.createElement('div');status.className='comparison-method-status';
    for(const selector of ['.denoise-variant-cell-state','[data-denoise-cell]','.denoise-variant-pipeline','[data-method-viewed]']) {
      const node=row.querySelector(selector);if(node) status.append(node);
    }
    row.append(status);
  });
  if(comparisonStageKey()==='denoise') {
    const grid=content.querySelector('.denoise-variant-grid');
    const rows=[...grid.querySelectorAll('.comparison-method-row')];
    grid.replaceChildren();grid.classList.add('comparison-noise-methods');
    for(const method of DENOISE_METHODS) {
      const group=document.createElement('section');group.className='comparison-noise-method';
      const heading=document.createElement('div');heading.className='comparison-noise-heading';
      const name=document.createElement('strong');name.textContent=method.label;
      heading.append(name,comparisonEffortButton(method.key));group.append(heading);
      const variants=document.createElement('div');variants.className='comparison-noise-presets';
      rows.filter(row=>row.querySelector('[data-denoise-cell]').dataset.method===method.key).forEach(row=>variants.append(row));
      group.append(variants);grid.append(group);
    }
    const batch=content.querySelector('.denoise-batch-export');
    if(batch) footer.querySelector('.comparison-result-details').append(batch);
  }
  content.querySelectorAll('.deflicker-parameters').forEach(parameters=>{
    parameters.querySelectorAll('[data-field]').forEach(field=>{
      const enclosing=field.closest('label');
      if(enclosing && comparisonStageKey()==='deflicker') {
        const owners={localized_confirm_single_shot:'localized_research',localized_confirm_film_experiment:'localized_stream'};
        if(owners[field.dataset.field]) {
          const row=document.createElement('div');row.className='comparison-param-row comparison-param-confirmation';row.dataset.methods=owners[field.dataset.field];enclosing.before(row);row.append(enclosing);
        }
        return;
      }
      const label=field.previousElementSibling;
      if(label?.tagName!=='LABEL') return;
      if(!field.id) field.id='comparison-'+comparisonStageKey()+'-'+field.dataset.field;
      label.htmlFor=field.id;
      const row=document.createElement('div');row.className='comparison-param-row';label.before(row);row.append(label,field);
      if(comparisonStageKey()==='deflicker') {
        const owners={window:'ffmpeg_median,ffmpeg_am',strength:'arp',max_correction:'arp',localized_confirm_single_shot:'localized_research',localized_confirm_film_experiment:'localized_stream'};
        if(owners[field.dataset.field]) row.dataset.methods=owners[field.dataset.field];
      }
    });
    const guidance=document.createElement('details');guidance.className='comparison-method-guidance comparison-method-help';guidance.innerHTML='<summary>About this method & limitations</summary><p data-method-help></p>';
    parameters.querySelectorAll('p:not([data-method-help]), .field-help').forEach(node=>guidance.append(node));parameters.append(guidance);
    if(comparisonStageKey()==='enhancement') parameters.querySelector('h3')?.insertAdjacentHTML('afterend','<p class="comparison-parameter-scope">Preset: classical methods only. Mix: all methods.</p>');
  });
  const heading=page.querySelector('.denoise-compare-heading');
  const methodHelp=document.createElement('details');methodHelp.className='comparison-method-help';
  methodHelp.innerHTML='<summary>About this method</summary><p data-method-help></p>';
  if(!content.querySelector('[data-method-help]')) content.append(methodHelp);
  const warning=document.createElement('p');warning.dataset.methodWarning='';warning.className='inline-warning';warning.hidden=true;
  content.querySelector('.deflicker-parameters')?.prepend(warning);
  const guidance=content.querySelector('.comparison-method-help');
  const fieldHelp=document.createElement('dl');fieldHelp.className='comparison-field-help';
  content.querySelectorAll('[data-field]').forEach(field=>{
    const text=comparisonHelpFields[field.dataset.field];if(field.type==='hidden' || !text) return;
    const term=document.createElement('dt');term.textContent=field.dataset.field.replaceAll('_',' ');
    const definition=document.createElement('dd');definition.textContent=text;fieldHelp.append(term,definition);
  });
  guidance.append(fieldHelp);
  const methodGuidance=guidance;
  footer.querySelector('.comparison-result-details').before(methodGuidance);
  page.querySelectorAll('[data-field]').forEach(field=>{
    field.removeAttribute('title');field.closest('label')?.removeAttribute('title');
    if(field.previousElementSibling?.tagName==='LABEL') field.previousElementSibling.removeAttribute('title');
  });
  const reference=document.createElement('div');reference.className='comparison-reference';
  reference.setAttribute('role','group');reference.setAttribute('aria-label','Comparison reference');
  reference.innerHTML='<span class="sr-only">Compare against</span><button type="button" data-reference="input" onclick="denoiseWorkspaceFor().setReference(\'input\')">Stage input</button><button type="button" data-reference="original" onclick="denoiseWorkspaceFor().setReference(\'original\')">Original source</button><small>Display only · processing input stays unchanged</small>';
  heading.after(reference);
  const transport=document.createElement('div');transport.className='comparison-transport-strip';
  transport.tabIndex=0;transport.setAttribute('role','group');
  transport.setAttribute('aria-label','Comparison playback and display controls; scroll horizontally on narrow windows');
  transport.addEventListener('focusin',event=>event.target.scrollIntoView({block:'nearest',inline:'nearest'}));
  transport.append(heading.querySelector('.denoise-compare-transport'),page.querySelector('.denoise-compare-position'));
  page.querySelector('#denoiseComparePlayers').before(transport);
  const context=document.createElement('p');context.dataset.reviewContext='';context.className='comparison-review-context';transport.after(context);
  const inputContext=document.createElement('details');inputContext.className='comparison-input-context';
  inputContext.innerHTML='<summary>Processing input &amp; provenance · <span data-processing-input-label></span></summary>';
  inputContext.append(page.querySelector('[data-processing-input]'));
  page.querySelector('#denoiseComparePlayers').after(inputContext);
  if(comparisonStageKey()==='stabilize') {
    const warning=document.createElement('p');warning.className='inline-warning';
    warning.textContent='0% means no additional fixed zoom, not complete picture preservation. Translation, rotation and bicubic resampling still change pixels and can expose black borders or clip source content. 3% does not guarantee border-free output. Difference shows geometry changes, not restoration quality.';
    page.querySelector('#denoiseComparePlayers').before(warning);
    const report=document.createElement('details');report.className='comparison-input-context';
    report.innerHTML='<summary data-stabilization-scene-summary>Stabilization scene report</summary><div data-stabilization-scenes role="status"></div>';
    report.insertAdjacentHTML('beforeend','<section data-stabilization-footprint><h3>Geometric footprint</h3><p>Pixel-centre raster estimate, independent of film brightness. This is not bicubic kernel support or a restoration-quality score.</p><button type="button" onclick="inspectStabilizationFootprint()">Inspect footprint at current frame</button><output data-footprint-status role="status"></output><img data-footprint-image hidden alt="Geometric output footprint: white is source-supported, black is outside the source" style="max-width:480px;width:100%;height:auto"></section>');
    inputContext.before(report);
  }
  updateComparisonInputContext();
  const analysis=document.createElement('details');analysis.dataset.comparisonAnalysis='';analysis.className='comparison-analysis';
  const caption=document.createElement('summary');caption.textContent='Frame analysis & display options';
  const chartTitle=page.querySelector('.denoise-compact-rail-heading strong');
  if(chartTitle) chartTitle.textContent='Frame changes vs stage input';
  analysis.append(caption,page.querySelector('.denoise-compact-rail'),page.querySelector('.denoise-preview-settings'));
  const zoomHint=page.querySelector('.review-zoom-hint');
  zoomHint.textContent='Alt + wheel: zoom · Drag: pan · Wheel: scroll page';heading.after(zoomHint);
  heading.classList.add('comparison-display-controls');
  heading.setAttribute('role','group');heading.setAttribute('aria-label','Comparison display controls');
  heading.querySelector('h2').classList.add('sr-only');
  transport.append(heading,heading.querySelector('[data-ds-quality]'));
  page.querySelector('#denoiseCompareNotice').after(analysis);
  for(const step of [-10,-1,1,10]) {
    const button=transport.querySelector(`[onclick="denoiseCompareStep(${step})"]`);
    button.insertAdjacentHTML('beforeend',`<span>${step>0?'+':''}${step}</span>`);
  }
  setComparisonProfile(comparisonProfile());
  updateComparisonInspector();
  mountComparisonHelp(sidebar,true);
}
