const OVERVIEW_WORKFLOW_STAGE_KEYS = {
  globalDeflicker:['deflicker'],globalEnhancement:['enhancement'],globalDenoise:['denoise'],
  globalCleanup:['cleanup'],globalStabilize:['stabilize'],globalDust:['dust'],globalDetail:['detail'],
  globalExpandOutpaint:['outpaint'],globalColorize:['shots','references','colour'],
  globalSoundtrack:['audio'],globalUpscale:['upscale'],
};

function drawGlobal() {
  const global = state.settings.global || {};
  const source = global.source || '';
  const commercialGuard = global.commercial_model_guard === 'true';
  const denoise = global.denoise === 'true';
  const deflicker = global.deflicker === 'true';
  const cleanup = global.cleanup === 'true';
  const stabilize = global.stabilize === 'true';
  const detail = global.detail === 'true';
  const expandOutpaint = global.expand_outpaint !== 'false';
  const colorize = global.colorize !== 'false';
  const upscale = global.upscale === 'true';
  const soundtrack = global.add_soundtrack === 'true';
  const noProcessing = !denoise && !deflicker && global.enhancement !== 'true' && !cleanup && !stabilize && global.dust !== 'true' && !detail && !expandOutpaint && !colorize && !upscale && !soundtrack;
  const analysis = state.source_analysis || {};
  const sourceTone = source && !analysis.ready
    ? (analysis.message || 'Analyzing source material')
    : (state.source_monochrome ? 'Source looks black and white' : 'Source appears to contain colour');

  document.getElementById('app').innerHTML = `
    <section class="card">
        <div class="global-top project-workspace-header">
          <div class="project-intro"><img class="project-mark" src="/media?path=assets/branding/arp-app-icon-192.png" alt=""><h1>Projects</h1><p>Save this restoration as a new project, or open a saved project to continue working.</p></div>
          <div class="project-buttons" role="group" aria-label="Project actions">
            <button type="button" onclick="saveProjectAs()">Create project…</button>
            <button type="button" onclick="loadProject()">Open project…</button>
            <button type="button" onclick="saveProject()">Save</button>
            <button type="button" onclick="clearOverview()">Reset settings…</button>
          </div>
        </div>
      ${overviewSourcePicker(source, global)}
      ${sourceAnalysisHtml(analysis)}
      ${overviewSectionPicker(global, source, state.source_playback || source)}
      <div class="overview-workflow-layout workflow-tiles">
      <div class="workflow-paired-head">
        <div><h2>Processing stages</h2><p>Top to bottom. Off stages are skipped. Arrows reorder the first three stages only.</p></div>
        <section class="overview-run-panel" aria-labelledby="overviewRunTitle">
          <h2 id="overviewRunTitle">Whole remaster</h2>
          ${overviewActions()}
        </section>
      </div>
      ${workflowPickerHtml(denoise, cleanup, stabilize, detail, expandOutpaint, colorize, upscale, soundtrack, sourceTone)}
      <div id="overviewProgressTable">${overviewProgressTable()}</div>
      <p class="overview-rerun-note">Safe to rerun: completed work is reused when its effective inputs are unchanged. Changed settings, source files, masks, prompts, guides, or upstream outputs invalidate only the affected phase and its downstream results. A phase with Force enabled always reruns.</p>
      <div class="inline-warning">
        <label for="globalCommercialModelGuard">
          <input id="globalCommercialModelGuard" type="checkbox" ${commercialGuard ? 'checked' : ''}>
          <strong>Commercial model guard (strict, opt-in)</strong>
        </label>
        Blocks non-commercial and unaudited AI model bundles before they run. This does not certify the footage, other software or output files.
        See <a href="https://github.com/DennLie/ai-remaster-pipeline/blob/feature/commercial-safe-foundation/docs/commercial-model-guard.md" target="_blank" rel="noopener">licensing notes</a>.
      </div>
      ${noProcessing ? '<div class="inline-warning"><strong>No processing stages selected.</strong> Choose at least one workflow step, otherwise Run Whole Remaster has nothing to do.</div>' : ''}
      </div>
    </section>
  `;

  const workflow=document.querySelector('[data-preprocessing-stages]') || document.querySelector('.workflow-list');
  const ids={denoise:'globalDenoise',deflicker:'globalDeflicker',enhancement:'globalEnhancement',cleanup:'globalCleanup',stabilize:'globalStabilize',dust:'globalDust',detail:'globalDetail',outpaint:'globalExpandOutpaint',colour:'globalColorize',audio:'globalSoundtrack',upscale:'globalUpscale'};
  for(const key of pipelineOrder().slice().reverse()) {const tile=document.getElementById(ids[key])?.closest('.workflow-stage-tile');if(tile) workflow.prepend(tile);}
  document.getElementById('globalSource').addEventListener('change', saveGlobal);
  document.getElementById('globalCommercialModelGuard').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalDenoise').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalDeflicker').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalEnhancement').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalCleanup').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalStabilize').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalDust').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalDetail').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalExpandOutpaint').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalColorize').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalUpscale').addEventListener('change', saveGlobalPipelineOptions);
  document.getElementById('globalSoundtrack').addEventListener('change', saveGlobalPipelineOptions);
  bindOverviewSectionControls();
}

function workflowPickerHtml(denoise, cleanup, stabilize, detail, expandOutpaint, colorize, upscale, soundtrack, sourceTone) {
  return `
    <div class="workflow-list workflow-tiles">
      <p>Reorder phases using arrows or tab handles. Shot Detection, References and Recomposition have their own tab handles. This changes actual inputs; unmatched video/artifact bindings block processing rather than substituting old results.</p>
      ${state.pipeline_plan?.problems?.length ? `<p class="inline-warning" role="alert">${esc(state.pipeline_plan.problems.join(' · '))}</p>` : ''}
      <ol class="workflow-sequence-list" data-preprocessing-stages>
      ${workflowOptionHtml({
        id: 'globalDenoise',
        checked: denoise,
        icon: 'cleanup',
        title: 'Remove Noise',
        body: 'Optional independent denoising with Off, hqdn3d, NLMeans or BM3D at your chosen video-pipeline position. Retain different versions for comparison.',
      })}
      ${workflowOptionHtml({
        id: 'globalDeflicker',
        checked: state.settings.global.deflicker === 'true',
        icon: 'cleanup',
        title: 'Light Deflicker',
        body: 'Optional model-free brightness correction at your chosen video-pipeline position. Compare ARP, FFmpeg and experimental Localized methods without changing the pipeline selection.',
      })}
      ${workflowOptionHtml({
        id: 'globalEnhancement',
        checked: state.settings.global.enhancement === 'true',
        icon: 'cleanup',
        title: 'Detail Enhancement / Deblur',
        body: 'Optional native 1× enhancement in your chosen preprocessing order. Compare classical sharpening, Richardson–Lucy, NAFNet and Restormer before explicitly adopting an output.',
      })}
      ${workflowOptionHtml({
        id: 'globalCleanup',
        checked: cleanup,
        icon: 'cleanup',
        title: 'Clean Up',
        body: 'Run optional DeVignette and masked ProPainter AI DeScratch, then optionally use the LTX 2.3 Dearchive LoRA before any framing or colour work. Output keeps the source frame rate and resolution.',
      })}
      ${workflowOptionHtml({
        id: 'globalStabilize',
        checked: stabilize,
        icon: 'stabilize',
        title: 'Stabilize',
        body: 'Remove gate weave, frame jitter, and unwanted rotation per shot before outpainting and colour propagation. Shot cuts reset motion tracking.',
      })}
      ${workflowOptionHtml({
        id: 'globalDust',
        checked: state.settings.global.dust === 'true',
        icon: 'cleanup',
        title: 'Dust & Scratches',
        body: 'Inspect defect proposals at the chosen video-pipeline position, compare Telea, ProPainter or experimental DART, then explicitly adopt a mask-gated repair. Off by default.',
      })}
      ${workflowOptionHtml({
        id: 'globalDetail',
        checked: detail,
        icon: 'cleanup',
        title: 'Detail Enhancement',
        body: 'Optional legacy 1x image sharpening at your chosen video-pipeline position. Choose Off, Unsharp, CAS or Multiscale, strength and original-image mix.',
      })}
      ${workflowOptionHtml({
        id: 'globalExpandOutpaint',
        checked: expandOutpaint,
        icon: 'wide',
        title: 'Outpainting',
        body: 'Extend the frame to the chosen aspect ratio before the rest of the pipeline runs. Skip this when the source is already framed the way you want.',
      })}
      ${workflowOptionHtml({
        id: 'globalColorize',
        checked: colorize,
        icon: 'colour',
        title: 'Colorize',
        body: 'Detect shots, make reference frames, and colorize the video before recomposition.',
        note: sourceTone,
      })}
      ${workflowOptionHtml({
        id: 'globalSoundtrack',
        checked: soundtrack,
        icon: 'audio',
        title: 'Create Audio Track',
        body: 'Generate a musical score and/or synchronized sound effects for a silent film, after recomposition and before upscaling. Defaults off when the source already has audio.',
      })}
      ${workflowOptionHtml({
        id: 'globalUpscale',
        checked: upscale,
        icon: 'upscale',
        title: 'Upscale',
        body: 'Run the selected AI video upscaler after recomposition, or directly on the selected source section if no earlier processing is selected.',
      })}
      </ol>
    </div>
  `;
}

function workflowOptionHtml(item) {
  const orderKey=({globalDenoise:'denoise',globalDeflicker:'deflicker',globalEnhancement:'enhancement',globalCleanup:'cleanup',globalStabilize:'stabilize',globalDust:'dust',globalDetail:'detail',globalExpandOutpaint:'outpaint',globalColorize:'colour',globalSoundtrack:'audio',globalUpscale:'upscale'})[item.id];
  const summaries={
    globalDeflicker:'Reduce brightness fluctuations.',
    globalDenoise:'Reduce noise; protect grain.',
    globalEnhancement:'Sharpen and deblur at native size.',
    globalCleanup:'Vignetting, scratches and damage.',
    globalStabilize:'Correct unwanted frame motion.',
    globalDetail:'Existing post-stabilization stage.',
    globalExpandOutpaint:'Extend the picture framing.',
    globalColorize:'References and colour propagation.',
    globalSoundtrack:'Generate music or sound effects.',
    globalUpscale:'Increase delivery resolution.',
  };
  return `
    <li class="workflow-stage-tile${orderKey ? ' is-preprocessing' : ''}">
    <div class="workflow-paired-row">
    <div class="workflow-stage-control">
    <label class="workflow-option ${item.checked ? 'is-on' : ''}" for="${item.id}" title="${esc(item.body)}">
      <input id="${item.id}" type="checkbox" aria-labelledby="${item.id}Title" aria-describedby="${item.id}Description" ${item.checked ? 'checked' : ''}>
      <span class="workflow-thumb workflow-${esc(item.icon)}" aria-hidden="true"></span>
      <strong id="${item.id}Title">${esc(item.title)}${item.id==='globalDetail' ? ' · legacy' : ''}</strong>
      <span class="workflow-stage-state" aria-hidden="true"></span>
      <span class="workflow-summary">${esc(summaries[item.id] || item.body)}${item.note ? `<em>${esc(item.note)}</em>` : ''}</span>
      <span class="sr-only" id="${item.id}Description">${esc(item.body)}</span>
    </label>
    ${orderKey ? workflowOrderControlsHtml(orderKey,item.title) : ''}
    </div>
    <div class="workflow-stage-progress" data-workflow-stage-status="${item.id}" aria-label="Whole remaster status for ${esc(item.title)}">${workflowStageProgressHtml(item.id,item.checked)}</div>
    </div>
    </li>
  `;
}

function workflowStageProgressHtml(id,enabled) {
  if(!enabled) return '<div class="workflow-status-entry"><strong>Off · skipped</strong><p>Not in the current queue.</p></div>';
  const keys=OVERVIEW_WORKFLOW_STAGE_KEYS[id] || [];
  const stages=((state.phase_progress && state.phase_progress.stages) || []).filter(item=>keys.includes(item.key));
  if(!stages.length) return '<div class="workflow-status-entry"><strong>Selected</strong><p>No status reported yet.</p></div>';
  return stages.map(sp=>{
    const row=(state.progress || []).find(item=>item.stage===sp.stage);
    const status=row ? row.status : sp.label;
    return `<div class="workflow-status-entry">
      ${stages.length>1 ? `<h3>${esc(sp.stage)}</h3>` : ''}
      <strong>${esc(status || 'Selected')}</strong>
      ${progressHtml(sp.percent,sp.label)}
      <p>Latest output: ${esc((row && row.latest) || '—')}</p>
    </div>`;
  }).join('');
}

function workflowOrderControlsHtml(key,title) {
  const order=pipelineOrder(),index=order.indexOf(key),busy=state.running || state.queue_running || pipelineOrderPending;
  return `<div class="workflow-order-buttons" role="group" aria-label="Processing order for ${esc(title)}">
    ${[-1,1].map(direction=>`<button type="button" data-workflow-order-key="${key}" data-workflow-order-direction="${direction}" onclick="movePipelineStage('${key}',${direction})" ${busy || index+direction<0 || index+direction>=order.length ? 'disabled' : ''} aria-label="Move ${esc(title)} ${direction<0 ? 'earlier' : 'later'}" title="Move ${esc(title)} ${direction<0 ? 'earlier' : 'later'} in processing order">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="${direction<0 ? 'm6 10 6-6 6 6M12 4v16' : 'm6 14 6 6 6-6M12 4v16'}"/></svg>
    </button>`).join('')}
  </div>`;
}

function sourceAnalysisHtml(analysis) {
  if (!analysis || !analysis.message || analysis.ready) return '<div id="overviewSourceAnalysis"></div>';
  const percent = Math.max(1, Math.min(100, Number(analysis.percent || 1)));
  return `
    <div id="overviewSourceAnalysis" class="inline-warning">
      <strong>${esc(analysis.message)}</strong>
      ${progressHtml(percent, percent + '%')}
    </div>
  `;
}

function overviewSourcePicker(source, global) {
  const sequence = state.source_sequence || {};
  const sequenceFps = sequence.fps || global.source_sequence_fps || '24';
  const shortName = value => String(value || '').split(/[\\/]/).pop();
  const sequenceSummary = sequence.count
    ? `<p class="source-sequence-summary"><strong>${sequence.count} images</strong> (${esc((sequence.formats || []).join(', '))}), naturally sorted from <span title="${esc(sequence.first || '')}">${esc(shortName(sequence.first))}</span> to <span title="${esc(sequence.last || '')}">${esc(shortName(sequence.last))}</span>.</p>`
    : '';
  return `
    <label>Source material</label>
    <div class="field-row">
      <input id="globalSource" value="${esc(source)}">
      <button type="button" onclick="browseGlobalSource()">Browse…</button>
    </div>
    <div class="source-sequence-controls">
      <label for="globalSequenceFps">Image sequence FPS</label>
      <input id="globalSequenceFps" type="number" min="1" max="240" step="0.001" value="${esc(sequenceFps)}" ${sequence.count ? 'onchange="saveSourceSequenceFps()"' : ''}>
      <span>Select one video, or multiple PNG, BMP, JPG, DPX, TIFF, or WebP images.</span>
    </div>
    ${sequenceSummary}
  `;
}

function overviewFilmstrip() {
  return overviewFilmstripInner();
}

function overviewFilmstripInner() {
  const thumbs = (state.source_previews || [])
    .map((path,index) => `<figure><img src="${esc(media(path))}" alt="Source sample ${index+1}"><figcaption>Source sample ${index+1}</figcaption></figure>`)
    .join('');
  return thumbs ? `<div class="source-samples"><h3>Source samples</h3><p>Stills from the source for a quick visual check. Not restored outputs.</p><div class="source-sample-images">${thumbs}</div></div>` : '';
}

function overviewSectionPicker(global, source, playbackSource) {
  const start = Number(global.section_start || 0);
  const duration = parseDuration((state.source_info && state.source_info.duration) || '0');
  const end = Number(global.section_end || duration || 0);
  const fpsText = (state.source_info && (state.source_info.fps || state.source_info.frame_rate)) || '';
  const fps = Math.max(1, Number(String(fpsText).split(' ')[0]) || 24);
  const sliderMax = Math.max(duration, end, 1);
  const frameStep = (1 / fps).toFixed(6);
  return `
    <div class="section-picker overview-section-picker">
      <div class="overview-source-layout">
      ${source ? `<video id="sourceSectionVideo" class="section-video" src="${media(playbackSource)}" controls preload="metadata"></video>` : ''}
      <div id="overviewFilmstrip">${source ? overviewFilmstripInner() : ''}</div>
      <div id="overviewSourceInfo">${source ? sourceInfoHtml(state.source_info || {}) : ''}</div>
      </div>
      <div class="editor-controls">
        <div>
          <label>Start: <span id="sectionStartLabel">${formatSeconds(start)}</span></label>
          <div class="frame-nudge-row">
            <input id="sectionStart" type="range" min="0" max="${sliderMax}" step="${frameStep}" value="${start}" data-fps="${fps}">
            <button type="button" class="frame-nudge" onclick="nudgeSectionBoundary('start',-1)">−1</button>
            <button type="button" class="frame-nudge" onclick="nudgeSectionBoundary('start',1)">+1</button>
          </div>
        </div>
        <div>
          <label>End: <span id="sectionEndLabel">${end ? formatSeconds(end) : 'End'}</span></label>
          <div class="frame-nudge-row">
            <input id="sectionEnd" type="range" min="0" max="${sliderMax}" step="${frameStep}" value="${end || duration || 0}" data-fps="${fps}">
            <button type="button" class="frame-nudge" onclick="nudgeSectionBoundary('end',-1)">−1</button>
            <button type="button" class="frame-nudge" onclick="nudgeSectionBoundary('end',1)">+1</button>
          </div>
        </div>
      </div>
      <div class="shot-tools">
        <button type="button" onclick="markSourceSection('start')">Mark Start</button>
        <button type="button" onclick="markSourceSection('end')">Mark End</button>
      </div>
    </div>
  `;
}

function bindOverviewSectionControls() {
  const start = document.getElementById('sectionStart');
  const end = document.getElementById('sectionEnd');
  if (start) start.addEventListener('input', () => document.getElementById('sectionStartLabel').textContent = formatSeconds(start.value));
  if (end) end.addEventListener('input', () => document.getElementById('sectionEndLabel').textContent = formatSeconds(end.value));
  if (start) start.addEventListener('change', saveGlobalSection);
  if (end) end.addEventListener('change', saveGlobalSection);
}

function overviewActions() {
  const progress = (state.phase_progress && state.phase_progress.global) || { percent: 0, label: 'Waiting' };
  return `
    <div class="actions">
      <button class="primary" onclick="runAll()">Run Whole Remaster</button>
      <button class="warn" onclick="stopRun()" ${state.running ? '' : 'disabled'}>Stop</button>
    </div>
    <div id="overviewPipelineProgress">${progressHtml(progress.percent, progress.label)}</div>
  `;
}

function overviewProgressTable() {
  const mappedKeys=Object.values(OVERVIEW_WORKFLOW_STAGE_KEYS).flat();
  const mappedTitles=new Set(((state.phase_progress && state.phase_progress.stages) || []).filter(item=>mappedKeys.includes(item.key)).map(item=>item.stage));
  const rows = (state.progress || []).filter(item=>!mappedTitles.has(item.stage)).map(progressRow => {
    const sp = stageProgressByTitle(progressRow.stage);
    return `
      <article class="overview-progress-stage">
        <header><h3>${esc(progressRow.stage)}</h3><span>Status: ${esc(progressRow.status)}</span></header>
        ${progressHtml(sp.percent, sp.label)}
        <p>Latest output: ${esc(progressRow.latest || '—')}</p>
      </article>
    `;
  }).join('');

  return rows ? `
    <div class="overview-stage-progress-list">
      <h3>Pipeline work without a stage switch</h3>
      ${rows}
    </div>
  ` : '';
}
