/* Only the stage-specific picker; the comparison, clock, stills and crop are shared. */
function deflickerPickerHtml() {
  const stage=state.stages.find(item=>item.key==='deflicker');
  const values=state.settings.deflicker || {};
  const selected=denoiseSelectedVariant(),pipeline=denoiseStageVariant();
  return `<section class="denoise-variant-picker" aria-labelledby="denoiseVariantHeading">
    <input type="hidden" data-field="method" value="${esc(values.method || 'arp')}">
    <div class="denoise-variant-head"><div><h3 id="denoiseVariantHeading">Compare variants</h3>
      <p class="denoise-variant-help">Choose up to five methods. Comparing does not change the pipeline.</p></div>
      <output id="denoiseMatrixStatus" class="denoise-matrix-status" role="status">${denoiseMatrixState().ready || 0}/5 rendered</output></div>
    <div class="deflicker-variant-list" role="group" aria-label="Deflicker methods included in comparison" onkeydown="denoiseMatrixKeyboard(event)">
      ${DEFLICKER_METHODS.map(method=>denoiseMatrixCellHtml(method,{key:'settings',label:'Current settings'},selected)).join('')}
    </div>
    <div class="denoise-selection-tools"><span data-dm-comparison-count role="status"></span>
      <button type="button" onclick="denoiseWorkspaceSelectAll()">Select all</button><button type="button" onclick="denoiseWorkspaceClear()">Clear</button></div>
    <div class="denoise-generation-prompt" data-dm-generation hidden><p data-dm-generation-text></p><div class="actions">
      <button type="button" class="primary" data-dm-generate onclick="denoiseWorkspaceGenerate()">Render missing variants</button>
      <button type="button" onclick="denoiseWorkspaceDismissGeneration()">Not now</button></div><output data-dm-generation-status role="status"></output></div>
    <button type="button" class="denoise-off-option" onclick="denoiseMatrixUseOff()">Pipeline: use unchanged input</button>
    ${comparisonSelectedHtml(denoiseVariantLabelFor(selected.method,selected.preset),denoiseVariantLabelFor(pipeline.method,pipeline.preset),Boolean(state.running))}
    <p class="shot-empty" data-df-analysis hidden></p>
    <div class="deflicker-parameters"><h3>Method settings</h3>
      <p class="shot-empty">ARP uses strength and correction limit; FFmpeg uses the temporal window. Edits invalidate only affected outputs. Nothing renders until you confirm.</p>
      ${stage.fields.filter(field=>field[0]!=='method').map(field=>fieldHtml(stage,field)).join('')}
      <p class="inline-warning">Localized methods are experimental and require the corresponding confirmation above. Short experiment: one shot, at most 240 frames and 75 million total pixels. Streaming: bounded five-frame buffer, no fixed film-length limit. Both use fixed 60% strength and an 8% cap; inspect cuts, fades, movement and deliberate exposure changes.</p>
    </div>
    <p class="shot-empty">Stage input is the selected output of the preceding enabled stage in your processing order, or the original source when this stage runs first. Pixel changes are not proof of restoration quality.</p>
  </section>`;
}
