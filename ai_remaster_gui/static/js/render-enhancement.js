/* Stage-specific choices; the retained workspace owns comparison and display lifetime. */
function enhancementPickerHtml() {
  const stage=state.stages.find(item=>item.key==='enhancement');
  const values=state.settings.enhancement || {};
  const selected=denoiseSelectedVariant(),pipeline=denoiseStageVariant();
  return `<section class="denoise-variant-picker" aria-labelledby="denoiseVariantHeading">
    <input type="hidden" data-field="method" value="${esc(values.method || 'off')}">
    <div class="denoise-variant-head"><div><h3 id="denoiseVariantHeading">Compare methods</h3>
      <p class="denoise-variant-help">Choose methods to compare at the current settings. Selection does not change the pipeline.</p></div>
      <output id="denoiseMatrixStatus" class="denoise-matrix-status" role="status">${denoiseMatrixState().ready || 0}/6 rendered</output></div>
    <div class="deflicker-variant-list" role="group" aria-label="Enhancement methods included in comparison" onkeydown="denoiseMatrixKeyboard(event)">
      ${ENHANCEMENT_METHODS.map(method=>denoiseMatrixCellHtml(method,{key:'settings',label:'Current settings'},selected)).join('')}
    </div>
    <div class="denoise-selection-tools"><span data-dm-comparison-count role="status"></span>
      <button type="button" onclick="denoiseWorkspaceSelectAll()">Select all</button><button type="button" onclick="denoiseWorkspaceClear()">Clear</button></div>
    <div class="denoise-generation-prompt" data-dm-generation hidden><p data-dm-generation-text></p><div class="actions">
      <button type="button" class="primary" data-dm-generate onclick="denoiseWorkspaceGenerate()">Render missing variants</button>
      <button type="button" onclick="denoiseWorkspaceDismissGeneration()">Not now</button></div><output data-dm-generation-status role="status"></output></div>
    <button type="button" class="denoise-off-option" onclick="denoiseMatrixUseOff()">Pipeline: use unchanged input</button>
    ${comparisonSelectedHtml(denoiseVariantLabelFor(selected.method,selected.preset),denoiseVariantLabelFor(pipeline.method,pipeline.preset),Boolean(state.running))}
    <div class="deflicker-parameters"><h3>Method settings</h3>
      <p class="shot-empty">Strength presets apply to the four classical methods, not to the neural checkpoints. Mix applies to every method. CAS Gentle is a starting point for inspection, not a proven quality winner.</p>
      ${stage.fields.filter(field=>field[0]!=='method').map(field=>fieldHtml(stage,field)).join('')}
      <p class="inline-warning">Richardson–Lucy assumes a Gaussian blur, not a measured film PSF. NAFNet and Restormer use GoPro-trained image models: inspect grain, faces, text, ringing and temporal flicker. Neural methods require prepared checkpoints and CUDA; the commercial model guard blocks unaudited weights.</p>
    </div>
    <p class="shot-empty">Stage input is the selected output of the preceding enabled stage in your processing order, or the original source if this stage runs first. Native 1×: no upscaling. More sharpness or larger pixel changes do not prove recovered historical detail. Later legacy processing is unchanged.</p>
  </section>`;
}
