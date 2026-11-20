/* Drafts, inspection and pipeline adoption are separate decisions. */
let dustDraft=null,dustProject='',dustWorkspaceCatalog='',dustRunPending=false,dustUpdateStatus=null;
let dustEpoch=0,dustInspectEpoch=0,dustInspectionKey='',dustView='mask',dustStatus=null,dustFollowDraft=false;
let dustRepairOpen=false,dustInputMode='input',dustOpacity=0.5;
let dustPaintFrame=null;
let dustAlignmentObserver=null;
let dustSourceMode='stage';
let dustAbSplitActive=false,dustSplitPos=50;
let dustChannel='mask';

function dustMatrix() {return dustSourceMode==='original' ? state.dust_matrix?.original || {} : state.dust_matrix || {};}
function dustSourceLabel() {return dustSourceMode==='original' ? 'Original input' : 'Stage input';}
function dustInputBlocked() {return dustSourceMode==='original' ? !dustMatrix().source : Boolean(state.processing_input?.problem);}
function dustSelectSource(mode) {
  if(!['stage','original'].includes(mode) || mode===dustSourceMode) return;
  if(typeof dustMaskEditorFlushSave==='function') void dustMaskEditorFlushSave();
  dustSourceMode=mode;dustFollowDraft=false;dustInspectEpoch++;dustInspectionKey='';dustStatus=null;
  if(typeof dustMaskEditorResetLocalCache==='function') dustMaskEditorResetLocalCache();
  denoiseWorkspaces.delete('dust');drawDust(state.stages.find(stage=>stage.key==='dust'));
}

const DUST_DETECTOR_META = {
  farneback_3f: {tier: 'Classical Detectors', label: 'Farneback 3F', readiness: 'pretrained_operational', commercial: true, license: 'Apache-2.0 / BSD-3-Clause', supported_channels: ['scratch', 'dust'], unsupported_channels: []},
  dis_flow: {tier: 'Classical Detectors', label: 'DIS Flow', readiness: 'pretrained_operational', commercial: true, license: 'Apache-2.0 / BSD-3-Clause', supported_channels: ['scratch', 'dust'], unsupported_channels: []},
  ridge: {tier: 'Classical Detectors', label: 'Temporal Ridge', readiness: 'pretrained_operational', commercial: true, license: 'Apache-2.0 / BSD-3-Clause', supported_channels: ['scratch'], unsupported_channels: ['dust']},
  consensus_5f: {tier: 'Classical Detectors', label: 'Consensus 5F', readiness: 'pretrained_operational', commercial: true, license: 'Apache-2.0 / BSD-3-Clause', supported_channels: ['scratch', 'dust'], unsupported_channels: []},
  legacy: {tier: 'Classical Detectors', label: 'ARP Legacy Scratch Mask', readiness: 'pretrained_operational', commercial: true, license: 'Apache-2.0 / BSD-3-Clause', supported_channels: ['scratch'], unsupported_channels: ['dust']},
  temporal_scratch: {tier: 'Classical Detectors', label: 'Temporal Scratch (Motion-Comp)', readiness: 'pretrained_operational', commercial: true, license: 'Apache-2.0 / BSD-3-Clause', supported_channels: ['scratch'], unsupported_channels: ['dust', 'uncertain']},
  dart: {tier: 'Neural Baseline (DART)', label: 'DART MaskNet', readiness: 'pretrained_operational', commercial: false, license: 'Restricted Research (RTN)', supported_channels: ['scratch', 'dust'], unsupported_channels: []},
  bopb: {tier: 'Neural Candidates', label: 'BOPB Scratch UNet', readiness: 'pretrained_operational', commercial: false, license: 'Restricted (Microsoft Academic Non-Commercial)', supported_channels: ['scratch'], unsupported_channels: ['dust']},
  film_artefact_segmenter: {tier: 'Neural Candidates', label: 'Film Artefact Segmenter', readiness: 'pretrained_operational', commercial: true, license: 'Permissive (MIT Clean-room reproduction)', supported_channels: ['scratch', 'dust'], unsupported_channels: []},
  cnn_blotches: {tier: 'Neural Candidates', label: 'CNN Blotches & Scratches', readiness: 'pretrained_operational', commercial: true, license: 'Permissive (MIT Clean-room reproduction)', supported_channels: ['scratch', 'dust'], unsupported_channels: []},
  patchcore: {tier: 'Neural Candidates', label: 'PatchCore Anomaly Detector', readiness: 'trained_experimental', commercial: true, license: 'Permissive (Apache-2.0 / BSD-3-Clause)', supported_channels: ['combined'], unsupported_channels: ['scratch', 'dust', 'uncertain']},
  padim: {tier: 'Neural Candidates', label: 'PaDiM Anomaly Detector', readiness: 'trained_experimental', commercial: true, license: 'Permissive (Apache-2.0 / BSD-3-Clause)', supported_channels: ['combined'], unsupported_channels: ['scratch', 'dust', 'uncertain']},
  correl_net: {tier: 'Neural Candidates', label: 'Correl-Net', readiness: 'training_required', commercial: false, license: 'Training Required (Architecture clean-room implemented)', supported_channels: ['scratch', 'dust'], unsupported_channels: []},
  experimental_e2: {tier: 'Experimental Prototypes', label: 'Experimental E2 · Scene-vs-Film (Experimental)', readiness: 'trained_experimental', commercial: true, license: 'Apache-2.0 / BSD-3-Clause (OpenCV + NumPy; Experimental Unvalidated)', supported_channels: ['scratch', 'uncertain'], unsupported_channels: ['dust'], experimental_opt_in_required: true, research_confirmed_required: true, three_state_outputs: ['high_confidence_damage', 'uncertain_damage_candidate', 'likely_intact_scene_structure']},
};

const CANONICAL_DUST_PRESETS = {
  balanced: {detector:'farneback_3f', sensitivity:0.65, dilate:1, threshold:160, repair:'telea', commercial_cleared:true},
  fast_preview: {detector:'dis_flow', sensitivity:0.60, dilate:0, threshold:160, repair:'telea', commercial_cleared:true},
  high_quality_precision: {detector:'consensus_5f', sensitivity:0.70, dilate:0, threshold:160, repair:'telea', commercial_cleared:true},
  scratch_focused: {detector:'ridge', sensitivity:0.65, dilate:0, threshold:160, repair:'telea', commercial_cleared:true},
  dust_focused: {detector:'dis_flow', sensitivity:0.75, dilate:1, threshold:160, repair:'telea', commercial_cleared:true},
  high_recall: {detector:'consensus_5f', sensitivity:0.85, dilate:2, threshold:160, repair:'telea', commercial_cleared:true},
  conservative_low_fp: {detector:'farneback_3f', sensitivity:0.50, dilate:0, threshold:160, repair:'telea', commercial_cleared:true},
  dart_optimal_f1: {detector:'dart', sensitivity:0.65, dilate:0, threshold:224, repair:'telea', commercial_cleared:false},
  dart_high_recall: {detector:'dart', sensitivity:0.65, dilate:0, threshold:192, repair:'telea', commercial_cleared:false},
  dart_conservative: {detector:'dart', sensitivity:0.65, dilate:0, threshold:240, repair:'telea', commercial_cleared:false},
  bopb_scratch_balanced: {detector:'bopb', sensitivity:0.50, dilate:0, threshold:128, repair:'telea', commercial_cleared:false},
  bopb_scratch_high_recall: {detector:'bopb', sensitivity:0.30, dilate:0, threshold:76, repair:'telea', commercial_cleared:false},
  film_artefact_balanced: {detector:'film_artefact_segmenter', sensitivity:0.50, dilate:0, threshold:128, repair:'telea', commercial_cleared:true},
  cnn_blotches_precision: {detector:'cnn_blotches', sensitivity:0.80, dilate:0, threshold:204, repair:'telea', commercial_cleared:true},
  patchcore_anomaly_research: {detector:'patchcore', sensitivity:0.50, dilate:0, threshold:128, repair:'telea', commercial_cleared:true},
  padim_anomaly_research: {detector:'padim', sensitivity:0.50, dilate:0, threshold:128, repair:'telea', commercial_cleared:true},
  temporal_scratch_balanced: {detector:'temporal_scratch', sensitivity:0.65, dilate:0, threshold:160, repair:'telea', commercial_cleared:true},
  experimental_e2_research: {detector:'experimental_e2', sensitivity:0.65, dilate:0, threshold:160, repair:'telea', commercial_cleared:true},
};

const DETECTOR_DEFAULT_PRESETS = {
  farneback_3f: 'balanced',
  temporal: 'balanced',
  dis_flow: 'fast_preview',
  consensus_5f: 'high_quality_precision',
  ridge: 'scratch_focused',
  legacy: 'legacy_baseline',
  temporal_scratch: 'temporal_scratch_balanced',
  dart: 'dart_optimal_f1',
  dart_harness: 'dart_optimal_f1',
  bopb: 'bopb_scratch_balanced',
  film_artefact_segmenter: 'film_artefact_balanced',
  cnn_blotches: 'cnn_blotches_precision',
  patchcore: 'patchcore_anomaly_research',
  padim: 'padim_anomaly_research',
  correl_net: 'custom',
  experimental_e2: 'experimental_e2_research',
};

const DUST_PRESET_MAP = new Proxy(CANONICAL_DUST_PRESETS, {
  get(target, prop, receiver) {
    const live = (typeof state !== 'undefined' ? state?.dust_matrix?.presets : null);
    if (live && typeof live === 'object' && prop in live) {
      return live[prop];
    }
    return Reflect.get(target, prop, receiver);
  },
  has(target, prop) {
    const live = (typeof state !== 'undefined' ? state?.dust_matrix?.presets : null);
    if (live && typeof live === 'object' && prop in live) return true;
    return Reflect.has(target, prop);
  },
  ownKeys(target) {
    const live = (typeof state !== 'undefined' ? state?.dust_matrix?.presets : null);
    if (live && typeof live === 'object') {
      const keys = Object.keys(live);
      if (keys.length > 0) return keys;
    }
    return Reflect.ownKeys(target);
  },
  getOwnPropertyDescriptor(target, prop) {
    const live = (typeof state !== 'undefined' ? state?.dust_matrix?.presets : null);
    if (live && typeof live === 'object' && prop in live) {
      return Object.getOwnPropertyDescriptor(live, prop);
    }
    return Reflect.getOwnPropertyDescriptor(target, prop);
  }
});

if (typeof globalThis !== 'undefined') {
  globalThis.DUST_PRESET_MAP = DUST_PRESET_MAP;
  globalThis.DETECTOR_DEFAULT_PRESETS = DETECTOR_DEFAULT_PRESETS;
  globalThis.CANONICAL_DUST_PRESETS = CANONICAL_DUST_PRESETS;
  globalThis.DUST_DETECTOR_META = DUST_DETECTOR_META;
}

function readinessBadge(readiness) {
  if (readiness === 'pretrained_operational') return '[Operational]';
  if (readiness === 'trained_experimental') return '[Experimental]';
  if (readiness === 'training_required') return '[Training Required]';
  if (readiness === 'blocked') return '[Blocked]';
  return `[${readiness || 'Unknown'}]`;
}

function channelOrigin(channelKey, detector) {
  const detKey = detector === 'temporal' ? 'farneback_3f' : (detector || 'farneback_3f');
  const meta = (state.dust_matrix?.available_detectors || []).find(d => d.name === detKey) || DUST_DETECTOR_META[detKey] || {};
  const supported = meta.supported_channels || ['scratch', 'dust'];
  if (channelKey === 'mask') return 'Combined mask';
  if (channelKey === 'vetoed') return detKey === 'experimental_e2' ? 'Native E2 vetoed intact structure' : 'Unavailable for this detector';
  if (['scratch','dust','uncertain'].includes(channelKey)) return supported.includes(channelKey) ? 'Native channel' : 'No native channel';
  return '';
}

const dustNames={
  legacy:'ARP legacy scratch mask',
  temporal:'Temporal Dust',
  farneback_3f:'Farneback 3F',
  dis_flow:'DIS Flow',
  ridge:'Temporal Ridge · experimental',
  consensus_5f:'Consensus 5F',
  dart:'DART · neural MaskNet (research)',
  dart_harness:'DART · neural MaskNet (research)',
  bopb:'BOPB Scratch UNet (research)',
  film_artefact_segmenter:'Film Artefact Segmenter (commercial)',
  cnn_blotches:'CNN Blotches & Scratches (commercial)',
  patchcore:'PatchCore Anomaly Detector · [Experimental · High FP Risk]',
  padim:'PaDiM Anomaly Detector · [Experimental · High FP Risk]',
  correl_net:'Correl-Net (training required)',
  temporal_scratch:'Temporal Scratch (Motion-Comp · Commercial)',
  experimental_e2:'Experimental E2 · Scene-vs-Film (Experimental)',
  telea:'OpenCV Telea',
  propainter:'ProPainter · non-commercial',
  balanced:'Balanced · Farneback 3F (Commercial)',
  fast_preview:'Fast Preview · DIS Flow (Commercial)',
  high_quality_precision:'High Precision · Consensus 5F (Commercial)',
  scratch_focused:'Scratch Focused · Ridge (Commercial)',
  dust_focused:'Dust Focused · DIS Flow (Commercial)',
  high_recall:'High Recall · Consensus 5F (Commercial)',
  conservative_low_fp:'Conservative · Farneback 3F (Commercial)',
  dart_optimal_f1:'DART Optimal F1 · th=224 (Research-only)',
  dart_high_recall:'DART High Recall · th=192 (Research-only)',
  dart_conservative:'DART Conservative · th=240 (Research-only)',
  bopb_scratch_balanced:'BOPB Scratch Balanced · th=128 (Research-only)',
  bopb_scratch_high_recall:'BOPB Scratch High Recall · th=76 (Research-only)',
  film_artefact_balanced:'Film Artefact Balanced · th=128 (Commercial)',
  cnn_blotches_precision:'CNN Blotches Precision · th=204 (Commercial)',
  patchcore_anomaly_research:'PatchCore Anomaly Research · [Not Recommended]',
  padim_anomaly_research:'PaDiM Anomaly Research · [Not Recommended]',
  temporal_scratch_balanced:'Temporal Scratch Balanced · radius=3 (Commercial)',
  experimental_e2_research:'Experimental E2 Research · (Experimental · 3-State)',
  custom:'Custom Parameters',
};

function dustAvailableFrames(res) {
  return Number(res?.frames || (res?.base_mask_ready ? res?.base_frames : 0) || 0);
}
function dustActiveEditSignature(sourceMode) {
  const srcMode = sourceMode || (typeof dustSourceMode === 'string' ? dustSourceMode : 'stage');
  if (
    typeof dustMaskEditor !== 'undefined' &&
    dustMaskEditor &&
    dustMaskEditor.analysisSource === srcMode &&
    (dustMaskEditor.editSignature || dustMaskEditor.timelineId)
  ) {
    return String(dustMaskEditor.editSignature || '');
  }
  const edits =
    srcMode === 'original'
      ? (state?.dust_edits?.original || state?.dust_matrix?.original?.dust_edits || null)
      : (state?.dust_edits?.stage || state?.dust_edits || state?.dust_matrix?.dust_edits || null);
  return String(edits?.edit_signature || (srcMode === (typeof dustSourceMode === 'string' ? dustSourceMode : 'stage') ? dustMatrix()?.active_edit_signature : '') || '');
}
function dustActiveTimelineId(sourceMode) {
  const srcMode = sourceMode || (typeof dustSourceMode === 'string' ? dustSourceMode : 'stage');
  if (
    typeof dustMaskEditor !== 'undefined' &&
    dustMaskEditor &&
    dustMaskEditor.analysisSource === srcMode &&
    dustMaskEditor.timelineId
  ) {
    return String(dustMaskEditor.timelineId || '');
  }
  const edits =
    srcMode === 'original'
      ? (state?.dust_edits?.original || state?.dust_matrix?.original?.dust_edits || null)
      : (state?.dust_edits?.stage || state?.dust_edits || state?.dust_matrix?.dust_edits || null);
  return String(edits?.timeline_id || '');
}
function dustContext() {return JSON.stringify([state.project_path,state.settings.global.source,dustSourceMode,dustMatrix().source_revision]);}
function dustCatalog() {return JSON.stringify([dustContext(),(dustMatrix().variants || []).map(v=>[v.method,v.repair_complete])]);}
function dustLabel(values) {return (values.detector==='dart' ? 'DART · joint restoration' : `${dustNames[values.detector] || values.detector} · ${dustNames[values.repair] || values.repair}`)+(values.analysis_source==='original'?' · Original input':'')+(values.mask_version ? ` · mask ${values.mask_version.slice(0,6)}` : '');}
function dustMatches(values,draft=dustDraft) {
  if(!draft || (values.analysis_source || 'stage')!==(draft.analysis_source || 'stage') || (values.mask_version || '')!==(draft.mask_version || '')) return false;
  const draftSource = draft.analysis_source || (typeof dustSourceMode === 'string' ? dustSourceMode : 'stage');
  const activeEditSig = dustActiveEditSignature(draftSource);
  const activeTid = dustActiveTimelineId(draftSource);
  const expectedSig = String(draft.edit_signature ?? activeEditSig ?? '');
  if(String(values.edit_signature || '')!==expectedSig) return false;
  const expectedTid = String(draft.timeline_id ?? (expectedSig ? activeTid : '') ?? '');
  if(expectedTid && String(values.timeline_id || '')!==expectedTid) return false;
  if(!expectedTid && String(values.timeline_id || '')) return false;
  return Object.entries(values).every(([key,value])=>{
    if(key==='edit_signature' || key==='timeline_id') return true;
    return String(draft[key] ?? '')===String(value);
  });
}

function dustActionState(result,busy=false,values=dustDraft || {},repairRequested=dustRepairOpen) {
  if(busy) return {kind:'',label:'Processing…',message:'A job is running. No new processing can start. Stop waits for the current operation.'};
  const detKey = values.detector === 'temporal' ? 'farneback_3f' : (values.detector || 'farneback_3f');
  const meta = (state.dust_matrix?.available_detectors || []).find(d => d.name === detKey) || DUST_DETECTOR_META[detKey] || {};
  const commercialGuard = state.settings?.global?.commercial_model_guard === 'true';

  const rState = meta.readiness_state || meta.readiness;
  if (rState === 'training_required') {
    return {
      kind: '',
      label: 'Training Required · Inexecutable',
      message: `${dustNames[detKey] || meta.label || values.detector} is in [Training Required] state. Multi-frame temporal training or checkpoint provisioning is required before activation. Accidental execution is blocked.`
    };
  }
  if (rState === 'blocked') {
    return {
      kind: '',
      label: 'Model Blocked',
      message: `${dustNames[detKey] || meta.label || values.detector} is in [Blocked] state and cannot be executed.`
    };
  }
  const isCommercial = meta.commercial_cleared !== undefined ? meta.commercial_cleared : (meta.commercial !== false);
  if (commercialGuard && !isCommercial) {
    return {
      kind: '',
      label: 'Commercial Safe: Blocked',
      message: `Commercial Model Guard is active. Non-commercial model ${dustNames[detKey] || meta.label || values.detector} (${meta.license_status || meta.license || 'restricted'}) cannot be executed in commercial safe mode. Turn off Commercial Model Guard in Global settings to test research models.`
    };
  }
  if (commercialGuard && values.repair === 'propainter') {
    return {
      kind: '',
      label: 'Commercial Safe: Blocked',
      message: 'Commercial Model Guard is active. ProPainter is non-commercial. Turn off Commercial Model Guard in Global settings to use ProPainter.'
    };
  }
  if (detKey === 'experimental_e2' && String(values.research_confirmed || 'false') !== 'true') {
    return {
      kind: '',
      label: 'Experimental Opt-In Required',
      message: 'Experimental E2 requires explicit experimental acknowledgment (research_confirmed) before detection or repair can run.'
    };
  }

  const activeEditSig = dustActiveEditSignature();
  const editsStaleForRepair = Boolean(activeEditSig && String(result?.edit_signature || '') !== activeEditSig);

  if(result?.coverage_warning) return {kind:'',label:'Excessive Mask Coverage · Repair Blocked',message:result.coverage_warning};
  const availableFrames = dustAvailableFrames(result);
  if(availableFrames && values.analysis_source==='original') return {kind:'',label:'Original mask ready · Saved',message:'Original input mask ready · Saved. Analysis only; switch to Stage input for repair. Pipeline selection unchanged.'};
  if(availableFrames && !repairRequested) return {kind:'',label:'Mask ready · Saved',message:activeEditSig ? 'Mask & human edits ready · Saved. Inspect or open repair to compose with your edits.' : 'Mask ready · Saved. Inspect or open the mask. Creating a repair is optional.'};
  if(result?.repair_error) return {kind:'',label:'Repair unavailable',message:'The saved repair is missing or changed. Its mask is still usable. Existing files will not be overwritten.'};
  if(result?.repair_complete && !editsStaleForRepair) return {kind:'',label:'Repair ready',message:'Repair ready. Compare the result before explicitly using it for the pipeline.'};
  if(availableFrames) return {kind:'repair',label:values.detector==='dart' ? 'Compose masked repair' : 'Repair masked areas',message:editsStaleForRepair ? 'Human mask edits are active and differ from the saved repair. Run repair to compose updated masks and output.' : 'Saved mask ready for this repair. Processing starts only on request.'};
  return {kind:'mask',label:values.detector==='dart' ? 'Calculate DART mask & restoration' : 'Calculate mask',message:'No mask for these settings. Calculate a mask first; no repair starts automatically.'};
}

function dustFieldHtml(field) {
  const [key,label,kind]=field,value=dustDraft[key] ?? '';
  if(kind==='checkbox') return `<label><input type="checkbox" data-dust="${key}" ${value==='true'?'checked':''}> ${esc(label)}</label>`;
  const commercialGuard = state.settings?.global?.commercial_model_guard === 'true';

  if (key === 'detector') {
    const isSelected = v => value === v || (v === 'farneback_3f' && value === 'temporal') || (v === 'temporal' && value === 'farneback_3f');
    const optgroups = [
      {
        name: 'Classical Detectors',
        keys: ['farneback_3f', 'dis_flow', 'ridge', 'consensus_5f', 'legacy', 'temporal_scratch']
      },
      {
        name: 'Neural Baseline (DART)',
        keys: ['dart']
      },
      {
        name: 'Neural Candidates',
        keys: ['bopb', 'film_artefact_segmenter', 'cnn_blotches', 'patchcore', 'padim', 'correl_net']
      },
      {
        name: 'Experimental Prototypes',
        keys: ['experimental_e2']
      }
    ];
    const groupHtml = optgroups.map(grp => {
      const opts = grp.keys.map(k => {
        const meta = (state.dust_matrix?.available_detectors || []).find(d => d.name === k) || DUST_DETECTOR_META[k] || {};
        const readiness = meta.readiness_state || meta.readiness || 'pretrained_operational';
        const badge = readinessBadge(readiness);
        const isCommercial = meta.commercial_cleared !== undefined ? meta.commercial_cleared : (meta.commercial !== false);
        const commBadge = !isCommercial ? (commercialGuard ? ' · [Non-Commercial · Blocked]' : ' · [Restricted]') : '';
        const disabled = commercialGuard && !isCommercial ? 'disabled' : '';
        return `<option value="${k}" ${isSelected(k)?'selected':''} ${disabled}>${esc(dustNames[k] || meta.label || k)} · ${badge}${commBadge}</option>`;
      }).join('');
      return `<optgroup label="${grp.name}">${opts}</optgroup>`;
    }).join('');
    return `<label>${esc(label)}<select data-dust="detector">${groupHtml}</select></label>`;
  }

  const options=kind.startsWith('select:') ? kind.slice(7).split('|').filter(v=>
    (key!=='repair' || v!=='dart' || dustDraft?.detector==='dart') &&
    (key!=='preset' || v==='custom' || (DUST_PRESET_MAP[v]?.detector==='temporal' ? 'farneback_3f' : DUST_PRESET_MAP[v]?.detector)===(dustDraft.detector==='temporal' ? 'farneback_3f' : dustDraft.detector))) : [];
  const isSelected = v => value === v;
  const control=options.length ? `<select data-dust="${key}">${options.map(v=>{
    let extra = '';
    if (key === 'preset') {
      const p = DUST_PRESET_MAP[v];
      const pDet = p ? p.detector : null;
      const meta = pDet ? ((state.dust_matrix?.available_detectors || []).find(d => d.name === pDet) || DUST_DETECTOR_META[pDet] || {}) : {};
      const isCommercial = meta.commercial_cleared !== undefined ? meta.commercial_cleared : (meta.commercial !== false);
      if (!isCommercial) extra = commercialGuard ? ' [Non-Commercial · Blocked]' : ' [Restricted]';
    }
    return `<option value="${v}" ${isSelected(v)?'selected':''}>${esc(dustNames[v] || v)}${extra}</option>`;
  }).join('')}</select>`
    : `<input data-dust="${key}" type="${kind==='text'?'text':'number'}" ${key==='sensitivity'?'min="0" max="1" step="0.05"':''} value="${esc(value)}">`;
  return `<label>${esc(label)}${control}</label>`;
}

function dustPickerHtml() {
  const selected=denoiseSelectedVariant(),st=state.stages.find(s=>s.key==='dust'),dart=dustDraft.detector==='dart';
  const commercialGuard = state.settings?.global?.commercial_model_guard === 'true';
  const detKey = dustDraft.detector === 'temporal' ? 'farneback_3f' : (dustDraft.detector || 'farneback_3f');
  const meta = (state.dust_matrix?.available_detectors || []).find(d => d.name === detKey) || DUST_DETECTOR_META[detKey] || {};
  const isCommercial = meta.commercial_cleared !== undefined ? meta.commercial_cleared : (meta.commercial !== false);
  const fields=keys=>keys.map(key=>st.fields.find(field=>field[0]===key)).filter(Boolean).map(dustFieldHtml).join('');

  let modelNotice = '';
  if (commercialGuard && !isCommercial) {
    modelNotice += `<p class="inline-warning"><strong>Commercial Safe Mode:</strong> ${dustNames[dustDraft.detector] || dustDraft.detector} is restricted under a non-commercial license (${meta.license_status || meta.license || 'restricted'}). Execution is blocked by Commercial Model Guard.</p>`;
  }
  if (dustDraft.detector === 'ridge') {
    modelNotice += '<p class="inline-warning">Temporal Ridge detects vertical scratches only; dust and blotches are unsupported. Unvalidated on authentic non-rigid historical damage.</p>';
  } else if (dustDraft.detector === 'consensus_5f') {
    modelNotice += '<p class="inline-note">Consensus 5F uses a 5-frame temporal window (radius 2). Boundary frames abstain with zero masks.</p>';
  } else if (dustDraft.detector === 'legacy') {
    modelNotice += '<p class="inline-note">ARP Legacy Scratch Mask uses single-frame spatial filtering without motion tracking.</p>';
  } else if (dustDraft.detector === 'dis_flow') {
    modelNotice += '<p class="inline-note">DIS Flow uses fast inverse-search optical flow for lower latency.</p>';
  } else if (dustDraft.detector === 'dart') {
    modelNotice += '<p class="inline-warning">DART uses neural Dilation Pyramid MaskNet (research-only; adapted RTN rights remain unresolved). Optimal F1 at threshold 224.</p>';
  } else if (dustDraft.detector === 'bopb') {
    modelNotice += '<p class="inline-warning">BOPB Scratch UNet is restricted under Microsoft Research Academic Non-Commercial License. Vertical scratch channel only.</p>';
  } else if (dustDraft.detector === 'film_artefact_segmenter') {
    modelNotice += '<p class="inline-note">Film Artefact Segmenter provides multi-task scratch and dust segmentation (commercial-safe clean-room reproduction).</p>';
  } else if (dustDraft.detector === 'cnn_blotches') {
    modelNotice += '<p class="inline-note">CNN Blotches & Scratches provides high-precision facial and edge-preserving defect detection (commercial-safe).</p>';
  } else if (dustDraft.detector === 'patchcore') {
    modelNotice += '<p class="inline-warning">PatchCore is a memory-bank anomaly detector (experimental). Sensitive to authentic film grain; recommended for low-grain archival textures.</p>';
  } else if (dustDraft.detector === 'padim') {
    modelNotice += '<p class="inline-warning">PaDiM is a patch distribution anomaly detector (experimental). Sensitive to authentic film grain; recommended for low-grain archival textures.</p>';
  } else if (dustDraft.detector === 'correl_net') {
    modelNotice += '<p class="inline-warning"><strong>Training Required:</strong> Correl-Net multi-frame temporal correlation defect detector is clean-room implemented but requires training or checkpoint provisioning before activation. Accidental execution is blocked.</p>';
  } else if (dustDraft.detector === 'temporal_scratch') {
    modelNotice += '<p class="inline-note">Temporal Scratch uses motion-compensated optical flow (DIS/Farneback) with telecine de-duplication to detect authentic archival scratches.</p>';
  } else if (dustDraft.detector === 'experimental_e2') {
    modelNotice += '<p class="inline-warning"><strong>Experimental (Opt-In):</strong> Experimental E2 separates high-confidence scratch proposals, uncertain candidates, and vetoed intact scene structure (<code>vetoed_mask.mkv</code>). Dust channel is unsupported. Requires explicit research confirmation below; unvalidated on independent historical ground truth.</p>';
  }
  if (dustDraft.enable_temporal_scratch === 'true' && dustDraft.detector !== 'temporal_scratch') {
    modelNotice += '<p class="inline-note"><strong>Temporal Enhancement Active:</strong> Fuses motion-compensated scratch proposals with primary detector. High-confidence scratches are added to the repair mask; uncertain candidates are routed to the preview channel only.</p>';
  }

   return `<section class="denoise-variant-picker"><h3>Prepare mask</h3>
    <section class="dust-parameter-group"><label>Source<select data-dust-source><option value="stage" ${dustSourceMode==='stage'?'selected':''}>Stage input</option><option value="original" ${dustSourceMode==='original'?'selected':''}>Original input · analysis only</option></select></label>${dustSourceMode==='original'?'<p>Analysis only. Switch to Stage input for repairs; masks are not transferred between sources.</p>':''}<div class="dust-fields">${fields(['detector','preset'])}</div>${dustDraft.detector !== 'temporal_scratch' ? `<div class="dust-fields">${fields(['enable_temporal_scratch'])}</div>` : ''}${modelNotice}<details class="dust-fine-tuning"><summary>Fine tuning</summary><div class="dust-fields">${fields(dart ? ['threshold','dilate'] : ['sensitivity','dilate'])}</div></details></section>
    ${dustRepairOpen ? `<section class="dust-parameter-group"><h3>Repair</h3><p>${dart && dustDraft.repair==='dart' ? 'DART creates its own mask and restoration together. The next step composes only the masked areas.' : 'Repairs the marked areas using the saved mask.'}</p><div class="dust-fields">${fields(dustDraft.repair==='telea' ? ['repair','radius'] : ['repair'])}</div><button type="button" data-dust-close-repair>Return to masks</button></section>` : ''}
    ${['ridge','dart','bopb','patchcore','padim','experimental_e2'].includes(dustDraft.detector) ? `<div class="dust-fields">${fields(['research_confirmed'])}</div>` : ''}
    ${['dart','bopb'].includes(dustDraft.detector) || dustRepairOpen && dustDraft.repair==='propainter' ? `<p class="inline-warning">${dustDraft.detector==='dart' ? 'DART is research-only; adapted RTN rights remain unresolved. Mask creation also runs restoration inference.' : dustDraft.detector==='bopb' ? 'BOPB is restricted under Microsoft Research Academic Non-Commercial License.' : 'ProPainter is non-commercial.'}</p><div class="dust-fields">${fields(['license_confirmed'])}</div>` : ''}
    ${dart ? `<details><summary>DART installation · managed by ARP</summary><p>Default: tools/dart with its own Python 3.12 environment. Set up once with install_windows.bat -DartOnly. No ComfyUI node required. Leave both overrides blank to use the managed installation.</p><div class="dust-fields">${fields(['dart_repo','dart_python'])}</div></details>` : ''}
    <p data-dust-draft role="status">Settings are a draft. Saved variants and pipeline selection stay unchanged.</p>
    <section class="dust-saved-results"><h3 id="denoiseVariantHeading">Saved masks & repairs</h3>
    <output id="denoiseMatrixStatus" role="status"></output>
    <div class="denoise-variant-grid">${denoiseMethods().map(m=>{
      const row=dustMatrix().variants?.find(v=>v.method===m.key),values=row?.values || {};
      const summary=[dustNames[values.preset] || values.preset,values.sensitivity!=null?'Sensitivity '+values.sensitivity:'',values.threshold!=null?'Threshold '+values.threshold:'',values.dilate!=null?'Expansion '+values.dilate:''].filter(Boolean).join(' · ');
      return `<div>${denoiseMatrixCellHtml(m,{key:'settings'},selected)}<p class="dust-saved-summary">${esc(summary)}</p></div>`;
    }).join('') || '<p>No saved masks for this input. Choose a method above to prepare one.</p>'}</div></section>
    ${comparisonSelectedHtml(denoiseVariantLabelFor(selected.method,'settings'),denoiseVariantLabelFor(denoiseStageVariant().method,'settings'),Boolean(state.running))}</section>`;
}
function dustSetView(view) {
  const editActive = typeof dustMaskEditorIsEditActive==='function' && dustMaskEditorIsEditActive();
  dustView=(!editActive && dustInputMode==='input' && dustSourceMode!=='original') ? 'result' : 'mask';
  const host=document.querySelector('[data-dust-catalog]');if(!host) return;
  const mask=dustView==='mask',repair=Boolean(dustRepairOpen && dustStatus?.repair_complete);
  host.classList.toggle('dust-has-repair',repair);host.classList.toggle('dust-input-mask',mask);
  host.querySelector('[data-dust-mask-view]').hidden=!mask;
  host.querySelector('#denoiseComparePlayers').hidden=mask && !repair;
  host.querySelector('[data-dust-opacity]').hidden=dustInputMode!=='overlay';
  const play=host.querySelector('#denoiseComparePlay');if(play) {play.disabled=mask && !dustAvailableFrames(dustStatus);play.title=mask?'Play stage input and saved mask on the shared frame clock':'Play repair comparison';}
   host.querySelectorAll('[data-dust-input]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.dustInput===dustInputMode || button.dataset.dustInput==='mask' && ['scratch','dust','uncertain','vetoed'].includes(dustInputMode))));
   const channel=host.querySelector('[data-dust-channel]');
   if(channel) {channel.disabled=['input','overlay'].includes(dustInputMode);channel.value=dustChannel;}
   const origin=host.querySelector('[data-dust-channel-origin]');
   if(origin) origin.textContent=['input','overlay'].includes(dustInputMode) ? (dustInputMode==='overlay'?'Overlay shows auto proposals, human edits, and protect intact layers.':'') : channelOrigin(dustInputMode,dustDraft.detector)+(dustInputMode!=='mask'?' · Exact still only. Missing channels are not replaced by the combined mask.':'');
  if(mask) {
    dustInvalidateInspection();
  }
  const playbackKey=dustPlaybackIdentity();
  if(host.dataset.maskPlayback!==playbackKey) {host.dataset.maskPlayback=playbackKey;denoiseWorkspaceRefresh();}
  if(mask && !denoiseWorkspaceSnapshot().exact) dustPaintFrame?.();
  dustUpdateInspectionContext();
  dustAlignImages();
}
function dustAlignImages() {
  const host=document.querySelector('.dust-input-mask.dust-has-repair');if(!host) return;
  const grid=host.querySelector('.dust-comparison-images'),caption=grid?.querySelector('[data-dust-images] figcaption');
  const wrap=[...host.querySelectorAll('#denoiseComparePlayers [data-dc-slot]')].find(panel=>Number(panel.dataset.dcSlot)>0 && !panel.hidden)?.querySelector('.denoise-compare-video-wrap');
  if(!caption || !wrap) return;
  caption.style.minHeight=innerWidth>600 ? Math.max(0,wrap.getBoundingClientRect().top-grid.getBoundingClientRect().top)+'px' : '';
}
function dustUpdateInspectionContext() {
  if(active!=='dust') return;
  const host=document.querySelector('[data-dust-catalog]');if(!host) return;
  const rows=dustMatrix().variants || [],summary=host.querySelector('#denoiseMatrixStatus');
  if(summary) summary.textContent=`${new Set(rows.map(row=>row.values?.mask_version || row.mask_sha256 || row.method)).size} saved masks · ${rows.filter(row=>row.repair_complete).length} repairs`;
  const selected=denoiseSelectedVariant(),row=dustMatrix().variants?.find(v=>v.method===selected.method);
  if(dustView==='mask' && (!row || !dustMatches(row.values))) {
    const context=host.querySelector('[data-review-context]');if(context) context.textContent=`Inspecting: ${dustAvailableFrames(dustStatus)?'saved':'draft'} mask · ${dustNames[dustDraft.detector]} · Pipeline selection unchanged`;
    host.querySelectorAll('[data-method-viewed]').forEach(badge=>badge.hidden=true);
    const label=host.querySelector('[data-dm-selected-label]');if(label) label.textContent='Draft mask · '+dustLabel(dustDraft);
    const use=host.querySelector('#denoiseUseVariant');if(use) use.hidden=true;
  }
  const notice=host.querySelector('#denoiseCompareNotice');
  if(notice && dustView==='mask') notice.textContent=denoiseWorkspaceSnapshot().exact?'Exact mask still · press Play for frame-locked compressed playback.':'Frame-locked mask playback · compressed display. Use Exact still to inspect mask boundaries.';
}
function dustMaskPlaybackActive() {return active==='dust' && (dustInputMode!=='input' || dustSourceMode==='original') && Boolean(dustAvailableFrames(dustStatus));}
function dustPlaybackIdentity() {return dustMaskPlaybackActive() ? JSON.stringify([dustContext(),dustDraft]) : '';}
function dustMaskCanvas() {
  const images=document.querySelector('[data-dust-images]');if(!images) return null;
  if(!images.querySelector('canvas')) images.innerHTML='<figure><figcaption></figcaption><div class="dust-image-viewport"><div class="dust-image-layer"><canvas hidden aria-label="Stage input mask inspection"></canvas></div></div></figure>';
  return images.querySelector('canvas');
}
function dustAbSplitShouldPan(event) {
  if(!event) return true;
  if(typeof event.button==='number' && event.button!==0) return true;
  if(event.shiftKey || event.altKey || event.ctrlKey || event.metaKey) return true;
  return Boolean(typeof dustMaskEditor!=='undefined' && dustMaskEditor && dustMaskEditor.spacePressed);
}
function dustAbSplitStartPointer(canvas, event, redrawFn, onFinish) {
  if(!dustAbSplitActive || dustAbSplitShouldPan(event)) return false;
  if(typeof event.stopPropagation==='function') event.stopPropagation();
  if(typeof event.preventDefault==='function') event.preventDefault();
  const host=document.querySelector('[data-dust-catalog]');
  const setPos=(evt)=>{
    const rect=typeof canvas.getBoundingClientRect==='function' ? canvas.getBoundingClientRect() : null;
    if(!rect || !rect.width) return;
    const x=evt.clientX - rect.left;
    dustSplitPos=Math.max(0, Math.min(100, Math.round((x / rect.width) * 100)));
    const slider=host?.querySelector('[data-dust-ab-slider-wrap] input');
    if(slider) slider.value=dustSplitPos;
    const out=host?.querySelector('[data-dust-ab-slider-wrap] output');
    if(out) out.textContent=dustSplitPos + '%';
    if(typeof redrawFn==='function') redrawFn();
  };
  const isPointer=typeof event.pointerId==='number' || (typeof event.type==='string' && event.type.startsWith('pointer'));
  if(isPointer && typeof canvas.setPointerCapture==='function' && typeof event.pointerId==='number') {
    try { canvas.setPointerCapture(event.pointerId); } catch {}
  }
  const moveEvt=isPointer ? 'pointermove' : 'mousemove';
  const upEvt=isPointer ? 'pointerup' : 'mouseup';
  const cancelEvt=isPointer ? 'pointercancel' : null;
  const win=typeof window!=='undefined' ? window : globalThis;
  const onMove=(me)=>{
    if(typeof me?.stopPropagation==='function') me.stopPropagation();
    setPos(me);
  };
  const onUp=(ue)=>{
    if(typeof ue?.stopPropagation==='function') ue.stopPropagation();
    if(isPointer && typeof canvas.releasePointerCapture==='function' && typeof event.pointerId==='number') {
      try {
        if(typeof canvas.hasPointerCapture!=='function' || canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      } catch {}
    }
    win?.removeEventListener?.(moveEvt, onMove);
    win?.removeEventListener?.(upEvt, onUp);
    if(cancelEvt) win?.removeEventListener?.(cancelEvt, onUp);
    if(typeof onFinish==='function') onFinish();
  };
  win?.addEventListener?.(moveEvt, onMove);
  win?.addEventListener?.(upEvt, onUp);
  if(cancelEvt) win?.addEventListener?.(cancelEvt, onUp);
  setPos(event);
  return true;
}
function dustWireCanvasInteraction(canvas, redrawFn, allowEdit = true) {
  if(!canvas) return;
  const editActive=Boolean(allowEdit && typeof dustMaskEditorIsEditActive==='function' && dustMaskEditorIsEditActive());
  if(canvas.style) {
    canvas.style.cursor=editActive ? 'crosshair' : (dustAbSplitActive ? 'ew-resize' : 'default');
    canvas.style.pointerEvents=(editActive || dustAbSplitActive) ? 'auto' : '';
  }
  let splitPointerActive=false;
  canvas.onpointerdown=(e)=>{
    if(allowEdit && typeof dustMaskEditorIsEditActive==='function' && dustMaskEditorIsEditActive()) {
      if(typeof dustMaskEditorHandlePointerDown==='function') dustMaskEditorHandlePointerDown(canvas, e);
      return;
    }
    if(dustAbSplitActive && dustAbSplitStartPointer(canvas, e, redrawFn, ()=>{ splitPointerActive=false; })) {
      splitPointerActive=true;
    }
  };
  canvas.onmousedown=(e)=>{
    if(allowEdit && typeof dustMaskEditorIsEditActive==='function' && dustMaskEditorIsEditActive()) {
      if(typeof dustMaskEditorHandleMouseDown==='function') dustMaskEditorHandleMouseDown(canvas, e);
      return;
    }
    if(splitPointerActive) return;
    if(dustAbSplitActive) dustAbSplitStartPointer(canvas, e, redrawFn);
  };
}
function dustPaintPlayback(atlas,manifest,frame) {
   if(!dustMaskPlaybackActive() || !manifest.regions?.mask) return;
   const identity=dustPlaybackIdentity(),canvas=dustMaskCanvas();if(!canvas) return;
   if(['scratch','dust','uncertain','vetoed'].includes(dustInputMode)) {
     canvas.hidden=true;
     const status=document.querySelector('[data-dust-frame-status]');
     if(status) status.textContent='Individual mask channels require Exact still. Select Exact still or use the combined mask for playback.';
     return;
   }
  const paint=()=>{
    if(identity!==dustPlaybackIdentity() || !canvas.isConnected || !atlas.width) return;
    const input=manifest.regions.original,mask=manifest.regions.mask;
    canvas.width=manifest.width;canvas.height=manifest.height;
    const ctx=canvas.getContext('2d');

    if (dustAbSplitActive) {
      const splitX = Math.max(0, Math.min(canvas.width, Math.round(canvas.width * (dustSplitPos / 100))));
      // Side A (Left): input
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, splitX, canvas.height);
      ctx.clip();
      ctx.drawImage(atlas, ...input, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      // Side B (Right): overlay or mask
      ctx.save();
      ctx.beginPath();
      ctx.rect(splitX, 0, canvas.width - splitX, canvas.height);
      ctx.clip();
      if(dustInputMode==='overlay') {
        ctx.drawImage(atlas,...mask,0,0,canvas.width,canvas.height);
        const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
        for(let i=0;i<pixels.data.length;i+=4) {const on=pixels.data[i]>=128;pixels.data[i]=255;pixels.data[i+1]=0;pixels.data[i+2]=255;pixels.data[i+3]=on?Math.round(255*dustOpacity):0;}
        const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;overlay.getContext('2d').putImageData(pixels,0,0);
        ctx.drawImage(atlas,...input,0,0,canvas.width,canvas.height);ctx.drawImage(overlay,0,0);
      } else if(dustInputMode==='input') {
        ctx.drawImage(atlas,...input,0,0,canvas.width,canvas.height);
      } else {
        ctx.drawImage(atlas,...mask,0,0,canvas.width,canvas.height);
      }
      ctx.restore();

      // Split line
      ctx.fillStyle='rgba(0,0,0,0.7)';
      ctx.fillRect(splitX-2, 0, 4, canvas.height);
      ctx.fillStyle='#ffffff';
      ctx.fillRect(splitX-1, 0, 2, canvas.height);
    } else {
      ctx.drawImage(atlas,...mask,0,0,canvas.width,canvas.height);
      if(dustInputMode==='input') ctx.drawImage(atlas,...input,0,0,canvas.width,canvas.height);
      if(dustInputMode==='overlay') {
        const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
        for(let i=0;i<pixels.data.length;i+=4) {const on=pixels.data[i]>=128;pixels.data[i]=255;pixels.data[i+1]=0;pixels.data[i+2]=255;pixels.data[i+3]=on?Math.round(255*dustOpacity):0;}
        const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;overlay.getContext('2d').putImageData(pixels,0,0);
        ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(atlas,...input,0,0,canvas.width,canvas.height);ctx.drawImage(overlay,0,0);
      }
    }
    canvas.hidden=false;canvas.dataset.dpFrame=String(frame);canvas.dataset.dpQuality='preview';
    const splitLabel = dustAbSplitActive ? `A/B Split (${dustSplitPos}%): Left: ${dustSourceLabel()} ↔ Right: ${dustInputMode==='mask'?'Mask':'Overlay'} · ` : '';
    canvas.closest('figure').querySelector('figcaption').textContent=`${splitLabel}${dustInputMode==='mask'?'Effective mask':dustSourceLabel()+(dustInputMode==='overlay'?' + magenta mask':'')} · frame ${frame+1} · compressed display`;
    const status=document.querySelector('[data-dust-frame-status]');if(status) status.textContent=`Saved mask · frame ${frame+1}/${manifest.frames}. Frame-locked compressed playback; no detection or repair.`;
    dustAlignImages();
    dustWireCanvasInteraction(canvas, paint, false);
  };
  paint();dustPaintFrame=paint;
}
async function dustRun(newMask=false) {
  if(state.running || state.queue_running || dustRunPending) return;
  const action=newMask ? {kind:'mask'} : dustActionState(dustStatus,false);if(!action.kind) return;
  if(newMask && !confirm('Recalculate the mask as a new saved version with the same settings? Existing masks, repairs and the pipeline selection stay unchanged.'+(dustDraft.detector==='dart' ? '\nDART will run GPU inference again, generating both its mask and full restoration.' : '\nDetection will run again.'))) return;
  if(typeof dustMaskEditorFlushSave==='function' && typeof dustMaskEditorIsBusy==='function' && dustMaskEditorIsBusy()) {
    await dustMaskEditorFlushSave();
  }
  const context=dustContext(),values={...dustDraft};
  if(newMask) {values.mask_version=crypto.randomUUID().replaceAll('-','');delete values.edit_signature;delete values.timeline_id;}
  dustRunPending=true;dustFollowDraft=true;
  const host=document.querySelector('[data-dust-catalog]'),button=host?.querySelector('[data-dust-next]');
  const recalculate=host?.querySelector('[data-dust-recalculate]');let started=false;
  if(button) button.disabled=true;
  if(recalculate) recalculate.disabled=true;
  try {
    const result=await postJson('/api/dust-run',{values,kind:action.kind,source:dustMatrix().source,project:state.project_path || ''});
    if(context!==dustContext() || active!=='dust') return;
    if(!result?.ok) throw Error(result?.error || result?.message || 'Processing could not start.');
    started=true;
    if(newMask) {dustDraft=values;dustStatus=null;dustRepairOpen=false;dustInputMode='input';dustInspectionKey='';drawDust(state.stages.find(stage=>stage.key==='dust'));}
    await refresh();
  } catch(error) {if(context===dustContext() && host?.isConnected) host.querySelector('[data-dust-state]').textContent=error.message;}
   finally {dustRunPending=false;if(button?.isConnected) button.disabled=false;if(recalculate?.isConnected) recalculate.disabled=Boolean(state.running || state.queue_running || dustInputBlocked());if(started) dustUpdateStatus?.();}
}
function dustEditorControlsHtml() {
  return `<div class="dust-edit-strip" data-dust-edit-strip>
    <span class="dust-edited-badge" data-dust-edited-badge>Edited frames: 0</span>
    <div class="dust-edit-strip-actions">
      <button type="button" data-dust-prev-edited title="Jump to previous edited frame" disabled>&lsaquo; Prev edited</button>
      <button type="button" data-dust-next-edited title="Jump to next edited frame" disabled>Next edited &rsaquo;</button>
      <button type="button" data-dust-export-provenance title="Export non-destructive provenance &amp; separated layer PNGs (operational edits only, not independent GT)">Export Provenance…</button>
    </div>
    <span class="dust-edit-status" data-dust-edit-status role="status"></span>
  </div>
  <div class="inline-warning dust-compat-banner" data-dust-compat-banner role="alert" hidden><span data-dust-compat-text></span> <button type="button" class="warn" data-dust-compat-reset-all>Reset All</button></div>
  <div class="dust-editor-bar" data-dust-editor-bar hidden>
    <div class="dust-editor-row">
      <div class="dust-tool-group" role="group" aria-label="Mask edit brush tools">
        <button type="button" class="dust-tool-btn" data-dust-tool="scratch_add" aria-pressed="true" title="Add missed scratch pixels"><span class="dust-swatch swatch-human-scratch" aria-hidden="true"></span>Scratch +</button>
        <button type="button" class="dust-tool-btn" data-dust-tool="scratch_reject" aria-pressed="false" title="Reject false-positive scratch pixels">Scratch -</button>
        <button type="button" class="dust-tool-btn" data-dust-tool="dust_add" aria-pressed="false" title="Add missed dust or blotch pixels"><span class="dust-swatch swatch-human-dust" aria-hidden="true"></span>Dust +</button>
        <button type="button" class="dust-tool-btn" data-dust-tool="dust_reject" aria-pressed="false" title="Reject false-positive dust pixels">Dust -</button>
        <button type="button" class="dust-tool-btn" data-dust-tool="reject_all" aria-pressed="false" title="Suppress all automatic proposals under brush">Reject All</button>
        <button type="button" class="dust-tool-btn" data-dust-tool="protect" aria-pressed="false" title="Protect intact film/scene pixels against all repair (highest safety veto)"><span class="dust-swatch swatch-protect" aria-hidden="true"></span>Protect Intact</button>
        <button type="button" class="dust-tool-btn" data-dust-tool="unprotect" aria-pressed="false" title="Remove Protect Intact lock">Unprotect</button>
        <button type="button" class="dust-tool-btn" data-dust-tool="clear_edits" aria-pressed="false" title="Clear all manual edit bits back to raw detector state">Clear Pixel</button>
      </div>
    </div>
    <div class="dust-editor-row">
      <label class="dust-brush-control">Brush radius <input type="range" min="0" max="30" step="1" value="3" data-dust-brush-radius aria-label="Brush radius in pixels"><output data-dust-brush-readout>7 px (r=3)</output></label>
      <div class="dust-history-group">
        <button type="button" data-dust-undo title="Undo stroke (Ctrl+Z)" disabled>Undo</button>
        <button type="button" data-dust-redo title="Redo stroke (Ctrl+Y)" disabled>Redo</button>
        <button type="button" data-dust-mark-reviewed title="Mark frame operationally reviewed (not independent ground truth)">Mark Reviewed</button>
        <button type="button" data-dust-save disabled>Saved</button>
        <button type="button" data-dust-reset-frame title="Reset manual edits on current frame">Reset Frame</button>
        <button type="button" class="warn" data-dust-reset-all title="Clear all manual edits for this timeline">Reset All</button>
      </div>
      <span class="dust-editor-hint">[ / ] size · Ctrl+Z/Y undo/redo · Space+drag pan</span>
    </div>
    <div class="dust-editor-row">
      <div class="dust-layer-group" role="group" aria-label="Mask overlay layers">
        <strong>Layers:</strong>
        <label><input type="checkbox" data-dust-layer="auto" checked> Auto Proposals</label>
        <label><input type="checkbox" data-dust-layer="human" checked> Human Edits</label>
        <label><input type="checkbox" data-dust-layer="rejected" checked> Rejected</label>
        <label><input type="checkbox" data-dust-layer="protect" checked> Protect Intact</label>
        <label data-dust-layer-wrap="vetoed" hidden><input type="checkbox" data-dust-layer="vetoed" checked> E2 Vetoed</label>
      </div>
      <span class="dust-frame-stats" data-dust-frame-stats></span>
    </div>
    <div class="dust-layer-legend" aria-label="Mask color legend">
      <span class="dust-legend-item"><span class="dust-swatch swatch-auto-scratch" aria-hidden="true"></span>Auto Scratch</span>
      <span class="dust-legend-item"><span class="dust-swatch swatch-auto-dust" aria-hidden="true"></span>Auto Dust</span>
      <span class="dust-legend-item"><span class="dust-swatch swatch-auto-uncertain" aria-hidden="true"></span>Auto Uncertain</span>
      <span class="dust-legend-item" data-dust-legend-vetoed hidden><span class="dust-swatch swatch-e2-vetoed" aria-hidden="true"></span>E2 Vetoed Intact</span>
      <span class="dust-legend-item"><span class="dust-swatch swatch-human-scratch" aria-hidden="true"></span>Human Scratch +</span>
      <span class="dust-legend-item"><span class="dust-swatch swatch-human-dust" aria-hidden="true"></span>Human Dust +</span>
      <span class="dust-legend-item"><span class="dust-swatch swatch-rejected" aria-hidden="true"></span>Rejected FP</span>
      <span class="dust-legend-item"><span class="dust-swatch swatch-protect" aria-hidden="true"></span>Protect Intact (Veto)</span>
    </div>
  </div>`;
}

function drawDust(st) {
   const fineTuningOpen=document.querySelector('.dust-fine-tuning')?.open || false;
  const openDialog=document.querySelector('.comparison-inspector-dialog[open]');
  const scroll=document.querySelector('.comparison-inspector-content')?.scrollTop || 0;
  const focus=document.activeElement?.dataset?.dust;
  const project=dustContext();
  const changedProject=!dustDraft || dustProject!==project;
  if(changedProject) {
    if(typeof dustMaskEditorResetProject==='function') dustMaskEditorResetProject();
    dustDraft={...state.settings.dust};if(dustSourceMode==='original') dustDraft.analysis_source='original';else delete dustDraft.analysis_source;dustProject=project;dustView='result';dustInputMode='input';dustRepairOpen=false;dustStatus=null;dustInspectionKey='';dustInspectEpoch++;
  }
  const epoch=++dustEpoch;
  const previous=denoiseWorkspaces.has('dust') ? denoiseWorkspaceSnapshot() : null;
  denoiseWorkspaceDeactivate();
  if(dustWorkspaceCatalog!==dustCatalog()) denoiseWorkspaces.delete('dust');
  dustWorkspaceCatalog=dustCatalog();
  drawDenoise(st,state.settings.dust,state.expected_outputs?.dust || [],{percent:0,label:state.running?'Processing':'Idle'});
  if(previous && !changedProject) {denoiseWorkspaceFor().restoreComparison(previous);denoiseWorkspaceRefresh();denoiseWorkspaceFor().seekFrame(previous.frame);denoiseWorkspaceFor().setViewport({zoom:previous.zoom,pan:previous.pan,native:previous.native,grid:previous.grid});}
  const host=document.querySelector('.comparison-workbench');host.classList.add('dust-editor');host.dataset.dustCatalog=dustCatalog();
  dustInspectionKey='';dustInspectEpoch++;dustPaintFrame=null;
   host.querySelector('#denoiseVariantHeading').textContent='Saved masks & repairs';
  host.querySelector('[data-dm-generation]')?.remove();
  host.querySelector('#denoiseCopyVariant')?.remove();
  const use=host.querySelector('#denoiseUseVariant');if(use) use.hidden=true;
  host.querySelector('[onclick="runStage(\'dust\')"]')?.remove();
  const footer=host.querySelector('.comparison-inspector-footer');
  const actions=document.createElement('div');actions.className='dust-next-action';
    actions.innerHTML='<p data-dust-state role="status">Checking saved mask…</p><button type="button" class="primary" data-dust-next disabled>Checking mask…</button><button type="button" data-dust-recalculate disabled>Recalculate mask…</button><a data-dust-open-mask hidden target="_blank" rel="noopener">Open saved mask</a><button type="button" class="primary" data-dust-create-repair hidden>Create repair…</button>';
   footer.prepend(actions);actions.querySelector('[data-dust-next]').onclick=()=>dustRun();
   actions.querySelector('[data-dust-recalculate]').onclick=()=>dustRun(true);
  actions.querySelector('[data-dust-create-repair]').onclick=()=>{dustRepairOpen=true;drawDust(st);};
  host.querySelector('[data-dust-close-repair]')?.addEventListener('click',()=>{dustRepairOpen=false;drawDust(st);});
  host.querySelector('[data-dust-source]').onchange=event=>dustSelectSource(event.target.value);
  const details=document.createElement('details');details.className='comparison-input-context';
  details.innerHTML='<summary>Mask measurements & technical details</summary><p data-dust-candidate-context></p><div data-dust-measurements></div><div data-dust-files></div><p>Counts describe proposals and changes, not detection accuracy. Models use BGR8 copies; native 10-bit preservation is verified separately. DART full restoration is diagnostic only.</p>';
  host.querySelector('[data-comparison-analysis]').append(details);

   const channels = [
    ['input', 'Input'],
    ['overlay', 'Mask overlay'],
    ['mask', 'Combined Defect'],
    ['scratch', 'Scratch'],
    ['dust', 'Dust'],
    ['uncertain', 'Uncertain'],
    ['vetoed', 'Vetoed']
  ];
  const channelButtons = channels.map(([key, label]) => {
     if(!['input','overlay','mask'].includes(key)) return '';
     return `<button type="button" data-dust-input="${key}">${key==='overlay'?'Overlay':key==='mask'?'Mask':label}</button>`;
  }).join('');

  const viewer=document.createElement('section');viewer.className='dust-mask-review';
  viewer.innerHTML=`<div class="dust-view-switch" role="group" aria-label="Stage input display"><strong>Stage input</strong>${channelButtons}<label data-dust-opacity hidden>Overlay opacity <input type="range" min="0" max="100" value="${dustOpacity*100}" aria-label="Mask overlay opacity"><output>${Math.round(dustOpacity*100)}%</output></label><div class="dust-ab-controls"><button type="button" data-dust-ab-toggle class="${dustAbSplitActive?'is-active':''}" aria-pressed="${String(dustAbSplitActive)}">${dustAbSplitActive?'A/B Split: ON':'A/B Split View'}</button><label data-dust-ab-slider-wrap ${dustAbSplitActive?'':'hidden'}>Split <input type="range" min="0" max="100" value="${dustSplitPos}" class="dust-split-slider" aria-label="A/B split position"><output>${dustSplitPos}%</output></label></div><button type="button" data-dust-edit-toggle aria-pressed="false" title="Edit defect &amp; protect masks on exact frame still">Edit Mask</button></div><div class="dust-comparison-images"><div data-dust-mask-view hidden><p data-dust-frame-status role="status">Calculate a mask to inspect marked areas here.</p><button type="button" data-dust-retry hidden>Retry mask frame</button><div class="dust-images" data-dust-images></div></div></div>`;
  host.querySelector('#denoiseComparePlayers').before(viewer);
   viewer.querySelector('.dust-view-switch strong').textContent='View';
   const sourceCaption=document.createElement('p');sourceCaption.className='dust-source-caption';sourceCaption.textContent='Source: '+dustSourceLabel();viewer.prepend(sourceCaption);
   const channelControl=document.createElement('label');channelControl.className='dust-channel-control';
   channelControl.innerHTML='Channel <select data-dust-channel><option value="mask">Combined defects</option><option value="scratch">Scratches</option><option value="dust">Dust</option><option value="uncertain">Unclassified</option><option value="vetoed">E2 Vetoed (intact)</option></select>';
   viewer.querySelector('.dust-view-switch').append(channelControl);
   const originNote=document.createElement('p');originNote.dataset.dustChannelOrigin='';originNote.className='dust-channel-origin';viewer.querySelector('.dust-view-switch').after(originNote);
   originNote.insertAdjacentHTML('afterend', dustEditorControlsHtml());
  viewer.querySelector('.dust-view-switch').setAttribute('aria-label',dustSourceLabel()+' display');
  viewer.before(viewer.querySelector('[data-dust-frame-status]'));
  viewer.querySelector('.dust-comparison-images').append(host.querySelector('#denoiseComparePlayers'));
  if(typeof dustMaskEditorSyncFromState==='function') dustMaskEditorSyncFromState(state);
  if(typeof dustMaskEditorMountUi==='function') dustMaskEditorMountUi(viewer);
  dustAlignmentObserver?.disconnect();
  if(typeof ResizeObserver==='function') {dustAlignmentObserver=new ResizeObserver(dustAlignImages);dustAlignmentObserver.observe(viewer);}
   host.querySelectorAll('[data-dust-input]').forEach(button=>button.onclick=()=>{dustInputMode=button.dataset.dustInput==='mask'?dustChannel:button.dataset.dustInput;dustInspectionKey='';if(['scratch','dust','uncertain','vetoed'].includes(dustInputMode)) denoiseWorkspaceSetExact(true);dustSetView();});
   channelControl.querySelector('select').onchange=event=>{dustChannel=event.target.value;dustInputMode=dustChannel;dustInspectionKey='';if(dustChannel!=='mask') denoiseWorkspaceSetExact(true);dustSetView();};
  host.querySelector('[data-dust-opacity] input').oninput=event=>{dustOpacity=Number(event.target.value)/100;host.querySelector('[data-dust-opacity] output').textContent=Math.round(dustOpacity*100)+'%';dustPaintFrame?.();};
  host.querySelector('[data-dust-retry]').onclick=()=>{if(!denoiseWorkspaceSnapshot().exact) {denoiseWorkspaceFor().playPause();return;}dustInspectionKey='';dustInvalidateInspection();};

  const abToggle=host.querySelector('[data-dust-ab-toggle]');
  if(abToggle) abToggle.onclick=()=>{
    dustAbSplitActive=!dustAbSplitActive;
    abToggle.classList.toggle('is-active', dustAbSplitActive);
    abToggle.setAttribute('aria-pressed', String(dustAbSplitActive));
    abToggle.textContent=dustAbSplitActive?'A/B Split: ON':'A/B Split View';
    const wrap=host.querySelector('[data-dust-ab-slider-wrap]');
    if(wrap) wrap.hidden=!dustAbSplitActive;
    dustPaintFrame?.();
    dustInvalidateInspection();
  };
  const abSlider=host.querySelector('[data-dust-ab-slider-wrap] input');
  if(abSlider) abSlider.oninput=e=>{
    dustSplitPos=Number(e.target.value);
    const out=host.querySelector('[data-dust-ab-slider-wrap] output');
    if(out) out.textContent=dustSplitPos+'%';
    dustPaintFrame?.();
  };

  const valid=()=>epoch===dustEpoch && host.isConnected && project===dustContext() && active==='dust';
  let statusEpoch=0;
  async function status() {
    if(typeof dustMaskEditorIsBusy==='function' && dustMaskEditorIsBusy()) return;
    const values=JSON.stringify(dustDraft),request=++statusEpoch;
    try {
      const result=await postJson('/api/dust-status',{values:{...dustDraft},source:dustMatrix().source,project:state.project_path || ''});
      if(!valid() || request!==statusEpoch || values!==JSON.stringify(dustDraft)) return;
      if(typeof dustMaskEditorIsBusy==='function' && dustMaskEditorIsBusy()) return;
      if(!result?.ok) throw Error(result?.error || result?.message || 'Could not check this mask.');
      if(dustAvailableFrames(result) && !dustAvailableFrames(dustStatus)) dustInspectionKey='';
      dustStatus=result;
      if(typeof dustMaskEditorSyncFromState==='function') dustMaskEditorSyncFromState(state);
      if(dustDraft.detector==='dart' && result.mask_values) {
        for(const key of ['dart_repo','dart_python']) if(!dustDraft[key]) dustDraft[key]=result.mask_values[key];
      }
      const action=dustActionState(result,state.running || state.queue_running || dustRunPending);
      host.querySelector('[data-dust-state]').textContent=state.dust_matrix?.stop_requested ? 'Stop requested. Waiting for the current operation.' : action.message;
      const next=host.querySelector('[data-dust-next]');next.textContent=action.label;next.hidden=!action.kind && !state.running;next.disabled=!action.kind || Boolean(state.running || dustRunPending || dustInputBlocked());
      if(result.installation?.available===false && action.kind==='mask') {next.disabled=true;next.textContent='Set up DART first';host.querySelector('[data-dust-state]').textContent=result.installation.message;}
      const availableFrames=dustAvailableFrames(result);
      const resultFiles=Object.keys(result.files || {}).length ? result.files : (result.base_files || {});
      const recalculate=host.querySelector('[data-dust-recalculate]');recalculate.hidden=!availableFrames;
      recalculate.disabled=Boolean(state.running || state.queue_running || dustRunPending || dustInputBlocked());
      const row=dustMatrix().variants?.find(v=>dustMatches(v.values));
      const create=host.querySelector('[data-dust-create-repair]');create.hidden=dustSourceMode==='original' || !availableFrames || dustRepairOpen;create.disabled=Boolean(state.running);
      const open=host.querySelector('[data-dust-open-mask]');open.hidden=!resultFiles?.['effective_mask.mkv'];if(!open.hidden) open.href='/media?path='+encodeURIComponent(resultFiles['effective_mask.mkv']);
      const use=host.querySelector('#denoiseUseVariant');if(use) use.hidden=!row?.repair_complete || denoiseSelectedVariant().method!==row.method;
      if(use && !dustRepairOpen) use.hidden=true;
      host.querySelector('[data-dust-draft]').textContent=row ? 'Settings match a saved variant. Pipeline selection is unchanged.' : 'Draft settings. Saved variants and pipeline selection are unchanged.';
      const context=host.querySelector('[data-review-context]');if(context && (!row || row.method!==denoiseSelectedVariant().method)) context.textContent=`Inspecting: draft mask · ${dustLabel(dustDraft)} · Pipeline selection unchanged`;
      host.querySelectorAll('[data-method-viewed]').forEach(badge=>{if(!row || row.method!==denoiseSelectedVariant().method) badge.hidden=true;});
      host.querySelectorAll('.comparison-method-row').forEach(element=>{
        const candidate=dustMatrix().variants?.find(v=>v.method===element.querySelector('[data-denoise-cell]')?.dataset.method);
        const label=element.querySelector('.denoise-variant-cell-state');
        if(label && candidate) label.textContent=candidate.repair_complete?'Repair ready':candidate.job?.state==='stopped'?'Stopped · resume':candidate.job?.state==='failed'?'Failed · inspect':'Mask ready';
      });
      host.querySelector('[data-dust-files]').innerHTML=Object.entries(resultFiles || {}).filter(([name])=>/csv|mkv/.test(name)).map(([name,path])=>`<a href="/media?path=${encodeURIComponent(path)}" target="_blank" rel="noopener">${esc(name)}</a>`).join(' · ');
      host.querySelector('[data-dust-candidate-context]').textContent=row ? `Effective mask SHA-256: ${row.mask_sha256}. Saved parameters: ${JSON.stringify(row.values)}` : 'No saved mask for this draft.';
      if(dustFollowDraft && row && !state.running && !dustRunPending) {
        dustFollowDraft=false;inspectComparisonVariant(row.method,'settings');if(!row.repair_complete) dustInputMode='overlay';
      }
      const editActive = typeof dustMaskEditorIsEditActive==='function' && dustMaskEditorIsEditActive();
      if((!row?.repair_complete || editActive) && dustView==='result') dustView='mask';
      dustSetView(dustView);
    } catch(error) {if(valid() && request===statusEpoch) host.querySelector('[data-dust-state]').textContent=error.message;}
  }
  dustUpdateStatus=status;

  host.querySelectorAll('[data-dust]').forEach(input=>input.onchange=()=>{
    if(typeof dustMaskEditorFlushSave==='function') void dustMaskEditorFlushSave();
    const k = input.dataset.dust, v = input.type==='checkbox'?String(input.checked):input.value;
    dustDraft[k]=v;
    if(k==='preset') {
      delete dustDraft.edit_signature;
      delete dustDraft.timeline_id;
      if(v!=='custom' && DUST_PRESET_MAP[v]) {
        const pCfg = DUST_PRESET_MAP[v];
        dustDraft.detector = pCfg.detector;
        dustDraft.sensitivity = pCfg.sensitivity;
        dustDraft.dilate = pCfg.dilate;
        dustDraft.threshold = pCfg.threshold;
        if(pCfg.repair) dustDraft.repair = pCfg.repair;
        dustDraft.preset = v;
      }
    } else if(k==='detector') {
      delete dustDraft.edit_signature;
      delete dustDraft.timeline_id;
      if(v!=='dart' && dustDraft.repair==='dart') dustDraft.repair='telea';
      const curP = DUST_PRESET_MAP[dustDraft.preset];
      const curDet = curP?.detector === 'temporal' ? 'farneback_3f' : curP?.detector;
      const newDet = v === 'temporal' ? 'farneback_3f' : v;
      if (!curP || curDet !== newDet) {
        const defPresetName = DETECTOR_DEFAULT_PRESETS[v] || DETECTOR_DEFAULT_PRESETS[newDet];
        if (defPresetName && DUST_PRESET_MAP[defPresetName]) {
          const pCfg = DUST_PRESET_MAP[defPresetName];
          dustDraft.detector = pCfg.detector;
          dustDraft.sensitivity = pCfg.sensitivity;
          dustDraft.dilate = pCfg.dilate;
          dustDraft.threshold = pCfg.threshold;
          if (pCfg.repair) dustDraft.repair = pCfg.repair;
          dustDraft.preset = defPresetName;
        } else {
          dustDraft.preset = 'custom';
        }
      }
    } else if(['sensitivity','dilate','threshold'].includes(k)) {
      delete dustDraft.edit_signature;
      delete dustDraft.timeline_id;
      const match = Object.entries(DUST_PRESET_MAP).find(([pName, pCfg]) =>
        pName !== 'custom' &&
        pCfg.detector === (dustDraft.detector || 'farneback_3f') &&
        Math.abs(Number(dustDraft.sensitivity ?? 0.65) - pCfg.sensitivity) < 1e-4 &&
        Number(dustDraft.dilate ?? 0) === pCfg.dilate &&
        Number(dustDraft.threshold ?? 160) === pCfg.threshold
      );
      dustDraft.preset = match ? match[0] : 'custom';
    }
    dustStatus=null;dustView='mask';dustInspectionKey='';dustInspectEpoch++;drawDust(st);
  });
  host.addEventListener('click',event=>{
    const rowElement=event.target.closest('.comparison-method-row');if(!rowElement) return;
    const row=dustMatrix().variants.find(v=>v.method===rowElement.querySelector('[data-denoise-cell]')?.dataset.method);
    if(!row || event.target.closest('[data-denoise-cell]')) return;
    if(typeof dustMaskEditorFlushSave==='function') void dustMaskEditorFlushSave();
    dustDraft={...row.values};dustStatus=null;dustRepairOpen=Boolean(row.repair_complete);dustInputMode=row.repair_complete?'input':'overlay';dustInspectionKey='';drawDust(st);
  });
  if(openDialog && !changedProject) showComparisonInspector();
  const inspector=host.querySelector('.comparison-inspector-content');if(inspector) inspector.scrollTop=scroll;
   const fineTuning=host.querySelector('.dust-fine-tuning');if(fineTuning) fineTuning.open=fineTuningOpen;
   if(focus) host.querySelector(`[data-dust="${focus}"]`)?.focus({preventScroll:true});
   status();
}
function dustInspectMediaUrl(result, key, preferAutoProposal = false) {
  if (preferAutoProposal) {
    if (result?.auto_proposal_urls?.[key]) return result.auto_proposal_urls[key];
    if (result?.auto_proposal_images?.[key]) return '/media?path=' + encodeURIComponent(result.auto_proposal_images[key]);
    if (result?.base_urls?.[key]) return result.base_urls[key];
    if (result?.base_images?.[key]) return '/media?path=' + encodeURIComponent(result.base_images[key]);
  }
  if (key === 'vetoed_mask.mkv' && result?.vetoed_url) return result.vetoed_url;
  if (result?.urls?.[key]) return result.urls[key];
  if (result?.images?.[key]) return '/media?path=' + encodeURIComponent(result.images[key]);
  return '';
}

function dustInvalidateInspection() {
  if(active!=='dust' || dustView!=='mask') return;
  const host=document.querySelector('[data-dust-mask-view]');if(!host) return;
  const snapshot=denoiseWorkspaceSnapshot(),frame=snapshot.frame;
  if(!snapshot.exact) {
    if(typeof dustMaskEditorFlushSave==='function' && typeof dustMaskEditorIsBusy==='function' && dustMaskEditorIsBusy()) {
      void dustMaskEditorFlushSave();
    }
    dustInspectEpoch++;dustInspectionKey='';
    const canvas=dustMaskCanvas();
    canvas?.classList.toggle('is-awaiting-frame',Boolean(snapshot.loading || snapshot.error || canvas.dataset.dpFrame!==String(frame)));
    const status=document.querySelector('[data-dust-frame-status]'),retry=host.querySelector('[data-dust-retry]');
    if(status && (snapshot.error || snapshot.loading)) status.textContent=snapshot.error ? snapshot.error+' Retry mask playback or choose Exact still.' : 'Preparing frame-locked saved mask playback…';
    retry.hidden=!snapshot.error;
    return;
  }
  const inspectionKey=()=>JSON.stringify([dustContext(),dustDraft,denoiseWorkspaceSnapshot().frame,dustInputMode,dustAbSplitActive,dustSplitPos]);
  const key=inspectionKey();
  if(key===dustInspectionKey) return;
  if(typeof dustMaskEditorFlushSave==='function' && typeof dustMaskEditorIsBusy==='function' && dustMaskEditorIsBusy()) {
    void dustMaskEditorFlushSave();
  }
  dustInspectionKey=key;const epoch=++dustInspectEpoch;
  const images=host.querySelector('[data-dust-images]'),message=document.querySelector('[data-dust-frame-status]');
  const retry=host.querySelector('[data-dust-retry]');retry.hidden=true;
  images.replaceChildren();
  dustPaintFrame=null;
  if(!dustAvailableFrames(dustStatus)) {message.textContent='No mask for these settings. Choose Calculate mask to inspect marked areas.';return;}
  message.textContent=`Decoding mask and ${dustSourceLabel().toLowerCase()} at frame ${frame+1}…`;

  const isUint8 =
    typeof dustMaskEditor !== 'undefined' &&
    dustMaskEditor &&
    (typeof dustMaskEditorIsUint8 === 'function' ? dustMaskEditorIsUint8(dustMaskEditor.bitmask) : Boolean(dustMaskEditor.bitmask));
  const alreadyLoaded =
    typeof dustMaskEditor !== 'undefined' &&
    dustMaskEditor &&
    !dustMaskEditor.loading &&
    dustMaskEditor.frame === frame &&
    dustMaskEditor.analysisSource === dustSourceMode &&
    isUint8;
  const alreadyLoadingSame =
    typeof dustMaskEditor !== 'undefined' &&
    dustMaskEditor &&
    dustMaskEditor.loading &&
    dustMaskEditor.loadingFrame === frame &&
    dustMaskEditor.analysisSource === dustSourceMode;
  const needsEditLoad =
    typeof dustMaskEditorLoadFrame === 'function' &&
    typeof dustMaskEditor !== 'undefined' &&
    !alreadyLoaded &&
    !alreadyLoadingSame;
  const editLoadPromise = needsEditLoad ? dustMaskEditorLoadFrame(frame, dustSourceMode) : Promise.resolve(null);

  postJson('/api/dust-inspect',{values:{...dustDraft},source:dustMatrix().source,project:state.project_path || '',frame}).then(result=>{
    if(epoch!==dustInspectEpoch || !host.isConnected || dustView!=='mask' || key!==inspectionKey()) return;
    if(!result?.ok) throw Error(result?.error || result?.message || 'Mask frame unavailable.');
    let figTitle = '';
    const orig = channelOrigin(dustInputMode, dustDraft.detector);
    const origBadge = orig ? ` [${orig}]` : '';
    if (dustInputMode === 'mask') figTitle = `Combined Defect${origBadge} · white marks effective areas`;
    else if (dustInputMode === 'scratch') figTitle = `Scratch channel${origBadge} · detected & edited vertical scratches`;
    else if (dustInputMode === 'dust') figTitle = `Dust channel${origBadge} · detected & edited dust & blotches`;
    else if (dustInputMode === 'uncertain') figTitle = `Uncertain channel${origBadge} · unclassified defect proposals`;
    else if (dustInputMode === 'vetoed') figTitle = `E2 Vetoed${origBadge} · likely intact scene structure`;
    else if (dustInputMode === 'overlay') figTitle = dustSourceLabel() + ' + multi-layer mask overlay';
    else figTitle = dustSourceLabel();

    if (dustAbSplitActive) {
      figTitle = `A/B Split (${dustSplitPos}%): Left: ${dustSourceLabel()} ↔ Right: ${figTitle.split('·')[0].trim()}`;
    }

    images.innerHTML=`<figure><figcaption>${figTitle} · frame ${frame+1}</figcaption><div class="dust-image-viewport"><div class="dust-image-layer"><canvas hidden aria-label="${dustSourceLabel()} mask inspection"></canvas></div></div></figure>`;
    const maskUrl = dustInspectMediaUrl(result, 'effective_mask.mkv', false);
    const rawCombinedUrl = dustInspectMediaUrl(result, 'mask.mkv', true) || maskUrl;
    const inputUrl = dustInspectMediaUrl(result, 'input.mkv', false);
    const scratchUrl = dustInspectMediaUrl(result, 'scratch_mask.mkv', true);
    const dustUrl = dustInspectMediaUrl(result, 'dust_mask.mkv', true);
    const uncertainUrl = dustInspectMediaUrl(result, 'uncertain_mask.mkv', true);
    const vetoedUrl = dustInspectMediaUrl(result, 'vetoed_mask.mkv', true);

    const mask=new Image();mask.src=maskUrl;
    const input=new Image();input.src=inputUrl;
    const rawCombinedImg = rawCombinedUrl && rawCombinedUrl !== maskUrl ? new Image() : null;
    if(rawCombinedImg) rawCombinedImg.src=rawCombinedUrl;
    const scratch=scratchUrl ? new Image() : null;
    if(scratch) scratch.src=scratchUrl;
    const dustImg=dustUrl ? new Image() : null;
    if(dustImg) dustImg.src=dustUrl;
    const uncertainImg=uncertainUrl ? new Image() : null;
    if(uncertainImg) uncertainImg.src=uncertainUrl;
    const vetoedImg=vetoedUrl ? new Image() : null;
    if(vetoedImg) vetoedImg.src=vetoedUrl;

    const extractedPlanes = {};
    let viewportMounted = false;

    input.onerror=mask.onerror=()=>{if(epoch===dustInspectEpoch && host.isConnected) {message.textContent='Could not load this mask frame. Retry the frame.';retry.hidden=false;}};
    const paint=()=>{
      if(epoch!==dustInspectEpoch || !mask.naturalWidth || !input.naturalWidth) return;
      const canvas=images.querySelector('canvas');
      if(!canvas) return;
      const w = mask.naturalWidth, h = mask.naturalHeight;
      if(canvas.width!==w || canvas.height!==h) {canvas.width=w;canvas.height=h;}

      const editActive = typeof dustMaskEditorIsEditActive==='function' && dustMaskEditorIsEditActive();
      const canUseEditorRenderer = typeof dustMaskEditorAttachFrameInspection==='function' && typeof dustExtractBinaryPlane==='function';

      if(canUseEditorRenderer) {
        if(!extractedPlanes.mask) extractedPlanes.mask = dustExtractBinaryPlane(mask, w, h);
        if(rawCombinedImg && rawCombinedImg.naturalWidth && !extractedPlanes.combined) extractedPlanes.combined = dustExtractBinaryPlane(rawCombinedImg, w, h);
        if(scratch && scratch.naturalWidth && !extractedPlanes.scratch) extractedPlanes.scratch = dustExtractBinaryPlane(scratch, w, h);
        if(dustImg && dustImg.naturalWidth && !extractedPlanes.dust) extractedPlanes.dust = dustExtractBinaryPlane(dustImg, w, h);
        if(uncertainImg && uncertainImg.naturalWidth && !extractedPlanes.uncertain) extractedPlanes.uncertain = dustExtractBinaryPlane(uncertainImg, w, h);
        if(vetoedImg && vetoedImg.naturalWidth && !extractedPlanes.vetoed) extractedPlanes.vetoed = dustExtractBinaryPlane(vetoedImg, w, h);

        const editorDustAddCount = typeof dustMaskEditor !== 'undefined' && dustMaskEditor?.frameStats?.dust_add_pixels ? dustMaskEditor.frameStats.dust_add_pixels : 0;
        const editorScratchAddCount = typeof dustMaskEditor !== 'undefined' && dustMaskEditor?.frameStats?.scratch_add_pixels ? dustMaskEditor.frameStats.scratch_add_pixels : 0;
        const channelMissing =
          (dustInputMode === 'scratch' && !(scratch && scratch.naturalWidth) && editorScratchAddCount === 0) ||
          (dustInputMode === 'dust' && !(dustImg && dustImg.naturalWidth) && editorDustAddCount === 0) ||
          (dustInputMode === 'uncertain' && !(uncertainImg && uncertainImg.naturalWidth)) ||
          (dustInputMode === 'vetoed' && !(vetoedImg && vetoedImg.naturalWidth));

        if(channelMissing) {
          const ctx=canvas.getContext('2d');
          const msg = dustInputMode==='scratch' ? 'Scratch channel unavailable' :
            dustInputMode==='dust' ? 'Dust channel unsupported for this detector' :
            dustInputMode==='uncertain' ? 'Uncertain channel unsupported for this detector' :
            'Vetoed layer unavailable for this detector';
          if(dustAbSplitActive) {
            const splitX = Math.max(0, Math.min(w, Math.round(w * (dustSplitPos / 100))));
            ctx.save();ctx.beginPath();ctx.rect(0,0,splitX,h);ctx.clip();ctx.drawImage(input,0,0);ctx.restore();
            ctx.save();ctx.beginPath();ctx.rect(splitX,0,w-splitX,h);ctx.clip();
            ctx.fillStyle='#111';ctx.fillRect(splitX,0,w-splitX,h);ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText(msg,splitX+20,40);
            ctx.restore();
            ctx.fillStyle='rgba(0,0,0,0.7)';ctx.fillRect(splitX-2,0,4,h);ctx.fillStyle='#ffffff';ctx.fillRect(splitX-1,0,2,h);
          } else {
            ctx.fillStyle='#111';ctx.fillRect(0,0,w,h);ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText(msg,20,40);
          }
        } else {
          dustMaskEditorAttachFrameInspection(canvas, {
            frame,
            width: w,
            height: h,
            baseImage: input,
            mode: dustInputMode,
            autoScratch: extractedPlanes.scratch || null,
            autoDust: extractedPlanes.dust || null,
            autoUncertain: extractedPlanes.uncertain || null,
            autoVetoed: extractedPlanes.vetoed || null,
            autoCombined: extractedPlanes.combined || extractedPlanes.mask || null,
            opacity: dustOpacity,
            splitActive: dustAbSplitActive,
            splitPos: dustSplitPos,
          });
        }
      } else {
        const ctx=canvas.getContext('2d');
        ctx.drawImage(mask,0,0);
        const pixels=ctx.getImageData(0,0,canvas.width,canvas.height);
        for(let i=0;i<pixels.data.length;i+=4) {const on=pixels.data[i]>=128;pixels.data[i]=255;pixels.data[i+1]=0;pixels.data[i+2]=255;pixels.data[i+3]=on?Math.round(255*dustOpacity):0;}
        ctx.putImageData(pixels,0,0);
        const overlay=document.createElement('canvas');overlay.width=canvas.width;overlay.height=canvas.height;overlay.getContext('2d').putImageData(pixels,0,0);
        ctx.clearRect(0,0,canvas.width,canvas.height);

        if (dustAbSplitActive) {
          const splitX = Math.max(0, Math.min(canvas.width, Math.round(canvas.width * (dustSplitPos / 100))));
          ctx.save();
          ctx.beginPath();
          ctx.rect(0, 0, splitX, canvas.height);
          ctx.clip();
          ctx.drawImage(input, 0, 0);
          ctx.restore();

          ctx.save();
          ctx.beginPath();
          ctx.rect(splitX, 0, canvas.width - splitX, canvas.height);
          ctx.clip();
          if(dustInputMode==='mask') {
            ctx.drawImage(mask,0,0);
          } else if(dustInputMode==='scratch') {
            if(scratch && scratch.naturalWidth) ctx.drawImage(scratch,0,0);
             else {ctx.fillStyle='#111';ctx.fillRect(splitX,0,canvas.width-splitX,canvas.height);ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Scratch channel unavailable',splitX+20,40);}
          } else if(dustInputMode==='dust') {
            if(dustImg && dustImg.naturalWidth) {
              ctx.drawImage(dustImg,0,0);
            } else {
              ctx.fillStyle='#111';ctx.fillRect(splitX,0,canvas.width-splitX,canvas.height);
              ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Dust channel unsupported',splitX+20,40);
            }
          } else if(dustInputMode==='uncertain') {
            if(uncertainImg && uncertainImg.naturalWidth) {
              ctx.drawImage(uncertainImg,0,0);
            } else {
              ctx.fillStyle='#111';ctx.fillRect(splitX,0,canvas.width-splitX,canvas.height);
              ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Uncertain channel unsupported',splitX+20,40);
            }
          } else if(dustInputMode==='vetoed') {
            if(vetoedImg && vetoedImg.naturalWidth) ctx.drawImage(vetoedImg,0,0);
            else {ctx.fillStyle='#111';ctx.fillRect(splitX,0,canvas.width-splitX,canvas.height);ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Vetoed layer unavailable',splitX+20,40);}
          } else {
            ctx.drawImage(input,0,0);
            if(dustInputMode==='overlay') ctx.drawImage(overlay,0,0);
          }
          ctx.restore();

          ctx.fillStyle='rgba(0,0,0,0.7)';
          ctx.fillRect(splitX-2, 0, 4, canvas.height);
          ctx.fillStyle='#ffffff';
          ctx.fillRect(splitX-1, 0, 2, canvas.height);
        } else {
          if(dustInputMode==='mask') {
            ctx.drawImage(mask,0,0);
          } else if(dustInputMode==='scratch') {
            if(scratch && scratch.naturalWidth) ctx.drawImage(scratch,0,0);
             else {ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Scratch channel unavailable',20,40);}
          } else if(dustInputMode==='dust') {
            if(dustImg && dustImg.naturalWidth) {
              ctx.drawImage(dustImg,0,0);
            } else {
              ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);
              ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Dust channel unsupported for this detector',20,40);
            }
          } else if(dustInputMode==='uncertain') {
            if(uncertainImg && uncertainImg.naturalWidth) {
              ctx.drawImage(uncertainImg,0,0);
            } else {
              ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);
              ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Uncertain channel unsupported for this detector',20,40);
            }
          } else if(dustInputMode==='vetoed') {
            if(vetoedImg && vetoedImg.naturalWidth) ctx.drawImage(vetoedImg,0,0);
            else {ctx.fillStyle='#111';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#eee';ctx.font='16px sans-serif';ctx.fillText('Vetoed layer unavailable for this detector',20,40);}
          } else {
            ctx.drawImage(input,0,0);
            if(dustInputMode==='overlay') ctx.drawImage(overlay,0,0);
          }
        }
      }

      canvas.hidden=false;
      canvas.dataset.dpFrame=String(frame);canvas.dataset.dpQuality='exact';
      dustWireCanvasInteraction(canvas, paint, true);

      if(!viewportMounted) {
        viewportMounted = true;
        const viewport=denoiseWorkspaceSnapshot();
        denoiseWorkspaceFor().setViewport({zoom:viewport.zoom,pan:viewport.pan,native:viewport.native,grid:viewport.grid});
      }
      dustPaintFrame=paint;
      dustAlignImages();
    };
    mask.onload=paint;input.onload=paint;
    if(rawCombinedImg) rawCombinedImg.onload=paint;
    if(scratch) scratch.onload=paint;
    if(dustImg) dustImg.onload=paint;
    if(uncertainImg) uncertainImg.onload=paint;
    if(vetoedImg) vetoedImg.onload=paint;
    void editLoadPromise.then(() => { if(epoch===dustInspectEpoch && host.isConnected) paint(); });
    if(mask.complete && input.complete && (!rawCombinedImg || rawCombinedImg.complete) && (!scratch || scratch.complete) && (!dustImg || dustImg.complete) && (!uncertainImg || uncertainImg.complete) && (!vetoedImg || vetoedImg.complete)) paint();
    message.textContent=`${dustSourceLabel()} · saved mask · exact frame ${frame+1}/${dustAvailableFrames(result) || dustAvailableFrames(dustStatus)}. Display only.`;
    const measures=document.querySelector('[data-dust-measurements]');if(measures) measures.textContent=JSON.stringify(result.measurements || {});
  }).catch(error=>{if(epoch===dustInspectEpoch && host.isConnected) {message.textContent=error.message;retry.hidden=false;}});
}
