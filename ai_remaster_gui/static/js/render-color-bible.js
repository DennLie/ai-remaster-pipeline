// Color Bible Workbench frontend renderer for ARP

let cbState = {
  data: null,
  activeFilmId: '',
  activeTab: 'character', // 'character' | 'object' | 'location'
  selectedEntityId: null,
  selectedVariantId: null,
  loading: false,
  error: null,
  newCharOpen: false,
  newVarOpen: false,
  newObjOpen: false,
  newLocOpen: false,
  newRefOpen: false,
};

async function loadColorBibleData(filmId = '', force = false) {
  if (cbState.loading && !force) return;
  cbState.loading = true;
  cbState.error = null;

  try {
    const params = new URLSearchParams();
    if (filmId) params.set('film_id', filmId);
    const url = '/api/color-bible/state' + (params.toString() ? '?' + params.toString() : '');
    const res = await api(url);
    if (!res.ok) {
      throw new Error(res.error || 'Failed to load Color Bible state.');
    }
    cbState.data = res;
    cbState.activeFilmId = res.film_id || '';

    // If selected entity no longer exists, reset selection
    if (cbState.selectedEntityId) {
      let found = false;
      if (cbState.activeTab === 'character') {
        found = (res.characters || []).some(c => c.character_id === cbState.selectedEntityId);
      } else if (cbState.activeTab === 'object') {
        found = (res.objects || []).some(o => o.object_id === cbState.selectedEntityId);
      } else if (cbState.activeTab === 'location') {
        found = (res.locations || []).some(l => l.location_id === cbState.selectedEntityId);
      }
      if (!found) {
        cbState.selectedEntityId = null;
        cbState.selectedVariantId = null;
      }
    }

    // Default select first character if none selected
    if (!cbState.selectedEntityId && cbState.activeTab === 'character' && res.characters?.length > 0) {
      cbState.selectedEntityId = res.characters[0].character_id;
      const vars = (res.appearance_variants || []).filter(v => v.character_id === cbState.selectedEntityId);
      if (vars.length > 0) {
        cbState.selectedVariantId = vars[0].variant_id;
      }
    }
  } catch (err) {
    cbState.error = err.message || String(err);
  } finally {
    cbState.loading = false;
    drawColorBible();
  }
}

function drawColorBible(followLogs = false) {
  if (!cbState.data && !cbState.loading && !cbState.error) {
    loadColorBibleData();
    document.getElementById('app').innerHTML = `
      <section class="card stage-loading" aria-busy="true">
        <h1>Color Bible</h1>
        <p role="status">Loading Color Bible state…</p>
        <output data-tab-loading>Connecting to Color Bible database…</output>
      </section>
    `;
    return;
  }

  const data = cbState.data || {
    film_id: '',
    films: [],
    characters: [],
    appearance_variants: [],
    objects: [],
    locations: [],
    assignments: [],
    references: [],
    snapshot_hash: null,
  };

  const films = data.films || [];
  const activeFilm = films.find(f => f.film_id === cbState.activeFilmId) || null;

  document.getElementById('app').innerHTML = `
    <div class="color-bible-container">
      ${cbState.error ? `<div class="cb-error-banner" role="alert"><span>&#9888; ${esc(cbState.error)}</span> <button type="button" class="icon-button" onclick="cbDismissError()">Dismiss</button></div>` : ''}

      <!-- Top Toolbar -->
      <section class="card color-bible-top-bar">
        <div class="cb-top-left">
          <div class="cb-film-selector-wrap">
            <label for="cbFilmSelect"><strong>Film:</strong></label>
            <select id="cbFilmSelect" onchange="cbChangeFilm(this.value)">
              ${films.length ? films.map(f => `<option value="${esc(f.film_id)}"${f.film_id === cbState.activeFilmId ? ' selected' : ''}>${esc(f.title || f.film_id)} (${esc(f.film_id)})</option>`).join('') : `<option value="">No films registered</option>`}
            </select>
            <button type="button" class="small" onclick="cbPromptNewFilm()">+ New Film</button>
          </div>
          <button type="button" class="accent" onclick="cbSyncManifest()" title="Import or sync shots from current project manifest">&#128259; Sync Manifest</button>
          <button type="button" class="small" onclick="cbRefresh()" title="Refresh database">&#8635; Refresh</button>
        </div>

        <div class="cb-top-right">
          <div class="cb-hash-badge-wrap">
            <span class="cb-hash-label">Snapshot Hash:</span>
            <code class="cb-hash-code" id="cbSnapshotHashDisplay" title="${esc(data.snapshot_hash || 'No approved references/assignments')}">${data.snapshot_hash ? esc(data.snapshot_hash.slice(0, 19) + '…') : 'None'}</code>
            ${data.snapshot_hash ? `<button type="button" class="small icon-button" onclick="cbCopyHash('${esc(data.snapshot_hash)}')" title="Copy full snapshot hash">&#128203; Copy</button>` : ''}
          </div>
        </div>
      </section>

      <!-- Split Workbench -->
      <div class="color-bible-split">
        <!-- Left Column: Entities & Shot Range Assignments -->
        <div class="color-bible-left">
          ${renderEntitiesCard(data)}
          ${renderAssignmentsCard(data)}
        </div>

        <!-- Right Column: Color References, Palettes & Approvals -->
        <div class="color-bible-right">
          ${renderReferencesCard(data)}
        </div>
      </div>
    </div>
  `;
}

// --- Left Panel: Entities & Variants ---

function renderEntitiesCard(data) {
  const characters = data.characters || [];
  const variants = data.appearance_variants || [];
  const objects = data.objects || [];
  const locations = data.locations || [];

  return `
    <section class="card cb-card">
      <div class="cb-card-header">
        <h2>Entities & Variants</h2>
        <div class="cb-tab-pills" role="tablist">
          <button type="button" class="cb-pill ${cbState.activeTab === 'character' ? 'active' : ''}" onclick="cbSetEntityTab('character')">Characters (${characters.length})</button>
          <button type="button" class="cb-pill ${cbState.activeTab === 'object' ? 'active' : ''}" onclick="cbSetEntityTab('object')">Props (${objects.length})</button>
          <button type="button" class="cb-pill ${cbState.activeTab === 'location' ? 'active' : ''}" onclick="cbSetEntityTab('location')">Locations (${locations.length})</button>
        </div>
      </div>

      <div class="cb-entity-content">
        ${cbState.activeTab === 'character' ? renderCharactersList(characters, variants) : ''}
        ${cbState.activeTab === 'object' ? renderObjectsList(objects) : ''}
        ${cbState.activeTab === 'location' ? renderLocationsList(locations) : ''}
      </div>
    </section>
  `;
}

function renderCharactersList(characters, variants) {
  return `
    <div class="cb-list-toolbar">
      <button type="button" class="small" onclick="cbToggleNewChar()">${cbState.newCharOpen ? 'Cancel' : '+ New Character'}</button>
    </div>

    ${cbState.newCharOpen ? `
      <form class="cb-form-card" onsubmit="cbSubmitCharacter(event)">
        <h3>Add New Character</h3>
        <label>Name: <input type="text" id="cbNewCharName" required placeholder="e.g. Elena" /></label>
        <label>Notes: <input type="text" id="cbNewCharNotes" placeholder="e.g. Female protagonist, 30s" /></label>
        <div class="cb-form-actions">
          <button type="submit" class="accent small">Create Character</button>
          <button type="button" class="small" onclick="cbToggleNewChar()">Cancel</button>
        </div>
      </form>
    ` : ''}

    <div class="cb-entity-tree">
      ${!characters.length ? '<p class="shot-empty">No characters created yet. Click "+ New Character" above.</p>' : characters.map(c => {
        const isSelected = cbState.selectedEntityId === c.character_id;
        const charVars = variants.filter(v => v.character_id === c.character_id);
        return `
          <div class="cb-entity-item ${isSelected ? 'selected' : ''}">
            <div class="cb-entity-row" onclick="cbSelectCharacter('${esc(c.character_id)}')">
              <span class="cb-entity-icon">&#128100;</span>
              <div class="cb-entity-info">
                <strong>${esc(c.name)}</strong>
                ${c.notes ? `<span class="cb-entity-notes">${esc(c.notes)}</span>` : ''}
              </div>
              <span class="cb-badge-count">${charVars.length} variant${charVars.length === 1 ? '' : 's'}</span>
            </div>

            ${isSelected ? `
              <div class="cb-variants-container">
                <div class="cb-variants-header">
                  <span>Costume / Appearance Variants:</span>
                  <button type="button" class="small icon-button" onclick="cbToggleNewVar()" title="Add variant">${cbState.newVarOpen ? '&#10005;' : '+ Add Variant'}</button>
                </div>

                ${cbState.newVarOpen ? `
                  <form class="cb-form-card cb-nested-form" onsubmit="cbSubmitVariant(event, '${esc(c.character_id)}')">
                    <h4>New Appearance Variant for ${esc(c.name)}</h4>
                    <label>Costume ID*: <input type="text" id="cbNewVarCostume" required placeholder="e.g. costume_1_wool_trench" /></label>
                    <label>Lighting Context: <input type="text" id="cbNewVarLighting" placeholder="e.g. interior_warm, exterior_overcast" /></label>
                    <label>Hair Profile: <input type="text" id="cbNewVarHair" placeholder="e.g. dark_auburn_updo" /></label>
                    <label>Skin Reference: <input type="text" id="cbNewVarSkin" placeholder="e.g. fitzpatrick_II_warm" /></label>
                    <label>Notes: <input type="text" id="cbNewVarNotes" placeholder="e.g. Key scene in train station" /></label>
                    <div class="cb-form-actions">
                      <button type="submit" class="accent small">Create Variant</button>
                      <button type="button" class="small" onclick="cbToggleNewVar()">Cancel</button>
                    </div>
                  </form>
                ` : ''}

                <div class="cb-variants-list">
                  ${!charVars.length ? '<p class="shot-empty">No appearance variants yet. Add one to assign shot ranges.</p>' : charVars.map(v => {
                    const isVarSelected = cbState.selectedVariantId === v.variant_id;
                    return `
                      <div class="cb-variant-item ${isVarSelected ? 'selected' : ''}" onclick="cbSelectVariant('${esc(v.variant_id)}')">
                        <span class="cb-var-icon">&#128087;</span>
                        <div class="cb-var-info">
                          <strong>${esc(v.costume_id)}</strong>
                          ${v.lighting_context ? `<span class="cb-pill-badge">${esc(v.lighting_context)}</span>` : ''}
                          ${v.hair_profile ? `<span class="cb-pill-badge muted">${esc(v.hair_profile)}</span>` : ''}
                          ${v.notes ? `<small>${esc(v.notes)}</small>` : ''}
                        </div>
                      </div>
                    `;
                  }).join('')}
                </div>
              </div>
            ` : ''}
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function renderObjectsList(objects) {
  return `
    <div class="cb-list-toolbar">
      <button type="button" class="small" onclick="cbToggleNewObj()">${cbState.newObjOpen ? 'Cancel' : '+ New Prop/Object'}</button>
    </div>

    ${cbState.newObjOpen ? `
      <form class="cb-form-card" onsubmit="cbSubmitObject(event)">
        <h3>Add New Object / Prop</h3>
        <label>Kind:
          <select id="cbNewObjKind">
            <option value="prop">Prop</option>
            <option value="vehicle">Vehicle</option>
            <option value="other">Other</option>
          </select>
        </label>
        <label>Name*: <input type="text" id="cbNewObjName" required placeholder="e.g. Vintage Sedan" /></label>
        <label>Target Palette (comma-separated hex codes): <input type="text" id="cbNewObjPalette" placeholder="e.g. #2b3a4a, #8c1d1d" /></label>
        <label>Notes: <input type="text" id="cbNewObjNotes" placeholder="e.g. Dark blue body with chrome trim" /></label>
        <div class="cb-form-actions">
          <button type="submit" class="accent small">Create Object</button>
          <button type="button" class="small" onclick="cbToggleNewObj()">Cancel</button>
        </div>
      </form>
    ` : ''}

    <div class="cb-entity-tree">
      ${!objects.length ? '<p class="shot-empty">No objects registered yet.</p>' : objects.map(o => {
        const isSelected = cbState.selectedEntityId === o.object_id;
        const palette = parsePaletteJson(o.palette_json);
        return `
          <div class="cb-entity-item ${isSelected ? 'selected' : ''}" onclick="cbSelectObject('${esc(o.object_id)}')">
            <div class="cb-entity-row">
              <span class="cb-entity-icon">&#128663;</span>
              <div class="cb-entity-info">
                <strong>${esc(o.name)}</strong>
                <span class="cb-pill-badge">${esc(o.kind)}</span>
                ${o.notes ? `<span class="cb-entity-notes">${esc(o.notes)}</span>` : ''}
              </div>
            </div>
            ${palette.length ? `
              <div class="cb-mini-swatches">
                ${palette.map(hex => `<span class="cb-mini-swatch" style="background:${esc(hex)}" title="${esc(hex)}"></span>`).join('')}
              </div>
            ` : ''}
          </div>
        `;
      }).join('')}
    </div>
  `;
}

function renderLocationsList(locations) {
  return `
    <div class="cb-list-toolbar">
      <button type="button" class="small" onclick="cbToggleNewLoc()">${cbState.newLocOpen ? 'Cancel' : '+ New Location'}</button>
    </div>

    ${cbState.newLocOpen ? `
      <form class="cb-form-card" onsubmit="cbSubmitLocation(event)">
        <h3>Add New Location</h3>
        <label>Set Name*: <input type="text" id="cbNewLocName" required placeholder="e.g. Grand Ballroom" /></label>
        <label>Lighting Context: <input type="text" id="cbNewLocLighting" placeholder="e.g. Chandelier candlelight, warm gold" /></label>
        <label>Target Palette (comma-separated hex codes): <input type="text" id="cbNewLocPalette" placeholder="e.g. #d4af37, #3b1e08" /></label>
        <label>Notes: <input type="text" id="cbNewLocNotes" placeholder="e.g. Sequences 2 & 5" /></label>
        <div class="cb-form-actions">
          <button type="submit" class="accent small">Create Location</button>
          <button type="button" class="small" onclick="cbToggleNewLoc()">Cancel</button>
        </div>
      </form>
    ` : ''}

    <div class="cb-entity-tree">
      ${!locations.length ? '<p class="shot-empty">No locations registered yet.</p>' : locations.map(l => {
        const isSelected = cbState.selectedEntityId === l.location_id;
        const palette = parsePaletteJson(l.palette_json);
        return `
          <div class="cb-entity-item ${isSelected ? 'selected' : ''}" onclick="cbSelectLocation('${esc(l.location_id)}')">
            <div class="cb-entity-row">
              <span class="cb-entity-icon">&#127963;</span>
              <div class="cb-entity-info">
                <strong>${esc(l.set_name)}</strong>
                ${l.lighting_context ? `<span class="cb-pill-badge">${esc(l.lighting_context)}</span>` : ''}
                ${l.notes ? `<span class="cb-entity-notes">${esc(l.notes)}</span>` : ''}
              </div>
            </div>
            ${palette.length ? `
              <div class="cb-mini-swatches">
                ${palette.map(hex => `<span class="cb-mini-swatch" style="background:${esc(hex)}" title="${esc(hex)}"></span>`).join('')}
              </div>
            ` : ''}
          </div>
        `;
      }).join('')}
    </div>
  `;
}

// --- Left Panel: Shot Range Assignments ---

function renderAssignmentsCard(data) {
  const assignments = data.assignments || [];
  const selectedLabel = getSelectedEntityLabel(data);

  return `
    <section class="card cb-card cb-assignments-card">
      <div class="cb-card-header">
        <h2>Shot Range Assignments</h2>
      </div>

      <div class="cb-target-banner">
        <span>Target: <strong>${esc(selectedLabel)}</strong></span>
      </div>

      <form class="cb-assign-form" onsubmit="cbSubmitRangeAssignment(event)">
        <div class="cb-range-input-row">
          <label>Shot Ranges:
            <input type="text" id="cbRangeInput" required placeholder="e.g. 12-18, 410-435" />
          </label>
          <button type="submit" class="accent small" ${!cbState.selectedEntityId ? 'disabled' : ''}>Assign Ranges</button>
        </div>
        <p class="cb-helper-text">Enter comma-separated single shots or ranges (e.g. <code>5, 12-18, 100-115</code>). Supports disjoint, non-contiguous intervals.</p>
      </form>

      <div class="cb-assignments-table-wrap">
        <h3>Current Film Assignments (${assignments.length})</h3>
        ${!assignments.length ? '<p class="shot-empty">No shot assignments yet.</p>' : `
          <table class="cb-table">
            <thead>
              <tr>
                <th>Entity</th>
                <th>Shot Range</th>
                <th>Status</th>
                <th>Confidence</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              ${assignments.map(a => {
                const entityName = getEntityNameFromAssignment(data, a);
                return `
                  <tr>
                    <td><strong>${esc(entityName)}</strong></td>
                    <td><span class="range-badge">Shots ${a.valid_shot_range_start} – ${a.valid_shot_range_end}</span></td>
                    <td><span class="approval-badge ${esc(a.approval_status)}">${esc(a.approval_status)}</span></td>
                    <td>${(a.confidence * 100).toFixed(0)}%</td>
                    <td>
                      <button type="button" class="icon-button warn inline" title="Delete assignment" onclick="cbDeleteAssignment('${esc(a.assignment_id)}')">&#128465;</button>
                    </td>
                  </tr>
                `;
              }).join('')}
            </tbody>
          </table>
        `}
      </div>
    </section>
  `;
}

// --- Right Panel: References, Palettes & Approvals ---

function renderReferencesCard(data) {
  const references = data.references || [];
  const selectedLabel = getSelectedEntityLabel(data);

  return `
    <section class="card cb-card cb-references-card">
      <div class="cb-card-header">
        <h2>Color References & Palettes</h2>
        <button type="button" class="accent small" onclick="cbToggleNewRef()">${cbState.newRefOpen ? 'Cancel' : '+ Add Reference'}</button>
      </div>

      ${cbState.newRefOpen ? `
        <form class="cb-form-card" onsubmit="cbSubmitReference(event)">
          <h3>Register Keyframe Reference for ${esc(selectedLabel)}</h3>
          <label>Keyframe Image Path*:
            <input type="text" id="cbRefKeyframePath" required placeholder="e.g. manifests/references/shot_0012_ref0.png or full disk path" />
          </label>
          <label>Optional Binary Mask Path:
            <input type="text" id="cbRefMaskUri" placeholder="e.g. manifests/references/shot_0012_mask.png" />
          </label>
          <div class="cb-inline-fields">
            <label>Provenance:
              <select id="cbRefProvenance">
                <option value="manual">Manual Colorist</option>
                <option value="historical_research">Historical Research</option>
                <option value="ddcolor">DDColor Candidate</option>
                <option value="qwen2511">Qwen2.5-11B Candidate</option>
              </select>
            </label>
            <label class="cb-checkbox-label">
              <input type="checkbox" id="cbRefAutoApprove" checked /> Auto-approve immediately
            </label>
          </div>
          <div class="cb-form-actions">
            <button type="submit" class="accent small">Analyze & Register</button>
            <button type="button" class="small" onclick="cbToggleNewRef()">Cancel</button>
          </div>
        </form>
      ` : ''}

      <div class="cb-references-list">
        ${!references.length ? `
          <div class="cb-empty-state">
            <p>No color references registered for this film yet.</p>
            <p class="cb-helper-text">Register approved keyframes to extract CIE-LAB percentile distributions and dominant RGB swatches.</p>
          </div>
        ` : references.map(ref => renderReferenceItem(ref, data)).join('')}
      </div>
    </section>
  `;
}

function renderReferenceItem(ref, data) {
  const entityName = getEntityNameFromRef(data, ref);
  const palette = parsePaletteRecords(ref.rgb_palette_json);
  const labStats = parseLabStats(ref.lab_stats_json);
  const isApproved = ref.approval_status === 'approved';
  const isRejected = ref.approval_status === 'rejected';

  return `
    <div class="cb-ref-item card">
      <div class="cb-ref-item-header">
        <div class="cb-ref-title-group">
          <strong>${esc(entityName)}</strong>
          <span class="cb-pill-badge">${esc(ref.provenance)}</span>
          <span class="cb-pill-badge muted">v${ref.version}</span>
        </div>
        <div class="cb-ref-approval-actions">
          <span class="approval-badge ${esc(ref.approval_status)}">${esc(ref.approval_status)}</span>
          ${!isApproved ? `<button type="button" class="accent small" onclick="cbApproveReference('${esc(ref.ref_id)}')">&#10003; Approve</button>` : ''}
          ${!isRejected ? `<button type="button" class="warn small" onclick="cbRejectReference('${esc(ref.ref_id)}')">&#10007; Reject</button>` : ''}
        </div>
      </div>

      <div class="cb-ref-body">
        <!-- Thumbnail -->
        <div class="cb-ref-thumbnail-wrap">
          <img src="/media?path=${encodeURIComponent(ref.source_keyframe_path)}" alt="Reference keyframe" class="cb-ref-thumbnail" loading="lazy" onerror="this.onerror=null;this.parentElement.innerHTML='<span class=\\'cb-thumb-missing\\'>Image not found</span>';" />
          <span class="cb-thumb-path" title="${esc(ref.source_keyframe_path)}">${esc(ref.source_keyframe_path.split(/[\\/]/).pop())}</span>
        </div>

        <!-- Palettes & Stats -->
        <div class="cb-ref-details">
          <!-- Dominant RGB Swatches -->
          <div class="cb-swatches-section">
            <h4>Dominant Target RGB Palette</h4>
            <div class="palette-swatches-grid">
              ${palette.map(swatch => `
                <div class="palette-swatch-chip" title="Click to copy ${esc(swatch.hex)}" onclick="cbCopyHex('${esc(swatch.hex)}')">
                  <span class="cb-swatch-color" style="background-color:${esc(swatch.hex)}"></span>
                  <div class="cb-swatch-meta">
                    <code>${esc(swatch.hex)}</code>
                    <span>${swatch.percentage ? (swatch.percentage * 100).toFixed(1) + '%' : ''}</span>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>

          <!-- CIE-LAB Percentiles -->
          ${labStats ? `
            <div class="cb-lab-section">
              <h4>CIE-LAB Percentile Distributions (D65)</h4>
              <table class="lab-table">
                <thead>
                  <tr>
                    <th>Axis</th>
                    <th>p10</th>
                    <th>p25</th>
                    <th>p50 (Median)</th>
                    <th>p75</th>
                    <th>p90</th>
                  </tr>
                </thead>
                <tbody>
                  ${['L', 'a', 'b'].map(axis => {
                    const row = labStats[axis] || {};
                    const axisLabel = axis === 'L' ? 'L* (Luma)' : axis === 'a' ? 'a* (Green–Red)' : 'b* (Blue–Yellow)';
                    return `
                      <tr>
                        <td><strong>${axisLabel}</strong></td>
                        <td>${formatNum(row.p10)}</td>
                        <td>${formatNum(row.p25)}</td>
                        <td><strong>${formatNum(row.p50)}</strong></td>
                        <td>${formatNum(row.p75)}</td>
                        <td>${formatNum(row.p90)}</td>
                      </tr>
                    `;
                  }).join('')}
                </tbody>
              </table>
            </div>
          ` : ''}
        </div>
      </div>
    </div>
  `;
}

// --- Event Handlers & Actions ---

function cbSetEntityTab(tab) {
  cbState.activeTab = tab;
  cbState.selectedEntityId = null;
  cbState.selectedVariantId = null;
  drawColorBible();
}

function cbSelectCharacter(id) {
  cbState.selectedEntityId = id;
  const vars = (cbState.data?.appearance_variants || []).filter(v => v.character_id === id);
  cbState.selectedVariantId = vars.length > 0 ? vars[0].variant_id : null;
  drawColorBible();
}

function cbSelectVariant(varId) {
  cbState.selectedVariantId = varId;
  drawColorBible();
}

function cbSelectObject(id) {
  cbState.selectedEntityId = id;
  cbState.selectedVariantId = null;
  drawColorBible();
}

function cbSelectLocation(id) {
  cbState.selectedEntityId = id;
  cbState.selectedVariantId = null;
  drawColorBible();
}

function cbToggleNewChar() {
  cbState.newCharOpen = !cbState.newCharOpen;
  drawColorBible();
}

function cbToggleNewVar() {
  cbState.newVarOpen = !cbState.newVarOpen;
  drawColorBible();
}

function cbToggleNewObj() {
  cbState.newObjOpen = !cbState.newObjOpen;
  drawColorBible();
}

function cbToggleNewLoc() {
  cbState.newLocOpen = !cbState.newLocOpen;
  drawColorBible();
}

function cbToggleNewRef() {
  cbState.newRefOpen = !cbState.newRefOpen;
  drawColorBible();
}

function cbDismissError() {
  cbState.error = null;
  drawColorBible();
}

function cbChangeFilm(filmId) {
  cbState.activeFilmId = filmId;
  cbState.selectedEntityId = null;
  cbState.selectedVariantId = null;
  loadColorBibleData(filmId, true);
}

function cbRefresh() {
  loadColorBibleData(cbState.activeFilmId, true);
}

async function cbPromptNewFilm() {
  const filmId = prompt('Enter new Film ID (alphanumeric, e.g. "my_feature_film"):');
  if (!filmId || !filmId.trim()) return;
  const title = prompt('Enter Film Title (e.g. "My Feature Film"):') || filmId;
  try {
    const res = await api('/api/color-bible/film', {
      method: 'POST',
      body: JSON.stringify({ film_id: filmId.trim(), title: title.trim() })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to initialize new film');
    await loadColorBibleData(filmId.trim(), true);
  } catch (err) {
    alert('Error: ' + (err.message || err));
  }
}

async function cbSyncManifest() {
  const filmId = cbState.activeFilmId;
  if (!filmId) {
    alert('Please select or create a film first.');
    return;
  }
  const defaultManifest = state.settings?.references?.manifest || state.settings?.shots?.manifest || '';
  const manifest = prompt('Enter manifest path to sync with Color Bible:', defaultManifest);
  if (!manifest || !manifest.trim()) return;

  try {
    const res = await api('/api/color-bible/sync-manifest', {
      method: 'POST',
      body: JSON.stringify({ film_id: filmId, manifest: manifest.trim() })
    });
    if (!res.ok) throw new Error(res.error || 'Manifest sync failed');
    alert(`Successfully synced ${res.shots_synced} shots into Color Bible!`);
    await loadColorBibleData(filmId, true);
  } catch (err) {
    alert('Manifest sync failed: ' + (err.message || err));
  }
}

async function cbSubmitCharacter(event) {
  event.preventDefault();
  const name = document.getElementById('cbNewCharName')?.value?.trim();
  const notes = document.getElementById('cbNewCharNotes')?.value?.trim() || '';
  if (!name) return;

  try {
    const res = await api('/api/color-bible/character', {
      method: 'POST',
      body: JSON.stringify({ film_id: cbState.activeFilmId, name, notes })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to create character');
    cbState.newCharOpen = false;
    cbState.selectedEntityId = res.character.character_id;
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Failed to create character: ' + (err.message || err));
  }
}

async function cbSubmitVariant(event, characterId) {
  event.preventDefault();
  const costume_id = document.getElementById('cbNewVarCostume')?.value?.trim();
  const lighting_context = document.getElementById('cbNewVarLighting')?.value?.trim() || '';
  const hair_profile = document.getElementById('cbNewVarHair')?.value?.trim() || '';
  const skin_reference = document.getElementById('cbNewVarSkin')?.value?.trim() || '';
  const notes = document.getElementById('cbNewVarNotes')?.value?.trim() || '';
  if (!costume_id) return;

  try {
    const res = await api('/api/color-bible/variant', {
      method: 'POST',
      body: JSON.stringify({ character_id: characterId, costume_id, lighting_context, hair_profile, skin_reference, notes })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to create appearance variant');
    cbState.newVarOpen = false;
    cbState.selectedVariantId = res.variant.variant_id;
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Failed to create variant: ' + (err.message || err));
  }
}

async function cbSubmitObject(event) {
  event.preventDefault();
  const kind = document.getElementById('cbNewObjKind')?.value || 'prop';
  const name = document.getElementById('cbNewObjName')?.value?.trim();
  const rawPalette = document.getElementById('cbNewObjPalette')?.value?.trim() || '';
  const notes = document.getElementById('cbNewObjNotes')?.value?.trim() || '';
  if (!name) return;

  const palette = rawPalette ? rawPalette.split(',').map(s => s.trim()).filter(Boolean) : null;
  try {
    const res = await api('/api/color-bible/object', {
      method: 'POST',
      body: JSON.stringify({ film_id: cbState.activeFilmId, kind, name, palette, notes })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to create object');
    cbState.newObjOpen = false;
    cbState.selectedEntityId = res.object.object_id;
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Failed to create object: ' + (err.message || err));
  }
}

async function cbSubmitLocation(event) {
  event.preventDefault();
  const set_name = document.getElementById('cbNewLocName')?.value?.trim();
  const lighting_context = document.getElementById('cbNewLocLighting')?.value?.trim() || '';
  const rawPalette = document.getElementById('cbNewLocPalette')?.value?.trim() || '';
  const notes = document.getElementById('cbNewLocNotes')?.value?.trim() || '';
  if (!set_name) return;

  const palette = rawPalette ? rawPalette.split(',').map(s => s.trim()).filter(Boolean) : null;
  try {
    const res = await api('/api/color-bible/location', {
      method: 'POST',
      body: JSON.stringify({ film_id: cbState.activeFilmId, set_name, lighting_context, palette, notes })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to create location');
    cbState.newLocOpen = false;
    cbState.selectedEntityId = res.location.location_id;
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Failed to create location: ' + (err.message || err));
  }
}

async function cbSubmitRangeAssignment(event) {
  event.preventDefault();
  if (!cbState.selectedEntityId) {
    alert('Please select an entity or variant on the left first.');
    return;
  }
  const rangesText = document.getElementById('cbRangeInput')?.value?.trim();
  if (!rangesText) return;

  try {
    const res = await api('/api/color-bible/assign-ranges', {
      method: 'POST',
      body: JSON.stringify({
        film_id: cbState.activeFilmId,
        entity_type: cbState.activeTab,
        entity_id: cbState.selectedEntityId,
        variant_id: cbState.selectedVariantId,
        ranges: rangesText,
        confidence: 1.0,
        approval_status: 'approved',
        provenance: 'manual',
      })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to assign shot ranges');
    const input = document.getElementById('cbRangeInput');
    if (input) input.value = '';
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Failed to assign shot ranges: ' + (err.message || err));
  }
}

async function cbDeleteAssignment(assignmentId) {
  if (!confirm('Are you sure you want to remove this shot range assignment?')) return;
  try {
    const res = await api('/api/color-bible/assign-ranges', {
      method: 'POST',
      body: JSON.stringify({
        film_id: cbState.activeFilmId,
        action: 'delete',
        delete_assignment_id: assignmentId,
      })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to delete assignment');
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Failed to delete assignment: ' + (err.message || err));
  }
}

async function cbSubmitReference(event) {
  event.preventDefault();
  if (!cbState.selectedEntityId) {
    alert('Please select a character, object, or location on the left first.');
    return;
  }
  const keyframe_path = document.getElementById('cbRefKeyframePath')?.value?.trim();
  const mask_uri = document.getElementById('cbRefMaskUri')?.value?.trim() || null;
  const provenance = document.getElementById('cbRefProvenance')?.value || 'manual';
  const auto_approve = !!document.getElementById('cbRefAutoApprove')?.checked;
  if (!keyframe_path) return;

  try {
    const res = await api('/api/color-bible/reference', {
      method: 'POST',
      body: JSON.stringify({
        film_id: cbState.activeFilmId,
        entity_type: cbState.activeTab,
        entity_id: cbState.selectedEntityId,
        variant_id: cbState.selectedVariantId,
        keyframe_path,
        mask_uri,
        provenance,
        auto_approve,
      })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to register reference');
    cbState.newRefOpen = false;
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Failed to register reference: ' + (err.message || err));
  }
}

async function cbApproveReference(refId) {
  try {
    const res = await api('/api/color-bible/reference-approval', {
      method: 'POST',
      body: JSON.stringify({ ref_id: refId, action: 'approve', approved_by: 'colorist' })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to approve reference');
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Approval failed: ' + (err.message || err));
  }
}

async function cbRejectReference(refId) {
  try {
    const res = await api('/api/color-bible/reference-approval', {
      method: 'POST',
      body: JSON.stringify({ ref_id: refId, action: 'reject' })
    });
    if (!res.ok) throw new Error(res.error || 'Failed to reject reference');
    await loadColorBibleData(cbState.activeFilmId, true);
  } catch (err) {
    alert('Reject failed: ' + (err.message || err));
  }
}

function cbCopyHash(hash) {
  navigator.clipboard.writeText(hash).then(() => {
    const el = document.getElementById('cbSnapshotHashDisplay');
    if (el) {
      const orig = el.textContent;
      el.textContent = 'Copied!';
      setTimeout(() => { el.textContent = orig; }, 1500);
    }
  }).catch(() => {
    prompt('Snapshot hash:', hash);
  });
}

function cbCopyHex(hex) {
  navigator.clipboard.writeText(hex).catch(() => {});
}

// --- Formatting Helpers ---

function getSelectedEntityLabel(data) {
  if (!cbState.selectedEntityId) return 'None (select an entity on the left)';

  if (cbState.activeTab === 'character') {
    const char = (data.characters || []).find(c => c.character_id === cbState.selectedEntityId);
    const charName = char ? char.name : cbState.selectedEntityId;
    if (cbState.selectedVariantId) {
      const variant = (data.appearance_variants || []).find(v => v.variant_id === cbState.selectedVariantId);
      const varName = variant ? variant.costume_id : cbState.selectedVariantId;
      return `${charName} (${varName})`;
    }
    return charName;
  }
  if (cbState.activeTab === 'object') {
    const obj = (data.objects || []).find(o => o.object_id === cbState.selectedEntityId);
    return obj ? `${obj.name} [${obj.kind}]` : cbState.selectedEntityId;
  }
  if (cbState.activeTab === 'location') {
    const loc = (data.locations || []).find(l => l.location_id === cbState.selectedEntityId);
    return loc ? loc.set_name : cbState.selectedEntityId;
  }
  return 'Selected Entity';
}

function getEntityNameFromAssignment(data, asgn) {
  if (asgn.entity_type === 'character') {
    const char = (data.characters || []).find(c => c.character_id === asgn.entity_id);
    const charName = char ? char.name : asgn.entity_id;
    if (asgn.variant_id) {
      const variant = (data.appearance_variants || []).find(v => v.variant_id === asgn.variant_id);
      const varName = variant ? variant.costume_id : asgn.variant_id;
      return `${charName} · ${varName}`;
    }
    return charName;
  }
  if (asgn.entity_type === 'object') {
    const obj = (data.objects || []).find(o => o.object_id === asgn.entity_id);
    return obj ? `${obj.name} (Prop)` : asgn.entity_id;
  }
  if (asgn.entity_type === 'location') {
    const loc = (data.locations || []).find(l => l.location_id === asgn.entity_id);
    return loc ? `${loc.set_name} (Location)` : asgn.entity_id;
  }
  return asgn.entity_id;
}

function getEntityNameFromRef(data, ref) {
  if (ref.entity_type === 'character') {
    const char = (data.characters || []).find(c => c.character_id === ref.entity_id);
    const charName = char ? char.name : ref.entity_id;
    if (ref.variant_id) {
      const variant = (data.appearance_variants || []).find(v => v.variant_id === ref.variant_id);
      const varName = variant ? variant.costume_id : ref.variant_id;
      return `${charName} (${varName})`;
    }
    return charName;
  }
  if (ref.entity_type === 'object') {
    const obj = (data.objects || []).find(o => o.object_id === ref.entity_id);
    return obj ? `${obj.name} (Prop)` : ref.entity_id;
  }
  if (ref.entity_type === 'location') {
    const loc = (data.locations || []).find(l => l.location_id === ref.entity_id);
    return loc ? `${loc.set_name} (Location)` : ref.entity_id;
  }
  return ref.entity_id;
}

function parsePaletteJson(jsonStr) {
  if (!jsonStr) return [];
  try {
    const parsed = JSON.parse(jsonStr);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function parsePaletteRecords(jsonStr) {
  if (!jsonStr) return [];
  try {
    const parsed = JSON.parse(jsonStr);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return [];
  }
}

function parseLabStats(jsonStr) {
  if (!jsonStr) return null;
  try {
    return JSON.parse(jsonStr);
  } catch (e) {
    return null;
  }
}

function formatNum(val) {
  if (val === undefined || val === null || isNaN(val)) return '—';
  return Number(val).toFixed(1);
}
