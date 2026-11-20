# Dust & Scratches Interactive Mask Editing Guide (Windows)

This guide explains how to inspect automatic defect proposals, paint non-destructive per-frame mask edits, protect intact film detail, run mask-gated repairs, and manage projects in **AI Remaster Pipeline (ARP)** on Windows.

---

## 1. Launch ARP on Windows

From your ARP workspace root in PowerShell or Command Prompt, launch the local web GUI using either:

```bat
.\launch_gui.bat
```

or directly with the workspace virtual environment:

```powershell
.\.venv\Scripts\python.exe -m ai_remaster_gui
```

ARP starts the local server (`http://127.0.0.1:8765`) and opens the GUI in your browser.

---

## 2. Open a Project & Navigate to Dust & Scratches

1. On the **Overview** tab:
   - Click **Open project…** to open an existing `.arpp` project bundle, **Create project…** to start a new `.arpp` file, or **Browse…** under **Source material** to select a source clip.
   - Under **Processing stages**, check **Dust & Scratches** (off by default) if you plan to run repairs or adopt a repair into the pipeline.
2. Click the **Dust & Scratches** tab in the top navigation bar.

---

## 3. Run Automatic Defect Detection

In the left **Prepare mask** panel:

- **Source**: Keep **Stage input** selected for repairable masks. Choose **Original input · analysis only** only if you want to inspect raw source detections without running repair.
- **Defect detector & Detector preset** (default):
  - **Defect detector**: `Farneback 3F` (`farneback_3f`)
  - **Detector preset**: `Balanced · Farneback 3F (Commercial)` (`balanced`)
  - Open **Fine tuning** to adjust **Detection sensitivity** or **Repair mask expansion (px)** (`dilate`) if needed.
- Click **Calculate mask** (or **Recalculate mask…** to create a separate saved mask version without overwriting earlier caches).

### Optional: Experimental E2 (`experimental_e2`)

Under **Defect detector → Experimental Prototypes**, you can optionally select **Experimental E2 · Scene-vs-Film (Experimental)** (preset `Experimental E2 Research · (Experimental · 3-State)`):

1. Check **I acknowledge experimental detection and unvalidated real-damage accuracy** (`research_confirmed`). Without this opt-in, ARP blocks execution with **Experimental Opt-In Required**.
2. Click **Calculate mask**.
3. E2 writes three non-flattened channels (`Auto Scratch` high-confidence damage, `Auto Uncertain` candidates, and `E2 Vetoed Intact` scene structure in `vetoed_mask.mkv`). Its **Dust** channel is unsupported.

---

## 4. Inspect & Edit Masks Interactively

### Enable Edit Mode

1. In the viewport toolbar (**View**), select **Overlay** (or **Mask** / **Channel** dropdown: `Combined defects`, `Scratches`, `Dust`, `Unclassified`, `E2 Vetoed (intact)`).
2. Click **Edit Mask** (switches to **Edit Mask: ON**). ARP automatically switches into **Exact still** (lossless 8-bit RGB frame view) and **Overlay** mode if **Input** was active.

### Brush Tools & Precedence

Pixel resolution always enforces strict deterministic precedence:

`Protect Intact > Explicit Reject > Human Add > Automatic Proposal`

Even when **Repair mask expansion (px)** (`dilate > 0`) is enabled, **Protect Intact** and explicit rejections are re-enforced after expansion so protected pixels are never modified.

| Button Label | Action |
|---|---|
| **Scratch +** | Add missed scratch pixels to repair (clears prior scratch rejection under stroke) |
| **Scratch -** | Reject false-positive scratch proposals (clears prior scratch addition under stroke) |
| **Dust +** | Add missed dust or blotch pixels to repair (clears prior dust rejection under stroke) |
| **Dust -** | Reject false-positive dust proposals (clears prior dust addition under stroke) |
| **Reject All** | Suppress all automatic proposals (`scratch`, `dust`, `uncertain`) and clear manual additions under stroke |
| **Protect Intact** | Highest-priority safety veto: locks intact film/scene pixels against all repair without erasing prior provenance bits |
| **Unprotect** | Remove the **Protect Intact** lock under stroke |
| **Clear Pixel** | Reset all manual edit bits under stroke back to raw automatic detector output |

### Brush Size, Layers & Opacity

- **Brush radius**: Use the **Brush radius** slider (`1 px (point)` at `r=0` up to `61 px` at `r=30`), or press `[` / `]` to shrink/grow the brush while editing.
- **Overlay opacity**: Adjust **Overlay opacity** when **Overlay** view is active.
- **Layers**: Toggle visibility checkboxes for **Auto Proposals**, **Human Edits**, **Rejected**, **Protect Intact**, and (when using E2) **E2 Vetoed**.

### Navigation, Zoom & Pan

- **Frame navigation**: Step with the transport controls (`-10`, `-1`, Play/Pause, `+1`, `+10`), drag the timeline scrubber, or click **‹ Prev edited** / **Next edited ›** in the edit strip to jump between edited frames.
- **Zoom**: Use `Alt + scroll wheel`, the zoom slider / `% of Fit` input, **Fit**, **1:1**, and **Pixel grid** (visible at high zoom).
- **Paint vs. Pan**:
  - **Left-click + drag** paints with the active brush tool when **Edit Mask: ON**.
  - **Hold `Space` + drag** pans the zoomed frame (`Space+drag pan`; cursor changes to a grab hand).

### Undo, Redo & Auto-Save

- **Undo** (`Ctrl+Z`, up to 30 strokes on the active frame) and **Redo** (`Ctrl+Y` or `Ctrl+Shift+Z`).
- Edits auto-save shortly after each stroke (`Save*` → `Saving…` → `Saved`). Switching frames, changing detectors/sources, or starting repair flushes pending edits automatically.

---

## 5. Review Status, Resetting & `.arpp` Project Persistence

- **Mark Reviewed**: Sets the operational review marker on the current frame (`reviewed` badge). This tracks workflow progress only—it is not independent ground truth.
- **Reset Frame**: Clears manual edits on the current frame only (recoverable via `Ctrl+Z` before leaving the frame). Automatic detector caches are preserved.
- **Reset All**: Clears all manual edits across the timeline while keeping automatic detector caches (`intermediate/dust_scratches/<base_key>/`) untouched.
- **Saving & Reopening Projects**:
  - Edits live in `manifests/dust_masks/<timeline_id>.zip` and `.meta.json` (checksum-verified and protected from **Cached Files** cleanup).
  - On **Overview**, click **Save** (or **Create project…**) to bundle your bitmask archives inside the `.arpp` project file.
  - Reopening via **Open project…** restores all per-frame bitmasks and review flags identically.
  - If the source clip's geometry, FPS, frame count, or checksum changes, ARP displays an **Edit archive mismatch** warning and locks editing until resolved (**Reset All**) to prevent misaligned masks.

---

## 6. Run Composed Repair & Inspect Before/After

1. Click **Create repair…** in the **Dust & Scratches** sidebar to open the **Repair** section.
2. Choose **Repair method**:
   - `telea` (OpenCV Telea, with **Telea radius (px)**) — commercial-safe default.
   - `propainter` or `dart` — research/non-commercial backends (require license/installation confirmation).
3. Click **Repair masked areas** (or **Compose masked repair** for DART).
   - ARP leaves the raw automatic detector cache (`<base_key>/`) immutable and composes an edit-bound mask folder keyed by `edit_signature`.
   - Restoration modifies **only** pixels inside the composed effective mask (`changed_outside_pixels == 0`, zero edits inside **Protect Intact**), preserving native 10-bit format where supported and leaving your source video untouched.
4. Compare results in the comparison panel using **Stage input**, **Result**, **Difference vs input**, or **Wipe**, or toggle **A/B Split View** (`Split` slider) in the mask viewer.
5. Click **Use for pipeline** to explicitly adopt the finished repair for downstream stages.

---

## 7. Optional Provenance Export (Operational Edits Only — Not Ground Truth)

Click **Export Provenance…** in the edit strip above the canvas to write separated PNG layers (`human_edits/`, `protect_intact/`, `effective_repair/`, and `auto_proposals/` when available) along with `metadata.json` and `provenance.json` into `exports/dust_mask_provenance/`.

> **Important:** Exported bundles and review markers record `ground_truth_status: "operational_edits_only_not_independent_gt"` (`protect_or_reject_auto_promoted_to_gt: false`, `untouched_auto_as_gt: false`). Manual strokes, false-positive rejections, **Protect Intact** regions, and untouched automatic proposals represent single-clip operational restoration choices and must **not** be treated as independent ground truth for benchmark evaluation.
