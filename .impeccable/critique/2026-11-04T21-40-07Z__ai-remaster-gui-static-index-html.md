---
target: critique . - current ARP GUI
total_score: 24
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 3
target_identity: "file:I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\index.html"
target_fingerprint: "sha256:3b3541125adec76e956d442e04a439455ef9ebb5bc90d5776e811f733c33efdc"
target_path: "I:\\ai-remaster-pipeline\\ai_remaster_gui\\static\\index.html"
timestamp: 2026-11-04T21-40-07Z
slug: ai-remaster-gui-static-index-html
---
⚠️ DEGRADED: single-context (the harness rejected both nested assessment agents at its depth limit; design review was completed before detector execution).

## Design specificity verdict

**This feels like a real restoration tool, not a generic dashboard.** Stage-input/result comparison, provenance, exact-still controls, and explicit pipeline adoption express ARP’s purpose. Preserve that direction. The biggest opportunity is making readiness and the next safe action unmistakable.

**Scope:** current GUI, anchored at `ai_remaster_gui/static/index.html`. Reviewed Overview, Remove Noise, Light Deflicker, both enhancement screens, and Output; desktop, tablet, and narrow captures. No processing or project changes were performed. Disabled shot/reference stages were not assessed.

## Design health: 24/40 — Acceptable

| Heuristic | Score | Key finding |
|---|---:|---|
| System status | 2/4 | “Ready / 100%” conflicts with current-input readiness |
| Real-world match | 3/4 | Good restoration language; overlapping enhancement names |
| User control | 3/4 | Explicit adoption and cancellation; reorder lacks undo |
| Consistency | 2/4 | Legacy and new comparison interfaces diverge |
| Error prevention | 2/4 | Navigation reordering changes processing immediately |
| Recognition | 3/4 | Labeled controls and visible provenance help |
| Efficiency | 3/4 | Useful keyboard navigation, zoom, and batch controls |
| Minimalist design | 2/4 | Overview buries the primary action |
| Error recovery | 2/4 | Some guidance exists; pending output lacks a next action |
| Help/documentation | 2/4 | Detailed explanations, but limited task-oriented guidance |
| **Total** | **24/40** | **Significant usability improvements needed** |

Error/recovery scores are provisional: destructive actions and long-running jobs were not exercised.

## What’s working

- **Input and result remain visually comparable.** The footage—not decorative chrome—dominates the workbench.
- **Comparison is separate from commitment.** “Use for pipeline” makes adoption deliberate.
- **Expert controls have substance.** Variant arrow-key navigation worked with a visible 2 px focus outline; zoom and native-pixel controls are clearly labeled.

## Priority issues

### 1. [P1] Readiness makes conflicting promises

Remove Noise simultaneously displays **“Ready / 100%,” “0/9 rendered,”** and **“No matching output for the current input.”**

**Why:** restorers cannot confidently decide whether expensive work is reusable.
**Fix:** distinguish historical completion from readiness for the current input. Show “Needs rerender” and identify the upstream change consistently in Overview, the sidebar, and result panes.
**Location:** `ai_remaster_gui/static/js/render-denoise.js:287`, `:379`; `ai_remaster_gui/static/js/core.js:475`.
**Command:** `/impeccable harden`

### 2. [P1] Overview hides the job beneath setup

At 1440 px, **Run Whole Remaster sits about 2,636 px down the document**. A large repeated logo, ten workflow options, ordering controls, and metadata precede it.

**Why:** returning users must traverse setup to resume work. At 1920 px, substantial width beside the source preview remains unused.
**Fix:** lead with a compact project/source summary, readiness, and Run/Review-next actions. Group optional stages and disclose advanced setup without removing capabilities.
**Location:** `ai_remaster_gui/static/js/render-overview.js:21–54`, `:75–155`.
**Command:** `/impeccable layout`

### 3. [P1] Reordering navigation silently changes processing

Dragging stage tabs—or choosing Move earlier/later—immediately saves the actual preprocessing order. The consequence is explained on Overview, not where the change occurs.

**Why:** a seemingly cosmetic navigation change can invalidate downstream work.
**Fix:** label this as processing order at the interaction point, show affected stages, and provide Undo. Retain the keyboard alternative.
**Location:** `ai_remaster_gui/static/js/core.js:426–433`, `:483–508`.
**Command:** `/impeccable harden`

### 4. [P2] Two enhancement stages are difficult to distinguish

**Detail Enhancement / Deblur** and **Detail Enhancement** appear together, with overlapping sharpening/deconvolution methods but different positions and interfaces.

**Why:** users must infer whether enabling both is intentional or redundant.
**Fix:** name their pipeline position or distinct purpose and explain when each is appropriate. Do not merge or remove functionality without a separate decision.
**Location:** `ai_remaster_gui/static/js/render-overview.js:99–124`.
**Command:** `/impeccable clarify`

### 5. [P2] Visible field labels are not reliably associated

The Source material input has **zero associated labels** in the DOM. Start/End range captions also lack explicit binding.

**Why:** visual proximity does not give screen-reader users an accessible field name.
**Fix:** bind labels and descriptions; verify the shared field renderer and test with NVDA.
**Location:** `ai_remaster_gui/static/js/render-overview.js:192–199`, `:231–240`; `ai_remaster_gui/static/js/render-helpers.js:360–375`.
**Command:** `/impeccable audit`

## Cognitive load and emotional journey

Overview fails **6/8 checklist items**: single focus, chunking, hierarchy, one decision at a time, minimal choices, and progressive disclosure. Grouping works; desktop comparisons reduce recall by keeping evidence visible. Choice points include **8 enabled tabs, 10 workflow options, 9 denoise variants, 5 deflicker methods, and 6 enhancement methods**. Preserve necessary expert choices, but organize their presentation.

The journey begins with credible footage and control, slows through lengthy setup, then loses confidence at conflicting readiness. Output ends with “Pending Detail Enhancement output,” without a direct recovery action.

## Persona red flags

- **Alex, power user:** the run action requires substantial scrolling; near-identical enhancement tabs demand repeated interpretation.
- **Jordan, first-timer:** “Ready” appears to contradict missing results; tab dragging looks like navigation personalization.
- **Sam, accessibility-dependent:** matrix keyboard focus works, but source/range label associations are incomplete. Full screen-reader testing remains outstanding.

## Detector and browser evidence

- **Impeccable source scan:** 2 findings—`repeating-stripes-gradient` attributed to `static/index.html` without a precise line, and `side-tab` at `static/styles.css:899`.
- **Runtime injection succeeded initially:** Overview reported 12 patterns, Remove Noise 2, enhancement 4. These were temporary headless-browser overlays; **no user-visible overlay remains**. A follow-up injection failed after the overlay server stopped.
- **App inspector:** 4 failures and 4 warnings, including missing Overview `h1`, **77 px horizontal overflow at 768 px**, crowded narrow-screen heading/actions, and extensive small text.
- **Source lint:** 10 failures and 18 warnings, largely stylistic. Several are not actionable defects: recomposition gradients implement feather masks, zero-padding formats time, modal shadows indicate elevation, and media viewports are legitimate nested containers. The warning border at `styles.css:899` communicates an experimental limitation.

The behavioral issues above matter more than cosmetic detector counts.

## Minor observations

- Change **“Labor”** to **“Lab”** in the otherwise English interface.
- Pending Output should show an honest unavailable state and a link to the required stage, rather than mounting an unavailable video.
- At narrow widths, comparison and variant selection become widely separated. This is secondary to ARP’s desktop-workstation use.
- Prefer tabular numerals for updating metadata and progress values.

**Questions to consider:** could one shared readiness model drive every status? Could Overview answer “What should I review or run next?” before exposing configuration?

The temporary browsers are closed; neither critique server is still running. No interface code was edited.
