# Detail Enhancement / Deblur

The optional `enhancement` stage follows Light Deflicker by default. It is
**disabled by default**, with method Off. Remove Noise, Light Deflicker and this
stage can be reordered on Overview or by dragging their tabs. The tab's Move
menu provides the same operation by keyboard or touch. This changes the actual
processing order, not just navigation, and is blocked during processing or a
whole-remaster queue. The order is saved with project settings; older projects
retain the original order. Disabled stages are skipped.
Legacy Clean Up, Stabilization and the later `detail` stage remain in place;
their settings are not migrated or automatically disabled. If both enhancement
stages are enabled, the film is processed twice, at their respective positions.

## Use

The shared comparison has two display profiles: **Review** gives the input and
results more image area and folds frame analysis; **Labor** shows a denser grid
and opens analysis. Both retain the same playback, stepping, zoom, exact-still,
Wipe and difference controls. Profile changes do not render, adopt results or
change processing settings. The preference is saved in this browser, separately
for each saved project (or workspace when no project is saved). Clearing browser
storage resets it to Review. On smaller screens, **Variants & settings** jumps
directly to the collapsible inspector. Full source identity remains available
under **Input provenance**. Zoom percentages are relative to Fit; **1:1** is
native physical-display-pixel mapping. Entering a custom zoom exits 1:1 mode.

Enable **Detail Enhancement / Deblur** on Overview. Include the methods to
compare, choose the current classical preset and picture mix, and explicitly
render missing variants. Selecting comparison rows does not adopt a pipeline
method. **Use for pipeline** adopts only the chosen method, retaining preset
and mix. Off passes the processing input through without enhancement.

### Shared comparison workspace

The left inspector uses compact method rows and **Effort low / medium / high**
badges in green/yellow/red. These are qualitative estimates for the actual ARP
implementation, not benchmarks, quality scores or guaranteed render times.
Method hover/focus and clicking a badge explain the effect, risks, CPU/CUDA path,
requirements and cost drivers. Badge clicks do not inspect, render or adopt.
Parameter labels/controls also explain their scope; the expandable method help
provides a touch-accessible text alternative. Experimental/model warnings and
required acknowledgements remain visible for the inspected method.

Remove Noise groups its nine strength variants under three methods. Redundant
headings and status lines no longer consume inspector height. **More actions &
details** contains pipeline rendering, export, copy settings and technical data.
Normal Review controls fit the tested 1440×800 desktop viewport without internal
scrolling; smaller windows, Labor, long warnings and expanded details can scroll.

The compact-inspector follow-up passed 129 JavaScript tests. Isolated Chrome
checks covered 1440×1000, 1440×800, 1200×720, 768×900 and 390×844; default
enhancement Review had no internal scrolling at the three desktop sizes, including
an unadopted ready candidate at 1440×800. Checks also exercised explicit pipeline
adoption and restoration, nine direct noise-preset choices, contextual warnings,
effort help without selection side effects, keyboard focus, and tooltip Escape
inside the mobile dialog. The existing exact/Wipe/reference and navigation
regression harness also passed. No new hardware benchmark, physical touch-device
or screen-reader audit was performed; processing code was unchanged.

Remove Noise, Light Deflicker and Detail Enhancement / Deblur use the same
workspace. **Review** shows the stage input and one inspected candidate;
**Labor** exposes the included results and frame analysis. Switching views
does not render, adopt, change processing settings, or discard frame, zoom,
pan and comparison membership. The view preference is saved per project;
frame and comparison choices are retained between tabs, not across reloads.

The method name or row selects a candidate, including unrendered methods.
Review inspection leaves Labor membership unchanged. Only in Labor,
**Compare in lab / In lab comparison** controls membership; inspecting a method
there also includes it and brings it into view. **Pipeline** identifies the
explicitly adopted output. Deflicker reveals parameters for the inspected method
in Review, and included/inspected methods in Labor; hidden fields retain values
and confirmations.
Method parameters remain next to the compact method list. The footer keeps
**Use for pipeline**, missing-result rendering, job progress and Stop reachable.
**Render pipeline selection** remains under **More actions & details**, while
settings scroll independently if needed. **Render missing variants**
is offered for the inspected missing Review candidate, or missing Labor members. Neither
render path adopts the last rendered candidate automatically. Pass-through and
copy/output information are under **More actions & details**.

At narrow widths, **Variants & settings** opens a keyboard-focused dialog;
closing it returns focus to the opener. Method help uses a keyboard/touch-accessible
disclosure instead of duplicate method-hover and info-button explanations.
Escape closes the inspector. Scope and limitations remain available as visible text or
an explicitly labelled disclosure, not only in tooltips.

Ordinary wheel input scrolls the workspace; **Alt+wheel** zooms the image.
Ctrl/Meta+wheel remains browser zoom. Source Sans 3 is bundled locally with
its OFL licence and source provenance; no Google Fonts request is needed at
runtime. Predecessor results used as input require current signed provenance,
not just the existence of an older output file. Missing predecessors use the
explicitly labelled standalone fallback described below, never an unsigned file.

**Compare against → Stage input / Original source** changes the displayed
reference only. Original source is the project's unprocessed source file, not
the preceding stage. Exact still, compressed comparison, Wipe and Difference
use the chosen reference; their labels follow that choice. Frame analysis stays
explicitly labelled **vs stage input**. Frame, viewport, comparison membership
and pipeline selection are preserved, independently for each comparison stage.
Candidate provenance is always validated against its actual processing input,
even while displaying the original source. Mix still blends with the processing
input; the reference choice does not enter processing settings or signatures.

Original comparisons require equal geometry and frame rate, and enough original
frames to cover the processing timeline. A source section uses an offset only
when its start is aligned to a source frame. Other mappings are explicitly
rejected: no silent resizing, approximate time matching or changed pixel meaning.
Return to Stage input if that limitation applies.

Membership uses a blue labelled switch only in Labor; Review inspection uses a
blue row and **In view** text. Adoption uses a green row and labelled Pipeline
tag, with **In view** added if inspected in Review. Rendered availability uses
petrol; missing outputs use neutral **Not rendered** text, warnings amber,
running feedback blue, and errors light red. Text and shape carry
the meaning independently of colour; disabled actions remain legible. A rejected
settings save aborts **Render pipeline selection** instead of requesting a job
with the old settings.

Methods are Unsharp, CAS, Multiscale, Richardson–Lucy, NAFNet and Restormer.
Gentle/Balanced/Strong refer to the existing classical implementations, not
equivalent apparent strengths and not three neural checkpoints. Mix is an
integer 0–100% for every method. Changing the classical preset leaves matching
neural results reusable; changing mix changes all affected result identities.

**Stage input** is the selected output of the preceding enabled stage, or the
original processing source when this stage runs first or uses the standalone
fallback described below. The reference label,
processing chain, selected method/settings and path identify that input. It is
not the last variant clicked or rendered. Changing order retains media on disk
but only current signed results are usable. Changed inputs reset frame/Fit
while preserving comparison membership; unchanged inputs retain their viewport.
The shared retained comparison supports
frame-locked compressed playback, exact RGB8 stills, differences, Wipe, pan,
zoom and native 1:1. Pixel changes and sharpness measures are not restoration
quality scores. FlashVSR+ remains a separate scaling research task.

## Neural preparation and integrity

The worker uses native-resolution FP32 CUDA inference, one RGB frame at a time,
without an all-film frame array. It preserves source audio and frame geometry,
requires nominal CFR, and never silently falls back to CPU or lower resolution.
NAFNet uses the official NAFNetLocal width64 GoPro configuration; Restormer
uses the official GoPro motion-deblurring architecture. Required `torch`,
`einops`, NumPy and FFmpeg must be available in ARP's processing environment.

The local installation uses `.cache/detail_deblur/code/NAFNet` at commit
`2b4af71ebe098a92a75910c233a3965a3e93ede4` and
`.cache/detail_deblur/code/Restormer` at commit
`68dc6ac472db26f16361150cb7a96a1bc87da93f`. Both must be clean checkouts,
including their architecture files and licence notices. The model-only
NAFNet loader avoids executing BasicSR's training/dataset import registry;
it does not install old BasicSR training packages into ARP.

Place these official README-linked files in `.cache/detail_deblur/models/`:

| Model/file | Official download | Locally measured SHA-256 |
|---|---|---|
| `NAFNet-GoPro-width64.pth` | [NAFNet GoPro width64](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view) | `329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f` |
| `Restormer-GoPro-motion_deblurring.pth` | [Restormer GoPro motion](https://drive.google.com/file/d/1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L/view) | `194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31` |

These are measured hashes of files obtained from the official links, **not
independently publisher-signed digests**. Workers verify the hash before
`torch.load(weights_only=True)` and strict tensor-state loading. The code does
not automatically download or unpickle arbitrary checkpoints. Code licences
do not settle weight, dataset or footage rights: the commercial model guard
continues to block NAFNet/Restormer pending an audit. Classical methods remain
available with that guard enabled.

## Publication and numerical safeguards

Results go to `intermediate/enhanced/`, keyed by input path, selected method,
effective parameters, checkpoint identity and intermediate profile. Comparison
and downstream use require the completed signature and current upstream
lineage. A failed render leaves the last matching success untouched; a failed
replacement/publication cannot keep the old ready marker. Explicit reuse can
refresh a touched source's provenance without encoding again. Polling never
hashes the whole source or checkpoint.

Neural inputs are reflect-padded to the architecture's multiple and cropped
back to the original size. Extreme synthetic inputs exposed catastrophic NAFNet
numeric excursions even though the checkpoint loaded strictly; finite tensors
alone were not sufficient validation. The runner rejects nonfinite predictions
or absolute values above 16 **before** normal display-range clipping and does
not publish that output. This is a catastrophic-failure guard, not a perceptual
quality threshold, a PSF estimate or a fix for every neural artifact.

## Validation limits

### Typographic workbench follow-up

Methods are now an open list: click the method name or row to inspect it, including
methods with **Not rendered** results. Selection never starts a render or adopts
the method. In **Review**, one inspected candidate appears beside the reference;
its row is blue and labelled **In view**. An inspected pipeline row stays green
and also says **In view**. **Compare in lab / In lab comparison** controls appear
only in **Labor**; Review inspection does not change retained Labor membership.
A petrol **Rendered** tag identifies availability, and a green row plus
**Pipeline** identifies adoption. Pending adoption does not move
the green marker before server confirmation. Source Sans 3 roles and colours are
recorded in `DESIGN.md`; colour roles combine rather than forming exclusive states.

There is one **About this method & limitations** disclosure as the accessible
fallback for targeted method/parameter hover and focus help, without per-field
info buttons. The Alt+wheel / drag / ordinary
wheel guidance remains beside the image controls. Native 1:1's pressed foreground
and background are explicitly paired for contrast.

Tab clicks immediately show the target workspace heading and a loading status.
Long waits show elapsed seconds while navigation stays available. Failed requests
offer **Try again**; cancelled or late responses cannot overwrite the current tab.
The server reuses signed readiness checks only within one state response and reuses
one progress snapshot. The next response checks readiness again; inspection and
rendering endpoints still validate current provenance independently.

The navigation/inspection follow-up passed **126 Node tests** and **126 Python
tests (125 passed, one opt-in neural test skipped)**. An isolated Chrome fixture
checked delayed, failed, retried and obsolete responses; selection without an
output; Review/Labor membership isolation; blue inspection beside green adoption;
and exact Wipe/reference retention at 1440/768/390 px. The rendered app probe had
no failures. Remaining advisories concern existing secondary tokens, code-font
fallback, panel edges and the deliberately height-limited inspector; its cream
palette warning refers to the off-white text, not the neutral background.

A read-only replica using the live project's settings reduced path resolutions
from **9,562 to 889** with identical relevant state. Two profiled runs measured
5.39–6.68 s for reconstructed former behaviour versus 1.03–1.22 s for the current implementation.
These are in-process diagnostic timings with cached source/system payloads,
not post-fix live HTTP timings. The user's running server was not restarted or
modified. Full GUI smoke and screen-reader audits were not repeated for this batch.

Individual preprocessing stages can render from the original processing source
(or prepared source section) when a configured upstream result is unavailable or
invalid. The input label names that fallback. Unsigned predecessor files are never
used as input. The whole-remaster queue and later downstream stages remain strict:
missing/failed predecessors are not silently skipped. Once matching predecessors
exist, the stage input and result identities change accordingly; source-based
trial results are not silently reused for a different input.

The typographic follow-up passed 118 Node tests and 116 relevant Python tests
(one opt-in neural test skipped). Browser checks at 1440/768/390 px cover mixed
membership states, green pipeline adoption and restoration, one method-help
disclosure, visible zoom guidance, pressed native-zoom foreground, mobile dialog
focus, and standalone original fallback with rendering enabled. The independent
finish review found no material correctness issues in its inspected scope;
full GUI smoke, physical-device touch and screen-reader audits remain separate.

Real pretrained CUDA video runs have been exercised on the same short FFV1
clip generated from the official NAFNet demo photograph, retaining native
960×720 geometry, cadence, audio, source bytes and signed reuse. Browser checks
exercise all six rendered methods, exact/compressed delivery, selective preset
invalidation, explicit adoption and retained tab state. The generated demo clip
is **not** a moving historical-film scene or a temporal-quality benchmark.

No long-film performance, 4K VRAM guarantee, authentic grain fidelity, motion
consistency or universal method ranking is established. NAFNet and Restormer
are framewise image models and may change grain, faces, text or temporal detail.
Richardson–Lucy still assumes, rather than measures, a Gaussian PSF. Inspect
short representative sections before committing expensive full-film runs.

Tests: `tests/test_enhancement_stage.py` and
`tests/test_enhancement_workspace_browser.test.cjs`. Set `ARP_TEST_NEURAL=1`
to explicitly include the local pretrained CUDA video test; it needs the
prepared checkpoints and official demo image at
`.cache/detail_deblur/smoke/official-nafnet-demo-blurry.jpg`.

The shared-workspace finish check (2026-11-04) passed 112 Node tests and
87 relevant Python tests (one opt-in neural test skipped). An isolated Chrome
fixture with six existing enhancement results exercised 1600/1440/768/390 px,
profile switching, exact/Wipe output, tab retention, explicit adoption,
parameter DOM/focus retention, wheel modifiers, keyboard help, the narrow
inspector dialog and invalid-upstream action gates. Remove Noise's nine-cell
picker and the blocked Light Deflicker view were also inspected. The independent
finish review's tooltip/dialog Escape conflict was fixed and regression-tested.
These checks are not a full GUI smoke pass, a screen-reader audit or a moving-film
quality validation. The visual probe reported no failures; remaining warnings
concern small secondary annotations, local monospace code text and the deliberately
height-limited inspector beside the larger image workspace.

The follow-up original-reference and workflow regression run passed **115 Node
tests** and **115 Python tests (one opt-in neural test skipped)**. It additionally
exercised original/input switching in exact and compressed display, actual
original-source inspection endpoints, preserved viewport/membership/settings,
ready-result Inspect without prior membership and bringing a Labor candidate
into view. A real FFV1/H.264 regression verifies section-offset subtraction uses
original frame 24 against result frame 0 without changing the result timeline.
This is functional evidence, not a new UX score or a full accessibility audit.
