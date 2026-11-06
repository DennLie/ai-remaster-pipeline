# Dust & Scratches — experimental workspace

This optional stage is **off by default**, by default after Stabilization and
before legacy Detail Enhancement and framing. It can now move within the seven
video-processing stages; see [video pipeline order](video-pipeline-order.md).
Existing Clean Up settings are
not migrated or deleted. Disable legacy AI DeScratch before comparing a second
scratch repair. Existing projects keep their order until explicitly reordered.

## Review workflow

1. Complete the configured upstream stages. CPU mask analysis does not require
   enabling Dust for pipeline repair, a repair-model licence or ComfyUI.
2. Choose a detector. These are local inspection drafts, not the
   pipeline selection. Calculate mask explicitly; changing a method starts no
   inference. Native dimensions are retained; large windows fail rather than
   silently resize.
3. Mask ready / Saved is a complete outcome. Open the saved mask, or inspect it
   on the full-width Stage input using Input / Mask overlay / Mask only. The
   magenta overlay has adjustable display-only opacity. Mask overlay and Mask only
   support Play on the shared frame clock. Playback is compressed; choose Exact
   still for authoritative mask-boundary inspection.
4. Optionally choose Create repair… to reveal repair settings; this does not
   start processing. Enable Dust on Overview before explicitly starting repair.
   The matching frozen CPU mask is reused without running the detector again.
   With a completed repair, the input/overlay stays left and the repair stays
   right. Inspect the gated result and absolute RGB
   difference, per-frame numerical counts and CSVs. These are not quality scores.
5. Adopt a repair explicitly. The pipeline checks the adopted report and all source,
    mask and result hashes again. Changes invalidate adoption, not old files.

### Shared Review / Labor workbench

Dust now uses the same comparison workspace as Remove Noise, Light Deflicker,
Enhancement and Stabilization: shared frame transport, compressed frame-locked
playback, exact BGR8 stills, matched zoom/pan, wipe and RGB/false-colour differences.
Review inspects one candidate; Labor retains multiple comparison members. Selecting
or displaying a result does not render it or change pipeline settings.

The saved-candidate list is scoped to the actual post-stabilization input and
validated report identities. It includes mask-only candidates, but only complete
mask-gated repairs enter playback or become adoptable. The analysis draft remains
independent: Calculate mask and Repair saved mask are explicit actions. Saved
mask-only/stopped candidates can request Repair / resume without regenerating the
mask. Failed/stopped states retain their diagnostic message; Stop requested waits
for the current model operation as documented below.

The shared analysis drawer retains mask measurements and technical details,
including numeric counts and CSV/master links. The effective mask and optional
overlay are displayed on the main Stage input, never on the repair. Diagnostics
use the inspected saved candidate's parameters, not an unrelated edited draft.
Their frame follows the shared transport; seeking or changing candidate invalidates
old diagnostic images and late replies. Effective-mask SHA-256 and saved parameters
identify identical frozen CPU masks. DART remains separately labelled joint
inference, and its full-frame diagnostic never becomes an adoptable/playback result.

Inspection, Labor membership, readiness and adoption stay independent. Pipeline
highlighting follows the confirmed report hash and current saved options. Browser
run/adoption requests carry the input and project context; the server rejects a
changed context, and the browser discards late project/frame responses. On small
windows the existing Variants & settings dialog owns the inspector; video panels
stack on phones. Offline historical benchmarks are not imported into this UI.

Validation uses isolated synthetic files, not the active film project: real CPU
provenance/inspection and compressed-playback generation, JS workspace contracts,
and a headless-browser fixture at 1440/768/390 with saved candidates, Review/Labor,
exact stills, retained zoom, draft isolation and a deliberately delayed mask reply.
Browser fixture playback endpoints are mocked; actual compressed reel generation
is tested separately on CPU. No new GPU/model runs are needed for this UI block.
The focused checks passed **93 Python tests** and **115 JS tests**. The rendered
fixture passed browser interaction/overflow checks and the visual inspector at
1440/768/390 with **0 failures, 4 warnings** (incumbent text palette, viewport/control
borders, synthetic colour-bar saturation and the existing 12px phone title inset).
Screenshots and rationale are local in `.cache/dust-workbench-inspect/` and
`.cache/dust-workbench-review.md`. The broader GUI suite's two previously unresolved
errors were not established green by these focused checks.

Masks and model working copies are FFV1/BGR8. The 8-bit repair path preserves
these working pixels outside the effective binary mask, not the original YUV
planes. Supported **10-bit SDR** inputs (`yuv420p10le`, `yuv422p10le`,
`yuv444p10le`, `gbrp10le`, `gray10le`) use a separate lossless native-plane
composition: original decoded samples outside the effective per-plane support
are preserved exactly. Chroma support is the union of source pixels in each
subsampled chroma block; it can extend beyond the luma mask. The encoded native
raster is decoded again and its digest checked against the composite stream.
This does not claim byte-identical compressed files or 10-bit model detail:
replacement content still originates from BGR8. Other higher-depth formats
remain blocked rather than silently converted.

`native_measurements.csv` records per-plane support/changed sample counts.
`repair_measurements.csv` describes actual BGR8-decoded master views, not native
sample identity. Chroma reconstruction can change displayed RGB pixels outside
the luma mask, even with exact preservation outside native-plane support.
Source audio is stream-copied into the repaired master. The stage uses FFV1
regardless of the global intermediate codec, avoiding another lossy generation.
Repair requires constant cadence and a zero-origin video timeline; HDR,
variable-rate and offset input are rejected rather than silently retimed.

## Methods and honest limits

- **Legacy:** unchanged `cleanup_video.ai_scratch_mask`, including its known
  whole-column false-positive risk. This is a baseline, not a recommended preset.
- **Temporal:** optical-flow aligned transient dust/small-defect proposals.
- **Ridge:** the PR #24 detector, pinned to `15af65261a733b6472553d4d32fefcfa43253210`,
  isolated from its ground-truth/evaluation harness. The adapter uses source-
  anchored 24-output-frame windows with one context frame at each side (up to
  26 input frames), raw-luma suspected-cut splitting (median difference >=22),
  and abstention at actual source/scene endpoints. Internal window boundaries
  no longer create artificial empty masks. No context crosses a detected cut.
  This is a **new experimental configuration**,
  not a reproduction of the published five-film pilot. Static scratches can be
  missed. False/missed cuts and moving healthy edges remain risks. Empty masks
  may mean detector safety rejection, not intact film. The original algorithms
  do not distinguish every safety rejection from an empty proposal.
- **Telea:** spatial CPU inpainting, not temporally reconstructed ground truth.
- **ProPainter:** existing ComfyUI node adapter, with the saved CPU mask and no
  additional node mask dilation. Alternative repair settings reuse the exact
  frozen mask bytes. Up to 24-frame jobs split at suspected cuts; overlaps,
  sustained motion and chunk seams are not quality-validated. The user must
  have a working ComfyUI/ProPainter installation. Non-commercial licence;
  blocked by the commercial guard. No model installation is performed here.
  Before repair, a read-only preflight checks the configured node-local weights,
  inventories SHA-256/byte counts, requires an idle live ComfyUI queue and checks
  the node input schema. Missing weights fail before queueing: this node normally
  downloads them during inference, which is not allowed as a hidden side effect.
- **DART:** optional ARP-managed installation, pinned DART v1.0.0 commit and weight
  SHA-256 checks from `run_dart_research.py`, plus clean tracked checkout check.
  Leave checkout/Python overrides blank for `tools/dart`, or retain both explicit
  external paths for compatibility. Acknowledge research/licence restrictions
  separately from installation. Joint inference produces its own soft mask AND repaired frames;
  it does not accept a CPU mask like Telea/ProPainter. Full DART frames are
  diagnostic; adoption uses a separately thresholded, mask-gated composite.
  Currently restricted to **2–96 reviewed, continuous-shot source frames**;
  suspected raw-luma cuts are rejected. Automatic checks do not prove a shot is
   cut-free. RTN rights remain unresolved; commercial guard blocks it.
   Both dimensions must be at least 128 pixels; smaller inputs fail without
   resizing. Invalid numerical casts during external export are fatal.

The inference adapters do not establish model quality or commercial clearance.
Seven-frame synthetic GPU smoke runs are recorded below. Full/native-resolution
neural performance, complete dependency inventories, GPU recovery under disconnects,
HDR/VFR support, full-film DART, extended comparison exports and
human-annotated authentic damage validation remain unfinished. CPU/mocked-model
tests must not be reported as actual ProPainter/DART inference.

### ARP-managed DART setup

Run `install_windows.bat -DartOnly` after base ARP setup. This standalone branch
exits before ComfyUI selection/setup or local configuration writes. It explicitly
asks for research-only acknowledgement; non-interactive setup requires
`-AcceptDartResearch`. The normal interactive full installer asks whether to
include DART (default **N / Enter**), with a separate research-rights confirmation
before installation. Declining either skips DART and permits normal base setup
to finish. The optional-component question warns about the separate environment
and potentially multi-GB downloads. `-InstallDart` selects the component directly;
`-AcceptDartResearch` skips only the rights prompt. Non-interactive full setup
requires both switches to include DART and never asks optional questions.

`scripts/setup_dart.py` clones the fixed DART commit under `tools/dart`, installs
its own Python **3.12.9** under `.python`, and synchronizes `.venv` using
**uv 0.13.6**, `uv.lock`, `--frozen --no-dev`. A different uv on PATH is not
upgraded: installer tooling goes into `tools/dart-installer` instead. Python is
not registered globally; no existing venv is copied. Inherited uv/active-runtime
overrides are removed for runtime installation. Managed environments, code and
weights remain outside ARP/ComfyUI's environments.

Weights come from the fixed GitHub v1.0.0 release and are verified against the
runner's pinned SHA-256 before publication. `-DartWeightsFrom <file>` copies a
verified existing file without changing its source. `-SkipModelDownloads` skips
missing weights unless that explicit local source is supplied; it does not skip
Python/package downloads. Normal installation needs network access and may
download several GB. No inference is started by setup; final preflight imports
the runtime and checks CUDA availability but is not a GPU benchmark.

Re-running setup retains the same pins, checks tracked changes and existing
weight hashes, and synchronizes missing packages. It refuses different/dirty
checkouts and corrupt existing weights rather than resetting or overwriting.
Interrupted setup can be retried; concurrent setup is locked. Managed directory
junctions pointing outside the managed tree are rejected. `I:\DART` is never
discovered, updated or adopted automatically.

Dust status reports a missing managed installation before inference; launch
also validates checkout/weights. Saved masks, repair viewing and saved DART
composition do not depend on reinstalling a runtime. Default path resolution is
shared by artifact identity, status, rendering and adoption, while existing
paired external paths retain their previous identity. Installation does not
enable Dust, grant commercial clearance or set processing confirmations.

Implementation verification is CPU/mock-only: installer tests cover the actual
PowerShell AST dispatch with a mocked subprocess, clean-install/re-run command
contracts, failed-sync retry, isolated bootstrap, checksum publication, Windows
junction rejection and dependency-free CLI help. The managed installation and
multi-GB downloads have **not** been executed; prior external-install GPU smoke
evidence below does not validate this new installation end to end. Final focused
verification: **147 Python tests passed**, including 13 isolated PowerShell
prompt/consent cases; the preceding JS regression run passed **114 tests**.
PowerShell AST syntax
and `git diff --check` passed. Python startup settings were isolated from the live
project. Neither the running ARP/ComfyUI instances nor user-film artifacts were
changed by these checks.

### Windows analysis publication and retained staging

Analysis now stages under `.<candidate-id>.analysis-<suffix>` without automatic
temporary-directory deletion on failure. Before this correction, a failed
directory rename invoked cleanup that could delete parts of an otherwise
finished DART result while another file was still locked. A CPU-only test with
an actual Windows handle denying delete-sharing reproduced that failure before
the change and passes afterward. This proves the cleanup defect, not which
process held the lock in the reported 72-frame user run.

Publication retries file-access failures five times (one second total backoff).
If still blocked, the complete stage remains with an actionable error. Retrying
mask calculation with identical source/settings revalidates source identity,
source hash and all required mask hashes (also `dart_full.mkv` for DART), then
retries only publication. It does not rerun inference or adopt a repair. Pending
stages are not offered as saved masks or repair candidates. Incomplete, corrupt,
inaccessible or ambiguous stages remain blocked for inspection, without automatic
overwrite, deletion or new inference. Interrupted incomplete analysis may consume
disk space and requires explicit human inspection before another calculation.

Five publication regressions cover persistent/injected and actual Windows locks,
transient retry, incomplete-stage retention, corrupted mask rejection, DART
diagnostic hashes, and inference-free retry. No new DART/GPU job was run. The old
`.dust-p4kqix0q` stage was already inaccessible after its cleanup failure; its
contents and permissions have not been changed, recovered or certified complete.

Offline benchmark films and results are not imported into restoration projects.
F1/IoU/precision/recall require synthetic or human-annotated ground truth; mask
coverage and changed-pixel counts on an ordinary project do not provide those.

Backend version 3 invalidates earlier candidate identities without deleting
their files. Explicit new analysis/repair/adoption is required. These changes
do not establish restoration quality, cut accuracy or authentic damage detection.

## Safe stop, retry and publication

Dust jobs are process-locked per candidate. A second worker cannot concurrently
write that candidate; the OS releases its lock even if the worker crashes.
The GUI's **Stop** sends a unique per-run cancellation marker instead of killing
the Dust worker tree. It does not stop ComfyUI, clear its queue, call a global
interrupt, or affect unrelated jobs. There is no instant/mid-inference stop:
the current ProPainter section or bounded DART invocation finishes first.
The worker checks the stop request before another section and before committing
a repair. An already-started file commit can finish; a stop observed afterward
still exits nonzero so a queued pipeline does not advance to another stage.
An unreachable ComfyUI can delay this cooperative stop while its existing
history-poll retry timeout runs. Repeated Stop does not force-kill the model.

Click **Repair** again with identical source/settings to retry. ProPainter's
verified completed sections in `.model-chunks/` are reused only when input/mask
hashes, settings, recorded model preflight, section range and output hash match.
Their inputs are retained as well, so an already-submitted prompt never depends
on disposable temporary files. A recorded acknowledged prompt can be reattached
if its successful history still exists; it is not submitted again. If a crash
or network failure leaves the submission outcome unknown, or history has been
lost/failed, retry refuses rather than risking a duplicate. Such cases need
human inspection of ComfyUI and the retained job record; there is no automatic
reset/overwrite. Mid-model state is not checkpointed. Telea restarts composition;
DART analysis restarts its bounded invocation rather than resuming neural state.

Dust retains the existing 24-frame/cut boundaries, model workflow and ProPainter
parameters. A multi-section test found that Dust had called the **outpainting**
split helper, which adds LTX repeat-padding and defaults to lossy encoding.
Only Dust's input slicing was replaced with exact frame-index FFV1/BGR8 trims:
no extra frames, no resampling of frozen binary masks, no overlap/extra dilation.
ProPainter requires at least two frames per section; a one-frame cut/tail section
is rejected **before any prompt** instead of silently padding or shifting it.
The shared adapter has only optional prompt-ID/acknowledgment lifecycle hooks;
legacy callers without them still submit/wait exactly as before.

Repairs are fully staged in `.repair-ready/`; the top-level `report.json` is the
completion marker and is atomically replaced last. Before publication the source,
frozen mask artifacts, staged master/CSV hashes, frame count, geometry and cadence
are checked again. Interrupted file publication can be finished on retry without
overwriting already-matching artifacts. Conflicting/changed artifacts fail closed
and are retained for inspection. Until that completion marker is valid, status,
inspection and explicit adoption cannot treat a stray `repaired.mkv` as complete.
Previously completed results and the active project's selection are not changed.

15 new CPU failure-injection/contract tests cover stop/retry, completed-section
reuse, acknowledged/unknown submissions, altered checkpoints/source/staged output,
interrupted publication, locking, exact frozen-mask trims, cut/tail handling,
fractional cadence, owned-worker stopping and unchanged legacy adapter behavior.
They use real FFV1 files and **model test doubles**, not new GPU quality evidence.
The focused suite passed **85 tests** (15 new job tests plus the existing 70
Dust/preflight/DART/HTTP/order/licensing/stabilization checks); `git diff --check`
also passed.
GPU recovery under disconnects, sustained motion/chunk-seam quality and
the two outstanding broad-GUI errors remain unvalidated/unresolved.

### Real ProPainter GPU stop/retry acceptance — 2026-11-06

The bounded acceptance run passed on the RTX 4090 with **50 synthetic frames,
256×192, 24 fps**, split into unchanged sections `[0,24)`, `[24,48)`, `[48,50)`.
Only the CPU detector was replaced by a known synthetic stripe-mask fixture;
preflight, ProPainter/ComfyUI inference, production Dust worker, checkpointing,
composition and publication were real. Candidate storage was redirected to
`.cache/dust-gpu-resume-20261106-01/`, outside the active film project. No settings
or outputs were adopted, and no model/workflow/production-code changes were
needed for this acceptance block.

- The first prompt was observed in ComfyUI's running queue, without a complete
  section checkpoint. At 1.7279 s after worker launch the harness wrote the same
  unique cancellation-file protocol used by GUI Stop. It did **not** exercise a
  browser click, request a global interrupt, or clear a queue.
- The current section finished; the worker exited nonzero after 6.0766 s, with
  exactly section 0 checkpointed, no final master, and repair validation refusing
  the incomplete candidate. This is a cooperative stop, not a killed inference.
- The second worker reused section 0: its model file, completion record and job
  record retained exact SHA-256 **and mtime_ns**. Only sections 24 and 48 submitted
  new prompts. ComfyUI history contains exactly three unique successful prompts,
  not four. Resume through final verification took 9.8178 s; this is a harness
  measurement, not a model-speed benchmark.
- All input/mask section frames were decoded and compared exactly to their frozen
  source slices. All 50 saved master frames matched the ordered source/model
  mask-gated construction; every outside-mask pixel remained unchanged. Frozen
  mask artifacts and source hash were unchanged, the final cadence/timeline
  passed validation, and decoded source/final audio MD5s matched.
- The isolated offline ComfyUI on `127.0.0.1:8193` used the existing ProPainter/VHS
  nodes and 20% PyTorch allocator cap. It was stopped afterward; training PID
  92184 retained its original start time and was confirmed running. No installed
  weights/dependencies or external tracked model files changed.

Source SHA-256: `33631b6f302ccbb5dccbf3d3e980bce80f532446b01ade14fc565fda17c9f64a`.
Final master: `c8b1005202b2fa9c50e21a140ea0b2470a18d8da02666ff6bef774a9daa724bc`.
The run folder retains `verified.json`, `additional-qa.json`, `frame-checks.csv`,
`stop-request.json`, `stopped.json`, full ComfyUI history, worker/server logs,
runtime/model preflight inventories and owned-process shutdown records. The 15
CPU job contract tests were rerun and passed separately.

The VHS log contains nonfatal audio-extraction warnings for the intentionally
silent model-input/mask sections. All three histories report success; the actual
source soundtrack was muxed separately and verified as above. The warnings are
retained, not represented as a warning-free run. CUDA peak memory was not measured.
This run does **not** validate browser interaction, mid-inference neural-state
resume, network/process-failure recovery, native-resolution cost, historical
repair quality or visible motion/chunk seams. DART was not rerun in this block.

## Model preflight and GPU smoke evidence — 2026-11-06

`scripts/dust_model_preflight.py` checks existing installations without queuing
prompts, starting servers, installing dependencies or downloading models:

```powershell
.\.venv\Scripts\python.exe scripts/dust_model_preflight.py `
  --dart-repo I:/DART --dart-python I:/DART/.venv/Scripts/python.exe `
  --comfy-dir I:/ComfyUI_windows_portable/ComfyUI `
  --output .cache/dust-model-preflight-new.json
```

The optional output file must not already exist. DART checks its pinned clean
tracked checkout, exact checkpoint and isolated Python/CUDA prerequisites.
The ProPainter hashes inventory installed files, not independent upstream pins
or a security/licensing audit. A successful prerequisite check neither reserves
the GPU nor proves inference, chunk quality or VRAM sufficiency.

On this workstation, `.cache/dust-model-preflight-20261106-02.json` records:

- DART pinned checkout/checkpoint verified; Python 3.12.9, PyTorch 2.9.0+cu126,
  CUDA available and the checked runtime modules present.
- All three ProPainter weights present and hashed; node checkout
  `9c27d5a0a508bae3296a1886ad026d8d4139d66c`, tracked files unmodified.
  ComfyUI was not reachable at configured `http://127.0.0.1:8188` during that
  initial check. The preflight itself started no server and sent no prompt.

### Actual model runs (after explicit parallel-GPU permission)

The user subsequently approved small tests alongside the existing training and
a separate ComfyUI startup. Runs used the RTX 4090 sequentially, separate output
folders and a 20% per-process PyTorch allocator cap (not a GPU reservation or a
cap on all driver allocations). No dependencies/weights were installed or
downloaded; neither external model checkout's tracked files were changed.

Common input: **seven synthetic frames, 256×192, 24 fps**, without audio.
This test fixture was nearest-neighbour enlarged from 128×96; production inputs
are not silently enlarged. Its SHA-256 is
`11591ad3cbf8f37f5bdb418fa14de297beca9a876eb00918eade7e71001be00b`.

| Backend | Observed wall time | Result | CUDA memory measured |
| --- | --- | --- | --- |
| DART | 9.7736 s | 7 restored PNGs + 7 own soft-mask PNGs | peak allocated 277,811,200 B; reserved 352,321,536 B |
| ProPainter | 4.6547 s | adapter output: 7 frames, 256×192, 24 fps | peak not measured |

These wrapper timings include different startup/queue/export work, with another
training process present; they are **not** a speed ranking or native-resolution
throughput estimate. ProPainter used a known binary stripe mask, not DART's
predictions and not a validated detector mask. This was not a Telea/ProPainter
quality comparison or an end-to-end ARP project/adoption test.

- **DART provenance:** commit
  `fc6f3e722c74a3b0e3a43ef40bb1d5303fa5e332`, checkpoint SHA-256
  `d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3`;
  Python 3.12.9, PyTorch 2.9.0+cu126.
- **ProPainter runtime:** ComfyUI 0.43.0, commit
  `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`; Python 3.13.14,
  PyTorch 2.14.1+cu130; node commit as above. Installed model SHA-256:
  - `raft-things.pth`: `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`
  - `recurrent_flow_completion.pth`: `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283`
  - `ProPainter.pth`: `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591`
- ProPainter's frozen mask SHA-256:
  `ec10f8d2fff0d0b82e720e25e3d1b722a058d266dcd91d7a026b418d74fb0cd5`.
  Raw output: `60336d7a6d69a7784c7e6c1c9dd54b56318dee16a254dd0dd9006eb423582ef6`.
  Separately gated FFV1: `9c5173e339de0c5f106e0d8ba33a99a09d7c1296c6ddaf8e1bcd0edbc48c9bf9`.

The isolated offline ComfyUI listened only on `127.0.0.1:8193`, with ProPainter
and VHS custom nodes whitelisted and separate input/output/temp/user folders.
Its owned process was stopped afterward; the existing training PID 92184 was
confirmed still running. The original ARP server was not restarted. No outputs
were adopted into the active film project and no defaults were selected.

### Artifact checks and retained failure

Evidence is local under `.cache/dust-gpu-smoke-20261106-01/`:

- `dart-256x192/cuda-smoke.json`, `propainter-smoke.json`: actual run metrics,
  runtime/preflight inventory and output identity.
- `verified.json`: source/mask/output hashes, exact DART PNG file sets and
  dimensions, ProPainter frame-count/geometry/cadence, and exact preservation
  outside the known mask after decoding the saved gated FFV1. All seven frames
  passed; `propainter-gating-check.csv` records zero changed outside pixels for
  the gated result (the raw model output does change outside pixels).
- `dart-gating-verified.json`: all seven saved gated DART PNGs were decoded and
  compared against the exact source/model composition at soft-mask threshold
  160. Outside-mask BGR8 pixels are unchanged. The full DART output is separate
  diagnostic evidence, not the adoptable composite.
- `dart-qa/dart_restoration_diagnostics.json`: mask coverage 0.3197632 and
  changed-unmasked fraction 0.6126514 at an 8-intensity-unit change threshold.
  These are **not accuracy or repair-quality scores**.
- `dart/validation-failed.json` excludes the first 128×96 run: upstream exited
  zero despite invalid numerical casts. A strict rerun in `dart-small-strict/`
  failed at `scripts/inference.py:192`. RAFT's four-level correlation pyramid
  reaches a singleton axis for that input, and its bilinear sampler divides by
  `W-1`/`H-1`. The failure artifacts are retained, not successful test outputs.
  The pinned runner now rejects dimensions below 128 and makes invalid-cast
  RuntimeWarnings fatal; the strict 256×192 run passed. This does not prove all
  larger geometries or long sequences safe.

70 focused CPU/contract/regression tests passed, including preflight, DART
geometry/fatal-warning, Dust, HTTP, stage-order, commercial-guard and stabilization
checks. Those tests are separate from the actual GPU runs. The broader GUI suite
still has two unresolved errors and was not established green by this block.
Authentic-damage quality, sustained motion/chunk seams, native-resolution GPU
cost and safe model cancellation/resume still require validation. Licence
restrictions remain unchanged: ProPainter non-commercial, DART research-only.
# Guided Dust workbench

The Dust inspector inherits the shared Typographic Workbench, not a separate
settings-page style. Detection and Repair are grouped by their actual jobs.
Choose the detector first; selection starts no processing. Mask ready / Saved
completes the standalone analysis task. Create repair… optionally reveals the
repairer and its settings, without starting it. No second
generic Render action is needed. Saved variants use readable detector/repair names;
their hashes and file links remain in technical details.

Mask inspection is an optional display layer on the left Stage input, with
Input / Mask overlay / Mask only and display-only magenta opacity. It decodes the
stage input and effective mask at the shared frame ordinal. Without a requested
completed repair the input fills the viewer; a completed repair appears cleanly
on the right. Both use shared zoom/pan. Saved input and effective mask now share
one frame-indexed playback atlas, including when no repair exists. Play works in
both mask views. Preview encoding is compressed; exact stills remain authoritative
for mask boundaries. An incomplete repair cannot be
adopted. Changed settings remain a draft, never a mutation of a saved candidate.
Successful explicit jobs select their matching candidate for inspection, not for
the pipeline. Use for pipeline is still a separate, validated decision.

DART's first action names joint mask/restoration inference. Its subsequent action
is Compose masked repair, not another neural restoration pass. Research/licence
acknowledgements remain visible. No processing defaults or model parameters change.

UI verification uses synthetic fixtures, including missing-mask, mask-only,
completed-repair, draft changes and late exact-frame responses. This is not new
restoration-quality or end-to-end GPU evidence.

Current mask-playback verification: 114 focused JS tests and 110 focused Python tests
pass. Python startup uses isolated empty settings: importing the server against
the live settings encounters an existing access-denied Dust result path; no
permissions or live settings were changed. `.cache/dust-input-browser.mjs`
checks the actual Chromium UI at 1440/768/390, including opacity pixel changes,
independent mask completion, optional repair reveal and left-only overlay placement.
The direct Impeccable detector reports no findings. The earlier guided-layout
visual inspector reported 0 FAIL
and 3 warnings: the pinned warm-neutral text is mistaken for a cream background,
independent inspector/viewer regions are measured as peer cards, and the existing
mobile header has 12px margins. These are not model/quality measurements.
The earlier broader JS run was not green: 167/168 passed; the Overview workflow-count test
still expects ten inputs while the existing Dust-enabled Overview has eleven.
Overview is outside this inspector change and was not modified in this work.
Mask/repair availability is validated independently: a missing or changed repair
retains its valid saved mask in status, the catalogue and exact inspection, while
comparison/adoption of that repair stays blocked. CPU regressions cover corrupt
and missing repair files. The current Chromium fixture also asserts aligned raster
tops in split view at desktop/tablet sizes; opacity is repainted locally without
another frame decode request. Earlier still-only checks did not establish mask
playback or restoration quality. The subsequent `.cache/dust-mask-playback-browser.mjs`
check uses real synthetic CPU source/mask files and actual encoded video, verifies
advancing overlay/mask-only playback, seek across a two-second segment boundary,
and return to exact stills without inference/adoption. This is playback evidence,
not historical-damage quality evidence. Restart ARP to load the backend changes; then
hard-refresh the browser. No active server restart is performed by this work.
Combined playback also covers a synthetic completed CPU repair and a distinct
original-source reference: the mask stream remains bound to Stage input while
repair streams retain the selected Original source reference. A JS regression
checks per-slot references and atomic common-frame publication. Scoped finish
review approved this reference correction. Native-resolution full-film preview
cost and historical-film damage quality remain outside this acceptance test.
Earlier guided-layout browser assertions covered draft-vs-saved inspection labels, retaining
Labor membership when a candidate appears, live viewport state after a delayed
mask response, and keeping the mobile inspector/focus open after a field change.
