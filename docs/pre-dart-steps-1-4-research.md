# Pre-DART research: film pipeline stages 1–4 (2026-11-01)

**Research decision, not production approval.** The objective is damage-mask
accuracy and faithful restoration, **not** aesthetic modernization.
This is an upstream technical review of the ARP `main` branch, its open
experimental PR #6, and primary documentation. No stage is enabled by
this document; all branches remain drafts.

## Baseline: what ARP actually ships

| Video-described stage | In `main` | Development branch | Recommendation before DART |
|---|---|---|---|
| 1 Light Deflicker | No | PR #6 `feature/light-deflicker-cpu`, real Commons test | Opt-in, conditional per shot |
| 2 Remove Noise | No dedicated noise-filter stage; only generative backends downstream | None | Leave off by default, test only if identified transfer/sensor noise disrupts DART |
| 3 Detail Enhancement | No conservative standalone pre-DART stage. LTX 2.5 CQ Enhancer exists later under Upscale | Existing upscale is not an equivalent | **Never automatic before DART**; separate modernized mastering variant if explicitly requested |
| 4 Camera Stabilisation | Yes, `scripts/stabilize_video.py`, FFmpeg/libvidstab, `Stabilization` GUI | Existing tests in `tests/test_stabilize_video.py` | Opt-in on actual gate weave, preserve scene motion and frame edges |

**Actual current GUI ordering:** `Cleanup -> Stabilization -> Outpainting
-> Colorization/Finishing -> Upscale`. In the experimental Light-Deflicker
branch, the light correction is **inside Clean Up**, before DeVignette,
temporal OpenCV defect masks and optional DeScratch / Dearchive.
The external DART evaluation is not a GUI backend, so the pre-DART
ordering is still a research choice rather than an implemented
automatic production path.

## Evidence and concrete implementation notes

### 1. Light Deflicker — keep PR #6, improve fidelity before defaults change

`scripts/film_deflicker.py` (PR #6) is a two-pass OpenCV/NumPy algorithm
with central-crop robust luma estimates, brightness-standardized scene
histograms, 11-frame neighborhood (radius 5), monotonic-ramp protection,
gain strength 0.65 and **gain cap ±8%**. No learned checkpoint.
Its real `Neighbors` experiment (960x720, 30 fps, 120 frames) applied
changes to **45** frames; a proxy for frame-to-frame luminance changes
was **2.61345 -> 1.84874**. **This is not measured historical
faithfulness**. See [the successful GitHub Actions run](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41343808901)
and PR #6's docs.

Risks:
- Current process decodes to 8-bit BGR, applies one LUT gain uniformly
  across color channels, and re-encodes. Clipping/rounding and source
  color transfer are not modeled. Do not claim high-bit-depth preservation.
- Histogram cuts and monotonic-ramp checks are heuristics, not semantic
  detection of artistic lighting or historically original flicker.
- The FFmpeg `deflicker` filter offers median/mean temporal modes as
  an objective alternative, but is not inherently film-cut aware.
- Run 3-way **original / custom / FFmpeg** tests on different source
  scenes and manually check fades, light flashes, skin, titles,
  blown highlights and grain. Avoid tuning to minimize exposure
  variance alone.

Official source: https://ffmpeg.org/ffmpeg-filters.html#deflicker

### 2. Remove Noise — missing; justify it by noise provenance first

`main` contains no independent denoise stage or settings. In the
available 960x720 2016 digital transfer, native film grain, transfer
noise, VP8 compression artefacts, dirt, and scratches cannot be assumed
to be the same removable signal. Blind smoothing can destroy both
real grain and the bright/dark defects whose location DART MaskNet is
supposed to detect.

Conservative research candidates:
- **FFmpeg hqdn3d**: spatial *and* temporal denoising; the normal
  default strengths are not a safe historical-restoration preset.
  https://ffmpeg.org/ffmpeg-filters.html#hqdn3d
- **FFmpeg nlmeans**: nonlocal denoising baseline, expensive, with
  potential loss of fine film texture.
  https://ffmpeg.org/ffmpeg-filters.html#nlmeans
- **OpenCV fastNlMeansDenoisingMulti**: multi-frame NLM, sensitive to
  motion, occlusion and scene cuts; not automatically optical-flow
  aligned. https://docs.opencv.org/5.0/main_modules/photo_denoise.html
- **Future motion-compensated temporal method**: best motivated for
  true transfer/sensor noise, but implement only if the results of the
  raw DART baseline demonstrate a need. Need per-shot cut/occlusion
  handling, displacement bounds and conservative masks.

**Decision:** No default denoise before DART. In the ablation
experiment, start with raw source, then separately use mild noise
processing on one identified noisy scene. Measure changes in DART
mask sensitivity and source-detail retention rather than a subjective
smoothness score. Do not strip grain and add synthetic grain to hide
the intervention.

### 3. Detail Enhancement — intentionally skip before DART

No standalone pre-DART enhancement feature exists in ARP's main path.
The optional *LTX 2.5 CQ Enhancer* operates in ARP's later Upscaling
workflow; it is generative, its exact model bundle is not fully
commercially cleared, and it may reconstruct identities.

Generic sharpening/unsharp masking before DART amplifies scratches,
halos and compression ringing. This is counterproductive to model
mask generation and faithful defect repair.

**Decision:** Set pre-DART detail enhancement to **OFF** in every
restoration test. For an archival-oriented master, do not artificially
sharpen at all. If a separate distribution-oriented/modernized master
is wanted, assess explicit constrained enhancement **after** repairs,
with side-by-side disclosures and unchanged historical reference
master. This is a distinct creative deliverable rather than historical
restoration.

FIAF's authoritative position is especially restrictive: the
`Digital Statement Part III` defines artificial sharpening/smoothing,
grain removal/replacement, and coloring originally black-and-white
films as exceeding **ethical archival restoration**. Keep
colorized/modernized outputs separately labeled rather than claiming
they are the archival restoration master.

https://www.fiafnet.org/pages/E-Resources/Digital-Statement-part-III.html

### 4. Camera Stabilisation — already implemented, avoid frame loss

`scripts/stabilize_video.py` in main has shot-aware two-pass
`vidstabdetect` and `vidstabtransform`, scene boundaries from the
reference detector, optional user-reviewed shot manifest, untrusted
motion-fit fallback, native-size intermediates, restoration of
source audio and browser previews. Existing unit tests cover
multi-shot consistency, motion detection fallback and a video smoke
test.

Default UI values: smoothing **12 frames**, maximum translation **48 px**,
maximum rotation **3 degrees**, fixed safety zoom **3%**, scene
threshold **0.075**. In the transform filter:
`crop=black:optzoom=0:zoom=3:interpol=bicubic`. No *adaptive*
zoom flicker, but **fixed 3% zoom discards picture area**.

For historic material, distinguish gate-weave/sprocket jitter from
original camera pan, exposure changes and intentional unstable shots.
A static frame does not establish an historically correct version.
**Keep Stabilization opt-in**; compare source / 0% zoom with black
borders / fixed zoom only for actual defects, track crop and
detail interpolation. Do not automatically move it before DART:
DART already uses RAFT flow, and stabilization transforms can
destroy source pixels or create black-border occlusions.

FFmpeg filters: https://ffmpeg.org/ffmpeg-filters.html#vidstabdetect
Upstream library, including licence and two-pass warnings:
https://github.com/georgmartius/vid.stab

**Distribution licence:** current libvidstab states LGPL-2.1-or-later,
while upstream documents FFmpeg `--enable-gpl --enable-libvidstab`
is still required. This is not a ban on commercial use, but a
binary/distribution compliance item (check versions, compile flags,
third-party notices and obligations). Never mark the complete
third-party bundle commercially cleared solely because core library
is LGPL.

## Experiment to decide which preprocessing is really needed

Run identical short, manually reviewed, **cut-free** native-resolution
clips from *Neighbors* with the **same pinned DART v1.0.0 checkpoint**,
same input sizes, same frame counts and exact timestamps:

| ID | Preprocessing | Purpose |
|---|---|---|
| A — control | Original source, none | Required DART baseline |
| B — exposure | Light Deflicker only | Does normalization improve mask accuracy or repair? |
| C — texture | Mild denoise only | Only if objectively noisy transfer; look for lost grain/defect recall |
| D — geometry | Stabilization only | Only on a shot with genuine gate weave; monitor crop/detail |
| E — combined | Only stages justified by B/C/D, no pre-sharpen | Check interactions and compound artefacts |
| F — cosmetic | Optional detail enhancement **after** DART repair | Separate modernized master, *not* fidelity ground truth |

Always evaluate:
1. Annotated defect-mask precision, recall, IoU, false positives on
   hair, clothing, faces, typography and intentional high-frequency
   film texture; not only percentage of proposed masks.
2. Full restored frames against a human-reviewed reference, difference
   heatmaps **inside/outside DART soft masks** and changes in intact
   detail; compare a *mask-gated composite* against unrestricted DART.
3. Exposure/cut/fade preservation; shot boundary safety.
4. Source image geometry, crop pixels, grain texture and background
   jitter. Align original/restoration coordinates for evaluation,
   never silently resize mismatched masks.
5. Timing, frame count, audio, bit depth/codec changes, processing
   cost, licence chain and model/checkpoint hashes.

The known Wikimedia file is a **960x720, 30fps encoded digital
transfer**, not a documented original film-scanner master. The film
may have a different historical original presentation cadence, and
duplicate frames can mislead temporal algorithms. Do not infer
original frame rate from this container, or automatically decimate
frames without verifying source provenance and cadence.

**No commercial production deployment** of DART or uncured model
bundles before adapted RTN rights are resolved. Source-film
jurisdiction and digital transfer rights remain a separate issue.

## Implementation priorities

- **Now:** finish real pinned DART v1.0.0 mask/restoration evaluation
  from PR #7, with unprocessed source as control. No need to
  implement any new filter for that test.
- **After actual GPU inference:** 3-way custom vs FFmpeg deflicker
  comparison, limited to problematic scene segments, with explicit
  source-detail and historical-fade QC.
- **Only on evidence of harm from noise:** develop a classical
  motion-aware *optional* denoising stage, with a true unchanged
  baseline. Avoid removing grain as a quality target.
- **Already available:** use stabilization selectively and
  record crop/zoom, especially for old-film gate weave.
- **Do not prioritize:** pre-DART detail enhancement or automatically
  adding artificial grain.

**Bottom line:** do not copy the video's order literally.
The minimal high-fidelity candidate is
`source -> [optional bounded deflicker] -> [optional verified
gate-weave stabilization] -> DART masks + repair -> validation`;
noise suppression is conditional and any detail enhancement belongs
only to a distinctly labeled optional modernized deliverable.

Independent cultural-heritage principles:
https://www.fiafnet.org/pages/E-Resources/Digital-Statement-part-III.html
