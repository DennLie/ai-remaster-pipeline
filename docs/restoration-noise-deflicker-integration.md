# ARP integration: Remove Noise, Deflicker and the following restoration stages

**2026-11-03 — opt-in integration QA draft; no merge to `main`.**

## Already inherited, not duplicate work

The current integration base, `feature/localized-deflicker-feature-films-streaming`
(Draft PR #29), is already a descendant of the shared noise/detail development
chain. It contains the working classical Remove Noise backend from
[PR #9](https://github.com/DennLie/ai-remaster-pipeline/pull/9) and its
synchronized browser A/B/C comparison from
[PR #10](https://github.com/DennLie/ai-remaster-pipeline/pull/10).
**Do not cherry-pick/merge those PRs again or replace the newer shared GUI
JavaScript with the older viewer code.**

- **Present:** Denoise `off / hqdn3d / nlmeans / bm3d`, per-method
  `gentle / balanced / strong`, independent cached outputs and an A/B/C
  original-vs-variants preview. Optional `lossless` FFV1 intermediates.
- **Present:** ARP Conservative, FFmpeg Median, FFmpeg Arithmetic Mean,
  Localized short experiment and Localized movie-length streaming, plus
  source/result still inspection and turquoise/orange directional change
  visualizations.
- **Present:** optional 1× Detail Enhancement and Stabilization; later ARP
  outpainting, colourization, audio and upscale stages follow their existing
  independent settings.
- **Not adopted:** [PR #11](https://github.com/DennLie/ai-remaster-pipeline/pull/11)
  FastDVDnet is a separate opt-in **research-only CLI**. It has no approved
  executable checkpoint or GUI backend in this branch. Do not auto-download
  or auto-enable pretrained model assets. No actual FastDVDnet GPU inference
  on the user's source has been verified.
- **Research, not release evidence:** PRs #20/#22 measured controlled
  synthetic-noise recovery on short excerpts of digitized historic films.
  Genuine aged-film grain fidelity, long-film artifacts and DART compatibility
  remain unverified.

## The live pipeline order — important

The current *implementation* is:

```
Original selected source / source section
  -> Remove Noise (optional; selected algorithm + preset)
  -> Clean Up (Deflicker and optional other cleanup operations)
  -> Stabilization (optional)
  -> Detail Enhancement (optional)
  -> downstream outpaint / colour / audio / upscale stages
```

**This is NOT the original desired "Step 1 Deflicker, Step 2 Remove Noise" order.**
It is a pre-existing implementation decision, not proof that denoising before
flicker correction is scientifically superior. Reversing these two stages would
require changing the source routing, stage order, settings semantics, artifact
dependencies and tests. Do not silently reverse it on this QA branch; evaluate
both orders on matched clips before introducing an order selector.

Enabling Remove Noise makes Clean Up consume the **denoised output**. Changing
Denoise method/preset changes Clean Up's expected source and result/cache
identity. Disabling Remove Noise restores Clean Up's input directly to the
configured original video. This relationship is covered by
`tests/test_denoise_deflicker_integration.py`.

## First Windows test — same source and frames

Use the existing **Draft PR #30** branch (when available); keep your original
film, old branches and outputs in place.

1. Start GUI, refresh the browser with **Ctrl+F5**, open Overview and choose
   a short **single-scene** source (the already tested 72-frame clip works).
2. Turn **Remove Noise = on** and **Clean Up = on**. Turn off the unrelated
   restoration stages for the first isolated test.
3. In **Remove Noise** choose `NLMeans / Gentle` for a controlled candidate
   and leave original/Off, HQDN3D and BM3D as separate comparison variants.
   Do not infer that `NLMeans` best preserves authentic film grain merely
   because it performed best on injected synthetic noise.
4. In **Clean Up**, enable **Light Deflicker** and select
   **Localized · Langfilm-Streaming (experimentell)**. Confirm the
   film-experiment warning. **Disable DeArchive, DeScratch, DeVignette
   and Detect Film Damage** for this controlled research job.
5. Select a lossless intermediate profile if disk space permits.
   Run the pipeline. The denoised output is the next stage's input.
6. Review Remove Noise **A/B/C** with the *same* source start and method
   variants, then Clean Up's original/result videos, exact still frames
   and the turquoise/orange brightness-change chart.
7. Repeat **with Remove Noise disabled** but all Clean Up settings unchanged.
   This is an essential control for detecting whether denoising removes
   real grain, conceals defects or changes Deflicker behavior.
8. After verified cuts/fades/motion on a few-minute section, test a longer
   film. Full-feature-film runtime, storage requirements and restoration
   quality on Windows/RTX hardware are **still to be measured**.

**Note:** The A/B/C videos are H.264 *previews* and their HTML-video seeking is
approximate; the comparison is not scientific ground truth. Use matched exact
native FFV1 frames and known-reference measurements when deciding a release.
The long-film change chart samples at most 240 sparse positions; it can miss
short artifacts and is not a full-film QA record.

## Tests and acceptance gates

The dedicated cross-stage integration tests verify:

- GUI command routing: **Remove Noise -> Clean Up -> Stabilization -> Detail**
  and source/output dependency propagation;
- if a Denoise preset changes, downstream Clean Up identities also change;
- when Denoise is off, Clean Up reverts to the original source;
- a real **24-frame FFV1-with-audio** file survives `NLMeans Gentle ->
  Localized streaming with oscillation guard` and retains frames, dimensions,
  FPS and an audio stream;
- both intermediate and final sources remain unmodified; repeat runs reuse
  valid signatures.

They demonstrate plumbing and regression safety, **not** that the filtering
order or any method is a film-grain-preserving restoration winner.

A production release needs a separate, representative multi-scene/multi-film
assessment of: original grain, moving hair/cloth and faces, fine scratches,
subtle/local flicker, scene transitions, unintended exposure changes, temporal
ghosting, final audio synchronization, and long-film resource use. Compare
method selection *and pipeline order* using identical original frames.
No automatic merge into `main`; retain the established Draft PR stack.
