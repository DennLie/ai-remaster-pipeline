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
  `gentle / balanced / strong`, independent cached outputs and a matrix-selected
  original-vs-variants workspace. Optional `lossless` FFV1 intermediates.
- **Present:** ARP Conservative, FFmpeg Median, FFmpeg Arithmetic Mean,
  Localized short experiment and Localized movie-length streaming in an
  independent **Light Deflicker** tab. It reuses the Remove Noise comparison:
  frame-locked playback, exact stills, Wipe, differences, zoom and native 1:1.
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
  -> Light Deflicker (optional; selected method + its current parameters)
  -> Clean Up (optional other cleanup operations; no GUI Deflicker pass)
  -> Stabilization (optional)
  -> Detail Enhancement (optional)
  -> downstream outpaint / colour / audio / upscale stages
```

**This is NOT the original desired "Step 1 Deflicker, Step 2 Remove Noise" order.**
The approved separation preserves this processing order. It is not proof that
denoising before flicker correction is scientifically superior. Reversing these two stages would
require changing the source routing, stage order, settings semantics, artifact
dependencies and tests. Do not silently reverse it on this QA branch; evaluate
both orders on matched clips before introducing an order selector.

Enabling Remove Noise makes Light Deflicker consume the **selected denoised
output**. Its comparison's Original is that actual input, not the raw transfer.
Changing Denoise method/preset changes Deflicker's expected source and
downstream identities. Disabling Remove Noise restores Deflicker's input to
the configured source/section. Clean Up consumes Deflicker's selected output
when enabled, otherwise the denoised/source input. These relationships are
covered by `tests/test_deflicker_stage.py` and
`tests/test_denoise_deflicker_integration.py`.

Raw saved settings/projects migrate legacy Clean Up Deflicker values before
defaults merge. Explicit new-stage choices are preserved. Deflicker-only
legacy Clean Up becomes disabled instead of an empty job; mixed Clean Up
projects keep their remaining operations. The legacy Clean Up CLI still
supports its old flags, but GUI commands never pass them.

## First Windows test — same source and frames

Use the existing **Draft PR #30** branch (when available); keep your original
film, old branches and outputs in place.

1. Start GUI, refresh the browser with **Ctrl+F5**, open Overview and choose
   a short **single-scene** source (the already tested 72-frame clip works).
2. Turn **Remove Noise = on** and **Light Deflicker = on**. Turn off Clean Up
   and the unrelated restoration stages for the first isolated test. Choose a
   lossless intermediate profile before rendering if disk space permits.
3. In **Remove Noise** include `NLMeans / Gentle`, render the missing candidate,
   then adopt it with **Use for pipeline**. Original remains the reference;
   HQDN3D and BM3D stay separate comparison variants.
   Do not infer that `NLMeans` best preserves authentic film grain merely
   because it performed best on injected synthetic noise.
4. In **Light Deflicker**, include the methods you want to compare and confirm
   **Render missing variants**. The five methods use their existing parameters,
   not a nine-preset matrix. Localized short and streaming each require their
   own explicit confirmation. Short mode requires one shot, at most 240 frames
   and 75 million decoded pixels; streaming buffers at most five frames.
   **Use for pipeline** explicitly adopts one method; selection/rendering alone
   never changes the pipeline choice.
5. Run the pipeline after adopting the desired variants. The denoised output
   is the next stage's input; matching generated work is reused.
6. Review both tabs with their shared full-input comparison: Original, Result,
   Difference, exact stills and Wipe. Zoom/pan/native 1:1 and the playhead stay
   independent per tab. Check the brightness-change chart, without treating
   pixel changes as evidence of improved film restoration.
7. Repeat **with Remove Noise disabled** but all Deflicker settings unchanged.
   This is an essential control for detecting whether denoising removes
   real grain, conceals defects or changes Deflicker behavior.
8. After verified cuts/fades/motion on a few-minute section, test a longer
   film. Full-feature-film runtime, storage requirements and restoration
   quality on Windows/RTX hardware are **still to be measured**.

**Note:** Playback uses frame-indexed native-size H.264 comparison reels. It is
frame-locked but compressed; exact stills decode matched 8-bit RGB frames into
the same viewports. Neither mode is scientific ground truth. Use matched
native master frames and known-reference measurements when deciding a release.
The long-film change chart samples at most 240 sparse positions; it can miss
short artifacts and is not a full-film QA record.

## Tests and acceptance gates

The dedicated cross-stage integration tests verify:

- GUI command routing: **Remove Noise -> Light Deflicker -> Clean Up -> Stabilization -> Detail**
  and source/output dependency propagation;
- if a Denoise preset changes, downstream Deflicker/Clean Up identities also change;
- when Denoise is off, Deflicker reverts to the processing source/section;
- a real **24-frame FFV1-with-audio** file survives `NLMeans Gentle ->
  Localized streaming with oscillation guard` and retains frames, dimensions,
  FPS and an audio stream;
- both intermediate and final sources remain unmodified; repeat runs reuse
  valid signatures;
- migration through real settings/project loading, five existing algorithms,
  fractional FPS, audio retention, explicit subset rendering, experimental
  confirmations, server-owned output paths and stale-output rejection;
- independent per-tab selections/playheads/viewport state, with shared still,
  Wipe and playback implementation rather than a copied viewer.

They demonstrate plumbing and regression safety, **not** that the filtering
order or any method is a film-grain-preserving restoration winner.

A production release needs a separate, representative multi-scene/multi-film
assessment of: original grain, moving hair/cloth and faces, fine scratches,
subtle/local flicker, scene transitions, unintended exposure changes, temporal
ghosting, final audio synchronization, and long-film resource use. Compare
method selection *and pipeline order* using identical original frames.
No automatic merge into `main`; retain the established Draft PR stack.
