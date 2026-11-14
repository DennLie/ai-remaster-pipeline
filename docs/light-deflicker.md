# Light Deflicker — optional conservative archive-film prepass

**Status: experimental; opt-in and CPU-only.** This change is designed for
the `Neighbors` source-quality testing roadmap, **not** a claim of a
commercially cleared full ARP pipeline.

## What it does

The default **ARP Conservative** method estimates a robust brightness value for each source frame,
compares it to neighboring frames in the **same estimated shot**, and
applies a **bounded, uniform brightness gain** where there is an isolated
discrepancy. Unlike a denoiser, it does **not** spatially blur, invent
texture, generate new pixels with a learned model, or interpolate frames.
It requires only OpenCV, NumPy and FFmpeg; no model checkpoint or GPU.

Safety features:

- **Off by default** for new projects. Existing enabled Clean Up Deflicker settings
  migrate to the independent stage without silently disabling the correction.
- The default `strength=0.65` and `max_correction=0.08` limit
  per-frame intensity gain to at most **8%** brighter or darker.
- Frames are analyzed in a two-pass process. Exposure estimates use a
  central crop with clipped black/white regions excluded to avoid
  black letterbox bars and blown-out highlights.
- Scene changes are estimated using **brightness-centered tonal
  distributions** and sudden exposure+histogram differences. Neighbor
  values are **never carried across identified cuts**.
- Monotonic exposure ramps are exempt, to help preserve purposeful
  fades and gradual lighting changes.
- A frame-size-preserving scalar RGB gain is applied using an 8-bit
  lookup table. This preserves spatial geometry and avoids a blur filter;
  **per-channel clipping and quantization are still possible**.
- The Deflicker artifact key includes the selected method-specific settings and algorithm
  version to prevent accidentally reusing an incompatible render.
- Analysis includes a JSON summary and list of applied per-frame gains
  in `.cache/deflicker/<output-stem>/`. Finished masters live in
  `intermediate/deflickered/` and have a completed provenance sidecar.

**Caveats:** A full-frame brightness adjustment cannot distinguish all
deliberate flashes or camera autoexposure from film flicker, and some
similar-looking hard cuts may evade histogram detection. A video re-encode
can still change bit-exact pixels or chroma. It does not repair scratches
or remove noise. Keep a copy of the unmodified archival source.

## ARP GUI

Enable **Light Deflicker** on Overview, then open its own tab between
**Remove Noise** and **Clean Up**. It runs independently without ComfyUI.
There are five existing methods, each compared with its current settings:

- **ARP Conservative**: strength 0–1 (default **0.65**) and maximum
  brightness correction 0–0.15 (default **0.08**).
- **FFmpeg Median / Arithmetic Mean**: temporal window **3, 5, 7 or 9** frames
  (default **5**); ARP strength and correction limit do not apply.
- **Localized short experiment**: explicit single-shot confirmation, at most
  **240 frames** and **75 million decoded pixels**. No cuts, fades or intentional
  exposure changes. Fixed 60% strength and 8% cap.
- **Localized film streaming**: separate experiment confirmation, at most
  **five buffered frames**, no fixed film-length limit. Fixed 60% strength and
  8% cap; cut/fade protection is fallible. Review motion and transitions.

Selecting methods only includes them in comparison. **Render missing variants**
explicitly generates the selected missing subset; **Use for pipeline** separately
adopts one method. Editing parameters invalidates only affected outputs.
Newly rendered comparisons do not silently change the pipeline selection.

The shared Remove Noise / Deflicker workspace provides a full-input timeline,
frame-indexed compressed playback, exact 8-bit RGB stills in the same canvases,
original/result/difference/Wipe views, shared zoom and pan, native **1:1**, and
an optional pixel grid. Each tab retains its own comparison selection, playhead
and viewport state during the session. Differences are measured before preview
compression; preview video itself is not lossless. Long-film charts sample
at most 240 positions and can miss short artifacts.

**Original is the actual stage input**: the selected Remove Noise output when
enabled, otherwise the processing source/selected section. It is not always the
unprocessed original transfer.

The pipeline order is:

```text
Source video / selected section
  → Remove Noise (optional)
  → Light Deflicker (optional, independent, model-free)
  → Clean Up (optional: DeVignette, defect masks, AI DeScratch, Dearchive)
  → Stabilization / Detail / later enabled stages
```

Old saved settings and project files retain their algorithm, parameters and
experiment confirmations. An otherwise empty Deflicker-only Clean Up stage is
disabled after migration. The GUI no longer sends Deflicker flags to Clean Up,
so enabling both stages does not apply the correction twice.

This PR changes **no DART/ProPainter/Wan/Qwen** backend or licence rule.
The opt-in Commercial model guard permits Deflicker-only processing
when the excluded model stages are disabled; it does **not** certify
FFmpeg binary distribution rights or the underlying film transfer.

## CLI

```bash
python scripts/deflicker_video.py \
  --source input/Neighbors-1920-original.webm \
  --method arp \
  --strength 0.65 \
  --max-correction 0.08 \
  --intermediate-profile lossless
```

The lossless profile uses FFV1 for intermediates. The stage retains its input
geometry, frame cadence and available audio. The legacy
`scripts/cleanup_video.py --light-deflicker ...` CLI remains supported for
existing scripts; only the GUI has moved to the independent runner.

## Validation gates

**CPU regression tests** cover stable exposure, alternating-frame
flicker, preserving monotonic fades, shot boundaries, correction bounds,
GUI/CLI parity, commercial-guard behavior and a **real encoded
FFmpeg input/output smoke test**. These are algorithm-contract tests,
not a guarantee that real archival film looks better.

For actual archival footage, compare the *same native-resolution* source
before and after correction and check (a) suspected flicker reduction
within shots, (b) unchanged shot/fade transitions, (c) absence of
new clipping, pumping or false local-color changes, (d) grain structure.
Manual image review remains required before enabling Deflicker by default.

Original digital transfer provenance:
[Wikimedia Commons Neighbors 1920](https://commons.wikimedia.org/wiki/File:Neighbors_(1920).webm).
The file's **US public-domain** designation does **not** by itself
establish rights for Germany or rights to every specific digital transfer;
do not redistribute its full contents commercially without separate review.

A comparison with post-restoration AI upscaling is deliberately outside
the scope of this PR. The mask-resolution experiment already showed
that increasing pixels **before** restoration cannot be assumed to
improve detection accuracy.


## Native 960×720 public-source validation — November 1, 2026

[Completed GitHub Actions run #41343808901](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41343808901)
ran both the full CPU test suite and a real source comparison using
[the Wikimedia Commons digital transfer](https://commons.wikimedia.org/wiki/File:Neighbors_(1920).webm).
Both jobs succeeded. The source WebM SHA-256 was
`029c419a9323158ff3f1dcdaae04a2d696c3ba030a750460c01e68733f671966`.

The automated job extracted four seconds at 00:05:00, retaining
**960×720 resolution and 30 fps (120 frames)**. With defaults
`strength=0.65`, `max_correction=0.08`:

- **45 of 120 frames** received a bounded exposure correction;
- the maximum calculated gain adjustment was **8%**;
- **one shot boundary** was detected;
- the mean absolute change in the per-frame median between adjacent
  frames was **2.61345 before** versus **1.84874 afterward**
  (approximately **29% lower**);
- the largest measured change in frame median was **10 intensity levels**
  on the 8-bit scale.

The artifact `neighbors-native-light-deflicker-comparison` retains
a labeled before/after contact sheet, 4-second split-screen H.264
video, machine-readable measurements and a provenance file.
The **left** pane is source and the **right** pane is corrected.

**What this does not prove:** lower inter-frame luminance variation is
not equivalent to a more historically correct image, nor evidence that
film texture or shadows are perfectly preserved. Adjacent differences
mix flicker, movement, cuts and legitimate exposure changes. The
poster's selected frames have visually similar detail, but a careful
review of the whole sequence and more source scenes is needed before
enabling Light Deflicker by default. This was an experimental
source-resolution test, **not** a colorization, DART or AI upscale.
