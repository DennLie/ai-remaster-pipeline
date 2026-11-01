# Light Deflicker — optional conservative archive-film prepass

**Status: experimental; opt-in and CPU-only.** This change is designed for
the `Neighbors` source-quality testing roadmap, **not** a claim of a
commercially cleared full ARP pipeline.

## What it does

Light Deflicker estimates a robust brightness value for each source frame,
compares it to neighboring frames in the **same estimated shot**, and
applies a **bounded, uniform brightness gain** where there is an isolated
discrepancy. Unlike a denoiser, it does **not** spatially blur, invent
texture, generate new pixels with a learned model, or interpolate frames.
It requires only OpenCV, NumPy and FFmpeg; no model checkpoint or GPU.

Safety features:

- **Off by default** (including existing projects).
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
- The clean-up artifact key includes the selected settings and algorithm
  version to prevent accidentally reusing an incompatible render.
- Analysis includes a JSON summary and list of applied per-frame gains
  in the existing `.cache/cleanup_chunks/<film>_<identity>/` directory.

**Caveats:** A full-frame brightness adjustment cannot distinguish all
deliberate flashes or camera autoexposure from film flicker, and some
similar-looking hard cuts may evade histogram detection. A video re-encode
can still change bit-exact pixels or chroma. It does not repair scratches
or remove noise. Keep a copy of the unmodified archival source.

## ARP GUI

In **Clean Up**, select *Light Deflicker (conservative; no AI)* and leave
`Dearchive` and `AI DeScratch` off for a model-free test. A
Deflicker-only run works without ComfyUI. The additional switches are:

- `Deflicker strength`: 0–1, default **0.65**.
- `Maximum brightness correction`: 0–0.15, default **0.08**.

The order of the optional clean-up stages is:

```text
Source video
  → Light Deflicker (opt-in, model-free)
  → DeVignette (opt-in, model-free)
  → Temporal defect mask proposal (opt-in, diagnostic)
  → AI DeScratch (optional, ProPainter noncommercial)
  → Dearchive (optional, LTX subject to separate licensing)
```

This PR changes **no DART/ProPainter/Wan/Qwen** backend or licence rule.
The opt-in Commercial model guard permits Deflicker-only processing
when the excluded model stages are disabled; it does **not** certify
FFmpeg binary distribution rights or the underlying film transfer.

## CLI

```bash
python scripts/cleanup_video.py \
  --source input/Neighbors-1920-original.webm \
  --light-deflicker \
  --deflicker-strength 0.65 \
  --deflicker-max-correction 0.08 \
  --no-dearchive \
  --intermediate-profile lossless
```

The lossless profile uses FFV1 for intermediates. The normal Clean Up
output retains the source geometry, frame cadence and available audio.

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
