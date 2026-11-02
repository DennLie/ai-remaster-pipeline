# Step 3 — reproducible local GPU comparison pack (no model inference in CI)

**Status: draft research support, November 2026.** This adds a reproducible
benchmark *input and evaluation contract*, not another denoiser or
automatic commercial model. All upstream code and weights remain
opt-in outside the ARP production defaults.

## What this fixes

A prior comparison can be invalid even when each MP4 has the correct
fps and dimensions: different video decoders may seek to different
frames when a model is asked to open a long WebM at "05:00".
The model outputs can thus be one or more frames apart.

`scripts/step3_benchmark.py` first extracts **one fixed native
reference clip per scene**, then has each model process exactly
that already-extracted video **from time 0**. Every reference
has a recorded SHA-256 digest and verified frame count/cadence.
All variants are compared to the *same* prepared clip (or to
its fixed-size Lanczos reference for generative upscaling).

This separates **two distinct questions**:

- **Native 1× detail and deblur:** Original / Unsharp / CAS /
  Multiscale / Richardson–Lucy / research NAFNet / research Restormer
  all at the digital transfer's native 960×720 resolution.
- **Generative video super-resolution:** SeedVR2 / FlashVSR /
  FlashVSR+ vs a **Lanczos 1024×768 baseline**, each using
  precisely the same prepared native input. 1024×768 is
  an aspect-correct **1.0667×** output for this particular
  960×720 4:3 transfer. Do **not** label it native 1×
  enhancement or 2× upscaling.

Native and VSR diagnostic JSON files remain separate; these
scores are **not objective quality ratings** or evidence that
new historical detail was truly reconstructed.

## 1. Prepare the verified input, no GPU required

Choose a *reviewed* location in the actual source transfer.
The **05:00** excerpt below was used in prior experiments;
the label does **not** claim the clip is a face, intertitle or
other specific scene. Choose additional shots only after a human
has reviewed their contents and cut boundaries.

```bash
python scripts/step3_benchmark.py prepare \
  --source "input/Neighbors_original.webm" \
  --source-sha256 029c419a9323158ff3f1dcdaae04a2d696c3ba030a750460c01e68733f671966 \
  --output-dir "benchmark/step3_neighbours" \
  --shot sample_0500=300:24 \
  --target-width 1024 --target-height 768
```

To create multiple independent shots, repeat `--shot
name=start_seconds:frame_count`. Maximum eight shots and
72 frames per shot. There is no automatic film download.

The command creates:

```text
benchmark/step3_neighbours/
  manifest.json
  sample_0500/
    native.mkv                 # FFV1 source clip, 960x720
    lanczos_reference.mkv      # same frames, 1024x768
```

Both are lossless intermediates **relative to the decoded 8-bit
video**. They do not reproduce original analog film information
lost in the digital transfer. The command rejects an existing
benchmark directory rather than overwriting it.

## 2. Run models on `sample_0500/native.mkv` at time zero

### Native classical baselines

```bash
python scripts/detail_enhance_video.py \
  --source "benchmark/step3_neighbours/sample_0500/native.mkv" \
  --output "benchmark/candidates/cas_gentle.mkv" \
  --method cas --preset gentle --mix 100 \
  --intermediate-profile lossless

python scripts/detail_enhance_video.py \
  --source "benchmark/step3_neighbours/sample_0500/native.mkv" \
  --output "benchmark/candidates/richardson_lucy_gentle.mkv" \
  --method richardson_lucy --preset gentle --mix 100 \
  --intermediate-profile lossless
```

**Important:** Richardson–Lucy assumes an arbitrary Gaussian
PSF; its outputs need not be deblurred accurately. Native
backends use a stage cache `.mkv.sig.json` with source hash.

### NAFNet and Restormer (research only)

NAFNet is in **PR #13, a sibling research branch**; it is **not**
automatically included in the Step-3 benchmark branch.
Restormer is inherited from PR #15. Invoke each verified
research adapter separately with:
`--source .../sample_0500/native.mkv`,
`--start-seconds 0`,
`--frames 24`,
`--research-only`, exact externally reviewed model checkout,
weights and SHA-256. They generate `.research.json` provenance
beside the output.

Neither trained image model was GPU-tested in GitHub CI.
Even if the run succeeds, it can flicker across frames or
invent fabric, hair or facial details.

### FlashVSR+ and existing SR backends (research only)

For the inherited FlashVSR+ research adapter (PR #16):

```bash
python scripts/run_flashvsr_plus_research.py \
  --source "benchmark/step3_neighbours/sample_0500/native.mkv" \
  --output "benchmark/candidates/flashvsr_plus_turbo_1024x768.mp4" \
  --checkout "external/FlashVSR_Plus" \
  --manifest "external-weights/flashvsr_plus_turbo_sha256.json" \
  --mode turbo --width 1024 --height 768 \
  --start-seconds 0 --frames 24 \
  --research-only --dry-run
```

This dry-run requires the manually supplied *real* weights and
full per-asset digests, but does not execute the model. Only
after trust, CUDA compatibility and licensing checks should an
operator omit `--dry-run` to run on RTX 4090.

SeedVR2 and FlashVSR are **already ARP Upscaling candidates**.
For a fair experiment they must be given the *same prepared
native.mkv input* and emit the same target output dimensions.
Existing GUI outputs without per-source provenance must be
explicitly marked unverified in the evaluation (see below).

## 3. Evaluate the actual outputs

```bash
python scripts/step3_benchmark.py evaluate \
  --manifest "benchmark/step3_neighbours/manifest.json" \
  --shot sample_0500 \
  --native CAS="benchmark/candidates/cas_gentle.mkv" \
  --native RL="benchmark/candidates/richardson_lucy_gentle.mkv" \
  --vsr FlashVSRPlus="benchmark/candidates/flashvsr_plus_turbo_1024x768.mp4" \
  --output-dir "benchmark/evaluation_sample_0500"
```

A run without generated outputs will correctly fail. The
evaluation validates:

- exact prepared reference hashes, geometry, fps and number of
  decoded frames, for both native and VSR branches;
- each candidate's geometry, fps and exact frame count;
- **matching input-video SHA-256** in `.research.json` or the
  ARP `.mkv.sig.json`, when available;
- `start_seconds=0` and the exact number of requested frames
  for research-adapter variants;
- no output overwrite or self-referential benchmark paths.

If a model's output lacks a compatible provenance sidecar,
default evaluation **fails closed**. An operator may pass
`--allow-unverified` to accept externally produced outputs;
they are clearly labeled
`UNVERIFIED_MANUAL_ALIGNMENT` in the summary.
This option does **not** prove those outputs are aligned.

The result directory contains:

```text
step3_evaluation.json
native/denoising_comparison.json
native/denoising_comparison.jpg
vsr/denoising_comparison.json
vsr/denoising_comparison.jpg
```

The shared `compare_denoising.py` engine computes mean pixel
difference, a Laplacian spatial-frequency proxy and newly
clipped-pixel fractions. Values are only descriptive.
**Pixel shifts, high frequencies, color differences and
compression all affect metrics.** No quality winner should
be selected automatically. For artistic and archival decisions
also evaluate full-resolution footage, multi-shot motion,
hallucination, grain texture, temporal flicker, halos, text
and the original defect masks.

## 4. Acceptance and outstanding work

| Check | Required before production |
|---|---|
| CPU source bundle and manifest | Automated tests on synthetic FFV1 clips |
| Native 1x vs target-sized VSR separation | Enforced by compare mode |
| RTX 4090 driver / CUDA / VRAM / frame times | **Not yet tested** |
| NAFNet / Restormer / FlashVSR+ actual inference | **Not yet tested** |
| Other existing SeedVR2 / FlashVSR at identical frames | **Not yet completed** |
| Checkpoint, training-data, decoder, dependency rights | **Separate review required** |
| Sampled scene selection and film-grain fidelity | **Human review required** |
| Main branch integration | **Not authorized, intentionally unchanged** |

A safe first RTX4090 pilot tests **24 frames per run** on
one reviewed shot, measures peak VRAM and frame time, then
moves on to several different shots. No automatic full-film
denoising, pre-DART sharpening or model downloads are part
of this benchmark tool.

The original Wikimedia Commons upload is a 960×720 digital
transfer, not an authenticated original negative scan.
Its US public-domain listing does not automatically settle
German/EU film, transfer or music rights.
