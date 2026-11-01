# Native 1× Detail Enhancement — implementation and evaluation

**Draft branch `feature/native-detail-enhancement`**, based on experimental
`feature/denoise-synchronized-comparison` (PR #10). No production `main`
changes, learned models or checkpoint downloads.

## Why this exists

ARP's Video Upscaling stage already supports SeedVR2, FlashVSR and LTX CQ
for generative restoration and reconstruction, but there was no
independent, **native-resolution** Step 3. This adds an optional
post-Clean-Up, post-Stabilization and pre-Outpainting detail stage
without forcing an upscale or modifying original source images.

## Selectable backends

| Backend | Gentle | Balanced | Strong |
|---|---|---|---|
| `off` | no change | no change | no change |
| `unsharp` | luma 3×3, 0.25 | luma 5×5, 0.55 | luma 5×5, 0.95 |
| `cas` | strength 0.18 | strength 0.38 | strength 0.60 |
| `multiscale` | luma 3×3 @ 0.12 + 7×7 @ 0.10 | luma 3×3 @ 0.30 + 7×7 @ 0.22 | luma 3×3 @ 0.52 + 7×7 @ 0.38 |

All presets are intentionally conservative **starting points**:
not certified optimal, not calibrated to equivalent apparent sharpness,
and not historically verified. Multiscale currently means **two
sequential unsharp radii**, not mask-aware decomposition or true
deblurring. Color/chroma sharpening is disabled for the current
FFmpeg filter graph (CAS `planes=1`, Unsharp `chroma_amount=0`).

A separate integer `mix` control (0–100%) linearly blends
native video output with its original upstream frame. A zero mix
or method Off stream-copies source video/audio, while other options
render an independent .mkv profile; lossless profile uses FFV1.
This is pixel-domain interpolation, not an artistic/gamma-aware blend.

## Usage (on a research branch)

GUI: Overview → enable **Detail Enhancement**. Navigate to the
Detail Enhancement tab, select one of four methods and choose
Gentle / Balanced / Strong, set image mix, then Run Detail Enhancement.

This stage uses the direct upstream video: the stabilization output
when enabled, otherwise the cleaned or denoised clip as available.
It precedes Outpainting, Shot Detection, Colorization and Upscaling.
It is **disabled by default**. The existing A/B/C viewer from PR #10
can select the stage's direct input plus saved detail-enhanced
outputs, and synchronizes preview playback approximately.
The video-frame comparison CLI remains authoritative for
frame-accurate diagnostics.

Direct standalone CLI:

```bash
python scripts/detail_enhance_video.py \
  --source input/neighbors_960x720.mkv \
  --output intermediate/detail_enhanced/neighbors_unsharp_gentle_50.mkv \
  --method unsharp --preset gentle --mix 50 \
  --intermediate-profile lossless
```

Output filenames and a sidecar signature distinguish upstream
source, backend, preset, mix and codec profile; the source itself is
not overwritten and an exact-matching result is reused.

## Benchmark and verification

CPU CI runs `tests/test_detail_enhancement.py`:
- all filter graphs and parameter validation;
- actual 12-frame FFV1 video with PCM audio for Off, Unsharp,
  Unsharp @45% mix, CAS and Multiscale;
- correct geometry, cadence, decoded frame count, audio stream,
  original-bytes preservation and caching;
- GUI stage order and Outpainting upstream identity;
- regression tests for PR #9 and #10 remain in the workflow.

A separate conditional workflow job processes **36 same native
960×720 frames from 05:00 of the verified Commons uploaded WebM**
(sha256 `029c419a9323158ff3f1dcdaae04a2d696c3ba030a750460c01e68733f671966`).
It exports 4-way H.264 preview with original/unsharp/CAS/multiscale,
native-frame contact sheet and JSON metrics.

The shared `scripts/compare_denoising.py` measures mean absolute
changes, gradient energy, near clipping; these are **descriptive
proxies, not ground-truth restored-detail fidelity**. An increase in
Laplacian energy can mean accentuated scratches/grain, not recovered
detail. Browser H.264 comparison is separately compressed and not
frame-exact on user seek.

Manual tests on additional shots are needed to decide if a better
preset exists, whether to include a proper PSF deblurring method,
and whether gentle sharpening actually improves a DART repair.
A comparison of modern AI detail reconstruction via existing
SeedVR2/FlashVSR/LTX-CQ belongs in the next GPU research experiment,
with checkpoint and commercial-licensing review.

## Source rights and accurate attribution

`Neighbors` is from
https://commons.wikimedia.org/wiki/File:Neighbors_(1920).webm ;
the uploaded 960×720 30fps file is a digital transfer, not established
as an original film scan. US public-domain metadata does not settle
rights in German/EU territories or rights to its transfer/soundtrack.
FFmpeg binary licensing depends on actual build; no bundled model,
training data, weight or proprietary restoration system is part of
this classical detail-enhancement step.

## Provenance and limitations

This work follows the independent research note
`docs/detail-enhancement-sota-review-2026.md`
(on the separate PR #7 research branch), with the deliberate
conservation-oriented default **Off**, but supports stronger
modernized outputs on explicit user selection. It implements
sharpening and contrast enhancement, **not** learned
native-resolution deblurring or "restoration" of invisible details.

## Verified real-film first-pass results (November 2026)

[GitHub Actions run #41361640082](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41361640082):
**CPU unit and regression tests succeeded, as did the real-original-film comparison.**
The verified Commons digital transfer was sampled at 00:05:00 for
36 consecutive 960×720, 30 fps frames (1.2 seconds). All three
methods used their **gentle** preset, full 100% enhanced image mix,
and lossless intermediate output.

| Diagnostic, not fidelity ranking | Original | Unsharp | CAS | Multiscale |
|---|---:|---:|---:|---:|
| Mean 8-bit absolute channel difference from the source | 0 | 0.2683079 | 0.4936153 | 0.5560888 |
| Mean absolute spatial Laplacian | 12.1746443 | 14.0294955 | 15.3806047 | 16.0708239 |
| Newly clipped channel fraction | 0 | 0.0015261 | 0.0025531 | 0.0032160 |

Interpretation: multiscale is most invasive of these **unmatched**
preset strengths on this one short segment, Unsharp the least.
Greater Laplacian response means higher spatial frequencies;
it may indicate sharpening, amplified grain/scratches or
unwanted halos, not authentically recovered details. Clipping is
small but nonzero and should be inspected on high-contrast edges.
There is **no perceptual quality winner** established by these numbers.

The workflow artifact `neighbors-native-detail-four-method-comparison`
includes 4-way same-frame video, JPEG contact sheet and JSON summary.
The preview H.264 video is downscaled to four 320×240 panels for
side-by-side watching; numerical analysis used 960×720.
Longer scene diversity, 100%-scale inspection and comparison
against controlled synthetic blur ground truth remain necessary.
