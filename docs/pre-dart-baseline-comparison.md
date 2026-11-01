# ARP Deflicker and Stabilization — first reproducible baseline comparison

**Experimental, no changed production defaults.** Goal: answer whether
our existing classic pre-DART steps outperform a simpler baseline,
rather than replacing them prematurely with a purported SOTA model.

This branch is stacked on PR #6 and does **not** merge into `main`.

## What it measures

`scripts/compare_pre_dart_baselines.py` decodes exactly aligned clips
and stops on mismatched frame count, geometry, or reported frame rate.
It produces:

- `pre_dart_baselines.json`: native width, height, fps and frame count,
  central median luminance mean absolute variation between consecutive
  frames (excluding any user-marked shot cuts), mean per-channel absolute
  pixel difference from the original, and new near-black / near-white
  clipped channels.
- `paired_pre_dart_frames.jpg`: side-by-side **identical frame numbers**
  at a matching display size. The image is a contact sheet of the first
  3 frames (preview only); it is **not** a complete visual film inspection.

These are diagnostics, **not** image-fidelity scores or repair
accuracy. Reducing brightness variation could erase genuine scene
lighting and original flicker. Low absolute pixel difference can
mean ineffective processing; large differences can result from valid
stabilization geometry and codec choices. Cropping and black
borders must be evaluated separately. The original digital transfer
is preserved and never overwritten.

## Automatic synthetic experiments

CI runs `tests/test_pre_dart_baselines.py` on temporary lossless,
synthetically degraded material:

1. A textured, high-contrast still shot is alternately exposed at
   88% and 112%. Compare ARP Light Deflicker versus FFmpeg
   `deflicker=size=5:mode=median` at the same frame rate.
2. The same texture is shaken with known, tiny 2–3px translations
   as stand-in for film transport weave. Test ARP's genuine
   `scripts/stabilize_video.py` on **zero fixed zoom** versus the
   **current 3% fixed zoom** (both two-pass libvidstab).
3. A separate test fails if variants contain too few frames
   or if the editor is told to exclude a non-existent cut boundary.

This is a repeatable **integration and diagnostic contract test**,
not a demonstration of state-of-the-art restoration. In particular,
a synthetic jittered graphic is not historical camera motion.

## Real *Neighbors* and manual review

The existing GitHub Actions comparison in PR #6 produced a real
source-resolution side-by-side test (960x720 WebM digital upload,
four-second source excerpt at 05:00) without AI restoration.
Its key proxy was mean absolute adjacent-frame median-luma
difference **2.61345 before to 1.84874 after** with 45 of 120
frames corrected. That was a *descriptive* first result,
not a definitive image-quality finding.

For a direct three-way real-film comparison, use that same
native excerpt as `source.mkv` and produce:

```bash
# The branch includes the ARP Light Deflicker CLI.
python scripts/cleanup_video.py \
  --source source.mkv --output arp_deflicker.mkv \
  --light-deflicker --no-dearchive --intermediate-profile lossless

ffmpeg -hide_banner -y -i source.mkv \
  -vf "deflicker=size=5:mode=median" -an \
  -c:v ffv1 -pix_fmt yuv420p ffmpeg_deflicker.mkv

python scripts/compare_pre_dart_baselines.py \
  --original source.mkv \
  --variant arp=arp_deflicker.mkv \
  --variant ffmpeg=ffmpeg_deflicker.mkv \
  --output-dir comparison-deflicker
```

For stabilization, choose a real clip **with visible film transport
weave**, preferably with black overscan frame edges. Camera pans
cannot be interpreted as errors to remove. Test:

```bash
python scripts/stabilize_video.py --source source.mkv \
  --output stable_0.mkv --zoom 0 \
  --single-shot --intermediate-profile lossless

python scripts/stabilize_video.py --source source.mkv \
  --output stable_3.mkv --zoom 3 \
  --single-shot --intermediate-profile lossless

python scripts/compare_pre_dart_baselines.py \
  --original source.mkv \
  --variant no_zoom=stable_0.mkv --variant fixed_zoom=stable_3.mkv \
  --output-dir comparison-stabilizer
```

`--single-shot` is appropriate **only when the user has
confirmed** the clip contains one uninterrupted shot. On a
multi-shot video, use automatic scene-aware ARP stabilization instead.

If there is a reviewed cut between frames 24 and 25, pass
`--cut-after-frame 24`; no temporal flicker statistic should smooth
across that real edit.

### What the next evaluation must quantify

- Deflicker on multiple cuts, fades, flashes, moving objects,
  grain-heavy and high-contrast frames. Prefer known synthetic
  ground-truth exposure maps for claims of true error reduction;
  take a blind subjective review on archival footage.
- Stabilization: quantified *residual* background jitter against
  known transform truth, validity mask, black-edge area, and
  details lost through interpolation. **Do not conflate smoother
  camera pans with less gate weave.**
- Default ARP stabilizer currently uses `zoom=3`, so the
  theoretical crop from pure 1.03x scaling is approximately
  `1 - 1/1.03**2 = 5.74%` of the area. The actual image
  footprint additionally depends on the motion correction and
  sampling; this theoretical value is not a measured lost-area rate.
- For a source-preserving archival master, evaluate **zoom=0**
  with visible validity borders before choosing any irreversible
  crop. Do not silently fill missing image detail using generative AI.
- Rerun identical input through pinned external DART only **after**
  the raw-original mask/repair baseline is available; evaluate
  annotated masks and real restored frames, not just a smoother look.

## Licensing and evidence

All comparisons here are **classical OpenCV/FFmpeg and no AI weights**.
Current ARP commercial model guard is unchanged. The DART adapted RTN
third-party rights issue still blocks any production/commercial
release claim; the original transfer's German/EU copyright context
is also separate.

Research context: [Deflicker/Stabilization SOTA review](https://github.com/DennLie/ai-remaster-pipeline/blob/feature/dart-external-mask-and-restoration-eval/docs/deflicker-stabilization-sota-review.md).
FFmpeg upstream definitions:
https://ffmpeg.org/ffmpeg-filters.html#deflicker
https://ffmpeg.org/ffmpeg-filters.html#vidstabtransform
