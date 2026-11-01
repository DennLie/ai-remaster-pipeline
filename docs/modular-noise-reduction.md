# Remove Noise — optional multi-backend stage (research preview)

This is the experimental GUI/CLI implementation of **stage 2** of the
video-inspired workflow: `Source → Remove Noise → Clean Up → Stabilize`.
It is **off by default**, including existing ARP projects. All algorithms
in this PR are classical, local FFmpeg implementations. **No AI weights
or DART models are bundled or downloaded.**

## GUI selection

On Overview, check **Remove Noise**. Open its stage tab and choose a
denoising algorithm and method-specific preset.

| Method | FFmpeg filter | Kind of noise | Practical caveat |
|---|---|---|---|
| Off | Stream copy/remux | Unchanged baseline | Keeps source pixels but changes container to MKV |
| hqdn3d | `hqdn3d` | Spatial and temporal noise | Can soften moving details or grain |
| NLMeans | `nlmeans` | Similar-patch spatial noise | More expensive, can smear details |
| BM3D | `bm3d` (basic) | Transform-domain image noise | High CPU cost; basic single-pass implementation |

Presets `gentle`, `balanced` and `strong` are **method-specific**,
not universal percentages. Their source-native FFmpeg parameters are
defined centrally in `scripts/denoise_video.py`. No method is inherently
better for all film sources. Grain preservation is not claimed without
an annotated benchmark, and FFmpeg filters may change pixel format or
bit depth. Run with `--intermediate-profile lossless` for FFV1,
which prevents another lossy intermediate encoder generation but is
**not** evidence that a filter preserved all input information.

Different method/preset/profile combinations write separate names under
`intermediate/denoised/`, keyed by the **original source name and settings**.
A sidecar signature holds a SHA256 source fingerprint; a changed source
invalidates cache reuse even when its name is unchanged. Each successful
variant can be inspected in the stage's intermediate file list. The
original source is never overwritten.

### CLI

```bash
python scripts/denoise_video.py \
  --source input/my_archive.mkv \
  --output intermediate/denoised/example_hqdn3d.mkv \
  --method hqdn3d --preset gentle --intermediate-profile lossless

python scripts/denoise_video.py \
  --source input/my_archive.mkv \
  --output intermediate/denoised/example_bm3d.mkv \
  --method bm3d --preset balanced --intermediate-profile lossless
```

An `off` method copies video/audio streams into MKV without filtering
(FFmpeg stream copy). On edited film sources where the output contains
audio, it is copied. Source cadence/geometry are checked before and
after filtering. External subtitles/data streams are not retained in
the first research version; keep the original for a canonical master.

## Comparison design

1. Inspect `off` and multiple methods using the *same* source
   frame indexes, at native resolution and an identical encoding
   profile. Do not compare differently downsampled thumbnails.
2. On a *known synthetic noise model*, compare to the undamaged frame
   and measure PSNR/SSIM, motion ghosting, and fine texture loss.
3. On archive footage, compare grain texture, faces, hair, text,
   real dirt/scratches, scene cuts and luminance changes by eye.
4. Only then measure **DART** mask precision/recall and restored
   output before and after a given denoiser. Denoising before DART
   can weaken precisely the tiny defects DART should detect.

A future GUI A/B/C synchronized viewer, selectable DART-before/after
placement, automatic per-scene variants and ML backends are **not
implemented in this first PR**. This release establishes the backend
selection, independent stage, non-destructive variants and testable
native video contract. The downstream GUI stages consume the chosen
variant if Remove Noise is enabled.

## AI backends are intentionally deferred

- **FastDVDnet** has published code and original checkpoints in
  [the author's repository](https://github.com/m-tassano/fastdvdnet).
  README declares MIT. A responsible integration must pin an exact
  commit and original checkpoint hash, verify the weight/distribution
  provenance, and test compatibility with current PyTorch/Windows/RTX
  drivers; historical-film performance is unknown.
- **RVRT** is a research comparison with noncommercial licensing
  for the published implementation, not a production-commercial
  default. Do not install into the main ARP environment.
- The actual licensing situation of **DART's adapted RTN code**
  remains unresolved. This PR makes no commercial DART claim.

FFmpeg docs: https://ffmpeg.org/ffmpeg-filters.html#hqdn3d
https://ffmpeg.org/ffmpeg-filters.html#nlmeans
https://ffmpeg.org/ffmpeg-filters.html#bm3d

## Testing

`tests/test_denoise_video.py` uses real FFmpeg samples with audio
and verifies the output frame count, reported FPS, picture geometry,
method-specific settings, independent artifact identities, and cache
reuse. CI runs CPU video regression and JavaScript syntax checks.

**Not yet production-approved**: test on multiple source recordings,
including NTSC/variable-FPS transfers and nonstandard pixel formats.
No default processing settings have been changed.
