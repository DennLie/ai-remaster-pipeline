# Denoising comparison: one source, several interchangeable backends

**Research configuration / PR #9**, not a recommendation to remove original film grain. No neural model is currently wired into the GUI. `main` is unchanged.

## Available in ARP

The optional **Remove Noise** stage precedes Clean Up in the experimental GUI. Select:
`Off`, `hqdn3d`, `NLMeans`, or `BM3D`, then one of the
method-specific `Gentle`, `Balanced`, `Strong` presets.
Settings produce separate cached outputs. The original is not
overwritten and the downstream DART/cleanup backend is unchanged.

For comparability, a [GitHub Actions workflow](../.github/workflows/denoise-tests.yml)
takes **24 consecutive 960×720 digital-transfer frames** of
*Neighbors* (1920) starting at 00:05:00 and runs **all three**
classical denoisers using their **Gentle** preset. The pipeline
writes a 4-column synchronized H.264 preview, fixed-frame
contact sheet, and `denoising_comparison.json`. It checks
native frame alignment, geometry, frame rate and incomplete outputs.

The source is the Wikimedia Commons original uploaded 960×720
WebM, not a verified original 35mm scan. Rights outside the US,
including Germany, are not established merely by the file's
US public-domain marking:
https://commons.wikimedia.org/wiki/File:Neighbors_(1920).webm

Metrics include mean pixel changes, changed-pixel fraction above an
8-bit threshold, high-frequency Laplacian energy and newly clipped
color channels. **Lower high-frequency energy is not necessarily
better:** it can mean lost hair, clothing texture, film grain or small
scratches needed by DART. These scores are diagnostic descriptors,
*not* trained quality metrics or ground-truth restoration accuracy.
A 24-frame sample is too short to evaluate all temporal artifacts
or judge a full feature film. Evaluate diverse scenes separately.

## Research candidate: FastDVDnet

Official upstream: https://github.com/m-tassano/fastdvdnet

The official repository's `LICENSE` and README identify the **source
code as MIT**. It includes `model.pth` and
`model_clipped_noise.pth`; the README describes them as trained on
Gaussian noise (in the approximate stated training range 5–55
on 8-bit values), and notes historical PyTorch/DALI training-version
differences. A MIT-licensed repository is not proof that all training
data and *weights* have been separately cleared for ARP's specific
commercial deployment, nor that their synthetic Gaussian training
distribution generalizes to 1920s footage.

**Decision:** leave FastDVDnet as a *candidate* in this phase.
Do not silently download or install weights in the default ARP
environment. Before allowing production use, separately record:
the exact model checksum, repository commit, dependency and weight
licence chain, expected GPU VRAM, tested noise levels, provenance of
training data and original-source comparison. The user's RTX 4090
is a future optional local test environment.

## Longer-term UX

A synchronized A/B/C preview should independently select
original and cached candidate videos, offer a full-resolution
100% crop, a difference heatmap and optional grain/detail inspection,
and record metadata for each render. The artifacts from this
workflow are an **offline comparison prototype**, not yet an
interactive browser A/B/C implementation.

Next evaluate the effect of each noise-reduction choice on **DART's
real masks and repaired frames**, once the separate DART research
runner is available; do not optimize for merely smoother pictures.


## First verified native-source results (November 1, 2026)

The [GitHub Actions run #41357324909](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41357324909)
completed **both unit-test and real digital-source experiment jobs** successfully.
A previous identical-code experimental run #41357276847 had also
succeeded. This is a 24-frame (0.8-second), 30fps clip at 960×720,
from 00:05:00 of the 2016 Commons original uploaded WebM.

Source WebM SHA-256:
`029c419a9323158ff3f1dcdaae04a2d696c3ba030a750460c01e68733f671966`.

| Descriptive metric | Original | hqdn3d Gentle | NLMeans Gentle | BM3D Gentle |
|---|---:|---:|---:|---:|
| Mean absolute channel change from source (8-bit scale) | 0 | 0.0617958 | 0.6305905 | 0.0004066 |
| Mean absolute spatial Laplacian | 12.2333814 | 12.2015616 | 10.1587062 | 12.2315088 |
| Fraction newly clipped channels | 0 | 0.0002484 | 0.0019633 | 0.0000017 |
| Pixels with any channel changed ≥ 8/255 | 0 | 0 | 0 | 0 |

**Interpretation:** the preset strengths are not equivalent across
algorithms. NLMeans has the biggest visible/quantitative effect in this
sample. BM3D Gentle barely changes the source, raising the question
whether that preset is meaningful for this specific film/codec.
hqdn3d Gentle stays close to the source. A lower spatial Laplacian
can also indicate *lost fine detail or film grain*, not just less
noise. Do not rank restoration quality using these proxies alone.

GitHub artifact `neighbors-native-denoising-four-method-comparison`
contains a four-column synchronized MP4 preview, frame-indexed JPEG,
JSON report and source checksum. The original film and all outputs
remain separate; no DART model or neural denoiser was executed.

**Next test:** longer excerpts with cuts, moving subjects and prominent
grain; inspect matching frames at 100% and compare masked DART
restoration with and without preprocessing on the RTX 4090.
