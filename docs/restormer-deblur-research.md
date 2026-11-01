# Restormer Motion Deblurring — research-only ARP comparison

**Status 2026-11-02:** Research adapter; not a normal Step-3 GUI selection
and not cleared for any commercial deployment. It remains in a draft
branch stacked on classical Richardson–Lucy PR #14 (which extends
native Detail Enhancement PR #12).

## Why compare Restormer and NAFNet?

- **Restormer** is a transformer-based single-image restoration
  model (CVPR 2022) with an official **Motion Deblurring** checkpoint.
- **NAFNet** is a different, convolutional single-image architecture.
  Its isolated research runner is in PR #13.
- Richardson–Lucy is a classical *assumed-PSF* deconvolution,
  added in PR #14.
- All three can operate at the original 960×720 resolution.
  None can be presumed to recover verified historical details from
  a 1920s film just because an output looks sharper.
- Both trained image models may alter grain, create ringing, or
  flicker frame by frame. Their GoPro training domain differs
  from this 2016 digital transfer.

Upstream pinned at
`68dc6ac472db26f16361150cb7a96a1bc87da93f`:
https://github.com/swz30/Restormer

The official code is under MIT in `LICENSE.md`.
An MIT source-code license does **not** by itself prove commercial
rights over every supplied weight, dataset, dependency, transfer,
or original film content.

## Fail-closed research adapter

`scripts/run_restormer_deblur_research.py`:

1. Requires explicit `--research-only` (no default inference).
2. Requires a **clean external checkout** at the exact commit above.
   The weights must be stored **outside** that checkout so the
   clean-tree guard can operate.
3. Requires a manually supplied motion-deblurring checkpoint,
   plus a **matching complete SHA-256**; there is no network or
   automatic checkpoint download.
4. Uses `torch.load(..., weights_only=True)` and strictly checks
   the `params` tensor mapping and model architecture. It does
   **not** run upstream `demo.py`, whose unqualified
   `torch.load(weights)` is unsuitable for an untrusted file.
5. Decodes original RGB8 video frames, no synthetic added noise,
   and runs a framewise motion-deblurring model. Pads dimensions
   to multiples of 8 and crops outputs back to source size.
6. Uses a documented user-selectable **0–100% original-image mix**;
   outputs lossless FFV1 MKV, same nominal CFR/native geometry,
   **no audio** in short research excerpts.
7. Limits a run to 48 frames and at most 512 MiB decoded RGB
   frames. Rejects VFR, missing/invalid source, overwrite, hash
   failures and silent CUDA-to-CPU fallback.
8. Writes `.research.json` with original and checkpoint hashes,
   model revision, start time, mix and final output digest.

Revision checks and hashes demonstrate **reproducibility/integrity**;
they do not establish safety of upstream code, trust of a
checkpoint producer or commercial licensing.

## Example: RTX 4090 research trial

Prepare a reviewed modern CUDA/PyTorch environment with
`torch`, `einops`, `numpy`, `opencv-python` and FFmpeg.
Current versions must be independently compatibility-tested.

```powershell
git clone https://github.com/swz30/Restormer external/Restormer
git -C external/Restormer checkout 68dc6ac472db26f16361150cb7a96a1bc87da93f

# Download the official MOTION Deblurring weights following upstream's
# README and keep them OUTSIDE the Git checkout:
Get-FileHash external-weights/motion_deblurring.pth -Algorithm SHA256
$SHA256 = "<independently-verified-64-character-hash>"

# No Torch/Restormer import and no model execution:
python scripts/run_restormer_deblur_research.py `
  --source input/Neighbors_original.webm `
  --output intermediate/detail_enhanced/Restormer_24frames.mkv `
  --checkout external/Restormer `
  --weights external-weights/motion_deblurring.pth `
  --weights-sha256 $SHA256 --frames 24 --start-seconds 300 `
  --mix 50 --research-only --dry-run

# On the opt-in research GPU system, remove --dry-run only after
# reviewing upstream code, weights and all dependency licenses.
```

The output starts at time 0 but represents input frames around
05:00. **Always compare against an identically extracted source
clip** (same source, start time, number of frames, RGB conversion,
FPS and geometry), not the whole film at timestamp 0.

## Before offering a production menu option

- Actual RTX 4090 24GB inference, driver/CUDA compatibility,
  run time and peak VRAM must be tested. This PR includes
  **no model execution in CI** and no verified quality result.
- Compare Restormer, NAFNet, Richardson–Lucy, Unsharp/CAS
  and the original at native 960×720 and matched frame indexes.
- Assess false details, facial/hair/text reconstruction, scratch
  effects, halos, grain destruction and temporal flicker.
- Evaluate on controlled blurred clips with known reference for
  PSNR/SSIM, plus genuine degraded film with separate perceptual QA.
- Review weights, data, dependency and original-source provenance
  against ARP's fail-closed commercial guard before any deployment.

The Commons *Neighbors* WebM is a digital transfer, not an
authenticated original camera negative. German/EU film,
transfer and soundtrack rights need independent consideration.

Primary sources:
- Official upstream: https://github.com/swz30/Restormer
- CVPR article: https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html
