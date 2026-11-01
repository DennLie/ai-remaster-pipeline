# NAFNet GoPro Deblurring — research-only ARP adapter

**Status 2026-11-02:** Prepared in a stacked **draft PR** for offline
CPU integrity tests and later user-authorized RTX 4090 inference.
The model is *not* part of the normal commercial ARP selection list.
No model download, automatic code execution or weights are introduced
into the default environment.

## Why NAFNet is a useful comparison

The official Megvii [NAFNet](https://github.com/megvii-research/NAFNet)
is a single-image deblurring network from ECCV 2022.
The upstream README reports **33.71 dB PSNR and 0.9668 SSIM** for
`NAFNet-GoPro-width64` on **GoPro**, not on silent films.
These scores are not general guarantees on archival transfer
noise, grain, scratches, or fluctuating frame damage.

Original test instructions:
https://github.com/megvii-research/NAFNet/blob/main/readme.md

Upstream is pinned to exact commit
`2b4af71ebe098a92a75910c233a3965a3e93ede4`
and its **source code** is MIT licensed. However, code licensing
does not conclusively grant rights to every included checkpoint,
training dataset, output element or dependency for commercial reuse.

Note that upstream `basicsr/demo.py` accepts a **single image**
and calls BasicSR's network loader, which uses unqualified
`torch.load(...)` in that upstream revision. For model integrity,
our adapter does **not invoke that demonstration path**. Instead,
it instantiates the verified pinned `NAFNet` architecture,
loads a separately verified checkpoint with
`torch.load(weights_only=True)`, and processes original RGB frames
directly with a controlled 0–100% source-mix parameter.

## Fail-closed requirements

1. You must manually clone the *exact* upstream commit into a
   **clean external checkout**. The research runner will refuse
   a dirty checkout, untracked new files or a different Git HEAD.
2. You must supply the **complete SHA-256** for a **separately
   downloaded** GoPro width32 or width64 checkpoint. The runner
   hashes the file again before importing any external code.
3. You explicitly opt in with `--research-only`. It defaults to
   CUDA, does not silently fall back to CPU and refuses overwriting
   any existing output.
4. Clips are capped at 72 frames and 512 MB of decoded RGB data.
   There is no full-film mode or automated installing/downloading.
5. FFV1 RGB research output is in a new .mkv, same native
   resolution/frame cadence, with no audio. Its sidecar research
   JSON records exact source/checkpoint/output hashes and settings.
6. The default pipeline never imports this script; no
   FastDVDnet/NAFNet model invocation or weights are built into ARP.

SHA-256 verifies **file identity, not whether the publisher or the
checkpoint is legally or technically trustworthy**. Upstream code
and packaged dependencies should still be reviewed and sandboxed.

## Minimal RTX 4090 trial — on your own machine

Use a separate modern PyTorch environment that supports the
installed GPU and its CUDA runtime. The upstream README cites
older 2022 CUDA/PyTorch versions; **do not automatically
install those historical pins** on an RTX 4090.
Its `basicsr` and `NAFNet` imports and checkpoints have not yet
been validated in a live RTX 4090 session.

After reviewing the upstream and obtaining the GoPro checkpoint
from the official README:

```powershell
git clone https://github.com/megvii-research/NAFNet external/NAFNet
git -C external/NAFNet checkout 2b4af71ebe098a92a75910c233a3965a3e93ede4

# Keep downloaded weights outside the clean NAFNet checkout.
Get-FileHash external-weights/NAFNet-GoPro-width32.pth -Algorithm SHA256
$SHA256 = "<actual-reviewed-64-character-sha256>"

python scripts/run_nafnet_deblur_research.py `
  --source input/Neighbors_1920.webm `
  --output intermediate/detail_enhanced/NAFNet_GoPro32_24frames.mkv `
  --checkout external/NAFNet `
  --weights external-weights/NAFNet-GoPro-width32.pth `
  --weights-sha256 $SHA256 `
  --research-only --width 32 --frames 24 `
  --start-seconds 300 --mix 50 --dry-run

# Only after validating hashes/source provenance, omit --dry-run:
python scripts/run_nafnet_deblur_research.py `
  --source input/Neighbors_1920.webm `
  --output intermediate/detail_enhanced/NAFNet_GoPro32_24frames.mkv `
  --checkout external/NAFNet `
  --weights external-weights/NAFNet-GoPro-width32.pth `
  --weights-sha256 $SHA256 `
  --research-only --width 32 --frames 24 `
  --start-seconds 300 --mix 50 --device cuda
```

The `--width 32` or `--width 64` flag **must match the
chosen checkpoint architecture**. It cannot auto-select or
silently load the wrong model. With a 50% mix, the deblur
output is blended with the **exact decoded original RGB
frame**; no synthetic input noise is added.

The output sample is time-shifted to start at 00:00, so
`compare_denoising.py` must receive an **identical 24-frame
extracted original clip** rather than the full original
WebM. Match the same `-ss 300`, `-frames:v 24`,
RGB conversion, FPS and encoded reference:

```powershell
ffmpeg -hide_banner -loglevel error -y -ss 300 `
  -i input/Neighbors_1920.webm -frames:v 24 `
  -map 0:v:0 -an -c:v ffv1 `
  intermediate/detail_enhanced/original_24frames.mkv
```

Compare not just mean pixel differences but moving edges,
real hair/hat details, scratches and intertitle letters.
**NAFNet is a single-image model:** its outputs may
flicker or alter detailed grain between neighboring frames.
Test several representative shots, not only 0.8 seconds.

## What remains before any production release

- Verify this adapter with a **real pinned GoPro checkpoint**
  and a clean checkout on RTX 4090, including modern PyTorch
  compatibility, peak VRAM, latency and correct color ordering.
- Record numerical comparison of original vs Unsharp/CAS/
  Multiscale (PR #12) vs NAFNet at the **same 1x resolution**.
- Evaluate temporal stability (flow-warped residual excluding
  cuts/occlusions), halo/ringing, faces and written intertitles,
  not just an increased sharpness score.
- Test on synthetically blurred source footage with known
  intact ground truth to measure PSNR/SSIM/LPIPS for
  actual deblurring, alongside film-reference subjective QA.
- Reconcile full model/checkpoint/training/dependency chain,
  commercial rights and original film source usage.
- Later integrate an **explicit experimental AI Deblur**
  GUI option, while keeping default Step 3 disabled.
  If performance or model rights fail, prefer classical
  baselines and leave the option research-only.

## Sources

- Official upstream code/weights: https://github.com/megvii-research/NAFNet
- NAFNet article: https://arxiv.org/abs/2204.04676
- Related paper on directional Richardson–Lucy calibration,
  with pinned NAFNet/Restormer research comparisons:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13517317/
- Archive ethical considerations:
  https://www.fiafnet.org/pages/E-Resources/Digital-Statement-part-III.html

**Commercial and archival note:** the 960×720, 30fps
Wikimedia Commons 2016 upload of *Neighbors* is a digital
transfer, not an authenticated original film scan.
US public-domain metadata does not establish all
German/EU rights or soundtrack/transfer rights.
