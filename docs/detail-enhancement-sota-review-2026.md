# Detail Enhancement (Step 3) — SOTA research and ARP implementation choices
**Reviewed 2026-11-02. Research and design, not implemented or production-clearance.**
**Target:** *Neighbors* (1920), Wikimedia 960×720 digital transfer at 30fps
(not a verified pristine original film scan); RTX 4090 24GB; optional
heritage-faithful and openly modernized versions.

## Key distinction: There is no single SOTA for "enhance details"

1. **Sharpening / local contrast** makes existing edges more visible
   without recovering missing information (potential halos and scratch
   enhancement).
2. **Deblurring / deconvolution** tries to invert an identified blur
   (point-spread function, defocus, motion blur). Without an appropriate
   degradation model this may amplify grain, halos or ringing.
3. **Multi-frame reconstruction / blind video restoration** infers new
   information from adjacent frames, using temporal alignment. Some
   models perform both denoising and upscaling by design.
4. **Generative video detail enhancement** invents plausible detail
   under an image prior: a modern remastering choice, not documentary
   evidence of how the frame originally appeared.

An improvement in sharpness, pixel contrast, or human preference does
not prove historical detail was restored. This is particularly relevant
to eyelashes, facial features, text, costumes, moving people,
high-contrast white/black edges, and 1920s film texture.

## Directly applicable controlled baselines (native 960×720)

| Backend | Category | Typical controls | Trade-offs |
|---|---|---|---|
| **OFF** | Baseline | None | Exact source frame and grain |
| **FFmpeg `unsharp`** | Classic sharpening | luma kernel/radius, luma amount, zero chroma sharpening | Local halos, scratch exaggeration; baseline reproducible |
| **FFmpeg `cas`** | Contrast-adaptive sharpening | strength, processing planes | Sharpening can still amplify texture noise and ringing |
| **OpenCV custom multiscale unsharp / local contrast** | Conservative enhancement | strengths for microcontrast, detail mask, luma-only | Adjustable and mask-aware but must not be advertised as true detail recovery |
| **Richardson–Lucy** (`skimage.restoration`) | PSF deconvolution | PSF, iterations, regularization/early stopping | Useful with known, fairly stable blur kernel; noise/ringing risk |
| **NAFNet / Restormer** | Learned *image* deblurring | pretrained matched task, strength by source blend | Optical deblurring potential; single-frame flicker risk, training mismatch and separate checkpoint rights |

Primary references:
- https://ffmpeg.org/ffmpeg-filters.html#unsharp
- https://ffmpeg.org/ffmpeg-filters.html#cas
- https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.richardson_lucy
- https://github.com/megvii-research/NAFNet
- https://github.com/swz30/Restormer

A pre-DART unsharp pass can enhance *scratches and dirt* and change
where the detector places its masks. Therefore default pre-DART
sharpening must be **OFF**; make it an explicit, separately measured
ablation if the user chooses to follow the exact YouTube step ordering.

## Research and practical video reconstruction candidates (2026)

| Candidate | Evidence / novelty | ARP fit, important caveat |
|---|---|---|
| **RealBasicVSR (CVPR 2022)** | Learned motion aggregation + degradation pre-cleaning; showed long-term temporal propagation can amplify unseen real artifacts | Good historically significant *non-diffusion video* benchmark, but primarily VSR and needs correct checkpoints |
| **VideoGigaGAN (CVPR 2025)** | Rich texture + video consistency; perceptual detail-focused | Research/visual benchmark; do not imply official runnable commercial weights were verified |
| **SeedVR2 (ICLR 2026)** | One-step diffusion/adversarial video restoration; 3B/7B options | **Already in ARP Upscaling**, reuse before building another backend; often generative/upscaling and changes texture |
| **FlashVSR (CVPR 2026)** | One-step streaming diffusion-based VSR | **Already in ARP Upscaling**; upstream explicitly targets 4×, so not a native 1× sharpening implementation |
| **FlashVSR+ (Oct 13, 2026 preprint)** | Hybrid attention + reward-aligned one-step VSR; authors report up to 2× DiT speed and 45% lower peak memory; **claim untiled 4K VSR on single RTX4090 24GB** | **New and particularly relevant optional ARP candidate**, *not independently benchmarked*, use external GPU test and evaluate outputs at matched delivery size |
| **STCDiT (CVPR 2026)** | Anchor-frame guidance & motion-aware VAE for structural/temporal fidelity | Peer-reviewed research; practical downloadable checkpoints and full rights not validated |
| **DGAF-VSR (CVPR 2026)** | Optical-flow-guided dense feature alignment / temporal conditioning; reports gains in fidelity and temporal stability | Peer-reviewed research; practical downloadable checkpoints and full rights not validated |
| **LTX-2.5 CQ Enhancer** | Prompts/guidance to reconstruct texture at working resolution | **Already in ARP** under Upscaling `ltx25cq` with `cq_finish=none`; code/revenue terms, generative face and texture risks |

Primary sources:
- https://openaccess.thecvf.com/content/CVPR2022/html/Chan_Investigating_Tradeoffs_in_Real-World_Video_Super-Resolution_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/CVPR2025/html/Xu_VideoGigaGAN_Towards_Detail-rich_Video_Super-Resolution_CVPR_2025_paper.html
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/444d69470b24ded080183c907b711bbf-Abstract-Conference.html
- https://openaccess.thecvf.com/content/CVPR2026/html/Zhuang_FlashVSR_Towards_Real-time_Diffusion-Based_Streaming_Video_Super_Resolution_CVPR_2026_paper.html
- https://arxiv.org/abs/2610.17552
- https://github.com/OpenImagingLab/FlashVSR_Plus
- https://huggingface.co/ruchangyao/FlashVSR_plus
- https://openaccess.thecvf.com/content/CVPR2026/html/Chen_STCDiT_Spatio-Temporally_Consistent_Diffusion_Transformer_for_High-Quality_Video_Super-Resolution_CVPR_2026_paper.html
- https://openaccess.thecvf.com/content/CVPR2026/html/Xu_Rethinking_Diffusion_Model-Based_Video_Super-Resolution_Leveraging_Dense_Guidance_from_Aligned_CVPR_2026_paper.html

**Evidence levels:** FlashVSR+ is an **October 2026 arXiv preprint**,
not established peer-reviewed SOTA. STCDiT and DGAF are
CVPR 2026 papers, but their reported margins are
task/benchmark-dependent; not demonstrated on *Neighbors*.
The RTX4090 24GB claim applies to the paper's specific
**VSR configuration**, not arbitrary 1× detail enhancement at 4K.
The original FlashVSR's official README notes that its optimized
BSA backend's RTX40-series compatibility was unknown as of review;
FlashVSR+ instead documents a Triton attention backend for 4090
and PyTorch SDPA fallback.

## What ARP already contains / must avoid duplicating

At `main`, `ai_remaster_gui/models.py` lists
`upscale.method=flashvsr|seedvr2|ltx25|ltx25cq`.
`scripts/upscale_video.py` also exposes
`cq_finish=flashvsr|seedvr2|ltx25|lanczos|none`.

This means an LTX **CQ enhance-only** route is conceptually present
inside the upscaling code without forcing the final upscaler,
but it is **not yet** a standalone controlled Step-3 module,
nor proof of an independently successful 1× enhancement.
Avoid implementing a second CQ or SeedVR2 backend just to label a
new stage. Better add an adapter/route that can call tested
existing implementations while preserving separate cache metadata.

## Recommended placement and user controls

Two complementary modes, never silently conflated:

### A. Detail Enhancement @1× (native frame size)

`OFF` (default) / `unsharp` / `CAS` /
`multiscale-local-contrast` /
`Richardson-Lucy (with specified PSF)` /
`NAFNet deblur (model cleared only)`.
Generic settings: enable, chosen algorithm and backend-specific
parameters, gamma-vs-linear processing choices, output precision,
luma-only mode, edge/texture mask, optional protection for DART
defect pixels, per-shot enable/disable, source mix (0–100%).
Presets: gentle/balanced/strong, mapped *per algorithm*; do not treat
all filters as equivalent at e.g. "50%".

### B. Generative detail / texture reconstruction

Use `SeedVR2`, `FlashVSR`, optional `FlashVSR+` and
LTX CQ. Model checkpoint/version, reconstruction strength, seed,
temporal window, chunk overlap, native-vs-upscaled target geometry,
color-fidelity switch and source/detail blend are independent.
Label outputs **AI Enhanced / Modern Remaster**, not
documentary proof of recovered features.
Need a consistent x2/x4 pipeline comparison:
`basic source -> x2 Lanczos (baseline)` versus
`same source -> SeedVR2 x2`, etc.
If comparing 1× results, convert each restored x2/4 output
back to native 960×720 with documented kernel *only as a
special test*, not as a claim of native model capability.
Avoid making every candidate unnecessarily upscale then downscale.

**Default restoration order recommended for actual defects:**
`original -> [optional deflicker / noise reduction / transport
stabilization] -> DART masks+repair -> optional 1× detail
enhancement -> colorization -> optional SR enhancement -> finishing`.
An alternative **pre-DART** detail pass can be exposed
*only* as an explicit ablation with identical DART model,
mask thresholds and annotated defect GT. A modernized
output and a historically faithful master must remain distinct.

## 24GB RTX4090 feasibility

- CPU `unsharp`, `cas`, light multiscale filters:
  fast, negligible VRAM and good nightly regression tests.
- NAFNet and other image deblurrers: likely viable per frame;
  **not** necessarily temporally consistent; actual speed untested.
- SeedVR2 3B FP8: third-party ComfyUI RTX4090 benchmarks demonstrate
  feasible execution with block swapping and tiled VAE, not
  real-time quality; exact performance depends on resolution,
  precision, implementation, driver and frames-per-chunk.
- FlashVSR+ Turbo/Base: research authors explicitly target
  RTX4090 24GB and report 4K inference; actual Windows/CUDA
  dependencies and video quality remain to be verified in ARP.
  Its separate Transformer + decoder + Wan VAE checkpoints
  and attention extension are not a one-file drop-in.

## Licensing and source provenance gates

| Software/model | Public license evidence | Conservative ARP action |
|---|---|---|
| FFmpeg unsharp/CAS | FFmpeg license depends on actual binary compilation (LGPL/GPL options) | Record executable/configuration; no blanket commercial-safe claim |
| OpenCV 4.5+ | Apache-2.0 | Code baseline; audit packaged build |
| scikit-image | BSD-style open source; component audit | Controlled PSF-deblur prototype |
| NAFNet / Restormer | MIT upstream code | Check exact checkpoint license/training provenance and dependencies separately |
| SeedVR2 | Official SeedVR repo Apache-2.0 | Already supported by ARP; verify each model and dependency |
| FlashVSR+ | Official repo Apache-2.0; Hugging Face model card Apache-2.0 | Candidate, but verify original Wan2.1 VAE, TCDecoder, attention backend and checkpoints |
| LTX-2.5 | LTX-2.x community license dated 2026-08-11; entities ≥ US$10m annual revenue need paid commercial agreement, other restrictions apply | Keep under existing ARP commercial guard; not unrestricted |
| VideoGigaGAN / VEnhancer | Publication/repository ≠ verified free production license and weights | Research/reference only until separately verified |
| DART/RTN adaptation | Upstream MIT does not settle missing license on original RTN-derived code | No blanket commercial deployment claim |

Source licenses:
- https://github.com/OpenImagingLab/FlashVSR_Plus/blob/main/LICENSE
- https://huggingface.co/ruchangyao/FlashVSR_plus
- https://github.com/ByteDance-Seed/SeedVR/blob/main/LICENSE
- https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
- https://www.fiafnet.org/pages/E-Resources/Digital-Statement-part-III.html

The Commons source's 2016 uploaded 960×720 digital transfer is
not proven a fresh original scan. Archive/content/music rights in
Germany/other jurisdictions need separate investigation.

## Scientifically defensible *Neighbors* benchmark

Use the **same original native digital transfer**, same 30 fps and
identical, shot-aware frame indexes. Keep lossless/10-bit capable
intermediates where supported; avoid comparing a source WebM
against multiple differently compressed MP4 proxies as though
pixel differences were genuine restoration differences.

A focused factorial experiment:
1. A: untouched original (960×720)
2. B: `unsharp` gentle, native resolution, luminance only
3. C: FFmpeg CAS gentle, native resolution
4. D: measured-PSF Richardson–Lucy only if image blur is genuinely
   detected and kernel can be supported; else do not include
5. E: NAFNet motion deblur research-only if weights checked
6. F: LTX CQ-enhance-only (existing implementation, model cleared)
7. G: SeedVR2 3B (existing SR stage), x2 compared to Lanczos x2
8. H: FlashVSR (existing) and optional FlashVSR+ (new), x2/x4
   evaluated at **exactly the same target geometry and frame numbers**

At least 4–6 short shot types: still close-up, walking/motion,
fine black clothing/hat/hair, title card and thin lettering,
bright high-contrast edges, dense damage/grain/fades. Put a
test excerpt around 05:00 only if it contains relevant motion
and note shot boundaries. Use the user's frame-exact A/B/C GUI
(PR #10) for viewing, and offline *frame-exact* correspondence
for measurement (browser seeking is approximate).

### Measurements and acceptance gates

- **Actual blur:** edge-spread function, calibrated artificial
  blur experiments with a known reference, reconstruction loss,
  halo width/overshoot and new clipping.
- **Texture:** spatial frequency energy + viewer crop inspection.
  Increased Laplacian variance alone ≠ more authentic detail,
  can simply be amplified grain/scratches.
- **Temporal stability:** motion-compensated warped-frame error
  excluding cuts, disocclusions and true motion; optical flow
  itself introduces measurement uncertainty. Include a contact
  sheet and looping side-by-side videos.
- **Defects:** manual defect mask precision/recall and pixel-level
  reconstruction sanity on real DART export. Did sharpening
  magnify scratches or cause hallucinated repairs?
- **Faithfulness:** inspect faces, clothing seams, geometry,
  written intertitles and scene-specific features at 100%
  native pixels, plus 200% nearest-neighbor only as a pixel
  diagnostic. Record image/video versions and licensing.
- **Efficiency:** seconds/frame on actual 4090, peak allocated
  VRAM, CPU RAM, chunk/transition seams and output size.
- **With ground truth** (synthetically degraded sources
  with documented kernels), report PSNR, SSIM, LPIPS and
  temporal metrics, but never interpret those synthetic
  results as original-film GT.

### Decision

Implement independent **native-resolution** baseline methods
(OFF/FFmpeg unsharp/CAS/mask-aware local contrast) first with
GUI controls and A/B/C comparisons, but don't run them
automatically before DART. Reuse tested `ltx25cq`,
SeedVR2 and FlashVSR upscaling implementations as separately
labeled creative approaches. Pilot `FlashVSR+` in isolated
research mode with a pinned model/weights/third-party rights
inventory before integrating it into commercial paths.
