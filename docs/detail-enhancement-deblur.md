# Detail Enhancement / Deblur

The new optional `enhancement` stage follows Light Deflicker in both the tab
order and processing order. It is **disabled by default**, with method Off.
Legacy Clean Up, Stabilization and the later `detail` stage remain in place;
their settings are not migrated or automatically disabled. If both enhancement
stages are enabled, the film is processed twice, at their respective positions.

## Use

Enable **Detail Enhancement / Deblur** on Overview. Include the methods to
compare, choose the current classical preset and picture mix, and explicitly
render missing variants. Selecting comparison tiles does not adopt a pipeline
method. **Use for pipeline** adopts only the chosen method, retaining preset
and mix. Off passes the processing input through without enhancement.

Methods are Unsharp, CAS, Multiscale, Richardson–Lucy, NAFNet and Restormer.
Gentle/Balanced/Strong refer to the existing classical implementations, not
equivalent apparent strengths and not three neural checkpoints. Mix is an
integer 0–100% for every method. Changing the classical preset leaves matching
neural results reusable; changing mix changes all affected result identities.

Original is the actual input after enabled Remove Noise and Light Deflicker,
not necessarily the raw transfer. The shared retained comparison supports
frame-locked compressed playback, exact RGB8 stills, differences, Wipe, pan,
zoom and native 1:1. Pixel changes and sharpness measures are not restoration
quality scores. FlashVSR+ remains a separate scaling research task.

## Neural preparation and integrity

The worker uses native-resolution FP32 CUDA inference, one RGB frame at a time,
without an all-film frame array. It preserves source audio and frame geometry,
requires nominal CFR, and never silently falls back to CPU or lower resolution.
NAFNet uses the official NAFNetLocal width64 GoPro configuration; Restormer
uses the official GoPro motion-deblurring architecture. Required `torch`,
`einops`, NumPy and FFmpeg must be available in ARP's processing environment.

The local installation uses `.cache/detail_deblur/code/NAFNet` at commit
`2b4af71ebe098a92a75910c233a3965a3e93ede4` and
`.cache/detail_deblur/code/Restormer` at commit
`68dc6ac472db26f16361150cb7a96a1bc87da93f`. Both must be clean checkouts,
including their architecture files and licence notices. The model-only
NAFNet loader avoids executing BasicSR's training/dataset import registry;
it does not install old BasicSR training packages into ARP.

Place these official README-linked files in `.cache/detail_deblur/models/`:

| Model/file | Official download | Locally measured SHA-256 |
|---|---|---|
| `NAFNet-GoPro-width64.pth` | [NAFNet GoPro width64](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view) | `329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f` |
| `Restormer-GoPro-motion_deblurring.pth` | [Restormer GoPro motion](https://drive.google.com/file/d/1pwcOhDS5Erzk8yfAbu7pXTud606SB4-L/view) | `194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31` |

These are measured hashes of files obtained from the official links, **not
independently publisher-signed digests**. Workers verify the hash before
`torch.load(weights_only=True)` and strict tensor-state loading. The code does
not automatically download or unpickle arbitrary checkpoints. Code licences
do not settle weight, dataset or footage rights: the commercial model guard
continues to block NAFNet/Restormer pending an audit. Classical methods remain
available with that guard enabled.

## Publication and numerical safeguards

Results go to `intermediate/enhanced/`, keyed by input path, selected method,
effective parameters, checkpoint identity and intermediate profile. Comparison
and downstream use require the completed signature and current upstream
lineage. A failed render leaves the last matching success untouched; a failed
replacement/publication cannot keep the old ready marker. Explicit reuse can
refresh a touched source's provenance without encoding again. Polling never
hashes the whole source or checkpoint.

Neural inputs are reflect-padded to the architecture's multiple and cropped
back to the original size. Extreme synthetic inputs exposed catastrophic NAFNet
numeric excursions even though the checkpoint loaded strictly; finite tensors
alone were not sufficient validation. The runner rejects nonfinite predictions
or absolute values above 16 **before** normal display-range clipping and does
not publish that output. This is a catastrophic-failure guard, not a perceptual
quality threshold, a PSF estimate or a fix for every neural artifact.

## Validation limits

Real pretrained CUDA video runs have been exercised on the same short FFV1
clip generated from the official NAFNet demo photograph, retaining native
960×720 geometry, cadence, audio, source bytes and signed reuse. Browser checks
exercise all six rendered methods, exact/compressed delivery, selective preset
invalidation, explicit adoption and retained tab state. The generated demo clip
is **not** a moving historical-film scene or a temporal-quality benchmark.

No long-film performance, 4K VRAM guarantee, authentic grain fidelity, motion
consistency or universal method ranking is established. NAFNet and Restormer
are framewise image models and may change grain, faces, text or temporal detail.
Richardson–Lucy still assumes, rather than measures, a Gaussian PSF. Inspect
short representative sections before committing expensive full-film runs.

Tests: `tests/test_enhancement_stage.py` and
`tests/test_enhancement_workspace_browser.test.cjs`. Set `ARP_TEST_NEURAL=1`
to explicitly include the local pretrained CUDA video test; it needs the
prepared checkpoints and official demo image at
`.cache/detail_deblur/smoke/official-nafnet-demo-blurry.jpg`.
