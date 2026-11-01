# Optional FastDVDnet research inference (not commercial-cleared)

**Status: isolated research CLI only.** Not installed in ARP's default
environment, not wired into the normal GUI or `denoise_video.py`,
not part of the ARP commercial model allowlist. No automatic
downloads, checkpoint inference, or `main` changes.

## What this solves

The published FastDVDnet [test script](https://github.com/m-tassano/fastdvdnet/blob/c8fdf6182a0340e89dd18f5df25b47337cbede6f/test_fastdvdnet.py)
does **not** denoise a film directly: it builds a synthetic benchmark
by adding Gaussian noise to source frames before running inference.
Calling this script directly on *Neighbors* would distort the film
first and produce an invalid before/after restoration comparison.

`scripts/run_fastdvdnet_research.py` reads the original decoded
RGB8 frames *without added noise*, uses a five-frame reflected
temporal window, and passes a separately selected `sigma/255`
**noise-conditioning estimate** to the original pinned model
architecture. Sigma is **not** noise added to the footage.

It verifies (prior to loading any external Python or weights):

- External FastDVDnet checkout is clean and pinned to commit
  `c8fdf6182a0340e89dd18f5df25b47337cbede6f`
  ([upstream](https://github.com/m-tassano/fastdvdnet)).
- Manual checkpoint's **entire SHA-256** is supplied and matches.
- Explicit `--research-only` opt-in; no checkout/weight auto-downloads.
- Checkpoint is loaded only with `torch.load(..., weights_only=True)`.
  Malicious huge tensors or a compromised upstream environment are
  still risks; checksum verification is *integrity*, not a trust grant.
- Output is a **new .mkv**, never source overwrite.
- Max 120 frames in one invocation; no silent GPU-to-CPU fallback.
  Audio deliberately omitted from this **short research sample**.

## Preparation (Windows PowerShell, when the RTX 4090 is available)

Use a **separate virtual environment** for modern compatible PyTorch,
and an explicitly reviewed FFmpeg installation. Do not install the
upstream legacy `requirements.yml` with PyTorch 1.1 / CUDA 10 on
a modern RTX 4090.

After deciding to trust the pinned upstream code and the weight source:

```powershell
# Explicit user action: obtain a *separate* external checkout.
git clone https://github.com/m-tassano/fastdvdnet external/fastdvdnet
git -C external/fastdvdnet checkout c8fdf6182a0340e89dd18f5df25b47337cbede6f

# Independently inspect and record the exact checkpoint SHA-256.
Get-FileHash external/fastdvdnet/model.pth -Algorithm SHA256

# Set this to the value you independently recorded. Placeholder is not a real hash.
$MODEL_SHA256 = "<64-character-reviewed-sha256>"

# Validate code revision / weight integrity and video without importing Torch.
python scripts/run_fastdvdnet_research.py `
  --source "input/Neighbors_native_720p.mkv" `
  --output "intermediate/denoised/neighbors_fastdvdnet_sigma10.mkv" `
  --checkout "external/fastdvdnet" `
  --weights "external/fastdvdnet/model.pth" `
  --weights-sha256 $MODEL_SHA256 `
  --frames 24 --start-seconds 0 --sigma 10 --research-only --dry-run

# Only after verifying everything, run actual inference:
python scripts/run_fastdvdnet_research.py `
  --source "input/Neighbors_native_720p.mkv" `
  --output "intermediate/denoised/neighbors_fastdvdnet_sigma10.mkv" `
  --checkout "external/fastdvdnet" `
  --weights "external/fastdvdnet/model.pth" `
  --weights-sha256 $MODEL_SHA256 `
  --frames 24 --start-seconds 0 --sigma 10 --device cuda --research-only
```

If your source is the full *Neighbors* WebM, use `--start-seconds 300`
to target the 05:00 excerpt (and ensure the file is the original
digital transfer, not the previously upscaled copy). The sample
output begins at 00:00 and runs for exactly the requested frames.

**Artifacts:** FFV1 research-only RGB converted to `gbrp`, no audio;
adjacent `.research.json` metadata records the hash of the checkpoint
and output, output format, requested frames and source path.
Compare to the **matching original excerpt** produced using the
same FFmpeg `--start-seconds` and frame range, not to a full-length
video at time 0. Use the existing
`scripts/compare_denoising.py` to check frame-exact diagnostics.
For an interactive preview, import the finished variant into the
ARP A/B/C comparison (PR #10).

## Scope/limitations

- This research adapter uses 8-bit RGB as a controlled baseline,
  not high-bit-depth/linear-light science-grade reconstruction.
- The model was trained primarily for synthetic Gaussian noise,
  not scratches, digitization artifacts or 1920s film grain.
- It does not preserve or guarantee original film grain and is
  especially likely to alter moving texture.
- Short segments include reflected temporal boundaries; avoid
  drawing conclusions from first/last two frames. Longer film
  tests need shot-aware inference and consistent overlaps.
- No GPU checkpoint inference has been run in CI; automated tests
  exercise integrity guards, FFmpeg source decoding, temporal indexing,
  and output muxing using **mock passthrough frames**, not model quality.
- The repository code is MIT, but that alone does not clear
  official bundled weights, upstream training data, dependencies,
  the original movie transfer or other jurisdictions for paid release.
  Remain research-only pending an explicit separate legal review.
- Never merge any FastDVDnet code or weights into the core
  commercial pipeline without model inventory and clear provenance.

For authentic vs modernized results, keep source and all denoised
variants separate. Evaluate `Original`, `hqdn3d`, `NLMeans`,
`BM3D`, and optional `FastDVDnet` by eye at 100% and with
the **same original DART mask/repair evaluations** once RTX inference
is available. No result is automatically considered better merely
because it is smoother or sharper.
