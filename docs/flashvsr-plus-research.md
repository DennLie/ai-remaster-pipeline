# FlashVSR+ on the RTX 4090 — isolated ARP research adapter

**Status:** draft research integration only. No GPU inference, model downloads, trained-weight execution, model quality validation or commercial clearance performed by this PR. `main` is untouched.

This is a **video super-resolution / generative reconstruction** candidate, not a native-1x sharpening filter or a source-faithful certified historical film reconstruction. It belongs in the *modern remaster* branch of ARP Step 3 / Step 7 comparisons.

## Verified upstream interfaces (2026-11-02)

- Official FlashVSR+ repository: https://github.com/OpenImagingLab/FlashVSR_Plus
- Code pinned to Git SHA `b548a2cc2c929d3d4676f2973abe27876ec4dee7`, source Apache-2.0.
- Technical report: https://arxiv.org/abs/2610.17552 (October 13, 2026 **preprint**).
- Authors claim up to 2× DiT acceleration, up to 45% lower peak memory, spatially untiled 4K on a 24 GB RTX 4090 *for their specific setup*; **not independently reproduced in ARP**.
- Native **Linux**, NVIDIA CUDA, Python **3.10**. A Windows RTX 4090 may require WSL2/Linux and appropriate GPU passthrough and driver support. The upstream **does not support CPU inference**.
- Upstream command: `python -m inference.run_stream`, with
  `--input`, `--output`, `--mode turbo|base|v1`,
  `--height`, `--width`, `--max-frames`, `--ckpt-dir`,
  `--hardware-profile generic`, `--attn-kernel sdpa`.
- Both output dimensions must be **multiples of 128**. The input loader center-crops when the input/output aspect ratios differ, so ARP blocks any aspect ratio mismatch.

**Important for *Neighbors*:** the actual 960×720 transfer has 4:3 geometry; neither dimension is divisible by 128. We do **not** stretch the source or crop the frame. Instead the pilot explicitly requests output **1024×768** (4:3 and 128-divisible), a **1.0667×** increase in both dimensions, not "native 1x" or "2x". Higher resolution 2048×1536 is also a valid 4:3, 128-divisible output (2.1333×), subject to RTX 4090 VRAM. Upstream may interpolate and invent texture; comparison needs a separately resampled conventional baseline at exactly the same output resolution.

## Model assets and integrity

Upstream requires the following for Turbo or Base, including *three* external checkpoint assets and a pinned repository context tensor:

| Manifest key | Location under clean pinned checkout |
|---|---|
| `dit.safetensors` | `ckpts/{turbo,base}/dit.safetensors` |
| `lq_proj_in.safetensors` | `ckpts/{turbo,base}/lq_proj_in.safetensors` |
| `TCDecoder.ckpt` | `ckpts/TCDecoder.ckpt` |
| `posi_prompt.pth` | `prompt_tensor/posi_prompt.pth`, tracked by upstream Git |

The `ckpts/` directory is ignored by the upstream repo. You must manually obtain, inspect and place weights there before using the adapter. There are **no auto-downloads**. Record every full SHA-256 in an **external JSON manifest**, not inside the checkout, with exactly the four keys above:

```json
{
  "dit.safetensors": "<64-hex-checksum>",
  "lq_proj_in.safetensors": "<64-hex-checksum>",
  "TCDecoder.ckpt": "<64-hex-checksum>",
  "posi_prompt.pth": "<64-hex-checksum>"
}
```

Compute these digests **locally** with `sha256sum` or `Get-FileHash -Algorithm SHA256`. The values above are placeholders, not approved checksums. Checkpoint hashes verify the exact downloaded bytes, **not** their trustworthiness, full third-party licenses, or absence of malicious tensors. The pinned official streaming implementation may use its own PyTorch checkpoint-loading internals, so execution remains explicitly user-authorized *research only*. Review Wan architecture, TCDecoder checkpoint, pretrained base models, training-data and dependency licenses separately before any commercial enablement. The MIT/Apache status of an individual source repo does not grant rights to all dependencies and weights.

## Pilot from a Linux / WSL2 research environment

Run from the **ARP repository root**, with a separately reviewed Linux/CUDA Python 3.10 environment. The upstream README describes installing `requirements.txt` and `pip install -e . --no-deps` in that *separate* environment. Do not install or import FlashVSR+ as an ARP core dependency.

```bash
# Explicit manual action, once the external upstream is reviewed:
git clone https://github.com/OpenImagingLab/FlashVSR_Plus external/FlashVSR_Plus
git -C external/FlashVSR_Plus checkout b548a2cc2c929d3d4676f2973abe27876ec4dee7

# Store reviewed weights under external/FlashVSR_Plus/ckpts/turbo/
# and shared TCDecoder at external/FlashVSR_Plus/ckpts/TCDecoder.ckpt.
# Store independently verified SHA-256 JSON OUTSIDE that repository.

python scripts/run_flashvsr_plus_research.py \
  --research-only --dry-run \
  --source "input/Neighbors_original.webm" \
  --output "intermediate/flashvsr_plus/Neighbors_turbo_1024x768.mp4" \
  --checkout "external/FlashVSR_Plus" \
  --manifest "external-weights/flashvsr_plus_turbo_sha256.json" \
  --mode turbo --width 1024 --height 768 \
  --frames 24 --start-seconds 300 \
  --temporal-compression 4 --seed 1 \
  --python "external/FlashVSR_Plus/.venv/bin/python"
```

The `--dry-run` verifies source geometry, temporal bounds, original video SHA-256, exact upstream commit and **all four real model asset hashes**; it does not import PyTorch, start CUDA inference or write a result. After a reviewed CUDA environment is available, omit `--dry-run` for the actual explicitly authorized GPU test.

On the real run, ARP extracts exactly the 24 requested original frames to a temporary lossless FFV1 source clip, executes the upstream CLI **with SDPA** (no Triton/BSA custom kernel required initially), and verifies output resolution, FPS and frame count. Results go to a **new** H.264 MP4 and a JSON provenance sidecar. The upstream MP4 is **lossy**: do not treat the output or its difference map as lossless pixel-level reconstruction.

## How to compare fairly

1. Keep unmodified 960×720 source, as well as an **independent Lanczos 1024×768 reference** from precisely the same original 24 frames.
2. Compare FlashVSR+ Turbo 1024×768 with that reference, and then Base 1024×768 if both checkpoint sets and rights are separately verified.
3. Also inspect source *at its native 960×720*. Do not compare sharpness numbers from different dimensions directly.
4. Verify original 4:3 composition, Keaton's face/hair/hats, clothing seams, intertitles, bright outlines and moving objects. Document new texture, hallucinated details, flicker, color shifts and temporal inconsistencies.
5. Keep **DART scratch repairs** independently controllable, with defect-mask tests before/after any optional generative enhancement. A smoother/sharper image may be less faithful.
6. Measure actual RTX 4090 wall-clock time, peak VRAM, frame cadence, source fingerprint, exact pretrained weights hashes and output compression. The research paper's 4090 performance is **not** an ARP benchmark.

## CI scope

CPU-only CI tests verify the pinned Git/dirty-tree gate, all four mandatory hash entries, video frame metadata, aspect-ratio protection, actual FFmpeg FFV1 inputs and dry-run output arguments. It does **not** run any pretrained model, CUDA kernel, Triton extension or video quality comparison. `main` and the commercial allowlist are unchanged.

The 2016 Wikimedia digital transfer of *Neighbors* is not confirmed to be an original camera-negative scan. US-public-domain metadata does not by itself establish the distribution and soundtrack/transfer rights in Germany or the rest of the EU.
