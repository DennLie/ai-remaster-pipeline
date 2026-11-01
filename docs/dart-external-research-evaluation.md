# DART 1.0.0 — isolated mask AND restoration evaluation

**Status: experimental research evaluation, no permission for production or
unrestricted commercial use.** This is a follow-up to the independent mask
detector and the mask-scoring benchmark (PRs #3–#4). It does not
import, vendor, package or download DART as part of the ARP GUI.
It explicitly keeps the commercial model guard unchanged.

## Verified upstream interface (2026-11-01)

The [DART upstream README](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer)
documents two outputs from the **same inference call**:

- `frames/frame_000000.png` — fully restored 8-bit RGB frame (OpenCV BGR when read);
- `masks/frame_000000.png` — **soft** 8-bit defect mask (0–255);
- `restored.mp4` if `--save-video` is requested.

On an input video, frame names start at `frame_000000`. With a source
image folder, file stems are retained. The DART network uses a
bidirectional 7-frame window and a stride of 3 by default. This is
**not** a mask-only scratch segmenter; restoration can change
image details even where the output mask is dark.

A reproducible source revision is provided by DART's
[v1.0.0 release](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer/releases/tag/v1.0.0):

- Source Git commit: `fc6f3e722c74a3b0e3a43ef40bb1d5303fa5e332`.
- `pretrained_models/dart.safetensors` SHA-256:
  `d78a388c76ffb2d71cd0fee55f44e3dde695112a40c540a7b67cc0590cfa73d3`.
- The DART download script verifies its own checkpoint SHA-256; the
  ARP research runner verifies it again before subprocess execution.

## What is still legally unresolved?

DART states its own **code and weights are MIT**, but the project's
[third-party register](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer/blob/main/THIRD_PARTY_LICENSES.md)
explicitly lists adapted RTN code in `video_backbone.py`,
`temporal_modules.py`, and `mask_net.py` originating from
[Bringing-Old-Films-Back-to-Life](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life),
**which has no source licence declaration**. MIT on the rest of DART
cannot establish an onward grant for those adapted upstream files.
Bundled RAFT-Sintel weights/code and SwinIR components also require
their own attribution. Written upstream permission or a valid
independently reviewed replacement/rights chain is required **before**
any commercial deployment claim.

This research-only harness may compare file outputs obtained separately,
but it makes no legal certification or licensed product integration.

## 1. Prepare a short, same-shot native-resolution source

Download the original 960×720 uploaded Commons digital file, **not**
the generated 480p transcode. Do not confuse it with an untouched
original 35-mm scanner master (actual scan provenance is unknown).

```bash
curl -fL \
  "https://upload.wikimedia.org/wikipedia/commons/2/27/Neighbors_%281920%29.webm" \
  -o Neighbors_original_960x720.webm

ffmpeg -ss 00:05:00 -i Neighbors_original_960x720.webm -t 4 \
  -map 0:v:0 -an -c:v ffv1 -pix_fmt yuv420p Neighbors_native_720p.mkv
```

Inspect that the excerpt is predominantly **one continuous shot**.
DART's recurrent state must not be carried over an actual film cut.
Take a different cut-free 2–4-second excerpt if needed; this script
does not claim to identify scene edits automatically.

The Commons file is marked US public domain **with warnings about
copyright outside the US, including Germany**. Rights to the specific
digital transfer and musical additions may be separate. Do not assume
commercial distribution in Germany is cleared.

## 2. Install DART independently — research only

Do **not** install its modules into ARP's ComfyUI environment.

```bash
git clone https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer.git ../DART
git -C ../DART checkout fc6f3e722c74a3b0e3a43ef40bb1d5303fa5e332

cd ../DART
uv sync
uv run python scripts/download_weights.py
uv run python scripts/download_weights.py --verify
cd ../ai-remaster-pipeline
```

Use the actual local ARP path for the last `cd`. DART's pinned
Python/CUDA environment is independent. For example, on Windows or
a Linux GPU worker, the virtualenv Python path differs.

The runner will **refuse a different DART revision or checkpoint hash**;
it will not download or replace anything.

## 3. Generate DART masks AND restored frames

From ARP's repository root, with an installed DART checkout and a
working GPU (if available):

```bash
python scripts/run_dart_research.py \
  --dart-repo ../DART \
  --dart-python ../DART/.venv/bin/python \
  --source Neighbors_native_720p.mkv \
  --output-dir evaluation/dart_trial \
  --max-frames 72 --chunk-size 7 --stride 3 \
  --device auto --research-only
```

This executes the **official pinned** `scripts/inference.py` in a
subprocess. The external output directory must be empty so old
files cannot accidentally pass QA. No DART source/weights are copied
into this repository. The program then writes:
`evaluation/dart_trial/arp_qa/dart_restoration_diagnostics.json` and
`dart_mask_restoration_contact_sheet.jpg` with source, fully restored,
mask overlay, conservative composite and difference heatmap panels.

For GPU out-of-memory failures, try `--chunk-size 5 --stride 2`.
CPU inference is possible but can be **very slow at native 960×720**.
There is no verified native-resolution GPU benchmark yet.

## 4. Compare DART mask versus the existing OpenCV proposal

The ARP Clean Up option `--detect-defects --no-dearchive` produces
a **lossless** temporal mask cache in `.cache/cleanup_chunks`.
For reliable comparison, run at `--defect-mask-height source`, with
the **same input clip** as DART, and export the cached mask video as:

```bash
mkdir -p evaluation/classical_masks
ffmpeg -i temporal_defects_v1_960x720.mkv \
  -start_number 0 evaluation/classical_masks/frame_%06d.png

python scripts/assess_dart_exports.py \
  --source Neighbors_native_720p.mkv \
  --dart-output evaluation/dart_trial \
  --output-dir evaluation/dart_trial_mask_comparison \
  --max-frames 72 \
  --classical-masks evaluation/classical_masks
```

Mask agreement (Jaccard) is calculated **only at identical pixel
coordinates**, with the DART mask gated at 160/255 and the OpenCV
binary mask at 128. It is a *difference-of-predictions*, not a
precision/recall score without annotated ground truth. No
resizing, frame dropping or automatic name matching is tolerated.

## 5. Assess the actual repair, not just the masks

The assessment records:

- DART mask coverage and how many changed pixels are **inside vs.
  outside** the soft-mask gate. Difference means any BGR channel
  differs by **at least 8 intensity units**.
- Full frame difference heatmaps and a limited 3-frame contact sheet,
  prioritizing scenes with substantial changes outside selected masks.
- A research-only **mask-gated composite** experiment that retains the
  source pixels **exactly** outside the selected mask (threshold
  160/255, no blur or feathering):

```bash
python scripts/assess_dart_exports.py \
  --source Neighbors_native_720p.mkv \
  --dart-output evaluation/dart_trial \
  --output-dir evaluation/dart_trial_preservation \
  --max-frames 72 \
  --write-masked-composite --research-only
```

The result in `masked_composite_frames/` is a separate **proposed**
preservation-constrained restoration. It is **not automatically added
to ARP**, is not proven aesthetically correct, and carries the same
underlying DART provenance restrictions.

A small outside-mask change percentage is not enough to prove
historical accuracy. Check faces, hair, clothing, typography, grain,
ghosting at motion boundaries, persistent scratches and cut handling.
Then compare annotated masks using
[the existing ground-truth benchmark](dart-vs-temporal-mask-benchmark.md).

## Release gates

- Keep all DART integration **research-only**; no production model
  selection or commercial-guard allowlisting before RTN provenance is
  fully resolved and notices are reviewed.
- Check exact file/frame geometry, full output coverage and consistent
  frame names; stop with errors on mismatches rather than interpolating.
- Review one-shot source clips at **native transfer resolution before any
  AI upscaling**.
- Do not treat DART's published perceptual scores as a verified quality
  result for our *Neighbors* excerpt.
- Preserve the original film file and evaluate repairs on a separate
  copy only.
