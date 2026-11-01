# DART versus ARP temporal masks — reproducible evaluation

This **benchmark is an evaluation harness, not a DART integration**. It
uses image files that the external DART application already exports.
ARP never imports or bundles DART or RTN code, models, or checkpoints.

## Primary-source findings (2026-11-01)

DART's [official README](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer#quick-start)
documents a restoration command that also writes soft defect masks:

```bash
uv run python scripts/inference.py --input my_clip.mp4 --output outputs/my_clip --save-masks --save-video
```

[DART inference.py](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer/blob/main/scripts/inference.py)
writes `outputs/my_clip/masks/<source-frame-name>.png`. Mask gray values
are **0–255 soft confidences**, not binary 0/255 defect labels. It
averages bidirectional MaskNet predictions for interior frames,
using one direction at temporal clip boundaries. The released DART
checkpoint is a **restoration + supervised mask-prediction network**,
not merely an isolated scratch segmentation model.

DART's own MIT notice and weight statement do **not** settle the
third-party provenance issue: its
[THIRD_PARTY_LICENSES.md](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer/blob/main/THIRD_PARTY_LICENSES.md)
says `video_backbone.py`, `temporal_modules.py` and the
`_build_rtn_mask_net` part of `mask_net.py` adapt code from
[RTN](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life),
whose original repository lacks a licence file. **No unrestricted
commercial grant has been confirmed for that adapted code**.
Also retain notices for DART's RAFT/RAFT-Sintel BSD-3 and SwinIR
Apache-2.0 elements. Before using any DART integration commercially,
obtain a written RTN rights grant or independent legal verification.

## Dataset specification

Use approximately **100–300 original-resolution film frames** spread
across multiple shots from *Neighbors* (1920). Include clean frames,
fast movement, moving faces, presentation bars, titles, scene cuts,
persistent scratches, single-frame dust and damaged backgrounds.

Lay out the dataset so frame names and image dimensions correspond
**exactly** in all directories:

```text
evaluation/
  frames/
    shot01/000001.png
    shot01/000002.png
  ground_truth/
    shot01/000001.png
    shot01/000002.png
  predictions/
    temporal/
      shot01/000001.png
      shot01/000002.png
    dart/
      shot01/000001.png
      shot01/000002.png
  samples.csv
```

`samples.csv`:

```csv
clip,frame,split,damage_type
shot01,000001.png,dev,dust
shot01,000002.png,test,clean
```

**Ground-truth PNG convention** (one-channel 8-bit, original frame
dimensions): `0` = known intact, `255` = actual damaged pixel,
`127` = ignore (uncertain/motion occlusion/excluded area).
No antialiasing or other grayscale values in ground truth. The
prediction PNGs may contain soft gray values, such as DART outputs.
The benchmark thresholds predictions at 128/255 by default. Never
treat DART's brightness as a calibrated probability without checking.

The `clip` and `frame` fields are simple names (no `../`), and the
manifest must have no duplicate clip/frame pairs. Every method must
provide all listed predictions: missing files cause an error, not a
silently better-looking score.

## Generate comparable predictions

**ARP temporal detector**: enable *Clean Up → Detect film damage* on
PR #3 (turn off ProPainter and Dearchive for a mask-only run). Use the
**lossless cached** `temporal_defects_v1_*.mkv`, not the H.264
`*_defect_mask.mp4` browser proxy.

Extract the lossless mask video to images, for example:

```bash
mkdir -p evaluation/predictions/temporal/shot01
ffmpeg -i temporal_defects_v1_948x720.mkv -vsync 0 evaluation/predictions/temporal/shot01/%06d.png
```

Be careful: the ARP analysis canvas may be **smaller than the original**
(e.g. 720 pixels high). For a fair comparison, **run at
`defect_mask_height=source`**, or explicitly pre-process the masks with
documented nearest-neighbour resizing to the matching GT coordinates.
The benchmark refuses shape mismatches; it never silently resizes.
Run shot windows separately and ensure frame naming/timing is aligned.

**DART**: in a **separate DART environment**, run its official
`scripts/inference.py --input <shot frames> --output <shot> --save-masks`
and copy its `masks/*.png` into
`evaluation/predictions/dart/shot01/`. Verify source-frame names.
The benchmark does not invoke DART and is unrelated to its weight
licence. Do not change frame order, crop or color-transfer the input
between methods.

## Score

From ARP's repository root:

```bash
python scripts/benchmark_defect_masks.py \
  --manifest evaluation/samples.csv \
  --ground-truth evaluation/ground_truth \
  --prediction temporal=evaluation/predictions/temporal \
  --prediction dart=evaluation/predictions/dart \
  --threshold 128 \
  --report evaluation/defect_benchmark.json
```

Report includes **micro-aggregated pixel precision, recall, F1, IoU,
false-positive rate on intact pixels, mask coverage**, raw counts and
frame-level results grouped by clip, split and damage type.
Performance and real-footage results are **not yet measured**.

Use a separate `dev` split to choose sensitivity, resolution and
soft-mask thresholds; report unchanged decisions on `test` shots.
For DART confidence calibration, run several fixed thresholds on
`dev` (e.g. 64, 128, 192), but never select a threshold based on
final test performance. Report **persistent-scratch recall** and
**false positives on intact faces and title cards** separately.
Include subjective review and runtime/VRAM measurements before
deciding on any restoration backend.

## Commercial deployment gate

Evaluation with externally generated masks does **not** remove
third-party licence restrictions. This PR adds no DART code or
models and changes neither the production Clean Up repair
engine nor the strict commercial model guard. A separate approved
backend is needed before restored outputs can be sold with the
commercial assurance requested by this fork.
