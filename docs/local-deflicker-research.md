# Localized deflicker — CPU research prototype

**Status:** experiment-only, **not** an ARP Clean Up option. Based on the selectable ARP / FFmpeg comparison in draft PR #26. No trained weights or DART dependency.

## Hypothesis

A broad, spatially varying low-frequency exposure error (e.g. one corner of a transfer pumping) can persist even if a single global gain is insufficient. A conservative *local spatial gain* may attenuate that specific error while preserving textured source pixels.

This is **not** a reimplementation of Pitié et al.'s complete motion-aware local gain/bias method or Delon and Desolneux's local-contrast model. It is not motion compensated, does not estimate spatial bias, is not high-bit-depth, and must not be offered as a production restoration mode yet.

## Prototype

`scripts/local_deflicker_research.py` reads a **short, static-shot** input into memory, blurs its grayscale luminance to a low-frequency spatial field, estimates a temporal neighbour-reference field (excluding the current frame), and applies a smooth bounded per-pixel gain to all three BGR channels. A human can specify shot starts via `--cut-frame N` (zero based); correction never references frames across those boundaries. An obvious monotonic global fade is protected. Neither grain denoising nor detail synthesis is performed intentionally. Frames are written in a separate **FFV1 MKV** without audio and a JSON diagnostic report is produced. No existing file is overwritten.

Example from a short **single-shot** FFV1 reference (maximum 240 frames by default):

```powershell
python scripts/local_deflicker_research.py --source evaluation/Neighbors_72.mkv --output evaluation/Neighbors_72_local_experimental.mkv --radius 2 --sigma 18 --strength 0.6 --cap 0.08
```

Do not run this on an unreviewed multi-shot clip without human cut boundaries. The output is separate from the regular pipeline and is **not** selected by the GUI.

## Experiments needed before any integration

1. **Synthetic spatially varying flicker:** compare local prototype vs ARP Conservative, FFmpeg Median, FFmpeg Arithmetic Mean and original, on exactly aligned video with known exposure maps. Measure restoration of the clean digital-transfer reference, especially **inside and outside** the damaged region.
2. **Negative controls:** static clean film; intentional full-frame or local fades; daylight changes; pans, moving foregrounds, camera flashes, high-contrast scene cuts, grain and shadow/highlight clipping. Record false changes, not merely reduction in temporal variance.
3. **Authentic film sources:** compare at least three independent film transfers and multiple shots, with manual defect annotation and blinded visual review.
4. **Geometry and fidelity gates:** verify all frames, dimensions, FPS, bit depth, RGB/chroma and source signatures; quantify clipping, outside-mask modifications, grain damage, temporal artifacts and per-film uncertainty.
5. **Release decision:** only expose a local method in Clean Up if it improves controlled results and does not degrade protected regions. A true motion-aware gain/bias approach would be a separate, more ambitious implementation with explicit license/provenance review.

## Scientific limitations

The algorithm can confuse motion and occlusion with local exposure drift. Its per-channel multiplication can alter chroma in clipped pixels. Preserving shape and texture does not prove recovering historical negative detail. A synthetic test passing is evidence only of the particular controlled defect model. **No automated production recommendation** is justified.

## Controlled real-footage digital-reference pilot (2026-11-02)

The separate `scripts/evaluate_local_deflicker_pilot.py` now accepts a locally available short **real historical transfer**, injects a **known** alternating left-side exposure disturbance, and compares the research backend against the untouched digital source as an alignment-preserving proxy reference. It also runs the same backend **without synthetic damage** and measures how much it wrongly changes the original. It reports:

- Mean absolute pixel error *inside* the injected exposure zone, before and after correction
- Changes *outside* the target area, which should remain small
- False modification of the completely untouched real film, including a protected outside region
- The number of frames changed, maximum gain and warnings that no film-negative ground truth exists

This is **not** proof of correcting genuine, unannotated film defects, even when the injected-damage task improves. The negative-control results on moving footage can expose false corrections.

For the already available 72-frame *Neighbors* local sample, **only after fetching the research branch**:

```powershell
Set-Location I:\ai-remaster-pipeline
git fetch origin
git switch --track origin/research/localized-deflicker-v1
.\.venv\Scripts\python.exe scripts/evaluate_local_deflicker_pilot.py `
  --source evaluation/Neighbors_72.mkv `
  --report evaluation/Neighbors_72_local_pilot.json `
  --max-frames 120
```

The report is stored only locally and is intentionally not committed to GitHub. To save another result, choose another filename; the pilot refuses overwrites. The application GUI is **not** switched to this research branch for normal Clean Up rendering.

**CI scope:** synthetic static and moving-image negative controls, plus an FFmpeg-generated moving test clip in a deterministic video pilot. This is not a captured historical film. A multi-film genuine-defect blind review is still required before enabling a selectable Localized engine.
