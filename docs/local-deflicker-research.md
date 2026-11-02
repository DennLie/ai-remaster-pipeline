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
