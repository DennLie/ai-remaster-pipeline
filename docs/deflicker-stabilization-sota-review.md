# SOTA audit: ARP Light Deflicker & Stabilization — 2026-11-01

**Scope**: code-level inspection of `main/scripts/stabilize_video.py` and
experimental PR #6 `scripts/film_deflicker.py`; review of published
film-deflicker and video-stabilization research through Nov 2026.

**Verdict: neither is state of the art** in contemporary benchmark terms.
Both are useful, deterministic, bounded classical baselines. SOTA on
consumer video is **not automatically** the best choice for an archival
film restoration master.

## Light Deflicker — code audit

Reference: [ARP PR #6](https://github.com/DennLie/ai-remaster-pipeline/pull/6).

Implementation: central-crop luma median, clipped pixel rejection,
16-bin standardized luminance histogram for shot detection,
radius-5 temporal neighborhood, median neighbor brightness,
monotonic-ramp heuristic, per-frame RGB 8-bit LUT with gain
`1 +/- min(strength*(neighbor/current-1), cap)` (strength 0.65,
cap 0.08 by default). Two-pass decode-and-reencode via OpenCV/FFmpeg.
Disabled by default. The native 960x720 *Neighbors* test reported
45/120 corrected frames and mean absolute adjacent median luma
change 2.61345 to 1.84874:
[real CI experiment](https://github.com/DennLie/ai-remaster-pipeline/actions/runs/41343808901).
**That figure measures only frame-luma variability, not fidelity,
DART performance, or ground-truth flicker correction.**

Limitations:
- **One global gain/frame**. Cannot separately repair unevenly exposed
  patches or spatially varying aging/illumination instability.
- **8-bit, gamma-encoded BGR** as opposed to transfer-function-aware
  high-bit-depth luminance processing; clips highlights, rounds
  shadows, and changes chroma together with luma. A quality gate
  should detect/clamp affected pixels, and ideally preserve source
  precision/chroma.
- Simple cut/fade heuristics and luma median are affected by moving
  foregrounds, occlusion, camera pans, letterboxes and strong
  scene illumination changes.
- Entire first-pass video held in brightness/histogram arrays; memory
  for those arrays is not the limiting issue, but full decode is
  required twice and the corrections are not motion compensated.
- No labeled artifact benchmark or independent blinded visual review.

Relevant research:
- F. Pitié, B. Kent, B. Collis and A. Kokaram,
  *Localised Deflicker of Moving Images* (2006), professional archival
  practice. **Localized gain/bias plus motion-aware statistical
  correction**, not just a single global gain.
  https://francois.pitie.net/research/deflicker/
- Julie Delon and Agnès Desolneux, *Stabilization of Flicker-Like
  Effects in Image Sequences through Local Contrast Correction*
  (SIAM J. Imaging Sci., 2010). A patch-based local stabilization
  robust to motion/contrast changes in archival sequences.
  https://doi.org/10.1137/090766371
- Qiu et al., *BlazeBVD* (ECCV 2024). Histogram-assisted illumination
  and local exposure priors, learned spatial/temporal refinement;
  reports qualitative and quantitative advantages on its selected
  datasets. NOT directly validated on *Neighbors* or audited as a
  commercially redistributable model.
  https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2526_ECCV_2024_paper.php
- Cai et al., *WaveletMask* (Applied Sciences, June 2026), separates
  high-frequency structural film defects and low-frequency flicker
  cues as a recurrent mask *detector*. Not interchangeable with
  a frame-correcting deflicker, but pertinent to DART pre-processing.
  https://doi.org/10.3390/app16136415
- FFmpeg built-in `deflicker=size=...:mode=median` is a stable,
  permissive-source **classical baseline**, not an SOTA algorithm.
  https://ffmpeg.org/ffmpeg-filters.html#deflicker

**Recommendation:** Keep PR #6 opt-in; test identical native,
scene-aligned clips using `none`, `ARP`, and `FFmpeg median`,
and optionally an independently reviewed localized/robust
method. Mark protected shadows/highlights and film grain.
A potential v2 would first correct only true transfer-related
luma flicker, retain 10/16-bit processing and chroma, estimate
localized low-frequency gain/bias with temporal motion confidence,
and avoid interpolation of intact fine image structure.
Do not introduce a large deep model by default.

## Stabilization — code audit

Reference: [current ARP main stabilizer](https://github.com/DennLie/ai-remaster-pipeline/blob/main/scripts/stabilize_video.py).

Implementation: independent shot detection (with long-window
pan-sensitive rules disabled), optional human-shot-manifest,
FFmpeg `vidstabdetect` and `vidstabtransform`, bounded global
translation/rotation, smoothing 12 frames, optional fallback on
bad global motion-fit, FFV1/lossless intermediate option,
preserves frame cadence/audio, standard fixed **3% zoom**,
`optzoom=0`, `crop=black`, bicubic resampling.
Existing unit tests and a synthetic FFmpeg smoke test confirm
functionality but **not real-world residual weave quality**.

Limitations:
- Global planar translation+rotation model cannot track
  local deformation, multi-plane parallax, rolling shutter or
  non-rigid shrinkage.
- A default 3% zoom **discards picture information**, contrary to
  archival-master full-frame protection goals.
- Bicubic resampling may change film grain and soften details.
- Global image-content motion estimation can mistake intentional
  camera pan for transport/weave; source provenance must guide it.
- No explicit background-occlusion masks, scan/perforation anchor
  or quality-based confidence score/gate specific to historical
  gate-weave. Source `Neighbors` WebM lacks provenance of a full
  film-edge overscan, preventing validation of true camera-vs-transfer
  instability from perforations alone.

Contemporary research:
- Xu et al., *DUT* (IEEE TIP 2022): mesh trajectories, multiple
  homographies, learned dynamic smoothing; can handle richer
  content motion than one rigid transform.
  https://pubmed.ncbi.nlm.nih.gov/35724289/
- Peng et al., *RStab* (CVPR 2024): 3D multi-frame fusion / rendering,
  aiming at full-frame stabilization and improved geometrical
  consistency. https://openaccess.thecvf.com/content/CVPR2024/html/Peng_3D_Multi-frame_Fusion_for_Video_Stabilization_CVPR_2024_paper.html
- Liu et al., *Minimum Latency Deep Online Video Stabilization*
  (IEEE TPAMI 2025): neural camera-path smoothing, multiple
  motion densities including dense flow.
  https://pubmed.ncbi.nlm.nih.gov/39509300/
- You et al., *GaVS* (SIGGRAPH 2025): temporally consistent 3D
  reconstruction/rendering to reduce cropping and parallax.
  https://sinoyou.github.io/gavs/
- Zhu et al., *VS3R* (arXiv, March 2026): 3D reconstruction and
  **generative diffusion** for full-frame stabilization. In the
  publicly inspected repository, DVDM/LoRA model weights were
  still advertised as future releases; not verified deployment-ready
  or suitable for archival fidelity.
  https://arxiv.org/abs/2603.05851
  https://github.com/zmhcasmy/VS3R

These approaches are not compared with ARP on an equivalent
archival dataset, and paper SOTA claims concern their own
benchmarks and metrics, not objectively authentic old-film
reproduction. Diffusion-based filling can invent material that
never existed in the film.

**Recommendation:** Keep original source untouched and stabilization
disabled unless proven weave/transfer jitter. For a more
archive-specific **non-generative** mode, try background-stable
feature/phase/ECC alignment with outlier rejection, only
translation/low rotation where justified, *no default zoom*, and
a recorded valid-footprint mask. OpenCV documents
`findTransformECC` using translation, Euclidean, affine or
homography modes:
https://docs.opencv.org/5.0/main_modules/video_track.html

Use overscan/perforation alignment if the film source actually
contains edges; do not infer it from an ordinary encoded 960x720
transfer. FIAF recommends preserving original camera instability
while distinguishing deterioration, duplication and scan-induced
weave. https://www.fiafnet.org/pages/E-Resources/Digital-Statement-part-III.html
Film overscan: https://www.fiafnet.org/is/2/E-Resources/Digital-Statement-part-II.html

## Tests required before replacing either stage

| Experiment | Comparison | Required outcome |
|---|---|---|
| Deflicker 1 | source vs ARP vs FFmpeg `deflicker` | Less *unwanted* temporal luma change without false correction at cuts, pans, fades, highlights |
| Deflicker 2 | source vs conservative localized gain/bias | No shadow clipping or grain damage; evaluate spatially local flicker against known clean synthetic base |
| Stabilizer 1 | no stabilizer vs libvidstab 0% zoom vs 3% zoom | Measure residual background jitter (pixels), frame crop, interpolation blur and protected edges |
| Stabilizer 2 | libvidstab vs masked ECC/feature-based translation | Reduce *true film weave* without canceling deliberate camera motion or scene content |
| DART ablation | DART raw vs optional improved preprocessing | Better manually annotated mask precision/recall and repair fidelity; reject purely cosmetic improvements |

Preserve frame numbers, color space, frame rate, crop footprint,
mask coordinates, original transfer file hash and model version.
Require one review of no-zoom black-border/frame footprint before
using a zoom/crop on archival material. Test on multiple separate
`Neighbors` scenes, **not** only one four-second sample. Film
provenance and licensing remain independent gates.

## Decision

1. **Now**: no backend replacement. Existing classic methods are
   reliable *baselines*, not research SOTA; leave default opt-in.
2. **Before DART**: run a raw-original DART baseline first. Compare
   processed vs raw at exactly matching source timestamps and
   coordinates; account for stabilization warping when assessing
   mask overlap.
3. **Next improvement**: fixed 3% stabilizer zoom should not be the
   default for a preservation master. Add a no-crop/no-zoom
   QA mode and quantify valid-pixel area before selecting a
   new algorithm.
4. **Then**: design a non-destructive, local flicker v2 *if*
   comparisons show artifacts the current global approach misses.
5. **Avoid**: generative 'SOTA' filling, unsupported commercial
   licensing claims, automatic remaster sharpen/grain replacement
   and interpreting lower frame-to-frame luma variance as
   evidence of restoration accuracy.
