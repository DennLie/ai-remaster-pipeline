# Localized Deflicker for entire films — streaming research mode

**Status: explicitly opt-in experiment only. Not production-validated, not merged.**

The original PR #28 Localized research mode read **all** frames into RAM and deliberately rejected clips longer than 240 frames / 75 million decoded pixels. This guard is a research implementation constraint, **not** a film-restoration design target. The original mode remains unchanged for scientific ablation.

This branch adds **Localized · Langfilm-Streaming (experimentell)**, with:
- **Two-pass** work: (1) scan exact frame sequence for histograms, shot boundaries and approximate monotonic fade protection; (2) sequentially decode and encode, keeping at most **five decoded frames and their local illumination fields** in memory.
- No 240-frame hard stop, no fixed maximum total decoded pixels, no preloading a complete scene or movie, no per-frame correction array proportional to the full movie. Shot/fade statistics require O(number of frames) scalars in the first pass, not O(number of frames × resolution).
- The same *spatial* low-frequency local gain, ±8% cap, spatially blurred oscillation guard and radius-2 temporal neighbors as the validated research pilot, with only the **surrounding shot/fade guard and streaming implementation** changed. A new versioned output key prevents accidentally reusing the short-clip mode's results.
- A fresh output, full 1:1 encoded frame count, output/resume signature and existing ARP finalize step, including the existing stage's audio handling.
- A client frame-change chart for full films: up to 240 **sparsely sampled verified exact frame positions**, with turquoise brightening and orange darkening. This chart is NOT an exhaustive per-frame/full-movie statistic and may miss localized problem frames. The exact Frame Inspector still targets the selected frame.

## Safety boundaries

**Do not interpret ability to process a whole movie as evidence of restoration quality.** Automatic cut detection uses ARP's normalized histogram Bhattacharyya threshold, and fade protection uses approximate per-frame low-resolution medians. Neither is a perfect shot detector. Subtle transitions, camera flashes, pans, slow day/night exposure, moving objects, and color/texture may still be altered. There is no motion compensation or genuine film-negative ground truth. The manual experimental acknowledgment is required; ARP Conservative remains the default.

Other Clean Up operations (DeArchive, AI DeScratch, DeVignette and Defect Detection) must remain off for the Localized streaming *test* so that attribution to the research algorithm is possible. This restriction does not apply to other ARP stages in subsequent independent jobs.

The processing memory is bounded by roughly five full decoded BGR frames plus five blurred float32 luminance maps **and temporary NumPy arrays**; it is independent of film *duration*, but depends on spatial resolution. At 4K/8K, memory and CPU requirements are still substantial. Output disk space can be large, especially in lossless FFV1. A decoder reset, missing metadata, inconsistent frame count, and geometry mismatch cause hard errors rather than fabricated comparison results.

## Test plan

- 52-frame oscillating source: streaming pixel output must exactly match the short-clip research algorithm when shot boundaries do not intervene.
- 60-frame cross-shot/fade synthetic sample: no blending across boundaries and no changes to protected fades.
- **305-frame actual FFV1 render**: exceeds 240 frames, tests the CLI, all frames and geometry, cache reuse, guard refusal, and separate artifact identities.
- Frame-chart exact seek semantics for movie-sized frame counts, browser labels, negative controls from the original research PR.
- Multi-hour/real-feature-film end-to-end processing and extensive visual review are **not yet tested in CI**; those are the next acceptance gates.

## Operator trial

From a clean working tree after fetching this branch, select the full film as the current ARP source, disable DeArchive/DeScratch/DeVignette/Detect Defects, enable Light Deflicker, choose **Localized · Langfilm-Streaming (experimentell)**, explicitly acknowledge its research-only status, then Run Clean Up.

A 72-frame clip may be used as a regression test first, then a few-minute multi-shot excerpt with cuts and fades, followed by the desired feature film after visual inspection. Preserve original media and previous variants, compare visually at shot transitions. For a truly complete film, make sure that the source selected on the Overview page is the *whole* film rather than an auto-selected short source section.

No CLI length cap is imposed by this mode. Model weights are not used. No change to main/test or earlier PR branches.
