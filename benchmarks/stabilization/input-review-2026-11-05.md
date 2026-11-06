# Longer-input screening — 2026-11-05

## Method and limits

Screened the nine prepared 20-second development inputs using 20 evenly spaced
decoded frames per input, automatic cut proposals from the production detector
(`shot_threshold=0.075`, `min_shot_seconds=1`), adjacent downsampled grayscale
change peaks, and selected full frame-pair neighbourhoods. This was assistant
inspection of sampled images, **not exhaustive human playback review** and not
restoration-quality judgement. Input files and their hashes were not changed.

The images and machine-readable proposals are under:

`.cache/stabilization-corpus/development-windows-20s-pr18-b790a154-01/input-review-01/`

`timeline.jpg` uses frame-first labels. `cut-after-N.jpg` shows frames around
each automatic proposal; `input-review.json` records all proposed spans and
descriptive grayscale peaks. Those peaks are neither quality nor cut truth.

## Automatic proposals, not ground truth

Frame ordinals are zero-based and relative to each selected input. A cut after
N is the boundary between frames N and N+1. An empty proposal list does not
prove there are no cuts.

| Input | Automatically proposed cut-after frames |
|---|---|
| Corner in Wheat / field | none |
| Corner in Wheat / rural workers | 192, 299 |
| Corner in Wheat / late scene | none |
| Train Robbery / interior | 420, 444 |
| Train Robbery / moving train | 79, 177, 221, 292, 325, 401 |
| Lumiere / approach | 45, 76, 186, 228, 413 |
| Lumiere / crowd | none |
| Melies / stage | none |
| Melies / trick | 109, 139, 319, 353, 437, 466 |

## Directly observed examples

- **Corner rural workers, 192 → 193:** exterior changes to a title card.
  **299 → 300:** title changes to a shop/queue scene. Both proposed boundaries
  are supported by the inspected neighbouring frames.
- **Train interior, 420 → 421:** the interior remains the same, but a large
  tear/displacement appears in the source and clears by frame 423. This is
  source damage, not evidence of an editorial change of scene. Resetting
  tracking may be defensive, but must not be counted as accurate cut detection.
- **Train interior, 444 → 445:** interior changes to an exterior/ladder view.
  The proposed boundary is supported by the inspected neighbouring frames.
- **Train moving, 401 → 402:** same train/track composition around a bright
  disturbed source frame, not an observed editorial change of scene.
  A later change of view begins at **407 → 408**, with damaged transition
  frames 408–409 and a stable new view from 410. That boundary is absent from
  the detector's proposals. See `transition-405-413.jpg` for the actual frames.
- **Lumiere approach, 186 → 187:** the locomotive continues within the same
  platform composition; the inspected six-frame neighbourhood does not support
  an editorial cut at the proposed boundary.
- **Melies trick, 319 → 320:** figures and snow/particles continue in the same
  stage composition; the inspected neighbourhood does not support an editorial
  cut at this proposed boundary.

These are selected observations, not an exhaustive count of false or missed
cuts. No complete `cut_after_frames` annotation was inserted into the input
manifest, and no cut-detector accuracy score is claimed.

## Consequences for the smoothing comparison

Keep detector parameters and safety limits fixed for the production-path
6/12/24-frame comparison at **3% zoom**. Its automatic shot segmentation is
part of the measured system, not retrospectively corrected after seeing scores.
Check that all three runs actually use identical source hashes and cut spans.

Interpret the resulting motion and fallback numbers as **descriptive technical
diagnostics**. Frequent resets, input damage or missed changes can confound an
apparent smoothing benefit. A safe original-preserving fallback is not a
successful restoration. Do not tune thresholds or select a winner from these
numbers alone.

The sampled inputs cover moving subjects/vehicles, static stage/background
geometry, particles, titles and scene transitions. They do **not establish a
verified sustained intentional camera-pan test**; a pan-preservation claim
remains an explicit coverage gap. Held-out films must not be used to tune that
gap away. Human playback review and completed cut annotations remain open.
