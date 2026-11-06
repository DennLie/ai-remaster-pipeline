# Offline smoothing review — 2026-11-05

## Where to review

All three production-backend runs completed: **nine 20-second inputs each at
smoothing 6 / 12 / 24 frames and requested zoom 3%**, or 27 native outputs.
The existing CLI's `--zooms 3` option avoids rendering unused zoom variants;
the default remains the four-zoom sweep. The targeted regression suite passed
**45 tests**, including default and fixed-zoom real-backend runs.

Offline viewing artifacts are assembled separately under:

`.cache/stabilization-benchmark/development-smoothing-20261105-01/comparison/`

Use `review-all.m3u` in a normal video player to view the nine cases, or open
`<case>/review_original_s6_s12_s24.mp4` individually. These are not browser
pages, ARP project inputs or pipeline selections.

The comparison publication is complete only when `comparison.json` reports
`status: complete` and the playlist exists. The preparation/verifier script is
`.cache/stabilization-benchmark/compare-development-smoothing.py`; it checks
source/output fingerprints, identical cut lists, code/runtime fingerprints,
settings, decoded timelines, output frame count and rational cadence.

## Numbers beside the videos

`comparison/measurements.md` is the readable combined overview, and
`comparison/measurements.csv` contains its unrounded values: **36 rows**
(one original plus three smoothing variants for each of nine inputs). Each
case folder also contains its own `measurements.md` and `measurements.csv`
beside the review video. Source-report and native-master paths are retained.

The tables separate measured execution counts from motion estimates and raster
geometry estimates. They include median estimated translation/rotation/vector
change, trusted adjacent pairs, untrusted **non-cut** pairs, excluded automatic
cut pairs, fallback counts/percentages, mean/worst output support and estimated
mean cropped source area. The baseline row comes from the radius-12 run.
Missing/not-applicable values are blank in CSV or a dash in Markdown, not zero.

The original QA's rejected-pair count includes excluded cuts. The consolidated
tables subtract those explicitly, with the invariant
`trusted + untrusted_noncut + excluded_cut_pairs = frames - 1` checked on every
row. This is not cut ground truth: suspected false or missed cuts still limit
the estimates. A change between estimated translation vectors is not physical
camera acceleration. Raw pixel motion is not pooled across film geometries.

The aggregation uses only completed reports and never writes human judgements.
`visual_review.csv` remains separate. There is no automatic score, ranking,
winner, or PSNR-based quality claim for these real inputs.

## Panel layout and limitations

| Top left | Top right |
|---|---|
| Original / actual processing input | Smoothing 6, requested zoom 3% |
| **Bottom left: smoothing 12, requested zoom 3%** | **Bottom right: smoothing 24, requested zoom 3%** |

Images are kept at their native dimensions within the mosaic. Labels occupy
added black header bands, not source image pixels. All four panes start at
the same frame, have matching cadence and retain the complete selected frame
count. The player may still scale the combined mosaic to fit your display.

The mosaic uses a rational frame clock and matches **frame ordinal N to N**,
not the union of rounded Matroska presentation timestamps. A real multi-scene
input exposed 1 ms concatenation differences: timestamp-based stacking emitted
695 frames instead of 480. Frame-index stacking emitted exactly 480 at 24 fps.
Every final mosaic is checked for full frame count and exact rational cadence;
failed timestamp-based assemblies are retained separately in a debug folder,
not in the review playlist. Native files are not retimed or overwritten.

Yellow **UNSTABILIZED SAFETY FALLBACK** labels identify individual scenes
re-encoded without correction or zoom. A label saying correction was applied
is not approval of its visual quality.

The MP4 files are **lossy H.264/YUV420 playback proxies**. They are useful for
motion, cuts, pumping and exposed edges, but are not evidence of preserved
fine grain or unchanged pixels. For those questions use native lossless files:

`radius6|radius12|radius24/<case>/real/zoom3.mkv`

The unchanged selected input is recorded in each run's manifest, and the
original/native-master paths also appear in `comparison.json`.

## What to check

1. Play at normal speed. Does unwanted movement decrease, without introducing
   new swimming/pumping or suppressing intentional motion? Do not infer pan
   preservation here: a sustained intentional camera pan is not yet verified
   in these inputs.
2. Inspect cuts and damaged source frames. A tracking reset or safety fallback
   must not be mistaken for improved stabilization. Distinguish existing
   source defects from new changes introduced by correction.
3. Inspect exposed edges and crop, particularly Corner in Wheat's field scene.
4. Inspect native masters for resampling/detail loss; do not mark grain/detail
   as passed solely from the compressed four-panel video.
5. Record observations in `comparison/visual_review.csv`, one row per
   case/candidate. Human judgement cells remain blank until an actual review.
   Leave uncertain questions blank and describe uncertainty in notes.

## Technical flags to prioritize, not a winner ranking

- **Train Robbery / moving train:** all 480 frames use safety fallback at every
  smoothing radius. This input does not demonstrate successful stabilization
  or distinguish the smoothing choices, even when coverage is 100%.
- **Train Robbery / interior:** 59 of 480 frames use fallback at every radius.
  The input contains both source damage and an editorial transition.
- **Lumiere / approach:** measured median apparent adjacent translation is
  about 0.91 px on input versus 1.62–1.69 px on candidates. This is a warning
  to inspect, not proof of worse restoration: foreground motion and matching
  errors confound the proxy, and the detector produces suspect resets.
- **Corner / field:** worst raster support is about 99.765% at radius 12 and
  99.566% at radius 24 despite 3% zoom; radius 6 reports 100%. Inspect edges
  rather than assuming 3% guarantees support or selecting a radius by this
  single metric.

The [input-screening record](input-review-2026-11-05.md) documents false/suspect
resets and a missed transition. The three runs retain identical automatic cut
segmentation; thresholds were not tuned after inspecting scores. Descriptive
motion numbers are not true historical gate-weave measurements or quality
scores. No real-input PSNR or historical-fidelity claim is made.

No winner has been selected, no human-quality form has been filled, no held-out
candidate has been rendered, and no preset/default/project has been changed.
Zoom minimization and held-out validation remain later gates, not completed
evidence. There is no benchmark integration into ARP's browser interface.
