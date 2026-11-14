# PR24 positive-control replay — 2026-11-06

## Outcome

All five published Temporal Ridge pixel-F1 results reproduce to their published
four-decimal rounding. The current ARP core matches the unchanged PR24 detector
pixel for pixel across all 120 damaged frames. Current and historical Legacy
also have identical aggregate confusion counts in this replay.

The actual ARP wrapper reduces scores. This is a measurable integration effect,
not a different sensitivity value. No production detector or guard was changed.

| Film / original split | Historical Legacy F1 | PR24 / ARP core F1 | Actual wrapper F1 |
|---|---:|---:|---:|
| Lumière / development | 0.000000 | 0.807288 | 0.779336 |
| Train Robbery / development | 0.000000 | 0.307100 | 0.298591 |
| Méliès / development | 0.064281 | 0.497617 | 0.487424 |
| Caligari / historical holdout | 0.000000 | 0.750000 | 0.725275 |
| Gertie / historical holdout | 0.000000 | 0.611035 | 0.448816 |

The historical holdouts are reproduction controls, not new independent test
data for changes developed after seeing these results. No tuning was performed.

## Provenance and reproduction

```powershell
$env:PATH="$PWD\.cache\tools\ffmpeg;$env:PATH"
.\.venv\Scripts\python.exe scripts/replay_ridge_pr24.py --corpus .cache/stabilization-corpus/pr18-b790a154 --output .cache/ridge-positive-control-new
.\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_ridge*.py' -v
```

Output must not exist. The script reads only existing clips and local Git objects;
it makes no network requests. All five input SHA-256 values match both the
recovered PR18 corpus manifests and the pinned clip identities in the replay.
The manifests also retain the original transfer identities and extraction ranges.
Full holdout transfers were not acquired; their existing 24-frame clips suffice.

The generator, historical Legacy function and its column helper, Ridge reference,
and scoring functions are loaded from PR24 commit
`15af65261a733b6472553d4d32fefcfa43253210`. Only the generator module and selected
detector/scoring functions execute; the historical application is not imported.
The generator uses the historical film/scene-specific derived seed, not a new
generic random-line implementation. Settings: `gentle`, seed **2026**, **24 frames**,
sensitivity **0.65**, **0 additional dilation**. Intrinsic Ridge morphology stays.

All methods receive identical FFmpeg-decoded BGR8 damaged arrays. Only the frame
reader is frozen when invoking the actual ARP wrapper. The decoded stream ends
after 24 frames, just as the original pilot clips do. This is not the 72-frame
Neighbors run or a long-window streaming validation.

Completed outputs are in `.cache/ridge-pr24-positive-control-20261106-03/`:
- `report.json`: source/code hashes, settings, runtime versions, aggregate metrics
  by original split, per-film/per-frame TP/FP/FN/TN, Precision/Recall/F1/IoU and
  healthy-pixel false-positive fractions; wrapper disagreements and reason codes.
- `<film>/synthetic/`: reference/degraded FFV1 clips, exact GT PNGs and manifest.
- `<film>/<method>/`: all exact native-size predicted masks for five methods.
- `<film>/native-input-gt-legacy-pr24-wrapper-*.png`: native, unscaled five-panel
  comparisons at frames 0, 12, 23; panels in filename order, magenta = prediction.

Run `...-02/` independently produced the same numerical results. Run `...-01/`
stopped after synthesizing the first case because an initially omitted historical
helper was detected; it is incomplete, retained, and not used as evidence. A
regression test now exercises the historical Legacy function with that helper.

## Why the wrapper differs

For Lumière, Train Robbery, Méliès and Caligari, differences are confined to
frames **0 and 23**: the wrapper abstains for insufficient adjacent context.
The interior masks match PR24 exactly.

For Gertie, the raw-luma median-difference guard splits before frames **4, 6, 8**,
with values **24, 23, 24** against the existing threshold **22**. This creates
additional abstention boundaries and changes temporal percentile windows. Changed
masks occur at **0, 1, 3, 6, 7, 8, 16, 23**. Frames 1 and 16 differ without a
boundary reason because they use a different temporal reference window.

These are suspected cuts according to the implementation, not human-confirmed
scene edits. The data do not justify simply removing the guard or tuning its
threshold on Gertie. A future cut-handling change needs independently annotated
cuts, flicker negatives and a new held-out test set.

## What remains open

The PR24 improvement is real for its known synthetic damage. It does not establish
high precision on all historical damage: PR24 Precision here ranges from about
**0.193 to 0.677**. The tests also confirm a simple static vertical stripe can
remain unselected, while a moving stripe is detected. This establishes a synthetic
mechanism limitation, not the cause of any particular real Neighbors defect.

No Telea, DART or ProPainter restoration was run in this replay. Historical repair
MSE claims were not reproduced. No new generator/model was installed. The broader
GUI suite was not rerun because no application source was changed. Nine focused
CPU tests pass, including source/overwrite safeguards, pinned detector agreement,
historical helper execution and annotation-export contracts.

## Human annotation gate

**Updated next step:** the user prefers existing datasets/generators over new
manual annotation. The [AbsoluteDegradation preflight](absolute-degradation-benchmark-preflight.md)
records the alternative synthetic-mask route and its acquisition requirements.
The sample below remains available for optional later real-damage validation;
it is not a prerequisite for continuing synthetic detector evaluation.

A fixed exploratory sample from `evaluation/Neighbors_72.mkv` is prepared under
`.cache/dust-real-annotation-20261106-01/`, at zero-based frames
**0, 12, 23, 36, 48, 71**. `samples.csv` and `manifest.json` retain source/image
hashes, native dimensions, pending status and unknown defect classes. Images are
decoded full-size BGR8 exports; no GT masks were invented. Selection is independent
of detector masks but is not a fresh validation holdout.

Reproduce with:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_dust_annotation.py --source evaluation/Neighbors_72.mkv --output .cache/dust-annotation-new
```

Human review must identify dust versus scratches and persistent versus transient
damage using neighbouring frames. Use grayscale labels **0=reviewed healthy,
127=unknown/ignore, 255=defect**; keep defect classes distinguishable. An empty
automatic template is not a reviewed negative. If these frames omit an important
class, explicitly add a human-selected sample and record its selection basis.

**Real-damage validation input:** human-labelled real-defect masks or an existing
equivalent labelled dataset. Until then, real-damage precision remains unknown.
Synthetic research may proceed via the generator route above. Preserve current
production settings; do not infer a safe correction from synthetic F1 alone.
