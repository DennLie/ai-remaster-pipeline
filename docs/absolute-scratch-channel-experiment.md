# AbsoluteDegradation scratch-channel experiment — 2026-11-06

## Completed result

The full scorer completed all **36 method/condition combinations**: 144 damaged
frames plus 72 reference-activity controls, each scored by four paths. All code
hashes recorded at scoring still match the current source. No settings were tuned
after observing either development or reserved test-content results.

Micro-aggregated metrics on **test-content clips 010 and 020**, both seeds:

| Method | Precision | Recall | Pixel F1 | FP / evaluated unaffected pixels |
|---|---:|---:|---:|---:|
| Legacy wrapper | 0.011226 | 0.254226 | 0.021503 | 8.1683% |
| Temporal Dust wrapper | 0.104569 | 0.136361 | 0.118367 | 0.4260% |
| Ridge wrapper | 0.301839 | 0.058509 | 0.098019 | 0.0494% |
| Ridge core | 0.279701 | 0.060666 | 0.099706 | 0.0570% |

Counts are summed before computing metrics; F1 is not averaged across frames.
Weak-opacity ignored pixels are excluded. These are known added-scratch labels,
not manual assertions about every source-image structure.

On reference-only test-content frames, selected area is **8.2533% Legacy,
0.4143% Temporal, 0.0461% Ridge wrapper, 0.0531% Ridge core**. This measures
detector activity; it is not certified authentic-damage false-positive rate.

Development positive-control F1: **0.022161 Legacy, 0.150002 Temporal,
0.085737 Ridge wrapper, 0.105716 Ridge core**. No development-based tuning followed.

**Interpretation:** the PR24 gains on simple moving 3px lines do not imply strong
performance on these variable-width, partial/curved, drifting and short-lived
official parametric scratches. All existing methods are weak at this operating
point. Ridge abstains from much of the damage; it finds only about 5.9% of the
strong-label pixels on reserved test content. Removing its wrapper guards alone
would not solve this experiment: the core also has F1 about 0.10. Temporal's higher
F1 here comes with appreciably more off-target selection; it is not a universal
winner and is not adopted automatically.

The exact cause of misses (width, geometry filters, temporal novelty, intrinsic
expansion, etc.) has not yet been isolated. A targeted ablation is the next
research step, not a justified production correction. A subsequent tuned method
needs new independent source content and damage trajectories for confirmation.

## Scope and fixed protocol

This is a **partial generator benchmark**, not a reproduction of the complete
AbsoluteDegradation pipeline or a dust benchmark. It uses the unchanged official
`FilmScratchGenerator` from v1.0.0 commit
`542856fd7e275a6dcec80311f7f7b4541bdc4f70`.

The full generator's official RTN texture-template link returned HTTP 404.
No replacement textures or fabricated dust labels were substituted. The
parametric scratch component works independently of these assets.

Before scoring, fixed in `protocol.json` and the script:
- 24 frames each from reference clips **000, 010, 020** in Kaggle Synthetic
  Degraded Eval **version 6**; all methods see the same grayscale content as BGR8.
- Development clip **000**, reserved test-content clips **010 and 020**.
- Seeds **20261106, 20261107**, reused across clips for paired content controls.
  These produce two scratch trajectories on three backgrounds, not six fully
  independent damage trajectories. No statistical population claim is made.
- Canonical scratch settings from the official YAML, except `probability=1`
  conditions positive samples on damage presence. Generator cadence **24 fps**
  is an experimental setting, not recovered timing metadata for the source PNGs.
- Reference-only frames are separate detector-activity controls.
- Original float masks are saved losslessly. For binary metrics, opacity
  **>=0.10** is defect; **0<opacity<0.10** is ignored; zero is unaffected.
  This is a declared experimental convention, not a universal perceptual label.
- Detectors: unchanged Legacy, Temporal Dust, Ridge wrapper and Ridge core,
  sensitivity **0.65**, additional expansion **0**.

There is no tuning, repair, DART inference or production adoption. The split
supports a first paired comparison, not broad independent validation. DART's
training/evaluation relationship to these source data is not resolved here.

## Installation and acquisition

The user authorised up to **5 GB** of acquisition. The official source checkout
and its own frozen CPU environment are under `.cache/absolute-degradation-v1/`.
`uv sync --frozen --no-dev` created a new environment using Python 3.13.1; neither
ARP's nor either DART environment was installed into or modified.

The official blue-noise asset was acquired and verified against upstream SHA-256
`c6b1d40b4076b3c47dca86af88b24d74c0b04e8ac9b187dfba46bfd13d6e00fb`
(41,943,168 bytes). It is available for later full-pipeline use, **not used** by
this scratch-only experiment. Reference PNG payloads total **98,962,797 bytes**.
The script caps reference acquisition at **250 MB**. Dependency downloads were
on the order of 170 MiB; there was no 7.53GB full-dataset download.

REDS reference images retain their attribution requirements; synthetic dataset
third-party terms are not replaced by the generator's MIT licence. This work
does not establish commercial clearance for DART or RTN texture assets.

## Commands and artifacts

From the ARP repository root:

```powershell
git clone --branch v1.0.0 --depth 1 https://github.com/TytanMikJas/AbsoluteDegradation.git .cache/absolute-degradation-v1
```

Inside that new checkout:

```powershell
uv sync --frozen --no-dev --python I:\ai-remaster-pipeline\.venv\Scripts\python.exe
.\.venv\Scripts\python.exe scripts/download_blue_noise.py
```

From ARP root (output must be new):

```powershell
.\.cache\absolute-degradation-v1\.venv\Scripts\python.exe scripts/benchmark_absolute_scratches.py generate --upstream .cache/absolute-degradation-v1 --root .cache/absolute-scratch-benchmark-new
.\.venv\Scripts\python.exe scripts/benchmark_absolute_scratches.py score --root .cache/absolute-scratch-benchmark-new
```

Generation and scoring can each exceed two minutes. Allow at least ten minutes
for scoring on this CPU. Partial output is retained, not automatically resumed
or overwritten. A complete scoring report is published last.

Actual root: `.cache/absolute-scratch-benchmark-20261106-01/`:
- `protocol.json`: settings written before generation/scoring.
- `acquisition.json`: URLs, hashes, byte counts for 72 reference PNGs.
- `generated.json`: complete marker for 144 damaged frames.
- `<clip>/<seed>/`: degraded PNGs, float soft masks, binary/ignore labels, config.
- `artifact-audit.json`: reference hashes and mask geometry/range/support checks.
- `scored/`: exact predicted masks and final `report.json` when scoring completes.

The completed scorer additionally has `scored/summary.json`, derived from the
per-frame confusion counts, with the source report's SHA-256 and metrics by split
and condition. Six unscaled test-content grids are named
`scored/native-input-gt-legacy-temporal-ridge-core-<clip>-<frame>.png` for clips
010/020 and frames 0/12/23 at seed 20261106. Panels are in filename order;
magenta marks each prediction. Reference images are used at their supplied
1270x690 dimensions without new resizing. One grid was visually inspected as a
sanity check; that is not a human mask-quality ground-truth evaluation.

Both seeded generator runs were repeated and produced pixel-identical degraded
frames and float masks. All 72 acquired source hashes were rechecked. No changed
sample lies outside the generator's nonzero soft-mask support. A small number
of strong-support pixels remain numerically unchanged due to image clipping or
rendering/quantisation; opacity support is therefore not identical to a strict
nonzero image-difference label.

The initial foreground scorer hit its 240-second limit after seven conditions.
Its incomplete outputs are retained under `scored-incomplete-timeout-01/` and are
not a complete benchmark. A fresh full scorer was launched with a 600-second
limit and completed all nine input conditions. Interpret only its final complete
report, not the earlier partial directory. The generator also reached its initial
foreground timeout during/after process shutdown, after publishing all complete
markers; its actual saved inputs and masks were independently audited afterward.

Eleven focused CPU tests pass (nine existing Ridge/replay tests and two new
soft-mask threshold/range tests). No broad GUI regression is claimed.
