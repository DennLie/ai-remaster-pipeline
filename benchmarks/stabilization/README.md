# Offline stabilization benchmark

Run the **real ARP/libvidstab backend** with 0/1/2/3% additional zoom, identical
motion settings and one detected cut manifest shared across zooms. No browser,
model download, project write, adoption or change to production defaults.
Requires the existing Python environment, NumPy/OpenCV and FFmpeg/libvidstab.

## Reuse the already defined films

The sample definition is not new:

- [PR #18](https://github.com/DennLie/ai-remaster-pipeline/pull/18) defines the
  seven-film, 17-window corpus. Its catalogue is pinned at commit
  `b790a15487faa01005f8166451e9716c1526e0e0`. The documented first visual review
  covered the original 14 windows, not all 17. Some original motive labels were
  corrected after inspection; never infer scene content from a label alone.
- [PR #22](https://github.com/DennLie/ai-remaster-pipeline/pull/22) used the first
  24-frame native excerpt from Lumière, Great Train Robbery and Méliès.
- `pr22-pilot.json` selects exactly those three **input clips**, verifies the
  pinned transfer SHA-256 against `corpus_manifest.json`, then verifies each
  `original_native.mkv` against its manifest SHA-256. It preserves the source
  and rights metadata. It does not duplicate the downloader or merge stacked PRs.

Use an already prepared PR #18 corpus on disk:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_stabilization.py --manifest benchmarks/stabilization/pr22-pilot.json --corpus-root X:/film_corpus --output-dir .cache/stabilization-benchmark/pilot-01
```

`X:/film_corpus/<film>/<scene>/original_native.mkv` and
`X:/film_corpus/<film>/corpus_manifest.json` must exist. Nothing is acquired
automatically. Preparation is documented in PR #18; original/proxy transfer
rights remain a separate review. This checkout does not include those movies.

The new experiment adds rotations (up to 0.5 degrees), translations (up to
2 pixels) with seed 2026, and a four-zoom sweep. It is **not an exact replay**
of PR #22's translation-only degradation. These 24-frame inputs are regression
probes, not evidence about full scenes or long films.

## Full scenes and longer sequences

Use a JSON manifest for local sources, with end-exclusive frame ordinals:

```json
{
  "version": 1,
  "cases": [
    {
      "id": "reviewed_pan",
      "source": "my-film.mkv",
      "start_frame": 600,
      "end_frame": 1800,
      "cut_after_frames": [],
      "tags": ["pan", "human-reviewed"],
      "synthetic": {"seed": 2026, "translation_px": 2, "rotation_deg": 0.5}
    }
  ]
}
```

Paths are relative to the manifest. Optional `sha256` pins a local source;
every run records its actual SHA-256 regardless. `cut_after_frames` lists
complete, human-reviewed boundaries relative to the selected clip (zero-based:
cut after frame 9 means between clip frames 9 and 10). Omit that field if cuts
are **not annotated**; an explicit empty list declares no real cuts. Detection
counts are reported only with annotations. All detected plus annotated cuts
are excluded from adjacent-motion measurements, for every candidate equally.

Without `synthetic`, only the real-input condition runs, with **no PSNR**.
With it, both unmodified input and injected-jitter conditions run. Setting
both amplitudes to zero supplies an explicit synthetic identity control.
Select motion, cuts, fades, occlusions, low contrast, titles, grain and quiet
scenes deliberately, without changing settings after seeing the winning score.
An existing directory is refused; use a new output directory for each run.

## Outputs and interpretation

The [2026-11-05 execution record](pilot-2026-11-05.md) documents the completed
three-film short pilot and the remaining evidence needed for preset selection.
The [longer development-input record](development-inputs-2026-11-05.md) documents
the four verified full transfers and nine prepared, not yet approved,
20-second candidate windows.
The [input-screening record](input-review-2026-11-05.md) documents observed
source damage, suspect automatic resets, a missed transition and coverage gaps.
The [offline viewing guide](offline-review-2026-11-05.md) explains the nine
four-panel review videos, playlist, native masters and blank human-review form.

- `benchmark.json`: completion/failure status, sources, hashes, runtime and code
  fingerprints, settings, per-case measurements and rejected/trusted matches.
- `summary.csv`: per-case/per-zoom motion proxies, fallback scenes/frames,
  geometric output coverage and estimated cropped source area. No pooled
  film-wide winner or artificial confidence interval.
- Each mode: exact CLI `commands.json`, FFmpeg logs, lossless masters, scene
  reports, persisted analysis/masks, `motion_frames.csv`, and contact sheets
  sampled throughout the selected sequence rather than only its beginning.
- Synthetic conditions: injected affine matrices plus `reference_frames.csv`
  with full-frame decoded **8-bit BGR** MSE/PSNR and a translation-only phase
  proxy against the corresponding digital base frame. Phase response below
  0.2 is rejected (a heuristic, not calibrated confidence); rotation and zoom
  can confound it. Infinite PSNR is represented by `null` + `exact_match=true`.
- `visual_review.csv`: intentionally blank human judgements for pans, cuts,
  grain/details and edges. Filling these is separate from running the program.

Image-content motion is not true historical transport weave. Lower motion can
erase an intentional pan. Full-frame PSNR includes crop, zoom, borders and
resampling; it does not prove authentic recovered detail. OpenCV's diagnostic
8-bit decode is not a high-bit-depth grain-preservation assay. Coverage uses
the raster estimator from package 2, not continuous area/kernel validity.
Varying-cadence metadata is rejected; no frame alignment is guessed.

Benchmark review stays **entirely offline and separate from film projects**.
Use labelled, synchronized comparison videos in a normal video player for
motion/transition review, and the native lossless masters for fine detail.
Do not import benchmark films or masters into an active ARP project, introduce
benchmark controls in its browser UI, or bypass stage provenance. Only a
validated preset with documented limits would eventually become an ARP option.

## Preset-selection protocol

Freeze this protocol before inspecting candidate scores; the short pilot is
an execution check, not a preset-selection dataset.

1. **Development films:** Lumiere, Great Train Robbery, Melies and Corner in
   Wheat. **Held-out films:** Caligari, Gertie and Nosferatu. Do not tune on
   held-out candidate outputs; checking input integrity is allowed. Adjacent
   windows from one film must not leak across these groups.
2. Extend the existing catalogue time windows into reviewed, preferably
   10–30-second shots from the **same hash-pinned transfers**. Record exact
   frame ranges and annotate cuts after inspecting the longer inputs. The
   old 24-frame review does not validate the extended shot. The old preparation
   script caps excerpts at 96 frames, so it cannot be reused unchanged for this
   step. Never loop/interpolate short clips to manufacture longer evidence.
3. Check input coverage for quiet shots, unwanted instability, moving subjects,
   low contrast, pans and transitions. The catalogue does not prove that all
   these conditions are present. Missing conditions remain explicit gaps;
   seeded synthetic controls supplement, but do not replace, real examples.
4. Compare smoothing radii **6 / 12 / 24 frames at fixed 3% zoom** against
   the unmodified input. Radius 12 is the current ARP baseline. Keep maximum
   translation **48 px**, maximum angle **3 degrees**, cut detection and all
   other backend settings fixed. Record cadence: a radius in frames has a
   different temporal extent at different frame rates. Do not pool raw pixel
   motion across different film geometries.
5. For the visually acceptable motion candidate(s), compare **0 / 1 / 2 / 3%
   zoom**, with identical input and motion settings. Prefer the least crop that
   does not introduce objectionable exposed edges. Coverage estimates support
   this judgement; they are not an automatic pass threshold.
6. Review in this order: no new jumps/pumping or damaged intentional pans;
   useful reduction of unwanted instability; acceptable crop and resampling.
   Retain unmodified-input comparisons. Report fallback scenes separately:
   preserved originals are safe fallbacks, not successful stabilization.
   PSNR, motion proxies and short regression clips cannot select the winner.
7. Freeze the selected candidate before rendering held-out films. A holdout
   failure is reported, not tuned away on the same set. If coverage is missing
   or no candidate is consistently safe, publish that limitation and keep
   stabilization opt-in/per-scene instead of declaring a universal preset.

Use separate new output directories for each smoothing run; the existing CLI
supports `--smoothing 6`, `12` and `24`. Use **`--zooms 3`** for this motion
phase so crop is held fixed and unnecessary zoom candidates are not rendered.
Omitting `--zooms` retains the default four-zoom sweep for step 5. No new search
engine or benchmark UI is necessary. The human-review cells remain blank until
an actual review. Automatic cut proposals are not reviewed cut ground truth;
inspect suspected false resets before interpreting smoothing measurements.
No preset or production-default change is implied by executing this protocol.
