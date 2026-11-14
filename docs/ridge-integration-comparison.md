# Ridge: PR24 reference versus ARP integration

Follow-up: the [five-film positive-control replay](ridge-positive-control-replay.md)
reproduces all published Ridge F1 values and quantifies wrapper losses, including
Gertie's additional suspected-cut splits. Real-damage annotation remains open.

This offline comparison measures agreement between predictions, **not defect
accuracy**. It does not render repairs, download anything, change a project,
select a winner, or alter production parameters. Human-labelled real damage is
not available for the inspected clips.

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts/compare_ridge_integration.py --source evaluation/Neighbors_72.mkv --output .cache/ridge-comparison-new
```

The output directory must not exist. The script requires the local Git object
`15af65261a733b6472553d4d32fefcfa43253210` (PR #24). It loads only `_as_stack`
and `detect_sequence` from the pinned research file; it does not import its
benchmark harness, clean reference images or ground-truth dependencies.

Exactly 24 frames, starting at zero by default (`--start-frame` selects a different
range), are decoded sequentially at native geometry into BGR8. The same frozen
arrays go to three paths, with sensitivity **0.65** and **zero additional dilation**:

1. Unchanged PR24 reference detector.
2. Current isolated `film_scratch_ridge.detect_sequence` core.
3. Actual `dust.masks_for` window/cut/endpoint wrapper, with only its frame reader
   replaced by that frozen 24-frame stream. It does not read beyond this excerpt.

The intrinsic Ridge 1x3 dilation remains part of both cores. The input, settings,
pixel dimensions, runtime versions and source-code SHA-256 values are recorded.
Source file hashes are checked before/after computation. Each path saves 24 exact
native-size grayscale PNG masks. `comparison.png` is a **downsampled display** of
five sampled frames: input, three magenta-mask overlays, and red disagreement
between PR24 and the wrapper. Red means *different predictions*, not a proven
false positive or false negative. `report.json` is written last.

## Local result: 2026-11-06

Outputs: `.cache/ridge-pr24-comparison-20261106-01/`, reproduced under
`.cache/ridge-pr24-comparison-20261106-02/` with identical numerical findings;
each contains two independent comparisons.
Both use frames **0–23**, **960x720**, without added preprocessing. The existing
stabilized video was read as-is, not regenerated. Its preparation differs from
the original; comparing these two cases does not isolate which prior operation
caused any prediction change.

| Input | PR24 versus current core | PR24 versus ARP wrapper |
|---|---|---|
| `evaluation/Neighbors_72.mkv` | Exact match; all 24 masks empty | Exact match; all 24 masks empty |
| `intermediate/stabilized/Neighbors_stabilized_eb46d58c.mkv` | Exact match on all frames | Different only at frame 23 (24th frame): PR24 selects 922 pixels, wrapper selects zero |

On the stabilized excerpt, PR24 selects **1,222 pixels at frame 22** and **922 at
frame 23**; all other masks are empty. The wrapper matches frame 22 but abstains
at the excerpt's final frame with `source_or_cut_boundary: insufficient context`.
This is the documented endpoint guard, not a different sensitivity value.

**Conclusion at this scope:** no detector-core/parameter mismatch was found.
Simply restoring the PR24 numbers will not make the original excerpt's masks
nonempty. The research core itself also outputs empty masks there. Empty output
does not prove absence of damage: this detector requires transient, thin, long,
strong two-sided ridge evidence and can reject stationary or weaker defects.

This does **not** replay the full five-film synthetic benchmark, validate the
entire 72-frame production run, or settle whether longer-window percentiles and
cut splitting harm performance elsewhere. It does not justify removing guards
or tuning on the old holdout films. A real quality conclusion requires labelled
damage and representative cases, not matching implementation output.

Three CPU tests cover exact pinned/core agreement, actual wrapper endpoint
abstention on a moving-stripe probe, a static-line negative control, real FFV1
input, unchanged source bytes and output-overwrite rejection. No GPU, active ARP
server, user-film result or pipeline adoption was changed.
