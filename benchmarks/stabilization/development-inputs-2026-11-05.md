# Longer development inputs — 2026-11-05

## Completed

Acquired only the four approved **development-film transfers**, using the exact
URLs in the recovered PR #18 manifests. Each complete file's byte count and
SHA-256 match both its manifest and the pinned catalogue at commit
`b790a15487faa01005f8166451e9716c1526e0e0`. No alternate transfer was selected.

| Film | Bytes | Transfer SHA-256 |
|---|---:|---|
| Lumiere | 5,948,262 | `653527c6ad9df215580de278f471d35b290e1033a73e478e4adc3b5dc79d839f` |
| Great Train Robbery | 81,832,180 | `2fe29266a0c253addcda17b9591d020c7e4d343702be273097eeee89ebdfc5ce` |
| Melies | 76,751,098 | `991a4cc1e126dad41ff828ce561dc02b36030d51dfc38ea6f28fd0ad965a1e94` |
| Corner in Wheat | 81,496,454 | `3efa8d2056d2e8912a0fb743b23102849d2d2af44ed6a0972b8751f77b1d1609` |

Total: **246,027,994 bytes**. Research use only: these are the same historical
digital transfers/proxies, not newly verified scans or commercially cleared
restorations. The held-out Caligari/Gertie/Nosferatu full transfers were not
downloaded and no held-out stabilization candidates were rendered.

## Prepared candidate windows

Extended all nine existing development time windows to **20 seconds** each,
without choosing new time marks after viewing stabilization scores:

| Film / existing window | Source interval, seconds | Frames | Cadence |
|---|---|---:|---:|
| Lumiere / approach_probe | [5, 25) | 600 | 30 fps |
| Lumiere / crowd_probe | [25, 45) | 600 | 30 fps |
| Train Robbery / interior_action_probe | [95, 115) | 480 | 24 fps |
| Train Robbery / moving_train_probe | [480, 500) | 480 | 24 fps |
| Melies / stage_probe | [110, 130) | 480 | 24 fps |
| Melies / trick_probe | [485, 505) | 480 | 24 fps |
| Corner in Wheat / field_probe | [95, 115) | 480 | 24 fps |
| Corner in Wheat / rural_workers_probe | [525, 545) | 480 | 24 fps |
| Corner in Wheat / late_scene_probe | [660, 680) | 480 | 24 fps |

All **4,560 frames / 180 seconds** were decoded successfully. Native geometry
and rational cadence were retained; the extracted FFV1/BGR0 clips have no audio,
scaling, stabilization, interpolation or synthetic degradation. Crucially,
**every first 24 decoded BGR frames matches its existing PR #18 clip pixel for
pixel**. This checks extraction alignment and decoded-pixel preservation, not
authentic historical-image quality. The nine native clips total 465,540,769 bytes.

## Local artifacts and checks

Under `.cache/stabilization-corpus/`:

- `development-transfers-pr18-b790a154/`: four verified `.webm` files and
  `acquisition.json` with exact URLs, hashes, byte counts and rights notes.
- `development-windows-20s-pr18-b790a154-01/<film>/<window>/`: lossless
  `original_native.mkv`, exact `extraction-command.json`, `extraction.log` and
  an input contact sheet sampled once per second.
- `development-windows-20s-pr18-b790a154-01/preparation.json`: preparation
  status, FFmpeg version, source identities, selected ranges and clip hashes.
- `development-windows-20s-pr18-b790a154-01/benchmark-candidates.json`: all nine
  candidates accepted by the existing runner's manifest/hash validation.
- `prepare-development-windows.py`: local extraction/check record. It refuses
  to overwrite the prepared output directory.

At preparation time no application/backend code, presets, defaults, active
project or original corpus clips were changed, and no long-window stabilization
comparison had run. The subsequent comparison and viewing workflow is recorded
in the [offline viewing guide](offline-review-2026-11-05.md).

## Review gate still open

These are **candidate windows**, not nine approved continuous shots. No
`cut_after_frames` annotations were invented: all nine remain unannotated,
and the manifest explicitly marks input review as pending. The first contact
sheet inspection already shows a transition late in Train Robbery's extended
`moving_train_probe`; the 24-frame pilot could not reveal that transition.
Its exact boundary must be checked on the decoded timeline, not guessed from
one-second samples.

Before selecting a preset, review the longer timelines for shot boundaries,
intentional camera motion, quiet backgrounds, contrast and input artifacts.
Trim/annotate where necessary and record any missing test conditions. Contact
sheets alone cannot validate temporal behavior. Then follow the fixed
[smoothing/zoom/holdout protocol](README.md#preset-selection-protocol).
Human quality judgements remain unfilled; no preset winner is claimed.
