# AbsoluteDegradation mask benchmark — preflight 2026-11-06

Follow-up: the user approved up to 5 GB. The [isolated scratch-channel
experiment](absolute-scratch-channel-experiment.md) records actual installation,
bounded reference acquisition and the unavailable texture-asset blocker. The
statements below describe the pre-acquisition inventory.

## Decision

Use an existing generator with automatic defect-mask output instead of requiring
new human annotations for the next quantitative experiment. Human annotation
remains necessary only for a later accuracy claim about unlabelled real damage.

No generator installation, data/asset download, inference or production change
has been performed in this preflight. HTTP requests read source and metadata.

## Verified published dataset

Dataset: `absolutedegradation/absolutecinema-synthetic-degraded-eval`, Kaggle
version **6** at inspection time. The view API reports **7,534,325,287 bytes**.
Full enumeration of the listing API (31 pages, maximum 200 entries per page):

| Root | Files | Sum of listed uncompressed bytes |
|---|---:|---:|
| `degraded/` | 3,000 | 3,132,719,954 |
| `gt/` | 3,000 | 4,401,591,940 |
| `croissant.json` | 1 | 13,393 |

The description identifies 30 clips of 100 frames with degraded/reference pairs.
**There are no segmentation-mask files in this release.** `gt/` means reference
images, not defect labels. Differences from the references include grain, blur,
photometric and geometric processing; thresholding those differences would not
produce authoritative scratch/dust ground truth.

Metadata sources (no image download):
- https://www.kaggle.com/api/v1/datasets/view/absolutedegradation/absolutecinema-synthetic-degraded-eval
- https://www.kaggle.com/api/v1/datasets/list/absolutedegradation/absolutecinema-synthetic-degraded-eval?pageSize=200
  Follow each `nextPageToken` with `pageToken` until empty.

## Verified generator interface

Official release tag **v1.0.0**:
- https://github.com/TytanMikJas/AbsoluteDegradation/tree/v1.0.0
- `absolute_degradation/factory.py`: `include_mask=True` is forwarded to the pipeline.
- `absolute_degradation/distortions/degradation_pipeline.py`: returns degraded
  frames, aligned reference frames and float32 masks in [0, 1]. Texture and
  parametric scratch masks are combined; masks do not label all global degradation.

Preserve these soft masks, and specify evaluation threshold/ignore handling
before scoring. Do not silently turn all nonzero opacity into certain defect.
The default pipeline crops to 256px patches and uses geometry transformations;
record effective geometry and paired-mask alignment rather than claiming native
full-frame evaluation. Default mask output is combined, not automatically a
separate dust/scratch annotation for per-class scoring.

The standard pipeline requires external Bringing Old Films texture templates and
a blue-noise basis (approximately 40 MB). The templates are not distributed with
the MIT generator code and retain their own terms. A research experiment is not
commercial clearance. Check assets and record hashes before use.

## Local inventory and limitations

ARP `.venv` has `hydra`, `omegaconf`, `scipy`, `cv2`, `numpy`, but no importable
`absolute_degradation`. No matching generator/REDS/texture archive was found in
the inspected top-level Downloads entries, `.cache/downloads`, or inspected
candidate directories. This was a scoped inventory, not a scan of all disks.

`I:\DART` was listed read-only; it contains the DART checkout, not an adjacent
AbsoluteDegradation checkout/data directory. `tools/dart` exists but listing it
returns access denied. Its contents cannot be certified absent or usable.
Permissions and either DART environment were left unchanged.

## Next authorised work boundary

Before acquisition, obtain explicit permission for the isolated official
generator checkout/environment and its required assets. Prefer a bounded clean
REDS subset over the full 7.53GB paired dataset if new source images are needed.
Publish asset sizes/terms where available; request another approval if an agreed
budget would be exceeded. Existing historical development clips can support a
synthetic-added-damage control, but are not pristine negatives for true FPR.

Then freeze source identities, seeds, severity settings and development/test
split. Compare existing Legacy, Temporal Dust and Ridge on identical generated
arrays before tuning. Keep historical film holdouts as reproduction controls,
not fresh validation data. Start with CPU; any DART GPU run needs separate
approval and remains a training-domain-related comparison.
