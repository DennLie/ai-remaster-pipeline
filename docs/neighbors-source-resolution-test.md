# Neighbors (1920): digital source vs. resampling resolution

**Experimental test; no model licence or film-distribution clearance.**
This is a follow-up to the mask benchmark in PR #4. No footage is
committed to the repository.

## Source provenance

- [Wikimedia Commons: Neighbors (1920).webm](https://commons.wikimedia.org/wiki/File:Neighbors_(1920).webm)
- **Original uploaded digital video:** VP8 WebM, **960 × 720**, 17 min 31 s,
  ~272 MB; uploaded January 30, 2016. The description credits a
  YouTube-hosted version, not an archival scan master. The exact
  original scan resolution and processing history are unknown.
- Wikimedia also supplies a 480p VP9 transcode of that 720p digital video.
  **It is not a separate 480p film scan.** A separate 2024 1440 × 1080
  restoration also exists but is a different film transfer. We deliberately
  do not mix different masters in the controlled scaling experiment.
- Commons marks the 2016 file **public domain in the United States** and
  warns that copyright may differ outside the US, **including Germany**.
  Before public release or commercial redistribution, separately clear
  the film, particular digital transfer and any added soundtrack.

## How this experiment is performed

The GitHub Actions job downloads the **original 960 × 720 WebM file** from
Wikimedia Commons, records its SHA-256 and takes four seconds from the
five-minute mark (`00:05:00`). No Vimeo/YouTube or DVD 480p transcode is
used as a separate source.

Three **resolutions of the exact same underlying frames** are tested:

| Variant | Size | Operation |
|---|---:|---|
| Downsampled | 640 × 480 | OpenCV area reduction |
| Digital source | 960 × 720 | No spatial resampling |
| Lanczos 2× | 1920 × 1440 | Ordinary interpolation, **not AI upscaling** |

Every variant passes through the same independent
`film_defect_masks.temporal_defect_mask` function with sensitivity
`0.65`. All proposals are mapped back to 960 × 720 pixel coordinates
for spatial comparisons. The job saves a **JSON report** and a
**three-frame contact sheet** showing original imagery and red
defect proposals at each resolution.

### What the result can and cannot show

The report includes mask coverage and cross-resolution Jaccard agreement.
**These are not quality/accuracy metrics.** Without manually annotated
ground truth, lower mask coverage does not mean fewer false positives,
and higher coverage does not mean more actual scratches detected.

Likewise, interpolating 720p to 1440p **creates no genuine film details**.
A proper AI super-resolution experiment needs a fixed, audited model,
independent source-quality evidence and manually checked defect masks.
We expect no automatic commercial clearance from this experiment.

To interpret the masks, inspect whether real defects are covered and
intact details (hair, faces, garments, titles) are left untouched.
Then annotate at least a small held-out frame subset and compute
precision/recall using the [DART comparison harness](dart-vs-temporal-mask-benchmark.md).

## Re-run locally

With Python, NumPy, OpenCV and FFmpeg available:

```bash
curl -fL \
  "https://upload.wikimedia.org/wikipedia/commons/2/27/Neighbors_%281920%29.webm" \
  -o Neighbors-1920-original.webm

ffmpeg -ss 00:05:00 -i Neighbors-1920-original.webm -t 4 \
  -an -c:v ffv1 -pix_fmt yuv420p neighbors-clip.mkv

python scripts/compare_source_resolution.py \
  --input neighbors-clip.mkv \
  --output-dir resolution-analysis \
  --max-frames 72 --sensitivity 0.65
```

The JSON and JPEG are stored in `resolution-analysis/`. The repository
does not require or ship the source video or a third-party super-resolution
checkpoint. The CI diagnostic artifacts expire according to the workflow's
retention setting.

**Release gate:** Do not merge this branch or change commercial processing
defaults based solely on the mask coverage of one sample.
