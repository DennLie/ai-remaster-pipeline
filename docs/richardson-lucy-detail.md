# Richardson–Lucy detail deblurring — research-aware classic backend

**Status:** optional Step-3 backend in a draft PR, no default processing change.

## What it does

`Detail Enhancement -> richardson_lucy` applies classic iterative
Richardson–Lucy deconvolution to the **Y (luminance) channel only**
at native source dimensions. Unlike Unsharp or CAS, this inverts an
assumed blur point-spread function (PSF). **The assumed Gaussian PSF
has not been measured from the original transfer.** Therefore any
apparent recovery of detail may also be overshoot, ringing, grain
amplification or damage enhancement.

| Preset | Assumed Gaussian PSF sigma | Richardson–Lucy iterations |
|---|---:|---:|
| Gentle | 0.75 pixels | 3 |
| Balanced | 1.00 pixels | 5 |
| Strong | 1.35 pixels | 8 |

The existing `Detail Enhancement` Mix control blends the enhanced
luma with the upstream frame from 0–100%; original video and audio
remain available for A/B/C inspection. No generative model is used.

## Technical behavior

- CPU scikit-image implementation, depends on **optional**
  `scikit-image`, `numpy`, `opencv-python`. These are loaded
  when the backend is selected; they are not new mandatory model
  dependencies of the normal ARP installation.
- Independent lossless or configured FFmpeg MKV intermediate.
- Video decoded to 8-bit RGB; conversion and luma processing
  are **gamma-encoded** rather than physically linear-light.
  Source material with higher bit depth is downsampled in this
  research-mode implementation; no high-bit-depth restoration
  accuracy is claimed.
- Nominal constant frame cadence and geometry checked; variable
  frame-rate footage is rejected. Audio streams are stream-copied
  from the unprocessed source.
- Output identity includes stage, original source fingerprint,
  PSF preset, mix and encoder profile. No original overwrite.
- Source resolution cannot improve and underlying absent details
  are not guaranteed to return.

## Scientific comparison

Evaluate on *Neighbors* using: original, Unsharp gentle, CAS gentle,
Multiscale gentle, Richardson–Lucy gentle, and in separate GPU
research NAFNet (PR #13). Use the same unmodified shot and
exact source-relative frames with scene boundaries documented.

Metric improvement from synthesized blurred test frames with known
ground truth does **not** prove authentic recovery on film copies.
For real film examine halos around intertitles, faces, hair,
clothing, high-contrast architectural edges, original grain and
DART defect masks. Inspect 1:1 native pixels and temporal stability.
Richardson–Lucy runs per-frame so temporal flicker is possible.

## License / provenance

scikit-image is open-source under BSD-family terms; audit installed
versions and their dependencies for distribution. FFmpeg licensing
depends on the actual build. There is no pretrained model,
weights or training-data license involved in this classic method.

Official reference:
https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.richardson_lucy

The user may intentionally prefer a more modern, sharper remaster.
That is an artistic choice, not a claim about the lost original.
