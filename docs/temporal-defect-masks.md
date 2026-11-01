# Conservative temporal defect masks for archival film

**Status: experimental, CPU-only and non-destructive.** This is the first
restoration preparation step for the ARP fork; it is **not DART**. It
creates pixel-level *proposals* for transient dust, small blotches and
short scratches without changing the source imagery. It does not execute
DART, RTN, ProPainter, Wan VACE, or any other AI checkpoint.

## Use in ARP GUI

1. Select source footage and enable **Clean Up** on the Overview tab.
2. Open Clean Up; enable **Detect film damage (mask only; no AI)**.
3. For a mask-only run, **disable both AI DeScratch and Dearchive**.
   (DeVignette is optional; leave off for fully unchanged picture content.)
4. Start Clean Up. ARP writes the normal cleaned processing master and a
   companion `*_defect_mask.mp4` file beside it.
5. Inspect the mask **before** attempting any inpainting. White means
   *suspected defect*, black means *preserve the source*.

The `*_defect_mask.mp4` is a viewing proxy; it uses H.264 compression.
The underlying **lossless mask master** (`temporal_defects_v1_*.mkv`)
is retained under `.cache/cleanup_chunks/<source>_<identity>/`.
Use the lossless mask, not the MP4 proxy, for future reconstruction and
pixel-level evaluation. Each mask is geometrically scaled to the original
size for the browser preview. The analysis can also run at source size.

The Clean Up output is *not* restored by the detector. Like any ARP
processing master, it can be re-encoded, so this is **not a guarantee
of byte-identical original pixels**. The image content is not deliberately
repainted, generated or inpainted by this step.

## CLI

```bash
python scripts/cleanup_video.py --source input/film.mkv \
  --detect-defects --defect-sensitivity 0.65 \
  --defect-mask-height 720 --no-dearchive
```

The results can be cached/reused. The stage considers its preview output
required in addition to the cleaned video, and regenerates it when missing.

## Algorithm and limitations

The detector uses no learned weights:

- Optical-flow alignment (OpenCV Farneback) of previous and following
  frames to the current frame.
- Candidate pixels must be inconsistent with **both** aligned neighbors,
  while those neighbors substantially agree.
- A spatial median filter checks for impulse-like local contrast.
- Connected-component constraints exclude wide scene objects, bars,
  edges and implausibly extensive masks; frame cuts are suppressed.
- First and last frames are deliberately left unmasked because they
  lack two-sided temporal evidence.

This approach favours precision and preservation over recall. It is
not reliable for **persistent scratches visible on consecutive frames**,
severe tears, rapid motion, occlusions, dissolves, irregular exposures,
or large damaged regions. It does not claim that a mask is correct
or that the historically authentic image is known. No automated
inpainting is enabled by this PR.

### DART licensing and future backend

[DART's documented third-party licence inventory](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer/blob/main/THIRD_PARTY_LICENSES.md)
identifies adapted RTN code from an upstream repository without an explicit
licence. Until this is resolved, this fork does **not** import DART code or
download DART weights, nor mark a DART backend as commercially cleared.
A later review can integrate an appropriately licensed DART backend behind
the same `0/255` mask contract and benchmark it against this detector.

## How to validate on real footage

Annotate 100–300 frames from *Neighbors* (1920), including clean frames,
transient dust, vertical scratches, camera movement, faces, strong grain,
cuts and titles. Use human-drawn ground-truth masks and report:

- Pixel precision/recall and intersection-over-union of defect masks.
- False-positive *intact pixel* rate, in particular on faces, captions
  and genuine narrow structural edges.
- Per-scene false-positive counts, skipped frames and processing fps.
- A separate benchmark for motion-compensation failure and film flicker.

Measure at source dimensions with matching timestamps. Keep DART and
ProPainter out of any claim of unrestricted commercial use until their
own complete licence terms are verified. See
[model licence audit](model-license-audit.md).
