# Commercial model guard (experimental, opt-in)

The [model licence audit](model-license-audit.md) and its
[machine-readable inventory](model-license-inventory.json) describe the
primary-source evidence and unresolved model-chain issues.

The **Commercial model guard** option on the **Overview** tab enables a
conservative *model selection* preflight for GUI and HTTP-launched workflows.
It is **off by default** to preserve existing project behavior. Turn it on for
commercial projects; any selected generative backend without an audited
complete model bundle is blocked rather than assumed safe.

**Not a licence certification or grant.** Commercial clearance of the
application, any film footage, training data, exact model checkpoint,
quantized derivative, LoRA, ComfyUI custom node, FFmpeg build, dependency
and cloud provider must be assessed separately. Direct CLI execution of
`scripts/*.py` bypasses this GUI guard. Existing outputs are not audited.

## Backend decisions

| Stage / backend | Strict guard result | Rationale |
| --- | --- | --- |
| DeVignette only | Not blocked | No generative model; toolchain still needs review |
| ProPainter / AI DeScratch | Blocked | NTU S-Lab License 1.0, non-commercial |
| LTX Dearchive | Blocked | Commercial constraints and LoRA clearance |
| LTX outpainting (2.3/2.5) | Blocked | Full model/LoRA bundle not unrestricted |
| MiniMax H3 | Blocked | Territorial and model-use restrictions |
| Wan VACE 2.1 | Pending audit, blocked | Base weights may be Apache-2.0; exact GGUF, LoRA and nodes require verification |
| Qwen-Image-2.1 | Blocked | Research-only |
| Qwen-Image-Edit-2511 | Pending audit, blocked | Check exact quantized weights, Lightning LoRA, VAE and node chain |
| OpenAI image/colour | Blocked | API governed by separate service terms |
| ColorMNet / CMNET2 / both | Blocked | ColorMNet CC BY-NC-SA; `both` invokes ColorMNet |
| Deep Exemplar | Pending audit, blocked | MIT code does not automatically license every checkpoint/dependency |
| LTX Colorizer | Blocked | Full bundle not unrestricted |
| Stable Audio Open / MMAudio | Pending audit, blocked | Weight and dependency terms need review |
| FlashVSR / SeedVR2 / LTX upscale | Pending audit or restricted, blocked | Checkpoint / dependency clearance incomplete |
| Stabilization / shot detection / recomposition | Not blocked | No AI checkpoint selected; other obligations remain |

No supported generative backend is **certified for unrestricted commercial
use** by this initial PR. A later PR should add an inventory recording
repository, exact revision, downloaded file hashes, licences, notices and
evidence for every bundled component before selectively allowing backends.

## Enforcement points

- Whole-pipeline preflight before the background queue starts, including stages
  whose results might otherwise be reused.
- Individual stage run, outpaint-chunk regeneration, manual reference
  regeneration/edit previews, guide generation/editing and upscale preview.
- The setting is saved per project; existing projects default to guard off.
- Command-line scripts and installer downloads are not blocked. Treat this as
  a safety net for GUI workflows, not as a security boundary against users.
- Intermediate outputs created before guard activation are neither deleted nor
  retroactively cleared.

## Sources to re-check against pinned model revisions

- [ARP model inventory](installer-model-sources.md)
- [ProPainter](https://github.com/sczhou/ProPainter)
- [ColorMNet licences](https://github.com/yyang181/colormnet/blob/main/LICENSES)
- [CMNET2](https://github.com/dan64/cmnet2)
- [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [Wan 2.1 VACE](https://github.com/Wan-Video/Wan2.1)
- [Deep Exemplar](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)
