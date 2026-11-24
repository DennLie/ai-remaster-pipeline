# Commercial model guard

The [model licence audit](model-license-audit.md) and its
[machine-readable inventory](model-license-inventory.json) describe the
primary-source evidence and unresolved model-chain issues.

The **Commercial model guard** option on the **Overview** tab enables a
conservative *model selection* preflight for GUI and HTTP-launched workflows.
It is **ON by default (`True`)** as of Phase 1 to guarantee out-of-the-box
compliance safety. Any selected generative backend without an audited,
commercially cleared complete model bundle is blocked rather than assumed safe.

### Non-commercial benchmark override

For non-commercial research, evaluation, and comparative benchmarking where
restricted models (e.g. ProPainter, Qwen research models, ColorMNet) must be executed,
the guard can be bypassed via:
- Setting `non_commercial_benchmark_override: "true"` in project settings.
- Explicitly passing `--allow-non-commercial-benchmark` (or `--guard-mode benchmark_override`)
  in CLI tools (`scripts/compliance_report.py`, `scripts/ddcolor_reference.py`).
- Setting `commercial_model_guard: "false"` in settings or CLI.

When the benchmark override is active, execution proceeds without blocking, but generated
compliance reports permanently record `guard_mode: "benchmark_override"` and flag the
run as non-compliant (`is_commercial_compliant: false`, `overall_commercial_compliant: false`).

**Not a licence certification or grant.** Commercial clearance of the
application, any film footage, training data, exact model checkpoint,
quantized derivative, LoRA, ComfyUI custom node, FFmpeg build, dependency
and cloud provider must be assessed separately. Direct CLI execution of
`scripts/*.py` bypasses this GUI guard unless explicitly invoked with compliance
reporting enabled. Existing outputs are not audited.

## Backend decisions

| Stage / backend | Strict guard result | Rationale |
| --- | --- | --- |
| DDColor (reference generation) | Allowed (OSI) | Apache-2.0 upstream code and checkpoints, OSI-compliant |
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
| OSI Masked LAB (video colorization) | Allowed (OSI) | Apache-2.0 / OpenCV DIS Optical Flow + native He et al. Guided Filter + SAM 2.1 Apache-2.0 masks, 100% OSI commercial-cleared |
| Stable Audio Open / MMAudio | Pending audit, blocked | Weight and dependency terms need review |
| FlashVSR / SeedVR2 / LTX upscale | Pending audit or restricted, blocked | Checkpoint / dependency clearance incomplete |
| Stabilization / shot detection / recomposition | Not blocked | No AI checkpoint selected; other obligations remain |

DDColor is cleared under Apache-2.0 as an OSI-compliant reference still colorizer.
OSI Masked LAB is cleared under Apache-2.0 as an OSI-compliant video colorizer.
Other generative backends remain subject to audit and clearance before being allowed
under strict guard mode.

## Enforcement points

- Whole-pipeline preflight before the background queue starts, including stages
  whose results might otherwise be reused.
- Individual stage run, outpaint-chunk regeneration, manual reference
  regeneration/edit previews, guide generation/editing and upscale preview.
- The setting is saved per project; projects default to guard ON (`commercial_model_guard: "true"`).
- Intermediate outputs created before guard activation are neither deleted nor
  retroactively cleared.

## Sources to re-check against pinned model revisions

- [ARP model inventory](installer-model-sources.md)
- [DDColor](https://github.com/piddnad/DDColor)
- [ProPainter](https://github.com/sczhou/ProPainter)
- [ColorMNet licences](https://github.com/yyang181/colormnet/blob/main/LICENSES)
- [CMNET2](https://github.com/dan64/cmnet2)
- [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1)
- [Wan 2.1 VACE](https://github.com/Wan-Video/Wan2.1)
- [Deep Exemplar](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)
