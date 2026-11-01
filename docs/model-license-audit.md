# Model licence audit — first evidence pass (2026-11-01)

> **Status: preliminary / not a commercial clearance certificate.** This document
> collects primary-source evidence for ARP's current runtime *model choices*.
> The machine-readable companion is [model-license-inventory.json](model-license-inventory.json).
> It is designed to keep the opt-in Commercial model guard deliberately
> conservative until each deployed model bundle is reviewed.

## What has been verified

### Promising permissive building blocks (NOT yet an approved complete ARP backend)

- **Wan 2.1 VACE 14B:** the [original model](https://huggingface.co/Wan-AI/Wan2.1-VACE-14B)
  and the [exact QuantStack GGUF repository](https://huggingface.co/QuantStack/Wan2.1_14B_VACE-GGUF)
  used by ARP both **declare Apache-2.0**. ARP additionally downloads a
  [Kijai-hosted Lightx2v step-distill LoRA](https://huggingface.co/Kijai/WanVideo_comfy/tree/main/Lightx2v),
  [repackaged UMT5 text encoder and VAE](https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged),
  and uses a ComfyUI/custom-node stack. The full, exact-file and
  dependency licence chain has **not** been established. In particular a
  host repository's lack of a licence badge cannot be treated as permission.
- **Qwen-Image-Edit-2511:** the [original checkpoint](https://huggingface.co/Qwen/Qwen-Image-Edit-2511),
  [Unsloth GGUF conversion](https://huggingface.co/unsloth/Qwen-Image-Edit-2511-GGUF),
  [lightx2v Lightning LoRA](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)
  and [Comfy-Org Qwen component repository](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
  declare **Apache-2.0**. This is strong evidence for prioritising an exact-file
  audit, **not** permission to certify the entire configured workflow without
  inspecting revisions, notices, custom nodes and third-party components.
- **DDColor (future optional reference engine):** the
  [code](https://github.com/piddnad/DDColor) and
  [published model repository](https://huggingface.co/piddnad/DDColor-models)
  declare **Apache-2.0**. Not currently an ARP backend: inclusion requires
  dependency review, checkpoint pinning and integration tests.

The statuses above intentionally differentiate **declared individual licence**
from **verified commercial deployability of a complete backend**. Even Apache-2.0
requires complying with its terms, e.g. retained copyright/licence notices and
relevant NOTICE files.

### Confirmed restrictions or substantial provenance problems

- **ProPainter:** [NTU S-Lab License 1.0](https://github.com/sczhou/ProPainter);
  ARP's AI DeScratch currently runs this non-commercial model.
- **ColorMNet / CMNET2:** [ColorMNet licences](https://github.com/yyang181/colormnet/blob/main/LICENSES)
  include **CC BY-NC-SA 4.0**; [CMNET2](https://github.com/dan64/cmnet2)
  builds on ColorMNet. The ARP `colour.method=both` mode includes ColorMNet.
- **Qwen-Image-2.1:** ARP installs a
  [Viggle derivative](https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/blob/main/README.md)
  whose authors expressly identify the **Qwen Research License** and
  non-commercial-only use.
- **LTX-2.3 family:** [Lightricks' full community licence](https://huggingface.co/Lightricks/LTX-2.3/blob/main/LICENSE)
  requires a separate paid agreement for entities with annual revenue
  of at least **USD 10 million**, and includes additional permitted-use
  conditions. This is incompatible with the user's requirement for a
  revenue-independent, unconditional commercial backend. LoRAs and
  LTX-2.5 components require their own exact-version assessment.
- **Stable Audio Open 1.0:** the
  [model card](https://huggingface.co/stabilityai/stable-audio-open-1.0)
  declares a *Stable Audio Community License*, explicitly redirecting
  commercial users to Stability's licence terms.
- **DART (candidate, not yet integrated):** despite MIT licensing of
  most project code, DART's own
  [third-party licence register](https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer/blob/main/THIRD_PARTY_LICENSES.md)
  identifies adapted RTN code from
  [raywzy/Bringing-Old-Films-Back-to-Life](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life)
  with **no original licence file**. This is a material blocker for
  promising unrestricted commercial reuse without a licence grant or
  a properly independent replacement. RAFT and SwinIR are separately
  listed and must retain notices.
- **Deep Exemplar:** [MIT source code](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization/blob/main/LICENSE)
  is promising, but a source-code licence alone does not prove that
  every ARP-downloaded checkpoint or adapter is covered.

## Separate code-distribution obligations

- The [ComfyUI repository](https://github.com/comfyanonymous/ComfyUI)
  is **GPL-3.0**. GPL code can be used commercially; it is **not** equivalent
  to an NC model licence. Distribution of a combined application may impose
  copyleft and source-availability obligations, depending on the exact
  integration. Hosting a service and distributing binaries differ legally.
- [FFmpeg](https://ffmpeg.org/legal.html) is LGPL/GPL depending on
  enabled features and build flags. The binaries ARP installs or packages
  require a **build-specific** assessment. No conclusion is drawn from
  `ffmpeg -version` alone.
- Model output rights and film-source rights are independent. A public-domain
  claim for a 1920 film does not automatically clear every surviving score,
  specific digital restoration, transfer or local jurisdictional rights.

## Required evidence before allowing any model

1. **Identify exact runtime artifact:** repository, revision, file path,
   model family, LoRA/quantization lineage, declared licence, source of grant.
2. **Pin and fingerprint:** immutable HF/Git revision plus SHA-256 of each
   downloaded checkpoint, LoRA, encoder, VAE and executable custom node.
3. **Review derivative terms:** a quantized derivative cannot convey rights
   absent from its base model; a LoRA may have independent restrictions.
4. **Review software distribution:** the Python packages, ComfyUI nodes,
   FFmpeg builds, CUDA extensions and notices; decide distribution vs SaaS.
5. **Document legal decision:** who checked, date, evidence URL/revision,
   notices and permitted uses. Default guard state stays *blocked* before
   an explicit, tested allowlist change.

## Next engineering work (not part of this PR)

Prioritise (A) a reproducible **Qwen 2511** pinned bundle audit,
(B) a **Wan 2.1 VACE without the optional unverified distillation LoRA**
path if image quality and performance permit, and (C) **DDColor** image
reference generation. Keep ProPainter, ColorMNet, Qwen-Image-2.1 and LTX
restricted in strict mode. Evaluate a separately implemented
temporal defect detector until DART's adapted RTN provenance is resolved.

**Not assessed in this pass:** exact FFmpeg binaries, local installed ComfyUI
nodes, the model files' SHA-256 hashes, runtime weights for FlashVSR,
SeedVR2, MMAudio and Deep Exemplar, or downstream film rights. These
remain explicitly **pending** in the companion inventory rather than
being silently allowed.
