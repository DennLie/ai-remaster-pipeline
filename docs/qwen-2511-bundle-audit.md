# Qwen-Image-Edit-2511: file provenance and offline verification

**Status: research / identity only, NOT licence clearance.** This is a
separate follow-up to PR #1. The [manifest](qwen-2511-file-manifest.json)
records the four model files that ARP's
`scripts/dependency_manager.py::QWEN_IMAGE_EDIT_MODELS` installs, their
Hugging Face repository identities, SHA-256 values documented by their
publishers, and available commit references.

All four source repositories currently **declare Apache-2.0**, but this
is **not** a legal finding for the full ARP workflow. Exact ComfyUI
node revisions, software licences, notices, model derivative terms and
optional companion models must also be checked.

## Verify actual installed files without downloading anything

From the repository root (activate your existing environment if using
external model paths described by `extra_model_paths.yaml`):

```bash
python scripts/audit_qwen_2511_bundle.py --check-manifest-only
python scripts/audit_qwen_2511_bundle.py --comfy-dir tools/comfyui --report output/qwen-2511-audit.json
```

The script compares its inventory to the downloader's actual four-file
list, resolves existing shared ComfyUI model paths using ARP's own path
resolver and calculates the installed files' SHA-256. It writes a JSON
**file-identity** report. It does not download, execute, run a GPU,
modify model files or switch on the commercial model guard allowlist.
It exits nonzero for missing or differing files.

A mismatch **does not by itself mean a licence violation**: the
Unsloth GGUF is actively updated and can legitimately change its
SHA-256 at `main`. The reference hash here is tied to a documented
source revision; inspect the actual checkpoint and update the manifest
through a reviewed PR if ARP needs a newer version.

## Important hidden dependency

In `scripts/qwen_colorize_references.py`,
`--describe-references` is **enabled by default**. When another color
reference is present, `reference_colours()` imports
`QWEN_IMAGE_21_SHARED_MODELS` from `scripts/dependency_manager.py`
and downloads a `qwen3vl_8b_int8_convrot.safetensors` encoder from
**Comfy-Org/Qwen-Image-2.1**. That is not part of the four-file
Qwen-Image-Edit-2511 bundle and potentially inherits separate,
restrictive model licensing.

Do **not** interpret a clean four-file hash report as permission to run
this optional reference-description path commercially. The commercial
guard from PR #1 intentionally still blocks Qwen reference generation.

Before a commercial allowlist entry is introduced, engineers should
review that dependency and, if needed, expose and enforce a
`--no-describe-references` option, while preserving desired color
consistency with approved methods. Also examine any Ollama-based
continuity descriptions, checkpoint revisions and model licences.

## Next steps

- Capture exact installed bytes and ComfyUI/custom-node revisions on the
  deployment GPU machine.
- Pin downloader revisions if/when all four artefacts are confirmed
  compatible with the selected ComfyUI graph and workflow.
- Audit any additional models used on Qwen outpaint guides, image edits
  and continuity references.
- Add a commercial-mode allowlist **only after** the full workflow is
  reviewed. Matching SHA-256 is necessary for provenance but not
  sufficient for commercial use.

See [first-pass licence audit](model-license-audit.md) and
[Commercial model guard](commercial-model-guard.md).
