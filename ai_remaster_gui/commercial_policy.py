"""Conservative model-launch guard for commercial remaster workflows.

This is NOT a licence grant, legal opinion, or certification. It checks
GUI/API model selection only: running scripts directly bypasses the guard.
An audited model must include its exact weights, LoRAs and dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping


def enabled(settings: Mapping[str, Mapping[str, str]]) -> bool:
    return settings.get("global", {}).get("commercial_model_guard") == "true"


def _on(values: Mapping[str, str], key: str, default: str = "false") -> bool:
    return values.get(key, default) == "true"


def stage_issue(
    settings: Mapping[str, Mapping[str, str]],
    stage: str,
    *,
    model_override: str | None = None,
) -> str:
    """Return an explanation when a stage is not cleared, or an empty string."""
    if not enabled(settings):
        return ""

    values = settings.get(stage, {})
    reasons: list[str] = []
    if stage == 'enhancement':
        method = model_override or values.get('method', 'off')
        if method in {'nafnet', 'restormer'}:
            reasons.append(f'{method} checkpoint and dependencies await commercial-use audit')
        elif method not in {'off', 'unsharp', 'cas', 'multiscale', 'richardson_lucy'}:
            reasons.append(f'unknown enhancement method {method!r} has not been reviewed')
    elif stage == 'dust':
        method = model_override or values.get('repair', 'telea')
        det = values.get('detector', 'temporal')
        preset = values.get('preset', '')
        if method == 'propainter':
            reasons.append('ProPainter is non-commercial (NTU S-Lab License 1.0)')
        elif method == 'dart' or det in ('dart', 'dart_harness') or preset.startswith('dart_'):
            reasons.append('DART adapted RTN code rights remain unresolved')
        elif method != 'telea':
            reasons.append('unknown Dust & Scratches repair method')
        if det not in ('dart', 'dart_harness'):
            try:
                from film_defect_detectors import DEFAULT_REGISTRY
                if not DEFAULT_REGISTRY.has(det, commercial_only=True):
                    reasons.append(f'detector {det!r} is not cleared for commercial use')
            except ImportError:
                pass
    elif stage == "cleanup":
        if _on(values, "ai_descratch"):
            reasons.append("ProPainter / AI DeScratch is non-commercial (NTU S-Lab License 1.0)")
        if _on(values, "dearchive", "true"):
            reasons.append("LTX 2.3 Dearchive is not cleared for unrestricted commercial use")
        if _on(values, "detect_defects"):
            det = values.get("defect_detector", "farneback_3f")
            try:
                from film_defect_detectors import DEFAULT_REGISTRY
                if not DEFAULT_REGISTRY.has(det, commercial_only=True):
                    reasons.append(f"defect detector {det!r} is not cleared for commercial use")
            except ImportError:
                pass
        # DeVignette alone has no generative checkpoint.
    elif stage == "outpaint":
        model = model_override or values.get("outpaint_model", "oumoumad")
        if model == "h3":
            reasons.append("MiniMax H3 has territorial and other use restrictions")
        elif model in {"oumoumad", "official", "ltx25"}:
            reasons.append(f"{model} uses an LTX model/LoRA bundle without unrestricted clearance")
        elif model == "wanvace":
            reasons.append("Wan VACE bundle, quantization, step-distill LoRA and custom nodes await a licence audit")
        else:
            reasons.append(f"unknown Outpainting model {model!r} has not been reviewed")
    elif stage == "references":
        model = model_override or values.get("method", "qwen")
        if model == "qwen21":
            reasons.append("Qwen-Image-2.1 has a research-only licence")
        elif model == "qwen":
            reasons.append("Qwen 2511 quantized weights, Lightning LoRA and dependencies await full audit")
        elif model == "openai":
            reasons.append("OpenAI API operates under separate service terms, not unrestricted open-source rights")
        else:
            reasons.append(f"unknown reference model {model!r} has not been reviewed")
    elif stage == "colour":
        model = model_override or values.get("method", "deepexemplar")
        if model in {"colormnet", "cmnet2", "both"}:
            reasons.append(f"{model} includes ColorMNet under CC BY-NC-SA (non-commercial)")
        elif model == "ltx":
            reasons.append("LTX Colorizer is not cleared for unrestricted commercial use")
        elif model == "deepexemplar":
            reasons.append("Deep Exemplar checkpoints and dependencies need licence verification")
        elif model == "openai":
            reasons.append("OpenAI Cloud Colorization uses separate service terms")
        else:
            reasons.append(f"unknown colorization model {model!r} has not been reviewed")
    elif stage == "audio":
        if _on(values, "create_music", "true"):
            reasons.append("Stable Audio Open checkpoint has not been cleared for unrestricted use")
        if _on(values, "create_sfx", "true"):
            reasons.append("MMAudio weights and dependencies have not been fully audited")
    elif stage == "upscale":
        model = model_override or values.get("method", "flashvsr")
        if model in {"ltx25", "ltx25cq"}:
            reasons.append("LTX upscaling models/LoRAs are not cleared for unrestricted commercial use")
        else:
            reasons.append(f"{model} upscaler weights and dependencies have not been fully audited")
    elif stage not in {"denoise", "deflicker", "stabilize", "shots", "recomp", "output"}:
        reasons.append(f"unrecognized stage {stage!r} has not been reviewed")

    if not reasons:
        return ""
    return f"Commercial model guard blocked {stage}: {'; '.join(reasons)}. See docs/commercial-model-guard.md."


def pipeline_issues(settings: Mapping[str, Mapping[str, str]], stages: Iterable[str]) -> list[str]:
    """Reject the queue before starting any stage if selected backends are unaudited."""
    return [reason for stage in stages if (reason := stage_issue(settings, stage))]
