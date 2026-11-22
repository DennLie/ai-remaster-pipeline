"""Conservative model-launch guard for commercial remaster workflows.

This is NOT a licence grant, legal opinion, or certification. It checks
GUI/API model selection and component provenance. An audited model must include
its exact weights, LoRAs, text encoders, and dependencies.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ROOT

REGISTRY_PATH = ROOT / "configs" / "commercial_policy_registry.json"


class CommercialPolicyError(RuntimeError):
    """Raised when a model component fails commercial policy validation."""


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of commercial policy evaluation for a component."""

    allowed: bool
    status: str  # "allowed_osi", "conditional_cleared", "conditional_blocked", "hard_blocked", "un-audited", "unknown"
    license: str
    reason: str
    sha256: str | None = None
    component_id: str = ""


def load_policy_registry(path: Path | str | None = None) -> dict[str, Any]:
    """Load the commercial policy registry, failing closed on missing or corrupt files."""
    registry_file = Path(path) if path is not None else REGISTRY_PATH
    if not registry_file.exists():
        return {"schema_version": "1.0.0", "policies": {}}
    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"schema_version": "1.0.0", "policies": {}}
            return data
    except Exception:
        return {"schema_version": "1.0.0", "policies": {}}


def enabled(settings: Mapping[str, Mapping[str, str]] | None) -> bool:
    """Return True if Commercial Model Guard is enabled (default ON).

    Can be disabled when non_commercial_benchmark_override is set to 'true',
    or when commercial_model_guard is explicitly 'false'.
    """
    if not settings:
        return True
    global_cfg = settings.get("global", {})
    if str(global_cfg.get("non_commercial_benchmark_override", "false")).lower() == "true":
        return False
    return str(global_cfg.get("commercial_model_guard", "true")).lower() == "true"


def compute_sha256(file_path: Path | str) -> str:
    """Compute the SHA-256 hex digest for a local file."""
    path_obj = Path(file_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file not found at {path_obj}")
    h = hashlib.sha256()
    with open(path_obj, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def _find_model_in_category(models_dict: dict[str, Any], cid: str) -> tuple[str, dict[str, Any]] | None:
    norm_cid = cid.strip().lower().replace("_", "-").replace(" ", "-")
    for key, entry in models_dict.items():
        if not isinstance(entry, dict):
            continue
        norm_key = key.strip().lower().replace("_", "-").replace(" ", "-")
        if norm_cid == norm_key:
            return key, entry
        aliases = [str(a).strip().lower().replace("_", "-").replace(" ", "-") for a in entry.get("aliases", [])]
        if norm_cid in aliases:
            return key, entry
    return None


def evaluate_component(
    component_id: str,
    file_path: Path | str | None = None,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    conditional_attestations: Mapping[str, bool | str] | None = None,
    *,
    component_type: str = "checkpoint",
    registry: dict[str, Any] | None = None,
) -> PolicyDecision:
    """Granularly evaluate a checkpoint, LoRA, encoder, or model against commercial policy.

    Fails closed on any unknown or un-audited component.
    """
    reg = registry if registry is not None else load_policy_registry()
    policies = reg.get("policies", {})
    cid = component_id.strip()

    computed_sha256: str | None = None
    if file_path is not None:
        p = Path(file_path)
        if p.is_file():
            computed_sha256 = compute_sha256(p)

    # 1. Check HARD_BLOCK_COMMERCIAL
    hard_blocks = policies.get("HARD_BLOCK_COMMERCIAL", {}).get("models", {})
    match = _find_model_in_category(hard_blocks, cid)
    if match:
        _, entry = match
        lic = entry.get("license", "non-commercial")
        reason = entry.get("reason", f"{entry.get('name', cid)} is strictly prohibited for commercial use.")
        return PolicyDecision(
            allowed=False,
            status="hard_blocked",
            license=lic,
            reason=f"{entry.get('name', cid)}: {reason}",
            sha256=computed_sha256,
            component_id=component_id,
        )

    # 2. Check CONDITIONAL_COMMERCIAL
    conditional_models = policies.get("CONDITIONAL_COMMERCIAL", {}).get("models", {})
    match = _find_model_in_category(conditional_models, cid)
    if match:
        key, entry = match
        lic = entry.get("license", "conditional-commercial")
        attestations: dict[str, Any] = {}
        if settings:
            attestations.update(settings.get("global", {}))
            attestations.update(settings.get("colour", {}))
            attestations.update(settings.get("cleanup", {}))
            attestations.update(settings.get("outpaint", {}))
        if conditional_attestations:
            attestations.update(conditional_attestations)

        if key == "ltx":
            revenue_ok = _is_truthy(attestations.get("company_revenue_under_10m_usd"))
            ic_lora_ok = _is_truthy(attestations.get("ic_lora_audited")) or _is_truthy(attestations.get("ltx_ic_lora_audited"))
            gemma_ok = _is_truthy(attestations.get("gemma_encoder_audited")) or _is_truthy(attestations.get("ltx_gemma_encoder_audited"))
            bundle_ok = _is_truthy(attestations.get("ltx_components_audited"))
            components_ok = bundle_ok or (ic_lora_ok and gemma_ok)

            if revenue_ok and components_ok:
                return PolicyDecision(
                    allowed=True,
                    status="conditional_cleared",
                    license=lic,
                    reason="LTX conditional commercial clearance satisfied (revenue < $10M USD and audited components).",
                    sha256=computed_sha256,
                    component_id=component_id,
                )
            missing = []
            if not revenue_ok:
                missing.append("company_revenue_under_10m_usd: true")
            if not components_ok:
                missing.append("audited entries for IC-LoRA + Gemma-3 encoder (ic_lora_audited: true, gemma_encoder_audited: true)")
            return PolicyDecision(
                allowed=False,
                status="conditional_blocked",
                license=lic,
                reason=f"LTX conditional commercial criteria not satisfied: requires {'; '.join(missing)}.",
                sha256=computed_sha256,
                component_id=component_id,
            )

        if key == "deepexemplar":
            vgg_ok = _is_truthy(attestations.get("weight_vgg19_audited"))
            if vgg_ok:
                return PolicyDecision(
                    allowed=True,
                    status="conditional_cleared",
                    license=lic,
                    reason="Deep Exemplar conditional commercial clearance satisfied (weight_vgg19_audited: true).",
                    sha256=computed_sha256,
                    component_id=component_id,
                )
            return PolicyDecision(
                allowed=False,
                status="conditional_blocked",
                license=lic,
                reason="Deep Exemplar requires explicit weight and VGG19 audit flag (weight_vgg19_audited: true).",
                sha256=computed_sha256,
                component_id=component_id,
            )

    # 3. Check ALLOWED_OSI
    allowed_osi = policies.get("ALLOWED_OSI", {}).get("models", {})
    match = _find_model_in_category(allowed_osi, cid)
    if match:
        key, entry = match
        lic = entry.get("license", "OSI-Approved")
        name = entry.get("name", cid)
        audited_hashes = set(entry.get("audited_hashes", []))

        # Check local file against audited hashes if present
        if file_path is not None:
            p = Path(file_path)
            if computed_sha256 is not None:
                if audited_hashes and computed_sha256 not in audited_hashes:
                    return PolicyDecision(
                        allowed=False,
                        status="un-audited",
                        license=lic,
                        reason=f"Un-audited {component_type} file '{p.name}' (SHA-256: {computed_sha256}) rejected by commercial policy.",
                        sha256=computed_sha256,
                        component_id=component_id,
                    )
                if not audited_hashes and key == "qwen-image-edit-2511":
                    return PolicyDecision(
                        allowed=False,
                        status="un-audited",
                        license=lic,
                        reason=f"Un-audited {component_type} file '{p.name}' (SHA-256: {computed_sha256}) rejected by commercial policy. Qwen-2511 requires audited SHA-256 registration.",
                        sha256=computed_sha256,
                        component_id=component_id,
                    )
            return PolicyDecision(
                allowed=True,
                status="allowed_osi",
                license=lic,
                reason=f"{name} is cleared under OSI license ({lic}).",
                sha256=computed_sha256,
                component_id=component_id,
            )

        # file_path is None
        if key == "qwen-image-edit-2511":
            attestations = {}
            if settings:
                attestations.update(settings.get("global", {}))
            if conditional_attestations:
                attestations.update(conditional_attestations)
            if _is_truthy(attestations.get("qwen_2511_audited")):
                return PolicyDecision(
                    allowed=True,
                    status="allowed_osi",
                    license=lic,
                    reason=f"{name} is cleared under OSI license ({lic}) with audited weights.",
                    sha256=None,
                    component_id=component_id,
                )
            return PolicyDecision(
                allowed=False,
                status="un-audited",
                license=lic,
                reason="Qwen-Image-Edit-2511 quantized weights, Lightning LoRA and dependencies await full audit.",
                sha256=None,
                component_id=component_id,
            )

        return PolicyDecision(
            allowed=True,
            status="allowed_osi",
            license=lic,
            reason=f"{name} is cleared under OSI license ({lic}).",
            sha256=None,
            component_id=component_id,
        )

    # 4. Unknown / Fail-Closed
    if file_path is not None and computed_sha256 is not None:
        p = Path(file_path)
        reason = f"Un-audited {component_type} file '{p.name}' (SHA-256: {computed_sha256}) rejected by commercial policy. Unknown component '{component_id}' has not been reviewed."
    else:
        reason = f"Unknown component '{component_id}' has not been reviewed under commercial policy."

    return PolicyDecision(
        allowed=False,
        status="unknown",
        license="unknown",
        reason=reason,
        sha256=computed_sha256,
        component_id=component_id,
    )


def enforce_component(
    component_id: str,
    file_path: Path | str | None = None,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    conditional_attestations: Mapping[str, bool | str] | None = None,
    *,
    component_type: str = "checkpoint",
    registry: dict[str, Any] | None = None,
) -> PolicyDecision:
    """Enforce commercial policy on a component, raising CommercialPolicyError on rejection."""
    decision = evaluate_component(
        component_id,
        file_path=file_path,
        settings=settings,
        conditional_attestations=conditional_attestations,
        component_type=component_type,
        registry=registry,
    )
    if not decision.allowed:
        raise CommercialPolicyError(decision.reason)
    return decision


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
    global_cfg = settings.get("global", {})

    if stage == "enhancement":
        method = model_override or values.get("method", "off")
        if method in {"nafnet", "restormer"}:
            reasons.append(f"{method} checkpoint and dependencies await commercial-use audit")
        elif method not in {"off", "unsharp", "cas", "multiscale", "richardson_lucy"}:
            reasons.append(f"unknown enhancement method {method!r} has not been reviewed")
    elif stage == "dust":
        method = model_override or values.get("repair", "telea")
        det = values.get("detector", "temporal")
        preset = values.get("preset", "")
        if method == "propainter":
            reasons.append("ProPainter is non-commercial (NTU S-Lab License 1.0)")
        elif method == "dart" or det in ("dart", "dart_harness") or preset.startswith("dart_"):
            reasons.append("DART adapted RTN code rights remain unresolved")
        elif method != "telea":
            reasons.append("unknown Dust & Scratches repair method")
        if det not in ("dart", "dart_harness"):
            from .dust_e2_adapter import DETECTOR_NAME as E2_DETECTOR_NAME, E2_DETECTOR_INFO
            if det == E2_DETECTOR_NAME:
                if not E2_DETECTOR_INFO.get("commercial_cleared", False):
                    reasons.append(f"detector {det!r} is not cleared for commercial use")
            else:
                try:
                    from film_defect_detectors import DEFAULT_REGISTRY
                    if not DEFAULT_REGISTRY.has(det, commercial_only=True):
                        reasons.append(f"detector {det!r} is not cleared for commercial use")
                except ImportError:
                    pass
    elif stage == "cleanup":
        if _on(values, "ai_descratch"):
            reasons.append("ProPainter / AI DeScratch is non-commercial (NTU S-Lab License 1.0)")
        if _on(values, "dearchive", "true"):
            revenue_ok = _on(global_cfg, "company_revenue_under_10m_usd")
            components_ok = _on(global_cfg, "ltx_components_audited") or (
                _on(global_cfg, "ic_lora_audited") and _on(global_cfg, "gemma_encoder_audited")
            )
            if not (revenue_ok and components_ok):
                reasons.append("LTX 2.3 Dearchive is not cleared for unrestricted commercial use")
        if _on(values, "detect_defects"):
            det = values.get("defect_detector", "farneback_3f")
            try:
                from film_defect_detectors import DEFAULT_REGISTRY
                if not DEFAULT_REGISTRY.has(det, commercial_only=True):
                    reasons.append(f"defect detector {det!r} is not cleared for commercial use")
            except ImportError:
                pass
    elif stage == "outpaint":
        model = model_override or values.get("outpaint_model", "oumoumad")
        if model == "h3":
            reasons.append("MiniMax H3 has territorial and other use restrictions")
        elif model in {"oumoumad", "official", "ltx25"}:
            revenue_ok = _on(global_cfg, "company_revenue_under_10m_usd")
            components_ok = _on(global_cfg, "ltx_components_audited") or (
                _on(global_cfg, "ic_lora_audited") and _on(global_cfg, "gemma_encoder_audited")
            )
            if not (revenue_ok and components_ok):
                reasons.append(f"{model} uses an LTX model/LoRA bundle without unrestricted clearance")
        elif model == "wanvace":
            reasons.append("Wan VACE bundle, quantization, step-distill LoRA and custom nodes await a licence audit")
        else:
            reasons.append(f"unknown Outpainting model {model!r} has not been reviewed")
    elif stage == "references":
        model = model_override or values.get("method", "qwen")
        if model == "ddcolor":
            pass
        elif model == "qwen21":
            reasons.append("Qwen-Image-2.1 has a research-only licence")
        elif model == "qwen":
            if not _on(global_cfg, "qwen_2511_audited"):
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
            revenue_ok = _on(global_cfg, "company_revenue_under_10m_usd")
            components_ok = _on(global_cfg, "ltx_components_audited") or (
                _on(global_cfg, "ic_lora_audited") and _on(global_cfg, "gemma_encoder_audited")
            )
            if not (revenue_ok and components_ok):
                reasons.append("LTX Colorizer is not cleared for unrestricted commercial use")
        elif model == "deepexemplar":
            vgg_audited = _on(global_cfg, "weight_vgg19_audited") or _on(values, "weight_vgg19_audited")
            if not vgg_audited:
                reasons.append("Deep Exemplar checkpoints and dependencies need licence verification")
        elif model == "openai":
            reasons.append("OpenAI Cloud Colorization uses separate service terms")
        elif model == "deoldify":
            pass
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
            revenue_ok = _on(global_cfg, "company_revenue_under_10m_usd")
            components_ok = _on(global_cfg, "ltx_components_audited") or (
                _on(global_cfg, "ic_lora_audited") and _on(global_cfg, "gemma_encoder_audited")
            )
            if not (revenue_ok and components_ok):
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
