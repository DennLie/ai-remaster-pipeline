"""Automated compliance reporter and EU AI Act Article 50 disclosure generator.

Maintains cumulative execution records in output/compliance_report.json,
tracks per-run model provenance, hashes, licenses, seeds, and attestations,
and evaluates overall commercial compliance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
import uuid

from .commercial_policy import (
    PolicyDecision,
    compute_sha256,
    evaluate_component,
    load_policy_registry,
)

SCHEMA_VERSION: str = "1.0.0"

DEFAULT_EU_AI_ACT_DISCLOSURE: dict[str, Any] = {
    "role": "deployer",
    "ai_generated_or_manipulated": True,
    "disclosure_notice": (
        "This media is an AI-assisted historical film restoration/colorization "
        "pursuant to EU AI Act Article 50(4)."
    ),
    "machine_readable_watermark_or_metadata": False,
    "compliance_effective_date": "2026-08-02",
    "applicable_articles": ["Art. 50(4)"],
}

VALID_STATUSES: set[str] = {
    "allowed_osi",
    "conditional_cleared",
    "conditional_blocked",
    "hard_blocked",
    "unknown",
}

VALID_COMPONENT_TYPES: set[str] = {
    "checkpoint",
    "lora",
    "encoder",
    "tool",
    "backend",
    "colorizer",
}

VALID_GUARD_MODES: set[str] = {
    "strict",
    "benchmark_override",
    "disabled",
}


def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() == "true"


def lookup_model_source_and_license(
    model_id: str,
    registry: dict[str, Any] | None = None,
) -> tuple[str | None, str | None]:
    """Look up source repository URL and license tag from the policy registry."""
    reg = registry if registry is not None else load_policy_registry()
    policies = reg.get("policies", {})
    norm_id = model_id.strip().lower().replace("_", "-").replace(" ", "-")

    for category in ("ALLOWED_OSI", "CONDITIONAL_COMMERCIAL", "HARD_BLOCK_COMMERCIAL"):
        models_dict = policies.get(category, {}).get("models", {})
        for key, entry in models_dict.items():
            if not isinstance(entry, dict):
                continue
            norm_key = key.strip().lower().replace("_", "-").replace(" ", "-")
            aliases = [
                str(a).strip().lower().replace("_", "-").replace(" ", "-")
                for a in entry.get("aliases", [])
            ]
            if norm_id == norm_key or norm_id in aliases:
                source = entry.get("source_repo") or entry.get("source_url")
                lic = entry.get("license")
                return source, lic
    return None, None


def determine_guard_mode(settings: Mapping[str, Any] | None) -> str:
    """Determine guard mode ('strict', 'benchmark_override', 'disabled') from settings."""
    if not settings:
        return "strict"

    if "guard_mode" in settings and str(settings["guard_mode"]) in VALID_GUARD_MODES:
        return str(settings["guard_mode"])

    global_cfg: Mapping[str, Any]
    if isinstance(settings.get("global"), Mapping):
        global_cfg = settings["global"]  # type: ignore[assignment]
    else:
        global_cfg = settings

    if "guard_mode" in global_cfg and str(global_cfg["guard_mode"]) in VALID_GUARD_MODES:
        return str(global_cfg["guard_mode"])

    override = _is_truthy(global_cfg.get("non_commercial_benchmark_override", "false"))
    if override:
        return "benchmark_override"

    guard_val = str(global_cfg.get("commercial_model_guard", "true")).strip().lower()
    if guard_val == "false":
        return "disabled"

    return "strict"


def is_run_compliant(guard_mode: str, models: list[dict[str, Any]]) -> bool:
    """Evaluate whether an individual run is commercially compliant.

    A run is compliant only if guard_mode is 'strict' and all models have
    status 'allowed_osi' or 'conditional_cleared'.
    """
    if guard_mode != "strict":
        return False
    for m in models:
        st = m.get("status")
        if st not in {"allowed_osi", "conditional_cleared"}:
            return False
    return True


def evaluate_overall_compliance(runs: list[dict[str, Any]]) -> bool:
    """Evaluate overall commercial compliance across all recorded runs.

    True only if ALL runs in the report are commercially compliant.
    """
    if not runs:
        return True
    return all(bool(r.get("is_commercial_compliant")) is True for r in runs)


def get_eu_ai_act_disclosure(
    settings: Mapping[str, Any] | None = None,
    role: str = "deployer",
    machine_readable_watermark: bool = False,
    custom_notice: str | None = None,
) -> dict[str, Any]:
    """Generate the EU AI Act Article 50 disclosure metadata block."""
    disclosure = dict(DEFAULT_EU_AI_ACT_DISCLOSURE)
    if settings:
        global_cfg: Mapping[str, Any]
        if isinstance(settings.get("global"), Mapping):
            global_cfg = settings["global"]  # type: ignore[assignment]
        else:
            global_cfg = settings

        if "eu_ai_act_role" in global_cfg:
            disclosure["role"] = str(global_cfg["eu_ai_act_role"])
        if "machine_readable_watermark" in global_cfg:
            disclosure["machine_readable_watermark_or_metadata"] = _is_truthy(
                global_cfg["machine_readable_watermark"]
            )
        if "disclosure_notice" in global_cfg:
            disclosure["disclosure_notice"] = str(global_cfg["disclosure_notice"])

    if role != "deployer":
        disclosure["role"] = role
    if machine_readable_watermark:
        disclosure["machine_readable_watermark_or_metadata"] = True
    if custom_notice:
        disclosure["disclosure_notice"] = custom_notice

    return disclosure


def extract_conditional_attestations(
    settings: Mapping[str, Any] | None = None,
    conditional_attestations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract and combine conditional attestations from settings and overrides."""
    attestations: dict[str, Any] = {}
    if settings:
        global_cfg: Mapping[str, Any]
        if isinstance(settings.get("global"), Mapping):
            global_cfg = settings["global"]  # type: ignore[assignment]
        else:
            global_cfg = settings

        known_attestation_keys = (
            "company_revenue_under_10m_usd",
            "ic_lora_audited",
            "gemma_encoder_audited",
            "ltx_components_audited",
            "weight_vgg19_audited",
            "qwen_2511_audited",
        )
        for k in known_attestation_keys:
            if k in global_cfg:
                attestations[k] = _is_truthy(global_cfg[k])

    if conditional_attestations:
        for k, v in conditional_attestations.items():
            if isinstance(v, (bool, str)):
                attestations[k] = _is_truthy(v)
            else:
                attestations[k] = v

    return attestations


def format_seeds(seeds: Mapping[str, int] | None) -> dict[str, int]:
    """Format seeds mapping to dict[str, int]."""
    if not seeds:
        return {}
    res: dict[str, int] = {}
    for k, v in seeds.items():
        try:
            res[str(k)] = int(v)
        except (ValueError, TypeError):
            continue
    return res


def normalize_model_record(
    model: dict[str, Any] | PolicyDecision | Any,
    settings: Mapping[str, Any] | None = None,
    conditional_attestations: Mapping[str, Any] | None = None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a model entry into the canonical schema required by compliance reports."""
    if isinstance(model, PolicyDecision):
        reg_source, reg_license = lookup_model_source_and_license(
            model.component_id, registry=registry
        )
        status = model.status if model.status in VALID_STATUSES else "unknown"
        return {
            "model_id": model.component_id,
            "component_type": "checkpoint",
            "file_path": None,
            "sha256": model.sha256,
            "source_url": reg_source,
            "license": model.license or reg_license or "unknown",
            "status": status,
            "reason": model.reason,
        }

    if not isinstance(model, Mapping):
        raise TypeError(
            f"Expected dict or PolicyDecision for model entry, got {type(model).__name__}"
        )

    raw_dict = dict(model)
    model_id = str(raw_dict.get("model_id") or raw_dict.get("component_id") or "").strip()

    # Component type normalization
    comp_type = str(raw_dict.get("component_type", "checkpoint")).strip().lower()
    if comp_type in {"text_encoder", "text-encoder"}:
        comp_type = "encoder"
    elif comp_type in {"tracker", "detector", "filter"}:
        comp_type = "tool"
    elif comp_type not in VALID_COMPONENT_TYPES:
        comp_type = "checkpoint"

    file_path = str(raw_dict["file_path"]) if raw_dict.get("file_path") else None
    sha256 = raw_dict.get("sha256")
    if sha256 is None and file_path:
        fp = Path(file_path)
        if fp.is_file():
            sha256 = compute_sha256(fp)

    reg_source, reg_license = lookup_model_source_and_license(model_id, registry=registry)
    source_url = raw_dict.get("source_url") or reg_source
    license_tag = raw_dict.get("license") or reg_license or "unknown"

    status = raw_dict.get("status")
    reason = raw_dict.get("reason")

    if not status or not reason:
        settings_map: Mapping[str, Mapping[str, str]] | None = None
        if isinstance(settings, Mapping):
            settings_map = {  # type: ignore[assignment]
                str(k): {str(subk): str(subv) for subk, subv in v.items()}
                for k, v in settings.items()
                if isinstance(v, Mapping)
            }
        decision = evaluate_component(
            model_id,
            file_path=file_path,
            settings=settings_map,
            conditional_attestations=conditional_attestations,
            component_type=comp_type,
            registry=registry,
        )
        if not status:
            status = decision.status if decision.status in VALID_STATUSES else "unknown"
        if not reason:
            reason = decision.reason
        if license_tag == "unknown" and decision.license != "unknown":
            license_tag = decision.license
        if sha256 is None and decision.sha256:
            sha256 = decision.sha256

    if status not in VALID_STATUSES:
        status = "unknown"

    return {
        "model_id": model_id,
        "component_type": comp_type,
        "file_path": file_path,
        "sha256": sha256,
        "source_url": source_url,
        "license": license_tag,
        "status": status,
        "reason": reason or "",
    }


def load_compliance_report(path: Path | str) -> dict[str, Any] | None:
    """Load an existing compliance_report.json file if it exists and is valid."""
    target_path = Path(path)
    if target_path.is_dir():
        target_path = target_path / "compliance_report.json"

    if not target_path.is_file():
        return None

    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return None
    return None


def resolve_color_bible_version(
    film_id: str | None = None,
    db_path: Path | str | None = None,
    settings: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve Color Bible snapshot version hash for a film if configured and available.

    If film_id is provided (or discoverable from settings.get("global", {}).get("color_bible_film_id")
    or settings.get("global", {}).get("film_id")), attempts to instantiate ColorBibleService(db_path=db_path)
    and call service.get_snapshot_hash(film_id). If DB does not exist, or film has no approved
    references/assignments, or any error occurs, cleanly returns None.
    """
    try:
        target_film_id = film_id
        target_db_path = db_path

        if not target_film_id and settings:
            global_cfg: Mapping[str, Any]
            if isinstance(settings.get("global"), Mapping):
                global_cfg = settings["global"]  # type: ignore[assignment]
            else:
                global_cfg = settings
            target_film_id = global_cfg.get("color_bible_film_id") or global_cfg.get("film_id")
            if target_db_path is None and "color_bible_db" in global_cfg:
                target_db_path = global_cfg["color_bible_db"]

        if not target_film_id or not str(target_film_id).strip():
            return None

        target_film_id = str(target_film_id).strip()

        from .color_bible_db import resolve_color_bible_db
        from .color_bible_service import ColorBibleService

        if target_db_path != ":memory:":
            resolved_path = resolve_color_bible_db(target_db_path)
            if not resolved_path.is_file():
                return None

        with ColorBibleService(db_path=target_db_path) as service:
            assignments = service.list_assignments(target_film_id)
            references = service.list_approved_references(target_film_id)
            if not assignments and not references:
                return None
            return service.get_snapshot_hash(target_film_id)
    except Exception:
        return None


def append_compliance_record(
    output_dir: Path | str,
    stage: str,
    models: list[dict[str, Any]],
    settings: Mapping[str, Any] | None = None,
    seeds: Mapping[str, int] | None = None,
    color_bible_version: str | None = None,
    conditional_attestations: Mapping[str, bool | str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Record a stage run into output/compliance_report.json, preserving multi-stage history.

    Atomically appends stage records to a cumulative runs list and updates the top-level
    overall_commercial_compliant status and EU AI Act Article 50 disclosure.
    """
    target_path = Path(output_dir)
    if target_path.suffix == ".json" or target_path.is_file():
        report_path = target_path
        target_dir = target_path.parent
    else:
        target_dir = target_path
        report_path = target_dir / "compliance_report.json"

    target_dir.mkdir(parents=True, exist_ok=True)

    guard_mode = determine_guard_mode(settings)
    attestations = extract_conditional_attestations(settings, conditional_attestations)
    normalized_models = [
        normalize_model_record(
            m,
            settings=settings,
            conditional_attestations=attestations,
        )
        for m in models
    ]
    run_compliant = is_run_compliant(guard_mode, normalized_models)
    formatted_seeds = format_seeds(seeds)
    if color_bible_version is None:
        cb_version = resolve_color_bible_version(settings=settings)
    else:
        cb_version = str(color_bible_version)
    disclosure = get_eu_ai_act_disclosure(settings=settings)

    now_iso = datetime.now(timezone.utc).isoformat()

    run_record: dict[str, Any] = {
        "stage": stage,
        "timestamp": now_iso,
        "guard_mode": guard_mode,
        "is_commercial_compliant": run_compliant,
        "models": normalized_models,
        "seeds": formatted_seeds,
        "color_bible_version": cb_version,
        "conditional_attestations": attestations,
        "eu_ai_act_article_50_disclosure": disclosure,
    }
    if metadata:
        run_record["metadata"] = dict(metadata)

    existing_data: dict[str, Any] = {}
    if report_path.is_file():
        loaded = load_compliance_report(report_path)
        if isinstance(loaded, dict):
            existing_data = loaded

    runs: list[dict[str, Any]] = existing_data.get("runs", [])
    if not isinstance(runs, list):
        runs = []

    runs.append(run_record)
    overall_compliant = evaluate_overall_compliance(runs)

    report_data: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "report_id": existing_data.get("report_id") or str(uuid.uuid4()),
        "created_at": existing_data.get("created_at") or now_iso,
        "updated_at": now_iso,
        "overall_commercial_compliant": overall_compliant,
        "runs": runs,
        "eu_ai_act_article_50_disclosure": disclosure,
    }

    # Atomic write: write to unique temporary file then atomic rename
    tmp_file = target_dir / f".compliance_report_{uuid.uuid4().hex}.tmp"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, report_path)
    finally:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass

    return report_path


def validate_compliance_report(
    report_or_path: dict[str, Any] | Path | str,
) -> tuple[bool, list[str]]:
    """Validate a compliance report dictionary or file against the schema.

    Returns (is_valid, errors).
    """
    errors: list[str] = []
    data: dict[str, Any]

    if isinstance(report_or_path, (str, Path)):
        p = Path(report_or_path)
        if p.is_dir():
            p = p / "compliance_report.json"
        if not p.is_file():
            return False, [f"Compliance report file not found at {p}"]
        try:
            with open(p, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, dict):
                    return False, ["Report JSON root must be an object"]
                data = loaded
        except Exception as e:
            return False, [f"Failed to parse compliance report JSON: {e}"]
    elif isinstance(report_or_path, dict):
        data = report_or_path
    else:
        return False, [f"Invalid input type: {type(report_or_path).__name__}"]

    # Validate top-level schema
    if "schema_version" not in data:
        errors.append("Missing required field 'schema_version'")
    if "overall_commercial_compliant" not in data or not isinstance(
        data["overall_commercial_compliant"], bool
    ):
        errors.append("Missing or non-boolean 'overall_commercial_compliant'")
    if "runs" not in data or not isinstance(data["runs"], list):
        errors.append("Missing or non-list 'runs'")
        return False, errors

    runs = data["runs"]
    for idx, run in enumerate(runs):
        if not isinstance(run, dict):
            errors.append(f"Run #{idx} is not an object")
            continue
        if "stage" not in run or not isinstance(run["stage"], str) or not run["stage"].strip():
            errors.append(f"Run #{idx} missing valid 'stage' string")
        if "timestamp" not in run or not isinstance(run["timestamp"], str):
            errors.append(f"Run #{idx} missing 'timestamp' string")
        if run.get("guard_mode") not in VALID_GUARD_MODES:
            errors.append(
                f"Run #{idx} has invalid guard_mode: {run.get('guard_mode')!r} (expected one of {VALID_GUARD_MODES})"
            )
        if "is_commercial_compliant" not in run or not isinstance(
            run["is_commercial_compliant"], bool
        ):
            errors.append(f"Run #{idx} missing or non-boolean 'is_commercial_compliant'")

        # Validate models
        if "models" not in run or not isinstance(run["models"], list):
            errors.append(f"Run #{idx} missing or non-list 'models'")
        else:
            for m_idx, m in enumerate(run["models"]):
                if not isinstance(m, dict):
                    errors.append(f"Run #{idx} model #{m_idx} is not an object")
                    continue
                if "model_id" not in m or not isinstance(m["model_id"], str) or not m["model_id"].strip():
                    errors.append(f"Run #{idx} model #{m_idx} missing valid 'model_id'")
                if m.get("component_type") not in VALID_COMPONENT_TYPES:
                    errors.append(
                        f"Run #{idx} model #{m_idx} invalid component_type: {m.get('component_type')!r}"
                    )
                if m.get("status") not in VALID_STATUSES:
                    errors.append(
                        f"Run #{idx} model #{m_idx} invalid status: {m.get('status')!r}"
                    )
                if "license" not in m or not isinstance(m["license"], str):
                    errors.append(f"Run #{idx} model #{m_idx} missing 'license' string")
                if "reason" not in m or not isinstance(m["reason"], str):
                    errors.append(f"Run #{idx} model #{m_idx} missing 'reason' string")

        # Validate seeds
        if "seeds" not in run or not isinstance(run["seeds"], dict):
            errors.append(f"Run #{idx} missing or non-dict 'seeds'")

        # Validate color_bible_version
        if "color_bible_version" in run and run["color_bible_version"] is not None:
            if not isinstance(run["color_bible_version"], str):
                errors.append(f"Run #{idx} 'color_bible_version' must be string or null")

        # Validate conditional_attestations
        if "conditional_attestations" not in run or not isinstance(
            run["conditional_attestations"], dict
        ):
            errors.append(f"Run #{idx} missing or non-dict 'conditional_attestations'")

        # Validate EU AI Act Article 50 disclosure
        if "eu_ai_act_article_50_disclosure" not in run or not isinstance(
            run["eu_ai_act_article_50_disclosure"], dict
        ):
            errors.append(f"Run #{idx} missing or non-dict 'eu_ai_act_article_50_disclosure'")
        else:
            disc = run["eu_ai_act_article_50_disclosure"]
            if "role" not in disc or not isinstance(disc["role"], str):
                errors.append(f"Run #{idx} disclosure missing 'role' string")
            if "ai_generated_or_manipulated" not in disc or not isinstance(
                disc["ai_generated_or_manipulated"], bool
            ):
                errors.append(f"Run #{idx} disclosure missing 'ai_generated_or_manipulated' bool")
            if "disclosure_notice" not in disc or not isinstance(disc["disclosure_notice"], str):
                errors.append(f"Run #{idx} disclosure missing 'disclosure_notice' string")
            if "machine_readable_watermark_or_metadata" not in disc or not isinstance(
                disc["machine_readable_watermark_or_metadata"], bool
            ):
                errors.append(
                    f"Run #{idx} disclosure missing 'machine_readable_watermark_or_metadata' bool"
                )
            if "compliance_effective_date" not in disc or not isinstance(
                disc["compliance_effective_date"], str
            ):
                errors.append(f"Run #{idx} disclosure missing 'compliance_effective_date' string")
            if "applicable_articles" not in disc or not isinstance(
                disc["applicable_articles"], list
            ):
                errors.append(f"Run #{idx} disclosure missing 'applicable_articles' list")

    # Validate overall consistency
    expected_overall = evaluate_overall_compliance(runs)
    if data["overall_commercial_compliant"] != expected_overall:
        errors.append(
            f"overall_commercial_compliant is {data['overall_commercial_compliant']} but calculated {expected_overall} from runs"
        )

    return len(errors) == 0, errors
