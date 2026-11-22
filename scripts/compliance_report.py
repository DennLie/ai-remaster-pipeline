#!/usr/bin/env python3
"""CLI tool for generating, inspecting, and validating compliance reports.

Supports cumulative multi-stage execution records, EU AI Act Article 50 disclosure
audits, and commercial model policy verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.compliance import (
    append_compliance_record,
    load_compliance_report,
    validate_compliance_report,
    determine_guard_mode,
    evaluate_overall_compliance,
    get_eu_ai_act_disclosure,
    normalize_model_record,
)


def _parse_key_val_pairs(items: list[str] | None) -> dict[str, Any]:
    """Parse a list of 'key=value' strings into a dictionary."""
    if not items:
        return {}
    res: dict[str, Any] = {}
    for item in items:
        if "=" in item:
            k, v = item.split("=", 1)
            k = k.strip()
            v = v.strip()
            # check boolean or integer
            if v.lower() in ("true", "1", "yes"):
                res[k] = True
            elif v.lower() in ("false", "0", "no"):
                res[k] = False
            else:
                try:
                    res[k] = int(v)
                except ValueError:
                    res[k] = v
        else:
            # Bare value, default to True
            res[item.strip()] = True
    return res


def _parse_seeds(seed_args: list[str] | None) -> dict[str, int]:
    """Parse seed arguments from 'name=int' or bare integers."""
    if not seed_args:
        return {}
    seeds: dict[str, int] = {}
    for idx, s in enumerate(seed_args):
        if "=" in s:
            k, v = s.split("=", 1)
            try:
                seeds[k.strip()] = int(v.strip())
            except ValueError:
                continue
        else:
            try:
                val = int(s.strip())
                key = "seed" if idx == 0 else f"seed_{idx}"
                seeds[key] = val
            except ValueError:
                continue
    return seeds


def inspect_report(
    path: Path | str,
    as_json: bool = False,
) -> int:
    """Inspect and display an existing compliance report."""
    target_path = Path(path)
    if target_path.is_dir():
        target_path = target_path / "compliance_report.json"

    data = load_compliance_report(target_path)
    if not data:
        print(f"[ERROR] Could not read compliance report at: {target_path}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(data, indent=2))
        return 0

    overall_ok = data.get("overall_commercial_compliant", False)
    status_str = "COMPLIANT (PASS)" if overall_ok else "NON-COMPLIANT (FAIL)"

    print("=" * 80)
    print(f"COMPLIANCE REPORT: {target_path}")
    print("=" * 80)
    print(f"Schema Version : {data.get('schema_version', 'unknown')}")
    print(f"Report ID      : {data.get('report_id', 'unknown')}")
    print(f"Created At     : {data.get('created_at', 'unknown')}")
    print(f"Updated At     : {data.get('updated_at', 'unknown')}")
    print(f"Overall Status : {status_str}")
    runs = data.get("runs", [])
    print(f"Total Runs     : {len(runs)}")

    # Top-level EU AI Act disclosure
    disc = data.get("eu_ai_act_article_50_disclosure", {})
    if disc:
        print("\nEU AI Act Article 50 Disclosure:")
        print(f"  Role               : {disc.get('role', 'deployer')}")
        print(f"  Effective Date     : {disc.get('compliance_effective_date', '2026-08-02')}")
        print(f"  Articles           : {', '.join(disc.get('applicable_articles', []))}")
        print(f"  Watermark/Metadata : {disc.get('machine_readable_watermark_or_metadata', False)}")
        print(f"  Notice             : {disc.get('disclosure_notice', '')}")

    print("\nExecution History:")
    for idx, run in enumerate(runs, 1):
        run_ok = run.get("is_commercial_compliant", False)
        run_status = "COMPLIANT" if run_ok else "NON-COMPLIANT"
        print("-" * 80)
        print(f"Run #{idx}: Stage '{run.get('stage', 'unknown')}' | {run.get('timestamp', '')}")
        print(f"  Guard Mode         : {run.get('guard_mode', 'strict')}")
        print(f"  Run Status         : {run_status}")
        print(f"  Seeds              : {run.get('seeds', {})}")
        print(f"  Color Bible        : {run.get('color_bible_version')}")
        print(f"  Attestations       : {run.get('conditional_attestations', {})}")

        models = run.get("models", [])
        print(f"  Models ({len(models)}):")
        for m in models:
            print(f"    - [{m.get('component_type', 'checkpoint')}] {m.get('model_id')}")
            print(f"        License : {m.get('license', 'unknown')}")
            print(f"        Status  : {m.get('status', 'unknown')}")
            if m.get("source_url"):
                print(f"        Source  : {m.get('source_url')}")
            if m.get("sha256"):
                print(f"        SHA-256 : {m.get('sha256')}")
            if m.get("file_path"):
                print(f"        Path    : {m.get('file_path')}")
            print(f"        Reason  : {m.get('reason', '')}")

    print("=" * 80)
    return 0


def validate_report(
    path: Path | str,
    strict: bool = True,
    quiet: bool = False,
) -> int:
    """Validate compliance report against schema and commercial rules."""
    is_valid, errors = validate_compliance_report(path)
    if not is_valid:
        if not quiet:
            print(f"[FAIL] Schema validation failed for: {path}", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
        return 1

    data = load_compliance_report(path)
    if not data:
        if not quiet:
            print(f"[FAIL] Unable to load report at: {path}", file=sys.stderr)
        return 1

    overall_ok = data.get("overall_commercial_compliant", False)
    if strict and not overall_ok:
        if not quiet:
            print(
                f"[FAIL] Report schema is valid, but overall_commercial_compliant is False: {path}",
                file=sys.stderr,
            )
        return 1

    if not quiet:
        status_label = "PASS (Commercially Compliant)" if overall_ok else "WARN (Non-Commercial Runs Recorded)"
        print(f"[PASS] Compliance verification successful [{status_label}]: {path}")
    return 0


def generate_record(
    output_dir: Path | str,
    stage: str,
    model_ids: list[str] | None = None,
    component_type: str = "checkpoint",
    file_path: str | None = None,
    source_url: str | None = None,
    license_tag: str | None = None,
    status: str | None = None,
    guard_mode: str = "strict",
    seeds: list[str] | None = None,
    color_bible_version: str | None = None,
    attestations: list[str] | None = None,
    watermark: bool = False,
    role: str = "deployer",
    disclosure_notice: str | None = None,
) -> int:
    """Generate or append a compliance record."""
    models: list[dict[str, Any]] = []
    if model_ids:
        for mid in model_ids:
            entry: dict[str, Any] = {
                "model_id": mid,
                "component_type": component_type,
            }
            if file_path:
                entry["file_path"] = file_path
            if source_url:
                entry["source_url"] = source_url
            if license_tag:
                entry["license"] = license_tag
            if status:
                entry["status"] = status
            models.append(entry)

    settings: dict[str, Any] = {
        "global": {
            "commercial_model_guard": "false" if guard_mode == "disabled" else "true",
            "non_commercial_benchmark_override": "true" if guard_mode == "benchmark_override" else "false",
            "eu_ai_act_role": role,
            "machine_readable_watermark": "true" if watermark else "false",
        }
    }
    if disclosure_notice:
        settings["global"]["disclosure_notice"] = disclosure_notice

    parsed_seeds = _parse_seeds(seeds)
    parsed_attestations = _parse_key_val_pairs(attestations)

    out_file = append_compliance_record(
        output_dir=output_dir,
        stage=stage,
        models=models,
        settings=settings,
        seeds=parsed_seeds,
        color_bible_version=color_bible_version,
        conditional_attestations=parsed_attestations,
    )

    print(f"[SUCCESS] Appended compliance record for stage '{stage}' -> {out_file}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for compliance_report.py."""
    parser = argparse.ArgumentParser(
        description="Compliance Report CLI: Generate, inspect, and validate multi-stage compliance reports and EU AI Act Article 50 disclosures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Root optional shortcut flags
    parser.add_argument(
        "--inspect",
        dest="root_inspect",
        metavar="PATH",
        nargs="?",
        const="output",
        help="Inspect an existing compliance report at PATH (default: output).",
    )
    parser.add_argument(
        "--validate",
        dest="root_validate",
        metavar="PATH",
        nargs="?",
        const="output",
        help="Validate compliance report at PATH against schema and commercial rules.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        default="output",
        help="Output directory containing compliance_report.json (default: output).",
    )
    parser.add_argument(
        "--stage",
        "-s",
        dest="stage",
        help="Pipeline stage name (e.g. references, colour, cleanup, upscale).",
    )
    parser.add_argument(
        "--model-id",
        "-m",
        dest="model_ids",
        action="append",
        help="Model ID to record (e.g. ddcolor, sam2, qwen). May be repeated.",
    )
    parser.add_argument(
        "--component-type",
        dest="component_type",
        choices=["checkpoint", "lora", "encoder", "tool"],
        default="checkpoint",
        help="Component type for recorded models (default: checkpoint).",
    )
    parser.add_argument(
        "--file-path",
        dest="file_path",
        help="Local path to model file (computes SHA-256 hash).",
    )
    parser.add_argument(
        "--source-url",
        dest="source_url",
        help="Source repository or Hugging Face URL for model.",
    )
    parser.add_argument(
        "--license",
        dest="license_tag",
        help="SPDX license tag or identifier.",
    )
    parser.add_argument(
        "--status",
        dest="status",
        choices=["allowed_osi", "conditional_cleared", "conditional_blocked", "hard_blocked", "unknown"],
        help="Policy status override.",
    )
    parser.add_argument(
        "--guard-mode",
        dest="guard_mode",
        choices=["strict", "benchmark_override", "disabled"],
        default="strict",
        help="Commercial guard mode (default: strict).",
    )
    parser.add_argument(
        "--allow-non-commercial-benchmark",
        dest="benchmark_override",
        action="store_true",
        help="Shortcut to set guard-mode to benchmark_override.",
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        action="append",
        help="RNG seed as 'name=123' or '123'. May be repeated.",
    )
    parser.add_argument(
        "--color-bible-version",
        dest="color_bible_version",
        help="Color Bible version tag or commit hash.",
    )
    parser.add_argument(
        "--attestation",
        dest="attestations",
        action="append",
        help="Conditional attestation flag as 'key=true'. May be repeated.",
    )
    parser.add_argument(
        "--watermark",
        dest="watermark",
        action="store_true",
        help="Flag indicating machine-readable watermark or metadata embedded.",
    )
    parser.add_argument(
        "--role",
        dest="role",
        default="deployer",
        help="EU AI Act role (deployer | provider, default: deployer).",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output raw JSON instead of formatted text.",
    )
    parser.add_argument(
        "--strict",
        dest="strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In validate mode: fail if overall_commercial_compliant is False (default: True, use --no-strict to disable).",
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        help="Quiet mode: suppress output, return exit code 0 or 1.",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="subcommand", help="Subcommand to run")

    # generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate or append a stage record")
    gen_parser.add_argument("--output-dir", "-o", default="output", help="Output directory")
    gen_parser.add_argument("--stage", "-s", required=True, help="Stage name")
    gen_parser.add_argument("--model-id", "-m", dest="model_ids", action="append", help="Model ID")
    gen_parser.add_argument("--component-type", choices=["checkpoint", "lora", "encoder", "tool"], default="checkpoint")
    gen_parser.add_argument("--file-path", help="Local model file path")
    gen_parser.add_argument("--source-url", help="Upstream source URL")
    gen_parser.add_argument("--license", dest="license_tag", help="License tag")
    gen_parser.add_argument("--status", choices=["allowed_osi", "conditional_cleared", "conditional_blocked", "hard_blocked", "unknown"])
    gen_parser.add_argument("--guard-mode", choices=["strict", "benchmark_override", "disabled"], default="strict")
    gen_parser.add_argument("--allow-non-commercial-benchmark", dest="benchmark_override", action="store_true")
    gen_parser.add_argument("--seed", dest="seeds", action="append", help="Seed key=val")
    gen_parser.add_argument("--color-bible-version", help="Color bible version")
    gen_parser.add_argument("--attestation", dest="attestations", action="append", help="Attestation key=val")
    gen_parser.add_argument("--watermark", action="store_true")
    gen_parser.add_argument("--role", default="deployer")
    gen_parser.add_argument("--notice", dest="disclosure_notice", help="Custom disclosure notice")

    # inspect subcommand
    insp_parser = subparsers.add_parser("inspect", help="Inspect an existing compliance report")
    insp_parser.add_argument("path", nargs="?", default="output", help="Path to report or output dir")
    insp_parser.add_argument("--json", dest="as_json", action="store_true", help="Output JSON")

    # validate subcommand
    val_parser = subparsers.add_parser("validate", help="Validate a compliance report against schema and rules")
    val_parser.add_argument("path", nargs="?", default="output", help="Path to report or output dir")
    val_parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strict exit code 1 if non-compliant (default: True, use --no-strict to disable)",
    )
    val_parser.add_argument("--quiet", action="store_true", help="Suppress output")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # 1. Handle root shortcuts if supplied
    if args.root_inspect is not None:
        return inspect_report(args.root_inspect, as_json=args.as_json)

    if args.root_validate is not None:
        return validate_report(args.root_validate, strict=args.strict, quiet=args.quiet)

    # 2. Handle subcommands
    if args.subcommand == "inspect":
        return inspect_report(args.path, as_json=args.as_json)

    if args.subcommand == "validate":
        return validate_report(args.path, strict=args.strict, quiet=args.quiet)

    if args.subcommand == "generate":
        g_mode = "benchmark_override" if args.benchmark_override else args.guard_mode
        return generate_record(
            output_dir=args.output_dir,
            stage=args.stage,
            model_ids=args.model_ids,
            component_type=args.component_type,
            file_path=args.file_path,
            source_url=args.source_url,
            license_tag=args.license_tag,
            status=args.status,
            guard_mode=g_mode,
            seeds=args.seeds,
            color_bible_version=args.color_bible_version,
            attestations=args.attestations,
            watermark=args.watermark,
            role=args.role,
            disclosure_notice=getattr(args, "disclosure_notice", None),
        )

    # 3. If stage provided at root level, treat as generate
    if args.stage:
        g_mode = "benchmark_override" if args.benchmark_override else args.guard_mode
        return generate_record(
            output_dir=args.output_dir,
            stage=args.stage,
            model_ids=args.model_ids,
            component_type=args.component_type,
            file_path=args.file_path,
            source_url=args.source_url,
            license_tag=args.license_tag,
            status=args.status,
            guard_mode=g_mode,
            seeds=args.seeds,
            color_bible_version=args.color_bible_version,
            attestations=args.attestations,
            watermark=args.watermark,
            role=args.role,
        )

    # Default to printing help
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
