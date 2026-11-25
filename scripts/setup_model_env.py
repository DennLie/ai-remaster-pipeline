#!/usr/bin/env python3
"""CLI for inspecting, installing, and removing ARP managed isolated model environments."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.commercial_policy import CommercialPolicyError
from ai_remaster_gui.managed_model_envs import (
    inspect_all_envs,
    inspect_env,
    install_env,
    remove_env,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage isolated Python environments under .cache/model_envs/<env_name>/.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Inspect installation and policy status of managed model envs.")
    p_status.add_argument("--env", default=None, help="Optional single env name to inspect.")
    p_status.add_argument("--base-dir", type=Path, default=None, help="Optional custom model_envs base directory.")

    p_install = sub.add_parser("install", help="Create and populate a managed isolated model environment.")
    p_install.add_argument("env_name", help="Managed environment name (e.g. sam2_family, ddcolor_env).")
    p_install.add_argument("--cpu", action="store_true", help="Install CPU-only PyTorch wheels instead of CUDA 12.4.")
    p_install.add_argument("--dry-run", action="store_true", help="Print install commands without creating files or installing.")
    p_install.add_argument(
        "--allow-non-commercial-benchmark",
        action="store_true",
        help="Required to unlock installation of noncommercial_research (SAM 3 / SAM 3.1).",
    )
    p_install.add_argument("--base-dir", type=Path, default=None, help="Optional custom model_envs base directory.")

    p_remove = sub.add_parser("remove", help="Remove a managed isolated model environment directory.")
    p_remove.add_argument("env_name", help="Managed environment name to remove.")
    p_remove.add_argument("--base-dir", type=Path, default=None, help="Optional custom model_envs base directory.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "status":
            if args.env:
                payload = {"env": inspect_env(args.env, base_dir=args.base_dir)}
            else:
                payload = {"envs": inspect_all_envs(base_dir=args.base_dir)}
            print(json.dumps(payload, indent=2))
            return 0

        if args.command == "install":
            res = install_env(
                args.env_name,
                cuda=not bool(args.cpu),
                allow_non_commercial=bool(args.allow_non_commercial_benchmark),
                dry_run=bool(args.dry_run),
                base_dir=args.base_dir,
            )
            print(json.dumps({"ok": True, **res}, indent=2))
            return 0

        if args.command == "remove":
            res = remove_env(args.env_name, base_dir=args.base_dir)
            print(json.dumps(res, indent=2))
            return 0
    except CommercialPolicyError as exc:
        print(f"Commercial Policy Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Managed Env Error: {exc}", file=sys.stderr)
        return 1

    return 1


if __name__ == "__main__":
    sys.exit(main())
