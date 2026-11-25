#!/usr/bin/env python3
"""Unified CLI for pluggable shot-local video segmentation into compressed RLE .masks.npz."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.commercial_policy import CommercialPolicyError
from ai_remaster_gui.video_segmenters import (
    DEFAULT_SEGMENTER_REGISTRY,
    run_shot_segmentation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Track shot-local object masks with any pluggable VideoSegmenterBackend and save .masks.npz.",
    )
    parser.add_argument(
        "--list-segmenters",
        action="store_true",
        help="List all registered video segmenters with license and text-prompt metadata as JSON.",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="sam21",
        help="Registered segmenter name (default: sam21).",
    )
    parser.add_argument("--shot-id", type=str, default="shot_0000", help="Shot identifier stored in .masks.npz.")
    parser.add_argument("--frames", type=str, default="", help="Path to frame directory or shot video file.")
    parser.add_argument(
        "--prompts-json",
        type=str,
        default=None,
        help="Inline JSON object or path to JSON file defining spatial/mask prompts.",
    )
    parser.add_argument(
        "--text-prompts",
        action="append",
        default=[],
        help="Optional text query prompt (repeatable; supported by grounded_sam2, sam3, sam31).",
    )
    parser.add_argument("--output", type=str, default="", help="Destination .masks.npz path.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path or model ID.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu').")
    parser.add_argument("--min-mask-area", type=int, default=16, help="Minimum foreground mask area in pixels.")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="Probability/IoU binarization threshold.")
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=0.5,
        help="Alpha binarization threshold for matting adapters such as videomama.",
    )
    parser.add_argument(
        "--compliance-output-dir",
        type=str,
        default="output",
        help="Directory for compliance_report.json logging (pass empty string to skip).",
    )
    parser.add_argument(
        "--allow-non-commercial-benchmark",
        action="store_true",
        help="Explicitly unlock non-commercial/benchmark-only segmenters (sam3, sam31) and log run non-compliant.",
    )
    return parser


def _load_prompts_json(raw: str | None) -> dict[int, dict[str, Any]]:
    if not raw:
        return {}
    candidate = Path(raw)
    if candidate.is_file():
        loaded = json.loads(candidate.read_text(encoding="utf-8"))
    else:
        loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("--prompts-json must decode to a JSON object mapping object IDs to prompt specs.")
    return {int(k): v for k, v in loaded.items()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_segmenters:
        items = [
            {
                "name": s.name,
                "label": s.label,
                "component_id": s.component_id,
                "license": s.license,
                "commercial_tier": s.commercial_tier,
                "supports_text_prompts": s.supports_text_prompts,
                "aliases": list(s.aliases),
            }
            for s in DEFAULT_SEGMENTER_REGISTRY.list_specs()
        ]
        print(json.dumps({"segmenters": items}, indent=2))
        return 0

    if not args.frames or not args.output:
        parser.error("--frames and --output are required unless --list-segmenters is set.")

    prompts = _load_prompts_json(args.prompts_json)
    compliance_dir = Path(args.compliance_output_dir) if str(args.compliance_output_dir or "").strip() else None

    try:
        out = run_shot_segmentation(
            shot_id=args.shot_id,
            frames=Path(args.frames),
            prompts=prompts,
            output_npz_path=Path(args.output),
            segmenter=args.segmenter,
            text_prompts=args.text_prompts or None,
            checkpoint_path=args.checkpoint,
            device=args.device,
            min_mask_area=args.min_mask_area,
            iou_threshold=args.iou_threshold,
            alpha_threshold=args.alpha_threshold,
            allow_non_commercial_benchmark=bool(args.allow_non_commercial_benchmark),
            compliance_output_dir=compliance_dir,
        )
    except CommercialPolicyError as exc:
        print(f"Commercial Policy Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Tracking Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, "segmenter": args.segmenter, "output_npz_path": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
