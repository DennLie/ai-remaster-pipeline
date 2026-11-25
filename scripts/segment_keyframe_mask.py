#!/usr/bin/env python3
"""CLI for single-image Keyframe / Reference Still segmentation into binary PNG + JSON sidecar."""

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
from ai_remaster_gui.image_segmenters import (
    DEFAULT_IMAGE_SEGMENTER_REGISTRY,
    normalize_target_classes,
    run_keyframe_segmentation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment a single keyframe still using any registered ImageSegmenterBackend.",
    )
    parser.add_argument(
        "--list-segmenters",
        action="store_true",
        help="List all registered image segmenters with license and capability metadata as JSON.",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="birefnet",
        help="Registered keyframe image segmenter name (default: birefnet).",
    )
    parser.add_argument("--image", type=str, default="", help="Path to reference keyframe image.")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Destination binary PNG mask path (defaults to intermediate/keyframe_masks/<stem>_<segmenter>.png).",
    )
    parser.add_argument("--text-prompt", type=str, default=None, help="Optional referring/text prompt.")
    parser.add_argument(
        "--target-classes",
        type=str,
        default=None,
        help="Optional comma-separated class names for panoptic/detection models (e.g. 'person,sky,coat').",
    )
    parser.add_argument("--points-json", type=str, default=None, help="Optional JSON [[x, y], ...] coordinates.")
    parser.add_argument("--boxes-json", type=str, default=None, help="Optional JSON [[x1, y1, x2, y2], ...] boxes.")
    parser.add_argument("--alpha-threshold", type=float, default=0.5, help="Alpha threshold for matting models.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path or HuggingFace ID.")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu').")
    parser.add_argument(
        "--compliance-output-dir",
        type=str,
        default="output",
        help="Directory for compliance_report.json logging (pass empty string to skip).",
    )
    parser.add_argument(
        "--allow-non-commercial-benchmark",
        action="store_true",
        help="Unlock sam3_image / sam31_image benchmark-only segmenters and log run non-compliant.",
    )
    return parser


def _parse_json_arg(raw: str | None) -> Any:
    if not raw or not str(raw).strip():
        return None
    p = Path(raw)
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return json.loads(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_segmenters:
        items = [
            {
                "name": s.name,
                "label": s.label,
                "component_id": s.component_id,
                "category": s.category,
                "license": s.license,
                "commercial_tier": s.commercial_tier,
                "supports_text_prompts": s.supports_text_prompts,
                "supports_point_box_prompts": s.supports_point_box_prompts,
                "supports_target_classes": s.supports_target_classes,
                "available": DEFAULT_IMAGE_SEGMENTER_REGISTRY.is_available(s.name),
            }
            for s in DEFAULT_IMAGE_SEGMENTER_REGISTRY.list_specs()
        ]
        print(json.dumps({"image_segmenters": items}, indent=2))
        return 0

    if not args.image:
        parser.error("--image is required unless --list-segmenters is set.")

    img_path = Path(args.image)
    out_path = (
        Path(args.output)
        if str(args.output or "").strip()
        else Path("intermediate") / "keyframe_masks" / f"{img_path.stem}_{args.segmenter}.png"
    )
    compliance_dir = Path(args.compliance_output_dir) if str(args.compliance_output_dir or "").strip() else None

    try:
        res = run_keyframe_segmentation(
            image=img_path,
            output_mask_path=out_path,
            segmenter=args.segmenter,
            text_prompt=args.text_prompt,
            points=_parse_json_arg(args.points_json),
            boxes=_parse_json_arg(args.boxes_json),
            target_classes=normalize_target_classes(args.target_classes) or None,
            alpha_threshold=args.alpha_threshold,
            checkpoint_path=args.checkpoint,
            device=args.device,
            allow_non_commercial_benchmark=bool(args.allow_non_commercial_benchmark),
            compliance_output_dir=compliance_dir,
        )
    except CommercialPolicyError as exc:
        print(f"Commercial Policy Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Keyframe Segmentation Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"ok": True, **res}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
