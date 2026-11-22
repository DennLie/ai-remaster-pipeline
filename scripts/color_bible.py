#!/usr/bin/env python3
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

from ai_remaster_gui.color_bible_service import ColorBibleService
from ai_remaster_gui.color_science import (
    compute_lab_percentiles,
    extract_rgb_palette,
    load_image_rgb,
    load_mask,
)


def parse_ranges(ranges_str: str) -> list[tuple[int, int]]:
    """Parse comma-separated shot ranges e.g. '12-18,410-435' or '5' into (start, end) tuples."""
    if not ranges_str or not ranges_str.strip():
        raise ValueError("Ranges string must not be empty.")
    results: list[tuple[int, int]] = []
    for raw_part in ranges_str.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            parts = part.split("-", 1)
            try:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
            except ValueError:
                raise ValueError(f"Invalid range format '{part}'. Expected integer range e.g. '12-18'.")
        else:
            try:
                start = int(part)
                end = start
            except ValueError:
                raise ValueError(f"Invalid shot number '{part}'. Expected integer.")
        if start > end:
            raise ValueError(f"Invalid range '{part}': start ({start}) > end ({end}).")
        results.append((start, end))
    if not results:
        raise ValueError(f"No valid shot ranges could be parsed from '{ranges_str}'.")
    return results


def handle_init_db(args: argparse.Namespace) -> int:
    with ColorBibleService(db_path=args.db) as service:
        db_path = service.repo.db_path
        print(f"Initialized Color Bible database at: {db_path}")
    return 0


def handle_sync_manifest(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with ColorBibleService(db_path=args.db) as service:
        res = service.sync_manifest(
            film_id=args.film_id,
            manifest_path=manifest_path,
            title=args.title,
        )
        shots_synced = res.get("shots_synced", 0)
        print(f"Synced {shots_synced} shot(s) for film '{args.film_id}' from manifest '{manifest_path}'.")
    return 0


def handle_add_character(args: argparse.Namespace) -> int:
    with ColorBibleService(db_path=args.db) as service:
        char = service.create_character(
            film_id=args.film_id,
            name=args.name,
            notes=args.notes or "",
        )
        print(f"Created character: {char.character_id}")
    return 0


def handle_add_variant(args: argparse.Namespace) -> int:
    with ColorBibleService(db_path=args.db) as service:
        variant = service.create_appearance_variant(
            character_id=args.character_id,
            costume_id=args.costume_id,
            hair_profile=args.hair_profile or "",
            skin_reference=args.skin_reference or "",
            lighting_context=args.lighting_context or "",
            notes=args.notes or "",
        )
        print(f"Created variant: {variant.variant_id}")
    return 0


def handle_assign_range(args: argparse.Namespace) -> int:
    ranges = parse_ranges(args.ranges)
    with ColorBibleService(db_path=args.db) as service:
        assignments = service.assign_entity_ranges(
            film_id=args.film_id,
            entity_type=args.entity_type,
            entity_id=args.entity_id,
            ranges=ranges,
            variant_id=args.variant_id,
            confidence=args.confidence,
            approval_status=args.approval_status,
            provenance=args.provenance,
        )
        print(f"Created {len(assignments)} assignment(s).")
    return 0


def handle_extract_palette(args: argparse.Namespace) -> int:
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mask_path = Path(args.mask) if args.mask else None
    if mask_path and not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    img_rgb = load_image_rgb(image_path)
    mask = load_mask(mask_path) if mask_path else None

    lab_stats = compute_lab_percentiles(img_rgb, mask)
    rgb_palette = extract_rgb_palette(img_rgb, mask, num_colors=args.num_colors)

    if args.json:
        payload = {
            "image": str(image_path),
            "mask": str(mask_path) if mask_path else None,
            "lab_percentiles": lab_stats,
            "rgb_palette": rgb_palette,
        }
        print(json.dumps(payload, indent=2))
    else:
        print("CIE-LAB Percentiles (L*, a*, b*):")
        print(f"  L*: {lab_stats.get('L', [])}")
        print(f"  a*: {lab_stats.get('a', [])}")
        print(f"  b*: {lab_stats.get('b', [])}")
        print(f"Dominant RGB Palette ({len(rgb_palette)} swatches):")
        for idx, entry in enumerate(rgb_palette, 1):
            hex_col = entry.get("hex", "")
            rgb = entry.get("rgb", [])
            pct = entry.get("percentage", 0.0) * 100.0
            print(f"  {idx}. {hex_col} RGB{tuple(rgb)} ({pct:.1f}%)")
    return 0


def handle_register_reference(args: argparse.Namespace) -> int:
    keyframe_path = Path(args.keyframe)
    if not keyframe_path.is_file():
        raise FileNotFoundError(f"Keyframe image not found: {keyframe_path}")

    mask_path = Path(args.mask) if args.mask else None
    if mask_path and not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    with ColorBibleService(db_path=args.db) as service:
        ref = service.register_reference_image(
            film_id=args.film_id,
            entity_type=args.entity_type,
            entity_id=args.entity_id,
            keyframe_path=keyframe_path,
            mask_uri=mask_path,
            variant_id=args.variant_id,
            provenance=args.provenance,
            source_shot_id=args.source_shot_id,
            auto_approve=args.auto_approve,
            approved_by=args.approved_by,
        )
        print(f"Created reference: {ref.ref_id}")
    return 0


def handle_approve_ref(args: argparse.Namespace) -> int:
    with ColorBibleService(db_path=args.db) as service:
        ref = service.approve_reference(
            ref_id=args.ref_id,
            approved_by=args.approved_by or "colorist",
        )
        print(f"Approved reference: {ref.ref_id}")
    return 0


def handle_reject_ref(args: argparse.Namespace) -> int:
    with ColorBibleService(db_path=args.db) as service:
        ref = service.reject_reference(ref_id=args.ref_id)
        print(f"Rejected reference: {ref.ref_id}")
    return 0


def handle_snapshot_hash(args: argparse.Namespace) -> int:
    with ColorBibleService(db_path=args.db) as service:
        hash_val = service.get_snapshot_hash(film_id=args.film_id)
        print(hash_val)
    return 0


def handle_list_assignments(args: argparse.Namespace) -> int:
    with ColorBibleService(db_path=args.db) as service:
        if args.shot_index is not None:
            assignments_data = service.get_shot_palette_assignments(
                film_id=args.film_id,
                shot_index=args.shot_index,
            )
            if args.json:
                print(json.dumps(assignments_data, indent=2))
            else:
                print(
                    f"Assignments for film '{args.film_id}' at shot index {args.shot_index}: "
                    f"({len(assignments_data)} active)"
                )
                for item in assignments_data:
                    asgn = item["assignment"]
                    ent_type = item["entity_type"]
                    ent_id = item["entity_id"]
                    var_id = item.get("variant_id") or "None"
                    status = asgn.get("approval_status", "")
                    conf = asgn.get("confidence", 1.0)
                    print(
                        f"  - [{asgn.get('assignment_id')}] Type: {ent_type} | ID: {ent_id} | "
                        f"Variant: {var_id} | Status: {status} | Conf: {conf}"
                    )
                    refs = item.get("color_references", [])
                    if refs:
                        print(f"    Approved References: {len(refs)}")
                        for r in refs:
                            print(f"      * Ref: {r.get('ref_id')} (v{r.get('version')}) - {r.get('provenance')}")
        else:
            assignments = service.list_assignments(film_id=args.film_id)
            if args.json:
                print(json.dumps([a.to_dict() for a in assignments], indent=2))
            else:
                print(f"Assignments for film '{args.film_id}': ({len(assignments)} total)")
                for a in assignments:
                    print(
                        f"  - [{a.assignment_id}] Range: {a.valid_shot_range_start}-{a.valid_shot_range_end} | "
                        f"Type: {a.entity_type} | ID: {a.entity_id} | Variant: {a.variant_id or 'None'} | "
                        f"Status: {a.approval_status} | Conf: {a.confidence}"
                    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build and return top-level argument parser with all Color Bible subcommands."""
    db_parent = argparse.ArgumentParser(add_help=False)
    db_parent.add_argument(
        "--db",
        "-d",
        default=None,
        help="Path to Color Bible SQLite database (defaults to ARP_COLOR_BIBLE_DB or manifests/color_bible.db).",
    )

    parser = argparse.ArgumentParser(
        prog="color_bible.py",
        description="Color Bible CLI for character, wardrobe, and palette management.",
        parents=[db_parent],
    )
    subparsers = parser.add_subparsers(dest="command", help="Color Bible subcommands")

    # init-db
    p_init = subparsers.add_parser(
        "init-db",
        parents=[db_parent],
        help="Initialize Color Bible database schema.",
    )
    p_init.set_defaults(func=handle_init_db)

    # sync-manifest
    p_sync = subparsers.add_parser(
        "sync-manifest",
        parents=[db_parent],
        help="Synchronize an ARP shot manifest into database.",
    )
    p_sync.add_argument("--film-id", required=True, help="Canonical film identifier.")
    p_sync.add_argument("--manifest", required=True, help="Path to shot manifest CSV file.")
    p_sync.add_argument("--title", default=None, help="Optional human-readable film title.")
    p_sync.set_defaults(func=handle_sync_manifest)

    # add-character
    p_add_char = subparsers.add_parser(
        "add-character",
        parents=[db_parent],
        help="Add a character to a film.",
    )
    p_add_char.add_argument("--film-id", required=True, help="Canonical film identifier.")
    p_add_char.add_argument("--name", required=True, help="Character display name.")
    p_add_char.add_argument("--notes", default="", help="Optional notes on the character.")
    p_add_char.set_defaults(func=handle_add_character)

    # add-variant
    p_add_var = subparsers.add_parser(
        "add-variant",
        parents=[db_parent],
        help="Add an appearance variant to a character.",
    )
    p_add_var.add_argument("--character-id", required=True, help="Character identifier.")
    p_add_var.add_argument("--costume-id", required=True, help="Costume / wardrobe identifier.")
    p_add_var.add_argument("--hair-profile", default="", help="Hair color and style profile.")
    p_add_var.add_argument("--skin-reference", default="", help="Skin tone reference metadata.")
    p_add_var.add_argument("--lighting-context", default="", help="Lighting setup or context.")
    p_add_var.add_argument("--notes", default="", help="Optional notes on the variant.")
    p_add_var.set_defaults(func=handle_add_variant)

    # assign-range
    p_assign = subparsers.add_parser(
        "assign-range",
        parents=[db_parent],
        help="Assign an entity across shot ranges.",
    )
    p_assign.add_argument("--film-id", required=True, help="Canonical film identifier.")
    p_assign.add_argument(
        "--entity-type",
        required=True,
        choices=["character", "object", "location"],
        help="Entity type.",
    )
    p_assign.add_argument("--entity-id", required=True, help="Entity ID.")
    p_assign.add_argument("--variant-id", default=None, help="Optional appearance variant ID.")
    p_assign.add_argument(
        "--ranges",
        required=True,
        help="Comma-separated shot ranges e.g. '12-18,410-435' or '5'.",
    )
    p_assign.add_argument(
        "--confidence",
        type=float,
        default=1.0,
        help="Confidence score [0.0, 1.0]. Default 1.0.",
    )
    p_assign.add_argument(
        "--approval-status",
        default="approved",
        choices=["draft", "pending_review", "approved", "rejected"],
        help="Approval status.",
    )
    p_assign.add_argument(
        "--provenance",
        default="manual",
        help="Assignment provenance (e.g. 'manual', 'script_import').",
    )
    p_assign.set_defaults(func=handle_assign_range)

    # extract-palette
    p_palette = subparsers.add_parser(
        "extract-palette",
        parents=[db_parent],
        help="Extract CIE-LAB percentiles and RGB palette from an image.",
    )
    p_palette.add_argument("--image", required=True, help="Path to input image.")
    p_palette.add_argument("--mask", default=None, help="Optional path to binary mask image.")
    p_palette.add_argument(
        "--num-colors",
        type=int,
        default=6,
        help="Number of dominant palette colors (default: 6).",
    )
    p_palette.add_argument("--json", action="store_true", help="Output results in JSON format.")
    p_palette.set_defaults(func=handle_extract_palette)

    # register-reference
    p_reg_ref = subparsers.add_parser(
        "register-reference",
        parents=[db_parent],
        help="Register a color reference image.",
    )
    p_reg_ref.add_argument("--film-id", required=True, help="Canonical film identifier.")
    p_reg_ref.add_argument(
        "--entity-type",
        required=True,
        choices=["character", "object", "location"],
        help="Entity type.",
    )
    p_reg_ref.add_argument("--entity-id", required=True, help="Entity ID.")
    p_reg_ref.add_argument("--variant-id", default=None, help="Optional appearance variant ID.")
    p_reg_ref.add_argument("--keyframe", required=True, help="Path to keyframe image.")
    p_reg_ref.add_argument("--mask", default=None, help="Optional path to binary mask image.")
    p_reg_ref.add_argument(
        "--provenance",
        default="manual",
        choices=["historical_research", "ddcolor", "qwen2511", "manual"],
        help="Reference provenance.",
    )
    p_reg_ref.add_argument("--source-shot-id", default=None, help="Optional source shot ID.")
    p_reg_ref.add_argument(
        "--auto-approve",
        action="store_true",
        help="Automatically mark reference as approved.",
    )
    p_reg_ref.add_argument("--approved-by", default=None, help="Optional approver username.")
    p_reg_ref.set_defaults(func=handle_register_reference)

    # approve-ref
    p_app_ref = subparsers.add_parser(
        "approve-ref",
        parents=[db_parent],
        help="Approve a color reference.",
    )
    p_app_ref.add_argument("--ref-id", required=True, help="Color reference ID.")
    p_app_ref.add_argument(
        "--approved-by",
        default="colorist",
        help="Approver username (default: 'colorist').",
    )
    p_app_ref.set_defaults(func=handle_approve_ref)

    # reject-ref
    p_rej_ref = subparsers.add_parser(
        "reject-ref",
        parents=[db_parent],
        help="Reject a color reference.",
    )
    p_rej_ref.add_argument("--ref-id", required=True, help="Color reference ID.")
    p_rej_ref.set_defaults(func=handle_reject_ref)

    # snapshot-hash
    p_hash = subparsers.add_parser(
        "snapshot-hash",
        parents=[db_parent],
        help="Compute canonical SHA-256 snapshot hash for a film.",
    )
    p_hash.add_argument("--film-id", required=True, help="Canonical film identifier.")
    p_hash.set_defaults(func=handle_snapshot_hash)

    # list-assignments
    p_list = subparsers.add_parser(
        "list-assignments",
        parents=[db_parent],
        help="List entity assignments for a film.",
    )
    p_list.add_argument("--film-id", required=True, help="Canonical film identifier.")
    p_list.add_argument(
        "--shot-index",
        type=int,
        default=None,
        help="Optional shot index to query active assignments.",
    )
    p_list.add_argument("--json", action="store_true", help="Output results in JSON format.")
    p_list.set_defaults(func=handle_list_assignments)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Color Bible CLI."""
    parser = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help()
        return 1

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
