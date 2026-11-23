#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.rle_masks import (
    VALID_ENTITY_TYPES,
    MaskObjectMetadata,
    ShotMaskArchive,
    decode_mask_rle,
    encode_mask_rle,
    load_shot_masks_npz,
    save_shot_masks_npz,
)

__all__ = [
    "VALID_ENTITY_TYPES",
    "MaskObjectMetadata",
    "ShotMaskArchive",
    "decode_mask_rle",
    "encode_mask_rle",
    "load_shot_masks_npz",
    "save_shot_masks_npz",
    "main",
]


def print_archive_info(archive: ShotMaskArchive, as_json: bool = False) -> None:
    """Print metadata and metrics for a ShotMaskArchive."""
    ratio = archive.compression_ratio()
    stored_masks = archive.stored_mask_count()
    total_slots = len(archive.objects) * archive.frame_count

    info_dict = {
        "shot_id": archive.shot_id,
        "frame_count": archive.frame_count,
        "height": archive.height,
        "width": archive.width,
        "stored_mask_count": stored_masks,
        "total_slots": total_slots,
        "compression_ratio": round(ratio, 2),
        "objects": [obj.to_dict() for obj in archive.objects],
    }

    if as_json:
        print(json.dumps(info_dict, indent=2))
        return

    print("=" * 64)
    print(f"Shot Mask Archive: {archive.shot_id}")
    print("=" * 64)
    print(f"Resolution:        {archive.width}x{archive.height} (WxH)")
    print(f"Frame Count:       {archive.frame_count}")
    print(f"Stored Masks:      {stored_masks} / {total_slots} slots")
    print(f"Compression Ratio: {ratio:.1f}x vs raw uint8")
    print(f"Objects ({len(archive.objects)}):")
    for obj in archive.objects:
        print(
            f"  - [ID {obj.obj_id}] '{obj.name}' "
            f"(type: {obj.entity_type}, entity_id: '{obj.entity_id}', variant_id: '{obj.variant_id}')"
        )
    print("=" * 64)


def extract_archive_masks(
    archive: ShotMaskArchive,
    output_dir: Path,
    target_obj_id: int | None = None,
    target_frame: int | None = None,
) -> int:
    """Extract individual mask frames to PNG images."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    frame_range = (
        range(target_frame, target_frame + 1)
        if target_frame is not None
        else range(archive.frame_count)
    )
    objects_to_extract = (
        [obj for obj in archive.objects if obj.obj_id == target_obj_id]
        if target_obj_id is not None
        else archive.objects
    )

    for obj in objects_to_extract:
        for frame_idx in frame_range:
            if not (0 <= frame_idx < archive.frame_count):
                continue
            mask = archive.get_mask(frame_idx, obj.obj_id)
            if mask is None:
                continue

            img = Image.fromarray(mask)
            filename = f"{archive.shot_id}_obj{obj.obj_id:02d}_frame{frame_idx:04d}.png"
            img.save(output_dir / filename)
            extracted_count += 1

    return extracted_count


def build_parser() -> argparse.ArgumentParser:
    """Construct CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Inspect and extract Run-Length-Encoded (RLE) .masks.npz archives."
    )
    parser.add_argument(
        "archive",
        type=Path,
        nargs="?",
        help="Path to .masks.npz archive file.",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Display archive metadata, object hierarchy, and compression metrics.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Format --info output as structured JSON.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract stored masks as 8-bit grayscale PNG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("extracted_masks"),
        help="Destination directory for extracted PNG masks (defaults to ./extracted_masks).",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=None,
        help="Filter extraction or inspection by specific object ID.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Filter extraction by specific frame index.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.archive is None:
        parser.print_help()
        return 1

    archive_path = Path(args.archive)
    if not archive_path.is_file():
        print(f"Error: Archive file not found: {archive_path}", file=sys.stderr)
        return 1

    try:
        archive = ShotMaskArchive.load_npz(archive_path)
    except Exception as exc:
        print(f"Error loading archive {archive_path}: {exc}", file=sys.stderr)
        return 1

    if args.extract:
        count = extract_archive_masks(
            archive=archive,
            output_dir=args.output_dir,
            target_obj_id=args.obj_id,
            target_frame=args.frame,
        )
        print(f"Successfully extracted {count} mask(s) to '{args.output_dir}'.")
        return 0

    # Default action: show info
    print_archive_info(archive, as_json=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
