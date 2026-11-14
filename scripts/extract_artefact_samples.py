"""Isolated ARTeFACT dataset extractor for offline benchmark evaluation.

Runs strictly inside an isolated Python environment (such as I:\\DART\\.venv)
that contains pyarrow, pandas, and Pillow. Never imported directly into the
commercial ARP production runtime or ComfyUI.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys

from PIL import Image
import numpy as np
import pyarrow.parquet as pq


TARGET_CLASSES = {
    3: "Dust",
    4: "Scratch",
    5: "Hair",
    6: "Dirt",
}


def extract_samples(
    parquet_path: Path,
    output_dir: Path,
    *,
    max_samples: int = 10,
    require_target_classes: bool = True,
) -> dict:
    """Extract sample images and raw annotation arrays from an ARTeFACT Parquet shard."""
    parquet_path = Path(parquet_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = output_dir / "raw_images"
    annotations_dir = output_dir / "raw_annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(parquet_path))
    num_rows = table.num_rows

    extracted = []
    for row_idx in range(num_rows):
        if len(extracted) >= max_samples:
            break

        row_dict = table.slice(row_idx, 1).to_pylist()[0]
        sample_id = str(row_dict.get("id") or f"sample_{row_idx:04d}")
        material = str(row_dict.get("material") or "unknown")

        ann_bytes = row_dict.get("annotation", {}).get("bytes")
        img_bytes = row_dict.get("image", {}).get("bytes")
        if not ann_bytes or not img_bytes:
            continue

        ann_img = Image.open(io.BytesIO(ann_bytes))
        ann_arr = np.array(ann_img)

        unique_classes = sorted(int(c) for c in np.unique(ann_arr))
        target_hits = sorted(set(unique_classes) & set(TARGET_CLASSES))
        if require_target_classes and not target_hits:
            continue

        # Save raw image
        raw_img_path = images_dir / f"{sample_id}.png"
        raw_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        raw_img.save(str(raw_img_path), format="PNG")

        # Save raw uint8 annotation
        raw_ann_path = annotations_dir / f"{sample_id}_ann.png"
        # Clip to uint8 if needed
        ann_u8 = np.clip(ann_arr, 0, 255).astype(np.uint8)
        Image.fromarray(ann_u8).save(str(raw_ann_path), format="PNG")

        extracted.append(
            {
                "id": sample_id,
                "row_index": row_idx,
                "material": material,
                "width": raw_img.width,
                "height": raw_img.height,
                "image_path": str(raw_img_path),
                "annotation_path": str(raw_ann_path),
                "unique_classes": unique_classes,
                "target_classes": target_hits,
            }
        )

    manifest_path = output_dir / "extracted_manifest.json"
    manifest_data = {
        "parquet_path": str(parquet_path),
        "total_extracted": len(extracted),
        "samples": extracted,
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    return manifest_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--no-filter-target", action="store_true")
    args = parser.parse_args(argv)

    res = extract_samples(
        args.parquet_path,
        args.output_dir,
        max_samples=args.max_samples,
        require_target_classes=not args.no_filter_target,
    )
    print(f"Extracted {res['total_extracted']} ARTeFACT samples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
