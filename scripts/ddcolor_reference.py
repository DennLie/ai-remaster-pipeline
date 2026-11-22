#!/usr/bin/env python3
"""DDColor reference draft generator.

Provides fast, deterministic still reference colorization using DDColor (Apache-2.0).
Preserves original grayscale luminance L in Lab color space ("The original pixels win"),
predicting only chromaticity a* and b*.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

# Ensure repository root is on sys.path for internal imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.common import ROOT, resolve_path, root_relative
except ImportError:
    ROOT = REPO_ROOT

    def resolve_path(path: Path | str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (ROOT / p).resolve()

    def root_relative(path: Path | str) -> str:
        try:
            return str(Path(path).resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")

try:
    from ai_remaster_gui.compliance import append_compliance_record
except ImportError:
    append_compliance_record = None  # type: ignore[assignment]


DEFAULT_MODEL_PATH = ROOT / "models" / "ddcolor" / "ddcolor_paper_tiny.pth"
DEFAULT_OUTPUT_ROOT = ROOT / "intermediate" / "outpainted_references_color"


class DDColorModel(nn.Module):
    """PyTorch neural architecture for DDColor chromaticity prediction.

    Dual-decoder / multi-scale architecture representation predicting
    chromaticity (a*, b*) from grayscale/luminance input.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 2) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        ab = self.decoder(feat)
        return ab


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve target torch.device."""
    dev = str(device or "auto").lower().strip()
    if dev in ("cuda", "gpu"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: Path | str, device: str = "auto") -> nn.Module:
    """Load DDColor checkpoint from model_path onto target device."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"DDColor model weights not found at: {path}. "
            f"Model weights are required for normal execution. Pass --dry-run for synthetic mock mode."
        )
    target_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=target_device)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("params") or checkpoint.get("state_dict") or checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint dictionary at {path} does not contain valid state dict parameters.")
        model = DDColorModel()
        total_keys = len(list(model.state_dict().keys()))
        incompatible = model.load_state_dict(state_dict, strict=False)
        if len(incompatible.missing_keys) == total_keys:
            raise ValueError(f"Checkpoint at {path} does not match DDColorModel architecture (0 matching keys).")
    else:
        raise ValueError(
            f"Checkpoint at {path} has unsupported format: expected nn.Module or dict, got {type(checkpoint).__name__}"
        )
    model.to(target_device)
    model.eval()
    return model


def _synthetic_dry_run_chroma(
    orig_l: np.ndarray,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deterministic synthetic chromaticity (a*, b*) preserving luminance."""
    h, w = orig_l.shape[:2]
    rng_seed = seed if seed is not None and seed >= 0 else 42
    rng = np.random.RandomState(rng_seed)
    y = np.linspace(-1.0, 1.0, h)[:, None]
    x = np.linspace(-1.0, 1.0, w)[None, :]
    l_norm = (orig_l.astype(np.float32) - 128.0) / 128.0

    # Seed-dependent deterministic phase and gain modulation
    phase_a = float(rng.uniform(-np.pi, np.pi))
    phase_b = float(rng.uniform(-np.pi, np.pi))
    gain_a = float(rng.uniform(8.0, 16.0))
    gain_b = float(rng.uniform(10.0, 20.0))

    # Natural tone offsets (mild warm tint, modulated by coordinates, seed, and luma)
    # Gently scale chromaticity towards neutral near pure black and pure white to stay within sRGB gamut
    chroma_scale = np.clip(orig_l.astype(np.float32) / 24.0, 0.0, 1.0) * np.clip(
        (255.0 - orig_l.astype(np.float32)) / 24.0, 0.0, 1.0
    )
    base_a = 128.0 + (gain_a * np.sin(1.5 * x + 0.5 * y + phase_a) + 6.0 * l_norm) * chroma_scale
    base_b = 128.0 + (gain_b * np.cos(1.0 * x - 1.2 * y + phase_b) + 8.0 * l_norm) * chroma_scale

    a = np.clip(base_a, 80, 175).astype(np.uint8)
    b = np.clip(base_b, 80, 175).astype(np.uint8)
    return a, b


def colorize_frame(
    image: np.ndarray,
    model_path: Path | str,
    device: str = "auto",
    dry_run: bool = False,
    input_size: int = 512,
    seed: int | None = None,
    model: nn.Module | None = None,
) -> np.ndarray:
    """Colorize a single image frame using DDColor or synthetic dry-run.

    Preserves original grayscale luminance L in Lab space ("The original pixels win"),
    predicting only chromaticity a* and b*.

    Args:
        image: Input image as numpy array (RGB or grayscale, uint8).
        model_path: Path to DDColor checkpoint (.pth).
        device: "auto" | "cuda" | "cpu".
        dry_run: If True, uses deterministic synthetic colorization without weights/GPU.
        input_size: Dimension to resize image for model inference (default 512).
        seed: Optional RNG seed for deterministic colorization.
        model: Optional pre-loaded PyTorch model.

    Returns:
        RGB numpy array of shape (H, W, 3), uint8.
    """
    model_file = Path(model_path)

    # 1. Extract original luminance L in Lab space
    if image.ndim == 2:
        rgb_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        orig_l = lab[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 1:
        rgb_img = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        orig_l = lab[:, :, 0]
    elif image.ndim == 3 and image.shape[2] >= 3:
        lab = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2LAB)
        orig_l = lab[:, :, 0]
    else:
        raise ValueError(f"Unsupported image shape for colorization: {image.shape}")

    orig_h, orig_w = orig_l.shape[:2]

    # 2. Dry-run mode: deterministic synthetic colorization
    if dry_run:
        a, b = _synthetic_dry_run_chroma(orig_l, seed=seed)
        lab_out = np.dstack([orig_l, a, b])
        return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)

    # 3. Normal mode: MUST fail closed if weights are missing
    if not model_file.is_file():
        raise FileNotFoundError(
            f"DDColor model checkpoint not found at: {model_file}. "
            f"Model weights are required for normal execution. Pass --dry-run for synthetic mock mode."
        )

    # 4. Neural inference
    target_device = resolve_device(device)
    if model is None:
        model = load_model(model_file, device=device)

    # Prepare normalized model input
    resized_l = cv2.resize(orig_l, (input_size, input_size), interpolation=cv2.INTER_AREA)
    l_tensor = torch.from_numpy(resized_l).float() / 255.0
    l_tensor = l_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(target_device)

    with torch.no_grad():
        pred_ab = model(l_tensor)

    # Upsample predicted chromaticity back to original frame dimensions
    if pred_ab.shape[2:] != (orig_h, orig_w):
        pred_ab = nn.functional.interpolate(
            pred_ab, size=(orig_h, orig_w), mode="bilinear", align_corners=False
        )

    ab_np = pred_ab.squeeze(0).cpu().numpy()  # shape (2, H, W)

    # Handle common output ranges: [-1, 1], [-128, 127], or [0, 255]
    ab_min, ab_max = float(ab_np.min()), float(ab_np.max())
    if -1.5 <= ab_min and ab_max <= 1.5:
        a_channel = np.clip(ab_np[0] * 128.0 + 128.0, 0, 255).astype(np.uint8)
        b_channel = np.clip(ab_np[1] * 128.0 + 128.0, 0, 255).astype(np.uint8)
    elif -130.0 <= ab_min and ab_max <= 130.0:
        a_channel = np.clip(ab_np[0] + 128.0, 0, 255).astype(np.uint8)
        b_channel = np.clip(ab_np[1] + 128.0, 0, 255).astype(np.uint8)
    else:
        a_channel = np.clip(ab_np[0], 0, 255).astype(np.uint8)
        b_channel = np.clip(ab_np[1], 0, 255).astype(np.uint8)

    lab_out = np.dstack([orig_l, a_channel, b_channel])
    return cv2.cvtColor(lab_out, cv2.COLOR_LAB2RGB)


def read_manifest_lines(manifest_path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Read manifest preserving leading comments and returning headers and row dicts."""
    comments: list[str] = []
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        csv_lines: list[str] = []
        for line in f:
            if line.startswith("#"):
                comments.append(line)
            else:
                csv_lines.append(line)

    if csv_lines:
        reader = csv.DictReader(csv_lines)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    return comments, fieldnames, rows


def write_manifest_lines(
    manifest_path: Path,
    comments: list[str],
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    """Write updated manifest preserving comments via atomic temp file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        for c in comments:
            f.write(c if c.endswith("\n") else c + "\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    tmp_path.replace(manifest_path)


def process_manifest(
    manifest_path: Path,
    output_manifest_path: Path | None,
    output_root: Path,
    model_path: Path,
    device: str = "auto",
    dry_run: bool = False,
    input_size: int = 512,
    seed: int | None = None,
    limit: int | None = None,
) -> int:
    """Process all reference rows in a manifest and update color_reference column."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    comments, fieldnames, rows = read_manifest_lines(manifest_path)
    if "color_reference" not in fieldnames:
        fieldnames.append("color_reference")

    model: nn.Module | None = None
    if not dry_run:
        model = load_model(model_path, device=device)

    count = 0
    for idx, row in enumerate(rows):
        if limit is not None and count >= limit:
            break
        if row.get("enabled", "true").strip().lower() in {"false", "0", "no", "off"}:
            continue

        source_rel = row.get("source_reference") or row.get("reference") or ""
        if not source_rel:
            continue
        source_path = resolve_path(source_rel)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source reference still not found: {source_path}")

        # Destination color reference path
        if output_root:
            target_path = resolve_path(output_root) / source_path.name
        elif row.get("color_reference"):
            target_path = resolve_path(row["color_reference"])
        else:
            target_path = DEFAULT_OUTPUT_ROOT / source_path.name

        with Image.open(source_path) as im:
            rgb_in = np.asarray(im.convert("RGB"))

        colorized_rgb = colorize_frame(
            rgb_in,
            model_path=model_path,
            device=device,
            dry_run=dry_run,
            input_size=input_size,
            seed=seed,
            model=model,
        )

        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colorized_rgb).save(target_path)
        row["color_reference"] = root_relative(target_path)
        count += 1
        print(f"[{count}] Colorized reference: {source_path.name} -> {target_path}")

    dst_manifest = output_manifest_path or manifest_path
    write_manifest_lines(dst_manifest, comments, fieldnames, rows)
    print(f"Updated manifest saved to: {dst_manifest} ({count} stills colorized)")
    return count


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DDColor fast deterministic reference draft generator (Apache-2.0)."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Path to shot references CSV manifest.",
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        help="Path to single input image to colorize.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (with --input-image) or output manifest path (with --manifest).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory for colorized reference stills (default: intermediate/outpainted_references_color).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to DDColor checkpoint (.pth). Defaults to models/ddcolor/ddcolor_paper_tiny.pth.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Execution device: auto | cuda | cpu (default: auto).",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=512,
        help="Inference image resolution (default: 512).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Deterministic synthetic colorization for testing and CI without GPU/weights.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic colorization.",
    )
    parser.add_argument(
        "--compliance-output-dir",
        type=Path,
        default=None,
        help="Output directory to record run into compliance_report.json via append_compliance_record.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional integer row limit for manifest processing.",
    )
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.manifest and not args.input_image:
        parser.error("Either --manifest or --input-image must be provided.")

    model_path = resolve_path(args.model_path)

    # Single-image colorization mode
    if args.input_image:
        input_image = resolve_path(args.input_image)
        if not input_image.is_file():
            raise FileNotFoundError(f"Input image not found: {input_image}")

        if args.output:
            output_image = resolve_path(args.output)
        elif args.output_root:
            output_image = resolve_path(args.output_root) / input_image.name
        else:
            output_image = input_image.parent / f"{input_image.stem}_colorized{input_image.suffix}"

        with Image.open(input_image) as im:
            rgb_in = np.asarray(im.convert("RGB"))

        colorized_rgb = colorize_frame(
            rgb_in,
            model_path=model_path,
            device=args.device,
            dry_run=args.dry_run,
            input_size=args.input_size,
            seed=args.seed,
        )

        output_image.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(colorized_rgb).save(output_image)
        print(f"Colorized still saved to: {output_image}")

    # Manifest processing mode
    if args.manifest:
        manifest_path = resolve_path(args.manifest)
        output_manifest = resolve_path(args.output) if args.output else None
        output_root = resolve_path(args.output_root) if args.output_root else DEFAULT_OUTPUT_ROOT

        process_manifest(
            manifest_path=manifest_path,
            output_manifest_path=output_manifest,
            output_root=output_root,
            model_path=model_path,
            device=args.device,
            dry_run=args.dry_run,
            input_size=args.input_size,
            seed=args.seed,
            limit=args.limit,
        )

    # Compliance reporting
    if args.compliance_output_dir and append_compliance_record is not None:
        comp_dir = resolve_path(args.compliance_output_dir)
        append_compliance_record(
            output_dir=comp_dir,
            stage="references",
            models=[
                {
                    "model_id": "ddcolor",
                    "component_type": "checkpoint",
                    "file_path": str(model_path) if model_path.is_file() else None,
                    "source_url": "https://github.com/piddnad/DDColor",
                    "license": "Apache-2.0",
                    "status": "allowed_osi",
                    "reason": "DDColor Apache-2.0 still reference draft generator",
                }
            ],
            settings={"references": {"method": "ddcolor"}},
            seeds={"references": args.seed} if args.seed is not None else None,
        )
        print(f"Recorded reference run to compliance report in: {comp_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
