"""CPU-only batch renderer and source-frame-exact comparison grid for Remove Noise.

Invoked by ARP using validated absolute source/output paths, never from arbitrary
browser file arguments. Creates ZIPs atomically; original media is read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path

from .config import ROOT, SCRIPTS
from .frame_differences import _validate_pair
from .frame_inspector import _decode, _metadata
from .paths import resolve, rel

METHODS = ("hqdn3d", "nlmeans", "bm3d")
PRESETS = ("gentle", "balanced", "strong")
VARIANTS = tuple((m, p) for m in METHODS for p in PRESETS)
EXPORT_ROOT = ROOT / "output" / "denoise_comparisons"
GRID_VERSION = 1
TILE_W, TILE_H = 320, 240
ROW_H, TOP_H, PAD = 298, 70, 12


def parse_frames(text: str, count: int) -> list[int]:
    """Input is one-based frame numbers, or empty for 10%, 50%, 90% of clip."""
    if count < 1:
        raise ValueError("No source frames.")
    if not str(text).strip():
        return sorted(set(min(count - 1, round((count - 1) * position))
                          for position in (0.10, 0.50, 0.90)))
    parts = [part.strip() for part in str(text).replace(";", ",").split(",")]
    if not 1 <= len(parts) <= 5 or any(not item.isdecimal() for item in parts):
        raise ValueError("Enter 1–5 frame numbers separated by commas.")
    frames = [int(value) - 1 for value in parts]
    if any(frame < 0 or frame >= count for frame in frames) or len(set(frames)) != len(frames):
        raise ValueError(f"Frame indices must be unique and between 1 and {count}.")
    return sorted(frames)


def validate_variants(rows: list[dict]) -> None:
    if not isinstance(rows, list) or not 1 <= len(rows) <= 9:
        raise ValueError("Expected 1–9 Remove Noise variants.")
    if any(not isinstance(row, dict) or
           not isinstance(row.get("method"), str) or
           not isinstance(row.get("preset"), str) for row in rows):
        raise ValueError("Expected Remove Noise method and preset pairs.")
    keys = [(row.get("method"), row.get("preset")) for row in rows]
    if len(set(keys)) != len(keys) or any(key not in VARIANTS for key in keys):
        raise ValueError("Unsupported or duplicate Remove Noise variant.")


def batch(source: Path, rows: list[dict], profile: str) -> None:
    sys.path.insert(0, str(SCRIPTS))
    from denoise_video import run
    from common import find_ffmpeg
    ffmpeg = find_ffmpeg(None)
    total = len(rows)
    failures = []
    for index, item in enumerate(rows, 1):
        label = f"{item['method'].upper()} / {item['preset']}"
        print(f"Remove Noise batch {index}/{total}: {label}", flush=True)
        try:
            result = run(source, resolve(item["output"]), method=item["method"],
                         preset=item["preset"], intermediate_profile=profile,
                         ffmpeg=ffmpeg)
            status = "reused" if result.get("reused") else "rendered"
            print(f"Remove Noise batch complete {index}/{total}: {label} ({status})", flush=True)
        except Exception as exc:
            failures.append(label)
            print(f"Remove Noise batch failed {index}/{total}: {label}: {exc}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)}/{total} Remove Noise variants failed: {', '.join(failures)}")
    print("Remove Noise batch all complete", flush=True)


def export_identity(source: Path, rows: list[dict], frames: list[int], profile: str) -> str:
    def file_identity(p: Path) -> list:
        stat = p.stat()
        return [str(p.resolve()), stat.st_size, stat.st_mtime_ns]
    payload = {
        "version": GRID_VERSION, "source": file_identity(source),
        "variants": [[row["method"], row["preset"], file_identity(resolve(row["output"]))]
                     for row in rows],
        "frames": frames, "profile": profile
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def output_path(source: Path, identity: str) -> Path:
    safe_name = "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in source.stem)[:48]
    return EXPORT_ROOT / f"{safe_name}_remove_noise_grid_{identity}.zip"


def _luma_difference(before, after):
    """Exact decoded RGB deltas with the same directional color and gamma as the UI."""
    import numpy as np
    a = np.asarray(before.convert("RGB"), dtype=np.int16)
    b = np.asarray(after.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        raise ValueError("Source and result frame dimensions differ.")
    rgb_delta = b.astype(np.float32) - a.astype(np.float32)
    delta = rgb_delta[:, :, 0] * 0.2126 + rgb_delta[:, :, 1] * 0.7152 + rgb_delta[:, :, 2] * 0.0722
    value = np.rint(255.0 * np.sqrt(np.minimum(1.0, np.abs(delta) * 32.0 / 255.0))).astype(np.uint8)
    out = np.empty_like(a, dtype=np.uint8)
    out[:, :, 0] = np.where(delta > 0, np.rint(value * .05), value)
    out[:, :, 1] = np.where(delta > 0, value, np.rint(value * .52))
    out[:, :, 2] = np.where(delta > 0, value, np.rint(value * .06))
    unchanged = np.abs(delta) < 1e-9
    out[unchanged] = (15, 23, 31)
    from PIL import Image
    return Image.fromarray(out, mode="RGB"), {
        "mean_absolute_luminance": round(float(np.abs(delta).mean()), 4),
        "pixel_change_percent": round(float((np.abs(delta) >= 1).mean() * 100), 2),
    }


def _image(path: Path, frame: int):
    from PIL import Image
    png = resolve(_decode(path, frame))
    with Image.open(png) as img:
        return img.convert("RGB")


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _sheet(frame: int, fps: float, source_frame, rows: list[dict], results: list[tuple]):
    from PIL import Image, ImageDraw
    width = 2 * TILE_W + 3 * PAD
    height = TOP_H + (1 + len(rows)) * ROW_H + PAD
    sheet = Image.new("RGB", (width, height), "#101b22")
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD, 11), f"FRAME {frame + 1}   |   {frame / fps:.3f} s   |   EXACT SOURCE INDEX",
              fill="#ffffff", font=_font(19))
    draw.text((PAD, 40), "Original / Ergebnis (links) | Pixeldifferenz 32x + Gamma (rechts)",
              fill="#b6c9d5", font=_font(13))
    entries = [("ORIGINAL — unverändert", source_frame, None, None)]
    entries.extend((f"{row['method'].upper()} / {row['preset'].upper()}", result, diff, metric)
                   for row, (result, diff, metric) in zip(rows, results))
    for idx, (name, image, diff, metric) in enumerate(entries):
        y = TOP_H + idx * ROW_H
        draw.rectangle((PAD - 2, y, width - PAD + 2, y + ROW_H - 7), fill="#1a2a35")
        draw.text((PAD + 5, y + 5), name, fill="#f5f8fa", font=_font(16))
        if metric:
            score = f"|ΔY| = {metric['mean_absolute_luminance']:.2f}/255  ·  ΔY≥1: {metric['pixel_change_percent']:.1f}%"
            draw.text((PAD + 5, y + 29), score, fill="#b8ced8", font=_font(12))
        image = image.copy()
        image.thumbnail((TILE_W, TILE_H), Image.Resampling.LANCZOS)
        px = PAD + (TILE_W - image.width) // 2
        py = y + 40 + (TILE_H - image.height) // 2
        sheet.paste(image, (px, py))
        if diff is not None:
            d = diff.copy()
            d.thumbnail((TILE_W, TILE_H), Image.Resampling.LANCZOS)
            px2 = TILE_W + 2 * PAD + (TILE_W - d.width) // 2
            sheet.paste(d, (px2, y + 40 + (TILE_H - d.height) // 2))
        else:
            draw.text((TILE_W + 2 * PAD + 10, y + 118),
                      "Referenz / keine Differenz", fill="#a3b5c3", font=_font(13))
    return sheet


def _png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, "PNG", optimize=False)
    return buf.getvalue()


def grid(source: Path, rows: list[dict], frames: list[int], profile: str, target: Path) -> None:
    from PIL import Image
    count = int(_metadata(source)["frames"])
    if not 1 <= len(frames) <= 5 or len(frames) != len(set(frames)) or min(frames) < 0 or max(frames) >= count:
        raise ValueError("Grid needs 1–5 distinct source-frame indices.")
    for row in rows:
        _validate_pair(source, resolve(row["output"]))
    sheets = []
    metrics = []
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.stem + ".partial.zip")
    temp.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            fps = _metadata(source)["fps"]
            for n, frame in enumerate(frames, 1):
                print(f"Remove Noise grid frame {n}/{len(frames)}: {frame + 1}", flush=True)
                original = _image(source, frame)
                rendered = []
                for row in rows:
                    variant = _image(resolve(row["output"]), frame)
                    difference, metric = _luma_difference(original, variant)
                    metrics.append({"frame": frame + 1, "seconds": round(frame / fps, 4),
                                    "method": row["method"], "preset": row["preset"], **metric})
                    rendered.append((variant, difference, metric))
                    archive.writestr(
                        f"differences/frame_{frame + 1:06d}_{row['method']}_{row['preset']}.png",
                        _png_bytes(difference))
                sheet = _sheet(frame, fps, original, rows, rendered)
                sheets.append(sheet)
                archive.writestr(f"frames/frame_{frame + 1:06d}_grid.png", _png_bytes(sheet))
                print(f"Remove Noise grid complete {n}/{len(frames)}", flush=True)
            overview = Image.new("RGB", (sum(s.width for s in sheets), max(s.height for s in sheets)), "#101b22")
            x = 0
            for sheet in sheets:
                overview.paste(sheet, (x, 0))
                x += sheet.width
            archive.writestr("overview_all_frames.png", _png_bytes(overview))
            csvbuf = io.StringIO()
            writer = csv.DictWriter(csvbuf, fieldnames=["frame", "seconds", "method", "preset",
                                                         "mean_absolute_luminance", "pixel_change_percent"])
            writer.writeheader()
            writer.writerows(metrics)
            archive.writestr("measurement.csv", csvbuf.getvalue().encode("utf-8-sig"))
            archive.writestr("README.txt", (
                "ARP Remove Noise – exact decoded source-frame comparison\n"
                f"Source: {source.name}\nProfile: {profile}\nFrames (1-based): "
                + ", ".join(str(i + 1) for i in frames) + "\n"
                "Rows: original, then completed method/preset variants.\n"
                "Left panel: source or processed result. Right panel: pixel differences.\n"
                "Cyan means brighter; orange means darker; dark graphite means unchanged.\n"
                "Difference visualization: luminance delta, 32x contrast + sqrt gamma.\n"
                "Original images and rendered videos are never modified.\n"
                "Pixel differences do NOT prove denoising, grain preservation or image quality.\n"
                "Contact sheets are downscaled for comparison; separate difference PNGs have source resolution.\n"
                "measurement.csv contains unamplified per-pixel luminance statistics.\n"
            ).encode("utf-8"))
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Remove Noise grid saved: {rel(target)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=("batch", "grid"))
    parser.add_argument("--source", required=True)
    parser.add_argument("--profile", choices=("low", "medium", "high", "lossless"), required=True)
    parser.add_argument("--variants", required=True, help="Server-generated JSON rows")
    parser.add_argument("--frames", default="", help="JSON list of zero-based frames for grid")
    parser.add_argument("--output", default="", help="ZIP target for grid")
    args = parser.parse_args()
    source = resolve(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = json.loads(args.variants)
    if not isinstance(rows, list):
        raise ValueError("Expected array of method/preset rows.")
    validate_variants(rows)
    if args.job == "batch":
        batch(source, rows, args.profile)
    else:
        frames = json.loads(args.frames)
        if not isinstance(frames, list) or not all(type(x) is int for x in frames):
            raise ValueError("Invalid frame indices.")
        target = resolve(args.output)
        if target.resolve().parent != EXPORT_ROOT.resolve() or target.suffix.lower() != ".zip":
            raise ValueError("Grid output must be in the dedicated export directory.")
        grid(source, rows, frames, args.profile, target)


if __name__ == "__main__":
    main()
