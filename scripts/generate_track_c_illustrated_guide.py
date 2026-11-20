#!/usr/bin/env python3
"""Generate raw-only illustrated annotation guide for Track C archival film labeling.

Uses ONLY raw archival pilot keyframes and clean synthetic schematic diagrams.
Never loads detectors, residuals, optical flow, or restored outputs.
"""
from __future__ import annotations

from pathlib import Path
import textwrap
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACK_DIR = PROJECT_ROOT / ".cache" / "benchmarks" / "track_c_pilot_pack"
OUT_PATH = PROJECT_ROOT / "docs" / "benchmarks" / "track_c_annotation_illustrated_guide.png"


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _load_rgb(path: Path) -> Image.Image:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read raw image: {path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _crop_fit(img: Image.Image, box: tuple[int, int, int, int] | None, size: tuple[int, int]) -> Image.Image:
    src = img.crop(box) if box is not None else img
    return src.resize(size, Image.Resampling.NEAREST if box is not None else Image.Resampling.BICUBIC)


def build_illustrated_guide(pack_dir: Path = PACK_DIR, out_path: Path = OUT_PATH) -> Path:
    W, H = 1480, 960
    canvas = Image.new("RGB", (W, H), (16, 20, 26))
    draw = ImageDraw.Draw(canvas)

    f_title = _load_font(24, bold=True)
    f_sub = _load_font(15, bold=False)
    f_h2 = _load_font(17, bold=True)
    f_body = _load_font(13, bold=False)
    f_bold = _load_font(13, bold=True)

    # Header
    draw.rectangle([0, 0, W, 76], fill=(23, 30, 39))
    draw.text((24, 14), "Track C Independent Human Annotation Guide (Raw Archival Frames Only)", fill=(235, 242, 250), font=f_title)
    draw.text((24, 46), "3-Tier Ground Truth: 0 = Verified Intact  |  127 = Uncertain / Unreviewed  |  255 = Verified Defect (Separate Scratch & Dust Channels)", fill=(125, 211, 252), font=f_sub)

    # Panel 1: Left Top - 3-Tier Boundary Schematic (Zero model data)
    p1 = (24, 96, 728, 505)
    draw.rounded_rectangle(p1, radius=10, fill=(24, 31, 40), outline=(52, 65, 82), width=2)
    draw.text((40, 110), "1. Three-Tier Labeling Principle (Zoomed Pixel-Boundary Rules)", fill=(240, 245, 250), font=f_h2)
    draw.text((40, 136), "Never paint unreviewed pixels as 0. Leave uncertain boundaries as 127.", fill=(165, 180, 196), font=f_body)

    # Schematic A: Clean vs Defect vs 127 boundary buffer
    sx0, sy0 = 44, 170
    cell = 16
    gw, gh = 18, 13
    for gy in range(gh):
        for gx in range(gw):
            x0 = sx0 + gx * cell
            y0 = sy0 + gy * cell
            # Vertical bright scratch core around gx in {8, 9} -> 255
            # Guard band gx in {7, 10} -> 127
            # Outer region -> 0
            is_dust = (gx in (14, 15) and gy in (3, 4))
            is_dust_halo = (13 <= gx <= 16 and 2 <= gy <= 5 and not is_dust)
            if gx in (8, 9) and 1 <= gy <= 11:
                col = (0, 229, 255)  # Scratch 255
                lbl = "255"
                tcol = (10, 20, 30)
            elif is_dust:
                col = (255, 152, 0)  # Dust 255
                lbl = "255"
                tcol = (10, 20, 30)
            elif (gx in (7, 10) and 0 <= gy <= 12) or (gx in (8, 9) and gy in (0, 12)) or is_dust_halo:
                col = (95, 108, 124)  # Uncertain 127 halo
                lbl = "127"
                tcol = (240, 245, 250)
            elif gx <= 2:
                col = (60, 70, 84)  # Left unreviewed 127
                lbl = "127"
                tcol = (210, 220, 230)
            else:
                col = (34, 160, 82)  # Verified Intact 0
                lbl = "0"
                tcol = (240, 255, 245)
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=col)
    cap_lines = textwrap.wrap(
        "Cyan (255) = Core Scratch | Orange (255) = Core Dust | Green (0) = Intact | Gray (127) = Halo / Unreviewed",
        width=42,
    )
    for ci, cline in enumerate(cap_lines):
        draw.text((sx0, sy0 + gh * cell + 8 + ci * 18), cline, fill=(203, 213, 225), font=f_body)

    # Schematic B: Legend cards inside Panel 1
    lx = 358
    boxes = [
        ((0, 229, 255), "Scratch Channel = 255 (Cyan)", "Linear vertical/angled/branching film damage verified across neighbors."),
        ((255, 152, 0), "Dust Channel = 255 (Orange)", "Compact blobs, specks, hair/fibers differing from neighboring frames."),
        ((34, 197, 94), "Verified Intact = 0 (Green)", "Confidently clean background, grain, sky, actors, walls, or poles."),
        ((232, 121, 249), "Neg Control ROI = 255 + Intact", "Fragile intact edges/structures that must NEVER suffer false repair."),
        ((100, 116, 139), "Uncertain / Unreviewed = 127", "Default on load. Also use on soft defect halos & ambiguous motion blur."),
    ]
    for idx, (c, title_b, desc_b) in enumerate(boxes):
        by = 168 + idx * 62
        draw.rectangle([lx, by, lx + 20, by + 20], fill=c, outline=(255, 255, 255))
        draw.text((lx + 28, by), title_b, fill=(240, 245, 250), font=f_bold)
        wrapped = textwrap.wrap(desc_b, width=48)
        for wi, wline in enumerate(wrapped[:2]):
            draw.text((lx + 28, by + 20 + wi * 16), wline, fill=(165, 180, 196), font=f_body)

    # Panel 2: Right Top - Temporal Neighbor Flipping on Raw Archival Frames
    p2 = (748, 96, W - 24, 505)
    draw.rounded_rectangle(p2, radius=10, fill=(24, 31, 40), outline=(52, 65, 82), width=2)
    draw.text((764, 110), "2. Use Raw Neighbors (t-2..t+2) — Never Model Outputs", fill=(240, 245, 250), font=f_h2)
    draw.text((764, 136), "Neighbors_72 raw frames f60, f61, f62 (flip Q / W or Blink T in annotator):", fill=(165, 180, 196), font=f_body)

    strip_61 = _load_rgb(pack_dir / "neighbors_72" / "neighbor_strips" / "strip_000061.png")
    # Extract t-1 (f60), t (f61), t+1 (f62) from raw strip (width = 5 * 960, header = 28)
    rw, rh = 960, 720
    slots = [("f60 (t-1)", 1), ("f61 (t)", 2), ("f62 (t+1)", 3)]
    for i, (cap, slot) in enumerate(slots):
        crop = strip_61.crop((slot * rw, 28, (slot + 1) * rw, 28 + rh)).resize((220, 165), Image.Resampling.BICUBIC)
        px = 764 + i * 232
        py = 166
        canvas.paste(crop, (px, py))
        draw.rectangle([px, py, px + 220, py + 165], outline=(56, 189, 248) if slot == 2 else (71, 85, 105), width=2)
        draw.text((px + 6, py + 172), cap, fill=(226, 232, 240), font=f_bold)

    draw.text((764, 372), "Decision cues purely from raw frames:", fill=(125, 211, 252), font=f_bold)
    cues = [
        "- Persistent scratch / fissure: stays at ~same film coordinate or slowly drifts while cutting across scene contours.",
        "- Transient dust / speck: pops in frame t and is absent at that scene location in both t-1 and t+1.",
        "- Intact architecture / actor edge: moves smoothly with camera pan or articulated body across t-1 -> t -> t+1.",
        "- Genuinely ambiguous thin vertical line on moving panning wall? Leave border or region at 127 (Uncertain).",
    ]
    for ci, line in enumerate(cues):
        draw.text((764, 396 + ci * 22), line, fill=(203, 213, 225), font=f_body)

    # Panel 3 & 4 Bottom Row: Critical Intact Negative-Control Structures (Raw Zooms)
    p3 = (24, 525, W - 24, H - 24)
    draw.rounded_rectangle(p3, radius=10, fill=(24, 31, 40), outline=(52, 65, 82), width=2)
    draw.text((40, 539), "3. Archival Negative-Control Structures (Verify Intact = 0 and Mark Neg Control ROI = 255 Where Verified)", fill=(240, 245, 250), font=f_h2)
    draw.text((40, 565), "Inspect zoomed raw crops across t-2..t+2 before marking; never copy coordinates blindly if a real speck falls near a structure.", fill=(165, 180, 196), font=f_body)

    examples = [
        (
            "neighbors_72 f25 — Intact Pipe",
            pack_dir / "neighbors_72" / "raw_frames" / "frame_000025.png",
            (130, 30, 250, 300),
            (180, 48, 192, 282),
            "Vertical pipe x:180..192, y:48..282",
        ),
        (
            "paleface_seq_a f8 — Jacket Edge",
            pack_dir / "paleface_seq_a" / "raw_frames" / "frame_000008.png",
            (205, 165, 325, 400),
            (258, 188, 268, 381),
            "Jacket seam x:258..268, y:188..381",
        ),
        (
            "paleface_seq_b f7 — Wooden Wall",
            pack_dir / "paleface_seq_b" / "raw_frames" / "frame_000007.png",
            (175, 55, 305, 205),
            (200, 80, 280, 180),
            "Wall/curtain x:200..280, y:80..180",
        ),
        (
            "college_seq_c f8 — Door Jamb",
            pack_dir / "college_seq_c" / "raw_frames" / "frame_000008.png",
            (245, 125, 375, 305),
            (280, 150, 340, 280),
            "Jamb/student x:280..340, y:150..280",
        ),
        (
            "neighbors_seq_d f30 — Utility Pole",
            pack_dir / "neighbors_seq_d" / "raw_frames" / "frame_000030.png",
            (115, 175, 255, 425),
            (150, 200, 220, 400),
            "Pole x:150..220, y:200..400",
        ),
    ]
    card_w = 266
    for idx, (title_e, path_e, crop_box, roi_box, sub_e) in enumerate(examples):
        im = _load_rgb(path_e)
        cx0, cy0, cx1, cy1 = crop_box
        cropped = im.crop(crop_box)
        disp_w, disp_h = 250, 250
        resized = cropped.resize((disp_w, disp_h), Image.Resampling.NEAREST)
        bx = 40 + idx * (card_w + 14)
        by = 598
        canvas.paste(resized, (bx, by))
        # Map ROI rectangle cleanly into resized coordinates
        sx = disp_w / max(1, cx1 - cx0)
        sy = disp_h / max(1, cy1 - cy0)
        rx0 = bx + int(round((roi_box[0] - cx0) * sx))
        ry0 = by + int(round((roi_box[1] - cy0) * sy))
        rx1 = bx + int(round((roi_box[2] - cx0) * sx))
        ry1 = by + int(round((roi_box[3] - cy0) * sy))
        draw.rectangle([rx0, ry0, rx1, ry1], outline=(232, 121, 249), width=2)
        draw.rectangle([bx, by, bx + disp_w, by + disp_h], outline=(71, 85, 105), width=2)
        draw.text((bx, by + disp_h + 8), title_e, fill=(240, 245, 250), font=f_bold)
        draw.text((bx, by + disp_h + 26), sub_e, fill=(165, 180, 196), font=f_body)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="PNG")
    return out_path


if __name__ == "__main__":
    p = build_illustrated_guide()
    print(f"Generated raw-only Track C illustrated guide: {p}")
