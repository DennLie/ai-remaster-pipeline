"""Frame-exact, non-generative QA of archival footage stabilization.

Adjacent image-content motion can be estimated with sparse optical flow
and robust 2D fits; it is NOT automatically gate weave rather than an
intentional camera pan. No ground-truth transforms are claimed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from compare_pre_dart_baselines import open_video, tile


def estimate_motion(before: np.ndarray, after: np.ndarray) -> dict[str, Any] | None:
    """Measure apparent global translation between consecutive image frames.

    Exclude uncertain matches, heavy occlusions and camera cuts. Features
    come from a central crop to reduce black film borders; that does
    not fully distinguish real camera motion from transfer weave.
    """
    if before.shape != after.shape or before.dtype != np.uint8:
        raise ValueError("Consecutive frames must have matching uint8 geometry")
    h, w = before.shape[:2]
    gray1 = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    roi = np.zeros(gray1.shape, np.uint8)
    roi[h//10:h-h//10, w//10:w-w//10] = 255
    pts = cv2.goodFeaturesToTrack(gray1, maxCorners=800, qualityLevel=0.015,
                                  minDistance=7, mask=roi, blockSize=5)
    if pts is None or len(pts) < 25:
        return None
    target, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, pts, None, winSize=(25, 25), maxLevel=3
    )
    if target is None or status is None:
        return None
    backward, status_back, _ = cv2.calcOpticalFlowPyrLK(
        gray2, gray1, target, None, winSize=(25, 25), maxLevel=3
    )
    if backward is None or status_back is None:
        return None
    forward_back = np.linalg.norm(backward.reshape(-1, 2) - pts.reshape(-1, 2), axis=1)
    valid = (status.ravel() == 1) & (status_back.ravel() == 1) & (forward_back <= 1.5)
    if int(np.count_nonzero(valid)) < 25:
        return None
    matrix, inliers = cv2.estimateAffinePartial2D(
        pts.reshape(-1, 2)[valid], target.reshape(-1, 2)[valid],
        method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000,
        confidence=0.99, refineIters=8
    )
    if matrix is None or inliers is None:
        return None
    ratio = float(np.count_nonzero(inliers)) / len(inliers)
    if ratio < 0.40:
        return None
    tx, ty = float(matrix[0, 2]), float(matrix[1, 2])
    return {
        "tx_px": round(tx, 5), "ty_px": round(ty, 5),
        "magnitude_px": round(float(np.hypot(tx, ty)), 5),
        "rotation_deg": round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 5),
        "inlier_ratio": round(ratio, 4), "inlier_points": int(np.count_nonzero(inliers)),
    }


def summarize_motion(motion: list[dict[str, Any] | None]) -> dict[str, Any]:
    good = [m for m in motion if m is not None]
    result = {"trusted_adjacent_pairs": len(good), "rejected_adjacent_pairs": len(motion),
              "median_translation_px": None, "median_translation_acceleration_px": None,
              "median_rotation_deg": None}
    result["rejected_adjacent_pairs"] -= len(good)
    if not good:
        return result
    result["median_translation_px"] = round(float(np.median([m["magnitude_px"] for m in good])), 5)
    result["median_rotation_deg"] = round(float(np.median([abs(m["rotation_deg"]) for m in good])), 5)
    accelerations = []
    for a, b in zip(motion, motion[1:]):
        if a is not None and b is not None:
            accelerations.append(float(np.hypot(
                b["tx_px"] - a["tx_px"], b["ty_px"] - a["ty_px"]
            )))
    if accelerations:
        result["median_translation_acceleration_px"] = round(float(np.median(accelerations)), 5)
    return result


def evaluate(source: Path, variants: dict[str, Path], output_dir: Path,
              *, cut_after_frames: set[int] | None = None,
              sample_frames: set[int] | None = None) -> dict[str, Any]:
    if not variants:
        raise ValueError("Provide at least one stabilization variant")
    cuts = set(cut_after_frames or ())
    if any(type(i) is not int or i < 0 for i in cuts):
        raise ValueError("Cut markers must be nonnegative frame indices")
    names = ["original", *variants]
    handles: dict[str, Any] = {}
    try:
        for name, path in [("original", source), *variants.items()]:
            handles[name] = open_video(path)
        original_size, fps = handles["original"][1:]
        for name, (_, size, rate) in handles.items():
            if size != original_size or abs(rate - fps) > 0.02:
                raise ValueError(f"Frame geometry or FPS mismatch for {name}")
        previous: dict[str, np.ndarray] = {}
        motions: dict[str, list[dict[str, Any] | None]] = {name: [] for name in names}
        edge_values: dict[str, list[float]] = {name: [] for name in names}
        sheets: list[np.ndarray] = []
        frame_no = 0
        while True:
            reads = {name: cap.read() for name, (cap, _, _) in handles.items()}
            if not reads["original"][0]:
                if any(success for name, (success, _) in reads.items() if name != "original"):
                    raise ValueError("A stabilized candidate has extra frames")
                break
            if not all(success for success, _ in reads.values()):
                raise ValueError(f"A stabilized candidate ended before frame {frame_no}")
            current = {name: img for name, (_, img) in reads.items()}
            if any(img.shape != current["original"].shape for img in current.values()):
                raise ValueError("Frame dimensions differ after decode")
            h, w = current["original"].shape[:2]
            # Deliberately a proxy, not measured source coverage: archival
            # originals can themselves have dark borders and vignette.
            border = max(2, round(min(h, w) * 0.04))
            edge = np.zeros((h, w), dtype=bool)
            edge[:border] = True
            edge[-border:] = True
            edge[:, :border] = True
            edge[:, -border:] = True
            for name in names:
                gray = cv2.cvtColor(current[name], cv2.COLOR_BGR2GRAY)
                edge_values[name].append(float(np.mean(gray[edge] <= 2)))
                if name in previous:
                    # A cut after index N occurs between N and N+1.
                    pair = (None if frame_no - 1 in cuts else estimate_motion(previous[name], current[name]))
                    motions[name].append(pair)
                previous[name] = current[name]
            if frame_no in (sample_frames if sample_frames is not None else (0, 16, 31, 63, 95)):
                sheets.append(np.concatenate(
                    [tile(current[name], f"frame {frame_no} | {name}", width=320) for name in names],
                    axis=1
                ))
            frame_no += 1
        if frame_no < 3:
            raise ValueError("Need at least three frames")
        if cuts and max(cuts) >= frame_no - 1:
            raise ValueError("Cut index out of range")
        result = {
            "schema_version": 1, "source": source.name,
            "width": original_size[0], "height": original_size[1],
            "fps": fps, "frames": frame_no, "cut_after_frames_excluded": sorted(cuts),
            "measurements": {
                name: {
                    **summarize_motion(motions[name]),
                    "edge_near_black_fraction_proxy": round(float(np.mean(edge_values[name])), 6),
                    "valid_source_footprint_measured": False,
                } for name in names
            },
            "warning": (
                "Motion of image content is not a measurement of historical film transport jitter: "
                "it includes camera pans, subject movement, edits and matching errors. "
                "Near-black edge pixels are not a source-footprint mask; "
                "3 percent fixed zoom loses source field-of-view independently of edge color. "
                "Manually inspect restored motion and frame edges."
            ),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        if not sheets:
            raise ValueError("No contact-sheet frames")
        if not cv2.imwrite(str(output_dir / "stabilization_contact_sheet.jpg"),
                           np.concatenate(sheets, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise IOError("Could not write contact sheet")
        (output_dir / "stabilization_qa.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8"
        )
        with (output_dir / 'motion_frames.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['frame', 'variant', 'excluded_cut', 'trusted',
                'tx_px', 'ty_px', 'magnitude_px', 'rotation_deg', 'inlier_ratio', 'inlier_points'])
            writer.writeheader()
            for name, rows in motions.items():
                for index, motion in enumerate(rows, 1):
                    writer.writerow({'frame': index, 'variant': name, 'excluded_cut': index-1 in cuts,
                                     'trusted': motion is not None, **(motion or {})})
        return result
    finally:
        for cap, _, _ in handles.values():
            cap.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--variant", action="append", required=True, help="LABEL=PATH")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cut-after-frame", type=int, action="append", default=[])
    args = p.parse_args()
    from compare_pre_dart_baselines import parse_variants
    report = evaluate(args.source, parse_variants(args.variant), args.output_dir,
                      cut_after_frames=set(args.cut_after_frame))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
