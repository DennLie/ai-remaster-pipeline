"""Diagnostic only: compare exactly aligned source/Light-Deflicker video frames.

This is a visual QA utility, NOT a quality ground-truth metric. Differences
in frame luminance do not establish restoration accuracy or film authenticity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from film_deflicker import frame_luma


def compare_videos(original: Path, corrected: Path, output_dir: Path) -> dict:
    source = cv2.VideoCapture(str(original))
    processed = cv2.VideoCapture(str(corrected))
    if not source.isOpened() or not processed.isOpened():
        source.release()
        processed.release()
        raise FileNotFoundError("Source or corrected video cannot be decoded")
    size_a = (int(source.get(cv2.CAP_PROP_FRAME_WIDTH)), int(source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    size_b = (int(processed.get(cv2.CAP_PROP_FRAME_WIDTH)), int(processed.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps_a = float(source.get(cv2.CAP_PROP_FPS))
    fps_b = float(processed.get(cv2.CAP_PROP_FPS))
    if size_a != size_b or min(fps_a, fps_b) <= 0 or abs(fps_a - fps_b) > 0.02:
        source.release()
        processed.release()
        raise ValueError(f"Mismatched video geometry/cadence: {size_a}/{fps_a} vs {size_b}/{fps_b}")
    values_a, values_b = [], []
    selected: list[tuple[float, int, np.ndarray]] = []
    try:
        while True:
            ok_a, a = source.read()
            ok_b, b = processed.read()
            if ok_a != ok_b:
                raise ValueError("Frame count mismatch; cannot compare mismatched footage")
            if not ok_a:
                break
            i = len(values_a)
            luma_a, _ = frame_luma(a)
            luma_b, _ = frame_luma(b)
            values_a.append(luma_a)
            values_b.append(luma_b)
            # Keep three examples with the largest measured brightness change.
            priority = abs(luma_a - luma_b)
            if len(selected) < 3 or priority > min(item[0] for item in selected):
                h = max(160, int(round(size_a[1] * 480 / size_a[0])))
                a_small = cv2.resize(a, (480, h), interpolation=cv2.INTER_AREA)
                b_small = cv2.resize(b, (480, h), interpolation=cv2.INTER_AREA)
                paired = np.concatenate([a_small, b_small], axis=1)
                cv2.putText(paired, f"Frame {i} | ORIGINAL", (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(paired, f"Frame {i} | LIGHT DEFLICKER", (492, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                selected.append((priority, i, paired))
                selected = sorted(selected, key=lambda item: item[0], reverse=True)[:3]
    finally:
        source.release()
        processed.release()
    if len(values_a) < 3:
        raise ValueError("At least three aligned video frames are needed")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected.sort(key=lambda item: item[1])
    cv2.imwrite(str(output_dir / "light_deflicker_before_after.jpg"),
                np.concatenate([item[2] for item in selected], axis=0),
                [cv2.IMWRITE_JPEG_QUALITY, 90])
    original_delta = np.abs(np.diff(np.asarray(values_a, dtype=np.float32)))
    corrected_delta = np.abs(np.diff(np.asarray(values_b, dtype=np.float32)))
    report = {
        "schema_version": 1,
        "purpose": "Descriptive paired-frame QA; not restoration accuracy or authenticity",
        "source": original.name, "corrected": corrected.name,
        "width": size_a[0], "height": size_a[1], "fps": fps_a,
        "frames": len(values_a),
        "changed_median_luma_frames": sum(a != b for a, b in zip(values_a, values_b)),
        "max_median_luma_change": round(max(abs(a-b) for a,b in zip(values_a, values_b)), 5),
        "mean_abs_adjacent_median_change_source": round(float(np.mean(original_delta)), 5),
        "mean_abs_adjacent_median_change_corrected": round(float(np.mean(corrected_delta)), 5),
        "warning": "Adjacent changes include real scene edits, motion and intentional lighting; lower is not necessarily more correct.",
        "example_frames": [item[1] for item in selected],
    }
    (output_dir / "light_deflicker_compare.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_videos(args.original, args.corrected, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
