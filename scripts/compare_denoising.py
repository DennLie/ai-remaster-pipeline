"""Frame-exact exploratory comparison of multiple ARP denoising methods.

This is a visual/technical diagnostic, not a no-reference denoising
quality metric: grain, true fine detail and compression noise can all
register as high spatial frequency. Never interpret reduction as better.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_options(options: list[str]) -> dict[str, Path]:
    if not options:
        raise ValueError("At least one variant LABEL=PATH is required")
    variants: dict[str, Path] = {}
    for pair in options:
        label, sep, path = pair.partition("=")
        if not sep or not label or not path or not label.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"Invalid variant: {pair!r}")
        if label in variants or label == "original":
            raise ValueError(f"Duplicate/reserved label: {label}")
        variants[label] = Path(path)
    return variants


def open_capture(path: Path) -> tuple[cv2.VideoCapture, tuple[int, int], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot read video {path}")
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if min(*size, fps) <= 0:
        capture.release()
        raise ValueError(f"Bad video geometry/cadence {path}")
    return capture, size, fps


def metrics(original: np.ndarray, processed: np.ndarray) -> dict[str, float]:
    if original.shape != processed.shape:
        raise ValueError("Frame geometry changed")
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    # High-frequency gradient measure, NOT a grain/noise estimator.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    diff = cv2.absdiff(original, processed)
    return {
        "mean_abs_change": float(np.mean(diff)),
        "changed_pixel_fraction_gte_8": float(np.mean(np.max(diff, axis=2) >= 8)),
        "high_frequency_laplacian_abs_mean": float(np.mean(np.abs(lap))),
        "new_clipped_channel_fraction": float(np.mean(
            ((processed <= 1) & (original > 1)) |
            ((processed >= 254) & (original < 254))
        )),
    }


def tile(frame: np.ndarray, label: str, width: int = 320) -> np.ndarray:
    h = max(120, int(round(frame.shape[0] * width / frame.shape[1])))
    small = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
    footer = np.full((28, width, 3), 25, np.uint8)
    cv2.putText(footer, label[:38], (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([small, footer], axis=0)


def compare(source: Path, variants: dict[str, Path], output_dir: Path) -> dict:
    if not variants:
        raise ValueError("No Denoising variants")
    paths = {"original": source, **variants}
    captures: dict[str, tuple[cv2.VideoCapture, tuple[int, int], float]] = {}
    try:
        for label, path in paths.items():
            captures[label] = open_capture(path)
        size, fps = captures["original"][1:]
        for label, (_, geometry, rate) in captures.items():
            if geometry != size or abs(rate-fps) > 0.02:
                raise ValueError(f"{label} has mismatched geometry / fps: {geometry}, {rate}")
        totals = {k: {"mean_abs_change": 0.0,
                      "changed_pixel_fraction_gte_8": 0.0,
                      "high_frequency_laplacian_abs_mean": 0.0,
                      "new_clipped_channel_fraction": 0.0} for k in paths}
        chosen: list[np.ndarray] = []
        count = 0
        while True:
            fetched = {name: captures[name][0].read() for name in paths}
            if not fetched["original"][0]:
                if any(ok for name, (ok, _) in fetched.items() if name != "original"):
                    raise ValueError("Variant contains extra frames")
                break
            if any(not ok for ok, _ in fetched.values()):
                raise ValueError(f"Variant ended early at frame {count}")
            original = fetched["original"][1]
            if original.shape != (size[1], size[0], 3):
                raise ValueError("Unexpected original frame geometry")
            panels = []
            for label in paths:
                frame = fetched[label][1]
                if frame.shape != original.shape or frame.dtype != np.uint8:
                    raise ValueError(f"Unexpected {label} frame geometry/bitdepth at {count}")
                for key, value in metrics(original, frame).items():
                    totals[label][key] += value
                if count in (0, 5, 12, 20):
                    panels.append(tile(frame, f"Frame {count:02d}: {label}"))
            if panels:
                chosen.append(np.concatenate(panels, axis=1))
            count += 1
        if count < 3:
            raise ValueError("Need at least 3 frames to compare")
        output_dir.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_dir / "denoising_comparison.jpg"),
                           np.concatenate(chosen, axis=0),
                           [cv2.IMWRITE_JPEG_QUALITY, 91]):
            raise IOError("Could not write comparison poster")
        report = {
            "schema_version": 1,
            "original": source.name, "frames": count, "width": size[0],
            "height": size[1], "fps": fps,
            "variants": {name: {key: round(value/count, 7) for key, value in data.items()}
                         for name, data in totals.items()},
            "warning": (
                "These are descriptive differences, not denoising accuracy. "
                "Laplacian energy includes desirable film grain and real fine detail, "
                "and lower scores can mean unwanted blur. Inspect at native resolution."
            ),
        }
        (output_dir / "denoising_comparison.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        for capture, _, _ in captures.values():
            capture.release()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--variant", action="append", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(compare(args.source, parse_options(args.variant), args.output_dir),
                     indent=2))


if __name__ == "__main__":
    main()
