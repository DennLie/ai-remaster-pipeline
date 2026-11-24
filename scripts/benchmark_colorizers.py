#!/usr/bin/env python3
"""Internal Non-Commercial Benchmark: osi_masked_lab vs CMNET2.

This standalone benchmark evaluates quantitative fidelity and temporal consistency
metrics between 100%-OSI color propagation (osi_masked_lab) and exemplar/memory-based
colorization (CMNET2).

Compliance Policy:
- Fails closed immediately unless explicitly authorized via allow_non_commercial=True
  or --allow-non-commercial-benchmark.
- Prohibits writing into commercial master directories (ROOT / "output").
- Emits compliance audit records marked with NON_COMMERCIAL_BENCHMARK.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import cv2
import numpy as np

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ai_remaster_gui.color_qc import ciede2000, compute_cdc
from ai_remaster_gui.color_science import load_image_rgb, srgb_to_lab
from ai_remaster_gui.commercial_policy import CommercialPolicyError
from ai_remaster_gui.paths import ROOT, is_within

try:
    from ai_remaster_gui.compliance import append_compliance_record
except ImportError:
    append_compliance_record = None


def validate_benchmark_output_path(output_dir: Path | str) -> Path:
    """Validate that output_dir is strictly in benchmark_reports/ or an isolated temp sandbox.

    Prohibits writing into commercial master directories (e.g. ROOT / "output").
    """
    target = Path(output_dir).resolve()
    commercial_master = (ROOT / "output").resolve()

    # 1. Reject if within or equal to ROOT / "output"
    if target == commercial_master or is_within(target, commercial_master):
        raise CommercialPolicyError(
            f"Prohibited output path '{output_dir}': benchmarks cannot write into commercial "
            f"master directory '{commercial_master}'."
        )

    # 2. Reject if any path part matches commercial deliverables markers
    parts_lower = {p.lower() for p in target.parts}
    if "commercial" in parts_lower or "commercial_master" in parts_lower:
        raise CommercialPolicyError(
            f"Prohibited output path '{output_dir}': output path contains commercial deliverables folder."
        )

    # 3. If within workspace (ROOT), must be within benchmark_reports
    if is_within(target, ROOT.resolve()):
        allowed_report_dir = (ROOT / "benchmark_reports").resolve()
        if not is_within(target, allowed_report_dir):
            raise CommercialPolicyError(
                f"Prohibited output path '{output_dir}': inside workspace, benchmarks must write "
                f"strictly to 'benchmark_reports/'."
            )

    return target


def _collect_image_files(directory: Path | str) -> list[Path]:
    """Collect sorted image file paths from a directory."""
    p = Path(directory)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Directory not found: {p}")
    valid_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted(
        [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions],
        key=lambda x: x.name,
    )
    return files


def generate_mock_cmnet2_frames(
    frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    """Generate deterministic mock colorized frames for CPU testing.

    Simulates exemplar colorization by introducing a subtle chroma shift on grayscale input.
    """
    mock_frames: list[np.ndarray] = []
    for idx, f in enumerate(frames):
        if f.ndim == 3 and f.shape[2] == 3:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        elif f.ndim == 2:
            gray = f
        else:
            gray = f[..., 0]

        # Apply simulated chroma shift in CIE Lab space
        L = (gray.astype(np.float32) / 255.0) * 100.0
        # Warm tint with subtle temporal perturbation
        a = np.full_like(L, 10.0 + 2.0 * np.sin(idx * 0.5))
        b = np.full_like(L, 15.0 + 2.0 * np.cos(idx * 0.5))

        L_u8 = np.clip(L * (255.0 / 100.0), 0, 255).astype(np.uint8)
        a_u8 = np.clip(a + 128.0, 0, 255).astype(np.uint8)
        b_u8 = np.clip(b + 128.0, 0, 255).astype(np.uint8)
        lab_cv = np.stack([L_u8, a_u8, b_u8], axis=-1)
        rgb = cv2.cvtColor(lab_cv, cv2.COLOR_Lab2RGB)
        mock_frames.append(rgb)
    return mock_frames


def compute_colorizer_metrics(
    colorized_frames: Sequence[np.ndarray],
    gt_frames: Sequence[np.ndarray],
    stride: int = 1,
) -> dict[str, Any]:
    """Compute PSNR, CIEDE2000 (mean & p90), and CDC against ground truth frames."""
    stride = max(1, int(stride))
    col_strided = list(colorized_frames[::stride])
    gt_strided = list(gt_frames[::stride])

    psnr_values: list[float] = []
    ciede_means: list[float] = []
    pixel_deltas: list[np.ndarray] = []

    for c_frame, g_frame in zip(col_strided, gt_strided):
        # 1. PSNR (Peak Signal-to-Noise Ratio in dB)
        mse = float(np.mean((g_frame.astype(np.float64) - c_frame.astype(np.float64)) ** 2))
        psnr = 100.0 if mse == 0.0 else float(10.0 * np.log10((255.0 ** 2) / mse))
        psnr_values.append(psnr)

        # 2. CIEDE2000
        lab_gt = srgb_to_lab(g_frame)
        lab_col = srgb_to_lab(c_frame)
        dE = ciede2000(lab_gt, lab_col)
        dE_arr = np.asarray(dE, dtype=np.float64)
        ciede_means.append(float(np.mean(dE_arr)))
        pixel_deltas.append(dE_arr.ravel())

    mean_psnr = float(np.mean(psnr_values)) if psnr_values else 0.0
    mean_ciede = float(np.mean(ciede_means)) if ciede_means else 0.0
    if pixel_deltas:
        p90_ciede = float(np.percentile(np.concatenate(pixel_deltas), 90.0))
    else:
        p90_ciede = 0.0

    # 3. CDC (Color Distribution Consistency)
    cdc_score = float(compute_cdc(col_strided, stride=1))

    return {
        "psnr_db": round(mean_psnr, 2),
        "mean_psnr_db": round(mean_psnr, 2),
        "mean_ciede2000": round(mean_ciede, 3),
        "p90_ciede2000": round(p90_ciede, 3),
        "cdc": round(cdc_score, 4),
        "per_frame_psnr": [round(p, 2) for p in psnr_values],
        "per_frame_ciede2000": [round(c, 3) for c in ciede_means],
    }


def generate_markdown_report(report_data: dict[str, Any]) -> str:
    """Generate GitHub Flavored Markdown summary report."""
    osi = report_data["methods"]["osi_masked_lab"]
    cmnet = report_data["methods"]["cmnet2"]
    comp = report_data["comparison"]

    lines = [
        "# Colorizer Benchmark Report (Internal Non-Commercial)",
        "",
        "> [!WARNING]",
        "> **Non-Commercial Research Only**: This benchmark compares `osi_masked_lab` against `CMNET2`.",
        "> CMNET2 utilizes weights under CC BY-NC-SA 4.0 and must never be used for commercial deliverables.",
        "",
        "## Configuration",
        f"- **Benchmark Type**: `{report_data.get('benchmark_type', 'NON_COMMERCIAL_BENCHMARK')}`",
        f"- **Frames Evaluated**: {report_data.get('frames_count', 0)}",
        f"- **Stride**: {report_data.get('stride', 1)}",
        f"- **Timestamp**: {report_data.get('timestamp', '')}",
        "",
        "## Quantitative Evaluation",
        "",
        "| Metric | `osi_masked_lab` (100% OSI) | `CMNET2` (CC BY-NC-SA 4.0) | Difference (OSI - CMNET2) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **PSNR (dB)** | {osi['psnr_db']:.2f} dB | {cmnet['psnr_db']:.2f} dB | {comp['psnr_diff_db']:+.2f} dB |",
        f"| **Mean CIEDE2000 (ΔE₀₀)** | {osi['mean_ciede2000']:.3f} | {cmnet['mean_ciede2000']:.3f} | {comp['mean_ciede2000_diff']:+.3f} |",
        f"| **p90 CIEDE2000 (ΔE₀₀)** | {osi['p90_ciede2000']:.3f} | {cmnet['p90_ciede2000']:.3f} | {comp['p90_ciede2000_diff']:+.3f} |",
        f"| **CDC (Color Consistency)** | {osi['cdc']:.4f} | {cmnet['cdc']:.4f} | {comp['cdc_diff']:+.4f} |",
        "",
        "## Summary",
        f"- **PSNR Gain**: `{comp['psnr_diff_db']:+.2f} dB` comparing OSI against CMNET2.",
        f"- **Perceptual Error**: OSI ΔE₀₀ = `{osi['mean_ciede2000']:.3f}`, CMNET2 ΔE₀₀ = `{cmnet['mean_ciede2000']:.3f}`.",
        f"- **Color Distribution Consistency**: OSI CDC = `{osi['cdc']:.4f}`, CMNET2 CDC = `{cmnet['cdc']:.4f}`.",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    frames_dir: Path | str,
    ground_truth_dir: Path | str,
    output_dir: Path | str,
    allow_non_commercial: bool = False,
    mock_cmnet2: bool = False,
    stride: int = 1,
    cmnet2_backend_fn: Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], Sequence[np.ndarray]] | None = None,
    osi_backend_fn: Callable[[Sequence[np.ndarray], Sequence[np.ndarray]], Sequence[np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Execute non-commercial colorizer benchmark comparing osi_masked_lab against CMNET2.

    Parameters:
        frames_dir: Directory containing input (grayscale) frames.
        ground_truth_dir: Directory containing ground-truth color frames.
        output_dir: Target directory for benchmark reports.
        allow_non_commercial: Explicit authorization gate for non-commercial benchmarking.
        mock_cmnet2: When True, uses simulated chroma shift on CPU.
        stride: Temporal frame sampling stride.
        cmnet2_backend_fn: Optional injectable callable for CMNET2.
        osi_backend_fn: Optional injectable callable for osi_masked_lab.

    Returns:
        Structured dictionary containing metrics, comparisons, and paths.
    """
    # 1. Compliance Guard: fail closed immediately unless allow_non_commercial is True
    if not allow_non_commercial:
        raise CommercialPolicyError(
            "Commercial Model Guard blocked non-commercial benchmark: CMNET2 contains non-commercial "
            "components (CC BY-NC-SA 4.0). Pass allow_non_commercial=True or --allow-non-commercial-benchmark "
            "to execute this internal benchmark."
        )

    # 2. Strict output path validation
    target_out = validate_benchmark_output_path(output_dir)
    target_out.mkdir(parents=True, exist_ok=True)

    # 3. Load frames
    in_paths = _collect_image_files(frames_dir)
    gt_paths = _collect_image_files(ground_truth_dir)
    count = min(len(in_paths), len(gt_paths))
    if count == 0:
        raise ValueError("Zero matching frames found between frames_dir and ground_truth_dir.")

    in_frames = [load_image_rgb(p) for p in in_paths[:count]]
    gt_frames = [load_image_rgb(p) for p in gt_paths[:count]]

    # 4. Run osi_masked_lab
    if osi_backend_fn is not None:
        osi_colorized = list(osi_backend_fn(in_frames, gt_frames))
    else:
        from scripts.osi_masked_lab import colorize_shot_osi_masked_lab
        osi_colorized = colorize_shot_osi_masked_lab(
            source=in_frames,
            keyframes={0: gt_frames[0]},
        )

    # 5. Run CMNET2
    if cmnet2_backend_fn is not None:
        cmnet2_colorized = list(cmnet2_backend_fn(in_frames, gt_frames))
    elif mock_cmnet2:
        cmnet2_colorized = generate_mock_cmnet2_frames(in_frames, gt_frames)
    else:
        try:
            from scripts.cmnet2_runtime import CMNet2Session  # noqa: F401
            raise RuntimeError("CMNet2 runtime requires GPU and downloaded weights.")
        except Exception as exc:
            raise RuntimeError(
                f"CMNET2 runtime or weights unavailable: {exc}. Use --mock-cmnet2 for CPU mock mode."
            ) from exc

    # 6. Compute metrics
    osi_metrics = compute_colorizer_metrics(osi_colorized, gt_frames, stride=stride)
    cmnet2_metrics = compute_colorizer_metrics(cmnet2_colorized, gt_frames, stride=stride)
    cmnet2_metrics["mock"] = mock_cmnet2 or (cmnet2_backend_fn is not None)

    comparison = {
        "psnr_diff_db": round(osi_metrics["psnr_db"] - cmnet2_metrics["psnr_db"], 2),
        "mean_ciede2000_diff": round(osi_metrics["mean_ciede2000"] - cmnet2_metrics["mean_ciede2000"], 3),
        "p90_ciede2000_diff": round(osi_metrics["p90_ciede2000"] - cmnet2_metrics["p90_ciede2000"], 3),
        "cdc_diff": round(osi_metrics["cdc"] - cmnet2_metrics["cdc"], 4),
    }

    timestamp_str = datetime.now(timezone.utc).isoformat()
    report_data = {
        "timestamp": timestamp_str,
        "benchmark_type": "NON_COMMERCIAL_BENCHMARK",
        "frames_count": count,
        "stride": stride,
        "methods": {
            "osi_masked_lab": osi_metrics,
            "cmnet2": cmnet2_metrics,
        },
        "metrics": {
            "osi_masked_lab": osi_metrics,
            "cmnet2": cmnet2_metrics,
        },
        "osi_masked_lab": osi_metrics,
        "cmnet2": cmnet2_metrics,
        "comparison": comparison,
    }

    # 7. Write report.json and report.md
    report_json_path = target_out / "report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    report_md_path = target_out / "report.md"
    markdown_content = generate_markdown_report(report_data)
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # 8. Record compliance audit entry if append_compliance_record is available
    if append_compliance_record is not None:
        compliance_models = [
            {
                "model_id": "CMNET2",
                "component_type": "colorizer",
                "status": "hard_blocked",
                "reason": "NON_COMMERCIAL_BENCHMARK",
                "license": "CC BY-NC-SA 4.0",
            },
            {
                "model_id": "osi_masked_lab",
                "component_type": "colorizer",
                "status": "allowed_osi",
                "reason": "100% OSI-compliant guided filter Lab propagation",
                "license": "Apache-2.0",
            },
        ]
        append_compliance_record(
            output_dir=target_out,
            stage="benchmark_colorizers",
            models=compliance_models,
            settings={
                "global": {
                    "guard_mode": "benchmark_override",
                    "non_commercial_benchmark_override": "true",
                    "commercial_model_guard": "false",
                }
            },
            metadata={
                "benchmark_type": "NON_COMMERCIAL_BENCHMARK",
                "allow_non_commercial": True,
                "mock_cmnet2": mock_cmnet2,
            },
        )

    return report_data


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Internal Non-Commercial Benchmark: osi_masked_lab vs CMNET2."
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Path to directory containing input frames.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=str,
        required=True,
        help="Path to directory containing ground-truth color frames.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "benchmark_reports" / "colorizers"),
        help="Target directory for benchmark reports (default: benchmark_reports/colorizers).",
    )
    parser.add_argument(
        "--allow-non-commercial-benchmark",
        action="store_true",
        default=False,
        help="Explicitly permit non-commercial benchmarking of CMNET2.",
    )
    parser.add_argument(
        "--mock-cmnet2",
        action="store_true",
        default=False,
        help="Run CMNET2 mock mode on CPU with simulated chroma shift.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Temporal frame evaluation stride (default: 1).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.allow_non_commercial_benchmark:
        raise CommercialPolicyError(
            "Commercial Model Guard blocked benchmark: CMNET2 contains non-commercial components "
            "(CC BY-NC-SA 4.0). You must pass --allow-non-commercial-benchmark to run."
        )

    run_benchmark(
        frames_dir=args.frames_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_dir=args.output_dir,
        allow_non_commercial=args.allow_non_commercial_benchmark,
        mock_cmnet2=args.mock_cmnet2,
        stride=args.stride,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
