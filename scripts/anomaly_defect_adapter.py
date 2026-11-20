"""Neural Anomaly Detectors Adapter Bridge.

Connects PatchCore and PaDiM feature memory-bank anomaly detectors
to ARP's defect detector registry and MultiChannelDefectMask contract.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.film_defect_detectors import (
        BaseDefectDetector,
        ChannelOrigin,
        ChannelProvenance,
        DefectMaskResult,
        MultiChannelDefectMask,
        ReadinessState,
        normalize_frames,
    )
except ImportError:
    from film_defect_detectors import (
        BaseDefectDetector,
        ChannelOrigin,
        ChannelProvenance,
        DefectMaskResult,
        MultiChannelDefectMask,
        ReadinessState,
        normalize_frames,
    )

MANAGED_ANOMALIB_DIR = ROOT / "tools/anomalib"


def find_venv_python() -> Path:
    """Resolve project virtual environment python executable."""
    candidates = [
        ROOT / ".venv/Scripts/python.exe",
        ROOT / ".venv/bin/python",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return Path(sys.executable)


# ---------------------------------------------------------------------------
# 1. PatchCore Anomaly Detector Adapter
# ---------------------------------------------------------------------------

class PatchCoreDefectAdapter(BaseDefectDetector):
    """Adapter for PatchCore Neural Anomaly Detector (Roth et al., CVPR 2022).

    Extracts deep feature memory banks and localizes visual film defects as
    statistical anomalies. Produces native combined defect proposal masks with
    continuous anomaly scores. Scratch and Dust channels are marked unsupported.
    """

    name = "patchcore"
    aliases = ("patchcore_anomaly_detector", "anomalib_patchcore")
    description = "PatchCore deep feature memory-bank anomaly detector (Recovered: high-resolution layer1+layer2 representation with calibrated archival core-set memory bank)."
    license_status = "Permissive (Apache-2.0 / BSD-3-Clause) · Commercial Cleared"
    commercial_cleared = True
    temporal_radius = 0
    readiness_state = ReadinessState.TRAINED_EXPERIMENTAL
    supported_channels = ("combined",)
    unsupported_channels = ("scratch", "dust", "uncertain")

    def __init__(self, tool_dir: Path = MANAGED_ANOMALIB_DIR) -> None:
        self.tool_dir = Path(tool_dir).resolve()
        self.runner_script = self.tool_dir / "run_anomaly_detector.py"
        self.weights_path = ROOT / "tools/checkpoints/patchcore/patchcore_memory_bank.pt"
        self.manifest_path = ROOT / "tools/checkpoints/manifests/patchcore.json"
        self._cached_detector: Any = None
        self._cached_device: str | None = None

    def _get_detector(self, device: str, threshold: float | None = None) -> Any:
        try:
            import torch
            from tools.anomalib.models import load_anomaly_detector
        except ImportError:
            return None

        dev_str = "cuda" if (device in ("auto", "") and torch.cuda.is_available()) or device.startswith("cuda") else "cpu"
        if self._cached_detector is None or self._cached_device != dev_str:
            self._cached_detector = load_anomaly_detector(self.weights_path, model_name="patchcore", device=dev_str)
            self._cached_device = dev_str
        if threshold is not None:
            self._cached_detector.threshold = float(threshold)
        return self._cached_detector

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "threshold": {
                "type": "float",
                "default": 0.75,
                "min": 0.0,
                "max": 1.0,
                "description": "Anomaly score binarization threshold.",
            },
            "tile_size": {
                "type": "int",
                "default": 0,
                "min": 0,
                "max": 2048,
                "description": "Tile size in pixels (0 for full-frame).",
            },
            "overlap": {
                "type": "int",
                "default": 64,
                "min": 0,
                "max": 512,
                "description": "Tile overlap in pixels with cosine blending.",
            },
            "sigma": {
                "type": "float",
                "default": 2.0,
                "min": 0.5,
                "max": 20.0,
                "description": "Gaussian smoothing sigma.",
            },
            "device": {
                "type": "str",
                "default": "auto",
                "description": "Target compute device ('auto', 'cuda', 'cpu').",
            },
        }

    def check_readiness(self) -> dict[str, Any]:
        ready = self.weights_path.is_file() and self.runner_script.is_file()
        issues: list[str] = []
        if not self.weights_path.is_file():
            issues.append(f"Memory bank missing at {self.weights_path}")
        if not self.runner_script.is_file():
            issues.append(f"Runner script missing at {self.runner_script}")

        return {
            "ready": ready,
            "readiness_state": self.readiness_state.value if ready else ReadinessState.TRAINING_REQUIRED.value,
            "commercial_cleared": self.commercial_cleared,
            "weights_present": self.weights_path.is_file(),
            "weights_path": str(self.weights_path) if self.weights_path.is_file() else None,
            "manifest_present": self.manifest_path.is_file(),
            "issues": issues,
        }

    def detect_defects(self, frame: np.ndarray, options: dict[str, Any] | None = None) -> MultiChannelDefectMask:
        res = self.detect([frame], options)
        return res.multi_channel_masks[0]

    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        params = self.validate_config(config)
        threshold = float(params["threshold"])
        tile_size = int(params["tile_size"])
        overlap = int(params["overlap"])
        sigma = float(params["sigma"])
        device = str(params["device"]).strip()

        seq = normalize_frames(frames)
        n = len(seq.bgr_frames)
        h, w = seq.height, seq.width

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"PatchCore checkpoint missing: {self.weights_path}")

        detector = self._get_detector(device, threshold)
        if detector is not None:
            from tools.anomalib.run_anomaly_detector import predict_frame
            combined_masks: list[np.ndarray] = []
            scratch_masks: list[np.ndarray] = []
            dust_masks: list[np.ndarray] = []
            uncertain_masks: list[np.ndarray] = []
            mcdm_list: list[MultiChannelDefectMask] = []

            for idx, f in enumerate(seq.bgr_frames):
                c_prob, c_mask = predict_frame(
                    detector=detector,
                    img_bgr=f,
                    tile_size=tile_size,
                    overlap=overlap,
                    gaussian_sigma=sigma,
                )
                c_mask = np.where(c_mask >= 128, np.uint8(255), np.uint8(0))
                c_prob = np.clip(c_prob, 0.0, 1.0).astype(np.float32)

                prov = {
                    "combined": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="patchcore_anomaly_scoring",
                        confidence_threshold=threshold,
                    ),
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "uncertain": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                }

                mcdm = MultiChannelDefectMask(
                    combined_probability=c_prob,
                    combined_mask=c_mask,
                    scratch_mask=None,
                    dust_mask=None,
                    uncertain_mask=None,
                    provenance=prov,
                    metadata={"model": "patchcore", "index": idx},
                )
                combined_masks.append(c_mask)
                scratch_masks.append(np.zeros((h, w), dtype=np.uint8))
                dust_masks.append(np.zeros((h, w), dtype=np.uint8))
                uncertain_masks.append(np.zeros((h, w), dtype=np.uint8))
                mcdm_list.append(mcdm)

            return DefectMaskResult(
                mask=combined_masks,
                scratch_mask=scratch_masks,
                dust_mask=dust_masks,
                uncertain_mask=uncertain_masks,
                multi_channel_masks=mcdm_list,
                metadata={"detector": self.name, "device": str(detector.device)},
            )

        py = find_venv_python()
        with tempfile.TemporaryDirectory(prefix="patchcore-run-") as tmp:
            tpath = Path(tmp)
            in_dir = tpath / "in"
            out_dir = tpath / "out"
            in_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)

            for idx, f in enumerate(seq.bgr_frames):
                cv2.imwrite(str(in_dir / f"frame_{idx:06d}.png"), f)

            cmd = [
                str(py),
                str(self.runner_script),
                "--input", str(in_dir),
                "--output-dir", str(out_dir),
                "--model-name", "patchcore",
                "--weights", str(self.weights_path),
                "--threshold", str(threshold),
                "--tile-size", str(tile_size),
                "--overlap", str(overlap),
                "--sigma", str(sigma),
                "--device", device,
                "--quiet",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(300, n * 30))
            if proc.returncode != 0:
                raise RuntimeError(f"PatchCore runner failed ({proc.returncode}):\n{proc.stderr}")

            masks_dir = out_dir / "masks"
            prob_dir = out_dir / "probabilities"

            combined_masks: list[np.ndarray] = []
            scratch_masks: list[np.ndarray] = []
            dust_masks: list[np.ndarray] = []
            uncertain_masks: list[np.ndarray] = []
            mcdm_list: list[MultiChannelDefectMask] = []

            for idx in range(n):
                stem = f"frame_{idx:06d}"
                c_mask_f = masks_dir / f"{stem}.png"
                c_prob_f = prob_dir / f"{stem}.npy"

                c_mask = cv2.imread(str(c_mask_f), cv2.IMREAD_UNCHANGED)
                c_prob = np.load(str(c_prob_f)).astype(np.float32)

                c_mask = np.where(c_mask >= 128, np.uint8(255), np.uint8(0))
                c_prob = np.clip(c_prob, 0.0, 1.0).astype(np.float32)

                prov = {
                    "combined": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="padim_mahalanobis_distance",
                        confidence_threshold=threshold,
                    ),
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "uncertain": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                }

                mcdm = MultiChannelDefectMask(
                    combined_probability=c_prob,
                    combined_mask=c_mask,
                    scratch_mask=None,
                    dust_mask=None,
                    uncertain_mask=None,
                    provenance=prov,
                    metadata={"model": "padim", "index": idx},
                )

                combined_masks.append(c_mask)
                scratch_masks.append(np.zeros((h, w), dtype=np.uint8))
                dust_masks.append(np.zeros((h, w), dtype=np.uint8))
                uncertain_masks.append(np.zeros((h, w), dtype=np.uint8))
                mcdm_list.append(mcdm)

            return DefectMaskResult(
                mask=combined_masks,
                scratch_mask=scratch_masks,
                dust_mask=dust_masks,
                uncertain_mask=uncertain_masks,
                multi_channel_masks=mcdm_list,
                metadata={"detector": self.name, "device": device},
            )


# ---------------------------------------------------------------------------
# 2. PaDiM Anomaly Detector Adapter
# ---------------------------------------------------------------------------

class PaDiMDefectAdapter(BaseDefectDetector):
    """Adapter for PaDiM Anomaly Detector (Defard et al., ICPR 2021).

    Fits position-wise multivariate Gaussian distributions on clean film patches
    and localizes defects via Mahalanobis distance.
    """

    name = "padim"
    aliases = ("padim_anomaly_detector", "anomalib_padim")
    description = "PaDiM multivariate Gaussian anomaly detector (Recovered: spatially invariant distribution modeling with regularized covariance)."
    license_status = "Permissive (Apache-2.0 / BSD-3-Clause) · Commercial Cleared"
    commercial_cleared = True
    temporal_radius = 0
    readiness_state = ReadinessState.TRAINED_EXPERIMENTAL
    supported_channels = ("combined",)
    unsupported_channels = ("scratch", "dust", "uncertain")

    def __init__(self, tool_dir: Path = MANAGED_ANOMALIB_DIR) -> None:
        self.tool_dir = Path(tool_dir).resolve()
        self.runner_script = self.tool_dir / "run_anomaly_detector.py"
        self.weights_path = ROOT / "tools/checkpoints/padim/padim_distribution.pt"
        self.manifest_path = ROOT / "tools/checkpoints/manifests/padim.json"
        self._cached_detector: Any = None
        self._cached_device: str | None = None

    def _get_detector(self, device: str, threshold: float | None = None) -> Any:
        try:
            import torch
            from tools.anomalib.models import load_anomaly_detector
        except ImportError:
            return None

        dev_str = "cuda" if (device in ("auto", "") and torch.cuda.is_available()) or device.startswith("cuda") else "cpu"
        if self._cached_detector is None or self._cached_device != dev_str:
            self._cached_detector = load_anomaly_detector(self.weights_path, model_name="padim", device=dev_str)
            self._cached_device = dev_str
        if threshold is not None:
            self._cached_detector.threshold = float(threshold)
        return self._cached_detector

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "threshold": {"type": "float", "default": 0.55, "min": 0.0, "max": 1.0},
            "tile_size": {"type": "int", "default": 0, "min": 0, "max": 2048},
            "overlap": {"type": "int", "default": 64, "min": 0, "max": 512},
            "sigma": {"type": "float", "default": 2.0, "min": 0.5, "max": 20.0},
            "device": {"type": "str", "default": "auto"},
        }

    def check_readiness(self) -> dict[str, Any]:
        ready = self.weights_path.is_file() and self.runner_script.is_file()
        return {
            "ready": ready,
            "readiness_state": self.readiness_state.value if ready else ReadinessState.TRAINING_REQUIRED.value,
            "commercial_cleared": self.commercial_cleared,
            "weights_present": self.weights_path.is_file(),
            "manifest_present": self.manifest_path.is_file(),
            "issues": [] if ready else ["Distribution checkpoint or runner script missing"],
        }

    def detect_defects(self, frame: np.ndarray, options: dict[str, Any] | None = None) -> MultiChannelDefectMask:
        res = self.detect([frame], options)
        return res.multi_channel_masks[0]

    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        params = self.validate_config(config)
        threshold = float(params["threshold"])
        tile_size = int(params["tile_size"])
        overlap = int(params["overlap"])
        sigma = float(params["sigma"])
        device = str(params["device"]).strip()

        seq = normalize_frames(frames)
        n = len(seq.bgr_frames)
        h, w = seq.height, seq.width

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"PaDiM checkpoint missing: {self.weights_path}")

        detector = self._get_detector(device, threshold)
        if detector is not None:
            from tools.anomalib.run_anomaly_detector import predict_frame
            combined_masks: list[np.ndarray] = []
            scratch_masks: list[np.ndarray] = []
            dust_masks: list[np.ndarray] = []
            uncertain_masks: list[np.ndarray] = []
            mcdm_list: list[MultiChannelDefectMask] = []

            for idx, f in enumerate(seq.bgr_frames):
                c_prob, c_mask = predict_frame(
                    detector=detector,
                    img_bgr=f,
                    tile_size=tile_size,
                    overlap=overlap,
                    gaussian_sigma=sigma,
                )
                c_mask = np.where(c_mask >= 128, np.uint8(255), np.uint8(0))
                c_prob = np.clip(c_prob, 0.0, 1.0).astype(np.float32)

                prov = {
                    "combined": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="padim_mahalanobis_distance",
                        confidence_threshold=threshold,
                    ),
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "uncertain": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                }

                mcdm = MultiChannelDefectMask(
                    combined_probability=c_prob,
                    combined_mask=c_mask,
                    scratch_mask=None,
                    dust_mask=None,
                    uncertain_mask=None,
                    provenance=prov,
                    metadata={"model": "padim", "index": idx},
                )
                combined_masks.append(c_mask)
                scratch_masks.append(np.zeros((h, w), dtype=np.uint8))
                dust_masks.append(np.zeros((h, w), dtype=np.uint8))
                uncertain_masks.append(np.zeros((h, w), dtype=np.uint8))
                mcdm_list.append(mcdm)

            return DefectMaskResult(
                mask=combined_masks,
                scratch_mask=scratch_masks,
                dust_mask=dust_masks,
                uncertain_mask=uncertain_masks,
                multi_channel_masks=mcdm_list,
                metadata={"detector": self.name, "device": str(detector.device)},
            )

        py = find_venv_python()
        with tempfile.TemporaryDirectory(prefix="padim-run-") as tmp:
            tpath = Path(tmp)
            in_dir = tpath / "in"
            out_dir = tpath / "out"
            in_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)

            for idx, f in enumerate(seq.bgr_frames):
                cv2.imwrite(str(in_dir / f"frame_{idx:06d}.png"), f)

            cmd = [
                str(py),
                str(self.runner_script),
                "--input", str(in_dir),
                "--output-dir", str(out_dir),
                "--model-name", "padim",
                "--weights", str(self.weights_path),
                "--threshold", str(threshold),
                "--tile-size", str(tile_size),
                "--overlap", str(overlap),
                "--sigma", str(sigma),
                "--device", device,
                "--quiet",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(300, n * 30))
            if proc.returncode != 0:
                raise RuntimeError(f"PaDiM runner failed ({proc.returncode}):\n{proc.stderr}")

            masks_dir = out_dir / "masks"
            prob_dir = out_dir / "probabilities"

            combined_masks: list[np.ndarray] = []
            scratch_masks: list[np.ndarray] = []
            dust_masks: list[np.ndarray] = []
            uncertain_masks: list[np.ndarray] = []
            mcdm_list: list[MultiChannelDefectMask] = []

            for idx in range(n):
                stem = f"frame_{idx:06d}"
                c_mask_f = masks_dir / f"{stem}.png"
                c_prob_f = prob_dir / f"{stem}.npy"

                c_mask = cv2.imread(str(c_mask_f), cv2.IMREAD_UNCHANGED)
                c_prob = np.load(str(c_prob_f)).astype(np.float32)

                c_mask = np.where(c_mask >= 128, np.uint8(255), np.uint8(0))
                c_prob = np.clip(c_prob, 0.0, 1.0).astype(np.float32)

                prov = {
                    "combined": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="padim_mahalanobis_distance",
                        confidence_threshold=threshold,
                    ),
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                    "uncertain": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="unsupported",
                    ),
                }

                mcdm = MultiChannelDefectMask(
                    combined_probability=c_prob,
                    combined_mask=c_mask,
                    scratch_mask=None,
                    dust_mask=None,
                    uncertain_mask=None,
                    provenance=prov,
                    metadata={"model": "padim", "index": idx},
                )

                combined_masks.append(c_mask)
                scratch_masks.append(np.zeros((h, w), dtype=np.uint8))
                dust_masks.append(np.zeros((h, w), dtype=np.uint8))
                uncertain_masks.append(np.zeros((h, w), dtype=np.uint8))
                mcdm_list.append(mcdm)

            return DefectMaskResult(
                mask=combined_masks,
                scratch_mask=scratch_masks,
                dust_mask=dust_masks,
                uncertain_mask=uncertain_masks,
                multi_channel_masks=mcdm_list,
                metadata={"detector": self.name, "device": device},
            )


def register_anomaly_detectors(registry: Any = None) -> None:
    """Register anomaly defect adapters into the provided or default registry."""
    try:
        from scripts.film_defect_detectors import DEFAULT_REGISTRY, register_defect_detector
    except ImportError:
        from film_defect_detectors import DEFAULT_REGISTRY, register_defect_detector

    target = registry or DEFAULT_REGISTRY
    target.register(PatchCoreDefectAdapter(), replace=True)
    target.register(PaDiMDefectAdapter(), replace=True)

