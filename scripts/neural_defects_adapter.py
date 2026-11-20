"""Neural Defect Detectors Adapter Bridge.

Connects clean-room deep learning defect models (FilmArtefactSegmenter,
CNNBlotchesScratches, CorrelNet) to ARP's defect detector registry and
MultiChannelDefectMask contract.
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

MANAGED_NEURAL_DIR = ROOT / "tools/neural_defects"


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
# 1. Film Artefact Segmenter Adapter
# ---------------------------------------------------------------------------

class FilmArtefactSegmenterAdapter(BaseDefectDetector):
    """Adapter for Film Artefact Segmenter (Ivanova et al., 2023 / FilmDamageSimulator).

    Executes multi-task neural defect segmentation. Outputs native Scratch and Dust
    defect channels with continuous probability maps and binary masks adhering to
    ARP's MultiChannelDefectMask contract.
    """

    name = "film_artefact_segmenter"
    aliases = ("fas", "film_artefact_segmentation", "ivanova_segmenter")
    description = "Multi-head neural network for film scratches and dust/dirt segmentation (commercial-safe)."
    license_status = "Permissive (MIT Clean-room reproduction, commercial-cleared)"
    commercial_cleared = True
    temporal_radius = 0
    readiness_state = ReadinessState.PRETRAINED_OPERATIONAL
    supported_channels = ("scratch", "dust")
    unsupported_channels = ("uncertain",)

    def __init__(self, tool_dir: Path = MANAGED_NEURAL_DIR) -> None:
        self.tool_dir = Path(tool_dir).resolve()
        self.runner_script = self.tool_dir / "run_neural_detector.py"
        self.weights_path = ROOT / "tools/checkpoints/film_artefact_segmenter/film_artefact_segmenter.pt"
        self.manifest_path = ROOT / "tools/checkpoints/manifests/film_artefact_segmenter.json"
        self._cached_model = None
        self._cached_device = None

    def _get_model(self, device: Any) -> Any:
        if self._cached_model is None or self._cached_device != device:
            try:
                from tools.neural_defects.models import load_neural_model
            except ImportError:
                sys.path.insert(0, str(ROOT / "tools/neural_defects"))
                from tools.neural_defects.models import load_neural_model
            self._cached_model = load_neural_model(
                str(self.weights_path),
                model_name=self.name,
                device=device,
            )
            self._cached_device = device
        return self._cached_model

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "General defect binarization threshold.",
            },
            "scratch_threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Scratch channel binarization threshold.",
            },
            "dust_threshold": {
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "description": "Dust channel binarization threshold.",
            },
            "tile_size": {
                "type": "int",
                "default": 256,
                "min": 0,
                "max": 2048,
                "description": "Tile size in pixels (0 for full frame).",
            },
            "overlap": {
                "type": "int",
                "default": 64,
                "min": 0,
                "max": 512,
                "description": "Tile overlap in pixels with cosine blending.",
            },
            "device": {
                "type": "str",
                "default": "auto",
                "description": "Target compute device ('auto', 'cuda', 'cpu').",
            },
        }

    def check_readiness(self) -> dict[str, Any]:
        """Verify weights presence and operational readiness."""
        weights_present = self.weights_path.is_file()
        manifest_present = self.manifest_path.is_file()
        ready = weights_present and self.runner_script.is_file()
        issues: list[str] = []
        if not weights_present:
            issues.append(f"Checkpoint weights missing at {self.weights_path}")
        if not self.runner_script.is_file():
            issues.append(f"Runner script missing at {self.runner_script}")

        return {
            "ready": ready,
            "readiness_state": self.readiness_state.value if ready else ReadinessState.TRAINING_REQUIRED.value,
            "commercial_cleared": self.commercial_cleared,
            "weights_present": weights_present,
            "weights_path": str(self.weights_path) if weights_present else None,
            "manifest_present": manifest_present,
            "issues": issues,
        }

    def detect_defects(
        self,
        frame: np.ndarray,
        options: dict[str, Any] | None = None,
    ) -> MultiChannelDefectMask:
        """Run single-frame detection and return MultiChannelDefectMask."""
        res = self.detect([frame], options)
        if not res.multi_channel_masks:
            raise RuntimeError("FilmArtefactSegmenter produced no masks.")
        return res.multi_channel_masks[0]

    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        """Execute detection across frame sequence."""
        params = self.validate_config(config)
        threshold = float(params["threshold"])
        scratch_threshold = float(params.get("scratch_threshold", threshold))
        dust_threshold = float(params.get("dust_threshold", threshold))
        tile_size = int(params["tile_size"])
        overlap = int(params["overlap"])
        device = str(params["device"]).strip()

        seq = normalize_frames(frames)
        n = len(seq.bgr_frames)
        h, w = seq.height, seq.width

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {self.weights_path}")

        # High-performance in-process execution with cached model
        target_device = None
        try:
            import torch
            clean_dev = device.strip().lower()
            if clean_dev in ("auto", ""):
                if torch.cuda.is_available():
                    target_device = torch.device("cuda")
                else:
                    raise RuntimeError("Film Artefact Segmenter auto device requires CUDA.")
            elif clean_dev == "cuda" or clean_dev.startswith("cuda:"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested for Film Artefact Segmenter but not available.")
                target_device = torch.device(clean_dev)
            elif clean_dev == "cpu":
                target_device = torch.device("cpu")
        except Exception:
            target_device = None

        if target_device is not None:
            try:
                try:
                    from tools.neural_defects.run_neural_detector import run_tiled_inference
                except ImportError:
                    sys.path.insert(0, str(ROOT / "tools/neural_defects"))
                    from tools.neural_defects.run_neural_detector import run_tiled_inference
                model = self._get_model(target_device)
                combined_masks = []
                scratch_masks = []
                dust_masks = []
                uncertain_masks = []
                mcdm_list = []
                empty_mask = np.zeros((h, w), dtype=np.uint8)

                for idx, frame in enumerate(seq.bgr_frames):
                    s_prob, d_prob, c_prob = run_tiled_inference(
                        model=model,
                        img_bgr=frame,
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=16,
                        device=target_device,
                    )
                    s_mask = np.where(s_prob >= scratch_threshold, np.uint8(255), np.uint8(0))
                    d_mask = np.where(d_prob >= dust_threshold, np.uint8(255), np.uint8(0))
                    c_mask = np.where(c_prob >= threshold, np.uint8(255), np.uint8(0))

                    prov = {
                        "scratch": ChannelProvenance(
                            origin=ChannelOrigin.NATIVE,
                            is_available=True,
                            source_algorithm="film_artefact_segmenter_scratch_head",
                            confidence_threshold=scratch_threshold,
                        ),
                        "dust": ChannelProvenance(
                            origin=ChannelOrigin.NATIVE,
                            is_available=True,
                            source_algorithm="film_artefact_segmenter_dust_head",
                            confidence_threshold=dust_threshold,
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
                        scratch_mask=s_mask,
                        dust_mask=d_mask,
                        uncertain_mask=None,
                        provenance=prov,
                        metadata={"model": "film_artefact_segmenter", "index": idx},
                    )

                    combined_masks.append(c_mask)
                    scratch_masks.append(s_mask)
                    dust_masks.append(d_mask)
                    uncertain_masks.append(empty_mask.copy())
                    mcdm_list.append(mcdm)

                return DefectMaskResult(
                    mask=combined_masks,
                    scratch_mask=scratch_masks,
                    dust_mask=dust_masks,
                    uncertain_mask=uncertain_masks,
                    multi_channel_masks=mcdm_list,
                    metadata={"detector": self.name, "device": str(target_device)},
                )
            except Exception:
                pass

        py = find_venv_python()
        with tempfile.TemporaryDirectory(prefix="fas-run-") as tmp:
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
                "--model-name", "film_artefact_segmenter",
                "--weights", str(self.weights_path),
                "--threshold", str(threshold),
                "--scratch-threshold", str(scratch_threshold),
                "--dust-threshold", str(dust_threshold),
                "--tile-size", str(tile_size),
                "--overlap", str(overlap),
                "--device", device,
                "--quiet",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(300, n * 30))
            if proc.returncode != 0:
                raise RuntimeError(f"FilmArtefactSegmenter runner failed ({proc.returncode}):\n{proc.stderr}")

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
                s_mask_f = masks_dir / f"{stem}_scratch.png"
                d_mask_f = masks_dir / f"{stem}_dust.png"
                c_prob_f = prob_dir / f"{stem}.npy"

                c_mask = cv2.imread(str(c_mask_f), cv2.IMREAD_UNCHANGED)
                s_mask = cv2.imread(str(s_mask_f), cv2.IMREAD_UNCHANGED)
                d_mask = cv2.imread(str(d_mask_f), cv2.IMREAD_UNCHANGED)
                c_prob = np.load(str(c_prob_f)).astype(np.float32)

                # Strictly enforce uint8 {0, 255}
                c_mask = np.where(c_mask >= 128, np.uint8(255), np.uint8(0))
                s_mask = np.where(s_mask >= 128, np.uint8(255), np.uint8(0))
                d_mask = np.where(d_mask >= 128, np.uint8(255), np.uint8(0))
                c_prob = np.clip(c_prob, 0.0, 1.0).astype(np.float32)

                prov = {
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="film_artefact_segmenter_scratch_head",
                        confidence_threshold=scratch_threshold,
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="film_artefact_segmenter_dust_head",
                        confidence_threshold=dust_threshold,
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
                    scratch_mask=s_mask,
                    dust_mask=d_mask,
                    uncertain_mask=None,
                    provenance=prov,
                    metadata={"model": "film_artefact_segmenter", "index": idx},
                )

                combined_masks.append(c_mask)
                scratch_masks.append(s_mask)
                dust_masks.append(d_mask)
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
# 2. CNN Blotches & Scratches Adapter
# ---------------------------------------------------------------------------

class CNNBlotchesDetectorAdapter(BaseDefectDetector):
    """Adapter for CNN Blotches and Scratches (Yous & Serir).

    Dual-branch CNN with anisotropic kernels for scratches and isotropic kernels
    for archival blotches.
    """

    name = "cnn_blotches"
    aliases = ("cnn_blotches_scratches", "yous_serir_cnn")
    description = "Dual-branch anisotropic CNN for film blotches and scratches (commercial-safe)."
    license_status = "Permissive (MIT Clean-room reproduction, commercial-cleared)"
    commercial_cleared = True
    temporal_radius = 0
    readiness_state = ReadinessState.PRETRAINED_OPERATIONAL
    supported_channels = ("scratch", "dust")
    unsupported_channels = ("uncertain",)

    def __init__(self, tool_dir: Path = MANAGED_NEURAL_DIR) -> None:
        self.tool_dir = Path(tool_dir).resolve()
        self.runner_script = self.tool_dir / "run_neural_detector.py"
        self.weights_path = ROOT / "tools/checkpoints/cnn_blotches/cnn_blotches.pt"
        self.manifest_path = ROOT / "tools/checkpoints/manifests/cnn_blotches.json"
        self._cached_model = None
        self._cached_device = None

    def _get_model(self, device: Any) -> Any:
        if self._cached_model is None or self._cached_device != device:
            try:
                from tools.neural_defects.models import load_neural_model
            except ImportError:
                sys.path.insert(0, str(ROOT / "tools/neural_defects"))
                from tools.neural_defects.models import load_neural_model
            self._cached_model = load_neural_model(
                str(self.weights_path),
                model_name=self.name,
                device=device,
            )
            self._cached_device = device
        return self._cached_model

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "scratch_threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "dust_threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "tile_size": {"type": "int", "default": 256, "min": 0, "max": 2048},
            "overlap": {"type": "int", "default": 64, "min": 0, "max": 512},
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
            "issues": [] if ready else ["Checkpoint weights or runner script missing"],
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
        scratch_threshold = float(params.get("scratch_threshold", threshold))
        dust_threshold = float(params.get("dust_threshold", threshold))
        tile_size = int(params["tile_size"])
        overlap = int(params["overlap"])
        device = str(params["device"]).strip()

        seq = normalize_frames(frames)
        n = len(seq.bgr_frames)
        h, w = seq.height, seq.width

        if not self.weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint missing: {self.weights_path}")

        # High-performance in-process execution with cached model
        target_device = None
        try:
            import torch
            clean_dev = device.strip().lower()
            if clean_dev in ("auto", ""):
                if torch.cuda.is_available():
                    target_device = torch.device("cuda")
                else:
                    raise RuntimeError("CNN Blotches auto device requires CUDA.")
            elif clean_dev == "cuda" or clean_dev.startswith("cuda:"):
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested for CNN Blotches but not available.")
                target_device = torch.device(clean_dev)
            elif clean_dev == "cpu":
                target_device = torch.device("cpu")
        except Exception:
            target_device = None

        if target_device is not None:
            try:
                try:
                    from tools.neural_defects.run_neural_detector import run_tiled_inference
                except ImportError:
                    sys.path.insert(0, str(ROOT / "tools/neural_defects"))
                    from tools.neural_defects.run_neural_detector import run_tiled_inference
                model = self._get_model(target_device)
                combined_masks = []
                scratch_masks = []
                dust_masks = []
                uncertain_masks = []
                mcdm_list = []
                empty_mask = np.zeros((h, w), dtype=np.uint8)

                for idx, frame in enumerate(seq.bgr_frames):
                    s_prob, d_prob, c_prob = run_tiled_inference(
                        model=model,
                        img_bgr=frame,
                        tile_size=tile_size,
                        overlap=overlap,
                        batch_size=16,
                        device=target_device,
                    )
                    s_mask = np.where(s_prob >= scratch_threshold, np.uint8(255), np.uint8(0))
                    d_mask = np.where(d_prob >= dust_threshold, np.uint8(255), np.uint8(0))
                    c_mask = np.where(c_prob >= threshold, np.uint8(255), np.uint8(0))

                    prov = {
                        "scratch": ChannelProvenance(
                            origin=ChannelOrigin.NATIVE,
                            is_available=True,
                            source_algorithm="cnn_blotches_anisotropic_head",
                            confidence_threshold=scratch_threshold,
                        ),
                        "dust": ChannelProvenance(
                            origin=ChannelOrigin.NATIVE,
                            is_available=True,
                            source_algorithm="cnn_blotches_blotch_head",
                            confidence_threshold=dust_threshold,
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
                        scratch_mask=s_mask,
                        dust_mask=d_mask,
                        uncertain_mask=None,
                        provenance=prov,
                        metadata={"model": "cnn_blotches", "index": idx},
                    )

                    combined_masks.append(c_mask)
                    scratch_masks.append(s_mask)
                    dust_masks.append(d_mask)
                    uncertain_masks.append(empty_mask.copy())
                    mcdm_list.append(mcdm)

                return DefectMaskResult(
                    mask=combined_masks,
                    scratch_mask=scratch_masks,
                    dust_mask=dust_masks,
                    uncertain_mask=uncertain_masks,
                    multi_channel_masks=mcdm_list,
                    metadata={"detector": self.name, "device": str(target_device)},
                )
            except Exception:
                pass

        py = find_venv_python()
        with tempfile.TemporaryDirectory(prefix="cnn-blotches-run-") as tmp:
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
                "--model-name", "cnn_blotches",
                "--weights", str(self.weights_path),
                "--threshold", str(threshold),
                "--scratch-threshold", str(scratch_threshold),
                "--dust-threshold", str(dust_threshold),
                "--tile-size", str(tile_size),
                "--overlap", str(overlap),
                "--device", device,
                "--quiet",
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=max(300, n * 30))
            if proc.returncode != 0:
                raise RuntimeError(f"CNNBlotches runner failed ({proc.returncode}):\n{proc.stderr}")

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
                s_mask_f = masks_dir / f"{stem}_scratch.png"
                d_mask_f = masks_dir / f"{stem}_dust.png"
                c_prob_f = prob_dir / f"{stem}.npy"

                c_mask = cv2.imread(str(c_mask_f), cv2.IMREAD_UNCHANGED)
                s_mask = cv2.imread(str(s_mask_f), cv2.IMREAD_UNCHANGED)
                d_mask = cv2.imread(str(d_mask_f), cv2.IMREAD_UNCHANGED)
                c_prob = np.load(str(c_prob_f)).astype(np.float32)

                c_mask = np.where(c_mask >= 128, np.uint8(255), np.uint8(0))
                s_mask = np.where(s_mask >= 128, np.uint8(255), np.uint8(0))
                d_mask = np.where(d_mask >= 128, np.uint8(255), np.uint8(0))
                c_prob = np.clip(c_prob, 0.0, 1.0).astype(np.float32)

                prov = {
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="cnn_blotches_anisotropic_head",
                        confidence_threshold=scratch_threshold,
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="cnn_blotches_blotch_head",
                        confidence_threshold=dust_threshold,
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
                    scratch_mask=s_mask,
                    dust_mask=d_mask,
                    uncertain_mask=None,
                    provenance=prov,
                    metadata={"model": "cnn_blotches", "index": idx},
                )

                combined_masks.append(c_mask)
                scratch_masks.append(s_mask)
                dust_masks.append(d_mask)
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
# 3. Correl-Net Adapter (Training Required)
# ---------------------------------------------------------------------------

class CorrelNetDetectorAdapter(BaseDefectDetector):
    """Adapter for Correl-Net Multi-Frame Defect Detector (Renaudeau et al., 2023).

    Architecture clean-room implemented in tools/neural_defects/models.py.
    State classified as TRAINING_REQUIRED pending multi-frame temporal corpus training.
    """

    name = "correl_net"
    aliases = ("correlnet", "temporal_correl_net")
    description = "Correl-Net multi-frame temporal correlation defect detector (Renaudeau et al., 2023)."
    license_status = "Training Required (Architecture clean-room implemented)"
    commercial_cleared = False
    temporal_radius = 1
    readiness_state = ReadinessState.TRAINING_REQUIRED
    supported_channels = ("scratch", "dust")
    unsupported_channels = ("uncertain",)

    def __init__(self, tool_dir: Path = MANAGED_NEURAL_DIR) -> None:
        self.tool_dir = Path(tool_dir).resolve()
        self.weights_path = ROOT / "tools/checkpoints/correl_net/correl_net.pt"
        self.manifest_path = ROOT / "tools/checkpoints/manifests/correl_net.json"

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "threshold": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
        }

    def check_readiness(self) -> dict[str, Any]:
        return {
            "ready": False,
            "readiness_state": ReadinessState.TRAINING_REQUIRED.value,
            "commercial_cleared": False,
            "weights_present": self.weights_path.is_file(),
            "manifest_present": self.manifest_path.is_file(),
            "issues": [
                "Full temporal video sequence training required. Architecture implemented in tools/neural_defects/models.py."
            ],
        }

    def detect_defects(self, frame: np.ndarray, options: dict[str, Any] | None = None) -> MultiChannelDefectMask:
        raise NotImplementedError(
            "Correl-Net is in state TRAINING_REQUIRED. Checkpoint weights require multi-frame temporal training."
        )

    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        raise NotImplementedError(
            "Correl-Net is in state TRAINING_REQUIRED. Checkpoint weights require multi-frame temporal training."
        )


def register_neural_defects(registry: Any = None) -> None:
    """Register neural defect adapters into the provided or default registry."""
    try:
        from scripts.film_defect_detectors import DEFAULT_REGISTRY, register_defect_detector
    except ImportError:
        from film_defect_detectors import DEFAULT_REGISTRY, register_defect_detector

    target = registry or DEFAULT_REGISTRY
    target.register(FilmArtefactSegmenterAdapter(), replace=True)
    target.register(CNNBlotchesDetectorAdapter(), replace=True)
    target.register(CorrelNetDetectorAdapter(), replace=True)

