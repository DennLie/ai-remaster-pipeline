"""BOPB Scratch Detector Adapter Bridge.

Connects the isolated Microsoft Bringing Old Photos Back to Life scratch detector
to ARP's defect detector registry and multi-channel defect mask architecture.
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
    from scripts.setup_bopb import (
        BOPB_SHA256,
        MANAGED_TOOL_DIR,
        check_readiness as check_bopb_readiness,
        find_venv_python,
        get_weights_path,
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
    from setup_bopb import (
        BOPB_SHA256,
        MANAGED_TOOL_DIR,
        check_readiness as check_bopb_readiness,
        find_venv_python,
        get_weights_path,
    )


class BopbPersistentWorker:
    """Persistent background model worker running inside tools/bopb/.venv."""

    def __init__(self, py_path: Path, runner_script: Path, weights: Path, device: str = "auto") -> None:
        self.py_path = Path(py_path).resolve()
        self.runner_script = Path(runner_script).resolve()
        self.weights = Path(weights).resolve()
        self.device = device
        self.proc: subprocess.Popen | None = None
        self._start()

    def _start(self) -> None:
        cmd = [
            str(self.py_path),
            str(self.runner_script),
            "--worker",
            "--weights", str(self.weights),
            "--device", self.device,
            "--quiet",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready_line = self.proc.stdout.readline() if self.proc.stdout else ""
        try:
            ready = json.loads(ready_line) if ready_line else {}
        except Exception:
            ready = {}
        if ready.get("status") != "ready":
            err = self.proc.stderr.read() if self.proc.stderr else ""
            self.close()
            raise RuntimeError(f"BOPB persistent worker failed to initialize: {err or ready_line}")

    def detect_batch(
        self,
        in_dir: Path,
        out_dir: Path,
        threshold: float,
        tile_size: int,
        overlap: int,
        input_mode: str,
        batch_size: int = 16,
    ) -> dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        req = {
            "cmd": "detect",
            "in_dir": str(in_dir),
            "out_dir": str(out_dir),
            "threshold": threshold,
            "tile_size": tile_size,
            "overlap": overlap,
            "input_mode": input_mode,
            "batch_size": batch_size,
        }
        assert self.proc and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        resp_line = self.proc.stdout.readline()
        if not resp_line:
            err = self.proc.stderr.read() if self.proc.stderr else ""
            self.close()
            raise RuntimeError(f"BOPB worker terminated unexpectedly during inference: {err}")
        resp = json.loads(resp_line)
        if resp.get("status") != "ok":
            raise RuntimeError(f"BOPB worker inference error: {resp.get('error')}")
        return resp

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                    self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None


class BopbScratchDetectorAdapter(BaseDefectDetector):
    """Adapter bridge for Microsoft Bringing Old Photos Back to Life Scratch Detector.

    Executes neural scratch detection in an isolated virtual environment and returns
    validated MultiChannelDefectMask objects adhering strictly to ARP's defect contract.
    Native output semantics: Scratch only! Dust and Uncertain are marked unsupported.
    """

    name = "bopb"
    aliases = ("bopb_scratch_detector", "bopb_scratch", "microsoft_bopb")
    description = "Microsoft Bringing Old Photos Back to Life UNet scratch detector (research-only)."
    license_status = "Restricted (Microsoft Research Academic Non-Commercial License)"
    commercial_cleared = False
    temporal_radius = 0
    readiness_state = ReadinessState.PRETRAINED_OPERATIONAL
    supported_channels = ("scratch",)
    unsupported_channels = ("dust", "uncertain")

    def __init__(self, tool_dir: Path = MANAGED_TOOL_DIR) -> None:
        self.tool_dir = Path(tool_dir).resolve()
        self.runner_script = self.tool_dir / "run_bopb_detector.py"
        self._worker: BopbPersistentWorker | None = None

    def close(self) -> None:
        """Shut down background worker process if active."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def __del__(self) -> None:
        self.close()

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "threshold": {
                "type": "float",
                "default": 0.4,
                "min": 0.0,
                "max": 1.0,
                "description": "Continuous scratch probability binarization threshold in [0.0, 1.0].",
            },
            "tile_size": {
                "type": "int",
                "default": 256,
                "min": 0,
                "max": 2048,
                "description": "Tile size in pixels (default 256; 0 for direct full-frame).",
            },
            "overlap": {
                "type": "int",
                "default": 64,
                "min": 0,
                "max": 512,
                "description": "Tile overlap in pixels for seamless Gaussian/Cosine blending.",
            },
            "device": {
                "type": "str",
                "default": "auto",
                "description": "Target compute device ('auto', 'cuda', 'cpu').",
            },
            "input_mode": {
                "type": "str",
                "default": "tile",
                "description": "Sizing mode: 'tile' (default), 'scale_256', or 'full_size'.",
            },
        }

    def check_readiness(self) -> dict[str, Any]:
        """Inspect isolated environment and weights status."""
        return check_bopb_readiness(tool_dir=self.tool_dir)

    def _get_python_runner(self) -> Path:
        """Resolve python interpreter for isolated runner execution."""
        py = find_venv_python(self.tool_dir)
        if py and py.is_file():
            return py
        # Fallback to current interpreter if dependencies are available
        return Path(sys.executable)

    def _get_worker(self, py: Path, weights: Path, device: str) -> BopbPersistentWorker:
        """Obtain or initialize the persistent BOPB model worker process."""
        if (
            self._worker is None
            or self._worker.device != device
            or (self._worker.proc and self._worker.proc.poll() is not None)
        ):
            if self._worker is not None:
                self._worker.close()
            self._worker = BopbPersistentWorker(py, self.runner_script, weights, device=device)
        return self._worker

    def detect_defects(
        self,
        frame: np.ndarray,
        options: dict[str, Any] | None = None,
    ) -> MultiChannelDefectMask:
        """Execute scratch detection on a single frame and return a MultiChannelDefectMask."""
        results = self.detect([frame], options)
        if not results.multi_channel_masks:
            raise RuntimeError("BOPB defect detection produced no multi-channel masks.")
        return results.multi_channel_masks[0]

    def detect(
        self,
        frames: Sequence[np.ndarray] | np.ndarray,
        config: Mapping[str, Any] | None = None,
    ) -> DefectMaskResult:
        """Execute BOPB scratch detection on a sequence of frames."""
        params = self.validate_config(config)
        threshold = float(params["threshold"])
        tile_size = int(params["tile_size"])
        overlap = int(params["overlap"])
        device = str(params["device"]).strip()
        input_mode = str(params["input_mode"]).strip()

        seq = normalize_frames(frames)
        n = len(seq.bgr_frames)
        h, w = seq.height, seq.width

        py = self._get_python_runner()
        weights = get_weights_path()
        if not weights or not weights.is_file():
            raise FileNotFoundError(
                "BOPB checkpoint (FT_Epoch_latest.pt) not found. "
                "Run 'python scripts/setup_bopb.py --download-weights' to provision weights."
            )

        with tempfile.TemporaryDirectory(prefix="bopb-adapter-") as tmp:
            tmp_path = Path(tmp)
            in_dir = tmp_path / "in"
            out_dir = tmp_path / "out"
            in_dir.mkdir(parents=True)
            out_dir.mkdir(parents=True)

            # Write input frames
            for idx, f in enumerate(seq.bgr_frames):
                cv2.imwrite(str(in_dir / f"frame_{idx:06d}.png"), f)

            # Execute via persistent worker with automatic fallback
            worker_succeeded = False
            try:
                worker = self._get_worker(py, weights, device)
                worker.detect_batch(
                    in_dir=in_dir,
                    out_dir=out_dir,
                    threshold=threshold,
                    tile_size=tile_size,
                    overlap=overlap,
                    input_mode=input_mode,
                    batch_size=16,
                )
                worker_succeeded = True
            except Exception:
                worker_succeeded = False

            if not worker_succeeded:
                cmd = [
                    str(py),
                    str(self.runner_script),
                    "--input", str(in_dir),
                    "--output-dir", str(out_dir),
                    "--weights", str(weights),
                    "--threshold", str(threshold),
                    "--tile-size", str(tile_size),
                    "--overlap", str(overlap),
                    "--input-mode", input_mode,
                    "--device", device,
                    "--quiet",
                ]
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=max(300, n * 30),
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"BOPB inference failed with exit code {proc.returncode}:\n{proc.stderr[:1000]}"
                    )

            # Collect outputs
            combined_masks: list[np.ndarray] = []
            scratch_masks: list[np.ndarray] = []
            dust_masks: list[np.ndarray] = []
            uncertain_masks: list[np.ndarray] = []
            mcdm_list: list[MultiChannelDefectMask] = []

            masks_dir = out_dir / "masks"
            prob_dir = out_dir / "probabilities"
            empty_mask = np.zeros((h, w), dtype=np.uint8)

            for idx in range(n):
                stem = f"frame_{idx:06d}"
                mask_file = masks_dir / f"{stem}.png"
                prob_file = prob_dir / f"{stem}.npy"

                if not mask_file.is_file():
                    raise FileNotFoundError(f"Expected BOPB mask output missing: {mask_file}")

                raw_mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
                if raw_mask is None or raw_mask.shape != (h, w):
                    raise ValueError(f"Invalid mask geometry for frame {idx}: {raw_mask.shape if raw_mask is not None else None}")

                # Ensure strictly {0, 255} uint8
                binary_scratch = np.where(raw_mask >= 128, np.uint8(255), np.uint8(0))

                if prob_file.is_file():
                    prob_map = np.load(str(prob_file)).astype(np.float32)
                else:
                    prob_png = prob_dir / f"{stem}.png"
                    if prob_png.is_file():
                        p_raw = cv2.imread(str(prob_png), cv2.IMREAD_UNCHANGED)
                        prob_map = (p_raw.astype(np.float32) / 255.0).clip(0.0, 1.0)
                    else:
                        prob_map = (binary_scratch.astype(np.float32) / 255.0).clip(0.0, 1.0)

                # Ensure probability invariants
                prob_map = np.clip(prob_map, 0.0, 1.0).astype(np.float32)

                # Build MultiChannelDefectMask
                # BOPB is natively Scratch only. Dust and Uncertain are marked unsupported.
                provenance = {
                    "scratch": ChannelProvenance(
                        origin=ChannelOrigin.NATIVE,
                        is_available=True,
                        source_algorithm="bopb_scratch_unet",
                        derivation_parameters={
                            "threshold": threshold,
                            "tile_size": tile_size,
                            "overlap": overlap,
                            "input_mode": input_mode,
                        },
                        confidence_threshold=threshold,
                    ),
                    "dust": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="none",
                    ),
                    "uncertain": ChannelProvenance(
                        origin=ChannelOrigin.UNSUPPORTED,
                        is_available=False,
                        source_algorithm="none",
                    ),
                }

                mcdm = MultiChannelDefectMask(
                    combined_probability=prob_map,
                    combined_mask=binary_scratch.copy(),
                    scratch_mask=binary_scratch.copy(),
                    dust_mask=None,
                    uncertain_mask=None,
                    provenance=provenance,
                    metadata={
                        "detector": self.name,
                        "frame_index": idx,
                        "threshold": threshold,
                        "weights_sha256": BOPB_SHA256,
                        "commercial_cleared": self.commercial_cleared,
                        "license": self.license_status,
                    },
                )

                combined_masks.append(binary_scratch.copy())
                scratch_masks.append(binary_scratch.copy())
                dust_masks.append(empty_mask.copy())
                uncertain_masks.append(empty_mask.copy())
                mcdm_list.append(mcdm)

            return DefectMaskResult(
                mask=combined_masks,
                scratch_mask=scratch_masks,
                dust_mask=dust_masks,
                uncertain_mask=uncertain_masks,
                multi_channel_masks=mcdm_list,
                metadata={
                    "detector": self.name,
                    "config": params,
                    "frames": n,
                    "weights_sha256": BOPB_SHA256,
                    "commercial_cleared": self.commercial_cleared,
                    "license": self.license_status,
                    "channel_semantics": {
                        "scratch": "native",
                        "dust": "unsupported",
                        "uncertain": "unsupported",
                    },
                },
            )
