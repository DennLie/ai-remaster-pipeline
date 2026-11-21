"""Unit and integration tests for PatchCore and PaDiM anomaly detection models and adapters."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts.anomaly_defect_adapter import (
    PaDiMDefectAdapter,
    PatchCoreDefectAdapter,
    register_anomaly_detectors,
)
from scripts.film_defect_detectors import (
    ChannelOrigin,
    MultiChannelDefectMask,
    build_default_registry,
)
from tools.anomalib.models import (
    PaDiMAnomalyDetector,
    PatchCoreAnomalyDetector,
    load_anomaly_detector,
)
from tools.anomalib.run_anomaly_detector import run_anomaly_inference

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "tools/checkpoints/manifests"


def test_patchcore_manifest() -> None:
    manifest_file = MANIFEST_DIR / "patchcore.json"
    assert manifest_file.is_file(), f"Manifest missing: {manifest_file}"
    with open(manifest_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["model_name"] == "patchcore"
    assert m["readiness_state"] == "pretrained_operational"
    assert m["licensing"]["commercial_cleared"] is True
    assert m["output_schema"]["channel_semantics"]["combined"]["origin"] == "native"
    assert m["output_schema"]["channel_semantics"]["scratch"]["origin"] == "unsupported"
    assert m["output_schema"]["channel_semantics"]["dust"]["origin"] == "unsupported"


def test_padim_manifest() -> None:
    manifest_file = MANIFEST_DIR / "padim.json"
    assert manifest_file.is_file(), f"Manifest missing: {manifest_file}"
    with open(manifest_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["model_name"] == "padim"
    assert m["readiness_state"] == "pretrained_operational"
    assert m["licensing"]["commercial_cleared"] is True
    assert m["output_schema"]["channel_semantics"]["combined"]["origin"] == "native"


def test_patchcore_checkpoint_loading_and_inference() -> None:
    weights = ROOT / "tools/checkpoints/patchcore/patchcore_memory_bank.pt"
    assert weights.is_file(), f"PatchCore weights missing: {weights}"
    detector = load_anomaly_detector(weights, model_name="patchcore", device="cpu")
    assert isinstance(detector, PatchCoreAnomalyDetector)
    assert detector.memory_bank is not None

    dummy = torch.randn(1, 3, 128, 128).clamp(0.0, 1.0)
    prob_map, mask = detector.predict(dummy)

    assert prob_map.shape == (128, 128)
    assert prob_map.dtype == np.float32
    assert float(prob_map.min()) >= 0.0 and float(prob_map.max()) <= 1.0
    assert set(np.unique(mask).tolist()).issubset({0, 255})


def test_padim_checkpoint_loading_and_inference() -> None:
    weights = ROOT / "tools/checkpoints/padim/padim_distribution.pt"
    assert weights.is_file(), f"PaDiM weights missing: {weights}"
    detector = load_anomaly_detector(weights, model_name="padim", device="cpu")
    assert isinstance(detector, PaDiMAnomalyDetector)
    assert detector.mean is not None
    assert detector.inv_cov is not None

    dummy = torch.randn(1, 3, 256, 256).clamp(0.0, 1.0)
    prob_map, mask = detector.predict(dummy)

    assert prob_map.shape == (256, 256)
    assert prob_map.dtype == np.float32
    assert float(prob_map.min()) >= 0.0 and float(prob_map.max()) <= 1.0
    assert set(np.unique(mask).tolist()).issubset({0, 255})


def test_anomaly_runner_inference_execution() -> None:
    with tempfile.TemporaryDirectory(prefix="test-anom-run-") as tmp:
        tpath = Path(tmp)
        in_dir = tpath / "in"
        out_dir = tpath / "out"
        in_dir.mkdir(parents=True)

        frame = np.full((128, 128, 3), 120, dtype=np.uint8)
        cv2.circle(frame, (64, 64), 10, (10, 10, 10), -1)
        cv2.imwrite(str(in_dir / "frame_000.png"), frame)

        summary = run_anomaly_inference(
            input_path=in_dir,
            output_dir=out_dir,
            model_name="patchcore",
            device_name="cpu",
            tile_size=0,
            quiet=True,
        )

        assert summary["total_frames"] == 1
        assert (out_dir / "masks/frame_000.png").is_file()
        assert (out_dir / "probabilities/frame_000.npy").is_file()

        prob = np.load(str(out_dir / "probabilities/frame_000.npy"))
        assert prob.shape == (128, 128)
        assert prob.dtype == np.float32
        assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0


def test_patchcore_adapter_contract() -> None:
    adapter = PatchCoreDefectAdapter()
    readiness = adapter.check_readiness()
    assert readiness["ready"] is True
    assert readiness["commercial_cleared"] is True

    frame = np.full((128, 128, 3), 120, dtype=np.uint8)
    mcdm = adapter.detect_defects(frame, {"tile_size": 128, "device": "cpu"})
    assert isinstance(mcdm, MultiChannelDefectMask)
    assert mcdm.combined_probability.shape == (128, 128)
    assert mcdm.combined_mask.shape == (128, 128)
    assert mcdm.scratch_mask is None
    assert mcdm.dust_mask is None
    assert mcdm.provenance["combined"].origin == ChannelOrigin.NATIVE
    assert mcdm.provenance["scratch"].origin == ChannelOrigin.UNSUPPORTED
    assert mcdm.provenance["dust"].origin == ChannelOrigin.UNSUPPORTED


def test_padim_adapter_contract() -> None:
    adapter = PaDiMDefectAdapter()
    readiness = adapter.check_readiness()
    assert readiness["ready"] is True
    assert readiness["commercial_cleared"] is True

    frame = np.full((128, 128, 3), 120, dtype=np.uint8)
    mcdm = adapter.detect_defects(frame, {"tile_size": 128, "device": "cpu"})
    assert isinstance(mcdm, MultiChannelDefectMask)
    assert mcdm.combined_probability.shape == (128, 128)
    assert mcdm.provenance["combined"].origin == ChannelOrigin.NATIVE
    assert mcdm.provenance["scratch"].origin == ChannelOrigin.UNSUPPORTED


def test_registry_integration_anomaly() -> None:
    registry = build_default_registry()
    register_anomaly_detectors(registry)
    assert registry.has("patchcore")
    assert registry.has("padim")

    pc = registry.get("patchcore", commercial_only=True)
    assert isinstance(pc, PatchCoreDefectAdapter)
