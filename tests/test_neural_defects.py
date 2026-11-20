"""Unit and integration tests for neural film defect detection models and adapters."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts.film_defect_detectors import (
    ChannelOrigin,
    MultiChannelDefectMask,
    ReadinessState,
    build_default_registry,
)
from scripts.neural_defects_adapter import (
    CNNBlotchesDetectorAdapter,
    CorrelNetDetectorAdapter,
    FilmArtefactSegmenterAdapter,
    register_neural_defects,
)
from tools.neural_defects.models import (
    CNNBlotchesScratches,
    CorrelNet,
    FilmArtefactSegmenter,
    load_neural_model,
)
from tools.neural_defects.run_neural_detector import run_inference

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "tools/checkpoints/manifests"


def test_film_artefact_segmenter_manifest() -> None:
    manifest_file = MANIFEST_DIR / "film_artefact_segmenter.json"
    assert manifest_file.is_file(), f"Manifest missing: {manifest_file}"
    with open(manifest_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["model_name"] == "film_artefact_segmenter"
    assert m["readiness_state"] == "pretrained_operational"
    assert m["licensing"]["commercial_cleared"] is True
    assert m["output_schema"]["channel_semantics"]["scratch"]["origin"] == "native"
    assert m["output_schema"]["channel_semantics"]["dust"]["origin"] == "native"
    assert m["checkpoint"]["sha256"] is not None


def test_cnn_blotches_manifest() -> None:
    manifest_file = MANIFEST_DIR / "cnn_blotches.json"
    assert manifest_file.is_file(), f"Manifest missing: {manifest_file}"
    with open(manifest_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["model_name"] == "cnn_blotches"
    assert m["readiness_state"] == "pretrained_operational"
    assert m["licensing"]["commercial_cleared"] is True
    assert m["output_schema"]["channel_semantics"]["scratch"]["origin"] == "native"
    assert m["output_schema"]["channel_semantics"]["dust"]["origin"] == "native"


def test_correl_net_manifest() -> None:
    manifest_file = MANIFEST_DIR / "correl_net.json"
    assert manifest_file.is_file(), f"Manifest missing: {manifest_file}"
    with open(manifest_file, "r", encoding="utf-8") as f:
        m = json.load(f)

    assert m["model_name"] == "correl_net"
    assert m["readiness_state"] == "training_required"
    assert m["licensing"]["commercial_cleared"] is False
    assert "blocker_report" in m
    assert m["blocker_report"]["level"].startswith("Level A")


def test_film_artefact_segmenter_forward_pass() -> None:
    model = FilmArtefactSegmenter(in_channels=3, base_channels=16)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x, return_probs=True)

    assert out["scratch_prob"].shape == (2, 1, 64, 64)
    assert out["dust_prob"].shape == (2, 1, 64, 64)
    assert out["combined_prob"].shape == (2, 1, 64, 64)
    assert float(out["combined_prob"].min()) >= 0.0
    assert float(out["combined_prob"].max()) <= 1.0


def test_correl_net_forward_pass() -> None:
    model = CorrelNet(in_channels=3, base_channels=16, search_radius=1)
    x = torch.randn(2, 3, 3, 64, 64)  # [B, T, C, H, W]
    with torch.no_grad():
        out = model(x, return_probs=True)

    assert out["scratch_prob"].shape == (2, 1, 64, 64)
    assert out["dust_prob"].shape == (2, 1, 64, 64)
    assert out["combined_prob"].shape == (2, 1, 64, 64)


def test_cnn_blotches_forward_pass() -> None:
    model = CNNBlotchesScratches(in_channels=3, base_channels=16)
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(x, return_probs=True)

    assert out["scratch_prob"].shape == (2, 1, 64, 64)
    assert out["dust_prob"].shape == (2, 1, 64, 64)
    assert out["combined_prob"].shape == (2, 1, 64, 64)


def test_film_artefact_segmenter_weights_loading() -> None:
    weights = ROOT / "tools/checkpoints/film_artefact_segmenter/film_artefact_segmenter.pt"
    assert weights.is_file(), f"FAS checkpoint missing: {weights}"
    model = load_neural_model(str(weights), model_name="film_artefact_segmenter", device="cpu")
    assert isinstance(model, FilmArtefactSegmenter)


def test_cnn_blotches_weights_loading() -> None:
    weights = ROOT / "tools/checkpoints/cnn_blotches/cnn_blotches.pt"
    assert weights.is_file(), f"CNN blotches checkpoint missing: {weights}"
    model = load_neural_model(str(weights), model_name="cnn_blotches", device="cpu")
    assert isinstance(model, CNNBlotchesScratches)


def test_neural_detector_runner_inference() -> None:
    with tempfile.TemporaryDirectory(prefix="test-neural-run-") as tmp:
        tpath = Path(tmp)
        in_dir = tpath / "in"
        out_dir = tpath / "out"
        in_dir.mkdir(parents=True)

        # Create synthetic test frame with a vertical line (scratch) and a circle (dust)
        frame = np.full((128, 128, 3), 120, dtype=np.uint8)
        cv2.line(frame, (64, 10), (64, 118), (240, 240, 240), thickness=2)
        cv2.circle(frame, (32, 32), 4, (20, 20, 20), -1)
        cv2.imwrite(str(in_dir / "frame_000.png"), frame)

        summary = run_inference(
            input_path=in_dir,
            output_dir=out_dir,
            model_name="film_artefact_segmenter",
            device_name="cpu",
            tile_size=0,
            quiet=True,
        )

        assert summary["total_frames"] == 1
        assert (out_dir / "masks/frame_000.png").is_file()
        assert (out_dir / "masks/frame_000_scratch.png").is_file()
        assert (out_dir / "masks/frame_000_dust.png").is_file()
        assert (out_dir / "probabilities/frame_000.npy").is_file()

        prob = np.load(str(out_dir / "probabilities/frame_000.npy"))
        assert prob.shape == (128, 128)
        assert prob.dtype == np.float32
        assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0

        mask = cv2.imread(str(out_dir / "masks/frame_000.png"), cv2.IMREAD_UNCHANGED)
        assert set(np.unique(mask).tolist()).issubset({0, 255})


def test_film_artefact_segmenter_adapter_contract() -> None:
    adapter = FilmArtefactSegmenterAdapter()
    readiness = adapter.check_readiness()
    assert readiness["ready"] is True
    assert readiness["commercial_cleared"] is True

    frame = np.full((128, 128, 3), 120, dtype=np.uint8)
    cv2.line(frame, (64, 10), (64, 118), (255, 255, 255), thickness=2)

    mcdm = adapter.detect_defects(frame, {"tile_size": 128, "device": "cpu"})
    assert isinstance(mcdm, MultiChannelDefectMask)
    assert mcdm.combined_probability.shape == (128, 128)
    assert mcdm.combined_mask.shape == (128, 128)
    assert mcdm.scratch_mask is not None and mcdm.scratch_mask.shape == (128, 128)
    assert mcdm.dust_mask is not None and mcdm.dust_mask.shape == (128, 128)

    assert mcdm.provenance["scratch"].origin == ChannelOrigin.NATIVE
    assert mcdm.provenance["dust"].origin == ChannelOrigin.NATIVE
    assert mcdm.provenance["uncertain"].origin == ChannelOrigin.UNSUPPORTED


def test_cnn_blotches_adapter_contract() -> None:
    adapter = CNNBlotchesDetectorAdapter()
    readiness = adapter.check_readiness()
    assert readiness["ready"] is True
    assert readiness["commercial_cleared"] is True

    frame = np.full((128, 128, 3), 120, dtype=np.uint8)
    mcdm = adapter.detect_defects(frame, {"tile_size": 128, "device": "cpu"})
    assert isinstance(mcdm, MultiChannelDefectMask)
    assert mcdm.combined_probability.shape == (128, 128)
    assert mcdm.provenance["scratch"].origin == ChannelOrigin.NATIVE
    assert mcdm.provenance["dust"].origin == ChannelOrigin.NATIVE


def test_correl_net_adapter_training_required() -> None:
    adapter = CorrelNetDetectorAdapter()
    readiness = adapter.check_readiness()
    assert readiness["ready"] is False
    assert readiness["readiness_state"] == ReadinessState.TRAINING_REQUIRED.value
    assert len(readiness["issues"]) > 0

    with pytest.raises(NotImplementedError):
        adapter.detect_defects(np.zeros((64, 64, 3), dtype=np.uint8))


def test_registry_integration() -> None:
    registry = build_default_registry()
    register_neural_defects(registry)
    assert registry.has("film_artefact_segmenter")
    assert registry.has("cnn_blotches")
    assert registry.has("correl_net")

    fas = registry.get("film_artefact_segmenter", commercial_only=True)
    assert isinstance(fas, FilmArtefactSegmenterAdapter)
