"""Unit tests for BOPB Scratch Detector runner, setup manager, and adapter bridge."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from scripts.bopb_defect_adapter import BopbScratchDetectorAdapter
from scripts.film_defect_detectors import (
    ChannelOrigin,
    MultiChannelDefectMask,
    RepairInclusionPolicy,
    UnsupportedPolicyError,
)
from scripts.setup_bopb import (
    BOPB_SHA256,
    MANIFEST_FILE,
    check_readiness,
    get_weights_path,
)
from tools.bopb.models import BOPBScratchUNet, load_bopb_model
from tools.bopb.run_bopb_detector import run_inference


def test_bopb_manifest_schema_and_licensing() -> None:
    """Verify manifest exists, is valid JSON, and declares non-commercial license."""
    assert MANIFEST_FILE.is_file(), f"Manifest file missing: {MANIFEST_FILE}"
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["model_name"] == "bopb_scratch_detector"
    assert "bopb" in manifest["aliases"]
    assert manifest["checkpoint"]["sha256"] == BOPB_SHA256
    assert manifest["checkpoint"]["size_bytes"] == 451663663
    assert manifest["licensing"]["commercial_cleared"] is False
    assert "Academic Non-Commercial" in manifest["licensing"]["weights_license"]
    assert manifest["architecture"]["type"] == "UNet"
    assert manifest["architecture"]["antialiasing"] is True
    assert manifest["output_schema"]["channel_semantics"]["scratch"]["origin"] == "native"
    assert manifest["output_schema"]["channel_semantics"]["dust"]["origin"] == "unsupported"


def test_bopb_setup_readiness_check() -> None:
    """Verify setup_bopb check_readiness reports operational state with verified weights."""
    status = check_readiness(verify_sha=True)
    assert status["ready"] is True
    assert status["readiness_state"] == "pretrained_operational"
    assert status["commercial_cleared"] is False
    assert status["weights_present"] is True
    assert status["weights_valid"] is True
    assert status["weights_sha256"] == BOPB_SHA256
    assert status["manifest_present"] is True
    assert len(status["issues"]) == 0


def test_bopb_model_architecture_shapes() -> None:
    """Verify BOPBScratchUNet forward pass preserves spatial resolution divisible by 16."""
    model = BOPBScratchUNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        antialiasing=True,
    )
    dummy_input = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (1, 1, 64, 64)


def test_bopb_weights_loading_strict() -> None:
    """Verify load_bopb_model loads checkpoint weights with strict state_dict match."""
    weights = get_weights_path()
    assert weights is not None and weights.is_file()
    model = load_bopb_model(str(weights), device="cpu")
    assert isinstance(model, BOPBScratchUNet)
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params == 37626177


def test_bopb_runner_inference_execution() -> None:
    """Verify standalone run_inference generates probability map, binary mask, and summary."""
    with tempfile.TemporaryDirectory(prefix="test-bopb-run-") as tmp:
        tpath = Path(tmp)
        in_dir = tpath / "in"
        out_dir = tpath / "out"
        in_dir.mkdir(parents=True)

        # Create synthetic test image with dark background and bright vertical scratch line
        img = np.full((256, 256, 3), 40, dtype=np.uint8)
        cv2.line(img, (128, 20), (128, 236), (240, 240, 240), thickness=2)
        test_img_path = in_dir / "test_scratch.png"
        cv2.imwrite(str(test_img_path), img)

        summary = run_inference(
            input_path=test_img_path,
            output_dir=out_dir,
            threshold=0.4,
            tile_size=256,
            device_name="cpu",
            export_prob_npy=True,
            verify_hash=True,
        )

        assert summary["model"] == "bopb_scratch_detector"
        assert summary["checkpoint_sha256"] == BOPB_SHA256
        assert summary["frames_processed"] == 1
        assert summary["channel_semantics"]["scratch"] == "native"
        assert summary["channel_semantics"]["dust"] == "unsupported"

        mask_file = out_dir / "masks" / "test_scratch.png"
        prob_npy_file = out_dir / "probabilities" / "test_scratch.npy"
        prob_png_file = out_dir / "probabilities" / "test_scratch.png"

        assert mask_file.is_file()
        assert prob_npy_file.is_file()
        assert prob_png_file.is_file()

        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        assert mask.shape == (256, 256)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

        prob = np.load(str(prob_npy_file))
        assert prob.shape == (256, 256)
        assert prob.dtype == np.float32
        assert 0.0 <= float(prob.min()) <= float(prob.max()) <= 1.0


def test_bopb_adapter_multi_channel_contract() -> None:
    """Verify BopbScratchDetectorAdapter adheres strictly to MultiChannelDefectMask contract."""
    adapter = BopbScratchDetectorAdapter()
    assert adapter.name == "bopb"
    assert adapter.commercial_cleared is False
    assert adapter.supported_channels == ("scratch",)
    assert adapter.unsupported_channels == ("dust", "uncertain")

    img = np.full((128, 128, 3), 50, dtype=np.uint8)
    cv2.line(img, (64, 10), (64, 118), (250, 250, 250), thickness=2)

    mcdm = adapter.detect_defects(img, {"threshold": 0.4, "device": "cpu", "tile_size": 128})
    assert isinstance(mcdm, MultiChannelDefectMask)
    assert mcdm.combined_probability.dtype == np.float32
    assert mcdm.combined_probability.shape == (128, 128)
    assert mcdm.combined_mask.dtype == np.uint8
    assert mcdm.combined_mask.shape == (128, 128)
    assert set(np.unique(mcdm.combined_mask)).issubset({0, 255})

    # Scratch is native, dust/uncertain are None
    assert mcdm.scratch_mask is not None
    assert mcdm.dust_mask is None
    assert mcdm.uncertain_mask is None

    assert mcdm.provenance["scratch"].origin == ChannelOrigin.NATIVE
    assert mcdm.provenance["scratch"].is_available is True
    assert mcdm.provenance["dust"].origin == ChannelOrigin.UNSUPPORTED
    assert mcdm.provenance["dust"].is_available is False
    assert mcdm.provenance["uncertain"].origin == ChannelOrigin.UNSUPPORTED
    assert mcdm.provenance["uncertain"].is_available is False

    # Repair inclusion policies
    s_only = mcdm.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)
    assert np.array_equal(s_only, mcdm.scratch_mask)

    conserv = mcdm.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
    assert np.array_equal(conserv, mcdm.scratch_mask)

    comb = mcdm.get_effective_repair_mask(RepairInclusionPolicy.COMBINED_DEFECT)
    assert np.array_equal(comb, mcdm.combined_mask)

    # Legacy compatibility bridge
    s_bridge, d_bridge, manifest = mcdm.to_legacy_compat_bridge(RepairInclusionPolicy.SCRATCH_ONLY)
    assert np.array_equal(s_bridge, mcdm.scratch_mask)
    assert np.count_nonzero(d_bridge) == 0
    assert manifest["scratch_available"] is True
    assert manifest["dust_available"] is False


def test_bopb_adapter_sequence_detection() -> None:
    """Verify BopbScratchDetectorAdapter.detect() handles sequences returning DefectMaskResult."""
    adapter = BopbScratchDetectorAdapter()
    f1 = np.full((128, 128, 3), 60, dtype=np.uint8)
    f2 = np.full((128, 128, 3), 70, dtype=np.uint8)

    result = adapter.detect([f1, f2], {"threshold": 0.4, "device": "cpu", "tile_size": 128})
    assert len(result) == 2
    assert len(result.mask) == 2
    assert len(result.scratch_mask) == 2
    assert len(result.dust_mask) == 2
    assert result.multi_channel_masks is not None
    assert len(result.multi_channel_masks) == 2
    assert result.metadata["detector"] == "bopb"
    assert result.metadata["weights_sha256"] == BOPB_SHA256
