"""Unit tests for MultiChannelDefectMask and multi-channel defect detection contracts.

Verifies:
1. Exact Two-Pixel Behavioral Test:
   Scratch=[255, 0], Dust=[0, 0], Uncertain=[0, 255]
   - CONSERVATIVE -> strictly [255, 0] (Uncertain excluded)
   - HIGH_RECALL -> strictly [255, 255] (Uncertain included)
   - SCRATCH_ONLY -> strictly [255, 0]
   - COMBINED_DEFECT -> strictly [255, 255]
2. All Four RepairInclusionPolicy Behaviors:
   - CONSERVATIVE: Scratch | Dust. Fails with UnsupportedPolicyError when model provides only combined mask.
   - HIGH_RECALL: Scratch | Dust | Uncertain. Fails with UnsupportedPolicyError when model provides no channels.
   - SCRATCH_ONLY: Scratch only. Fails with UnsupportedPolicyError when Scratch is None.
   - COMBINED_DEFECT: Combined mask only. Works reliably on combined-only models.
3. Bitwise Union & Arithmetic Overflow Protection:
   - When Scratch, Dust, and Uncertain overlap at identical pixel coordinates, verify bitwise OR
     unions produce strictly 255 without integer wraparound.
4. Input Validation & Invariant Enforcement:
   - 2D array and shape checks.
   - Dtype enforcement: float32 for probability, uint8 for masks.
   - Finite float validation: NaN / Inf in probability raises MaskContractValidationError.
   - Probability range validation: values < 0.0 or > 1.0 raise MaskContractValidationError.
   - Strict binary mask validation: values other than 0 or 255 (e.g. 1, 128, 254) raise MaskContractValidationError.
   - Channel provenance consistency checks: available marked True with None mask, or False with non-None mask.
5. Legacy Compatibility Bridge (to_legacy_compat_bridge):
   - Invariant: np.bitwise_or(scratch, dust) == get_effective_repair_mask(policy) for all 4 policies (100% pixel-identical).
   - Manifest verification: json.dumps(manifest) succeeds with zero raw numpy arrays.
   - Combined-only models: is_compatibility_mapping is True with warning, and effective mask is delivered without silent drop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from film_defect_detectors import (  # noqa: E402
    ChannelOrigin,
    ChannelProvenance,
    MaskContractValidationError,
    MultiChannelDefectMask,
    ReadinessState,
    RepairInclusionPolicy,
    UnsupportedPolicyError,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def two_pixel_defect_data():
    """Provides the canonical 2-pixel test fixtures:

    Pixel 0: Defect is a Scratch (Scratch=255, Dust=0, Uncertain=0, Combined=255)
    Pixel 1: Defect is Uncertain (Scratch=0, Dust=0, Uncertain=255, Combined=255)
    """
    prob = np.array([[1.0, 1.0]], dtype=np.float32)
    combined = np.array([[255, 255]], dtype=np.uint8)
    scratch = np.array([[255, 0]], dtype=np.uint8)
    dust = np.array([[0, 0]], dtype=np.uint8)
    uncertain = np.array([[0, 255]], dtype=np.uint8)

    provenance = {
        "scratch": ChannelProvenance(
            origin=ChannelOrigin.DERIVED,
            is_available=True,
            source_algorithm="ridge_filter",
        ),
        "dust": ChannelProvenance(
            origin=ChannelOrigin.DERIVED,
            is_available=True,
            source_algorithm="morphological_cc",
        ),
        "uncertain": ChannelProvenance(
            origin=ChannelOrigin.DERIVED,
            is_available=True,
            source_algorithm="ambiguity_heuristic",
        ),
    }

    mask_obj = MultiChannelDefectMask(
        combined_probability=prob,
        combined_mask=combined,
        scratch_mask=scratch,
        dust_mask=dust,
        uncertain_mask=uncertain,
        provenance=provenance,
        metadata={"test": "two_pixel_fixture"},
    )
    return {
        "mask_obj": mask_obj,
        "prob": prob,
        "combined": combined,
        "scratch": scratch,
        "dust": dust,
        "uncertain": uncertain,
    }


@pytest.fixture
def base_2d_masks():
    """Provides a synthetic 4x4 multi-channel mask set with distinct and overlapping regions."""
    prob = np.full((4, 4), 0.75, dtype=np.float32)
    combined = np.full((4, 4), 255, dtype=np.uint8)

    scratch = np.zeros((4, 4), dtype=np.uint8)
    scratch[:, 1] = 255  # vertical scratch line

    dust = np.zeros((4, 4), dtype=np.uint8)
    dust[2, 2] = 255  # dust spot
    dust[0, 1] = 255  # overlap with scratch at (0, 1)

    uncertain = np.zeros((4, 4), dtype=np.uint8)
    uncertain[3, 3] = 255
    uncertain[0, 1] = 255  # 3-way overlap at (0, 1)

    provenance = {
        "scratch": ChannelProvenance(ChannelOrigin.NATIVE, True, "native_neural_scratch"),
        "dust": ChannelProvenance(ChannelOrigin.NATIVE, True, "native_neural_dust"),
        "uncertain": ChannelProvenance(ChannelOrigin.DERIVED, True, "anomaly_uncertainty"),
    }

    return MultiChannelDefectMask(
        combined_probability=prob,
        combined_mask=combined,
        scratch_mask=scratch,
        dust_mask=dust,
        uncertain_mask=uncertain,
        provenance=provenance,
    )


# ==============================================================================
# 1. Exact Two-Pixel Behavioral Test
# ==============================================================================


class TestTwoPixelBehavior:
    """Rigorous verification of the exact 2-pixel behavior:

    Scratch=[255, 0], Dust=[0, 0], Uncertain=[0, 255]
    """

    def test_two_pixel_conservative_policy_excludes_uncertain(self, two_pixel_defect_data):
        """CONSERVATIVE policy: strictly [255, 0] (Uncertain excluded)."""
        mask_obj = two_pixel_defect_data["mask_obj"]
        result = mask_obj.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)

        expected = np.array([[255, 0]], dtype=np.uint8)
        assert np.array_equal(result, expected), (
            f"Expected CONSERVATIVE to resolve strictly [[255, 0]], got {result}"
        )
        assert result[0, 0] == 255, "Pixel 0 (scratch) must be included"
        assert result[0, 1] == 0, "Pixel 1 (uncertain) must be strictly excluded under CONSERVATIVE"

    def test_two_pixel_high_recall_policy_includes_uncertain(self, two_pixel_defect_data):
        """HIGH_RECALL policy: strictly [255, 255] (Uncertain included)."""
        mask_obj = two_pixel_defect_data["mask_obj"]
        result = mask_obj.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)

        expected = np.array([[255, 255]], dtype=np.uint8)
        assert np.array_equal(result, expected), (
            f"Expected HIGH_RECALL to resolve strictly [[255, 255]], got {result}"
        )
        assert result[0, 0] == 255, "Pixel 0 (scratch) must be included"
        assert result[0, 1] == 255, "Pixel 1 (uncertain) must be included under HIGH_RECALL"

    def test_two_pixel_scratch_only_policy_includes_only_scratch(self, two_pixel_defect_data):
        """SCRATCH_ONLY policy: strictly [255, 0]."""
        mask_obj = two_pixel_defect_data["mask_obj"]
        result = mask_obj.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)

        expected = np.array([[255, 0]], dtype=np.uint8)
        assert np.array_equal(result, expected), (
            f"Expected SCRATCH_ONLY to resolve strictly [[255, 0]], got {result}"
        )
        assert result[0, 0] == 255, "Pixel 0 (scratch) must be included"
        assert result[0, 1] == 0, "Pixel 1 (uncertain) must be strictly excluded under SCRATCH_ONLY"

    def test_two_pixel_combined_defect_policy_includes_all_proposals(self, two_pixel_defect_data):
        """COMBINED_DEFECT policy: strictly [255, 255]."""
        mask_obj = two_pixel_defect_data["mask_obj"]
        result = mask_obj.get_effective_repair_mask(RepairInclusionPolicy.COMBINED_DEFECT)

        expected = np.array([[255, 255]], dtype=np.uint8)
        assert np.array_equal(result, expected), (
            f"Expected COMBINED_DEFECT to resolve strictly [[255, 255]], got {result}"
        )
        assert result[0, 0] == 255
        assert result[0, 1] == 255


# ==============================================================================
# 2. All Four RepairInclusionPolicy Behaviors & Error Conditions
# ==============================================================================


class TestRepairInclusionPolicies:
    """Verifies all four RepairInclusionPolicy behaviors and required failure modes."""

    def test_conservative_policy_combines_scratch_and_dust(self):
        """CONSERVATIVE combines Scratch | Dust without including Uncertain."""
        prob = np.full((3, 3), 0.5, dtype=np.float32)
        combined = np.full((3, 3), 255, dtype=np.uint8)
        scratch = np.zeros((3, 3), dtype=np.uint8)
        dust = np.zeros((3, 3), dtype=np.uint8)
        uncertain = np.zeros((3, 3), dtype=np.uint8)

        scratch[0, 0] = 255
        dust[1, 1] = 255
        uncertain[2, 2] = 255

        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=scratch,
            dust_mask=dust,
            uncertain_mask=uncertain,
        )

        effective = mask.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        expected = np.bitwise_or(scratch, dust)
        assert np.array_equal(effective, expected)
        assert effective[0, 0] == 255
        assert effective[1, 1] == 255
        assert effective[2, 2] == 0, "Uncertain channel must be excluded in CONSERVATIVE"

    def test_conservative_policy_works_with_scratch_only_or_dust_only(self):
        """CONSERVATIVE succeeds if either scratch or dust is provided."""
        prob = np.full((2, 2), 0.5, dtype=np.float32)
        combined = np.full((2, 2), 255, dtype=np.uint8)
        scratch = np.array([[255, 0], [0, 0]], dtype=np.uint8)

        # Scratch provided, dust is None
        mask_scratch = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=scratch,
            dust_mask=None,
        )
        eff_scratch = mask_scratch.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        assert np.array_equal(eff_scratch, scratch)

        # Dust provided, scratch is None
        dust = np.array([[0, 255], [0, 0]], dtype=np.uint8)
        mask_dust = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=None,
            dust_mask=dust,
        )
        eff_dust = mask_dust.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        assert np.array_equal(eff_dust, dust)

    def test_conservative_policy_fails_when_model_only_provides_combined_mask(self):
        """CONSERVATIVE fails with UnsupportedPolicyError when both scratch and dust are None."""
        prob = np.full((2, 2), 0.5, dtype=np.float32)
        combined = np.full((2, 2), 255, dtype=np.uint8)
        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=None,
            dust_mask=None,
        )

        with pytest.raises(UnsupportedPolicyError, match="Policy 'CONSERVATIVE' requires Scratch or Dust"):
            mask.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)

    def test_high_recall_policy_combines_scratch_dust_and_uncertain(self):
        """HIGH_RECALL combines Scratch | Dust | Uncertain."""
        prob = np.full((3, 3), 0.5, dtype=np.float32)
        combined = np.full((3, 3), 255, dtype=np.uint8)
        scratch = np.zeros((3, 3), dtype=np.uint8)
        dust = np.zeros((3, 3), dtype=np.uint8)
        uncertain = np.zeros((3, 3), dtype=np.uint8)

        scratch[0, 0] = 255
        dust[1, 1] = 255
        uncertain[2, 2] = 255

        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=scratch,
            dust_mask=dust,
            uncertain_mask=uncertain,
        )

        effective = mask.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)
        expected = np.bitwise_or(np.bitwise_or(scratch, dust), uncertain)
        assert np.array_equal(effective, expected)
        assert effective[0, 0] == 255
        assert effective[1, 1] == 255
        assert effective[2, 2] == 255

    def test_high_recall_policy_fails_when_no_channels_provided(self):
        """HIGH_RECALL fails with UnsupportedPolicyError when all classified/uncertain channels are None."""
        prob = np.full((2, 2), 0.5, dtype=np.float32)
        combined = np.full((2, 2), 255, dtype=np.uint8)
        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=None,
            dust_mask=None,
            uncertain_mask=None,
        )

        with pytest.raises(
            UnsupportedPolicyError, match="Policy 'HIGH_RECALL' requested but model provides no classified"
        ):
            mask.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)

    def test_scratch_only_policy_returns_scratch_mask(self):
        """SCRATCH_ONLY returns the scratch channel copy and excludes dust and uncertain."""
        prob = np.full((3, 3), 0.5, dtype=np.float32)
        combined = np.full((3, 3), 255, dtype=np.uint8)
        scratch = np.zeros((3, 3), dtype=np.uint8)
        dust = np.zeros((3, 3), dtype=np.uint8)
        uncertain = np.zeros((3, 3), dtype=np.uint8)

        scratch[0, 1] = 255
        dust[1, 1] = 255
        uncertain[2, 2] = 255

        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=scratch,
            dust_mask=dust,
            uncertain_mask=uncertain,
        )

        effective = mask.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)
        assert np.array_equal(effective, scratch)
        assert effective[0, 1] == 255
        assert effective[1, 1] == 0
        assert effective[2, 2] == 0

    def test_scratch_only_policy_fails_when_scratch_is_none(self):
        """SCRATCH_ONLY fails with UnsupportedPolicyError when Scratch is None."""
        prob = np.full((2, 2), 0.5, dtype=np.float32)
        combined = np.full((2, 2), 255, dtype=np.uint8)
        dust = np.full((2, 2), 255, dtype=np.uint8)

        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=None,
            dust_mask=dust,
        )

        with pytest.raises(UnsupportedPolicyError, match="Policy 'SCRATCH_ONLY' requested but model does not provide a Scratch channel"):
            mask.get_effective_repair_mask(RepairInclusionPolicy.SCRATCH_ONLY)

    def test_combined_defect_policy_works_on_combined_only_models(self):
        """COMBINED_DEFECT works reliably on models that provide only a combined defect output."""
        prob = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        combined = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=None,
            dust_mask=None,
            uncertain_mask=None,
        )

        effective = mask.get_effective_repair_mask(RepairInclusionPolicy.COMBINED_DEFECT)
        assert np.array_equal(effective, combined)

        # Modifying returned array does not mutate internal combined_mask
        effective[0, 0] = 255
        assert mask.combined_mask[0, 0] == 0

    @pytest.mark.parametrize(
        "policy_input,expected_enum",
        [
            ("conservative", RepairInclusionPolicy.CONSERVATIVE),
            ("CONSERVATIVE", RepairInclusionPolicy.CONSERVATIVE),
            ("Conservative", RepairInclusionPolicy.CONSERVATIVE),
            ("high_recall", RepairInclusionPolicy.HIGH_RECALL),
            ("HIGH_RECALL", RepairInclusionPolicy.HIGH_RECALL),
            ("scratch_only", RepairInclusionPolicy.SCRATCH_ONLY),
            ("SCRATCH_ONLY", RepairInclusionPolicy.SCRATCH_ONLY),
            ("combined_defect", RepairInclusionPolicy.COMBINED_DEFECT),
            ("COMBINED_DEFECT", RepairInclusionPolicy.COMBINED_DEFECT),
        ],
    )
    def test_policy_resolution_from_string(self, base_2d_masks, policy_input, expected_enum):
        """Policy resolution handles case-insensitive strings correctly."""
        mask = base_2d_masks
        res_str = mask.get_effective_repair_mask(policy_input)
        res_enum = mask.get_effective_repair_mask(expected_enum)
        assert np.array_equal(res_str, res_enum)

    def test_unknown_policy_raises_unsupported_policy_error(self, base_2d_masks):
        """Invalid policy string raises UnsupportedPolicyError."""
        with pytest.raises(UnsupportedPolicyError, match="Unknown or unsupported RepairInclusionPolicy"):
            base_2d_masks.get_effective_repair_mask("non_existent_policy")


# ==============================================================================
# 3. Bitwise Union & Arithmetic Overflow Protection
# ==============================================================================


class TestBitwiseUnionAndOverflowProtection:
    """Verifies that multi-channel unions use bitwise OR to avoid arithmetic wraparound."""

    def test_overlapping_defects_do_not_wrap_around(self):
        """When Scratch, Dust, and Uncertain overlap at identical pixel coordinates,

        arithmetic sum: 255 + 255 + 255 = 765 -> (765 % 256) = 253 (corrupted uint8!).
        bitwise OR: 255 | 255 | 255 = 255 (strictly valid binary defect).
        """
        shape = (3, 3)
        prob = np.full(shape, 1.0, dtype=np.float32)
        combined = np.full(shape, 255, dtype=np.uint8)

        # All 3 channels have 255 at every coordinate
        scratch = np.full(shape, 255, dtype=np.uint8)
        dust = np.full(shape, 255, dtype=np.uint8)
        uncertain = np.full(shape, 255, dtype=np.uint8)

        # Demonstrate the overflow that would happen with arithmetic addition
        arithmetic_sum_two = scratch + dust  # 255 + 255 = 254 in uint8!
        assert arithmetic_sum_two[0, 0] == 254, "Sanity check: uint8 addition wraps to 254"
        arithmetic_sum_three = scratch + dust + uncertain  # 255 + 255 + 255 = 253 in uint8!
        assert arithmetic_sum_three[0, 0] == 253, "Sanity check: uint8 addition wraps to 253"

        mask = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=scratch,
            dust_mask=dust,
            uncertain_mask=uncertain,
        )

        # 1. Conservative (scratch | dust)
        eff_conservative = mask.get_effective_repair_mask(RepairInclusionPolicy.CONSERVATIVE)
        assert eff_conservative.dtype == np.uint8
        assert np.all(eff_conservative == 255), "Conservative union must be strictly 255, avoiding 254 wraparound"

        # 2. High Recall (scratch | dust | uncertain)
        eff_high_recall = mask.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL)
        assert eff_high_recall.dtype == np.uint8
        assert np.all(eff_high_recall == 255), "High recall union must be strictly 255, avoiding 253 wraparound"

    def test_bitwise_or_pixel_values_strictly_binary(self, base_2d_masks):
        """All pixels in resolved effective masks must be strictly in {0, 255}."""
        for policy in RepairInclusionPolicy:
            mask = base_2d_masks.get_effective_repair_mask(policy)
            unique_vals = set(np.unique(mask))
            assert unique_vals.issubset({0, 255}), (
                f"Policy {policy} produced non-binary values: {unique_vals}"
            )


# ==============================================================================
# 4. Input Validation & Invariant Enforcement
# ==============================================================================


class TestInputValidationAndInvariants:
    """Verifies strict invariant checking in MultiChannelDefectMask.__post_init__."""

    @pytest.fixture
    def valid_inputs(self):
        """Valid base inputs for mutation testing."""
        return {
            "combined_probability": np.full((4, 4), 0.5, dtype=np.float32),
            "combined_mask": np.full((4, 4), 255, dtype=np.uint8),
            "scratch_mask": np.zeros((4, 4), dtype=np.uint8),
            "dust_mask": np.zeros((4, 4), dtype=np.uint8),
            "uncertain_mask": np.zeros((4, 4), dtype=np.uint8),
        }

    # 4.1 2D array and shape checks
    def test_probability_not_ndarray_raises(self, valid_inputs):
        valid_inputs["combined_probability"] = [[0.5, 0.5], [0.5, 0.5]]
        with pytest.raises(MaskContractValidationError, match="combined_probability must be a numpy.ndarray"):
            MultiChannelDefectMask(**valid_inputs)

    def test_probability_non_2d_raises(self, valid_inputs):
        # 1D
        valid_inputs["combined_probability"] = np.full((16,), 0.5, dtype=np.float32)
        with pytest.raises(MaskContractValidationError, match="combined_probability must be 2D, got 1D"):
            MultiChannelDefectMask(**valid_inputs)

        # 3D
        valid_inputs["combined_probability"] = np.full((4, 4, 1), 0.5, dtype=np.float32)
        with pytest.raises(MaskContractValidationError, match="combined_probability must be 2D, got 3D"):
            MultiChannelDefectMask(**valid_inputs)

    def test_probability_empty_array_raises(self, valid_inputs):
        valid_inputs["combined_probability"] = np.zeros((0, 0), dtype=np.float32)
        with pytest.raises(MaskContractValidationError, match="combined_probability cannot be empty"):
            MultiChannelDefectMask(**valid_inputs)

    def test_combined_mask_not_ndarray_raises(self, valid_inputs):
        valid_inputs["combined_mask"] = "invalid_mask"
        with pytest.raises(MaskContractValidationError, match="combined_mask must be a numpy.ndarray"):
            MultiChannelDefectMask(**valid_inputs)

    def test_combined_mask_non_2d_raises(self, valid_inputs):
        valid_inputs["combined_mask"] = np.full((4, 4, 1), 255, dtype=np.uint8)
        with pytest.raises(MaskContractValidationError, match="combined_mask must be 2D, got 3D"):
            MultiChannelDefectMask(**valid_inputs)

    def test_probability_and_mask_shape_mismatch_raises(self, valid_inputs):
        valid_inputs["combined_probability"] = np.full((4, 4), 0.5, dtype=np.float32)
        valid_inputs["combined_mask"] = np.full((4, 5), 255, dtype=np.uint8)
        with pytest.raises(MaskContractValidationError, match="Shape mismatch: probability"):
            MultiChannelDefectMask(**valid_inputs)

    @pytest.mark.parametrize("channel_name", ["scratch_mask", "dust_mask", "uncertain_mask"])
    def test_optional_channel_not_ndarray_raises(self, valid_inputs, channel_name):
        valid_inputs[channel_name] = [0, 255]
        with pytest.raises(MaskContractValidationError, match=f"{channel_name.replace('_mask', '')}_mask must be a numpy.ndarray"):
            MultiChannelDefectMask(**valid_inputs)

    @pytest.mark.parametrize("channel_name", ["scratch_mask", "dust_mask", "uncertain_mask"])
    def test_optional_channel_shape_mismatch_raises(self, valid_inputs, channel_name):
        valid_inputs[channel_name] = np.zeros((3, 3), dtype=np.uint8)
        with pytest.raises(MaskContractValidationError, match=f"Shape mismatch for {channel_name.replace('_mask', '')}_mask"):
            MultiChannelDefectMask(**valid_inputs)

    # 4.2 Dtype enforcement
    @pytest.mark.parametrize("bad_dtype", [np.float64, np.float16, np.int32, np.uint8])
    def test_probability_dtype_must_be_float32(self, valid_inputs, bad_dtype):
        valid_inputs["combined_probability"] = np.full((4, 4), 0.5, dtype=bad_dtype)
        with pytest.raises(MaskContractValidationError, match="combined_probability must be float32"):
            MultiChannelDefectMask(**valid_inputs)

    @pytest.mark.parametrize("bad_dtype", [np.int32, np.float32, np.bool_])
    def test_combined_mask_dtype_must_be_uint8(self, valid_inputs, bad_dtype):
        valid_inputs["combined_mask"] = np.full((4, 4), 255, dtype=bad_dtype)
        with pytest.raises(MaskContractValidationError, match="combined_mask must be uint8"):
            MultiChannelDefectMask(**valid_inputs)

    @pytest.mark.parametrize("channel_name", ["scratch_mask", "dust_mask", "uncertain_mask"])
    @pytest.mark.parametrize("bad_dtype", [np.int32, np.float32, np.bool_])
    def test_optional_channels_dtype_must_be_uint8(self, valid_inputs, channel_name, bad_dtype):
        valid_inputs[channel_name] = np.zeros((4, 4), dtype=bad_dtype)
        with pytest.raises(MaskContractValidationError, match=f"{channel_name.replace('_mask', '')}_mask must be uint8"):
            MultiChannelDefectMask(**valid_inputs)

    # 4.3 Finite float validation
    def test_probability_contains_nan_raises(self, valid_inputs):
        prob = np.full((4, 4), 0.5, dtype=np.float32)
        prob[1, 1] = np.nan
        valid_inputs["combined_probability"] = prob
        with pytest.raises(MaskContractValidationError, match="combined_probability contains non-finite values"):
            MultiChannelDefectMask(**valid_inputs)

    def test_probability_contains_pos_inf_raises(self, valid_inputs):
        prob = np.full((4, 4), 0.5, dtype=np.float32)
        prob[2, 2] = np.inf
        valid_inputs["combined_probability"] = prob
        with pytest.raises(MaskContractValidationError, match="combined_probability contains non-finite values"):
            MultiChannelDefectMask(**valid_inputs)

    def test_probability_contains_neg_inf_raises(self, valid_inputs):
        prob = np.full((4, 4), 0.5, dtype=np.float32)
        prob[0, 3] = -np.inf
        valid_inputs["combined_probability"] = prob
        with pytest.raises(MaskContractValidationError, match="combined_probability contains non-finite values"):
            MultiChannelDefectMask(**valid_inputs)

    # 4.4 Probability range validation
    def test_probability_values_below_zero_raises(self, valid_inputs):
        prob = np.full((4, 4), 0.5, dtype=np.float32)
        prob[0, 0] = -0.001
        valid_inputs["combined_probability"] = prob
        with pytest.raises(MaskContractValidationError, match=r"combined_probability values must be in \[0.0, 1.0\]"):
            MultiChannelDefectMask(**valid_inputs)

    def test_probability_values_above_one_raises(self, valid_inputs):
        prob = np.full((4, 4), 0.5, dtype=np.float32)
        prob[0, 0] = 1.0001
        valid_inputs["combined_probability"] = prob
        with pytest.raises(MaskContractValidationError, match=r"combined_probability values must be in \[0.0, 1.0\]"):
            MultiChannelDefectMask(**valid_inputs)

    def test_probability_boundary_values_succeed(self, valid_inputs):
        prob = np.zeros((4, 4), dtype=np.float32)
        prob[0, 0] = 1.0
        prob[1, 1] = 0.0
        prob[2, 2] = 0.5
        valid_inputs["combined_probability"] = prob
        mask = MultiChannelDefectMask(**valid_inputs)
        assert mask.combined_probability[0, 0] == 1.0
        assert mask.combined_probability[1, 1] == 0.0

    # 4.5 Strict binary mask validation
    @pytest.mark.parametrize("invalid_val", [1, 2, 127, 128, 254])
    def test_combined_mask_invalid_binary_values_raises(self, valid_inputs, invalid_val):
        mask = np.full((4, 4), 0, dtype=np.uint8)
        mask[1, 1] = invalid_val
        valid_inputs["combined_mask"] = mask
        with pytest.raises(MaskContractValidationError, match="combined_mask must contain strictly binary values: 0 or 255"):
            MultiChannelDefectMask(**valid_inputs)

    @pytest.mark.parametrize("channel_name", ["scratch_mask", "dust_mask", "uncertain_mask"])
    @pytest.mark.parametrize("invalid_val", [1, 100, 254])
    def test_optional_channels_invalid_binary_values_raises(self, valid_inputs, channel_name, invalid_val):
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[0, 0] = invalid_val
        valid_inputs[channel_name] = arr
        with pytest.raises(
            MaskContractValidationError,
            match=f"{channel_name.replace('_mask', '')}_mask must contain strictly binary values: 0 or 255",
        ):
            MultiChannelDefectMask(**valid_inputs)

    # 4.6 Channel provenance consistency checks
    def test_provenance_marked_available_with_none_mask_raises(self, valid_inputs):
        valid_inputs["scratch_mask"] = None
        valid_inputs["provenance"] = {
            "scratch": ChannelProvenance(
                origin=ChannelOrigin.NATIVE,
                is_available=True,  # available, but array is None!
                source_algorithm="test_algo",
            )
        }
        with pytest.raises(
            MaskContractValidationError,
            match="Channel scratch marked available in provenance but array is None",
        ):
            MultiChannelDefectMask(**valid_inputs)

    def test_provenance_marked_unavailable_with_provided_mask_raises(self, valid_inputs):
        valid_inputs["dust_mask"] = np.zeros((4, 4), dtype=np.uint8)
        valid_inputs["provenance"] = {
            "dust": ChannelProvenance(
                origin=ChannelOrigin.UNSUPPORTED,
                is_available=False,  # unavailable, but array is given!
                source_algorithm="none",
            )
        }
        with pytest.raises(
            MaskContractValidationError,
            match="Channel dust marked unavailable in provenance but array is provided",
        ):
            MultiChannelDefectMask(**valid_inputs)

    def test_provenance_consistent_success(self, valid_inputs):
        valid_inputs["scratch_mask"] = np.full((4, 4), 255, dtype=np.uint8)
        valid_inputs["dust_mask"] = None
        valid_inputs["provenance"] = {
            "scratch": ChannelProvenance(ChannelOrigin.NATIVE, True, "native_scratch"),
            "dust": ChannelProvenance(ChannelOrigin.UNSUPPORTED, False, "none"),
        }
        mask = MultiChannelDefectMask(**valid_inputs)
        assert mask.scratch_mask is not None
        assert mask.dust_mask is None


# ==============================================================================
# 5. Legacy Compatibility Bridge (to_legacy_compat_bridge)
# ==============================================================================


class TestLegacyCompatibilityBridge:
    """Verifies that to_legacy_compat_bridge adheres to the 100% pixel-identical invariant

    and outputs a strictly JSON-serializable manifest with zero raw numpy types.
    """

    @pytest.mark.parametrize(
        "policy",
        [
            RepairInclusionPolicy.CONSERVATIVE,
            RepairInclusionPolicy.HIGH_RECALL,
            RepairInclusionPolicy.SCRATCH_ONLY,
            RepairInclusionPolicy.COMBINED_DEFECT,
        ],
    )
    def test_legacy_bridge_union_invariant_for_all_policies(self, base_2d_masks, policy):
        """Invariant: np.bitwise_or(scratch, dust) == get_effective_repair_mask(policy)

        must hold 100% pixel-identically for all policies.
        """
        mask_obj = base_2d_masks
        effective = mask_obj.get_effective_repair_mask(policy)
        s, d, manifest = mask_obj.to_legacy_compat_bridge(policy)

        # 1. Union invariant
        legacy_union = np.bitwise_or(s, d)
        assert np.array_equal(legacy_union, effective), (
            f"Policy {policy} broke the legacy union invariant: "
            f"bitwise_or(scratch, dust) != get_effective_repair_mask"
        )

        # 2. Dtype and shape invariants
        assert s.dtype == np.uint8
        assert d.dtype == np.uint8
        assert s.shape == effective.shape
        assert d.shape == effective.shape

    def test_legacy_bridge_manifest_json_serializability_and_types(self, base_2d_masks):
        """Manifest must serialize cleanly with json.dumps and contain no raw numpy types."""
        mask_obj = base_2d_masks
        _, _, manifest = mask_obj.to_legacy_compat_bridge(RepairInclusionPolicy.CONSERVATIVE)

        # Verify json.dumps succeeds without TypeError
        serialized = json.dumps(manifest)
        assert isinstance(serialized, str)
        deserialized = json.loads(serialized)
        assert deserialized == manifest

        # Explicit check for zero numpy scalar types or ndarrays
        forbidden_numpy_types = (np.integer, np.floating, np.bool_, np.ndarray)
        for key, value in manifest.items():
            assert not isinstance(value, forbidden_numpy_types), (
                f"Manifest field '{key}' has raw numpy type {type(value)}; must be a Python native primitive"
            )
            # Verify it is one of standard json primitive types
            assert isinstance(value, (int, float, str, bool, list, dict, type(None))), (
                f"Manifest field '{key}' has unsupported type {type(value)}"
            )

        assert "policy_resolved" in manifest
        assert manifest["policy_resolved"] == "conservative"
        assert "mask_pixel_count" in manifest
        assert isinstance(manifest["mask_pixel_count"], int)

    def test_legacy_bridge_combined_only_model_behavior(self):
        """Combined-only models:

        - is_compatibility_mapping is True
        - includes explicit warning against dust-specific erosion
        - delivers effective mask via dust channel without silent drop
        - scratch channel is strictly empty (zeros)
        """
        prob = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
        combined = np.array([[0, 255], [255, 0]], dtype=np.uint8)

        mask_obj = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=None,
            dust_mask=None,
            uncertain_mask=None,
        )

        s, d, manifest = mask_obj.to_legacy_compat_bridge(RepairInclusionPolicy.COMBINED_DEFECT)

        # Invariant: np.bitwise_or(s, d) matches combined_mask exactly
        assert np.array_equal(np.bitwise_or(s, d), combined)

        # Scratch channel is all zeros
        assert np.all(s == 0)

        # Dust channel carries the full combined mask
        assert np.array_equal(d, combined)

        # Compatibility manifest checks
        assert manifest["is_compatibility_mapping"] is True
        assert manifest["semantic_type"] == "generic_combined_defect"
        assert manifest["policy_resolved"] == "combined_defect"
        assert manifest["mask_pixel_count"] == 2
        assert "warning" in manifest
        assert "do not apply dust-specific erosion" in manifest["warning"]

        # json serializable
        assert json.dumps(manifest) is not None

    def test_legacy_bridge_combined_only_model_rejects_conservative_policy(self):
        """Calling to_legacy_compat_bridge with CONSERVATIVE on combined-only model raises UnsupportedPolicyError."""
        prob = np.full((2, 2), 0.5, dtype=np.float32)
        combined = np.full((2, 2), 255, dtype=np.uint8)

        mask_obj = MultiChannelDefectMask(
            combined_probability=prob,
            combined_mask=combined,
            scratch_mask=None,
            dust_mask=None,
        )

        with pytest.raises(UnsupportedPolicyError, match="Policy 'CONSERVATIVE' requires Scratch or Dust"):
            mask_obj.to_legacy_compat_bridge(RepairInclusionPolicy.CONSERVATIVE)

    def test_legacy_bridge_string_policy_conversion(self, base_2d_masks):
        """to_legacy_compat_bridge correctly handles string policy parameter."""
        s, d, manifest = base_2d_masks.to_legacy_compat_bridge("high_recall")
        assert manifest["policy_resolved"] == "high_recall"
        assert np.array_equal(
            np.bitwise_or(s, d),
            base_2d_masks.get_effective_repair_mask(RepairInclusionPolicy.HIGH_RECALL),
        )

    def test_legacy_bridge_invalid_policy_string_raises(self, base_2d_masks):
        """to_legacy_compat_bridge raises UnsupportedPolicyError for invalid policy string."""
        with pytest.raises(UnsupportedPolicyError, match="Unknown or unsupported RepairInclusionPolicy"):
            base_2d_masks.to_legacy_compat_bridge("unknown_policy_type")


# ==============================================================================
# 6. Serialization & Method Contract Tests
# ==============================================================================


class TestMultiChannelDefectMaskContractDetails:
    """Verifies as_dict serialization and metadata integrity."""

    def test_as_dict_structure_and_copy(self, base_2d_masks):
        """as_dict returns a complete dictionary with copy of metadata and provenance dicts."""
        base_2d_masks.metadata["custom_param"] = 42
        d = base_2d_masks.as_dict()

        assert "combined_probability" in d
        assert "combined_mask" in d
        assert "scratch_mask" in d
        assert "dust_mask" in d
        assert "uncertain_mask" in d
        assert "provenance" in d
        assert "metadata" in d

        assert d["metadata"]["custom_param"] == 42
        # Mutating dictionary representation does not mutate the original object
        d["metadata"]["custom_param"] = 999
        assert base_2d_masks.metadata["custom_param"] == 42

        # Provenance entries are converted to dicts
        assert isinstance(d["provenance"]["scratch"], dict)
        assert d["provenance"]["scratch"]["origin"] == ChannelOrigin.NATIVE
        assert d["provenance"]["scratch"]["is_available"] is True

    def test_readiness_state_and_channel_origin_enums(self):
        """Enum sanity checks for readiness and channel origin."""
        assert ReadinessState.PRETRAINED_OPERATIONAL.value == "pretrained_operational"
        assert ReadinessState.TRAINED_EXPERIMENTAL.value == "trained_experimental"
        assert ReadinessState.TRAINING_REQUIRED.value == "training_required"
        assert ReadinessState.BLOCKED.value == "blocked"

        assert ChannelOrigin.NATIVE.value == "native"
        assert ChannelOrigin.DERIVED.value == "derived"
        assert ChannelOrigin.UNSUPPORTED.value == "unsupported"
