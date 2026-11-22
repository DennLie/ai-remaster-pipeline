"""GPU-free tests for the DDColor Reference Generator script, wrapper, and integration."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from ai_remaster_gui.commercial_policy import (
    CommercialPolicyError,
    enforce_component,
    evaluate_component,
    stage_issue,
)
from ai_remaster_gui.config import ROOT
from ai_remaster_gui.runtime_settings import base_settings
from scripts.ddcolor_reference import (
    DDColorModel,
    build_parser,
    colorize_frame,
    load_model,
    process_manifest,
    read_manifest_lines,
    resolve_device,
    write_manifest_lines,
)
import scripts.ddcolor_reference as ddcolor_script


class DDColorReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def _create_test_image(self, width: int = 64, height: int = 64) -> tuple[Path, np.ndarray]:
        """Create a synthetic test image file with varying luminance and color."""
        img_path = self.tmp_dir / f"test_input_{width}x{height}.png"
        # Gradient pattern with diverse luminance levels
        y, x = np.mgrid[0:height, 0:width]
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        arr[:, :, 0] = (x * (255 / max(1, width - 1))).astype(np.uint8)
        arr[:, :, 1] = (y * (255 / max(1, height - 1))).astype(np.uint8)
        arr[:, :, 2] = ((x + y) * (127 / max(1, width + height - 2))).astype(np.uint8)
        Image.fromarray(arr).save(img_path)
        return img_path, arr

    # --------------------------------------------------------------------------
    # 1. CLI Parameter Parsing and Wrapper Invocation
    # --------------------------------------------------------------------------

    def test_cli_parser_defaults_and_options(self):
        parser = build_parser()

        # Requires either --manifest or --input-image when invoked via CLI main
        with patch.object(sys, "argv", ["ddcolor_reference.py"]):
            with self.assertRaises(SystemExit) as ctx:
                ddcolor_script.main()
            self.assertEqual(ctx.exception.code, 2)

        # Parses input-image mode
        args = parser.parse_args(["--input-image", "input.png", "--output", "out.png", "--dry-run"])
        self.assertEqual(args.input_image, Path("input.png"))
        self.assertEqual(args.output, Path("out.png"))
        self.assertTrue(args.dry_run)
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.input_size, 512)

        # Parses manifest mode with custom parameters
        args = parser.parse_args([
            "--manifest", "shots.csv",
            "--output-root", "out_stills",
            "--device", "cpu",
            "--seed", "1234",
            "--limit", "5",
            "--compliance-output-dir", "comp_out",
        ])
        self.assertEqual(args.manifest, Path("shots.csv"))
        self.assertEqual(args.output_root, Path("out_stills"))
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.seed, 1234)
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.compliance_output_dir, Path("comp_out"))

    def test_wrapper_invocation_and_help(self):
        wrapper_path = ROOT / "wrappers" / "ddcolor_reference.bat"
        self.assertTrue(wrapper_path.is_file(), f"Wrapper script missing: {wrapper_path}")

        # Run wrapper with --help on Windows
        if sys.platform == "win32":
            res = subprocess.run(
                ["cmd.exe", "/c", str(wrapper_path), "--help"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(res.returncode, 0)
            self.assertIn("DDColor fast deterministic reference draft generator", res.stdout)
            self.assertIn("--dry-run", res.stdout)
            self.assertIn("--manifest", res.stdout)
            self.assertIn("--input-image", res.stdout)

    def test_cli_main_dry_run_single_image(self):
        img_path, _ = self._create_test_image(32, 32)
        out_path = self.tmp_dir / "out_cli.png"
        comp_dir = self.tmp_dir / "compliance"

        test_args = [
            "ddcolor_reference.py",
            "--input-image", str(img_path),
            "--output", str(out_path),
            "--dry-run",
            "--seed", "42",
            "--compliance-output-dir", str(comp_dir),
        ]
        with patch.object(sys, "argv", test_args):
            exit_code = ddcolor_script.main()
            self.assertEqual(exit_code, 0)

        self.assertTrue(out_path.is_file())
        with Image.open(out_path) as im:
            self.assertEqual(im.size, (32, 32))

        # Check compliance report recorded
        comp_report = comp_dir / "compliance_report.json"
        self.assertTrue(comp_report.is_file())
        with comp_report.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertTrue(data["overall_commercial_compliant"])
        self.assertEqual(data["runs"][0]["stage"], "references")
        self.assertEqual(data["runs"][0]["models"][0]["model_id"], "ddcolor")
        self.assertEqual(data["runs"][0]["seeds"], {"references": 42})

    # --------------------------------------------------------------------------
    # 2. Deterministic Synthetic Dry-Run Inference Preserving Luminance
    # --------------------------------------------------------------------------

    def test_dry_run_strictly_preserves_grayscale_luminance(self):
        """Verify 'The original pixels win': luminance L in Lab space is preserved within integer conversion tolerance."""
        # 1. Midtone grayscale image test (where sRGB gamut clipping does not occur)
        h, w = 48, 48
        gray_midtones = np.random.randint(60, 200, size=(h, w), dtype=np.uint8)
        out_mid_rgb = colorize_frame(gray_midtones, model_path="fake.pth", dry_run=True, seed=42)
        self.assertEqual(out_mid_rgb.shape, (h, w, 3))
        self.assertEqual(out_mid_rgb.dtype, np.uint8)

        # Extract L channel from input and output
        in_mid_rgb = cv2.cvtColor(gray_midtones, cv2.COLOR_GRAY2RGB)
        orig_l_mid = cv2.cvtColor(in_mid_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        out_l_mid = cv2.cvtColor(out_mid_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]

        # On midtones, L must be preserved within integer rounding tolerance (atol=3)
        np.testing.assert_allclose(
            orig_l_mid,
            out_l_mid,
            atol=3,
            err_msg="Midtone luminance L must be preserved within integer conversion tolerance (atol=3)",
        )
        self.assertLess(
            float(np.mean(np.abs(orig_l_mid.astype(float) - out_l_mid.astype(float)))),
            0.5,
            "Mean luminance error across midtones must be minimal (<0.5)",
        )

        # 2. Full dynamic range [0, 255] gradient test
        y, x = np.mgrid[0:h, 0:w]
        gray_gradient = (x * (255 / (w - 1))).astype(np.uint8)
        out_grad_rgb = colorize_frame(gray_gradient, model_path="fake.pth", dry_run=True, seed=42)
        in_grad_rgb = cv2.cvtColor(gray_gradient, cv2.COLOR_GRAY2RGB)
        orig_l_grad = cv2.cvtColor(in_grad_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        out_l_grad = cv2.cvtColor(out_grad_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]

        # Across full dynamic range (including deep shadows where sRGB gamut limits apply),
        # max deviation is bounded (atol=10) and mean absolute deviation is under 1.0.
        np.testing.assert_allclose(
            orig_l_grad,
            out_l_grad,
            atol=10,
            err_msg="Full-range luminance must be preserved within gamut conversion bounds (atol=10)",
        )
        self.assertLess(
            float(np.mean(np.abs(orig_l_grad.astype(float) - out_l_grad.astype(float)))),
            1.0,
            "Mean luminance error across gradient must be under 1.0",
        )

        # 3. Grayscale 3D array input (H, W, 1)
        gray_3d = np.expand_dims(gray_midtones, axis=2)
        out_rgb_gray3d = colorize_frame(gray_3d, model_path="fake.pth", dry_run=True, seed=42)
        out_l_3d = cv2.cvtColor(out_rgb_gray3d, cv2.COLOR_RGB2LAB)[:, :, 0]
        np.testing.assert_allclose(orig_l_mid, out_l_3d, atol=3)

    def test_dry_run_determinism_and_seed_variation(self):
        h, w = 32, 32
        gray_midtones = np.random.randint(60, 200, size=(h, w), dtype=np.uint8)

        # Same seed produces bit-identical RGB output
        out1 = colorize_frame(gray_midtones, model_path="fake.pth", dry_run=True, seed=999)
        out2 = colorize_frame(gray_midtones, model_path="fake.pth", dry_run=True, seed=999)
        np.testing.assert_array_equal(out1, out2)

        # Different seed produces distinct chromaticity while preserving luminance
        out3 = colorize_frame(gray_midtones, model_path="fake.pth", dry_run=True, seed=111)
        self.assertFalse(np.array_equal(out1, out3), "Different seeds should produce different color tones")

        lab1 = cv2.cvtColor(out1, cv2.COLOR_RGB2LAB)
        lab3 = cv2.cvtColor(out3, cv2.COLOR_RGB2LAB)
        np.testing.assert_allclose(
            lab1[:, :, 0],
            lab3[:, :, 0],
            atol=2,
            err_msg="Luminance must remain preserved across seeds within integer conversion tolerance",
        )

    # --------------------------------------------------------------------------
    # 3. Fail-Closed Behavior on Missing Weights
    # --------------------------------------------------------------------------

    def test_missing_weights_raises_fatal_file_not_found(self):
        missing_path = self.tmp_dir / "nonexistent_ddcolor_weights_9999.pth"
        _, rgb_arr = self._create_test_image(16, 16)

        # Running colorize_frame without dry-run when weights are missing MUST raise FileNotFoundError
        with self.assertRaises(FileNotFoundError) as ctx:
            colorize_frame(rgb_arr, model_path=missing_path, dry_run=False)
        self.assertIn("DDColor model checkpoint not found", str(ctx.exception))

        # load_model also fails closed
        with self.assertRaises(FileNotFoundError) as ctx:
            load_model(missing_path)
        self.assertIn("DDColor model weights not found", str(ctx.exception))

        # CLI invocation without --dry-run fails closed
        img_path, _ = self._create_test_image(16, 16)
        test_args = [
            "ddcolor_reference.py",
            "--input-image", str(img_path),
            "--model-path", str(missing_path),
        ]
        with patch.object(sys, "argv", test_args):
            with self.assertRaises(FileNotFoundError):
                ddcolor_script.main()

    # --------------------------------------------------------------------------
    # 4. Mocked Neural Inference in CI without GPU
    # --------------------------------------------------------------------------

    def test_mocked_neural_inference_pipeline(self):
        """Verify real non-dry-run neural execution path on CPU with mocked PyTorch weights."""
        orig_h, orig_w = 40, 40
        _, rgb_arr = self._create_test_image(orig_w, orig_h)
        orig_lab = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB)
        orig_l = orig_lab[:, :, 0]

        # Construct a real PyTorch module or mock model on CPU
        mock_model = DDColorModel(in_channels=3, out_channels=2)
        mock_model.eval()

        dummy_path = self.tmp_dir / "mock_weights.pth"
        dummy_path.write_bytes(b"dummy checkpoint content")

        # Colorize through neural path (dry_run=False)
        out_rgb = colorize_frame(
            rgb_arr,
            model_path=dummy_path,
            device="cpu",
            dry_run=False,
            input_size=64,
            model=mock_model,
        )

        self.assertEqual(out_rgb.shape, (orig_h, orig_w, 3))
        self.assertEqual(out_rgb.dtype, np.uint8)

        # Verify luminance preservation through neural path within integer conversion tolerance
        out_lab = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2LAB)
        np.testing.assert_allclose(
            orig_l,
            out_lab[:, :, 0],
            atol=10,
            err_msg="Neural colorization path must preserve original luminance L within conversion bounds",
        )

        # Test with neutral mock model (zero chroma) preserving exact luminance
        zero_model = MagicMock()
        zero_model.to.return_value = zero_model
        zero_model.eval.return_value = zero_model
        zero_model.return_value = torch.zeros((1, 2, 64, 64), dtype=torch.float32)

        out_zero_rgb = colorize_frame(
            rgb_arr,
            model_path=dummy_path,
            device="cpu",
            dry_run=False,
            input_size=64,
            model=zero_model,
        )
        out_zero_lab = cv2.cvtColor(out_zero_rgb, cv2.COLOR_RGB2LAB)
        np.testing.assert_allclose(
            orig_l,
            out_zero_lab[:, :, 0],
            atol=2,
            err_msg="Zero-chroma neural path must preserve original luminance within atol=2",
        )

    def test_mocked_checkpoint_loading_via_torch_load(self):
        """Verify load_model correctly unpacks state_dict checkpoint from torch.load."""
        dummy_path = self.tmp_dir / "mock_checkpoint.pth"
        dummy_path.write_bytes(b"mock state dict")

        # Create dummy state dict matching DDColorModel
        dummy_state = DDColorModel().state_dict()
        with patch("torch.load", return_value={"state_dict": dummy_state}):
            loaded = load_model(dummy_path, device="cpu")
            self.assertIsInstance(loaded, nn.Module)

        # Mismatched checkpoint dictionary (0 matching keys) raises ValueError
        with patch("torch.load", return_value={"state_dict": {"completely_unrelated_layer.weight": torch.zeros(10)}}):
            with self.assertRaises(ValueError) as ctx:
                load_model(dummy_path, device="cpu")
            self.assertIn("does not match DDColorModel architecture (0 matching keys)", str(ctx.exception))

        # Unsupported checkpoint object type raises ValueError
        with patch("torch.load", return_value="invalid_checkpoint_object"):
            with self.assertRaises(ValueError) as ctx:
                load_model(dummy_path, device="cpu")
            self.assertIn("unsupported format", str(ctx.exception))

    def test_ddcolor_references_stage_does_not_require_comfy(self):
        """Verify REV-MAJ-01: DDColor references stage does not trigger ensure_comfy_available_for_stage."""
        from ai_remaster_gui.server import PipelineApp
        app_inst = PipelineApp.__new__(PipelineApp)
        app_inst.settings = {
            "global": {"source": "fake_input.mp4", "colorize": "true", "commercial_model_guard": "false"},
            "cloud": {},
            "references": {"method": "ddcolor", "manifest": "shots.csv"},
        }
        app_inst.quitting = False
        app_inst.colorize_enabled = lambda: True
        app_inst.upscale_enabled = lambda: False
        app_inst.cleanup_enabled = lambda: False
        app_inst.stabilize_enabled = lambda: False
        app_inst.detail_enabled = lambda: False
        app_inst.preprocessing_problem = lambda _: ""
        app_inst.ensure_pipeline_source = lambda: None
        app_inst.lock = MagicMock()
        app_inst.process = None
        app_inst.stage_stamp = MagicMock(return_value="")
        app_inst.command_for = MagicMock(return_value=["echo", "done"])
        app_inst.log = []

        with (
            patch("ai_remaster_gui.server.ensure_comfy_available_for_stage") as mock_ensure,
            patch("subprocess.Popen") as mock_popen,
        ):
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            mock_popen.return_value = mock_proc

            # 1. ddcolor method does NOT require comfy
            app_inst.run_stage("references")
            mock_ensure.assert_not_called()

            # 2. qwen method DOES require comfy
            app_inst.settings["references"]["method"] = "qwen"
            mock_ensure.return_value = (True, "")
            app_inst.run_stage("references")
            self.assertTrue(mock_ensure.called)

    # --------------------------------------------------------------------------
    # 5. Manifest Processing and Header Comment Preservation
    # --------------------------------------------------------------------------

    def test_manifest_processing_preserves_comments_and_updates_color_reference(self):
        # 1. Create source still images
        shot1_path = self.tmp_dir / "shot_001.png"
        shot2_path = self.tmp_dir / "shot_002.png"
        shot3_path = self.tmp_dir / "shot_003.png"
        _, arr1 = self._create_test_image(32, 32)
        Image.fromarray(arr1).save(shot1_path)
        Image.fromarray(arr1).save(shot2_path)
        Image.fromarray(arr1).save(shot3_path)

        # 2. Write CSV manifest with leading '#' comments
        manifest_path = self.tmp_dir / "references_manifest.csv"
        manifest_lines = [
            "# AI Remaster Pipeline Shot References Manifest\n",
            "# Source Video: test_archive_1928.mp4\n",
            "# Generated by shots detector stage\n",
            "index,shot_id,source_reference,reference,enabled\n",
            f"0,shot_1,{shot1_path.name},{shot1_path.name},true\n",
            f"1,shot_2,{shot2_path.name},{shot2_path.name},true\n",
            f"2,shot_3,{shot3_path.name},{shot3_path.name},false\n",
        ]
        manifest_path.write_text("".join(manifest_lines), encoding="utf-8")

        # 3. Process manifest in dry-run mode
        out_root = self.tmp_dir / "colorized_stills"
        out_manifest = self.tmp_dir / "updated_manifest.csv"

        # Mock resolve_path to locate shot stills in tmp_dir
        with patch("scripts.ddcolor_reference.resolve_path", side_effect=lambda p: (self.tmp_dir / p) if not Path(p).is_absolute() else Path(p)):
            count = process_manifest(
                manifest_path=manifest_path,
                output_manifest_path=out_manifest,
                output_root=out_root,
                model_path=Path("dummy.pth"),
                device="cpu",
                dry_run=True,
                seed=42,
            )

        self.assertEqual(count, 2, "Only enabled shots (shot_1, shot_2) should be processed")

        # 4. Verify output files created
        self.assertTrue((out_root / shot1_path.name).is_file())
        self.assertTrue((out_root / shot2_path.name).is_file())
        self.assertFalse((out_root / shot3_path.name).is_file(), "Disabled shot_3 must not be colorized")

        # 5. Verify manifest content and comment preservation
        self.assertTrue(out_manifest.is_file())
        raw_manifest_text = out_manifest.read_text(encoding="utf-8")
        self.assertTrue(raw_manifest_text.startswith("# AI Remaster Pipeline Shot References Manifest"))
        self.assertIn("# Source Video: test_archive_1928.mp4", raw_manifest_text)
        self.assertIn("# Generated by shots detector stage", raw_manifest_text)

        comments, fieldnames, rows = read_manifest_lines(out_manifest)
        self.assertEqual(len(comments), 3)
        self.assertIn("color_reference", fieldnames)
        self.assertEqual(len(rows), 3)

        # Check color_reference column values
        self.assertIn(shot1_path.name, rows[0]["color_reference"])
        self.assertIn(shot2_path.name, rows[1]["color_reference"])
        self.assertEqual(rows[2].get("color_reference", ""), "", "Disabled shot must not have color_reference")

    def test_manifest_limit_parameter(self):
        shot1 = self.tmp_dir / "limit_1.png"
        shot2 = self.tmp_dir / "limit_2.png"
        _, arr = self._create_test_image(16, 16)
        Image.fromarray(arr).save(shot1)
        Image.fromarray(arr).save(shot2)

        manifest = self.tmp_dir / "limit_manifest.csv"
        manifest.write_text(
            f"index,source_reference,reference,enabled\n0,{shot1.name},{shot1.name},true\n1,{shot2.name},{shot2.name},true\n",
            encoding="utf-8",
        )

        out_root = self.tmp_dir / "limit_out"
        with patch("scripts.ddcolor_reference.resolve_path", side_effect=lambda p: (self.tmp_dir / p) if not Path(p).is_absolute() else Path(p)):
            count = process_manifest(
                manifest_path=manifest,
                output_manifest_path=None,
                output_root=out_root,
                model_path=Path("dummy.pth"),
                dry_run=True,
                limit=1,
            )
        self.assertEqual(count, 1)

    # --------------------------------------------------------------------------
    # 6. Acceptance Gate Verification
    # --------------------------------------------------------------------------

    def test_acceptance_gate_dry_run_reference_pass_commercial_models(self):
        """Acceptance Gate: dry-run commercial reference pass succeeds with DDColor and audited Qwen-2511."""
        settings = base_settings()
        settings["global"]["commercial_model_guard"] = "true"
        settings["global"]["non_commercial_benchmark_override"] = "false"

        # DDColor in references succeeds
        self.assertEqual(stage_issue(settings, "references", model_override="ddcolor"), "")
        dec_ddcolor = evaluate_component("ddcolor")
        self.assertTrue(dec_ddcolor.allowed)
        self.assertEqual(dec_ddcolor.license, "Apache-2.0")

        # Qwen-2511 in references succeeds when audited
        settings["global"]["qwen_2511_audited"] = "true"
        self.assertEqual(stage_issue(settings, "references", model_override="qwen"), "")
        dec_qwen = evaluate_component("qwen-image-edit-2511", conditional_attestations={"qwen_2511_audited": True})
        self.assertTrue(dec_qwen.allowed)

    def test_acceptance_gate_reference_pass_fails_cleanly_on_non_commercial_models(self):
        """Acceptance Gate: reference pass fails cleanly when any NC model is selected."""
        settings = base_settings()
        settings["global"]["commercial_model_guard"] = "true"

        # 1. Qwen-Image-2.1 (research-only)
        issue = stage_issue(settings, "references", model_override="qwen21")
        self.assertTrue(issue)
        self.assertIn("Qwen-Image-2.1 has a research-only licence", issue)
        with self.assertRaises(CommercialPolicyError):
            enforce_component("qwen21")

        # 2. ColorMNet / CMNET2 (CC BY-NC-SA)
        for nc_col in ("colormnet", "cmnet2", "both"):
            issue = stage_issue(settings, "colour", model_override=nc_col)
            self.assertTrue(issue)
            self.assertIn("non-commercial", issue)
            with self.assertRaises(CommercialPolicyError):
                enforce_component(nc_col)

        # 3. ProPainter (NTU S-Lab License 1.0)
        issue = stage_issue(settings, "dust", model_override="propainter")
        self.assertTrue(issue)
        self.assertIn("ProPainter is non-commercial", issue)
        with self.assertRaises(CommercialPolicyError):
            enforce_component("propainter")

        # 4. MiniMax H3 (territorial / non-commercial restrictions)
        issue = stage_issue(settings, "outpaint", model_override="h3")
        self.assertTrue(issue)
        self.assertIn("MiniMax H3", issue)
        with self.assertRaises(CommercialPolicyError):
            enforce_component("h3")

        # 5. SAM 3.1 (non-commercial research)
        dec_sam = evaluate_component("sam3.1")
        self.assertFalse(dec_sam.allowed)
        self.assertEqual(dec_sam.status, "hard_blocked")
        with self.assertRaises(CommercialPolicyError):
            enforce_component("sam3.1")


if __name__ == "__main__":
    unittest.main()
