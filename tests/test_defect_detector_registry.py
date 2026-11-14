"""Unit and integration tests for BaseDefectDetector, DefectDetectorRegistry, CLI, and GUI integration."""
from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import artifact_ids as aid  # noqa: E402
import benchmark_defect_masks as bench  # noqa: E402
import cleanup_video  # noqa: E402
import film_defect_detectors as fdd  # noqa: E402
import film_defect_masks as fdm  # noqa: E402
from ai_remaster_gui import dust  # noqa: E402


class _MockCustomDetector(fdd.BaseDefectDetector):
    name = "custom_test_detector"
    aliases = ("custom_alias",)
    description = "Unit-test custom strategy"
    license_status = "MIT"
    commercial_cleared = True
    temporal_radius = 0

    @property
    def schema(self) -> dict[str, dict[str, Any]]:
        return {
            "sensitivity": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            "window": {"type": "int", "default": 3, "min": 1, "max": 9},
            "strict": {"type": "bool", "default": False},
            "mode": {"type": "str", "default": "fast", "choices": ("fast", "accurate")},
            "scales": {"type": "tuple", "default": (2, 4)},
        }

    def detect(self, frames: Any, config: Any = None) -> fdd.DefectMaskResult:
        params = self.validate_config(config)
        seq = fdd.normalize_frames(frames)
        n = len(seq.gray_frames)
        h, w = seq.height, seq.width
        masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(n)]
        if params["strict"]:
            for m in masks:
                m[0, 0] = 255
        return fdd.DefectMaskResult(
            mask=masks,
            scratch_mask=[m.copy() for m in masks],
            dust_mask=[np.zeros((h, w), dtype=np.uint8) for _ in range(n)],
            metadata={"detector": self.name, "config": params},
        )


class _MockRestrictedDetector(_MockCustomDetector):
    name = "restricted_research_detector"
    aliases = ("restricted_alias",)
    license_status = "Non-Commercial Research Only"
    commercial_cleared = False


class DefectDetectorRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(77)
        self.clean_frames = [
            rng.integers(112, 122, size=(64, 96, 3), dtype=np.uint8)
            for _ in range(6)
        ]

    def test_contract_and_registry_discovery_and_aliases(self) -> None:
        registry = fdd.build_default_registry()
        available = registry.list_available(commercial_only=True)
        for expected in ("farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy"):
            self.assertIn(expected, available)
        self.assertNotIn("dart_harness", available)

        all_with_restricted = registry.list_available(commercial_only=False)
        self.assertIn("dart_harness", all_with_restricted)

        with_aliases = registry.list_available(commercial_only=True, include_aliases=True)
        for alias in ("temporal", "dis", "multiscale_ridge", "median_5f", "spatial_scratch"):
            self.assertIn(alias, with_aliases)

        # Primary and alias lookups resolve to identical strategy instances
        self.assertIs(registry.get("farneback_3f"), registry.get("temporal"))
        self.assertIs(registry.get("dis_flow"), registry.get("dis"))
        self.assertIs(registry.get("ridge"), registry.get("multiscale_ridge"))
        self.assertIs(registry.get("consensus_5f"), registry.get("median_5f"))
        self.assertIs(registry.get("legacy"), registry.get("spatial_scratch"))

        # Schema and descriptor inspection
        descriptions = registry.describe_all(commercial_only=True)
        self.assertEqual(len(descriptions), 5)
        for desc in descriptions:
            self.assertTrue(desc["commercial_cleared"])
            self.assertIn("sensitivity", desc["schema"])

    def test_custom_registration_replacement_and_unregistration(self) -> None:
        registry = fdd.DefectDetectorRegistry()
        inst = registry.register(_MockCustomDetector)
        self.assertIsInstance(inst, fdd.BaseDefectDetector)
        self.assertTrue(registry.has("custom_test_detector"))
        self.assertTrue(registry.has("custom_alias"))
        self.assertIs(registry.get("custom_alias"), inst)

        # Duplicate registration without replace=True fails
        with self.assertRaises(ValueError):
            registry.register(_MockCustomDetector)

        # Replacement succeeds with replace=True
        inst2 = registry.register(_MockCustomDetector(), replace=True)
        self.assertIs(registry.get("custom_test_detector"), inst2)

        # Invalid detector types rejected
        with self.assertRaises(TypeError):
            registry.register("not_a_detector")  # type: ignore[arg-type]

        # Unregister removes primary name and aliases
        registry.unregister("custom_alias")
        self.assertFalse(registry.has("custom_test_detector"))
        self.assertFalse(registry.has("custom_alias"))
        with self.assertRaises(fdd.UnknownDetectorError):
            registry.get("custom_test_detector")
        with self.assertRaises(fdd.UnknownDetectorError):
            registry.unregister("nonexistent")

    def test_commercial_safety_firewall_and_imports(self) -> None:
        # 1. Verify AST of film_defect_detectors.py contains only commercial-safe imports
        source_path = ROOT / "scripts" / "film_defect_detectors.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden_modules = {"torch", "torchvision", "dart", "rtn", "propainter"}
        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0].lower())
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module.split(".")[0].lower())
        self.assertTrue(imported_modules.isdisjoint(forbidden_modules))

        # 2. Verify commercial_only=True blocks un-cleared / research detectors
        registry = fdd.build_default_registry()
        registry.register(_MockRestrictedDetector())

        for restricted_name in ("dart_harness", "dart", "restricted_research_detector", "restricted_alias"):
            self.assertFalse(registry.has(restricted_name, commercial_only=True))
            self.assertTrue(registry.has(restricted_name, commercial_only=False))
            with self.assertRaises(fdd.CommercialLicenseError):
                registry.get(restricted_name, commercial_only=True)
            with self.assertRaises(PermissionError):
                registry.detect(restricted_name, self.clean_frames, commercial_only=True)

            # Allowed only when commercial_only=False is explicitly set for isolated evaluation
            det = registry.get(restricted_name, commercial_only=False)
            self.assertFalse(det.commercial_cleared)

    def test_parameter_schema_validation_across_all_types(self) -> None:
        det = _MockCustomDetector()
        defaults = det.validate_config(None)
        self.assertEqual(
            defaults,
            {
                "sensitivity": 0.5,
                "window": 3,
                "strict": False,
                "mode": "fast",
                "scales": (2, 4),
            },
        )

        valid = det.validate_config({
            "sensitivity": "0.8",
            "window": 5,
            "strict": "true",
            "mode": "accurate",
            "scales": [3, 5, 7],
        })
        self.assertEqual(valid["sensitivity"], 0.8)
        self.assertEqual(valid["window"], 5)
        self.assertTrue(valid["strict"])
        self.assertEqual(valid["mode"], "accurate")
        self.assertEqual(valid["scales"], (3, 5, 7))

        # Invalid configs rejected cleanly
        invalid_configs: list[dict[str, Any]] = [
            {"unknown_param": 1},
            {"sensitivity": -0.1},
            {"sensitivity": 1.1},
            {"sensitivity": float("nan")},
            {"sensitivity": True},
            {"window": 2.5},
            {"window": 0},
            {"window": False},
            {"strict": "maybe"},
            {"mode": "invalid_choice"},
            {"scales": "2,4"},
            {"scales": (0, 2)},
        ]
        for bad_cfg in invalid_configs:
            with self.subTest(bad_cfg=bad_cfg), self.assertRaises(ValueError):
                det.validate_config(bad_cfg)

        with self.assertRaises(TypeError):
            det.validate_config(["not", "a", "mapping"])  # type: ignore[arg-type]

    def test_all_builtin_detectors_satisfy_contract_and_detect_damage(self) -> None:
        scratch_frames, _ = bench.inject_scratches(
            self.clean_frames,
            bench.ScratchConfig(width=2, opacity=0.9, duration=4, drift_px=3.0, seed=21),
        )
        dust_frames, _ = bench.inject_dust(
            self.clean_frames,
            bench.DustConfig(clusters_per_frame=3, contrast=0.9, seed=22),
        )

        for name in fdd.list_defect_detectors(commercial_only=True):
            detector = fdd.get_defect_detector(name)
            with self.subTest(detector=name):
                res_s = detector.detect(scratch_frames, {"sensitivity": 0.65})
                res_d = detector.detect(dust_frames, {"sensitivity": 0.65})

                for res in (res_s, res_d):
                    self.assertIsInstance(res, fdd.DefectMaskResult)
                    self.assertEqual(len(res), len(self.clean_frames))
                    self.assertIs(res.masks, res.mask)
                    self.assertIs(res.scratch_masks, res.scratch_mask)
                    self.assertIs(res.dust_masks, res.dust_mask)
                    as_d = res.as_dict()
                    self.assertIn("mask", as_d)
                    self.assertIn("scratch", as_d)
                    self.assertIn("dust", as_d)
                    for m in res.mask + res.scratch_mask + res.dust_mask:
                        self.assertEqual(m.shape, (64, 96))
                        self.assertEqual(m.dtype, np.uint8)
                        self.assertTrue(set(np.unique(m).tolist()).issubset({0, 255}))

                if name in ("farneback_3f", "dis_flow", "consensus_5f"):
                    self.assertGreater(sum(np.count_nonzero(m) for m in res_d.dust_mask), 0)
                if name in ("ridge", "consensus_5f", "legacy"):
                    self.assertGreater(sum(np.count_nonzero(m) for m in res_s.scratch_mask), 0)

        # Verify Farneback exact parity with temporal_defect_mask on interior frames
        fb_res = fdd.get_defect_detector("farneback_3f").detect(dust_frames, {"sensitivity": 0.65})
        self.assertEqual(np.count_nonzero(fb_res.mask[0]), 0)
        self.assertEqual(np.count_nonzero(fb_res.mask[-1]), 0)
        for t in range(1, len(dust_frames) - 1):
            expected = fdm.temporal_defect_mask(
                dust_frames[t - 1], dust_frames[t], dust_frames[t + 1], 0.65
            )
            np.testing.assert_array_equal(fb_res.mask[t], expected)
            np.testing.assert_array_equal(
                np.where((fb_res.scratch_mask[t] > 0) | (fb_res.dust_mask[t] > 0), np.uint8(255), np.uint8(0)),
                fb_res.mask[t],
            )

    def test_cleanup_video_cli_flag_and_backward_compatible_identity(self) -> None:
        parser = cleanup_video.build_parser()
        default_args = parser.parse_args(["--source", "input.mp4", "--detect-defects", "--no-dearchive"])
        self.assertEqual(default_args.defect_detector, "farneback_3f")

        temporal_args = parser.parse_args([
            "--source", "input.mp4", "--detect-defects", "--defect-detector", "temporal", "--no-dearchive"
        ])
        dis_args = parser.parse_args([
            "--source", "input.mp4", "--detect-defects", "--defect-detector", "dis_flow", "--no-dearchive"
        ])

        # Default and 'temporal' produce the exact legacy identity key
        id_default = cleanup_video.cleanup_identity(Path("input.mp4"), default_args)
        id_temporal = cleanup_video.cleanup_identity(Path("input.mp4"), temporal_args)
        id_legacy_call = aid.cleanup_identity(
            "input.mp4",
            default_args.cleanup_lora,
            default_args.prompt,
            default_args.negative_prompt,
            default_args.lora_strength,
            default_args.seed,
            detect_defects=True,
            defect_sensitivity=0.65,
            defect_mask_height="720",
            dearchive=False,
        )
        self.assertEqual(id_default, id_legacy_call)
        self.assertEqual(id_temporal, id_legacy_call)
        self.assertEqual(id_default["defect_algorithm"], "temporal_motion_v1")

        # Non-default detector produces distinct identity key
        id_dis = cleanup_video.cleanup_identity(Path("input.mp4"), dis_args)
        self.assertEqual(id_dis["defect_algorithm"], "dis_flow_v1")
        self.assertNotEqual(aid.artifact_key(id_default), aid.artifact_key(id_dis))

        # End-to-end prepare_temporal_defect_masks across farneback_3f, dis_flow, and consensus_5f
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.mkv"
            writer = dust.Writer(source, {"frames": 5, "width": 96, "height": 64, "fps": "12"})
            dust_seq, _ = bench.inject_dust(
                self.clean_frames[:5],
                bench.DustConfig(clusters_per_frame=3, contrast=0.9, seed=33),
            )
            for f in dust_seq:
                writer.write(f)
            writer.close()

            info = {"width": 96, "height": 64, "fps": 12.0, "frames": 5}
            ffmpeg = dust.find_ffmpeg()
            for det_name in ("farneback_3f", "dis_flow", "consensus_5f", "legacy"):
                out_video = root / f"cleaned_{det_name}.mkv"
                work_dir = root / f"work_{det_name}"
                args = parser.parse_args([
                    "--source", str(source),
                    "--output", str(out_video),
                    "--detect-defects",
                    "--defect-detector", det_name,
                    "--defect-mask-height", "source",
                    "--no-dearchive",
                ])
                preview = cleanup_video.prepare_temporal_defect_masks(
                    ffmpeg, source, out_video, work_dir, info, args
                )
                self.assertTrue(preview.is_file())

    def test_gui_dust_integration_inspection_routes_and_cached_mask_preservation(self) -> None:
        # Verify new detectors are exposed in dust.DETECTORS and dust.available_detectors()
        for det_name in ("legacy", "temporal", "ridge", "farneback_3f", "dis_flow", "consensus_5f", "dart"):
            self.assertIn(det_name, dust.DETECTORS)
        avail_names = {d["name"] for d in dust.available_detectors(commercial_only=True)}
        self.assertEqual(avail_names, {"farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy"})

        with tempfile.TemporaryDirectory() as td, patch.object(dust, "ROOT", Path(td)):
            source = Path(td) / "source.mkv"
            writer = dust.Writer(source, {"frames": 5, "width": 96, "height": 64, "fps": "12"})
            dust_seq, _ = bench.inject_dust(
                self.clean_frames[:5],
                bench.DustConfig(clusters_per_frame=3, contrast=0.9, seed=44),
            )
            for f in dust_seq:
                writer.write(f)
            writer.close()

            app = SimpleNamespace(
                stabilized_source_for_downstream=lambda: str(source),
                settings={"dust": {}},
                running_stage_key="",
                project_path="proj",
                process=None,
            )

            # Run mask analysis with dis_flow
            values = {"detector": "dis_flow", "repair": "telea", "sensitivity": 0.65}
            dust.run(source, values, "mask")

            # Verify status() and inspect() expose detector_info and respect cached mask
            st = dust.status(app, values)
            self.assertEqual(st["frames"], 5)
            self.assertEqual(st["detector_info"]["name"], "dis_flow")
            self.assertTrue(st["detector_info"]["commercial_cleared"])

            with patch("ai_remaster_gui.frame_inspector._decode", return_value="frame.png"):
                insp = dust.inspect(app, values, 2)
            self.assertEqual(insp["frame"], 2)
            self.assertEqual(insp["detector_info"]["name"], "dis_flow")

            mat = dust.matrix_state(app)
            self.assertEqual(len(mat["variants"]), 1)
            self.assertTrue(len(mat["available_detectors"]) >= 5)

            # Changing repair backend reuses the frozen dis_flow mask without re-running detection
            changed_repair = {**values, "repair": "propainter", "license_confirmed": "true"}
            with patch.object(dust, "masks_for", side_effect=AssertionError("Must reuse cached mask")):
                folder, _ = dust.saved_mask(source, changed_repair)
                self.assertEqual(folder, dust.folder_for(source, values))
                dust.run(source, changed_repair, "mask")

            # Switching to another detector ('consensus_5f') does not overwrite or invalidate dis_flow cache
            c5_values = {"detector": "consensus_5f", "repair": "telea", "sensitivity": 0.65}
            dust.run(source, c5_values, "mask")
            self.assertNotEqual(dust.folder_for(source, values), dust.folder_for(source, c5_values))
            variants = dust.matrix_state(app)["variants"]
            self.assertEqual(len(variants), 3)
            self.assertEqual(
                dust.digest(dust.folder_for(source, values) / "effective_mask.mkv"),
                dust.digest(dust.folder_for(source, changed_repair) / "effective_mask.mkv"),
            )


if __name__ == "__main__":
    unittest.main()
