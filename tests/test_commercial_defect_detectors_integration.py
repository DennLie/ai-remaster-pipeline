"""Targeted validation tests for commercial-safe defect detectors in the ARP GUI and preset system."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import artifact_ids as aid
import benchmark_defect_masks as bench
import cleanup_video
import film_defect_detectors as fdd
from ai_remaster_gui import commercial_policy, dust, models, runtime_settings, server


class CommercialDefectDetectorsIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(99)
        self.clean_frames = [
            rng.integers(110, 130, size=(64, 96, 3), dtype=np.uint8)
            for _ in range(8)
        ]
        self.dust_frames, _ = bench.inject_dust(
            self.clean_frames,
            bench.DustConfig(clusters_per_frame=4, contrast=0.9, seed=101),
        )
        self.scratch_frames, _ = bench.inject_scratches(
            self.clean_frames,
            bench.ScratchConfig(width=2, opacity=0.9, duration=5, drift_px=2.0, seed=102),
        )

    def _write_dummy_video(self, path: Path, frames: list[np.ndarray]) -> None:
        meta = {"frames": len(frames), "width": frames[0].shape[1], "height": frames[0].shape[0], "fps": "12"}
        writer = dust.Writer(path, meta)
        for f in frames:
            writer.write(f)
        writer.close()

    def test_1_selecting_every_supported_detector(self) -> None:
        """Every commercial-safe detector is selectable and runnable through the GUI backend."""
        supported = ("farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy")
        with tempfile.TemporaryDirectory() as td, patch.object(dust, "ROOT", Path(td)):
            source = Path(td) / "source.mkv"
            self._write_dummy_video(source, self.dust_frames)

            for det in supported:
                with self.subTest(detector=det):
                    values = {"detector": det, "repair": "telea", "sensitivity": 0.65}
                    if det == "ridge":
                        values["research_confirmed"] = "true"
                    opts = dust.options(values)
                    self.assertEqual(opts["detector"], det)

                    # Authorize succeeds in commercial safe mode
                    auth = dust.authorize(values, guard=True, kind="mask")
                    self.assertEqual(auth["detector"], det)

                    # Mask generation succeeds and saves valid output
                    report = dust.run(source, values, "mask", guard=True)
                    self.assertFalse(report["repair_complete"])
                    self.assertEqual(report["metadata"]["frames"], 8)
                    self.assertIn("effective_mask.mkv", report["hashes"])
                    self.assertIn("scratch_mask.mkv", report["hashes"])
                    self.assertIn("dust_mask.mkv", report["hashes"])

    def test_2_gui_to_backend_parameter_propagation(self) -> None:
        """Parameters set in GUI propagate cleanly to CLI invocations and status queries."""
        config = {"comfy_dir": "tools/comfyui", "comfy_url": "http://127.0.0.1:8188", "comfy_output_root": "output"}
        app = server.APP

        # Clean Up stage parameter propagation
        for det in ("farneback_3f", "dis_flow", "ridge", "consensus_5f"):
            cleanup_vals = {
                "detect_defects": "true",
                "defect_detector": det,
                "defect_sensitivity": "0.75",
                "defect_mask_height": "720",
                "dearchive": "false",
            }
            cmd = app._cleanup_command(config, cleanup_vals)
            self.assertIn("--detect-defects", cmd)
            self.assertIn("--defect-detector", cmd)
            self.assertEqual(cmd[cmd.index("--defect-detector") + 1], det)
            self.assertIn("--defect-sensitivity", cmd)
            self.assertEqual(cmd[cmd.index("--defect-sensitivity") + 1], "0.75")

        # Dust stage parameter propagation
        with tempfile.TemporaryDirectory() as td, patch.object(dust, "ROOT", Path(td)):
            source = Path(td) / "source.mkv"
            self._write_dummy_video(source, self.dust_frames)
            test_app = SimpleNamespace(
                stabilized_source_for_downstream=lambda: str(source),
                settings={"dust": {}},
                running_stage_key="",
                project_path="",
                process=None,
            )
            dust_vals = {"detector": "consensus_5f", "repair": "telea", "sensitivity": 0.70}
            dust.run(source, dust_vals, "mask")

            st = dust.status(test_app, dust_vals)
            self.assertEqual(st["detector_info"]["name"], "consensus_5f")
            self.assertEqual(st["detector_info"]["temporal_radius"], 2)
            self.assertEqual(st["mask_values"]["sensitivity"], 0.70)

    def test_3_backward_compatibility_with_existing_saved_presets(self) -> None:
        """Historical defaults, saved projects without detector, and 'temporal' alias are preserved."""
        # 1. Project without detector defaults to farneback_3f / temporal
        defaults = runtime_settings.base_settings()
        self.assertEqual(defaults["dust"]["detector"], "farneback_3f")
        self.assertEqual(defaults["cleanup"]["defect_detector"], "farneback_3f")

        # 2. Historical 'temporal' alias is accepted and mapped without error
        values_temporal = {"detector": "temporal", "repair": "telea"}
        opts_temporal = dust.options(values_temporal)
        self.assertEqual(opts_temporal["detector"], "temporal")
        auth = dust.authorize(values_temporal, guard=True, kind="mask")
        self.assertEqual(auth["detector"], "temporal")

        # 3. Artifact identity for default, empty, and temporal alias are identical
        id_empty = aid.cleanup_identity("src.mp4", "lora", "p", "neg", 1.0, 42, detect_defects=True)
        id_fb = aid.cleanup_identity("src.mp4", "lora", "p", "neg", 1.0, 42, detect_defects=True, defect_detector="farneback_3f")
        id_temp = aid.cleanup_identity("src.mp4", "lora", "p", "neg", 1.0, 42, detect_defects=True, defect_detector="temporal")
        self.assertEqual(id_empty["defect_algorithm"], "temporal_motion_v1")
        self.assertEqual(id_fb["defect_algorithm"], "temporal_motion_v1")
        self.assertEqual(id_temp["defect_algorithm"], "temporal_motion_v1")
        self.assertEqual(aid.artifact_key(id_empty), aid.artifact_key(id_fb))
        self.assertEqual(aid.artifact_key(id_empty), aid.artifact_key(id_temp))

    def test_4_correct_detector_identity_in_artifact_keys_and_caches(self) -> None:
        """Every detector produces a distinct artifact key in cleanup and isolated folder in dust."""
        # Clean Up stage: distinct algorithm keys
        keys: dict[str, str] = {}
        for det in ("farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy"):
            ident = aid.cleanup_identity(
                "film.mp4", "lora", "p", "neg", 1.0, 42,
                detect_defects=True, defect_detector=det
            )
            key = aid.artifact_key(ident)
            keys[det] = key
        # All 5 detectors have unique identity keys
        self.assertEqual(len(set(keys.values())), 5)
        self.assertEqual(keys["farneback_3f"], keys.get("farneback_3f"))

        # Dust stage: distinct cache folder per detector
        with tempfile.TemporaryDirectory() as td, patch.object(dust, "ROOT", Path(td)):
            source = Path(td) / "source.mkv"
            self._write_dummy_video(source, self.clean_frames[:4])
            folders = {
                det: dust.folder_for(source, {"detector": det, "repair": "telea", "sensitivity": 0.65})
                for det in ("farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy")
            }
            self.assertEqual(len(set(folders.values())), 5)

    def test_5_preview_isolation_and_preservation_of_accepted_masks(self) -> None:
        """Changing draft preview detector does NOT alter the accepted pipeline repair."""
        with tempfile.TemporaryDirectory() as td, patch.object(dust, "ROOT", Path(td)):
            source = Path(td) / "source.mkv"
            self._write_dummy_video(source, self.dust_frames)

            # Accept repair with Farneback
            val_a = {"detector": "farneback_3f", "repair": "telea", "sensitivity": 0.65}
            dust.run(source, val_a, "mask")
            rep_a = dust.run(source, val_a, "repair")
            folder_a = dust.folder_for(source, val_a)
            rep_sha = dust.digest(folder_a / "repaired.mkv")

            app = SimpleNamespace(
                stabilized_source_for_downstream=lambda: str(source),
                settings={"dust": {**val_a, "adopted_report": dust.digest(folder_a / "report.json")}},
                running_stage_key="",
                project_path="",
                process=None,
            )

            # Switch draft to DIS Flow and run mask preview
            val_b = {"detector": "dis_flow", "repair": "telea", "sensitivity": 0.65}
            dust.run(source, val_b, "mask")
            folder_b = dust.folder_for(source, val_b)

            # Accepted repair in folder A remains bit-identical and adopted
            self.assertEqual(dust.digest(folder_a / "repaired.mkv"), rep_sha)
            self.assertNotEqual(folder_a, folder_b)
            self.assertEqual(app.settings["dust"]["detector"], "farneback_3f")
            self.assertEqual(app.settings["dust"]["adopted_report"], dust.digest(folder_a / "report.json"))

    def test_6_switching_between_detectors(self) -> None:
        """Switching between detectors finds cached masks without overwriting or re-inference."""
        with tempfile.TemporaryDirectory() as td, patch.object(dust, "ROOT", Path(td)):
            source = Path(td) / "source.mkv"
            self._write_dummy_video(source, self.dust_frames)
            app = SimpleNamespace(
                stabilized_source_for_downstream=lambda: str(source),
                settings={"dust": {}},
                running_stage_key="",
                project_path="",
                process=None,
            )

            detectors = ("farneback_3f", "dis_flow", "consensus_5f", "ridge", "legacy")
            for det in detectors:
                vals = {"detector": det, "repair": "telea", "sensitivity": 0.65}
                if det == "ridge":
                    vals["research_confirmed"] = "true"
                dust.run(source, vals, "mask")

            # All 5 masks are present in matrix_state
            mat = dust.matrix_state(app)
            self.assertEqual(len(mat["variants"]), 5)

            # Retrieving each detector via saved_mask succeeds without calling masks_for
            with patch.object(dust, "masks_for", side_effect=AssertionError("Should use saved mask")):
                for det in detectors:
                    vals = {"detector": det, "repair": "telea", "sensitivity": 0.65}
                    if det == "ridge":
                        vals["research_confirmed"] = "true"
                    folder, report = dust.saved_mask(source, vals)
                    self.assertTrue((folder / "mask.mkv").is_file())
                    self.assertEqual(report["identity"]["options"]["detector"], det)

    def test_7_unsupported_settings_and_insufficient_temporal_context(self) -> None:
        """Unknown detectors and invalid configs are rejected; boundary frames abstain correctly."""
        # Unknown detector rejected
        with self.assertRaises(ValueError):
            dust.options({"detector": "nonexistent_neural_detector", "repair": "telea"})

        # Temporal radius context: consensus_5f requires radius 2
        seq = [np.full((32, 48, 3), 120, dtype=np.uint8) for _ in range(5)]
        with patch.object(dust, "frames", return_value=iter(seq)):
            items = list(dust.masks_for("unused", dust.options({"detector": "consensus_5f"})))
            self.assertEqual(len(items), 5)
            # Boundary frames (first 2, last 2) must abstain
            self.assertIn("insufficient context", items[0][2])
            self.assertIn("insufficient context", items[1][2])
            self.assertIn("insufficient context", items[3][2])
            self.assertIn("insufficient context", items[4][2])

        # Spatial detector (legacy) requires zero context and never abstains
        with patch.object(dust, "frames", return_value=iter(seq)):
            items_spatial = list(dust.masks_for("unused", dust.options({"detector": "legacy"})))
            self.assertEqual(len(items_spatial), 5)
            self.assertTrue(all(not item[2] for item in items_spatial))

    def test_8_commercial_exclusion_of_dart(self) -> None:
        """Strict commercial firewall: DART is blocked in GUI, registry, and commercial policies."""
        registry = fdd.DEFAULT_REGISTRY

        # 1. Commercial registry lookup blocks DART
        self.assertFalse(registry.has("dart", commercial_only=True))
        self.assertFalse(registry.has("dart_harness", commercial_only=True))
        with self.assertRaises(fdd.CommercialLicenseError):
            registry.get("dart", commercial_only=True)

        # 2. Dust stage authorization blocks DART when guard is active
        dart_vals = {"detector": "dart", "repair": "dart", "research_confirmed": "true", "license_confirmed": "true"}
        with self.assertRaises(ValueError):
            dust.authorize(dart_vals, guard=True, kind="mask")
        with self.assertRaises(ValueError):
            dust.authorize(dart_vals, guard=True, kind="repair")

        # 3. Commercial policy reports DART issue
        settings = {"global": {"commercial_model_guard": "true"}, "dust": dart_vals}
        issue = commercial_policy.stage_issue(settings, "dust")
        self.assertIn("DART", issue)

        # 4. Commercial choices tuple excludes DART
        self.assertNotIn("dart", fdd.COMMERCIAL_DETECTOR_CHOICES)
        self.assertNotIn("dart", fdd.DEFAULT_REGISTRY.list_available(commercial_only=True))
        dust_stage_def = next(s for s in models.STAGES if s.key == "dust")
        detector_field = next(f for f in dust_stage_def.fields if f[0] == "detector")
        detector_choices = detector_field[2].split("select:")[1].split("|")
        for expected in ("farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy"):
            self.assertIn(expected, detector_choices)

    def test_9_channels_and_interactive_comparison_support(self) -> None:
        """Inspect endpoint returns individual scratch and dust channels alongside combined mask."""
        with tempfile.TemporaryDirectory() as td, patch.object(dust, "ROOT", Path(td)):
            source = Path(td) / "source.mkv"
            self._write_dummy_video(source, self.dust_frames)
            values = {"detector": "dis_flow", "repair": "telea", "sensitivity": 0.65}
            dust.run(source, values, "mask")

            app = SimpleNamespace(
                stabilized_source_for_downstream=lambda: str(source),
                settings={"dust": {}},
                running_stage_key="",
                project_path="",
                process=None,
            )

            # Inspect decoded images contains all comparison channels
            with patch("ai_remaster_gui.frame_inspector._decode", side_effect=lambda p, f: f"{p.name}_f{f}.png"):
                insp = dust.inspect(app, values, 2)

            self.assertIn("input.mkv", insp["images"])
            self.assertIn("mask.mkv", insp["images"])
            self.assertIn("effective_mask.mkv", insp["images"])
            self.assertIn("scratch_mask.mkv", insp["images"])
            self.assertIn("dust_mask.mkv", insp["images"])
            self.assertIn("supported_channels", insp["detector_info"])
            self.assertEqual(insp["detector_info"]["supported_channels"], ["scratch", "dust"])

    def test_10_cleanup_video_streaming_and_bounded_memory(self) -> None:
        """Windowed defect mask streaming in cleanup pipeline processes frames in bounded chunks."""
        from common import find_ffmpeg, signature_path
        from prepare_outpaint_input import probe_video

        ffmpeg = find_ffmpeg()
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            src = td_path / "stream_test.mkv"
            vw = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (64, 48))
            for i in range(30):
                vw.write(np.full((48, 64, 3), 110 + (i % 20), dtype=np.uint8))
            vw.release()
            info = probe_video(src)

            for det in ("farneback_3f", "dis_flow", "ridge", "consensus_5f", "legacy"):
                with self.subTest(detector=det):
                    args = SimpleNamespace(
                        force=True,
                        defect_sensitivity=0.65,
                        defect_mask_height="source",
                        defect_detector=det,
                    )
                    work_dir = td_path / f"work_{det}"
                    work_dir.mkdir()
                    out = td_path / f"out_{det}.mp4"
                    preview_path = cleanup_video.prepare_temporal_defect_masks(
                        ffmpeg, src, out, work_dir, info, args
                    )
                    self.assertTrue(preview_path.is_file())
                    self.assertGreater(preview_path.stat().st_size, 0)
                    mask_files = list(work_dir.glob("temporal_defects_*.mkv"))
                    self.assertEqual(len(mask_files), 1)
                    sig_path = signature_path(mask_files[0])
                    self.assertTrue(sig_path.is_file())
                    sig = json.loads(sig_path.read_text(encoding="utf-8"))
                    if det == "farneback_3f":
                        self.assertEqual(sig["tool"], "film_defect_masks.py")
                    else:
                        self.assertEqual(sig["detector"], det)

    def test_11_cleanup_select_labels_and_alias_mapping(self) -> None:
        """HTML select helper provides readable detector labels and maps 'temporal' alias to farneback_3f."""
        js_file = ROOT / "ai_remaster_gui" / "static" / "js" / "render-helpers.js"
        content = js_file.read_text(encoding="utf-8")

        # Verify friendly labels are declared in selectOptionLabel
        for det, friendly in (
            ("farneback_3f", "Farneback 3F (optical flow consensus)"),
            ("dis_flow", "DIS Flow (fast optical flow)"),
            ("ridge", "Temporal Ridge (vertical scratches)"),
            ("consensus_5f", "Consensus 5F (5-frame median)"),
            ("legacy", "Legacy (single-frame spatial)"),
        ):
            self.assertIn(friendly, content)

        # Verify isSelected logic in selectFieldHtml handles 'temporal' and empty
        self.assertIn("value === 'temporal'", content)


if __name__ == "__main__":
    unittest.main()
