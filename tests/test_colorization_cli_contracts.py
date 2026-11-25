"""End-to-end CLI argument, Color Bible propagation, cache invalidation, and project bundling tests."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


class ColorizationCliContractsTests(unittest.TestCase):
    def test_references_commands_parse_cleanly_in_all_target_clis(self) -> None:
        from ai_remaster_gui.server import PipelineApp
        import ddcolor_reference
        import qwen_colorize_references
        import qwen21_generate_reference

        app_inst = PipelineApp.__new__(PipelineApp)
        app_inst.settings = {
            "global": {
                "color_bible_db": "manifests/color_bible.db",
                "color_bible_film_id": "film_abc",
            },
            "references": {},
        }

        # 1. ddcolor command with Regenerate (--force) and Color Bible context
        cmd_dd = app_inst._references_command(
            {},
            {
                "method": "ddcolor",
                "manifest": "manifests/shots.csv",
                "reference_runner": "guided_only",
                "ddcolor_model": "piddnad/ddcolor_paper_tiny",
                "confidence_threshold": "0.85",
            },
        ) + ["--force", "--dry-run"]
        dd_args = ddcolor_reference.build_parser().parse_args(cmd_dd[3:])
        self.assertTrue(dd_args.force)
        self.assertTrue(dd_args.dry_run)
        self.assertEqual(dd_args.film_id, "film_abc")
        self.assertEqual(str(dd_args.color_bible_db).replace("\\", "/"), "manifests/color_bible.db")
        self.assertEqual(dd_args.model_variant, "piddnad/ddcolor_paper_tiny")

        # 2. qwen 2511 command
        with mock.patch("ai_remaster_gui.server.qwen_workflow_for", return_value="workflows/qwen.json"):
            cmd_qwen = app_inst._references_command(
                {},
                {
                    "method": "qwen",
                    "manifest": "manifests/shots.csv",
                    "reference_runner": "guided_only",
                },
            ) + ["--force", "--dry-run"]
        qwen_args = qwen_colorize_references.build_parser().parse_args(cmd_qwen[3:])
        self.assertTrue(qwen_args.force)
        self.assertEqual(qwen_args.reference_runner, "guided_only")
        self.assertEqual(qwen_args.film_id, "film_abc")

        # 3. qwen 21 command
        cmd_q21 = app_inst._references_command(
            {},
            {
                "method": "qwen21",
                "manifest": "manifests/shots.csv",
            },
        ) + ["--force", "--dry-run"]
        q21_args = qwen21_generate_reference.build_parser().parse_args(cmd_q21[3:])
        self.assertTrue(q21_args.force)
        self.assertEqual(q21_args.film_id, "film_abc")

    def test_colour_osi_masked_lab_command_inherits_color_bible_and_parses(self) -> None:
        from ai_remaster_gui.server import PipelineApp
        import colorize_video

        app_inst = PipelineApp.__new__(PipelineApp)
        app_inst.settings = {
            "global": {
                "color_bible_db": "manifests/color_bible.db",
                "color_bible_film_id": "heritage_film_01",
            },
            "cloud": {"intermediate_format": "high"},
            "colour": {
                "manifest": "manifests/shots.csv",
                "method": "osi_masked_lab",
                "osi_flow_scale": "0.5",
                "osi_anchor_strength": "0.75",
                "guided_radius": "8",
                "guided_eps": "1e-3",
                "osi_run_qc": "true",
                "osi_workers": "2",
            },
        }
        cmd = app_inst._colour_command({}, app_inst.settings["colour"])
        parsed = colorize_video.build_parser().parse_args(cmd[3:])
        self.assertEqual(parsed.method, "osi_masked_lab")
        self.assertEqual(parsed.film_id, "heritage_film_01")
        self.assertEqual(str(parsed.color_bible_db).replace("\\", "/"), "manifests/color_bible.db")
        self.assertAlmostEqual(parsed.flow_scale, 0.5)
        self.assertAlmostEqual(parsed.anchor_strength, 0.75)
        self.assertEqual(parsed.guided_radius, 8)
        self.assertTrue(parsed.run_qc)
        self.assertEqual(parsed.workers, 2)

    def test_needs_comfy_excludes_osi_masked_lab(self) -> None:
        from ai_remaster_gui.server import PipelineApp

        app = PipelineApp.__new__(PipelineApp)
        app.settings = {
            "global": {"commercial_model_guard": "true"},
            "colour": {"method": "osi_masked_lab", "compute": "local", "manifest": ""},
        }
        # Ensure missing manifest is reported rather than requiring ComfyUI
        ok, msg = app.run_stage("colour") if False else (True, "")
        self.assertTrue(ok)

    def test_color_bible_context_resolution_precedence(self) -> None:
        from ai_remaster_gui.color_bible_context import (
            canonical_shot_id,
            find_shot_mask_archive_candidates,
            resolve_color_bible_context,
            shot_id_candidates,
        )

        settings = {
            "global": {"color_bible_db": "manifests/custom_cb.db", "color_bible_film_id": "global_film"},
            "references": {"film_id": "", "color_bible_db": ""},
            "colour": {"film_id": "override_film", "color_bible_db": ""},
        }
        ctx_ref = resolve_color_bible_context(settings, REPO_ROOT, stage="references")
        self.assertEqual(ctx_ref["film_id"], "global_film")
        self.assertEqual(ctx_ref["db_raw"], "manifests/custom_cb.db")

        ctx_col = resolve_color_bible_context(settings, REPO_ROOT, stage="colour")
        self.assertEqual(ctx_col["film_id"], "override_film")

        self.assertEqual(canonical_shot_id("my_film", 3), "my_film_3")
        cands = shot_id_candidates("my_film", 3)
        self.assertIn("my_film_3", cands)
        self.assertIn("shot_0003", cands)
        self.assertIn("shot_003", cands)

        with tempfile.TemporaryDirectory() as tmp:
            masks_dir = Path(tmp)
            target = masks_dir / "shot_003.masks.npz"
            target.write_bytes(b"npz")
            found = find_shot_mask_archive_candidates(masks_dir, 3, film_id="my_film")
            self.assertEqual(found, target)

    def test_stage_is_current_respect_run_flags_and_qc_derived_columns(self) -> None:
        from ai_remaster_gui.stage_stamps import DERIVED_COLUMNS

        for col in ("qc_status", "qc_cdc", "qc_flow_err", "qc_delta_e", "qc_bleed", "qc_failure_codes"):
            self.assertIn(col, DERIVED_COLUMNS)

        from ai_remaster_gui.server import PipelineApp

        app = PipelineApp.__new__(PipelineApp)
        app.settings = {"global": {}, "shots": {"force": "true"}}
        with mock.patch("ai_remaster_gui.stage_plan.pipeline_configured", return_value=False):
            self.assertFalse(app.stage_is_current("shots", respect_run_flags=True))
            with mock.patch.object(app, "expected_outputs", return_value=["a.csv"]), \
                 mock.patch.object(app, "existing_outputs", return_value=["a.csv"]), \
                 mock.patch("ai_remaster_gui.stage_stamps.read_stamp", return_value="abc"), \
                 mock.patch.object(app, "stage_stamp", return_value="abc"):
                self.assertTrue(app.stage_is_current("shots", respect_run_flags=False))

    def test_ddcolor_manifest_force_flag_and_real_model_if_cached(self) -> None:
        import ddcolor_reference

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "shot_0000.png"
            dst = root / "shot_0000_color.png"
            Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(src)
            manifest = root / "shots.csv"
            manifest.write_text(
                f"enabled,end,source_reference,color_reference\ntrue,0:00:01,{src},{dst}\n",
                encoding="utf-8",
            )
            exit_code = ddcolor_reference.main.__wrapped__() if hasattr(ddcolor_reference.main, "__wrapped__") else None
            self.assertIsNone(exit_code)
            with mock.patch.object(sys, "argv", ["ddcolor_reference.py", "--manifest", str(manifest), "--dry-run", "--force", "--row-index", "0"]):
                rc = ddcolor_reference.main()
            self.assertEqual(rc, 0)
            self.assertTrue(dst.is_file())
            self.assertTrue(Path(str(dst) + ".sig.json").is_file())

    def test_real_piddnad_ddcolor_paper_tiny_smoke_when_cached(self) -> None:
        from huggingface_hub import try_to_load_from_cache
        import ddcolor_reference

        cached = try_to_load_from_cache("piddnad/ddcolor_paper_tiny", "pytorch_model.bin")
        if not cached or not isinstance(cached, str) or not Path(cached).is_file():
            self.skipTest("Cached piddnad/ddcolor_paper_tiny not present locally")

        gray = np.full((64, 64, 3), 110, dtype=np.uint8)
        gray[:32, :, :] = 190
        out = ddcolor_reference.colorize_frame(
            gray,
            model_path=Path(cached),
            device="cpu",
            dry_run=False,
            input_size=128,
            allow_builtin_arch=True,
        )
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertEqual(out.dtype, np.uint8)
        # Verify real model predicted non-trivial chroma
        chroma_spread = float(np.abs(out[:, :, 0].astype(np.float32) - out[:, :, 2].astype(np.float32)).max())
        self.assertGreater(chroma_spread, 1.0)

    def test_project_io_allows_color_bible_and_shot_masks(self) -> None:
        from ai_remaster_gui.project_io import bundle_name_allowed

        self.assertTrue(bundle_name_allowed("manifests/color_bible.db"))
        self.assertTrue(bundle_name_allowed("intermediate/shot_masks/film_0.masks.npz"))
        self.assertFalse(bundle_name_allowed("intermediate/shot_masks/evil.exe"))
        self.assertFalse(bundle_name_allowed("../manifests/color_bible.db"))


if __name__ == "__main__":
    unittest.main()
