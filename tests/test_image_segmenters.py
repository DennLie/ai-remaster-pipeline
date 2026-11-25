"""CPU-only unit & integration tests for Keyframe / Still Image Segmenters."""

from __future__ import annotations

import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from ai_remaster_gui import http_handler, state
from ai_remaster_gui import server  # noqa: F401
from ai_remaster_gui.color_bible_service import ColorBibleService
from ai_remaster_gui.commercial_policy import (
    CommercialPolicyError,
    evaluate_component,
    stage_issue,
)
from ai_remaster_gui.compliance import load_compliance_report, validate_compliance_report
from ai_remaster_gui.image_segmenters import (
    DEFAULT_IMAGE_SEGMENTER_REGISTRY,
    GroundingDINOImageSegmenter,
    ImageSegmenterBackend,
    Mask2FormerImageSegmenter,
    OneFormerImageSegmenter,
    load_seed_mask_for_tracker,
    run_keyframe_segmentation,
)
from ai_remaster_gui.rle_masks import ShotMaskArchive
from ai_remaster_gui.runtime_settings import base_settings
from ai_remaster_gui.video_segmenters import run_shot_segmentation
from scripts.osi_masked_lab import apply_strict_l_lock
from scripts.segment_keyframe_mask import main as img_cli_main


class _ImageHttpHandler(http_handler.Handler):
    def __init__(self, path: str, body: dict | None = None, headers: dict[str, str] | None = None) -> None:
        self.path = path
        self.headers = headers or {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        self._body = body or {}
        self.responses: dict[str, object] = {}

    def read_json(self) -> dict:
        return self._body

    def send_json(self, payload) -> None:
        self.responses["json"] = payload

    def send_error(self, code, message=None, explain=None) -> None:
        self.responses["error"] = code


class TestImageSegmenterRegistryAndSidecars(unittest.TestCase):
    """(a) All 10 image segmenters register cleanly and output binary PNG masks + JSON sidecars."""

    def test_all_10_image_segmenters_register_and_write_mask_and_sidecar(self) -> None:
        specs = DEFAULT_IMAGE_SEGMENTER_REGISTRY.list_specs()
        names = [s.name for s in specs]
        expected = [
            "evf_sam2",
            "florence2",
            "birefnet",
            "mask2former",
            "oneformer",
            "grounding_dino",
            "rf_detr_seg",
            "sam21_image",
            "sam3_image",
            "sam31_image",
        ]
        self.assertEqual(names, expected)

        h, w = 48, 64
        img_rgb = np.full((h, w, 3), 120, dtype=np.uint8)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        gt_mask[10:36, 16:48] = 255

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_path = tmp_path / "keyframe.png"
            Image.fromarray(img_rgb).save(img_path)

            for seg_name in expected:
                with self.subTest(image_segmenter=seg_name):
                    kwargs: dict[str, object] = {}
                    seg_call_kwargs: dict[str, object] = {}

                    if seg_name == "evf_sam2":
                        runner = mock.MagicMock()
                        runner.predict.return_value = {"mask": gt_mask > 0, "score": 0.94}
                        kwargs["runner"] = runner
                        seg_call_kwargs["text_prompt"] = "protagonist coat"
                    elif seg_name == "florence2":
                        runner = mock.MagicMock()
                        runner.predict.return_value = {
                            "<REFERRING_EXPRESSION_SEGMENTATION>": {
                                "polygons": [[[16, 10, 47, 10, 47, 35, 16, 35]]],
                                "labels": ["coat"],
                            }
                        }
                        kwargs["runner"] = runner
                        seg_call_kwargs["text_prompt"] = "coat"
                    elif seg_name == "birefnet":
                        runner = mock.MagicMock()
                        runner.predict.return_value = (gt_mask.astype(np.float32) / 255.0)
                        kwargs["runner"] = runner
                    elif seg_name in {"mask2former", "oneformer"}:
                        seg_map = np.zeros((h, w), dtype=np.int32)
                        seg_map[10:36, 16:48] = 1
                        seg_map[:10, :] = 2
                        runner = mock.MagicMock()
                        runner.predict.return_value = {
                            "segmentation": seg_map,
                            "segments_info": [
                                {"id": 1, "label": "person", "score": 0.98},
                                {"id": 2, "label": "sky", "score": 0.95},
                            ],
                        }
                        kwargs["runner"] = runner
                        seg_call_kwargs["target_classes"] = ["person"]
                    elif seg_name == "grounding_dino":
                        runner = mock.MagicMock()
                        runner.predict_boxes.return_value = [("coat", [16, 10, 48, 36], 0.91)]
                        refiner = mock.MagicMock()
                        refiner.predict.return_value = (np.stack([gt_mask > 0], axis=0), np.array([0.96]), None)
                        kwargs["runner"] = runner
                        kwargs["refiner"] = refiner
                        seg_call_kwargs["text_prompt"] = "coat"
                    elif seg_name == "rf_detr_seg":
                        runner = mock.MagicMock()
                        runner.predict.return_value = {
                            "masks": np.stack([gt_mask > 0], axis=0),
                            "boxes": [[16, 10, 48, 36]],
                            "scores": [0.89],
                            "labels": ["person"],
                        }
                        kwargs["runner"] = runner
                    elif seg_name in {"sam21_image", "sam3_image", "sam31_image"}:
                        pred = mock.MagicMock()
                        pred.predict.return_value = (np.stack([gt_mask > 0], axis=0), np.array([0.97]), None)
                        kwargs["predictor"] = pred
                        if seg_name == "sam21_image":
                            seg_call_kwargs["points"] = [[30, 20]]
                        else:
                            seg_call_kwargs["text_prompt"] = "protagonist"

                    allow_nc = seg_name in {"sam3_image", "sam31_image"}
                    backend = DEFAULT_IMAGE_SEGMENTER_REGISTRY.create(
                        seg_name,
                        allow_non_commercial_benchmark=allow_nc,
                        **kwargs,
                    )
                    self.assertIsInstance(backend, ImageSegmenterBackend)
                    backend.close()

                    out_mask = tmp_path / f"mask_{seg_name}.png"
                    res = run_keyframe_segmentation(
                        image=img_path,
                        output_mask_path=out_mask,
                        segmenter=seg_name,
                        allow_non_commercial_benchmark=allow_nc,
                        **seg_call_kwargs,
                        **kwargs,
                    )
                    self.assertTrue(out_mask.is_file())
                    sidecar = out_mask.with_suffix(".json")
                    self.assertTrue(sidecar.is_file())
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    self.assertEqual(meta["segmenter"], seg_name)
                    self.assertEqual(len(meta["mask_sha256"]), 64)
                    self.assertEqual((meta["width"], meta["height"]), (w, h))
                    self.assertGreater(res["foreground_pixels"], 0)
                    saved_arr = load_seed_mask_for_tracker(out_mask)
                    self.assertEqual(set(np.unique(saved_arr).tolist()), {0, 255})

    def test_panoptic_target_classes_filtering_and_vram_cleanup(self) -> None:
        h, w = 30, 40
        img = np.zeros((h, w, 3), dtype=np.uint8)
        seg_map = np.zeros((h, w), dtype=np.int32)
        seg_map[:10, :] = 10  # sky
        seg_map[10:25, 5:20] = 20  # person
        runner = mock.MagicMock()
        runner.predict.return_value = {
            "segmentation": seg_map,
            "segments_info": [
                {"id": 10, "label": "sky", "score": 0.9},
                {"id": 20, "label": "person", "score": 0.95},
            ],
        }
        for cls in (Mask2FormerImageSegmenter, OneFormerImageSegmenter):
            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.empty_cache") as empty_cache:
                seg = cls(runner=runner)
                res_sky = seg.segment_image(img, target_classes="sky")
                self.assertEqual(res_sky.labels, ["sky"])
                self.assertEqual(int(res_sky.mask[2, 2]), 255)
                self.assertEqual(int(res_sky.mask[15, 10]), 0)
                seg.close()
                empty_cache.assert_called()

    def test_grounding_dino_box_fallback_without_refiner(self) -> None:
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        runner = mock.MagicMock()
        runner.predict_boxes.return_value = [("hat", [4, 6, 18, 20], 0.88)]
        seg = GroundingDINOImageSegmenter(runner=runner)
        res = seg.segment_image(img, text_prompt="hat")
        self.assertEqual(res.labels, ["hat"])
        self.assertEqual(res.boxes, [(4.0, 6.0, 18.0, 20.0)])
        self.assertEqual(int(res.mask[10, 10]), 255)
        self.assertEqual(int(res.mask[0, 0]), 0)


class TestImageSegmenterPolicyAndCompliance(unittest.TestCase):
    """(b) & (c) Commercial Guard blocking and non-commercial benchmark override compliance logging."""

    def test_commercial_mode_allows_8_osi_and_blocks_sam3_image(self) -> None:
        settings = base_settings()
        settings["global"]["commercial_model_guard"] = "true"
        settings["colour"]["method"] = "osi_masked_lab"

        osi_image = ["evf_sam2", "florence2", "birefnet", "mask2former", "oneformer", "grounding_dino", "rf_detr_seg", "sam21_image"]
        for name in osi_image:
            dec = DEFAULT_IMAGE_SEGMENTER_REGISTRY.check_policy(name, settings=settings)
            self.assertTrue(dec.allowed)
            settings["colour"]["image_segmenter"] = name
            self.assertEqual(stage_issue(settings, "colour"), "")

        for nc_name in ("sam3_image", "sam31_image"):
            dec = evaluate_component(nc_name, settings=settings, component_type="tool")
            self.assertFalse(dec.allowed)
            self.assertEqual(dec.status, "hard_blocked")
            self.assertIn("Meta SAM License", dec.reason)
            with self.assertRaises(CommercialPolicyError):
                DEFAULT_IMAGE_SEGMENTER_REGISTRY.check_policy(nc_name, settings=settings)
            settings["colour"]["image_segmenter"] = nc_name
            self.assertIn("Meta SAM License", stage_issue(settings, "colour"))

    def test_benchmark_override_unlocks_sam3_image_and_logs_non_compliant(self) -> None:
        h, w = 32, 32
        img = np.zeros((h, w, 3), dtype=np.uint8)
        gt = np.zeros((h, w), dtype=bool)
        gt[6:22, 6:22] = True
        pred = mock.MagicMock()
        pred.predict.return_value = (np.stack([gt], axis=0), np.array([0.95]), None)

        override_settings = base_settings()
        override_settings["global"]["commercial_model_guard"] = "false"
        override_settings["global"]["non_commercial_benchmark_override"] = "true"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_dir = tmp_path / "output"
            mask_out = tmp_path / "sam31_mask.png"

            res = run_keyframe_segmentation(
                image=img,
                output_mask_path=mask_out,
                segmenter="sam31_image",
                text_prompt="actor",
                settings=override_settings,
                allow_non_commercial_benchmark=True,
                compliance_output_dir=out_dir,
                predictor=pred,
            )
            self.assertTrue(Path(res["mask_path"]).is_file())
            valid, errs = validate_compliance_report(out_dir)
            self.assertTrue(valid, f"Invalid compliance report: {errs}")
            report = load_compliance_report(out_dir)
            self.assertFalse(report["overall_commercial_compliant"])
            self.assertFalse(report["runs"][0]["is_commercial_compliant"])
            self.assertEqual(report["runs"][0]["guard_mode"], "benchmark_override")
            self.assertTrue(report["runs"][0]["metadata"]["non_commercial_benchmark_run"])

    def test_cli_allow_non_commercial_benchmark_and_strict_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_path = tmp_path / "still.png"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(img_path)
            out_mask = tmp_path / "still_sam3.png"
            comp_dir = tmp_path / "comp"

            rc_blocked = img_cli_main([
                "--segmenter", "sam3_image",
                "--image", str(img_path),
                "--output", str(out_mask),
                "--text-prompt", "coat",
                "--compliance-output-dir", str(comp_dir),
            ])
            self.assertEqual(rc_blocked, 2)

            gt = np.zeros((32, 32), dtype=bool)
            gt[4:24, 4:24] = True
            fake_sam3 = mock.MagicMock()
            fake_pred = mock.MagicMock()
            fake_pred.predict.return_value = (np.stack([gt], axis=0), np.array([0.9]), None)
            fake_sam3.build_sam3_image_predictor.return_value = fake_pred

            with mock.patch.dict(sys.modules, {"sam3": fake_sam3}):
                rc_ok = img_cli_main([
                    "--segmenter", "sam3_image",
                    "--image", str(img_path),
                    "--output", str(out_mask),
                    "--text-prompt", "coat",
                    "--compliance-output-dir", str(comp_dir),
                    "--allow-non-commercial-benchmark",
                ])
            self.assertEqual(rc_ok, 0)
            self.assertTrue(out_mask.is_file())
            self.assertFalse(load_compliance_report(comp_dir)["overall_commercial_compliant"])


class TestKeyframeToColorBibleAndVideoTrackerE2E(unittest.TestCase):
    """(d) Keyframe mask plugs seamlessly into BOTH ColorBibleService and VideoSegmenterRegistry."""

    def test_keyframe_mask_feeds_color_bible_and_video_tracker(self) -> None:
        h, w, t_count = 48, 48, 3
        gray = np.full((h, w), 128, dtype=np.uint8)
        # Keyframe still: blue background (a=-10, b=-30), vibrant warm coat inside center [12:36, 12:36] (a=45, b=25)
        chroma = np.full((h, w, 2), [-10.0, -30.0], dtype=np.float32)
        chroma[12:36, 12:36] = [45.0, 25.0]
        ref_rgb = apply_strict_l_lock(gray, chroma)

        soft_matte = np.zeros((h, w), dtype=np.float32)
        soft_matte[12:36, 12:36] = 0.92
        biref_runner = mock.MagicMock()
        biref_runner.predict.return_value = soft_matte

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ref_path = tmp_path / "ref_still.png"
            Image.fromarray(ref_rgb).save(ref_path)
            mask_path = tmp_path / "keyframe_masks" / "ref_still_birefnet.png"

            # Step 1: Segment keyframe still via BiRefNet
            seg_res = run_keyframe_segmentation(
                image=ref_path,
                output_mask_path=mask_path,
                segmenter="birefnet",
                runner=biref_runner,
            )
            self.assertTrue(Path(seg_res["mask_path"]).is_file())

            # Step 2a: Register reference in Color Bible with generated mask_uri -> LAB stats reflect warm coat only!
            db_path = tmp_path / "color_bible.db"
            with ColorBibleService(db_path=db_path) as cb:
                cb.create_film("film_kf", title="Keyframe E2E")
                char = cb.create_character("film_kf", "Elena")
                ref_rec = cb.register_reference_image(
                    film_id="film_kf",
                    entity_type="character",
                    entity_id=char.character_id,
                    keyframe_path=ref_path,
                    mask_uri=mask_path,
                    provenance="manual",
                    auto_approve=True,
                )
                self.assertEqual(ref_rec.mask_uri, str(mask_path))
                lab_stats = json.loads(ref_rec.lab_stats_json)
                median_a = lab_stats["a"][2]
                median_b = lab_stats["b"][2]
                self.assertGreater(median_a, 35.0)
                self.assertGreater(median_b, 15.0)

            # Step 2b: Pass saved mask PNG into VideoSegmenterRegistry as seed mask -> .masks.npz
            seed_mask = load_seed_mask_for_tracker(mask_path)
            cutie_runner = mock.MagicMock()
            cutie_runner.step.side_effect = [
                np.stack([1.0 - (seed_mask > 0).astype(np.float32), (seed_mask > 0).astype(np.float32)], axis=0)
                for _ in range(t_count)
            ]
            out_npz = tmp_path / "shot_0000.masks.npz"
            shot_frames = [ref_rgb.copy() for _ in range(t_count)]
            run_shot_segmentation(
                shot_id="shot_0000",
                frames=shot_frames,
                prompts={1: {"frame_idx": 0, "mask": seed_mask, "name": "Elena", "entity_type": "character"}},
                output_npz_path=out_npz,
                segmenter="cutie",
                runner=cutie_runner,
            )
            archive = ShotMaskArchive.load_npz(out_npz)
            self.assertEqual(archive.frame_count, t_count)
            for t in range(t_count):
                np.testing.assert_array_equal(archive.get_mask(t, 1), seed_mask)


class TestMissingImageSegmenterDepsRaiseActionableErrors(unittest.TestCase):
    """(e) Missing packages/weights raise explicit RuntimeError without fake masks."""

    def test_missing_packages_raise_runtime_error_for_all_10(self) -> None:
        real_import = builtins.__import__
        blocked = ("evf_sam", "transformers", "birefnet", "groundingdino", "rfdetr", "sam2", "sam3")

        def _fail_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == p or name.startswith(p + ".") for p in blocked):
                raise ImportError(f"Simulated missing module '{name}'")
            return real_import(name, globals, locals, fromlist, level)

        saved = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if any(k == p or k.startswith(p + ".") for p in blocked)}
        try:
            with mock.patch("builtins.__import__", side_effect=_fail_import):
                for spec in DEFAULT_IMAGE_SEGMENTER_REGISTRY.list_specs():
                    with self.subTest(image_segmenter=spec.name):
                        with self.assertRaises(RuntimeError):
                            DEFAULT_IMAGE_SEGMENTER_REGISTRY.create(
                                spec.name,
                                allow_non_commercial_benchmark=(spec.commercial_tier == "hard_blocked"),
                            )
        finally:
            sys.modules.update(saved)


class TestImageSegmenterHttpRoutes(unittest.TestCase):
    """(f) HTTP GET /api/segmenters and POST /api/segment-keyframe policy, origin, and path-traversal checks."""

    def setUp(self) -> None:
        self._real_app = state.APP
        self.app = mock.MagicMock()
        self.app.log = []
        self.app.settings = base_settings()
        self.app.state.return_value = {}
        state.APP = self.app

    def tearDown(self) -> None:
        state.APP = self._real_app

    def test_get_segmenters_lists_both_registries_and_enforces_host(self) -> None:
        h = _ImageHttpHandler("/api/segmenters")
        h.do_GET()
        payload = h.responses["json"]
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["image_segmenters"]), 10)
        self.assertEqual(len(payload["video_trackers"]), 9)
        img_map = {item["name"]: item for item in payload["image_segmenters"]}
        self.assertTrue(img_map["birefnet"]["allowed"])
        self.assertFalse(img_map["sam3_image"]["allowed"])
        self.assertFalse(img_map["sam31_image"]["allowed"])

        remote = _ImageHttpHandler("/api/segmenters", headers={"Host": "evil.example"})
        remote.do_GET()
        self.assertEqual(remote.responses.get("error"), 403)

    def test_post_segment_keyframe_policy_security_and_path_traversal(self) -> None:
        # 1. Cross-origin CSRF blocked
        csrf = _ImageHttpHandler(
            "/api/segment-keyframe",
            {"image_path": "README.md", "segmenter": "birefnet"},
            headers={"Host": "127.0.0.1:8765", "Origin": "https://attacker.example"},
        )
        csrf.do_POST()
        self.assertEqual(csrf.responses.get("error"), 403)

        # 2. Path traversal on image_path blocked
        trav = _ImageHttpHandler("/api/segment-keyframe", {
            "image_path": "../../outside/secret.png",
            "segmenter": "birefnet",
        })
        trav.do_POST()
        self.assertFalse(trav.responses["json"]["ok"])
        self.assertIn("path traversal", trav.responses["json"]["error"].lower())

        # 3. Commercial Guard blocks sam3_image / sam31_image
        for nc_seg in ("sam3_image", "sam31_image"):
            blocked = _ImageHttpHandler("/api/segment-keyframe", {
                "image_path": "intermediate/ref.png",
                "segmenter": nc_seg,
                "text_prompt": "coat",
            })
            blocked.do_POST()
            self.assertFalse(blocked.responses["json"]["ok"])
            self.assertIn("Meta SAM License", blocked.responses["json"]["error"])

        # 4. Success with valid served path and mocked run_keyframe_segmentation
        with (
            mock.patch("ai_remaster_gui.http_handler.resolve_served") as mock_served,
            mock.patch("ai_remaster_gui.image_segmenters.run_keyframe_segmentation") as mock_run,
        ):
            fake_img = mock.MagicMock(spec=Path)
            fake_img.is_file.return_value = True
            fake_img.stem = "ref_0"
            mock_served.return_value = fake_img
            mock_run.return_value = {
                "mask_path": "intermediate/keyframe_masks/ref_0_evf_sam2.png",
                "sidecar_path": "intermediate/keyframe_masks/ref_0_evf_sam2.json",
                "mask_sha256": "a" * 64,
                "segmenter": "evf_sam2",
                "category": "promptable",
                "labels": ["coat"],
                "foreground_pixels": 120,
                "width": 64,
                "height": 48,
            }
            ok_h = _ImageHttpHandler("/api/segment-keyframe", {
                "image_path": "manifests/references/ref_0.png",
                "segmenter": "evf_sam2",
                "text_prompt": "coat",
            })
            ok_h.do_POST()
            self.assertTrue(ok_h.responses["json"]["ok"])
            self.assertEqual(ok_h.responses["json"]["segmenter"], "evf_sam2")
            self.assertEqual(mock_run.call_args.kwargs["text_prompt"], "coat")


if __name__ == "__main__":
    unittest.main()
