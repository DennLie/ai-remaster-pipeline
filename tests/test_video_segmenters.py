"""CPU-only unit & integration tests for pluggable VideoSegmenterBackend & VideoSegmenterRegistry."""

from __future__ import annotations

import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from ai_remaster_gui import http_handler, state
from ai_remaster_gui import server  # noqa: F401
from ai_remaster_gui.commercial_policy import (
    CommercialPolicyError,
    evaluate_component,
    stage_issue,
)
from ai_remaster_gui.compliance import load_compliance_report, validate_compliance_report
from ai_remaster_gui.rle_masks import MaskObjectMetadata, ShotMaskArchive, load_shot_masks_npz
from ai_remaster_gui.runtime_settings import base_settings
from ai_remaster_gui.video_segmenters import (
    DEFAULT_SEGMENTER_REGISTRY,
    GroundedSAM2SegmenterBackend,
    SAMURAISegmenterBackend,
    VideoMaMaSegmenterBackend,
    VideoSegmenterBackend,
    run_shot_segmentation,
)
from scripts.osi_masked_lab import apply_strict_l_lock, colorize_shot_osi_masked_lab
from scripts.track_shot_masks import main as cli_main


def _make_sam_like_predictor(gt_masks_by_t: dict[int, dict[int, np.ndarray]]) -> mock.MagicMock:
    pred = mock.MagicMock()
    pred.init_state.return_value = {"state": "active"}

    def _prop(_state):
        for t in sorted(gt_masks_by_t.keys()):
            obj_ids = sorted(gt_masks_by_t[t].keys())
            logits = [
                np.where(gt_masks_by_t[t][oid] > 0, 4.0, -4.0).astype(np.float32)[np.newaxis, ...]
                for oid in obj_ids
            ]
            yield (t, obj_ids, logits)

    pred.propagate_in_video.side_effect = _prop
    return pred


class _RouteTestHandler(http_handler.Handler):
    def __init__(self, path: str, body: dict | None = None) -> None:
        self.path = path
        self.headers = {"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"}
        self._body = body or {}
        self.responses: dict[str, object] = {}

    def read_json(self) -> dict:
        return self._body

    def send_json(self, payload) -> None:
        self.responses["json"] = payload

    def send_error(self, code, message=None, explain=None) -> None:
        self.responses["error"] = code


class TestVideoSegmenterRegistryAndArchives(unittest.TestCase):
    """(a) Verify all 9 segmenters register cleanly and produce bit-exact .masks.npz usable by osi_masked_lab."""

    def test_all_9_segmenters_registered_and_produce_osi_compatible_npz(self) -> None:
        specs = DEFAULT_SEGMENTER_REGISTRY.list_specs()
        names = [s.name for s in specs]
        expected_names = [
            "sam21",
            "samurai",
            "edgetam",
            "cutie",
            "grounded_sam2",
            "rf_detr_seg",
            "videomama",
            "sam3",
            "sam31",
        ]
        self.assertEqual(names, expected_names)

        h, w, t_count = 48, 64, 3
        gray_frames = [np.full((h, w), 110, dtype=np.uint8) for _ in range(t_count)]
        rgb_frames = [cv2.cvtColor(g, cv2.COLOR_GRAY2RGB) for g in gray_frames]

        gt_by_t: dict[int, dict[int, np.ndarray]] = {}
        for t in range(t_count):
            m = np.zeros((h, w), dtype=np.uint8)
            m[10:34, 14 + t * 4 : 38 + t * 4] = 255
            gt_by_t[t] = {1: m}

        # Color keyframe 0: Red inside object mask, blue outside
        chroma_k0 = np.zeros((h, w, 2), dtype=np.float32)
        chroma_k0[:, :] = [-10.0, -30.0]
        chroma_k0[gt_by_t[0][1] > 0] = [42.0, 22.0]
        key_0 = apply_strict_l_lock(gray_frames[0], chroma_k0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            for seg_name in expected_names:
                with self.subTest(segmenter=seg_name):
                    prompts: dict[int, dict[str, object]] = {
                        1: {
                            "frame_idx": 0,
                            "box": [14, 10, 38, 34],
                            "mask": gt_by_t[0][1],
                            "name": "hero",
                            "entity_type": "character",
                            "entity_id": "char_hero",
                        }
                    }
                    text_prompts = None
                    backend_kwargs: dict[str, object] = {}

                    if seg_name in {"sam21", "samurai", "edgetam", "sam3", "sam31"}:
                        backend_kwargs["predictor"] = _make_sam_like_predictor(gt_by_t)
                    elif seg_name == "cutie":
                        runner = mock.MagicMock()

                        def _cutie_step(frame, mask=None, objects=None, _gt=gt_by_t, _idx=[0]):
                            t_cur = _idx[0]
                            _idx[0] += 1
                            prob = (_gt[t_cur][1] > 0).astype(np.float32)
                            return np.stack([1.0 - prob, prob], axis=0)

                        runner.step.side_effect = _cutie_step
                        backend_kwargs["runner"] = runner
                    elif seg_name == "grounded_sam2":
                        detector = mock.MagicMock()
                        detector.predict_boxes.return_value = [("hero", [14, 10, 38, 34])]
                        backend_kwargs["detector"] = detector
                        backend_kwargs["predictor"] = _make_sam_like_predictor(gt_by_t)
                        prompts = {}
                        text_prompts = ["hero"]
                    elif seg_name == "rf_detr_seg":
                        runner = mock.MagicMock()

                        def _rfdetr_pred(_rgb, threshold=0.4, _gt=gt_by_t, _idx=[0]):
                            t_cur = _idx[0]
                            _idx[0] += 1
                            return {"masks": np.stack([_gt[t_cur][1] > 0], axis=0), "obj_ids": [1]}

                        runner.predict.side_effect = _rfdetr_pred
                        backend_kwargs["runner"] = runner
                    elif seg_name == "videomama":
                        runner = mock.MagicMock()

                        def _matte_shot(_frames, guide_mask=None, _gt=gt_by_t):
                            return [(_gt[t][1].astype(np.float32) / 255.0) for t in range(t_count)]

                        runner.matte_shot.side_effect = _matte_shot
                        backend_kwargs["runner"] = runner

                    out_npz = tmp_path / f"{seg_name}.masks.npz"
                    allow_nc = seg_name in {"sam3", "sam31"}
                    backend = DEFAULT_SEGMENTER_REGISTRY.create(
                        seg_name,
                        allow_non_commercial_benchmark=allow_nc,
                        **backend_kwargs,
                    )
                    self.assertIsInstance(backend, VideoSegmenterBackend)
                    backend.close()

                    produced = run_shot_segmentation(
                        shot_id=f"shot_{seg_name}",
                        frames=rgb_frames,
                        prompts=prompts,
                        output_npz_path=out_npz,
                        segmenter=seg_name,
                        text_prompts=text_prompts,
                        allow_non_commercial_benchmark=allow_nc,
                        **backend_kwargs,
                    )
                    self.assertEqual(produced, out_npz)
                    self.assertTrue(out_npz.is_file())

                    archive = load_shot_masks_npz(out_npz)
                    self.assertEqual(archive.frame_count, t_count)
                    self.assertEqual((archive.height, archive.width), (h, w))
                    for t in range(t_count):
                        decoded = archive.get_mask(t, 1)
                        self.assertIsNotNone(decoded)
                        np.testing.assert_array_equal(decoded, gt_by_t[t][1])

                    # Consume directly in osi_masked_lab
                    colorized_dir = tmp_path / f"col_{seg_name}"
                    out_frames = colorize_shot_osi_masked_lab(
                        source=gray_frames,
                        keyframes={0: key_0},
                        mask_archive=out_npz,
                        output_path=colorized_dir,
                    )
                    self.assertEqual(len(out_frames), t_count)

    def test_videomama_alpha_threshold_and_vram_cleanup(self) -> None:
        h, w = 32, 32
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(2)]
        soft_matte = np.full((h, w), 0.2, dtype=np.float32)
        soft_matte[8:24, 8:24] = 0.72

        runner = mock.MagicMock()
        runner.matte_shot.return_value = [soft_matte, soft_matte]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_hi = Path(tmpdir) / "hi.masks.npz"
            out_lo = Path(tmpdir) / "lo.masks.npz"
            with mock.patch("torch.cuda.is_available", return_value=True), mock.patch("torch.cuda.empty_cache") as mock_empty:
                run_shot_segmentation(
                    shot_id="s1",
                    frames=frames,
                    prompts={1: {"mask": (soft_matte > 0.5).astype(np.uint8) * 255}},
                    output_npz_path=out_hi,
                    segmenter="videomama",
                    alpha_threshold=0.8,
                    runner=runner,
                )
                run_shot_segmentation(
                    shot_id="s1",
                    frames=frames,
                    prompts={1: {"mask": (soft_matte > 0.5).astype(np.uint8) * 255}},
                    output_npz_path=out_lo,
                    segmenter="videomama",
                    alpha_threshold=0.6,
                    runner=runner,
                )
                mock_empty.assert_called()

            a_hi = ShotMaskArchive.load_npz(out_hi)
            a_lo = ShotMaskArchive.load_npz(out_lo)
            self.assertIsNone(a_hi.get_mask(0, 1))
            self.assertIsNotNone(a_lo.get_mask(0, 1))

    def test_grounded_sam2_supports_inner_trackers(self) -> None:
        h, w = 40, 40
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(2)]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:25, 10:25] = 255

        for inner in ("sam21", "edgetam", "samurai"):
            detector = mock.MagicMock()
            detector.predict_boxes.return_value = [("actor", [10, 10, 25, 25])]
            pred = _make_sam_like_predictor({0: {1: mask}, 1: {1: mask}})
            backend = GroundedSAM2SegmenterBackend(
                inner_tracker=inner,
                detector=detector,
                predictor=pred,
            )
            with backend:
                backend.init_shot(frames)
                metas = backend.add_prompts({}, text_prompts="actor")
                self.assertEqual(len(metas), 1)
                self.assertEqual(metas[0].name, "actor")
                tracked = list(backend.track_shot())
                self.assertEqual(len(tracked), 2)
                np.testing.assert_array_equal(tracked[1][1][1], mask)


class TestSegmenterCommercialPolicyAndCompliance(unittest.TestCase):
    """(b) & (c) Commercial Guard enforcement and non-commercial benchmark override compliance logging."""

    def test_commercial_mode_allows_7_osi_and_blocks_sam3_sam31(self) -> None:
        strict_settings = base_settings()
        strict_settings["global"]["commercial_model_guard"] = "true"
        strict_settings["colour"]["method"] = "osi_masked_lab"

        osi_segmenters = ["sam21", "samurai", "edgetam", "cutie", "grounded_sam2", "rf_detr_seg", "videomama"]
        for seg in osi_segmenters:
            with self.subTest(osi_segmenter=seg):
                decisions = DEFAULT_SEGMENTER_REGISTRY.check_policy(seg, settings=strict_settings)
                self.assertTrue(all(d.allowed for d in decisions))
                strict_settings["colour"]["segmenter"] = seg
                self.assertEqual(stage_issue(strict_settings, "colour"), "")

        for nc_seg in ("sam3", "sam31"):
            with self.subTest(blocked_segmenter=nc_seg):
                dec = evaluate_component(nc_seg, settings=strict_settings, component_type="tool")
                self.assertFalse(dec.allowed)
                self.assertEqual(dec.status, "hard_blocked")
                self.assertIn("Meta SAM License", dec.reason)

                with self.assertRaises(CommercialPolicyError) as ctx:
                    DEFAULT_SEGMENTER_REGISTRY.check_policy(nc_seg, settings=strict_settings)
                self.assertIn("Meta SAM License", str(ctx.exception))

                strict_settings["colour"]["segmenter"] = nc_seg
                issue = stage_issue(strict_settings, "colour")
                self.assertIn("Meta SAM License", issue)

    def test_benchmark_override_unlocks_sam3_and_marks_compliance_non_commercial(self) -> None:
        h, w = 40, 40
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(2)]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[10:25, 10:25] = 255

        override_settings = base_settings()
        override_settings["global"]["commercial_model_guard"] = "false"
        override_settings["global"]["non_commercial_benchmark_override"] = "true"
        self.assertEqual(stage_issue(override_settings, "colour"), "")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            out_dir = tmp_path / "output"
            # 1. Strict OSI run is compliant
            osi_npz = tmp_path / "osi.masks.npz"
            run_shot_segmentation(
                shot_id="shot_osi",
                frames=frames,
                prompts={1: {"box": [10, 10, 25, 25]}},
                output_npz_path=osi_npz,
                segmenter="sam21",
                settings=base_settings(),
                compliance_output_dir=out_dir,
                predictor=_make_sam_like_predictor({0: {1: mask}, 1: {1: mask}}),
            )
            rep_osi = load_compliance_report(out_dir)
            self.assertIsNotNone(rep_osi)
            self.assertTrue(rep_osi["overall_commercial_compliant"])
            self.assertTrue(rep_osi["runs"][0]["is_commercial_compliant"])

            # 2. SAM 3.1 benchmark run unlocks and flips overall_commercial_compliant -> False
            nc_out_dir = tmp_path / "output_nc"
            sam31_npz = tmp_path / "sam31.masks.npz"
            run_shot_segmentation(
                shot_id="shot_bench",
                frames=frames,
                prompts={1: {"box": [10, 10, 25, 25]}},
                output_npz_path=sam31_npz,
                segmenter="sam31",
                text_prompts="actor",
                settings=override_settings,
                allow_non_commercial_benchmark=True,
                compliance_output_dir=nc_out_dir,
                predictor=_make_sam_like_predictor({0: {1: mask}, 1: {1: mask}}),
            )
            valid, errs = validate_compliance_report(nc_out_dir)
            self.assertTrue(valid, f"Compliance report invalid: {errs}")
            rep_nc = load_compliance_report(nc_out_dir)
            self.assertFalse(rep_nc["overall_commercial_compliant"])
            self.assertFalse(rep_nc["runs"][0]["is_commercial_compliant"])
            self.assertEqual(rep_nc["runs"][0]["guard_mode"], "benchmark_override")
            self.assertTrue(rep_nc["runs"][0]["metadata"]["non_commercial_benchmark_run"])
            self.assertEqual(rep_nc["runs"][0]["metadata"]["segmenter"], "sam31")

    def test_cli_allow_non_commercial_benchmark_flag_and_strict_block(self) -> None:
        h, w = 32, 32
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir()
            for idx in range(2):
                cv2.imwrite(str(frames_dir / f"{idx:05d}.png"), np.zeros((h, w, 3), dtype=np.uint8))

            out_npz = tmp_path / "cli_sam3.masks.npz"
            comp_dir = tmp_path / "compliance_out"

            # Without override: exit code 2 (CommercialPolicyError)
            rc_blocked = cli_main([
                "--segmenter", "sam3",
                "--shot-id", "shot_cli",
                "--frames", str(frames_dir),
                "--output", str(out_npz),
                "--compliance-output-dir", str(comp_dir),
            ])
            self.assertEqual(rc_blocked, 2)

            # With --allow-non-commercial-benchmark and mocked sam3 module: succeeds & logs non-compliant
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[4:20, 4:20] = 255
            fake_sam3 = mock.MagicMock()
            fake_sam3.build_sam3_video_predictor.return_value = _make_sam_like_predictor({0: {1: mask}, 1: {1: mask}})
            with mock.patch.dict(sys.modules, {"sam3": fake_sam3}):
                rc_ok = cli_main([
                    "--segmenter", "sam3",
                    "--shot-id", "shot_cli",
                    "--frames", str(frames_dir),
                    "--text-prompts", "hero",
                    "--output", str(out_npz),
                    "--compliance-output-dir", str(comp_dir),
                    "--allow-non-commercial-benchmark",
                ])
            self.assertEqual(rc_ok, 0)
            self.assertTrue(out_npz.is_file())
            rep = load_compliance_report(comp_dir)
            self.assertFalse(rep["overall_commercial_compliant"])


class TestSamuraiKalmanOcclusionStability(unittest.TestCase):
    """(d) SAMURAI Kalman bbox gating maintains stable IDs across crossing & occlusion swaps."""

    def test_samurai_prevents_id_swap_across_crossing_occlusion(self) -> None:
        h, w = 100, 160

        def _box_mask(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
            m = np.zeros((h, w), dtype=np.uint8)
            m[y1:y2, x1:x2] = 255
            return m

        # Object 1 moves right in upper-middle lane: x = 10, 32, 54, 76, 98
        # Object 2 moves left in intersecting lane: x = 110, 88, (occluded at t=2), 44, 22
        m1_frames = [_box_mask(10 + t * 22, 30, 32 + t * 22, 55) for t in range(5)]
        m2_frames = [
            _box_mask(110, 34, 132, 59),  # t=0
            _box_mask(88, 34, 110, 59),   # t=1
            np.zeros((h, w), dtype=np.uint8),  # t=2 occluded!
            _box_mask(44, 34, 66, 59),    # t=3 re-emerges on the left
            _box_mask(22, 34, 44, 59),    # t=4 continues left
        ]

        # Simulate raw predictor swapping raw IDs (1 <-> 2) right after t=2 occlusion!
        raw_stream = []
        for t in range(5):
            if t < 2:
                raw_1, raw_2 = m1_frames[t], m2_frames[t]
            elif t == 2:
                raw_1, raw_2 = m1_frames[t], m2_frames[t]
            else:
                # Raw predictor erroneously labels obj 2 as raw ID 1 and obj 1 as raw ID 2
                raw_1, raw_2 = m2_frames[t], m1_frames[t]
            logits = [
                np.where(raw_1 > 0, 5.0, -5.0).astype(np.float32)[np.newaxis, ...],
                np.where(raw_2 > 0, 5.0, -5.0).astype(np.float32)[np.newaxis, ...],
            ]
            raw_stream.append((t, [1, 2], logits))

        pred = mock.MagicMock()
        pred.init_state.return_value = {"id": "crossing"}
        pred.propagate_in_video.return_value = raw_stream

        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(5)]
        tracker = SAMURAISegmenterBackend(predictor=pred)
        tracker.init_shot(frames)
        tracker.add_prompts({
            1: {"frame_idx": 0, "box": [10, 30, 32, 55], "name": "runner_right"},
            2: {"frame_idx": 0, "box": [110, 34, 132, 59], "name": "runner_left"},
        })
        tracked = dict(tracker.track_shot())
        tracker.close()

        # Verify tracked ID 1 stayed with m1_frames and tracked ID 2 recovered m2_frames without swapping!
        for t in (0, 1, 3, 4):
            np.testing.assert_array_equal(tracked[t][1], m1_frames[t])
            np.testing.assert_array_equal(tracked[t][2], m2_frames[t])


class TestMissingOptionalDependenciesRaiseActionableErrors(unittest.TestCase):
    """(e) Missing packages raise explicit RuntimeError without generating dummy masks."""

    def test_missing_optional_packages_raise_runtime_error(self) -> None:
        real_import = builtins.__import__
        blocked_prefixes = ("sam2", "edgetam", "cutie", "groundingdino", "transformers", "rfdetr", "videomama", "sam3")

        def _failing_import(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == p or name.startswith(p + ".") for p in blocked_prefixes):
                raise ImportError(f"Simulated missing package '{name}'")
            return real_import(name, globals, locals, fromlist, level)

        saved = {k: sys.modules.pop(k) for k in list(sys.modules.keys()) if any(k == p or k.startswith(p + ".") for p in blocked_prefixes)}
        try:
            with mock.patch("builtins.__import__", side_effect=_failing_import):
                for seg_name in ("sam21", "samurai", "edgetam", "cutie", "grounded_sam2", "rf_detr_seg", "videomama", "sam3", "sam31"):
                    with self.subTest(segmenter=seg_name):
                        with self.assertRaises(RuntimeError):
                            DEFAULT_SEGMENTER_REGISTRY.create(
                                seg_name,
                                allow_non_commercial_benchmark=(seg_name in {"sam3", "sam31"}),
                            )
        finally:
            sys.modules.update(saved)


class TestSegmenterHttpEndpoint(unittest.TestCase):
    """(f) HTTP endpoint validation for POST /api/sam2-track-shot & /api/track-shot."""

    def setUp(self) -> None:
        self._real_app = state.APP
        self.app = mock.MagicMock()
        self.app.log = []
        self.app.settings = base_settings()
        self.app.state.return_value = {}
        state.APP = self.app

    def tearDown(self) -> None:
        state.APP = self._real_app

    def test_track_shot_routes_accept_segmenter_and_enforce_policy(self) -> None:
        for route in ("/api/sam2-track-shot", "/api/track-shot"):
            with self.subTest(route=route):
                with mock.patch("ai_remaster_gui.sam2_tracker.run_shot_tracking_pipeline") as mock_run:
                    mock_run.return_value = Path("intermediate/shot_masks/shot_001.masks.npz")
                    h = _RouteTestHandler(route, {
                        "shot_id": "shot_001",
                        "frames": "intermediate",
                        "segmenter": "grounded_sam2",
                        "text_prompts": ["coat", "hat"],
                        "prompts": {},
                    })
                    h.do_POST()
                    self.assertTrue(h.responses["json"]["ok"])
                    self.assertEqual(mock_run.call_args.kwargs["segmenter"], "grounded_sam2")
                    self.assertEqual(mock_run.call_args.kwargs["text_prompts"], ["coat", "hat"])

    def test_track_shot_blocks_sam3_in_commercial_mode_and_allows_when_overridden(self) -> None:
        # Commercial Guard ON -> sam3 & sam31 blocked
        for seg in ("sam3", "sam31"):
            h = _RouteTestHandler("/api/track-shot", {
                "shot_id": "shot_001",
                "frames": "intermediate",
                "segmenter": seg,
                "prompts": {},
            })
            h.do_POST()
            self.assertFalse(h.responses["json"]["ok"])
            self.assertIn("Meta SAM License", h.responses["json"]["error"])

        # Commercial Guard OFF + benchmark override -> unlocked
        self.app.settings["global"]["commercial_model_guard"] = "false"
        self.app.settings["global"]["non_commercial_benchmark_override"] = "true"
        with mock.patch("ai_remaster_gui.sam2_tracker.run_shot_tracking_pipeline") as mock_run:
            mock_run.return_value = Path("intermediate/shot_masks/shot_001.masks.npz")
            h = _RouteTestHandler("/api/track-shot", {
                "shot_id": "shot_001",
                "frames": "intermediate",
                "segmenter": "sam31",
                "text_prompts": "hero",
                "prompts": {},
            })
            h.do_POST()
            self.assertTrue(h.responses["json"]["ok"])
            self.assertEqual(mock_run.call_args.kwargs["segmenter"], "sam31")

    def test_track_shot_rejects_unknown_segmenter_and_path_traversal(self) -> None:
        bad_seg = _RouteTestHandler("/api/sam2-track-shot", {
            "shot_id": "shot_001",
            "frames": "intermediate",
            "segmenter": "unknown_seg_xyz",
            "prompts": {},
        })
        bad_seg.do_POST()
        self.assertFalse(bad_seg.responses["json"]["ok"])
        self.assertIn("Unknown video segmenter", bad_seg.responses["json"]["error"])

        bad_frames = _RouteTestHandler("/api/track-shot", {
            "shot_id": "shot_001",
            "frames": "../../outside_secret_frames",
            "segmenter": "cutie",
            "prompts": {},
        })
        bad_frames.do_POST()
        self.assertFalse(bad_frames.responses["json"]["ok"])
        self.assertIn("path traversal", bad_frames.responses["json"]["error"].lower())


if __name__ == "__main__":
    unittest.main()
