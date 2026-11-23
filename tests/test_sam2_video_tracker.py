from __future__ import annotations

import gc
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_remaster_gui.commercial_policy import CommercialPolicyError
from ai_remaster_gui.rle_masks import MaskObjectMetadata, ShotMaskArchive
from ai_remaster_gui.sam2_tracker import (
    ByteTrackAdapterHook,
    NullTrackerAdapter,
    PreTrackerAdapter,
    ShotVideoTracker,
    run_shot_tracking_pipeline,
    verify_sam2_installation,
)
from scripts.sam2_video_tracker import select_best_candidate_mask


def _create_mock_official_sam2_module() -> types.ModuleType:
    """Create a mock module simulating official facebookresearch/sam2."""
    mod = types.ModuleType("sam2")
    mod.__file__ = str(Path("site-packages/sam2/__init__.py"))
    mod.__version__ = "2.1.0"
    mod.build_sam = types.ModuleType("sam2.build_sam")
    mod.sam2_video_predictor = types.ModuleType("sam2.sam2_video_predictor")
    mod.build_sam2_video_predictor = mock.MagicMock()
    mod.SAM2VideoPredictor = mock.MagicMock()
    mod.modeling = types.ModuleType("sam2.modeling")
    return mod


class TestSAM2InstallationVerification(unittest.TestCase):
    """Unit tests for verify_sam2_installation fail-closed sanity checks."""

    def test_reject_sam3_checkpoint_names(self) -> None:
        """Verify strict rejection of SAM 3.x / non-commercial research checkpoints."""
        blocked_checkpoints = [
            "sam3.1.pt",
            "sam3.1_hiera_large.pt",
            "sam3_base.pt",
            "sam-3.1-hiera.pt",
            "facebook/sam-3.1",
            "checkpoints/sam3.pt",
            "sam 3.1 checkpoint.pt",
        ]
        for ckpt in blocked_checkpoints:
            with self.subTest(checkpoint=ckpt):
                with self.assertRaises(CommercialPolicyError) as ctx:
                    verify_sam2_installation(ckpt)
                self.assertIn("SAM 3", str(ctx.exception))

    def test_reject_non_apache_or_unsupported_checkpoint(self) -> None:
        """Verify rejection of non-Apache or unknown model checkpoints."""
        unsupported = [
            "proprietary_tracker.pt",
            "custom_yolo_weights.pth",
            "some_non_apache_weights.pt",
        ]
        for ckpt in unsupported:
            with self.subTest(checkpoint=ckpt):
                with self.assertRaises(CommercialPolicyError):
                    verify_sam2_installation(ckpt)

    def test_reject_unofficial_pypi_fork(self) -> None:
        """Verify fail-closed rejection when unofficial PyPI sam2 package is installed."""
        # Case A: Missing video predictor / hiera modeling components
        fork_mod = types.ModuleType("sam2")
        fork_mod.__file__ = str(Path("site-packages/sam2/__init__.py"))
        fork_mod.__version__ = "0.1.0"
        with mock.patch.dict(sys.modules, {"sam2": fork_mod}):
            with self.assertRaises(RuntimeError) as ctx:
                verify_sam2_installation("sam2.1_hiera_large.pt")
            self.assertIn("Unofficial PyPI", str(ctx.exception))

        # Case B: Module explicitly flagged as unofficial
        flagged_mod = _create_mock_official_sam2_module()
        flagged_mod._is_unofficial_pypi = True
        with mock.patch.dict(sys.modules, {"sam2": flagged_mod}):
            with self.assertRaises(RuntimeError) as ctx:
                verify_sam2_installation("sam2.1_hiera_large.pt")
            self.assertIn("Unofficial PyPI", str(ctx.exception))

        # Case C: Module file path has 'unofficial'
        path_mod = _create_mock_official_sam2_module()
        path_mod.__file__ = "C:/env/site-packages/unofficial_sam2/__init__.py"
        with mock.patch.dict(sys.modules, {"sam2": path_mod}):
            with self.assertRaises(RuntimeError) as ctx:
                verify_sam2_installation("sam2.1_hiera_large.pt")
            self.assertIn("Unofficial PyPI", str(ctx.exception))

        # Case D: Module has builder and modeling, but lacks video predictor
        no_video_mod = types.ModuleType("sam2")
        no_video_mod.__file__ = str(Path("site-packages/sam2/__init__.py"))
        no_video_mod.__version__ = "2.1.0"
        no_video_mod.build_sam = types.ModuleType("sam2.build_sam")
        no_video_mod.modeling = types.ModuleType("sam2.modeling")
        with mock.patch.dict(sys.modules, {"sam2": no_video_mod}):
            with self.assertRaises(RuntimeError) as ctx:
                verify_sam2_installation("sam2.1_hiera_large.pt")
            self.assertIn("Unofficial PyPI", str(ctx.exception))

    def test_accept_valid_sam21_checkpoints(self) -> None:
        """Verify acceptance of valid official SAM 2.1 checkpoints."""
        official_mod = _create_mock_official_sam2_module()
        valid_checkpoints = [
            "sam2.1_hiera_base_plus.pt",
            "sam2.1_hiera_large.pt",
            "sam2.1_hiera_tiny.pt",
            "sam2.1_hiera_small.pt",
            "facebook/sam2.1-hiera-large",
            "models/checkpoints/sam2.1_hiera_large.pt",
        ]
        with mock.patch.dict(sys.modules, {"sam2": official_mod}):
            for ckpt in valid_checkpoints:
                with self.subTest(checkpoint=ckpt):
                    res = verify_sam2_installation(ckpt)
                    self.assertEqual(res["status"], "verified")
                    self.assertEqual(res["license"], "Apache-2.0")
                    self.assertEqual(res["package"], "facebookresearch/sam2")
                    self.assertEqual(res["checkpoint"], ckpt)

    def test_missing_sam2_package_raises_runtime_error(self) -> None:
        """Verify RuntimeError is raised when sam2 package is not installed."""
        orig_sam2 = sys.modules.pop("sam2", None)
        try:
            with mock.patch.dict(sys.modules, {}):
                with mock.patch("builtins.__import__", side_effect=ImportError("No module named 'sam2'")):
                    with self.assertRaises(RuntimeError) as ctx:
                        verify_sam2_installation("sam2.1_hiera_large.pt")
                    self.assertIn("not installed", str(ctx.exception))
        finally:
            if orig_sam2 is not None:
                sys.modules["sam2"] = orig_sam2


class TestShotVideoTrackerLifecycle(unittest.TestCase):
    """Unit tests for ShotVideoTracker lifecycle and memory management."""

    def setUp(self) -> None:
        self.mock_predictor = mock.MagicMock()
        self.mock_predictor.init_state.return_value = {"inference_state_id": "state_123"}
        self.frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(5)]

    def test_init_shot_and_close_lifecycle(self) -> None:
        """Verify predictor.init_state and reset_state are strictly invoked."""
        tracker = ShotVideoTracker(predictor=self.mock_predictor)
        self.assertIsNone(tracker.inference_state)

        # 1. Initialize shot
        state = tracker.init_shot(self.frames)
        self.assertEqual(state, {"inference_state_id": "state_123"})
        self.assertEqual(tracker.inference_state, {"inference_state_id": "state_123"})
        self.mock_predictor.init_state.assert_called_once()

        # 2. Close tracker
        tracker.close()
        self.assertIsNone(tracker.inference_state)
        self.assertTrue(tracker._is_closed)
        self.mock_predictor.reset_state.assert_called_once_with({"inference_state_id": "state_123"})

    def test_consecutive_shots_memory_cleanup(self) -> None:
        """Verify subsequent init_shot call cleans up prior shot state without leaking."""
        tracker = ShotVideoTracker(predictor=self.mock_predictor)
        tracker.init_shot(self.frames)

        # Initial state present
        self.mock_predictor.reset_state.assert_not_called()

        # Re-initialize with new frames
        new_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(3)]
        self.mock_predictor.init_state.return_value = {"inference_state_id": "state_456"}
        tracker.init_shot(new_frames)

        # Ensure reset_state was called for the previous state
        self.mock_predictor.reset_state.assert_called_once_with({"inference_state_id": "state_123"})
        self.assertEqual(tracker.inference_state, {"inference_state_id": "state_456"})

        tracker.close()
        self.assertEqual(self.mock_predictor.reset_state.call_count, 2)

    def test_context_manager_normal_and_exception(self) -> None:
        """Verify context manager guarantees cleanup on both clean exit and exceptions."""
        # Clean exit
        with ShotVideoTracker(predictor=self.mock_predictor) as tracker:
            tracker.init_shot(self.frames)
        self.mock_predictor.reset_state.assert_called_once()
        self.assertIsNone(tracker.inference_state)

        self.mock_predictor.reset_state.reset_mock()

        # Exception exit
        with self.assertRaises(ValueError):
            with ShotVideoTracker(predictor=self.mock_predictor) as tracker:
                tracker.init_shot(self.frames)
                raise ValueError("Simulated pipeline error")

        self.mock_predictor.reset_state.assert_called_once()
        self.assertIsNone(tracker.inference_state)

    @mock.patch("torch.cuda.empty_cache")
    @mock.patch("torch.cuda.is_available", return_value=True)
    def test_cuda_cache_cleared_on_close(self, mock_is_available: mock.MagicMock, mock_empty_cache: mock.MagicMock) -> None:
        """Verify torch.cuda.empty_cache is triggered on close when CUDA is available."""
        tracker = ShotVideoTracker(predictor=self.mock_predictor)
        tracker.init_shot(self.frames)
        tracker.close()
        mock_empty_cache.assert_called()

    def test_init_shot_slice_staging_in_multi_image_folder(self) -> None:
        """Verify passing a slice of image files creates a managed temporary directory and cleans up."""
        import cv2

        with tempfile.TemporaryDirectory() as src_tmp:
            src_dir = Path(src_tmp)
            frame_paths = []
            for i in range(5):
                fp = src_dir / f"frame_{i:03d}.jpg"
                cv2.imwrite(str(fp), np.zeros((32, 32, 3), dtype=np.uint8))
                frame_paths.append(fp)

            tracker = ShotVideoTracker(predictor=self.mock_predictor)

            # 1. Pass full folder (all 5 frames) -> no staging directory needed
            tracker.init_shot(frame_paths)
            self.assertIsNone(tracker._temp_dir)
            self.mock_predictor.init_state.assert_called_with(video_path=str(src_dir))

            self.mock_predictor.init_state.reset_mock()

            # 2. Pass a slice of 2 frames ([1, 2]) -> staging directory created
            slice_paths = [frame_paths[1], frame_paths[2]]
            tracker.init_shot(slice_paths)
            self.assertIsNotNone(tracker._temp_dir)
            staged_dir = Path(tracker._temp_dir.name)
            self.assertTrue(staged_dir.is_dir())
            staged_files = sorted(staged_dir.iterdir())
            self.assertEqual(len(staged_files), 2)
            self.assertEqual(staged_files[0].name, "00000.jpg")
            self.assertEqual(staged_files[1].name, "00001.jpg")
            self.mock_predictor.init_state.assert_called_with(video_path=tracker._temp_dir.name)

            # 3. Close tracker -> cleans up temporary directory
            tracker.close()
            self.assertIsNone(tracker._temp_dir)
            self.assertFalse(staged_dir.exists())

    def test_track_shot_runs_in_inference_mode(self) -> None:
        """Verify propagate_in_video is executed within torch.inference_mode context."""
        tracker = ShotVideoTracker(predictor=self.mock_predictor)
        tracker.init_shot(self.frames)
        self.mock_predictor.propagate_in_video.return_value = [
            (0, [1], np.zeros((1, 1, 64, 64), dtype=np.float32))
        ]

        with mock.patch("torch.inference_mode") as mock_inf:
            mock_ctx = mock.MagicMock()
            mock_inf.return_value = mock_ctx
            results = list(tracker.track_shot())
            self.assertEqual(len(results), 1)
            mock_inf.assert_called_once()
            mock_ctx.__enter__.assert_called()


class TestCandidateSelectionAndEmptyBackgroundHeuristic(unittest.TestCase):
    """Unit tests for float32 (3, H, W) candidate selection and empty-background heuristics."""

    def test_select_best_candidate_mask_highest_activation(self) -> None:
        """Verify selection of the best candidate mask channel among 3 candidates."""
        h, w = 30, 30
        candidates = np.zeros((3, h, w), dtype=np.float32)

        # Candidate 0: Tiny noise patch (5 pixels < min_mask_area 16)
        candidates[0, :5, :1] = 5.0

        # Candidate 1: High-confidence strong foreground mask (50 pixels, mean logit 4.5)
        candidates[1, 10:20, 10:15] = 4.5

        # Candidate 2: Moderate confidence foreground mask (30 pixels, mean logit 1.2)
        candidates[2, 5:15, 5:8] = 1.2

        best = select_best_candidate_mask(candidates, min_mask_area=16)
        np.testing.assert_array_equal(best, candidates[1])

    def test_select_best_candidate_mask_explicit_scores(self) -> None:
        """Verify IoU scores override internal heuristics when provided."""
        h, w = 20, 20
        candidates = np.zeros((3, h, w), dtype=np.float32)
        candidates[0] = 1.0
        candidates[1] = 2.0
        candidates[2] = 3.0

        scores = [0.1, 0.95, 0.4]
        best = select_best_candidate_mask(candidates, scores=scores)
        np.testing.assert_array_equal(best, candidates[1])

    def test_empty_background_heuristic_below_min_area(self) -> None:
        """Verify masks with fewer than min_mask_area pixels are suppressed to all zeros."""
        mock_predictor = mock.MagicMock()
        mock_predictor.init_state.return_value = {"id": 1}

        # Candidate mask has only 8 positive pixels (< min_mask_area 16)
        tiny_mask = np.zeros((1, 1, 40, 40), dtype=np.float32)
        tiny_mask[0, 0, :2, :4] = 3.0

        mock_predictor.propagate_in_video.return_value = [
            (0, [1], tiny_mask)
        ]

        tracker = ShotVideoTracker(predictor=mock_predictor)
        tracker.init_shot([np.zeros((40, 40, 3), dtype=np.uint8)])

        results = list(tracker.track_shot(min_mask_area=16))
        self.assertEqual(len(results), 1)
        frame_idx, frame_masks = results[0]
        self.assertEqual(frame_idx, 0)
        self.assertIn(1, frame_masks)
        # Suppressed to all zeros
        self.assertEqual(np.count_nonzero(frame_masks[1]), 0)

    def test_empty_background_heuristic_background_prompt(self) -> None:
        """Verify objects prompted with background points suppress spurious full-frame masks."""
        mock_predictor = mock.MagicMock()
        mock_predictor.init_state.return_value = {"id": 1}

        # Predictor returns a spurious full-frame mask
        full_mask = np.full((1, 1, 50, 50), 4.0, dtype=np.float32)
        mock_predictor.propagate_in_video.return_value = [
            (0, [1], full_mask)
        ]

        tracker = ShotVideoTracker(predictor=mock_predictor)
        tracker.init_shot([np.zeros((50, 50, 3), dtype=np.uint8)])

        # Register prompt with all-zero labels (background only)
        tracker.add_object_points_or_mask(
            frame_idx=0,
            obj_id=1,
            points=[[10.0, 10.0], [20.0, 20.0]],
            labels=[0, 0],
        )

        results = list(tracker.track_shot(min_mask_area=16))
        _, frame_masks = results[0]
        # Spurious mask suppressed to zeros because prompt was empty background
        self.assertEqual(np.count_nonzero(frame_masks[1]), 0)

    def test_discard_empty_background_on_positive_prompt(self) -> None:
        """Verify _empty_background_objects is cleared when valid foreground prompts are added."""
        mock_predictor = mock.MagicMock()
        mock_predictor.init_state.return_value = {"id": 1}
        tracker = ShotVideoTracker(predictor=mock_predictor)
        tracker.init_shot([np.zeros((50, 50, 3), dtype=np.uint8)])

        # 1. Negative points latch obj_id 1 to background
        tracker.add_object_points_or_mask(0, 1, points=[[5.0, 5.0]], labels=[0])
        self.assertIn(1, tracker._empty_background_objects)

        # 2. Subsequent positive point clears latch
        tracker.add_object_points_or_mask(0, 1, points=[[15.0, 15.0]], labels=[1])
        self.assertNotIn(1, tracker._empty_background_objects)

        # 3. Latch obj_id 2 to background, then clear with valid box
        tracker.add_object_points_or_mask(0, 2, points=[[5.0, 5.0]], labels=[0])
        self.assertIn(2, tracker._empty_background_objects)
        tracker.add_object_points_or_mask(0, 2, box=[10.0, 10.0, 30.0, 30.0])
        self.assertNotIn(2, tracker._empty_background_objects)

        # 4. Latch obj_id 3 to background, then clear with mask (>= 16 pixels)
        tracker.add_object_points_or_mask(0, 3, points=[[5.0, 5.0]], labels=[0])
        self.assertIn(3, tracker._empty_background_objects)
        pos_mask = np.ones((10, 10), dtype=np.uint8)  # 100 pixels >= 16
        tracker.add_object_points_or_mask(0, 3, mask=pos_mask)
        self.assertNotIn(3, tracker._empty_background_objects)

        # 5. Empty mask (< 16 pixels) latches obj_id 4
        small_mask = np.ones((3, 3), dtype=np.uint8)  # 9 pixels < 16
        tracker.add_object_points_or_mask(0, 4, mask=small_mask)
        self.assertIn(4, tracker._empty_background_objects)


class TestByteTrackAdapterHook(unittest.TestCase):
    """Unit tests for PreTrackerAdapter, NullTrackerAdapter, and ByteTrackAdapterHook."""

    def test_null_tracker_adapter_protocol(self) -> None:
        """Verify NullTrackerAdapter conforms to PreTrackerAdapter protocol."""
        adapter = NullTrackerAdapter()
        self.assertIsInstance(adapter, PreTrackerAdapter)
        self.assertTrue(adapter.is_available())

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [(10.0, 12.0, 40.0, 45.0), (60.0, 62.0, 90.0, 95.0)]
        associated = adapter.update_and_associate(0, frame, boxes)

        self.assertEqual(len(associated), 2)
        self.assertEqual(associated[0], (0, (10.0, 12.0, 40.0, 45.0)))
        self.assertEqual(associated[1], (1, (60.0, 62.0, 90.0, 95.0)))

    def test_bytetrack_adapter_graceful_degradation(self) -> None:
        """Verify graceful fallback to NullTrackerAdapter when ByteTrack is unavailable."""
        with mock.patch("scripts.sam2_video_tracker._BYTE_TRACK_AVAILABLE", False):
            hook = ByteTrackAdapterHook()
            self.assertFalse(hook.is_available())

            frame = np.zeros((80, 80, 3), dtype=np.uint8)
            boxes = [(5.0, 5.0, 25.0, 25.0)]
            res = hook.update_and_associate(0, frame, boxes)
            self.assertEqual(res, [(0, (5.0, 5.0, 25.0, 25.0))])

    def test_bytetrack_adapter_with_active_tracker(self) -> None:
        """Verify ByteTrackAdapterHook delegates to underlying tracker when provided."""
        mock_tracker = mock.MagicMock()
        mock_track = mock.MagicMock(track_id=99, tlbr=np.array([15.0, 18.0, 55.0, 58.0]))
        mock_tracker.update.return_value = [mock_track]

        hook = ByteTrackAdapterHook(tracker=mock_tracker)
        self.assertTrue(hook.is_available())

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [(10.0, 15.0, 50.0, 55.0)]
        res = hook.update_and_associate(0, frame, boxes)

        mock_tracker.update.assert_called_once()
        self.assertEqual(res, [(99, (15.0, 18.0, 55.0, 58.0))])

    def test_bytetrack_adapter_with_custom_update_and_associate(self) -> None:
        """Verify ByteTrackAdapterHook supports custom trackers implementing update_and_associate."""
        class CustomTracker:
            def update_and_associate(
                self,
                frame_idx: int,
                frame_bgr: np.ndarray,
                bboxes: list[tuple[float, float, float, float]],
            ) -> list[tuple[int, tuple[float, float, float, float]]]:
                return [(777, bboxes[0])]

        hook = ByteTrackAdapterHook(tracker=CustomTracker())
        self.assertTrue(hook.is_available())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        res = hook.update_and_associate(0, frame, [(10.0, 20.0, 30.0, 40.0)])
        self.assertEqual(res, [(777, (10.0, 20.0, 30.0, 40.0))])


class TestEndToEndTrackingAndArchive(unittest.TestCase):
    """End-to-end integration tests saving into .masks.npz and verifying bit-exact load."""

    def test_track_and_save_archive_roundtrip(self) -> None:
        """Verify tracking outputs serialize to .masks.npz and reload bit-exact."""
        h, w = 60, 80
        frames_count = 3
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(frames_count)]

        # Synthetic ground truth masks
        gt_masks: dict[tuple[int, int], np.ndarray] = {}

        # Object 1: moving box
        for t in range(frames_count):
            m1 = np.zeros((h, w), dtype=np.uint8)
            m1[10 + t * 4 : 30 + t * 4, 15 + t * 5 : 45 + t * 5] = 255
            gt_masks[(1, t)] = m1

        # Object 2: static box
        for t in range(frames_count):
            m2 = np.zeros((h, w), dtype=np.uint8)
            m2[35:50, 50:70] = 255
            gt_masks[(2, t)] = m2

        # Mock predictor propagation stream
        mock_predictor = mock.MagicMock()
        mock_predictor.init_state.return_value = {"state": "active"}

        stream_yields = []
        for t in range(frames_count):
            # Output logits: logits > 0 gives foreground
            l1 = np.where(gt_masks[(1, t)] > 0, 3.0, -3.0).astype(np.float32)
            l2 = np.where(gt_masks[(2, t)] > 0, 3.0, -3.0).astype(np.float32)
            # 3-candidate format (3, H, W) for object 1, (1, H, W) for object 2
            cand1 = np.stack([l1 * 0.1, l1, l1 * 0.5])  # Channel 1 is best
            cand2 = l2[np.newaxis, ...]  # (1, H, W)
            logits_batch = [cand1, cand2]
            stream_yields.append((t, [1, 2], logits_batch))

        mock_predictor.propagate_in_video.return_value = stream_yields

        metas = [
            MaskObjectMetadata(obj_id=1, name="hero_coat", entity_type="character", entity_id="char_hero"),
            MaskObjectMetadata(obj_id=2, name="ancient_relic", entity_type="object", entity_id="obj_relic"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_npz = Path(tmpdir) / "shot_e2e_001.masks.npz"

            tracker = ShotVideoTracker(predictor=mock_predictor)
            tracker.init_shot(frames)
            tracker.add_object_points_or_mask(0, 1, box=[15, 10, 45, 30])
            tracker.add_object_points_or_mask(0, 2, box=[50, 35, 70, 50])

            archive = tracker.track_and_save_archive(
                shot_id="shot_e2e_001",
                output_npz_path=out_npz,
                object_metadatas=metas,
            )
            tracker.close()

            self.assertTrue(out_npz.is_file())

            # Load back via ShotMaskArchive and verify bit-exact consistency
            loaded = ShotMaskArchive.load_npz(out_npz)
            self.assertEqual(loaded.shot_id, "shot_e2e_001")
            self.assertEqual(loaded.frame_count, frames_count)
            self.assertEqual(loaded.height, h)
            self.assertEqual(loaded.width, w)
            self.assertEqual(len(loaded.objects), 2)

            o1 = loaded.get_object(1)
            self.assertIsNotNone(o1)
            self.assertEqual(o1.name, "hero_coat")
            self.assertEqual(o1.entity_type, "character")
            self.assertEqual(o1.entity_id, "char_hero")

            o2 = loaded.get_object(2)
            self.assertIsNotNone(o2)
            self.assertEqual(o2.name, "ancient_relic")
            self.assertEqual(o2.entity_type, "object")

            # Verify bit-exact decoded masks
            for (obj_id, t), expected_mask in gt_masks.items():
                retrieved = loaded.get_mask(t, obj_id)
                self.assertIsNotNone(retrieved)
                np.testing.assert_array_equal(retrieved, expected_mask)

    def test_run_shot_tracking_pipeline_service_wrapper(self) -> None:
        """Verify high-level run_shot_tracking_pipeline service wrapper."""
        h, w = 50, 50
        frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(2)]

        expected_mask = np.zeros((h, w), dtype=np.uint8)
        expected_mask[10:30, 10:30] = 255

        mock_predictor = mock.MagicMock()
        mock_predictor.init_state.return_value = {"id": "pipeline_state"}
        logits = np.where(expected_mask > 0, 5.0, -5.0).astype(np.float32)[np.newaxis, ...]
        mock_predictor.propagate_in_video.return_value = [
            (0, [1], [logits]),
            (1, [1], [logits]),
        ]

        prompts = {
            1: {
                "frame_idx": 0,
                "points": [[20, 20]],
                "labels": [1],
                "name": "protagonist",
                "entity_type": "character",
                "entity_id": "p01",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "pipeline_shot.masks.npz"
            saved_path = run_shot_tracking_pipeline(
                shot_id="shot_pipeline_01",
                frames=frames,
                prompts=prompts,
                output_npz_path=out_file,
                predictor=mock_predictor,
            )

            self.assertEqual(saved_path, out_file)
            self.assertTrue(out_file.is_file())

            archive = ShotMaskArchive.load_npz(out_file)
            self.assertEqual(archive.shot_id, "shot_pipeline_01")
            self.assertEqual(archive.frame_count, 2)
            obj = archive.get_object(1)
            self.assertIsNotNone(obj)
            self.assertEqual(obj.name, "protagonist")
            self.assertEqual(obj.entity_type, "character")

            # Bit-exact mask verification
            for t in range(2):
                mask = archive.get_mask(t, 1)
                self.assertIsNotNone(mask)
                np.testing.assert_array_equal(mask, expected_mask)


if __name__ == "__main__":
    unittest.main()
