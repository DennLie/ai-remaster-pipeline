"""Integration and contract tests for candidate film defect detectors."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_defect_masks as bench  # noqa: E402
import candidate_detector_eval as cde  # noqa: E402


class CandidateDetectorsTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(101)
        self.clean_frames = [
            rng.integers(116, 124, size=(64, 80, 3), dtype=np.uint8)
            for _ in range(6)
        ]

    def test_preprocess_frames_normalizes_grayscale_and_bgr_identically(self) -> None:
        gray_in = [np.full((32, 48), 120, dtype=np.uint8) for _ in range(3)]
        bgr_in = [np.full((32, 48, 3), 120, dtype=np.uint8) for _ in range(3)]

        seq_g = cde.preprocess_frames(gray_in)
        seq_b = cde.preprocess_frames(bgr_in)

        self.assertEqual(seq_g.height, 32)
        self.assertEqual(seq_g.width, 48)
        for g1, g2, b1, b2 in zip(
            seq_g.gray_frames, seq_b.gray_frames, seq_g.bgr_frames, seq_b.bgr_frames
        ):
            self.assertTrue(np.array_equal(g1, g2))
            self.assertTrue(np.array_equal(b1, b2))

        with self.assertRaises(ValueError):
            cde.preprocess_frames([])
        with self.assertRaises(ValueError):
            cde.preprocess_frames([np.zeros((32, 32), dtype=np.float32)])
        with self.assertRaises(ValueError):
            cde.preprocess_frames([
                np.zeros((32, 32), dtype=np.uint8),
                np.zeros((32, 40), dtype=np.uint8),
            ])

    def test_all_candidates_satisfy_unified_contract_on_scratches_and_dust(self) -> None:
        scratch_frames, scratch_gt = bench.inject_scratches(
            self.clean_frames,
            bench.ScratchConfig(width=2, opacity=0.9, duration=4, drift_px=3.0, seed=11),
        )
        dust_frames, dust_gt = bench.inject_dust(
            self.clean_frames,
            bench.DustConfig(clusters_per_frame=3, contrast=0.9, seed=12),
        )

        for name in cde.CANDIDATE_NAMES:
            detector = cde.CANDIDATE_REGISTRY[name]
            with self.subTest(candidate=name):
                res_s = detector(scratch_frames, sensitivity=0.65)
                res_d = detector(dust_frames, sensitivity=0.65)

                for res in (res_s, res_d):
                    self.assertIn("scratch", res)
                    self.assertIn("dust", res)
                    self.assertEqual(len(res["scratch"]), len(self.clean_frames))
                    self.assertEqual(len(res["dust"]), len(self.clean_frames))
                    for m in res["scratch"] + res["dust"]:
                        self.assertEqual(m.shape, (64, 80))
                        self.assertEqual(m.dtype, np.uint8)
                        self.assertTrue(set(np.unique(m).tolist()).issubset({0, 255}))

                # Ridge is a specialized scratch detector: its dust channel must be all zeros
                if name == "ridge":
                    self.assertTrue(all(np.count_nonzero(m) == 0 for m in res_d["dust"]))
                    self.assertGreater(sum(np.count_nonzero(m) for m in res_s["scratch"]), 0)
                else:
                    # Temporal consensus detectors should detect transient dust
                    self.assertGreater(sum(np.count_nonzero(m) for m in res_d["dust"]), 0)

    def test_dart_harness_reads_external_masks_when_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mask_dir = Path(tmp) / "masks"
            mask_dir.mkdir(parents=True)
            for idx in range(len(self.clean_frames)):
                m = np.zeros((64, 80), dtype=np.uint8)
                # Add a vertical scratch and a dust spot above threshold 160
                m[10:45, 20:22] = 200
                m[30:33, 50:53] = 220
                cv2.imwrite(str(mask_dir / f"{idx + 1:06d}.png"), m)

            out = cde.detect_dart_harness(
                self.clean_frames,
                external_mask_dir=Path(tmp),
                threshold=160,
            )
            self.assertEqual(len(out["scratch"]), len(self.clean_frames))
            self.assertEqual(len(out["dust"]), len(self.clean_frames))
            self.assertGreater(np.count_nonzero(out["scratch"][0]), 0)
            self.assertGreater(np.count_nonzero(out["dust"][0]), 0)

    def test_error_map_and_six_panel_contact_sheet_generation(self) -> None:
        gt = np.zeros((20, 20), dtype=np.uint8)
        pred = np.zeros((20, 20), dtype=np.uint8)
        gt[2, 2] = 255
        pred[2, 2] = 255  # TP -> Green (0, 255, 0)
        pred[5, 5] = 255  # FP -> Red (0, 0, 255)
        gt[8, 8] = 255    # FN -> Blue (255, 0, 0)
        gt[10, 10] = 127  # Ignored -> (64, 64, 64)

        err_map = cde.build_error_map(gt, pred)
        self.assertEqual(tuple(int(x) for x in err_map[2, 2]), (0, 255, 0))
        self.assertEqual(tuple(int(x) for x in err_map[5, 5]), (0, 0, 255))
        self.assertEqual(tuple(int(x) for x in err_map[8, 8]), (255, 0, 0))
        self.assertEqual(tuple(int(x) for x in err_map[10, 10]), (64, 64, 64))
        self.assertEqual(tuple(int(x) for x in err_map[0, 0]), (0, 0, 0))

        with tempfile.TemporaryDirectory() as tmp:
            sheet_path = Path(tmp) / "sheet.png"
            masks = [gt.copy() for _ in range(len(self.clean_frames))]
            frames_20 = [f[:20, :20] for f in self.clean_frames]
            sheet = cde.generate_contact_sheet(
                frames_20,
                masks,
                masks,
                masks,
                masks,
                frame_index=2,
                method_name="dis_flow",
                output_path=sheet_path,
                panel_width=120,
            )
            self.assertTrue(sheet_path.is_file())
            # 6 panels horizontally, each width 120 -> total width 720
            self.assertEqual(sheet.shape[1], 120 * 6)
            self.assertEqual(sheet.shape[2], 3)

    def test_temporal_metrics_and_profiling(self) -> None:
        gt = [np.zeros((32, 32), dtype=np.uint8) for _ in range(5)]
        for t in (1, 2, 3):
            gt[t][5:20, 10] = 255

        # Perfect prediction: 0 flicker, 1.0 continuity
        perfect_tm = cde.compute_temporal_metrics(gt, gt)
        self.assertEqual(perfect_tm["mask_flicker_rate"], 0.0)
        self.assertEqual(perfect_tm["detection_continuity"], 1.0)
        self.assertEqual(perfect_tm["missed_persistent_rate"], 0.0)

        # Flickering FP on frame 2 at (0, 0) and missed scratch on frame 2
        flicker_pred = [m.copy() for m in gt]
        flicker_pred[2][5:20, 10] = 0
        flicker_pred[2][0, 0] = 255
        noisy_tm = cde.compute_temporal_metrics(gt, flicker_pred)
        self.assertGreater(noisy_tm["mask_flicker_rate"], 0.0)
        self.assertAlmostEqual(noisy_tm["detection_continuity"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(noisy_tm["missed_persistent_rate"], 1.0 / 3.0, places=5)

        # Verify standard resolution constants and profiling output structure
        self.assertEqual(cde.STANDARD_RESOLUTIONS["720p"], (1280, 720))
        self.assertEqual(cde.STANDARD_RESOLUTIONS["1080p"], (1920, 1080))
        self.assertEqual(cde.STANDARD_RESOLUTIONS["4K"], (3840, 2160))

        prof = cde.profile_detector_performance(
            cde.detect_consensus_5f,
            resolutions={"tiny": (64, 48)},
            num_frames=3,
        )
        self.assertIn("tiny", prof)
        self.assertGreater(prof["tiny"]["ms_per_frame"], 0.0)
        self.assertGreater(prof["tiny"]["peak_ram_mb"], 0.0)

    def test_cli_end_to_end_evaluation_and_markdown_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "eval_out"
            md_path = Path(tmp) / "comparison.md"
            rc = cde.main([
                "--candidates", "farneback_3f", "dis_flow", "ridge", "consensus_5f", "dart_harness",
                "--num-frames", "5",
                "--height", "48",
                "--width", "64",
                "--output-dir", str(out_dir),
                "--markdown-out", str(md_path),
            ])
            self.assertEqual(rc, 0)
            report_file = out_dir / "candidate_evaluation_report.json"
            self.assertTrue(report_file.is_file())
            self.assertTrue(md_path.is_file())

            payload = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["candidates"].keys()), set(cde.CANDIDATE_NAMES))
            for name in cde.CANDIDATE_NAMES:
                sheet_file = out_dir / "contact_sheets" / f"contact_sheet_{name}.png"
                self.assertTrue(sheet_file.is_file())

    def test_validation_and_edge_cases_across_candidates_and_helpers(self) -> None:
        # 4D numpy array input to preprocess_frames
        arr_4d = np.full((3, 24, 32, 3), 120, dtype=np.uint8)
        seq_4d = cde.preprocess_frames(arr_4d)
        self.assertEqual(len(seq_4d.bgr_frames), 3)
        self.assertEqual((seq_4d.height, seq_4d.width), (24, 32))

        with self.assertRaises(ValueError):
            cde.preprocess_frames(np.zeros((24, 32), dtype=np.uint8))  # 2D array invalid

        # Short or tiny sequences return all-zero masks cleanly for all candidates
        tiny_frames = [np.full((12, 12, 3), 100, dtype=np.uint8) for _ in range(2)]
        for name in cde.CANDIDATE_NAMES:
            res = cde.CANDIDATE_REGISTRY[name](tiny_frames)
            self.assertEqual(len(res["scratch"]), 2)
            self.assertEqual(len(res["dust"]), 2)
            self.assertTrue(all(np.count_nonzero(m) == 0 for m in res["scratch"] + res["dust"]))

        # Sensitivity bounds validation
        for fn in (cde.detect_farneback_3f, cde.detect_dis_flow, cde.detect_multiscale_ridge, cde.detect_consensus_5f):
            with self.assertRaises(ValueError):
                fn(self.clean_frames, sensitivity=1.5)

        # DART harness error cases
        with self.assertRaises(ValueError):
            cde.detect_dart_harness(self.clean_frames, threshold=0)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                cde.detect_dart_harness(self.clean_frames, external_mask_dir=Path(tmp))
            with self.assertRaises(ValueError):
                cde.detect_dart_harness(
                    self.clean_frames,
                    external_mask_dir=Path(tmp),
                    frame_names=["000001.png"],
                )

        # Profile detector performance validation
        with self.assertRaises(ValueError):
            cde.profile_detector_performance(cde.detect_consensus_5f, num_frames=2)

        # Empirical comparison document exists and covers all candidates and standard resolutions
        doc_path = Path(__file__).resolve().parents[1] / "docs" / "defect-detector-empirical-comparison.md"
        self.assertTrue(doc_path.is_file())
        doc_text = doc_path.read_text(encoding="utf-8")
        for name in cde.CANDIDATE_NAMES:
            self.assertIn(name, doc_text)
        for res_label in cde.STANDARD_RESOLUTIONS:
            self.assertIn(res_label, doc_text)


if __name__ == "__main__":
    unittest.main()
