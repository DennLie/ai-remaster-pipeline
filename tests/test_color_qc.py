"""Comprehensive tests for Color Consistency Vectorized Engine and CLI."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

import cv2
import numpy as np

from ai_remaster_gui.color_bible_db import (
    ColorReferenceRecord,
    Film,
    QCEvent,
    ShotRecord,
    apply_migrations,
    get_connection,
    list_qc_events,
)
from ai_remaster_gui.color_bible_service import ColorBibleService
from ai_remaster_gui.color_qc import (
    ObjectQCScore,
    QCResult,
    ciede2000,
    compute_cdc,
    compute_optical_flow_chroma_error,
    detect_hue_jumps,
    detect_mask_bleed,
    evaluate_object_color,
    evaluate_shot_qc,
)
from ai_remaster_gui.color_science import srgb_to_lab
from ai_remaster_gui.manifests import read_manifest_details
from ai_remaster_gui.rle_masks import MaskObjectMetadata, ShotMaskArchive
from scripts.qc_color_consistency import main as cli_main

# Published Sharma et al. (2005) 34 CIE L*a*b* benchmark test pairs:
# ((L1, a1, b1), (L2, a2, b2), expected_dE00)
SHARMA_2005_TEST_PAIRS: tuple[tuple[tuple[float, float, float], tuple[float, float, float], float], ...] = (
    ((50.0000, 2.6772, -79.7751), (50.0000, 0.0000, -82.7485), 2.0425),
    ((50.0000, 3.1571, -77.2803), (50.0000, 0.0000, -82.7485), 2.8615),
    ((50.0000, 2.8361, -74.0200), (50.0000, 0.0000, -82.7485), 3.4412),
    ((50.0000, -1.3802, -84.2814), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -1.1848, -84.8006), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, -0.9009, -85.5211), (50.0000, 0.0000, -82.7485), 1.0000),
    ((50.0000, 0.0000, 0.0000), (50.0000, -1.0000, 2.0000), 2.3669),
    ((50.0000, -1.0000, 2.0000), (50.0000, 0.0000, 0.0000), 2.3669),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0009), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0010), 7.1792),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0011), 7.2195),
    ((50.0000, 2.4900, -0.0010), (50.0000, -2.4900, 0.0012), 7.2195),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0009, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0010, -2.4900), 4.8045),
    ((50.0000, -0.0010, 2.4900), (50.0000, 0.0011, -2.4900), 4.7461),
    ((50.0000, 2.5000, 0.0000), (50.0000, 0.0000, -2.5000), 4.3065),
    ((50.0000, 2.5000, 0.0000), (73.0000, 25.0000, -18.0000), 27.1492),
    ((50.0000, 2.5000, 0.0000), (61.0000, -5.0000, 29.0000), 22.8977),
    ((50.0000, 2.5000, 0.0000), (56.0000, -27.0000, -3.0000), 31.9030),
    ((50.0000, 2.5000, 0.0000), (58.0000, 24.0000, 15.0000), 19.4535),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.1736, 0.5854), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2972, 0.0000), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 1.8634, 0.5757), 1.0000),
    ((50.0000, 2.5000, 0.0000), (50.0000, 3.2592, 0.3350), 1.0000),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((63.0109, -31.0961, -5.8663), (62.8187, -29.7946, -4.0864), 1.2630),
    ((61.2901, 3.7196, -5.3901), (61.4292, 2.2480, -4.9620), 1.8731),
    ((35.0831, -44.1164, 3.7933), (35.0232, -40.0716, 1.5901), 1.8645),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
    ((36.4612, 47.8580, 18.3852), (36.2715, 50.5065, 21.2231), 1.4146),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
    ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381),
    ((6.7747, -0.2908, -2.4247), (5.8714, -0.0985, -2.2286), 0.6377),
    ((2.0776, 0.0795, -1.1350), (0.9033, -0.0636, -0.5514), 0.9082),
)


class ColorQCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    # 1. CIE Ground-Truth Benchmark
    def test_ciede2000_sharma_benchmark_individual(self) -> None:
        """Verify individual 1D calls against all 34 Sharma et al. (2005) published pairs within 1e-4."""
        for idx, (p1, p2, expected) in enumerate(SHARMA_2005_TEST_PAIRS, 1):
            calculated = ciede2000(p1, p2)
            self.assertIsInstance(calculated, float)
            self.assertAlmostEqual(
                calculated,
                expected,
                delta=1e-4,
                msg=f"Sharma benchmark pair #{idx} failed: got {calculated}, expected {expected}",
            )

    def test_ciede2000_sharma_benchmark_vectorized(self) -> None:
        """Verify batch vectorized CIEDE2000 against all 34 published pairs simultaneously."""
        lab1 = np.array([p[0] for p in SHARMA_2005_TEST_PAIRS], dtype=np.float64)
        lab2 = np.array([p[1] for p in SHARMA_2005_TEST_PAIRS], dtype=np.float64)
        expected = np.array([p[2] for p in SHARMA_2005_TEST_PAIRS], dtype=np.float64)

        calculated = ciede2000(lab1, lab2)
        self.assertIsInstance(calculated, np.ndarray)
        self.assertEqual(calculated.shape, (len(SHARMA_2005_TEST_PAIRS),))

        diff = np.abs(calculated - expected)
        max_diff = float(np.max(diff))
        self.assertLessEqual(max_diff, 1e-4, f"Vectorized max diff {max_diff} exceeded 1e-4 threshold")

    # 2. Lightness Invariance
    def test_ciede2000_lightness_invariance(self) -> None:
        """Verify normalize_lightness=True isolates chroma drift and eliminates Delta L effect."""
        ref = np.array([50.0, 25.0, 30.0])

        # Test shifting L* by +30 and -30 with identical a*, b*
        test_plus = np.array([80.0, 25.0, 30.0])
        test_minus = np.array([20.0, 25.0, 30.0])

        # When normalize_lightness=False, Delta E must be substantial (> 20.0)
        dE_std_plus = ciede2000(ref, test_plus, normalize_lightness=False)
        dE_std_minus = ciede2000(ref, test_minus, normalize_lightness=False)
        self.assertGreater(dE_std_plus, 20.0)
        self.assertGreater(dE_std_minus, 20.0)

        # When normalize_lightness=True, Delta E must be exactly 0.0
        dE_norm_plus = ciede2000(ref, test_plus, normalize_lightness=True)
        dE_norm_minus = ciede2000(ref, test_minus, normalize_lightness=True)
        self.assertAlmostEqual(dE_norm_plus, 0.0, places=5)
        self.assertAlmostEqual(dE_norm_minus, 0.0, places=5)

    # 3. Injected Hue Defect
    def test_evaluate_shot_qc_injected_hue_defect(self) -> None:
        """Verify injecting a wrong hue on an object mask triggers CHROMA_DELTA_E_FAIL and status 'fail'."""
        h, w = 60, 60
        num_frames = 5

        # Reference object is vivid red: sRGB (220, 20, 20)
        ref_rgb = np.full((10, 10, 3), [220, 20, 20], dtype=np.uint8)
        ref_lab = srgb_to_lab(ref_rgb).reshape(-1, 3)
        ref_mean = [float(x) for x in np.mean(ref_lab, axis=0)]
        ref_stats = {"mean_lab": ref_mean}

        # Rendered shot frames: background is neutral gray (128, 128, 128)
        # Foreground object is injected with green hue: sRGB (20, 180, 20)
        frames: list[np.ndarray] = []
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        mask_2d[15:45, 15:45] = 255  # 30x30 object mask

        for _ in range(num_frames):
            frame = np.full((h, w, 3), 128, dtype=np.uint8)
            frame[mask_2d > 0] = [20, 180, 20]  # Injected green defect
            frames.append(frame)

        archive = ShotMaskArchive(
            shot_id="shot_hue_defect",
            frame_count=num_frames,
            height=h,
            width=w,
            objects=[
                MaskObjectMetadata(
                    obj_id=1,
                    name="hero_jacket",
                    entity_type="object",
                    entity_id="hero_jacket",
                )
            ],
        )
        for t in range(num_frames):
            archive.set_mask(t, 1, mask_2d)

        qc_res = evaluate_shot_qc(
            shot_id="shot_hue_defect",
            frames=frames,
            mask_archive=archive,
            ref_lab_stats_map={"hero_jacket": ref_stats},
            stride=1,
        )

        self.assertEqual(qc_res.qc_status, "fail")
        self.assertIn("CHROMA_DELTA_E_FAIL", qc_res.failure_codes)
        self.assertIn("hero_jacket", qc_res.object_scores)
        self.assertGreater(qc_res.object_scores["hero_jacket"].chroma_delta_e, 8.0)

    # 4. Temporal Metrics
    def test_compute_cdc_flicker(self) -> None:
        """Verify CDC is 1.0 on static frames and drops below threshold on alternating flicker."""
        h, w = 50, 50
        # Static frames
        static_frames = [np.full((h, w, 3), 150, dtype=np.uint8) for _ in range(6)]
        cdc_static = compute_cdc(static_frames)
        self.assertAlmostEqual(cdc_static, 1.0, places=4)

        # Alternating flicker frames (dark/bright)
        flicker_frames = [
            np.full((h, w, 3), 40 if i % 2 == 0 else 220, dtype=np.uint8)
            for i in range(6)
        ]
        cdc_flicker = compute_cdc(flicker_frames)
        self.assertLess(cdc_flicker, 0.50)

    def test_compute_optical_flow_chroma_error(self) -> None:
        """Verify optical flow chroma warping error detects temporal chroma displacement."""
        h, w = 60, 60
        # Frame 0: Gray background with a green patch at (10..30, 10..30)
        f0_rgb = np.full((h, w, 3), 128, dtype=np.uint8)
        f0_rgb[10:30, 10:30] = [0, 200, 0]

        # Frame 1: Same grayscale intensity structure, but patch turns red in chroma
        f1_rgb = np.full((h, w, 3), 128, dtype=np.uint8)
        f1_rgb[10:30, 10:30] = [200, 0, 0]

        g0 = cv2.cvtColor(f0_rgb, cv2.COLOR_RGB2GRAY)
        g1 = cv2.cvtColor(f1_rgb, cv2.COLOR_RGB2GRAY)
        lab0 = srgb_to_lab(f0_rgb)
        lab1 = srgb_to_lab(f1_rgb)

        # Grayscale identical frames with chroma changed -> optical flow produces high chroma residual
        err = compute_optical_flow_chroma_error([g0, g1], [lab0, lab1])
        self.assertGreater(err, 12.0)

    def test_detect_hue_jumps(self) -> None:
        """Verify hue jump detector identifies sharp hue shifts (> 25 degrees)."""
        h, w = 40, 40
        frames_lab = []

        # Frames 0..2 have mean hue around 20 degrees
        for _ in range(3):
            f = np.zeros((h, w, 3), dtype=np.float64)
            f[..., 0] = 50.0
            f[..., 1] = 40.0
            f[..., 2] = 15.0
            frames_lab.append(f)

        # Frame 3 jumps abruptly to hue around 180 degrees (a=-40, b=0)
        f_jump = np.zeros((h, w, 3), dtype=np.float64)
        f_jump[..., 0] = 50.0
        f_jump[..., 1] = -40.0
        f_jump[..., 2] = 0.0
        frames_lab.append(f_jump)

        # Frame 4 stays at new hue
        f_stay = np.zeros((h, w, 3), dtype=np.float64)
        f_stay[..., 0] = 50.0
        f_stay[..., 1] = -40.0
        f_stay[..., 2] = 0.0
        frames_lab.append(f_stay)

        jumps = detect_hue_jumps(frames_lab, threshold_degrees=25.0)
        self.assertEqual(jumps, [3])

    def test_detect_mask_bleed(self) -> None:
        """Verify mask bleed detector flags chroma spill while rejecting clean high-contrast edges."""
        h, w = 80, 80
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[25:55, 25:55] = 255

        # Case A: Clean sharp edge.
        # Neutral background (0, 0 in a*, b*), red object (60, 20 in a*, b*)
        frame_clean = np.zeros((h, w, 3), dtype=np.float64)
        frame_clean[..., 0] = 50.0
        frame_clean[mask > 0, 1] = 60.0
        frame_clean[mask > 0, 2] = 20.0

        bleed_clean = detect_mask_bleed(frame_clean, mask, radius=3, threshold=8.0)
        self.assertAlmostEqual(bleed_clean, 0.0, places=4)

        # Case B: Bleed into immediate outer boundary ring.
        frame_bleed = frame_clean.copy()
        k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        near_ring = (cv2.dilate(mask, k_near) > 0) & (mask == 0)
        # Spill red chroma into near ring
        frame_bleed[near_ring, 1] = 25.0
        frame_bleed[near_ring, 2] = 8.33

        bleed_spill = detect_mask_bleed(frame_bleed, mask, radius=3, threshold=8.0)
        self.assertGreater(bleed_spill, 8.0)

    # 5. CLI & Persistence End-to-End
    def test_cli_and_persistence_e2e(self) -> None:
        """Test scripts/qc_color_consistency.py end-to-end on synthetic shot data."""
        film_id = "film_test_qc"
        shot_id = "shot_001"
        num_frames = 4
        h, w = 64, 64

        # 1. Setup SQLite database with Color Bible schema
        db_path = self.tmp_dir / "color_bible.db"
        conn = get_connection(db_path)
        apply_migrations(conn)

        # Register film and reference
        conn.execute(
            "INSERT INTO films (film_id, title, source_video, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (film_id, "Test Film", "source.mp4", "2026-11-24T00:00:00Z", "2026-11-24T00:00:00Z"),
        )
        ref_stats_json = json.dumps({
            "mean_lab": [55.0, 45.0, 30.0],
            "lab_percentiles": {
                "L": [50.0, 52.0, 55.0, 58.0, 60.0],
                "a": [40.0, 42.0, 45.0, 48.0, 50.0],
                "b": [25.0, 28.0, 30.0, 32.0, 35.0],
            },
        })
        conn.execute(
            """INSERT INTO color_references (
                ref_id, film_id, entity_type, entity_id, source_keyframe_path,
                lab_stats_json, rgb_palette_json, provenance, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "ref_coat",
                film_id,
                "object",
                "red_coat",
                "ref.png",
                ref_stats_json,
                "[]",
                "manual",
                "2026-11-24T00:00:00Z",
            ),
        )
        conn.commit()
        conn.close()

        # 2. Setup synthetic rendered frames matching reference
        frames_dir = self.tmp_dir / "frames" / shot_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        mask_2d = np.zeros((h, w), dtype=np.uint8)
        mask_2d[20:50, 20:50] = 255

        # Red coat color corresponding to roughly L=55, a=45, b=30
        for i in range(num_frames):
            frame = np.full((h, w, 3), 100, dtype=np.uint8)
            frame[mask_2d > 0] = [180, 50, 40]
            cv2.imwrite(str(frames_dir / f"frame_{i:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 3. Setup mask archive
        masks_dir = self.tmp_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        archive = ShotMaskArchive(
            shot_id=shot_id,
            frame_count=num_frames,
            height=h,
            width=w,
            objects=[
                MaskObjectMetadata(
                    obj_id=1,
                    name="red_coat",
                    entity_type="object",
                    entity_id="red_coat",
                )
            ],
        )
        for t in range(num_frames):
            archive.set_mask(t, 1, mask_2d)
        archive.save_npz(masks_dir / f"{shot_id}.masks.npz")

        # 4. Setup shot manifest CSV
        manifest_path = self.tmp_dir / "manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["shot_id", "shot_index", "start_frame", "end_frame", "frames_path"])
            writer.writerow([shot_id, "0", "0", str(num_frames), str(frames_dir)])

        output_json = self.tmp_dir / "qc_report.json"

        # 5. Run CLI
        cli_args = [
            "--manifest", str(manifest_path),
            "--frames-dir", str(frames_dir),
            "--masks-dir", str(masks_dir),
            "--color-bible-db", str(db_path),
            "--film-id", film_id,
            "--stride", "1",
            "--output-json", str(output_json),
            "--update-manifest",
            "--record-db",
        ]
        ret_code = cli_main(cli_args)
        self.assertEqual(ret_code, 0)

        # 6. Verify manifest CSV was updated with columns
        _src, fieldnames, rows = read_manifest_details(manifest_path)
        self.assertIn("qc_status", fieldnames)
        self.assertIn("qc_delta_e", fieldnames)
        self.assertIn("qc_cdc", fieldnames)
        self.assertIn("qc_failure_codes", fieldnames)
        self.assertEqual(rows[0]["qc_status"], "pass")
        self.assertNotEqual(rows[0]["qc_cdc"], "")

        # 7. Verify JSON report was created
        self.assertTrue(output_json.is_file())
        with output_json.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["film_id"], film_id)
        self.assertEqual(data["summary"]["passed"], 1)
        self.assertEqual(data["summary"]["failed"], 0)

        # 8. Verify SQLite qc_events table contains record
        conn = get_connection(db_path)
        events = list_qc_events(conn, film_id=film_id)
        conn.close()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].shot_id, shot_id)
        self.assertEqual(events[0].qc_status, "pass")

    def test_cli_failure_exit_code(self) -> None:
        """Verify CLI returns exit code 1 when a shot fails QC."""
        film_id = "film_fail_test"
        shot_id = "shot_failing"
        h, w = 40, 40

        frames_dir = self.tmp_dir / "failing_frames" / shot_id
        frames_dir.mkdir(parents=True, exist_ok=True)
        # 4 frames with huge temporal flicker to cause CDC_FAIL
        for i in range(4):
            val = 20 if i % 2 == 0 else 240
            frame = np.full((h, w, 3), val, dtype=np.uint8)
            cv2.imwrite(str(frames_dir / f"f_{i:04d}.png"), frame)

        manifest_path = self.tmp_dir / "manifest_failing.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["shot_id", "frames_path"])
            writer.writerow([shot_id, str(frames_dir)])

        ret = cli_main([
            "--manifest", str(manifest_path),
            "--film-id", film_id,
            "--update-manifest",
        ])
        self.assertEqual(ret, 1)

        _src, _fields, rows = read_manifest_details(manifest_path)
        self.assertEqual(rows[0]["qc_status"], "fail")
        self.assertIn("CDC_FAIL", rows[0]["qc_failure_codes"])

    def test_cli_closes_color_bible_service(self) -> None:
        """Verify CLI ensures service.close() is called in finally block."""
        manifest_path = self.tmp_dir / "manifest_close_test.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["shot_id", "frames_path"])
            writer.writerow(["shot_0", "nonexistent"])

        db_path = self.tmp_dir / "test_cb.db"

        from unittest import mock
        with mock.patch("scripts.qc_color_consistency.ColorBibleService") as mock_service_cls:
            mock_inst = mock.MagicMock()
            mock_service_cls.return_value = mock_inst
            cli_main([
                "--manifest", str(manifest_path),
                "--color-bible-db", str(db_path),
                "--record-db",
            ])
            mock_inst.close.assert_called_once()

    def test_ciede2000_broadcasting(self) -> None:
        """Verify CIEDE2000 supports broadcasting across arbitrary dimensions."""
        ref = np.array([50.0, 10.0, 20.0])  # (3,)
        targets = np.zeros((4, 5, 3), dtype=np.float64)  # (4, 5, 3)
        targets[..., 0] = 60.0
        targets[..., 1] = 15.0
        targets[..., 2] = 25.0

        dE = ciede2000(ref, targets)
        self.assertEqual(dE.shape, (4, 5))
        self.assertTrue(np.all(dE > 0.0))

    def test_evaluate_shot_qc_warn_threshold(self) -> None:
        """Verify a moderate delta_e in range (5.0, 8.0] triggers CHROMA_DELTA_E_WARN and status 'warn'."""
        h, w = 40, 40
        num_frames = 3

        ref_lab = [50.0, 0.0, 0.0]
        ref_stats = {"mean_lab": ref_lab}

        # Create frames where object chroma delta E is ~6.5 (between 5.0 and 8.0)
        # Using a target Lab with slight chroma shift
        target_rgb = [150, 120, 120]  # Slight warmth
        mask_2d = np.zeros((h, w), dtype=np.uint8)
        mask_2d[10:30, 10:30] = 255

        frames = []
        for _ in range(num_frames):
            f = np.full((h, w, 3), 128, dtype=np.uint8)
            f[mask_2d > 0] = target_rgb
            frames.append(f)

        archive = ShotMaskArchive(
            shot_id="shot_warn",
            frame_count=num_frames,
            height=h,
            width=w,
            objects=[MaskObjectMetadata(obj_id=1, name="subtle_obj", entity_type="object", entity_id="subtle_obj")],
        )
        for t in range(num_frames):
            archive.set_mask(t, 1, mask_2d)

        # Let's test with custom thresholds to explicitly test warn boundary
        qc_res = evaluate_shot_qc(
            shot_id="shot_warn",
            frames=frames,
            mask_archive=archive,
            ref_lab_stats_map={"subtle_obj": ref_stats},
            thresholds={"chroma_delta_e_pass": 1.0, "chroma_delta_e_warn": 50.0},
        )
        self.assertEqual(qc_res.qc_status, "warn")
        self.assertIn("CHROMA_DELTA_E_WARN", qc_res.failure_codes)


if __name__ == "__main__":
    unittest.main()
