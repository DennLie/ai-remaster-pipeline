"""Research mask detector: controlled synthetic true defects, static-edge controls, source integrity."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import restoration_evaluation as ev
import temporal_ridge_scratch_research as ridge


class TemporalRidgeResearchTests(unittest.TestCase):
    def test_static_architecture_edge_is_negative_control(self):
        h, w = 96, 160
        frame = np.full((h, w, 3), 100, np.uint8)
        frame[:, 35:38] = 255   # vertical window-like ridge in every frame
        frame[:, 80:] = 215     # real architectural step edge
        movie = [frame.copy() for _ in range(15)]
        outputs = ridge.detect_sequence(movie, sensitivity=0.65)
        self.assertEqual(len(outputs), 15)
        self.assertTrue(all(np.count_nonzero(x) == 0 for x in outputs))

    def test_fixed_temporal_sequence_and_fail_closed_coverage(self):
        frame = np.full((96, 160, 3), 120, np.uint8)
        frame[:, 15:18] = 245
        seq = []
        known = []
        for index in range(24):
            output, ground_truth = ev.corrupt(
                frame, kind="vertical_scratches", strength="gentle",
                index=index, seed=73831)
            seq.append(cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            known.append(ground_truth)
        first = ridge.detect_sequence(seq)
        second = ridge.detect_sequence(seq)
        self.assertEqual(len(first), 24)
        for p, q, gt in zip(first, second, known):
            np.testing.assert_array_equal(p, q)
            self.assertEqual(p.dtype, np.uint8)
            self.assertEqual(p.shape, gt.shape)
            self.assertLessEqual(np.mean(p > 0), 0.03 + 1e-9)
        with self.assertRaises(ValueError):
            ridge.detect_sequence(seq, sensitivity=float("nan"))
        with self.assertRaises(ValueError):
            ridge.detect_sequence(seq[:2])

    def test_full_ffv1_source_integrity_and_no_oracle_access(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-ridge-") as td:
            root = Path(td)
            source = root / "input.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "testsrc2=size=128x96:rate=12:duration=1",
                 "-c:v", "ffv1", str(source)],
                check=True, capture_output=True, timeout=40)
            stage = root / "experiment"
            manifest = ev.synthesize(
                source=source, output_dir=stage,
                film="synthetic_unit", scene="buildings",
                kind="vertical_scratches", preset="gentle",
                seed=2026, ffmpeg=ffmpeg, ffprobe=ffprobe)
            original_checksum = ev.digest(source)
            report = ridge.benchmark(
                manifest=stage / "manifest.json", output_dir=root / "research",
                split="development", sensitivity=0.65,
                ffmpeg=ffmpeg, ffprobe=ffprobe)
            self.assertEqual(report["frames"], 12)
            self.assertEqual(report["source_damaged_sha256"], manifest["degraded_sha256"])
            self.assertEqual(ev.digest(source), original_checksum)
            self.assertEqual(set(report["methods"]), {
                "existing_production", "research_temporal_ridge"})
            for label, vals in report["methods"].items():
                m = vals["mask_metrics"]
                self.assertEqual(sum(m[k] for k in ("tp", "fp", "fn", "tn")), 128 * 96 * 12)
                self.assertGreaterEqual(m["pixel_f1"], 0)
                self.assertLessEqual(m["pixel_f1"], 1)
                self.assertGreaterEqual(vals["negative_control_coverage_mean"], 0)
                self.assertTrue((root / "research" / f"{label}_000000.png").exists())
                self.assertIsNotNone(vals["telea_known_scratch_mse"])
            with self.assertRaises(FileExistsError):
                ridge.benchmark(
                    manifest=stage / "manifest.json", output_dir=root / "research",
                    split="holdout", ffmpeg=ffmpeg, ffprobe=ffprobe)
            damaged = stage / "degraded.mkv"
            damaged.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "integrity"):
                ridge.benchmark(
                    manifest=stage / "manifest.json",
                    output_dir=root / "modified", split="development",
                    ffmpeg=ffmpeg, ffprobe=ffprobe)


if __name__ == "__main__":
    unittest.main()
