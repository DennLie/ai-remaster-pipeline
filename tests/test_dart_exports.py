"""CPU-only DART export evaluator contracts. No DART binaries or model required."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assess_dart_exports as assessment  # noqa: E402


def fixture(root: Path, *, n: int = 3) -> tuple[Path, Path]:
    original = root / "source"
    dart = root / "dart"
    for folder in (original, dart / "frames", dart / "masks"):
        folder.mkdir(parents=True)
    for i in range(n):
        frame = np.full((64, 96, 3), 110, dtype=np.uint8)
        frame[12:38, 55:62] = 190
        frame[30:34, 21:25] = 248  # Synthetic transient dirt
        restored = frame.copy()
        restored[30:34, 21:25] = 110
        restored[10:12, 10:12] = 5  # Unwanted change outside defect mask
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[30:34, 21:25] = 220
        name = f"frame_{i:06d}.png"
        for target, image in [
            (original / name, frame),
            (dart / "frames" / name, restored),
            (dart / "masks" / name, mask),
        ]:
            if not cv2.imwrite(str(target), image):
                raise RuntimeError(f"Could not create fixture {target}")
    return original, dart


class DartExportAnalysisTests(unittest.TestCase):
    def test_only_masked_pixels_changed_in_preservation_composite(self):
        original = np.full((24, 32, 3), 126, np.uint8)
        restored = np.full_like(original, 210)
        mask = np.zeros((24, 32), dtype=np.uint8)
        mask[7:11, 8:13] = 175
        mask[3:7, 8:13] = 159
        composite = assessment.masked_composite(original, restored, mask, 160)
        np.testing.assert_array_equal(composite[mask < 160], original[mask < 160])
        np.testing.assert_array_equal(composite[mask >= 160], restored[mask >= 160])
        self.assertEqual(np.count_nonzero(np.any(composite != original, axis=2)), 20)

    def test_stored_report_identifies_restoration_changes_outside_mask(self):
        with tempfile.TemporaryDirectory(prefix="dart-eval-test-") as td:
            root = Path(td)
            src, dart = fixture(root)
            out = root / "result"
            report = assessment.evaluate(src, dart, out, threshold=160)
            self.assertEqual(report["frames"], 3)
            self.assertEqual(report["width"], 96)
            self.assertEqual(report["height"], 64)
            self.assertGreater(report["changed_unmasked_fraction"], 0)
            self.assertGreater(report["changed_masked_fraction"], 0)
            self.assertFalse(report["commercially_cleared"])
            self.assertFalse(report["masked_composite_written"])
            self.assertTrue((out / "dart_mask_restoration_contact_sheet.jpg").is_file())
            saved = json.loads((out / "dart_restoration_diagnostics.json").read_text())
            self.assertEqual(saved["frames"], 3)

    def test_research_composite_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory(prefix="dart-eval-test-") as td:
            root = Path(td)
            src, dart = fixture(root)
            with self.assertRaisesRegex(ValueError, "research-only"):
                assessment.evaluate(src, dart, root / "out", write_masked_composite=True)
            report = assessment.evaluate(src, dart, root / "out", write_masked_composite=True, research_only=True)
            self.assertTrue(report["masked_composite_written"])
            image = cv2.imread(str(root / "out" / "masked_composite_frames" / "frame_000000.png"))
            original = cv2.imread(str(src / "frame_000000.png"))
            self.assertEqual(image.shape, original.shape)
            np.testing.assert_array_equal(image[10:12, 10:12], original[10:12, 10:12])
            self.assertEqual(int(image[31, 22, 0]), 110)

    def test_missing_mask_is_failure_not_silent_skip(self):
        with tempfile.TemporaryDirectory(prefix="dart-eval-test-") as td:
            root = Path(td)
            src, dart = fixture(root)
            (dart / "masks" / "frame_000001.png").unlink()
            with self.assertRaises(FileNotFoundError):
                assessment.evaluate(src, dart, root / "result")

    def test_mismatched_masks_are_rejected_without_resizing(self):
        with tempfile.TemporaryDirectory(prefix="dart-eval-test-") as td:
            root = Path(td)
            src, dart = fixture(root)
            cv2.imwrite(str(dart / "masks" / "frame_000001.png"), np.zeros((32, 48), np.uint8))
            with self.assertRaisesRegex(ValueError, "source geometry"):
                assessment.evaluate(src, dart, root / "result")

    def test_extra_stale_output_frames_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="dart-eval-test-") as td:
            root = Path(td)
            src, dart = fixture(root)
            cv2.imwrite(str(dart / "frames" / "stale.png"), np.zeros((64, 96, 3), np.uint8))
            with self.assertRaisesRegex(ValueError, "unexpected frames"):
                assessment.evaluate(src, dart, root / "result")

    def test_video_input_names_match_official_dart_naming(self):
        with tempfile.TemporaryDirectory(prefix="dart-eval-test-") as td:
            root = Path(td)
            path = root / "source.avi"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12, (96, 64))
            self.assertTrue(writer.isOpened())
            for i in range(4):
                writer.write(np.full((64, 96, 3), 100 + i, np.uint8))
            writer.release()
            names = [name for name, _ in assessment.source_frames(path, max_frames=3)]
            self.assertEqual(names, ["frame_000000", "frame_000001", "frame_000002"])

    def test_reject_unapproved_mask_threshold_and_size(self):
        orig = np.zeros((10, 14, 3), np.uint8)
        rest = orig.copy()
        small = np.zeros((4, 14), np.uint8)
        with self.assertRaisesRegex(ValueError, "geometries"):
            assessment.masked_composite(orig, rest, small, 160)
        with self.assertRaisesRegex(ValueError, "threshold"):
            assessment.masked_composite(orig, rest, np.zeros((10, 14), np.uint8), 256)


if __name__ == "__main__":
    unittest.main()
