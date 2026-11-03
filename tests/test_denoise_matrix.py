"""Regression coverage for the Remove Noise method matrix and contact-sheet export."""
from __future__ import annotations

import csv
import io
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from ai_remaster_gui import denoise_matrix as matrix


class DenoiseMatrixTests(unittest.TestCase):
    def test_frame_selection_is_one_based_and_capped(self):
        self.assertEqual(matrix.parse_frames("", 11), [1, 5, 9])
        self.assertEqual(matrix.parse_frames("1, 4, 11", 11), [0, 3, 10])
        self.assertEqual(matrix.parse_frames("", 1), [0])
        for invalid in ("0", "12", "1,1", "1,2,3,4,5,6", "1;2;3;4;5;6", "abc"):
            with self.subTest(text=invalid):
                with self.assertRaises(ValueError):
                    matrix.parse_frames(invalid, 11)

    def test_variant_matrix_rejects_invalid_methods_and_duplicates(self):
        self.assertEqual(len(matrix.VARIANTS), 9)
        matrix.validate_variants([{"method": a, "preset": b} for a, b in matrix.VARIANTS])
        for bad in (
            None, {}, "hqdn3d", [], [None], [{"method": [], "preset": "gentle"}],
            [{"method": "off", "preset": "gentle"}],
            [{"method": "nlmeans", "preset": "extreme"}],
            [{"method": "nlmeans", "preset": "gentle"}] * 2,
        ):
            with self.assertRaises(ValueError):
                matrix.validate_variants(bad)

    def test_selected_batch_uses_server_paths_and_does_not_adopt_pipeline_settings(self):
        from ai_remaster_gui import server
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mkv"
            source.write_bytes(b"source")
            app = server.PipelineApp.__new__(server.PipelineApp)
            app.settings = {"global": {"source": str(source)}, "denoise": {"method": "off", "preset": "gentle"}}
            before = json.dumps(app.settings, sort_keys=True)
            app.quitting = False
            app.process = None
            app.lock = threading.Lock()
            app.denoise_enabled = Mock(return_value=True)
            app.ensure_pipeline_source = Mock()
            app.denoise_matrix_rows = Mock(return_value=[
                {"method": method, "preset": preset, "output": f"server/{method}_{preset}.mkv"}
                for method, preset in matrix.VARIANTS
            ])
            app._start_denoise_matrix = Mock(return_value=(True, "started"))
            requested = [{"method": "hqdn3d", "preset": "gentle", "output": "untrusted/overwrite.mkv"},
                         {"method": "bm3d", "preset": "strong"}]
            self.assertTrue(app.run_denoise_batch(requested)[0])
            command = app._start_denoise_matrix.call_args.args[1]
            rows = json.loads(command[command.index("--variants") + 1])
            self.assertEqual(rows, [
                {"method": "hqdn3d", "preset": "gentle", "output": "server/hqdn3d_gentle.mkv"},
                {"method": "bm3d", "preset": "strong", "output": "server/bm3d_strong.mkv"},
            ])
            self.assertEqual(json.dumps(app.settings, sort_keys=True), before)
            app._start_denoise_matrix.reset_mock()
            self.assertFalse(app.run_denoise_batch([{"method": "off", "preset": "gentle"}])[0])
            app._start_denoise_matrix.assert_not_called()
            app.ensure_pipeline_source.reset_mock()
            app.process = Mock()
            app.process.poll.return_value = None
            self.assertFalse(app.run_denoise_batch(requested)[0])
            app.ensure_pipeline_source.assert_not_called()

    def test_batch_cli_accepts_a_single_validated_variant(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mkv"
            source.write_bytes(b"source")
            rows = [{"method": "bm3d", "preset": "strong", "output": "server/result.mkv"}]
            args = ["denoise_matrix", "batch", "--source", str(source), "--profile", "high", "--variants", json.dumps(rows)]
            with patch("sys.argv", args), patch.object(matrix, "batch") as batch:
                matrix.main()
            batch.assert_called_once_with(source, rows, "high")

    def test_matrix_state_exposes_matching_outputs_and_source_revision(self):
        from ai_remaster_gui import server
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.mkv"
            source.write_bytes(b"source")
            app = server.PipelineApp.__new__(server.PipelineApp)
            app.settings = {"global": {"source": str(source)}}
            app.denoise_grid_request = None
            app.running_stage = ""
            rows = [{"method": "bm3d", "preset": "gentle", "output": "server/ready.mkv", "revision": "123-456"}]
            app.denoise_matrix_ready = Mock(return_value=rows)
            first = app.denoise_matrix_state()
            self.assertEqual(first["variants"], rows)
            self.assertEqual(first["ready"], 1)
            self.assertTrue(first["source_revision"])
            source.write_bytes(b"changed source")
            self.assertNotEqual(first["source_revision"], app.denoise_matrix_state()["source_revision"])

    def test_directional_difference_preserves_polarity_and_values(self):
        original = Image.new("RGB", (2, 1))
        original.putdata([(100, 100, 100), (100, 100, 100)])
        result = Image.new("RGB", (2, 1))
        result.putdata([(110, 110, 110), (90, 90, 90)])
        image, metrics = matrix._luma_difference(original, result)
        bright, dark = list(image.getdata())
        self.assertGreater(bright[1], bright[0])
        self.assertGreater(bright[2], bright[0])
        self.assertGreater(dark[0], dark[1])
        self.assertAlmostEqual(metrics["mean_absolute_luminance"], 10.0)
        self.assertEqual(metrics["pixel_change_percent"], 100.0)
        unchanged, zero = matrix._luma_difference(original, original)
        self.assertEqual(unchanged.getpixel((0, 0)), (15, 23, 31))
        self.assertEqual(zero["mean_absolute_luminance"], 0)

    def test_grid_zip_contains_overview_individual_sheets_fullres_diffs_and_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            source = base / "source.mkv"
            source.write_bytes(b"source")
            rendered = base / "result.mkv"
            rendered.write_bytes(b"result")
            target = base / "grid.zip"
            source_image = Image.new("RGB", (12, 8), (100, 100, 100))
            result_image = Image.new("RGB", (12, 8), (102, 102, 102))
            rows = [{"method": "hqdn3d", "preset": "gentle", "output": str(rendered)}]
            def image_for(path, frame):
                return source_image.copy() if path == source else result_image.copy()
            with patch.object(matrix, "_metadata", return_value={"frames": 8, "fps": 24.0}), \
                 patch.object(matrix, "_validate_pair", return_value=8), \
                 patch.object(matrix, "_image", side_effect=image_for):
                matrix.grid(source, rows, [0, 7], "high", target)
            self.assertTrue(target.is_file())
            self.assertFalse(target.with_name("grid.partial.zip").exists())
            with zipfile.ZipFile(target) as archive:
                names = set(archive.namelist())
                self.assertIn("overview_all_frames.png", names)
                self.assertIn("frames/frame_000001_grid.png", names)
                self.assertIn("frames/frame_000008_grid.png", names)
                self.assertIn("differences/frame_000001_hqdn3d_gentle.png", names)
                self.assertIn("measurement.csv", names)
                with Image.open(io.BytesIO(archive.read("overview_all_frames.png"))) as image:
                    self.assertEqual(image.width, 2 * (2 * matrix.TILE_W + 3 * matrix.PAD))
                with Image.open(io.BytesIO(archive.read("differences/frame_000001_hqdn3d_gentle.png"))) as image:
                    self.assertEqual(image.size, (12, 8))
                rows_csv = list(csv.DictReader(io.StringIO(archive.read("measurement.csv").decode("utf-8-sig"))))
                self.assertEqual(len(rows_csv), 2)
                self.assertEqual(rows_csv[0]["method"], "hqdn3d")
                self.assertIn("not prove denoising", archive.read("README.txt").decode("utf-8").lower())

    def test_export_key_changes_when_inputs_change(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "clip.mkv"
            result = Path(folder) / "render.mkv"
            source.write_bytes(b"a")
            result.write_bytes(b"b")
            rows = [{"method": "bm3d", "preset": "strong", "output": str(result)}]
            a = matrix.export_identity(source, rows, [0], "high")
            self.assertEqual(a, matrix.export_identity(source, rows, [0], "high"))
            self.assertNotEqual(a, matrix.export_identity(source, rows, [1], "high"))
            result.write_bytes(b"modified")
            self.assertNotEqual(a, matrix.export_identity(source, rows, [0], "high"))


if __name__ == "__main__":
    unittest.main()
