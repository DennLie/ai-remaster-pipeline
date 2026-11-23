from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ai_remaster_gui.rle_masks import (
    VALID_ENTITY_TYPES,
    MaskObjectMetadata,
    ShotMaskArchive,
    decode_mask_rle,
    encode_mask_rle,
    load_shot_masks_npz,
    save_shot_masks_npz,
)
import scripts.rle_masks as rle_cli


class TestRLEMaskCodec(unittest.TestCase):
    """Unit tests for low-level alternating Run-Length Encoding codec."""

    def test_empty_mask_all_zeros(self) -> None:
        """Test encoding and decoding of an all-zero mask across different resolutions."""
        for shape in [(1, 1), (10, 10), (720, 1280), (1080, 1920)]:
            h, w = shape
            mask = np.zeros((h, w), dtype=np.uint8)
            encoded = encode_mask_rle(mask)
            self.assertEqual(encoded.dtype, np.int32)
            self.assertEqual(len(encoded), 1)
            self.assertEqual(encoded[0], h * w)

            decoded = decode_mask_rle(encoded, h, w)
            self.assertEqual(decoded.dtype, np.uint8)
            self.assertEqual(decoded.shape, (h, w))
            np.testing.assert_array_equal(decoded, mask)

    def test_full_mask_all_ones(self) -> None:
        """Test encoding and decoding of a full 255 mask across different resolutions."""
        for shape in [(1, 1), (10, 10), (720, 1280), (1080, 1920)]:
            h, w = shape
            mask = np.full((h, w), 255, dtype=np.uint8)
            encoded = encode_mask_rle(mask)
            self.assertEqual(encoded.dtype, np.int32)
            self.assertEqual(len(encoded), 2)
            self.assertEqual(encoded[0], 0)  # Starts with 0 zeros
            self.assertEqual(encoded[1], h * w)  # Followed by h*w ones

            decoded = decode_mask_rle(encoded, h, w)
            self.assertEqual(decoded.dtype, np.uint8)
            self.assertEqual(decoded.shape, (h, w))
            np.testing.assert_array_equal(decoded, mask)

    def test_single_blob_geometry(self) -> None:
        """Test rectangular foreground blob bit-exact roundtrip."""
        h, w = 100, 150
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:70, 30:110] = 255

        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, h, w)
        np.testing.assert_array_equal(decoded, mask)

    def test_multiple_disconnected_components(self) -> None:
        """Test multiple disconnected foreground islands bit-exact roundtrip."""
        h, w = 200, 300
        mask = np.zeros((h, w), dtype=np.uint8)
        # Component 1
        mask[10:30, 20:50] = 255
        # Component 2
        mask[80:120, 150:200] = 255
        # Component 3
        mask[160:190, 40:80] = 255
        # Component 4 (tiny 2x2 dot)
        mask[5:7, 280:282] = 255

        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, h, w)
        np.testing.assert_array_equal(decoded, mask)

    def test_thin_lines_geometry(self) -> None:
        """Test 1-pixel horizontal, vertical, and diagonal thin lines."""
        h, w = 120, 120
        mask = np.zeros((h, w), dtype=np.uint8)
        # Horizontal line
        mask[15, :] = 255
        # Vertical line
        mask[:, 40] = 255
        # Diagonal line
        for i in range(h):
            mask[i, i] = 255

        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, h, w)
        np.testing.assert_array_equal(decoded, mask)

    def test_checkerboard_pattern(self) -> None:
        """Test high-frequency checkerboard pattern roundtrip."""
        h, w = 40, 50
        y, x = np.ogrid[:h, :w]
        mask = (((x + y) % 2) * 255).astype(np.uint8)

        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, h, w)
        np.testing.assert_array_equal(decoded, mask)

    def test_boolean_dtype_roundtrip(self) -> None:
        """Test boolean mask inputs and outputs with dtype=bool."""
        h, w = 60, 80
        mask_bool = np.zeros((h, w), dtype=bool)
        mask_bool[10:40, 20:60] = True

        encoded = encode_mask_rle(mask_bool)
        decoded_bool = decode_mask_rle(encoded, h, w, dtype=bool)
        self.assertEqual(decoded_bool.dtype, bool)
        np.testing.assert_array_equal(decoded_bool, mask_bool)

    def test_uint8_binary_values(self) -> None:
        """Test values_255=False to produce 0 and 1 uint8 masks."""
        h, w = 30, 30
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[5:15, 5:15] = 1

        encoded = encode_mask_rle(mask)
        decoded = decode_mask_rle(encoded, h, w, dtype=np.uint8, values_255=False)
        np.testing.assert_array_equal(decoded, mask)


class TestMaskObjectMetadata(unittest.TestCase):
    """Unit tests for MaskObjectMetadata model and Color Bible entity links."""

    def test_valid_entity_types(self) -> None:
        """Test instantiation with valid entity types (character, object, location)."""
        for et in VALID_ENTITY_TYPES:
            meta = MaskObjectMetadata(
                obj_id=1,
                name=f"test_{et}",
                entity_type=et,
                entity_id="char_01",
                variant_id="var_a",
            )
            self.assertEqual(meta.entity_type, et)
            d = meta.to_dict()
            restored = MaskObjectMetadata.from_dict(d)
            self.assertEqual(restored.obj_id, meta.obj_id)
            self.assertEqual(restored.name, meta.name)
            self.assertEqual(restored.entity_type, meta.entity_type)
            self.assertEqual(restored.entity_id, meta.entity_id)
            self.assertEqual(restored.variant_id, meta.variant_id)

    def test_entity_type_case_normalization(self) -> None:
        """Test that entity_type string is normalized to lowercase."""
        meta = MaskObjectMetadata(obj_id=2, name="Hero", entity_type="  CHARACTER  ")
        self.assertEqual(meta.entity_type, "character")

    def test_invalid_entity_type_raises(self) -> None:
        """Test that invalid entity types raise ValueError."""
        with self.assertRaises(ValueError):
            MaskObjectMetadata(obj_id=1, name="invalid", entity_type="vehicle")
        with self.assertRaises(ValueError):
            MaskObjectMetadata(obj_id=1, name="invalid", entity_type="")


class TestShotMaskArchive(unittest.TestCase):
    """Unit tests for ShotMaskArchive serialization, retrieval, and compression."""

    def test_save_and_load_roundtrip(self) -> None:
        """Test complete save and load roundtrip with multiple objects and metadata."""
        h, w, frames = 100, 160, 5
        obj1 = MaskObjectMetadata(
            obj_id=1,
            name="protagonist_coat",
            entity_type="character",
            entity_id="char_01",
            variant_id="costume_02",
        )
        obj2 = MaskObjectMetadata(
            obj_id=2,
            name="vintage_lamp",
            entity_type="object",
            entity_id="prop_09",
            variant_id="default",
        )

        archive = ShotMaskArchive(
            shot_id="shot_001",
            frame_count=frames,
            height=h,
            width=w,
            objects=[obj1, obj2],
        )

        # Generate distinct masks for each object and frame
        saved_masks: dict[tuple[int, int], np.ndarray] = {}
        for t in range(frames):
            # Object 1: moving box
            m1 = np.zeros((h, w), dtype=np.uint8)
            m1[10 + t * 5 : 40 + t * 5, 20 + t * 8 : 70 + t * 8] = 255
            archive.set_mask(t, 1, m1)
            saved_masks[(1, t)] = m1

            # Object 2: static circle-ish patch
            m2 = np.zeros((h, w), dtype=np.uint8)
            m2[60:85, 90:120] = 255
            archive.set_mask(t, 2, m2)
            saved_masks[(2, t)] = m2

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "shot_001.masks.npz"
            save_shot_masks_npz(archive, archive_path)
            self.assertTrue(archive_path.is_file())

            # Load back using both classmethod and module function
            loaded = load_shot_masks_npz(archive_path)

            self.assertEqual(loaded.shot_id, "shot_001")
            self.assertEqual(loaded.frame_count, frames)
            self.assertEqual(loaded.height, h)
            self.assertEqual(loaded.width, w)
            self.assertEqual(len(loaded.objects), 2)

            o1 = loaded.get_object(1)
            self.assertIsNotNone(o1)
            self.assertEqual(o1.name, "protagonist_coat")
            self.assertEqual(o1.entity_type, "character")
            self.assertEqual(o1.entity_id, "char_01")
            self.assertEqual(o1.variant_id, "costume_02")

            o2 = loaded.get_object(2)
            self.assertIsNotNone(o2)
            self.assertEqual(o2.name, "vintage_lamp")
            self.assertEqual(o2.entity_type, "object")

            # Verify bit-exact masks
            for (obj_id, t), expected_mask in saved_masks.items():
                retrieved = loaded.get_mask(t, obj_id)
                self.assertIsNotNone(retrieved)
                np.testing.assert_array_equal(retrieved, expected_mask)

            # Test get_frame_masks
            frame_masks_0 = loaded.get_frame_masks(0)
            self.assertIn(1, frame_masks_0)
            self.assertIn(2, frame_masks_0)
            np.testing.assert_array_equal(frame_masks_0[1], saved_masks[(1, 0)])

    def test_compression_ratio_realistic_1080p(self) -> None:
        """Verify compression ratio is strictly >50x on realistic 1080p masks."""
        h, w = 1080, 1920
        frames = 20
        obj = MaskObjectMetadata(
            obj_id=1,
            name="lead_actor",
            entity_type="character",
            entity_id="c_lead",
        )
        archive = ShotMaskArchive("shot_bench", frames, h, w, [obj])

        # Realistic foreground object covering ~15% of frame area (600x520 = 312,000 px / 2.07M px ~ 15%)
        for t in range(frames):
            mask = np.zeros((h, w), dtype=np.uint8)
            y0 = 150 + t * 2
            x0 = 600 + t * 4
            mask[y0 : y0 + 600, x0 : x0 + 520] = 255
            archive.set_mask(t, 1, mask)

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "shot_bench.masks.npz"
            archive.save_npz(archive_path)

            ratio = archive.compression_ratio()
            # Must achieve >50x compression
            self.assertGreater(ratio, 50.0)

            # Also check loaded archive calculates same ratio from disk
            loaded = ShotMaskArchive.load_npz(archive_path)
            loaded_ratio = loaded.compression_ratio()
            self.assertAlmostEqual(ratio, loaded_ratio, places=2)

    def test_random_access_single_frame_decode(self) -> None:
        """Verify retrieving a single object mask at frame t does not decode entire video."""
        h, w, frames = 200, 300, 15
        obj = MaskObjectMetadata(obj_id=1, name="car", entity_type="object")
        archive = ShotMaskArchive("shot_random_access", frames, h, w, [obj])

        for t in range(frames):
            m = np.zeros((h, w), dtype=np.uint8)
            m[20:60, 20:80] = 255
            archive.set_mask(t, 1, m)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random.masks.npz"
            archive.save_npz(path)

            loaded = ShotMaskArchive.load_npz(path)

            # Spy on decode_mask_rle
            with mock.patch("ai_remaster_gui.rle_masks.decode_mask_rle", wraps=decode_mask_rle) as mock_decode:
                target_frame = 7
                retrieved = loaded.get_mask(target_frame, 1)
                self.assertIsNotNone(retrieved)
                # Ensure decode_mask_rle was called exactly ONCE for the requested frame
                self.assertEqual(mock_decode.call_count, 1)
                args, _ = mock_decode.call_args
                self.assertEqual(args[1], h)
                self.assertEqual(args[2], w)

    def test_sparse_frames_handling(self) -> None:
        """Verify handling of frames where objects are absent."""
        h, w, frames = 50, 50, 10
        obj = MaskObjectMetadata(obj_id=1, name="sword", entity_type="object")
        archive = ShotMaskArchive("shot_sparse", frames, h, w, [obj])

        # Object is present only at frames 2, 4, 6
        active_frames = {2, 4, 6}
        for t in active_frames:
            m = np.full((h, w), 255, dtype=np.uint8)
            archive.set_mask(t, 1, m)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sparse.masks.npz"
            archive.save_npz(path)
            loaded = ShotMaskArchive.load_npz(path)

            self.assertEqual(loaded.stored_mask_count(), 3)
            for t in range(frames):
                if t in active_frames:
                    self.assertTrue(loaded.has_mask(t, 1))
                    self.assertIsNotNone(loaded.get_mask(t, 1))
                    self.assertIn(1, loaded.get_frame_masks(t))
                else:
                    self.assertFalse(loaded.has_mask(t, 1))
                    self.assertIsNone(loaded.get_mask(t, 1))
                    self.assertNotIn(1, loaded.get_frame_masks(t))

    def test_clear_mask(self) -> None:
        """Verify clear_mask correctly removes a mask before saving."""
        h, w = 40, 40
        obj = MaskObjectMetadata(obj_id=1, name="prop", entity_type="object")
        archive = ShotMaskArchive("shot_clear", 3, h, w, [obj])
        archive.set_mask(0, 1, np.full((h, w), 255, dtype=np.uint8))
        self.assertTrue(archive.has_mask(0, 1))

        archive.clear_mask(0, 1)
        self.assertFalse(archive.has_mask(0, 1))
        self.assertIsNone(archive.get_mask(0, 1))


class TestInvalidInputs(unittest.TestCase):
    """Unit tests for input validation, boundary checking, and defensive error handling."""

    def test_invalid_mask_dimensions_in_codec(self) -> None:
        """encode_mask_rle requires 2D arrays."""
        with self.assertRaises(ValueError):
            encode_mask_rle(np.zeros((10, 10, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            encode_mask_rle(np.zeros((10,), dtype=np.uint8))

    def test_invalid_dimensions_in_decode(self) -> None:
        """decode_mask_rle rejects negative dimensions, empty RLE, or mismatched sum."""
        with self.assertRaises(ValueError):
            decode_mask_rle([10], -5, 2)
        with self.assertRaises(ValueError):
            decode_mask_rle([], 10, 10)
        with self.assertRaises(ValueError):
            decode_mask_rle([50], 10, 10)  # sum 50 != 100
        with self.assertRaises(ValueError):
            decode_mask_rle([-5, 105], 10, 10)  # negative length

    def test_mismatched_mask_dimensions_in_archive(self) -> None:
        """ShotMaskArchive.set_mask rejects masks not matching (height, width)."""
        obj = MaskObjectMetadata(obj_id=1, name="obj", entity_type="object")
        archive = ShotMaskArchive("shot_test", 5, 100, 200, [obj])

        wrong_shape_mask = np.zeros((100, 150), dtype=np.uint8)
        with self.assertRaises(ValueError):
            archive.set_mask(0, 1, wrong_shape_mask)

    def test_out_of_bounds_frame_indices(self) -> None:
        """ShotMaskArchive rejects frame indices outside [0, frame_count)."""
        obj = MaskObjectMetadata(obj_id=1, name="obj", entity_type="object")
        archive = ShotMaskArchive("shot_test", 5, 50, 50, [obj])

        with self.assertRaises(ValueError):
            archive.set_mask(5, 1, np.zeros((50, 50), dtype=np.uint8))
        with self.assertRaises(ValueError):
            archive.set_mask(-1, 1, np.zeros((50, 50), dtype=np.uint8))
        with self.assertRaises(ValueError):
            archive.get_mask(10, 1)
        with self.assertRaises(ValueError):
            archive.get_frame_masks(10)

    def test_unknown_object_id(self) -> None:
        """ShotMaskArchive.set_mask rejects unregistered object IDs."""
        archive = ShotMaskArchive("shot_test", 5, 50, 50)
        with self.assertRaises(ValueError):
            archive.set_mask(0, 999, np.zeros((50, 50), dtype=np.uint8))

    def test_load_nonexistent_file(self) -> None:
        """ShotMaskArchive.load_npz raises FileNotFoundError for missing files."""
        with self.assertRaises(FileNotFoundError):
            ShotMaskArchive.load_npz("non_existent_file.masks.npz")

    def test_load_corrupted_archive(self) -> None:
        """ShotMaskArchive.load_npz raises ValueError for corrupted NPZ missing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupt_path = Path(tmpdir) / "corrupt.npz"
            np.savez_compressed(corrupt_path, dummy=np.array([1, 2, 3]))
            with self.assertRaises(ValueError):
                ShotMaskArchive.load_npz(corrupt_path)


class TestCLIHelper(unittest.TestCase):
    """Unit tests for scripts/rle_masks.py CLI inspection and extraction."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.archive_path = Path(self.tmpdir.name) / "cli_test.masks.npz"

        h, w, frames = 60, 80, 4
        obj = MaskObjectMetadata(
            obj_id=1,
            name="cli_char",
            entity_type="character",
            entity_id="actor_99",
            variant_id="suit",
        )
        archive = ShotMaskArchive("cli_shot", frames, h, w, [obj])
        for t in range(frames):
            m = np.zeros((h, w), dtype=np.uint8)
            m[10:30, 10:40] = 255
            archive.set_mask(t, 1, m)
        archive.save_npz(self.archive_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_cli_info(self) -> None:
        """Test scripts/rle_masks.py --info prints formatted summary."""
        code = rle_cli.main([str(self.archive_path), "--info"])
        self.assertEqual(code, 0)

    def test_cli_info_json(self) -> None:
        """Test scripts/rle_masks.py --info --json prints valid JSON."""
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            code = rle_cli.main([str(self.archive_path), "--info", "--json"])
            self.assertEqual(code, 0)

        output = buf.getvalue().strip()
        data = json.loads(output)
        self.assertEqual(data["shot_id"], "cli_shot")
        self.assertEqual(data["frame_count"], 4)
        self.assertEqual(data["height"], 60)
        self.assertEqual(data["width"], 80)
        self.assertEqual(len(data["objects"]), 1)
        self.assertEqual(data["objects"][0]["name"], "cli_char")

    def test_cli_extract(self) -> None:
        """Test scripts/rle_masks.py --extract writes PNG mask frames."""
        out_dir = Path(self.tmpdir.name) / "extracted"
        code = rle_cli.main([str(self.archive_path), "--extract", "--output-dir", str(out_dir)])
        self.assertEqual(code, 0)
        png_files = list(out_dir.glob("*.png"))
        self.assertEqual(len(png_files), 4)


if __name__ == "__main__":
    unittest.main()
