"""Offline contracts for public-domain film curation and real media packaging."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_public_domain_films as film  # noqa: E402


class CuratedPublicDomainFilmsTests(unittest.TestCase):
    def test_curated_sources_have_named_distinct_films_and_two_shots(self):
        catalog = film.load_catalog(ROOT / "benchmarks/public_domain_films/catalog.json")
        self.assertGreaterEqual(len(catalog["films"]), 5)
        self.assertGreaterEqual(sum(len(x["shots"]) for x in catalog["films"]), 10)
        self.assertEqual(len({x["id"] for x in catalog["films"]}), len(catalog["films"]))
        for entry in catalog["films"]:
            self.assertIn("transfer_note", entry)
            self.assertIn("right_status", entry)
            self.assertTrue(entry["source_page"].startswith("https://commons.wikimedia.org/"))
            self.assertGreaterEqual(len(entry["shots"]), 2)

    def test_360p_derivatives_from_api_no_invented_url(self):
        source = {
            "id": "testfilm", "commons_file": "Test.webm",
            "acquire": "vp9_360p", "source_original_sha1": None,
        }
        response = {"query": {"pages": [{"pageid": 23, "videoinfo": [{
            "url": "https://upload.wikimedia.org/wikipedia/commons/original.webm",
            "sha1": "a" * 40, "size": 1500000000,
            "derivatives": [
                {"transcodekey": "240p.vp9.webm", "height": 240,
                 "src": "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/240p.vp9.webm"},
                {"transcodekey": "360p.vp9.webm", "height": 360,
                 "src": "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/360p.vp9.webm"},
            ],
        }]}]}}
        with patch.object(film, "request_json", return_value=response):
            selected = film.remote_video(source)
            self.assertTrue(selected["selected_media_url"].endswith("360p.vp9.webm"))
            self.assertEqual(selected["commons_original_sha1"], "a" * 40)
        source["source_original_sha1"] = "b" * 40
        with patch.object(film, "request_json", return_value=response):
            with self.assertRaisesRegex(ValueError, "changed"):
                film.remote_video(source)
        source["source_original_sha1"] = None
        source["acquire"] = "original"
        with patch.object(film, "request_json", return_value=response):
            selected = film.remote_video(source)
            self.assertTrue(selected["selected_media_url"].endswith("original.webm"))

    def test_vsr_geometry_never_stretches_or_crops(self):
        self.assertEqual(film.compatible_vsr_geometry(480, 360),
                         {"width": 512, "height": 384, "ratio_preserved": True})
        self.assertEqual(film.compatible_vsr_geometry(540, 360),
                         {"width": 768, "height": 512, "ratio_preserved": True})
        self.assertIsNone(film.compatible_vsr_geometry(462, 360))
        self.assertRaises(FileNotFoundError, film.load_catalog, Path("/not-a-real-json-file.json"))

    def test_real_local_synthetic_source_generates_audio_free_research_clips(self):
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-corpus-test-") as td:
            root = Path(td)
            source = root / "source.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12:duration=4",
                 "-f", "lavfi", "-i", "sine=frequency=300:duration=4",
                 "-map", "0:v", "-map", "1:a",
                 "-c:v", "ffv1", "-c:a", "pcm_s16le", str(source)],
                capture_output=True, check=True, timeout=40,
            )
            sha = film.hash_file(source)
            entry = {
                "id": "testfilm", "title": "Synthetic test-only film", "year": 1900,
                "creator": "Unit test", "commons_file": "Test.webm",
                "source_page": "https://commons.wikimedia.org/wiki/File:Test.webm",
                "acquire": "original", "right_status": "test only",
                "transfer_note": "This is not an actual film", "tests": ["synthetic"],
                "shots": [
                    {"name": "shot_a", "seconds": 0, "frames": 12},
                    {"name": "shot_b", "seconds": 2, "frames": 12},
                ],
            }
            result_dir = root / "generated"
            report = film.render_one(
                {"schema_version": 1}, entry, source, result_dir,
                source_provenance={"source": "local", "downloaded_sha256": sha},
                ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            self.assertEqual(len(report["clips"]), 2)
            self.assertFalse(report["original_audio_redistributed"])
            self.assertEqual(sha, film.hash_file(source))
            for shot in entry["shots"]:
                folder = result_dir / shot["name"]
                self.assertTrue((folder / "original_native.mkv").is_file())
                self.assertTrue((folder / "synthetic_blur_mild.mkv").is_file())
                self.assertTrue((folder / "synthetic_blur_medium.mkv").is_file())
                self.assertTrue((folder / "unsharp_gentle.mkv").is_file())
                self.assertTrue((folder / "cas_gentle.mkv").is_file())
                self.assertTrue((folder / "lanczos_reference.mkv").is_file())
                self.assertTrue((folder / "diagnostics/denoising_comparison.json").is_file())
                self.assertEqual(film.probe(ffprobe, folder / "original_native.mkv", count_frames=True)["frames"], 12)
                audio = subprocess.run(
                    [ffprobe, "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=index", "-of", "csv=p=0",
                     str(folder / "original_native.mkv")],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
                self.assertEqual(audio, "")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                film.render_one({"schema_version": 1}, entry, source, result_dir,
                                source_provenance={"downloaded_sha256": sha},
                                ffmpeg=ffmpeg, ffprobe=ffprobe)


if __name__ == "__main__":
    unittest.main()
