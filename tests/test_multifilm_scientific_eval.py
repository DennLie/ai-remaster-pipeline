"""Known-source multi-film orchestration and no-leak provenance contracts."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_multifilm_scientific_eval as pilot


def prepare_fake_corpus(parent: Path, original: Path, film: str) -> None:
    import restoration_evaluation as ev
    folder = parent / film
    scene = folder / "scene1"
    scene.mkdir(parents=True)
    shutil.copyfile(original, scene / "original_native.mkv")
    folder.joinpath("corpus_manifest.json").write_text(json.dumps({
        "schema_version": 1, "id": film,
        "source_provenance": {"downloaded_sha256": ev.digest(original)},
        "rights_status": "synthetic unit test, no actual archive film",
        "clips": [{"name": "scene1", "files": {
            "original_native.mkv": {"sha256": ev.digest(scene / "original_native.mkv")},
        }}],
    }))


class MultiFilmPilotTests(unittest.TestCase):
    def test_duplicate_and_escaping_film_names_refused(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td)
            with self.assertRaises(ValueError):
                pilot.source_shots(corpus, ["a", "a"], 1)
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                pilot.source_shots(corpus, ["../escaping"], 1)
            with self.assertRaises(ValueError):
                pilot.parse_csv("caligari,caligari")
            with self.assertRaises(ValueError):
                pilot.parse_csv("")
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                pilot.execute(corpus_root=corpus, output_root=corpus / "test",
                              films=["film1"], stages=["madeup"])

    def test_true_detail_and_film_level_output_on_verified_ffv1_corpus(self):
        import restoration_evaluation as ev
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-film-pilot-") as td:
            root = Path(td)
            film = root / "synthetic.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                 "-i", "testsrc2=size=96x72:rate=12:duration=1",
                 "-an", "-c:v", "ffv1", str(film)],
                check=True, capture_output=True, timeout=35,
            )
            folder = root / "corpus"
            prepare_fake_corpus(folder, film, "film_a")
            sources = pilot.source_shots(folder, ["film_a"], 1)
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0]["sha256"], ev.digest(film))
            report = pilot.execute(
                corpus_root=folder, output_root=root / "scores",
                films=["film_a"], stages=["detail"],
                ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            self.assertEqual(report["stage_runs"], 1)
            self.assertEqual(len(report["film_level_summary"]["results"]), 3)
            self.assertTrue(all(x["ci95"] is None for x in report["film_level_summary"]["results"]))
            self.assertEqual(report["outputs"][0]["degradation"], "gaussian_blur")
            self.assertTrue((root / "scores" / "multifilm_report.json").exists())
            with self.assertRaises(FileExistsError):
                pilot.execute(corpus_root=folder, output_root=root / "scores",
                              films=["film_a"], stages=["detail"],
                              ffmpeg=ffmpeg, ffprobe=ffprobe)
            # Modifying the actual source must invalidate the corpus reference.
            (folder / "film_a" / "scene1" / "original_native.mkv").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Native film scene changed"):
                pilot.source_shots(folder, ["film_a"], 1)


if __name__ == "__main__":
    unittest.main()
