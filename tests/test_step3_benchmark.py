"""CPU-only, real FFmpeg Step-3 multi-shot alignment/provenance tests."""
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
import step3_benchmark as bench  # noqa: E402


class Step3BenchmarkTests(unittest.TestCase):
    def test_invalid_and_duplicate_shots_and_geometry_fail_closed(self):
        for spec in (["bad/label=0:6"], ["A=nan:6"], ["A=-1:6"],
                     ["A=0:0"], ["A=0:73"], ["A=0:abc"]):
            with self.assertRaises(ValueError):
                bench.parse_shots(spec)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            bench.parse_shots(["Scene=0:6", "scene=1:6"])
        with self.assertRaises(ValueError):
            bench.parse_shots([])
        original = {"width": 160, "height": 120}
        bench.validate_geometry(original, 512, 384)
        with self.assertRaisesRegex(ValueError, "aspect ratio"):
            bench.validate_geometry(original, 512, 512)
        with self.assertRaisesRegex(ValueError, "multiples of 128"):
            bench.validate_geometry(original, 500, 384)
        with self.assertRaisesRegex(ValueError, "Variant"):
            bench.parse_variants(["unknown"])

    def test_manifest_protects_against_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaisesRegex(ValueError, "escapes"):
                bench.checked_relative(root, "../other.mkv")
            with self.assertRaisesRegex(ValueError, "Missing"):
                bench.checked_relative(root, "")

    def test_research_sidecars_and_stage_signatures_are_checked(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            output = root / "research.mkv"
            output.write_bytes(b"never execute weights")
            expected = hashlib.sha256(b"native clip").hexdigest()
            with self.assertRaisesRegex(ValueError, "No matching source signature"):
                bench.validate_provenance(output, expected, 6)
            d = bench.validate_provenance(output, expected, 6, allow_unverified=True)
            self.assertIn("UNVERIFIED", d["status"])
            sidecar = output.with_suffix(".research.json")
            base = {
                "source_sha256": expected, "start_seconds": 0.,
                "frames_requested": 6, "output_sha256": bench.file_hash(output),
            }
            sidecar.write_text(json.dumps(base))
            self.assertEqual(bench.validate_provenance(output, expected, 6)["status"],
                             "research_sidecar_verified")
            base["start_seconds"] = 1.
            sidecar.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "frame zero"):
                bench.validate_provenance(output, expected, 6)
            base["start_seconds"] = 0.
            base["source_sha256"] = "0" * 64
            sidecar.write_text(json.dumps(base))
            with self.assertRaisesRegex(ValueError, "source SHA mismatch"):
                bench.validate_provenance(output, expected, 6)
            sidecar.unlink()
            stage = output.with_suffix(".mkv.sig.json")
            stage.write_text(json.dumps({"source_fingerprint": {"sha256": expected}}))
            self.assertEqual(bench.validate_provenance(output, expected, 6)["status"],
                             "arp_stage_signature_verified")

    def test_real_source_prepare_and_native_vsr_comparisons(self):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        self.assertTrue(ffmpeg and ffprobe)
        with tempfile.TemporaryDirectory(prefix="arp-step3-benchmark-") as td:
            root = Path(td)
            source = root / "source.mkv"
            subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                 "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=12:duration=3",
                 "-c:v", "ffv1", str(source)],
                check=True, capture_output=True, timeout=30,
            )
            digest = bench.file_hash(source)
            pack = root / "prepared"
            shots = bench.parse_shots(["motion=0:6", "other=1:6"])
            result = bench.prepare(
                source=source, destination=pack, shots=shots,
                width=512, height=384, expected_source_sha256=digest,
                ffmpeg=ffmpeg, ffprobe=ffprobe,
            )
            self.assertEqual(len(result["shots"]), 2)
            self.assertEqual((pack / "manifest.json").is_file(), True)
            for shot in result["shots"]:
                native = pack / shot["native_path"]
                scaled = pack / shot["lanczos_path"]
                self.assertEqual(bench.video_probe(ffprobe, native)["frame_count"], 6)
                self.assertEqual(bench.video_probe(ffprobe, scaled)["frame_count"], 6)
                self.assertEqual(bench.video_probe(ffprobe, scaled)["width"], 512)
                self.assertEqual(bench.file_hash(native), shot["native_sha256"])
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                bench.prepare(source=source, destination=pack, shots=shots,
                              width=512, height=384, ffmpeg=ffmpeg, ffprobe=ffprobe)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                bench.prepare(source=source, destination=root / "wrong-hash",
                              shots=shots, width=512, height=384,
                              expected_source_sha256="0"*64, ffmpeg=ffmpeg, ffprobe=ffprobe)

            shot = result["shots"][0]
            native_variant = root / "classic.mkv"
            vsr_variant = root / "modern.mp4"
            shutil.copyfile(pack / shot["native_path"], native_variant)
            shutil.copyfile(pack / shot["lanczos_path"], vsr_variant)
            native_variant.with_suffix(".mkv.sig.json").write_text(json.dumps({
                "source_fingerprint": {"sha256": shot["native_sha256"]},
            }))
            vsr_variant.with_suffix(".research.json").write_text(json.dumps({
                "original_sha256": shot["native_sha256"],
                "start_seconds": 0., "frames_requested": 6,
                "output_sha256": bench.file_hash(vsr_variant),
            }))
            output_dir = root / "comparison"
            compared = bench.evaluate(
                manifest_path=pack / "manifest.json", shot_name="motion",
                native={"pass_native": native_variant},
                vsr={"pass_vsr": vsr_variant},
                destination=output_dir, ffprobe=ffprobe,
            )
            self.assertEqual(compared["diagnostics"]["native"]["frames"], 6)
            self.assertEqual(compared["diagnostics"]["vsr"]["frames"], 6)
            self.assertEqual(compared["diagnostics"]["native"]["variants"]["pass_native"]["mean_abs_change"], 0.)
            self.assertEqual(compared["diagnostics"]["vsr"]["variants"]["pass_vsr"]["mean_abs_change"], 0.)
            self.assertTrue((output_dir / "native" / "denoising_comparison.jpg").is_file())
            self.assertTrue((output_dir / "vsr" / "denoising_comparison.jpg").is_file())
            self.assertTrue((output_dir / "step3_evaluation.json").is_file())
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                bench.evaluate(
                    manifest_path=pack / "manifest.json", shot_name="motion",
                    native={"pass_native": native_variant}, vsr={},
                    destination=output_dir, ffprobe=ffprobe,
                )
            # Research original pointer from ANOTHER shot must never pass even
            # when size/fps/frame count happen to match.
            sidecar = vsr_variant.with_suffix(".research.json")
            changed = json.loads(sidecar.read_text())
            changed["original_sha256"] = result["shots"][1]["native_sha256"]
            sidecar.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "source SHA mismatch"):
                bench.evaluate(
                    manifest_path=pack / "manifest.json", shot_name="motion",
                    native={}, vsr={"pass_vsr": vsr_variant},
                    destination=root / "wrong-shot", ffprobe=ffprobe,
                )


if __name__ == "__main__":
    unittest.main()
