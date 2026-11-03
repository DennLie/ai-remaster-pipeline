"""Exact Remove Noise review: source lineage, frame accuracy and chart security.

Real FFV1 is used for reference/variant frame decoding. No trained models.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_remaster_gui import denoise_review, frame_inspector


class MinimalApp:
    settings = {"global": {"source": "unused"}}


class DenoiseReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="arp_denoise_exact_")
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.input = self.folder / "original.mkv"
        self.outputs = self.folder / "intermediate" / "denoised"
        self.outputs.mkdir(parents=True)
        self.variant = self.outputs / "Neighbors_denoised_hqdn3d_gentle.mkv"
        self.ffmpeg = shutil.which("ffmpeg")
        if not self.ffmpeg:
            self.skipTest("Real FFmpeg required")
        made = subprocess.run([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=64x48:rate=12:duration=1.5",
            "-an", "-c:v", "ffv1", str(self.input),
        ], capture_output=True, text=True, timeout=30)
        self.assertEqual(made.returncode,0,made.stderr)
        processed = subprocess.run([
            self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(self.input),
            "-vf", "hqdn3d=luma_spatial=0.9:chroma_spatial=0.7:luma_tmp=1.2:chroma_tmp=0.8",
            "-an", "-c:v", "ffv1", str(self.variant),
        ],capture_output=True,text=True,timeout=30)
        self.assertEqual(processed.returncode,0,processed.stderr)
        self.sidecar = self.variant.with_suffix(".mkv.sig.json")
        self.write_provenance()
        self.patches = [
            patch.object(denoise_review,"_OUTPUTS",self.outputs),
            patch.object(denoise_review,"_CACHE",self.folder / "difference_cache"),
            patch.object(denoise_review,"pipeline_source_text",return_value=str(self.input)),
            patch.object(frame_inspector,"ROOT",self.folder),
            patch.object(frame_inspector,"CACHE",self.folder / "image_cache"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def write_provenance(self,source=None,mtime=None):
        source = source or self.input
        stat = self.input.stat()
        self.sidecar.write_text(json.dumps({
            "tool":"denoise_video.py","version":1,
            "source":str(source.resolve()),
            "source_fingerprint":{
                "size":stat.st_size,
                "mtime_ns":stat.st_mtime_ns if mtime is None else mtime,
                "sha256":"placeholder-for-verified-production-checksum"
            },
            "method":"hqdn3d","preset":"gentle","intermediate_profile":"lossless",
        }),encoding="utf-8")

    def test_safe_pair_and_exact_stills(self):
        src,dst = denoise_review.review_pair(MinimalApp(),str(self.variant))
        self.assertEqual(src,self.input)
        self.assertEqual(dst,self.variant)
        item = denoise_review.inspect_denoise(MinimalApp(),str(self.variant),8)
        self.assertEqual(item["frame"],8)
        self.assertEqual(item["frames"],18)
        self.assertEqual(item["stage"],"denoise")
        self.assertTrue(item["exact"])
        self.assertEqual(item["method"],"hqdn3d")
        for key in ("before","after"):
            self.assertTrue((self.folder / item[key]).is_file())
        with self.assertRaises(IndexError):
            denoise_review.inspect_denoise(MinimalApp(),str(self.variant),18)

    def test_directional_chart_and_reused_report(self):
        first = denoise_review.denoise_frame_differences(MinimalApp(),str(self.variant))
        self.assertEqual(first["frames"],18)
        self.assertEqual(len(first["samples"]),18)
        self.assertEqual(first["metric"],"directional_mean_luma_8bit")
        with patch.object(denoise_review,"_compare_sampled_frames",side_effect=AssertionError("Cache miss")):
            again = denoise_review.denoise_frame_differences(MinimalApp(),str(self.variant))
        self.assertEqual(first,again)

    def test_reject_unsigned_stale_and_unrelated_results(self):
        self.sidecar.unlink()
        with self.assertRaisesRegex(ValueError,"provenance"):
            denoise_review.review_pair(MinimalApp(),str(self.variant))
        self.write_provenance()
        self.write_provenance(mtime=0)
        with self.assertRaisesRegex(ValueError,"stale"):
            denoise_review.review_pair(MinimalApp(),str(self.variant))
        self.write_provenance(source=self.folder/"unrelated.mkv")
        with self.assertRaisesRegex(ValueError,"different source"):
            denoise_review.review_pair(MinimalApp(),str(self.variant))
        self.write_provenance()
        other = self.folder / "unrelated.mkv"
        other.write_bytes(self.variant.read_bytes())
        with self.assertRaisesRegex(ValueError,"Only finished"):
            denoise_review.review_pair(MinimalApp(),str(other))
        with self.assertRaises(ValueError):
            denoise_review.review_pair(MinimalApp(),str(self.outputs/"clip.partial.mkv"))
        with self.assertRaises(ValueError):
            denoise_review.review_pair(MinimalApp(),"")
        with self.assertRaises(ValueError):
            denoise_review.inspect_denoise(MinimalApp(),str(self.variant),-1)

    def test_input_source_is_actual_processing_input(self):
        self.assertNotEqual(str(self.variant),str(self.input))
        path = denoise_review.review_pair(MinimalApp(),str(self.variant))
        self.assertEqual(path[0],self.input)


if __name__=="__main__":
    unittest.main()
