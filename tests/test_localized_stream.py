"""Feature-length Localized research adapter: bounded memory and continuity tests.

The 300+ frame FFV1 test intentionally exceeds the original 240-frame cap.
Full-length movie performance and optical flow remain unverified.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
import artifact_ids as ids  # noqa: E402
from local_deflicker_research import local_deflicker_frames  # noqa: E402
from localized_deflicker_stream import (  # noqa: E402
    MAX_BUFFER_FRAMES, scan_shot_boundaries, stream_corrected_frames,
    validate_stream_request, _correct_center, _blur_luma,
)
from ai_remaster_gui import app, frame_differences  # noqa: E402
from ai_remaster_gui.runtime_settings import base_settings  # noqa: E402


def research_args(**changes):
    args = dict(
        localized_confirm_film_experiment=True,
        dearchive=False, ai_descratch=False, devignette=False, detect_defects=False,
    )
    args.update(changes)
    return argparse.Namespace(**args)


def textured_frame(width=96, height=64):
    yy, xx = np.mgrid[:height, :width]
    raw = (85 + 24*np.sin(xx/7) + 14*np.cos(yy/4) +
           12*((xx//11)%2)).astype(np.float32)
    return cv2.cvtColor(np.clip(raw,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)


class MemoryReader:
    def __init__(self,frames):
        self.frames=frames
        self.position=0
    def read(self):
        if self.position>=len(self.frames):
            return False, None
        f=self.frames[self.position]
        self.position+=1
        return True, f.copy()
    def grab(self):
        return self.position<len(self.frames)


def save_ffv1(path: Path, images, fps=12):
    height,width=images[0].shape[:2]
    cmd=[
        "ffmpeg","-hide_banner","-nostdin","-loglevel","error","-y",
        "-f","rawvideo","-pix_fmt","bgr24","-s:v",f"{width}x{height}",
        "-r",str(fps),"-i","pipe:0","-an","-c:v","ffv1","-f","matroska",str(path)
    ]
    completed=subprocess.run(
        cmd,input=b"".join(f.tobytes() for f in images),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90
    )
    if completed.returncode:
        raise AssertionError(completed.stderr.decode(errors="replace"))


class FilmStreamingTests(unittest.TestCase):
    def test_uncapped_frame_count_still_requires_valid_metadata_and_opt_in(self):
        info=dict(frames=280000,width=1920,height=1080,fps=24)
        self.assertEqual(validate_stream_request(research_args(),info),
                         (280000,1920,1080,24))
        self.assertEqual(MAX_BUFFER_FRAMES,5)
        with self.assertRaisesRegex(ValueError,"confirmation"):
            validate_stream_request(
                research_args(localized_confirm_film_experiment=False),info
            )
        with self.assertRaisesRegex(ValueError,"ALONE"):
            validate_stream_request(research_args(dearchive=True),info)
        with self.assertRaises(ValueError):
            validate_stream_request(research_args(),dict(info,frames=0))

    def test_exact_stream_matches_research_pilot_on_same_uncut_frames(self):
        base=textured_frame()
        frames=[
            np.clip(np.rint(base.astype(np.float32)*(1.16 if i%2 else 0.84)),0,255).astype(np.uint8)
            for i in range(52)
        ]
        expected,_=local_deflicker_frames(
            frames,radius=2,sigma=18,strength=0.6,cap=0.08,
            temporal_guard="oscillation"
        )
        report=list(stream_corrected_frames(
            MemoryReader(frames),[(0,len(frames),False)],
            base.shape[1],base.shape[0]
        ))
        self.assertEqual(len(report),len(frames))
        self.assertTrue(all(
            np.array_equal(expected[i],img)
            for i,img,_ in report
        ),"Streaming should match the original guarded numerical algorithm")

    def test_protected_fades_and_shot_boundaries_keep_frame_count(self):
        base=textured_frame().astype(np.float32)
        fade=[
            np.rint(np.clip(base*(0.60+0.015*i),0,255)).astype(np.uint8)
            for i in range(36)
        ]
        second=[(255-textured_frame()).copy() for _ in range(24)]
        frames=fade+second
        reader=MemoryReader(frames)
        output=list(stream_corrected_frames(
            reader,[(0,36,True),(36,60,False)],96,64
        ))
        self.assertEqual([i for i,_,_ in output],list(range(60)))
        self.assertTrue(all(np.array_equal(output[i][1],frames[i])
                            for i in range(36)))
        self.assertTrue(np.array_equal(output[36][1],frames[36]))

    def test_300_frame_real_video_and_distinct_identity(self):
        with tempfile.TemporaryDirectory(prefix="longfilm_") as temp:
            tmp=Path(temp)
            base=textured_frame()
            frames=[np.clip(
                np.rint(base.astype(np.float32)*(1.08 if i%2 else 0.92)),0,255
            ).astype(np.uint8) for i in range(305)]
            source=tmp/"source.mkv"
            output=tmp/"result.mkv"
            save_ffv1(source,frames)
            shots,scan=scan_shot_boundaries(source,305,96,64)
            self.assertEqual(sum(end-start for start,end,_ in shots),305)
            self.assertEqual(scan["frames"],305)
            cmd=[
                sys.executable,str(ROOT/"scripts"/"cleanup_video.py"),
                "--source",str(source),"--output",str(output),
                "--no-dearchive","--light-deflicker",
                "--deflicker-method","localized_stream",
                "--localized-confirm-film-experiment",
                "--intermediate-profile","lossless"
            ]
            r=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT,timeout=200)
            self.assertEqual(r.returncode,0,r.stdout[-2200:]+r.stderr[-2200:])
            self.assertIn("Localized streaming complete: 305 frames",r.stdout)
            cap=cv2.VideoCapture(str(output))
            try:
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),305)
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),96)
            finally:
                cap.release()
            second=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT,timeout=60)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertIn("Reuse Clean Up output",second.stdout)
            rejected=subprocess.run(
                [x for x in cmd if x!="--localized-confirm-film-experiment"],
                capture_output=True,text=True,cwd=ROOT,timeout=40
            )
            self.assertNotEqual(rejected.returncode,0)
            self.assertIn("confirmation",rejected.stderr+rejected.stdout)

            common=dict(
                source_name=source.name,lora="unused.safetensors",prompt="",
                negative_prompt="",strength=1,seed=42,dearchive=False,
                light_deflicker=True
            )
            stream=ids.cleanup_identity(**common,deflicker_method="localized_stream")
            short=ids.cleanup_identity(**common,deflicker_method="localized_research")
            self.assertNotEqual(ids.artifact_key(stream),ids.artifact_key(short))
            self.assertEqual(stream["shot_detection"],"arp_histogram_bhattacharyya_v1")

    def test_long_movie_chart_is_sparse_without_arbitrary_frame_cap(self):
        with tempfile.TemporaryDirectory(prefix="longchart_") as tmp:
            from unittest.mock import patch
            a=Path(tmp)/"a.mkv";b=Path(tmp)/"b.mkv"
            a.touch();b.touch()
            with patch.object(frame_differences,"_metadata",return_value={
                "frames":190000,"width":1920,"height":1080,"fps_fraction":"24/1"
            }):
                self.assertEqual(frame_differences._validate_pair(a,b),190000)


if __name__=="__main__":
    unittest.main()
