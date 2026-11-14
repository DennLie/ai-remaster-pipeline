"""Content-independent, pixel-centre footprint replay, not interpolation-quality QA.

The same local motions and optimizer act on a synthetic white canvas. Nearest
sampling measures geometric support, not the bicubic kernel or dark film pixels.
Source-area estimates use the planar libvidstab backward-map determinant
(1 - zoom/100)^2; translation and rotation have unit determinant.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from common import file_fingerprint


def replay(ffmpeg, work_dir, index, frames, width, height, fps, transform, fallback):
    folder = work_dir / 'geometry'
    folder.mkdir(exist_ok=True)
    stem = f'shot_{index:04d}'
    analysis = folder / (stem + '.trf')
    global_motion = folder / (stem + '_global.trf')
    shutil.copy2(work_dir / (stem + '.trf'), analysis)
    shutil.copy2(work_dir / 'global_motions.trf', global_motion)
    mask = folder / (stem + '_mask.mkv')
    if fallback:
        graph = 'format=gray'
    else:
        # Remove the source trim: the artificial canvas already starts at frame zero.
        graph = 'format=yuv444p,' + transform[transform.index('vidstabtransform='):]
        graph = graph.replace('interpol=bicubic', 'interpol=no') + ',format=gray'
    subprocess.run([ffmpeg, '-v', 'error', '-y', '-f', 'lavfi', '-i',
        f'color=c=white:s={width}x{height}:r={fps:.10f},format=yuv444p', '-vf', graph,
        '-frames:v', str(frames), '-an', '-c:v', 'ffv1', str(mask)], cwd=work_dir, check=True)
    if not fallback and (work_dir / 'global_motions.trf').read_bytes() != global_motion.read_bytes():
        raise ValueError('Mask replay produced different motion fits; coverage is unavailable.')
    counts = []
    capture = cv2.VideoCapture(str(mask))
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if image.shape[:2] != (height, width):
                raise ValueError('Footprint geometry mismatch.')
            counts.append(int(np.count_nonzero(image[:, :, 0] > 127)))
    finally:
        capture.release()
    if len(counts) != frames:
        raise ValueError('Footprint frame count mismatch.')
    return {'mask': mask.name, 'analysis': analysis.name, 'global_motion': global_motion.name,
            'replay_filter': graph, 'valid_pixels': counts, 'mask_fingerprint': file_fingerprint(mask)}


def publish(output, signature, scenes, width, height, work_dir, ffmpeg):
    destination = output.with_suffix('.geometry')
    destination.mkdir(exist_ok=True)
    for scene in scenes:
        if 'mask' not in scene:
            continue
        for key in ('mask', 'analysis', 'global_motion'):
            source = work_dir / 'geometry' / scene[key]
            partial = destination / (source.name + '.tmp')
            shutil.copy2(source, partial)
            partial.replace(destination / source.name)
            scene[key + '_fingerprint'] = file_fingerprint(destination / source.name)
        scene['mask_fingerprint'] = file_fingerprint(destination / scene['mask'])
    version = subprocess.check_output([ffmpeg, '-version'], text=True)
    report = {'version': 1, 'model': 'libvidstab_white_canvas_nearest_pixel_centres',
        'signature': signature, 'output_fingerprint': file_fingerprint(output),
        'width': width, 'height': height, 'ffmpeg': version, 'scenes': scenes}
    target = output.with_suffix(output.suffix + '.geometry.json')
    partial = target.with_suffix('.json.tmp')
    partial.write_text(json.dumps(report) + '\n', encoding='utf-8')
    partial.replace(target)
