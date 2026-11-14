"""Offline PR24/core/wrapper comparison on the exact same 24 decoded frames.

No model, download, repair, project changes or accuracy claims without annotations.
Requires the pinned PR24 Git object locally; never fetches it automatically.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ai_remaster_gui import dust
from film_scratch_ridge import detect_sequence

REFERENCE_COMMIT = '15af65261a733b6472553d4d32fefcfa43253210'
REFERENCE_PATH = 'scripts/temporal_ridge_scratch_research.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reference_detector():
    source = subprocess.check_output(['git','-C',str(ROOT),'show',f'{REFERENCE_COMMIT}:{REFERENCE_PATH}'])
    tree = ast.parse(source)
    # Import only the pinned detector, not its evaluation harness or GT dependencies.
    nodes = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ('_as_stack','detect_sequence')]
    if {node.name for node in nodes} != {'_as_stack','detect_sequence'}:
        raise ValueError('Pinned PR24 detector functions are missing.')
    namespace = {'cv2':cv2,'np':np,'math':math}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),REFERENCE_PATH,'exec'),namespace)
    return namespace['detect_sequence'], hashlib.sha256(source).hexdigest()


def compare_frames(images, reference):
    if len(images) != 24:
        raise ValueError('This controlled comparison requires exactly 24 frames.')
    predictions = {'pr24':reference(images,sensitivity=.65), 'arp_core':detect_sequence(images,.65)}
    # Freeze the reader only: execute ARP's actual window/cut/endpoint wrapper.
    with patch.object(dust,'frames',side_effect=lambda _path:iter(images)):
        wrapped = list(dust.masks_for(Path('frozen-24-frame-input'),dust.options({'detector':'ridge'})))
    predictions['arp_wrapper'] = [mask for _,mask,_ in wrapped]
    if any(len(masks) != 24 for masks in predictions.values()):
        raise ValueError('Detector output count differs from the frozen input.')
    rows = []
    for index in range(24):
        masks = {key:values[index] for key,values in predictions.items()}
        if any(mask.shape != images[index].shape[:2] or mask.dtype != np.uint8 for mask in masks.values()):
            raise ValueError('Detector output geometry/type differs.')
        rows.append({'frame':index,'selected_pixels':{key:int(np.count_nonzero(mask)) for key,mask in masks.items()},
            'different_pixels_core':int(np.count_nonzero(masks['pr24'] != masks['arp_core'])),
            'different_pixels_wrapper':int(np.count_nonzero(masks['pr24'] != masks['arp_wrapper'])),
            'wrapper_reason':wrapped[index][2]})
    return predictions, rows


def compare(source, output, start_frame=0):
    source, output = Path(source).resolve(), Path(output).resolve()
    if type(start_frame) is not int or start_frame < 0:
        raise ValueError('Start frame must be a nonnegative integer.')
    if output.exists():
        raise FileExistsError('Comparison output already exists; nothing will be overwritten.')
    before = sha(source)
    reference, reference_sha = reference_detector()
    capture = cv2.VideoCapture(str(source))
    images = []
    try:
        if not capture.isOpened():raise ValueError('Cannot decode source.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:raise ValueError('Invalid source cadence.')
        for index in range(start_frame+24):
            ok, image = capture.read()
            if not ok:raise ValueError('Source does not contain the requested 24 frames.')
            if index >= start_frame:
                height,width=image.shape[:2]
                if height < 32 or width < 32 or height*width*24 > 96_000_000:
                    raise ValueError('PR24 geometry/memory limit exceeded; no resizing applied.')
                images.append(image.copy())
    finally:
        capture.release()
    predictions, rows = compare_frames(images,reference)
    if sha(source) != before:raise ValueError('Source changed during comparison; refusing publication.')
    output.mkdir(parents=True)
    for name,masks in predictions.items():
        folder=output/name;folder.mkdir()
        for index,mask in enumerate(masks):
            if not cv2.imwrite(str(folder/f'frame_{index:06d}.png'),mask):raise OSError('Could not save exact mask.')
    sheets=[]
    for index in (0,1,12,22,23):
        tiles=[]
        for name in ('input','pr24','arp_core','arp_wrapper','disagreement'):
            image=images[index].copy()
            if name=='disagreement':
                selected=predictions['pr24'][index] != predictions['arp_wrapper'][index]
                image[selected]=(0,0,255)
            elif name!='input':
                selected=predictions[name][index]>0
                image[selected]=(image[selected]*.5+np.array([255,0,255])*.5).astype(np.uint8)
            tile=cv2.resize(image,(320,round(image.shape[0]*320/image.shape[1])),interpolation=cv2.INTER_AREA)
            bar=np.full((32,320,3),20,np.uint8)
            cv2.putText(bar,f'{start_frame+index}: {name}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            tiles.append(np.vstack((bar,tile)))
        sheets.append(np.hstack(tiles))
    if not cv2.imwrite(str(output/'comparison.png'),np.vstack(sheets)):raise OSError('Could not save comparison.')
    report={'scope':'agreement of predictions, NOT defect accuracy; no GT or automatic winner',
        'source':str(source),'source_sha256':before,'start_frame':start_frame,'frames':24,'fps':fps,
        'width':images[0].shape[1],'height':images[0].shape[0],
        'input':'same frozen BGR8 frames for all three paths; no resize or added preprocessing',
        'reference_commit':REFERENCE_COMMIT,'reference_path':REFERENCE_PATH,'reference_file_sha256':reference_sha,
        'arp_core_sha256':sha(ROOT/'scripts/film_scratch_ridge.py'),'arp_wrapper_sha256':sha(ROOT/'ai_remaster_gui/dust.py'),
        'numpy':np.__version__,'opencv':cv2.__version__,'sensitivity':.65,'additional_dilation':0,
        'core_equal':all(row['different_pixels_core']==0 for row in rows),
        'wrapper_equal':all(row['different_pixels_wrapper']==0 for row in rows),
        'different_frames_wrapper':[row['frame'] for row in rows if row['different_pixels_wrapper']],
        'both_empty_frames':[i for i in range(24) if not np.any(predictions['pr24'][i]) and not np.any(predictions['arp_wrapper'][i])],
        'per_frame':rows,'display':'comparison.png is downsampled; separate PNG masks are exact native geometry'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start-frame',type=int,default=0)
    args=parser.parse_args()
    report=compare(args.source,args.output,args.start_frame)
    print(json.dumps({key:report[key] for key in ('source','width','height','core_equal','wrapper_equal','different_frames_wrapper','both_empty_frames')},indent=2))


if __name__=='__main__':main()
