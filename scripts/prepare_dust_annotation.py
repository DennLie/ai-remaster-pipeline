"""Export a fixed, unlabelled native-frame sample for human defect annotation."""
import argparse
import csv
import json
from pathlib import Path

import cv2

from compare_ridge_integration import sha

SAMPLE = (0, 12, 23, 36, 48, 71)


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Output exists; refusing overwrite.')
    before = sha(source)
    capture = cv2.VideoCapture(str(source))
    samples = {}
    try:
        for index in range(max(SAMPLE) + 1):
            ok, image = capture.read()
            if not ok:
                raise ValueError('Need at least 72 decodable frames; no partial sample published.')
            if index in SAMPLE:
                samples[index] = image
    finally:
        capture.release()
    if sha(source) != before:
        raise ValueError('Source changed while decoding.')
    output.mkdir(parents=True)
    rows = []
    for index, image in samples.items():
        name = f'frame_{index:06d}.png'
        if not cv2.imwrite(str(output / name), image):
            raise OSError('Cannot save native sample')
        rows.append({'frame_index_zero_based': index, 'image': name, 'image_sha256': sha(output / name),
                     'width': image.shape[1], 'height': image.shape[0], 'annotation_status': 'pending',
                     'damage_classes': 'unknown', 'split': 'unassigned'})
    with (output / 'samples.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {'source': str(source), 'source_sha256': before, 'opencv': cv2.__version__,
                'selection': 'Fixed frame indices; not selected from detector output. Exploratory annotation, not independent holdout.',
                'samples': rows,
                'instructions': 'Annotate only human-reviewed pixels: 0=healthy, 127=unknown/ignore, 255=defect. '
                'Separate dust and scratch classes. Review neighbouring frames for persistence and scene motion. '
                'No ground-truth mask is created automatically; a blank mask must not stand for a reviewed negative.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
