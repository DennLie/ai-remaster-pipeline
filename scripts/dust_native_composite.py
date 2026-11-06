"""Lossless 10-bit composition; only declared native-plane support is replaced.

Models still produce BGR8. This preserves native source samples elsewhere, not
10-bit model detail. Chroma support is the union of each subsampled pixel block.
"""
from contextlib import ExitStack
import hashlib
import subprocess
import tempfile

import cv2
import numpy as np


FORMATS = {'yuv420p10le': (2, 2), 'yuv422p10le': (2, 1),
           'yuv444p10le': (1, 1), 'gbrp10le': (1, 1), 'gray10le': None}


def plane_shapes(meta):
    fmt = meta['pixel_format']
    if fmt not in FORMATS:
        raise ValueError('Unsupported native repair pixel format.')
    h, w = meta['height'], meta['width']
    chroma = FORMATS[fmt]
    return [(h, w)] if chroma is None else [(h, w)] + [((h+chroma[1]-1)//chroma[1], (w+chroma[0]-1)//chroma[0])] * 2


def plane_masks(mask, meta):
    shapes = plane_shapes(meta)
    result = [mask.astype(bool)]
    for h, w in shapes[1:]:
        sx = (meta['width'] + w-1)//w
        sy = (meta['height'] + h-1)//h
        padded = np.pad(mask, ((0, h*sy-mask.shape[0]), (0, w*sx-mask.shape[1])), constant_values=False)
        result.append(padded.reshape(h, sy, w, sx).any(axis=(1, 3)))
    return result


def read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size-len(data))
        if not chunk:
            raise ValueError('Native frame stream ended early.')
        data.extend(chunk)
    return bytes(data)


def composite_frame(original, patch, mask, meta):
    shapes = plane_shapes(meta)
    supports = plane_masks(mask, meta)
    chunks, counts, offset = [], [], 0
    for (h, w), support in zip(shapes, supports, strict=True):
        size = h*w*2
        before = np.frombuffer(original[offset:offset+size], '<u2').reshape(h, w)
        replacement = np.frombuffer(patch[offset:offset+size], '<u2').reshape(h, w)
        composed = before.copy()
        composed[support] = replacement[support]
        changed = composed != before
        counts.append({'support_samples': int(support.sum()), 'changed_inside_samples': int((changed & support).sum()),
                       'changed_outside_samples': int((changed & ~support).sum())})
        chunks.append(composed.tobytes())
        offset += size
    return b''.join(chunks), counts


def compose(ffmpeg, source, patch, mask_path, output, meta):
    """Stream native samples, verify encoded raster digest, retain source audio."""
    fmt = meta['pixel_format']
    size = sum(h*w*2 for h,w in plane_shapes(meta))
    with ExitStack() as stack:
        def process(args, **kwargs):
            log = stack.enter_context(tempfile.TemporaryFile())
            p = subprocess.Popen([ffmpeg, '-v', 'error', *args], stderr=log, **kwargs)
            def close():
                if p.poll() is None:
                    p.kill()
                p.wait()
                for stream in (p.stdin, p.stdout):
                    if stream:
                        stream.close()
            stack.callback(close)
            return p
        common = ['-map', '0:v:0', '-an', '-pix_fmt', fmt, '-f', 'rawvideo', 'pipe:1']
        base = process(['-i', str(source), *common], stdout=subprocess.PIPE)
        filters = []
        if fmt.startswith('yuv'):
            matrices = {'bt709':'bt709', 'bt470bg':'bt601', 'smpte170m':'bt601', 'bt2020nc':'bt2020'}
            space = meta.get('color_space', 'unknown')
            if space not in (*matrices, 'unknown'):
                raise ValueError('Unsupported source colour matrix; no implicit conversion.')
            filters = ['-vf', 'scale=out_color_matrix=' + matrices.get(space, 'bt601') + ':out_range=' + ('pc' if meta.get('color_range') == 'pc' else 'tv')]
        replacement = process(['-i', str(patch), *filters, *common], stdout=subprocess.PIPE)
        colour = []
        for field, flag in (('color_range','-color_range'), ('color_space','-colorspace'), ('transfer','-color_trc'), ('color_primaries','-color_primaries')):
            value = meta.get(field, 'unknown')
            if value not in ('unknown', 'unspecified', None):
                colour += [flag, value]
        # Tag the raw INPUT too: output-only colour tags can make FFmpeg insert
        # an unwanted range/matrix conversion into otherwise identical samples.
        writer = process(['-n', '-f', 'rawvideo', '-pixel_format', fmt, '-video_size', f"{meta['width']}x{meta['height']}",
            '-framerate', meta['fps'], *colour, '-i', 'pipe:0', '-i', str(source), '-map', '0:v:0', '-map', '1:a?',
            '-c:v', 'ffv1', '-pix_fmt', fmt, *colour, '-c:a', 'copy', str(output)], stdin=subprocess.PIPE)
        capture = cv2.VideoCapture(str(mask_path))
        stack.callback(capture.release)
        expected = hashlib.sha256()
        rows = []
        for frame in range(meta['frames']):
            ok, mask = capture.read()
            if not ok:
                raise ValueError('Effective mask ended early.')
            raw, counts = composite_frame(read_exact(base.stdout, size), read_exact(replacement.stdout, size), mask[:, :, 0] > 127, meta)
            writer.stdin.write(raw)
            expected.update(raw)
            rows.append({'frame': frame, **{f'plane{i}_{key}': value for i, c in enumerate(counts) for key,value in c.items()}})
        if base.stdout.read(1) or replacement.stdout.read(1) or capture.read()[0]:
            raise ValueError('Extra native/mask frames.')
        writer.stdin.close()
        if any(p.wait() for p in (base, replacement, writer)):
            raise ValueError('Native composition process failed.')
        verified = process(['-i', str(output), *common], stdout=subprocess.PIPE)
        actual = hashlib.sha256()
        while chunk := verified.stdout.read(1024*1024):
            actual.update(chunk)
        if verified.wait() or actual.digest() != expected.digest():
            raise ValueError('Encoded native samples differ from the verified composite.')
        return rows
