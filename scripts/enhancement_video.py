"""Native 1x Detail Enhancement / Deblur, separate from legacy Detail."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from common import ROOT, file_fingerprint, find_ffmpeg, resumable_output, root_relative, signature_path, write_signature
from intermediate_video import canonical_profile, codec_args, container_args
import detail_enhance_video as classical
from deblur_models import MODELS, load_model, model_identity, restore_rgb

VERSION = 1
METHODS = ('unsharp', 'cas', 'multiscale', 'richardson_lucy', 'nafnet', 'restormer')


def options(values: dict) -> dict:
    method = values.get('method', 'off')
    preset = values.get('preset', 'gentle')
    raw_mix = values.get('mix', 100)
    mix = int(raw_mix)
    if isinstance(raw_mix, bool) or isinstance(raw_mix, float) and raw_mix != mix:
        raise ValueError('Enhanced picture mix must be an integer percentage.')
    if method not in ('off', *METHODS):
        raise ValueError('Choose a supported Detail Enhancement / Deblur method.')
    classical.validate('off' if method in MODELS else method, preset, mix)
    result = {'method': method, 'mix': mix}
    if method not in MODELS:
        result['preset'] = preset
    else:
        result['model'] = model_identity(method)
    return result


def output_for(source: Path, values: dict, profile: str = 'high') -> Path:
    selected = options(values)
    identity = [VERSION, str(source.resolve()), selected, canonical_profile(profile)]
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    from artifact_ids import source_word
    return ROOT / 'intermediate' / 'enhanced' / f"{source_word(source.name)}_enhanced_{selected['method']}_{token}.mkv"


def neural_video(source, target, selected, profile, ffmpeg):
    import numpy as np
    from fractions import Fraction
    info = classical.probe(ffmpeg, source)
    width, height = int(info['width']), int(info['height'])
    probe = str(Path(ffmpeg).with_name('ffprobe.exe' if Path(ffmpeg).suffix.lower() == '.exe' else 'ffprobe'))
    timing = json.loads(subprocess.check_output([probe, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,r_frame_rate', '-of', 'json', str(source)], text=True))['streams'][0]
    if Fraction(timing['avg_frame_rate']) != Fraction(timing['r_frame_rate']):
        raise ValueError('Neural Deblur currently requires constant-frame-rate input.')
    # The full worker is streaming: no all-film RGB array or PNG frame directory.
    model = load_model(selected['method'])
    decode = [ffmpeg, '-v', 'error', '-nostdin', '-i', str(source), '-map', '0:v:0', '-an',
              '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    encode = [ffmpeg, '-v', 'error', '-nostdin', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
              '-s', f'{width}x{height}', '-r', info['avg_frame_rate'], '-i', 'pipe:0', '-i', str(source),
              '-map', '0:v:0', '-map', '1:a?', *codec_args(profile, fast=True), '-c:a', 'copy',
              *container_args(str(target), profile), str(target)]
    frames = 0
    with subprocess.Popen(decode, stdout=subprocess.PIPE) as decoder, subprocess.Popen(encode, stdin=subprocess.PIPE) as encoder:
        try:
            while True:
                raw = bytearray()
                while len(raw) < width * height * 3:
                    packet = decoder.stdout.read(width * height * 3 - len(raw))
                    if not packet:
                        break
                    raw.extend(packet)
                if not raw:
                    break
                if len(raw) != width * height * 3:
                    raise ValueError('Incomplete RGB input frame.')
                rgb = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                restored = restore_rgb(model, rgb, method=selected['method'], mix=selected['mix'])
                encoder.stdin.write(restored.tobytes())
                frames += 1
                if frames % 24 == 0:
                    print(f'Detail Enhancement / Deblur: {frames} frames', flush=True)
            encoder.stdin.close()
            decoder.stdout.close()
            if decoder.wait() or encoder.wait() or not frames:
                raise RuntimeError('Deblur decoder or encoder failed.')
        finally:
            for process in (decoder, encoder):
                if process.poll() is None:
                    process.kill()
                    process.wait()
    tool = str(Path(ffmpeg).with_name('ffprobe.exe' if Path(ffmpeg).suffix.lower() == '.exe' else 'ffprobe'))
    count = subprocess.check_output([tool, '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames', '-of', 'csv=p=0', str(target)], text=True).strip()
    after = classical.probe(ffmpeg, target)
    if int(count) != frames or any(after[k] != info[k] for k in ('width', 'height')) or Fraction(after['avg_frame_rate']) != Fraction(info['avg_frame_rate']):
        raise ValueError('Deblur output lost frame mapping or dimensions.')


def run(source: Path, values: dict, profile: str = 'high', *, output: Path | None = None, force=False, dry_run=False):
    source = source.resolve()
    selected, profile = options(values), canonical_profile(profile)
    if not source.is_file():
        raise FileNotFoundError('Enhancement input missing. Run the selected upstream stage first.')
    output = output or output_for(source, values, profile)
    if output.resolve() == source or output.suffix.lower() != '.mkv':
        raise ValueError('Enhancement needs a separate MKV output.')
    sig = {'tool': 'enhancement_video.py', 'version': VERSION, 'source': root_relative(source),
           'source_fingerprint': file_fingerprint(source), 'parameters': selected,
           'method': selected['method'], 'intermediate_profile': profile}
    if not force and resumable_output(output, sig, video_like=source):
        if not dry_run:
            write_signature(output, sig)
        print(f'Reuse Detail Enhancement / Deblur output: {output}', flush=True)
        return output
    if dry_run:
        print(f'Would enhance {source} -> {output}: {selected}', flush=True)
        return output
    ffmpeg = find_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.stem + '.partial.mkv')
    try:
        if selected['method'] in MODELS and selected['mix']:
            neural_video(source, partial, selected, profile, ffmpeg)
        else:
            classical.run(source, partial, method='off' if selected['method'] in MODELS else selected['method'],
                preset=values.get('preset', 'gentle'), mix=selected['mix'], intermediate_profile=profile, ffmpeg=ffmpeg, force=True)
        signature_path(output).unlink(missing_ok=True)
        partial.replace(output)
        write_signature(output, sig)
        print(f'Wrote Detail Enhancement / Deblur output: {output}', flush=True)
    finally:
        partial.unlink(missing_ok=True)
        signature_path(partial).unlink(missing_ok=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--method', choices=('off', *METHODS), default='off')
    parser.add_argument('--preset', choices=classical.PRESETS, default='gentle')
    parser.add_argument('--mix', type=int, default=100)
    parser.add_argument('--intermediate-profile', default='high')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    run(args.source, vars(args), args.intermediate_profile, output=args.output, force=args.force, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
