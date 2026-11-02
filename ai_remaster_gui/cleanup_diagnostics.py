"""Read existing per-frame Clean Up diagnostics, without executing models."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path


def _clean_stem(name: str) -> str:
    stem = Path(name).stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def cleanup_diagnostics_for(root: Path, source: Path | None,
                            output: Path | None, settings: dict) -> dict:
    """Only show reports belonging to the exact current cleanup artifact."""
    empty = {"ready": False, "deflicker": None, "defect_mask": "", "scratch_mask": ""}
    if not source or not output or not output.is_file() or not source.is_file():
        return empty
    # A fresh input invalidates previously generated reports.
    try:
        if source.stat().st_mtime_ns > output.stat().st_mtime_ns:
            return empty
    except OSError:
        return empty
    result = dict(empty, ready=True)
    if settings.get('detect_defects') == 'true':
        preview = output.with_name(output.stem + '_defect_mask.mp4')
        if preview.is_file():
            result['defect_mask'] = str(preview)
    if settings.get('ai_descratch') == 'true' and settings.get('save_scratch_mask') == 'true':
        preview = output.with_name(output.stem + '_scratch_mask.mp4')
        if preview.is_file():
            result['scratch_mask'] = str(preview)
    # Only ARP Conservative produces meaningful per-frame gain factors.
    # FFmpeg Median/Arithmetic Mean instead use direct pixel-difference charts.
    if settings.get('light_deflicker') != 'true' or settings.get('deflicker_method', 'arp') != 'arp':
        return result
    match = re.fullmatch(r'.+_cleanup_([a-f0-9]{8})', output.stem)
    if not match:
        return result
    workdir = root / '.cache' / 'cleanup_chunks' / f'{_clean_stem(source.name)}_{match.group(1)}'
    if not workdir.is_dir():
        return result
    try:
        strength = float(settings.get('deflicker_strength', '0.65') or 0.65)
        limit = float(settings.get('deflicker_max_correction', '0.08') or 0.08)
    except (ValueError, TypeError):
        return result
    try:
        reports = sorted(workdir.glob('light_deflicker_*_analysis.json'),
                         key=lambda p: p.stat().st_mtime_ns, reverse=True)
    except OSError:
        return result
    for path in reports[:30]:
        try:
            if path.stat().st_mtime_ns > output.stat().st_mtime_ns + 2_000_000_000:
                continue
            data = json.loads(path.read_text(encoding='utf-8'))
            if Path(data.get('source', '')).resolve() != source.resolve():
                continue
            if abs(float(data.get('strength', -1)) - strength) > 1e-7:
                continue
            if abs(float(data.get('max_correction', -1)) - limit) > 1e-7:
                continue
            gains = data.get('gains')
            if not isinstance(gains, list) or not gains or len(gains) > 2_000_000:
                continue
            if int(data.get('frames', -1)) != len(gains):
                continue
            if not all(isinstance(g, (int, float)) and math.isfinite(g)
                       and 0.80 <= g <= 1.20 for g in gains):
                continue
            total = len(gains)
            step = max(1, math.ceil(total / 240))
            samples = []
            for start in range(0, total, step):
                end = min(total, start + step)
                at = max(range(start, end), key=lambda k: abs(gains[k] - 1.0))
                samples.append({'frame': at, 'gain': round(gains[at], 7)})
            result['deflicker'] = {
                'frames': total,
                'corrected': sum(abs(g - 1.0) >= 1e-4 for g in gains),
                'max_gain_change_pct': round(100 * max(abs(g - 1) for g in gains), 3),
                'mean_abs_gain_change_pct': round(100 * sum(abs(g - 1) for g in gains) / total, 3),
                'strength': strength,
                'limit_pct': round(100 * limit, 2),
                'cut_boundaries': max(0, len(data.get('cuts', [])) - 1),
                'samples': samples,
                'sampled': step > 1,
            }
            return result
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
    return result
