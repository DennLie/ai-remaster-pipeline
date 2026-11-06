"""Read-only model readiness checks. Never downloads, starts a server or queues work.

Hashes inventory installed files; only DART has independently pinned expected
bytes here. Presence/hash reporting does not certify model quality or rights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from comfy_api import http_json
from run_dart_research import DART_COMMIT, DART_WEIGHTS_SHA256, validate_checkout

PROPAINTER_WEIGHTS = ('raft-things.pth', 'recurrent_flow_completion.pth', 'ProPainter.pth')
PROPAINTER_NODES = ('VHS_LoadVideoPath', 'ImageToMask', 'ProPainterInpaint', 'VHS_VideoCombine')


def inventory(path):
    path = Path(path)
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f'Required model file missing/empty: {path}. No automatic download was attempted.')
    with path.open('rb') as f:
        sha = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha}


def inspect_dart(repo, python, *, probe_runtime=True):
    repo, python = Path(repo).resolve(), Path(python).resolve()
    result = {'backend': 'dart', 'ready': False, 'blockers': [], 'weights': [],
              'licence': 'research only; unresolved adapted RTN rights', 'expected_commit': DART_COMMIT}
    try:
        validate_checkout(repo, repo/'pretrained_models/dart.safetensors')
        dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if dirty.strip():
            raise ValueError('DART tracked checkout has modifications; pinned HEAD alone is insufficient.')
        if not python.is_file():
            raise ValueError('External DART Python executable missing.')
        result['weights'] = [{'path': str(repo/'pretrained_models/dart.safetensors'), 'sha256': DART_WEIGHTS_SHA256,
                              'bytes': (repo/'pretrained_models/dart.safetensors').stat().st_size}]
        if probe_runtime:
            code = ("import json,sys,importlib.util,torch;print(json.dumps({'python':sys.version.split()[0],"
                    "'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),"
                    "'missing_modules':[m for m in ['albumentations','hydra','safetensors','cv2','timm','pytorch_lightning','torchvision'] "
                    "if importlib.util.find_spec(m) is None]}))")
            runtime = json.loads(subprocess.check_output([str(python), '-I', '-c', code], text=True, timeout=45))
            result['runtime'] = runtime
            if runtime['missing_modules'] or not runtime['cuda_available']:
                raise ValueError('DART GPU runtime is incomplete or CUDA unavailable; no silent CPU inference fallback.')
        result['ready'] = True
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result['blockers'].append(str(exc))
    return result


def inspect_propainter(comfy_dir, url):
    node = Path(comfy_dir).resolve()/'custom_nodes/ComfyUI_ProPainter_Nodes'
    result = {'backend': 'propainter', 'ready': False, 'blockers': [], 'weights': [],
              'licence': 'non-commercial; no commercial clearance', 'comfy_url': url}
    try:
        result['weights'] = [inventory(node/'weights'/name) for name in PROPAINTER_WEIGHTS]
        result['node_commit'] = subprocess.check_output(['git','-C',str(node),'rev-parse','HEAD'],text=True).strip()
        result['node_tracked_dirty'] = bool(subprocess.check_output(['git','-C',str(node),'status','--porcelain','--untracked-files=no'],text=True).strip())
        queue = http_json('GET', url.rstrip('/')+'/queue', timeout=5)
        result['queue'] = {k: len(queue.get(k, [])) for k in ('queue_running','queue_pending')}
        if any(result['queue'].values()):
            raise ValueError('ComfyUI has running/pending work. Do not enqueue validation into another job queue.')
        info = http_json('GET', url.rstrip('/')+'/object_info', timeout=10)
        missing = [name for name in PROPAINTER_NODES if name not in info]
        if missing:
            raise ValueError('Required live ComfyUI nodes missing: '+', '.join(missing))
        required = info['ProPainterInpaint'].get('input', {}).get('required', {})
        fields = ('image','mask','width','height','mask_dilates','flow_mask_dilates','ref_stride','neighbor_length','subvideo_length','raft_iter','fp16')
        if any(key not in required for key in fields):
            raise ValueError('Installed ProPainter node schema differs from the adapter contract.')
        result['ready'] = True
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        result['blockers'].append(str(exc))
    return result


def require_ready(report):
    if not report['ready']:
        raise ValueError('Model preflight blocked '+report['backend']+': '+'; '.join(report['blockers']))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dart-repo', type=Path, required=True)
    p.add_argument('--dart-python', type=Path, required=True)
    p.add_argument('--comfy-dir', type=Path, required=True)
    p.add_argument('--comfy-url', default='http://127.0.0.1:8188')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    report = {'scope':'read-only prerequisite check; NOT GPU reservation, inference or quality validation',
        'dart': inspect_dart(args.dart_repo,args.dart_python),
        'propainter': inspect_propainter(args.comfy_dir,args.comfy_url)}
    text = json.dumps(report,indent=2)
    if args.output:
        with args.output.open('x',encoding='utf-8') as f:
            f.write(text+'\n')
    print(text)


if __name__ == '__main__':
    main()
