"""Pinned, model-only native-resolution inference. No downloads or CPU fallback."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import runpy
import subprocess
import sys
import types

from common import ROOT

MODEL_ROOT = ROOT / '.cache' / 'detail_deblur'
MODELS = {
    'nafnet': ('NAFNet', '2b4af71ebe098a92a75910c233a3965a3e93ede4', 'NAFNet-GoPro-width64.pth',
               '329d3ab4077b8d6b7ff61de376e483714667960bf85be027bf4335cda701196f'),
    'restormer': ('Restormer', '68dc6ac472db26f16361150cb7a96a1bc87da93f', 'Restormer-GoPro-motion_deblurring.pth',
                  '194e38fb5b607c9dc5a5b3e08e65b2e79ee2bf0ef5048e0612f6b2ff2f79da31'),
}


def model_identity(method: str) -> dict:
    name, revision, filename, digest = MODELS[method]
    return {'name': name, 'revision': revision, 'checkpoint': filename, 'sha256': digest}


def load_model(method: str):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for neural Deblur; no implicit CPU fallback.')
    name, revision, filename, digest = MODELS[method]
    repo, weights = MODEL_ROOT / 'code' / name, MODEL_ROOT / 'models' / filename
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if head != revision or dirty:
        raise ValueError(f'{name} needs its clean, pinned model checkout.')
    with weights.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != digest:
        raise ValueError(f'{name} checkpoint SHA-256 mismatch.')
    loaded = torch.load(weights, map_location='cpu', weights_only=True)
    if not isinstance(loaded, dict):
        raise ValueError('Expected a tensor checkpoint dictionary.')
    params = loaded.get('params', loaded.get('params_ema', loaded))
    if not isinstance(params, dict) or not params or any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in params.items()):
        raise ValueError('Checkpoint contains invalid/non-tensor parameters.')
    if method == 'nafnet':
        # Inference does not need BasicSR's training registry, LMDB or dataset plugins.
        existing = sys.modules.get('basicsr')
        if existing is not None and list(getattr(existing, '__path__', [])) != [str(repo / 'basicsr')]:
            raise RuntimeError('NAFNet cannot share a worker with another BasicSR installation.')
        for package in ('basicsr', 'basicsr.models', 'basicsr.models.archs', 'basicsr.utils'):
            if package in sys.modules:
                continue
            module = types.ModuleType(package)
            module.__path__ = [str(repo.joinpath(*package.split('.')))]
            sys.modules[package] = module
        logger = importlib.import_module('basicsr.utils.logger')
        sys.modules['basicsr.utils'].get_root_logger = logger.get_root_logger
        arch = importlib.import_module('basicsr.models.archs.NAFNet_arch')
        model = arch.NAFNetLocal(width=64, enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])
    else:
        arch = runpy.run_path(str(repo / 'basicsr/models/archs/restormer_arch.py'))
        model = arch['Restormer'](inp_channels=3, out_channels=3, dim=48,
            num_blocks=[4, 6, 6, 8], num_refinement_blocks=4, heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', dual_pixel_task=False)
    model.load_state_dict({k.removeprefix('module.'): v for k, v in params.items()}, strict=True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return model.eval().to('cuda')


def restore_rgb(model, rgb, *, method: str, mix: int, device: str = 'cuda'):
    import numpy as np
    import torch
    import torch.nn.functional as F
    if method not in MODELS or type(mix) is not int or not 0 <= mix <= 100:
        raise ValueError('Invalid neural Deblur parameters.')
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Deblur input must be RGB8.')
    if mix == 0:
        return rgb.copy()
    height, width = rgb.shape[:2]
    if min(height, width) < 32:
        raise ValueError('Deblur needs frames at least 32 pixels wide and high.')
    multiple = 16 if method == 'nafnet' else 8
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
    pad = (0, (-width) % multiple, 0, (-height) % multiple)
    with torch.inference_mode():
        predicted = model(F.pad(x, pad, mode='reflect') if any(pad) else x)
        if predicted.ndim != 4 or predicted.shape[:2] != (1, 3) or predicted.shape[2:] != (height + pad[3], width + pad[1]):
            raise ValueError('Deblur model changed frame geometry.')
        predicted = predicted[:, :, :height, :width]
        # Catch catastrophic excursions before clamping can conceal a broken inference.
        if not torch.isfinite(predicted).all() or predicted.abs().max() > 16:
            raise ValueError('Deblur output has a nonfinite or pathological numeric range; result was not published.')
        predicted = predicted.clamp(0, 1) * (mix / 100) + x * (1 - mix / 100)
        return predicted.squeeze(0).permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
