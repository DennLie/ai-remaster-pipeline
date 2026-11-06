"""Explicit opt-in installer for ARP's isolated, pinned DART research backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

from run_dart_research import DART_COMMIT, DART_WEIGHTS_SHA256

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer.git'
UV_VERSION = '0.13.6'
PYTHON_VERSION = '3.12.9'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command, **kwargs):
    subprocess.run([str(arg) for arg in command], check=True, **kwargs)


def checked_checkout(repo):
    revision = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if revision != DART_COMMIT:
        raise ValueError(f'DART checkout differs from the ARP pin: {revision}. No update/reset was attempted.')
    dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],text=True)
    if dirty.strip():
        raise ValueError('DART tracked files have changes. Refusing to modify this checkout.')


def install_weights(repo, copy_from=None, skip=False):
    target = repo/'pretrained_models/dart.safetensors'
    if not target.parent.resolve().is_relative_to(repo.resolve()) or target.is_symlink():
        raise ValueError('Managed DART weights must not redirect to an external installation.')
    if target.exists():
        if digest(target) != DART_WEIGHTS_SHA256:
            raise ValueError('Existing DART weights have a different checksum; not overwritten.')
        return True
    if skip and not copy_from:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.dart-weights-',dir=target.parent) as temporary:
        staged = Path(temporary)/'dart.safetensors'
        if copy_from:
            if digest(copy_from) != DART_WEIGHTS_SHA256:
                raise ValueError('Supplied DART weights do not match the pinned SHA-256.')
            shutil.copyfile(copy_from, staged)
        else:
            url = ('https://github.com/TytanMikJas/DART-Degradation-Aware-Recurrent-Transformer/'
                   'releases/download/v1.0.0/dart.safetensors')
            with urllib.request.urlopen(url, timeout=60) as response, staged.open('wb') as stream:
                shutil.copyfileobj(response, stream)
        if digest(staged) != DART_WEIGHTS_SHA256:
            raise ValueError('Downloaded DART weights failed SHA-256 verification; not published.')
        if target.exists():
            raise ValueError('DART weights appeared during setup; refusing to overwrite them.')
        staged.rename(target)
    return True


def uv_command(root):
    binary = shutil.which('uv')
    if binary:
        version = subprocess.check_output([binary,'--version'],text=True).split()
        if len(version)>1 and version[1] == UV_VERSION:
            return [binary]
    # Installer tooling is isolated as well: never pip-install into ARP/ComfyUI.
    bootstrap = root/'tools/dart-installer'
    if not bootstrap.resolve().is_relative_to((root/'tools').resolve()) or not (root/'tools').resolve().is_relative_to(root):
        raise ValueError('DART installer tooling redirects outside the managed tools directory.')
    python = bootstrap/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if os.name == 'nt' and not python.resolve().is_relative_to(bootstrap.resolve()):
        raise ValueError('DART installer Python redirects outside the managed tooling directory.')
    if not python.exists():
        run([sys.executable,'-I','-m','venv',bootstrap])
    run([python,'-I','-m','pip','--isolated','--require-virtualenv','install',f'uv=={UV_VERSION}'])
    return [python,'-I','-m','uv']


def install(root=ROOT, *, acknowledged=False, weights_from=None, skip_weights=False):
    if not acknowledged:
        raise ValueError('DART is optional research software with unresolved adapted RTN rights. Explicit acknowledgement required.')
    root = Path(root).resolve()
    repo = root/'tools/dart'
    repo.parent.mkdir(parents=True, exist_ok=True)
    if not repo.parent.resolve().is_relative_to(root) or not repo.resolve().is_relative_to(repo.parent.resolve()):
        raise ValueError('Managed DART path redirects to an external checkout; not modified.')
    for name in ('.venv','.python'):
        if not (repo/name).resolve().is_relative_to(repo.resolve()):
            raise ValueError('DART runtime path redirects outside its managed checkout; not modified.')
    # The OS releases this lock on interruption; concurrent setup must not race sync/publication.
    from ai_remaster_gui.dust_jobs import candidate_lock
    with candidate_lock(repo.parent/'dart-setup'):
        if not repo.exists():
            with tempfile.TemporaryDirectory(prefix='.dart-checkout-',dir=repo.parent) as temporary:
                staged = Path(temporary)/'checkout'
                run(['git','clone','--no-checkout',REPOSITORY,staged])
                run(['git','-C',staged,'checkout','--detach',DART_COMMIT])
                checked_checkout(staged)
                staged.rename(repo)
        checked_checkout(repo)
        uv = uv_command(root)
        # Inherited uv/active-environment overrides must not select another project's runtime.
        env = {key:value for key,value in os.environ.items()
               if not key.upper().startswith('UV_') and key.upper() not in ('VIRTUAL_ENV','PYTHONHOME','PYTHONPATH')}
        env.update({'UV_PYTHON_INSTALL_DIR':str(repo/'.python'),
                    'UV_PROJECT_ENVIRONMENT':str(repo/'.venv'), 'UV_PYTHON_DOWNLOADS':'automatic'})
        run([*uv,'python','install',PYTHON_VERSION,'--no-bin','--no-registry'],env=env,cwd=repo)
        managed_python = Path(subprocess.check_output(
            [*uv,'python','find','--system','--no-project','--managed-python','--no-python-downloads',PYTHON_VERSION],
            env=env,cwd=repo,text=True).strip())
        if not managed_python.resolve().is_relative_to((repo/'.python').resolve()):
            raise ValueError('DART Python must come from its own managed installation.')
        run([*uv,'sync','--project',repo,'--python',managed_python,'--frozen','--no-dev'],env=env,cwd=repo)
        ready = install_weights(repo, weights_from, skip_weights)
        python = repo/('.venv/Scripts/python.exe' if os.name=='nt' else '.venv/bin/python')
        if ready:
            from dust_model_preflight import inspect_dart, require_ready
            require_ready(inspect_dart(repo, python))
        print(json.dumps({'dart_repo':str(repo),'dart_python':str(python),'weights_ready':ready,
                          'commit':DART_COMMIT,'research_only':True},indent=2))
        return repo, python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--acknowledge-research',action='store_true')
    parser.add_argument('--weights-from',type=Path,help='Copy verified existing weights; never alter their source.')
    parser.add_argument('--skip-weights',action='store_true')
    args = parser.parse_args()
    install(acknowledged=args.acknowledge_research,weights_from=args.weights_from,skip_weights=args.skip_weights)


if __name__=='__main__':
    sys.path.insert(0,str(ROOT))
    main()
