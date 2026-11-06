"""Resolve the managed backend without installing, downloading or changing settings."""
from pathlib import Path
import sys
from .config import ROOT, SCRIPTS


def installation(values, root=ROOT):
    repo, python = (str(values.get(key,'')).strip() for key in ('dart_repo','dart_python'))
    if bool(repo) != bool(python):
        raise ValueError('Specify both external DART paths, or leave both blank for the ARP-managed installation.')
    if repo:
        return repo, python
    managed = Path(root)/'tools/dart'
    return str(managed), str(managed/'.venv/Scripts/python.exe')


def require_installation(values, root=ROOT):
    repo, python = installation(values,root)
    if not (Path(repo)/'scripts/inference.py').is_file() or not Path(python).is_file():
        raise ValueError('DART is not set up. Run install_windows.bat -DartOnly (no ComfyUI changes). No automatic install was attempted.')
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from dust_model_preflight import inspect_dart, require_ready
    return require_ready(inspect_dart(repo,python,probe_runtime=False))
