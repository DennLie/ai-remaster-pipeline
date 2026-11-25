"""Managed isolated model environments and pinned subprocess runner under .cache/model_envs/."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .commercial_policy import (
    CommercialPolicyError,
    enabled as commercial_guard_enabled,
    evaluate_component,
)
from .config import ROOT

DEFAULT_MODEL_ENVS_DIR = ROOT / ".cache" / "model_envs"
MANAGED_ENV_ACTIVE_VAR = "ARP_MANAGED_ENV_ACTIVE"
MANAGED_ENV_NAME_VAR = "ARP_MANAGED_ENV_NAME"
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu124"
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
OFFICIAL_SAM2_GIT_URL = "git+https://github.com/facebookresearch/sam2.git"

_STRIPPED_ENV_KEYS = frozenset({
    "VIRTUAL_ENV",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONUSERBASE",
    "CONDA_PREFIX",
    "UV_PROJECT_ENVIRONMENT",
    "UV_PYTHON",
})


@dataclass(frozen=True)
class ModelEnvSpec:
    """Descriptor for a conflict-free isolated Python environment group."""

    name: str
    label: str
    description: str
    python_version: str
    supported_python: tuple[str, ...]
    commercial_safe: bool
    models: tuple[str, ...]
    aliases: tuple[str, ...]
    torch_packages: tuple[str, ...]
    packages: tuple[str, ...]
    probe_imports: tuple[str, ...]
    component_ids: tuple[str, ...]


MODEL_ENV_SPECS: dict[str, ModelEnvSpec] = {
    "transformers_seg": ModelEnvSpec(
        name="transformers_seg",
        label="Transformers Segmentation & Matting (OSI)",
        description="Isolated Python env for Florence-2, BiRefNet, Mask2Former, OneFormer, and RF-DETR-Seg.",
        python_version="3.11",
        supported_python=("3.11", "3.12"),
        commercial_safe=True,
        models=("florence2", "birefnet", "mask2former", "oneformer", "rf_detr_seg"),
        aliases=("florence-2", "birefnet-lite", "birefnet_lite", "mask2-former", "one-former", "rf-detr-seg", "rfdetr-seg", "rfdetr"),
        torch_packages=("torch", "torchvision"),
        packages=(
            "transformers>=4.45.0",
            "timm",
            "einops",
            "accelerate",
            "kornia",
            "rfdetr",
            "pillow",
            "numpy",
            "opencv-python-headless",
            "safetensors",
            "huggingface-hub",
        ),
        probe_imports=("transformers", "rfdetr"),
        component_ids=("florence2", "birefnet", "mask2former", "oneformer", "rf_detr_seg"),
    ),
    "sam2_family": ModelEnvSpec(
        name="sam2_family",
        label="Official SAM 2.1 / EdgeTAM / Grounding Ecosystem (OSI)",
        description="Isolated Python env with official facebookresearch/sam2 (never PyPI sam2), EdgeTAM, SAMURAI, EVF-SAM-2, and Grounding DINO.",
        python_version="3.11",
        supported_python=("3.11", "3.12"),
        commercial_safe=True,
        models=("sam21", "sam21_image", "samurai", "edgetam", "evf_sam2", "grounded_sam2", "grounding_dino"),
        aliases=(
            "sam2",
            "sam2.1",
            "sam-2.1",
            "sam21-image",
            "sam2.1-image",
            "samurai-sam21",
            "edge-tam",
            "evf-sam2",
            "evf-sam-2",
            "grounded-sam2",
            "grounded-sam-2",
            "grounding-dino",
            "groundingdino",
        ),
        torch_packages=("torch", "torchvision"),
        packages=(
            OFFICIAL_SAM2_GIT_URL,
            "git+https://github.com/facebookresearch/EdgeTAM.git",
            "transformers>=4.45.0",
            "hydra-core",
            "iopath",
            "pillow",
            "numpy",
            "opencv-python-headless",
            "safetensors",
            "huggingface-hub",
        ),
        probe_imports=("sam2", "edgetam", "transformers"),
        component_ids=("sam2", "sam21_image", "samurai", "edgetam", "evf_sam2", "grounded_sam2", "grounding_dino"),
    ),
    "ddcolor_env": ModelEnvSpec(
        name="ddcolor_env",
        label="DDColor Reference Generator (Apache-2.0)",
        description="Isolated Python 3.10/3.11 env for official piddnad/DDColor and compatible basicsr/timm stack.",
        python_version="3.11",
        supported_python=("3.10", "3.11"),
        commercial_safe=True,
        models=("ddcolor",),
        aliases=("ddcolor_modelscope", "ddcolor_paper_tiny", "ddcolor_artistic"),
        torch_packages=("torch", "torchvision"),
        packages=(
            "git+https://github.com/piddnad/DDColor.git",
            "timm",
            "einops",
            "pillow",
            "numpy",
            "opencv-python-headless",
            "safetensors",
            "huggingface-hub",
        ),
        probe_imports=("ddcolor",),
        component_ids=("ddcolor",),
    ),
    "vos_matting": ModelEnvSpec(
        name="vos_matting",
        label="Cutie VOS & VideoMaMa Matting (OSI)",
        description="Isolated Python env for hkchengrex/Cutie (MIT) and VideoMaMa (Apache-2.0).",
        python_version="3.11",
        supported_python=("3.11", "3.12"),
        commercial_safe=True,
        models=("cutie", "videomama"),
        aliases=("hkchengrex-cutie", "video-mama"),
        torch_packages=("torch", "torchvision"),
        packages=(
            "git+https://github.com/hkchengrex/Cutie.git",
            "git+https://github.com/cvlab-kaist/VideoMaMa.git",
            "hydra-core",
            "omegaconf",
            "scipy",
            "pillow",
            "numpy",
            "opencv-python-headless",
            "safetensors",
        ),
        probe_imports=("cutie", "videomama"),
        component_ids=("cutie", "videomama"),
    ),
    "noncommercial_research": ModelEnvSpec(
        name="noncommercial_research",
        label="SAM 3 / SAM 3.1 Research Only (Non-Commercial)",
        description="Strictly isolated env for Meta SAM 3 & 3.1. Blocked when Commercial Model Guard is ON.",
        python_version="3.11",
        supported_python=("3.11", "3.12"),
        commercial_safe=False,
        models=("sam3", "sam3_image", "sam31", "sam31_image"),
        aliases=("sam-3", "sam3-image", "sam3.1", "sam-3.1", "sam31-image", "sam3.1-image"),
        torch_packages=("torch", "torchvision"),
        packages=(
            "git+https://github.com/facebookresearch/sam3.git",
            "hydra-core",
            "iopath",
            "pillow",
            "numpy",
            "opencv-python-headless",
            "safetensors",
            "huggingface-hub",
        ),
        probe_imports=("sam3",),
        component_ids=("sam3", "sam3.1", "sam3_image", "sam31_image"),
    ),
}

_MODEL_LOOKUP: dict[str, str] = {}
for _env_name, _spec in MODEL_ENV_SPECS.items():
    for _tok in (*_spec.models, *_spec.aliases):
        _norm = _tok.strip().lower()
        _MODEL_LOOKUP[_norm] = _env_name
        _MODEL_LOOKUP[_norm.replace("-", "_")] = _env_name


def get_env_spec(env_name: str) -> ModelEnvSpec:
    """Return the ModelEnvSpec for env_name or raise ValueError."""
    key = str(env_name or "").strip().lower()
    if key not in MODEL_ENV_SPECS:
        valid = ", ".join(MODEL_ENV_SPECS.keys())
        raise ValueError(f"Unknown managed model environment {env_name!r}. Valid environments: {valid}")
    return MODEL_ENV_SPECS[key]


def env_for_model(model_id: str) -> str | None:
    """Resolve a canonical model or segmenter ID (or alias) to its managed env name."""
    raw = str(model_id or "").strip().lower()
    if not raw:
        return None
    return _MODEL_LOOKUP.get(raw) or _MODEL_LOOKUP.get(raw.replace("-", "_"))


def model_to_env_map() -> dict[str, str]:
    """Return a clean mapping of every canonical model ID -> managed env name."""
    mapping: dict[str, str] = {}
    for env_name, spec in MODEL_ENV_SPECS.items():
        for model_id in spec.models:
            mapping[model_id] = env_name
    return mapping


def env_directory(env_name: str, *, base_dir: Path | str | None = None) -> Path:
    """Resolve the managed environment directory and enforce containment under base_dir."""
    spec = get_env_spec(env_name)
    base = Path(base_dir).resolve() if base_dir is not None else DEFAULT_MODEL_ENVS_DIR.resolve()
    target = (base / spec.name).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Managed environment directory {target} escapes base directory {base}.")
    return target


def env_python_executable(env_name: str, *, base_dir: Path | str | None = None) -> Path:
    """Return the expected Python executable path inside the managed env's .venv."""
    env_dir = env_directory(env_name, base_dir=base_dir)
    if os.name == "nt":
        return env_dir / ".venv" / "Scripts" / "python.exe"
    return env_dir / ".venv" / "bin" / "python"


def is_managed_env_active(env: Mapping[str, str] | None = None) -> bool:
    """Return True when the current process is already running inside a managed model env."""
    source = env if env is not None else os.environ
    return str(source.get(MANAGED_ENV_ACTIVE_VAR, "")).strip().lower() in {"1", "true", "yes"}


def check_env_policy(
    env_name: str,
    *,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    allow_non_commercial: bool = False,
) -> tuple[bool, str]:
    """Evaluate commercial policy for installing or invoking a managed model env."""
    spec = get_env_spec(env_name)
    if spec.commercial_safe:
        for cid in spec.component_ids:
            dec = evaluate_component(cid, settings=settings, component_type="tool")
            if not dec.allowed:
                return False, dec.reason
        return True, f"{spec.label} is cleared for commercial use."

    # Non-commercial research environment (SAM 3 / SAM 3.1)
    if settings is not None:
        guard_on = commercial_guard_enabled(settings)
        global_cfg = settings.get("global", {}) if isinstance(settings.get("global"), Mapping) else {}
        override_on = str(global_cfg.get("non_commercial_benchmark_override", "false")).strip().lower() == "true"
        if (not guard_on) and override_on:
            return True, "Non-commercial research environment unlocked via explicit benchmark override."
        return (
            False,
            f"Commercial Model Guard blocked '{spec.name}' (Meta SAM License): disable commercial_model_guard and set non_commercial_benchmark_override='true' (or pass --allow-non-commercial-benchmark) to unlock for benchmark use.",
        )

    if allow_non_commercial:
        return True, "Non-commercial research environment unlocked via explicit --allow-non-commercial-benchmark."
    return (
        False,
        f"Commercial Model Guard blocked '{spec.name}' (Meta SAM License): non-commercial research/benchmark use requires --allow-non-commercial-benchmark or non_commercial_benchmark_override='true'.",
    )


def require_env_policy(
    env_name: str,
    *,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    allow_non_commercial: bool = False,
) -> None:
    """Enforce commercial policy for an environment, raising CommercialPolicyError on block."""
    allowed, reason = check_env_policy(env_name, settings=settings, allow_non_commercial=allow_non_commercial)
    if not allowed:
        raise CommercialPolicyError(reason)


def _manifest_path(env_dir: Path) -> Path:
    return env_dir / "env_manifest.json"


def _write_manifest_atomic(manifest_file: Path, payload: Mapping[str, Any]) -> None:
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_file.parent,
        prefix=".env_manifest_",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(dict(payload), indent=2))
        tmp_path = Path(tmp.name)
    tmp_path.replace(manifest_file)


def _read_manifest(manifest_file: Path) -> dict[str, Any] | None:
    if not manifest_file.is_file():
        return None
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def inspect_env(
    env_name: str,
    *,
    base_dir: Path | str | None = None,
    settings: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Fast non-spawning inspection of a single managed model environment."""
    spec = get_env_spec(env_name)
    env_dir = env_directory(spec.name, base_dir=base_dir)
    py_exe = env_python_executable(spec.name, base_dir=base_dir)
    manifest_file = _manifest_path(env_dir)
    manifest = _read_manifest(manifest_file) or {}
    manifest_status = str(manifest.get("status") or "").strip().lower()
    py_exists = py_exe.is_file()

    if manifest_status == "ready" and py_exists:
        status = "ready"
    elif manifest_status == "error":
        status = "error"
    else:
        status = "missing"

    allow_nc = False
    if settings is None and not spec.commercial_safe:
        # Report raw policy status conservatively as blocked unless settings unlock it
        policy_allowed, policy_reason = check_env_policy(spec.name, settings=None, allow_non_commercial=False)
    else:
        policy_allowed, policy_reason = check_env_policy(spec.name, settings=settings, allow_non_commercial=allow_nc)

    return {
        "env_name": spec.name,
        "label": spec.label,
        "description": spec.description,
        "status": status,
        "ready": status == "ready",
        "commercial_safe": spec.commercial_safe,
        "policy_allowed": policy_allowed,
        "policy_reason": policy_reason,
        "python_version": str(manifest.get("python_version") or spec.python_version),
        "supported_python": list(spec.supported_python),
        "python_executable": str(py_exe),
        "env_dir": str(env_dir),
        "manifest_path": str(manifest_file),
        "models": list(spec.models),
        "packages": list(manifest.get("packages") or [*spec.torch_packages, *spec.packages]),
        "torch_flavor": str(manifest.get("torch_flavor") or "uninstalled"),
        "installer": str(manifest.get("installer") or ""),
        "installed_at": str(manifest.get("installed_at") or ""),
        "error": str(manifest.get("error") or ""),
    }


def inspect_all_envs(
    *,
    base_dir: Path | str | None = None,
    settings: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Inspect all 5 managed model environments without spawning subprocesses."""
    return {
        env_name: inspect_env(env_name, base_dir=base_dir, settings=settings)
        for env_name in MODEL_ENV_SPECS
    }


def _strip_host_venv_from_path(raw_path: str) -> str:
    if not raw_path:
        return ""
    try:
        host_prefix = Path(sys.prefix).resolve()
    except Exception:
        host_prefix = None
    try:
        base_prefix = Path(sys.base_prefix).resolve()
    except Exception:
        base_prefix = None

    kept: list[str] = []
    for entry in raw_path.split(os.pathsep):
        cleaned = entry.strip()
        if not cleaned:
            continue
        if host_prefix is not None and (base_prefix is None or host_prefix != base_prefix):
            try:
                p = Path(cleaned).resolve()
                if p.is_relative_to(host_prefix):
                    continue
            except Exception:
                pass
        kept.append(cleaned)
    return os.pathsep.join(kept)


def _sanitized_install_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = base_env if base_env is not None else os.environ
    env: dict[str, str] = {}
    for k, v in source.items():
        up = k.upper()
        if up in _STRIPPED_ENV_KEYS or up.startswith("UV_"):
            continue
        env[k] = v
    if "PATH" in env:
        env["PATH"] = _strip_host_venv_from_path(env["PATH"])
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def build_sanitized_subprocess_env(
    *,
    repo_root: Path | str | None = None,
    base_env: Mapping[str, str] | None = None,
    env_name: str = "",
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build sanitized subprocess env isolating out main .venv while keeping repo root importable."""
    env = _sanitized_install_env(base_env=base_env)
    resolved_repo = Path(repo_root).resolve() if repo_root is not None else ROOT.resolve()
    if extra_env:
        for k, v in extra_env.items():
            if k.upper() in _STRIPPED_ENV_KEYS or k.upper().startswith("UV_"):
                continue
            env[k] = str(v)
    env["PYTHONPATH"] = str(resolved_repo)
    env[MANAGED_ENV_ACTIVE_VAR] = "1"
    if env_name:
        env[MANAGED_ENV_NAME_VAR] = env_name
    return env


def _build_install_commands(
    spec: ModelEnvSpec,
    venv_dir: Path,
    py_exe: Path,
    *,
    cuda: bool,
    uv_bin: str | None,
) -> tuple[str, list[list[str]]]:
    torch_index = TORCH_CUDA_INDEX_URL if cuda else TORCH_CPU_INDEX_URL
    if uv_bin:
        cmds = [
            [uv_bin, "venv", str(venv_dir), "--python", spec.python_version],
            [uv_bin, "pip", "install", "--python", str(py_exe), "--index-url", torch_index, *spec.torch_packages],
            [uv_bin, "pip", "install", "--python", str(py_exe), *spec.packages],
        ]
        return "uv", cmds

    cmds = [
        [sys.executable, "-m", "venv", str(venv_dir)],
        [str(py_exe), "-m", "pip", "--isolated", "--require-virtualenv", "install", "--index-url", torch_index, *spec.torch_packages],
        [str(py_exe), "-m", "pip", "--isolated", "--require-virtualenv", "install", *spec.packages],
    ]
    return "venv+pip", cmds


def install_env(
    env_name: str,
    *,
    cuda: bool = True,
    allow_non_commercial: bool = False,
    dry_run: bool = False,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    base_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Create and populate an isolated model environment under .cache/model_envs/<env_name>/."""
    spec = get_env_spec(env_name)
    require_env_policy(spec.name, settings=settings, allow_non_commercial=allow_non_commercial)

    env_dir = env_directory(spec.name, base_dir=base_dir)
    venv_dir = env_dir / ".venv"
    py_exe = env_python_executable(spec.name, base_dir=base_dir)
    manifest_file = _manifest_path(env_dir)
    torch_flavor = "cuda" if cuda else "cpu"

    uv_bin = shutil.which("uv")
    installer, commands = _build_install_commands(spec, venv_dir, py_exe, cuda=cuda, uv_bin=uv_bin)

    if dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "env_name": spec.name,
            "installer": installer,
            "python_version": spec.python_version,
            "torch_flavor": torch_flavor,
            "env_dir": str(env_dir),
            "python_executable": str(py_exe),
            "packages": [*spec.torch_packages, *spec.packages],
            "commands": commands,
        }

    env_dir.mkdir(parents=True, exist_ok=True)
    install_env_vars = _sanitized_install_env()
    try:
        for cmd in commands:
            subprocess.run(cmd, check=True, env=install_env_vars, cwd=str(ROOT))
    except Exception as exc:
        err_manifest = {
            "schema_version": 1,
            "env_name": spec.name,
            "status": "error",
            "python_version": spec.python_version,
            "packages": [*spec.torch_packages, *spec.packages],
            "torch_flavor": torch_flavor,
            "installer": installer,
            "commercial_safe": spec.commercial_safe,
            "models": list(spec.models),
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
        _write_manifest_atomic(manifest_file, err_manifest)
        raise

    ready_manifest = {
        "schema_version": 1,
        "env_name": spec.name,
        "status": "ready",
        "python_version": spec.python_version,
        "packages": [*spec.torch_packages, *spec.packages],
        "torch_flavor": torch_flavor,
        "installer": installer,
        "commercial_safe": spec.commercial_safe,
        "models": list(spec.models),
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "error": "",
    }
    _write_manifest_atomic(manifest_file, ready_manifest)
    return inspect_env(spec.name, base_dir=base_dir, settings=settings)


def remove_env(env_name: str, *, base_dir: Path | str | None = None) -> dict[str, Any]:
    """Delete an isolated model environment directory if present."""
    spec = get_env_spec(env_name)
    env_dir = env_directory(spec.name, base_dir=base_dir)
    existed = env_dir.exists()
    if existed:
        shutil.rmtree(env_dir)
    return {
        "ok": True,
        "env_name": spec.name,
        "removed": existed,
        "env_dir": str(env_dir),
        "status": "missing",
    }


def run_in_managed_env(
    env_name: str,
    script_path: Path | str,
    argv: Sequence[str] = (),
    *,
    cwd: Path | str | None = None,
    timeout: float | None = None,
    capture_output: bool = True,
    check: bool = True,
    settings: Mapping[str, Mapping[str, str]] | None = None,
    allow_non_commercial: bool = False,
    base_dir: Path | str | None = None,
    repo_root: Path | str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute script_path inside env_name's isolated Python interpreter with sanitized env."""
    spec = get_env_spec(env_name)
    require_env_policy(spec.name, settings=settings, allow_non_commercial=allow_non_commercial)

    if is_managed_env_active():
        active_name = os.environ.get(MANAGED_ENV_NAME_VAR, spec.name)
        raise RuntimeError(
            f"Recursive managed environment invocation prevented ({MANAGED_ENV_ACTIVE_VAR}=1 inside '{active_name}' while calling '{spec.name}')."
        )

    info = inspect_env(spec.name, base_dir=base_dir, settings=settings)
    if not info["ready"]:
        raise RuntimeError(
            f"Managed model environment '{spec.name}' is not installed or not ready (status: {info['status']}). "
            f"Run 'python scripts/setup_model_env.py install {spec.name}' or click the 'Install Environment' button in the GUI."
        )

    py_exe = Path(info["python_executable"])
    resolved_script = Path(script_path)
    if not resolved_script.is_absolute():
        resolved_script = (Path(repo_root) if repo_root else ROOT) / resolved_script

    cmd = [str(py_exe), "-u", str(resolved_script), *(str(a) for a in argv)]
    sub_env = build_sanitized_subprocess_env(
        repo_root=repo_root,
        env_name=spec.name,
        extra_env=extra_env,
    )
    run_cwd = str(Path(cwd).resolve() if cwd is not None else ROOT.resolve())

    proc = subprocess.run(
        cmd,
        cwd=run_cwd,
        env=sub_env,
        capture_output=capture_output,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            f"Managed env '{spec.name}' failed for {resolved_script.name} (exit {proc.returncode}): {detail or 'unknown error'}"
        )
    return proc


__all__ = [
    "DEFAULT_MODEL_ENVS_DIR",
    "MANAGED_ENV_ACTIVE_VAR",
    "MANAGED_ENV_NAME_VAR",
    "MODEL_ENV_SPECS",
    "ModelEnvSpec",
    "OFFICIAL_SAM2_GIT_URL",
    "TORCH_CPU_INDEX_URL",
    "TORCH_CUDA_INDEX_URL",
    "build_sanitized_subprocess_env",
    "check_env_policy",
    "env_directory",
    "env_for_model",
    "env_python_executable",
    "get_env_spec",
    "inspect_all_envs",
    "inspect_env",
    "install_env",
    "is_managed_env_active",
    "model_to_env_map",
    "remove_env",
    "require_env_policy",
    "run_in_managed_env",
]
