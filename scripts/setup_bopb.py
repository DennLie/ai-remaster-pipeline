#!/usr/bin/env python3
"""Automated Setup Manager and Readiness Verifier for BOPB Scratch Detector.

Manages isolated virtual environment creation, dependency installation, and official
checkpoint download/verification under tools/checkpoints/bopb/ with strict SHA-256 validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.common import ensure_inherited_permissions, staging_directory
except ImportError:
    try:
        from common import ensure_inherited_permissions, staging_directory
    except ImportError:
        def ensure_inherited_permissions(p: Path | str) -> None:
            pass

        from contextlib import contextmanager
        @contextmanager
        def staging_directory(prefix: str = ".tmp-", dir: Path | str | None = None):
            tmp = tempfile.mkdtemp(prefix=prefix, dir=dir)
            try:
                yield Path(tmp)
            finally:
                if Path(tmp).exists():
                    shutil.rmtree(tmp, ignore_errors=True)

BOPB_SHA256 = "b2d7ab04e9b3885c6b1991bb7a0b823129dd6e3ac078a9fd059ebd2a7ba59a95"
BOPB_SIZE_BYTES = 451663663
UPSTREAM_REPO = "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life"
PRIMARY_WEIGHTS_URL = (
    "https://huggingface.co/databuzzword/bringing-old-photos-back-to-life/resolve/main/"
    "Global/checkpoints/detection/FT_Epoch_latest.pt"
)
ZIP_WEIGHTS_URL = (
    "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/"
    "global_checkpoints.zip"
)

MANAGED_TOOL_DIR = ROOT / "tools" / "bopb"
MANAGED_WEIGHTS_DIR = ROOT / "tools" / "checkpoints" / "bopb"
MANAGED_WEIGHTS_FILE = MANAGED_WEIGHTS_DIR / "FT_Epoch_latest.pt"
MANIFEST_FILE = ROOT / "tools" / "checkpoints" / "manifests" / "bopb.json"


def sha256_file(path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_venv_python(tool_dir: Path = MANAGED_TOOL_DIR) -> Path | None:
    """Find python executable inside managed tools/bopb/.venv."""
    candidates = [
        tool_dir / ".venv" / "Scripts" / "python.exe",
        tool_dir / ".venv" / "bin" / "python",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def get_weights_path() -> Path | None:
    """Find existing valid BOPB weights checkpoint."""
    candidates = [
        MANAGED_WEIGHTS_FILE,
        MANAGED_TOOL_DIR / "checkpoints" / "detection" / "FT_Epoch_latest.pt",
        ROOT / "tools" / "checkpoints" / "detection" / "FT_Epoch_latest.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def check_environment(tool_dir: Path = MANAGED_TOOL_DIR) -> tuple[bool, str, dict[str, Any]]:
    """Test Python interpreter and package imports inside tools/bopb/.venv."""
    py = find_venv_python(tool_dir)
    if not py:
        return False, "Virtual environment not found", {}

    probe_code = (
        "import sys, json, torch, torchvision, numpy, cv2, PIL; "
        "print(json.dumps({"
        "'torch_version': torch.__version__, "
        "'torch_cuda': torch.cuda.is_available(), "
        "'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0, "
        "'numpy_version': numpy.__version__, "
        "'cv2_version': cv2.__version__, "
        "'python_version': sys.version.split()[0]"
        "}))"
    )
    try:
        proc = subprocess.run(
            [str(py), "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0:
            details = json.loads(proc.stdout.strip())
            return True, "Environment operational", details
        return False, f"Import error: {proc.stderr.strip()[:300]}", {}
    except Exception as exc:
        return False, f"Execution failed: {exc}", {}


def check_readiness(
    tool_dir: Path = MANAGED_TOOL_DIR,
    weights_path: Path | None = None,
    verify_sha: bool = True,
) -> dict[str, Any]:
    """Inspect complete BOPB readiness and return structured diagnostic JSON."""
    issues: list[str] = []

    # 1. Environment check
    py = find_venv_python(tool_dir)
    env_ok, env_msg, env_details = check_environment(tool_dir)
    if not env_ok:
        issues.append(f"Runtime environment issue: {env_msg}")

    # 2. Checkpoint check
    wpath = weights_path or get_weights_path()
    weights_present = bool(wpath and wpath.is_file())
    weights_valid = False
    actual_hash = ""

    if weights_present and wpath:
        if verify_sha:
            try:
                actual_hash = sha256_file(wpath)
                weights_valid = (actual_hash == BOPB_SHA256)
                if not weights_valid:
                    issues.append(
                        f"Weights SHA-256 mismatch: expected {BOPB_SHA256}, got {actual_hash}"
                    )
            except Exception as exc:
                issues.append(f"Failed to hash weights at {wpath}: {exc}")
        else:
            weights_valid = True
    else:
        issues.append("BOPB checkpoint (FT_Epoch_latest.pt) not found")

    # 3. Manifest check
    manifest_present = MANIFEST_FILE.is_file()
    if not manifest_present:
        issues.append(f"Manifest missing at {MANIFEST_FILE}")

    # Determine readiness state
    ready = env_ok and weights_present and weights_valid
    readiness_state = "pretrained_operational" if ready else (
        "training_required" if not weights_present else "blocked"
    )

    return {
        "model": "bopb_scratch_detector",
        "ready": ready,
        "readiness_state": readiness_state,
        "commercial_cleared": False,
        "license": "Microsoft Research / Non-Commercial",
        "tool_dir": str(tool_dir),
        "venv_python": str(py) if py else None,
        "environment_valid": env_ok,
        "environment_message": env_msg,
        "environment_details": env_details,
        "weights_path": str(wpath) if wpath else None,
        "weights_present": weights_present,
        "weights_valid": weights_valid,
        "weights_sha256": actual_hash,
        "expected_sha256": BOPB_SHA256,
        "manifest_path": str(MANIFEST_FILE),
        "manifest_present": manifest_present,
        "issues": issues,
    }


def download_weights(
    target_path: Path = MANAGED_WEIGHTS_FILE,
    copy_from: Path | None = None,
    force: bool = False,
) -> Path:
    """Download or copy BOPB weights with SHA-256 verification and atomic publishing."""
    target_path = target_path.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.is_file() and not force:
        existing_hash = sha256_file(target_path)
        if existing_hash == BOPB_SHA256:
            return target_path
        print(f"Existing weights hash mismatch ({existing_hash[:12]}...). Re-downloading.")

    with staging_directory(prefix=".bopb-weights-", dir=target_path.parent) as tmp_dir:
        staged_file = tmp_dir / "FT_Epoch_latest.pt"

        if copy_from:
            src = Path(copy_from).resolve()
            if not src.is_file():
                raise FileNotFoundError(f"Source weights file not found: {src}")
            src_hash = sha256_file(src)
            if src_hash != BOPB_SHA256:
                raise ValueError(
                    f"Supplied weights hash mismatch: expected {BOPB_SHA256}, got {src_hash}"
                )
            print(f"Copying verified weights from {src}...")
            shutil.copyfile(src, staged_file)
        else:
            print(f"Downloading BOPB scratch weights from {PRIMARY_WEIGHTS_URL}...")
            req = urllib.request.Request(
                PRIMARY_WEIGHTS_URL,
                headers={"User-Agent": "ARP-BOPB-Installer/1.0"},
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, staged_file.open("wb") as out:
                    shutil.copyfileobj(resp, out)
            except Exception as dl_err:
                print(f"Direct download failed ({dl_err}). Attempting release zip extraction...")
                zip_staged = tmp_dir / "checkpoints.zip"
                with urllib.request.urlopen(ZIP_WEIGHTS_URL, timeout=180) as resp, zip_staged.open("wb") as out:
                    shutil.copyfileobj(resp, out)
                with zipfile.ZipFile(zip_staged, "r") as zf:
                    matching = [name for name in zf.namelist() if name.endswith("FT_Epoch_latest.pt")]
                    if not matching:
                        raise FileNotFoundError("FT_Epoch_latest.pt not found inside release zip archive.")
                    with zf.open(matching[0]) as zf_stream, staged_file.open("wb") as out:
                        shutil.copyfileobj(zf_stream, out)

        actual_hash = sha256_file(staged_file)
        if actual_hash != BOPB_SHA256:
            raise ValueError(
                f"Downloaded BOPB weights failed SHA-256 verification!\n"
                f"Expected: {BOPB_SHA256}\n"
                f"Actual:   {actual_hash}"
            )

        if target_path.exists():
            target_path.unlink()
        staged_file.rename(target_path)
        ensure_inherited_permissions(target_path)
        print(f"BOPB checkpoint successfully published at {target_path}")

    return target_path


def create_virtualenv(tool_dir: Path = MANAGED_TOOL_DIR) -> Path:
    """Create isolated virtual environment under tool_dir/.venv with CUDA compatibility."""
    tool_dir = tool_dir.resolve()
    tool_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = tool_dir / ".venv"

    # Prefer python with CUDA compatibility (Python 3.10 or 3.11 preferred over 3.13 for prebuilt CUDA wheels)
    py_candidates = [
        Path(r"C:\Users\ogn81\AppData\Local\Programs\Python\Python310\python.exe"),
        Path(r"C:\Python312\python.exe"),
        Path(sys.executable),
    ]
    py_target = None
    for c in py_candidates:
        if c.is_file():
            py_target = c
            break

    uv_bin = shutil.which("uv")
    if uv_bin and py_target:
        print(f"Creating isolated virtualenv with uv ({py_target}) at {venv_dir}...")
        cmd = [uv_bin, "venv", "--python", str(py_target), "--system-site-packages", str(venv_dir)]
        subprocess.run(cmd, check=True)
    elif py_target:
        print(f"Creating isolated virtualenv with {py_target} at {venv_dir}...")
        subprocess.run([str(py_target), "-m", "venv", "--system-site-packages", str(venv_dir)], check=True)
    else:
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)], check=True)

    py = find_venv_python(tool_dir)
    if not py:
        raise RuntimeError(f"Virtualenv created but Python binary not found in {venv_dir}")
    return py


def install_dependencies(py: Path, use_cuda: bool = True) -> None:
    """Install minimal required dependencies into isolated BOPB environment."""
    print("Installing BOPB dependencies into isolated environment...")
    uv_bin = shutil.which("uv")

    packages = ["numpy", "pillow", "opencv-python"]
    torch_packages = ["torch", "torchvision"]

    if uv_bin:
        # Install base packages
        subprocess.run(
            [uv_bin, "pip", "install", "--python", str(py), *packages],
            check=True,
        )
        # Install PyTorch
        torch_cmd = [uv_bin, "pip", "install", "--python", str(py), *torch_packages]
        if use_cuda:
            torch_cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu126"])
        try:
            subprocess.run(torch_cmd, check=True)
        except subprocess.CalledProcessError:
            print("CUDA wheel install failed or not available; installing standard PyTorch build...")
            subprocess.run([uv_bin, "pip", "install", "--python", str(py), *torch_packages], check=True)
    else:
        pip_cmd = [str(py), "-m", "pip", "install", "--isolated", "--no-warn-script-location"]
        subprocess.run([*pip_cmd, *packages], check=True)
        torch_cmd = [*pip_cmd, *torch_packages]
        if use_cuda:
            torch_cmd.extend(["--extra-index-url", "https://download.pytorch.org/whl/cu126"])
        try:
            subprocess.run(torch_cmd, check=True)
        except subprocess.CalledProcessError:
            print("CUDA wheel install failed; falling back to standard PyTorch...")
            subprocess.run([*pip_cmd, *torch_packages], check=True)


def clean(tool_dir: Path = MANAGED_TOOL_DIR, delete_weights: bool = False) -> None:
    """Non-destructively remove tools/bopb/.venv and temporary caches."""
    venv_dir = tool_dir / ".venv"
    if venv_dir.is_dir():
        print(f"Removing virtual environment at {venv_dir}...")
        shutil.rmtree(venv_dir, ignore_errors=True)

    for item in tool_dir.glob(".tmp*"):
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        elif item.is_file():
            item.unlink(missing_ok=True)

    if delete_weights and MANAGED_WEIGHTS_FILE.is_file():
        print(f"Removing weights checkpoint at {MANAGED_WEIGHTS_FILE}...")
        MANAGED_WEIGHTS_FILE.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--check", action="store_true", help="Non-destructive status check returning JSON readiness")
    parser.add_argument("--download-weights", action="store_true", help="Download and verify FT_Epoch_latest.pt checkpoint")
    parser.add_argument("--weights-from", type=Path, default=None, help="Copy verified existing weights checkpoint from local path")
    parser.add_argument("--install", action="store_true", help="Perform full setup: virtualenv creation, dependencies, weights")
    parser.add_argument("--skip-weights", action="store_true", help="Skip checkpoint download during installation")
    parser.add_argument("--clean", action="store_true", help="Clean virtual environment and temporary staging directories")
    parser.add_argument("--delete-weights", action="store_true", help="Also delete weights checkpoint during clean (destructive)")
    parser.add_argument("--force", action="store_true", help="Force re-download / overwrite")
    parser.add_argument("--tool-dir", type=Path, default=MANAGED_TOOL_DIR, help="Target tools/bopb directory")

    args = parser.parse_args(argv)

    if args.clean:
        clean(args.tool_dir, delete_weights=args.delete_weights)
        print("BOPB cleanup complete.")
        return 0

    if args.check:
        readiness = check_readiness(tool_dir=args.tool_dir)
        print(json.dumps(readiness, indent=2))
        return 0 if readiness["ready"] else 1

    if args.download_weights:
        target = download_weights(copy_from=args.weights_from, force=args.force)
        readiness = check_readiness(tool_dir=args.tool_dir, weights_path=target)
        print(json.dumps(readiness, indent=2))
        return 0 if readiness["ready"] else 1

    if args.install:
        py = create_virtualenv(args.tool_dir)
        install_dependencies(py)
        if not args.skip_weights:
            download_weights(copy_from=args.weights_from, force=args.force)
        readiness = check_readiness(tool_dir=args.tool_dir)
        print(json.dumps(readiness, indent=2))
        return 0 if readiness["ready"] else 1

    # Default action if no arguments: run check
    readiness = check_readiness(tool_dir=args.tool_dir)
    print(json.dumps(readiness, indent=2))
    return 0 if readiness["ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
