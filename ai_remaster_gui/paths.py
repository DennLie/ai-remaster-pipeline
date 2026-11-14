from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

from .config import ROOT, VIDEO_EXTS, comfy_dir_for


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def resolve(text: str) -> Path:
    path = Path(text).expanduser()
    return path if path.is_absolute() else ROOT / path


def check_artifact_access(path: Path | str | None, *, require_file: bool = False) -> tuple[bool, OSError | None]:
    """Safely check if an artifact exists (and optionally is a regular file).

    Returns (is_ready, error):
    - Accessible and exists (and is regular file if require_file): (True, None)
    - Accessible and does not exist: (False, None)
    - Raises OSError (e.g. PermissionError, WinError 5, sharing violation): (False, exc)
    """
    if not path:
        return False, None
    try:
        resolved = resolve(str(path))
        if require_file:
            return resolved.is_file(), None
        return resolved.exists(), None
    except OSError as exc:
        return False, exc


def safe_is_file(path: Path | str | None, on_error: Callable[[Path, OSError], None] | None = None) -> bool:
    """Return True if path exists, is a regular file, and is accessible without OSError."""
    ok, exc = check_artifact_access(path, require_file=True)
    if exc is not None and on_error:
        try:
            resolved = resolve(str(path))
        except Exception:
            resolved = Path(str(path))
        on_error(resolved, exc)
    return ok


def safe_exists(path: Path | str | None, on_error: Callable[[Path, OSError], None] | None = None) -> bool:
    """Return True if path exists and is accessible without OSError."""
    ok, exc = check_artifact_access(path, require_file=False)
    if exc is not None and on_error:
        try:
            resolved = resolve(str(path))
        except Exception:
            resolved = Path(str(path))
        on_error(resolved, exc)
    return ok


def ensure_inherited_permissions(path: Path | str) -> None:
    """Ensure path inherits permissions from its parent directory on Windows.

    Prevents restrictive ACL inheritance breakage when staging directories
    or files are created or moved across privilege contexts (e.g., elevated
    sessions, same-volume renames). Leaves permissions on non-Windows
    platforms untouched.
    """
    if sys.platform != "win32":
        return
    try:
        p = Path(path)
        if not p.exists():
            return
        cmd = ["icacls", str(p), "/reset"]
        if p.is_dir():
            cmd.append("/t")
        subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        # Non-fatal defense-in-depth: do not abort the pipeline if icacls is blocked.
        pass


class staging_directory:
    """Context manager for staging directories that maintain parent ACL inheritance.

    Unlike tempfile.TemporaryDirectory (which uses mode 0o700, creating an explicit
    non-inheriting DACL on Windows that locks out non-elevated users if created in an
    elevated session), staging_directory creates folders with default permissions
    (CreateDirectoryW with NULL security attributes), ensuring newly generated
    artifacts inherit the workspace/parent folder ACLs and remain accessible to
    normal, non-elevated users.
    """

    def __init__(
        self,
        prefix: str = ".stage-",
        suffix: str = "",
        dir: Path | str | None = None,
        delete: bool = True,
    ) -> None:
        self.dir = Path(dir) if dir is not None else Path(tempfile.gettempdir())
        self.prefix = prefix
        self.suffix = suffix
        self.delete = delete
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        for _ in range(100):
            token = uuid.uuid4().hex[:12]
            candidate = self.dir / f"{self.prefix}{token}{self.suffix}"
            try:
                candidate.mkdir(parents=False, exist_ok=False)
                self.path = candidate
                ensure_inherited_permissions(self.path)
                return self.path
            except FileExistsError:
                continue
        raise RuntimeError(f"Could not allocate staging directory under {self.dir}")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.delete and self.path is not None and self.path.exists():
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception:
                pass


def is_unfinished_media_artifact(path: Path) -> bool:
    """Exclude in-progress encoder files from GUI previews and media requests.

    Our FFmpeg writers create e.g. `movie.partial.mkv` and atomically
    publish `movie.mkv` only after the container is finalized. A suffix-only
    check mistakes the unfinished Matroska file for a playable video.
    """
    return any(piece in {"partial", "tmp", "part"} for piece in path.name.lower().split("."))


def is_within(path: Path, root: Path) -> bool:
    """True when path is root itself or nested under it (both resolved first)."""
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def served_roots(extra_paths: Iterable[str] = ()) -> list[Path]:
    """Directories the GUI may read files from when answering browser requests: the project
    tree, the ComfyUI install (and its output folder), and the folder of any explicitly
    chosen file passed in extra_paths (the source video legitimately lives anywhere on disk)."""
    comfy_dir = Path(comfy_dir_for())
    roots = [ROOT, comfy_dir, comfy_dir / "output"]
    for text in extra_paths:
        if not text:
            continue
        candidate = resolve(text)
        roots.append(candidate if candidate.is_dir() else candidate.parent)
    return roots


def resolve_served(text: str, extra_paths: Iterable[str] = ()) -> Path | None:
    """Resolve a browser-supplied path, returning it only when it stays inside an allowed
    root (see served_roots). Returns None for paths that escape — callers answer 404 — so a
    malicious page cannot read arbitrary files such as C:/Windows/win.ini or /etc/passwd."""
    if not text:
        return None
    path = resolve(text)
    return path if any(is_within(path, root) for root in served_roots(extra_paths)) else None


def resolve_video_source(text: str) -> Path:
    path = resolve(text)
    try:
        if path.exists():
            return path
    except OSError:
        return path
    if path.suffix.lower() not in VIDEO_EXTS:
        return path
    name = path.name
    if not any(ch in name for ch in '<>:"|?*'):
        return path
    parent = path.parent
    try:
        if not parent.exists():
            return path
    except OSError:
        return path

    def comparable(filename: str) -> str:
        stripped = "".join("" if ch in '<>:"|?*\uff5c\xa6' else ch for ch in filename)
        return " ".join(stripped.lower().split())

    wanted = comparable(name)
    try:
        matches = sorted(
            (
                candidate
                for candidate in parent.iterdir()
                if safe_is_file(candidate)
                and candidate.suffix.lower() == path.suffix.lower()
                and comparable(candidate.name) == wanted
            ),
            key=lambda candidate: candidate.name.lower(),
        )
        return matches[0] if len(matches) == 1 else path
    except OSError:
        return path


def newest(folder: Path, exts: set[str]) -> Path | None:
    try:
        if not folder.exists():
            return None
    except OSError:
        return None
    files: list[Path] = []
    try:
        for path in folder.rglob("*"):
            try:
                if path.is_file() and path.suffix.lower() in exts:
                    files.append(path)
            except OSError:
                continue
    except OSError:
        return None
    try:
        return max(files, key=lambda path: path.stat().st_mtime_ns) if files else None
    except OSError:
        return None


def aspect_slug(value: str) -> str:
    return value.replace(":", "x").replace(".", "_")


def safe_stem(path_text: str) -> str:
    stem = Path(path_text).stem.replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in stem)


def even_int(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def parse_aspect(value: str) -> float:
    if ":" in value:
        left, right = value.split(":", 1)
        return float(left) / float(right)
    return float(value)


def format_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
