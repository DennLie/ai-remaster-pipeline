"""Dust-only cooperative stop and process-scoped candidate locking."""
from contextlib import contextmanager
from pathlib import Path
import os


def check_cancel(path):
    if path and Path(path).exists():
        raise ValueError('Dust stop requested. Verified model sections are retained for retry.')


@contextmanager
def candidate_lock(target):
    folder = target.parent / '.locks'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / target.name).open('a+b') as handle:
        if not handle.tell():
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another worker owns this Dust candidate. Wait for it to stop.') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
