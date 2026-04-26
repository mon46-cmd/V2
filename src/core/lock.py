"""Cross-platform single-process lock.

Prevents two concurrent instances of the same loop from running and
racing each other (e.g. double-billing the AI watchlist call on a fast
systemd restart).

POSIX: uses ``fcntl.flock`` -- the kernel releases the lock automatically
       if the process crashes.
Windows: falls back to a stale-PID check (best-effort, not kernel-enforced).

Usage::

    from core.lock import file_lock, LockBusy

    try:
        with file_lock(cfg.data_root / "scanner.lock"):
            run_scanner()
    except LockBusy as exc:
        log.warning("already running: %s", exc)
        sys.exit(1)
"""
from __future__ import annotations

import errno
import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator

log = logging.getLogger(__name__)


class LockBusy(RuntimeError):
    """Raised when another live process already holds the lock."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
            )
            if h == 0:
                return False
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Acquire an exclusive process-level lock at *path*.

    Raises :class:`LockBusy` immediately if another live process holds it.
    Always releases the lock on exit (normal or exceptional).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fh: IO | None = None

    if sys.platform != "win32":
        # POSIX -- fcntl.flock.
        import fcntl
        fh = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            fh.close()
            raise LockBusy(
                f"another process already holds {path} (errno={e.errno})"
            ) from e
        try:
            fh.seek(0)
            fh.truncate()
            fh.write(str(os.getpid()))
            fh.flush()
            try:
                yield
            finally:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            fh.close()
            try:
                path.unlink()
            except OSError:
                pass
        return

    # Windows -- stale-PID check.
    if path.exists():
        try:
            other = int(path.read_text(encoding="utf-8").strip() or "0")
        except (OSError, ValueError):
            other = 0
        if other and _pid_alive(other):
            raise LockBusy(f"another process (pid={other}) holds {path}")

    fd = os.open(str(path), os.O_CREAT | os.O_TRUNC | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode("utf-8"))
    finally:
        os.close(fd)
    try:
        yield
    finally:
        try:
            path.unlink()
        except OSError:
            pass
