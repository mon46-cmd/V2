"""Filesystem helpers for the data tree.

All writes elsewhere in the codebase assume these directories exist.
``ensure_dirs(cfg)`` is idempotent and safe to call repeatedly.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


def ensure_dirs(cfg: "Config") -> None:
    """Create the full data directory tree if any directories are missing."""
    for p in (
        cfg.data_root,
        cfg.cache_root,
        cfg.feature_root,
        cfg.run_root,
        cfg.log_root,
    ):
        Path(p).mkdir(parents=True, exist_ok=True)


def run_dir(cfg: "Config", run_id: str) -> Path:
    """Return (and create if needed) the directory for a given run id."""
    d = cfg.run_root / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_write(path: Path, content: str | bytes, *, encoding: str = "utf-8") -> None:
    """Write *content* to *path* atomically via a sibling temp file.

    Guarantees that readers never see a partially written file.
    Parent directories are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if isinstance(content, str):
            tmp.write_text(content, encoding=encoding)
        else:
            tmp.write_bytes(content)
        tmp.replace(path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
