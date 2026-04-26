"""Parquet cache with atomic writes and safe merge-append.

Layout on disk::

    <cache_root>/
      klines/BTCUSDT/15/BTCUSDT_klines_15.parquet
      funding/BTCUSDT/BTCUSDT_funding.parquet
      ticks_live/BTCUSDT/2026-04-26.parquet

Guarantees:
- Writes are atomic: a ``.tmp`` sibling is written then ``os.replace``d
  over the target path. A crash mid-write leaves the previous good file.
- Reads tolerate missing or corrupt files (removed and ``None`` returned).
- ``append()`` merges new rows into existing data, deduplicates on a key
  column, and overwrites the file atomically.

Usage::

    from downloader.cache import ParquetCache
    from core import load_config

    cfg   = load_config()
    cache = ParquetCache(cfg.cache_root)

    # Write
    cache.write(df, kind="klines", symbol="BTCUSDT", subkey="15")

    # Read back
    df = cache.read(kind="klines", symbol="BTCUSDT", subkey="15")

    # Append new rows (dedup on timestamp)
    cache.append(new_rows, kind="klines", symbol="BTCUSDT", subkey="15")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from downloader.constants import MIN_PARQUET_BYTES
from downloader.errors import CacheError

log = logging.getLogger(__name__)


class ParquetCache:
    """File-backed parquet store with atomic writes.

    Args:
        root: Directory where all cache files will be written.
              Created automatically if it does not exist.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def path(self, kind: str, symbol: str, subkey: str = "") -> Path:
        """Return the canonical path for a (kind, symbol, subkey) triple.

        Examples::

            cache.path("klines", "BTCUSDT", "15")
            # -> <root>/klines/BTCUSDT/15/BTCUSDT_klines_15.parquet

            cache.path("funding", "BTCUSDT")
            # -> <root>/funding/BTCUSDT/BTCUSDT_funding.parquet
        """
        if subkey:
            d = self.root / kind / symbol / subkey
            d.mkdir(parents=True, exist_ok=True)
            return d / f"{symbol}_{kind}_{subkey}.parquet"
        d = self.root / kind / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol}_{kind}.parquet"

    def daily_path(self, kind: str, symbol: str, date: str) -> Path:
        """Return the path for a daily-partitioned file (e.g. live ticks).

        Example::

            cache.daily_path("ticks_live", "BTCUSDT", "2026-04-26")
            # -> <root>/ticks_live/BTCUSDT/2026-04-26.parquet
        """
        d = self.root / kind / symbol
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{date}.parquet"

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read(self, kind: str, symbol: str, subkey: str = "") -> pd.DataFrame | None:
        """Read a parquet file.  Returns ``None`` if missing or corrupt."""
        return self._read(self.path(kind, symbol, subkey))

    def read_daily(self, kind: str, symbol: str, date: str) -> pd.DataFrame | None:
        """Read a daily-partitioned parquet file."""
        return self._read(self.daily_path(kind, symbol, date))

    def _read(self, p: Path) -> pd.DataFrame | None:
        if not p.exists():
            return None
        if p.stat().st_size < MIN_PARQUET_BYTES:
            log.warning("undersized parquet %s (%d B), removing", p, p.stat().st_size)
            p.unlink(missing_ok=True)
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            log.warning("corrupt parquet %s, removing: %s", p, exc)
            p.unlink(missing_ok=True)
            return None
        return df if not df.empty else None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, df: pd.DataFrame, kind: str, symbol: str, subkey: str = "") -> Path:
        """Write ``df`` atomically.  Returns the final path."""
        p = self.path(kind, symbol, subkey)
        _atomic_write(p, df)
        return p

    def write_daily(self, df: pd.DataFrame, kind: str, symbol: str, date: str) -> Path:
        """Write a daily-partitioned file atomically."""
        p = self.daily_path(kind, symbol, date)
        _atomic_write(p, df)
        return p

    # ------------------------------------------------------------------
    # Append / merge
    # ------------------------------------------------------------------

    def append(
        self,
        df:     pd.DataFrame,
        kind:   str,
        symbol: str,
        subkey: str = "",
        *,
        key:    str = "timestamp",
    ) -> pd.DataFrame:
        """Merge ``df`` into the existing cache file.

        New rows are combined with existing ones; duplicates on ``key``
        are dropped (keeping the newest version), and the result is
        sorted ascending before being written back atomically.

        Returns the full merged DataFrame.
        """
        if df is None or df.empty:
            existing = self.read(kind, symbol, subkey)
            return existing if existing is not None else pd.DataFrame()

        existing = self.read(kind, symbol, subkey)
        if existing is None or existing.empty:
            merged = df.sort_values(key).reset_index(drop=True)
        else:
            merged = (
                pd.concat([existing, df], ignore_index=True)
                .drop_duplicates(subset=[key], keep="last")
                .sort_values(key)
                .reset_index(drop=True)
            )
        self.write(merged, kind, symbol, subkey)
        return merged

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def last_timestamp(self, kind: str, symbol: str, subkey: str = "") -> pd.Timestamp | None:
        """Return the most recent timestamp in a cached file, or None."""
        df = self.read(kind, symbol, subkey)
        if df is None or df.empty or "timestamp" not in df.columns:
            return None
        return pd.Timestamp(df["timestamp"].max())

    def first_timestamp(self, kind: str, symbol: str, subkey: str = "") -> pd.Timestamp | None:
        """Return the oldest timestamp in a cached file, or None."""
        df = self.read(kind, symbol, subkey)
        if df is None or df.empty or "timestamp" not in df.columns:
            return None
        return pd.Timestamp(df["timestamp"].min())

    def inventory(self) -> pd.DataFrame:
        """List every parquet file under the cache root (debug helper)."""
        rows: list[dict] = []
        if not self.root.exists():
            return pd.DataFrame()
        for p in sorted(self.root.rglob("*.parquet")):
            rel   = p.relative_to(self.root)
            parts = rel.parts
            rows.append({
                "kind":   parts[0] if len(parts) > 0 else "",
                "symbol": parts[1] if len(parts) > 1 else "",
                "subkey": "/".join(parts[2:-1]) if len(parts) > 3 else "",
                "file":   parts[-1],
                "bytes":  p.stat().st_size,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Internal atomic write
# ---------------------------------------------------------------------------

def _atomic_write(path: Path, df: pd.DataFrame) -> None:
    """Write ``df`` to ``path`` via a .tmp sibling, then atomically replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, compression="zstd", index=False)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise CacheError(f"atomic write failed for {path}: {exc}") from exc
