"""Time helpers.  Single source of truth for 'now'.

Tests monkeypatch ``now_utc`` to freeze the clock.

All timestamps flowing through the system should be tz-aware UTC
pandas Timestamps.  Never use naive datetimes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


def now_utc() -> pd.Timestamp:
    """Return the current UTC instant as a tz-aware pandas Timestamp."""
    return pd.Timestamp(datetime.now(timezone.utc))


def to_utc(ts: "pd.Timestamp | datetime | str | int | float") -> pd.Timestamp:
    """Coerce an arbitrary timestamp to a tz-aware UTC pandas Timestamp.

    Integers and floats are interpreted as *milliseconds* since epoch
    (Bybit's native convention).
    """
    if isinstance(ts, (int, float)):
        return pd.Timestamp(int(ts), unit="ms", tz="UTC")
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def utc_isoformat(ts: pd.Timestamp | None = None) -> str:
    """Return an ISO-8601 UTC string, e.g. ``'2026-04-26T09:00:00.000Z'``.

    If *ts* is None the current time is used.
    """
    t = ts if ts is not None else now_utc()
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(t.microsecond / 1000):03d}Z"


def ms_since_epoch(ts: pd.Timestamp | None = None) -> int:
    """Return Unix time in milliseconds (Bybit convention)."""
    t = ts if ts is not None else now_utc()
    return int(t.value // 1_000_000)
