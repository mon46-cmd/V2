"""Identifier helpers: run ids and ULID-like monotonic ids.

Three utilities are exported:

* ``run_id()``   -- human-readable timestamp id for run directories.
* ``ulid()``     -- 26-char sortable opaque id (ULID spec compatible).
* ``short_id()`` -- short hex token for debug correlation.
"""
from __future__ import annotations

import os
import secrets
import time

import pandas as pd

from .time import now_utc


def run_id(now: pd.Timestamp | None = None) -> str:
    """Return a run id like ``'20260426T090000Z'``.  UTC, second precision."""
    t = now if now is not None else now_utc()
    return t.strftime("%Y%m%dT%H%M%SZ")


# Crockford's base-32 alphabet (no I, L, O, U) for ULID compatibility.
_B32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _b32_encode(n: int, length: int) -> str:
    out: list[str] = []
    for _ in range(length):
        out.append(_B32[n & 0x1F])
        n >>= 5
    return "".join(reversed(out))


def ulid() -> str:
    """Return a 26-char ULID: 48-bit ms timestamp + 80-bit random suffix.

    Lexicographically sortable by creation time.  Not strictly monotonic
    within the same millisecond across processes -- use the random suffix
    to break ties.
    """
    ms   = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = int.from_bytes(secrets.token_bytes(10), "big")
    return _b32_encode(ms, 10) + _b32_encode(rand, 16)


def short_id(n_bytes: int = 4) -> str:
    """Return a short hex token for debug correlation (e.g. ``'a3f1b2c0'``)."""
    return os.urandom(n_bytes).hex()
