"""Tests for the 1-bar lag (closed-bar-only) entry guard."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from loops.scanner import _drop_open_bar


def _df(timestamps: list[datetime]) -> pd.DataFrame:
    idx = pd.DatetimeIndex(timestamps, tz="UTC")
    return pd.DataFrame({"close": list(range(len(timestamps)))}, index=idx)


class TestDropOpenBar:

    def test_drops_in_progress_15m_bar(self):
        # Pretend 'now' is 7 minutes into the current 15m bar.
        now      = datetime.now(tz=timezone.utc)
        last_ts  = now - timedelta(minutes=7)            # bar opened 7m ago
        prev_ts  = last_ts - timedelta(minutes=15)
        df       = _df([prev_ts, last_ts])
        out      = _drop_open_bar(df, interval_min=15)
        assert len(out) == 1
        assert out.index[-1] == pd.Timestamp(prev_ts).tz_convert("UTC")

    def test_keeps_closed_bar(self):
        now     = datetime.now(tz=timezone.utc)
        last_ts = now - timedelta(minutes=20)            # closed 5m ago
        prev_ts = last_ts - timedelta(minutes=15)
        df      = _df([prev_ts, last_ts])
        out     = _drop_open_bar(df, interval_min=15)
        assert len(out) == 2

    def test_empty_input_returned(self):
        df  = pd.DataFrame()
        out = _drop_open_bar(df, interval_min=15)
        assert out.empty

    def test_naive_timestamp_treated_as_utc(self):
        now     = datetime.now(tz=timezone.utc)
        last_ts = (now - timedelta(minutes=5)).replace(tzinfo=None)  # naive
        prev_ts = (last_ts - timedelta(minutes=15)).replace(tzinfo=None)
        idx = pd.DatetimeIndex([prev_ts, last_ts])  # naive index
        df  = pd.DataFrame({"close": [1, 2]}, index=idx)
        out = _drop_open_bar(df, interval_min=15)
        assert len(out) == 1
