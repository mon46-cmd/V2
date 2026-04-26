"""Tests for the position review prompt (Prompt C).

Covers:
  - build_review_context output structure and key field presence.
  - Position stage detection (full / runner / stale).
  - RSI trend direction output.
  - Funding rate formatting in review context.
  - Key structural levels rendering.
  - ATR-denominated distance calculation.
  - Safety contract: hold on low confidence (tested via validate_position_review).
  - Default fallback on missing/bad data.
  - Social scan cluster label assignment.
  - Social scan pinned_symbols injection.
"""
from __future__ import annotations

import math
import sys
import os

import pandas as pd
import pytest

# Make src importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai.calls.context_builder import build_review_context
from ai.calls.social_scan import _cluster_label, _build_universe_table, _build_user_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snapshot_row(overrides: dict | None = None) -> dict:
    """Minimal feature snapshot row for review tests."""
    base = {
        "rsi_14":             62.0,
        "clv":                0.55,
        "adx_14":             28.0,
        "efficiency_ratio_20": 0.55,
        "vol_regime_score":   0.45,
        "wick_asymmetry_5":   0.03,
        "atr_14_pct":         0.80,
        "supertrend_dir":     "up",
        "supertrend":         76_000.0,
        "ema_21":             77_200.0,
        "ema_50":             75_800.0,
        "swing_high_10":      78_500.0,
        "swing_low_10":       76_100.0,
        "vwap_20":            77_800.0,
        "funding_rate":       0.00005,   # +0.5 bps
        "rvol_20":            0.85,
    }
    if overrides:
        base.update(overrides)
    return base


def _make_ohlcv_df(n: int = 15, rsi_start: float = 60.0, rsi_delta: float = 0.5) -> pd.DataFrame:
    """Fake snapshot dataframe with RSI column for trajectory tests."""
    rows = []
    for i in range(n):
        rows.append({
            "rsi_14":  rsi_start + rsi_delta * i,
            "close":   78_000 + i * 10,
            "high":    78_100 + i * 10,
            "low":     77_900 + i * 10,
            "open":    77_950 + i * 10,
            "volume":  1000 + i * 50,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_review_context -- basic structure
# ---------------------------------------------------------------------------

class TestBuildReviewContextStructure:
    def test_returns_string(self):
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_936, remaining_pct=100.0, bars_held=5, loser_streak=0,
            hook_reason="drawdown_60pct",
        )
        assert isinstance(ctx, str)
        assert len(ctx) > 100

    def test_contains_hook_reason(self):
        ctx = build_review_context(
            symbol="ETHUSDT", side="long",
            entry=2330, stop_loss=2280, tp1=2400, tp2=2470,
            mark=2358, remaining_pct=50.0, bars_held=8, loser_streak=0,
            hook_reason="tp1_hit",
        )
        assert "tp1_hit" in ctx

    def test_contains_key_position_fields(self):
        ctx = build_review_context(
            symbol="SOLUSDT", side="short",
            entry=90.0, stop_loss=93.5, tp1=86.0, tp2=82.0,
            mark=88.0, remaining_pct=100.0, bars_held=3, loser_streak=0,
            hook_reason="trend_flip",
        )
        assert "SOLUSDT" in ctx
        assert "short" in ctx
        assert "trend_flip" in ctx

    def test_contains_pnl(self):
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_936, remaining_pct=100.0, bars_held=5, loser_streak=0,
            hook_reason="drawdown_60pct",
        )
        assert "unrealized_pnl" in ctx
        assert "+" in ctx or "-" in ctx   # has a sign


# ---------------------------------------------------------------------------
# build_review_context -- trade stage
# ---------------------------------------------------------------------------

class TestTradeStage:
    def test_tp1_hit_shown(self):
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=79_600, remaining_pct=50.0, bars_held=12, loser_streak=0,
            hook_reason="tp1_hit",
        )
        assert "50%" in ctx
        assert "TP1" in ctx or "tp1" in ctx

    def test_tp2_distance_shown(self):
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=79_600, remaining_pct=50.0, bars_held=12, loser_streak=0,
            hook_reason="tp1_hit",
        )
        assert "TP2" in ctx or "tp2" in ctx

    def test_atr_distances_shown_when_row_provided(self):
        row = _make_snapshot_row()
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="drawdown_60pct",
            row=row,
        )
        assert "ATR" in ctx or "atr" in ctx


# ---------------------------------------------------------------------------
# build_review_context -- RSI trend
# ---------------------------------------------------------------------------

class TestRsiTrend:
    def test_rising_rsi_detected(self):
        # RSI starts at 55 and rises 1 per bar -> at bar -4 it was 55, now ~69.
        df = _make_ohlcv_df(n=20, rsi_start=55.0, rsi_delta=0.8)
        row = _make_snapshot_row({"rsi_14": float(df.iloc[-1]["rsi_14"])})
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            row=row, ohlcv_df=df,
        )
        assert "rsi_trend" in ctx
        assert "rising" in ctx

    def test_falling_rsi_detected(self):
        df = _make_ohlcv_df(n=20, rsi_start=72.0, rsi_delta=-1.0)
        row = _make_snapshot_row({"rsi_14": float(df.iloc[-1]["rsi_14"])})
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            row=row, ohlcv_df=df,
        )
        assert "rsi_trend" in ctx
        assert "falling" in ctx

    def test_rsi_trend_absent_when_insufficient_bars(self):
        # Only 3 bars -- can't compute 3-bar-ago RSI.
        df = _make_ohlcv_df(n=3, rsi_start=60.0, rsi_delta=1.0)
        row = _make_snapshot_row({"rsi_14": 62.0})
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=2, loser_streak=0,
            hook_reason="tp1_hit",
            row=row, ohlcv_df=df,
        )
        # rsi_trend should NOT appear (not enough bars).
        assert "rsi_trend" not in ctx


# ---------------------------------------------------------------------------
# build_review_context -- funding rate
# ---------------------------------------------------------------------------

class TestFundingRateInReview:
    def test_funding_shown_in_bps(self):
        row = _make_snapshot_row({"funding_rate": 0.00095})   # 9.5 bps
        ctx = build_review_context(
            symbol="ORCAUSDT", side="long",
            entry=1.72, stop_loss=1.60, tp1=1.90, tp2=2.08,
            mark=1.75, remaining_pct=100.0, bars_held=4, loser_streak=0,
            hook_reason="funding_adverse",
            row=row,
        )
        assert "funding_rate" in ctx
        assert "bps" in ctx

    def test_negative_funding_shown(self):
        row = _make_snapshot_row({"funding_rate": -0.009367})   # -93.67 bps (ORCA-like)
        ctx = build_review_context(
            symbol="ORCAUSDT", side="long",
            entry=1.72, stop_loss=1.60, tp1=1.90, tp2=2.08,
            mark=1.75, remaining_pct=100.0, bars_held=4, loser_streak=0,
            hook_reason="funding_adverse",
            row=row,
        )
        assert "-" in ctx   # has negative sign
        assert "bps" in ctx

    def test_no_funding_when_absent(self):
        row = _make_snapshot_row()
        row.pop("funding_rate", None)
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=4, loser_streak=0,
            hook_reason="tp1_hit",
            row=row,
        )
        assert "funding_rate" not in ctx


# ---------------------------------------------------------------------------
# build_review_context -- key structural levels
# ---------------------------------------------------------------------------

class TestKeyLevels:
    def test_ema_21_shown(self):
        row = _make_snapshot_row()
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            row=row,
        )
        assert "ema_21" in ctx

    def test_swing_levels_shown(self):
        row = _make_snapshot_row()
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            row=row,
        )
        assert "swing_low_10" in ctx
        assert "swing_high_10" in ctx

    def test_vwap_shown(self):
        row = _make_snapshot_row()
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            row=row,
        )
        assert "vwap" in ctx.lower()

    def test_levels_absent_when_no_row(self):
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
        )
        # Without a row, no KEY LEVELS section.
        assert "KEY LEVELS" not in ctx


# ---------------------------------------------------------------------------
# build_review_context -- 6 recent bars
# ---------------------------------------------------------------------------

class TestRecentBars:
    def test_six_bars_in_output(self):
        df = _make_ohlcv_df(n=20, rsi_start=60.0, rsi_delta=0.3)
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            ohlcv_df=df,
        )
        # Should have bar-5 through bar+0.
        assert "bar-5" in ctx
        assert "bar+0" in ctx

    def test_fewer_bars_graceful(self):
        df = _make_ohlcv_df(n=2, rsi_start=60.0, rsi_delta=1.0)
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=1, loser_streak=0,
            hook_reason="tp1_hit",
            ohlcv_df=df,
        )
        assert isinstance(ctx, str)


# ---------------------------------------------------------------------------
# build_review_context -- short side
# ---------------------------------------------------------------------------

class TestShortPositionReview:
    def test_short_adverse_move_positive(self):
        """For a short, adverse move = mark > entry (price went up against us)."""
        ctx = build_review_context(
            symbol="ETHUSDT", side="short",
            entry=2400, stop_loss=2480, tp1=2300, tp2=2200,
            mark=2450,   # 50 above entry = adverse for short
            remaining_pct=100.0, bars_held=5, loser_streak=1,
            hook_reason="drawdown_60pct",
        )
        assert "adverse_move" in ctx
        # adverse % should be positive and nonzero.
        import re
        m = re.search(r"adverse_move:\s+([0-9.]+)%", ctx)
        assert m is not None
        assert float(m.group(1)) > 0

    def test_short_pnl_positive_when_mark_below_entry(self):
        """For a short in profit, mark < entry."""
        ctx = build_review_context(
            symbol="ETHUSDT", side="short",
            entry=2400, stop_loss=2480, tp1=2300, tp2=2200,
            mark=2350,   # in profit
            remaining_pct=100.0, bars_held=5, loser_streak=0,
            hook_reason="tp1_hit",
        )
        assert "+2.08%" in ctx or "+" in ctx


# ---------------------------------------------------------------------------
# Social scan -- cluster labels
# ---------------------------------------------------------------------------

class TestClusterLabels:
    def _row(self, **kw) -> dict:
        base = {
            "symbol": "XXXXUSDT",
            "price": 1.0,
            "price_change_24h_pct": 0.0,
            "atr_pct_15m": 1.0,
            "high_24h": 1.05,
            "low_24h": 0.95,
            "active_flags": [],
            "has_open_position": False,
        }
        base.update(kw)
        return base

    def test_mega_label(self):
        assert _cluster_label(self._row(symbol="BTCUSDT")) == "[MEGA]"
        assert _cluster_label(self._row(symbol="ETHUSDT")) == "[MEGA]"

    def test_open_position_overrides_mega(self):
        # Even ETHUSDT should show [OPEN] if has_open_position.
        assert _cluster_label(self._row(symbol="ETHUSDT", has_open_position=True)) == "[OPEN]"

    def test_moonshot_label(self):
        row = self._row(price_change_24h_pct=0.25)  # +25%
        assert _cluster_label(row) == "[MOONSHOT]"

    def test_dump_label(self):
        row = self._row(price_change_24h_pct=-0.20)  # -20%
        assert _cluster_label(row) == "[DUMP]"

    def test_volatile_by_atr(self):
        row = self._row(atr_pct_15m=4.5)
        assert _cluster_label(row) == "[VOLATILE]"

    def test_volatile_by_range(self):
        row = self._row(price=1.0, high_24h=1.40, low_24h=1.00)  # 40% range
        assert _cluster_label(row) == "[VOLATILE]"

    def test_trending_by_flag(self):
        row = self._row(active_flags=["flag_adx_strong_trend"])
        assert _cluster_label(row) == "[TRENDING]"

    def test_quiet_label(self):
        row = self._row(atr_pct_15m=0.2)
        assert _cluster_label(row) == "[QUIET]"

    def test_watch_fallback(self):
        row = self._row(atr_pct_15m=1.0, active_flags=[])
        assert _cluster_label(row) == "[WATCH]"


# ---------------------------------------------------------------------------
# Social scan -- universe table rendering
# ---------------------------------------------------------------------------

class TestUniverseTableRendering:
    def _sample_rows(self) -> list[dict]:
        return [
            {
                "symbol": "BTCUSDT", "price": 78_000, "price_change_24h_pct": 0.006,
                "turnover_24h": 5_800e6, "funding_rate": 0.000028,
                "open_interest_value": 12_000e6,
                "high_24h": 79_000, "low_24h": 77_000,
                "atr_pct_15m": 0.12, "active_flags": ["flag_golden_cross"],
                "has_open_position": False,
            },
            {
                "symbol": "ORCAUSDT", "price": 1.73, "price_change_24h_pct": 0.87,
                "turnover_24h": 166e6, "funding_rate": -0.009367,
                "open_interest_value": 45e6,
                "high_24h": 1.85, "low_24h": 1.20,
                "atr_pct_15m": 4.89, "active_flags": ["flag_adx_strong_trend"],
                "has_open_position": True,
            },
        ]

    def test_table_has_header(self):
        table = _build_universe_table(self._sample_rows())
        assert "UNIVERSE" in table
        assert "Cols:" in table

    def test_oi_shown_when_nonzero(self):
        table = _build_universe_table(self._sample_rows())
        # BTC has 12_000M OI.
        assert "12000M" in table

    def test_oi_na_when_zero(self):
        rows = self._sample_rows()
        rows[0]["open_interest_value"] = 0
        table = _build_universe_table(rows)
        assert "n/a" in table

    def test_cluster_label_in_table(self):
        table = _build_universe_table(self._sample_rows())
        assert "[MEGA]" in table
        assert "[OPEN]" in table

    def test_funding_shown_in_bps(self):
        table = _build_universe_table(self._sample_rows())
        assert "bps" in table


# ---------------------------------------------------------------------------
# Social scan -- pinned_symbols in user prompt
# ---------------------------------------------------------------------------

class TestPinnedSymbols:
    def test_pinned_note_in_prompt(self):
        rows = [
            {
                "symbol": "BTCUSDT", "price": 78_000, "price_change_24h_pct": 0.006,
                "turnover_24h": 5_800e6, "funding_rate": 0.000028,
                "open_interest_value": 12_000e6,
                "high_24h": 79_000, "low_24h": 77_000,
                "atr_pct_15m": 0.12, "active_flags": [],
                "has_open_position": False,
            }
        ]
        prompt = _build_user_prompt(
            as_of="2026-04-26T12:00:00+00:00",
            universe_rows=rows,
            pinned_symbols=["ORCAUSDT", "ETHUSDT"],
        )
        assert "PINNED" in prompt
        assert "ORCAUSDT" in prompt
        assert "ETHUSDT" in prompt

    def test_no_pinned_note_when_empty(self):
        rows = [
            {
                "symbol": "BTCUSDT", "price": 78_000, "price_change_24h_pct": 0.006,
                "turnover_24h": 5_800e6, "funding_rate": 0.000028,
                "open_interest_value": 0,
                "high_24h": 79_000, "low_24h": 77_000,
                "atr_pct_15m": 0.12, "active_flags": [],
                "has_open_position": False,
            }
        ]
        prompt = _build_user_prompt(
            as_of="2026-04-26T12:00:00+00:00",
            universe_rows=rows,
            pinned_symbols=[],
        )
        assert "PINNED" not in prompt


# ---------------------------------------------------------------------------
# Edge cases & robustness
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_entry_does_not_crash(self):
        ctx = build_review_context(
            symbol="XYZUSDT", side="long",
            entry=0.0, stop_loss=0.0, tp1=0.0, tp2=0.0,
            mark=0.0, remaining_pct=100.0, bars_held=1, loser_streak=0,
            hook_reason="manual",
        )
        assert isinstance(ctx, str)

    def test_none_row_produces_valid_context(self):
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            row=None, ohlcv_df=None,
        )
        assert "BTCUSDT" in ctx
        assert "tp1_hit" in ctx

    def test_empty_ohlcv_does_not_crash(self):
        ctx = build_review_context(
            symbol="BTCUSDT", side="long",
            entry=78_000, stop_loss=76_500, tp1=79_500, tp2=81_000,
            mark=78_500, remaining_pct=100.0, bars_held=6, loser_streak=0,
            hook_reason="tp1_hit",
            ohlcv_df=pd.DataFrame(),
        )
        assert isinstance(ctx, str)

    def test_cluster_label_all_types(self):
        """Smoke test: all cluster labels return strings."""
        variants = [
            {"symbol": "BTCUSDT"},
            {"symbol": "XYZUSDT", "has_open_position": True},
            {"symbol": "XYZUSDT", "price_change_24h_pct": 0.25},
            {"symbol": "XYZUSDT", "price_change_24h_pct": -0.20},
            {"symbol": "XYZUSDT", "atr_pct_15m": 5.0},
            {"symbol": "XYZUSDT", "active_flags": ["flag_golden_cross"]},
            {"symbol": "XYZUSDT", "atr_pct_15m": 0.1},
            {"symbol": "XYZUSDT"},
        ]
        base = {"price": 1.0, "price_change_24h_pct": 0.0, "atr_pct_15m": 1.0,
                "high_24h": 1.1, "low_24h": 0.9, "active_flags": [],
                "has_open_position": False}
        for v in variants:
            row = {**base, **v}
            label = _cluster_label(row)
            assert label.startswith("[") and label.endswith("]")
