"""Tests for the deterministic backtest replay module."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest import (
    BTReport,
    ConservativePolicy,
    Decision,
    DecisionPolicy,
    run_backtest,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root           = tmp_path / "runs",
        log_root           = tmp_path / "logs",
        max_open_positions = 1,
        max_position_usd   = 50_000.0,
        risk_pct_per_trade = 0.01,
        atr_mult_sl        = 1.5,
        tp1_rr             = 1.5,
        tp2_rr             = 3.0,
        trail_atr_mult     = 1.0,
        trigger_flags      = ("flag_volume_climax", "flag_sweep_up", "flag_sweep_dn"),
        prompt_cooldown_candles  = 3,
        cooldown_bypass_atr_mult = 0.8,
        cooldown_bypass_floor_pct= 0.01,
        adx_min_trade            = 18.0,
        efficiency_min_trade     = 0.35,
        min_action_confidence    = 0.55,
        max_drawdown_pct         = 0.0,
        bar_max_age_sec          = 0,
    )


def _trending_ohlcv(n: int = 500, drift: float = 0.001) -> pd.DataFrame:
    """Build a deterministic uptrending OHLCV series with mild noise."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    rets = rng.normal(loc=drift, scale=0.005, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    high  = close * (1.0 + np.abs(rng.normal(0, 0.002, n)))
    low   = close * (1.0 - np.abs(rng.normal(0, 0.002, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol   = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ── basic shape ──────────────────────────────────────────────────────────────

class TestRunBacktestSmoke:

    def test_empty_input_returns_empty_report(self, tmp_path):
        rep = run_backtest(ohlcv=pd.DataFrame(), symbol="X", cfg=_cfg(tmp_path))
        assert isinstance(rep, BTReport)
        assert rep.trades == []
        assert rep.stats()["n_trades"] == 0

    def test_runs_on_synthetic_uptrend(self, tmp_path):
        df  = _trending_ohlcv(n=400, drift=0.001)
        rep = run_backtest(ohlcv=df, symbol="BTCUSDT", cfg=_cfg(tmp_path))
        # Should at least not crash and produce a stats dict.
        stats = rep.stats()
        assert "n_trades"      in stats
        assert "total_return"  in stats
        assert "sharpe"        in stats
        assert "max_drawdown"  in stats
        assert stats["max_drawdown"] <= 0.0     # drawdowns are non-positive
        # Equity curve is monotonic in time.
        if len(rep.equity_curve) > 1:
            assert rep.equity_curve.index.is_monotonic_increasing


# ── policy plug-in ───────────────────────────────────────────────────────────

class _AlwaysLongPolicy(DecisionPolicy):
    def decide(self, bar):
        return Decision("long", 0.7)


class _AlwaysSkipPolicy(DecisionPolicy):
    def decide(self, bar):
        return Decision("skip", 0.0)


class TestPolicyPluggability:

    def test_skip_policy_yields_no_trades(self, tmp_path):
        df  = _trending_ohlcv(n=300)
        rep = run_backtest(
            ohlcv=df, symbol="BTCUSDT", cfg=_cfg(tmp_path),
            policy=_AlwaysSkipPolicy(),
        )
        assert rep.stats()["n_trades"] == 0
        # Equity unchanged.
        assert rep.stats()["final_equity"] == pytest.approx(10_000.0)

    def test_default_policy_is_conservative_with_cfg_adx(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.adx_min_trade = 9999.0           # impossibly high -> never enter
        df  = _trending_ohlcv(n=300)
        rep = run_backtest(ohlcv=df, symbol="BTCUSDT", cfg=cfg)
        assert rep.stats()["n_trades"] == 0


# ── conservative policy unit ─────────────────────────────────────────────────

class TestConservativePolicy:

    def test_low_adx_skips(self):
        bar = pd.Series({"adx_14": 10.0, "supertrend_dir": 1.0, "rsi_14": 60.0})
        d   = ConservativePolicy(adx_min=18).decide(bar)
        assert d.side == "skip"

    def test_long_signal(self):
        bar = pd.Series({"adx_14": 30.0, "supertrend_dir": 1.0, "rsi_14": 60.0})
        d   = ConservativePolicy(adx_min=18).decide(bar)
        assert d.side == "long"
        assert 0.0 < d.confidence <= 1.0

    def test_short_signal(self):
        bar = pd.Series({"adx_14": 30.0, "supertrend_dir": -1.0, "rsi_14": 40.0})
        d   = ConservativePolicy(adx_min=18).decide(bar)
        assert d.side == "short"

    def test_mixed_signal_skips(self):
        bar = pd.Series({"adx_14": 30.0, "supertrend_dir": 1.0, "rsi_14": 40.0})
        d   = ConservativePolicy(adx_min=18).decide(bar)
        assert d.side == "skip"


# ── stats ────────────────────────────────────────────────────────────────────

class TestStats:

    def test_stats_zero_trades(self, tmp_path):
        rep = run_backtest(ohlcv=pd.DataFrame(), symbol="X", cfg=_cfg(tmp_path))
        s   = rep.stats()
        assert s["n_trades"]      == 0
        assert s["hit_rate"]      == 0.0
        assert s["profit_factor"] == 0.0
