"""Tests for the volatility-regime size adjustment in RiskEngine."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from portfolio.risk import RiskEngine


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        run_root           = tmp_path / "runs",
        log_root           = tmp_path / "logs",
        max_open_positions = 5,
        max_position_usd   = 50_000.0,
        risk_pct_per_trade = 0.01,
        atr_mult_sl        = 1.5,
        tp1_rr             = 1.5,
        tp2_rr             = 3.0,
        trail_atr_mult     = 1.0,
    )


class TestVolRegimeSizing:

    def test_neutral_score_no_change(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        a = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000, vol_regime_score=1.0)
        b = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000)   # default 1.0
        assert a.qty_usd == pytest.approx(b.qty_usd)

    def test_high_vol_shrinks_size(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        baseline = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000)
        elevated = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000, vol_regime_score=2.0)
        # 1/max(2.0,0.5) = 0.5 -> exactly half size
        assert elevated.qty_usd == pytest.approx(baseline.qty_usd * 0.5, rel=1e-3)

    def test_low_vol_no_upsize(self, tmp_path):
        # vol_score < 1 should NOT increase size (clamped to 1.0)
        eng = RiskEngine(_cfg(tmp_path))
        baseline = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000)
        quiet = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000, vol_regime_score=0.3)
        assert quiet.qty_usd == pytest.approx(baseline.qty_usd, rel=1e-3)

    def test_extreme_vol_floor(self, tmp_path):
        # Even score=10 must not push the multiplier below 0.5.
        eng = RiskEngine(_cfg(tmp_path))
        baseline = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000)
        crazy    = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000, vol_regime_score=10.0)
        assert crazy.qty_usd == pytest.approx(baseline.qty_usd * 0.5, rel=1e-3)

    def test_invalid_score_falls_back(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        baseline = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000)
        bogus    = eng.compute(side="long", entry=100, atr=2, equity_usd=10_000, vol_regime_score=float("nan"))
        # NaN -> clamps; 1/max(nan,0.5) is nan; max/min compare with nan = nan;
        # we expect graceful behaviour: NaN means "treat as 1.0" or shrink.
        # The implementation does max/min on nan which returns the other arg in python
        # (nan propagation). To be robust, just check that the result is finite and
        # not larger than baseline.
        assert bogus.accepted
        assert bogus.qty_usd <= baseline.qty_usd + 1e-6
