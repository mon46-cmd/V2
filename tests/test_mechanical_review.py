"""Tests for the deterministic mechanical position-review fallback.

Run::
    pytest tests/test_mechanical_review.py -v
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from portfolio.mechanical_review import (
    MechanicalReviewConfig,
    mechanical_review,
)
from portfolio.paper_broker import PaperBroker
from portfolio.risk import RiskEngine


# ── helpers ──────────────────────────────────────────────────────────────────

def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        paper_equity_usd     = 10_000.0,
        run_root             = tmp_path / "runs",
        log_root             = tmp_path / "logs",
        max_open_positions   = 3,
        max_position_usd     = 5_000.0,
        risk_pct_per_trade   = 0.01,
        atr_mult_sl          = 1.5,
        tp1_rr               = 1.5,
        tp2_rr               = 3.0,
        trail_atr_mult       = 1.0,
        max_drawdown_pct     = 0.0,
    )


def _open_long(broker: PaperBroker, *, entry: float = 100.0, atr: float = 2.0) -> "Position":  # noqa: F821
    eng = RiskEngine(_cfg(Path(broker._pos_path).parent.parent))
    sizing = eng.compute(side="long", entry=entry, atr=atr, equity_usd=10_000.0)
    pos = broker.open("BTCUSDT", sizing=sizing)
    assert pos is not None
    return pos


# ── default hold ─────────────────────────────────────────────────────────────

class TestDefaultHold:

    def test_fresh_position_holds(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker)
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=100.0, atr=2.0, bars_held=1,
        )
        assert rev["action"] == "hold"
        assert rev["new_stop_loss"] == 0.0


# ── stale exit ───────────────────────────────────────────────────────────────

class TestStaleExit:

    def test_stale_underwater_exits(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker)
        pos.pnl_usd = -10.0
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=99.0, atr=2.0, bars_held=33,
        )
        assert rev["action"] == "exit"
        assert rev["urgency"] == "high"
        assert "stale" in rev["rationale"].lower()

    def test_stale_but_profitable_does_not_exit(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker)
        pos.pnl_usd = 25.0
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=102.0, atr=2.0, bars_held=33,
        )
        assert rev["action"] != "exit"

    def test_threshold_boundary(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker)
        pos.pnl_usd = -1.0
        cfg_mr = MechanicalReviewConfig(stale_bars_max=10)
        rev1 = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=99.0, atr=2.0, bars_held=9, cfg=cfg_mr,
        )
        assert rev1["action"] == "hold"
        rev2 = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=99.0, atr=2.0, bars_held=10, cfg=cfg_mr,
        )
        assert rev2["action"] == "exit"


# ── lock-in after TP1 ────────────────────────────────────────────────────────

class TestPostTp1LockIn:

    def test_long_tightens_to_breakeven_plus_cushion(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker, entry=100.0, atr=2.0)
        pos.tp1_hit = True
        pos.pnl_usd = 50.0     # well above min
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=104.5, atr=2.0, bars_held=8,
        )
        assert rev["action"] == "tighten_stop"
        # cushion default 0.25 -> 100 + 0.5 = 100.5
        assert rev["new_stop_loss"] == pytest.approx(100.5, abs=0.01)
        assert rev["new_stop_loss"] > pos.stop_loss        # an improvement
        assert rev["new_stop_loss"] < 104.5                 # below mark

    def test_short_tightens_to_breakeven(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        eng    = RiskEngine(_cfg(tmp_path))
        sizing = eng.compute(side="short", entry=100.0, atr=2.0, equity_usd=10_000.0)
        pos    = broker.open("BTCUSDT", sizing=sizing)
        assert pos is not None
        pos.tp1_hit = True
        pos.pnl_usd = 30.0
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=95.5, atr=2.0, bars_held=8,
        )
        assert rev["action"] == "tighten_stop"
        # short cushion: 100 - 0.25 * 2 = 99.5
        assert rev["new_stop_loss"] == pytest.approx(99.5, abs=0.01)
        assert rev["new_stop_loss"] < pos.stop_loss
        assert rev["new_stop_loss"] > 95.5

    def test_tp1_hit_but_pnl_too_small_holds(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker)
        pos.tp1_hit = True
        pos.pnl_usd = 0.5     # below min_pnl_for_tighten_usd default 1.0
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=100.05, atr=2.0, bars_held=5,
        )
        assert rev["action"] == "hold"


# ── adverse retracement ──────────────────────────────────────────────────────

class TestAdverseRetracement:

    def test_giveback_triggers_be_tighten(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker, entry=100.0, atr=2.0)
        pos.peak_pnl = 100.0
        pos.pnl_usd  = 30.0     # gave back 70%
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=101.0, atr=2.0, bars_held=10,
        )
        assert rev["action"] == "tighten_stop"
        assert rev["new_stop_loss"] == pytest.approx(100.0, abs=0.01)  # exact BE

    def test_small_giveback_holds(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker, entry=100.0, atr=2.0)
        pos.peak_pnl = 100.0
        pos.pnl_usd  = 80.0     # gave back 20%
        rev = mechanical_review(
            symbol="BTCUSDT", position=pos, mark_price=103.0, atr=2.0, bars_held=10,
        )
        assert rev["action"] == "hold"


# ── PaperBroker.tighten_stop ─────────────────────────────────────────────────

class TestTightenStopBroker:

    def test_long_tighten_accepted(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker, entry=100.0, atr=2.0)
        old_sl = pos.stop_loss
        ok = broker.tighten_stop("BTCUSDT", new_stop=98.5)
        assert ok
        assert broker.get("BTCUSDT").stop_loss == pytest.approx(98.5)
        assert broker.get("BTCUSDT").stop_loss > old_sl

    def test_long_widen_rejected(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        pos    = _open_long(broker, entry=100.0, atr=2.0)
        old_sl = pos.stop_loss
        # try to widen (move stop further from price)
        ok = broker.tighten_stop("BTCUSDT", new_stop=old_sl - 1.0)
        assert not ok
        assert broker.get("BTCUSDT").stop_loss == pytest.approx(old_sl)

    def test_short_tighten_accepted(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        eng    = RiskEngine(_cfg(tmp_path))
        sizing = eng.compute(side="short", entry=100.0, atr=2.0, equity_usd=10_000.0)
        broker.open("BTCUSDT", sizing=sizing)
        old_sl = broker.get("BTCUSDT").stop_loss   # ~103
        ok = broker.tighten_stop("BTCUSDT", new_stop=old_sl - 1.0)
        assert ok
        assert broker.get("BTCUSDT").stop_loss < old_sl

    def test_short_widen_rejected(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        eng    = RiskEngine(_cfg(tmp_path))
        sizing = eng.compute(side="short", entry=100.0, atr=2.0, equity_usd=10_000.0)
        broker.open("BTCUSDT", sizing=sizing)
        old_sl = broker.get("BTCUSDT").stop_loss
        ok = broker.tighten_stop("BTCUSDT", new_stop=old_sl + 1.0)
        assert not ok

    def test_unknown_symbol(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        assert not broker.tighten_stop("XYZUSDT", new_stop=100.0)
