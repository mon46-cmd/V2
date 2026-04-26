"""Tests for PaperBroker and RiskEngine — core paper trading path.

Run::
    pytest tests/test_paper_trading.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from portfolio.paper_broker import ClosedTrade, PaperBroker, Position
from portfolio.risk import Levels, RiskEngine, SizingResult, TrailingStop


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _cfg(tmp_path: Path) -> SimpleNamespace:
    """Minimal config-like object."""
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
        review_every_bars    = 4,
    )


def _make_sizing(side: str = "long", entry: float = 100.0, atr: float = 2.0) -> SizingResult:
    risk = RiskEngine(_cfg(Path(tempfile.mkdtemp())))
    return risk.compute(side=side, entry=entry, atr=atr, equity_usd=10_000.0, n_open=0)


# ── RiskEngine unit tests ─────────────────────────────────────────────────────

class TestRiskEngine:

    def test_long_levels(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        lv = eng.compute_levels(side="long", entry=100.0, atr=2.0)
        # stop = 100 - 1.5*2 = 97
        assert lv.stop_loss == pytest.approx(97.0, abs=0.01)
        # tp1 = 100 + 3 * 1.5 = 104.5
        assert lv.tp1 == pytest.approx(104.5, abs=0.01)
        # tp2 = 100 + 3 * 3.0 = 109.0
        assert lv.tp2 == pytest.approx(109.0, abs=0.01)
        assert lv.risk_reward == 3.0

    def test_short_levels(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        lv = eng.compute_levels(side="short", entry=100.0, atr=2.0)
        assert lv.stop_loss == pytest.approx(103.0, abs=0.01)   # 100 + 3
        assert lv.tp1 == pytest.approx(95.5, abs=0.01)           # 100 - 4.5
        assert lv.tp2 == pytest.approx(91.0, abs=0.01)           # 100 - 9.0

    def test_sizing_basic(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        lv  = eng.compute_levels(side="long", entry=100.0, atr=2.0)
        sr  = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=0)
        # risk = 10000 * 0.01 = $100; stop_pct = 3/100 = 0.03; qty = 100/0.03 = 3333
        assert sr.accepted
        assert sr.qty_usd == pytest.approx(3333.33, abs=10)
        assert sr.risk_usd == pytest.approx(100.0, abs=1)

    def test_sizing_capped_at_max_pos(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        lv  = eng.compute_levels(side="long", entry=100.0, atr=0.01)   # tiny stop → huge size
        sr  = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=0)
        assert sr.accepted
        assert sr.qty_usd <= 5_000.0

    def test_sizing_rejected_max_open(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        lv  = eng.compute_levels(side="long", entry=100.0, atr=2.0)
        sr  = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=3)   # at max
        assert not sr.accepted
        assert "max_open_positions" in sr.reason

    def test_circuit_breaker_half(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        lv  = eng.compute_levels(side="long", entry=100.0, atr=2.0)
        sr3 = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=0, loser_streak=3)
        sr0 = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=0, loser_streak=0)
        assert sr3.qty_usd == pytest.approx(sr0.qty_usd * 0.5, rel=0.01)

    def test_circuit_breaker_quarter(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        lv  = eng.compute_levels(side="long", entry=100.0, atr=2.0)
        sr5 = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=0, loser_streak=5)
        sr0 = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=0, loser_streak=0)
        assert sr5.qty_usd == pytest.approx(sr0.qty_usd * 0.25, rel=0.01)

    def test_atr_from_pct(self, tmp_path):
        eng = RiskEngine(_cfg(tmp_path))
        assert eng.atr_from_pct(100.0, 2.0) == pytest.approx(2.0)
        assert eng.atr_from_pct(50000.0, 1.5) == pytest.approx(750.0)


# ── TrailingStop unit tests ───────────────────────────────────────────────────

class TestTrailingStop:

    def test_long_ratchet(self):
        ts = TrailingStop(side="long", initial_sl=95.0, trail_atr_mult=1.0)
        ts.activate(100.0)
        sl1 = ts.update(mark=105.0, atr=2.0)   # best=105 → trail=103
        assert sl1 == pytest.approx(103.0)
        sl2 = ts.update(mark=102.0, atr=2.0)   # best=105 → trail still 103 (ratchet)
        assert sl2 == pytest.approx(103.0)

    def test_long_hit(self):
        ts = TrailingStop(side="long", initial_sl=95.0, trail_atr_mult=1.0)
        ts.activate(100.0)
        ts.update(mark=105.0, atr=2.0)   # trail=103
        assert not ts.is_hit(104.0)
        assert ts.is_hit(103.0)

    def test_short_ratchet(self):
        ts = TrailingStop(side="short", initial_sl=105.0, trail_atr_mult=1.0)
        ts.activate(100.0)
        sl1 = ts.update(mark=95.0, atr=2.0)   # best=95 → trail=97
        assert sl1 == pytest.approx(97.0)
        sl2 = ts.update(mark=98.0, atr=2.0)   # best=95 → trail still 97 (ratchet)
        assert sl2 == pytest.approx(97.0)

    def test_inactive_returns_initial_sl(self):
        ts = TrailingStop(side="long", initial_sl=90.0)
        assert ts.update(mark=110.0, atr=2.0) == 90.0


# ── PaperBroker unit tests ────────────────────────────────────────────────────

class TestPaperBroker:

    def test_open_position(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        sizing = _make_sizing()
        pos    = broker.open("BTCUSDT", sizing=sizing)
        assert pos is not None
        assert broker.n_open == 1
        assert "BTCUSDT" in broker.open_symbols

    def test_open_duplicate_rejected(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        sizing = _make_sizing()
        broker.open("BTCUSDT", sizing=sizing)
        pos2 = broker.open("BTCUSDT", sizing=sizing)
        assert pos2 is None
        assert broker.n_open == 1

    def test_open_rejected_sizing(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        eng    = RiskEngine(_cfg(tmp_path))
        lv     = eng.compute_levels(side="long", entry=100.0, atr=2.0)
        sr     = eng.compute_size(levels=lv, equity_usd=10_000.0, n_open=3)   # rejected
        pos    = broker.open("BTCUSDT", sizing=sr)
        assert pos is None

    def test_sl_hit_long(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        sizing = _make_sizing(entry=100.0, atr=2.0)
        broker.open("BTCUSDT", sizing=sizing)
        # SL at ~97; low=96 should trigger
        closed = broker.update_bar("BTCUSDT", high=101.0, low=96.0, close=99.0, atr=2.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "sl_hit"
        assert closed[0].pnl_usd < 0
        assert broker.n_open == 0
        assert broker.loser_streak == 1

    def test_tp2_hit_long(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        sizing = _make_sizing(entry=100.0, atr=2.0)
        broker.open("BTCUSDT", sizing=sizing)
        # TP2 at ~109; high=112 should trigger
        closed = broker.update_bar("BTCUSDT", high=112.0, low=108.0, close=111.0, atr=2.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "tp2_hit"
        assert closed[0].pnl_usd > 0
        assert broker.loser_streak == 0

    def test_tp1_then_trailing_stop(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        sizing = _make_sizing(entry=100.0, atr=2.0)
        broker.open("BTCUSDT", sizing=sizing)
        pos = broker.get("BTCUSDT")
        # TP1 at ~104.5
        broker.update_bar("BTCUSDT", high=106.0, low=103.0, close=105.0, atr=2.0)
        pos = broker.get("BTCUSDT")
        assert pos.tp1_hit
        # Trailing stop activates; price reverses back to sl
        closed = broker.update_bar("BTCUSDT", high=105.5, low=94.0, close=94.5, atr=2.0)
        assert len(closed) == 1
        assert closed[0].exit_reason in ("trailing_stop", "sl_hit")

    def test_equity_updated_on_close(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        eq0    = broker.equity
        sizing = _make_sizing(entry=100.0, atr=2.0)
        broker.open("BTCUSDT", sizing=sizing)
        broker.update_bar("BTCUSDT", high=112.0, low=108.0, close=111.0, atr=2.0)
        assert broker.equity != eq0   # equity changed after TP2

    def test_ai_close(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        broker.open("BTCUSDT", sizing=_make_sizing())
        trade = broker.ai_close("BTCUSDT", exit_price=101.0)
        assert trade is not None
        assert trade.exit_reason == "ai_close"
        assert broker.n_open == 0

    def test_multi_position(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            broker.open(sym, sizing=_make_sizing())
        assert broker.n_open == 3

    def test_max_positions_block(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        eng    = RiskEngine(_cfg(tmp_path))
        # Open 3 (max)
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            broker.open(sym, sizing=_make_sizing())
        # 4th should be rejected at sizing level
        lv = eng.compute_levels(side="long", entry=100.0, atr=2.0)
        sr = eng.compute_size(levels=lv, equity_usd=broker.equity, n_open=broker.n_open)
        assert not sr.accepted

    def test_save_load_roundtrip(self, tmp_path):
        broker1 = PaperBroker(_cfg(tmp_path))
        broker1.open("BTCUSDT", sizing=_make_sizing(entry=100.0))
        broker1.save()

        broker2 = PaperBroker(_cfg(tmp_path))
        broker2.load()
        assert broker2.n_open == 1
        assert broker2.equity == pytest.approx(broker1.equity, abs=0.01)
        pos = broker2.get("BTCUSDT")
        assert pos is not None
        assert pos.entry == pytest.approx(100.0)

    def test_trade_history_written(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        broker.open("BTCUSDT", sizing=_make_sizing(entry=100.0, atr=2.0))
        broker.update_bar("BTCUSDT", high=112.0, low=109.0, close=111.0, atr=2.0)
        hist_path = broker._hist_path
        assert hist_path.exists()
        lines = hist_path.read_text().strip().splitlines()
        assert len(lines) == 1
        trade = json.loads(lines[0])
        assert trade["symbol"] == "BTCUSDT"
        assert trade["exit_reason"] == "tp2_hit"

    def test_short_sl_hit(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        sizing = _make_sizing(side="short", entry=100.0, atr=2.0)
        broker.open("ETHUSDT", sizing=sizing)
        # SL at ~103; high=105 should trigger
        closed = broker.update_bar("ETHUSDT", high=105.0, low=99.0, close=104.0, atr=2.0)
        assert len(closed) == 1
        assert closed[0].exit_reason == "sl_hit"
        assert closed[0].pnl_usd < 0

    def test_loser_streak_resets_on_win(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        # Simulate a loss
        broker.open("BTCUSDT", sizing=_make_sizing(entry=100.0, atr=2.0))
        broker.update_bar("BTCUSDT", high=99.0, low=96.0, close=97.0, atr=2.0)  # SL
        assert broker.loser_streak == 1
        # Now a win
        broker.open("ETHUSDT", sizing=_make_sizing(entry=100.0, atr=2.0))
        broker.update_bar("ETHUSDT", high=112.0, low=109.0, close=111.0, atr=2.0)  # TP2
        assert broker.loser_streak == 0

    def test_summary(self, tmp_path):
        broker = PaperBroker(_cfg(tmp_path))
        broker.open("BTCUSDT", sizing=_make_sizing())
        s = broker.summary()
        assert s["n_open"] == 1
        assert s["equity"] == pytest.approx(10_000.0)
        assert "BTCUSDT" in s["open_symbols"]


# ── Integration: trigger → open → bar update → close ─────────────────────────

class TestIntegration:

    def test_full_lifecycle_long(self, tmp_path):
        """Trigger fires → position opens → TP2 hit → closes with profit."""
        from loops.triggers import TriggerDecision, DEC_FRESH
        from features.layer7_flags import FLAG_COLUMNS

        eng    = RiskEngine(_cfg(tmp_path))
        broker = PaperBroker(_cfg(tmp_path))

        # Simulate a trigger decision
        dec = TriggerDecision(
            symbol   = "BTCUSDT",
            bar_ts   = __import__("pandas").Timestamp("2024-01-01 12:00", tz="UTC"),
            decision = DEC_FRESH,
            flag     = "flag_volume_climax",
            close    = 100.0,
            atr_pct  = 2.0,
        )
        assert dec.fired

        # Open position
        atr    = eng.atr_from_pct(dec.close, dec.atr_pct)
        sizing = eng.compute(side="long", entry=dec.close, atr=atr, equity_usd=broker.equity)
        broker.open("BTCUSDT", sizing=sizing)
        assert broker.n_open == 1

        # TP2 hit
        pos    = broker.get("BTCUSDT")
        closed = broker.update_bar("BTCUSDT",
                                   high=pos.tp2 + 1, low=pos.tp2 - 0.5,
                                   close=pos.tp2, atr=atr)
        assert len(closed) == 1
        trade = closed[0]
        assert trade.pnl_usd > 0
        assert trade.exit_reason == "tp2_hit"
        assert broker.n_open == 0
        assert broker.equity > 10_000.0


# ── Drawdown halt (portfolio-level circuit breaker) ───────────────────────────

class TestDrawdownHalt:

    def _cfg_with_dd(self, tmp_path: Path, dd_pct: float) -> SimpleNamespace:
        cfg = _cfg(tmp_path)
        cfg.max_drawdown_pct = dd_pct
        return cfg

    def test_not_halted_at_start(self, tmp_path):
        broker = PaperBroker(self._cfg_with_dd(tmp_path, 0.05))
        assert broker.drawdown_pct == 0.0
        assert not broker.is_halted()

    def test_disabled_when_threshold_zero(self, tmp_path):
        broker = PaperBroker(self._cfg_with_dd(tmp_path, 0.0))
        # Force a synthetic loss
        broker._equity = 5_000.0
        broker._peak_equity = 10_000.0
        assert broker.drawdown_pct == pytest.approx(0.5)
        assert not broker.is_halted()   # threshold 0 disables halt

    def test_halts_when_threshold_exceeded(self, tmp_path):
        broker = PaperBroker(self._cfg_with_dd(tmp_path, 0.05))
        broker._equity = 9_400.0          # 6% DD
        broker._peak_equity = 10_000.0
        assert broker.drawdown_pct == pytest.approx(0.06, abs=0.001)
        assert broker.is_halted()

    def test_peak_tracks_winners(self, tmp_path):
        broker = PaperBroker(self._cfg_with_dd(tmp_path, 0.05))
        # Win raises peak
        broker.open("BTCUSDT", sizing=_make_sizing(entry=100.0, atr=2.0))
        broker.update_bar("BTCUSDT", high=112.0, low=109.0, close=111.0, atr=2.0)  # TP2
        assert broker.peak_equity == pytest.approx(broker.equity)
        assert broker.peak_equity > 10_000.0

    def test_peak_persists_across_save_load(self, tmp_path):
        cfg = self._cfg_with_dd(tmp_path, 0.05)
        b1 = PaperBroker(cfg)
        b1.open("BTCUSDT", sizing=_make_sizing(entry=100.0, atr=2.0))
        b1.update_bar("BTCUSDT", high=112.0, low=109.0, close=111.0, atr=2.0)
        peak = b1.peak_equity
        b1.save()

        b2 = PaperBroker(cfg)
        b2.load()
        assert b2.peak_equity == pytest.approx(peak, abs=0.01)

    def test_halt_engages_after_loss_streak(self, tmp_path):
        cfg = self._cfg_with_dd(tmp_path, 0.02)   # 2% threshold (easy to trip)
        broker = PaperBroker(cfg)
        # Force a big loss by short-circuiting equity
        broker._equity = 9_700.0          # 3% drawdown
        broker._peak_equity = 10_000.0
        assert broker.is_halted()
        s = broker.summary()
        assert s["halted"] is True
        assert s["drawdown_pct"] == pytest.approx(3.0, abs=0.01)
