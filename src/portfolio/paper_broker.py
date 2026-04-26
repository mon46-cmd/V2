"""Paper trading position manager.

Simulates a live futures account without placing real orders.
All state is persisted to ``run_root/positions.json`` so restarts
don't lose open trade context.

Each position lifecycle:
  open  → (optional) tp1_hit → trailing_active → close

Positions are closed by:
  - SL hit
  - TP2 hit
  - Trailing stop ratcheted and hit (after TP1)
  - AI position_review returning CLOSE
  - Manual close

Usage::

    from portfolio.paper_broker import PaperBroker, Position
    from portfolio.risk import RiskEngine, SizingResult

    broker = PaperBroker(cfg)
    broker.load()

    sizing = risk_engine.compute(side=\"long\", entry=67200, atr=810, equity_usd=broker.equity)
    if sizing.accepted:
        pos = broker.open(symbol=\"BTCUSDT\", sizing=sizing)

    # On each new bar:
    closed = broker.update_bar(symbol=\"BTCUSDT\", high=68000, low=66800, close=67900, atr=820)
    for c in closed:
        print(c.symbol, c.pnl_usd, c.exit_reason)

    broker.save()
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

from core.config import Config
from portfolio.risk import Levels, SizingResult, TrailingStop

log = logging.getLogger(__name__)

Side = Literal["long", "short"]
CloseReason = Literal["sl_hit", "tp2_hit", "trailing_stop", "ai_close", "manual", "tp1_partial"]


# ── Open Position ──────────────────────────────────────────────────────────────

@dataclass
class Position:
    """A single open paper trade."""

    id:            str        # uuid
    symbol:        str
    side:          Side
    entry:         float
    qty_usd:       float      # notional
    qty_contracts: float
    stop_loss:     float
    tp1:           float
    tp2:           float
    stop_distance: float
    atr_used:      float
    risk_usd:      float
    opened_at:     str        # ISO UTC

    # Runtime state (not persisted as primary keys — derived on reload)
    tp1_hit:       bool  = False
    pnl_usd:       float = 0.0   # running unrealised P&L
    peak_pnl:      float = 0.0   # best seen P&L (for drawdown tracking)

    # Trailing stop (serialised separately)
    trail_sl:      float | None = None   # current trailing stop price (None until TP1)

    @property
    def is_long(self) -> bool:
        return self.side == "long"

    def mark_to_market(self, price: float) -> float:
        """Compute unrealised P&L in USD at a given mark price."""
        if self.entry <= 0:
            return 0.0
        pnl_pct = (price - self.entry) / self.entry if self.is_long else (self.entry - price) / self.entry
        return self.qty_usd * pnl_pct

    def sl_hit(self, low: float, high: float) -> bool:
        if self.is_long:
            return low <= self.stop_loss
        return high >= self.stop_loss

    def tp1_reached(self, high: float, low: float) -> bool:
        if self.tp1_hit:
            return False
        return (high >= self.tp1) if self.is_long else (low <= self.tp1)

    def tp2_reached(self, high: float, low: float) -> bool:
        return (high >= self.tp2) if self.is_long else (low <= self.tp2)


# ── Closed Trade Record ────────────────────────────────────────────────────────

@dataclass
class ClosedTrade:
    """Immutable record of a completed trade."""

    id:           str
    symbol:       str
    side:         Side
    entry:        float
    exit_price:   float
    qty_usd:      float
    pnl_usd:      float
    pnl_pct:      float
    risk_usd:     float
    risk_reward:  float        # actual R:R achieved
    opened_at:    str
    closed_at:    str
    exit_reason:  CloseReason
    tp1_hit:      bool


# ── Paper Broker ───────────────────────────────────────────────────────────────

class PaperBroker:
    """File-backed paper trading account.

    Args:
        cfg:            Loaded Config.
        positions_path: Override for the positions JSON file.
        history_path:   Override for the trade history JSONL file.
    """

    def __init__(
        self,
        cfg: Config,
        *,
        positions_path: Path | None = None,
        history_path:   Path | None = None,
    ) -> None:
        self._cfg        = cfg
        self._pos_path   = positions_path or (cfg.run_root / "positions.json")
        self._hist_path  = history_path   or (cfg.run_root / "trade_history.jsonl")
        self._positions: dict[str, Position] = {}   # symbol -> Position
        self._equity:    float = float(cfg.paper_equity_usd)
        self._peak_equity: float = float(cfg.paper_equity_usd)
        self._loser_streak: int = 0
        # Portfolio-level circuit breaker: halt opening new positions when
        # drawdown from peak exceeds this fraction (0 disables).
        self._max_drawdown_pct: float = float(
            getattr(cfg, "max_drawdown_pct", 0.0) or 0.0
        )

    # ── Public read ─────────────────────────────────────────────────────────

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak as a fraction (0.0 = at peak, 0.05 = 5% DD)."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._equity) / self._peak_equity)

    def is_halted(self) -> bool:
        """True when drawdown from peak equity exceeds ``max_drawdown_pct``.

        While halted, the scanner must NOT open new positions.  Existing
        positions continue to be managed (SL/TP/trailing/AI close) so they
        can naturally exit. Returns False if no limit configured.
        """
        if self._max_drawdown_pct <= 0:
            return False
        return self.drawdown_pct >= self._max_drawdown_pct

    @property
    def loser_streak(self) -> int:
        return self._loser_streak

    @property
    def n_open(self) -> int:
        return len(self._positions)

    @property
    def open_symbols(self) -> list[str]:
        return list(self._positions.keys())

    def get(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    def open_positions(self) -> list[Position]:
        return list(self._positions.values())

    # ── Open ────────────────────────────────────────────────────────────────

    def open(self, symbol: str, *, sizing: SizingResult) -> Position | None:
        """Open a new paper position from a SizingResult.

        Returns None if:
        - Symbol already has an open position.
        - sizing.accepted is False.
        """
        if not sizing.accepted:
            log.warning("broker.open rejected: %s — %s", symbol, sizing.reason)
            return None
        if symbol in self._positions:
            log.warning("broker.open skipped: %s already has open position", symbol)
            return None

        lv  = sizing.levels
        pos = Position(
            id            = str(uuid.uuid4())[:8],
            symbol        = symbol,
            side          = lv.side,
            entry         = lv.entry,
            qty_usd       = sizing.qty_usd,
            qty_contracts = sizing.qty_contracts,
            stop_loss     = lv.stop_loss,
            tp1           = lv.tp1,
            tp2           = lv.tp2,
            stop_distance = lv.stop_distance,
            atr_used      = lv.atr_used,
            risk_usd      = sizing.risk_usd,
            opened_at     = _now_iso(),
        )
        self._positions[symbol] = pos
        log.info(
            "OPEN  %-10s %s  entry=%.4f  sl=%.4f  tp1=%.4f  tp2=%.4f  qty=$%.0f",
            symbol, pos.side.upper(), pos.entry, pos.stop_loss, pos.tp1, pos.tp2, pos.qty_usd,
        )
        return pos

    # ── Bar update ──────────────────────────────────────────────────────────

    def update_bar(
        self,
        symbol: str,
        *,
        high:  float,
        low:   float,
        close: float,
        atr:   float,
    ) -> list[ClosedTrade]:
        """Process one closed bar for a symbol.  Returns list of newly closed trades.

        Order of checks (matches realistic intra-bar sequence):
          1. TP1 check → activate trailing stop
          2. TP2 check → full close
          3. Stop-loss check (initial or trailing)
        """
        pos = self._positions.get(symbol)
        if pos is None:
            return []

        closed: list[ClosedTrade] = []

        # ── TP1 hit: activate trailing stop ──────────────────────────────
        if pos.tp1_reached(high, low):
            pos.tp1_hit = True
            pos.trail_sl = pos.stop_loss   # trailing starts at hard SL
            log.info("TP1   %-10s %s  tp1=%.4f", symbol, pos.side.upper(), pos.tp1)

        # ── TP2: full close ───────────────────────────────────────────────
        if pos.tp2_reached(high, low):
            exit_px = pos.tp2
            closed.append(self._close(symbol, exit_price=exit_px, reason="tp2_hit"))
            return closed

        # ── Trailing stop update & check ──────────────────────────────────
        if pos.tp1_hit and pos.trail_sl is not None:
            ts = TrailingStop(
                side           = pos.side,
                initial_sl     = pos.stop_loss,
                trail_atr_mult = getattr(self._cfg, "trail_atr_mult", 1.0),
                active         = True,
            )
            ts._current_sl  = pos.trail_sl
            ts._best_price  = high if pos.is_long else low
            new_sl = ts.update(mark=close, atr=atr)
            pos.trail_sl = new_sl

            if ts.is_hit(close):
                closed.append(self._close(symbol, exit_price=new_sl, reason="trailing_stop"))
                return closed

        # ── Hard SL check ────────────────────────────────────────────────
        if pos.sl_hit(low, high):
            exit_px = pos.stop_loss
            closed.append(self._close(symbol, exit_price=exit_px, reason="sl_hit"))
            return closed

        # ── Mark-to-market P&L update ─────────────────────────────────────
        pos.pnl_usd  = pos.mark_to_market(close)
        pos.peak_pnl = max(pos.peak_pnl, pos.pnl_usd)

        return closed

    # ── AI-triggered close ───────────────────────────────────────────────────

    def ai_close(self, symbol: str, *, exit_price: float) -> ClosedTrade | None:
        """Close a position on AI recommendation."""
        if symbol not in self._positions:
            return None
        return self._close(symbol, exit_price=exit_price, reason="ai_close")

    def manual_close(self, symbol: str, *, exit_price: float) -> ClosedTrade | None:
        """Close a position manually (for testing / override)."""
        if symbol not in self._positions:
            return None
        return self._close(symbol, exit_price=exit_price, reason="manual")

    # ── Stop adjustment ──────────────────────────────────────────────────────

    def tighten_stop(self, symbol: str, *, new_stop: float) -> bool:
        """Tighten an open position's stop-loss (one-way ratchet).

        The stop may only move in the direction of profit:
          - Longs:  new_stop must be strictly greater than current stop
                    AND strictly below the entry-side risk floor (i.e. > current SL).
          - Shorts: new_stop must be strictly less than current stop.

        If the position has TP1 hit (trailing active), the same rule is applied
        to ``trail_sl`` so subsequent bars use the tightened level.

        Returns True if the stop was moved, False otherwise.
        """
        pos = self._positions.get(symbol)
        if pos is None or new_stop <= 0:
            return False

        if pos.is_long and new_stop <= pos.stop_loss:
            return False
        if (not pos.is_long) and new_stop >= pos.stop_loss:
            return False

        old = pos.stop_loss
        pos.stop_loss = float(new_stop)
        # Keep the trailing-stop level consistent: never let trail_sl be
        # looser than the new hard stop.
        if pos.trail_sl is not None:
            if pos.is_long:
                pos.trail_sl = max(pos.trail_sl, pos.stop_loss)
            else:
                pos.trail_sl = min(pos.trail_sl, pos.stop_loss)
        log.info(
            "TIGHTEN %-10s %s  sl: %.6f -> %.6f",
            symbol, pos.side.upper(), old, pos.stop_loss,
        )
        return True

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Atomically persist open positions and equity to disk."""
        self._pos_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "equity":       self._equity,
            "peak_equity":  self._peak_equity,
            "loser_streak": self._loser_streak,
            "saved_at":     _now_iso(),
            "positions":    {sym: asdict(pos) for sym, pos in self._positions.items()},
        }
        _atomic_write(self._pos_path, json.dumps(payload, indent=2))
        log.debug("broker.save: %d open, equity=$%.2f", len(self._positions), self._equity)

    def load(self) -> None:
        """Restore state from disk (best-effort)."""
        if not self._pos_path.exists():
            log.info("broker.load: no positions file, starting fresh equity=$%.2f", self._equity)
            return
        try:
            raw = json.loads(self._pos_path.read_text(encoding="utf-8"))
            self._equity       = float(raw.get("equity", self._equity))
            # Backward compatible: if no peak persisted, take max(starting, current).
            self._peak_equity  = float(raw.get("peak_equity", max(self._peak_equity, self._equity)))
            self._loser_streak = int(raw.get("loser_streak", 0))
            for sym, d in raw.get("positions", {}).items():
                # Handle missing fields with defaults for forward-compat
                d.setdefault("tp1_hit", False)
                d.setdefault("pnl_usd", 0.0)
                d.setdefault("peak_pnl", 0.0)
                d.setdefault("trail_sl", None)
                self._positions[sym] = Position(**{k: v for k, v in d.items() if k in Position.__dataclass_fields__})
            log.info(
                "broker.load: %d open positions, equity=$%.2f, streak=%d",
                len(self._positions), self._equity, self._loser_streak,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("broker.load failed: %s", exc)

    def summary(self) -> dict:
        """Return a human-readable summary dict for logging / status endpoint."""
        return {
            "equity":        round(self._equity, 2),
            "peak_equity":   round(self._peak_equity, 2),
            "drawdown_pct":  round(self.drawdown_pct * 100, 3),
            "halted":        self.is_halted(),
            "n_open":        self.n_open,
            "loser_streak":  self._loser_streak,
            "open_symbols":  self.open_symbols,
            "unrealised_pnl": round(sum(p.pnl_usd for p in self._positions.values()), 2),
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _close(
        self,
        symbol: str,
        *,
        exit_price: float,
        reason: CloseReason,
    ) -> ClosedTrade:
        pos = self._positions.pop(symbol)
        pnl = pos.mark_to_market(exit_price)
        pnl_pct  = pnl / pos.qty_usd if pos.qty_usd else 0.0
        rr_actual = pnl / pos.risk_usd if pos.risk_usd > 0 else 0.0

        # Update equity (and peak high-water mark)
        self._equity = max(0.0, self._equity + pnl)
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Circuit breaker streak
        if pnl < 0:
            self._loser_streak += 1
        else:
            self._loser_streak = 0

        # Halt warning when crossing the drawdown threshold
        if self.is_halted():
            log.warning(
                "DRAWDOWN HALT: equity=$%.2f peak=$%.2f dd=%.2f%% threshold=%.2f%% "
                "-- new entries blocked until equity recovers",
                self._equity, self._peak_equity,
                self.drawdown_pct * 100, self._max_drawdown_pct * 100,
            )

        trade = ClosedTrade(
            id          = pos.id,
            symbol      = symbol,
            side        = pos.side,
            entry       = pos.entry,
            exit_price  = exit_price,
            qty_usd     = pos.qty_usd,
            pnl_usd     = round(pnl, 4),
            pnl_pct     = round(pnl_pct * 100, 4),
            risk_usd    = pos.risk_usd,
            risk_reward = round(rr_actual, 3),
            opened_at   = pos.opened_at,
            closed_at   = _now_iso(),
            exit_reason = reason,
            tp1_hit     = pos.tp1_hit,
        )

        sign = "+" if pnl >= 0 else ""
        log.info(
            "CLOSE %-10s %s  entry=%.4f  exit=%.4f  pnl=%s$%.2f  rr=%.2fR  reason=%s  equity=$%.2f",
            symbol, pos.side.upper(), pos.entry, exit_price,
            sign, pnl, rr_actual, reason, self._equity,
        )

        # Append to trade history JSONL
        self._append_history(trade)

        return trade

    def _append_history(self, trade: ClosedTrade) -> None:
        try:
            self._hist_path.parent.mkdir(parents=True, exist_ok=True)
            with self._hist_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(trade)) + "\n")
        except OSError as exc:
            log.warning("broker history write failed: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
