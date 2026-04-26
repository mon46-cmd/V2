"""ATR-based stop-loss, take-profit, and position sizing engine.

The AI gives us direction and conviction.  We do NOT trust it for precise
price levels -- it doesn't know tick sizes, current ATR, or our equity.

This module computes ALL price levels mechanically from the feature snapshot.
The AI's entry/SL/TP suggestions are used only as soft signals; the final
levels always come from this engine.

Strategy:
- Stop-loss:     Entry ± (ATR_MULT * ATR_14).  Minimum 0.5x ATR, max 2x ATR.
- Take-profit 1: Entry ± (TP1_RR * stop_distance).  Default 1.5R.
- Take-profit 2: Entry ± (TP2_RR * stop_distance).  Default 3.0R.
- Size:          Risk 1% of equity / stop_distance. Clamped by max_position_usd.

Circuit breaker:
- After 3 consecutive losses, size is halved.
- After 5 consecutive losses, size is quartered (halt mode).
- Size restores on first winner.

Usage::

    from portfolio.risk import RiskEngine, SizingResult
    from core import load_config

    cfg    = load_config()
    engine = RiskEngine(cfg)

    result = engine.compute(
        side="long",
        entry=67_200.0,
        atr=810.0,           # from feature snapshot
        equity_usd=10_000.0,
        n_open=1,
        loser_streak=0,
    )
    print(result.stop_loss, result.tp1, result.tp2, result.qty_usd)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.config import Config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (sensible defaults, all overridable via Config)
# ---------------------------------------------------------------------------

DEFAULT_ATR_MULT_SL  = 1.5    # stop = entry ± 1.5 * ATR
DEFAULT_TP1_RR       = 1.5    # TP1 at 1.5R
DEFAULT_TP2_RR       = 3.0    # TP2 at 3.0R
DEFAULT_RISK_PCT     = 0.01   # risk 1% of equity per trade
DEFAULT_MAX_OPEN     = 3      # max concurrent positions
DEFAULT_MAX_POS_USD  = 5_000  # max notional per position


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Levels:
    """Mechanically computed price levels for one trade."""
    side:          str     # "long" | "short"
    entry:         float   # intended entry price
    stop_loss:     float   # hard stop
    tp1:           float   # first target (partial close)
    tp2:           float   # final target (full close)
    stop_distance: float   # abs(entry - stop_loss)
    atr_used:      float   # ATR that drove the calculation
    risk_reward:   float   # tp2 / stop_distance (actual R:R)


@dataclass
class SizingResult:
    """Combined levels + position size decision."""
    levels:        Levels
    qty_usd:       float   # notional size in USD
    qty_contracts: float   # qty / entry (approximate contract count)
    risk_usd:      float   # actual risk in USD (qty_usd * stop_distance / entry)
    accepted:      bool    # False if sizing was rejected (too small / too large)
    reason:        str     # why accepted or rejected
    risk_mult:     float   # circuit breaker multiplier applied


# ---------------------------------------------------------------------------
# RiskEngine
# ---------------------------------------------------------------------------

class RiskEngine:
    """Compute mechanically sound stop-loss, TP, and position size.

    All price levels are ATR-driven.  The AI's suggested entry is used as
    the reference but its SL/TP suggestions are IGNORED -- we compute our own.

    Args:
        cfg:  Loaded Config.
    """

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_levels(
        self,
        *,
        side:  str,    # "long" | "short"
        entry: float,  # intended entry price
        atr:   float,  # ATR_14 in price units (not %)
    ) -> Levels:
        """Compute stop-loss and take-profit levels for a trade.

        Args:
            side:  Trade direction.
            entry: Entry price.
            atr:   ATR(14) in price units.

        Returns:
            ``Levels`` with stop, tp1, tp2, and R:R.
        """
        cfg         = self._cfg
        atr_mult    = getattr(cfg, "atr_mult_sl",  DEFAULT_ATR_MULT_SL)
        tp1_rr      = getattr(cfg, "tp1_rr",       DEFAULT_TP1_RR)
        tp2_rr      = getattr(cfg, "tp2_rr",       DEFAULT_TP2_RR)
        min_atr_mult = 0.5
        max_atr_mult = 2.0

        # Clamp ATR multiplier to sane range.
        atr_mult = max(min_atr_mult, min(max_atr_mult, float(atr_mult)))

        stop_dist = atr * atr_mult
        if stop_dist <= 0 or entry <= 0:
            log.warning("compute_levels: invalid atr=%.6f or entry=%.6f", atr, entry)
            stop_dist = entry * 0.01  # fallback: 1% of price

        if side == "long":
            sl  = entry - stop_dist
            tp1 = entry + stop_dist * tp1_rr
            tp2 = entry + stop_dist * tp2_rr
        else:
            sl  = entry + stop_dist
            tp1 = entry - stop_dist * tp1_rr
            tp2 = entry - stop_dist * tp2_rr

        return Levels(
            side          = side,
            entry         = entry,
            stop_loss     = round(sl,  8),
            tp1           = round(tp1, 8),
            tp2           = round(tp2, 8),
            stop_distance = stop_dist,
            atr_used      = atr,
            risk_reward   = tp2_rr,
        )

    def compute_size(
        self,
        *,
        levels:       Levels,
        equity_usd:   float,
        n_open:       int       = 0,
        loser_streak: int       = 0,
        vol_regime_score: float = 1.0,
    ) -> SizingResult:
        """Compute position size given levels and current portfolio state.

        Args:
            levels:       From ``compute_levels()``.
            equity_usd:   Current account equity in USD.
            n_open:       Number of currently open positions.
            loser_streak: Consecutive losing trades for circuit breaker.
            vol_regime_score:
                Realised-vol regime score, typically in ``[0.5, 2.5]`` where
                1.0 = neutral, > 1.0 = elevated vol (= reduce size), < 1.0 =
                quiet vol (no upsize -- floor at 1.0).  Defaults to 1.0
                (no adjustment).  The applied multiplier is
                ``clip(1.0 / max(score, 0.5), 0.5, 1.0)`` so size only
                ever shrinks, never grows, in volatile regimes.

        Returns:
            ``SizingResult`` with qty_usd, risk_usd, accepted, reason.
        """
        cfg         = self._cfg
        risk_pct    = getattr(cfg, "risk_pct_per_trade",  DEFAULT_RISK_PCT)
        max_open    = getattr(cfg, "max_open_positions",  DEFAULT_MAX_OPEN)
        max_pos_usd = getattr(cfg, "max_position_usd",   DEFAULT_MAX_POS_USD)

        # --- Circuit breaker ---
        risk_mult = _circuit_breaker_mult(loser_streak)

        # --- Vol-regime mult (size DOWN only, never up) ---
        try:
            score = float(vol_regime_score) if vol_regime_score is not None else 1.0
            if score != score:   # NaN check
                score = 1.0
        except (TypeError, ValueError):
            score = 1.0
        vol_mult = max(0.5, min(1.0, 1.0 / max(score, 0.5)))
        risk_mult = risk_mult * vol_mult

        # --- Capacity check ---
        if n_open >= max_open:
            return SizingResult(
                levels=levels, qty_usd=0, qty_contracts=0, risk_usd=0,
                accepted=False, risk_mult=risk_mult,
                reason=f"max_open_positions={max_open} reached (n_open={n_open})",
            )

        if equity_usd <= 0:
            return SizingResult(
                levels=levels, qty_usd=0, qty_contracts=0, risk_usd=0,
                accepted=False, risk_mult=risk_mult,
                reason="equity_usd <= 0",
            )

        # --- Size: risk $R per trade ---
        # qty_usd = (equity * risk_pct * risk_mult) / (stop_distance / entry)
        risk_usd_target  = equity_usd * risk_pct * risk_mult
        stop_pct_of_entry = levels.stop_distance / levels.entry if levels.entry > 0 else 0.01
        qty_usd           = risk_usd_target / stop_pct_of_entry

        # Clamp to max position notional.
        qty_usd = min(qty_usd, float(max_pos_usd))

        # Minimum trade size: don't open < $10 positions.
        if qty_usd < 10.0:
            return SizingResult(
                levels=levels, qty_usd=0, qty_contracts=0, risk_usd=0,
                accepted=False, risk_mult=risk_mult,
                reason=f"position too small: ${qty_usd:.2f} < $10 minimum",
            )

        actual_risk = qty_usd * stop_pct_of_entry
        qty_contracts = qty_usd / levels.entry if levels.entry > 0 else 0.0

        log.info(
            "sizing accepted symbol=%s side=%s entry=%.4f sl=%.4f tp1=%.4f tp2=%.4f "
            "qty_usd=%.2f risk_usd=%.2f rr=%.1f mult=%.2f",
            "?", levels.side, levels.entry, levels.stop_loss,
            levels.tp1, levels.tp2, qty_usd, actual_risk, levels.risk_reward, risk_mult,
        )

        return SizingResult(
            levels        = levels,
            qty_usd       = round(qty_usd, 2),
            qty_contracts = qty_contracts,
            risk_usd      = round(actual_risk, 2),
            accepted      = True,
            reason        = "ok",
            risk_mult     = risk_mult,
        )

    def compute(
        self,
        *,
        side:         str,
        entry:        float,
        atr:          float,
        equity_usd:   float,
        n_open:       int = 0,
        loser_streak: int = 0,
        vol_regime_score: float = 1.0,
    ) -> SizingResult:
        """One-shot: compute levels and size together.

        Convenience wrapper for ``compute_levels`` + ``compute_size``.
        """
        levels = self.compute_levels(side=side, entry=entry, atr=atr)
        return self.compute_size(
            levels=levels,
            equity_usd=equity_usd,
            n_open=n_open,
            loser_streak=loser_streak,
            vol_regime_score=vol_regime_score,
        )

    def atr_from_pct(self, entry: float, atr_pct: float) -> float:
        """Convert ATR expressed as % of price back to price units.

        Feature pipeline stores ``atr_14_pct`` (ATR as % of close).
        This converts it back for use in ``compute_levels``.

        Args:
            entry:   Current entry/close price.
            atr_pct: ATR as percentage (e.g. 1.24 means 1.24%).
        """
        return entry * (atr_pct / 100.0)


# ---------------------------------------------------------------------------
# Trailing stop
# ---------------------------------------------------------------------------

@dataclass
class TrailingStop:
    """Simple ATR-trailing stop tracker for an open position.

    Once TP1 is hit, the stop trails by ``trail_atr_mult * ATR`` behind
    the best-seen price.  This locks in profit without closing early.

    Usage::

        ts = TrailingStop(side="long", initial_sl=66500.0, trail_atr_mult=1.0)
        # on each new bar:
        new_sl = ts.update(mark=68000.0, atr=810.0)
        if mark < new_sl:  # stop hit
            close_position()
    """

    side:            str    # "long" | "short"
    initial_sl:      float  # the hard SL set at entry
    trail_atr_mult:  float  = 1.0  # trail distance in ATR units
    active:          bool   = False  # becomes True after TP1 hit
    _best_price:     float  = field(default=0.0, init=False, repr=False)
    _current_sl:     float  = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._current_sl = self.initial_sl

    def activate(self, price: float) -> None:
        """Activate trailing stop at TP1 price."""
        self.active       = True
        self._best_price  = price
        self._current_sl  = self.initial_sl  # start from hard SL

    def update(self, *, mark: float, atr: float) -> float:
        """Update trailing stop with current mark price and ATR.

        Returns the current stop-loss level.
        Does NOT move the stop adversely (ratchet only).
        """
        if not self.active:
            return self._current_sl

        trail_dist = atr * self.trail_atr_mult

        if self.side == "long":
            self._best_price = max(self._best_price, mark)
            new_sl           = self._best_price - trail_dist
            self._current_sl = max(self._current_sl, new_sl)   # ratchet up only
        else:
            self._best_price = min(self._best_price, mark)
            new_sl           = self._best_price + trail_dist
            self._current_sl = min(self._current_sl, new_sl)   # ratchet down only

        return self._current_sl

    @property
    def current_sl(self) -> float:
        return self._current_sl

    def is_hit(self, mark: float) -> bool:
        """Return True if the current mark has hit the trailing stop."""
        if self.side == "long":
            return mark <= self._current_sl
        return mark >= self._current_sl


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

def _circuit_breaker_mult(loser_streak: int) -> float:
    """Return a risk multiplier based on consecutive losses.

    0-2 losses:  full size (1.0x)
    3-4 losses:  half size (0.5x)
    5+ losses:   quarter size (0.25x) -- near-halt mode
    """
    if loser_streak >= 5:
        return 0.25
    if loser_streak >= 3:
        return 0.50
    return 1.0
