"""Mechanical (deterministic) position-review fallback.

When the AI position-review call fails (network error, budget exhausted,
schema mismatch, kill switch on, ...) the position-review prompt's safety
contract returns ``action="hold"``.  That is correct from a *do-no-harm*
standpoint, but it leaves the position drifting without active risk
management when the AI is unavailable for an extended period.

This module provides a small, deterministic risk-protective fallback that
the scanner can apply *instead of* a blind hold when the AI is down.

Rules (applied in priority order; the first matching rule wins):

  1. **Profit lock-in after TP1.**  If ``tp1_hit`` and the trail is still
     at the original stop, tighten the stop to break-even (or break-even
     plus a small ATR cushion).  This guarantees the trade cannot turn
     into a loss after taking partial profit.

  2. **Time-based stale exit.**  If the position has been open for more
     than ``stale_bars_max`` bars and is still under water, exit.
     Holding losers indefinitely is the most expensive failure mode.

  3. **Adverse-move tighten.**  If the position is unrealised-positive
     and the close has retraced more than ``adverse_retrace_pct`` of the
     peak unrealised gain, tighten to break-even.

  4. **Hold** otherwise.

The output schema is identical to ``ai.schemas.PositionReview`` so the
scanner can act on it through the same code path it uses for AI reviews.

This is intentionally conservative: it never *opens* anything, never
*scales out*, never *flips*.  Its only job is to protect capital while
the AI is unavailable.

Usage::

    from portfolio.mechanical_review import mechanical_review

    # When the AI call has failed or is disabled:
    review = mechanical_review(
        symbol     = pos.symbol,
        position   = pos,
        mark_price = float(last_bar["close"]),
        atr        = atr,
        bars_held  = bars_held,
    )
    # review is shaped like ai.schemas.PositionReview
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ai.schemas import PositionReview
from portfolio.paper_broker import Position

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration (all overridable via Config)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MechanicalReviewConfig:
    """Tunable thresholds for the deterministic fallback."""

    # Hard exit when underwater for more than this many 15m bars.
    # 32 bars * 15m = 8h, matching the "time_exit_8h" hook reason.
    stale_bars_max: int = 32

    # ATR cushion above break-even when locking in profit (in ATR units).
    # 0.0 = exact break-even; 0.25 = entry + 0.25 * ATR for longs.
    breakeven_cushion_atr: float = 0.25

    # Fraction of peak unrealised PnL that must have been given back
    # before tightening to break-even (0.6 = 60% giveback).
    adverse_retrace_pct: float = 0.60

    # Minimum unrealised PnL (in $) before any tightening rule fires.
    # Avoids thrashing on tiny noise.
    min_pnl_for_tighten_usd: float = 1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mechanical_review(
    *,
    symbol:     str,
    position:   Position,
    mark_price: float,
    atr:        float,
    bars_held:  int,
    cfg:        MechanicalReviewConfig | None = None,
) -> PositionReview:
    """Apply deterministic risk rules to an open position.

    Args:
        symbol:     Ticker symbol.
        position:   Open ``Position``.
        mark_price: Current close price.
        atr:        Current ATR in price units.
        bars_held:  Number of 15m bars the position has been open.
        cfg:        Optional override for thresholds.

    Returns:
        A ``PositionReview``-shaped dict.  Always one of:
          - ``action="exit"`` (stale loser),
          - ``action="tighten_stop"`` (lock in profit),
          - ``action="hold"`` (default).

        The reply is fully self-explanatory in ``rationale`` so that the
        audit log makes clear this came from the mechanical fallback,
        not the AI.
    """
    cfg = cfg or MechanicalReviewConfig()

    # 1. Stale + underwater exit.
    if bars_held >= cfg.stale_bars_max and position.pnl_usd <= 0.0:
        log.info(
            "mechanical_review %s: stale exit bars_held=%d pnl=$%.2f",
            symbol, bars_held, position.pnl_usd,
        )
        return _result(
            symbol     = symbol,
            action     = "exit",
            urgency    = "high",
            rationale  = (
                f"mechanical fallback: stale_bars={bars_held} >= "
                f"{cfg.stale_bars_max} and pnl=${position.pnl_usd:.2f} <= 0"
            ),
        )

    # 2. Lock in profit at break-even after TP1.
    if (
        position.tp1_hit
        and atr > 0
        and position.pnl_usd >= cfg.min_pnl_for_tighten_usd
    ):
        new_sl = _breakeven_stop(position, atr, cfg.breakeven_cushion_atr)
        if _is_improvement(position, new_sl, mark_price):
            log.info(
                "mechanical_review %s: lock-in BE after TP1 new_sl=%.6f",
                symbol, new_sl,
            )
            return _result(
                symbol        = symbol,
                action        = "tighten_stop",
                urgency       = "medium",
                new_stop_loss = new_sl,
                rationale     = (
                    f"mechanical fallback: TP1 hit, locking break-even+"
                    f"{cfg.breakeven_cushion_atr:.2f}*ATR (was {position.stop_loss:.6f})"
                ),
            )

    # 3. Adverse-move giveback: was up, given back >X% of peak.
    if (
        position.peak_pnl >= cfg.min_pnl_for_tighten_usd
        and position.pnl_usd < position.peak_pnl
        and atr > 0
    ):
        giveback = (position.peak_pnl - position.pnl_usd) / position.peak_pnl
        if giveback >= cfg.adverse_retrace_pct:
            new_sl = _breakeven_stop(position, atr, 0.0)
            if _is_improvement(position, new_sl, mark_price):
                log.info(
                    "mechanical_review %s: giveback %.0f%% of peak -> BE",
                    symbol, giveback * 100,
                )
                return _result(
                    symbol        = symbol,
                    action        = "tighten_stop",
                    urgency       = "medium",
                    new_stop_loss = new_sl,
                    rationale     = (
                        f"mechanical fallback: gave back {giveback*100:.0f}% "
                        f"of peak pnl (${position.peak_pnl:.2f} -> "
                        f"${position.pnl_usd:.2f}); tighten to BE"
                    ),
                )

    # 4. Default hold.
    return _result(
        symbol    = symbol,
        action    = "hold",
        urgency   = "low",
        rationale = "mechanical fallback: no protective rule matched",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _breakeven_stop(pos: Position, atr: float, cushion_atr: float) -> float:
    """Compute the break-even stop price (entry + cushion in trade direction)."""
    if pos.is_long:
        return pos.entry + cushion_atr * atr
    return pos.entry - cushion_atr * atr


def _is_improvement(pos: Position, new_sl: float, mark: float) -> bool:
    """True when ``new_sl`` is *strictly* a tighter (more protective) stop.

    For longs: must be greater than the current stop *and* below the mark.
    For shorts: must be less than the current stop *and* above the mark.
    Floor at zero / sanity check.
    """
    if new_sl <= 0:
        return False
    if pos.is_long:
        return new_sl > pos.stop_loss and new_sl < mark
    return new_sl < pos.stop_loss and new_sl > mark


def _result(
    *,
    symbol:        str,
    action:        str,
    urgency:       str,
    rationale:     str,
    new_stop_loss: float = 0.0,
) -> PositionReview:
    """Build a ``PositionReview``-shaped dict with safe defaults."""
    return PositionReview(
        review_type   = "position_review",
        symbol        = symbol,
        action        = action,
        confidence    = 1.0,                 # deterministic rule -> full confidence
        new_stop_loss = float(new_stop_loss),
        rationale     = rationale[:500],
        urgency       = urgency,
    )
