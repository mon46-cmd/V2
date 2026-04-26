"""Pure trigger-gate logic.

``detect_trigger`` is a pure function: given the latest feature snapshot row
for a symbol and the symbol's cooldown state, it returns a ``TriggerDecision``.

Decision codes (string enum for log / JSONL readability):
  fresh          -- first time ever, no cooldown history.
  post_cooldown  -- cooldown expired naturally (enough bars elapsed).
  bypass_move    -- cooldown still active but price moved enough to override.
  cooldown_active -- cooldown blocking, price move insufficient.
  no_flag        -- no configured trigger flag fired on this bar.
  no_bar         -- bar is None or missing required fields.
  dup_bar        -- this bar_ts was already processed for this symbol.

The function does NOT mutate state.  The scanner records results back into
``CooldownStore`` after a positive decision.

Usage::

    from loops.triggers import detect_trigger, TriggerDecision
    from loops.cooldowns import CooldownStore

    store = CooldownStore(path=run_dir / "cooldowns.json")
    store.load()

    for symbol in watchlist_symbols:
        bar   = snapshot_df.loc[symbol].to_dict()
        state = store.state_for(symbol, current_bar_ts=bar["timestamp"])
        dec   = detect_trigger(symbol=symbol, bar=bar, state=state, cfg=cfg)

        if dec.fired:
            store.record(symbol, bar_ts=dec.bar_ts, close=dec.close)
            await on_trigger(symbol, bar, dec)

    store.save()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from core.config import Config
from loops.cooldowns import CooldownState

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision codes
# ---------------------------------------------------------------------------

DEC_FRESH          = "fresh"
DEC_POST_COOLDOWN  = "post_cooldown"
DEC_BYPASS_MOVE    = "bypass_move"
DEC_COOLDOWN       = "cooldown_active"
DEC_NO_FLAG        = "no_flag"
DEC_NO_BAR         = "no_bar"
DEC_DUP_BAR        = "dup_bar"

POSITIVE_DECISIONS: frozenset[str] = frozenset({
    DEC_FRESH, DEC_POST_COOLDOWN, DEC_BYPASS_MOVE,
})


# ---------------------------------------------------------------------------
# Flag catalog
# ---------------------------------------------------------------------------

# All L7 binary flags available from the feature pipeline.
# Grouped by quality tier so the scanner can apply different thresholds.
# Tier A: high-conviction structural events.
# Tier B: momentum / context flags.
# Tier C: supplementary, use only in combination.

FLAG_TIERS: dict[str, list[str]] = {
    "A": [
        "flag_volume_climax",      # volume > 95th pct -- unusual institutional activity
        "flag_sweep_up",           # price swept above swing high but closed below (bull trap)
        "flag_sweep_dn",           # price swept below swing low but reclaimed (bear trap)
        "flag_squeeze_release",    # Bollinger squeeze released -- volatility expansion
        "flag_macd_cross_up",      # MACD histogram crossed above zero (bullish momentum)
        "flag_macd_cross_dn",      # MACD histogram crossed below zero (bearish momentum)
        "flag_regime_flip",        # Supertrend changed direction (trend reversal)
    ],
    "B": [
        "flag_golden_cross",       # EMA-50 crossed above EMA-200
        "flag_death_cross",        # EMA-50 crossed below EMA-200
        "flag_adx_strong_trend",   # ADX >= 25 (strong directional trend)
    ],
    "C": [
        "flag_rsi_oversold",       # RSI <= 30
        "flag_rsi_overbought",     # RSI >= 70
    ],
}

# Default trigger set: any Tier A flag fires a deep analysis call.
# Configurable via cfg.trigger_flags.
DEFAULT_TRIGGER_FLAGS: tuple[str, ...] = tuple(FLAG_TIERS["A"])


# ---------------------------------------------------------------------------
# TriggerDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TriggerDecision:
    """Result of evaluating one bar against the trigger gate."""

    symbol:         str
    bar_ts:         pd.Timestamp | None
    decision:       str                       # one of the DEC_* constants
    flag:           str | None  = None        # which flag fired (if any)
    close:          float | None = None       # close price at the trigger bar
    atr_pct:        float | None = None       # ATR as % of price
    move_pct:       float | None = None       # price move since last trigger
    threshold_pct:  float | None = None       # bypass threshold applied
    bars_elapsed:   int | None   = None       # bars since last cooldown
    reason:         str          = ""         # human-readable explanation

    @property
    def fired(self) -> bool:
        return self.decision in POSITIVE_DECISIONS


# ---------------------------------------------------------------------------
# Pure detection function
# ---------------------------------------------------------------------------

def detect_trigger(
    *,
    symbol:        str,
    bar:           Mapping[str, Any] | pd.Series | None,
    state:         CooldownState,
    cfg:           Config,
    trigger_flags: tuple[str, ...] | None = None,
) -> TriggerDecision:
    """Evaluate one freshly-closed 15m bar against the trigger gate.

    Args:
        symbol:        Ticker symbol.
        bar:           Feature snapshot row as a dict or pd.Series.
                       Must contain: ``timestamp`` (or ``bar_ts``), ``close``,
                       ``atr_14_pct``, and all configured trigger flag columns.
        state:         Current cooldown state for this symbol.
        cfg:           Loaded Config.
        trigger_flags: Override the flags to check. Defaults to ``cfg.trigger_flags``
                       or ``DEFAULT_TRIGGER_FLAGS`` if cfg doesn't specify.

    Returns:
        A frozen ``TriggerDecision`` dataclass.
    """
    # --- Parse bar ---
    if bar is None:
        return _no_bar(symbol, "bar is None")

    if isinstance(bar, pd.Series):
        bar = bar.to_dict()

    bar_ts = _parse_ts(bar.get("timestamp") or bar.get("bar_ts"))
    if bar_ts is None:
        return _no_bar(symbol, "missing/invalid timestamp")

    close = _flt(bar.get("close"))
    if close is None or close <= 0:
        return _no_bar(symbol, "missing/invalid close")

    # --- Dedup: same bar already processed ---
    if state.last_bar_ts is not None and bar_ts == state.last_bar_ts:
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close,
            decision=DEC_DUP_BAR, reason="bar_ts already processed",
        )

    # --- Which configured flag fired? ---
    flags_to_check = trigger_flags or tuple(cfg.trigger_flags) or DEFAULT_TRIGGER_FLAGS
    fired_flag = _find_fired_flag(bar, flags_to_check)

    if fired_flag is None:
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close,
            decision=DEC_NO_FLAG, reason="no trigger flag active",
        )

    # --- ATR in % ---
    atr_pct = _flt(bar.get("atr_14_pct")) or 0.0

    # --- Fresh symbol: never triggered before ---
    if state.last_bar_ts is None or state.last_close is None:
        log.debug("trigger fresh symbol=%s flag=%s close=%.4f", symbol, fired_flag, close)
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
            decision=DEC_FRESH, flag=fired_flag,
            reason="first trigger for symbol",
        )

    bars_elapsed = max(0, state.bars_since)
    cooldown     = max(0, int(cfg.prompt_cooldown_candles))

    # --- Cooldown expired naturally ---
    if bars_elapsed >= cooldown:
        log.debug(
            "trigger post_cooldown symbol=%s flag=%s bars_elapsed=%d",
            symbol, fired_flag, bars_elapsed,
        )
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
            decision=DEC_POST_COOLDOWN, flag=fired_flag,
            bars_elapsed=bars_elapsed,
            reason=f"cooldown elapsed: {bars_elapsed} >= {cooldown} bars",
        )

    # --- Cooldown still active: check price-move bypass ---
    move_pct      = abs(close - state.last_close) / state.last_close
    threshold_pct = max(
        cfg.cooldown_bypass_atr_mult * (atr_pct / 100.0),
        cfg.cooldown_bypass_floor_pct,
    )

    if move_pct >= threshold_pct:
        log.debug(
            "trigger bypass symbol=%s flag=%s move=%.4f threshold=%.4f",
            symbol, fired_flag, move_pct, threshold_pct,
        )
        return TriggerDecision(
            symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
            decision=DEC_BYPASS_MOVE, flag=fired_flag,
            move_pct=move_pct, threshold_pct=threshold_pct,
            bars_elapsed=bars_elapsed,
            reason=(f"bypass: move {move_pct:.4f} >= threshold {threshold_pct:.4f}"),
        )

    # --- Still in cooldown, no bypass ---
    return TriggerDecision(
        symbol=symbol, bar_ts=bar_ts, close=close, atr_pct=atr_pct,
        decision=DEC_COOLDOWN, flag=fired_flag,
        move_pct=move_pct, threshold_pct=threshold_pct,
        bars_elapsed=bars_elapsed,
        reason=(
            f"cooldown active: {bars_elapsed}/{cooldown} bars, "
            f"move {move_pct:.4f} < threshold {threshold_pct:.4f}"
        ),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _no_bar(symbol: str, reason: str) -> TriggerDecision:
    return TriggerDecision(symbol=symbol, bar_ts=None, decision=DEC_NO_BAR, reason=reason)


def _parse_ts(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _flt(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
        return v if v == v else None  # NaN check
    except (TypeError, ValueError):
        return None


def _find_fired_flag(bar: Mapping[str, Any], flags: tuple[str, ...]) -> str | None:
    """Return the first flag name that is set to >= 1.0, or None."""
    for fname in flags:
        v = _flt(bar.get(fname))
        if v is not None and v >= 1.0:
            return fname
    return None
