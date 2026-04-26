"""Context builder -- serialize feature snapshot rows into token-efficient prompt text.

The model gets a structured text block, NOT raw JSON.
Text blocks are ~40% more token-efficient than JSON for numeric data because
there are no quotes, colons-as-separators, or array brackets.

Two context types:
  ``build_deep_context``    -- full feature block for Prompt B (deep analysis)
  ``build_review_context``  -- lean position state block for Prompt C (review)

Both functions also inject ``past context``: a summary of recent price action
(last N bars) so the model understands the trajectory, not just the snapshot.

Token estimates:
  Deep context:   ~350-400 tokens
  Review context: ~200-250 tokens
  Past N bars:    ~80-120 tokens (5-bar OHLCV summary)

Usage::

    from ai.calls.context_builder import build_deep_context, build_review_context

    ctx = build_deep_context(
        row=snapshot_row,          # dict from pipeline snapshot last row
        ohlcv_df=recent_ohlcv,    # last 20 bars of OHLCV
        symbol="BTCUSDT",
        as_of="2026-04-26T12:00:00Z",
        social_reason="ETF inflows catalyst",
        watchlist_direction="long",
        htf_bias="bullish",       # from higher TF EMA check
    )
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Number formatting helpers
# ---------------------------------------------------------------------------

def _p(v: Any, decimals: int = 2) -> str:
    """Format a numeric value for prompt output. Treats -0 as 0."""
    if v is None:
        return "n/a"
    try:
        f = float(v)
        if not math.isfinite(f):
            return "n/a"
        if decimals == 0:
            return str(int(round(f)))
        result = f"{f:.{decimals}f}"
        # Suppress negative-zero presentation ("-0.00" -> "0.00")
        if result.lstrip("-") == "0" + ("." + "0" * decimals if decimals else ""):
            result = result.lstrip("-")
        return result
    except (TypeError, ValueError):
        return "n/a"

def _price(v: Any) -> str:
    """Format a price: round to int if > 10, else 4dp."""
    try:
        f = float(v)
        if not math.isfinite(f):
            return "n/a"
        return str(int(round(f))) if f > 10 else f"{f:.4f}"
    except (TypeError, ValueError):
        return "n/a"

def _sign(v: Any, decimals: int = 2) -> str:
    """Format a signed number explicitly (+/-). Suppresses -0."""
    try:
        f = float(v)
        if not math.isfinite(f):
            return "n/a"
        # Round first to avoid -0 from floating point
        r = round(f, decimals)
        if r == 0.0:
            return f"+0.{'0'*decimals}"
        return f"{r:+.{decimals}f}"
    except (TypeError, ValueError):
        return "n/a"

def _pct(v: Any) -> str:
    """Format a ratio as percentage: 0.73 -> '73.0%'."""
    try:
        f = float(v)
        return f"{f * 100:.1f}%" if math.isfinite(f) else "n/a"
    except (TypeError, ValueError):
        return "n/a"

def _dir(v: Any) -> str:
    """Convert +1/-1 supertrend direction to up/down text."""
    try:
        f = float(v)
        return "up" if f > 0 else "down" if f < 0 else "flat"
    except (TypeError, ValueError):
        return "n/a"

def _bps(v: Any) -> str:
    """Format a rate (e.g. 0.0001) as basis points with 2dp."""
    try:
        f = float(v)
        if not math.isfinite(f):
            return "n/a"
        bps = f * 10_000
        r = round(bps, 2)
        if r == 0.0:
            return "0.00bps"
        return f"{r:+.2f}bps"
    except (TypeError, ValueError):
        return "n/a"

def _obv_direction(v: Any) -> str:
    """OBV slope is in absolute units; just show direction + magnitude label."""
    try:
        f = float(v)
        if not math.isfinite(f) or f == 0:
            return "flat"
        sign = "+" if f > 0 else "-"
        # Classify magnitude relative to typical range.
        mag = abs(f)
        label = "weak" if mag < 20 else "moderate" if mag < 100 else "strong"
        return f"{sign}{label}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# Past context: recent OHLCV summary
# ---------------------------------------------------------------------------

def _build_past_bars(ohlcv_df: pd.DataFrame | None, n: int = 5) -> str:
    """Summarize the last N bars as compact OHLCV text.

    Gives the model a sense of recent trajectory:
    whether price is accelerating, decelerating, or reversing.

    Example output (each bar = one line):
        OHLCV last 5 bars (oldest first, 15m):
        bar-5: O=67100 H=67380 L=66950 C=67200 V=1842 (green)
        bar-4: O=67200 H=67550 L=67180 C=67490 V=2104 (green)
        ...
    """
    if ohlcv_df is None or ohlcv_df.empty:
        return ""

    tail = ohlcv_df.tail(n).copy()
    if tail.empty:
        return ""

    lines = [f"RECENT {min(n, len(tail))} BARS (oldest->newest, 15m):"]
    for i, (_, row) in enumerate(tail.iterrows()):
        o = _price(row.get("open",  row.get("close")))
        h = _price(row.get("high",  row.get("close")))
        l = _price(row.get("low",   row.get("close")))
        c = _price(row.get("close"))
        v = row.get("volume", 0)
        v_str  = f"{float(v):.0f}" if v else "n/a"
        candle = "green" if float(row.get("close", 0)) >= float(row.get("open", 0)) else "red"
        offset = -(min(n, len(tail)) - 1 - i)
        lines.append(f"  bar{offset:+d}: O={o} H={h} L={l} C={c} V={v_str} ({candle})")
    return "\n".join(lines)


def _build_recent_flags(row: dict[str, Any], lookback_rows: list[dict] | None) -> str:
    """Show which flags were active in the last N bars (recent history)."""
    if not lookback_rows:
        return ""
    all_flags = [k for k in row if k.startswith("flag_")]
    if not all_flags:
        return ""

    # Find flags that fired in recent history (any of last N bars).
    recently_fired: dict[str, int] = {}  # flag -> bars ago
    for ago, hist_row in enumerate(reversed(lookback_rows[-8:]), start=1):
        for f in all_flags:
            v = hist_row.get(f)
            try:
                if float(v) >= 1.0 and f not in recently_fired:
                    recently_fired[f] = ago
            except (TypeError, ValueError):
                pass

    if not recently_fired:
        return ""

    lines = ["RECENT FLAG HISTORY (bars ago):"]
    for f, ago in sorted(recently_fired.items(), key=lambda x: x[1]):
        lines.append(f"  {f}: {ago}b ago")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deep context builder (Prompt B)
# ---------------------------------------------------------------------------

def build_deep_context(
    *,
    row:                  dict[str, Any],
    symbol:               str,
    as_of:                str,
    close_price:          float | None      = None,   # from OHLCV, not feature row
    social_reason:        str               = "",
    watchlist_direction:  str               = "",
    htf_bias:             str               = "unknown",
    ohlcv_df:             pd.DataFrame | None = None,
    lookback_rows:        list[dict]        | None = None,
    fired_flag:           str               = "",
) -> str:
    """Assemble the full context block for the deep analysis prompt (Prompt B).

    Args:
        row:                  Last snapshot row as dict (from feature pipeline).
        symbol:               Ticker symbol.
        as_of:                ISO UTC timestamp string.
        close_price:          Last close from OHLCV frame (feature snapshot may not carry it).
        social_reason:        Grok's reason for picking this symbol.
        watchlist_direction:  "long" or "short" from social scan.
        htf_bias:             Higher-timeframe bias string.
        ohlcv_df:             Recent OHLCV bars for past context.
        lookback_rows:        Last N snapshot rows for flag history.
        fired_flag:           Which trigger flag fired on this bar.
    """
    g  = row.get  # shorthand

    # Active flags only.
    active_flags = [
        k for k in row
        if k.startswith("flag_")
        and _try_float(g(k)) >= 1.0
    ]

    past_bars   = _build_past_bars(ohlcv_df, n=6)
    flag_hist   = _build_recent_flags(row, lookback_rows)

    sections: list[str] = []

    # Header.
    sections.append(
        f"Symbol: {symbol}  |  TF: 15m  |  HTF bias: {htf_bias}  |  UTC: {as_of}"
    )
    if social_reason:
        sections.append(f"Social context ({watchlist_direction.upper()}): {social_reason}")
    if fired_flag:
        sections.append(f"Trigger flag: {fired_flag}")

    # Past context.
    if past_bars:
        sections.append(past_bars)
    if flag_hist:
        sections.append(flag_hist)

    # Resolve close: prefer explicitly passed value, fall back to row, fall back to OHLCV.
    close_val = close_price
    if close_val is None or not math.isfinite(close_val if close_val else float('nan')):
        close_val = _try_float(g('close'))
    if not math.isfinite(close_val):
        # Last resort: read directly from ohlcv_df.
        if ohlcv_df is not None and not ohlcv_df.empty and 'close' in ohlcv_df.columns:
            close_val = float(ohlcv_df['close'].iloc[-1])

    atr_price_val = _atr_price(close_val, g('atr_14_pct'))
    funding_bps   = _try_float(g('funding_rate_z'))   # z-score from L5 (may be nan)
    funding_raw   = _try_float(g('funding_rate'))     # raw rate if available

    # Price & structure.
    sections.append("\n".join([
        "PRICE & STRUCTURE:",
        f"  close:              {_price(close_val)}",
        f"  atr_14:             {_price(atr_price_val)}  ({_p(g('atr_14_pct'))}% of price)",
        f"  ema_9:              {_price(g('ema_9'))}",
        f"  ema_21:             {_price(g('ema_21'))}",
        f"  ema_50:             {_price(g('ema_50'))}",
        f"  ema_200:            {_price(g('ema_200'))}",
        f"  supertrend:         {_dir(g('supertrend_dir'))}  (band: {_price(g('supertrend'))})",
        # BB width: show 3dp to avoid 0.00% for tight markets
        f"  bb_%b:              {_p(g('bb_pct_b_20'))}  (>1=above upper, <0=below lower | width={_p(g('bb_width_20'), 3)}%)",
        f"  price_rank_100b:    {_pct(g('price_pct_rank_100'))}  (0%=100b low, 100%=100b high)",
        f"  swing_high_10:      {_price(g('swing_high_10'))}  swing_low_10: {_price(g('swing_low_10'))}",
        f"  vwap_dist:          {_p(g('vwap_20_dist'))}%  (price vs 20-bar VWAP)",
        f"  donchian_pct:       {_p(g('donchian_pct_20'))}  (0=at 20b low, 1=at 20b high)",
    ]))

    # Momentum -- with improved formatting.
    sections.append("\n".join([
        "MOMENTUM:",
        f"  rsi_14:             {_p(g('rsi_14'))}  (oversold<30, overbought>70)",
        f"  macd_hist:          {_sign(g('macd_hist'))}  (macd={_sign(g('macd'))} sig={_sign(g('macd_signal'))})",
        f"  adx_14:             {_p(g('adx_14'))}  +DI={_p(g('plus_di_14'))} -DI={_p(g('minus_di_14'))}",
        f"  stoch_k:            {_p(g('stoch_k_14'))}  d={_p(g('stoch_d_3'))}",
        f"  williams_r:         {_p(g('williams_r_14'))}  (0=overbought, -100=oversold)",
        f"  cci_20:             {_p(g('cci_20'))}",
        f"  cmf_20:             {_p(g('cmf_20'))}  (>0=buying pressure, <0=selling)",
        f"  obv_trend:          {_obv_direction(g('obv_slope_20'))}  (OBV slope direction/magnitude)",
    ]))

    # Derivatives (L5) -- only show if available.
    deriv_lines = ["DERIVATIVES:"]
    for key, label in [
        ("funding_rate",      "funding_rate"),
        ("funding_rate_z",    "funding_rate_z (z-score vs 30d)"),
        ("open_interest_z",   "oi_change_z    (z-score)"),
        ("basis_bps",         "basis_bps      (mark-index spread)"),
        ("ls_ratio",          "ls_ratio       (longs/shorts)"),
    ]:
        v = g(key)
        if v is not None and math.isfinite(_try_float(v)):
            # Show funding rate as bps for clarity.
            val = _bps(v) if key == "funding_rate" else _p(v)
            deriv_lines.append(f"  {label}: {val}")
    if len(deriv_lines) > 1:
        sections.append("\n".join(deriv_lines))

    # Microstructure (L8).
    micro_lines = ["MICROSTRUCTURE:"]
    for key, label, fmt in [
        ("clv",                "clv            [-1=bear close, +1=bull close]", "p2"),
        ("efficiency_ratio_20","eff_ratio      [0=choppy, 1=trending]",         "p2"),
        ("rvol_20",            "rvol           [1=avg, >1.5=unusual]",          "p2"),
        ("nrei",               "nrei           [1=avg range, >2=expansion]",    "p2"),
        ("ud_vol_ratio_20",    "ud_vol_ratio   [>1=buy dom, <1=sell dom]",      "p2"),
        ("tqi",                "tqi            [signed ADX: +up/-down]",        "p1"),
        ("vol_regime_score",   "vol_regime     [0=calm, 1=historically extreme]","p2"),
        ("sharpe_20",          "sharpe_20b     [>0.5=momentum quality, range ~-3 to +3]","p2"),
        ("ret_skew_20",        "ret_skew_20b   [<0=left tail risk]",            "p2"),
        ("bull_streak",        "bull_streak    [consecutive green closes]",     "p0"),
        ("bear_streak",        "bear_streak    [consecutive red closes]",       "p0"),
        ("wick_asymmetry_5",   "wick_asym_5b   [+ve=upper wick dom (sellers)]","p2"),
        ("vrds_20",            "vrds_20        [bull flow - bear flow]",        "p3"),
        # Amihud omitted: raw scale is arbitrary cross-symbol
    ]:
        v = g(key)
        if v is not None and math.isfinite(_try_float(v)):
            dp = int(fmt[1])
            micro_lines.append(f"  {label}: {_p(v, dp)}")
    if len(micro_lines) > 1:
        sections.append("\n".join(micro_lines))

    # Partial-bar warning: if last bar has very low RVOL it means it's still open.
    # Warn Grok so it doesn't over-interpret zeroed-out bar metrics.
    try:
        rvol_last = _try_float(g("rvol_20"))
        if math.isfinite(rvol_last) and rvol_last < 0.1:
            sections.append(
                "NOTE: Current bar is partially open (very low volume). "
                "Efficiency ratio, RVOL, and NREI for bar+0 may be near-zero -- "
                "weight the previous bars more heavily for microstructure signals."
            )
    except Exception:  # noqa: BLE001
        pass

    # Active flags.
    if active_flags:
        sections.append("ACTIVE FLAGS: " + ", ".join(active_flags))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Review context builder (Prompt C)
# ---------------------------------------------------------------------------

def build_review_context(
    *,
    symbol:           str,
    side:             str,
    entry:            float,
    stop_loss:        float,
    tp1:              float,
    tp2:              float,
    mark:             float,
    remaining_pct:    float,      # 100=full, 50=TP1 hit
    bars_held:        int,
    loser_streak:     int,
    hook_reason:      str,
    row:              dict[str, Any] | None = None,   # latest snapshot row
    ohlcv_df:         pd.DataFrame | None   = None,   # recent bars for context
) -> str:
    """Assemble the enriched context block for the review prompt (Prompt C).

    Provides:
    - Full position state with P&L, distances, and ATR multiples.
    - Trade progress (how far to TP2, stage awareness).
    - Key structural levels (EMA 21/50, swing H/L, VWAP).
    - RSI trend (current vs 3 bars ago) for momentum trajectory.
    - Funding rate context for cost and squeeze risk.
    - 6 recent bars for price action trajectory.
    """
    g = (row or {}).get

    safe_entry = entry if entry != 0 else 1e-9   # guard division by zero

    unrealized_pct = ((mark - safe_entry) / safe_entry * 100) if side == "long" \
        else ((safe_entry - mark) / safe_entry * 100)
    pnl_str = f"{unrealized_pct:+.2f}%"

    sl_dist_pct  = abs(mark - stop_loss) / safe_entry * 100
    tp1_dist_pct = abs(tp1  - mark)      / safe_entry * 100
    tp2_dist_pct = abs(tp2  - mark)      / safe_entry * 100

    # How far have we moved against the position?
    if side == "long":
        adverse_pct = max(0.0, (safe_entry - mark) / safe_entry * 100)
    else:
        adverse_pct = max(0.0, (mark - safe_entry) / safe_entry * 100)

    # ATR in price units for distance calculations.
    atr_price = _atr_price(mark, g("atr_14_pct"))

    # RSI trajectory: compare current vs ohlcv_df bar 3 ago (if available).
    rsi_current = _try_float(g("rsi_14"))
    rsi_3b_ago  = float("nan")
    if ohlcv_df is not None and not ohlcv_df.empty and "rsi_14" in ohlcv_df.columns:
        tail3 = ohlcv_df.dropna(subset=["rsi_14"])
        if len(tail3) >= 4:
            rsi_3b_ago = float(tail3.iloc[-4]["rsi_14"])

    past_bars = _build_past_bars(ohlcv_df, n=6)

    sections: list[str] = []

    sections.append(f"REVIEW TRIGGER: {hook_reason}")

    # --- Position state ---
    pos_lines = [
        "POSITION STATE:",
        f"  symbol:          {symbol}",
        f"  side:            {side}",
        f"  entry:           {_price(entry)}",
        f"  mark_price:      {_price(mark)}",
        f"  unrealized_pnl:  {pnl_str}",
        f"  remaining_size:  {remaining_pct:.0f}%  (100=full, 50=TP1 hit, runner)",
        f"  stop_loss:       {_price(stop_loss)}   ({sl_dist_pct:.2f}% from mark)",
        f"  tp1:             {_price(tp1)}   ({tp1_dist_pct:.2f}% to TP1)",
        f"  tp2:             {_price(tp2)}   ({tp2_dist_pct:.2f}% to TP2)",
        f"  bars_held:       {bars_held}",
        f"  adverse_move:    {adverse_pct:.2f}%  (vs entry)",
        f"  loser_streak:    {loser_streak}",
    ]
    # ATR multiples for SL and TP distances.
    if atr_price and math.isfinite(atr_price) and atr_price > 0:
        sl_atr  = abs(mark - stop_loss) / atr_price
        tp1_atr = abs(tp1 - mark)       / atr_price
        tp2_atr = abs(tp2 - mark)       / atr_price
        pos_lines.append(
            f"  distances_in_ATR: sl={sl_atr:.1f}x  tp1={tp1_atr:.1f}x  tp2={tp2_atr:.1f}x"
        )
    sections.append("\n".join(pos_lines))

    # --- Key structural signals ---
    if row is not None:
        signal_lines = ["CURRENT SIGNALS:"]
        for key, label in [
            ("supertrend_dir",    "supertrend"),
            ("rsi_14",            "rsi_14"),
            ("clv",               "clv"),
            ("adx_14",            "adx_14"),
            ("efficiency_ratio_20", "efficiency"),
            ("vol_regime_score",  "vol_regime"),
            ("wick_asymmetry_5",  "wick_asym_5b"),
            ("atr_14_pct",        "atr_pct"),
        ]:
            v = g(key)
            if v is not None:
                formatted = _dir(v) if key == "supertrend_dir" else _p(v)
                signal_lines.append(f"  {label}: {formatted}")

        # RSI trend.
        if math.isfinite(rsi_current) and math.isfinite(rsi_3b_ago):
            delta = rsi_current - rsi_3b_ago
            direction = "rising" if delta > 2 else "falling" if delta < -2 else "flat"
            signal_lines.append(
                f"  rsi_trend:       {direction} ({rsi_3b_ago:.1f} -> {rsi_current:.1f}, delta={delta:+.1f})"
            )

        # Funding rate.
        fr = g("funding_rate")
        if fr is not None and math.isfinite(_try_float(fr)):
            signal_lines.append(f"  funding_rate:    {_bps(fr)}  (>+5bps=longs pay, <-2bps=shorts pay)")

        # Partial-bar warning.
        try:
            rvol_last = _try_float(g("rvol_20"))
            if math.isfinite(rvol_last) and rvol_last < 0.1:
                signal_lines.append(
                    "  NOTE: bar+0 is partially open -- efficiency/rvol may be near-zero."
                )
        except Exception:  # noqa: BLE001
            pass

        if len(signal_lines) > 1:
            sections.append("\n".join(signal_lines))

    # --- Key structural levels ---
    if row is not None:
        level_lines = ["KEY LEVELS:"]
        for key, label in [
            ("ema_21",        "ema_21       (dynamic support/resistance)"),
            ("ema_50",        "ema_50       (medium structure)"),
            ("supertrend",    "supertrend_band"),
            ("swing_high_10", "swing_high_10"),
            ("swing_low_10",  "swing_low_10"),
            ("vwap_20",       "vwap_20      (intraday fair value)"),
        ]:
            v = g(key)
            if v is not None and math.isfinite(_try_float(v)):
                level_lines.append(f"  {label}: {_price(v)}")
        if len(level_lines) > 1:
            sections.append("\n".join(level_lines))

    if past_bars:
        sections.append(past_bars)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")

def _atr_price(close: Any, atr_pct: Any) -> float | None:
    try:
        c = float(close)
        a = float(atr_pct)
        return c * a / 100.0
    except (TypeError, ValueError):
        return None
