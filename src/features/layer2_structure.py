"""Layer 2 -- price structure and trend features.

Computed from OHLCV candles only.  Pure, no I/O, no lookahead.

Output columns
--------------
ema_9 / ema_21 / ema_50 / ema_200   Exponential moving averages.
ema_9_dist ... ema_200_dist          (close - ema) / close  as a %.
ema_9_slope ... ema_200_slope        1-bar pct change in the EMA.
price_pct_rank_100                   Percentile rank of close in the last 100 bars (0-1).
swing_high_10 / swing_low_10         N-bar pivot high / low price levels.
close_vs_swing_high_pct              (close - swing_high) / close * 100.
close_vs_swing_low_pct              (close - swing_low)  / close * 100.
candle_body_pct                      Abs body / (high - low): measures indecision.
upper_wick_pct / lower_wick_pct     Wick size relative to bar range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return a frame with Layer-2 columns aligned to ``df.index``."""
    close = df["close"].astype("float64")
    high  = df["high"].astype("float64")
    low   = df["low"].astype("float64")
    openp = df["open"].astype("float64") if "open" in df.columns else close

    out: dict[str, pd.Series] = {}

    # EMAs, distance from close, slope.
    for w in cfg.ema_windows:
        ema = close.ewm(span=w, adjust=False, min_periods=w).mean()
        out[f"ema_{w}"]       = ema
        out[f"ema_{w}_dist"]  = (close - ema) / close.replace(0, np.nan) * 100.0
        out[f"ema_{w}_slope"] = ema.pct_change()

    # Price percentile rank over a look-back window.
    out["price_pct_rank_100"] = close.rolling(cfg.price_rank_window, min_periods=cfg.price_rank_window).rank(pct=True)

    # Swing highs / lows (N-bar pivot detection).
    n = cfg.swing_lookback
    prior_high = high.shift(1).rolling(n, min_periods=n).max()
    prior_low  = low.shift(1).rolling(n, min_periods=n).min()
    out["swing_high_10"]           = prior_high
    out["swing_low_10"]            = prior_low
    out["close_vs_swing_high_pct"] = (close - prior_high) / close.replace(0, np.nan) * 100.0
    out["close_vs_swing_low_pct"]  = (close - prior_low)  / close.replace(0, np.nan) * 100.0

    # Candle anatomy (indecision / wick analysis).
    bar_range = (high - low).replace(0, np.nan)
    body      = (close - openp).abs()
    out["candle_body_pct"]   = body      / bar_range
    out["upper_wick_pct"]    = (high - pd.concat([close, openp], axis=1).max(axis=1)) / bar_range
    out["lower_wick_pct"]    = (pd.concat([close, openp], axis=1).min(axis=1) - low)  / bar_range

    return pd.DataFrame(out, index=df.index)
