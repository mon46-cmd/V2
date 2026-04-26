"""Layer 1 -- volatility features.

Computed from OHLCV candles only.  Pure, no I/O, no lookahead.

Output columns
--------------
ret              Simple 1-bar return (close/prev_close - 1).
log_ret          Log return (ln(close/prev_close)).
rv_20            Realized volatility: rolling std of log returns.
atr_14           Wilder ATR (absolute, in price units).
atr_14_pct       ATR as % of close price (normalised for cross-symbol comparison).
parkinson_20     Parkinson high-low estimator of intrabar volatility.
garman_klass_20  Garman-Klass OHLC estimator (more efficient than Parkinson).
yang_zhang_20    Yang-Zhang estimator (accounts for overnight gap + intrabar).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return a frame with Layer-1 columns aligned to ``df.index``.

    Args:
        df:  OHLCV DataFrame with columns ``open, high, low, close`` (volume optional).
             Index must be a UTC-tz-aware DatetimeIndex.
        cfg: FeatureConfig instance.
    """
    close  = df["close"].astype("float64")
    high   = df["high"].astype("float64")
    low    = df["low"].astype("float64")
    openp  = df["open"].astype("float64") if "open" in df.columns else close

    prev_close = close.shift(1)
    ret     = close / prev_close - 1.0
    log_ret = np.log(close / prev_close)

    # Realized volatility (std of log-returns).
    rv = log_ret.rolling(cfg.window_rv, min_periods=cfg.window_rv).std(ddof=1)

    # Wilder ATR.
    tr  = _true_range(high, low, prev_close)
    atr = _wilder(tr, cfg.window_atr)

    # Parkinson HL estimator.
    hl_log      = np.log(high / low.replace(0, np.nan))
    park_var    = (hl_log ** 2).rolling(cfg.window_parkinson, min_periods=cfg.window_parkinson).mean()
    parkinson   = np.sqrt(park_var / (4.0 * math.log(2.0)))

    # Garman-Klass estimator.
    gk_term = 0.5 * hl_log ** 2 - (2.0 * math.log(2.0) - 1.0) * np.log(close / openp) ** 2
    gk_var  = gk_term.rolling(cfg.window_garman_klass, min_periods=cfg.window_garman_klass).mean()
    garman_klass = np.sqrt(gk_var.clip(lower=0.0))

    # Yang-Zhang estimator.
    yz = _yang_zhang(openp, high, low, close, cfg.window_yang_zhang)

    return pd.DataFrame({
        "ret":             ret,
        "log_ret":         log_ret,
        "rv_20":           rv,
        "atr_14":          atr,
        "atr_14_pct":      atr / close.replace(0, np.nan) * 100.0,
        "parkinson_20":    parkinson,
        "garman_klass_20": garman_klass,
        "yang_zhang_20":   yz,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)


def _wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder smoothing: EMA with alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _yang_zhang(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series, n: int) -> pd.Series:
    """Yang-Zhang (2000) OHLC volatility estimator.

    sigma_yz^2 = sigma_overnight^2 + k * sigma_open_close^2 + (1-k) * sigma_rs^2
    with k = 0.34 / (1.34 + (n+1)/(n-1)).
    """
    if n < 2:
        return pd.Series(np.nan, index=c.index, dtype="float64")
    prev_c = c.shift(1)
    overnight    = np.log(o / prev_c)
    open_close   = np.log(c / o)
    # Rogers-Satchell intrabar term.
    rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

    sigma_o  = overnight.rolling(n, min_periods=n).var(ddof=1)
    sigma_c  = open_close.rolling(n, min_periods=n).var(ddof=1)
    sigma_rs = rs.rolling(n, min_periods=n).mean()

    k   = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    var = sigma_o + k * sigma_c + (1.0 - k) * sigma_rs
    return np.sqrt(var.clip(lower=0.0))
