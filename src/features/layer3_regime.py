"""Layer 3 -- market regime features.

Computed from a series of log-returns (or close prices).
Pure Python -- no Rust dependency.  Pure, no I/O, no lookahead.

Output columns
--------------
hurst_100   Rolling Hurst exponent (R/S method) over 100 bars.
            ~0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting.
vr_2_100    Lo-MacKinlay variance ratio (q=2) over 100 bars.
            ~1.0 = random walk, >1 = positive autocorrelation (momentum).
acf1_50     Lag-1 autocorrelation of log-returns over 50 bars.
            Positive = momentum, negative = mean-reverting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return a frame with Layer-3 columns aligned to ``df.index``."""
    if "log_ret" in df.columns:
        ret = df["log_ret"].to_numpy(dtype="float64")
    elif "ret" in df.columns:
        ret = df["ret"].to_numpy(dtype="float64")
    else:
        close = df["close"].to_numpy(dtype="float64")
        ret   = np.concatenate([[np.nan], np.diff(np.log(np.where(close > 0, close, np.nan)))])

    out = pd.DataFrame(index=df.index)
    out["hurst_100"] = _rolling_hurst(ret, cfg.hurst_window)
    out["vr_2_100"]  = _rolling_vr(ret,   cfg.hurst_window, q=2)
    out["acf1_50"]   = _rolling_acf1(ret,  cfg.acf_window)
    return out


# ---------------------------------------------------------------------------
# Rolling kernels (pure Python -- fast enough for typical 1k-5k bar windows)
# ---------------------------------------------------------------------------

def _rolling_hurst(ret: np.ndarray, window: int) -> np.ndarray:
    n   = ret.size
    out = np.full(n, np.nan)
    if window < 16 or window > n:
        return out
    for i in range(window - 1, n):
        out[i] = _hurst_rs(ret[i - window + 1: i + 1])
    return out


def _rolling_vr(ret: np.ndarray, window: int, q: int) -> np.ndarray:
    n   = ret.size
    out = np.full(n, np.nan)
    if window < q * 2 or window > n:
        return out
    for i in range(window - 1, n):
        out[i] = _variance_ratio(ret[i - window + 1: i + 1], q)
    return out


def _rolling_acf1(x: np.ndarray, window: int) -> np.ndarray:
    n   = x.size
    out = np.full(n, np.nan)
    if window < 4 or window > n:
        return out
    for i in range(window - 1, n):
        s  = x[i - window + 1: i + 1]
        m  = s.mean()
        d  = s - m
        c0 = float((d * d).mean())
        if c0 <= 0:
            continue
        out[i] = float((d[1:] * d[:-1]).sum() / window / c0)
    return out


def _hurst_rs(returns: np.ndarray) -> float:
    """R/S Hurst exponent for a single window."""
    n = returns.size
    if n < 16 or not np.all(np.isfinite(returns)):
        return float("nan")
    # Sub-period sizes.
    sizes: list[int] = []
    k = 8
    while k <= n:
        sizes.append(k)
        k = int(np.ceil(k * 1.7))
    if not sizes or sizes[-1] != n:
        sizes.append(n)

    log_n, log_rs = [], []
    for m in sizes:
        rs = _mean_rs(returns, m)
        if np.isfinite(rs) and rs > 0:
            log_n.append(np.log(m))
            log_rs.append(np.log(rs))
    if len(log_n) < 3:
        return float("nan")
    a, b  = np.asarray(log_n), np.asarray(log_rs)
    am, bm = a.mean(), b.mean()
    den = ((a - am) ** 2).sum()
    return float("nan") if den <= 0 else float(((a - am) * (b - bm)).sum() / den)


def _mean_rs(returns: np.ndarray, m: int) -> float:
    n      = returns.size
    chunks = n // m
    if m < 4 or chunks == 0:
        return float("nan")
    acc = []
    for c in range(chunks):
        s   = returns[c * m: c * m + m]
        d   = s - s.mean()
        cum = np.cumsum(d)
        rng = float(cum.max() - cum.min())
        std = float(np.sqrt((d * d).mean()))
        if std > 0 and np.isfinite(rng):
            acc.append(rng / std)
    return float(np.mean(acc)) if acc else float("nan")


def _variance_ratio(returns: np.ndarray, q: int) -> float:
    n = returns.size
    if q < 2 or n < q * 2:
        return float("nan")
    var1 = float(returns.var(ddof=1))
    if var1 <= 0:
        return float("nan")
    sums  = np.convolve(returns, np.ones(q), mode="valid")
    var_q = float(sums.var(ddof=1))
    return var_q / (q * var1)
