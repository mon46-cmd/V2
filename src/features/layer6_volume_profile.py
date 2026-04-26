"""Layer 6 -- bar-approximated volume profile.

For each bar, look at the trailing ``cfg.window_vp`` bars, bin the
price range into ``cfg.vp_num_bins`` cells, spread each bar's volume
uniformly across the cells it touches, then compute:

Output columns
--------------
poc_price       Point of control: price level with max traded volume.
vah_price       Value area high: upper edge of the 70% volume zone.
val_price       Value area low:  lower edge of the 70% volume zone.
poc_dist_pct    (close - POC) / close * 100.  Positive = above POC.
vah_dist_pct    (close - VAH) / close * 100.
val_dist_pct    (close - VAL) / close * 100.
value_area_width (VAH - VAL) / POC: normalised width of the value area.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return a frame with Layer-6 columns aligned to ``df.index``."""
    n    = len(df)
    win  = cfg.window_vp
    bins = cfg.vp_num_bins
    area = cfg.vp_value_area

    high  = df["high"].to_numpy(dtype="float64")
    low   = df["low"].to_numpy(dtype="float64")
    vol   = df["volume"].to_numpy(dtype="float64") if "volume" in df.columns else np.zeros(n)
    close = df["close"].astype("float64")

    poc_arr, vah_arr, val_arr = _rolling_vp(low, high, vol, win, bins, area)

    poc = pd.Series(poc_arr, index=df.index, dtype="float64")
    vah = pd.Series(vah_arr, index=df.index, dtype="float64")
    val = pd.Series(val_arr, index=df.index, dtype="float64")

    c0  = close.replace(0, np.nan)
    poc0 = poc.replace(0, np.nan)

    return pd.DataFrame({
        "poc_price":        poc,
        "vah_price":        vah,
        "val_price":        val,
        "poc_dist_pct":     (close - poc) / c0 * 100.0,
        "vah_dist_pct":     (close - vah) / c0 * 100.0,
        "val_dist_pct":     (close - val) / c0 * 100.0,
        "value_area_width": (vah - val) / poc0,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Pure-Python rolling volume profile
# ---------------------------------------------------------------------------

def _rolling_vp(
    low: np.ndarray, high: np.ndarray, vol: np.ndarray,
    win: int, bins: int, area: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n   = len(low)
    poc = np.full(n, np.nan)
    vah = np.full(n, np.nan)
    val = np.full(n, np.nan)

    for i in range(win - 1, n):
        lo_w  = low[i - win + 1: i + 1]
        hi_w  = high[i - win + 1: i + 1]
        vol_w = vol[i - win + 1: i + 1]

        p_lo = float(np.nanmin(lo_w))
        p_hi = float(np.nanmax(hi_w))
        if not np.isfinite(p_lo) or not np.isfinite(p_hi) or p_hi <= p_lo:
            continue

        edges   = np.linspace(p_lo, p_hi, bins + 1)
        bin_mid = 0.5 * (edges[:-1] + edges[1:])
        hist    = _build_hist(lo_w, hi_w, vol_w, edges)
        if hist.sum() <= 0:
            continue

        poc_idx     = int(np.argmax(hist))
        poc[i]      = float(bin_mid[poc_idx])
        vah_i, val_i = _value_area(hist, poc_idx, area)
        vah[i]      = float(bin_mid[vah_i])
        val[i]      = float(bin_mid[val_i])

    return poc, vah, val


def _build_hist(
    low: np.ndarray, high: np.ndarray, vol: np.ndarray, edges: np.ndarray,
) -> np.ndarray:
    """Spread each bar's volume uniformly across the price bins it overlaps."""
    nbins = len(edges) - 1
    hist  = np.zeros(nbins, dtype="float64")
    for l, h, v in zip(low, high, vol):
        if not np.isfinite(l) or not np.isfinite(h) or v <= 0 or h < l:
            continue
        lo_idx = int(np.clip(np.searchsorted(edges, l, side="right") - 1, 0, nbins - 1))
        hi_idx = int(np.clip(np.searchsorted(edges, h, side="left"),      0, nbins - 1))
        hi_idx = max(hi_idx, lo_idx)
        bar_span = max(h - l, edges[1] - edges[0])
        for b in range(lo_idx, hi_idx + 1):
            overlap = max(0.0, min(edges[b + 1], h) - max(edges[b], l))
            hist[b] += v * (overlap / bar_span)
    return hist


def _value_area(hist: np.ndarray, poc_idx: int, area: float) -> tuple[int, int]:
    """Expand outward from POC until ``area`` fraction of volume is covered."""
    total  = hist.sum()
    target = total * area
    lo = hi = poc_idx
    accum   = hist[poc_idx]
    nbins   = len(hist)

    while accum < target and (lo > 0 or hi < nbins - 1):
        up = hist[hi + 1] if hi < nbins - 1 else -1.0
        dn = hist[lo - 1] if lo > 0        else -1.0
        if up < 0 and dn < 0:
            break
        if up >= dn:
            hi    += 1
            accum += hist[hi]
        else:
            lo    -= 1
            accum += hist[lo]
    return hi, lo
