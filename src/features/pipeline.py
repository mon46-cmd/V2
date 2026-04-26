"""Feature pipeline: assemble all layers into one wide snapshot.

Usage
-----
Simple (OHLCV only)::

    from features.pipeline import compute_snapshot
    from features.config   import FeatureConfig

    snapshot = compute_snapshot(ohlcv_df)

With derivatives data::

    snapshot = compute_snapshot(
        ohlcv_df,
        funding  = funding_df,
        oi       = oi_df,
        mark     = mark_df,
        index_p  = index_df,
        ls_ratio = ls_df,
    )

The returned DataFrame has one row per OHLCV bar and all feature columns
from every layer concatenated horizontally.  Columns from later layers
that depend on earlier layers are computed in the correct order.

Layers
------
L1  Volatility        -- returns, ATR, Parkinson, Garman-Klass, Yang-Zhang
L2  Price structure   -- EMAs, swing highs/lows, candle anatomy, percentile rank
L3  Regime            -- Hurst exponent, variance ratio, lag-1 autocorrelation
L4  Technicals        -- RSI, MACD, Bollinger, ADX, Stoch, CCI, MFI, VWAP, OBV, CMF, Supertrend
L5  Derivatives       -- funding, OI, basis, premium, L/S ratio (optional)
L6  Volume profile    -- POC, VAH, VAL, distances
L7  Flags / signals   -- 12 binary event flags computed from L1-L4 + OHLCV
L8  Advanced          -- Kyle lambda, Amihud, VRDS, realized skew/kurt, efficiency ratio,
                         RVOL, NREI, rolling Sharpe, CLV, up/down vol ratio,
                         vol regime score, TQI, Waddah Attar Explosion,
                         wick asymmetry, bull/bear streak
"""
from __future__ import annotations

import logging

import pandas as pd

from features.config import FeatureConfig
import features.layer1_volatility   as l1
import features.layer2_structure    as l2
import features.layer3_regime       as l3
import features.layer4_technicals   as l4
import features.layer5_derivatives  as l5
import features.layer6_volume_profile as l6
import features.layer7_flags        as l7
import features.layer8_advanced     as l8

log = logging.getLogger(__name__)


def compute_snapshot(
    ohlcv:    pd.DataFrame,
    cfg:      FeatureConfig | None = None,
    *,
    funding:  pd.DataFrame | None = None,
    oi:       pd.DataFrame | None = None,
    mark:     pd.DataFrame | None = None,
    index_p:  pd.DataFrame | None = None,
    premium:  pd.DataFrame | None = None,
    ls_ratio: pd.DataFrame | None = None,
    layers:   tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Compute all feature layers and return a single wide DataFrame.

    Args:
        ohlcv:    OHLCV DataFrame. Must have columns ``open, high, low, close``
                  and a UTC DatetimeIndex.  ``volume`` is optional but many
                  layer-4 features require it.
        cfg:      FeatureConfig.  Defaults to ``FeatureConfig()`` (sensible defaults).
        funding:  Funding-rate DataFrame (layer 5).
        oi:       Open-interest DataFrame (layer 5).
        mark:     Mark-price klines (layer 5, basis computation).
        index_p:  Index-price klines (layer 5, basis computation).
        premium:  Premium-index klines (layer 5).
        ls_ratio: Long/short ratio DataFrame (layer 5).
        layers:   Optional tuple of layer numbers to compute, e.g. ``(1, 2, 4)``.
                  If None all layers are computed.

    Returns:
        Wide DataFrame with all feature columns, same index as ``ohlcv``.
    """
    if cfg is None:
        cfg = FeatureConfig()

    want = set(layers) if layers is not None else {1, 2, 3, 4, 5, 6, 7, 8}
    frames: list[pd.DataFrame] = []

    def _run(n: int, fn, *args, **kwargs):
        if n not in want:
            return pd.DataFrame(index=ohlcv.index)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            log.warning("layer %d failed: %s", n, exc, exc_info=True)
            return pd.DataFrame(index=ohlcv.index)

    f1 = _run(1, l1.compute, ohlcv, cfg)
    f2 = _run(2, l2.compute, ohlcv, cfg)
    f3 = _run(3, l3.compute, _with_cols(ohlcv, f1), cfg)    # L3 can use log_ret from L1
    f4 = _run(4, l4.compute, ohlcv, cfg)
    f5 = _run(5, l5.compute, ohlcv, cfg,
               funding=funding, oi=oi, mark=mark,
               index_p=index_p, premium=premium, ls_ratio=ls_ratio)
    f6 = _run(6, l6.compute, ohlcv, cfg)

    # L7 flags need the wide frame containing all L1-L4 outputs.
    wide_for_flags = pd.concat([ohlcv, f1, f2, f4], axis=1)
    f7 = _run(7, l7.compute, wide_for_flags, cfg)

    # L8 advanced features reuse L1/L2/L4 to avoid recomputing ATR/wick/ADX.
    f8 = _run(8, l8.compute, ohlcv, cfg,
               l1_frame=f1 if not f1.empty else None,
               l2_frame=f2 if not f2.empty else None,
               l4_frame=f4 if not f4.empty else None)

    frames = [f1, f2, f3, f4, f5, f6, f7, f8]
    non_empty = [f for f in frames if not f.empty]
    if not non_empty:
        return pd.DataFrame(index=ohlcv.index)
    result = pd.concat(non_empty, axis=1)
    return result


def get_last_row(snapshot: pd.DataFrame) -> pd.Series:
    """Return the most recent (last closed bar) row as a Series.

    Useful for building the per-symbol context dict to pass to the AI.
    """
    if snapshot.empty:
        return pd.Series(dtype="float64")
    return snapshot.iloc[-1]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _with_cols(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """Return *base* with non-duplicate columns from *extra* appended."""
    new_cols = [c for c in extra.columns if c not in base.columns]
    if not new_cols:
        return base
    return pd.concat([base, extra[new_cols]], axis=1)
