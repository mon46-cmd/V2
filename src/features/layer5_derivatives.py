"""Layer 5 -- derivatives-market features.

Requires data from separate Bybit endpoints (funding, OI, mark/index klines,
long/short ratio). All inputs are passed as optional DataFrames.  If a frame
is None or empty its output columns are filled with NaN -- the pipeline
never crashes on partial data bundles.

Output columns
--------------
Funding:
  funding_rate        Latest settlement rate merged asof.
  funding_chg         1-period change in funding rate.
  funding_z_20        Z-score vs. 20-period rolling mean.
  funding_annualized  fr * 3 * 365 (three 8h settlements per day).

Open interest:
  oi                  Contracts (or USD value) from OI endpoint.
  oi_chg              Absolute 1-bar change.
  oi_chg_pct          % change.
  oi_z_50             Z-score vs. 50-bar rolling mean.
  oi_price_divergence 1.0 when price and OI move in opposite directions.

Basis / mark-index:
  basis_bps           (mark - index) / index * 10000.
  basis_z_50          Z-score of basis.

Premium index:
  premium_index       Raw premium-index close (asof merged).

Long/short ratio:
  ls_buy_ratio        Aggregate long ratio (0-1).
  ls_sell_ratio       Aggregate short ratio (0-1).
  ls_buy_ratio_z_20   Z-score of long ratio.
  ls_sell_ratio_z_20  Z-score of short ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute(
    base:     pd.DataFrame,
    cfg:      FeatureConfig,
    *,
    funding:  pd.DataFrame | None = None,
    oi:       pd.DataFrame | None = None,
    mark:     pd.DataFrame | None = None,
    index_p:  pd.DataFrame | None = None,
    premium:  pd.DataFrame | None = None,
    ls_ratio: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a frame with Layer-5 columns aligned to ``base.index``.

    All auxiliary DataFrames must have a UTC-tz-aware DatetimeIndex and a
    ``timestamp`` or index representing time.  Columns are merged backward
    asof (at each bar in *base* we use the latest auxiliary value <= that bar).

    Args:
        base:     The primary OHLCV DataFrame (provides the time grid).
        cfg:      FeatureConfig.
        funding:  DataFrame with ``funding_rate`` column.
        oi:       DataFrame with ``open_interest`` column.
        mark:     DataFrame with ``close`` column (mark-price klines).
        index_p:  DataFrame with ``close`` column (index-price klines).
        premium:  DataFrame with ``close`` column (premium-index klines).
        ls_ratio: DataFrame with ``buy_ratio`` and ``sell_ratio`` columns.
    """
    out = pd.DataFrame(index=base.index)

    # ------------------------------------------------------------------ Funding
    if funding is not None and not funding.empty and "funding_rate" in funding.columns:
        fr = _asof_col(base, funding, "funding_rate")
        out["funding_rate"]       = fr
        out["funding_chg"]        = fr.diff()
        out["funding_z_20"]       = _zscore(fr, cfg.window_funding_z)
        out["funding_annualized"] = fr * 3.0 * 365.0

    # ------------------------------------------------------------------ OI
    if oi is not None and not oi.empty and "open_interest" in oi.columns:
        oi_s = _asof_col(base, oi, "open_interest")
        out["oi"]         = oi_s
        out["oi_chg"]     = oi_s.diff()
        out["oi_chg_pct"] = oi_s.pct_change()
        out["oi_z_50"]    = _zscore(oi_s, cfg.window_oi_z)
        if "close" in base.columns:
            price_sign = np.sign(base["close"].diff().fillna(0.0))
            oi_sign    = np.sign(out["oi_chg"].fillna(0.0))
            out["oi_price_divergence"] = (price_sign != oi_sign).astype("float64")

    # ------------------------------------------------------------------ Basis
    if (
        mark    is not None and not mark.empty    and "close" in mark.columns
        and index_p is not None and not index_p.empty and "close" in index_p.columns
    ):
        mk_c  = _asof_col(base, mark,    "close")
        ix_c  = _asof_col(base, index_p, "close")
        basis = (mk_c - ix_c) / ix_c.replace(0, np.nan) * 1e4
        out["basis_bps"]  = basis
        out["basis_z_50"] = _zscore(basis, cfg.window_basis_z)

    # ------------------------------------------------------------------ Premium index
    if premium is not None and not premium.empty and "close" in premium.columns:
        out["premium_index"] = _asof_col(base, premium, "close")

    # ------------------------------------------------------------------ Long/Short ratio
    if ls_ratio is not None and not ls_ratio.empty:
        for col in ("buy_ratio", "sell_ratio"):
            if col in ls_ratio.columns:
                s                      = _asof_col(base, ls_ratio, col)
                out[f"ls_{col}"]       = s
                out[f"ls_{col}_z_20"]  = _zscore(s, cfg.window_ls_z)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _asof_col(base: pd.DataFrame, aux: pd.DataFrame, col: str) -> pd.Series:
    """Backward asof-merge a single column from *aux* onto *base*'s index.

    Returns a Series aligned to base.index, filled NaN where aux has no data.
    Both frames must have a UTC DatetimeIndex.

    Returns an all-NaN series gracefully if *aux* is missing the column or
    does not have a usable time index -- the caller decides whether to error.
    """
    nan_series = pd.Series(float("nan"), index=base.index, dtype="float64")

    if aux is None or aux.empty:
        return nan_series

    aux_work = aux.copy()

    # Promote timestamp column to index if needed.
    if not isinstance(aux_work.index, pd.DatetimeIndex):
        if "timestamp" in aux_work.columns:
            aux_work = aux_work.set_index("timestamp")
        else:
            return nan_series   # can't align -- return NaN silently

    if col not in aux_work.columns:
        return nan_series

    aux_s = aux_work[[col]].sort_index().groupby(level=0).last()

    merged = pd.merge_asof(
        base[[]].reset_index().rename(columns={base.index.name or "index": "__ts__"}),
        aux_s[[col]].reset_index().rename(columns={aux_s.index.name or "index": "__ts__"}),
        on="__ts__",
        direction="backward",
    )
    result = merged[col].values
    return pd.Series(result, index=base.index, dtype="float64")


def _zscore(s: pd.Series, n: int) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    std  = s.rolling(n, min_periods=n).std(ddof=1).replace(0, np.nan)
    return (s - mean) / std
