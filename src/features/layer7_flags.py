"""Layer 7 -- binary event flags / signals.

All flags are computed from OHLCV + earlier layer columns already merged
into a single wide DataFrame.  They fire at bar-close with no lookahead.

All outputs are float64 ``{0.0, 1.0}`` so they fit cleanly into the
numeric feature matrix and can be shipped verbatim to an LLM as 0/1.

Output columns (FLAG_COLUMNS)
------------------------------
flag_volume_climax     Volume > N x rolling mean (surge / institutional print).
flag_sweep_up          High pierced N-bar high but close pulled back (bull trap / stop hunt).
flag_sweep_dn          Low pierced N-bar low but close reclaimed (bear trap / stop hunt).
flag_squeeze_release   Bollinger Width was in bottom-10th percentile and jumped above median.
flag_macd_cross_up     MACD histogram flipped from <= 0 to > 0 (bullish momentum shift).
flag_macd_cross_dn     MACD histogram flipped from >= 0 to < 0 (bearish momentum shift).
flag_regime_flip       Supertrend direction changed sign (trend reversal).
flag_rsi_overbought    RSI >= 70.
flag_rsi_oversold      RSI <= 30.
flag_adx_strong_trend  ADX >= 25 (strong directional trend in place).
flag_golden_cross      EMA-50 crossed above EMA-200 on this bar.
flag_death_cross       EMA-50 crossed below EMA-200 on this bar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig

FLAG_COLUMNS: tuple[str, ...] = (
    "flag_volume_climax",
    "flag_sweep_up",
    "flag_sweep_dn",
    "flag_squeeze_release",
    "flag_macd_cross_up",
    "flag_macd_cross_dn",
    "flag_regime_flip",
    "flag_rsi_overbought",
    "flag_rsi_oversold",
    "flag_adx_strong_trend",
    "flag_golden_cross",
    "flag_death_cross",
)


def compute(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Compute all flags from a wide DataFrame containing OHLCV + L1-L4 columns.

    Missing indicator columns are handled gracefully -- the associated flag
    is set to 0.0 rather than raising.
    """
    out = pd.DataFrame(0.0, index=df.index, columns=list(FLAG_COLUMNS))

    vol = _col(df, "volume")

    # ------------------------------------------------------------------ Volume climax
    roll_mean = vol.shift(1).rolling(cfg.climax_lookback, min_periods=cfg.climax_lookback).mean()
    out["flag_volume_climax"] = (vol > cfg.climax_mult * roll_mean).astype("float64").fillna(0.0)

    # ------------------------------------------------------------------ Liquidity sweeps
    if {"high", "low", "close", "open"}.issubset(df.columns):
        n          = cfg.sweep_lookback
        prior_high = df["high"].shift(1).rolling(n, min_periods=n).max()
        prior_low  = df["low"].shift(1).rolling(n,  min_periods=n).min()
        sweep_up   = (df["high"] > prior_high) & (df["close"] < prior_high) & (df["close"] < df["open"])
        sweep_dn   = (df["low"]  < prior_low)  & (df["close"] > prior_low)  & (df["close"] > df["open"])
        out["flag_sweep_up"] = sweep_up.astype("float64").fillna(0.0)
        out["flag_sweep_dn"] = sweep_dn.astype("float64").fillna(0.0)

    # ------------------------------------------------------------------ Bollinger squeeze release
    # A squeeze release fires when Bollinger Bandwidth:
    #   - Was in a compressed state (below q10 of recent history), AND
    #   - Crosses above the q50 on the current bar (bw jumps from tight to expanding).
    # Implementation: bw > q50 AND bw_prev <= q50 (crossing event, not a state check).
    # This avoids the bug where checking prior <= q10 fails because by the time
    # bw is in expansion, the prior bar is already the first expansion bar.
    if "bb_width_20" in df.columns:
        bw    = df["bb_width_20"].astype("float64")
        prev  = bw.shift(1)
        sr    = cfg.squeeze_lookback
        q50   = bw.rolling(sr, min_periods=sr).quantile(0.50).shift(1)  # q50 of history
        q10   = bw.rolling(sr, min_periods=sr).quantile(0.10).shift(1)  # q10 of history
        # Squeeze state: previous bar was below q10 (compressed).
        # Release: current bar crossed above q50 while prior was compressed.
        was_compressed = prev <= q10
        crossed_above  = (prev < q50) & (bw >= q50)
        out["flag_squeeze_release"] = (
            (was_compressed | crossed_above).astype("float64").fillna(0.0)
        )
        # More precise: only fire when bw actually crossed q50 upward AND was tight recently.
        # Use 3-bar look-back for "was in squeeze":
        was_tight_3b = prev.rolling(3, min_periods=1).min() <= q10
        out["flag_squeeze_release"] = (
            (was_tight_3b & (bw > q50)).astype("float64").fillna(0.0)
        )

    # ------------------------------------------------------------------ MACD cross
    if "macd_hist" in df.columns:
        h    = df["macd_hist"].astype("float64")
        prev = h.shift(1)
        out["flag_macd_cross_up"] = ((prev <= 0) & (h > 0)).astype("float64").fillna(0.0)
        out["flag_macd_cross_dn"] = ((prev >= 0) & (h < 0)).astype("float64").fillna(0.0)

    # ------------------------------------------------------------------ Supertrend regime flip
    if "supertrend_dir" in df.columns:
        st   = df["supertrend_dir"].astype("float64")
        flip = (np.sign(st) != np.sign(st.shift(1))).astype("float64")
        out["flag_regime_flip"] = flip.fillna(0.0)

    # ------------------------------------------------------------------ RSI extremes
    if "rsi_14" in df.columns:
        r = df["rsi_14"].astype("float64")
        out["flag_rsi_overbought"] = (r >= cfg.rsi_overbought).astype("float64").fillna(0.0)
        out["flag_rsi_oversold"]   = (r <= cfg.rsi_oversold).astype("float64").fillna(0.0)

    # ------------------------------------------------------------------ ADX strong trend
    if "adx_14" in df.columns:
        out["flag_adx_strong_trend"] = (df["adx_14"].astype("float64") >= 25.0).astype("float64").fillna(0.0)

    # ------------------------------------------------------------------ EMA crosses (Golden / Death)
    if "ema_50" in df.columns and "ema_200" in df.columns:
        e50   = df["ema_50"].astype("float64")
        e200  = df["ema_200"].astype("float64")
        above = e50 > e200
        out["flag_golden_cross"] = (above & ~above.shift(1).fillna(False)).astype("float64").fillna(0.0)
        out["flag_death_cross"]  = (~above & above.shift(1).fillna(True)).astype("float64").fillna(0.0)

    # Guarantee stable column order and dtype.
    return out[list(FLAG_COLUMNS)].astype("float64")


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name].astype("float64") if name in df.columns else pd.Series(0.0, index=df.index)
