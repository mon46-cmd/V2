"""Layer 8 -- advanced predictive features.

These are non-trivial, research-backed indicators with documented
predictive edge.  All are pure, no-lookahead, and computed from
data available at bar close.

Features implemented
--------------------

1. KYLE LAMBDA -- price-impact coefficient.
   Regresses abs(price change) / volume over a rolling window.
   High lambda = thin book / high market impact = institutional flow.
   Source: Kyle (1985), Glosten-Milgrom. Used in HFT flow research.

2. AMIHUD ILLIQUIDITY -- daily price impact per unit of volume.
   |ret| / volume.  Rolling mean.  High = illiquid = easier to move price.
   Source: Amihud (2002). Widely used in crypto market micro-structure.

3. VOLUME-WEIGHTED RETURN DISPERSION (VRDS).
   Compares up-volume return vs down-volume return.
   Large positive = bulls absorbing at price; negative = distribution.

4. REALIZED SKEWNESS & KURTOSIS (rolling).
   Skewness of log-returns over N bars.  Negative skew = fat left tail
   (crash risk). Excess kurtosis > 3 = fat tails = regime instability.

5. PRICE EFFICIENCY RATIO (Kaufman).
   Net directional move / sum of absolute moves.  ~1 = trending cleanly,
   ~0 = choppy / noise.  Basis for KAMA.
   Source: Kaufman (1995). Used in trend-following systems.

6. RELATIVE VOLUME (RVOL).
   Current bar volume / rolling median volume.
   RVOL > 2 = unusual activity / potential breakout catalyst.

7. NORMALIZED RANGE EXPANSION INDEX (NREI).
   (High - Low) / ATR(N).  Bar range as a multiple of typical range.
   Spike = expansion / exhaustion candle.

8. ROLLING SHARPE RATIO.
   Mean(ret) / Std(ret) over N bars (annualized by sqrt bars/year).
   Momentum quality metric: high Sharpe = momentum without huge swings.

9. CLOSE LOCATION VALUE (CLV).
   (Close - Low - (High - Close)) / (High - Low).
   Range: [-1, +1].  +1 = close at high (strong buying pressure).
   Basis for accumulation/distribution line.

10. UP / DOWN VOLUME RATIO.
    Volume when close > open / volume when close < open, rolling.
    > 1 = buying dominates. Useful divergence signal vs price.

11. VOLATILITY REGIME SCORE.
    Percentile rank of current rv_20 over 252 bars.
    0 = historically calm, 1 = historically extreme volatility.

12. TREND QUALITY INDEX (TQI).
    ADX * directional_bias (plus_di - minus_di) / (plus_di + minus_di).
    Combines trend strength + direction into a single signed metric.
    Positive = strong up-trend, Negative = strong down-trend.

13. WADDAH ATTAR EXPLOSION.
    Momentum of MACD histogram relative to current Bollinger bandwidth.
    Detects when momentum spike exceeds volatility -- strong breakout signal.
    Source: Waddah Attar indicator, popularized in MT4 trading community.

14. INTRABAR VOLATILITY ASYMMETRY.
    upper_wick_pct - lower_wick_pct (from L2).
    Positive = sellers pushed back from high = supply overhead.
    Negative = buyers defended the low = demand present.

15. PRICE PRESSURE (exhaustion proxy).
    Number of consecutive bars close > open (bullish pressure streak) or
    close < open (bearish pressure streak). Extreme streaks often precede
    reversals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute(
    df:  pd.DataFrame,
    cfg: FeatureConfig,
    *,
    l1_frame: pd.DataFrame | None = None,
    l2_frame: pd.DataFrame | None = None,
    l4_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a frame with Layer-8 columns aligned to ``df.index``.

    Args:
        df:       OHLCV base DataFrame.
        cfg:      FeatureConfig.
        l1_frame: Output of layer1_volatility.compute() (optional, used for rv).
        l2_frame: Output of layer2_structure.compute()  (optional, used for wick).
        l4_frame: Output of layer4_technicals.compute() (optional, used for ADX, BB).
    """
    close = df["close"].astype("float64")
    high  = df["high"].astype("float64")
    low   = df["low"].astype("float64")
    openp = df["open"].astype("float64") if "open" in df.columns else close
    vol   = df["volume"].astype("float64") if "volume" in df.columns else pd.Series(1.0, index=df.index)

    # Helper: grab a pre-computed column from an earlier layer frame or recompute inline.
    def _get(frame: pd.DataFrame | None, col: str) -> pd.Series | None:
        if frame is not None and col in frame.columns:
            return frame[col].astype("float64")
        return None

    prev_close = close.shift(1)
    log_ret    = np.log(close / prev_close)

    out: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------ 1. Kyle Lambda
    abs_ret = log_ret.abs()
    # Clip zero-volume bars to avoid inf.
    vol_safe = vol.replace(0, np.nan)
    price_impact = abs_ret / vol_safe
    out["kyle_lambda_20"] = price_impact.rolling(cfg.window_rv, min_periods=cfg.window_rv).mean()

    # ------------------------------------------------------------------ 2. Amihud Illiquidity
    out["amihud_20"] = price_impact.rolling(cfg.window_rv, min_periods=cfg.window_rv).mean() * 1e6

    # ------------------------------------------------------------------ 3. Volume-Weighted Return Dispersion
    # Separate up-bar and down-bar flows.
    ret    = close / prev_close - 1.0
    is_up  = (ret > 0).astype("float64")
    is_dn  = (ret < 0).astype("float64")
    n_vwrd = cfg.window_rv

    up_vol = (vol * is_up).rolling(n_vwrd, min_periods=n_vwrd).sum()
    dn_vol = (vol * is_dn).rolling(n_vwrd, min_periods=n_vwrd).sum()
    up_pv  = (vol * is_up * ret).rolling(n_vwrd, min_periods=n_vwrd).sum()
    dn_pv  = (vol * is_dn * ret.abs()).rolling(n_vwrd, min_periods=n_vwrd).sum()

    vw_up_ret = up_pv / up_vol.replace(0, np.nan)
    vw_dn_ret = dn_pv / dn_vol.replace(0, np.nan)
    out["vrds_20"] = vw_up_ret - vw_dn_ret

    # ------------------------------------------------------------------ 4. Realized Skewness & Kurtosis
    n_sk = cfg.window_rv

    def _skew(y: np.ndarray) -> float:
        m = np.nanmean(y)
        s = np.nanstd(y, ddof=1)
        if s < 1e-12:
            return 0.0
        return float(np.nanmean(((y - m) / s) ** 3))

    def _kurt(y: np.ndarray) -> float:
        m = np.nanmean(y)
        s = np.nanstd(y, ddof=1)
        if s < 1e-12:
            return 0.0
        return float(np.nanmean(((y - m) / s) ** 4) - 3.0)  # excess kurtosis

    lr_arr = log_ret.to_numpy()
    out["ret_skew_20"]  = pd.Series(
        _rolling_apply(lr_arr, n_sk, _skew), index=df.index, dtype="float64"
    )
    out["ret_kurt_20"]  = pd.Series(
        _rolling_apply(lr_arr, n_sk, _kurt), index=df.index, dtype="float64"
    )

    # ------------------------------------------------------------------ 5. Price Efficiency Ratio
    # Kaufman ER = net directional log-move / sum of |bar log-returns|.
    # Range [0, 1]: 1 = perfectly trending, 0 = random walk.
    # BUG fix: must compare log-space moves, not raw price vs log path.
    n_er = cfg.window_rv
    net_log_move  = log_ret.rolling(n_er, min_periods=n_er).sum().abs()  # |net log move|
    total_log_path = log_ret.abs().rolling(n_er, min_periods=n_er).sum()  # sum |bar log returns|
    out["efficiency_ratio_20"] = net_log_move / total_log_path.replace(0, np.nan)

    # ------------------------------------------------------------------ 6. Relative Volume (RVOL)
    # Normalize against median of same-window bars, not raw bar volume.
    # Use lagged median (shift 1) to avoid look-ahead on the current bar.
    vol_median = vol.shift(1).rolling(cfg.window_rv, min_periods=max(5, cfg.window_rv // 4)).median().replace(0, np.nan)
    out["rvol_20"] = vol / vol_median

    # ------------------------------------------------------------------ 7. Normalized Range Expansion Index
    bar_range = high - low
    # ATR: either reuse from L1 or recompute.
    atr_col = _get(l1_frame, "atr_14")
    if atr_col is None:
        tr  = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_col = tr.ewm(alpha=1.0 / cfg.window_atr, adjust=False, min_periods=cfg.window_atr).mean()
    out["nrei"] = bar_range / atr_col.replace(0, np.nan)

    # ------------------------------------------------------------------ 8. Rolling Sharpe Ratio (unnormalized)
    # Intentionally NOT annualized: for a 20-bar window, mean/std gives a clean
    # [-3, +3] ratio the AI can reason about directly.
    # Annualizing with sqrt(35040) produces values like 40+ which are meaningless.
    n_sh     = cfg.window_rv
    roll_ret = log_ret.rolling(n_sh, min_periods=n_sh).mean()
    roll_std = log_ret.rolling(n_sh, min_periods=n_sh).std(ddof=1).replace(0, np.nan)
    out["sharpe_20"] = roll_ret / roll_std   # range ~[-3, +3], >0.5 = positive momentum

    # ------------------------------------------------------------------ 9. Close Location Value (CLV)
    hl = (high - low).replace(0, np.nan)
    out["clv"] = ((close - low) - (high - close)) / hl

    # ------------------------------------------------------------------ 10. Up / Down Volume Ratio
    up_vol_roll = (vol * (close >= openp).astype("float64")).rolling(cfg.window_rv, min_periods=cfg.window_rv).sum()
    dn_vol_roll = (vol * (close <  openp).astype("float64")).rolling(cfg.window_rv, min_periods=cfg.window_rv).sum()
    out["ud_vol_ratio_20"] = up_vol_roll / dn_vol_roll.replace(0, np.nan)

    # ------------------------------------------------------------------ 11. Volatility Regime Score
    rv_col = _get(l1_frame, "rv_20")
    if rv_col is None:
        rv_col = log_ret.rolling(cfg.window_rv, min_periods=cfg.window_rv).std(ddof=1)
    out["vol_regime_score"] = rv_col.rolling(252, min_periods=50).rank(pct=True)

    # ------------------------------------------------------------------ 12. Trend Quality Index (TQI)
    adx_col   = _get(l4_frame, "adx_14")
    plus_col  = _get(l4_frame, "plus_di_14")
    minus_col = _get(l4_frame, "minus_di_14")
    if adx_col is not None and plus_col is not None and minus_col is not None:
        di_sum  = (plus_col + minus_col).replace(0, np.nan)
        dir_bias = (plus_col - minus_col) / di_sum        # range (-1, +1)
        out["tqi"] = adx_col * dir_bias                   # range (-100, +100)

    # ------------------------------------------------------------------ 13. Waddah Attar Explosion
    macd_hist_col = _get(l4_frame, "macd_hist")
    bb_width_col  = _get(l4_frame, "bb_width_20")
    if macd_hist_col is not None and bb_width_col is not None:
        # Momentum magnitude: change in MACD histogram vs bandwidth
        macd_delta = macd_hist_col - macd_hist_col.shift(1)
        # Explosion = abs(momentum change) / volatility proxy
        out["wae_explosion"] = macd_delta.abs() / bb_width_col.replace(0, np.nan)
        # Sign carries direction: positive = bullish thrust
        out["wae_direction"] = np.sign(macd_delta)

    # ------------------------------------------------------------------ 14. Intrabar Volatility Asymmetry
    uw_col = _get(l2_frame, "upper_wick_pct")
    lw_col = _get(l2_frame, "lower_wick_pct")
    if uw_col is not None and lw_col is not None:
        out["wick_asymmetry"] = uw_col - lw_col
        # Smooth over 5 bars to reduce bar-by-bar noise.
        out["wick_asymmetry_5"] = out["wick_asymmetry"].rolling(5, min_periods=3).mean()

    # ------------------------------------------------------------------ 15. Price Pressure (streak)
    bullish = (close > openp).astype(int)
    bearish = (close < openp).astype(int)
    out["bull_streak"] = pd.Series(
        _streak(bullish.to_numpy()), index=df.index, dtype="float64"
    )
    out["bear_streak"] = pd.Series(
        _streak(bearish.to_numpy()), index=df.index, dtype="float64"
    )

    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rolling_apply(arr: np.ndarray, window: int, fn) -> np.ndarray:
    """Rolling apply of a scalar function over a 1-D numpy array."""
    n   = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = fn(arr[i - window + 1: i + 1])
    return out


def _streak(arr: np.ndarray) -> np.ndarray:
    """Return the current consecutive-run length for each bar.

    +N = N consecutive 1s (bullish bars),
    -N = N consecutive 0-or-1 transition (bearish bars, use separate input).
    Returns +count when the current bar is 1, 0 otherwise.
    """
    n   = len(arr)
    out = np.zeros(n, dtype="float64")
    cur = 0
    for i in range(n):
        if arr[i] == 1:
            cur += 1
        else:
            cur = 0
        out[i] = float(cur)
    return out
