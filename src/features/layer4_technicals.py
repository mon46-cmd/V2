"""Layer 4 -- classical technical indicators.

Computed from OHLCV candles only.  All Wilder-style indicators use
EMA with alpha=1/n.  Pure, no I/O, no lookahead.

Output columns
--------------
Momentum / oscillators:
  rsi_14                  Relative Strength Index (Wilder).
  macd / macd_signal / macd_hist  Standard MACD(12,26,9).
  stoch_k_14 / stoch_d_3  Stochastic oscillator.
  williams_r_14           Williams %R (-100 to 0).
  cci_20                  Commodity Channel Index.
  mfi_14                  Money Flow Index (volume-weighted RSI).

Trend / bands:
  bb_mid/upper/lower_20   Bollinger Bands (20, 2).
  bb_width_20             Bandwidth = (upper-lower)/mid.
  bb_pct_b_20             %B = (close-lower)/(upper-lower).
  keltner_upper/lower_20  Keltner Channels (EMA +/- 1.5*ATR).
  donchian_hi/lo/pct_20   Donchian Channel.
  supertrend              Supertrend band level.
  supertrend_dir          +1 = bullish, -1 = bearish.
  adx_14 / plus_di_14 / minus_di_14  ADX and directional indicators.

Volume:
  vwap_20_dist            (close - rolling VWAP) / VWAP * 100.
  obv                     On-Balance Volume (cumulative).
  obv_slope_20            20-bar linear-regression slope of OBV.
  cmf_20                  Chaikin Money Flow.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.config import FeatureConfig


def compute(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return a frame with Layer-4 columns aligned to ``df.index``."""
    close = df["close"].astype("float64")
    high  = df["high"].astype("float64")
    low   = df["low"].astype("float64")
    vol   = df["volume"].astype("float64") if "volume" in df.columns else pd.Series(0.0, index=df.index)

    out: dict[str, pd.Series] = {}

    # ------------------------------------------------------------------ RSI
    out["rsi_14"] = _rsi(close, cfg.window_rsi)

    # ------------------------------------------------------------------ MACD
    ema_fast     = close.ewm(span=cfg.macd_fast,   adjust=False, min_periods=cfg.macd_fast).mean()
    ema_slow     = close.ewm(span=cfg.macd_slow,   adjust=False, min_periods=cfg.macd_slow).mean()
    macd         = ema_fast - ema_slow
    macd_signal  = macd.ewm(span=cfg.macd_signal, adjust=False, min_periods=cfg.macd_signal).mean()
    out["macd"]        = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"]   = macd - macd_signal

    # ------------------------------------------------------------------ Bollinger Bands
    bb_mid   = close.rolling(cfg.window_bb, min_periods=cfg.window_bb).mean()
    bb_std   = close.rolling(cfg.window_bb, min_periods=cfg.window_bb).std(ddof=1)
    bb_upper = bb_mid + cfg.bb_stdev_mult * bb_std
    bb_lower = bb_mid - cfg.bb_stdev_mult * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_mid_20"]    = bb_mid
    out["bb_upper_20"]  = bb_upper
    out["bb_lower_20"]  = bb_lower
    out["bb_width_20"]  = bb_range / bb_mid.replace(0, np.nan)
    out["bb_pct_b_20"]  = (close - bb_lower) / bb_range

    # ------------------------------------------------------------------ Keltner Channels
    kelt_ema   = close.ewm(span=cfg.window_keltner, adjust=False, min_periods=cfg.window_keltner).mean()
    tr         = _true_range(high, low, close.shift(1))
    kelt_atr   = _wilder(tr, cfg.window_keltner)
    out["keltner_upper_20"] = kelt_ema + cfg.keltner_mult * kelt_atr
    out["keltner_lower_20"] = kelt_ema - cfg.keltner_mult * kelt_atr

    # ------------------------------------------------------------------ Donchian Channel
    don_hi  = high.rolling(cfg.window_donchian, min_periods=cfg.window_donchian).max()
    don_lo  = low.rolling(cfg.window_donchian,  min_periods=cfg.window_donchian).min()
    don_rng = (don_hi - don_lo).replace(0, np.nan)
    out["donchian_hi_20"]  = don_hi
    out["donchian_lo_20"]  = don_lo
    out["donchian_pct_20"] = (close - don_lo) / don_rng

    # ------------------------------------------------------------------ ADX / DI
    adx_frame         = _adx(high, low, close, cfg.window_adx)
    out["adx_14"]      = adx_frame["adx"]
    out["plus_di_14"]  = adx_frame["+di"]
    out["minus_di_14"] = adx_frame["-di"]

    # ------------------------------------------------------------------ Stochastic
    stoch_hi = high.rolling(cfg.window_stoch_k, min_periods=cfg.window_stoch_k).max()
    stoch_lo = low.rolling(cfg.window_stoch_k,  min_periods=cfg.window_stoch_k).min()
    stoch_k  = 100.0 * (close - stoch_lo) / (stoch_hi - stoch_lo).replace(0, np.nan)
    out["stoch_k_14"] = stoch_k
    out["stoch_d_3"]  = stoch_k.rolling(cfg.window_stoch_d, min_periods=cfg.window_stoch_d).mean()

    # ------------------------------------------------------------------ Williams %R
    w_hi = high.rolling(cfg.window_williams, min_periods=cfg.window_williams).max()
    w_lo = low.rolling(cfg.window_williams,  min_periods=cfg.window_williams).min()
    out["williams_r_14"] = -100.0 * (w_hi - close) / (w_hi - w_lo).replace(0, np.nan)

    # ------------------------------------------------------------------ CCI
    tp      = (high + low + close) / 3.0
    sma_tp  = tp.rolling(cfg.window_cci, min_periods=cfg.window_cci).mean()
    mad     = tp.rolling(cfg.window_cci, min_periods=cfg.window_cci).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    out["cci_20"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # ------------------------------------------------------------------ MFI
    out["mfi_14"] = _mfi(high, low, close, vol, cfg.window_mfi)

    # ------------------------------------------------------------------ Rolling VWAP
    typical  = (high + low + close) / 3.0
    pv       = typical * vol
    roll_pv  = pv.rolling(cfg.window_vwap, min_periods=cfg.window_vwap).sum()
    roll_v   = vol.rolling(cfg.window_vwap, min_periods=cfg.window_vwap).sum().replace(0, np.nan)
    vwap     = roll_pv / roll_v
    out["vwap_20_dist"] = (close - vwap) / vwap.replace(0, np.nan) * 100.0

    # ------------------------------------------------------------------ OBV
    obv = (vol * np.sign(close.diff().fillna(0.0))).cumsum()
    out["obv"]          = obv
    out["obv_slope_20"] = _rolling_slope(obv, cfg.window_obv_slope)

    # ------------------------------------------------------------------ CMF
    hl  = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl
    mfv = mfm * vol
    out["cmf_20"] = (
        mfv.rolling(cfg.window_cmf, min_periods=cfg.window_cmf).sum()
        / vol.rolling(cfg.window_cmf, min_periods=cfg.window_cmf).sum().replace(0, np.nan)
    )

    # ------------------------------------------------------------------ Supertrend
    st, st_dir = _supertrend(high, low, close, cfg.supertrend_atr, cfg.supertrend_mult)
    out["supertrend"]     = st
    out["supertrend_dir"] = st_dir

    return pd.DataFrame(out, index=df.index)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _true_range(high: pd.Series, low: pd.Series, prev_close: pd.Series) -> pd.Series:
    return pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta   = close.diff()
    up      = delta.clip(lower=0.0)
    down    = (-delta).clip(lower=0.0)
    avg_up  = _wilder(up, n)
    avg_dn  = _wilder(down, n)
    rs      = avg_up / avg_dn.replace(0, np.nan)
    rsi     = 100.0 - 100.0 / (1.0 + rs)
    return rsi.where(~avg_dn.eq(0), 100.0)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.DataFrame:
    up_move  = high.diff()
    dn_move  = -low.diff()
    plus_dm  = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=high.index)
    atr      = _wilder(_true_range(high, low, close.shift(1)), n)
    plus_di  = 100.0 * _wilder(plus_dm,  n) / atr.replace(0, np.nan)
    minus_di = 100.0 * _wilder(minus_dm, n) / atr.replace(0, np.nan)
    dx       = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx      = _wilder(dx, n)
    return pd.DataFrame({"+di": plus_di, "-di": minus_di, "adx": adx})


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, vol: pd.Series, n: int) -> pd.Series:
    tp    = (high + low + close) / 3.0
    rmf   = tp * vol
    prev  = tp.shift(1)
    up    = rmf.where(tp > prev, 0.0)
    dn    = rmf.where(tp < prev, 0.0)
    mfr   = (
        up.rolling(n, min_periods=n).sum()
        / dn.rolling(n, min_periods=n).sum().replace(0, np.nan)
    )
    return 100.0 - 100.0 / (1.0 + mfr)


def _rolling_slope(s: pd.Series, n: int) -> pd.Series:
    """Closed-form OLS slope over a rolling window of n bars."""
    if n < 2:
        return pd.Series(np.nan, index=s.index)
    x     = np.arange(n, dtype="float64")
    x_bar = x.mean()
    denom = ((x - x_bar) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return float("nan")
        return float(((x - x_bar) * (y - y.mean())).sum() / denom)

    return s.rolling(n, min_periods=n).apply(_slope, raw=True)


def _supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    atr_n: int, mult: float,
) -> tuple[pd.Series, pd.Series]:
    hl2         = (high + low) / 2.0
    tr          = _true_range(high, low, close.shift(1))
    atr         = _wilder(tr, atr_n)
    upper_basic = hl2 + mult * atr
    lower_basic = hl2 - mult * atr

    n         = len(close)
    upper     = upper_basic.to_numpy().copy()
    lower     = lower_basic.to_numpy().copy()
    st        = np.full(n, np.nan, dtype="float64")
    direction = np.full(n, np.nan, dtype="float64")   # NaN until warm-up done
    c_arr     = close.to_numpy()

    # Find the first fully-warmed bar (first non-NaN ATR).
    atr_arr   = atr.to_numpy()
    finite_mask = np.isfinite(atr_arr)
    if len(atr_arr) == 0 or not finite_mask.any():
        # Empty or all-NaN series — return empty result without crashing.
        return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)
    first = int(np.argmax(finite_mask))
    if not np.isfinite(atr_arr[first]):
        # Entire series is NaN -- return empty.
        return pd.Series(st, index=close.index), pd.Series(direction, index=close.index)

    direction[first] = 1.0
    st[first]        = lower[first]

    for i in range(first + 1, n):
        # Skip bars where the current bands are not yet valid.
        if not np.isfinite(upper[i]) or not np.isfinite(lower[i]):
            continue

        # Previous bars may still be NaN during warm-up -- treat them as the
        # basic band value so the clamp has no effect.
        prev_upper = upper[i - 1] if np.isfinite(upper[i - 1]) else upper[i]
        prev_lower = lower[i - 1] if np.isfinite(lower[i - 1]) else lower[i]
        prev_c     = c_arr[i - 1]

        # Band-clamping: bands only move in the direction of the trend.
        if upper[i] > prev_upper and prev_c <= prev_upper:
            upper[i] = prev_upper
        if lower[i] < prev_lower and prev_c >= prev_lower:
            lower[i] = prev_lower

        # Use previous direction; if still NaN (shouldn't happen after first)
        # default to bullish.
        prev_dir = direction[i - 1] if np.isfinite(direction[i - 1]) else 1.0

        if prev_dir == 1.0:
            direction[i] = -1.0 if c_arr[i] < lower[i] else 1.0
        else:
            direction[i] = 1.0  if c_arr[i] > upper[i] else -1.0
        st[i] = lower[i] if direction[i] == 1.0 else upper[i]

    return (
        pd.Series(st,        index=close.index, dtype="float64"),
        pd.Series(direction, index=close.index, dtype="float64"),
    )
