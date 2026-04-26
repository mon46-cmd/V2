"""Feature computation configuration.

All tunable parameters live here in one place.
No magic numbers should appear in any compute_* function.

Usage::

    from features.config import FeatureConfig
    cfg = FeatureConfig()                      # sensible defaults
    cfg = FeatureConfig(window_rsi=21)         # override one field
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    # ------------------------------------------------------------------ Layer 1: volatility
    window_rv:           int   = 20   # realized-vol (log-return std)
    window_atr:          int   = 14   # Wilder ATR
    window_parkinson:    int   = 20   # Parkinson HL estimator
    window_garman_klass: int   = 20   # Garman-Klass OHLC estimator
    window_yang_zhang:   int   = 20   # Yang-Zhang OHLC estimator

    # ------------------------------------------------------------------ Layer 2: price structure
    swing_lookback:  int = 10    # n-bar pivot high/low detection
    ema_windows:     tuple[int, ...] = field(default_factory=lambda: (9, 21, 50, 200))

    # ------------------------------------------------------------------ Layer 3: regime
    hurst_window: int = 100   # R/S Hurst window (bars of log-returns)
    acf_window:   int = 50    # lag-1 autocorrelation window

    # ------------------------------------------------------------------ Layer 4: technical indicators
    window_rsi:       int   = 14
    macd_fast:        int   = 12
    macd_slow:        int   = 26
    macd_signal:      int   = 9
    window_bb:        int   = 20
    bb_stdev_mult:    float = 2.0
    window_adx:       int   = 14
    window_vwap:      int   = 20    # rolling VWAP period (bars)
    window_obv_slope: int   = 20
    supertrend_atr:   int   = 10
    supertrend_mult:  float = 3.0
    window_stoch_k:   int   = 14
    window_stoch_d:   int   = 3
    window_cci:       int   = 20
    window_williams:  int   = 14
    window_donchian:  int   = 20
    window_keltner:   int   = 20
    keltner_mult:     float = 1.5
    window_cmf:       int   = 20    # Chaikin Money Flow
    window_mfi:       int   = 14    # Money Flow Index
    price_rank_window: int  = 100   # percentile rank look-back

    # ------------------------------------------------------------------ Layer 5: derivatives
    window_funding_z: int = 20    # z-score window for funding rate
    window_oi_z:      int = 50    # z-score window for open interest
    window_basis_z:   int = 50    # z-score window for mark/index basis
    window_ls_z:      int = 20    # z-score window for L/S ratio
    oi_chg_bars_1h:   int = 4     # bars in 1 h at 15-min cadence
    oi_chg_bars_24h:  int = 96    # bars in 24 h at 15-min cadence

    # ------------------------------------------------------------------ Layer 6: volume profile
    window_vp:      int   = 200   # look-back bars for VP computation
    vp_num_bins:    int   = 50    # price buckets in the histogram
    vp_value_area:  float = 0.70  # fraction of volume defining VA

    # ------------------------------------------------------------------ Layer 7: flags / signals
    climax_lookback: int   = 20    # rolling mean window for volume climax
    climax_mult:     float = 2.5   # volume > mult * rolling mean => climax
    sweep_lookback:  int   = 10    # N-bar prior high/low for sweep detection
    squeeze_lookback: int  = 100   # window for Bollinger squeeze percentile
    rsi_overbought:  float = 70.0
    rsi_oversold:    float = 30.0
