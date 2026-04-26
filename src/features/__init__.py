"""Features package: compute market features from OHLCV + derivatives data.

All feature computation is:
  - Pure (no I/O, no global state, no network calls).
  - No-lookahead: every row at index t uses only data with index <= t.
  - Graceful: missing optional inputs (funding, OI, etc.) produce NaN
    columns rather than raising.

Quick-start
-----------
::

    from features.pipeline import compute_snapshot
    from features.config   import FeatureConfig

    cfg      = FeatureConfig()
    snapshot = compute_snapshot(ohlcv_df, cfg)
    last_bar = snapshot.iloc[-1]        # dict-like row for the AI context

Layer map
---------
1 layer1_volatility    -- ret, log_ret, rv_20, atr_14, parkinson_20, garman_klass_20, yang_zhang_20
2 layer2_structure     -- EMA(9/21/50/200) + dist + slope, price_pct_rank, swing H/L, candle anatomy
3 layer3_regime        -- hurst_100, vr_2_100, acf1_50
4 layer4_technicals    -- RSI, MACD, Bollinger, ADX, Stoch, Williams %R, CCI, MFI, VWAP, OBV, CMF, Supertrend
5 layer5_derivatives   -- funding_rate/z, OI/z, basis_bps/z, premium, L/S ratio/z
6 layer6_volume_profile -- POC, VAH, VAL, distances, value_area_width
7 layer7_flags         -- 12 binary event flags (volume climax, sweep, MACD cross, etc.)
8 layer8_advanced      -- Kyle lambda, Amihud, VRDS, realized skew/kurt, efficiency ratio,
                          RVOL, NREI, rolling Sharpe, CLV, up/down vol ratio,
                          vol regime score, TQI, Waddah Attar Explosion,
                          wick asymmetry, bull/bear streak
"""
from __future__ import annotations

from features.config   import FeatureConfig
from features.pipeline import compute_snapshot, get_last_row
from features.layer7_flags import FLAG_COLUMNS

__all__ = [
    "FeatureConfig",
    "compute_snapshot",
    "get_last_row",
    "FLAG_COLUMNS",
]
