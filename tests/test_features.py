"""Tests for the feature computation layers.

These tests are pure (no network calls).  They use synthetic OHLCV data
generated from a seeded random walk so results are deterministic.

Run with::

    pytest tests/test_features.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.config  import FeatureConfig
from features.pipeline import compute_snapshot, get_last_row
import features.layer1_volatility    as l1
import features.layer2_structure     as l2
import features.layer3_regime        as l3
import features.layer4_technicals    as l4
import features.layer5_derivatives   as l5
import features.layer6_volume_profile as l6
import features.layer7_flags         as l7
import features.layer8_advanced      as l8
from features.layer7_flags import FLAG_COLUMNS


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture(scope="module")
def ohlcv() -> pd.DataFrame:
    """300 bars of synthetic OHLCV data on a 15-min UTC grid."""
    rng   = np.random.default_rng(42)
    n     = 300
    ts    = pd.date_range("2026-04-01", periods=n, freq="15min", tz="UTC")
    close = 30_000 + np.cumsum(rng.normal(0, 100, n))
    noise = rng.uniform(50, 300, n)
    df    = pd.DataFrame({
        "open":   close - rng.uniform(0, 50, n),
        "high":   close + noise,
        "low":    close - noise,
        "close":  close,
        "volume": rng.uniform(10, 500, n),
    }, index=ts)
    # Ensure OHLC consistency.
    df["high"]  = df[["open", "close", "high"]].max(axis=1)
    df["low"]   = df[["open", "close", "low"]].min(axis=1)
    return df.astype("float64")


def _assert_output(df_out: pd.DataFrame, df_in: pd.DataFrame, expected_cols: list[str]) -> None:
    """Common checks: same index, expected columns, no all-NaN columns."""
    assert df_out.index.equals(df_in.index), "output index does not match input index"
    for col in expected_cols:
        assert col in df_out.columns, f"missing column: {col!r}"
    # Allow a warm-up NaN prefix but never an entirely NaN column.
    for col in expected_cols:
        assert df_out[col].notna().any(), f"column {col!r} is entirely NaN"


# ===========================================================================
# Layer 1 -- volatility
# ===========================================================================

class TestLayer1:
    def test_output_columns(self, ohlcv, cfg):
        out = l1.compute(ohlcv, cfg)
        _assert_output(out, ohlcv, ["ret", "log_ret", "rv_20", "atr_14", "atr_14_pct",
                                     "parkinson_20", "garman_klass_20", "yang_zhang_20"])

    def test_ret_log_ret_relationship(self, ohlcv, cfg):
        """log_ret ~= ret for small returns."""
        out   = l1.compute(ohlcv, cfg).dropna()
        diff  = (out["log_ret"] - out["ret"]).abs()
        # Max absolute difference should be small.
        assert diff.max() < 0.05, "log_ret and ret diverge unexpectedly"

    def test_atr_positive(self, ohlcv, cfg):
        out = l1.compute(ohlcv, cfg)
        assert (out["atr_14"].dropna() > 0).all(), "ATR must be strictly positive"

    def test_volatility_estimators_positive(self, ohlcv, cfg):
        out = l1.compute(ohlcv, cfg)
        for col in ("rv_20", "parkinson_20", "garman_klass_20", "yang_zhang_20"):
            vals = out[col].dropna()
            assert (vals >= 0).all(), f"{col!r} has negative values"

    def test_no_lookahead_spot_check(self, ohlcv, cfg):
        """Recomputing on a prefix must give the same last-common-row value."""
        full = l1.compute(ohlcv, cfg)
        half = l1.compute(ohlcv.iloc[:150], cfg)
        row  = 149
        assert abs(float(full.iloc[row]["atr_14"]) - float(half.iloc[row]["atr_14"])) < 1e-9


# ===========================================================================
# Layer 2 -- price structure
# ===========================================================================

class TestLayer2:
    def test_output_columns(self, ohlcv, cfg):
        out = l2.compute(ohlcv, cfg)
        _assert_output(out, ohlcv, [
            "ema_9", "ema_21", "ema_50", "ema_200",
            "price_pct_rank_100",
            "swing_high_10", "swing_low_10",
            "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
        ])

    def test_ema_smoothness(self, ohlcv, cfg):
        """EMA-200 should be smoother (lower std) than EMA-9."""
        out = l2.compute(ohlcv, cfg).dropna()
        assert out["ema_200"].std() < out["ema_9"].std(), "EMA-200 should be smoother than EMA-9"

    def test_swing_high_gte_swing_low(self, ohlcv, cfg):
        out = l2.compute(ohlcv, cfg).dropna()
        assert (out["swing_high_10"] >= out["swing_low_10"]).all()

    def test_candle_anatomy_bounds(self, ohlcv, cfg):
        out = l2.compute(ohlcv, cfg).dropna()
        assert out["candle_body_pct"].between(0.0, 1.0).all(), "candle_body_pct out of [0,1]"
        assert (out["upper_wick_pct"] >= 0).all()
        assert (out["lower_wick_pct"] >= 0).all()

    def test_pct_rank_bounds(self, ohlcv, cfg):
        out = l2.compute(ohlcv, cfg)
        rank = out["price_pct_rank_100"].dropna()
        assert rank.between(0.0, 1.0).all(), "price_pct_rank must be in [0, 1]"


# ===========================================================================
# Layer 3 -- regime
# ===========================================================================

class TestLayer3:
    def test_output_columns(self, ohlcv, cfg):
        out = l3.compute(ohlcv, cfg)
        _assert_output(out, ohlcv, ["hurst_100", "vr_2_100", "acf1_50"])

    def test_hurst_range(self, ohlcv, cfg):
        """Hurst should be in (0, 1) for typical price series."""
        h = l3.compute(ohlcv, cfg)["hurst_100"].dropna()
        assert h.between(0.0, 1.0).all(), f"Hurst out of (0,1): min={h.min():.3f} max={h.max():.3f}"

    def test_vr_positive(self, ohlcv, cfg):
        vr = l3.compute(ohlcv, cfg)["vr_2_100"].dropna()
        assert (vr > 0).all(), "variance ratio must be positive"

    def test_acf1_range(self, ohlcv, cfg):
        """ACF1 is correlation, must be in (-1, 1)."""
        acf = l3.compute(ohlcv, cfg)["acf1_50"].dropna()
        assert acf.between(-1.0, 1.0).all()


# ===========================================================================
# Layer 4 -- technicals
# ===========================================================================

class TestLayer4:
    def test_output_columns(self, ohlcv, cfg):
        out = l4.compute(ohlcv, cfg)
        _assert_output(out, ohlcv, [
            "rsi_14", "macd", "macd_hist", "bb_mid_20", "bb_width_20",
            "adx_14", "stoch_k_14", "vwap_20_dist", "obv", "cmf_20",
            "supertrend", "supertrend_dir",
        ])

    def test_rsi_bounds(self, ohlcv, cfg):
        rsi = l4.compute(ohlcv, cfg)["rsi_14"].dropna()
        assert rsi.between(0.0, 100.0).all(), f"RSI out of [0,100]: {rsi.describe()}"

    def test_macd_signal_relationship(self, ohlcv, cfg):
        """macd_hist = macd - macd_signal."""
        out  = l4.compute(ohlcv, cfg).dropna()
        diff = (out["macd_hist"] - (out["macd"] - out["macd_signal"])).abs()
        assert diff.max() < 1e-8

    def test_bb_ordering(self, ohlcv, cfg):
        out = l4.compute(ohlcv, cfg).dropna()
        assert (out["bb_upper_20"] >= out["bb_mid_20"]).all()
        assert (out["bb_mid_20"]   >= out["bb_lower_20"]).all()

    def test_bb_pct_b_range(self, ohlcv, cfg):
        """For a random walk %B should have values both below 0 and above 1 occasionally."""
        pct_b = l4.compute(ohlcv, cfg)["bb_pct_b_20"].dropna()
        assert pct_b.min() < 0.3, "expected some bars near/below lower band"
        assert pct_b.max() > 0.7, "expected some bars near/above upper band"

    def test_stoch_bounds(self, ohlcv, cfg):
        stoch = l4.compute(ohlcv, cfg)["stoch_k_14"].dropna()
        assert stoch.between(0.0, 100.0).all()

    def test_supertrend_dir_values(self, ohlcv, cfg):
        st_dir = l4.compute(ohlcv, cfg)["supertrend_dir"].dropna()
        assert st_dir.isin([-1.0, 1.0]).all(), "supertrend_dir must be +/-1"

    def test_adx_non_negative(self, ohlcv, cfg):
        adx = l4.compute(ohlcv, cfg)["adx_14"].dropna()
        assert (adx >= 0).all()

    def test_obv_monotonic_consistency(self, ohlcv, cfg):
        """OBV should be non-zero (it accumulates volume)."""
        obv = l4.compute(ohlcv, cfg)["obv"]
        assert obv.abs().max() > 0


# ===========================================================================
# Layer 5 -- derivatives (synthetic data)
# ===========================================================================

class TestLayer5:
    @pytest.fixture
    def funding_df(self, ohlcv):
        ts = pd.date_range(ohlcv.index[0], periods=30, freq="8h", tz="UTC")
        return pd.DataFrame({"funding_rate": np.random.default_rng(1).normal(0.0001, 0.0002, 30)}, index=ts)

    @pytest.fixture
    def oi_df(self, ohlcv):
        return pd.DataFrame({"open_interest": np.linspace(1e9, 1.2e9, 300)}, index=ohlcv.index)

    def test_funding_columns(self, ohlcv, cfg, funding_df):
        out = l5.compute(ohlcv, cfg, funding=funding_df)
        for col in ("funding_rate", "funding_z_20", "funding_annualized"):
            assert col in out.columns

    def test_oi_columns(self, ohlcv, cfg, oi_df):
        out = l5.compute(ohlcv, cfg, oi=oi_df)
        for col in ("oi", "oi_chg_pct", "oi_z_50"):
            assert col in out.columns

    def test_no_derivatives_returns_empty(self, ohlcv, cfg):
        out = l5.compute(ohlcv, cfg)
        assert out.empty or out.columns.tolist() == []

    def test_basis_requires_both_mark_and_index(self, ohlcv, cfg):
        """Passing only mark (no index_p) should NOT produce basis_bps."""
        mark = pd.DataFrame({"close": ohlcv["close"] + 10}, index=ohlcv.index)
        out  = l5.compute(ohlcv, cfg, mark=mark)  # index_p=None
        assert "basis_bps" not in out.columns


# ===========================================================================
# Layer 6 -- volume profile
# ===========================================================================

class TestLayer6:
    def test_output_columns(self, ohlcv, cfg):
        out = l6.compute(ohlcv, cfg)
        _assert_output(out, ohlcv, ["poc_price", "vah_price", "val_price",
                                     "poc_dist_pct", "value_area_width"])

    def test_ordering(self, ohlcv, cfg):
        """VAH >= POC >= VAL for every non-NaN row."""
        out = l6.compute(ohlcv, cfg).dropna()
        assert (out["vah_price"] >= out["poc_price"]).all()
        assert (out["poc_price"] >= out["val_price"]).all()

    def test_value_area_width_positive(self, ohlcv, cfg):
        width = l6.compute(ohlcv, cfg)["value_area_width"].dropna()
        assert (width >= 0).all()


# ===========================================================================
# Layer 7 -- flags
# ===========================================================================

class TestLayer7:
    def test_all_flag_columns_present(self, ohlcv, cfg):
        wide = pd.concat([ohlcv, l1.compute(ohlcv, cfg), l4.compute(ohlcv, cfg)], axis=1)
        out  = l7.compute(wide, cfg)
        for col in FLAG_COLUMNS:
            assert col in out.columns

    def test_flags_binary(self, ohlcv, cfg):
        wide = pd.concat([ohlcv, l4.compute(ohlcv, cfg)], axis=1)
        out  = l7.compute(wide, cfg)
        for col in FLAG_COLUMNS:
            assert out[col].isin([0.0, 1.0]).all(), f"{col!r} has non-binary values"

    def test_volume_climax_fires(self, ohlcv, cfg):
        """At least one bar should have a volume climax in 300 bars."""
        # Inject a guaranteed 4x volume spike at bar 150 to reliably trigger the flag
        ohlcv_mod = ohlcv.copy()
        spike_idx = 150
        mean_vol  = ohlcv_mod["volume"].iloc[:spike_idx].mean()
        ohlcv_mod.iloc[spike_idx, ohlcv_mod.columns.get_loc("volume")] = mean_vol * 4.0

        wide = pd.concat([ohlcv_mod, l4.compute(ohlcv_mod, cfg)], axis=1)
        out  = l7.compute(wide, cfg)
        assert out["flag_volume_climax"].sum() > 0, "no volume climax detected even with 4x spike"


# ===========================================================================
# Layer 8 -- advanced predictive features
# ===========================================================================

class TestLayer8:
    @pytest.fixture
    def frames(self, ohlcv, cfg):
        f1 = l1.compute(ohlcv, cfg)
        f2 = l2.compute(ohlcv, cfg)
        f4 = l4.compute(ohlcv, cfg)
        return f1, f2, f4

    def test_output_columns(self, ohlcv, cfg, frames):
        f1, f2, f4 = frames
        out = l8.compute(ohlcv, cfg, l1_frame=f1, l2_frame=f2, l4_frame=f4)
        # ud_vol_ratio_20 can be all-NaN when the synthetic fixture has no down-bars
        # (close < open) in the rolling window — this is expected, not a bug.
        # It is verified separately with a fixture that guarantees down-bars.
        check_cols = [
            "kyle_lambda_20", "amihud_20", "vrds_20",
            "ret_skew_20", "ret_kurt_20",
            "efficiency_ratio_20", "rvol_20", "nrei",
            "sharpe_20", "clv",
            "vol_regime_score", "bull_streak", "bear_streak",
        ]
        _assert_output(out, ohlcv, check_cols)
        # ud_vol_ratio_20 must exist as a column (may be NaN for degenerate data)
        assert "ud_vol_ratio_20" in out.columns, "ud_vol_ratio_20 column missing"

    def test_ud_vol_ratio_with_mixed_bars(self, cfg):
        """ud_vol_ratio_20 is non-NaN when there are both up-bars and down-bars."""
        n     = 100
        idx   = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        # Alternate up/down bars explicitly
        close = pd.Series([100.0 + (1.0 if i % 2 == 0 else -1.0) for i in range(n)], index=idx)
        openp = pd.Series([100.0] * n, index=idx)
        df    = pd.DataFrame({
            "open":   openp,
            "high":   close + 0.5,
            "low":    close - 0.5,
            "close":  close,
            "volume": pd.Series(100.0, index=idx),
        })
        out   = l8.compute(df, cfg)
        ratio = out["ud_vol_ratio_20"].dropna()
        assert len(ratio) > 0, "ud_vol_ratio_20 all NaN even with explicit down-bars"
        assert (ratio > 0).all(), "ud_vol_ratio_20 must be positive"

    def test_clv_bounds(self, ohlcv, cfg):
        """CLV is in [-1, +1] by construction."""
        out = l8.compute(ohlcv, cfg)
        clv = out["clv"].dropna()
        assert clv.between(-1.0, 1.0).all(), f"CLV out of [-1,1]: min={clv.min():.3f} max={clv.max():.3f}"

    def test_efficiency_ratio_bounds(self, ohlcv, cfg):
        """ER in [0, 1]: 0 = complete chop, 1 = perfect trend."""
        out = l8.compute(ohlcv, cfg)
        er  = out["efficiency_ratio_20"].dropna()
        assert (er >= 0.0).all(), "efficiency ratio below 0"
        assert (er <= 1.0 + 1e-9).all(), f"efficiency ratio above 1: max={er.max():.4f}"

    def test_rvol_positive(self, ohlcv, cfg):
        out = l8.compute(ohlcv, cfg)
        rvol = out["rvol_20"].dropna()
        assert (rvol > 0).all(), "RVOL must be positive"

    def test_nrei_positive(self, ohlcv, cfg):
        """NREI = bar_range / ATR, both positive, so result >= 0."""
        out = l8.compute(ohlcv, cfg)
        nrei = out["nrei"].dropna()
        assert (nrei >= 0).all()

    def test_streaks_non_negative(self, ohlcv, cfg):
        out = l8.compute(ohlcv, cfg)
        assert (out["bull_streak"] >= 0).all()
        assert (out["bear_streak"] >= 0).all()

    def test_streaks_increase_then_reset(self, ohlcv, cfg):
        """A streak counter must only increase by 1 or reset to 0/1."""
        out  = l8.compute(ohlcv, cfg)
        bull = out["bull_streak"].to_numpy()
        for i in range(1, len(bull)):
            prev, cur = bull[i - 1], bull[i]
            assert cur == 0 or cur == prev + 1 or cur == 1, \
                f"invalid streak transition {prev} -> {cur} at index {i}"

    def test_realized_skew_varies(self, ohlcv, cfg):
        """Skewness over a random walk should not be constant."""
        out  = l8.compute(ohlcv, cfg)
        skew = out["ret_skew_20"].dropna()
        assert skew.std() > 0.0, "skewness is constant -- likely a bug"

    def test_kyle_lambda_positive(self, ohlcv, cfg):
        out = l8.compute(ohlcv, cfg)
        kl  = out["kyle_lambda_20"].dropna()
        assert (kl >= 0).all(), "Kyle lambda must be non-negative"

    def test_tqi_present_when_l4_supplied(self, ohlcv, cfg, frames):
        f1, f2, f4 = frames
        out = l8.compute(ohlcv, cfg, l4_frame=f4)
        assert "tqi" in out.columns, "TQI should be computed when l4_frame is supplied"

    def test_wae_present_when_l4_supplied(self, ohlcv, cfg, frames):
        f1, f2, f4 = frames
        out = l8.compute(ohlcv, cfg, l4_frame=f4)
        assert "wae_explosion" in out.columns
        assert "wae_direction" in out.columns

    def test_wick_asymmetry_when_l2_supplied(self, ohlcv, cfg, frames):
        f1, f2, f4 = frames
        out = l8.compute(ohlcv, cfg, l2_frame=f2)
        assert "wick_asymmetry"   in out.columns
        assert "wick_asymmetry_5" in out.columns

    def test_no_lookahead_efficiency_ratio(self, ohlcv, cfg):
        """Recomputing on a prefix must give the same value at the cut-off row."""
        full = l8.compute(ohlcv, cfg)
        half = l8.compute(ohlcv.iloc[:150], cfg)
        v_full = float(full.iloc[149]["efficiency_ratio_20"])
        v_half = float(half.iloc[149]["efficiency_ratio_20"])
        assert abs(v_full - v_half) < 1e-9, "efficiency_ratio has lookahead bias"


# ===========================================================================
# Full pipeline
# ===========================================================================

class TestPipeline:
    def test_compute_snapshot_runs(self, ohlcv, cfg):
        snap = compute_snapshot(ohlcv, cfg)
        assert isinstance(snap, pd.DataFrame)
        assert len(snap) == len(ohlcv)
        assert snap.index.equals(ohlcv.index)

    def test_snapshot_has_all_layer_columns(self, ohlcv, cfg):
        snap = compute_snapshot(ohlcv, cfg)
        # Spot-check one column per layer.
        for col in ("atr_14", "ema_21", "hurst_100", "rsi_14",
                    "poc_price", "flag_volume_climax",
                    "clv", "efficiency_ratio_20"):       # L8
            assert col in snap.columns, f"missing {col!r} from snapshot"

    def test_get_last_row_returns_series(self, ohlcv, cfg):
        snap     = compute_snapshot(ohlcv, cfg)
        last_row = get_last_row(snap)
        assert isinstance(last_row, pd.Series)
        assert len(last_row) == len(snap.columns)

    def test_selective_layers(self, ohlcv, cfg):
        """Requesting only layers 1 and 4 should NOT include layer 3 or 8 columns."""
        snap = compute_snapshot(ohlcv, cfg, layers=(1, 4))
        assert "atr_14"              in snap.columns
        assert "rsi_14"              in snap.columns
        assert "hurst_100"           not in snap.columns    # L3 excluded
        assert "poc_price"           not in snap.columns    # L6 excluded
        assert "efficiency_ratio_20" not in snap.columns    # L8 excluded

    def test_empty_ohlcv_graceful(self, cfg):
        empty = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            dtype="float64",
        )
        empty.index = pd.DatetimeIndex([], tz="UTC")
        snap = compute_snapshot(empty, cfg)
        assert snap.empty or len(snap) == 0
