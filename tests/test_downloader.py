"""Real-network tests for the Bybit REST downloader.

These tests hit the live Bybit PUBLIC endpoints (no API key required).
They verify that:
  - The HTTP client can connect and parse responses.
  - Every DataFrame method returns the correct column schema.
  - Pagination works (klines for a multi-page window returns > 1000 rows).
  - Empty windows return correctly-typed empty DataFrames.
  - The ticker and orderbook snapshots contain expected keys.
  - The cache round-trip (write -> read -> append) is lossless.

Run with::

    pytest tests/test_downloader.py -v

Skip with::

    pytest tests/test_downloader.py -v -m "not live"

The ``live`` marker is applied to every test in this file.
"""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from downloader        import HttpClient, ParquetCache, RestClient
from downloader.constants import (
    COLS_FUNDING,
    COLS_KLINE,
    COLS_LS,
    COLS_OI,
    COLS_TICK,
)
from core.config import Config, load_config

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.live  # pytest -m "not live" to skip all

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
SYMBOL   = "BTCUSDT"
CATEGORY = "linear"

# One week window (always available).
WINDOW_START = "2026-04-01"
WINDOW_END   = "2026-04-08"

# Short window that fits in a single page.
SMALL_START = "2026-04-25"
SMALL_END   = "2026-04-26"


@pytest.fixture(scope="module")
def cfg() -> Config:
    """Minimal config pointing at Bybit public REST."""
    return load_config()


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop so we do not recreate it per test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_df(df: pd.DataFrame, expected_cols: tuple[str, ...]) -> None:
    """Assert a DataFrame has the right shape and column schema."""
    assert isinstance(df, pd.DataFrame), "return value must be a DataFrame"
    assert list(df.columns) == list(expected_cols), (
        f"column mismatch: got {list(df.columns)}, expected {list(expected_cols)}"
    )
    if not df.empty:
        # Accept both ns and ms resolution UTC timestamps (pandas 2.x changed default)
        ts_dtype = str(df["timestamp"].dtype)
        assert "UTC" in ts_dtype and "datetime64" in ts_dtype, (
            f"timestamp must be UTC datetime64, got {ts_dtype}"
        )
        # Only numeric columns must be float64; string columns (e.g. 'symbol', 'side') are excluded
        float_cols = [
            c for c in df.columns
            if c != "timestamp" and df[c].dtype not in (object,) and not str(df[c].dtype).startswith("str")
        ]
        for c in float_cols:
            assert df[c].dtype == "float64", f"column {c!r} must be float64, got {df[c].dtype}"


# ===========================================================================
# OHLCV Klines
# ===========================================================================

class TestKlines:
    @pytest.mark.asyncio
    async def test_klines_schema(self, cfg: Config) -> None:
        """klines() returns correct columns and types."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.klines(SYMBOL, "60", SMALL_START, SMALL_END)
        _assert_df(df, COLS_KLINE)
        assert len(df) > 0, "expected at least one candle"

    @pytest.mark.asyncio
    async def test_klines_sorted(self, cfg: Config) -> None:
        """Timestamps must be strictly ascending (no duplicates)."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.klines(SYMBOL, "15", WINDOW_START, WINDOW_END)
        assert df["timestamp"].is_monotonic_increasing, "timestamps not sorted"
        assert not df["timestamp"].duplicated().any(), "duplicate timestamps found"

    @pytest.mark.asyncio
    async def test_klines_pagination(self, cfg: Config) -> None:
        """A multi-week window must return > 1000 rows (proves pagination works)."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            # 15-min candles over 2 weeks = ~1344 rows -> forces pagination.
            df = await rest.klines(SYMBOL, "15", "2026-04-01", "2026-04-15")
        assert len(df) > 1000, f"expected > 1000 rows, got {len(df)}"

    @pytest.mark.asyncio
    async def test_klines_empty_window(self, cfg: Config) -> None:
        """An inverted window must return an empty DataFrame with correct schema."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.klines(SYMBOL, "15", SMALL_END, SMALL_START)  # end < start
        _assert_df(df, COLS_KLINE)
        assert df.empty, "inverted window must be empty"

    @pytest.mark.asyncio
    async def test_mark_klines_schema(self, cfg: Config) -> None:
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.mark_klines(SYMBOL, "60", SMALL_START, SMALL_END)
        from downloader.constants import COLS_PRICE_KLINE
        _assert_df(df, COLS_PRICE_KLINE)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_index_klines_schema(self, cfg: Config) -> None:
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.index_klines(SYMBOL, "60", SMALL_START, SMALL_END)
        from downloader.constants import COLS_PRICE_KLINE
        _assert_df(df, COLS_PRICE_KLINE)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_premium_klines_schema(self, cfg: Config) -> None:
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.premium_klines(SYMBOL, "60", SMALL_START, SMALL_END)
        from downloader.constants import COLS_PRICE_KLINE
        _assert_df(df, COLS_PRICE_KLINE)
        assert len(df) > 0


# ===========================================================================
# Funding rates
# ===========================================================================

class TestFunding:
    @pytest.mark.asyncio
    async def test_funding_schema(self, cfg: Config) -> None:
        """funding() returns correct columns and types."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.funding(SYMBOL, WINDOW_START, WINDOW_END)
        _assert_df(df, COLS_FUNDING)
        assert len(df) > 0, "expected at least one funding settlement"

    @pytest.mark.asyncio
    async def test_funding_values_plausible(self, cfg: Config) -> None:
        """Funding rates should be small (typically -0.01 to +0.01)."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.funding(SYMBOL, WINDOW_START, WINDOW_END)
        assert df["funding_rate"].abs().max() < 0.1, "implausibly large funding rate"


# ===========================================================================
# Open interest
# ===========================================================================

class TestOpenInterest:
    @pytest.mark.asyncio
    async def test_oi_schema(self, cfg: Config) -> None:
        """open_interest() returns correct columns and types."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.open_interest(SYMBOL, "1h", WINDOW_START, WINDOW_END)
        _assert_df(df, COLS_OI)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_oi_positive(self, cfg: Config) -> None:
        """Open interest must be positive."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.open_interest(SYMBOL, "1h", WINDOW_START, WINDOW_END)
        assert (df["open_interest"] > 0).all(), "open interest contains non-positive values"


# ===========================================================================
# Long/Short ratio
# ===========================================================================

class TestLongShortRatio:
    @pytest.mark.asyncio
    async def test_ls_schema(self, cfg: Config) -> None:
        """long_short_ratio() returns correct columns and types."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.long_short_ratio(SYMBOL, "1h", limit=100)
        _assert_df(df, COLS_LS)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_ls_ratios_sum_to_one(self, cfg: Config) -> None:
        """buy_ratio + sell_ratio should be approximately 1.0."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.long_short_ratio(SYMBOL, "1h", limit=50)
        total = (df["buy_ratio"] + df["sell_ratio"])
        assert total.between(0.98, 1.02).all(), "buy_ratio + sell_ratio not near 1.0"


# ===========================================================================
# Market snapshots
# ===========================================================================

class TestMarketSnapshots:
    @pytest.mark.asyncio
    async def test_tickers_returns_list(self, cfg: Config) -> None:
        """tickers() must return a non-empty list of dicts."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            tickers = await rest.tickers()
        assert isinstance(tickers, list), "tickers() must return a list"
        assert len(tickers) > 50, "expected at least 50 instruments"
        t = tickers[0]
        assert "symbol"      in t
        assert "price"       in t
        assert "turnover_24h" in t

    @pytest.mark.asyncio
    async def test_ticker_single(self, cfg: Config) -> None:
        """ticker() returns a single normalized dict with expected keys."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            t = await rest.ticker(SYMBOL)
        assert t["symbol"]  == SYMBOL
        assert t["price"]   > 0
        assert t["bid"]     > 0
        assert t["ask"]     > 0
        assert t["ask"]     >= t["bid"]

    @pytest.mark.asyncio
    async def test_ticker_missing_symbol(self, cfg: Config) -> None:
        """ticker() with an invalid symbol raises BybitApiError or returns empty/no-price dict."""
        from downloader.errors import BybitApiError
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            try:
                t = await rest.ticker("THISDOESNOTEXISTUSDT")
                # If it doesn't raise, it should return a dict with no positive price
                assert isinstance(t, dict)
                assert t.get("price", 0) == 0 or t == {}
            except BybitApiError:
                pass  # Expected: Bybit returns params error for invalid symbols

    @pytest.mark.asyncio
    async def test_instruments_schema(self, cfg: Config) -> None:
        """instruments() returns a DataFrame with expected columns."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.instruments()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 50
        for col in ("symbol", "status", "quoteCoin"):
            assert col in df.columns, f"missing column {col!r}"


# ===========================================================================
# Orderbook
# ===========================================================================

class TestOrderbook:
    @pytest.mark.asyncio
    async def test_orderbook_structure(self, cfg: Config) -> None:
        """orderbook() returns correct keys and non-empty bid/ask lists."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            ob = await rest.orderbook(SYMBOL, depth=10)
        assert ob["symbol"]    == SYMBOL
        assert ob["ts_ms"]     > 0
        assert len(ob["bids"]) > 0
        assert len(ob["asks"]) > 0

    @pytest.mark.asyncio
    async def test_orderbook_best_spread(self, cfg: Config) -> None:
        """Best ask must be >= best bid (no crossed book)."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            ob = await rest.orderbook(SYMBOL, depth=5)
        best_bid = ob["bids"][0][0]
        best_ask = ob["asks"][0][0]
        assert best_ask >= best_bid, f"crossed book: bid={best_bid} ask={best_ask}"


# ===========================================================================
# Recent trades
# ===========================================================================

class TestRecentTrades:
    @pytest.mark.asyncio
    async def test_recent_trades_schema(self, cfg: Config) -> None:
        """recent_trades() returns correct columns."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.recent_trades(SYMBOL, limit=50)
        _assert_df(df, COLS_TICK)
        assert len(df) > 0

    @pytest.mark.asyncio
    async def test_recent_trades_side_values(self, cfg: Config) -> None:
        """Trade sides must be 'Buy' or 'Sell'."""
        async with HttpClient(cfg) as http:
            rest = RestClient(http, cfg)
            df = await rest.recent_trades(SYMBOL, limit=100)
        assert df["side"].isin(["Buy", "Sell"]).all(), "unexpected side values"


# ===========================================================================
# Parquet cache
# ===========================================================================

class TestParquetCache:
    def test_write_read_roundtrip(self, cfg: Config) -> None:
        """Write a DataFrame and read it back -- data must be identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(Path(tmpdir))
            df = pd.DataFrame({
                "timestamp": pd.date_range("2026-04-01", periods=5, freq="1h", tz="UTC"),
                "open":  [1.0, 2.0, 3.0, 4.0, 5.0],
                "close": [1.1, 2.1, 3.1, 4.1, 5.1],
            })
            cache.write(df, kind="klines", symbol="TEST", subkey="60")
            result = cache.read(kind="klines", symbol="TEST", subkey="60")
            assert result is not None
            pd.testing.assert_frame_equal(df, result)

    def test_append_deduplicates(self, cfg: Config) -> None:
        """Appending overlapping rows must not create duplicates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(Path(tmpdir))
            ts = pd.date_range("2026-04-01", periods=4, freq="1h", tz="UTC")
            df1 = pd.DataFrame({"timestamp": ts[:3], "value": [1.0, 2.0, 3.0]})
            df2 = pd.DataFrame({"timestamp": ts[2:], "value": [99.0, 4.0]})  # row 2 overlaps

            cache.write(df1, "data", "TEST")
            merged = cache.append(df2, "data", "TEST")

            assert len(merged) == 4, f"expected 4 rows, got {len(merged)}"
            # The overlapping row should use the newer value (99.0).
            assert merged.loc[merged["timestamp"] == ts[2], "value"].iloc[0] == 99.0

    def test_missing_file_returns_none(self, cfg: Config) -> None:
        """Reading a non-existent cache entry returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(Path(tmpdir))
            assert cache.read("klines", "NOBODY", "15") is None

    def test_last_timestamp(self, cfg: Config) -> None:
        """last_timestamp() returns the max timestamp in the cached file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(Path(tmpdir))
            ts = pd.date_range("2026-04-01", periods=3, freq="1h", tz="UTC")
            df = pd.DataFrame({"timestamp": ts, "value": [1.0, 2.0, 3.0]})
            cache.write(df, "data", "SYM")
            last = cache.last_timestamp("data", "SYM")
            assert last == ts[-1]

    def test_inventory(self, cfg: Config) -> None:
        """inventory() lists all parquet files in the cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ParquetCache(Path(tmpdir))
            df = pd.DataFrame({"timestamp": pd.date_range("2026-04-01", periods=2, freq="1h", tz="UTC"), "v": [1.0, 2.0]})
            cache.write(df, "klines", "AAVE", "15")
            cache.write(df, "funding", "AAVE")
            inv = cache.inventory()
            assert len(inv) == 2
            assert set(inv["kind"].tolist()) == {"klines", "funding"}
