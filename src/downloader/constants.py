"""Bybit v5 API constants: endpoints, limits, intervals, column schemas.

All magic strings and numbers used by the downloader live here.
No other file in src/downloader/ should hard-code an endpoint path,
a page-size cap, or a column name.
"""
from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Base URLs
# ---------------------------------------------------------------------------
BYBIT_REST_BASE: Final[str]    = "https://api.bybit.com"
BYBIT_ARCHIVE_BASE: Final[str] = "https://public.bybit.com/trading"

# ---------------------------------------------------------------------------
# REST endpoint paths (all relative to BYBIT_REST_BASE)
# ---------------------------------------------------------------------------
PATH_KLINE:         Final[str] = "/v5/market/kline"
PATH_MARK_KLINE:    Final[str] = "/v5/market/mark-price-kline"
PATH_INDEX_KLINE:   Final[str] = "/v5/market/index-price-kline"
PATH_PREMIUM_KLINE: Final[str] = "/v5/market/premium-index-price-kline"
PATH_FUNDING:       Final[str] = "/v5/market/funding/history"
PATH_OI:            Final[str] = "/v5/market/open-interest"
PATH_TICKERS:       Final[str] = "/v5/market/tickers"
PATH_INSTRUMENTS:   Final[str] = "/v5/market/instruments-info"
PATH_ORDERBOOK:     Final[str] = "/v5/market/orderbook"
PATH_RECENT_TRADES: Final[str] = "/v5/market/recent-trade"
PATH_LS_RATIO:      Final[str] = "/v5/market/account-ratio"

# ---------------------------------------------------------------------------
# Pagination caps (Bybit v5 public hard limits)
# ---------------------------------------------------------------------------
LIMIT_KLINE:         Final[int] = 1000
LIMIT_FUNDING:       Final[int] = 200
LIMIT_OI:            Final[int] = 200
LIMIT_LS:            Final[int] = 500
LIMIT_RECENT_TRADES: Final[int] = 1000
LIMIT_ORDERBOOK:     Final[int] = 500

# ---------------------------------------------------------------------------
# Valid interval strings
# ---------------------------------------------------------------------------
KLINE_INTERVALS: Final[tuple[str, ...]] = (
    "1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M",
)
OI_INTERVALS: Final[tuple[str, ...]] = (
    "5min", "15min", "30min", "1h", "4h", "1d",
)
LS_INTERVALS: Final[tuple[str, ...]] = (
    "5min", "15min", "30min", "1h", "4h", "4d",
)

# ---------------------------------------------------------------------------
# Canonical output column schemas
# Each method in RestClient returns a DataFrame with exactly these columns.
# ---------------------------------------------------------------------------
COLS_KLINE:       Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close", "volume", "turnover")
COLS_PRICE_KLINE: Final[tuple[str, ...]] = ("timestamp", "open", "high", "low", "close")
COLS_FUNDING:     Final[tuple[str, ...]] = ("timestamp", "symbol", "funding_rate")
COLS_OI:          Final[tuple[str, ...]] = ("timestamp", "open_interest")
COLS_LS:          Final[tuple[str, ...]] = ("timestamp", "buy_ratio", "sell_ratio")
COLS_TICK:        Final[tuple[str, ...]] = ("timestamp", "symbol", "side", "size", "price", "trade_id")

# ---------------------------------------------------------------------------
# WebSocket URLs
# ---------------------------------------------------------------------------
WS_LINEAR:  Final[str] = "wss://stream.bybit.com/v5/public/linear"
WS_SPOT:    Final[str] = "wss://stream.bybit.com/v5/public/spot"
WS_INVERSE: Final[str] = "wss://stream.bybit.com/v5/public/inverse"

WS_PING_INTERVAL_SEC:  Final[float] = 20.0
WS_RECONNECT_BASE_SEC: Final[float] = 1.0
WS_RECONNECT_MAX_SEC:  Final[float] = 30.0

# ---------------------------------------------------------------------------
# Cache constants
# ---------------------------------------------------------------------------
MIN_PARQUET_BYTES: Final[int] = 256  # files smaller than this are corrupt
