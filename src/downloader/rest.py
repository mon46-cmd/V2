"""Bybit v5 public REST client.

Every method returns a ``pandas.DataFrame`` with:
- UTC-tz-aware ``timestamp`` column.
- ``float64`` numeric columns.
- Canonical column set (defined in ``constants.py``).
- Duplicates already dropped, sorted ascending by ``timestamp``.

Pagination is handled internally: pass a time window, receive every row.
Empty windows return an empty DataFrame with the correct schema -- never
a shape surprise for callers.

Usage::

    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from core import load_config

    cfg = load_config()
    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)

        # OHLCV candles
        df = await rest.klines("BTCUSDT", "15", "2026-04-01", "2026-04-26")

        # Single ticker
        t  = await rest.ticker("ETHUSDT")

        # Full market snapshot
        tickers     = await rest.tickers()
        instruments = await rest.instruments()
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from core.config import Config
from downloader.constants import (
    COLS_FUNDING,
    COLS_KLINE,
    COLS_LS,
    COLS_OI,
    COLS_PRICE_KLINE,
    COLS_TICK,
    LIMIT_FUNDING,
    LIMIT_KLINE,
    LIMIT_LS,
    LIMIT_OI,
    LIMIT_ORDERBOOK,
    LIMIT_RECENT_TRADES,
    PATH_FUNDING,
    PATH_INDEX_KLINE,
    PATH_INSTRUMENTS,
    PATH_KLINE,
    PATH_LS_RATIO,
    PATH_MARK_KLINE,
    PATH_OI,
    PATH_ORDERBOOK,
    PATH_PREMIUM_KLINE,
    PATH_RECENT_TRADES,
    PATH_TICKERS,
)
from downloader.http import HttpClient

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ms(ts: "str | pd.Timestamp | int | float") -> int:
    """Convert any timestamp type to Bybit milliseconds since epoch."""
    if isinstance(ts, (int, float)):
        return int(ts)
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp() * 1000)


def _float(v: Any) -> float:
    """Parse a value to float64, returning NaN on failure."""
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _empty_df(cols: tuple[str, ...], *, float_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema."""
    data: dict[str, pd.Series] = {}
    for c in cols:
        if c == "timestamp":
            data[c] = pd.Series(dtype="datetime64[ns, UTC]")
        elif c in float_cols:
            data[c] = pd.Series(dtype="float64")
        else:
            data[c] = pd.Series(dtype="object")
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# RestClient
# ---------------------------------------------------------------------------

class RestClient:
    """Thin normalizing wrapper over the Bybit v5 public REST API.

    Args:
        http: Open ``HttpClient`` instance (from ``async with HttpClient(cfg)``).
        cfg:  Loaded Config (provides ``category``, ``quote_currency``, etc.).
    """

    def __init__(self, http: HttpClient, cfg: Config) -> None:
        self._http     = http
        self._category = cfg.category

    # ------------------------------------------------------------------ klines

    async def klines(
        self,
        symbol:   str,
        interval: str,
        start:    "str | pd.Timestamp | int",
        end:      "str | pd.Timestamp | int",
    ) -> pd.DataFrame:
        """OHLCV candles (trade price).

        Returns columns: ``timestamp, open, high, low, close, volume, turnover``.

        Args:
            symbol:   e.g. ``"BTCUSDT"``.
            interval: ``"1"``, ``"5"``, ``"15"``, ``"60"``, ``"240"``, ``"D"`` ...
                      (see ``KLINE_INTERVALS`` in constants.py).
            start:    Start of the window (inclusive). Any format accepted by
                      ``pd.Timestamp``, or unix ms integer.
            end:      End of the window (exclusive).
        """
        return await self._klines(PATH_KLINE, symbol, interval, start, end, price_only=False)

    async def mark_klines(
        self,
        symbol:   str,
        interval: str,
        start:    "str | pd.Timestamp | int",
        end:      "str | pd.Timestamp | int",
    ) -> pd.DataFrame:
        """Mark-price candles.  Returns ``timestamp, open, high, low, close``."""
        return await self._klines(PATH_MARK_KLINE, symbol, interval, start, end, price_only=True)

    async def index_klines(
        self,
        symbol:   str,
        interval: str,
        start:    "str | pd.Timestamp | int",
        end:      "str | pd.Timestamp | int",
    ) -> pd.DataFrame:
        """Index-price candles.  Returns ``timestamp, open, high, low, close``."""
        return await self._klines(PATH_INDEX_KLINE, symbol, interval, start, end, price_only=True)

    async def premium_klines(
        self,
        symbol:   str,
        interval: str,
        start:    "str | pd.Timestamp | int",
        end:      "str | pd.Timestamp | int",
    ) -> pd.DataFrame:
        """Premium-index candles (basis proxy).  Returns ``timestamp, open, high, low, close``."""
        return await self._klines(PATH_PREMIUM_KLINE, symbol, interval, start, end, price_only=True)

    async def _klines(
        self,
        path:       str,
        symbol:     str,
        interval:   str,
        start:      "str | pd.Timestamp | int",
        end:        "str | pd.Timestamp | int",
        *,
        price_only: bool,
    ) -> pd.DataFrame:
        start_ms, end_ms = _ms(start), _ms(end)
        if end_ms <= start_ms:
            return _empty_df(
                COLS_PRICE_KLINE if price_only else COLS_KLINE,
                float_cols=tuple(c for c in (COLS_PRICE_KLINE if price_only else COLS_KLINE) if c != "timestamp"),
            )

        rows: list[list[Any]] = []
        cur_end = end_ms

        while cur_end > start_ms:
            result = await self._http.get_json(path, {
                "category": self._category,
                "symbol":   symbol,
                "interval": interval,
                "start":    start_ms,
                "end":      cur_end,
                "limit":    LIMIT_KLINE,
            })
            got = result.get("list") or []
            if not got:
                break
            rows.extend(got)
            oldest = int(got[-1][0])
            if len(got) < LIMIT_KLINE or oldest <= start_ms:
                break
            cur_end = oldest - 1

        if not rows:
            return _empty_df(
                COLS_PRICE_KLINE if price_only else COLS_KLINE,
                float_cols=tuple(c for c in (COLS_PRICE_KLINE if price_only else COLS_KLINE) if c != "timestamp"),
            )

        cols = COLS_PRICE_KLINE if price_only else COLS_KLINE
        trimmed = [r[:len(cols)] for r in rows]
        df = pd.DataFrame(trimmed, columns=list(cols))
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        for c in df.columns:
            if c != "timestamp":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        return df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)

    # ------------------------------------------------------------------ funding

    async def funding(
        self,
        symbol: str,
        start:  "str | pd.Timestamp | int",
        end:    "str | pd.Timestamp | int",
    ) -> pd.DataFrame:
        """Historical funding rates.

        Returns columns: ``timestamp, symbol, funding_rate``.

        Note: Bybit only stores funding history back ~6 months.
        """
        start_ms, end_ms = _ms(start), _ms(end)
        out: list[dict[str, Any]] = []
        cur_end = end_ms

        while True:
            result = await self._http.get_json(PATH_FUNDING, {
                "category":  self._category,
                "symbol":    symbol,
                "startTime": start_ms,
                "endTime":   cur_end,
                "limit":     LIMIT_FUNDING,
            })
            got = result.get("list") or []
            if not got:
                break
            out.extend(got)
            oldest = min(int(r["fundingRateTimestamp"]) for r in got)
            if len(got) < LIMIT_FUNDING or oldest <= start_ms:
                break
            cur_end = oldest - 1

        if not out:
            return _empty_df(COLS_FUNDING, float_cols=("funding_rate",))

        df = pd.DataFrame(out)
        df["timestamp"]    = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
        df["funding_rate"] = pd.to_numeric(df["fundingRate"], errors="coerce").astype("float64")
        df["symbol"]       = df["symbol"].astype(str)
        return (
            df[list(COLS_FUNDING)]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ open interest

    async def open_interest(
        self,
        symbol:   str,
        interval: str,
        start:    "str | pd.Timestamp | int",
        end:      "str | pd.Timestamp | int",
    ) -> pd.DataFrame:
        """Historical open interest.

        Returns columns: ``timestamp, open_interest``.

        Args:
            interval: One of ``"5min"``, ``"15min"``, ``"30min"``,
                      ``"1h"``, ``"4h"``, ``"1d"``.
        """
        start_ms, end_ms = _ms(start), _ms(end)
        out:    list[dict[str, Any]] = []
        cursor: str = ""

        while True:
            params: dict[str, Any] = {
                "category":     self._category,
                "symbol":       symbol,
                "intervalTime": interval,
                "startTime":    start_ms,
                "endTime":      end_ms,
                "limit":        LIMIT_OI,
            }
            if cursor:
                params["cursor"] = cursor
            result = await self._http.get_json(PATH_OI, params)
            got    = result.get("list") or []
            if not got:
                break
            out.extend(got)
            cursor = result.get("nextPageCursor") or ""
            if not cursor or len(got) < LIMIT_OI:
                break

        if not out:
            return _empty_df(COLS_OI, float_cols=("open_interest",))

        df = pd.DataFrame(out)
        df["timestamp"]     = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["open_interest"] = pd.to_numeric(df["openInterest"], errors="coerce").astype("float64")
        return (
            df[list(COLS_OI)]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ long/short ratio

    async def long_short_ratio(
        self,
        symbol:   str,
        interval: str,
        *,
        limit: int = LIMIT_LS,
    ) -> pd.DataFrame:
        """Aggregate account long/short ratio (most recent N points).

        Returns columns: ``timestamp, buy_ratio, sell_ratio``.

        Note: Bybit does not support start/end for this endpoint; only
        the most recent ``limit`` data points are available.

        Args:
            interval: One of ``"5min"``, ``"15min"``, ``"30min"``,
                      ``"1h"``, ``"4h"``, ``"4d"``.
            limit:    Number of records to return (max 500).
        """
        result = await self._http.get_json(PATH_LS_RATIO, {
            "category": self._category,
            "symbol":   symbol,
            "period":   interval,
            "limit":    min(limit, LIMIT_LS),
        })
        got = result.get("list") or []
        if not got:
            return _empty_df(COLS_LS, float_cols=("buy_ratio", "sell_ratio"))

        df = pd.DataFrame(got)
        df["timestamp"]  = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["buy_ratio"]  = pd.to_numeric(df["buyRatio"],  errors="coerce").astype("float64")
        df["sell_ratio"] = pd.to_numeric(df["sellRatio"], errors="coerce").astype("float64")
        return (
            df[list(COLS_LS)]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------ market snapshots

    async def tickers(self) -> list[dict[str, Any]]:
        """Full market snapshot for every instrument in the category.

        Returns a list of normalized ticker dicts (see ``_norm_ticker``).
        """
        result = await self._http.get_json(PATH_TICKERS, {"category": self._category})
        return [_norm_ticker(t) for t in (result.get("list") or [])]

    async def ticker(self, symbol: str) -> dict[str, Any]:
        """Snapshot for a single symbol.  Returns ``{}`` if not found."""
        result = await self._http.get_json(
            PATH_TICKERS, {"category": self._category, "symbol": symbol},
        )
        lst = result.get("list") or []
        return _norm_ticker(lst[0]) if lst else {}

    async def instruments(self) -> pd.DataFrame:
        """Contract specs for all instruments (paginated).

        Columns include: ``symbol``, ``status``, ``quoteCoin``,
        ``contractType``, ``launchTime``, ``priceFilter.tickSize``,
        ``lotSizeFilter.minOrderQty``, etc.
        """
        out:    list[dict[str, Any]] = []
        cursor: str = ""

        while True:
            params: dict[str, Any] = {"category": self._category, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            result = await self._http.get_json(PATH_INSTRUMENTS, params)
            got    = result.get("list") or []
            if not got:
                break
            out.extend(got)
            cursor = result.get("nextPageCursor") or ""
            if not cursor:
                break

        if not out:
            return pd.DataFrame()

        df = pd.json_normalize(out)
        if "launchTime" in df.columns:
            df["launchTime"] = pd.to_datetime(
                pd.to_numeric(df["launchTime"], errors="coerce"), unit="ms", utc=True,
            )
        return df

    # ------------------------------------------------------------------ orderbook

    async def orderbook(self, symbol: str, *, depth: int = 50) -> dict[str, Any]:
        """Level-2 orderbook snapshot.

        Returns a dict with keys:
        ``symbol``, ``ts_ms``, ``update_id``, ``bids``, ``asks``
        where ``bids``/``asks`` are lists of ``(price, qty)`` float tuples,
        best price first.

        Args:
            symbol: e.g. ``"BTCUSDT"``.
            depth:  Number of price levels per side (max 500).
        """
        result = await self._http.get_json(PATH_ORDERBOOK, {
            "category": self._category,
            "symbol":   symbol,
            "limit":    min(depth, LIMIT_ORDERBOOK),
        })
        return {
            "symbol":    str(result.get("s", symbol)),
            "ts_ms":     int(result.get("ts", 0)),
            "update_id": int(result.get("u", 0)),
            "bids":      [(float(p), float(q)) for p, q in (result.get("b") or [])],
            "asks":      [(float(p), float(q)) for p, q in (result.get("a") or [])],
        }

    # ------------------------------------------------------------------ recent trades

    async def recent_trades(self, symbol: str, *, limit: int = 200) -> pd.DataFrame:
        """Most recent public trades (up to 1 000).

        Returns columns: ``timestamp, symbol, side, size, price, trade_id``.

        Args:
            symbol: e.g. ``"BTCUSDT"``.
            limit:  Number of trades to return (max 1 000).
        """
        result = await self._http.get_json(PATH_RECENT_TRADES, {
            "category": self._category,
            "symbol":   symbol,
            "limit":    min(limit, LIMIT_RECENT_TRADES),
        })
        got = result.get("list") or []
        if not got:
            return _empty_df(COLS_TICK, float_cols=("size", "price"))

        df = pd.DataFrame(got)
        df["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        df["symbol"]    = df["symbol"].astype(str) if "symbol" in df.columns else symbol
        df["side"]      = df["side"].astype(str)
        df["size"]      = pd.to_numeric(df["size"],  errors="coerce").astype("float64")
        df["price"]     = pd.to_numeric(df["price"], errors="coerce").astype("float64")
        df["trade_id"]  = df["execId"].astype(str) if "execId" in df.columns else ""
        return (
            df[list(COLS_TICK)]
            .drop_duplicates("trade_id")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Internal normalization helpers
# ---------------------------------------------------------------------------

def _norm_ticker(t: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw Bybit ticker dict to consistent field names."""
    return {
        "symbol":              str(t.get("symbol", "")),
        "price":               _float(t.get("lastPrice")),
        "bid":                 _float(t.get("bid1Price")),
        "ask":                 _float(t.get("ask1Price")),
        "mark_price":          _float(t.get("markPrice")),
        "index_price":         _float(t.get("indexPrice")),
        "volume_24h":          _float(t.get("volume24h")),
        "turnover_24h":        _float(t.get("turnover24h")),
        "open_interest":       _float(t.get("openInterest")),
        "open_interest_value": _float(t.get("openInterestValue")),
        "funding_rate":        _float(t.get("fundingRate")),
        "next_funding_ms":     int(t.get("nextFundingTime", 0) or 0),
        "price_change_24h_pct": _float(t.get("price24hPcnt")),
        "high_24h":            _float(t.get("highPrice24h")),
        "low_24h":             _float(t.get("lowPrice24h")),
    }
