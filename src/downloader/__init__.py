"""Downloader package: public API surface.

Nothing here performs I/O.  Import from sub-modules for full control,
or import from here for convenience.

Typical usage
-------------
::

    from downloader import HttpClient, RestClient, ParquetCache
    from core import load_config

    cfg = load_config()

    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)

        # OHLCV candles
        df = await rest.klines("BTCUSDT", "15", "2026-04-01", "2026-04-26")

        # Funding history
        fr = await rest.funding("BTCUSDT", "2026-04-01", "2026-04-26")

        # Market snapshot
        tickers = await rest.tickers()

    # Cache the result
    cache = ParquetCache(cfg.cache_root)
    cache.write(df, kind="klines", symbol="BTCUSDT", subkey="15")

What you can download
---------------------
Via RestClient (REST, real-time or historical):
  * klines()          -- OHLCV trade-price candles
  * mark_klines()     -- Mark-price candles
  * index_klines()    -- Index-price candles
  * premium_klines()  -- Premium-index (basis proxy) candles
  * funding()         -- Historical funding rates (8-h settlements)
  * open_interest()   -- Historical open interest snapshots
  * long_short_ratio()-- Aggregate account L/S ratio (recent only)
  * tickers()         -- Full market snapshot (all symbols)
  * ticker()          -- Single-symbol snapshot
  * instruments()     -- Contract specs (tick size, qty step, status ...)
  * orderbook()       -- Level-2 orderbook snapshot
  * recent_trades()   -- Most recent public trades
"""
from __future__ import annotations

from downloader.cache    import ParquetCache
from downloader.errors   import BybitApiError, CacheError, HttpError
from downloader.http     import HttpClient
from downloader.rest     import RestClient

__all__ = [
    "HttpClient",
    "RestClient",
    "ParquetCache",
    "HttpError",
    "BybitApiError",
    "CacheError",
]
