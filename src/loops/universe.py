"""Universe builder -- fetch and filter the tradeable Bybit perpetual universe.

Responsibilities:
- Fetch all linear-USDT tickers from Bybit REST.
- Apply hard filters (min volume, min price, exclusion lists, spread).
- Apply soft ranking (sort by 24h turnover descending).
- Return a clean DataFrame with one row per tradeable symbol.
- Cache result to disk for ``cache_ttl_sec`` to avoid hammering Bybit.

The universe is intentionally large-ish (~50-100 symbols) because it is the
raw pool.  The social scan then narrows this to 5 watchlist symbols.

Usage::

    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from core import load_config
    from loops.universe import UniverseBuilder

    cfg = load_config()
    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)
        builder = UniverseBuilder(cfg, rest)
        symbols = await builder.get()    # returns list[str], cached
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import Config
from downloader.rest import RestClient

log = logging.getLogger(__name__)

# Stablecoins and pegged assets that should never be traded.
_DEFAULT_EXCLUDE: frozenset[str] = frozenset({
    "USDCUSDT", "USDEUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT",
    "BUSDUSDT", "FRAXUSDT", "USDTUSDT", "PYUSDUSDT", "EURUSDT",
    "GBPUSDT", "JPYUSDT",
})

# Symbols containing these substrings are excluded (leveraged tokens, etc.).
_EXCLUDE_SUBSTRINGS: tuple[str, ...] = ("-", "UP", "DOWN", "BEAR", "BULL")


class UniverseBuilder:
    """Fetch, filter, and cache the tradeable symbol universe.

    Args:
        cfg:           Loaded Config.
        rest:          Open RestClient.
        cache_ttl_sec: Seconds before re-fetching from Bybit. Default 3600 (1h).
        cache_path:    Optional path to persist cache between process restarts.
                       Defaults to ``cfg.data_root / "universe.json"``.
    """

    def __init__(
        self,
        cfg:           Config,
        rest:          RestClient,
        *,
        cache_ttl_sec: int        = 3_600,
        cache_path:    Path | None = None,
    ) -> None:
        self._cfg          = cfg
        self._rest         = rest
        self._ttl          = cache_ttl_sec
        self._cache_path   = cache_path or (cfg.data_root / "universe.json")
        self._cached_at    = 0.0
        self._cached_syms: list[str] = []
        self._cached_df:   pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def get(self, *, force: bool = False) -> list[str]:
        """Return the filtered symbol list, using cache if fresh.

        Args:
            force: If True, bypass TTL and re-fetch unconditionally.

        Returns:
            Ordered list of symbol strings (e.g. ``["BTCUSDT", "ETHUSDT", ...]``).
        """
        if not force and self._is_cache_fresh():
            return list(self._cached_syms)

        # Try loading from disk first (survives process restarts).
        if not force and self._load_disk_cache():
            return list(self._cached_syms)

        return await self._fetch_and_filter()

    async def get_df(self, *, force: bool = False) -> pd.DataFrame:
        """Return the full filtered ticker DataFrame (includes volume, price, etc.)."""
        await self.get(force=force)
        return self._cached_df if self._cached_df is not None else pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_cache_fresh(self) -> bool:
        return (
            bool(self._cached_syms)
            and (time.monotonic() - self._cached_at) < self._ttl
        )

    def _load_disk_cache(self) -> bool:
        """Load persisted cache. Returns True if valid and fresh."""
        if not self._cache_path.exists():
            return False
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            saved_at = float(data.get("saved_at", 0))
            if (time.time() - saved_at) > self._ttl:
                return False
            syms = [str(s) for s in data.get("symbols", [])]
            if not syms:
                return False
            self._cached_syms = syms
            self._cached_at   = time.monotonic()
            log.info("universe loaded from disk n=%d", len(syms))
            return True
        except Exception:  # noqa: BLE001
            return False

    def _save_disk_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"saved_at": time.time(), "symbols": self._cached_syms}
            tmp = self._cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            import os; os.replace(tmp, self._cache_path)  # noqa: E702
        except OSError as exc:
            log.warning("universe disk cache save failed: %s", exc)

    async def _fetch_and_filter(self) -> list[str]:
        log.info("universe fetch from Bybit...")
        try:
            raw: list[dict[str, Any]] = await self._rest.tickers()
        except Exception as exc:  # noqa: BLE001
            log.error("universe fetch failed: %s", exc)
            return list(self._cached_syms)  # return stale cache if available

        df = pd.DataFrame(raw)
        if df.empty:
            log.warning("universe: empty ticker response")
            return []

        df = self._filter(df)
        df = df.sort_values("turnover_24h", ascending=False).reset_index(drop=True)

        self._cached_df   = df
        self._cached_syms = df["symbol"].tolist()
        self._cached_at   = time.monotonic()
        self._save_disk_cache()

        log.info(
            "universe filtered: %d symbols (from %d total)",
            len(self._cached_syms), len(raw),
        )
        return list(self._cached_syms)

    def _filter(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self._cfg
        original = len(df)

        # --- USDT quote only ---
        if "symbol" in df.columns:
            df = df[df["symbol"].str.endswith("USDT", na=False)]

        # --- Hard exclusions ---
        exclude = _DEFAULT_EXCLUDE | set(cfg.exclude_symbols)
        df = df[~df["symbol"].isin(exclude)]

        # --- Substring exclusions ---
        excl_subs = _EXCLUDE_SUBSTRINGS + tuple(cfg.exclude_substrings)
        for sub in excl_subs:
            df = df[~df["symbol"].str.contains(sub, na=False, regex=False)]

        # --- Minimum price (avoid micro-cap dust) ---
        if "price" in df.columns and cfg.min_price_usd > 0:
            df = df[df["price"].fillna(0) >= cfg.min_price_usd]

        # --- Minimum 24h turnover (liquidity gate) ---
        if "turnover_24h" in df.columns and cfg.min_turnover_usd_24h > 0:
            df = df[df["turnover_24h"].fillna(0) >= cfg.min_turnover_usd_24h]

        # --- Maximum bid-ask spread ---
        if all(c in df.columns for c in ("bid", "ask", "price")) and cfg.max_spread_bps > 0:
            mid    = df["price"].replace(0, float("nan"))
            spread = ((df["ask"] - df["bid"]) / mid * 10_000).fillna(9999)
            df     = df[spread <= cfg.max_spread_bps]

        # --- Universe size cap ---
        if cfg.universe_size > 0 and len(df) > cfg.universe_size:
            df = df.head(cfg.universe_size)

        log.debug("universe filter: %d -> %d symbols", original, len(df))
        return df.reset_index(drop=True)
