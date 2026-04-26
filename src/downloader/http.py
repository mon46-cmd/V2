"""Async HTTP client with rate limiting and automatic retry.

One ``HttpClient`` is opened per process via ``async with`` and shared
between ``RestClient`` and ``ArchiveClient`` so they all respect the
same global rate limit and reuse the same connection pool.

Design rules:
- ``get_json()`` unwraps Bybit's ``{retCode, retMsg, result}`` envelope
  and raises ``BybitApiError`` on retCode != 0.
- ``get_bytes()`` fetches raw binary (archive CSV.gz on a different host).
- 429 / 5xx responses are retried with jittered back-off automatically.
- On failure a structured ``HttpError`` is raised so callers can branch
  on ``exc.status`` without parsing strings.

Usage::

    from downloader.http import HttpClient
    from core import load_config

    cfg = load_config()
    async with HttpClient(cfg) as http:
        data = await http.get_json("/v5/market/tickers", {"category": "linear"})
"""
from __future__ import annotations

import asyncio
import logging
import random
from types import TracebackType
from typing import Any

import aiohttp

from core.config import Config
from downloader.errors import BybitApiError, HttpError

log = logging.getLogger(__name__)

# HTTP status codes that are safe to retry.
_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})


class _RateLimiter:
    """Token-bucket: enforces a minimum delay between successive requests."""

    def __init__(self, min_delay_sec: float) -> None:
        self._min_delay = max(0.0, min_delay_sec)
        self._last      = 0.0
        self._lock      = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            loop      = asyncio.get_event_loop()
            sleep_for = self._min_delay - (loop.time() - self._last)
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._last = loop.time()


class HttpClient:
    """Async HTTP client for Bybit REST + archive.

    Always use as ``async with HttpClient(cfg) as http: ...``.

    Args:
        cfg:      Loaded Config object (provides timeouts, retry counts, etc.).
        base_url: Override the REST base URL (default: ``cfg.bybit_rest_base``).
    """

    def __init__(self, cfg: Config, *, base_url: str | None = None) -> None:
        self._base_url    = (base_url or cfg.bybit_rest_base).rstrip("/")
        self._timeout     = aiohttp.ClientTimeout(total=cfg.http_timeout_sec)
        self._max_retries = cfg.http_max_retries
        self._backoff     = cfg.http_backoff_base_sec
        self._rate        = _RateLimiter(cfg.http_rate_delay_sec)
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "HttpClient":
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            headers={"User-Agent": "bybit-downloader/1.0"},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc:      BaseException | None,
        tb:       TracebackType | None,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    async def get_json(
        self,
        path:   str,
        params: dict[str, Any] | None = None,
        *,
        envelope: bool = True,
    ) -> Any:
        """GET a path relative to the base URL and return parsed JSON.

        Args:
            path:     API path, e.g. ``"/v5/market/kline"``.
            params:   Query parameters.
            envelope: If True (default) unwrap Bybit's
                      ``{retCode, retMsg, result}`` wrapper and raise
                      ``BybitApiError`` when retCode != 0.
                      Set False for non-Bybit hosts.

        Returns:
            The ``result`` dict (envelope=True) or the full parsed JSON.
        """
        self._assert_open()
        url = f"{self._base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            await self._rate.wait()
            try:
                async with self._session.get(url, params=params) as resp:  # type: ignore[union-attr]
                    if resp.status in _RETRY_STATUSES:
                        last_exc = HttpError(resp.status, url, await resp.text())
                        if attempt < self._max_retries:
                            await self._sleep(attempt)
                            continue
                        raise last_exc
                    if resp.status != 200:
                        raise HttpError(resp.status, url, await resp.text())
                    data = await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    log.debug("transport error %s: %s (attempt %d)", path, exc, attempt + 1)
                    await self._sleep(attempt)
                    continue
                raise HttpError(0, url, str(exc)) from exc

            if not envelope:
                return data

            ret_code = int(data.get("retCode", -1))
            if ret_code != 0:
                raise BybitApiError(ret_code, str(data.get("retMsg", "")), url)
            return data.get("result") or {}

        raise HttpError(0, url, f"all retries exhausted: {last_exc!r}")

    async def get_bytes(self, url: str) -> bytes:
        """Fetch raw bytes from an absolute URL (e.g. archive CSV.gz).

        Args:
            url: Absolute URL (may be on a different host).

        Returns:
            Raw response bytes.

        Raises:
            HttpError: On any non-200 or transport failure.
        """
        self._assert_open()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            await self._rate.wait()
            try:
                async with self._session.get(url) as resp:  # type: ignore[union-attr]
                    if resp.status == 404:
                        raise HttpError(404, url, "not found")
                    if resp.status in _RETRY_STATUSES:
                        last_exc = HttpError(resp.status, url, "")
                        if attempt < self._max_retries:
                            await self._sleep(attempt)
                            continue
                        raise last_exc
                    if resp.status != 200:
                        raise HttpError(resp.status, url, await resp.text())
                    return await resp.read()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await self._sleep(attempt)
                    continue
                raise HttpError(0, url, str(exc)) from exc

        raise HttpError(0, url, f"all retries exhausted: {last_exc!r}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_open(self) -> None:
        if self._session is None:
            raise RuntimeError("HttpClient must be used inside an 'async with' block")

    async def _sleep(self, attempt: int) -> None:
        delay = self._backoff * (2 ** attempt)
        await asyncio.sleep(random.uniform(0.0, delay))
