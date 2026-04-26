"""Downloader-specific exceptions.

These extend the project-wide hierarchy in ``core.errors`` where appropriate
so that callers can catch at either level of specificity.
"""
from __future__ import annotations

from core.errors import ExchangeError, NetworkError


class HttpError(NetworkError):
    """Transport failure: non-2xx status, timeout, or connection reset.

    Attributes:
        status: HTTP status code (0 for connection-level errors).
        url:    Request URL.
        body:   First 200 chars of the response body (for diagnosis).
    """

    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"HTTP {status} {url}: {body[:200]}", status=status, url=url)
        self.status = status
        self.url    = url
        self.body   = body


class BybitApiError(ExchangeError):
    """Bybit JSON envelope error: retCode != 0.

    Attributes:
        ret_code: Bybit retCode integer.
        ret_msg:  Bybit retMsg string.
        url:      Request URL.
    """

    def __init__(self, ret_code: int, ret_msg: str, url: str) -> None:
        super().__init__(
            f"Bybit ret={ret_code} msg={ret_msg!r}",
            ret_code=ret_code,
            url=url,
        )
        self.ret_code = ret_code
        self.ret_msg  = ret_msg
        self.url      = url


class CacheError(ExchangeError):
    """Unrecoverable cache read/write failure."""
