"""Retry utilities with jittered exponential back-off.

Two decorators are provided -- one for ``async`` callables, one for plain
synchronous functions.  Both share the same back-off formula:

    delay = min(base * 2**attempt, cap) + uniform(0, jitter)

Usage
-----
Async (most common in this project)::

    from core.retry import async_retry
    from core.errors import NetworkError, RateLimitError

    @async_retry(retries=5, retryable=(NetworkError, RateLimitError))
    async def fetch_candles(symbol: str) -> list:
        ...

Sync::

    from core.retry import sync_retry

    @sync_retry(retries=3)
    def read_file(path: str) -> str:
        ...

Manual (inside a loop)::

    from core.retry import Retrier

    async def run():
        r = Retrier(retries=4)
        async for attempt in r:
            try:
                result = await risky_call()
                break
            except NetworkError as exc:
                await r.on_error(exc)
"""
from __future__ import annotations

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Coroutine, Sequence, Type, TypeVar

from .errors import is_retryable

log = logging.getLogger(__name__)

F  = TypeVar("F",  bound=Callable[..., Any])
CF = TypeVar("CF", bound=Callable[..., Coroutine[Any, Any, Any]])

_DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (Exception,)


def _backoff(attempt: int, base: float = 0.5, cap: float = 60.0, jitter: float = 1.0) -> float:
    """Jittered exponential back-off in seconds."""
    return min(base * (2 ** attempt), cap) + random.uniform(0.0, jitter)


# ---------------------------------------------------------------------------
# Async decorator
# ---------------------------------------------------------------------------

def async_retry(
    retries:   int                           = 5,
    base:      float                         = 0.5,
    cap:       float                         = 60.0,
    jitter:    float                         = 1.0,
    retryable: Sequence[Type[Exception]]     = _DEFAULT_RETRYABLE,
) -> Callable[[CF], CF]:
    """Retry an ``async`` function on retryable exceptions with back-off."""
    def decorator(fn: CF) -> CF:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except tuple(retryable) as exc:  # type: ignore[misc]
                    last_exc = exc
                    if not is_retryable(exc) or attempt >= retries:
                        raise
                    delay = _backoff(attempt, base, cap, jitter)
                    log.warning(
                        "retrying after error",
                        extra={"func": fn.__qualname__, "attempt": attempt + 1,
                               "retries": retries, "delay": round(delay, 2), "exc": repr(exc)},
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sync decorator
# ---------------------------------------------------------------------------

def sync_retry(
    retries:   int                       = 3,
    base:      float                     = 0.5,
    cap:       float                     = 30.0,
    jitter:    float                     = 0.5,
    retryable: Sequence[Type[Exception]] = _DEFAULT_RETRYABLE,
) -> Callable[[F], F]:
    """Retry a synchronous function on retryable exceptions with back-off."""
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except tuple(retryable) as exc:  # type: ignore[misc]
                    last_exc = exc
                    if not is_retryable(exc) or attempt >= retries:
                        raise
                    delay = _backoff(attempt, base, cap, jitter)
                    log.warning(
                        "retrying after error",
                        extra={"func": fn.__qualname__, "attempt": attempt + 1,
                               "retries": retries, "delay": round(delay, 2), "exc": repr(exc)},
                    )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Manual retrier (for use inside loops)
# ---------------------------------------------------------------------------

class Retrier:
    """Async iterator for manual retry loops.

    Example::

        r = Retrier(retries=4)
        async for attempt in r:
            try:
                result = await risky()
                break              # success -- exit the loop
            except NetworkError as exc:
                await r.on_error(exc)
    """

    def __init__(self, retries: int = 5, base: float = 0.5,
                 cap: float = 60.0, jitter: float = 1.0) -> None:
        self._retries = retries
        self._base    = base
        self._cap     = cap
        self._jitter  = jitter
        self._attempt = 0

    def __aiter__(self) -> "Retrier":
        self._attempt = 0
        return self

    async def __anext__(self) -> int:
        if self._attempt > self._retries:
            raise StopAsyncIteration
        current       = self._attempt
        self._attempt += 1
        return current

    async def on_error(self, exc: Exception) -> None:
        """Sleep and then re-raise if no attempts remain or error is fatal."""
        if not is_retryable(exc) or self._attempt > self._retries:
            raise exc
        delay = _backoff(self._attempt - 1, self._base, self._cap, self._jitter)
        log.warning("retrying after error",
                    extra={"attempt": self._attempt, "retries": self._retries,
                           "delay": round(delay, 2), "exc": repr(exc)})
        await asyncio.sleep(delay)
