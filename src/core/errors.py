"""Project-wide exception hierarchy and structured error handling.

Exception hierarchy
-------------------
AppError                         -- base for all application errors
  ConfigError                    -- bad or missing configuration
  DataError                      -- unexpected / corrupt data shape
    ValidationError              -- Pydantic / schema mismatch
  ExchangeError                  -- exchange API errors
    RateLimitError               -- HTTP 429 / rate-limit reached
    AuthError                    -- 401 / 403 from the exchange
    SymbolNotFoundError          -- symbol does not exist on exchange
  AIError                        -- LLM / OpenRouter errors
    AIBudgetExceededError        -- daily spend cap reached
    AITimeoutError               -- AI call timed out
  NetworkError                   -- generic HTTP / socket failure
  LockBusy                       -- another process holds the lock

Utility
-------
``handle(exc, logger, *, context)`` -- log + optionally re-raise in a
structured, consistent way.

``is_retryable(exc)`` -- True when the caller *should* retry after backoff.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class AppError(Exception):
    """Base class for all application-level exceptions.

    Carry structured context via the ``ctx`` mapping so that callers can
    inspect individual fields without parsing the message string.
    """

    def __init__(self, message: str, **ctx: Any) -> None:
        super().__init__(message)
        self.ctx: dict[str, Any] = ctx

    def __str__(self) -> str:
        base = super().__str__()
        if self.ctx:
            pairs = ", ".join(f"{k}={v!r}" for k, v in self.ctx.items())
            return f"{base} [{pairs}]"
        return base


# --- Configuration ---

class ConfigError(AppError):
    """Raised when a required configuration value is missing or invalid."""


# --- Data integrity ---

class DataError(AppError):
    """Raised when received data has an unexpected shape or content."""


class ValidationError(DataError):
    """Raised when Pydantic or manual schema validation fails."""


# --- Exchange ---

class ExchangeError(AppError):
    """Raised on any error response from the exchange REST / WS API."""


class RateLimitError(ExchangeError):
    """HTTP 429 or exchange-specific rate-limit error.  Always retryable."""


class AuthError(ExchangeError):
    """HTTP 401 / 403 -- bad credentials or expired session.  Not retryable."""


class SymbolNotFoundError(ExchangeError):
    """The requested symbol does not exist on the exchange."""


# --- AI / LLM ---

class AIError(AppError):
    """Raised on any error from the AI / OpenRouter layer."""


class AIBudgetExceededError(AIError):
    """Daily AI spend cap has been reached.  Not retryable until reset."""


class AITimeoutError(AIError):
    """AI call exceeded the configured timeout.  May be retried."""


# --- Network ---

class NetworkError(AppError):
    """Generic HTTP or socket-level failure.  Usually retryable."""


# --- Process locking ---

class LockBusy(AppError):
    """Raised when another live process already holds the lock."""


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

#: Exceptions that warrant an automatic retry with backoff.
_RETRYABLE: tuple[type[Exception], ...] = (
    RateLimitError,
    AITimeoutError,
    NetworkError,
)

#: Exceptions that should *never* be retried.
_FATAL: tuple[type[Exception], ...] = (
    ConfigError,
    AuthError,
    AIBudgetExceededError,
    LockBusy,
)


def is_retryable(exc: BaseException) -> bool:
    """Return True when the exception is safe to retry after a backoff.

    The classification is conservative: unknown exception types return
    False so that the default behaviour is to surface, not swallow, them.
    """
    if isinstance(exc, _FATAL):
        return False
    return isinstance(exc, (_RETRYABLE))


# ---------------------------------------------------------------------------
# Structured error handler
# ---------------------------------------------------------------------------

def handle(
    exc: BaseException,
    logger: logging.Logger,
    *,
    context: str = "",
    reraise: bool = True,
    level: int = logging.ERROR,
) -> None:
    """Log `exc` in a structured way and optionally re-raise it.

    Args:
        exc:     The caught exception.
        logger:  Module-scoped logger (use ``get_logger(__name__)``).
        context: Human-readable context string, e.g. ``"fetching OHLCV"``.
        reraise: If True (default), re-raise ``exc`` after logging.
        level:   Log level for the message (default ``ERROR``).

    Example::

        try:
            data = await exchange.fetch(symbol)
        except Exception as exc:
            handle(exc, log, context="fetch OHLCV", reraise=False)
    """
    extra: dict[str, Any] = {
        "exc_type": type(exc).__name__,
    }
    if context:
        extra["context"] = context
    if isinstance(exc, AppError) and exc.ctx:
        extra.update(exc.ctx)

    tb = traceback.format_exc()
    if tb and tb.strip() != "NoneType: None":
        extra["traceback"] = tb

    msg = f"{context + ': ' if context else ''}{exc}"
    logger.log(level, msg, extra=extra)

    if reraise:
        raise exc
