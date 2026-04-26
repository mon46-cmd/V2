"""Core utilities: config, logging, paths, time, ids, errors, retry, bus.

Nothing in this package performs network I/O.  Any code that touches the
exchange, the AI, or external services lives *outside* core.

Startup sequence (every process entry point)
--------------------------------------------
::

    from core import load_config
    from core.log import configure, get_logger

    cfg = load_config()
    configure(cfg, process="scanner")

    log = get_logger(__name__)
    log.info("process started")
"""
from __future__ import annotations

from .bus    import EventBus, bus
from .config import Config, load_config
from .errors import (
    AIBudgetExceededError,
    AIError,
    AITimeoutError,
    AppError,
    AuthError,
    ConfigError,
    DataError,
    ExchangeError,
    LockBusy,
    NetworkError,
    RateLimitError,
    SymbolNotFoundError,
    ValidationError,
    handle,
    is_retryable,
)
from .ids   import run_id, short_id, ulid
from .log   import configure, get_logger
from .paths import ensure_dirs, run_dir, safe_write
from .retry import Retrier, async_retry, sync_retry
from .time  import ms_since_epoch, now_utc, to_utc, utc_isoformat

__all__ = [
    # config
    "Config",
    "load_config",
    # logging
    "configure",
    "get_logger",
    # paths
    "ensure_dirs",
    "run_dir",
    "safe_write",
    # time
    "now_utc",
    "to_utc",
    "utc_isoformat",
    "ms_since_epoch",
    # ids
    "run_id",
    "ulid",
    "short_id",
    # errors
    "AppError",
    "ConfigError",
    "DataError",
    "ValidationError",
    "ExchangeError",
    "RateLimitError",
    "AuthError",
    "SymbolNotFoundError",
    "AIError",
    "AIBudgetExceededError",
    "AITimeoutError",
    "NetworkError",
    "LockBusy",
    "handle",
    "is_retryable",
    # retry
    "async_retry",
    "sync_retry",
    "Retrier",
    # bus
    "EventBus",
    "bus",
]
