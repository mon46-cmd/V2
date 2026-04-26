"""Structured JSON logging.

One call per process at startup:

    from core import load_config
    from core.log import configure, get_logger
    cfg = load_config()
    configure(cfg, process="scanner")

    log = get_logger(__name__)
    log.info("started", extra={"symbol": "BTCUSDT"})

Output streams:
  * File  -- JSON-lines, daily rotation, 14-day retention, all levels >= cfg.log_level.
  * Console -- plain text, WARNING and above only (operator feedback on VPS).

Extra fields added to the log record via `extra={...}` are merged into
the JSON payload automatically.

Secrets are never logged intentionally; but `SafeFilter` also strips obvious
API keys from log messages as a last-resort safety net.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config


# ---------------------------------------------------------------------------
# Secret masking
# ---------------------------------------------------------------------------

_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(Authorization['\"]?\s*[:=]\s*['\"]?)[^'\"\s,}]+", re.IGNORECASE), r"\1***"),
    (re.compile(r"(Bearer\s+)[A-Za-z0-9\-_.=]+"), r"\1***"),
    (re.compile(r"sk-[A-Za-z0-9\-_]{10,}"), "sk-***"),
    (re.compile(r"(api[_-]?(?:key|secret)\s*[:=]\s*)[^\s,}\"']+", re.IGNORECASE), r"\1***"),
]


def _mask(msg: str) -> str:
    for pat, repl in _SECRET_PATTERNS:
        msg = pat.sub(repl, msg)
    return msg


class SafeFilter(logging.Filter):
    """Strip obvious secret patterns from log messages before they are
    written to any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = _mask(str(record.msg))
        except Exception:  # noqa: BLE001
            pass
        return True


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

# Fields that are part of LogRecord's internal bookkeeping -- we never want
# to emit these as extra payload keys.
_SKIP_KEYS: frozenset[str] = frozenset({
    "args", "msg", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created",
    "msecs", "relativeCreated", "thread", "threadName", "processName",
    "process", "name", "message", "taskName",
})


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts":     self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        # Merge any extra fields added via extra={...} kwarg.
        for k, v in record.__dict__.items():
            if k in _SKIP_KEYS:
                continue
            try:
                json.dumps(v)
                payload[k] = v
            except (TypeError, ValueError):
                payload[k] = repr(v)

        return json.dumps(payload, ensure_ascii=True)


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def configure(cfg: "Config", process: str) -> None:
    """Install the structured file handler and a plain console handler.

    Idempotent -- safe to call multiple times (previous root handlers are
    removed first so you never end up with duplicate log lines).

    Args:
        cfg:     Loaded Config object.
        process: Logical name for this process, used as the log filename
                 (e.g. ``"scanner"``, ``"exec"``).
    """
    root = logging.getLogger()
    root.setLevel(cfg.log_level)

    # Remove stale handlers from previous calls or pytest captures.
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass

    # --- File handler (JSON, daily rotation) ---
    cfg.log_root.mkdir(parents=True, exist_ok=True)
    log_file = cfg.log_root / f"{process}.log"

    file_h = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        backupCount=14,
        utc=True,
        encoding="utf-8",
    )
    file_h.setFormatter(JsonFormatter())
    file_h.addFilter(SafeFilter())
    file_h.setLevel(cfg.log_level)
    root.addHandler(file_h)

    # --- Console handler (plain text, WARNING+) ---
    console_h = logging.StreamHandler()
    console_h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    console_h.addFilter(SafeFilter())
    console_h.setLevel("WARNING")
    root.addHandler(console_h)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Prefer ``get_logger(__name__)`` at the top of every module so that
    log records carry the correct dotted path automatically.

    Example::

        log = get_logger(__name__)
        log.info("fetched candles", extra={"symbol": "ETHUSDT", "rows": 200})
    """
    return logging.getLogger(name)
