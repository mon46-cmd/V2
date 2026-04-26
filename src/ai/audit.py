"""Prompt/response audit log.

Every AI call (live or dry-run) is appended to a newline-delimited JSON
file (``<log_root>/ai_calls/YYYY-MM-DD.jsonl``) and to the structured
application logger.

Each JSONL record is a self-contained JSON object on a single line:
  {
    "call_id":    "a1b2c3d4e5f6",
    "ts_utc":     "2026-04-26T11:38:00.123Z",
    "call_type":  "analysis",
    "model":      "openai/gpt-4o-mini",
    "symbol":     "BTCUSDT",          // null for multi-symbol calls
    "dry_run":    false,
    "messages":   [{"role": "system", "content": "..."}, ...],
    "response": {
      "content":       "...",
      "finish_reason": "stop",
      "usage":         {"prompt_tokens": 412, "completion_tokens": 88},
      "latency_ms":    741,
      "http_status":   200
    },
    "cost_usd":    0.0003,
    "error":       null              // or error message string
  }

The file is rotated daily (UTC).  Old files accumulate; no cleanup is done
here so the operator can decide retention policy.

Usage::

    from ai.audit import AuditLog
    from core import load_config

    cfg = load_config()
    audit = AuditLog(cfg)
    audit.record(call_id, call_type, model, messages, response, cost_usd)
    audit.record_error(call_id, call_type, model, messages, error_msg)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from ai.http import Message, Response
from core.config import Config

log = logging.getLogger(__name__)

# Truncate very long message content in the structured log (not in JSONL file).
_LOG_CONTENT_LIMIT = 400


class AuditLog:
    """Thread-safe JSONL audit log for AI calls.

    One ``AuditLog`` instance should be shared for the lifetime of the
    process.  It is thread-safe and safe to use from async contexts.

    Args:
        cfg:  Loaded Config.  Uses ``cfg.log_root / "ai_calls"``.
    """

    def __init__(self, cfg: Config) -> None:
        self._dir  = cfg.log_root / "ai_calls"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        call_id:   str,
        call_type: str,
        model:     str,
        messages:  list[Message],
        response:  Response,
        *,
        symbol:    str | None = None,
        cost_usd:  float      = 0.0,
        dry_run:   bool       = False,
    ) -> None:
        """Write a successful call record to the JSONL log and structured logger.

        Args:
            call_id:   Unique identifier for this call (hex string).
            call_type: Human label, e.g. ``"analysis"``, ``"watchlist"``.
            model:     Model identifier returned by OpenRouter.
            messages:  The messages list sent to the API.
            response:  The ``Response`` object returned by ``OpenRouterHTTP``.
            symbol:    Optional symbol the call relates to.
            cost_usd:  Estimated or actual cost in USD.
            dry_run:   Whether this was a mock/dry-run call.
        """
        entry = self._build_entry(
            call_id=call_id, call_type=call_type, model=model,
            messages=messages, symbol=symbol, dry_run=dry_run,
            response={
                "content":       response.content,
                "finish_reason": response.finish_reason,
                "usage":         response.usage,
                "latency_ms":    response.latency_ms,
                "http_status":   response.http_status,
            },
            cost_usd=cost_usd,
            error=None,
        )
        self._write(entry)
        self._log_call(entry, response.content)

    def record_error(
        self,
        call_id:   str,
        call_type: str,
        model:     str,
        messages:  list[Message],
        error:     str,
        *,
        symbol:   str | None = None,
        dry_run:  bool       = False,
    ) -> None:
        """Write a failed call record.

        Args:
            error: String representation of the exception.
        """
        entry = self._build_entry(
            call_id=call_id, call_type=call_type, model=model,
            messages=messages, symbol=symbol, dry_run=dry_run,
            response=None,
            cost_usd=0.0,
            error=error,
        )
        self._write(entry)
        log.error(
            "AI call failed",
            extra={
                "call_id":   call_id,
                "call_type": call_type,
                "model":     model,
                "symbol":    symbol,
                "error":     error,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        *,
        call_id:   str,
        call_type: str,
        model:     str,
        messages:  list[Message],
        symbol:    str | None,
        dry_run:   bool,
        response:  dict[str, Any] | None,
        cost_usd:  float,
        error:     str | None,
    ) -> dict[str, Any]:
        return {
            "call_id":   call_id,
            "ts_utc":    _now_iso(),
            "call_type": call_type,
            "model":     model,
            "symbol":    symbol,
            "dry_run":   dry_run,
            "messages":  [{"role": m.role, "content": m.content} for m in messages],
            "response":  response,
            "cost_usd":  round(cost_usd, 8),
            "error":     error,
            "prompt_version": _prompt_hash(messages),
        }

    def _write(self, entry: dict[str, Any]) -> None:
        path = self._dir / f"{_utc_day()}.jsonl"
        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)

    def _log_call(self, entry: dict[str, Any], content: str) -> None:
        resp = entry.get("response") or {}
        log.info(
            "AI call completed",
            extra={
                "call_id":           entry["call_id"],
                "call_type":         entry["call_type"],
                "model":             entry["model"],
                "symbol":            entry["symbol"],
                "dry_run":           entry["dry_run"],
                "latency_ms":        resp.get("latency_ms"),
                "prompt_tokens":     (resp.get("usage") or {}).get("prompt_tokens"),
                "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                "cost_usd":          entry["cost_usd"],
                "finish_reason":     resp.get("finish_reason"),
                "content_preview":   content[:_LOG_CONTENT_LIMIT],
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _utc_day() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")


def _prompt_hash(messages: list[Message]) -> str:
    """Stable 12-char hash of the system messages.

    Captures only the ``system`` role content because that is the part of
    the prompt that defines the contract / rules / schema.  User-role
    messages contain per-call data (snapshot rows, watchlist) and would
    make every call-id unique, defeating the purpose of a *version* tag.

    The hash lets us A/B-test prompt revisions: filter audit records by
    ``prompt_version`` to compare hit rate, cost, and downstream PnL
    between two prompt variants.
    """
    h = hashlib.sha1()
    for m in messages:
        if m.role == "system":
            h.update(b"system\n")
            h.update(m.content.encode("utf-8", errors="replace"))
            h.update(b"\n--\n")
    return h.hexdigest()[:12]
