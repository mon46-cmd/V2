"""Central AI client: the single entry point for all OpenRouter calls.

``AIClient`` wires together:
  - ``OpenRouterHTTP``  -- actual HTTP calls
  - ``MockClient``      -- dry-run / offline fallback
  - ``BudgetTracker``   -- daily USD spend guard
  - ``AuditLog``        -- JSONL + structured log of every call

Public interface
----------------
Every call goes through ``AIClient.chat()``.  The caller provides:
  - the model name (or lets the client pick from config defaults)
  - a list of ``Message`` objects
  - optional metadata (call_type, symbol) for logging

``chat()`` returns the model's raw string response.  Parsing / validation
is the caller's responsibility -- this keeps the client simple and reusable.

Dry-run / offline mode
-----------------------
The client automatically falls back to ``MockClient`` when:
  - ``cfg.ai_dry_run = True``
  - ``cfg.openrouter_api_key`` is empty
  - environment variable ``AI_KILL_SWITCH=1``

Budget guard
------------
If a ``BudgetTracker`` is supplied and the estimated cost exceeds the daily
cap, ``chat()`` raises ``BudgetExhaustedError`` *before* sending anything.
This is a hard stop -- the caller can catch it and decide whether to skip
the call, use a cheaper model, etc.

Usage::

    import aiohttp
    from ai.client import AIClient, BudgetExhaustedError
    from ai.http   import Message
    from ai.budget import BudgetTracker
    from ai.audit  import AuditLog
    from core      import load_config

    cfg    = load_config()
    budget = BudgetTracker(daily_cap_usd=1.00, state_path=cfg.run_root / "budget.json")
    audit  = AuditLog(cfg)

    async with aiohttp.ClientSession() as session:
        client = AIClient(cfg, session=session, budget=budget, audit=audit)

        try:
            reply = await client.chat(
                model     = cfg.model_deep,
                messages  = [
                    Message("system", "You are a concise crypto analyst."),
                    Message("user",   "Is BTCUSDT bullish right now?"),
                ],
                call_type = "analysis",
                symbol    = "BTCUSDT",
            )
            print(reply)   # raw string from the model
        except BudgetExhaustedError as exc:
            print(f"Daily budget exhausted: {exc}")
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import aiohttp

from ai.audit   import AuditLog
from ai.budget  import BudgetTracker
from ai.http    import AIHttpError, Message, OpenRouterHTTP, Response
from ai.mock    import MockClient, mock_call_id
from ai.pricing import cost_usd, get_registry
from core.config import Config
from core.errors import AIError

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BudgetExhaustedError(AIError):
    """Raised before a call is made when the estimated cost exceeds the cap."""

    def __init__(self, needed: float, remaining: float) -> None:
        super().__init__(
            f"Daily AI budget exhausted: need ${needed:.4f}, remaining ${remaining:.4f}",
            needed=needed,
            remaining=remaining,
        )
        self.needed    = needed
        self.remaining = remaining


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _new_call_id() -> str:
    return uuid.uuid4().hex[:16]


def _estimate_pre_call(model: str, messages: list[Message], max_tokens: int) -> float:
    """Rough pre-call cost estimate (chars / 4 ~= tokens)."""
    total_chars   = sum(len(m.content) for m in messages)
    prompt_tokens = total_chars // 4
    return cost_usd(model, prompt_tokens, max_tokens)

class AIClient:
    """Route AI chat calls to OpenRouter or the dry-run mock.

    Args:
        cfg:     Loaded Config.
        session: Open ``aiohttp.ClientSession`` (caller owns lifecycle).
        budget:  Optional ``BudgetTracker``.  If supplied, calls are blocked
                 before they are made when the cap would be exceeded.
        audit:   Optional ``AuditLog``.  If supplied, every call result is
                 written to the JSONL file and structured logger.
    """

    def __init__(
        self,
        cfg:     Config,
        *,
        session: aiohttp.ClientSession | None = None,
        budget:  BudgetTracker | None         = None,
        audit:   AuditLog | None              = None,
    ) -> None:
        self._cfg    = cfg
        self._budget = budget
        self._audit  = audit

        kill_switch = os.getenv("AI_KILL_SWITCH", "").strip().lower() in ("1", "true", "yes")
        self._offline = (
            kill_switch
            or cfg.ai_dry_run
            or not cfg.openrouter_api_key
        )

        if self._offline:
            self._http = None
            self._mock = MockClient()
            if kill_switch:
                log.warning("AI_KILL_SWITCH is set -- all AI calls will use the mock")
            elif cfg.ai_dry_run:
                log.info("AI_DRY_RUN=1 -- using mock responses (no API spend)")
            else:
                log.info("No OPENROUTER_API_KEY -- using mock responses")
        else:
            if session is None:
                raise ValueError("A live aiohttp.ClientSession must be provided when not in dry-run mode")
            self._http = OpenRouterHTTP(cfg, session)
            self._mock = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def chat(
        self,
        model:       str,
        messages:    list[Message],
        *,
        call_type:   str        = "chat",
        symbol:      str | None = None,
        temperature: float      = 0.2,
        max_tokens:  int        = 1500,
        json_mode:   bool       = True,
        web_search:  bool       = False,
        max_results: int        = 5,
    ) -> str:
        """Send a chat request and return the model's response as a string.

        Args:
            model:       OpenRouter model ID.
            messages:    System + user (+ any prior assistant turns) as
                         ``Message`` objects.
            call_type:   Label used in logs and audit (e.g. ``"analysis"``).
            symbol:      Optional ticker symbol, recorded in audit.
            temperature: Sampling temperature.
            max_tokens:  Maximum completion tokens.
            json_mode:   Request JSON output format.
            web_search:  Enable ``openrouter:web_search`` tool (Prompt A only).
            max_results: Max search results when ``web_search=True``.

        Returns:
            Raw content string from the model.

        Raises:
            ``BudgetExhaustedError`` if the budget cap would be exceeded.
            ``AIHttpError`` if all HTTP retries fail (live mode only).
        """
        call_id  = _new_call_id()
        est_cost = _estimate_pre_call(model, messages, max_tokens)

        # Budget pre-check.
        if self._budget is not None and not self._offline:
            if not self._budget.can_afford(est_cost):
                log.warning(
                    "Budget exhausted -- blocking AI call",
                    extra={"call_id": call_id, "needed": est_cost,
                           "remaining": self._budget.remaining_usd},
                )
                raise BudgetExhaustedError(est_cost, self._budget.remaining_usd)

        log.debug(
            "AI call starting",
            extra={"call_id": call_id, "call_type": call_type,
                   "model": model, "symbol": symbol,
                   "messages": len(messages), "dry_run": self._offline},
        )

        # Dispatch.
        error: str | None = None
        response: Response | None = None
        try:
            if self._offline:
                response = self._mock.chat(  # type: ignore[union-attr]
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                )
            else:
                response = await self._http.chat(  # type: ignore[union-attr]
                    model=model, messages=messages,
                    temperature=temperature, max_tokens=max_tokens, json_mode=json_mode,
                    web_search=web_search, max_results=max_results,
                )
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            if self._audit is not None:
                self._audit.record_error(
                    call_id=call_id, call_type=call_type, model=model,
                    messages=messages, error=error, symbol=symbol, dry_run=self._offline,
                )
            log.error("AI call error", extra={"call_id": call_id, "error": error})
            raise

        # Charge budget on actual token usage (live only).
        actual_cost = 0.0
        if response is not None and self._budget is not None and not self._offline:
            actual_cost = cost_usd(
                model,
                response.usage.get("prompt_tokens", 0),
                response.usage.get("completion_tokens", 0),
            )
            self._budget.charge(actual_cost)
            log.debug(
                "Budget charged",
                extra={
                    "call_id":    call_id,
                    "cost_usd":   actual_cost,
                    "summary":    self._budget.summary(),
                },
            )

        # Audit record.
        if self._audit is not None and response is not None:
            self._audit.record(
                call_id=call_id, call_type=call_type, model=model,
                messages=messages, response=response,
                symbol=symbol, cost_usd=actual_cost, dry_run=self._offline,
            )

        return response.content if response is not None else ""
