"""AI package: OpenRouter communication.

All AI calls go through ``AIClient.chat()``.

Quick-start
-----------
::

    import aiohttp
    from ai         import AIClient, Message
    from ai.budget  import BudgetTracker
    from ai.audit   import AuditLog
    from core       import load_config

    cfg    = load_config()
    budget = BudgetTracker(daily_cap_usd=1.00, state_path=cfg.run_root / "budget.json")
    audit  = AuditLog(cfg)

    async with aiohttp.ClientSession() as session:
        client = AIClient(cfg, session=session, budget=budget, audit=audit)
        reply  = await client.chat(
            model     = cfg.model_deep,
            messages  = [
                Message("system", "You are a concise crypto analyst."),
                Message("user",   "Is BTCUSDT bullish right now?"),
            ],
            call_type = "analysis",
            symbol    = "BTCUSDT",
        )

Dry-run
-------
Set ``AI_DRY_RUN=1`` in your ``.env`` (or ``cfg.ai_dry_run=True``) and the
client returns canned mock responses without spending any money.

Module map
----------
client.py  -- ``AIClient``: the single entry point for all calls.
http.py    -- ``OpenRouterHTTP``: raw HTTP to OpenRouter.
audit.py   -- ``AuditLog``: JSONL + structured log of every call.
budget.py  -- ``BudgetTracker``: daily USD spend cap with file persistence.
mock.py    -- ``MockClient``: deterministic offline mock.
"""
from __future__ import annotations

from ai.client  import AIClient, BudgetExhaustedError
from ai.http    import AIHttpError, Message, Response
from ai.audit   import AuditLog
from ai.budget  import BudgetTracker
from ai.mock    import MockClient
from ai.pricing import cost_usd as estimate_cost   # backward-compat alias

__all__ = [
    "AIClient",
    "BudgetExhaustedError",
    "estimate_cost",
    "AIHttpError",
    "Message",
    "Response",
    "AuditLog",
    "BudgetTracker",
    "MockClient",
]
