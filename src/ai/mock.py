"""Dry-run mock: returns deterministic canned responses without spending money.

Activated automatically when ``cfg.ai_dry_run=True``, when
``cfg.openrouter_api_key`` is empty, or when ``AI_KILL_SWITCH=1`` is set.

The mock returns valid JSON strings that satisfy whatever the caller expects.
It also records itself in the audit log (with ``dry_run=True``) so the log
stays continuous in dry-run mode.

Usage::

    from ai.mock import MockClient

    mock = MockClient()
    resp = mock.chat(model="openai/gpt-4o-mini",
                     messages=[Message("user", "Hello")])
    # resp.content is a canned JSON string
    # resp.latency_ms is 0 (instant)
"""
from __future__ import annotations

import json
import uuid

from ai.http import Message, Response


# Default canned response -- valid JSON so downstream parsers don't crash.
_DEFAULT_CONTENT = json.dumps({
    "dry_run": True,
    "note":    "This is a mock response.  Set AI_DRY_RUN=0 and provide an API key for live calls.",
})


class MockClient:
    """Synchronous mock that mimics the interface of ``OpenRouterHTTP.chat``.

    ``chat()`` is intentionally *not* async here -- it is always called
    from the ``AIClient`` which wraps it appropriately.

    Args:
        default_content: JSON string to return as the model's reply.
                         Override this in tests to inject specific payloads.
    """

    def __init__(self, default_content: str = _DEFAULT_CONTENT) -> None:
        self._content = default_content

    def chat(
        self,
        model:       str,
        messages:    list[Message],
        *,
        temperature: float = 0.2,
        max_tokens:  int   = 1500,
        json_mode:   bool  = True,
    ) -> Response:
        """Return a canned ``Response`` without making a network call."""
        return Response(
            model         = model,
            content       = self._content,
            usage         = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            http_status   = 200,
            latency_ms    = 0,
            finish_reason = "stop",
            raw_envelope  = {"dry_run": True},
        )


def mock_call_id() -> str:
    """Generate a deterministic-looking call ID for dry-run records."""
    return "dry" + uuid.uuid4().hex[:12]
