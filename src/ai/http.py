"""OpenRouter async HTTP client.

One focused responsibility: send a list of messages to the
``/chat/completions`` endpoint and return a structured ``Response``
dataclass.  No prompt rendering, no schema validation, no business logic.

Design:
- Uses ``aiohttp`` and is always used inside ``async with`` via the
  caller -- the session is passed in so a single pool can be reused.
- 429 / 5xx responses are retried with jittered back-off (3 attempts).
- Every request/response is returned verbatim; the caller decides what
  to do with the raw text.
- Raises ``AIHttpError`` (-> ``core.errors.AIError``) on non-recoverable
  failure so callers can catch at whatever level they need.

Usage::

    import aiohttp
    from ai.http import OpenRouterHTTP, Message

    async with aiohttp.ClientSession() as session:
        http = OpenRouterHTTP(cfg, session)
        resp = await http.chat(
            model   = "openai/gpt-4o-mini",
            messages= [Message("system", "You are helpful."),
                       Message("user", "Hello!")],
        )
        print(resp.content)   # raw string from the model
        print(resp.usage)     # {prompt_tokens, completion_tokens, total_tokens}
"""
from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from core.config import Config
from core.errors import AIError


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A single chat turn."""
    role:    str   # "system" | "user" | "assistant"
    content: str


@dataclass
class Response:
    """Parsed OpenRouter response."""
    model:          str
    content:        str                  # raw text from the model
    usage:          dict[str, int]       # prompt/completion/total tokens
    http_status:    int
    latency_ms:     int
    finish_reason:  str                  # "stop" | "length" | "content_filter" | ...
    raw_envelope:   dict[str, Any] = field(repr=False)  # full JSON from API


class AIHttpError(AIError):
    """Non-recoverable HTTP failure after all retries are exhausted."""

    def __init__(self, status: int, url: str, body: str) -> None:
        super().__init__(f"OpenRouter HTTP {status}: {body[:300]}", status=status, url=url)
        self.status = status
        self.url    = url


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

_RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES    = 3
_BACKOFF_BASE   = 1.0


class OpenRouterHTTP:
    """Thin async wrapper around the OpenRouter chat/completions endpoint.

    Args:
        cfg:     Loaded Config (provides api key, base url, title, referer).
        session: An open ``aiohttp.ClientSession``.  Caller owns lifecycle.
    """

    _URL_PATH = "/chat/completions"

    def __init__(self, cfg: Config, session: aiohttp.ClientSession) -> None:
        self._cfg     = cfg
        self._session = session
        self._url     = cfg.openrouter_base_url.rstrip("/") + self._URL_PATH
        self._headers = self._build_headers(cfg)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def chat(
        self,
        model:       str,
        messages:    list[Message],
        *,
        temperature: float          = 0.2,
        max_tokens:  int            = 1500,
        json_mode:   bool           = True,
        web_search:  bool           = False,
        max_results: int            = 5,
    ) -> Response:
        """Send a chat request and return a parsed ``Response``.

        Args:
            model:       OpenRouter model ID, e.g. ``"x-ai/grok-4.1-fast"``.
            messages:    Ordered list of ``Message`` objects.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens:  Maximum completion tokens.
            json_mode:   If True, requests ``response_format={"type":"json_object"}``.
                         The model is expected to return valid JSON.
            web_search:  If True, enables the ``openrouter:web_search`` server tool.
                         The model will search the web when it deems necessary.
                         NOTE: The old ``:online`` suffix is deprecated -- this is
                         the current recommended approach (April 2026).
                         For xAI models, OpenRouter routes to xAI's native search.
            max_results: Number of search results to retrieve (default 5).
                         Only relevant when ``web_search=True``.

        Returns:
            ``Response`` dataclass with content, usage, and metadata.

        Raises:
            ``AIHttpError`` if all retries fail.
        """
        body: dict[str, Any] = {
            "model":       model,
            "messages":    [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        if web_search:
            body["tools"] = [{
                "type": "function",
                "function": {
                    "name":        "openrouter:web_search",
                    "description": "Search the web for recent information.",
                    "parameters": {
                        "type":       "object",
                        "properties": {
                            "query":       {"type": "string"},
                            "max_results": {"type": "integer", "default": max_results},
                            "engine":      {"type": "string",  "default": "native"},
                        },
                        "required": ["query"],
                    },
                },
            }]

        return await self._post_with_retry(body)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _post_with_retry(self, body: dict[str, Any]) -> Response:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return await self._post_once(body)
            except AIHttpError as exc:
                if exc.status not in _RETRY_STATUSES:
                    raise
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(random.uniform(0.0, _BACKOFF_BASE * (2 ** attempt)))
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(random.uniform(0.0, _BACKOFF_BASE * (2 ** attempt)))
        raise AIHttpError(0, self._url, f"all retries exhausted: {last_exc!r}")

    async def _post_once(self, body: dict[str, Any]) -> Response:
        t0 = time.perf_counter()
        timeout = aiohttp.ClientTimeout(total=self._cfg.ai_timeout_sec)
        async with self._session.post(
            self._url,
            headers=self._headers,
            json=body,
            timeout=timeout,
        ) as resp:
            status = resp.status
            text   = await resp.text()
        latency_ms = int((time.perf_counter() - t0) * 1000)

        if status in _RETRY_STATUSES:
            raise AIHttpError(status, self._url, text)
        if status != 200:
            raise AIHttpError(status, self._url, text)

        try:
            envelope: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AIHttpError(0, self._url, f"non-JSON response: {text[:200]}") from exc

        choices = envelope.get("choices") or []
        content = ""
        finish  = "unknown"
        if choices:
            choice  = choices[0]
            content = (choice.get("message") or {}).get("content") or ""
            finish  = choice.get("finish_reason") or "unknown"

        usage: dict[str, int] = {
            "prompt_tokens":     int(envelope.get("usage", {}).get("prompt_tokens", 0)),
            "completion_tokens": int(envelope.get("usage", {}).get("completion_tokens", 0)),
            "total_tokens":      int(envelope.get("usage", {}).get("total_tokens", 0)),
        }

        return Response(
            model         = str(envelope.get("model", body.get("model", ""))),
            content       = content,
            usage         = usage,
            http_status   = status,
            latency_ms    = latency_ms,
            finish_reason = finish,
            raw_envelope  = envelope,
        )

    @staticmethod
    def _build_headers(cfg: Config) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {cfg.openrouter_api_key}",
            "Content-Type":  "application/json",
        }
        if cfg.openrouter_title:
            h["X-Title"] = cfg.openrouter_title
        if cfg.openrouter_referer:
            h["HTTP-Referer"] = cfg.openrouter_referer
        return h
