"""Live model pricing fetched from OpenRouter + local fallback table.

OpenRouter exposes pricing at ``/api/v1/models`` (no auth required).
``PriceRegistry`` fetches this on startup and caches it in memory.
A hardcoded fallback table is used when the API is unreachable.

Prices are in USD per 1,000,000 tokens (the OpenRouter convention).
Budget enforcement always uses USD/token; this module converts cleanly.

Usage::

    from ai.pricing import PriceRegistry, cost_usd

    registry = PriceRegistry()
    await registry.refresh()                 # fetch from OpenRouter

    usd = registry.cost_usd("x-ai/grok-3", prompt_tokens=500, completion_tokens=200)
    print(f"${usd:.6f}")

    # Or use the module-level singleton (auto-refreshed on first call):
    from ai.pricing import cost_usd
    usd = cost_usd("x-ai/grok-3", 500, 200)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded fallback (USD per 1M tokens).
# Format: model_id -> (prompt_usd_per_1m, completion_usd_per_1m)
# Last verified: 2026-04-26 via web search.
#
# Web search on OpenRouter:
#   - The :online suffix is DEPRECATED. Use the openrouter:web_search
#     server tool instead (pass it in the "tools" field of the request body).
#   - For xAI models, OpenRouter uses xAI's native search when available.
#
# Grok 4.20 has tiered pricing:
#   <= 200K context: $2/M in, $6/M out
#   >  200K context: $4/M in, $12/M out
# We budget the cheaper tier since our prompts are well under 200K.
# ---------------------------------------------------------------------------
_FALLBACK: dict[str, tuple[float, float]] = {
    # --- xAI Grok 4.x family (April 2026) ---
    # Flagship -- full reasoning, highest quality.  Use for social scan only.
    "x-ai/grok-4.20":              (2.00,   6.00),
    "x-ai/grok-4.20-multi-agent":  (2.00,   6.00),   # same retail tier
    # Fast -- 10x cheaper, still very capable.  Use for deep + review.
    "x-ai/grok-4.1-fast":          (0.20,   0.50),
    # Legacy aliases (may still appear in old configs)
    "x-ai/grok-4":                 (2.00,   6.00),
    "x-ai/grok-4-fast":            (0.20,   0.50),
    "x-ai/grok-3":                 (3.00,  15.00),
    "x-ai/grok-3-mini":            (0.30,   0.50),
    "x-ai/grok-3-mini:online":     (0.30,   0.50),   # deprecated suffix, still routes
    "x-ai/grok-2-1212":            (2.00,  10.00),
    "x-ai/grok-beta":              (5.00,  15.00),

    # --- OpenAI ---
    "openai/gpt-4o":               (5.00,  15.00),
    "openai/gpt-4o-mini":          (0.15,   0.60),
    "openai/o3-mini":              (1.10,   4.40),
    "openai/o4-mini":              (1.10,   4.40),

    # --- Anthropic ---
    "anthropic/claude-3-5-sonnet": (3.00,  15.00),
    "anthropic/claude-3-5-haiku":  (0.80,   4.00),
    "anthropic/claude-sonnet-4":   (3.00,  15.00),
    "anthropic/claude-haiku-4":    (0.80,   4.00),

    # --- Google ---
    "google/gemini-2.0-flash":     (0.10,   0.40),
    "google/gemini-2.5-flash":     (0.15,   0.60),
    "google/gemini-2.5-pro":       (1.25,   5.00),

    # --- DeepSeek ---
    "deepseek/deepseek-chat":      (0.14,   0.28),
    "deepseek/deepseek-r1":        (0.55,   2.19),

    # --- Meta ---
    "meta-llama/llama-3.3-70b-instruct": (0.12, 0.30),

    # --- Mock (tests) ---
    "mock/mock-1":                 (0.00,   0.00),
}

# Conservative fallback for completely unknown models.
_DEFAULT_PRICE: tuple[float, float] = (3.00, 15.00)

# Cache lifetime -- re-fetch from OpenRouter every 6 hours.
_CACHE_TTL_SEC = 6 * 3600


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PriceRegistry:
    """Holds per-model pricing, optionally refreshed from OpenRouter.

    Thread/coroutine safe: ``refresh()`` is an async method protected by
    an asyncio lock, so parallel refreshes collapse to one.

    The registry always has a valid state (the fallback table) even before
    ``refresh()`` is called.
    """

    _MODELS_URL = "https://openrouter.ai/api/v1/models"

    def __init__(self) -> None:
        self._prices: dict[str, tuple[float, float]] = dict(_FALLBACK)
        self._fetched_at: float = 0.0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def refresh(self, *, force: bool = False) -> bool:
        """Fetch the model list from OpenRouter and update prices.

        Args:
            force: If True, ignore the cache TTL and always re-fetch.

        Returns:
            True if a live fetch was performed, False if cache was valid.
        """
        async with self._lock:
            age = time.monotonic() - self._fetched_at
            if not force and age < _CACHE_TTL_SEC:
                return False
            return await self._do_fetch()

    def cost_usd(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        """Return the USD cost for one call.

        Args:
            model:             OpenRouter model ID.
            prompt_tokens:     Input token count.
            completion_tokens: Output token count.

        Returns:
            Cost in USD (>= 0).
        """
        p_per_m, c_per_m = self._prices.get(model, _DEFAULT_PRICE)
        return (prompt_tokens * p_per_m + completion_tokens * c_per_m) / 1_000_000.0

    def get_price(self, model: str) -> tuple[float, float]:
        """Return (prompt_usd_per_1m, completion_usd_per_1m) for a model."""
        return self._prices.get(model, _DEFAULT_PRICE)

    def known_models(self) -> list[str]:
        """Return all model IDs in the registry."""
        return list(self._prices.keys())

    def price_table(self) -> dict[str, tuple[float, float]]:
        """Return a snapshot of the full price table (defensive copy)."""
        return dict(self._prices)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _do_fetch(self) -> bool:
        """Hit OpenRouter, parse the model list, merge into _prices."""
        try:
            timeout = aiohttp.ClientTimeout(total=15.0)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.get(self._MODELS_URL) as resp:
                    if resp.status != 200:
                        log.warning("PriceRegistry fetch failed http=%d", resp.status)
                        return False
                    data: dict[str, Any] = await resp.json(content_type=None)
        except Exception as exc:  # noqa: BLE001
            log.warning("PriceRegistry fetch error: %s", exc)
            return False

        updated = 0
        for model_obj in data.get("data", []):
            model_id: str = model_obj.get("id", "")
            pricing    = model_obj.get("pricing") or {}
            try:
                # OpenRouter reports USD/token (not per 1M); multiply by 1M.
                p_rate = float(pricing.get("prompt",     0)) * 1_000_000
                c_rate = float(pricing.get("completion", 0)) * 1_000_000
                if model_id and (p_rate >= 0) and (c_rate >= 0):
                    self._prices[model_id] = (p_rate, c_rate)
                    updated += 1
            except (TypeError, ValueError):
                continue

        self._fetched_at = time.monotonic()
        log.info("PriceRegistry refreshed n_models=%d updated=%d", len(data.get("data", [])), updated)
        return True


# ---------------------------------------------------------------------------
# Module-level singleton -- lazy, no auto-refresh (call refresh() yourself)
# ---------------------------------------------------------------------------

_registry = PriceRegistry()


def cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Module-level convenience: cost in USD using the singleton registry.

    Does NOT auto-refresh from OpenRouter -- call ``get_registry().refresh()``
    at startup if you want live prices.
    """
    return _registry.cost_usd(model, prompt_tokens, completion_tokens)


def get_registry() -> PriceRegistry:
    """Return the module-level singleton registry."""
    return _registry
