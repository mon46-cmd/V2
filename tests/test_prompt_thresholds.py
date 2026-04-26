"""Tests for templated prompt thresholds (M1).

Verifies that the configured threshold values appear in the rendered
system prompts for ``position_review`` and ``deep_analysis``.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from ai.calls import deep_analysis, position_review
from ai.http import Message


# ── helpers ──────────────────────────────────────────────────────────────────

class _CapturingClient:
    """AIClient stand-in that records the messages and returns a canned reply."""

    def __init__(self, reply: str = '{"action":"hold","confidence":0.0}') -> None:
        self.reply = reply
        self.captured: list[Message] = []

    async def chat(self, *, model, messages, call_type, symbol=None,
                   max_tokens=350, web_search=False, **_):
        self.captured = list(messages)
        return self.reply


def _cfg(**over):
    base = SimpleNamespace(
        model_review          = "stub-review",
        model_deep            = "stub-deep",
        min_action_confidence = 0.55,
        adx_min_trade         = 18.0,
        efficiency_min_trade  = 0.35,
    )
    for k, v in over.items():
        setattr(base, k, v)
    return base


# ── position_review template ─────────────────────────────────────────────────

class TestPositionReviewTemplate:

    def test_default_threshold_in_prompt(self):
        cli = _CapturingClient()
        cfg = _cfg()
        asyncio.run(position_review.chat_position_review(
            cli, cfg, symbol="BTCUSDT", context="ctx",
        ))
        sys_msg = cli.captured[0].content
        assert "0.55" in sys_msg
        assert "{min_conf" not in sys_msg          # placeholder substituted

    def test_custom_threshold_in_prompt(self):
        cli = _CapturingClient()
        cfg = _cfg(min_action_confidence=0.70)
        asyncio.run(position_review.chat_position_review(
            cli, cfg, symbol="BTCUSDT", context="ctx",
        ))
        sys_msg = cli.captured[0].content
        assert "0.70" in sys_msg
        assert "0.55" not in sys_msg.replace("0.555", "")  # exact value moved


# ── deep_analysis template ───────────────────────────────────────────────────

class TestDeepAnalysisTemplate:

    def test_default_thresholds_in_prompt(self):
        cli = _CapturingClient(reply='{"action":"skip","confidence":0.0}')
        cfg = _cfg()
        asyncio.run(deep_analysis.chat_deep_analysis(
            cli, cfg, symbol="BTCUSDT", context="ctx",
        ))
        sys_msg = cli.captured[0].content
        assert "ADX < 18" in sys_msg
        assert "efficiency_ratio < 0.35" in sys_msg
        assert "{adx_min" not in sys_msg
        assert "{eff_min" not in sys_msg

    def test_custom_thresholds_in_prompt(self):
        cli = _CapturingClient(reply='{"action":"skip","confidence":0.0}')
        cfg = _cfg(adx_min_trade=25, efficiency_min_trade=0.50)
        asyncio.run(deep_analysis.chat_deep_analysis(
            cli, cfg, symbol="BTCUSDT", context="ctx",
        ))
        sys_msg = cli.captured[0].content
        assert "ADX < 25" in sys_msg
        assert "efficiency_ratio < 0.50" in sys_msg
