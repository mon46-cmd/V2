"""Tests for the AI communication module.

All tests are pure (no network calls, no API key required).
They test:
  - MockClient returns a well-formed Response
  - AIClient uses the mock when ai_dry_run=True
  - AIClient uses the mock when api_key is empty
  - AuditLog writes a JSONL record with correct fields
  - BudgetTracker: charges, rolls, and persists correctly
  - BudgetExhaustedError is raised when cap is exceeded
  - estimate_cost returns non-negative values for known models

Run with::

    pytest tests/test_ai.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.audit  import AuditLog
from ai.budget import BudgetTracker
from ai.client import AIClient, BudgetExhaustedError
from ai        import estimate_cost            # re-exported from ai/__init__.py
from ai.http   import Message, Response
from ai.mock   import MockClient
from core.config import Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides) -> Config:
    """Build a minimal Config with safe defaults."""
    base = dict(
        openrouter_api_key  = "",   # no key -> offline by default
        openrouter_base_url = "https://openrouter.ai/api/v1",
        openrouter_referer  = "",
        openrouter_title    = "",
        model_deep          = "openai/gpt-4o-mini",
        model_watchlist     = "openai/gpt-4o-mini",
        model_review        = "openai/gpt-4o-mini",
        ai_dry_run          = True,
        ai_timeout_sec      = 30.0,
        log_root            = Path(tempfile.mkdtemp()),
        run_root            = Path(tempfile.mkdtemp()),
    )
    base.update(overrides)
    # Config is frozen -- use __dataclass_fields__ (not hasattr) because fields
    # with default_factory don't appear as class attributes.
    valid_fields = Config.__dataclass_fields__
    return Config(**{k: v for k, v in base.items() if k in valid_fields})


def _make_response(content: str = '{"ok": true}') -> Response:
    return Response(
        model         = "openai/gpt-4o-mini",
        content       = content,
        usage         = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70},
        http_status   = 200,
        latency_ms    = 42,
        finish_reason = "stop",
        raw_envelope  = {},
    )


# ===========================================================================
# MockClient
# ===========================================================================

class TestMockClient:
    def test_returns_response(self):
        mock = MockClient()
        resp = mock.chat("openai/gpt-4o-mini", [Message("user", "hello")])
        assert isinstance(resp, Response)

    def test_content_is_json_string(self):
        mock = MockClient()
        resp = mock.chat("openai/gpt-4o-mini", [Message("user", "hello")])
        # Default content must be valid JSON.
        parsed = json.loads(resp.content)
        assert isinstance(parsed, dict)

    def test_custom_content(self):
        payload = '{"signal": "long"}'
        mock    = MockClient(default_content=payload)
        resp    = mock.chat("openai/gpt-4o-mini", [Message("user", "hello")])
        assert resp.content == payload

    def test_zero_latency(self):
        mock = MockClient()
        resp = mock.chat("openai/gpt-4o-mini", [Message("user", "hello")])
        assert resp.latency_ms == 0

    def test_http_status_200(self):
        mock = MockClient()
        resp = mock.chat("openai/gpt-4o-mini", [Message("user", "hello")])
        assert resp.http_status == 200


# ===========================================================================
# AIClient (offline / dry-run)
# ===========================================================================

class TestAIClientOffline:
    @pytest.mark.asyncio
    async def test_chat_returns_string_in_dry_run(self):
        cfg    = _make_config(ai_dry_run=True)
        client = AIClient(cfg)
        reply  = await client.chat(
            model    = "openai/gpt-4o-mini",
            messages = [Message("system", "Be concise."), Message("user", "Hello")],
        )
        assert isinstance(reply, str)
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_chat_uses_mock_when_no_api_key(self):
        cfg    = _make_config(ai_dry_run=False, openrouter_api_key="")
        client = AIClient(cfg)
        assert client._offline is True
        reply  = await client.chat(
            model    = "openai/gpt-4o-mini",
            messages = [Message("user", "hi")],
        )
        assert isinstance(reply, str)

    @pytest.mark.asyncio
    async def test_kill_switch_forces_offline(self, monkeypatch):
        monkeypatch.setenv("AI_KILL_SWITCH", "1")
        cfg    = _make_config(ai_dry_run=False, openrouter_api_key="sk-real-key")
        client = AIClient(cfg)
        assert client._offline is True

    @pytest.mark.asyncio
    async def test_call_type_and_symbol_dont_break(self):
        cfg    = _make_config()
        client = AIClient(cfg)
        reply  = await client.chat(
            model     = "openai/gpt-4o-mini",
            messages  = [Message("user", "test")],
            call_type = "analysis",
            symbol    = "BTCUSDT",
        )
        assert isinstance(reply, str)

    @pytest.mark.asyncio
    async def test_budget_exhausted_error_raised(self):
        cfg    = _make_config(ai_dry_run=False, openrouter_api_key="sk-real-key")
        budget = BudgetTracker(daily_cap_usd=0.0)  # zero cap
        import aiohttp
        session = MagicMock(spec=aiohttp.ClientSession)
        client  = AIClient(cfg, session=session, budget=budget)
        # Force offline off so budget check runs.
        client._offline = False

        with pytest.raises(BudgetExhaustedError):
            await client.chat(
                model    = "openai/gpt-4o-mini",
                messages = [Message("user", "a" * 400)],  # large enough to cost something
            )

    @pytest.mark.asyncio
    async def test_audit_record_written(self, tmp_path):
        cfg    = _make_config(ai_dry_run=True, log_root=tmp_path)
        audit  = AuditLog(cfg)
        client = AIClient(cfg, audit=audit)
        await client.chat(
            model     = "openai/gpt-4o-mini",
            messages  = [Message("user", "test")],
            call_type = "test",
            symbol    = "ETHUSDT",
        )
        # Find the JSONL file that was written.
        jsonl_files = list((tmp_path / "ai_calls").glob("*.jsonl"))
        assert len(jsonl_files) == 1, "expected exactly one JSONL file"
        line  = jsonl_files[0].read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["call_type"] == "test"
        assert record["symbol"]    == "ETHUSDT"
        assert record["dry_run"]   is True
        assert "messages"  in record
        assert "response"  in record
        assert "cost_usd"  in record


# ===========================================================================
# AuditLog
# ===========================================================================

class TestAuditLog:
    def test_creates_dir(self, tmp_path):
        cfg = _make_config(log_root=tmp_path / "logs")
        AuditLog(cfg)
        assert (tmp_path / "logs" / "ai_calls").is_dir()

    def test_record_writes_valid_json(self, tmp_path):
        cfg   = _make_config(log_root=tmp_path)
        audit = AuditLog(cfg)
        msgs  = [Message("system", "Be helpful."), Message("user", "Hi")]
        resp  = _make_response()
        audit.record("abc123", "test", "openai/gpt-4o-mini", msgs, resp, cost_usd=0.0001)
        lines = (tmp_path / "ai_calls").glob("*.jsonl")
        records = [json.loads(l) for f in lines for l in f.read_text().splitlines() if l.strip()]
        assert len(records) == 1
        r = records[0]
        assert r["call_id"]    == "abc123"
        assert r["call_type"]  == "test"
        assert r["error"]      is None
        assert len(r["messages"]) == 2

    def test_record_error_writes_valid_json(self, tmp_path):
        cfg   = _make_config(log_root=tmp_path)
        audit = AuditLog(cfg)
        msgs  = [Message("user", "crash")]
        audit.record_error("err001", "analysis", "openai/gpt-4o-mini", msgs, "TimeoutError()")
        lines   = list((tmp_path / "ai_calls").glob("*.jsonl"))
        record  = json.loads(lines[0].read_text().strip())
        assert record["error"]    == "TimeoutError()"
        assert record["response"] is None

    def test_multiple_records_accumulate(self, tmp_path):
        cfg   = _make_config(log_root=tmp_path)
        audit = AuditLog(cfg)
        for i in range(3):
            audit.record(f"id{i}", "test", "openai/gpt-4o-mini",
                         [Message("user", "hi")], _make_response(), cost_usd=0.0)
        lines = list((tmp_path / "ai_calls").glob("*.jsonl"))
        records = [json.loads(l) for l in lines[0].read_text().splitlines() if l.strip()]
        assert len(records) == 3


# ===========================================================================
# BudgetTracker
# ===========================================================================

class TestBudgetTracker:
    def test_initial_state(self):
        b = BudgetTracker(daily_cap_usd=2.0)
        assert b.spent_usd     == 0.0
        assert b.remaining_usd == 2.0

    def test_can_afford_within_cap(self):
        b = BudgetTracker(daily_cap_usd=1.0)
        assert b.can_afford(0.5) is True

    def test_can_afford_exceeds_cap(self):
        b = BudgetTracker(daily_cap_usd=0.10)
        assert b.can_afford(0.20) is False

    def test_charge_updates_spent(self):
        b = BudgetTracker(daily_cap_usd=1.0)
        b.charge(0.30)
        assert abs(b.spent_usd - 0.30) < 1e-9

    def test_charge_returns_cumulative(self):
        b = BudgetTracker(daily_cap_usd=1.0)
        b.charge(0.10)
        total = b.charge(0.20)
        assert abs(total - 0.30) < 1e-9

    def test_remaining_decreases(self):
        b = BudgetTracker(daily_cap_usd=1.0)
        b.charge(0.25)
        assert abs(b.remaining_usd - 0.75) < 1e-9

    def test_remaining_floor_at_zero(self):
        b = BudgetTracker(daily_cap_usd=0.10)
        b.charge(1.00)
        assert b.remaining_usd == 0.0

    def test_reset(self):
        b = BudgetTracker(daily_cap_usd=1.0)
        b.charge(0.50)
        b.reset()
        assert b.spent_usd == 0.0

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "budget.json"
        b1   = BudgetTracker(daily_cap_usd=1.0, state_path=path)
        b1.charge(0.33)
        # New instance should reload from file.
        b2   = BudgetTracker(daily_cap_usd=1.0, state_path=path)
        assert abs(b2.spent_usd - 0.33) < 1e-6

    def test_summary_dict(self):
        b = BudgetTracker(daily_cap_usd=1.0)
        b.charge(0.42)
        s = b.summary()
        assert "spent_usd"     in s
        assert "remaining_usd" in s
        assert "cap_usd"       in s


# ===========================================================================
# estimate_cost
# ===========================================================================

class TestEstimateCost:
    def test_known_model_positive(self):
        cost = estimate_cost("openai/gpt-4o-mini", 500, 100)
        assert cost > 0.0

    def test_zero_tokens_zero_cost(self):
        assert estimate_cost("openai/gpt-4o-mini", 0, 0) == 0.0

    def test_unknown_model_fallback(self):
        cost = estimate_cost("some/unknown-model-xyz", 1000, 500)
        assert cost > 0.0

    def test_gpt4_more_expensive_than_mini(self):
        mini = estimate_cost("openai/gpt-4o-mini", 1000, 1000)
        gpt4 = estimate_cost("openai/gpt-4o",      1000, 1000)
        assert gpt4 > mini
