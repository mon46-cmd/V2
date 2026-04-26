"""Tests for the AI prompt-version audit hash."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ai.audit import AuditLog, _prompt_hash
from ai.http import Message, Response


def _cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(log_root=tmp_path / "logs")


def _resp() -> Response:
    return Response(
        model="m", content="{}", usage={"prompt_tokens": 100, "completion_tokens": 20},
        http_status=200, latency_ms=42, finish_reason="stop", raw_envelope={},
    )


class TestPromptVersionHash:

    def test_same_system_same_hash(self):
        a = [Message("system", "rules"), Message("user", "data-1")]
        b = [Message("system", "rules"), Message("user", "data-2")]
        assert _prompt_hash(a) == _prompt_hash(b)

    def test_changed_system_changes_hash(self):
        a = [Message("system", "rules v1"), Message("user", "x")]
        b = [Message("system", "rules v2"), Message("user", "x")]
        assert _prompt_hash(a) != _prompt_hash(b)

    def test_hash_length_and_charset(self):
        h = _prompt_hash([Message("system", "abc")])
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_no_system_messages_returns_stable_hash(self):
        a = _prompt_hash([Message("user", "x")])
        b = _prompt_hash([Message("user", "y")])
        assert a == b   # neither has system content -> identical hash


class TestAuditLogPromptVersion:

    def test_record_writes_prompt_version(self, tmp_path):
        audit = AuditLog(_cfg(tmp_path))
        msgs  = [Message("system", "system X"), Message("user", "u")]
        audit.record(
            call_id="abc", call_type="t", model="m",
            messages=msgs, response=_resp(), cost_usd=0.001,
        )
        files = list((tmp_path / "logs" / "ai_calls").glob("*.jsonl"))
        assert len(files) == 1
        line = files[0].read_text(encoding="utf-8").strip().splitlines()[-1]
        entry = json.loads(line)
        assert "prompt_version" in entry
        assert entry["prompt_version"] == _prompt_hash(msgs)

    def test_error_record_writes_prompt_version(self, tmp_path):
        audit = AuditLog(_cfg(tmp_path))
        msgs  = [Message("system", "system Y"), Message("user", "u")]
        audit.record_error(
            call_id="abc", call_type="t", model="m",
            messages=msgs, error="boom",
        )
        files = list((tmp_path / "logs" / "ai_calls").glob("*.jsonl"))
        line = files[0].read_text(encoding="utf-8").strip().splitlines()[-1]
        entry = json.loads(line)
        assert entry["prompt_version"] == _prompt_hash(msgs)
        assert entry["error"] == "boom"
