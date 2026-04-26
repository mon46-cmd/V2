"""Per-symbol trigger cooldown store.

After a trigger fires on a symbol, we suppress further fires for
``cfg.prompt_cooldown_candles`` bars.  This prevents the deep analysis
from being called 10 times on the same move.

Bypass rule: if price has moved more than ``cfg.cooldown_bypass_atr_mult * ATR``
since the last trigger, the cooldown is overridden regardless of bars elapsed.
This captures continuation moves and fast breakouts.

State is persisted atomically so restarts don't lose cooldown tracking.

Usage::

    from loops.cooldowns import CooldownStore
    from core import load_config

    cfg   = load_config()
    store = CooldownStore(path=cfg.run_root / "cooldowns.json")
    store.load()

    state = store.state_for("BTCUSDT", current_bar_ts=ts)
    # pass state to detect_trigger()

    if trigger_fired:
        store.record("BTCUSDT", bar_ts=ts, close=67400.0)
        store.save()
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CooldownState:
    """Snapshot of one symbol's cooldown.

    ``bars_since`` is the number of 15m bars since the last trigger.
    It starts at a very large number (999) for symbols that never fired,
    so the first trigger always passes the cooldown gate.
    """
    last_bar_ts:  pd.Timestamp | None = None
    last_close:   float | None        = None
    bars_since:   int                 = 999   # "infinite" for fresh symbols


class CooldownStore:
    """Thread-safe (single-process) per-symbol cooldown state.

    The store tracks, for each symbol:
    - The bar timestamp when the last trigger fired.
    - The close price at that bar.
    - How many 15m bars have elapsed since then.

    Args:
        path:              Path to the JSON state file.
        bar_duration_sec:  Bar length in seconds (default 900 = 15m).
    """

    def __init__(
        self,
        path:             Path,
        *,
        bar_duration_sec: int = 900,
    ) -> None:
        self._path     = path
        self._bar_sec  = bar_duration_sec
        self._states:  dict[str, dict] = {}   # symbol -> {last_bar_ts, last_close}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load persisted state from disk (best-effort, silent on missing/corrupt)."""
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._states = {str(k): v for k, v in raw.items()}
            log.debug("cooldowns loaded: %d symbols", len(self._states))
        except Exception as exc:  # noqa: BLE001
            log.warning("cooldowns load failed: %s", exc)

    def save(self) -> None:
        """Atomically persist state to disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._states, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            log.error("cooldowns save failed: %s", exc)

    def state_for(self, symbol: str, *, current_bar_ts: pd.Timestamp | None = None) -> CooldownState:
        """Return the current CooldownState for a symbol.

        Args:
            symbol:          Ticker symbol.
            current_bar_ts:  Timestamp of the bar being evaluated (used to
                             compute ``bars_since`` dynamically).
        """
        rec = self._states.get(symbol)
        if rec is None:
            return CooldownState()

        last_ts_raw = rec.get("last_bar_ts")
        last_close  = rec.get("last_close")

        last_ts: pd.Timestamp | None = None
        if last_ts_raw is not None:
            try:
                last_ts = pd.Timestamp(last_ts_raw, tz="UTC")
            except Exception:  # noqa: BLE001
                last_ts = None

        bars_since = 999
        if last_ts is not None and current_bar_ts is not None:
            try:
                delta_sec  = (current_bar_ts - last_ts).total_seconds()
                bars_since = max(0, int(delta_sec // self._bar_sec))
            except Exception:  # noqa: BLE001
                bars_since = 999

        return CooldownState(
            last_bar_ts = last_ts,
            last_close  = float(last_close) if last_close is not None else None,
            bars_since  = bars_since,
        )

    def record(self, symbol: str, *, bar_ts: pd.Timestamp, close: float) -> None:
        """Record a trigger fire for ``symbol``.  Call AFTER trigger is confirmed fired.

        Args:
            symbol: Ticker symbol.
            bar_ts: Bar timestamp of the trigger.
            close:  Close price at the trigger bar.
        """
        self._states[str(symbol)] = {
            "last_bar_ts": bar_ts.isoformat() if hasattr(bar_ts, "isoformat") else str(bar_ts),
            "last_close":  float(close),
        }

    def reset(self, symbol: str) -> None:
        """Clear cooldown for a symbol (e.g. if the position was closed)."""
        self._states.pop(str(symbol), None)

    def reset_all(self) -> None:
        """Clear all cooldowns."""
        self._states.clear()

    def summary(self) -> dict[str, int]:
        """Return ``{symbol: bars_since}`` for all tracked symbols."""
        now = pd.Timestamp.now(tz="UTC")
        return {sym: self.state_for(sym, current_bar_ts=now).bars_since
                for sym in self._states}
