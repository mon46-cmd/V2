"""Daily USD spend tracker.

State is persisted as JSON so a process restart does not reset the counter.
The tracker is process-local (protected by a threading Lock) -- do not
share state files across multiple processes.

Day boundaries are UTC.  The first call of a new UTC day automatically
resets the counter and writes the fresh state.

Usage::

    from ai.budget import BudgetTracker
    from pathlib import Path

    budget = BudgetTracker(daily_cap_usd=1.00, state_path=Path("data/budget.json"))

    if budget.can_afford(0.002):
        budget.charge(actual_cost)

    print(f"Spent: ${budget.spent_usd:.4f}  Remaining: ${budget.remaining_usd:.4f}")
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class BudgetTracker:
    """Thread-safe daily USD budget with optional file persistence.

    Args:
        daily_cap_usd: Maximum USD to spend in a UTC calendar day.
        state_path:    Optional path to persist state across restarts.
                       If None, state resets on every process start.
    """

    def __init__(
        self,
        daily_cap_usd: float,
        state_path:    Path | None = None,
    ) -> None:
        self._cap       = float(daily_cap_usd)
        self._path      = state_path
        self._lock      = Lock()
        self._day       = _utc_day()
        self._spent_usd = 0.0
        self._load()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def daily_cap_usd(self) -> float:
        return self._cap

    @property
    def spent_usd(self) -> float:
        with self._lock:
            self._maybe_roll()
            return self._spent_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self._cap - self.spent_usd)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_afford(self, estimated_cost_usd: float) -> bool:
        """Return True if the estimated cost fits within the remaining budget."""
        return estimated_cost_usd <= self.remaining_usd + 1e-9

    def charge(self, cost_usd: float) -> float:
        """Add ``cost_usd`` to the running total.

        Returns the new cumulative spend for the day.
        """
        with self._lock:
            self._maybe_roll()
            self._spent_usd += max(0.0, float(cost_usd))
            self._save()
            return self._spent_usd

    def reset(self) -> None:
        """Manually reset the counter (e.g. for testing)."""
        with self._lock:
            self._spent_usd = 0.0
            self._day       = _utc_day()
            self._save()

    def summary(self) -> dict:
        """Return a dict suitable for logging."""
        return {
            "day":           self._day,
            "cap_usd":       self._cap,
            "spent_usd":     round(self._spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_roll(self) -> None:
        today = _utc_day()
        if today != self._day:
            self._day       = today
            self._spent_usd = 0.0
            self._save()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            if data.get("day") == self._day:
                self._spent_usd = float(data.get("spent_usd", 0.0))
        except Exception:  # noqa: BLE001
            # Corrupt file -- start fresh without crashing.
            self._spent_usd = 0.0

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "day":       self._day,
            "spent_usd": round(self._spent_usd, 8),
            "cap_usd":   self._cap,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise


def _utc_day() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
