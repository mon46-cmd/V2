"""Watchlist store -- manages the 6h Grok social scan result.

The watchlist answers: ``which 5 symbols should we watch right now?``

It is backed by a JSON file so:
- Process restarts within the 6h window don't rebill the social scan.
- The scanner loop can always load the last known watchlist immediately.

The caller is responsible for calling ``refresh()`` when the watchlist expires.

Usage::

    from loops.watchlist import WatchlistStore
    from ai.schemas import SocialScan

    store = WatchlistStore(cfg)
    wl    = store.load()                  # returns SocialScan | None
    if wl is None or store.is_expired():
        wl = await call_social_scan(...)   # your AI call
        store.save(wl)

    symbols = store.symbols()             # ["BTCUSDT", "ETHUSDT", ...]
    reasons = store.reason_for("BTCUSDT") # "ETF inflows + ..."
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from ai.schemas import SocialScan, default_social_scan
from core.config import Config

log = logging.getLogger(__name__)

# Default cadence: refresh every 6 hours.
WATCHLIST_TTL_SEC = 6 * 3_600


class WatchlistStore:
    """File-backed store for the active SocialScan watchlist.

    Args:
        cfg:       Loaded Config.
        ttl_sec:   Seconds before the watchlist is considered stale.
        path:      Optional override for storage path.
                   Defaults to ``cfg.data_root / "watchlist.json"``.
    """

    def __init__(
        self,
        cfg:     Config,
        *,
        ttl_sec: int        = WATCHLIST_TTL_SEC,
        path:    Path | None = None,
    ) -> None:
        self._cfg  = cfg
        self._ttl  = ttl_sec
        self._path = path or (cfg.data_root / "watchlist.json")
        self._wl:  SocialScan | None = None
        self._saved_at: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def load(self) -> SocialScan | None:
        """Load watchlist from disk. Returns None if file missing or corrupt."""
        if not self._path.exists():
            return None
        try:
            raw  = json.loads(self._path.read_text(encoding="utf-8"))
            data = raw.get("scan") or {}
            meta = raw.get("meta") or {}
            # Reconstruct minimal SocialScan from disk.
            wl: SocialScan = {
                "scan_type":   "social_scan",
                "as_of":       str(data.get("as_of", "")),
                "market_mood": str(data.get("market_mood", "neutral")),
                "mood_score":  float(data.get("mood_score", 0.0)),
                "picks":       data.get("picks", []),
                "noise_level": str(data.get("noise_level", "medium")),
                "model_note":  str(data.get("model_note", "")),
            }
            self._wl       = wl
            self._saved_at = float(meta.get("saved_at", 0.0))
            return wl
        except Exception as exc:  # noqa: BLE001
            log.warning("watchlist load failed: %s", exc)
            return None

    def save(self, scan: SocialScan) -> None:
        """Persist a fresh SocialScan to disk."""
        self._wl       = scan
        self._saved_at = _now_ts()
        payload = {
            "meta": {
                "saved_at":   self._saved_at,
                "expires_at": self._saved_at + self._ttl,
                "ttl_sec":    self._ttl,
            },
            "scan": dict(scan),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            import os
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self._path)
            log.info(
                "watchlist saved: %d picks, mood=%s, score=%.2f",
                len(scan.get("picks", [])),
                scan.get("market_mood", "?"),
                float(scan.get("mood_score", 0)),
            )
        except OSError as exc:
            log.error("watchlist save failed: %s", exc)

    def is_expired(self) -> bool:
        """Return True if the stored watchlist is older than TTL."""
        if self._saved_at <= 0:
            # Not loaded from disk yet -- try to load.
            self.load()
        if self._saved_at <= 0:
            return True
        return (_now_ts() - self._saved_at) > self._ttl

    def age_sec(self) -> float:
        """Seconds since the watchlist was last saved (0 if never saved)."""
        if self._saved_at <= 0:
            return float("inf")
        return _now_ts() - self._saved_at

    def symbols(self) -> list[str]:
        """Return the list of watchlist symbols, ordered by conviction desc."""
        wl = self._wl
        if wl is None:
            wl = self.load()
        if wl is None:
            return []
        picks = sorted(
            wl.get("picks", []),
            key=lambda p: float(p.get("conviction", 0)),
            reverse=True,
        )
        return [str(p["symbol"]) for p in picks if p.get("symbol")]

    def direction_for(self, symbol: str) -> str | None:
        """Return ``"long"`` or ``"short"`` for a symbol, or None if not in watchlist."""
        wl = self._wl or self.load()
        if wl is None:
            return None
        for pick in wl.get("picks", []):
            if str(pick.get("symbol", "")).upper() == symbol.upper():
                return str(pick.get("direction", "long"))
        return None

    def reason_for(self, symbol: str) -> str:
        """Return the social reason string for a symbol (empty if not found)."""
        wl = self._wl or self.load()
        if wl is None:
            return ""
        for pick in wl.get("picks", []):
            if str(pick.get("symbol", "")).upper() == symbol.upper():
                return str(pick.get("reason", ""))
        return ""

    def conviction_for(self, symbol: str) -> float:
        """Return conviction score [0, 1] for a symbol (0.0 if not found)."""
        wl = self._wl or self.load()
        if wl is None:
            return 0.0
        for pick in wl.get("picks", []):
            if str(pick.get("symbol", "")).upper() == symbol.upper():
                try:
                    return float(pick.get("conviction", 0.0))
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def current(self) -> SocialScan:
        """Return the current scan, or a neutral default if none loaded."""
        wl = self._wl or self.load()
        return wl if wl is not None else default_social_scan()

    def mood(self) -> tuple[str, float]:
        """Return ``(market_mood, mood_score)`` from the current scan."""
        wl = self.current()
        return str(wl.get("market_mood", "neutral")), float(wl.get("mood_score", 0.0))


def _now_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()
