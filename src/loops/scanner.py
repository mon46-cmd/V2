"""Main async scanner loop for the V8 paper trading engine.

Architecture
------------
One coroutine runs perpetually, sleeping until the next 15m bar close.
On each tick it:

  1. Checks if the social scan watchlist needs refreshing (6h TTL).
  2. For each watchlist symbol:
     a. Downloads the latest 15m klines + computes feature snapshot.
     b. Runs detect_trigger() — fires if any configured Tier-A flag is set.
     c. If triggered → builds context → calls chat_deep_analysis.
     d. If AI says LONG/SHORT → sizes position → opens via PaperBroker.
  3. For each *open* paper position (every bar):
     a. Calls broker.update_bar() → SL/TP/trailing stop.
     b. Every N bars → calls chat_position_review → acts on CLOSE.

Usage::

    python -m loops.runner           # preferred
    from loops.scanner import Scanner
    asyncio.run(Scanner(cfg).run())
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import aiohttp
import pandas as pd

from ai.audit import AuditLog
from ai.budget import BudgetTracker
from ai.calls.context_builder import build_deep_context, build_review_context
from ai.calls.deep_analysis import chat_deep_analysis
from ai.calls.position_review import chat_position_review
from ai.calls.social_scan import chat_social_scan
from ai.client import AIClient
from core.config import Config
from downloader.http import HttpClient
from downloader.rest import RestClient
from features.config import FeatureConfig
from features.pipeline import compute_snapshot
from loops.cooldowns import CooldownStore
from loops.triggers import detect_trigger
from loops.universe import UniverseBuilder
from loops.watchlist import WatchlistStore
from portfolio.mechanical_review import mechanical_review
from portfolio.paper_broker import PaperBroker
from portfolio.risk import RiskEngine

log = logging.getLogger(__name__)


def _drop_open_bar(snap: pd.DataFrame, *, interval_min: int = 15) -> pd.DataFrame:
    """Remove the last row when it represents an in-progress (unclosed) bar.

    A 15m candle with index timestamp ``T`` closes at ``T + 15min``. If
    ``now < T + 15min`` the candle is still updating and using its values
    for an entry decision is a forward-looking bias (the indicators see
    a partial bar but the trade will fill at the close). Dropping it
    enforces a strict 1-bar lag, identical to the discipline used in
    proper backtests.

    Returns the (possibly trimmed) DataFrame.
    """
    if snap is None or snap.empty:
        return snap
    last_ts = snap.index[-1]
    if hasattr(last_ts, "to_pydatetime"):
        last_ts = last_ts.to_pydatetime()
    if getattr(last_ts, "tzinfo", None) is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)
    close_at = last_ts + timedelta(minutes=interval_min)
    now = datetime.now(tz=timezone.utc)
    if now < close_at:
        return snap.iloc[:-1]
    return snap


class Scanner:
    """V8 autonomous paper trading scanner."""

    def __init__(self, cfg: Config) -> None:
        self._cfg        = cfg
        self._feat       = FeatureConfig()
        self._watchlist  = WatchlistStore(cfg)
        self._cooldowns  = CooldownStore(cfg.run_root / "cooldowns.json")
        self._broker     = PaperBroker(cfg)
        self._risk       = RiskEngine(cfg)
        self._review_cnt: dict[str, int] = {}   # symbol → bars since last review
        # Persistent across ticks: budget cap survives restarts via state file,
        # audit log appends every AI call to data/runs/audit.jsonl.
        self._budget     = BudgetTracker(
            daily_cap_usd=cfg.daily_budget_usd,
            state_path=cfg.run_root / "budget.json",
        )
        self._audit      = AuditLog(cfg)

    # ── Entry ──────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Run perpetually, ticking every 15m bar."""
        log.info("Scanner starting — equity=$%.2f", self._broker.equity)
        self._cooldowns.load()
        self._broker.load()

        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                log.info("Scanner stopped — saving state")
                self._save()
                raise
            except Exception as exc:  # noqa: BLE001
                log.error("tick error: %s", exc, exc_info=True)
            await self._sleep_until_next_bar()

    # ── Tick ───────────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        now = datetime.now(tz=timezone.utc)
        log.info("── TICK %s  open=%d  equity=$%.2f",
                 now.strftime("%H:%M UTC"), self._broker.n_open, self._broker.equity)

        async with HttpClient(self._cfg) as http:
            rest    = RestClient(http, self._cfg)
            # Dedicated aiohttp session for OpenRouter calls. Kept separate
            # from the exchange HttpClient so connection pools / timeouts
            # don't interfere. Skipped entirely when running in offline /
            # dry-run mode (AIClient handles that itself).
            ai_timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=ai_timeout) as ai_session:
                ai = AIClient(
                    self._cfg,
                    session=ai_session,
                    budget=self._budget,
                    audit=self._audit,
                )
                builder = UniverseBuilder(self._cfg, rest)

                # 1. Open position updates (run first — SL/TP may fire on this bar)
                for sym in list(self._broker.open_symbols):
                    await self._process_open(sym, rest=rest, ai=ai)

                # 2. Maybe refresh watchlist
                await self._maybe_social_scan(rest=rest, ai=ai, builder=builder)

                watchlist = self._watchlist.symbols()
                if not watchlist:
                    log.warning("watchlist empty — nothing to scan")
                    return

                # 3. Trigger scan on watchlist symbols (skip already-open ones)
                if self._broker.is_halted():
                    log.warning(
                        "DRAWDOWN HALT active (dd=%.2f%%) -- skipping new entry scan; existing positions still managed",
                        self._broker.drawdown_pct * 100,
                    )
                else:
                    for sym in watchlist:
                        if sym in self._broker.open_symbols:
                            continue
                        await self._scan(sym, rest=rest, ai=ai)

        self._save()

    # ── Social scan ────────────────────────────────────────────────────────

    async def _maybe_social_scan(
        self,
        rest: RestClient,
        ai: AIClient,
        builder: UniverseBuilder,
    ) -> None:
        if not self._watchlist.is_expired():
            log.debug("watchlist fresh (%.0fm old)", self._watchlist.age_sec() / 60)
            return

        log.info("watchlist expired — running social scan")
        uni_df = await builder.get_df(force=False)

        # Build universe rows for social prompt
        universe_rows: list[dict] = []
        if uni_df is not None and not uni_df.empty:
            universe_rows = uni_df.to_dict(orient="records")

        pinned = list(self._broker.open_symbols)   # keep open positions in watchlist

        scan = await chat_social_scan(
            ai, self._cfg,
            universe_rows  = universe_rows,
            pinned_symbols = pinned,
        )
        self._watchlist.save(scan)
        log.info("social scan: %d picks  mood=%s", len(scan.get("picks", [])), scan.get("market_mood"))

    # ── Trigger scan ───────────────────────────────────────────────────────

    async def _scan(self, symbol: str, *, rest: RestClient, ai: AIClient) -> None:
        """Feature computation + trigger gate + deep analysis for one symbol."""
        try:
            snap = await self._snapshot(symbol, rest)
            if snap is None or snap.empty:
                return

            # 1-bar lag: drop the in-progress bar so entry decisions are
            # made on a *closed* candle.  Position management (_process_open)
            # still uses the live tail because trailing stops & SL hits
            # need real-time price.
            snap = _drop_open_bar(snap, interval_min=15)
            if snap.empty:
                return

            bar_ts = snap.index[-1]
            last   = snap.iloc[-1]
            bar_d  = {**last.to_dict(), "timestamp": bar_ts}

            state = self._cooldowns.state_for(symbol, current_bar_ts=bar_ts)
            dec   = detect_trigger(symbol=symbol, bar=bar_d, state=state, cfg=self._cfg)

            log.debug("trigger %s → %s  flag=%s", symbol, dec.decision, dec.flag)
            if not dec.fired:
                return

            # Record cooldown immediately (before AI call) to avoid double-fire
            self._cooldowns.record(symbol, bar_ts=dec.bar_ts, close=dec.close)
            log.info("TRIGGER %-10s  flag=%-28s  close=%.4f", symbol, dec.flag, dec.close)

            # Build context + call AI
            context = build_deep_context(
                row                 = last.to_dict(),
                symbol              = symbol,
                as_of               = bar_ts.isoformat(),
                close_price         = float(last.get("close", dec.close)),
                social_reason       = self._watchlist.reason_for(symbol),
                watchlist_direction = self._watchlist.direction_for(symbol) or "",
                ohlcv_df            = snap.reset_index(),
                fired_flag          = dec.flag or "",
            )

            result = await chat_deep_analysis(ai, self._cfg, symbol=symbol, context=context)

            action = str(result.get("action", "skip")).upper()
            conf   = float(result.get("confidence", 0))
            log.info("AI %s → %s  conf=%.2f  horizon=%s", symbol, action, conf, result.get("time_horizon"))

            if action in ("LONG", "SHORT"):
                side = "long" if action == "LONG" else "short"
                atr_pct = float(last.get("atr_14_pct", 1.0))
                atr = self._risk.atr_from_pct(dec.close, atr_pct)
                # Vol-regime score: current ATR% vs its rolling median.
                # >1 = elevated vol -> RiskEngine will downscale size.
                try:
                    atr_series = snap["atr_14_pct"].dropna()
                    median_atr = float(atr_series.tail(60).median())
                    vol_score  = (atr_pct / median_atr) if median_atr > 0 else 1.0
                except Exception:  # noqa: BLE001
                    vol_score = 1.0
                sizing = self._risk.compute(
                    side             = side,
                    entry            = dec.close,
                    atr              = atr,
                    equity_usd       = self._broker.equity,
                    n_open           = self._broker.n_open,
                    loser_streak     = self._broker.loser_streak,
                    vol_regime_score = vol_score,
                )
                self._broker.open(symbol, sizing=sizing)

        except Exception as exc:  # noqa: BLE001
            log.error("scan %s error: %s", symbol, exc, exc_info=True)

    # ── Open position management ───────────────────────────────────────────

    async def _process_open(self, symbol: str, *, rest: RestClient, ai: AIClient) -> None:
        """Update bar + optionally run position review for an open position."""
        try:
            snap = await self._snapshot(symbol, rest, bars=20)
            if snap is None or snap.empty:
                return

            last    = snap.iloc[-1]
            atr_pct = float(last.get("atr_14_pct", 1.0))
            pos     = self._broker.get(symbol)
            atr     = self._risk.atr_from_pct(float(last.get("close", pos.entry if pos else 1)), atr_pct)

            closed = self._broker.update_bar(
                symbol = symbol,
                high   = float(last.get("high", 0)),
                low    = float(last.get("low",  0)),
                close  = float(last.get("close", 0)),
                atr    = atr,
            )
            for trade in closed:
                log.info("CLOSED %s  pnl=$%.2f  rr=%.2fR  reason=%s",
                         trade.symbol, trade.pnl_usd, trade.risk_reward, trade.exit_reason)
                self._cooldowns.reset(trade.symbol)
                self._review_cnt.pop(symbol, None)

            if symbol not in self._broker.open_symbols:
                return

            # Position review cadence
            review_every = int(getattr(self._cfg, "review_every_bars", 4))
            cnt = self._review_cnt.get(symbol, 0) + 1
            self._review_cnt[symbol] = cnt
            if cnt < review_every:
                return

            self._review_cnt[symbol] = 0
            pos = self._broker.get(symbol)
            if pos is None:
                return

            context = build_review_context(
                symbol        = symbol,
                side          = pos.side,
                entry         = pos.entry,
                stop_loss     = pos.stop_loss,
                tp1           = pos.tp1,
                tp2           = pos.tp2,
                mark          = float(last.get("close", pos.entry)),
                remaining_pct = 50.0 if pos.tp1_hit else 100.0,
                bars_held     = cnt,
                loser_streak  = self._broker.loser_streak,
                hook_reason   = "review_cadence",
                row           = last.to_dict(),
                ohlcv_df      = snap.reset_index(),
            )
            review = await chat_position_review(ai, self._cfg, symbol=symbol, context=context)
            action = str(review.get("action", "HOLD")).upper()
            log.info("review %s → %s", symbol, action)

            # Mechanical fallback: if the AI returned the synthetic-default
            # "hold" (signalled by confidence == 0.0 and the safe-default
            # rationale), apply deterministic risk rules instead of
            # drifting blindly while the AI is unavailable.
            if action == "HOLD" and float(review.get("confidence", 0.0)) <= 0.0:
                fb = mechanical_review(
                    symbol     = symbol,
                    position   = pos,
                    mark_price = float(last.get("close", pos.entry)),
                    atr        = atr,
                    bars_held  = cnt,
                )
                fb_action = fb["action"].upper()
                if fb_action != "HOLD":
                    log.info(
                        "mechanical_fallback %s → %s  (%s)",
                        symbol, fb_action, fb.get("rationale", ""),
                    )
                    review = fb
                    action = fb_action

            if action == "EXIT" or action == "CLOSE":
                close_px = float(last.get("close", pos.entry))
                self._broker.ai_close(symbol, exit_price=close_px)
            elif action == "TIGHTEN_STOP":
                new_sl = float(review.get("new_stop_loss", 0.0))
                if new_sl > 0:
                    self._broker.tighten_stop(symbol, new_stop=new_sl)

        except Exception as exc:  # noqa: BLE001
            log.error("process_open %s error: %s", symbol, exc, exc_info=True)

    # ── Feature snapshot ──────────────────────────────────────────────────

    async def _snapshot(self, symbol: str, rest: RestClient, bars: int = 320) -> pd.DataFrame | None:
        """Fetch klines and compute full feature snapshot."""
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(minutes=15 * (bars + 20))
        try:
            ohlcv = await rest.klines(symbol, "15", start, end)
            if ohlcv.empty or len(ohlcv) < 60:
                log.warning("snapshot %s: only %d bars", symbol, len(ohlcv))
                return None
            # Bar freshness guard: skip if the latest bar is too old (stale data).
            max_age = int(getattr(self._cfg, "bar_max_age_sec", 0) or 0)
            if max_age > 0:
                last_ts = ohlcv["timestamp"].iloc[-1]
                if hasattr(last_ts, "to_pydatetime"):
                    last_ts = last_ts.to_pydatetime()
                if getattr(last_ts, "tzinfo", None) is None:
                    last_ts = last_ts.replace(tzinfo=timezone.utc)
                age = (end - last_ts).total_seconds()
                if age > max_age:
                    log.warning(
                        "snapshot %s: last bar is stale (age=%.0fs > max=%ds) -- skipping",
                        symbol, age, max_age,
                    )
                    return None
            return compute_snapshot(ohlcv.set_index("timestamp"), self._feat)
        except Exception as exc:  # noqa: BLE001
            log.error("snapshot %s: %s", symbol, exc)
            return None

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self) -> None:
        self._broker.save()
        self._cooldowns.save()

    # ── Timing ────────────────────────────────────────────────────────────

    @staticmethod
    async def _sleep_until_next_bar(buffer_sec: int = 5) -> None:
        now      = datetime.now(tz=timezone.utc)
        elapsed  = (now.minute % 15) * 60 + now.second
        wait_sec = (15 * 60 - elapsed) + buffer_sec
        log.info("next bar in %.0fs  (%s)",
                 wait_sec,
                 (now + timedelta(seconds=wait_sec)).strftime("%H:%M:%S UTC"))
        await asyncio.sleep(max(wait_sec, 1))
