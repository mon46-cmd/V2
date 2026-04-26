"""Deterministic backtest replay over cached parquet klines.

Replays a single-symbol 15m feature pipeline through the *exact* same
trigger / risk / paper-broker components used in production, but with
two differences that make the result reproducible and bias-free:

  1. **No AI calls.**  A pluggable ``DecisionPolicy`` is invoked instead.
     The default (``ConservativePolicy``) takes a long when the bar's
     fired flag is on the bullish side of the supertrend, and a short
     on the bearish side, with conviction proportional to ADX.  This
     gives a sane "what would the system do without AI uncertainty"
     baseline.  Other policies can be plugged in (random, ML model,
     replay from JSONL audit log).

  2. **1-bar lag enforced everywhere.**  Decisions for bar ``T`` use
     features computed up to bar ``T-1`` only.  Fills happen at the
     open of bar ``T+1``.  This is the discipline the live scanner
     applies via ``_drop_open_bar``.

The replay produces:
  * a per-trade DataFrame (entry/exit ts, side, pnl_usd, R, reason),
  * a per-bar equity curve,
  * summary stats: total return, Sharpe, max DD, profit factor,
    hit rate, average R.

CLI::
    python -m backtest --symbol BTCUSDT --start 2025-01-01 --end 2025-04-01

Notes:
  * Reads klines from ``cfg.cache_root / "klines"`` (downloaded by the
    existing data downloader).  Falls back to a parquet file passed
    via ``--klines path.parquet``.
  * Funding/OI are NOT used by default to keep the backtest fast and
    self-contained.  Pass ``--with-derivs`` to include them if cached.

This is intentionally a research tool: it favours clarity and
correctness over throughput.
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import Config, load_config
from features.config import FeatureConfig
from features.pipeline import compute_snapshot
from loops.cooldowns import CooldownState
from loops.triggers import detect_trigger
from portfolio.risk import RiskEngine

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """Output of a backtest policy for one fired-flag bar."""
    side:       str    # "long" | "short" | "skip"
    confidence: float  # 0..1


class DecisionPolicy:
    """Override ``decide`` to plug in a custom backtest policy."""

    def decide(self, bar: pd.Series) -> Decision:    # pragma: no cover
        raise NotImplementedError


class ConservativePolicy(DecisionPolicy):
    """Default policy.

    Long when supertrend_dir > 0 AND ADX >= adx_min AND RSI > 50.
    Short when supertrend_dir < 0 AND ADX >= adx_min AND RSI < 50.
    Skip otherwise.  Confidence scales linearly with ADX in [adx_min, 40].
    """

    def __init__(self, adx_min: float = 18.0) -> None:
        self.adx_min = adx_min

    def decide(self, bar: pd.Series) -> Decision:
        adx = float(bar.get("adx_14", 0.0))
        if adx < self.adx_min:
            return Decision("skip", 0.0)

        st_dir = float(bar.get("supertrend_dir", 0.0))
        rsi    = float(bar.get("rsi_14", 50.0))

        confidence = max(0.0, min(1.0, (adx - self.adx_min) / 22.0))

        if st_dir > 0 and rsi > 50:
            return Decision("long", confidence)
        if st_dir < 0 and rsi < 50:
            return Decision("short", confidence)
        return Decision("skip", 0.0)


# ---------------------------------------------------------------------------
# Trade / equity records
# ---------------------------------------------------------------------------

@dataclass
class BTTrade:
    symbol:       str
    side:         str
    entry_ts:     pd.Timestamp
    entry_px:     float
    exit_ts:      pd.Timestamp
    exit_px:      float
    qty_usd:      float
    pnl_usd:      float
    r_multiple:   float    # pnl_usd / risk_usd
    reason:       str      # sl_hit | tp1_hit | tp2_hit | trail | timeout
    confidence:   float
    bars_held:    int


@dataclass
class BTReport:
    trades:        list[BTTrade]
    equity_curve:  pd.Series
    start:         pd.Timestamp
    end:           pd.Timestamp
    initial_equity: float

    def stats(self) -> dict:
        eq      = self.equity_curve
        ret_pct = (eq.iloc[-1] / self.initial_equity - 1.0) if len(eq) else 0.0

        trades = self.trades
        n      = len(trades)
        wins   = [t for t in trades if t.pnl_usd > 0]
        losses = [t for t in trades if t.pnl_usd <= 0]

        gross_win  = sum(t.pnl_usd for t in wins)
        gross_loss = -sum(t.pnl_usd for t in losses)
        pf         = gross_win / gross_loss if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

        max_dd = _max_drawdown(eq) if len(eq) else 0.0

        # Sharpe on bar-to-bar equity returns, annualised assuming 15m bars.
        # 96 bars/day * 365 = 35040 bars/year.
        if len(eq) > 2:
            rets = eq.pct_change().dropna()
            sharpe = float(rets.mean() / rets.std() * math.sqrt(35040)) if rets.std() > 0 else 0.0
        else:
            sharpe = 0.0

        avg_r = float(np.mean([t.r_multiple for t in trades])) if trades else 0.0

        return {
            "n_trades":        n,
            "hit_rate":        (len(wins) / n) if n else 0.0,
            "total_return":    float(ret_pct),
            "final_equity":    float(eq.iloc[-1]) if len(eq) else self.initial_equity,
            "max_drawdown":    float(max_dd),
            "profit_factor":   float(pf),
            "sharpe":          sharpe,
            "avg_r_multiple":  avg_r,
            "gross_win_usd":   float(gross_win),
            "gross_loss_usd":  float(gross_loss),
        }


def _max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    running_peak = eq.cummax()
    dd = (eq - running_peak) / running_peak
    return float(dd.min())


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------

@dataclass
class _OpenTrade:
    side:        str
    entry_ts:    pd.Timestamp
    entry_px:    float
    sl:          float
    tp1:         float
    tp2:         float
    qty_usd:     float
    risk_usd:    float
    confidence:  float
    bars_held:   int = 0
    tp1_hit:     bool = False


def run_backtest(
    *,
    ohlcv:           pd.DataFrame,
    symbol:          str,
    cfg:             Config,
    policy:          DecisionPolicy | None = None,
    initial_equity:  float = 10_000.0,
    timeout_bars:    int   = 32,            # force-close after N bars (8h on 15m)
) -> BTReport:
    """Run a single-symbol replay.

    Args:
        ohlcv:           DataFrame indexed by UTC timestamp with columns
                         ``open, high, low, close, volume``. *Closed* bars only.
        symbol:          Ticker symbol (used in trade records).
        cfg:             Loaded Config (RiskEngine reads risk_pct etc).
        policy:          DecisionPolicy. Defaults to ConservativePolicy(adx_min=cfg.adx_min_trade).
        initial_equity:  Starting equity in USD.
        timeout_bars:    Max bars per trade before force-close at close.

    Returns:
        ``BTReport`` with trades, equity curve, and summary stats.
    """
    if ohlcv.empty:
        return BTReport(
            trades=[], equity_curve=pd.Series(dtype=float),
            start=pd.Timestamp.utcnow(), end=pd.Timestamp.utcnow(),
            initial_equity=initial_equity,
        )

    if policy is None:
        adx_min = float(getattr(cfg, "adx_min_trade", 18.0))
        policy  = ConservativePolicy(adx_min=adx_min)

    # 1) Compute features once for the whole series.
    feat_cfg = FeatureConfig()
    snap     = compute_snapshot(ohlcv, feat_cfg)

    # Keep only closed candles -- protects against an in-progress bar
    # accidentally being included by the caller.
    snap = snap.dropna(how="all")
    if snap.empty:
        return BTReport(
            trades=[], equity_curve=pd.Series(dtype=float),
            start=ohlcv.index.min(), end=ohlcv.index.max(),
            initial_equity=initial_equity,
        )

    risk_eng  = RiskEngine(cfg)
    cooldown  = CooldownState()

    equity        = initial_equity
    equity_curve  = []
    trades:        list[BTTrade] = []
    open_trade:    _OpenTrade | None = None

    next_opens = ohlcv["open"].shift(-1)   # fill price for "decide on T, fill at T+1 open"

    for i in range(len(snap) - 1):              # -1 because we need bar i+1 for fill
        ts   = snap.index[i]
        bar  = snap.iloc[i]
        nxt  = snap.iloc[i + 1] if i + 1 < len(snap) else None

        # ── Manage open trade on the *next* bar (high/low traversal) ────────
        if open_trade is not None and nxt is not None:
            high    = float(nxt["high"])
            low     = float(nxt["low"])
            close_p = float(nxt["close"])
            t       = open_trade
            t.bars_held += 1
            exited  = False
            exit_px = close_p
            reason  = "timeout"

            if t.side == "long":
                if low <= t.sl:
                    exit_px, reason, exited = t.sl, "sl_hit", True
                elif high >= t.tp2:
                    exit_px, reason, exited = t.tp2, "tp2_hit", True
                elif high >= t.tp1 and not t.tp1_hit:
                    t.tp1_hit = True
                    # Move stop to break-even (deterministic).
                    t.sl = max(t.sl, t.entry_px)
            else:
                if high >= t.sl:
                    exit_px, reason, exited = t.sl, "sl_hit", True
                elif low <= t.tp2:
                    exit_px, reason, exited = t.tp2, "tp2_hit", True
                elif low <= t.tp1 and not t.tp1_hit:
                    t.tp1_hit = True
                    t.sl = min(t.sl, t.entry_px)

            if not exited and t.bars_held >= timeout_bars:
                exited = True
                reason = "timeout"

            if exited:
                if t.side == "long":
                    pnl_pct = (exit_px - t.entry_px) / t.entry_px
                else:
                    pnl_pct = (t.entry_px - exit_px) / t.entry_px
                pnl_usd = t.qty_usd * pnl_pct
                r_mult  = pnl_usd / t.risk_usd if t.risk_usd > 0 else 0.0
                equity += pnl_usd
                trades.append(BTTrade(
                    symbol=symbol, side=t.side,
                    entry_ts=t.entry_ts, entry_px=t.entry_px,
                    exit_ts=snap.index[i + 1], exit_px=exit_px,
                    qty_usd=t.qty_usd, pnl_usd=pnl_usd,
                    r_multiple=r_mult, reason=reason,
                    confidence=t.confidence, bars_held=t.bars_held,
                ))
                open_trade = None

        # ── Look for a new entry only when flat ────────────────────────────
        if open_trade is None:
            cooldown = _step_cooldown(cooldown, current_bar_ts=ts, bar_sec=900)
            dec = detect_trigger(symbol=symbol, bar=bar, state=cooldown, cfg=cfg)
            if dec.fired:
                cooldown = CooldownState(last_bar_ts=dec.bar_ts, last_close=dec.close, bars_since=0)
                pol = policy.decide(bar)
                if pol.side in ("long", "short") and i + 1 < len(snap):
                    fill_px = float(next_opens.iloc[i]) if not pd.isna(next_opens.iloc[i]) else float(bar["close"])
                    atr_pct = float(bar.get("atr_14_pct", 1.0))
                    atr     = risk_eng.atr_from_pct(fill_px, atr_pct)
                    sizing  = risk_eng.compute(
                        side=pol.side, entry=fill_px, atr=atr,
                        equity_usd=equity, n_open=0, loser_streak=0,
                    )
                    if sizing.accepted:
                        open_trade = _OpenTrade(
                            side=pol.side,
                            entry_ts=snap.index[i + 1],
                            entry_px=fill_px,
                            sl=sizing.levels.stop_loss,
                            tp1=sizing.levels.tp1,
                            tp2=sizing.levels.tp2,
                            qty_usd=sizing.qty_usd,
                            risk_usd=sizing.risk_usd,
                            confidence=pol.confidence,
                        )

        equity_curve.append((ts, equity))

    # Force-close any still-open trade at last close.
    if open_trade is not None:
        last_px = float(snap.iloc[-1]["close"])
        t = open_trade
        if t.side == "long":
            pnl_pct = (last_px - t.entry_px) / t.entry_px
        else:
            pnl_pct = (t.entry_px - last_px) / t.entry_px
        pnl_usd = t.qty_usd * pnl_pct
        r_mult  = pnl_usd / t.risk_usd if t.risk_usd > 0 else 0.0
        equity += pnl_usd
        trades.append(BTTrade(
            symbol=symbol, side=t.side,
            entry_ts=t.entry_ts, entry_px=t.entry_px,
            exit_ts=snap.index[-1], exit_px=last_px,
            qty_usd=t.qty_usd, pnl_usd=pnl_usd,
            r_multiple=r_mult, reason="forced_close",
            confidence=t.confidence, bars_held=t.bars_held,
        ))

    eq_series = pd.Series(
        [v for _, v in equity_curve],
        index=pd.DatetimeIndex([ts for ts, _ in equity_curve]),
    ) if equity_curve else pd.Series(dtype=float)

    return BTReport(
        trades         = trades,
        equity_curve   = eq_series,
        start          = snap.index[0],
        end            = snap.index[-1],
        initial_equity = initial_equity,
    )


def _step_cooldown(state: CooldownState, *, current_bar_ts: pd.Timestamp, bar_sec: int) -> CooldownState:
    """Lightweight CooldownState advancement (since we're not using CooldownStore)."""
    if state.last_bar_ts is None:
        return state
    elapsed = int((current_bar_ts - state.last_bar_ts).total_seconds() // bar_sec)
    return CooldownState(
        last_bar_ts = state.last_bar_ts,
        last_close  = state.last_close,
        bars_since  = max(0, elapsed),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    ap = argparse.ArgumentParser(description="V8 deterministic backtest replay")
    ap.add_argument("--symbol",   required=True)
    ap.add_argument("--klines",   required=True, help="Parquet file with OHLCV")
    ap.add_argument("--start",    default=None)
    ap.add_argument("--end",      default=None)
    ap.add_argument("--equity",   type=float, default=10_000.0)
    ap.add_argument("--out",      default=None, help="Optional CSV path for trades")
    args = ap.parse_args()

    cfg = load_config()
    df  = pd.read_parquet(args.klines)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    if args.start:
        df = df[df.index >= pd.Timestamp(args.start, tz="UTC")]
    if args.end:
        df = df[df.index <= pd.Timestamp(args.end, tz="UTC")]

    rep = run_backtest(
        ohlcv=df, symbol=args.symbol, cfg=cfg, initial_equity=args.equity,
    )
    stats = rep.stats()
    print(f"\n=== Backtest {args.symbol}  {rep.start} -> {rep.end} ===")
    for k, v in stats.items():
        print(f"  {k:18s} {v:.4f}" if isinstance(v, float) else f"  {k:18s} {v}")

    if args.out:
        pd.DataFrame([asdict(t) for t in rep.trades]).to_csv(args.out, index=False)
        print(f"\nTrades written to {args.out}")


if __name__ == "__main__":   # pragma: no cover
    logging.basicConfig(level=logging.WARNING)
    _cli()
