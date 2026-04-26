"""V8 Dashboard — REST API routes.

All endpoints are prefixed with ``/api`` by the main app.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query, Request

log = logging.getLogger(__name__)
router = APIRouter()

# ── Helpers ─────────────────────────────────────────────────────────────────

def _cfg(request: Request):
    return request.app.state.cfg


def _safe_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        f = float(v)
        return None if f != f else f   # NaN → None for JSON
    except (TypeError, ValueError):
        return None


# ── Universe / Tickers ─────────────────────────────────────────────────────

@router.get("/universe")
async def universe(request: Request):
    """Top 30 tradeable pairs with live ticker snapshot."""
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from loops.universe import UniverseBuilder

    cfg = _cfg(request)
    async with HttpClient(cfg) as http:
        rest    = RestClient(http, cfg)
        builder = UniverseBuilder(cfg, rest)
        # Force re-fetch if df cache is empty (e.g. fresh container with stale disk cache)
        df      = await builder.get_df()
        if df.empty:
            df = await builder.get_df(force=True)

    # Build response rows
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "symbol":        str(r.get("symbol", "")),
            "price":         _safe_float(r.get("price")),
            "change_24h":    _safe_float(r.get("price_change_24h_pct")),
            "volume_24h":    _safe_float(r.get("volume_24h")),
            "turnover_24h":  _safe_float(r.get("turnover_24h")),
            "open_interest": _safe_float(r.get("open_interest")),
            "funding_rate":  _safe_float(r.get("funding_rate")),
            "high_24h":      _safe_float(r.get("high_24h")),
            "low_24h":       _safe_float(r.get("low_24h")),
        })

    return {"count": len(rows), "symbols": rows}


# ── Flags ───────────────────────────────────────────────────────────────────

_FLAG_LABELS = {
    "flag_volume_climax":    {"label": "Volume Climax",           "color": "orange"},
    "flag_sweep_up":         {"label": "Sweep Up (Bull Trap)",   "color": "red"},
    "flag_sweep_dn":         {"label": "Sweep Down (Bear Trap)", "color": "green"},
    "flag_macd_cross_up":    {"label": "MACD Cross Up",          "color": "green"},
    "flag_macd_cross_dn":    {"label": "MACD Cross Down",        "color": "red"},
    "flag_adx_strong_trend": {"label": "ADX Strong Trend",       "color": "gold"},
    "flag_golden_cross":     {"label": "Golden Cross",           "color": "gold"},
    "flag_death_cross":      {"label": "Death Cross",            "color": "red"},
    "flag_rsi_overbought":   {"label": "RSI Overbought",         "color": "red"},
    "flag_rsi_oversold":     {"label": "RSI Oversold",           "color": "green"},
    "flag_regime_flip":      {"label": "Regime Flip",            "color": "purple"},
    "flag_squeeze_release":  {"label": "Squeeze Release",        "color": "blue"},
}


@router.get("/flags/{symbol}")
async def flags(request: Request, symbol: str):
    """Compute live flag status for a symbol using last 320 bars of 15m data."""
    import numpy as np
    import pandas as pd
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from features.pipeline import compute_snapshot, FeatureConfig

    cfg  = _cfg(request)
    feat = FeatureConfig()
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(minutes=15 * 340)

    async with HttpClient(cfg) as http:
        rest  = RestClient(http, cfg)
        ohlcv = await rest.klines(symbol, "15", start, end)

    if ohlcv.empty or len(ohlcv) < 60:
        return {"symbol": symbol, "flags": [], "indicators": {}, "error": "Insufficient data"}

    ohlcv_idx = ohlcv.set_index("timestamp")
    try:
        snap = compute_snapshot(ohlcv_idx, feat)
    except Exception as exc:
        return {"symbol": symbol, "flags": [], "indicators": {}, "error": str(exc)}

    last = snap.iloc[-1]

    # Collect flag states
    flags_out = []
    for col, meta in _FLAG_LABELS.items():
        if col in snap.columns:
            val = last.get(col, 0)
            active = bool(pd.notna(val) and float(val) >= 1.0)
            # Count recent fires (last 20 bars)
            recent = snap[col].tail(20)
            fire_rate = round(float(recent.sum() / len(recent) * 100), 1)
            flags_out.append({
                "key":       col,
                "label":     meta["label"],
                "color":     meta["color"],
                "active":    active,
                "fire_rate": fire_rate,
            })

    # Key indicators
    indicators = {}
    for col in ["rsi_14", "adx_14", "macd_hist", "supertrend_dir",
                 "bb_width_20", "ema_21", "ema_50", "atr_14"]:
        if col in snap.columns:
            v = last.get(col)
            if pd.notna(v):
                indicators[col] = round(float(v), 6)

    ts = snap.index[-1].isoformat() if not snap.empty else None
    return {
        "symbol":       symbol,
        "flags":        flags_out,
        "indicators":   indicators,
        "active_count": sum(1 for f in flags_out if f["active"]),
        "timestamp":    ts,
    }


# ── Klines ─────────────────────────────────────────────────────────────────

_TF_MAP = {
    "1m": "1", "5m": "5", "15m": "15", "1h": "60",
    "4h": "240", "1d": "D", "1w": "W",
}

@router.get("/klines/{symbol}")
async def klines(
    request: Request,
    symbol: str,
    tf: str = Query("15m", description="Timeframe: 1m, 5m, 15m, 1h, 4h, 1d"),
    bars: int = Query(200, ge=10, le=1000, description="Number of bars"),
):
    """OHLCV candle data for charting."""
    from downloader.http import HttpClient
    from downloader.rest import RestClient

    cfg = _cfg(request)
    interval = _TF_MAP.get(tf, "15")

    # Compute time window
    tf_minutes = {"1": 1, "5": 5, "15": 15, "60": 60, "240": 240, "D": 1440, "W": 10080}
    mins = tf_minutes.get(interval, 15)
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(minutes=mins * (bars + 10))

    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)
        df   = await rest.klines(symbol, interval, start, end)

    if df.empty:
        return {"symbol": symbol, "tf": tf, "candles": []}

    candles = []
    for _, r in df.iterrows():
        candles.append({
            "time": int(r["timestamp"].timestamp()),
            "open": round(float(r["open"]), 8),
            "high": round(float(r["high"]), 8),
            "low":  round(float(r["low"]), 8),
            "close": round(float(r["close"]), 8),
            "volume": round(float(r.get("volume", 0)), 4),
        })

    return {"symbol": symbol, "tf": tf, "candles": candles}


# ── AI Calls Audit ─────────────────────────────────────────────────────────

def _read_jsonl_files(log_dir: Path, max_files: int = 7) -> list[dict]:
    """Read recent JSONL audit files and return entries as a list."""
    if not log_dir.exists():
        return []

    files = sorted(log_dir.glob("*.jsonl"), reverse=True)[:max_files]
    entries = []
    for f in files:
        try:
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    entries.append(json.loads(line))
        except Exception as exc:
            log.warning("Failed to parse %s: %s", f.name, exc)

    return sorted(entries, key=lambda e: e.get("ts_utc", ""), reverse=True)


@router.get("/ai/calls")
async def ai_calls(
    request: Request,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    call_type: str | None = Query(None, description="Filter by call_type"),
):
    """Paginated AI call audit log."""
    cfg = _cfg(request)
    log_dir = cfg.log_root / "ai_calls"
    entries = _read_jsonl_files(log_dir)

    if call_type:
        entries = [e for e in entries if e.get("call_type") == call_type]

    total = len(entries)
    start = (page - 1) * per_page
    page_entries = entries[start:start + per_page]

    # Strip full message content for the list view (keep preview only)
    summaries = []
    for e in page_entries:
        msgs = e.get("messages", [])
        resp = e.get("response") or {}
        summaries.append({
            "call_id":    e.get("call_id"),
            "ts_utc":     e.get("ts_utc"),
            "call_type":  e.get("call_type"),
            "model":      e.get("model"),
            "symbol":     e.get("symbol"),
            "dry_run":    e.get("dry_run"),
            "cost_usd":   e.get("cost_usd"),
            "error":      e.get("error"),
            "latency_ms": resp.get("latency_ms"),
            "tokens_in":  (resp.get("usage") or {}).get("prompt_tokens"),
            "tokens_out": (resp.get("usage") or {}).get("completion_tokens"),
            "n_messages": len(msgs),
            "response_preview": (resp.get("content") or "")[:300],
        })

    return {
        "total": total,
        "page": page,
        "per_page": per_page,
        "calls": summaries,
    }


@router.get("/ai/calls/{call_id}")
async def ai_call_detail(request: Request, call_id: str):
    """Full detail for a single AI call, including complete prompts and response."""
    cfg = _cfg(request)
    log_dir = cfg.log_root / "ai_calls"
    entries = _read_jsonl_files(log_dir)

    for e in entries:
        if e.get("call_id") == call_id:
            return e

    return {"error": "Call not found", "call_id": call_id}


# ── Watchlist / AI Picks ───────────────────────────────────────────────────

@router.get("/watchlist")
async def watchlist(request: Request):
    """Current AI-selected watchlist (top 5 from latest social scan)."""
    cfg = _cfg(request)
    log_dir = cfg.log_root / "ai_calls"
    entries = _read_jsonl_files(log_dir, max_files=3)

    # Find the most recent social_scan call
    for e in entries:
        if e.get("call_type") in ("social_scan", "social"):
            resp = e.get("response") or {}
            content = resp.get("content", "")
            try:
                parsed = json.loads(content)
                picks = parsed.get("picks", [])
                return {
                    "ts_utc": e.get("ts_utc"),
                    "model": e.get("model"),
                    "picks": picks,
                    "call_id": e.get("call_id"),
                }
            except (json.JSONDecodeError, TypeError):
                pass

    return {"picks": [], "ts_utc": None, "note": "No social scan found in recent logs"}


# ── System Status ──────────────────────────────────────────────────────────

@router.get("/status")
async def status(request: Request):
    """System health and configuration summary."""
    cfg = _cfg(request)
    log_dir = cfg.log_root / "ai_calls"
    entries = _read_jsonl_files(log_dir, max_files=1)

    today_cost = sum(
        e.get("cost_usd", 0)
        for e in entries
        if e.get("ts_utc", "").startswith(datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"))
    )

    return {
        "status": "running",
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "config": {
            "model_social":   cfg.model_social,
            "model_deep":     cfg.model_deep,
            "model_review":   cfg.model_review,
            "universe_size":  cfg.universe_size,
            "ai_dry_run":     cfg.ai_dry_run,
            "ai_kill_switch": cfg.ai_kill_switch,
            "daily_budget":   cfg.daily_budget_usd,
            "today_spent":    round(today_cost, 6),
        },
        "risk": {
            "equity_usd":         cfg.paper_equity_usd,
            "risk_per_trade_pct": cfg.risk_pct_per_trade * 100,
            "max_positions":      cfg.max_open_positions,
            "atr_mult_sl":        cfg.atr_mult_sl,
        },
        "ai_calls_today": len([
            e for e in entries
            if e.get("ts_utc", "").startswith(datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"))
        ]),
    }


# ── Color helpers ──────────────────────────────────────────────────────────

_HEX = {
    "green":  "#1db88c", "red":    "#e85555", "gold":   "#f0c040",
    "orange": "#f07840", "blue":   "#3f9fe8", "purple": "#a070f0",
}

def _hex(name: str) -> str:
    return _HEX.get(name, "#6b84a0")


# ── Chart Markers ──────────────────────────────────────────────────────────

_MARKER_FLAGS: dict[str, tuple[str, str, str]] = {
    "flag_volume_climax":    ("Vol Climax",    "orange", "arrowUp"),
    "flag_sweep_up":         ("Sweep ▲",       "red",    "arrowDown"),
    "flag_sweep_dn":         ("Sweep ▼",       "green",  "arrowUp"),
    "flag_macd_cross_up":    ("MACD ↑",        "green",  "arrowUp"),
    "flag_macd_cross_dn":    ("MACD ↓",        "red",    "arrowDown"),
    "flag_adx_strong_trend": ("ADX Strong",    "gold",   "circle"),
    "flag_golden_cross":     ("Golden X",      "gold",   "arrowUp"),
    "flag_death_cross":      ("Death X",       "red",    "arrowDown"),
    "flag_rsi_overbought":   ("RSI OB",        "red",    "arrowDown"),
    "flag_rsi_oversold":     ("RSI OS",        "green",  "arrowUp"),
    "flag_regime_flip":      ("Regime Flip",   "purple", "circle"),
    "flag_squeeze_release":  ("Squeeze",       "blue",   "circle"),
}


@router.get("/markers/{symbol}")
async def chart_markers(
    request: Request,
    symbol: str,
    tf: str = Query("15m"),
):
    """Return chart markers: flag events + AI call events for a symbol."""
    import pandas as pd
    from downloader.http import HttpClient
    from downloader.rest import RestClient
    from features.pipeline import compute_snapshot, FeatureConfig

    cfg      = _cfg(request)
    interval = _TF_MAP.get(tf, "15")
    tf_min   = {"1":1,"5":5,"15":15,"60":60,"240":240,"D":1440,"W":10080}.get(interval, 15)
    end      = datetime.now(tz=timezone.utc)
    start    = end - timedelta(minutes=tf_min * 340)

    async with HttpClient(cfg) as http:
        rest  = RestClient(http, cfg)
        ohlcv = await rest.klines(symbol, interval, start, end)

    if ohlcv.empty or len(ohlcv) < 50:
        return {"symbol": symbol, "tf": tf, "markers": []}

    ohlcv_idx = ohlcv.set_index("timestamp")
    feat = FeatureConfig()
    try:
        snap = compute_snapshot(ohlcv_idx, feat)
    except Exception as exc:
        return {"symbol": symbol, "tf": tf, "markers": [], "error": str(exc)}

    markers: list[dict] = []

    # ── Flag markers ──────────────────────────────────────────────────────
    for col, (label, color, shape) in _MARKER_FLAGS.items():
        if col not in snap.columns:
            continue
        pos = "belowBar" if shape == "arrowUp" else "aboveBar"
        for ts, val in snap[col].items():
            if pd.notna(val) and float(val) >= 1.0:
                markers.append({
                    "time":     int(ts.timestamp()),
                    "position": pos,
                    "color":    _hex(color),
                    "shape":    shape,
                    "text":     label,
                    "size":     1,
                    "type":     "flag",
                    "key":      col,
                })

    # ── AI call markers ───────────────────────────────────────────────────
    log_dir = cfg.log_root / "ai_calls"
    for e in _read_jsonl_files(log_dir):
        if e.get("symbol") != symbol:
            continue
        ts_str = e.get("ts_utc", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            resp    = e.get("response") or {}
            content = resp.get("content", "")
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {}
            action  = parsed.get("action", e.get("call_type", "?"))
            ai_col  = "#a070f0"
            pos     = "belowBar"
            shape   = "circle"
            if action in ("LONG",):
                ai_col = "#1db88c"; shape = "arrowUp"; pos = "belowBar"
            elif action in ("SHORT", "CLOSE"):
                ai_col = "#e85555"; shape = "arrowDown"; pos = "aboveBar"
            markers.append({
                "time":     int(ts.timestamp()),
                "position": pos,
                "color":    ai_col,
                "shape":    shape,
                "text":     f"AI:{action}",
                "size":     2,
                "type":     "ai_call",
                "call_id":  e.get("call_id"),
                "call_type": e.get("call_type"),
            })
        except Exception:
            pass

    markers.sort(key=lambda m: m["time"])
    return {"symbol": symbol, "tf": tf, "markers": markers,
            "flag_count": sum(1 for m in markers if m["type"] == "flag"),
            "ai_count":   sum(1 for m in markers if m["type"] == "ai_call")}


# ── Application Logs ───────────────────────────────────────────────────────

@router.get("/logs")
async def app_logs(
    request: Request,
    n: int = Query(300, ge=1, le=2000),
    level: str | None = Query(None),
):
    """Return recent captured log lines from the ring buffer."""
    from web_api.app import LOG_BUFFER
    lines = list(LOG_BUFFER)
    if level:
        lines = [l for l in lines if l.get("level") == level.upper()]
    return {
        "lines": lines[-n:],
        "total": len(LOG_BUFFER),
    }


# ── Portfolio ──────────────────────────────────────────────────────────────

@router.get("/portfolio")
async def portfolio(request: Request):
    """Portfolio summary derived from AI call history and config defaults."""
    cfg     = _cfg(request)
    log_dir = cfg.log_root / "ai_calls"
    entries = _read_jsonl_files(log_dir, max_files=30)

    equity0 = cfg.paper_equity_usd
    equity  = equity0
    total_cost  = sum(e.get("cost_usd", 0) for e in entries)
    calls_total = len(entries)

    # Build per-symbol call counts
    per_sym: dict[str, dict] = {}
    for e in entries:
        sym = e.get("symbol")
        if not sym:
            continue
        if sym not in per_sym:
            per_sym[sym] = {"symbol": sym, "analyses": 0, "reviews": 0, "last_action": None, "last_ts": None}
        ct = e.get("call_type", "")
        if "analysis" in ct:
            per_sym[sym]["analyses"] += 1
            resp    = e.get("response") or {}
            content = resp.get("content", "")
            try:
                parsed = json.loads(content)
                per_sym[sym]["last_action"] = parsed.get("action")
                per_sym[sym]["last_ts"]     = e.get("ts_utc")
            except Exception:
                pass
        elif "review" in ct:
            per_sym[sym]["reviews"] += 1

    # Simple equity curve from total cost deduction
    equity_curve = [{"time": "start", "equity": equity0}]
    for e in sorted(entries, key=lambda x: x.get("ts_utc", "")):
        cost = e.get("cost_usd", 0)
        equity -= cost
        ts = e.get("ts_utc", "")
        if ts:
            equity_curve.append({"time": ts[:16], "equity": round(equity, 4)})

    return {
        "equity_start": equity0,
        "equity_current": round(equity, 4),
        "total_ai_cost": round(total_cost, 6),
        "calls_total": calls_total,
        "ai_picks": list(request.app.state.cfg.trigger_flags)
                    if hasattr(request.app.state, "cfg") else [],
        "watched_symbols": list(per_sym.values()),
        "equity_curve": equity_curve[-100:],  # last 100 points
    }
