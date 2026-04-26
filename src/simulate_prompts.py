"""Prompt dry-run simulation.

Fetches real Bybit data, runs the full feature pipeline on live pairs,
then assembles the EXACT prompt text that would be sent to Grok for each
call type (social scan, deep analysis, position review).

Nothing is sent to any AI. Everything is saved to files for review.

Output structure
----------------
data/prompt_review/<run_ts>/
    00_run_summary.txt          -- what was fetched, which pairs, which flags fired
    01_social_scan_prompt.txt   -- Prompt A (system + user) as it would be sent
    02_deep_analysis_<SYM>.txt  -- Prompt B for each triggered symbol
    03_position_review_<SYM>.txt-- Prompt C (simulated TP1 hook)
    symbols/
        <SYM>_snapshot.csv      -- last 20 rows of feature snapshot
        <SYM>_last_bar.txt      -- human-readable feature summary

Usage (inside Docker)
---------------------
    docker run --rm -e PYTHONPATH=/app/src v8-app python src/simulate_prompts.py

Or directly:
    python src/simulate_prompts.py [--symbols BTCUSDT ETHUSDT] [--out-dir /tmp/review]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap path so the script can be run directly.
# ---------------------------------------------------------------------------
_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ai.calls.context_builder import build_deep_context, build_review_context
from ai.calls.social_scan    import _SYSTEM as _SOCIAL_SYSTEM, _build_user_prompt as _social_user, _build_universe_table
from ai.calls.deep_analysis  import _SYSTEM as _DEEP_SYSTEM,   _build_user_prompt as _deep_user
from ai.calls.position_review import _SYSTEM as _REVIEW_SYSTEM, _build_user_prompt as _review_user
from core.config             import load_config
from downloader.http         import HttpClient
from downloader.rest         import RestClient
from features                import FeatureConfig, compute_snapshot, get_last_row, FLAG_COLUMNS
from loops.triggers          import DEFAULT_TRIGGER_FLAGS
from loops.universe          import UniverseBuilder
from portfolio.risk          import RiskEngine

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("simulate_prompts")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TOP_N     = 10       # how many top-volume pairs to scan
BARS_15M          = 200      # candle history per symbol
HTF_INTERVAL      = "60"     # 1h bars for higher-TF bias
HTF_BARS          = 50
REVIEW_SIM_PROFIT = 0.012    # simulated TP1 hit: +1.2% unrealized
REVIEW_SIM_BARS   = 8        # bars held

_SOCIAL_MARKETS   = ["BTCUSDT", "ETHUSDT"]  # always include for market context


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(symbols_override: list[str], out_dir: Path) -> None:
    cfg        = load_config()
    feat_cfg   = FeatureConfig()
    engine     = RiskEngine(cfg)
    run_ts     = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root   = out_dir / run_ts
    sym_dir    = out_root / "symbols"
    sym_dir.mkdir(parents=True, exist_ok=True)

    log.info("Output directory: %s", out_root)

    # HttpClient is an async context manager that manages its own aiohttp session.
    async with HttpClient(cfg) as http:
        rest = RestClient(http, cfg)

        # ----------------------------------------------------------------
        # Step 1: Universe filter -> pick top symbols
        # ----------------------------------------------------------------
        if symbols_override:
            symbols = symbols_override
            log.info("Using override symbols: %s", symbols)
        else:
            builder = UniverseBuilder(cfg, rest, cache_ttl_sec=0)  # no cache in sim
            log.info("Fetching Bybit universe...")
            df_uni  = await builder.get_df(force=True)
            if df_uni.empty:
                log.error("Empty universe -- check connectivity")
                return
            top = df_uni.head(DEFAULT_TOP_N)
            symbols = top["symbol"].tolist()
            log.info("Top %d symbols by turnover: %s", len(symbols), symbols)

        # ----------------------------------------------------------------
        # Step 2: Fetch market context (BTC/ETH) for social scan prompt
        # ----------------------------------------------------------------
        btc_24h_pct = eth_24h_pct = None
        for ref_sym in _SOCIAL_MARKETS:
            try:
                t = await rest.ticker(ref_sym)
                pct = t.get("price_change_24h_pct")
                if ref_sym == "BTCUSDT":
                    btc_24h_pct = float(pct) * 100 if pct else None
                elif ref_sym == "ETHUSDT":
                    eth_24h_pct = float(pct) * 100 if pct else None
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not fetch ticker for %s: %s", ref_sym, exc)

        log.info("Market context: BTC 24h=%.2f%%  ETH 24h=%.2f%%",
                 btc_24h_pct or 0, eth_24h_pct or 0)

        # ----------------------------------------------------------------
        # Step 3: Download OHLCV + compute features for each symbol
        # ----------------------------------------------------------------
        end   = datetime.now(tz=timezone.utc)
        start_15m = pd.Timestamp(end) - pd.Timedelta(minutes=15 * BARS_15M)
        start_htf = pd.Timestamp(end) - pd.Timedelta(hours=HTF_BARS)

        symbol_data: dict[str, dict[str, Any]] = {}

        for sym in symbols:
            log.info("Fetching %s...", sym)
            try:
                ohlcv = await rest.klines(sym, "15", start_15m, end)
                if ohlcv.empty or len(ohlcv) < 50:
                    log.warning("%s: insufficient bars (%d), skipping", sym, len(ohlcv))
                    continue

                # Set DatetimeIndex (required by feature pipeline).
                ohlcv = ohlcv.set_index("timestamp")

                # HTF (1h) for bias detection.
                htf_ohlcv = await rest.klines(sym, HTF_INTERVAL, start_htf, end)
                htf_bias  = _detect_htf_bias(htf_ohlcv) if not htf_ohlcv.empty else "unknown"

                # Derivatives (best-effort).
                try:
                    funding = await rest.funding(sym, start_15m, end)
                except Exception:  # noqa: BLE001
                    funding = None

                # Feature pipeline.
                snapshot = compute_snapshot(ohlcv, feat_cfg, funding=funding)
                last_bar = get_last_row(snapshot)

                # Detect active flags.
                active_flags = _get_active_flags(last_bar)

                symbol_data[sym] = {
                    "ohlcv":        ohlcv,
                    "snapshot":     snapshot,
                    "last_bar":     last_bar,
                    "htf_bias":     htf_bias,
                    "active_flags": active_flags,
                }

                # Save snapshot CSV and human-readable summary.
                snapshot.tail(20).to_csv(sym_dir / f"{sym}_snapshot.csv")
                _write_last_bar(sym_dir / f"{sym}_last_bar.txt", sym, last_bar, active_flags, htf_bias)

                log.info(
                    "  %s: %d bars, %d active flags: %s",
                    sym, len(ohlcv), len(active_flags), active_flags or "none",
                )

            except Exception as exc:  # noqa: BLE001
                log.error("%s failed: %s", sym, exc, exc_info=True)

    if not symbol_data:
        log.error("No symbol data collected -- aborting")
        return

    as_of = datetime.now(tz=timezone.utc).isoformat()

    # ----------------------------------------------------------------
    # Step 4: Build enriched universe rows for social scan
    # Use the ticker DataFrame when available (has OI, 24h high/low from Bybit).
    # Fall back to OHLCV-derived estimates for override-symbol runs.
    # ----------------------------------------------------------------
    # Grab the ticker DF from the universe builder if we used it.
    ticker_map: dict[str, dict] = {}
    try:
        uni_df = await universe.get_df()
        if not uni_df.empty and "symbol" in uni_df.columns:
            for _, row in uni_df.iterrows():
                ticker_map[str(row["symbol"])] = row.to_dict()
    except Exception:  # noqa: BLE001
        pass

    universe_rows: list[dict[str, Any]] = []
    # Simulated open positions to pin (real system reads from PositionStore).
    pinned_symbols = ["ORCAUSDT", "RAVEUSDT"]   # example – replace with real store

    for sym in symbols:
        d = symbol_data.get(sym)
        if d is None:
            continue
        last_bar     = d["last_bar"]
        ohlcv        = d["ohlcv"]
        active_flags = d["active_flags"]
        tk           = ticker_map.get(sym, {})

        close  = float(ohlcv["close"].iloc[-1])
        high24 = float(tk.get("high_24h") or ohlcv["high"].max())
        low24  = float(tk.get("low_24h")  or ohlcv["low"].min())
        vol24  = float(tk.get("turnover_24h") or
                       (ohlcv["turnover"].sum() if "turnover" in ohlcv.columns
                        else ohlcv["volume"].sum() * close))
        oi_val = float(tk.get("open_interest_value") or 0)
        fr     = float(tk.get("funding_rate") or last_bar.get("funding_rate") or 0)
        chg24  = float(tk.get("price_change_24h_pct") or
                       ((close / float(ohlcv["open"].iloc[0]) - 1) if not ohlcv.empty else 0))
        atr_pct = float(last_bar.get("atr_14_pct") or 0)

        universe_rows.append({
            "symbol":                sym,
            "price":                 close,
            "price_change_24h_pct":  chg24,
            "turnover_24h":          vol24,
            "funding_rate":          fr,
            "high_24h":              high24,
            "low_24h":               low24,
            "open_interest_value":   oi_val,
            "atr_pct_15m":           atr_pct,
            "active_flags":          active_flags,
            "has_open_position":     sym in pinned_symbols,
        })

    # ----------------------------------------------------------------
    # Step 5: Prompt A -- social scan
    # ----------------------------------------------------------------
    log.info("Building Prompt A (social scan)...")
    social_user_msg = _social_user(
        as_of            = as_of,
        universe_rows    = universe_rows,
        btc_24h_pct      = btc_24h_pct,
        eth_24h_pct      = eth_24h_pct,
        pinned_symbols   = pinned_symbols,
    )
    _write_prompt(
        out_root / "01_social_scan_prompt.txt",
        call_type    = "Prompt A -- Social Scan (Grok 4.20 + web_search=True)",
        model        = cfg.model_social,
        system       = _SOCIAL_SYSTEM,
        user         = social_user_msg,
        note         = (
            "This prompt is sent every 6h. web_search=True means Grok will\n"
            "search X/Twitter, Reddit, and news before responding.\n"
            f"Estimated tokens: ~700 in / ~400 out\n"
            f"Estimated cost:   ~$0.022/call"
        ),
    )

    # ----------------------------------------------------------------
    # Step 6: Prompt B -- deep analysis for each triggered symbol
    # ----------------------------------------------------------------
    triggered: list[str] = []
    all_syms_sorted = sorted(symbol_data.keys(), key=lambda s: len(symbol_data[s]["active_flags"]), reverse=True)

    for sym in all_syms_sorted:
        d            = symbol_data[sym]
        last_bar     = d["last_bar"]
        ohlcv        = d["ohlcv"]
        active_flags = d["active_flags"]
        htf_bias     = d["htf_bias"]

        # Get close directly from OHLCV (feature snapshot doesn't carry it).
        try:
            close_price = float(ohlcv["close"].iloc[-1])
        except (KeyError, IndexError, ValueError):
            close_price = None

        # Use the first active flag if any, otherwise force one for simulation.
        fired_flag   = active_flags[0] if active_flags else "flag_volume_climax (simulated)"

        ctx = build_deep_context(
            row                  = last_bar.to_dict(),
            symbol               = sym,
            as_of                = as_of,
            close_price          = close_price,
            social_reason        = f"Simulated social pick: {sym} showing {fired_flag}",
            watchlist_direction  = _guess_direction(last_bar),
            htf_bias             = htf_bias,
            ohlcv_df             = ohlcv.reset_index(),
            lookback_rows        = _lookback_rows(d["snapshot"], n=10),
            fired_flag           = fired_flag,
        )

        deep_user_msg = _deep_user(symbol=sym, context=ctx)
        _write_prompt(
            out_root / f"02_deep_analysis_{sym}.txt",
            call_type    = f"Prompt B -- Deep Analysis (Grok 4.1 Fast) -- {sym}",
            model        = cfg.model_deep,
            system       = _DEEP_SYSTEM,
            user         = deep_user_msg,
            note         = (
                f"Triggered by: {fired_flag}\n"
                f"Active flags: {active_flags or 'none (simulated)'}\n"
                f"HTF bias:     {htf_bias}\n"
                f"Estimated tokens: ~650 in / ~200 out\n"
                f"Estimated cost:   ~$0.00023/call"
            ),
        )
        triggered.append(sym)
        log.info("  Prompt B written for %s (flag=%s)", sym, fired_flag)

        # Only write Prompt B for top 5 to keep output manageable.
        if len(triggered) >= 5:
            break

    # ----------------------------------------------------------------
    # Step 6: Prompt C -- simulated position review (TP1 hit scenario)
    # ----------------------------------------------------------------
    log.info("Building Prompt C (position review simulations)...")
    for sym in triggered[:3]:   # simulate review for top 3
        d        = symbol_data[sym]
        last_bar = d["last_bar"]
        ohlcv    = d["ohlcv"]

        # close is in the OHLCV frame (last_bar is the feature snapshot row).
        try:
            close = float(ohlcv["close"].iloc[-1])
        except (KeyError, IndexError, TypeError, ValueError):
            close = 0.0

        if close <= 0:
            log.warning("  Prompt C skipped for %s -- could not determine close price", sym)
            continue

        atr_pct = float(last_bar.get("atr_14_pct", 1.2) or 1.2)
        atr     = engine.atr_from_pct(close, atr_pct)
        side    = _guess_direction(last_bar)
        levels  = engine.compute_levels(side=side, entry=close * 0.998, atr=atr)
        mark    = levels.entry * (1 + REVIEW_SIM_PROFIT) if side == "long" else levels.entry * (1 - REVIEW_SIM_PROFIT)

        # Pass raw OHLCV as ohlcv_df for bar display (has open/high/low/close/volume).
        # Attach RSI from the snapshot so rsi_trend can be computed.
        row_dict   = last_bar.to_dict()
        snap_reset = d["snapshot"].reset_index()
        ohlcv_for_review = ohlcv.reset_index().copy()
        if "rsi_14" in snap_reset.columns:
            # Align by index (both are date-ordered; snapshot may have more rows).
            rsi_series = snap_reset["rsi_14"].values[-len(ohlcv_for_review):]
            if len(rsi_series) == len(ohlcv_for_review):
                ohlcv_for_review["rsi_14"] = rsi_series
            else:
                # Pad with NaN if lengths differ.
                ohlcv_for_review["rsi_14"] = float("nan")

        ctx = build_review_context(
            symbol        = sym,
            side          = side,
            entry         = levels.entry,
            stop_loss     = levels.stop_loss,
            tp1           = levels.tp1,
            tp2           = levels.tp2,
            mark          = mark,
            remaining_pct = 50.0,           # TP1 already hit, 50% remains
            bars_held     = REVIEW_SIM_BARS,
            loser_streak  = 0,
            hook_reason   = "tp1_hit",
            row           = row_dict,
            ohlcv_df      = ohlcv_for_review,   # raw OHLCV with RSI attached
        )

        review_user_msg = _review_user(symbol=sym, context=ctx)
        _write_prompt(
            out_root / f"03_position_review_{sym}.txt",
            call_type    = f"Prompt C -- Position Review (Grok 4.1 Fast) -- {sym}",
            model        = cfg.model_review,
            system       = _REVIEW_SYSTEM,
            user         = review_user_msg,
            note         = (
                f"Hook reason:   tp1_hit (simulated at +1.2% from entry)\n"
                f"Position side: {side}\n"
                f"Entry: {levels.entry:.4f}  SL: {levels.stop_loss:.4f}  "
                f"TP1: {levels.tp1:.4f}  TP2: {levels.tp2:.4f}\n"
                f"Mark:  {mark:.4f}  Remaining: 50%  Bars held: {REVIEW_SIM_BARS}\n"
                f"Estimated tokens: ~550 in / ~150 out\n"
                f"Estimated cost:   ~$0.00018/call"
            ),
        )
        log.info("  Prompt C written for %s (side=%s close=%.4f)", sym, side, close)

    # ----------------------------------------------------------------
    # Step 7: Run summary
    # ----------------------------------------------------------------
    _write_summary(
        out_root / "00_run_summary.txt",
        run_ts      = run_ts,
        as_of       = as_of,
        symbols     = list(symbol_data.keys()),
        triggered   = triggered,
        btc_24h_pct = btc_24h_pct,
        eth_24h_pct = eth_24h_pct,
        symbol_data = symbol_data,
        cfg_model_social = cfg.model_social,
        cfg_model_deep   = cfg.model_deep,
        cfg_model_review = cfg.model_review,
    )

    log.info("=" * 60)
    log.info("Simulation complete. Files saved to:")
    log.info("  %s", out_root)
    log.info("=" * 60)
    for f in sorted(out_root.rglob("*")):
        if f.is_file():
            log.info("  %s (%d bytes)", f.relative_to(out_root), f.stat().st_size)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_prompt(path: Path, *, call_type: str, model: str, system: str, user: str, note: str) -> None:
    sep = "=" * 70
    content = "\n".join([
        sep,
        f"  {call_type}",
        sep,
        f"Model:  {model}",
        "",
        note,
        "",
        sep,
        "SYSTEM PROMPT",
        sep,
        system.strip(),
        "",
        sep,
        "USER PROMPT",
        sep,
        user.strip(),
        "",
        sep,
        f"  Token estimate: ~{(len(system)+len(user))//4} prompt tokens",
        sep,
    ])
    path.write_text(content, encoding="utf-8")


def _write_last_bar(path: Path, sym: str, row: pd.Series, flags: list[str], htf_bias: str) -> None:
    lines = [f"Last bar summary: {sym}  HTF bias: {htf_bias}", ""]
    for key in sorted(row.index):
        v = row[key]
        if pd.isna(v):
            continue
        if key.startswith("flag_") and v == 0:
            continue  # skip silent flags
        lines.append(f"  {key:<35} {v}")
    if flags:
        lines.append("")
        lines.append(f"Active flags: {flags}")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_summary(
    path: Path,
    *,
    run_ts:       str,
    as_of:        str,
    symbols:      list[str],
    triggered:    list[str],
    btc_24h_pct:  float | None,
    eth_24h_pct:  float | None,
    symbol_data:  dict,
    cfg_model_social: str,
    cfg_model_deep:   str,
    cfg_model_review: str,
) -> None:
    lines = [
        "=" * 70,
        "V8 PROMPT DRY-RUN SIMULATION",
        "=" * 70,
        f"Run timestamp: {run_ts}",
        f"As of (UTC):   {as_of}",
        "",
        "MARKET CONTEXT",
        f"  BTC 24h: {btc_24h_pct:+.2f}%".replace("None", "n/a") if btc_24h_pct else "  BTC 24h: n/a",
        f"  ETH 24h: {eth_24h_pct:+.2f}%".replace("None", "n/a") if eth_24h_pct else "  ETH 24h: n/a",
        "",
        "MODELS",
        f"  Prompt A (social scan):  {cfg_model_social}",
        f"  Prompt B (deep analysis):{cfg_model_deep}",
        f"  Prompt C (review):       {cfg_model_review}",
        "",
        f"UNIVERSE SCANNED ({len(symbols)} symbols)",
        *[f"  {s}" for s in symbols],
        "",
        f"DEEP ANALYSIS PROMPTS GENERATED ({len(triggered)})",
        *[f"  {s} -> 02_deep_analysis_{s}.txt" for s in triggered],
        "",
        "PER-SYMBOL FLAG STATUS",
    ]
    for sym, d in symbol_data.items():
        flags = d["active_flags"]
        bias  = d["htf_bias"]
        lines.append(f"  {sym:<15} HTF={bias:<12} flags={flags or ['none']}")

    lines += [
        "",
        "FILES",
        "  01_social_scan_prompt.txt      -- Prompt A",
        *[f"  02_deep_analysis_{s}.txt      -- Prompt B" for s in triggered],
        *[f"  03_position_review_{s}.txt    -- Prompt C (sim)" for s in triggered[:3]],
        *[f"  symbols/{s}_snapshot.csv" for s in symbols],
        *[f"  symbols/{s}_last_bar.txt" for s in symbols],
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def _get_active_flags(row: pd.Series) -> list[str]:
    """Return list of flag columns that are >= 1.0 in the last bar."""
    flags = []
    for col in FLAG_COLUMNS:
        if col in row.index:
            try:
                if float(row[col]) >= 1.0:
                    flags.append(col)
            except (TypeError, ValueError):
                pass
    return flags


def _detect_htf_bias(htf_ohlcv: pd.DataFrame) -> str:
    """Simple 1h EMA-21 vs EMA-50 bias check."""
    try:
        c   = htf_ohlcv["close"].astype(float)
        e21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
        e50 = c.ewm(span=50, adjust=False).mean().iloc[-1]
        last = c.iloc[-1]
        if last > e21 > e50:
            return "bullish"
        if last < e21 < e50:
            return "bearish"
        return "mixed"
    except Exception:  # noqa: BLE001
        return "unknown"


def _guess_direction(row: pd.Series) -> str:
    """Guess likely trade direction from supertrend + EMA alignment."""
    try:
        st = float(row.get("supertrend_dir", 0) or 0)
        if st > 0:
            return "long"
        if st < 0:
            return "short"
    except (TypeError, ValueError):
        pass
    return "long"


def _lookback_rows(snapshot: pd.DataFrame, n: int = 10) -> list[dict]:
    """Return last N rows of snapshot as list of dicts."""
    tail = snapshot.iloc[-n:] if len(snapshot) >= n else snapshot
    return [row.to_dict() for _, row in tail.iterrows()]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V8 prompt dry-run simulation")
    p.add_argument(
        "--symbols", nargs="+", metavar="SYM", default=[],
        help="Override symbol list (e.g. BTCUSDT ETHUSDT SOLUSDT)",
    )
    p.add_argument(
        "--out-dir", metavar="DIR", default="",
        help="Output directory (default: data/prompt_review)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args    = _parse_args()
    cfg     = load_config()
    out_dir = Path(args.out_dir) if args.out_dir else (cfg.data_root / "prompt_review")
    asyncio.run(main(symbols_override=args.symbols, out_dir=out_dir))
