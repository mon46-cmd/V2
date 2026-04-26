"""Prompt A -- 6h Grok social sentiment scan.

DESIGN:
  We feed Grok the top-30 filtered universe as a structured table with live
  market data (price, 24h change, volume, funding rate, ATR%, active flags).
  Grok picks from THIS list -- it cannot invent symbols outside the universe.

  web_search=True lets Grok search X/Twitter, Reddit, Telegram, news for
  each candidate it is considering.  The table provides technical context;
  the search provides catalysts.  Together they produce a high-quality shortlist.

  Output: SocialScan schema (top 5 picks, direction, conviction, reason).

Cost: ~$0.030/call (Grok 4.20 + search tokens).  Called every 6h only.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ai.client import AIClient
from ai.http import Message
from ai.schemas import (
    SocialScan,
    default_social_scan,
    parse_json_response,
    validate_social_scan,
)
from core.config import Config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a senior crypto derivatives trader and quantitative analyst specialised \
in Bybit USDT perpetual futures on short-to-medium time horizons (6-48h moves).

Your edge is combining live social intelligence with technical market data. \
You will receive a pre-filtered universe of the top tradeable Bybit pairs with \
live market data already attached.  Your job is to identify which 1-5 of these \
pairs have the highest probability of a SIGNIFICANT directional move in the next \
6-48 hours -- and to back every pick with a VERIFIABLE, SPECIFIC catalyst.

=== MANDATORY PROCESS ===
Step 1 -- SCAN SOCIALS: For each plausible candidate in the universe table, \
actively search X/Twitter (last 6h), Reddit (r/CryptoCurrency, r/altcoin, \
r/Bitcoin, r/ethereum, r/defi), Telegram public channels, and crypto news sites \
(Coindesk, Cointelegraph, The Block, Decrypt).
Step 2 -- FILTER NOISE: Discard anonymous shills, coordinated bot activity, \
price predictions without catalysts, meme hype, and copy-paste threads.
Step 3 -- FIND CATALYSTS: Only proceed with picks where you found at least one \
of: protocol upgrade, exchange listing, regulatory news, ETF/institutional flow, \
major on-chain anomaly (whale wallet, bridge flow, smart money), macro trigger \
(FOMC, CPI, geopolitical), or documented technical breakout confirmation.
Step 4 -- APPLY CONTRARIAN FILTER: If a coin trends #1 on CT and mood_score \
is above +0.80, lean short or skip -- extreme greed is a fade signal. If \
sentiment is deeply negative (< -0.70) but the technical data shows accumulation \
(positive funding, stable OI), lean long.
Step 5 -- CROSS-CHECK TECHNICALS: Use the provided table columns. Prefer pairs \
where the social catalyst ALIGNS with the technical structure:
  - Long bias: price above EMA structure, ATR expanding, funding not extreme.
  - Short bias: price failing at resistance, negative CVD divergence implied by \
    high funding rate (> 0.03%) with price stalling.

=== UNIVERSE CLUSTER LABELS ===
Pairs are tagged with a cluster based on live data:
  [MEGA]     : Top-tier liquidity (BTC/ETH/SOL). Core market pairs.
  [TRENDING] : ADX > 25 or golden/death cross active. Momentum confirmed.
  [VOLATILE] : ATR > 3% or 24h range > 30%. High-risk, high-reward.
  [MOONSHOT] : 24h change > +20%. Already moved -- need continuation catalyst.
  [DUMP]     : 24h change < -15%. Potential dead-cat or capitulation.
  [QUIET]    : ATR < 0.5%. Consolidating. Skip unless major catalyst.
  [OPEN]     : We currently hold an open position. MUST be included in picks
               if the thesis is still intact (do NOT drop an [OPEN] pair from
               the watchlist without a clear invalidation reason).

=== STRICT RULES ===
- ONLY pick symbols from the provided universe table. Never suggest symbols outside it.
- Every pick MUST have a specific verifiable catalyst. "People are bullish" is not a catalyst.
- Pick 1-5 symbols. If conviction is absent for any, pick fewer. If the entire market is noise
  (no verifiable catalysts found anywhere), return an empty picks list and explain in model_note.
- Do NOT pick a symbol if its 24h change is already > +15% or < -15% without
  a continuation catalyst -- that ship has sailed.
- [OPEN] positions: include them UNLESS you have a specific invalidation catalyst.
  If you drop an [OPEN] pair, you MUST explain the exact invalidation in key_risk.
- Funding rate > +5bps: extreme long crowding, reduces long conviction (longs paying heavily).
- Funding rate < -2bps: shorts paying longs -- potential short squeeze setup.
- ATR% < 0.3%: pair is consolidating -- skip unless major breakout catalyst confirmed.
- High OI with flat price: potential coil (big move building). Factor into conviction.
- Low OI with big price move: thin positioning, move may not sustain.

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object.  No markdown, no prose outside the JSON. \
The format is specified in the user message.\
"""


# ---------------------------------------------------------------------------
# Universe table builder
# ---------------------------------------------------------------------------

def _cluster_label(r: dict) -> str:
    """Assign a momentum cluster label to a universe row."""
    sym       = str(r.get("symbol", ""))
    chg_pct   = (r.get("price_change_24h_pct") or 0) * 100
    atr_pct   = r.get("atr_pct_15m") or 0
    # Normalise flag names: strip "flag_" prefix and underscores for matching.
    raw_flags = r.get("active_flags") or []
    flags     = [f.replace("flag_", "").replace("_", "") for f in raw_flags]
    has_open  = bool(r.get("has_open_position"))
    is_mega   = sym in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"}

    if has_open:
        return "[OPEN]"
    if is_mega:
        return "[MEGA]"
    if chg_pct >= 20:
        return "[MOONSHOT]"
    if chg_pct <= -15:
        return "[DUMP]"
    if atr_pct > 3.0 or (r.get("high_24h", 0) - r.get("low_24h", 0)) / max(r.get("price", 1), 1e-9) * 100 > 30:
        return "[VOLATILE]"
    if any(f in flags for f in ("adxstrongtrend", "goldencross", "macdcrossup", "deathcross", "macdcrossdn")):
        return "[TRENDING]"
    if atr_pct < 0.5:
        return "[QUIET]"
    return "[WATCH]"


def _build_universe_table(universe_rows: list[dict[str, Any]]) -> str:
    """Format the top-N ticker rows as a compact ASCII table for the prompt.

    Each row: cluster | symbol | price | 24h_chg% | vol_usd_M | funding_bps | oi_M | atr% | flags
    Cluster labels group pairs by momentum regime for fast AI scanning.
    Range column: computed from ticker high_24h/low_24h (true 24h) or OHLCV window.
    """
    lines = [
        "UNIVERSE (top filtered pairs by 24h volume, live Bybit data):",
        "Cols: cluster | symbol | price | 24hChg% | vol24h_M | fundRate_bps | OI_M | atr% | priceRange% | [flags]",
        "",
    ]

    for r in universe_rows:
        sym          = str(r.get("symbol", ""))
        price        = r.get("price", 0) or 0
        chg_pct      = (r.get("price_change_24h_pct") or 0) * 100
        vol_m        = (r.get("turnover_24h") or 0) / 1e6
        funding      = (r.get("funding_rate") or 0) * 10_000    # bps
        oi_usd       = (r.get("open_interest_value") or r.get("open_interest", 0) * price) / 1e6
        high_24h     = r.get("high_24h") or price
        low_24h      = r.get("low_24h")  or price
        atr_pct      = r.get("atr_pct_15m", None)
        active_flags = r.get("active_flags", []) or []
        cluster      = _cluster_label(r)

        price_str = str(int(round(price))) if price > 10 else f"{price:.4f}"
        range_pct = (high_24h - low_24h) / price * 100 if price > 0 else 0.0
        flag_str  = ",".join(f.replace("flag_", "").replace("_", "") for f in active_flags) or "-"
        atr_str   = f"{atr_pct:.2f}%" if atr_pct is not None else "n/a"
        oi_str    = f"{oi_usd:.0f}M" if oi_usd > 0 else "n/a"

        lines.append(
            f"  {cluster:<10} {sym:<14} {price_str:>10} "
            f"| {chg_pct:>+6.2f}% "
            f"| {vol_m:>7.1f}M "
            f"| {funding:>+6.2f}bps "
            f"| {oi_str:>7} "
            f"| {atr_str} "
            f"| rng:{range_pct:.1f}% "
            f"| [{flag_str}]"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(
    *,
    as_of:           str,
    universe_rows:   list[dict[str, Any]],
    btc_24h_pct:     float | None = None,
    eth_24h_pct:     float | None = None,
    fear_greed:      int   | None = None,
    extra_context:   str          = "",
    pinned_symbols:  list[str]    = (),
) -> str:
    """Build the user-facing message for the social scan."""
    lines: list[str] = []

    lines.append(f"Current UTC: {as_of}")
    lines.append("")

    # --- Market macro context ---
    lines.append("MARKET MACRO:")
    if btc_24h_pct is not None:
        lines.append(f"  BTC 24h: {btc_24h_pct:+.2f}%")
    if eth_24h_pct is not None:
        lines.append(f"  ETH 24h: {eth_24h_pct:+.2f}%")
    if fear_greed is not None:
        label = "Extreme Fear" if fear_greed < 25 else \
                "Fear"         if fear_greed < 45 else \
                "Neutral"      if fear_greed < 55 else \
                "Greed"        if fear_greed < 75 else "Extreme Greed"
        lines.append(f"  Fear & Greed Index: {fear_greed}/100 ({label})")
    if extra_context:
        lines.append(f"  {extra_context}")
    lines.append("")

    # --- Open positions note ---
    if pinned_symbols:
        lines.append(
            f"PINNED (open positions -- keep on watchlist unless thesis broken): "
            + ", ".join(pinned_symbols)
        )
        lines.append("")

    # --- Universe table ---
    lines.append(_build_universe_table(universe_rows))
    lines.append("")

    # --- Task ---
    lines.append(
        "TASK: Search socials for the past 6 hours. For each candidate in the "
        "table above, check X/Twitter, Reddit, and Telegram for verifiable "
        "catalysts. Pick 1-5 pairs from the table that have the clearest "
        "directional setup supported by BOTH a social catalyst AND the technical "
        "data above. [OPEN] pairs must remain unless explicitly invalidated."
    )
    lines.append("")

    # --- Schema ---
    lines.append(
        'Return JSON:\n'
        '{\n'
        '  "as_of": "ISO datetime",\n'
        '  "market_mood": "bullish|bearish|mixed|neutral",\n'
        '  "mood_score": -1.0 to +1.0,\n'
        '  "noise_level": "low|medium|high",\n'
        '  "picks": [\n'
        '    {\n'
        '      "symbol": "XXXUSDT",\n'
        '      "direction": "long|short",\n'
        '      "conviction": 0.0-1.0,\n'
        '      "catalyst": "One precise sentence: what specific event/data supports this move?",\n'
        '      "social_source": ["twitter","reddit","telegram","news","onchain"],\n'
        '      "technical_alignment": "1 sentence: does the table data support the direction?",\n'
        '      "key_risk": "1 sentence: what would invalidate this pick?"\n'
        '    }\n'
        '  ],\n'
        '  "model_note": "Any market-wide caveat, dominant narrative, or uncertainty"\n'
        '}'
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main call function
# ---------------------------------------------------------------------------

async def chat_social_scan(
    client: AIClient,
    cfg:    Config,
    *,
    as_of:           str | None            = None,
    universe_rows:   list[dict[str, Any]]  = (),
    btc_24h_pct:     float | None          = None,
    eth_24h_pct:     float | None          = None,
    fear_greed:      int   | None          = None,
    extra_context:   str                   = "",
    pinned_symbols:  list[str]             = (),
) -> SocialScan:
    """Execute the 6h social sentiment scan via Grok 4.20.

    Args:
        client:          Initialised AIClient.
        cfg:             Loaded Config.
        as_of:           ISO-8601 UTC timestamp. Defaults to now.
        universe_rows:   Full ticker rows from UniverseBuilder (top 30).
        btc_24h_pct:     BTC 24h % change for macro context.
        eth_24h_pct:     ETH 24h % change for macro context.
        fear_greed:      Fear & Greed index 0-100 (optional).
        extra_context:   Additional freeform context string.
        pinned_symbols:  Symbols with open positions -- must remain on watchlist.
    """
    as_of = as_of or datetime.now(tz=timezone.utc).isoformat()
    model = cfg.model_social or "x-ai/grok-4.20"

    # Truncate universe to top 30 to keep tokens manageable.
    rows = list(universe_rows)[:30]

    messages = [
        Message("system", _SYSTEM),
        Message("user", _build_user_prompt(
            as_of           = as_of,
            universe_rows   = rows,
            btc_24h_pct     = btc_24h_pct,
            eth_24h_pct     = eth_24h_pct,
            fear_greed      = fear_greed,
            extra_context   = extra_context,
            pinned_symbols  = list(pinned_symbols),
        )),
    ]

    log.info("social_scan start model=%s n_pairs=%d as_of=%s", model, len(rows), as_of)

    try:
        response = await client.chat(
            model       = model,
            messages    = messages,
            call_type   = "social_scan",
            symbol      = "MARKET",
            max_tokens  = 1000,
            web_search  = True,
            max_results = 10,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("social_scan failed: %s", exc)
        return default_social_scan()

    parsed, err = parse_json_response(response)
    if err or parsed is None:
        log.warning("social_scan parse error: %s | raw=%r", err, response[:200])
        return default_social_scan()

    result = validate_social_scan(parsed)
    if isinstance(result, str):
        log.warning("social_scan validation error: %s | parsed=%s", result, parsed)
        return default_social_scan()

    log.info(
        "social_scan ok: mood=%s score=%.2f picks=%d noise=%s",
        result["market_mood"], result["mood_score"],
        len(result["picks"]), result["noise_level"],
    )
    return result
