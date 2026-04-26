"""Prompt B -- per-trigger deep analysis.

Called when a Tier-A flag fires on a watchlist symbol.
Grok 4.1 Fast. No web search -- all context is injected.

DESIGN CONTRACT:
  - AI gives: action (long/short/skip), confidence, time_horizon, rationale, invalidation.
  - AI does NOT give: entry price, stop-loss, or take-profit levels.
  - All price levels are computed mechanically by RiskEngine (ATR-based).
  - AI output below 0.60 confidence is automatically demoted to "skip".

The system prompt embeds a full professional trade analysis framework so Grok
reasons like a quantitative trader, not a chatbot.

Cost: ~$0.00025/call (Grok 4.1 Fast, no search). Can be called frequently.
"""
from __future__ import annotations

import logging

from ai.client import AIClient
from ai.http import Message
from ai.schemas import (
    DeepAnalysis,
    default_deep_analysis,
    parse_json_response,
    validate_deep_analysis,
)
from core.config import Config

log = logging.getLogger(__name__)

# Minimum AI confidence before we act. Below this, force skip.
MIN_CONFIDENCE = 0.60

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a quantitative crypto derivatives trader specialised in Bybit USDT \
perpetual futures on the 15-minute timeframe. You analyse structured market \
data snapshots and decide whether to open a position, in which direction, \
and with how much conviction.

You DO NOT suggest entry prices, stop-losses, or take-profits. Those are \
computed mechanically by the risk engine after your decision. Your job is \
purely directional intelligence: is there an edge here, and if so, which way?

=== YOUR ANALYSIS FRAMEWORK ===

Step 1 -- REGIME CHECK (non-negotiable gates, check EACH independently):
  - ADX < {adx_min:.0f}: No trend. Output: skip regardless of other signals.
  - efficiency_ratio < {eff_min:.2f} AND vol_regime_score < 0.20 simultaneously:
    Choppy low-vol regime. Skip unless a Tier-A flag is active (sweep/climax).
    Note: if ONLY efficiency_ratio is low but vol_regime > 0.30, do NOT skip.
  - RSI between 46-54 AND no active Tier-A flag: No directional momentum. Skip.
  - vol_regime_score > 0.90: Historically extreme volatility (top 10%).
    Reduce conviction by 0.15, note heightened risk in rationale.

Step 2 -- TREND STRUCTURE:
  - Supertrend direction is your primary bias filter. Never take a position
    directly against a strong supertrend (ADX > 30) unless you have a reversal
    flag (sweep or engulf pattern).
  - EMA alignment: price > ema_21 > ema_200 = strong long bias.
    Price < ema_21 < ema_200 = strong short bias.
    Mixed (price between EMAs) = wait for momentum confirmation.
  - Price rank (100b): > 85% = overbought in local range (reduces long conviction).
    < 15% = oversold in local range (reduces short conviction).
  - Swing high/low: if close is within 0.3% of swing_high_10 = resistance
    overhead (longs need momentum to push through).
    If close near swing_low_10 = support test (shorts have limited downside).
  - Bollinger %B > 1.0: price above upper band -- short squeeze or breakout.
    Confirm with volume. < 0.0: below lower band -- breakdown or capitulation.

Step 3 -- MOMENTUM QUALITY:
  - RSI: 55-72 for long momentum, 28-45 for short momentum.
    RSI > 75: overbought, avoid fresh longs. RSI < 25: oversold, avoid fresh shorts.
  - MACD histogram: positive and growing = long momentum accelerating.
    Negative and shrinking = short momentum fading (potential reversal).
    Watch for histogram divergence vs price (momentum leading indicator).
  - Stochastic K: below 25 = oversold zone (long opportunity if trend is up).
    Above 75 = overbought zone (short opportunity if trend is down).
    Stoch crossing 20 upward in an uptrend = high-quality long signal.
  - ADX 18-25: emerging trend. 25-40: established trend. > 40: strong trend
    (but can be near exhaustion). Use TQI sign to confirm direction.

Step 4 -- MICROSTRUCTURE (critical for 15m timeframe):
  - CLV (close location value) range [-1, +1]:
    > +0.5 = strong buying pressure (bulls control close). Long bias.
    < -0.5 = strong selling pressure (bears pushed price down). Short bias.
    Near 0 = indecision, reduce conviction.
  - Efficiency ratio [0, 1]: > 0.65 = trending cleanly, good for momentum trades.
    < {eff_min:.2f} = choppy (but only SKIP if also vol_regime < 0.20 -- see Step 1).
    0.35-0.65 = moderate trend quality, proceed with lower conviction.
  - RVOL (relative volume): > 1.5 = unusual activity, breakout potential.
    < 0.5 = low interest, avoid. RVOL spike on a flag = high quality signal.
    NOTE: If the prompt says bar+0 is partially open, ignore current RVOL/NREI.
  - NREI (normalized range expansion): > 1.5 = expansion candle (momentum),
    > 2.5 = potential exhaustion spike (contrarian fade signal).
  - Up/Down volume ratio: > 1.5 = buying dominant. < 0.7 = selling dominant.
    Divergence from price direction = hidden distribution or accumulation.
  - Wick asymmetry: positive (upper_wick > lower_wick) = sellers overhead,
    reduces long conviction. Negative = buyers absorbing, supports longs.
  - Bull/bear streak: >= 5 consecutive closes in same direction = momentum carry.
    >= 8 = exhaustion warning (reversal risk increases).
  - TQI (trend quality index, signed ADX): > +20 = strong uptrend momentum.
    < -20 = strong downtrend momentum. Near 0 = directional uncertainty.
  - Vol regime score [0, 1]: 0 = historically quiet. 1 = historically extreme.
    Context: 0.3-0.7 is normal trading range for active markets.
  - Sharpe_20b range [-3, +3]: > 0.5 = positive momentum quality. < -0.5 = degrading.
    This is a 20-bar rolling mean/std ratio, NOT an annualized figure.

Step 5 -- FLAG INTERPRETATION:
  flag_volume_climax:  Extreme volume spike. Check direction via CLV and price action.
                       If CLV > 0: buyers overwhelming sellers (potential continuation or reversal).
                       If CLV < 0: distribution. High conviction reversal signal.
  flag_sweep_up:       Price swept above recent highs then rejected. Bearish.
                       Classic stop-hunt followed by reversal. Short bias.
  flag_sweep_dn:       Price swept below recent lows then recovered. Bullish.
                       Stop-hunt completed. Long bias.
  flag_macd_cross_up:  MACD line crossed above signal. Momentum turning positive.
                       Confirm with ADX > 20 and RSI > 50 for high quality.
  flag_macd_cross_dn:  MACD line crossed below signal. Momentum turning negative.
                       Confirm with ADX > 20 and RSI < 50.
  flag_adx_strong_trend: ADX just crossed 25. Trend is now established.
                       Trade with the direction of TQI (positive = long, negative = short).
  flag_golden_cross:   EMA-21 crossed above EMA-50. Bullish structure shift.
                       Best used for swing entries on pullbacks.
  flag_death_cross:    EMA-21 crossed below EMA-50. Bearish structure shift.
  flag_bb_squeeze:     Bollinger bandwidth compressed to 52-week low. Breakout imminent.
                       Direction unknown: wait for price to break BB boundary + volume.
  flag_rsi_divergence: Price making new high/low but RSI diverging. Reversal warning.
  NOTE: Flags marked "1b ago" fired on the previous closed bar -- this is normal timing.
        Flags marked ">= 6b ago" are stale and should be weighted down in your analysis.

Step 6 -- HIGHER TIMEFRAME BIAS:
  Always weight the provided HTF bias heavily:
  - HTF=bullish: add +0.10 to long confidence, subtract from short confidence.
  - HTF=bearish: add +0.10 to short confidence, subtract from long confidence.
  - HTF=mixed: reduce all confidence by 0.05 (no structural clarity).
  Social scan direction MUST match your action to get full confidence score.
  If social says "long" but your analysis shows short, cap confidence at 0.60.

Step 7 -- RECENT PRICE TRAJECTORY:
  Read the last 6 bars carefully. They tell you if the current bar is:
  - A continuation of existing momentum (act with trend)
  - A pullback into support/resistance (potential entry)
  - A topping/bottoming pattern (potential reversal)
  RVOL on recent bars matters: rising volume on green bars = accumulation.
  Rising volume on red bars = distribution.

=== CONFIDENCE CALIBRATION ===
  0.90-1.00: All signals aligned, clear catalyst, high volume, HTF agreement. Rare.
  0.75-0.89: Strong setup, most signals agree, minor conflicts.
  0.60-0.74: Reasonable setup, some uncertainty. Default action zone.
  < 0.60:    Skip. Do not trade uncertainty.

=== OUTPUT ===
Return ONLY valid JSON. No markdown. No prose outside the JSON object.\
"""

# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(*, symbol: str, context: str) -> str:
    return (
        f"Analyse {symbol} (15m Bybit perp) and decide: long, short, or skip.\n\n"
        f"{context}\n\n"
        "Apply your 7-step framework. Return JSON:\n"
        "{\n"
        '  "symbol": "XXXUSDT",\n'
        '  "action": "long|short|skip",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "time_horizon": "scalp_1_4h|swing_4_24h|position_1_3d",\n'
        '  "regime": "trending|choppy|volatile|reversing",\n'
        '  "rationale": "2-3 sentences: which signals align, why this setup has edge now",\n'
        '  "invalidation": "1 sentence: specific price action that would prove this thesis wrong"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Main call function
# ---------------------------------------------------------------------------

async def chat_deep_analysis(
    client:  AIClient,
    cfg:     Config,
    *,
    symbol:  str,
    context: str,
) -> DeepAnalysis:
    """Execute the per-trigger deep analysis prompt (Prompt B).

    Args:
        client:  Initialised AIClient.
        cfg:     Loaded Config (provides model_deep).
        symbol:  Ticker symbol being analysed.
        context: Token-efficient context block from build_deep_context().

    Returns:
        Validated DeepAnalysis. Returns default_deep_analysis(symbol) (skip)
        on any failure -- never acts on broken AI response.
    """
    model = cfg.model_deep or "x-ai/grok-4.1-fast"

    adx_min = float(getattr(cfg, "adx_min_trade", 18.0))
    eff_min = float(getattr(cfg, "efficiency_min_trade", 0.35))
    system_prompt = (
        _SYSTEM_TEMPLATE
        .replace("{adx_min:.0f}", f"{adx_min:.0f}")
        .replace("{eff_min:.2f}", f"{eff_min:.2f}")
    )
    messages = [
        Message("system", system_prompt),
        Message("user", _build_user_prompt(symbol=symbol, context=context)),
    ]

    log.info("deep_analysis start symbol=%s model=%s", symbol, model)

    try:
        response = await client.chat(
            model      = model,
            messages   = messages,
            call_type  = "deep_analysis",
            symbol     = symbol,
            max_tokens = 500,
            web_search = False,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("deep_analysis failed symbol=%s: %s", symbol, exc)
        return default_deep_analysis(symbol)

    parsed, err = parse_json_response(response)
    if err or parsed is None:
        log.warning("deep_analysis parse error symbol=%s: %s | raw=%r",
                    symbol, err, response[:200])
        return default_deep_analysis(symbol)

    result = validate_deep_analysis(parsed, symbol)
    if isinstance(result, str):
        log.warning("deep_analysis validation error symbol=%s: %s", symbol, result)
        return default_deep_analysis(symbol)

    # Enforce minimum confidence gate.
    if result["action"] != "skip" and result["confidence"] < MIN_CONFIDENCE:
        log.info(
            "deep_analysis below threshold symbol=%s confidence=%.2f -> skip",
            symbol, result["confidence"],
        )
        result = {**result, "action": "skip"}  # type: ignore[assignment]

    log.info(
        "deep_analysis ok symbol=%s action=%s confidence=%.2f horizon=%s regime=%s",
        symbol, result["action"], result["confidence"],
        result.get("time_horizon", "?"), result.get("regime", "?"),
    )
    return result  # type: ignore[return-value]
