"""Prompt C -- open position review.

Called by hooks in the exec loop when specific events require active oversight:
  hook_reason values:
    "tp1_hit"          -- TP1 reached, runner still open. Manage the trail.
    "drawdown_60pct"   -- Price moved 60% of the way to stop-loss.
    "trend_flip"       -- Supertrend flipped against the position.
    "funding_adverse"  -- Funding rate > 0.03% against position direction.
    "time_exit_8h"     -- Position held > 8h without hitting TP1. Stale trade.
    "manual"           -- Operator-triggered review.

SAFETY CONTRACTS (ABSOLUTE -- NEVER OVERRIDE):
  1. On ANY failure (network, parse, validation, schema mismatch): return "hold".
     We never exit a live position on a broken AI response.
  2. "tighten_stop" may only IMPROVE the stop (move it in the direction of profit).
     The hard SL set by RiskEngine at entry is the floor -- we never widen it.
  3. Below 0.55 confidence: always return "hold". No action on uncertainty.
  4. "exit" is a serious action. Reserve for clear thesis invalidation.
  5. "flip" is extremely rare and should only occur when a strong counter-setup
     is confirmed on the same bar.

Cost: ~$0.00018/call (Grok 4.1 Fast, ~550 in / 150 out tokens).
"""
from __future__ import annotations

import logging

from ai.client import AIClient
from ai.http import Message
from ai.schemas import (
    PositionReview,
    default_position_review,
    parse_json_response,
    validate_position_review,
)
from core.config import Config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a professional crypto risk manager reviewing an open Bybit USDT \
perpetual futures position on the 15-minute timeframe. Your sole \
responsibility is protecting this position from unnecessary loss while \
not closing profitable runners prematurely.

You operate in DEFENSE mode. NOT looking for new setups.
Key questions: (1) Is the original thesis still valid? \
(2) Can we lock in more profit? (3) Is it time to cut?

=== TRADE STAGE CONTEXT ===
You will see a "remaining_size" field:
  100% = full position, TP1 not yet hit. Be patient but vigilant.
   50% = TP1 already hit. We trade with house money. Give much more room.
    0% = runner only (TP1 and TP2 hit). Only trail.

"bars_held" matters:
  < 4 bars  = too early. Give the trade room. Default: hold.
  4-16 bars = active management window. Apply framework normally.
  > 32 bars = approaching stale threshold. Raise bar for holding.

=== DECISION FRAMEWORK ===

--- HOLD ---
DEFAULT. Return "hold" unless you have STRONG, SPECIFIC evidence to act.
Conditions that override discomfort -- HOLD regardless:
  - bars_held < 4 and no severe adverse move (< 30% of SL distance).
  - TP1 already hit (remaining_size <= 50%) and supertrend still aligned.
  - Supertrend intact AND efficiency >= 0.40: trend still has legs.
  - Price within 1x ATR of a support EMA (ema_21 for longs, ema_21 for shorts).
  - CLV > 0 for longs / CLV < 0 for shorts: buyers/sellers still in control.
  - RSI in directional zone (> 50 for longs / < 50 for shorts).

--- TIGHTEN_STOP ---
Move stop-loss closer to price to lock in profit.
WHEN TO USE:
  a) unrealized_pnl > 0 AND bars_held >= 6 AND efficiency >= 0.40.
  b) Price has moved at least 1.5x ATR beyond entry.
  c) A clear structural level (EMA, swing high/low, VWAP) sits between
     current stop and price -- place new stop just below that level.
  d) NOT during active expansion bars (rvol > 1.5 or nrei > 1.5).
     Wait for bar to close before adjusting.
CRITICAL: "new_stop_loss" MUST be between current stop and current price.
  For longs: new_stop > current_stop and new_stop < mark_price.
  For shorts: new_stop < current_stop and new_stop > mark_price.

--- SCALE_OUT ---
Close 50% of remaining position NOW.
ONLY when TP1 is already hit AND BOTH conditions are true:
  (a) Momentum fading hard: RSI_trend shows rolling over from overbought
      (long: RSI was > 65, now declining) AND CLV turned negative (< -0.3).
  (b) Runner is NOT running: price stalled within 0.5x ATR above TP1 for
      more than 4 bars -- the continuation momentum never materialized.
Never scale out in the first 4 bars.

--- EXIT ---
Close 100% of remaining position.
Reserve ONLY for clear structural invalidation:
  - Supertrend has flipped AGAINST position AND ADX > 25 (confirmed trend, not noise).
  - RSI crossed 50 decisively against direction (long: RSI < 47 AND declining).
  - efficiency < 0.30 AND price has moved > 0.5x ATR against position.
  - adverse_move > 50% of original SL distance AND no key structural level defending.
  - hook="time_exit_8h" AND unrealized_pnl <= 0 AND regime is choppy (efficiency < 0.40).
  - funding_adverse: funding rate > 8bps against direction AND TP2 is > 2x ATR away.
    Holding a losing position into adverse funding is a double penalty. Cut if underwater.
  - key_level broken: for longs, close < ema_21 AND close < vwap AND vol_regime > 0.5.
Do NOT exit because: one red candle, discomfort, or "it might reverse".

--- FLIP ---
Close current position + signal counter-position. Extremely rare.
ALL of these must be true simultaneously:
  - Supertrend flipped against position.
  - A Tier-A flag (sweep, volume climax) firing in opposite direction.
  - HTF bias now aligned with new direction.
  - Fresh deep analysis confidence for counter-trade would be >= 0.70.
If ANY condition is missing: use EXIT, not FLIP.

=== STRUCTURAL LEVELS INTERPRETATION ===
Use the provided key levels to make your stop placement and thesis decisions:
  - ema_21: primary dynamic support/resistance on 15m.
    Long position: ema_21 acts as support. Break below = warning.
  - swing_low_10 (long) / swing_high_10 (short): nearest structural pivot.
    Used for stop placement and thesis invalidation.
  - vwap: intraday fair value. Price below vwap = bearish bias.
  - For tighten_stop on longs: typical targets are swing_low_10, ema_21,
    or last consolidation low visible in recent bars.

=== HOOK CONTEXT ===
  tp1_hit:        Focus on protecting runner. Hold OR tighten stop below key level.
  drawdown_60pct: Critical moment. Is this retracement or reversal? Check supertrend + ADX.
  trend_flip:     Supertrend flipped. Evaluate: whipsaw (ADX < 20) or real (ADX > 25)?
  funding_adverse: High funding cost. Check direction (long paying = bearish signal).
                   Factor into exit threshold: lower bar for exit if already in loss.
  time_exit_8h:   Stale trade protocol. Honest assessment: is there a reason to stay?
                  If pnl <= 0 and no momentum, exit cleanly. Cut losers, let winners run.

=== RSI TREND INTERPRETATION ===
The context provides current rsi_14 and rsi_3bar_ago. Use these together:
  - rsi_3bar_ago > rsi_14 + 5: RSI rolling over (momentum fading). Warning for longs.
  - rsi_14 > rsi_3bar_ago + 5: RSI building (momentum strengthening). Good for longs.
  - RSI crossing 50: directional change signal. Treat as soft exit trigger.

=== FUNDING RATE CONTEXT ===
  Positive funding (longs pay): > +5bps is elevated, > +10bps is extreme.
    For long positions: high positive funding HURTS us -- raises exit threshold.
    For short positions: high positive funding HELPS us (shorts receive).
  Negative funding (shorts pay): < -5bps is elevated, < -15bps is extreme.
    For short positions: extreme negative funding is a squeeze risk.

=== CONFIDENCE CALIBRATION ===
  0.85-1.00: Crystal clear structural evidence. Multiple signals agree. Use sparingly.
  0.65-0.84: Good evidence, minor uncertainty.
  {min_conf:.2f}-0.64: Borderline. Hold is fine here. Exit/flip need 0.65+.
  < {min_conf:.2f}:    Return "hold". Never act on uncertainty.

Return ONLY valid JSON. No markdown. No prose outside the JSON object.
{
  "symbol": "XXXUSDT",
  "action": "hold|tighten_stop|scale_out|exit|flip",
  "confidence": 0.0-1.0,
  "new_stop_loss": price_if_tighten_else_0,
  "urgency": "low|medium|high",
  "thesis_intact": true_or_false,
  "rationale": "2-3 sentences: specific evidence referencing actual data values"
}\
"""

# ---------------------------------------------------------------------------
# User prompt builder
# ---------------------------------------------------------------------------

def _build_user_prompt(*, symbol: str, context: str) -> str:
    return (
        f"Review the open {symbol} position and decide the next action.\n\n"
        f"{context}\n\n"
        "Apply your decision framework. Return JSON:\n"
        "{\n"
        '  "symbol": "XXXUSDT",\n'
        '  "action": "hold|tighten_stop|scale_out|exit|flip",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "new_stop_loss": price_if_tighten_else_0,\n'
        '  "urgency": "low|medium|high",\n'
        '  "thesis_intact": true_or_false,\n'
        '  "rationale": "2-3 sentences citing specific indicator values"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Main call function
# ---------------------------------------------------------------------------

async def chat_position_review(
    client:  AIClient,
    cfg:     Config,
    *,
    symbol:  str,
    context: str,
) -> PositionReview:
    """Execute a position review prompt (Prompt C).

    Args:
        client:  Initialised AIClient.
        cfg:     Loaded Config (provides model_review).
        symbol:  Ticker symbol of the open position.
        context: Context string from build_review_context().

    Returns:
        Validated PositionReview. Falls back to default_position_review(symbol)
        (action="hold") on ANY failure -- the safety contract is absolute.
    """
    model = cfg.model_review or "x-ai/grok-4.1-fast"

    min_conf = float(getattr(cfg, "min_action_confidence", 0.55))
    system_prompt = _SYSTEM_TEMPLATE.replace("{min_conf:.2f}", f"{min_conf:.2f}")
    messages = [
        Message("system", system_prompt),
        Message("user", _build_user_prompt(symbol=symbol, context=context)),
    ]

    log.info("position_review start symbol=%s model=%s", symbol, model)

    try:
        response = await client.chat(
            model      = model,
            messages   = messages,
            call_type  = "position_review",
            symbol     = symbol,
            max_tokens = 350,
            web_search = False,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("position_review failed symbol=%s: %s -- hold", symbol, exc)
        return default_position_review(symbol)

    parsed, err = parse_json_response(response)
    if err or parsed is None:
        log.warning("position_review parse error symbol=%s: %s | raw=%r",
                    symbol, err, response[:200])
        return default_position_review(symbol)

    result = validate_position_review(parsed, symbol)
    if isinstance(result, str):
        log.warning("position_review validation error symbol=%s: %s", symbol, result)
        return default_position_review(symbol)

    # Safety gate: low confidence -> hold unconditionally.
    if result["action"] != "hold" and result["confidence"] < min_conf:
        log.info(
            "position_review below threshold symbol=%s confidence=%.2f < %.2f -> hold",
            symbol, result["confidence"], min_conf,
        )
        result = {**result, "action": "hold"}  # type: ignore[assignment]

    log.info(
        "position_review ok symbol=%s action=%s confidence=%.2f urgency=%s intact=%s",
        symbol, result["action"], result["confidence"],
        result.get("urgency", "?"), result.get("thesis_intact", "?"),
    )
    return result  # type: ignore[return-value]
