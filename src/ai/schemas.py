"""Structured JSON schemas for AI responses.

Every AI call in the pipeline returns JSON that must match one of these
TypedDicts.  We use TypedDicts (not Pydantic) to keep things fast and
dependency-light.  Validation is done via ``validate_*`` functions that
return either a clean object or a clear error string.

Three response types match the three pipeline stages:

SocialScan   -- 6h Grok social scan: top 5 directional picks + reasoning.
DeepAnalysis -- Flag-triggered deep analysis: entry decision with risk params.
PositionReview -- Review prompt: hold / tighten / exit verdict.

All fields are str/float/int/bool -- no nested objects except ``picks``.
The AI must always return valid JSON.  If it doesn't, ``validate_*``
returns an error string and the caller falls back to a safe default.
"""
from __future__ import annotations

import json
from typing import Any, Literal, TypedDict


# ---------------------------------------------------------------------------
# 1. Social Scan (runs every 6h)
# ---------------------------------------------------------------------------

class Pick(TypedDict):
    symbol:    str    # e.g. "BTCUSDT"
    direction: str    # "long" | "short"
    conviction: float # 0.0 - 1.0
    reason:    str    # 1-2 sentence human rationale
    sources:   list[str]  # non-empty list of source types e.g. ["twitter", "reddit"]


class SocialScan(TypedDict):
    scan_type:    Literal["social_scan"]
    as_of:        str          # ISO-8601 UTC
    market_mood:  str          # "bullish" | "bearish" | "mixed" | "neutral"
    mood_score:   float        # -1.0 (extreme fear) to +1.0 (extreme greed)
    picks:        list[Pick]   # exactly 1-5 items
    noise_level:  str          # "low" | "medium" | "high" -- how much trash was filtered
    model_note:   str          # any caveat the model wants to surface


def validate_social_scan(raw: Any) -> SocialScan | str:
    """Validate and coerce a raw dict into SocialScan.

    Returns the validated TypedDict, or an error string if invalid.
    """
    if not isinstance(raw, dict):
        return "response is not a JSON object"

    required = ("market_mood", "mood_score", "picks")
    for field in required:
        if field not in raw:
            return f"missing required field: {field!r}"

    picks = raw.get("picks", [])
    if not isinstance(picks, list) or not picks:
        return "picks must be a non-empty list"
    if len(picks) > 5:
        picks = picks[:5]

    validated_picks: list[Pick] = []
    for i, p in enumerate(picks):
        if not isinstance(p, dict):
            return f"picks[{i}] is not an object"
        if not p.get("symbol"):
            return f"picks[{i}] missing symbol"
        direction = str(p.get("direction", "")).lower()
        if direction not in ("long", "short"):
            return f"picks[{i}].direction must be 'long' or 'short', got {direction!r}"
        try:
            conviction = float(p.get("conviction", 0.5))
            conviction = max(0.0, min(1.0, conviction))
        except (TypeError, ValueError):
            conviction = 0.5
        validated_picks.append(Pick(
            symbol    = str(p["symbol"]).upper().strip(),
            direction = direction,
            conviction= conviction,
            reason    = str(p.get("reason", ""))[:500],
            sources   = [str(s) for s in p.get("sources", [])] or ["unknown"],
        ))

    try:
        mood_score = float(raw.get("mood_score", 0.0))
        mood_score = max(-1.0, min(1.0, mood_score))
    except (TypeError, ValueError):
        mood_score = 0.0

    noise = str(raw.get("noise_level", "medium")).lower()
    if noise not in ("low", "medium", "high"):
        noise = "medium"

    mood = str(raw.get("market_mood", "mixed")).lower()
    if mood not in ("bullish", "bearish", "mixed", "neutral"):
        mood = "mixed"

    return SocialScan(
        scan_type   = "social_scan",
        as_of       = str(raw.get("as_of", "")),
        market_mood = mood,
        mood_score  = mood_score,
        picks       = validated_picks,
        noise_level = noise,
        model_note  = str(raw.get("model_note", ""))[:300],
    )


# ---------------------------------------------------------------------------
# 2. Deep Analysis (runs on flag fire, per symbol)
# ---------------------------------------------------------------------------

class DeepAnalysis(TypedDict):
    analysis_type: Literal["deep_analysis"]
    symbol:        str
    action:        str    # "long" | "short" | "skip"
    confidence:    float  # 0.0 - 1.0
    entry:         float  # suggested entry price (0 = market)
    stop_loss:     float  # hard stop price
    take_profit_1: float  # first target (partial close)
    take_profit_2: float  # final target (full close)
    time_horizon:  str    # "scalp" | "swing" | "position"
    rationale:     str    # why this setup
    invalidation:  str    # what would invalidate the thesis
    risk_reward:   float  # estimated R:R ratio


def validate_deep_analysis(raw: Any, symbol: str) -> DeepAnalysis | str:
    """Validate a raw dict into DeepAnalysis.

    Args:
        raw:    Dict parsed from model response.
        symbol: Expected symbol (used as fallback).

    Returns the validated TypedDict, or an error string.
    """
    if not isinstance(raw, dict):
        return "response is not a JSON object"

    action = str(raw.get("action", "skip")).lower()
    if action not in ("long", "short", "skip"):
        action = "skip"

    try:
        confidence = float(raw.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.0

    def _float(key: str, default: float = 0.0) -> float:
        try:
            return float(raw.get(key, default))
        except (TypeError, ValueError):
            return default

    entry       = _float("entry")
    stop_loss   = _float("stop_loss")
    tp1         = _float("take_profit_1")
    tp2         = _float("take_profit_2")
    risk_reward = _float("risk_reward", 0.0)

    # Sanity: if action is not skip, key levels must be non-zero.
    if action != "skip":
        if stop_loss <= 0 or tp1 <= 0:
            return f"action={action!r} but stop_loss/take_profit_1 are zero -- invalid"

    horizon = str(raw.get("time_horizon", "swing")).lower()
    if horizon not in ("scalp", "swing", "position"):
        horizon = "swing"

    return DeepAnalysis(
        analysis_type = "deep_analysis",
        symbol        = str(raw.get("symbol", symbol)).upper(),
        action        = action,
        confidence    = confidence,
        entry         = entry,
        stop_loss     = stop_loss,
        take_profit_1 = tp1,
        take_profit_2 = tp2,
        time_horizon  = horizon,
        rationale     = str(raw.get("rationale", ""))[:1000],
        invalidation  = str(raw.get("invalidation", ""))[:500],
        risk_reward   = risk_reward,
    )


# ---------------------------------------------------------------------------
# 3. Position Review (runs on regime flip, drawdown, TP1, funding approach)
# ---------------------------------------------------------------------------

class PositionReview(TypedDict):
    review_type:    Literal["position_review"]
    symbol:         str
    action:         str    # "hold" | "tighten_stop" | "scale_out" | "exit" | "flip"
    confidence:     float  # 0.0 - 1.0
    new_stop_loss:  float  # only relevant when action="tighten_stop" (0 = no change)
    rationale:      str    # why this verdict
    urgency:        str    # "low" | "medium" | "high" -- how fast to act


def validate_position_review(raw: Any, symbol: str) -> PositionReview | str:
    """Validate a raw dict into PositionReview."""
    if not isinstance(raw, dict):
        return "response is not a JSON object"

    action = str(raw.get("action", "hold")).lower()
    valid_actions = ("hold", "tighten_stop", "scale_out", "exit", "flip")
    if action not in valid_actions:
        action = "hold"

    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    try:
        new_sl = float(raw.get("new_stop_loss", 0.0))
    except (TypeError, ValueError):
        new_sl = 0.0

    urgency = str(raw.get("urgency", "medium")).lower()
    if urgency not in ("low", "medium", "high"):
        urgency = "medium"

    return PositionReview(
        review_type   = "position_review",
        symbol        = str(raw.get("symbol", symbol)).upper(),
        action        = action,
        confidence    = confidence,
        new_stop_loss = new_sl,
        rationale     = str(raw.get("rationale", ""))[:500],
        urgency       = urgency,
    )


# ---------------------------------------------------------------------------
# Safe defaults (returned when validation fails)
# ---------------------------------------------------------------------------

def default_social_scan() -> SocialScan:
    """Empty safe social scan -- no picks, neutral mood."""
    return SocialScan(
        scan_type   = "social_scan",
        as_of       = "",
        market_mood = "neutral",
        mood_score  = 0.0,
        picks       = [],
        noise_level = "high",
        model_note  = "synthetic-default: response invalid or budget exhausted",
    )


def default_deep_analysis(symbol: str) -> DeepAnalysis:
    """Safe skip -- do not trade."""
    return DeepAnalysis(
        analysis_type = "deep_analysis",
        symbol        = symbol,
        action        = "skip",
        confidence    = 0.0,
        entry         = 0.0,
        stop_loss     = 0.0,
        take_profit_1 = 0.0,
        take_profit_2 = 0.0,
        time_horizon  = "swing",
        rationale     = "synthetic-default: response invalid or budget exhausted",
        invalidation  = "n/a",
        risk_reward   = 0.0,
    )


def default_position_review(symbol: str) -> PositionReview:
    """Safe hold -- keep the position open."""
    return PositionReview(
        review_type   = "position_review",
        symbol        = symbol,
        action        = "hold",
        confidence    = 0.0,
        new_stop_loss = 0.0,
        rationale     = "synthetic-default: response invalid or budget exhausted",
        urgency       = "low",
    )


# ---------------------------------------------------------------------------
# JSON parse helper
# ---------------------------------------------------------------------------

def parse_json_response(content: str) -> tuple[dict | None, str | None]:
    """Try to parse the model's raw string as JSON.

    Returns (parsed_dict, None) on success, (None, error_message) on failure.
    Handles markdown code fences that some models add around JSON.
    """
    text = content.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        text  = "\n".join(lines).strip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"
