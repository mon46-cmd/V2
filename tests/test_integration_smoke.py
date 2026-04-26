"""Integration smoke test -- verifies all new modules import and compute correctly."""
from core.config import load_config
from ai.calls import (
    chat_social_scan, chat_deep_analysis, chat_position_review,
    build_deep_context, build_review_context,
)
from loops import (
    UniverseBuilder, WatchlistStore, CooldownStore,
    detect_trigger, TriggerDecision, FLAG_TIERS,
)
from portfolio.risk import RiskEngine, TrailingStop
from ai.pricing import cost_usd

cfg    = load_config()
engine = RiskEngine(cfg)

# 1. Risk engine levels
result = engine.compute(side="long", entry=67200.0, atr=810.0, equity_usd=10000.0)
assert result.accepted, f"Expected accepted, got: {result.reason}"
assert result.levels.stop_loss < 67200, f"Long SL should be below entry"
assert result.levels.tp1 > 67200, f"Long TP1 should be above entry"
assert result.levels.tp2 > result.levels.tp1, f"TP2 should be above TP1"
print(f"[OK] RiskEngine: SL={result.levels.stop_loss:.0f}  TP1={result.levels.tp1:.0f}  TP2={result.levels.tp2:.0f}  qty=${result.qty_usd:.2f}")

# 2. Circuit breaker
r0 = engine.compute(side="long", entry=67200.0, atr=810.0, equity_usd=10000.0, loser_streak=0)
r3 = engine.compute(side="long", entry=67200.0, atr=810.0, equity_usd=10000.0, loser_streak=3)
r5 = engine.compute(side="long", entry=67200.0, atr=810.0, equity_usd=10000.0, loser_streak=5)
assert r0.risk_mult == 1.0
assert r3.risk_mult == 0.5
assert r5.risk_mult == 0.25
print(f"[OK] Circuit breaker: streak=0->{r0.risk_mult}  streak=3->{r3.risk_mult}  streak=5->{r5.risk_mult}")

# 3. Trailing stop
ts = TrailingStop(side="long", initial_sl=66500.0, trail_atr_mult=1.0)
ts.activate(68500.0)
sl_at_69k = ts.update(mark=69000.0, atr=810.0)
assert sl_at_69k > 66500, f"Trailing SL should have moved up from initial"
assert not ts.is_hit(68300), f"68300 should not have hit the trailing stop"
print(f"[OK] TrailingStop: initial=66500 -> after mark 69000: sl={sl_at_69k:.0f}")

# 4. Pricing
cost_social = cost_usd("x-ai/grok-4.20", 700, 400)
cost_deep   = cost_usd("x-ai/grok-4.1-fast", 650, 200)
cost_review = cost_usd("x-ai/grok-4.1-fast", 450, 120)
print(f"[OK] Pricing: social=${cost_social:.5f}  deep=${cost_deep:.5f}  review=${cost_review:.5f}")
print(f"     Daily est (4+5+6 calls): ${cost_social*4 + cost_deep*5 + cost_review*6:.4f}")

# 5. Flag tiers
assert "flag_volume_climax" in FLAG_TIERS["A"]
assert len(FLAG_TIERS["A"]) >= 4
print(f"[OK] FLAG_TIERS: A={len(FLAG_TIERS['A'])} flags, B={len(FLAG_TIERS['B'])} flags")

# 6. Trigger gate (pure function)
import pandas as pd
bar = {
    "timestamp": pd.Timestamp("2026-04-26 12:00:00", tz="UTC"),
    "close": 67200.0,
    "atr_14_pct": 1.21,
    "flag_volume_climax": 1.0,
}
from loops.cooldowns import CooldownState
state = CooldownState()
dec = detect_trigger(symbol="BTCUSDT", bar=bar, state=state, cfg=cfg)
assert dec.fired, f"Expected trigger to fire, got: {dec.decision}"
assert dec.flag == "flag_volume_climax"
print(f"[OK] detect_trigger: decision={dec.decision}  flag={dec.flag}")

# 7. Context builder
ctx_deep = build_deep_context(
    row={"close": 67200.0, "atr_14_pct": 1.21, "rsi_14": 62.3, "supertrend_dir": 1.0},
    symbol="BTCUSDT",
    as_of="2026-04-26T12:00:00Z",
    social_reason="ETF inflows",
    watchlist_direction="long",
    fired_flag="flag_volume_climax",
)
assert "BTCUSDT" in ctx_deep
assert "PRICE & STRUCTURE" in ctx_deep
print(f"[OK] build_deep_context: {len(ctx_deep)} chars (~{len(ctx_deep)//4} tokens)")

ctx_review = build_review_context(
    symbol="BTCUSDT", side="long", entry=67200.0, stop_loss=66000.0,
    tp1=68800.0, tp2=71400.0, mark=68100.0, remaining_pct=50.0,
    bars_held=8, loser_streak=0, hook_reason="tp1_hit",
    row={"supertrend_dir": 1.0, "rsi_14": 58.1, "clv": 0.31, "atr_14_pct": 1.21},
)
assert "tp1_hit" in ctx_review
assert "POSITION STATE" in ctx_review
print(f"[OK] build_review_context: {len(ctx_review)} chars (~{len(ctx_review)//4} tokens)")

print("\nALL SMOKE TESTS PASSED")
