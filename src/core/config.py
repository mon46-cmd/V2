"""Central configuration.

Values come exclusively from environment variables (optionally via a .env
file).  Nothing here performs network I/O -- this module is a pure typed
container and must never import from outside core.

Typical usage (once per process, at the entry point):

    from core import load_config
    cfg = load_config()

Pass `cfg` around explicitly; do not store it in a module-level global
outside of core itself.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load the local .env if present; explicit path so we are not at the
    # mercy of the current working directory at import time.
    _DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    if _DOTENV_PATH.is_file():
        load_dotenv(_DOTENV_PATH)
except ImportError as _exc:  # pragma: no cover - dependency listed in requirements
    import warnings
    warnings.warn(
        f"python-dotenv not installed ({_exc}); falling back to raw env vars only. "
        "Install requirements.txt to load .env files.",
        RuntimeWarning,
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# This file lives at  src/core/config.py  so two parents up is the repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def _env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default) or default


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int = 0) -> int:
    raw = os.getenv(key, "")
    if not raw:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    raw = os.getenv(key, "")
    if not raw:
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key, "")
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    """Typed, immutable snapshot of the runtime environment.

    Fields are grouped by concern.  Every field has a conservative
    default so the system can boot and fail gracefully when variables
    are missing rather than crashing at import time.
    """

    # --- Repository layout ---
    repo_root:    Path = REPO_ROOT
    data_root:    Path = field(default_factory=lambda: REPO_ROOT / "data")
    cache_root:   Path = field(default_factory=lambda: REPO_ROOT / "data" / "cache")
    feature_root: Path = field(default_factory=lambda: REPO_ROOT / "data" / "features")
    run_root:     Path = field(default_factory=lambda: REPO_ROOT / "data" / "runs")
    log_root:     Path = field(default_factory=lambda: REPO_ROOT / "data" / "logs")

    # --- AI / OpenRouter ---
    openrouter_api_key:  str   = ""
    openrouter_base_url: str   = "https://openrouter.ai/api/v1"
    openrouter_referer:  str   = ""
    openrouter_title:    str   = ""
    model_social:        str   = "x-ai/grok-4.20"       # 6h social scan
    model_deep:          str   = "x-ai/grok-4.1-fast"   # per-trigger analysis
    model_review:        str   = "x-ai/grok-4.1-fast"   # position review
    model_watchlist:     str   = ""                      # legacy alias
    ai_timeout_sec:      float = 90.0
    ai_dry_run:          bool  = False
    openrouter_live:     bool  = False
    ai_kill_switch:      bool  = False
    daily_budget_usd:    float = 1.0

    # --- Exchange (Bybit) ---
    bybit_api_key:      str   = ""
    bybit_api_secret:   str   = ""
    bybit_rest_base:    str   = "https://api.bybit.com"
    bybit_archive_base: str   = "https://public.bybit.com/trading"
    category:           str   = "linear"
    quote_currency:     str   = "USDT"
    bybit_offline:      bool  = False

    # --- HTTP ---
    http_timeout_sec:      float = 20.0
    http_rate_delay_sec:   float = 0.10
    http_max_retries:      int   = 5
    http_backoff_base_sec: float = 0.5

    # --- Paper broker ---
    paper_equity_usd:         float = 10_000.0
    per_trade_risk_pct:       float = 0.01
    max_concurrent_positions: int   = 3
    taker_fee_bps:            float = 6.0
    slippage_bps:             float = 2.0
    tp1_scale_out_pct:        float = 0.50

    # --- Risk engine (portfolio/risk.py) ---
    atr_mult_sl:          float = 1.5    # stop = entry ± 1.5 * ATR
    tp1_rr:               float = 1.5    # TP1 at 1.5R
    tp2_rr:               float = 3.0    # TP2 at 3.0R
    risk_pct_per_trade:   float = 0.01   # 1% of equity risked per trade
    max_open_positions:   int   = 3      # max concurrent open positions
    max_position_usd:     float = 5_000  # max notional per position
    max_drawdown_pct:     float = 0.05   # halt new entries above this DD from peak (0=disabled)
    bar_max_age_sec:      int   = 0      # skip snapshot if last bar older than this (0=disabled)

    # --- AI prompt thresholds (templated into system prompts) ---
    adx_min_trade:         float = 18.0   # deep_analysis: skip below this ADX
    efficiency_min_trade:  float = 0.35   # deep_analysis: skip below this efficiency
    min_action_confidence: float = 0.55   # position_review: hold below this confidence

    # --- Universe filter ---
    universe_size:         int   = 30
    min_turnover_usd_24h:  float = 20_000_000.0
    max_spread_bps:        float = 10.0
    min_listing_age_days:  int   = 30
    min_price_usd:         float = 0.0001
    exclude_symbols:       tuple[str, ...] = field(default_factory=lambda: (
        "USDCUSDT", "USDEUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT",
    ))
    exclude_substrings: tuple[str, ...] = field(default_factory=lambda: ("-",))

    # --- Timeframes ---
    watch_interval: str           = "15"
    higher_tfs:     tuple[str, ...] = ("60", "240")

    # --- Scanner / cooldowns ---
    trigger_flags: tuple[str, ...] = field(default_factory=lambda: (
        "flag_volume_climax", "flag_sweep_up", "flag_sweep_dn",
    ))
    prompt_cooldown_candles:    int   = 3
    cooldown_bypass_atr_mult:   float = 0.8
    cooldown_bypass_floor_pct:  float = 0.01
    scanner_watchlist_reuse_sec: int  = 900

    # --- Feature engine ---
    snapshot_concurrency: int = 6

    # --- Logging ---
    log_level: str = "INFO"

    # --- HTTP API ---
    api_host: str = "127.0.0.1"
    api_port: int = 8765


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def load_config() -> Config:
    """Build the frozen Config from the current environment and ensure
    the data directory tree exists.

    Call once at process startup.
    """
    data_root = _env_path("DATA_ROOT", REPO_ROOT / "data")

    cfg = Config(
        # layout
        repo_root    = REPO_ROOT,
        data_root    = data_root,
        cache_root   = data_root / "cache",
        feature_root = data_root / "features",
        run_root     = data_root / "runs",
        log_root     = data_root / "logs",
        # AI
        openrouter_api_key  = _env_str("OPENROUTER_API_KEY"),
        openrouter_referer  = _env_str("OPENROUTER_REFERER"),
        openrouter_title    = _env_str("OPENROUTER_TITLE"),
        model_social        = _env_str("MODEL_SOCIAL",  "x-ai/grok-4.20"),
        model_deep          = _env_str("MODEL_DEEP",    "x-ai/grok-4.1-fast"),
        model_review        = _env_str("MODEL_REVIEW",  "x-ai/grok-4.1-fast"),
        model_watchlist     = _env_str("MODEL_WATCHLIST"),
        ai_dry_run          = _env_bool("AI_DRY_RUN"),
        openrouter_live     = _env_bool("OPENROUTER_LIVE"),
        ai_kill_switch      = _env_bool("AI_KILL_SWITCH"),
        daily_budget_usd    = _env_float("DAILY_BUDGET_USD", 1.0),
        # risk engine
        atr_mult_sl        = _env_float("ATR_MULT_SL",        1.5),
        tp1_rr             = _env_float("TP1_RR",             1.5),
        tp2_rr             = _env_float("TP2_RR",             3.0),
        risk_pct_per_trade = _env_float("RISK_PCT_PER_TRADE", 0.01),
        max_open_positions = _env_int(  "MAX_OPEN_POSITIONS",  3),
        max_position_usd   = _env_float("MAX_POSITION_USD",   5_000),
        max_drawdown_pct   = _env_float("MAX_DRAWDOWN_PCT",   0.05),
        bar_max_age_sec    = _env_int(  "BAR_MAX_AGE_SEC",      0),
        adx_min_trade         = _env_float("ADX_MIN_TRADE",         18.0),
        efficiency_min_trade  = _env_float("EFFICIENCY_MIN_TRADE",  0.35),
        min_action_confidence = _env_float("MIN_ACTION_CONFIDENCE", 0.55),
        # exchange
        bybit_api_key    = _env_str("BYBIT_API_KEY"),
        bybit_api_secret = _env_str("BYBIT_API_SECRET"),
        bybit_offline    = _env_bool("BYBIT_OFFLINE"),
        # scanner cache
        scanner_watchlist_reuse_sec = _env_int("SCANNER_WATCHLIST_REUSE_SEC", 900),
        # logging / API
        log_level = _env_str("LOG_LEVEL", "INFO").upper(),
        api_host  = _env_str("API_HOST", "127.0.0.1"),
        api_port  = _env_int("API_PORT",  8765),
    )

    # Ensure directories exist as a side-effect.
    from .paths import ensure_dirs
    ensure_dirs(cfg)
    return cfg
