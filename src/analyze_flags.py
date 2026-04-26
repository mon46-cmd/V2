"""Flag analysis and visualization on real market data.

Usage::
    python src/analyze_flags.py
    python src/analyze_flags.py --symbols ETHUSDT SOLUSDT --bars 400
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from core.config import load_config
from downloader.http import HttpClient
from downloader.rest import RestClient
from features.pipeline import compute_snapshot, FeatureConfig

log = logging.getLogger("analyze_flags")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")

# ── Palette ────────────────────────────────────────────────────────────────
C_GREEN  = "#26a69a"; C_RED   = "#ef5350"; C_GOLD  = "#ffd700"
C_BLUE   = "#42a5f5"; C_PURP  = "#ab47bc"; C_ORAN  = "#ffa726"
C_BG     = "#0d1117"; C_PANEL = "#161b22"; C_GRID  = "#21262d"
C_TEXT   = "#c9d1d9"; C_ACCT  = "#58a6ff"

FLAG_META: dict[str, dict] = {
    "flag_volume_climax":   {"label":"Volume Climax",         "ind":"volume",        "ind_lbl":"Volume vs 2.5x Mean",      "col":C_ORAN, "desc":"Volume > 2.5× 20-bar mean"},
    "flag_sweep_up":        {"label":"Sweep Up (Bull Trap)",  "ind":"high",          "ind_lbl":"Price High",               "col":C_RED,  "desc":"High pierced N-bar high, bearish close"},
    "flag_sweep_dn":        {"label":"Sweep Down (Bear Trap)","ind":"low",           "ind_lbl":"Price Low",                "col":C_GREEN,"desc":"Low pierced N-bar low, bullish close"},
    "flag_macd_cross_up":   {"label":"MACD Cross Up",         "ind":"macd_hist",     "ind_lbl":"MACD Histogram",           "col":C_GREEN,"desc":"MACD histogram ≤0 → >0"},
    "flag_macd_cross_dn":   {"label":"MACD Cross Down",       "ind":"macd_hist",     "ind_lbl":"MACD Histogram",           "col":C_RED,  "desc":"MACD histogram ≥0 → <0"},
    "flag_adx_strong_trend":{"label":"ADX Strong Trend",      "ind":"adx_14",        "ind_lbl":"ADX-14  (+DI green / -DI red)","col":C_GOLD,"desc":"ADX ≥ 25"},
    "flag_golden_cross":    {"label":"Golden Cross EMA50/200","ind":"ema_spread_pct", "ind_lbl":"EMA50-EMA200 spread %",   "col":C_GOLD, "desc":"EMA-50 crossed above EMA-200"},
    "flag_death_cross":     {"label":"Death Cross EMA50/200", "ind":"ema_spread_pct","ind_lbl":"EMA50-EMA200 spread %",   "col":C_RED,  "desc":"EMA-50 crossed below EMA-200"},
    "flag_rsi_overbought":  {"label":"RSI Overbought",        "ind":"rsi_14",        "ind_lbl":"RSI-14",                  "col":C_RED,  "desc":"RSI ≥ 70"},
    "flag_rsi_oversold":    {"label":"RSI Oversold",          "ind":"rsi_14",        "ind_lbl":"RSI-14",                  "col":C_GREEN,"desc":"RSI ≤ 30"},
    "flag_regime_flip":     {"label":"Supertrend Regime Flip","ind":"supertrend_dir","ind_lbl":"Supertrend Direction",    "col":C_PURP, "desc":"Supertrend ±1 sign flip"},
    "flag_squeeze_release": {"label":"Bollinger Squeeze Release","ind":"bb_width_20","ind_lbl":"Bollinger Bandwidth",     "col":C_BLUE, "desc":"BB width compressed → burst above median"},
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _style(ax, title=""):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(C_GRID)
    ax.yaxis.label.set_color(C_TEXT)
    ax.xaxis.label.set_color(C_TEXT)
    if title:
        ax.set_title(title, color=C_TEXT, fontsize=9, pad=4)
    ax.grid(color=C_GRID, lw=0.5, ls="--", alpha=0.4)


def _candles(ax, ohlcv_tail: pd.DataFrame, dates=None):
    """Simple OHLC bars coloured green/red."""
    d = ohlcv_tail.reset_index(drop=True)
    x = range(len(d))
    for i, row in d.iterrows():
        c = C_GREEN if row["close"] >= row["open"] else C_RED
        ax.plot([i, i], [row["low"], row["high"]], color=c, lw=0.6, alpha=0.6)
        ax.plot([i, i], [row["open"], row["close"]], color=c, lw=2.2)


def _flag_vlines(ax, fire_idx: list[int], n_total: int, color: str):
    """Draw subtle vertical lines at flag bars. Skip if >20% fire rate (too dense)."""
    rate = len(fire_idx) / max(n_total, 1)
    if rate > 0.20:
        return          # Too dense — markers on price chart are enough
    for i in fire_idx:
        ax.axvline(i, color=color, lw=0.5, alpha=0.35)


def _adaptive_bins(arr: np.ndarray, n=40) -> np.ndarray:
    """Histogram bins scaled to actual data spread (ignores fat-tail outliers)."""
    if len(arr) == 0:
        return np.linspace(-2, 2, n)
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99)
    span = max(abs(p1), abs(p99), 0.05) * 1.5
    return np.linspace(-span, span, n)


def _add_dates(ax, dates: pd.Index, n_ticks=6):
    """Label x-axis with human-readable timestamps sampled at n_ticks positions."""
    if dates is None or len(dates) == 0:
        return
    step = max(len(dates) // n_ticks, 1)
    ticks = list(range(0, len(dates), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [pd.Timestamp(dates[i]).strftime("%m/%d %H:%M") for i in ticks],
        fontsize=7, color=C_TEXT, rotation=20, ha="right"
    )


# ── Per-flag plot ────────────────────────────────────────────────────────────

def plot_flag(snap: pd.DataFrame, ohlcv: pd.DataFrame,
              flag: str, meta: dict, symbol: str, out_dir: Path):

    flag_mask  = snap[flag] >= 1.0
    n_fires    = int(flag_mask.sum())
    n_bars     = len(snap)

    if n_fires == 0:
        log.info("  %s — %s: 0 fires, skipping", symbol, flag)
        return

    log.info("  %s — %s: %d fires / %d bars (%.1f%%)",
             symbol, flag, n_fires, n_bars, n_fires / n_bars * 100)

    # ── Align tails ──────────────────────────────────────────────────────
    N = min(300, len(snap), len(ohlcv))
    ohlcv_tail = ohlcv.tail(N).reset_index(drop=True)
    snap_tail  = snap.tail(N).reset_index(drop=True)
    dates      = ohlcv.tail(N).index  # keep DatetimeIndex for labels

    # Precompute EMA spread % (avoids raw-price-diff scaling issues in panel 2)
    if "ema_50" in snap.columns and "ema_200" in snap.columns:
        e200 = snap["ema_200"].replace(0, np.nan)
        snap = snap.copy()
        snap["ema_spread_pct"] = (snap["ema_50"] - snap["ema_200"]) / e200 * 100
        snap_tail = snap.tail(N).reset_index(drop=True)

    fire_idx = [i for i in range(len(snap_tail)) if snap_tail[flag].iloc[i] >= 1.0]
    fire_pct = n_fires / n_bars * 100

    # ── Layout ───────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 11), facecolor=C_BG)
    fig.suptitle(
        f"{symbol}  ·  {meta['label']}\n"
        f"{meta['desc']}  |  {n_fires} fires / {n_bars} bars ({fire_pct:.1f}%)",
        color=C_TEXT, fontsize=12, y=0.99,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           height_ratios=[2.8, 1.6, 2.0],
                           hspace=0.55, wspace=0.28)

    ax_price = fig.add_subplot(gs[0, :])
    ax_ind   = fig.add_subplot(gs[1, :])
    ax_r1h   = fig.add_subplot(gs[2, 0])
    ax_r4h   = fig.add_subplot(gs[2, 1])

    # ─── Panel 1: Price ──────────────────────────────────────────────────
    _candles(ax_price, ohlcv_tail)
    x = range(len(snap_tail))

    # EMA overlays (only plot where non-NaN)
    for col, color, lw, lbl in [
        ("ema_21",  C_ACCT,  1.0, "EMA-21"),
        ("ema_50",  C_GOLD,  0.9, "EMA-50"),
        ("ema_200", C_PURP,  0.7, "EMA-200"),
    ]:
        if col in snap_tail.columns:
            s = snap_tail[col]
            mask = s.notna()
            if mask.any():
                xi = [i for i in x if mask.iloc[i]]
                yi = s[mask].values
                ax_price.plot(xi, yi, color=color, lw=lw, label=lbl, alpha=0.85)

    # Supertrend — draw as coloured line segments (green=bull, red=bear)
    if "supertrend" in snap_tail.columns and "supertrend_dir" in snap_tail.columns:
        st_vals = snap_tail["supertrend"].values
        st_dir  = snap_tail["supertrend_dir"].values
        seg_x, seg_y, seg_c = [], [], None
        def _flush(ax, sx, sy, sc):
            if len(sx) > 1:
                ax.plot(sx, sy, color=sc, lw=1.0, alpha=0.6, zorder=2)
        for i in range(len(snap_tail)):
            c = C_GREEN if st_dir[i] == 1 else C_RED
            if not np.isfinite(st_vals[i]):
                _flush(ax_price, seg_x, seg_y, seg_c)
                seg_x, seg_y, seg_c = [], [], c
                continue
            if seg_c != c and seg_x:
                seg_x.append(i); seg_y.append(st_vals[i])
                _flush(ax_price, seg_x, seg_y, seg_c)
                seg_x, seg_y = [i], [st_vals[i]]
            else:
                seg_x.append(i); seg_y.append(st_vals[i])
            seg_c = c
        _flush(ax_price, seg_x, seg_y, seg_c)

    # Flag markers — placed 0.5% below the bar's low, so they're always visible
    lows = ohlcv_tail["low"].values if "low" in ohlcv_tail.columns else None
    for i in fire_idx:
        if lows is not None and i < len(lows):
            yp = lows[i] * 0.9975
            ax_price.annotate("▲", xy=(i, yp), color=meta["col"],
                              fontsize=8, ha="center", va="top",
                              fontweight="bold")

    _style(ax_price, "Price (15m) + Flag Markers ▲ + EMAs")
    ax_price.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT, loc="upper left",
                    framealpha=0.7)
    _add_dates(ax_price, dates)

    # ─── Panel 2: Indicator ──────────────────────────────────────────────
    ind = meta["ind"]

    # Volume: must come from ohlcv_tail, not snapshot
    if ind == "volume":
        if "volume" in ohlcv_tail.columns:
            vol = ohlcv_tail["volume"].values.astype(float)
            roll_mean = pd.Series(vol).rolling(20, min_periods=1).mean().values
            ax_ind.bar(x, vol, color=C_BLUE, alpha=0.55, width=0.8, label="Volume")
            ax_ind.plot(x, roll_mean * 2.5, color=C_ORAN, lw=1.0,
                        ls="--", label="2.5× mean")
            _flag_vlines(ax_ind, fire_idx, len(snap_tail), meta["col"])
            ax_ind.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
        else:
            ax_ind.text(0.5, 0.5, "No volume data", ha="center", va="center",
                        color=C_TEXT, transform=ax_ind.transAxes)

    # High/Low sweeps: show price range bands
    elif ind in ("high", "low"):
        if all(c in ohlcv_tail.columns for c in ("high", "low", "close")):
            ax_ind.plot(x, ohlcv_tail["high"].values, color=C_GREEN, lw=0.8, label="High")
            ax_ind.plot(x, ohlcv_tail["low"].values,  color=C_RED,   lw=0.8, label="Low")
            ax_ind.plot(x, ohlcv_tail["close"].values, color=C_TEXT, lw=0.6, alpha=0.5)
            # N-bar prior high/low
            cfg = FeatureConfig()
            ph = ohlcv_tail["high"].shift(1).rolling(cfg.sweep_lookback, min_periods=cfg.sweep_lookback).max()
            pl = ohlcv_tail["low"].shift(1).rolling(cfg.sweep_lookback, min_periods=cfg.sweep_lookback).min()
            ax_ind.plot(x, ph.values, color=C_GOLD, lw=0.8, ls="--", label=f"{cfg.sweep_lookback}-bar prior high")
            ax_ind.plot(x, pl.values, color=C_PURP, lw=0.8, ls="--", label=f"{cfg.sweep_lookback}-bar prior low")
            _flag_vlines(ax_ind, fire_idx, len(snap_tail), meta["col"])
            ax_ind.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)

    elif ind == "macd_hist":
        if "macd_hist" in snap_tail.columns:
            vals = snap_tail["macd_hist"].values
            colors = [C_GREEN if v >= 0 else C_RED for v in vals]
            ax_ind.bar(x, vals, color=colors, alpha=0.8, width=0.8)
            ax_ind.axhline(0, color=C_TEXT, lw=0.5)
            _flag_vlines(ax_ind, fire_idx, len(snap_tail), meta["col"])

    elif ind == "rsi_14":
        if "rsi_14" in snap_tail.columns:
            ax_ind.plot(x, snap_tail["rsi_14"].values, color=C_ACCT, lw=1.2)
            ax_ind.axhline(70, color=C_RED,   lw=0.8, ls="--", alpha=0.7, label="OB 70")
            ax_ind.axhline(30, color=C_GREEN, lw=0.8, ls="--", alpha=0.7, label="OS 30")
            ax_ind.axhline(50, color=C_TEXT,  lw=0.4, ls=":",  alpha=0.4)
            ax_ind.set_ylim(0, 100)
            _flag_vlines(ax_ind, fire_idx, len(snap_tail), meta["col"])
            ax_ind.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)

    elif ind == "adx_14":
        if "adx_14" in snap_tail.columns:
            ax_ind.plot(x, snap_tail["adx_14"].values, color=C_GOLD, lw=1.3, label="ADX")
            if "plus_di_14" in snap_tail.columns:
                ax_ind.plot(x, snap_tail["plus_di_14"].values,  color=C_GREEN, lw=0.8, ls="--", label="+DI")
            if "minus_di_14" in snap_tail.columns:
                ax_ind.plot(x, snap_tail["minus_di_14"].values, color=C_RED,   lw=0.8, ls="--", label="-DI")
            ax_ind.axhline(25, color=C_ORAN, lw=0.8, ls="--", alpha=0.7, label="ADX 25")
            # Only draw vlines if sparse enough
            _flag_vlines(ax_ind, fire_idx, len(snap_tail), meta["col"])
            ax_ind.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)

    elif ind == "ema_spread_pct":
        if "ema_spread_pct" in snap_tail.columns:
            vals = snap_tail["ema_spread_pct"].fillna(0).values
            colors = [C_GREEN if v >= 0 else C_RED for v in vals]
            # Only plot where EMA200 is valid (skip warm-up NaN zone)
            valid = snap_tail["ema_200"].notna()
            xi_v  = [i for i in x if valid.iloc[i]]
            vi_v  = [vals[i] for i in xi_v]
            cl_v  = [colors[i] for i in xi_v]
            ax_ind.bar(xi_v, vi_v, color=cl_v, alpha=0.7, width=0.8)
            ax_ind.axhline(0, color=C_TEXT, lw=0.6)
            ax_ind.set_ylabel("EMA spread %", color=C_TEXT, fontsize=8)
            _flag_vlines(ax_ind, [i for i in fire_idx if valid.iloc[i]], len(xi_v), meta["col"])

    elif ind == "supertrend_dir":
        if "supertrend_dir" in snap_tail.columns:
            ax_ind.step(x, snap_tail["supertrend_dir"].fillna(0).values,
                        color=meta["col"], lw=1.5)
            ax_ind.axhline(0, color=C_TEXT, lw=0.5, ls=":")
            ax_ind.set_ylim(-1.5, 1.5)
            ax_ind.set_yticks([-1, 0, 1])
            ax_ind.set_yticklabels(["Bear", "0", "Bull"], color=C_TEXT, fontsize=8)
            _flag_vlines(ax_ind, fire_idx, len(snap_tail), meta["col"])

    elif ind == "bb_width_20":
        if "bb_width_20" in snap_tail.columns:
            bw = snap_tail["bb_width_20"].values
            ax_ind.plot(x, bw, color=C_BLUE, lw=1.2, label="BB Width")
            q10 = pd.Series(bw).rolling(100, min_periods=20).quantile(0.10).values
            q50 = pd.Series(bw).rolling(100, min_periods=20).quantile(0.50).values
            ax_ind.plot(x, q10, color=C_RED,   lw=0.7, ls="--", label="P10 (squeeze threshold)")
            ax_ind.plot(x, q50, color=C_GREEN, lw=0.7, ls="--", label="P50 (release threshold)")
            _flag_vlines(ax_ind, fire_idx, len(snap_tail), meta["col"])
            ax_ind.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)

    _style(ax_ind, meta["ind_lbl"])
    _add_dates(ax_ind, dates)

    # ─── Panels 3 & 4: Forward return distributions ───────────────────────
    if "close" in snap.columns:
        close = snap["close"]
        for ax, h, lbl in [(ax_r1h, 4, "1h (4 bars)"), (ax_r4h, 16, "4h (16 bars)")]:
            fwd = np.log(close.shift(-h) / close) * 100
            flag_ret  = fwd[flag_mask].dropna().values
            other_ret = fwd[~flag_mask].dropna().values

            all_ret = np.concatenate([flag_ret, other_ret])
            bins = _adaptive_bins(all_ret)

            _style(ax, f"Return Dist — {lbl}")
            ax.hist(other_ret, bins=bins, color=C_BLUE,     alpha=0.45, density=True,
                    label=f"No flag (n={len(other_ret)})")
            ax.hist(flag_ret,  bins=bins, color=meta["col"], alpha=0.75, density=True,
                    label=f"Flag fires (n={len(flag_ret)})")
            ax.axvline(0, color=C_TEXT, lw=0.8, ls="--")
            if len(flag_ret) > 0:
                mu = flag_ret.mean()
                pos_pct = (flag_ret > 0).mean() * 100
                ax.axvline(mu, color=C_GOLD, lw=1.5,
                           label=f"μ={mu:+.3f}%  pos={pos_pct:.0f}%")
            ax.set_xlabel("Forward log-return %", color=C_TEXT, fontsize=8)
            ax.legend(fontsize=7, facecolor=C_PANEL, labelcolor=C_TEXT)
    else:
        for ax in (ax_r1h, ax_r4h):
            _style(ax)
            ax.text(0.5, 0.5, "No close data", ha="center", va="center",
                    color=C_TEXT, transform=ax.transAxes)

    out_path = out_dir / f"{symbol}_{flag}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    log.info("  → %s", out_path.name)


# ── Stats table ──────────────────────────────────────────────────────────────

def flag_stats_table(snap: pd.DataFrame) -> pd.DataFrame:
    from features.layer7_flags import FLAG_COLUMNS
    rows = []
    close = snap.get("close")
    for flag in FLAG_COLUMNS:
        if flag not in snap.columns:
            continue
        mask   = snap[flag] >= 1.0
        n_fire = int(mask.sum())
        row    = {"flag": flag, "fires": n_fire,
                  "rate%": round(n_fire / len(snap) * 100, 2)}
        if close is not None and n_fire > 0:
            for h, k1, k2 in [(4, "fwd1h_mean%", None), (16, "fwd4h_mean%", "fwd4h_pos%")]:
                fwd = np.log(close.shift(-h) / close)[mask].dropna() * 100
                row[k1 if k2 is None else "fwd1h_mean%" if h == 4 else "fwd4h_mean%"] = round(fwd.mean(), 3)
                if k2:
                    row[k2] = round((fwd > 0).mean() * 100, 1)
        else:
            row.update({"fwd1h_mean%": np.nan, "fwd4h_mean%": np.nan, "fwd4h_pos%": np.nan})
        rows.append(row)
    return pd.DataFrame(rows).set_index("flag")


# ── Main ─────────────────────────────────────────────────────────────────────

async def main(symbols: list[str], n_bars: int, out_dir: Path):
    cfg  = load_config()
    feat = FeatureConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    async with HttpClient(cfg) as http:
        rest  = RestClient(http, cfg)
        end   = datetime.now(tz=timezone.utc)
        start = end - timedelta(minutes=(n_bars + 20) * 15)

        for sym in symbols:
            log.info("Fetching %s (%d bars)...", sym, n_bars)
            ohlcv = await rest.klines(sym, "15", start, end)
            if ohlcv.empty or len(ohlcv) < 60:
                log.warning("  %s: insufficient data", sym)
                continue

            ohlcv_idx = ohlcv.set_index("timestamp")
            snap = compute_snapshot(ohlcv_idx, feat)

            # Attach close for forward-return computation
            snap = snap.copy()
            snap["close"] = ohlcv_idx["close"]

            log.info("  %s: %d bars, %d flag cols",
                     sym, len(snap), sum(c.startswith("flag_") for c in snap.columns))

            stats = flag_stats_table(snap)
            (out_dir / f"{sym}_flag_stats.txt").write_text(stats.to_string())
            print(f"\n{'='*60}\n  {sym} ({len(snap)} bars)\n{'='*60}")
            print(stats.to_string())

            for flag, meta in FLAG_META.items():
                if flag not in snap.columns:
                    continue
                plot_flag(snap, ohlcv_idx, flag, meta, sym, out_dir)

    log.info("Done → %s", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    ap.add_argument("--bars",    type=int,  default=300)
    ap.add_argument("--out",     type=str,  default="data/flag_analysis")
    args = ap.parse_args()
    asyncio.run(main(args.symbols, args.bars, Path(args.out)))
