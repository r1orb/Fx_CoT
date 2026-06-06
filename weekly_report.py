"""
COT Weekly Report Generator
Detects active MM extreme signals, generates an HTML report with embedded
charts, and sends it by email.

Usage:
    python weekly_report.py              # full pipeline + send email
    python weekly_report.py --no-email  # generate report, skip email
    python weekly_report.py --refresh   # force re-download price cache

Setup:
    1. pip install python-dotenv
    2. Copy .env.example to .env and fill in credentials
    3. Run weekly after updating COT files in COT/
"""

import sys
import os
import base64
import smtplib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats as scipy_stats

# Silence yfinance noise in logs
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    log.warning("python-dotenv not installed. Reading env vars from system environment only.")

# Import core analysis pipeline
sys.path.insert(0, str(Path(__file__).parent))
from cot_analysis import (
    load_cot_files,
    compute_net_mm,
    fetch_prices,
    align_and_forward_returns,
    analyze_extremes,
    Z_EXTREME,
    FWD_HORIZONS,
    INSTRUMENT_MAP,
    OUTPUT_DIR,
    CACHE_FILE,
)

REPORTS_DIR = OUTPUT_DIR / "reports"
SIGNAL_CHARTS_DIR = OUTPUT_DIR / "signal_charts"

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COT Weekly Report")
    p.add_argument("--no-email", action="store_true",
                   help="Generate report but skip sending email")
    p.add_argument("--refresh", action="store_true",
                   help="Force re-download of price data (ignore cache)")
    return p.parse_args()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(refresh: bool = False):
    """Run full COT analysis and return (cot_df, merged_df, prices, stats_df)."""
    if refresh and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        log.info("Price cache deleted — will re-download.")

    log.info("Loading COT files...")
    cot_df = load_cot_files()
    log.info(f"  {len(cot_df):,} rows | date range: "
             f"{cot_df['date'].min().date()} to {cot_df['date'].max().date()}")

    log.info("Computing NET MM z-scores...")
    cot_df = compute_net_mm(cot_df)

    log.info("Fetching price data...")
    tickers = list(dict.fromkeys(t for _, t, _ in INSTRUMENT_MAP))
    prices = fetch_prices(tickers)

    log.info("Aligning COT with prices...")
    merged_df = align_and_forward_returns(cot_df, prices)

    log.info("Computing edge statistics...")
    stats_df = analyze_extremes(merged_df)
    try:
        stats_df.to_csv(OUTPUT_DIR / "results_table.csv", index=False)
    except PermissionError:
        log.warning("results_table.csv is locked (open in another app?). Skipping save.")

    return cot_df, merged_df, prices, stats_df


# ── Signal Detection ──────────────────────────────────────────────────────────

def count_weeks_at_extreme(z_series: pd.Series, direction: int,
                            threshold: float = Z_EXTREME) -> int:
    """Count consecutive weeks the instrument has been at the extreme (newest-first)."""
    count = 0
    for z in z_series.sort_index(ascending=False):
        if pd.isna(z):
            break
        if direction == 1 and z > threshold:
            count += 1
        elif direction == -1 and z < -threshold:
            count += 1
        else:
            break
    return count


def detect_signals(cot_df: pd.DataFrame, stats_df: pd.DataFrame,
                   threshold: float = Z_EXTREME) -> list[dict]:
    """
    For each instrument check the latest two weeks of z-scores.
    Classify signals as NEW or ONGOING and cross-reference with edge stats.
    """
    signals = []

    for instrument, grp in cot_df.groupby("instrument"):
        grp = grp.sort_values("date")
        recent = grp.dropna(subset=["z_score"]).tail(2)
        if recent.empty:
            continue

        current = recent.iloc[-1]
        z_now   = current["z_score"]
        z_prev  = recent.iloc[-2]["z_score"] if len(recent) == 2 else None

        if z_now > threshold:
            extreme_type = "LONG extreme"
            direction    = 1
        elif z_now < -threshold:
            extreme_type = "SHORT extreme"
            direction    = -1
        else:
            continue  # not currently at extreme

        # NEW = just crossed threshold; ONGOING = was already there
        if z_prev is None:
            status = "NEW"
        elif direction == 1 and z_prev > threshold:
            status = "ONGOING"
        elif direction == -1 and z_prev < -threshold:
            status = "ONGOING"
        else:
            status = "NEW"

        streak = count_weeks_at_extreme(grp.set_index("date")["z_score"],
                                        direction, threshold)

        # Pull historical edge for 4W horizon
        edge_row = stats_df[
            (stats_df["instrument"] == instrument) &
            (stats_df["extreme_type"] == extreme_type) &
            (stats_df["horizon_weeks"] == 4)
        ]
        edge = edge_row.iloc[0].to_dict() if not edge_row.empty else {}

        signals.append({
            "instrument":    instrument,
            "ticker":        current["ticker"],
            "invert":        current["invert"],
            "report_date":   current["date"].date(),
            "z_score":       round(z_now, 2),
            "pct_rank":      round(current.get("pct_rank_1y", np.nan) * 100, 1)
                             if pd.notna(current.get("pct_rank_1y")) else "—",
            "extreme_type":  extreme_type,
            "direction":     direction,
            "status":        status,
            "streak_weeks":  streak,
            "edge":          edge,
        })

    signals.sort(key=lambda s: (s["status"] != "NEW", abs(s["z_score"])), reverse=True)
    return signals


def detect_faded_signals(cot_df: pd.DataFrame, threshold: float = Z_EXTREME) -> list[dict]:
    """Signals that were extreme last week but are no longer this week."""
    faded = []
    for instrument, grp in cot_df.groupby("instrument"):
        grp = grp.sort_values("date")
        recent = grp.dropna(subset=["z_score"]).tail(2)
        if len(recent) < 2:
            continue
        z_now  = recent.iloc[-1]["z_score"]
        z_prev = recent.iloc[-2]["z_score"]
        was_extreme = abs(z_prev) > threshold
        is_extreme  = abs(z_now)  > threshold
        if was_extreme and not is_extreme:
            direction = "long" if z_prev > 0 else "short"
            faded.append({
                "instrument": instrument,
                "ticker":     recent.iloc[-1]["ticker"],
                "direction":  direction,
                "z_prev":     round(z_prev, 2),
                "z_now":      round(z_now, 2),
            })
    return faded


# ── Signal Charts ─────────────────────────────────────────────────────────────

def _img_to_base64(fig) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def make_signal_chart(instrument: str, merged_df: pd.DataFrame,
                      prices: pd.DataFrame) -> str:
    """
    2-year focused chart for an active signal instrument.
    Returns a base64 PNG string.
    """
    grp = merged_df[merged_df["instrument"] == instrument].sort_values("date")
    ticker = grp["ticker"].iloc[0]
    invert = grp["invert"].iloc[0]

    # 2-year window
    cutoff = grp["date"].max() - pd.DateOffset(years=2)
    grp = grp[grp["date"] >= cutoff]

    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()

    if ticker not in prices.columns:
        return ""

    px = prices[ticker].dropna()
    px_2y = px[px.index >= cutoff]

    fig = plt.figure(figsize=(10, 6.5))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[1.6, 1], hspace=0.12)

    # -- Price panel --
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(px_2y.index, px_2y.values, lw=1.3, color="#2c7bb6")
    ax1.set_yscale("log")
    ax1.set_title(
        f"{instrument}  ({ticker})"
        + ("  [returns sign-flipped for COT direction]" if invert else ""),
        fontsize=11, fontweight="bold", pad=6
    )
    ax1.set_ylabel("Price", fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelbottom=False)

    # Shade extreme-period columns on price panel
    ex_long  = grp[grp["z_score"] > Z_EXTREME]["entry_date"]
    ex_short = grp[grp["z_score"] < -Z_EXTREME]["entry_date"]
    for d in ex_long:
        ax1.axvline(d, color="#d62728", alpha=0.07, lw=5)
    for d in ex_short:
        ax1.axvline(d, color="#2c7bb6", alpha=0.07, lw=5)

    # -- Z-score panel --
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(grp["date"], grp["z_score"], lw=1.2, color="#2ca02c", label="NET MM z-score")
    ax2.axhline(0,          color="#888", lw=0.8, ls="--")
    ax2.axhline( Z_EXTREME, color="#d62728", lw=1.0, ls="--", label=f"+{Z_EXTREME}")
    ax2.axhline(-Z_EXTREME, color="#d62728", lw=1.0, ls="--", label=f"-{Z_EXTREME}")
    ax2.fill_between(grp["date"],  Z_EXTREME, grp["z_score"].clip(lower= Z_EXTREME),
                     alpha=0.20, color="#d62728")
    ax2.fill_between(grp["date"], -Z_EXTREME, grp["z_score"].clip(upper=-Z_EXTREME),
                     alpha=0.20, color="#2c7bb6")

    # Annotate latest z-score
    last_z    = grp["z_score"].iloc[-1]
    last_date = grp["date"].iloc[-1]
    ax2.annotate(
        f" z = {last_z:.2f}",
        xy=(last_date, last_z),
        xytext=(10, 0), textcoords="offset points",
        fontsize=10, fontweight="bold",
        color="#d62728" if abs(last_z) > Z_EXTREME else "#333",
        va="center",
    )
    ax2.set_ylabel("Z-Score (3yr)", fontsize=9)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.2)

    fig.autofmt_xdate(rotation=20, ha="right")
    result = _img_to_base64(fig)
    plt.close(fig)
    return result


def make_overview_scatter(instrument: str, merged_df: pd.DataFrame) -> str:
    """Z-score vs 4W forward return scatter for historical edge visualisation."""
    grp = merged_df[merged_df["instrument"] == instrument].dropna(subset=["z_score", "fwd_4w"])

    fig, ax = plt.subplots(figsize=(7, 4))
    is_ext = grp["z_score"].abs() > Z_EXTREME
    ax.scatter(grp.loc[~is_ext, "z_score"], grp.loc[~is_ext, "fwd_4w"] * 100,
               color="#aec7e8", alpha=0.4, s=14, label="Normal")
    ax.scatter(grp.loc[is_ext, "z_score"], grp.loc[is_ext, "fwd_4w"] * 100,
               color="#d62728", alpha=0.7, s=20, zorder=3, label="Extreme")
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlabel("NET MM Z-Score", fontsize=9)
    ax.set_ylabel("4W Forward Return (%)", fontsize=9)
    ax.set_title(f"{instrument} — Z-Score vs 4W Return (full history)", fontsize=10)
    ax.grid(True, alpha=0.2)
    if len(grp) > 10:
        sl, ic, rv, pv, _ = scipy_stats.linregress(grp["z_score"], grp["fwd_4w"] * 100)
        xs = np.linspace(grp["z_score"].min(), grp["z_score"].max(), 50)
        ax.plot(xs, sl * xs + ic, color="#ff7f0e", lw=1.5,
                label=f"r={rv:.2f}  p={pv:.3f}")
    ax.legend(fontsize=8)
    result = _img_to_base64(fig)
    plt.close(fig)
    return result


# ── HTML Report ───────────────────────────────────────────────────────────────

_CSS = """
body{font-family:Arial,Helvetica,sans-serif;max-width:960px;margin:0 auto;
     padding:20px;color:#222;line-height:1.45}
h1{color:#1a1a2e;border-bottom:3px solid #e63946;padding-bottom:8px;margin-bottom:4px}
h2{color:#457b9d;margin-top:36px;margin-bottom:8px}
h3{color:#1a1a2e;margin-bottom:6px}
.subtitle{color:#666;margin-top:0;font-size:14px}
.warn-banner{background:#fff3cd;border:1px solid #ffc107;border-radius:6px;
             padding:10px 16px;margin:16px 0;color:#856404;font-size:13px}
.signal-card{border:2px solid;border-radius:8px;padding:18px 20px;margin:18px 0}
.card-new    {border-color:#e63946;background:#fff5f5}
.card-ongoing{border-color:#f4a261;background:#fffaf0}
.badge{display:inline-block;padding:3px 9px;border-radius:4px;
       font-size:11px;font-weight:bold;letter-spacing:.5px;margin-right:8px}
.badge-new    {background:#e63946;color:#fff}
.badge-ongoing{background:#f4a261;color:#fff}
.stat-row{display:flex;gap:14px;flex-wrap:wrap;margin:12px 0}
.stat-box{background:#f0f4f8;border-radius:6px;padding:10px 16px;text-align:center;
          min-width:90px}
.stat-val{font-size:21px;font-weight:bold;color:#1a1a2e}
.stat-lbl{font-size:10px;color:#666;margin-top:2px}
.no-signal{background:#f8f9fa;border:1px solid #dee2e6;border-radius:6px;
           padding:14px 18px;color:#6c757d;margin:10px 0}
table{width:100%;border-collapse:collapse;font-size:13px;margin:12px 0}
th{background:#1a1a2e;color:#fff;padding:8px 11px;text-align:left;white-space:nowrap}
td{padding:7px 11px;border-bottom:1px solid #eee;white-space:nowrap}
tr:nth-child(even) td{background:#fafafa}
tr:hover td{background:#e8f4fd}
.sig{color:#e63946;font-weight:bold}
.pos{color:#2d6a4f}
.neg{color:#c0392b}
.center{text-align:center}
img{max-width:100%;border-radius:6px;margin:10px 0;box-shadow:0 2px 6px rgba(0,0,0,.12)}
.faded{background:#f0f8e8;border:1px solid #a8d5a2;border-radius:6px;
       padding:10px 16px;margin:10px 0;font-size:13px;color:#2d6a4f}
.footer{margin-top:48px;color:#999;font-size:12px;border-top:1px solid #eee;padding-top:12px}
"""

def _edge_html(edge: dict) -> str:
    if not edge:
        return "<em style='color:#999'>No historical stats</em>"
    n    = int(edge.get("n", 0))
    mean = edge.get("mean_return_pct", float("nan"))
    hit  = edge.get("hit_rate_pct", float("nan"))
    t    = edge.get("t_stat", float("nan"))
    p    = edge.get("p_value", float("nan"))
    sig  = "**" if (not np.isnan(p) and p < 0.05) else ("*" if (not np.isnan(p) and p < 0.10) else "")
    mean_col = "pos" if (not np.isnan(mean) and mean > 0) else "neg"
    return (
        f"<b>N={n}</b> | "
        f"Mean 4W: <span class='{mean_col}'><b>{mean:+.2f}%</b></span> | "
        f"Hit rate: <b>{hit:.1f}%</b> | "
        f"t={t:.2f}  p={p:.4f} {sig}"
    )


def generate_html_report(signals: list[dict], faded: list[dict],
                          stats_df: pd.DataFrame, chart_map: dict[str, dict],
                          warning: str | None, run_date: datetime) -> str:
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>",
             f"<title>COT Report {run_date.strftime('%Y-%m-%d')}</title>",
             f"<style>{_CSS}</style></head><body>"]
    parts.append(f"<h1>COT NET MM Weekly Report</h1>")
    parts.append(f"<p class='subtitle'>Generated: {run_date.strftime('%A, %d %B %Y  %H:%M')} "
                 f"| Data as of: "
                 f"{signals[0]['report_date'] if signals else '—'}</p>")

    if warning:
        parts.append(f"<div class='warn-banner'>&#9888; {warning}</div>")

    # ── Active signals section ──
    parts.append("<h2>Active Signals</h2>")
    if not signals:
        parts.append("<div class='no-signal'>No instruments currently at z-score extremes "
                     f"(threshold: &plusmn;{Z_EXTREME})</div>")
    else:
        for sig in signals:
            card_cls   = "card-new" if sig["status"] == "NEW" else "card-ongoing"
            badge_cls  = "badge-new" if sig["status"] == "NEW" else "badge-ongoing"
            direction  = "LONG (bullish)" if sig["direction"] == 1 else "SHORT (bearish)"
            parts.append(f"<div class='signal-card {card_cls}'>")
            parts.append(f"<h3><span class='badge {badge_cls}'>{sig['status']}</span>"
                         f"{sig['instrument']} &nbsp;&mdash;&nbsp; {direction}</h3>")

            parts.append("<div class='stat-row'>")
            parts.append(f"<div class='stat-box'><div class='stat-val'>{sig['z_score']:+.2f}</div>"
                         f"<div class='stat-lbl'>Z-Score</div></div>")
            parts.append(f"<div class='stat-box'><div class='stat-val'>{sig['streak_weeks']}</div>"
                         f"<div class='stat-lbl'>Weeks at Extreme</div></div>")
            pct = sig["pct_rank"]
            pct_str = f"{pct:.0f}th" if pct != "—" else "—"
            parts.append(f"<div class='stat-box'><div class='stat-val'>{pct_str}</div>"
                         f"<div class='stat-lbl'>1Y Pct Rank</div></div>")
            parts.append(f"<div class='stat-box'><div class='stat-val'>{sig['ticker']}</div>"
                         f"<div class='stat-lbl'>Price Feed</div></div>")
            parts.append("</div>")

            parts.append(f"<p><b>Historical edge (4W horizon):</b> {_edge_html(sig['edge'])}</p>")
            if sig.get("invert"):
                parts.append("<p style='font-size:12px;color:#888'>"
                              "Note: COT contract direction is non-USD vs USD; "
                              "return sign is adjusted accordingly.</p>")

            charts = chart_map.get(sig["instrument"], {})
            inst_name = sig["instrument"]
            if charts.get("signal"):
                parts.append(f"<img src='data:image/png;base64,{charts['signal']}' "
                              f"alt='{inst_name} signal chart'>")
            if charts.get("scatter"):
                parts.append(f"<img src='data:image/png;base64,{charts['scatter']}' "
                              f"alt='{inst_name} scatter chart'>")
            parts.append("</div>")

    # ── Faded signals ──
    if faded:
        parts.append("<h2>Signals Exited This Week</h2>")
        for f in faded:
            parts.append(f"<div class='faded'>&#10003; <b>{f['instrument']}</b> — "
                         f"{f['direction'].upper()} extreme exited | "
                         f"z: {f['z_prev']:+.2f} &rarr; {f['z_now']:+.2f}</div>")

    # ── Full statistics table (4W horizon) ──
    parts.append("<h2>Full Edge Statistics (4-Week Horizon)</h2>")
    focus = stats_df[stats_df["horizon_weeks"] == 4].copy()
    focus = focus.sort_values("t_stat", key=abs, ascending=False, na_position="last")
    rows = ["<table><tr>",
            "<th>Instrument</th><th>Extreme Type</th>",
            "<th class='center'>N</th><th class='center'>Mean 4W %</th>",
            "<th class='center'>Hit %</th><th class='center'>t-stat</th>",
            "<th class='center'>p-value</th><th class='center'>Sig</th>",
            "</tr>"]
    for _, r in focus.iterrows():
        if pd.isna(r["t_stat"]):
            continue
        sig_mark = '<span class="sig">**</span>' if r["p_value"] < 0.05 else (
                   '<span class="sig">*</span>'  if r["p_value"] < 0.10 else "")
        mean_cls = "pos" if r["mean_return_pct"] > 0 else "neg"
        rows.append(
            f"<tr><td>{r['instrument']}</td><td>{r['extreme_type']}</td>"
            f"<td class='center'>{int(r['n'])}</td>"
            f"<td class='center {mean_cls}'><b>{r['mean_return_pct']:+.2f}%</b></td>"
            f"<td class='center'>{r['hit_rate_pct']:.1f}%</td>"
            f"<td class='center'>{r['t_stat']:.2f}</td>"
            f"<td class='center'>{r['p_value']:.4f}</td>"
            f"<td class='center'>{sig_mark}</td></tr>"
        )
    rows.append("</table>")
    rows.append("<p style='font-size:12px;color:#666'>** p &lt; 0.05 &nbsp; * p &lt; 0.10 "
                "| Hit rate = % of extreme observations where return matched expected direction</p>")
    parts.append("".join(rows))

    # ── All horizon breakdown for significant instruments ──
    sig_instruments = stats_df[
        (stats_df["p_value"] < 0.05) & (stats_df["hit_rate_pct"] > 55)
    ]["instrument"].unique()
    if len(sig_instruments) > 0:
        parts.append("<h2>Horizon Breakdown — Significant Instruments</h2>")
        full = stats_df[stats_df["instrument"].isin(sig_instruments)].copy()
        full = full.sort_values(["instrument", "extreme_type", "horizon_weeks"])
        rows2 = ["<table><tr>",
                 "<th>Instrument</th><th>Type</th><th class='center'>Horizon</th>",
                 "<th class='center'>N</th><th class='center'>Mean %</th>",
                 "<th class='center'>Hit %</th><th class='center'>t</th>",
                 "<th class='center'>p</th></tr>"]
        for _, r in full.iterrows():
            if pd.isna(r["t_stat"]):
                continue
            mean_cls = "pos" if r["mean_return_pct"] > 0 else "neg"
            rows2.append(
                f"<tr><td>{r['instrument']}</td><td>{r['extreme_type']}</td>"
                f"<td class='center'>{int(r['horizon_weeks'])}W</td>"
                f"<td class='center'>{int(r['n'])}</td>"
                f"<td class='center {mean_cls}'>{r['mean_return_pct']:+.2f}%</td>"
                f"<td class='center'>{r['hit_rate_pct']:.1f}%</td>"
                f"<td class='center'>{r['t_stat']:.2f}</td>"
                f"<td class='center'>{r['p_value']:.4f}</td></tr>"
            )
        rows2.append("</table>")
        parts.append("".join(rows2))

    parts.append(
        f"<div class='footer'>"
        f"Data: CFTC Commitment of Traders (Financial TFF report) | "
        f"Prices: Yahoo Finance (weekly close) | "
        f"Z-score: 3-year rolling (156-week window, min 52 weeks) | "
        f"Extreme threshold: &plusmn;{Z_EXTREME} standard deviations | "
        f"Edge criterion: p &lt; 0.05 AND hit rate &gt; 55% at 4-week horizon"
        f"</div>"
    )
    parts.append("</body></html>")
    return "\n".join(parts)


# ── Email Body (compact, no images) ──────────────────────────────────────────

def generate_email_body(signals: list[dict], faded: list[dict],
                         warning: str | None, run_date: datetime) -> str:
    css_mini = (
        "body{font-family:Arial,sans-serif;color:#222;max-width:700px;margin:0 auto}"
        "h2{color:#457b9d}h3{color:#1a1a2e}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th{background:#1a1a2e;color:#fff;padding:7px 10px;text-align:left}"
        "td{padding:6px 10px;border-bottom:1px solid #eee}"
        ".pos{color:#2d6a4f;font-weight:bold}.neg{color:#c0392b;font-weight:bold}"
        ".badge-new{background:#e63946;color:#fff;padding:2px 7px;border-radius:3px;"
        "font-size:11px;font-weight:bold}"
        ".badge-on{background:#f4a261;color:#fff;padding:2px 7px;border-radius:3px;"
        "font-size:11px;font-weight:bold}"
        ".sig{color:#e63946;font-weight:bold}"
    )
    body = [f"<html><head><style>{css_mini}</style></head><body>"]
    body.append(f"<h2>COT Weekly Report &mdash; {run_date.strftime('%d %B %Y')}</h2>")

    if warning:
        body.append(f"<p style='background:#fff3cd;padding:8px;border-radius:4px;"
                    f"color:#856404'><b>&#9888; Warning:</b> {warning}</p>")

    # Active signals
    body.append("<h3>Active Signals</h3>")
    if not signals:
        body.append(f"<p style='color:#666'>No instruments at z-score extremes "
                    f"(&plusmn;{Z_EXTREME}) this week.</p>")
    else:
        body.append("<table><tr><th>Instrument</th><th>Status</th><th>Direction</th>"
                    "<th>Z-Score</th><th>Weeks</th>"
                    "<th>Mean 4W%</th><th>Hit%</th><th>t</th><th>p</th></tr>")
        for s in signals:
            badge = (f"<span class='badge-new'>NEW</span>"
                     if s["status"] == "NEW" else "<span class='badge-on'>ONGOING</span>")
            direction = "LONG" if s["direction"] == 1 else "SHORT"
            e = s["edge"]
            mean_r = e.get("mean_return_pct", float("nan"))
            hit    = e.get("hit_rate_pct",    float("nan"))
            t_s    = e.get("t_stat",          float("nan"))
            p_v    = e.get("p_value",         float("nan"))
            mean_cls = "pos" if (not np.isnan(mean_r) and mean_r > 0) else "neg"
            sig_m = '<span class="sig">**</span>' if (not np.isnan(p_v) and p_v < 0.05) else (
                    '<span class="sig">*</span>'  if (not np.isnan(p_v) and p_v < 0.10) else "")
            body.append(
                f"<tr><td><b>{s['instrument']}</b></td><td>{badge}</td>"
                f"<td>{direction}</td><td><b>{s['z_score']:+.2f}</b></td>"
                f"<td>{s['streak_weeks']}</td>"
                f"<td class='{mean_cls}'>{mean_r:+.2f}%</td>"
                f"<td>{hit:.1f}%</td><td>{t_s:.2f}</td>"
                f"<td>{p_v:.4f}{sig_m}</td></tr>"
            )
        body.append("</table>")

    # Faded signals
    if faded:
        body.append("<h3>Exited This Week</h3><ul>")
        for f in faded:
            body.append(f"<li><b>{f['instrument']}</b> &mdash; {f['direction'].upper()} extreme "
                        f"exited (z: {f['z_prev']:+.2f} &rarr; {f['z_now']:+.2f})</li>")
        body.append("</ul>")

    body.append("<p style='color:#888;font-size:12px;margin-top:24px'>"
                "Full charts and horizon breakdown in the attached HTML report. "
                "Open it in a browser for the best experience.</p>")
    body.append("<p style='color:#999;font-size:11px'>"
                "Data: CFTC COT | Prices: Yahoo Finance | "
                f"Threshold: &plusmn;{Z_EXTREME} std devs (3yr rolling)</p>")
    body.append("</body></html>")
    return "\n".join(body)


# ── Email Sending ─────────────────────────────────────────────────────────────

def send_email(subject: str, body_html: str, report_html: str,
               csv_path: Path, date_str: str) -> None:
    smtp_host = os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
    smtp_user = os.environ.get("EMAIL_USER", "")
    smtp_pass = os.environ.get("EMAIL_PASSWORD", "")
    recipients_raw = os.environ.get("EMAIL_TO", "")

    if not smtp_user or not smtp_pass or not recipients_raw:
        raise EnvironmentError(
            "EMAIL_USER, EMAIL_PASSWORD, and EMAIL_TO must be set in .env or environment."
        )

    recipients = [r.strip() for r in recipients_raw.split(",") if r.strip()]

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = smtp_user
    msg["To"]      = ", ".join(recipients)

    # Plain text fallback
    msg.set_content("COT Weekly Report — open this email in an HTML-capable client.")

    # HTML body (tables, no images)
    msg.add_alternative(body_html, subtype="html")

    # Attachment 1: full HTML report with embedded charts
    filename_html = f"cot_report_{date_str}.html"
    msg.add_attachment(
        report_html.encode("utf-8"),
        maintype="text",
        subtype="html",
        filename=filename_html,
    )

    # Attachment 2: CSV data
    if csv_path.exists():
        msg.add_attachment(
            csv_path.read_bytes(),
            maintype="text",
            subtype="csv",
            filename=f"cot_results_{date_str}.csv",
        )

    log.info(f"Sending email to: {recipients}")
    import ssl
    is_local = smtp_host in ("127.0.0.1", "localhost", "::1")
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        if is_local:
            # Bridge uses a self-signed cert; disable verification for localhost
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            try:
                server.starttls(context=ctx)
            except smtplib.SMTPException:
                pass  # some Bridge versions don't require STARTTLS
        else:
            server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    log.info("Email sent successfully.")


# ── Freshness Check ───────────────────────────────────────────────────────────

def check_freshness(cot_df: pd.DataFrame, max_days: int = 10) -> str | None:
    latest = cot_df["date"].max()
    age = (datetime.now() - latest).days
    if age > max_days:
        return (f"Latest COT data is {age} days old ({latest.date()}). "
                f"Update COT files in COT/ before relying on this report.")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    args = parse_args()
    run_date = datetime.now()
    date_str = run_date.strftime("%Y-%m-%d")

    log.info("=== COT Weekly Report ===")

    # 1. Run full analysis pipeline
    cot_df, merged_df, prices, stats_df = run_pipeline(refresh=args.refresh)

    # 2. Freshness check
    warning = check_freshness(cot_df)
    if warning:
        log.warning(warning)

    # 3. Detect signals
    log.info("Detecting current signals...")
    signals = detect_signals(cot_df, stats_df)
    faded   = detect_faded_signals(cot_df)

    n_new     = sum(1 for s in signals if s["status"] == "NEW")
    n_ongoing = sum(1 for s in signals if s["status"] == "ONGOING")
    log.info(f"  Active: {len(signals)}  (NEW: {n_new}, ONGOING: {n_ongoing})")
    if faded:
        log.info(f"  Exited this week: {[f['instrument'] for f in faded]}")

    # 4. Generate signal charts for active instruments
    log.info("Generating signal charts...")
    chart_map: dict[str, dict] = {}
    for sig in signals:
        inst = sig["instrument"]
        log.info(f"  Charting {inst}...")
        chart_map[inst] = {
            "signal":  make_signal_chart(inst, merged_df, prices),
            "scatter": make_overview_scatter(inst, merged_df),
        }

    # 5. Generate HTML report
    log.info("Building HTML report...")
    html_report = generate_html_report(
        signals, faded, stats_df, chart_map, warning, run_date
    )
    report_path = REPORTS_DIR / f"cot_report_{date_str}.html"
    report_path.write_text(html_report, encoding="utf-8")
    log.info(f"  Report saved: {report_path}")

    # 6. Email
    if args.no_email:
        log.info("--no-email flag set: skipping email.")
    else:
        sig_names = ", ".join(s["instrument"] for s in signals[:3])
        suffix    = " +" if len(signals) > 3 else ""
        subject   = (
            f"COT Report {date_str} | "
            f"{n_new} NEW, {n_ongoing} ONGOING"
            + (f" | {sig_names}{suffix}" if signals else " | No Signals")
        )
        email_body = generate_email_body(signals, faded, warning, run_date)
        try:
            send_email(subject, email_body, html_report,
                       OUTPUT_DIR / "results_table.csv", date_str)
        except EnvironmentError as e:
            log.error(f"Email not sent: {e}")
            log.error("Set EMAIL_USER, EMAIL_PASSWORD, EMAIL_TO in .env and retry.")

    # 7. Console summary
    print("\n" + "=" * 72)
    print(f"  COT REPORT — {date_str}")
    print("=" * 72)
    if signals:
        print(f"\n  ACTIVE SIGNALS ({len(signals)}):\n")
        for s in signals:
            e = s["edge"]
            mean_r = e.get("mean_return_pct", float("nan"))
            hit    = e.get("hit_rate_pct",    float("nan"))
            t_s    = e.get("t_stat",          float("nan"))
            p_v    = e.get("p_value",         float("nan"))
            sig_m  = " **" if (not np.isnan(p_v) and p_v < 0.05) else (
                     " *"  if (not np.isnan(p_v) and p_v < 0.10) else "")
            print(f"  [{s['status']:7s}] {s['instrument']:<22} "
                  f"z={s['z_score']:+.2f}  {s['extreme_type']:<16} "
                  f"{s['streak_weeks']}wk streak")
            print(f"            Edge: mean 4W={mean_r:+.2f}%  hit={hit:.1f}%  "
                  f"t={t_s:.2f}  p={p_v:.4f}{sig_m}")
            print()
    else:
        print("\n  No instruments currently at extremes.\n")

    if faded:
        print(f"  EXITED: {', '.join(f['instrument'] for f in faded)}")

    print(f"\n  Report: {report_path}")
    print(f"  Data:   {OUTPUT_DIR / 'results_table.csv'}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
