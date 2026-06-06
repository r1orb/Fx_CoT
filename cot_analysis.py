"""
COT NET Money Manager vs Price Action — Correlation Study
FX (EUR, GBP, JPY, AUD, CAD, CHF, NZD) + Crypto (BTC, ETH)

Outputs:
  results_table.csv   — stats per instrument × horizon
  cot_charts/         — one PNG per instrument
  price_cache.parquet — cached weekly prices (re-used on subsequent runs)
"""

import glob
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows so Unicode chars in print() don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────

COT_DIR    = Path(r"C:\Users\127\FX_COT\COT")
OUTPUT_DIR = Path(r"C:\Users\127\FX_COT")
CHARTS_DIR = OUTPUT_DIR / "cot_charts"
CACHE_FILE = OUTPUT_DIR / "price_cache.parquet"

Z_EXTREME       = 1.5   # z-score threshold for "extreme" classification
Z_WINDOW        = 156   # rolling window for z-score (3 years of weekly data)
Z_MIN_PERIODS   = 52    # minimum history before z-score is meaningful
FWD_HORIZONS    = [1, 2, 4, 12]   # forward return horizons in weeks
PRICE_START     = "2010-01-01"

# (substring to match in COT market name, yfinance ticker, invert_sign)
# invert_sign=True when COT contract is long non-USD but yfinance ticker is USD-base
# e.g. long JPY futures = bullish JPY = USDJPY falls → flip return sign
INSTRUMENT_MAP = [
    ("EURO FX",            "EURUSD=X", False),
    ("BRITISH POUND",      "GBPUSD=X", False),
    ("JAPANESE YEN",       "USDJPY=X", True),
    ("AUSTRALIAN DOLLAR",  "AUDUSD=X", False),
    ("CANADIAN DOLLAR",    "USDCAD=X", True),
    ("SWISS FRANC",        "USDCHF=X", True),
    ("NZ DOLLAR",          "NZDUSD=X", False),
    ("BITCOIN",            "BTC-USD",  False),
    ("ETHER CASH SETTLED", "ETH-USD",  False),
]

# ── Step 1: Load COT data ─────────────────────────────────────────────────────

def load_cot_files() -> pd.DataFrame:
    files = sorted(glob.glob(str(COT_DIR / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files in {COT_DIR}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=",", low_memory=False, quotechar='"')
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: could not read {f}: {e}")

    raw = pd.concat(dfs, ignore_index=True)

    date_col = "Report_Date_as_YYYY-MM-DD"
    raw["date"] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=["date"])

    for col in ["Lev_Money_Positions_Long_All", "Lev_Money_Positions_Short_All",
                "Open_Interest_All"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    name_col = "Market_and_Exchange_Names"
    tagged = []
    for match_str, ticker, invert in INSTRUMENT_MAP:
        mask = raw[name_col].str.upper().str.contains(match_str.upper(), na=False)
        n = mask.sum()
        if n == 0:
            print(f"  Warning: no COT rows matched '{match_str}'")
            continue
        sub = raw[mask].copy()
        sub["instrument"] = match_str
        sub["ticker"]     = ticker
        sub["invert"]     = invert
        tagged.append(sub)

    if not tagged:
        raise ValueError("No instruments matched — check COT_DIR and INSTRUMENT_MAP.")

    combined = pd.concat(tagged, ignore_index=True)

    # Prefer "Combined" rows over "FutOnly" when both exist for same date
    combined["_is_combined"] = (
        combined["FutOnly_or_Combined"].str.strip().str.lower() == "combined"
    )
    combined = combined.sort_values(
        ["instrument", "date", "_is_combined"], ascending=[True, True, False]
    )
    combined = combined.drop_duplicates(subset=["instrument", "date"], keep="first")
    combined = combined.drop(columns=["_is_combined"])
    combined = combined.sort_values(["instrument", "date"]).reset_index(drop=True)

    return combined


# ── Step 2: Compute NET MM z-scores ───────────────────────────────────────────

def compute_net_mm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["net_mm"] = (
        df["Lev_Money_Positions_Long_All"] - df["Lev_Money_Positions_Short_All"]
    )
    df["oi"] = df["Open_Interest_All"]
    df["net_mm_pct_oi"] = df["net_mm"] / df["oi"].replace(0, np.nan)

    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).mean()
        sd = s.rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS).std()
        return (s - mu) / sd.replace(0, np.nan)

    def _pct_rank(s: pd.Series) -> pd.Series:
        return s.rolling(52, min_periods=26).rank(pct=True)

    df["z_score"]     = df.groupby("instrument")["net_mm"].transform(_zscore)
    df["pct_rank_1y"] = df.groupby("instrument")["net_mm"].transform(_pct_rank)

    return df


# ── Step 3: Fetch prices ───────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], start: str = PRICE_START) -> pd.DataFrame:
    if CACHE_FILE.exists():
        cached = pd.read_parquet(CACHE_FILE)
        missing = [t for t in tickers if t not in cached.columns]
        if not missing:
            print("  Using cached price data.")
            return cached
        print(f"  Cache present but missing {missing}. Refreshing.")

    print(f"  Downloading weekly closes for: {tickers}")
    raw = yf.download(tickers, start=start, interval="1wk",
                      auto_adjust=True, progress=False)

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(
        columns={"Close": tickers[0]}
    )
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.to_parquet(CACHE_FILE)
    print(f"  Cached to {CACHE_FILE}")
    return close


# ── Step 4: Align COT with prices, compute forward returns ────────────────────

def align_and_forward_returns(cot_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    COT report date = Tuesday.  Entry = following Friday (+3 days).
    Forward return for horizon h = log(price_h_weeks_later / entry_price).
    Invert sign for pairs where COT contract direction is inverse to yfinance quote.
    """
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()

    results = []

    for instrument, grp in cot_df.groupby("instrument"):
        ticker = grp["ticker"].iloc[0]
        invert = grp["invert"].iloc[0]

        if ticker not in prices.columns:
            print(f"  Warning: {ticker} not in price data, skipping {instrument}.")
            continue

        px = prices[ticker].dropna()

        # Build price table with pre-computed forward log returns
        px_df = px.to_frame("entry_price").reset_index().rename(columns={"index": "price_date"})
        if "Date" in px_df.columns:
            px_df = px_df.rename(columns={"Date": "price_date"})
        px_df = px_df.sort_values("price_date").reset_index(drop=True)

        for h in FWD_HORIZONS:
            fwd_vals = np.log(px.shift(-h) / px)
            if invert:
                fwd_vals = -fwd_vals
            px_df[f"fwd_{h}w"] = fwd_vals.values

        # Entry date = COT Tuesday + 3 days → nearest Friday
        g = grp[["date", "net_mm", "z_score", "pct_rank_1y"]].copy()
        g["entry_date"] = (g["date"] + pd.Timedelta(days=3)).dt.normalize()
        # Normalize datetime precision to avoid merge_asof dtype mismatch
        g["entry_date"] = g["entry_date"].astype("datetime64[s]")
        px_df["price_date"] = px_df["price_date"].astype("datetime64[s]")
        g = g.sort_values("entry_date").reset_index(drop=True)

        merged = pd.merge_asof(
            g, px_df,
            left_on="entry_date",
            right_on="price_date",
            direction="nearest",
            tolerance=pd.Timedelta(days=5),
        )
        merged = merged.dropna(subset=["entry_price"])
        merged["instrument"] = instrument
        merged["ticker"]     = ticker
        merged["invert"]     = invert
        results.append(merged)

    if not results:
        raise ValueError("Alignment produced no rows — check date ranges.")

    return pd.concat(results, ignore_index=True)


# ── Step 5a: Statistics ────────────────────────────────────────────────────────

def analyze_extremes(df: pd.DataFrame, threshold: float = Z_EXTREME) -> pd.DataFrame:
    records = []

    for instrument, grp in df.groupby("instrument"):
        grp = grp.dropna(subset=["z_score"])
        ticker = grp["ticker"].iloc[0]

        subsets = [
            ("LONG extreme",  grp[grp["z_score"] >  threshold],  1),
            ("SHORT extreme", grp[grp["z_score"] < -threshold], -1),
        ]

        for label, sub, direction in subsets:
            for h in FWD_HORIZONS:
                col = f"fwd_{h}w"
                rets = sub[col].dropna()
                n = len(rets)

                if n < 4:
                    records.append({
                        "instrument": instrument, "ticker": ticker,
                        "extreme_type": label, "horizon_weeks": h,
                        "n": n, "mean_return_pct": np.nan,
                        "std_return_pct": np.nan, "hit_rate_pct": np.nan,
                        "t_stat": np.nan, "p_value": np.nan,
                    })
                    continue

                mean_r  = rets.mean()
                std_r   = rets.std()
                t, p    = stats.ttest_1samp(rets, 0)
                hit     = (np.sign(rets) == direction).mean()

                records.append({
                    "instrument":     instrument,
                    "ticker":         ticker,
                    "extreme_type":   label,
                    "horizon_weeks":  h,
                    "n":              n,
                    "mean_return_pct": round(mean_r * 100, 3),
                    "std_return_pct":  round(std_r  * 100, 3),
                    "hit_rate_pct":    round(hit    * 100, 1),
                    "t_stat":          round(t, 3),
                    "p_value":         round(p, 4),
                })

    return pd.DataFrame(records)


# ── Step 5b: Charts ────────────────────────────────────────────────────────────

def make_charts(merged_df: pd.DataFrame, prices: pd.DataFrame) -> None:
    CHARTS_DIR.mkdir(exist_ok=True)
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index).normalize()

    for instrument, grp in merged_df.groupby("instrument"):
        ticker = grp["ticker"].iloc[0]
        invert = grp["invert"].iloc[0]

        if ticker not in prices.columns:
            continue

        px = prices[ticker].dropna()
        grp = grp.sort_values("date")

        fig = plt.figure(figsize=(14, 11))
        gs  = gridspec.GridSpec(3, 1, height_ratios=[2, 1.5, 1.3], hspace=0.38)

        # --- Panel 1: Price ---
        ax1 = fig.add_subplot(gs[0])
        direction_note = " (USDJPY/USDCAD/USDCHF shown; returns flipped)" if invert else ""
        ax1.plot(px.index, px.values, lw=1.1, color="#1f77b4")
        ax1.set_yscale("log")
        ax1.set_title(
            f"{instrument}  •  {ticker}{direction_note}",
            fontsize=12, fontweight="bold", pad=6
        )
        ax1.set_ylabel("Price (log scale)", fontsize=8)
        ax1.grid(True, alpha=0.25)

        # Mark extreme periods on price chart
        ex_long  = grp[grp["z_score"] >  Z_EXTREME]
        ex_short = grp[grp["z_score"] < -Z_EXTREME]
        for d in ex_long["entry_date"]:
            ax1.axvline(d, color="#d62728", alpha=0.08, lw=0.8)
        for d in ex_short["entry_date"]:
            ax1.axvline(d, color="#1f77b4", alpha=0.08, lw=0.8)

        # --- Panel 2: Z-score ---
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.plot(grp["date"], grp["z_score"], lw=1.0, color="#2ca02c", label="NET MM z-score")
        ax2.axhline(0,          color="gray",    lw=0.7, ls="--")
        ax2.axhline( Z_EXTREME, color="#d62728", lw=1.0, ls="--", label=f"±{Z_EXTREME}")
        ax2.axhline(-Z_EXTREME, color="#d62728", lw=1.0, ls="--")
        ax2.fill_between(
            grp["date"],
            Z_EXTREME, grp["z_score"].clip(lower=Z_EXTREME),
            alpha=0.18, color="#d62728"
        )
        ax2.fill_between(
            grp["date"],
            -Z_EXTREME, grp["z_score"].clip(upper=-Z_EXTREME),
            alpha=0.18, color="#1f77b4"
        )
        ax2.set_ylabel("Z-Score (3yr rolling)", fontsize=8)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(True, alpha=0.25)

        # --- Panel 3: Scatter z-score vs 4W forward return ---
        ax3 = fig.add_subplot(gs[2])
        valid = grp.dropna(subset=["z_score", "fwd_4w"])
        is_extreme = valid["z_score"].abs() > Z_EXTREME
        ax3.scatter(
            valid.loc[~is_extreme, "z_score"],
            valid.loc[~is_extreme, "fwd_4w"] * 100,
            color="#aec7e8", alpha=0.4, s=12, label="Normal"
        )
        ax3.scatter(
            valid.loc[is_extreme, "z_score"],
            valid.loc[is_extreme, "fwd_4w"] * 100,
            color="#d62728", alpha=0.65, s=18, zorder=3, label="Extreme"
        )
        ax3.axhline(0, color="gray", lw=0.7, ls="--")
        ax3.axvline(0, color="gray", lw=0.7, ls="--")
        ax3.set_xlabel("NET MM Z-Score", fontsize=8)
        ax3.set_ylabel("4-Week Fwd Return (%)", fontsize=8)
        ax3.set_title("Z-Score vs 4-Week Forward Return", fontsize=8)
        ax3.grid(True, alpha=0.25)

        if len(valid) > 10:
            slope, intercept, r_val, p_val, _ = stats.linregress(
                valid["z_score"], valid["fwd_4w"] * 100
            )
            x_line = np.linspace(valid["z_score"].min(), valid["z_score"].max(), 50)
            ax3.plot(
                x_line, slope * x_line + intercept,
                color="#ff7f0e", lw=1.5,
                label=f"r = {r_val:.2f}   p = {p_val:.3f}"
            )
        ax3.legend(fontsize=8)

        safe = instrument.replace(" ", "_").replace("/", "_")
        path = CHARTS_DIR / f"{safe}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path.name}")


# ── Step 5c: Console summary ──────────────────────────────────────────────────

def print_summary(stats_df: pd.DataFrame) -> None:
    print("\n" + "=" * 82)
    print("  COT NET MM EXTREMES  —  EDGE SUMMARY  (4-Week Horizon)")
    print("=" * 82)

    focus = stats_df[
        (stats_df["horizon_weeks"] == 4) &
        (stats_df["extreme_type"].isin(["LONG extreme", "SHORT extreme"]))
    ].dropna(subset=["t_stat"]).copy()

    focus["abs_t"] = focus["t_stat"].abs()
    focus = focus.sort_values("abs_t", ascending=False)

    header = (
        f"\n{'Instrument':<22} {'Type':<16} {'N':>4}  "
        f"{'Mean%':>7}  {'Hit%':>6}  {'t-stat':>7}  {'p-val':>7}  {'Sig':>4}"
    )
    print(header)
    print("-" * 82)

    for _, row in focus.iterrows():
        if row["p_value"] < 0.05:
            sig = "**"
        elif row["p_value"] < 0.10:
            sig = " *"
        else:
            sig = "  "
        print(
            f"{row['instrument']:<22} {row['extreme_type']:<16} {int(row['n']):>4}  "
            f"{row['mean_return_pct']:>7.2f}  {row['hit_rate_pct']:>6.1f}  "
            f"{row['t_stat']:>7.2f}  {row['p_value']:>7.4f}  {sig}"
        )

    print("\n** p < 0.05   * p < 0.10")

    edges = focus[(focus["p_value"] < 0.05) & (focus["hit_rate_pct"] > 55)]
    print(f"\n→ Instruments with significant edge (p<0.05 AND hit rate>55%): {len(edges)}")
    if not edges.empty:
        print()
        for _, row in edges.iterrows():
            direction = "bullish" if "LONG" in row["extreme_type"] else "bearish"
            print(
                f"  [OK] {row['instrument']} ({row['extreme_type']}):  "
                f"when MM extremely {direction} => mean 4W return = {row['mean_return_pct']:.2f}%,  "
                f"hit rate = {row['hit_rate_pct']:.1f}%,  t = {row['t_stat']:.2f}"
            )

    # Also print full horizon table for significant pairs
    sig_instruments = edges["instrument"].unique()
    if len(sig_instruments) > 0:
        print("\n" + "-" * 82)
        print("  FULL HORIZON BREAKDOWN (significant instruments only)")
        print("-" * 82)
        full = stats_df[
            stats_df["instrument"].isin(sig_instruments) &
            stats_df["extreme_type"].isin(["LONG extreme", "SHORT extreme"])
        ].dropna(subset=["t_stat"])
        print(full.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    CHARTS_DIR.mkdir(exist_ok=True)

    print("\n[1/5] Loading COT data...")
    cot_df = load_cot_files()
    instruments_found = cot_df["instrument"].value_counts()
    print(f"  {len(cot_df):,} rows  |  {cot_df['instrument'].nunique()} instruments")
    print(f"  Date range: {cot_df['date'].min().date()} to {cot_df['date'].max().date()}")
    print(f"  Rows per instrument:\n{instruments_found.to_string()}")

    print("\n[2/5] Computing NET MM positions and z-scores...")
    cot_df = compute_net_mm(cot_df)
    print(f"  Z-score stats (all instruments):")
    print(cot_df.groupby("instrument")["z_score"].describe()[
        ["count", "min", "max", "mean", "std"]
    ].round(2).to_string())

    print("\n[3/5] Fetching weekly price data from yfinance...")
    tickers = list(dict.fromkeys(t for _, t, _ in INSTRUMENT_MAP))  # preserve order, dedupe
    prices = fetch_prices(tickers, start=PRICE_START)
    print(f"  Price data: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"  Available tickers: {list(prices.columns)}")

    print("\n[4/5] Aligning COT with prices and computing forward returns...")
    merged_df = align_and_forward_returns(cot_df, prices)
    print(f"  Aligned rows: {len(merged_df):,}")
    print(f"  Coverage per instrument:")
    print(merged_df.groupby("instrument")["fwd_4w"].count().to_string())

    print("\n[5/5] Analyzing extremes and generating outputs...")
    stats_df = analyze_extremes(merged_df)

    out_csv = OUTPUT_DIR / "results_table.csv"
    stats_df.to_csv(out_csv, index=False)
    print(f"  Results saved → {out_csv}")

    print("\n  Generating charts...")
    make_charts(merged_df, prices)

    print_summary(stats_df)

    print(f"\n{'='*82}")
    print(f"  Done.  Charts: {CHARTS_DIR}")
    print(f"  Full table: {out_csv}")
    print(f"{'='*82}\n")


if __name__ == "__main__":
    main()
