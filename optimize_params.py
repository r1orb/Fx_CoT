"""
COT Parameter Optimizer
=======================
Grid-searches Z_WINDOW and Z_EXTREME using anchored expanding-window
cross-validation, then writes the best combination to params.json.

Run manually or schedule monthly (before the Friday weekly report).
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Force UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cot_analysis import (
    INSTRUMENT_MAP,
    OUTPUT_DIR,
    PRICE_START,
    align_and_forward_returns,
    analyze_extremes,
    fetch_prices,
    load_cot_files,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

PARAMS_FILE = OUTPUT_DIR / "params.json"

GRID = {
    "Z_WINDOW":  [78, 104, 130, 156, 182, 208],
    "Z_EXTREME": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
}

# Anchored CV folds: (train_start, train_end, test_start, test_end)
# Burn-in ends at 2014 to guarantee ≥ 208 weeks of history for the widest window.
FOLDS = [
    ("2014-01-01", "2018-01-01", "2018-01-01", "2020-01-01"),
    ("2014-01-01", "2020-01-01", "2020-01-01", "2022-01-01"),
    ("2014-01-01", "2022-01-01", "2022-01-01", "2024-01-01"),
]
HOLDOUT_START = "2024-01-01"

OPT_HORIZONS  = (4, 12)   # forward-return horizons to score
MIN_N         = 8          # discard rows with fewer observations
HIT_FLOOR     = 55.0       # hit rate must exceed this to score positively
P_THRESHOLD   = 0.05       # significance gate
GAP_ALLOWANCE = 0.05       # tolerated train-test gap before penalty kicks in


# ── Core helpers ───────────────────────────────────────────────────────────────

def _compute_net_mm_parametrized(
    df: pd.DataFrame, z_window: int, z_min_periods: int
) -> pd.DataFrame:
    """Identical to cot_analysis.compute_net_mm with explicit window params."""
    df = df.copy()
    df["net_mm"] = (
        df["Lev_Money_Positions_Long_All"] - df["Lev_Money_Positions_Short_All"]
    )
    df["oi"] = df["Open_Interest_All"]
    df["net_mm_pct_oi"] = df["net_mm"] / df["oi"].replace(0, np.nan)

    def _zscore(s: pd.Series) -> pd.Series:
        mu = s.rolling(z_window, min_periods=z_min_periods).mean()
        sd = s.rolling(z_window, min_periods=z_min_periods).std()
        return (s - mu) / sd.replace(0, np.nan)

    def _pct_rank(s: pd.Series) -> pd.Series:
        return s.rolling(52, min_periods=26).rank(pct=True)

    df["z_score"]     = df.groupby("instrument")["net_mm"].transform(_zscore)
    df["pct_rank_1y"] = df.groupby("instrument")["net_mm"].transform(_pct_rank)
    return df


def composite_score(stats_df: pd.DataFrame) -> float:
    """
    Aggregate edge quality across all instrument-extreme pairs at OPT_HORIZONS.

    Each qualifying row contributes:
        ((hit_rate - HIT_FLOOR) / 45) × (1 - p_value) × (1 - exp(-n / 20))
    Rows below MIN_N observations or p >= P_THRESHOLD contribute 0.
    Returns the mean across all rows (including zero contributors).
    """
    rows = stats_df[stats_df["horizon_weeks"].isin(OPT_HORIZONS)]
    scores = []
    for _, row in rows.iterrows():
        n   = row["n"]
        hit = row["hit_rate_pct"]
        p   = row["p_value"]
        if n < MIN_N or pd.isna(p) or pd.isna(hit) or p >= P_THRESHOLD:
            scores.append(0.0)
            continue
        reliability = 1.0 - np.exp(-n / 20.0)
        scores.append(((hit - HIT_FLOOR) / 45.0) * (1.0 - p) * reliability)
    return float(np.mean(scores)) if scores else 0.0


def _score_window(
    merged_full: pd.DataFrame, start: str, end: str, z_extreme: float
) -> float:
    """Score a single time slice at a given z_extreme threshold."""
    slc = merged_full[
        (merged_full["date"] >= pd.Timestamp(start))
        & (merged_full["date"] < pd.Timestamp(end))
    ]
    if slc.empty:
        return 0.0
    return composite_score(analyze_extremes(slc, threshold=z_extreme))


# ── Grid search ────────────────────────────────────────────────────────────────

def run_grid_search(
    cot_df: pd.DataFrame, prices: pd.DataFrame
) -> tuple[dict, list[dict]]:
    """
    Exhaustively search GRID, return (best_result_dict, all_results_list).
    Z_WINDOW is the outer loop so align_and_forward_returns is called only 6×.
    """
    all_results = []
    n_combos = len(GRID["Z_WINDOW"]) * len(GRID["Z_EXTREME"])
    done = 0

    for z_window in GRID["Z_WINDOW"]:
        z_min = z_window // 3
        cot_z = _compute_net_mm_parametrized(cot_df, z_window, z_min)
        merged = align_and_forward_returns(cot_z, prices)
        merged["date"] = pd.to_datetime(merged["date"])

        for z_extreme in GRID["Z_EXTREME"]:
            done += 1
            log.info(f"  [{done}/{n_combos}] Z_WINDOW={z_window}  Z_EXTREME={z_extreme}")

            fold_train, fold_test = [], []
            fold_detail = []
            for train_start, train_end, test_start, test_end in FOLDS:
                ts = _score_window(merged, train_start, train_end, z_extreme)
                ts_out = _score_window(merged, test_start, test_end, z_extreme)
                fold_train.append(ts)
                fold_test.append(ts_out)
                fold_detail.append({
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "test_score": round(ts_out, 6),
                })

            mean_test  = float(np.mean(fold_test))
            mean_train = float(np.mean(fold_train))
            gap        = mean_train - mean_test
            adjusted   = mean_test - max(0.0, gap - GAP_ALLOWANCE)

            all_results.append({
                "Z_WINDOW":           z_window,
                "Z_EXTREME":          z_extreme,
                "mean_test_score":    round(mean_test, 6),
                "mean_train_score":   round(mean_train, 6),
                "generalization_gap": round(gap, 6),
                "adjusted_score":     round(adjusted, 6),
                "folds":              fold_detail,
            })

    best = max(all_results, key=lambda r: r["adjusted_score"])
    return best, all_results


# ── Holdout evaluation ─────────────────────────────────────────────────────────

def evaluate_on_holdout(
    cot_df: pd.DataFrame,
    prices: pd.DataFrame,
    z_window: int,
    z_extreme: float,
) -> dict:
    """Out-of-sample check on the reserved holdout period (informational only)."""
    z_min  = z_window // 3
    cot_z  = _compute_net_mm_parametrized(cot_df, z_window, z_min)
    merged = align_and_forward_returns(cot_z, prices)
    merged["date"] = pd.to_datetime(merged["date"])
    holdout = merged[merged["date"] >= pd.Timestamp(HOLDOUT_START)]
    if holdout.empty:
        return {"holdout_score": None, "n_holdout_weeks": 0}
    score = composite_score(analyze_extremes(holdout, threshold=z_extreme))
    n_wks = int(holdout["date"].nunique())
    return {"holdout_score": round(score, 6), "n_holdout_weeks": n_wks}


# ── Output ─────────────────────────────────────────────────────────────────────

def write_params(best: dict, holdout: dict, all_results: list[dict]) -> None:
    payload = {
        "Z_WINDOW":      best["Z_WINDOW"],
        "Z_EXTREME":     best["Z_EXTREME"],
        "Z_MIN_PERIODS": best["Z_WINDOW"] // 3,
        "optimization_metadata": {
            "run_timestamp":      datetime.now().isoformat(timespec="seconds"),
            "mean_test_score":    best["mean_test_score"],
            "mean_train_score":   best["mean_train_score"],
            "generalization_gap": best["generalization_gap"],
            "adjusted_score":     best["adjusted_score"],
            "holdout_score":      holdout["holdout_score"],
            "n_holdout_weeks":    holdout["n_holdout_weeks"],
            "folds":              best["folds"],
            "all_results":        all_results,
        },
    }
    with open(PARAMS_FILE, "w") as f:
        json.dump(payload, f, indent=2)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== COT Parameter Optimizer ===")

    log.info("Loading COT data...")
    cot_df = load_cot_files()

    log.info("Fetching prices (uses cache when available)...")
    tickers = list(dict.fromkeys(t for _, t, _ in INSTRUMENT_MAP))
    prices  = fetch_prices(tickers, start=PRICE_START)

    n_combos = len(GRID["Z_WINDOW"]) * len(GRID["Z_EXTREME"])
    log.info(
        f"Grid search: {len(GRID['Z_WINDOW'])} Z_WINDOW × "
        f"{len(GRID['Z_EXTREME'])} Z_EXTREME = {n_combos} combos × "
        f"{len(FOLDS)} folds..."
    )
    best, all_results = run_grid_search(cot_df, prices)

    log.info(
        f"Best: Z_WINDOW={best['Z_WINDOW']}  Z_EXTREME={best['Z_EXTREME']}  "
        f"adj_score={best['adjusted_score']:.4f}  "
        f"(train={best['mean_train_score']:.4f}  "
        f"test={best['mean_test_score']:.4f}  "
        f"gap={best['generalization_gap']:.4f})"
    )

    log.info("Evaluating on holdout (2024-present)...")
    holdout = evaluate_on_holdout(cot_df, prices, best["Z_WINDOW"], best["Z_EXTREME"])
    h_score = holdout["holdout_score"]
    log.info(
        f"Holdout score: {h_score:.4f}  ({holdout['n_holdout_weeks']} weeks)"
        if h_score is not None
        else "Holdout: no data"
    )

    write_params(best, holdout, all_results)
    log.info(f"Wrote {PARAMS_FILE}")


if __name__ == "__main__":
    main()
