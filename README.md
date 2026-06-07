# FX COT — NET Money Manager Correlation Study

Weekly analysis of CFTC Commitment of Traders (COT) data for FX and crypto.
Detects NET Money Manager position extremes, measures their historical edge
against subsequent price action, and delivers an automated email report every Friday.

## What It Does

- Loads CFTC Financial TFF reports (2006–present)
- Computes NET Money Manager position z-scores (optimized rolling window)
- Downloads weekly prices from Yahoo Finance (yfinance)
- Calculates forward returns at 1, 2, 4, and 12-week horizons
- Runs t-tests to identify statistically significant edges at extremes
- Detects NEW and ONGOING signals each week
- Generates a standalone HTML report with embedded charts
- Sends the report by email via ProtonMail Bridge (or any SMTP server)
- Self-optimizes z-score parameters monthly via walk-forward grid search

## Instruments Covered

| COT Market | Price Feed | Notes |
|---|---|---|
| EURO FX | EURUSD=X | |
| BRITISH POUND | GBPUSD=X | |
| JAPANESE YEN | USDJPY=X | Return sign flipped |
| AUSTRALIAN DOLLAR | AUDUSD=X | |
| CANADIAN DOLLAR | USDCAD=X | Return sign flipped |
| SWISS FRANC | USDCHF=X | Return sign flipped |
| NZ DOLLAR | NZDUSD=X | |
| BITCOIN | BTC-USD | CFTC data from 2018 |
| ETHER CASH SETTLED | ETH-USD | CFTC data from 2021 |

## Edge Criterion

A signal is considered statistically significant when:
- **p < 0.05** (t-test against zero return)
- **Hit rate > 55%** at the 4-week horizon

Extreme threshold: **±Z_EXTREME standard deviations** (optimized rolling z-score window).
Default fallback values: Z_WINDOW = 156 weeks, Z_EXTREME = 1.5.

## Scripts

| Script | Purpose |
|---|---|
| `run.sh` | **Main entry point.** Runs optimizer then report in sequence. |
| `optimize_params.py` | Grid-searches Z_WINDOW and Z_EXTREME; writes `params.json`. |
| `weekly_report.py` | Generates HTML report and sends email. |
| `cot_analysis.py` | Core analysis library (z-scores, forward returns, edge stats). |

## Setup

### 1. Install dependencies

```bash
pip install pandas yfinance matplotlib scipy pyarrow python-dotenv
```

### 2. Add COT data

Download Financial TFF reports from the CFTC:
[https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm](https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm)

Place `.txt` files in the `COT/` folder. Naming convention used: `FinCom{YEAR}.txt`.

### 3. Configure email

```bash
cp .env.example .env
```

Edit `.env` with your SMTP credentials. For ProtonMail Bridge:

```
EMAIL_SMTP_HOST=127.0.0.1
EMAIL_SMTP_PORT=1025
EMAIL_USER=you@proton.me
EMAIL_PASSWORD=your_bridge_password
EMAIL_TO=you@proton.me,other@example.com
```

Bridge must be running when the report is sent.
Bridge password is shown in the Bridge app under **Account → SMTP settings**.

### 4. Schedule (Windows)

Two Task Scheduler jobs cover the full weekly cycle.

**Friday 18:30 — weekly report with email**

```powershell
$python  = "C:\Users\127\AppData\Local\Programs\Python\Python314\python.exe"
$script  = "C:\Users\127\FX_COT\weekly_report.py"
$workdir = "C:\Users\127\FX_COT"
$action  = New-ScheduledTaskAction -Execute $python -Argument $script -WorkingDirectory $workdir
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "18:30"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable -RunOnlyIfNetworkAvailable
Register-ScheduledTask -TaskName "COT_Weekly_Report" -TaskPath "\FX_COT\" -Action $action -Trigger $trigger -Settings $settings -Force
```

**Saturday 07:00 — optimize parameters + analysis (no email)**

```powershell
$bash    = "C:\Program Files\Git\usr\bin\bash.exe"
$script  = "C:\Users\127\FX_COT\run.sh"
$workdir = "C:\Users\127\FX_COT"
$action  = New-ScheduledTaskAction -Execute $bash -Argument "--login `"$script`" --no-email" -WorkingDirectory $workdir
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "07:00"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 2) -StartWhenAvailable -RunOnlyIfNetworkAvailable
Register-ScheduledTask -TaskName "COT_Full_Pipeline" -TaskPath "\FX_COT\" -Action $action -Trigger $trigger -Settings $settings -Force
```

## Usage

```bash
# Full pipeline — optimize parameters, then generate report + send email
bash run.sh

# Full pipeline — skip email
bash run.sh --no-email

# Full pipeline — force price-cache refresh
bash run.sh --refresh

# Skip optimizer, use existing params.json (faster weekly run)
bash run.sh --skip-optimize

# Report only (no optimizer, no email)
bash run.sh --skip-optimize --no-email

# Run optimizer standalone
python optimize_params.py

# Run analysis only (saves results_table.csv and charts, no email)
python cot_analysis.py
```

## Parameter Optimization

`optimize_params.py` automatically finds the best z-score parameters using
3-fold anchored walk-forward cross-validation:

- **Grid:** Z_WINDOW ∈ {78, 104, 130, 156, 182, 208} weeks × Z_EXTREME ∈ {1.0, 1.25, 1.5, 1.75, 2.0, 2.5}
- **Objective:** maximize directional hit rate × statistical significance at 4W and 12W horizons
- **Validation:** anchored expanding-window CV (2014→2018/2020/2022 train, 2018–2024 test) + held-out 2024–present
- **Output:** `params.json` (ignored by git) — loaded automatically by `cot_analysis.py` on every run

If `params.json` is absent or invalid, the system falls back to hardcoded defaults (Z_WINDOW=156, Z_EXTREME=1.5).

## Outputs

| File | Description |
|---|---|
| `reports/cot_report_YYYY-MM-DD.html` | Standalone HTML report (open in browser) |
| `results_table.csv` | Full stats: all instruments × all horizons |
| `cot_charts/` | Per-instrument overview charts (price / z-score / scatter) |
| `price_cache.parquet` | Cached weekly prices — delete to force refresh |
| `params.json` | Optimized parameters + full CV diagnostics (generated, not committed) |

## Weekly Workflow

1. Friday ~15:30 ET — CFTC publishes new COT data
2. Download the updated `FinCom{YEAR}.txt` and drop it into `COT/`
3. Friday 18:30 — `COT_Weekly_Report` task fires, sends email report
4. Saturday 07:00 — `COT_Full_Pipeline` task fires, re-optimizes parameters and refreshes analysis

## Data Source

CFTC Commitments of Traders — Financial TFF (Traders in Financial Futures)
Released weekly, every Friday, reporting positions as of the prior Tuesday.
