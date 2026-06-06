# FX COT — NET Money Manager Correlation Study

Weekly analysis of CFTC Commitment of Traders (COT) data for FX and crypto.
Detects NET Money Manager position extremes, measures their historical edge
against subsequent price action, and delivers an automated email report every Friday.

## What It Does

- Loads CFTC Financial TFF reports (2006–present)
- Computes NET Money Manager position z-scores (3-year rolling window)
- Downloads weekly prices from Yahoo Finance (yfinance)
- Calculates forward returns at 1, 2, 4, and 12-week horizons
- Runs t-tests to identify statistically significant edges at extremes
- Detects NEW and ONGOING signals each week
- Generates a standalone HTML report with embedded charts
- Sends the report by email via ProtonMail Bridge (or any SMTP server)

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

Extreme threshold: **±1.5 standard deviations** (3-year rolling z-score).

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

The repo includes a PowerShell one-liner to register a Task Scheduler job
that fires every **Friday at 18:30** (after the CFTC release):

```powershell
$python  = "C:\Users\127\AppData\Local\Programs\Python\Python314\python.exe"
$script  = "C:\Users\127\FX_COT\weekly_report.py"
$workdir = "C:\Users\127\FX_COT"
$action  = New-ScheduledTaskAction -Execute $python -Argument $script -WorkingDirectory $workdir
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "18:30"
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable -RunOnlyIfNetworkAvailable
Register-ScheduledTask -TaskName "COT_Weekly_Report" -TaskPath "\FX_COT\" -Action $action -Trigger $trigger -Settings $settings -Force
```

## Usage

```bash
# Full run — analysis + email
python weekly_report.py

# Generate report without sending email
python weekly_report.py --no-email

# Force re-download of price cache
python weekly_report.py --refresh

# Run analysis only (no email, saves results_table.csv and charts)
python cot_analysis.py
```

## Outputs

| File | Description |
|---|---|
| `reports/cot_report_YYYY-MM-DD.html` | Standalone HTML report (open in browser) |
| `results_table.csv` | Full stats: all instruments × all horizons |
| `cot_charts/` | Per-instrument overview charts (price / z-score / scatter) |
| `price_cache.parquet` | Cached weekly prices — delete to force refresh |

## Weekly Workflow

1. Friday ~15:30 ET — CFTC publishes new COT data
2. Download the updated `FinCom{YEAR}.txt` and drop it into `COT/`
3. 18:30 — Task Scheduler fires automatically
4. Report lands in configured inboxes

## Data Source

CFTC Commitments of Traders — Financial TFF (Traders in Financial Futures)
Released weekly, every Friday, reporting positions as of the prior Tuesday.
