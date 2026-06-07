#!/usr/bin/env bash
# run.sh — Full COT pipeline: optimize parameters, then generate + send report.
#
# Usage:
#   bash run.sh                # optimize + full report with email
#   bash run.sh --no-email     # optimize + report, skip sending
#   bash run.sh --refresh      # optimize + force price-cache refresh
#   bash run.sh --skip-optimize  # skip optimizer, use existing params.json
#   bash run.sh --no-email --skip-optimize

set -euo pipefail

PYTHON="C:/Users/127/AppData/Local/Programs/Python/Python314/python.exe"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_OPTIMIZE=0
REPORT_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --skip-optimize) SKIP_OPTIMIZE=1 ;;
        *)               REPORT_ARGS+=("$arg") ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== COT Pipeline ==="
log "Working directory: $DIR"
cd "$DIR"

# ── Step 1: Parameter optimization ────────────────────────────────────────────
if [[ $SKIP_OPTIMIZE -eq 1 ]]; then
    log "Skipping optimizer (--skip-optimize)"
else
    log "Step 1/2 — Optimizing parameters..."
    "$PYTHON" optimize_params.py
    log "Step 1/2 — Done."
fi

# ── Step 2: Weekly report ──────────────────────────────────────────────────────
log "Step 2/2 — Generating report${REPORT_ARGS:+ (${REPORT_ARGS[*]})}..."
"$PYTHON" weekly_report.py "${REPORT_ARGS[@]}"
log "Step 2/2 — Done."

log "Pipeline complete."
