#!/usr/bin/env bash
# ======================================================================
# QuantaAlpha — Daily Live Signal Generation
# ======================================================================
# Usage:
#   bash docs/live/run_signals.sh              # full pipeline (ingest + signals)
#   bash docs/live/run_signals.sh --signals    # signals only (skip data update)
#   bash docs/live/run_signals.sh --ingest     # data update only (skip signals)
#   bash docs/live/run_signals.sh --report     # regenerate today's report only
#
# This script:
#   1. Activates the quantaalpha-ollama conda environment
#   2. Downloads the latest EOD market data (data ingest)
#   3. Generates model scores for all tickers
#   4. Compares scores against current holdings (data/live/positions.json)
#   5. Produces buy/sell orders via TopkDropout rebalancing
#   6. Saves pending orders to data/live/pending_orders_{date}.json
#   7. Generates a Markdown report to data/live/reports/report_{date}.md
#
# Run after US market close (recommended: 16:45–17:30 ET).
# ======================================================================

set -euo pipefail

# ── Resolve paths ────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${REPO_ROOT}/configs/live.yaml"
CONDA_BIN="${CONDA_BIN:-${HOME}/anaconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-quantaalpha-ollama}"
POSITIONS_FILE="${REPO_ROOT}/data/live/positions.json"
ORDERS_DIR="${REPO_ROOT}/data/live"

# ── Parse flags ──────────────────────────────────────────────────────
DO_INGEST=true
DO_SIGNALS=true
DO_REPORT_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --signals)  DO_INGEST=false ;;
        --ingest)   DO_SIGNALS=false ;;
        --report)   DO_INGEST=false; DO_SIGNALS=false; DO_REPORT_ONLY=true ;;
        --help|-h)
            head -18 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown flag: $arg (use --ingest, --signals, --report, or no flag for full pipeline)"
            exit 1 ;;
    esac
done

# ── Validate environment ─────────────────────────────────────────────
if [[ ! -x "$CONDA_BIN" ]]; then
    echo "ERROR: conda not found at $CONDA_BIN"
    echo "Set CONDA_BIN=/path/to/conda before running."
    exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: live config not found at $CONFIG"
    exit 1
fi

# ── Helper: run inside conda env ─────────────────────────────────────
run_conda() {
    "$CONDA_BIN" run -n "$CONDA_ENV" --cwd "$REPO_ROOT" "$@"
}

# ── Display current state ────────────────────────────────────────────
echo "========================================"
echo " QuantaAlpha Live Signal Generation"
echo " $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================"
echo "Repo:      $REPO_ROOT"
echo "Config:    $CONFIG"
echo "Conda env: $CONDA_ENV"
echo "Ingest:    $DO_INGEST  |  Signals: $DO_SIGNALS"
echo ""

if [[ -f "$POSITIONS_FILE" ]]; then
    echo "── Current Holdings ──"
    run_conda python -c "
import json, sys
with open('$POSITIONS_FILE') as f:
    state = json.load(f)
positions = state.get('positions', {})
acct = state.get('account_value', 'N/A')
dt = state.get('date', 'N/A')
pnl = state.get('pnl', {})
cash = state.get('cash', 0)
print(f'  Date:           {dt}')
print(f'  Account value:  \${acct:,.2f}' if isinstance(acct, (int, float)) else f'  Account value:  {acct}')
print(f'  Positions:      {len(positions)}')
cum_pnl = pnl.get('cumulative_pnl', 0)
cum_ret = pnl.get('cumulative_excess_return', 0)
print(f'  Cumul P&L:      \${cum_pnl:+,.2f} ({cum_ret:+.2%})')
if cash:
    print(f'  Cash:           \${cash:+,.2f}')
print(f'  Holdings:       {\"  \".join(sorted(positions.keys()))}')
"
    echo ""
else
    echo "  No existing positions (fresh start with \$1M capital)."
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 1: Data Ingest — download latest EOD data
# ══════════════════════════════════════════════════════════════════════
if [[ "$DO_INGEST" == "true" ]]; then
    echo "── Step 1: Data Ingest ──"
    run_conda python -c "
import sys, yaml, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from quantaalpha.live.data_ingestor import DataIngestor

with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)

h5_path = cfg['data']['h5_path']
print(f'  H5 path: {h5_path}')
ingestor = DataIngestor(h5_path=h5_path)
rows = ingestor.ingest(force_full=False)
print(f'  New rows added: {rows}')
print('  Ingest complete.')
"
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 2: Signal Generation — score tickers, rebalance, save orders
# ══════════════════════════════════════════════════════════════════════
if [[ "$DO_SIGNALS" == "true" ]]; then
    echo "── Step 2: Signal Generation ──"
    run_conda python -c "
import sys, json, yaml, logging, os

# Suppress noisy factor computation output — only show warnings+
logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')

from quantaalpha.live.scheduler import TradingScheduler

scheduler = TradingScheduler('$CONFIG')

# Redirect stdout during signal generation to suppress factor progress lines
from io import StringIO
_capture = StringIO()
_orig_stdout = sys.stdout
sys.stdout = _capture
try:
    result = scheduler.run_signal()
finally:
    sys.stdout = _orig_stdout
    # Show warnings from captured output (if any)
    captured = _capture.getvalue()
    for line in captured.splitlines():
        if 'WARNING' in line or 'ERROR' in line or 'KILL' in line:
            print(f'  {line}', file=sys.stderr)

if not result:
    print('  WARNING: No orders generated (empty scores or error).')
    sys.exit(1)

if result.get('kill_switch'):
    print(f'  KILL-SWITCH TRIGGERED — daily P&L: \${result[\"daily_pnl\"]:+,.2f}')
    print('  No orders generated. Review risk limits.')
    sys.exit(2)

# scheduler.run_signal() already saved the Markdown report.
# Print it to stdout so the terminal shows the full report.
report_file = result.get('report_file')
if report_file:
    from pathlib import Path
    md_path = Path(report_file)
    if md_path.exists():
        print(md_path.read_text())
        print(f'  (Report saved: {report_file})', file=sys.stderr)
    else:
        # Fallback: generate inline if file not written
        from quantaalpha.live.position_tracker import PositionTracker
        import yaml as _y
        with open('$CONFIG') as _f:
            _cfg = _y.safe_load(_f)
        _out = _cfg.get('output', {})
        tracker = PositionTracker(
            positions_file=_out.get('positions_file', 'data/live/positions.json'),
            pnl_dir=_out.get('pnl_dir', 'data/live/pnl'),
        )
        ledger = tracker.load_ledger(last_n=10)
        from quantaalpha.live.report_md import generate_report
        print(generate_report(result, ledger=ledger))
"
    echo ""
fi

# ══════════════════════════════════════════════════════════════════════
# STEP 3 (optional): Regenerate report from existing orders file
# ══════════════════════════════════════════════════════════════════════
if [[ "$DO_REPORT_ONLY" == "true" ]]; then
    echo "── Regenerating report from existing orders ──"
    run_conda python -c "
import json, sys, glob
from pathlib import Path

# Find the most recent orders file
files = sorted(glob.glob('$ORDERS_DIR/pending_orders_*.json'))
if not files:
    print('ERROR: No pending_orders_*.json found in $ORDERS_DIR', file=sys.stderr)
    sys.exit(1)
orders_path = Path(files[-1])
print(f'  Using: {orders_path}', file=sys.stderr)
data = json.loads(orders_path.read_text())

# Load ledger
import yaml
from quantaalpha.live.position_tracker import PositionTracker
with open('$CONFIG') as _f:
    _cfg = yaml.safe_load(_f)
_out = _cfg.get('output', {})
tracker = PositionTracker(
    positions_file=_out.get('positions_file', 'data/live/positions.json'),
    pnl_dir=_out.get('pnl_dir', 'data/live/pnl'),
)
ledger = tracker.load_ledger(last_n=10)

from quantaalpha.live.report_md import generate_report, save_report
report_path = save_report(data, ledger=ledger, output_dir=Path('$ORDERS_DIR/reports'))
print(report_path.read_text())
print(f'  (Report saved: {report_path})', file=sys.stderr)
"
    echo ""
fi

echo "========================================"
echo " Done — $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "========================================"
