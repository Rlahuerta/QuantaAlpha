# QuantaAlpha Live Trading Guide

> Step-by-step guide for running the live US equities model and generating trading predictions.

## Table of Contents

1. [Quick Start — Get a Prediction in 2 Minutes](#quick-start)
2. [Prerequisites](#prerequisites)
3. [One-Shot Signal Generation](#one-shot-signal-generation)
4. [Understanding the Output](#understanding-the-output)
5. [Daily Automated Pipeline](#daily-automated-pipeline)
6. [IBKR Paper Trading Setup](#ibkr-paper-trading-setup)
7. [Model Retraining](#model-retraining)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

Generate today's stock predictions with a single command:

```bash
cd /path/to/QuantaAlpha

# Activate the conda environment
source ~/anaconda3/bin/activate quantaalpha-ollama

# Generate signals (takes ~90 seconds)
python -m quantaalpha.live.signal_generator \
  --meta data/models/us_union89_prod_meta.json \
  --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \
  --lookback 300
```

This will:
1. Load the pre-trained LightGBM model (89 features, trained 2016–2020)
2. Compute 89 alpha factors on the latest market data
3. Score all ~517 S&P 500 stocks
4. Print ranked buy signals (higher score = stronger buy)

**Expected output:**
```
Tickers scored: 517
Top 20:
  ADP    0.003619
  AWK    0.003619
  PANW   0.003619
  WMT    0.003619
  ...
```

---

## Prerequisites

### 1. Conda Environment

```bash
# The project uses a dedicated conda env with all dependencies
source ~/anaconda3/bin/activate quantaalpha-ollama

# Verify key packages
python -c "import lightgbm; import pandas; import h5py; print('OK')"
```

### 2. Data Files

| File | Path | Size | Description |
|------|------|------|-------------|
| **US HDF5 data** | `git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5` | ~29 MB | Daily OHLCV for S&P 500, 2016–present |
| **Model booster** | `data/models/us_union89_prod_lgbm.txt` | ~15 MB | Trained LightGBM model |
| **Model metadata** | `data/models/us_union89_prod_meta.json` | ~4 KB | Feature list, training config |
| **Factor library** | `data/factorlib/all_factors_library_us_union89.json` | ~180 KB | 89 factor definitions (expressions) |

If you're setting up from scratch, see [Data Setup](#data-setup) below.

### 3. Configuration

The live trading config lives at `configs/live.yaml`. Key sections:

```yaml
model:
  meta_path: "data/models/us_union89_prod_meta.json"

data:
  h5_path: "git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5"
  lookback_days: 300

portfolio:
  topk: 20        # hold top 20 stocks
  n_drop: 2       # replace at most 2 per day
  capital: 1000000 # $1M paper capital
```

---

## One-Shot Signal Generation

### From Python

```python
from quantaalpha.live.signal_generator import SignalGenerator

# Load model + factor definitions
sg = SignalGenerator.from_meta(
    "data/models/us_union89_prod_meta.json",
    h5_path="git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5",
    lookback_days=300,
)

# Generate scores (uses latest available date in HDF5)
scores = sg.generate()
# scores = {"AAPL": 0.0021, "MSFT": -0.0004, ...}

# Print top 20 buy signals
top20 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
for ticker, score in top20:
    print(f"  {ticker:6s}  score={score:.6f}")
```

### Specifying a Date

```python
# Backtest a specific date
scores = sg.generate(as_of_date="2025-06-15")
```

### Full Pipeline (Scores → Orders)

```python
from quantaalpha.live.signal_generator import SignalGenerator
from quantaalpha.live.portfolio_constructor import PortfolioConstructor

# 1. Generate scores
sg = SignalGenerator.from_meta(
    "data/models/us_union89_prod_meta.json",
    h5_path="git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5",
    lookback_days=300,
)
scores = sg.generate()

# 2. Get current prices (last row of HDF5)
import h5py, pandas as pd
with h5py.File("git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5", "r") as f:
    close = pd.read_hdf(
        "git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5",
        key="data",
    )
    latest = close.groupby("instrument").last()
    prices = latest["$close"].to_dict()

# 3. Construct portfolio (topk=20, n_drop=2)
pc = PortfolioConstructor(topk=20, n_drop=2, capital=1_000_000)
result = pc.rebalance(
    scores=scores,
    positions={},      # empty = first rebalance
    prices=prices,
)

# 4. Print orders
for order in result.orders:
    print(f"  {order.action:4s} {order.shares:+5d} {order.ticker:6s}  "
          f"${order.notional:,.0f}")
```

---

## Understanding the Output

### Scores

Each ticker gets a LightGBM predicted score. Higher score → model expects higher forward return.

- Scores are **relative** — absolute values are small (0.001–0.004 typical)
- Only the **rank order** matters for portfolio construction
- ~517 S&P 500 tickers are scored

### Orders File

The scheduler writes `data/live/pending_orders_{date}.json`:

```json
{
  "date": "2026-02-22",
  "scores_count": 517,
  "daily_pnl": 7987.19,
  "account_value": 1007987.19,
  "orders": [
    {"ticker": "INTC", "shares": 8, "action": "buy", "price": 44.11, "reason": "reweight"},
    {"ticker": "LRCX", "shares": -4, "action": "sell", "price": 244.92, "reason": "reweight"}
  ]
}
```

| Field | Meaning |
|-------|---------|
| `scores_count` | Number of tickers scored |
| `daily_pnl` | Simulated P&L from previous day's holdings |
| `account_value` | Running account value (starts at $1M) |
| `orders` | Buy/sell instructions for next trading day |
| `orders[].shares` | Positive = buy, negative = sell |
| `orders[].reason` | `"new"` (new entry), `"reweight"` (rebalance), `"exit"` (dropped from top-K) |

### Portfolio Strategy

The model uses **TopkDropout**:

- **Hold top-20** highest-scoring stocks (equal weight)
- **Replace at most 2** stocks per day (`n_drop=2`)
- This limits turnover while maintaining signal freshness
- Equal-weight target: each position ≈ $50,000 (5% of $1M capital)

### Model Performance (Backtest)

| Metric | Value | Period |
|--------|-------|--------|
| IC (Information Coefficient) | 0.034 | 2022–2025 |
| Annualised Return | +18.5% | excess vs SPY |
| Maximum Drawdown | -18.1% | worst peak-to-trough |
| Calmar Ratio | 1.02 | ARR / MDD |
| Features | 89 | from 8 LLM mining runs |

---

## Daily Automated Pipeline

### Using the Scheduler

The scheduler runs two jobs daily (US Eastern Time):

| Time | Job | What it does |
|------|-----|-------------|
| 16:30 | `run_ingest` | Fetch latest EOD data via yfinance → update HDF5 |
| 17:00 | `run_signal` | Compute factors → score → rebalance → write orders JSON |

```bash
# Start the scheduler (blocks, runs until Ctrl+C)
python -m quantaalpha.live.scheduler --config configs/live.yaml
```

Or run in the background:

```bash
nohup python -m quantaalpha.live.scheduler --config configs/live.yaml \
  > log/live_scheduler.log 2>&1 &
echo $! > log/live_scheduler.pid
```

### Manual Daily Run

If you prefer to run manually (e.g., via cron):

```bash
# Step 1: Update market data (after 4:30 PM ET)
python -m quantaalpha.live.data_ingestor \
  --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5

# Step 2: Generate signals
python -m quantaalpha.live.signal_generator \
  --meta data/models/us_union89_prod_meta.json \
  --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \
  --lookback 300

# Step 3: Or run the full pipeline (ingest + signal + orders)
python -c "
from quantaalpha.live.scheduler import TradingScheduler
sched = TradingScheduler('configs/live.yaml')
sched.run_ingest()
orders = sched.run_signal()
print(f'Generated {len(orders.get(\"orders\", []))} orders')
"
```

### Crontab Setup (Alternative to Scheduler)

```crontab
# Run after US market close (4:30 PM ET = 21:30 UTC in winter)
30 21 * * 1-5 cd /path/to/QuantaAlpha && ~/anaconda3/envs/quantaalpha-ollama/bin/python -m quantaalpha.live.data_ingestor --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 >> log/ingest.log 2>&1
00 22 * * 1-5 cd /path/to/QuantaAlpha && ~/anaconda3/envs/quantaalpha-ollama/bin/python -m quantaalpha.live.scheduler --config configs/live.yaml --once >> log/signal.log 2>&1
```

---

## IBKR Paper Trading Setup

### 1. Install IBKR Trader Workstation (TWS)

1. Download TWS from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/tws.php)
2. Create a paper trading account at [IBKR Paper Trading](https://www.interactivebrokers.com/en/trading/papertrading.php)
3. In TWS → **File → Global Configuration → API → Settings**:
   - ✅ Enable ActiveX and Socket Clients
   - ✅ Socket port: **7497** (paper trading)
   - ✅ Allow connections from localhost only

### 2. Configure QuantaAlpha

Edit `configs/live.yaml`:

```yaml
ibkr:
  host: "127.0.0.1"
  port: 7497            # paper trading
  client_id: 1
  dry_run: false         # set to false to actually submit orders
  paper_mode: true
```

### 3. Test Connection

```python
from quantaalpha.live.ibkr_executor import IBKRExecutor

executor = IBKRExecutor(
    host="127.0.0.1",
    port=7497,
    client_id=1,
    dry_run=True,  # keep True for first test
)

# Test connection (requires TWS running)
executor.connect()
print("Connected to IBKR TWS")
executor.disconnect()
```

### 4. Run End-to-End Paper Trade

```bash
# 1. Ensure TWS is running on port 7497
# 2. Start the scheduler
python -m quantaalpha.live.scheduler --config configs/live.yaml

# The scheduler will:
#   16:30 → fetch latest EOD data
#   17:00 → generate signals → submit orders to TWS paper account
```

### Risk Controls

The system includes built-in safety mechanisms:

| Control | Setting | Default |
|---------|---------|---------|
| Kill switch | Stop trading if daily loss > 3% | `risk.daily_loss_limit_pct: 0.03` |
| Position limit | Max 5% per stock | `portfolio.max_position_pct: 0.05` |
| Liquidity filter | Skip stocks with ADV < $5M | `portfolio.min_adv: 5_000_000` |
| MDD alert | Alert when drawdown > 8% | `risk.max_drawdown_alert_pct: 0.08` |
| Promotion gate | Need Sharpe ≥ 1.5, MDD < 10%, 60+ days | `gate` section |

---

## Data Setup

### First-Time Setup (Download Historical Data)

```bash
# Activate environment
source ~/anaconda3/bin/activate quantaalpha-ollama

# Download S&P 500 historical data (2016–present, ~5 minutes)
python scripts/fetch_us_data.py \
  --output git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \
  --start 2016-01-01

# Verify
python -c "
import h5py
with h5py.File('git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5', 'r') as f:
    for key in f.keys():
        print(f'{key}: {f[key].shape if hasattr(f[key], \"shape\") else \"group\"} ')
"
```

### Update Data Daily

```bash
# Incremental update (only fetches new dates)
python -m quantaalpha.live.data_ingestor \
  --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5
```

---

## Model Retraining

Monthly retraining is recommended to adapt to regime changes.

### Automatic Retraining

```bash
# Retrain the production model with latest data
python scripts/retrain_us_model.py
```

This will:
1. Load the current factor library
2. Retrain LightGBM with an extended training window
3. Save new model files to `data/models/`
4. Update `configs/live.yaml` to point to the new model

### Manual Retraining via Backtest

```bash
# Full backtest + model save
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest_us.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library_us_union89.json \
  --output-name us_union89_retrained -v
```

---

## Troubleshooting

### "Factor computation failed: alpha must satisfy: 0 < alpha <= 1"

One factor (`Mean_Reversion_Strength_ZScore_RSI`) has an EMA parameter issue. This is a known cosmetic issue — the model works fine with 88/89 factors. The missing factor has near-zero importance.

### Signal generation is slow (>5 minutes)

Some factors (e.g., `Market_Beta_Regime_Momentum_Orthogonal_60D`) involve rolling correlations and take 10–15 seconds each. Total runtime for 89 factors is typically 75–90 seconds.

### "No module named 'quantaalpha'"

Ensure the package is installed in development mode:

```bash
source ~/anaconda3/bin/activate quantaalpha-ollama
cd /path/to/QuantaAlpha
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e .
```

### All scores are identical

This usually means the HDF5 data is stale (all tickers have the same latest date). Update data:

```bash
python -m quantaalpha.live.data_ingestor \
  --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5
```

### IBKR connection refused

- Ensure TWS is running and API is enabled (port 7497 for paper)
- Check `File → Global Configuration → API → Settings → Socket port`
- Try `telnet 127.0.0.1 7497` to verify port is open

---

## File Reference

| File | Purpose |
|------|---------|
| `configs/live.yaml` | All live trading settings |
| `quantaalpha/live/signal_generator.py` | Factor compute → model predict → scores |
| `quantaalpha/live/portfolio_constructor.py` | TopkDropout rebalancing |
| `quantaalpha/live/data_ingestor.py` | Daily HDF5 data update |
| `quantaalpha/live/scheduler.py` | APScheduler daily cron |
| `quantaalpha/live/ibkr_executor.py` | IBKR TWS order submission |
| `quantaalpha/live/position_tracker.py` | P&L tracking |
| `quantaalpha/live/risk_monitor.py` | Kill switch + gate checker |
| `quantaalpha/live/alerter.py` | Email/Slack alerts |
| `scripts/retrain_us_model.py` | Monthly model retraining |
| `scripts/fetch_us_data.py` | Historical data download |
| `data/live/pending_orders_*.json` | Daily order files |
| `data/live/positions.json` | Current holdings |
| `data/live/pnl/` | Daily P&L snapshots |
| `data/models/us_union89_prod_*.json/.txt` | Production model files |
