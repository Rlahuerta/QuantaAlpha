# QuantaAlpha Paper Portfolio — Daily Operations Guide

## How the Strategy Works

This is a **daily rebalanced long-only portfolio** of 10 S&P 500 stocks.

### The Concept

Every day after market close, the model:
1. Scores all ~517 S&P 500 stocks using 89 alpha factors
2. Picks the **top 10 highest-scoring** stocks
3. Allocates **equal weight** (~$100K each with 1M capital)
4. Generates **buy/sell orders** for the next trading day

### Key Rule: TopkDropout (topk=10, n_drop=1)
- **Hold 10 stocks** at all times
- **Replace at most 1 stock per day** — this limits turnover and trading costs
- If a held stock drops out of the top-10, it's replaced by the highest-scoring stock not already held
- Positions that stay in the portfolio are **reweighted** to maintain equal weight

---

## Daily Workflow

### 1. Check Today's Orders (after 5:00 PM ET)

```bash
cat data/live/pending_orders_$(date +%Y-%m-%d).json | python3 -m json.tool
```

The orders file tells you **exactly what to do tomorrow morning**:

```json
{
  "date": "2026-02-23",
  "orders": [
    {"ticker": "CRWD", "shares": 141, "action": "buy", "price": 352.37},
    {"ticker": "GEN",  "shares": -50, "action": "sell", "price": 21.57}
  ]
}
```

### 2. Execute Orders (at market open, ~9:30 AM ET)

| Action | What to do |
|--------|-----------|
| `"buy"` with positive shares | **Buy** that many shares at market open |
| `"sell"` with negative shares | **Sell** that many shares at market open |
| `"buy"` with `"reason": "reweight"` | Adjust position size (buy more shares) |
| `"sell"` with `"reason": "reweight"` | Trim position size (sell some shares) |

**Order type: Use MARKET orders at open** (MOO — Market on Open) for best replication of the model's backtested performance. The backtest assumes execution at next-day open price.

### 3. What About Target Prices?

**There are no target prices.** This is NOT a price-target strategy. Instead:

- **Entry**: Buy at market open (any price — the model predicts relative rank, not price levels)
- **Exit**: Only when the model tells you to sell (next day's orders file says `"sell"`)
- **Hold period**: Typically days to weeks (a stock stays until it drops out of top-10)
- **No stop-losses needed**: The model naturally rotates out underperformers via daily re-ranking

### 4. Check Portfolio Status

```bash
# Current holdings
cat data/live/positions.json

# P&L history
ls data/live/pnl/

# Scheduler log
tail -20 log/paper_trading.log
```

---

## Today's Portfolio (Feb 23, 2026)

This is the **initial build** — all 10 positions are new buys.

| # | Ticker | Company | Shares | ~Notional | Weight |
|---|--------|---------|-------:|----------:|-------:|
| 1 | CRWD | CrowdStrike | 141 | $49,684 | 10% |
| 2 | ERIE | Erie Indemnity | 188 | $49,809 | 10% |
| 3 | ADP | Automatic Data Processing | 241 | $49,942 | 10% |
| 4 | GEN | Gen Digital | 2,318 | $49,999 | 10% |
| 5 | GDDY | GoDaddy | 570 | $49,915 | 10% |
| 6 | IFF | Intl Flavors & Fragrances | 617 | $49,971 | 10% |
| 7 | PSA | Public Storage | 163 | $49,759 | 10% |
| 8 | FIS | Fidelity National Info Svcs | 1,045 | $49,982 | 10% |
| 9 | PANW | Palo Alto Networks | 343 | $49,955 | 10% |
| 10 | FOX | Fox Corporation | 980 | $49,960 | 10% |

**Total deployed: ~$500K** (equal weight across 10 positions)

---

## What Happens Each Day — Example Scenarios

### Scenario A: No changes (most common)
```
Orders: []   (empty — all 10 stocks still in top-10)
Action: Do nothing. Hold all positions.
```

### Scenario B: One stock swapped
```
Orders:
  SELL  GEN   -2318 shares  (dropped out of top-10)
  BUY   MSFT  +135 shares   (new entry to top-10)
Action: Sell all GEN shares at open, buy 135 MSFT at open.
```

### Scenario C: Reweight only
```
Orders:
  BUY   CRWD   +5 shares   reason: "reweight"
  SELL  FOX   -20 shares   reason: "reweight"
Action: Buy 5 more CRWD, sell 20 FOX. Keeps weights equal as prices change.
```

---

## Risk Management

| Rule | Setting | What happens |
|------|---------|-------------|
| Kill switch | Daily loss > 3% | System halts, no new orders generated |
| Max position | 10% per stock | Equal weight with 10 stocks = exactly 10% |
| Liquidity filter | ADV > $5M | Won't pick illiquid stocks |
| Max drawdown alert | MDD > 8% | Warning in logs |

---

## Key Things to Remember

1. **Execute orders at market open** — the model is backtested on open-to-open returns
2. **Don't add stop-losses** — the model handles exits via daily re-ranking
3. **Don't skip days** — consistency matters; the TopkDropout logic assumes daily execution
4. **Most days = no trades** — with n_drop=1, at most 1 stock rotates per day
5. **Reweight orders are small** — just adjusting share counts to maintain equal weight
6. **The model has NO opinion on intraday prices** — it only predicts relative next-day performance

---

## Manual Signal Generation (if scheduler is down)

```bash
cd /path/to/QuantaAlpha
source ~/anaconda3/bin/activate quantaalpha-ollama

# Step 1: Update data (after 4:30 PM ET)
python -m quantaalpha.live.data_ingestor \
  --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5

# Step 2: Generate signals
python -c "
from quantaalpha.live.scheduler import TradingScheduler
s = TradingScheduler('configs/live.yaml')
s.run_signal()
print('Orders written to data/live/pending_orders_*.json')
"
```

---

## Performance Expectations (from walk-forward backtest)

| Metric | Value | Note |
|--------|-------|------|
| Avg annual return | +26.3% total (+10.3% vs SPY) | 5-year walk-forward |
| Beat SPY | 5/5 years (100%) | Every single year 2021-2025 |
| Beat NASDAQ | 4/5 years (80%) | Only lost in 2023 AI rally |
| Worst year | +4.2% excess (2023) | Still positive |
| Best year | +16.7% excess (2025) | |
| Typical turnover | ~1 trade/week | Very low with n_drop=1 |

**Important**: Past backtest performance does not guarantee future results. This is a paper trading trial to validate live performance.
