# Backtest Module (V2)

This module runs independent backtests for baseline and custom factor libraries.

## Supported Factor Sources

- `alpha158`
- `alpha158_20`
- `alpha360`
- `custom` (JSON factor library)
- `combined` (baseline + custom)

## Quick Start

### Custom factors

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json /path/to/factors.json
```

### Combined mode

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source combined \
  --factor-json /path/to/factors.json
```

### Dry run (load only)

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json /path/to/factors.json \
  --dry-run -v
```

## Config Highlights

- `data.provider_uri`, market, date ranges
- `dataset.segments` (train/valid/test)
- `model.params` (LightGBM)
- `backtest.strategy` and transaction costs

## Outputs

- Backtest metrics JSON
- Cumulative excess return CSV
- Logs for factor loading and model execution
