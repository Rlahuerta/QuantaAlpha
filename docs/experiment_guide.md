# Experiment Guide

This guide explains how to run, debug, and reproduce QuantaAlpha experiments.

## 1. Prerequisites

- Python 3.10+
- Installed package (`pip install -e .`)
- Valid `.env` configuration
- Qlib data available in `QLIB_DATA_DIR`

## 2. Core Config Files

- `configs/experiment.yaml`: factor mining pipeline settings
- `configs/backtest.yaml`: independent backtest settings
- `configs/.env.example`: environment variable template

## 3. Execution Flow

QuantaAlpha mining follows a 5-step loop:

1. Propose hypothesis
2. Construct factor expression
3. Calculate factor values
4. Backtest
5. Feedback

When evolution is enabled, rounds are organized as:

- Original round
- Mutation round
- Crossover round
- Repeat until `max_rounds`

## 4. Run Commands

### Full mining

```bash
./run.sh "price-volume factor mining"
```

### Controlled steps

```bash
python -m quantaalpha.cli mine \
  --direction "momentum + liquidity" \
  --config_path configs/experiment.yaml \
  --step_n 5
```

### Independent backtest (custom factors)

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library.json
```

## 5. Reproducibility Checklist

- Pin model/provider in `.env`
- Save experiment config snapshot
- Keep `DATA_RESULTS_DIR` stable
- Use deterministic settings where available (`seed`, fixed model version)

## 6. Debugging Checklist

### LLM issues

- Verify `OPENAI_BASE_URL`, `OPENAI_API_KEY`
- Confirm model is available from provider endpoint
- Check JSON-mode responses for strict parsability

### Data issues

- Ensure `QLIB_DATA_DIR` has `calendars/`, `features/`, `instruments/`
- Ensure factor source HDF5 exists (`daily_pv.h5`)

### Runtime issues

- Reduce run to `step_n=1` or `step_n=5`
- Disable evolution for smoke testing
- Review generated workspace logs and backtest stdout

## 7. Output Artifacts

Typical outputs:

- Factor library JSON: `data/factorlib/all_factors_library*.json`
- Workspace and cache under `DATA_RESULTS_DIR`
- Branch/evolution logs under `log/`

## 8. Performance Tuning Tips

- Start with 1 direction + 1 loop, then scale up.
- Increase `num_directions` first, then `max_rounds`.
- Keep complexity gates enabled to reduce unstable factors.
- Use local embeddings + cloud chat split to optimize cost/performance.
