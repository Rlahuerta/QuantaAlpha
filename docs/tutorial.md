# QuantaAlpha Tutorial (End-to-End)

This tutorial walks through setup, mining, and backtesting with practical commands.

## Step 1: Environment Setup

```bash
git clone https://github.com/QuantaAlpha/QuantaAlpha.git
cd QuantaAlpha
conda env create -f environment.ollama.yml
conda activate quantaalpha-ollama
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e .
```

## Step 2: Configure Runtime

```bash
cp configs/.env.example .env
```

Minimum required values in `.env`:

- `QLIB_DATA_DIR`
- `DATA_RESULTS_DIR`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `CHAT_MODEL`
- `REASONING_MODEL`

Recommended Ollama split:

- Cloud LLM: `OPENAI_BASE_URL=https://ollama.com/v1`, `CHAT_MODEL=minimax-m2.5`
- Local embeddings: `EMBEDDING_BASE_URL=http://localhost:11434/v1`, `EMBEDDING_MODEL=mxbai-embed-large`

## Step 3: Data Preparation

Prepare:

- Qlib market data (`cn_data`)
- HDF5 factor source (`daily_pv.h5`)

Place HDF5 files into:

- `git_ignore_folder/factor_implementation_source_data/daily_pv.h5`
- `git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5`

## Step 4: Preflight Checks

```bash
# Cloud model list
curl -sS https://ollama.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY" | head

# Local embedding models
ollama list | grep -E "mxbai-embed-large|nomic-embed-text"

# Embedding endpoint smoke test
curl -sS http://localhost:11434/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"mxbai-embed-large","input":"embedding smoke test"}' | head
```

## Step 5: Run Mining

Quick smoke run:

```bash
python -m quantaalpha.cli mine \
  --direction "price-volume factor mining" \
  --config_path configs/experiment.yaml \
  --step_n 5
```

Full run:

```bash
./run.sh "price-volume factor mining"
```

Output factor library is typically saved to:

- `data/factorlib/all_factors_library.json`

## Step 6: Run Independent Backtest

Dry run first:

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library.json \
  --dry-run -v
```

Then full backtest:

```bash
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library.json
```

## Step 7: Optional Web UI

```bash
cd frontend-v2
bash start.sh
```

Open:

- `http://localhost:3000`

## Troubleshooting

- **No factor output**: verify HDF5 source paths and Qlib data path.
- **Backtest fails**: verify `QLIB_DATA_DIR` and provider URI consistency.
- **JSON parse errors**: reduce run scope (`step_n=1` or `5`) and inspect logs.
- **Slow embeddings**: switch to `nomic-embed-text` for speed.
