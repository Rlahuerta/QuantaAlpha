# QuantaAlpha Tutorial (End-to-End)

This tutorial explains the method in the paper (`2602.07085v1.pdf`) and shows how to run the full workflow in this repository.

## 1) Repository and Paper Overview

QuantaAlpha is an LLM-driven alpha mining framework designed for noisy, non-stationary financial markets.

According to the paper, the main idea is to treat each full mining run as a **trajectory** and improve mining quality through **trajectory-level self-evolution**:

- **Diversified planning initialization** to start from multiple complementary research directions.
- **Controllable factor realization** from hypothesis -> symbolic expression -> executable code.
- **Consistency and quality constraints** to reduce semantic drift, excessive complexity, and redundancy.
- **Evolutionary refinement** with mutation (targeted repair) and crossover (reuse of strong trajectory segments).

Reported results in the paper include strong performance on CSI 300 and effective transfer to CSI 500 / S&P 500 under distribution shift.

## 2) Theoretical Background (Intuition)

The paper formulates alpha mining as maximizing predictive utility with regularization:

$$
f^* = \arg\max_{f \in \mathcal{F}} \mathcal{L}(f(X), y) - \lambda \mathcal{R}(f)
$$

Then it optimizes a trajectory-generation policy:

$$
\pi^* = \arg\max_{\pi} \mathbb{E}_{\tau \sim \pi}[R(\tau)]
$$

where each trajectory \(\tau\) is a full research run (hypothesis generation -> factor construction -> evaluation).

In practical terms, this gives three useful properties:

1. **Broader exploration**: diversified initialization avoids getting stuck near one seed idea.
2. **Controllable refinement**: mutation repairs specific failing trajectory steps without rewriting everything.
3. **Experience reuse**: crossover recombines high-reward trajectory parts to accelerate convergence.

The framework also enforces complexity/redundancy controls to limit overfit or crowded factor designs.

## 3) Typical Applications of the Method

You can use this repo in three common modes:

- **Single-market alpha discovery**: mine interpretable factors on CSI 300 with a closed-loop workflow.
- **Cross-market transfer testing**: evaluate whether mined factors generalize to CSI 500 or S&P 500.
- **Research productivity workflow**: automate repetitive hypothesis -> implementation -> backtest iterations while keeping auditable trajectories.

## 4) Practical Workflow

### Step 1: Environment Setup

```bash
git clone https://github.com/QuantaAlpha/QuantaAlpha.git
cd QuantaAlpha
conda env create -f environment.ollama.yml
conda activate quantaalpha-ollama
SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 pip install -e .
```

### Step 2: Configure Runtime

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

### Step 3: Data Preparation

Prepare:

- Qlib market data (`cn_data`)
- HDF5 factor source (`daily_pv.h5`)

Place HDF5 files into:

- `git_ignore_folder/factor_implementation_source_data/daily_pv.h5`
- `git_ignore_folder/factor_implementation_source_data_debug/daily_pv.h5`

### Step 4: Preflight Checks

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

### Step 5: Run Mining

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

### Step 6: Run Independent Backtest

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

### Step 7: Optional Web UI

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
