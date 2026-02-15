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

## 4) How the Algorithm Works (Clear Flow)

Use this as a mental model of the full loop:

1. **Input research direction**  
   You provide a natural-language direction (example: "price-volume factor mining with short-term reversal focus").

2. **Diversified planning initialization**  
   The system generates multiple complementary hypotheses so search starts from different areas, not one local neighborhood.

3. **Controllable factor construction**  
   For each hypothesis, QuantaAlpha builds:
   - semantic description -> symbolic factor expression -> AST structure -> executable code.
   This keeps implementation aligned with the original hypothesis.

4. **Constraint and consistency gates**  
   The system checks:
   - hypothesis/expression/code consistency,
   - complexity limits,
   - redundancy against existing factors.
   Failing candidates are rewritten.

5. **Backtest evaluation**  
   Valid factors are evaluated by predictive metrics and return/risk metrics; each full run is stored as a trajectory.

6. **Trajectory-level self-evolution**  
   QuantaAlpha improves trajectories by:
   - **Mutation**: fix weak steps inside a trajectory,
   - **Crossover**: recombine strong segments from high-performing trajectories.
   This loop repeats to improve factor quality over rounds.

7. **Final factor pool and strategy testing**  
   Mined factors are saved into factor-library JSON files, then converted into investable strategies through independent backtesting.

## 5) Practical Workflow

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

## 6) How to Use the Frontend (GUI)

After opening `http://localhost:3000`, use the GUI in this order:

1. **Settings page (first stop)**  
   Configure API endpoint/key/model, data paths, default experiment parameters, and mining directions.  
   Save settings before starting tasks.

2. **Home page (start mining)**  
   Enter a natural-language mining request in the bottom input (or enable custom mining direction from settings), then submit.

3. **Mining dashboard (monitor execution)**  
   Track phase progress, logs, and live metrics while the run is executing.  
   You can stop the task if needed.

4. **Factor Library page (review outputs)**  
   Inspect generated factors, filter by quality tier, open factor details, and compare metrics before selecting candidates.

5. **Backtest page (build strategy candidates)**  
   Choose a factor library JSON, select factor source mode:
   - `custom`: only mined factors
   - `combined`: mined factors + Alpha158 baseline  
   Run backtest and compare return/risk outputs.

6. **Iterate and version**  
   Run multiple GUI experiments with different direction prompts and keep library files versioned so comparisons stay reproducible.

## 7) How to Find or Create an Investment Strategy (Detailed)

Use this pipeline to go from idea -> factor library -> strategy candidate:

### 7.1 Define strategy requirements first

Before mining, write a one-page mandate with:

- Market/universe (for this repo, start with CSI 300 baseline setup).
- Holding horizon and rebalance frequency.
- Turnover tolerance and transaction-cost sensitivity.
- Risk constraints (max drawdown tolerance, volatility tolerance, concentration limits).
- Success criteria (for example: stable RankIC + acceptable drawdown, not just high return).

This prevents overfitting to one attractive metric.

### 7.2 Convert an investment thesis into testable mining directions

Convert a broad thesis into specific prompts:

- Thesis: "short-term reversal after volume shock"
- Mechanism: "mean reversion after temporary liquidity imbalance"
- Mining prompt: "mine short-horizon reversal factors with volume-confirmation and volatility filter"

Run several variants (different horizons/filters), not only one prompt.

### 7.3 Run an experiment batch (A/B style)

Run a quick screening pass first:

```bash
python -m quantaalpha.cli mine \
  --direction "short-term reversal with volume confirmation" \
  --config_path configs/experiment.yaml \
  --step_n 5
```

After each run, archive the produced library so runs stay comparable:

```bash
cp data/factorlib/all_factors_library.json data/factorlib/all_factors_library_reversal_v1.json
```

Then repeat for other hypotheses (`trend`, `quality`, `multi-horizon momentum`, etc.).

### 7.4 Build a factor shortlist with a strict checklist

When reading factor libraries, prioritize:

1. **Signal quality stability**: IC/RankIC consistency across rounds, not one lucky spike.
2. **Risk-adjusted quality**: evaluate ARR together with MDD and information ratio.
3. **Redundancy control**: avoid selecting many near-duplicate factors.
4. **Interpretability**: keep factors with understandable economic mechanisms when possible.

### 7.5 Convert shortlisted factors into strategy candidates

For each shortlisted library, run both strategy forms:

```bash
# Candidate as pure mined-factor strategy
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source custom \
  --factor-json data/factorlib/all_factors_library_reversal_v1.json

# Candidate as blended strategy (mined + Alpha158 baseline)
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest.yaml \
  --factor-source combined \
  --factor-json data/factorlib/all_factors_library_reversal_v1.json
```

Treat these as two different portfolio hypotheses and compare them directly.

### 7.6 Perform robustness checks before promotion

A candidate is stronger if it survives:

- Time robustness: different train/validation/test windows.
- Market robustness: transfer tests (e.g., CSI 500 / S&P 500 style checks).
- Cost robustness: performance still acceptable under realistic frictions.
- Concentration robustness: returns are not dominated by a very small subset of factors.

### 7.7 Use a deployment ladder

Promote candidates progressively:

1. Offline backtest only.
2. Paper/simulated trading.
3. Small-capital pilot with strict limits.
4. Scale gradually only if live behavior matches research expectations.

Maintain a live monitoring sheet: return, drawdown, turnover, exposure drift, and factor decay warnings.

> Important: this repository provides a quantitative research workflow, not personalized financial advice. Always perform independent risk, compliance, and execution review before real trading.

## Troubleshooting

- **No factor output**: verify HDF5 source paths and Qlib data path.
- **Backtest fails**: verify `QLIB_DATA_DIR` and provider URI consistency.
- **JSON parse errors**: reduce run scope (`step_n=1` or `5`) and inspect logs.
- **Slow embeddings**: switch to `nomic-embed-text` for speed.
