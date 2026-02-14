# AlphaAgent Experiment Hyperparameters

This document summarizes the practical hyperparameters used by QuantaAlpha experiments.

## 1) Model Configuration

Key environment variables:

```bash
REASONING_MODEL=<model_name>
CHAT_MODEL=<model_name>
OPENAI_API_KEY=<api_key>
OPENAI_BASE_URL=<base_url>
```

Common provider pattern:

- OpenAI-compatible endpoint + model pair
- Separate chat/reasoning model if needed

## 2) Planning Configuration (`planning`)

| Key | Description |
| --- | --- |
| `enabled` | Enable parallel planning |
| `num_directions` | Number of generated directions |
| `max_attempts` | Retry count for invalid planning output |
| `use_llm` | Use LLM for direction generation |
| `allow_fallback` | Fallback to built-in templates |
| `prompt_file` | Planning prompt template |

## 3) Execution Configuration (`execution`)

| Key | Description |
| --- | --- |
| `max_loops` | Max loops (if `step_n` not set) |
| `steps_per_loop` | Fixed 5-step workflow |
| `step_n` | Total steps (highest priority) |
| `use_local` | Run locally vs Docker |
| `parallel_execution` | Parallel branch execution |

## 4) Evolution Configuration (`evolution`)

| Key | Description |
| --- | --- |
| `enabled` | Enable evolution mode |
| `mutation_enabled` | Enable mutation rounds |
| `crossover_enabled` | Enable crossover rounds |
| `max_rounds` | Total rounds |
| `crossover_size` | Parent count per crossover |
| `crossover_n` | Crossover combinations per round |
| `parallel_enabled` | Parallel tasks in each phase |
| `prefer_diverse_crossover` | Prefer diverse parent mixes |

Parent selection options:

- `best`
- `random`
- `weighted`
- `weighted_inverse`
- `top_percent_plus_random`

## 5) Factor Configuration (`factor`)

| Key | Description |
| --- | --- |
| `factors_per_hypothesis` | Factors generated per hypothesis |
| `complexity.symbol_length_threshold` | Max expression length |
| `complexity.base_features_threshold` | Max base feature count |
| `complexity.free_args_ratio_threshold` | Max constant ratio |
| `duplication.enabled` | Enable duplicate checks |
| `duplication.threshold` | Duplicate subtree threshold |

## 6) Backtest Configuration (`backtest`)

| Key | Description |
| --- | --- |
| `use_docker` | Execute in Docker |
| `timeout` | Per-backtest timeout (seconds) |
| `qlib.config_name` | Qlib config template |

## 7) LLM/Logging/Path Settings

- `llm.factor_mining_timeout`
- `llm.max_retries`
- `llm.retry_delay`
- `logging.level`
- `logging.save_snapshots`
- `logging.save_trajectory_pool`

## Recommended Tuning Order

1. Set model/provider and verify API connectivity.
2. Tune `num_directions` + `max_rounds` for exploration budget.
3. Tighten factor complexity constraints to reduce noisy factors.
4. Increase backtest timeout only when necessary.
5. Enable parallel modes only after baseline stability is confirmed.
