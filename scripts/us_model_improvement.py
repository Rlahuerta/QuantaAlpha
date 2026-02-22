#!/usr/bin/env python3
"""
scripts/us_model_improvement.py — US model improvement experiments.

Runs three experiments to improve IC from the current 0.018:
1. Feature importance pruning (drop low-importance features)
2. Hyperparameter sweep (grid search on key LightGBM params)
3. Label construction experiments (different forward return horizons)

Usage:
    python scripts/us_model_improvement.py --experiment all
    python scripts/us_model_improvement.py --experiment feature_selection
    python scripts/us_model_improvement.py --experiment hyperparam_sweep
    python scripts/us_model_improvement.py --experiment label_experiment
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FACTOR_JSON = "data/factorlib/all_factors_library_exp_us_combined.json"
BASE_CONFIG = "configs/backtest_us.yaml"
OUTPUT_DIR = Path("data/results/us_improvement")


def _load_config() -> dict:
    return yaml.safe_load(Path(BASE_CONFIG).read_text())


def _run_backtest(cfg: dict, factor_json: str, run_name: str) -> dict | None:
    """Run a backtest with the given config and return metrics."""
    from quantaalpha.backtest.runner import BacktestRunner

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(cfg, tmp, default_flow_style=False, allow_unicode=True)
        tmp_path = tmp.name

    try:
        runner = BacktestRunner(tmp_path)
        metrics = runner.run(
            factor_json=[factor_json],
            factor_source="custom",
            output_name=run_name,
        )

        # Runner.run() returns the metrics dict directly.
        # Also check saved metrics file as fallback.
        if not metrics:
            results_dir = Path(cfg.get("experiment", {}).get("output_dir", "data/results/backtest_v2_results"))
            metrics_file = results_dir / f"{run_name}_backtest_metrics.json"
            if metrics_file.exists():
                saved = json.loads(metrics_file.read_text())
                metrics = saved.get("metrics", saved)

        return metrics
    except Exception as e:
        log.error("Backtest %s failed: %s", run_name, e, exc_info=True)
        return None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _extract_metrics_from_log(run_name: str) -> dict:
    """Parse metrics from the backtest log output."""
    # BacktestRunner prints metrics to log; also saved to model meta
    meta_dir = Path("data/models")
    for f in sorted(meta_dir.glob(f"{run_name}*_meta.json")):
        try:
            return json.loads(f.read_text())
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# Experiment 1: Feature importance pruning
# ---------------------------------------------------------------------------

def run_feature_selection(cfg: dict) -> dict:
    """Analyze feature importances and retrain with top features only."""
    import numpy as np

    log.info("=" * 70)
    log.info("EXPERIMENT 1: Feature Importance Pruning")
    log.info("=" * 70)

    # First, load the current model and get feature importances
    meta_path = Path("data/models/us_retrain_20260222_112859_meta.json")
    lgbm_path = Path("data/models/us_retrain_20260222_112859_lgbm.txt")
    feat_path = Path("data/models/us_retrain_20260222_112859_feature_cols.json")

    if not lgbm_path.exists():
        log.warning("No existing model found, training baseline first")
        return {"error": "no model"}

    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(lgbm_path))
    importances = booster.feature_importance(importance_type="gain")
    feature_names = json.loads(feat_path.read_text())

    # Rank features by importance
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    log.info("Feature importances (top 20):")
    for name, imp in ranked[:20]:
        log.info("  %6.1f  %s", imp, name)

    total_imp = sum(importances)
    zero_count = sum(1 for _, imp in ranked if imp == 0)
    log.info("Total features: %d, Zero-importance: %d", len(ranked), zero_count)

    results = {"baseline_features": len(feature_names), "rankings": []}

    # Try different pruning levels: top 75%, top 50%, top 25%, top 100 features
    thresholds = [
        ("top75pct", int(len(ranked) * 0.75)),
        ("top50pct", int(len(ranked) * 0.50)),
        ("top25pct", int(len(ranked) * 0.25)),
        ("top100", min(100, len(ranked))),
        ("top50", min(50, len(ranked))),
        ("nonzero", len(ranked) - zero_count),
    ]

    for label, n_features in thresholds:
        if n_features < 10:
            continue
        top_features = [name for name, _ in ranked[:n_features]]

        # Create a pruned factor library with only selected features
        pruned_lib = _create_pruned_library(FACTOR_JSON, top_features)
        if pruned_lib is None:
            continue

        run_name = f"us_featsel_{label}_{n_features}"
        log.info("Running %s with %d features...", run_name, n_features)

        exp_cfg = copy.deepcopy(cfg)
        metrics = _run_backtest(exp_cfg, pruned_lib, run_name)

        result = {
            "label": label,
            "n_features": n_features,
            "metrics": metrics,
        }
        results["rankings"].append(result)
        log.info("  %s → %s", label, json.dumps(metrics or {}, indent=2)[:200])

    return results


def _create_pruned_library(factor_json: str, keep_features: list[str]) -> str | None:
    """Create a pruned factor library JSON with only specified features."""
    lib = json.loads(Path(factor_json).read_text())
    factors = lib.get("factors", {})

    # Keep only factors whose names are in keep_features
    pruned = {}
    for fname, fdata in factors.items():
        if fname in keep_features:
            pruned[fname] = fdata

    if not pruned:
        log.warning("No matching factors found for pruning")
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"pruned_{len(pruned)}_factors.json"
    pruned_lib = {
        "metadata": {
            "version": "1.0",
            "total_factors": len(pruned),
            "pruned_from": len(factors),
        },
        "factors": pruned,
    }
    out_path.write_text(json.dumps(pruned_lib, indent=2))
    return str(out_path)


# ---------------------------------------------------------------------------
# Experiment 2: Hyperparameter sweep
# ---------------------------------------------------------------------------

def run_hyperparam_sweep(cfg: dict) -> dict:
    """Grid search over key LightGBM hyperparameters."""
    log.info("=" * 70)
    log.info("EXPERIMENT 2: Hyperparameter Sweep")
    log.info("=" * 70)

    # Key parameters to sweep
    param_grid = [
        # Lower learning rates with more rounds
        {"learning_rate": 0.01, "num_boost_round": 1000, "num_leaves": 64, "max_depth": 6},
        {"learning_rate": 0.01, "num_boost_round": 1000, "num_leaves": 128, "max_depth": 7},
        {"learning_rate": 0.05, "num_boost_round": 500, "num_leaves": 64, "max_depth": 6},
        {"learning_rate": 0.05, "num_boost_round": 500, "num_leaves": 128, "max_depth": 7},
        {"learning_rate": 0.05, "num_boost_round": 500, "num_leaves": 210, "max_depth": 8},
        # Higher regularization
        {"learning_rate": 0.05, "num_boost_round": 500, "num_leaves": 64, "max_depth": 6,
         "lambda_l1": 500, "lambda_l2": 1000, "min_child_samples": 200},
        # Less aggressive subsampling
        {"learning_rate": 0.01, "num_boost_round": 1000, "num_leaves": 32, "max_depth": 5,
         "colsample_bytree": 0.5, "subsample": 0.6},
        # Very simple model (anti-overfit)
        {"learning_rate": 0.01, "num_boost_round": 2000, "num_leaves": 16, "max_depth": 4,
         "min_child_samples": 500, "lambda_l1": 1000, "lambda_l2": 2000},
    ]

    results = []
    for i, params in enumerate(param_grid):
        exp_cfg = copy.deepcopy(cfg)
        exp_cfg.setdefault("model", {}).setdefault("params", {})
        exp_cfg["model"]["params"].update(params)

        run_name = f"us_hp_{i:02d}_lr{params['learning_rate']}_nl{params.get('num_leaves', 'def')}"
        log.info("Running %s with params: %s", run_name, params)

        metrics = _run_backtest(exp_cfg, FACTOR_JSON, run_name)
        result = {"index": i, "params": params, "metrics": metrics, "run_name": run_name}
        results.append(result)
        log.info("  HP %d → %s", i, json.dumps(metrics or {}, indent=2)[:200])

    return {"experiments": results}


# ---------------------------------------------------------------------------
# Experiment 3: Label construction
# ---------------------------------------------------------------------------

def run_label_experiment(cfg: dict) -> dict:
    """Test different label constructions."""
    log.info("=" * 70)
    log.info("EXPERIMENT 3: Label Construction")
    log.info("=" * 70)

    labels = {
        # Pure 1-day return
        "ret_1d": "Ref($close, -2) / Ref($close, -1) - 1",
        # Pure 5-day return
        "ret_5d": "Ref($close, -6) / Ref($close, -1) - 1",
        # Pure 20-day return (momentum)
        "ret_20d": "Ref($close, -21) / Ref($close, -1) - 1",
        # Blend: 70% short + 30% long (emphasize short-term)
        "blend_70_30": "(Ref($close, -2) / Ref($close, -1) - 1) * 0.7 + (Ref($close, -6) / Ref($close, -1) - 1) * 0.3",
        # Original multi-period (baseline)
        "blend_50_30_20": "(Ref($close, -2) / Ref($close, -1) - 1) * 0.5 + (Ref($close, -6) / Ref($close, -1) - 1) * 0.3 + (Ref($close, -21) / Ref($close, -1) - 1) * 0.2",
        # 3-day return (alternative short horizon)
        "ret_3d": "Ref($close, -4) / Ref($close, -1) - 1",
    }

    results = []
    for label_name, label_expr in labels.items():
        exp_cfg = copy.deepcopy(cfg)
        exp_cfg["dataset"]["label"] = label_expr

        # Simpler model for label testing (faster)
        exp_cfg.setdefault("model", {}).setdefault("params", {})
        exp_cfg["model"]["params"]["learning_rate"] = 0.05
        exp_cfg["model"]["params"]["num_boost_round"] = 500
        exp_cfg["model"]["params"]["num_leaves"] = 64
        exp_cfg["model"]["params"]["max_depth"] = 6

        run_name = f"us_label_{label_name}"
        log.info("Running %s with label: %s", run_name, label_expr[:80])

        metrics = _run_backtest(exp_cfg, FACTOR_JSON, run_name)
        result = {"label_name": label_name, "label_expr": label_expr, "metrics": metrics}
        results.append(result)
        log.info("  %s → %s", label_name, json.dumps(metrics or {}, indent=2)[:200])

    return {"experiments": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["all", "feature_selection", "hyperparam_sweep", "label_experiment"])
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()
    all_results = {}

    experiments = {
        "feature_selection": run_feature_selection,
        "label_experiment": run_label_experiment,
        "hyperparam_sweep": run_hyperparam_sweep,
    }

    if args.experiment == "all":
        run_exps = list(experiments.keys())
    else:
        run_exps = [args.experiment]

    for exp_name in run_exps:
        log.info("\n" + "=" * 70)
        log.info("Starting experiment: %s", exp_name)
        log.info("=" * 70)
        try:
            all_results[exp_name] = experiments[exp_name](cfg)
        except Exception as e:
            log.error("Experiment %s failed: %s", exp_name, e, exc_info=True)
            all_results[exp_name] = {"error": str(e)}

    # Save all results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = OUTPUT_DIR / f"improvement_results_{ts}.json"
    results_file.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("\nAll results saved to %s", results_file)

    # Print summary
    print("\n" + "=" * 70)
    print("IMPROVEMENT EXPERIMENT SUMMARY")
    print("=" * 70)
    for exp_name, exp_results in all_results.items():
        print(f"\n--- {exp_name} ---")
        if "error" in exp_results:
            print(f"  ERROR: {exp_results['error']}")
        elif "experiments" in exp_results:
            for r in exp_results["experiments"]:
                m = r.get("metrics") or {}
                ic = m.get("IC", "N/A")
                arr = m.get("excess_return_annualized", m.get("ARR", "N/A"))
                name = r.get("label_name", r.get("run_name", r.get("index", "")))
                print(f"  {name}: IC={ic}, ARR={arr}")
        elif "rankings" in exp_results:
            for r in exp_results["rankings"]:
                m = r.get("metrics") or {}
                ic = m.get("IC", "N/A")
                print(f"  {r['label']} ({r['n_features']}): IC={ic}")


if __name__ == "__main__":
    main()
