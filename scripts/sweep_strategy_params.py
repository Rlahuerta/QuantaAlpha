#!/usr/bin/env python3
"""Hyperparameter sweep: topk × n_drop for the portfolio strategy.

Runs multiple backtests in parallel (--parallel N) or sequentially,
collects results, and prints a comparison table.

Usage:
    python scripts/sweep_strategy_params.py \\
        --factor-jsons data/factorlib/all_factors_library_decay5d.json \\
                       data/factorlib/all_factors_library_exp_phaseABC_decay5d.json \\
                       data/factorlib/all_factors_library_exp_phaseABC2_decay5d.json \\
        --parallel 2
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parent.parent.resolve()
PYTHON = sys.executable


def _make_temp_config(base_config: dict, topk: int, n_drop: int, output_name: str) -> Path:
    """Write a temp backtest.yaml with given topk/n_drop strategy kwargs."""
    cfg = json.loads(json.dumps(base_config))  # deep copy via json round-trip
    strategy_kwargs = cfg["backtest"]["strategy"]["kwargs"]
    strategy_kwargs["topk"] = topk
    strategy_kwargs["n_drop"] = n_drop
    # Ensure we always use equal-weight strategy (not signal-weighted which has high costs)
    cfg["backtest"]["strategy"]["class"] = "TopkDropoutStrategy"
    cfg["backtest"]["strategy"]["module_path"] = "qlib.contrib.strategy"
    # Remove weight_scheme if present
    strategy_kwargs.pop("weight_scheme", None)

    tmp = Path(tempfile.mktemp(suffix=".yaml", prefix=f"sweep_{output_name}_"))
    with open(tmp, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    return tmp


def _run_one(args_tuple):
    topk, n_drop, factor_jsons, base_config_path, results_dir = args_tuple
    output_name = f"sweep_topk{topk}_ndrop{n_drop}"
    result_file = results_dir / f"{output_name}_backtest_metrics.json"

    if result_file.exists():
        print(f"[{output_name}] Already exists, loading cached result.")
        with open(result_file) as f:
            return output_name, topk, n_drop, json.load(f)

    # Load base config
    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    tmp_config = _make_temp_config(base_config, topk, n_drop, output_name)

    cmd = [
        PYTHON, "-m", "quantaalpha.backtest.run_backtest",
        "-c", str(tmp_config),
        "--factor-source", "combined",
        "--output-name", output_name,
    ]
    for fj in factor_jsons:
        cmd += ["--factor-json", str(fj)]

    t0 = time.time()
    print(f"[{output_name}] Starting... (topk={topk}, n_drop={n_drop})")
    try:
        proc = subprocess.run(
            cmd, cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=1200
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            print(f"[{output_name}] FAILED after {elapsed:.0f}s")
            print(proc.stderr[-2000:])
            return output_name, topk, n_drop, None

        if not result_file.exists():
            print(f"[{output_name}] Done ({elapsed:.0f}s) but no output file found")
            return output_name, topk, n_drop, None

        with open(result_file) as f:
            metrics = json.load(f)
        print(f"[{output_name}] Done ({elapsed:.0f}s): "
              f"Net ARR={metrics['metrics']['annualized_return']:.1%}, "
              f"MDD={metrics['metrics']['max_drawdown']:.1%}, "
              f"Calmar={metrics['metrics']['calmar_ratio']:.1f}")
        return output_name, topk, n_drop, metrics
    except subprocess.TimeoutExpired:
        print(f"[{output_name}] TIMEOUT after 1200s")
        return output_name, topk, n_drop, None
    finally:
        tmp_config.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-jsons", nargs="+", required=True,
                        help="Factor library JSON files (combined mode)")
    parser.add_argument("--base-config", default="configs/backtest.yaml",
                        help="Base backtest config YAML")
    parser.add_argument("--topk-values", nargs="+", type=int, default=[20, 30, 50],
                        help="topk values to sweep")
    parser.add_argument("--ndrop-values", nargs="+", type=int, default=[1, 2, 5],
                        help="n_drop values to sweep")
    parser.add_argument("--parallel", type=int, default=2,
                        help="Max parallel backtests")
    parser.add_argument("--results-dir", default="data/results/backtest_v2_results",
                        help="Where backtest results are saved")
    args = parser.parse_args()

    base_config_path = (REPO_ROOT / args.base_config).resolve()
    results_dir = (REPO_ROOT / args.results_dir).resolve()
    factor_jsons = [(REPO_ROOT / fj).resolve() for fj in args.factor_jsons]

    # Build grid
    grid = [
        (topk, n_drop, factor_jsons, base_config_path, results_dir)
        for topk in args.topk_values
        for n_drop in args.ndrop_values
        if n_drop <= topk  # n_drop must be ≤ topk
    ]

    print(f"Sweep: {len(grid)} combinations, parallel={args.parallel}")
    print(f"Grid: topk={args.topk_values} × n_drop={args.ndrop_values}\n")

    all_results = []
    with ProcessPoolExecutor(max_workers=args.parallel) as ex:
        futures = {ex.submit(_run_one, g): g for g in grid}
        for fut in as_completed(futures):
            all_results.append(fut.result())

    # Sort and print comparison table
    all_results.sort(key=lambda r: (r[1], r[2]))  # sort by topk, n_drop

    print("\n" + "=" * 90)
    print(f"{'Config':<25} {'Net ARR':>9} {'Gross ARR':>10} {'Cost/yr':>8} {'MDD':>7} {'Calmar':>8} {'IR':>7}")
    print("=" * 90)
    for name, topk, n_drop, metrics in all_results:
        if metrics is None:
            print(f"  topk={topk:2d} n_drop={n_drop:2d}  {'FAILED':>9}")
            continue
        m = metrics["metrics"]
        print(
            f"  topk={topk:2d} n_drop={n_drop:2d}      "
            f"{m['annualized_return']:>9.1%} "
            f"{m['annualized_return_gross']:>10.1%} "
            f"{m['annual_cost_rate']:>8.2%} "
            f"{m['max_drawdown']:>7.1%} "
            f"{m['calmar_ratio']:>8.2f} "
            f"{m['information_ratio']:>7.2f}"
        )

    # Find best by calmar
    valid = [(n, t, d, m) for n, t, d, m in all_results if m is not None]
    if valid:
        best = max(valid, key=lambda r: r[3]["metrics"]["calmar_ratio"])
        print(f"\nBest Calmar: topk={best[1]}, n_drop={best[2]}")
        best_arr = max(valid, key=lambda r: r[3]["metrics"]["annualized_return"])
        print(f"Best Net ARR: topk={best_arr[1]}, n_drop={best_arr[2]}")


if __name__ == "__main__":
    main()
