import json

import yaml

from quantaalpha.backtest.runner import BacktestRunner


def _write_runner_config(tmp_path):
    config = {
        "data": {
            "start_time": "2021-01-01",
            "end_time": "2021-12-31",
            "market": "csi300",
        },
        "dataset": {
            "segments": {
                "test": ["2021-10-01", "2021-12-31"],
            }
        },
        "backtest": {
            "backtest": {
                "start_time": "2021-10-01",
                "end_time": "2021-12-31",
                "benchmark": "SH000300",
            }
        },
        "experiment": {
            "output_dir": str(tmp_path / "bt_out"),
            "output_metrics_file": "metrics.json",
        },
    }
    path = tmp_path / "runner_output_config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_save_results_writes_metrics_and_summary(tmp_path):
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))
    metrics = {
        "IC": 0.1,
        "ICIR": 0.2,
        "Rank IC": 0.3,
        "Rank ICIR": 0.4,
        "annualized_return": 0.25,
        "information_ratio": 1.2,
        "max_drawdown": -0.1,
    }

    runner._save_results(
        metrics=metrics,
        exp_name="exp_a",
        factor_source="custom",
        num_factors=5,
        elapsed=12.5,
        output_name="run_a",
    )
    runner._save_results(
        metrics=metrics,
        exp_name="exp_b",
        factor_source="custom",
        num_factors=6,
        elapsed=13.5,
        output_name=None,
    )

    output_dir = tmp_path / "bt_out"
    run_a_path = output_dir / "run_a_backtest_metrics.json"
    exp_b_path = output_dir / "metrics.json"
    summary_path = output_dir / "batch_summary.json"

    assert run_a_path.exists()
    assert exp_b_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(summary) == 2
    assert summary[0]["name"] == "run_a"
    assert summary[0]["calmar_ratio"] == 2.5
    assert summary[1]["name"] == "exp_b"


def test_print_results_handles_missing_metrics(capsys, tmp_path):
    runner = BacktestRunner(str(_write_runner_config(tmp_path)))
    runner._print_results(metrics={}, total_time=1.23)

    out = capsys.readouterr().out
    assert "Backtest Results" in out
    assert "IC: N/A" in out
    assert "Total time: 1.2s" in out
