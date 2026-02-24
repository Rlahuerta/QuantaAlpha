import sys
import types
from pathlib import Path

import pandas as pd
import pytest
import yaml

from quantaalpha.backtest.runner import BacktestRunner


def _write_runner_config(tmp_path):
    config = {
        "data": {
            "provider_uri": str(tmp_path / "qlib_data"),
            "region": "us",
            "start_time": "2021-01-01",
            "end_time": "2021-01-31",
            "market": "csi300",
        },
        "factor_source": {
            "type": "custom",
            "custom": {"json_files": []},
        },
        "llm": {
            "cache_dir": str(tmp_path / "cache"),
            "auto_extract_cache": False,
        },
        "dataset": {
            "label": "Ref($close, -2) / Ref($close, -1) - 1",
            "learn_processors": [],
            "infer_processors": [],
            "segments": {
                "train": ["2021-01-01", "2021-01-10"],
                "valid": ["2021-01-11", "2021-01-20"],
                "test": ["2021-01-21", "2021-01-31"],
            },
        },
        "model": {"type": "lgb", "params": {}},
        "backtest": {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"topk": 2, "n_drop": 1},
            },
            "backtest": {
                "start_time": "2021-01-21",
                "end_time": "2021-01-31",
                "account": 1_000_000,
                "benchmark": "SH000300",
                "exchange_kwargs": {},
            },
        },
        "experiment": {
            "name": "exp",
            "recorder": "rec",
            "output_dir": str(tmp_path / "out"),
            "output_metrics_file": "metrics.json",
        },
    }
    config_path = tmp_path / "runner_full_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture
def fake_qlib_modules(monkeypatch):
    init_calls = []

    qlib_module = types.ModuleType("qlib")
    qlib_data_module = types.ModuleType("qlib.data")
    qlib_dataset_module = types.ModuleType("qlib.data.dataset")
    qlib_handler_module = types.ModuleType("qlib.data.dataset.handler")
    qlib_processor_module = types.ModuleType("qlib.data.dataset.processor")
    qlib_workflow_module = types.ModuleType("qlib.workflow")
    qlib_record_module = types.ModuleType("qlib.workflow.record_temp")
    qlib_backtest_module = types.ModuleType("qlib.backtest")
    qlib_contrib_module = types.ModuleType("qlib.contrib")
    qlib_contrib_model_module = types.ModuleType("qlib.contrib.model")
    qlib_contrib_gbdt_module = types.ModuleType("qlib.contrib.model.gbdt")
    qlib_contrib_eval_module = types.ModuleType("qlib.contrib.evaluate")

    def _init(provider_uri=None, region=None):
        init_calls.append({"provider_uri": provider_uri, "region": region})

    qlib_module.init = _init

    class FakeDataHandler:
        pass

    class FakeDataHandlerLP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeDatasetH:
        def __init__(self, handler, segments):
            self.handler = handler
            self.segments = segments

    class _NoOpProcessor:
        def __init__(self, *args, **kwargs):
            pass

    class FakeRecorder:
        def load_object(self, path):
            if path.endswith("ric.pkl"):
                return pd.Series([0.03, 0.05])
            if path.endswith("ic.pkl"):
                return pd.Series([0.10, 0.20])
            raise FileNotFoundError(path)

    class _StartContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeR:
        _recorder = FakeRecorder()

        @classmethod
        def start(cls, experiment_name, recorder_name):
            return _StartContext()

        @classmethod
        def get_recorder(cls):
            return cls._recorder

    class FakeSignalRecord:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self):
            return None

    class FakeSigAnaRecord(FakeSignalRecord):
        pass

    class FakeD:
        @staticmethod
        def instruments(market):
            return ["AAA", "BBB"]

        @staticmethod
        def list_instruments(instruments, start_time, end_time, as_list=True):
            return list(instruments)

        @staticmethod
        def features(stock_list, fields, start_time, end_time, freq="day"):
            dates = pd.to_datetime(["2021-01-21", "2021-01-22"])
            if fields == ["$close"]:
                idx = pd.MultiIndex.from_product([stock_list, dates], names=["instrument", "datetime"])
                close_values = [0.0, 10.0, 12.0, 13.0]
                return pd.DataFrame({"$close": close_values[: len(idx)]}, index=idx)

            idx = pd.MultiIndex.from_product([stock_list, dates], names=["instrument", "datetime"])
            data = {field: [0.1 * (i + 1) for i in range(len(idx))] for field in fields}
            return pd.DataFrame(data, index=idx)

    class FakeLGBModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, dataset):
            self.dataset = dataset

        def predict(self, dataset):
            idx = pd.MultiIndex.from_product(
                [pd.to_datetime(["2021-01-21", "2021-01-22"]), ["AAA", "BBB"]],
                names=["datetime", "instrument"],
            )
            return pd.Series([0.1, 0.2, 0.3, 0.4], index=idx)

    def fake_backtest(**kwargs):
        report_df = pd.DataFrame(
            {
                "return": [0.01, 0.03],
                "bench": [0.0, 0.01],
                "cost": [0.001, 0.001],
            },
            index=pd.to_datetime(["2021-01-21", "2021-01-22"]),
        )
        return {"1day": (report_df, pd.DataFrame())}, {}

    def fake_risk_analysis(_):
        return pd.DataFrame(
            {
                "risk": {
                    "annualized_return": 0.3,
                    "information_ratio": 1.2,
                    "max_drawdown": -0.1,
                }
            }
        )

    qlib_data_module.D = FakeD
    qlib_dataset_module.DatasetH = FakeDatasetH
    qlib_handler_module.DataHandler = FakeDataHandler
    qlib_handler_module.DataHandlerLP = FakeDataHandlerLP
    qlib_processor_module.Fillna = _NoOpProcessor
    qlib_processor_module.ProcessInf = _NoOpProcessor
    qlib_processor_module.CSRankNorm = _NoOpProcessor
    qlib_processor_module.DropnaLabel = _NoOpProcessor
    qlib_workflow_module.R = FakeR
    qlib_record_module.SignalRecord = FakeSignalRecord
    qlib_record_module.SigAnaRecord = FakeSigAnaRecord
    qlib_backtest_module.backtest = fake_backtest
    qlib_contrib_gbdt_module.LGBModel = FakeLGBModel
    qlib_contrib_eval_module.risk_analysis = fake_risk_analysis

    monkeypatch.setitem(sys.modules, "qlib", qlib_module)
    monkeypatch.setitem(sys.modules, "qlib.data", qlib_data_module)
    monkeypatch.setitem(sys.modules, "qlib.data.dataset", qlib_dataset_module)
    monkeypatch.setitem(sys.modules, "qlib.data.dataset.handler", qlib_handler_module)
    monkeypatch.setitem(sys.modules, "qlib.data.dataset.processor", qlib_processor_module)
    monkeypatch.setitem(sys.modules, "qlib.workflow", qlib_workflow_module)
    monkeypatch.setitem(sys.modules, "qlib.workflow.record_temp", qlib_record_module)
    monkeypatch.setitem(sys.modules, "qlib.backtest", qlib_backtest_module)
    monkeypatch.setitem(sys.modules, "qlib.contrib", qlib_contrib_module)
    monkeypatch.setitem(sys.modules, "qlib.contrib.model", qlib_contrib_model_module)
    monkeypatch.setitem(sys.modules, "qlib.contrib.model.gbdt", qlib_contrib_gbdt_module)
    monkeypatch.setitem(sys.modules, "qlib.contrib.evaluate", qlib_contrib_eval_module)

    return {"init_calls": init_calls, "backtest_module": qlib_backtest_module, "data_module": qlib_data_module}


def test_init_qlib_prefers_config_uri_and_is_idempotent(tmp_path, fake_qlib_modules, monkeypatch):
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    # Env var set, but config-file provider_uri should win (enables cross-market backtests)
    monkeypatch.setenv("QLIB_DATA_DIR", "~/qlib_env_data")
    monkeypatch.setenv("QLIB_PROVIDER_URI", "/should/not/be/used")

    runner._init_qlib()
    runner._init_qlib()  # second call is a no-op

    calls = fake_qlib_modules["init_calls"]
    assert len(calls) == 1
    # Config-file provider_uri wins over env (B16 fix: cross-market backtest support)
    assert calls[0]["provider_uri"] == str(tmp_path / "qlib_data")
    assert calls[0]["region"] == "us"


def test_run_updates_inputs_and_skips_custom_compute_when_no_custom(tmp_path, monkeypatch):
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    monkeypatch.setattr(runner, "_init_qlib", lambda: None)
    captured = {}

    monkeypatch.setattr(runner, "_load_factors", lambda: ({"qlib_factor": "$close"}, [{"name": "cf"}]))
    def _fake_compute_custom_factors(factors, skip_compute=False):
        captured["skip_compute"] = skip_compute
        return pd.DataFrame(
            {"cf": [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2021-01-21"), "AAA")],
                names=["datetime", "instrument"],
            ),
        )

    monkeypatch.setattr(runner, "_compute_custom_factors", _fake_compute_custom_factors)
    monkeypatch.setattr(runner, "_create_dataset", lambda factor_expressions, computed_factors: "dataset")
    monkeypatch.setattr(runner, "_train_and_backtest", lambda dataset, exp_name, rec_name, output_name=None: {"IC": 0.1})
    monkeypatch.setattr(runner, "_print_results", lambda metrics, total_time: None)
    monkeypatch.setattr(
        runner,
        "_save_results",
        lambda metrics, exp_name, factor_source, num_factors, elapsed, output_name=None: captured.setdefault(
            "saved",
            {"exp_name": exp_name, "factor_source": factor_source, "num_factors": num_factors, "output_name": output_name},
        ),
    )

    metrics = runner.run(
        factor_source="combined",
        factor_json=["/tmp/my_factor_file.json"],
        skip_uncached=True,
    )
    assert metrics["IC"] == 0.1
    assert runner.config["factor_source"]["type"] == "combined"
    assert runner.config["factor_source"]["custom"]["json_files"] == ["/tmp/my_factor_file.json"]
    assert captured["skip_compute"] is True
    assert captured["saved"]["output_name"] == "my_factor_file"
    assert captured["saved"]["num_factors"] == 2

    monkeypatch.setattr(runner, "_load_factors", lambda: ({"only_qlib": "$close"}, []))
    monkeypatch.setattr(runner, "_compute_custom_factors", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")))
    second = runner.run(output_name="explicit_name")
    assert second["IC"] == 0.1


def test_load_factors_and_compute_custom_factors_variants(tmp_path, monkeypatch):
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    import quantaalpha.backtest.factor_loader as factor_loader_module
    import quantaalpha.backtest.custom_factor_calculator as calculator_module

    class FakeLoader:
        def __init__(self, config):
            self.config = config

        def load_factors(self):
            return {"A": "$close"}, [{"name": "customA"}]

    monkeypatch.setattr(factor_loader_module, "FactorLoader", FakeLoader)
    qlib_factors, custom_factors = runner._load_factors()
    assert qlib_factors == {"A": "$close"}
    assert custom_factors == [{"name": "customA"}]

    class FakeCalculator:
        return_value = None
        last_cache_dir = None
        last_auto_extract = None
        last_skip_compute = None

        def __init__(self, data_df=None, cache_dir=None, auto_extract_cache=True, config=None):
            FakeCalculator.last_cache_dir = cache_dir
            FakeCalculator.last_auto_extract = auto_extract_cache
            self.config = config

        def calculate_factors_batch(self, factors, use_cache=True, skip_compute=False):
            FakeCalculator.last_skip_compute = skip_compute
            return FakeCalculator.return_value

    monkeypatch.setattr(calculator_module, "CustomFactorCalculator", FakeCalculator)

    FakeCalculator.return_value = None
    assert runner._compute_custom_factors([{"name": "x"}]) is None

    FakeCalculator.return_value = ["bad-type"]
    assert runner._compute_custom_factors([{"name": "x"}]) is None

    FakeCalculator.return_value = pd.DataFrame()
    assert runner._compute_custom_factors([{"name": "x"}]) is None

    non_multi = pd.DataFrame({"x": [1.0]}, index=pd.Index([0], name="row"))
    FakeCalculator.return_value = non_multi
    result = runner._compute_custom_factors([{"name": "x"}], skip_compute=True)
    assert isinstance(result, pd.DataFrame)
    assert FakeCalculator.last_skip_compute is True
    assert str(FakeCalculator.last_cache_dir).endswith("cache")
    assert FakeCalculator.last_auto_extract is False


def test_create_dataset_and_handler_fetch_branches(tmp_path, fake_qlib_modules, monkeypatch):
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    with pytest.raises(ValueError, match="No factor expressions available"):
        runner._create_dataset({}, computed_factors=None)

    dataset = runner._create_dataset({"A": "$close"}, computed_factors="bad-type")
    assert dataset.handler.kwargs["data_loader"]["class"] == "QlibDataLoader"
    assert dataset.segments["test"] == ["2021-01-21", "2021-01-31"]

    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2021-01-21", "2021-01-22"]), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    computed = pd.DataFrame({"customF": [0.1, 0.2, 0.3, 0.4]}, index=idx)
    labels = pd.DataFrame(
        {"LABEL0": [0.01, 0.02, 0.03, 0.04]},
        index=idx.swaplevel().sort_values(),
    )
    qlib_features = pd.DataFrame(
        {"qlibF": [1.0, 2.0, 3.0, 4.0]},
        index=idx.swaplevel().sort_values(),
    )
    monkeypatch.setattr(runner, "_compute_label", lambda label_expr: labels)
    monkeypatch.setattr(runner, "_load_qlib_factors", lambda factor_expressions: qlib_features)

    ds = runner._create_dataset_with_computed_factors({"qlibF": "$close"}, computed)
    handler = ds.handler

    feature_df = handler.fetch(col_set="feature")
    label_df = handler.fetch(col_set="label")
    full_df = handler.fetch(col_set="__all")
    picked = handler.fetch(col_set=[("feature", "customF")])
    ranged_tuple = handler.fetch(selector=("2021-01-21", "2021-01-21"), col_set="feature")
    ranged_slice = handler.fetch(selector=slice("2021-01-21", "2021-01-21"), col_set="feature")
    squeezed = handler.fetch(col_set="label", squeeze=True)

    assert handler.data_loader is None
    assert set(handler.instruments) == {"AAA", "BBB"}
    assert list(handler.get_cols("feature")) == ["customF", "qlibF"]
    assert "customF" in handler.get_cols("not_exists")
    assert len(feature_df) == len(label_df) == len(full_df)
    assert picked.shape[1] == 1
    assert len(ranged_tuple) > 0 and len(ranged_slice) > 0
    assert isinstance(squeezed, pd.Series)
    handler.setup_data()
    handler.config()

    handler._data.index = handler._data.index.set_names(["dt", "ins"])
    assert set(handler.instruments) == {"AAA", "BBB"}
    fallback_fetch = handler.fetch(selector=("2021-01-21", "2021-01-21"), col_set="feature")
    assert len(fallback_fetch) > 0


def test_label_loading_factor_loading_and_train_backtest_branches(tmp_path, fake_qlib_modules, monkeypatch):
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    label_df = runner._compute_label("Ref($close, -2) / Ref($close, -1) - 1")
    assert list(label_df.columns) == ["LABEL0"]
    assert len(label_df) > 0

    loaded_factors = runner._load_qlib_factors({"A": "$close", "B": "$open"})
    assert list(loaded_factors.columns) == ["A", "B"]

    monkeypatch.setattr(fake_qlib_modules["data_module"].D, "features", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("read fail")))
    assert runner._load_qlib_factors({"A": "$close"}) is None

    # Restore a working features loader for train/backtest checks.
    def _working_features(stock_list, fields, start_time, end_time, freq="day"):
        dates = pd.to_datetime(["2021-01-21", "2021-01-22"])
        if fields == ["$close"]:
            idx = pd.MultiIndex.from_product([stock_list, dates], names=["instrument", "datetime"])
            return pd.DataFrame({"$close": [0.0, 10.0, 11.0, 12.0][: len(idx)]}, index=idx)
        idx = pd.MultiIndex.from_product([stock_list, dates], names=["instrument", "datetime"])
        return pd.DataFrame({f: [0.1, 0.2, 0.3, 0.4][: len(idx)] for f in fields}, index=idx)

    monkeypatch.setattr(fake_qlib_modules["data_module"].D, "features", _working_features)

    metrics = runner._train_and_backtest(dataset=object(), exp_name="exp", rec_name="rec", output_name="run_ok")
    assert metrics["IC"] == pytest.approx(0.15)
    assert metrics["Rank IC"] == pytest.approx(0.04)
    assert metrics["annualized_return"] == pytest.approx(0.3)
    assert metrics["calmar_ratio"] == pytest.approx(3.0)
    assert (tmp_path / "out" / "run_ok_cumulative_excess.csv").exists()

    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("csv fail")))
    metrics_csv_fail = runner._train_and_backtest(dataset=object(), exp_name="exp", rec_name="rec", output_name="run_csv_fail")
    assert metrics_csv_fail["annualized_return"] == pytest.approx(0.3)

    monkeypatch.setattr(
        fake_qlib_modules["backtest_module"],
        "backtest",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("portfolio failed")),
    )
    metrics_bt_fail = runner._train_and_backtest(dataset=object(), exp_name="exp", rec_name="rec", output_name="run_bt_fail")
    assert "IC" in metrics_bt_fail

    runner.config["model"]["type"] = "xgb"
    with pytest.raises(ValueError, match="Unsupported model type"):
        runner._train_and_backtest(dataset=object(), exp_name="exp", rec_name="rec", output_name="run_bad_model")


def test_save_results_recovers_from_corrupted_summary(tmp_path):
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "batch_summary.json"
    summary_file.write_text("not-json", encoding="utf-8")

    runner._save_results(
        metrics={"annualized_return": 0.2, "max_drawdown": -0.1},
        exp_name="exp_corrupt",
        factor_source="custom",
        num_factors=3,
        elapsed=2.5,
        output_name="repair_case",
    )

    repaired = yaml.safe_load(summary_file.read_text(encoding="utf-8"))
    assert isinstance(repaired, list)
    assert repaired[-1]["name"] == "repair_case"


def test_train_and_backtest_saves_model_files(tmp_path, fake_qlib_modules, monkeypatch):
    """Model booster + feature_cols.json + meta.json saved when LGBModel.model.save_model exists."""
    import json
    import types
    import numpy as np

    model_save_dir = tmp_path / "models"
    # Write config with model_save_dir
    config = {
        "data": {
            "provider_uri": str(tmp_path / "qlib_data"),
            "region": "us",
            "start_time": "2021-01-01",
            "end_time": "2021-01-31",
            "market": "csi300",
        },
        "factor_source": {
            "type": "custom",
            "custom": {"json_files": ["lib.json"]},
        },
        "llm": {"cache_dir": str(tmp_path / "cache"), "auto_extract_cache": False},
        "dataset": {
            "label": "Ref($close,-2)/Ref($close,-1)-1",
            "learn_processors": [],
            "infer_processors": [],
            "segments": {
                "train": ["2021-01-01", "2021-01-10"],
                "valid": ["2021-01-11", "2021-01-20"],
                "test": ["2021-01-21", "2021-01-31"],
            },
        },
        "model": {"type": "lgb", "params": {}},
        "backtest": {
            "strategy": {
                "class": "TopkDropoutStrategy",
                "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"topk": 2, "n_drop": 1},
            },
            "backtest": {
                "start_time": "2021-01-21", "end_time": "2021-01-31",
                "account": 1_000_000, "benchmark": "SH000300",
                "exchange_kwargs": {},
            },
        },
        "experiment": {
            "name": "exp",
            "recorder": "rec",
            "output_dir": str(tmp_path / "out"),
            "output_metrics_file": "metrics.json",
            "model_save_dir": str(model_save_dir),
        },
    }
    import yaml as _yaml
    cfg_path = tmp_path / "cfg_with_save.yaml"
    cfg_path.write_text(_yaml.safe_dump(config))
    runner = BacktestRunner(str(cfg_path))

    # Keep the save_model calls side-effect-free; write a sentinel file
    saved_paths = []

    class FakeLGBBooster:
        def save_model(self, path):
            saved_paths.append(path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake_booster")

        def feature_importance(self, importance_type="gain"):
            return np.array([1.0, 2.0])

        def feature_name(self):
            return ["Column_0", "Column_1"]

        best_iteration = 10

    class FakeLGBModelWithBooster:
        def __init__(self, **kwargs):
            self.model = FakeLGBBooster()

        def fit(self, dataset):
            pass

        def predict(self, dataset):
            idx = pd.MultiIndex.from_product(
                [pd.to_datetime(["2021-01-21", "2021-01-22"]), ["AAA", "BBB"]],
                names=["datetime", "instrument"],
            )
            return pd.Series([0.1, 0.2, 0.3, 0.4], index=idx)

    # Patch the LGBModel that the runner imports
    gbdt_mod = sys.modules.get("qlib.contrib.model.gbdt")
    assert gbdt_mod is not None, "fake qlib modules must be installed by fixture"
    original_lgb = gbdt_mod.LGBModel
    gbdt_mod.LGBModel = FakeLGBModelWithBooster

    # Stash feature_cols for the column-name mapping
    runner._feature_cols = ["momentum_5d", "volatility_10d"]

    try:
        metrics = runner._train_and_backtest(dataset=object(), exp_name="exp", rec_name="rec", output_name="my_run")
    finally:
        gbdt_mod.LGBModel = original_lgb  # restore

    # Booster file was created
    assert len(saved_paths) == 1
    assert "my_run_lgbm.txt" in saved_paths[0]

    # Feature cols JSON was written
    fc_path = model_save_dir / "my_run_feature_cols.json"
    assert fc_path.exists()
    assert json.loads(fc_path.read_text()) == ["momentum_5d", "volatility_10d"]

    # Metadata JSON was written
    meta_path = model_save_dir / "my_run_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model_stem"] == "my_run"
    assert meta["num_features"] == 2
    assert "momentum_5d" in meta["feature_cols"]


def test_train_and_backtest_model_save_failure_does_not_crash(tmp_path, fake_qlib_modules, monkeypatch):
    """If save_model raises, _train_and_backtest still returns metrics (no crash)."""
    class FakeLGBBoosterBroken:
        def save_model(self, path):
            raise OSError("disk full")

        def feature_importance(self, importance_type="gain"):
            return []

        def feature_name(self):
            return []

    class FakeLGBModelBroken:
        def __init__(self, **kwargs):
            self.model = FakeLGBBoosterBroken()

        def fit(self, dataset):
            pass

        def predict(self, dataset):
            idx = pd.MultiIndex.from_product(
                [pd.to_datetime(["2021-01-21", "2021-01-22"]), ["AAA", "BBB"]],
                names=["datetime", "instrument"],
            )
            return pd.Series([0.1, 0.2, 0.3, 0.4], index=idx)

    gbdt_mod = sys.modules.get("qlib.contrib.model.gbdt")
    original_lgb = gbdt_mod.LGBModel
    gbdt_mod.LGBModel = FakeLGBModelBroken
    config_path = _write_runner_config(tmp_path)
    runner = BacktestRunner(str(config_path))
    runner._feature_cols = []

    try:
        metrics = runner._train_and_backtest(dataset=object(), exp_name="exp", rec_name="rec", output_name="run_save_fail")
    finally:
        gbdt_mod.LGBModel = original_lgb

    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# Regression: PrecomputedDataHandler must handle list selectors (YAML segments)
# ---------------------------------------------------------------------------

class TestPrecomputedDataHandlerListSelector:
    """Regression for bug where YAML list segments bypassed date filtering."""

    @staticmethod
    def _make_handler():
        """Create a minimal PrecomputedDataHandler with 3 years of data."""
        import pandas as pd
        import numpy as np
        from qlib.data.dataset.handler import DataHandler

        dates = pd.bdate_range("2016-01-01", "2018-12-31")
        instruments = ["AAPL", "MSFT"]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        n = len(idx)
        cols = pd.MultiIndex.from_tuples([
            ("feature", "f1"), ("feature", "f2"), ("label", "LABEL0")
        ])
        data = pd.DataFrame(np.random.randn(n, 3).astype(np.float32), index=idx, columns=cols)

        class PrecomputedDataHandler(DataHandler):
            def __init__(self, data_df, segments):
                self._data = data_df
                self._segments = segments
            @property
            def data_loader(self):
                return None
            @property
            def instruments(self):
                return list(self._data.index.get_level_values(1).unique())
            def fetch(self, selector=None, level='datetime', col_set='feature',
                     data_key=None, squeeze=False, proc_func=None):
                if col_set in ('feature', 'label'):
                    result = self._data[col_set].copy()
                elif col_set == '__all' or col_set is None:
                    result = self._data.copy()
                else:
                    if isinstance(col_set, (list, tuple)):
                        result = self._data[list(col_set)].copy()
                    else:
                        result = self._data.copy()
                if selector is not None:
                    try:
                        dates = result.index.get_level_values('datetime')
                    except KeyError:
                        dates = result.index.get_level_values(0)
                    if isinstance(selector, (tuple, list)) and len(selector) == 2:
                        start, end = selector
                        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                        result = result.loc[mask]
                    elif isinstance(selector, slice):
                        start, end = selector.start, selector.stop
                        if start is not None and end is not None:
                            mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                            result = result.loc[mask]
                if squeeze and result.shape[1] == 1:
                    result = result.iloc[:, 0]
                return result
            def get_cols(self, col_set='feature'):
                return list(self._data[col_set].columns) if col_set in self._data.columns.get_level_values(0) else []
            def setup_data(self, **kwargs):
                pass
            def config(self, **kwargs):
                pass

        segments = {
            "train": ["2016-01-01", "2016-12-31"],
            "valid": ["2017-01-01", "2017-12-31"],
            "test":  ["2018-01-01", "2018-12-31"],
        }
        return PrecomputedDataHandler(data, segments), segments

    def test_list_selector_filters_dates(self):
        """List segments (from YAML) must filter dates, not return all data."""
        import pandas as pd
        from qlib.data.dataset import DatasetH

        handler, segments = self._make_handler()
        dataset = DatasetH(handler=handler, segments=segments)
        train_df = dataset.prepare("train", col_set="feature")
        dates = train_df.index.get_level_values("datetime")
        assert dates.max() <= pd.Timestamp("2016-12-31"), \
            f"Train data leaked past 2016: max date = {dates.max()}"
        assert dates.min() >= pd.Timestamp("2016-01-01"), \
            f"Train data before 2016: min date = {dates.min()}"

    def test_tuple_selector_filters_dates(self):
        """Tuple selectors must also filter correctly."""
        import pandas as pd
        from qlib.data.dataset import DatasetH

        handler, segments = self._make_handler()
        segments_tuple = {k: tuple(v) for k, v in segments.items()}
        dataset = DatasetH(handler=handler, segments=segments_tuple)
        test_df = dataset.prepare("test", col_set="feature")
        dates = test_df.index.get_level_values("datetime")
        assert dates.min() >= pd.Timestamp("2018-01-01")
        assert dates.max() <= pd.Timestamp("2018-12-31")

    def test_train_valid_test_no_overlap(self):
        """Train/valid/test segments must not overlap."""
        import pandas as pd
        from qlib.data.dataset import DatasetH

        handler, segments = self._make_handler()
        dataset = DatasetH(handler=handler, segments=segments)
        train = set(dataset.prepare("train", col_set="feature").index.get_level_values("datetime"))
        valid = set(dataset.prepare("valid", col_set="feature").index.get_level_values("datetime"))
        test  = set(dataset.prepare("test",  col_set="feature").index.get_level_values("datetime"))
        assert not (train & valid), "train/valid overlap"
        assert not (train & test), "train/test overlap"
        assert not (valid & test), "valid/test overlap"
