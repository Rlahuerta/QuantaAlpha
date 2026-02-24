import sys
import types

import pandas as pd
import pytest
import yaml

from quantaalpha.backtest.runner import BacktestRunner


def _write_train_backtest_config(tmp_path):
    config = {
        'data': {
            'start_time': '2021-01-01',
            'end_time': '2021-12-31',
            'market': 'csi300',
        },
        'dataset': {
            'segments': {
                'train': ['2021-01-01', '2021-06-30'],
                'valid': ['2021-07-01', '2021-09-30'],
                'test': ['2021-10-01', '2021-12-31'],
            },
        },
        'model': {
            'type': 'lgb',
            'params': {},
        },
        'backtest': {
            'strategy': {
                'class': 'TopkDropoutStrategy',
                'module_path': 'qlib.contrib.strategy.signal_strategy',
                'kwargs': {
                    'topk': 10,
                    'n_drop': 2,
                },
            },
            'backtest': {
                'start_time': '2021-01-01',
                'end_time': '2021-01-31',
                'account': 1_000_000,
                'benchmark': 'SH000300',
                'exchange_kwargs': {},
            },
        },
        'experiment': {
            'output_dir': str(tmp_path),
            'output_metrics_file': 'metrics.json',
        },
    }
    config_path = tmp_path / 'backtest_train_config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return config_path


@pytest.fixture
def fake_qlib_train_modules(monkeypatch):
    qlib_module = types.ModuleType('qlib')
    qlib_data_module = types.ModuleType('qlib.data')
    qlib_workflow_module = types.ModuleType('qlib.workflow')
    qlib_record_module = types.ModuleType('qlib.workflow.record_temp')
    qlib_backtest_module = types.ModuleType('qlib.backtest')
    qlib_contrib_module = types.ModuleType('qlib.contrib')
    qlib_contrib_model_module = types.ModuleType('qlib.contrib.model')
    qlib_contrib_gbdt_module = types.ModuleType('qlib.contrib.model.gbdt')
    qlib_contrib_eval_module = types.ModuleType('qlib.contrib.evaluate')

    class FakeLGBModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, dataset):
            self.dataset = dataset

        def predict(self, dataset):
            pred_index = pd.MultiIndex.from_product(
                [pd.to_datetime(['2021-01-04']), ['AAA', 'BBB']],
                names=['datetime', 'instrument'],
            )
            return pd.Series([0.1, 0.2], index=pred_index)

    class FakeRecorder:
        def load_object(self, _):
            raise FileNotFoundError('IC artifact not found')

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
            return market

        @staticmethod
        def list_instruments(instruments, start_time, end_time, as_list=True):
            return ['AAA', 'BBB']

        @staticmethod
        def features(stock_list, fields, start_time, end_time, freq='day'):
            index = pd.MultiIndex.from_product(
                [stock_list, pd.to_datetime(['2021-01-04'])],
                names=['instrument', 'datetime'],
            )
            return pd.DataFrame({'$close': [10.0] * len(index)}, index=index)

    def fake_backtest(**kwargs):
        report_df = pd.DataFrame(
            {'return': [0.01, 0.02]},
            index=pd.to_datetime(['2021-01-04', '2021-01-05']),
        )
        return {'1day': (report_df, pd.DataFrame())}, {}

    def fake_risk_analysis(_):
        return pd.Series(
            {
                'annualized_return': 0.2,
                'information_ratio': 1.5,
                'max_drawdown': -0.1,
            }
        )

    qlib_data_module.D = FakeD
    qlib_workflow_module.R = FakeR
    qlib_record_module.SignalRecord = FakeSignalRecord
    qlib_record_module.SigAnaRecord = FakeSigAnaRecord
    qlib_backtest_module.backtest = fake_backtest
    qlib_contrib_gbdt_module.LGBModel = FakeLGBModel
    qlib_contrib_eval_module.risk_analysis = fake_risk_analysis

    monkeypatch.setitem(sys.modules, 'qlib', qlib_module)
    monkeypatch.setitem(sys.modules, 'qlib.data', qlib_data_module)
    monkeypatch.setitem(sys.modules, 'qlib.workflow', qlib_workflow_module)
    monkeypatch.setitem(sys.modules, 'qlib.workflow.record_temp', qlib_record_module)
    monkeypatch.setitem(sys.modules, 'qlib.backtest', qlib_backtest_module)
    monkeypatch.setitem(sys.modules, 'qlib.contrib', qlib_contrib_module)
    monkeypatch.setitem(sys.modules, 'qlib.contrib.model', qlib_contrib_model_module)
    monkeypatch.setitem(sys.modules, 'qlib.contrib.model.gbdt', qlib_contrib_gbdt_module)
    monkeypatch.setitem(sys.modules, 'qlib.contrib.evaluate', qlib_contrib_eval_module)


def test_train_backtest_handles_missing_ic_and_missing_bench_cost(
    tmp_path, fake_qlib_train_modules
):
    config_path = _write_train_backtest_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    metrics = runner._train_and_backtest(dataset=object(), exp_name='exp', rec_name='rec')

    assert 'IC' not in metrics
    assert metrics['annualized_return'] == pytest.approx(0.2)
    assert metrics['information_ratio'] == pytest.approx(1.5)
    assert metrics['max_drawdown'] == pytest.approx(-0.1)
    assert metrics['calmar_ratio'] == pytest.approx(2.0)
