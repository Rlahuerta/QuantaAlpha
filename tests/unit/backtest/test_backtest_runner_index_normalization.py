import sys
import types

import pandas as pd
import pytest
import yaml

from quantaalpha.backtest.runner import BacktestRunner


def _write_minimal_backtest_config(tmp_path):
    config = {
        'data': {
            'start_time': '2021-01-01',
            'end_time': '2021-12-31',
            'market': 'csi300',
        },
        'dataset': {
            'label': ['Ref($close, -2) / Ref($close, -1) - 1'],
            'learn_processors': [],
            'infer_processors': [],
            'segments': {
                'train': ['2021-01-01', '2021-06-30'],
                'valid': ['2021-07-01', '2021-09-30'],
                'test': ['2021-10-01', '2021-12-31'],
            },
        },
    }
    config_path = tmp_path / 'backtest_config.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return config_path


@pytest.fixture
def fake_qlib_modules(monkeypatch):
    qlib_module = types.ModuleType('qlib')
    qlib_data_module = types.ModuleType('qlib.data')
    qlib_dataset_module = types.ModuleType('qlib.data.dataset')
    qlib_handler_module = types.ModuleType('qlib.data.dataset.handler')
    qlib_processor_module = types.ModuleType('qlib.data.dataset.processor')

    class FakeDataHandler:
        pass

    class FakeDatasetH:
        def __init__(self, handler, segments):
            self.handler = handler
            self.segments = segments

    class _NoOpProcessor:
        def __init__(self, *args, **kwargs):
            pass

    qlib_dataset_module.DatasetH = FakeDatasetH
    qlib_handler_module.DataHandler = FakeDataHandler
    qlib_handler_module.DataHandlerLP = FakeDataHandler
    qlib_data_module.D = object()
    qlib_processor_module.Fillna = _NoOpProcessor
    qlib_processor_module.ProcessInf = _NoOpProcessor
    qlib_processor_module.CSRankNorm = _NoOpProcessor
    qlib_processor_module.DropnaLabel = _NoOpProcessor

    monkeypatch.setitem(sys.modules, 'qlib', qlib_module)
    monkeypatch.setitem(sys.modules, 'qlib.data', qlib_data_module)
    monkeypatch.setitem(sys.modules, 'qlib.data.dataset', qlib_dataset_module)
    monkeypatch.setitem(sys.modules, 'qlib.data.dataset.handler', qlib_handler_module)
    monkeypatch.setitem(sys.modules, 'qlib.data.dataset.processor', qlib_processor_module)


def test_create_dataset_normalizes_to_datetime_instrument_index(tmp_path, fake_qlib_modules, monkeypatch):
    config_path = _write_minimal_backtest_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    datetimes = pd.to_datetime(['2021-01-04', '2021-01-05'])
    instruments = ['AAA', 'BBB']

    computed_index = pd.MultiIndex.from_product(
        [datetimes, instruments], names=['datetime', 'instrument']
    )
    computed_factors = pd.DataFrame({'custom_factor': [0.1, 0.2, 0.3, 0.4]}, index=computed_index)

    swapped_index = pd.MultiIndex.from_tuples(
        [(ins, dt) for dt in datetimes for ins in instruments],
        names=['instrument', 'datetime'],
    )
    qlib_factors = pd.DataFrame({'qlib_factor': [1.0, 2.0, 3.0, 4.0]}, index=swapped_index)
    labels = pd.DataFrame({'LABEL0': [0.01, 0.02, 0.03, 0.04]}, index=swapped_index)

    monkeypatch.setattr(runner, '_compute_label', lambda _: labels)
    monkeypatch.setattr(runner, '_load_qlib_factors', lambda _: qlib_factors)

    dataset = runner._create_dataset_with_computed_factors(
        factor_expressions={'qlib_factor': '$close'},
        computed_factors=computed_factors,
    )
    prepared = dataset.handler._data

    assert prepared.index.names == ['datetime', 'instrument']
    assert pd.api.types.is_datetime64_any_dtype(prepared.index.get_level_values('datetime'))
    assert ('feature', 'custom_factor') in prepared.columns
    assert ('feature', 'qlib_factor') in prepared.columns
    assert ('label', 'LABEL0') in prepared.columns


def test_create_dataset_rejects_non_multiindex_computed_factors(tmp_path, fake_qlib_modules, monkeypatch):
    config_path = _write_minimal_backtest_config(tmp_path)
    runner = BacktestRunner(str(config_path))

    valid_index = pd.MultiIndex.from_product(
        [pd.to_datetime(['2021-01-04']), ['AAA']], names=['datetime', 'instrument']
    )
    labels = pd.DataFrame({'LABEL0': [0.01]}, index=valid_index)

    monkeypatch.setattr(runner, '_compute_label', lambda _: labels)
    monkeypatch.setattr(runner, '_load_qlib_factors', lambda _: pd.DataFrame(index=valid_index))

    invalid_computed = pd.DataFrame({'custom_factor': [0.1]}, index=pd.Index([0], name='row'))

    with pytest.raises(ValueError, match='computed_features index must be MultiIndex'):
        runner._create_dataset_with_computed_factors({}, invalid_computed)
