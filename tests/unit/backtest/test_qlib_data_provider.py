import sys
import types

import numpy as np
import pandas as pd

from quantaalpha.backtest.factor_calculator import QlibDataProvider


def _install_fake_qlib(monkeypatch):
    init_calls = {"count": 0, "args": None}

    qlib_module = types.ModuleType("qlib")
    qlib_config_module = types.ModuleType("qlib.config")
    qlib_data_module = types.ModuleType("qlib.data")

    def fake_init(provider_uri=None, region=None):
        init_calls["count"] += 1
        init_calls["args"] = {"provider_uri": provider_uri, "region": region}

    class FakeD:
        @staticmethod
        def instruments(market):
            return market

        @staticmethod
        def features(stock_list, fields, start_time, end_time, freq="day"):
            idx = pd.MultiIndex.from_product(
                [["AAA", "BBB"], pd.date_range("2024-01-01", periods=3)],
                names=["instrument", "datetime"],
            )
            values = np.arange(len(idx) * len(fields), dtype=float).reshape(len(idx), len(fields)) + 1
            return pd.DataFrame(values, index=idx, columns=fields)

    qlib_module.init = fake_init
    qlib_config_module.REG_CN = "REG_CN"
    qlib_config_module.REG_US = "REG_US"
    qlib_data_module.D = FakeD

    monkeypatch.setitem(sys.modules, "qlib", qlib_module)
    monkeypatch.setitem(sys.modules, "qlib.config", qlib_config_module)
    monkeypatch.setitem(sys.modules, "qlib.data", qlib_data_module)
    return init_calls


def test_qlib_data_provider_initializes_once_and_respects_region(monkeypatch):
    init_calls = _install_fake_qlib(monkeypatch)
    provider = QlibDataProvider({"data": {"provider_uri": "/tmp/qlib", "region": "us"}})

    provider._init_qlib()
    provider._init_qlib()

    assert provider._initialized is True
    assert init_calls["count"] == 1
    assert init_calls["args"]["provider_uri"] == "/tmp/qlib"
    assert init_calls["args"]["region"] == "REG_US"


def test_qlib_data_provider_builds_stock_dataframe_with_return(monkeypatch):
    _install_fake_qlib(monkeypatch)
    provider = QlibDataProvider(
        {
            "data": {
                "provider_uri": "/tmp/qlib",
                "region": "cn",
                "start_time": "2024-01-01",
                "end_time": "2024-01-03",
                "market": "csi300",
            }
        }
    )

    df = provider.get_stock_data()

    assert {"$open", "$high", "$low", "$close", "$volume", "$vwap", "$return"}.issubset(df.columns)
    assert len(df) == 6
    assert df["$return"].isna().sum() == 2
