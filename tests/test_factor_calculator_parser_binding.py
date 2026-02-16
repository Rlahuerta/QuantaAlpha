import numpy as np
import pandas as pd

from quantaalpha.backtest.factor_calculator import FactorCalculator


def _sample_df(columns):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    base = np.arange(len(idx), dtype=float)
    data = {col: base + i + 1 for i, col in enumerate(columns)}
    return pd.DataFrame(data, index=idx)


def _build_calculator(tmp_path, df):
    config = {
        "llm": {
            "enabled": False,
            "cache_results": False,
            "cache_dir": str(tmp_path / "factor_cache"),
        },
        "factor_calculation": {
            "output_dir": str(tmp_path / "computed_factors"),
        },
    }
    return FactorCalculator(config=config, data_df=df)


def test_factor_calculator_parser_supports_plain_columns(tmp_path):
    calc = _build_calculator(tmp_path, _sample_df(["return", "volume"]))
    result = calc._calculate_with_parser("RANK(TS_SUM($return, 2) * TS_SUM($volume, 2))")
    assert isinstance(result, pd.Series)
    assert len(result) == 8
    assert result.notna().any()


def test_factor_calculator_parser_supports_dollar_prefixed_columns(tmp_path):
    calc = _build_calculator(tmp_path, _sample_df(["$return", "$volume"]))
    result = calc._calculate_with_parser("RANK(TS_SUM($return, 2) * TS_SUM($volume, 2))")
    assert isinstance(result, pd.Series)
    assert len(result) == 8
    assert result.notna().any()

