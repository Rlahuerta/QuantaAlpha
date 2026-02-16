import numpy as np
import pandas as pd

from quantaalpha.backtest.factor_calculator import FactorCalculator


def _sample_data():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    base = np.arange(len(idx), dtype=float) + 1
    return pd.DataFrame(
        {
            "$close": base,
            "$open": base + 0.5,
            "$volume": base * 10,
        },
        index=idx,
    )


def _build_calculator(tmp_path, llm_enabled=False, cache_results=True):
    config = {
        "llm": {
            "enabled": llm_enabled,
            "cache_results": cache_results,
            "cache_dir": str(tmp_path / "factor_cache"),
        },
        "factor_calculation": {
            "output_dir": str(tmp_path / "computed_factors"),
        },
    }
    return FactorCalculator(config=config, data_df=_sample_data())


def test_validate_expression_rules(tmp_path):
    calc = _build_calculator(tmp_path)
    assert calc._validate_expression("TS_MEAN($close, 2)") is True
    assert calc._validate_expression("") is False
    assert calc._validate_expression("TS_MEAN(close, 2)") is False
    assert calc._validate_expression("TS_MEAN($close, 2") is False


def test_calculate_factors_uses_cache_when_available(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=False, cache_results=True)
    expr = "TS_MEAN($close, 2)"
    cached = pd.Series(np.arange(len(calc.data_df), dtype=float), index=calc.data_df.index)
    calc._save_to_cache(expr, cached)

    monkeypatch.setattr(calc, "_calculate_with_parser", lambda _expr: (_ for _ in ()).throw(AssertionError("parser called")))

    result = calc.calculate_factors([{"factor_name": "cached_factor", "factor_expression": expr}])

    assert "cached_factor" in result.columns
    assert result["cached_factor"].equals(cached)


def test_calculate_factors_falls_back_to_llm_when_parser_returns_none(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=True, cache_results=False)
    llm_series = pd.Series(np.ones(len(calc.data_df)), index=calc.data_df.index)

    monkeypatch.setattr(calc, "_calculate_with_parser", lambda _expr: None)
    monkeypatch.setattr(calc, "_calculate_with_llm", lambda _factor_info: llm_series)

    result = calc.calculate_factors([{"factor_name": "llm_factor", "factor_expression": "BAD_EXPR"}])

    assert result["llm_factor"].equals(llm_series)


def test_calculate_factors_returns_empty_when_parser_fails_and_llm_disabled(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=False, cache_results=False)
    monkeypatch.setattr(calc, "_calculate_with_parser", lambda _expr: None)

    result = calc.calculate_factors([{"factor_name": "bad_factor", "factor_expression": "BAD_EXPR"}])

    assert result.empty


def test_generate_factor_code_accepts_valid_llm_expression(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=True, cache_results=False)

    class FakeAPIBackend:
        def build_messages_and_create_chat_completion(self, **kwargs):
            return '"TS_MEAN($close, 3)"'

    monkeypatch.setattr("quantaalpha.llm.client.APIBackend", FakeAPIBackend)

    expr = calc._generate_factor_code(
        {
            "factor_name": "x",
            "factor_expression": "placeholder",
            "factor_description": "",
            "variables": {},
        }
    )

    assert expr == "TS_MEAN($close, 3)"


def test_generate_factor_code_rejects_invalid_llm_expression(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=True, cache_results=False)

    class FakeAPIBackend:
        def build_messages_and_create_chat_completion(self, **kwargs):
            return "invalid_expression_without_dollar"

    monkeypatch.setattr("quantaalpha.llm.client.APIBackend", FakeAPIBackend)

    expr = calc._generate_factor_code(
        {
            "factor_name": "x",
            "factor_expression": "placeholder",
            "factor_description": "",
            "variables": {},
        }
    )

    assert expr is None
