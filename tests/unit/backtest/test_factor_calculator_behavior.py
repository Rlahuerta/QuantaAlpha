import numpy as np
import pandas as pd
import pytest

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


def test_calculate_factors_requires_data_then_set_data(tmp_path, monkeypatch):
    config = {
        "llm": {"enabled": False, "cache_results": False, "cache_dir": str(tmp_path / "cache")},
        "factor_calculation": {"output_dir": str(tmp_path / "out")},
    }
    calc = FactorCalculator(config=config, data_df=None)
    with pytest.raises(ValueError, match="Data not set"):
        calc.calculate_factors([{"factor_name": "x", "factor_expression": "TS_MEAN($close, 2)"}])

    calc.set_data(_sample_data())
    monkeypatch.setattr(calc, "_calculate_with_parser", lambda _expr: pd.Series(1.0, index=calc.data_df.index))
    result = calc.calculate_factors([{"factor_name": "x", "factor_expression": "TS_MEAN($close, 2)"}])
    assert "x" in result.columns


def test_calculate_with_parser_dataframe_scalar_and_exception_paths(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=False, cache_results=False)

    import quantaalpha.factors.coder.expr_parser as expr_parser

    monkeypatch.setattr(expr_parser, "parse_symbol", lambda expr, cols: expr)  # noqa: ARG005
    monkeypatch.setattr(
        expr_parser,
        "parse_expression",
        lambda expr: "pd.DataFrame({'x': np.arange(len(df))}, index=df.index)",  # noqa: ARG005
    )
    df_result = calc._calculate_with_parser("dummy")
    assert isinstance(df_result, pd.Series)

    monkeypatch.setattr(expr_parser, "parse_expression", lambda expr: "1.23")  # noqa: ARG005
    scalar_result = calc._calculate_with_parser("dummy")
    assert isinstance(scalar_result, pd.Series)
    assert (scalar_result == 1.23).all()

    monkeypatch.setattr(expr_parser, "parse_symbol", lambda expr, cols: (_ for _ in ()).throw(ValueError("bad")))  # noqa: ARG005
    assert calc._calculate_with_parser("dummy") is None


def test_calculate_with_llm_cache_none_and_exception_paths(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=True, cache_results=True)
    expr = "TS_MEAN($close, 2)"
    cached = pd.Series(np.arange(len(calc.data_df), dtype=float), index=calc.data_df.index)
    calc._save_to_cache(expr, cached)
    assert calc._calculate_with_llm({"factor_name": "cached", "factor_expression": expr}) is not None

    monkeypatch.setattr(calc, "_generate_factor_code", lambda info: None)  # noqa: ARG005
    assert calc._calculate_with_llm({"factor_name": "none", "factor_expression": "x"}) is None

    monkeypatch.setattr(calc, "_generate_factor_code", lambda info: (_ for _ in ()).throw(RuntimeError("boom")))  # noqa: ARG005
    assert calc._calculate_with_llm({"factor_name": "err", "factor_expression": "y"}) is None


def test_execute_factor_code_cache_io_and_generation_error_paths(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=True, cache_results=True)

    monkeypatch.setattr(calc, "_calculate_with_parser", lambda expr: (_ for _ in ()).throw(RuntimeError("bad")))  # noqa: ARG005
    assert calc._execute_factor_code("expr", "name") is None

    bad_expr = "BAD_CACHE_EXPR"
    bad_key = calc._get_cache_key(bad_expr)
    bad_cache_file = calc.cache_dir / f"{bad_key}.pkl"
    bad_cache_file.write_text("not-a-pickle", encoding="utf-8")
    assert calc._load_from_cache(bad_expr) is None

    monkeypatch.setattr(pd.Series, "to_pickle", lambda self, path: (_ for _ in ()).throw(OSError("io error")))
    calc._save_to_cache("SAVE_FAIL_EXPR", pd.Series([1.0]))

    class RaisingAPIBackend:
        def build_messages_and_create_chat_completion(self, **kwargs):
            raise RuntimeError("api down")

    monkeypatch.setattr("quantaalpha.llm.client.APIBackend", RaisingAPIBackend)
    assert (
        calc._generate_factor_code(
            {
                "factor_name": "x",
                "factor_expression": "TS_MEAN($close, 5)",
                "factor_description": "",
                "variables": {},
            }
        )
        is None
    )


def test_calculate_factors_continues_after_exception(tmp_path, monkeypatch):
    calc = _build_calculator(tmp_path, llm_enabled=False, cache_results=False)
    monkeypatch.setattr(calc, "_calculate_with_parser", lambda expr: (_ for _ in ()).throw(RuntimeError("crash")))  # noqa: ARG005
    result = calc.calculate_factors([{"factor_name": "boom", "factor_expression": "x"}])
    assert result.empty
