import builtins
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

import quantaalpha.backtest.custom_factor_calculator as calculator_module
from quantaalpha.backtest.custom_factor_calculator import (
    CustomFactorCalculator,
    CustomFactorDataLoader,
    get_qlib_stock_data,
)


def _sample_data(with_duplicate=False):
    base_idx = [
        (pd.Timestamp("2024-01-01"), "AAA"),
        (pd.Timestamp("2024-01-02"), "AAA"),
        (pd.Timestamp("2024-01-01"), "BBB"),
        (pd.Timestamp("2024-01-02"), "BBB"),
    ]
    if with_duplicate:
        base_idx.append((pd.Timestamp("2024-01-02"), "BBB"))
    idx = pd.MultiIndex.from_tuples(base_idx, names=["datetime", "instrument"])
    return pd.DataFrame(
        {
            "$open": [10, 11, 20, 21] + ([21] if with_duplicate else []),
            "$high": [11, 12, 21, 22] + ([22] if with_duplicate else []),
            "$low": [9, 10, 19, 20] + ([20] if with_duplicate else []),
            "$close": [10.5, 11.5, 20.5, 21.5] + ([21.5] if with_duplicate else []),
            "$volume": [100, 101, 200, 201] + ([201] if with_duplicate else []),
            "$vwap": [10.2, 11.2, 20.2, 21.2] + ([21.2] if with_duplicate else []),
        },
        index=idx,
    )


def test_init_property_prepare_and_cache_helpers(tmp_path, monkeypatch):
    raw = _sample_data(with_duplicate=True)
    calc = CustomFactorCalculator(data_df=raw, cache_dir=tmp_path / "cache")
    assert calc._data_prepared is True
    assert "$return" in calc.data_df.columns
    assert not calc.data_df.index.duplicated().any()

    key = calc._get_cache_key("TS_MEAN($close, 5)")
    assert len(key) == 32

    s = pd.Series([1.0, 2.0], index=pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "AAA"), (pd.Timestamp("2024-01-02"), "AAA")],
        names=["datetime", "instrument"],
    ))
    calc._save_to_cache("expr_a", s)
    assert calc._load_from_cache("expr_a") is not None
    assert calc._load_from_cache("missing_expr") is None

    monkeypatch.setattr(pd, "read_pickle", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad pickle")))
    assert calc._load_from_cache("expr_a") is None

    h5_path = tmp_path / "result.h5"
    s.to_frame("factor").to_hdf(h5_path, key="data")
    assert calc._load_from_cache_location({"result_h5_path": str(h5_path)}) is not None
    assert calc._load_from_cache_location({"result_h5_path": str(tmp_path / "no_file.h5")}) is None
    assert calc._load_from_cache_location({}) is None
    assert calc._load_from_cache_location(None) is None

    monkeypatch.setattr(pd, "read_hdf", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad hdf")))
    assert calc._load_from_cache_location({"result_h5_path": str(h5_path)}) is None

    bad_dir = tmp_path / "not_a_dir"
    bad_dir.write_text("x", encoding="utf-8")
    calc.cache_dir = bad_dir
    calc._save_to_cache("expr_b", s)  # warning path, no raise


def test_process_cached_result_and_auto_extract_branches(tmp_path, monkeypatch):
    calc = CustomFactorCalculator(data_df=_sample_data(), cache_dir=tmp_path / "cache")
    idx_swapped = pd.MultiIndex.from_tuples(
        [("AAA", pd.Timestamp("2024-01-01")), ("AAA", pd.Timestamp("2024-01-02"))],
        names=["instrument", "datetime"],
    )

    one_col = pd.DataFrame({"x": [1.0, 2.0]}, index=idx_swapped)
    factor_col = pd.DataFrame({"factor": [1.0, 2.0], "other": [3.0, 4.0]}, index=idx_swapped)
    multi_col = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=idx_swapped)

    out_one = calc._process_cached_result(one_col, "one")
    out_factor = calc._process_cached_result(factor_col, "factor")
    out_multi = calc._process_cached_result(multi_col, "multi")
    out_bad = calc._process_cached_result(123, "bad")

    for out in [out_one, out_factor, out_multi]:
        assert isinstance(out, pd.Series)
        assert out.index.names == ["datetime", "instrument"]
    assert out_bad is None

    calc._cache_extracted = True
    calc._auto_extract_cache_from_logs()  # early return path

    calc._cache_extracted = False
    sys.modules.pop("tools.factor_cache_extractor", None)
    sys.modules.pop("tools", None)
    calc._auto_extract_cache_from_logs()  # import error path

    calc._cache_extracted = False
    tools_pkg = types.ModuleType("tools")
    extract_mod = types.ModuleType("tools.factor_cache_extractor")
    extract_mod.extract_factors_to_cache = lambda output_dir, verbose=False: 2
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.factor_cache_extractor", extract_mod)
    calc._auto_extract_cache_from_logs()

    calc._cache_extracted = False
    extract_mod.extract_factors_to_cache = lambda output_dir, verbose=False: (_ for _ in ()).throw(RuntimeError("boom"))
    calc._auto_extract_cache_from_logs()  # generic exception path


def test_lazy_data_loading_and_validate_align_paths(tmp_path, monkeypatch):
    calc = CustomFactorCalculator(data_df=None, cache_dir=tmp_path / "cache", config={"data": {"market": "csi300"}})
    monkeypatch.setattr(calculator_module, "get_qlib_stock_data", lambda config: _sample_data())
    loaded = calc.data_df
    assert "$return" in loaded.columns

    no_data_calc = CustomFactorCalculator(data_df=None, cache_dir=tmp_path / "cache", config=None)
    try:
        _ = no_data_calc.data_df
        raise AssertionError("Expected ValueError for missing data/config")
    except ValueError:
        pass

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "AAA"), (pd.Timestamp("2024-01-02"), "AAA")],
        names=["datetime", "instrument"],
    )
    series_ok = pd.Series([1.0, 2.0], index=idx)

    # target_idx from property raises -> fallback return non-empty result
    fallback = no_data_calc._validate_and_align_result(series_ok, "f1", reference_index=None)
    assert isinstance(fallback, pd.Series)

    # low index match path -> None
    disjoint_idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2025-01-01"), "ZZZ")],
        names=["datetime", "instrument"],
    )
    low_match = calc._validate_and_align_result(series_ok, "f2", reference_index=disjoint_idx)
    assert low_match is None

    # duplicate index alignment path
    dup_series = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01"), "AAA"),
                (pd.Timestamp("2024-01-01"), "AAA"),
                (pd.Timestamp("2024-01-02"), "AAA"),
            ],
            names=["datetime", "instrument"],
        ),
    )
    aligned = calc._validate_and_align_result(dup_series, "f3", reference_index=idx)
    assert isinstance(aligned, pd.Series)
    assert aligned.index.equals(idx)

    all_nan = pd.Series([np.nan, np.nan], index=idx)
    assert calc._validate_and_align_result(all_nan, "f4", reference_index=idx) is None


def test_calculate_factor_paths_with_patched_parser(tmp_path, monkeypatch):
    calc = CustomFactorCalculator(data_df=_sample_data(), cache_dir=tmp_path / "cache")

    import quantaalpha.factors.coder.expr_parser as expr_parser_module

    monkeypatch.setattr(expr_parser_module, "parse_symbol", lambda expr, cols: "$close")
    monkeypatch.setattr(expr_parser_module, "parse_expression", lambda expr: expr)
    result_series = calc.calculate_factor("f_series", "ignored")
    assert isinstance(result_series, pd.Series)
    assert result_series.name == "f_series"

    original_eval = builtins.eval
    monkeypatch.setattr(expr_parser_module, "parse_expression", lambda expr: "$close")
    monkeypatch.setattr(
        builtins,
        "eval",
        lambda _expr, _globals: pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]}, index=calc.data_df.index),
    )
    result_df = calc.calculate_factor("f_df", "ignored")
    assert isinstance(result_df, pd.Series)
    assert result_df.name == "f_df"
    monkeypatch.setattr(builtins, "eval", original_eval)

    monkeypatch.setattr(expr_parser_module, "parse_expression", lambda expr: "1.5")
    result_scalar = calc.calculate_factor("f_scalar", "ignored")
    assert isinstance(result_scalar, pd.Series)
    assert (result_scalar == 1.5).all()

    monkeypatch.setattr(
        expr_parser_module,
        "parse_symbol",
        lambda expr, cols: "CONST_EXPR",
    )
    monkeypatch.setattr(expr_parser_module, "parse_expression", lambda expr: "CONST_EXPR")
    monkeypatch.setattr(
        builtins,
        "eval",
        lambda _expr, _globals: pd.Series(_globals["df"]["$close"].values, index=[_globals["df"].index[0]] * len(_globals["df"])),
    )
    result_dup_idx = calc.calculate_factor("f_dup", "ignored")
    assert isinstance(result_dup_idx, pd.Series)
    monkeypatch.setattr(builtins, "eval", original_eval)

    monkeypatch.setattr(expr_parser_module, "parse_expression", lambda expr: (_ for _ in ()).throw(RuntimeError("parse fail")))
    assert calc.calculate_factor("f_error", "ignored") is None


def test_calculate_factors_from_json_and_batch_modes(tmp_path, monkeypatch):
    calc = CustomFactorCalculator(data_df=_sample_data(), cache_dir=tmp_path / "cache", auto_extract_cache=True)

    factors_json = {
        "factors": {
            "f1": {"factor_name": "A", "factor_expression": "exprA"},
            "f2": {"factor_name": "B", "factor_expression": ""},
            "f3": {"factor_name": "C", "factor_expression": "exprC"},
        }
    }
    json_path = tmp_path / "factors.json"
    json_path.write_text(json.dumps(factors_json), encoding="utf-8")

    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "AAA"), (pd.Timestamp("2024-01-02"), "AAA")],
        names=["datetime", "instrument"],
    )

    monkeypatch.setattr(
        calc,
        "calculate_factor",
        lambda factor_name, factor_expression: (
            pd.Series([1.0, 2.0], index=idx, name=factor_name) if factor_name == "A" else None
        ),
    )
    out_df = calc.calculate_factors_from_json(str(json_path), max_factors=2)
    assert list(out_df.columns) == ["A"]

    # skip_compute branch
    calc_skip = CustomFactorCalculator(data_df=_sample_data(), cache_dir=tmp_path / "cache2", auto_extract_cache=True)
    monkeypatch.setattr(calc_skip, "_auto_extract_cache_from_logs", lambda: setattr(calc_skip, "_cache_extracted", True))
    monkeypatch.setattr(calc_skip, "_load_from_cache_location", lambda cache_location: None)
    monkeypatch.setattr(calc_skip, "_load_from_cache", lambda expr: None)
    monkeypatch.setattr(calc_skip, "calculate_factor", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not compute")))
    skipped = calc_skip.calculate_factors_batch(
        [{"factor_name": "S1", "factor_expression": "e1"}],
        use_cache=True,
        skip_compute=True,
    )
    assert skipped.empty
    assert calc_skip._cache_extracted is True

    # full branch: cache hits + compute outcomes + align/validate
    calc_full = CustomFactorCalculator(data_df=_sample_data(), cache_dir=tmp_path / "cache3", auto_extract_cache=False)
    factors = [
        {"factor_name": "EMPTY", "factor_expression": ""},
        {"factor_name": "H5", "factor_expression": "e_h5", "cache_location": {"result_h5_path": "x.h5"}},
        {"factor_name": "MD5", "factor_expression": "e_md5"},
        {"factor_name": "OK", "factor_expression": "e_ok"},
        {"factor_name": "ALLNAN", "factor_expression": "e_nan"},
        {"factor_name": "NONE", "factor_expression": "e_none"},
        {"factor_name": "ERR", "factor_expression": "e_err"},
    ]

    monkeypatch.setattr(
        calc_full,
        "_load_from_cache_location",
        lambda cache_location: pd.Series([10.0, 11.0], index=idx, name="H5")
        if cache_location
        else None,
    )
    monkeypatch.setattr(
        calc_full,
        "_load_from_cache",
        lambda expr: pd.Series([20.0, 21.0], index=idx, name="MD5") if expr == "e_md5" else None,
    )

    saved = {"count": 0}
    monkeypatch.setattr(calc_full, "_save_to_cache", lambda expr, result: saved.__setitem__("count", saved["count"] + 1))
    monkeypatch.setattr(calc_full, "_validate_and_align_result", lambda result, factor_name, reference_index=None: result)

    def _fake_calculate_factor(name, expr):
        if name == "OK":
            return pd.Series([30.0, 31.0], index=idx, name=name)
        if name == "ALLNAN":
            return pd.Series([np.nan, np.nan], index=idx, name=name)
        if name == "NONE":
            return None
        if name == "ERR":
            raise RuntimeError("compute err")
        return None

    monkeypatch.setattr(calc_full, "calculate_factor", _fake_calculate_factor)

    # trigger signal setup fallback branch
    import signal as signal_module

    monkeypatch.setattr(signal_module, "signal", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("no signal")))
    result_df = calc_full.calculate_factors_batch(factors, use_cache=True, skip_compute=False)
    assert set(result_df.columns) == {"H5", "MD5", "OK"}
    assert saved["count"] == 1


def test_custom_factor_data_loader_and_get_qlib_stock_data(tmp_path, monkeypatch):
    idx = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), "AAA"), (pd.Timestamp("2024-01-02"), "AAA")],
        names=["datetime", "instrument"],
    )
    factor_df = pd.DataFrame({"f1": [0.1, 0.2]}, index=idx)
    data_df = _sample_data().loc[idx]
    loader = CustomFactorDataLoader(factor_df=factor_df)

    import quantaalpha.factors.coder.expr_parser as expr_parser_module

    monkeypatch.setattr(expr_parser_module, "parse_symbol", lambda expr, cols: "$close")
    monkeypatch.setattr(expr_parser_module, "parse_expression", lambda expr: "$close")
    features, labels = loader.to_qlib_format(data_df)
    assert features.equals(factor_df)
    assert list(labels.columns) == ["LABEL0"]

    original_eval = builtins.eval
    monkeypatch.setattr(
        builtins,
        "eval",
        lambda _expr, _globals: pd.DataFrame({"label": [1.0, 2.0]}, index=data_df.index),
    )
    _, labels_from_df = loader.to_qlib_format(data_df)
    assert list(labels_from_df.columns) == ["LABEL0"]
    monkeypatch.setattr(builtins, "eval", original_eval)

    # get_qlib_stock_data paths
    qlib_module = types.ModuleType("qlib")
    qlib_data_module = types.ModuleType("qlib.data")
    init_calls = {"count": 0}

    def _init(provider_uri=None, region=None):
        init_calls["count"] += 1
        raise RuntimeError("already init")

    class FakeD:
        @staticmethod
        def instruments(market):
            return ["AAA", "BBB"]

        @staticmethod
        def features(stock_list, fields, start_time, end_time, freq="day"):
            idx2 = pd.MultiIndex.from_product(
                [stock_list, pd.to_datetime(["2024-01-01", "2024-01-02"])],
                names=["instrument", "datetime"],
            )
            data = {field: np.arange(len(idx2), dtype=float) + i for i, field in enumerate(fields)}
            return pd.DataFrame(data, index=idx2)

    qlib_module.init = _init
    qlib_data_module.D = FakeD
    monkeypatch.setitem(sys.modules, "qlib", qlib_module)
    monkeypatch.setitem(sys.modules, "qlib.data", qlib_data_module)
    monkeypatch.setenv("QLIB_DATA_DIR", "~/qlib_env_data")

    cfg = {
        "data": {
            "provider_uri": "/fallback/provider",
            "region": "us",
            "start_time": "2024-01-01",
            "end_time": "2024-01-02",
            "market": "csi300",
        }
    }
    loaded = get_qlib_stock_data(cfg)
    assert init_calls["count"] == 1
    assert list(loaded.columns) == ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
    assert len(loaded) > 0
