import numpy as np
import pandas as pd

from quantaalpha.factors.coder.function_lib import (
    ADD,
    AND,
    BB_LOWER,
    BB_MIDDLE,
    BB_UPPER,
    DIVIDE,
    EQ,
    EXP,
    FLOOR,
    GE,
    GT,
    INV,
    LE,
    LOG,
    LT,
    MACD,
    MULTIPLY,
    NE,
    OR,
    POW,
    REGRESI,
    REGBETA,
    RSI,
    SCALE,
    SUBTRACT,
    TS_MAD,
    TS_QUANTILE,
    TS_CORR,
    TS_MEAN,
    TS_SUM,
    WHERE,
)
import quantaalpha.factors.regulator.factor_regulator as regulator_module
from quantaalpha.factors.regulator.factor_regulator import FactorRegulator


def _sample_series():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    return pd.Series([1.0, 2.0, 2.0, 4.0, 3.0, 6.0], index=idx)


def _sample_by_date():
    return pd.Series(
        [10.0, 20.0, 30.0],
        index=pd.date_range("2024-01-01", periods=3),
    )


def test_time_series_mean_and_sum_by_instrument():
    values = _sample_series()
    mean_result = TS_MEAN(values, 2)
    sum_result = TS_SUM(values, 2)

    assert mean_result.xs("AAA", level="instrument").tolist() == [1.0, 1.5, 2.5]
    assert mean_result.xs("BBB", level="instrument").tolist() == [2.0, 3.0, 5.0]
    assert sum_result.xs("AAA", level="instrument").tolist() == [1.0, 3.0, 5.0]
    assert sum_result.xs("BBB", level="instrument").tolist() == [2.0, 6.0, 10.0]


def test_time_series_corr_returns_rolling_correlation():
    values = _sample_series()
    corr = TS_CORR(values, values * 2, p=2)
    aaa = corr.xs("AAA", level="instrument")
    bbb = corr.xs("BBB", level="instrument")

    assert np.isnan(aaa.iloc[0]) and np.isnan(bbb.iloc[0])
    assert np.allclose(aaa.iloc[1:], [1.0, 1.0], atol=1e-6)
    assert np.allclose(bbb.iloc[1:], [1.0, 1.0], atol=1e-6)


def test_arithmetic_helpers_align_datetime_and_multiindex():
    values = _sample_series()
    by_date = _sample_by_date()

    add_result = ADD(values, by_date)
    sub_result = SUBTRACT(values, by_date)
    mul_result = MULTIPLY(values, by_date)
    div_result = DIVIDE(values, by_date)

    assert add_result.xs("AAA", level="instrument").tolist() == [11.0, 22.0, 33.0]
    assert sub_result.xs("AAA", level="instrument").tolist() == [-9.0, -18.0, -27.0]
    assert mul_result.xs("AAA", level="instrument").tolist() == [10.0, 40.0, 90.0]
    assert div_result.xs("AAA", level="instrument").tolist() == [0.1, 0.1, 0.1]


def test_arithmetic_compare_and_where_support_scalar_and_alignment():
    values = _sample_series()
    by_date = _sample_by_date()

    assert ADD(1, 2) == 3
    assert SUBTRACT(5, 2) == 3
    assert MULTIPLY(3, 4) == 12
    assert DIVIDE(8, 2) == 4

    gt_result = GT(values, by_date)
    lt_result = LT(1, values)
    ge_result = GE(values, by_date)
    le_result = LE(values, by_date)
    eq_result = EQ(values, values)
    ne_result = NE(values, by_date)

    assert gt_result.index.equals(values.index)
    assert lt_result.index.equals(values.index)
    assert ge_result.index.equals(values.index)
    assert le_result.index.equals(values.index)
    assert eq_result.all()
    assert ne_result.index.equals(values.index)

    cond = OR(gt_result, LT(values, 2))
    and_result = AND(cond, gt_result)
    where_result = WHERE(cond, by_date, -1)
    where_scalar = WHERE(False, values, 0)

    assert and_result.index.equals(values.index)
    assert where_result.index.equals(values.index)
    assert where_scalar.eq(0).all()


def test_bb_middle_supports_kwargs_on_decorated_function():
    values = _sample_series()
    result = BB_MIDDLE(values, 2, n_jobs=1)
    assert len(result) == len(values)
    assert result.notna().any()


def test_bb_middle_dynamic_window_series_does_not_crash():
    values = _sample_series()
    dynamic_window = pd.Series([2] * len(values), index=values.index)
    result = BB_MIDDLE(values, dynamic_window, 1)
    assert len(result) == len(values)
    assert result.notna().any()


def test_bollinger_bands_macd_rsi_and_regression_helpers():
    values = _sample_series()
    dynamic_window = pd.Series([1, 2, 3, 1, 2, 3], index=values.index)

    upper = BB_UPPER(values, dynamic_window, n_jobs=1)
    lower = BB_LOWER(values, dynamic_window, n_jobs=1)
    spread = (upper - lower).dropna()
    assert len(upper) == len(values)
    assert len(lower) == len(values)
    assert (spread >= 0).all()

    macd = MACD(values, short_window=2, long_window=3)
    rsi = RSI(values, window=2)
    assert len(macd) == len(values)
    assert len(rsi) == len(values)

    ref = np.array([1.0, 2.0, 3.0])
    beta_from_array = REGBETA(values, ref, p=3, n_jobs=1)
    beta_from_swapped = REGBETA(ref, values, p=3, n_jobs=1)
    residual_from_array = REGRESI(values, ref, p=3, n_jobs=1)
    residual_from_swapped = REGRESI(ref, values, p=3, n_jobs=1)

    for output in [beta_from_array, beta_from_swapped, residual_from_array, residual_from_swapped]:
        assert len(output) == len(values)
        assert output.notna().sum() >= 2


def test_math_and_distribution_helpers_cover_quantile_swap_branch():
    values = _sample_series()
    q_default = TS_QUANTILE(values, p=3, q=0.5)
    q_swapped = TS_QUANTILE(values, p=0.5, q=3)
    mad = TS_MAD(values, p=2)
    scaled = SCALE(values, target_sum=1.0)

    assert q_default.equals(q_swapped)
    assert len(mad) == len(values)
    assert np.isclose(scaled.groupby("datetime").apply(lambda x: x.abs().sum()).iloc[0], 1.0)

    assert np.isfinite(EXP(values)).all()
    assert np.isfinite(LOG(values)).all()
    assert np.isfinite(INV(values)).all()
    assert (POW(values, 2) == values**2).all()
    assert (FLOOR(values + 0.5) == np.floor(values + 0.5)).all()


def test_factor_regulator_evaluate_and_acceptance(monkeypatch):
    monkeypatch.setattr(regulator_module, "match_alphazoo", lambda expr, alphazoo: (2, "subtree", "alpha_001"))
    monkeypatch.setattr(regulator_module, "count_free_args", lambda expr: 1)
    monkeypatch.setattr(regulator_module, "count_unique_vars", lambda expr: 2)
    monkeypatch.setattr(regulator_module, "count_all_nodes", lambda expr: 10)
    monkeypatch.setattr(regulator_module, "calculate_symbol_length", lambda expr: 80)
    monkeypatch.setattr(regulator_module, "count_base_features", lambda expr: 3)

    regulator = FactorRegulator(duplication_threshold=8, symbol_length_threshold=300, base_features_threshold=6)
    ok, eval_dict = regulator.evaluate("TS_MEAN($close, 5)")

    assert ok is True
    assert eval_dict["duplicated_subtree_size"] == 2
    assert eval_dict["matched_alpha"] == "alpha_001"
    assert regulator.is_expression_acceptable(eval_dict) is True

    too_many_free_args = dict(eval_dict)
    too_many_free_args["num_free_args"] = 10
    too_many_free_args["num_all_nodes"] = 10
    assert regulator.is_expression_acceptable(too_many_free_args) is False


def test_factor_regulator_is_parsable_rejects_empty_expression():
    regulator = FactorRegulator()
    assert regulator.is_parsable("") is False


def test_factor_regulator_add_factor_accepts_scalar_and_list():
    regulator = FactorRegulator()
    regulator.add_factor("A", "TS_SUM($close, 2)")
    regulator.add_factor(["B", "C"], ["TS_MEAN($close, 2)", "TS_STD($close, 2)"])

    assert len(regulator.alphazoo) == 3
    assert regulator.get_new_factors() == [
        ("A", "TS_SUM($close, 2)"),
        ("B", "TS_MEAN($close, 2)"),
        ("C", "TS_STD($close, 2)"),
    ]
