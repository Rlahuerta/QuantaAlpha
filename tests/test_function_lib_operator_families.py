import numpy as np
import operator
import pandas as pd
import pytest

import quantaalpha.factors.coder.function_lib as function_lib_module
from quantaalpha.factors.coder.function_lib import (
    COUNT,
    DECAYLINEAR,
    DELAY,
    EMA,
    KURT,
    FILTER,
    GE,
    GT,
    HIGHDAY,
    INV,
    MAX,
    MEAN,
    MEDIAN,
    MIN,
    LOG,
    LOWDAY,
    PERCENTILE,
    PROD,
    POW,
    REGBETA,
    REGRESI,
    SCALE,
    SEQUENCE,
    SIGN,
    SMA,
    SQRT,
    STD,
    SUMAC,
    SUMIF,
    TS_ARGMAX,
    TS_ARGMIN,
    TS_COVARIANCE,
    TS_KURT,
    TS_MAX,
    TS_MEDIAN,
    TS_MIN,
    TS_PCTCHANGE,
    TS_RANK,
    TS_SKEW,
    TS_STD,
    TS_VAR,
    TS_ZSCORE,
    WHERE,
    WMA,
    ZSCORE,
)


def _sample_series():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    return pd.Series([1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0], index=idx)


def _sample_by_date():
    return pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2024-01-01", periods=4))


def test_datatype_adapter_handles_array_and_scalar_inputs():
    arr = np.array([1.0, 4.0, 9.0])

    sqrt_arr = SQRT(arr)
    sqrt_scalar = SQRT(9)
    pow_arr = POW(arr, 2)
    pow_swapped = POW(2, arr)

    assert isinstance(sqrt_arr, pd.DataFrame)
    assert sqrt_scalar == pytest.approx(3.0)
    assert isinstance(pow_arr, pd.DataFrame)
    assert isinstance(pow_swapped, pd.DataFrame)
    assert np.isfinite(LOG(1)).all()
    assert np.isfinite(INV(2)).all()


def test_rolling_statistical_operators_cover_rank_min_max_and_percentiles():
    values = _sample_series()

    ts_rank = TS_RANK(values, p=3)
    ts_max = TS_MAX(values, p=3)
    ts_min = TS_MIN(values, p=3)
    ts_median = TS_MEDIAN(values, p=3)
    ts_argmax = TS_ARGMAX(values, p=3)
    ts_argmin = TS_ARGMIN(values, p=3)
    pct_roll = PERCENTILE(values, q=0.5, p=3)
    pct_global = PERCENTILE(values, q=0.5)
    ts_std = TS_STD(values, p=3)
    ts_var = TS_VAR(values, p=3)
    ts_skew = TS_SKEW(values, p=3)
    ts_z = TS_ZSCORE(values, p=3)
    z = ZSCORE(values)
    scaled = SCALE(values, target_sum=1.0)

    for series in [
        ts_rank,
        ts_max,
        ts_min,
        ts_median,
        ts_argmax,
        ts_argmin,
        pct_roll,
        pct_global,
        ts_std,
        ts_var,
        ts_skew,
        ts_z,
        z,
        scaled,
    ]:
        assert len(series) == len(values)

    per_day_abs = scaled.groupby("datetime").apply(lambda x: x.abs().sum())
    assert np.isclose(per_day_abs.iloc[0], 1.0)


def test_covariance_delay_moving_average_and_conditional_operators():
    values = _sample_series()
    by_date = _sample_by_date()
    cond = GT(values, 2)

    cov_series = TS_COVARIANCE(values, values * 2, p=3)
    cov_array = TS_COVARIANCE(values, np.array([1.0, 2.0, 3.0]), p=3)
    delayed = DELAY(values, p=1)
    ema = EMA(values, 3)
    sma_window = SMA(values, m=3)
    sma_alpha = SMA(values, m=2, n=1)
    wma = WMA(values, p=3)
    count = COUNT(cond, p=2)
    sum_if = SUMIF(values, p=2, cond=cond)
    filtered = FILTER(values, cond)
    prod_roll = PROD(values, p=3)
    prod_mul = PROD(values, p=cond)
    decay = DECAYLINEAR(values, p=3)
    highday = HIGHDAY(values, p=3)
    lowday = LOWDAY(values, p=3)
    seq = SEQUENCE(5)
    sumac = SUMAC(values, p=3)
    pct_change = TS_PCTCHANGE(values, p=1)
    sign = SIGN(values - 3)
    where_scalar_only = WHERE(True, 1, 0)
    where_aligned = WHERE(GE(values, by_date), values, 0)

    assert len(cov_series) == len(values)
    assert len(cov_array) == len(values)
    assert len(delayed) == len(values)
    assert len(ema) == len(values)
    assert len(sma_window) == len(values)
    assert len(sma_alpha) == len(values)
    assert len(wma) == len(values)
    assert len(count) == len(values)
    assert len(sum_if) == len(values)
    assert len(filtered) == len(values)
    assert len(prod_roll) == len(values)
    assert len(prod_mul) == len(values)
    assert len(decay) == len(values)
    assert len(highday) == len(values)
    assert len(lowday) == len(values)
    assert len(sumac) == len(values)
    assert len(pct_change) == len(values)
    assert len(sign) == len(values)
    assert where_scalar_only == 1
    assert where_aligned.index.equals(values.index)
    assert seq.tolist() == [1, 2, 3, 4, 5]


def test_covariance_and_delay_input_validation_paths():
    values = _sample_series()

    with pytest.raises(TypeError):
        TS_COVARIANCE(values, {"bad": "type"}, p=2)

    with pytest.raises(AssertionError):
        DELAY(values, p=-1)


def test_regression_operators_mismatch_and_alignment_paths():
    values = _sample_series()
    by_date = _sample_by_date()

    mismatch_index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["AAA"]],
        names=["datetime", "instrument"],
    )
    mismatch_values = pd.Series([1.0, 2.0, 3.0, 4.0], index=mismatch_index)

    with pytest.raises(AssertionError, match="indices must align"):
        REGBETA(values, mismatch_values, p=3, n_jobs=1)

    aligned_residuals = REGRESI(values, by_date, p=3, n_jobs=1)
    assert len(aligned_residuals) == len(values)
    assert aligned_residuals.notna().sum() >= 2


def test_cross_sectional_stats_and_ternary_min_max_branches():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=5), ["AAA", "BBB", "CCC", "DDD"]],
        names=["datetime", "instrument"],
    )
    values = pd.Series(np.arange(1, len(idx) + 1, dtype=float), index=idx)

    mean = MEAN(values)
    std = STD(values)
    skew = TS_SKEW(values, p=4)
    cs_skew = function_lib_module.SKEW(values)
    kurt = KURT(values)
    median = MEDIAN(values)
    ts_kurt = TS_KURT(values, p=4)
    max2 = MAX(values, values * 2)
    max3 = MAX(values, values * 2, values * 0.5)
    min2 = MIN(values, values * 2)
    min3 = MIN(values, values * 2, values * 0.5)

    assert len(mean) == 5
    assert len(std) == 5
    assert len(skew) == len(values)
    assert len(cs_skew) == len(values)
    assert len(kurt) == len(values)
    assert len(median) == 5
    assert len(ts_kurt) == len(values)
    assert len(max2) == len(values)
    assert len(max3) == len(values)
    assert len(min2) == len(values)
    assert len(min3) == len(values)


def test_ts_corr_and_covariance_additional_type_paths():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["AAA"]],
        names=["datetime", "instrument"],
    )
    const = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
    arr = np.array([1.0, 2.0, 3.0, 4.0])
    by_date = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.Index(pd.date_range("2024-01-01", periods=4), name="datetime"),
    )

    corr_arr = function_lib_module.TS_CORR(const, arr, p=2)
    corr_date = function_lib_module.TS_CORR(const, by_date, p=2)
    cov_arr = TS_COVARIANCE(const, arr, p=2)
    cov_date = TS_COVARIANCE(const, by_date, p=2)

    assert len(corr_arr) == len(const)
    assert (corr_arr.dropna() == 0).all()
    assert len(corr_date) == len(const)
    assert len(cov_arr) == len(const)
    assert len(cov_date) == len(const)

    with pytest.raises(TypeError):
        function_lib_module.TS_CORR(const, {"bad": 1}, p=2)


def test_regresi_additional_alignment_and_group_count_paths():
    values = _sample_series()
    by_date = _sample_by_date()

    aligned = REGRESI(values, by_date, p=2, n_jobs=1)
    assert len(aligned) == len(values)

    mismatch_left = pd.Series([1.0, 2.0, 3.0], index=pd.Index(pd.date_range("2024-01-01", periods=3), name="d"))
    mismatch_right = pd.Series([1.0, 2.0, 3.0], index=pd.Index(pd.date_range("2024-02-01", periods=3), name="d"))
    with pytest.raises(AssertionError, match="indices must align"):
        REGRESI(mismatch_left, mismatch_right, p=2, n_jobs=1)

    one_instrument = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=4), ["AAA"]],
            names=["datetime", "instrument"],
        ),
    )
    one_instrument_res = REGRESI(values, one_instrument, p=2, n_jobs=1)
    assert len(one_instrument_res) == len(values)

    idx_inst = pd.Index(["AAA", "AAA", "BBB", "BBB"], name="instrument")
    inst_left = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx_inst)
    inst_right = pd.Series([2.0, 3.0, 4.0, 5.0], index=idx_inst)
    inst_residual = REGRESI(inst_left, inst_right, p=2, n_jobs=1)
    assert len(inst_residual) == 4


def test_alignment_helpers_and_where_edge_branches():
    values = _sample_series()
    by_date_idx = pd.date_range("2024-01-01", periods=4)
    by_date_df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=by_date_idx)
    values_df = values.to_frame("x")

    assert function_lib_module._arithmetic_with_alignment(1, 2, operator.add) == 3
    assert function_lib_module._arithmetic_with_alignment(1, values, operator.add).index.equals(values.index)
    assert function_lib_module._arithmetic_with_alignment(values, 1, operator.add).index.equals(values.index)
    assert function_lib_module._arithmetic_with_alignment(values_df, by_date_df, operator.add).index.equals(values_df.index)
    assert function_lib_module._arithmetic_with_alignment(by_date_df, values_df, operator.add).index.equals(values_df.index)

    dup_idx = pd.Index([by_date_idx[0], by_date_idx[0], by_date_idx[1]], name="datetime")
    dup_series = pd.Series([1.0, 2.0, 3.0], index=dup_idx)
    left_series = pd.Series([1.0, 2.0, 3.0], index=by_date_idx[:3])
    result_dup = function_lib_module._arithmetic_with_alignment(left_series, dup_series, operator.add)
    assert len(result_dup) >= 3

    calls = {"n": 0}

    def flaky_op(a, b):
        if calls["n"] == 0:
            calls["n"] += 1
            raise ValueError("identically-labeled")
        return a + b

    recovered = function_lib_module._arithmetic_with_alignment(values_df, values_df.copy(), flaky_op)
    assert recovered.index.equals(values_df.index)
    assert calls["n"] == 1

    with pytest.raises(ValueError):
        function_lib_module._arithmetic_with_alignment(
            values_df,
            values_df.copy(),
            lambda a, b: (_ for _ in ()).throw(ValueError("other-error")),
        )

    assert function_lib_module._align_for_operation(1, 2) == (1, 2)
    assert function_lib_module._align_for_operation(1, values)[0] == 1
    assert function_lib_module._align_for_operation(values, 1)[1] == 1

    bad_right = pd.Series([1.0, 2.0], index=pd.Index([by_date_idx[0], by_date_idx[0]], name="datetime"))
    aligned_left, aligned_right = function_lib_module._align_for_operation(left_series, bad_right)
    assert aligned_left is left_series
    assert aligned_right is bad_right

    assert function_lib_module._compare_with_alignment(1, 2, operator.lt) is True
    lt_from_scalar = function_lib_module._compare_with_alignment(1, values, operator.lt)
    assert lt_from_scalar.index.equals(values.index)
    gt_from_series = function_lib_module._compare_with_alignment(values, 1, operator.gt)
    assert gt_from_series.index.equals(values.index)

    cmp_calls = {"n": 0}

    def flaky_cmp(a, b):
        if cmp_calls["n"] == 0:
            cmp_calls["n"] += 1
            raise ValueError("identically-labeled")
        return a > b

    cmp_result = function_lib_module._compare_with_alignment(values_df, values_df.copy(), flaky_cmp)
    assert cmp_result.index.equals(values_df.index)
    with pytest.raises(ValueError):
        function_lib_module._compare_with_alignment(
            values_df,
            values_df.copy(),
            lambda a, b: (_ for _ in ()).throw(ValueError("other-error")),
        )

    plain_idx = pd.Index([0, 1, 2], name="i")
    out_false_idx = WHERE(True, 1, pd.Series([0, 1, 2], index=plain_idx))
    assert out_false_idx.index.equals(plain_idx)

    cond_plain = pd.Series([True, False, True], index=plain_idx)
    true_mismatch = pd.Series([10, 20, 30], index=pd.Index([1, 2, 3], name="i"))
    out_true_reindex = WHERE(cond_plain, true_mismatch, 0)
    assert out_true_reindex.index.equals(plain_idx)

    cond_multi = pd.Series(
        [True, False, True, False],
        index=pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=2), ["AAA", "BBB"]],
            names=["datetime", "instrument"],
        ),
    )
    false_by_date = pd.Series([0.0, 1.0], index=pd.date_range("2024-01-01", periods=2))
    out_false_multi = WHERE(cond_multi, 1, false_by_date)
    assert out_false_multi.index.equals(cond_multi.index)

    out_cond_same = WHERE(cond_plain, pd.Series([1, 2, 3], index=plain_idx), -1)
    assert out_cond_same.index.equals(plain_idx)


def test_dynamic_bollinger_window_dataframe_and_int_paths():
    values = _sample_series()
    dynamic_window_df = pd.DataFrame({"w": [0, 2, 0, 2, 0, 2, 0, 2]}, index=values.index)

    middle_dynamic = function_lib_module.BB_MIDDLE(values, dynamic_window_df, n_jobs=1)
    upper_dynamic = function_lib_module.BB_UPPER(values, dynamic_window_df, n_jobs=1)
    lower_dynamic = function_lib_module.BB_LOWER(values, dynamic_window_df, n_jobs=1)
    upper_int = function_lib_module.BB_UPPER(values, 2, n_jobs=1)
    lower_int = function_lib_module.BB_LOWER(values, 2, n_jobs=1)

    assert len(middle_dynamic) == len(values)
    assert len(upper_dynamic) == len(values)
    assert len(lower_dynamic) == len(values)
    assert len(upper_int) == len(values)
    assert len(lower_int) == len(values)
