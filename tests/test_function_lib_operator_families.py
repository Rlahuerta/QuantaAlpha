import numpy as np
import pandas as pd
import pytest

from quantaalpha.factors.coder.function_lib import (
    COUNT,
    DECAYLINEAR,
    DELAY,
    EMA,
    FILTER,
    GE,
    GT,
    HIGHDAY,
    INV,
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
    SUMAC,
    SUMIF,
    TS_ARGMAX,
    TS_ARGMIN,
    TS_COVARIANCE,
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
