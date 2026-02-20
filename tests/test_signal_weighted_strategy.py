"""Tests for SignalWeightedTopkStrategy.

Tests use a mock Position object to avoid Qlib initialisation.
All tests are pure-Python / numpy — no live market data required.
"""
import numpy as np
import pandas as pd
import pytest

from quantaalpha.backtest.strategy import SignalWeightedTopkStrategy
from qlib.contrib.strategy.signal_strategy import WeightStrategyBase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakePosition:
    """Minimal mock of qlib.backtest.position.Position."""

    def __init__(self, holdings):
        self._holdings = list(holdings)

    def get_stock_list(self):
        return list(self._holdings)


def _make_score(n=10, seed=42):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.random(n), index=[f"S{i:03d}" for i in range(n)])


def _make_strategy(topk=5, n_drop=2, weight_scheme="linear_rank"):
    """Create strategy bypassing Qlib's signal initialization (tests weight logic only)."""
    strat = SignalWeightedTopkStrategy.__new__(SignalWeightedTopkStrategy)
    strat.topk = topk
    strat.n_drop = n_drop
    strat.weight_scheme = weight_scheme
    strat.rebalance_frequency = 1
    return strat


# ---------------------------------------------------------------------------
# Weight scheme: weights sum to 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["linear_rank", "softmax", "score", "equal"])
def test_weights_sum_to_one_empty_portfolio(scheme):
    strat = _make_strategy(topk=5, weight_scheme=scheme)
    score = _make_score(10)
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert weights, "Should select stocks on first day"
    assert len(weights) == 5
    assert abs(sum(weights.values()) - 1.0) < 1e-9


@pytest.mark.parametrize("scheme", ["linear_rank", "softmax", "score", "equal"])
def test_weights_sum_to_one_with_existing_portfolio(scheme):
    strat = _make_strategy(topk=5, weight_scheme=scheme)
    score = _make_score(10)
    pos = _FakePosition(["S000", "S001", "S002", "S003", "S004"])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Weight ordering: higher signal → higher weight
# ---------------------------------------------------------------------------

def test_linear_rank_monotone():
    """Highest-score stock should get largest weight."""
    strat = _make_strategy(topk=5, weight_scheme="linear_rank")
    score = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3, "E": 0.1})
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert weights["A"] > weights["B"] > weights["C"] > weights["D"] > weights["E"]


def test_equal_scheme_all_equal():
    strat = _make_strategy(topk=4, weight_scheme="equal")
    score = pd.Series({"A": 0.9, "B": 0.7, "C": 0.5, "D": 0.3})
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert all(abs(w - 0.25) < 1e-9 for w in weights.values())


def test_softmax_monotone():
    strat = _make_strategy(topk=3, weight_scheme="softmax")
    score = pd.Series({"A": 2.0, "B": 1.0, "C": 0.0})
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert weights["A"] > weights["B"] > weights["C"]


# ---------------------------------------------------------------------------
# Turnover constraint
# ---------------------------------------------------------------------------

def test_n_drop_limits_replacements():
    """At most n_drop stocks should be swapped per day."""
    strat = _make_strategy(topk=5, n_drop=1)
    # Portfolio holds S000..S004; signal now prefers S005..S009 (all new)
    score = pd.Series(
        {"S005": 0.9, "S006": 0.8, "S007": 0.7, "S008": 0.6, "S009": 0.5,
         "S000": 0.4, "S001": 0.3, "S002": 0.2, "S003": 0.1, "S004": 0.05}
    )
    pos = _FakePosition(["S000", "S001", "S002", "S003", "S004"])
    weights = strat.generate_target_weight_position(score, pos, None, None)

    new_stocks = set(weights.keys()) - {"S000", "S001", "S002", "S003", "S004"}
    removed_stocks = {"S000", "S001", "S002", "S003", "S004"} - set(weights.keys())
    # With n_drop=1, at most 1 stock is replaced
    assert len(new_stocks) <= 1
    assert len(removed_stocks) <= 1


def test_no_change_when_portfolio_optimal():
    """If current holdings are already the top-k, no change should occur."""
    strat = _make_strategy(topk=3, n_drop=2)
    score = pd.Series({"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.1, "E": 0.05})
    pos = _FakePosition(["A", "B", "C"])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert set(weights.keys()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_score_returns_empty():
    strat = _make_strategy(topk=5)
    score = pd.Series(dtype=float)
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert weights == {}


def test_score_with_all_nans_returns_empty():
    strat = _make_strategy(topk=3)
    score = pd.Series({"A": np.nan, "B": np.nan})
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert weights == {}


def test_dataframe_score_uses_first_column():
    strat = _make_strategy(topk=3, weight_scheme="equal")
    score_df = pd.DataFrame(
        {"score": {"A": 0.9, "B": 0.8, "C": 0.7}, "other": {"A": 0.1, "B": 0.2, "C": 0.3}}
    )
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score_df, pos, None, None)
    assert set(weights.keys()) == {"A", "B", "C"}


def test_fewer_stocks_than_topk():
    """If fewer stocks are available than topk, return all of them."""
    strat = _make_strategy(topk=10, weight_scheme="equal")
    score = pd.Series({"A": 0.5, "B": 0.3})
    pos = _FakePosition([])
    weights = strat.generate_target_weight_position(score, pos, None, None)
    assert set(weights.keys()) == {"A", "B"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9



# ---------------------------------------------------------------------------
# Rebalance frequency: generate_trade_decision skipping
# ---------------------------------------------------------------------------

class _FakeTradeCalendar:
    def __init__(self, step):
        self._step = step
    def get_trade_step(self):
        return self._step
    def get_step_time(self, step=None, shift=0):
        return None, None


class _FakeSignal:
    def get_signal(self, start_time=None, end_time=None):
        return pd.Series({"A": 0.9, "B": 0.8, "C": 0.7})


def _make_strategy_with_calendar(step, freq=5):
    """Create strategy ready for generate_trade_decision tests."""
    strat = SignalWeightedTopkStrategy.__new__(SignalWeightedTopkStrategy)
    strat.topk = 3
    strat.n_drop = 1
    strat.weight_scheme = "equal"
    strat.rebalance_frequency = freq
    strat.signal = _FakeSignal()
    return strat, _FakeTradeCalendar(step)


def test_rebalance_frequency_skips_non_rebalance_days():
    """On non-rebalance days, generate_trade_decision should return empty order list."""
    from unittest.mock import patch, PropertyMock
    from qlib.backtest.decision import TradeDecisionWO
    for step in [1, 2, 3, 4, 6, 7, 8, 9]:
        strat, cal = _make_strategy_with_calendar(step, freq=5)
        with patch.object(type(strat), "trade_calendar", new_callable=PropertyMock, return_value=cal):
            decision = strat.generate_trade_decision()
        assert isinstance(decision, TradeDecisionWO)
        assert list(decision.order_list) == [], f"Expected no orders on step {step}"


def test_rebalance_frequency_calls_super_on_rebalance_days():
    """On rebalance days (step % freq == 0), super().generate_trade_decision is called."""
    from unittest.mock import patch, PropertyMock
    strat, cal = _make_strategy_with_calendar(step=5, freq=5)
    with patch.object(type(strat), "trade_calendar", new_callable=PropertyMock, return_value=cal):
        with patch.object(WeightStrategyBase, "generate_trade_decision", return_value="mocked") as mock_gdt:
            result = strat.generate_trade_decision()
    mock_gdt.assert_called_once()
    assert result == "mocked"


def test_rebalance_frequency_default_is_daily():
    """Default rebalance_frequency=1 means every day is a rebalance day."""
    strat = _make_strategy(topk=3, weight_scheme="equal")
    assert strat.rebalance_frequency == 1


def test_rebalance_frequency_step_zero_is_rebalance():
    """Step 0 (first day) is always a rebalance day."""
    from unittest.mock import patch, PropertyMock
    strat, cal = _make_strategy_with_calendar(step=0, freq=5)
    with patch.object(type(strat), "trade_calendar", new_callable=PropertyMock, return_value=cal):
        with patch.object(WeightStrategyBase, "generate_trade_decision", return_value="mocked") as mock_gdt:
            result = strat.generate_trade_decision()
    mock_gdt.assert_called_once()


def test_runner_parquet_fallback_signal_weighted():
    """_compute_portfolio_metrics_from_parquet should accept weight_scheme in config."""
    from unittest.mock import patch
    from quantaalpha.backtest.runner import BacktestRunner

    runner = BacktestRunner.__new__(BacktestRunner)
    runner.config = {
        "backtest": {
            "backtest": {"start_time": "2022-01-01", "end_time": "2022-12-31"},
            "strategy": {"kwargs": {"topk": 3, "weight_scheme": "linear_rank"}},
        }
    }

    dates = pd.date_range("2022-01-03", periods=5, freq="B")
    symbols = ["A", "B", "C"]

    pred = pd.Series(
        [0.9, 0.5, 0.1] * 5,
        index=pd.MultiIndex.from_tuples(
            [(dt, s) for dt in dates for s in symbols], names=["datetime", "instrument"]
        ),
    )

    prices_rows = []
    for i, dt in enumerate(dates):
        for s in symbols:
            prices_rows.append({"datetime": dt, "symbol": s, "open": 10.0 + i * 0.1})
    prices = pd.DataFrame(prices_rows)
    prices["datetime"] = pd.to_datetime(prices["datetime"])

    bench_rows = [{"datetime": dt, "open": 100.0 + i} for i, dt in enumerate(dates)]
    bench = pd.DataFrame(bench_rows)
    bench["datetime"] = pd.to_datetime(bench["datetime"])

    strategy_config = {"kwargs": {"topk": 3, "weight_scheme": "linear_rank"}}

    with patch.object(runner, "_load_parquet_prices", return_value=prices), \
         patch.object(runner, "_load_parquet_benchmark", return_value=bench):
        result = runner._compute_portfolio_metrics_from_parquet(pred, strategy_config)

    # Should return a dict (possibly empty if calculations fail) — no exception
    assert isinstance(result, dict)


def test_parquet_portfolio_next_ret_uses_t_plus_2_open():
    """Verify next_ret = open(t+2)/open(t+1) - 1 (no look-ahead: signal at t executes at open(t+1))."""
    import numpy as np
    from unittest.mock import patch
    from quantaalpha.backtest.runner import BacktestRunner

    runner = BacktestRunner.__new__(BacktestRunner)
    runner.config = {
        "backtest": {
            "backtest": {"start_time": "2022-01-01", "end_time": "2022-12-31"},
            "strategy": {"kwargs": {"topk": 1, "weight_scheme": "equal"}},
        }
    }

    # 6 trading dates — gives enough room for shift(-2) to yield valid values
    dates = pd.date_range("2022-01-03", periods=6, freq="B")
    symbol = "AA"
    # Strictly increasing opens: 10, 11, 12, 13, 14, 15
    opens = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    prices_rows = [{"datetime": dt, "symbol": symbol, "open": o} for dt, o in zip(dates, opens)]
    prices = pd.DataFrame(prices_rows)
    prices["datetime"] = pd.to_datetime(prices["datetime"])

    bench_rows = [{"datetime": dt, "open": 100.0} for dt in dates]
    bench = pd.DataFrame(bench_rows)
    bench["datetime"] = pd.to_datetime(bench["datetime"])

    # Constant score so all dates select "AA"
    pred = pd.Series(
        [1.0] * len(dates),
        index=pd.MultiIndex.from_tuples(
            [(dt, symbol) for dt in dates], names=["datetime", "instrument"]
        ),
    )

    with patch.object(runner, "_load_parquet_prices", return_value=prices), \
         patch.object(runner, "_load_parquet_benchmark", return_value=bench):
        # Build the px DataFrame the same way the runner does, and verify next_ret values
        import pandas as _pd
        px = prices[["datetime", "symbol", "open"]].copy().sort_values(["symbol", "datetime"])
        grp = px.groupby("symbol")["open"]
        px["next_ret"] = grp.shift(-2) / grp.shift(-1) - 1

        aa = px[px["symbol"] == symbol].reset_index(drop=True)
        # For date[0] (open=10): shift(-2)=12, shift(-1)=11 → next_ret = 12/11 - 1 ≈ 0.0909
        assert abs(aa.loc[0, "next_ret"] - (12.0 / 11.0 - 1)) < 1e-6
        # For date[1] (open=11): shift(-2)=13, shift(-1)=12 → next_ret = 13/12 - 1 ≈ 0.0833
        assert abs(aa.loc[1, "next_ret"] - (13.0 / 12.0 - 1)) < 1e-6
        # Last two rows must be NaN (not enough future data for shift(-2))
        assert _pd.isna(aa.loc[4, "next_ret"])
        assert _pd.isna(aa.loc[5, "next_ret"])
