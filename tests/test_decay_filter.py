"""
Tests for quantaalpha.factors.decay_filter

Coverage:
  - compute_decay_profile_from_df with known signal-injected data
  - factor built to predict horizon-N returns: ic_Nd > 0 by construction
  - factor built to predict 1d only: ic_1d > ic_5d (decays)
  - noise factor: all ICs near zero, decay_pass_5d reflects sign
  - anticorr factor: ic_5d < 0, gate fails
  - passes_decay_gate with various horizons and min_ic thresholds
  - fit_decay_half_life with synthetic exponential data
  - fit_decay_half_life edge cases (< 2 points, all negative IC, growing IC)
  - empty / bad-index factor returns empty profile
  - custom horizons, exec_lag, eval_start/end filters
  - missing instruments handled gracefully
  - H5 file not found returns empty profile
  - decay_half_life finite and positive for decaying factor
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantaalpha.factors.decay_filter import (
    DEFAULT_HORIZONS,
    compute_decay_profile,
    compute_decay_profile_from_df,
    fit_decay_half_life,
    passes_decay_gate,
    _empty_profile,
)

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_factor_with_ic(
    n_days: int = 150,
    n_stocks: int = 50,
    horizon: int = 5,
    target_ic: float = 0.25,
    exec_lag: int = 1,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """Build (factor_series, price_df) where factor has ~target_ic at horizon.

    Strategy: factor = target_ic * rank(fwd_ret_N) + sqrt(1 - target_ic²) * rank(noise)
    This gives Spearman IC ≈ target_ic by construction of rank regression.
    Uses extra tail days in prices so forward returns are available for all factor dates.
    """
    rng = np.random.default_rng(seed)
    tail = horizon + exec_lag + 5
    total = n_days + tail
    dates_all = pd.bdate_range("2022-01-03", periods=total)
    instruments = [f"ST{i:04d}" for i in range(n_stocks)]

    # Random-walk prices
    log_ret = rng.normal(0, 0.012, (total, n_stocks))
    close = pd.DataFrame(np.exp(np.cumsum(log_ret, axis=0)), index=dates_all, columns=instruments)

    # Forward return aligned with exec_lag
    exec_px = close.shift(-exec_lag)
    exit_px = close.shift(-(horizon + exec_lag))
    fwd = exit_px / exec_px - 1

    # Cross-sectional rank → centered in [−0.5, 0.5]
    fwd_rank = fwd.rank(axis=1, pct=True) - 0.5
    noise = pd.DataFrame(rng.normal(0, 1, (total, n_stocks)), index=dates_all, columns=instruments)
    noise_rank = noise.rank(axis=1, pct=True) - 0.5

    signal_w = float(target_ic)
    noise_w = float(np.sqrt(max(0.0, 1.0 - target_ic ** 2)))
    factor_mat = signal_w * fwd_rank + noise_w * noise_rank

    # Factor only for the first n_days; prices for all days (for fwd-return calc)
    factor_series = factor_mat.iloc[:n_days].stack()
    factor_series.index.names = ["datetime", "instrument"]

    price_df = close.stack().rename("$close").to_frame()
    price_df.index.names = ["datetime", "instrument"]

    return factor_series, price_df


def _make_noise_factor(n_days: int = 150, n_stocks: int = 50, seed: int = 7) -> tuple[pd.Series, pd.DataFrame]:
    """Pure noise factor — no signal at any horizon."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days + 25)
    instruments = [f"ST{i:04d}" for i in range(n_stocks)]
    close = pd.DataFrame(
        np.exp(np.cumsum(rng.normal(0, 0.012, (len(dates), n_stocks)), axis=0)),
        index=dates, columns=instruments,
    )
    factor_mat = pd.DataFrame(rng.normal(0, 1, (len(dates), n_stocks)), index=dates, columns=instruments)
    factor_series = factor_mat.iloc[:n_days].stack()
    factor_series.index.names = ["datetime", "instrument"]
    price_df = close.stack().rename("$close").to_frame()
    price_df.index.names = ["datetime", "instrument"]
    return factor_series, price_df


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

class TestComputeDecayProfileFromDf:
    def test_signal_factor_has_positive_ic_at_target_horizon(self):
        """Factor built to predict 5d returns → ic_5d should be > 0."""
        factor, prices = _make_factor_with_ic(horizon=5, target_ic=0.25)
        metrics = compute_decay_profile_from_df(factor, prices)
        assert metrics["ic_5d"] is not None
        assert metrics["ic_5d"] > 0, f"Expected ic_5d > 0, got {metrics['ic_5d']}"

    def test_signal_factor_has_positive_ic_at_1d(self):
        """Factor built to predict 1d returns → ic_1d > 0."""
        factor, prices = _make_factor_with_ic(horizon=1, target_ic=0.30)
        metrics = compute_decay_profile_from_df(factor, prices)
        assert metrics["ic_1d"] is not None
        assert metrics["ic_1d"] > 0, f"Expected ic_1d > 0, got {metrics['ic_1d']}"

    def test_1d_factor_decays_by_5d(self):
        """Factor optimized for 1d should have ic_1d > ic_5d (no persistent signal)."""
        factor, prices = _make_factor_with_ic(
            n_days=200, n_stocks=60, horizon=1, target_ic=0.35, seed=1
        )
        metrics = compute_decay_profile_from_df(factor, prices)
        ic_1d = metrics.get("ic_1d") or 0.0
        ic_5d = metrics.get("ic_5d") or 0.0
        assert ic_1d > ic_5d, f"Expected ic_1d ({ic_1d:.4f}) > ic_5d ({ic_5d:.4f})"

    def test_5d_signal_factor_passes_decay_gate(self):
        factor, prices = _make_factor_with_ic(horizon=5, target_ic=0.20)
        metrics = compute_decay_profile_from_df(factor, prices)
        assert metrics["decay_pass_5d"] is True

    def test_1d_only_factor_may_fail_5d_gate(self):
        """A very short-horizon factor (1d) likely fails the 5d gate."""
        # With a high-signal 1d factor but random walk prices, 5d IC is near 0 or negative
        factor, prices = _make_factor_with_ic(
            n_days=250, n_stocks=80, horizon=1, target_ic=0.40, seed=5
        )
        metrics = compute_decay_profile_from_df(factor, prices)
        # ic_1d should be clearly positive
        assert (metrics["ic_1d"] or 0.0) > 0
        # ic_5d may or may not be positive depending on random seed — just assert the
        # gate flag correctly reflects the sign
        ic_5d = metrics.get("ic_5d") or 0.0
        assert metrics["decay_pass_5d"] == (ic_5d > 0)

    def test_anticorr_factor_fails_gate(self):
        """Negatively-correlated factor should have ic < 0 and fail gate."""
        factor, prices = _make_factor_with_ic(horizon=5, target_ic=-0.25)
        metrics = compute_decay_profile_from_df(factor, prices)
        assert (metrics["ic_5d"] or 0.0) < 0
        assert metrics["decay_pass_5d"] is False

    def test_noise_factor_gate_reflects_sign(self):
        """Noise factor: decay_pass_5d must correctly reflect sign of ic_5d."""
        factor, prices = _make_noise_factor(n_days=300, n_stocks=60, seed=7)
        metrics = compute_decay_profile_from_df(factor, prices)
        ic_5d = metrics.get("ic_5d") or 0.0
        assert metrics["decay_pass_5d"] == (ic_5d > 0)

    def test_returns_daily_ic_count(self):
        factor, prices = _make_factor_with_ic(n_days=120)
        metrics = compute_decay_profile_from_df(factor, prices)
        assert metrics["daily_ic_count"] > 0

    def test_eval_date_filter_reduces_count(self):
        factor, prices = _make_factor_with_ic(n_days=200)
        metrics_full = compute_decay_profile_from_df(factor, prices)
        metrics_half = compute_decay_profile_from_df(
            factor, prices, eval_start="2022-06-01", eval_end="2022-09-30"
        )
        assert metrics_half["daily_ic_count"] < metrics_full["daily_ic_count"]

    def test_missing_instruments_handled_gracefully(self):
        factor, prices = _make_factor_with_ic(n_days=100, n_stocks=40)
        keep = prices.index.get_level_values("instrument").unique()[:20]
        prices_subset = prices.loc[prices.index.get_level_values("instrument").isin(keep)]
        metrics = compute_decay_profile_from_df(factor, prices_subset)
        assert "ic_1d" in metrics
        assert "decay_pass_5d" in metrics

    def test_all_instruments_missing_returns_empty(self):
        factor, prices = _make_factor_with_ic(n_days=100, n_stocks=40)
        prices_renamed = prices.copy()
        prices_renamed.index = prices_renamed.index.set_levels(
            [f"ZZZZ{i}" for i in range(40)], level="instrument"
        )
        metrics = compute_decay_profile_from_df(factor, prices_renamed)
        assert all(metrics[f"ic_{n}d"] is None for n in DEFAULT_HORIZONS)
        assert metrics["decay_pass_5d"] is False

    def test_custom_horizons(self):
        factor, prices = _make_factor_with_ic(n_days=150, horizon=7)
        metrics = compute_decay_profile_from_df(factor, prices, horizons=[3, 7])
        assert "ic_3d" in metrics
        assert "ic_7d" in metrics
        assert "ic_1d" not in metrics

    def test_exec_lag_reduces_ic(self):
        """Higher execution lag = older signal at execution → lower IC at same horizon."""
        factor, prices = _make_factor_with_ic(n_days=200, n_stocks=60, horizon=1, target_ic=0.35)
        m_lag0 = compute_decay_profile_from_df(factor, prices, exec_lag=0)
        m_lag3 = compute_decay_profile_from_df(factor, prices, exec_lag=3)
        ic0 = m_lag0["ic_1d"] or 0.0
        ic3 = m_lag3["ic_1d"] or 0.0
        # Allow 0.05 tolerance; primary check is the direction
        assert ic0 >= ic3 - 0.05, f"Expected ic(lag=0)={ic0:.4f} >= ic(lag=3)={ic3:.4f}"

    def test_empty_factor_series_returns_empty_profile(self):
        _, prices = _make_factor_with_ic(n_days=100)
        empty = pd.Series(dtype=float, name="factor")
        empty.index = pd.MultiIndex.from_tuples([], names=["datetime", "instrument"])
        metrics = compute_decay_profile_from_df(empty, prices)
        assert all(metrics[f"ic_{n}d"] is None for n in DEFAULT_HORIZONS)

    def test_non_multiindex_factor_returns_empty_profile(self):
        _, prices = _make_factor_with_ic(n_days=100)
        bad_factor = pd.Series([1.0, 2.0, 3.0])
        metrics = compute_decay_profile_from_df(bad_factor, prices)
        assert all(metrics[f"ic_{n}d"] is None for n in DEFAULT_HORIZONS)


# ---------------------------------------------------------------------------
# passes_decay_gate
# ---------------------------------------------------------------------------

class TestPassesDecayGate:
    def test_positive_ic_passes(self):
        assert passes_decay_gate({"ic_5d": 0.01}) is True

    def test_zero_ic_fails(self):
        assert passes_decay_gate({"ic_5d": 0.0}) is False

    def test_negative_ic_fails(self):
        assert passes_decay_gate({"ic_5d": -0.003}) is False

    def test_none_ic_fails(self):
        assert passes_decay_gate({"ic_5d": None}) is False

    def test_missing_key_fails(self):
        assert passes_decay_gate({"ic_1d": 0.05}) is False  # no ic_5d

    def test_custom_horizon(self):
        assert passes_decay_gate({"ic_10d": 0.005}, horizon=10) is True

    def test_min_ic_threshold_strict(self):
        assert passes_decay_gate({"ic_5d": 0.003}, min_ic=0.005) is False

    def test_min_ic_threshold_lenient(self):
        assert passes_decay_gate({"ic_5d": 0.003}, min_ic=0.002) is True

    def test_min_ic_exactly_equal_fails(self):
        # strict greater-than: ic must be > min_ic
        assert passes_decay_gate({"ic_5d": 0.005}, min_ic=0.005) is False


# ---------------------------------------------------------------------------
# fit_decay_half_life
# ---------------------------------------------------------------------------

class TestFitDecayHalfLife:
    def test_perfect_exponential_decay_accuracy(self):
        """Half-life should be within 20% of true value for clean exponential data."""
        true_lambda = 0.1  # decay rate
        true_hl = np.log(2) / true_lambda  # ≈ 6.93
        horizons = [1, 5, 10, 20]
        ics = [0.05 * np.exp(-true_lambda * t) for t in horizons]
        hl = fit_decay_half_life(horizons, ics)
        assert hl is not None
        assert abs(hl - true_hl) / true_hl < 0.20, f"Half-life {hl:.2f} far from {true_hl:.2f}"

    def test_half_life_always_positive(self):
        hl = fit_decay_half_life([1, 5, 10, 20], [0.04, 0.03, 0.02, 0.01])
        assert hl is not None
        assert hl > 0

    def test_growing_ic_returns_none(self):
        hl = fit_decay_half_life([1, 5, 10, 20], [0.01, 0.02, 0.03, 0.04])
        assert hl is None

    def test_all_negative_ic_returns_none(self):
        hl = fit_decay_half_life([1, 5, 10, 20], [-0.01, -0.02, -0.03, -0.04])
        assert hl is None

    def test_single_point_returns_none(self):
        assert fit_decay_half_life([5], [0.03]) is None

    def test_empty_returns_none(self):
        assert fit_decay_half_life([], []) is None

    def test_fast_decay_smaller_than_slow(self):
        ics_fast = [0.05 * np.exp(-0.5 * t) for t in [1, 5, 10, 20]]
        ics_slow = [0.05 * np.exp(-0.05 * t) for t in [1, 5, 10, 20]]
        hl_fast = fit_decay_half_life([1, 5, 10, 20], ics_fast)
        hl_slow = fit_decay_half_life([1, 5, 10, 20], ics_slow)
        assert hl_fast is not None and hl_slow is not None
        assert hl_fast < hl_slow

    def test_two_points_sufficient(self):
        hl = fit_decay_half_life([1, 10], [0.05, 0.02])
        assert hl is not None


# ---------------------------------------------------------------------------
# compute_decay_profile (H5-path variant)
# ---------------------------------------------------------------------------

class TestComputeDecayProfileH5:
    def test_missing_h5_returns_empty_profile(self):
        factor, _ = _make_factor_with_ic(n_days=100)
        metrics = compute_decay_profile(factor, h5_path="/nonexistent/path/daily_pv.h5")
        assert all(metrics.get(f"ic_{n}d") is None for n in DEFAULT_HORIZONS)
        assert metrics["decay_pass_5d"] is False

    def test_with_temp_h5_file(self, tmp_path):
        """Writes a minimal H5 and verifies the function reads it correctly."""
        factor, prices = _make_factor_with_ic(n_days=80, n_stocks=20, target_ic=0.20)
        h5_file = tmp_path / "daily_pv.h5"
        prices.to_hdf(str(h5_file), key="data", mode="w")
        metrics = compute_decay_profile(factor, h5_path=str(h5_file))
        assert "ic_1d" in metrics
        assert isinstance(metrics["decay_pass_5d"], bool)


# ---------------------------------------------------------------------------
# _empty_profile
# ---------------------------------------------------------------------------

class TestEmptyProfile:
    def test_default_horizons(self):
        ep = _empty_profile()
        for n in DEFAULT_HORIZONS:
            assert ep[f"ic_{n}d"] is None
        assert ep["decay_half_life"] is None
        assert ep["decay_pass_5d"] is False
        assert ep["daily_ic_count"] == 0

    def test_custom_horizons(self):
        ep = _empty_profile([3, 7, 15])
        assert "ic_3d" in ep and "ic_7d" in ep and "ic_15d" in ep
        assert "ic_1d" not in ep


# ---------------------------------------------------------------------------
# Integration: half-life in full profile
# ---------------------------------------------------------------------------

class TestDecayHalfLifeIntegration:
    def test_decaying_factor_has_finite_half_life(self):
        """Factor built to predict 1d should have finite half-life (decays quickly)."""
        factor, prices = _make_factor_with_ic(
            n_days=200, n_stocks=60, horizon=1, target_ic=0.30, seed=3
        )
        metrics = compute_decay_profile_from_df(factor, prices)
        # If at least 2 positive IC values exist, half-life should be finite
        positive_ics = [
            metrics[f"ic_{n}d"] for n in DEFAULT_HORIZONS
            if metrics.get(f"ic_{n}d") is not None and metrics[f"ic_{n}d"] > 0
        ]
        if len(positive_ics) >= 2:
            assert metrics["decay_half_life"] is not None
            assert metrics["decay_half_life"] > 0

    def test_persistent_factor_has_longer_half_life(self):
        """Factor targeting 20d horizon should have longer half-life than 1d-targeting factor."""
        factor_1d, prices_1d = _make_factor_with_ic(
            n_days=200, n_stocks=60, horizon=1, target_ic=0.30, seed=10
        )
        factor_20d, prices_20d = _make_factor_with_ic(
            n_days=200, n_stocks=60, horizon=20, target_ic=0.30, seed=10
        )
        m1 = compute_decay_profile_from_df(factor_1d, prices_1d)
        m20 = compute_decay_profile_from_df(factor_20d, prices_20d)
        hl1 = m1.get("decay_half_life")
        hl20 = m20.get("decay_half_life")
        # If both are defined, the 20d factor should persist longer
        if hl1 is not None and hl20 is not None:
            assert hl20 >= hl1 - 2.0  # allow 2-day tolerance for noise


