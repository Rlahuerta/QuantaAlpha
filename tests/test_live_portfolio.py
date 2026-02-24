"""Tests for live trading portfolio management.

Covers:
- PositionTracker idempotency (Bug #1)
- Cash tracking (Bug #2) — account_value = prev + daily_pnl; cash is residual
- num_positions correctness (Bug #3)
- TopkDropout n_drop enforcement (Bug #4)
- Multi-day P&L accumulation consistency
- Model switch flow
"""

import json
import tempfile
from datetime import date
from pathlib import Path

import pytest

from quantaalpha.live.position_tracker import PositionTracker
from quantaalpha.live.portfolio_constructor import PortfolioConstructor, Order


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_live_dir(tmp_path):
    """Create a temp directory with positions.json and pnl/ subdir."""
    positions_file = tmp_path / "positions.json"
    pnl_dir = tmp_path / "pnl"
    pnl_dir.mkdir()
    return positions_file, pnl_dir


@pytest.fixture
def tracker(tmp_live_dir):
    positions_file, pnl_dir = tmp_live_dir
    return PositionTracker(positions_file=positions_file, pnl_dir=pnl_dir)


@pytest.fixture
def sample_positions():
    return {"AAPL": 100, "MSFT": 80, "GOOG": 50}


@pytest.fixture
def sample_prices():
    return {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 150.0}


# ---------------------------------------------------------------------------
# Bug #1: Idempotency — same-day re-run must NOT double-count P&L
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_same_day_rerun_no_double_count(self, tracker, sample_positions, sample_prices):
        """Running record_day twice on the same date must not double cumulative P&L."""
        pnl_dict = {"daily_pnl": 1000.0, "daily_excess_return": 0.005, "portfolio_return": 0.005}

        # First run
        state1 = tracker.record_day(
            positions=sample_positions,
            account_value=101_000,
            daily_pnl_dict=pnl_dict,
            as_of=date(2026, 2, 20),
            prices=sample_prices,
        )
        cum1 = state1["pnl"]["cumulative_pnl"]

        # Second run (same date) — must NOT add another 1000
        state2 = tracker.record_day(
            positions=sample_positions,
            account_value=101_000,
            daily_pnl_dict=pnl_dict,
            as_of=date(2026, 2, 20),
            prices=sample_prices,
        )
        cum2 = state2["pnl"]["cumulative_pnl"]

        assert cum1 == cum2, f"Cumulative P&L doubled: {cum1} → {cum2}"

    def test_same_day_rerun_preserves_cumulative_excess(self, tracker):
        """Cumulative excess return must be identical on same-day re-run."""
        prices = {"AAPL": 200.0}
        pos = {"AAPL": 100}
        pnl = {"daily_pnl": 500, "daily_excess_return": 0.003, "portfolio_return": 0.003}

        state1 = tracker.record_day(pos, 50_500, pnl, as_of=date(2026, 2, 20), prices=prices)
        state2 = tracker.record_day(pos, 50_500, pnl, as_of=date(2026, 2, 20), prices=prices)

        assert state1["pnl"]["cumulative_excess_return"] == state2["pnl"]["cumulative_excess_return"]

    def test_different_day_does_accumulate(self, tracker):
        """P&L on a NEW date must accumulate normally."""
        prices = {"AAPL": 200.0}
        pos = {"AAPL": 100}
        pnl = {"daily_pnl": 1000, "daily_excess_return": 0.01, "portfolio_return": 0.01}

        tracker.record_day(pos, 101_000, pnl, as_of=date(2026, 2, 20), prices=prices)
        state2 = tracker.record_day(pos, 102_000, pnl, as_of=date(2026, 2, 21), prices=prices)

        assert state2["pnl"]["cumulative_pnl"] == 2000.0
        assert state2["pnl"]["cumulative_excess_return"] == 0.02

    def test_triple_run_same_day(self, tracker):
        """Three runs on the same day must produce same cumulative as one run."""
        prices = {"SPY": 500.0}
        pos = {"SPY": 200}
        pnl = {"daily_pnl": 2000, "daily_excess_return": 0.02, "portfolio_return": 0.02}

        s1 = tracker.record_day(pos, 102_000, pnl, as_of=date(2026, 3, 1), prices=prices)
        s2 = tracker.record_day(pos, 102_000, pnl, as_of=date(2026, 3, 1), prices=prices)
        s3 = tracker.record_day(pos, 102_000, pnl, as_of=date(2026, 3, 1), prices=prices)

        assert s1["pnl"]["cumulative_pnl"] == s2["pnl"]["cumulative_pnl"] == s3["pnl"]["cumulative_pnl"]


# ---------------------------------------------------------------------------
# Bug #2: Cash tracking — account_value is preserved; cash is residual
# ---------------------------------------------------------------------------

class TestCashTracking:
    def test_cash_is_residual(self, tracker):
        """Cash = account_value − mark_to_market when prices are given."""
        positions = {"AAPL": 100, "MSFT": 50}
        prices = {"AAPL": 200.0, "MSFT": 400.0}
        pnl = {"daily_pnl": 0, "daily_excess_return": 0, "portfolio_return": 0}

        account_value = 100_000.0
        mtm = 100 * 200 + 50 * 400  # 40,000
        state = tracker.record_day(positions, account_value, pnl,
                                   as_of=date(2026, 2, 20), prices=prices)

        assert state["account_value"] == account_value
        assert state["cash"] == account_value - mtm  # 60,000

    def test_account_value_not_overridden_by_mtm(self, tracker):
        """account_value should be the passed value, not mark-to-market."""
        positions = {"GOOG": 10}
        prices = {"GOOG": 3000.0}
        pnl = {"daily_pnl": 0, "daily_excess_return": 0, "portfolio_return": 0}

        state = tracker.record_day(positions, 50_000, pnl,
                                   as_of=date(2026, 2, 20), prices=prices)
        assert state["account_value"] == 50_000
        assert state["cash"] == 50_000 - 30_000  # 20,000

    def test_no_prices_zero_cash(self, tracker):
        """When prices not provided, cash defaults to 0."""
        positions = {"AAPL": 100}
        pnl = {"daily_pnl": 0, "daily_excess_return": 0, "portfolio_return": 0}

        state = tracker.record_day(positions, 50_000, pnl, as_of=date(2026, 2, 20))
        assert state["account_value"] == 50_000
        assert state["cash"] == 0.0

    def test_mark_to_market_method(self, tracker):
        """Test the mark_to_market helper directly."""
        positions = {"A": 100, "B": 200, "C": 50}
        prices = {"A": 10.0, "B": 20.0, "C": 30.0}
        assert tracker.mark_to_market(positions, prices) == 100*10 + 200*20 + 50*30

    def test_mark_to_market_missing_price(self, tracker):
        """Tickers without prices contribute 0."""
        positions = {"A": 100, "B": 200}
        prices = {"A": 10.0}  # B missing
        assert tracker.mark_to_market(positions, prices) == 1000.0

    def test_fully_invested_zero_cash(self, tracker):
        """When all capital is in positions, cash ≈ 0."""
        positions = {"X": 100}
        prices = {"X": 1000.0}
        pnl = {"daily_pnl": 0, "daily_excess_return": 0, "portfolio_return": 0}

        state = tracker.record_day(positions, 100_000, pnl,
                                   as_of=date(2026, 2, 20), prices=prices)
        assert state["cash"] == 0.0


# ---------------------------------------------------------------------------
# Bug #3: num_positions must match actual holdings
# ---------------------------------------------------------------------------

class TestNumPositions:
    def test_num_positions_matches_holdings(self, tracker):
        """num_positions field must equal len(positions), not price-counted."""
        positions = {"AAPL": 100, "MSFT": 80, "GOOG": 50}
        prices = {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 150.0}
        pnl = {"daily_pnl": 0, "daily_excess_return": 0, "num_positions": 2}

        state = tracker.record_day(positions, 100_000, pnl,
                                   as_of=date(2026, 2, 20), prices=prices)
        assert state["pnl"]["num_positions"] == 3  # actual count, not 2

    def test_empty_positions(self, tracker):
        """num_positions is 0 for empty portfolio."""
        pnl = {"daily_pnl": 0, "daily_excess_return": 0}
        state = tracker.record_day({}, 100_000, pnl, as_of=date(2026, 2, 20), prices={})
        assert state["pnl"]["num_positions"] == 0

    def test_ten_positions(self, tracker):
        """10 positions yields num_positions=10."""
        positions = {f"T{i}": 100 for i in range(10)}
        prices = {f"T{i}": 50.0 for i in range(10)}
        pnl = {"daily_pnl": 0, "daily_excess_return": 0}
        state = tracker.record_day(positions, 100_000, pnl,
                                   as_of=date(2026, 2, 20), prices=prices)
        assert state["pnl"]["num_positions"] == 10


# ---------------------------------------------------------------------------
# Bug #4: TopkDropout must respect n_drop when topk decreases
# ---------------------------------------------------------------------------

class TestTopkDropout:
    def _make_scores(self, tickers, base=1.0):
        """Generate descending scores for tickers."""
        return {t: base - i * 0.01 for i, t in enumerate(tickers)}

    def test_normal_rebalance_respects_n_drop(self):
        """Standard case: at most n_drop stocks exit per rebalance."""
        pc = PortfolioConstructor(topk=5, n_drop=1, capital=100_000)
        old = ["A", "B", "C", "D", "E"]
        # New scores: F is top, E drops out
        scores = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "F": 0.95, "E": 0.1}
        target, dropped = pc._select_topk_with_dropout(scores, old)
        assert len(dropped) <= 1

    def test_topk_decrease_respects_n_drop(self):
        """When topk shrinks from 10→5, should NOT drop >n_drop positions at once."""
        pc = PortfolioConstructor(topk=5, n_drop=2, capital=100_000)
        old_holdings = [f"S{i}" for i in range(10)]  # currently holding 10
        # New scores: S0-S4 are top-5, S5-S9 should be dropped
        scores = {f"S{i}": 1.0 - i * 0.1 for i in range(10)}
        scores.update({f"N{i}": 0.5 + i * 0.01 for i in range(5)})  # new candidates

        target, dropped = pc._select_topk_with_dropout(scores, old_holdings)

        # With n_drop=2, we should only drop at most 2 per day
        assert len(dropped) <= 2
        # Total positions should still include old ones minus dropped
        remaining_old = set(old_holdings) - set(dropped)
        assert remaining_old.issubset(set(target))

    def test_topk_decrease_30_to_10(self):
        """Simulate the exact production bug: 30→10 positions with n_drop=1."""
        pc = PortfolioConstructor(topk=10, n_drop=1, capital=1_000_000)
        old_holdings = [f"OLD{i}" for i in range(30)]
        # All old holdings score low, new top-10 are different tickers
        scores = {}
        for i in range(30):
            scores[f"OLD{i}"] = 0.1 - i * 0.001  # all low
        for i in range(10):
            scores[f"NEW{i}"] = 0.9 - i * 0.01  # all high

        target, dropped = pc._select_topk_with_dropout(scores, old_holdings)

        # Key assertion: should NOT drop all 30 old positions at once
        assert len(dropped) <= 1
        # Should still be holding most old positions
        remaining_old = len([t for t in target if t.startswith("OLD")])
        assert remaining_old >= 28  # at most 1 dropped + maybe 1 added

    def test_same_topk_only_drops_n_drop(self):
        """With stable topk, exactly n_drop exits."""
        pc = PortfolioConstructor(topk=5, n_drop=2, capital=100_000)
        old = ["A", "B", "C", "D", "E"]
        # A-C still good, D-E fall out, F-G enter
        scores = {"A": 0.9, "B": 0.8, "C": 0.7, "F": 0.85, "G": 0.75, "D": 0.2, "E": 0.1}
        target, dropped = pc._select_topk_with_dropout(scores, old)
        assert len(dropped) <= 2

    def test_no_holdings_fills_topk(self):
        """Fresh start (no holdings) should create exactly topk positions."""
        pc = PortfolioConstructor(topk=10, n_drop=2, capital=100_000)
        scores = {f"T{i}": 1.0 - i * 0.05 for i in range(20)}
        target, dropped = pc._select_topk_with_dropout(scores, [])
        assert len(target) == 10
        assert len(dropped) == 0

    def test_rebalance_order_generation(self):
        """Full rebalance produces valid orders for new entries and exits."""
        pc = PortfolioConstructor(topk=3, n_drop=1, capital=30_000)
        scores = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.1}
        positions = {"A": 100, "B": 100, "D": 100}
        prices = {"A": 100, "B": 100, "C": 100, "D": 100}

        result = pc.rebalance(scores, positions, prices)
        actions = {o.ticker: o.action for o in result.orders}

        assert "D" in actions and actions["D"] == "sell"
        assert "C" in actions and actions["C"] == "buy"


# ---------------------------------------------------------------------------
# Multi-day P&L consistency
# ---------------------------------------------------------------------------

class TestMultiDayPnL:
    def test_three_day_cumulative(self, tracker):
        """Cumulative P&L after 3 days must equal sum of daily P&Ls."""
        prices = {"X": 100.0}
        pos = {"X": 100}
        daily_pnls = [1000, -500, 200]
        account = 100_000.0

        for i, dp in enumerate(daily_pnls):
            account += dp
            pnl = {"daily_pnl": dp, "daily_excess_return": dp / 10_000, "portfolio_return": dp / 10_000}
            state = tracker.record_day(pos, account, pnl, as_of=date(2026, 3, i + 1), prices=prices)

        assert state["pnl"]["cumulative_pnl"] == sum(daily_pnls)

    def test_five_day_with_reruns(self, tracker):
        """Multi-day with some same-day re-runs must still be consistent."""
        prices = {"Y": 50.0}
        pos = {"Y": 200}
        expected_cum = 0
        account = 100_000.0

        for day in range(1, 6):
            dp = 100 * day
            expected_cum += dp
            account += dp
            pnl = {"daily_pnl": dp, "daily_excess_return": 0.001 * day, "portfolio_return": 0.001 * day}
            # Run each day twice to stress idempotency
            tracker.record_day(pos, account, pnl, as_of=date(2026, 4, day), prices=prices)
            state = tracker.record_day(pos, account, pnl, as_of=date(2026, 4, day), prices=prices)

        assert state["pnl"]["cumulative_pnl"] == expected_cum

    def test_cash_tracked_across_days(self, tracker):
        """Cash field is recomputed each day from account_value − mtm."""
        for day, (price, acct) in enumerate([(100.0, 150_000), (110.0, 152_000), (95.0, 147_000)], 1):
            prices = {"Z": price}
            pos = {"Z": 100}
            pnl = {"daily_pnl": 0, "daily_excess_return": 0, "portfolio_return": 0}
            state = tracker.record_day(pos, acct, pnl, as_of=date(2026, 5, day), prices=prices)
            assert state["account_value"] == acct
            assert state["cash"] == acct - 100 * price


# ---------------------------------------------------------------------------
# Model switch / position reset flow
# ---------------------------------------------------------------------------

class TestModelSwitch:
    def test_complete_portfolio_change_preserves_cumulative(self, tracker):
        """Switching all positions should not corrupt cumulative P&L."""
        prices_old = {"OLD1": 100, "OLD2": 200}
        prices_new = {"NEW1": 150, "NEW2": 250}

        # Day 1: old portfolio
        pnl1 = {"daily_pnl": 500, "daily_excess_return": 0.005, "portfolio_return": 0.005}
        tracker.record_day(
            {"OLD1": 100, "OLD2": 50}, 100_500, pnl1,
            as_of=date(2026, 6, 1), prices=prices_old,
        )

        # Day 2: complete switch to new portfolio
        pnl2 = {"daily_pnl": -200, "daily_excess_return": -0.002, "portfolio_return": -0.002}
        state = tracker.record_day(
            {"NEW1": 80, "NEW2": 40}, 100_300, pnl2,
            as_of=date(2026, 6, 2), prices=prices_new,
        )

        # Cumulative should be 500 + (-200) = 300
        assert state["pnl"]["cumulative_pnl"] == 300.0
        # Account value is preserved (not overridden by mtm)
        assert state["account_value"] == 100_300
        # Cash = account - mtm(80×150 + 40×250 = 22,000)
        assert state["cash"] == 100_300 - (80 * 150 + 40 * 250)


# ---------------------------------------------------------------------------
# P&L history files
# ---------------------------------------------------------------------------

class TestPnLHistory:
    def test_history_files_created(self, tracker):
        """Each day creates a pnl_YYYY-MM-DD.json file."""
        prices = {"A": 100}
        pos = {"A": 10}
        pnl = {"daily_pnl": 0, "daily_excess_return": 0, "portfolio_return": 0}

        tracker.record_day(pos, 100_000, pnl, as_of=date(2026, 7, 1), prices=prices)
        tracker.record_day(pos, 100_000, pnl, as_of=date(2026, 7, 2), prices=prices)

        files = list(tracker.pnl_dir.glob("pnl_*.json"))
        assert len(files) == 2

    def test_history_overwritten_on_rerun(self, tracker):
        """Same-day re-run overwrites the history file (not appends)."""
        prices = {"A": 100}
        pos = {"A": 10}
        pnl = {"daily_pnl": 500, "daily_excess_return": 0.005, "portfolio_return": 0.005}

        tracker.record_day(pos, 100_500, pnl, as_of=date(2026, 7, 1), prices=prices)
        tracker.record_day(pos, 100_500, pnl, as_of=date(2026, 7, 1), prices=prices)

        files = list(tracker.pnl_dir.glob("pnl_2026-07-01.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["pnl"]["cumulative_pnl"] == 500.0  # NOT 1000
