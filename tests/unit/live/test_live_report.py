"""Tests for quantaalpha.live.report — trading report renderer."""

from __future__ import annotations

import io

import pytest

from quantaalpha.live.report import render_report


def _base_data(**overrides):
    """Minimal valid order_data dict."""
    d = {
        "date": "2026-02-25",
        "as_of": "2026-02-25",
        "scores_count": 500,
        "daily_pnl": -500.0,
        "account_value": 1_000_000.0,
        "cash": 0.0,
        "previous_positions": {"AAPL": 100, "MSFT": 200},
        "prices": {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 2800.0},
        "benchmark_return": 0.001,
        "cumulative_pnl": -1500.0,
        "cumulative_excess_return": -0.002,
        "position_pnl": {
            "AAPL": {
                "shares": 100,
                "price_start": 151.0,
                "price_end": 150.0,
                "pnl": -100.0,
            },
            "MSFT": {
                "shares": 200,
                "price_start": 302.0,
                "price_end": 300.0,
                "pnl": -400.0,
            },
        },
        "orders": [],
        "target_positions": {"AAPL": 100, "MSFT": 200},
        "orders_file": "data/live/pending_orders_2026-02-25.json",
    }
    d.update(overrides)
    return d


class TestReportRenders:
    """Basic smoke tests: render_report produces output without errors."""

    def test_empty_orders(self):
        buf = io.StringIO()
        render_report(_base_data(), file=buf)
        text = buf.getvalue()
        assert "ACCOUNT SUMMARY" in text
        assert "TARGET PORTFOLIO" in text
        assert "HOLD" in text  # both prev positions are in target, no orders

    def test_sell_order(self):
        data = _base_data(
            orders=[
                {
                    "ticker": "AAPL",
                    "shares": -100,
                    "action": "sell",
                    "price": 150.0,
                    "reason": "dropped",
                }
            ],
            target_positions={"MSFT": 200},
        )
        buf = io.StringIO()
        render_report(data, file=buf)
        text = buf.getvalue()
        assert "SELL" in text
        assert "AAPL" in text
        assert "dropped" in text

    def test_buy_new_order(self):
        data = _base_data(
            orders=[
                {
                    "ticker": "GOOG",
                    "shares": 10,
                    "action": "buy",
                    "price": 2800.0,
                    "reason": "new entry",
                }
            ],
            target_positions={"AAPL": 100, "MSFT": 200, "GOOG": 10},
        )
        buf = io.StringIO()
        render_report(data, file=buf)
        text = buf.getvalue()
        assert "BUY NEW" in text
        assert "GOOG" in text

    def test_increase_order(self):
        data = _base_data(
            orders=[
                {
                    "ticker": "AAPL",
                    "shares": 50,
                    "action": "buy",
                    "price": 150.0,
                    "reason": "reweight",
                }
            ],
            target_positions={"AAPL": 150, "MSFT": 200},
        )
        buf = io.StringIO()
        render_report(data, file=buf)
        text = buf.getvalue()
        assert "INCREASE" in text
        assert "150" in text  # target shares

    def test_reduce_order(self):
        data = _base_data(
            orders=[
                {
                    "ticker": "AAPL",
                    "shares": -50,
                    "action": "sell",
                    "price": 150.0,
                    "reason": "reweight",
                }
            ],
            target_positions={"AAPL": 50, "MSFT": 200},
        )
        buf = io.StringIO()
        render_report(data, file=buf)
        text = buf.getvalue()
        assert "REDUCE" in text


class TestReportContent:
    """Verify specific fields appear in the report."""

    def test_account_value(self):
        buf = io.StringIO()
        render_report(_base_data(account_value=987_654.32), file=buf)
        assert "987,654.32" in buf.getvalue()

    def test_daily_pnl(self):
        buf = io.StringIO()
        render_report(_base_data(daily_pnl=-1234.56), file=buf)
        assert "1,234.56" in buf.getvalue()

    def test_benchmark_shown(self):
        buf = io.StringIO()
        render_report(_base_data(benchmark_return=0.0123), file=buf)
        assert "SPY" in buf.getvalue()

    def test_target_weights(self):
        buf = io.StringIO()
        data = _base_data(
            target_positions={"AAPL": 100},
            prices={"AAPL": 100.0},
            account_value=10_000.0,
        )
        render_report(data, file=buf)
        # 100 shares × $100 = $10,000 → 100.0% weight
        assert "100.0%" in buf.getvalue()

    def test_cash_displayed(self):
        buf = io.StringIO()
        render_report(_base_data(cash=50_000.0), file=buf)
        assert "50,000" in buf.getvalue()

    def test_per_position_pnl_sorted(self):
        buf = io.StringIO()
        render_report(_base_data(), file=buf)
        text = buf.getvalue()
        # MSFT had -400, AAPL -100; worst first
        msft_pos = text.index("MSFT")
        aapl_pos = text.index("AAPL")
        # In the P&L section, MSFT (worse) should appear before AAPL
        # Find positions in the P&L section specifically
        pnl_section = text[text.index("POSITION P&L") :]
        assert pnl_section.index("MSFT") < pnl_section.index("AAPL")


class TestEdgeCases:
    """Edge cases and empty data."""

    def test_no_positions(self):
        buf = io.StringIO()
        data = _base_data(
            previous_positions={},
            position_pnl={},
            target_positions={},
            orders=[],
        )
        render_report(data, file=buf)
        text = buf.getvalue()
        assert "NO TRADES" in text

    def test_zero_account_value(self):
        buf = io.StringIO()
        data = _base_data(account_value=0)
        render_report(data, file=buf)
        # Should not crash on division by zero
        assert "ACCOUNT SUMMARY" in buf.getvalue()

    def test_mixed_actions(self):
        """Report with sell, buy new, and increase all at once."""
        data = _base_data(
            previous_positions={"AAPL": 100, "MSFT": 200, "TSLA": 50},
            prices={"AAPL": 150.0, "MSFT": 300.0, "GOOG": 2800.0, "TSLA": 250.0},
            orders=[
                {
                    "ticker": "TSLA",
                    "shares": -50,
                    "action": "sell",
                    "price": 250.0,
                    "reason": "dropped",
                },
                {
                    "ticker": "GOOG",
                    "shares": 10,
                    "action": "buy",
                    "price": 2800.0,
                    "reason": "new entry",
                },
                {
                    "ticker": "AAPL",
                    "shares": 50,
                    "action": "buy",
                    "price": 150.0,
                    "reason": "reweight",
                },
            ],
            target_positions={"AAPL": 150, "MSFT": 200, "GOOG": 10},
        )
        buf = io.StringIO()
        render_report(data, file=buf)
        text = buf.getvalue()
        assert "SELL" in text
        assert "BUY NEW" in text
        assert "INCREASE" in text
        assert "HOLD" in text  # MSFT should be held
