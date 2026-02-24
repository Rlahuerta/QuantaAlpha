"""Unit tests for quantaalpha.live.report_md — Markdown report generator.

Tests cover:
- Normal trading day with SELL + BUY orders
- Hold-only day (no orders)
- Kill-switch day
- Missing / empty data fields (backward compat)
- Performance history section with ledger rows
- Report file persistence (save_report)
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures — minimal order_data dicts
# ---------------------------------------------------------------------------

BASE_DATA: dict = {
    "date": "2026-02-24",
    "initial_capital": 1_000_000.0,
    "scores_count": 508,
    "daily_pnl": -381.99,
    "account_value": 989_645.73,
    "cash": -9_922.81,
    "benchmark_return": 0.0012,
    "cumulative_pnl": -8_183.47,
    "cumulative_excess_return": -0.008973,
    "previous_positions": {
        "GEN": 4591, "FIS": 2114, "GDDY": 1147, "FOX": 1958,
        "IFF": 1207, "BRO": 1434, "ERIE": 385, "AIZ": 456,
        "CSGP": 2088, "ARES": 876,
    },
    "prices": {
        "CSGP": 47.89, "ERIE": 259.30, "GPN": 79.31, "BRO": 69.69,
        "AIZ": 218.94, "FOX": 51.05, "IFF": 82.82, "ARES": 114.15,
        "GDDY": 87.18, "GEN": 21.78, "FIS": 47.29,
    },
    "orders": [
        {"ticker": "AIZ", "shares": -456, "action": "sell", "price": 218.94, "reason": "dropped"},
        {"ticker": "GPN", "shares": 1260, "action": "buy",  "price": 79.31,  "reason": "new entry"},
    ],
    "target_positions": {
        "ERIE": 385, "GPN": 1260, "BRO": 1434, "FOX": 1958,
        "IFF": 1207, "GDDY": 1147, "GEN": 4591, "FIS": 2114,
        "CSGP": 2088, "ARES": 876,
    },
    "position_pnl": {
        "GEN":  {"shares": 4591, "price_start": 21.65, "price_end": 21.78, "pnl":  596.83},
        "FIS":  {"shares": 2114, "price_start": 47.46, "price_end": 47.29, "pnl": -359.38},
        "GDDY": {"shares": 1147, "price_start": 87.76, "price_end": 87.18, "pnl": -665.26},
    },
    "orders_file": "data/live/pending_orders_2026-02-24.json",
}

LEDGER_ROWS: list = [
    {"date": "2026-02-21", "account_value": 1_000_000.0, "daily_pnl": 0.0,
     "cumulative_pnl": 0.0, "daily_return": 0.0, "cum_excess_return": 0.0,
     "num_positions": 0, "invested": 0, "cash": 0, "benchmark_return": 0},
    {"date": "2026-02-22", "account_value": 1_007_987.19, "daily_pnl": 7987.19,
     "cumulative_pnl": 7987.19, "daily_return": 0.007987, "cum_excess_return": 0.007987,
     "num_positions": 30, "invested": 0, "cash": 0, "benchmark_return": 0},
    {"date": "2026-02-23", "account_value": 992_198.52, "daily_pnl": -15788.67,
     "cumulative_pnl": -7801.48, "daily_return": -0.016574, "cum_excess_return": -0.008587,
     "num_positions": 30, "invested": 0, "cash": 0, "benchmark_return": 0},
    {"date": "2026-02-24", "account_value": 989_645.73, "daily_pnl": -381.99,
     "cumulative_pnl": -8183.47, "daily_return": -0.000386, "cum_excess_return": -0.008973,
     "num_positions": 10, "invested": 999_568.54, "cash": -9922.81, "benchmark_return": 0.0012},
]


# ---------------------------------------------------------------------------
# Import under test
# ---------------------------------------------------------------------------

from quantaalpha.live.report_md import generate_report, save_report  # noqa: E402


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_returns_string(self):
        md = generate_report(BASE_DATA)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_title_contains_date(self):
        md = generate_report(BASE_DATA)
        assert "2026-02-24" in md

    def test_executive_summary_section(self):
        md = generate_report(BASE_DATA)
        assert "Executive Summary" in md
        assert "989" in md          # account value
        assert "1,000,000" in md    # initial capital or nearby

    def test_sell_section_present(self):
        md = generate_report(BASE_DATA)
        assert "SELL" in md
        assert "AIZ" in md

    def test_buy_section_present(self):
        md = generate_report(BASE_DATA)
        assert "BUY" in md
        assert "GPN" in md

    def test_hold_section_lists_unchanged_tickers(self):
        md = generate_report(BASE_DATA)
        assert "Hold" in md
        # Tickers in both prev_pos and target with no order
        assert "ERIE" in md
        assert "BRO" in md

    def test_position_pnl_section(self):
        md = generate_report(BASE_DATA)
        assert "Position P&L" in md
        assert "GEN" in md
        assert "+$596.83" in md or "596" in md

    def test_target_portfolio_table(self):
        md = generate_report(BASE_DATA)
        assert "Target Portfolio" in md
        assert "TOTAL" in md

    def test_account_summary_section(self):
        md = generate_report(BASE_DATA)
        assert "Account Summary" in md
        assert "Initial Capital" in md
        assert "Total Return" in md

    def test_footer_contains_generated_at(self):
        from datetime import datetime, timezone
        ts = datetime(2026, 2, 24, 21, 34, tzinfo=timezone.utc)
        md = generate_report(BASE_DATA, generated_at=ts)
        assert "2026-02-24 21:34" in md

    def test_performance_history_with_ledger(self):
        md = generate_report(BASE_DATA, ledger=LEDGER_ROWS)
        assert "Performance History" in md
        assert "2026-02-22" in md
        assert "2026-02-23" in md


class TestHoldOnlyDay:
    def test_no_trades_section(self):
        data = {**BASE_DATA, "orders": [], "previous_positions": BASE_DATA["target_positions"]}
        md = generate_report(data)
        assert "No Trades" in md or "no trades" in md.lower()

    def test_hold_tickers_listed(self):
        data = {**BASE_DATA, "orders": [], "previous_positions": BASE_DATA["target_positions"]}
        md = generate_report(data)
        assert "ERIE" in md


class TestKillSwitchDay:
    def test_kill_switch_banner(self):
        data = {**BASE_DATA, "kill_switch": True, "orders": []}
        md = generate_report(data)
        assert "KILL-SWITCH" in md

    def test_no_orders_on_kill_switch(self):
        data = {**BASE_DATA, "kill_switch": True, "orders": []}
        md = generate_report(data)
        # Should not have SELL/BUY order tables
        assert "| AIZ" not in md


class TestMissingFields:
    def test_empty_data_does_not_crash(self):
        md = generate_report({})
        assert isinstance(md, str)

    def test_missing_prices_falls_back_to_order_prices(self):
        data = {**BASE_DATA, "prices": {}}
        # Should not raise; prices come from order dicts
        md = generate_report(data)
        assert "AIZ" in md

    def test_no_ledger_omits_history_section(self):
        md = generate_report(BASE_DATA, ledger=[])
        assert "Performance History" not in md

    def test_single_ledger_row_no_trend(self):
        # With only 1 row, _trend returns "→" without crashing
        md = generate_report(BASE_DATA, ledger=[LEDGER_ROWS[0]])
        assert isinstance(md, str)

    def test_zero_initial_capital(self):
        data = {**BASE_DATA, "initial_capital": 0}
        md = generate_report(data)
        # Should not divide by zero
        assert isinstance(md, str)


class TestSaveReport:
    def test_saves_to_correct_path(self, tmp_path):
        path = save_report(BASE_DATA, output_dir=tmp_path)
        assert path.exists()
        assert path.name == "report_2026-02-24.md"

    def test_file_content_is_markdown(self, tmp_path):
        path = save_report(BASE_DATA, ledger=LEDGER_ROWS, output_dir=tmp_path)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("# QuantaAlpha")
        assert "## Executive Summary" in content

    def test_idempotent_overwrite(self, tmp_path):
        save_report(BASE_DATA, output_dir=tmp_path)
        path = save_report(BASE_DATA, output_dir=tmp_path)
        assert path.exists()
        # File should contain exactly one title line
        lines = path.read_text().splitlines()
        title_lines = [l for l in lines if l.startswith("# QuantaAlpha Daily")]
        assert len(title_lines) == 1

    def test_creates_output_dir(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "reports"
        path = save_report(BASE_DATA, output_dir=nested)
        assert path.exists()


class TestOrderCategorisation:
    """Verify SELL/REDUCE/BUY/INCREASE logic via markdown output."""

    def test_increase_order_categorised(self):
        data = {
            **BASE_DATA,
            "orders": [
                # INCREASE: ticker already in prev_pos
                {"ticker": "GEN", "shares": 500, "action": "buy", "price": 21.78, "reason": "add"},
            ],
        }
        md = generate_report(data)
        assert "INCREASE" in md
        assert "GEN" in md

    def test_reduce_order_categorised(self):
        data = {
            **BASE_DATA,
            "orders": [
                # REDUCE: sell but ticker still in target
                {"ticker": "ERIE", "shares": -100, "action": "sell", "price": 259.30, "reason": "trim"},
            ],
            "target_positions": {**BASE_DATA["target_positions"], "ERIE": 285},
        }
        md = generate_report(data)
        assert "REDUCE" in md
        assert "ERIE" in md

    def test_sell_and_buy_action_summary(self):
        md = generate_report(BASE_DATA)
        # Executive summary should note both actions
        assert "SELL" in md
        assert "BUY" in md
