"""Unit tests for quantaalpha.live.report_md — chain-of-blocks format.

Tests cover:
- Single-day chain (initial capital block + day block)
- Multi-day chain with cash flow arithmetic
- SELL → BUY cash flow waterfall correctness
- Hold-only day (no orders)
- Current-day top-N picks section
- Missing prices / partial data graceful degradation
- load_all_orders deduplication (live wins over archive)
- save_chain_report writes trading_journal.md
- backward-compat wrappers: generate_report, save_report
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from quantaalpha.live.report_md import (
    generate_chain_report,
    generate_report,
    load_all_orders,
    save_chain_report,
    save_report,
)

# ─── Minimal day fixtures ──────────────────────────────────────────────────────

DAY1: dict = {
    "date": "2026-02-20",
    "account_value": 990_000.0,
    "daily_pnl": 0.0,
    "initial_capital": 1_000_000.0,
    "orders": [
        {"ticker": "AAPL", "shares": 500, "action": "buy", "price": 200.0, "reason": "new"},
        {"ticker": "MSFT", "shares": 200, "action": "buy", "price": 500.0, "reason": "new"},
    ],
    "target_positions": {"AAPL": 500, "MSFT": 200},
    "previous_positions": {},
    "prices": {},           # no prices at the top level — must use order prices
    "position_pnl": {},
}

DAY2: dict = {
    "date": "2026-02-21",
    "account_value": 1_002_000.0,
    "daily_pnl": 12_000.0,
    "initial_capital": 1_000_000.0,
    "benchmark_return": 0.005,
    "orders": [
        {"ticker": "AAPL", "shares": -500, "action": "sell", "price": 220.0, "reason": "exit"},
        {"ticker": "GOOG", "shares": 100, "action": "buy",  "price": 300.0, "reason": "new"},
    ],
    "target_positions": {"MSFT": 200, "GOOG": 100},
    "previous_positions": {"AAPL": 500, "MSFT": 200},
    "prices": {"AAPL": 220.0, "MSFT": 510.0, "GOOG": 300.0},
    "position_pnl": {
        "AAPL": {"shares": 500, "price_start": 200.0, "price_end": 220.0, "pnl": 10_000.0},
        "MSFT": {"shares": 200, "price_start": 500.0, "price_end": 510.0, "pnl":  2_000.0},
    },
}

FULL_DATA: dict = {
    "date": "2026-02-24",
    "initial_capital": 1_000_000.0,
    "daily_pnl": -381.99,
    "account_value": 989_645.73,
    "cash": -9_922.81,
    "benchmark_return": 0.0012,
    "cumulative_pnl": -8_183.47,
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
}

# ─── generate_chain_report ─────────────────────────────────────────────────────

def test_chain_report_has_initial_block():
    md = generate_chain_report([DAY1], initial_capital=1_000_000.0, topk=5)
    assert "Block #0" in md
    assert "Initial State" in md
    assert "$1,000,000.00" in md


def test_chain_report_block_numbering():
    md = generate_chain_report([DAY1, DAY2], initial_capital=1_000_000.0, topk=5)
    assert "Block #1" in md
    assert "Block #2" in md


def test_chain_report_dates_appear():
    md = generate_chain_report([DAY1, DAY2], initial_capital=1_000_000.0)
    assert "2026-02-20" in md
    assert "2026-02-21" in md


def test_chain_report_current_label_on_last_block():
    md = generate_chain_report([DAY1, DAY2], initial_capital=1_000_000.0)
    # Only the last block gets the CURRENT marker
    assert "CURRENT" in md
    # The CURRENT marker must appear alongside DAY2's date
    current_line = [l for l in md.splitlines() if "CURRENT" in l][0]
    assert "2026-02-21" in current_line


def test_chain_report_no_current_on_earlier_blocks():
    """Block #1 must NOT be tagged CURRENT when there are multiple blocks."""
    md = generate_chain_report([DAY1, DAY2], initial_capital=1_000_000.0)
    lines = md.splitlines()
    block1_line = [l for l in lines if "Block #1" in l][0]
    assert "CURRENT" not in block1_line


# ─── Cash flow arithmetic ──────────────────────────────────────────────────────

def test_cash_flow_sell_increases_cash():
    """Selling 500 AAPL @ 220 should add $110,000 to opening cash."""
    initial = 1_000_000.0
    # After DAY1 buys: 500×200 + 200×500 = 200,000; opening cash = 800,000
    # DAY2: SELL 500 AAPL @ 220 → +110,000 → running cash = 910,000
    # then BUY 100 GOOG @ 300 → -30,000 → closing = 880,000
    md = generate_chain_report([DAY1, DAY2], initial_capital=initial, topk=5)
    assert "Opening Cash" in md
    # SELL proceeds appear as positive in the waterfall
    assert "+$110,000.00" in md


def test_cash_flow_buy_decreases_cash():
    """Buying 100 GOOG @ 300 should deduct $30,000."""
    md = generate_chain_report([DAY1, DAY2], initial_capital=1_000_000.0, topk=5)
    assert "−$30,000.00" in md


def test_first_day_opening_cash_is_initial_capital():
    """First day's opening cash must equal initial_capital."""
    md = generate_chain_report([DAY1], initial_capital=1_000_000.0)
    # Opening Cash row must show $1,000,000.00
    lines_with_opening = [l for l in md.splitlines() if "Opening Cash" in l]
    assert lines_with_opening, "No 'Opening Cash' row found"
    assert "$1,000,000.00" in lines_with_opening[0]


def test_sell_before_buy_order():
    """SELLs must appear before BUYs in the cash flow table."""
    md = generate_chain_report([DAY2], initial_capital=1_000_000.0)
    sell_idx = md.index("SELL ALL")
    buy_idx  = md.index("BUY NEW")
    assert sell_idx < buy_idx


# ─── topk row-limiting ────────────────────────────────────────────────────────

def _make_day_with_n_buys(n: int) -> dict:
    """Day with n BUY NEW orders of equal value."""
    orders = [
        {"ticker": f"T{i:02d}", "shares": 100, "action": "buy",
         "price": 100.0, "reason": "new"}
        for i in range(n)
    ]
    return {
        "date": "2026-03-01",
        "account_value": 1_000_000.0 - n * 10_000,
        "daily_pnl": 0.0,
        "orders": orders,
        "target_positions": {f"T{i:02d}": 100 for i in range(n)},
        "previous_positions": {},
        "prices": {f"T{i:02d}": 100.0 for i in range(n)},
        "position_pnl": {},
    }


def test_topk_limits_buy_rows_displayed():
    """With topk=3 and 10 buys, only 3 individual rows + 1 collapsed row shown."""
    day = _make_day_with_n_buys(10)
    md = generate_chain_report([day], initial_capital=1_000_000.0, topk=3)
    # 3 visible BUY rows
    assert md.count("🟢 BUY NEW") == 3
    # 1 collapsed row for the remaining 7
    assert "7 more buys" in md


def test_topk_collapsed_row_shows_correct_cash():
    """Cash arithmetic must include hidden orders in the collapsed row."""
    day = _make_day_with_n_buys(5)  # 5 × $10,000 = $50,000 total buys
    md = generate_chain_report([day], initial_capital=1_000_000.0, topk=2)
    # Opening $1M, show 2 buys (-$10k each = $980k), collapse 3 (-$30k = $950k)
    assert "$950,000.00" in md   # closing cash after all 5 buys


def test_topk_no_collapse_when_orders_lte_topk():
    """No collapsed row if orders ≤ topk."""
    day = _make_day_with_n_buys(3)
    md = generate_chain_report([day], initial_capital=1_000_000.0, topk=5)
    assert "more buys" not in md
    assert md.count("🟢 BUY NEW") == 3


def test_topk_sells_always_shown():
    """All SELL orders are always shown regardless of topk."""
    day = {
        **_make_day_with_n_buys(5),
        "orders": [
            {"ticker": "OLD1", "shares": -100, "action": "sell", "price": 200.0},
            {"ticker": "OLD2", "shares": -50,  "action": "sell", "price": 150.0},
        ] + [
            {"ticker": f"T{i:02d}", "shares": 100, "action": "buy", "price": 100.0}
            for i in range(8)
        ],
        "previous_positions": {"OLD1": 100, "OLD2": 50},
    }
    md = generate_chain_report([day], initial_capital=1_000_000.0, topk=3)
    # Both sells must appear
    assert "OLD1" in md
    assert "OLD2" in md
    assert md.count("SELL ALL") == 2


def test_topk_holds_collapsed():
    """Hold list collapses excess tickers beyond topk."""
    holds = {f"H{i:02d}": i * 10 for i in range(20)}
    day = {
        "date": "2026-03-01",
        "account_value": 1_000_000.0,
        "daily_pnl": 0.0,
        "orders": [],
        "target_positions": holds,
        "previous_positions": holds,
        "prices": {k: 100.0 for k in holds},
        "position_pnl": {},
    }
    md = generate_chain_report([day], initial_capital=1_000_000.0, topk=5)
    assert "more:" in md   # collapsed suffix
    assert "**Hold** (20 positions)" in md


def test_topk_pnl_table_collapses_middle():
    """P&L table with > topk rows shows worst/best N/2 + collapsed middle."""
    pos_pnl = {
        f"S{i}": {"shares": 100, "price_start": 100.0,
                  "price_end": 100.0 - i, "pnl": -i * 100.0}
        for i in range(1, 13)   # 12 positions, topk=4 → show 2 + middle(8) + 2
    }
    day = {
        "date": "2026-03-01",
        "account_value": 990_000.0,
        "daily_pnl": -sum(i * 100 for i in range(1, 13)),
        "orders": [],
        "target_positions": {},
        "previous_positions": {},
        "prices": {},
        "position_pnl": pos_pnl,
    }
    md = generate_chain_report([day], initial_capital=1_000_000.0, topk=4)
    assert "more)" in md   # collapsed middle row



# ─── P&L attribution section ──────────────────────────────────────────────────

def test_position_pnl_table_present():
    md = generate_chain_report([DAY2], initial_capital=1_000_000.0)
    assert "Price Movements" in md
    assert "AAPL" in md
    assert "MSFT" in md


def test_position_pnl_totals():
    md = generate_chain_report([DAY2], initial_capital=1_000_000.0)
    assert "TOTAL" in md
    # DAY2 daily_pnl = 12,000
    assert "+$12,000.00" in md


def test_no_pnl_table_when_position_pnl_empty():
    day = {**DAY1, "position_pnl": {}}
    md = generate_chain_report([day], initial_capital=1_000_000.0)
    assert "Price Movements" not in md


# ─── Top-N picks section ──────────────────────────────────────────────────────

def test_top_picks_only_on_current_day():
    md = generate_chain_report([DAY1, DAY2], initial_capital=1_000_000.0, topk=3)
    # Top picks section present
    assert "Algorithm's Top" in md
    # Only one picks section (for the last block)
    assert md.count("Algorithm's Top") == 1


def test_top_picks_count_limited_by_topk():
    """With topk=2 only 2 tickers should appear in the picks table."""
    md = generate_chain_report([FULL_DATA], initial_capital=1_000_000.0, topk=2)
    assert "Algorithm's Top 2 Picks" in md


def test_top_picks_shows_weights():
    md = generate_chain_report([FULL_DATA], initial_capital=1_000_000.0, topk=5)
    # Weight column header
    assert "Weight" in md


def test_top_picks_total_row():
    md = generate_chain_report([FULL_DATA], initial_capital=1_000_000.0, topk=5)
    assert "Top-5 total" in md


# ─── SELL / BUY icons ─────────────────────────────────────────────────────────

def test_sell_icon_in_table():
    md = generate_chain_report([DAY2], initial_capital=1_000_000.0)
    assert "🔴 SELL ALL" in md


def test_buy_new_icon_in_table():
    md = generate_chain_report([DAY2], initial_capital=1_000_000.0)
    assert "🟢 BUY NEW" in md


def test_benchmark_line_when_present():
    md = generate_chain_report([DAY2], initial_capital=1_000_000.0)
    assert "SPY" in md


# ─── Hold-only day ────────────────────────────────────────────────────────────

def test_hold_only_day_no_orders_section():
    hold_day = {**DAY1, "orders": [], "previous_positions": DAY1["target_positions"]}
    md = generate_chain_report([hold_day], initial_capital=1_000_000.0)
    assert "No rebalancing orders" in md


def test_hold_tickers_listed():
    hold_day = {
        **DAY1,
        "orders": [],
        "target_positions": {"AAPL": 500, "MSFT": 200},
        "previous_positions": {"AAPL": 500, "MSFT": 200},
    }
    md = generate_chain_report([hold_day], initial_capital=1_000_000.0)
    assert "Hold" in md


# ─── Missing / partial data ───────────────────────────────────────────────────

def test_missing_account_value_graceful():
    day = {**DAY1, "account_value": None}
    md = generate_chain_report([day], initial_capital=1_000_000.0)
    assert "Block #1" in md


def test_missing_daily_pnl_graceful():
    day = {**DAY1, "daily_pnl": None}
    md = generate_chain_report([day], initial_capital=1_000_000.0)
    assert "Block #1" in md


def test_zero_orders_no_crash():
    day = {**DAY1, "orders": []}
    md = generate_chain_report([day], initial_capital=1_000_000.0)
    assert "Block #1" in md


def test_empty_days_list():
    """Empty list should at minimum produce the initial state block."""
    md = generate_chain_report([], initial_capital=1_000_000.0)
    assert "Block #0" in md
    assert "Initial State" in md


# ─── load_all_orders ──────────────────────────────────────────────────────────

def _write_orders(directory: Path, date: str, account_value: float) -> None:
    data = {
        "date": date,
        "account_value": account_value,
        "orders": [],
        "target_positions": {},
    }
    (directory / f"pending_orders_{date}.json").write_text(json.dumps(data))


def test_load_orders_basic(tmp_path: Path):
    _write_orders(tmp_path, "2026-02-20", 1_000_000.0)
    _write_orders(tmp_path, "2026-02-21", 1_002_000.0)
    days = load_all_orders(tmp_path)
    assert len(days) == 2
    assert days[0]["date"] == "2026-02-20"
    assert days[1]["date"] == "2026-02-21"


def test_load_orders_deduplication_live_wins(tmp_path: Path):
    """Live-directory version must override archive for the same date."""
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_orders(archive, "2026-02-23", 1_000_000.0)   # corrupted reset
    _write_orders(tmp_path, "2026-02-23", 936_845.0)    # correct live version
    days = load_all_orders(tmp_path)
    assert len(days) == 1
    assert days[0]["account_value"] == 936_845.0


def test_load_orders_sorted_ascending(tmp_path: Path):
    _write_orders(tmp_path, "2026-02-22", 1_007_987.0)
    _write_orders(tmp_path, "2026-02-20", 1_000_000.0)
    _write_orders(tmp_path, "2026-02-21", 1_000_000.0)
    days = load_all_orders(tmp_path)
    dates = [d["date"] for d in days]
    assert dates == sorted(dates)


def test_load_orders_archive_only(tmp_path: Path):
    archive = tmp_path / "archive"
    archive.mkdir()
    _write_orders(archive, "2026-02-20", 1_000_000.0)
    _write_orders(archive, "2026-02-21", 1_002_000.0)
    days = load_all_orders(tmp_path)
    assert len(days) == 2


def test_load_orders_empty_directory(tmp_path: Path):
    days = load_all_orders(tmp_path)
    assert days == []


def test_load_orders_ignores_malformed_json(tmp_path: Path):
    (tmp_path / "pending_orders_2026-02-20.json").write_text("{ broken json >>>")
    _write_orders(tmp_path, "2026-02-21", 1_002_000.0)
    days = load_all_orders(tmp_path)
    assert len(days) == 1


# ─── save_chain_report ────────────────────────────────────────────────────────

def test_save_chain_report_creates_file(tmp_path: Path):
    _write_orders(tmp_path, "2026-02-24", 990_000.0)
    path = save_chain_report(tmp_path, initial_capital=1_000_000.0, topk=5)
    assert path.exists()
    assert path.name == "trading_journal.md"


def test_save_chain_report_creates_reports_subdir(tmp_path: Path):
    _write_orders(tmp_path, "2026-02-24", 990_000.0)
    path = save_chain_report(tmp_path, initial_capital=1_000_000.0)
    assert path.parent.name == "reports"


def test_save_chain_report_custom_output_dir(tmp_path: Path):
    _write_orders(tmp_path, "2026-02-24", 990_000.0)
    out = tmp_path / "custom_output"
    path = save_chain_report(tmp_path, output_dir=out, initial_capital=1_000_000.0)
    assert path.parent == out


def test_save_chain_report_content(tmp_path: Path):
    _write_orders(tmp_path, "2026-02-24", 990_000.0)
    path = save_chain_report(tmp_path, initial_capital=1_000_000.0)
    content = path.read_text()
    assert "Trading Journal" in content
    assert "Block #0" in content


# ─── Backward-compat wrappers ─────────────────────────────────────────────────

def test_generate_report_backward_compat():
    md = generate_report(FULL_DATA)
    assert "Block #1" in md
    assert "2026-02-24" in md


def test_save_report_backward_compat(tmp_path: Path):
    path = save_report(FULL_DATA, output_dir=tmp_path)
    assert path.exists()
    assert path.name == "trading_journal.md"


# ─── Timestamp param ──────────────────────────────────────────────────────────

def test_generated_at_in_footer():
    ts = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    md = generate_chain_report([DAY1], initial_capital=1_000_000.0, generated_at=ts)
    assert "2026-03-01 12:00" in md


# ─── algo_info rendering ──────────────────────────────────────────────────────

ALGO_INFO: dict = {
    "model_stem": "us_union89_prod",
    "num_features": 89,
    "factor_json": "all_factors_library_us.json",
    "topk": 10,
    "n_drop": 1,
    "capital": 1_000_000.0,
    "max_position_pct": 0.10,
    "market": "sp500",
    "benchmark": "SPY",
    "train_range": "2016-01-01 ~ 2025-12-31",
    "test_range": "2022-01-01 ~ 2025-12-31",
    "arr": 0.18496,
    "ir": 0.4944,
    "mdd": -0.18129,
    "calmar": 1.0202,
    "ic": 0.03354,
    "rank_ic": 0.03192,
    "metrics_source": "us_union89_prod_backtest_metrics.json",
}


def test_algo_info_model_name_in_header():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "us_union89_prod" in md


def test_algo_info_algorithm_profile_section():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "Algorithm Profile" in md
    assert "89 custom alpha factors" in md


def test_algo_info_backtest_performance_section():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "Backtest Performance" in md
    assert "2022-01-01 ~ 2025-12-31" in md


def test_algo_info_arr_shown():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "+18.5%" in md


def test_algo_info_mdd_shown():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "-18.1%" in md


def test_algo_info_calmar_shown():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "1.020" in md


def test_algo_info_ic_shown():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "0.0335" in md


def test_algo_info_strategy_params_section():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "Strategy Parameters" in md
    assert "top-10 stocks" in md
    assert "drop 1 position" in md
    assert "10% of capital" in md


def test_algo_info_metrics_source_shown():
    md = generate_chain_report([DAY1], algo_info=ALGO_INFO)
    assert "us_union89_prod_backtest_metrics.json" in md


def test_algo_info_none_no_crash():
    """Report renders fine with no algo_info."""
    md = generate_chain_report([DAY1], algo_info=None)
    assert "Block #0" in md
    assert "Strategy Parameters" in md


def test_algo_info_empty_dict_no_crash():
    md = generate_chain_report([DAY1], algo_info={})
    assert "Block #0" in md


def test_algo_info_partial_no_crash():
    """Only some fields provided — no KeyError."""
    md = generate_chain_report([DAY1], algo_info={"model_stem": "my_model", "arr": 0.15})
    assert "my_model" in md
    assert "+15.0%" in md


# ─── load_algo_info ───────────────────────────────────────────────────────────

def _write_live_yaml(path: Path, meta_path: str = "", metrics_path: str = "") -> None:
    path.write_text(f"""
portfolio:
  topk: 10
  n_drop: 1
  capital: 1000000
  max_position_pct: 0.10
model:
  meta_path: "{meta_path}"
""")


def _write_meta_json(path: Path, stem: str = "test_model") -> None:
    path.write_text(json.dumps({
        "model_stem": stem,
        "num_features": 42,
        "factor_json": "factors.json",
        "booster_path": "model.txt",
    }))


def _write_metrics_json(path: Path) -> None:
    path.write_text(json.dumps({
        "metrics": {
            "IC": 0.03,
            "Rank IC": 0.028,
            "annualized_return": 0.18,
            "information_ratio": 0.5,
            "max_drawdown": -0.20,
            "calmar_ratio": 0.9,
        },
        "config": {
            "market": "sp500",
            "benchmark": "SPY",
            "data_range": "2016~2025",
            "test_range": "2022~2025",
        },
    }))


from quantaalpha.live.report_md import load_algo_info


def test_load_algo_info_missing_config(tmp_path: Path):
    info = load_algo_info(tmp_path / "nonexistent.yaml")
    assert info == {}


def test_load_algo_info_no_model_path(tmp_path: Path):
    cfg = tmp_path / "live.yaml"
    _write_live_yaml(cfg, meta_path="")
    info = load_algo_info(cfg)
    assert info.get("topk") == 10


def test_load_algo_info_reads_portfolio_params(tmp_path: Path):
    cfg = tmp_path / "live.yaml"
    _write_live_yaml(cfg)
    info = load_algo_info(cfg)
    assert info["topk"] == 10
    assert info["n_drop"] == 1
    assert info["capital"] == 1_000_000
    assert info["max_position_pct"] == 0.10


def test_load_algo_info_loads_meta(tmp_path: Path):
    meta = tmp_path / "model_meta.json"
    _write_meta_json(meta, stem="my_model")
    cfg = tmp_path / "live.yaml"
    _write_live_yaml(cfg, meta_path=str(meta))
    info = load_algo_info(cfg)
    assert info["model_stem"] == "my_model"
    assert info["num_features"] == 42


def test_load_algo_info_loads_metrics(tmp_path: Path):
    meta = tmp_path / "my_model_meta.json"
    _write_meta_json(meta, stem="my_model")
    metrics = tmp_path / "my_model_backtest_metrics.json"
    _write_metrics_json(metrics)
    cfg = tmp_path / "live.yaml"
    _write_live_yaml(cfg, meta_path=str(meta))
    info = load_algo_info(cfg)
    assert abs(info["arr"] - 0.18) < 1e-9
    assert info["benchmark"] == "SPY"
    assert info["mdd"] == pytest.approx(-0.20)


def test_load_algo_info_missing_metrics_graceful(tmp_path: Path):
    meta = tmp_path / "nomet_meta.json"
    _write_meta_json(meta, stem="nomet")
    cfg = tmp_path / "live.yaml"
    _write_live_yaml(cfg, meta_path=str(meta))
    info = load_algo_info(cfg)
    assert info["model_stem"] == "nomet"
    assert info.get("arr") is None


def test_save_chain_report_with_config_path(tmp_path: Path):
    """save_chain_report passes config info into the rendered report."""
    meta = tmp_path / "m_meta.json"
    _write_meta_json(meta, stem="m")
    metrics = tmp_path / "m_backtest_metrics.json"
    _write_metrics_json(metrics)
    cfg = tmp_path / "live.yaml"
    _write_live_yaml(cfg, meta_path=str(meta))
    _write_orders(tmp_path, "2026-02-24", 990_000.0)

    path = save_chain_report(tmp_path, config_path=cfg)
    content = path.read_text()
    assert "Algorithm Profile" in content
    assert "m" in content

