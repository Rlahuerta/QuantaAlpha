"""Tests for quantaalpha/live/scheduler.py, ibkr_executor.py, and position_tracker.py.

All tests are fully offline — no TWS connection, no H5 files, no LightGBM model.
"""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def live_config(tmp_path: Path) -> Path:
    """Write a minimal live.yaml to a temp dir and return its path."""
    cfg = {
        "model": {"meta_path": str(tmp_path / "meta.json")},
        "data": {"h5_path": str(tmp_path / "daily_pv.h5"), "lookback_days": 60},
        "portfolio": {"topk": 5, "n_drop": 2, "capital": 100_000},
        "schedule": {"timezone": "America/New_York", "ingest_time": "16:30", "signal_time": "17:00"},
        "ibkr": {"host": "127.0.0.1", "port": 7497, "client_id": 1, "timeout": 5, "dry_run": True},
        "output": {
            "orders_dir": str(tmp_path / "orders"),
            "positions_file": str(tmp_path / "positions.json"),
            "pnl_dir": str(tmp_path / "pnl"),
        },
    }
    config_file = tmp_path / "live.yaml"
    config_file.write_text(yaml.dump(cfg))
    return config_file


# ===========================================================================
# TradingScheduler tests
# ===========================================================================

class TestTradingSchedulerInit:
    def test_loads_config(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        assert s.config["portfolio"]["topk"] == 5
        assert s._scheduler is None

    def test_missing_config_raises(self, tmp_path: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        with pytest.raises(FileNotFoundError):
            TradingScheduler(tmp_path / "nonexistent.yaml")

    def test_timezone_default(self, tmp_path: Path):
        cfg = {"model": {}, "data": {}, "portfolio": {}, "ibkr": {}, "output": {}}
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml.dump(cfg))
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(p)
        assert s._timezone() == "America/New_York"

    def test_timezone_from_config(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        assert s._timezone() == "America/New_York"

    def test_parse_time(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        assert s._parse_time("ingest_time", "16:00") == (16, 30)
        assert s._parse_time("signal_time", "16:00") == (17, 0)

    def test_parse_time_default_used_when_key_absent(self, tmp_path: Path):
        cfg = {"schedule": {"timezone": "UTC"}}
        p = tmp_path / "s.yaml"
        p.write_text(yaml.dump(cfg))
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(p)
        assert s._parse_time("ingest_time", "15:45") == (15, 45)


class TestTradingSchedulerJobs:
    def test_get_jobs_before_start_returns_empty(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        assert s.get_jobs() == []

    def test_start_nonblocking_registers_two_jobs(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        s.start(blocking=False)
        jobs = s.get_jobs()
        assert len(jobs) == 2
        ids = {j["id"] for j in jobs}
        assert "ingest_job" in ids
        assert "signal_job" in ids
        s.stop()

    def test_stop_after_start(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        s.start(blocking=False)
        s.stop()
        assert not s._scheduler.running

    def test_stop_before_start_is_noop(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        s.stop()  # should not raise

    def test_job_names_correct(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        s = TradingScheduler(live_config)
        s.start(blocking=False)
        names = {j["name"] for j in s.get_jobs()}
        assert "EOD Data Ingest" in names
        assert "Signal Generation + Order Save" in names
        s.stop()


class TestTradingSchedulerRunIngest:
    def test_run_ingest_calls_ingestor(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        mock_ingestor = MagicMock()
        mock_ingestor.ingest.return_value = 42
        with patch.object(sched_mod, "DataIngestor", return_value=mock_ingestor):
            rows = s.run_ingest()
        assert rows == 42
        mock_ingestor.ingest.assert_called_once_with(force_full=False)

    def test_run_ingest_uses_h5_path_from_config(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        captured = {}
        def fake_ingestor(h5_path, **_):
            captured["h5_path"] = h5_path
            m = MagicMock(); m.ingest.return_value = 0; return m
        with patch.object(sched_mod, "DataIngestor", side_effect=fake_ingestor):
            s.run_ingest()
        assert "daily_pv.h5" in captured["h5_path"]


class TestTradingSchedulerRunSignal:
    def _make_mock_sg(self, scores=None):
        sg = MagicMock()
        if scores is None:
            scores = {"AAPL": 1.0, "MSFT": 0.9, "GOOGL": 0.8, "AMZN": 0.7, "META": 0.6}
        sg.generate.return_value = scores
        sg._load_h5_window.return_value = None  # simplify: skip price loading
        return sg

    def test_run_signal_creates_orders_file(self, live_config: Path, tmp_path: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        mock_sg = self._make_mock_sg()
        mock_constructor = MagicMock()
        result = MagicMock()
        result.orders = []
        result.target_portfolio = {"AAPL": 100}
        mock_constructor.rebalance.return_value = result
        with patch.object(sched_mod, "SignalGenerator") as MockSG, \
             patch.object(sched_mod, "PortfolioConstructor", return_value=mock_constructor):
            MockSG.from_meta.return_value = mock_sg
            order_data = s.run_signal()
        assert "date" in order_data
        orders_dir = Path(s.config["output"]["orders_dir"])
        files = list(orders_dir.glob("pending_orders_*.json"))
        assert len(files) == 1

    def test_run_signal_empty_scores_returns_empty(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        mock_sg = self._make_mock_sg(scores={})
        with patch.object(sched_mod, "SignalGenerator") as MockSG:
            MockSG.from_meta.return_value = mock_sg
            result = s.run_signal()
        assert result == {}

    def test_run_signal_saves_last_orders(self, live_config: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        mock_sg = self._make_mock_sg()
        mock_constructor = MagicMock()
        rebalance_result = MagicMock()
        rebalance_result.orders = []
        rebalance_result.target_portfolio = {}
        mock_constructor.rebalance.return_value = rebalance_result
        with patch.object(sched_mod, "SignalGenerator") as MockSG, \
             patch.object(sched_mod, "PortfolioConstructor", return_value=mock_constructor):
            MockSG.from_meta.return_value = mock_sg
            s.run_signal()
        assert s._last_orders is not None
        assert "orders" in s._last_orders

    def test_run_signal_existing_positions_loaded(self, live_config: Path, tmp_path: Path):
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        pos_file = Path(s.config["output"]["positions_file"])
        pos_file.parent.mkdir(parents=True, exist_ok=True)
        pos_file.write_text(json.dumps({"positions": {"AAPL": 50}}))
        captured = {}
        mock_sg = self._make_mock_sg()
        mock_constructor = MagicMock()
        def capture_rebalance(**kwargs):
            captured.update(kwargs)
            r = MagicMock(); r.orders = []; r.target_portfolio = {}
            return r
        mock_constructor.rebalance.side_effect = capture_rebalance
        with patch.object(sched_mod, "SignalGenerator") as MockSG, \
             patch.object(sched_mod, "PortfolioConstructor", return_value=mock_constructor):
            MockSG.from_meta.return_value = mock_sg
            s.run_signal()
        assert captured.get("positions", {}).get("AAPL") == 50

    def test_run_signal_records_pnl_snapshot(self, live_config: Path, tmp_path: Path):
        """run_signal() writes a P&L snapshot via PositionTracker."""
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        mock_sg = self._make_mock_sg()
        mock_constructor = MagicMock()
        result = MagicMock()
        result.orders = []
        result.target_portfolio = {"AAPL": 100, "MSFT": 80}
        mock_constructor.rebalance.return_value = result
        with patch.object(sched_mod, "SignalGenerator") as MockSG, \
             patch.object(sched_mod, "PortfolioConstructor", return_value=mock_constructor):
            MockSG.from_meta.return_value = mock_sg
            s.run_signal()
        # Check positions.json was written with new target portfolio
        pos_file = Path(s.config["output"]["positions_file"])
        assert pos_file.exists()
        state = json.loads(pos_file.read_text())
        assert state["positions"]["AAPL"] == 100
        assert state["positions"]["MSFT"] == 80
        assert "pnl" in state
        # Check pnl history file was written
        pnl_dir = Path(s.config["output"]["pnl_dir"])
        pnl_files = list(pnl_dir.glob("pnl_*.json"))
        assert len(pnl_files) == 1

    def test_run_signal_accumulates_pnl(self, live_config: Path, tmp_path: Path):
        """Two consecutive runs accumulate cumulative P&L."""
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        mock_sg = self._make_mock_sg()
        mock_constructor = MagicMock()
        result = MagicMock()
        result.orders = []
        result.target_portfolio = {"AAPL": 100}
        mock_constructor.rebalance.return_value = result
        with patch.object(sched_mod, "SignalGenerator") as MockSG, \
             patch.object(sched_mod, "PortfolioConstructor", return_value=mock_constructor):
            MockSG.from_meta.return_value = mock_sg
            s.run_signal()
            s.run_signal()
        pos_file = Path(s.config["output"]["positions_file"])
        state = json.loads(pos_file.read_text())
        assert state["account_value"] > 0
        assert "cumulative_pnl" in state["pnl"]

    def test_run_signal_kill_switch_halts_orders(self, live_config: Path, tmp_path: Path):
        """KillSwitch triggers when daily P&L exceeds loss limit — no orders file."""
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        # Seed positions and config with low capital so small loss triggers kill switch
        s.config["risk"] = {"daily_loss_limit_pct": 0.01}
        s.config["portfolio"]["capital"] = 10_000
        # Write existing positions with high value so P&L is computed
        pos_file = Path(s.config["output"]["positions_file"])
        pos_file.parent.mkdir(parents=True, exist_ok=True)
        pos_file.write_text(json.dumps({
            "positions": {"BAD_STOCK": 100},
            "account_value": 10_000,
            "pnl": {"cumulative_pnl": 0, "cumulative_excess_return": 0},
        }))
        mock_sg = self._make_mock_sg()
        # Return prices that show a massive loss
        import pandas as pd
        idx = pd.MultiIndex.from_tuples(
            [("2026-01-01", "BAD_STOCK"), ("2026-01-02", "BAD_STOCK")],
            names=["datetime", "instrument"],
        )
        window_df = pd.DataFrame(
            {"$close": [200.0, 50.0]},  # -75% drop → loss = 100 * 150 = $15k > 1% of $10k
            index=idx,
        )
        mock_sg._load_h5_window.return_value = window_df
        mock_constructor = MagicMock()
        with patch.object(sched_mod, "SignalGenerator") as MockSG, \
             patch.object(sched_mod, "PortfolioConstructor", return_value=mock_constructor):
            MockSG.from_meta.return_value = mock_sg
            result = s.run_signal()
        assert result.get("kill_switch") is True
        assert result.get("daily_pnl") < 0
        # No rebalance should have been called
        mock_constructor.rebalance.assert_not_called()
        # P&L should still be recorded even on kill-switch day
        pnl_dir = Path(s.config["output"]["pnl_dir"])
        assert len(list(pnl_dir.glob("pnl_*.json"))) == 1

    def test_run_signal_includes_daily_pnl_in_orders(self, live_config: Path):
        """Order data includes daily_pnl and account_value fields."""
        from quantaalpha.live.scheduler import TradingScheduler
        import quantaalpha.live.scheduler as sched_mod
        s = TradingScheduler(live_config)
        mock_sg = self._make_mock_sg()
        mock_constructor = MagicMock()
        result = MagicMock()
        result.orders = []
        result.target_portfolio = {"AAPL": 50}
        mock_constructor.rebalance.return_value = result
        with patch.object(sched_mod, "SignalGenerator") as MockSG, \
             patch.object(sched_mod, "PortfolioConstructor", return_value=mock_constructor):
            MockSG.from_meta.return_value = mock_sg
            order_data = s.run_signal()
        assert "daily_pnl" in order_data
        assert "account_value" in order_data


# ===========================================================================
# IBKRExecutor tests
# ===========================================================================

class TestIBKRExecutorInit:
    def test_defaults(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor()
        assert ex.host == "127.0.0.1"
        assert ex.port == 7497
        assert ex.dry_run is True

    def test_from_config(self, live_config: Path):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor.from_config(live_config)
        assert ex.port == 7497
        assert ex.dry_run is True
        assert ex.client_id == 1

    def test_custom_params(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(host="10.0.0.1", port=7496, dry_run=False)
        assert ex.host == "10.0.0.1"
        assert ex.port == 7496
        assert ex.dry_run is False


class TestIBKRExecutorDryRun:
    def test_connect_is_noop(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        ex.connect()  # should not raise
        assert ex._ib is None

    def test_disconnect_is_noop(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        ex.disconnect()  # should not raise

    def test_is_connected_true_in_dry_run(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        assert ex.is_connected is True

    def test_get_account_value_returns_default(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        assert ex.get_account_value() == 1_000_000.0

    def test_get_positions_returns_empty(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        assert ex.get_positions() == {}

    def test_get_latest_prices_returns_empty(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        assert ex.get_latest_prices(["AAPL", "MSFT"]) == {}

    def test_cancel_open_orders_returns_zero(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        assert ex.cancel_open_orders() == 0


class TestIBKRExecutorPlaceOrder:
    def test_buy_order_dry_run(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor, TradeResult
        ex = IBKRExecutor(dry_run=True)
        result = ex.place_market_order("AAPL", 100)
        assert isinstance(result, TradeResult)
        assert result.ticker == "AAPL"
        assert result.shares == 100
        assert result.status == "dry_run"
        assert result.is_buy

    def test_sell_order_dry_run(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        result = ex.place_market_order("MSFT", -50)
        assert result.shares == -50
        assert result.status == "dry_run"
        assert result.is_sell

    def test_zero_shares_skipped(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        result = ex.place_market_order("AAPL", 0)
        assert result.status == "skipped"

    def test_trade_result_notional(self):
        from quantaalpha.live.ibkr_executor import TradeResult
        r = TradeResult(ticker="AAPL", shares=10, status="dry_run", price=150.0)
        assert abs(r.notional - 1500.0) < 0.01

    def test_trade_result_notional_no_price(self):
        from quantaalpha.live.ibkr_executor import TradeResult
        r = TradeResult(ticker="AAPL", shares=10, status="dry_run")
        assert r.notional == 0.0


class TestIBKRExecutorExecuteOrders:
    def test_sells_before_buys(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        orders = [
            {"ticker": "AAPL", "shares": 100},
            {"ticker": "MSFT", "shares": -50},
            {"ticker": "GOOGL", "shares": 75},
        ]
        results = ex.execute_orders(orders)
        assert len(results) == 3
        # Sell should come first
        assert results[0].ticker == "MSFT"
        assert results[0].is_sell

    def test_all_results_dry_run(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        orders = [{"ticker": "AAPL", "shares": 10}, {"ticker": "MSFT", "shares": -5}]
        results = ex.execute_orders(orders)
        assert all(r.status == "dry_run" for r in results)

    def test_empty_orders(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        results = ex.execute_orders([])
        assert results == []


class TestIBKRExecutorContextManager:
    def test_context_manager_connects_disconnects(self):
        from quantaalpha.live.ibkr_executor import IBKRExecutor
        ex = IBKRExecutor(dry_run=True)
        with ex as executor:
            assert executor.is_connected
        # disconnect called — no error


# ===========================================================================
# PositionTracker tests
# ===========================================================================

class TestPositionTrackerState:
    def test_load_state_missing_returns_empty(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(
            positions_file=tmp_path / "positions.json",
            pnl_dir=tmp_path / "pnl",
        )
        assert pt.load_state() == {}

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(positions_file=tmp_path / "positions.json", pnl_dir=tmp_path / "pnl")
        state = {"date": "2026-02-20", "positions": {"AAPL": 100}}
        pt.save_state(state)
        loaded = pt.load_state()
        assert loaded == state

    def test_save_creates_parent_dir(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(
            positions_file=tmp_path / "nested" / "positions.json",
            pnl_dir=tmp_path / "pnl",
        )
        pt.save_state({"date": "2026-02-20"})
        assert (tmp_path / "nested" / "positions.json").exists()


class TestPositionTrackerUpdatePositions:
    def test_buy_increases_position(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        updated = pt.update_positions(
            [{"ticker": "AAPL", "shares": 100}],
            current_positions={"AAPL": 50},
        )
        assert updated["AAPL"] == 150

    def test_sell_decreases_position(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        updated = pt.update_positions(
            [{"ticker": "AAPL", "shares": -50}],
            current_positions={"AAPL": 100},
        )
        assert updated["AAPL"] == 50

    def test_full_sell_removes_position(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        updated = pt.update_positions(
            [{"ticker": "AAPL", "shares": -100}],
            current_positions={"AAPL": 100},
        )
        assert "AAPL" not in updated

    def test_new_position_added(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        updated = pt.update_positions(
            [{"ticker": "MSFT", "shares": 80}],
            current_positions={"AAPL": 100},
        )
        assert updated["MSFT"] == 80
        assert updated["AAPL"] == 100

    def test_loads_from_disk_when_positions_none(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pos_file = tmp_path / "positions.json"
        pos_file.write_text(json.dumps({"positions": {"AAPL": 200}}))
        pt = PositionTracker(pos_file, tmp_path / "pnl")
        updated = pt.update_positions([{"ticker": "AAPL", "shares": -100}])
        assert updated["AAPL"] == 100


class TestPositionTrackerPnL:
    def test_daily_pnl_computed_correctly(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        result = pt.compute_daily_pnl(
            positions_start={"AAPL": 100, "MSFT": 50},
            prices_start={"AAPL": 150.0, "MSFT": 300.0},
            prices_end={"AAPL": 152.0, "MSFT": 302.0},
            account_value_start=30_000.0,
            benchmark_return=0.005,
        )
        # AAPL: 100*(152-150)=200, MSFT: 50*(302-300)=100, total=300
        assert abs(result["daily_pnl"] - 300.0) < 0.01
        assert abs(result["portfolio_return"] - 300.0 / 30_000.0) < 1e-6
        excess = 300.0 / 30_000.0 - 0.005
        assert abs(result["daily_excess_return"] - excess) < 1e-6

    def test_empty_positions_returns_zeros(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        result = pt.compute_daily_pnl({}, {}, {}, 100_000.0, 0.01)
        assert result["daily_pnl"] == 0.0
        assert result["portfolio_return"] == 0.0

    def test_zero_account_value_returns_zeros(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        result = pt.compute_daily_pnl({"AAPL": 100}, {"AAPL": 150.0}, {"AAPL": 155.0}, 0.0, 0.0)
        assert result["portfolio_return"] == 0.0

    def test_missing_end_price_skipped(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "p.json", tmp_path / "pnl")
        result = pt.compute_daily_pnl(
            {"AAPL": 100, "MSFT": 50},
            {"AAPL": 150.0, "MSFT": 300.0},
            {"AAPL": 155.0},  # MSFT missing
            30_000.0,
            0.0,
        )
        assert abs(result["daily_pnl"] - 500.0) < 0.01
        assert result["num_positions"] == 1


class TestPositionTrackerRecordDay:
    def test_record_day_creates_history_file(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        today = date(2026, 2, 20)
        pt.record_day(
            positions={"AAPL": 100},
            account_value=105_000.0,
            daily_pnl_dict={"daily_pnl": 1000.0, "portfolio_return": 0.01, "daily_excess_return": 0.005, "num_positions": 1},
            as_of=today,
        )
        assert (tmp_path / "pnl" / "pnl_2026-02-20.json").exists()

    def test_record_day_accumulates_cumulative(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        # Day 1
        pt.record_day(
            positions={"AAPL": 100},
            account_value=100_000.0,
            daily_pnl_dict={"daily_pnl": 500.0, "daily_excess_return": 0.002, "portfolio_return": 0.005, "num_positions": 1},
            as_of=date(2026, 2, 19),
        )
        # Day 2
        state2 = pt.record_day(
            positions={"AAPL": 100},
            account_value=100_500.0,
            daily_pnl_dict={"daily_pnl": 300.0, "daily_excess_return": 0.001, "portfolio_return": 0.003, "num_positions": 1},
            as_of=date(2026, 2, 20),
        )
        assert abs(state2["pnl"]["cumulative_pnl"] - 800.0) < 0.01
        assert abs(state2["pnl"]["cumulative_excess_return"] - 0.003) < 1e-6

    def test_record_day_updates_positions_file(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        pt.record_day(
            positions={"AAPL": 100, "MSFT": 50},
            account_value=50_000.0,
            daily_pnl_dict={"daily_pnl": 0.0, "daily_excess_return": 0.0, "portfolio_return": 0.0, "num_positions": 2},
            as_of=date(2026, 2, 20),
        )
        state = pt.load_state()
        assert state["positions"]["AAPL"] == 100
        assert state["positions"]["MSFT"] == 50


class TestPositionTrackerSummary:
    def test_summary_no_data(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        s = pt.summary()
        assert s == {"status": "no data"}

    def test_summary_after_record(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        pt.record_day(
            positions={"AAPL": 100},
            account_value=110_000.0,
            daily_pnl_dict={"daily_pnl": 500.0, "daily_excess_return": 0.003, "portfolio_return": 0.005, "num_positions": 1},
            as_of=date(2026, 2, 20),
        )
        s = pt.summary()
        assert s["date"] == "2026-02-20"
        assert s["num_positions"] == 1
        assert s["account_value"] == 110_000.0

    def test_load_pnl_history_empty(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        assert pt.load_pnl_history() == []

    def test_load_pnl_history_returns_records(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        for d in [date(2026, 2, 18), date(2026, 2, 19), date(2026, 2, 20)]:
            pt.record_day(
                positions={"AAPL": 100},
                account_value=100_000.0,
                daily_pnl_dict={"daily_pnl": 0.0, "daily_excess_return": 0.0, "portfolio_return": 0.0, "num_positions": 1},
                as_of=d,
            )
        history = pt.load_pnl_history(days=2)
        assert len(history) == 2
        assert history[-1]["date"] == "2026-02-20"

    def test_load_pnl_history_sorted_ascending(self, tmp_path: Path):
        from quantaalpha.live.position_tracker import PositionTracker
        pt = PositionTracker(tmp_path / "positions.json", tmp_path / "pnl")
        for d in [date(2026, 2, 20), date(2026, 2, 18), date(2026, 2, 19)]:
            pt.record_day(
                positions={"AAPL": 100},
                account_value=100_000.0,
                daily_pnl_dict={"daily_pnl": 0.0, "daily_excess_return": 0.0, "portfolio_return": 0.0, "num_positions": 1},
                as_of=d,
            )
        history = pt.load_pnl_history()
        dates = [h["date"] for h in history]
        assert dates == sorted(dates)
