"""
tests/test_phase5_risk.py — Phase 5 unit tests

Covers:
  - PortfolioConstructor: max_position_pct cap + liquidity filter (ADV)
  - KillSwitch: triggered / not triggered / edge cases
  - GateChecker: pass / fail (Sharpe, MDD, days) scenarios
  - Alerter: dry_run, format templates, from_config, from_dict
  - retrain_us_model: _update_live_yaml, arg parsing
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# ────────────────────────────────────────────────────────────────────
# PortfolioConstructor — risk controls
# ────────────────────────────────────────────────────────────────────

class TestPortfolioConstructorRiskControls:

    def _make_scores(self, n=10) -> dict:
        return {f"T{i:02d}": float(n - i) for i in range(n)}

    def _make_prices(self, tickers, price=100.0) -> dict:
        return {t: price for t in tickers}

    # --- max_position_pct ---

    def test_default_max_position_pct_is_five_percent(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=5, capital=100_000)
        assert pc.max_position_pct == 0.05

    def test_max_position_pct_caps_shares(self):
        """With 4 stocks equal-weight = 25% each; cap at 5% → each capped to $5k."""
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=4, capital=100_000, max_position_pct=0.05)
        scores = {f"T{i}": float(10 - i) for i in range(10)}
        prices = {f"T{i}": 100.0 for i in range(10)}
        result = pc.rebalance(scores=scores, positions={}, prices=prices)
        for ticker, shares in result.target_portfolio.items():
            notional = shares * 100.0
            assert notional <= 100_000 * 0.05 + 100, \
                f"{ticker} notional ${notional:,.0f} exceeds 5% cap"

    def test_max_position_pct_one_means_uncapped(self):
        """Setting max_position_pct=1.0 means no cap (full equal weight)."""
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=2, capital=100_000, max_position_pct=1.0)
        scores = {"AAPL": 2.0, "MSFT": 1.0}
        prices = {"AAPL": 100.0, "MSFT": 100.0}
        result = pc.rebalance(scores=scores, positions={}, prices=prices)
        # Equal weight = 50% each = 500 shares at $100
        for shares in result.target_portfolio.values():
            assert shares == 500

    def test_position_cap_does_not_increase_shares(self):
        """Cap should only reduce, never increase, target allocation."""
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        # topk=2 equal-weight = 50% each → cap at 30% reduces to 30%
        pc = PortfolioConstructor(topk=2, capital=100_000, max_position_pct=0.30)
        result = pc.rebalance(
            scores={"AAPL": 2.0, "MSFT": 1.0},
            positions={},
            prices={"AAPL": 100.0, "MSFT": 100.0},
        )
        for shares in result.target_portfolio.values():
            assert shares <= 300  # 30% of $100k / $100 = 300 shares

    # --- liquidity filter (ADV) ---

    def test_min_adv_zero_disables_filter(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=3, min_adv=0)
        scores = {"AAPL": 3.0, "TINY": 2.0, "MSFT": 1.0}
        filtered = pc._apply_liquidity_filter(scores, adv={})
        assert filtered == scores  # unchanged

    def test_liquidity_filter_removes_low_adv(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=5, min_adv=5_000_000)
        scores = {"AAPL": 3.0, "TINY": 2.0, "MSFT": 1.0}
        adv = {"AAPL": 10_000_000, "TINY": 1_000, "MSFT": 8_000_000}
        filtered = pc._apply_liquidity_filter(scores, adv)
        assert "TINY" not in filtered
        assert "AAPL" in filtered
        assert "MSFT" in filtered

    def test_liquidity_filter_keeps_stocks_at_threshold(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=3, min_adv=5_000_000)
        scores = {"X": 1.0}
        adv = {"X": 5_000_000}  # exactly at threshold
        filtered = pc._apply_liquidity_filter(scores, adv)
        assert "X" in filtered

    def test_liquidity_filter_applied_in_rebalance(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=5, min_adv=5_000_000, max_position_pct=1.0)
        scores = {"AAPL": 5.0, "TINY": 4.0, "MSFT": 3.0, "GOOG": 2.0, "META": 1.0}
        prices = {t: 100.0 for t in scores}
        adv = {"AAPL": 10e6, "TINY": 100, "MSFT": 8e6, "GOOG": 9e6, "META": 7e6}
        result = pc.rebalance(scores=scores, positions={}, prices=prices, adv=adv)
        assert "TINY" not in result.target_portfolio

    def test_rebalance_adv_none_does_not_crash(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        pc = PortfolioConstructor(topk=3, min_adv=5_000_000)
        scores = {"AAPL": 3.0, "MSFT": 2.0, "GOOG": 1.0}
        prices = {t: 100.0 for t in scores}
        # adv=None should not raise
        result = pc.rebalance(scores=scores, positions={}, prices=prices, adv=None)
        assert isinstance(result.target_portfolio, dict)


# ────────────────────────────────────────────────────────────────────
# KillSwitch
# ────────────────────────────────────────────────────────────────────

class TestKillSwitch:

    def test_not_triggered_on_profit(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        assert ks.is_triggered(daily_pnl=5_000) is False

    def test_not_triggered_when_loss_below_limit(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        assert ks.is_triggered(daily_pnl=-20_000) is False  # limit is $30k

    def test_triggered_when_loss_exceeds_limit(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        assert ks.is_triggered(daily_pnl=-35_000) is True

    def test_triggered_exactly_at_limit(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        # limit_usd = 30_000; -30_000 is NOT < -30_000 → not triggered
        assert ks.is_triggered(daily_pnl=-30_000) is False

    def test_triggered_just_past_limit(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        assert ks.is_triggered(daily_pnl=-30_001) is True

    def test_limit_usd_property(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        assert ks.limit_usd == pytest.approx(30_000.0)

    def test_check_return_uses_capital(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        assert ks.check_return(-0.04) is True   # -4% > 3% limit
        assert ks.check_return(-0.02) is False  # -2% < 3% limit

    def test_invalid_limit_raises(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        with pytest.raises(ValueError):
            KillSwitch(daily_loss_limit_pct=0.0)
        with pytest.raises(ValueError):
            KillSwitch(daily_loss_limit_pct=1.5)

    def test_zero_pnl_not_triggered(self):
        from quantaalpha.live.risk_monitor import KillSwitch
        ks = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
        assert ks.is_triggered(0.0) is False


# ────────────────────────────────────────────────────────────────────
# GateChecker
# ────────────────────────────────────────────────────────────────────

class TestGateChecker:

    def _write_history(self, pnl_dir: Path, returns: list[float], capital=1_000_000.0):
        """Write synthetic pnl_*.json snapshots to pnl_dir."""
        pnl_dir.mkdir(parents=True, exist_ok=True)
        cum = 0.0
        for i, r in enumerate(returns):
            d = f"2026-01-{i+1:02d}" if i < 28 else f"2026-02-{i-27:02d}"
            snap = {
                "date": d,
                "daily_pnl": r * capital,
                "portfolio_return": r,
                "cumulative_pnl": cum + r * capital,
                "capital": capital,
            }
            cum += r * capital
            (pnl_dir / f"pnl_{d}.json").write_text(json.dumps(snap))

    def test_insufficient_days_fails(self, tmp_path):
        from quantaalpha.live.risk_monitor import GateChecker
        gate = GateChecker(min_days=60)
        self._write_history(tmp_path / "pnl", [0.001] * 10)
        result = gate.check(tmp_path / "pnl")
        assert result.passed is False
        assert result.days == 10
        assert "Insufficient" in result.reason

    def test_good_strategy_passes(self, tmp_path):
        from quantaalpha.live.risk_monitor import GateChecker
        # Daily return of +0.3% with no drawdown → Sharpe ≈ sqrt(252)*0.3/tiny_std >> 1.5
        gate = GateChecker(min_sharpe=1.5, max_mdd=0.10, min_days=60)
        good_returns = [0.003] * 60  # consistent +0.3%/day
        self._write_history(tmp_path / "pnl", good_returns)
        result = gate.check(tmp_path / "pnl")
        assert result.passed is True
        assert result.sharpe > 1.5
        assert result.days == 60

    def test_low_sharpe_fails(self, tmp_path):
        from quantaalpha.live.risk_monitor import GateChecker
        gate = GateChecker(min_sharpe=1.5, max_mdd=0.10, min_days=60)
        # Alternating +/-0.1% → near-zero Sharpe
        noisy = [0.001 if i % 2 == 0 else -0.001 for i in range(60)]
        self._write_history(tmp_path / "pnl", noisy)
        result = gate.check(tmp_path / "pnl")
        assert result.passed is False
        assert "Sharpe" in result.reason

    def test_large_drawdown_fails(self, tmp_path):
        from quantaalpha.live.risk_monitor import GateChecker
        gate = GateChecker(min_sharpe=0.0, max_mdd=0.10, min_days=60)
        # 20% crash on day 30
        returns = [0.003] * 30 + [-0.20] + [0.003] * 29
        self._write_history(tmp_path / "pnl", returns)
        result = gate.check(tmp_path / "pnl")
        assert result.passed is False
        assert "MDD" in result.reason

    def test_empty_dir_fails_with_no_history(self, tmp_path):
        from quantaalpha.live.risk_monitor import GateChecker
        (tmp_path / "pnl").mkdir()
        gate = GateChecker(min_days=1)
        result = gate.check(tmp_path / "pnl")
        assert result.passed is False
        assert result.days == 0

    def test_max_drawdown_calculation(self, tmp_path):
        from quantaalpha.live.risk_monitor import GateChecker
        gate = GateChecker(min_days=3)
        # +10%, +10%, -20% → peak = 1.21, trough = 0.968 → MDD ≈ -20%/1.21
        returns = [0.10, 0.10, -0.20]
        self._write_history(tmp_path / "pnl", returns)
        result = gate.check(tmp_path / "pnl")
        assert result.mdd < 0  # drawdown is negative
        assert abs(result.mdd) > 0.10

    def test_gate_result_fields_populated(self, tmp_path):
        from quantaalpha.live.risk_monitor import GateChecker
        gate = GateChecker(min_days=5)
        returns = [0.002] * 10
        self._write_history(tmp_path / "pnl", returns)
        result = gate.check(tmp_path / "pnl")
        assert isinstance(result.sharpe, float)
        assert isinstance(result.mdd, float)
        assert isinstance(result.total_return, float)
        assert isinstance(result.reason, str)
        assert result.days == 10


# ────────────────────────────────────────────────────────────────────
# Alerter
# ────────────────────────────────────────────────────────────────────

class TestAlerter:

    def _dry_run_alerter(self):
        from quantaalpha.live.alerter import Alerter, AlertConfig
        return Alerter(AlertConfig(dry_run=True))

    # --- from_dict ---

    def test_from_dict_dry_run_default(self):
        from quantaalpha.live.alerter import Alerter
        a = Alerter.from_dict({})
        assert a.config.dry_run is True

    def test_from_dict_sets_fields(self):
        from quantaalpha.live.alerter import Alerter
        a = Alerter.from_dict({"dry_run": False, "email_enabled": True, "smtp_host": "mail.example.com"})
        assert a.config.email_enabled is True
        assert a.config.smtp_host == "mail.example.com"

    # --- from_config ---

    def test_from_config_reads_yaml(self, tmp_path):
        from quantaalpha.live.alerter import Alerter
        cfg = {
            "alerting": {
                "dry_run": True,
                "subject_prefix": "[TEST]",
                "email": {"enabled": False},
                "slack": {"enabled": False, "webhook_url": ""},
            }
        }
        f = tmp_path / "live.yaml"
        f.write_text(yaml.dump(cfg))
        a = Alerter.from_config(f)
        assert a.config.subject_prefix == "[TEST]"
        assert a.config.dry_run is True

    def test_from_config_with_live_yaml(self):
        """Load real live.yaml if it exists."""
        from quantaalpha.live.alerter import Alerter
        live_yaml = Path("configs/live.yaml")
        if live_yaml.exists():
            a = Alerter.from_config(live_yaml)
            assert a.config.dry_run is True  # default is dry_run

    # --- send_alert (dry_run) ---

    def test_dry_run_returns_true(self):
        a = self._dry_run_alerter()
        assert a.send_alert("info", {"message": "test"}) is True

    def test_dry_run_does_not_call_email(self):
        a = self._dry_run_alerter()
        with patch.object(a, "_send_email") as mock_email:
            a.send_alert("error", {"message": "fail"})
            mock_email.assert_not_called()

    def test_dry_run_does_not_call_slack(self):
        a = self._dry_run_alerter()
        with patch.object(a, "_send_slack") as mock_slack:
            a.send_alert("kill_switch", {"daily_pnl": -50000, "limit_pct": 0.03})
            mock_slack.assert_not_called()

    # --- message formatting ---

    def test_kill_switch_template(self):
        a = self._dry_run_alerter()
        msg = a._format_message("kill_switch", {"daily_pnl": -35000, "limit_pct": 0.03})
        assert "KILL-SWITCH" in msg
        assert "35,000" in msg

    def test_gate_passed_template(self):
        a = self._dry_run_alerter()
        msg = a._format_message("gate_passed", {"days": 65, "sharpe": 1.8, "mdd": -0.07})
        assert "GATE CHECK PASSED" in msg
        assert "65" in msg

    def test_gate_failed_template(self):
        a = self._dry_run_alerter()
        msg = a._format_message("gate_failed", {"reason": "Sharpe 0.5 < 1.5"})
        assert "GATE CHECK FAILED" in msg
        assert "Sharpe" in msg

    def test_drawdown_breach_template(self):
        a = self._dry_run_alerter()
        msg = a._format_message("drawdown_breach", {"mdd": -0.12, "threshold": 0.10})
        assert "DRAWDOWN" in msg

    def test_unknown_event_type_does_not_raise(self):
        a = self._dry_run_alerter()
        result = a.send_alert("custom_event", {"foo": "bar"})
        assert result is True

    def test_empty_details_does_not_raise(self):
        a = self._dry_run_alerter()
        result = a.send_alert("info")
        assert result is True

    # --- non-dry-run path (mocked) ---

    def test_non_dry_run_calls_email_when_enabled(self):
        from quantaalpha.live.alerter import Alerter, AlertConfig
        cfg = AlertConfig(dry_run=False, email_enabled=True, to_addrs=["x@y.com"])
        a = Alerter(cfg)
        with patch.object(a, "_send_email", return_value=True) as mock_email:
            result = a.send_alert("info", {"message": "hi"})
        mock_email.assert_called_once()
        assert result is True

    def test_non_dry_run_calls_slack_when_enabled(self):
        from quantaalpha.live.alerter import Alerter, AlertConfig
        cfg = AlertConfig(dry_run=False, slack_enabled=True, slack_webhook_url="https://hooks.slack.com/x")
        a = Alerter(cfg)
        with patch.object(a, "_send_slack", return_value=True) as mock_slack:
            result = a.send_alert("error", {"message": "oops"})
        mock_slack.assert_called_once()
        assert result is True

    def test_non_dry_run_no_channels_enabled_returns_false(self):
        from quantaalpha.live.alerter import Alerter, AlertConfig
        cfg = AlertConfig(dry_run=False, email_enabled=False, slack_enabled=False)
        a = Alerter(cfg)
        result = a.send_alert("info", {"message": "hi"})
        assert result is False


# ────────────────────────────────────────────────────────────────────
# retrain_us_model — _update_live_yaml
# ────────────────────────────────────────────────────────────────────

class TestRetrainScript:

    def test_update_live_yaml_sets_meta_path(self, tmp_path):
        import sys
        scripts_dir = Path(__file__).parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from retrain_us_model import _update_live_yaml

        live_yaml = tmp_path / "live.yaml"
        live_yaml.write_text(yaml.dump({"model": {"meta_path": "old_path.json"}, "ibkr": {"port": 7497}}))

        _update_live_yaml(live_yaml, "new_path.json")

        updated = yaml.safe_load(live_yaml.read_text())
        assert updated["model"]["meta_path"] == "new_path.json"
        assert updated["ibkr"]["port"] == 7497  # other keys untouched

    def test_update_live_yaml_creates_model_key_if_missing(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from retrain_us_model import _update_live_yaml

        live_yaml = tmp_path / "live.yaml"
        live_yaml.write_text(yaml.dump({"ibkr": {}}))
        _update_live_yaml(live_yaml, "new.json")
        updated = yaml.safe_load(live_yaml.read_text())
        assert updated["model"]["meta_path"] == "new.json"

    def test_retrain_dry_run_writes_placeholder_meta(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import retrain_us_model as rm

        config_path = tmp_path / "backtest_us.yaml"
        factor_json = tmp_path / "factors.json"
        factor_json.write_text(json.dumps({"factors": {}}))
        # Minimal backtest config
        config_path.write_text(yaml.dump({
            "factor_source": {"custom": {"factor_json": str(factor_json)}},
        }))

        mock_runner = MagicMock()
        mock_runner.run.return_value = None

        with patch("quantaalpha.backtest.runner.BacktestRunner", return_value=mock_runner):
            meta = rm.retrain(
                config_path=config_path,
                factor_json=str(factor_json),
                model_dir=tmp_path / "models",
                dry_run=True,
            )

        assert meta.exists()
        data = json.loads(meta.read_text())
        assert data["dry_run"] is True
        assert "us_retrain_" in data["run_name"]

    def test_retrain_missing_factor_json_raises(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import retrain_us_model as rm

        config_path = tmp_path / "backtest_us.yaml"
        config_path.write_text(yaml.dump({}))

        with pytest.raises(ValueError, match="factor-json"):
            rm.retrain(
                config_path=config_path,
                factor_json=None,
                model_dir=tmp_path / "models",
                dry_run=True,
            )

    def test_retrain_nonexistent_factor_json_raises(self, tmp_path):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        import retrain_us_model as rm

        config_path = tmp_path / "backtest_us.yaml"
        config_path.write_text(yaml.dump({}))

        with pytest.raises(FileNotFoundError):
            rm.retrain(
                config_path=config_path,
                factor_json="/no/such/file.json",
                model_dir=tmp_path / "models",
                dry_run=True,
            )
