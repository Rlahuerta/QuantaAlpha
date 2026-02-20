"""
risk_monitor.py — Kill-switch and paper-trading gate checker.

KillSwitch
----------
Checks whether today's P&L has breached the daily loss limit.  When
triggered the scheduler should skip order generation and alert.

    monitor = KillSwitch(daily_loss_limit_pct=0.03, capital=1_000_000)
    if monitor.is_triggered(daily_pnl=-35_000):
        # halt trading today

GateChecker
-----------
Reads the P&L history and decides whether paper-trading has met the
promotion criteria for live capital.

    gate = GateChecker(min_sharpe=1.5, max_mdd=0.10, min_days=60)
    result = gate.check("data/live/pnl")
    if result.passed:
        print("Ready for live capital!")
"""

from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kill-switch
# ---------------------------------------------------------------------------


class KillSwitch:
    """
    Halt daily trading when the intraday loss exceeds a configured limit.

    Parameters
    ----------
    daily_loss_limit_pct : float
        Maximum allowed daily loss as a fraction of capital (default 0.03 = 3%).
    capital : float
        Portfolio capital used to convert the percentage limit to dollar terms.
    """

    def __init__(
        self,
        daily_loss_limit_pct: float = 0.03,
        capital: float = 1_000_000.0,
    ) -> None:
        if not 0 < daily_loss_limit_pct < 1:
            raise ValueError("daily_loss_limit_pct must be in (0, 1)")
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.capital = capital

    @property
    def limit_usd(self) -> float:
        return self.capital * self.daily_loss_limit_pct

    def is_triggered(self, daily_pnl: float) -> bool:
        """Return True when daily_pnl is negative and exceeds the loss limit."""
        triggered = daily_pnl < -self.limit_usd
        if triggered:
            log.warning(
                "Kill-switch triggered: daily P&L $%.0f < -$%.0f (%.1f%% limit)",
                daily_pnl,
                self.limit_usd,
                self.daily_loss_limit_pct * 100,
            )
        return triggered

    def check_return(self, daily_return: float) -> bool:
        """Return True when daily_return (fraction) breaches the limit."""
        return self.is_triggered(daily_return * self.capital)


# ---------------------------------------------------------------------------
# Gate checker
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Outcome of a paper-trading gate check."""

    passed: bool
    days: int
    sharpe: float
    mdd: float          # max drawdown (negative fraction, e.g. -0.08 = −8%)
    total_return: float # cumulative return fraction
    reason: str         # human-readable explanation


class GateChecker:
    """
    Evaluate whether paper-trading results meet live-capital promotion criteria.

    Parameters
    ----------
    min_sharpe : float
        Minimum annualised Sharpe ratio (default 1.5).
    max_mdd : float
        Maximum acceptable max-drawdown as a *positive* fraction (default 0.10 = 10%).
    min_days : int
        Minimum number of trading days required (default 60).
    risk_free_rate : float
        Annual risk-free rate used in Sharpe calculation (default 0.05).
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        min_sharpe: float = 1.5,
        max_mdd: float = 0.10,
        min_days: int = 60,
        risk_free_rate: float = 0.05,
    ) -> None:
        self.min_sharpe = min_sharpe
        self.max_mdd = max_mdd
        self.min_days = min_days
        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_history(pnl_dir: Path) -> List[dict]:
        """Load all pnl_*.json snapshots, sorted chronologically."""
        files = sorted(pnl_dir.glob("pnl_*.json"))
        history = []
        for f in files:
            try:
                history.append(json.loads(f.read_text()))
            except Exception:
                log.warning("Could not parse %s", f)
        return history

    def _daily_returns(self, history: List[dict]) -> List[float]:
        """Extract daily portfolio returns (fraction of capital)."""
        returns = []
        for snap in history:
            # Prefer portfolio_return if available; else derive from daily_pnl / capital
            r = snap.get("portfolio_return")
            if r is None:
                pnl = snap.get("daily_pnl", 0.0)
                capital = snap.get("portfolio_value") or snap.get("capital") or 1_000_000.0
                r = pnl / capital if capital else 0.0
            returns.append(float(r))
        return returns

    def _sharpe(self, returns: List[float]) -> float:
        if len(returns) < 2:
            return 0.0
        n = len(returns)
        mean_r = sum(returns) / n
        daily_rf = self.risk_free_rate / self.TRADING_DAYS_PER_YEAR
        excess = [r - daily_rf for r in returns]
        mean_ex = sum(excess) / n
        variance = sum((r - mean_ex) ** 2 for r in excess) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 1e-10
        return mean_ex / std * math.sqrt(self.TRADING_DAYS_PER_YEAR)

    def _max_drawdown(self, returns: List[float]) -> float:
        """Return max drawdown as a negative fraction (e.g. -0.08)."""
        cum = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in returns:
            cum *= (1 + r)
            if cum > peak:
                peak = cum
            dd = (cum - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return max_dd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, pnl_dir: str | Path) -> GateResult:
        """
        Evaluate promotion readiness from the P&L snapshot directory.

        Parameters
        ----------
        pnl_dir : str | Path
            Directory containing ``pnl_YYYY-MM-DD.json`` files.

        Returns
        -------
        GateResult
        """
        pnl_dir = Path(pnl_dir)
        history = self._load_history(pnl_dir)
        days = len(history)

        if days < self.min_days:
            return GateResult(
                passed=False,
                days=days,
                sharpe=0.0,
                mdd=0.0,
                total_return=0.0,
                reason=f"Insufficient history: {days} days < {self.min_days} required",
            )

        returns = self._daily_returns(history)
        sharpe = self._sharpe(returns)
        mdd = self._max_drawdown(returns)      # negative value
        total_return = sum(returns)            # approximate cumulative

        reasons = []
        if sharpe < self.min_sharpe:
            reasons.append(f"Sharpe {sharpe:.2f} < {self.min_sharpe}")
        if abs(mdd) > self.max_mdd:
            reasons.append(f"MDD {mdd:.1%} > -{self.max_mdd:.0%}")

        passed = len(reasons) == 0
        reason = "All criteria met — ready for live capital" if passed else "; ".join(reasons)

        log.info(
            "Gate check: days=%d sharpe=%.2f mdd=%.1%% passed=%s",
            days, sharpe, mdd * 100, passed,
        )
        return GateResult(
            passed=passed,
            days=days,
            sharpe=sharpe,
            mdd=mdd,
            total_return=total_return,
            reason=reason,
        )
