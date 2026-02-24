"""Daily position and P&L tracker.

Maintains a rolling JSON snapshot of current equity positions and
computes cumulative excess return vs a benchmark (default: SPY).

Snapshot format (``data/live/positions.json``)::

    {
      "date": "2026-02-20",
      "positions": {"AAPL": 100, "MSFT": 80, ...},
      "account_value": 1050234.50,
      "pnl": {
        "daily_pnl": 1234.56,
        "cumulative_pnl": 5678.90,
        "daily_excess_return": 0.0023,
        "cumulative_excess_return": 0.0567
      }
    }

P&L history is written to ``data/live/pnl/pnl_{date}.json`` for each day.
"""

from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PositionTracker:
    """Track daily positions and compute P&L vs benchmark.

    Parameters
    ----------
    positions_file:
        Path to the current-state JSON (overwritten each day).
    pnl_dir:
        Directory where per-day P&L history files are written.
    benchmark_ticker:
        Ticker used as the benchmark return (default "SPY").
    """

    def __init__(
        self,
        positions_file: str | Path = "data/live/positions.json",
        pnl_dir: str | Path = "data/live/pnl",
        benchmark_ticker: str = "SPY",
    ) -> None:
        self.positions_file = Path(positions_file)
        self.pnl_dir = Path(pnl_dir)
        self.benchmark_ticker = benchmark_ticker

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def load_state(self) -> Dict:
        """Load the current snapshot.  Returns ``{}`` if file absent."""
        if not self.positions_file.exists():
            return {}
        try:
            return json.loads(self.positions_file.read_text())
        except Exception as exc:
            logger.warning("Could not load state: %s", exc)
            return {}

    def save_state(self, state: Dict) -> None:
        """Persist snapshot to *positions_file*."""
        self.positions_file.parent.mkdir(parents=True, exist_ok=True)
        self.positions_file.write_text(json.dumps(state, indent=2))

    def update_positions(
        self,
        trade_results: List[Dict],
        current_positions: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """Apply executed trades to current positions.

        Parameters
        ----------
        trade_results:
            List of ``{"ticker": str, "shares": int}`` dicts (signed).
        current_positions:
            Starting position dict.  If *None*, loads from disk.

        Returns
        -------
        Updated ``{ticker: shares}`` dict (zero-share positions dropped).
        """
        if current_positions is None:
            state = self.load_state()
            current_positions = state.get("positions", {})

        updated = dict(current_positions)
        for trade in trade_results:
            ticker = trade["ticker"]
            shares = int(trade["shares"])
            updated[ticker] = updated.get(ticker, 0) + shares

        # Drop zero / tiny positions
        updated = {t: s for t, s in updated.items() if s != 0}
        return updated

    # ------------------------------------------------------------------
    # P&L computation
    # ------------------------------------------------------------------

    def compute_daily_pnl(
        self,
        positions_start: Dict[str, int],
        prices_start: Dict[str, float],
        prices_end: Dict[str, float],
        account_value_start: float,
        benchmark_return: float,
    ) -> Dict:
        """Compute daily P&L and excess return vs benchmark.

        Parameters
        ----------
        positions_start:
            Holdings at start of day (before any rebalancing).
        prices_start, prices_end:
            Price dicts keyed by ticker.
        account_value_start:
            Portfolio value at start of day.
        benchmark_return:
            Benchmark daily return (e.g. SPY return for the day).

        Returns
        -------
        Dict with keys: daily_pnl, portfolio_return, daily_excess_return,
        num_positions.
        """
        if not positions_start or account_value_start <= 0:
            return {
                "daily_pnl": 0.0,
                "portfolio_return": 0.0,
                "daily_excess_return": 0.0,
                "num_positions": 0,
            }

        pnl = 0.0
        counted = 0
        for ticker, shares in positions_start.items():
            p0 = prices_start.get(ticker)
            p1 = prices_end.get(ticker)
            if p0 and p1 and p0 > 0:
                pnl += shares * (p1 - p0)
                counted += 1

        portfolio_return = pnl / account_value_start if account_value_start > 0 else 0.0
        excess_return = portfolio_return - benchmark_return

        return {
            "daily_pnl": round(pnl, 2),
            "portfolio_return": round(portfolio_return, 6),
            "daily_excess_return": round(excess_return, 6),
            "num_positions": counted,
        }

    def mark_to_market(
        self,
        positions: Dict[str, int],
        prices: Dict[str, float],
    ) -> float:
        """Compute portfolio value as sum(shares × price) for all positions."""
        total = 0.0
        for ticker, shares in positions.items():
            price = prices.get(ticker, 0.0)
            if price > 0:
                total += shares * price
        return round(total, 2)

    def record_day(
        self,
        positions: Dict[str, int],
        account_value: float,
        daily_pnl_dict: Dict,
        as_of: Optional[date] = None,
        prices: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Build and persist the daily snapshot.

        Cumulative P&L is accumulated from the previous day's state.
        Idempotent: if already ran today, overwrites without re-accumulating.

        For paper trading, ``account_value`` should already include daily P&L
        (i.e. ``prev_account + daily_pnl``).  Rebalancing at current prices is
        zero-sum, so this drift approach is exact.

        When *prices* is provided, ``cash`` is derived as
        ``account_value − mark_to_market`` so both invested and uninvested
        capital are tracked.

        Returns the full state dict written to disk.
        """
        today = (as_of or date.today()).isoformat()
        prev_state = self.load_state()

        # Bug fix #2: compute cash as residual (positions may not use 100% capital)
        cash = 0.0
        if prices and positions:
            mtm = self.mark_to_market(positions, prices)
            cash = round(account_value - mtm, 2)

        # Bug fix #1: idempotency — detect same-day re-run
        prev_date = prev_state.get("date")
        if prev_date == today:
            logger.info("Same-day re-run detected for %s — overwriting without re-accumulating", today)
            prev_cum = prev_state.get("pnl", {}).get("_prev_day_cumulative_excess_return", 0.0)
            prev_cum_pnl = prev_state.get("pnl", {}).get("_prev_day_cumulative_pnl", 0.0)
        else:
            prev_cum = prev_state.get("pnl", {}).get("cumulative_excess_return", 0.0)
            prev_cum_pnl = prev_state.get("pnl", {}).get("cumulative_pnl", 0.0)

        state = {
            "date": today,
            "positions": positions,
            "account_value": round(account_value, 2),
            "cash": cash,
            "pnl": {
                "daily_pnl": daily_pnl_dict.get("daily_pnl", 0.0),
                "cumulative_pnl": round(prev_cum_pnl + daily_pnl_dict.get("daily_pnl", 0.0), 2),
                "daily_return": daily_pnl_dict.get("portfolio_return", 0.0),
                "daily_excess_return": daily_pnl_dict.get("daily_excess_return", 0.0),
                "cumulative_excess_return": round(
                    prev_cum + daily_pnl_dict.get("daily_excess_return", 0.0), 6
                ),
                # Bug fix #3: actual position count, not price-available count
                "num_positions": len(positions),
                # Store previous-day base for idempotent re-runs
                "_prev_day_cumulative_pnl": prev_cum_pnl,
                "_prev_day_cumulative_excess_return": prev_cum,
            },
        }

        self.save_state(state)

        # Append to history
        self.pnl_dir.mkdir(parents=True, exist_ok=True)
        history_file = self.pnl_dir / f"pnl_{today}.json"
        history_file.write_text(json.dumps(state, indent=2))
        logger.info(
            "P&L recorded: date=%s daily_pnl=%.2f excess=%.4f cumulative=%.4f",
            today,
            state["pnl"]["daily_pnl"],
            state["pnl"]["daily_excess_return"],
            state["pnl"]["cumulative_excess_return"],
        )
        return state

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def load_pnl_history(self, days: int = 30) -> List[Dict]:
        """Load last *days* of P&L history records, sorted ascending."""
        if not self.pnl_dir.exists():
            return []
        records = []
        for path in sorted(self.pnl_dir.glob("pnl_*.json")):
            try:
                records.append(json.loads(path.read_text()))
            except Exception:
                pass
        return records[-days:] if len(records) > days else records

    def summary(self) -> Dict:
        """Return a concise performance summary from the current state."""
        state = self.load_state()
        if not state:
            return {"status": "no data"}
        pnl = state.get("pnl", {})
        return {
            "date": state.get("date"),
            "num_positions": len(state.get("positions", {})),
            "account_value": state.get("account_value"),
            "cumulative_pnl": pnl.get("cumulative_pnl"),
            "cumulative_excess_return": pnl.get("cumulative_excess_return"),
            "daily_excess_return": pnl.get("daily_excess_return"),
        }
