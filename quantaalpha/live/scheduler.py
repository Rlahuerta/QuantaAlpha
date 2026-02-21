"""Daily EOD trading scheduler using APScheduler.
...
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from quantaalpha.live.data_ingestor import DataIngestor
from quantaalpha.live.signal_generator import SignalGenerator
from quantaalpha.live.portfolio_constructor import PortfolioConstructor
from quantaalpha.live.position_tracker import PositionTracker
from quantaalpha.live.risk_monitor import KillSwitch

logger = logging.getLogger(__name__)


class TradingScheduler:
    """Daily EOD pipeline: ingest → signal → save orders.

    Parameters
    ----------
    config_path:
        Path to ``configs/live.yaml`` (or equivalent).
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path) as fh:
            self.config: Dict[str, Any] = yaml.safe_load(fh)
        self._scheduler: Any = None
        self._last_orders: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def run_ingest(self) -> int:
        """Fetch latest EOD data.  Returns number of new rows added."""
        h5_path = self.config["data"]["h5_path"]
        ingestor = DataIngestor(h5_path=h5_path)
        rows_added = ingestor.ingest(force_full=False)
        logger.info("Ingest complete: %d new rows", rows_added)
        return rows_added

    def run_signal(self) -> Dict:
        """Generate signals, compute orders, track P&L, persist to JSON.

        Flow:
        1. Load previous positions & prices
        2. Compute daily P&L from overnight holdings
        3. KillSwitch check — abort if daily loss exceeds limit
        4. Generate new scores via SignalGenerator
        5. Rebalance → target portfolio + orders
        6. Record day via PositionTracker (P&L + new positions)
        7. Persist pending orders JSON

        Returns the order dict written to disk (or ``{}`` on failure).
        """
        model_cfg = self.config["model"]
        port_cfg = self.config.get("portfolio", {})
        out_cfg = self.config.get("output", {})
        risk_cfg = self.config.get("risk", {})
        capital = float(port_cfg.get("capital", 1_000_000))

        # -- Setup helpers --
        tracker = PositionTracker(
            positions_file=out_cfg.get("positions_file", "data/live/positions.json"),
            pnl_dir=out_cfg.get("pnl_dir", "data/live/pnl"),
        )
        kill_switch = KillSwitch(
            daily_loss_limit_pct=float(risk_cfg.get("daily_loss_limit_pct", 0.03)),
            capital=capital,
        )

        # 1. Load last-known positions
        prev_state = tracker.load_state()
        current_positions: Dict[str, int] = {
            str(k): int(v)
            for k, v in prev_state.get("positions", {}).items()
        }
        prev_account_value = prev_state.get("account_value", capital)

        # 2. Generate scores (also gives us access to H5 prices)
        h5_path = self.config.get("data", {}).get("h5_path")
        sg = SignalGenerator.from_meta(
            model_cfg["meta_path"],
            **({"h5_path": h5_path} if h5_path else {}),
        )
        scores: Dict[str, float] = sg.generate(as_of_date=None)
        if not scores:
            logger.warning("Signal generator returned empty scores; skipping orders")
            return {}

        # 3. Extract today's and yesterday's prices from H5
        prices_today: Dict[str, float] = {}
        prices_yesterday: Dict[str, float] = {}
        benchmark_return: float = 0.0
        try:
            import pandas as pd
            window_df = sg._load_h5_window(pd.Timestamp("today"))
            if window_df is not None and not window_df.empty:
                close_col = "$close" if "$close" in window_df.columns else "close"
                if close_col in window_df.columns:
                    closes = window_df[close_col].unstack(level="instrument")
                    if len(closes) >= 1:
                        prices_today = closes.iloc[-1].dropna().to_dict()
                    if len(closes) >= 2:
                        prices_yesterday = closes.iloc[-2].dropna().to_dict()
                        # Approximate SPY benchmark return
                        spy_today = prices_today.get("SPY")
                        spy_yest = prices_yesterday.get("SPY")
                        if spy_today and spy_yest and spy_yest > 0:
                            benchmark_return = (spy_today - spy_yest) / spy_yest
        except Exception as exc:
            logger.warning("Could not extract prices from H5: %s", exc)

        # 4. Compute daily P&L from overnight holdings
        daily_pnl_dict = tracker.compute_daily_pnl(
            positions_start=current_positions,
            prices_start=prices_yesterday or prices_today,
            prices_end=prices_today,
            account_value_start=prev_account_value,
            benchmark_return=benchmark_return,
        )

        # 5. KillSwitch — halt if daily loss exceeds limit
        daily_pnl = daily_pnl_dict.get("daily_pnl", 0.0)
        if current_positions and kill_switch.is_triggered(daily_pnl):
            logger.warning("Kill-switch active — skipping order generation")
            # Still record the bad day
            account_value = prev_account_value + daily_pnl
            tracker.record_day(
                positions=current_positions,
                account_value=account_value,
                daily_pnl_dict=daily_pnl_dict,
            )
            return {"kill_switch": True, "daily_pnl": daily_pnl}

        # 6. Rebalance
        constructor = PortfolioConstructor(
            topk=int(port_cfg.get("topk", 20)),
            n_drop=int(port_cfg.get("n_drop", 5)),
            capital=capital,
            max_position_pct=float(port_cfg.get("max_position_pct", 0.05)),
            min_adv=float(port_cfg.get("min_adv", 0)),
        )
        result = constructor.rebalance(
            scores=scores,
            positions=current_positions,
            prices=prices_today,
        )

        # 7. Record day — use target portfolio as new positions (paper = instant fill)
        new_positions = result.target_portfolio
        account_value = prev_account_value + daily_pnl
        tracker.record_day(
            positions=new_positions,
            account_value=round(account_value, 2),
            daily_pnl_dict=daily_pnl_dict,
        )

        # 8. Persist orders
        orders_dir = Path(out_cfg.get("orders_dir", "data/live"))
        orders_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        orders_file = orders_dir / f"pending_orders_{today}.json"
        order_data: Dict[str, Any] = {
            "date": today,
            "as_of": today,
            "scores_count": len(scores),
            "daily_pnl": daily_pnl,
            "account_value": round(account_value, 2),
            "orders": [
                {
                    "ticker": o.ticker,
                    "shares": o.shares,
                    "action": o.action,
                    "price": o.price,
                    "reason": o.reason,
                }
                for o in result.orders
            ],
            "target_positions": result.target_portfolio,
        }
        orders_file.write_text(json.dumps(order_data, indent=2))
        logger.info("Orders saved: %s (%d orders)", orders_file, len(result.orders))
        self._last_orders = order_data
        return order_data

    # ------------------------------------------------------------------
    # Scheduler control
    # ------------------------------------------------------------------

    def _build_scheduler(self, blocking: bool) -> Any:
        if blocking:
            from apscheduler.schedulers.blocking import BlockingScheduler
            return BlockingScheduler(timezone=self._timezone())
        from apscheduler.schedulers.background import BackgroundScheduler
        return BackgroundScheduler(timezone=self._timezone())

    def _timezone(self) -> str:
        return self.config.get("schedule", {}).get("timezone", "America/New_York")

    def _parse_time(self, key: str, default: str) -> tuple[int, int]:
        t = self.config.get("schedule", {}).get(key, default)
        h, m = map(int, t.split(":"))
        return h, m

    def start(self, blocking: bool = True) -> None:
        """Start the scheduler.

        Parameters
        ----------
        blocking:
            When *True* (default) the call blocks until the process is
            interrupted.  Pass *False* for tests or embedding in a larger
            application.
        """
        ingest_h, ingest_m = self._parse_time("ingest_time", "16:30")
        signal_h, signal_m = self._parse_time("signal_time", "17:00")

        self._scheduler = self._build_scheduler(blocking)
        self._scheduler.add_job(
            self.run_ingest,
            "cron",
            hour=ingest_h,
            minute=ingest_m,
            id="ingest_job",
            name="EOD Data Ingest",
            misfire_grace_time=300,
        )
        self._scheduler.add_job(
            self.run_signal,
            "cron",
            hour=signal_h,
            minute=signal_m,
            id="signal_job",
            name="Signal Generation + Order Save",
            misfire_grace_time=300,
        )
        logger.info(
            "Scheduler starting — ingest=%02d:%02d signal=%02d:%02d tz=%s",
            ingest_h, ingest_m, signal_h, signal_m, self._timezone(),
        )
        if blocking:
            try:
                self._scheduler.start()
            except (KeyboardInterrupt, SystemExit):
                logger.info("Scheduler stopped")
        else:
            self._scheduler.start()

    def stop(self) -> None:
        """Gracefully stop a non-blocking scheduler."""
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped")

    def get_jobs(self) -> List[Dict]:
        """Return summary of scheduled jobs (id, name, next_run_time)."""
        if self._scheduler is None:
            return []
        return [
            {
                "id": j.id,
                "name": j.name,
                "next_run_time": str(j.next_run_time),
            }
            for j in self._scheduler.get_jobs()
        ]


def main() -> None:  # pragma: no cover
    """CLI entry point: ``python -m quantaalpha.live.scheduler configs/live.yaml``."""
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/live.yaml"
    TradingScheduler(config_path).start()


if __name__ == "__main__":  # pragma: no cover
    main()
