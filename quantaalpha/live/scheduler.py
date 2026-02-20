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
        """Generate signals, compute orders, persist to JSON.

        Returns the order dict that was written to disk (or ``{}`` on
        failure).
        """
        model_cfg = self.config["model"]
        port_cfg = self.config.get("portfolio", {})
        out_cfg = self.config.get("output", {})

        # 1. Load last-known positions
        positions_file = Path(out_cfg.get("positions_file", "data/live/positions.json"))
        current_positions: Dict[str, int] = {}
        if positions_file.exists():
            try:
                state = json.loads(positions_file.read_text())
                current_positions = {str(k): int(v) for k, v in state.get("positions", {}).items()}
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not load positions file: %s", exc)

        # 2. Generate scores
        sg = SignalGenerator.from_meta(model_cfg["meta_path"])
        as_of = datetime.now()
        scores: Dict[str, float] = sg.generate(as_of_date=as_of)
        if not scores:
            logger.warning("Signal generator returned empty scores; skipping orders")
            return {}

        # 3. Latest prices for position sizing (last close from H5 window)
        lookback = self.config.get("data", {}).get("lookback_days", 300)
        window_df = sg._load_h5_window(as_of)
        prices: Dict[str, float] = {}
        if window_df is not None and not window_df.empty:
            inst_col = "instrument" if "instrument" in window_df.columns else window_df.columns[1]
            close_col = "close" if "close" in window_df.columns else "Close"
            dt_col = "datetime" if "datetime" in window_df.columns else window_df.columns[0]
            latest = (
                window_df.sort_values(dt_col)
                .groupby(inst_col)[close_col]
                .last()
            )
            prices = latest.to_dict()

        # 4. Rebalance
        constructor = PortfolioConstructor(
            topk=int(port_cfg.get("topk", 20)),
            n_drop=int(port_cfg.get("n_drop", 5)),
            capital=float(port_cfg.get("capital", 1_000_000)),
        )
        result = constructor.rebalance(
            scores=scores,
            positions=current_positions,
            prices=prices,
        )

        # 5. Persist orders
        orders_dir = Path(out_cfg.get("orders_dir", "data/live"))
        orders_dir.mkdir(parents=True, exist_ok=True)
        today = date.today().isoformat()
        orders_file = orders_dir / f"pending_orders_{today}.json"
        order_data: Dict[str, Any] = {
            "date": today,
            "as_of": as_of.isoformat(),
            "scores_count": len(scores),
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
            "target_positions": result.target_positions,
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
