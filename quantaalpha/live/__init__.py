"""
quantaalpha.live — Daily US live trading pipeline

Modules:
  data_ingestor         — Fetch EOD OHLCV via yfinance, append to daily_pv.h5
  signal_generator      — Load trained LightGBM, compute factors, return ranked scores
  portfolio_constructor — TopkDropout without Qlib dependency (+ risk controls)
  scheduler             — APScheduler daily cron: ingest → signal → save orders
  ibkr_executor         — IBKR TWS paper/live order execution (dry_run safe)
  position_tracker      — Daily P&L snapshot vs SPY benchmark
  risk_monitor          — KillSwitch (daily loss limit) + GateChecker (paper→live gate)
  alerter               — Email + Slack alerting for kill-switch / drawdown events
"""

__all__ = [
    "data_ingestor",
    "signal_generator",
    "portfolio_constructor",
    "scheduler",
    "ibkr_executor",
    "position_tracker",
    "risk_monitor",
    "alerter",
]
