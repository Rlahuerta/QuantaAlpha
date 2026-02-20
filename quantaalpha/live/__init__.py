"""
quantaalpha.live — Daily US live trading pipeline

Modules:
  data_ingestor       — Fetch EOD OHLCV via yfinance, append to daily_pv.h5
  signal_generator    — Load trained LightGBM, compute factors, return ranked scores
  portfolio_constructor — TopkDropout without Qlib dependency
  ibkr_executor       — IBKR TWS paper/live order execution (Phase 3)
  position_tracker    — Daily P&L snapshot vs SPY benchmark
  scheduler           — APScheduler daily cron orchestration (Phase 3)
"""
from importlib import import_module as _import

__all__ = [
    "data_ingestor",
    "signal_generator",
    "portfolio_constructor",
]
