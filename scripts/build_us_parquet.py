#!/usr/bin/env python3
"""
Build US parquet bundles for the backtest runner's parquet-fallback path.

Creates:
  <output_dir>/sp500_prices.parquet   — long-format OHLCV (datetime, symbol, open, close, …)
  <output_dir>/benchmark_gspc.parquet — SPY open prices (datetime, open)

Usage:
  python scripts/build_us_parquet.py
  python scripts/build_us_parquet.py --h5 git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5 \
      --output git_ignore_folder/parquet_us
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("build_us_parquet")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def build_prices(h5_path: Path, output_dir: Path) -> Path:
    """Convert daily_pv.h5 → sp500_prices.parquet."""
    log.info(f"Loading US H5: {h5_path}")
    df = pd.read_hdf(str(h5_path), key="data")
    log.info(f"  Rows: {len(df):,}  Columns: {list(df.columns)}")

    prices = df[["$open", "$close", "$high", "$low", "$volume"]].copy()
    prices = prices.reset_index()
    prices.columns = ["datetime", "symbol", "open", "close", "high", "low", "volume"]
    prices["datetime"] = pd.to_datetime(prices["datetime"]).dt.normalize()
    prices["symbol"] = prices["symbol"].astype(str)
    prices = prices.sort_values(["symbol", "datetime"]).reset_index(drop=True)

    out = output_dir / "sp500_prices.parquet"
    prices.to_parquet(str(out), index=False)
    log.info(f"  Saved: {out}  ({len(prices):,} rows, {prices['symbol'].nunique()} symbols)")
    return out


def build_benchmark(output_dir: Path, start: str = "2015-01-01", end: str = "2026-12-31") -> Path:
    """Download SPY via yfinance → benchmark_gspc.parquet."""
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance not installed — pip install yfinance")

    log.info(f"Downloading SPY benchmark {start} → {end}...")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if spy.empty:
        raise SystemExit("SPY download returned empty DataFrame")

    spy = spy.reset_index()
    # Handle MultiIndex columns that yfinance sometimes returns
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = [c[0].lower() if c[1] == '' or c[1] == 'SPY' else '_'.join(c).lower()
                       for c in spy.columns]
    else:
        spy.columns = [c.lower() for c in spy.columns]

    # Normalise date column name
    date_col = next((c for c in spy.columns if "date" in c.lower()), spy.columns[0])
    spy = spy.rename(columns={date_col: "datetime"})
    spy["datetime"] = pd.to_datetime(spy["datetime"]).dt.normalize()

    bench = spy[["datetime", "open"]].copy().reset_index(drop=True)
    out = output_dir / "benchmark_gspc.parquet"
    bench.to_parquet(str(out), index=False)
    log.info(f"  Saved: {out}  ({len(bench):,} rows, {bench['datetime'].min()} → {bench['datetime'].max()})")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build US parquet bundles for backtest runner")
    parser.add_argument("--h5", default="git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5",
                        help="Path to US daily_pv.h5")
    parser.add_argument("--output", default="git_ignore_folder/parquet_us",
                        help="Output directory for parquet files")
    parser.add_argument("--spy-start", default="2015-01-01", help="SPY download start date")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    h5_path = Path(args.h5) if Path(args.h5).is_absolute() else repo_root / args.h5
    output_dir = Path(args.output) if Path(args.output).is_absolute() else repo_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if not h5_path.exists():
        raise SystemExit(f"H5 file not found: {h5_path}")

    build_prices(h5_path, output_dir)
    build_benchmark(output_dir, start=args.spy_start)
    log.info("Done — parquet bundle ready.")


if __name__ == "__main__":
    main()
