#!/usr/bin/env python3
"""Build compact parquet bundle from Yahoo SP500 CSV downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SP500 parquet bundle from Yahoo raw CSV files")
    parser.add_argument("--input-dir", required=True, help="Directory containing raw/*.csv from Yahoo downloader")
    parser.add_argument("--output-dir", required=True, help="Directory for parquet bundle and manifest")
    parser.add_argument("--target-start", default="2016-01-01")
    parser.add_argument("--target-end", default="2025-12-26")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    raw_dir = input_dir / "raw"
    if not raw_dir.exists():
        raise SystemExit(f"raw dir not found: {raw_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    benchmark_df = None
    for csv_file in sorted(raw_dir.glob("*.csv")):
        if csv_file.stem == "^GSPC":
            benchmark_df = pd.read_csv(csv_file, parse_dates=["datetime"])
            continue

        df = pd.read_csv(csv_file, parse_dates=["datetime"])
        if df.empty:
            continue
        df["symbol"] = csv_file.stem
        frames.append(df)

    if not frames:
        raise SystemExit("No symbol csv files found to build parquet bundle")

    prices = pd.concat(frames, ignore_index=True)
    prices = prices[["datetime", "symbol", "open", "high", "low", "close", "volume", "adjclose", "vwap_proxy"]]
    prices = prices.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    prices_path = out_dir / "sp500_prices.parquet"
    prices.to_parquet(prices_path, index=False)

    bench_path = None
    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_df = benchmark_df.sort_values("datetime").reset_index(drop=True)
        bench_path = out_dir / "benchmark_gspc.parquet"
        benchmark_df.to_parquet(bench_path, index=False)

    symbol_cov = (
        prices.groupby("symbol")["datetime"]
        .agg(min_date="min", max_date="max", rows="count")
        .reset_index()
    )
    cov_path = out_dir / "symbol_coverage.parquet"
    symbol_cov.to_parquet(cov_path, index=False)

    failures_path = input_dir / "failures.txt"
    failures = failures_path.read_text(encoding="utf-8").splitlines() if failures_path.exists() else []

    manifest = {
        "bundle_format": "parquet",
        "target_window": {"start": args.target_start, "end": args.target_end},
        "paths": {
            "prices": str(prices_path.resolve()),
            "symbol_coverage": str(cov_path.resolve()),
            "benchmark_gspc": str(bench_path.resolve()) if bench_path else None,
            "input_symbols": str((input_dir / "sp500_symbols.txt").resolve()),
            "input_instruments": str((input_dir / "instruments_sp500.txt").resolve()),
            "input_failures": str(failures_path.resolve()) if failures_path.exists() else None,
        },
        "stats": {
            "rows": int(len(prices)),
            "symbols": int(prices["symbol"].nunique()),
            "min_date": str(prices["datetime"].min().date()),
            "max_date": str(prices["datetime"].max().date()),
            "failed_symbols": int(len(failures)),
        },
    }

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "rows": manifest["stats"]["rows"], "symbols": manifest["stats"]["symbols"]}))


if __name__ == "__main__":
    main()

