"""
Build CoSTEER HDF5 data for S&P500 factor evaluation.

Reads: data/sp500_parquet_bundle/sp500_prices.parquet
Writes:
  git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5  (all symbols)
  git_ignore_folder/factor_implementation_source_data_us_debug/daily_pv.h5  (first 100)

Usage:
    python scripts/build_us_hdf5_from_parquet.py [--n_debug 100]
"""
import argparse
import os
from pathlib import Path

import pandas as pd


def build(parquet_path: str, output_dir: str, output_debug_dir: str, n_debug: int = 100) -> None:
    print("Loading parquet ...")
    raw = pd.read_parquet(parquet_path)

    # Normalise datetime to midnight (Qlib convention)
    raw["datetime"] = pd.to_datetime(raw["datetime"]).dt.normalize()
    raw = raw.rename(columns={"vwap_proxy": "vwap"})

    # Build MultiIndex (datetime, instrument) sorted — matching CN HDF5 structure
    raw = raw.set_index(["datetime", "symbol"])
    raw.index.names = ["datetime", "instrument"]
    raw = raw.sort_index()

    # Create $ prefixed columns + compute $return
    cols_map = {
        "open": "$open",
        "close": "$close",
        "high": "$high",
        "low": "$low",
        "volume": "$volume",
        "vwap": "$vwap",
    }
    df = raw[list(cols_map.keys())].rename(columns=cols_map)

    # $return: per-instrument daily pct change
    df["$return"] = (
        df.groupby(level="instrument")["$close"].pct_change().fillna(0)
    )

    print(f"Full dataset shape: {df.shape}")
    print(f"Date range: {df.index.get_level_values('datetime').min()} → {df.index.get_level_values('datetime').max()}")
    print(f"Instruments: {df.index.get_level_values('instrument').nunique()}")

    # Write full dataset
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_path = out / "daily_pv.h5"
    df.to_hdf(str(full_path), key="data", mode="w", complevel=4)
    print(f"Written full HDF5: {full_path} ({full_path.stat().st_size / 1e6:.1f} MB)")

    # Write debug dataset (first n_debug instruments)
    instruments = sorted(df.index.get_level_values("instrument").unique())[:n_debug]
    df_debug = df.loc[df.index.get_level_values("instrument").isin(instruments)]
    out_debug = Path(output_debug_dir)
    out_debug.mkdir(parents=True, exist_ok=True)
    debug_path = out_debug / "daily_pv.h5"
    df_debug.to_hdf(str(debug_path), key="data", mode="w", complevel=4)
    print(f"Written debug HDF5: {debug_path} ({debug_path.stat().st_size / 1e6:.1f} MB)")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build US CoSTEER HDF5 from SP500 parquet bundle")
    parser.add_argument("--parquet", default="data/sp500_parquet_bundle/sp500_prices.parquet")
    parser.add_argument("--output", default="git_ignore_folder/factor_implementation_source_data_us")
    parser.add_argument("--output-debug", default="git_ignore_folder/factor_implementation_source_data_us_debug")
    parser.add_argument("--n-debug", type=int, default=100)
    args = parser.parse_args()
    build(args.parquet, args.output, args.output_debug, args.n_debug)


if __name__ == "__main__":
    main()
