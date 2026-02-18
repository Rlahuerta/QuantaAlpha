"""
Build Qlib binary data for S&P500 from the pre-built parquet bundle.

Input:  data/sp500_parquet_bundle/sp500_prices.parquet
Output: data/qlib/us_data_2025/

Usage:
    python scripts/build_us_qlib_from_parquet.py [--output data/qlib/us_data_2025]
"""
import argparse
import os
import struct
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["open", "high", "low", "close", "volume", "adjclose", "vwap_proxy"]
# Map parquet column -> Qlib feature name
FEATURE_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "adjclose": "adjclose",
    "vwap_proxy": "vwap",
}


def write_bin(path: Path, start_idx: int, values: np.ndarray) -> None:
    """Write a Qlib .bin feature file: [start_idx float32, val1, val2, ...]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.concatenate([[start_idx], values]).astype("<f")
    with open(path, "wb") as fp:
        arr.tofile(fp)


def build(parquet_path: str, output_dir: str) -> None:
    out = Path(output_dir)

    print("Loading parquet ...")
    df = pd.read_parquet(parquet_path)
    # Normalise datetime to date-only string (trading date)
    df["date"] = pd.to_datetime(df["datetime"]).dt.normalize()
    df = df.set_index(["date", "symbol"]).sort_index()

    # Build calendar from all unique dates in the data
    all_dates = sorted(df.index.get_level_values("date").unique())
    calendar_strs = [d.strftime("%Y-%m-%d") for d in all_dates]
    date_to_idx: dict[pd.Timestamp, int] = {d: i for i, d in enumerate(all_dates)}

    # Write calendar
    cal_path = out / "calendars" / "day.txt"
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.write_text("\n".join(calendar_strs) + "\n")
    print(f"Calendar: {len(calendar_strs)} trading days  [{calendar_strs[0]} → {calendar_strs[-1]}]")

    # Group by symbol for efficient iteration
    symbols = sorted(df.index.get_level_values("symbol").unique())
    print(f"Processing {len(symbols)} symbols ...")

    instr_rows = []

    for sym in symbols:
        sym_df = df.loc[(slice(None), sym), :].copy()
        sym_df.index = sym_df.index.get_level_values("date")
        sym_df = sym_df.sort_index()

        if sym_df.empty:
            continue

        start_date = sym_df.index.min()
        end_date = sym_df.index.max()
        start_idx = date_to_idx[start_date]
        end_idx = date_to_idx[end_date]
        n = end_idx - start_idx + 1

        instr_rows.append(f"{sym}\t{start_date.strftime('%Y-%m-%d')}\t{end_date.strftime('%Y-%m-%d')}")

        feat_dir = out / "features" / sym.lower()
        feat_dir.mkdir(parents=True, exist_ok=True)

        for src_col, qlib_name in FEATURE_MAP.items():
            values = np.full(n, np.nan, dtype=np.float32)
            for d, row in sym_df[[src_col]].iterrows():
                idx_in_range = date_to_idx[d] - start_idx
                values[idx_in_range] = row[src_col]
            write_bin(feat_dir / f"{qlib_name}.day.bin", start_idx, values)

        # factor = adjclose / close (adjustment factor for splits/dividends)
        factor_values = np.full(n, np.nan, dtype=np.float32)
        for d, row in sym_df[["adjclose", "close"]].iterrows():
            idx_in_range = date_to_idx[d] - start_idx
            if row["close"] != 0:
                factor_values[idx_in_range] = row["adjclose"] / row["close"]
            else:
                factor_values[idx_in_range] = 1.0
        write_bin(feat_dir / "factor.day.bin", start_idx, factor_values)

    # Write instruments file
    instr_dir = out / "instruments"
    instr_dir.mkdir(parents=True, exist_ok=True)
    instr_file = instr_dir / "sp500.txt"
    instr_file.write_text("\n".join(instr_rows) + "\n")
    # all.txt (same content)
    (instr_dir / "all.txt").write_text("\n".join(instr_rows) + "\n")
    print(f"Instruments: {len(instr_rows)} symbols written to {instr_file}")
    print(f"Done! Qlib data written to: {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build US Qlib binary data from SP500 parquet bundle")
    parser.add_argument("--parquet", default="data/sp500_parquet_bundle/sp500_prices.parquet")
    parser.add_argument("--output", default="data/qlib/us_data_2025")
    args = parser.parse_args()
    build(args.parquet, args.output)


if __name__ == "__main__":
    main()
