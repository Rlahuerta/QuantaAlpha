#!/usr/bin/env python3
"""Rebuild data/live/ledger.csv from existing PnL history files.

Scans ``data/live/pnl/`` and ``data/live/archive/`` for ``pnl_*.json``
files and reconstructs a complete CSV ledger.  Duplicate dates are
resolved by keeping the version from ``pnl/`` (live) over ``archive/``.

Usage:
    python scripts/rebuild_ledger.py                     # default paths
    python scripts/rebuild_ledger.py --pnl-dir data/live/pnl --archive-dir data/live/archive
    python scripts/rebuild_ledger.py --initial-capital 500000
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

LEDGER_COLUMNS = [
    "date", "initial_capital", "account_value", "daily_pnl",
    "cumulative_pnl", "daily_return", "cum_excess_return",
    "num_positions", "invested", "cash", "benchmark_return",
]


def _load_pnl_files(directory: Path) -> Dict[str, Dict]:
    """Load all pnl_*.json files from *directory*, keyed by date."""
    records: Dict[str, Dict] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.glob("pnl_*.json")):
        try:
            data = json.loads(path.read_text())
            date_str = data.get("date")
            if date_str:
                records[date_str] = data
        except Exception:
            continue
    return records


def rebuild(
    pnl_dir: Path,
    archive_dir: Path,
    output: Path,
    initial_capital: float,
) -> int:
    """Rebuild ledger CSV.  Returns number of rows written."""
    # Archive first, then live overwrites duplicates
    all_records: Dict[str, Dict] = {}
    all_records.update(_load_pnl_files(archive_dir))
    all_records.update(_load_pnl_files(pnl_dir))

    if not all_records:
        print("No PnL records found — nothing to rebuild.")
        return 0

    rows: List[Dict] = []
    for date_str in sorted(all_records):
        data = all_records[date_str]
        pnl = data.get("pnl", {})
        rows.append({
            "date": date_str,
            "initial_capital": data.get("initial_capital", initial_capital),
            "account_value": data.get("account_value", 0),
            "daily_pnl": pnl.get("daily_pnl", 0),
            "cumulative_pnl": pnl.get("cumulative_pnl", 0),
            "daily_return": pnl.get("daily_return", pnl.get("portfolio_return", 0)),
            "cum_excess_return": pnl.get("cumulative_excess_return", 0),
            "num_positions": pnl.get("num_positions", len(data.get("positions", {}))),
            "invested": 0,  # not available in old PnL files
            "cash": data.get("cash", 0),
            "benchmark_return": 0,  # not available in old PnL files
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Ledger rebuilt: {output} ({len(rows)} rows)")
    for row in rows:
        pnl_str = f"${float(row['daily_pnl']):+,.2f}"
        cum_str = f"${float(row['cumulative_pnl']):+,.2f}"
        print(f"  {row['date']}  acct=${float(row['account_value']):>12,.2f}  "
              f"daily={pnl_str:>12s}  cumul={cum_str:>12s}  pos={row['num_positions']}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild live trading ledger CSV")
    parser.add_argument("--pnl-dir", default="data/live/pnl", help="PnL history dir")
    parser.add_argument("--archive-dir", default="data/live/archive", help="Archive dir")
    parser.add_argument("--output", default="data/live/ledger.csv", help="Output CSV")
    parser.add_argument("--initial-capital", type=float, default=1_000_000,
                        help="Starting capital (default: 1000000)")
    args = parser.parse_args()
    rebuild(
        pnl_dir=Path(args.pnl_dir),
        archive_dir=Path(args.archive_dir),
        output=Path(args.output),
        initial_capital=args.initial_capital,
    )


if __name__ == "__main__":
    main()
