"""
data_ingestor.py — Fetch EOD OHLCV for the US universe and append to daily_pv.h5.

Usage (daily cron, ~4:30 PM ET after market close):
    from quantaalpha.live.data_ingestor import DataIngestor
    ingestor = DataIngestor(h5_path="git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5")
    new_rows = ingestor.ingest()   # returns count of new rows added

CLI:
    python -m quantaalpha.live.data_ingestor --h5 <path> [--tickers-file <path>]
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_H5 = _REPO_ROOT / "git_ignore_folder" / "factor_implementation_source_data_us" / "daily_pv.h5"
_DEFAULT_INSTRUMENTS = _REPO_ROOT / "git_ignore_folder" / "factor_implementation_source_data_us" / "instruments.txt"


def _engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert raw yfinance OHLCV to the daily_pv.h5 schema.

    Input:  MultiIndex(datetime, instrument), columns: open/high/low/close/volume
    Output: MultiIndex(datetime, instrument), columns: $open/$high/$low/$close/$volume/$vwap/$return (float32)
    """
    df = raw.copy()
    df = df.dropna(subset=["close"])

    rename = {"open": "$open", "high": "$high", "low": "$low", "close": "$close", "volume": "$volume"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["$vwap"] = (df["$high"] + df["$low"] + df["$close"]) / 3.0

    # Per-instrument % return; fill first-day NaN with 0
    df["$return"] = (
        df.groupby("instrument", group_keys=False)["$close"]
        .transform(lambda x: x.pct_change(fill_method=None))
        .fillna(0)
    )
    keep = ["$open", "$high", "$low", "$close", "$volume", "$vwap", "$return"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.astype("float32")
    df.index.names = ["datetime", "instrument"]
    return df.sort_index()


class DataIngestor:
    """Incremental EOD data ingestion for the US H5 store."""

    def __init__(
        self,
        h5_path: str | Path = _DEFAULT_H5,
        instruments_file: str | Path = _DEFAULT_INSTRUMENTS,
        tickers: Optional[List[str]] = None,
    ):
        self.h5_path = Path(h5_path)
        self.instruments_file = Path(instruments_file)
        self._tickers = tickers  # override; otherwise loaded from instruments_file

    # ------------------------------------------------------------------
    def _load_tickers(self) -> List[str]:
        if self._tickers:
            return self._tickers
        if self.instruments_file.exists():
            lines = self.instruments_file.read_text().strip().splitlines()
            return [t.strip() for t in lines if t.strip()]
        raise FileNotFoundError(f"Instruments file not found: {self.instruments_file}")

    def _last_stored_date(self) -> Optional[date]:
        if not self.h5_path.exists():
            return None
        df = pd.read_hdf(str(self.h5_path), key="data")
        if df.empty:
            return None
        last_dt = df.index.get_level_values("datetime").max()
        return pd.Timestamp(last_dt).date()

    def _download(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        """Download OHLCV from yfinance in batches; return MultiIndex DataFrame."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed — pip install yfinance")

        batch_size = 50
        rows = []
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                raw = yf.download(
                    batch, start=start, end=end,
                    auto_adjust=True, progress=False, threads=True,
                )
            except Exception as e:
                log.warning(f"Batch {i//batch_size + 1} failed: {e}")
                continue
            if raw.empty:
                continue
            # yfinance returns MultiIndex columns (field, ticker) when >1 ticker
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns.names = ["field", "symbol"]
                for ticker in batch:
                    if ticker not in raw.columns.get_level_values("symbol"):
                        continue
                    t_df = raw.xs(ticker, axis=1, level="symbol").copy()
                    t_df.columns = [c.lower() for c in t_df.columns]
                    if "close" not in t_df.columns:
                        continue
                    t_df = t_df.dropna(subset=["close"])
                    t_df.index = pd.to_datetime(t_df.index).normalize()
                    t_df.index.name = "datetime"
                    t_df["instrument"] = ticker
                    t_df = t_df.reset_index().set_index(["datetime", "instrument"])
                    rows.append(t_df)
            else:
                raw.columns = [c.lower() for c in raw.columns]
                if "close" not in raw.columns:
                    continue
                ticker = batch[0]
                raw = raw.dropna(subset=["close"])
                raw.index = pd.to_datetime(raw.index).normalize()
                raw.index.name = "datetime"
                raw["instrument"] = ticker
                raw = raw.reset_index().set_index(["datetime", "instrument"])
                rows.append(raw)

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows).sort_index()

    def _merge(self, existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
        if existing.empty:
            return new
        if new.empty:
            return existing
        combined = pd.concat([existing, new])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    # ------------------------------------------------------------------
    def ingest(self, force_full: bool = False) -> int:
        """Fetch missing trading days and append to H5. Returns new row count."""
        tickers = self._load_tickers()
        last_date = None if force_full else self._last_stored_date()

        today = date.today()
        if last_date is not None and last_date >= today:
            log.info("Data already up to date (last_date=%s). Nothing to do.", last_date)
            return 0

        start = (
            "2016-01-01" if last_date is None
            else (last_date - timedelta(days=5)).strftime("%Y-%m-%d")  # small overlap for safety
        )
        end = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        log.info("Downloading %d tickers from %s to %s...", len(tickers), start, end)
        raw = self._download(tickers, start=start, end=end)
        if raw.empty:
            log.warning("Download returned no data.")
            return 0

        new_featured = _engineer_features(raw)
        log.info("Downloaded %d rows for %d instruments.", len(new_featured), new_featured.index.get_level_values("instrument").nunique())

        existing = pd.DataFrame()
        if self.h5_path.exists():
            existing = pd.read_hdf(str(self.h5_path), key="data")

        merged = self._merge(existing, new_featured)
        new_rows = len(merged) - len(existing)

        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_hdf(str(self.h5_path), key="data", mode="w", complevel=5, complib="blosc")
        log.info("H5 updated: %d total rows (+%d new). Last date: %s",
                 len(merged), new_rows,
                 merged.index.get_level_values("datetime").max())
        return max(0, new_rows)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest EOD US data into daily_pv.h5")
    parser.add_argument("--h5", default=str(_DEFAULT_H5), help="Path to daily_pv.h5")
    parser.add_argument("--tickers-file", default=str(_DEFAULT_INSTRUMENTS),
                        help="Newline-separated ticker list")
    parser.add_argument("--force-full", action="store_true",
                        help="Re-download full history (ignore last stored date)")
    args = parser.parse_args()

    ingestor = DataIngestor(h5_path=args.h5, instruments_file=args.tickers_file)
    new_rows = ingestor.ingest(force_full=args.force_full)
    print(f"Ingestion complete: {new_rows} new rows added.")


if __name__ == "__main__":
    main()
