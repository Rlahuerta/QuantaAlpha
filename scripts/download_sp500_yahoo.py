#!/usr/bin/env python3
"""Download S&P 500 constituents and daily OHLCV from Yahoo Finance."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"


@dataclass
class DownloadStats:
    ok: int = 0
    failed: int = 0


def _to_unix(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _normalize_symbol(symbol: str) -> str:
    # Yahoo Finance uses "-" instead of "." in tickers (e.g., BRK.B -> BRK-B).
    return symbol.strip().replace(".", "-")


def _load_symbols_from_instruments_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    symbols: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        symbols.append(_normalize_symbol(parts[0]))
    return sorted(set(symbols))


def fetch_sp500_constituents(symbols_file: str | None = None) -> list[str]:
    if symbols_file:
        from_file = _load_symbols_from_instruments_file(Path(symbols_file))
        if from_file:
            return from_file

    default_instruments = Path("data/qlib/us_data/instruments/sp500.txt")
    from_default = _load_symbols_from_instruments_file(default_instruments)
    if from_default:
        return from_default

    try:
        tables = pd.read_html(WIKI_SP500_URL)
    except ImportError as e:
        raise RuntimeError(
            "Could not load symbols from local instruments and pandas.read_html requires lxml. "
            "Either install lxml or pass --symbols-file data/qlib/us_data/instruments/sp500.txt"
        ) from e
    if not tables:
        raise RuntimeError("Failed to parse S&P 500 constituents from Wikipedia")
    df = tables[0]
    if "Symbol" not in df.columns:
        raise RuntimeError("Unexpected Wikipedia table format: missing 'Symbol' column")
    symbols = sorted({_normalize_symbol(s) for s in df["Symbol"].dropna().astype(str).tolist()})
    return symbols


def fetch_yahoo_chart(symbol: str, start_unix: int, end_unix: int) -> pd.DataFrame:
    params = urllib.parse.urlencode(
        {
            "period1": start_unix,
            "period2": end_unix,
            "interval": "1d",
            "events": "div,splits",
            "includeAdjustedClose": "true",
        }
    )
    url = f"{YAHOO_CHART_URL.format(symbol=urllib.parse.quote(symbol))}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    result = payload.get("chart", {}).get("result")
    if not result:
        raise RuntimeError(f"No data returned for symbol={symbol}")
    result0 = result[0]
    timestamps = result0.get("timestamp") or []
    quote = (((result0.get("indicators") or {}).get("quote") or [{}])[0]) or {}
    adj = (((result0.get("indicators") or {}).get("adjclose") or [{}])[0]) or {}
    if not timestamps:
        raise RuntimeError(f"Empty timestamp data for symbol={symbol}")

    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(timestamps, unit="s", utc=True).tz_localize(None),
            "open": quote.get("open", []),
            "high": quote.get("high", []),
            "low": quote.get("low", []),
            "close": quote.get("close", []),
            "volume": quote.get("volume", []),
            "adjclose": adj.get("adjclose", []),
        }
    )
    df = df.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates("datetime")
    # Yahoo does not provide vwap directly for daily candles; use a simple OHLC proxy.
    df["vwap_proxy"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0
    return df


def download_symbol(
    symbol: str,
    out_dir: Path,
    start_unix: int,
    end_unix: int,
    retries: int,
    sleep_sec: float,
) -> tuple[bool, str | None]:
    last_err: str | None = None
    for i in range(retries):
        try:
            df = fetch_yahoo_chart(symbol, start_unix, end_unix)
            if df.empty:
                raise RuntimeError("empty DataFrame")
            out_path = out_dir / f"{symbol}.csv"
            df.to_csv(out_path, index=False)
            return True, None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as e:
            last_err = f"{type(e).__name__}: {e}"
            if i + 1 < retries:
                time.sleep(sleep_sec)
    return False, last_err


def build_instrument_file(raw_dir: Path, instrument_path: Path) -> None:
    rows: list[str] = []
    for csv_file in sorted(raw_dir.glob("*.csv")):
        if csv_file.stem.startswith("^"):
            continue
        df = pd.read_csv(csv_file, usecols=["datetime"])
        if df.empty:
            continue
        start = str(pd.to_datetime(df["datetime"]).min().date())
        end = str(pd.to_datetime(df["datetime"]).max().date())
        rows.append(f"{csv_file.stem}\t{start}\t{end}")
    instrument_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SP500 data from Yahoo Finance")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-26")
    parser.add_argument("--output-dir", default="data/yahoo_sp500")
    parser.add_argument(
        "--symbols-file",
        default=os.environ.get("SP500_SYMBOLS_FILE"),
        help="Optional tab-separated instruments file (symbol\\tstart\\tend)",
    )
    parser.add_argument("--limit", type=int, default=0, help="download only first N symbols (0 = all)")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    symbols = fetch_sp500_constituents(args.symbols_file)
    if args.limit > 0:
        symbols = symbols[: args.limit]

    (out_root / "sp500_symbols.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")

    start_unix = _to_unix(args.start)
    end_unix = _to_unix(args.end)
    stats = DownloadStats()
    failures: list[str] = []

    for symbol in symbols:
        ok, err = download_symbol(
            symbol=symbol,
            out_dir=raw_dir,
            start_unix=start_unix,
            end_unix=end_unix,
            retries=args.retries,
            sleep_sec=args.sleep_sec,
        )
        if ok:
            stats.ok += 1
        else:
            stats.failed += 1
            failures.append(f"{symbol}: {err}")

    # Save benchmark for backtest config checks.
    download_symbol("^GSPC", raw_dir, start_unix, end_unix, args.retries, args.sleep_sec)

    build_instrument_file(raw_dir=raw_dir, instrument_path=out_root / "instruments_sp500.txt")
    if failures:
        (out_root / "failures.txt").write_text("\n".join(failures) + "\n", encoding="utf-8")

    print(
        f"Downloaded symbols: ok={stats.ok}, failed={stats.failed}, "
        f"output={out_root.resolve()}"
    )


if __name__ == "__main__":
    main()
