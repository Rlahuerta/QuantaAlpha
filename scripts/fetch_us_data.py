#!/usr/bin/env python3
"""
Phase 0 — US Data Pipeline
Fetches S&P 500 + NASDAQ-100 daily OHLCV from yfinance (2016-present) and
stores it as an HDF5 file matching the layout of daily_pv.h5 used by
QuantaAlpha's factor computation engine.

Output: git_ignore_folder/factor_implementation_source_data_us/daily_pv.h5
Format: pd.DataFrame with MultiIndex(datetime, instrument), float32 columns:
        $open, $close, $high, $low, $volume, $vwap, $return

Usage:
    python scripts/fetch_us_data.py [--start 2016-01-01] [--end 2025-12-31]
                                    [--output git_ignore_folder/factor_implementation_source_data_us]
                                    [--incremental]
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe helpers
# ---------------------------------------------------------------------------

_WIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _fetch_html(url: str) -> str:
    """Fetch URL HTML with browser-like headers to avoid 403."""
    import requests as _req
    resp = _req.get(url, headers=_WIKI_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def get_sp500_tickers() -> List[str]:
    """Scrape current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        from io import StringIO
        html = _fetch_html(url)
        tables = pd.read_html(StringIO(html), header=0)
        tickers = tables[0]["Symbol"].tolist()
        tickers = [t.replace(".", "-") for t in tickers]
        log.info(f"S&P 500: {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        log.warning(f"Wikipedia scrape failed ({e}), using fallback list")
        return _sp500_fallback()


def get_nasdaq100_tickers() -> List[str]:
    """Scrape current NASDAQ-100 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        from io import StringIO
        html = _fetch_html(url)
        tables = pd.read_html(StringIO(html), header=0)
        for tbl in tables:
            for col in ["Ticker", "Symbol", "Ticker symbol"]:
                if col in tbl.columns:
                    tickers = tbl[col].tolist()
                    tickers = [str(t).replace(".", "-") for t in tickers if str(t) not in ("nan", "")]
                    if len(tickers) > 50:
                        log.info(f"NASDAQ-100: {len(tickers)} tickers from Wikipedia")
                        return tickers
        raise ValueError("Could not find ticker column in NASDAQ-100 table")
    except Exception as e:
        log.warning(f"Wikipedia scrape failed ({e}), using fallback list")
        return _nasdaq100_fallback()


def get_universe() -> List[str]:
    """Return deduplicated S&P 500 + NASDAQ-100 ticker list."""
    sp500 = set(get_sp500_tickers())
    ndq100 = set(get_nasdaq100_tickers())
    combined = sorted(sp500 | ndq100)
    log.info(f"Combined universe: {len(combined)} unique tickers (SP500={len(sp500)}, NDQ100={len(ndq100)}, overlap={len(sp500 & ndq100)})")
    return combined


# ---------------------------------------------------------------------------
# Fallback static ticker lists (used if Wikipedia is unavailable)
# ---------------------------------------------------------------------------

def _sp500_fallback() -> List[str]:
    # S&P 500 constituents as of Feb 2026; deduplicated via dict.fromkeys
    return list(dict.fromkeys([
        "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB",
        "AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN",
        "AMCR","AEE","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI",
        "ANSS","AON","APA","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ",
        "T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BKNG",
        "BAX","BBY","BIO","TECH","BK","BA","BSX","BRK-B","BMY","AVGO","BR","BRO",
        "BF-B","BLDR","BG","CDNS","CPT","CPB","COF","CAH","KMX","CCL","CARR","CAT",
        "CBOE","CBRE","CDW","CE","COR","CNC","CDAY","CF","CRL","SCHW","CHTR","CVX",
        "CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS",
        "KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ","CEG","CPRT","GLW","CTVA",
        "CSGP","COST","CTRA","CRWD","CCI","CSX","CMI","CVS","DHI","DHR","DRI",
        "DVA","DECK","DE","DAL","DVN","DXCM","FANG","DLR","DG","DLTR","D","DPZ",
        "DOV","DOW","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA",
        "ELV","LLY","EMR","ENPH","ETR","EOG","EPAM","EFX","EQIX","EQT","EL",
        "ETSY","EG","EVRG","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS",
        "FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FLT","FMC","F",
        "FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GEHC","GEV","GEN",
        "GIS","GPC","GILD","GPN","GL","GDDY","GS","HAL","HIG","HAS","HCA","DOC",
        "HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ",
        "HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD",
        "INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM",
        "JBHT","JBL","JKHY","J","JNJ","JCI","JPM","K","KVUE","KR","KMB","KIM",
        "KMI","KLAC","KHC","KDP","KEY","KEYS","KKR","LHX","LH","LRCX","LW","LVS",
        "LDOS","LEN","LIN","LYV","LMT","L","LOW","LYB","MTB","MPC","MAR","MMC",
        "MLM","MAS","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA",
        "MHK","TAP","MDLZ","MPWR","MOH","MOS","MSI","MSCI","MKC","MCD","MCK",
        "MDT","MRK","NKE","NI","NDSN","NEE","NEM","NFLX","NWS","NWSA","NOC",
        "NCLH","NRG","NSC","NTRS","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL",
        "OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PANW","PH","PAYX","PAYC",
        "PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL",
        "PFG","PG","PGR","PLD","PRU","PEG","PTC","PSA","PHM","PWR","QCOM","DGX",
        "RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL",
        "ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SEE","SRE","NOW",
        "SHW","SPG","SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT",
        "STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP",
        "TGT","TEL","TDY","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT",
        "TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP",
        "UAL","UPS","URI","UNH","UHS","VLO","VTR","VRSN","VRSK","VZ","VRTX",
        "VTRS","VLTO","V","VST","VICI","VMC","WRB","GWW","WAB","WMT","DIS","WBD",
        "WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WHR","WMB","WTW","WDAY",
        "WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS","CIEN","CRH","CVNA","FIX",
        "ARES","SNDK","EME","HOOD","APP","IBKR","TTD","DDOG","COIN","DASH","TKO",
        "WSM","EXE","APO","LII","TPL","BBWI","PLTR","DELL","ERIE",
        "BX","LULU","STLD","GEV","CEG","GEHC",
    ]))


def _nasdaq100_fallback() -> List[str]:
    # NASDAQ-100 constituents as of January 20, 2026
    return [
        "ADBE","AMD","ABNB","ALNY","GOOGL","GOOG","AMZN","AEP","AMGN","ADI",
        "AAPL","AMAT","APP","ARM","ASML","TEAM","ADSK","ADP","AXON","BKR",
        "BKNG","AVGO","CDNS","CHTR","CTAS","CSCO","CCEP","CTSH","CMCSA","CEG",
        "CPRT","CSGP","COST","CRWD","CSX","DDOG","DXCM","FANG","DASH","EA",
        "EXC","FAST","FER","FTNT","GEHC","GILD","HON","IDXX","INSM","INTC",
        "INTU","ISRG","KDP","KLAC","KHC","LRCX","LIN","MAR","MRVL","MELI",
        "META","MCHP","MU","MSFT","MSTR","MDLZ","MPWR","MNST","NFLX","NVDA",
        "NXPI","ORLY","ODFL","PCAR","PLTR","PANW","PAYX","PYPL","PDD","PEP",
        "QCOM","REGN","ROP","ROST","STX","SHOP","SBUX","SNPS","TMUS","TTWO",
        "TSLA","TXN","TRI","VRSK","VRTX","WMT","WBD","WDC","WDAY","XEL","ZS",
    ]


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_tickers(
    tickers: List[str],
    start: str,
    end: str,
    batch_size: int = 50,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """
    Download daily OHLCV for all tickers from yfinance in batches.
    Returns a DataFrame with MultiIndex(datetime, instrument).
    """
    all_frames: List[pd.DataFrame] = []
    failed: List[str] = []
    total_batches = (len(tickers) + batch_size - 1) // batch_size

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        log.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} tickers [{batch[0]}…{batch[-1]}]")

        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(
                    tickers=batch,
                    start=start,
                    end=end,
                    interval="1d",
                    auto_adjust=True,   # prices adjusted for splits/dividends
                    progress=False,
                    threads=True,
                )
                break
            except Exception as e:
                log.warning(f"    Attempt {attempt} failed: {e}")
                if attempt == max_retries:
                    log.error(f"    Batch {batch_num} failed after {max_retries} retries, skipping")
                    failed.extend(batch)
                    raw = None
                    break
                time.sleep(retry_delay)

        if raw is None or raw.empty:
            continue

        # yfinance multi-ticker returns columns: (field, ticker)
        if isinstance(raw.columns, pd.MultiIndex):
            frames = []
            for ticker in batch:
                try:
                    t_data = raw.xs(ticker, axis=1, level=1).copy()
                    t_data.columns = [c.lower() for c in t_data.columns]
                    t_data["instrument"] = ticker
                    t_data.index.name = "datetime"
                    frames.append(t_data.reset_index().set_index(["datetime", "instrument"]))
                except KeyError:
                    failed.append(ticker)
        else:
            # Single ticker
            t_data = raw.copy()
            t_data.columns = [c.lower() for c in t_data.columns]
            ticker = batch[0]
            t_data["instrument"] = ticker
            t_data.index.name = "datetime"
            frames = [t_data.reset_index().set_index(["datetime", "instrument"])]

        if frames:
            all_frames.extend(frames)

        time.sleep(0.5)  # be polite

    if not all_frames:
        raise RuntimeError("No data downloaded — check network and ticker list")

    df = pd.concat(all_frames)
    df = df.sort_index()

    if failed:
        log.warning(f"Failed tickers ({len(failed)}): {failed}")

    log.info(f"Raw download: {len(df)} rows, {df.index.get_level_values('instrument').nunique()} instruments")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename yfinance columns to QuantaAlpha DSL names and add derived fields.

    Input columns (lowercase): open, high, low, close, volume
    Output columns: $open, $high, $low, $close, $volume, $vwap, $return
    """
    rename = {
        "open":   "$open",
        "high":   "$high",
        "low":    "$low",
        "close":  "$close",
        "volume": "$volume",
    }
    # Drop any unexpected columns (e.g. 'dividends', 'stock splits')
    df = df[[c for c in rename if c in df.columns]].rename(columns=rename)

    # Daily VWAP approximation: (H+L+C)/3
    df["$vwap"] = (df["$high"] + df["$low"] + df["$close"]) / 3.0

    # Daily return: pct_change per instrument
    df = df.sort_index()
    df["$return"] = (
        df.groupby(level="instrument")["$close"]
        .transform(lambda x: x.pct_change(fill_method=None))
    )
    df["$return"] = df["$return"].fillna(0.0)

    # Drop rows where close is NaN (delisted / not yet listed)
    df = df.dropna(subset=["$close"])

    # Cast to float32 to match CN H5 dtype
    for col in df.columns:
        df[col] = df[col].astype("float32")

    log.info(f"After feature engineering: {len(df)} rows, {df.index.get_level_values('instrument').nunique()} instruments")
    return df


# ---------------------------------------------------------------------------
# Incremental update
# ---------------------------------------------------------------------------

def load_existing(h5_path: Path) -> pd.DataFrame:
    """Load existing H5 if present."""
    if h5_path.exists():
        log.info(f"Loading existing data from {h5_path}")
        df = pd.read_hdf(str(h5_path), key="data")
        log.info(f"  Existing rows: {len(df)}")
        return df
    return pd.DataFrame()


def merge_incremental(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Merge new data into existing, deduplicating by index."""
    if existing.empty:
        return new
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def quality_check(df: pd.DataFrame) -> None:
    """Log basic data quality stats."""
    instruments = df.index.get_level_values("instrument").nunique()
    dates = df.index.get_level_values("datetime").nunique()
    nan_pct = df.isna().mean().mean() * 100

    # IC sanity: $return should have reasonable daily std
    ret_std = df["$return"].std()

    log.info("=== Data Quality Report ===")
    log.info(f"  Instruments : {instruments}")
    log.info(f"  Trading days: {dates}")
    log.info(f"  Total rows  : {len(df):,}")
    log.info(f"  NaN rate    : {nan_pct:.2f}%")
    log.info(f"  $return std : {ret_std:.4f}  (expected 0.005–0.03 for US daily)")
    log.info(f"  Date range  : {df.index.get_level_values('datetime').min()} → {df.index.get_level_values('datetime').max()}")
    log.info(f"  Columns     : {list(df.columns)}")

    if nan_pct > 5:
        log.warning("NaN rate > 5% — check for delisted or unavailable tickers")
    if not (0.005 < ret_std < 0.05):
        log.warning(f"Unexpected $return std ({ret_std:.4f}) — check data correctness")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch US OHLCV data for QuantaAlpha")
    parser.add_argument("--start", default="2016-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default=None,         help="End date (default: today)")
    parser.add_argument(
        "--output",
        default="git_ignore_folder/factor_implementation_source_data_us",
        help="Output directory for daily_pv.h5",
    )
    parser.add_argument("--incremental", action="store_true",
                        help="Only download data newer than what exists in H5")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Tickers per yfinance batch request (default 50)")
    args = parser.parse_args()

    if args.end is None:
        args.end = pd.Timestamp.today().strftime("%Y-%m-%d")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "daily_pv.h5"

    log.info(f"=== QuantaAlpha US Data Fetch ===")
    log.info(f"Period  : {args.start} → {args.end}")
    log.info(f"Output  : {h5_path}")

    # 1. Universe
    tickers = get_universe()

    # 2. Incremental: advance start date if data already exists
    start = args.start
    if args.incremental and h5_path.exists():
        existing = load_existing(h5_path)
        if not existing.empty:
            last_date = existing.index.get_level_values("datetime").max()
            # Start from the day after last available date
            start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            log.info(f"Incremental mode: fetching from {start}")
            if start >= args.end:
                log.info("Data already up to date, nothing to download")
                quality_check(existing)
                return
    else:
        existing = pd.DataFrame()

    # 3. Download
    log.info(f"Downloading {len(tickers)} tickers in batches of {args.batch_size}...")
    raw = download_tickers(tickers, start=start, end=args.end, batch_size=args.batch_size)

    # 4. Feature engineering
    featured = engineer_features(raw)

    # 5. Merge with existing if incremental
    if not existing.empty:
        featured = merge_incremental(existing, featured)

    # 6. Quality check
    quality_check(featured)

    # 7. Save as HDF5
    log.info(f"Saving to {h5_path} ...")
    featured.to_hdf(str(h5_path), key="data", mode="w", complevel=5, complib="blosc")
    size_mb = h5_path.stat().st_size / 1024 / 1024
    log.info(f"Saved {len(featured):,} rows → {h5_path} ({size_mb:.1f} MB)")

    # 8. Save instruments list (for Qlib and live trading)
    instruments = sorted(featured.index.get_level_values("instrument").unique().tolist())
    inst_path = out_dir / "instruments.txt"
    inst_path.write_text("\n".join(instruments))
    log.info(f"Instruments list: {len(instruments)} tickers → {inst_path}")

    log.info("=== Phase 0 complete ===")


if __name__ == "__main__":
    main()
