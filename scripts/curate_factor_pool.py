#!/usr/bin/env python3
"""Factor pool curation per paper methodology (arXiv:2602.07085, Section 5.3).

Implements:
1. Compute RankIC for each factor on validation set
2. Sort by RankIC descending
3. Greedy admission: add factor only if |corr| < threshold with all admitted factors
4. Cap pool at max_ratio × total_mined

Memory-efficient: uses float16, loads only validation slice, streams factors.

Usage:
    python scripts/curate_factor_pool.py \
        --input data/factorlib/all_factors_library_expanded_decay.json \
        --output data/factorlib/all_factors_library_curated.json \
        --corr-threshold 0.7 \
        --max-ratio 0.5

    # With custom validation period
    python scripts/curate_factor_pool.py \
        --input data/factorlib/all_factors_library_expanded_decay.json \
        --output data/factorlib/all_factors_library_curated.json \
        --valid-start 2021-01-01 --valid-end 2021-12-31
"""

import argparse
import ctypes
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Reuse project-wide precision setting
FACTOR_DTYPE = np.float16

# Force glibc to release freed pages back to OS
try:
    _LIBC = ctypes.CDLL("libc.so.6")
    def _release_memory():
        gc.collect()
        _LIBC.malloc_trim(0)
except Exception:
    def _release_memory():
        gc.collect()


def _rss_mb() -> float:
    """Current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def load_factor_validation_slice(
    expr: str, cache_dir: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
) -> pd.Series | None:
    """Load only the validation slice of a factor, cast to float16.

    Immediately frees the full-size pickle to keep RSS low.
    """
    md5 = hashlib.md5(expr.encode()).hexdigest()
    pkl_path = os.path.join(cache_dir, f"{md5}.pkl")
    if not os.path.exists(pkl_path):
        return None
    try:
        raw = pd.read_pickle(pkl_path)
        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[:, 0]
        # Ensure (datetime, instrument) index
        if raw.index.names == ["instrument", "datetime"]:
            raw = raw.swaplevel().sort_index()

        # Slice to validation period only — massive memory savings
        try:
            sliced = raw.loc[start_ts:end_ts].copy()
        except KeyError:
            mask = (
                (raw.index.get_level_values("datetime") >= start_ts)
                & (raw.index.get_level_values("datetime") <= end_ts)
            )
            sliced = raw[mask].copy()

        # Free full-size pickle ASAP (114 MB → ~2 MB slice)
        del raw
        _release_memory()

        return sliced.astype(FACTOR_DTYPE)
    except Exception:
        return None


def compute_label(qlib_data_dir: str, start: str, end: str, market: str = "cn",
                  prices_parquet: str | None = None) -> pd.Series:
    """Compute next-day return label: Ref($close, -2) / Ref($close, -1) - 1.

    For US market, uses parquet prices instead of Qlib D.features().
    """
    if market == "us":
        return _compute_label_us(prices_parquet or "data/sp500_parquet_bundle/sp500_prices.parquet",
                                 start, end)
    import qlib
    from qlib.data import D

    qlib.init(provider_uri=qlib_data_dir, region="cn")
    instruments = D.instruments("csi300")
    fields = ["Ref($close, -2) / Ref($close, -1) - 1"]
    df = D.features(instruments, fields, start_time=start, end_time=end)
    label = df.iloc[:, 0].astype(np.float32)
    label.name = "label"
    # Qlib returns (instrument, datetime) — normalize to (datetime, instrument)
    if label.index.names == ["instrument", "datetime"]:
        label = label.swaplevel().sort_index()
    return label


def _compute_label_us(prices_parquet: str, start: str, end: str) -> pd.Series:
    """Compute 1-day forward return label from US parquet prices."""
    df = pd.read_parquet(prices_parquet, columns=["datetime", "symbol", "close"])
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()  # strip time component
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    df = df.loc[mask].sort_values(["symbol", "datetime"])
    # Ref($close, -2) / Ref($close, -1) - 1
    close_grp = df.groupby("symbol")["close"]
    label = close_grp.shift(-2) / close_grp.shift(-1) - 1
    idx = pd.MultiIndex.from_arrays(
        [df["datetime"].values, df["symbol"].values],
        names=["datetime", "instrument"],
    )
    out = pd.Series(label.values, index=idx, dtype=np.float32, name="label")
    return out.dropna().sort_index()


def compute_rankic_per_factor(
    factor_series: pd.Series,
    label: pd.Series,
) -> float:
    """Compute mean daily RankIC (Spearman correlation). Both series pre-sliced.

    Uses numpy arrays directly to minimize pandas overhead and memory copies.
    """
    common_idx = factor_series.index.intersection(label.index)
    if len(common_idx) < 100:
        return 0.0

    # Promote to float32 for Spearman — float16 lacks precision for ranking
    f_aligned = factor_series.loc[common_idx].astype(np.float32)
    l_aligned = label.loc[common_idx]

    mask = f_aligned.notna() & l_aligned.notna()
    f_aligned = f_aligned[mask]
    l_aligned = l_aligned[mask]

    if len(f_aligned) < 100:
        return 0.0

    # Group by date using numpy for speed
    dt_idx = f_aligned.index.get_level_values("datetime")
    dates = dt_idx.unique()
    f_vals = f_aligned.values
    l_vals = l_aligned.values
    dt_codes = dt_idx.codes if hasattr(dt_idx, 'codes') else pd.Categorical(dt_idx).codes

    daily_ics = []
    for d_code, dt in enumerate(dates):
        day_mask = dt_codes == d_code
        fv = f_vals[day_mask]
        lv = l_vals[day_mask]

        if len(fv) < 10:
            continue

        # Check for constant arrays (skip to avoid warnings)
        if fv.std() < 1e-10 or lv.std() < 1e-10:
            continue

        ic, _ = spearmanr(fv, lv)
        if not np.isnan(ic):
            daily_ics.append(ic)

    return float(np.mean(daily_ics)) if daily_ics else 0.0


def compute_pairwise_corr(
    factor_a: pd.Series,
    factor_b: pd.Series,
) -> float:
    """Compute correlation between two factor series (both pre-sliced, float16)."""
    common_idx = factor_a.index.intersection(factor_b.index)
    if len(common_idx) < 100:
        return 0.0

    a = factor_a.loc[common_idx].astype(np.float32)
    b = factor_b.loc[common_idx].astype(np.float32)
    mask = a.notna() & b.notna()
    a = a[mask]
    b = b[mask]

    if len(a) < 100:
        return 0.0

    return float(np.abs(a.corr(b)))


def _build_factor_matrix(
    factors: dict,
    cache_dir: str,
    label: pd.Series,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build aligned factor matrix (float32) + label vector on common index.

    Returns:
        factor_matrix: shape (n_common, n_factors), float32, NaN-filled
        label_vec: shape (n_common,), float32
        names: list of factor names (matching columns)
    """
    print("  Building aligned factor matrix...")
    # Use label's index as reference
    ref_idx = label.index

    factor_cols = {}  # name -> np.array aligned to ref_idx
    names_order = []

    for i, (name, v) in enumerate(factors.items()):
        expr = v.get("factor_expression", "")
        if not expr:
            continue

        s = load_factor_validation_slice(expr, cache_dir, start_ts, end_ts)
        if s is None:
            continue

        # Reindex to common reference (fills missing with NaN)
        aligned = s.reindex(ref_idx).astype(np.float32).values
        factor_cols[name] = aligned
        names_order.append(name)
        del s

        if (i + 1) % 100 == 0:
            _release_memory()
            print(f"    Loaded {len(names_order)}/{len(factors)} factors (RSS: {_rss_mb():.0f} MB)")

    _release_memory()
    n_factors = len(names_order)
    n_rows = len(ref_idx)
    print(f"  Matrix: {n_rows} rows × {n_factors} factors")

    # Stack into matrix
    mat = np.column_stack([factor_cols[n] for n in names_order])  # (n_rows, n_factors)
    label_vec = label.values.astype(np.float32)

    del factor_cols
    _release_memory()
    print(f"  Matrix memory: {mat.nbytes / 1e6:.0f} MB (RSS: {_rss_mb():.0f} MB)")

    return mat, label_vec, names_order


def compute_rankic_vectorized(
    mat: np.ndarray,
    label_vec: np.ndarray,
    date_codes: np.ndarray,
    n_dates: int,
) -> np.ndarray:
    """Compute mean daily RankIC for all factors at once.

    Args:
        mat: (n_rows, n_factors) float32
        label_vec: (n_rows,) float32
        date_codes: (n_rows,) int, maps each row to a date index
        n_dates: number of unique dates

    Returns:
        rankic: (n_factors,) mean daily RankIC
    """
    n_factors = mat.shape[1]
    ic_sums = np.zeros(n_factors, dtype=np.float64)
    ic_counts = np.zeros(n_factors, dtype=np.int32)

    for d in range(n_dates):
        day_mask = date_codes == d
        n_stocks = day_mask.sum()
        if n_stocks < 10:
            continue

        lv = label_vec[day_mask]
        if np.isnan(lv).all() or np.nanstd(lv) < 1e-10:
            continue

        for j in range(n_factors):
            fv = mat[day_mask, j]
            valid = ~(np.isnan(fv) | np.isnan(lv))
            if valid.sum() < 10:
                continue
            fv_v = fv[valid]
            lv_v = lv[valid]
            if np.std(fv_v) < 1e-10:
                continue
            ic, _ = spearmanr(fv_v, lv_v)
            if not np.isnan(ic):
                ic_sums[j] += ic
                ic_counts[j] += 1

        if (d + 1) % 50 == 0:
            print(f"    RankIC: {d+1}/{n_dates} dates processed...")

    with np.errstate(invalid='ignore'):
        result = np.where(ic_counts > 0, ic_sums / ic_counts, 0.0)
    return result


def greedy_decorrelation_numpy(
    mat: np.ndarray,
    rankic: np.ndarray,
    names: list[str],
    corr_threshold: float,
    max_pool: int,
) -> list[int]:
    """Greedy admission using numpy correlation — O(admitted × candidates) but fast.

    Returns indices of admitted factors.
    """
    # Sort by |RankIC| descending
    order = np.argsort(-np.abs(rankic))

    admitted_idx = []
    # Pre-compute per-column stats for fast correlation
    # corr(a, b) = cov(a,b) / (std_a * std_b)
    # We'll compute correlation on-the-fly using numpy columns

    for rank, fi in enumerate(order):
        if len(admitted_idx) >= max_pool:
            break
        if abs(rankic[fi]) < 1e-6:
            continue

        col = mat[:, fi]

        # Check correlation with all admitted
        too_correlated = False
        for ai in admitted_idx:
            acol = mat[:, ai]
            # Fast correlation: handle NaNs
            valid = ~(np.isnan(col) | np.isnan(acol))
            if valid.sum() < 100:
                continue
            c = np.corrcoef(col[valid], acol[valid])[0, 1]
            if abs(c) > corr_threshold:
                too_correlated = True
                break

        if not too_correlated:
            admitted_idx.append(fi)
            if len(admitted_idx) % 20 == 0:
                print(f"    Admitted {len(admitted_idx)} factors (checked {rank+1}/{len(order)})...")

    return admitted_idx


def curate_pool(
    input_path: str,
    output_path: str,
    cache_dir: str = "data/results/factor_cache",
    qlib_data_dir: str = "~/.qlib/qlib_data/cn_data",
    valid_start: str = "2021-01-01",
    valid_end: str = "2021-12-31",
    corr_threshold: float = 0.7,
    max_ratio: float = 0.5,
    market: str = "cn",
    prices_parquet: str | None = None,
):
    """Main curation pipeline. Matrix-based for speed, float32 for accuracy."""
    print(f"Loading factor library: {input_path}")
    print(f"  Market: {market}, RSS at start: {_rss_mb():.0f} MB")
    with open(input_path) as f:
        lib = json.load(f)

    factors = lib["factors"]
    total = len(factors)
    max_pool = int(total * max_ratio)
    print(f"Total factors: {total}, max pool size: {max_pool}")

    start_ts = pd.Timestamp(valid_start)
    end_ts = pd.Timestamp(valid_end)

    # Step 1: Compute label
    print(f"\n[1/4] Computing label ({valid_start} to {valid_end})...")
    label = compute_label(qlib_data_dir, valid_start, valid_end, market=market,
                          prices_parquet=prices_parquet)
    print(f"  Label: {len(label)} rows, {label.nbytes/1e6:.1f} MB")

    # Step 2: Build aligned matrix (one disk pass)
    print(f"\n[2/4] Loading factors into aligned matrix...")
    mat, label_vec, names = _build_factor_matrix(
        factors, cache_dir, label, start_ts, end_ts,
    )
    del label
    _release_memory()

    # Date codes for daily RankIC
    # Reconstruct date index from the label's original index
    label_reloaded = compute_label(qlib_data_dir, valid_start, valid_end,
                                   market=market, prices_parquet=prices_parquet)
    dt_level = label_reloaded.index.get_level_values("datetime")
    date_cats = pd.Categorical(dt_level)
    date_codes = date_cats.codes.astype(np.int32)
    n_dates = len(date_cats.categories)
    del label_reloaded
    _release_memory()
    print(f"  {n_dates} trading days in validation period")

    # Step 3: Compute RankIC for all factors (vectorized over dates)
    print(f"\n[3/4] Computing RankIC ({len(names)} factors × {n_dates} days)...")
    rankic = compute_rankic_vectorized(mat, label_vec, date_codes, n_dates)

    # Report top factors
    sorted_idx = np.argsort(-np.abs(rankic))
    print(f"\n  Top 10 by |RankIC|:")
    for i in sorted_idx[:10]:
        print(f"    {names[i]}: RankIC={rankic[i]:.4f}")
    positive = (rankic > 0).sum()
    print(f"  Positive RankIC: {positive}/{len(rankic)}")

    # Step 4: Greedy decorrelation (numpy — fast)
    print(f"\n[4/4] Greedy pool admission (|corr|<{corr_threshold}, cap={max_pool})...")
    admitted_idx = greedy_decorrelation_numpy(mat, rankic, names, corr_threshold, max_pool)

    admitted_names = [names[i] for i in admitted_idx]
    admitted_ics = [float(rankic[i]) for i in admitted_idx]

    del mat, label_vec
    _release_memory()
    print(f"  Final pool: {len(admitted_names)} factors (from {total} total)")

    # Build output library
    print(f"\nWriting curated library: {output_path}")
    curated_factors = {}
    for name, ic in zip(admitted_names, admitted_ics):
        entry = factors[name].copy()
        entry["rankic_validation"] = ic
        curated_factors[name] = entry

    output = {
        "metadata": {
            "version": "1.0",
            "total_factors": len(curated_factors),
            "source": os.path.basename(input_path),
            "curation": {
                "method": "rankic_greedy_decorrelation",
                "corr_threshold": corr_threshold,
                "max_ratio": max_ratio,
                "valid_period": f"{valid_start} to {valid_end}",
                "original_count": total,
            },
        },
        "factors": curated_factors,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Curated pool: {len(curated_factors)} factors")
    print(f"Reduction: {total} → {len(curated_factors)} ({100*len(curated_factors)/total:.0f}%)")
    if admitted_ics:
        print(f"RankIC range: {min(admitted_ics):.4f} to {max(admitted_ics):.4f} (mean={np.mean(admitted_ics):.4f})")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Curate factor pool per paper methodology")
    parser.add_argument("--input", required=True, help="Input factor library JSON")
    parser.add_argument("--output", required=True, help="Output curated library JSON")
    parser.add_argument("--cache-dir", default="data/results/factor_cache")
    parser.add_argument("--qlib-data-dir", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--valid-start", default="2021-01-01")
    parser.add_argument("--valid-end", default="2021-12-31")
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--max-ratio", type=float, default=0.5)
    parser.add_argument("--market", default="cn", choices=["cn", "us"],
                        help="Market: cn (CSI300) or us (S&P500)")
    parser.add_argument("--prices-parquet", default=None,
                        help="US prices parquet path (required for --market us)")
    args = parser.parse_args()

    curate_pool(
        input_path=args.input,
        output_path=args.output,
        cache_dir=args.cache_dir,
        qlib_data_dir=args.qlib_data_dir,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        corr_threshold=args.corr_threshold,
        max_ratio=args.max_ratio,
        market=args.market,
        prices_parquet=args.prices_parquet,
    )


if __name__ == "__main__":
    main()
