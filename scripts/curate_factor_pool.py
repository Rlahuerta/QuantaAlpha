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


def compute_label(qlib_data_dir: str, start: str, end: str) -> pd.Series:
    """Compute next-day return label: Ref($close, -2) / Ref($close, -1) - 1."""
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


def curate_pool(
    input_path: str,
    output_path: str,
    cache_dir: str = "data/results/factor_cache",
    qlib_data_dir: str = "~/.qlib/qlib_data/cn_data",
    valid_start: str = "2021-01-01",
    valid_end: str = "2021-12-31",
    corr_threshold: float = 0.7,
    max_ratio: float = 0.5,
):
    """Main curation pipeline. Memory-efficient: streams factors, uses float16."""
    print(f"Loading factor library: {input_path}")
    print(f"  RSS at start: {_rss_mb():.0f} MB")
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
    label = compute_label(qlib_data_dir, valid_start, valid_end)
    print(f"  Label: {len(label)} rows, {label.nbytes/1e6:.1f} MB")

    # Step 2: Stream factors — compute RankIC one at a time, keep only validation slice
    print(f"\n[2/4] Computing RankIC (streaming, {FACTOR_DTYPE.__name__})...")
    rankic_scores = {}
    expr_map = {}  # name -> expression (to reload later)
    loaded = 0

    for i, (name, v) in enumerate(factors.items()):
        expr = v.get("factor_expression", "")
        if not expr:
            continue

        s = load_factor_validation_slice(expr, cache_dir, start_ts, end_ts)
        if s is None:
            continue

        loaded += 1
        ic = compute_rankic_per_factor(s, label)
        rankic_scores[name] = ic
        expr_map[name] = expr
        del s  # free immediately
        _release_memory()

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{total}] RankIC computed for {loaded} factors... (RSS: {_rss_mb():.0f} MB)")

    _release_memory()
    print(f"  Loaded & scored: {loaded} / {total} (RSS: {_rss_mb():.0f} MB)")

    # Sort by absolute RankIC descending
    sorted_factors = sorted(
        rankic_scores.items(), key=lambda x: abs(x[1]), reverse=True
    )

    print(f"\n  Top 10 by |RankIC|:")
    for name, ic in sorted_factors[:10]:
        print(f"    {name}: RankIC={ic:.4f}")

    positive = sum(1 for _, ic in sorted_factors if ic > 0)
    print(f"  Positive RankIC: {positive}/{len(sorted_factors)}")

    # Step 3: Greedy admission with correlation filter
    # Only admitted factors stay in memory (as float16 validation slices)
    print(f"\n[3/4] Greedy pool admission (corr<{corr_threshold}, cap={max_pool})...")
    admitted = []
    admitted_series = []  # float16 validation slices only

    for name, ic in sorted_factors:
        if len(admitted) >= max_pool:
            break

        if abs(ic) < 1e-6:
            continue

        # Reload validation slice (float16)
        candidate = load_factor_validation_slice(
            expr_map[name], cache_dir, start_ts, end_ts
        )
        if candidate is None:
            continue

        # Check correlation with all admitted factors
        too_correlated = False
        for admitted_s in admitted_series:
            corr = compute_pairwise_corr(candidate, admitted_s)
            if corr > corr_threshold:
                too_correlated = True
                break

        if not too_correlated:
            admitted.append(name)
            admitted_series.append(candidate)
            if len(admitted) % 20 == 0:
                mem_mb = sum(s.nbytes for s in admitted_series) / 1e6
                print(f"  Admitted {len(admitted)} factors ({mem_mb:.0f} MB)...")
        else:
            del candidate

    del admitted_series
    _release_memory()
    print(f"  Final pool: {len(admitted)} factors (from {total} total)")

    # Step 4: Build output library
    print(f"\n[4/4] Writing curated library: {output_path}")
    curated_factors = {}
    for name in admitted:
        entry = factors[name].copy()
        entry["rankic_validation"] = rankic_scores[name]
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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Curated pool: {len(curated_factors)} factors")
    print(f"Reduction: {total} → {len(curated_factors)} ({100*len(curated_factors)/total:.0f}%)")
    ics = [rankic_scores[n] for n in admitted]
    print(f"RankIC range: {min(ics):.4f} to {max(ics):.4f} (mean={np.mean(ics):.4f})")
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
    )


if __name__ == "__main__":
    main()
