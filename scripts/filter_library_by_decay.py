#!/usr/bin/env python
"""
Standalone script to apply IC decay filter to an existing factor JSON library.

Usage:
    python scripts/filter_library_by_decay.py \\
        --input  data/factorlib/all_factors_library.json \\
        --output data/factorlib/all_factors_library_decay5d.json \\
        --horizon 5 \\
        --min-ic 0.0 \\
        --data-path git_ignore_folder/factor_implementation_source_data/daily_pv.h5 \\
        --report

Factors whose IC at --horizon is > --min-ic are kept; the rest are dropped.
Decay metrics (ic_1d, ic_5d, ic_10d, ic_20d, decay_half_life) are attached to each
factor's backtest_results before writing the output file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Make sure the project root is on PYTHONPATH when running standalone
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pandas as pd
import numpy as np

from quantaalpha.factors.decay_filter import DEFAULT_HORIZONS


def _load_factor_series_from_h5(result_h5_path: str) -> "pd.Series | None":
    """Load factor values from a workspace result.h5 file via h5py (fast path)."""
    try:
        import h5py
        import numpy as np

        with h5py.File(result_h5_path, "r") as fh:
            keys = list(fh.keys())
            top_key = keys[0]
            grp = fh[top_key]
            grp_keys = list(grp.keys())

            # Fast path: pandas Series fixed format (values + index_label0/1)
            if "values" in grp_keys and "index_label0" in grp_keys:
                raw = grp["values"][:]                       # float64, shape (N,)
                dates_ns = grp["index_level0"][:]            # int64 nanoseconds
                inst_bytes = grp["index_level1"][:]          # bytes, shape (n_insts,)
                label_dates = grp["index_label0"][:]         # int16, index into dates_ns
                label_insts = grp["index_label1"][:]         # int16, index into inst_bytes
                all_insts = np.array([b.decode() for b in inst_bytes])
                dt_idx = pd.to_datetime(dates_ns[label_dates], unit="ns")
                inst_idx = all_insts[label_insts]
                mi = pd.MultiIndex.from_arrays(
                    [dt_idx, inst_idx], names=["datetime", "instrument"]
                )
                return pd.Series(raw.astype("float32"), index=mi)

            # Fast path: pandas DataFrame fixed format (block0_values)
            if "block0_values" in grp_keys:
                raw = grp["block0_values"][:]
                if raw.ndim == 2:
                    raw = raw[:, 0]
                dates_ns = grp["axis1_level0"][:]
                inst_bytes = grp["axis1_level1"][:]
                label_dates = grp["axis1_label0"][:]
                label_insts = grp["axis1_label1"][:]
                all_insts = np.array([b.decode() for b in inst_bytes])
                dt_idx = pd.to_datetime(dates_ns[label_dates], unit="ns")
                inst_idx = all_insts[label_insts]
                mi = pd.MultiIndex.from_arrays(
                    [dt_idx, inst_idx], names=["datetime", "instrument"]
                )
                return pd.Series(raw.astype("float32"), index=mi)

        # Pandas fallback
        if "factor" in keys:
            vals = pd.read_hdf(result_h5_path, key="factor")
        elif len(keys) == 1:
            vals = pd.read_hdf(result_h5_path, key=keys[0])
        else:
            return None

        if isinstance(vals, pd.DataFrame):
            vals = vals.iloc[:, 0]
        if not isinstance(vals.index, pd.MultiIndex):
            vals = vals.stack()
            vals.index.names = ["datetime", "instrument"]
        return vals
    except Exception as e:
        print(f"  [warn] Cannot load {result_h5_path}: {e}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter factor library by IC decay at N-day horizon.")
    parser.add_argument("--input", required=True, help="Input factor JSON library path")
    parser.add_argument("--output", required=True, help="Output (filtered) JSON library path")
    parser.add_argument(
        "--horizon", type=int, default=5, help="Horizon in trading days (default: 5)"
    )
    parser.add_argument(
        "--min-ic", type=float, default=0.0, help="Minimum IC at horizon (default: 0.0)"
    )
    parser.add_argument(
        "--data-path",
        default="git_ignore_folder/factor_implementation_source_data/daily_pv.h5",
        help="Path to daily_pv.h5 with $close column",
    )
    parser.add_argument(
        "--instruments-file",
        default="data/qlib/cn_data/instruments/csi300.txt",
        help="Qlib instruments file to filter IC to (e.g. csi300.txt). Empty = all instruments.",
    )
    parser.add_argument(
        "--use-abs-ic", action="store_true",
        help="Filter by |IC| rather than signed IC (keeps contrarian factors)"
    )
    parser.add_argument(
        "--report", action="store_true", help="Print per-factor IC breakdown table"
    )
    args = parser.parse_args()

    # Load instrument universe filter (e.g. CSI300 only)
    instruments_filter: Optional[set] = None
    if args.instruments_file:
        inst_file = Path(args.instruments_file)
        if inst_file.exists():
            inst_df = pd.read_csv(
                inst_file, sep="\t", header=None, names=["instrument", "start", "end"]
            )
            instruments_filter = set(inst_df["instrument"].str.upper())
            print(f"Instrument filter: {len(instruments_filter)} from {inst_file.name}", flush=True)
        else:
            print(f"Instrument file not found: {inst_file} — using all instruments", flush=True)
    else:
        print("No instrument filter — using all instruments", flush=True)

    input_path = Path(args.input)
    output_path = Path(args.output)
    h5_path = Path(args.data_path)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if not h5_path.exists():
        print(f"ERROR: H5 data file not found: {h5_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        library = json.load(f)

    factors = library.get("factors", {})
    total = len(factors)
    print(f"Loaded {total} factors from {input_path}", flush=True)

    horizons = sorted({1, args.horizon, 10, 20})
    print(f"Building forward-return matrices (2016-2025) ...", flush=True)
    from quantaalpha.factors.decay_filter import (
        _build_fwd_returns,
        _load_factor_wide_from_h5,
        compute_ic_wide_with_fwd_rets,
    )
    import gc
    _date_range = (pd.Timestamp("2016-01-01"), pd.Timestamp("2025-12-31"))
    # Load price_df as fallback when h5py fast path fails
    price_df = None
    try:
        price_df = pd.read_hdf(str(h5_path), key="data")[["$close"]]
        print(f"  Loaded price_df: {len(price_df)} rows", flush=True)
    except Exception as exc:
        print(f"  Could not load price_df: {exc}", flush=True)
    _, _, fwd_rets = _build_fwd_returns(
        price_df, horizons, exec_lag=1,
        date_range=_date_range, h5_path=str(h5_path),
        instruments=instruments_filter,
    )
    if not fwd_rets:
        print("ERROR: could not build forward return matrices", file=sys.stderr)
        sys.exit(1)
    gc.collect()
    print(f"Forward-return matrices ready (horizons={horizons}).", flush=True)

    # MD5-pkl cache dir: fall back when H5 workspace files are missing
    _pkl_cache_dir = Path(os.environ.get("FACTOR_CACHE_DIR", "data/results/factor_cache"))

    def _load_factor_wide_from_pkl(factor_expr: str) -> "pd.DataFrame | None":
        """Load factor series from MD5-keyed pkl cache and reshape to wide (date × instrument)."""
        import hashlib, pickle
        cache_key = hashlib.md5(factor_expr.encode()).hexdigest()
        cache_file = _pkl_cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "rb") as fh:
                series = pickle.load(fh)
            if not isinstance(series, pd.Series):
                return None
            # Normalise index to (datetime, instrument)
            if series.index.names == ["instrument", "datetime"]:
                series.index = series.index.swaplevel("instrument", "datetime")
            series.index.names = ["datetime", "instrument"]
            wide = series.unstack("instrument")
            if instruments_filter:
                cols = [c for c in wide.columns if str(c).upper() in instruments_filter]
                wide = wide[cols]
            # Apply date range filter
            if _date_range:
                wide = wide.loc[_date_range[0]:_date_range[1]]
            return wide if not wide.empty else None
        except Exception as e:
            print(f"  [warn] pkl cache load failed [{cache_key}]: {e}", flush=True)
            return None

    # Stream factors one by one — peak RAM = fwd_rets + one factor at a time
    no_cache_ids: list[str] = []
    all_metrics: dict[str, dict] = {}

    for i, (fid, finfo) in enumerate(factors.items(), 1):
        fname = finfo.get("factor_name", fid)
        factor_expr = finfo.get("factor_expression", "")
        result_h5 = finfo.get("cache_location", {}).get("result_h5_path")
        factor_wide = None
        if result_h5 and Path(result_h5).exists():
            factor_wide = _load_factor_wide_from_h5(
                result_h5, date_range=_date_range, instruments=instruments_filter
            )
        if factor_wide is None and factor_expr:
            factor_wide = _load_factor_wide_from_pkl(factor_expr)
        if factor_wide is None:
            no_cache_ids.append(fid)
            print(f"  [{i}/{total}] SKIP {fname}: no cached values", flush=True)
            continue
        if factor_wide is None:
            no_cache_ids.append(fid)
            print(f"  [{i}/{total}] SKIP {fname}: load failed", flush=True)
            continue
        metrics = compute_ic_wide_with_fwd_rets(factor_wide, fwd_rets, horizons=horizons)
        del factor_wide
        all_metrics[fid] = metrics
        ic_at_h = metrics.get(f"ic_{args.horizon}d")
        ic_check = abs(ic_at_h) if (args.use_abs_ic and ic_at_h is not None) else ic_at_h
        passes = ic_check is not None and ic_check > args.min_ic
        print(
            f"  [{i}/{total}] {'PASS' if passes else 'FAIL'} {fname}: "
            f"ic_1d={metrics.get('ic_1d')}, ic_{args.horizon}d={ic_at_h}",
            flush=True,
        )

    del fwd_rets
    gc.collect()

    kept_ids = list(no_cache_ids)  # conservatively keep factors with no cache
    dropped_ids = []
    rows = []

    for fid, metrics in all_metrics.items():
        finfo = factors[fid]
        fname = finfo.get("factor_name", fid)

        br = finfo.setdefault("backtest_results", {})
        for k, v in metrics.items():
            br[k] = v

        ic_at_horizon = metrics.get(f"ic_{args.horizon}d")
        ic_check = abs(ic_at_horizon) if (args.use_abs_ic and ic_at_horizon is not None) else ic_at_horizon
        passes = ic_check is not None and ic_check > args.min_ic

        row = {
            "name": fname,
            **{f"ic_{n}d": metrics.get(f"ic_{n}d") for n in horizons},
            "half_life": metrics.get("decay_half_life"),
            "pass": passes,
        }
        rows.append(row)

        if passes:
            kept_ids.append(fid)
        else:
            dropped_ids.append(fid)

    # Build output library
    out_factors = {fid: factors[fid] for fid in kept_ids}
    out_library = dict(library)
    out_library["factors"] = out_factors
    out_library.setdefault("metadata", {})
    out_library["metadata"]["total_factors"] = len(out_factors)
    out_library["metadata"]["decay_filter"] = {
        "horizon": args.horizon,
        "min_ic": args.min_ic,
        "original_count": total,
        "kept_count": len(kept_ids),
        "dropped_count": len(dropped_ids),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out_library, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Total:   {total}")
    print(f"Kept:    {len(kept_ids)}  ({100*len(kept_ids)//max(total,1)}%)")
    print(f"Dropped: {len(dropped_ids)}  ({100*len(dropped_ids)//max(total,1)}%)")
    print(f"Output:  {output_path}")

    if args.report and rows:
        df = pd.DataFrame(rows).set_index("name")
        ic_cols = [c for c in df.columns if c.startswith("ic_")]
        print(f"\nPer-factor IC decay (horizon={args.horizon}d):\n")
        print(df[ic_cols + ["half_life", "pass"]].to_string())

    if dropped_ids:
        print(f"\nDropped factors:")
        for fid in dropped_ids:
            fname = factors[fid].get("factor_name", fid)
            ic_h = factors[fid].get("backtest_results", {}).get(f"ic_{args.horizon}d")
            print(f"  {fname}: ic_{args.horizon}d={ic_h}")


if __name__ == "__main__":
    main()
