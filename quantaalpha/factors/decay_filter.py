"""
Factor Decay Filter — multi-horizon IC analysis.

For each factor, computes the cross-sectional Spearman IC against forward returns
at 1, 5, 10 and 20-day horizons (with a 1-day execution lag). Factors whose IC
is non-positive at the target horizon should be discarded: they cannot be traded
profitably once execution delay is accounted for.

Usage:
    from quantaalpha.factors.decay_filter import compute_decay_profile, passes_decay_gate

    # From factor values (MultiIndex Series)
    metrics = compute_decay_profile(factor_series, h5_path="git_ignore_folder/.../daily_pv.h5")
    if not passes_decay_gate(metrics, horizon=5):
        # reject factor

Public API:
    compute_decay_profile(factor_series, h5_path, horizons, exec_lag) -> dict
    compute_decay_profile_from_df(factor_series, price_df, horizons, exec_lag) -> dict
    passes_decay_gate(metrics, horizon, min_ic) -> bool
    fit_decay_half_life(horizons, ic_values) -> float | None
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default horizons (trading days) at which IC is evaluated
DEFAULT_HORIZONS: List[int] = [1, 5, 10, 20]

# Minimum number of (date, instrument) observations per day to include that
# day's IC in the rolling average.  Days with fewer cross-sectional points are
# skipped (e.g. IPO/suspension heavy days).
MIN_STOCKS_PER_DAY = 10

# Minimum number of valid daily IC observations needed before we trust the mean.
MIN_IC_DAYS = 20


def compute_decay_profile(
    factor_series: pd.Series,
    h5_path: str,
    horizons: List[int] = DEFAULT_HORIZONS,
    exec_lag: int = 1,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Dict:
    """Compute multi-horizon IC decay profile for a single factor.

    Loads ``$close`` prices from *h5_path* (``daily_pv.h5``, key ``data``),
    then delegates to :func:`compute_decay_profile_from_df`.

    Args:
        factor_series: MultiIndex(datetime, instrument) factor values.
        h5_path: Path to ``daily_pv.h5`` containing a ``$close`` column under
            key ``data``.
        horizons: Forward-return horizons in trading days.
        exec_lag: Execution lag in days (default 1 — buy at next open modelled
            as next close).
        eval_start: Optional start date filter (inclusive, e.g. ``"2022-01-01"``).
        eval_end: Optional end date filter (inclusive).

    Returns:
        Dict with keys ``ic_Nd`` for each horizon *N*, ``decay_half_life``,
        ``decay_pass_5d``, and ``daily_ic_count``.
    """
    h5_path = str(h5_path)
    try:
        price_df = pd.read_hdf(h5_path, key="data")[["$close"]]
    except Exception as exc:
        logger.warning("decay_filter: failed to load H5 %s: %s", h5_path, exc)
        return _empty_profile()

    return compute_decay_profile_from_df(
        factor_series, price_df, horizons=horizons, exec_lag=exec_lag,
        eval_start=eval_start, eval_end=eval_end,
    )


def _load_close_from_h5_fast(
    h5_path: str,
    date_range: Optional[tuple] = None,
    instruments: Optional[set] = None,
) -> Optional[pd.Series]:
    """Load ``$close`` from a daily_pv.h5 file using h5py directly.

    Avoids the slow pandas ``read_hdf`` MultiIndex reconstruction by reading
    raw arrays and filtering in numpy before constructing any pandas objects.
    This is typically 10–30× faster than ``pd.read_hdf`` for this file format.

    Returns a MultiIndex(datetime, instrument) float32 Series, or None on error.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        logger.debug("h5py not available; falling back to pandas read_hdf")
        return None

    try:
        with h5py.File(h5_path, "r") as fh:
            grp = fh["data"]
            items = [x.decode() for x in grp["block0_items"][:]]
            if "$close" not in items:
                logger.warning("$close not found in H5 items: %s", items)
                return None
            close_idx = items.index("$close")

            # Raw values — shape (N, n_cols), float32
            close_vals = grp["block0_values"][:, close_idx].astype("float32")

            # MultiIndex: dates encoded as int16 → level0 (nanoseconds), instruments as int16 → level1
            dates_ns = grp["axis1_level0"][:]          # shape (n_unique_dates,)
            inst_bytes = grp["axis1_level1"][:]        # shape (n_unique_insts,)
            label_dates = grp["axis1_label0"][:]       # int16, shape (N,) → index into dates_ns
            label_insts = grp["axis1_label1"][:]       # int16, shape (N,) → index into inst_bytes

        # Build instrument string array for all unique instruments
        all_insts = np.array([b.decode() for b in inst_bytes])  # shape (n_unique_insts,)

        # Row mask: date filter (in numpy, fast)
        row_mask = np.ones(len(close_vals), dtype=bool)
        if date_range is not None:
            min_dt, max_dt = date_range
            min_ns = int(pd.Timestamp(min_dt).value)
            max_ns = int(pd.Timestamp(max_dt).value)
            date_mask = (dates_ns >= min_ns) & (dates_ns <= max_ns)  # per unique-date
            row_mask &= date_mask[label_dates]

        # Instrument filter (optional)
        if instruments is not None:
            inst_mask = np.isin(all_insts, list(instruments))   # per unique-inst
            row_mask &= inst_mask[label_insts]

        if not row_mask.any():
            return None

        close_f = close_vals[row_mask]
        label_d = label_dates[row_mask]
        label_i = label_insts[row_mask]

        # Build final MultiIndex Series
        dt_idx = pd.to_datetime(dates_ns[label_d], unit="ns")
        inst_idx = all_insts[label_i]
        mi = pd.MultiIndex.from_arrays([dt_idx, inst_idx], names=["datetime", "instrument"])
        return pd.Series(close_f.astype("float32"), index=mi)

    except Exception as exc:
        logger.warning("_load_close_from_h5_fast failed: %s", exc)
        return None


def _load_close_wide_from_h5(
    h5_path: str,
    date_range: Optional[tuple] = None,
    instruments: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """Load ``$close`` as a wide DataFrame (days × instruments) via h5py.

    Builds the dense matrix directly in numpy — no pandas MultiIndex created,
    so peak RAM is much lower (~30 MB for full A-share universe 2016-2025
    in float16 vs >500 MB via the Series + unstack path).

    Returns a float16 DataFrame indexed by datetime, columns = instruments.
    Returns None on error.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        return None

    try:
        with h5py.File(h5_path, "r") as fh:
            grp = fh["data"]
            items = [x.decode() for x in grp["block0_items"][:]]
            if "$close" not in items:
                return None
            close_idx = items.index("$close")

            close_vals = grp["block0_values"][:, close_idx]   # float32, shape (N,)
            dates_ns = grp["axis1_level0"][:]                  # int64 nanos
            inst_bytes = grp["axis1_level1"][:]
            label_dates = grp["axis1_label0"][:]               # int16 → date index
            label_insts = grp["axis1_label1"][:]               # int16 → inst index

        all_insts = np.array([b.decode() for b in inst_bytes])

        # Date filter in numpy
        row_mask = np.ones(len(close_vals), dtype=bool)
        if date_range is not None:
            min_dt, max_dt = date_range
            min_ns = int(pd.Timestamp(min_dt).value)
            max_ns = int(pd.Timestamp(max_dt).value)
            date_mask = (dates_ns >= min_ns) & (dates_ns <= max_ns)
            row_mask &= date_mask[label_dates]
        if instruments is not None:
            inst_mask = np.isin(all_insts, list(instruments))
            row_mask &= inst_mask[label_insts]

        if not row_mask.any():
            return None

        close_f = close_vals[row_mask].astype("float32")
        label_d = label_dates[row_mask]
        label_i = label_insts[row_mask]

        # Map compressed int16 codes to dense row/col indices
        unique_dcodes, row_idx = np.unique(label_d, return_inverse=True)
        unique_icodes, col_idx = np.unique(label_i, return_inverse=True)
        n_dates, n_insts = len(unique_dcodes), len(unique_icodes)

        # Build dense matrix directly in float16 (no float64 temporary)
        dense = np.full((n_dates, n_insts), np.nan, dtype=np.float16)
        dense[row_idx, col_idx] = close_f.astype("float16")

        dt_index = pd.DatetimeIndex(
            pd.to_datetime(dates_ns[unique_dcodes], unit="ns")
        ).sort_values()
        inst_cols = all_insts[unique_icodes]

        df = pd.DataFrame(dense, index=pd.to_datetime(dates_ns[unique_dcodes], unit="ns"), columns=inst_cols)
        return df.sort_index()

    except Exception as exc:
        logger.warning("_load_close_wide_from_h5 failed: %s", exc)
        return None


def _build_fwd_returns(
    price_df: pd.DataFrame,
    horizons: List[int],
    exec_lag: int = 1,
    instruments: Optional[set] = None,
    date_range: Optional[tuple] = None,
    h5_path: Optional[str] = None,
) -> tuple:
    """Pre-compute forward return matrices (call once per price_df).

    When *h5_path* is provided, uses the fast h5py loader to bypass the slow
    pandas MultiIndex reconstruction.  Filters to *instruments* and
    *date_range* **before** unstacking so the dense matrix is minimal.

    Returns:
        (close_by_inst, None, fwd_rets_dict) where fwd_rets_dict maps
        horizon → float16 DataFrame(days × instruments).
    """
    # --- Fast path: load directly from H5 via h5py (builds wide matrix, no pandas MultiIndex) ---
    if h5_path is not None:
        tail_days = (max(horizons) + exec_lag) * 2 if date_range else None
        dr = date_range
        if dr is not None and tail_days:
            min_d, max_d = dr
            dr = (min_d, max_d + pd.Timedelta(days=tail_days))
        close_by_inst = _load_close_wide_from_h5(h5_path, date_range=dr, instruments=instruments)
        if close_by_inst is None:
            logger.warning("Fast H5 wide load failed; falling back to price_df")
        else:
            exec_price = close_by_inst.shift(-exec_lag)
            fwd_rets: Dict[int, pd.DataFrame] = {}
            for n in horizons:
                exit_price = close_by_inst.shift(-(n + exec_lag))
                fwd_rets[n] = (exit_price / exec_price - 1).astype("float16")
                del exit_price
            del exec_price
            return close_by_inst, None, fwd_rets

    # --- Slow path: extract from pre-loaded DataFrame ---
    close = None
    if close is None:
        if price_df is None:
            return None, None, {}
        close_col = "$close" if "$close" in price_df.columns else price_df.columns[0]
        close = _normalise_series(price_df[close_col])
        if close is None:
            return None, None, {}

        if instruments is not None:
            mask_inst = close.index.get_level_values("instrument").isin(instruments)
            close = close[mask_inst]

        if date_range is not None:
            min_date, max_date = date_range
            tail_days = (max(horizons) + exec_lag) * 2
            max_date_ext = max_date + pd.Timedelta(days=tail_days)
            dates_idx = close.index.get_level_values("datetime")
            close = close[(dates_idx >= min_date) & (dates_idx <= max_date_ext)]

    if len(close) == 0:
        return None, None, {}

    # Unstack in float64 (pandas Cython kernel requires float32/64), then cast to float16
    close_by_inst = close.astype("float64").unstack("instrument").sort_index()
    close_by_inst = close_by_inst.astype("float16")
    del close

    exec_price = close_by_inst.shift(-exec_lag)
    fwd_rets: Dict[int, pd.DataFrame] = {}
    for n in horizons:
        exit_price = close_by_inst.shift(-(n + exec_lag))
        fwd_rets[n] = (exit_price / exec_price - 1).astype("float16")
        del exit_price

    del exec_price
    return close_by_inst, None, fwd_rets


def _load_factor_wide_from_h5(
    h5_path: str,
    date_range: Optional[tuple] = None,
    instruments: Optional[set] = None,
) -> Optional[pd.DataFrame]:
    """Load a factor result.h5 workspace file as wide DataFrame (days × instruments).

    Uses h5py directly to avoid pandas MultiIndex construction overhead.
    Returns float32 wide DataFrame, or None on error.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        return None

    try:
        with h5py.File(h5_path, "r") as fh:
            keys = list(fh.keys())
            top_key = keys[0]
            grp = fh[top_key]

            if "values" not in grp:
                return None

            raw = grp["values"][:]                     # float64, shape (N,)
            dates_ns = grp["index_level0"][:]          # int64 nanos
            inst_bytes = grp["index_level1"][:]
            label_dates = grp["index_label0"][:]       # int16
            label_insts = grp["index_label1"][:]       # int16

        all_insts = np.array([b.decode() for b in inst_bytes])

        row_mask = np.ones(len(raw), dtype=bool)
        if date_range is not None:
            min_dt, max_dt = date_range
            min_ns = int(pd.Timestamp(min_dt).value)
            max_ns = int(pd.Timestamp(max_dt).value)
            row_mask &= (dates_ns[label_dates] >= min_ns) & (dates_ns[label_dates] <= max_ns)
        if instruments is not None:
            inst_mask = np.isin(all_insts, list(instruments))
            row_mask &= inst_mask[label_insts]

        if not row_mask.any():
            return None

        raw_f = raw[row_mask].astype("float32")
        label_d = label_dates[row_mask]
        label_i = label_insts[row_mask]

        unique_dcodes, row_idx = np.unique(label_d, return_inverse=True)
        unique_icodes, col_idx = np.unique(label_i, return_inverse=True)

        dense = np.full((len(unique_dcodes), len(unique_icodes)), np.nan, dtype=np.float32)
        dense[row_idx, col_idx] = raw_f

        df = pd.DataFrame(
            dense,
            index=pd.to_datetime(dates_ns[unique_dcodes], unit="ns"),
            columns=all_insts[unique_icodes],
        )
        return df.sort_index()

    except Exception as exc:
        logger.debug("_load_factor_wide_from_h5 failed for %s: %s", h5_path, exc)
        return None


def compute_decay_profiles_batch(
    factors: Dict[str, pd.Series],
    price_df: pd.DataFrame,
    horizons: List[int] = DEFAULT_HORIZONS,
    exec_lag: int = 1,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Dict[str, Dict]:
    """Compute IC decay profiles for many factors efficiently.

    Collects the union of instruments and date range across all factors,
    filters *price_df* down to that subset **before** unstacking, then
    builds forward-return matrices once (as float16) and iterates factors.

    Args:
        factors: Mapping of factor_name → MultiIndex(datetime, instrument) Series.
        price_df: Pre-loaded long-form DataFrame with ``$close`` column.
        horizons: Forward-return horizons in trading days.
        exec_lag: Execution lag in trading days.
        eval_start / eval_end: Optional date range filter.

    Returns:
        Dict mapping factor_name → decay profile dict.
    """
    if not factors:
        return {}

    # --- Collect union of instruments and date range across all factors ---
    all_inst: set = set()
    min_date = pd.Timestamp.max
    max_date = pd.Timestamp.min
    for s in factors.values():
        s = _normalise_series(s)
        if s is None:
            continue
        all_inst.update(s.index.get_level_values("instrument").unique())
        dates = s.index.get_level_values("datetime")
        if len(dates):
            min_date = min(min_date, dates.min())
            max_date = max(max_date, dates.max())

    if not all_inst or min_date > max_date:
        return {name: _empty_profile(horizons) for name in factors}

    # --- Build forward-return matrices (filtered, float16) ---
    close_by_inst, _, fwd_rets = _build_fwd_returns(
        price_df, horizons, exec_lag,
        instruments=all_inst,
        date_range=(min_date, max_date),
    )
    if close_by_inst is None:
        return {name: _empty_profile(horizons) for name in factors}

    all_instruments = set(close_by_inst.columns)
    results = {}

    for name, factor_series in factors.items():
        factor_series = _normalise_series(factor_series)
        if factor_series is None or len(factor_series) == 0:
            results[name] = _empty_profile(horizons)
            continue

        fac_instruments = set(factor_series.index.get_level_values("instrument").unique())
        common = fac_instruments & all_instruments
        if not common:
            results[name] = _empty_profile(horizons)
            continue

        factor_series = factor_series.loc[
            factor_series.index.get_level_values("instrument").isin(common)
        ]
        # Unstack in float64 then cast — float16 unstack not supported by pandas
        factor_wide = factor_series.astype("float64").unstack("instrument").sort_index().astype("float16")

        if eval_start:
            factor_wide = factor_wide.loc[factor_wide.index >= pd.Timestamp(eval_start)]
        if eval_end:
            factor_wide = factor_wide.loc[factor_wide.index <= pd.Timestamp(eval_end)]
        if len(factor_wide) == 0:
            results[name] = _empty_profile(horizons)
            continue

        ic_by_horizon: Dict[int, float] = {}
        daily_count = 0
        for n in horizons:
            ret_wide = fwd_rets[n].reindex(factor_wide.index)
            daily_ics = _daily_spearman_ic(factor_wide, ret_wide)
            valid = daily_ics.dropna()
            daily_count = len(valid)
            ic_by_horizon[n] = float(valid.mean()) if daily_count >= MIN_IC_DAYS else float("nan")
            del ret_wide

        del factor_wide, factor_series

        result: Dict = {}
        for n in horizons:
            result[f"ic_{n}d"] = None if np.isnan(ic_by_horizon[n]) else round(ic_by_horizon[n], 6)
        result["daily_ic_count"] = daily_count

        valid_pairs = [(n, ic_by_horizon[n]) for n in horizons if not np.isnan(ic_by_horizon[n])]
        result["decay_half_life"] = fit_decay_half_life(
            [p[0] for p in valid_pairs], [p[1] for p in valid_pairs]
        ) if len(valid_pairs) >= 2 else None

        ic_5d = ic_by_horizon.get(5, float("nan"))
        result["decay_pass_5d"] = bool(not np.isnan(ic_5d) and ic_5d > 0)
        results[name] = result

    return results


def compute_decay_profile_from_df(
    factor_series: pd.Series,
    price_df: pd.DataFrame,
    horizons: List[int] = DEFAULT_HORIZONS,
    exec_lag: int = 1,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Dict:
    """Compute multi-horizon IC decay profile using a pre-loaded price DataFrame.

    Args:
        factor_series: MultiIndex(datetime, instrument) factor values.
        price_df: DataFrame with MultiIndex(datetime, instrument) and a
            ``$close`` column (or just a single column treated as close).
        horizons: Forward-return horizons in trading days.
        exec_lag: Execution lag in trading days.
        eval_start: Optional start date filter.
        eval_end: Optional end date filter.

    Returns:
        Dict with ``ic_1d``, ``ic_5d``, ``ic_10d``, ``ic_20d`` (or the keys for
        each horizon in *horizons*), ``decay_half_life`` (float or None),
        ``decay_pass_5d`` (bool), ``daily_ic_count`` (int).
    """
    factor_series = _normalise_series(factor_series)
    if factor_series is None or len(factor_series) == 0:
        return _empty_profile(horizons)

    # Identify close column
    close_col = "$close" if "$close" in price_df.columns else price_df.columns[0]
    close = _normalise_series(price_df[close_col])
    if close is None:
        return _empty_profile(horizons)

    # Align on common instruments
    common_instruments = factor_series.index.get_level_values("instrument").unique().intersection(
        close.index.get_level_values("instrument").unique()
    )
    if len(common_instruments) == 0:
        logger.warning("decay_filter: no common instruments between factor and price data")
        return _empty_profile(horizons)

    factor_series = factor_series.loc[
        factor_series.index.get_level_values("instrument").isin(common_instruments)
    ]
    close = close.loc[
        close.index.get_level_values("instrument").isin(common_instruments)
    ]

    # Pre-compute forward returns once — unstack in float64, then cast to float16
    close_by_inst = close.astype("float64").unstack("instrument").sort_index().astype("float16")
    del close
    exec_price = close_by_inst.shift(-exec_lag)
    fwd_rets: Dict[int, pd.DataFrame] = {}
    for n in horizons:
        exit_price = close_by_inst.shift(-(n + exec_lag))
        fwd_rets[n] = (exit_price / exec_price - 1).astype("float16")
        del exit_price
    del exec_price

    # Unstack factor in float64, then cast to float16
    factor_wide = factor_series.astype("float64").unstack("instrument").sort_index().astype("float16")

    # Optional date filter
    if eval_start:
        start_ts = pd.Timestamp(eval_start)
        factor_wide = factor_wide.loc[factor_wide.index >= start_ts]
    if eval_end:
        end_ts = pd.Timestamp(eval_end)
        factor_wide = factor_wide.loc[factor_wide.index <= end_ts]

    if len(factor_wide) == 0:
        return _empty_profile(horizons)

    # Compute daily cross-sectional Spearman IC for each horizon
    ic_by_horizon: Dict[int, float] = {}
    daily_count = None
    for n in horizons:
        ret_wide = fwd_rets[n].reindex(factor_wide.index)
        daily_ics = _daily_spearman_ic(factor_wide, ret_wide)
        valid = daily_ics.dropna()
        daily_count = len(valid)
        if daily_count < MIN_IC_DAYS:
            ic_by_horizon[n] = float("nan")
        else:
            ic_by_horizon[n] = float(valid.mean())

    # Build output dict
    result: Dict = {}
    for n in horizons:
        result[f"ic_{n}d"] = None if np.isnan(ic_by_horizon[n]) else round(ic_by_horizon[n], 6)
    result["daily_ic_count"] = daily_count or 0

    # Decay half-life fit
    valid_pairs = [(n, ic_by_horizon[n]) for n in horizons if not np.isnan(ic_by_horizon[n])]
    if len(valid_pairs) >= 2:
        ns = [p[0] for p in valid_pairs]
        ics = [p[1] for p in valid_pairs]
        result["decay_half_life"] = fit_decay_half_life(ns, ics)
    else:
        result["decay_half_life"] = None

    ic_5d = ic_by_horizon.get(5, float("nan"))
    result["decay_pass_5d"] = bool(not np.isnan(ic_5d) and ic_5d > 0)

    return result


def compute_ic_wide_with_fwd_rets(
    factor_wide: pd.DataFrame,
    fwd_rets: Dict[int, pd.DataFrame],
    horizons: List[int] = DEFAULT_HORIZONS,
) -> Dict:
    """Compute IC decay profile from a pre-unstacked wide factor DataFrame.

    Use this when the factor is already in wide format (days × instruments),
    e.g. loaded via :func:`_load_factor_wide_from_h5`.  No unstack step needed.

    Args:
        factor_wide: Wide float32 DataFrame (days × instruments).
        fwd_rets: Dict from :func:`_build_fwd_returns` (horizon → float16 DF).
        horizons: Horizons to compute IC for.
    """
    if factor_wide is None or len(factor_wide) == 0:
        return _empty_profile(horizons)

    # Restrict columns to intersection with fwd_rets
    all_instruments = set()
    for ret_df in fwd_rets.values():
        all_instruments.update(ret_df.columns)
    common_cols = [c for c in factor_wide.columns if c in all_instruments]
    if not common_cols:
        return _empty_profile(horizons)
    # Keep factor values in float32 — better rank precision than float16
    factor_wide = factor_wide[common_cols]

    ic_by_horizon: Dict[int, float] = {}
    daily_count = 0
    for n in horizons:
        ret_wide = fwd_rets[n].reindex(factor_wide.index)[common_cols]
        daily_ics = _daily_spearman_ic(factor_wide, ret_wide)
        valid = daily_ics.dropna()
        daily_count = len(valid)
        ic_by_horizon[n] = float(valid.mean()) if daily_count >= MIN_IC_DAYS else float("nan")
        del ret_wide

    del factor_wide

    result: Dict = {}
    for n in horizons:
        result[f"ic_{n}d"] = None if np.isnan(ic_by_horizon[n]) else round(ic_by_horizon[n], 6)
    result["daily_ic_count"] = daily_count

    valid_pairs = [(n, ic_by_horizon[n]) for n in horizons if not np.isnan(ic_by_horizon[n])]
    result["decay_half_life"] = fit_decay_half_life(
        [p[0] for p in valid_pairs], [p[1] for p in valid_pairs]
    ) if len(valid_pairs) >= 2 else None

    ic_5d = ic_by_horizon.get(5, float("nan"))
    result["decay_pass_5d"] = bool(not np.isnan(ic_5d) and ic_5d > 0)
    return result


def compute_ic_from_prebuilt(
    factor_series: pd.Series,
    fwd_rets: Dict[int, pd.DataFrame],
    horizons: List[int] = DEFAULT_HORIZONS,
    eval_start: Optional[str] = None,
    eval_end: Optional[str] = None,
) -> Dict:
    """Compute IC decay profile using pre-built forward return matrices.

    Call this inside a loop over many factors; build *fwd_rets* ONCE outside
    the loop via :func:`_build_fwd_returns` to avoid rebuilding it per factor.

    Args:
        factor_series: MultiIndex(datetime, instrument) factor values.
        fwd_rets: Dict mapping horizon → wide DataFrame(days × instruments)
            as returned by ``_build_fwd_returns(...)[2]``.
        horizons: Horizons to compute IC for.
        eval_start / eval_end: Optional date filters.

    Returns:
        Same dict structure as :func:`compute_decay_profile`.
    """
    factor_series = _normalise_series(factor_series)
    if factor_series is None or len(factor_series) == 0:
        return _empty_profile(horizons)

    all_instruments = set()
    for ret_df in fwd_rets.values():
        all_instruments.update(ret_df.columns)

    fac_instruments = set(factor_series.index.get_level_values("instrument").unique())
    common = fac_instruments & all_instruments
    if not common:
        return _empty_profile(horizons)

    factor_series = factor_series.loc[
        factor_series.index.get_level_values("instrument").isin(common)
    ]

    # Unstack in float64 (pandas requirement), then cast to float16
    factor_wide = (
        factor_series.astype("float64")
        .unstack("instrument")
        .sort_index()
        .astype("float16")
    )
    del factor_series

    if eval_start:
        factor_wide = factor_wide.loc[factor_wide.index >= pd.Timestamp(eval_start)]
    if eval_end:
        factor_wide = factor_wide.loc[factor_wide.index <= pd.Timestamp(eval_end)]

    if len(factor_wide) == 0:
        return _empty_profile(horizons)

    ic_by_horizon: Dict[int, float] = {}
    daily_count = 0
    for n in horizons:
        ret_wide = fwd_rets[n].reindex(factor_wide.index)
        daily_ics = _daily_spearman_ic(factor_wide, ret_wide)
        valid = daily_ics.dropna()
        daily_count = len(valid)
        ic_by_horizon[n] = float(valid.mean()) if daily_count >= MIN_IC_DAYS else float("nan")
        del ret_wide

    del factor_wide

    result: Dict = {}
    for n in horizons:
        result[f"ic_{n}d"] = None if np.isnan(ic_by_horizon[n]) else round(ic_by_horizon[n], 6)
    result["daily_ic_count"] = daily_count

    valid_pairs = [(n, ic_by_horizon[n]) for n in horizons if not np.isnan(ic_by_horizon[n])]
    result["decay_half_life"] = fit_decay_half_life(
        [p[0] for p in valid_pairs], [p[1] for p in valid_pairs]
    ) if len(valid_pairs) >= 2 else None

    ic_5d = ic_by_horizon.get(5, float("nan"))
    result["decay_pass_5d"] = bool(not np.isnan(ic_5d) and ic_5d > 0)
    return result


def passes_decay_gate(
    metrics: Dict,
    horizon: int = 5,
    min_ic: float = 0.0,
) -> bool:
    """Return True if the factor passes the decay gate at *horizon* days.

    Args:
        metrics: Output from :func:`compute_decay_profile`.
        horizon: Horizon in trading days (must be in metrics).
        min_ic: Minimum required IC (default 0 — strictly positive).
    """
    key = f"ic_{horizon}d"
    ic = metrics.get(key)
    if ic is None:
        return False
    return float(ic) > min_ic


def fit_decay_half_life(horizons: List[int], ic_values: List[float]) -> Optional[float]:
    """Fit exponential decay ic(t) = ic_0 * exp(-λt) and return half-life = ln2 / λ.

    Returns None if the fit fails or if IC is not decaying (λ ≤ 0).
    """
    if len(horizons) < 2:
        return None

    ns = np.asarray(horizons, dtype=float)
    ics = np.asarray(ic_values, dtype=float)

    # Only use positive IC values for exponential fit (log undefined for ≤ 0)
    mask = ics > 0
    if mask.sum() < 2:
        return None

    ns_pos = ns[mask]
    ics_pos = ics[mask]

    # log-linear fit: log(ic) = log(ic_0) - λ * t
    try:
        coeffs = np.polyfit(ns_pos, np.log(ics_pos), deg=1)
        lam = -coeffs[0]  # decay rate (positive = decaying)
        if lam <= 0:
            return None  # IC is growing, no meaningful half-life
        return round(float(np.log(2) / lam), 2)
    except (np.linalg.LinAlgError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise_series(series: pd.Series) -> Optional[pd.Series]:
    """Ensure series has a 2-level MultiIndex named (datetime, instrument)."""
    if not isinstance(series, pd.Series):
        return None
    if not isinstance(series.index, pd.MultiIndex):
        return None
    idx = series.index
    names = list(idx.names)
    if "datetime" not in names or "instrument" not in names:
        # Try to infer: first level datetime-like, second string-like
        if len(names) == 2:
            try:
                if pd.api.types.is_datetime64_any_dtype(idx.get_level_values(0)):
                    series.index.names = ["datetime", "instrument"]
                    return series
            except Exception:
                pass
        return None
    return series


def _daily_spearman_ic(
    factor_wide: pd.DataFrame,  # (days × instruments)
    ret_wide: pd.DataFrame,     # (days × instruments)
) -> pd.Series:
    """Return a Series of cross-sectional Spearman IC, one value per date.

    Vectorized implementation: rank both matrices cross-sectionally then
    compute row-wise Pearson correlation on ranks, handling NaN per row.
    ~100× faster than iterating over dates in Python.
    """
    common_cols = factor_wide.columns.intersection(ret_wide.columns)
    if len(common_cols) == 0:
        return pd.Series(dtype=float)

    # Align to common dates and columns
    common_dates = factor_wide.index.intersection(ret_wide.index)
    f = factor_wide.loc[common_dates, common_cols]
    r = ret_wide.loc[common_dates, common_cols]

    # Cross-sectional ranks (NaN stays NaN)
    f_rank = f.rank(axis=1)
    r_rank = r.rank(axis=1)

    # Mask: only positions where both factor and return are non-NaN
    mask = f_rank.notna() & r_rank.notna()
    n_valid = mask.sum(axis=1)  # per-row valid count

    # Fill NaN with 0 for arithmetic (we account for this in means)
    f_r = f_rank.fillna(0.0)
    r_r = r_rank.fillna(0.0)
    m = mask.astype(float)

    # Row-wise mean of ranks (only over valid positions)
    f_mean = (f_r * m).sum(axis=1) / n_valid.clip(lower=1)
    r_mean = (r_r * m).sum(axis=1) / n_valid.clip(lower=1)

    # Centered ranks (zero out invalid positions)
    f_c = (f_rank.sub(f_mean, axis=0)).fillna(0.0)
    r_c = (r_rank.sub(r_mean, axis=0)).fillna(0.0)

    # Row-wise Pearson on centered ranks = Spearman IC
    num = (f_c * r_c).sum(axis=1)
    denom = np.sqrt((f_c ** 2).sum(axis=1) * (r_c ** 2).sum(axis=1))

    ic_series = num / denom.replace(0, np.nan)
    ic_series[n_valid < MIN_STOCKS_PER_DAY] = np.nan

    # Re-index to original factor dates (dates not in ret get NaN)
    return ic_series.reindex(factor_wide.index)


def _empty_profile(horizons: List[int] = DEFAULT_HORIZONS) -> Dict:
    """Return a profile dict with all None values."""
    result = {f"ic_{n}d": None for n in horizons}
    result["decay_half_life"] = None
    result["decay_pass_5d"] = False
    result["daily_ic_count"] = 0
    return result
