#!/usr/bin/env python3
"""
Custom factor calculator using AlphaAgent expression parser.
Supports the same expression syntax as factor mining.

Features:
1. Parse factor expressions (expr_parser)
2. Compute factor values (function_lib)
3. Output Qlib DataLoader-compatible format
4. Load precomputed factors from cache
"""

import hashlib
import json
import logging
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Add project root (from quantaalpha/backtest/ up two levels)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='quantaalpha')

# Use thread backend for joblib to avoid subprocess importing LLM modules
os.environ.setdefault('JOBLIB_START_METHOD', 'loky')

logger = logging.getLogger(__name__)

# Precision used for all factor Series and DataFrames.
# float16 (half-precision) is the default; override with FACTOR_PRECISION=float32 env var.
# float16 is safe for cross-sectionally normalised factors (RANK/ZSCORE output in [-3, 3]).
_FACTOR_PRECISION = os.environ.get("FACTOR_PRECISION", "float16").lower()
FACTOR_DTYPE = np.float32 if _FACTOR_PRECISION == "float32" else np.float16

DEFAULT_CACHE_DIR = Path(os.environ.get("FACTOR_CACHE_DIR", "data/results/factor_cache"))


class CustomFactorCalculator:
    """
    Custom factor calculator using AlphaAgent expression parser and function lib.
    Loads precomputed factors from cache; can auto-extract cache from main program logs.
    """
    
    def __init__(self, data_df: Optional[pd.DataFrame] = None, cache_dir: Optional[Path] = None, 
                 auto_extract_cache: bool = True, config: Optional[Dict] = None):
        """
        Args:
            data_df: Stock data DataFrame (optional, lazy-loaded).
            cache_dir: Cache directory path (optional).
            auto_extract_cache: Whether to auto-extract cache from main program logs (default True).
            config: Config dict for lazy loading data (optional).
        """
        self._raw_data_df = data_df
        self._data_prepared = False
        self._config = config
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.auto_extract_cache = auto_extract_cache
        self._cache_extracted = False
        # Skip market-specific caches when running cross-market (non-CN) backtest
        data_cfg = (config or {}).get('data', {})
        self._skip_market_cache = data_cfg.get('region', 'cn').lower() != 'cn'
        # Market universe for filtering (loaded lazily, reduces per-factor RAM ~20×)
        self._market: str = data_cfg.get('market', 'csi300')
        self._market_instruments: Optional[frozenset] = None
        
        if data_df is not None and len(data_df) > 0:
            self._prepare_data()
    
    @property
    def data_df(self) -> pd.DataFrame:
        """Lazy-load stock data."""
        if not self._data_prepared:
            if self._raw_data_df is None or len(self._raw_data_df) == 0:
                if self._config is not None:
                    print("  Loading stock data (needed for expression-based factor computation)...")
                    self._raw_data_df = get_qlib_stock_data(self._config)
                else:
                    raise ValueError("No stock data provided and no config for loading")
            self._prepare_data()
        return self._raw_data_df
        
    def _prepare_data(self):
        """Prepare data and add common derived columns."""
        if self._data_prepared:
            return
        
        df = self._raw_data_df.copy()
        
        if '$return' not in df.columns:
            df['$return'] = df.groupby('instrument')['$close'].transform(
                lambda x: x / x.shift(1) - 1
            )
        
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            logger.warning(f"Data has {dup_count} duplicate index entries, deduplicated")
            df = df[~df.index.duplicated(keep='last')]
        
        self._raw_data_df = df
        self._data_prepared = True
        logger.debug(f"Data prepared: {len(df)} rows, cols: {list(df.columns)}")
    
    def _get_cache_key(self, expr: str) -> str:
        """Cache key from expression MD5 hash."""
        return hashlib.md5(expr.encode()).hexdigest()

    def _get_market_instruments(self) -> Optional[frozenset]:
        """Lazily load the set of instrument codes for the configured market (e.g. CSI300).
        
        Filters factor Series to this set immediately after loading, reducing RAM ~20× when
        the cached H5 files cover the full A-share universe (~6000 stocks) but the backtest
        only needs the market subset (~300 stocks).
        """
        if self._market_instruments is not None:
            return self._market_instruments
        try:
            from qlib.data import D
            inst_list = D.list_instruments(D.instruments(self._market), as_list=True)
            if inst_list:
                self._market_instruments = frozenset(str(i).upper() for i in inst_list)
                logger.debug(f"Loaded {len(self._market_instruments)} instruments for market '{self._market}'")
                return self._market_instruments
        except Exception as e:
            logger.debug(f"Could not load market instruments from qlib ({e}); skipping universe filter")
        return None

    def _filter_to_instruments(self, series: pd.Series, instruments: frozenset) -> pd.Series:
        """Keep only rows whose instrument-level value is in `instruments`."""
        if not isinstance(series.index, pd.MultiIndex):
            return series
        try:
            inst_vals = series.index.get_level_values('instrument').astype(str).str.upper()
        except KeyError:
            # Level might be positional
            inst_vals = series.index.get_level_values(1).astype(str).str.upper()
        mask = inst_vals.isin(instruments)
        return series.loc[mask]

    def _load_factor_h5_fast(self, h5_path: str,
                              instruments: Optional[frozenset] = None) -> Optional[pd.Series]:
        """Load a factor result.h5 via h5py, optionally filtering to `instruments`.
        
        Avoids constructing a full 14 M-row pandas MultiIndex when only ~300 rows per day
        are needed, cutting load time from minutes to sub-second.
        """
        try:
            import h5py  # type: ignore
        except ImportError:
            return None
        try:
            with h5py.File(h5_path, "r") as fh:
                keys = list(fh.keys())
                if not keys:
                    return None
                grp = fh[keys[0]]
                if "values" not in grp:
                    return None
                raw = grp["values"][:]             # float64, shape (N,)
                dates_ns = grp["index_level0"][:]  # int64 nanoseconds
                inst_bytes = grp["index_level1"][:]
                label_dates = grp["index_label0"][:]  # int16/int32 indices
                label_insts = grp["index_label1"][:]

            all_insts = np.array([b.decode() for b in inst_bytes])

            if instruments is not None:
                inst_upper = np.char.upper(all_insts)
                in_market = np.isin(inst_upper, list(instruments))
                row_mask = in_market[label_insts]
            else:
                row_mask = np.ones(len(raw), dtype=bool)

            if not row_mask.any():
                return None

            raw_f = raw[row_mask]
            ld = label_dates[row_mask]
            li = label_insts[row_mask]

            dt_vals = pd.to_datetime(dates_ns[ld], unit="ns")
            inst_vals = all_insts[li]
            idx = pd.MultiIndex.from_arrays([dt_vals, inst_vals],
                                            names=["datetime", "instrument"])
            return pd.Series(raw_f, index=idx, dtype=FACTOR_DTYPE).sort_index()
        except Exception as exc:
            logger.debug("_load_factor_h5_fast failed for %s: %s", h5_path, exc)
            return None

    def _load_from_cache(self, expr: str) -> Optional[pd.Series]:
        """Load factor values from MD5 cache. Skipped for non-CN markets (stale data risk)."""
        if self._skip_market_cache:
            return None
        cache_key = self._get_cache_key(expr)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                result = pd.read_pickle(cache_file)
                return self._process_cached_result(result, cache_key)
            except Exception as e:
                logger.debug(f"Cache load failed [{cache_key}]: {e}")
                return None
        return None
    
    def _load_from_cache_location(self, cache_location: Dict) -> Optional[pd.Series]:
        """Load factor from path given in cache_location. Skipped for non-CN markets."""
        if self._skip_market_cache:
            return None
        if not cache_location:
            return None

        result_h5_path = cache_location.get('result_h5_path', '')
        if not result_h5_path:
            return None

        h5_file = Path(result_h5_path)
        if not h5_file.exists():
            logger.debug(f"Cache file not found: {result_h5_path}")
            return None

        # Fast path: h5py with immediate instrument filter (avoids 14 M-row MultiIndex)
        instruments = self._get_market_instruments()
        result = self._load_factor_h5_fast(str(h5_file), instruments)
        if result is not None:
            return result

        # Fallback: pandas read_hdf then filter
        try:
            result = pd.read_hdf(str(h5_file))
            result = self._process_cached_result(result, result_h5_path)
            if result is not None and instruments is not None:
                result = self._filter_to_instruments(result, instruments)
            return result
        except Exception as e:
            logger.debug(f"Load from cache_location failed [{result_h5_path}]: {e}")
            return None
    
    def _process_cached_result(self, result: Any, source: str) -> Optional[pd.Series]:
        """Normalize cached result format (does not touch self.data_df to avoid lazy load)."""
        try:
            if isinstance(result, pd.DataFrame):
                if len(result.columns) == 1:
                    result = result.iloc[:, 0]
                elif 'factor' in result.columns:
                    result = result['factor']
                else:
                    result = result.iloc[:, 0]
            
            # Standard order: (datetime, instrument)
            if isinstance(result.index, pd.MultiIndex):
                cache_idx_names = list(result.index.names)
                expected_order = ['datetime', 'instrument']
                if cache_idx_names != expected_order and set(cache_idx_names) == set(expected_order):
                    result = result.swaplevel()
                    result = result.sort_index()

            result = result.astype(FACTOR_DTYPE)
            # Filter to market universe immediately to reduce accumulated RAM
            instruments = self._get_market_instruments()
            if instruments is not None:
                result = self._filter_to_instruments(result, instruments)
            return result
        except Exception as e:
            logger.debug(f"Process cached result failed [{source}]: {e}")
            return None
    
    def _save_to_cache(self, expr: str, result: pd.Series):
        """Save factor values to cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self._get_cache_key(expr)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            result.to_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Save to cache failed: {e}")
    
    def _auto_extract_cache_from_logs(self):
        """Auto-extract cache from main program logs; runs once on first need."""
        if self._cache_extracted:
            return
        
        self._cache_extracted = True
        
        try:
            from tools.factor_cache_extractor import extract_factors_to_cache
            
            logger.debug("Auto-extracting main program cache...")
            new_count = extract_factors_to_cache(
                output_dir=self.cache_dir,
                verbose=False
            )
            if new_count > 0:
                logger.debug(f"Extracted {new_count} factors to cache")
        except ImportError:
            logger.debug("Cache extractor not available, skip auto-extract")
        except Exception as e:
            logger.warning(f"Auto-extract cache failed: {e}")
        
    def calculate_factor(self, factor_name: str, factor_expression: str) -> Optional[pd.Series]:
        """
        Compute a single factor.
        Returns: pd.Series with MultiIndex (datetime, instrument).
        """
        try:
            import io
            import sys as _sys
            from joblib import parallel_backend
            
            from quantaalpha.factors.coder.expr_parser import (
                parse_expression, parse_symbol
            )
            import quantaalpha.factors.coder.function_lib as func_lib
            
            df = self.data_df.copy()
            
            expr = parse_symbol(factor_expression, df.columns)
            
            old_stdout = _sys.stdout
            _sys.stdout = io.StringIO()
            try:
                expr = parse_expression(expr)
            finally:
                _sys.stdout = old_stdout
            
            for col in df.columns:
                col_name = str(col)
                base_name = col_name[1:] if col_name.startswith('$') else col_name
                expr = re.sub(
                    rf"(?<![A-Za-z0-9_\$])\$?{re.escape(base_name)}(?![A-Za-z0-9_])",
                    f"df[{col_name!r}]",
                    expr,
                )
            
            exec_globals = {
                'df': df,
                'np': np,
                'pd': pd,
            }
            
            for name in dir(func_lib):
                if not name.startswith('_'):
                    obj = getattr(func_lib, name)
                    if callable(obj):
                        exec_globals[name] = obj
            
            with parallel_backend('threading', n_jobs=1):
                result = eval(expr, exec_globals)
            
            if isinstance(result, pd.DataFrame):
                result = result.iloc[:, 0]
            
            if isinstance(result, pd.Series):
                result.name = factor_name
                # Align result index with raw data (duplicate-safe)
                if not result.index.equals(df.index):
                    try:
                        if result.index.duplicated().any():
                            result = result[~result.index.duplicated(keep='last')]
                        result = result.reindex(df.index)
                    except Exception:
                        logger.debug(f"reindex fallback for [{factor_name}]")
                        result = result[~result.index.duplicated(keep='last')]
                        clean_idx = df.index[~df.index.duplicated(keep='last')]
                        result = result.reindex(clean_idx)
                return result.astype(FACTOR_DTYPE)
            else:
                return pd.Series(result, index=df.index, name=factor_name).astype(FACTOR_DTYPE)
                
        except Exception as e:
            logger.warning(f"Factor computation failed [{factor_name}]: {str(e)[:200]}")
            return None
    
    def calculate_factors_from_json(self, json_path: str, 
                                   max_factors: Optional[int] = None) -> pd.DataFrame:
        """Batch compute factors from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        factors = data.get('factors', {})
        
        results = {}
        success_count = 0
        fail_count = 0
        
        factor_items = list(factors.items())
        if max_factors:
            factor_items = factor_items[:max_factors]
        
        total = len(factor_items)
        logger.debug(f"Computing {total} factors...")
        
        for i, (factor_id, factor_info) in enumerate(factor_items):
            factor_name = factor_info.get('factor_name', factor_id)
            factor_expr = factor_info.get('factor_expression', '')
            
            if not factor_expr:
                fail_count += 1
                continue
            
            if (i + 1) % 10 == 0 or i == 0:
                logger.debug(f"  Progress: {i+1}/{total}")
            
            result = self.calculate_factor(factor_name, factor_expr)
            
            if result is not None:
                results[factor_name] = result
                success_count += 1
            else:
                fail_count += 1
        
        print(f"Factor computation done: success {success_count}, failed {fail_count}")
        
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    
    def calculate_factors_batch(self, factors: List[Dict], use_cache: bool = True,
                                skip_compute: bool = False) -> pd.DataFrame:
        """
        Streaming batch loader. Factors are loaded one at a time and their numpy values
        are appended to a column list; the Series objects are freed immediately.

        Memory profile: O(n_rows × n_cols × 2 bytes) — no accumulation of pandas
        MultiIndex objects, which saved ~12 GB for 338 factors vs the old dict approach.
        """
        import gc
        import time as _time
        import signal as _signal

        class _FactorTimeout(Exception):
            pass

        if use_cache and self.auto_extract_cache:
            self._auto_extract_cache_from_logs()

        instruments = self._get_market_instruments()

        # reference_index: use full H5 data index to avoid truncation when first
        # cached factor has a shorter date range than the training window.
        reference_index = None
        if self._raw_data_df is not None and isinstance(self._raw_data_df.index, pd.MultiIndex):
            reference_index = self._raw_data_df.index
            logger.debug(f"  reference_index from data_df: {len(reference_index)} rows, "
                         f"{reference_index.get_level_values(0).min()} to "
                         f"{reference_index.get_level_values(0).max()}")
        columns_data: List[np.ndarray] = []   # one 1-D float16 array per factor
        col_names: List[str] = []

        success_count = fail_count = cache_location_hit = cache_hit = compute_count = 0
        failed_names: List[str] = []
        need_compute_factors: List = []
        total = len(factors)

        def _accept(series: pd.Series, name: str) -> bool:
            """Extract aligned numpy values and append to columns_data. Frees no refs."""
            nonlocal reference_index
            if series is None or series.isna().all():
                return False
            try:
                if reference_index is None:
                    reference_index = series.index
                vals = series.reindex(reference_index).values.astype(FACTOR_DTYPE)
                columns_data.append(vals)
                col_names.append(name)
                return True
            except Exception as exc:
                logger.debug("_accept alignment failed for %s: %s", name, exc)
                return False

        # Pass 1: load from cache (h5py fast path, already filtered to market instruments)
        for i, factor_info in enumerate(factors):
            factor_name = factor_info.get('factor_name', 'unknown')
            factor_expr = factor_info.get('factor_expression', '')
            cache_location = factor_info.get('cache_location')

            if not factor_expr:
                fail_count += 1
                failed_names.append(factor_name)
                continue

            series = None

            if use_cache and cache_location:
                h5_path = cache_location.get('result_h5_path', '')
                if h5_path:
                    series = self._load_from_cache_location(cache_location)
                    if series is not None:
                        cache_location_hit += 1

            if series is None and use_cache:
                series = self._load_from_cache(factor_expr)
                if series is not None:
                    cache_hit += 1

            if series is not None:
                if _accept(series, factor_name):
                    success_count += 1
                    print(f"  [{i+1}/{total}] ✓ cache: {factor_name}")
                else:
                    fail_count += 1
                    failed_names.append(factor_name)
                del series  # free MultiIndex memory immediately
            else:
                need_compute_factors.append((i, factor_info))
                print(f"  [{i+1}/{total}] ⏳ Pending: {factor_name}")

            if (i + 1) % 50 == 0:
                gc.collect()

        # Pass 2: compute uncached factors
        if need_compute_factors:
            if skip_compute:
                skipped = [fi.get('factor_name', 'unknown') for _, fi in need_compute_factors]
                print(f"  Skipping {len(skipped)} uncached factors (skip_compute=True)")
                fail_count += len(skipped)
                failed_names.extend(skipped)
            else:
                print(f"  Computing {len(need_compute_factors)} factors from expressions...")
                for idx, (orig_i, factor_info) in enumerate(need_compute_factors):
                    factor_name = factor_info.get('factor_name', 'unknown')
                    factor_expr = factor_info.get('factor_expression', '')
                    print(f"  Compute [{idx+1}/{len(need_compute_factors)}]: {factor_name} ...",
                          end='', flush=True)
                    t0 = _time.time()

                    old_handler = None
                    try:
                        def _timeout_handler(s, f):
                            raise _FactorTimeout()
                        old_handler = _signal.signal(_signal.SIGALRM, _timeout_handler)
                        _signal.alarm(120)
                    except (AttributeError, ValueError):
                        pass

                    try:
                        result = self.calculate_factor(factor_name, factor_expr)
                    except _FactorTimeout:
                        print(f" ✗ Timeout ({_time.time()-t0:.1f}s)")
                        fail_count += 1
                        failed_names.append(f"{factor_name}(timeout)")
                        result = None
                    except Exception as e:
                        print(f" ✗ Error ({_time.time()-t0:.1f}s): {str(e)[:80]}")
                        fail_count += 1
                        failed_names.append(factor_name)
                        result = None
                    finally:
                        try:
                            _signal.alarm(0)
                            if old_handler is not None:
                                _signal.signal(_signal.SIGALRM, old_handler)
                        except (AttributeError, ValueError):
                            pass

                    if result is None:
                        continue

                    elapsed = _time.time() - t0
                    if len(result) > 0 and not result.isna().all():
                        if instruments is not None:
                            result = self._filter_to_instruments(result, instruments)
                        result = result.astype(FACTOR_DTYPE)
                        if _accept(result, factor_name):
                            success_count += 1
                            compute_count += 1
                            print(f" ✓ ({elapsed:.1f}s)")
                            if use_cache:
                                self._save_to_cache(factor_expr, result)
                        else:
                            fail_count += 1
                            failed_names.append(factor_name)
                            print(f" ✗ NaN ({elapsed:.1f}s)")
                        del result
                    else:
                        fail_count += 1
                        failed_names.append(factor_name)
                        print(f" ✗ Failed ({_time.time()-t0:.1f}s)")

        print(f"Factor load done: success {success_count}, failed {fail_count} | "
              f"H5 cache {cache_location_hit}, MD5 cache {cache_hit}, computed {compute_count}")
        if failed_names:
            print(f"  Failed: {', '.join(failed_names)}")

        if not columns_data:
            return pd.DataFrame()

        gc.collect()
        # np.column_stack: one allocation of (n_rows × n_cols), then free the list
        data_matrix = np.column_stack(columns_data).astype(FACTOR_DTYPE)
        del columns_data
        gc.collect()
        result_df = pd.DataFrame(data_matrix, index=reference_index, columns=col_names)
        del data_matrix
        logger.debug("  Streaming result DataFrame: %s", result_df.shape)
        return result_df
    
    def _validate_and_align_result(self, result: pd.Series, factor_name: str, 
                                    reference_index: Optional[pd.Index] = None) -> Optional[pd.Series]:
        """Validate and align cached result index."""
        if result is None:
            return None
        
        target_idx = reference_index
        if target_idx is None:
            try:
                target_idx = self.data_df.index
            except Exception:
                return result if len(result) > 0 and not result.isna().all() else None
        
        # Align index (duplicate-safe)
        if not result.index.equals(target_idx):
            try:
                if result.index.duplicated().any():
                    result = result[~result.index.duplicated(keep='last')]
                if target_idx.duplicated().any():
                    target_idx = target_idx[~target_idx.duplicated(keep='last')]
                
                common_idx = result.index.intersection(target_idx)
                if len(common_idx) > len(target_idx) * 0.5:
                    result = result.reindex(target_idx)
                    logger.debug(f"    Index align: common {len(common_idx)}, target {len(target_idx)}")
                else:
                    logger.warning(f"    Cache index match rate too low ({len(common_idx)}/{len(target_idx)}), will recompute")
                    return None
            except Exception as e:
                logger.warning(f"    Index align failed: {e}, will recompute")
                return None
        
        # Validate data
        if result is None or len(result) == 0 or result.isna().all():
            return None
        
        return result


class CustomFactorDataLoader:
    """
    Converts computed factor values to Qlib-compatible format.
    """
    
    def __init__(self, factor_df: pd.DataFrame, label_expr: str = "Ref($close, -2) / Ref($close, -1) - 1"):
        self.factor_df = factor_df
        self.label_expr = label_expr
        
    def to_qlib_format(self, data_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert to Qlib data format."""
        from quantaalpha.factors.coder.expr_parser import (
            parse_expression, parse_symbol
        )
        import quantaalpha.factors.coder.function_lib as func_lib
        
        df = data_df.copy()
        
        expr = parse_symbol(self.label_expr, df.columns)
        expr = parse_expression(expr)
        
        for col in df.columns:
            col_name = str(col)
            base_name = col_name[1:] if col_name.startswith('$') else col_name
            expr = re.sub(
                rf"(?<![A-Za-z0-9_\$])\$?{re.escape(base_name)}(?![A-Za-z0-9_])",
                f"df[{col_name!r}]",
                expr,
            )
        
        exec_globals = {'df': df, 'np': np, 'pd': pd}
        for name in dir(func_lib):
            if not name.startswith('_'):
                obj = getattr(func_lib, name)
                if callable(obj):
                    exec_globals[name] = obj
        
        label = eval(expr, exec_globals)
        if isinstance(label, pd.DataFrame):
            label = label.iloc[:, 0]
        
        labels_df = pd.DataFrame({'LABEL0': label})
        
        return self.factor_df, labels_df


def get_qlib_stock_data(config: Dict) -> pd.DataFrame:
    """Load stock data from Qlib."""
    import qlib
    from qlib.data import D
    
    data_config = config.get('data', {})
    
    # Config-file provider_uri wins; fall back to env vars then default
    provider_uri = (
        data_config.get('provider_uri')
        or os.environ.get('QLIB_DATA_DIR')
        or os.environ.get('QLIB_PROVIDER_URI')
        or os.path.expanduser('~/.qlib/qlib_data/cn_data')
    )
    provider_uri = os.path.expanduser(provider_uri)
    region = data_config.get('region', 'cn')
    
    try:
        qlib.init(provider_uri=provider_uri, region=region)
    except Exception:
        pass  # Already initialized
    
    start_time = data_config.get('start_time', '2016-01-01')
    end_time = data_config.get('end_time', '2025-12-31')
    market = data_config.get('market', 'csi300')
    
    stock_list = D.instruments(market)
    
    fields = ['$open', '$high', '$low', '$close', '$volume', '$vwap']
    df = D.features(
        stock_list,
        fields,
        start_time=start_time,
        end_time=end_time,
        freq='day'
    )
    
    df.columns = fields
    
    logger.debug(f"Loaded stock data: {len(df)} rows")
    
    return df


if __name__ == '__main__':
    """Test factor computation."""
    import yaml
    
    logging.basicConfig(level=logging.INFO)
    
    _project_root = Path(__file__).resolve().parents[2]
    config_path = _project_root / 'configs' / 'backtest.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("Loading stock data...")
    data_df = get_qlib_stock_data(config)
    
    calculator = CustomFactorCalculator(data_df)
    
    test_expr = "RANK(-1 * TS_PCTCHANGE($close, 10))"
    print(f"\nTest expression: {test_expr}")
    
    result = calculator.calculate_factor("test_factor", test_expr)
    if result is not None:
        print(f"Success! Result shape: {result.shape}")
        print(result.head())
    else:
        print("Failed!")
