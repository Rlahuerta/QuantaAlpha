"""
signal_generator.py — Load a pre-trained LightGBM model and generate ranked
trading signals for the US universe.

Usage:
    from quantaalpha.live.signal_generator import SignalGenerator
    gen = SignalGenerator.from_meta("data/models/us_baseline_meta.json")
    scores = gen.generate(as_of_date="2025-01-15")
    # Returns: {"AAPL": 0.82, "MSFT": 0.79, ...}  (higher = stronger buy signal)

CLI:
    python -m quantaalpha.live.signal_generator --meta data/models/us_baseline_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


class SignalGenerator:
    """Compute factor values → LightGBM scores → ranked ticker dict."""

    def __init__(
        self,
        booster_path: str | Path,
        feature_cols: List[str],
        factor_json_files: List[str | Path],
        h5_path: str | Path,
        cache_dir: Optional[str | Path] = None,
        lookback_days: int = 120,
    ):
        """
        Args:
            booster_path:    Path to the LightGBM .txt booster file.
            feature_cols:    Ordered list of factor names (must match model training order).
            factor_json_files: Paths to the factor library JSON files used during training.
            h5_path:         Path to the US daily_pv.h5 data store.
            cache_dir:       Factor cache directory (default: data/results/factor_cache_us).
            lookback_days:   Days of history to load for factor computation (default 120).
        """
        self.booster_path = Path(booster_path)
        self.feature_cols = feature_cols
        self.factor_json_files = [Path(p) for p in (factor_json_files or [])]
        self.h5_path = Path(h5_path)
        self.cache_dir = Path(cache_dir) if cache_dir else (_REPO_ROOT / "data/results/factor_cache_us")
        self.lookback_days = lookback_days
        self._booster = None

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_meta(cls, meta_path: str | Path, **kwargs) -> "SignalGenerator":
        """Load a SignalGenerator from the metadata JSON saved by the runner."""
        meta = json.loads(Path(meta_path).read_text())
        return cls(
            booster_path=meta["booster_path"],
            feature_cols=meta["feature_cols"],
            factor_json_files=meta.get("factor_json", []),
            h5_path=meta.get("data_file", ""),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_booster(self):
        if self._booster is None:
            try:
                import lightgbm as lgb
            except ImportError:
                raise ImportError("lightgbm not installed — pip install lightgbm")
            self._booster = lgb.Booster(model_file=str(self.booster_path))
            log.info("Booster loaded from %s (%d features)", self.booster_path, self._booster.num_trees())

    def _load_h5_window(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Load the H5 store, sliced to the lookback window ending on as_of."""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 not found: {self.h5_path}")
        start = (as_of - pd.Timedelta(days=self.lookback_days * 1.6)).strftime("%Y-%m-%d")
        end = as_of.strftime("%Y-%m-%d")
        full = pd.read_hdf(str(self.h5_path), key="data")
        # Slice on datetime level of MultiIndex
        dts = full.index.get_level_values("datetime")
        mask = (dts >= start) & (dts <= end)
        return full.loc[mask]

    def _load_factors(self) -> List[Dict]:
        """Load factor definitions from all JSON files."""
        factors = {}
        for p in self.factor_json_files:
            if not p.exists():
                log.warning("Factor JSON not found: %s", p)
                continue
            lib = json.loads(p.read_text())
            for name, info in lib.get("factors", {}).items():
                if name not in factors:
                    factors[name] = info
        return [{"name": k, **v} for k, v in factors.items()]

    def _compute_factors(self, data_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        """Compute factor values for all instruments on as_of date."""
        from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator

        calc = CustomFactorCalculator(
            data_df=data_df,
            cache_dir=self.cache_dir,
            auto_extract_cache=False,
            config={"data": {"region": "us", "market": "sp500"}},
        )
        factor_defs = self._load_factors()
        if not factor_defs:
            raise ValueError("No factors loaded from factor_json_files")

        # Only compute factors we need (those in feature_cols)
        needed = set(self.feature_cols)
        factor_defs_filtered = [f for f in factor_defs if f["name"] in needed]
        log.info("Computing %d factors for %s...", len(factor_defs_filtered), as_of.date())

        factor_map = calc.calculate_factors_batch(factor_defs_filtered, use_cache=True)

        rows: Dict[str, Dict[str, float]] = {}
        as_of_str = as_of.strftime("%Y-%m-%d")
        for name, series in factor_map.items():
            if series is None:
                continue
            # Get values for as_of date only
            try:
                day_vals = series.xs(as_of_str, level="datetime", drop_level=True)
            except KeyError:
                continue
            for instrument, val in day_vals.items():
                if instrument not in rows:
                    rows[instrument] = {}
                rows[instrument][name] = float(val) if np.isfinite(val) else np.nan

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "instrument"
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        as_of_date: Optional[str | date] = None,
        min_valid_fraction: float = 0.5,
    ) -> Dict[str, float]:
        """Compute ranked trading signals for as_of_date.

        Args:
            as_of_date: Date to generate signals for. Defaults to today.
            min_valid_fraction: Drop instruments with fewer than this fraction
                                of non-NaN factor values. Default 0.5.

        Returns:
            Dict mapping instrument → raw model score (higher = stronger buy).
        """
        if as_of_date is None:
            as_of_date = date.today()
        as_of = pd.Timestamp(as_of_date)

        self._load_booster()

        data_df = self._load_h5_window(as_of)
        if data_df.empty:
            log.warning("No H5 data in window ending %s", as_of.date())
            return {}

        factors_df = self._compute_factors(data_df, as_of)
        if factors_df.empty:
            log.warning("Factor computation returned empty DataFrame")
            return {}

        # Align to training feature order, fill missing with 0
        aligned = factors_df.reindex(columns=self.feature_cols, fill_value=np.nan)

        # Drop instruments with too many NaN features
        valid_frac = aligned.notna().mean(axis=1)
        aligned = aligned.loc[valid_frac >= min_valid_fraction]
        if aligned.empty:
            log.warning("All instruments dropped by min_valid_fraction filter")
            return {}

        # Impute remaining NaNs with column median
        aligned = aligned.fillna(aligned.median())

        scores = self._booster.predict(aligned.values)
        result = dict(zip(aligned.index.tolist(), scores.tolist()))
        log.info("Generated signals for %d instruments on %s", len(result), as_of.date())
        return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate live trading signals")
    parser.add_argument("--meta", required=True, help="Path to model meta.json")
    parser.add_argument("--date", default=None, help="As-of date (YYYY-MM-DD, default: today)")
    parser.add_argument("--top-n", type=int, default=30, help="Print top N tickers by score")
    args = parser.parse_args()

    gen = SignalGenerator.from_meta(args.meta)
    scores = gen.generate(as_of_date=args.date)
    if not scores:
        print("No signals generated.")
        return
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print(f"Top-{args.top_n} signals for {args.date or date.today()}:")
    for rank, (ticker, score) in enumerate(ranked[: args.top_n], 1):
        print(f"  {rank:3d}. {ticker:<8s}  score={score:.4f}")


if __name__ == "__main__":
    main()
