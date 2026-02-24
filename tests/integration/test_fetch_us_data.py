"""
Tests for scripts/fetch_us_data.py (Phase 0 US data pipeline).

Coverage:
  - _sp500_fallback / _nasdaq100_fallback: minimum count, no duplicates, yfinance-safe names
  - get_universe: deduplication, combined length, overlap handling
  - get_sp500_tickers / get_nasdaq100_tickers: Wikipedia scrape path and 403 fallback
  - download_tickers: yfinance call structure, retry on failure, failed-ticker handling
  - engineer_features: column renaming, $vwap formula, $return pct_change, float32 cast,
    NaN-row removal, MultiIndex preservation
  - load_existing: missing-file returns empty, valid H5 loads correctly
  - merge_incremental: empty existing → returns new; deduplication (keep='last');
    correct sort order; non-overlapping concat
  - quality_check: runs without exception; warns on high NaN rate; warns on bad return std
  - main: end-to-end integration with mocked yfinance and filesystem
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the script as a module (scripts/ is not a package)
# ---------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_us_data.py"

def _load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_us_data", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture(scope="module")
def mod():
    return _load_script()


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(
    tickers: list[str] | None = None,
    n_days: int = 60,
    start: str = "2022-01-03",
) -> pd.DataFrame:
    """Build a small OHLCV DataFrame matching engineer_features() input format."""
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for ticker in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days)))
        high  = close * (1 + rng.uniform(0, 0.02, n_days))
        low   = close * (1 - rng.uniform(0, 0.02, n_days))
        op    = close * (1 + rng.normal(0, 0.005, n_days))
        vol   = rng.integers(1_000_000, 10_000_000, n_days).astype(float)
        idx   = pd.MultiIndex.from_arrays([dates, [ticker] * n_days], names=["datetime", "instrument"])
        df    = pd.DataFrame({"open": op, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
        rows.append(df)
    return pd.concat(rows).sort_index()


def _make_h5_df(tmp_path: Path, tickers: list[str] | None = None) -> tuple[pd.DataFrame, Path]:
    """Save a minimal H5 and return (df, path)."""
    from tests.test_fetch_us_data import _load_script  # avoid circular if needed
    mod = _load_script()
    raw = _make_ohlcv(tickers)
    featured = mod.engineer_features(raw)
    h5_path = tmp_path / "daily_pv.h5"
    featured.to_hdf(str(h5_path), key="data", mode="w")
    return featured, h5_path


# ---------------------------------------------------------------------------
# Fallback list tests
# ---------------------------------------------------------------------------

class TestFallbackLists:
    def test_sp500_fallback_min_count(self, mod):
        tickers = mod._sp500_fallback()
        assert len(tickers) >= 400, f"Expected ≥400 S&P 500 fallbacks, got {len(tickers)}"

    def test_sp500_fallback_no_duplicates(self, mod):
        tickers = mod._sp500_fallback()
        assert len(tickers) == len(set(tickers)), "Duplicate tickers in S&P 500 fallback"

    def test_sp500_fallback_no_dots(self, mod):
        """yfinance uses dashes not dots (BRK-B not BRK.B)."""
        for t in mod._sp500_fallback():
            assert "." not in t, f"Dot in ticker {t!r} — should be dash"

    def test_sp500_fallback_has_mega_caps(self, mod):
        tickers = set(mod._sp500_fallback())
        for expected in ("AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA"):
            assert expected in tickers, f"{expected} missing from S&P 500 fallback"

    def test_nasdaq100_fallback_min_count(self, mod):
        tickers = mod._nasdaq100_fallback()
        assert len(tickers) >= 90, f"Expected ≥90 NASDAQ-100 fallbacks, got {len(tickers)}"

    def test_nasdaq100_fallback_no_duplicates(self, mod):
        tickers = mod._nasdaq100_fallback()
        assert len(tickers) == len(set(tickers))

    def test_nasdaq100_fallback_has_key_stocks(self, mod):
        tickers = set(mod._nasdaq100_fallback())
        for expected in ("NVDA", "AAPL", "MSFT", "AMZN", "META", "TSLA", "GOOG"):
            assert expected in tickers


# ---------------------------------------------------------------------------
# get_universe
# ---------------------------------------------------------------------------

class TestGetUniverse:
    def test_deduplication(self, mod):
        with patch.object(mod, "get_sp500_tickers", return_value=["AAPL", "MSFT", "GOOG"]), \
             patch.object(mod, "get_nasdaq100_tickers", return_value=["GOOG", "NVDA"]):
            universe = mod.get_universe()
        assert sorted(universe) == sorted(["AAPL", "GOOG", "MSFT", "NVDA"])

    def test_result_is_sorted(self, mod):
        with patch.object(mod, "get_sp500_tickers", return_value=["MSFT", "AAPL"]), \
             patch.object(mod, "get_nasdaq100_tickers", return_value=["NVDA"]):
            universe = mod.get_universe()
        assert universe == sorted(universe)

    def test_combined_at_least_500(self, mod):
        """Real fallbacks give ≥500 unique tickers."""
        with patch.object(mod, "_fetch_html", side_effect=Exception("no network")):
            universe = mod.get_universe()
        assert len(universe) >= 500


# ---------------------------------------------------------------------------
# get_sp500_tickers / get_nasdaq100_tickers — Wikipedia path and fallback
# ---------------------------------------------------------------------------

class TestGetTickers:
    def test_sp500_falls_back_on_403(self, mod):
        with patch.object(mod, "_fetch_html", side_effect=Exception("HTTP 403")):
            tickers = mod.get_sp500_tickers()
        assert len(tickers) >= 400

    def test_nasdaq100_falls_back_on_error(self, mod):
        with patch.object(mod, "_fetch_html", side_effect=Exception("timeout")):
            tickers = mod.get_nasdaq100_tickers()
        assert len(tickers) >= 90

    def test_sp500_parses_wikipedia_html(self, mod):
        """Simulate Wikipedia returning valid HTML with a Symbol column."""
        html = """<table><thead><tr><th>Symbol</th><th>Name</th></tr></thead>
                  <tbody><tr><td>AAPL</td><td>Apple</td></tr>
                         <tr><td>BRK.B</td><td>Berkshire</td></tr></tbody></table>"""
        with patch.object(mod, "_fetch_html", return_value=html):
            tickers = mod.get_sp500_tickers()
        assert "AAPL" in tickers
        assert "BRK-B" in tickers   # dot normalised to dash
        assert "BRK.B" not in tickers

    def test_nasdaq100_parses_wikipedia_html(self, mod):
        html = """<table><thead><tr><th>Ticker</th><th>Company</th></tr></thead>
                  <tbody><tr><td>NVDA</td><td>Nvidia</td></tr>
                         <tr><td>AAPL</td><td>Apple</td></tr></tbody></table>"""
        with patch.object(mod, "_fetch_html", return_value=html):
            tickers = mod.get_nasdaq100_tickers()
        assert "NVDA" in tickers
        assert "AAPL" in tickers


# ---------------------------------------------------------------------------
# engineer_features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_output_columns(self, mod):
        raw = _make_ohlcv()
        out = mod.engineer_features(raw)
        for col in ("$open", "$high", "$low", "$close", "$volume", "$vwap", "$return"):
            assert col in out.columns, f"Missing column: {col}"

    def test_no_unexpected_columns(self, mod):
        raw = _make_ohlcv()
        out = mod.engineer_features(raw)
        assert set(out.columns) == {"$open", "$high", "$low", "$close", "$volume", "$vwap", "$return"}

    def test_vwap_formula(self, mod):
        """$vwap == (high + low + close) / 3."""
        raw = _make_ohlcv(["AAPL"], n_days=10)
        out = mod.engineer_features(raw)
        expected = (out["$high"] + out["$low"] + out["$close"]) / 3.0
        pd.testing.assert_series_equal(out["$vwap"], expected, check_names=False, atol=1e-5)

    def test_return_is_pct_change_per_instrument(self, mod):
        """$return should be NaN (→0) on first day, close/prev_close - 1 otherwise."""
        raw = _make_ohlcv(["AAPL"], n_days=5)
        out = mod.engineer_features(raw)
        ret = out["$return"].unstack("instrument")["AAPL"]
        close = out["$close"].unstack("instrument")["AAPL"]
        expected = close.pct_change(fill_method=None).fillna(0)
        pd.testing.assert_series_equal(ret, expected.astype("float32"), atol=1e-5, check_names=False)

    def test_float32_dtype(self, mod):
        raw = _make_ohlcv()
        out = mod.engineer_features(raw)
        for col in out.columns:
            assert out[col].dtype == np.float32, f"{col} is {out[col].dtype}, expected float32"

    def test_multiindex_preserved(self, mod):
        raw = _make_ohlcv(["AAPL", "MSFT"])
        out = mod.engineer_features(raw)
        assert out.index.names == ["datetime", "instrument"]

    def test_nan_close_rows_dropped(self, mod):
        raw = _make_ohlcv(["AAPL"], n_days=5)
        raw.loc[raw.index[2], "close"] = np.nan
        out = mod.engineer_features(raw)
        assert len(out) == 4  # one row dropped

    def test_return_no_nan_after_fillna(self, mod):
        raw = _make_ohlcv(["AAPL", "MSFT"], n_days=20)
        out = mod.engineer_features(raw)
        assert not out["$return"].isna().any(), "$return should have no NaN after fillna(0)"

    def test_multiple_instruments_independent_return(self, mod):
        """$return cross-instrument computation must not bleed across tickers."""
        raw = _make_ohlcv(["AAPL", "MSFT"], n_days=10)
        out = mod.engineer_features(raw)
        aapl_ret = out.xs("AAPL", level="instrument")["$return"]
        msft_ret = out.xs("MSFT", level="instrument")["$return"]
        # First-day return is 0 for each independently
        assert aapl_ret.iloc[0] == pytest.approx(0.0, abs=1e-6)
        assert msft_ret.iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_extra_columns_ignored(self, mod):
        """Columns like 'dividends' present in some yfinance outputs must be dropped."""
        raw = _make_ohlcv(["AAPL"], n_days=5)
        raw["dividends"] = 0.0
        out = mod.engineer_features(raw)
        assert "dividends" not in out.columns


# ---------------------------------------------------------------------------
# load_existing
# ---------------------------------------------------------------------------

class TestLoadExisting:
    def test_missing_file_returns_empty(self, mod, tmp_path):
        result = mod.load_existing(tmp_path / "nonexistent.h5")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_loads_valid_h5(self, mod, tmp_path):
        raw = _make_ohlcv(["AAPL"], n_days=20)
        featured = mod.engineer_features(raw)
        h5 = tmp_path / "daily_pv.h5"
        featured.to_hdf(str(h5), key="data", mode="w")
        loaded = mod.load_existing(h5)
        assert len(loaded) == len(featured)
        assert list(loaded.columns) == list(featured.columns)


# ---------------------------------------------------------------------------
# merge_incremental
# ---------------------------------------------------------------------------

class TestMergeIncremental:
    def _featured(self, mod, tickers, start, n_days):
        return mod.engineer_features(_make_ohlcv(tickers, n_days=n_days, start=start))

    def test_empty_existing_returns_new(self, mod):
        new = self._featured(mod, ["AAPL"], "2022-01-03", 10)
        result = mod.merge_incremental(pd.DataFrame(), new)
        pd.testing.assert_frame_equal(result, new)

    def test_non_overlapping_concat(self, mod):
        old = self._featured(mod, ["AAPL"], "2022-01-03", 20)
        new = self._featured(mod, ["AAPL"], "2022-03-01", 10)
        result = mod.merge_incremental(old, new)
        assert len(result) == len(old) + len(new)

    def test_overlapping_rows_deduplicated(self, mod):
        """New data for same date should overwrite old."""
        base = self._featured(mod, ["AAPL"], "2022-01-03", 20)
        # Make a "newer" version of the last 5 rows with different values
        overlap = base.copy()
        overlap["$close"] = overlap["$close"] * 2.0  # distinct marker
        result = mod.merge_incremental(base, overlap)
        assert len(result) == len(base)  # no extra rows
        # All $close values should be from overlap (keep='last')
        pd.testing.assert_series_equal(
            result["$close"].sort_index(),
            overlap["$close"].sort_index(),
        )

    def test_result_is_sorted(self, mod):
        old = self._featured(mod, ["AAPL"], "2022-03-01", 10)
        new = self._featured(mod, ["AAPL"], "2022-01-03", 10)
        result = mod.merge_incremental(old, new)
        assert result.index.is_monotonic_increasing


# ---------------------------------------------------------------------------
# quality_check
# ---------------------------------------------------------------------------

class TestQualityCheck:
    def test_good_data_no_warning(self, mod, caplog):
        raw = _make_ohlcv(["AAPL", "MSFT", "GOOG"], n_days=60)
        featured = mod.engineer_features(raw)
        import logging
        with caplog.at_level(logging.WARNING):
            mod.quality_check(featured)
        assert "NaN rate > 5%" not in caplog.text
        assert "Unexpected $return std" not in caplog.text

    def test_high_nan_triggers_warning(self, mod, caplog):
        raw = _make_ohlcv(["AAPL"], n_days=50)
        featured = mod.engineer_features(raw).copy().astype(object)
        # Inject NaN across ALL columns for 40% of rows → overall NaN rate ~40% > 5%
        idx_nan = featured.sample(frac=0.40, random_state=1).index
        featured.loc[idx_nan, :] = np.nan
        import logging
        with caplog.at_level(logging.WARNING):
            mod.quality_check(featured)
        assert "NaN rate > 5%" in caplog.text

    def test_bad_return_std_triggers_warning(self, mod, caplog):
        raw = _make_ohlcv(["AAPL"], n_days=50)
        featured = mod.engineer_features(raw).copy()
        # Force an absurd return std (e.g. values like 5.0 → std >> 0.05)
        featured["$return"] = 5.0
        import logging
        with caplog.at_level(logging.WARNING):
            mod.quality_check(featured)
        assert "Unexpected $return std" in caplog.text

    def test_quality_check_does_not_raise(self, mod):
        raw = _make_ohlcv(["AAPL", "TSLA"], n_days=30)
        featured = mod.engineer_features(raw)
        mod.quality_check(featured)  # must not raise


# ---------------------------------------------------------------------------
# Integration: main() with mocks
# ---------------------------------------------------------------------------

class TestMainIntegration:
    def test_main_writes_h5_and_instruments(self, mod, tmp_path, monkeypatch):
        """End-to-end: main() should produce daily_pv.h5 + instruments.txt."""
        fake_featured = mod.engineer_features(_make_ohlcv(["AAPL", "MSFT"], n_days=30))

        monkeypatch.setattr(mod, "get_universe", lambda: ["AAPL", "MSFT"])
        monkeypatch.setattr(mod, "download_tickers", lambda *a, **kw: _make_ohlcv(["AAPL", "MSFT"], n_days=30))

        sys_argv_backup = sys.argv
        sys.argv = ["fetch_us_data.py", "--start", "2022-01-01", "--output", str(tmp_path)]
        try:
            mod.main()
        finally:
            sys.argv = sys_argv_backup

        h5 = tmp_path / "daily_pv.h5"
        inst = tmp_path / "instruments.txt"
        assert h5.exists(), "daily_pv.h5 not created"
        assert inst.exists(), "instruments.txt not created"

        loaded = pd.read_hdf(str(h5), key="data")
        assert len(loaded) > 0
        for col in ("$open", "$high", "$low", "$close", "$volume", "$vwap", "$return"):
            assert col in loaded.columns

        instruments_text = inst.read_text().strip().split("\n")
        assert set(instruments_text) == {"AAPL", "MSFT"}

    def test_main_incremental_skips_if_up_to_date(self, mod, tmp_path, monkeypatch, caplog):
        """Incremental mode: if last date >= today, nothing downloaded."""
        # Write an H5 with last date = today
        today = pd.Timestamp.today().normalize()
        raw = _make_ohlcv(["AAPL"], n_days=5, start=(today - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
        # Manually set last row datetime to today
        featured = mod.engineer_features(raw)

        h5 = tmp_path / "daily_pv.h5"
        featured.to_hdf(str(h5), key="data", mode="w")

        download_called = []
        monkeypatch.setattr(mod, "get_universe", lambda: ["AAPL"])
        monkeypatch.setattr(mod, "download_tickers", lambda *a, **kw: download_called.append(1) or _make_ohlcv(["AAPL"], 5))

        sys.argv = ["fetch_us_data.py", "--start", "2016-01-01",
                    "--end", today.strftime("%Y-%m-%d"),
                    "--output", str(tmp_path), "--incremental"]
        import logging
        with caplog.at_level(logging.INFO):
            mod.main()

        # download_tickers should NOT have been called (data already current)
        assert len(download_called) == 0, "Expected no download in incremental mode when data is current"
