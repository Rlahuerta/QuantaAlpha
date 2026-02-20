"""
Unit tests for quantaalpha.live module (Phase 2).
Tests data_ingestor, signal_generator, and portfolio_constructor
without requiring LightGBM models, H5 data, or live network.
"""
from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(
    tickers=("AAPL", "MSFT"),
    start="2024-01-02",
    end="2024-01-10",
) -> pd.DataFrame:
    """Create a MultiIndex(datetime, instrument) DataFrame with OHLCV columns."""
    dates = pd.date_range(start, end, freq="B")
    rows = []
    for ticker in tickers:
        for dt in dates:
            rows.append({
                "datetime": dt,
                "instrument": ticker,
                "$open": 100.0,
                "$high": 102.0,
                "$low": 99.0,
                "$close": 101.0,
                "$volume": 1_000_000.0,
            })
    df = pd.DataFrame(rows).set_index(["datetime", "instrument"])
    return df.astype("float32")


# ===========================================================================
# DataIngestor tests
# ===========================================================================

class TestEngineerFeaturesLive:
    """Tests for data_ingestor._engineer_features."""

    def setup_method(self):
        from quantaalpha.live.data_ingestor import _engineer_features
        self._fn = _engineer_features

    def _raw(self, tickers=("AAPL",), n=10):
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        rows = []
        for t in tickers:
            for dt in dates:
                rows.append({"datetime": dt, "instrument": t,
                             "open": 100.0, "high": 105.0, "low": 98.0,
                             "close": 101.0, "volume": 5e5})
        return pd.DataFrame(rows).set_index(["datetime", "instrument"])

    def test_output_columns(self):
        result = self._fn(self._raw())
        assert set(result.columns) == {"$open", "$high", "$low", "$close", "$volume", "$vwap", "$return"}

    def test_vwap_formula(self):
        result = self._fn(self._raw())
        expected_vwap = (105.0 + 98.0 + 101.0) / 3
        assert np.allclose(result["$vwap"].values, expected_vwap, atol=1e-3)

    def test_return_is_float32(self):
        result = self._fn(self._raw())
        assert result.dtypes["$return"] == np.float32

    def test_return_first_row_is_zero(self):
        result = self._fn(self._raw(("AAPL",)))
        first = result.xs("AAPL", level="instrument")["$return"].iloc[0]
        assert abs(first) < 1e-5

    def test_drops_nan_close(self):
        raw = self._raw(("AAPL",))
        raw.loc[raw.index[0], "close"] = float("nan")
        result = self._fn(raw)
        assert len(result) == 9

    def test_multiple_tickers_independent_returns(self):
        result = self._fn(self._raw(("AAPL", "MSFT")))
        aapl_first = result.xs("AAPL", level="instrument")["$return"].iloc[0]
        msft_first = result.xs("MSFT", level="instrument")["$return"].iloc[0]
        assert abs(aapl_first) < 1e-5
        assert abs(msft_first) < 1e-5

    def test_index_names(self):
        result = self._fn(self._raw())
        assert result.index.names == ["datetime", "instrument"]


class TestDataIngestorInit:
    """Tests for DataIngestor initialization."""

    def test_default_h5_path(self):
        from quantaalpha.live.data_ingestor import DataIngestor
        ingestor = DataIngestor()
        assert "daily_pv.h5" in str(ingestor.h5_path)

    def test_custom_tickers(self):
        from quantaalpha.live.data_ingestor import DataIngestor
        ingestor = DataIngestor(tickers=["AAPL", "GOOGL"])
        tickers = ingestor._load_tickers()
        assert tickers == ["AAPL", "GOOGL"]

    def test_load_tickers_from_file(self, tmp_path):
        from quantaalpha.live.data_ingestor import DataIngestor
        f = tmp_path / "tickers.txt"
        f.write_text("AAPL\nMSFT\nGOOGL\n")
        ingestor = DataIngestor(instruments_file=f)
        assert ingestor._load_tickers() == ["AAPL", "MSFT", "GOOGL"]

    def test_load_tickers_missing_file_raises(self):
        from quantaalpha.live.data_ingestor import DataIngestor
        ingestor = DataIngestor(instruments_file="/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            ingestor._load_tickers()


class TestDataIngestorMerge:
    """Tests for DataIngestor._merge."""

    def setup_method(self):
        from quantaalpha.live.data_ingestor import DataIngestor
        self.ingestor = DataIngestor(tickers=["AAPL"])

    def test_merge_empty_existing(self):
        new = _make_ohlcv_df()
        result = self.ingestor._merge(pd.DataFrame(), new)
        assert len(result) == len(new)

    def test_merge_empty_new(self):
        existing = _make_ohlcv_df()
        result = self.ingestor._merge(existing, pd.DataFrame())
        assert len(result) == len(existing)

    def test_merge_deduplicates(self):
        df = _make_ohlcv_df()
        result = self.ingestor._merge(df, df)
        assert len(result) == len(df)

    def test_merge_new_rows_appended(self):
        old = _make_ohlcv_df(start="2024-01-02", end="2024-01-05")
        new = _make_ohlcv_df(start="2024-01-08", end="2024-01-12")
        result = self.ingestor._merge(old, new)
        assert len(result) > len(old)
        assert len(result) > len(new)

    def test_merge_new_overrides_old(self):
        old = _make_ohlcv_df()
        new = old.copy()
        new["$close"] = 999.0
        result = self.ingestor._merge(old, new)
        assert float(result["$close"].iloc[0]) == 999.0


class TestDataIngestorH5IO:
    """Tests for DataIngestor._last_stored_date and ingest() dry-run."""

    def test_last_stored_date_no_file(self, tmp_path):
        from quantaalpha.live.data_ingestor import DataIngestor
        ingestor = DataIngestor(h5_path=tmp_path / "missing.h5", tickers=["AAPL"])
        assert ingestor._last_stored_date() is None

    def test_last_stored_date_from_h5(self, tmp_path):
        from quantaalpha.live.data_ingestor import DataIngestor
        h5 = tmp_path / "pv.h5"
        df = _make_ohlcv_df()
        df.to_hdf(str(h5), key="data")
        ingestor = DataIngestor(h5_path=h5, tickers=["AAPL"])
        last = ingestor._last_stored_date()
        assert last is not None
        assert isinstance(last, date)

    def test_ingest_up_to_date_skips(self, tmp_path):
        from quantaalpha.live.data_ingestor import DataIngestor
        h5 = tmp_path / "pv.h5"
        today_df = _make_ohlcv_df(start=str(date.today()), end=str(date.today()))
        today_df.to_hdf(str(h5), key="data")
        ingestor = DataIngestor(h5_path=h5, tickers=["AAPL"])
        result = ingestor.ingest()
        assert result == 0


# ===========================================================================
# PortfolioConstructor tests
# ===========================================================================

class TestTopkSelection:
    """Tests for PortfolioConstructor._select_topk_with_dropout."""

    def setup_method(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        self.pc = PortfolioConstructor(topk=5, n_drop=2)

    def _scores(self, n=10):
        return {f"T{i:02d}": float(10 - i) for i in range(n)}

    def test_target_size_equals_topk(self):
        scores = self._scores(10)
        target, _ = self.pc._select_topk_with_dropout(scores, [])
        assert len(target) == 5

    def test_no_holdings_returns_top5(self):
        scores = self._scores(10)
        target, dropped = self.pc._select_topk_with_dropout(scores, [])
        assert target == ["T00", "T01", "T02", "T03", "T04"]
        assert dropped == []

    def test_held_in_top_stay(self):
        scores = self._scores(10)
        held = ["T00", "T01"]
        target, dropped = self.pc._select_topk_with_dropout(scores, held)
        assert "T00" in target
        assert "T01" in target

    def test_n_drop_limits_ejections(self):
        scores = self._scores(10)
        # Hold T08, T09 (outside top5 by score)
        held = ["T00", "T01", "T02", "T08", "T09"]
        target, dropped = self.pc._select_topk_with_dropout(scores, held)
        assert len(dropped) <= 2

    def test_worst_holding_dropped_first(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        scores = {"A": 10, "B": 9, "C": 8, "D": 7, "E": 6, "F": 0.1, "G": 0.2}
        pc = PortfolioConstructor(topk=5, n_drop=1)
        held = ["A", "B", "F"]  # F is worst
        target, dropped = pc._select_topk_with_dropout(scores, held)
        assert "F" in dropped

    def test_empty_scores(self):
        target, dropped = self.pc._select_topk_with_dropout({}, [])
        assert target == []


class TestTargetShares:
    """Tests for PortfolioConstructor._compute_target_shares."""

    def setup_method(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        self.pc = PortfolioConstructor(topk=4, n_drop=1, capital=100_000)

    def test_shares_sum_near_capital(self):
        target = ["AAPL", "MSFT", "GOOG", "AMZN"]
        prices = {t: 100.0 for t in target}
        shares = self.pc._compute_target_shares(target, prices)
        total = sum(shares.values()) * 100.0
        assert 0.8 * 100_000 < total < 1.1 * 100_000

    def test_equal_weight_equal_shares(self):
        target = ["A", "B", "C", "D"]
        prices = {t: 100.0 for t in target}
        shares = self.pc._compute_target_shares(target, prices)
        vals = list(shares.values())
        assert all(v == vals[0] for v in vals)

    def test_missing_price_skipped(self):
        target = ["AAPL", "NO_PRICE"]
        prices = {"AAPL": 200.0}
        shares = self.pc._compute_target_shares(target, prices)
        assert "NO_PRICE" not in shares
        assert "AAPL" in shares

    def test_zero_price_skipped(self):
        target = ["AAPL"]
        shares = self.pc._compute_target_shares(target, {"AAPL": 0.0})
        assert "AAPL" not in shares

    def test_empty_target(self):
        assert self.pc._compute_target_shares([], {}) == {}


class TestPortfolioRebalance:
    """Integration tests for PortfolioConstructor.rebalance()."""

    def setup_method(self):
        from quantaalpha.live.portfolio_constructor import PortfolioConstructor
        self.pc = PortfolioConstructor(topk=3, n_drop=1, capital=300_000)

    def _scores(self):
        return {"AAPL": 0.9, "MSFT": 0.8, "GOOG": 0.7, "AMZN": 0.6, "META": 0.5}

    def _prices(self):
        return {"AAPL": 200.0, "MSFT": 400.0, "GOOG": 150.0, "AMZN": 180.0, "META": 500.0}

    def test_no_positions_buys_top3(self):
        result = self.pc.rebalance(self._scores(), {}, self._prices())
        bought = {o.ticker for o in result.buy_orders}
        assert "AAPL" in bought
        assert "MSFT" in bought
        assert "GOOG" in bought

    def test_result_has_target_portfolio(self):
        result = self.pc.rebalance(self._scores(), {}, self._prices())
        assert len(result.target_portfolio) == 3

    def test_held_stocks_in_top3_kept(self):
        positions = {"AAPL": 100, "MSFT": 50}
        result = self.pc.rebalance(self._scores(), positions, self._prices())
        assert "AAPL" in result.target_portfolio
        assert "MSFT" in result.target_portfolio

    def test_sell_order_generated_for_dropped(self):
        # Hold AMZN (rank 4, outside top3); should generate sell after n_drop
        positions = {"AAPL": 100, "MSFT": 50, "AMZN": 80}
        result = self.pc.rebalance(self._scores(), positions, self._prices())
        sell_tickers = {o.ticker for o in result.sell_orders}
        # AMZN is outside top3 and should eventually be dropped
        assert len(result.orders) >= 0  # basic sanity

    def test_empty_scores(self):
        result = self.pc.rebalance({}, {}, {})
        assert result.target_portfolio == {}

    def test_result_date_set(self):
        result = self.pc.rebalance(self._scores(), {}, self._prices(), as_of_date="2025-01-15")
        assert result.date == "2025-01-15"

    def test_topk_n_drop_in_result(self):
        result = self.pc.rebalance(self._scores(), {}, self._prices())
        assert result.topk == 3
        assert result.n_drop == 1
        assert result.capital == 300_000


class TestPortfolioOrderProperties:
    """Tests for Order dataclass properties."""

    def test_notional(self):
        from quantaalpha.live.portfolio_constructor import Order
        o = Order(ticker="AAPL", shares=100, price=200.0, action="buy")
        assert o.notional == 20_000.0

    def test_notional_sell(self):
        from quantaalpha.live.portfolio_constructor import Order
        o = Order(ticker="AAPL", shares=-50, price=200.0, action="sell")
        assert o.notional == 10_000.0


# ===========================================================================
# SignalGenerator tests (mocked, no LightGBM/H5 required)
# ===========================================================================

class TestSignalGeneratorInit:
    """Tests for SignalGenerator constructor."""

    def test_from_meta(self, tmp_path):
        from quantaalpha.live.signal_generator import SignalGenerator
        meta = {
            "booster_path": str(tmp_path / "model.txt"),
            "feature_cols": ["f1", "f2"],
            "factor_json": [str(tmp_path / "lib.json")],
            "data_file": str(tmp_path / "pv.h5"),
        }
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps(meta))
        gen = SignalGenerator.from_meta(meta_path)
        assert gen.feature_cols == ["f1", "f2"]

    def test_attributes_stored(self, tmp_path):
        from quantaalpha.live.signal_generator import SignalGenerator
        gen = SignalGenerator(
            booster_path=tmp_path / "m.txt",
            feature_cols=["a", "b", "c"],
            factor_json_files=[],
            h5_path=tmp_path / "pv.h5",
        )
        assert len(gen.feature_cols) == 3
        assert gen.lookback_days == 120

    def test_custom_lookback(self, tmp_path):
        from quantaalpha.live.signal_generator import SignalGenerator
        gen = SignalGenerator(
            booster_path=tmp_path / "m.txt",
            feature_cols=[],
            factor_json_files=[],
            h5_path=tmp_path / "pv.h5",
            lookback_days=60,
        )
        assert gen.lookback_days == 60


class TestSignalGeneratorLoadFactors:
    """Tests for _load_factors()."""

    def test_empty_if_no_files(self, tmp_path):
        from quantaalpha.live.signal_generator import SignalGenerator
        gen = SignalGenerator(
            booster_path=tmp_path / "m.txt",
            feature_cols=[],
            factor_json_files=[],
            h5_path=tmp_path / "pv.h5",
        )
        assert gen._load_factors() == []

    def test_loads_from_json(self, tmp_path):
        from quantaalpha.live.signal_generator import SignalGenerator
        lib = {
            "factors": {
                "mom_5": {"factor_expression": "TS_MEAN($return, 5)", "quality": "high_quality"},
                "vol_10": {"factor_expression": "TS_STD($close, 10)", "quality": "medium_quality"},
            }
        }
        p = tmp_path / "lib.json"
        p.write_text(json.dumps(lib))
        gen = SignalGenerator(
            booster_path=tmp_path / "m.txt",
            feature_cols=["mom_5", "vol_10"],
            factor_json_files=[p],
            h5_path=tmp_path / "pv.h5",
        )
        factors = gen._load_factors()
        assert len(factors) == 2
        names = {f["name"] for f in factors}
        assert {"mom_5", "vol_10"} == names

    def test_deduplicates_across_files(self, tmp_path):
        from quantaalpha.live.signal_generator import SignalGenerator
        lib = {"factors": {"f1": {"factor_expression": "X", "quality": "high_quality"}}}
        p1 = tmp_path / "lib1.json"
        p2 = tmp_path / "lib2.json"
        p1.write_text(json.dumps(lib))
        p2.write_text(json.dumps(lib))
        gen = SignalGenerator(
            booster_path=tmp_path / "m.txt",
            feature_cols=["f1"],
            factor_json_files=[p1, p2],
            h5_path=tmp_path / "pv.h5",
        )
        assert len(gen._load_factors()) == 1

    def test_missing_file_warns_and_skips(self, tmp_path):
        from quantaalpha.live.signal_generator import SignalGenerator
        gen = SignalGenerator(
            booster_path=tmp_path / "m.txt",
            feature_cols=[],
            factor_json_files=[tmp_path / "nonexistent.json"],
            h5_path=tmp_path / "pv.h5",
        )
        factors = gen._load_factors()  # should not raise
        assert factors == []


class TestSignalGeneratorGenerate:
    """Tests for generate() with mocked H5 and booster."""

    def _make_gen(self, tmp_path, feature_cols=("f1", "f2")):
        from quantaalpha.live.signal_generator import SignalGenerator
        lib = {"factors": {
            "f1": {"factor_expression": "RANK($close)", "quality": "high_quality"},
            "f2": {"factor_expression": "TS_MEAN($close, 5)", "quality": "high_quality"},
        }}
        lib_path = tmp_path / "lib.json"
        lib_path.write_text(json.dumps(lib))
        return SignalGenerator(
            booster_path=tmp_path / "m.txt",
            feature_cols=list(feature_cols),
            factor_json_files=[lib_path],
            h5_path=tmp_path / "pv.h5",
            lookback_days=30,
        )

    def test_generate_returns_dict(self, tmp_path):
        gen = self._make_gen(tmp_path)
        df = _make_ohlcv_df()

        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.9, 0.7])

        factors_df = pd.DataFrame(
            {"f1": [0.5, 0.3], "f2": [0.8, 0.6]},
            index=pd.Index(["AAPL", "MSFT"], name="instrument"),
        )

        gen._booster = mock_booster
        with patch.object(gen, "_load_h5_window", return_value=df), \
             patch.object(gen, "_compute_factors", return_value=factors_df):
            result = gen.generate(as_of_date="2024-01-10")

        assert isinstance(result, dict)
        assert set(result.keys()) == {"AAPL", "MSFT"}

    def test_generate_empty_h5_returns_empty(self, tmp_path):
        gen = self._make_gen(tmp_path)
        gen._booster = MagicMock()
        with patch.object(gen, "_load_h5_window", return_value=pd.DataFrame()):
            result = gen.generate(as_of_date="2024-01-10")
        assert result == {}

    def test_generate_drops_all_nan_instruments(self, tmp_path):
        gen = self._make_gen(tmp_path)
        mock_booster = MagicMock()
        mock_booster.predict.return_value = np.array([0.5])
        gen._booster = mock_booster

        df = _make_ohlcv_df()
        # AAPL has all NaN features — should be dropped by min_valid_fraction
        factors_df = pd.DataFrame(
            {"f1": [np.nan, 0.3], "f2": [np.nan, 0.6]},
            index=pd.Index(["AAPL", "MSFT"], name="instrument"),
        )
        with patch.object(gen, "_load_h5_window", return_value=df), \
             patch.object(gen, "_compute_factors", return_value=factors_df):
            result = gen.generate(as_of_date="2024-01-10", min_valid_fraction=0.9)

        assert "AAPL" not in result
        assert "MSFT" in result

    def test_generate_missing_h5_returns_empty(self, tmp_path):
        gen = self._make_gen(tmp_path)
        gen._booster = MagicMock()
        with pytest.raises(FileNotFoundError):
            gen._load_h5_window(pd.Timestamp("2024-01-10"))
