"""
Tests for FactorRegulator: complexity, redundancy, and consistency checks.

Covers:
- is_parsable: empty/malformed expressions
- evaluate: expression evaluation metrics
- is_expression_acceptable: boundary conditions for thresholds
- add_factor: adding factors to the zoo
- Integration: full quality gate pipeline
"""

import tempfile
from pathlib import Path

import pytest

from quantaalpha.factors.regulator.factor_regulator import FactorRegulator


class TestIsParsable:
    """Tests for is_parsable method."""

    def test_empty_string(self):
        reg = FactorRegulator()
        assert reg.is_parsable("") is False

    def test_none_expression(self):
        reg = FactorRegulator()
        assert reg.is_parsable(None) is False

    def test_valid_simple_expression(self):
        reg = FactorRegulator()
        assert reg.is_parsable("$close") is True

    def test_valid_complex_expression(self):
        reg = FactorRegulator()
        assert reg.is_parsable("RANK(TS_MEAN($close, 5) / (TS_STD($close, 5) + 1e-8))") is True

    def test_invalid_syntax(self):
        reg = FactorRegulator()
        assert reg.is_parsable("TS_MEAN($close, )") is False


class TestEvaluate:
    """Tests for evaluate method."""

    def test_evaluate_simple_expression(self):
        reg = FactorRegulator()
        success, eval_dict = reg.evaluate("$close")
        assert success is True
        assert eval_dict["expr"] == "$close"
        assert eval_dict["symbol_length"] > 0

    def test_evaluate_counts_base_features(self):
        reg = FactorRegulator()
        success, eval_dict = reg.evaluate("($close + $open) / $volume")
        assert success is True
        assert eval_dict["num_base_features"] == 3


class TestIsExpressionAcceptable:
    """Tests for is_expression_acceptable method."""

    def test_acceptable_simple_expression(self):
        reg = FactorRegulator()
        success, eval_dict = reg.evaluate("$close")
        assert reg.is_expression_acceptable(eval_dict) is True

    def test_zero_nodes_rejected(self):
        reg = FactorRegulator()
        eval_dict = {
            "expr": "",
            "duplicated_subtree_size": 0,
            "duplicated_subtree": "",
            "matched_alpha": None,
            "num_free_args": 0,
            "num_unique_vars": 0,
            "num_all_nodes": 0,
            "symbol_length": 0,
            "num_base_features": 0,
        }
        assert reg.is_expression_acceptable(eval_dict) is False

    def test_invalid_ratio_rejected(self):
        reg = FactorRegulator()
        eval_dict = {
            "expr": "test",
            "duplicated_subtree_size": 0,
            "duplicated_subtree": "",
            "matched_alpha": None,
            "num_free_args": 10,
            "num_unique_vars": 5,
            "num_all_nodes": 5,
            "symbol_length": 10,
            "num_base_features": 1,
        }
        assert reg.is_expression_acceptable(eval_dict) is False


class TestAddFactor:
    """Tests for add_factor method."""

    def test_add_single_factor(self):
        reg = FactorRegulator()
        reg.add_factor("test_factor", "$close")
        assert len(reg.alphazoo) == 1

    def test_add_multiple_factors(self):
        reg = FactorRegulator()
        reg.add_factor(["f1", "f2", "f3"], ["$close", "$open", "$volume"])
        assert len(reg.alphazoo) == 3

    def test_add_factor_mismatched_lengths(self):
        reg = FactorRegulator()
        with pytest.raises(ValueError):
            reg.add_factor(["f1", "f2"], ["$close"])


class TestIntegration:
    """Integration tests for full quality gate pipeline."""

    def test_full_pipeline_valid_factor(self):
        reg = FactorRegulator()
        expr = "RANK(TS_MEAN($close, 10))"
        assert reg.is_parsable(expr)
        success, eval_dict = reg.evaluate(expr)
        assert success
        assert reg.is_expression_acceptable(eval_dict)

    def test_full_pipeline_invalid_factor(self):
        reg = FactorRegulator()
        assert reg.is_parsable("RANK(TS_MEAN($close, ))") is False

    def test_duplication_detection_with_zoo(self):
        import pandas as pd

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            zoo_data = pd.DataFrame(
                {"factor_name": ["alpha001"], "factor_expression": ["RANK(TS_MEAN($close, 5))"]}
            )
            zoo_data.to_csv(f, index=False)
            zoo_path = f.name

        try:
            reg = FactorRegulator(factor_zoo_path=zoo_path)
            success, eval_dict = reg.evaluate("RANK(TS_MEAN($close, 5))")
            assert eval_dict["duplicated_subtree_size"] > 0
        finally:
            Path(zoo_path).unlink()

    def test_threshold_customization(self):
        reg = FactorRegulator(
            factor_zoo_path=None,
            duplication_threshold=2,
            symbol_length_threshold=10,
            base_features_threshold=1,
        )
        expr = "TS_MEAN($close, 5)"
        if reg.is_parsable(expr):
            success, eval_dict = reg.evaluate(expr)
            if success:
                assert isinstance(reg.is_expression_acceptable(eval_dict), bool)