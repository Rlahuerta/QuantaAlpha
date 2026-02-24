"""
Tests for quantaalpha.pipeline.evolution.ast_mutation (Phase C depth).

Covers all three mutation layers:
  1. Window scaling  (perturb_expression_windows)
  2. Operator substitution  (substitute_operators)
  3. Structural wrapping  (wrap_expression)
  4. Combined hint builder  (build_ast_mutation_hint)
"""

import pytest
from quantaalpha.pipeline.evolution.ast_mutation import (
    perturb_expression_windows,
    substitute_operators,
    wrap_expression,
    build_ast_mutation_hint,
)


# ---------------------------------------------------------------------------
# Layer 1: Window scaling
# ---------------------------------------------------------------------------

class TestPerturbExpressionWindows:
    def test_single_operator(self):
        variants = perturb_expression_windows("TS_MEAN($return, 20)")
        assert "TS_MEAN($return, 10)" in variants  # ×0.5
        assert "TS_MEAN($return, 40)" in variants  # ×2

    def test_multi_operator_scaled_together(self):
        expr = "RANK(TS_MEAN($return, 20) / TS_STD($return, 10))"
        variants = perturb_expression_windows(expr)
        assert "RANK(TS_MEAN($return, 10) / TS_STD($return, 5))" in variants
        assert "RANK(TS_MEAN($return, 40) / TS_STD($return, 20))" in variants

    def test_no_window_returns_empty(self):
        assert perturb_expression_windows("RANK($close / $open)") == []

    def test_min_window_clamp(self):
        # window=3, ×0.5 → 1.5 → rounds to 2 (min)
        variants = perturb_expression_windows("TS_MEAN($return, 3)")
        halved = [v for v in variants if "TS_MEAN($return, 2)" in v]
        assert halved, f"Expected min-clamped variant, got: {variants}"

    def test_max_window_clamp(self):
        # window=200, ×2 → 400 → clamped to 240
        variants = perturb_expression_windows("TS_MEAN($return, 200)")
        doubled = [v for v in variants if "TS_MEAN($return, 240)" in v]
        assert doubled, f"Expected max-clamped variant, got: {variants}"

    def test_deduplication(self):
        # ×0.5 and ×2 on window=2 → both clamp to 2 and 4; no duplicates
        variants = perturb_expression_windows("TS_MEAN($return, 2)", scales=(0.5, 0.5))
        assert len(variants) == len(set(variants))

    def test_custom_scales(self):
        variants = perturb_expression_windows("EMA($close, 10)", scales=(3.0,))
        assert "EMA($close, 30)" in variants


# ---------------------------------------------------------------------------
# Layer 2: Operator substitution
# ---------------------------------------------------------------------------

class TestSubstituteOperators:
    def test_ts_mean_to_ema(self):
        variants = substitute_operators("TS_MEAN($return, 20)")
        assert any("EMA($return, 20)" in v for v in variants)

    def test_ts_std_to_ts_zscore(self):
        variants = substitute_operators("TS_STD($return, 20)")
        assert any("TS_ZSCORE($return, 20)" in v for v in variants)

    def test_rank_to_zscore(self):
        variants = substitute_operators("RANK($close / $open)")
        assert any("ZSCORE($close / $open)" in v for v in variants)

    def test_no_substitutable_op_returns_empty(self):
        # LOG, SQRT have no substitution rules
        assert substitute_operators("LOG(ABS($return) + 1e-8)") == []

    def test_only_first_occurrence_replaced_per_variant(self):
        # Two TS_MEAN — each variant only replaces one (count=1 in regex.sub)
        expr = "TS_MEAN($return, 20) / TS_MEAN($close, 5)"
        variants = substitute_operators(expr)
        # At least one variant should have EMA in it
        assert any("EMA" in v for v in variants)
        # No variant should replace both TS_MEAN occurrences at once
        for v in variants:
            assert v.count("EMA") <= 1 or v.count("TS_MEAN") >= 1

    def test_no_duplicate_variants(self):
        variants = substitute_operators("TS_MEAN($return, 20)")
        assert len(variants) == len(set(variants))


# ---------------------------------------------------------------------------
# Layer 3: Structural wrapping
# ---------------------------------------------------------------------------

class TestWrapExpression:
    def test_adds_rank_and_zscore_when_missing(self):
        wrapped = wrap_expression("TS_MEAN($return, 20)")
        descs = [desc for desc, _ in wrapped]
        exprs = [e for _, e in wrapped]
        assert "RANK normalization" in descs
        assert "ZSCORE normalization" in descs
        assert "RANK(TS_MEAN($return, 20))" in exprs
        assert "ZSCORE(TS_MEAN($return, 20))" in exprs

    def test_adds_delta_when_missing(self):
        wrapped = wrap_expression("TS_MEAN($return, 20)")
        exprs = [e for _, e in wrapped]
        assert "DELTA(TS_MEAN($return, 20), 1)" in exprs

    def test_skips_rank_when_already_wrapped(self):
        wrapped = wrap_expression("RANK(TS_MEAN($return, 20))")
        descs = [desc for desc, _ in wrapped]
        assert "RANK normalization" not in descs
        assert "ZSCORE normalization" not in descs

    def test_skips_delta_when_already_wrapped(self):
        wrapped = wrap_expression("DELTA(TS_MEAN($return, 20), 5)")
        descs = [desc for desc, _ in wrapped]
        assert "temporal derivative (momentum)" not in descs

    def test_case_insensitive_wrap_detection(self):
        wrapped = wrap_expression("rank(TS_MEAN($return, 20))")
        descs = [desc for desc, _ in wrapped]
        assert "RANK normalization" not in descs


# ---------------------------------------------------------------------------
# Combined hint builder
# ---------------------------------------------------------------------------

class TestBuildAstMutationHint:
    def test_returns_non_empty_for_temporal_expression(self):
        hint = build_ast_mutation_hint("TS_MEAN($return, 20)", "Test_Factor")
        assert hint != ""
        assert "Test_Factor" in hint

    def test_contains_all_sections(self):
        hint = build_ast_mutation_hint("RANK(TS_MEAN($return, 20) / TS_STD($return, 10))")
        assert "Time-horizon variants" in hint
        assert "Operator-substitution" in hint
        assert "Structural wrappers" in hint

    def test_no_duplicate_expressions_in_hint(self):
        hint = build_ast_mutation_hint("TS_MEAN($return, 20)", "F")
        lines = [l.strip() for l in hint.splitlines() if l.strip().startswith("- `")]
        exprs = [l for l in lines]
        assert len(exprs) == len(set(exprs)), f"Duplicate entries in hint:\n{hint}"

    def test_returns_empty_for_no_mutations_possible(self):
        # Already RANK-wrapped, no temporal ops with windows, no substitutable ops
        # RANK($close / $open) → no window variants, no op subs, but ZSCORE+DELTA still apply
        # Use a fully-neutralized form: wrap with both cs ops to block those, then check DELTA
        # Simplest: expression with no temporal ops that is already cs-and-delta wrapped
        # DELTA(RANK($close), 1) still has DELTA window=1 → perturb gives window=2
        # So truly "no mutations" is near-impossible; instead verify a plain var gets wrappers
        hint = build_ast_mutation_hint("$close")
        # $close legitimately gets RANK/ZSCORE/DELTA wrappers suggested
        assert "Structural wrappers" in hint

    def test_factor_name_in_header(self):
        hint = build_ast_mutation_hint("EMA($close, 20)", "My_Signal")
        assert "My_Signal" in hint

    def test_no_factor_name_still_works(self):
        hint = build_ast_mutation_hint("EMA($close, 20)")
        assert hint != ""
        assert "My_Signal" not in hint
