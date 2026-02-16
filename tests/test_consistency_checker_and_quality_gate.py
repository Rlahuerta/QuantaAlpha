from types import SimpleNamespace

import quantaalpha.factors.coder.function_lib  # noqa: F401
import quantaalpha.factors.regulator.consistency_checker as checker_module
from quantaalpha.factors.regulator.consistency_checker import (
    ComplexityChecker,
    ConsistencyCheckResult,
    FactorConsistencyChecker,
    FactorQualityGate,
    RedundancyChecker,
)


def test_consistency_check_result_to_dict_roundtrip():
    result = ConsistencyCheckResult(
        is_consistent=False,
        hypothesis_to_description="h2d",
        description_to_formulation="d2f",
        formulation_to_expression="f2e",
        overall_feedback="bad",
        corrected_expression="expr2",
        corrected_description="desc2",
        severity="major",
    )
    as_dict = result.to_dict()

    assert as_dict["is_consistent"] is False
    assert as_dict["corrected_expression"] == "expr2"
    assert as_dict["severity"] == "major"


def test_factor_consistency_checker_disabled_and_should_proceed():
    checker = FactorConsistencyChecker(enabled=False, strict_mode=True)
    result = checker.check_consistency(
        hypothesis="h",
        factor_name="f",
        factor_description="d",
        factor_formulation="form",
        factor_expression="expr",
    )

    assert result.is_consistent is True
    assert "disabled" in result.overall_feedback.lower()
    assert checker.should_proceed_to_backtest(result) is True


def test_factor_consistency_checker_success_and_exception(monkeypatch):
    checker = FactorConsistencyChecker(enabled=True)
    monkeypatch.setattr(
        checker_module,
        "consistency_prompts",
        {
            "consistency_check_system": "SYS",
            "consistency_check_user": "{{factor_name}}|{{hypothesis}}|{{variables}}",
        },
    )

    class _FakeAPIBackend:
        def build_messages_and_create_chat_completion(self, **kwargs):
            assert kwargs["json_mode"] is True
            assert "factorA" in kwargs["user_prompt"]
            return (
                '{"is_consistent": false, "severity": "minor",'
                '"overall_feedback": "needs fix", "corrected_expression": "x2"}'
            )

    monkeypatch.setattr(checker_module, "APIBackend", lambda: _FakeAPIBackend())
    result = checker.check_consistency(
        hypothesis="h",
        factor_name="factorA",
        factor_description="desc",
        factor_formulation="form",
        factor_expression="expr",
        variables={"$close": "close price"},
    )
    assert result.is_consistent is False
    assert result.severity == "minor"
    assert result.corrected_expression == "x2"

    class _BrokenAPIBackend:
        def build_messages_and_create_chat_completion(self, **kwargs):
            raise RuntimeError("llm unavailable")

    monkeypatch.setattr(checker_module, "APIBackend", lambda: _BrokenAPIBackend())
    result_error = checker.check_consistency(
        hypothesis="h",
        factor_name="factorB",
        factor_description="desc",
        factor_formulation="form",
        factor_expression="expr",
    )
    assert result_error.is_consistent is True
    assert "failed with error" in result_error.overall_feedback


def test_check_and_correct_expression_description_and_strict_paths(monkeypatch):
    checker = FactorConsistencyChecker(enabled=True, strict_mode=False, max_correction_attempts=3)

    seq = [
        ConsistencyCheckResult(
            is_consistent=False,
            hypothesis_to_description="",
            description_to_formulation="",
            formulation_to_expression="",
            overall_feedback="fix expr",
            corrected_expression="expr-2",
            severity="major",
        ),
        ConsistencyCheckResult(
            is_consistent=True,
            hypothesis_to_description="",
            description_to_formulation="",
            formulation_to_expression="",
            overall_feedback="ok",
            severity="none",
        ),
    ]
    state = {"i": 0}

    def _fake_check_consistency(**kwargs):
        out = seq[state["i"]]
        state["i"] += 1
        return out

    monkeypatch.setattr(checker, "check_consistency", _fake_check_consistency)
    result, final_expr, final_desc = checker.check_and_correct(
        hypothesis="h",
        factor_name="f",
        factor_description="d1",
        factor_formulation="form",
        factor_expression="expr-1",
    )
    assert result.is_consistent is True
    assert final_expr == "expr-2"
    assert final_desc == "d1"

    checker_desc = FactorConsistencyChecker(enabled=True, strict_mode=False, max_correction_attempts=2)
    seq_desc = [
        ConsistencyCheckResult(
            is_consistent=False,
            hypothesis_to_description="",
            description_to_formulation="",
            formulation_to_expression="",
            overall_feedback="fix desc",
            corrected_description="d2",
            severity="minor",
        ),
        ConsistencyCheckResult(
            is_consistent=True,
            hypothesis_to_description="",
            description_to_formulation="",
            formulation_to_expression="",
            overall_feedback="ok",
            severity="none",
        ),
    ]
    state_desc = {"i": 0}
    monkeypatch.setattr(checker_desc, "check_consistency", lambda **kwargs: seq_desc.pop(0))
    result_desc, expr_desc, final_desc_val = checker_desc.check_and_correct(
        hypothesis="h",
        factor_name="f",
        factor_description="d1",
        factor_formulation="form",
        factor_expression="expr",
    )
    assert result_desc.is_consistent is True
    assert expr_desc == "expr"
    assert final_desc_val == "d2"

    checker_strict = FactorConsistencyChecker(enabled=True, strict_mode=True, max_correction_attempts=2)
    monkeypatch.setattr(
        checker_strict,
        "check_consistency",
        lambda **kwargs: ConsistencyCheckResult(
            is_consistent=False,
            hypothesis_to_description="",
            description_to_formulation="",
            formulation_to_expression="",
            overall_feedback="fail",
            severity="major",
        ),
    )
    strict_result, strict_expr, strict_desc = checker_strict.check_and_correct(
        hypothesis="h",
        factor_name="f",
        factor_description="d1",
        factor_formulation="form",
        factor_expression="expr",
    )
    assert strict_result.is_consistent is False
    assert strict_expr == "expr"
    assert strict_desc == "d1"


def test_batch_check_and_should_proceed_paths(monkeypatch):
    checker = FactorConsistencyChecker(enabled=True, strict_mode=False)

    def _fake_check_and_correct(**kwargs):
        result = ConsistencyCheckResult(
            is_consistent=False,
            hypothesis_to_description="h2d",
            description_to_formulation="d2f",
            formulation_to_expression="f2e",
            overall_feedback="minor issue",
            severity="minor",
        )
        return result, "expr-fixed", "desc-fixed"

    monkeypatch.setattr(checker, "check_and_correct", _fake_check_and_correct)
    factors = [
        {"name": "f1", "description": "d1", "formulation": "fo1", "expression": "e1"},
        {"name": "f2", "description": "d2", "formulation": "fo2", "expression": "e2"},
    ]
    batch = checker.batch_check(hypothesis="h", factors=factors)

    assert len(batch) == 2
    updated_factor, result = batch[0]
    assert updated_factor["expression"] == "expr-fixed"
    assert updated_factor["description"] == "desc-fixed"
    assert updated_factor["consistency_check"]["severity"] == "minor"
    assert checker.should_proceed_to_backtest(result) is True

    checker.strict_mode = True
    assert checker.should_proceed_to_backtest(result) is False
    checker.strict_mode = False
    major_result = ConsistencyCheckResult(
        is_consistent=False,
        hypothesis_to_description="",
        description_to_formulation="",
        formulation_to_expression="",
        overall_feedback="major",
        severity="major",
    )
    assert checker.should_proceed_to_backtest(major_result) is False


def test_complexity_checker_disabled_pass_fail_and_exception(monkeypatch):
    disabled_checker = ComplexityChecker(enabled=False)
    assert disabled_checker.check("x")[0] is True

    import quantaalpha.factors.coder.factor_ast as ast_module

    checker = ComplexityChecker(enabled=True, symbol_length_threshold=10, base_features_threshold=2, free_args_ratio_threshold=0.5)
    monkeypatch.setattr(ast_module, "calculate_symbol_length", lambda expr: 8)
    monkeypatch.setattr(ast_module, "count_base_features", lambda expr: 2)
    monkeypatch.setattr(ast_module, "count_free_args", lambda expr: 1)
    monkeypatch.setattr(ast_module, "count_all_nodes", lambda expr: 4)
    passed, feedback = checker.check("expr")
    assert passed is True
    assert "passed" in feedback.lower()

    monkeypatch.setattr(ast_module, "calculate_symbol_length", lambda expr: 11)
    monkeypatch.setattr(ast_module, "count_base_features", lambda expr: 3)
    monkeypatch.setattr(ast_module, "count_free_args", lambda expr: 3)
    monkeypatch.setattr(ast_module, "count_all_nodes", lambda expr: 4)
    failed, failed_feedback = checker.check("expr")
    assert failed is False
    assert "Symbol Length" in failed_feedback
    assert "Base Features" in failed_feedback
    assert "Free Args Ratio" in failed_feedback

    monkeypatch.setattr(ast_module, "calculate_symbol_length", lambda expr: (_ for _ in ()).throw(RuntimeError("boom")))
    skipped, skipped_feedback = checker.check("expr")
    assert skipped is True
    assert "skipped due to error" in skipped_feedback


def test_redundancy_checker_disabled_lazy_property_and_check_paths(monkeypatch):
    disabled_checker = RedundancyChecker(enabled=False)
    assert disabled_checker.check("x") == (True, "Redundancy check disabled", {})

    import quantaalpha.factors.regulator.factor_regulator as regulator_module

    class _LazyFakeRegulator:
        def __init__(self, factor_zoo_path=None, duplication_threshold=None):
            self.factor_zoo_path = factor_zoo_path
            self.duplication_threshold = duplication_threshold

    monkeypatch.setattr(regulator_module, "FactorRegulator", _LazyFakeRegulator)
    lazy_checker = RedundancyChecker(enabled=True, duplication_threshold=9, factor_zoo_path="zoo.json")
    lazy_reg = lazy_checker.factor_regulator
    assert isinstance(lazy_reg, _LazyFakeRegulator)
    assert lazy_reg.duplication_threshold == 9
    assert lazy_reg.factor_zoo_path == "zoo.json"

    checker = RedundancyChecker(enabled=True, duplication_threshold=5)
    checker._factor_regulator = SimpleNamespace(
        is_parsable=lambda expr: False,
        evaluate=lambda expr: (True, {}),
    )
    assert checker.check("expr")[0] is False
    assert "cannot be parsed" in checker.check("expr")[1]

    checker._factor_regulator = SimpleNamespace(
        is_parsable=lambda expr: True,
        evaluate=lambda expr: (False, {}),
    )
    assert checker.check("expr")[0] is False
    assert "Failed to evaluate" in checker.check("expr")[1]

    checker._factor_regulator = SimpleNamespace(
        is_parsable=lambda expr: True,
        evaluate=lambda expr: (True, {"duplicated_subtree_size": 6, "matched_alpha": "alpha_x", "duplicated_subtree": "TS_MEAN"}),
    )
    failed, feedback, details = checker.check("expr")
    assert failed is False
    assert "Redundancy Check Failed" in feedback
    assert details["matched_alpha"] == "alpha_x"

    checker._factor_regulator = SimpleNamespace(
        is_parsable=lambda expr: True,
        evaluate=lambda expr: (True, {"duplicated_subtree_size": 1}),
    )
    passed, pass_feedback, pass_details = checker.check("expr")
    assert passed is True
    assert "passed" in pass_feedback
    assert pass_details["duplicated_subtree_size"] == 1

    checker._factor_regulator = SimpleNamespace(
        is_parsable=lambda expr: (_ for _ in ()).throw(RuntimeError("boom")),
        evaluate=lambda expr: (True, {}),
    )
    skipped, skipped_feedback, skipped_details = checker.check("expr")
    assert skipped is True
    assert "skipped due to error" in skipped_feedback
    assert skipped_details == {}


def test_factor_quality_gate_all_pass_and_failure_paths():
    passing_consistency = SimpleNamespace(
        enabled=True,
        check_and_correct=lambda **kwargs: (
            ConsistencyCheckResult(
                is_consistent=True,
                hypothesis_to_description="ok",
                description_to_formulation="ok",
                formulation_to_expression="ok",
                overall_feedback="ok",
                severity="none",
            ),
            "expr-fixed",
            "desc-fixed",
        ),
        should_proceed_to_backtest=lambda result: True,
    )
    passing_complexity = SimpleNamespace(enabled=True, check=lambda expression: (True, "Complexity ok"))
    passing_redundancy = SimpleNamespace(enabled=True, check=lambda expression: (True, "Redundancy ok", {"m": 1}))
    gate = FactorQualityGate(
        consistency_checker=passing_consistency,
        complexity_checker=passing_complexity,
        redundancy_checker=passing_redundancy,
        consistency_enabled=True,
        complexity_enabled=True,
        redundancy_enabled=True,
    )

    passed, feedback, details = gate.evaluate(
        hypothesis="h",
        factor_name="f1",
        factor_description="d1",
        factor_formulation="fo1",
        factor_expression="expr1",
    )
    assert passed is True
    assert "passed all quality gates" in feedback
    assert details["corrected_expression"] == "expr-fixed"
    assert details["redundancy"]["passed"] is True

    failing_consistency = SimpleNamespace(
        enabled=True,
        check_and_correct=lambda **kwargs: (
            ConsistencyCheckResult(
                is_consistent=False,
                hypothesis_to_description="bad",
                description_to_formulation="bad",
                formulation_to_expression="bad",
                overall_feedback="consistency fail",
                severity="major",
            ),
            "expr-still-bad",
            "desc-still-bad",
        ),
        should_proceed_to_backtest=lambda result: False,
    )
    failing_complexity = SimpleNamespace(enabled=True, check=lambda expression: (False, "complexity fail"))
    failing_redundancy = SimpleNamespace(enabled=True, check=lambda expression: (False, "redundancy fail", {"dup": 10}))
    failing_gate = FactorQualityGate(
        consistency_checker=failing_consistency,
        complexity_checker=failing_complexity,
        redundancy_checker=failing_redundancy,
        consistency_enabled=True,
        complexity_enabled=True,
        redundancy_enabled=True,
    )
    failed, fail_feedback, fail_details = failing_gate.evaluate(
        hypothesis="h",
        factor_name="f2",
        factor_description="d2",
        factor_formulation="fo2",
        factor_expression="expr2",
    )
    assert failed is False
    assert "[Consistency]" in fail_feedback
    assert "[Complexity]" in fail_feedback
    assert "[Redundancy]" in fail_feedback
    assert fail_details["complexity"]["passed"] is False
    assert fail_details["redundancy"]["passed"] is False
