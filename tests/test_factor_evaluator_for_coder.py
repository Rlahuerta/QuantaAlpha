from types import SimpleNamespace

import pandas as pd

import quantaalpha.factors.coder.function_lib  # noqa: F401
import quantaalpha.factors.coder.evaluators as evaluators_module
from quantaalpha.factors.coder.evaluators import FactorEvaluatorForCoder


class _DummyImplementation:
    def __init__(self, code="expr = 'TS_MEAN($close, 5)'", code_dict=None, execute_result=None):
        self.code = code
        self.code_dict = code_dict if code_dict is not None else {"factor.py": code}
        self._execute_result = execute_result if execute_result is not None else ("ok", pd.DataFrame({"f": [1.0]}))

    def execute(self):
        return self._execute_result


class _FakeFactorRegulator:
    def __init__(
        self,
        factor_zoo_path=None,
        duplication_threshold=5,
        symbol_length_threshold=300,
        base_features_threshold=6,
    ):
        self.factor_zoo_path = factor_zoo_path
        self.duplication_threshold = duplication_threshold
        self.symbol_length_threshold = symbol_length_threshold
        self.base_features_threshold = base_features_threshold
        self.alphazoo = []
        self._parsable = True
        self._evaluate_ok = True
        self._acceptable = True
        self._eval_dict = {
            "duplicated_subtree_size": 0,
            "num_free_args": 0,
            "num_all_nodes": 1,
            "num_unique_vars": 0,
            "symbol_length": 1,
            "num_base_features": 1,
        }

    def is_parsable(self, expr):
        if isinstance(self._parsable, Exception):
            raise self._parsable
        return self._parsable

    def evaluate(self, expr):
        return self._evaluate_ok, self._eval_dict

    def is_expression_acceptable(self, eval_dict):
        return self._acceptable


def _build_evaluator(monkeypatch):
    monkeypatch.setattr(evaluators_module, "FactorRegulator", _FakeFactorRegulator)
    return FactorEvaluatorForCoder(scen=None)


def test_extract_expr_and_ast_regularization_core_paths(monkeypatch):
    evaluator = _build_evaluator(monkeypatch)

    assert evaluator.extract_expr("expr = 'TS_MEAN($close, 5)'") == "TS_MEAN($close, 5)"
    assert evaluator.extract_expr("no expression") == ""

    class _NoCodeImpl:
        pass

    skipped_ok, skipped_feedback = evaluator.check_ast_regularization(_NoCodeImpl())
    assert skipped_ok is True
    assert "Skipped" in skipped_feedback

    empty_expr_ok, empty_expr_feedback = evaluator.check_ast_regularization(
        _DummyImplementation(code="print('hello')")
    )
    assert empty_expr_ok is True
    assert empty_expr_feedback == ""

    evaluator.factor_regulator._parsable = False
    unparsable_ok, unparsable_feedback = evaluator.check_ast_regularization(_DummyImplementation())
    assert unparsable_ok is False
    assert "cannot be parsed" in unparsable_feedback

    evaluator.factor_regulator._parsable = True
    evaluator.factor_regulator._evaluate_ok = False
    eval_fail_ok, eval_fail_feedback = evaluator.check_ast_regularization(_DummyImplementation())
    assert eval_fail_ok is False
    assert "Failed to evaluate expression" in eval_fail_feedback


def test_ast_regularization_note_failure_and_exception_paths(monkeypatch):
    evaluator = _build_evaluator(monkeypatch)
    evaluator.factor_regulator._evaluate_ok = True
    evaluator.factor_regulator._acceptable = False
    evaluator.factor_regulator._eval_dict = {
        "duplicated_subtree_size": 1,
        "num_free_args": 0,
        "num_all_nodes": 0,
        "num_unique_vars": 0,
        "symbol_length": 1,
        "num_base_features": 1,
    }
    evaluator.factor_regulator.factor_zoo_path = None
    evaluator.factor_regulator.alphazoo = []
    note_ok, note_feedback = evaluator.check_ast_regularization(_DummyImplementation())
    assert note_ok is True
    assert "Novelty check skipped" in note_feedback

    evaluator.factor_regulator.factor_zoo_path = "zoo.json"
    evaluator.factor_regulator.alphazoo = ["alpha_001"]
    evaluator.factor_regulator.symbol_length_threshold = 10
    evaluator.factor_regulator.base_features_threshold = 2
    evaluator.factor_regulator._eval_dict = {
        "duplicated_subtree_size": 8,
        "duplicated_subtree": "TS_MEAN($close, 5)",
        "matched_alpha": "alpha_001",
        "num_free_args": 6,
        "num_all_nodes": 10,
        "num_unique_vars": 6,
        "symbol_length": 15,
        "num_base_features": 4,
    }
    fail_ok, fail_feedback = evaluator.check_ast_regularization(_DummyImplementation())
    assert fail_ok is False
    assert "AST Regularization Check Failed" in fail_feedback
    assert "Free Arguments Ratio Check Failed" in fail_feedback

    evaluator.factor_regulator._parsable = RuntimeError("boom")
    exc_ok, exc_feedback = evaluator.check_ast_regularization(_DummyImplementation())
    assert exc_ok is True
    assert "Skipped" in exc_feedback


def test_evaluate_returns_cached_or_failed_task_feedback(monkeypatch):
    evaluator = _build_evaluator(monkeypatch)
    target_task = SimpleNamespace(get_task_information=lambda: "task-info", version=1)
    implementation = _DummyImplementation()

    queried_success = SimpleNamespace(
        success_task_to_knowledge_dict={"task-info": SimpleNamespace(feedback="cached-feedback")},
        failed_task_info_set=set(),
    )
    assert evaluator.evaluate(target_task, implementation, queried_knowledge=queried_success) == "cached-feedback"

    queried_failed = SimpleNamespace(
        success_task_to_knowledge_dict={},
        failed_task_info_set={"task-info"},
    )
    failed_feedback = evaluator.evaluate(target_task, implementation, queried_knowledge=queried_failed)
    assert failed_feedback.final_decision is False
    assert "failed too many times" in failed_feedback.execution_feedback


def test_evaluate_ast_failure_and_value_decision_branches(monkeypatch):
    evaluator = _build_evaluator(monkeypatch)
    target_task = SimpleNamespace(get_task_information=lambda: "task-info", version=1)

    class _ShouldNotExecute(_DummyImplementation):
        def execute(self):
            raise AssertionError("execute should not be called on AST failure")

    monkeypatch.setattr(evaluator, "check_ast_regularization", lambda implementation: (False, "ast bad"))
    ast_failed = evaluator.evaluate(target_task, _ShouldNotExecute())
    assert ast_failed.final_decision is False
    assert "rejected due to AST regularization violations" in ast_failed.final_feedback

    implementation = _DummyImplementation(
        execute_result=("line1\nwarning hidden\nline2", pd.DataFrame({"f": [1.0, 2.0]}))
    )
    monkeypatch.setattr(evaluator, "check_ast_regularization", lambda implementation: (True, "AST passed"))
    evaluator.value_evaluator = SimpleNamespace(evaluate=lambda **kwargs: ("value ok", True))
    evaluator.code_evaluator = SimpleNamespace(evaluate=lambda **kwargs: ("code should not run", None))
    evaluator.final_decision_evaluator = SimpleNamespace(
        evaluate=lambda **kwargs: (False, "final should not run")
    )
    value_true = evaluator.evaluate(target_task, implementation)
    assert value_true.final_decision is True
    assert value_true.code_feedback == "Final decision is True and there are no code critics."
    assert "warning" not in value_true.execution_feedback.lower()
    assert value_true.final_feedback == "Value evaluation passed, skip final decision evaluation."

    evaluator.value_evaluator = SimpleNamespace(evaluate=lambda **kwargs: ("value bad", False))
    evaluator.code_evaluator = SimpleNamespace(evaluate=lambda **kwargs: ("code critics", None))
    evaluator.final_decision_evaluator = SimpleNamespace(
        evaluate=lambda **kwargs: (True, "final should not run")
    )
    value_false = evaluator.evaluate(target_task, implementation)
    assert value_false.final_decision is False
    assert value_false.code_feedback == "code critics"
    assert value_false.final_feedback == "Value evaluation failed, skip final decision evaluation."


def test_evaluate_with_no_generated_df_uses_final_decision(monkeypatch):
    evaluator = _build_evaluator(monkeypatch)
    target_task = SimpleNamespace(get_task_information=lambda: "task-info", version=1)
    implementation = _DummyImplementation(execute_result=("ok", None))

    monkeypatch.setattr(evaluator, "check_ast_regularization", lambda implementation: (True, "AST passed"))
    evaluator.value_evaluator = SimpleNamespace(
        evaluate=lambda **kwargs: (_ for _ in ()).throw(AssertionError("value evaluator should not run"))
    )
    evaluator.code_evaluator = SimpleNamespace(evaluate=lambda **kwargs: ("code reviewed", None))
    evaluator.final_decision_evaluator = SimpleNamespace(evaluate=lambda **kwargs: (True, "approved"))

    feedback = evaluator.evaluate(target_task, implementation)
    assert feedback.value_generated_flag is False
    assert "No factor value generated" in feedback.value_feedback
    assert feedback.code_feedback == "code reviewed"
    assert feedback.final_decision is True
    assert feedback.final_feedback == "approved"
