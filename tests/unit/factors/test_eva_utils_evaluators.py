from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import quantaalpha.factors.coder.eva_utils as eva_utils_module
from quantaalpha.factors.coder.factor import FactorExecutionResult
from quantaalpha.factors.coder.eva_utils import (
    FactorCodeEvaluator,
    FactorCorrelationEvaluator,
    FactorDatetimeDailyEvaluator,
    FactorEqualValueRatioEvaluator,
    FactorEvaluator,
    FactorFinalDecisionEvaluator,
    FactorIndexEvaluator,
    FactorInfEvaluator,
    FactorMissingValuesEvaluator,
    FactorOutputFormatEvaluator,
    FactorRowCountEvaluator,
    FactorSingleColumnEvaluator,
    FactorValueEvaluator,
)


class DummyWorkspace:
    def __init__(self, df, code="pass"):
        self._df = df
        self.code = code
        self.target_task = None

    def execute(self):
        success = self._df is not None
        return FactorExecutionResult(success=success, feedback="ok", result=self._df)


class DummyWorkspaceNone(DummyWorkspace):
    def __init__(self, code="pass"):
        super().__init__(None, code=code)


def _daily_df(values, col="factor"):
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=len(values) // 2), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({col: values}, index=idx)


def test_basic_evaluators_for_inf_single_column_and_daily_checks():
    gen_df = _daily_df([1.0, np.inf, 2.0, 3.0])
    gt_df = _daily_df([1.0, 2.0, 2.0, 3.0])
    gen_ws = DummyWorkspace(gen_df)
    gt_ws = DummyWorkspace(gt_df)

    inf_feedback, inf_ok = FactorInfEvaluator().evaluate(gen_ws, gt_ws)
    single_feedback, single_ok = FactorSingleColumnEvaluator().evaluate(gen_ws, gt_ws)
    daily_feedback, daily_ok = FactorDatetimeDailyEvaluator().evaluate(gen_ws, gt_ws)

    assert "infinite values" in inf_feedback
    assert inf_ok is False
    assert single_ok is True
    assert daily_ok is True
    assert "daily" in daily_feedback


def test_datetime_daily_evaluator_rejects_minute_frequency():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2, freq="min"), ["AAA"]],
        names=["datetime", "instrument"],
    )
    gen_ws = DummyWorkspace(pd.DataFrame({"factor": [1.0, 2.0]}, index=idx))
    gt_ws = DummyWorkspace(pd.DataFrame({"factor": [1.0, 2.0]}, index=idx))

    feedback, ok = FactorDatetimeDailyEvaluator().evaluate(gen_ws, gt_ws)

    assert ok is False
    assert "not daily" in feedback


def test_row_index_missing_equal_and_correlation_evaluators():
    gt_df = _daily_df([1.0, 2.0, 3.0, 4.0])
    gen_df = gt_df.copy()

    gen_ws = DummyWorkspace(gen_df)
    gt_ws = DummyWorkspace(gt_df)

    _, row_ratio = FactorRowCountEvaluator().evaluate(gen_ws, gt_ws)
    _, idx_similarity = FactorIndexEvaluator().evaluate(gen_ws, gt_ws)
    _, missing_ok = FactorMissingValuesEvaluator().evaluate(gen_ws, gt_ws)
    _, equal_ratio = FactorEqualValueRatioEvaluator().evaluate(gen_ws, gt_ws)
    _, corr_ok = FactorCorrelationEvaluator(hard_check=True).evaluate(gen_ws, gt_ws)

    assert row_ratio == 1.0
    assert idx_similarity == 1.0
    assert missing_ok is True
    assert equal_ratio > 0.99
    assert corr_ok is True


def test_equal_value_ratio_handles_incompatible_dtypes_without_crash():
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["AAA"]],
        names=["datetime", "instrument"],
    )
    gen_ws = DummyWorkspace(pd.DataFrame({"factor": ["a", "b"]}, index=idx))
    gt_ws = DummyWorkspace(pd.DataFrame({"factor": [1.0, 2.0]}, index=idx))

    feedback, ratio = FactorEqualValueRatioEvaluator().evaluate(gen_ws, gt_ws)

    assert ratio == 0.0
    assert "differ" in feedback


def test_factor_value_evaluator_returns_true_for_identical_dataframes():
    gt_df = _daily_df([1.0, 2.0, 3.0, 4.0])
    gen_df = gt_df.copy()
    gen_ws = DummyWorkspace(gen_df)
    gt_ws = DummyWorkspace(gt_df)

    conclusion, decision = FactorValueEvaluator().evaluate(gen_ws, gt_ws)

    assert decision is True
    assert "highly correlated" in conclusion


def test_output_format_and_final_decision_evaluators_with_mocked_api(monkeypatch):
    class FakeAPIBackend:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_messages_and_calculate_token(self, **kwargs):
            return 0

        def build_messages_and_create_chat_completion(self, **kwargs):
            if "output dataframe info" in kwargs["user_prompt"]:
                return '{"output_format_feedback":"ok","output_format_decision":true}'
            return '{"final_decision": true, "final_feedback": "approved"}'

    monkeypatch.setattr(eva_utils_module, "APIBackend", FakeAPIBackend)

    gen_df = _daily_df([1.0, 2.0, 3.0, 4.0])
    gen_ws = DummyWorkspace(gen_df)
    fake_task = SimpleNamespace(get_task_information=lambda: "task info")

    output_feedback, output_ok = FactorOutputFormatEvaluator().evaluate(gen_ws, None)
    final_decision, final_feedback = FactorFinalDecisionEvaluator().evaluate(
        target_task=fake_task,
        execution_feedback="exec",
        value_feedback="value",
        code_feedback="code",
    )

    assert output_ok is True
    assert output_feedback == "ok"
    assert final_decision is True
    assert final_feedback == "approved"


def test_factor_evaluator_base_paths_and_series_conversion():
    class _BaseEvaluator(FactorEvaluator):
        def evaluate(self, target_task, implementation, gt_implementation, **kwargs):  # type: ignore[override]
            return super().evaluate(target_task, implementation, gt_implementation, **kwargs)

    evaluator = _BaseEvaluator()
    with pytest.raises(NotImplementedError):
        evaluator.evaluate(None, None, None)

    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=2), ["AAA"]],
        names=["datetime", "instrument"],
    )
    src = DummyWorkspace(pd.Series([1.0, 2.0], index=idx, name="my_factor"))
    gt = DummyWorkspace(pd.Series([1.0, 2.0], index=idx))
    gt_df, gen_df = evaluator._get_df(gt, src)

    assert list(gt_df.columns) == ["gt_factor"]
    # gen_df: unnamed Series → column "factor"; named Series → uses series name
    assert list(gen_df.columns) == ["my_factor"]
    assert str(evaluator) == "_BaseEvaluator"


def test_factor_code_evaluator_token_truncation_and_feedback(monkeypatch):
    class _Scen:
        def get_scenario_all_desc(self, *args, **kwargs):
            return "scenario"

    class _Task:
        def get_task_information(self):
            return "task-info"

    class _FakeAPI:
        token_calls = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_messages_and_calculate_token(self, **kwargs):
            _FakeAPI.token_calls += 1
            return 999 if _FakeAPI.token_calls == 1 else 0

        def build_messages_and_create_chat_completion(self, **kwargs):
            return "code-critic-feedback"

    monkeypatch.setattr(eva_utils_module, "APIBackend", _FakeAPI)
    monkeypatch.setattr(eva_utils_module.LLM_SETTINGS, "chat_token_limit", 10)

    implementation = DummyWorkspace(_daily_df([1.0, 2.0, 3.0, 4.0]), code="print('x')")
    gt_implementation = DummyWorkspace(_daily_df([1.0, 2.0, 3.0, 4.0]), code="print('gt')")
    feedback, decision = FactorCodeEvaluator(scen=_Scen()).evaluate(
        target_task=_Task(),
        implementation=implementation,
        execution_feedback="very long execution feedback text" * 5,
        value_feedback="value-check",
        gt_implementation=gt_implementation,
    )

    assert feedback == "code-critic-feedback"
    assert decision is None
    assert _FakeAPI.token_calls >= 2


def test_output_format_evaluator_retry_keyerror_path(monkeypatch):
    class _BadAPI:
        def __init__(self, **kwargs):
            pass

        def build_messages_and_create_chat_completion(self, **kwargs):
            return '{"wrong":"shape"}'

    monkeypatch.setattr(eva_utils_module, "APIBackend", _BadAPI)
    ws = DummyWorkspace(_daily_df([1.0, 2.0, 3.0, 4.0]))

    with pytest.raises(KeyError, match="Wrong JSON Response"):
        FactorOutputFormatEvaluator().evaluate(ws, None)


def test_basic_evaluators_handle_none_or_invalid_inputs():
    gt_ws = DummyWorkspace(_daily_df([1.0, 2.0, 3.0, 4.0]))
    none_ws = DummyWorkspaceNone()

    inf_feedback, inf_ok = FactorInfEvaluator().evaluate(none_ws, gt_ws)
    single_feedback, single_ok = FactorSingleColumnEvaluator().evaluate(none_ws, gt_ws)
    output_feedback, output_ok = FactorOutputFormatEvaluator().evaluate(none_ws, gt_ws)
    daily_feedback, daily_ok = FactorDatetimeDailyEvaluator().evaluate(none_ws, gt_ws)
    row_feedback, row_ok = FactorRowCountEvaluator().evaluate(none_ws, gt_ws)
    index_feedback, index_ok = FactorIndexEvaluator().evaluate(none_ws, gt_ws)
    missing_feedback, missing_ok = FactorMissingValuesEvaluator().evaluate(none_ws, gt_ws)
    equal_feedback, equal_ratio = FactorEqualValueRatioEvaluator().evaluate(none_ws, gt_ws)
    corr_feedback, corr_ok = FactorCorrelationEvaluator(hard_check=True).evaluate(none_ws, gt_ws)

    assert "None" in inf_feedback and inf_ok is False
    assert "None" in single_feedback and single_ok is False
    assert "Skip" in output_feedback and output_ok is False
    assert "Skip" in daily_feedback and daily_ok is False
    assert "None" in row_feedback and row_ok is False
    assert "None" in index_feedback and index_ok is False
    assert "None" in missing_feedback and missing_ok is False
    assert "None" in equal_feedback and equal_ratio == -1
    assert "None" in corr_feedback and corr_ok is False

    no_datetime_ws = DummyWorkspace(pd.DataFrame({"factor": [1.0, 2.0]}, index=[0, 1]))
    no_datetime_feedback, no_datetime_ok = FactorDatetimeDailyEvaluator().evaluate(no_datetime_ws, gt_ws)
    assert no_datetime_ok is False
    assert "does not have a datetime index" in no_datetime_feedback

    bad_idx = pd.MultiIndex.from_tuples(
        [("not-a-date", "AAA"), ("still-bad", "BBB")],
        names=["datetime", "instrument"],
    )
    bad_datetime_ws = DummyWorkspace(pd.DataFrame({"factor": [1.0, 2.0]}, index=bad_idx))
    bad_datetime_feedback, bad_datetime_ok = FactorDatetimeDailyEvaluator().evaluate(bad_datetime_ws, gt_ws)
    assert bad_datetime_ok is False
    assert "not in the correct format" in bad_datetime_feedback


def test_correlation_and_value_evaluator_additional_decision_paths():
    gt_df = _daily_df([1.0, 2.0, 3.0, 4.0])
    neg_df = _daily_df([-1.0, -2.0, -3.0, -4.0])
    gt_ws = DummyWorkspace(gt_df)
    neg_ws = DummyWorkspace(neg_df)

    corr_feedback_soft, corr_value = FactorCorrelationEvaluator(hard_check=False).evaluate(neg_ws, gt_ws)
    corr_feedback_hard, corr_ok = FactorCorrelationEvaluator(hard_check=True).evaluate(neg_ws, gt_ws)
    assert "rankic" in corr_feedback_soft
    assert isinstance(corr_value, float)
    assert corr_ok is False
    assert "not sufficiently high correlated" in corr_feedback_hard

    scen_v2 = SimpleNamespace(input_shape=(4, 1))
    gen_v2 = gt_df.copy()
    gen_v2["extra"] = [0.0, 1.0, 2.0, 3.0]
    gt_shifted = gt_df.copy()
    gt_shifted.index = pd.MultiIndex.from_product(
        [pd.date_range("2024-02-01", periods=2), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )

    conclusion_none, decision_none = FactorValueEvaluator(scen=scen_v2).evaluate(
        DummyWorkspace(gen_v2),
        DummyWorkspace(gt_shifted),
        version=2,
    )
    assert decision_none is None
    assert "different index" in conclusion_none
    assert "more columns than input feature" in conclusion_none

    bad_inf = gen_v2.copy()
    bad_inf.iloc[0, 0] = np.inf
    conclusion_false, decision_false = FactorValueEvaluator(scen=scen_v2).evaluate(
        DummyWorkspace(bad_inf),
        DummyWorkspace(gt_shifted),
        version=2,
    )
    assert decision_false is False
    assert "infinite values" in conclusion_false


def test_final_decision_evaluator_retry_and_error_paths(monkeypatch):
    class _Task:
        def get_task_information(self):
            return "task-info"

    class _KeylessAPI:
        token_calls = 0

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_messages_and_calculate_token(self, **kwargs):
            _KeylessAPI.token_calls += 1
            return 999 if _KeylessAPI.token_calls == 1 else 0

        def build_messages_and_create_chat_completion(self, **kwargs):
            return '{"only":"feedback"}'

    monkeypatch.setattr(eva_utils_module, "APIBackend", _KeylessAPI)
    monkeypatch.setattr(eva_utils_module.LLM_SETTINGS, "chat_token_limit", 10)
    evaluator = FactorFinalDecisionEvaluator()
    with pytest.raises(KeyError, match="missing 'final_decision'"):
        evaluator.evaluate(
            target_task=_Task(),
            execution_feedback="x" * 60,
            value_feedback="v",
            code_feedback="c",
        )
    assert _KeylessAPI.token_calls >= 2

    class _BadJsonAPI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_messages_and_calculate_token(self, **kwargs):
            return 0

        def build_messages_and_create_chat_completion(self, **kwargs):
            return "{invalid-json"

    monkeypatch.setattr(eva_utils_module, "APIBackend", _BadJsonAPI)
    with pytest.raises(ValueError, match="Failed to decode JSON response"):
        evaluator.evaluate(
            target_task=_Task(),
            execution_feedback="exec",
            value_feedback="value",
            code_feedback="code",
        )
