from types import SimpleNamespace

import numpy as np
import pandas as pd

import quantaalpha.factors.coder.eva_utils as eva_utils_module
from quantaalpha.factors.coder.eva_utils import (
    FactorCorrelationEvaluator,
    FactorDatetimeDailyEvaluator,
    FactorEqualValueRatioEvaluator,
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
        return None, self._df


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
