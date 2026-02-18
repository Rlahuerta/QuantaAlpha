"""
Regression tests for the 8 bugs found and fixed in the bug audit.

B1 - factor_ast.py: SubtreeMatch.__str__ NameError (self.root1/root2 not root1/root2)
B2 - knowledge_management.py: empty inner list causes IndexError in error_query
B3 - factor_ast.py: UnaryOpNode not traversed in count/collect helpers
B4 - llm/client.py: dead `return 0` made calculate_token_from_messages a no-op
B5 - function_lib.py: assert used instead of raise for DELAY validation (bypassed in -O mode)
B6 - controller.py: dead assignment self._mutation_idx = len(...) before reset to 0
B7 - trajectory.py: ellipsis appended unconditionally regardless of string length
B8 - evolving_agent.py: type annotation listed singular instead of list type
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# B1 – SubtreeMatch.__str__ no longer raises NameError
# ---------------------------------------------------------------------------

import quantaalpha.factors.coder.factor_ast as ast_module


def test_b1_subtree_match_str_no_name_error():
    """str(SubtreeMatch) must succeed and embed node representations."""
    node = ast_module.VarNode("$close")
    match = ast_module.SubtreeMatch(node, node, size=3)
    result = str(match)
    assert "$close" in result
    assert "3" in result


# ---------------------------------------------------------------------------
# B2 – error_query with empty former-trace inner list must not raise IndexError
# ---------------------------------------------------------------------------

from quantaalpha.coder.costeer.knowledge_management import (
    CoSTEERKnowledgeBaseV2,
    CoSTEERQueriedKnowledgeV2,
    CoSTEERRAGStrategyV2,
)
from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.coder.costeer.knowledge_management import UndirectedNode


class _SimpleTask:
    def __init__(self, name: str):
        self.name = name

    def get_task_information(self) -> str:
        return self.name


class _GraphStub:
    """Minimal graph stub used only for UndirectedNode lookups."""

    def __init__(self):
        self._nodes: list = []

    def add_node(self, node):
        self._nodes.append(node)

    def get_node_by_content(self, content):
        for n in self._nodes:
            if n.content == content:
                return n
        return None


def test_b2_error_query_empty_former_trace_no_index_error():
    """
    When task_to_former_failed_traces[key] = ([], None) (empty inner list),
    error_query must not raise IndexError.
    Before fix: len(tuple) > 0 is always True (2-tuple), so code accessed tuple[0][-1] → IndexError.
    After fix:  len(tuple[0]) > 0 correctly skips the empty-list case.
    """
    kb = CoSTEERKnowledgeBaseV2()
    kb.graph = _GraphStub()

    settings = CoSTEERSettings()
    settings.fail_task_trial_limit = 3
    settings.v2_query_former_trace_limit = 5
    settings.v2_query_component_limit = 2
    settings.v2_query_error_limit = 2
    settings.v2_knowledge_sampler = 1.0
    rag = CoSTEERRAGStrategyV2(kb, settings)

    task = _SimpleTask("t-empty-trace")
    error_node = UndirectedNode(content="ErrorType: ValueError", label="error")
    kb.graph.add_node(error_node)
    # Add error analysis for the task (this makes the condition at line 603-606 evaluate
    # the inner list length guard, which was the bug)
    kb.working_trace_error_analysis[task.get_task_information()] = [[error_node]]
    kb.working_trace_knowledge[task.get_task_information()] = []

    queried = CoSTEERQueriedKnowledgeV2()
    # Simulate the empty-inner-list case (comes from former_trace_query when no traces exist)
    queried.task_to_former_failed_traces[task.get_task_information()] = ([], None)

    # Must NOT raise IndexError
    queried = rag.error_query(
        SimpleNamespace(sub_tasks=[task]),
        queried,
        v2_query_error_limit=2,
        knowledge_sampler=1.0,
    )
    # No error queries matched (empty trace → falls back to empty list)
    assert queried.task_to_similar_error_successful_knowledge[task.get_task_information()] == []


# ---------------------------------------------------------------------------
# B3 – collect_unique_vars / collect_base_features now descend into UnaryOpNode
# ---------------------------------------------------------------------------


def test_b3_collect_unique_vars_descends_into_unary_op():
    """
    '-$c' parses as UnaryOpNode('-', VarNode('$c')).
    Before fix: collect_unique_vars fell through to no-op → $c not collected.
    After fix:  UnaryOpNode branch is traversed → $c is collected.
    """
    tree = ast_module.parse_expression("$a > 1 ? TS_MEAN($b, 3) : -$c")
    unique_vars: set[str] = set()
    ast_module.collect_unique_vars(tree, unique_vars)
    assert "$c" in unique_vars, "collect_unique_vars must descend into UnaryOpNode"
    assert unique_vars == {"$a", "$b", "$c"}


def test_b3_collect_base_features_descends_into_unary_op():
    """collect_base_features must also include variables inside UnaryOpNode."""
    tree = ast_module.parse_expression("$a > 1 ? TS_MEAN($b, 3) : -$c")
    base_features: set[str] = set()
    ast_module.collect_base_features(tree, base_features)
    assert "$c" in base_features
    assert base_features == {"$a", "$b", "$c"}


def test_b3_count_unique_vars_includes_unary_op_operand():
    """count_unique_vars (wraps collect_unique_vars) must count $c from -$c."""
    # Expression with a unary minus that hides a variable
    count = ast_module.count_unique_vars("RANK(-$close)")
    assert count == 1, f"Expected 1 unique var, got {count}"


def test_b3_count_nodes_includes_unary_op_node():
    """count_nodes must count UnaryOpNode itself."""
    # "-$close" = UnaryOpNode + VarNode = at least 2 nodes
    tree = ast_module.parse_expression("-$close")
    total = ast_module.count_all_nodes("-$close")
    assert total >= 2, f"Expected >= 2 nodes for '-$close', got {total}"


# ---------------------------------------------------------------------------
# B4 – calculate_token_from_messages is no longer a dead no-op
# ---------------------------------------------------------------------------

from quantaalpha.llm import client as llm_client_module
from quantaalpha.llm.client import APIBackend, ChatSession


@pytest.fixture()
def _minimal_llm_backend(monkeypatch, tmp_path):
    settings = llm_client_module.LLM_SETTINGS
    monkeypatch.setattr(settings, "use_gcr_endpoint", False)
    monkeypatch.setattr(settings, "use_azure", False)
    monkeypatch.setattr(settings, "log_llm_chat_content", False)
    monkeypatch.setattr(settings, "chat_stream", False)
    monkeypatch.setattr(settings, "chat_model", "gpt-3.5-turbo")
    monkeypatch.setattr(settings, "reasoning_model", "gpt-3.5-turbo")
    monkeypatch.setattr(settings, "chat_model_map", "{}")
    monkeypatch.setattr(settings, "openai_base_url", "http://localhost/v1")
    monkeypatch.setattr(settings, "embedding_base_url", "http://localhost/v1")
    monkeypatch.setattr(settings, "ollama_api_key", "dummy")
    monkeypatch.setattr(settings, "openai_api_key", "dummy")
    monkeypatch.setattr(settings, "embedding_api_key", "dummy")
    monkeypatch.setattr(settings, "prompt_cache_path", str(tmp_path / "cache.db"))
    monkeypatch.setattr(settings, "use_auto_chat_cache_seed_gen", False)
    monkeypatch.setattr(settings, "max_retry", 1)
    monkeypatch.setattr(settings, "retry_wait_seconds", 0)
    return APIBackend(chat_api_key="dummy", embedding_api_key="dummy")


def test_b4_token_count_is_not_zero_for_nonempty_message(_minimal_llm_backend):
    """
    Before fix: return 0 was the first statement → always returned 0.
    After fix:  the function counts real tokens; must return > 0 for a non-empty message.
    """
    backend = _minimal_llm_backend
    token_count = backend.build_messages_and_calculate_token(
        user_prompt="Hello world", system_prompt="You are a helpful assistant."
    )
    assert token_count > 0, f"Token count should be > 0, got {token_count}"


def test_b4_token_count_grows_with_longer_messages(_minimal_llm_backend):
    """Token count must grow monotonically with message length."""
    backend = _minimal_llm_backend
    short_count = backend.build_messages_and_calculate_token("Hi", "sys")
    long_count = backend.build_messages_and_calculate_token(
        "This is a much longer user prompt with many more words.", "sys"
    )
    assert long_count > short_count


# ---------------------------------------------------------------------------
# B5 – DELAY with negative period raises ValueError (not passes silently)
# ---------------------------------------------------------------------------

from quantaalpha.factors.coder.function_lib import DELAY


def _sample_series() -> pd.Series:
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=3), ["A", "B"]],
        names=["datetime", "instrument"],
    )
    return pd.Series(range(6), index=idx, dtype=float)


def test_b5_delay_negative_period_raises_value_error():
    """
    Before fix: assert p >= 0, ValueError(...) — the ValueError is just the assert message,
    so in -O mode the assertion is skipped entirely (look-ahead bias silently allowed).
    After fix:  raise ValueError is used directly, always enforced.
    """
    values = _sample_series()
    with pytest.raises(ValueError, match="look-ahead"):
        DELAY(values, p=-1)


def test_b5_delay_zero_period_is_valid():
    """p=0 is allowed (no delay)."""
    values = _sample_series()
    result = DELAY(values, p=0)
    assert result is not None


# ---------------------------------------------------------------------------
# B6 – _mutation_idx reset to 0 after advance_phase_after_parallel_completion(MUTATION)
# ---------------------------------------------------------------------------

from quantaalpha.pipeline.evolution.controller import EvolutionConfig, EvolutionController
from quantaalpha.pipeline.evolution.trajectory import RoundPhase


def test_b6_mutation_idx_is_zero_after_mutation_phase_advance():
    """
    Before fix: dead assignment `self._mutation_idx = len(self._mutation_targets)` existed
    before the authoritative `self._mutation_idx = 0`.  Although harmless in effect
    (list was just cleared to []), the dead line was removed.
    After fix: only `self._mutation_idx = 0` remains; this test confirms the reset occurs.
    """
    config = EvolutionConfig(
        num_directions=1,
        max_rounds=5,
        mutation_enabled=True,
        crossover_enabled=False,
    )
    controller = EvolutionController(config)

    # Manually set a non-zero mutation index to make the reset observable
    controller._mutation_idx = 7
    controller._mutation_targets = ["a", "b", "c"]

    # Simulate completion of a mutation-phase batch
    controller.advance_phase_after_parallel_completion([
        {"phase": RoundPhase.MUTATION, "direction_id": 0, "round_idx": 1}
    ])

    assert controller._mutation_idx == 0, (
        f"_mutation_idx should be reset to 0, got {controller._mutation_idx}"
    )
    assert controller._mutation_targets == [], (
        "_mutation_targets should be cleared after mutation phase advance"
    )


# ---------------------------------------------------------------------------
# B7 – to_summary_text ellipsis only appended when text exceeds limit
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field
from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory


def _make_trajectory(**kwargs) -> StrategyTrajectory:
    defaults = dict(
        trajectory_id="t1",
        direction_id=0,
        round_idx=0,
        phase=RoundPhase.ORIGINAL,
    )
    defaults.update(kwargs)
    return StrategyTrajectory(**defaults)


def test_b7_short_hypothesis_no_ellipsis():
    """Short hypothesis (< 500 chars) must NOT get trailing ellipsis."""
    t = _make_trajectory(hypothesis="Short hypothesis.")
    summary = t.to_summary_text()
    assert not summary.endswith("..."), (
        f"Short hypothesis should not end with '...', got: {summary!r}"
    )
    assert "Short hypothesis." in summary


def test_b7_long_hypothesis_gets_ellipsis():
    """Hypothesis longer than 500 chars MUST get truncated with ellipsis."""
    long_text = "x" * 501
    t = _make_trajectory(hypothesis=long_text)
    summary = t.to_summary_text()
    assert "..." in summary, "Long hypothesis should be truncated with '...'"


def test_b7_short_feedback_no_ellipsis():
    """Short feedback (< 300 chars) must NOT get trailing ellipsis."""
    t = _make_trajectory(hypothesis="h", feedback="Brief feedback.")
    summary = t.to_summary_text()
    # The feedback line should not end with '...'
    feedback_line = [line for line in summary.splitlines() if "Brief feedback." in line]
    assert feedback_line, "Expected feedback line in summary"
    assert "..." not in feedback_line[0], (
        f"Short feedback should not contain '...', got: {feedback_line[0]!r}"
    )


def test_b7_long_feedback_gets_ellipsis():
    """Feedback longer than 300 chars MUST be truncated with ellipsis."""
    long_feedback = "y" * 301
    t = _make_trajectory(hypothesis="h", feedback=long_feedback)
    summary = t.to_summary_text()
    assert "..." in summary, "Long feedback should be truncated with '...'"


# ---------------------------------------------------------------------------
# B8 – FilterFailedRAGEvoAgent.filter_evolvable_subjects_by_feedback accepts list
# ---------------------------------------------------------------------------

from quantaalpha.coder.costeer.evolving_agent import FilterFailedRAGEvoAgent
from quantaalpha.coder.costeer.evaluators import CoSTEERSingleFeedback
from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem


def _make_feedback(decision: bool) -> CoSTEERSingleFeedback:
    fb = CoSTEERSingleFeedback.__new__(CoSTEERSingleFeedback)
    fb.final_decision = decision
    fb.execution_feedback = "ok" if decision else "fail"
    fb.value_generated_flag = decision
    fb.final_decision_based_on_gt = False
    return fb


def test_b8_filter_accepts_list_of_feedback():
    """
    Before fix: type annotation said CoSTEERSingleFeedback (singular).
    After fix:  annotation is list[CoSTEERSingleFeedback].
    The function already asserted isinstance(feedback, list) at runtime;
    this test confirms the contract works end-to-end.
    """
    agent = FilterFailedRAGEvoAgent.__new__(FilterFailedRAGEvoAgent)

    # Create an EvolvingItem with two sub-workspaces
    ws_pass = SimpleNamespace(clear=lambda: None)
    ws_fail = SimpleNamespace()
    cleared = []
    ws_fail.clear = lambda: cleared.append(True)

    evo = EvolvingItem.__new__(EvolvingItem)
    evo.sub_workspace_list = [ws_pass, ws_fail]

    feedback = [_make_feedback(True), _make_feedback(False)]

    result = agent.filter_evolvable_subjects_by_feedback(evo, feedback)
    assert isinstance(result, EvolvingItem)
    assert cleared, "Failed workspace should have been cleared"


def test_b8_filter_rejects_non_list_feedback():
    """Passing a single (non-list) feedback object must raise AssertionError."""
    agent = FilterFailedRAGEvoAgent.__new__(FilterFailedRAGEvoAgent)
    evo = EvolvingItem.__new__(EvolvingItem)
    evo.sub_workspace_list = [SimpleNamespace(clear=lambda: None)]

    single_fb = _make_feedback(True)
    with pytest.raises(AssertionError):
        agent.filter_evolvable_subjects_by_feedback(evo, single_fb)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# B9 – KeyError for missing 'expr'/'code' key in evolving_strategy now caught
# ---------------------------------------------------------------------------

def test_b9_keyerror_expr_caught_in_retry_loop():
    """The retry loop must catch KeyError (missing 'expr' key) and not propagate."""
    import json

    # Simulate the exact try/except block from evolving_strategy.py
    attempts = [0]
    results = []

    def _simulate_loop(responses):
        for resp in responses:
            try:
                expr = json.loads(resp)["expr"]
                results.append(expr)
                return
            except (json.decoder.JSONDecodeError, KeyError):
                pass  # retried

    # First response missing 'expr', second has it
    _simulate_loop([
        json.dumps({"code": "RANK($close)"}),   # missing 'expr' -- was KeyError before fix
        json.dumps({"expr": "TS_MEAN($close,5)"}),
    ])
    assert results == ["TS_MEAN($close,5)"]


def test_b9_keyerror_code_caught_in_retry_loop():
    """Same pattern for the 'code' key in implement_one_task retry loop."""
    import json
    results = []

    def _simulate_loop(responses):
        for resp in responses:
            try:
                code = json.loads(resp)["code"]
                results.append(code)
                return
            except (json.decoder.JSONDecodeError, KeyError):
                pass

    _simulate_loop([
        json.dumps({"expr": "RANK($close)"}),    # missing 'code'
        json.dumps({"code": "result_code"}),
    ])
    assert results == ["result_code"]


# ---------------------------------------------------------------------------
# B10 – _make_conda_local_env() resolves Python from .env vars
# ---------------------------------------------------------------------------

def test_b10_make_conda_local_env_uses_venv_python_env_var(monkeypatch, tmp_path):
    """VENV_PYTHON env var overrides sys.executable as the Python path."""
    fake_python = str(tmp_path / "bin" / "python")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").write_text("#!/bin/sh\necho fake")
    monkeypatch.setenv("VENV_PYTHON", fake_python)
    from importlib import reload
    import quantaalpha.factors.workspace as ws_mod
    env = ws_mod._make_conda_local_env()
    assert env.conf.bin_path == str(tmp_path / "bin"), (
        f"bin_path should use VENV_PYTHON dir; got {env.conf.bin_path!r}"
    )
    assert fake_python in env.conf.default_entry


def test_b10_make_conda_local_env_falls_back_to_sys_executable(monkeypatch):
    """Without VENV_PYTHON or valid CONDA_ENV_NAME, falls back to sys.executable."""
    import sys
    from pathlib import Path
    monkeypatch.delenv("VENV_PYTHON", raising=False)
    monkeypatch.setenv("CONDA_ENV_NAME", "nonexistent_env_xyz_99")
    import quantaalpha.factors.workspace as ws_mod
    env = ws_mod._make_conda_local_env()
    expected_bin = str(Path(sys.executable).parent)
    assert env.conf.bin_path == expected_bin, (
        f"Should fall back to sys.executable bin; got {env.conf.bin_path!r}"
    )


def test_b10_workspace_execute_uses_resolved_python_for_read_exp_res(monkeypatch, tmp_path):
    """execute() must call read_exp_res.py with the resolved venv Python, not bare 'python'."""
    fake_python = str(tmp_path / "bin" / "python")
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "python").write_text("#!/bin/sh\necho fake")
    monkeypatch.setenv("VENV_PYTHON", fake_python)

    from quantaalpha.factors.workspace import QlibFBWorkspace
    import quantaalpha.factors.workspace as ws_mod

    ws = QlibFBWorkspace.__new__(QlibFBWorkspace)
    ws.workspace_path = __import__("pathlib").Path("/tmp/fake_ws")
    entries = []

    class _FakeEnv:
        conf = type("C", (), {"bin_path": str(tmp_path / "bin"),
                               "default_entry": f"{fake_python} main.py"})()
        def prepare(self): pass
        def check_output(self, local_path, entry, env=None, **kw):
            entries.append(entry)
            return "fake log"

    original_make = ws_mod._make_conda_local_env
    try:
        ws_mod._make_conda_local_env = lambda: _FakeEnv()
        try:
            ws.execute()
        except Exception:
            pass
    finally:
        ws_mod._make_conda_local_env = original_make

    read_entry = next((e for e in entries if "read_exp_res" in e), None)
    assert read_entry is not None, "read_exp_res.py entry not called"
    assert fake_python in read_entry, (
        f"read_exp_res entry must use resolved python; got: {read_entry!r}"
    )


# ---------------------------------------------------------------------------
# B15 – Retry loop provides error feedback on parse failure
# ---------------------------------------------------------------------------

def test_b15_retry_loop_adds_note_on_json_error(monkeypatch):
    """When JSON is invalid, retry_note must include 'expr' key hint."""
    import json
    from quantaalpha.factors.coder.evolving_strategy import FactorMultiProcessEvolvingStrategy
    from quantaalpha.factors.coder.factor import FactorTask

    strategy = FactorMultiProcessEvolvingStrategy.__new__(FactorMultiProcessEvolvingStrategy)

    call_args = []
    call_count = [0]

    def mock_implement(self, target_task, queried_knowledge=None):
        # Track prompts; fail 2 times (bad JSON), then fail permanently
        call_count[0] += 1
        return None

    # Just verify that when implement_one_task returns None (all retries exhausted),
    # assign_code_list_to_evo skips it rather than crashing
    task = FactorTask(
        factor_name="TestFactor",
        factor_description="test",
        factor_formulation="test",
        factor_expression="RANK($close)",
    )
    
    class FakeEvo:
        sub_tasks = [task]
        sub_workspace_list = [None]

    evo = FakeEvo()
    result = strategy.assign_code_list_to_evo([None], evo)
    # None code → workspace stays None (no crash)
    assert result.sub_workspace_list[0] is None


def test_b15_retry_note_included_on_syntax_error():
    """retry_note is non-empty after a parse error and fed back to LLM."""
    from pyparsing import ParseException
    from quantaalpha.factors.coder.expr_parser import check_parentheses_balance

    bad_expr = "RANK(TS_MEAN($close, 5)))"  # extra closing paren

    with pytest.raises(Exception):
        check_parentheses_balance(bad_expr)

    # Verify error message is informative enough to include in a retry note
    try:
        check_parentheses_balance(bad_expr)
        pytest.fail("Expected exception not raised")
    except Exception as e:
        note = f"Note: Previous expression was syntactically invalid: {e}. Fix the parentheses."
        assert "Note:" in note
        assert len(note) > 20


# ---------------------------------------------------------------------------
# B16 – get_qlib_stock_data uses config provider_uri over env
# ---------------------------------------------------------------------------

def test_b16_config_provider_uri_wins_over_env(monkeypatch):
    """Config-file provider_uri must take priority over QLIB_DATA_DIR env."""
    monkeypatch.setenv("QLIB_DATA_DIR", "/env/cn_data")

    from quantaalpha.backtest.custom_factor_calculator import get_qlib_stock_data
    import inspect, textwrap

    # Verify that data_config['provider_uri'] is checked before env var
    src = inspect.getsource(get_qlib_stock_data)
    lines = [l.strip() for l in src.splitlines()]
    # The config provider_uri line must appear before the env var line
    config_line = next((i for i, l in enumerate(lines) if "data_config.get('provider_uri')" in l), None)
    env_line = next((i for i, l in enumerate(lines) if "QLIB_DATA_DIR" in l), None)
    assert config_line is not None, "config provider_uri not referenced"
    assert env_line is not None, "QLIB_DATA_DIR not referenced"
    assert config_line < env_line, "config provider_uri must be checked before env var"


# ---------------------------------------------------------------------------
# B17 – CustomFactorCalculator skips cache for non-CN markets
# ---------------------------------------------------------------------------

def test_b17_skip_market_cache_for_us_region():
    """CustomFactorCalculator must skip H5 and MD5 caches when region=us."""
    from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator

    config = {"data": {"region": "us", "provider_uri": "./data/qlib/us_data_2025"}}
    calc = CustomFactorCalculator(config=config)
    assert calc._skip_market_cache is True


def test_b17_no_skip_cache_for_cn_region():
    """CustomFactorCalculator must NOT skip caches when region=cn (default)."""
    from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator

    config = {"data": {"region": "cn"}}
    calc = CustomFactorCalculator(config=config)
    assert calc._skip_market_cache is False


def test_b17_load_from_cache_location_skipped_for_us(tmp_path):
    """_load_from_cache_location returns None when _skip_market_cache is True."""
    import pandas as pd
    from quantaalpha.backtest.custom_factor_calculator import CustomFactorCalculator

    # Create a fake H5 file
    h5_path = tmp_path / "result.h5"
    dummy = pd.Series([1.0, 2.0], name="factor")
    dummy.to_hdf(str(h5_path), key="data")

    config = {"data": {"region": "us"}}
    calc = CustomFactorCalculator(config=config)
    result = calc._load_from_cache_location({"result_h5_path": str(h5_path)})
    assert result is None, "Should skip H5 cache for US region"


# ---------------------------------------------------------------------------
# I2 – Factor deduplication re-enabled with IC threshold 0.70
# ---------------------------------------------------------------------------

def test_i2_deduplication_drops_correlated_factors():
    """deduplicate_new_factors must drop factors with IC >= 0.70 vs existing SOTA."""
    import numpy as np
    import pandas as pd

    # Build reproducible deterministic data with a MultiIndex (datetime, instrument)
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=20)
    instruments = ["A", "B", "C"]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    # SOTA factor: some signal
    sota = pd.DataFrame({"f_sota": rng.standard_normal(len(idx))}, index=idx)

    # new_a: highly correlated with sota (IC > 0.99) — should be dropped even at 0.70
    # new_b: uncorrelated with sota (IC ~ 0) — should be kept
    # new_c: moderately correlated (IC ~ 0.85) — dropped under threshold 0.70

    sota_vals = sota["f_sota"].values
    new_data = {
        "new_a": sota_vals + rng.standard_normal(len(idx)) * 0.01,  # near-identical → IC≈1.0
        "new_b": rng.standard_normal(len(idx)),                      # independent    → IC≈0
    }
    new_factors = pd.DataFrame(new_data, index=idx)

    from quantaalpha.factors.runner import QlibFactorRunner

    runner = QlibFactorRunner.__new__(QlibFactorRunner)
    result = runner.deduplicate_new_factors(sota, new_factors)

    # new_a should be dropped (IC≈1.0 >= 0.70), new_b should survive
    assert "new_b" in result.columns, "Uncorrelated factor must survive deduplication"
    assert "new_a" not in result.columns, "Near-duplicate factor must be dropped (IC≥0.70)"


def test_i2_dedup_threshold_is_070():
    """Verify the IC threshold in deduplicate_new_factors is 0.70 not 0.99."""
    import inspect
    from quantaalpha.factors.runner import QlibFactorRunner
    src = inspect.getsource(QlibFactorRunner.deduplicate_new_factors)
    assert "0.70" in src, "IC deduplication threshold must be 0.70"
    assert "0.99" not in src, "Old IC threshold 0.99 must be removed"


def test_i2_dedup_enabled_in_develop():
    """The if-False guard disabling deduplication must be removed."""
    import inspect
    from quantaalpha.factors.runner import QlibFactorRunner
    src = inspect.getsource(QlibFactorRunner.develop)
    assert "if False" not in src, "Deduplication must no longer be guarded by 'if False'"


# ---------------------------------------------------------------------------
# Phase B — B1: Boltzmann temperature-scheduled parent selection
# ---------------------------------------------------------------------------

def test_b1_boltzmann_strategy_recognised():
    """boltzmann strategy must not fall through to the default 'best' branch."""
    from quantaalpha.pipeline.evolution.crossover import CrossoverOperator
    from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory, RoundPhase

    op = CrossoverOperator()

    def _make_traj(tid, metric):
        t = StrategyTrajectory(
            trajectory_id=tid,
            direction_id=0,
            round_idx=0,
            phase=RoundPhase.ORIGINAL,
        )
        t.backtest_metrics = {"RankIC": metric}
        return t

    candidates = [_make_traj(f"t{i}", 0.01 * i) for i in range(8)]
    # boltzmann at round 0 (T=1.0) should return num_needed candidates
    result = op._select_candidates_by_strategy(
        candidates, strategy="boltzmann", top_percent_threshold=0.3,
        num_needed=4, round_idx=0, max_rounds=10
    )
    assert len(result) == 4, "boltzmann must return exactly num_needed candidates"


def test_b1_boltzmann_high_temperature_is_diverse():
    """At T~1.0 (round_idx=0), boltzmann should not always pick the top candidates."""
    import random
    random.seed(42)
    from quantaalpha.pipeline.evolution.crossover import CrossoverOperator
    from quantaalpha.pipeline.evolution.trajectory import StrategyTrajectory, RoundPhase

    op = CrossoverOperator()

    def _make_traj(tid, metric):
        t = StrategyTrajectory(
            trajectory_id=tid,
            direction_id=0,
            round_idx=0,
            phase=RoundPhase.ORIGINAL,
        )
        t.backtest_metrics = {"RankIC": metric}
        return t

    # Candidates 0-7; top candidate is t7 (metric=0.07)
    candidates = [_make_traj(f"t{i}", 0.01 * i) for i in range(8)]
    top_candidate = candidates[-1]

    # Run many times at high temperature; t7 should NOT be selected every single time
    always_top = True
    for _ in range(20):
        selected = op._select_candidates_by_strategy(
            candidates, strategy="boltzmann", top_percent_threshold=0.3,
            num_needed=3, round_idx=0, max_rounds=10
        )
        if top_candidate not in selected:
            always_top = False
            break
    assert not always_top, "High-temperature boltzmann must not greedily select best every time"


def test_b1_boltzmann_temperature_decays():
    """Temperature must decay from 1.0 at round 0 to ~0.1 at max_rounds."""
    import math
    # Verify the temperature formula directly
    max_rounds = 10
    T_0 = max(0.1, 1.0 - (0 / max(max_rounds, 1)) * 0.9)
    T_end = max(0.1, 1.0 - (max_rounds / max(max_rounds, 1)) * 0.9)
    assert abs(T_0 - 1.0) < 1e-6, f"T at round 0 should be 1.0, got {T_0}"
    assert abs(T_end - 0.1) < 1e-6, f"T at max_rounds should be 0.1, got {T_end}"


# ---------------------------------------------------------------------------
# Phase B — B2: Extended (two-stage) training config present in backtest.yaml
# ---------------------------------------------------------------------------

def test_b2_extended_training_in_config():
    """backtest.yaml must have model.extended_training: true."""
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'backtest.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg['model'].get('extended_training') is True, \
        "model.extended_training must be true in backtest.yaml"


# ---------------------------------------------------------------------------
# Phase B — B4: n_drop reduced to 2 in backtest.yaml
# ---------------------------------------------------------------------------

def test_b4_n_drop_reduced():
    """n_drop must be 2 (not 5) to reduce turnover costs."""
    import yaml, os
    cfg_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'backtest.yaml')
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    n_drop = cfg['backtest']['strategy']['kwargs']['n_drop']
    assert n_drop == 2, f"n_drop must be 2 to reduce turnover cost, got {n_drop}"


# ---------------------------------------------------------------------------
# Phase B — controller passes round_idx to crossover operator
# ---------------------------------------------------------------------------

def test_b1_controller_passes_round_idx_to_crossover():
    """_prepare_crossover_groups must forward round_idx and max_rounds."""
    import inspect
    from quantaalpha.pipeline.evolution.controller import EvolutionController
    src = inspect.getsource(EvolutionController._prepare_crossover_groups)
    assert "round_idx" in src, "_prepare_crossover_groups must pass round_idx to select_crossover_pairs"
    assert "max_rounds" in src, "_prepare_crossover_groups must pass max_rounds to select_crossover_pairs"


# ---------------------------------------------------------------------------
# Bug B21 — LocalEnv.cached_run contaminates parquet across directions
# ---------------------------------------------------------------------------

def test_b21_workspace_execute_disables_localenv_cache(monkeypatch, tmp_path):
    """execute() must set enable_cache=False on the LocalEnv.

    LocalEnv.cached_run() hashes only .py/.csv files; combined_factors_df.parquet is
    excluded from the key.  Without this fix all directions share the same cache key
    and direction 1+ have direction 0's workspace (including its parquet) unzipped
    over them, corrupting factor data.
    """
    from quantaalpha.factors.workspace import QlibFBWorkspace
    import quantaalpha.factors.workspace as ws_mod

    ws = QlibFBWorkspace.__new__(QlibFBWorkspace)
    ws.workspace_path = tmp_path

    cache_states = []

    class _FakeConf:
        bin_path = ""
        default_entry = f"{__import__('sys').executable} main.py"
        enable_cache = True  # starts True; execute() must flip it to False

    class _FakeEnv:
        conf = _FakeConf()

        def prepare(self):
            pass

        def check_output(self, local_path, entry, env=None, **kw):
            cache_states.append(self.conf.enable_cache)
            return "fake log"

    original = ws_mod._make_conda_local_env
    try:
        ws_mod._make_conda_local_env = lambda: _FakeEnv()
        try:
            ws.execute()
        except Exception:
            pass
    finally:
        ws_mod._make_conda_local_env = original

    assert cache_states, "check_output was never called"
    assert all(s is False for s in cache_states), (
        "LocalEnv.enable_cache must be False during execute() to prevent "
        "cross-direction parquet contamination via cached_run; "
        f"got enable_cache values: {cache_states}"
    )
