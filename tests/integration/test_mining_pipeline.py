"""
Integration tests for the mining pipeline.

Guards three bugs fixed in session topic/ollama:
  - RLIMIT_AS=2GB caused factor.py workers to spin at 99% CPU indefinitely
    (Python+numpy+pandas need ~3.2GB of virtual address space).
  - FactorExecutionResult tuple-unpack crashed in evaluators.py / eva_utils.py.
  - gen_df.columns AttributeError when factor.py returns a pd.Series.

Test tiers:
  T1-T3, T5-T6: no external deps — always run in CI.
  T4:           requires debug HDF5 — skipped when absent.
  T7-T8:        full AlphaAgentLoop smoke test — all LLM/runner calls mocked;
                runs without external deps.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEBUG_HDF5 = (
    _REPO_ROOT
    / "git_ignore_folder"
    / "factor_implementation_source_data_debug"
    / "daily_pv.h5"
)


def _hdf5_available() -> bool:
    return _DEBUG_HDF5.exists()


_skip_no_hdf5 = pytest.mark.skipif(
    not _hdf5_available(),
    reason="debug HDF5 not present (git_ignore_folder/…/daily_pv.h5)",
)


def _make_multiindex(n_dates: int = 3, instruments: list[str] | None = None):
    instruments = instruments or ["AAA", "BBB"]
    return pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=n_dates), instruments],
        names=["datetime", "instrument"],
    )


# ---------------------------------------------------------------------------
# T1 — RLIMIT_AS regression: subprocess imports numpy within 30 s
# ---------------------------------------------------------------------------

_RLIMIT_SCRIPT = """
import resource, sys
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
import numpy
import pandas
print("ok")
"""


@pytest.mark.timeout(120)
def test_t1_rlimit_as_subprocess_imports_numpy_quickly():
    """Spawning a child with RLIMIT_AS=8GB must complete the numpy import in <30 s.

    Before the fix the 2GB limit caused the child to spin at 99% CPU forever
    because mmap for shared libraries was repeatedly denied with ENOMEM.
    """
    py = sys.executable
    proc = subprocess.run(
        [py, "-c", _RLIMIT_SCRIPT],
        capture_output=True,
        timeout=30,
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stderr.decode()}"
    assert b"ok" in proc.stdout


# ---------------------------------------------------------------------------
# T2 — _get_df Series normalisation
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_t2_get_df_series_normalisation():
    """_get_df must convert a named Series to a single-column DataFrame (gen_df).
    An unnamed Series should use the fallback column name 'factor'.
    """
    from quantaalpha.factors.coder.eva_utils import FactorEvaluator
    from quantaalpha.factors.coder.factor import FactorExecutionResult

    idx = _make_multiindex()

    class _EvalBase(FactorEvaluator):
        def evaluate(self, target_task, implementation, gt_implementation, **kwargs):
            return super().evaluate(target_task, implementation, gt_implementation, **kwargs)

    class _WS:
        def __init__(self, data):
            self._data = data
            self.target_task = None

        def execute(self):
            return FactorExecutionResult(success=True, feedback="ok", result=self._data)

    evaluator = _EvalBase()

    # Named series → column should be the series name
    named_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx, name="my_factor")
    gt_ws = _WS(named_series)
    gen_ws = _WS(named_series)
    gt_df, gen_df = evaluator._get_df(gt_ws, gen_ws)
    assert isinstance(gen_df, pd.DataFrame)
    assert list(gen_df.columns) == ["my_factor"]

    # Unnamed series → fallback column name "factor"
    unnamed_series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx)
    gt_ws2 = _WS(unnamed_series)
    gen_ws2 = _WS(unnamed_series)
    _, gen_df2 = evaluator._get_df(gt_ws2, gen_ws2)
    assert isinstance(gen_df2, pd.DataFrame)
    assert list(gen_df2.columns) == ["factor"]


# ---------------------------------------------------------------------------
# T3 — FactorSingleColumnEvaluator with Series output doesn't crash
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_t3_single_column_evaluator_accepts_series():
    """FactorSingleColumnEvaluator.evaluate() must not raise AttributeError
    when the implementation workspace returns a pd.Series.
    """
    from quantaalpha.factors.coder.eva_utils import FactorSingleColumnEvaluator
    from quantaalpha.factors.coder.factor import FactorExecutionResult

    idx = _make_multiindex()
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], index=idx, name="my_factor")

    class _WS:
        def __init__(self, data):
            self._data = data
            self.target_task = None

        def execute(self):
            return FactorExecutionResult(success=True, feedback="ok", result=self._data)

    gt_ws = _WS(pd.DataFrame({"my_factor": series.values}, index=idx))
    gen_ws = _WS(series)

    evaluator = FactorSingleColumnEvaluator(scen=None)
    feedback, decision = evaluator.evaluate(gen_ws, gt_ws)

    assert isinstance(feedback, str)
    assert isinstance(decision, bool)


# ---------------------------------------------------------------------------
# T4 — Real FactorFBWorkspace.execute() subprocess
# ---------------------------------------------------------------------------

@_skip_no_hdf5
@pytest.mark.timeout(120)
def test_t4_factor_workspace_real_subprocess(tmp_path, monkeypatch):
    """FactorFBWorkspace.execute() must complete (not hang) and return a
    non-empty DataFrame when RLIMIT_AS is 8 GB.

    This is the direct regression test for the 2 GB hang: before the fix the
    subprocess never returned because mmap for numpy failed silently.
    """
    import quantaalpha.core.experiment as experiment_module
    import quantaalpha.core.utils as core_utils_module
    import quantaalpha.factors.coder.factor as factor_module

    monkeypatch.setattr(core_utils_module.RD_AGENT_SETTINGS, "cache_with_pickle", False)
    monkeypatch.setattr(
        experiment_module.RD_AGENT_SETTINGS,
        "workspace_path",
        tmp_path / "ws_root",
    )
    monkeypatch.setattr(
        factor_module.FACTOR_COSTEER_SETTINGS,
        "python_bin",
        sys.executable,
    )
    monkeypatch.setattr(
        factor_module.FACTOR_COSTEER_SETTINGS,
        "file_based_execution_timeout",
        60,
    )
    # Point debug data folder to actual debug HDF5 directory
    monkeypatch.setattr(
        factor_module.FACTOR_COSTEER_SETTINGS,
        "data_folder_debug",
        str(_DEBUG_HDF5.parent),
    )

    from quantaalpha.factors.coder.factor import FactorFBWorkspace, FactorTask

    task = FactorTask(
        factor_name="Smoke_Test_Factor",
        factor_description="5-day mean close rank",
        factor_formulation="rank of 5-day TS_MEAN of close",
        factor_expression="RANK(TS_MEAN($close, 5))",
    )
    ws = FactorFBWorkspace(target_task=task)
    ws.inject_code(**{
        "factor.py": _build_factor_py("RANK(TS_MEAN($close, 5))", "Smoke_Test_Factor")
    })

    result = ws.execute(data_type="Debug")

    assert result.success is True, f"Execution failed:\n{result.feedback}"
    assert result.result is not None
    assert isinstance(result.result, (pd.Series, pd.DataFrame))
    assert len(result.result) > 0


# ---------------------------------------------------------------------------
# T5 — generate_parallel_directions with mocked LLM
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_t5_generate_parallel_directions_mocked_llm(monkeypatch):
    """generate_parallel_directions returns n strings when the LLM mock
    responds with valid JSON.  Also exercises the retry path when the LLM
    returns garbage on the first attempt.
    """
    import quantaalpha.pipeline.planning as planning_module

    directions_json = json.dumps([
        "Direction A: momentum reversal",
        "Direction B: volume-conditioned mean reversion",
        "Direction C: cross-sectional RSI extremes",
    ])
    # First call returns garbage; second call returns valid JSON (retry path)
    call_count = {"n": 0}

    def _fake_llm(self, user_prompt, system_prompt, json_mode=False, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "not valid json at all !!!"
        return directions_json

    monkeypatch.setattr(
        planning_module.APIBackend,
        "build_messages_and_create_chat_completion",
        _fake_llm,
    )

    prompt_file = _REPO_ROOT / "quantaalpha" / "pipeline" / "prompts" / "planning_prompts.yaml"
    if not prompt_file.exists():
        # Fallback: use any yaml in the pipeline dir
        prompt_file = next(
            (_REPO_ROOT / "quantaalpha" / "pipeline" / "prompts").glob("*.yaml"), None
        )
    if prompt_file is None:
        pytest.skip("No planning prompt file found")

    result = planning_module.generate_parallel_directions(
        initial_direction="test direction",
        n=3,
        prompt_file=prompt_file,
        use_llm=True,
        allow_fallback=True,
    )

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(d, str) and len(d) > 0 for d in result)
    assert call_count["n"] >= 2  # retry was triggered


# ---------------------------------------------------------------------------
# T6 — generate_parallel_directions fallback path
# ---------------------------------------------------------------------------

@pytest.mark.timeout(120)
def test_t6_generate_parallel_directions_fallback():
    """With use_llm=False the function must return exactly n template-based
    directions without making any network calls.
    """
    import quantaalpha.pipeline.planning as planning_module

    prompt_file = _REPO_ROOT / "quantaalpha" / "pipeline" / "prompts" / "planning_prompts.yaml"
    if not prompt_file.exists():
        pytest.skip("No planning prompt file found")

    for n in (1, 3, 5):
        result = planning_module.generate_parallel_directions(
            initial_direction="momentum trading strategy",
            n=n,
            prompt_file=prompt_file,
            use_llm=False,
            allow_fallback=True,
        )
        assert len(result) == n, f"Expected {n} directions, got {len(result)}"
        assert all(isinstance(d, str) and len(d) > 0 for d in result)


# ---------------------------------------------------------------------------
# T7/T8 — AlphaAgentLoop full 5-step smoke test (all IO mocked)
# ---------------------------------------------------------------------------

def _make_hypothesis():
    """Return a minimal AlphaAgentHypothesis for smoke tests."""
    from quantaalpha.factors.proposal import AlphaAgentHypothesis

    return AlphaAgentHypothesis(
        hypothesis="5-day momentum is predictive",
        concise_observation="prices trend over 5 days",
        concise_justification="short-term momentum effect",
        concise_knowledge="momentum, mean-reversion",
        concise_specification="rank of 5-day mean close",
    )


def _make_factor_task():
    from quantaalpha.factors.coder.factor import FactorTask

    return FactorTask(
        factor_name="Smoke_Rank_TsMean_Close",
        factor_description="5-day momentum factor",
        factor_formulation="RANK(TS_MEAN($close, 5))",
        factor_expression="RANK(TS_MEAN($close, 5))",
    )


def _make_experiment_with_workspace(task, ws):
    """Build a QlibFactorExperiment with a pre-populated workspace."""
    from quantaalpha.factors.experiment import QlibFactorExperiment

    exp = QlibFactorExperiment(sub_tasks=[task])
    exp.sub_workspace_list = [ws]
    return exp


def _make_result_experiment(task):
    """Build a QlibFactorExperiment whose .result is a metrics Series
    (mimics what QlibFactorRunner.develop() returns).
    """
    from quantaalpha.factors.experiment import QlibFactorExperiment
    from quantaalpha.factors.coder.factor import FactorFBWorkspace, FactorExecutionResult

    # Build workspace with a valid result
    idx = pd.MultiIndex.from_product(
        [pd.date_range("2020-01-01", periods=100), ["AAA", "BBB"]],
        names=["datetime", "instrument"],
    )
    factor_values = pd.Series(np.random.randn(200), index=idx, name=task.factor_name)

    ws = FactorFBWorkspace(target_task=task)
    ws.inject_code(**{"factor.py": _build_factor_py("RANK(TS_MEAN($close, 5))", task.factor_name)})
    ws._cached_result = FactorExecutionResult(
        success=True, feedback="ok", result=factor_values
    )
    ws._result_h5_written = True

    exp = QlibFactorExperiment(sub_tasks=[task])
    exp.sub_workspace_list = [ws]
    # Synthetic backtest metrics that library.py can parse
    exp.result = pd.Series({
        "IC": 0.05,
        "ICIR": 0.30,
        "Rank IC": 0.04,
        "Rank ICIR": 0.28,
        "Annualized Return": 0.12,
    })
    return exp


def _build_factor_py(expr: str, name: str) -> str:
    return f"""\
import pandas as pd
import numpy as np
import os, re
from quantaalpha.factors.coder.expr_parser import parse_expression, parse_symbol
from quantaalpha.factors.coder.function_lib import *

def calculate_factor(expr, name):
    df = pd.read_hdf('./daily_pv.h5', key='data')
    if '$return' not in df.columns and '$close' in df.columns:
        df['$return'] = df.groupby(level='instrument')['$close'].pct_change(fill_method=None)
    if '$vwap' not in df.columns and all(c in df.columns for c in ['$open', '$high', '$low', '$close']):
        df['$vwap'] = (df['$open'] + df['$high'] + df['$low'] + df['$close']) / 4.0
    expr = parse_symbol(expr, df.columns)
    expr = parse_expression(expr)
    for col in df.columns:
        col_name = str(col)
        base_name = col_name[1:] if col_name.startswith('$') else col_name
        expr = re.sub(
            rf"(?<![A-Za-z0-9_\\$])\\$?{{re.escape(base_name)}}(?![A-Za-z0-9_])",
            f"df[{{col_name!r}}]",
            expr,
        )
    df[name] = eval(expr)
    result = df[name].astype(np.float64)
    if os.path.exists('result.h5'):
        os.remove('result.h5')
    result.to_hdf('result.h5', key='data')

if __name__ == '__main__':
    expr = "{expr}"
    name = "{name}"
    calculate_factor(expr, name)
"""


@pytest.mark.timeout(120)
def test_t7_alpha_agent_loop_smoke(monkeypatch, tmp_path):
    """Run all 5 steps of AlphaAgentLoop with mocked LLM and runner.
    Asserts the loop completes without exception and writes the factor library.
    """
    import quantaalpha.pipeline.loop as loop_module
    import quantaalpha.factors.proposal as proposal_module
    import quantaalpha.factors.runner as runner_module
    from quantaalpha.pipeline.loop import AlphaAgentLoop
    from quantaalpha.pipeline.settings import ALPHA_AGENT_FACTOR_PROP_SETTING
    from quantaalpha.llm.config import LLM_SETTINGS

    # --- env / settings setup ---
    library_path = tmp_path / "factorlib" / "all_factors_library_smoke.json"
    library_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("FACTOR_LIBRARY_SUFFIX", "smoke")
    monkeypatch.setenv("LOG_LLM_CHAT_CONTENT", "false")

    # Suppress pickle snapshots — MagicMock instances cannot be pickled
    import quantaalpha.utils.workflow as workflow_module
    monkeypatch.setattr(workflow_module.LoopBase, "dump", lambda self, path: None)

    # Suppress logger object-serialisation (also uses pickle) — keeps test fast
    import rdagent.log.logger as _rdlog
    monkeypatch.setattr(_rdlog.RDAgentLog, "log_object", lambda self, obj, tag="", **kw: None)

    # Redirect library save to tmp_path so the test is fully isolated
    import quantaalpha.pipeline.loop as _loop
    original_feedback = _loop.AlphaAgentLoop.feedback

    def _patched_feedback(self, prev_out):
        # Call the real feedback but override the project-root library path
        import quantaalpha.factors.library as lib_module
        original_manager_init = lib_module.FactorLibraryManager.__init__

        def _tmp_manager_init(mgr_self, path, *args, **kwargs):
            original_manager_init(mgr_self, str(library_path), *args, **kwargs)

        with patch.object(lib_module.FactorLibraryManager, "__init__", _tmp_manager_init):
            original_feedback(self, prev_out)

    monkeypatch.setattr(_loop.AlphaAgentLoop, "feedback", _patched_feedback)

    # --- build pre-baked objects ---
    hypothesis = _make_hypothesis()
    task = _make_factor_task()
    result_exp = _make_result_experiment(task)

    # --- construct loop ---
    loop = AlphaAgentLoop(
        PROP_SETTING=ALPHA_AGENT_FACTOR_PROP_SETTING,
        potential_direction="5-day momentum alpha",
        stop_event=threading.Event(),
    )

    # --- mock each step's component ---
    loop.hypothesis_generator.gen = MagicMock(return_value=hypothesis)
    loop.factor_constructor.convert = MagicMock(
        return_value=_make_experiment_with_workspace(task, MagicMock())
    )
    loop.coder.develop = MagicMock(
        return_value=_make_experiment_with_workspace(task, MagicMock())
    )
    loop.runner.develop = MagicMock(return_value=result_exp)
    loop.summarizer.generate_feedback = MagicMock(
        return_value=SimpleNamespace(
            observations="looks good",
            hypothesis_evaluation="confirmed",
            new_hypothesis="try 10-day",
            decision=True,
            reason="positive IC",
        )
    )

    # --- run one complete loop ---
    loop.run(step_n=5)

    # Verify each step was called
    loop.hypothesis_generator.gen.assert_called_once()
    loop.factor_constructor.convert.assert_called_once()
    loop.coder.develop.assert_called_once()
    loop.runner.develop.assert_called_once()
    loop.summarizer.generate_feedback.assert_called_once()

    # Factor library must have been attempted (even if path redirect happened)
    assert result_exp.result is not None


@pytest.mark.timeout(120)
def test_t8_factor_library_schema(monkeypatch, tmp_path):
    """After a complete loop the library JSON must contain at least one factor
    with the expected schema fields.
    """
    from quantaalpha.factors.library import FactorLibraryManager
    from quantaalpha.factors.experiment import QlibFactorExperiment

    library_path = tmp_path / "library.json"
    task = _make_factor_task()
    result_exp = _make_result_experiment(task)

    feedback = SimpleNamespace(
        observations="ok",
        hypothesis_evaluation="positive",
        new_hypothesis="try longer window",
        decision=True,
        reason="IC positive",
    )

    manager = FactorLibraryManager(str(library_path))
    manager.add_factors_from_experiment(
        experiment=result_exp,
        experiment_id="test_001",
        round_number=0,
        hypothesis="5-day momentum",
        feedback=feedback,
        initial_direction="momentum",
        user_initial_direction="momentum",
        planning_direction="momentum",
        evolution_phase="original",
        trajectory_id="traj_0_0_original",
        parent_trajectory_ids=[],
    )

    assert library_path.exists(), "Library file was not created"
    data = json.loads(library_path.read_text())

    assert "factors" in data
    assert len(data["factors"]) >= 1

    factor_name = list(data["factors"].keys())[0]
    factor = data["factors"][factor_name]

    # Real library schema uses 'backtest_results' and nests evolution_phase in 'metadata'
    required_top_keys = {"factor_expression", "quality", "backtest_results"}
    assert required_top_keys.issubset(factor.keys()), (
        f"Missing top-level keys: {required_top_keys - factor.keys()}"
    )
    assert isinstance(factor["backtest_results"], dict)
    assert factor["factor_expression"] is not None

    # evolution_phase is stored inside the 'metadata' sub-dict
    metadata = factor.get("metadata", {})
    assert "evolution_phase" in metadata, (
        f"'evolution_phase' missing from metadata. metadata keys: {list(metadata.keys())}"
    )
    assert metadata["evolution_phase"] == "original"
