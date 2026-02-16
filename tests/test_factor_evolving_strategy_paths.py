from __future__ import annotations

from types import SimpleNamespace

import quantaalpha.factors.coder.evolving_strategy as es_module
from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.coder.costeer.knowledge_management import CoSTEERQueriedKnowledgeV2
from quantaalpha.core.scenario import Scenario
from quantaalpha.factors.coder.factor import FactorTask
from quantaalpha.factors.coder.evolving_strategy import (
    FactorMultiProcessEvolvingStrategy,
    FactorParsingStrategy,
    FactorRunningStrategy,
)


class _Scenario(Scenario):
    @property
    def background(self) -> str:
        return "bg"

    @property
    def interface(self) -> str:
        return "itf"

    @property
    def output_format(self) -> str:
        return "out"

    @property
    def simulator(self) -> str:
        return "sim"

    @property
    def rich_style_description(self) -> str:
        return "rich"

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None) -> str:  # noqa: ANN001, ANN201
        return "scenario-desc"


class _FakeWorkspace:
    def __init__(self, target_task=None):  # noqa: ANN001
        self.target_task = target_task
        self.code_dict = {}

    def inject_code(self, **files):
        self.code_dict.update(files)


class _FakeKnowledge:
    def __init__(self, code: str = "expr = 'TS_MEAN($close, 2)'"):
        self.target_task = SimpleNamespace(
            get_task_description=lambda: "good factor",
            get_task_information=lambda: "good factor info",
        )
        self.implementation = SimpleNamespace(code=code)
        self.feedback = "former feedback"

    def get_implementation_and_feedback_str(self):
        return "implementation-and-feedback"


def _task(name: str, expr: str = "TS_MEAN($close, 2)"):
    return FactorTask(
        factor_name=name,
        factor_description=f"{name}-desc",
        factor_formulation=f"{name}-form",
        factor_expression=expr,
        version=1,
    )


def _settings():
    s = CoSTEERSettings()
    s.select_threshold = 10
    return s


def _queried(task: FactorTask, *, former=None, latest=None, success=None, errors=None):
    info = task.get_task_information()
    return CoSTEERQueriedKnowledgeV2(
        success_task_to_knowledge_dict={},
        failed_task_info_set=set(),
        task_to_former_failed_traces={info: (former or [], latest)},
        task_to_similar_task_successful_knowledge={info: success or []},
        task_to_similar_error_successful_knowledge={info: errors or []},
    )


def _patch_fake_api(monkeypatch, *, token_values=None, completion_values=None, fallback_completion='{"code":"x"}'):
    class _FakeAPIBackend:
        _token_values = list(token_values or [1])
        _completion_values = list(completion_values or [fallback_completion])

        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def build_messages_and_calculate_token(self, **kwargs):  # noqa: ANN201
            if self._token_values:
                return self._token_values.pop(0)
            return 1

        def build_messages_and_create_chat_completion(self, **kwargs):  # noqa: ANN201
            if self._completion_values:
                return self._completion_values.pop(0)
            return fallback_completion

    monkeypatch.setattr(es_module, "APIBackend", _FakeAPIBackend)
    return _FakeAPIBackend


def _patch_factor_prompts(monkeypatch):
    monkeypatch.setitem(
        es_module.implement_prompts,
        "evolving_strategy_error_summary_v2_system",
        "{{scenario}} {{factor_information_str}} {{code_and_feedback}}",
    )
    monkeypatch.setitem(
        es_module.implement_prompts,
        "evolving_strategy_error_summary_v2_user",
        "{{queried_similar_error_knowledge}}",
    )
    monkeypatch.setitem(
        es_module.implement_prompts,
        "evolving_strategy_factor_implementation_v1_system",
        "{{scenario}} {{queried_former_failed_knowledge|length}}",
    )
    monkeypatch.setitem(
        es_module.implement_prompts,
        "evolving_strategy_factor_implementation_v2_user",
        "{{factor_information_str}} {{queried_similar_error_knowledge}} {{error_summary_critics}} "
        "{{similar_successful_factor_description}} {{similar_successful_expression}} "
        "{{latest_attempt_to_latest_successful_execution}}",
    )


def test_factor_multi_process_error_summary_and_implement(monkeypatch):
    monkeypatch.setattr(es_module.LLM_SETTINGS, "chat_token_limit", 100)
    _patch_factor_prompts(monkeypatch)
    _patch_fake_api(
        monkeypatch,
        token_values=[1000, 50, 50],
        completion_values=["error-summary", "not-json", '{"code":"print(1)"}'],
    )

    scen = _Scenario()
    strategy = FactorMultiProcessEvolvingStrategy(scen=scen, settings=_settings())
    strategy.extract_expr = lambda code: "TS_MEAN($close, 2)"
    target_task = _task("factor-a")
    former = [_FakeKnowledge("expr = 'BAD1'"), _FakeKnowledge("expr = 'BAD2'")]
    error_pairs = [("err", (_FakeKnowledge("expr = 'BAD3'"), _FakeKnowledge("expr = 'GOOD'")))]
    queried = _queried(target_task, former=former, success=[], errors=error_pairs, latest="latest")

    summary = strategy.error_summary(target_task, former, error_pairs)
    code = strategy.implement_one_task(target_task, queried)

    assert summary == "error-summary"
    assert code == "print(1)"


def test_factor_multi_process_implement_handles_token_pruning_branches(monkeypatch):
    monkeypatch.setattr(es_module.LLM_SETTINGS, "chat_token_limit", 100)
    _patch_factor_prompts(monkeypatch)
    _patch_fake_api(
        monkeypatch,
        token_values=[1000, 1000, 50],
        completion_values=['{"code":"print(2)"}'],
    )

    scen = _Scenario()
    strategy = FactorMultiProcessEvolvingStrategy(scen=scen, settings=_settings())
    strategy.extract_expr = lambda code: "TS_MEAN($close, 2)"
    target_task = _task("factor-b")
    success_knowledge = [_FakeKnowledge(), _FakeKnowledge()]
    queried = _queried(
        target_task,
        former=[_FakeKnowledge()],
        success=success_knowledge,
        errors=[("err", (_FakeKnowledge("expr='bad'"), _FakeKnowledge("expr='good'")))],
    )
    code = strategy.implement_one_task(target_task, queried)
    assert code == "print(2)"


def test_factor_multi_process_assign_code_list_to_evo(monkeypatch):
    monkeypatch.setattr(es_module, "FactorFBWorkspace", _FakeWorkspace)
    scen = _Scenario()
    strategy = FactorMultiProcessEvolvingStrategy(scen=scen, settings=_settings())
    t1, t2 = _task("x1"), _task("x2")
    evo = SimpleNamespace(sub_tasks=[t1, t2], sub_workspace_list=[None, None])

    out = strategy.assign_code_list_to_evo([None, "print('ok')"], evo)
    assert out.sub_workspace_list[0] is None
    assert out.sub_workspace_list[1].code_dict["factor.py"] == "print('ok')"


def test_factor_parsing_extract_expr_and_template_first_round():
    strategy = FactorParsingStrategy(scen=_Scenario(), settings=_settings())
    assert strategy.extract_expr("x=1\nexpr = \"TS_MEAN($close, 5)\"") == "TS_MEAN($close, 5)"
    assert strategy.extract_expr("x=1") == ""

    task = _task("parser-first")
    queried = _queried(task, former=[], success=[], errors=[])
    code = strategy.implement_one_task(task, queried)
    assert "def calculate_factor" in code
    assert "parser-first" in code


def test_factor_parsing_with_history_json_retry_and_none_path(monkeypatch):
    monkeypatch.setattr(es_module, "FactorFBWorkspace", _FakeWorkspace)
    monkeypatch.setattr(es_module.LLM_SETTINGS, "chat_token_limit", 100)
    monkeypatch.setattr(es_module.FACTOR_COSTEER_SETTINGS, "v2_error_summary", False)
    _patch_fake_api(
        monkeypatch,
        token_values=[50, 50],
        completion_values=["not-json", '{"expr":"TS_SUM($close, 3)"}'],
    )

    strategy = FactorParsingStrategy(scen=_Scenario(), settings=_settings())
    task = _task("parser-retry")
    queried = _queried(task, former=[_FakeKnowledge("expr = 'BROKEN'")], success=[_FakeKnowledge()], errors=[])
    code = strategy.implement_one_task(task, queried)
    assert "TS_SUM($close, 3)" in code

    _patch_fake_api(monkeypatch, token_values=[50], completion_values=["bad-json"], fallback_completion="bad-json")
    none_code = strategy.implement_one_task(task, queried)
    assert none_code is None

    evo = SimpleNamespace(sub_tasks=[task], sub_workspace_list=[None])
    out = strategy.assign_code_list_to_evo(["print(3)"], evo)
    assert out.sub_workspace_list[0].code_dict["factor.py"] == "print(3)"


def test_factor_running_strategy_implement_assign_and_evolve(monkeypatch):
    monkeypatch.setattr(es_module, "FactorFBWorkspace", _FakeWorkspace)
    monkeypatch.setattr(es_module.RD_AGENT_SETTINGS, "multi_proc_n", 1)

    def _fake_mp_wrapper(tasks, n):  # noqa: ANN001, ANN201
        assert n == 1
        return ["print('r1')", "print('r2')"]

    monkeypatch.setattr(es_module, "multiprocessing_wrapper", _fake_mp_wrapper)

    strategy = FactorRunningStrategy(scen=_Scenario(), settings=_settings())
    t1, t2 = _task("run-1"), _task("run-2")
    evo = SimpleNamespace(sub_tasks=[t1, t2], sub_workspace_list=[None, None], corresponding_selection=None)

    rendered = strategy.implement_one_task(t1, queried_knowledge=None)
    assert "run-1" in rendered

    out = strategy.evolve(evo=evo, queried_knowledge=None)
    assert out.corresponding_selection == [0, 1]
    assert out.sub_workspace_list[0].code_dict["factor.py"] == "print('r1')"
    assert out.sub_workspace_list[1].code_dict["factor.py"] == "print('r2')"
