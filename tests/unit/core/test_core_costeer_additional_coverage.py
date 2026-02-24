from __future__ import annotations

from types import SimpleNamespace

import dill as pickle
import pytest

import quantaalpha.coder.costeer.evaluators as costeer_eval_module
import quantaalpha.coder.costeer.evolvable_subjects as evolvable_module
import quantaalpha.coder.costeer.scheduler as scheduler_module
import quantaalpha.coder.costeer.evolving_strategy as mp_strategy_module
import quantaalpha.core.evolving_agent as core_agent_module
import quantaalpha.core.knowledge_base as kb_module
from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.coder.costeer.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiEvaluator,
    CoSTEERSingleFeedback,
)
from quantaalpha.coder.costeer.evolving_agent import FilterFailedRAGEvoAgent
from quantaalpha.coder.costeer.evolving_strategy import MultiProcessEvolvingStrategy
from quantaalpha.coder.costeer.evolvable_subjects import EvolvingItem
from quantaalpha.coder.costeer.scheduler import random_select
from quantaalpha.coder.costeer.task import CoSTEERTask
from quantaalpha.core.developer import Developer
from quantaalpha.core.evaluation import Evaluator, Feedback
from quantaalpha.core.evolving_agent import RAGEvoAgent
from quantaalpha.core.evolving_framework import EvolvingStrategy
from quantaalpha.core.experiment import Experiment, FBWorkspace, Task
from quantaalpha.core.knowledge_base import KnowledgeBase
from quantaalpha.core.scenario import Scenario


class _Scenario(Scenario):
    @property
    def background(self) -> str:
        return "bg"

    @property
    def interface(self) -> str:
        return "iface"

    @property
    def output_format(self) -> str:
        return "json"

    @property
    def simulator(self) -> str:
        return "sim"

    @property
    def rich_style_description(self) -> str:
        return "rich"

    def get_scenario_all_desc(
        self,
        task: Task | None = None,
        filtered_tag: str | None = None,
        simple_background: bool | None = None,
    ) -> str:
        return "desc"


class _Task(Task):
    def get_task_information(self) -> str:
        return self.name


class _DeveloperBase(Developer):
    def develop(self, exp):  # type: ignore[override]
        return super().develop(exp)


class _EvaluatorBase(Evaluator):
    def evaluate(self, target_task, implementation, gt_implementation, **kwargs):  # type: ignore[override]
        return super().evaluate(target_task, implementation, gt_implementation, **kwargs)


class _PassStrategy(EvolvingStrategy):
    def evolve(self, *evo, evolving_trace=None, queried_knowledge=None, **kwargs):  # type: ignore[override]
        if evo:
            return evo[0]
        return kwargs["evo"]


class _SingleEvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task,
        implementation,
        gt_implementation,
        queried_knowledge=None,
        **kwargs,
    ):  # type: ignore[override]
        return CoSTEERSingleFeedback(
            final_decision=(target_task.name == "ok"),
            final_feedback="done",
        )


class _ConcreteCoSTEERTask(CoSTEERTask):
    def get_task_information(self) -> str:
        return self.name


class _DummyMPStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(self, target_task, queried_knowledge=None):  # type: ignore[override]
        ws = FBWorkspace(target_task=target_task)
        ws.code_dict["impl.py"] = "pass"
        return ws

    def assign_code_list_to_evo(self, code_list, evo):  # type: ignore[override]
        for idx, ws in enumerate(code_list):
            if ws is not None:
                evo.sub_workspace_list[idx] = ws
        return evo


def test_knowledge_base_load_dump_and_warning(tmp_path, monkeypatch):
    kb_path = tmp_path / "kb" / "state.pkl"
    kb = KnowledgeBase(kb_path)
    kb.alpha = 123
    kb.dump()
    assert kb_path.exists()

    loaded = KnowledgeBase(kb_path)
    assert loaded.alpha == 123
    assert loaded.path == kb_path

    with kb_path.open("wb") as f:
        pickle.dump({"beta": 7, "path": "ignore-me"}, f)
    loaded_dict = KnowledgeBase(kb_path)
    assert loaded_dict.beta == 7
    assert loaded_dict.path == kb_path

    class _Obj:
        def __init__(self):
            self.gamma = "ok"
            self.path = "not-used"

    with kb_path.open("wb") as f:
        pickle.dump(_Obj(), f)
    loaded_obj = KnowledgeBase(kb_path)
    assert loaded_obj.gamma == "ok"
    assert loaded_obj.path == kb_path

    warning_messages: list[str] = []
    monkeypatch.setattr(kb_module.logger, "warning", lambda msg: warning_messages.append(msg))
    KnowledgeBase().dump()
    assert warning_messages
    assert "path is not set" in warning_messages[0]


def test_abstract_developer_and_evaluator_raise_not_implemented():
    scen = _Scenario()
    dev = _DeveloperBase(scen)
    with pytest.raises(NotImplementedError, match="generate method is not implemented"):
        dev.develop(object())

    eva = _EvaluatorBase(scen)
    with pytest.raises(NotImplementedError):
        eva.evaluate(
            _Task("task-a"),
            FBWorkspace(target_task=_Task("task-a")),
            FBWorkspace(target_task=_Task("task-a")),
        )
    assert isinstance(Feedback(), Feedback)


def test_scheduler_task_and_evolving_item_branches(monkeypatch):
    task = _ConcreteCoSTEERTask(name="task-1", base_code="print('x')")
    assert task.base_code == "print('x')"

    selected_logs: list[str] = []
    monkeypatch.setattr(scheduler_module.random, "sample", lambda seq, n: [seq[-1]])
    monkeypatch.setattr(scheduler_module.logger, "info", lambda msg: selected_logs.append(msg))
    selected = random_select([0, 1, 2], evo=None, selected_num=1, queried_knowledge=None, scen=_Scenario())
    assert selected == [2]
    assert selected_logs

    warn_logs: list[str] = []
    monkeypatch.setattr(evolvable_module.logger, "warning", lambda msg: warn_logs.append(msg))
    mismatch_item = EvolvingItem(
        sub_tasks=[_Task("a"), _Task("b")],
        sub_gt_implementations=[FBWorkspace(target_task=_Task("gt-only"))],
    )
    assert mismatch_item.sub_gt_implementations is None
    assert warn_logs

    exp = Experiment(sub_tasks=[_Task("x"), _Task("y")])
    exp.based_experiments = ["base"]
    exp.experiment_workspace = "workspace"
    item = EvolvingItem.from_experiment(exp)
    assert item.based_experiments == ["base"]
    assert item.experiment_workspace == "workspace"


def test_costeer_multi_evaluator_and_filter_failed_agent(monkeypatch):
    scen = _Scenario()
    single = _SingleEvaluator(scen)
    multi = CoSTEERMultiEvaluator(single_evaluator=single, scen=scen)
    monkeypatch.setattr(costeer_eval_module.RD_AGENT_SETTINGS, "multi_proc_n", 1)

    info_logs: list[str] = []
    monkeypatch.setattr(costeer_eval_module.logger, "info", lambda msg: info_logs.append(msg))
    evo = EvolvingItem(
        sub_tasks=[_Task("ok"), _Task("bad")],
        sub_gt_implementations=[None, None],
    )
    evo.sub_workspace_list = [SimpleNamespace(clear=lambda: None), SimpleNamespace(clear=lambda: None)]
    feedback = multi.evaluate(evo, queried_knowledge=SimpleNamespace())
    assert isinstance(feedback, list)
    assert feedback[0].final_decision is True
    assert feedback[1].final_decision is False
    assert evo.sub_tasks[0].factor_implementation is True
    assert not hasattr(evo.sub_tasks[1], "factor_implementation")
    assert info_logs and "Final decisions" in info_logs[0]

    text = str(CoSTEERSingleFeedback(final_decision=False))
    assert "No execution feedback" in text
    assert "FAIL" in text

    class _Clearable:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    ws_fail = _Clearable()
    ws_pass = _Clearable()
    evo_for_filter = EvolvingItem(sub_tasks=[_Task("f"), _Task("p")])
    evo_for_filter.sub_workspace_list = [ws_fail, ws_pass]
    filter_agent = FilterFailedRAGEvoAgent(
        max_loop=1,
        evolving_strategy=_PassStrategy(scen),
        rag=None,
    )
    filtered = filter_agent.filter_evolvable_subjects_by_feedback(
        evo_for_filter,
        [CoSTEERSingleFeedback(final_decision=False), CoSTEERSingleFeedback(final_decision=True)],
    )
    assert filtered is evo_for_filter
    assert ws_fail.cleared is True
    assert ws_pass.cleared is False


def test_multiprocess_strategy_evolve_and_rag_agent_branches(monkeypatch):
    scen = _Scenario()
    settings = CoSTEERSettings(select_threshold=1)
    strategy = _DummyMPStrategy(scen=scen, settings=settings)
    strategy.settings.select_threshold = 1

    selected_calls: list[tuple[list[int], int]] = []

    def _select(indices, evo_obj, selected_num, queried, scen_obj):
        selected_calls.append((indices.copy(), selected_num))
        return [indices[0]]

    monkeypatch.setattr(strategy, "select_one_round_tasks", _select)
    monkeypatch.setattr(
        mp_strategy_module,
        "multiprocessing_wrapper",
        lambda func_calls, n: [f(*args) for f, args in func_calls],
    )
    monkeypatch.setattr(mp_strategy_module.RD_AGENT_SETTINGS, "multi_proc_n", 1)

    tasks = [_Task("cached"), _Task("failed"), _Task("todo-1"), _Task("todo-2")]
    evo = EvolvingItem(sub_tasks=tasks)
    queried_knowledge = SimpleNamespace(
        success_task_to_knowledge_dict={
            tasks[0].get_task_information(): SimpleNamespace(implementation="cached-impl")
        },
        failed_task_info_set={tasks[1].get_task_information()},
    )
    evolved = strategy.evolve(evo=evo, queried_knowledge=queried_knowledge)
    assert selected_calls == [([2, 3], 1)]
    assert evolved.sub_workspace_list[0] == "cached-impl"
    assert evolved.sub_workspace_list[2] is not None
    assert evolved.sub_workspace_list[3] is None
    assert evolved.corresponding_selection == [2]

    class _RAG:
        def __init__(self):
            self.generated = 0
            self.queried = 0

        def generate_knowledge(self, evolving_trace):
            self.generated += 1

        def query(self, evo, evolving_trace):
            self.queried += 1
            return {"trace_len": len(evolving_trace)}

    class _Eval:
        def __init__(self):
            self.calls = 0

        def evaluate(self, evo, queried_knowledge=None):
            self.calls += 1
            return CoSTEERSingleFeedback(final_decision=True)

    monkeypatch.setattr(core_agent_module, "tqdm", lambda it, *_args, **_kwargs: it)
    monkeypatch.setattr(core_agent_module.logger, "log_object", lambda *args, **kwargs: None)
    monkeypatch.setattr(core_agent_module.logger, "info", lambda *args, **kwargs: None)

    pass_strategy = _PassStrategy(scen)
    evo_subject = EvolvingItem(sub_tasks=[_Task("t1")])
    evo_subject.sub_workspace_list = ["workspace-1"]
    rag = _RAG()
    eva = _Eval()

    rag_agent = RAGEvoAgent(
        max_loop=2,
        evolving_strategy=pass_strategy,
        rag=rag,
        with_knowledge=True,
        with_feedback=True,
        knowledge_self_gen=True,
    )
    result = rag_agent.multistep_evolve(evo_subject, eva, filter_final_evo=False)
    assert result is evo_subject
    assert len(rag_agent.evolving_trace) == 2
    assert rag.generated == 2
    assert rag.queried == 2
    assert eva.calls == 2

    rag_agent_filter = RAGEvoAgent(
        max_loop=1,
        evolving_strategy=pass_strategy,
        rag=None,
        with_knowledge=False,
        with_feedback=True,
        knowledge_self_gen=False,
    )
    filtered_result = rag_agent_filter.multistep_evolve(evo_subject, Feedback(), filter_final_evo=True)
    assert filtered_result is None
    assert rag_agent_filter.filter_evolvable_subjects_by_feedback(evo_subject, None) is None
