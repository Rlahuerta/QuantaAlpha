from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

import quantaalpha.coder.costeer as costeer_module
from quantaalpha.coder.costeer import CoSTEER
from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.coder.costeer.knowledge_management import CoSTEERKnowledgeBaseV1, CoSTEERKnowledgeBaseV2
from quantaalpha.core.evaluation import Evaluator
from quantaalpha.core.evolving_framework import EvolvingStrategy
from quantaalpha.core.experiment import Experiment, Task
from quantaalpha.core.scenario import Scenario


class _Scenario(Scenario):
    @property
    def background(self) -> str:
        return "bg"

    @property
    def interface(self) -> str:
        return "interface"

    @property
    def output_format(self) -> str:
        return "output"

    @property
    def simulator(self) -> str:
        return "sim"

    @property
    def rich_style_description(self) -> str:
        return "rich"

    def get_scenario_all_desc(self, task=None, filtered_tag=None, simple_background=None) -> str:  # noqa: ANN001, ANN201
        return "all-desc"


class _Evaluator(Evaluator):
    def evaluate(self, target_task, implementation=None, gt_implementation=None, **kwargs):  # noqa: ANN001, ANN201
        return [SimpleNamespace(final_decision=True)]


class _EvolvingStrategy(EvolvingStrategy):
    def evolve(self, *evo, evolving_trace=None, queried_knowledge=None, **kwargs):  # noqa: ANN002, ANN201
        return kwargs.get("evo", evo[0] if evo else None)


class _Task(Task):
    def get_task_information(self) -> str:
        return self.name


def _build_costeer(evolving_version: int, *, knowledge_base_path: Path | None = None, new_knowledge_base_path: Path | None = None):
    scen = _Scenario()
    settings = CoSTEERSettings()
    settings.max_loop = 1
    settings.knowledge_base_path = str(knowledge_base_path) if knowledge_base_path else None
    settings.new_knowledge_base_path = str(new_knowledge_base_path) if new_knowledge_base_path else None
    return CoSTEER(settings, _Evaluator(scen), _EvolvingStrategy(scen), evolving_version, scen)


def test_costeer_load_or_init_knowledge_base_paths(tmp_path):
    c1 = _build_costeer(1)
    c2 = _build_costeer(2)
    assert isinstance(c1.knowledge_base, CoSTEERKnowledgeBaseV1)
    assert isinstance(c2.knowledge_base, CoSTEERKnowledgeBaseV2)

    incompatible_path = tmp_path / "kb_incompatible.pkl"
    with Path.open(incompatible_path, "wb") as fh:
        pickle.dump(CoSTEERKnowledgeBaseV2(), fh)

    with pytest.raises(ValueError, match="not compatible"):
        c1.load_or_init_knowledge_base(incompatible_path, component_init_list=[])


def test_costeer_develop_uses_evo_agent_and_saves_knowledge(monkeypatch, tmp_path):
    out_kb = tmp_path / "new_kb.pkl"
    c1 = _build_costeer(1, new_knowledge_base_path=out_kb)

    class _FakeEvoAgent:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        def multistep_evolve(self, experiment, evaluator, filter_final_evo=False):  # noqa: ANN001, ANN201
            experiment.sub_workspace_list = ["workspace-from-evo"]
            return experiment

    monkeypatch.setattr(costeer_module, "FilterFailedRAGEvoAgent", _FakeEvoAgent)

    exp = Experiment(sub_tasks=[_Task("sub-task-1")])
    out = c1.develop(exp)

    assert out is exp
    assert out.sub_workspace_list == ["workspace-from-evo"]
    assert out_kb.exists()
