from __future__ import annotations

from types import SimpleNamespace

import pytest

import quantaalpha.coder.costeer.knowledge_management as km_module
from quantaalpha.coder.costeer.config import CoSTEERSettings
from quantaalpha.coder.costeer.knowledge_management import (
    CoSTEERKnowledge,
    CoSTEERKnowledgeBaseV1,
    CoSTEERKnowledgeBaseV2,
    CoSTEERQueriedKnowledgeV2,
    CoSTEERRAGStrategyV1,
    CoSTEERRAGStrategyV2,
)
from quantaalpha.coder.knowledge.graph import UndirectedNode


class _Task:
    def __init__(self, info: str):
        self.info = info
        self.name = info

    def get_task_information(self) -> str:
        return self.info


class _Workspace:
    def __init__(self, code: str):
        self.code = code

    def copy(self):
        return _Workspace(self.code)

    def __str__(self):
        return f"WS({self.code})"


def _fb(
    *,
    final_decision: bool = False,
    value_generated_flag: bool = False,
    execution_feedback: str = "",
    value_feedback: str = "",
    final_decision_based_on_gt: bool = False,
):
    return SimpleNamespace(
        final_decision=final_decision,
        value_generated_flag=value_generated_flag,
        execution_feedback=execution_feedback,
        value_feedback=value_feedback,
        final_decision_based_on_gt=final_decision_based_on_gt,
    )


class _GraphStub:
    def __init__(self):
        self.nodes: list[UndirectedNode] = []

    def size(self):
        return len(self.nodes)

    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.append(node)

    def add_nodes(self, node, neighbors):
        self.add_node(node)
        for n in neighbors:
            self.add_node(n)

    def get_node_by_content(self, content):
        for n in self.nodes:
            if n.content == content:
                return n
        return None

    def get_all_nodes_by_label(self, label):
        return [n for n in self.nodes if n.label == label]

    def get_all_nodes_by_label_list(self, labels):
        return [n for n in self.nodes if n.label in labels]

    def query_by_content(self, **kwargs):
        return [self.get_node_by_content(kwargs["content"])] if self.get_node_by_content(kwargs["content"]) else []

    def query_by_node(self, node, **kwargs):
        constraint_labels = kwargs.get("constraint_labels") or []
        if node.label == "component" and constraint_labels == ["task_description"]:
            return [n for n in self.nodes if n.label == "task_description"]
        if node.label == "error" and constraint_labels == ["task_trace"]:
            return [n for n in self.nodes if n.label == "task_trace"]
        if node.label in ("task_trace", "task_description") and "task_success_implement" in constraint_labels:
            return [n for n in self.nodes if n.label == "task_success_implement"]
        return []

    def get_nodes_intersection(self, node_list, **kwargs):  # noqa: ARG002
        return [n for n in self.nodes if n.label == "task_description"]


def _mk_knowledge(task: _Task, code: str, feedback) -> CoSTEERKnowledge:
    return CoSTEERKnowledge(target_task=task, implementation=_Workspace(code), feedback=feedback)


def test_costeer_knowledge_and_v1_not_implemented_paths():
    task = _Task("task-v1")
    k = _mk_knowledge(task, "print(1)", _fb(final_decision=True))
    assert "implementation code" in k.get_implementation_and_feedback_str()
    assert "implementation feedback" in k.get_implementation_and_feedback_str()

    rag_v1 = CoSTEERRAGStrategyV1(CoSTEERKnowledgeBaseV1(), CoSTEERSettings())
    with pytest.raises(NotImplementedError):
        rag_v1.generate_knowledge([])
    with pytest.raises(NotImplementedError):
        rag_v1.query(SimpleNamespace(sub_tasks=[task]), [])


def test_rag_v2_analyze_component_and_analyze_error_paths(monkeypatch):
    kb = CoSTEERKnowledgeBaseV2()
    kb.graph = _GraphStub()
    comp_1 = UndirectedNode(content="comp-a", label="component")
    comp_2 = UndirectedNode(content="comp-b", label="component")
    kb.graph.add_node(comp_1)
    kb.graph.add_node(comp_2)
    rag = CoSTEERRAGStrategyV2(kb, CoSTEERSettings())

    monkeypatch.setattr(
        km_module.APIBackend,
        "build_messages_and_create_chat_completion",
        lambda self, **kwargs: '{"component_no_list":[1,2]}',
    )
    selected = rag.analyze_component("some task")
    assert selected == [comp_1, comp_2]

    monkeypatch.setattr(
        km_module.APIBackend,
        "build_messages_and_create_chat_completion",
        lambda self, **kwargs: "not-json",
    )
    assert rag.analyze_component("bad llm output") == []

    execution_feedback = (
        'File "factor.py", line 2, in run\n'
        "    bad_line\n"
        "TypeError: boom"
    )
    analyzed_exec = rag.analyze_error(execution_feedback, feedback_type="execution")
    assert analyzed_exec and "ErrorType: TypeError" in analyzed_exec[0]

    analyzed_undefined = rag.analyze_error("n/a", feedback_type="execution")
    assert analyzed_undefined == ["Undefined Error"]

    value_feedback = "The source dataframe and the ground truth dataframe have different rows count."
    analyzed_value = rag.analyze_error(value_feedback, feedback_type="value")
    assert analyzed_value == [value_feedback]

    assert rag.analyze_error("x", feedback_type="unknown") == ["Undefined Error"]

    existing_error = UndirectedNode(content="Undefined Error", label="error")
    kb.graph.add_node(existing_error)
    analyzed_existing = rag.analyze_error("anything", feedback_type="execution")
    assert existing_error in analyzed_existing


def test_rag_v2_former_trace_component_and_error_queries(monkeypatch):
    kb = CoSTEERKnowledgeBaseV2()
    kb.graph = _GraphStub()
    settings = CoSTEERSettings()
    settings.fail_task_trial_limit = 3
    settings.v2_query_former_trace_limit = 5
    settings.v2_query_component_limit = 2
    settings.v2_query_error_limit = 2
    settings.v2_knowledge_sampler = 1.0
    settings.v2_add_fail_attempt_to_latest_successful_execution = True
    rag = CoSTEERRAGStrategyV2(kb, settings)

    t_keep = _Task("task-keep")
    t_fail = _Task("task-fail")
    t_component = _Task("task-component")
    t_error = _Task("task-error")
    t_done = _Task("task-done")

    k_keep_1 = _mk_knowledge(t_keep, "code-1", _fb(value_generated_flag=True, execution_feedback="err-a"))
    k_keep_2 = _mk_knowledge(t_keep, "code-2", _fb(value_generated_flag=False, execution_feedback="err-b"))
    kb.working_trace_knowledge[t_keep.get_task_information()] = [k_keep_1, k_keep_2]

    kb.working_trace_knowledge[t_fail.get_task_information()] = [
        _mk_knowledge(t_fail, f"code-{i}", _fb(value_generated_flag=True, execution_feedback=f"e{i}"))
        for i in range(3)
    ]

    queried = CoSTEERQueriedKnowledgeV2()
    queried = rag.former_trace_query(
        SimpleNamespace(sub_tasks=[t_keep, t_fail]),
        queried,
        v2_query_former_trace_limit=5,
        v2_add_fail_attempt_to_latest_successful_execution=True,
    )
    keep_traces, keep_latest_attempt = queried.task_to_former_failed_traces[t_keep.get_task_information()]
    assert keep_traces == [k_keep_1]
    assert keep_latest_attempt is k_keep_2
    assert t_fail.get_task_information() in queried.failed_task_info_set

    comp_a = UndirectedNode(content="component-a", label="component")
    comp_b = UndirectedNode(content="component-b", label="component")
    task_desc = UndirectedNode(content="desc-a", label="task_description")
    success_node = UndirectedNode(content="success", label="task_success_implement")
    kb.graph.add_node(comp_a)
    kb.graph.add_node(comp_b)
    kb.graph.add_node(task_desc)
    kb.graph.add_node(success_node)
    kb.task_to_component_nodes[t_component.get_task_information()] = [comp_a, comp_b]

    knowledge_gt = _mk_knowledge(
        _Task("gt"),
        "gt-code",
        _fb(final_decision=True, final_decision_based_on_gt=True),
    )
    knowledge_non_gt = _mk_knowledge(
        _Task("non-gt"),
        "non-gt-code",
        _fb(final_decision=True, final_decision_based_on_gt=False),
    )
    kb.node_to_implementation_knowledge_dict[success_node.id] = knowledge_gt
    kb.success_task_to_knowledge_dict = {"gt-task": knowledge_gt, "non-gt-task": knowledge_non_gt}

    monkeypatch.setattr(
        km_module,
        "calculate_embedding_distance_between_str_list",
        lambda query_list, target_list: [[0.9 for _ in target_list]],
    )
    queried.task_to_similar_task_successful_knowledge = {}
    queried.failed_task_info_set = {t_done.get_task_information()}
    queried = rag.component_query(
        SimpleNamespace(sub_tasks=[t_component, t_done]),
        queried,
        v2_query_component_limit=2,
        knowledge_sampler=1.0,
    )
    assert queried.task_to_similar_task_successful_knowledge[t_done.get_task_information()] == []
    assert queried.task_to_similar_task_successful_knowledge[t_component.get_task_information()]

    error_node = UndirectedNode(content="ErrorType: TypeError\nError line: bad_line", label="error")
    trace_node = UndirectedNode(content="trace", label="task_trace")
    success_node_2 = UndirectedNode(content="success2", label="task_success_implement")
    kb.graph.add_node(error_node)
    kb.graph.add_node(trace_node)
    kb.graph.add_node(success_node_2)

    trace_knowledge = _mk_knowledge(t_error, "trace-code", _fb(final_decision=False))
    success_knowledge = _mk_knowledge(t_error, "success-code", _fb(final_decision=True))
    kb.node_to_implementation_knowledge_dict[trace_node.id] = trace_knowledge
    kb.node_to_implementation_knowledge_dict[success_node_2.id] = success_knowledge
    kb.working_trace_knowledge[t_error.get_task_information()] = [trace_knowledge]
    kb.working_trace_error_analysis[t_error.get_task_information()] = [[error_node]]

    queried.task_to_former_failed_traces[t_error.get_task_information()] = ([trace_knowledge], None)
    queried.failed_task_info_set = {t_done.get_task_information()}
    queried = rag.error_query(
        SimpleNamespace(sub_tasks=[t_error, t_done]),
        queried,
        v2_query_error_limit=2,
        knowledge_sampler=1.0,
    )
    pairs = queried.task_to_similar_error_successful_knowledge[t_error.get_task_information()]
    assert pairs and "ErrorType: TypeError" in pairs[0][0]
    assert queried.task_to_similar_error_successful_knowledge[t_done.get_task_information()] == []


def test_knowledge_base_v2_update_success_and_graph_wrapper_paths():
    kb = CoSTEERKnowledgeBaseV2()
    kb.graph = _GraphStub()
    task = _Task("task-update")
    info = task.get_task_information()
    comp = UndirectedNode(content="comp-x", label="component")
    kb.task_to_component_nodes[info] = [comp]

    k1 = _mk_knowledge(task, "trace-code", _fb(final_decision=False, execution_feedback="trace-fb"))
    k2 = _mk_knowledge(task, "success-code", _fb(final_decision=True, execution_feedback="success-fb"))
    kb.working_trace_knowledge[info] = [k1, k2]
    kb.working_trace_error_analysis[info] = [["Undefined Error"]]

    kb.update_success_task(info)
    assert any(node.label == "task_trace" for node in kb.graph.nodes)
    assert any(node.label == "task_success_implement" for node in kb.graph.nodes)
    assert kb.node_to_implementation_knowledge_dict

    by_content = kb.graph_get_node_by_content("Undefined Error")
    queried_content = kb.graph_query_by_content(content="Undefined Error")
    assert by_content is not None
    assert queried_content and queried_content[0] is by_content

    trace_node = next(node for node in kb.graph.nodes if node.label == "task_trace")
    queried_node = kb.graph_query_by_node(node=trace_node, step=1, constraint_labels=["task_success_implement"], block=True)
    assert queried_node

    node_a = UndirectedNode(content="a", label="component")
    node_b = UndirectedNode(content="b", label="component")
    kb.graph.add_node(node_a)
    kb.graph.add_node(node_b)
    inter = kb.graph_query_by_intersection([node_a, node_b], steps=1, constraint_labels=["task_description"])
    inter_with_origin = kb.graph_query_by_intersection(
        [node_a, node_b],
        steps=1,
        constraint_labels=["task_description"],
        output_intersection_origin=True,
    )
    assert inter
    assert inter_with_origin and isinstance(inter_with_origin[0], list)

    with pytest.raises(AssertionError):
        kb.graph_query_by_intersection([node_a], steps=1, constraint_labels=["task_description"])


def test_rag_v2_generate_knowledge_and_query_dispatch_paths(monkeypatch):
    kb = CoSTEERKnowledgeBaseV2()
    kb.graph = _GraphStub()
    settings = CoSTEERSettings()
    settings.v2_query_former_trace_limit = 2
    settings.v2_add_fail_attempt_to_latest_successful_execution = True
    settings.v2_query_component_limit = 2
    settings.v2_query_error_limit = 2
    settings.v2_knowledge_sampler = 1.0
    rag = CoSTEERRAGStrategyV2(kb, settings)

    t_success = _Task("task-success")
    t_fail = _Task("task-fail")
    kb.task_to_component_nodes[t_success.get_task_information()] = []
    kb.task_to_component_nodes[t_fail.get_task_information()] = []
    monkeypatch.setattr(rag, "analyze_error", lambda single_feedback, feedback_type="execution": ["Undefined Error"])  # noqa: ARG005

    evo_step = SimpleNamespace(
        evolvable_subjects=SimpleNamespace(
            sub_tasks=[t_success, t_fail],
            sub_workspace_list=[_Workspace("ok"), _Workspace("bad")],
        ),
        feedback=[
            _fb(final_decision=True, value_generated_flag=True, execution_feedback="ok"),
            _fb(final_decision=False, value_generated_flag=False, execution_feedback="boom"),
        ],
    )
    assert rag.generate_knowledge([evo_step]) is None
    assert rag.generate_knowledge([evo_step]) is None
    assert t_success.get_task_information() in kb.success_task_to_knowledge_dict
    assert t_fail.get_task_information() in kb.working_trace_error_analysis

    called = {"former": 0, "component": 0, "error": 0}

    def _former(evo, queried_knowledge_v2, v2_query_former_trace_limit=5, v2_add_fail_attempt_to_latest_successful_execution=False):  # noqa: ANN001, ANN201, ARG001
        called["former"] += 1
        return queried_knowledge_v2

    def _component(evo, queried_knowledge_v2, v2_query_component_limit=5, knowledge_sampler=1.0):  # noqa: ANN001, ANN201, ARG001
        called["component"] += 1
        return queried_knowledge_v2

    def _error(evo, queried_knowledge_v2, v2_query_error_limit=5, knowledge_sampler=1.0):  # noqa: ANN001, ANN201, ARG001
        called["error"] += 1
        return queried_knowledge_v2

    monkeypatch.setattr(rag, "former_trace_query", _former)
    monkeypatch.setattr(rag, "component_query", _component)
    monkeypatch.setattr(rag, "error_query", _error)
    queried = rag.query(SimpleNamespace(sub_tasks=[t_success]), evolving_trace=[])

    assert called == {"former": 1, "component": 1, "error": 1}
    assert queried.success_task_to_knowledge_dict == kb.success_task_to_knowledge_dict


def test_knowledge_base_v2_init_components_and_query_noop_path(monkeypatch):
    # Mock create_embedding to avoid requiring a real EMBEDDING_MODEL / API.
    # Use a deterministic non-zero vector keyed by content so cosine similarity
    # between identical strings is 1.0 (above the 0.999 threshold).
    import hashlib

    def _fake_embedding(self):
        h = hashlib.md5(self.content.encode()).digest()
        self.embedding = [float(b) / 255.0 + 0.01 for b in h[:8]]

    monkeypatch.setattr(
        "quantaalpha.coder.knowledge.vector_base.KnowledgeMetaData.create_embedding",
        _fake_embedding,
    )
    kb = CoSTEERKnowledgeBaseV2(init_component_list=["component-init-a"])
    node = kb.graph_get_node_by_content("component-init-a")
    assert node is not None
    with pytest.raises(AttributeError):
        kb.get_all_nodes_by_label("component")
    assert kb.query() is None
