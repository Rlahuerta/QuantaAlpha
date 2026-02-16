from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import quantaalpha.coder.knowledge.graph as graph_module
import quantaalpha.coder.knowledge.vector_base as vector_module
import quantaalpha.llm.config as llm_config_module
from quantaalpha.coder.knowledge.graph import (
    Graph,
    UndirectedGraph,
    UndirectedNode,
    assign_isometric_coordinate_to_node,
    assign_random_coordinate_to_node,
    curly_node_coordinate,
    graph_to_edges,
)
from quantaalpha.coder.knowledge.vector_base import (
    Document,
    KnowledgeMetaData,
    PDVectorBase,
    VectorBase,
    contents_to_documents,
)


class _FakeAPIBackend:
    def create_embedding(self, input_content):
        if isinstance(input_content, list):
            return [[float(len(v)), 1.0] for v in input_content]
        return [float(len(input_content)), 1.0]


def test_knowledge_metadata_embedding_trunk_and_document_conversion(monkeypatch):
    monkeypatch.setattr(vector_module, "APIBackend", lambda: _FakeAPIBackend())

    doc = KnowledgeMetaData(content="abcdef", label="L1")
    doc.split_into_trunk(size=2)
    assert doc.trunks == ["ab", "cd", "ef"]
    assert len(doc.trunks_embedding) == 3

    doc.create_embedding()
    assert doc.embedding == [6.0, 1.0]
    original_embedding = doc.embedding
    doc.create_embedding()
    assert doc.embedding == original_embedding

    from_dict_doc = KnowledgeMetaData().from_dict({"id": "id1", "label": "LBL", "content": "X"})
    assert from_dict_doc.id == "id1"
    assert from_dict_doc.label == "LBL"
    assert "Document(id=id1" in repr(from_dict_doc)

    monkeypatch.setattr(llm_config_module.LLM_SETTINGS, "embedding_max_str_num", 2)
    docs = contents_to_documents(["a", "bb", "ccc"], label="X")
    assert [d.label for d in docs] == ["X", "X", "X"]
    assert [d.embedding[0] for d in docs] == [1.0, 2.0, 3.0]


def test_pdvectorbase_and_base_vectorbase_paths(monkeypatch):
    monkeypatch.setattr(vector_module, "APIBackend", lambda: _FakeAPIBackend())

    base = VectorBase()
    assert base.add(Document(content="x")) is None
    assert base.search("x") is None

    pd_base = PDVectorBase()
    assert pd_base.shape() == (0, 4)
    empty_docs, empty_scores = pd_base.search("x")
    assert empty_docs == []
    assert empty_scores == []

    doc = Document(content="hello", label="greet")
    doc.trunks = ["he", "llo"]
    doc.trunks_embedding = [[2.0, 1.0], [3.0, 1.0]]
    pd_base.add(doc)
    pd_base.add([Document(content="bye", label="bye", embedding=[3.0, 1.0])])

    docs, scores = pd_base.search("hello", topk_k=3, similarity_threshold=-1.0)
    assert len(docs) >= 1
    assert len(scores) >= 1
    assert all(isinstance(item, Document) for item in docs)


def test_graph_base_and_batch_embedding_paths(monkeypatch):
    monkeypatch.setattr(graph_module, "APIBackend", lambda: _FakeAPIBackend())
    monkeypatch.setattr(vector_module, "APIBackend", lambda: _FakeAPIBackend())
    monkeypatch.setattr(llm_config_module.LLM_SETTINGS, "embedding_max_str_num", 2)

    graph = Graph()
    assert graph.size() == 0
    assert graph.get_node("missing") is None
    assert graph.get_all_nodes() == []
    assert "Graph(nodes={})" in str(graph)
    with pytest.raises(NotImplementedError):
        graph.add_node()

    n1 = UndirectedNode(content="node1", label="L")
    n2 = UndirectedNode(content="node2", label="L")
    graph.nodes[n1.id] = n1
    graph.nodes[n2.id] = n2
    assert graph.find_node("node1", "L") is n1
    assert graph.find_node("none", "L") is None
    assert graph.get_all_nodes_by_label_list(["L"]) == [n1, n2]

    embedded = Graph.batch_embedding([n1, n2])
    assert all(item.embedding is not None for item in embedded)


def test_undirected_graph_node_operations_and_queries(monkeypatch):
    monkeypatch.setattr(graph_module, "APIBackend", lambda: _FakeAPIBackend())
    monkeypatch.setattr(vector_module, "APIBackend", lambda: _FakeAPIBackend())
    monkeypatch.setattr(llm_config_module.LLM_SETTINGS, "embedding_max_str_num", 2)

    u1 = UndirectedNode(content="A", label="group")
    u2 = UndirectedNode(content="B", label="group")
    u1.add_neighbor(u2)
    assert u2 in u1.get_neighbors()
    u1.remove_neighbor(u2)
    assert u2 not in u1.get_neighbors()
    assert "UndirectedNode(" in str(u1)
    assert "UndirectedNode(" in repr(u1)

    graph = UndirectedGraph()
    n1 = UndirectedNode(content="A", label="L1")
    n2 = UndirectedNode(content="B", label="L1")
    n3 = UndirectedNode(content="C", label="L2")
    n4 = UndirectedNode(content="A", label="L1")
    n4.id = "same-content-different-id"

    graph.add_node(n1)
    graph.add_node(n1)  # existing id branch
    graph.add_node(n4)  # find_node branch
    graph.add_node(n2, neighbor=n1)
    graph.add_nodes(n3, [])
    graph.add_nodes(n3, [n1, n2])
    assert graph.get_node(n1.id) is not None
    assert n1 in graph.get_node(n2.id).neighbors
    assert "UndirectedGraph(nodes=" in str(graph)

    near_model = graph.get_node_by_content("Model")
    assert near_model is None or isinstance(near_model, UndirectedNode)

    within_1 = graph.get_nodes_within_steps(n1, steps=1)
    assert all(isinstance(x, UndirectedNode) for x in within_1)
    within_block = graph.get_nodes_within_steps(n1, steps=2, constraint_labels=["L1"], block=True)
    assert all(node.label == "L1" for node in within_block)
    within_filtered = graph.get_nodes_within_steps(n1, steps=2, constraint_labels=["L2"])
    assert all(node.label == "L2" for node in within_filtered)

    with pytest.raises(AssertionError):
        graph.get_nodes_intersection([n1], steps=1)
    inter = graph.get_nodes_intersection([n1, n2], steps=2)
    assert isinstance(inter, list)

    sem_by_str = graph.semantic_search("A", topk_k=3, similarity_threshold=-1.0)
    sem_by_node = graph.semantic_search(n1, topk_k=3, similarity_threshold=-1.0)
    assert isinstance(sem_by_str, list)
    assert isinstance(sem_by_node, list)

    constraint_far = graph.query_by_node(
        n1,
        step=2,
        constraint_node=UndirectedNode(content="ZZZZZZ", label="Lx", embedding=[999.0, 999.0]),
        constraint_distance=0.999999,
    )
    assert constraint_far == []
    constraint_near = graph.query_by_node(
        n1,
        step=2,
        constraint_node=UndirectedNode(content="A", label="L1", embedding=n1.embedding),
        constraint_distance=0.0,
    )
    assert isinstance(constraint_near, list)

    with pytest.raises(TypeError, match="unexpected keyword argument 'content'"):
        graph.query_by_content("A", topk_k=3, step=2, similarity_threshold=-1.0)
    with pytest.raises(TypeError, match="unexpected keyword argument 'content'"):
        graph.query_by_content(["A", "B"], topk_k=2, step=2, similarity_threshold=-1.0)

    intersection = graph.intersection([n1, n2], [n2, n3])
    different = graph.different([n1, n2], [n2, n3])
    distance = graph.cal_distance(n1, n2)
    filtered = graph.filter_label([n1, n2, n3], ["L1"])
    assert intersection == [n2]
    assert n1 in different and n3 in different
    assert isinstance(distance, float)
    assert filtered == [n1, n2]

    graph.clear()
    assert graph.size() == 0


def test_undirected_graph_add_node_error_paths(monkeypatch):
    warnings = []
    monkeypatch.setattr(graph_module, "APIBackend", lambda: _FakeAPIBackend())
    monkeypatch.setattr(vector_module, "APIBackend", lambda: _FakeAPIBackend())
    monkeypatch.setattr(llm_config_module.LLM_SETTINGS, "embedding_max_str_num", 2)
    monkeypatch.setattr("quantaalpha.log.logger.warning", lambda msg: warnings.append(msg))

    graph = UndirectedGraph()
    bad_node = UndirectedNode(content="bad", label="L")
    bad_neighbor = UndirectedNode(content="bad-neighbor", label="L")

    monkeypatch.setattr(bad_node, "create_embedding", lambda: (_ for _ in ()).throw(RuntimeError("boom-node")))
    graph.add_node(bad_node)
    assert any("embedding failed" in str(msg) for msg in warnings)

    good_node = UndirectedNode(content="good", label="L")
    monkeypatch.setattr(
        bad_neighbor,
        "create_embedding",
        lambda: (_ for _ in ()).throw(RuntimeError("boom-neighbor")),
    )
    graph.add_node(good_node, neighbor=bad_neighbor)
    assert len(warnings) >= 2


def test_graph_coordinate_and_edge_utilities(monkeypatch):
    monkeypatch.setattr(graph_module.random, "SystemRandom", lambda: SimpleNamespace(uniform=lambda a, b: 0.5))
    edges = graph_to_edges({"a": ["b", "c"], "b": ["a"], "c": ["a"]})
    assert sorted(edges) == [("a", "b"), ("a", "c")]

    rnd_coords = assign_random_coordinate_to_node(["n1", "n2"], scope=1.0, origin=(1.0, 2.0))
    iso_coords = assign_isometric_coordinate_to_node(["n1", "n2"], x_step=2.0, x_origin=1.0, y_origin=3.0)
    curved = curly_node_coordinate({"n1": (0.0, 0.0), "n2": (0.5, 0.0)}, center_y=1.0, r=1.0)

    assert rnd_coords["n1"] == (1.5, 2.5)
    assert iso_coords["n1"] == (1.0, 3.0)
    assert iso_coords["n2"] == (3.0, 3.0)
    assert curved["n1"][1] == pytest.approx(2.0)


def test_graph_query_by_content_breaks_on_topk(monkeypatch):
    monkeypatch.setattr(graph_module, "APIBackend", lambda: _FakeAPIBackend())
    monkeypatch.setattr(vector_module, "APIBackend", lambda: _FakeAPIBackend())

    graph = UndirectedGraph()
    n1 = UndirectedNode(content="A", label="L1")
    n2 = UndirectedNode(content="B", label="L1")
    n3 = UndirectedNode(content="C", label="L1")
    graph.add_node(n1, n2)
    graph.add_node(n2, n3)

    with pytest.raises(TypeError, match="unexpected keyword argument 'content'"):
        graph.query_by_content(["A", "B", "C"], topk_k=1, step=1, similarity_threshold=-1.0)
