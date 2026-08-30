

import damast.core
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.processing_graph import Node, ProcessingGraph
from damast.core.transformations import MultiCycleTransformer, PipelineElement


def test_processing_graph():
    n0 = Node(name="0", transformer=MultiCycleTransformer(features=["lat"], n=180))
    n1 = Node(name="1", transformer=MultiCycleTransformer(features=["lat"], n=180))
    n2 = Node(name="2", transformer=MultiCycleTransformer(features=["lat"], n=180))
    n3 = Node(name="3", transformer=MultiCycleTransformer(features=["lat"], n=180))
    n4 = Node(name="4", transformer=MultiCycleTransformer(features=["lat"], n=180))

    graph = ProcessingGraph()
    graph.add_node(node=n0)
    graph.add_node(node=n1)
    graph.add_node(node=n2)
    graph.add_node(node=n3)
    graph.add_node(node=n4)

    for n in graph.nodes():
        node = graph[n.uuid]
        assert node == n


class Identity(PipelineElement):
    """
    Minimal single-input PipelineElement, defined at module level so it can be reconstructed
    by 'module_name:class_name' via a save/load roundtrip - see
    test_processing_graph_roundtrip_with_join below. MultiCycleTransformer (used in
    test_processing_graph above) is a plain Transformer, not a PipelineElement, and so isn't
    serializable via dict() - not suitable here.
    """
    @damast.core.describe("Identity")
    @damast.core.input({"x": {}})
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class JoinTwo(PipelineElement):
    """
    Minimal two-input operator, defined at module level for the same reason as `Identity`.
    """
    @damast.core.describe("Join two inputs")
    @damast.core.input({"x": {}})
    @damast.core.input({"x": {}}, label="other")
    @damast.core.output({"x": {}})
    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


def test_processing_graph_roundtrip_with_join():
    """
    Ensure multiple inputs are supported
    """
    graph = ProcessingGraph()
    graph.add_node(Node(name="a", transformer=Identity()))
    graph.join(name="other", operator=JoinTwo())

    # ProcessingGraph.from_dict() used to raise outright on the join node below - the
    # roundtrip itself succeeding is the primary regression check
    restored = ProcessingGraph.from_dict(dict(graph))

    assert {n.uuid for n in restored.nodes()} == {n.uuid for n in graph.nodes()}
    assert {(u.uuid, v.uuid) for u, v in restored.edges()} == {(u.uuid, v.uuid) for u, v in graph.edges()}

    join_node = next(n for n in restored.nodes() if n.name == "other" and not n.is_datasource())
    slots = {
        restored.get_edge_data(predecessor, join_node)["slot"]
        for predecessor in restored.predecessors(join_node)
    }
    assert slots == {"df", "other"}

    join_node = next(n for n in restored.nodes() if n.name == "other" and not n.is_datasource())
    slots = {
        restored.get_edge_data(predecessor, join_node)["slot"]
        for predecessor in restored.predecessors(join_node)
    }
    assert slots == {"df", "other"}
