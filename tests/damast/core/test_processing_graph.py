import pytest
import uuid
from damast.core.processing_graph import Node, ProcessingGraph
from damast.core.transformations import PipelineElement, CycleTransformer

def test_processing_graph():
    n0 = Node(name="0", transformer=CycleTransformer(features=["lat"], n=180))
    n1 = Node(name="1", transformer=CycleTransformer(features=["lat"], n=180))
    n2 = Node(name="2", transformer=CycleTransformer(features=["lat"], n=180))
    n3 = Node(name="3", transformer=CycleTransformer(features=["lat"], n=180))
    n4 = Node(name="4", transformer=CycleTransformer(features=["lat"], n=180))

    graph = ProcessingGraph()
    graph.add(node=n0)
    graph.add(node=n1)
    graph.add(node=n2)

    for n in graph.nodes():
        node = graph[n.uuid]
        assert node == n
