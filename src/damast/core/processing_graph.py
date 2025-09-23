from __future__ import annotations

import networkx as nx
import uuid as uuid_module
from damast.core.transformations import PipelineElement
from damast.core.formatting import DEFAULT_INDENT
from .decorators import (
    artifacts,
    describe,
    output,
    input
)

class DataSource(PipelineElement):
    def __init__(self):
        pass

    @describe("Node for DataSource")
    @input({})
    @output({})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class Node:
    name: str
    transformer: PipelineElement

    def __init__(self, name: str, transformer: PipelineElement, uuid: str | None = None):
        self.name = name
        self.transformer = transformer

        if uuid is None:
            self.transformer.set_uuid(str(uuid_module.uuid4()))
        else:
            self.transformer.set_uuid(uuid)


    def __repr__(self):
        return f"name={self.name} {self.transformer.__class__.__name__} (uuid={self.uuid})"

    @property
    def uuid(self):
        return self.transformer.uuid

    def to_str(self, indent_level: int = 0) -> str:
        """
        Output as string.

        :param indent_level: Indentation per step.
            It is multiplied by :attr:`damast.core.formatting.DEFAULT_INDENT`.
        """
        hspace = DEFAULT_INDENT * indent_level

        data = hspace + DEFAULT_INDENT + self.name + ":\n"
        data += self.transformer.to_str(indent_level+4)
        return data

    def __iter__(self):
        yield "uuid", str(self.uuid)
        yield "name", self.name
        yield "transformer", dict(self.transformer)

    @classmethod
    def from_dict(cls, data: dict[str, any]):
        return Node(uuid=data['uuid'],
                    name=data['name'],
                    transformer=PipelineElement.create_new(**data['transformer'])
                )


class ProcessingGraph:
    _graph: nx.DiGraph

    _root_node: Node
    _leaf_node: Node

    def __init__(self):
        self._graph = nx.DiGraph()
        self._leaf_node = None
        self._root_node = None

    def __getitem__(self, uuid: str):
        for n in self._graph.nodes():
            if n.uuid == uuid:
                return n

        raise KeyError(f"Could not find node with {uuid=}")

    def __iter__(self):
        yield "nodes", [dict(x) for x in self._graph.nodes()]
        yield "edges", [{'from': str(x[0].uuid), 'to': str(x[1].uuid)} for x in self._graph.edges()]

    @classmethod
    def from_dict(cls, data: dict[str, any]): 
        if 'nodes' not in data: 
            raise KeyError("ProcessingGraph.from_dict: missing 'nodes'")

        if 'edges' not in data: 
            raise KeyError("ProcessingGraph.from_dict: missing 'edges'")

        graph = cls()
        for node_dict in data['nodes']:
            node = Node.from_dict(node_dict)
            graph.add(node)

        for edge_dict in data['edges']:
            from_node = graph[edge_dict['from']]
            to_node = graph[edge_dict['to']]

            graph.add_edge(from_node, to_node)

        return graph

    def __eq__(self, other: ProcessingGraph) -> bool:
        return nx.utils.graphs_equal(self._graph, other._graph)

    def add(self, node: Node):
        if not self._root_node:
            self._root_node = node

        self._graph.add_node(node)
        # automatically create an edge between the nodes
        if self._leaf_node is not None:
            self._graph.add_edge(self._leaf_node, node)

        self._leaf_node = node

    def join(self, name: str, operator: PipelineOperator, processing_graph: ProcessingGraph | None = None):
        leaf_node = None
        if processing_graph:
            for n in processing_graph.nodes():
                self._graph.add_node(n)
                if self._graph.out_degree(n) == 0:
                    leaf_node = n

            for e in processing_graph.edges():
                self._graph.add_edge(e)

        if not leaf_node:
            # we require at least a data loader stub node
            leaf_node = Node(f"__{name}__", DataSource())
            self._graph.add_node(leaf_node)

        node = Node(name, operator)
        self._graph.add_node(node)

        self._graph.add_edge(leaf_node, node)
        self._graph.add_edge(self._leaf_node, node)

        self._leaf_node = node

    def to_str(self, indent_level = 0):
        return '\n'.join(list(nx.generate_network_text(self._graph, ascii_only=True)))

        data = ""
        #nx.write_network_text(self._graph)
        #reversed_graph = self._graph.reverse()
        #for node in nx.bfs_tree(reversed_graph, self._leaf_node):
        #    if len(list(self._graph.successors(node))) == 2:
        #        print("JOIN NODE")
        #    data += node.to_str(indent_level=indent_level)
        ##nx.write_network_text(reversed_graph)
        #return data

    def get_joins(self):
        joins = []
        for n in self._graph.nodes():
            if self._graph.in_degree(n) == 2:
                joins.append(n)
        return joins

    def __getattr__(self, name):
        return getattr(self._graph, name)


