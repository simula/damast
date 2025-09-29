from __future__ import annotations

import copy
import inspect
import uuid as uuid_module

import networkx as nx

from damast.core.formatting import DEFAULT_INDENT
from damast.core.transformations import PipelineElement

from .decorators import DAMAST_DEFAULT_DATASOURCE, artifacts, describe, input, output


class DataSource(PipelineElement):
    """
    PipelineElement that marks a datasource, e.g.,
    an annotated dataframe that needs to be processed
    """
    def __init__(self):
        super().__init__()

    @describe("Node for marking a plain data entry")
    @input({})
    @output({})
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class Node:
    name: str
    transformer: PipelineElement

    #: Cache computation results
    result: AnnotatedDataFrame
    #: Cache validation results
    validation_output_spec: list[DataSpecification]

    def __init__(self, name: str, transformer: PipelineElement, uuid: str | None = None):
        self.name = name
        self.transformer = transformer

        if uuid is None:
            self.transformer.set_uuid(str(uuid_module.uuid4()))
        else:
            self.transformer.set_uuid(uuid)

    def __repr__(self):
        return f"name={self.name} {self.transformer.__class__.__name__} (uuid={self.uuid})"

    def inputs(self) -> list[str]:
        """
        Get all inputs for AnnotatedDataFrame
        """
        inputs = []
        for x, y in inspect.signature(self.transformer.transform).parameters.items():
            if y.annotation == 'AnnotatedDataFrame' or y.annotation.__name__ == 'AnnotatedDataFrame':
                inputs.append(x)
        return inputs

    def is_datasource(self) -> bool:
        return type(self.transformer) == DataSource

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

    def __init__(self, with_datasource: bool = True):
        self._graph = nx.DiGraph()
        self._root_node = None
        self._leaf_node = None

        if with_datasource:
            self.add_datasource()

    def add_datasource(self):
        """
        Add a datasource node as root node to the graph
        (only if the root node has not been set)
        """
        if self._root_node:
            raise RuntimeError("ProcessingGraph.add_datasource: adding a "
                               " datasource is only possible for an empty graph")

        self._root_node = Node(DAMAST_DEFAULT_DATASOURCE, DataSource())
        self._graph.add_node(self._root_node)

        self._leaf_node = self._root_node

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
        """
        Load the graph from dictionary containing nodes and edges.
        Note, that no internal states can be loaded.
        """
        if 'nodes' not in data:
            raise KeyError("ProcessingGraph.from_dict: missing 'nodes'")

        if 'edges' not in data:
            raise KeyError("ProcessingGraph.from_dict: missing 'edges'")

        graph = cls(with_datasource=False)

        for node_dict in data['nodes']:
            node = Node.from_dict(node_dict)
            graph.add(node)

        for edge_dict in data['edges']:
            from_node = graph[edge_dict['from']]
            to_node = graph[edge_dict['to']]

            graph.add_edge(from_node, to_node)

        return graph

    def __eq__(self, other: ProcessingGraph) -> bool:
        """
        Compares the underlying graphs
        """
        return nx.utils.graphs_equal(self._graph, other._graph)

    def add(self, node: Node):
        """
        Add (or rather appaned) a node to the current graph
        It will attach to the graph's current leaf node
        """
        if not self._root_node:
            self._root_node = node

        self._graph.add_node(node)

        required_slots = node.inputs()
        if len(required_slots) != 1:
            raise RuntimeError(f"{node} needs to have exactly one dataframe input argument",
                               " found {required_slots}")

        if self._leaf_node is not None:
            self._graph.add_edge(self._leaf_node, node, slot=required_slots[0])
        self._leaf_node = node

    def join(self, name: str, operator: PipelineOperator, processing_graph: ProcessingGraph | None = None):
        """
        Join another processing graph into the current one.
        This will an new 'join' node for the given operator.
        If no processing_graph is given, then a DataSource node will be created and added to the
        graph label by :name
        """
        leaf_node = None

        # if this is to join a processing graph
        # add all nodes and edges
        if processing_graph:
            for n in processing_graph.nodes():
                # Reset the source node of this processing graph
                # to the name of this join
                if n.is_datasource():
                    n.name = name

                self._graph.add_node(n)
                if self._graph.out_degree(n) == 0:
                    leaf_node = n

            for from_n, to_n, data in processing_graph.edges(data=True):
                self._graph.add_edge(from_n, to_n, **data)

        if not leaf_node:
            # we require at least a data loader stub node
            # by convention name after the 'join' node
            leaf_node = Node(name, DataSource())
            self._graph.add_node(leaf_node)

        # we consider order here and assume max 2
        node = Node(name, operator)
        required_slots = node.inputs()
        if len(required_slots) != 2:
            raise RuntimeError(f"ProcessingGraph.join: requires exactly two dataframe argument, but found {required_slots}")

        self._graph.add_node(node)

        # Allow to point to the argument in the transformer / operator which will
        # receive the input from the connection predecessor
        self._graph.add_edge(leaf_node, node, slot=required_slots[1])
        self._graph.add_edge(self._leaf_node, node, slot=required_slots[0])

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

    def get_joins(self, in_degree: int = 2):
        """
        Get nodes, that have an in_degree of :degree (default is two)
        """
        if in_degree < 2:
            raise ValueError("ProcessingGraph.get_joins: no join nodes with an in degree"
                             " of less than 2 possible")
        joins = []
        for n in self._graph.nodes():
            if self._graph.in_degree(n) == in_degree:
                joins.append(n)
        return joins

    def __getattr__(self, name):
        # forward calls to the underlying graph
        return getattr(self._graph, name)

    def inputs_ready(self, node: Node) -> bool:
        """
        Check if the results from predecessor nodes that this
        node computation depends upon are ready for ingestion
        """
        for i in node.inputs():
            for x,y in self._graph.in_edges(node):
                if not hasattr(x, 'result') or x.result is None:
                    return False

        return True

    def get_current_inputs(self, node: Node) -> bool:
        """
        Get the current inputs
        """
        inputs = {}
        for i in node.inputs():
            for x,y,data in self._graph.in_edges(node, data=True):
                if hasattr(x, 'result'):
                    inputs[data['slot']] = x.result
        return inputs

    def clear_state(self):
        """
        Clear the internal state including computation and validation results
        """
        for n in self._graph.nodes():
            n.result = None
            n.validation_output_spec = None

    def __len__(self) -> int:
        return len(self._graph)

    def __deepcopy__(self, memo) -> ProcessingGraph:
        new_graph = ProcessingGraph(with_datasource=False)
        node_map = {}
        for n in self._graph.nodes():
            new_node = Node(name=f"{n.name}", transformer=copy.deepcopy(n.transformer), uuid=f"{n.uuid}")
            new_graph.add_node(new_node)

        for from_n, to_n, data in self._graph.edges(data=True):
            new_graph.add_edge(new_graph[from_n.uuid], new_graph[to_n.uuid], **data.copy())

        return new_graph


    def execute(self, node: Node) -> AnnotatedDataFrame:
        """
        Run the transformation of this node
        :raise RuntimeError when the results from other nodes are not yet available
        """
        kwargs = {}
        if not self.inputs_ready(node):
            raise RuntimeError(f"ProcessingGraph.node: {node} inputs are not ready")

        for i in node.inputs():
            for from_node,to_node,data in self._graph.in_edges(node, data=True):
                if data['slot'] == i:
                    kwargs[data['slot']] = from_node.result
        return node.transformer.fit_transform(**kwargs)


