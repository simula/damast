"""
Generate an SVG visualization of a `DataProcessingPipeline` - its steps, the dataflow between
them, and the input/output contract of each step and of the pipeline as a whole - via Graphviz
(through `pydot`).

Each node's content is built from `PipelineExporter.datasource_facts`/`step_facts`/
`output_columns`, which in turn read the same sources `DataProcessingPipeline.describe` and
`PipelineElement.to_str` already read - a step's declared ``@describe``/``@input``/``@output``
decorators - plus `DataProcessingPipeline.input_metadata`/`output_metadata` for the pipeline's
own minimal input/output contract, so the diagram cannot drift from the text description.

.. note::
    Rendering to SVG (`SvgExporter.to_svg`/`export_svg`) requires the Graphviz ``dot``
    executable on ``PATH`` - `pydot` alone is not enough. `SvgExporter.to_pydot` needs neither
    and can be used to inspect the built graph structure without Graphviz installed.
"""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import pydot

from damast.viz.pipeline_exporter import (
    ColumnInfo,
    DataSourceFacts,
    PipelineExporter,
    StepFacts,
    )

__all__ = ["SvgExporter"]

#: Visual style for datasource boxes - distinguishes them from regular step boxes
_DATASOURCE_STYLE = {"style": "filled", "fillcolor": "#eef2ff"}
#: Visual style for the synthetic "pipeline output" box
_OUTPUT_STYLE = {"style": "filled", "fillcolor": "#f0fdf4"}


class SvgExporter(PipelineExporter):
    """
    Renders a `DataProcessingPipeline`'s steps, dataflow and interfaces to SVG via Graphviz.

    Example:

    ```python
    exporter = SvgExporter(pipeline)
    exporter.export_svg(path="pipeline.svg")
    ```

    Args:
        pipeline: The pipeline to visualize
    """

    def supported_filetypes(self) -> list[str]:
        return [".svg"]

    @staticmethod
    def _table(rows: list[str]) -> str:
        """Wrap ``rows`` of already-escaped HTML into a Graphviz HTML-like node label."""
        cells = "".join(f'<TR><TD ALIGN="LEFT">{row}</TD></TR>' for row in rows)
        return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">{cells}</TABLE>>'

    @staticmethod
    def _column_list(columns: list[ColumnInfo]) -> str:
        """Render a list of `ColumnInfo` as escaped, ``<BR/>``-separated column names."""
        return "<BR/>".join(escape(column.name) for column in columns) or "-"

    def _node_label(self, facts: DataSourceFacts | StepFacts) -> str:
        """Build the HTML-like label for one datasource/step node's facts."""
        rows = [f"<B>{escape(facts.name)}</B> ({escape(facts.class_name)})"]

        if facts.description:
            rows.append(escape(facts.description))

        if isinstance(facts, DataSourceFacts):
            rows.append(f"required:<BR/>{self._column_list(facts.required_columns)}")
        else:
            show_labels = len(facts.input_slots) > 1
            for label, columns in facts.input_slots.items():
                prefix = f"in ({escape(label)}):" if show_labels else "in:"
                rows.append(f"{prefix}<BR/>{self._column_list(columns)}")
            rows.append(f"out:<BR/>{self._column_list(facts.output_columns)}")

        return self._table(rows)

    def to_pydot(self) -> pydot.Dot:
        """
        Build the `pydot.Dot` graph for this exporter's pipeline - one box per processing step
        (plus a synthetic box for the pipeline's overall output), one arrow per dataflow
        connection.

        Needs only the pipeline's processing graph and its declared decorators - no data, and
        no prior call to `prepare`, is required. Unlike `to_svg`/`export_svg`, this does not
        need Graphviz installed.

        Example:

        ```python
        exporter = SvgExporter(pipeline)
        graph = exporter.to_pydot()
        ```

        Returns:
            A `pydot.Dot` graph, ready for `pydot.Dot.create_svg` or another Graphviz format

        Raises:
            RuntimeError: See `DataProcessingPipeline._declared_interface`
        """
        graph = pydot.Dot(self._pipeline.name, graph_type="digraph", rankdir="LR")

        nodes = list(self._pipeline.processing_graph.nodes())
        for node in nodes:
            if node.is_datasource():
                style, facts = _DATASOURCE_STYLE, self.datasource_facts(node)
            else:
                style, facts = {}, self.step_facts(node)
            graph.add_node(pydot.Node(
                node.uuid, shape="plain", label=self._node_label(facts), **style
            ))

        for from_node, to_node, data in self._pipeline.processing_graph.edges(data=True):
            # only disambiguate the slot on a join's two incoming edges - a single-input step
            # has nothing to disambiguate
            edge_kwargs = {"label": data["slot"]} if len(to_node.transformer.input_specs) > 1 else {}
            graph.add_edge(pydot.Edge(from_node.uuid, to_node.uuid, **edge_kwargs))

        if nodes:
            sink = nodes[-1]
            output_columns = self._column_list(self.output_columns())
            output_id = f"{sink.uuid}-output"
            graph.add_node(pydot.Node(
                output_id, shape="plain",
                label=self._table(["<B>pipeline output</B>", f"out:<BR/>{output_columns}"]),
                **_OUTPUT_STYLE,
            ))
            graph.add_edge(pydot.Edge(sink.uuid, output_id))

        return graph

    def to_svg(self) -> str:
        """
        Render this exporter's pipeline to an SVG document via Graphviz.

        Example:

        ```python
        exporter = SvgExporter(pipeline)
        svg = exporter.to_svg()
        ```

        Returns:
            The rendered SVG document

        Raises:
            RuntimeError: If the Graphviz ``dot`` executable is not installed or not on ``PATH``
        """
        try:
            svg_bytes = self.to_pydot().create_svg()
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{self.__class__.__name__}.to_svg: rendering requires the Graphviz 'dot'"
                " executable to be installed and on PATH - see https://graphviz.org/download/"
            ) from e

        return svg_bytes.decode("utf-8")

    def export_svg(self, path: str | Path) -> Path:
        """
        Render this exporter's pipeline to SVG and write it to a `.svg` file.

        Example:

        ```python
        exporter = SvgExporter(pipeline)
        exporter.export_svg(path="pipeline.svg")
        ```

        Args:
            path: Destination `.svg` file - parent directories are created as needed

        Returns:
            The path that was written

        Raises:
            RuntimeError: If the Graphviz ``dot`` executable is not installed or not on ``PATH``
        """
        svg = self.to_svg()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(svg)
        return path
