"""
Generate a Mermaid flowchart of a `DataProcessingPipeline` - its steps, the dataflow between
them, and the input/output contract of each step and of the pipeline as a whole.

Mermaid (https://mermaid.js.org) is a plain-text diagram language rendered client-side in the
browser, so unlike `damast.viz.svg_export.SvgExporter` this needs no external binary at
render time. `to_mermaid`/`export_mermaid` produce the diagram source alone (e.g. to embed in a
Markdown file, or a page that already loads Mermaid); `to_html`/`export_html` wrap it into a
self-contained HTML page that loads Mermaid from a CDN.

Each processing element is drawn as a cluster of its own: an "Input (min required)" block
(one lean trapezoid per required column), the `transform` call, and an "Output (guaranteed)"
block - built from `PipelineExporter.datasource_facts`/`step_facts`/`output_columns`, which in
turn read the same sources `DataProcessingPipeline.describe`/`SvgExporter` already do (a step's
declared ``@input``/``@output`` decorators, plus `input_metadata`/`output_metadata` for a
datasource's requirement and the pipeline's own guaranteed output), so the diagram cannot drift
from the text description.

Rendering is Jinja-templated (`src/damast/viz/templates/mermaid/*.j2`, one small file per
element kind) rather than hard-coded in Python, so the Mermaid syntax/styling of a datasource,
a processing element, a single column, etc. can each be edited independently - either in the
shipped defaults, or by pointing `MermaidExporter(pipeline, template_dir=...)` at a directory
that overrides just the files it cares about.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union
from xml.sax.saxutils import escape

import jinja2

from damast.core.dataprocessing import DataProcessingPipeline
from damast.viz.pipeline_exporter import ColumnInfo, DataSourceFacts, PipelineExporter, StepFacts

__all__ = ["MermaidExporter"]

#: Mermaid build loaded by the HTML page `to_html`/`export_html` produce - pinned to the v11
#: line since the ``id@{ shape: ..., label: ... }`` node syntax used for the lean-trapezoid
#: input/output nodes needs Mermaid >= 11.3
_MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"

#: Package directory holding the default ``.j2`` templates, one per element kind
_DEFAULT_TEMPLATES_DIR = "templates/mermaid"

#: Max columns per row before a `ColumnBlock` wraps into multiple rows (`column_block.j2`,
#: exposed to it as the ``max_columns_per_row`` template global) - kept as one constant so the
#: row ids `_style_and_click_lines` derives to style them can't drift from what the template
#: actually renders
_MAX_COLUMNS_PER_ROW = 4

_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="{cdn}"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
  h1 {{ font-size: 1.25rem; }}
  .mermaid {{ margin-top: 1.5rem; }}
</style>
</head>
<body>
<h1>{title}</h1>
<pre class="mermaid">
{diagram}</pre>
<script>
  mermaid.initialize({{ startOnLoad: true }});
</script>
</body>
</html>
"""

class MermaidExporter(PipelineExporter):
    """
    Renders a `DataProcessingPipeline`'s steps, dataflow and interfaces to a Mermaid flowchart.

    Example:

    ```python
    exporter = MermaidExporter(pipeline)
    exporter.export_html(path="pipeline.html")
    ```

    Args:
        pipeline: The pipeline to visualize
        template_dir: Optional directory of ``.j2`` templates overriding the shipped defaults
            in `templates/mermaid/` - only the files present there are overridden (e.g. drop in
            a single ``datasource_block.j2`` to restyle just datasources); everything else
            keeps rendering from the default template of the same name
    """

    @dataclass
    class LeafNode:
        """One column - a required input or a guaranteed output - drawn as a lean trapezoid
        (pointing right for an input flowing in, left for an output flowing out)."""
        #: Mermaid node id (must be unique within the diagram)
        id: str
        #: ``"lean-r"`` for a required input, ``"lean-l"`` for a guaranteed output
        shape: str
        #: Already-escaped column label, e.g. ``"lat [unit: m]"``
        label: str
        #: ``"inputsStyle"`` or ``"outputsStyle"``
        style_class: str
        #: The column's description, if any - shown as a click-tooltip
        tooltip: Optional[str] = None

    @dataclass
    class TransformNode:
        """The step's ``transform`` call, drawn as a stadium between its input and output
        blocks."""
        #: Mermaid node id (must be unique within the diagram)
        id: str
        #: Node label - always ``"transform"``, the name of the method every step implements
        label: str = "⚙ transform"

    @dataclass
    class ColumnBlock:
        """A titled cluster of `LeafNode` columns - a processing element's input or output
        block, or the pipeline's overall guaranteed output."""
        #: Mermaid subgraph id (must be unique within the diagram)
        id: str
        #: Cluster title, e.g. ``"Input (min required)"``
        title: str
        #: ``"inputsBlockStyle"`` or ``"outputsBlockStyle"``
        style_class: str
        #: The columns in this block, in render order
        columns: List["MermaidExporter.LeafNode"]

    @dataclass
    class DataSourceBlock:
        """A datasource's required columns - structurally a `ColumnBlock`, but its own type so
        it renders from its own template (``datasource_block.j2``), independent of the column
        blocks nested inside a `ProcessingElement`."""
        #: Mermaid subgraph id (must be unique within the diagram)
        id: str
        #: Cluster title, e.g. ``"Input (DataSource)"``
        title: str
        #: The required columns, in render order
        columns: List["MermaidExporter.LeafNode"]
        #: Always ``"dataSourceNodeStyle"`` - kept as a field (rather than hard-coded in the
        #: template) so `datasource_block.j2` stays a trivial include of `column_block.j2`
        style_class: str = "dataSourceNodeStyle"

    @dataclass
    class ProcessingElement:
        """One processing step: one `ColumnBlock` per input slot, its `TransformNode`, and one
        `ColumnBlock` for its output - a fixed shape, not a generic recursive container."""
        #: Mermaid subgraph id (must be unique within the diagram)
        id: str
        #: The transformer's class name, shown as the cluster's title
        class_name: str
        #: One input block per input slot (more than one only for a join)
        input_blocks: List["MermaidExporter.ColumnBlock"]
        #: The step's ``transform`` call
        transform: "MermaidExporter.TransformNode"
        #: The step's guaranteed output columns
        output_block: "MermaidExporter.ColumnBlock"
        #: The step's ``@describe`` text, if any - shown as a click-tooltip on the cluster
        tooltip: Optional[str] = None

    @dataclass
    class Edge:
        """One dataflow connection - between two top-level elements, or from a processing
        element's input block to its `TransformNode` and on to its output block."""
        #: Id of the source node/subgraph
        source: str
        #: Id of the target node/subgraph
        target: str
        #: Input slot name - set only when the target has more than one input (a join)
        label: Optional[str] = None

    def __init__(self, pipeline: DataProcessingPipeline, template_dir: Optional[Union[str, Path]] = None) -> None:
        super().__init__(pipeline)

        loaders = []
        if template_dir is not None:
            loaders.append(jinja2.FileSystemLoader(str(template_dir)))
        loaders.append(jinja2.PackageLoader("damast.viz", _DEFAULT_TEMPLATES_DIR))

        self._env = jinja2.Environment(
            loader=jinja2.ChoiceLoader(loaders),
            autoescape=False,  # Mermaid isn't HTML; values are pre-escaped in Python instead
            trim_blocks=True,
            lstrip_blocks=True,
            # Jinja strips a template file's own final newline by default, which would glue an
            # {% include %}'d block's last statement onto whatever follows it in the parent -
            # Mermaid needs one statement per line, so keep it.
            keep_trailing_newline=True,
            undefined=jinja2.StrictUndefined,
        )
        self._env.globals["max_columns_per_row"] = _MAX_COLUMNS_PER_ROW

    def supported_filetypes(self) -> list[str]:
        return [".html", ".mmd"]

    @staticmethod
    def _safe_id(uuid: str) -> str:
        """Turn a `Node.uuid` into a valid Mermaid id (letters/digits/underscore only)."""
        return "n" + uuid.replace("-", "_")

    @staticmethod
    def _leaf_nodes(
            columns: List[ColumnInfo], *, shape: str, style_class: str, prefix: str
    ) -> List["MermaidExporter.LeafNode"]:
        """Build one `LeafNode` per column in ``columns``."""
        nodes = []
        for i, column in enumerate(columns):
            label = escape(column.name)
            if column.unit is not None:
                label += f" [unit: {escape(column.unit)}]"
            tooltip = escape(column.description) if column.description else None
            nodes.append(MermaidExporter.LeafNode(
                id=f"{prefix}{i}", shape=shape, label=label, style_class=style_class, tooltip=tooltip
            ))
        return nodes

    def _datasource_element(self, node_id: str, facts: DataSourceFacts) -> "MermaidExporter.DataSourceBlock":
        """Build the top-level cluster for a datasource node - its required columns only,
        since `DataSource`'s own declared input/output is always empty (see `input_metadata`)."""
        pe_id = self._safe_id(node_id)
        class_name = escape(facts.class_name)
        columns = self._leaf_nodes(
            facts.required_columns, shape="lean-r", style_class="inputsStyle", prefix=f"{pe_id}_I"
        )

        return self.DataSourceBlock(id=pe_id, title=f"⎆ {class_name}", columns=columns)

    def _processing_element(self, node_id: str, facts: StepFacts) -> "MermaidExporter.ProcessingElement":
        """Build the top-level cluster for a processing step: one "Input (min required)" block
        per input slot, its `transform` call, and one "Output (guaranteed)" block."""
        pe_id = self._safe_id(node_id)
        class_name = escape(facts.class_name)
        multi_slot = len(facts.input_slots) > 1

        input_blocks = []
        for label, slot_columns in facts.input_slots.items():
            title = f"Input ({escape(label)})" if multi_slot else "Input"
            input_id = f"{pe_id}_INPUTS_{label}"
            columns = self._leaf_nodes(
                slot_columns, shape="lean-r", style_class="inputsStyle", prefix=f"{input_id}_"
            )
            input_blocks.append(self.ColumnBlock(
                id=input_id, title=title, style_class="inputsBlockStyle", columns=columns
            ))

        transform = self.TransformNode(id=f"{pe_id}_TRANSFORM")

        output_id = f"{pe_id}_OUTPUTS"
        output_columns = self._leaf_nodes(
            facts.output_columns, shape="lean-l", style_class="outputsStyle", prefix=f"{output_id}_"
        )
        output_block = self.ColumnBlock(
            id=output_id, title="Output ", style_class="outputsBlockStyle", columns=output_columns
        )

        tooltip = escape(facts.description) if facts.description else None

        return self.ProcessingElement(
            id=pe_id, class_name=class_name, input_blocks=input_blocks,
            transform=transform, output_block=output_block, tooltip=tooltip,
        )

    def to_elements(
            self,
    ) -> tuple[
        List[Union["MermaidExporter.DataSourceBlock", "MermaidExporter.ProcessingElement", "MermaidExporter.ColumnBlock"]],
        List["MermaidExporter.Edge"],
    ]:
        """
        Build the top-level element/`Edge` representation of this exporter's pipeline - one
        cluster per processing step (plus a synthetic `ColumnBlock` for the pipeline's overall
        output), one edge per dataflow connection between them.

        Needs only the pipeline's processing graph and its declared decorators - no data, and
        no prior call to `prepare`, is required.

        Example:

        ```python
        exporter = MermaidExporter(pipeline)
        elements, edges = exporter.to_elements()
        ```

        Returns:
            A ``(top_level_clusters, edges)`` tuple

        Raises:
            RuntimeError: See `DataProcessingPipeline._declared_interface`
        """
        graph_nodes = list(self._pipeline.processing_graph.nodes())

        top_level = []
        element_id = {}
        for node in graph_nodes:
            if node.is_datasource():
                element = self._datasource_element(node.uuid, self.datasource_facts(node))
            else:
                element = self._processing_element(node.uuid, self.step_facts(node))
            top_level.append(element)
            element_id[node.uuid] = element.id

        edges = []
        for from_node, to_node, data in self._pipeline.processing_graph.edges(data=True):
            # only disambiguate the slot on a join's two incoming edges - a single-input step
            # has nothing to disambiguate
            label = data["slot"] if len(to_node.transformer.input_specs) > 1 else None
            edges.append(self.Edge(
                source=element_id[from_node.uuid], target=element_id[to_node.uuid], label=label
            ))

        if graph_nodes:
            sink = graph_nodes[-1]
            output_columns = self._leaf_nodes(
                self.output_columns(), shape="lean-l", style_class="outputsStyle",
                prefix="PIPELINE_OUTPUT_",
            )
            output_element = self.ColumnBlock(
                id="PIPELINE_OUTPUT", title="Output (guaranteed)",
                style_class="outputsBlockStyle", columns=output_columns,
            )
            top_level.append(output_element)
            edges.append(self.Edge(source=element_id[sink.uuid], target=output_element.id))

        return top_level, edges

    def _render_top_level(self, element: object) -> str:
        """Render one top-level element with the template matching its kind."""
        if isinstance(element, self.DataSourceBlock):
            return self._env.get_template("datasource_block.j2").render(block=element)
        if isinstance(element, self.ProcessingElement):
            return self._env.get_template("processing_element.j2").render(element=element)
        if isinstance(element, self.ColumnBlock):
            return self._env.get_template("column_block.j2").render(block=element)
        raise TypeError(f"{self.__class__.__name__}._render_top_level: unexpected element type {type(element)}")

    def _style_and_click_lines(
            self,
            top_level: List[Union["MermaidExporter.DataSourceBlock", "MermaidExporter.ProcessingElement",
                                   "MermaidExporter.ColumnBlock"]],
    ) -> tuple[List[str], List[str]]:
        """
        Collect every ``class``/``click`` statement for ``top_level``, to emit as one block at
        the end of the document - Mermaid only renders classes/clicks correctly when they come
        after the ``classDef`` declarations and after the nodes/subgraphs they refer to, not
        interleaved inside a ``subgraph ... end``.
        """
        class_lines: List[str] = []
        click_lines: List[str] = []

        def add_leaf(leaf: "MermaidExporter.LeafNode") -> None:
            class_lines.append(f"class {leaf.id} {leaf.style_class}")
            if leaf.tooltip:
                click_lines.append(f'click {leaf.id} "javascript:void(0)" "{leaf.tooltip}"')

        def add_block(block: Union["MermaidExporter.ColumnBlock", "MermaidExporter.DataSourceBlock"]) -> None:
            class_lines.append(f"class {block.id} {block.style_class}")
            for leaf in block.columns:
                add_leaf(leaf)

        for element in top_level:
            if isinstance(element, self.ProcessingElement):
                class_lines.append(f"class {element.id} processingElementStyle")
                if element.tooltip:
                    click_lines.append(f'click {element.id} "javascript:void(0)" "{element.tooltip}"')
                for input_block in element.input_blocks:
                    add_block(input_block)
                add_block(element.output_block)
            else:
                # DataSourceBlock and the top-level pipeline-output ColumnBlock
                add_block(element)

        return class_lines, click_lines

    def to_mermaid(self) -> str:
        """
        Render this exporter's pipeline to Mermaid flowchart source, via the Jinja templates in
        `templates/mermaid/` (or `template_dir`, for any overridden ones).

        Example:

        ```python
        exporter = MermaidExporter(pipeline)
        print(exporter.to_mermaid())
        ```

        Returns:
            The Mermaid ``flowchart`` diagram source

        Raises:
            jinja2.UndefinedError: If an overridden template references an undefined variable
        """
        top_level, edges = self.to_elements()

        body_parts = [self._render_top_level(element) for element in top_level]
        body_parts += [self._env.get_template("edge.j2").render(edge=edge) for edge in edges]

        class_lines, click_lines = self._style_and_click_lines(top_level)

        return self._env.get_template("document.j2").render(
            class_defs=self._env.get_template("class_defs.j2").render(),
            body="\n".join(body_parts),
            class_lines="\n".join(class_lines),
            click_lines="\n".join(click_lines),
        )

    def export_mermaid(self, path: Union[str, Path]) -> Path:
        """
        Render this exporter's pipeline to Mermaid flowchart source and write it to a `.mmd`
        file - just the diagram, not wrapped in an HTML page (see `export_html` for that).

        Example:

        ```python
        exporter = MermaidExporter(pipeline)
        exporter.export_mermaid(path="pipeline.mmd")
        ```

        Args:
            path: Destination file - parent directories are created as needed

        Returns:
            The path that was written
        """
        diagram = self.to_mermaid()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(diagram)
        return path

    def to_html(self, title: Optional[str] = None) -> str:
        """
        Render this exporter's pipeline to a self-contained HTML page with the flowchart,
        loading Mermaid from a CDN - open it in any browser, no local Mermaid install needed.

        Example:

        ```python
        exporter = MermaidExporter(pipeline)
        html = exporter.to_html()
        ```

        Args:
            title: Page title - defaults to ``"<pipeline.name> - pipeline flowchart"``

        Returns:
            The rendered HTML document
        """
        return _HTML_TEMPLATE.format(
            title=escape(title or f"{self._pipeline.name} - pipeline flowchart"),
            cdn=_MERMAID_CDN,
            diagram=self.to_mermaid(),
        )

    def export_html(self, path: Union[str, Path], title: Optional[str] = None) -> Path:
        """
        Render this exporter's pipeline to a self-contained HTML page and write it to a `.html`
        file.

        Example:

        ```python
        exporter = MermaidExporter(pipeline)
        exporter.export_html(path="pipeline.html")
        ```

        Args:
            path: Destination `.html` file - parent directories are created as needed
            title: Page title - defaults to ``"<pipeline.name> - pipeline flowchart"``

        Returns:
            The path that was written
        """
        html = self.to_html(title=title)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return path
