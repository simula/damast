"""
Generate a Mermaid flowchart of a `DataProcessingPipeline` - its steps, the dataflow between
them, and the input/output contract of each step and of the pipeline as a whole.

Mermaid (https://mermaid.js.org) is a plain-text diagram language rendered client-side in the
browser, so unlike `damast.viz.svg_export.SvgExporter` this needs no external binary at
render time. `to_mermaid`/`export_mermaid` produce the diagram source alone (e.g. to embed in a
Markdown file, or a page that already loads Mermaid); `to_html`/`export_html` wrap it into a
self-contained HTML page that loads Mermaid from a CDN.

Each processing element is drawn as a cluster of its own: an "Input" block
(one lean trapezoid per required column), the `transform` call, and an "Output"
block - built from `PipelineExporter.datasource_facts`/`step_facts`/`output_columns`, which in
turn read the same sources `DataProcessingPipeline.describe`/`SvgExporter` already do (a step's
declared ``@input``/``@output`` decorators, plus `input_metadata`/`output_metadata` for a
datasource's requirement and the pipeline's own output), so the diagram cannot drift
from the text description.

Rendering is Jinja-templated (`src/damast/viz/templates/mermaid/*.j2`, one small file per
element kind) rather than hard-coded in Python, so the Mermaid syntax/styling of a datasource,
a processing element, a single column, etc. can each be edited independently - either in the
shipped defaults, or by pointing `MermaidExporter(pipeline, template_dir=...)` at a directory
that overrides just the files it cares about.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

import jinja2

from damast.core.dataprocessing import DataProcessingPipeline
from damast.viz.pipeline_exporter import (
    ColumnInfo,
    DataSourceFacts,
    PipelineExporter,
    StepFacts,
)

__all__ = ["MermaidExporter"]

#: Mermaid build loaded by the HTML page `to_html`/`export_html` produce - pinned to the v11
#: line since the ``id@{ shape: ..., label: ... }`` node syntax used for the lean-trapezoid
#: input/output nodes needs Mermaid >= 11.3, and the collapsible-subgraph
#: ``id@{ view: collapsed }`` metadata `to_html`'s click-to-collapse relies on needs >= 11.17
_MERMAID_CDN = "https://cdn.jsdelivr.net/npm/mermaid@11.17/dist/mermaid.min.js"

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
  #graph {{ margin-top: 1.5rem; }}
  /* every occurrence of a hovered column, see wireHoverHighlight */
  .col-highlight path, .col-highlight rect, .col-highlight polygon {{
    stroke: #ff5722 !important;
    stroke-width: 3px !important;
  }}
  /* Mermaid sets its click-tooltip's font-size as a plain (non-!important) inline style -
     match it to the Input/Output block title size (inputsBlockStyle/outputsBlockStyle) */
  .mermaidTooltip {{
    font-size: 18px !important;
    font-family: sans-serif;
    background-color: #cceec8 !important;
    max-width: 400px !important;
    border-radius: 15px;
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<div id="graph"></div>
<script>
  // securityLevel 'loose' is required for the per-column description tooltips (a plain
  // 'click id "..."' binding) to work - collapsing itself no longer goes through Mermaid's
  // click mechanism at all, see wireCollapseClicks below
  mermaid.initialize({{ startOnLoad: false, securityLevel: 'loose' }});

  // must match the id passed to mermaid.render() below - used again to strip its prefix off
  // rendered element ids in wireCollapseClicks
  const RENDER_ID = "graph-svg";

  // the plain diagram source, with no 'view: collapsed' lines - collapsed state lives only in
  // this page's `collapsed` set below, appended back in before every (re-)render
  const baseDiagram = {diagram_json};
  // {{leaf node id: column name}}, for wireHoverHighlight - kept out of the Mermaid source
  // itself so a column name never needs sanitizing into a Mermaid class identifier
  const columnNames = {column_names_json};
  const collapsed = new Set();

  async function renderDiagram() {{
    const overrides = [...collapsed].map((id) => `${{id}}@{{ view: collapsed }}`).join("\\n");
    const source = overrides ? `${{baseDiagram}}\\n${{overrides}}` : baseDiagram;
    const {{ svg, bindFunctions }} = await mermaid.render(RENDER_ID, source);
    const container = document.getElementById("graph");
    container.innerHTML = svg;
    // mermaid.render() does not wire up 'click' bindings itself (unlike its startOnLoad path) -
    // bindFunctions attaches the tooltip clicks to the SVG just inserted
    if (bindFunctions) {{
      bindFunctions(container);
    }}
    wireCollapseClicks(container);
    wireHoverHighlight(container);
  }}

  // the same column shows up under several ids (required by a datasource, a step's
  // input/output, the pipeline's overall output) - group this render's leaf nodes by column
  // name (via the columnNames map built server-side) and highlight every occurrence together
  // on hover, so a column's path through the pipeline is easy to trace visually
  function wireHoverHighlight(container) {{
    const nodePrefix = RENDER_ID + "-flowchart-";
    const elementsByName = new Map();
    container.querySelectorAll("g.node[id]").forEach((el) => {{
      if (!el.id.startsWith(nodePrefix)) {{
        return;
      }}
      // mermaid renders a node's id as "<nodePrefix><ourId>-<counter>" - our own ids never
      // end in "-<digits>" (they use underscores), so this only strips mermaid's own suffix
      const id = el.id.slice(nodePrefix.length).replace(/-\\d+$/, "");
      const name = columnNames[id];
      if (!name) {{
        return;
      }}
      if (!elementsByName.has(name)) {{
        elementsByName.set(name, []);
      }}
      elementsByName.get(name).push(el);
    }});
    elementsByName.forEach((elements) => {{
      if (elements.length < 2) {{
        return;
      }}
      // a leaf node's own shape (the "label-container" polygon/path/rect) carries an inline
      // style="...!important" from its inputsStyle/outputsStyle classDef, which beats the
      // .col-highlight stylesheet rule even though that rule also uses !important - so besides
      // toggling the class (for any shape with no inline style), overwrite every occurrence's
      // inline style directly on hover, restoring each one's original on mouseleave
      const shapes = elements.map((el) => el.querySelector(".label-container") || el.querySelector("polygon, path, rect"));
      const originalStyles = shapes.map((shape) => (shape ? shape.getAttribute("style") : null));
      elements.forEach((el) => {{
        el.addEventListener("mouseenter", () => {{
          elements.forEach((e) => e.classList.add("col-highlight"));
          shapes.forEach((shape) => {{
            if (shape) {{
              shape.style.setProperty("stroke", "#ff5722", "important");
              shape.style.setProperty("stroke-width", "3px", "important");
            }}
          }});
        }});
        el.addEventListener("mouseleave", () => {{
          elements.forEach((e) => e.classList.remove("col-highlight"));
          shapes.forEach((shape, i) => {{
            if (shape) {{
              shape.setAttribute("style", originalStyles[i] || "");
            }}
          }});
        }});
      }});
    }});
  }}

  // Mermaid does not fire 'click' bindings on a subgraph's own id at all, expanded or
  // collapsed (mermaid-js/mermaid#5428) - so instead of relying on Mermaid's click mechanism,
  // find every element carrying the 'collapsible' class ourselves (ColumnBlock/DataSourceBlock/
  // ProcessingElement all get it, from MermaidExporter._style_and_click_lines) - matches the
  // cluster while expanded and, once collapsed, the single node Mermaid replaces it with -
  // and attach a real click listener directly
  function wireCollapseClicks(container) {{
    const prefix = RENDER_ID + "-";
    container.querySelectorAll("g.collapsible").forEach((el) => {{
      if (!el.id.startsWith(prefix)) {{
        return;
      }}
      const id = el.id.slice(prefix.length);
      el.addEventListener("click", (event) => {{
        // stop a click on a nested block (e.g. a step's "Input" block) from also toggling
        // every ancestor subgraph it sits inside (e.g. the step itself)
        event.stopPropagation();
        if (collapsed.has(id)) {{
          collapsed.delete(id);
        }} else {{
          collapsed.add(id);
        }}
        renderDiagram();
      }});
    }});
  }}

  renderDiagram();
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
        """One column - a input or a output - drawn as a lean trapezoid
        (pointing right for an input flowing in, left for an output flowing out)."""
        #: Mermaid node id (must be unique within the diagram)
        id: str
        #: ``"lean-r"`` for input, ``"lean-l"`` for output
        shape: str
        #: Already-escaped column label, e.g. ``"lat [unit: m]"``
        label: str
        #: ``"inputsStyle"`` or ``"outputsStyle"``
        style_class: str
        #: The bare column name (unescaped) - the same column shows up under several ids
        #: (required by a datasource, a step's input/output, the pipeline's overall output);
        #: `to_html`'s JS uses this to highlight every occurrence of one column on hover
        name: str
        #: The column's description, if any - shown as a click-tooltip
        tooltip: str | None = None

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
        block, or the pipeline's overall output."""
        #: Mermaid subgraph id (must be unique within the diagram)
        id: str
        #: Cluster title, e.g. ``"Input"``
        title: str
        #: ``"inputsBlockStyle"`` or ``"outputsBlockStyle"``
        style_class: str
        #: The columns in this block, in render order
        columns: list[MermaidExporter.LeafNode]

    @dataclass
    class DataSourceBlock:
        """A datasource's required columns - structurally a `ColumnBlock`, but its own type so
        it renders from its own template (``datasource_block.j2``), independent of the column
        blocks nested inside a `ProcessingElement`."""
        #: Mermaid subgraph id (must be unique within the diagram)
        id: str
        #: Cluster title, e.g. ``"DataSource"``
        title: str
        #: The required columns, in render order
        columns: list[MermaidExporter.LeafNode]
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
        input_blocks: list[MermaidExporter.ColumnBlock]
        #: The step's ``transform`` call
        transform: MermaidExporter.TransformNode
        #: The step's output columns
        output_block: MermaidExporter.ColumnBlock
        #: The step's ``@describe`` text, if any - shown as a click-tooltip on the cluster
        tooltip: str | None = None

    @dataclass
    class Edge:
        """One dataflow connection - between two top-level elements, or from a processing
        element's input block to its `TransformNode` and on to its output block."""
        #: Id of the source node/subgraph
        source: str
        #: Id of the target node/subgraph
        target: str
        #: Input slot name - set only when the target has more than one input (a join)
        label: str | None = None

    def __init__(self, pipeline: DataProcessingPipeline, template_dir: str | Path | None = None) -> None:
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
            columns: list[ColumnInfo], *, shape: str, style_class: str, prefix: str
    ) -> list[MermaidExporter.LeafNode]:
        """Build one `LeafNode` per column in ``columns``."""
        nodes = []
        for i, column in enumerate(columns):
            label = escape(column.name)
            if column.unit is not None:
                label += f"\nunit: {escape(column.unit)}"

            tooltip = ""
            if column.representation_type:
                tooltip += f"type: {escape(column.representation_type)}"
            if column.description:
                if tooltip != "":
                    tooltip += "<br/>---<br/>"
                tooltip += f"{escape(column.description)}"

            nodes.append(MermaidExporter.LeafNode(
                id=f"{prefix}{i}", shape=shape, label=label, style_class=style_class,
                name=column.name, tooltip=tooltip,
            ))
        return nodes

    def _datasource_element(self, node_id: str, facts: DataSourceFacts) -> MermaidExporter.DataSourceBlock:
        """Build the top-level cluster for a datasource node - its required columns only,
        since `DataSource`'s own declared input/output is always empty (see `input_metadata`)."""
        pe_id = self._safe_id(node_id)
        class_name = escape(facts.class_name)
        columns = self._leaf_nodes(
            facts.required_columns, shape="lean-r", style_class="inputsStyle", prefix=f"{pe_id}_I"
        )

        return self.DataSourceBlock(id=pe_id, title=f"⎆ {class_name}", columns=columns)

    def _processing_element(self, node_id: str, facts: StepFacts) -> MermaidExporter.ProcessingElement:
        """Build the top-level cluster for a processing step: one "Input" block
        per input slot, its `transform` call, and one "Output" block."""
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
        list[MermaidExporter.DataSourceBlock | MermaidExporter.ProcessingElement | MermaidExporter.ColumnBlock],
        list[MermaidExporter.Edge],
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
                id="PIPELINE_OUTPUT", title="Output",
                style_class="pipelineOutputsBlockStyle", columns=output_columns,
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
            top_level: list[MermaidExporter.DataSourceBlock | MermaidExporter.ProcessingElement | MermaidExporter.ColumnBlock],
            *,
            interactive: bool,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """
        Collect every ``class``/``click`` statement for ``top_level``, to emit as one block at
        the end of the document - Mermaid only renders classes/clicks correctly when they come
        after the ``classDef`` declarations and after the nodes/subgraphs they refer to, not
        interleaved inside a ``subgraph ... end`` - plus a ``{leaf id: column name}`` map for
        every `LeafNode`, so `to_html`'s JS can highlight every occurrence of one column on
        hover without sanitizing column names into Mermaid class identifiers itself.

        If ``interactive``, every subgraph (`DataSourceBlock`, `ColumnBlock`,
        `ProcessingElement`) also gets the ``collapsible`` marker class. Nothing in the Mermaid
        source binds a click to it: Mermaid does not fire ``click`` bindings on a subgraph's own
        id at all, expanded or collapsed (github.com/mermaid-js/mermaid/issues/5428) - so
        `to_html`'s JS instead renders the SVG, queries it for ``g.collapsible`` elements itself
        (both the expanded cluster and, once collapsed, the single node Mermaid replaces it
        with carry this class) and attaches a real click listener to each, deriving which id to
        toggle from the element's own DOM id.

        A subgraph's `class` assignment is dropped once collapsed unless some edge references
        its id (verified against a real Mermaid render) - every collapsible element here already
        has one from the pipeline's own dataflow (a `ColumnBlock` connects to its
        `ProcessingElement`'s `transform`; every top-level element sits on the edges `to_elements`
        already draws between steps), so no extra edge needs adding just for this.

        `to_mermaid`/`export_mermaid` pass ``interactive=False``: a portable ``.mmd`` file has no
        page-side JS to match ``collapsible`` elements, so it must not depend on one.
        """
        class_lines: list[str] = []
        click_lines: list[str] = []
        column_names: dict[str, str] = {}

        def add_leaf(leaf: MermaidExporter.LeafNode) -> None:
            class_lines.append(f"class {leaf.id} {leaf.style_class}")
            if leaf.tooltip:
                click_lines.append(f'click {leaf.id} "javascript:void(0)" "{leaf.tooltip}"')
            column_names[leaf.id] = leaf.name

        def mark_collapsible(subgraph_id: str) -> None:
            if interactive:
                class_lines.append(f"class {subgraph_id} collapsible")

        def add_block(block: MermaidExporter.ColumnBlock | MermaidExporter.DataSourceBlock) -> None:
            class_lines.append(f"class {block.id} {block.style_class}")
            mark_collapsible(block.id)
            for leaf in block.columns:
                add_leaf(leaf)

        for element in top_level:
            if isinstance(element, self.ProcessingElement):
                class_lines.append(f"class {element.id} processingElementStyle")
                mark_collapsible(element.id)
                if element.tooltip:
                    click_lines.append(f'click {element.id}_TRANSFORM "javascript:void(0)" "{element.tooltip}"')
                for input_block in element.input_blocks:
                    add_block(input_block)
                add_block(element.output_block)
            else:
                # DataSourceBlock and the top-level pipeline-output ColumnBlock
                add_block(element)

        return class_lines, click_lines, column_names

    def _render(self, *, interactive: bool) -> tuple[str, dict[str, str]]:
        """Shared implementation of `to_mermaid` (``interactive=False``) and `to_html`
        (``interactive=True``, adding the ``collapsible`` marker class `to_html`'s JS matches
        against to wire up click-to-collapse). Returns ``(diagram, column_names)`` - the
        ``{leaf id: column name}`` map `to_html` embeds separately for its hover-highlight."""
        top_level, edges = self.to_elements()

        body_parts = [self._render_top_level(element) for element in top_level]
        body_parts += [self._env.get_template("edge.j2").render(edge=edge) for edge in edges]

        class_lines, click_lines, column_names = self._style_and_click_lines(top_level, interactive=interactive)

        diagram = self._env.get_template("document.j2").render(
            class_defs=self._env.get_template("class_defs.j2").render(),
            body="\n".join(body_parts),
            class_lines="\n".join(class_lines),
            click_lines="\n".join(click_lines),
        )
        return diagram, column_names

    def to_mermaid(self) -> str:
        """
        Render this exporter's pipeline to Mermaid flowchart source, via the Jinja templates in
        `templates/mermaid/` (or `template_dir`, for any overridden ones).

        Portable, plain diagram source - unlike `to_html`, its subgraphs carry no ``collapsible``
        marker class, since matching against one depends on JS only the HTML page defines.

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
        diagram, _ = self._render(interactive=False)
        return diagram

    def export_mermaid(self, path: str | Path) -> Path:
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
        path.write_text(diagram, encoding="utf-8")
        return path

    def to_html(self, title: str | None = None) -> str:
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
            The rendered HTML document - every subgraph is clickable, toggling it between
            Mermaid's ``view: collapsed``/``view: expanded`` states client-side, and hovering a
            column highlights every other occurrence of that same column

        Raises:
            RuntimeError: See `DataProcessingPipeline._declared_interface`
        """
        diagram, column_names = self._render(interactive=True)
        return _HTML_TEMPLATE.format(
            title=escape(title or f"{self._pipeline.name} - pipeline flowchart"),
            cdn=_MERMAID_CDN,
            # JSON-encoded, not embedded as a JS template literal, so a column name/description
            # containing a quote, backslash or backtick can't break out of the JS source
            # ensure_ascii=False: the page is UTF-8, and labels/titles use real Unicode glyphs
            # (e.g. "⚙") - keep them literal in the page source rather than \uXXXX-escaped
            diagram_json=json.dumps(diagram, ensure_ascii=False),
            column_names_json=json.dumps(column_names, ensure_ascii=False),
        )

    def export_html(self, path: str | Path, title: str | None = None) -> Path:
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
        path.write_text(html, encoding="utf-8")
        return path
