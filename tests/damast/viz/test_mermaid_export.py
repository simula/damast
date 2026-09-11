import re

import jinja2
import pytest

from damast.viz.mermaid_export import _MERMAID_CDN, MermaidExporter


def _find(elements, predicate):
    return next(e for e in elements if predicate(e))


def test_to_elements_has_one_cluster_per_step_plus_output(chained_pipeline):
    top_level, edges = MermaidExporter(chained_pipeline).to_elements()
    # df, step-one, step-two, plus the synthetic pipeline-output cluster
    assert len(top_level) == 4
    assert len(edges) == 3


def test_datasource_block_shows_required_columns(chained_pipeline):
    top_level, _ = MermaidExporter(chained_pipeline).to_elements()
    datasource = _find(top_level, lambda e: isinstance(e, MermaidExporter.DataSourceBlock))

    assert datasource.style_class == "dataSourceNodeStyle"
    names = {leaf.label.split("\n")[0] for leaf in datasource.columns}
    assert names == {"alpha", "beta"}
    assert all(leaf.shape == "lean-r" and leaf.style_class == "inputsStyle" for leaf in datasource.columns)
    # "alpha" carries a unit and a description (see conftest) - both show up on its label
    alpha = _find(datasource.columns, lambda c: c.label.startswith("alpha"))
    assert "unit: m" in alpha.label
    assert alpha.tooltip == "type: &lt;class 'int'&gt;<br/>---<br/>raw reading"
    # the bare, unescaped name - not the label with its unit/tooltip glyph baked in - so
    # to_html's hover-highlight can group same-named columns across the diagram by it
    assert alpha.name == "alpha"


def test_processing_element_nests_input_transform_output(chained_pipeline):
    top_level, _ = MermaidExporter(chained_pipeline).to_elements()
    step_one = _find(top_level, lambda e: isinstance(e, MermaidExporter.ProcessingElement)
                      and e.class_name == "_StepOne")

    assert step_one.tooltip == "doubles alpha"

    (input_block,) = step_one.input_blocks
    assert [leaf.label.split("\n")[0] for leaf in input_block.columns] == ["alpha"]
    assert input_block.columns[0].tooltip == "type: &lt;class 'int'&gt;<br/>---<br/>raw reading"

    assert [leaf.label for leaf in step_one.output_block.columns] == ["alpha_doubled"]


def test_pipeline_output_block_accumulates_across_the_chain(chained_pipeline):
    top_level, edges = MermaidExporter(chained_pipeline).to_elements()
    output = _find(top_level, lambda e: e.id == "PIPELINE_OUTPUT")

    assert isinstance(output, MermaidExporter.ColumnBlock)
    assert output.style_class == "pipelineOutputsBlockStyle"
    names = {leaf.label.split("\n")[0] for leaf in output.columns}
    assert names == {"alpha", "alpha_doubled", "beta", "gamma"}
    assert all(leaf.shape == "lean-l" and leaf.style_class == "outputsStyle" for leaf in output.columns)

    step_two = _find(top_level, lambda e: isinstance(e, MermaidExporter.ProcessingElement)
                      and e.class_name == "_StepTwo")
    assert any(e.source == step_two.id and e.target == output.id for e in edges)


def test_join_pipeline_gets_one_input_block_per_slot(join_pipeline):
    top_level, edges = MermaidExporter(join_pipeline).to_elements()
    join_step = _find(top_level, lambda e: isinstance(e, MermaidExporter.ProcessingElement)
                       and e.class_name == "_JoinStep")

    assert {block.title for block in join_step.input_blocks} == {"Input (df)", "Input (other)"}

    slot_labels = {e.label for e in edges if e.label}
    assert slot_labels == {"df", "other"}


def test_supported_filetypes(chained_pipeline):
    assert MermaidExporter(chained_pipeline).supported_filetypes() == [".html", ".mmd"]


def test_to_mermaid_contains_class_defs_and_assignments(chained_pipeline):
    diagram = MermaidExporter(chained_pipeline).to_mermaid()
    assert "flowchart TB" in diagram
    assert "classDef dataSourceNodeStyle" in diagram
    assert "classDef processingElementStyle" in diagram
    assert 'label: "alpha' in diagram and "unit: m" in diagram
    assert 'shape: lean-l, label: "gamma"' in diagram
    assert '"⚙ transform")' in diagram
    assert "class " in diagram and " inputsStyle" in diagram
    assert 'click' in diagram and 'raw reading' in diagram
    assert 'click' in diagram and 'doubles alpha' in diagram


def test_to_mermaid_has_no_synthetic_layout_nodes(chained_pipeline):
    # an earlier design added an invisible sibling node to every collapsible subgraph (needed
    # to keep its class alive once collapsed) - dagre's layout counted it when centering each
    # rank, throwing the whole top-level chain visibly off a shared vertical axis. Every
    # collapsible element already sits on a real edge from the pipeline's own dataflow (a
    # ColumnBlock connects to its ProcessingElement's transform; every top-level element sits on
    # the edges between steps), so that requirement is met for free - no synthetic node needed
    diagram = MermaidExporter(chained_pipeline).to_mermaid()
    assert "_anchor" not in diagram


def test_to_mermaid_has_no_collapse_bindings(chained_pipeline):
    # to_mermaid()/export_mermaid() are portable, plain diagram source - no element is ever
    # actually assigned the 'collapsible' marker class (the classDef declaration itself is
    # harmless boilerplate and stays either way), since matching against it depends on JS only
    # the HTML page defines
    diagram = MermaidExporter(chained_pipeline).to_mermaid()
    assert not re.search(r"^class \S+ collapsible$", diagram, re.MULTILINE)
    assert "view: collapsed" not in diagram


def test_to_mermaid_puts_class_defs_and_assignments_after_the_diagram_body(chained_pipeline):
    # classDef/class/click statements interleaved into a subgraph don't render correctly in
    # Mermaid - they must all trail the diagram body, classDef declarations before the class
    # assignments that use them
    diagram = MermaidExporter(chained_pipeline).to_mermaid()

    last_subgraph_end = max(i for i, line in enumerate(diagram.splitlines()) if line.strip() == "end")
    first_class_def = next(i for i, line in enumerate(diagram.splitlines()) if line.startswith("classDef "))
    first_class_assignment = next(i for i, line in enumerate(diagram.splitlines()) if line.startswith("class "))
    first_click = next(i for i, line in enumerate(diagram.splitlines()) if line.startswith("click "))

    assert last_subgraph_end < first_class_def < first_class_assignment < first_click


def test_to_mermaid_keeps_one_statement_per_line(chained_pipeline):
    # a template's own trailing newline must survive {% include %} composition, or two
    # statements end up glued onto one line and Mermaid fails to parse it
    diagram = MermaidExporter(chained_pipeline).to_mermaid()
    for line in diagram.splitlines():
        assert line.count("class ") <= 1
        assert not (" --> " in line and "class " in line)


def test_export_mermaid_writes_raw_diagram_source(chained_pipeline, tmp_path):
    path = MermaidExporter(chained_pipeline).export_mermaid(path=tmp_path / "pipeline.mmd")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "flowchart TB" in text
    assert "<!doctype html>" not in text
    assert not re.search(r"^class \S+ collapsible$", text, re.MULTILINE)


def test_to_html_embeds_diagram_and_mermaid_cdn(chained_pipeline):
    html = MermaidExporter(chained_pipeline).to_html()
    assert f'<script src="{_MERMAID_CDN}"></script>' in html
    assert "flowchart TB" in html
    assert "chained - pipeline flowchart" in html


def test_to_html_matches_tooltip_font_size_to_block_titles(chained_pipeline):
    html = MermaidExporter(chained_pipeline).to_html()
    assert re.search(r".mermaidTooltip {\n\s+font-size: [12][0-9]px !important", ''.join(html))


def test_to_html_wires_up_click_to_collapse_per_subgraph(chained_pipeline):
    # Mermaid does not fire 'click' bindings on a subgraph's own id at all, expanded or
    # collapsed (mermaid-js/mermaid#5428), so collapsing does not go through Mermaid's click
    # mechanism: the page renders the SVG itself, then finds every 'collapsible'-classed
    # element (verified against a real Mermaid render to also match the single node a subgraph
    # becomes once collapsed) and attaches a real click listener directly - see
    # MermaidExporter._style_and_click_lines and the renderDiagram/wireCollapseClicks pair below
    html = MermaidExporter(chained_pipeline).to_html()
    top_level, _ = MermaidExporter(chained_pipeline).to_elements()

    assert "startOnLoad: false" in html
    assert "securityLevel: 'loose'" in html
    assert "async function renderDiagram" in html
    assert "function wireCollapseClicks" in html
    assert "bindFunctions(container)" in html  # needed for the (unrelated) tooltip clicks
    assert 'querySelectorAll("g.collapsible")' in html
    assert "event.stopPropagation()" in html  # a nested block's click must not also toggle its parent

    datasource = _find(top_level, lambda e: isinstance(e, MermaidExporter.DataSourceBlock))
    step = _find(top_level, lambda e: isinstance(e, MermaidExporter.ProcessingElement))
    output = _find(top_level, lambda e: e.id == "PIPELINE_OUTPUT")
    for element in (datasource, step, output):
        assert f"class {element.id} collapsible" in html


def test_to_html_wires_up_hover_highlight_for_same_named_columns(chained_pipeline):
    # "alpha" is required by the datasource, consumed as _StepOne's input, and re-appears in the
    # pipeline's overall output - hovering any one of those three should highlight all of them,
    # see MermaidExporter._style_and_click_lines and wireHoverHighlight below
    html = MermaidExporter(chained_pipeline).to_html()

    assert "function wireHoverHighlight" in html
    assert "const columnNames = " in html

    top_level, _ = MermaidExporter(chained_pipeline).to_elements()
    datasource = _find(top_level, lambda e: isinstance(e, MermaidExporter.DataSourceBlock))
    alpha = _find(datasource.columns, lambda c: c.label.startswith("alpha"))
    assert f'"{alpha.id}": "alpha"' in html


def test_export_html_writes_file(chained_pipeline, tmp_path):
    path = MermaidExporter(chained_pipeline).export_html(path=tmp_path / "nested" / "pipeline.html")
    assert path.exists()
    assert "flowchart TB" in path.read_text(encoding="utf-8")


def test_template_dir_overrides_only_the_files_it_provides(chained_pipeline, tmp_path):
    (tmp_path / "datasource_block.j2").write_text(
        'subgraph {{ block.id }} ["CUSTOM: {{ block.title }}"]\n'
        "    direction TB\n"
        "end\n"
        "class {{ block.id }} {{ block.style_class }}\n"
    )

    top_level, _ = MermaidExporter(chained_pipeline).to_elements()
    datasource = _find(top_level, lambda e: isinstance(e, MermaidExporter.DataSourceBlock))
    diagram = MermaidExporter(chained_pipeline, template_dir=tmp_path).to_mermaid()

    assert f"CUSTOM: {datasource.title}" in diagram
    # everything else still resolves to the shipped default
    assert "_StepOne" in diagram
    assert "gamma" in diagram


def test_template_dir_typo_raises_instead_of_rendering_blank(chained_pipeline, tmp_path):
    (tmp_path / "column_block.j2").write_text('{{ block.this_attribute_does_not_exist }}\n')

    with pytest.raises(jinja2.UndefinedError):
        MermaidExporter(chained_pipeline, template_dir=tmp_path).to_mermaid()
