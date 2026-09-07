import jinja2
import pytest

from damast.viz.mermaid_export import MermaidExporter


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

    assert datasource.title == "Input (DataSource)"
    assert datasource.style_class == "dataSourceNodeStyle"
    labels = {leaf.label for leaf in datasource.columns}
    assert labels == {"alpha [unit: m]", "beta"}
    assert all(leaf.shape == "lean-r" and leaf.style_class == "inputsStyle" for leaf in datasource.columns)


def test_processing_element_nests_input_transform_output(chained_pipeline):
    top_level, _ = MermaidExporter(chained_pipeline).to_elements()
    step_one = _find(top_level, lambda e: isinstance(e, MermaidExporter.ProcessingElement)
                      and e.class_name == "_StepOne")

    assert step_one.tooltip == "doubles alpha"

    (input_block,) = step_one.input_blocks
    assert input_block.title == "Input (min required)"
    assert [leaf.label for leaf in input_block.columns] == ["alpha [unit: m]"]
    assert input_block.columns[0].tooltip == "raw reading"

    assert step_one.transform.label == "transform"
    assert [leaf.label for leaf in step_one.output_block.columns] == ["alpha_doubled"]


def test_pipeline_output_block_accumulates_across_the_chain(chained_pipeline):
    top_level, edges = MermaidExporter(chained_pipeline).to_elements()
    output = _find(top_level, lambda e: e.id == "PIPELINE_OUTPUT")

    assert isinstance(output, MermaidExporter.ColumnBlock)
    assert output.style_class == "outputsBlockStyle"
    labels = {leaf.label for leaf in output.columns}
    assert labels == {"alpha [unit: m]", "alpha_doubled", "beta", "gamma"}
    assert all(leaf.shape == "lean-l" and leaf.style_class == "outputsStyle" for leaf in output.columns)

    step_two = _find(top_level, lambda e: isinstance(e, MermaidExporter.ProcessingElement)
                      and e.class_name == "_StepTwo")
    assert any(e.source == step_two.id and e.target == output.id for e in edges)


def test_join_pipeline_gets_one_input_block_per_slot(join_pipeline):
    top_level, edges = MermaidExporter(join_pipeline).to_elements()
    join_step = _find(top_level, lambda e: isinstance(e, MermaidExporter.ProcessingElement)
                       and e.class_name == "_JoinStep")

    assert {block.title for block in join_step.input_blocks} == {
        "Input (df) (min required)", "Input (other) (min required)",
    }

    slot_labels = {e.label for e in edges if e.label}
    assert slot_labels == {"df", "other"}


def test_supported_filetypes(chained_pipeline):
    assert MermaidExporter(chained_pipeline).supported_filetypes() == [".html", ".mmd"]


def test_to_mermaid_contains_class_defs_and_assignments(chained_pipeline):
    diagram = MermaidExporter(chained_pipeline).to_mermaid()
    assert diagram.startswith("flowchart TB\n")
    assert "classDef dataSourceNodeStyle" in diagram
    assert "classDef processingElementStyle" in diagram
    assert 'shape: lean-r, label: "alpha [unit: m]"' in diagram
    assert 'shape: lean-l, label: "gamma"' in diagram
    assert '("transform")' in diagram
    assert "class " in diagram and " inputsStyle" in diagram
    assert 'click' in diagram and 'raw reading' in diagram
    assert 'click' in diagram and 'doubles alpha' in diagram


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
    text = path.read_text()
    assert text.startswith("flowchart TB\n")
    assert "<!doctype html>" not in text


def test_to_html_embeds_diagram_and_mermaid_cdn(chained_pipeline):
    html = MermaidExporter(chained_pipeline).to_html()
    assert "<script src=\"https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js\"></script>" in html
    assert "flowchart TB" in html
    assert "mermaid.initialize" in html
    assert "chained - pipeline flowchart" in html


def test_export_html_writes_file(chained_pipeline, tmp_path):
    path = MermaidExporter(chained_pipeline).export_html(path=tmp_path / "nested" / "pipeline.html")
    assert path.exists()
    assert "flowchart TB" in path.read_text()


def test_template_dir_overrides_only_the_files_it_provides(chained_pipeline, tmp_path):
    (tmp_path / "datasource_block.j2").write_text(
        'subgraph {{ block.id }} ["CUSTOM: {{ block.title }}"]\n'
        "    direction TB\n"
        "end\n"
        "class {{ block.id }} {{ block.style_class }}\n"
    )

    diagram = MermaidExporter(chained_pipeline, template_dir=tmp_path).to_mermaid()

    assert "CUSTOM: Input (DataSource)" in diagram
    # everything else still resolves to the shipped default
    assert "Input (min required)" in diagram
    assert "Output (guaranteed)" in diagram


def test_template_dir_typo_raises_instead_of_rendering_blank(chained_pipeline, tmp_path):
    (tmp_path / "leaf.j2").write_text('{{ node.this_attribute_does_not_exist }}\n')

    with pytest.raises(jinja2.UndefinedError):
        MermaidExporter(chained_pipeline, template_dir=tmp_path).to_mermaid()
