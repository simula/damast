import shutil

import pydot
import pytest

from damast.viz.svg_export import SvgExporter

_NO_DOT = shutil.which("dot") is None


def test_to_pydot_has_one_node_per_step_plus_output(chained_pipeline):
    graph = SvgExporter(chained_pipeline).to_pydot()
    assert isinstance(graph, pydot.Dot)
    # df, step-one, step-two, plus the synthetic output node
    assert len(graph.get_node_list()) == 4
    assert len(graph.get_edge_list()) == 3


def test_to_pydot_labels_include_required_and_accumulated_columns(chained_pipeline):
    graph = SvgExporter(chained_pipeline).to_pydot()
    labels = {node.get_name(): node.get_attributes().get("label", "") for node in graph.get_node_list()}

    datasource_label = next(label for name, label in labels.items() if "DataSource" in label)
    assert "alpha" in datasource_label
    assert "beta" in datasource_label

    step_two_label = next(label for label in labels.values() if "step-two" in label)
    assert "adds beta" in step_two_label
    assert "alpha_doubled" in step_two_label
    assert "gamma" in step_two_label

    output_label = next(label for label in labels.values() if "pipeline output" in label)
    for column in ["alpha", "alpha_doubled", "beta", "gamma"]:
        assert column in output_label


def test_to_pydot_labels_join_edges_with_their_slot(join_pipeline):
    graph = SvgExporter(join_pipeline).to_pydot()
    slot_labels = {edge.get_attributes().get("label") for edge in graph.get_edge_list()}
    assert {"df", "other"} <= slot_labels


def test_to_pydot_does_not_require_graphviz(chained_pipeline, monkeypatch):
    # to_pydot only builds the graph structure - it must not shell out to 'dot'
    monkeypatch.setenv("PATH", "")
    SvgExporter(chained_pipeline).to_pydot()


def test_supported_filetypes(chained_pipeline):
    assert SvgExporter(chained_pipeline).supported_filetypes() == [".svg"]


@pytest.mark.skipif(_NO_DOT, reason="graphviz 'dot' executable not installed")
def test_to_svg_renders_an_svg_document(chained_pipeline):
    svg = SvgExporter(chained_pipeline).to_svg()
    assert svg.startswith("<?xml")
    assert "<svg" in svg


@pytest.mark.skipif(_NO_DOT, reason="graphviz 'dot' executable not installed")
def test_export_svg_writes_file(chained_pipeline, tmp_path):
    path = SvgExporter(chained_pipeline).export_svg(path=tmp_path / "nested" / "pipeline.svg")
    assert path.exists()
    assert "<svg" in path.read_text()


def test_to_svg_reports_missing_graphviz_clearly(chained_pipeline, monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="Graphviz"):
        SvgExporter(chained_pipeline).to_svg()
