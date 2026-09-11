import pytest

from damast.core.decorators import DAMAST_DEFAULT_DATASOURCE
from damast.viz.pipeline_exporter import ColumnInfo, DataSourceFacts, PipelineExporter


def _node(pipeline, name):
    return next(n for n in pipeline.processing_graph.nodes() if n.name == name)


def test_datasource_facts_has_required_columns(chained_pipeline):
    exporter = PipelineExporter(chained_pipeline)
    node = _node(chained_pipeline, DAMAST_DEFAULT_DATASOURCE)
    facts = exporter.datasource_facts(node)

    assert isinstance(facts, DataSourceFacts)
    assert {c.name for c in facts.required_columns} == {"alpha", "beta"}
    alpha = next(c for c in facts.required_columns if c.name == "alpha")
    assert alpha == ColumnInfo(name="alpha", unit="m", description="raw reading", representation_type="<class 'int'>")


def test_step_facts_has_description_input_slots_and_output_columns(chained_pipeline):
    exporter = PipelineExporter(chained_pipeline)
    step_one = exporter.step_facts(_node(chained_pipeline, "step-one"))
    step_two = exporter.step_facts(_node(chained_pipeline, "step-two"))

    assert step_one.class_name == "_StepOne"
    assert step_one.description == "doubles alpha"
    assert list(step_one.input_slots.keys()) == ["df"]
    assert step_one.input_slots["df"] == [ColumnInfo(name="alpha", unit="m", description="raw reading", representation_type="<class 'int'>")]
    assert step_one.output_columns == [ColumnInfo(name="alpha_doubled", representation_type="<class 'int'>")]

    assert step_two.description == "adds beta"
    assert {c.name for c in step_two.input_slots["df"]} == {"alpha_doubled", "beta"}
    assert step_two.output_columns == [ColumnInfo(name="gamma", representation_type="<class 'int'>")]


def test_step_facts_gets_one_input_slot_per_side_of_a_join(join_pipeline):
    exporter = PipelineExporter(join_pipeline)
    join_node = next(
        n for n in join_pipeline.processing_graph.nodes()
        if not n.is_datasource() and type(n.transformer).__name__ == "_JoinStep"
    )

    assert set(exporter.step_facts(join_node).input_slots.keys()) == {"df", "other"}


def test_output_columns_accumulates_across_the_chain(chained_pipeline):
    columns = PipelineExporter(chained_pipeline).output_columns()
    assert {c.name for c in columns} == {"alpha", "alpha_doubled", "beta", "gamma"}


def test_supported_filetypes_is_not_implemented_on_the_base_class(chained_pipeline):
    with pytest.raises(NotImplementedError):
        PipelineExporter(chained_pipeline).supported_filetypes()
