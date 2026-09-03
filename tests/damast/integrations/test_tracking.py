from astropy import units

from damast.core.annotations import Annotation
from damast.core.data_description import MinMax
from damast.core.data_description import NumericValueStats
from damast.core.metadata import DataSpecification, MetaData
from damast.integrations.tracking import flatten_metadata, flatten_step_stats


def test_flatten_metadata_splits_contract_from_stats():
    spec = DataSpecification(
        name="speed",
        unit=units.m / units.s,
        value_range=MinMax(0.0, 10.0),
        value_stats=NumericValueStats(
            mean=1.0, stddev=0.5, total_count=100, null_count=2
        ),
    )
    metadata = MetaData(
        columns=[spec],
        annotations=[Annotation(name="institution", value="Simula")],
    )

    params, metrics, tags = flatten_metadata(metadata)

    # unit/value_range describe the declared contract -> params
    assert params["col.speed.unit"] == spec.unit.to_string()
    assert "col.speed.value_range" in params

    # value_stats are numeric, per-run computed values -> metrics
    assert metrics["col.speed.mean"] == 1.0
    assert metrics["col.speed.stddev"] == 0.5
    assert metrics["col.speed.null_count"] == 2

    # scalar annotations -> tags
    assert tags["institution"] == "Simula"


def test_flatten_metadata_skips_non_scalar_annotations():
    metadata = MetaData(columns=[DataSpecification(name="x")], annotations=[])
    metadata.add_annotation(Annotation(name="comment", value="a plain string"))

    _, _, tags = flatten_metadata(metadata)
    assert tags["comment"] == "a plain string"


def test_flatten_step_stats():
    processing_stats = {
        "step_a": {
            "processing_time_in_s": 1.5,
            "output_dataframe_length": 42,
            "input_dataframe_length": {"df": 50},
        }
    }

    metrics = flatten_step_stats(processing_stats)

    assert metrics["step.step_a.processing_time_in_s"] == 1.5
    assert metrics["step.step_a.output_dataframe_length"] == 42.0
    assert metrics["step.step_a.input_dataframe_length.df"] == 50.0


def test_flatten_step_stats_empty():
    assert flatten_step_stats({}) == {}
