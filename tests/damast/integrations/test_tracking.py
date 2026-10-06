import datetime
import json
import re

from astropy import units

from damast.core.annotations import Annotation
from damast.core.data_description import MinMax, NumericValueStats
from damast.core.metadata import DataSpecification, MetaData
from damast.integrations.tracking import (
    flatten_metadata,
    flatten_step_params,
    flatten_step_stats,
)


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


def test_flatten_step_stats_includes_removed_rows_and_keys():
    processing_stats = {
        "step_a": {
            "processing_time_in_s": 1.5,
            "output_dataframe_length": 42,
            "input_dataframe_length": {"df": 50},
            "rows_removed": 8,
            "keys": {"mmsi": {"unique_in": 7, "unique_out": 3, "unique_removed": 4,
                              "removed_sample": [1, 2, 3], "removed_truncated": True}},
        }
    }

    metrics = flatten_step_stats(processing_stats)

    assert metrics["step.step_a.rows_removed"] == 8.0
    assert metrics["step.step_a.keys.mmsi.unique_in"] == 7.0
    assert metrics["step.step_a.keys.mmsi.unique_out"] == 3.0
    assert metrics["step.step_a.keys.mmsi.unique_removed"] == 4.0

    # a list and a bool are not metrics - the contract is dict[str, float]
    assert not [name for name in metrics if "removed_sample" in name or "removed_truncated" in name]
    assert all(isinstance(value, float) for value in metrics.values())


def test_flatten_step_stats_empty():
    assert flatten_step_stats({}) == {}


# --- flattening is generic over whatever a tracker reports -----------------------------------

#: what MLflow accepts in a metric or parameter name
MLFLOW_SAFE_NAME = re.compile(r"^[0-9a-zA-Z_\-./ ]+$")

STATS_WITH_AN_UNKNOWN_TRACKER = {
    "step_a": {
        "rows_removed": 8,
        "my_block": {"my_count": 3, "nested": {"deeper": 1.5}},
    }
}


def test_flatten_step_stats_reports_an_unknown_trackers_measurements():
    """The point of the generic walk: a tracker this module knows nothing about is reported."""
    metrics = flatten_step_stats(STATS_WITH_AN_UNKNOWN_TRACKER)

    assert metrics["step.step_a.my_block.my_count"] == 3.0
    # the path carries every level, so a nested block cannot collide with a top-level one
    assert metrics["step.step_a.my_block.nested.deeper"] == 1.5
    assert "step.step_a.nested.deeper" not in metrics
    assert metrics["step.step_a.rows_removed"] == 8.0


def test_flatten_step_params_takes_what_is_not_a_measurement():
    stats = {
        "step_a": {
            "rows_removed": 8,
            "start_time": datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc),
            "keys": {"mmsi": {"unique_removed": 4,
                              "removed_sample": [1, 2, 3],
                              "removed_truncated": True}},
        }
    }

    metrics = flatten_step_stats(stats)
    params = flatten_step_params(stats)

    # a flag is not a quantity, even though isinstance(True, int) holds
    assert "step.step_a.keys.mmsi.removed_truncated" not in metrics
    assert params["step.step_a.keys.mmsi.removed_truncated"] is True
    assert params["step.step_a.start_time"] == stats["step_a"]["start_time"]
    # a container is JSON-encoded, as flatten_metadata does
    assert json.loads(params["step.step_a.keys.mmsi.removed_sample"]) == [1, 2, 3]
    assert metrics["step.step_a.keys.mmsi.unique_removed"] == 4.0


def test_metrics_and_params_partition_the_stats():
    """Exact complements, so nothing a tracker reports is silently dropped."""
    stats = {
        "step_a": {
            "rows_removed": 8,
            "processing_time_in_s": 1.5,
            "start_time": datetime.datetime(2026, 6, 1, tzinfo=datetime.timezone.utc),
            "input_dataframe_length": {"df": 50, "other": 10},
            "keys": {"mmsi": {"unique_in": 7, "removed_sample": [], "removed_truncated": False}},
        }
    }

    metrics = flatten_step_stats(stats)
    params = flatten_step_params(stats)

    assert not set(metrics) & set(params)
    # 8 leaves in total: 2 scalars + 1 datetime + 2 lengths + 3 key entries
    assert len(metrics) + len(params) == 8


def test_names_are_sanitised_for_mlflow():
    """A tracked column can carry characters MLflow rejects - damast writes 'latitude (deg)'."""
    stats = {"step:a": {"keys": {"latitude (deg)": {"unique_in": 3},
                                 "a:b": {"unique_in": 1}}}}

    metrics = flatten_step_stats(stats)

    assert all(MLFLOW_SAFE_NAME.match(name) for name in metrics), metrics
    # parentheses are not in MLflow's set either, so they go too - a space is fine
    assert metrics["step.step_a.keys.latitude _deg_.unique_in"] == 3.0
    assert metrics["step.step_a.keys.a_b.unique_in"] == 1.0


def test_flatten_step_params_empty():
    assert flatten_step_params({}) == {}
