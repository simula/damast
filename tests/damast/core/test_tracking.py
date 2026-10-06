import polars
import pytest

import damast.core
from damast.core.constants import DAMAST_DEFAULT_DATASOURCE
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import DataSpecification, MetaData
from damast.core.tracking import KeyTracker, RowCountTracker, TimingTracker


def adf_of(**columns) -> AnnotatedDataFrame:
    return AnnotatedDataFrame(
        polars.LazyFrame(columns),
        metadata=MetaData([DataSpecification(name=name) for name in columns]),
        validation_mode=damast.core.ValidationMode.IGNORE,
    )


class FakeStep:
    """Stands in for a PipelineElement - a tracker only ever reads its uuid."""

    def __init__(self, uuid: str = "step-1"):
        self.uuid = uuid


def test_timing_tracker_reports_a_duration():
    step = FakeStep()
    tracker = TimingTracker()

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(x=[1])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(x=[1])})

    assert stats["end_time"] >= stats["start_time"]
    assert stats["processing_time_in_s"] >= 0.0
    assert stats["processing_time_in_s"] == pytest.approx(
        (stats["end_time"] - stats["start_time"]).total_seconds())


def test_timing_tracker_without_a_start_still_reports_the_end():
    """A step whose start was never seen must not make a tracker raise."""
    stats = TimingTracker().on_step_end(FakeStep(), {DAMAST_DEFAULT_DATASOURCE: adf_of(x=[1])})

    assert "end_time" in stats
    assert "processing_time_in_s" not in stats


def test_row_count_tracker_reports_removed_rows():
    step = FakeStep()
    tracker = RowCountTracker()

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(x=[1, 2, 3, 4])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(x=[1, 2])})

    assert stats["input_dataframe_length"] == {DAMAST_DEFAULT_DATASOURCE: 4}
    assert stats["output_dataframe_length"] == 2
    assert stats["rows_removed"] == 2


def test_row_count_tracker_totals_the_inputs_of_a_join():
    """A join consumes several frames, so 'removed' is against their total."""
    step = FakeStep()
    tracker = RowCountTracker()

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(x=[1, 2, 3]),
                                 "other": adf_of(x=[4, 5])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(x=[1, 2, 3, 4, 5])})

    assert stats["input_dataframe_length"] == {DAMAST_DEFAULT_DATASOURCE: 3, "other": 2}
    assert stats["output_dataframe_length"] == 5
    assert stats["rows_removed"] == 0


def test_key_tracker_counts_distinct_entities():
    """1000000 appears twice but is one entity, so rows and keys differ."""
    step = FakeStep()
    tracker = KeyTracker(["mmsi"])

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, 2, 3, 1000000, 1000000])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1000000, 1000000])})

    assert stats["keys"]["mmsi"] == {
        "unique_in": 4,
        "unique_out": 1,
        "unique_removed": 3,
        "removed_sample": [1, 2, 3],
        "removed_truncated": False,
    }


def test_key_tracker_bounds_the_removed_sample():
    step = FakeStep()
    tracker = KeyTracker(["mmsi"], max_removed_keys=2)

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, 2, 3, 4, 9])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[9])})

    assert stats["keys"]["mmsi"]["unique_removed"] == 4
    assert stats["keys"]["mmsi"]["removed_sample"] == [1, 2]
    assert stats["keys"]["mmsi"]["removed_truncated"] is True


def test_key_tracker_ignores_nulls():
    """A null identifies no entity, so it is neither counted nor reported as removed."""
    step = FakeStep()
    tracker = KeyTracker(["mmsi"])

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, None, 2])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, None])})

    assert stats["keys"]["mmsi"]["unique_in"] == 2
    assert stats["keys"]["mmsi"]["unique_out"] == 1
    assert stats["keys"]["mmsi"]["removed_sample"] == [2]


def test_key_tracker_skips_a_key_the_step_drops():
    """A step may drop the column - that is not 'every entity was removed'."""
    step = FakeStep()
    tracker = KeyTracker(["mmsi"])

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, 2], value=[3, 4])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(value=[3, 4])})

    assert stats == {}


def test_key_tracker_skips_a_key_that_was_never_there():
    """Tracking a column the data does not have is harmless."""
    step = FakeStep()
    tracker = KeyTracker(["mmsi"])

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(value=[1, 2])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(value=[1])})

    assert stats == {}


def test_key_tracker_unions_the_keys_of_a_join():
    step = FakeStep()
    tracker = KeyTracker(["mmsi"])

    tracker.on_step_start(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, 2]),
                                 "other": adf_of(mmsi=[2, 3])})
    stats = tracker.on_step_end(step, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, 2, 3])})

    assert stats["keys"]["mmsi"]["unique_in"] == 3
    assert stats["keys"]["mmsi"]["unique_removed"] == 0


def test_key_tracker_tracks_each_step_separately():
    """The transient 'before' state is per step, so two steps cannot bleed into each other."""
    tracker = KeyTracker(["mmsi"])
    first, second = FakeStep("a"), FakeStep("b")

    tracker.on_step_start(first, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1, 2, 3])})
    tracker.on_step_start(second, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[7, 8])})

    assert tracker.on_step_end(second, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[7])})[
        "keys"]["mmsi"]["removed_sample"] == [8]
    assert tracker.on_step_end(first, {DAMAST_DEFAULT_DATASOURCE: adf_of(mmsi=[1])})[
        "keys"]["mmsi"]["removed_sample"] == [2, 3]


def test_key_tracker_rejects_a_negative_bound():
    with pytest.raises(ValueError, match="max_removed_keys must not be negative"):
        KeyTracker(["mmsi"], max_removed_keys=-1)
