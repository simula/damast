import datetime
import math

import polars as pl
import pytest
from astropy import units

import damast.core
from damast.core import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.metadata import DataSpecification, MetaData
from damast.domains.maritime.transformers.features import Heading


def _headings(tmp_path, rows: list[tuple[int, int, float, float]]) -> pl.DataFrame:
    """Run Heading over (mmsi, seconds, lat, lon) rows."""
    start = datetime.datetime(2026, 1, 1)
    df = pl.LazyFrame({
        "mmsi": [r[0] for r in rows],
        "timestamp": [start + datetime.timedelta(seconds=r[1]) for r in rows],
        "lat": [r[2] for r in rows],
        "lon": [r[3] for r in rows],
    })
    metadata = MetaData([
        DataSpecification(name="mmsi", representation_type=int),
        DataSpecification(name="timestamp"),
        DataSpecification(name="lat", unit=units.deg),
        DataSpecification(name="lon", unit=units.deg),
    ])
    adf = AnnotatedDataFrame(df, metadata, validation_mode=damast.core.ValidationMode.IGNORE)
    pipeline = DataProcessingPipeline(name="heading", base_dir=tmp_path).add(
        "heading", Heading(), name_mappings={"group": "mmsi", "sort": "timestamp"})
    return pipeline.transform(df=adf).collect().sort("mmsi", "timestamp")


def test_heading_is_the_bearing_to_the_next_position(tmp_path):
    # east along the equator, then north, then (nearly) east again
    result = _headings(tmp_path, [
        (1, 0, 0.0, 0.0),
        (1, 60, 0.0, 1.0),
        (1, 120, 1.0, 1.0),
        (1, 180, 1.0, 2.0),
    ])

    # the first position's heading has no predecessor to change from, the last one has no next
    # position to head to - both are dropped
    seconds = (result["timestamp"] - datetime.datetime(2026, 1, 1)).dt.total_seconds()
    assert seconds.to_list() == [60, 120]
    assert result["heading"].to_list() == pytest.approx([0.0, math.pi / 2], abs=1e-3)
    assert result["delta_heading"].to_list() == pytest.approx([-math.pi / 2, math.pi / 2], abs=1e-3)
    assert result["angular_velocity"].to_list() == pytest.approx(
        [-math.pi / 2 / 60, math.pi / 2 / 60], abs=1e-4)


def test_heading_changes_are_never_taken_across_objects(tmp_path):
    result = _headings(tmp_path, [
        (1, 0, 0.0, 0.0),
        (1, 60, 0.0, 1.0),
        (1, 120, 1.0, 1.0),
        # a second object: its first heading must not be compared with the first object's last
        (2, 0, 10.0, 10.0),
        (2, 60, 10.0, 11.0),
        (2, 120, 11.0, 11.0),
    ])

    assert result["mmsi"].to_list() == [1, 2]
    # both: due north after heading east - the same change, independent of the other object
    assert result["delta_heading"].to_list() == pytest.approx([-math.pi / 2, -1.5708], abs=2e-2)


def test_a_turn_across_south_is_a_small_heading_change(tmp_path):
    # heading south, first slightly east of south (bearing ~ +179.4 deg), then slightly west of
    # it (~ -179.4 deg): a turn of ~1.2 degrees, not of ~358
    result = _headings(tmp_path, [
        (1, 0, 0.0, 0.0),
        (1, 60, -1.0, 0.01),
        (1, 120, -2.0, 0.0),
        (1, 180, -3.0, 0.0),
    ])

    assert all(abs(h) > math.radians(179) for h in result["heading"])
    assert result["delta_heading"].abs().max() < math.radians(2)
    assert result["angular_velocity"].abs().max() < math.radians(2) / 60


def test_a_turn_takes_from_the_middle_of_one_step_to_the_middle_of_the_next(tmp_path):
    # 2 s east, then 60 s north: the two bearings are the course at 1 s and at 32 s, so the
    # quarter turn takes 31 s - not the 2 s of the incoming step
    result = _headings(tmp_path, [
        (1, 0, 0.0, 0.0),
        (1, 2, 0.0, 0.001),
        (1, 62, 0.01, 0.001),
        (1, 122, 0.02, 0.001),
    ])

    assert result["delta_heading"].to_list() == pytest.approx([-math.pi / 2, 0.0], abs=1e-3)
    assert result["angular_velocity"].to_list() == pytest.approx([-math.pi / 2 / 31, 0.0], abs=1e-4)


def test_a_repeated_position_keeps_the_course(tmp_path):
    # heading east; the position at 120 s is reported again at 180 s - that step has no bearing
    # (not due north), so the course stays east throughout
    result = _headings(tmp_path, [
        (1, 0, 0.0, 0.0),
        (1, 60, 0.0, 1.0),
        (1, 120, 0.0, 2.0),
        (1, 180, 0.0, 2.0),
        (1, 240, 0.0, 3.0),
        (1, 300, 0.0, 4.0),
    ])

    assert result["heading"].to_list() == pytest.approx([math.pi / 2] * 4, abs=1e-3)
    assert result["delta_heading"].to_list() == pytest.approx([0.0] * 4, abs=1e-6)
    assert result["angular_velocity"].to_list() == pytest.approx([0.0] * 4, abs=1e-6)


def test_an_object_before_its_first_move_takes_its_first_course(tmp_path):
    result = _headings(tmp_path, [
        (1, 0, 0.0, 0.0),
        (1, 60, 0.0, 0.0),
        (1, 120, 0.0, 0.0),
        (1, 180, 1.0, 0.0),
        (1, 240, 2.0, 0.0),
    ])

    assert result["heading"].to_list() == pytest.approx([0.0] * 3, abs=1e-6)
    assert result["delta_heading"].to_list() == pytest.approx([0.0] * 3, abs=1e-6)


def test_an_object_that_never_moves_has_no_heading_and_no_change(tmp_path):
    result = _headings(tmp_path, [
        (1, 0, 5.0, 5.0),
        (1, 60, 5.0, 5.0),
        (1, 120, 5.0, 5.0),
        (1, 180, 5.0, 5.0),
    ])

    # kept like any other object - only its first and last positions are dropped
    assert result.height == 2
    assert result["heading"].is_null().all()
    assert result["delta_heading"].to_list() == [0.0, 0.0]
    assert result["angular_velocity"].to_list() == [0.0, 0.0]


@pytest.mark.parametrize("angle, expected", [
    (0.0, 0.0),
    (math.radians(358), math.radians(-2)),
    (math.radians(-358), math.radians(2)),
    (math.radians(90), math.radians(90)),
    (3 * math.pi, -math.pi),
])
def test_wrapped_angle_lies_within_a_half_turn(angle, expected):
    from damast.domains.maritime.transformers.features import wrapped_angle

    value = pl.select(wrapped_angle(pl.lit(angle))).item()

    assert value == pytest.approx(expected)
    assert -math.pi <= value < math.pi
