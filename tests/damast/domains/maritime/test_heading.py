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
