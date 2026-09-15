import datetime as dt

import polars
import pytest

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import DataSpecification, MetaData
from damast.core.partitioning import ByColumn, ByExpr, ByTime


@pytest.fixture()
def timeseries_adf():
    df = polars.DataFrame(
        {
            "mmsi": [1, 1, 2, 2, 3],
            "timestamp": [
                dt.datetime(2026, 1, 1, 3),
                dt.datetime(2026, 1, 1, 5),
                dt.datetime(2026, 1, 2, 1),
                dt.datetime(2026, 1, 2, 2),
                dt.datetime(2026, 1, 3, 10),
            ],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    columns = [
        DataSpecification(name="mmsi"),
        DataSpecification(name="timestamp"),
        DataSpecification(name="x"),
    ]
    return AnnotatedDataFrame(df, MetaData(columns=columns))


def test_by_column_writes_one_file_per_distinct_value(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(tmp_path, ByColumn("mmsi"))

    assert sorted(p.name for p in written) == [
        "mmsi_1.parquet",
        "mmsi_2.parquet",
        "mmsi_3.parquet",
    ]
    for p in written:
        assert p.exists()
        assert p.with_suffix(".spec.yaml").exists()


def test_by_time_daily_buckets_rows_by_calendar_day(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(
        tmp_path, ByTime("timestamp", every="1d")
    )

    assert sorted(p.name for p in written) == [
        "2026-01-01.parquet",
        "2026-01-02.parquet",
        "2026-01-03.parquet",
    ]


def test_by_time_hourly_buckets_rows_by_hour(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(
        tmp_path, ByTime("timestamp", every="1h")
    )

    assert sorted(p.name for p in written) == [
        "2026-01-01T03.parquet",
        "2026-01-01T05.parquet",
        "2026-01-02T01.parquet",
        "2026-01-02T02.parquet",
        "2026-01-03T10.parquet",
    ]


def test_by_time_prefix_is_prepended_to_the_default_filename(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(
        tmp_path, ByTime("timestamp", every="1d", prefix="ais")
    )

    assert sorted(p.name for p in written) == [
        "ais_2026-01-01.parquet",
        "ais_2026-01-02.parquet",
        "ais_2026-01-03.parquet",
    ]


def test_by_time_format_overrides_the_default_filename(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(
        tmp_path, ByTime("timestamp", every="1d", format="day-%j-%Y")
    )

    assert sorted(p.name for p in written) == [
        "day-001-2026.parquet",
        "day-002-2026.parquet",
        "day-003-2026.parquet",
    ]


def test_by_time_format_with_prefix_combines_both(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(
        tmp_path, ByTime("timestamp", every="1d", format="%j", prefix="ais")
    )

    assert sorted(p.name for p in written) == [
        "ais_001.parquet",
        "ais_002.parquet",
        "ais_003.parquet",
    ]


def test_by_time_format_with_nested_path_creates_parent_directories(
    timeseries_adf, tmp_path
):
    written = timeseries_adf.export_partitioned(
        tmp_path, ByTime("timestamp", every="1d", format="%Y/%m/%d")
    )

    # .as_posix(), not str(): a relative path's separator is OS-native ("\\" on Windows),
    # but the "%Y/%m/%d" format always produces forward slashes.
    assert sorted(p.relative_to(tmp_path).as_posix() for p in written) == [
        "2026/01/01.parquet",
        "2026/01/02.parquet",
        "2026/01/03.parquet",
    ]
    for p in written:
        assert p.exists()


def test_by_expr_allows_a_custom_key_and_filename(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(
        tmp_path,
        ByExpr(polars.col("mmsi") % 2, filename_fn=lambda key: f"parity_{key}"),
    )

    assert sorted(p.name for p in written) == ["parity_0.parquet", "parity_1.parquet"]


def test_export_partitioned_round_trips_through_from_files(timeseries_adf, tmp_path):
    timeseries_adf.export_partitioned(tmp_path, ByTime("timestamp", every="1d"))

    loaded = AnnotatedDataFrame.from_files(
        sorted(str(p) for p in tmp_path.glob("*.parquet"))
    )

    assert loaded.dataframe.collected().height == 5
    assert sorted(loaded.dataframe.collected()["mmsi"].to_list()) == [1, 1, 2, 2, 3]


def test_export_partitioned_without_spec_skips_sidecar_yaml(timeseries_adf, tmp_path):
    written = timeseries_adf.export_partitioned(
        tmp_path, ByColumn("mmsi"), save_spec=False
    )

    for p in written:
        assert p.exists()
        assert not p.with_suffix(".spec.yaml").exists()


def test_export_partitioned_creates_missing_directory(timeseries_adf, tmp_path):
    target = tmp_path / "nested" / "out"
    written = timeseries_adf.export_partitioned(target, ByColumn("mmsi"))

    assert target.is_dir()
    assert all(p.parent == target for p in written)
