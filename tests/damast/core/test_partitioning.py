import datetime as dt

import polars
import pytest

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import DataSpecification, MetaData, ValidationMode
from damast.core.partitioning import ByColumn, ByExpr, ByTime, SaveAs


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


def test_export_partitioned_creates_missing_directory(timeseries_adf, tmp_path):
    target = tmp_path / "nested" / "out"
    written = timeseries_adf.export_partitioned(target, ByColumn("mmsi"))

    assert target.is_dir()
    assert all(p.parent == target for p in written)


def test_save_as_parse_plain_path_is_unaffected():
    from pathlib import Path

    save_as = SaveAs.parse("out/result.parquet")

    assert save_as.path == Path("out/result.parquet")
    assert save_as.strategy is None


def test_save_as_parse_windows_drive_letter_is_not_mistaken_for_a_strategy():
    # "C:" must not be parsed as a (nonexistent) "C" strategy prefix.
    save_as = SaveAs.parse(r"C:\Users\me\out.parquet")

    assert save_as.strategy is None
    assert str(save_as.path) == r"C:\Users\me\out.parquet"


def test_save_as_export_writes_a_single_file_for_a_plain_path(timeseries_adf, tmp_path):
    output_file = tmp_path / "result.parquet"

    written = SaveAs.parse(str(output_file)).export(timeseries_adf)

    assert written == output_file
    assert output_file.exists()
    # SaveAs.export goes through AnnotatedDataFrame.save (like export_partitioned's per-
    # partition files), not the lower-level export - so a plain path also gets its sidecar.
    loaded = AnnotatedDataFrame.from_files([str(output_file)])
    assert loaded.dataframe.collected().height == 5


def test_save_as_export_supports_hdf5_for_a_plain_path(timeseries_adf, tmp_path):
    output_file = tmp_path / "result.hdf5"

    written = SaveAs.parse(str(output_file)).export(timeseries_adf)

    assert written == output_file
    assert output_file.exists()


def test_save_as_export_time_strategy(timeseries_adf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    written = SaveAs.parse("time:timestamp+daily:out/AIS_%Y_%m_%d").export(
        timeseries_adf
    )

    # .as_posix(), not str(): a relative path's separator is OS-native ("\\" on Windows),
    # but the template's own "/" is always a forward slash.
    assert sorted(p.as_posix() for p in written) == [
        "out/AIS_2026_01_01.parquet",
        "out/AIS_2026_01_02.parquet",
        "out/AIS_2026_01_03.parquet",
    ]


def test_save_as_export_column_strategy(timeseries_adf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    written = SaveAs.parse("column:mmsi:out/vessel_{mmsi}").export(timeseries_adf)

    assert sorted(p.as_posix() for p in written) == [
        "out/vessel_1.parquet",
        "out/vessel_2.parquet",
        "out/vessel_3.parquet",
    ]


def test_save_as_export_time_plus_column_strategy(
    timeseries_adf, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    written = SaveAs.parse(
        "time+column:timestamp+daily+mmsi:out/{mmsi}/AIS_%Y_%m_%d"
    ).export(timeseries_adf)

    assert sorted(p.as_posix() for p in written) == [
        "out/1/AIS_2026_01_01.parquet",
        "out/2/AIS_2026_01_02.parquet",
        "out/3/AIS_2026_01_03.parquet",
    ]


@pytest.mark.parametrize(
    "value",
    [
        "time:timestamp:out/AIS_%Y",  # missing "+<interval>"
        "time+column:timestamp+daily:out/AIS",  # missing "+<column>"
        "column::out/x",  # empty <column>
        "time:+daily:out/x",  # empty <timestamp_column>
        "time:timestamp+:out/x",  # empty <interval>
        "time+column:+daily+mmsi:out/x",  # empty <timestamp_column>
        "time+column:timestamp++mmsi:out/x",  # empty <interval>
        "time+column:timestamp+daily+:out/x",  # empty <column>
    ],
)
def test_save_as_parse_rejects_malformed_spec(value):
    with pytest.raises(ValueError, match="SaveAs.parse"):
        SaveAs.parse(value)


def test_expected_paths_plain_path_is_unaffected():
    from pathlib import Path

    assert SaveAs.expected_paths(
        "out/result.parquet", start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 1, 3)
    ) == [Path("out/result.parquet")]


def test_expected_paths_time_strategy_matches_what_export_writes(timeseries_adf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    value = "time:timestamp+daily:out/AIS_%Y_%m_%d"

    written = SaveAs.parse(value).export(timeseries_adf)
    predicted = SaveAs.expected_paths(
        value, start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 1, 3, 23, 59, 59)
    )

    assert sorted(str(p) for p in predicted) == sorted(str(p) for p in written)


def test_expected_paths_time_strategy_only_covers_the_requested_range(timeseries_adf, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    value = "time:timestamp+daily:out/AIS_%Y_%m_%d"

    predicted = SaveAs.expected_paths(
        value, start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 1, 2)
    )

    assert [p.as_posix() for p in predicted] == ["out/AIS_2026_01_01.parquet", "out/AIS_2026_01_02.parquet"]


def test_expected_paths_column_strategy_wildcards_the_column(timeseries_adf):
    from pathlib import Path

    predicted = SaveAs.expected_paths(
        "column:mmsi:out/vessel_{mmsi}", start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 1, 3)
    )

    assert predicted == [Path("out/vessel_*.parquet")]


def test_expected_paths_time_plus_column_strategy_wildcards_the_column_per_bucket():
    from pathlib import Path

    predicted = SaveAs.expected_paths(
        "time+column:timestamp+daily+mmsi:out/{mmsi}/AIS_%Y_%m_%d",
        start=dt.datetime(2026, 1, 1),
        end=dt.datetime(2026, 1, 2),
    )

    assert predicted == [
        Path("out/*/AIS_2026_01_01.parquet"),
        Path("out/*/AIS_2026_01_02.parquet"),
    ]


@pytest.mark.parametrize(
    "value",
    [
        "time:timestamp:out/AIS_%Y",  # missing "+<interval>"
        "time+column:timestamp+daily:out/AIS",  # missing "+<column>"
        "column::out/x",  # empty <column>
        "time:+daily:out/x",  # empty <timestamp_column>
        "time:timestamp+:out/x",  # empty <interval>
    ],
)
def test_expected_paths_rejects_malformed_spec(value):
    with pytest.raises(ValueError, match="SaveAs.expected_paths"):
        SaveAs.expected_paths(value, start=dt.datetime(2026, 1, 1), end=dt.datetime(2026, 1, 2))


def _zoned_adf(time_zone: str | None):
    """The `timeseries_adf` frame with its timestamps as instants in `time_zone` (None: naive)."""
    df = polars.DataFrame(
        {
            "mmsi": [1, 2],
            # 2026-01-01 20:00 UTC is already 2026-01-02 in Tokyo (UTC+9)
            "timestamp": [dt.datetime(2026, 1, 1, 20), dt.datetime(2026, 1, 1, 23)],
            "x": [10.0, 20.0],
        }
    )
    if time_zone is not None:
        df = df.with_columns(
            polars.col("timestamp").dt.replace_time_zone("UTC").dt.convert_time_zone(time_zone)
        )
    columns = [DataSpecification(name=name) for name in ("mmsi", "timestamp", "x")]
    return AnnotatedDataFrame(df, MetaData(columns=columns))


def test_by_time_buckets_follow_the_column_time_zone(tmp_path):
    """A non-UTC column partitions by *its* days - the instants above are a Tokyo 02 Jan."""
    written = _zoned_adf("Asia/Tokyo").export_partitioned(tmp_path, ByTime("timestamp", every="1d"))

    assert [p.name for p in written] == ["2026-01-02.parquet"]
    assert [p.name for p in _zoned_adf("UTC").export_partitioned(tmp_path, ByTime("timestamp", every="1d"))] == [
        "2026-01-01.parquet"
    ]


def test_export_partitioned_warns_only_for_a_non_utc_time_bucket(tmp_path, caplog):
    for time_zone, expected in [("Asia/Tokyo", True), ("UTC", False), (None, False)]:
        caplog.clear()
        with caplog.at_level("WARNING", logger="damast.core.partitioning"):
            _zoned_adf(time_zone).export_partitioned(tmp_path, ByTime("timestamp", every="1d"))
        warned = any("not UTC days" in record.message for record in caplog.records)
        assert warned is expected, f"{time_zone=} should {'' if expected else 'not '}warn"


def test_export_partitioned_warns_for_a_non_utc_bucket_inside_a_struct_key(tmp_path, caplog):
    """'time+column:' keys are a struct - the timestamp field still has to be checked."""
    save_as = SaveAs.parse("time+column:timestamp+daily+mmsi:AIS_%Y_%m_%d")

    with caplog.at_level("WARNING", logger="damast.core.partitioning"):
        _zoned_adf("Asia/Tokyo").export_partitioned(tmp_path, save_as.strategy)

    assert any("not UTC days" in record.message for record in caplog.records)


def test_expected_paths_names_do_not_match_a_non_utc_archive(tmp_path):
    """The mismatch the warning is about: UTC bounds predict a name the archive doesn't have."""
    written = _zoned_adf("Asia/Tokyo").export_partitioned(tmp_path, ByTime("timestamp", every="1d"))
    predicted = SaveAs.expected_paths(
        "time:timestamp+daily:%Y-%m-%d",
        start=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2026, 1, 1, 23, 59, tzinfo=dt.UTC),
    )

    assert [p.name for p in predicted] == ["2026-01-01.parquet"]
    assert [p.name for p in written] == ["2026-01-02.parquet"]


def test_export_partitioned_keeps_declared_metadata_in_every_partition(tmp_path):
    """Per-partition metadata is re-inferred, but declarations must survive the split -
    ranges narrow to the partition, annotations and `is_optional` do not."""
    from damast.core.annotations import Annotation

    df = polars.DataFrame(
        {
            "timestamp": [dt.datetime(2026, 1, 1, 3), dt.datetime(2026, 1, 2, 4)],
            "x": [10.0, 20.0],
        }
    )
    metadata = MetaData(
        columns=[DataSpecification(name="timestamp"),
                 DataSpecification(name="x", description="the x", is_optional=True)],
        annotations=[Annotation(name="origin", value="a-test")],
    )
    adf = AnnotatedDataFrame(df, metadata, validation_mode=ValidationMode.IGNORE)

    written = adf.export_partitioned(tmp_path, ByTime("timestamp", every="1d"))

    assert len(written) == 2
    for path in written:
        part = AnnotatedDataFrame.from_files([str(path)], metadata_required=False)
        assert {name: a.value for name, a in part.metadata.annotations.items()
                if name == "origin"} == {"origin": "a-test"}
        assert part.metadata["x"].is_optional is True
        assert part.metadata["x"].description == "the x"

    # ... while the value range is each partition's own, not the whole frame's
    ranges = {p.name: AnnotatedDataFrame.from_files([str(p)], metadata_required=False)
              .metadata["x"].value_range for p in written}
    assert [(r.min, r.max) for r in ranges.values()] == [(10.0, 10.0), (20.0, 20.0)]


def test_export_partitioned_suffix_is_configurable(timeseries_adf, tmp_path):
    """`suffix=""` hands full filenames to the strategy - e.g. names from an external
    naming scheme; a suffix without a leading dot is normalized."""
    named = ByExpr(polars.col("mmsi"), filename_fn=lambda key: f"vessel-{key}.parquet")

    written = timeseries_adf.export_partitioned(tmp_path / "own", named, suffix="")
    assert sorted(p.name for p in written) == ["vessel-1.parquet", "vessel-2.parquet", "vessel-3.parquet"]
    assert all(p.exists() for p in written)

    written = timeseries_adf.export_partitioned(tmp_path / "dotless", ByColumn("mmsi"), suffix="dat")
    assert sorted(p.name for p in written) == ["mmsi_1.dat", "mmsi_2.dat", "mmsi_3.dat"]

    # the default is unchanged, and the files stay readable whatever they are called
    written = timeseries_adf.export_partitioned(tmp_path / "default", ByColumn("mmsi"))
    assert sorted(p.name for p in written) == ["mmsi_1.parquet", "mmsi_2.parquet", "mmsi_3.parquet"]
