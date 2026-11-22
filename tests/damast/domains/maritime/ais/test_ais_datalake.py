import datetime as dt
import logging
from pathlib import Path

import polars as pl
import pytest

from damast.domains.maritime.ais import AISDataLake

MAPPING = """
aisdk:
  timestamp_format: "%d/%m/%Y %H:%M:%S"
  columns:
    mmsi: MMSI
    timestamp: '# Timestamp'
    latitude: Latitude
    longitude: Longitude
barentswatch:
  scale:
    draught: 0.1
  columns:
    mmsi: mmsi
    timestamp: msgtime
    latitude: latitude
    longitude: longitude
    draught: draught
"""


def _write(base: Path, source: str, day: dt.date, frame: pl.DataFrame) -> Path:
    folder = base / f"source={source}" / f"date={day.isoformat()}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source}-{day.isoformat()}.parquet"
    frame.write_parquet(path)
    return path


@pytest.fixture
def lake_dir(tmp_path) -> Path:
    base = tmp_path / "lake"
    for d in (1, 2, 3):
        _write(
            base,
            "aisdk",
            dt.date(2026, 6, d),
            pl.DataFrame(
                {
                    "MMSI": [111, 111, 222],
                    "# Timestamp": [f"0{d}/06/2026 10:00:00", f"0{d}/06/2026 12:00:00", f"0{d}/06/2026 11:00:00"],
                    "Latitude": [55.0, 55.1, 40.0],
                    "Longitude": [10.0, 10.1, -70.0],
                }
            ),
        )
    for d in (2, 3):
        _write(
            base,
            "barentswatch",
            dt.date(2026, 6, d),
            pl.DataFrame(
                {
                    "mmsi": [111],
                    "msgtime": [f"2026-06-0{d}T13:00:00.000+02:00"],
                    "latitude": [56.0],
                    "longitude": [11.0],
                    "draught": [75],
                }
            ),
        )
    (base / "source=barentswatch" / "notes.txt").write_text("not a day")
    (base / "source-mapping.yaml").write_text(MAPPING)
    return base


def test_sources_and_time_intervals(lake_dir):
    lake = AISDataLake(lake_dir)
    assert lake.sources == {"aisdk", "barentswatch"}
    assert lake.time_intervals["aisdk"].from_date == dt.date(2026, 6, 1)
    assert lake.time_intervals["aisdk"].to_date == dt.date(2026, 6, 3)
    assert lake.time_intervals["barentswatch"].from_date == dt.date(2026, 6, 2)


def test_matching_files_by_source_and_days(lake_dir):
    lake = AISDataLake(lake_dir)
    found = list(lake.matching_files(dt.datetime(2026, 6, 2), dt.datetime(2026, 6, 3), sources=["barentswatch"]))
    assert [(s, d) for s, d, _ in found] == [
        ("barentswatch", dt.date(2026, 6, 2)),
        ("barentswatch", dt.date(2026, 6, 3)),
    ]
    assert len(list(lake.matching_files())) == 5
    assert list(lake.matching_files(sources=["unknown"])) == []
    assert [d for _, d, _ in lake.matching_files(to_time=dt.date(2026, 6, 1))] == [dt.date(2026, 6, 1)]


def test_an_empty_lake_has_no_sources(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        lake = AISDataLake(tmp_path)
        assert lake.sources == set()
    assert list(lake.matching_files()) == []
    assert lake.get_track(111).is_empty()
    assert "no mapping file" in caplog.text


def test_a_source_without_days_has_no_interval(lake_dir):
    (lake_dir / "source=empty").mkdir()
    lake = AISDataLake(lake_dir)
    assert "empty" in lake.sources
    assert "empty" not in lake.time_intervals


def test_missing_directory():
    with pytest.raises(FileNotFoundError, match="no such directory"):
        AISDataLake("/does/not/exist")


def test_mapping_rejects_unknown_canonical_columns(lake_dir):
    (lake_dir / "source-mapping.yaml").write_text("aisdk:\n  columns:\n    mmsi: MMSI\n    speed: SOG\n")
    with pytest.raises(ValueError, match="(?s)aisdk.*speed"):
        AISDataLake(lake_dir).refresh()


def test_source_mapping(lake_dir):
    lake = AISDataLake(lake_dir)
    assert lake.source_mapping("aisdk").columns["timestamp"] == "# Timestamp"
    assert lake.get_source_mapping("aisdk")["MMSI"] == "mmsi"
    assert "sog" in lake.source_mapping("aisdk").missing
    assert lake.source_mapping("unknown") is None
    with pytest.raises(ValueError):
        lake.get_source_mapping("unknown")


def test_get_track_parses_timestamps_scales_and_sorts(lake_dir):
    lake = AISDataLake(lake_dir)
    track = lake.get_track(111, columns=["mmsi", "timestamp", "latitude", "longitude", "draught"])
    assert track.schema["timestamp"] == pl.Datetime("us")
    assert track["timestamp"].is_sorted()
    assert track.height == 6 + 2
    bw = track.filter(pl.col("source") == "barentswatch")
    # 13:00 +02:00 is 11:00 UTC.
    assert bw["timestamp"].to_list()[0] == dt.datetime(2026, 6, 2, 11, 0)
    assert bw["draught"].to_list() == pytest.approx([7.5, 7.5])
    # aisdk has no draught column.
    assert track.filter(pl.col("source") == "aisdk")["draught"].null_count() == 6


def test_get_track_within_a_time_window(lake_dir):
    lake = AISDataLake(lake_dir)
    track = lake.get_track(111, dt.datetime(2026, 6, 2, 11), dt.datetime(2026, 6, 3, 10))
    assert track["timestamp"].to_list() == [
        dt.datetime(2026, 6, 2, 11),
        dt.datetime(2026, 6, 2, 12),
        dt.datetime(2026, 6, 3, 10),
    ]
    assert track["source"].to_list() == ["barentswatch", "aisdk", "aisdk"]


def test_by_mmsi_quickview(lake_dir):
    coverage = AISDataLake(lake_dir).by_mmsi_quickview()
    assert set(coverage) == {111, 222}
    assert coverage[111]["aisdk"].dates == [dt.date(2026, 6, d) for d in (1, 2, 3)]
    assert len(coverage[111]["aisdk"].files) == 3
    assert coverage[111]["barentswatch"].dates == [dt.date(2026, 6, 2), dt.date(2026, 6, 3)]
    assert set(coverage[222]) == {"aisdk"}
