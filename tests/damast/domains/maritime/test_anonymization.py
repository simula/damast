import datetime as dt

import polars
import pytest

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.metadata import DataSpecification, MetaData
from damast.data_handling.transformers.filters import DropMissingOrNan
from damast.domains.maritime.anonymization import (
    AISColumns,
    GeneralizeSpaceTime,
    KAnonymityFilter,
    PseudonymizeVoyages,
    SegmentVoyages,
    SuppressVoyageEndpoints,
    VoyageKAnonymity,
)

KEY = "ab" * 32
T0 = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)

SEGMENT_MAPPINGS = {"timestamp": "msgtime", "speed": "speedOverGround", "status": "navigationalStatus"}
PSEUDONYMIZE_MAPPINGS = {"timestamp": "msgtime", "lat": "latitude", "lon": "longitude",
                         "speed": "speedOverGround", "course": "courseOverGround",
                         "status": "navigationalStatus", "ship_type": "shipType"}


def annotate(df: polars.DataFrame, columns: AISColumns = AISColumns()) -> AnnotatedDataFrame:
    units = {columns.lat: "deg", columns.lon: "deg", columns.course: "deg"}
    ints = {columns.mmsi, columns.status, columns.ship_type}
    return AnnotatedDataFrame(df, MetaData(columns=[
        DataSpecification(c, unit=units.get(c), representation_type=int if c in ints else None)
        for c in df.columns
    ]))


def ais(rows: list[tuple]) -> polars.DataFrame:
    """Rows of (mmsi, minutes after T0, latitude, speed, navigational status) at longitude 5."""
    return polars.DataFrame({
        "mmsi": [r[0] for r in rows],
        "msgtime": [T0 + dt.timedelta(minutes=r[1]) for r in rows],
        "latitude": [r[2] for r in rows],
        "longitude": [5.0] * len(rows),
        "speedOverGround": [r[3] for r in rows],
        "courseOverGround": [0.0] * len(rows),
        "navigationalStatus": [r[4] for r in rows],
        "shipType": [71] * len(rows),
        "name": ["SECRET"] * len(rows),
    })


def segment(df: polars.DataFrame, tmp_path) -> polars.DataFrame:
    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("segment", SegmentVoyages(), name_mappings=SEGMENT_MAPPINGS)
    return pipeline.transform(df=annotate(df)).lazyframe.collect()


def pseudonymize(df: polars.DataFrame, tmp_path, **kwargs) -> polars.DataFrame:
    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("segment", SegmentVoyages(), name_mappings=SEGMENT_MAPPINGS)
    pipeline.add("pseudonymize", PseudonymizeVoyages(**kwargs), name_mappings=PSEUDONYMIZE_MAPPINGS)
    return pipeline.transform(df=annotate(df)).lazyframe.collect()


@pytest.fixture
def key_env(monkeypatch):
    monkeypatch.setenv("AIS_TEST_KEY", KEY)
    return "AIS_TEST_KEY"


def test_voyages_split_at_rest_and_after_gaps(tmp_path):
    df = ais([
        (257000000, 0, 60.0, 10.0, 0),
        (257000000, 10, 60.1, 10.0, 0),
        (257000000, 20, 60.2, 0.0, 5),    # moored - dropped, ends the voyage
        (257000000, 30, 60.2, 10.0, 0),
        (257000000, 200, 60.3, 10.0, 0),  # gap of 170 min > 1 h
        (258000000, 0, 59.0, 10.0, 0),
    ])
    result = segment(df, tmp_path).sort("msgtime", "latitude")

    assert len(result) == 5
    voyages = dict(zip(result["latitude"].to_list(), result["voyage"].to_list(), strict=True))
    assert voyages[60.0] == voyages[60.1] == "257000000|2026-07-01 00:00:00.000000+00:00"
    assert len({voyages[60.0], voyages[60.2], voyages[60.3], voyages[59.0]}) == 4


def test_only_whitelisted_columns_remain(tmp_path, key_env):
    result = pseudonymize(ais([(257000000, 0, 60.0, 10.0, 0)]), tmp_path, key_env=key_env)

    assert set(result.columns) == {"voyage_id", *PSEUDONYMIZE_MAPPINGS.values()}
    assert result["shipType"].to_list() == [70]  # 71 coarsened to its category


def test_pseudonyms_depend_on_key(tmp_path, key_env, monkeypatch):
    df = ais([(257000000, 0, 60.0, 10.0, 0)])
    first = pseudonymize(df, tmp_path, key_env=key_env)["voyage_id"].to_list()
    again = pseudonymize(df, tmp_path, key_env=key_env)["voyage_id"].to_list()
    monkeypatch.setenv(key_env, "cd" * 32)
    other_key = pseudonymize(df, tmp_path, key_env=key_env)["voyage_id"].to_list()
    random_key = pseudonymize(df, tmp_path)["voyage_id"].to_list()

    assert first == again
    assert first != other_key
    assert first != random_key


def test_saved_pipeline_holds_no_key(tmp_path, key_env):
    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("segment", SegmentVoyages(), name_mappings=SEGMENT_MAPPINGS)
    pipeline.add("pseudonymize", PseudonymizeVoyages(key_env=key_env), name_mappings=PSEUDONYMIZE_MAPPINGS)
    pipeline.transform(df=annotate(ais([(257000000, 0, 60.0, 10.0, 0)])))
    text = pipeline.save(tmp_path).read_text()

    assert key_env in text
    assert KEY not in text


def test_suppress_voyage_endpoints(tmp_path):
    # one voyage due north in steps of 0.01 deg (~1.1 km)
    df = polars.DataFrame({
        "voyage_id": ["a"] * 7,
        "msgtime": [T0 + dt.timedelta(minutes=i) for i in range(7)],
        "latitude": [60.0 + 0.01 * i for i in range(7)],
        "longitude": [5.0] * 7,
    })
    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("endpoints", SuppressVoyageEndpoints(radius_km=2.0, min_fixes=1),
                 name_mappings={"group": "voyage_id", "sort": "msgtime", "lat": "latitude", "lon": "longitude"})
    result = pipeline.transform(df=annotate(df)).lazyframe.collect()

    # 0.02 deg ~ 2.2 km: only the fixes at least that far from both ends remain
    assert result["latitude"].to_list() == pytest.approx([60.02, 60.03, 60.04])

    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("endpoints", SuppressVoyageEndpoints(radius_km=2.0, min_fixes=4),
                 name_mappings={"group": "voyage_id", "sort": "msgtime", "lat": "latitude", "lon": "longitude"})
    assert len(pipeline.transform(df=annotate(df)).lazyframe.collect()) == 0


def test_generalize_space_time(tmp_path):
    df = polars.DataFrame({
        "msgtime": [T0 + dt.timedelta(minutes=13, seconds=7)] * 2,
        "latitude": [60.0, 60.0001],
        "longitude": [5.0, 5.0001],
        "speedOverGround": [10.4, 10.6],
        "courseOverGround": [356.0, 14.0],
    })
    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("generalize", GeneralizeSpaceTime(h3_resolution=7, time_bucket="10m"),
                 name_mappings={"timestamp": "msgtime", "lat": "latitude", "lon": "longitude",
                                "speed": "speedOverGround", "course": "courseOverGround"})
    result = pipeline.transform(df=annotate(df)).lazyframe.collect()

    assert result["cell"].n_unique() == 1
    assert result["latitude"].n_unique() == 1 and result["longitude"].n_unique() == 1
    assert result["msgtime"].to_list() == [T0 + dt.timedelta(minutes=10)] * 2
    assert result["speedOverGround"].to_list() == [10.0, 11.0]
    assert result["courseOverGround"].to_list() == [0.0, 10.0]


def test_k_anonymity_filter(tmp_path):
    df = polars.DataFrame({
        "cell": [1, 1, 1, 1, 2, 2],
        "msgtime": [T0] * 6,
        "voyage_id": ["a", "b", "c", "c", "a", "a"],
    })
    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("k", KAnonymityFilter(k=3), name_mappings={"timestamp": "msgtime", "id": "voyage_id"})
    result = pipeline.transform(df=annotate(df)).lazyframe.collect()

    assert result["cell"].to_list() == [1, 1, 1, 1]


def test_k_counts_vessels_not_voyages(tmp_path, key_env):
    # vessel 257000000 makes three voyages through the same cell within ten minutes, split by
    # stops; vessel 258000000 passes it once. Counted by voyage there would be 4 >= k
    rows = [(257000000, m, 60.0, s, 0) for m, s in [(0, 10.0), (1, 0.0), (2, 10.0), (3, 0.0), (4, 10.0)]]
    rows.append((258000000, 0, 60.0, 10.0, 0))
    pipeline = DataProcessingPipeline(name="anon", base_dir=str(tmp_path))
    pipeline.add("segment", SegmentVoyages(), name_mappings=SEGMENT_MAPPINGS)
    pipeline.add("generalize", GeneralizeSpaceTime(),
                 name_mappings={"timestamp": "msgtime", "lat": "latitude", "lon": "longitude",
                                "speed": "speedOverGround", "course": "courseOverGround"})
    pipeline.add("k", KAnonymityFilter(k=3), name_mappings={"timestamp": "msgtime", "id": "mmsi"})
    pipeline.add("pseudonymize", PseudonymizeVoyages(key_env=key_env), name_mappings=PSEUDONYMIZE_MAPPINGS)
    result = pipeline.transform(df=annotate(ais(rows))).lazyframe.collect()

    assert len(result) == 0


def crowd(n_vessels: int) -> polars.DataFrame:
    """n vessels sailing north side by side, 100 m apart, one fix a minute for an hour."""
    return ais([(257000000 + v, m, 60.0 + 0.001 * m, 10.0, 0) for v in range(n_vessels) for m in range(60)])


def test_strategy_releases_only_whitelisted_columns(tmp_path, key_env):
    columns = AISColumns(mmsi="MMSI", lat="lat")
    df = crowd(6).rename({"mmsi": "MMSI", "latitude": "lat"})
    anonymize = VoyageKAnonymity(key_env=key_env, k=5, time_bucket="1h", columns=columns, base_dir=tmp_path)
    result = anonymize.transform(df=annotate(df, columns)).lazyframe.collect()

    assert len(result) > 0
    assert set(result.columns) == {"voyage_id", "msgtime", "lat", "longitude", "speedOverGround",
                                   "courseOverGround", "navigationalStatus", "shipType"}
    assert result["voyage_id"].n_unique() == 6


def test_strategy_drops_crowds_smaller_than_k(tmp_path, key_env):
    anonymize = VoyageKAnonymity(key_env=key_env, k=7, time_bucket="1h", base_dir=tmp_path)

    assert len(anonymize.transform(df=annotate(crowd(6))).lazyframe.collect()) == 0


def test_strategy_chains_after_other_steps(tmp_path, key_env):
    df = crowd(6).vstack(ais([(257000000, 0, 61.0, 10.0, 0)]).with_columns(polars.lit(None).alias("name")))
    anonymize = VoyageKAnonymity(key_env=key_env, time_bucket="1h", base_dir=tmp_path)
    pipeline = DataProcessingPipeline(name="release", base_dir=str(tmp_path))
    pipeline.add("drop unnamed", DropMissingOrNan(), name_mappings={"x": "name"})
    pipeline.extend(anonymize)

    steps = [node.name for node in pipeline.processing_graph.nodes()]
    assert steps[:3] == ["df", "drop unnamed", "anonymize: segment voyages"]
    assert pipeline.transform(df=annotate(df)).lazyframe.collect().equals(
        anonymize.transform(df=annotate(crowd(6))).lazyframe.collect())


def test_saved_strategy_pipeline_reproduces_its_output(tmp_path, key_env):
    pipeline = VoyageKAnonymity(key_env=key_env, time_bucket="1h", base_dir=tmp_path)
    expected = pipeline.transform(df=annotate(crowd(6))).lazyframe.collect()

    loaded = DataProcessingPipeline.load(pipeline.save(tmp_path))
    result = loaded.transform(df=annotate(crowd(6))).lazyframe.collect()

    assert result.sort("voyage_id", "msgtime").equals(expected.sort("voyage_id", "msgtime"))
