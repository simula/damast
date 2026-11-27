import datetime as dt

import polars
import pytest
import yaml

from damast.cli.data_privacy import ais_columns
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import DataSpecification, MetaData
from damast.domains.maritime.anonymization import AISColumns, PrivacyReport, VoyageKAnonymity

T0 = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
LONE = 258000000


def annotate(df: polars.DataFrame) -> AnnotatedDataFrame:
    columns = AISColumns()
    units = {columns.lat: "deg", columns.lon: "deg", columns.course: "deg"}
    ints = {columns.mmsi, columns.status, columns.ship_type, columns.imo}
    return AnnotatedDataFrame(df, MetaData(columns=[
        DataSpecification(c, unit=units.get(c), representation_type=int if c in ints else None)
        for c in df.columns
    ]))


def ais(rows: list[tuple]) -> polars.DataFrame:
    """Rows of (mmsi, minutes after T0, latitude, speed, navigational status, ship type) at longitude 5."""
    return polars.DataFrame({
        "mmsi": [r[0] for r in rows],
        "msgtime": [T0 + dt.timedelta(minutes=r[1]) for r in rows],
        "latitude": [r[2] for r in rows],
        "longitude": [5.0] * len(rows),
        "speedOverGround": [r[3] for r in rows],
        "courseOverGround": [0.0] * len(rows),
        "navigationalStatus": [r[4] for r in rows],
        "shipType": [r[5] for r in rows],
        "imoNumber": polars.Series([None] * len(rows), dtype=polars.Int64),
        "reportClass": ["A"] * len(rows),
        "name": ["SECRET"] * len(rows),
    })


def crowd(n_vessels: int) -> list[tuple]:
    """n cargo ships sailing north together, one fix a minute for an hour."""
    return [(257000000 + v, m, 60.0 + 0.001 * m, 10.0, 0, 71) for v in range(n_vessels) for m in range(60)]


@pytest.fixture
def data() -> polars.DataFrame:
    """A crowd of 6 ships, and a pleasure craft on its own far away - moored for its last 2 fixes."""
    lone = [(LONE, m, 70.0, 5.0 if m < 8 else 0.0, 0 if m < 8 else 5, 37) for m in range(10)]
    return ais(crowd(6) + lone)


def test_report_of_raw_data(data):
    report = PrivacyReport.assess(annotate(data), k=5, time_bucket="1h").to_dict()

    assert report["dataset"]["rows"] == 370
    assert report["dataset"]["vessels"] == 7
    assert report["direct_identifiers"]["columns"] == {"mmsi": 7, "imoNumber": 1, "name": 1}
    assert report["owner_risk"]["high"] == {"vessels": 1, "rows": 10, "rows_percent": 2.7}
    assert report["owner_risk"]["medium"]["vessels"] == 6
    assert report["at_rest"]["rows"] == 2
    assert report["precision"]["min_time_step_s"] == 60.0
    assert report["k_anonymity"]["rows_below_k"] == 10
    assert report["k_anonymity"]["vessels_affected"] == 1
    assert set(report["minimization"]["not_released"]) == {"mmsi", "imoNumber", "reportClass", "name"}
    assert len(report["findings"]) == 5


def test_sections_without_their_columns_are_skipped(data):
    report = PrivacyReport.assess(annotate(data.drop("shipType", "navigationalStatus", "speedOverGround"))).to_dict()

    assert report["owner_risk"] == {"skipped": "missing columns: shipType"}
    assert "skipped" in report["at_rest"]
    assert "skipped" not in report["k_anonymity"]


def test_report_of_released_data_shows_no_risk(tmp_path, monkeypatch):
    monkeypatch.setenv("AIS_TEST_KEY", "ab" * 32)
    released = VoyageKAnonymity(key_env="AIS_TEST_KEY", k=5, time_bucket="1h", base_dir=tmp_path) \
        .transform(df=annotate(ais(crowd(6))))

    report = PrivacyReport.assess(released, k=5, time_bucket="1h").to_dict()

    assert report["direct_identifiers"]["columns"] == {}
    assert report["k_anonymity"]["vessel_column"] == "voyage_id"
    assert report["k_anonymity"]["rows_below_k"] == 0
    assert report["minimization"]["not_released"] == []


def test_report_saves_as_yaml(data, tmp_path):
    report = PrivacyReport.assess(annotate(data))

    path = report.save_yaml(tmp_path / "sub" / "data.privacy.yaml")

    assert yaml.safe_load(path.read_text()) == report.to_dict()


def test_ais_columns_from_arguments():
    assert ais_columns(["mmsi=MMSI", "lat=Latitude"]) == AISColumns(mmsi="MMSI", lat="Latitude")
    with pytest.raises(ValueError, match="field=column"):
        ais_columns(["vessel=MMSI"])
