import datetime as dt

import polars
import yaml

from damast.core.dataframe import AnnotatedDataFrame


def test_privacy_writes_report(tmp_path, script_runner):
    data = tmp_path / "ais.parquet"
    polars.DataFrame({
        "MMSI": [257000000, 257000000, 257000001],
        "time": [dt.datetime(2026, 7, 1, 0, m) for m in range(3)],
        "latitude": [60.0, 60.001, 60.002],
        "longitude": [5.0, 5.0, 5.0],
    }).write_parquet(data)
    report = tmp_path / "ais.privacy.yaml"

    result = script_runner.run(["damast", "privacy", "-f", str(data), "-o", str(report),
                                "--column", "mmsi=MMSI", "timestamp=time", "-k", "3"])

    assert result.returncode == 0, result.stderr
    content = yaml.safe_load(report.read_text())
    assert content["settings"]["k"] == 3
    assert content["direct_identifiers"]["columns"] == {"MMSI": 2}
    assert content["k_anonymity"]["rows_below_k"] == 3


def crowd_file(path, n_vessels: int = 6):
    """n cargo ships sailing north together for an hour, with BarentsWatch's text timestamps."""
    rows = [(257000000 + v, m) for v in range(n_vessels) for m in range(60)]
    df = polars.DataFrame({
        "mmsi": [r[0] for r in rows],
        "msgtime": [(dt.datetime(2026, 7, 1) + dt.timedelta(minutes=r[1])).strftime("%Y-%m-%dT%H:%M:%S+00:00")
                    for r in rows],
        "latitude": [60.0 + 0.001 * r[1] for r in rows],
        "longitude": [5.0] * len(rows),
        "speedOverGround": [10.0] * len(rows),
        "courseOverGround": [0.0] * len(rows),
        "navigationalStatus": [0] * len(rows),
        "shipType": [71] * len(rows),
        "imoNumber": polars.Series([None] * len(rows), dtype=polars.Int64),
        "callSign": ["LA1234"] * len(rows),
        "name": ["SECRET"] * len(rows),
    })
    adf = AnnotatedDataFrame(df, AnnotatedDataFrame.infer_annotation(df))
    for column in ("latitude", "longitude", "courseOverGround"):
        adf.metadata[column].unit = "deg"
    adf.export(path)
    return path


def test_convert_privacy_minimize(tmp_path, script_runner):
    data = crowd_file(tmp_path / "ais.parquet")
    output = tmp_path / "out" / "ais.min.parquet"

    result = script_runner.run(["damast", "convert", "-f", str(data), "-o", str(output), "--privacy", "minimize"])

    assert result.returncode == 0, result.stderr
    converted = polars.read_parquet(output)
    assert "mmsi" in converted.columns
    assert not {"imoNumber", "callSign", "name"} & set(converted.columns)
    report = yaml.safe_load((tmp_path / "out" / "ais.min.privacy.yaml").read_text())
    assert report["privacy"] == "minimize"
    assert report["dropped_columns"] == ["imoNumber", "callSign", "name"]
    assert report["output"]["direct_identifiers"]["columns"] == {"mmsi": 6}
    assert report["output"]["precision"]["latitude_decimals"] <= 4
    assert (tmp_path / "out" / "ais.min.damast.ppl").exists()


def test_convert_privacy_k_anonymity(tmp_path, script_runner, monkeypatch):
    monkeypatch.setenv("AIS_TEST_KEY", "ab" * 32)
    data = crowd_file(tmp_path / "ais.parquet")
    output = tmp_path / "ais.anon.parquet"
    report_path = tmp_path / "report.yaml"

    result = script_runner.run(["damast", "convert", "-f", str(data), "-o", str(output),
                                "--privacy", "k-anonymity", "--privacy-key-env", "AIS_TEST_KEY",
                                "--privacy-report", str(report_path), "--time-bucket", "1h"])

    assert result.returncode == 0, result.stderr
    assert "mmsi" not in polars.read_parquet(output).columns
    report = yaml.safe_load(report_path.read_text())
    assert report["output"]["direct_identifiers"]["columns"] == {}
    assert report["output"]["k_anonymity"]["rows_below_k"] == 0
    assert (tmp_path / "report.damast.ppl").exists()


def test_convert_privacy_report_keeps_data(tmp_path, script_runner):
    data = crowd_file(tmp_path / "ais.parquet")
    output = tmp_path / "copy.parquet"

    result = script_runner.run(["damast", "convert", "-f", str(data), "-o", str(output), "--privacy", "report"])

    assert result.returncode == 0, result.stderr
    original = polars.read_parquet(data)
    assert polars.read_parquet(output).select(original.columns).equals(original)
    report = yaml.safe_load((tmp_path / "copy.privacy.yaml").read_text())
    assert "output" not in report
    assert report["input"]["dataset"]["rows"] == 360


def test_convert_privacy_needs_one_report_path(tmp_path, script_runner):
    data = crowd_file(tmp_path / "ais.parquet")

    per_file = script_runner.run(["damast", "convert", "-f", str(data), "--output-dir", str(tmp_path / "out"),
                                  "--privacy", "report"])
    partitioned = script_runner.run(["damast", "convert", "-f", str(data), "--privacy", "report",
                                     "--save-as", f"column:mmsi:{tmp_path}/by-mmsi/{{mmsi}}"])

    assert per_file.returncode == 1
    assert "--save-as or --output-file" in per_file.stdout
    assert partitioned.returncode == 1
    assert "--privacy-report" in partitioned.stdout
