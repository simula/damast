import datetime as dt

import polars
import yaml


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
