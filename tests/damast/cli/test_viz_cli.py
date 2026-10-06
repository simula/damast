from pathlib import Path

import pytest
import yaml

from damast.cli.data_visualize import resolve_kind


@pytest.mark.parametrize("name,expected", [
    ("prepare.stats.yaml", "stats"),
    ("prepare.damast.ppl", "pipeline"),
    ("/some/dir/ais_join_sources.damast.ppl", "pipeline"),
])
def test_resolve_kind_follows_the_suffix(name: str, expected: str):
    assert resolve_kind(Path(name)) == expected


@pytest.mark.parametrize("name", ["README.md", "data.parquet", "prepare.yaml"])
def test_resolve_kind_rejects_anything_else(name: str):
    with pytest.raises(ValueError, match="expected a pipeline"):
        resolve_kind(Path(name))


def test_viz_renders_a_statistics_report(script_runner, tmp_path):
    report = tmp_path / "prepare.stats.yaml"
    report.write_text(yaml.dump({"name": "prepare", "steps": {
        "df": {"processing_time_in_s": 1.0,
               "input_dataframe_length": {"df": 10}, "output_dataframe_length": 10},
    }}, sort_keys=False))
    output = tmp_path / "run.svg"

    result = script_runner.run(["damast", "viz", "-f", str(report), "-o", str(output)])

    assert result.returncode == 0, result.stderr
    assert output.exists()


def test_viz_rejects_a_format_the_input_cannot_be_drawn_as(script_runner, tmp_path):
    report = tmp_path / "prepare.stats.yaml"
    report.write_text(yaml.dump({"name": "p", "steps": {}}, sort_keys=False))

    result = script_runner.run(["damast", "viz", "-f", str(report), "--format", "mermaid"])

    assert result.returncode != 0
    assert "Cannot write a stats visualization as 'mermaid'" in result.stdout + result.stderr
