import re
import sys

import pytest

from damast.cli.data_processing import resolve_input_data
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.transformations import plugin_manager
from damast.data_handling.transformers.filters import DropMissingOrNan

JOIN_TRANSFORMER_SOURCE = '''
import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.transformations import PipelineElement


class JoinByTimestamp(PipelineElement):
    @damast.core.describe("Join by timestamp")
    @damast.core.input({"timestamp": {}, "lon": {}, "lat": {}})
    @damast.core.input({"timestamp": {}, "lat": {}, "lon": {}}, label="other")
    @damast.core.output({
        "{{other:event_type}}": {"representation_type": str},
        "{{other:lat}}": {},
        "{{other:lon}}": {},
    })
    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame) -> AnnotatedDataFrame:
        other_timestamp = self.get_name("timestamp", datasource="other")
        df_timestamp = self.get_name("timestamp")
        df.lazyframe = df.join(other.lazyframe, left_on=df_timestamp, right_on=other_timestamp)
        df._metadata = df._metadata.merge(other._metadata).drop(other_timestamp)
        return df
'''


# --- _resolve_input_data ------------------------------------------------------------------

def test_resolve_input_data_bare_form_single_datasource():
    assert resolve_input_data([["1.csv"]], ["df"]) == {"df": ["1.csv"]}


def test_resolve_input_data_flattens_repeated_bare_occurrences():
    assert resolve_input_data([["1.csv"], ["2.csv"]], ["df"]) == {"df": ["1.csv", "2.csv"]}


def test_resolve_input_data_named_datasources():
    resolved = resolve_input_data([["df=1.csv"], ["osint=2.csv", "3.csv"]], ["df", "osint"])
    assert resolved == {"df": ["1.csv"], "osint": ["2.csv", "3.csv"]}


def test_resolve_input_data_requires_input():
    with pytest.raises(RuntimeError, match="--input-data is required"):
        resolve_input_data(None, ["df"])


def test_resolve_input_data_requires_name_prefix_for_multiple_datasources():
    with pytest.raises(RuntimeError, match="prefix each --input-data occurrence"):
        resolve_input_data([["1.csv"]], ["df", "osint"])


def test_resolve_input_data_rejects_unknown_datasource():
    with pytest.raises(RuntimeError, match="unknown datasource 'foo'"):
        resolve_input_data([["foo=1.csv"], ["osint=2.csv"]], ["df", "osint"])


def test_resolve_input_data_rejects_duplicate_datasource():
    with pytest.raises(RuntimeError, match="given more than once"):
        resolve_input_data([["df=1.csv"], ["df=2.csv"]], ["df", "osint"])


def test_resolve_input_data_reports_missing_datasource():
    with pytest.raises(RuntimeError, match="missing input for datasource\\(s\\) osint"):
        resolve_input_data([["df=1.csv"]], ["df", "osint"])


# --- end-to-end: single-datasource pipeline ------------------------------------------------

@pytest.fixture
def simple_pipeline_path(tmp_path):
    pipeline = DataProcessingPipeline(name="simple", base_dir=tmp_path) \
        .add("drop_missing_mmsi", DropMissingOrNan(), name_mappings={"x": "mmsi"})
    return pipeline.save(tmp_path)


def test_process_bare_input_data_is_backward_compatible(data_path, simple_pipeline_path, tmp_path, script_runner):
    output_file = tmp_path / "result.parquet"
    result = script_runner.run([
        "damast", "process",
        "--pipeline", str(simple_pipeline_path),
        "--input-data", str(data_path / "test_ais.csv"),
        "--output-file", str(output_file),
    ])

    assert result.returncode == 0, result.stdout
    assert output_file.exists()


def test_process_requires_input_data_or_describe(simple_pipeline_path, script_runner):
    result = script_runner.run(["damast", "process", "--pipeline", str(simple_pipeline_path)])

    assert result.returncode != 0
    assert "--input-data is required" in result.stdout


def test_process_describe_single_datasource(simple_pipeline_path, script_runner):
    result = script_runner.run(["damast", "process", "--pipeline", str(simple_pipeline_path), "--describe"])

    assert result.returncode == 0, result.stdout
    assert "Pipeline: simple" in result.stdout
    assert "Interface:" in result.stdout
    assert re.search(r"df:\s*\n\s*mmsi", result.stdout)
    assert "Steps:" in result.stdout
    assert "DropMissingOrNan" in result.stdout


# --- end-to-end: multi-datasource (join) pipeline ------------------------------------------

@pytest.fixture
def join_pipeline_path(tmp_path, monkeypatch):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "join_transformer.py").write_text(JOIN_TRANSFORMER_SOURCE)
    monkeypatch.setenv("DAMAST_PLUGIN_PATH", str(plugin_dir))
    plugin_manager.reload()

    from damast.plugins import JoinByTimestamp

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    pipeline = DataProcessingPipeline(name="ais_osint", description="Join AIS with OSINT events",
                                      base_dir=output_dir) \
        .join("osint", JoinByTimestamp(),
              name_mappings={
                  "df": {"timestamp": "date_time_utc"},
                  "other": {"timestamp": "timestamp", "lat": "latitude", "lon": "longitude"},
              })
    path = pipeline.save(output_dir)

    yield path

    # drop the cached plugin module so it doesn't leak into other tests sharing the
    # process-wide plugin_manager - mirrors _reset_plugin_manager() in test_plugins.py
    for module_name in list(plugin_manager.local_files):
        sys.modules.pop(module_name, None)
    plugin_manager._local_modules.clear()
    plugin_manager._local_files.clear()
    plugin_manager._requirement_cache.clear()
    plugin_manager._loaded = False


def test_process_multi_datasource_input_data(data_path, join_pipeline_path, tmp_path, script_runner):
    output_file = tmp_path / "joined.parquet"
    result = script_runner.run([
        "damast", "process",
        "--pipeline", str(join_pipeline_path),
        "--input-data", f"df={data_path / 'test_ais.csv'}",
        "--input-data", f"osint={data_path / 'osint.csv'}",
        "--output-file", str(output_file),
    ])

    assert result.returncode == 0, result.stdout
    assert output_file.exists()


def test_process_multi_datasource_requires_name_prefix(data_path, join_pipeline_path, script_runner):
    result = script_runner.run([
        "damast", "process",
        "--pipeline", str(join_pipeline_path),
        "--input-data", str(data_path / "test_ais.csv"),
    ])

    assert result.returncode != 0
    assert "prefix each --input-data occurrence" in result.stdout


def test_process_describe_multi_datasource(join_pipeline_path, script_runner):
    result = script_runner.run(["damast", "process", "--pipeline", str(join_pipeline_path), "--describe"])

    assert result.returncode == 0, result.stdout
    assert "Pipeline: ais_osint" in result.stdout
    assert "Description: Join AIS with OSINT events" in result.stdout
    assert "Interface:" in result.stdout
    assert re.search(r"df:\s*\n\s*date_time_utc", result.stdout)
    assert re.search(r"osint:\s*\n\s*timestamp", result.stdout)
    assert "Steps:" in result.stdout
    assert "JoinByTimestamp" in result.stdout
