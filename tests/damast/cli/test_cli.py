
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from zipfile import ZipFile

import polars
import pytest
import yaml

import damast.cli.main as cli_main
from damast.cli.data_annotate import DataAnnotateParser
from damast.cli.data_converter import DataConvertParser
from damast.cli.data_inspect import DataInspectParser
from damast.cli.data_processing import DataProcessingParser
from damast.cli.experiment import ExperimentParser
from damast.core.dataframe import DAMAST_SPEC_SUFFIX, AnnotatedDataFrame
from damast.domains.maritime.ais.data_generator import AISTestData


@pytest.fixture
def subparsers():
    return [
        "annotate",
        "convert",
        "experiment",
        "inspect",
        "process",
    ]

def test_help(subparsers, capsys, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['damast'])
    cli_main.run()
    captured = capsys.readouterr()

    for subparser in subparsers:
        assert re.search(subparser, captured.out)


@pytest.mark.parametrize("name, klass", [
    [ "annotate", DataAnnotateParser ],
    [ "convert", DataConvertParser ],
    [ "inspect", DataInspectParser ],
    [ "experiment", ExperimentParser ],
    [ "process", DataProcessingParser ],
])
def test_subparser(name, klass, script_runner):
    result = script_runner.run(['damast', name, "--help"])
    assert result.returncode == 0

    test_parser = ArgumentParser()
    klass(parser=test_parser)

    for a in test_parser._actions:
        if a.help == "==SUPPRESS==":
            continue

        for option in a.option_strings:
            assert re.search(option, result.stdout) is not None, f"Should have {option=}"

@pytest.mark.parametrize("filename", [
    "data.hdf5",
    "test_ais.parquet",
    "test_ais.csv",
    "test_dataframe.csv",
    "test_dataframe.hdf5",
])
def test_inspect(data_path, filename, script_runner):
    result = script_runner.run(['damast', 'inspect', '-f', str(data_path / filename)])

    assert re.search(r"Loading dataframe \(1 files\)", result.stdout) is not None, "Process dataframe"
    assert re.search("shape:", result.stdout) is not None, "Show dataframe"

@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_annotate(data_path, filename, spec_filename, tmp_path, script_runner):
    result = script_runner.run(['damast', 'annotate', '-f', str(data_path / filename), '-o', tmp_path])

    assert result.returncode == 0

    with open(tmp_path / spec_filename, "r") as f:
        written_spec = yaml.load(f, Loader=yaml.SafeLoader)

    with open(data_path / spec_filename, "r") as f:
        expected_spec = yaml.load(f, Loader=yaml.SafeLoader)
        expected_spec["annotations"]["source"] = [str(data_path / filename)]

    assert written_spec['annotations'] == expected_spec['annotations']

    written_columns = written_spec['columns']
    expected_columns = expected_spec['columns']

    for idx, column in enumerate(written_columns):
        for field in column:
            if type(field) is float:
                column.assert_approx(expected_columns[idx][field])
            elif type(field) is dict:
                for subfield in field:
                    expected_subfield_value = expected_columns[idx][field][subfield]
                    written_subfield_value = field[subfield]
                    expected_subfield_value.assert_approx(written_subfield_value)
            else:
                column == expected_columns[idx][field]


@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_annotate_non_existing_column(data_path, filename, spec_filename, tmp_path, script_runner):
    result = script_runner.run(['damast', 'annotate', '-f', str(data_path / filename), '-o', tmp_path, '--set-unit', 'no-column:deg'])

    assert result.returncode == 1
    assert re.search(r"'no-column' does not exist", result.stdout) is not None, "Error on missing column presented"


@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_convert(data_path, filename, spec_filename, tmp_path, script_runner):

    output_file = Path(tmp_path) / (Path(filename).stem + ".parquet")

    result = script_runner.run(['damast', 'convert', '-f', str(data_path / filename), '--output-dir', tmp_path])
    assert result.returncode == 0
    assert output_file

    result = script_runner.run(['damast', 'convert', '-f', spec_filename, '--output-dir', tmp_path])
    assert result.returncode != 0

@pytest.mark.skipif(sys.platform.startswith("win"), reason="ratarmount does not (easily) run on windows")
@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_convert_zip(data_path, filename, spec_filename, tmp_path, script_runner):
    output_zip = tmp_path / f"{Path(filename)}.zip"
    with ZipFile(output_zip, 'w') as f:
        f.write(str(data_path / filename), arcname=filename)
        f.write(str(data_path / spec_filename), arcname=spec_filename)


    assert Path(output_zip).exists()

    output_file = Path(tmp_path) / (Path(filename).stem + ".parquet")

    result = script_runner.run(['damast', 'convert', '-f', output_zip, '--output-dir', tmp_path])
    assert result.returncode == 0
    assert output_file.exists()

#@pytest.mark.parametrize("filename, spec_filename", [
#    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
#])
#def test_fail_convert_zip(data_path, filename, spec_filename, tmp_path, script_runner, monkeypatch):
#    import damast
#    monkeypatch.setattr(damast.utils.io, "DAMAST_ARCHIVE_SUPPORT_AVAILABLE", False)
#
#    output_zip = tmp_path / f"{Path(filename)}.zip"
#    with ZipFile(output_zip, 'w') as f:
#        f.write(str(data_path / filename), arcname=filename)
#        f.write(str(data_path / spec_filename), arcname=spec_filename)
#
#    assert Path(output_zip).exists()
#    output_file = Path(tmp_path) / (Path(filename).stem + ".parquet")
#
#    result = script_runner.run(['damast', 'convert', '-f', output_zip, '--output-dir', tmp_path])
#    assert result.returncode == 1
#    assert not output_file.exists()


@pytest.mark.skipif(sys.platform.startswith("win"), reason="ratarmount does not (easily) run on windows")
@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_convert_zip_zip(data_path, filename, spec_filename, tmp_path, script_runner):

    output_zip = Path("/tmp") / f"{Path(filename)}.zip"
    with ZipFile(output_zip, 'w') as f:
        f.write(str(data_path / filename), arcname=filename)
        f.write(str(data_path / spec_filename), arcname=spec_filename)

    assert Path(output_zip).exists()

    output_zip_zip = tmp_path / f"{Path(filename)}.zip.zip"
    with ZipFile(output_zip_zip, 'w') as f:
        f.write(output_zip, arcname=output_zip_zip.name)

    assert Path(output_zip_zip).exists()

    output_file = Path(tmp_path) / (Path(filename).stem + ".parquet")

    result = script_runner.run(['damast', 'convert', '-f', output_zip_zip, '--output-dir', tmp_path])
    assert result.returncode == 0
    assert output_file.exists()

def test_convert_n_to_1(tmp_path, script_runner):
    input_files = []
    output_file = Path(tmp_path) / "test-convert-n-to-1.parquet"

    number_of_trajectories = 10
    max_range = 3
    for i in range(0, max_range):
        input_file = Path(tmp_path) / f"{i}.csv"
        input_files.append(input_file.resolve())

        ais_test_data = AISTestData(number_of_trajectories=number_of_trajectories)
        ais_test_data.dataframe.write_csv(str(input_file))

    run_cmd = ['damast', 'convert', '-f']
    run_cmd += input_files
    run_cmd += ['--output-file', str(output_file)]

    result = script_runner.run(run_cmd)
    assert result.returncode == 0
    assert output_file

    df = AnnotatedDataFrame.from_file(output_file, metadata_required=False)
    number_of_mmsis = len(set(df.mmsi.collect().to_series()))
    assert number_of_mmsis <= number_of_trajectories*max_range
    assert number_of_mmsis >= number_of_trajectories*(max_range-1)

def test_convert_n_to_n(tmp_path, script_runner):
    input_files = []

    number_of_trajectories = 10
    max_range = 3
    for i in range(0, max_range):
        input_file = Path(tmp_path) / f"{i}.csv"
        input_files.append(input_file.resolve())

        ais_test_data = AISTestData(number_of_trajectories=number_of_trajectories)
        ais_test_data.dataframe.write_csv(str(input_file))

    run_cmd = ['damast', 'convert', '-f']
    run_cmd += input_files
    run_cmd += ['--output-dir', str(tmp_path)]

    result = script_runner.run(run_cmd)
    assert result.returncode == 0
    for f in input_files:
        output_file = tmp_path / f"{f.stem}.parquet"
        assert output_file

        df = AnnotatedDataFrame.from_file(output_file, metadata_required=False)
        number_of_mmsis = len(set(df.mmsi.collect().to_series()))
        assert number_of_mmsis == number_of_trajectories


def test_annotate_representation_type(tmp_path, script_runner):
    """
    Use case:
        a column is automatically inferred to be of a numeric type, but should be actually of string type
        the user corrects the type using damast annotate --set-representation_type column_name:String -f <filename>
        user checks the result via damast inspect -f <filename>

    While polar keeps autoinferring the type to int, the embedded schema will be used to
    cast the data to the spec/expected type
    """
    df = polars.DataFrame({
            "a": [1,2,3,4],
            "b": ["A", "B", "C", "D"]
        })

    data_file = tmp_path / "test.parquet"
    df.write_parquet(data_file)

    result = script_runner.run(['damast', "annotate", "-f", data_file, "--set-representation_type", "a:String"])
    expected_data_file = Path(tmp_path / "test-annotated.parquet")
    expected_spec_file = Path(tmp_path / "test-annotated.spec.yaml")

    assert expected_data_file.exists()
    assert expected_spec_file.exists()

    result = script_runner.run(['damast', "inspect", "-f", expected_data_file])

    # The new representation type should now be str
    assert re.search(r"a:\n\s+is_optional: False\n\s+representation_type: str",result.stdout)
    assert result.returncode == 0


