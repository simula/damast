from pathlib import Path
from zipfile import ZipFile

import pytest

from damast.core.dataframe import DAMAST_SPEC_SUFFIX, AnnotatedDataFrame
from damast.utils.io import Archive


@pytest.mark.parametrize("filename, spec_filename", [
    ["test_ais.csv", f"test_ais{DAMAST_SPEC_SUFFIX}"]
])
def test_archive(data_path, filename, spec_filename, tmp_path):
    output_zip = tmp_path / f"{Path(filename)}.zip"
    with ZipFile(output_zip, 'w') as f:
        f.write(str(data_path / filename), arcname=filename)
        f.write(str(data_path / spec_filename), arcname=spec_filename)


    assert Path(output_zip).exists()

    # default no filter
    with Archive(filenames=[output_zip]) as input_files:
        assert len(input_files) == 2

        filenames = [x.name for x in input_files]

        assert filename in filenames
        assert spec_filename in filenames

    # permit only supported files
    with Archive(filenames=[output_zip], filter_fn = lambda x : AnnotatedDataFrame.get_supported_format(Path(x).suffix) is None) as input_files:
        assert len(input_files) == 1

        assert filename in [x.name for x in input_files]


