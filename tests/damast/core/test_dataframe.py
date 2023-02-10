import pandas as pd
import pytest
import vaex
import astropy.units as units

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import DataCategory, DataSpecification, MetaData


@pytest.fixture()
def metadata():
    column_spec = DataSpecification(name="height",
                                    category=DataCategory.STATIC,
                                    unit=units.m)

    metadata = MetaData(columns=column_spec)
    return metadata


@pytest.fixture()
def vaex_dataframe():
    data = [
        [0, "a"],
        [1, "b"],
        [2, "c"]
    ]
    columns = [
        [
            "number", "letter"
        ]
    ]
    pandas_df = pd.DataFrame(data, columns=columns)
    return vaex.from_pandas(pandas_df)


def test_dataframe(metadata, vaex_dataframe, tmp_path):
    adf = AnnotatedDataFrame(dataframe=vaex_dataframe,
                             metadata=metadata)

    test_file = tmp_path / "test_dataframe.hdf5"
    adf.export_hdf5(test_file)
    assert test_file.exists()
