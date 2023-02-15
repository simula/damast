from pathlib import Path

import astropy.units as units
import pandas as pd
import pytest
import vaex
from vaex.dataframe import astropy

from damast.core.dataframe import AnnotatedDataFrame
from damast.core.datarange import MinMax
from damast.core.metadata import DataCategory, DataSpecification, MetaData


@pytest.fixture()
def metadata():
    column_spec = DataSpecification(name="height",
                                    category=DataCategory.STATIC,
                                    unit=units.m)

    metadata = MetaData(columns=[column_spec])
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


def test_annotated_dataframe_export(metadata, vaex_dataframe, tmp_path):
    """
    Simple test of the annotated dataframe export

    :param metadata: metadata to use
    :param vaex_dataframe: vaex dataframe to use
    :param tmp_path: where to temporarily save the data
    """
    adf = AnnotatedDataFrame(dataframe=vaex_dataframe,
                             metadata=metadata)

    test_file = tmp_path / "test_dataframe.hdf5"
    adf.export_hdf5(test_file)
    assert test_file.exists()


def test_01_dataframe_composition():
    """
    Test the dataframe composition, i.e. metadata in combination with an actual dataframe
    """
    data_path = Path(__file__).parent.parent / "data" / "01_dataframe_composition"
    csv_path = data_path / "data.csv"
    spec_path = data_path / "dataspec.yaml"

    md = MetaData.load_yaml(filename=spec_path)
    df = vaex.from_csv(filename_or_buffer=csv_path)

    md.apply(df=df)

    # If the units are given in the spec, ensure that the dataframe is augmented
    assert df.units["fullname-a"] == astropy.units.s
    assert df.units["fullname-b"] == astropy.units.m

    # FIXME: Should `adf` be unused?
    adf = AnnotatedDataFrame(dataframe=df,
                             metadata=md)

    md.columns[0].value_range = MinMax(min=0, max=1)
    with pytest.raises(ValueError, match="lies outside of range"):
        md.apply(df=df)

    with pytest.raises(ValueError, match="lies outside of range"):
        AnnotatedDataFrame(dataframe=df, metadata=md)
