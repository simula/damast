from pathlib import Path

import astropy.units as units
import pandas as pd
import pytest
import vaex
from vaex.dataframe import astropy

from damast.core.annotations import Annotation
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.datarange import MinMax
from damast.core.metadata import DataCategory, DataSpecification, MetaData


@pytest.fixture()
def metadata():
    column_spec = DataSpecification(name="height",
                                    category=DataCategory.STATIC,
                                    unit=units.m)

    license = Annotation(name=Annotation.Key.License, value="MIT License")
    comment = Annotation(name=Annotation.Key.Comment, value="test dataframe")
    annotations = {
        Annotation.Key.License.value: license,
        Annotation.Key.Comment.value: comment
    }

    metadata = MetaData(columns=[column_spec], annotations=annotations)
    return metadata


@pytest.fixture()
def vaex_dataframe():
    data = [
        [0, "a"],
        [1, "b"],
        [2, "c"]
    ]
    columns = ["height", "letter"]
    pandas_df = pd.DataFrame(data, columns=columns)
    return vaex.from_pandas(pandas_df)


def test_annotated_dataframe_export_hdf5(metadata, vaex_dataframe, tmp_path):
    """
    Simple test of the annotated dataframe export to HDF5

    :param metadata: metadata to use
    :param vaex_dataframe: vaex dataframe to use
    :param tmp_path: where to temporarily save the data to HDF5
    """
    adf = AnnotatedDataFrame(dataframe=vaex_dataframe,
                             metadata=metadata)

    test_file = tmp_path / "test_dataframe.hdf5"
    adf.save(filename=test_file)
    assert test_file.exists()


def test_annotated_dataframe_export_csv(metadata, vaex_dataframe, tmp_path):
    """
    Simple test of the annotated dataframe export to csv

    :param metadata: metadata to use
    :param vaex_dataframe: vaex dataframe to use
    :param tmp_path: where to temporarily save the data to csv
    """
    adf = AnnotatedDataFrame(dataframe=vaex_dataframe,
                             metadata=metadata)

    test_file = tmp_path / "test_dataframe.csv"
    metadata_test_file = tmp_path / "test_dataframe.spec.yaml"
    adf.save(filename=test_file)
    assert test_file.exists()
    assert metadata_test_file.exists()


def test_annotated_dataframe_import_hdf5():
    """
    Simple test of the annotated dataframe import for HDF5
    """
    data_path = Path(__file__).parent.parent / "data"
    hdf5_path = data_path / "data.hdf5"

    adf = AnnotatedDataFrame.from_file(hdf5_path)
    assert adf.column_names == ["height", "letter"]
    assert adf._dataframe.to_pandas_df().equals(vaex.open(hdf5_path).to_pandas_df())
    assert adf._metadata.annotations["license"] == Annotation(name="license", value="MIT License")
    assert adf._metadata.annotations["comment"] == Annotation(name="comment", value="test dataframe")
    assert adf._metadata.columns[0] == DataSpecification(
        name="height", abbreviation="height", category=DataCategory.STATIC, unit=units.m)


def test_annotated_dataframe_import_csv():
    """
    Simple test of the annotated dataframe import for HDF5
    """
    data_path = Path(__file__).parent.parent / "data"
    hdf5_path = data_path / "test_dataframe.csv"

    adf = AnnotatedDataFrame.from_file(hdf5_path)
    assert adf.column_names == ["height", "letter"]
    assert adf._dataframe.to_pandas_df().equals(vaex.open(hdf5_path).to_pandas_df())
    assert adf._metadata.annotations["license"] == Annotation(name="license", value="MIT License")
    assert adf._metadata.annotations["comment"] == Annotation(name="comment", value="test dataframe")
    assert adf._metadata.columns[0] == DataSpecification(
        name="height", abbreviation="height", category=DataCategory.STATIC, unit=units.m)


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

    adf = AnnotatedDataFrame(dataframe=df,
                             metadata=md)

    assert adf._metadata == md
    assert adf._dataframe == df

    assert adf.column_names == df.column_names

    md.columns[0].value_range = MinMax(min=0, max=1)
    with pytest.raises(ValueError, match="lies outside of range"):
        md.apply(df=df)

    with pytest.raises(ValueError, match="lies outside of range"):
        AnnotatedDataFrame(dataframe=df, metadata=md)
