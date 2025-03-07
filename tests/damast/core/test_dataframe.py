import copy
from pathlib import Path

import astropy
import astropy.units as units
import numpy as np
import pandas as pd
import polars
import polars.testing
import pytest

from damast.core.annotations import Annotation
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.datarange import MinMax
from damast.core.metadata import (
    DataCategory,
    DataSpecification,
    MetaData,
    ValidationMode,
    )
from damast.core.types import XDataFrame


@pytest.fixture()
def metadata():
    column_spec = DataSpecification(name="height",
                                    category=DataCategory.STATIC,
                                    unit=units.m,
                                    abbreviation="height",
                                    value_range=MinMax(min=0, max=40))

    license = Annotation(name=Annotation.Key.License, value="MIT License")
    comment = Annotation(name=Annotation.Key.Comment, value="test dataframe")
    annotations = [license, comment]

    return MetaData(columns=[column_spec], annotations=annotations)


@pytest.fixture()
def polars_dataframe():
    data = [
        [0, "a"],
        [1, "b"],
        [2, "c"]
    ]
    columns = ["height", "letter"]
    pandas_df = pd.DataFrame(data, columns=columns)
    return polars.from_pandas(pandas_df)

def test_annotated_dataframe_wrong_init(polars_dataframe):
    with pytest.raises(ValueError, match="must be of type 'DataFrame'"):
        AnnotatedDataFrame(dataframe="test",
                           metadata="any")

    with pytest.raises(ValueError, match="must be of type 'MetaData'"):
        AnnotatedDataFrame(dataframe=polars_dataframe,
                           metadata="any")

def test_annotated_dataframe_deep_copy(metadata, polars_dataframe):
    """
    Validate the deep copy functionality of the dataframe"
    """

    adf = AnnotatedDataFrame(dataframe=polars_dataframe,
                             metadata=metadata)

    adf_copy = copy.deepcopy(adf)

    assert adf_copy.metadata.columns == adf.metadata.columns
    assert adf_copy.dataframe.column_names == adf.dataframe.column_names

    old_name = adf_copy.metadata.columns[0].name
    adf_copy.metadata.columns[0].name = "new-name"
    assert adf.metadata.columns[0].name == old_name

    column_names = adf.dataframe.column_names

    adf_copy.drop(adf_copy.dataframe.column_names)
    assert adf.dataframe.column_names == column_names
    assert adf_copy.dataframe.column_names != column_names


def test_annotated_dataframe_export_hdf5(metadata, polars_dataframe, tmp_path):
    """
    Simple test of the annotated dataframe export to HDF5

    :param metadata: metadata to use
    :param polars_dataframe: polars dataframe to use
    :param tmp_path: where to temporarily save the data to HDF5
    """
    adf = AnnotatedDataFrame(dataframe=polars_dataframe,
                             metadata=metadata)

    test_file = tmp_path / "test_dataframe.hdf5"
    adf.save(filename=test_file)
    assert test_file.exists()

    with pytest.raises(ValueError, match="no dataframe to save"):
        adf._dataframe = None
        adf.save(filename=test_file)

    loaded_adf = AnnotatedDataFrame.from_file(filename=test_file)
    assert loaded_adf.dataframe.collect().equals(polars_dataframe)
    assert metadata == loaded_adf.metadata

    # Test the manipulation of metadata for import and export
    test_file = tmp_path / "test_dataframe_reload.hdf5"

    extra_column = "extra_column"
    loaded_adf.metadata.columns.append(DataSpecification(name=extra_column))
    from_column = loaded_adf.column_names[0]
    loaded_adf._dataframe = loaded_adf._dataframe.with_columns(
        polars.col(from_column).alias(extra_column)
    )
    loaded_adf.save(filename=test_file)

    # Check updated dataframe (with virtual column)
    loaded_adf = AnnotatedDataFrame.from_file(filename=test_file)
    polars.testing.assert_series_equal(
            loaded_adf.select(extra_column).collect().to_series(0),
            polars_dataframe[from_column],
            check_names=False)

    assert extra_column in loaded_adf.column_names
    assert extra_column in loaded_adf.metadata



def test_annotated_dataframe_export_csv(metadata, polars_dataframe, tmp_path):
    """
    Simple test of the annotated dataframe export to csv

    :param metadata: metadata to use
    :param polars_dataframe: polars dataframe to use
    :param tmp_path: where to temporarily save the data to csv
    """
    adf = AnnotatedDataFrame(dataframe=polars_dataframe,
                             metadata=metadata)

    test_file = tmp_path / "test_dataframe.csv"
    metadata_test_file = tmp_path / "test_dataframe.spec.yaml"
    adf.save(filename=test_file)
    assert test_file.exists()
    assert metadata_test_file.exists()

def test_annotated_dataframe_import_vaex_hdf5():
    """
    Simple test of the annotated dataframe import for HDF5
    """
    data_path = Path(__file__).parent.parent / "data"
    hdf5_path = data_path / "data.hdf5"

    adf = AnnotatedDataFrame.from_file(hdf5_path)
    assert adf.column_names == ["height", "letter"]

    pandas_df = pd.DataFrame(data = {'height': [0,1,2], 'letter': ['a','b','c']})
    assert adf.dataframe.collect().to_pandas().equals(pandas_df)

    assert adf._metadata.annotations["license"] == Annotation(name="license", value="MIT License")
    assert adf._metadata.annotations["comment"] == Annotation(name="comment", value="test dataframe")
    assert adf._metadata.columns[0] == DataSpecification(
        name="height", abbreviation="height", category=DataCategory.STATIC, unit=units.m,
        value_range=MinMax(min=0, max=40))


def test_annotated_dataframe_import_csv():
    """
    Simple test of the annotated dataframe import for csv
    """
    data_path = Path(__file__).parent.parent / "data"
    csv_path = data_path / "test_dataframe.csv"

    adf = AnnotatedDataFrame.from_file(csv_path)
    assert adf.column_names == ["height", "letter"]
    assert XDataFrame(adf._dataframe).equals(XDataFrame(polars.scan_csv(csv_path)))
    assert adf._metadata.annotations["license"] == Annotation(name="license", value="MIT License")
    assert adf._metadata.annotations["comment"] == Annotation(name="comment", value="test dataframe")
    assert adf._metadata.columns[0] == DataSpecification(
        name="height", abbreviation="height", category=DataCategory.STATIC,
        unit=units.m, value_range=MinMax(min=0, max=40), representation_type=int)

def test_01_dataframe_composition():
    """
    Test the dataframe composition, i.e. metadata in combination with an actual dataframe
    """
    data_path = Path(__file__).parent.parent / "data" / "01_dataframe_composition"
    csv_path = data_path / "data.csv"
    spec_path = data_path / "dataspec.yaml"

    md = MetaData.load_yaml(filename=spec_path)

    df = polars.scan_csv(source=csv_path)

    md.apply(df=df)

    # If the units are given in the spec, ensure that the dataframe is augmented
    #assert df.units["fullname-a"] == astropy.units.s
    #assert df.units["fullname-b"] == astropy.units.m

    adf = AnnotatedDataFrame(dataframe=df,
                             metadata=md)

    assert adf._metadata == md
    assert XDataFrame(adf._dataframe).equals(XDataFrame(df))

    assert adf.column_names == df.compat.column_names

    md.columns[0].value_range = MinMax(min=0, max=1)
    with pytest.raises(ValueError, match="lies outside of range"):
        md.apply(df=df)

    with pytest.raises(ValueError, match="lies outside of range"):
        AnnotatedDataFrame(dataframe=df, metadata=md)


def test_force_range():
    mmsi = np.array([0, -1, 2, 3, 8, 12, 52, 40, 18], dtype=np.int64)
    column_a = DataSpecification(
                name="mmsi",
                is_optional=False,
                representation_type=np.int64,
                missing_value=None,
                value_range=MinMax(min=0, max=40)
            )

    df = polars.LazyFrame(mmsi, ["mmsi"])
    df_filtered = df.filter(
            (polars.col("mmsi") >= 0) & (polars.col("mmsi") <= 40)
    )
    metadata = MetaData([column_a])
    adf = AnnotatedDataFrame(df,
            metadata=metadata,
            validation_mode=ValidationMode.UPDATE_DATA
          )

    assert XDataFrame(adf._dataframe).equals(XDataFrame(df_filtered))

def test_convert_csv_to_adf(tmp_path):
    output_filename =  tmp_path / "test-convert-csv.pq" #hdf5"
    test_path = Path(__file__).parent.parent / "data"
    AnnotatedDataFrame.convert_csv_to_adf(csv_filenames=[test_path / "test_dataframe.csv"],
                                          metadata_filename=test_path / "test_dataframe.spec.yaml",
                                          output_filename=output_filename,
                                          csv_sep=",")

    adf = AnnotatedDataFrame.from_file(filename=output_filename)
    assert "height" in adf.column_names
    assert "letter" in adf.column_names

    columns_in_metadata = [ x.name for x in adf.metadata.columns]

    assert "height" in columns_in_metadata
    assert "letter" in columns_in_metadata
