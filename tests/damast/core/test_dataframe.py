import copy
from pathlib import Path

import astropy.units as units
import numpy as np
import pandas as pd
import polars
import polars.testing
import pytest

from damast.core.annotations import Annotation
from damast.core.data_description import ListOfValues, MinMax
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.metadata import (
    DataCategory,
    DataSpecification,
    MetaData,
    ValidationMode,
)
from damast.core.types import XDataFrame


@pytest.fixture()
def metadata():
    column_spec_height = DataSpecification(name="height",
                                    category=DataCategory.STATIC,
                                    unit=units.m,
                                    abbreviation="height",
                                    value_range=MinMax(min=0, max=40))

    license = Annotation(name=Annotation.Key.License, value="MIT License")
    comment = Annotation(name=Annotation.Key.Comment, value="test dataframe")
    annotations = [license, comment]

    column_spec_letter = DataSpecification(name="letter",
                                    category=DataCategory.STATIC,
                                    abbreviation="letter")

    annotations = [license, comment]
    return MetaData(columns=[column_spec_height, column_spec_letter], annotations=annotations)


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

def test_incomplete_metadata(metadata, polars_dataframe, tmp_path):
    """
    Simple test of the annotated dataframe export to HDF5

    :param metadata: metadata to use
    :param polars_dataframe: polars dataframe to use
    :param tmp_path: where to temporarily save the data to HDF5
    """
    incomplete_metadata = MetaData(columns=metadata.columns[:-1])
    with pytest.raises(ValueError, match=f"missing column metadata for columns: {metadata.columns[-1].name}"):
         AnnotatedDataFrame(dataframe=polars_dataframe,
                            metadata=incomplete_metadata)

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
        adf.lazyframe = None
        adf.save(filename=test_file)

    loaded_adf = AnnotatedDataFrame.from_file(filename=test_file)
    assert loaded_adf.dataframe.collect().equals(polars_dataframe)
    assert metadata == loaded_adf.metadata

    # Test the manipulation of metadata for import and export
    test_file = tmp_path / "test_dataframe_reload.hdf5"

    extra_column = "extra_column"
    loaded_adf.metadata.columns.append(DataSpecification(name=extra_column))
    from_column = loaded_adf.column_names[0]
    loaded_adf.lazyframe = loaded_adf.lazyframe.with_columns(
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

def test_annotated_dataframe_import_vaex_hdf5(data_path):
    """
    Simple test of the annotated dataframe import for HDF5
    """
    hdf5_path = data_path / "data.hdf5"

    with pytest.raises(ValueError, match="missing column metadata"):
        AnnotatedDataFrame.from_file(hdf5_path)

    adf = AnnotatedDataFrame.from_file(hdf5_path, metadata_required=False, validation_mode=ValidationMode.IGNORE)
    assert adf.column_names == ["height", "letter"]

    pandas_df = pd.DataFrame(data = {'height': [0,1,2], 'letter': ['a','b','c']})
    assert adf.dataframe.collect().to_pandas().equals(pandas_df)

    assert adf._metadata.annotations["license"] == Annotation(name="license", value="MIT License")
    assert adf._metadata.annotations["comment"] == Annotation(name="comment", value="test dataframe")
    assert adf._metadata.columns[0] == DataSpecification(
        name="height", abbreviation="height", category=DataCategory.STATIC, unit=units.m,
        value_range=MinMax(min=0, max=40))


def test_annotated_dataframe_import_csv(data_path):
    """
    Simple test of the annotated dataframe import for csv
    """
    csv_path = data_path / "test_dataframe.csv"

    adf = AnnotatedDataFrame.from_file(csv_path)
    assert adf.column_names == ["height", "letter"]
    assert adf.dtype('height') == polars.Int64, "None types should be properly handled"

    assert XDataFrame(adf.lazyframe).equals(XDataFrame(polars.scan_csv(csv_path, null_values=["None", "none"])))
    assert adf._metadata.annotations["license"] == Annotation(name="license", value="MIT License")
    assert adf._metadata.annotations["comment"] == Annotation(name="comment", value="test dataframe")
    assert adf._metadata.columns[0] == DataSpecification(
        name="height", abbreviation="height", category=DataCategory.STATIC,
        unit=units.m, value_range=MinMax(min=0, max=40), representation_type=int)

def test_annotated_dataframe_import_csv_with_quotes(data_path):
    """
    Simple test of the annotated dataframe import for csv
    """
    csv_path = data_path / "test_dataframe_with_quotes.csv"

    adf = AnnotatedDataFrame.from_file(csv_path)
    assert adf.column_names == ["id", "name"]
    assert adf.dtype('id') == polars.Int64
    assert adf.dtype('name') == polars.String

    assert XDataFrame(adf.lazyframe).equals(XDataFrame(polars.scan_csv(csv_path, null_values=["None", "none"])))

    df = adf.dataframe.collect()
    assert df[0,1] == "a,b;c"
    assert df[1,1] == "d;e,f"

def test_set_dtype(data_path):
    """
    Test if conversion from int -> str in representation_type is consistent
    """
    csv_path = data_path / "test_dataframe.csv"
    adf = AnnotatedDataFrame.from_file(csv_path)

    assert adf.dtype('height') != polars.String
    assert polars.String == adf.set_dtype('height', polars.String)

    assert adf.dtype('height') == polars.String


def test_01_dataframe_composition(data_path):
    """
    Test the dataframe composition, i.e. metadata in combination with an actual dataframe
    """
    data_path = data_path / "01_dataframe_composition"
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
    assert XDataFrame(adf.lazyframe).equals(XDataFrame(df))

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
                value_range=MinMax(min=0, max=40, allow_missing=False)
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

    assert XDataFrame(adf.lazyframe).equals(XDataFrame(df_filtered))

def test_force_range_allow_missing():
    mmsi = polars.DataFrame({'mmsi': [0, -1, 2, 3, 8, 12, 52, 40, 18, None]})
    column_a = DataSpecification(
                name="mmsi",
                is_optional=False,
                representation_type=np.int64,
                missing_value=None,
                value_range=MinMax(min=0, max=40, allow_missing=True)
            )

    df = mmsi.lazy()
    df_filtered = df.filter(
            ((polars.col("mmsi") >= 0) & (polars.col("mmsi") <= 40)) | polars.col("mmsi").is_null()
    )
    metadata = MetaData([column_a])
    adf = AnnotatedDataFrame(df,
            metadata=metadata,
            validation_mode=ValidationMode.UPDATE_DATA
          )

    assert XDataFrame(adf.lazyframe).equals(XDataFrame(df_filtered))


def test_list_of_values_rejects_invalid_category_not_at_extremes():
    """
    Regression: READONLY validation of a ListOfValues range used to reuse the MinMax
    check, i.e. only compare the lexicographically smallest/largest value against the
    allowed list - an invalid category sorting between two valid ones silently passed.
    """
    df = polars.DataFrame({"category": ["a", "x", "c"]}).lazy()
    column_spec = DataSpecification(name="category", value_range=ListOfValues(["a", "b", "c"]))
    metadata = MetaData([column_spec])

    with pytest.raises(ValueError, match="lie outside"):
        metadata.apply(df=df, validation_mode=ValidationMode.READONLY)


def test_list_of_values_accepts_all_valid_categories():
    df = polars.DataFrame({"category": ["a", "b", "c", "a"]}).lazy()
    column_spec = DataSpecification(name="category", value_range=ListOfValues(["a", "b", "c"]))
    metadata = MetaData([column_spec])

    # Should not raise
    metadata.apply(df=df, validation_mode=ValidationMode.READONLY)


def test_list_of_values_allows_null_when_listed():
    df = polars.DataFrame({"category": ["a", None, "b"]}).lazy()
    column_spec = DataSpecification(name="category", value_range=ListOfValues(["a", "b", None]))
    metadata = MetaData([column_spec])

    # Should not raise
    metadata.apply(df=df, validation_mode=ValidationMode.READONLY)


def test_readonly_validation_batches_minmax_into_single_collect(monkeypatch):
    """
    Regression: MetaData.apply() used to call DataSpecification.apply() once per
    column, and each call independently collect()-ed its own min/max - N columns
    meant N separate query executions. Validation should now precompute all MinMax
    aggregates in a single collect() up front.
    """
    df = polars.DataFrame({
        "a": [0, 1, 2],
        "b": [10, 11, 12],
        "c": [100, 101, 102],
    }).lazy()

    column_specs = [
        DataSpecification(name="a", value_range=MinMax(min=0, max=2)),
        DataSpecification(name="b", value_range=MinMax(min=10, max=12)),
        DataSpecification(name="c", value_range=MinMax(min=100, max=102)),
    ]
    metadata = MetaData(column_specs)

    original_collect = polars.LazyFrame.collect
    call_count = 0

    def counting_collect(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_collect(self, *args, **kwargs)

    monkeypatch.setattr(polars.LazyFrame, "collect", counting_collect)

    metadata.apply(df=df, validation_mode=ValidationMode.READONLY)

    assert call_count == 1


def test_xdataframe_minmax_cache_invalidated_on_lazyframe_reassignment():
    xdf = XDataFrame(polars.DataFrame({"a": [0, 1, 2], "b": [10, 11, 12]}).lazy())

    xdf.precompute_minmax(["a", "b"])
    assert xdf.minmax("a") == (0, 2)
    assert xdf.minmax("b") == (10, 12)

    # Reassigning the lazyframe must drop the stale cache
    xdf.lazyframe = polars.DataFrame({"a": [5, 6, 7], "b": [10, 11, 12]}).lazy()
    assert xdf.minmax("a") == (5, 7)


def test_convert_csv_to_adf(tmp_path):
    output_filename =  tmp_path / "test-convert-csv.pq"
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


def test_update_refreshes_existing_column_without_declared_range():
    """
    Regression test: AnnotatedDataFrame.update() used to only add a DataSpecification for a
    column that was not yet part of the metadata - if the column already existed (e.g. because
    it was loaded from a previously exported file, and a pipeline step recomputes/overwrites it),
    the pre-existing spec - including a stale value_range - was left untouched. A pipeline step
    that legitimately produces values outside that stale range (e.g. AddDeltaTime producing 0.0
    for the first ping of a group, when the loaded file's inferred range started at 1.0) would
    then fail the READONLY validate_metadata() check run right after update(), even though the
    step's own declared output spec makes no claim about the value range at all.
    """
    height = np.array([-5, 0, 10, 100], dtype=np.int64)
    df = polars.LazyFrame({"height": height})

    # Simulate a column that already has a (now stale) value_range attached, e.g. inherited from
    # a previously exported file.
    stale_spec = DataSpecification(
        name="height",
        representation_type=int,
        value_range=MinMax(min=0, max=40),
    )
    adf = AnnotatedDataFrame(df, metadata=MetaData([stale_spec]), validation_mode=ValidationMode.IGNORE)

    # The step producing 'height' does not itself declare any value_range constraint - matching
    # how PipelineElement subclasses declare their outputs in practice, e.g.
    # @damast.core.output({"height": {"representation_type": int}})
    step_output_spec = DataSpecification(name="height", representation_type=int)
    adf.update(expectations=[step_output_spec])

    assert adf.metadata["height"].value_range is None
    # Must not raise, even though actual data (-5, 100) lies outside the old MinMax(0, 40).
    adf.validate_metadata()


def test_update_replaces_existing_column_with_newly_declared_range():
    """
    Companion to test_update_refreshes_existing_column_without_declared_range: when the current
    pipeline step *does* declare its own value_range for a column that already existed in the
    metadata, update() must use the step's newly declared range (its own guarantee about what it
    produces) rather than keep the old spec's range or silently drop the check entirely.
    """
    x = np.array([-0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    df = polars.LazyFrame({"x": x})

    old_spec = DataSpecification(
        name="x",
        representation_type=float,
        value_range=MinMax(min=0, max=40),
    )
    adf = AnnotatedDataFrame(df, metadata=MetaData([old_spec]), validation_mode=ValidationMode.IGNORE)

    new_spec = DataSpecification(
        name="x",
        representation_type=float,
        value_range=MinMax(min=-1.0, max=1.0),
    )
    adf.update(expectations=[new_spec])

    assert adf.metadata["x"].value_range == MinMax(min=-1.0, max=1.0)
    adf.validate_metadata()


def test_update_preserves_representation_type_when_step_output_declares_none():
    """
    Regression test: fixing the stale value_range/value_stats issue above (by replacing the
    existing spec with the step's own declared output spec) went too far and also reset
    representation_type/description/unit/etc. to None whenever a step's declared output for an
    already-existing column was empty or partial - e.g. a pass-through filter like
    @damast.core.output({"x": {}}) on a column that already had representation_type=str set.
    That silently wiped type information a later pipeline step's @input check could depend on.
    Structural fields must carry over from the existing spec when the step doesn't declare them,
    while value_range/value_stats still reset (see test above).
    """
    df = polars.LazyFrame({"date_time_utc": ["2020-01-01", "2020-01-02"]})

    existing_spec = DataSpecification(
        name="date_time_utc",
        representation_type=str,
        description="original description",
        unit=units.deg,
    )
    adf = AnnotatedDataFrame(df, metadata=MetaData([existing_spec]), validation_mode=ValidationMode.IGNORE)

    # A pass-through step (e.g. DropMissingOrNan/FilterWithin) declares no constraints at all -
    # @damast.core.output({"x": {}}) - for a column that already exists in the metadata.
    step_output_spec = DataSpecification(name="date_time_utc")
    adf.update(expectations=[step_output_spec])

    assert adf.metadata["date_time_utc"].representation_type is str
    assert adf.metadata["date_time_utc"].description == "original description"
    assert adf.metadata["date_time_utc"].unit == units.deg
    adf.validate_metadata()
