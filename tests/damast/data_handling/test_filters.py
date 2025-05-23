from pathlib import Path

import polars as pl
import pytest

import damast.core
import damast.data_handling.transformers.filters
from damast.domains.maritime.ais.data_generator import AISTestData, AISTestDataSpec
from damast.domains.maritime.data_specification import ColumnName


@pytest.fixture()
def adf() -> damast.core.AnnotatedDataFrame:
    test_data = AISTestData(number_of_trajectories=10, min_length=25, max_length=200)
    return damast.core.AnnotatedDataFrame(test_data.dataframe, damast.core.MetaData.from_dict(AISTestDataSpec))


@pytest.mark.parametrize("inplace", [True, False])
def test_remove_values(tmpdir, adf: damast.core.AnnotatedDataFrame, inplace: bool):
    """
    Test that removal of sources work on test data
    """
    pipeline = damast.core.DataProcessingPipeline(
        name="test removal of source",
        base_dir=Path(tmpdir),
        inplace_transformation=inplace
    )

    pipeline.add("Remove rows with ground as source", damast.data_handling.transformers.filters.RemoveValueRows("g"),
                 name_mappings={"x": ColumnName.SOURCE})

    original_sources = adf[ColumnName.SOURCE].collect()
    num_sources = len(original_sources)
    num_invalid_sources = len(original_sources.filter(pl.col(ColumnName.SOURCE) == "g"))
    num_valid_sources = num_sources - num_invalid_sources

    new_adf = pipeline.transform(adf)

    filtered_sources = new_adf[ColumnName.SOURCE].collect()
    assert len(filtered_sources) == num_valid_sources
    assert len(filtered_sources.filter(pl.col(ColumnName.SOURCE) != "s")) == 0

    original_df_length = len(adf.filter(pl.col(ColumnName.SOURCE) == "g").collect())
    if inplace:
        # frame has been updated, so no invalid entries should be left
        assert original_df_length == 0
    else:
        # frame has been not been updated inplace, so invalid entries should be still left
        assert original_df_length == num_invalid_sources


@pytest.mark.parametrize("inplace", [True, False])
def test_drop_missing(tmpdir,  adf: damast.core.AnnotatedDataFrame, inplace: bool):

    pipeline = damast.core.DataProcessingPipeline(
        name="test removal of source",
        base_dir=Path(tmpdir),
        inplace_transformation=inplace
    )
    pipeline.add("Remove rows with ground as source",
                 damast.data_handling.transformers.filters.DropMissingOrNan(),
                 name_mappings={"x": ColumnName.DATE_TIME_UTC})


    num_missing = adf.dataframe.select(ColumnName.DATE_TIME_UTC).null_count().collect()[0,0]

    assert num_missing > 0
    new_adf = pipeline.transform(adf)

    new_num_missing = new_adf.dataframe.select(ColumnName.DATE_TIME_UTC).null_count().collect()[0,0]

    assert new_num_missing == 0
    num_missing_post = adf.dataframe.select(ColumnName.DATE_TIME_UTC).null_count().collect()[0,0]

    if inplace:
        assert num_missing_post == 0
    else:
        assert num_missing == num_missing_post


@pytest.mark.parametrize("inplace", [True, False])
def test_filter_within(tmpdir,  adf: damast.core.AnnotatedDataFrame, inplace: bool):

    pipeline = damast.core.DataProcessingPipeline(
        name="test removal of source",
        base_dir=Path(tmpdir),
        inplace_transformation=inplace
    )
    unique_values = adf.select("message_nr").unique().collect().to_numpy().flatten()
    assert len(unique_values) > 1

    pipeline.add("Filter rows within message types",
                 damast.data_handling.transformers.filters.FilterWithin(unique_values[:1]),
                 name_mappings={"x": "message_nr"})

    num_all = len(adf.dataframe)
    num_eq = len(adf.filter(pl.col("message_nr") == unique_values[[0]]).collect())
    new_adf = pipeline.transform(adf)

    assert len(new_adf.dataframe) == num_eq
    assert len(new_adf.filter(pl.col("message_nr") == unique_values[[0]]).collect()) == num_eq
    if inplace:
        assert len(adf.dataframe.collect()) == len(new_adf.dataframe.collect())
    else:
        assert len(adf.dataframe.collect()) == num_all
