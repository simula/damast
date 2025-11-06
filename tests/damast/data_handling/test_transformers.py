from pathlib import Path

import pytest

import damast.core
import damast.data_handling.transformers.filters
from damast.core.types import XDataFrame
from damast.domains.maritime.ais.data_generator import AISTestData, AISTestDataSpec
from damast.domains.maritime.data_specification import ColumnName


@pytest.fixture()
def adf() -> damast.core.AnnotatedDataFrame:
    test_data = AISTestData(number_of_trajectories=10, min_length=25, max_length=200)
    return damast.core.AnnotatedDataFrame(test_data.dataframe, damast.core.MetaData.from_dict(AISTestDataSpec))


@pytest.mark.parametrize("inplace", [True, False])
def test_timestamp(tmpdir, adf: damast.core.AnnotatedDataFrame, inplace: bool):
    """
    Test if time-stamp is sensible to work by
    """
    pipeline = damast.core.DataProcessingPipeline(
        name="test removal of source",
        base_dir=Path(tmpdir),
        inplace_transformation=inplace
    )
    pipeline.add("Add time stamp", damast.data_handling.transformers.AddTimestamp(),
                 name_mappings={"from": ColumnName.DATE_TIME_UTC, "to": ColumnName.TIMESTAMP})

    adf_copy = adf.clone()
    new_adf = pipeline.transform(adf)

    # Drop remaining null values after conversion
    subset = new_adf.drop_nulls(subset=[ColumnName.TIMESTAMP])

    # Drop missing values from input
    ref_subset = adf_copy.drop_nulls(subset=[ColumnName.DATE_TIME_UTC])

    ref_lat_sorted = ref_subset.sort(ColumnName.DATE_TIME_UTC).select(ColumnName.LATITUDE.lower()).collect()
    lat_sorted = subset.sort(ColumnName.TIMESTAMP).select(ColumnName.LATITUDE.lower()).collect()

    assert XDataFrame(ref_lat_sorted).equals(XDataFrame(lat_sorted))

    if inplace:
        assert len(adf.dataframe.drop_nulls([ColumnName.TIMESTAMP]).collect()) == len(subset.collect())
    else:
        assert ColumnName.TIMESTAMP not in adf.column_names
