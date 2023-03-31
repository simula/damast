from damast.domains.maritime.ais.data_generator import AISTestData, AISTestDataSpec
from damast.domains.maritime.data_specification import ColumnName
import damast.data_handling.transformers.filters
import damast.core
from pathlib import Path
import pytest
import numpy as np


@pytest.fixture()
def adf() -> damast.core.AnnotatedDataFrame:
    test_data = AISTestData(number_of_trajectories=10, min_length=25, max_length=200)
    return damast.core.AnnotatedDataFrame(test_data.dataframe, damast.core.MetaData.from_dict(AISTestDataSpec))


@pytest.mark.parametrize("inplace", [True, False])
def test_timestamp(tmpdir, adf: damast.core.AnnotatedDataFrame, inplace: bool):
    """
    Test if time-stamp is sensible to work by
    """
    pipeline = damast.core.DataProcessingPipeline("test removal of source", Path(tmpdir))
    pipeline.add("Add time stamp", damast.data_handling.transformers.AddTimestamp(inplace),
                 name_mappings={"from": ColumnName.DATE_TIME_UTC, "to": ColumnName.TIMESTAMP})
    adf_copy = adf.dataframe.copy()
    new_adf = pipeline.transform(adf)

    # Drop converted nan values
    subset = new_adf.dataframe.dropnan([ColumnName.TIMESTAMP])

    # Drop missing values from input
    ref_subset = adf_copy.dropmissing([ColumnName.DATE_TIME_UTC])

    ref_lat_sorted = ref_subset.sort([ColumnName.DATE_TIME_UTC])[ColumnName.LATITUDE.lower()].evaluate()
    lat_sorted = subset.sort([ColumnName.TIMESTAMP])[ColumnName.LATITUDE.lower()].evaluate()
    assert np.allclose(ref_lat_sorted, lat_sorted)
    if inplace:
        assert len(adf.dataframe.dropnan([ColumnName.TIMESTAMP])) == len(subset)
    else:
        assert not (ColumnName.TIMESTAMP in adf.dataframe.column_names)
