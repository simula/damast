import datetime as dt

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline

from damast.data_handling.transformers.filters import (
    AreaFilter,
    DuplicateNeighboursFilter,
    MMSIFilter
)
from damast.domains.maritime.data_specification import MMSI, ColumnName


@pytest.fixture
def lat_long_dataframe():
    """
    A simple data frame with latitude longitude
    :return:
    """
    data = [[-180.0, -90.0], [-181.0, -91.0]]
    column_names = ["longitude", "latitude"]

    return pd.DataFrame(np.array(data), columns=column_names)


@pytest.fixture
def dataframe_with_duplicates():
    """
    A simple data frame with
    :return:
    """
    data = [[-180.0, -90.0], [-180.0, -90.0]]
    column_names = ["a", "b"]

    return pd.DataFrame(np.array(data), columns=column_names)


@pytest.fixture
def mmsi_dataframe():
    """
    This mmsi dataframe contains only one valid MMSI
    """
    utcnow = dt.datetime.utcnow()
    utcnow + dt.timedelta(seconds=1)

    out_of_lower_range_mmsi = MMSI.min_value - 1
    out_of_upper_range_mmsi = MMSI.max_value + 1
    valid_mmsi = MMSI.min_value

    data = [[out_of_lower_range_mmsi, utcnow],
            [valid_mmsi, utcnow + dt.timedelta(seconds=1)],
            [valid_mmsi, utcnow + dt.timedelta(seconds=1)],
            [valid_mmsi, utcnow + dt.timedelta(seconds=2)],
            [out_of_upper_range_mmsi, utcnow + dt.timedelta(seconds=2)],
            ]
    column_names = [ColumnName.MMSI, ColumnName.TIMESTAMP]

    return pd.DataFrame(np.array(data), columns=column_names)


def test_filter_areas(lat_long_dataframe):
    assert lat_long_dataframe[lat_long_dataframe["latitude"] < -90.0].shape[0] > 0
    assert lat_long_dataframe[lat_long_dataframe["longitude"] < -180.0].shape[0] > 0

    area_filter = AreaFilter(latitude_name="latitude",
                             longitude_name="longitude")

    area_filter.fit(lat_long_dataframe)
    filtered_df = area_filter.fit_transform(lat_long_dataframe)

    assert filtered_df[filtered_df["latitude"] < -90.0].shape[0] == 0
    assert filtered_df[filtered_df["longitude"] < -180.0].shape[0] == 0


def test_filter_areas_pipeline(lat_long_dataframe):
    assert lat_long_dataframe[lat_long_dataframe["latitude"] < -90.0].shape[0] > 0

    simple_pipeline = Pipeline([
        ("area_filter", AreaFilter(latitude_name="latitude",
                                   longitude_name="longitude"))
    ])

    filtered_df = simple_pipeline.fit_transform(lat_long_dataframe)
    assert filtered_df[filtered_df["latitude"] < -90.0].shape[0] == 0
    assert filtered_df[filtered_df["longitude"] < -180.0].shape[0] == 0

    # Displaying the pipeline, run: pytest -s ...
    sklearn.set_config(display="text")
    print(simple_pipeline)


def test_filter_duplicates(dataframe_with_duplicates):
    duplicates_filter = DuplicateNeighboursFilter(column_names=["a", "b"])
    filtered_df = duplicates_filter.fit_transform(dataframe_with_duplicates)

    assert filtered_df.shape[0] == 1


def test_mmsi_filter(mmsi_dataframe):
    """
    Test if the MMSIFilter is operating correctly and produces only unique columns
    :param mmsi_dataframe: fixture for the mmsi data
    :return:
    """
    mmsi_filter = MMSIFilter()
    filtered_df = mmsi_filter.fit_transform(mmsi_dataframe)

    assert filtered_df[filtered_df[ColumnName.MMSI] < MMSI.min_value].shape[0] == 0
    assert filtered_df[filtered_df[ColumnName.MMSI] > MMSI.min_value].shape[0] == 0

    number_of_rows = filtered_df[ColumnName.MMSI].shape[0]
    number_of_unique_timestamps = np.unique(filtered_df[ColumnName.TIMESTAMP].values).shape[0]

    assert number_of_unique_timestamps == number_of_rows
