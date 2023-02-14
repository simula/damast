import pandas as pd
import pytest
import vaex
from astropy import units
from vaex.ml import CycleTransformer

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.datarange import CyclicMinMax, MinMax
from damast.core.metadata import DataSpecification, DataCategory, MetaData


@pytest.fixture()
def height_metadata():
    column_spec = DataSpecification(name="height",
                                    category=DataCategory.STATIC,
                                    unit=units.m)

    metadata = MetaData(columns=[column_spec])
    return metadata


@pytest.fixture()
def height_dataframe():
    data = [
        [0, "a"],
        [1, "b"],
        [2, "c"]
    ]
    columns = [
        "height", "letter"
    ]
    pandas_df = pd.DataFrame(data, columns=columns)
    return vaex.from_pandas(pandas_df)


@pytest.fixture()
def lat_lon_metadata():
    lat_column_spec = DataSpecification(name="latitude",
                                        category=DataCategory.DYNAMIC,
                                        unit=units.deg,
                                        value_range=CyclicMinMax(-90.0, 90.0))
    lon_column_spec = DataSpecification(name="longitude",
                                        category=DataCategory.DYNAMIC,
                                        unit=units.deg,
                                        value_range=CyclicMinMax(-180.0, 180.0))

    metadata = MetaData(columns=[lat_column_spec, lon_column_spec])
    return metadata


@pytest.fixture()
def lat_lon_dataframe():
    data = [
        [-90.0, -180.0],
        [0.0, 0.0],
        [90.0, 180.0]
    ]
    columns = [
        "latitude", "longitude"
    ]
    pandas_df = pd.DataFrame(data, columns=columns)
    return vaex.from_pandas(pandas_df)


def test_data_processor_input(height_dataframe, height_metadata):
    height_adf = AnnotatedDataFrame(dataframe=height_dataframe,
                                    metadata=height_metadata)

    class CustomDataProcessor:
        # Consider:
        # - mapping of input names
        # - use regex to match columns
        #
        @damast.core.input(
            [
                {"longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)}},
                {"latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}}
            ]
        )
        @damast.core.output(
            [
                {"longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"latitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"latitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)}}
            ]
        )
        def apply_lat_lon(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            return df

        @damast.core.input(
            [
                {"height": {"unit": units.m}}
            ]
        )
        @damast.core.output(
            [{"height": {"unit": units.km}}]
        )
        def apply_height(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            return df

    cdp = CustomDataProcessor()
    with pytest.raises(RuntimeError, match="Input requirements are not fulfilled"):
        cdp.apply_lat_lon(df=height_adf)

    cdp.apply_height(df=height_adf)


def test_data_processor_output(lat_lon_dataframe, lat_lon_metadata):
    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

    class CustomDataProcessor:
        # Consider:
        # - mapping of input names
        # - use regex to match columns
        #
        @damast.core.input(
            [
                {"longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)}},
                {"latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}}
            ]
        )
        @damast.core.output(
            [
                {"longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"latitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"latitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)}}
            ]
        )
        def apply_lat_lon(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            transformer = CycleTransformer(features=["latitude", "longitude"])
            df._dataframe = transformer.fit_transform(df._dataframe)
            return df

        @damast.core.input(
            [
                {"longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)}},
                {"latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}}
            ]
        )
        @damast.core.output(
            [
                {"longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"latitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)}},
                {"latitude_xasdf": {"unit": None, "value_range": MinMax(0.0, 1.0)}}
            ]
        )
        def apply_lat_lon_fail(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            transformer = CycleTransformer(features=["latitude", "longitude"])
            df._dataframe = transformer.fit_transform(df._dataframe)
            return df

    cdp = CustomDataProcessor()
    adf = cdp.apply_lat_lon(df=adf)
    assert "latitude_x" in adf.column_names

    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

    with pytest.raises(RuntimeError, match="is not present"):
        cdp.apply_lat_lon_fail(df=adf)
