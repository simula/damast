import pandas as pd
import pytest
import vaex
from astropy import units
from vaex.ml import CycleTransformer

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement
from damast.core.datarange import CyclicMinMax, MinMax
from damast.core.metadata import DataCategory, DataSpecification, MetaData


class DataProcessorA:
    # Consider:
    # - mapping of input names
    # - use regex to match columns
    #
    @damast.core.input({
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)},
    })
    @damast.core.output({
        "longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)},
        "latitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)},
        "longitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)},
        "latitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)}
    })
    def apply_lat_lon(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        transformer = CycleTransformer(features=["latitude", "longitude"])
        df._dataframe = transformer.fit_transform(df._dataframe)
        return df

    @damast.core.input({
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}
    })
    @damast.core.output({
        "longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)},
        "latitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)},
        "longitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)},
        "latitude_xy": {"unit": None, "value_range": MinMax(0.0, 1.0)}
    })
    def apply_lat_lon_fail(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        transformer = CycleTransformer(features=["latitude", "longitude"])
        df._dataframe = transformer.fit_transform(df._dataframe)
        return df

    @damast.core.input({
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}
    })
    @damast.core.output({
        "longitude": {"unit": units.deg}
    })
    def apply_lat_lon_remove_col(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        df._dataframe.drop(columns=["latitude"], inplace=True)
        return df


class TransformerA(PipelineElement):
    @damast.core.describe("latitude x/y generation")
    @damast.core.input({
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}
    })
    @damast.core.output({
        "latitude_x": {"unit": units.deg},
        "latitude_y": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class TransformerB(PipelineElement):
    @damast.core.describe("delta computation")
    @damast.core.input({
        "latitude_x": {"unit": units.deg},
        "latitude_y": {"unit": units.deg}
    })
    @damast.core.output({
        "delta_longitude": {"unit": units.deg},
        "delta_latitude": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


class TransformerC(PipelineElement):
    @damast.core.describe("label generation")
    @damast.core.input({
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)},
        "delta_longitude": {"unit": units.deg},
        "delta_latitude": {"unit": units.deg}
    })
    @damast.core.output({
        "label": {}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df


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
        @damast.core.input({
            "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
            "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}
        })
        @damast.core.output({
            "longitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)},
            "latitude_x": {"unit": None, "value_range": MinMax(0.0, 1.0)},
            "longitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)},
            "latitude_y": {"unit": None, "value_range": MinMax(0.0, 1.0)}
        })
        def apply_lat_lon(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            return df

        @damast.core.input({"height": {"unit": units.m}})
        @damast.core.output({"height": {"unit": units.km}})
        def apply_height(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            return df

    cdp = CustomDataProcessor()
    with pytest.raises(RuntimeError, match="Input requirements are not fulfilled"):
        cdp.apply_lat_lon(df=height_adf)

    cdp.apply_height(df=height_adf)


def test_data_processor_output(lat_lon_dataframe, lat_lon_metadata):
    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

    cdp = DataProcessorA()
    adf = cdp.apply_lat_lon(df=adf)
    assert "latitude_x" in adf.column_names


def test_data_processor_output_fail(lat_lon_dataframe, lat_lon_metadata):
    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

    cdp = DataProcessorA()
    with pytest.raises(RuntimeError, match="is not present"):
        cdp.apply_lat_lon_fail(df=adf)

    with pytest.raises(RuntimeError, match="was removed by"):
        cdp.apply_lat_lon_remove_col(df=adf)


def test_access_decorator_info():
    input_specs = {
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}
    }
    output_specs = {"longitude": {"unit": units.deg}}

    class CustomDataProcessor:
        @damast.core.input(input_specs)
        @damast.core.output(output_specs)
        def apply_lat_lon_remove_col(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            return df

    assert getattr(CustomDataProcessor.apply_lat_lon_remove_col, damast.core.DECORATED_INPUT_SPECS) == \
           DataSpecification.from_requirements(requirements=input_specs)

    assert getattr(CustomDataProcessor.apply_lat_lon_remove_col, damast.core.DECORATED_OUTPUT_SPECS) == \
           DataSpecification.from_requirements(requirements=output_specs)


def test_data_processing_valid_pipeline(tmp_path):
    pipeline = DataProcessingPipeline(name="abc", base_dir=tmp_path) \
        .add("transform-a", TransformerA()) \
        .add("transform-b", TransformerB()) \
        .add("transform-c", TransformerC())

    with pytest.raises(RuntimeError, match="set the correct output specs"):
        pipeline.output_specs

    pipeline.prepare()
    assert pipeline.output_specs is not None

    output_columns = [x.name for x in pipeline.output_specs]
    for column in ["longitude", "latitude",
                   "latitude_x", "latitude_y",
                   "delta_longitude", "delta_latitude",
                   "label"]:
        assert column in output_columns

    representation = pipeline.to_str(indent_level=0)
    print("\n")
    print(representation)
    assert representation != ""


def test_data_processing_invalid_pipeline(tmp_path):
    pipeline = DataProcessingPipeline(name="acb", base_dir=tmp_path) \
        .add("transform-a", TransformerA()) \
        .add("transform-c", TransformerC()) \
        .add("transform-b", TransformerB())
    with pytest.raises(RuntimeError, match="insufficient output declared"):
        pipeline.prepare()
