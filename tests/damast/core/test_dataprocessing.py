import pandas as pd
import pytest
import vaex
from astropy import units
from vaex.ml import CycleTransformer

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement, DECORATED_INPUT_SPECS, \
    DECORATED_OUTPUT_SPECS
from damast.core.datarange import CyclicMinMax, MinMax
from damast.core.metadata import DataCategory, DataSpecification, MetaData


class DataProcessorA(PipelineElement):
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
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        transformer = CycleTransformer(features=["latitude", "longitude"])
        df._dataframe = transformer.fit_transform(df._dataframe)
        return df


class DataProcessorAFail(PipelineElement):
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
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        transformer = CycleTransformer(features=["latitude", "longitude"])
        df._dataframe = transformer.fit_transform(df._dataframe)
        return df


class DataProcessorARemoveCol(PipelineElement):
    @damast.core.input({
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}
    })
    @damast.core.output({
        "longitude": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
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
        "latitude_y": {"unit": units.deg},
        "longitude_x": {"unit": units.deg},
        "longitude_y": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat_cyclic_transformer = vaex.ml.CycleTransformer(features=["latitude"], n=180.0)
        lon_cyclic_transformer = vaex.ml.CycleTransformer(features=["longitude"], n=360.0)

        _df = lat_cyclic_transformer.fit_transform(df=df)
        _df = lon_cyclic_transformer.fit_transform(df=_df)
        df._dataframe = _df
        return df


class TransformerB(PipelineElement):
    @damast.core.describe("delta computation")
    @damast.core.input({
        "latitude_x": {"unit": units.deg},
        "latitude_y": {"unit": units.deg},
        "longitude_x": {"unit": units.deg},
        "longitude_y": {"unit": units.deg}
    })
    @damast.core.output({
        "delta_longitude": {"unit": units.deg},
        "delta_latitude": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        # This operation is does not really make sense, but acts as a placeholder to generate
        # the desired output columns
        df["delta_longitude"] = df["longitude_x"] - df["longitude_y"]
        df["delta_latitude"] = df["latitude_x"] - df["latitude_y"]
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
        df["label"] = "data-label"
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


@pytest.fixture()
def lat_lon_annotated_dataframe(lat_lon_dataframe, lat_lon_metadata):
    return AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                              metadata=lat_lon_metadata)


def test_data_processor_input(height_dataframe, height_metadata):
    height_adf = AnnotatedDataFrame(dataframe=height_dataframe,
                                    metadata=height_metadata)

    class ApplyLatLonProcessor(PipelineElement):
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
        def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            return df

    class ApplyHeight(PipelineElement):
        @damast.core.input({"height": {"unit": units.m}})
        @damast.core.output({"height": {"unit": units.km}})
        def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            return df

    lat_lon_processor = ApplyLatLonProcessor()
    with pytest.raises(RuntimeError, match="Input requirements are not fulfilled"):
        lat_lon_processor.transform(df=height_adf)

    height_processor = ApplyHeight()
    height_processor.transform(df=height_adf)


def test_data_processor_output(lat_lon_dataframe, lat_lon_metadata):
    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

    cdp = DataProcessorA()
    adf = cdp.transform(df=adf)
    assert "latitude_x" in adf.column_names


def test_data_processor_output_fail(lat_lon_dataframe, lat_lon_metadata):
    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

    a_fail = DataProcessorAFail()
    with pytest.raises(RuntimeError, match="is not present"):
        a_fail.transform(df=adf)

    a_remove_col = DataProcessorARemoveCol()
    with pytest.raises(RuntimeError, match="was removed by"):
        a_remove_col.transform(df=adf)


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


def test_data_processing_valid_pipeline(lat_lon_dataframe, lat_lon_metadata, tmp_path):
    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

    pipeline = DataProcessingPipeline(name="abc", base_dir=tmp_path) \
        .add("transform-a", TransformerA()) \
        .add("transform-b", TransformerB()) \
        .add("transform-c", TransformerC())

    with pytest.raises(RuntimeError, match="set the correct output specs"):
        pipeline.output_specs

    pipeline.prepare(df=adf)

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

    pipeline.transform(df=adf)


def test_data_processing_invalid_pipeline(lat_lon_annotated_dataframe, tmp_path):
    pipeline = DataProcessingPipeline(name="acb", base_dir=tmp_path) \
        .add("transform-a", TransformerA()) \
        .add("transform-c", TransformerC()) \
        .add("transform-b", TransformerB())
    with pytest.raises(RuntimeError, match="Input requirements are not fulfilled"):
        pipeline.prepare(df=lat_lon_annotated_dataframe)


def test_single_element_pipeline(tmp_path):
    data = [["10000000", 0]]
    column_names = ["mmsi", "status"]

    column_specs = [
        DataSpecification(name="mmsi"),
        DataSpecification(name="status", unit=units.deg)
    ]

    df_pd = pd.DataFrame(data, columns=column_names)
    df = vaex.from_pandas(df_pd)
    adf = AnnotatedDataFrame(df, MetaData(columns=column_specs))

    class TransformX(PipelineElement):

        @damast.core.describe("Generic transform of x")
        @damast.core.input({"x": {"unit": units.deg}})
        @damast.core.output({"{{x}}_suffix": {"unit": units.deg}})
        def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            df[f"{self.get_name('x')}_suffix"] = df[self.get_name('x')]
            return df

    pipeline = DataProcessingPipeline("TransformStatus", tmp_path)
    pipeline.add("Transform status",
                 TransformX(),
                 name_mappings={"x": "status"})

    pipeline.transform(df=adf)


def test_decorator_renaming(tmp_path):
    data = [["10000000", 0]]
    column_names = ["mmsi", "status"]

    column_specs = [
        DataSpecification(name="mmsi"),
        DataSpecification(name="status", unit=units.deg)
    ]

    df_pd = pd.DataFrame(data, columns=column_names)
    df = vaex.from_pandas(df_pd)
    adf = AnnotatedDataFrame(df, MetaData(columns=column_specs))

    class TransformX(PipelineElement):

        @damast.core.describe("Generic transform of x")
        @damast.core.input({"x": {"unit": units.deg}})
        @damast.core.output({"{{x}}_suffix": {"unit": units.deg}})
        def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            df[f"{self.get_name('x')}_suffix"] = df[self.get_name('x')]
            return df

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[0].name == "x"
    assert getattr(TransformX.transform, DECORATED_OUTPUT_SPECS)[0].name == "{{x}}_suffix"

    with pytest.raises(RuntimeError, match="Input requirements are not fulfilled"):
        pipeline = DataProcessingPipeline("TransformStatus", tmp_path)
        pipeline.add("Transform status",
                     TransformX(),
                     name_mappings={"x": "extra_status"})
        pipeline.transform(df=adf)

    pipeline = DataProcessingPipeline("TransformStatus", tmp_path)
    pipeline.add("Transform status",
                 TransformX(),
                 name_mappings={"x": "status"})

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[0].name == "x"
    assert getattr(TransformX.transform, DECORATED_OUTPUT_SPECS)[0].name == "{{x}}_suffix"

    name, transformer = pipeline.steps[0]
    assert transformer.input_specs[0].name == "status"
    assert transformer.output_specs[0].name == "status_suffix"

    repr_before = pipeline.to_str()
    pipeline.transform(df=adf)
    repr_after = pipeline.to_str()
    assert repr_before == repr_after

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[0].name == "x"
    assert getattr(TransformX.transform, DECORATED_OUTPUT_SPECS)[0].name == "{{x}}_suffix"


def test_toplevel_decorators(tmp_path):
    data = [["10000000", 0]]
    column_names = ["mmsi", "status"]

    column_specs = [
        DataSpecification(name="mmsi"),
        DataSpecification(name="status", unit=units.deg)
    ]

    df_pd = pd.DataFrame(data, columns=column_names)
    df = vaex.from_pandas(df_pd)
    adf = AnnotatedDataFrame(df, MetaData(columns=column_specs))

    class TransformX(PipelineElement):

        @damast.describe("Generic transform of x")
        @damast.input({"x": {"unit": units.deg}})
        @damast.output({"{{x}}_suffix": {"unit": units.deg}})
        @damast.artifacts({"file": "test.damast"})
        def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            df[f"{self.get_name('x')}_suffix"] = df[self.get_name('x')]

            filename = self.parent_pipeline.base_dir / "test.damast"
            with open(filename, "w") as f:
                f.write("test")
            return df

    p = DataProcessingPipeline(name="toplevel-test",
                               base_dir=tmp_path) \
        .add(name="status-transform",
             transformer=TransformX(),
             name_mappings={"x": "status"})
    p.transform(df=adf)
