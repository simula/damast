import numpy as np
import pandas as pd
import polars
import pytest
from astropy import units

import damast
from damast.core.data_description import CyclicMinMax, MinMax
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.decorators import (
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS
)

from damast.core.dataprocessing import (
    DataProcessingPipeline,
    PipelineElement
)
from damast.core.metadata import DataCategory, DataSpecification, MetaData
from damast.core.transformations import CycleTransformer, Transformer
from damast.core.types import XDataFrame


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
        lat_cyclic_transformer = CycleTransformer(features=[self.get_name("latitude")], n=180.0)
        lon_cyclic_transformer = CycleTransformer(features=[self.get_name("longitude")], n=360.0)

        df = lat_cyclic_transformer.fit_transform(df=df)
        df = lon_cyclic_transformer.fit_transform(df=df)
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
        transformer = CycleTransformer(features=["latitude", "longitude"], n=360)
        return transformer.fit_transform(df)


class DataProcessorARemoveCol(PipelineElement):
    @damast.core.input({
        "longitude": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)},
        "latitude": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)}
    })
    @damast.core.output({
        "longitude": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        return df.drop(columns=["latitude"])


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
        lat_cyclic_transformer = CycleTransformer(features=["latitude"], n=180.0)
        lon_cyclic_transformer = CycleTransformer(features=["longitude"], n=360.0)

        df = lat_cyclic_transformer.fit_transform(df=df)
        df = lon_cyclic_transformer.fit_transform(df=df)
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
        df._dataframe = df.with_columns(
                (polars.col("longitude_x") - polars.col("longitude_y")).alias("delta_longitude")
            )
        df._dataframe = df.with_columns(
                (polars.col("latitude_x") - polars.col("latitude_y")).alias("delta_latitude")
            )
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
        df._dataframe = df._dataframe.with_columns(
                polars.lit("data-label").alias("label")
        )
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
    return polars.LazyFrame(data, columns, orient="row")


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
    return polars.LazyFrame(data, columns, orient="row")


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
    with pytest.raises(RuntimeError, match="Failed to update metadata"):
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


def test_data_processing_valid_pipeline(lat_lon_dataframe, lat_lon_metadata, height_metadata, tmp_path):
    pipeline = DataProcessingPipeline(name="abc", base_dir=tmp_path) \
        .add("transform-a", TransformerA()) \
        .add("transform-b", TransformerB()) \
        .add("transform-c", TransformerC())

    with pytest.raises(RuntimeError, match="set the correct output specs"):
        pipeline.output_specs

    with pytest.raises(RuntimeError, match="ensure consistency"):
        invalid_adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                                         metadata=lat_lon_metadata)
        invalid_adf._metadata = height_metadata
        pipeline.prepare(df=invalid_adf)

    adf = AnnotatedDataFrame(dataframe=lat_lon_dataframe,
                             metadata=lat_lon_metadata)

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

    df = polars.LazyFrame(data, column_names, orient="row")
    adf = AnnotatedDataFrame(df, MetaData(columns=column_specs))

    class TransformX(PipelineElement):

        @damast.core.describe("Generic transform of x")
        @damast.core.input({"x": {"unit": units.deg}})
        @damast.core.output({"{{x}}_suffix": {"unit": units.deg}})
        def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            df._dataframe = df._dataframe.with_columns(polars.col(self.get_name('x')).alias(f"{self.get_name('x')}_suffix"))
            return df

    pipeline = DataProcessingPipeline(name="TransformStatus",
                                      base_dir=tmp_path)
    pipeline.add("Transform status",
                 TransformX(),
                 name_mappings={"x": "status"})

    adf = pipeline.transform(df=adf)

    assert "status_suffix" in adf.column_names, "Expect 'status_suffix' to be a new column"

    assert adf.metadata['status_suffix'], "Expect metadata to be available for 'status_suffix'"
    assert adf.metadata['status_suffix'].representation_type == polars.Int64, "Expect representation_type Int64 for 'status_suffix'"

@pytest.mark.parametrize("varname",["x","xyz"])
def test_decorator_renaming(varname, tmp_path):
    data = [["10000000", 0]]
    column_names = ["mmsi", "status"]

    column_specs = [
        DataSpecification(name="mmsi"),
        DataSpecification(name="status", unit=units.deg)
    ]

    df = polars.LazyFrame(data, column_names, orient="row")
    adf = AnnotatedDataFrame(df, MetaData(columns=column_specs))

    class TransformX(PipelineElement):

        @damast.core.describe("Generic transform of x")
        @damast.core.input({f"{varname}": {"unit": units.deg}})
        @damast.core.output({"{{" + varname + "}}_suffix": {"unit": units.deg}})
        def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
            df[f"{self.get_name(varname)}_suffix"] = df[self.get_name(varname)]
            return df

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[0].name == varname
    assert getattr(TransformX.transform, DECORATED_OUTPUT_SPECS)[0].name == "{{" + varname + "}}_suffix"

    with pytest.raises(RuntimeError, match="Input requirements are not fulfilled"):
        pipeline = DataProcessingPipeline(name="TransformStatus",
                                          base_dir=tmp_path)
        pipeline.add("Transform status",
                     TransformX(),
                     name_mappings={varname: "extra_status"})
        pipeline.transform(df=adf)

    pipeline = DataProcessingPipeline(name="TransformStatus",
                                      base_dir=tmp_path)
    pipeline.add("Transform status",
                 TransformX(),
                 name_mappings={varname: "status"})

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[0].name == varname
    assert getattr(TransformX.transform, DECORATED_OUTPUT_SPECS)[0].name == "{{" + varname + "}}_suffix"

    for node in pipeline.processing_graph.nodes():
        name = node.name
        transformer = node.transformer

    assert transformer.input_specs[0].name == "status"
    assert transformer.output_specs[0].name == "status_suffix"

    repr_before = pipeline.to_str()
    pipeline.transform(df=adf)
    repr_after = pipeline.to_str()
    assert repr_before == repr_after

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[0].name == varname
    assert getattr(TransformX.transform, DECORATED_OUTPUT_SPECS)[0].name == "{{" + varname + "}}_suffix"


def test_toplevel_decorators(tmp_path):
    data = [["10000000", 0]]
    column_names = ["mmsi", "status"]

    column_specs = [
        DataSpecification(name="mmsi"),
        DataSpecification(name="status", unit=units.deg)
    ]

    df = polars.LazyFrame(data, column_names, orient="row")
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


def test_save(tmp_path):
    pipeline = DataProcessingPipeline(name="abc",
                                      base_dir=tmp_path) \
        .add("transform-a", TransformerA()) \
        .add("transform-b", TransformerB()) \
        .add("transform-c", TransformerC())

    pipeline_file = pipeline.save(tmp_path)
    loaded_pipeline = DataProcessingPipeline.load(pipeline_file)
    print(loaded_pipeline)


def test_save_load_state(lat_lon_annotated_dataframe, lat_lon_metadata, tmp_path):
    """
    Test the load/save_state functionality of the pipeline
    """
    pipeline = DataProcessingPipeline(name="a",
                                      base_dir=tmp_path) \
        .add("transform-a", DataProcessorA())

    df_transformed = pipeline.transform(df=lat_lon_annotated_dataframe)
    pipeline.save_state(df=df_transformed, dir=tmp_path)

    loaded_adf = pipeline.load_state(df=lat_lon_annotated_dataframe, dir=tmp_path)

    #for i in range(len(df_transformed)):
        #assert np.array_equiv(df_transformed[i], loaded_adf[i])
    assert XDataFrame(df_transformed).equals(XDataFrame(loaded_adf))
