
import polars
import pytest
from astropy import units

import damast
from damast.core.data_description import CyclicMinMax, MinMax
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement
from damast.core.decorators import (
    DAMAST_DEFAULT_DATASOURCE,
    DECORATED_INPUT_SPECS,
    DECORATED_OUTPUT_SPECS,
    )
from damast.core.metadata import DataCategory, DataSpecification, MetaData
from damast.core.transformations import MultiCycleTransformer
from damast.core.types import XDataFrame
from damast.data_handling.transformers.cycle_transformer import CycleTransformer
from damast.domains.maritime.math.spatial import great_circle_distance
from damast.utils import fromisoformat


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
        lat_cyclic_transformer = MultiCycleTransformer(features=[self.get_name("latitude")], n=180.0)
        lon_cyclic_transformer = MultiCycleTransformer(features=[self.get_name("longitude")], n=360.0)

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
        transformer = MultiCycleTransformer(features=["latitude", "longitude"], n=360)
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
        lat_cyclic_transformer = MultiCycleTransformer(features=["latitude"], n=180.0)
        lon_cyclic_transformer = MultiCycleTransformer(features=["longitude"], n=360.0)

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

class JoinByTimestamp(PipelineElement):
    def __init__(self):
        pass

    @damast.core.describe("JoinByTimestamp")
    @damast.core.input({
                           "timestamp": {},
                           "lon": {},
                           "lat": {},
                       })
    @damast.core.input({
                            "timestamp": {},
                            "lat": {},
                            "lon": {}
                        }, label='other'
    )
    @damast.core.output({})
    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame) -> AnnotatedDataFrame:
        other_timestamp = self.get_name('timestamp', datasource='other')
        df_timestamp = self.get_name('timestamp')

        df._dataframe = df.join(other._dataframe, left_on=df_timestamp, right_on=other_timestamp)
        return df

class JoinSpatioTemporal(PipelineElement):
    distance_in_km: float
    before_time_in_s: float
    after_time_in_s: float

    def __init__(self,
                 distance_in_km: float,
                 before_time_in_s: float,
                 after_time_in_s: float):
        self.distance_in_km = distance_in_km
        self.before_time_in_s = before_time_in_s
        self.after_time_in_s = after_time_in_s
        pass

    @damast.core.describe("JoinSpatioTemporal")
    @damast.core.input({
                           "mmsi": {},
                           "timestamp": { 'unit': 's'},
                           "lon": { 'unit': 'deg'},
                           "lat": { 'unit': 'deg' },
                       })
    @damast.core.input({
                            "timestamp": { 'unit': 's'},
                            "lat": { 'unit': 'deg' },
                            "lon": { 'unit': 'deg' }
                        }, label='other'
    )
    @damast.core.output({
        'event_timestamp': { 'unit': 's' },
        'event_type': {},
        'event_delta_distance': { 'description': "Distance between vessel and event", 'unit': 'km'},
        'event_delta_time': { 'description': "Timedelta between event and vessel message", 'unit': 's'},
    })
    def transform(self, df: AnnotatedDataFrame, other: AnnotatedDataFrame) -> AnnotatedDataFrame:
        import polars as pl
        import polars.selectors as cs

        other_timestamp = self.get_name('timestamp', datasource='other')
        df_timestamp = self.get_name('timestamp')

        filtered_df = df.join_where(other._dataframe, \
                                      (pl.col(df_timestamp) - self.before_time_in_s) <= pl.col(other_timestamp), \
                                      (pl.col(df_timestamp) + self.after_time_in_s) >= pl.col(other_timestamp), \
                                      great_circle_distance(pl.col(self.get_name('lat')),
                                                            pl.col(self.get_name('lon')),
                                                            pl.col(self.get_name('lat', datasource='other')),
                                                            pl.col(self.get_name('lon', datasource='other'))) <= self.distance_in_km
                    )
        df._dataframe = df.join(filtered_df,
                    how="left",
                    left_on=[self.get_name('mmsi'), df_timestamp],
                    right_on=[self.get_name('mmsi'), df_timestamp],
                    suffix="_redundant",
                    ).drop(cs.ends_with("_redundant"))

        df._dataframe = df.with_columns(
                  event_delta_distance = great_circle_distance(pl.col(self.get_name('lat')),
                                        pl.col(self.get_name('lon')),
                                        pl.col(self.get_name('lat', datasource='other')),
                                        pl.col(self.get_name('lon', datasource='other'))
                  ),
                  event_delta_time = pl.col(other_timestamp) - pl.col(df_timestamp)
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
            { DAMAST_DEFAULT_DATASOURCE: DataSpecification.from_requirements(requirements=input_specs) }

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

    prepared_pipeline_a = pipeline.prepare(df=adf)
    prepared_pipeline_b = pipeline.prepare(df=adf)

    assert prepared_pipeline_a.processing_graph != prepared_pipeline_b.processing_graph

    with pytest.raises(RuntimeError, match="is not yet ready"):
        assert pipeline.output_specs is None

    assert prepared_pipeline_a.output_specs is not None
    output_columns = [x.name for x in prepared_pipeline_a.output_specs]
    for column in ["longitude", "latitude",
                   "latitude_x", "latitude_y",
                   "delta_longitude", "delta_latitude",
                   "label"]:
        assert column in output_columns

    representation = prepared_pipeline_a.to_str(indent_level=0)
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

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[DAMAST_DEFAULT_DATASOURCE][0].name == varname
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

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[DAMAST_DEFAULT_DATASOURCE][0].name == varname
    assert getattr(TransformX.transform, DECORATED_OUTPUT_SPECS)[0].name == "{{" + varname + "}}_suffix"

    for node in pipeline.processing_graph.nodes():
        name = node.name
        transformer = node.transformer

    assert transformer.input_specs[DAMAST_DEFAULT_DATASOURCE][0].name == "status"
    assert transformer.output_specs[0].name == "status_suffix"

    repr_before = pipeline.to_str()
    pipeline.transform(df=adf)
    repr_after = pipeline.to_str()
    assert repr_before == repr_after

    assert getattr(TransformX.transform, DECORATED_INPUT_SPECS)[DAMAST_DEFAULT_DATASOURCE][0].name == varname
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

def test_join_operation(data_path, tmp_path):
    pipeline = DataProcessingPipeline(name="ais_preparation",
                                      base_dir=tmp_path) \
        .join("osint", JoinByTimestamp(),
                  name_mappings = {
                      'df': {
                          "timestamp": "date_time_utc",
                      },
                      'other': {
                          "timestamp": "timestamp",
                          "lat": "latitude",
                          "lon": "longitude",
                      }
                  },
        )

    ais_csv = data_path / "test_ais.csv"
    ais_df = AnnotatedDataFrame.from_file(ais_csv)

    osint_csv = data_path / "osint.csv"
    osint_df = AnnotatedDataFrame.from_file(osint_csv, metadata_required=False)

    with pytest.raises(TypeError, match="expected an annotated dataframe"):
        new_adf = pipeline.transform(ais_csv, osint=osint_df)

    joined_df = pipeline.transform(ais_df, osint=osint_df)

    assert len(osint_df) != len(ais_df)
    assert len(joined_df) == 1
    for column in ["event_type", "lat", "latitude"]:
        assert column in joined_df.columns


def test_join_pipeline(data_path, tmp_path):
    event_pipeline = DataProcessingPipeline(name="event_preparation",
                                      base_dir=tmp_path) \
                .add("lat_cycle_transform", CycleTransformer(n=180), name_mappings={'x': 'latitude'}) \
                .add("lon_cycle_transform", CycleTransformer(n=90), name_mappings={'x': 'longitude'})

    pipeline = DataProcessingPipeline(name="ais_preparation",
                                      base_dir=tmp_path) \
        .join("osint", JoinByTimestamp(), data_source=event_pipeline,
                  name_mappings = {
                      'df': {
                          "timestamp": "date_time_utc",
                      },
                      'other': {
                          "timestamp": "timestamp",
                          "lat": "latitude",
                          "lon": "longitude",
                      }
                  },
        )

    ais_csv = data_path / "test_ais.csv"
    ais_df = AnnotatedDataFrame.from_file(ais_csv)

    osint_csv = data_path / "osint.csv"
    osint_df = AnnotatedDataFrame.from_file(osint_csv, metadata_required=False)

    with pytest.raises(TypeError, match="expected an annotated dataframe"):
        new_adf = pipeline.transform(ais_csv, osint=osint_df)

    joined_df = pipeline.transform(ais_df, osint=osint_df)

    assert len(osint_df) != len(ais_df)
    assert len(joined_df) == 1
    for column in ["event_type", "lat", "latitude"]:
        assert column in joined_df.columns

def test_join_spatio_temporal_pipeline(data_path, tmp_path):

    from damast.data_handling.transformers import AddTimestamp


    distance_in_km=50
    after_time_in_s=3600*24*7
    before_time_in_s=3600*24

    event_pipeline = DataProcessingPipeline(name="event_preparation",
                                      base_dir=tmp_path
        ).add("event_timestamp",
                 AddTimestamp(),
                 name_mappings={
                     "from": "timestamp",
                     "to": "event_timestamp"
                 }
        )

    pipeline = DataProcessingPipeline(name="ais_preparation",
                                      base_dir=tmp_path
        ).add("message_timestamp",
                 AddTimestamp(),
                 name_mappings={
                     "from": "date_time_utc",
                     "to": "message_timestamp"
                 }
        ).join("osint", JoinSpatioTemporal(
                distance_in_km=distance_in_km,
                after_time_in_s=after_time_in_s,
                before_time_in_s=before_time_in_s,
            ), data_source=event_pipeline,
                  name_mappings = {
                      'df': {
                          "timestamp": "message_timestamp",
                      },
                      'other': {
                          "timestamp": "event_timestamp",
                          "lat": "latitude",
                          "lon": "longitude",
                      }
                  },
        )

    ais_csv = data_path / "test_ais.csv"
    ais_df = AnnotatedDataFrame.from_file(ais_csv)
    ais_df.metadata['lon'].unit = 'deg'
    ais_df.metadata['lat'].unit = 'deg'

    osint_csv = data_path / "osint.csv"
    osint_df = AnnotatedDataFrame.from_file(osint_csv, metadata_required=False)
    osint_df.metadata['longitude'].unit = 'deg'
    osint_df.metadata['latitude'].unit = 'deg'

    event_type_messages = {}
    for e in osint_df.collected().rows(named=True):
        event_type = e['event_type']
        event_longitude = e['longitude']
        event_latitude = e['latitude']
        event_timestamp = e['timestamp']

        matches = []
        for msg in ais_df.collected().rows(named=True):
            msg_latitude = msg['lat']
            msg_longitude = msg['lon']
            msg_timestamp = msg['date_time_utc']

            time_delta = (fromisoformat(msg_timestamp) - fromisoformat(event_timestamp)).total_seconds()
            if time_delta > -before_time_in_s and time_delta < after_time_in_s:
                delta_distance_in_km = great_circle_distance(msg_latitude, msg_longitude,
                                      event_latitude, event_longitude)
                if delta_distance_in_km <= distance_in_km:
                    matches.append(msg)

        event_type_messages[event_type] = matches


    joined_df = pipeline.transform(ais_df, osint=osint_df)
    for column in ["event_type", "lat", "latitude"]:
        assert column in joined_df.columns

    for column in joined_df.columns:
        assert not column.endswith("_redundant")

    assert len(joined_df) == len(ais_df)
    messages_with_events = joined_df.filter(polars.col("event_type").is_not_null())
    for e in messages_with_events.collect().rows(named=True):
        event_type = e['event_type']
        assert len([x for x in event_type_messages[event_type] if x['lat'] == e['lat'] and x['lon'] == e['lon'] and x['date_time_utc'] == e['date_time_utc']]) == 1


