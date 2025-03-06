from pathlib import Path
from typing import Union

import polars

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline, PipelineElement
from damast.core.datarange import CyclicMinMax, MinMax
from damast.core.metadata import MetaData
from damast.core.transformations import CycleTransformer
from damast.core.units import units

ais_spec = Path(__file__).parent / "ais_dataspec.yaml"
ais_data = Path(__file__).parent.parent.parent / "damast" / "data" / "test_ais.csv"

md = MetaData.load_yaml(ais_spec)
df = polars.scan_csv(ais_data, separator=";")

adf = AnnotatedDataFrame(dataframe=df,
                         metadata=md)


# adf.save(filename="/tmp/test.hdf5")
# new_adf = AnnotatedDataFrame.from_file(filename="/tmp/test.hdf5")

class HDF5Export(PipelineElement):
    filename: Path

    def __init__(self, filename: Union[str, Path]):
        self.filename = Path(filename)

    @damast.core.input({})
    @damast.core.artifacts({
        "hdf5_export": "*.hdf5"
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        df.save(filename=self.parent_pipeline.base_dir / self.filename)
        return df


class LatLonTransformer(PipelineElement):
    @damast.core.describe("Lat/Lon cyclic transformation")
    @damast.core.input({
        "lat": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)},
        "lon": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)}
    })
    @damast.core.output({
        "lat_x": {"value_range": MinMax(-1.0, 1.0)},
        "lat_y": {"value_range": MinMax(-1.0, 1.0)},
        "lon_x": {"value_range": MinMax(-1.0, 1.0)},
        "lon_y": {"value_range": MinMax(-1.0, 1.0)}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat_cyclic_transformer = CycleTransformer(features=["lat"], n=180.0)
        lon_cyclic_transformer = CycleTransformer(features=["lon"], n=360.0)

        _df = lat_cyclic_transformer.fit_transform(df=df)
        _df = lon_cyclic_transformer.fit_transform(df=_df)
        df._dataframe = _df
        return df


base_dir = Path(__file__).parent / "output"
base_dir.mkdir(parents=True, exist_ok=True)

pipeline = DataProcessingPipeline(name="ais_preparation",
                                  base_dir=base_dir) \
    .add("geo_cycle_transform", LatLonTransformer()) \
    .add("export-hdf5", HDF5Export("ais-processed.hdf5"))

new_df = pipeline.transform(adf)
print(pipeline.to_str(indent_level=2))
print(new_df._dataframe.collect())

# Start ML
# ml = MLPipeline(name="train")
