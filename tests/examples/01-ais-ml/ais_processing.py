from pathlib import Path

import vaex
from vaex.ml.transformations import Transformer

import damast
from damast.core.dataframe import AnnotatedDataFrame
from damast.core.dataprocessing import DataProcessingPipeline
from damast.core.datarange import CyclicMinMax
from damast.core.metadata import MetaData
from damast.core.units import units

ais_spec = Path(__file__).parent / "ais_dataspec.yaml"
ais_data = Path(__file__).parent.parent.parent / "damast" / "data" / "test_ais.csv"

md = MetaData.load_yaml(ais_spec)
df = vaex.from_csv(ais_data, sep=";")

adf = AnnotatedDataFrame(dataframe=df,
                         metadata=md)


class LatLonTransformer(Transformer):
    @damast.core.describe("Lat/Lon cyclic transformation")
    @damast.core.input({
        "lat": {"unit": units.deg, "value_range": CyclicMinMax(-90.0, 90.0)},
        "lon": {"unit": units.deg, "value_range": CyclicMinMax(-180.0, 180.0)}
    })
    @damast.core.output({
        "lat_x": {"unit": units.deg},
        "lat_y": {"unit": units.deg},
        "lon_x": {"unit": units.deg},
        "lon_y": {"unit": units.deg}
    })
    def transform(self, df: AnnotatedDataFrame) -> AnnotatedDataFrame:
        lat_cyclic_transformer = vaex.ml.CycleTransformer(features=["lat"], n=180.0)
        lon_cyclic_transformer = vaex.ml.CycleTransformer(features=["lon"], n=360.0)

        _df = lat_cyclic_transformer.fit_transform(df=df)
        _df = lon_cyclic_transformer.fit_transform(df=_df)
        df._dataframe = _df
        return df


pipeline = DataProcessingPipeline([
    ("geolocation", LatLonTransformer())
])

new_df = pipeline.transform(adf)
print(pipeline.to_str(indent_level=2))
print(new_df._dataframe)
